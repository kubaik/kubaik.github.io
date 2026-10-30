# Offline-first money apps with DynamoDB Streams

Tutorials usually show the happy path for offline-first payments: queue the transaction, retry later. That model holds up for one user. It breaks down when thousands of agents each push dozens of transactions per hour into a backend that has to reconcile them against a shared wallet ledger. Three failure modes show up repeatedly.

**Ordering guarantees disappear.** DynamoDB Streams preserve order per shard, but once you fan out to SQS or EventBridge the ordering is only per message group, not per user or per transaction. A retry loop that reorders transfers can produce duplicate debits unless the write path is idempotent on its own.

**Balance checks race.** A wallet balance can change between the moment an offline transaction is saved and the moment it syncs. Naïve eventual-consistency reads return stale values, and a read-then-write balance check will over- or under-deduct whenever two syncs for the same phone overlap.

**Cost grows quietly.** Using DynamoDB transactions for every offline write multiplies consumed capacity units. Teams often respond by bumping provisioned capacity, watch most of it sit idle, then switch to on-demand and get throttled during month-end spikes.

Eventual consistency is not a toggle you flip once. It is a contract you enforce with concrete policies: shard-key design, conditional writes that express the balance check inside the write itself, and retry backoff that respects a latency budget. This article walks through those levers.

## Prerequisites and what you'll build

You'll end up with a small Node.js service (Node 20 LTS, arm64) that:

- Accepts `POST /tx/offline` with `{ agentId, phone, amount, idempotencyKey }`
- Stores the transaction in DynamoDB with a TTL of 7 days and a status of `queued`
- Uses DynamoDB Streams → Lambda to reconcile to the ledger table on success, or move the record to a DLQ on failure
- Exposes `/tx/status/{idempotencyKey}` that returns the current status without leaking internal state

You'll need:

- An AWS account with IAM permissions for DynamoDB, Lambda, CloudWatch Logs, SQS, EventBridge, and IAM roles
- AWS CLI v2
- Node 20 LTS
- A DynamoDB table for transactions (`OfflineTx`) with partition key `agentId` and sort key `createdAt`
- A DynamoDB table for the ledger (`WalletLedger`) with partition key `phone` and sort key `txId`

The ledger table uses a single-table design: one GSI on `GSI1PK = STATUS` and `GSI1SK = createdAt` so the reconciler can query only queued records instead of scanning the whole table. Any query that filters by status will be cheaper than a full scan, though the exact saving depends on how many records are in each state — measure it with consumed-capacity metrics rather than assuming a fixed percentage.

## Step 1 — set up the environment

Create a new directory and initialize:

```bash
mkdir mobile-money-offline && cd mobile-money-offline
npm init -y
npm install @aws-sdk/client-dynamodb @aws-sdk/lib-dynamodb @aws-sdk/client-sqs uuid
```

Install the CDK CLI globally and bootstrap once per region:

```bash
npm install -g aws-cdk
cdk bootstrap aws://ACCOUNT-NUMBER/REGION
```

Define `cdk.json`:

```json
{
  "app": "node bin/app.js",
  "context": {
    "@aws-cdk/aws-lambda:reservedConcurrentExecutions": 1000,
    "@aws-cdk/core:enableStackNameDuplicates": true
  }
}
```

Create `bin/app.js`:

```javascript
#!/usr/bin/env node
const cdk = require('aws-cdk-lib');
const { OfflineStack } = require('../lib/offline-stack');

const app = new cdk.App();
new OfflineStack(app, 'OfflineStack', { env: { region: 'eu-central-1' } });
```

Create `lib/offline-stack.js`:

```javascript
const cdk = require('aws-cdk-lib');
const dynamodb = require('aws-cdk-lib/aws-dynamodb');
const lambda = require('aws-cdk-lib/aws-lambda');
const eventsources = require('aws-cdk-lib/aws-lambda-event-sources');
const sqs = require('aws-cdk-lib/aws-sqs');
const { Duration } = cdk;

class OfflineStack extends cdk.Stack {
  constructor(scope, id, props) {
    super(scope, id, props);

    // OfflineTx table: on-demand billing, TTL, stream with old and new images
    const offlineTx = new dynamodb.Table(this, 'OfflineTx', {
      partitionKey: { name: 'agentId', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'createdAt', type: dynamodb.AttributeType.NUMBER },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      timeToLiveAttribute: 'expiresAt',
      stream: dynamodb.StreamViewType.NEW_AND_OLD_IMAGES,
      globalSecondaryIndexes: [
        {
          indexName: 'StatusCreatedAtGSI',
          partitionKey: { name: 'STATUS', type: dynamodb.AttributeType.STRING },
          sortKey: { name: 'createdAt', type: dynamodb.AttributeType.NUMBER },
        },
      ],
    });

    // Ledger table: single-table design
    const ledger = new dynamodb.Table(this, 'WalletLedger', {
      partitionKey: { name: 'phone', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'txId', type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      globalSecondaryIndexes: [
        {
          indexName: 'PhoneStatusGSI',
          partitionKey: { name: 'phone', type: dynamodb.AttributeType.STRING },
          sortKey: { name: 'STATUS', type: dynamodb.AttributeType.STRING },
        },
      ],
    });

    // Dead-letter queue for failed reconciliations
    const sqsQueue = new sqs.Queue(this, 'TxDLQ', { retentionPeriod: Duration.days(14) });

    // Lambda consumer
    const consumer = new lambda.Function(this, 'TxConsumer', {
      runtime: lambda.Runtime.NODEJS_20_X,
      code: lambda.Code.fromAsset('lambda'),
      handler: 'index.handler',
      memorySize: 512,
      timeout: Duration.seconds(15),
      environment: {
        OFFLINE_TABLE: offlineTx.tableName,
        LEDGER_TABLE: ledger.tableName,
        DLQ_URL: sqsQueue.queueUrl,
      },
      reservedConcurrentExecutions: 200,
    });

    // Event source
    consumer.addEventSource(new eventsources.DynamoEventSource(offlineTx, {
      startingPosition: lambda.StartingPosition.LATEST,
      batchSize: 100,
      bisectBatchOnError: true,
      retryAttempts: 3,
    }));

    // Permissions
    offlineTx.grantStreamRead(consumer);
    offlineTx.grantReadWriteData(consumer);
    ledger.grantReadWriteData(consumer);
    sqsQueue.grantSendMessages(consumer);
  }
}

module.exports = { OfflineStack };
```

Deploy once to verify the stack:

```bash
cdk deploy --require-approval never
```

## Step 2 — core implementation

Create `lambda/index.js`:

```javascript
const { DynamoDBClient } = require('@aws-sdk/client-dynamodb');
const { DynamoDBDocumentClient, PutCommand, GetCommand, UpdateCommand, TransactWriteItemsCommand } = require('@aws-sdk/lib-dynamodb');
const { v4: uuidv4 } = require('uuid');
const { SQSClient, SendMessageCommand } = require('@aws-sdk/client-sqs');

const ddb = new DynamoDBClient({ region: process.env.AWS_REGION });
const docClient = DynamoDBDocumentClient.from(ddb);
const sqs = new SQSClient({ region: process.env.AWS_REGION });

const OFFLINE_TABLE = process.env.OFFLINE_TABLE;
const LEDGER_TABLE = process.env.LEDGER_TABLE;
const DLQ_URL = process.env.DLQ_URL;

async function enqueueOfflineTx(agentId, phone, amount, idempotencyKey) {
  const now = Date.now();
  const expiresAt = Math.floor(now / 1000) + 7 * 24 * 3600; // TTL is in epoch seconds

  await docClient.send(new PutCommand({
    TableName: OFFLINE_TABLE,
    Item: {
      agentId,
      createdAt: now,
      expiresAt,
      phone,
      amount: Number(amount),
      idempotencyKey,
      status: 'queued',
    },
    ConditionExpression: 'attribute_not_exists(idempotencyKey)',
  }));
  return { idempotencyKey };
}

// Convert a DynamoDB stream image into plain JS values.
function unmarshallImage(image) {
  const out = {};
  for (const [key, value] of Object.entries(image)) {
    const type = Object.keys(value)[0];
    out[key] = value[type];
  }
  return out;
}

async function reconcileTx(record) {
  if (record.eventName !== 'INSERT' && record.eventName !== 'MODIFY') return;

  const newImage = record.dynamodb.NewImage;
  const { agentId, phone, amount, idempotencyKey, status, createdAt } = unmarshallImage(newImage);

  if (status !== 'queued') return; // only process queued records

  // 1. Guard against duplicate debits using the idempotency key.
  const existing = await docClient.send(new GetCommand({
    TableName: LEDGER_TABLE,
    Key: { phone, txId: idempotencyKey },
  }));

  if (existing.Item) {
    await docClient.send(new UpdateCommand({
      TableName: OFFLINE_TABLE,
      Key: { agentId, createdAt },
      UpdateExpression: 'SET #status = :status',
      ExpressionAttributeNames: { '#status': 'status' },
      ExpressionAttributeValues: { ':status': 'synced' },
    }));
    return;
  }

  // 2. Atomic debit: write the tx row and decrement the balance in one transaction.
  // The condition on the balance row makes the check part of the write itself.
  try {
    await docClient.send(new TransactWriteItemsCommand({
      TransactItems: [
        {
          Put: {
            TableName: LEDGER_TABLE,
            Item: {
              phone,
              txId: idempotencyKey,
              amount: Number(amount),
              createdAt: Date.now(),
              status: 'completed',
              type: 'debit',
            },
            ConditionExpression: 'attribute_not_exists(txId)',
          },
        },
        {
          Update: {
            TableName: LEDGER_TABLE,
            Key: { phone, txId: `BALANCE_${phone}` },
            UpdateExpression: 'SET #balance = #balance - :amount',
            ConditionExpression: 'attribute_exists(#balance) AND #balance >= :amount',
            ExpressionAttributeNames: { '#balance': 'balance' },
            ExpressionAttributeValues: { ':amount': Number(amount) },
          },
        },
      ],
    }));

    await docClient.send(new UpdateCommand({
      TableName: OFFLINE_TABLE,
      Key: { agentId, createdAt },
      UpdateExpression: 'SET #status = :status',
      ExpressionAttributeNames: { '#status': 'status' },
      ExpressionAttributeValues: { ':status': 'synced' },
    }));
  } catch (err) {
    if (err.name === 'TransactionCanceledException' || err.name === 'ConditionalCheckFailedException') {
      await docClient.send(new UpdateCommand({
        TableName: OFFLINE_TABLE,
        Key: { agentId, createdAt },
        UpdateExpression: 'SET #status = :status, #reason = :reason',
        ExpressionAttributeNames: { '#status': 'status', '#reason': 'reason' },
        ExpressionAttributeValues: { ':status': 'failed', ':reason': 'INSUFFICIENT_BALANCE' },
      }));

      await sqs.send(new SendMessageCommand({
        QueueUrl: DLQ_URL,
        MessageBody: JSON.stringify({
          idempotencyKey,
          phone,
          amount,
          error: 'INSUFFICIENT_BALANCE',
        }),
      }));
      return;
    }

    // Transient or unexpected error: rethrow so the stream retries the batch.
    throw err;
  }
}

module.exports = { enqueueOfflineTx, reconcileTx };
```

Two things to note about the rewrite above versus the naive version:

- The balance is decremented with `SET #balance = #balance - :amount` inside the transaction, and the guard `#balance >= :amount` is part of the same write. There is no separate read, so there is no window for a concurrent sync to slip in between the read and the write.
- The ledger row's primary key is `(phone, idempotencyKey)`, so a replayed stream record collides with the existing item and the `attribute_not_exists(txId)` condition fails. That is how the idempotency guard is enforced at the storage layer, not just in application code.
- Transient errors are rethrown, not swallowed into the DLQ. If the Lambda throws, the stream retries the batch; only permanent business failures (insufficient balance) go to the DLQ.

Add the handler file (`lambda/index.handler.js`):

```javascript
const { reconcileTx } = require('./index');

module.exports.handler = async (event) => {
  const records = event.Records || [];
  const results = await Promise.allSettled(records.map(reconcileTx));

  const batchItemFailures = [];
  results.forEach((result, index) => {
    if (result.status === 'rejected') {
      batchItemFailures.push({ itemIdentifier: records[index].eventID });
    }
  });

  return { batchItemFailures };
};
```

Returning `batchItemFailures` (rather than an empty array) requires the event source mapping to have `reportBatchItemFailures: true`. Without it, Lambda retries the whole batch on any failure, which re-processes records that already succeeded.

## Step 3 — edge cases and failure modes

DynamoDB Streams delivers **at least once**, so the same record can fire more than once. If `reconcileTx` is not idempotent, you double-debit the ledger. The idempotency key plus the conditional put handles that. The remaining edge cases:

- **Concurrent balance updates.** If two reconciliations start at the same time for the same phone, the second one hits the balance condition and fails the transaction. The offline record moves to `failed`, and the agent's client sees a failure and can retry. That is the intended behavior — better to fail a debit than to allow an overdraft.
- **Duplicate idempotency keys from the agent.** If the mobile app retries the same transfer before the offline record syncs, the ledger `Get` finds the existing item and marks the offline record `synced` with no second debit.
- **Stale balance reads.** Not applicable here, because the balance is read inside the transaction. If you ever add a separate read for a display value, treat it as advisory only.
- **Lambda timeouts at scale.** With many agents syncing at once, a batch of 100 can take several seconds when the ledger GSI has hot keys. Raising memory to 1 GB and timeout to 30 s is a common adjustment; verify with the `Duration` metric on the function before changing anything.
- **Poison records.** A record that always throws (e.g. malformed payload) will exhaust the event source's retry attempts and land in the Lambda destination / on-failure destination. Configure that destination explicitly; don't rely on the business DLQ for it.

For the agent SDK, use exponential backoff with jitter:

```javascript
function retry(fn, retries = 3, delay = 100) {
  return new Promise((resolve, reject) => {
    fn()
      .then(resolve)
      .catch((err) => {
        if (retries <= 0) return reject(err);
        const nextDelay = Math.min(delay * 2 + Math.random() * 100, 5000);
        setTimeout(() => retry(fn, retries - 1, nextDelay).then(resolve).catch(reject), nextDelay);
      });
  });
}
```

Jitter matters because synchronized retries create a thundering herd: when a branch office comes back online, every agent's client fires at once. Spreading retries over a random window keeps the reconcile requests from clustering on the same second.

## Step 4 — observability and tests

Alarm on DLQ depth so a stuck reconciler is visible before customers notice:

```bash
aws cloudwatch put-metric-alarm \
  --alarm-name "OfflineTx-DLQ-Alarm" \
  --alarm-description "DLQ contains failed offline transactions" \
  --metric-name "ApproximateNumberOfMessagesVisible" \
  --namespace "AWS/SQS" \
  --statistic "Sum" \
  --period 60 \
  --threshold 1 \
  --comparison-operator "GreaterThanOrEqualToThreshold" \
  --evaluation-periods 1 \
  --alarm-actions arn:aws:sns:eu-central-1:123456789012:AlarmTopic
```

A minimal integration test against a local DynamoDB (DynamoDB Local, or `amazon/dynamodb-local` in Docker) looks like this:

```javascript
const { DynamoDBClient } = require('@aws-sdk/client-dynamodb');
const { DynamoDBDocumentClient, GetCommand, PutCommand } = require('@aws-sdk/lib-dynamodb');
const { enqueueOfflineTx, reconcileTx } = require('../lambda/index');

const ddb = new DynamoDBClient({
  region: 'eu-central-1',
  endpoint: 'http://localhost:8000',
  credentials: { accessKeyId: 'local', secretAccessKey: 'local' },
});
const docClient = DynamoDBDocumentClient.from(ddb);

test('duplicate idempotency key results in a single ledger row', async () => {
  const key = 'TEST_IDEMPOTENT_001';
  await enqueueOfflineTx('AGENT_001', '254712345678', 100, key);

  const record = {
    eventName: 'INSERT',
    dynamodb: {
      NewImage: {
        agentId: { S: 'AGENT_001' },
        createdAt: { N: String(Date.now()) },
        phone: { S: '254712345678' },
        amount: { N: '100' },
        idempotencyKey: { S: key },
        status: { S: 'queued' },
      },
    },
  };

  await reconcileTx(record);
  await reconcileTx(record); // replay

  const ledgerRow = await docClient.send(new GetCommand({
    TableName: 'WalletLedger',
    Key: { phone: '254712345678', txId: key },
  }));
  expect(ledgerRow.Item).toBeDefined();
  expect(ledgerRow.Item.amount).toBe(100);
});
```

The replay is the important part. A test that only runs the happy path won't catch the double-debit bug that at-least-once delivery causes in production.

For metrics, emit counters and timers from the handler:

```javascript
const client = require('prom-client');
const register = new client.Registry();
const txCounter = new client.Counter({ name: 'offline_tx_total', help: 'Total offline tx', registers: [register] });
const syncDuration = new client.Histogram({ name: 'offline_sync_duration_ms', help: 'Duration of reconcileTx', buckets: [100, 200, 400, 800, 1600], registers: [register] });

module.exports.handler = async (event) => {
  const end = syncDuration.startTimer();
  const results = await Promise.allSettled(event.Records.map(reconcileTx));
  txCounter.inc(results.length);
  end();
  return { batchItemFailures: [] };
};
```

## How to measure whether this actually helps

The published numbers you'll see for this kind of pattern are usually specific to one deployment and one traffic shape. Instead of copying them, instrument four things and compare before/after on your own workload:

1. **Duplicate debits per million transactions.** Count ledger rows where two entries share the same `(phone, idempotencyKey)` — there should be zero, but counting catches bugs in the guard. Query the ledger GSI by phone and idempotency key, or emit a CloudWatch metric from the reconciler when the `Get` on the ledger returns an existing item.
2. **Reconcile latency p50/p95/p99.** Use the Lambda `Duration` metric plus your own histogram. Compare the stream-to-synced delta by recording `Date.now() - createdAt` when the offline record flips to `synced`.
3. **DLQ depth and age.** `ApproximateNumberOfMessagesVisible` and `ApproximateAgeOfOldestMessage` on the DLQ. A rising age means the reconciler is failing faster than the DLQ is being drained.
4. **Consumed capacity per million transactions.** The `ConsumedWriteCapacityUnits` and `ConsumedReadCapacityUnits` metrics on both tables, summed over the window and divided by the transaction count. This is what actually tells you whether the transaction-based write path is worth its cost versus a simpler non-atomic path.

Run the same load against both implementations — a replay harness that pushes N offline records through the stream — and compare the four metrics. That gives you a real answer for your traffic, not someone else's.

## Common questions and variations

**How do I handle agent refunds or reversals offline?**

Add a second Lambda that listens for `status = refund_requested` on the `OfflineTx` table. When a refund is queued, write a new ledger row with `type: 'credit'` and a deterministic tx id derived from the original (e.g. `refund:${originalIdempotencyKey}`). The conditional put on that id makes the refund idempotent the same way the debit is.

**Is DynamoDB Streams ordering per shard good enough for many agents?**

Per-shard ordering is strong. Because the partition key is `agentId`, all events for a given agent land on the same shard and are delivered in order. If you need ordering by phone instead, you'd have to shard by phone and accept the hot-key risk on popular numbers, or use a stream service with explicit ordering keys. In practice, agent-keyed ordering plus idempotent writes covers most money-app workloads.

**What if the Lambda consumer fails 100% of the time?**

DynamoDB Streams retries the batch up to the configured `retryAttempts` (3 in the CDK example above), then routes the failed records to the Lambda destination you configure. If the failure is a programming error, fix the code and replay the destination. Don't let a permanent bug keep burning stream retries — that blocks the shard.

**Can I use this with PostgreSQL instead of DynamoDB?**

Yes, but you lose the per-shard ordering guarantee and the single-table design. You'd use logical replication or `LISTEN/NOTIFY` to stream changes, and you'd manage the ordering key yourself. The cost comparison depends heavily on your workload and instance sizing; don't assume a ratio — measure consumed capacity against your Postgres instance-hour cost for the same traffic.

## Decision checklist

Before shipping this pattern, confirm:

- [ ] The balance check is a condition inside the write, not a separate read followed by a write.
- [ ] The ledger primary key includes the idempotency key, so replays collide instead of duplicating.
- [ ] The stream event source has `reportBatchItemFailures: true` and the handler returns per-record failures.
- [ ] Transient errors are rethrown (so the stream retries); only permanent business failures go to the DLQ.
- [ ] The DLQ has an alarm on depth and oldest-message age.
- [ ] The agent SDK uses exponential backoff with jitter, not fixed-interval retries.
- [ ] You have a replay harness that pushes duplicate records through the reconciler.

## Action for the next 30 minutes

Open your wallet service and find the balance check for an offline transaction. If it reads the balance and then writes in a separate call, replace it with a single `TransactWriteItems` call that puts the ledger row and decrements the balance under a `#balance >= :amount` condition — the code in Step 2 is a working template. Then write one test that calls the reconcile function twice with the same record and asserts that exactly one ledger row exists. That test is the difference between a queue that retries and a queue that double-debits.
