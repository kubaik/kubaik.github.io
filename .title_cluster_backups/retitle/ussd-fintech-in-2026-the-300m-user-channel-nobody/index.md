# USSD fintech in 2026: the 300M user channel nobody

USSD tutorials tend to stop at the happy path: a menu, a PIN, a confirmation. Production is what happens when the carrier edge times out, the handset drops mid-session, and the user presses the wrong key on a 2G connection. This guide covers the parts that decide whether a USSD flow survives contact with a live network.

## Why USSD still matters for fintech distribution

USSD has a set of properties that no app can replicate: it works on any GSM handset, requires no install, no data plan, and no storage. The user dials a short code and is inside a session within a second or two. For mass-market financial services in markets where smartphone penetration and storage are constrained, that combination keeps USSD relevant regardless of how polished the native app is.

The common failure mode is treating USSD as a legacy fallback rather than a first-class product surface. Teams build the app first, bolt on a USSD menu later, and then discover the channel has hard constraints the app never had:

- **160 characters per screen.** Anything longer is truncated or rejected by the carrier.
- **A short session budget.** Carriers commonly enforce a session timeout in the range of tens of seconds. When it expires, the session is dropped and the user must dial again.
- **No client-side state.** Every screen is a fresh HTTP request to your endpoint. You own all session state.
- **No error recovery UI.** You cannot show a spinner, a retry button, or a stack trace. You have one string.
- **Per-session carrier fees.** Failed sessions still cost money. At scale, silent failures are a recurring line item.

None of these are reasons to avoid the channel. They are reasons to design for it deliberately.

This guide builds a stateless USSD service on AWS: a Lambda handler behind an HTTP API, a DynamoDB session table, a small explicit state machine, and CloudWatch metrics that tell you where latency actually lives. The code targets Node 20 LTS on Lambda, but the architecture is runtime-agnostic.

## Prerequisites and what you will build

You do not need a carrier contract to start. Most USSD aggregators offer a sandbox that lets you point a short code at an HTTPS endpoint and drive the flow from a real handset. The sandbox is where you validate the state machine; the carrier integration is where you validate certificates, timeouts, and edge behavior.

By the end of this guide you will have:

- A stateless USSD service on AWS Lambda (Node 20 LTS) behind an HTTP API
- A four-state flow: welcome, authenticate, menu, transfer
- A DynamoDB session table with TTL-based expiry
- Idempotency handling so carrier retries do not double-charge or double-execute
- CloudWatch metrics and an alarm on p95 latency
- A load test you can run before carrier integration

The flow mirrors what most wallet USSD menus look like:

1. User dials the short code.
2. System greets and asks for a PIN.
3. User enters the PIN; the system validates it.
4. Menu shows options: transfer, balance, help.
5. User selects transfer, enters amount and recipient.
6. System confirms and sends an OTP by SMS.

Each screen is capped at 160 characters. The whole session must complete inside the carrier's timeout window, or the user redials and you pay the session fee again.

**Environment checklist:**

- Node 20 LTS (`node --version`)
- AWS CDK v2 installed and bootstrapped in your target account
- An aggregator sandbox account with a short code and API credentials
- An AWS region chosen for proximity to the carrier's gateway, not for your team's convenience

The region choice matters more than most teams expect. Carrier session timeouts are measured from the carrier's edge, not from your Lambda. If the round trip from carrier edge to your region adds several hundred milliseconds before your code even runs, you have spent part of your budget on geography. Pick a region close to the carrier POP, and measure rather than assume.

## Step 1 — project setup

Create the project and install dependencies:

```bash
mkdir ussd-fintech && cd ussd-fintech
npm init -y
npm install typescript @types/node --save-dev
tsc --init
npm install aws-cdk-lib constructs aws-cdk-lib/aws-lambda-nodejs @aws-sdk/client-dynamodb @aws-sdk/lib-dynamodb @aws-sdk/client-cloudwatch
```

Scaffold the stack in `lib/ussd-stack.ts`:

```typescript
import * as cdk from 'aws-cdk-lib';
import * as lambda from 'aws-cdk-lib/aws-lambda-nodejs';
import * as apigateway from 'aws-cdk-lib/aws-apigatewayv2';
import * as integrations from 'aws-cdk-lib/aws-apigatewayv2-integrations';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import * as logs from 'aws-cdk-lib/aws-logs';

export class UssdStack extends cdk.Stack {
  constructor(scope: cdk.App, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    const table = new dynamodb.Table(this, 'SessionTable', {
      partitionKey: { name: 'id', type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      timeToLiveAttribute: 'expiresAt',
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    const handler = new lambda.NodejsFunction(this, 'UssdHandler', {
      runtime: lambda.Runtime.NODEJS_20_X,
      memorySize: 512,
      timeout: cdk.Duration.seconds(15),
      logRetention: logs.RetentionDays.ONE_MONTH,
      environment: {
        SESSION_TABLE: table.tableName,
        AT_USERNAME: process.env.AT_USERNAME ?? '',
        AT_API_KEY: process.env.AT_API_KEY ?? '',
      },
    });

    table.grantReadWriteData(handler);

    const httpApi = new apigateway.HttpApi(this, 'UssdApi', {
      defaultIntegration: new integrations.HttpLambdaIntegration('Handler', handler),
    });

    new cdk.CfnOutput(this, 'ApiUrl', { value: httpApi.url! });
  }
}
```

Create `bin/app.ts`:

```typescript
#!/usr/bin/env node
import 'source-map-support/register';
import * as cdk from 'aws-cdk-lib';
import { UssdStack } from '../lib/ussd-stack';

const app = new cdk.App();
new UssdStack(app, 'UssdStack', {
  env: { account: process.env.CDK_DEFAULT_ACCOUNT, region: process.env.CDK_DEFAULT_REGION },
});
```

Deploy:

```bash
export CDK_DEFAULT_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
export CDK_DEFAULT_REGION=eu-west-1
cdk bootstrap
cdk deploy
```

The `ApiUrl` output is the HTTPS endpoint your aggregator will call when a user dials the short code.

**Failure mode to plan for:** production carrier endpoints commonly require mutual TLS. Sandbox endpoints usually relax this, but the production integration will need a client certificate issued by a recognized CA, attached and rotated on a schedule the carrier dictates. Treat certificate provisioning and rotation testing as a distinct workstream with its own lead time, not as a deployment detail.

## Step 2 — the state machine

The core of a USSD service is a pure function: given the previous session state and the user's input, return the next message and the next state. Keeping it pure makes it testable without AWS, which matters because the interesting bugs are in the transitions.

Create `src/ussd.ts`:

```typescript
export type State = 'welcome' | 'auth' | 'menu' | 'transfer' | 'done';

export type Session = {
  phoneNumber: string;
  state: State;
  pin?: string;
};

export type UssdResponse = {
  message: string;
  newState: State;
};

const MENU = '1. Transfer 2. Balance 3. Help';

export function handleUssd(
  session: Session | null,
  text: string | null
): UssdResponse {
  if (!session) {
    return { message: 'Welcome. Enter PIN.', newState: 'auth' };
  }

  switch (session.state) {
    case 'auth':
      if (text === '1234') {
        return { message: MENU, newState: 'menu' };
      }
      return { message: 'Invalid PIN. Try again.', newState: 'auth' };

    case 'menu':
      if (text === '1') {
        return {
          message: 'Enter amount and phone, e.g. 500 08012345678',
          newState: 'transfer',
        };
      }
      if (text === '2') {
        return { message: `Balance: 1,250 NGN. ${MENU}`, newState: 'menu' };
      }
      return { message: `Invalid option. ${MENU}`, newState: 'menu' };

    case 'transfer': {
      const parts = (text ?? '').split(' ');
      const amount = parts[0];
      const phone = parts[1];
      if (amount && phone && /^\d{11}$/.test(phone)) {
        return { message: 'Enter OTP sent to your phone', newState: 'done' };
      }
      return {
        message: 'Invalid format. Use: 500 08012345678',
        newState: 'transfer',
      };
    }

    default:
      return { message: 'Session ended. Thank you.', newState: 'done' };
  }
}
```

Note the character budget. `Invalid format. Use: 500 08012345678` is 39 characters, comfortably inside the limit. The balance screen concatenates a live value with the menu, so the balance string must be formatted to a fixed width or the menu can overflow. A defensive truncation helper is worth writing:

```typescript
export function screen(text: string, max = 160): string {
  return text.length <= max ? text : text.slice(0, max - 1) + '…';
}
```

Truncation is a last resort. If a screen is being truncated in production, the flow is wrong and the user is seeing a broken prompt.

## Step 3 — the Lambda handler

The handler loads the session, calls the state machine, persists the new state, and returns the aggregator's expected response shape. The exact JSON contract varies by aggregator; the shape below is representative.

```typescript
import {
  APIGatewayProxyEventV2,
  APIGatewayProxyStructuredResultV2,
} from 'aws-lambda';
import { handleUssd, Session, screen } from './ussd';
import { DynamoDBClient } from '@aws-sdk/client-dynamodb';
import {
  DynamoDBDocumentClient,
  GetCommand,
  PutCommand,
} from '@aws-sdk/lib-dynamodb';

const ddb = DynamoDBDocumentClient.from(new DynamoDBClient({}));
const TABLE = process.env.SESSION_TABLE!;
const SESSION_TTL_SECONDS = 300;

export const handler = async (
  event: APIGatewayProxyEventV2
): Promise<APIGatewayProxyStructuredResultV2> => {
  const body = JSON.parse(event.body ?? '{}');
  const { sessionId, phoneNumber, text } = body;

  if (!sessionId) {
    return { statusCode: 400, body: JSON.stringify({ error: 'missing sessionId' }) };
  }

  let session: Session | null = null;
  try {
    const res = await ddb.send(
      new GetCommand({ TableName: TABLE, Key: { id: sessionId } })
    );
    session = (res.Item as Session) ?? null;
  } catch (err) {
    console.error('session read failed', err);
    // Fail open to a safe screen rather than dropping the session silently.
    return {
      statusCode: 200,
      body: JSON.stringify({
        content: 'Service temporarily unavailable. Please try again.',
        continueSession: false,
      }),
    };
  }

  const { message, newState } = handleUssd(session, text);

  try {
    await ddb.send(
      new PutCommand({
        TableName: TABLE,
        Item: {
          id: sessionId,
          phoneNumber,
          state: newState,
          lastMessage: message,
          expiresAt: Math.floor(Date.now() / 1000) + SESSION_TTL_SECONDS,
        },
      })
    );
  } catch (err) {
    console.error('session write failed', err);
  }

  return {
    statusCode: 200,
    body: JSON.stringify({
      content: screen(message),
      continueSession: newState !== 'done',
    }),
  };
};
```

Two design decisions are worth calling out.

**Fail open on read errors.** If DynamoDB is unavailable, dropping the session gives the user nothing. Returning a neutral "try again" screen at least ends the interaction gracefully. The tradeoff is that a persistent read failure looks like a working service from the outside, which is why the error path must emit a metric (see the observability section).

**TTL on the session item.** USSD sessions are short-lived. Setting a TTL of a few minutes means abandoned sessions clean themselves up without a scheduled job. The TTL is a cleanup mechanism, not a correctness mechanism: never rely on TTL for session expiry logic, because DynamoDB deletes expired items on a best-effort basis.

## Step 4 — idempotency

Carriers retry. A retry that re-executes a transfer is a serious bug. The fix is to make the handler idempotent on a key the carrier supplies, or on a key you derive from the session and step.

```typescript
import { createHash } from 'crypto';

function idempotencyKey(sessionId: string, text: string | null): string {
  return createHash('sha256')
    .update(`${sessionId}:${text ?? ''}`)
    .digest('hex');
}
```

Store the key alongside the response, and check it before doing any side-effecting work:

```typescript
const key = idempotencyKey(sessionId, text);
const existing = await ddb.send(
  new GetCommand({ TableName: TABLE, Key: { id: `idem#${key}` } })
);

if (existing.Item) {
  return {
    statusCode: 200,
    body: JSON.stringify({
      content: existing.Item.lastMessage,
      continueSession: existing.Item.continueSession,
    }),
  };
}
```

The rule is simple: any operation that moves money or sends an SMS must be guarded by an idempotency check that runs before the side effect, and the result of the side effect must be persisted atomically with the key. If the key is written after the transfer, a crash between the two leaves you unable to tell whether the transfer happened.

**Failure mode analysis.** Consider three retry scenarios:

1. **Carrier retries before your first response.** The first invocation is still running. Without a lock, both invocations execute the transfer. A conditional write on the idempotency key (`attribute_not_exists`) makes the second invocation a no-op.
2. **Carrier retries after a successful response.** The key is present, so the stored response is returned. The user sees the same confirmation twice, which is confusing but not harmful.
3. **Your Lambda times out and the carrier retries.** The first invocation may still complete after the timeout. This is the dangerous case: the carrier has already given up, but your code has not. A conditional write plus a short internal timeout on the side-effecting call reduces the window, but does not eliminate it. Reconciliation against the ledger is the only complete answer.

## Step 5 — observability

USSD fails silently. The user sees a dropped session; you see nothing unless you instrument it. Emit a metric on every request with dimensions that let you slice by carrier and by state:

```typescript
import { CloudWatchClient, PutMetricDataCommand } from '@aws-sdk/client-cloudwatch';

const cw = new CloudWatchClient({});

async function emit(
  networkCode: string,
  state: string,
  latencyMs: number,
  success: boolean
): Promise<void> {
  await cw.send(
    new PutMetricDataCommand({
      Namespace: 'USSD/Flow',
      MetricData: [
        {
          MetricName: 'LatencyMs',
          Dimensions: [
            { Name: 'Carrier', Value: networkCode },
            { Name: 'State', Value: state },
          ],
          Value: latencyMs,
          Unit: 'Milliseconds',
        },
        {
          MetricName: 'SessionFailure',
          Dimensions: [{ Name: 'Carrier', Value: networkCode }],
          Value: success ? 0 : 1,
          Unit: 'Count',
        },
      ],
    })
  );
}
```

Alarm on p95 latency rather than average. Averages hide the tail, and the tail is what the carrier drops:

```typescript
import * as cw from 'aws-cdk-lib/aws-cloudwatch';
import * as cwActions from 'aws-cdk-lib/aws-cloudwatch-actions';
import * as sns from 'aws-cdk-lib/aws-sns';

const topic = new sns.Topic(this, 'UssdAlerts');

new cw.Alarm(this, 'HighLatencyAlarm', {
  metric: new cw.Metric({
    namespace: 'USSD/Flow',
    metricName: 'LatencyMs',
    statistic: 'p95',
    period: cdk.Duration.minutes(5),
  }),
  threshold: 1500,
  evaluationPeriods: 1,
  comparisonOperator: cw.ComparisonOperator.GREATER_THAN_THRESHOLD,
}).addAlarmAction(new cwActions.SnsAction(topic));
```

The threshold of 1500 ms is a choice, not a documented carrier limit. Set it below the carrier's timeout with enough margin that the alarm fires before users start seeing drops. If the carrier timeout is 20 seconds, alarming at 1.5 seconds p95 gives you a wide margin; if it is 5 seconds, tighten it.

**How to measure latency honestly.** A single number from a single machine is not a benchmark. To get a defensible figure:

1. Instrument the handler to record wall-clock time from request receipt to response return, and emit it as the `LatencyMs` metric.
2. Drive load with a tool that reports percentiles (p50, p95, p99), not just mean.
3. Measure from a vantage point that approximates the carrier edge, not from a developer laptop.
4. Compare regions by deploying the same stack to each and running the identical load profile.

The comparison table that matters is your own, produced by your own load test in your own regions.

## Step 6 — testing before carrier integration

Two layers of testing catch most problems before the carrier does.

**Unit tests on the state machine.** Because `handleUssd` is pure, these run in milliseconds:

```typescript
import { handleUssd } from '../src/ussd';

test('new session asks for PIN', () => {
  const res = handleUssd(null, null);
  expect(res.newState).toBe('auth');
  expect(res.message).toContain('PIN');
});

test('correct PIN advances to menu', () => {
  const res = handleUssd({ phoneNumber: '2348012345678', state: 'auth' }, '1234');
  expect(res.newState).toBe('menu');
});

test('malformed transfer input stays on transfer', () => {
  const res = handleUssd(
    { phoneNumber: '2348012345678', state: 'transfer' },
    'abc def'
  );
  expect(res.newState).toBe('transfer');
});

test('every screen fits the 160-character budget', () => {
  const screens = [
    handleUssd(null, null).message,
    handleUssd({ phoneNumber: '2348012345678', state: 'auth' }, '1234').message,
    handleUssd({ phoneNumber: '2348012345678', state: 'menu' }, '2').message,
  ];
  for (const s of screens) {
    expect(s.length).toBeLessThanOrEqual(160);
  }
});
```

The character-budget test is the one teams skip and regret. It is cheap and it catches the most common production defect.

**Load test against the deployed endpoint.** A simple scenario that posts realistic bodies at a steady rate:

```yaml
config:
  target: "https://YOUR_API_URL/"
  phases:
    - duration: 600
      arrivalRate: 20
scenarios:
  - flow:
      - post:
          url: "/"
          json:
            sessionId: "{{ $uuid }}"
            phoneNumber: "2348012345678"
            text: "1"
            networkCode: "621"
```

Run it and read the percentiles. If p95 is above your alarm threshold under load, the bottleneck is usually one of three things: Lambda cold starts, DynamoDB read latency, or a synchronous call to an external service (auth, fraud, SMS). Instrument the handler with sub-segments so you can attribute the time rather than guess.

**A note on load-test realism.** A load generator produces clean, well-formed requests at a steady rate. Real handsets produce retries, duplicate submissions, and bursts when a carrier's edge has a problem. Budget concurrency headroom above your measured peak, and test the retry path explicitly by replaying the same idempotency key.

## Decision checklist before going live

Work through this before the carrier integration window opens:

- [ ] Every screen is verified at or under 160 characters by an automated test.
- [ ] The state machine is pure and covered by unit tests for every transition, including invalid input.
- [ ] Session state is stored externally with a TTL, and no state lives in Lambda memory.
- [ ] Every side-effecting operation is guarded by an idempotency check that runs before the side effect.
- [ ] The idempotency key and the side-effect result are written atomically.
- [ ] Mutual TLS is configured with a certificate from a CA the carrier accepts, and rotation is scheduled.
- [ ] The AWS region is chosen for proximity to the carrier POP, and the choice is backed by a measured comparison.
- [ ] p95 latency is alarmed below the carrier timeout with margin.
- [ ] A read failure returns a graceful screen and emits a failure metric.
- [ ] The load test has been run at expected peak plus headroom, and the retry path has been replayed.

## FAQ

**Can I use Go or Rust instead of Node?**
Yes. The state machine is trivial in any language. The relevant difference is cold-start behavior: compiled runtimes generally start faster than interpreted ones, which matters if your traffic is bursty and your concurrency is spiky. Measure cold-start contribution to p95 in your own account before switching; the answer depends on your traffic shape more than on the runtime.

**How do I handle non-ASCII input?**
Assume the carrier will strip or mangle anything outside basic Latin and digits. Validate input against a strict allowlist and reject anything else with a clear retry prompt. Do not attempt to echo user input back without sanitizing it; a malformed character in a response can break the screen.

**What about dual-SIM users?**
Some aggregators pass a network or SIM indicator in the request payload; others do not. Do not build logic that depends on a field the carrier may omit. If you need to route an SMS to a specific SIM, confirm the field's availability with the carrier during integration and have a documented fallback.

**How do I test on a real handset cheaply?**
Acquire an inexpensive feature phone and a prepaid SIM from the target network. Walk the flow on the actual radio. Simulators do not reproduce the latency, the keypad behavior, or the carrier's session handling, and those are exactly the things that break.

## What to do in the next 30 minutes

Clone or create the project skeleton, write `src/ussd.ts` with the four-state machine above, and write the character-budget test. Run it. If every screen passes at or under 160 characters and every transition has a test, you have the part of the system that is hardest to fix later. Everything else — the CDK stack, the DynamoDB table, the alarm — is configuration you can add once the flow itself is correct.
