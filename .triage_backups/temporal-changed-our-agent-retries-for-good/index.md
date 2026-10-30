# Durable Execution for Multi-Step Agent Workflows

## The failure mode: retrying a step is not the same as retrying a workflow

Agent platforms commonly start out with cron jobs, message queues, or hand-rolled retry loops. That is adequate while an agent performs one bounded action. It stops being adequate once the agent orchestrates a sequence of side-effecting steps — for example, a refund that calls a payment gateway, then a fraud engine, then writes a row to PostgreSQL.

The assumption that trips teams up is this: *if a step fails, retrying from the same point will eventually succeed.* Three things break that assumption.

1. **Errors are not uniform over time.** A downstream API may return 500, 429, 409, or a 200 with a semantically failed body. Blind retry treats all of these identically.
2. **Agent state may have changed between attempts.** A retry that re-reads current state can act on data the original attempt never saw.
3. **Partial results may be invalid.** If step 2 succeeded and step 3 failed, the side effect of step 2 is still live. Retrying the whole sequence duplicates it.

The observable symptoms are cascading retries, duplicated side effects, and orphaned records that need manual reconciliation. Teams typically respond by adding idempotency keys, compensating transactions, and exponential backoff — all correct, but each one is a hand-built mechanism that must be reasoned about per workflow, and the reasoning gets harder as the number of steps grows.

Durable execution runtimes address this by moving retry, state persistence, and replay out of application code and into a runtime. The workflow function is written as ordinary sequential code; the runtime records each step's result and, after a crash, replays the workflow from the recorded history rather than re-executing completed steps. This article builds a refund workflow on that model, then examines where the model helps and where it does not.

## Prerequisites and what you will build

You will need:

- A Node.js 20 LTS project with TypeScript
- A durable execution runtime — either a managed service or a self-hosted cluster; the code below uses the Temporal SDK, but the concepts map to any runtime with the same execution model
- A database to store final results (PostgreSQL is used here)
- An event source for incoming work (any queue or HTTP endpoint will do)

What you will build: a refund workflow that calls three external services in sequence, writes the result to PostgreSQL, and retries failed steps without duplicating work. The workflow survives worker restarts, API timeouts, and transient errors without leaving half-processed refunds behind.

## Step 1 — project and worker setup

Initialize the project and install dependencies:

```bash
npm init -y
npm i -D typescript @types/node tsx
npx tsc --init
mkdir src
npm i @temporalio/worker @temporalio/client @temporalio/workflow
```

Create a worker in `src/worker.ts`:

```typescript
import { Worker } from '@temporalio/worker';
import * as activities from './activities';

async function run() {
  const worker = await Worker.create({
    workflowsPath: require.resolve('./workflows'),
    activities,
    taskQueue: 'refund-queue-v1',
  });
  await worker.run();
}

run().catch((err) => {
  console.error(err);
  process.exit(1);
});
```

Configure the connection to your runtime. A managed cluster normally requires mTLS on port 7233; a local development server listens on `localhost:7233` without TLS. The address and namespace are supplied through environment variables:

```bash
export TEMPORAL_ADDRESS=localhost:7233
export TEMPORAL_NAMESPACE=default
```

If your deployment requires client certificates, pass `tls` options to the connection object rather than relying on environment variables alone.

**Gotcha:** the port number is the same for local and managed deployments, but the transport is not. A worker pointed at a managed endpoint without TLS configuration fails at connection time, not at task execution time, so the error surfaces immediately on startup.

## Step 2 — the workflow and its activities

Define the workflow in `src/workflows.ts`. Note that the workflow function contains only orchestration logic — no network calls, no database access, no `Date.now()`, no randomness. Everything nondeterministic belongs in an activity.

```typescript
import {
  proxyActivities,
  defineSignal,
  setHandler,
  condition,
} from '@temporalio/workflow';
import type * as activities from './activities';
import type { RefundInput } from './types';

const { callPaymentGateway, callFraudEngine, writeToPostgres, reversePayment } =
  proxyActivities<typeof activities>({
    startToCloseTimeout: '30 seconds',
    retry: {
      maximumAttempts: 5,
      initialInterval: '1 second',
      maximumInterval: '10 seconds',
      backoffCoefficient: 2,
    },
  });

export const cancelSignal = defineSignal<[boolean]>('cancel');

export async function refundWorkflow(input: RefundInput): Promise<void> {
  let cancelled = false;
  setHandler(cancelSignal, () => {
    cancelled = true;
  });

  const paymentResult = await callPaymentGateway(input.paymentId, input.idempotencyKey);
  if (paymentResult.status === 'DECLINED') {
    throw new Error('Payment declined');
  }

  if (cancelled) {
    await reversePayment(input.paymentId, input.idempotencyKey);
    return;
  }

  const fraudResult = await callFraudEngine(input.userId, input.amount);
  if (fraudResult.status === 'REJECTED') {
    await reversePayment(input.paymentId, input.idempotencyKey);
    return;
  }

  await writeToPostgres({
    refundId: input.refundId,
    userId: input.userId,
    amount: input.amount,
    status: 'COMPLETED',
    timestamp: input.requestedAt,
  });
}
```

Two details matter here. First, `input.requestedAt` is passed in as data rather than computed inside the workflow, because calling `new Date()` inside a workflow body is nondeterministic and will diverge on replay. Second, cancellation is handled by checking a flag and performing compensation explicitly — throwing from a signal handler is not a reliable way to unwind a workflow, because the handler runs on a different code path than the main workflow body.

Implement the activities in `src/activities.ts`:

```typescript
import { createPool, sql } from 'slonik';
import type { RefundRecord } from './types';

const pool = createPool(process.env.DATABASE_URL!, {
  maximumPoolSize: 10,
  idleTimeout: 30,
  connectionTimeout: 2,
});

export async function callPaymentGateway(
  paymentId: string,
  idempotencyKey: string,
): Promise<{ status: 'APPROVED' | 'DECLINED' }> {
  const res = await fetch(
    `https://payment.example.com/api/v1/payments/${paymentId}/refund`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-KEY': process.env.PAYMENT_API_KEY!,
        'Idempotency-Key': idempotencyKey,
      },
    },
  );
  if (!res.ok) throw new Error(`Payment API ${res.status}`);
  return (await res.json()) as { status: 'APPROVED' | 'DECLINED' };
}

export async function callFraudEngine(
  userId: string,
  amount: number,
): Promise<{ status: 'APPROVED' | 'REJECTED' }> {
  const res = await fetch('https://fraud.example.com/v1/check', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-API-KEY': process.env.FRAUD_API_KEY!,
    },
    body: JSON.stringify({ userId, amount }),
  });
  if (!res.ok) throw new Error(`Fraud API ${res.status}`);
  return (await res.json()) as { status: 'APPROVED' | 'REJECTED' };
}

export async function reversePayment(
  paymentId: string,
  idempotencyKey: string,
): Promise<void> {
  const res = await fetch(
    `https://payment.example.com/api/v1/payments/${paymentId}/reversal`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-KEY': process.env.PAYMENT_API_KEY!,
        'Idempotency-Key': `${idempotencyKey}-reversal`,
      },
    },
  );
  if (!res.ok) throw new Error(`Reversal API ${res.status}`);
}

export async function writeToPostgres(record: RefundRecord): Promise<void> {
  await pool.query(sql`
    INSERT INTO refunds (refund_id, user_id, amount, status, timestamp)
    VALUES (${record.refundId}, ${record.userId}, ${record.amount},
            ${record.status}, ${record.timestamp})
    ON CONFLICT (refund_id) DO NOTHING
  `);
}
```

Create the table:

```sql
CREATE TABLE refunds (
  refund_id   TEXT PRIMARY KEY,
  user_id     TEXT NOT NULL,
  amount      NUMERIC NOT NULL,
  status      TEXT NOT NULL,
  timestamp   TIMESTAMPTZ NOT NULL
);

CREATE INDEX idx_refunds_user_id ON refunds(user_id);
```

Start the worker:

```bash
npx tsx src/worker.ts
```

**Timeout relationship.** Activity timeouts should be shorter than the workflow's overall timeout. If the workflow has a 5-minute execution timeout and each activity has a 2-minute `startToCloseTimeout`, a slow activity is retried by the runtime without restarting the workflow. If the activity timeout exceeds the workflow timeout, the workflow will be terminated while the activity is still in flight, and the activity's result is discarded.

## Step 3 — edge cases that actually bite

### Idempotency keys must be generated once per workflow run

The key must be stable across retries. Generate it at workflow start, from data already in the input, and pass it to every activity that performs a side effect:

```typescript
export async function refundWorkflow(input: RefundInput): Promise<void> {
  const idempotencyKey = `refund-${input.refundId}`;
  // ... pass idempotencyKey to every activity
}
```

Do not generate the key inside an activity. If the activity is retried, a freshly generated key defeats the purpose entirely — the downstream service sees a new request rather than a duplicate of the old one.

### Compensation must be explicit

If a later step fails, earlier side effects remain. The workflow must reverse them. Note that the reversal itself is an activity and therefore retryable:

```typescript
let paymentApproved = false;

try {
  const paymentResult = await callPaymentGateway(input.paymentId, idempotencyKey);
  paymentApproved = true;

  const fraudResult = await callFraudEngine(input.userId, input.amount);
  if (fraudResult.status === 'REJECTED') {
    throw new Error('Fraud check failed');
  }

  await writeToPostgres({ /* ... */ });
} catch (err) {
  if (paymentApproved) {
    await reversePayment(input.paymentId, idempotencyKey);
  }
  throw err;
}
```

The `paymentApproved` flag lives in workflow state, which is persisted. If the worker crashes mid-workflow, replay restores the flag correctly before the catch block re-executes.

### Transient database errors

PostgreSQL error codes `40P01` (deadlock detected) and `40001` (serialization failure) are safe to retry. Connection-level codes such as `57P01` and `57P02` indicate the server is shutting down and the connection is gone; the client should reconnect rather than retry on the same connection. A Slonik interceptor can distinguish these:

```typescript
import { createPool, createTypeParserPreset } from 'slonik';

const TRANSIENT = new Set(['40P01', '40001']);

export const retryOnTransientError = {
  name: 'retry-transient',
  queryExecutionError: async (ctx, query, error) => {
    const code = (error as { code?: string }).code;
    if (code && TRANSIENT.has(code)) {
      // Let the caller's retry policy handle it.
      throw error;
    }
    throw error;
  },
};
```

**Failure-mode note:** the interceptor above does not itself retry; it classifies. Retrying at two layers — the interceptor and the activity retry policy — produces multiplicative retry counts. Pick one layer. For database work inside a durable workflow, the activity retry policy is usually the better choice, because the runtime records the attempt count and visibility tooling can show it.

### Cancellation

Send a signal from a client process:

```typescript
import { Client, Connection } from '@temporalio/client';

const connection = await Connection.connect();
const client = new Client({ connection });

await client.workflow.getHandle(workflowId).signal(cancelSignal, true);
```

The workflow's main body observes the flag at the next await point and runs compensation. A signal handler that throws does not reliably unwind the workflow, because the handler executes outside the main body's call stack.

**Common mistake:** omitting a workflow execution timeout. Without one, a workflow stuck on a step that never resolves occupies a worker slot indefinitely. Set a `workflowExecutionTimeout` appropriate to the domain — minutes for automated refunds, days for workflows that wait on human approval.

## Step 4 — observability and testing

### Visibility

The runtime's web UI and CLI expose workflow state directly: which workflows are running, which are retrying, which have failed, and the full event history of each. For a refund workflow, the useful filters are by workflow type, by status, and by start time.

### Metrics

The worker SDK exposes metrics that can be scraped by Prometheus:

```typescript
import { Runtime, DefaultLogger } from '@temporalio/worker';

Runtime.install({
  logger: new DefaultLogger('info'),
  metrics: {
    otel: {
      metricsExporter: { url: process.env.OTEL_EXPORTER_URL },
    },
  },
});
```

The metrics worth alerting on:

- **Task queue backlog** — the number of pending tasks relative to worker capacity. This is the leading indicator that workers are undersized.
- **Activity failure rate by activity type** — a spike isolated to one activity points at a downstream dependency, not at the worker.
- **Workflow task failure rate** — this indicates nondeterminism, which is a code defect, not a load problem.
- **Schedule-to-start latency** — how long work waits before a worker picks it up. This is the metric that correlates with user-visible delay.

### Testing

The runtime provides a test environment that runs workflows and activities in-process without a cluster:

```typescript
import { TestWorkflowEnvironment } from '@temporalio/testing';
import { Worker } from '@temporalio/worker';
import { refundWorkflow } from './workflows';

describe('refundWorkflow', () => {
  let env: TestWorkflowEnvironment;
  let worker: Worker;

  beforeAll(async () => {
    env = await TestWorkflowEnvironment.createLocal();
  });

  afterAll(async () => {
    await env?.teardown();
  });

  it('completes a refund end to end', async () => {
    worker = await Worker.create({
      connection: env.nativeConnection,
      taskQueue: 'test-refund-queue',
      workflowsPath: require.resolve('./workflows'),
      activities: {
        callPaymentGateway: async () => ({ status: 'APPROVED' }),
        callFraudEngine: async () => ({ status: 'APPROVED' }),
        writeToPostgres: async () => undefined,
        reversePayment: async () => undefined,
      },
    });

    await worker.runUntil(async () => {
      const handle = await env.client.workflow.start(refundWorkflow, {
        taskQueue: 'test-refund-queue',
        workflowId: 'test-refund-wf-1',
        args: [{
          refundId: 'r-123',
          paymentId: 'p-456',
          userId: 'u-789',
          amount: 100,
          idempotencyKey: 'refund-r-123',
          requestedAt: '2025-01-01T00:00:00.000Z',
        }],
      });
      await handle.result();
    });
  });

  it('compensates when the fraud check rejects', async () => {
    const reversed: string[] = [];
    worker = await Worker.create({
      connection: env.nativeConnection,
      taskQueue: 'test-refund-queue',
      workflowsPath: require.resolve('./workflows'),
      activities: {
        callPaymentGateway: async () => ({ status: 'APPROVED' }),
        callFraudEngine: async () => ({ status: 'REJECTED' }),
        writeToPostgres: async () => undefined,
        reversePayment: async (paymentId: string) => {
          reversed.push(paymentId);
        },
      },
    });

    await worker.runUntil(async () => {
      const handle = await env.client.workflow.start(refundWorkflow, {
        taskQueue: 'test-refund-queue',
        workflowId: 'test-refund-wf-2',
        args: [{
          refundId: 'r-124',
          paymentId: 'p-457',
          userId: 'u-790',
          amount: 100,
          idempotencyKey: 'refund-r-124',
          requestedAt: '2025-01-01T00:00:00.000Z',
        }],
      });
      await handle.result().catch(() => undefined);
    });

    expect(reversed).toEqual(['p-457']);
  });
});
```

The second test is the one that matters. Happy-path tests pass on systems with broken compensation logic; the failure-path test is what catches it.

## How to measure whether this is actually better

Any claim that a durable execution runtime reduces latency or error rate is a claim about *your* system, and the only way to know is to measure it. Here is what to instrument.

**Baseline, before migration:**

- Record, per workflow instance, the wall-clock time from first attempt to terminal state (success or abandoned).
- Count duplicate side effects: rows in the downstream system with the same logical operation ID but different request IDs.
- Count orphans: records left in an intermediate state past a defined staleness threshold.
- Record the number of manual interventions per week (support tickets, scripts run against production data).

**After migration, measure the same four things.** Keep the measurement window the same length and, ideally, the same traffic mix. If you cannot hold traffic constant, report the numbers as rates rather than counts.

**What to expect, and why.** The p95 latency for a *successful* workflow should improve mainly because steps that previously triggered a full-sequence retry now retry only the failed step. The improvement scales with the number of steps and the failure rate of the later steps — if almost all failures happen at step 1, there is little to gain. The duplicate-side-effect count should go to zero if idempotency keys are implemented correctly; if it does not, the bug is in the key generation, not in the runtime.

**Cost.** Durable execution runtimes typically charge per action or per task, plus storage for event history. A hand-rolled Lambda-plus-queue system charges per invocation and per message. Which is cheaper depends on how many steps each workflow has and how long it waits between them. Workflows that spend most of their time waiting (human approval, long polling) are usually cheaper on a durable execution runtime, because waiting does not consume compute. Workflows that are a single short step are usually cheaper on a queue.

## Choosing between approaches

| Property | Durable execution runtime | Queue + retry loop | Managed step functions |
|---|---|---|---|
| State persistence | Runtime-managed, per workflow | Application-managed | Runtime-managed |
| Deterministic replay | Required (constrains workflow code) | N/A | N/A |
| Retry granularity | Per activity | Per message | Per state |
| Compensation | Explicit, in code | Explicit, in code | Explicit, in state machine |
| Visibility into in-flight work | Full event history | Depends on queue tooling | Full execution history |
| Operational overhead | Cluster or managed service | Low | Low |
| Best fit | Multi-step, long-lived, cross-service | Single-step, short-lived | AWS-native, bounded workflows |

The constraint worth understanding before committing is **determinism**. Workflow code must produce the same sequence of commands on replay as it did on the original execution. That means no `Date.now()`, no `Math.random()`, no direct network calls, and no iteration over unordered collections inside the workflow body. These rules are enforced by a sandbox at runtime, and violations produce a nondeterminism error that terminates the workflow. This is a real cost: it forces a discipline that ordinary application code does not require.

The benefit is that the workflow body reads as linear code — `await stepA(); await stepB(); await stepC();` — while the runtime handles persistence, retry, and recovery. For workflows with more than a handful of side-effecting steps, that trade is usually worth making. For a single step, it is not.

## FAQ

**What happens if the worker crashes mid-activity?**
The runtime records activity heartbeats. When a worker restarts, the activity is either resumed if it was heartbeating, or retried from the start if it was not. Activities that do not heartbeat are retried from the beginning, so they must be idempotent.

**Can this be done in Python or Go?**
Yes. The execution model — workflows, activities, signals, queries — is the same across SDKs. Workflow and activity definitions are not portable between languages, but the design is.

**Why not use a managed step function service?**
Managed step functions are a good fit when the workflow is confined to one cloud provider and the steps are all provider-native services. Durable execution runtimes are a better fit when the workflow spans multiple providers, calls arbitrary HTTP services, or needs to run for months.

**How does this interact with an existing queue?**
It sits in front of it. The queue delivers a message, a small handler starts a workflow with the message as input, and the workflow takes over from there. The queue is responsible for delivery; the runtime is responsible for execution.

**Does the workflow need a timeout?**
Yes. Set a `workflowExecutionTimeout` for the overall bound and a `workflowRunTimeout` for a single run if the workflow can be continued-as-new. Without a timeout, a workflow waiting on something that never arrives runs indefinitely.

## Do this in the next 30 minutes

Pick one workflow in your system that has more than two sequential side-effecting steps. Write down, for each step, what happens if it succeeds and a later step fails. If any step's answer is "we would have to clean that up manually," you have found the exact place where a durable execution runtime earns its keep — and you have a concrete test case to write before you migrate anything.
