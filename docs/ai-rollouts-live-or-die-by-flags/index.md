# AI rollouts live or die by flags

Most AI tutorials stop at the happy path: a prompt, a model call, a response. Production is different. Model versions get deprecated, prompt templates regress, safety filters misfire on a subset of inputs, and a temperature change can shift output quality in ways that only show up in aggregate. The practical question is not "how do I call a model" but "how do I change what the model does, for a subset of traffic, without a full redeploy and without a long incident."

Feature flags answer that question. A flag is a small piece of remote configuration — usually a boolean plus optional variant and targeting rules — that an application reads at runtime. Teams commonly put every inference call, prompt template, and safety filter behind one. The alternative is rebuilding and redeploying the inference stack for each tweak, which turns a five-second kill switch into a multi-hour rollback.

This article walks through a self-hosted flag service on AWS Lambda with DynamoDB, a Node client SDK, timeout and circuit-breaker handling, observability, tests, and a canary rollout pattern. It uses Node 20 LTS and ARM64 Lambda. Nothing here is specific to a particular model vendor; the flag layer sits in front of whatever inference client you already use.

## What you'll build

1. A small flag-evaluation service: an AWS Lambda function backed by a DynamoDB table, defined with AWS CDK in TypeScript.
2. A Node SDK that your inference code imports to decide which model, prompt version, or safety filter to use.
3. Client-side timeout and circuit-breaker logic so a slow flag service cannot take down inference.
4. Metrics and a CloudWatch dashboard, plus unit tests for the SDK.
5. A canary rollout pattern: 5% of traffic, then 50%, then 100%, with a kill switch at each step.

Prerequisites: Node 20 LTS locally, an AWS account with permission to create Lambda, DynamoDB, IAM, and CloudWatch resources, and basic familiarity with GitHub Actions for CI.

## Step 1 — Set up the infrastructure

AWS CDK defines the stack in code. Install it once, then bootstrap the account and create an app.

```bash
npm install -g aws-cdk
cdk bootstrap aws://ACCOUNT-NUMBER/REGION
mkdir ai-flag-service && cd ai-flag-service
cdk init app --language typescript
```

Edit `lib/ai-flag-service-stack.ts`. This creates a DynamoDB table with on-demand capacity, a Lambda function, and the IAM permission the function needs to read the table.

```typescript
import * as cdk from 'aws-cdk-lib';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as logs from 'aws-cdk-lib/aws-logs';

interface Props extends cdk.StackProps {
  stage: string;
}

export class AiFlagServiceStack extends cdk.Stack {
  constructor(scope: cdk.App, id: string, props: Props) {
    super(scope, id, props);

    const table = new dynamodb.Table(this, 'FlagsTable', {
      partitionKey: { name: 'flagId', type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      timeToLiveAttribute: 'expiresAt',
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    const fn = new lambda.Function(this, 'FlagEvaluator', {
      runtime: lambda.Runtime.NODEJS_20_X,
      architecture: lambda.Architecture.ARM_64,
      handler: 'index.handler',
      code: lambda.Code.fromAsset('lambda'),
      environment: {
        TABLE_NAME: table.tableName,
        STAGE: props.stage,
      },
      timeout: cdk.Duration.seconds(5),
      logRetention: logs.RetentionDays.ONE_MONTH,
    });

    table.grantReadData(fn);

    new cdk.CfnOutput(this, 'FunctionName', { value: fn.functionName });
    new cdk.CfnOutput(this, 'TableName', { value: table.tableName });
  }
}
```

Create the handler in `lambda/index.ts`. It reads a flag by ID and returns its state. Targeting logic is deliberately minimal here; the point is the shape of the contract, not a full rules engine.

```typescript
import { DynamoDBClient, GetItemCommand } from '@aws-sdk/client-dynamodb';
import { unmarshall } from '@aws-sdk/util-dynamodb';

const client = new DynamoDBClient({ region: process.env.AWS_REGION });
const TABLE_NAME = process.env.TABLE_NAME!;

export const handler = async (event: any) => {
  const { flagId, userId, context = {} } = event;

  if (!flagId || !userId) {
    return { error: 'Missing flagId or userId' };
  }

  const key = { flagId: { S: flagId } };
  const cmd = new GetItemCommand({ TableName: TABLE_NAME, Key: key });
  const res = await client.send(cmd);

  if (!res.Item) {
    return { enabled: false, reason: 'Flag not found' };
  }

  const item = unmarshall(res.Item);
  const enabled = item.enabled as boolean;
  const targeting = item.targeting as Record<string, unknown> | undefined;

  if (!targeting || Object.keys(targeting).length === 0) {
    return { enabled, variant: item.variant || 'default' };
  }

  // Extend with your own targeting rules, e.g. userId in targeting.userIds.
  return { enabled, variant: item.variant || 'default' };
};
```

Install dependencies and deploy.

```bash
npm install @aws-sdk/client-dynamodb @aws-sdk/util-dynamodb
cdk deploy --context stage=prod
```

The output includes the function name. Save it; the SDK needs it.

One operational note: with `RemovalPolicy.DESTROY`, deleting the stack deletes the table, and DynamoDB table deletion is not instantaneous. Redeploying immediately after a destroy can fail while the old table is still in `DELETING`. For development stacks, either wait for the table to disappear or switch the removal policy to `RETAIN` and clean up manually.

## Step 2 — Build the client SDK

The SDK wraps the Lambda invocation, records latency, and gives inference code a single function to call.

```bash
mkdir ai-flag-sdk && cd ai-flag-sdk
npm init -y
npm install @aws-sdk/client-lambda @aws-sdk/client-cloudwatch
```

Edit `src/index.ts`:

```typescript
import { LambdaClient, InvokeCommand } from '@aws-sdk/client-lambda';
import { CloudWatchClient, PutMetricDataCommand } from '@aws-sdk/client-cloudwatch';

const lambda = new LambdaClient({ region: process.env.AWS_REGION });
const cloudwatch = new CloudWatchClient({ region: process.env.AWS_REGION });
const FUNCTION_NAME = process.env.FLAG_FUNCTION_NAME!;

interface FlagOptions {
  flagId: string;
  userId: string;
  context?: Record<string, unknown>;
}

interface FlagResult {
  enabled: boolean;
  variant?: string;
}

async function evaluateFlagInternal(options: FlagOptions): Promise<FlagResult> {
  const payload = {
    flagId: options.flagId,
    userId: options.userId,
    context: options.context,
  };

  const cmd = new InvokeCommand({
    FunctionName: FUNCTION_NAME,
    Payload: JSON.stringify(payload),
  });

  const res = await lambda.send(cmd);

  if (res.StatusCode !== 200 || !res.Payload) {
    throw new Error(`Flag evaluation failed: ${res.StatusCode}`);
  }

  return JSON.parse(Buffer.from(res.Payload).toString('utf-8'));
}

export async function evaluateFlag(options: FlagOptions): Promise<FlagResult> {
  const start = Date.now();
  const result = await evaluateFlagInternal(options);
  const latency = Date.now() - start;

  await cloudwatch.send(
    new PutMetricDataCommand({
      Namespace: 'AI/FeatureFlags',
      MetricData: [
        {
          MetricName: 'LatencyMs',
          Dimensions: [{ Name: 'FlagId', Value: options.flagId }],
          Value: latency,
          Unit: 'Milliseconds',
        },
      ],
    })
  );

  return result;
}
```

Note the split: `evaluateFlagInternal` does the work, and `evaluateFlag` wraps it with metrics. That separation matters in Step 3, where the circuit breaker needs a function to wrap.

### Wire it into an inference handler

The flag layer decides which model and which prompt template to use. The inference code does not need to know how flags are stored.

```typescript
import { evaluateFlag } from 'ai-flag-sdk';
import { BedrockRuntimeClient, InvokeModelCommand } from '@aws-sdk/client-bedrock-runtime';

export const handler = async (event: any) => {
  const { userId, message } = event;

  const modelFlag = await evaluateFlag({ flagId: 'ai-model-v2', userId });
  const promptFlag = await evaluateFlag({ flagId: 'ai-prompt-v3', userId });

  const modelId = modelFlag.enabled
    ? process.env.MODEL_ID_PRIMARY!
    : process.env.MODEL_ID_FALLBACK!;
  const promptTemplate = promptFlag.enabled ? 'prompt-v3.jinja' : 'prompt-v2.jinja';

  const client = new BedrockRuntimeClient({ region: process.env.AWS_REGION });
  const cmd = new InvokeModelCommand({
    modelId,
    body: JSON.stringify({ prompt: `Use template ${promptTemplate}: ${message}` }),
  });

  const res = await client.send(cmd);
  const output = JSON.parse(Buffer.from(res.body).toString('utf-8'));

  return { ticket: output.summary };
};
```

Two details are worth calling out. First, model IDs come from environment variables rather than string literals, because model identifiers change and hardcoding them across a codebase is a maintenance problem. Second, the flag call is on the critical path of the request. That is why the next section exists.

## Step 3 — Handle timeouts, failures, and caching

A flag service that is slow or unavailable must not make inference slow or unavailable. There are three layers of defense: a client-side timeout, a circuit breaker, and a local cache.

### Client-side timeout

The evaluator Lambda has a 5-second timeout. A client-side timeout shorter than that bounds how long inference can be blocked.

```typescript
export async function evaluateFlag(
  options: FlagOptions,
  timeoutMs = 2000
): Promise<FlagResult> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    return await Promise.race([
      evaluateFlagInternal(options),
      new Promise<FlagResult>((_, reject) => {
        controller.signal.addEventListener('abort', () =>
          reject(new Error('Flag evaluation timed out'))
        );
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}
```

### Circuit breaker

Timeouts bound a single call. A circuit breaker bounds repeated calls to a dependency that is already failing, so inference does not pay the timeout on every request.

```typescript
import CircuitBreaker from 'opossum';

const breaker = new CircuitBreaker(evaluateFlagInternal, {
  timeout: 2000,
  errorThresholdPercentage: 50,
  resetTimeout: 30000,
});

export async function evaluateFlag(options: FlagOptions): Promise<FlagResult> {
  try {
    return await breaker.fire(options);
  } catch (err) {
    console.error('Flag evaluation failed, using default', err);
    return { enabled: false, variant: 'default' };
  }
}
```

The fallback matters as much as the breaker. Returning `enabled: false` means new behavior stays off during an outage, which is the safer default for a rollout. It is not the right default for every flag — a flag that turns a safety filter *on* should fail closed in the opposite direction. Decide per flag which state is safe when the flag service is unreachable, and encode that in the flag record rather than in the call site.

### Local cache

A short-lived in-process cache absorbs bursts and reduces DynamoDB reads.

```typescript
import NodeCache from 'node-cache';

const cache = new NodeCache({ stdTTL: 5 });

export async function evaluateFlag(options: FlagOptions): Promise<FlagResult> {
  const cacheKey = `${options.flagId}:${options.userId}`;
  const cached = cache.get<FlagResult>(cacheKey);
  if (cached) return cached;

  const result = await breaker.fire(options);
  cache.set(cacheKey, result);
  return result;
}
```

A 5-second TTL is a deliberate trade-off: it caps how long a kill switch takes to propagate. If a flag must take effect instantly, skip the cache for that flag or invalidate on write. Document the propagation delay next to the kill-switch runbook, because an operator who expects instant effect and gets five seconds of stale behavior will misread the incident.

### Flag schema

Validate flags on write so a malformed record cannot silently disable a feature.

```typescript
interface FeatureFlag {
  flagId: string;
  enabled: boolean;
  variant?: string;
  targeting?: {
    userIds?: string[];
    userSegments?: string[];
    percentage?: number;
  };
  expiresAt?: number;
  createdAt: number;
  updatedAt: number;
}
```

A CI job can read every row and assert it matches this shape. That catches targeting rules referencing segments that no longer exist, which is a common source of flags that quietly evaluate to `false` for everyone.

## Step 4 — Observability and tests

### Metrics worth emitting

Latency and error rate are the minimum. For AI rollouts, four signals are more useful:

- **Flag evaluation latency**, per flag, so a slow flag is visible before it slows inference.
- **Enabled ratio**, per flag, so an unexpected flip to 100% or 0% is visible immediately.
- **Variant distribution**, so a canary at 5% can be confirmed to actually be at 5%.
- **Fallback rate**, the count of evaluations that returned the default because of a timeout or open breaker. A rising fallback rate is the earliest signal that the flag layer is degrading.

The SDK emits the latency metric above. The same `PutMetricData` call pattern covers the others. A CloudWatch dashboard combining these four is enough to run a rollout.

### Unit tests

Tests should cover the contract, not the AWS SDK. Mock the client and assert on the SDK's behavior.

```typescript
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { evaluateFlag } from './index';
import { LambdaClient } from '@aws-sdk/client-lambda';

vi.mock('@aws-sdk/client-lambda');

beforeEach(() => {
  vi.clearAllMocks();
});

describe('evaluateFlag', () => {
  it('returns enabled false when the flag is not found', async () => {
    LambdaClient.prototype.send = vi.fn().mockResolvedValue({
      StatusCode: 200,
      Payload: Buffer.from(JSON.stringify({ enabled: false, reason: 'Flag not found' })),
    });
    const res = await evaluateFlag({ flagId: 'missing', userId: 'u1' });
    expect(res.enabled).toBe(false);
  });

  it('returns enabled true and the variant when the flag is found', async () => {
    LambdaClient.prototype.send = vi.fn().mockResolvedValue({
      StatusCode: 200,
      Payload: Buffer.from(JSON.stringify({ enabled: true, variant: 'beta' })),
    });
    const res = await evaluateFlag({ flagId: 'test', userId: 'u1' });
    expect(res.enabled).toBe(true);
    expect(res.variant).toBe('beta');
  });
});
```

Run in CI with `npx vitest run`. Add a test for the timeout path and one for the fallback path; those are the branches that matter during an incident and the ones least likely to be exercised manually.

## Canary rollouts for model and prompt changes

The pattern: a new model or prompt ships behind a flag that targets a small percentage of users, and that percentage is increased only after the metrics look right.

| Stage | Traffic | Minimum observation | Advance if | Roll back if |
|---|---|---|---|---|
| Canary | 5% | 30 minutes | Error rate within baseline; latency p95 within baseline | Error rate above baseline or any safety-filter failure |
| Partial | 50% | 1 hour | Same, plus no quality regression in sampled outputs | Same |
| Full | 100% | 24 hours | Sampling shows no drift | Same |

The percentages and durations are illustrative. The important part is that each stage has a stated advance condition and a stated rollback condition, decided before the rollout starts. "We will look at the dashboard" is not a condition.

Two things make this work in practice. First, the flag evaluation must be weighted consistently for a given user, or a user can flip between variants across requests and produce confusing telemetry. Second, quality is not the same as error rate. A model can return HTTP 200 with worse output. Sampling a fixed number of outputs per stage and reviewing them is the only reliable check; automated quality scoring is a separate system and out of scope here.

### How to measure whether flags actually help

Claims about incident reduction are easy to make and hard to verify. If you want a number, instrument it:

- Tag every incident with whether a flag change was involved in detection, mitigation, or cause.
- Record time-to-mitigation per incident, defined as the interval from first alert to the change that stopped the bleeding.
- Compare the distribution of time-to-mitigation for flag-mediated mitigations against redeploy-mediated mitigations.

This produces a defensible figure for your own environment. It will not match anyone else's, because it depends on your deploy pipeline, your on-call process, and how much of the system is behind flags.

## Managed services versus self-hosting

A managed flag service is a reasonable choice, and for many teams it is the right one. The comparison below is a decision aid, not a benchmark; latency and cost depend on your region, call volume, and network topology, and should be measured in your own environment.

| Consideration | Self-hosted | Managed service |
|---|---|---|
| Latency | One in-region Lambda invocation plus a DynamoDB read | Network call to a third party; measure from your region |
| Cost model | Lambda invocations + DynamoDB reads + CloudWatch metrics | Per-seat or per-evaluation pricing |
| Data residency | Full control; data stays in your account and region | Depends on the vendor's regions and contract |
| Audit trail | You build it | Usually built in |
| Targeting rules | You build them | Usually richer out of the box |
| Operational burden | You own uptime, scaling, and schema | Vendor owns it |
| Kill-switch latency | Bounded by your cache TTL | Bounded by the vendor's propagation guarantees |

The decision usually comes down to two questions. Do you have a hard data-residency or audit requirement that a vendor cannot meet? And do you have the on-call capacity to own another service? Self-hosting is not free; it is a service you now operate.

## Keeping flags from becoming permanent

Flags accumulate. A flag that stays at 100% for a year is dead configuration that still costs a read on every request and still has to be reasoned about during incidents. A workable lifecycle:

1. Every flag has an owner and a purpose, recorded at creation.
2. Every flag has a removal date set when it is created, not when someone remembers.
3. A scheduled job reports flags that have not changed state in 90 days.
4. Removing a flag is a normal PR, reviewed like any other change.

The removal date is the part teams skip. Setting it at creation is cheap; reconstructing intent six months later is not.

## Where to go from here

Open `lib/ai-flag-service-stack.ts` and add a second CloudWatch alarm: fire when the SDK's fallback metric exceeds 1% of evaluations over a 5-minute window. That alarm is the difference between noticing a degraded flag service from a dashboard and noticing it from a customer report. Then create a flag named `ai-safety-filter-v1`, set it to target a small percentage of users, and confirm from the variant-distribution metric that the percentage you configured is the percentage you are actually getting — a mismatch there means the targeting logic and the telemetry disagree, and that is worth finding on a quiet day rather than during a rollout.
