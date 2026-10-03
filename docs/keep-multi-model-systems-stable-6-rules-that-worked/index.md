# Stabilize multi-model agent routers: six production rules

Multi-model agent routers fail in ways that are boring and predictable: unbounded fan-out on retries, duplicate side effects, timeouts that outlive the caller, and caches that quietly stop being shared. The model, the prompt, and the vector store are usually not the problem. The plumbing is.

This article walks through six rules that hold up in production, with runnable code for a minimal router on AWS Lambda, Redis, and OpenTelemetry. Each rule gets a "how to measure it" note, because a rule you cannot measure is a rule you will not keep.

## Prerequisites and what you'll build

You need a project that already has:

- Node 20 LTS with TypeScript tests (Python 3.11 works equally well; the patterns are language-agnostic).
- AWS Lambda on arm64, with provisioned concurrency off initially.
- Redis 7.2 for shared state and rate limiting.
- An OpenTelemetry collector and a Prometheus endpoint for metrics.

You will build a minimal agent router that:

1. Receives events via API Gateway HTTP API.
2. Routes each event to one of three backends: a small quantised model, an external SaaS LLM, or a vector similarity endpoint.
3. Enforces a concurrency budget per window.
4. Retries with exponential backoff capped at three attempts.
5. Emits structured logs and traces so you can see where time is spent.

The goal is a router whose tail latency is bounded by your limits, not by your traffic.

## Rule 1 — Bound concurrency at the router, not the runtime

Serverless runtimes scale out; they do not scale *gracefully*. When a downstream dependency returns 503, retry logic that spawns a new invocation per retry turns one failure into a fan-out. A typical failure mode is a retry storm: each failed call schedules three more, and the queue depth grows faster than the autoscaler can react.

Bounded concurrency means an explicit ceiling on in-flight work, enforced before you call the model.

### Step 1 — set up the environment

```bash
mkdir agent-router && cd agent-router
npm init -y
npm install typescript @types/node --save-dev
npx tsc --init
npm install @opentelemetry/sdk-node @opentelemetry/auto-instrumentations-node \
  winston winston-transport-http @aws-sdk/client-lambda redis express-pino-logger pino pino-pretty
```

Create a `.env` file:

```ini
REDIS_URL=redis://cluster.example.cache.amazonaws.com:6379
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
AWS_REGION=us-east-1
AGENT_CONCURRENCY_LIMIT=100
```

Run the OpenTelemetry collector locally for development:

```bash
docker run -d --name otel-collector \
  -p 4317:4317 -p 4318:4318 -p 8888:8888 \
  otel/opentelemetry-collector-contrib:0.88.0 \
  --config=./otel-config.yaml
```

Create `otel-config.yaml`:

```yaml
receivers:
  otlp:
    protocols:
      grpc:
      http:

processors:
  batch:
  memory_limiter:
    limit_mib: 128
    spike_limit_mib: 32
    check_interval: 1s

exporters:
  logging:
    loglevel: debug
  prometheus:
    endpoint: "0.0.0.0:8889"

extensions:
  health_check:

service:
  extensions: [health_check]
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch, memory_limiter]
      exporters: [logging, prometheus]
    metrics:
      receivers: [otlp]
      processors: [batch, memory_limiter]
      exporters: [logging, prometheus]
```

Verify locally:

```bash
curl -X POST http://localhost:4318/v1/traces -H "Content-Type: application/json" \
  -d '{"resourceSpans":[{"resource":{"attributes":[{"key":"service.name","value":{"stringValue":"agent-router"}}]},' \
  '"scopeSpans":[{"spans":[{"traceId":"00000000000000000000000000000001","spanId":"0000000000000001",' \
  '"name":"test-span","startTimeUnixNano":"1677643683000000000","endTimeUnixNano":"1677643683100000000"}]}]}]
```

You should see debug logs and Prometheus metrics at `http://localhost:8888/metrics`.

Gotcha: the OpenTelemetry collector's memory limiter defaults to a 200 MiB limit. On a 512 MiB Lambda, that leaves little headroom for the collector's own buffers. Set `limit_mib` to 128 and `spike_limit_mib` to 32 as above, and watch the collector's resident memory under load.

### Step 2 — core implementation

Create `src/index.ts`:

```typescript
import { trace } from '@opentelemetry/api';
import { createClient } from 'redis';

type AgentType = 'small' | 'llm' | 'vector';
type ModelResult = { output: string; durationMs: number };

const tracer = trace.getTracer('agent-router');
const REDIS = createClient({ url: process.env.REDIS_URL });
REDIS.connect().catch(err => console.error('Redis connect failed', err));

const CONCURRENCY_LIMIT = parseInt(process.env.AGENT_CONCURRENCY_LIMIT || '100', 10);
const REQUEST_WINDOW_MS = 60 * 1000;

// Sliding-window counter: one sorted set per window, scored by timestamp.
async function checkConcurrency(): Promise<boolean> {
  const now = Date.now();
  const key = 'agent:concurrency';

  const results = await REDIS.multi()
    .zRemRangeByScore(key, 0, now - REQUEST_WINDOW_MS)
    .zCard(key)
    .exec();

  const current = (results?.[1] as number) ?? 0;
  if (current >= CONCURRENCY_LIMIT) {
    return false;
  }

  await REDIS.zAdd(key, { score: now, value: `${now}:${Math.random()}` });
  await REDIS.expire(key, Math.ceil(REQUEST_WINDOW_MS / 1000) * 2);
  return true;
}

async function routeAgent(input: string, agentType: AgentType): Promise<ModelResult> {
  const span = tracer.startSpan(`route:${agentType}`);
  span.setAttribute('agent.type', agentType);

  try {
    switch (agentType) {
      case 'small':
        return { output: `Small model: ${input.slice(0, 20)}`, durationMs: 12 };
      case 'llm':
        await new Promise(res => setTimeout(res, 150));
        return { output: `LLM: ${input.toUpperCase()}`, durationMs: 160 };
      case 'vector':
        await new Promise(res => setTimeout(res, 80));
        return { output: `Vector: ${input.length} chars`, durationMs: 85 };
      default:
        throw new Error(`Unknown agent type ${agentType}`);
    }
  } finally {
    span.end();
  }
}

export async function handleRequest(event: any) {
  const span = tracer.startSpan('handleRequest');
  span.setAttribute('http.method', event.requestContext?.http?.method ?? 'POST');

  try {
    const body = JSON.parse(event.body || '{}');
    const { input, agentType = 'llm' } = body;

    if (!input) throw new Error('Missing input');

    const allowed = await checkConcurrency();
    if (!allowed) {
      span.recordException(new Error('Concurrency limit exceeded'));
      span.setStatus({ code: 2, message: 'Too many requests' });
      return {
        statusCode: 429,
        body: JSON.stringify({ error: 'Too many requests' }),
      };
    }

    const result = await routeAgent(input, agentType);
    return { statusCode: 200, body: JSON.stringify(result) };
  } catch (err: any) {
    span.recordException(err);
    span.setStatus({ code: 2, message: err.message });
    return { statusCode: 500, body: JSON.stringify({ error: err.message }) };
  } finally {
    span.end();
  }
}

export const handler = async (event: any) => {
  if (event.routeKey === '$default') {
    return await handleRequest(event);
  }
  return { statusCode: 404 };
};
```

Design notes:

- The guard uses a Redis sorted set rather than a plain `INCR` counter. A counter resets on a fixed clock boundary, which produces a thundering herd at the boundary; a sliding window smooths it out. `zRemRangeByScore` drops entries older than the window, and `zCard` gives the current count.
- The three simulated latencies (12 ms, 160 ms, 85 ms) are placeholders for whatever your backends actually do. Measure your own.
- Every route call is wrapped in a span, so the trace shows which backend dominates the tail.

Deploy to AWS Lambda:

```bash
npm install esbuild --save-dev
npx esbuild src/index.ts --bundle --platform=node --outfile=dist/index.js --minify

zip -r function.zip dist/index.js node_modules package.json

aws lambda create-function \
  --function-name agent-router \
  --runtime nodejs20.x \
  --handler index.handler \
  --zip-file fileb://function.zip \
  --role arn:aws:iam::123456789012:role/lambda-execution-role \
  --architectures arm64 \
  --timeout 10 \
  --memory-size 512 \
  --environment Variables='{"REDIS_URL":"redis://...","AGENT_CONCURRENCY_LIMIT":"100"}'
```

Attach an API Gateway HTTP API:

```bash
aws apigatewayv2 create-api \
  --name agent-router-api \
  --protocol-type HTTP \
  --target arn:aws:lambda:us-east-1:123456789012:function:agent-router

aws apigatewayv2 create-route \
  --api-id <api-id> \
  --route-key '$default' \
  --target integrations/<integration-id>

aws apigatewayv2 deploy-api --api-id <api-id> --stage-name prod
```

How to measure bounded concurrency: instrument the router to emit a counter of accepted and rejected requests, plus a gauge of in-flight work. Compare the gauge against `AGENT_CONCURRENCY_LIMIT` in Prometheus. If the gauge ever exceeds the limit, the guard is not atomic and you have a race.

## Rule 2 — Make every side effect idempotent

Retries are mandatory in a distributed system. Duplicate work is not. Without idempotency keys, a retry can charge you twice for the same LLM call, write two rows to a database, or send two emails.

Add a Redis-backed idempotency store keyed on a client-supplied header:

```typescript
const IDEMPOTENCY_TTL_SECONDS = 24 * 60 * 60;

async function ensureIdempotency(event: any): Promise<void> {
  const headerKey = event.headers?.['idempotency-key'];
  if (!headerKey) return;

  const key = `idemp:${headerKey}`;
  // SET NX is atomic: only one caller wins.
  const acquired = await REDIS.set(key, '1', {
    NX: true,
    EX: IDEMPOTENCY_TTL_SECONDS,
  });

  if (acquired === null) {
    throw new Error('Duplicate request');
  }
}
```

Wire it into `handleRequest` before any model call:

```typescript
export async function handleRequest(event: any) {
  const span = tracer.startSpan('handleRequest');
  try {
    await ensureIdempotency(event);
    // ... rest of the handler
  } catch (err: any) {
    span.recordException(err);
    span.setStatus({ code: 2, message: err.message });
    return { statusCode: 409, body: JSON.stringify({ error: err.message }) };
  } finally {
    span.end();
  }
}
```

The important detail is `SET ... NX`. A read-then-write sequence (`EXISTS` then `SET`) has a race window: two concurrent requests can both see "not present" and both proceed. `SET NX` is a single atomic operation, so exactly one caller wins.

How to measure idempotency: count `Duplicate request` rejections as a metric. A healthy rate is nonzero — it means retries are actually being deduplicated. A rate of zero with retries enabled usually means the idempotency key is not reaching the router (check header casing and API Gateway mapping).

## Rule 3 — Cap timeouts below your caller's timeout

A Lambda timeout of 10 seconds is generous for the router but dangerous for the caller. If your API Gateway integration timeout is 29 seconds and your Lambda can run for 10, a slow model call holds a connection for the full 10 seconds, and every retry stacks on top.

Set the Lambda timeout to the smallest value that accommodates your slowest legitimate call, plus a small margin. If your LLM call has a 1.2-second budget, a 2-second Lambda timeout is reasonable:

```bash
aws lambda update-function-configuration \
  --function-name agent-router \
  --timeout 2
```

Provisioned concurrency is a separate lever. It removes cold-start latency at the cost of paying for idle capacity. Enable it only after you have measured cold-start impact:

```bash
aws lambda put-provisioned-concurrency-config \
  --function-name agent-router \
  --qualifier '$LATEST' \
  --provisioned-concurrent-executions 50
```

How to measure timeout headroom: emit a histogram of handler duration and compare its p99 to the configured timeout. If p99 is within 20% of the timeout, you are one traffic spike away from mass timeouts. Also emit a counter for timeout-induced failures; a nonzero rate means the timeout is too tight or a dependency is too slow.

## Rule 4 — Degrade gracefully when shared state fails

Redis is a single point of failure for both the concurrency guard and the idempotency store. When it is unavailable, you have three options: fail closed (reject everything), fail open (accept everything), or degrade to a local cache.

A local fallback keeps the service up but weakens the guarantee:

```typescript
const localCache = new Map<string, number>();

async function ensureIdempotency(event: any): Promise<void> {
  const headerKey = event.headers?.['idempotency-key'];
  if (!headerKey) return;

  const key = `idemp:${headerKey}`;

  try {
    const acquired = await REDIS.set(key, '1', {
      NX: true,
      EX: IDEMPOTENCY_TTL_SECONDS,
    });
    if (acquired === null) throw new Error('Duplicate request');
  } catch (err) {
    // Redis unavailable: fall back to an instance-local cache.
    const expiresAt = localCache.get(key);
    if (expiresAt && expiresAt > Date.now()) {
      throw new Error('Duplicate request');
    }
    localCache.set(key, Date.now() + IDEMPOTENCY_TTL_SECONDS * 1000);
    setTimeout(() => localCache.delete(key), IDEMPOTENCY_TTL_SECONDS * 1000);
  }
}
```

The honest caveat: an instance-local cache is not shared. During a rolling deployment, two Lambda instances can each accept the same idempotency key, so duplicates can still slip through. Treat the fallback as a availability trade, not a correctness fix, and alert when it is active.

How to measure the fallback: emit a counter every time the `catch` branch runs. If that counter is nonzero for more than a few seconds, page someone — you are running without a shared idempotency guarantee.

## Rule 5 — Instrument before you optimize

You cannot tune what you cannot see. Add structured logging and a metrics pipeline before you change any routing policy.

```typescript
import pino from 'pino';

const logger = pino({
  level: process.env.LOG_LEVEL || 'info',
  transport: { target: 'pino-pretty' },
});

// Inside handleRequest:
logger.info({ agentType, inputLength: input.length }, 'routing request');
logger.error({ err }, 'request failed');
```

Prometheus exporter configuration in the collector:

```yaml
exporters:
  prometheus:
    endpoint: "0.0.0.0:8889"
    metric_expiration: 15m
    resource_to_telemetry_conversion:
      enabled: true
```

A useful dashboard has at least these panels:

- p50/p95/p99 latency, broken down by agent type.
- Accepted vs. rejected requests (the concurrency guard).
- Idempotency duplicate rate.
- Redis eviction rate and memory usage.
- Cost per 1,000 requests, computed from Lambda duration × memory and Redis instance hours.

For the cost panel, the arithmetic is straightforward. AWS Lambda bills on GB-seconds: `duration_seconds × memory_MB / 1024 × price_per_GB_second`. Add your Redis instance-hour cost divided by requests per hour. Label the panel "illustrative" until you have a week of real traffic, because the per-request cost depends heavily on your mix of agent types.

Load test with k6:

```javascript
import http from 'k6/http';
import { check } from 'k6';

export const options = {
  vus: 200,
  duration: '2m',
};

export default function () {
  const payload = JSON.stringify({ input: 'test', agentType: 'llm' });
  const headers = {
    'Content-Type': 'application/json',
    'Idempotency-Key': `${__VU}-${__ITER}`,
  };
  const res = http.post(
    'https://<api-id>.execute-api.us-east-1.amazonaws.com/prod/',
    payload,
    { headers }
  );
  check(res, { 'status is 200 or 429': (r) => r.status === 200 || r.status === 429 });
}
```

Note the check accepts 429. Under a bounded-concurrency design, rejecting excess load is correct behavior, not a failure. A load test that only accepts 200 will report false failures.

How to measure: run the test, then compare p99 latency at 50, 100, and 200 VUs. Latency should stay flat until the concurrency limit is hit, then 429s should appear while latency stays flat. If latency climbs instead of 429s appearing, the guard is not working.

## Rule 6 — Test the failure paths, not just the happy path

Unit tests that only cover successful routing are close to useless for a router. Test the guards:

```typescript
import { handleRequest } from './index';

describe('agent router', () => {
  it('rejects a duplicate idempotency key', async () => {
    const event = {
      routeKey: '$default',
      headers: { 'idempotency-key': 'dup-123' },
      body: JSON.stringify({ input: 'test', agentType: 'llm' }),
    };
    const res1 = await handleRequest(event);
    expect(res1.statusCode).toBe(200);

    const res2 = await handleRequest(event);
    expect(res2.statusCode).toBe(409);
    expect(JSON.parse(res2.body).error).toContain('Duplicate');
  });

  it('routes short input to the small model', async () => {
    const event = {
      routeKey: '$default',
      body: JSON.stringify({ input: 'test', agentType: 'small' }),
    };
    const res = await handleRequest(event);
    expect(res.statusCode).toBe(200);
    const body = JSON.parse(res.body);
    expect(body.output).toContain('Small model');
  });

  it('returns 429 when the concurrency limit is exceeded', async () => {
    process.env.AGENT_CONCURRENCY_LIMIT = '0';
    const event = {
      routeKey: '$default',
      body: JSON.stringify({ input: 'test', agentType: 'llm' }),
    };
    const res = await handleRequest(event);
    expect(res.statusCode).toBe(429);
    process.env.AGENT_CONCURRENCY_LIMIT = '100';
  });
});
```

Run them:

```bash
npm install jest ts-jest @types/jest --save-dev
npx jest --detectOpenHandles
```

The third test is the important one. It asserts the guard actually rejects, which is the behavior that protects your downstream dependencies.

## A worked example: choosing a routing policy

Suppose you have three backends with these *illustrative* characteristics:

| Backend | Latency (p50) | Cost per 1k calls | Good at |
|---|---|---|---|
| Small quantised model | 12 ms | $0.001 | Short inputs, classification |
| External LLM | 160 ms | $0.030 | Open-ended generation |
| Vector endpoint | 85 ms | $0.011 | Retrieval, similarity |

If 70% of your traffic is short classification and 30% is generation, a naive policy that sends everything to the LLM costs:

`0.7 × $0.030 + 0.3 × $0.030 = $0.030` per call, or `$30` per 1,000 calls.

A policy that routes short inputs to the small model:

`0.7 × $0.001 + 0.3 × $0.030 = $0.0007 + $0.009 = $0.0097` per call, or `$9.70` per 1,000 calls.

That is a 68% reduction, derived entirely from the stated assumptions. The catch: the small model must be accurate enough for the classification task. Measure accuracy on a held-out set before you route production traffic to it. A cheap wrong answer is more expensive than an expensive right one.

## Failure modes to watch for

- **Thundering herd at window boundaries.** A fixed-window counter resets all at once. Use a sliding window (sorted set) or a token bucket.
- **Non-atomic idempotency.** `EXISTS` followed by `SET` has a race. Use `SET NX`.
- **Timeout inversion.** If the router's timeout exceeds the caller's, the caller gives up first and retries, doubling load. Keep router timeouts strictly below caller timeouts.
- **Silent fallback.** A local cache that activates during a Redis outage can mask duplicate processing. Alert on it.
- **Load tests that reject 429.** A bounded system is supposed to reject excess load. Test for it.
- **Unmeasured routing policy.** Changing which model handles which input without measuring accuracy and cost is guesswork.

## FAQ

**How do you handle model failures without losing data?**
Write failed events to a dead-letter queue with the original payload and error context. A separate consumer retries with exponential backoff and a cap, then alerts. The cap matters: unbounded retries are how a partial outage becomes a total one.

**Can this pattern work on Kubernetes instead of Lambda?**
Yes. Replace the Redis concurrency guard with a sidecar rate limiter or a Redis-backed sliding window in Lua. The idempotency store and observability stack stay the same. The trade-off is different: pods have slower cold starts but better long-lived connection reuse.

**What about streaming responses?**
Streaming breaks the request/response idempotency model. Use a separate channel (WebSocket or a Redis stream) for progress, keep the idempotency key on the initiating request, and make the stream resumable by sequence number.

**Should I route across multiple models or fine-tune one?**
Routing wins when your traffic is heterogeneous and your latency budget is tight. Fine-tuning wins when your task is narrow, your dataset is large enough to be representative, and you can tolerate higher per-call latency. Measure both on your own traffic; published comparisons rarely match your mix.

## Take action in the next 30 minutes

Open your router's metrics dashboard and find the p99 latency for your slowest agent backend. Write that number down. Then check whether your concurrency guard is a fixed-window counter or a sliding window. If it is fixed-window, replace it with the sorted-set implementation above — that single change is the most common fix for retry-storm tail latency.
