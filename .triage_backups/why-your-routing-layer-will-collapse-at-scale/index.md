# Why your routing layer will collapse at scale

A routing layer for LLM calls looks trivial in a demo: pick a provider, call it, return the result. The trouble starts when the same code has to survive latency spikes, cost pressure, data-residency rules, and provider-specific error behaviour at the same time. The sections below describe a design that handles those constraints, the failure modes that show up in practice, and how to measure whether it is actually working.

## Why a simple `if` statement is not enough

Routing is often described as one decision: which model do I call? In production, that decision has to be *stateful across retries*. Three constraints drive most of the complexity:

- **Latency variance.** Cold model loads and provider-side queueing mean the same call can take very different amounts of time. A retry policy tuned for the median will fire too early and create retry storms.
- **Cost.** Sending every request to the top-tier model first is the simplest policy and usually the most expensive one. A tiered policy needs a way to fall back without doubling spend.
- **Residency.** If a prompt must be processed in a specific jurisdiction, the region is part of the routing decision, not an afterthought. A retry that silently switches region can break a compliance obligation even when the first attempt was fine.

The retry loop is where these constraints collide. Provider documentation typically describes a single call. It rarely specifies what to persist between attempts, so teams ship code that works in staging and then fails an audit because the second attempt went somewhere the first one was not allowed to go.

A related trap is caching model responses without tying the cache key to the model version. When a provider rolls out a new embedding or chat model, cached entries can keep being served until someone notices a quality change. The symptom is usually a support ticket describing output that changed "overnight" with no code change.

Provider-specific error behaviour matters too. Different providers emit rate-limit and overload errors with different status codes, retry hints, and backoff expectations. A retry policy hardcoded for one provider will misbehave when traffic shifts to another.

## The core design

The system has one source of truth: a config file that maps a user context to model tiers, regions, and retry policies. A simplified entry looks like this:

```json
{
  "eu-users": {
    "tier": "standard",
    "providers": [
      {
        "name": "provider-a",
        "model": "provider-a-large",
        "region": "eu-west-1",
        "cost_per_1k_token": 0.0000018,
        "max_retries": 2,
        "retry_delay_ms": 1000,
        "timeout_ms": 30000
      },
      {
        "name": "provider-b",
        "model": "provider-b-flagship",
        "region": "eu-central-1",
        "cost_per_1k_token": 0.000010,
        "max_retries": 1,
        "retry_delay_ms": 500,
        "timeout_ms": 45000
      }
    ]
  }
}
```

The router reads this file at startup, validates it against a JSON schema, and builds an in-memory table. On each request it does three things:

1. **Context matching** — read the user's region and tier from headers or verified JWT claims.
2. **Provider selection** — pick the first provider that has not exhausted its retry budget.
3. **Stateful retry** — persist the attempt count and the provider used, so a retry does not repeat a provider that already failed and does not cross a region boundary.

State can live in a Redis list keyed by request ID. Each entry is a tuple of `{provider, attempt, status}`. A Lua script atomically checks the list length against `max_retries`, appends the new attempt, and returns the current budget. Doing this atomically matters under concurrency: a read-then-write sequence in application code will let two workers both believe they have budget left.

Residency is enforced by two rules baked into the config:

- Every provider entry must declare a region.
- The router injects a region header into the downstream call and logs it.

The audit trail is a separate Redis stream that records every decision: timestamp, request ID, a hashed user identifier, provider, region, token count, and error if any. A scheduled job flushes the stream to object storage, where a query engine can reconstruct the full history for a given user or time window.

### Choosing the retry delay

A common mistake is to set `retry_delay_ms` to a round number that has no relationship to observed latency. If the delay is shorter than the provider's typical response time, retries fire while the original request is still in flight, which multiplies load instead of relieving it.

The practical rule is to derive the delay from measured latency for that provider, not from intuition. Concretely:

- Instrument the provider client to record the duration of every call, success or failure.
- Compute a high percentile (p95 or p99) over a rolling window, for example the last 1,000 calls.
- Set the base delay to at least that percentile, then add jitter so retries from different requests do not align.
- Cap the delay at a value your latency budget can tolerate.

To measure this, you need per-provider latency histograms. A Prometheus histogram with buckets spanning 50 ms to 30 s, labelled by provider and model, is enough. Compare the configured `retry_delay_ms` against the `histogram_quantile(0.95, ...)` value for the same provider. If the configured delay is below the p95, retries will overlap with in-flight requests.

## Implementation walkthrough

### 1. Config schema and validation

Validate the config at startup and fail fast. In Node, a JSON schema validator such as Ajv can enforce required fields and region formats:

```javascript
import Ajv from 'ajv';
const ajv = new Ajv({ allErrors: true });

const schema = {
  $schema: 'http://json-schema.org/draft-07/schema#',
  type: 'object',
  additionalProperties: false,
  patternProperties: {
    '^[a-zA-Z0-9_-]+$': {
      type: 'object',
      properties: {
        tier: { type: 'string' },
        providers: {
          type: 'array',
          items: {
            type: 'object',
            required: ['name', 'model', 'region', 'cost_per_1k_token', 'max_retries'],
            properties: {
              region: { pattern: '^(us|eu|ap)-[a-z0-9-]+-[0-9]$' }
            }
          }
        }
      },
      required: ['tier', 'providers']
    }
  }
};

const validate = ajv.compile(schema);
```

In Kubernetes this becomes a readiness check; in a serverless function it throws an exception that your alerting catches. The important property is that an invalid config never reaches the request path.

### 2. Provider client factory with circuit breakers

Wrap each provider's SDK in a factory that returns a client configured for the region and guarded by a circuit breaker:

```javascript
import { CircuitBreaker } from 'opossum';

function createClient(provider) {
  const breaker = new CircuitBreaker(async (prompt, options) => {
    const client = new ProviderClient({ region: provider.region });
    return client.chat(provider.model, prompt, options);
  }, {
    timeout: provider.timeout_ms,
    errorThresholdPercentage: 50,
    resetTimeout: 30000
  });
  return breaker;
}
```

The breaker opens after the configured error percentage is exceeded within the rolling window, which stops traffic from hammering a provider that is already failing. It also emits events you can export as metrics.

### 3. Request router with stateful retries

The core router needs to read the attempt history, choose a provider, call it, and record the outcome:

```javascript
import express from 'express';
import { createHash } from 'crypto';
import { Redis } from 'ioredis';

const redis = new Redis(process.env.REDIS_URL || 'redis://localhost:6379');
const app = express();

app.post('/chat', async (req, res) => {
  const userRegion = req.headers['x-user-region'] || 'eu-users';
  const config = await loadConfig();
  const ctx = config[userRegion];
  const requestId = createHash('sha256').update(req.body.prompt).digest('hex');

  const key = `retry:${requestId}`;
  const attempts = await redis.lrange(key, 0, -1);

  const available = ctx.providers.filter(p =>
    !attempts.some(a => JSON.parse(a).provider === p.name)
  );

  if (available.length === 0) {
    return res.status(429).json({ error: 'All providers exhausted' });
  }

  const provider = available[0];
  const client = providerClients[provider.name];

  try {
    const response = await client.fire(req.body.prompt, {
      maxTokens: 2048,
      temperature: 0.7
    });

    await redis.xadd('model_audit', '*', {
      requestId,
      userRegion,
      provider: provider.name,
      region: provider.region,
      tokens: response.usage.total_tokens,
      status: 'success'
    });

    res.json(response);
  } catch (err) {
    await redis.rpush(key, JSON.stringify({ provider: provider.name, attempt: attempts.length + 1 }));
    await redis.expire(key, provider.timeout_ms / 1000 + 60);

    if (attempts.length + 1 < provider.max_retries) {
      return res.status(503).json({ error: 'retry' });
    }

    await redis.xadd('model_audit', '*', {
      requestId,
      userRegion,
      provider: provider.name,
      region: provider.region,
      status: 'failed',
      error: err.message
    });
    res.status(500).json({ error: 'All providers failed' });
  }
});
```

Two things are worth noting. First, the retry budget is per provider, not per request, so a provider that failed once is not retried beyond its own limit. Second, the TTL on the retry list should be longer than the longest timeout in the config, otherwise a slow request can outlive its own state.

### 4. Auto-tuning the retry delay

A background worker can read recent audit entries per provider and recompute the p95 latency. If the configured delay is below the p95, it updates the in-memory config and publishes the change so other instances pick it up.

```javascript
const p95 = (arr) => {
  const sorted = [...arr].sort((a, b) => a - b);
  const pos = Math.floor(sorted.length * 0.95);
  return sorted[pos];
};

setInterval(async () => {
  const stats = await redis.xrange('model_audit', '-', '+', 'COUNT', 1000);
  const latencies = stats
    .filter(e => e[1][6] === 'success')
    .map(e => parseInt(e[1][8], 10));

  const newDelay = Math.min(Math.max(p95(latencies) * 2, 1000), 5000);
  await updateProviderConfig(provider.name, { retry_delay_ms: newDelay });
}, 30000);
```

The multiplier and the floor are policy choices, not derived values. Pick them so the resulting delay exceeds your measured p95 and stays inside your latency budget, then revisit them when the provider's behaviour changes.

### 5. Audit export

A scheduled job reads the audit stream and writes partitioned files to object storage, organised by date. A query engine can then answer questions like "show every request processed for user X in region Y between two dates" without touching production Redis. Partitioning by date keeps scan costs bounded.

## Failure modes to plan for

1. **Config reload stampede.** When a config change is published, every instance may try to fetch the new file at once. Serialise reloads with a lock, or publish the change and have each instance wait a randomised jitter before fetching.

2. **Region drift on provider upgrade.** A provider may move a model between regions. If the config is not updated, requests can route to the wrong region and fail a residency check. Enforce region immutability in the schema and require an explicit migration step for region changes.

3. **Audit stream backpressure.** At high request rates, the audit stream can grow faster than the flush job drains it. Cap the stream length (`XADD ... MAXLEN`) and increase the flush frequency. Monitor the stream length as a first-class metric.

4. **Retry list eviction.** If Redis is configured to evict keys under memory pressure, retry lists can disappear mid-request, producing "all providers exhausted" errors when budget remains. Mark retry keys as non-evictable and rely on TTLs for cleanup.

5. **Trusting client-supplied region headers.** A client can claim any region. Validate the region claim against the allowed set for the user's identity, and log a warning when the header and the claim disagree. Never let an unverified header determine where a prompt is processed.

## What to measure

Before adding routing complexity, instrument the following and compare before and after:

| Metric | What to instrument | How to compare |
|---|---|---|
| Router overhead | Time spent in routing logic per request | Histogram of router duration, compare to total request duration |
| Retry rate | Count of retries per provider | Counter labelled by provider; alert if it exceeds your baseline |
| Retry overlap | Retries issued while the original call is in flight | Compare configured delay to provider p95 latency |
| Circuit breaker trips | Breaker open events per provider | Counter; correlate with provider status pages |
| Audit lag | Difference between stream length and flush rate | Gauge of stream length; alert on sustained growth |
| Residency violations | Requests whose region header does not match the allowed set | Counter; should be zero in steady state |

None of these require a specific vendor. They require that the routing layer emits structured events with provider, region, and outcome attached.

## When this approach is the wrong choice

- **Very low latency budgets.** Routing adds a round trip to Redis and some in-process work. If your p99 budget is under 100 ms, measure the overhead before committing; it may be the wrong place to spend latency.
- **Single provider, single region.** A thin wrapper around the SDK is enough. The routing table, retry state, and audit stream are all overhead you do not need.
- **Hard residency boundaries.** If regulation requires that data physically never leaves a jurisdiction, dynamic region selection is a liability. Pre-partition the config per region and disable cross-region routing entirely.
- **Low request volume.** The fixed cost of a managed Redis instance and a scheduled flush job can dominate at small scale. A single function with in-memory retry state may be cheaper until volume grows.

## FAQ

**How do I handle model version upgrades without downtime?**
Add a new provider entry with the new model name and region, and keep the old entry until the new model's latency and error rate have stabilised over a full traffic cycle. Use the audit stream to compare token counts and error rates between the two before shifting traffic.

**How large does the retry store need to be?**
Size it from the number of concurrent in-flight requests, not the daily volume. Each retry list holds at most `max_retries + 1` entries and expires after the longest timeout plus a margin. Measure actual memory per key under load and multiply by peak concurrency.

**How do I test the residency path in staging?**
Run a second Redis instance and force the router to use a single region for every request. Run an export script and assert that the region header in the downstream call matches the expected value. Automate it in CI so a missing or wrong header fails the build.

**Can this run outside Kubernetes?**
Yes. The only platform-specific piece is the health check endpoint. Keep the retry store external and durable, because ephemeral local storage will lose retry state on restart.

## One thing to do in the next 30 minutes

Grep your codebase for hardcoded provider names and region strings. Count the distinct call sites. If the same provider or region appears in more than one place, extract those values into a single config file and add a startup validation step that rejects unknown regions. That one change turns an implicit policy into something you can review, test, and audit.
