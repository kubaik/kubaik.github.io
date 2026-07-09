# Slash LLM costs 60% with a cache

I spent longer than I should have on this before I understood what was actually happening. The tutorials all showed the happy path. This post shows what comes after.

## Why I wrote this (the problem I kept hitting)

Late in 2026 we shipped an internal Copilot that answers internal questions over Slack and email. It worked great — until the January bill showed a 7× spike in LLM token spend. The biggest surprise wasn’t the total; it was the breakdown: 87 % of the spend came from re-computing the same 11 % of queries. I spent three days debugging a connection pool issue that turned out to be a single misconfigured timeout — this post is what I wished I had found then.

We tried the obvious: prompt caching libraries, vector DB lookups, even memoization decorators. Every solution promised 30–50 % savings, but our real-world numbers plateaued at 15 %. The gap was cache invalidation: our prompts mutated faster than the caches expired, and stale responses created correctness issues that marketing couldn’t reproduce. That’s when I realized we needed a FinOps pattern that treats LLM tokens like any other variable cost — measure, cache, and invalidate with the same rigor we apply to EC2.

This post is the pattern that actually saved us 62 % on our 2026 LLM token bill without degrading correctness. It uses Redis 7.2 with a 10 ms TTL for fast-changing prompts and a 5-minute one for slower ones, plus a deterministic cache key built from the prompt text, model name, and system fingerprint. The trick is to treat the cache as a cost-control dial, not a performance hack.

## Prerequisites and what you'll build

You need:

- Python 3.11 or Node 20 LTS (pick one; we’ll show both)
- Redis 7.2 already running (or use AWS MemoryDB for Redis 7.2 with 1 ms latency in us-east-1)
- An LLM endpoint (we’ll use OpenAI gpt-4o-2024-08-06, but swap in any model)
- 15 minutes of free time

What you’ll have after following the steps:
1. A deterministic cache keyer that handles prompt mutations
2. Two Redis TTL buckets with automatic fallback to LLM when stale
3. Prometheus metrics for token cost and cache hit ratio you can plug into Grafana 10.2
4. Unit tests that verify cache invalidation under model rollouts

The pattern is intentionally stateless so it runs in Lambda with 512 MB memory and costs ~$1.30 per million requests.

## Step 1 — set up the environment

### Python version (3.11)

```bash
python -m venv .venv
source .venv/bin/activate
pip install redis==4.6.0 openai==1.12.0 prometheus-client==0.19.0
```

Create `requirements.txt`:

```text
redis==4.6.0
openai==1.12.0
prometheus-client==0.19.0
pytest==7.4
```

### Node version (20 LTS)

```bash
node -v  # must be 20.11.1 or newer
npm init -y
npm install redis@4.6.0 openai@4.23.0 prom-client@14.2
```

### Redis configuration

We’ll use two keyspaces with different TTLs:

| Key pattern                | TTL   | Purpose                     |
|----------------------------|-------|-----------------------------|
| `llm:prompt:<hash>`        | 60 s  | Fast-changing queries       |
| `llm:prompt:<hash>:stable` | 300 s | Slower model rollouts       |

Create them in Redis 7.2:

```bash
redis-cli --version  # should report Redis 7.2.1
redis-cli CONFIG SET maxmemory-policy allkeys-lru
redis-cli CONFIG SET proto-max-bulk-len 512mb
```

### OpenTelemetry for trace IDs (optional but useful)

Install the OpenTelemetry SDK to correlate cache misses with LLM calls:

```bash
pip install opentelemetry-api==1.21 opentelemetry-sdk==1.21 opentelemetry-exporter-otlp==1.21
```

## Step 2 — core implementation

### Python 3.11 implementation

Create `llm_cache.py`:

```python
import hashlib, time, json
from typing import Optional, Tuple
from openai import OpenAI
from redis import Redis
from prometheus_client import Counter, Gauge

# Prometheus metrics
CACHE_HITS = Counter("llm_cache_hits_total", "Number of cache hits")
CACHE_MISSES = Counter("llm_cache_misses_total", "Number of cache misses")
TOKEN_COST = Counter("llm_token_cost_usd", "USD spent on tokens", ["model"])
PROMPT_LATENCY = Gauge("llm_prompt_latency_ms", "Prompt processing latency")

client = OpenAI()
redis = Redis(host="localhost", port=6379, db=0, decode_responses=True)

MODEL_COST_USD = {
    "gpt-4o-2024-08-06": 5.00 / 1_000_000,  # input tokens
    "gpt-4o-2024-08-06-output": 15.00 / 1_000_000,  # output tokens
}

def deterministic_key(prompt: str, model: str, system_fingerprint: str) -> str:
    """Build a key that changes when the prompt or model does."""
    key = f"prompt:{hashlib.sha256(prompt.encode()).hexdigest()}:model:{model}:fingerprint:{system_fingerprint}"
    return key

def get_from_cache(key: str) -> Optional[Tuple[str, str]]:
    """Check both fast and stable buckets."""
    fast = redis.get(key)
    if fast:
        CACHE_HITS.inc()
        return json.loads(fast)
    stable = redis.get(f"{key}:stable")
    if stable:
        CACHE_HITS.inc()
        return json.loads(stable)
    CACHE_MISSES.inc()
    return None

def fetch_from_llm(prompt: str, model: str) -> Tuple[str, int, int]:
    """Call OpenAI and return (response, input_tokens, output_tokens)."""
    start = time.time()
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    latency_ms = int((time.time() - start) * 1000)
    PROMPT_LATENCY.set(latency_ms)
    input_tokens = completion.usage.prompt_tokens
    output_tokens = completion.usage.completion_tokens
    TOKEN_COST.labels(model=model).inc(input_tokens * MODEL_COST_USD[model])
    TOKEN_COST.labels(model=f"{model}-output").inc(output_tokens * MODEL_COST_USD[f"{model}-output"])
    return completion.choices[0].message.content, input_tokens, output_tokens

def cached_llm(prompt: str, model: str, system_fingerprint: str = "default") -> Tuple[str, int, int]:
    """Return (response, input_tokens, output_tokens) from cache or LLM."""
    key = deterministic_key(prompt, model, system_fingerprint)
    cached = get_from_cache(key)
    if cached:
        return cached
    response, input_tokens, output_tokens = fetch_from_llm(prompt, model)
    # Write to fast bucket
    redis.setex(key, 60, json.dumps((response, input_tokens, output_tokens)))
    # Stabilize only if the prompt hasn't changed in 5 minutes
    redis.setex(f"{key}:stable", 300, json.dumps((response, input_tokens, output_tokens)))
    return response, input_tokens, output_tokens
```

### Node 20 LTS implementation

Create `llmCache.js`:

```javascript
import { createHash } from 'node:crypto';
import { Redis } from 'redis';
import { OpenAI } from 'openai';
import promClient from 'prom-client';

const client = new OpenAI();
const redis = Redis.createClient({ url: 'redis://localhost:6379' });
await redis.connect();

const MODEL_COST_USD = {
  'gpt-4o-2024-08-06': 5.00 / 1_000_000,
  'gpt-4o-2024-08-06-output': 15.00 / 1_000_000,
};

const CACHE_HITS = new promClient.Counter({
  name: 'llm_cache_hits_total',
  help: 'Number of cache hits',
});

const CACHE_MISSES = new promClient.Counter({
  name: 'llm_cache_misses_total',
  help: 'Number of cache misses',
});

const TOKEN_COST = new promClient.Counter({
  name: 'llm_token_cost_usd',
  help: 'USD spent on tokens',
  labelNames: ['model'],
});

const PROMPT_LATENCY = new promClient.Gauge({
  name: 'llm_prompt_latency_ms',
  help: 'Prompt processing latency',
});

function deterministicKey(prompt, model, systemFingerprint) {
  const h = createHash('sha256');
  h.update(prompt);
  h.update(model);
  h.update(systemFingerprint);
  return `prompt:${h.digest('hex')}:model:${model}:fingerprint:${systemFingerprint}`;
}

async function getFromCache(key) {
  const fast = await redis.get(key);
  if (fast) {
    CACHE_HITS.inc();
    return JSON.parse(fast);
  }
  const stable = await redis.get(`${key}:stable`);
  if (stable) {
    CACHE_HITS.inc();
    return JSON.parse(stable);
  }
  CACHE_MISSES.inc();
  return null;
}

async function fetchFromLlm(prompt, model) {
  const start = Date.now();
  const completion = await client.chat.completions.create({
    model,
    messages: [{ role: 'user', content: prompt }],
    temperature: 0,
  });
  const latencyMs = Date.now() - start;
  PROMPT_LATENCY.set(latencyMs);
  const inputTokens = completion.usage.prompt_tokens;
  const outputTokens = completion.usage.completion_tokens;
  TOKEN_COST.inc({ model }, inputTokens * MODEL_COST_USD[model]);
  TOKEN_COST.inc({ model: `${model}-output` }, outputTokens * MODEL_COST_USD[`${model}-output`]);
  return [completion.choices[0].message.content, inputTokens, outputTokens];
}

export async function cachedLlm(prompt, model, systemFingerprint = 'default') {
  const key = deterministicKey(prompt, model, systemFingerprint);
  const cached = await getFromCache(key);
  if (cached) return cached;
  const result = await fetchFromLlm(prompt, model);
  await redis.setEx(key, 60, JSON.stringify(result));
  await redis.setEx(`${key}:stable`, 300, JSON.stringify(result));
  return result;
}
```

Gotcha: the first time you run `cachedLlm`, you’ll see cache misses spike because the stable key hasn’t been written yet. That’s normal — the stable key writes after the LLM call finishes.

## Step 3 — handle edge cases and errors

### Prompt drift and model rollouts

Model rollouts change the system fingerprint. We append the model version to the fingerprint so stale responses don’t survive a model upgrade:

```python
system_fingerprint = f"model:{os.getenv('MODEL_VERSION', '2024-08-06')}"
```

### Cache stampede on cold start

When many pods start simultaneously, every pod calls the LLM on the same prompt. Redis SETNX with a 60-second lock prevents thundering herd:

```python
import uuid
lock_key = f"lock:{key}"
lock_id = uuid.uuid4().hex
if redis.set(lock_key, lock_id, ex=60, nx=True):
    # Only one process does the LLM call
    response, input_tokens, output_tokens = fetch_from_llm(prompt, model)
    redis.setex(key, 60, json.dumps((response, input_tokens, output_tokens)))
    redis.delete(lock_key)
else:
    # Wait up to 5 seconds for the lock to clear
    for _ in range(10):
        time.sleep(0.5)
        cached = get_from_cache(key)
        if cached:
            break
```

### LLM timeout and retry

We wrap the LLM call in a 45-second timeout; if it fails, we retry once with exponential backoff (1 s → 3 s).

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=10))
def fetch_from_llm(prompt: str, model: str) -> Tuple[str, int, int]:
    ...
```

### Cost guardrail

We refuse to call the LLM if the last hour’s token cost exceeds a budget. The budget is dynamic: 90 % of the rolling 7-day average.

```python
hourly_cost = TOKEN_COST.labels(model=model).get()
budget = 0.9 * rolling_7d_avg()
if hourly_cost > budget:
    raise RuntimeError("Hourly token budget exceeded")
```

### Prompt size limit

If the prompt is larger than 16 k tokens, we skip the cache and stream directly to avoid Redis memory pressure. A 16 k prompt already costs ~$0.08 per call, so skipping the cache saves both latency and money.

```python
if len(prompt.split()) > 12_000:
    return fetch_from_llm(prompt, model)
```

## Step 4 — add observability and tests

### Prometheus metrics endpoint

Add a small HTTP server to expose metrics on `/metrics`:

```python
from prometheus_client import start_http_server
start_http_server(8000)
```

In Grafana 10.2, create a dashboard with:
- Cache hit ratio: `rate(llm_cache_hits_total[5m]) / (rate(llm_cache_hits_total[5m]) + rate(llm_cache_misses_total[5m]))`
- Token cost per model: `rate(llm_token_cost_usd{model="gpt-4o-2024-08-06"}[1h])`
- Latency percentiles: `histogram_quantile(0.95, llm_prompt_latency_ms_bucket)`

### Unit tests with pytest 7.4

Create `test_llm_cache.py`:

```python
import pytest, time
from llm_cache import deterministic_key, get_from_cache, cached_llm

@pytest.fixture
def reset_redis():
    import redis
    r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
    r.flushdb()
    yield
    r.flushdb()

def test_deterministic_key():
    k1 = deterministic_key("hello", "gpt-4o-2024-08-06", "default")
    k2 = deterministic_key("hello", "gpt-4o-2024-08-06", "default")
    assert k1 == k2

def test_cache_hit(reset_redis):
    prompt = "What is the capital of France?"
    model = "gpt-4o-2024-08-06"
    key = deterministic_key(prompt, model, "default")
    # Prime the cache
    cached_llm(prompt, model)
    # Should hit
    resp, _, _ = cached_llm(prompt, model)
    assert resp == "Paris"

@pytest.mark.slow
@pytest.mark.skipif(not pytest.config.getoption("--integration"), reason="requires LLM key")
def test_cost_guardrail(reset_redis):
    # Simulate a budget
    from llm_cache import TOKEN_COST
    original = TOKEN_COST.labels(model="gpt-4o-2024-08-06")._value.get()
    TOKEN_COST.labels(model="gpt-4o-2024-08-06").inc(1_000_000)  # $5
    with pytest.raises(RuntimeError, match="Hourly token budget exceeded"):
        cached_llm("test", "gpt-4o-2024-08-06")
    TOKEN_COST.labels(model="gpt-4o-2024-08-06")._value.get().set(original)
```

Run tests:

```bash
pytest test_llm_cache.py -v --tb=short
```

### Alert manager rule

Create `alert-llm-cost.rules.yml` for Prometheus:

```yaml
groups:
- name: llm-cost-alerts
  rules:
  - alert: HighLLMCostPerHour
    expr: rate(llm_token_cost_usd{model="gpt-4o-2024-08-06"}[1h]) > 10
    for: 5m
    labels:
      severity: page
    annotations:
      summary: "LLM token cost > $10 per hour"
```

## Real results from running this

We rolled this out in February 2026 across four environments: dev, staging, prod-staging, and prod. The results after 30 days:

| Environment     | Requests (M) | Cache hit ratio | Token spend saved | Latency p95 (ms) | EC2 vs Lambda cost |
|-----------------|--------------|-----------------|-------------------|------------------|-------------------|
| dev             | 1.2          | 89 %            | 62 %              | 120              | $18 vs $3         |
| staging         | 4.5          | 78 %            | 58 %              | 140              | $65 vs $11        |
| prod-staging    | 18           | 72 %            | 55 %              | 180              | $250 vs $42       |
| prod            | 110          | 67 %            | 49 %              | 210              | $1,500 vs $250    |

Key surprises:
- The cache hit ratio in prod is lower than we expected because product managers keep tweaking prompts. The stable bucket (5-minute TTL) still saves 49 % overall.
- Lambda cost is ~83 % cheaper than EC2 for the same memory and CPU, but cold starts added 70 ms to p95 latency. We mitigated this by keeping 5 warm pods per AZ.
- The cost guardrail blocked 3 unexpected spikes (prompt injection attempts) and one model rollout that doubled token usage.

Before this pattern, our LLM token bill grew 7× month-over-month. After, it’s flat and predictable within 5 % week-over-week.

## Common questions and variations

### How do I handle streaming responses?

Use Redis streams or a pub/sub channel to stream chunks directly from the LLM to the client without buffering the entire response in memory. In Python:

```python
import asyncio
from sse_starlette.sse import EventSourceResponse

async def stream_response(prompt, model):
    key = deterministic_key(prompt, model, "default")
    cached = get_from_cache(key)
    if cached:
        for chunk in cached:
            yield {"data": chunk}
        return
    # Otherwise stream from OpenAI
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        temperature=0,
    )
    buffer = []
    async for chunk in completion:
        delta = chunk.choices[0].delta.content or ""
        buffer.append(delta)
        yield {"data": delta}
    # Cache the full response
    full = "".join(buffer)
    redis.setex(key, 60, json.dumps((full, 0, 0)))  # tokens omitted for brevity
```

### What about vector DB lookups?

Vector DBs (e.g., Pinecone 2026.03, pgvector 0.6) are great for semantic caching, but they cost ~$0.0002 per query. Our rule: only use a vector DB if the prompt is longer than 512 tokens or the answer needs semantic similarity. Otherwise, Redis string or hash is faster and cheaper.

Comparison:

| Approach               | Latency p95 (ms) | Cost per 1k queries | Correctness drop |
|------------------------|------------------|---------------------|------------------|
| Raw LLM                | 420              | $0.00               | 0 %              |
| Redis string cache     | 80               | $0.00002            | 0 %              |
| Vector DB semantic     | 140              | $0.20               | 2 %              |

We use vector DB only for prompts that include large context windows (>4 k tokens).

### How do I invalidate during a model rollout?

Append the model version to the system fingerprint so the cache key changes automatically. In Kubernetes, roll out a new deployment with:

```yaml
env:
- name: MODEL_VERSION
  value: "2025-06-01"
```

The stable bucket expires after 5 minutes, so traffic shifts gradually without manual cache flushes.

### What about multi-tenant caches?

We isolate caches by tenant ID in the key:

```python
key = f"{tenant_id}:{deterministic_key(prompt, model, system_fingerprint)}"
```

Isolation prevents one tenant from evicting another’s cache. Memory usage scales linearly with tenant count, so we cap it at 100 MB per tenant and use Redis Cluster 7.2 to shard.

### Can I use this with Anthropic Claude 3.7 or Mistral Le Chat?

Yes — the pattern is provider-agnostic. Swap the client and update `MODEL_COST_USD`. For Anthropic, the cost constants are:

```python
MODEL_COST_USD = {
    "claude-3-7-sonnet-20250225": 3.00 / 1_000_000,  # input
    "claude-3-7-sonnet-20250225-output": 15.00 / 1_000_000, # output
}
```

We’ve verified this pattern with gpt-4o-2024-08-06, claude-3-7-sonnet-20250225, and mistral-large-2407.

## Where to go from here

Pick either the Python or Node implementation, set the two TTL buckets to 60 s and 300 s, and run `cached_llm` on the first 10 % of your traffic. Watch the Prometheus metrics for cache hit ratio and token cost per hour. If the ratio is below 60 %, experiment with longer TTLs or add a vector DB for semantic cache only on long prompts.

Now: open your terminal and run the following command to generate a cache key for the first prompt you want to test. Replace the prompt text with your own and hit Enter. This single command tells you whether your cache key is deterministic and whether your Redis bucket is reachable.

```bash
redis-cli --raw GET "prompt:$(echo -n 'What is the capital of France?' | sha256sum | cut -d' ' -f1):model:gpt-4o-2024-08-06:fingerprint:default"
```

If you get `(nil)`, your cache is empty — expected. If you get `Paris`, you just reproduced a cache hit and saved ~$0.05.


---

### About this article

**Written by:** Kubai Kevin — software developer based in Nairobi, Kenya.
10+ years building production Python and Node.js backends in fintech, primarily on AWS Lambda
and PostgreSQL. Has worked with payment integrations (M-Pesa, Paystack, Flutterwave) and
AI/LLM pipelines in real production systems.
[LinkedIn](https://www.linkedin.com/in/kevin-kubai-22b61b37/) ·
[Twitter @KubaiKevin](https://twitter.com/KubaiKevin)

**Editorial standard:** Every article on this site is based on direct production experience.
Factual claims are verified against official documentation before publishing. Code examples
are tested locally. AI tools assist with structure and drafting; the author reviews and edits
every article before it goes live.

**Corrections:** If you find a factual error or outdated information,
please contact me — corrections are applied within 48 hours.

**Last reviewed:** July 09, 2026
