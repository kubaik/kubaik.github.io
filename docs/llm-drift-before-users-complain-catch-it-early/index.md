# LLM drift before users complain: catch it early

## Why accuracy-only evaluation misses production drift

Most LLM evaluation content focuses on offline quality: golden prompt sets, similarity scores, LLM-as-a-judge. Those are useful for gating a release. They are close to useless for detecting the failure modes that actually degrade a live service, because the things users feel first are latency and cost, not answer quality.

A model can keep producing correct-looking outputs while p99 latency doubles and token consumption climbs. By the time a quality metric moves, the incident is already hours old. The evaluation pipeline that catches drift early is therefore not a benchmark harness — it is a lightweight, always-on set of checks over operational metrics, compared against a rolling baseline rather than a fixed threshold.

Three metric families carry most of the signal:

- **Latency** (p50, p95, p99, and upstream dependency latency)
- **Cost proxies** (tokens per request, retries per request, cache hit rate)
- **Quality** (a sampled judge score or task-specific assertion, used as a backstop)

Everything below is about instrumenting those correctly, and about the failure modes that make them move without any model change.

## Failure mode 1: fixed thresholds instead of rolling baselines

A hardcoded alert like "p99 > 1s is bad" works on the day it is written and decays from then on. Traffic mix changes, prompt templates grow, upstream services get slower, and the threshold either fires constantly or never fires at all.

The fix is a rolling baseline per `(model_version, region)` pair. A practical definition:

- Window: the last 24 hours of production traffic.
- Statistic: p99 latency and mean tokens per request.
- Outlier handling: exclude samples above the 99.9th percentile before computing the baseline.
- Refresh: hourly, via a background job.
- Alert condition: current p99 exceeds baseline by a fixed absolute margin, or tokens per request exceed baseline by a fixed relative margin.

The absolute margin for latency and the relative margin for tokens are policy choices, not universal constants. What matters is that they are computed against a baseline that tracks the system, and that they are set per region so a 200 ms shift in a high-traffic region is not treated the same as the same shift in a region with a saturated upstream.

Instrumenting this requires three things: a histogram for latency, a counter for tokens, and a gauge for cache hit rate. A minimal FastAPI worker that records all three looks like this:

```python
from fastapi import FastAPI, Request
from prometheus_client import Counter, Histogram, Gauge
import redis.asyncio as redis
import time

app = FastAPI()

REQUEST_LATENCY = Histogram(
    "llm_request_latency_seconds",
    "Latency of LLM requests in seconds",
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
TOKENS_PER_REQUEST = Counter(
    "llm_tokens_total",
    "Total tokens processed",
    ["model_version", "region"],
)
CACHE_HIT_RATE = Gauge(
    "llm_cache_hit_rate",
    "Cache hit rate for LLM responses",
    ["region"],
)

redis_client = redis.Redis(
    host="redis-cache",
    port=6379,
    decode_responses=True,
    socket_timeout=5,
    socket_connect_timeout=5,
)

@app.middleware("http")
async def track_metrics(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    latency = time.time() - start

    region = request.headers.get("x-region", "unknown")
    REQUEST_LATENCY.labels(region=region).observe(latency)

    model_version = response.headers.get("x-model-version", "unknown")
    tokens = int(response.headers.get("x-tokens-used", "0"))
    TOKENS_PER_REQUEST.labels(
        model_version=model_version, region=region
    ).inc(tokens)

    return response

@app.get("/health")
async def health():
    return {"status": "ok"}
```

Two details matter more than the code. First, the region label must be on the histogram and counter, not only on the gauge — otherwise the baseline cannot be computed per region. Second, the token count must come from the response path, not be estimated from the prompt, because the whole point is to catch cases where actual consumption diverges from expectation.

### How to measure whether your baseline is any good

You do not need a benchmark table to know if this works. You need a backtest. Take 30 days of historical metric data, compute the rolling baseline as of each hour, and ask: for the incidents you already know about, how many hours before the first user complaint would this rule have fired? If the answer is "never" or "after," the baseline or the margin is wrong. This is a query against data you already have, and it is the only honest way to tune the margins.

## Failure mode 2: cache degradation that quality metrics cannot see

A response cache is part of the serving path, and when it degrades, latency and cost rise while output quality stays identical. This is the single most common reason a green quality dashboard coexists with a support queue.

The mechanism is usually eviction, not failure. A cache cluster under memory pressure evicts entries according to its `maxmemory-policy`. With `allkeys-lru`, the policy evicts across all keys, so a traffic spike can flush a large fraction of a working set that was previously stable. Requests then fall through to the model, latency climbs, and token spend climbs with it. Nothing errors. Nothing looks broken. The only signal is the hit rate.

A configuration that behaves more predictably under skewed access patterns uses an LFU policy rather than LRU:

```
# redis.conf — cluster mode
cluster-enabled yes
cluster-config-file nodes.conf
cluster-node-timeout 5000
maxmemory 16gb
maxmemory-policy allkeys-lfu
maxmemory-samples 5
lfu-log-factor 10
lfu-decay-time 1
```

`allkeys-lfu` tracks access frequency, so entries that are read often survive a burst of one-off keys that would otherwise displace them under LRU. `lfu-decay-time 1` means the frequency counter halves roughly once per minute, which keeps the policy responsive to changing access patterns rather than locking in historical popularity.

The monitoring rule is simple and does not depend on any particular traffic level: alert on a relative drop in hit rate against its own rolling baseline, per region. A fixed "below 75%" line is a starting point, but the same reasoning as latency applies — the baseline should move with the workload.

### A worked example of the cost arithmetic

Suppose a service serves 1,000,000 requests per day. The cache normally absorbs 85% of them, so 150,000 reach the model. Each model call averages 1,200 tokens. At a blended price of $0.50 per million tokens, the daily model spend is:

- 150,000 × 1,200 = 180,000,000 tokens
- 180,000,000 / 1,000,000 = 180 million-token units
- 180 × $0.50 = $90 per day

Now suppose the hit rate falls to 60%. Model calls rise to 400,000:

- 400,000 × 1,200 = 480,000,000 tokens
- 480 × $0.50 = $240 per day

The difference is $150 per day, or roughly $4,500 per month, from a change that produced no errors and no quality regression. These figures are illustrative — substitute your own request volume and token price — but the shape of the result is the point: hit rate is a first-class cost metric, and it belongs on the same dashboard as latency.

### Multi-level caching

A single cache tier concentrates risk. A common structure that reduces it:

- **In-process cache** (for example, Python's `functools.lru_cache`) for repeated identical calls within one worker.
- **Regional cache** (a Redis cluster per region) for cross-process sharing.
- **Global store** for model weights and other large artifacts, ideally replicated to each region.

The regional tier is what prevents a failover in one region from turning into a cross-region latency problem for every region.

## Failure mode 3: silent artifact and dependency drift

Model weights and tokenizers are treated as static artifacts, but they are pulled from somewhere, and that somewhere can change. A tokenizer revision that applies a new normalization rule will change token counts for the same input text without changing a single model weight. The outputs still look fine. The bill and the latency do not.

The defense is to pin the revision explicitly and verify it at load time:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import hashlib

MODEL_ID = "my-org/my-model"
MODEL_REVISION = "v1.2.3"
TOKENIZER_REVISION = "v1.2.3"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    revision=MODEL_REVISION,
)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    revision=TOKENIZER_REVISION,
)

vocab = tokenizer.get_vocab()
# Sort keys so the hash is stable across processes.
canonical = "\n".join(f"{k}\t{v}" for k, v in sorted(vocab.items()))
tokenizer_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
expected_hash = "a1b2c3..."  # recorded when the revision was validated
if tokenizer_hash != expected_hash:
    raise RuntimeError("Tokenizer revision mismatch detected")
```

Two corrections to the naive version of this check are worth stating explicitly. Hashing `tokenizer.get_vocab().tobytes()` is not reliable, because dict iteration order is not guaranteed to be stable across processes or Python versions; sort the items and hash a canonical string instead. And pinning a revision is only meaningful if the check runs at startup, so a mismatch fails the deploy rather than surfacing as a metric anomaly hours later.

The same reasoning applies to any managed artifact store or model registry: the category of tool matters less than the property that a revision identifier resolves to immutable content, and that the deployed process verifies it.

There is a second, quieter version of this failure mode: cold-start latency when artifacts are pulled from a distant region. If the artifact store lives in one region and the service runs in another, cold starts absorb cross-region transfer time. The evaluation pipeline will not see this if it only samples the primary region. Replicating the artifact store per region, and measuring cold-start latency as its own metric, is the fix.

## Verifying that the pipeline actually fires

An alerting rule that has never fired is untested. The verification approach is to inject each drift mode deliberately and confirm the alert arrives.

**Tokenizer drift.** In a staging environment, run the service with a different tokenizer revision and confirm that the startup hash check fails the deploy. This is the cheapest test and it catches the most common silent regression.

**Cache eviction.** Force evictions and watch the hit-rate gauge:

```bash
redis-cli --cluster call 127.0.0.1:6379 "DEBUG evict 10000"
```

Note that `DEBUG evict` is a debugging command and is not available in all managed Redis offerings; where it is unavailable, simulate the same effect by lowering `maxmemory` on a test cluster until eviction begins.

**Upstream latency.** Inject delay on the path to the vector store or another upstream dependency and confirm that the per-region p99 alert fires. A socket-level wrapper is one way to do this in a test harness:

```python
import socket
import time

_original_socket = socket.socket

def slow_socket(*args, **kwargs):
    s = _original_socket(*args, **kwargs)
    s.settimeout(10)
    time.sleep(0.3)  # injected delay
    return s

socket.socket = slow_socket
```

This monkeypatch is a test-only tool. It delays socket creation, not individual sends, so it is a crude approximation of real network latency — useful for confirming that an alert fires, not for measuring realistic latency distributions.

**Load shape.** To exercise a region under realistic concurrency, a load generator with per-request region headers is enough:

```python
from locust import HttpUser, task, between

class DriftUser(HttpUser):
    wait_time = between(0.5, 2.5)

    @task
    def request(self):
        self.client.get(
            "/api/v1/chat",
            headers={"x-region": "ap-southeast-1"},
        )

# locust -f drift_test.py --headless -u 100 -r 10 --host=https://example.internal
```

The important property of these tests is not that they run in staging — it is that they run against a copy of the metric pipeline with the same baselines and the same alert rules as production. A chaos test that fires an alert nobody receives proves nothing.

## A decision checklist

When a drift alert fires, or when you are deciding what to instrument next, work through this list in order:

1. **Is the alert comparing against a rolling baseline, per region and per model version?** If not, fix that first; every other signal is unreliable without it.
2. **Did the hit rate move?** Check the per-region cache hit-rate gauge against its baseline before looking at anything else. Eviction is the most common cause and the easiest to confirm.
3. **Did tokens per request move?** If yes, and latency moved with it, suspect an artifact change. Check the tokenizer and model revision hashes recorded at deploy time.
4. **Did upstream dependency latency move?** Break p99 down by dependency. A vector store or a rate limiter is more often the cause than the model itself.
5. **Is the drift confined to one region?** Regional isolation points at infrastructure — connection pools, replication lag, cross-region artifact pulls — rather than at the model.
6. **Has the alert ever fired in a test?** If not, treat the rule as unverified and run the injection tests above.

## FAQ

**Why would p99 latency rise with no model change?**
The most common causes are cache eviction reducing hit rate, an upstream dependency slowing down, or connection pool saturation under concurrency. Break the latency down by dependency and by region before assuming the model is involved.

**How should a rolling baseline be computed?**
Take the last 24 hours of production samples for a given `(model_version, region)`, drop samples above the 99.9th percentile, and compute the statistic you alert on. Refresh hourly. Store it wherever your alerting layer can read it cheaply.

**What is the minimum metric set for catching drift early?**
Per-region p99 latency, tokens per request, cache hit rate, and upstream dependency latency. Quality metrics are a backstop, not the primary signal, because latency and cost move first.

**How do I stop a tokenizer update from silently changing token counts?**
Pin the tokenizer revision to the model revision, verify a canonical hash of the vocabulary at process startup, and fail the deploy on mismatch. Verify at deploy time, not at request time.

## Action for the next 30 minutes

Open your metrics backend and run two queries for the last 24 hours, grouped by region: p99 latency and cache hit rate. Compare the most recent two hours against the preceding 22. If either has moved by more than roughly 15% in any single region, you have found a drift your current alerts are not catching — and you now have the specific metric and region to build the rolling baseline around.
