# AI feature drift: why your model died at midnight

## Why a model can fail only at certain hours

A recurring production pattern in AI feature pipelines: a model behaves well during the day, then degrades sharply during a specific window (often the small hours) and recovers on its own. The dashboard shows a metric collapse, and the logs point at the feature store with an error such as `FeatureStoreKeyNotFound: key not found in RedisCluster`.

The error message is usually a red herring. The feature store may be perfectly healthy. What changes at night is the *input distribution* and the *runtime environment*, not the store itself. Three mechanisms commonly produce this pattern, and they can stack:

1. **Tokenizer or vocabulary drift** after an upstream batch job lands new categorical values.
2. **Allocator fragmentation** in a memory-constrained container, triggered by a slightly larger working set.
3. **Cold-start networking** when connection setup coincides with a traffic peak.

The rest of this article explains each mechanism, gives a reproducible fix, and shows how to verify the fix with instrumentation rather than anecdotes.

## Mechanism 1: tokenizer and vocabulary drift

### What actually happens

A model is trained against a fixed vocabulary. For text encoders this is a tokenizer vocabulary; for tabular-plus-text hybrids it is often a set of categorical codes. When an upstream ETL job introduces new categories, one of two things occurs:

- The tokenizer maps unknown strings to its unknown token (`UNK` / `[UNK]`), so many distinct inputs collapse into one embedding. Signal is lost.
- The code path that builds an embedding table by index encounters an index beyond the trained range, which either raises or forces a resize.

Both are *silent* in the sense that no exception is raised at the point of the distribution change. The visible symptom appears downstream: a metric drop, a latency increase, or a memory error. A feature-store error can appear later if the process, under memory pressure, drops or fails to populate cached entries and then surfaces a cache-miss exception.

### The fix: version the vocabulary with the model, and validate before tokenizing

Two practices remove most of this class of bug:

- **Version the tokenizer/vocabulary as a model artifact.** The vocabulary file should be stored and loaded alongside the weights, referenced by the same version identifier. Never let the inference path read a vocabulary that was not shipped with the model.
- **Validate the payload against a schema before tokenization.** Reject or quarantine unknown categories explicitly, and emit a metric, rather than letting them flow into the tokenizer.

A minimal schema using Pydantic:

```python
# feature_schema.py
from pydantic import BaseModel, constr
from typing import Annotated

class LoanFeatures(BaseModel):
    loan_purpose: Annotated[str, constr(pattern=r'^[A-Za-z0-9 -]{1,32}$')]
    income: float
    credit_score: int

schema = LoanFeatures.model_json_schema()
```

Note what this schema does and does not do. It enforces a *format* (length, allowed characters) but not a *closed set* of categories. To catch drift you need the closed set. A more honest version keeps an explicit allow-list loaded from the same artifact as the model:

```python
# feature_schema.py
from pydantic import BaseModel, field_validator
import json

with open("vocab-v1.json") as f:
    VOCAB = set(json.load(f)["loan_purpose"])

class LoanFeatures(BaseModel):
    loan_purpose: str
    income: float
    credit_score: int

    @field_validator("loan_purpose")
    @classmethod
    def known_category(cls, v: str) -> str:
        if v not in VOCAB:
            raise ValueError(f"unknown category: {v!r}")
        return v
```

The inference handler validates, then tokenizes, then asserts the token count fits the model's context window:

```python
# inference_lambda.py
import json
import boto3
from feature_schema import LoanFeatures

s3 = boto3.client("s3")
vocab = json.loads(
    s3.get_object(Bucket="model-registry", Key="vocab-v1.json")["Body"].read()
)

model = load_model("model-v1.bert")
tokenizer = Tokenizer.from_file("tokenizer-v1.json")

MAX_TOKENS = 512  # must match the value used at training time

def handler(event, ctx):
    features = LoanFeatures(**event["features"])
    tokens = tokenizer.encode(features.loan_purpose)
    if len(tokens) > MAX_TOKENS:
        raise ValueError(f"token length {len(tokens)} exceeds {MAX_TOKENS}")
    return model.predict(tokens)
```

Two details matter for correctness:

- `MAX_TOKENS` must be a named constant that matches training. Hard-coding `512` in two places invites divergence; read it from the model artifact.
- Raising a typed error (for example a custom `FeatureValidationError`) lets you alarm on it directly. A bare `ValueError` is harder to distinguish from unrelated failures.

### How to measure drift

Instrumentation is the only reliable way to confirm this mechanism:

- **Emit a counter per rejected payload**, tagged by reason. If the count rises at the same hour each day, drift is the cause.
- **Track unknown-token rate** on the tokenizer output. For Hugging Face tokenizers, `tokenizer.encode(text).tokens` will contain the unknown token; count occurrences per request and publish a histogram.
- **Compare category cardinality** between the training snapshot and the live stream. A simple `len(set(live_categories) - set(train_categories))` computed hourly is enough to catch new values.

A concrete check you can run against a stored training vocabulary and a day of production values:

```python
train = set(json.load(open("vocab-v1.json"))["loan_purpose"])
live = set(json.load(open("today_categories.json")))
print("new categories:", sorted(live - train))
print("retired categories:", sorted(train - live))
```

If `new categories` is non-empty, the model is receiving inputs it was never trained to represent. Whether that is acceptable depends on the fallback behavior: an explicit unknown bucket that was present during training is a legitimate design; an accidental collapse into `UNK` is not.

## Mechanism 2: allocator fragmentation in a constrained container

### What actually happens

A process can sit comfortably below its memory limit in steady state and still fail when the working set grows slightly. The reason is fragmentation: the allocator holds free memory, but not in contiguous blocks large enough for the next request. The failure surfaces as a `MemoryError` or as the runtime killing the container, and any in-process caches disappear with it. That cache loss is what can later present as a feature-store miss.

The trigger is often small: a batch of inputs that produce a slightly larger intermediate tensor, or a tokenizer that allocates a temporary buffer per request. The steady-state headroom is not the same as the headroom available for a single large allocation.

### The fix: measure headroom, then reduce fragmentation

Two levers, in order of cost:

- **Increase the memory limit** so that peak usage has real headroom. If the documented limit is 512 MB and steady-state usage is above roughly 450 MB, the margin is thin. Moving to the next tier is a one-line change and removes the failure entirely. This is the cheapest correct fix and should be tried first.
- **Change the allocator.** Linking a different allocator (for example jemalloc) can reduce fragmentation for allocation-heavy Python workloads. This is a real technique but it adds build complexity and a new failure surface, so it should follow, not precede, the memory bump.

A container definition that preloads jemalloc:

```dockerfile
FROM public.ecr.aws/lambda/python:3.11
RUN yum install -y jemalloc
ENV LD_PRELOAD=/usr/lib64/libjemalloc.so.1 \
    MALLOC_CONF=background_thread:true
COPY app.py ${LAMBDA_TASK_ROOT}
CMD ["app.handler"]
```

Pin the base image tag and the package version in your own build; the snippet above is illustrative and the exact library path differs across distributions. Verify the preload actually took effect rather than assuming it: a process that fails to find the shared object will typically still start, but without the allocator you intended.

### How to measure fragmentation

You do not need a profiler to know whether you are close to the limit:

- **Publish peak memory usage**, not average. CloudWatch reports `MaxMemoryUsed` per invocation; alarm on the p99 of that value against the container limit.
- **Compute headroom explicitly.** If the limit is 640 MB and p99 peak usage is 590 MB, headroom is 50 MB, or roughly 8%. That is a thin margin for a workload with variable input size.
- **Watch for restarts.** A rising cold-start count with no deployment is a strong signal that containers are being recycled under memory pressure.

If you do want allocator-level detail, jemalloc ships with `jeprof`, which can produce a heap profile from a running process. Treat the output as a diagnostic, not a target: the actionable number is peak usage versus limit.

### A worked example

Assume a container limit of 512 MB and a measured p99 `MaxMemoryUsed` of 480 MB.

- Headroom = 512 − 480 = 32 MB, which is 32 / 512 = 6.25% of the limit.
- A single request that needs a contiguous 40 MB allocation cannot be satisfied even though 32 MB is free, because the free space is fragmented.
- Raising the limit to 640 MB gives 640 − 480 = 160 MB of headroom, i.e. 25% of the limit.

The arithmetic is trivial; the point is that the decision should be made from a measured peak, not from an average. An average of 300 MB tells you nothing about whether a 512 MB limit is safe.

## Mechanism 3: cold-start networking during a traffic peak

### What actually happens

When a container starts, it must establish network connections before it can serve requests. If connection setup involves DNS resolution or a new network interface, the first requests can time out. If the traffic peak coincides with a high rate of container starts, many requests fail at once, and the resulting retries can amplify load on the downstream store.

A common shape: a connection pool is created per invocation, DNS answers are cached for a short TTL, and the client's connect timeout is longer than the retry budget. The first request on a new container fails, the client retries, and the retry storm is what the error logs actually show.

### The fix: create connections once, per container

Move client construction to module scope so it runs during initialization, and configure timeouts that fail fast:

```python
# redis_client.py
import os
import redis

pool = redis.ConnectionPool(
    host=os.environ["REDIS_HOST"],
    port=6379,
    db=0,
    max_connections=50,
    socket_connect_timeout=2,
    socket_timeout=2,
    socket_keepalive=True,
    socket_keepalive_options={
        redis.SocketKeepAliveOptions.TCP_KEEPIDLE: 5
    },
    decode_responses=True,
)
client = redis.Redis(connection_pool=pool)
```

Because this module is imported at container start, the pool is built once and reused. Set `socket_timeout` as well as `socket_connect_timeout`; without it, a slow read can hang beyond the function timeout.

### How to measure cold-start cost

- **Log the initialization duration** from the start of module import to the end, and publish it as a metric. Compare its distribution before and after a change.
- **Count cold starts** by logging a line in module scope. Every container logs it exactly once, so the count over a window is the number of containers started.
- **Correlate** the cold-start count with the error rate. If errors cluster in the minutes following a burst of cold starts, connection setup is implicated.

If your platform supports snapshot-based cold-start acceleration, measure the improvement rather than assuming it: compare p99 initialization time before and after enabling it, on the same workload.

## A decision checklist for nightly failures

Work through these in order; each step is cheap and rules out a large class of causes.

1. **Confirm the time window is real.** Plot the failing metric by hour over at least 14 days. A genuine window repeats; a one-off spike does not.
2. **Check whether the input distribution changed.** Diff live category values against the training vocabulary. Non-empty difference means drift.
3. **Check peak memory against the limit.** If p99 peak usage exceeds roughly 90% of the limit, raise the limit before investigating anything else.
4. **Check cold-start frequency.** If container starts spike at the same time as errors, inspect connection setup.
5. **Check retry behavior.** If retries outnumber original requests, the retry policy is amplifying the failure. Add jitter and cap attempts.
6. **Only then** look at the downstream store. A store error that appears only under load is usually a symptom of the process above it, not a store fault.

## Verifying a fix without inventing numbers

A fix is verified when the metric that failed returns to its baseline and stays there under the same conditions. Concretely:

- **Replay the failing window.** Run a load test that reproduces the traffic shape of the failing hours, not a flat load. The shape matters more than the volume.
- **Compare like with like.** Compare the same metric over the same hours before and after the change. Comparing a daytime average to a nighttime average proves nothing.
- **Require more than one run.** A single passing run can be luck. Repeat the test enough times that a rare failure would show up.
- **Set the pass condition in advance.** Decide the acceptable threshold before running, so the result cannot be reinterpreted afterward.

What to instrument, at minimum:

- Requests per second, split by success and failure.
- The domain metric that degraded (for example AUC or another accuracy measure), computed on the same schedule as before.
- Peak memory per invocation (p99), against the container limit.
- Cold-start count and initialization duration.
- Downstream error rate and retry count.

## Preventing recurrence

### Version everything that affects the input representation

The model, tokenizer, vocabulary, and any preprocessing configuration should share one version identifier and be loaded together. A model that loads a vocabulary it was not trained with is a latent bug, not a configuration choice.

### Validate at the boundary

Reject unexpected inputs explicitly and count the rejections. An explicit rejection with a metric is far easier to debug than a silent substitution that degrades accuracy.

### Test with realistic inputs

Synthetic test data tends to be well-formed. Include null values, out-of-vocabulary categories, and long inputs in the test set, because those are the inputs that break pipelines.

### Canary changes

Route a small share of traffic to a new model version and compare its metrics with the incumbent's over the same period. A canary that is only checked for errors will miss an accuracy regression, so compare the domain metric too.

### Keep dependencies pinned

Pin exact versions and use a lockfile so that a rebuild produces the same environment:

```bash
pip-compile --generate-hashes requirements.in > requirements.txt
```

An unpinned rebuild can silently change tokenizer behavior, which is exactly the class of bug this article is about.

## Common errors and what they usually mean

| Error | Likely cause | First check |
|---|---|---|
| Unknown-token warnings in tokenizer output | New category not present at training time | Diff live categories against the training vocabulary |
| `MemoryError` under load | Peak usage close to the container limit | p99 peak memory versus limit |
| Connection timeout on first request after start | Connection created per invocation, not per container | Initialization duration and cold-start count |
| Cache-miss error only under load | Process recycled under memory pressure | Cold-start count correlated with error rate |
| Accuracy drop with no errors | Silent input substitution | Unknown-token rate and category cardinality |

## FAQ

**Why would a model fail only between certain hours?**

Because the input distribution or the runtime environment changes on a schedule. Batch jobs often land at fixed times, and traffic peaks often follow daily patterns. The model itself does not know what time it is.

**Is a feature-store error ever the real cause?**

Yes, but it is usually a symptom of something above it. If the store error appears only when the process is under memory pressure or restarting, look at the process first.

**Do I need a different memory allocator?**

Not necessarily. Raising the container limit is simpler and addresses the same problem when the issue is thin headroom. Changing the allocator is worth considering only after the limit has been raised and peak usage is still close to it.

**How do I know drift is the cause rather than a bug?**

Drift produces a gradual or scheduled change with no code deployment. A bug appears with a deployment. Correlating the metric change against your deployment log is the fastest way to tell them apart.

**Should I retrain the model?**

If the new categories are legitimate and will persist, yes. Validation and alerting buy time; they do not make the model accurate on inputs it never saw.

## Action for the next 30 minutes

Open your model's memory metric for the last 14 days and plot the p99 of peak usage against the container limit. If that ratio exceeds 90%, raise the limit to the next tier and redeploy. That single change removes the most common cause of nightly failures before you investigate anything else.
