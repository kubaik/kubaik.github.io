# Observability for LLM Features: Metrics That Drive Decisions

Most tutorials for wrapping an LLM endpoint stop at the happy path: send a prompt, get a completion, return it. The harder problem starts afterwards, when a service has accumulated several LLM-backed features and nobody can say which ones are worth their compute cost. Token-level tracing produces megabytes of intermediate noise for every byte of user-facing signal.

This article describes a lightweight observability layer for LLM features: one structured event per user interaction, a small set of Prometheus metrics, and alert rules tied to a stated SLO. It runs comfortably on a small VM, requires no SaaS vendor, and works with any HTTP LLM endpoint.

## The failure mode: observability that becomes the bottleneck

A common pattern in teams that add LLM features incrementally: each feature logs its own prompt, its own response, and some metadata "for future observability." A summarisation call might log a short system prompt, a long user message, an assistant reply, and per-request metadata. None of that is large on its own. The problem is multiplicative.

Work through the arithmetic with illustrative assumptions:

- 20,000 requests per day
- 150 tokens logged per request (prompt + response + metadata)
- 4 bytes per token as a rough text approximation

That is 20,000 × 150 = 3,000,000 tokens per day, or 3,000,000 × 4 bytes ≈ 12 MB per day of raw text, roughly 84 MB per week. If the logging format is verbose JSON with envelope fields, the on-disk figure is several times larger. At that point the logging pipeline — not the LLM — is often the thing that saturates disk, slows queries, and makes the dashboard unreadable.

The deeper problem is that token-level data answers "what did the model produce?" It does not answer "did the user get value?" A team can hold gigabytes of traces and still be unable to decide whether to keep a feature.

The fix is to log one row per user interaction containing only fields that map to a decision: which feature, which model, how long, how much, did it succeed, and what did the user think.

## Prerequisites and what you will build

Required:

- Python 3.11
- FastAPI
- Redis (for feature flags and a simple circuit breaker)
- PostgreSQL with the `pgvector` extension (used later for feedback clustering)
- A running LLM HTTP endpoint exposing an OpenAI-compatible chat completions route
- Node 20 LTS (only if you want the optional dashboard)

You will build:

1. A FastAPI service that wraps any LLM endpoint and emits one structured event per call.
2. A small Prometheus exporter exposing four metrics:
   - `llm_feature_duration_seconds` (histogram)
   - `llm_feature_success_total` (counter)
   - `llm_feature_user_feedback` (up-down counter)
   - `llm_feature_cost_cents` (counter)
3. A Redis-backed feature flag that enables a feature per user segment.
4. A backfill script that imports historical rows so before/after comparisons remain possible.

The structured event contains:

- `user_id`
- `feature_name` (for example `summarise`, `extract_entities`)
- `model_name`
- `prompt_length`
- `response_length`
- `duration_ms`
- `cost_cents`
- `success` (boolean)
- `user_feedback` (1–5, optional)
- `error_message` (optional)

That single row is enough to decide whether a feature earns its keep.

## Step 1 — set up the environment

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
```

Install the core stack:

```bash
pip install fastapi uvicorn redis httpx opentelemetry-api opentelemetry-sdk opentelemetry-exporter-prometheus prometheus-client psycopg2-binary python-dotenv
```

Create `.env`:

```ini
LLM_ENDPOINT=http://llama3:8000/v1/chat/completions
LLM_API_KEY=
REDIS_URL=redis://redis:6379/0
DATABASE_URL=postgresql://postgres:postgres@postgres:5432/llm_observability
PROMETHEUS_PORT=8001
```

Bring up the backing services with `docker-compose.yml`:

```yaml
version: '3.9'
services:
  redis:
    image: redis:7.2-alpine
    ports:
      - "6379:6379"
  postgres:
    image: pgvector/pgvector:pg15
    environment:
      POSTGRES_PASSWORD: postgres
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
  app:
    build: .
    ports:
      - "8000:8000"
      - "8001:8001"
    depends_on:
      - redis
      - postgres

volumes:
  pgdata:
```

A common failure mode when the LLM endpoint sits behind a proxy that requires an API key: the key is loaded from a file and retains a trailing newline, so every request returns 401. Strip whitespace explicitly when reading secrets from files.

## Step 2 — core implementation

Create `main.py`:

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import redis.asyncio as redis
import httpx
import time
import os
from pydantic import BaseModel
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.prometheus import PrometheusMetricExporter
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

app = FastAPI()

# --- Observability setup ---
resource = Resource.create({"service.name": "llm-feature-proxy"})
exporter = PrometheusMetricExporter(port=int(os.getenv("PROMETHEUS_PORT", "8001")))
reader = PeriodicExportingMetricReader(exporter)
meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
trace.set_tracer_provider(TracerProvider(resource=resource))

# --- Metrics definitions ---
meter = meter_provider.get_meter("llm.feature.metrics", version="1.0")
feature_duration = meter.create_histogram(
    "llm_feature_duration_seconds",
    unit="s",
    description="Duration of a single LLM feature call",
)
feature_success = meter.create_counter(
    "llm_feature_success_total",
    unit="1",
    description="Count of successful feature calls",
)
feature_feedback = meter.create_updown_counter(
    "llm_feature_user_feedback",
    unit="1",
    description="User feedback score",
)
feature_cost = meter.create_counter(
    "llm_feature_cost_cents",
    unit="cent",
    description="Cost in cents for this feature call",
)

# --- Redis feature flag ---
redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))


class FeatureRequest(BaseModel):
    user_id: str
    feature_name: str
    prompt: str
    model: str = "llama3-8b-v1"
    max_tokens: int = 256


@app.post("/feature/{feature_name}")
async def call_feature(feature_name: str, req: FeatureRequest):
    start = time.time()
    tracer = trace.get_tracer(__name__)

    flag_key = f"feature:{feature_name}:{req.user_id}"
    active = await redis_client.get(flag_key)
    if not active:
        raise HTTPException(status_code=403, detail="Feature disabled for user")

    async with tracer.start_as_current_span(f"feature:{feature_name}") as span:
        span.set_attribute("user.id", req.user_id)
        span.set_attribute("feature.name", feature_name)

        headers = {"Authorization": f"Bearer {os.getenv('LLM_API_KEY', '').strip()}"}
        payload = {
            "model": req.model,
            "messages": [{"role": "user", "content": req.prompt}],
            "max_tokens": req.max_tokens,
            "temperature": 0.3,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                resp = await client.post(
                    os.getenv("LLM_ENDPOINT"),
                    json=payload,
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()
                duration_ms = (time.time() - start) * 1000
                total_tokens = data.get("usage", {}).get("total_tokens", 0)
                cost_cents = total_tokens * 0.0002  # see note below

                feature_duration.record(
                    duration_ms / 1000,
                    {"feature": feature_name, "model": req.model},
                )
                feature_success.add(1, {"feature": feature_name})
                feature_cost.add(cost_cents, {"feature": feature_name})

                return JSONResponse(
                    {
                        "response": data["choices"][0]["message"]["content"],
                        "duration_ms": duration_ms,
                        "cost_cents": cost_cents,
                    }
                )
            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR))
                raise


@app.post("/feature/{feature_name}/feedback")
async def submit_feedback(feature_name: str, req: dict):
    score = req.get("score", 0)
    feature_feedback.add(score, {"feature": feature_name})
    return {"ok": True}
```

A note on the cost multiplier: the value `0.0002` cents per token above is a placeholder. Replace it with the figure from your own provider agreement, and keep the conversion in one function so it can be updated in a single place. Deriving cost from `usage.total_tokens` is only as accurate as the provider's usage reporting; see the fallback in the edge-cases section.

Key design decisions:

- One POST endpoint per feature, so the same wrapper serves summarisation, entity extraction, and anything else.
- Logging happens at the prompt/response boundary, not per token.
- The wrapper adds a small fixed overhead (typically single-digit milliseconds) relative to LLM latency measured in hundreds of milliseconds. Measure it rather than assuming: run `wrk` or `hey` against a stubbed endpoint and compare with the raw endpoint.

## Step 3 — handle edge cases and errors

Three edge cases tend to appear once a wrapper like this reaches production.

**1. Timeout cascades.** If the LLM endpoint is slow, every request waits for the client timeout. A circuit breaker limits the blast radius. The version below counts failures per feature (not per user, which would fragment the counter) and opens for a fixed window:

```python
from redis.exceptions import RedisError

CIRCUIT_THRESHOLD = 10
CIRCUIT_WINDOW_SECONDS = 60
CIRCUIT_OPEN_SECONDS = 30


async def check_circuit(feature_name: str):
    open_key = f"cb:open:{feature_name}"
    try:
        if await redis_client.get(open_key):
            raise HTTPException(status_code=503, detail="Service temporarily unavailable")
    except RedisError:
        # Fail open: a Redis outage should not take down the wrapper.
        return


async def record_failure(feature_name: str):
    key = f"cb:fail:{feature_name}"
    try:
        count = await redis_client.incr(key)
        if count == 1:
            await redis_client.expire(key, CIRCUIT_WINDOW_SECONDS)
        if count > CIRCUIT_THRESHOLD:
            await redis_client.set(open_key := f"cb:open:{feature_name}", "1", ex=CIRCUIT_OPEN_SECONDS)
    except RedisError:
        return
```

Two details matter here. First, the circuit state is keyed by feature, not by user, so one slow user does not trip the breaker for everyone. Second, an unreachable Redis fails open — the wrapper proceeds rather than rejecting every request, because Redis is a dependency of the breaker, not of the LLM call itself.

To verify a breaker behaves as intended, write a test that patches the LLM client to raise, calls the endpoint `CIRCUIT_THRESHOLD + 1` times, and asserts that the next call returns 503 without contacting the LLM.

**2. Prompt injection and log hygiene.** Log the prompt length, never the raw prompt, in span attributes. If a truncated copy is needed for debugging, cap it at a fixed byte length (512 bytes is a common choice) and store it in a separate, access-controlled table rather than in trace attributes, which are often widely readable.

**3. Cost spikes during rollout.** When a new feature goes live, enable it for a small cohort first. A simple approach uses a deterministic hash of the user id so a given user gets a stable answer:

```python
import hashlib


def in_rollout(user_id: str, feature_name: str, percent: float) -> bool:
    digest = hashlib.sha256(f"{feature_name}:{user_id}".encode()).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return bucket < percent
```

This avoids the trap of using `SET feature:summarise:beta_users 0.05` as a bare Redis value, which does not actually gate anything unless the application reads and interprets it as a percentage.

When a breaker is used, pair it with a health endpoint that reports current state, and make sure the open key always carries an expiry. A breaker key written without a TTL will keep a cohort returning 503s until someone notices.

## Step 4 — tests and alerting

Install test dependencies:

```bash
pip install pytest pytest-asyncio httpx
```

Create `test_main.py`:

```python
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


@pytest.fixture
def mock_redis(mocker):
    mocker.patch("redis.asyncio.Redis.get", return_value=b"1")
    yield


@pytest.mark.asyncio
async def test_feature_success(mock_redis, mocker):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [{"message": {"content": "summary"}}],
                "usage": {"total_tokens": 100},
            }

    async def fake_post(*args, **kwargs):
        return FakeResponse()

    mocker.patch("httpx.AsyncClient.post", side_effect=fake_post)

    resp = client.post(
        "/feature/summarise",
        json={
            "user_id": "u1",
            "feature_name": "summarise",
            "prompt": "long document",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["cost_cents"] == pytest.approx(100 * 0.0002)
```

The original version of this test returned a plain dict from the mocked `post`, which would fail because the code calls `raise_for_status()` on the result. The `FakeResponse` class above fixes that.

Prometheus scrapes the exporter's port every 15 seconds:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'llm-feature'
    scrape_interval: 15s
    static_configs:
      - targets: ['app:8001']
```

Alert rules:

```yaml
groups:
- name: llm-feature
  rules:
  - alert: HighFeatureLatency
    expr: histogram_quantile(0.95, sum(rate(llm_feature_duration_seconds_bucket{feature="summarise"}[5m])) by (le)) > 2
    for: 5m
    labels:
      severity: page
    annotations:
      summary: "Summarise feature 95th percentile above 2s"
```

The `sum(...) by (le)` wrapper is required because `histogram_quantile` needs a single bucket series per `le` value; without it, multiple label combinations produce ambiguous results.

### How to measure whether this is working

Do not trust a summary table from someone else's deployment. Measure your own. Instrument the following and compare a two-week window before and after the wrapper:

- **Log volume:** `du -sh /var/log/llm/` weekly, or the equivalent metric from your log sink.
- **API latency:** the wrapper's own `llm_feature_duration_seconds` histogram, compared against the LLM endpoint's own reported latency. The difference is the wrapper's overhead.
- **Alert quality:** count pages per week and classify each as actionable or not. A useful wrapper should reduce the ratio of non-actionable pages.
- **Cost:** sum `llm_feature_cost_cents` per feature per day. This is the figure that lets you argue for retiring a feature.

Only after collecting this data can anyone claim a percentage improvement.

## Common questions

**How do I measure LLM feature adoption without user ids?**
Collect `user_id` only if your privacy policy permits it. If it does not, derive a stable pseudonymous id from the session token (for example, a salted hash) and store only the hash. Omitting identity entirely means you cannot segment feedback by cohort, which limits the analysis but does not break the metrics.

**What if my LLM endpoint does not return token usage?**
Count tokens client-side and use that figure. For models with a published tokenizer, use it directly; for others, a word count multiplied by a documented ratio is a rough substitute. Whichever fallback you use, record whether the value was server-reported or estimated, so cost dashboards can distinguish the two.

```python
from functools import lru_cache


@lru_cache(maxsize=32)
def get_tokenizer(model_name: str):
    import tiktoken
    return tiktoken.encoding_for_model(model_name)


def safe_token_count(text: str, model_name: str) -> int:
    encoding = get_tokenizer(model_name)
    return len(encoding.encode(text))
```

Note that `tiktoken.encoding_for_model` only knows about OpenAI models; for other models you will need to pass an explicit encoding name or fall back to the word-count estimate.

**How do I run this cheaply on a serverless platform?**
Package the wrapper as a container image and deploy it to a function service. Replace the Prometheus exporter with the platform's native metrics emitter (for example, CloudWatch embedded metric format on AWS) so you do not need to run a scrape endpoint. Cold starts add latency; keep the image small by using a multi-stage build and excluding development tooling.

**When should I move from Redis feature flags to a dedicated flag service?**
Move when you need rule-based targeting (for example, "enable summarise only for users with more than five prior messages"), or when flag keys and evaluation logic outgrow what a Redis client can reasonably express. The migration is usually a single function swap in the wrapper.

## Optional: clustering feedback with pgvector

Storing feedback scores alone tells you a feature is disliked; it does not tell you why. Storing an embedding alongside each feedback row lets you find clusters of similar failing prompts.

```sql
CREATE TABLE IF NOT EXISTS user_feedback (
    id SERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    prompt TEXT NOT NULL,
    response TEXT,
    score SMALLINT NOT NULL CHECK (score BETWEEN 1 AND 5),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    embedding vector(384)
);
```

Backfill embeddings with a sentence-transformer model:

```python
from sentence_transformers import SentenceTransformer
import psycopg2
from psycopg2.extras import execute_batch
import os

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def backfill_embeddings():
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    cur = conn.cursor()
    cur.execute("SELECT id, prompt FROM user_feedback WHERE embedding IS NULL")
    rows = cur.fetchall()
    if not rows:
        return
    embeddings = model.encode([r[1] for r in rows])
    with conn.cursor() as update_cur:
        execute_batch(
            update_cur,
            "UPDATE user_feedback SET embedding = %s WHERE id = %s",
            [(emb.tolist(), row[0]) for row, emb in zip(rows, embeddings)],
        )
    conn.commit()
    conn.close()
```

The `emb.tolist()` conversion matters: passing a NumPy array directly to psycopg2 will not serialise correctly for a `vector` column.

To find the nearest neighbours of a low-scoring prompt, use the `<->` operator with an explicit vector literal, and remember to index the column for anything beyond a few thousand rows:

```sql
CREATE INDEX ON user_feedback USING hnsw (embedding vector_cosine_ops);
```

## Decision checklist before shipping an LLM feature

- Does the feature emit exactly one structured event per call, with a feature name and a success flag?
- Is there a cost figure attached, even if approximate, and is it labelled as an estimate?
- Is the feature gated by a rollout mechanism that can be turned off without a deploy?
- Is there a circuit breaker whose state is keyed by feature and whose keys always expire?
- Is there an alert rule tied to a stated SLO, and has it been tested by deliberately exceeding the threshold in staging?
- Does the dashboard answer "should we keep this feature?" in one glance?

## Do this next

Pick one LLM-backed feature currently in production. Add a single structured log line at its prompt/response boundary containing `feature_name`, `model_name`, `duration_ms`, `total_tokens`, and `success`. Deploy it, then run this against your endpoint for one hour:

```bash
hey -z 1h -c 4 -m POST -T application/json \
  -d '{"user_id":"test","feature_name":"summarise","prompt":"hello"}' \
  http://localhost:8000/feature/summarise
```

Compare the wrapper's reported latency against the LLM endpoint's own latency logs. That difference is the observability layer's true overhead, and it is the first number you need before deciding whether the rest of this design is worth adopting.
