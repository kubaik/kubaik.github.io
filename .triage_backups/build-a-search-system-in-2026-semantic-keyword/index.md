# Build a search system in 2026: semantic, keyword

Most search tutorials stop at the happy path: one query type, a small index, no load. Production search is less about writing the query and more about how keyword matching, vector similarity, caching, and reranking fit together when the index is large and traffic is uneven. This article walks through a working hybrid search service, then covers the failure modes that show up once real traffic arrives — and how to measure each one rather than guess.

## What you'll build

A small local search service exposing three endpoints:

1. `/keyword` — classic full-text search using PostgreSQL's `tsquery`
2. `/semantic` — cosine similarity using pgvector
3. `/hybrid` — reranks keyword candidates with semantic scores

The stack runs in Docker Compose so results are reproducible. You'll need a Unix-like shell, Docker with the Compose plugin, Python 3.11, and Node (only if you want to run the load test with k6). Install commands for macOS and Debian-family Linux:

```bash
# macOS with Homebrew
brew install docker docker-compose python@3.11 node redis
# Ubuntu/Debian
sudo apt update && sudo apt install docker.io docker-compose-plugin python3.11 nodejs npm redis-server
```

## Step 1 — environment setup

Create a project folder and lay out the files:

```bash
mkdir search-2026 && cd search-2026
mkdir -p postgres redis app
```

`docker-compose.yml`:

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    ports:
      - "5432:5432"
    environment:
      POSTGRES_USER: search
      POSTGRES_PASSWORD: search
      POSTGRES_DB: search
    volumes:
      - ./postgres/init.sql:/docker-entrypoint-initdb.d/init.sql
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U search -d search"]
      interval: 2s
      timeout: 1s
      retries: 5

  redis:
    image: redis:7.2-alpine
    ports:
      - "6379:6379"
    volumes:
      - ./redis/redis.conf:/usr/local/etc/redis/redis.conf
    command: redis-server /usr/local/etc/redis/redis.conf

  app:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DB_URL=postgresql://search:search@postgres:5432/search
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_started

volumes:
  pgdata:
```

Note: the `version:` key is obsolete in Compose v2 and produces a warning; it is omitted here. Use the official `pgvector/pgvector` image rather than a third-party build so the extension is installed for the correct PostgreSQL major version.

`postgres/init.sql`:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
  id BIGSERIAL PRIMARY KEY,
  title TEXT NOT NULL,
  body TEXT NOT NULL,
  tags TEXT[] NOT NULL,
  embedding vector(384)  -- sentence-transformers/all-MiniLM-L6-v2
);

CREATE INDEX idx_documents_title_tsv ON documents USING GIN (to_tsvector('english', title));
CREATE INDEX idx_documents_body_tsv ON documents USING GIN (to_tsvector('english', body));
CREATE INDEX idx_documents_tags_gin ON documents USING GIN (tags);
CREATE INDEX idx_documents_embedding_cosine ON documents USING hnsw (embedding vector_cosine_ops);
```

Two notes on the index choice. HNSW is the default recommendation for query latency in modern pgvector; IVFFlat requires enough rows to train its lists and needs a `lists` parameter tuned to row count. HNSW avoids that tuning step at the cost of higher memory use.

The `ALTER TABLE ... SET STORAGE PLAIN` line that often appears in tutorials is not needed here — pgvector already stores vectors in a compact form, and forcing plain storage can increase disk usage rather than reduce it.

`redis/redis.conf`:

```ini
bind 0.0.0.0
dir /data
appendonly yes
maxmemory 500mb
maxmemory-policy allkeys-lru
```

`app/requirements.txt`:

```
fastapi==0.110.1
uvicorn==0.29.0
sentence-transformers==2.6.1
pgvector==0.2.1
psycopg2-binary==2.9.9
redis==5.0.1
httpx==0.27.0
```

`Dockerfile`:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/main.py .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Bring the stack up and verify connectivity:

```bash
docker compose up -d --build
docker compose exec postgres pg_isready -U search -d search
docker compose exec redis redis-cli ping
```

If you see `could not open extension control file`, the extension is not present in the image — switch to the `pgvector/pgvector` image, which bundles it.

## Step 2 — core implementation

`app/main.py`. Embeddings are generated in Python; the database stays thin.

```python
from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer
import psycopg2, redis, os, json

DB_URL = os.getenv("DB_URL")
REDIS_URL = os.getenv("REDIS_URL")
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
model = SentenceTransformer(MODEL_NAME, device="cpu")

app = FastAPI()

conn = psycopg2.connect(DB_URL, connect_timeout=3)
redis_client = redis.from_url(REDIS_URL)

@app.post("/ingest")
async def ingest(title: str, body: str, tags: list[str]):
    embedding = model.encode(f"{title} {body}", convert_to_tensor=False)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents (title, body, tags, embedding)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (title, body, tags, embedding.tobytes())
        )
        new_id = cur.fetchone()[0]
        conn.commit()
    return {"id": new_id}

@app.get("/keyword")
async def keyword(q: str):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, title, body, ts_rank(
                setweight(to_tsvector('english', title), 'A') ||
                setweight(to_tsvector('english', body), 'B'),
                plainto_tsquery('english', %s)
            ) AS rank
            FROM documents
            WHERE to_tsvector('english', title || ' ' || body) @@ plainto_tsquery('english', %s)
            ORDER BY rank DESC
            LIMIT 20
            """,
            (q, q)
        )
        rows = cur.fetchall()
    return [{"id": r[0], "title": r[1], "body": r[2]} for r in rows]

@app.get("/semantic")
async def semantic(q: str):
    cache_key = f"semantic:emb:{q}"
    cached_emb = redis_client.get(cache_key)
    if cached_emb:
        query_embedding = bytes(cached_emb)
    else:
        query_embedding = model.encode(q, convert_to_tensor=False).tobytes()
        redis_client.setex(cache_key, 300, query_embedding)

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, title, body,
                   1 - (embedding <=> %s) AS score
            FROM documents
            ORDER BY embedding <=> %s
            LIMIT 20
            """,
            (query_embedding, query_embedding)
        )
        rows = cur.fetchall()
    return [{"id": r[0], "title": r[1], "score": float(r[3])} for r in rows]

@app.get("/hybrid")
async def hybrid(q: str):
    # Step 1: fetch keyword candidates
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, title, body FROM documents
            WHERE to_tsvector('english', title || ' ' || body) @@ plainto_tsquery('english', %s)
            ORDER BY ts_rank(
                setweight(to_tsvector('english', title), 'A') ||
                setweight(to_tsvector('english', body), 'B'),
                plainto_tsquery('english', %s)
            ) DESC
            LIMIT 200
            """,
            (q, q)
        )
        keyword_rows = cur.fetchall()

    if not keyword_rows:
        return []

    # Step 2: rerank with semantic
    query_embedding = model.encode(q, convert_to_tensor=False).tobytes()
    keyword_ids = [r[0] for r in keyword_rows]

    with conn.cursor() as cur:
        placeholders = ",".join(["%s"] * len(keyword_ids))
        cur.execute(
            f"""
            SELECT id, title, body,
                   1 - (embedding <=> %s) AS score
            FROM documents
            WHERE id IN ({placeholders})
            """,
            [query_embedding] + keyword_ids
        )
        semantic_rows = cur.fetchall()

    # Step 3: reorder
    id_to_semantic = {r[0]: float(r[3]) for r in semantic_rows}
    hybrid_rows = sorted(keyword_rows, key=lambda r: id_to_semantic.get(r[0], 0), reverse=True)

    return [{"id": r[0], "title": r[1], "score": id_to_semantic.get(r[0], 0)} for r in hybrid_rows[:20]]
```

Design decisions worth understanding:

- **Cosine distance (`<=>`)** is the right operator when embeddings are normalized, which the MiniLM family is. It is stable across corpus sizes and matches how the model was trained.
- **`setweight`** gives title matches more influence than body matches. Whether this helps depends on your corpus; treat it as a hypothesis to test with the recall measurement described below, not a guaranteed improvement.
- **Candidate count** (the `LIMIT 200` above) controls the recall/latency tradeoff. Larger candidate sets raise recall at the cost of a bigger `IN` list and more reranking work. Measure before changing it.
- **Ordering by the distance operator** rather than by the computed score lets the index do the ordering. Sorting on a derived column (`1 - (embedding <=> %s)`) forces a full scan.

Seed the index. Assuming a JSONL file where each line has `title`, `body`, and `tags`:

```bash
docker compose exec app python -c "
from main import conn, model
import json
with open('questions.jsonl') as f:
    for line in f:
        d = json.loads(line)
        emb = model.encode(f\"{d['title']} {d['body']}\", convert_to_tensor=False)
        with conn.cursor() as cur:
            cur.execute(
                'INSERT INTO documents (title, body, tags, embedding) VALUES (%s,%s,%s,%s)',
                (d['title'], d['body'], d['tags'], emb.tobytes())
            )
        conn.commit()
"
```

This commits one row at a time, which is far too slow for large corpora. Batch inserts (or `COPY`) in groups of a few hundred rows per transaction.

## Step 3 — failure modes and fixes

### 1. Cache miss storms

If the raw query embedding is not cached, every request pays the model inference cost. Cache the embedding bytes under a key derived from the query, with a short TTL. The `/semantic` endpoint above shows the pattern; the same cache should be shared by `/hybrid`.

Note that `redis_client.get` returns bytes, so `bytes(cached_emb)` is a no-op copy — but if you switch to a client that returns `str`, decode explicitly and be consistent.

### 2. Vector dimension mismatch

Changing the embedding model without migrating the column type produces an error like:

```
ERROR:  expected 384 dimensions, not 768
```

The fix is to `ALTER TABLE documents ALTER COLUMN embedding TYPE vector(768)`, drop and rebuild the index, and re-ingest every row. Pin the model name in an environment variable (as above) so the app and the schema can never disagree silently.

### 3. Connection handling

`psycopg2.connect` returns a single connection, not a pool. FastAPI's async endpoints run in a threadpool, so concurrent requests will interleave on that one connection and raise `InterfaceError` or corrupt cursors. Use `psycopg2.pool.ThreadedConnectionPool` (or an async driver) and acquire a connection per request:

```python
from psycopg2.pool import ThreadedConnectionPool
pool = ThreadedConnectionPool(minconn=2, maxconn=20, dsn=DB_URL)
```

Note that `min_size` and `max_size` are not valid `psycopg2.connect` keyword arguments — they belong to the pool class.

### 4. Cold start latency

The first call to `model.encode` loads weights into memory. On a small ARM instance this can take a second or more. Warm the model at startup rather than on first request:

```python
@app.on_event("startup")
async def warmup():
    model.encode("warmup")
```

### 5. Sort spilling to disk

Large sorts and index builds can spill when `work_mem` is too small. Rather than guessing a global value, check the actual plan:

```sql
EXPLAIN (ANALYZE, BUFFERS) SELECT id FROM documents ORDER BY embedding <=> '...' LIMIT 20;
```

Look for `Sort Method: external merge` in the output. If you see it, raise `work_mem` for the session or the role that runs search queries rather than system-wide:

```sql
SET work_mem = '64MB';
```

## Step 4 — observability and testing

Add tracing so you can see where time goes:

```bash
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp
```

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

trace.set_tracer_provider(TracerProvider())
otlp_exporter = OTLPSpanExporter(endpoint="http://otel-collector:4318/v1/traces")
trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(otlp_exporter))
tracer = trace.get_tracer(__name__)

@app.get("/semantic")
async def semantic(q: str):
    with tracer.start_as_current_span("semantic_search"):
        # ... existing code ...
        span = trace.get_current_span()
        span.set_attribute("hits_returned", len(rows))
        return [{"id": r[0], "title": r[1], "score": float(r[3])} for r in rows]
```

Set SLOs before you have data, then revise them once you do. A reasonable starting point for a single-node deployment is P95 under 150 ms for keyword, under 350 ms for semantic, and under 450 ms for hybrid — but these are targets, not guarantees, and depend heavily on index size and hardware.

Load test with k6:

```javascript
// loadtest.js
import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  stages: [
    { duration: '30s', target: 50 },
    { duration: '2m', target: 200 },
    { duration: '30s', target: 0 },
  ],
  thresholds: {
    http_req_duration: ['p(95)<450'],
  },
};

export default function () {
  const res = http.get('http://localhost:8000/hybrid?q=docker+compose');
  check(res, { 'status 200': (r) => r.status === 200 });
  sleep(0.5);
}
```

```bash
k6 run --vus 50 --duration 3m loadtest.js
```

Run this on the same machine class you intend to deploy to, or the numbers mean nothing.

Smoke tests:

```python
# tests/test_search.py
import os
import psycopg2
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

@pytest.fixture(autouse=True)
def reset_db():
    with psycopg2.connect(os.getenv("DB_URL")) as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE documents")
            conn.commit()

def test_keyword():
    client.post("/ingest", params={"title": "Hello", "body": "World", "tags": ["a"]})
    r = client.get("/keyword", params={"q": "hello"})
    assert r.status_code == 200
    assert len(r.json()) == 1

def test_semantic():
    client.post("/ingest", params={"title": "Docker", "body": "Containers", "tags": ["b"]})
    r = client.get("/semantic", params={"q": "docker containers"})
    assert r.status_code == 200
    assert r.json()[0]["score"] > 0.5

def test_hybrid():
    client.post("/ingest", params={"title": "FastAPI", "body": "Async", "tags": ["c"]})
    r = client.get("/hybrid", params={"q": "api"})
    assert r.status_code == 200
    assert len(r.json()) > 0
```

Note that the endpoints declare their inputs as query parameters, so `TestClient` calls use `params=`, not `json=`. The semantic score threshold is set to 0.5 rather than 0.8 because MiniLM cosine scores for loosely related text commonly land in the 0.3–0.6 range; a 0.8 threshold will fail on many valid pairs.

```bash
docker compose exec app pytest -q
```

## Measuring recall instead of guessing it

Claims like "hybrid beats semantic by X%" are only meaningful if you measure them on your own data. The procedure:

1. **Build a labeled set.** Take 100–500 queries and, for each, list the document IDs that a human would consider relevant. This is the expensive part; there is no shortcut.
2. **Run each approach** against the labeled set and record the top-10 IDs per query.
3. **Compute Recall@10** as `(relevant docs in top 10) / (total relevant docs)` and **MRR@10** as the reciprocal of the rank of the first relevant result, averaged across queries.
4. **Compare.** If hybrid does not beat both baselines on your corpus, the reranking step is not earning its latency.

A small script is enough:

```python
def recall_at_k(returned_ids, relevant_ids, k=10):
    top = set(returned_ids[:k])
    return len(top & set(relevant_ids)) / max(len(relevant_ids), 1)

def reciprocal_rank(returned_ids, relevant_ids):
    for i, doc_id in enumerate(returned_ids[:10], start=1):
        if doc_id in relevant_ids:
            return 1.0 / i
    return 0.0
```

For latency, instrument the endpoint with a histogram (OpenTelemetry, Prometheus, or even a simple in-process list during development) and report the 50th, 95th, and 99th percentiles. Cost per query is `(instance hourly cost) / (queries per hour)` — compute it from your own billing data rather than copying someone else's instance sizing.

## Common questions

**How do I add multi-language support without doubling the index?**
Use a multilingual embedding model that produces the same dimension as the current one, add a `language` column, and create partial indexes per language:

```sql
CREATE INDEX idx_documents_embedding_es ON documents USING hnsw (embedding vector_cosine_ops)
WHERE language = 'es';
```

Filter by language before ranking. Note that PostgreSQL's built-in full-text search needs a language-specific configuration (`to_tsvector('spanish', ...)`) to work correctly across languages — the `'english'` configuration used above will not stem non-English text properly.

**Can I use a dedicated search engine instead of PostgreSQL?**
Yes. Dedicated engines typically offer better keyword relevance tuning and built-in faceting, at the cost of running another service and losing the ability to join against your relational data. Whether hybrid reranking is supported depends on the specific engine and its version — check the documentation for your version rather than assuming. If your access patterns are read-heavy and you don't need joins, a dedicated engine is often the simpler operational choice.

**What happens as the index grows to millions of rows?**
Memory becomes the binding constraint. HNSW indexes must largely fit in RAM to deliver low latency; check the index size with `\di+` and compare against available memory. For very large corpora, partition the table by a natural key (tenant, date, category) and query only the relevant partitions. Rebuilding an HNSW index on a large table is expensive, so plan for it.

**How do I handle real-time updates?**
Insert or upsert the new row with its embedding, and let the index update incrementally. HNSW supports inserts without a full rebuild; IVFFlat does not handle inserts as gracefully and may require periodic reindexing. For high write volume, batch updates and reindex during low-traffic windows.

## Your next 30 minutes

Pick one endpoint — `/hybrid` is the best candidate — and instrument it. Add a timer around the database call and around the model call separately, run the k6 script for three minutes, and record the two numbers. That split tells you immediately whether the bottleneck is inference (model time dominates) or the vector index (database time dominates), and the two problems have completely different fixes. Do this before tuning any parameter, because tuning the wrong layer is the most common way to spend a week on a problem you don't have.
