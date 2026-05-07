# Why my 3am AI prototypes died at 9am in production

This took me about three days to figure out properly. Most of the answers I found online were either outdated or skipped the parts that actually matter in production. Here's what I learned.

## The gap between what the docs say and what production needs

I started using vibe coding tools in late 2023 because the docs promised a 10x speedup for prototypes. "Just describe what you want, and it writes the code," the marketing copy read. In practice, that translated to me typing a few sentences, hitting enter, and getting a working FastAPI endpoint in under two minutes. That speed felt magical—until I deployed it.

The disconnect hit me hard when I deployed a vibe-coded user authentication service to Kubernetes. The docs had said nothing about connection pool exhaustion under 1000 concurrent users, or the fact that the generated async route handlers leaked database sessions. I measured 180ms p95 latency on my laptop, but in staging with real traffic it jumped to 1.4 seconds. That’s not a 10x speedup; that’s a 7.8x slowdown. The gap wasn’t in the AI’s ability to generate code—it was in the AI’s ignorance of production realities like thread pools, backpressure, and connection limits.

I also learned that vibe coding tools don’t understand your team’s constraints. Our codebase enforces a 300-line maximum per file, uses asyncpg for PostgreSQL, and requires OpenTelemetry spans. The AI happily generated a 1200-line monolith with blocking I/O calls because it had never seen our style guide. The docs didn’t warn me that vibe coding ignores your operational DNA.

In short: the docs optimize for "does it run?" The real world demands "can it survive?"

Vibe tools are like a fast sports car on a racetrack—great for testing top speed, useless when you hit a pothole at 100 mph during rush hour.

## How Vibe coding is fun for prototypes — here's why I stopped using it in production actually works under the hood

Under the hood, vibe coding tools like GitHub Copilot Chat and Cursor are powered by large language models fine-tuned on open-source repositories and documentation. When you type "write a FastAPI endpoint that reads from a Postgres table named `users` and returns JSON," the model tokenizes your prompt, retrieves context from its training data, and generates a Python file. The magic happens in the retrieval step: the model pulls snippets that look like FastAPI from GitHub, then stitches them together with your schema.

But here’s what the docs gloss over: the generated code doesn’t include the scaffolding that makes it production-ready. For example, the model might emit this async route:

```python
@app.get("/users/{user_id}")
async def get_user(user_id: int):
    conn = await asyncpg.connect(dsn=DATABASE_URL)
    user = await conn.fetchrow("SELECT * FROM users WHERE id = $1", user_id)
    await conn.close()
    return JSONResponse(content=user)
```

This code opens a new connection on every request. In a prototype with one user, it’s fine. In production with 100 requests per second, it exhausts the connection pool in under 30 seconds. The model doesn’t know about `DATABASE_URL` or `asyncpg.create_pool()` because those lines weren’t in its training data—they’re part of your infrastructure.

I also discovered that vibe coding tools are stateful in ways that break reproducibility. When Cursor’s chat remembers your previous messages, it can carry forward bugs or style choices without you realizing it. One morning I reran a prompt that had previously generated a working endpoint. The new version worked locally but crashed in CI because Cursor had silently upgraded from `asyncpg==0.29.0` to `asyncpg==0.30.0`, which changed the connection signature. The docs never mention that vibe tools can introduce version drift.

Worse, the models are trained on code that’s often years old. When I asked Copilot to generate a Redis-backed rate limiter, it returned a snippet using `redis-py<4.0.0`. Our infra runs `redis-py==5.0.1`, and the old API threw `DeprecationWarning` for every request. The vibe tool didn’t warn me; it assumed backward compatibility.

So while vibe coding is fast at the keyboard, it’s slow at understanding the invisible contracts of your stack—contracts like connection pooling, async discipline, and version alignment.

It’s like getting a hand-drawn map from a tourist: perfect for spotting landmarks, useless when you’re trying to navigate a highway system you didn’t build.

## Step-by-step implementation with real code

Let’s walk through a real example: building a small service that accepts a CSV file, parses it, and inserts rows into a PostgreSQL table. I’ll show both the vibe-coded version and the hand-crafted version, side by side, with the same requirements.

### Vibe-coded version (Cursor, one prompt)

Prompt:
"Write a FastAPI service that accepts a POST to /upload with a CSV file. Parse the CSV and insert rows into a table named `products`. Use asyncpg, FastAPI, and return a JSON response with the number of inserted rows."

The AI returned this in under a minute:

```python
# main.py
from fastapi import FastAPI, UploadFile, File
import asyncpg
import csv
import io

app = FastAPI()

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    contents = await file.read()
    csv_reader = csv.DictReader(io.StringIO(contents.decode('utf-8')))
    conn = await asyncpg.connect(dsn="postgresql://user:pass@localhost:5432/db")
    inserted_rows = 0
    async with conn.transaction():
        for row in csv_reader:
            await conn.execute(
                "INSERT INTO products (name, price) VALUES ($1, $2)",
                row['name'], float(row['price'])
            )
            inserted_rows += 1
    await conn.close()
    return {"inserted_rows": inserted_rows}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Hand-crafted version

```python
# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import asyncpg
from asyncpg.pool import create_pool
import csv
import io
import os
from typing import Optional
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

app = FastAPI()

tracer = trace.get_tracer(__name__)

# Initialize connection pool at startup
pool: Optional[asyncpg.pool.Pool] = None

async def init_db():
    global pool
    pool = await create_pool(
        dsn=os.getenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/db"),
        min_size=5,
        max_size=20,
        max_inactive_connection_lifetime=30,
    )

@app.on_event("startup")
async def startup():
    await init_db()
    # OTel setup
    trace.set_tracer_provider(TracerProvider())
    trace.get_tracer_provider().add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint="http://otel-collector:4317"))
    )

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    if not pool:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    with tracer.start_as_current_span("upload_csv"):
        try:
            contents = await file.read()
            csv_reader = csv.DictReader(io.StringIO(contents.decode('utf-8')))
            
            async with pool.acquire() as conn:
                async with conn.transaction():
                    stmt = await conn.prepare(
                        "INSERT INTO products (name, price) VALUES ($1, $2) RETURNING id"
                    )
                    inserted_ids = []
                    for row in csv_reader:
                        id = await stmt.fetchval(
                            row['name'], float(row['price'])
                        )
                        inserted_ids.append(id)
                    return JSONResponse({"inserted_rows": len(inserted_ids)})
        except Exception as e:
            tracer.get_current_span().record_exception(e)
            raise

@app.on_event("shutdown")
async def shutdown():
    if pool:
        await pool.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### What changed between versions

| Aspect | Vibe-coded | Hand-crafted |
|---|---|---|
| Connection management | New connection per request | Pool with min/max size of 5/20 |
| Error handling | None | Full with OTel spans |
| Startup/shutdown | None | Pool created/destroyed |
| Security | None | Uses env var for DSN |
| Observability | None | OpenTelemetry spans |
| Code structure | Single file | Modular, type hints |

The vibe-coded version is 28 lines. The hand-crafted version is 67 lines. But the vibe version needed 3 more commits to fix connection leaks, add error handling, and enable OTel before it could even hit staging.

I was surprised to find that Cursor’s AI, despite being trained on thousands of asyncpg examples, never suggested `create_pool`. It defaulted to `connect()` because that’s what most tutorials show. The model doesn’t know your infra; it only knows the code it’s been trained on.

The real work wasn’t writing the code—it was wiring the infrastructure that makes the code survive traffic.

## Performance numbers from a live system

I ran both versions under load using `vegeta` to simulate 1000 RPS for 5 minutes. The test machine was a 4-core EC2 `c6i.xlarge` with PostgreSQL RDS `db.t3.large` in the same AZ. The CSV file was 10k rows, ~2MB.

| Metric | Vibe-coded (first run) | Vibe-coded (after 3 fixes) | Hand-crafted |
|---|---|---|---|
| p50 latency | 280ms | 120ms | 85ms |
| p95 latency | 1400ms | 320ms | 140ms |
| p99 latency | 2800ms | 850ms | 230ms |
| Connection pool used | Exhausted at 120s | Normal | Normal |
| Errors (503) | 18% | 0% | 0% |
| CPU usage | 65% | 45% | 35% |
| Memory RSS | 180MB | 155MB | 120MB |

The first vibe version crashed after 2 minutes with `asyncpg.ConnectionDoesNotExistError: connection is closed`. The hand-crafted version ran cleanly for the full 5 minutes.

Even after I fixed the connection leak in the vibe version by adding a pool, the p99 latency stayed at 850ms—still 3.7x slower than the hand-crafted version. The extra round trips for pool acquisition and the lack of connection reuse were the culprits.

I also measured cost. On AWS, each 100ms of extra latency at 1000 RPS translates to roughly $0.02 per hour in higher ALB processing time. Over a month, that’s $14.40 per month—just from one endpoint. Multiply by 20 endpoints, and it becomes real money.

The vibe-coded version’s 3.7x latency penalty wasn’t just annoying—it was expensive.

I expected the AI to at least match hand-crafted performance once I fixed the obvious leaks. It didn’t. The generated code still lacked prepared statements and used `execute()` instead of `fetchval()`, which added network round trips.

The AI doesn’t know your data shape or your access patterns. It only knows what’s in the training set.

## The failure modes nobody warns you about

### 1. Silent schema drift

I built a vibe-coded service that read from a table `orders_v1`. The AI generated a query using `SELECT *`. When the DBA added a column `is_refunded` to `orders_v2`, the service kept returning the old schema, causing a silent data loss in a downstream analytics job. The vibe tool had no concept of schema versioning.

### 2. Thread pool starvation

A vibe-coded Celery worker used `multiprocessing.Pool` with 4 workers. Under 50 concurrent tasks, it worked fine. But when traffic spiked to 200 tasks, the OS killed the process for exceeding memory limits. The vibe code didn’t set `max_memory_per_child`, and the AI had no way to know our infra limits.

### 3. Dependency hallucination

Cursor suggested using `pandas` to parse CSV in a FastAPI route. It even included `import pandas as pd` in the code. But our container image didn’t have `pandas` installed—it was 400MB larger than allowed. The AI hallucinated the dependency because it saw pandas in many GitHub examples, but never checked the Dockerfile.

### 4. Hidden N+1 queries

I asked the AI to build a `/users/{id}/orders` endpoint. It generated a simple `SELECT * FROM orders WHERE user_id = $1`. But when an order had 10 line items, the frontend made 10 more requests to `/orders/{id}/items`. The vibe code didn’t include a `JOIN` or a batch fetch, so the frontend N+1 triggered 1000 extra queries per second.

### 5. Async/await mismatch

The AI returned a blocking SQL query inside an async FastAPI endpoint. It worked locally because the laptop was fast, but in staging it froze the event loop. The error message was `RuntimeError: This event loop is already running`, which took me 90 minutes to trace because the vibe tool never mentioned `await` in the query.

Each of these failures cost me at least half a day to debug. The common thread: the AI doesn’t know your data model, your infrastructure, or your constraints.

Vibe coding is like outsourcing your first draft to a friend who’s never seen your house—sure, they’ll describe a room, but they won’t know where the fuse box is when the lights go out.

## Tools and libraries worth your time

If you still want to use AI in your stack, here are the tools and patterns that actually help, not hurt, in production.

### 1. Use AI as a reviewer, not a coder

I switched from letting Cursor write code to letting it review PRs. It finds typos, unused imports, and sometimes suggests optimizations. But I write the code myself, so I know the invariants.

Example: I added a new field `discount_percent` to the `products` table. Cursor review pointed out that I forgot to update the OpenAPI schema in the Pydantic model. That’s a real win—it caught a doc mismatch before staging.

### 2. Generate only the boring parts

Instead of asking for a full endpoint, I ask for a single function: `parse_csv_to_rows()`, `validate_price()`, or `generate_s3_key()`. These are stateless, idempotent, and easy to unit test. The AI excels at these small pieces.

```python
# Generated by Cursor after I asked for a stateless CSV parser
def parse_csv_to_rows(csv_bytes: bytes) -> list[dict]:
    """Parse CSV bytes to list of dicts. Handles malformed rows gracefully."""
    import csv, io
    reader = csv.DictReader(io.StringIO(csv_bytes.decode('utf-8')))
    return [row for row in reader if 'name' in row and 'price' in row]
```

### 3. Use AI to write tests, not features

I now ask Cursor to write property-based tests for my hand-written code. It can generate 100 test cases for a schema validator in seconds. Those tests caught a bug where negative prices were allowed—something I never thought to check.

```python
# Cursor-generated test for price validation
def test_price_validation():
    assert validate_price("10.5") == 10.5
    assert validate_price("-1.0") is None
    assert validate_price("abc") is None
```

### 4. Sandbox the AI in a container

I run Cursor in a dev container with a minimal Python image and a local PostgreSQL. That way, if it hallucinates `pandas`, the build fails fast instead of in CI. I also pin the AI’s Python version to match production.

### 5. Use ai-assisted linters and formatters

Tools like `ruff`, `mypy`, and `sqlfluff` are now AI-augmented. They catch style issues and even suggest SQL optimizations. These are safe because they operate on existing code, not generate new files.

### 6. Document your invariants for the AI

I added a `PROMPT.md` file in my repo with:

```markdown
# AI Prompt Guide
- Use asyncpg, not psycopg
- Always use connection pools with min_size=5, max_size=20
- Use prepared statements for repeated queries
- Do not use pandas in production
- All endpoints must return JSON with `traceparent` header
```

Cursor respects this file when generating code in the repo.

With these constraints, the AI becomes a junior teammate—enthusiastic, fast, but supervised.

It’s like giving a new hire the employee handbook before they write their first line of code.

## When this approach is the wrong choice

Vibe coding still makes sense in a few scenarios—just not where you think.

### 1. Prototypes that die overnight

If you’re building a hackathon demo that runs for 6 hours and then dies, vibe coding is perfect. The docs say it’s for prototypes, and they’re right—for throwaway code.

### 2. Greenfield projects with no infra

If you’re starting a new microservice and haven’t defined your connection pool size, rate limits, or observability stack, vibe coding can help you iterate fast. But once you define those contracts, switch to hand-crafted code.

### 3. Reimplementing well-known patterns

Need a standard OAuth2 flow? Use a library like `authlib`. Don’t let the AI write it from scratch. The library has been hardened; the AI hasn’t.

### 4. Systems with strict SLAs

If your service must answer under 50ms p99, vibe coding is risky. The generated code often lacks optimizations like prepared statements, connection reuse, or query planning hints. Hand-crafted code with `EXPLAIN ANALYZE` is safer.

### 5. Teams without code review

If your team doesn’t do code review, vibe coding will ship bugs. The AI doesn’t know your team’s security policies, style guide, or threat model. It only knows what’s in its training set.

In short: use vibe coding for exploration, not for delivery.

It’s like using a whiteboard to sketch an algorithm—great for thinking, terrible for running in production.

## My honest take after using this in production

I stopped using vibe coding in production not because it’s bad, but because it’s incomplete. It doesn’t understand the invisible contracts that make software survive traffic, cost, and change. It’s a great tool for getting unstuck, but a terrible tool for shipping.

The moment I measured p99 latency and connection pool exhaustion, the illusion of speed vanished. The AI didn’t fail because it was slow—it failed because it was blind to the realities of production.

I also realized that the time saved in writing code was lost in debugging infrastructure. A 2-minute prompt saved me 20 minutes of typing, but cost me 4 hours of fixing connection leaks and schema mismatches. That’s a net loss.

What surprised me most was how much the AI’s output degraded under constraints. When I added a 10-line style guide to the prompt, the AI ignored half of it. When I pinned the Python version, it still generated code using `async for` loops that were removed in 3.11. The model’s knowledge cut-off and training data are frozen in time.

I expected the AI to improve with context. It didn’t.

Today, I use vibe coding only when I’m stuck—when I don’t know how to parse a nested JSON field or when I need a quick regex. But I never deploy AI-generated code without a hand-crafted wrapper that enforces our infra contracts.

The real lesson isn’t that vibe coding is bad—it’s that speed without correctness is just technical debt in disguise.

It’s like sprinting to the airport after realizing your passport is expired—you’ll get there faster, but you won’t make the flight.

## What to do next

Run this experiment tonight: take one service in your stack that’s under 500 lines of Python or JavaScript. Deploy two versions side by side:

1. A vibe-coded version generated from a single prompt
2. A hand-crafted version using your team’s style guide and infra contracts

Load test both with `vegeta` or `k6` at 2x your normal traffic. Measure p50, p95, p99 latency, and error rate. If the vibe version is within 20% of the hand-crafted version on all metrics, then it might be safe to use for that service. If not, add a post-processing step: run the AI output through a linter, a formatter, and a connection pool checker before you merge.

Document the gap in your runbook: "Vibe-coded code must pass lint + pool check before staging." That single rule will save you from the most common failures.

Then, next week, pick the service with the highest latency p99 and rewrite it by hand. Measure the difference. You’ll know whether vibe coding belongs in your stack—or only in your prototype folder.

## Frequently Asked Questions

**Is vibe coding safe for internal tools that never go to production?**

Yes. Internal tools that run on your laptop or in a local dev environment are perfect for vibe coding. No users, no SLAs, no scaling. Just remember to delete the code when the tool is done—don’t let it leak into a shared repo.

**Can I use vibe coding for infrastructure as code like Terraform or Kubernetes manifests?**

Yes, but with caution. The AI can generate valid Terraform, but it often misses lifecycle rules, tags, or backend configurations. I’ve seen AI-generated Kubernetes manifests that don’t set resource limits, causing OOM kills in production. Always run `terraform plan` or `kubectl apply --dry-run=server` before merging.

**What’s the fastest way to add AI safety checks to a CI pipeline?**

Add a step that runs `ruff check`, `mypy`, and `sqlfluff lint` on any AI-generated file. If the AI added a comment with "TODO: add error handling", the linter will flag it. That catches 80% of vibe-related issues before staging.

**I’m a startup founder with one engineer. Should I use vibe coding?**

Only if your engineer is comfortable reviewing every line for connection pools, async discipline, and schema drift. If not, hand-craft your critical paths first. A startup can afford a slow launch if it avoids a 3am outage caused by a leaked connection.

## Tooling Cheat Sheet

| Purpose | Tool | Why it helps |
|---|---|---|
| Lint AI-generated Python | ruff==0.3.7 | Catches unused imports, async/await mismatches |
| Lint SQL in Python strings | sqlfluff==3.0.3 | Enforces consistent SQL style |
| Catch connection leaks | py-spy + asyncpg | Profiles connection usage in real time |
| Load test AI vs hand-crafted | vegeta 12.10.0 | Measures p50/p99 latency under traffic |
| AI-aware formatter | black==24.3.0 | Ensures consistent style across generations |
| Sandboxed AI dev container | devcontainers/cli 0.57.0 | Prevents dependency hallucination |
| Observability for AI-generated code | OpenTelemetry Python 1.22.0 | Tracks latency and errors automatically |