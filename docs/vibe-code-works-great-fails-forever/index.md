# Vibe code: works great, fails forever

## Why vibe-coded MVPs fail after the demo

Vibe coding is a real and rational strategy for a demo: one file, one process, hard-coded config, no tests. It optimizes for the only metric that matters before funding — time to something a stakeholder can click. The failure mode is not the speed; it is that the shortcuts are invisible. They do not announce themselves as debt. They sit quietly until the service is asked to do something the demo never did: run two instances, survive a dependency upgrade, produce an audit trail, or let a new engineer change one endpoint without touching the rest.

The ten items below are ranked by how expensive they are to reverse, not by how bad they look in review. Each one includes the concrete production symptom, the fix, and — where a number is useful — how to measure it yourself rather than trusting a borrowed figure.

## 1. Zero observability scaffolding

**Symptom.** Locally, failures print to a console. In production that console is replaced by a log drain, and when the incident happens the only signal available is an unstructured error line with no request context. Reconstructing what happened requires replaying traffic against a local replica.

**Fix.** Emit structured JSON logs from day one and attach a correlation ID to every line. That is roughly twenty lines of middleware, not a platform project.

```python
import json, logging, uuid
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("app")

class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        logger.info(json.dumps({
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
        }))
        return response
```

**How to measure whether it is working.** Pick one week, count the incidents where you could identify the failing request from logs alone without reproducing it. The target is all of them. If the answer is "we had to replay traffic," the logging is not done — regardless of how many dashboards exist.

## 2. Undocumented environment assumptions

**Symptom.** The code assumes Redis is on `localhost:6379` and that the database already has the expected schema and seed rows. When staging is rebuilt from scratch, or a connection string rotates, the app does not crash loudly — it falls back to something else and writes to the wrong place.

**Fix.** Fail fast on missing configuration. A twelve-factor app reads config from the environment and refuses to start without it.

```python
import os

def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value

DATABASE_URL = require_env("DATABASE_URL")
REDIS_URL = require_env("REDIS_URL")
JWT_SECRET = require_env("JWT_SECRET")
```

**Failure-mode analysis.** The dangerous version of this bug is the silent fallback, not the crash. A crash on startup is a five-minute fix. A silent fallback to SQLite corrupts data for however long it takes someone to notice, and the corrupted rows are usually not recoverable from the primary store.

**Checklist.**
- Every external dependency has an explicit, validated connection string.
- Startup fails loudly if any is missing.
- The same code path runs in local, staging, and production — only the values differ.

## 3. No dependency version pins

**Symptom.** The environment resolves to whatever is newest at install time. A minor release changes behavior, the breakage appears only in production, and the local environment still works because it was built months ago.

**Fix.** Pin direct dependencies and commit a lock file. Then automate the upgrade so pins do not become a fossil record.

```toml
# pyproject.toml
[project]
dependencies = [
    "fastapi==0.115.6",
    "uvicorn[standard]==0.34.0",
    "sqlalchemy==2.0.36",
]
```

Generate the lock file with the tool of your choice (`pip-compile`, `uv lock`, `poetry lock`) and commit it. A lock file without an automated upgrade PR is a slow-motion outage; see item 9 for the automation.

**How to measure.** Instrument one thing: the time between a dependency releasing a security patch and that patch being deployed. If you cannot answer that question with a number, you do not have a dependency process.

## 4. In-memory state in a multi-instance deployment

**Symptom.** Sessions or rate-limit counters live in a process-local dict. It works perfectly with one instance. The moment a second instance exists, a fraction of requests fail with 401 or bypass the limit, and the fraction scales with instance count.

**Fix.** Move shared state to a store both instances can see, and make the writes atomic.

```python
import redis.asyncio as redis

r = redis.from_url(REDIS_URL, decode_responses=True)

async def save_session(token: str, user_id: str, ttl_seconds: int = 86400) -> None:
    key = f"session:{token}"
    async with r.pipeline(transaction=True) as pipe:
        pipe.hset(key, mapping={"user_id": user_id})
        pipe.expire(key, ttl_seconds)
        await pipe.execute()
```

**Failure-mode analysis.** The subtle version of this bug is not the 401 — it is the rate limiter. A per-process counter silently allows N times the intended request rate, where N is the instance count. That is an abuse vector, not a UX bug, and it will not show up in your error rate.

## 5. Manual secrets management

**Symptom.** A secret is committed, pasted into a CI variable, or hard-coded "temporarily." When it leaks, the response is a scramble: find every place it was used, rotate, redeploy, and hope nothing was missed.

**Fix.** Secrets come from a secret manager at runtime, never from the repository. The rotation procedure should be a documented, tested runbook — not a script someone writes during the incident.

**Checklist.**
- No secret appears in git history, including in deleted files and CI configs.
- Rotation is a single command or a single pipeline run.
- Rotation does not require a human to remember which services consume the secret.
- A leaked secret can be rotated without a deploy.

**How to measure.** Time a rotation end-to-end in a staging environment. If it takes longer than the incident it is meant to resolve, the process is the vulnerability.

## 6. No structured error handling

**Symptom.** Every failure returns a generic 500 with a string. When someone asks for a breakdown of timeouts versus validation errors versus upstream failures, there is exactly one bucket.

**Fix.** Define error codes and return them consistently.

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

class AppError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        self.code = code
        self.message = message
        self.status = status

@app.exception_handler(AppError)
async def handle_app_error(request: Request, exc: AppError):
    return JSONResponse(
        status_code=exc.status,
        content={"error": {"code": exc.code, "message": exc.message}},
    )

@app.get("/session")
async def get_session():
    raise AppError("AUTH_TIMEOUT", "Session expired", status=401)
```

**Failure-mode analysis.** Error messages are a data-exfiltration surface. A handler that interpolates user input into the message will eventually leak an email address, a token fragment, or a row ID into a log aggregator with broader access than the database. Keep the machine-readable code separate from the human-readable message, and never put user data in either.

## 7. Optimistic concurrency assumptions

**Symptom.** The write path assumes the database always has capacity and that concurrent writes do not conflict. Under a traffic spike, writes time out; under concurrent writes to the same row, one silently overwrites the other.

**Fix.** Make writes idempotent and handle conflict explicitly.

```python
from sqlalchemy.dialects.postgresql import insert

async def upsert_user(session, user_id: str, email: str):
    stmt = insert(User).values(user_id=user_id, email=email)
    stmt = stmt.on_conflict_do_update(
        index_elements=[User.user_id],
        set_={"email": stmt.excluded.email},
    )
    await session.execute(stmt)
    await session.commit()
```

**How to measure.** Run a load test against a staging copy with a realistic write mix and watch database CPU and write latency, not just request latency. The database saturates before the API does, so API-level metrics will look healthy right up until they do not.

## 8. No API contract tests

**Symptom.** The frontend and backend agree on a payload shape in a document that drifts. A field is renamed from `userId` to `user_id`, the backend accepts both or neither, and the failure surfaces as a customer complaint rather than a test failure.

**Fix.** Test the contract, not the implementation. Assert on the response shape your consumers depend on.

```python
from fastapi.testclient import TestClient

client = TestClient(app)

def test_session_response_shape():
    resp = client.post("/session", json={"email": "a@example.com"})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"token", "expires_at"}
    assert isinstance(body["token"], str)
    assert isinstance(body["expires_at"], int)
```

**How to measure.** Count the number of production incidents caused by a payload mismatch in the last quarter. The goal is zero, and the only way to get there is a test that fails when the shape changes.

## 9. Single-file architecture

**Symptom.** Routes, models, middleware, and a background job all live in one file. It grows past the point where anyone can hold it in their head. Adding an endpoint means reasoning about every other endpoint.

**Fix.** Split by responsibility, not by layer count. For a small service, three modules are enough: `routes.py`, `models.py`, `main.py`. The point is not a particular structure; it is that a change to one concern does not require reading the others.

**Failure-mode analysis.** The refactor itself is a risk. A single-file service is often single-threaded by accident, and splitting it can expose a race condition that the original structure hid. Write the contract tests from item 8 *before* the refactor, so the refactor has a safety net.

**How to measure.** Time-to-first-fix: how long a new engineer takes to make a change to one endpoint without breaking another. Track it on the first three changes a new hire makes. If it is measured in days, the file is the problem.

## 10. No request-ID propagation

**Symptom.** No correlation ID flows through the system. When an upstream provider rejects a batch of requests, there is no way to group them, and the investigation becomes a manual grep.

**Fix.** Accept an inbound `x-request-id`, generate one if absent, propagate it to downstream calls, and include it in every log line and error response. The middleware in item 1 is the whole implementation.

**How to measure.** Take a single failed request and trace it from the client through your service to the downstream dependency using only the request ID. If that takes more than a few minutes, the propagation is incomplete.

## Advanced edge cases

### Memory growth under sustained load

A service can pass a demo and still grow unboundedly under sustained traffic. Common causes: background tasks created per request that are never awaited to completion, HTTP client sessions created per call instead of reused, and caches with no eviction. None of these show up in a short test.

**How to detect it.** Record resident memory on a fixed interval and load-test for longer than the intended deploy interval. If memory trends upward across the test and does not return to baseline after traffic stops, there is a leak. A process that restarts on a schedule is hiding the leak, not fixing it.

**How to fix it.** Reuse one HTTP client per process with an explicit timeout, bound every background task with a semaphore, and give every cache an eviction policy. Then re-run the same test and confirm memory returns to baseline.

### Shared-state races between instances

Two instances writing the same session key can interleave such that one overwrites the other's expiry. The symptom is intermittent 401s that correlate with instance count, not with load.

**How to detect it.** Deploy two instances behind a load balancer and run a test that logs in once and then makes authenticated requests in parallel. If a fraction fail, the state is not being written atomically.

**How to fix it.** Use a single atomic operation for the write (a Redis transaction or a Lua script) rather than a sequence of separate commands. Then repeat the test and confirm zero failures.

### Time handling in token expiry

Tokens that compute expiry from a naive local timestamp break around daylight-saving transitions and whenever a server's clock is not UTC. The symptom is a cluster of auth failures in one region at a specific hour, once or twice a year.

**How to detect it.** Add a test that generates a token with a mocked clock set to a DST boundary and asserts the expiry is correct in UTC.

**How to fix it.** Compute all expiry times in UTC, store them as UTC, and compare in UTC. Add a small grace period on the server side so clock skew between machines does not cause spurious expirations. The fix is small; the test is what keeps it fixed.

### Log pipeline backpressure

A log pipeline that drops events under load is worse than no pipeline, because it gives false confidence. The symptom is missing log lines during exactly the incidents you need them for.

**How to detect it.** Compare the number of requests your service reports with the number of log events that arrive in the aggregator over the same window. If the counts diverge, events are being dropped.

**How to fix it.** Use a shipper that buffers to disk and acknowledges delivery, and sample high-volume debug logs rather than shipping all of them. Verify the counts match after the change.

## Tooling categories worth adopting

The specific products matter less than the category. For each category below, the question is whether you have one and whether it is wired to the code.

| Category | What it must do | How to verify it works |
|---|---|---|
| Structured logging | Emit JSON with a correlation ID on every line | Trace one failed request from logs alone |
| Metrics and tracing | Capture request latency percentiles and dependency calls | Compare a load-test percentile against the same percentile in production |
| Dependency automation | Open upgrade PRs with CI on a schedule | Confirm a patch release reaches production without manual action |
| Secret management | Serve secrets at runtime, rotate without a deploy | Time a rotation in staging |
| Contract testing | Fail CI when a response shape changes | Rename a field and confirm the test fails |

## A hardening checklist

Run this against any service that has outgrown its demo. Each item is a yes/no, and the "no" answers are the backlog.

- [ ] Structured logs with a request ID on every line
- [ ] Startup fails if required configuration is missing
- [ ] Dependencies pinned, with a lock file committed
- [ ] No shared state in process memory
- [ ] Secrets loaded at runtime, rotation runbook tested
- [ ] Error responses use machine-readable codes
- [ ] Writes are idempotent and handle conflict
- [ ] Contract tests run in CI
- [ ] No file exceeds the size a new engineer can read in one sitting
- [ ] Request IDs propagate to downstream calls
- [ ] Memory is stable under a load test longer than the deploy interval
- [ ] Log event counts match request counts

## What to do in the next 30 minutes

Open the service that has outgrown its demo and add request-ID middleware — the twenty-line snippet in item 1. Deploy it to staging, make one request, and confirm the ID appears in both the response header and the log line. That single change turns every future incident from an archaeology project into a lookup, and it is the cheapest item on the checklist to get right.
