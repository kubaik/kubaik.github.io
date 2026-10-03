# AI interviews now ask for this

## What actually changed in AI-assisted engineering interviews

The shift in technical interviews is not that candidates are asked to write code with an AI assistant. It is that they are asked to debug, audit, and critique code that an AI assistant produced. The distinction matters. Generating code is cheap; understanding why generated code fails under load, leaks data, or passes tests that should have failed is the skill employers are now filtering for.

The reason is structural, not fashionable. AI code generators are trained to produce plausible code, and plausible code is usually code that works on the happy path. The failure modes that matter in production — ordering assumptions, shared mutable state, unbounded logging, over-broad exception handling, wildcard permissions — are exactly the ones a model has no strong signal to avoid. So teams that ship AI-assisted code accumulate a specific class of defects, and interviewers have started testing for the ability to find them.

This article covers the four areas where this shows up: test reliability, observability cost, secrets and credential rotation, and auditing generated code itself. Each section describes the failure mode, how to reproduce it, how to measure it, and what a defensible fix looks like.

## Flaky tests and the reproducibility problem

A flaky test is one that passes or fails without any change to the code under test. The most common mechanical cause in Python projects is test-order dependence: a test that assumes it runs first, or assumes a database starts empty, or asserts on an auto-incrementing identifier.

Two separate things get conflated here, and interviewers often probe the distinction:

1. **Nondeterministic ordering.** Plugins such as `pytest-random-order` shuffle test execution to surface hidden inter-test dependencies. If a suite only passes in one order, that is a real defect the shuffle has exposed.
2. **Nondeterministic state.** Even with a fixed order, a test that asserts `response.json()["id"] == 1` is asserting on state that depends on how many rows exist. That assertion is wrong regardless of ordering.

The fix for ordering is to pin a seed so runs are reproducible while still exercising shuffle:

```ini
# pytest.ini
[pytest]
# pytest-random-order reads this option; pin it so CI failures are reproducible.
random_order_seed = 42
```

The fix for state dependence is to stop asserting on values the test does not control. Assert on the shape and constraints of the response instead:

```python
def test_create_user(client):
    response = client.post(
        "/users", json={"name": "Alice", "email": "alice@example.com"}
    )
    assert response.status_code == 201
    body = response.json()
    assert isinstance(body["id"], int)
    assert body["name"] == "Alice"
```

### Reproducing and measuring flakiness

A single failing run tells you nothing useful. What you want is a failure rate over many runs, plus the seed that produces the failure.

```bash
# Run the suite 50 times with different seeds, stop on first failure, keep the seed.
for i in $(seq 1 50); do
  pytest -p no:randomly -q --random-order-seed=$i || { echo "failed at seed $i"; break; }
done
```

If the suite uses `pytest-random-order`, the seed is reported in the failure output; capture it and re-run with that exact seed to get a deterministic reproduction. Instrument by logging the seed and the test order to CI output — without that, a failure at 3 a.m. is unreproducible.

A useful metric to track is **failures per thousand CI runs** for the suite as a whole, segmented by whether a fixed seed reproduces the failure. A failure that reproduces under a fixed seed is a code or test defect. A failure that does not reproduce under any seed is usually an infrastructure or timing issue and belongs in a different bucket.

### The database-state fixture

If tests share a database, the fixture must put it in a known state and tear it down. Using a file-based SQLite database and deleting it before and after the module is one approach:

```python
import os
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from main import Base

@pytest.fixture(scope="module", autouse=True)
def setup_db():
    db_path = "./test.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(bind=engine)
    yield
    engine.dispose()
    if os.path.exists(db_path):
        os.remove(db_path)
```

Two caveats worth stating plainly. First, SQLite in-memory databases are per-connection; if the application opens a new connection per request, each request sees an empty database. That alone produces the "id is 1, then id is 2" class of failure. Second, `scope="module"` means the fixture runs once per module, not once per test — tests within the module still share state. Use `scope="function"` if isolation matters more than speed.

## Observability overhead and high-cardinality metrics

Instrumentation is not free. Every span, log line, and metric emission consumes CPU, allocation, and network. The failure mode that shows up in interviews is a candidate who instruments everything, then cannot explain why p99 latency rose after the change.

The specific trap is **high-cardinality labels**. A metric labelled by user ID, request ID, or full URL path will create one time series per distinct value. Most metrics backends charge per active time series and degrade in query performance as cardinality grows. A metric labelled by `http.route` (bounded) is fine; the same metric labelled by `http.target` (unbounded) is not.

### How to measure instrumentation overhead

You cannot reason your way to the overhead number; you have to measure it. The procedure:

1. Run a fixed load profile against the service with instrumentation disabled. Record p50, p95, p99 latency and CPU utilization.
2. Enable instrumentation with the same load profile. Record the same numbers.
3. Subtract. The difference is the overhead, expressed in milliseconds at each percentile.

For a local profile of a Python service, `py-spy` samples the running process without code changes:

```bash
py-spy top --pid <pid> --duration 10
```

For a Node.js service, use the built-in profiler or `--cpu-prof`. For any service, the cleanest comparison is the same load generator against two builds that differ only in whether the instrumentation SDK is initialized.

The documented trade-off is that distributed tracing typically adds low single-digit milliseconds per request when sampling is enabled and the exporter batches asynchronously. If the exporter is synchronous or sampling is at 100% under high throughput, the overhead can be much larger. The instrumentation itself is rarely the dominant cost; the export path usually is.

### Reducing cardinality

The fix is to constrain label values before they reach the metric backend:

```python
from opentelemetry import metrics

meter = metrics.get_meter(__name__)

# Bounded label: the route template, not the resolved path.
request_counter = meter.create_counter(
    "http.server.requests",
    description="Count of HTTP requests by route and status.",
)

def record_request(route_template: str, status_code: int) -> None:
    # route_template is e.g. "/users/{user_id}", never "/users/12345".
    request_counter.add(1, {"http.route": route_template, "http.status_code": status_code})
```

The rule: any label whose number of distinct values grows with traffic must not be a metric label. Put it in a trace attribute or a log field instead, where the backend is designed for high cardinality.

## Secrets rotation as a distributed systems problem

Rotating a credential without dropping traffic is a coordination problem, not a configuration change. The interview question — "how would you rotate a secret without a gap in availability" — is testing whether the candidate understands that two systems (the secret store and the application) hold the credential at different times.

The failure mode is straightforward. A rotation process updates the credential in the secret store and in the backing service. Between those two writes, one side has the new value and the other has the old. Any request that reads the secret in that window fails authentication.

### The dual-write pattern

The standard mitigation is to accept both old and new credentials for a grace period:

1. Generate the new credential. Store it alongside the old one, marked as the new active value but with the old value still accepted.
2. Update the backing service (database, third-party API) to accept both.
3. Roll the application to read the new value. Because the old value is still accepted, instances that have not yet picked up the new value continue to work.
4. After the grace period — long enough for every running instance to have restarted or refreshed — revoke the old value.

The grace period must exceed the maximum time any instance can hold a stale secret. If secrets are cached for 15 minutes and instances restart on a rolling schedule over 10 minutes, a 30-minute grace period is a reasonable starting point. That is arithmetic, not a magic number: `grace_period > cache_ttl + rollout_duration + clock_skew_margin`.

### Handling rotation failures

Rotation lambdas and scripts fail. The failure mode that matters is a partial rotation: the new secret is written to the store but the backing service still expects the old one, or vice versa. The application must not crash on a failed refresh — it should keep using the last known-good value and retry.

```python
import time
import logging

logger = logging.getLogger(__name__)

def refresh_secret_with_retry(fetch, current, max_attempts=5, base_delay=1.0):
    """Fetch a new secret, falling back to the previous value on failure."""
    for attempt in range(max_attempts):
        try:
            return fetch()
        except Exception as exc:
            delay = base_delay * (2 ** attempt)
            logger.warning(
                "secret refresh attempt %d failed: %s; retrying in %.1fs",
                attempt + 1, exc, delay,
            )
            time.sleep(delay)
    logger.error("secret refresh exhausted retries; keeping previous value")
    return current
```

The important property is the return of `current` on total failure. A rotation system that raises on failure and takes the service down is worse than one that keeps serving with a soon-to-expire credential.

## Auditing AI-generated code: a checklist

The interview exercise is usually: here is a generated function or module, find what is wrong. The following checklist covers the categories that generated code most often gets wrong. Each item is a question to ask of the code, not a rule to apply blindly.

### Correctness under concurrency

- Does the code hold a lock or transaction across an `await` or a network call? If so, it can hold the lock far longer than intended.
- Are there two `commit()` calls in one logical operation? Nested or repeated commits are a common source of partial writes and, under load, deadlocks.
- Does the code read a value, then write it back based on the read (read-modify-write)? Without a transaction or a compare-and-swap, that is a lost-update race.

```python
# Generated code with a repeated commit inside one logical operation.
def create_user(db, user_data):
    user = User(**user_data)
    db.add(user)
    db.commit()          # first commit: user row exists
    # ... more operations that may fail ...
    db.commit()          # second commit: partial state if the first succeeded
```

The fix is to make the operation atomic: one transaction, one commit, roll back on failure.

### Error handling

- Is there a bare `except:` or `except Exception:` that swallows the error and returns a success-shaped response? That masks failures and makes incidents invisible.
- Does the handler log the exception with enough context to diagnose it, or does it log only that something went wrong?

```python
from fastapi import HTTPException

@app.post("/process")
def process(payload: dict):
    try:
        result = do_work(payload)
        return {"status": "ok", "result": result}
    except ValueError as exc:
        # Expected, client-caused failure.
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        # Unexpected: log with context, return a generic message.
        logger.exception("unexpected failure processing payload")
        raise HTTPException(status_code=500, detail="internal error")
```

### Logging and data exposure

- Does the code log an entire request or event object? Request objects routinely contain credentials, tokens, and personal data.
- Are there fields that should be redacted before any log write? Redaction must happen at the logging boundary, not be assumed to happen downstream.

```javascript
const winston = require('winston');

const logger = winston.createLogger({
  format: winston.format.combine(
    winston.format.json()
  ),
  transports: [new winston.transports.Console()],
});

// Redact before logging, not after.
function safeLog(event) {
  const { cardNumber, cvv, ...rest } = event;
  logger.info(rest);
}
```

### Permissions and infrastructure

- Does an IAM policy, service account, or role use a wildcard action or resource? Generated infrastructure code frequently does, because it is the shortest path to "it works."
- Can the permission be scoped to the specific resource and the specific actions the code performs?

```yaml
Policies:
  - Effect: Allow
    Action:
      - 'dynamodb:GetItem'
      - 'dynamodb:PutItem'
    Resource: !GetAtt MyTable.Arn
```

### Dependencies

- Does the generated code add a dependency? If so, is it pinned to a range, and has it been checked for known vulnerabilities?

```bash
pip-audit --desc
```

Dependency auditing is a routine CI step, not a one-time review. The point of the checklist item is that generated code introduces dependencies silently, and an unpinned or unvetted dependency is a supply-chain risk.

### Performance assumptions

- Does the code assume the dataset fits in memory? Generated code often does, because the training examples did.
- Does the code make a network call inside a loop? That turns an O(n) operation into O(n) round trips.
- Does the code allocate a buffer sized by user input? That is a denial-of-service vector.

## A worked example: auditing one generated function

Consider this generated endpoint, of the kind a code assistant produces for a FastAPI service:

```python
from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker

app = FastAPI()
engine = create_engine("sqlite:///./app.db")
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String, unique=True)

Base.metadata.create_all(bind=engine)

class UserCreate(BaseModel):
    name: str
    email: str

@app.post("/users")
def create_user(user: UserCreate):
    db = SessionLocal()
    db_user = User(name=user.name, email=user.email)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user
```

Walking the checklist:

**Correctness.** The session is never closed. Under sustained load this exhausts the connection pool. The fix is a dependency that yields a session and closes it, or a context manager. The `unique=True` constraint on `email` means a duplicate insert raises `IntegrityError`, which is not handled — so the endpoint returns a 500 for a client error that should be a 409.

**Error handling.** No handling at all. A duplicate email produces an unhandled exception.

**Logging.** No logging, so failures are invisible except in the framework's default error output.

**Performance.** `create_engine` with SQLite is synchronous and, for a write-heavy endpoint, serializes on the database file. That is fine for a demo and wrong for production, but the more transferable point is that the generated code chose the simplest database configuration without any comment acknowledging the trade-off.

**Dependencies.** SQLAlchemy is a real dependency with a version range; the generated code does not pin one. Whether the installed version has a known vulnerability is a question for `pip-audit`, not for inspection.

The rewritten version addresses the concrete defects:

```python
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import declarative_base, sessionmaker, Session

app = FastAPI()
engine = create_engine("sqlite:///./app.db")
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String, unique=True)

Base.metadata.create_all(bind=engine)

class UserCreate(BaseModel):
    name: str
    email: str

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/users", status_code=201)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = User(name=user.name, email=user.email)
    db.add(db_user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="email already registered")
    db.refresh(db_user)
    return {"id": db_user.id, "name": db_user.name, "email": db_user.email}
```

The changes are: session lifecycle managed by a dependency, duplicate-email handled as a client error, explicit 201 status, and a response shape the test can assert on without depending on a specific database-assigned value.

## Decision checklist before shipping generated code

Use this as a review gate. Each question has a concrete artifact that answers it.

| Question | Artifact that answers it |
|---|---|
| Do the tests pass in a shuffled order, repeatedly? | CI job running the suite N times with different seeds |
| Are metric labels bounded? | Metric definitions with label allow-lists |
| Is instrumentation overhead measured, not assumed? | Latency comparison with SDK on and off |
| Can secrets rotate without a gap? | Dual-write documented with a grace period longer than cache TTL + rollout |
| Is every failure path logged with context? | Code review of exception handlers |
| Are permissions scoped to specific actions and resources? | IAM policy diff against a wildcard baseline |
| Are new dependencies pinned and audited? | `pip-audit` (or equivalent) output in CI |
| Is there a test that fails if the happy-path assumption breaks? | At least one negative test per endpoint |

## FAQ

**Does this mean candidates should refuse to use AI assistants?**
No. The interview tests whether the candidate can evaluate generated output critically. Using the assistant is expected; trusting it without review is what the questions are designed to catch.

**Are flaky tests always the test's fault?**
No. Flakiness can come from the test, the code under test, or the environment. The first step is always to determine which by reproducing with a fixed seed and fixed environment. Only then does the fix become obvious.

**How much observability overhead is acceptable?**
There is no universal number. The right approach is to measure it for the specific service and decide whether the diagnostic value exceeds the cost. What is not acceptable is not knowing the number.

**Is dual-write the only way to rotate secrets safely?**
It is the most general. Some systems support overlapping credential validity natively; others use a short-lived token exchange. The underlying requirement is always the same: at no point may there be a window where one side expects a value the other side has stopped sending.

**Should every generated function be rewritten?**
No. The point of the checklist is to find the small fraction of generated code that carries a real failure mode. Most generated code is fine. The skill is in identifying the code that is not.

## One thing to do in the next 30 minutes

Pick one test suite in a repository you have access to and run it 20 times with different random-order seeds, capturing the seed on each failure:

```bash
for i in $(seq 1 20); do
  pytest -q --random-order-seed=$i || echo "FAILED at seed $i"
done
```

If any run fails, you have found a real defect that a single run would have hidden — and you now have the exact seed to reproduce it.
