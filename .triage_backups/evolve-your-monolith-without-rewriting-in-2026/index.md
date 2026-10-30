# Evolve your monolith without rewriting in 2026

A monolith does not become a problem because it is a monolith. It becomes a problem when its structure makes every change expensive. The usual failure is not the technology choice; it is that the original shortcut — one file, one session, one import graph — hardens into something nobody wants to touch.

This article describes a small, boring structure for a Python service that keeps the option to change open. The example is a FastAPI backend, but the layering applies to any web framework. The code is deliberately small so the shape is visible. Everything below is reproducible on a laptop.

## The failure mode: a one-file backend

The common starting point looks like this: a single Python module with a few hundred lines of synchronous routes, SQLAlchemy models declared next to handlers, and a `main.py` that also owns configuration, connection setup, and a couple of background tasks. It works. It ships. Then it accumulates.

Symptoms that show up in this shape, in roughly the order teams notice them:

- **Latency spikes after deploy.** A single shared ORM session or connection is reused across concurrent requests. Under load, requests serialize on it. A p99 that was 120 ms can jump into the seconds without any code change, because the traffic pattern changed, not the code.
- **Slow releases.** The module is imported by everything, so every change touches everything. Test suites grow to cover unrelated paths, and a redeploy blocks other work.
- **Unreadable to non-engineers.** When a non-technical stakeholder asks how the product works, the only artifact is a file that mixes HTTP, business rules, and SQL.
- **Hidden coupling.** Cache keys, database sessions, and request context are module-level globals. The dependency graph is implicit and only discoverable by reading the whole file.

None of these are caused by FastAPI, SQLAlchemy, or Python. They are caused by the absence of a boundary between "how a request arrives", "what the business rule is", and "where the bytes live".

## The three layers

The structure below separates three concerns and nothing else.

1. **API surface** — HTTP routing, request/response shapes, status codes. Framework-specific.
2. **Domain** — plain Python functions and objects. No framework imports, no database imports. This is the part that describes what the product does.
3. **Infrastructure** — database engines, cache clients, queues, external HTTP calls. Framework-agnostic but implementation-specific.

The rule that makes this work: dependencies point inward. The API layer imports the domain. The domain imports nothing from the API or the infrastructure. Infrastructure is passed into the domain or called through an interface the domain owns.

Target layout:

```
myapp/
├── app/
│   ├── api/
│   │   └── v1/
│   │       └── users.py
│   ├── domain/
│   │   ├── errors.py
│   │   └── user_service.py
│   ├── infra/
│   │   ├── cache.py
│   │   └── database.py
├── tests/
│   ├── integration/
│   └── unit/
└── main.py
```

Roughly 250–300 lines of production code for a small service. The point is not the line count; it is that each file has one reason to change.

A common mistake is to split too late, after the codebase is already large. At that point the split becomes a migration project with a long-lived branch. Splitting at a few hundred lines is cheap and reversible.

## Step 1 — environment and smoke test

Pin versions explicitly so the example is reproducible. Exact pins matter less than the habit of pinning; substitute current versions of the same libraries if you prefer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install "fastapi" "uvicorn[standard]" "sqlalchemy[asyncio]" "asyncpg" "redis" "pytest" "pytest-asyncio" "mypy" "ruff"
```

For local development, a Redis container is convenient:

```bash
docker run -d --name redis-dev -p 6379:6379 redis:7-alpine
```

Minimal application:

```python
# main.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"status": "ok"}
```

```bash
uvicorn main:app --reload
curl http://localhost:8000/
# {"status":"ok"}
```

If you see `RuntimeError: no running event loop` in a test runner or CI, the usual causes are (a) a module-level `asyncio.get_event_loop()` call, or (b) an async fixture that is not marked as async. The fix is to use `pytest-asyncio` and mark the test, not to disable the lifespan.

## Step 2 — domain first, framework later

Write the domain layer before the routes. It has no FastAPI import.

```python
# app/domain/user_service.py
from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class UserDTO(BaseModel):
    id: int
    email: str
    created_at: datetime


class CreateUserRequest(BaseModel):
    email: str


class UserService:
    def __init__(self, repo: "UserRepository") -> None:
        self._repo = repo

    async def create_user(self, email: str) -> UserDTO:
        existing = await self._repo.find_by_email(email)
        if existing is not None:
            raise UserAlreadyExistsError(email)
        return await self._repo.insert(email)

    async def get_user(self, user_id: int) -> Optional[UserDTO]:
        return await self._repo.find_by_id(user_id)
```

`UserRepository` is a protocol or an abstract base class owned by the domain layer. The domain never imports SQLAlchemy. The repository is injected, which is what makes the domain testable without a database.

```python
# app/domain/errors.py
class DomainError(Exception):
    """Base class for errors the domain knows how to describe."""


class UserAlreadyExistsError(DomainError):
    def __init__(self, email: str) -> None:
        super().__init__(f"User with email {email!r} already exists")
        self.email = email
```

The domain layer here is small. Its value is the interface it exposes: `create_user`, `get_user`, and a small set of named exceptions. That interface is what the API layer depends on, and it is what you can move to another framework or another transport later.

## Step 3 — infrastructure behind the interface

The infrastructure layer implements the repository interface using a real database, and provides a cache client.

```python
# app/infra/database.py
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

DATABASE_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/dev"

engine = create_async_engine(DATABASE_URL, pool_size=5, max_overflow=0)
SessionLocal = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
```

Two notes on the defaults above:

- `pool_size=5, max_overflow=0` caps concurrent database connections at five. That is a deliberate ceiling, not a recommendation. The correct value depends on the database's own connection limit and on how many application instances are running. Setting `max_overflow` to zero means a request waits for a free connection instead of opening a new one; that converts a database overload into queueing latency, which is usually the failure you want.
- `expire_on_commit=False` avoids an implicit refresh after commit, which would otherwise trigger an extra round trip and can fail after the session is closed.

```python
# app/infra/cache.py
import redis.asyncio as redis

cache = redis.Redis(host="localhost", port=6379, decode_responses=True)
```

The repository implementation lives in the infrastructure layer and is the only place that knows SQLAlchemy exists:

```python
# app/infra/user_repository.py
from datetime import datetime, timezone

from sqlalchemy import select

from app.domain.user_service import UserDTO
from app.infra.database import SessionLocal
from app.infra.models import User


class SqlUserRepository:
    async def find_by_email(self, email: str) -> UserDTO | None:
        async with SessionLocal() as session:
            row = await session.scalar(select(User).where(User.email == email))
            return _to_dto(row) if row else None

    async def find_by_id(self, user_id: int) -> UserDTO | None:
        async with SessionLocal() as session:
            row = await session.get(User, user_id)
            return _to_dto(row) if row else None

    async def insert(self, email: str) -> UserDTO:
        async with SessionLocal() as session:
            row = User(email=email, created_at=datetime.now(timezone.utc))
            session.add(row)
            await session.commit()
            await session.refresh(row)
            return _to_dto(row)


def _to_dto(row: User) -> UserDTO:
    return UserDTO(id=row.id, email=row.email, created_at=row.created_at)
```

One session per operation. This is the single most important detail for the latency-spike failure mode described earlier: a session is not safe to share across concurrent tasks, and a request-scoped session is the simplest correct unit.

## Step 4 — the API layer

The routes translate HTTP into domain calls and domain exceptions into status codes. Nothing else.

```python
# app/api/v1/users.py
from fastapi import APIRouter, Depends, HTTPException

from app.domain.errors import DomainError, UserAlreadyExistsError
from app.domain.user_service import CreateUserRequest, UserService
from app.infra.user_repository import SqlUserRepository

router = APIRouter(prefix="/v1/users", tags=["users"])


def get_user_service() -> UserService:
    return UserService(repo=SqlUserRepository())


@router.post("/")
async def create_user(
    payload: CreateUserRequest,
    svc: UserService = Depends(get_user_service),
):
    try:
        user = await svc.create_user(payload.email)
    except UserAlreadyExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except DomainError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"id": user.id, "email": user.email}


@router.get("/{user_id}")
async def get_user(
    user_id: int,
    svc: UserService = Depends(get_user_service),
):
    user = await svc.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return {"id": user.id, "email": user.email}
```

```python
# main.py
from fastapi import FastAPI

from app.api.v1 import users

app = FastAPI()
app.include_router(users.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
```

The API layer now contains no SQL, no cache keys, and no business rules. Changing the database touches `app/infra/`, not the routes. Adding a new transport — a CLI, a worker, a WebSocket handler — means calling the same `UserService` from a new entry point.

## Failure mode: cache stampede

Caching `get_user` by id is a natural next step. The naive version has a well-known failure: when a popular key expires, every concurrent request misses the cache and hits the database at once. If the database call takes 200 ms and 50 requests arrive in that window, the database sees 50 identical queries.

The fix is a short-lived lock around the miss path, with a re-check inside the lock:

```python
# app/infra/cache.py
import asyncio
from contextlib import asynccontextmanager

import redis.asyncio as redis

cache = redis.Redis(host="localhost", port=6379, decode_responses=True)


@asynccontextmanager
async def cache_lock(key: str, ttl_seconds: int = 10, wait_seconds: float = 2.0):
    lock_key = f"lock:{key}"
    acquired = await cache.set(lock_key, "1", ex=ttl_seconds, nx=True)
    if acquired:
        try:
            yield True
        finally:
            await cache.delete(lock_key)
        return

    # Someone else holds the lock. Wait briefly, then proceed without it.
    deadline = asyncio.get_running_loop().time() + wait_seconds
    while asyncio.get_running_loop().time() < deadline:
        if not await cache.exists(lock_key):
            break
        await asyncio.sleep(0.02)
    yield False
```

Used in the service:

```python
async def get_user(self, user_id: int) -> UserDTO | None:
    cache_key = f"user:{user_id}"
    cached = await cache.get(cache_key)
    if cached is not None:
        return UserDTO.model_validate_json(cached)

    async with cache_lock(cache_key) as acquired:
        if acquired:
            cached = await cache.get(cache_key)
            if cached is not None:
                return UserDTO.model_validate_json(cached)
            user = await self._repo.find_by_id(user_id)
            if user is not None:
                await cache.set(cache_key, user.model_dump_json(), ex=300)
            return user

    # Lock was not acquired and the holder did not populate the cache in time.
    return await self._repo.find_by_id(user_id)
```

Two properties of this pattern are worth being explicit about:

- **The lock is best-effort.** If it cannot be acquired within `wait_seconds`, the request falls through to the database rather than failing. A stampede of a few requests is acceptable; a request that errors because a lock is held is not.
- **The lock has a TTL.** If the holder crashes, the lock expires. Without a TTL, a single failure would block the key indefinitely.

The trade-off is that cold-cache requests are slower by the lock round trip, and requests that lose the lock race may still hit the database. Both are usually acceptable compared to a thundering herd.

## How to measure whether this helps

Do not trust latency numbers from an article, including this one. Measure your own. The instrumentation is small:

1. **Instrument the request path.** Add a middleware that records the wall-clock duration of each request and emits it to your metrics backend as a histogram, labelled by route and status code.
2. **Instrument the database call.** Time each repository method and emit a separate histogram. Without this, you cannot tell whether a slow request is slow because of the database, the cache, or the framework.
3. **Generate load.** Use a load generator that can hold a fixed number of concurrent connections — `hey`, `wrk`, or `locust` all work. Run at a concurrency level that is realistic for your service, not the maximum your laptop can produce.
4. **Compare configurations, not absolutes.** Run the same load against the one-file version and the layered version, or against the cache with and without the lock, on the same machine. Report the delta.

A useful measurement recipe, stated generically:

- Fix the request mix (for example, 90% reads of an existing id, 10% creates).
- Warm the cache, then measure steady-state p50/p95/p99.
- Expire the cache, then measure the same percentiles during the cold window. This is where the stampede shows up.
- Repeat each run at least three times and report the spread. A single run on a shared machine is noise.

The specific numbers depend on your hardware, your database, and your network. What is stable across environments is the shape: a shared session produces p99 spikes under concurrency, and a lock around the cache miss converts a spike into a slightly higher median.

## Testing the layers

Unit tests exercise the domain with a fake repository. No database, no network, no fixtures that need cleanup.

```python
# tests/unit/test_user_service.py
import pytest

from app.domain.errors import UserAlreadyExistsError
from app.domain.user_service import UserDTO, UserService


class FakeRepo:
    def __init__(self, existing: dict[str, UserDTO] | None = None) -> None:
        self._by_email = existing or {}
        self._next_id = 100

    async def find_by_email(self, email: str) -> UserDTO | None:
        return self._by_email.get(email)

    async def find_by_id(self, user_id: int) -> UserDTO | None:
        return next((u for u in self._by_email.values() if u.id == user_id), None)

    async def insert(self, email: str) -> UserDTO:
        from datetime import datetime, timezone

        user = UserDTO(id=self._next_id, email=email, created_at=datetime.now(timezone.utc))
        self._next_id += 1
        self._by_email[email] = user
        return user


@pytest.mark.asyncio
async def test_create_user():
    svc = UserService(repo=FakeRepo())
    user = await svc.create_user("new@example.com")
    assert user.email == "new@example.com"


@pytest.mark.asyncio
async def test_create_duplicate_raises():
    from datetime import datetime, timezone

    existing = UserDTO(id=1, email="dup@example.com", created_at=datetime.now(timezone.utc))
    svc = UserService(repo=FakeRepo({"dup@example.com": existing}))
    with pytest.raises(UserAlreadyExistsError):
        await svc.create_user("dup@example.com")
```

Integration tests exercise the real database and cache. Keep them few and focused on the wiring, not the business rules.

```python
# tests/integration/test_users_api.py
import pytest
from httpx import ASGITransport, AsyncClient

from main import app


@pytest.mark.asyncio
async def test_create_and_get_user():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/v1/users/", json={"email": "alice@example.com"})
        assert resp.status_code in (200, 409)

        resp = await client.get("/v1/users/1")
        assert resp.status_code in (200, 404)
```

The assertions are loose on purpose: integration tests that assert on exact ids or exact status codes become brittle when the database is shared. Assert on the contract (the response shape, the error code for a known conflict) and leave the specifics to unit tests.

## Decision checklist

Use this when deciding whether to split a module.

- **Does the file mix HTTP concerns with SQL?** If yes, extract the repository first. That is the highest-value split and the least disruptive.
- **Is there a module-level session, client, or cache object?** If yes, replace it with an injected dependency. This is the cause of most concurrency bugs in this shape.
- **Can you describe what the service does without naming a framework or a database?** If not, the domain layer is missing.
- **Would a new transport (a worker, a CLI) be able to reuse the logic?** If yes, the layering is working.
- **Is the split reversible?** It should be. If merging the layers back would require rewriting tests, the boundary is in the wrong place.
- **Are there more than three layers?** Probably too many. API, domain, infra covers most small services. Add a layer only when a real second consumer exists.

## A comparison worth making

| Concern | Layered (this article) | Single module |
|---|---|---|
| Adding a new endpoint | New route file, reuses domain | Edit the shared file |
| Changing the database | Edit `app/infra/` | Edit routes and models together |
| Unit testing business rules | Fake repository, no I/O | Requires a database or heavy mocking |
| Concurrency safety | Session per operation | Depends on reviewer discipline |
| Onboarding a new contributor | Read three small files | Read one large file |
| Cost of reversal | Low — files can be merged | Low initially, high after growth |

The table is qualitative on purpose. Latency and cost figures depend entirely on your workload and infrastructure, and any specific number quoted without your measurements is not evidence.

## FAQ

**Do I need a message queue to do this?**
No. A queue is an infrastructure detail. If you add one, put the client in `app/infra/` and expose a small interface the domain calls. The domain should not import the queue library.

**How do I handle database migrations?**
Use a migration tool that runs outside the application process, such as Alembic. Point its environment at the same database URL the application uses, and run migrations as a separate deployment step. Do not run migrations on application startup; with more than one instance, they will race.

**What about WebSockets or background tasks?**
Add a new entry point under `app/api/` or a new worker module that calls the same domain services. The domain and infrastructure layers do not change. That is the test of whether the boundary is real.

**When should I stop splitting?**
When each layer has a single reason to change and you can name it. If you cannot articulate why a file exists separately, merge it back.

## Do this next

Add a `/health` endpoint that actually checks the database and the cache, and run it before you write any other code:

```python
# main.py
from fastapi import FastAPI
from sqlalchemy import text

from app.infra.cache import cache
from app.infra.database import SessionLocal

app = FastAPI()


@app.get("/health")
async def health():
    db_ok = False
    try:
        async with SessionLocal() as session:
            await session.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False

    try:
        redis_ok = bool(await cache.ping())
    except Exception:
        redis_ok = False

    status = "ok" if (db_ok and redis_ok) else "degraded"
    return {"status": status, "db": db_ok, "redis": redis_ok}
```

Then run `curl http://localhost:8000/health` and fix any connection errors immediately. A health check that only returns a static string will not tell you that a security group is blocking the cache port; this one will, and it takes about ten minutes to write.
