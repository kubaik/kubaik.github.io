# Retire API versions before they pile up

Version numbers are often treated as an insurance policy: ship `/v1/`, later ship `/v2/` for a breaking change, and let `/v1/` "stay around for compatibility." A year later the surface area includes `/v1/`, `/v2/`, and a half-finished `/v3/` that only two internal services call. The graveyard is rarely caused by versioning itself. It is caused by versioning without a deletion plan. A version with no scheduled death is not a compatibility strategy; it is a backlog item that never gets picked up.

This article covers versioning strategies that include an exit. It is not a general REST design guide. The focus is on the mechanics that make old versions cheap to retire, and the specific points where teams accidentally make retirement impossible.

By the end there is a working FastAPI example showing a version negotiation pattern, a deprecation header scheme, and a migration checklist that can be run against an existing API in an afternoon.

## The three versioning strategies, and which one survives

Three mainstream approaches exist. They are not equivalent in how much they cost later.

| Strategy | Where the version lives | Cost to add a version | Cost to remove a version |
|---|---|---|---|
| URI versioning | `/v1/resource` | Low | High — every client URL changes |
| Header versioning | `Accept: application/vnd.api+json;version=1` | Medium | Medium — clients must send a new header |
| Date-based versioning | `Stripe-Version: 2024-06-20` | High | Low — behavior can be pinned per request |

URI versioning is the default choice because it is visible and cacheable. The trap is that it makes the version part of the address rather than part of the contract. Retiring `/v1/` means convincing every consumer to change a URL, which requires coordinating with clients the team may not control. That is where graveyards come from.

Header versioning keeps the URL stable, so routing by header at the gateway does not require touching client URLs. The cost is that clients which do not set the header need a default, and that default becomes its own implicit version that must be supported.

Date-based versioning is the most expensive to implement and the easiest to retire. Each request declares the API behavior it expects, so old behavior can stay alive for old dates while new clients get new behavior on the same URL. The implementation cost is real: every response shape change needs to be conditional on the request's declared date.

If a team is starting fresh and controls all clients, URI versioning plus a strict deprecation policy is fine. If there are external clients that cannot be forced to upgrade, date-based versioning is the only one that lets the API move forward without a negotiation.

## Step 1 — Set up the environment

Python 3.10+ and FastAPI are required. The example uses only standard library and FastAPI dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install "fastapi[standard]"
```

Create a file `main.py`. The pattern below uses a single application with a version resolver that reads from either the URL path or a header, so migration can happen incrementally.

## Step 2 — Core implementation: version resolution as a dependency

A common mistake is scattering version checks through route handlers. If `if version == 1:` appears in a handler, the version logic is now coupled to business logic and cannot be removed without touching every endpoint.

Instead, resolve the version once, in a dependency, and pass it to handlers as a value. Handlers decide behavior based on the value, not on parsing.

```python
from fastapi import FastAPI, Header, HTTPException, Depends, Request
from typing import Annotated

app = FastAPI()

SUPPORTED_VERSIONS = {"1", "2"}
DEFAULT_VERSION = "2"
DEPRECATED_VERSIONS = {"1"}

async def resolve_version(
    request: Request,
    x_api_version: Annotated[str | None, Header()] = None,
) -> str:
    # Prefer explicit header, fall back to path segment, then default.
    explicit = x_api_version
    if explicit is None:
        path_parts = request.url.path.strip("/").split("/")
        if path_parts and path_parts[0].startswith("v") and path_parts[0][1:].isdigit():
            explicit = path_parts[0][1:]
    version = explicit or DEFAULT_VERSION
    if version not in SUPPORTED_VERSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported API version: {version}",
        )
    return version

Version = Annotated[str, Depends(resolve_version)]
```

Two things to notice. First, the resolver is the single place that knows how to extract a version. Second, unsupported versions fail loudly at the edge, not deep inside a handler.

Now a handler that behaves differently by version:

```python
@app.get("/v{version}/users/{user_id}")
async def get_user(version: Version, user_id: int):
    user = {"id": user_id, "name": "Ada", "email": "ada@example.com"}
    if version == "1":
        # v1 returned a flat `email` field. v2 nests it under contact.
        return user
    return {
        "id": user["id"],
        "name": user["name"],
        "contact": {"email": user["email"]},
    }
```

The handler's version branch is small and local. The version string came from the dependency, so removing v1 later means deleting one branch and one entry in `SUPPORTED_VERSIONS`.

## Step 3 — Make deprecation visible and machine-readable

A version that is intended to be removed must announce itself. The HTTP `Deprecation` header is defined in RFC 9745 (published 2025) and is the correct signal. Pair it with `Sunset` (RFC 8594) to give a date, and `Link` with `rel="deprecation"` to point at migration docs.

Add a middleware that stamps every response for a deprecated version:

```python
from datetime import datetime, timezone
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

SUNSET_DATES = {
    "1": "2026-06-01T00:00:00Z",
}

class DeprecationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        version = getattr(request.state, "api_version", None)
        if version in SUNSET_DATES:
            response.headers["Deprecation"] = "true"
            response.headers["Sunset"] = SUNSET_DATES[version]
            response.headers["Link"] = (
                '<https://docs.example.com/migrate-v1>; rel="deprecation"'
            )
        return response
```

To populate `request.state.api_version`, set it inside the resolver:

```python
async def resolve_version(request: Request, x_api_version: Annotated[str | None, Header()] = None) -> str:
    # ... same resolution logic ...
    request.state.api_version = version
    return version
```

Every v1 response now carries a machine-readable warning. Clients that log headers will see it. API gateways can be configured to alert on it. This is the difference between "there is a plan to remove v1" and "v1 is being removed on a known date."

### The gotcha: caching proxies strip unknown headers

A common trap: teams add `Deprecation` and `Sunset`, verify locally with curl, and assume clients see them. In practice, shared caches and CDNs may not forward response headers they do not recognize, and some HTTP client libraries expose only a fixed set. If the header alone is relied on to warn consumers, some fraction of them will never see it.

Mitigation: also include the deprecation notice in the response body for deprecated versions, and log the version on the server side so it is possible to identify which clients still call v1. The header is the standard, but the log is what tells the team who to contact.

## Step 4 — Track usage before announcing a sunset

A version cannot be sunset if it cannot be measured. Before publishing a sunset date, it must be possible to know which clients call which version. The minimum viable tracking is a counter keyed by (version, client identifier, endpoint).

In FastAPI, this is a middleware or a logging hook:

```python
import logging

logger = logging.getLogger("api.version")

@app.middleware("http")
async def log_version_usage(request: Request, call_next):
    response = await call_next(request)
    version = getattr(request.state, "api_version", "unknown")
    client = request.headers.get("x-client-id", "anonymous")
    logger.info(
        "version_call",
        extra={"version": version, "client": client, "path": request.url.path},
    )
    return response
```

What matters is not the logging library. It is whether the system can answer "which clients called v1 in the last 30 days" before anything is announced. If it cannot, the sunset date is a guess, and guesses cause outages.

A reasonable policy: do not announce a sunset until v1 traffic is below a threshold that can be enumerated by hand. If the remaining callers fit on one screen, they can be contacted. If they do not, the version is not ready to retire.

## Step 5 — Build the migration path, not just the deprecation notice

Deprecation notices tell clients something is ending. They do not tell clients what to do. The migration path is the part that actually retires versions.

Three things make a migration path work:

1. **A compatibility shim.** If v1 and v2 differ only in response shape, offer a query parameter or header that returns v1 shape from the v2 endpoint. This lets clients upgrade their URL without changing their parsing code, then change parsing separately. Two smaller changes are easier to schedule than one big one.

2. **A changelog keyed to behavior, not to version numbers.** "v2 changed the `email` field to `contact.email`" is useful. "v2 released" is not.

3. **A test fixture for each supported version.** If the old version's contract test cannot be run against the current code, the version is not actually supported — it is just hoped for. Contract tests per version are the cheapest insurance against accidental breakage during refactors.

Here is a minimal per-version contract test using FastAPI's test client:

```python
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_v1_shape():
    r = client.get("/v1/users/1")
    assert r.status_code == 200
    body = r.json()
    assert "email" in body
    assert "contact" not in body
    assert r.headers.get("Deprecation") == "true"

def test_v2_shape():
    r = client.get("/v2/users/1")
    assert r.status_code == 200
    body = r.json()
    assert "contact" in body
    assert "Deprecation" not in r.headers
```

These tests are boring and that is the point. When someone refactors the user serializer, the v1 test fails, and the question "did we mean to break v1?" gets asked before deploy instead of after.

## Step 6 — Decide when to actually delete

The hardest part of versioning is not adding versions, it is removing them. A workable rule:

- A version can be deleted when its traffic is zero for a full billing cycle, *or* when every remaining caller has been contacted and has confirmed migration.
- The sunset date must be at least one full release cycle away from the announcement. For external APIs, that is typically measured in months, not weeks. The exact number depends on the clients' release cadence, which should be asked about directly.
- Deletion means the route returns `410 Gone`, not `404`. `404` implies the resource might exist elsewhere; `410` says it is intentionally gone. This matters for client error handling and for search engines.

```python
@app.get("/v1/{rest:path}")
async def v1_gone(rest: str):
    raise HTTPException(
        status_code=410,
        detail="API v1 was retired on 2026-06-01. See https://docs.example.com/migrate-v1",
    )
```

Keep the `410` handler in place for at least one cycle after deletion. It costs almost nothing and it turns "my integration broke mysteriously" into "my integration broke with a clear message."

## Frequently Asked Questions

**How long should a deprecated API version be kept?**
Long enough that every known client can migrate on their own release schedule. For internal clients that is often weeks; for external clients it can be quarters. The right way to decide is to ask the clients, not to pick a number. If they cannot be asked, the version probably cannot be safely retired.

**URI versioning or header versioning?**
URI versioning if clients are controlled and visible, cacheable URLs are wanted. Header versioning if URLs need to stay stable across versions. Date-based versioning if there are external clients that cannot be coordinated with. The cost of URI versioning is that retirement requires every client to change a URL, which is the main driver of version graveyards.

**What is the difference between the Deprecation and Sunset headers?**
`Deprecation` (RFC 9745) signals that a version is deprecated. `Sunset` (RFC 8594) gives a date after which it will stop working. Use both — deprecation without a date is a warning nobody can act on, and a date without deprecation is a surprise.

**Can v1 just run forever?**
It can, but the cost compounds. Every supported version is a version that must be tested, monitored, and patched. The graveyard is not a storage problem, it is a maintenance problem. The point of a sunset policy is to bound that cost, not to avoid it entirely.

## How to measure readiness for a sunset

Before announcing anything, instrument the following and keep the results in a place the whole team can see:

1. **Per-version request counts, broken down by client identifier.** The `x-client-id` header in the logging middleware above is a starting point; if clients do not send one, fall back to API keys, OAuth client IDs, or source IP ranges, and document the limitation.
2. **Per-version error rates.** A version with low traffic but a high error rate may already be effectively dead, or may be failing silently for a small set of clients.
3. **The date each version was introduced and the last date each client called it.** If the last-call date cannot be filled in, that is the version that will be hard to retire.
4. **The set of contract tests that currently pass for each version.** A version with no passing contract tests is not supported in any meaningful sense.

A worked example, with illustrative numbers: suppose v1 receives 10,000 requests per day, spread across 40 client identifiers. That is not yet tractable to contact by hand. After a deprecation header and a migration guide are published, suppose v1 drops to 200 requests per day across 3 client identifiers. That is tractable. The sunset date can now be set, the 3 clients contacted, and the `410` handler deployed once their traffic reaches zero. The arithmetic is simple; the hard part is having the data to do it.

## Where to go from here

Versioning strategy is a decision about deletion, not about addition. The teams that avoid graveyards decide *before* shipping v1 what would have to be true to remove it, and then measure that condition continuously.

Next 30 minutes: open the API's route definitions and count how many versions are live. For each one, write down the date it was introduced and the last date a client is known to have called it. If the second column cannot be filled in, that is the version that will be hard to retire — and the logging middleware in Step 4 is the first thing to add.
