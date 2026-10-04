# Senior role projects: 3 real paths

Portfolios full of CRUD apps and todo lists no longer distinguish candidates for senior remote roles. What distinguishes them is evidence that the author can reason about production constraints: partial failures, shared infrastructure, data isolation, and latency that varies by region. The three projects below are chosen because each one forces a specific senior-level decision into the open. None of them is flashy. All of them can be built and measured on modest hardware, and every performance claim in this article is something you can verify yourself rather than take on faith.

## Prerequisites and what you will build

You need a laptop, Node.js 20 LTS or Python 3.11+, and a container runtime. The stack used throughout:

- **Node.js 20 LTS** with a lightweight HTTP framework for the API layer (the code below uses Fastify-style plugin registration, but any framework with hooks works)
- **PostgreSQL 15** for the database
- **Redis 7.2** for caching, rate limiting, and pub/sub
- **Docker** for reproducible local environments
- **GitHub Actions** (or any CI runner) for automated checks

The three projects:

1. A multi-tenant SaaS API using PostgreSQL row-level security for tenant isolation
2. A rate-limited service that survives retry storms without cascading failure
3. A real-time analytics pipeline built on event sourcing and eventual consistency

Each surfaces a different senior concern: security, reliability, and observability. The constraint that ties them together is running them on a small shared VPS (2 vCPU, 2GB RAM is a reasonable target) while still behaving acceptably for users spread across regions.

## Step 1 — set up the environment

Create a `docker-compose.yml` that wires up PostgreSQL 15 and Redis 7.2, then start it:

```bash
docker compose up -d
```

A note on memory: on a 2GB host, Docker Desktop or the container runtime itself can consume a large fraction of RAM before your services start. Cap the PostgreSQL container explicitly to avoid swap thrashing during load:

```yaml
services:
  postgres:
    mem_limit: 1g
    cpus: 1.5
```

This is not a performance trick; it is a guardrail. When PostgreSQL competes with Redis for the same 2GB, the kernel will swap, and swap latency is what turns a slow request into a timeout. Measuring the effect is straightforward: run `docker stats` while your load test runs and watch whether the `MEM USAGE / LIMIT` column for any container approaches its cap. If it does, you have found your bottleneck before your users did.

## Step 2 — core implementation

### Project 1: Multi-tenant SaaS API with row-level security

A single-tenant CRUD API proves almost nothing. Tenant isolation is where the interesting failure modes live, and PostgreSQL's row-level security (RLS) is the mechanism that makes the isolation a property of the database rather than a property of every query an application author remembers to write.

```sql
-- tenants table
CREATE TABLE tenants (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  slug TEXT UNIQUE NOT NULL
);

-- users table with tenant_id
CREATE TABLE users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email TEXT NOT NULL UNIQUE,
  tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable RLS on users table
ALTER TABLE users ENABLE ROW LEVEL SECURITY;

-- Policy: users can only see their own tenant
CREATE POLICY tenant_isolation_policy ON users
  USING (tenant_id = current_setting('app.current_tenant')::UUID);
```

Two things to notice. First, the policy uses `current_setting('app.current_tenant')`, which means the application must set that value on the connection before any query runs. Second, `current_setting` with a missing value raises an error by default rather than returning null, which is the behavior you want: a query with no tenant context should fail loudly, not silently return every row.

In the API layer, resolve the tenant from a header and set the session variable:

```typescript
// src/plugins/tenant.ts
import fp from 'fastify-plugin';

export default fp(async (fastify) => {
  fastify.addHook('preHandler', async (request) => {
    const tenantSlug = request.headers['x-tenant-slug'];
    if (!tenantSlug) throw fastify.httpErrors.badRequest('Missing tenant slug');

    const tenant = await fastify.db.queryOne<{ id: string }>(
      `SELECT id FROM tenants WHERE slug = $1`,
      [tenantSlug]
    );
    if (!tenant) throw fastify.httpErrors.notFound('Tenant not found');

    await fastify.db.query('SELECT set_config($1, $2, false)', [
      'app.current_tenant',
      tenant.id,
      false
    ]);
  });
});
```

Register the plugin in your app:

```typescript
fastify.register(tenant);
```

The `false` argument to `set_config` means the setting applies for the current session rather than the current transaction. That choice matters. With a connection pool, a session-scoped setting persists on the connection after the request finishes, so the next request that checks out the same connection inherits the previous tenant's context unless it sets its own. Setting the value at the start of every request, as above, makes that safe. An alternative is `SET LOCAL` inside an explicit transaction, which scopes the setting to the transaction and resets automatically on commit or rollback. Both are correct; the failure mode is mixing them.

**Failure mode to design against:** an application-level filter (`WHERE tenant_id = $1`) that a developer forgets on one query. RLS moves the guarantee into the database, so a missing filter returns zero rows instead of another tenant's data.

### Project 2: Rate-limited service with retry storms

A rate limiter is a common portfolio piece. A rate limiter that behaves correctly when clients retry aggressively is rarer, and that is where the senior signal is.

The naive version, using Redis `INCR` with a TTL:

```javascript
// src/plugins/rate-limit.js
import fp from 'fastify-plugin';
import Redis from 'ioredis';

const redis = new Redis(process.env.REDIS_URL);

export default fp(async (fastify) => {
  fastify.addHook('preHandler', async (request, reply) => {
    const clientId = request.headers['x-client-id'];
    if (!clientId) return reply.code(400).send('Missing client ID');

    const key = `rate_limit:${clientId}`;
    const count = await redis.incr(key);

    if (count === 1) {
      await redis.expire(key, 60); // 60 second window
    }

    if (count > 100) {
      reply.code(429).send('Too many requests');
      return;
    }
  });
});
```

This has a known race: if the process dies between `INCR` and `EXPIRE`, the key never expires and that client is locked out permanently. The fix is to make the two operations atomic. Redis 7.0 and later support `EXPIRE` with the `NX` flag, and the standard pattern is a small Lua script or a pipeline that sets the expiry only when the counter is created:

```javascript
const script = `
  local count = redis.call('INCR', KEYS[1])
  if count == 1 then
    redis.call('EXPIRE', KEYS[1], ARGV[1])
  end
  return count
`;
const count = await redis.eval(script, 1, key, 60);
```

Now the real problem: when a client is rate-limited, a naive client retries immediately. A thousand clients doing that turns a 429 response into a self-inflicted denial of service. The server-side mitigation is to tell the client exactly how long to wait, and the client-side mitigation is jitter so that all the retries do not arrive in lockstep.

```javascript
// With exponential backoff and jitter
const retryAfter = Math.min(10, Math.pow(2, retryCount) + Math.random() * 2);
reply.header('Retry-After', retryAfter);
```

The `Retry-After` header is the contract; the jitter is what prevents synchronized retries from re-creating the spike you just absorbed. Note that `Retry-After` accepts either a number of seconds or an HTTP date; the numeric form above is the one most clients handle without parsing ambiguity.

**Failure mode to design against:** a rate limiter that returns 429 with no `Retry-After`, or with a constant value, so every client retries at the same instant.

### Project 3: Real-time analytics with event sourcing

A REST API that writes to SQL and polls for updates does not scale to a real-time dashboard. Event sourcing makes the write path append-only and the read path a projection, which is what allows the two to be scaled and reasoned about independently.

```python
# src/events/handlers.py
from sqlalchemy import create_engine, Column, String, Integer, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import redis.asyncio as redis
import uuid

Base = declarative_base()

class Event(Base):
    __tablename__ = 'events'
    id = Column(String, primary_key=True)
    type = Column(String)
    user_id = Column(String)
    payload = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

engine = create_engine('postgresql://user:pass@localhost:5432/analytics')
Session = sessionmaker(bind=engine)

async def handle_event(event_type: str, user_id: str, payload: dict):
    session = Session()
    event = Event(id=uuid.uuid4().hex, type=event_type, user_id=user_id, payload=str(payload))
    session.add(event)
    session.commit()
    session.close()

    # Publish to Redis for real-time updates
    r = redis.Redis.from_url('redis://localhost:6379')
    await r.publish('events', f'{event_type}:{user_id}')
```

The `uuid` import is required and easy to forget; without it the handler raises `NameError` on the first event. More importantly, the write-to-database-then-publish-to-Redis sequence is not atomic. If the process crashes between `commit()` and `publish()`, the event is stored but never broadcast, and any subscriber that was waiting for it misses it. The standard remedies are the transactional outbox pattern (write the event and an outbox row in one transaction, then have a separate process publish from the outbox) or accepting at-least-once delivery and making consumers idempotent. Which one you choose is exactly the kind of trade-off a senior role is about, and it is worth stating explicitly in a README.

On the client, subscribe to updates:

```javascript
// src/plugins/analytics.js
import fp from 'fastify-plugin';
import Redis from 'ioredis';

const redis = new Redis(process.env.REDIS_URL);

export default fp(async (fastify) => {
  fastify.get('/analytics', { websocket: true }, (connection, req) => {
    const sub = redis.duplicate();
    sub.subscribe('events');

    sub.on('message', (channel, message) => {
      connection.socket.send(message);
    });

    connection.socket.on('close', () => sub.unsubscribe());
  });
});
```

**Failure mode to design against:** publishing to Redis inside the same request that commits to PostgreSQL and treating the two as if they succeed or fail together. They do not.

## Step 3 — handle edge cases and errors

### Cache stampedes

When a cached value expires and many requests arrive for it at the same moment, every one of them misses the cache and hits the database. That is a cache stampede, and it is most likely to happen on the hottest key, which is the worst possible time.

A common mitigation is probabilistic early refresh: refresh the value slightly before it expires, so the expiry never coincides with a burst of misses.

```javascript
// src/plugins/cache.js
import fp from 'fastify-plugin';

const cache = new Map();

async function getWithRefresh(key, ttlMs, fetchFn) {
  const value = cache.get(key);
  if (value && value.expiresAt > Date.now() + 100) {
    return value.data;
  }

  // Background refresh
  if (!value || value.expiresAt < Date.now()) {
    const newData = await fetchFn();
    cache.set(key, { data: newData, expiresAt: Date.now() + ttlMs });
    return newData;
  }

  return value.data;
}

export default fp(async (fastify) => {
  fastify.decorate('cache', { getWithRefresh });
});
```

To verify this actually helps, instrument two counters: cache hits and database queries for the same key. Run a load test with a short TTL and compare the database query count with and without the early-refresh path. If the count is unchanged, your refresh window is too short relative to your request rate.

### Retry storms with circuit breakers

Naive retry logic against a struggling dependency makes the dependency's problem worse. A circuit breaker stops calling a dependency that is already failing, and gives it room to recover.

```python
# src/libs/circuit_breaker.py
import asyncio
from functools import wraps
import time

class CircuitBreaker:
    def __init__(self, max_failures=5, reset_timeout=30):
        self.max_failures = max_failures
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.last_failure = 0
        self.state = "closed"

    async def call(self, func, *args, **kwargs):
        if self.state == "open":
            if time.time() - self.last_failure > self.reset_timeout:
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is open")

        try:
            result = await func(*args, **kwargs)
            if self.state == "half-open":
                self.state = "closed"
                self.failures = 0
            return result
        except Exception as e:
            self.failures += 1
            self.last_failure = time.time()
            if self.failures >= self.max_failures:
                self.state = "open"
            raise

def circuit(max_failures=5, reset_timeout=30):
    cb = CircuitBreaker(max_failures, reset_timeout)
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await cb.call(func, *args, **kwargs)
        return wrapper
    return decorator
```

Wrap your Redis calls:

```python
@circuit(max_failures=3, reset_timeout=10)
async def get_rate_limit(client_id: str):
    r = redis.Redis.from_url('redis://localhost:6379')
    return await r.incr(f'rate_limit:{client_id}')
```

The half-open state is the important part: after the reset timeout, the breaker allows a single request through. If it succeeds, the breaker closes and normal traffic resumes. If it fails, the breaker reopens. Without the half-open state you either never recover or you recover into a thundering herd.

### Tenant isolation at the edge

A connection pool shared across tenants is a subtle leak vector: if tenant context is set per session and the pool hands a connection to a different tenant without resetting, queries can see the wrong rows. One mitigation is a pool per tenant with a bounded connection count, so one tenant cannot exhaust the shared pool.

```typescript
// src/plugins/tenant-pool.ts
import fp from 'fastify-plugin';
import { Pool } from 'pg';

const tenantPools = new Map<string, Pool>();

async function getTenantPool(tenantId: string) {
  if (!tenantPools.has(tenantId)) {
    const pool = new Pool({
      connectionString: process.env.DATABASE_URL,
      max: 5, // Limit connections per tenant
    });
    tenantPools.set(tenantId, pool);
  }
  return tenantPools.get(tenantId)!;
}

export default fp(async (fastify) => {
  fastify.decorate('getTenantPool', getTenantPool);
});
```

This trades connection efficiency for isolation. On a small VPS with many tenants, the per-tenant `max` must be set low enough that the sum of pools does not exceed PostgreSQL's `max_connections` (default 100). If it does, new connections are refused, which is a worse failure than the one you were preventing.

## Step 4 — add observability and tests

### Logging and tracing

Distributed tracing is the difference between "the endpoint is slow" and "the endpoint is slow because the tenant lookup is doing a sequential scan." OpenTelemetry with a Jaeger backend is a common local setup:

```typescript
// src/plugins/observability.ts
import fp from 'fastify-plugin';
import { NodeSDK } from '@opentelemetry/sdk-node';
import { getNodeAutoInstrumentations } from '@opentelemetry/auto-instrumentations-node';
import { JaegerExporter } from '@opentelemetry/exporter-jaeger';

const sdk = new NodeSDK({
  traceExporter: new JaegerExporter({ endpoint: 'http://localhost:14268/api/traces' }),
  instrumentations: [getNodeAutoInstrumentations()],
});

sdk.start();

export default fp(async (fastify) => {
  fastify.decorate('tracer', sdk.getTracer('portfolio-api'));
});
```

Run Jaeger locally:

```bash
docker run -d --name jaeger \
  -e COLLECTOR_ZIPKIN_HTTP_PORT=9411 \
  -p 5775:5775/udp \
  -p 6831:6831/udp \
  -p 6832:6832/udp \
  -p 5778:5778 \
  -p 16686:16686 \
  -p 14268:14268 \
  -p 9411:9411 \
  jaegertracing/all-in-one:1.48
```

The Jaeger UI is on port 16686. The point of running it is not the dashboard; it is that you can point to a specific slow request and show which span consumed the time.

### Tests that simulate real constraints

A load test turns "it works on my machine" into a number. k6 is one option:

```javascript
// load-test.js
import http from 'k6/http';
import { check } from 'k6';

export const options = {
  stages: [
    { duration: '2m', target: 20 },
    { duration: '5m', target: 50 },
    { duration: '2m', target: 20 },
  ],
  thresholds: {
    http_req_duration: ['p(95)<250'], // 250ms p95
  },
};

export default function () {
  const res = http.get('http://localhost:3000/api/users', {
    tags: { name: 'get_users' },
    headers: { 'x-tenant-slug': 'acme' },
  });
  check(res, {
    'status was 200': (r) => r.status == 200,
  });
}
```

The `thresholds` block is the part that matters: it fails the run if p95 latency exceeds 250ms, so the test produces a pass/fail signal rather than a graph someone has to interpret.

### How to measure the claims in this article yourself

Every performance claim here should be reproducible on your own hardware. The measurement plan:

1. **Baseline latency.** Run the load test above against an endpoint with no cache. Record p50, p95, and p99 from the k6 summary output.
2. **Cache effect.** Add Redis in front of the same endpoint and rerun. Compare the same percentiles. Note the TTL you used; the result is only meaningful for that TTL and that request rate.
3. **Database load.** Query `pg_stat_statements` before and after, or watch `pg_stat_database` for the number of sequential scans. A cache that reduces latency but not database load is not doing what you think.
4. **Circuit breaker behavior.** Point the wrapped call at a dependency you can stop (`docker stop <redis-container>`), then run the load test and confirm that after the failure threshold is crossed, requests fail fast rather than waiting on connection timeouts.
5. **Memory headroom.** Run `docker stats` throughout. On a 2GB host, the interesting number is how close any container gets to its limit, not the average.

Record the numbers in the README along with the hardware, the TTLs, and the load profile. That is what makes the result a measurement rather than a claim.

## Common questions and variations

### What if you do not have a VPS to test on?

A free-tier managed PostgreSQL instance plus a free-tier managed Redis instance is enough to exercise every failure mode described here. The constraint you are simulating is a small connection limit and limited memory, and the free tiers impose both. The one thing you cannot simulate this way is network latency between regions; for that, run the load generator from a different region than the service.

### TypeScript or Python?

TypeScript is the more common choice for API-focused roles; Python is fine for data and analytics roles. The decision that matters more than the language is whether the project demonstrates a real trade-off with a stated reason.

### How do you handle tenant migrations?

Run migrations per tenant inside a transaction, and keep the migration tooling aware of which tenants have been migrated. `pg_dump` and `pg_restore` are for backup and restore, not for schema migration; using them for the latter means you lose the ability to apply a migration to one tenant without affecting others. A migration ledger table keyed by tenant is the usual approach.

| Scenario | Toolchain | Senior-level concern |
|---|---|---|
| Multi-tenant SaaS API | PostgreSQL RLS + TypeScript | Data isolation, connection pool safety |
| Rate-limited service | Redis + atomic increment + circuit breaker | Retry storms, cascading failure |
| Real-time analytics | Event sourcing + pub/sub + outbox | Delivery guarantees, eventual consistency |

## Where to go from here

Pick one project and deploy it. Add a load test with a threshold, not just a graph. Write a README section titled "Trade-offs and failure modes" and be specific about what you chose not to do and why.

**Do this in the next 30 minutes:** open your tenant middleware and add a comment above the `set_config` call explaining why the setting is session-scoped rather than transaction-scoped, and what would break if a request failed to set it. If you cannot write that comment confidently, that is the gap worth closing first.
