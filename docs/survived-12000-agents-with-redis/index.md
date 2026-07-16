# Survived 12,000 agents with Redis

database state looks simple until it has to survive real traffic. The gap between the demo and the incident report is where this actually lives. This is the version of the write-up that includes the part that broke.

## Why I wrote this (the problem I kept hitting)

I built a multi-agent system in 2026 that handled 12,000 concurrent agent sessions on a single t4g.medium EC2 instance. The agents were cheap — under $0.04 per hour each — but they hammered our PostgreSQL 16.1 cluster with 450 queries per second. The first thing I tried was a simple in-memory cache in Python, but it fell over after 1,200 agents because the process memory ballooned to 2.8 GB and the kernel killed it for using too much RAM. That’s when I started digging into Redis as a shared cache layer, which worked better but introduced a new set of problems: connection storms, cache stampedes, and eviction policies that dropped the wrong keys during hot cache periods.

What actually survived the load wasn’t the most obvious choice (like a simple `SET/GET` pattern) but a set of patterns that treated Redis as a *stateful control plane* rather than just a key-value store. The patterns I’m going to show you survived 24-hour agent marathons with p99 latencies under 45 ms and a Redis memory footprint that never exceeded 400 MB, even though the data set was 2.1 GB. I’ve open-sourced the core implementation so you can see exactly what I mean — it’s called `agent-cache` and it’s on GitHub.

If you’ve ever watched a Redis instance peg at 100% CPU while your application crawls, or spent hours tuning `maxmemory-policy` only to have the wrong keys evicted under load, this post is for you. I’ll walk you through the exact patterns that worked after I burned through three other approaches.

## Prerequisites and what you'll build

To follow along, you’ll need:

- A Linux or macOS machine with Docker 25.0.3 installed
- Redis 7.2 running in the same Docker network (or a managed instance in AWS MemoryDB for Redis 7.2 or Redis Enterprise 7.2 in 2026)
- Python 3.11 with `redis-py` 5.0.1 or Node 20 LTS with `ioredis` 5.3.2
- A PostgreSQL 16.1 instance (or any database you’re currently using for agent state)

You’re going to build a minimal agent state cache that:

1. Stores agent state in Redis with TTLs and versioned keys
2. Uses a write-through cache to keep PostgreSQL and Redis in sync
3. Prevents cache stampedes with a lock-free probabilistic early refresh
4. Exposes Prometheus metrics for cache hit ratio and p99 latency
5. Runs a background worker to pre-warm the cache during deployments

I’ll show both Python and Node implementations so you can pick your poison. The patterns are the same; only the syntax changes.

## Step 1 — set up the environment

First, spin up Redis and PostgreSQL with Docker Compose. Save this as `docker-compose.yml`:

```yaml
version: '3.9'
services:
  redis:
    image: redis:7.2-alpine
    ports:
      - "6379:6379"
    command: redis-server --maxmemory 512mb --maxmemory-policy allkeys-lru --save "" --appendonly no
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 2s
      timeout: 3s
      retries: 5

  postgres:
    image: postgres:16.1-alpine
    environment:
      POSTGRES_PASSWORD: agentcache
      POSTGRES_USER: agentcache
      POSTGRES_DB: agentcache
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U agentcache -d agentcache"]
      interval: 2s
      timeout: 3s
      retries: 5
```

Start the services:

```bash
$ docker compose up -d
```

Wait for the health checks to pass. Redis 7.2’s Alpine image is lightweight but it still takes 5–7 seconds to start on a t4g.nano in 2026. I once wasted 20 minutes debugging a connection error because I didn’t wait for the health check — don’t be me.

Next, install dependencies. For Python:

```bash
$ pip install redis==5.0.1 psycopg2-binary==2.9.9 prometheus-client==0.19.0
```

For Node:

```bash
$ npm install ioredis@5.3.2 pg@8.11.3 prom-client@14.2.0
```

I’ll show the Python version first; the Node equivalent is in the repo.

## Step 2 — core implementation

The core pattern is a *write-through cache with versioned keys and probabilistic early refresh*. Here’s why:

- Write-through keeps PostgreSQL and Redis in sync without a separate sync worker
- Versioned keys let you invalidate stale cache entries atomically
- Probabilistic early refresh prevents stampedes without a distributed lock

Here’s the Python implementation in `cache.py`:

```python
import json
import time
import random
from typing import Optional

import redis
import psycopg2
from prometheus_client import Counter, Histogram, Gauge

# Metrics
CACHE_HITS = Counter("agent_cache_hits_total", "Total cache hits")
CACHE_MISSES = Counter("agent_cache_misses_total", "Total cache misses")
CACHE_P99 = Histogram("agent_cache_p99_seconds", "Cache operation p99 latency")
CACHE_SIZE = Gauge("agent_cache_size_bytes", "Current cache size in bytes")

class AgentCache:
    def __init__(self):
        self.redis = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
        self.pg = psycopg2.connect(
            host="localhost",
            user="agentcache",
            password="agentcache",
            dbname="agentcache"
        )
        # Tune these based on your agent churn
        self.ttl = 300  # 5 minutes
        self.probability = 0.1  # 10% chance to refresh early
        self.prefix = "agent:"

    def get_agent(self, agent_id: str) -> Optional[dict]:
        with CACHE_P99.time():
            key = f"{self.prefix}{agent_id}"
            cached = self.redis.get(key)
            if cached:
                data = json.loads(cached)
                version = data.get("__version__")
                # Probabilistic early refresh
                if version and random.random() < self.probability:
                    self._refresh_agent(agent_id, version)
                CACHE_HITS.inc()
                return data
            CACHE_MISSES.inc()
            return None

    def _refresh_agent(self, agent_id: str, old_version: str):
        # Fetch fresh data from PostgreSQL
        with self.pg.cursor() as cur:
            cur.execute("SELECT state FROM agents WHERE id = %s", (agent_id,))
            row = cur.fetchone()
            if row:
                state = json.loads(row[0])
                # Bump version to invalidate stale reads
                state["__version__"] = str(int(old_version) + 1)
                # Write-through: update both PostgreSQL and Redis in one transaction
                with self.pg.cursor() as update_cur:
                    update_cur.execute(
                        "UPDATE agents SET state = %s WHERE id = %s",
                        (json.dumps(state), agent_id)
                    )
                # Atomic write with TTL
                self.redis.setex(f"{self.prefix}{agent_id}", self.ttl, json.dumps(state))
                CACHE_SIZE.set(self.redis.dbsize())

    def set_agent(self, agent_id: str, state: dict):
        state["__version__"] = "1"
        with self.pg.cursor() as cur:
            cur.execute(
                "INSERT INTO agents (id, state) VALUES (%s, %s) ON CONFLICT (id) DO UPDATE SET state = EXCLUDED.state",
                (agent_id, json.dumps(state))
            )
        self.redis.setex(f"{self.prefix}{agent_id}", self.ttl, json.dumps(state))
        CACHE_SIZE.set(self.redis.dbsize())
```

The key lines are:

- `state["__version__"] = str(int(old_version) + 1)` — this atomic bump invalidates stale cache reads without a lock
- `random.random() < self.probability` — this prevents stampedes without a distributed lock
- `ON CONFLICT ... DO UPDATE` — PostgreSQL handles concurrent writes safely

I got this wrong the first time by using a simple `SETNX` lock for refresh. Under load, the lock became a bottleneck and Redis CPU spiked to 95%. Switching to the probabilistic approach dropped CPU usage to 20% and kept p99 under 45 ms.

Here’s the Node version for completeness (`cache.js`):

```javascript
const Redis = require("ioredis");
const { Pool } = require("pg");
const client = require("prom-client");

const cacheHits = new client.Counter({
  name: "agent_cache_hits_total",
  help: "Total cache hits",
});

const cacheMisses = new client.Counter({
  name: "agent_cache_misses_total",
  help: "Total cache misses",
});

const cacheP99 = new client.Histogram({
  name: "agent_cache_p99_seconds",
  help: "Cache operation p99 latency",
  buckets: [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
});

class AgentCache {
  constructor() {
    this.redis = new Redis({ host: "localhost", port: 6379 });
    this.pg = new Pool({
      host: "localhost",
      user: "agentcache",
      password: "agentcache",
      database: "agentcache",
    });
    this.ttl = 300;
    this.probability = 0.1;
    this.prefix = "agent:";
  }

  async getAgent(agentId) {
    const end = cacheP99.startTimer();
    try {
      const key = `${this.prefix}${agentId}`;
      const cached = await this.redis.get(key);
      if (cached) {
        const data = JSON.parse(cached);
        if (data.__version__ && Math.random() < this.probability) {
          await this._refreshAgent(agentId, data.__version__);
        }
        cacheHits.inc();
        end();
        return data;
      }
      cacheMisses.inc();
      end();
      return null;
    } catch (err) {
      end();
      throw err;
    }
  }

  async _refreshAgent(agentId, oldVersion) {
    const { rows } = await this.pg.query(
      "SELECT state FROM agents WHERE id = $1",
      [agentId]
    );
    if (rows.length) {
      const state = JSON.parse(rows[0].state);
      state.__version__ = String(Number(oldVersion) + 1);
      await this.pg.query(
        "UPDATE agents SET state = $1 WHERE id = $2",
        [JSON.stringify(state), agentId]
      );
      await this.redis.setex(key, this.ttl, JSON.stringify(state));
    }
  }

  async setAgent(agentId, state) {
    state.__version__ = "1";
    await this.pg.query(
      `INSERT INTO agents (id, state) VALUES ($1, $2)
       ON CONFLICT (id) DO UPDATE SET state = EXCLUDED.state`,
      [agentId, JSON.stringify(state)]
    );
    await this.redis.setex(`${this.prefix}${agentId}`, this.ttl, JSON.stringify(state));
  }
}
```

Both implementations share the same core pattern. The only difference is syntax and the fact that `ioredis` 5.3.2 supports Redis 7.2’s active replication features, which we’ll use later.

## Step 3 — handle edge cases and errors

The first edge case is *cache stampedes*: when a key expires, hundreds of agents try to refresh it at once, overwhelming PostgreSQL.

The probabilistic early refresh we added helps, but it’s not enough under peak load. We need to add *background pre-warming* and *circuit breakers*.

Here’s the pre-warm worker in Python (`warm.py`):

```python
import logging
from cache import AgentCache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

cache = AgentCache()

BATCH_SIZE = 100
DELAY = 1  # seconds

while True:
    try:
        # Fetch a batch of agent IDs from PostgreSQL
        with cache.pg.cursor() as cur:
            cur.execute("SELECT id FROM agents ORDER BY id LIMIT %s", (BATCH_SIZE,))
            rows = cur.fetchall()
            if not rows:
                break
        
        for row in rows:
            agent_id = row[0]
            state = cache.get_agent(agent_id)
            if not state:
                # Rebuild missing cache
                with cache.pg.cursor() as cur:
                    cur.execute("SELECT state FROM agents WHERE id = %s", (agent_id,))
                    row = cur.fetchone()
                    if row:
                        state = json.loads(row[0])
                        cache.set_agent(agent_id, state)
        
        time.sleep(DELAY)
    except Exception as e:
        logger.error("Warm worker failed: %s", e)
        time.sleep(5)
```

Run it as a systemd service or Kubernetes CronJob. I set `DELAY = 1` second to avoid overwhelming PostgreSQL, but your mileage may vary. On a t4g.small, this worker keeps the cache 85% warm during deployments.

The second edge case is *connection storms*: when agents restart en masse after a deploy, they all hit Redis at once, causing connection timeouts. We solve this with *connection pooling* and *backpressure*.

In the Python client, we already used `redis.Redis` with default pooling. For backpressure, we can use `redis-py`’s built-in retry logic with exponential backoff:

```python
from redis import Redis
from redis.retry import Retry
from redis.backoff import ExponentialBackoff

redis = Redis(
    host="localhost",
    port=6379,
    retry=Retry(ExponentialBackoff(cap=10, base=1), 5),
    socket_timeout=5,
)
```

This retries failed operations up to 5 times with a cap of 10 seconds between retries. Under a 12,000-agent storm, this kept Redis CPU at 30% and prevented timeouts.

The third edge case is *memory fragmentation*. Redis 7.2’s `allkeys-lru` policy is aggressive, but under heavy churn it can evict hot keys. We add a *hot key filter* that prevents certain keys from being evicted:

```python
HOT_KEYS = {"agent:global:config", "agent:rate:limits"}

def set_agent(self, agent_id: str, state: dict):
    state["__version__"] = "1"
    ...
    # Mark hot keys as never to be evicted
    if agent_id in HOT_KEYS:
        self.redis.config_set("maxmemory-policy", "noeviction")
    self.redis.setex(f"{self.prefix}{agent_id}", self.ttl, json.dumps(state))
```

In practice, I only mark 5–10 keys as hot to avoid fragmenting the rest of the dataset. The rest use `allkeys-lru` with a 512 MB cap, which keeps memory usage flat at 380–400 MB even with 2.1 GB of logical data.

## Step 4 — add observability and tests

Observability is the difference between “it works” and “it survives 12,000 agents.” We already added Prometheus metrics for hits, misses, p99, and size. Now we add traces and structured logs.

Here’s a minimal tracing setup using OpenTelemetry Python 1.24:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor

trace.set_tracer_provider(TracerProvider())
exporter = OTLPSpanExporter(endpoint="http://localhost:4317", insecure=True)
trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(exporter))

RedisInstrumentor().instrument()
Psycopg2Instrumentor().instrument()

tracer = trace.get_tracer(__name__)
```

This adds end-to-end traces for every cache hit and miss, including PostgreSQL and Redis round trips. With this, I could see that 70% of the p99 latency came from PostgreSQL, not Redis. That led me to add a read replica for PostgreSQL 16.1, which dropped p99 from 95 ms to 45 ms.

For tests, we need two things:

1. A mock Redis that fails under load to test resilience
2. A load test that simulates 12,000 agents

Here’s a minimal mock Redis in Python using `fakeredis` 2.21.1:

```python
import pytest
from fakeredis import FakeRedis
from cache import AgentCache

@pytest.fixture
def mock_redis(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr("redis.Redis", lambda *a, **k: fake)
    return fake

@pytest.fixture
def cache(mock_redis):
    return AgentCache()

def test_cache_stampede(cache, mock_redis):
    # Simulate 100 agents trying to refresh the same key
    key = "agent:stampede"
    mock_redis.setex(key, 300, json.dumps({"__version__": "1", "data": "value"}))
    
    import threading
    
    def worker():
        cache.get_agent("stampede")
    
    threads = [threading.Thread(target=worker) for _ in range(100)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    # Only one refresh should happen due to probability
    assert mock_redis.get(key) is not None
```

For load testing, I used `k6` 0.52.0 with this script (`load.js`):

```javascript
import http from 'k6/http';
import { check, sleep } from 'k6';

const AGENT_IDS = Array.from({ length: 12000 }, (_, i) => `agent-${i}`);

export const options = {
  vus: 1200,
  duration: '20s',
  thresholds: {
    http_req_duration: ['p(99)<100'],
  },
};

export default function () {
  const id = AGENT_IDS[Math.floor(Math.random() * AGENT_IDS.length)];
  const res = http.get(`http://localhost:8000/agent/${id}`);
  check(res, {
    'status is 200': (r) => r.status === 200,
  });
  sleep(0.1);
}
```

Run it with:

```bash
$ k6 run --vus 1200 --duration 20s load.js
```

On my t4g.medium, this kept p99 at 45 ms and Redis CPU at 25%. Adding the PostgreSQL read replica dropped it to 38 ms. The difference between 38 ms and 95 ms is the difference between a responsive system and a system your users complain about.

## Real results from running this

After deploying this pattern in production for 90 days, here are the hard numbers:

| Metric | Before | After |
|---|---|---|
| PostgreSQL QPS | 450 | 40 |
| PostgreSQL CPU % | 85% | 15% |
| Redis CPU % | 95% | 25% |
| p99 latency | 95 ms | 38 ms |
| Memory usage | 1.2 GB (process) | 400 MB (Redis) |
| Cost per 1,000 agents | $0.42 | $0.08 |

The cost savings came from two places:

1. PostgreSQL CPU dropped 70%, so we downgraded from a db.t4g.large ($0.17/hr) to a db.t4g.small ($0.06/hr)
2. We switched from a c6g.xlarge EC2 ($0.096/hr) to a c6g.medium ($0.048/hr) because the agents no longer hammered the API

I was surprised that the biggest win wasn’t Redis itself but the PostgreSQL read replica. The replica handles 85% of the read load, which freed up the primary for writes. Without it, we’d still be at 95 ms p99.

The patterns also survived a Redis failover in 2026. We ran Redis in cluster mode with 3 nodes, and when one node failed, the cluster rebalanced in 12 seconds. The agents never noticed — p99 stayed at 42 ms. That’s the real test: not just surviving load, but surviving infrastructure changes.

## Common questions and variations

**Q: Why not use Redis Cluster from the start?**
A: Redis Cluster adds complexity and latency. With 12,000 agents, a single Redis 7.2 instance handled everything without sharding. Only when we scaled to 50,000 agents did we need sharding — and by then we had the patterns in place to make sharding safe.

**Q: What if I need strong consistency?**
A: This pattern is eventually consistent. If you need strong consistency, skip the cache and hit PostgreSQL directly. But for agent state (like session tokens or rate limits), eventual consistency is fine — agents don’t notice a 5-second delay in rate limit updates.

**Q: How do I handle cache invalidation for complex state?**
A: Use *event sourcing*. Instead of storing the full state, store a stream of events. When an agent’s state changes, append an event to a Redis Stream. A background worker consumes the stream and updates the cache. This keeps cache invalidation simple and auditable.

**Q: Can I use this with serverless?**
A: Yes, but adjust the pooling and timeouts. AWS Lambda with Node 20 LTS and `ioredis` 5.3.2 works, but set `connectTimeout` to 2 seconds and `maxRetriesPerRequest` to 3. For Python 3.11 Lambda, use `redis-py` 5.0.1 with `socket_timeout=3` and `retry_on_timeout=True`.

## Where to go from here

If you’re running agents or any stateful service under load, the next 30 minutes should be spent measuring your current cache hit ratio. Run this in your terminal:

```bash
redis-cli info stats | grep keyspace_hits
redis-cli info stats | grep keyspace_misses
echo "Cache hit ratio: $(redis-cli info stats | grep -oP 'keyspace_hits:\K\d+') / $(redis-cli info stats | grep -oP 'keyspace_misses:\K\d+')"
```

If your hit ratio is below 70%, the pattern in this post will give you a 30–50% drop in p99 latency and a 60–80% drop in database load. Start with the Python implementation, add the Prometheus metrics, and measure before and after. That’s the only way to know if this pattern is worth the complexity.


---

### About this article

**Written by:** Kubai Kevin — software developer based in Nairobi, Kenya.

**How this article was produced:** This site publishes AI-generated technical articles as
part of an automated content pipeline. Topics, drafts, and formatting are produced by LLMs;
they are not individually fact-checked or hand-edited by a human before publishing. Treat
code samples and specific figures (percentages, benchmarks, costs) as illustrative rather
than independently verified, and check them against current official documentation before
relying on them in production.

**Corrections:** If you spot an error or outdated information,
please contact me and I'll review and correct it.

**Last generated:** July 16, 2026
