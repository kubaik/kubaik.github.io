# Spot AI’s hidden cost spike in your stack

## Why a faster refactor can cost more

An AI coding assistant rewrites a loop into a set join. Local tests pass, latency drops, and the code looks cleaner. Two weeks later the database line item on the cloud bill has tripled while CPU utilisation and response time look unchanged.

This is a common failure mode, not a rare one. The confusion comes from the fact that AI tools optimise for the code in front of them, not for the system around it. A rewrite can make a function measurably faster in isolation while removing the implicit buffering, caching, or batching that was absorbing most of the production load. The surface symptoms — high database I/O, latency spikes, budget alerts — look like classic performance problems. The root cause is usually an architectural change that only manifests under real traffic.

Three broad categories account for most of these incidents:

- **Cache bypass.** The rewrite replaces a cached read path with a direct database read path.
- **Concurrency mismatch.** The rewrite changes the shape of the workload so that it no longer fits the connection pool, transaction timeout, or provisioned capacity.
- **Environment drift.** The rewrite changes how the application talks to a managed service, but the timeout, IAM policy, or capacity setting was not updated to match.

Each is diagnosable. None requires guessing.

## Cache bypass: the most common cause

A frequent rewrite pattern is N+1 query elimination. An LLM sees a loop that fetches a list of parent records and then fetches a child record for each one, and replaces it with a single `SELECT ... IN` or a join. That is often a genuine improvement. But if the original per-item fetch was served from a cache layer, the rewrite converts cache hits into database reads.

The arithmetic is worth stating explicitly. Suppose an endpoint serves 1,000 requests per minute, each request previously issuing 10 child fetches, and the cache absorbs 90% of those fetches. That is 10,000 fetches per minute, of which 1,000 reach the database and 9,000 are served from cache. Remove the cache from the path and the database now handles 10,000 fetches per minute — a tenfold increase in database work for the same user-visible traffic. Database I/O is typically the most expensive operation in a web application, so the bill scales roughly with that multiplier.

### How to detect it

Compare the cache hit ratio before and after the change. For a Redis-compatible cache, the relevant counters are exposed by `INFO stats`:

```bash
redis-cli -h your-cache-endpoint -p 6379 INFO stats | grep -E 'keyspace_hits|keyspace_misses'
```

Compute the ratio as `keyspace_hits / (keyspace_hits + keyspace_misses)`. A workload that previously sat above 85% and now sits below 70% is a strong signal that an architectural rewrite bypassed the cache.

Latency is a secondary signal, not a primary one. A cold or missing cache shows up as elevated command latency, but so do network issues and slow commands, so confirm with the hit ratio before acting.

To count queries per request, use the instrument your framework already provides. In Django, `django.db.connection.queries` records executed queries when `DEBUG` is enabled. A minimal Flask equivalent:

```python
from flask import Flask, g

app = Flask(__name__)

@app.before_request
def before_request():
    g.queries = []

@app.after_request
def after_request(response):
    app.logger.info("Queries executed: %d", len(g.queries))
    return response
```

In production, prefer a query counter that does not depend on debug mode — a database proxy that logs statement counts, or an APM agent that reports queries per transaction. Run the same endpoint 100 times before and after the change and compare the median query count. A jump from single digits to hundreds per request is the signature of a removed cache or a reintroduced N+1.

### The fix

Reintroduce the caching layer explicitly rather than relying on the AI to preserve it. In Django, that means `@cache_page` on a view or `cache.set(key, value, timeout=...)` around the expensive call. In Express, it means a response-caching middleware. The rule that matters: if an endpoint returns data that is not user-specific and does not change second to second, cache it, and choose the TTL deliberately. Five minutes is a reasonable starting point for dynamic data, one hour for semi-static reference data. Never cache user-specific pages under a shared key.

## Connection pool and transaction limits

The second category is a concurrency mismatch. An AI rewrite collapses many small operations into one batched operation, or splits one operation into many. Either direction can exceed a configured limit.

The symptom is high write latency while CPU is low and the database looks idle. The application is waiting for a connection, not for the database to work.

A representative error from psycopg2 is:

```
psycopg2.OperationalError: connection limit exceeded for non-replication connection
```

Note that the pool size that matters is the one your ORM or connection pooler enforces, not the database's `max_connections`. Most ORMs default to a pool in the range of 5 to 20 connections. If a rewrite turns 500 small writes into one batched write that occupies a connection for the duration of the batch, and your pool is 20, the remaining requests queue behind it.

In SQLAlchemy:

```python
engine = create_engine(
    "postgresql://user:pass@localhost/db",
    pool_size=50,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=300
)
```

In Django:

```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'db',
        'USER': 'user',
        'PASSWORD': 'pass',
        'HOST': 'localhost',
        'PORT': '5432',
        'CONN_MAX_AGE': 300,
        'OPTIONS': {
            'connect_timeout': 3,
        }
    }
}
```

Pool size is a hard-to-reverse decision in the sense that it is coupled to the database's own connection limit and to memory. Raising it without checking `max_connections` on the database simply moves the failure. A defensible starting point is to size the pool to the number of concurrent requests your load test actually sustains, not to a guess. Multiply concurrent requests by a small headroom factor, then verify against the database's connection limit and the memory cost per connection.

Transaction timeouts are the companion problem. A rewrite that merges ten short transactions into one long transaction can exceed `statement_timeout`. PostgreSQL's default is `0`, meaning no timeout, but many managed services set a finite value. If the transaction now exceeds it, the statement is cancelled:

```
psycopg2.errors.QueryCanceled: canceling statement due to statement timeout
```

The client retries, the retries stack up, and a thundering herd forms. Set a timeout deliberately and split long batches:

```sql
ALTER SYSTEM SET statement_timeout = '5000';
SELECT pg_reload_conf();
```

Then confirm the timeout is the cause by checking whether the cancellation errors correlate with the new code path. If they do, either lower the timeout further or chunk the batch.

## Managed-service rewrites

The third category is environment drift. The rewrite changes how the application talks to a managed service, but the configuration around that service was not updated.

### Synchronous call replaced by a queue publish

A common rewrite replaces a slow external API call with a publish to a message queue. That is often correct. But the compute function's timeout may still be set to the value that suited the original synchronous call. The function now finishes in milliseconds, yet it is configured with a multi-second timeout, and any downstream consumer that expects a synchronous result sees a timeout:

```
Task timed out after 5.01 seconds
```

The fix has two parts. First, lower the function timeout to something close to the actual work:

```python
import boto3
import os

sqs = boto3.client('sqs', region_name='eu-west-1')
queue_url = os.getenv('STRIPE_EVENT_QUEUE')

def handler(event, context):
    sqs.send_message(
        QueueUrl=queue_url,
        MessageBody=event['body'],
        MessageGroupId='stripe'
    )
    return {"statusCode": 202, "body": "Accepted"}
```

Second, decide whether the async boundary is acceptable. Once you publish to a queue, you have committed to eventual consistency. If the product promises a result within two seconds, you need either a synchronous path or a polling step with a deadline. This is an architectural decision, not a bug fix, and it should be made before the rewrite ships.

### Point read replaced by a query

Another recurring pattern replaces a single-item read with a query against a secondary index. The read capacity cost is not the same. A point read of a single item consumes a fixed, small amount of read capacity, while a query consumes capacity proportional to the number of items examined. If the query scans a thousand items where the original read touched one, the capacity consumed per request can be orders of magnitude higher.

If the access pattern genuinely needs a query, keep it but constrain the result set with a key condition and a projection. If it only needs one item, revert to the point read:

```python
import boto3

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('Orders')

# Query against a secondary index
response = table.query(
    IndexName='user_id-index',
    KeyConditionExpression='user_id = :uid',
    ExpressionAttributeValues={':uid': 'user123'}
)

# Point read when the primary key is known
response = table.get_item(
    Key={'order_id': 'order456'}
)
```

The point read above uses only the primary key. Passing `ExpressionAttributeValues` to `get_item` for an attribute that is not part of the key has no effect and is a common mistake in AI-generated code; remove it.

## How to verify a fix

Detection is not verification. A fix is verified when the same workload produces the same or better metrics on the new code path.

Start with a traffic replay against a staging environment that mirrors the post-change state. A production traffic slice, captured and replayed, is the closest thing to a controlled experiment available without risking live users. Measure three things:

1. **Cache hit ratio**, from the cache's own statistics endpoint.
2. **Query count per endpoint**, from your ORM instrumentation or database proxy logs.
3. **Tail latency**, such as p95 or p99 from your metrics backend.

Compare the same three metrics before and after the change, under the same load. If the hit ratio returns to its previous level and query counts fall, the cache fix worked. If tail latency remains high, you have masked the symptom rather than fixed the cause.

Then run a load test with a tool such as k6 or Locust at a concurrency level you expect in production. The replayed slice tells you whether the change is directionally correct; the load test tells you whether it holds under pressure. A fix that works at 10 concurrent users and fails at 100 is not a fix.

Finally, check the cost delta in your cloud provider's cost explorer, filtered to the service that changed. Cost data lags, so allow for the billing period before concluding. If the delta persists after the metrics have normalised, something else in the change set is responsible.

## Gating AI refactors before they ship

Prevention is cheaper than diagnosis. Three controls cover most of the risk.

**Gate the change behind a feature flag.** A flag lets you compare the old and new code paths in the same environment without affecting all users. A minimal Django decorator:

```python
from django.conf import settings
from functools import wraps
import logging

logger = logging.getLogger('ai_refactor')

def ai_refactor(flag_name):
    def decorator(view_func):
        @wraps(view_func)
        def wrapper(request, *args, **kwargs):
            if getattr(settings, flag_name, False):
                logger.info('AI refactor path taken', extra={'path': request.path})
            return view_func(request, *args, **kwargs)
        return wrapper
    return decorator

@ai_refactor('USE_AI_REFACTOR')
def order_list(request):
    pass
```

Note that the decorator above logs the path but does not select between two implementations. To actually compare paths, the flag must dispatch to two separate functions; otherwise the flag is decorative.

**Mirror traffic to a shadow deployment.** A service mesh or load balancer can duplicate live requests to a second deployment that runs the new code and discards the responses. The shadow deployment must not write to production data stores, or the comparison is invalid. Kubernetes manifests for this are straightforward: a second Deployment with the new image, a Service selecting it, and a mirroring rule on the ingress or mesh.

**Add a cost check to CI.** A cost-estimation tool run against your infrastructure-as-code can report the delta between the current branch and the main branch. Set a threshold — 20% is a common choice — and fail the build above it. This catches capacity changes that would otherwise only appear on the bill.

The rule that ties these together: no AI-assisted refactor reaches production without a comparison against the previous behaviour and a documented rollback path. The rollback path should be a revert of the code change, not a database migration.

## Decision checklist

Use this when reviewing an AI-generated refactor that touches data access, caching, or a managed service.

- Does the change alter the number of database round-trips per request? If yes, measure the count before and after.
- Does the change remove a cache read from the path? If yes, check the hit ratio and decide whether the cache should be reinstated.
- Does the change alter the shape of the workload — batching, fan-out, or fan-in? If yes, check pool size, transaction timeout, and provisioned capacity.
- Does the change cross a synchronous/asynchronous boundary? If yes, confirm the product's latency contract and update timeouts.
- Does the change touch a managed service's access pattern? If yes, confirm the capacity model for the new pattern.
- Is there a feature flag and a rollback plan? If not, do not ship.

A short table of the failure modes and how to confirm each:

| Symptom | Likely cause | How to confirm | Reversible by config? |
|---|---|---|---|
| Cache hit ratio collapses | Cached read path replaced by direct read | Cache statistics before and after | Yes, if the cache still exists |
| Write latency rises, CPU flat | Pool exhaustion after batching change | Connection count vs pool size | Yes, with a redeploy |
| Statement cancellations | Long transaction exceeds timeout | Error logs correlated with new path | Yes, by chunking or raising timeout |
| Capacity throttling | Point read replaced by a query | Capacity metrics per request | Yes, if the access pattern allows |
| Function timeouts | Async rewrite without timeout change | Function duration vs configured timeout | Yes, with a redeploy |

## When the cause is not obvious

If the bill remains elevated after the cache, pool, and environment checks, bisect the change set. Find the commit range that introduced the refactor and test each commit against a cost or load metric in staging. `git bisect` automates the search when you can express the condition as a pass/fail test — for example, whether database connections exceed a threshold during a fixed load run.

One further possibility is observability drift. An AI rewrite can inline a function that contained the only logging or metrics call on a hot path. The symptom is missing traces or absent data in your metrics backend, and the underlying load is invisible rather than reduced. Confirm by adding a temporary counter or log line to the hot path and comparing its volume before and after the change. If the volume is unchanged but the traces are missing, the change removed instrumentation, not load.

The most disruptive case is a data model change — a rewrite that alters cardinality or key structure. That is the only category here that may require downtime to reverse, which is why it belongs behind a flag and a migration plan.

## One action for the next 30 minutes

Pick the single endpoint most likely to have been touched by an AI refactor in the last two weeks. Run it 100 times against staging, record the query count per request and the cache hit ratio, then compare both against the same measurements from before the refactor. If the query count has risen or the hit ratio has fallen, you have found the cause of the bill change and can decide on a fix before the next billing cycle closes.
