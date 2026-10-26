# 3 skills to outrun AI salary cuts

AI coding tools have made scaffolding cheap. A CRUD dashboard, a GraphQL schema, and a first pass at tests can appear in an afternoon. What they rarely produce is the set of production safeguards that determine whether a system survives contact with real traffic: bounded connection pools, retry logic that does not amplify outages, and cache headers that stop error responses from being stored and replayed.

The gap matters commercially. When the visible work of writing code gets cheaper, the value of an engineer shifts toward reducing operational risk: fewer outages, fewer support tickets, fewer silent payment failures. This article covers three failure modes that show up repeatedly in AI-scaffolded applications, the code that fixes each one, and how to verify the fix with measurements rather than assurances.

## Why fast output is not the same as safe output

A language model generates the most probable continuation of the code it has seen. That corpus is dominated by examples that compile and demonstrate a concept, not examples that survive a traffic spike. The result is code that is correct in the narrow sense and fragile in the broad one.

A junior-level output is code that runs. A senior-level outcome is code that degrades predictably when a dependency fails, that does not exhaust a shared resource under load, and that does not turn a transient upstream error into a sustained outage.

Three safeguard categories cover a large share of the gap between those two states:

1. **Bounded resource use.** Database connections, file handles, and worker threads are finite. Code that acquires them without a ceiling will eventually hit that ceiling.
2. **Failure isolation.** Retrying a failing dependency without backoff or a circuit breaker converts one upstream problem into a client-facing outage.
3. **Response hygiene.** Caching an error response, or letting a cold start produce one, turns a one-second blip into minutes of visible downtime.

The sections below address each in turn. Every code sample is runnable, and every verification step is a command or a metric rather than a claim.

## Safeguard 1: Bound your database connection pool

**Symptom.** The application performs acceptably in development and degrades or stalls under concurrent production traffic. Users report timeouts that do not reproduce locally.

**Cause.** An unbounded or default-sized connection pool. Many managed Postgres providers set a modest default connection limit on the instance, and application-side pools often inherit a small default as well. A route handler that opens a connection per request, or a pool configured with a low `max`, will queue or fail once concurrent demand exceeds the limit.

The arithmetic is worth doing explicitly. Suppose a route handler issues three sequential queries per request, each taking 30 ms. That is 90 ms of connection-held time per request. With a pool of 20 connections, the theoretical ceiling is roughly 20 divided by 0.09 seconds, or about 220 requests per second, assuming perfectly uniform arrival. Real traffic is bursty, so the effective ceiling is lower. If each request instead holds a connection for 300 ms because of a slow query, the ceiling drops to about 66 requests per second. Those figures are illustrative arithmetic from the stated assumptions, not measured results; the point is that pool size and query latency jointly determine throughput.

**Fix.** Configure an explicit pool ceiling, set timeouts, and expose pool depth as a metric.

```javascript
// Node.js with pg
const { Pool } = require('pg');

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  max: 20,                       // keep below the database's own connection limit
  idleTimeoutMillis: 30000,      // close idle clients after 30s
  connectionTimeoutMillis: 2000, // fail fast instead of queueing forever
});

pool.on('error', (err) => {
  console.error('idle client error', err.message);
});
```

The `max` value should be derived, not guessed. A workable starting rule is to keep the application pool below the database's configured connection limit, leaving headroom for administrative connections and migrations. If the database allows 100 connections, a pool of 20 to 40 is a reasonable starting point for a single application instance. Multiply by the number of application instances and compare against the database limit before deploying.

Expose pool state so the ceiling is observable:

```javascript
app.get('/health', async (req, res) => {
  let dbOk = false;
  try {
    await pool.query('SELECT 1');
    dbOk = true;
  } catch (err) {
    console.error('health check failed', err.message);
  }

  res.status(dbOk ? 200 : 503).json({
    status: dbOk ? 'ok' : 'degraded',
    pool_size: pool.totalCount,
    pool_idle: pool.idleCount,
    pool_waiting: pool.waitingCount,
  });
});
```

`pool.waitingCount` is the important number. It is the count of callers blocked waiting for a connection. A sustained non-zero value means demand exceeds capacity, and it is the earliest reliable signal that the pool is the bottleneck.

**How to measure it.** Two measurements matter: pool saturation under load, and query latency. Drive concurrent traffic with a load tool and record `pool_waiting` alongside response latency. A simple loop using a load-testing tool or a shell loop against a staging endpoint will surface saturation if it exists:

```bash
# 200 requests, 20 at a time, against a staging endpoint
seq 1 200 | xargs -P 20 -I{} curl -s -o /dev/null -w "%{http_code} %{time_total}\n" \
  https://staging.example.com/api/users
```

Compare the latency distribution at low concurrency against the distribution at high concurrency. If p95 latency grows faster than concurrency, and `pool_waiting` is above zero during the run, the pool is the constraint. Raising `max` should move the knee of that curve; if it does not, the bottleneck is elsewhere, most likely a slow query.

## Safeguard 2: Retries that do not amplify failure

**Symptom.** A dependency has a brief outage. Your service returns errors long after the dependency recovers, or the dependency's rate limiter starts rejecting your traffic.

**Cause.** Naive retry logic. A loop that retries immediately on failure multiplies load on a service that is already struggling. If a hundred clients each retry five times with no delay, a one-second blip becomes five hundred requests arriving in the same window. Many upstream APIs respond to that pattern with rate limiting, which extends the outage.

**Fix.** Combine exponential backoff with a circuit breaker. Backoff spaces retries out; the breaker stops retrying entirely once the failure rate crosses a threshold, giving the dependency room to recover.

```javascript
import CircuitBreaker from 'opossum';
import pRetry from 'p-retry';

async function callUpstream() {
  const res = await fetch('https://api.example.com/v1/events', {
    headers: { Authorization: `Bearer ${process.env.UPSTREAM_KEY}` },
  });
  if (!res.ok) throw new Error(`upstream error: ${res.status}`);
  return res.json();
}

const breaker = new CircuitBreaker(callUpstream, {
  timeout: 5000,
  errorThresholdPercentage: 50,
  resetTimeout: 30000,
  volumeThreshold: 5, // do not trip on a tiny sample
});

breaker.on('open', () => console.warn('breaker open'));
breaker.on('halfOpen', () => console.warn('breaker half-open'));
breaker.on('close', () => console.info('breaker closed'));

const callWithRetry = () =>
  pRetry(() => breaker.fire(), {
    retries: 5,
    minTimeout: 100,
    maxTimeout: 5000,
    factor: 2,
  });
```

Two parameters deserve attention. `volumeThreshold` prevents the breaker from tripping on a single failure in a low-traffic window. `resetTimeout` controls how long the breaker stays open before allowing a trial request; too short and the dependency is hammered again, too long and recovery is delayed for your users.

The fallback path matters as much as the breaker. When the breaker is open, the caller should receive a defined response, not an unhandled rejection:

```javascript
async function getEventsWithFallback() {
  try {
    return await callWithRetry();
  } catch (err) {
    // serve last known good data or a degraded response
    return { data: [], degraded: true, reason: 'upstream_unavailable' };
  }
}
```

**How to measure it.** Measure two things: the latency cost of the breaker when everything is healthy, and the behaviour when the upstream fails. For the first, record p50 and p95 latency for a fixed request volume before and after introducing the breaker. The overhead of the breaker itself is small; the observable change should be within noise. For the second, point the client at a stub that returns 503 on demand and confirm that the breaker opens, that requests fail fast rather than hanging, and that the fallback path returns a usable response. The breaker should open after the configured error threshold is crossed and close again after a successful trial request following `resetTimeout`.

## Safeguard 3: Stop caching error responses

**Symptom.** A brief origin failure becomes minutes of downtime for all users, including users who never hit the failing instance.

**Cause.** A CDN or edge cache storing a 5xx response. Many caches do not store 5xx by default, but misconfiguration, a proxy in front of the origin, or an origin that returns a 200 with an error body can all produce a cached failure. Cold starts on serverless platforms make this worse: the first request after a scale-to-zero event can be slow enough to time out, and if that timeout is cached, subsequent requests receive it too.

**Fix.** Set explicit cache headers on API responses, and ensure error responses are never stored.

```javascript
// Next.js route handler
import { NextResponse } from 'next/server';

export async function GET() {
  try {
    const data = await loadData();
    return NextResponse.json(data, {
      headers: {
        'Cache-Control': 'private, no-store, must-revalidate',
      },
    });
  } catch (err) {
    console.error('route failed', err.message);
    return NextResponse.json(
      { error: 'temporarily_unavailable' },
      {
        status: 503,
        headers: {
          'Cache-Control': 'no-store, no-cache, must-revalidate',
          'Retry-After': '5',
        },
      }
    );
  }
}
```

The `Retry-After` header is a signal to well-behaved clients and crawlers that the failure is transient. It does not prevent a misconfigured intermediary from caching the response, which is why the `no-store` directive matters.

If an intermediary sits in front of the origin and insists on caching, a small edge function can rewrite headers on error responses before they are stored:

```javascript
// edge worker: strip cacheability from 5xx responses
export default {
  async fetch(request) {
    const response = await fetch(request);

    if (response.status >= 500) {
      const headers = new Headers(response.headers);
      headers.set('Cache-Control', 'no-store, no-cache, must-revalidate');
      headers.delete('Expires');
      return new Response(response.body, {
        status: response.status,
        statusText: response.statusText,
        headers,
      });
    }

    return response;
  },
};
```

**How to measure it.** Inspect the actual headers returned by the edge, not just the origin. A request that passes through a CDN can be cached even when the origin sends `no-store` if the CDN is configured to override it. Check the cache status header your provider exposes, and confirm that a forced 5xx response is not served from cache on a subsequent request:

```bash
# first request should miss, second should also miss for a 5xx
curl -sI https://example.com/api/health | grep -i -E 'cache-control|cache-status|age'
```

For cold-start mitigation, the goal is to keep the function warm enough that the first request after a quiet period does not time out. A scheduled request to a lightweight health endpoint at an interval shorter than the platform's idle timeout achieves this. The interval depends on the platform; check the documented idle timeout for the runtime in use and schedule accordingly.

## A production readiness checklist

The three safeguards above are the highest-value items, but they are not the only ones. The checklist below is a starting point for a review before any deployment that will receive real traffic.

| Check | What to confirm | Why it matters |
|-------|-----------------|----------------|
| Connection pool bounded | `max` set and below the database limit | Prevents connection exhaustion under load |
| Pool depth observable | `waitingCount` exported as a metric | Detects saturation before users do |
| Query latency measured | p95 recorded per route | Distinguishes pool limits from slow queries |
| Retries use backoff | Non-zero delay, capped retries | Avoids amplifying upstream failures |
| Circuit breaker configured | Threshold and reset timeout set | Fails fast during sustained outages |
| Fallback path defined | Degraded response, not an exception | Keeps the UI usable during outages |
| API responses uncacheable | `no-store` on all API routes | Prevents cached errors from persisting |
| Error responses uncacheable | Verified at the edge, not just origin | A CDN can override origin headers |
| Cold start mitigated | Warm-up schedule shorter than idle timeout | Avoids first-request timeouts |
| Secrets not logged | Log output reviewed for credentials | Prevents credential leakage |

## Failure modes that follow these fixes

Once the three primary issues are addressed, a second tier of problems tends to surface. Each is worth recognising before it costs a client relationship.

- **Connection leaks.** A code path that acquires a client and returns early on an error without releasing it will exhaust the pool slowly. The symptom is a pool that saturates at low traffic. The fix is `try/finally` around every acquisition, or a library that manages release automatically.
- **Breaker stuck open.** If `resetTimeout` is long and the health check that closes the breaker is itself failing, the breaker never closes. Confirm that the half-open trial request exercises the same path as normal traffic.
- **Cache stampede.** A cold cache plus a burst of traffic sends every request to the origin simultaneously. Mitigate with request coalescing or a short jittered TTL on cached values.
- **Serverless memory exhaustion.** A function that loads a large dataset into memory per invocation will fail under concurrency. Raise the memory limit only after confirming the allocation is necessary; often the fix is to stream or paginate.
- **Credential leakage in logs.** Structured logging that serialises entire request or config objects will capture API keys. Redact known-sensitive field names at the logger level rather than at each call site.

## An escalation path when the fixes are not enough

If the application still fails under load after the safeguards above, the problem is usually visibility rather than code.

1. **Confirm the metrics exist.** Pool depth, error rate, and latency percentiles should be collected and retained. Without them, diagnosis is guesswork. A managed observability service or a self-hosted metrics stack both work; the requirement is that the data exists before the incident.
2. **Reproduce the failure outside production.** Kill an instance, or point the client at a stub that returns errors, and observe whether the system degrades as designed. A breaker that has never been exercised in a test is an assumption, not a safeguard.
3. **Ask a precise question.** When requesting help, include the exact error, the relevant configuration values, and the health endpoint output. A question like "why does my pool reach its limit at 50 concurrent users when each request holds a connection for 200 ms" can be answered; "why is my app slow" cannot.
4. **Consider simplifying the architecture.** A single well-monitored instance is easier to reason about than a distributed system with many moving parts. If the operational surface is larger than the team can maintain, reducing it is a legitimate engineering decision, not a retreat.

## Frequently asked questions

**Why does AI-generated code omit these safeguards?**

Because they are rarely present in the examples the models learn from. Tutorials and sample repositories optimise for demonstrating a concept, not for surviving production. A connection pool with a conservative default, a retry loop with no delay, and an API route with no cache headers all look correct in isolation. The omission is a property of the training distribution, not a defect in any particular tool.

**How much latency does a circuit breaker add?**

The breaker itself adds a small constant overhead per call, typically in the low single-digit milliseconds, dominated by the bookkeeping of recording success and failure. The measurable benefit appears during outages: without a breaker, a failing dependency causes callers to wait for the full timeout on every request; with one, calls fail immediately once the threshold is crossed. Measure both states on your own stack rather than relying on a general figure.

**What pool size should a small application use?**

Start from the database's configured connection limit, subtract headroom for administrative connections, and divide by the number of application instances. Then verify with a load test: raise the pool size and observe whether the knee in the latency curve moves. If it does not, the pool was not the constraint. There is no universal correct value.

**Why does the first request after a quiet period fail?**

Serverless platforms scale to zero, and the first request after that pays an initialisation cost. If that cost exceeds the platform's request timeout, the platform returns a 5xx. If a cache stores that response, the failure persists for the cache's TTL. The fix is twofold: keep the function warm with scheduled requests, and ensure error responses are not cacheable.

**Is it worth paying for an AI coding assistant?**

That depends on whether the output is reviewed. These tools reduce the cost of producing a first draft, which is genuinely useful. They do not reduce the cost of reviewing that draft for production readiness, and that review is where the risk lives. The economics work when the time saved on scaffolding exceeds the time spent auditing the result.

## The skills that hold their value

The three areas covered here — bounded resource use, failure isolation, and response hygiene — are not new. They predate the current generation of tooling by decades. What has changed is the ratio: the cost of producing code has fallen, so the share of an engineer's value that comes from producing it has fallen with it. What remains is judgement about how systems fail and what to do about it.

That judgement is not demonstrated by a checklist alone. It is demonstrated by measurements: a latency curve that shows where the pool saturates, a breaker that opens and closes as configured, a cache status header that confirms an error was not stored. Each of those is something you can produce in an afternoon, and each one is evidence that the system was designed rather than merely generated.

**Your next step, in the next 30 minutes:** open the file that creates your database pool, set an explicit `max` value below your database's connection limit, and add a `/health` endpoint that returns `pool.totalCount`, `pool.idleCount`, and `pool.waitingCount`. Deploy it to staging and run a concurrent load against a route that queries the database. If `waitingCount` is above zero during the run, you have found your first bottleneck and you have the measurement to prove it.
