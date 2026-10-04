# 5 ways to find hidden latency before production

## What "hidden latency" actually means

Most latency that reaches production is not invisible to tooling. It is invisible to *load*. A service that responds in 40ms at 100 requests per second can respond in 400ms at 2,000 requests per second because of connection pool exhaustion, lock contention, garbage collection, or a query plan that changes shape once a table grows. None of these show up in a functional test suite, and most of them do not show up in a staging environment that runs at 1% of production volume.

There is a second category: latency that is visible in aggregate but not attributable. A p99 graph climbing from 80ms to 130ms tells you something changed, but not what. Without per-dependency or per-function breakdown, the only way to find the cause is to guess and bisect.

The five approaches below target different parts of that problem. They are complements, not substitutes. The goal of this article is to describe what each one actually does, where it fails, and how to measure whether it is working for you.

## A note on measurement

Every claim about "this caught a regression" is only meaningful alongside a measurement method. Throughout this article, when a technique is described as effective, the implied measurement is:

- **Detection latency**: time from deploying a change that introduces a known regression to the pipeline failing or alerting. Measure it by injecting a synthetic regression (for example, a deliberate `sleep(50ms)` behind a feature flag) and timing the alert.
- **False positive rate**: fraction of pipeline runs that fail without a real regression present, over a representative window (a few weeks of normal commit traffic).
- **Added CI time**: wall-clock seconds added to the median pipeline run, measured before and after enabling the check.
- **Coverage**: which latency sources the technique can see (CPU, I/O, network, database, cross-service) and which it structurally cannot.

Without those four numbers, "we added latency testing" is not a claim you can act on.

## 1. Traffic replay with latency injection

**What it does.** Capture real request traffic at a boundary (a load balancer, a sidecar proxy, or a host interface), then replay it against a staging deployment while injecting delay into specific dependencies. The delay injection is usually done with `tc` on Linux, which adds delay at the network interface, or with a TCP proxy that sits between the service and its dependency.

**Why it works.** Replay reproduces the request *mix*, which is the part synthetic tests almost always get wrong. A benchmark that sends uniform requests will never exercise the one endpoint that fans out to nine downstream calls. Replay does.

**Where it fails.**
- If you replay a small fraction of traffic, you replay a small fraction of the request mix. Rare paths stay rare and stay untested.
- If the staging data volume is smaller than production, queries behave differently. A query that does a sequential scan over 100 rows in staging may use an index in production, or vice versa. Replay does not fix that; you need comparable data volume.
- Latency injection with `tc` is applied at the interface, so it delays *all* traffic on that interface, not one dependency. Per-dependency delay requires a proxy in the path.

**How to measure it.** Record a fixed window of traffic, note the request count and the distinct endpoint distribution, then replay. Compare the endpoint distribution in the replay against the recorded distribution; if they diverge, your replay is not representative.

```bash
# Record 60 seconds of TLS traffic on any interface
sudo tcpdump -i any -w /tmp/trace.pcap -G 60 -W 1 'tcp port 443'

# Replay a fixed request rate from a targets file
vegeta attack -duration=60s -rate=100 -targets=targets.json | vegeta report
```

To inject delay for the duration of the replay, apply `tc` to the relevant interface before starting `vegeta` and remove it after:

```bash
sudo tc qdisc add dev eth0 root netem delay 100ms
# ... run replay ...
sudo tc qdisc del dev eth0 root netem
```

Note that `tc` delay is applied in both directions only if you configure both sides; a single `qdisc` on one interface delays egress only. Verify with a `ping` or a simple request before trusting the number.

**Best fit.** Teams that already have a staging environment close enough to production that replay results transfer. If staging and production differ substantially in instance type, data volume, or dependency versions, replay results will mislead you.

## 2. Synthetic probing from multiple regions

**What it does.** Deploy a small probe (a Lambda, a container, or a managed synthetic monitoring service) in each region you serve, hitting a known endpoint on a fixed interval and asserting on status, latency, and payload.

**Why it works.** It is the cheapest way to detect regional or edge-level problems: DNS resolution differences, TLS handshake latency, CDN misconfiguration, a load balancer in one region routing to a degraded backend.

**Where it fails.** Probes test a fixed set of endpoints, usually the health check. A health check that returns a cached value will pass while the real request path is broken. A health check that only verifies process liveness will pass while every database call is timing out. The probe tells you the endpoint you chose is healthy; it says nothing about the endpoints you did not choose.

**How to measure it.** Track the probe's own latency distribution. If the probe's latency has high variance (for example, p50 20ms but p99 300ms), the probe itself is noisy and will produce false positives at tight thresholds. Baseline the probe for a week before setting any alert threshold.

```javascript
// A minimal synthetic canary handler.
// The exact runtime API differs between providers; the structure is the same:
// issue a request, assert on status and duration, throw on failure.
const https = require('https');

async function checkEndpoint(url, maxDurationMs) {
  const start = Date.now();
  const statusCode = await new Promise((resolve, reject) => {
    const req = https.get(url, (res) => {
      res.resume();
      resolve(res.statusCode);
    });
    req.on('error', reject);
    req.setTimeout(maxDurationMs, () => {
      req.destroy(new Error('timeout'));
    });
  });
  const duration = Date.now() - start;
  if (statusCode !== 200) {
    throw new Error(`unexpected status ${statusCode}`);
  }
  if (duration > maxDurationMs) {
    throw new Error(`latency ${duration}ms exceeded ${maxDurationMs}ms`);
  }
  return duration;
}

exports.handler = async () => {
  await checkEndpoint('https://api.example.com/healthz', 200);
};
```

**Best fit.** Multi-region services with an availability or latency commitment, used as a coarse availability signal rather than a latency regression detector.

## 3. CPU profiling diffs in CI

**What it does.** Capture a CPU profile of a representative workload before and after a change, then compare the two to find functions whose cumulative time increased.

**Why it works.** It attributes latency to a function. When p99 rises and the cause is a hot loop or an inefficient serialization path, a profile diff points at the line. Aggregate metrics cannot do that.

**Where it fails.** It only sees CPU time. If the regression is waiting on a socket, a lock, or a disk, the profile will show the waiting function with roughly the same CPU time as before, and the diff will be clean while latency increases. Profiling also requires a workload that exercises the changed code; a profile of a startup path will not show a regression in a request handler.

**How to measure it.** Run the same workload against both builds with the same input, capture profiles for the same wall-clock duration, and compare cumulative time per function. Report the delta in absolute milliseconds, not just percentage, because a 3% increase in a function called 200,000 times per request is a different problem from a 3% increase in a function called once.

```bash
# Capture a profile of a running process for 10 seconds.
# --pid targets a live process; --duration controls sampling window.
py-spy record --pid 1234 --duration 10 --format speedscope -o after.json
```

Comparing two profiles is a diff of the aggregated stacks. A rough approach is to convert each profile to a sorted list of `(function, cumulative_ms)` pairs and compare entries that appear in both. Treat this as a signal to investigate, not a pass/fail gate, until you have baselined its noise on your workload.

**Best fit.** CPU-bound services where the dominant cost is computation rather than I/O. For most request-serving backends, CPU is a minority of wall-clock time, so treat this as one lens among several.

## 4. Query plan diffing

**What it does.** Capture the execution plan for a set of representative queries before and after a schema or query change, then compare the plans.

**Why it works.** A plan change is a common cause of sudden latency regressions that no amount of application profiling will explain, because the time is spent inside the database, not in the application. A migration that adds a column, changes a type, or updates statistics can flip a plan from an index scan to a sequential scan.

**Where it fails.** Plans depend on data distribution and statistics, so a plan captured on a small test dataset may not match the plan production will choose. Plans also change for reasons unrelated to your commit: autovacuum, statistics updates, or a different parameter value. Diffing plans without controlling for those produces noise.

**How to measure it.** Capture plans with `EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)` on a dataset with realistic row counts and distribution, and compare the estimated cost *and* the actual rows read. A plan that looks the same but reads ten times as many buffers is a regression.

```sql
-- Capture a plan with actual execution statistics and buffer usage.
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
SELECT * FROM orders WHERE user_id = 123;
```

Store the JSON output per query per build, and diff the plan trees. The comparison that matters is on the actual node types and row counts, not the total estimated cost alone.

**Best fit.** Services where a large share of request latency is database time, and where the query set is stable enough to enumerate.

## 5. Dependency latency budgets

**What it does.** Instrument every outbound call (cache, queue, database, third-party API) and track its latency distribution. Define a budget per dependency and fail a deployment or alert when the budget is exceeded.

**Why it works.** It makes cross-service latency a first-class contract. If a client library silently opens a new connection per request, the dependency's p99 will rise, and the budget will catch it even though the application code did not change.

**Where it fails.** Budgets are only as good as their thresholds. Set them too tight and normal variance fails the build; set them too loose and they never fire. They also require instrumentation coverage: an uninstrumented dependency is invisible to the budget.

**How to measure it.** Before setting a threshold, collect the dependency's latency distribution for a period that includes normal peak traffic. Set the budget above the observed p99 with margin, then tighten it over time as you eliminate variance. Record the false positive rate at each threshold.

```yaml
# Example budget file. Thresholds must be derived from observed baselines,
# not chosen arbitrarily.
services:
  api-service:
    dependencies:
      cache:
        p99_ms: 8
        p95_ms: 4
      queue:
        p99_ms: 25
        p95_ms: 15
```

**Best fit.** Microservice architectures where latency is dominated by a small number of shared dependencies and where the team owns the instrumentation.

## Choosing between them

| Situation | First choice | Second choice | Notes |
|---|---|---|---|
| Latency dominated by database time | Query plan diff | Traffic replay | Requires realistic data volume for plans to be meaningful |
| CPU-bound service with tight p99 | CPU profile diff | Dependency budgets | Profile diff sees CPU only |
| Multi-region with availability commitment | Synthetic probes | Dependency budgets | Probes are availability signals, not latency regression detectors |
| Unknown cause, need attribution | Dependency budgets | Traffic replay | Budgets localize the slow dependency; replay reproduces the load |
| Small team, limited CI budget | Synthetic probes | Dependency budgets | Lowest setup cost, lowest coverage |

The honest summary: no single technique covers all latency sources. Dependency budgets and traffic replay together cover the widest range, but both require instrumentation and a staging environment that resembles production. Synthetic probes are cheap and shallow. Profile and plan diffs are narrow but precise when they apply.

## Common failure modes

**Replaying at the wrong scale.** A replay that runs at 10 requests per second against a service that handles 5,000 will not reproduce contention. The regression appears only above a threshold, and the replay is below it.

**Probing a cached endpoint.** A health check that returns a constant will pass regardless of backend health. Verify that the probe's response actually depends on the components you care about.

**Trusting a single profile.** A single profile is a sample. Compare distributions across multiple runs before concluding a function regressed.

**Setting budgets from intuition.** A budget chosen without a baseline will either fail constantly or never fire. Baseline first, then set the threshold.

**Ignoring the cost of the check itself.** Every added CI step has a cost in time and maintenance. A check that adds 30 seconds and fails once a quarter may not be worth it; one that adds 3 seconds and catches a regression per month usually is.

## What to do in the next 30 minutes

Pick one endpoint in your service that you know is on a hot path, and measure its latency under increasing load using a local load generator. Record p50, p95, and p99 at three load levels: well below your peak, at your peak, and slightly above. If p99 rises non-linearly as load increases, you have found a hidden latency source, and you now have a baseline to compare against the next time you change that code path.
