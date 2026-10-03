# Designing Alert Triage That Suppresses Noise Safely

## Why alert noise is a design problem, not a discipline problem

Most alerting stacks treat every metric breach as an incident. A threshold fires, a notification is sent, and a human decides whether it mattered. That works until the number of breaches grows faster than the number of humans, at which point the pager becomes a random sampler of your infrastructure.

The confusion is rarely technical. Teams conflate coverage with safety: more rules feel safer until the volume of notifications buries the signals that matter. A metric breach is just data. It becomes an alert only when the breach crosses a defined risk threshold, and it becomes a page only when the risk is real and sustained.

The design goal is not "fewer alerts." It is "every page is actionable, and every suppressed event is still recorded." Those two properties together are what let an on-call engineer trust the pager.

## A four-class mental model

Think of an emergency department triage nurse. The nurse does not page a surgeon for every elevated temperature. They check the trend, the patient's history, and the protocol. An alert pipeline can do the same by classifying each breach into one of four shapes before deciding what to do with it.

1. **Spike** — a sudden jump well above baseline that returns to normal quickly. Common causes are upstream DNS, CDN, or cache stampede events. Response: record it, do not page.
2. **Drift** — a gradual shift sustained over a longer window, still inside the SLO but directionally bad. Common causes are config changes and slow leaks. Response: record it, page only if the SLO is at risk.
3. **Noise** — a breach with no measurable downstream impact. Response: record only.
4. **Failure** — a sustained breach that will exhaust the error budget within the burn-rate window, or already has. Response: page immediately.

The classification is the whole trick. Once each breach carries a class label, routing, suppression, and reporting all become queries over that label rather than ad-hoc human judgment.

## Worked example: classifying a cache CPU spike

Consider a regional cache cluster that shows a CPU spike lasting under a minute. The raw metric looks alarming in isolation:

```
redis_cpu_user_seconds_total{cache_cluster="sgp-cache-01",quantile="0.99"} 4.2 1678901234
```

Step 1 — classify the shape. The value jumped and recovered within the window. That is a spike, not a drift. The next question is whether it had downstream impact.

Step 2 — establish a baseline. A daily job computes rolling percentiles per cluster over a 30-day window. Suppose the 99th percentile for this cluster is normally 1.1 CPU-seconds. The observed 4.2 is roughly 3.8× baseline. That ratio, plus the recovery time, is what the classifier keys on.

Step 3 — check downstream impact. The API's p99 latency stayed inside its SLO target during the spike. If the SLO target is 150 ms and the observed value stayed below it, there is no burn to attribute to this event. The spike is real but has no user-visible consequence.

Step 4 — label and route. A recording rule attaches a `severity` label, and the routing layer sends `spike` to a log sink rather than a pager. The event is written to a time-series table with columns for timestamp, service, severity, upstream source, and downstream impact, so a dashboard can render it as a low-priority marker.

The outcome is that the event is visible in the morning review, attributable, and closed without waking anyone. The important property is that nothing was discarded — the event exists, it is queryable, and a human can audit the suppression decision later.

### How to measure whether this actually helps

Do not trust a narrative about reduced pages. Instrument the pipeline and compare before and after over the same window length:

- Count pages fired per rotation from your paging provider's API or export.
- Count total alert evaluations and total breaches from the alerting engine's own metrics.
- Count suppression decisions by class from the triage table.
- Sample suppressed events weekly and label each one manually as "correctly suppressed" or "should have paged." This gives you a precision estimate that is grounded in review, not assertion.

A suppression system with no manual audit is a system that will eventually hide a real failure. The audit is not optional.

## Where the four-class model connects to tools you already use

If you have used composite alarms in a managed monitoring service, you have already touched triage: composite alarms combine multiple conditions before deciding to notify. The difference here is granularity. Composite alarms still treat a breach as a potential page; this model labels the breach first so routing can decide.

If you have worked with SLO burn-rate math — the approach described in the Google SRE Workbook's chapter on alerting on SLOs — you already know that not every breach is an incident. This model automates the classification step so humans do not have to make the same call repeatedly at 3 a.m.

If you have tuned Prometheus relabeling, you know labels drive routing. Extend that idea: labels also drive suppression. A relabel stage that injects `severity` turns an existing alert rule into a triage-aware rule with no change to the underlying query.

## Common misconceptions, corrected

**"Suppressed alerts disappear."** They should not. A suppressed alert is written to durable storage with its class label and the reason for suppression. You can query it for patterns, and you can prove after the fact that a suppression decision was correct. If your suppression path drops data, fix that before adding more rules.

**"Triage requires machine learning."** Simple heuristics cover most cases: a spike is a multiple of baseline that recovers within a bounded window; a drift is a sustained shift over a longer window; noise is a breach with no downstream SLO impact. These are arithmetic, not inference. Start with arithmetic and only add complexity when you can show it improves the audit numbers.

**"If we only page failures, we will miss incidents."** Spikes and drifts are still logged. The distinction is about who gets woken, not about what gets recorded. An incident that starts as a spike will become a drift or a failure if it persists, and the classifier will re-evaluate it on the next evaluation interval.

**"This needs a separate system."** The minimal version is one relabeling stage on existing alert rules plus one table for triage records. The routing layer can be your existing alert router with an added match on the severity label.

## Advanced layer: probabilistic suppression and circuit breakers

Once the four classes are stable, two additions reduce noise further without hiding failures.

**Probabilistic suppression** uses historical frequency to compute a suppression score per service and class. If the score is below a threshold, the alert is logged but not paged. The score is recomputed on a schedule from the triage table:

```sql
SELECT
  service,
  severity,
  COUNT(*) / 30.0 AS daily_freq,
  1 - (COUNT(*) / 30.0) AS p_suppress
FROM alert_logs
WHERE ts >= NOW() - INTERVAL '30 days'
GROUP BY service, severity
HAVING COUNT(*) > 3;
```

The arithmetic here is straightforward and worth stating explicitly: if a service produces 6 events of a given class in 30 days, the daily frequency is 6 / 30 = 0.2, and the naive suppression score is 1 - 0.2 = 0.8. The threshold you pick determines how aggressive the suppression is. Pick it from the audit data, not from intuition.

**Circuit breakers** prevent alert storms during a sustained outage. When a service is already in a critical state, new spikes and drifts for that service are suppressed until the breaker expires. The breaker state lives in a key-value store with a TTL, set atomically:

```lua
-- set a breaker for a service with a TTL, atomically
local key = KEYS[1]
local ttl = tonumber(ARGV[1]) or 1800
redis.call('SET', key, '1', 'EX', ttl, 'NX')
```

The critical detail is the exception. A breaker must never suppress a failure-class alert, because the whole point of the breaker is to reduce noise, not to hide the incident that caused the noise. Encode that exception in the breaker logic:

```lua
local key = KEYS[1]
local ttl = tonumber(ARGV[1]) or 1800
local severity = ARGV[2]

if severity == 'failure' then
  return 0
end

return redis.call('SET', key, '1', 'EX', ttl, 'NX')
```

**Ownership tags** complete the picture. When an alert fires, it should page the owning team, and the breaker should be scoped per service so a single upstream outage produces one suppression decision rather than one per dependent service.

```yaml
routing:
  sgp-cache-01:
    team: infra-sgp
    escalation: infra-sgp-lead
    breaker_ttl: 1800
```

## Failure modes to design against

**The breaker race condition.** A breaker fires for a service, and minutes later a genuine failure occurs in a downstream dependency. If the breaker suppresses the failure alert, the outage is hidden. The mitigation is the severity exception shown above, plus a test that asserts a failure-class alert always routes to the pager regardless of breaker state.

**Cross-service duplication.** A single upstream fault can trigger alerts across many services. If each alert carries only its own service label, reports will count the same root cause many times. A synthetic root-cause label assigned during relabeling lets reports aggregate by cause rather than by symptom:

```yaml
- source_labels: [__address__, loadbalancer]
  separator: ':'
  regex: (.+);(.+)
  target_label: root_cause_id
  replacement: 'lb_upstream_5xx'
```

**Timezone drift in rolling windows.** A daily job that computes `ts >= NOW() - INTERVAL '30 days'` can silently skip an hour when daylight saving time changes. The fix is to make the window explicit in UTC:

```sql
WHERE date_trunc('day', ts AT TIME ZONE 'UTC')
      >= date_trunc('day', NOW() AT TIME ZONE 'UTC') - INTERVAL '30 days'
```

Pair that with a data-completeness check that flags any day with fewer than 23 hours of data, so a gap in the triage table cannot quietly change suppression behavior.

**Suppression that never expires.** A suppression rule added for a one-off event should have an expiry. Without one, the rule outlives the reason for it and becomes an invisible gap in coverage. Treat suppression rules as code: reviewed, versioned, and dated.

## Quick reference

| Class   | Trigger                                    | Response                | Recorded as              |
|---------|--------------------------------------------|-------------------------|--------------------------|
| Spike   | Multiple of baseline, recovers quickly     | log, do not page        | triage row, severity tag |
| Drift   | Sustained shift over a longer window       | log, page if SLO at risk| triage row, severity tag |
| Noise   | Breach with no downstream impact           | log only                | triage row, impact flag  |
| Failure | Burn rate exceeds the critical threshold   | page immediately        | triage row, page event   |

## Integration sketch

The pieces fit together with a small amount of glue. A recording rule classifies breaches and attaches a severity label:

```yaml
groups:
  - name: triage
    rules:
      - record: alert:cache_cpu:severity
        expr: |
          (redis_cpu_user_seconds_total{quantile="0.99"}
           / on(cache_cluster) group_left()
           avg_over_time(redis_cpu_user_seconds_total{quantile="0.99"}[30d]))
          > 3
        labels:
          severity: spike
```

A table stores the triage record so dashboards can render it without generating pages:

```sql
SELECT
  $__timeGroup(ts, '1m') AS time,
  severity,
  COUNT(*) AS count
FROM alert_logs
WHERE $__timeFilter(ts)
  AND service = 'sgp-cache-01'
GROUP BY 1, 2
ORDER BY 1;
```

And the routing layer consults the breaker before deciding to page:

```python
import redis

r = redis.Redis(host='breaker-store', port=6379, decode_responses=True)

def should_suppress(service: str, severity: str) -> bool:
    result = r.eval(
        script=open('breaker.lua').read(),
        numkeys=1,
        keys=[f'service:{service}'],
        args=[1800, severity],
    )
    return bool(result)
```

## How to evaluate the system honestly

Avoid the temptation to report a headline percentage. Instead, track four numbers over matched windows:

- Pages per on-call shift, from the paging provider.
- Suppressed events per class, from the triage table.
- Manual audit precision: of the suppressed events you sampled, what fraction were correctly suppressed.
- Missed incidents: failures that were suppressed and later confirmed as real.

The fourth number should be zero. If it is not, the classifier or the breaker exception is wrong, and no amount of reduction in page volume compensates for a hidden outage.

## FAQ

**What if a spike does cause a downstream outage?** Then it stops being a spike. If downstream metrics breach the SLO, the event is reclassified as a drift or failure on the next evaluation, and the failure path pages. The classifier must re-evaluate, not decide once and forget.

**Do we need to rewrite all our alerts?** No. Start with one service, add a severity label via relabeling, and record triage events. After a few weeks you will have enough data to see which rules are candidates for suppression and which are load-bearing.

**How do we handle multi-service incidents?** Assign a root-cause label during relabeling and aggregate reports by that label rather than by service. Route the page to the owning team, and scope the breaker per service so one upstream fault does not produce one page per dependent.

**What does it cost to store triage records?** The storage cost is a function of event volume and row width. Estimate it as rows per month multiplied by average row size, then compare that to the cost of an unnecessary page in engineer time. Do the arithmetic with your own numbers rather than assuming a figure.

**How do we know a suppression rule is still correct?** Audit it. Sample suppressed events weekly, label them, and retire any rule whose precision drops. A suppression rule with no audit is a coverage gap waiting to be discovered during an incident.

## One thing to do in the next 30 minutes

Open your largest Prometheus rule file, add a recording rule that computes the ratio of the current value to a 30-day average for one high-volume metric, and attach a `severity` label based on that ratio. Validate it with `promtool check rules rules/*.yml`, deploy to staging, and watch the new label appear in your alerting UI. You will immediately see which existing alerts would have been classified as spikes — and you will have the beginning of the data needed to justify suppressing them.
