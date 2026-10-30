# Agent SLOs: measure what agents actually do

## Why request-shaped SLOs fail on loop-shaped systems

A background report generator, an async approval flow, a retry-driven sync job: none of these are request handlers. They are loops that pick up work, persist state, retry, and eventually either finish or get stuck. The standard SLO pair — p99 latency and 5xx rate — measures the wrong thing for them.

The reason is structural. A loop that is stuck still returns 200 OK from whatever endpoint triggers it. A worker that has crammed ten thousand jobs into its queue because the backoff policy never engaged still reports zero errors. A renderer that leaks memory and slows to a crawl still completes some fraction of its work, so the latency histogram barely moves. In each case the signals look healthy while the system delivers nothing.

The gap is not tooling. Prometheus, OpenTelemetry, and Grafana all handle counters and histograms fine. The gap is the definition of "good" when the unit of work is a job that takes minutes and can silently fail. This article builds SLOs that track outcomes — jobs that actually reached a terminal success state — rather than the signals emitted along the way.

## What you need before starting

The approach assumes a system with three properties:

- A durable work tracker: a Postgres table, a Redis Stream, an SQS queue, or anything else that records individual units of work and their state.
- At least one recurring or background process that consumes from it: a cron job, a systemd timer, a container worker, a scheduled function.
- A way to mark a unit of work as terminally done or failed — a status column, a counter, a metric label.

If any of those is missing, add it first. Outcome SLOs are impossible without a terminal state, because the whole point is to count completions against attempts.

The worked example throughout is a report generator: a scheduled process that picks queued report jobs, renders a document, delivers it, and records the result. The same structure applies to invoice approvals, nightly syncs, and message delivery.

## Model the job lifecycle explicitly

Before writing any metric, decide what states a job can occupy and which of them count as success. A minimal schema:

```sql
CREATE TABLE reports (
  id BIGSERIAL PRIMARY KEY,
  status TEXT NOT NULL CHECK (status IN ('queued','in_progress','done','failed')),
  queued_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  started_at TIMESTAMPTZ,
  finished_at TIMESTAMPTZ,
  attempts INT NOT NULL DEFAULT 0,
  max_queue_age_minutes INT NOT NULL DEFAULT 30,
  report_name TEXT NOT NULL
);

CREATE INDEX idx_reports_status_queued_at ON reports(status, queued_at);
```

Two design choices matter here.

First, `failed` is a real terminal state, not an absence of `done`. If the only way a job leaves the queue is by succeeding, then a crashed worker leaves rows in `in_progress` forever and the completion rate silently drifts upward because the denominator never grows. Explicit failure keeps the denominator honest.

Second, `max_queue_age_minutes` is per-row rather than global. Different report types can legitimately have different deadlines, and encoding the deadline next to the job means the reaper query needs no join and no configuration lookup.

The index on `(status, queued_at)` supports the pickup query `WHERE status='queued' ORDER BY queued_at LIMIT 1`. Whether to keep it is a real tradeoff: every index adds write cost to inserts and updates. For a queue that never exceeds a few hundred rows, a sequential scan is faster than the index lookup. For a queue that routinely holds thousands, the index pays for itself. Measure it rather than guessing: run `EXPLAIN (ANALYZE, BUFFERS)` on the pickup query with the index present and absent, at realistic queue depth, and compare the actual execution time.

## Pick up work safely

The pickup step is where most silent-failure bugs originate. The correct pattern is a single atomic statement that claims one row and marks it in progress, so two workers cannot claim the same job:

```python
from sqlalchemy import create_engine, text

engine = create_engine('postgresql://postgres:pass@localhost:5432/reports')

def claim_next_job(conn):
    return conn.execute(text(
        """
        UPDATE reports
        SET status='in_progress', started_at=now(), attempts=attempts+1
        WHERE id = (
            SELECT id FROM reports
            WHERE status='queued'
            ORDER BY queued_at ASC
            FOR UPDATE SKIP LOCKED
            LIMIT 1
        )
        RETURNING id, report_name, attempts
        """
    )).fetchone()
```

`FOR UPDATE SKIP LOCKED` is the important part. Without `SKIP LOCKED`, concurrent workers serialize on the same row and the second worker blocks. Without `FOR UPDATE`, two workers can both read the same `queued` row before either writes, and the job runs twice.

Note what this does *not* do: it does not guarantee the job finishes. A worker that claims a job and then dies leaves the row in `in_progress`. That is expected — the reaper described below handles it.

## Define the metric as a completion ratio

Two metric families carry the SLO:

```python
from prometheus_client import Counter, Histogram

report_jobs_total = Counter(
    'report_jobs_total',
    'Count of report jobs by terminal status',
    ['status']
)

report_job_duration_seconds = Histogram(
    'report_job_duration_seconds',
    'Duration of report jobs in seconds',
    buckets=[1.0, 3.0, 10.0, 30.0, 60.0, 120.0, 300.0]
)
```

The counter is labelled by *terminal* status only — `done` or `failed`. Do not increment it on `in_progress`. The histogram is secondary: it answers "how slow is the slow path," not "did the work happen."

The SLO is the fraction of jobs that reach `done` within their deadline. In PromQL, over a six-hour window:

```promql
(
  sum by (report_name) (increase(report_jobs_total{status="done"}[6h]))
  /
  sum by (report_name) (increase(report_jobs_total[6h]))
) * 100
```

Use `increase()` rather than `rate()` here. `rate()` returns a per-second value, which is fine for ratios but awkward when you want to reason about counts, and it silently extrapolates at window edges in ways that surprise people reading the dashboard. `increase()` gives the count of events in the window, which is what the SLO language ("95% of jobs") actually refers to.

A subtlety: this ratio counts *terminal events*, not jobs. A job that fails, is retried, and then succeeds contributes one `failed` and one `done`. That is usually the right behavior — you want to know what fraction of attempts succeed — but if you want per-job success, you need a separate gauge that tracks the current state of each job, or you need to emit the terminal event only once per job ID. Decide which question you are answering before you build the alert.

## Choose the window from the work cycle

The window must be long enough to contain a full cycle of the work, plus slack for the slowest legitimate path.

For a generator that runs every 15 minutes, a 6-hour window contains 24 cycles. That is enough that a single bad cycle does not trip the alert, but short enough that a sustained problem surfaces within the hour. For an hourly job, a 24-hour window is the natural choice. For a daily job, 7 days.

The failure mode to avoid is a window shorter than the cycle. With a 1-hour window on a job that runs hourly, the ratio oscillates between 0 and 100 as each cycle lands, and the alert fires on every cycle boundary.

The second failure mode is a window so long that detection is useless. A 30-day window on a 15-minute job will not move measurably until the problem has persisted for days.

## Handle the failure paths explicitly

Three failure paths account for most silent SLO drift.

**The process dies mid-job.** The row stays `in_progress` and no counter is incremented. The completion ratio does not drop, because the denominator did not grow either. Fix this with a reaper that fails stale in-progress rows:

```sql
UPDATE reports
SET status='failed', finished_at=now()
WHERE status='in_progress'
  AND started_at < now() - interval '2 hours';
```

**The job never gets picked up.** The row sits `queued` past its deadline. Fix with a queue-age reaper:

```sql
UPDATE reports
SET status='failed', finished_at=now()
WHERE status='queued'
  AND queued_at < now() - (max_queue_age_minutes || ' minutes')::interval;
```

Run both reapers on a schedule shorter than the alert window — every 10 minutes is typical. Each reaper should also increment `report_jobs_total{status="failed"}` for the rows it flips, otherwise the metric and the database disagree and the dashboard is lying.

**The exception path skips the metric.** This is the most common bug. Wrap the work so that every exit path increments exactly one counter:

```python
import time

def run_job(conn, job_id, report_name):
    start = time.time()
    try:
        render_report(report_name)
        with conn.begin():
            conn.execute(text(
                "UPDATE reports SET status='done', finished_at=now() WHERE id=:id"
            ), {'id': job_id})
        report_jobs_total.labels(status='done').inc()
    except Exception:
        with conn.begin():
            conn.execute(text(
                "UPDATE reports SET status='failed', finished_at=now() WHERE id=:id"
            ), {'id': job_id})
        report_jobs_total.labels(status='failed').inc()
        raise
    finally:
        report_job_duration_seconds.observe(time.time() - start)
```

The `finally` block matters: duration is recorded whether the job succeeded or failed, so the histogram reflects reality rather than only the happy path.

## A worked example: diagnosing a stuck loop

Suppose the completion ratio drops from 99% to 91% over six hours. The latency histogram is flat. Where do you look?

Step 1: split the ratio by failure mode. Query `increase(report_jobs_total{status="failed"}[6h])` and compare it to the reaper counts. If the reaper is producing most of the failures, the problem is throughput — jobs are queued but not being claimed. If the exception path is producing them, the problem is in the work itself.

Step 2: if it is throughput, check whether workers are alive. A worker that has exited cleanly leaves `in_progress` rows that the reaper eventually fails; a worker that is alive but slow leaves rows `queued`. The distinction is visible in the timestamps: `started_at IS NULL` for queued, `started_at` set but stale for in-progress.

Step 3: if it is the work, look at the duration histogram's tail. A shift in the 60–300 second bucket with no change in the lower buckets points at a slow dependency rather than a slow renderer. A shift in every bucket points at the host.

Step 4: check attempt counts. `SELECT attempts, count(*) FROM reports WHERE status='failed' GROUP BY attempts` distinguishes a transient dependency outage (many rows at `attempts=1..3`) from a permanently broken job (many rows at the retry ceiling).

This sequence is the payoff of the outcome SLO: the ratio tells you *that* something is wrong, and the per-status breakdown tells you *where* to look. The latency histogram alone would have shown nothing.

## Alert on sustained drops, not spikes

A single failed cycle is not an incident. Alert on the ratio staying below threshold for longer than one cycle:

```yaml
groups:
  - name: report-slo
    rules:
      - alert: ReportCompletionSLOViolation
        expr: |
          (
            sum by (report_name) (increase(report_jobs_total{status="done"}[6h]))
            /
            sum by (report_name) (increase(report_jobs_total[6h]))
          ) * 100 < 95
        for: 30m
        labels:
          severity: page
        annotations:
          summary: "Completion rate below 95% for {{ $labels.report_name }}"
```

The `for: 30m` clause is what separates this from a latency alert. It requires the condition to hold continuously, which filters out the noise of a single slow cycle.

Add a second, lower-severity rule for the reaper itself: if the reaper has not run in the last 20 minutes, the failure counts are stale and every other alert is unreliable.

## Testing the pipeline

Two layers of testing are worth the effort.

Unit-level: verify that each exit path increments exactly one counter. The metric objects expose their current value, so a test can assert on it directly:

```python
def test_failed_job_increments_failure_counter(db_session, monkeypatch):
    monkeypatch.setattr('report_agent.render_report', lambda name: (_ for _ in ()).throw(RuntimeError('boom')))
    seed_job(db_session, 'test.pdf')
    before = report_jobs_total.labels(status='failed')._value.get()
    with pytest.raises(RuntimeError):
        run_job(db_session, 1, 'test.pdf')
    after = report_jobs_total.labels(status='failed')._value.get()
    assert after == before + 1
```

Reaching into `_value` is a private API and will break across prometheus_client versions; for anything long-lived, expose a small helper that reads the counter through the public `collect()` interface instead.

End-to-end: seed a synthetic job on a schedule and assert that it reaches a terminal state within the deadline. Run this from outside the system under test — a separate scheduler, a separate host — so that a failure of the main worker does not also disable the check.

## Instrumenting the measurement itself

If you want to claim a completion rate, you need to know the measurement is trustworthy. Three things to instrument:

- Reaper runs: a counter incremented every time each reaper executes, with a timestamp gauge for the last run.
- Counter/DB divergence: a periodic reconciliation query comparing `count(*)` of terminal rows by status against the counter's increase over the same period. A gap means a code path is skipping the metric.
- Claim latency: the difference between `queued_at` and `started_at`, as a histogram. This is the leading indicator — it rises before the completion ratio falls.

To measure the divergence, run the reconciliation as a scheduled job and expose the absolute difference as a gauge. Alert if it is non-zero for more than one reconciliation period.

## A decision checklist

Before shipping an outcome SLO, confirm:

- Every job has exactly one terminal state, and failure is explicit.
- The completion counter is incremented on every terminal transition, including those made by reapers.
- The reaper runs more often than the alert window.
- The SLO window is at least several times the work cycle.
- The alert uses `for:` to require sustained violation.
- There is a reconciliation check between the counter and the durable state.
- There is a synthetic end-to-end check running outside the system under test.

If any of those is missing, the SLO will either miss real incidents or fire on noise. Both outcomes erode trust in the metric, and a metric nobody trusts is worse than no metric.

## FAQ

**What if the work has no natural terminal state?** Then it is not a job and this pattern does not apply. Streaming pipelines and long-lived connections need a different model — usually a freshness or lag metric rather than a completion ratio.

**Should retries count as separate attempts or as one job?** Decide based on the question. "What fraction of attempts succeed" wants attempts. "What fraction of user requests are eventually served" wants jobs. The counter design above supports the first; the second needs a per-job terminal event emitted only once.

**Can this work without Prometheus?** Yes. Any system that can store a counter and evaluate a ratio over a window works. The specific query language changes; the structure does not.

**How do I set the threshold?** Start by measuring the current rate for a week, then set the threshold slightly below the observed floor. Setting an aspirational threshold before you have a baseline produces constant alerts.

## Do this in the next 30 minutes

Pick one background process you own and answer a single question: for a unit of work in that system, what column or field, read today, tells you whether it succeeded? If the answer is "nothing," add a terminal status field and increment a counter on both the success and failure paths. That change alone converts an unmeasured loop into something you can put an SLO on.
