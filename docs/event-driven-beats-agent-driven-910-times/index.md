# Event-driven beats agent-driven 9/10 times

## The problem in one sentence

An event-driven system treats an incoming event as the trigger for a decision; an agent-driven system has a long-running process that decides for itself what work to do next. Both use queues and workers, which is why the two are so easy to confuse — and why teams frequently bolt an agent onto an event-driven core without noticing until latency spikes correlate with a schedule rather than with traffic.

The usual symptom is a periodic spike: response times jump every few minutes while request volume is flat. The trace shows a scheduled invocation fanning out into retries, each retry starting a fresh worker, each worker contending for the same row or the same lock. The instinct is to blame the database, then the queue, then the concurrency limit. The actual cause is architectural: a scheduler has been allowed to make decisions.

This article covers how to recognise the shift, how to reverse it, and when an agent-driven design is genuinely the right call.

## The distinction that matters

Event-driven:

```
event arrives -> handler decides what to do -> side effect
```

Agent-driven:

```
agent polls for work -> agent decides what to do -> side effect
```

The difference is not "queues versus cron." It is *where the decision lives*. In the event-driven case, the decision is a pure function of the event payload and current state, evaluated once per event. In the agent-driven case, the decision is made by a process that maintains its own view of what remains to be done — which means it carries state, and state means concurrency control.

Three properties follow directly from that:

- **Event-driven handlers are naturally concurrent.** Each event is independent. Retries are safe when handlers are idempotent. Side effects are append-only or keyed by event ID.
- **Agent-driven processes are sequential by construction.** The agent is the single source of truth for "what next," so two agents running at once must coordinate.
- **Mixing the two produces contention.** Events and agents compete for the same rows, locks, and connection pool. Correctness starts to depend on execution order rather than on event arrival.

The trap is that the agent is usually introduced for a reasonable-sounding reason: cleanup, reporting, or retrying failed events. Each of those feels safer when it is explicit and scheduled. In practice each one imports state and locking into a system that did not need either.

## Failure mode: the feedback loop

A concrete and common shape:

1. A scheduled job fires and invokes a function.
2. That function publishes an event to a queue.
3. A consumer of that queue is the same function, or writes to a row the scheduled job is about to read.
4. The scheduled job's read is blocked by the consumer's write, or vice versa.
5. The scheduled job times out, retries, and the cycle repeats.

Nothing here is a load problem. The latency spike is contention between a polling loop and an event stream, and it will appear on a fixed cadence regardless of traffic. If your p99 correlates with a cron expression rather than with requests per second, this is the shape you are looking at.

A second failure mode is the orphan-sweeper. An event fails to process; rather than letting the queue retry, someone writes a periodic scan for "events older than N minutes with no success timestamp," marks them as retrying, and republishes. This introduces a race: two scans can select the same row, the first marks it, the second skips it, and the first one's publish is lost. The result is duplicate or dropped events that corrupt downstream aggregates — and because the sweeper is scheduled, the corruption arrives in batches.

A third is cleanup-by-agent in ephemeral environments. A job deletes old namespaces, pods, or volumes on a timer, and uses a leader-election lock stored in a resource that the job itself is about to delete. The lock disappears before it is released, the agent hangs, and resources are orphaned. The platform almost always has a better mechanism.

## Fix 1: stop polling, use the queue's retry

The most common cause of accidental agent behaviour is a polling loop added "to ensure delivery." If you are using a managed queue, it already has retry semantics.

- Amazon SQS: visibility timeout plus a redrive policy to a dead-letter queue. `maxReceiveCount` on the redrive policy controls how many receives before a message is moved.
- RabbitMQ: dead-letter exchanges, with optional per-message TTL.
- Redis Streams: consumer groups with `XACK`, `XPENDING`, and `XAUTOCLAIM` for reclaiming messages from dead consumers.

The fix is to delete the polling job and configure the queue. Nothing else changes.

The scheduler's only legitimate job is to emit an event. It should not decide anything. If your platform offers a managed scheduler that invokes a target directly, use it rather than a cron entry that shells out to a script.

```python
import json

import boto3

scheduler = boto3.client("scheduler")


def schedule_report_job(report_id: str) -> dict:
    """Create a one-shot schedule that invokes a Lambda with a fixed payload.

    The schedule carries no decision logic: it emits the same event the
    application would emit if a user had requested the report.
    """
    return scheduler.create_schedule(
        Name=f"report-{report_id}",
        ScheduleExpression="rate(10 minutes)",
        FlexibleTimeWindow={"Mode": "OFF"},
        Target={
            "Arn": "arn:aws:lambda:us-east-1:123456789012:function:report-generator",
            "RoleArn": "arn:aws:iam::123456789012:role/eventbridge-scheduler-role",
            "Input": json.dumps({"report_id": report_id}),
        },
    )
```

Note what the payload contains: the identity of the thing to act on, not an instruction about how many times to retry or which rows are stale. The handler derives everything else from state.

## Fix 2: let the dead-letter queue be the orphan handler

If you currently republish failed events from a database scan, replace that scan with a dead-letter queue plus a consumer.

The sequence:

1. The consumer reads from the source queue.
2. On failure, the message is not acknowledged and becomes visible again after the visibility timeout.
3. After `maxReceiveCount` receives, the queue moves the message to the DLQ automatically.
4. A separate consumer reads the DLQ and publishes a failure record for human review.

There is no polling for orphans, no status column, and no race, because the queue owns the retry counter.

```python
import json

import boto3

sqs = boto3.client("sqs")
sns = boto3.client("sns")

DLQ_URL = "https://sqs.us-east-1.amazonaws.com/123456789012/dlq"
FAILURE_TOPIC_ARN = "arn:aws:sns:us-east-1:123456789012:analytics-failures"


def drain_dlq() -> None:
    """Move messages from the DLQ to a topic for human review.

    This handler is invoked by the queue itself (event source mapping),
    not by a timer. It never decides that a message is orphaned; the
    queue's redrive policy already made that decision.
    """
    response = sqs.receive_message(QueueUrl=DLQ_URL, MaxNumberOfMessages=10)
    for msg in response.get("Messages", []):
        event = json.loads(msg["Body"])
        sns.publish(TopicArn=FAILURE_TOPIC_ARN, Message=json.dumps(event))
        sqs.delete_message(QueueUrl=DLQ_URL, ReceiptHandle=msg["ReceiptHandle"])
```

Two details worth getting right. First, the DLQ consumer must be idempotent, because a crash between `publish` and `delete_message` will replay the message. Second, set an alarm on the DLQ's approximate message count rather than on the source queue's depth — a growing DLQ is the signal that matters.

## Fix 3: use the platform's lifecycle hooks, not a cleanup agent

Cleanup is the one place where teams most often write an agent because the platform's mechanism is less familiar.

- Kubernetes Jobs support `ttlSecondsAfterFinished`, which deletes the Job object after completion.
- Kubernetes supports owner references and cascading deletion, so a Namespace or a custom resource can own its children and clean them up when it is removed.
- Controllers such as Argo CD support automated pruning of resources no longer present in the desired state.

A label-based sweep is still an agent, and it is worth saying so plainly:

```bash
# Illustrative only: this is an agent, not a lifecycle policy.
kubectl get ns -o json \
  | jq -r '.items[] | select(.metadata.creationTimestamp < "2026-01-01T00:00:00Z") | .metadata.name' \
  | xargs -I {} kubectl delete ns {}
```

The `--field-selector` form in many examples does not actually support timestamp comparison, which is one reason this pattern tends to be fragile. Prefer declarative ownership:

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: staging-1234
  labels:
    environment: staging
  annotations:
    # Consumed by your controller of choice; Kubernetes itself does not
    # interpret arbitrary TTL annotations.
    example.com/ttl: "7d"
```

The annotation is inert until a controller reads it, so the real work is choosing a controller and making it the single owner of cleanup. The point is that the controller reconciles desired state continuously; it is not a timer that decides which namespaces look old.

## How to tell whether the fix worked

Do not rely on a single latency number. Instrument four things and compare before and after.

**End-to-end latency, not queue latency.** Record a timestamp at event emission and at side-effect completion, and compute percentiles over that span. A load generator such as k6 or Locust can drive the entry point; the interesting number is the difference between the two timestamps, not the HTTP response time.

**Queue depth and age of oldest message.** Depth alone is misleading when consumers are fast. Age of the oldest unacknowledged message is the metric that reveals a stuck consumer.

**DLQ arrival rate.** If this goes up after the change, the problem moved rather than disappeared.

**Alert volume attributable to the removed job.** Count alerts whose source is the scheduled job or its retry table over a comparable window.

A small health check that reports depth and recent average depth:

```python
import time

import boto3

sqs = boto3.client("sqs")
cloudwatch = boto3.client("cloudwatch")


def check_queue_health(queue_url: str, depth_threshold: int = 100) -> dict:
    attrs = sqs.get_queue_attributes(
        QueueUrl=queue_url,
        AttributeNames=["ApproximateNumberOfMessages"],
    )
    depth = int(attrs["Attributes"]["ApproximateNumberOfMessages"])

    end_time = time.time()
    metrics = cloudwatch.get_metric_statistics(
        Namespace="AWS/SQS",
        MetricName="ApproximateNumberOfMessagesVisible",
        Dimensions=[{"Name": "QueueName", "Value": queue_url.rsplit("/", 1)[-1]}],
        StartTime=end_time - 300,
        EndTime=end_time,
        Period=60,
        Statistics=["Average"],
    )
    datapoints = metrics["Datapoints"]
    avg_depth = (
        sum(d["Average"] for d in datapoints) / len(datapoints) if datapoints else 0.0
    )

    return {
        "depth": depth,
        "avg_depth_last_5min": avg_depth,
        "healthy": depth < depth_threshold,
    }
```

The thresholds are placeholders. Derive yours from observed normal depth during your busiest hour, and alarm on the rate of change rather than the absolute value.

## Preventing the next one

Two mechanisms do most of the work: a design checklist applied before code is written, and a review guardrail that flags the patterns mechanically.

| Check | Why it matters | Preferred alternative |
|---|---|---|
| Does the job poll a queue or a table? | Polling is the definition of an agent | Event source mapping or a managed scheduler |
| Does it update a shared status column? | Shared mutable state requires locking | Append-only events, or an outbox table |
| Does it own a retry counter? | The queue already has one | Visibility timeout plus a redrive policy |
| Is it scheduled more often than hourly? | Frequent schedules usually mean batch thinking | Emit events at the moment of change |
| Does it hold a lock in a resource it deletes? | The lock can vanish mid-operation | Owner references and cascading deletion |

For the guardrail, a regex scan in CI catches most of these before review:

```yaml
name: agent-pattern-detector
on: [pull_request]
jobs:
  scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Flag scheduling and polling patterns
        run: |
          set -euo pipefail
          matches=$(grep -rEn \
            'cron|setInterval|schedule\.|retry_table|poll_queue|delete.*namespace' \
            --include='*.py' --include='*.js' --include='*.ts' \
            --include='*.yaml' --include='*.yml' . || true)
          if [ -n "$matches" ]; then
            echo "$matches"
            echo "::warning::Scheduling or polling patterns found; confirm they are not agents."
          fi
```

A warning rather than a failure is usually the right default, because legitimate uses exist. The value is that a reviewer sees the list.

## When agent-driven is the right choice

The claim in the original framing — that event-driven wins most of the time — is directionally right but needs a precise boundary. Agent-driven is appropriate when the work is genuinely stateful across invocations and the state cannot be derived from events.

Legitimate cases:

- **Long-running sagas with compensation.** A multi-step process that must roll back partially completed work needs to track which steps completed. Workflow engines exist precisely for this, and they are agents in the sense used here.
- **Work that must run exactly once at a wall-clock time.** Monthly billing, regulatory filings, scheduled exports. Even here, a managed scheduler that emits an event is preferable to a cron entry, because the event then flows through the same path as everything else.
- **Reconciliation against an external system that has no event stream.** If the only way to learn about changes is to poll, you must poll. Isolate the poller, keep its state minimal, and have it emit events rather than act directly.
- **Rate-limited batch operations against a third party.** An agent that paces requests is doing something an event handler cannot easily do.

The test is not "does it run on a schedule." It is: *does this process decide what to do next based on state it maintains itself?* If yes, it is an agent, and it deserves the same scrutiny as any other stateful service: explicit ownership, bounded retries, and a plan for two of them running at once.

## Escalation when the fixes do not help

If latency spikes persist after removing polling loops, replacing retry tables with redrive policies, and moving cleanup to platform lifecycle hooks, the remaining cause is usually a hidden chain.

**Look for event chains.** Trace a single request end to end and count how many distinct consumers it touches. If one event handler publishes an event that triggers another handler that publishes a third, you have an implicit workflow with no owner. Flatten it into one handler, or make the workflow explicit in a workflow engine where the state is visible.

**Check for state hidden in queue semantics.** FIFO queues with message groups preserve ordering per group, which is a form of state. If message groups map to user sessions, a single slow session blocks its entire group. Either avoid message groups where ordering is not required, or accept the head-of-line blocking deliberately.

**Profile for non-request queries.** On PostgreSQL, `pg_stat_statements` will show you queries executing during the spike that are not part of any user request. Those are your agent-driven reads and writes contending with user traffic. Moving them to a replica is a mitigation; removing them is the fix.

A full migration to a broker with native idempotency and replay, such as Kafka, is a rewrite rather than a fix. Treat it as such and budget accordingly.

## FAQ

**Why does event-driven feel slower at first?**
Because the response no longer waits for the side effect. The user action emits an event and returns; the work happens later. This adds a small, bounded delay to the background work and removes the large, unbounded spikes caused by contention. Measure end-to-end completion time for the background work, not the API response time.

**How do I migrate from cron without downtime?**
Three steps. First, add an event-driven trigger alongside the existing job so both paths exist. Second, make the cron job emit the same event the new trigger emits, so both paths converge on one handler. Third, remove the cron job once the event path has run for a full cycle without incident. Feature-flag the switch so it can be reverted without a deploy.

**What if the database cannot store an append-only event log?**
Use an outbox table. Write the event row in the same transaction as the state change, then have a publisher read unprocessed rows and emit them. The key detail is that the publisher must not decide *whether* to publish based on staleness — it publishes everything unprocessed, in order, and marks rows processed only after the emit succeeds.

```sql
CREATE TABLE outbox (
  id           bigserial PRIMARY KEY,
  aggregate_id varchar(255) NOT NULL,
  event_type   varchar(255) NOT NULL,
  payload      jsonb        NOT NULL,
  created_at   timestamptz  NOT NULL DEFAULT now(),
  processed_at timestamptz  NULL
);

INSERT INTO outbox (aggregate_id, event_type, payload)
VALUES ('user-123', 'user_created', '{"email": "user@example.com"}');

-- Publisher: select a batch, emit, then mark. Skipping rows is safe
-- because unprocessed rows are simply picked up on the next pass.
SELECT id, aggregate_id, event_type, payload
FROM outbox
WHERE processed_at IS NULL
ORDER BY id
LIMIT 100;
```

**Does this apply to serverless only?**
No. The pattern is about where the decision lives, not about the runtime. A Kubernetes CronJob that decides which resources to delete is an agent. A consumer reading from a topic is not, regardless of whether it runs in a container or a function.

## The 30-minute action

Open your repository and search for scheduling and polling constructs in one command:

```bash
grep -rEn 'cron|setInterval|schedule\.|retry_table|poll_queue' \
  --include='*.py' --include='*.js' --include='*.ts' \
  --include='*.yaml' --include='*.yml' .
```

For each match, ask the single question that separates the two patterns: *does this code decide what to do next based on state it maintains itself?* If the answer is yes, you have found an agent. Pick the highest-traffic one and convert it to an event source mapping with a redrive policy before your next deploy.
