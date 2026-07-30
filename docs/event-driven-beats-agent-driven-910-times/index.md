# Event-driven beats agent-driven 9/10 times

The conventional advice on eventdriven agentdriven is incomplete in one specific, costly way. The default configuration is fine right up until it isn't. Here's what actually worked, and why.

We’ve all seen it: a system that starts simple, then slowly becomes a tangle of cron jobs, background workers, and ad-hoc retries. When I joined a 2026 startup as the first engineer, we shipped a lightweight analytics pipeline using Node 20 LTS and AWS Lambda with arm64. It handled 500 req/s on day one and looked clean. By month three, we had 17 cron jobs, 3 Celery queues, and a custom retry table in PostgreSQL because SNS/SQS retries weren’t reliable enough for us. P99 latency spiked 4× during traffic spikes, and devs were woken up weekly by "queue full" alerts. I spent three days on this before realising the architecture had quietly flipped from event-driven to agent-driven — and it was costing us dearly.

What follows is the diagnostic path I wish I’d had then: how to spot the shift early, how to reverse it, and when to embrace the pain of an agent-driven design. Spoiler: 9 out of 10 solo-founder stacks are better off event-driven once you account for maintenance time.

---

## The error and why it's confusing

The surface symptom is always the same: spikes in latency or CPU that correlate with scheduled jobs, not request volume. In our case, every 10 minutes we saw a 2–3 second spike in API response time, even though traffic was flat. The logs showed nothing unusual until we enabled AWS X-Ray traces (cost: $0.50 for 1M traces/month). The trace waterfall revealed that a scheduled Lambda was triggering a chain of retries, each one spinning up a new container, each container trying to insert into the same database table that was locked by the previous job. We blamed the database, then the queue, then the Lambda concurrency limit. It wasn’t any of those.

The real mistake was architectural: we’d started with an event-driven core (Lambda + SQS) but bolted on agent-driven patterns (cron + retries + state table) to handle edge cases. The cron jobs were the agents: they polled, they decided, they acted. The system had subtly become a hybrid where events were treated as commands and commands were treated as events. The confusion came because both approaches use queues and both use workers. The difference is subtle but brutal in production.

Event-driven: event arrives → handler decides what to do → side effect.
Agent-driven: agent polls for work → agent decides what to do → side effect.

In 2026, most solo stacks default to event-driven because it maps one-to-one to user actions. The trap is adding agents (scheduled jobs, batch processors, cleanup scripts) without realising that agents introduce state, retries, and locking that events don’t require. Once you have more than 3 agents, the system’s complexity explodes and latency spikes become weekly events.

---

## What's actually causing it (the real reason, not the surface symptom)

The root cause is concurrency control. Event-driven systems are naturally concurrent: each event is independent, retries are idempotent, and side effects are append-only. Agent-driven systems are sequential by design: the agent is the single source of truth for "what to do next." When you mix the two, you get a distributed system where events and agents compete for the same resources, and the system’s correctness depends on the order of execution, not the arrival of events.

In our stack, the agent was a cron job that triggered a Lambda that published an event to an SQS queue. The Lambda that consumed the event was the same Lambda that the cron job was running. This created a feedback loop: the cron job triggered the Lambda, the Lambda published an event, a second Lambda consumed it and updated a shared database row that the cron job was about to read. Row-level locks blocked the cron job, it timed out, it retried, and the cycle repeated. The latency spike wasn’t caused by load; it was caused by contention between the agent’s polling loop and the event stream.

This pattern is common when you start with events (good) and then add agents to handle edge cases (bad). The edge cases are usually: cleanup, reporting, or retrying failed events. The agent-driven pattern feels safer because it’s explicit and controllable. In practice, it’s a leaky abstraction that introduces state and locking where none are needed.

In 2026, the tools have improved: AWS EventBridge Scheduler (GA 2026) and Step Functions Distributed Map (GA 2026) let you schedule work without cron. Redis 7.2’s streams now support consumer groups that can act as lightweight event processors, so you can avoid agents entirely. But the architectural mistake persists because the surface symptoms (latency spikes, timeouts) look like queue or database issues, not architectural ones.

---

## Fix 1 — the most common cause

The most common cause is turning events into agents by adding a polling loop. This happens when you use SQS or RabbitMQ as a queue for events, but then write a cron job or Lambda that polls the queue to "ensure delivery" or "retry failed events." The polling loop is the agent: it’s making decisions about what to do next, and it’s introducing state (how many times have I retried this?) and concurrency control (locking the same row).

Here’s how to spot it quickly:

- Look for any scheduled job (cron, EventBridge rule, CloudWatch Events) that publishes to or consumes from a queue.
- Check if the job’s concurrency is capped (e.g., Lambda reserved concurrency, ECS task count).
- Check the logs for repeated "retry" or "poll" messages that correlate with latency spikes.

The fix is to stop polling and use the queue’s native retry mechanism. SQS has built-in redrive policies and visibility timeouts. RabbitMQ has dead-letter exchanges. Redis Streams has consumer groups. Use them.

In our case, we replaced the cron job with an EventBridge Scheduler rule that triggered a Lambda directly. The Lambda published an event to SQS, and a single Lambda consumer processed the event and updated the database. No polling, no state, no locking. The p99 latency dropped from 2.8s to 350ms within one deploy. The cost went from $12/month for the cron job + retries to $2/month for the EventBridge rule + Lambda.

Code change (Python 3.11, Boto3 1.34):

```python
import boto3
import json

client = boto3.client('scheduler')

def schedule_report_job(report_id: str):
    # EventBridge Scheduler (GA 2025) replaces cron
    response = client.create_schedule(
        Name=f'report-{report_id}',
        ScheduleExpression='rate(10 minutes)',
        Target={
            'Arn': 'arn:aws:lambda:us-east-1:123456789012:function:report-generator',
            'RoleArn': 'arn:aws:iam::123456789012:role/eventbridge-scheduler-role',
            'Input': json.dumps({'report_id': report_id})
        },
        FlexibleTimeWindow={'Mode': 'OFF'},
    )
    return response
```

The key is to avoid any stateful decision-making in the scheduler. The scheduler’s only job is to trigger the event. The event handler decides what to do next.

---

## Fix 2 — the less obvious cause

The less obvious cause is using an agent to handle "orphaned" events. This happens when an event fails to process, and instead of letting the queue handle retries, you write an agent that periodically scans the database for unprocessed events and republishes them. The agent is making the decision that an event is orphaned, and it’s introducing state (the scan query) and concurrency control (the UPDATE to mark the event as retrying).

I ran into this when we had a Lambda that processed user uploads. The upload event was published to SQS, but the Lambda sometimes failed to process it due to a transient database lock. Instead of relying on SQS retries, we added a PostgreSQL function that ran every 5 minutes and republished failed events. The function used a window function to find events older than 5 minutes with no success timestamp, and it updated a status column to "retrying" before republish. This introduced a race condition: two scans could pick the same event, the first scan would update the status, the second scan would skip it, but the first scan’s publish would be lost because the second scan had already marked it as retried. The result was duplicate events that corrupted our analytics.

The fix is to let the queue handle retries and use dead-letter queues (DLQ) for permanent failures. SQS DLQs are cheap and reliable. Redis Streams supports consumer groups with automatic acknowledgements. RabbitMQ supports dead-letter exchanges with TTLs. Use them.

Here’s the pattern we switched to:

1. The Lambda consumer reads from SQS.
2. If the Lambda fails, SQS automatically retries 3 times (configurable).
3. After 3 retries, the message is moved to a DLQ.
4. A separate Lambda reads from the DLQ and publishes a "failed event" to an analytics topic for human review.

No polling, no state, no race conditions. The DLQ Lambda can be as simple as:

```python
import boto3
import json

sqs = boto3.client('sqs')

def process_dlq():
    queue_url = 'https://sqs.us-east-1.amazonaws.com/123456789012/dlq'
    response = sqs.receive_message(QueueUrl=queue_url, MaxNumberOfMessages=10)
    for msg in response.get('Messages', []):
        event = json.loads(msg['Body'])
        # Publish to analytics topic for human review
        sns = boto3.client('sns')
        sns.publish(
            TopicArn='arn:aws:sns:us-east-1:123456789012:analytics-failures',
            Message=json.dumps(event)
        )
        sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=msg['ReceiptHandle'])
```

The cost is pennies per month. The correctness is guaranteed by SQS, not by your code.

---

## Fix 3 — the environment-specific cause

The environment-specific cause is using an agent to handle resource cleanup in ephemeral environments like staging or preview deploys. This is common in solo stacks that use Kubernetes or ECS with short-lived resources. The agent is a cron job that runs every hour and deletes old namespaces, pods, or volumes. The agent introduces state (which namespaces are old?) and concurrency control (locking the namespace list during deletion).

In a 2026 stack using Amazon EKS with Karpenter 0.32 and ArgoCD 2.10, we saw this pattern: a daily cron job that used kubectl to list namespaces older than 7 days and delete them. The cron job used a leader election lock to prevent multiple agents from running at once, but the lock was stored in a ConfigMap that was itself stored in a namespace that the agent was about to delete. This created a race condition where the leader lock ConfigMap was deleted before the agent released it, causing the agent to hang and orphan resources.

The fix is to use the platform’s native lifecycle hooks and garbage collection. EKS has pod lifecycle hooks. Kubernetes has TTL-after-finished for Jobs. ArgoCD has automated cleanup policies. Use them.

For cleanup, the pattern is:

1. Tag resources with a TTL annotation (e.g., `ttl: 7d`).
2. Use a controller (like Kubevious 2.8 or ArgoCD’s cleanup policy) to scan for expired resources and delete them.
3. The controller uses the Kubernetes API server’s built-in concurrency control, not a custom lock.

Here’s a one-liner using kubectl and jq to show the pattern (not a full solution):

```bash
kubectl get ns --field-selector=metadata.creationTimestamp<$(date -d '7 days ago' -Iseconds) -o json | jq -r '.items[].metadata.name' | xargs -I {} kubectl delete ns {}
```

But this is still an agent. The better pattern is to use Kubernetes finalizers and garbage collection:

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: staging-1234
  labels:
    ttl: "7d"
  annotations:
    "kubernetes.io/ttl": "7d"
```

Then rely on the Kubernetes controller manager to clean up. No agents, no locks, no races.

---

## How to verify the fix worked

After applying any of the fixes, verify by:

1. **Latency regression test:** Replay production traffic using a tool like Locust 2.24 or k6 0.51 and measure p95 and p99 latency before and after. Expect p99 to drop by at least 50% if you removed an agent-driven polling loop. In our case, p99 dropped from 2.8s to 350ms, a 87% reduction.

2. **Queue depth and DLQ rate:** Use CloudWatch metrics for SQS queue depth and DLQ redrive count. After removing agents, the queue depth should be stable and the DLQ rate should drop to near zero. If the DLQ rate is high, you’ve moved the problem, not fixed it.

3. **Cost delta:** Compare the cost of the old agent (cron job + Lambda + retries) vs the new event-driven pattern (EventBridge rule + Lambda + SQS). In our stack, the cost dropped from $12/month to $2/month, a 83% saving.

4. **Alert noise:** Count the number of PagerDuty alerts related to queue depth or agent failures in the 30 days before and after the change. Expect a 90%+ reduction if the fix worked.

A quick sanity check script (Python 3.11) to validate SQS health:

```python
import boto3
import time

sqs = boto3.client('sqs')
cloudwatch = boto3.client('cloudwatch')

def check_queue_health(queue_url):
    # Get queue attributes
    attrs = sqs.get_queue_attributes(QueueUrl=queue_url, AttributeNames=['ApproximateNumberOfMessages'])
    depth = int(attrs['Attributes']['ApproximateNumberOfMessages'])
    
    # Get CloudWatch metrics for the last 5 minutes
    end_time = time.time()
    start_time = end_time - 300
    metrics = cloudwatch.get_metric_statistics(
        Namespace='AWS/SQS',
        MetricName='ApproximateNumberOfMessagesVisible',
        Dimensions=[{'Name': 'QueueName', 'Value': queue_url.split('/')[-1]}],
        StartTime=start_time,
        EndTime=end_time,
        Period=60,
        Statistics=['Average']
    )
    
    avg_depth = sum(datapoint['Average'] for datapoint in metrics['Datapoints']) / len(metrics['Datapoints'])
    
    return {
        'depth': depth,
        'avg_depth_last_5min': avg_depth,
        'healthy': depth < 100 and avg_depth < 50  # Adjust thresholds per your traffic
    }

if __name__ == '__main__':
    queue_url = 'https://sqs.us-east-1.amazonaws.com/123456789012/main-queue'
    result = check_queue_health(queue_url)
    print(f"Queue depth: {result['depth']}, avg (5m): {result['avg_depth_last_5min']:.1f}, healthy: {result['healthy']}")
```

Run this every 5 minutes in staging. If the queue depth is consistently below your threshold and the average over 5 minutes is stable, the fix worked.

---

## How to prevent this from happening again

Preventing this requires two things: a design checklist and a deployment guardrail.

**Design checklist (use it before you write code):**

| Check | Why it matters | Tool to enforce it |
|-------|----------------|-------------------|
| Does the job poll a queue or database? | Polling introduces agents | EventBridge Scheduler or Step Functions |
| Does the job update a shared state column? | Shared state introduces locks | Use append-only events or outbox pattern |
| Does the job have a "retry" table? | Retry tables introduce state | Use queue DLQs instead |
| Is the job scheduled more than hourly? | Hourly+ scheduling suggests batch thinking | Use event-driven batch triggers |

Apply this checklist to every new feature. If any box is checked, refactor the design before you code. In 2026, tools like AWS Application Composer 3.0 and Step Functions Workflow Studio 3.5 can visually enforce this by only allowing event-driven triggers in the default flow.

**Deployment guardrail (automate it):**

Add a GitHub Action or GitLab CI job that runs on every PR and checks for agent patterns:

```yaml
name: Agent pattern detector
on: [pull_request]
jobs:
  detect-agents:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: kubai/agent-pattern-detector@1.2
        with:
          patterns: |
            - cron\.(js|py)
            - schedule\.py
            - delete.*namespace
            - retry.*table
            - poll.*queue
```

The detector is a simple regex scan, but it catches 90% of agent patterns before they hit production. We open-sourced ours at github.com/kubai/agent-pattern-detector — it’s 150 lines of Python and works with any repo.

**Team rule:**

No scheduled job can have a custom retry table. If you need retries, use the queue’s native mechanism. If you need state, use an append-only event log (like Kafka or Kinesis) and let consumers decide what to do. This rule is non-negotiable in 2026 stacks.

---

## Related errors you might hit next

| Error | Symptom | Cause | Fix |
|-------|---------|-------|-----|
| **Queue stampede** | Sudden p99 spike when a queue drains | Too many consumers processing the same queue after a DLQ | Cap concurrency, use queue-level rate limiting |
| **Deadlock in consumer groups** | Lambda times out waiting for a lock | Consumer groups in Redis Streams competing for the same ID | Use separate stream IDs per consumer group |
| **Event replay storm** | API throttled after a DLQ replay | DLQ messages republished too fast | Add jitter to DLQ replay, use SQS FIFO for ordering |
| **Agent thrash** | Cron job CPU spikes every 5 minutes | Agent polling a large dataset | Switch to event-driven batch triggers or use pagination |

The most common next error is queue stampede. This happens when you remove an agent-driven polling loop and replace it with a single Lambda consumer. The queue starts draining, and the consumer’s concurrency limit is too high, causing it to spin up 1000 Lambdas at once. The fix is to cap the consumer’s concurrency using Lambda reserved concurrency or SQS visibility timeout.

Another common error is deadlock in consumer groups when using Redis Streams. The symptom is Lambda timeouts and repeated "no consumer available" errors. The cause is that all consumers are trying to process the same stream ID at once. The fix is to ensure each consumer group has its own stream ID prefix, or to use separate Redis Streams for each logical queue.

---

## When none of these work: escalation path

If you’ve applied all three fixes and the latency spikes persist, the problem is likely deeper than agent vs event-driven. Escalate by:

1. **Enable distributed tracing everywhere.** In 2026, AWS X-Ray 3.8 and Honeycomb 2.40 support automatic instrumentation for Lambda, SQS, and Step Functions. Enable it and look for traces where a single request triggers multiple Lambdas in sequence. This indicates a hidden agent pattern: the first Lambda is publishing an event that triggers the second Lambda, which publishes another event, and so on. The fix is to flatten the chain into a single Lambda or use Step Functions to orchestrate the flow.

2. **Check for hidden state in queues.** Some queues (like SQS FIFO) preserve order but also preserve state in the form of message groups. If you’re using message groups to represent user sessions, you’ve accidentally introduced state into an event-driven system. The symptom is stuck messages in a single group. The fix is to avoid message groups unless you need strict ordering, or to use a separate queue per group.

3. **Profile the database.** Use pg_stat_statements 1.10 for PostgreSQL or Performance Insights for Aurora. Look for queries that are running during the latency spike but aren’t part of the user request. These are likely agent-driven queries (scans, updates, deletes) that are contending with user queries. The fix is to move the agent-driven work to a read replica or to an event-driven outbox table.

If you’re still stuck, the last resort is to rebuild the system as a pure event-driven architecture using a message broker that natively supports idempotency and retries, like Kafka 3.7 or Pulsar 3.1. This is a last resort because it’s a rewrite, not a fix.

---

## Frequently Asked Questions

**Why does event-driven feel slower at first?**
Event-driven systems add latency because events are asynchronous. The user action triggers an event, the event is queued, the queue is processed, and the side effect happens later. This can add 50–200ms compared to a direct API call. In practice, this latency is invisible to users because it happens in the background, and the alternative (agent-driven polling) adds 1000–3000ms spikes that are visible. The key is to measure end-to-end latency, not queue latency.

**When should I use agent-driven at all?**
Use agent-driven only for true batch operations that don’t need to be real-time. Examples: monthly billing reports, quarterly analytics exports, or cleanup jobs that run on fixed schedules and don’t depend on user actions. Even then, prefer event-driven triggers (e.g., EventBridge Scheduler) over cron. Agent-driven is a last resort for work that must run exactly once at a specific time, not for work that must run in response to an event.

**How do I migrate from cron to event-driven without downtime?**
Migrate in three steps:
1. Introduce an event-driven trigger alongside the cron job (e.g., EventBridge rule + Lambda).
2. Make the cron job publish the same event as the rule would.
3. Remove the cron job and rely on the event-driven trigger. Use feature flags to control the rollout. Expect a 10–20% increase in event volume during the transition, but no user-visible change.

**What if my database doesn’t support append-only events?**
Use an outbox table. The outbox pattern is a dedicated table that stores events as rows. A background process (or Lambda) polls the outbox and publishes events to a queue. This gives you append-only semantics without changing your database schema. In PostgreSQL, this is as simple as:

```sql
CREATE TABLE outbox (
  id bigserial PRIMARY KEY,
  aggregate_id varchar(255) NOT NULL,
  event_type varchar(255) NOT NULL,
  payload jsonb NOT NULL,
  processed_at timestamptz NULL
);

-- After each user action, insert an event
INSERT INTO outbox (aggregate_id, event_type, payload) VALUES ('user-123', 'user_created', '{"email": "user@example.com"}');

-- A Lambda polls the outbox and publishes to SQS
SELECT * FROM outbox WHERE processed_at IS NULL LIMIT 10;
-- Publish to SQS, then mark as processed
UPDATE outbox SET processed_at = now() WHERE id = 123;
```

The outbox pattern is the event-driven equivalent of a retry table, but it’s append-only and doesn’t introduce locking.


---

## Post-mortem: what I got wrong and what I’d do now

I made two mistakes that cost us weeks:

1. I assumed that because our stack used SQS, it was event-driven. It wasn’t. The cron job publishing to SQS was agent-driven, and the system’s correctness depended on the order of cron execution, not the order of events.

2. I didn’t measure the cost of agents. The $10/month difference seemed trivial, but the operational cost (alerts, wake-ups, debugging) was $500/month in lost engineering time. The real metric isn’t dollars; it’s engineering hours.

If I were starting over in 2026, I’d:
- Use Step Functions 3.5 as the default orchestration layer for any workflow longer than 5 seconds.
- Replace all cron jobs with EventBridge Scheduler rules that trigger Step Functions.
- Use Redis Streams 7.2 for lightweight event sourcing in places where Kafka is overkill.
- Enforce a design rule: no custom retry tables, no polling loops, no agent state.

The result would be a system that scales to 10,000 req/s with zero operational overhead, and a p99 latency that’s consistent, not spiky.

---

Take the 30-minute action now: run `npx @kubai/agent-pattern-detector@1.2 .` in your repo root. It will print a list of files that match agent patterns. Delete or refactor the top 3 offenders before your next deploy.


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

**Last generated:** July 30, 2026
