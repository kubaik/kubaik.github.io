# Always-on agents cost more than you think

An agent that runs continuously and an agent that wakes on demand can do identical work and produce very different bills, latencies, and failure profiles. The difference is not the code. It is where the cost lands: idle time versus wake-up time, plus the retry storms that follow whichever one you chose badly.

This article builds a cost model for both patterns, walks through a working implementation of each, and catalogs the failure modes that show up under load. The examples use a payment-reconciliation workload — a queue of jobs that must be processed within a timeout window — because that workload punishes both patterns in visible ways.

## The gap between what the docs say and what production needs

Vendor documentation describes the happy path: deploy the function, let it run, pay for what you use. Two things are missing from that description.

First, "pay for what you use" hides a floor. A process that is running but not working still consumes memory, still holds a container slot, and still bills. A function that is not running costs nothing per second but pays a fixed latency penalty every time it starts. Neither pattern is free at zero traffic; they are free in different places.

Second, the timeout budget belongs to your upstream dependency, not to you. If a payment provider documents a 45-second window, your end-to-end path — queue pickup, cold start, HTTP call, retry, response — has to fit inside that window with margin. Cold-start latency is not a performance detail in that context. It is part of your success rate.

Three failure domains drive most of the surprise:

- **Power.** Hosted instances in regions with unstable grid supply may sit behind UPS or generator capacity. An always-on process keeps drawing through an outage; an on-demand process simply stops, which is cheaper but means queued work waits.
- **Network.** Mobile and last-mile networks drop packets and reset connections. Retries are the normal case, not the exception, and each retry is a billable unit of work.
- **Memory.** Caches and queues have eviction policies. Eviction is not a passive cleanup; it can generate a burst of new work at exactly the moment the system is already under pressure.

The rest of this article treats those three as first-class inputs to the cost model.

## How the two patterns actually work under the hood

### Always-on agents

An always-on agent is a long-lived process — a systemd unit, a container with a restart policy, a Kubernetes Deployment — that polls a queue or listens on a socket. Its cost structure is:

- **Fixed cost:** instance-hours × instance count, regardless of load.
- **Variable cost:** CPU and memory actually consumed while processing.
- **Hidden cost:** the instance is sized for peak, so it is over-provisioned for the ~90% of the day that is not peak.

The failure mode is not the idle bill by itself. It is that an always-on agent has no natural backpressure. When the queue grows, the agent keeps polling at its fixed interval, which adds load to the queue, which adds load to the cache, which can trigger eviction, which adds more work. The system has no mechanism to shed load because the polling loop is unconditional.

### On-demand agents

An on-demand agent is invoked per unit of work — a queue trigger, an HTTP request, a scheduled tick. Its cost structure is:

- **Fixed cost:** approximately zero when idle.
- **Variable cost:** invocation duration × memory × invocation count, including invocations that do nothing.
- **Hidden cost:** cold-start latency on every invocation that lands on a cold execution environment, plus the keep-alive traffic you add to avoid cold starts.

The failure mode is the inverse of the always-on case. There is natural backpressure (the queue absorbs the burst), but there is a latency tax on every wake-up, and that tax is worst exactly when traffic spikes, because that is when the platform is most likely to need new execution environments.

### The hybrid pattern

A hybrid keeps a durable buffer — a log-structured queue with consumer groups, or a managed queue with a dead-letter path — in front of on-demand consumers. The buffer absorbs bursts, the consumers scale with depth, and the timeout budget is enforced at the consumer, not at the producer.

The hybrid is not free. It adds a moving part (the buffer), and it adds a second place where messages can be lost or duplicated. The rest of this article shows how to reason about whether that trade is worth it.

## A worked cost model

The numbers below are illustrative. Substitute your own provider's published rates; the arithmetic is what matters.

Assume a job that takes 200 ms of wall time to process, a queue that receives 50,000 jobs per month, and a peak of 5× the average rate for two hours each evening.

**Always-on, one instance:**

- Instance cost: 0.02 USD/hour × 730 hours = 14.60 USD/month.
- Capacity: an instance handling 5 jobs/second at 200 ms each saturates at 5 concurrent jobs. To absorb a 5× peak you need headroom, so assume 3 instances for redundancy and burst: 43.80 USD/month.
- Idle fraction: if average load is 0.02 jobs/second and capacity is 15 jobs/second, the fleet is idle roughly 99.9% of the time. You are paying 43.80 USD/month to be ready.

**On-demand:**

- Invocations: 50,000 jobs + retries. Assume a 2% retry rate, so 51,000 invocations.
- Duration: 200 ms of work plus 400 ms of cold start on the fraction of invocations that hit a cold environment. If 20% are cold, the average duration is 200 + (0.20 × 400) = 280 ms.
- Cost: 51,000 × 0.280 s = 14,280 GB-s at 512 MB. At a representative rate of 0.0000166667 USD per GB-s, that is 0.24 USD/month of compute.
- Plus keep-alive: if you ping every 30 seconds to stay warm, that is 2,880 invocations/day × 30 days = 86,400 invocations, each ~50 ms, which is 2,160 GB-s, or 0.04 USD/month.

The gap is stark: roughly 44 USD/month versus roughly 0.28 USD/month. That is the honest version of the "always-on costs more" claim — it costs more *at low utilization*. The interesting question is what happens as utilization rises and as latency requirements tighten.

**Where the gap closes.** At 15 jobs/second sustained, the always-on fleet is fully utilized and the on-demand fleet is paying cold starts on a large fraction of invocations. Recompute: 15 jobs/second × 2,592,000 seconds/month = 38.9M jobs. On-demand at 280 ms average is 38.9M × 0.280 = 10.9M GB-s = 181 USD/month, versus 43.80 USD/month for the always-on fleet. The crossover is real and it depends entirely on your utilization curve.

**How to measure your own crossover.** Instrument three things and you can compute it without guessing:

1. **Idle fraction.** For always-on, sample CPU utilization every minute and count minutes below 10%. For on-demand, count invocations that return no work (empty polls).
2. **Cold-start rate.** Log the time from invocation start to handler entry. Anything above your platform's warm threshold is a cold start. Track the fraction and the P99.
3. **Retry amplification.** Count retries per successful job. If this is above 1.05, your cost model is dominated by retries, not by the base pattern.

With those three numbers you can compute both sides of the crossover for your own workload in a spreadsheet.

## Step-by-step implementation

Both implementations below process jobs from a queue and call an external payment API. The code is intentionally minimal; the point is the shape, not the production hardening.

### Always-on agent

A containerized Node process that polls a Redis list:

```dockerfile
FROM node:20-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
CMD ["node", "agent.js"]
```

```javascript
import { createClient } from 'redis';
import { setTimeout as sleep } from 'timers/promises';

const redis = createClient({ url: process.env.REDIS_URL });
await redis.connect();

const POLL_INTERVAL_MS = 5000;

while (true) {
  const start = Date.now();
  const job = await redis.lPop('payments:queue');
  if (job) {
    try {
      await fetch('https://api.example.com/payments', {
        method: 'POST',
        body: job,
        headers: { 'Content-Type': 'application/json' },
        signal: AbortSignal.timeout(45000),
      });
    } catch (err) {
      // Push back for retry; see the failure-modes section for why
      // an unbounded push-back is dangerous.
      await redis.rPush('payments:queue', job);
    }
  }
  const elapsed = Date.now() - start;
  if (elapsed < POLL_INTERVAL_MS) await sleep(POLL_INTERVAL_MS - elapsed);
}
```

Two things to notice. The poll interval is fixed, which means the agent adds a constant load to Redis regardless of queue depth. And the retry path pushes the job back to the same list with no attempt counter, which is the classic unbounded-retry bug.

### On-demand agent

A queue-triggered function with a visibility timeout matched to the upstream timeout:

```javascript
import { SQSClient, ReceiveMessageCommand, DeleteMessageCommand } from '@aws-sdk/client-sqs';

const sqs = new SQSClient({ region: 'af-south-1' });

export const handler = async () => {
  const { Messages } = await sqs.send(new ReceiveMessageCommand({
    QueueUrl: process.env.SQS_URL,
    MaxNumberOfMessages: 1,
    WaitTimeSeconds: 2,
  }));

  if (!Messages?.length) return;

  try {
    const res = await fetch('https://api.example.com/payments', {
      method: 'POST',
      body: Messages[0].Body,
      headers: { 'Content-Type': 'application/json' },
      signal: AbortSignal.timeout(45000),
    });
    if (!res.ok) throw new Error(`upstream ${res.status}`);
    await sqs.send(new DeleteMessageCommand({
      QueueUrl: process.env.SQS_URL,
      ReceiptHandle: Messages[0].ReceiptHandle,
    }));
  } catch (err) {
    // Do not delete: let the visibility timeout expire and the
    // redrive policy handle retries up to maxReceiveCount.
  }
};
```

The deployment template sets the two values that matter:

```yaml
Resources:
  PaymentsQueue:
    Type: AWS::SQS::Queue
    Properties:
      VisibilityTimeout: 60          # must exceed the 45s upstream timeout
      RedrivePolicy:
        maxReceiveCount: 3
        deadLetterTargetArn: !GetAtt PaymentsDLQ.Arn

  MpesaAgent:
    Type: AWS::Serverless::Function
    Properties:
      Runtime: nodejs20.x
      MemorySize: 512
      Timeout: 60                    # must also exceed the upstream timeout
      Architectures:
        - arm64
      Environment:
        Variables:
          SQS_URL: !Ref PaymentsQueue
      Events:
        SQSEvent:
          Type: SQS
          Properties:
            Queue: !GetAtt PaymentsQueue.Arn
            BatchSize: 1
```

The critical constraint: **visibility timeout and function timeout must both exceed the upstream timeout.** If the upstream takes 47 seconds and your visibility timeout is 30, the message becomes visible again while the first invocation is still running, and you get duplicate processing. This is the single most common configuration bug in queue-triggered agent systems.

### Hybrid: buffered consumers

The hybrid replaces the unconditional poll with a blocking read from a consumer group, so consumers only wake when there is work:

```javascript
// Producer
await redis.xAdd('payments:stream', '*', { body: JSON.stringify(job) });

// Consumer
const response = await redis.xReadGroup(
  'payments-group',
  process.env.CONSUMER_ID,
  [{ key: 'payments:stream', id: '>' }],
  { COUNT: 1, BLOCK: 5000 }
);
```

The consumer group gives you three things the list does not: at-least-once delivery with explicit acknowledgement (`XACK`), a pending-entries list you can inspect for stuck messages, and the ability to scale consumers horizontally without coordination. The cost is that you now have to manage the pending list, or it grows without bound.

## Failure modes worth designing for

### Unbounded retry amplification

A job that fails and is pushed back to the same queue with no attempt counter will be retried forever, and each retry consumes a worker slot. Under a partial outage — say, the upstream is returning 503 for 30% of requests — the effective queue depth grows by 1.43× per pass. Within a few minutes the queue is dominated by retries, and the system is doing more work on failures than it ever did on successes.

**Fix:** every job carries an attempt counter. The consumer increments it and routes to a dead-letter queue when it exceeds `maxReceiveCount`. Never rely on the queue's own redrive policy alone if your consumer can push work back to the same queue.

### Cold-start tail latency

Average cold start is not the number that matters. The number that matters is the 99.9th percentile, because that is the invocation that blows your timeout budget. Cold starts are heavy-tailed: most are fast, a few are several times the median, and the tail widens under platform load — which is exactly when your traffic peaks.

**Fix:** measure the cold-start distribution, not the mean. If your P99.9 cold start plus your P99 upstream latency exceeds your timeout budget, you need either provisioned concurrency, a warm pool, or a pattern that tolerates the tail (for example, a fast acknowledgement followed by asynchronous processing).

### Cache eviction as a load generator

When a cache crosses its memory threshold, the eviction policy starts removing keys. If those keys represent queued work or session state, the eviction itself creates new work — cache misses that must be recomputed, or jobs that must be re-enqueued. The eviction becomes a load generator at the worst possible moment.

**Fix:** size the cache so that eviction is a rare event, not a routine one. Cap the queue's memory with an explicit maximum length and a trimming strategy that removes old entries rather than active ones. Monitor eviction rate as a first-class metric; if it is nonzero during normal operation, the cache is undersized.

### Failover races on locks

Distributed locks built on a single primary node have a window during failover where two clients can both believe they hold the lock. The window is short — typically the time for the old primary to stop accepting writes plus the time for the new primary to be promoted — but it is long enough for a duplicate payment.

**Fix:** if correctness depends on mutual exclusion, use a lock service with quorum semantics rather than a single-node cache. If you must use the cache, use its replication-wait primitive before treating a lock as acquired, and accept that this adds latency to every lock acquisition.

### Timeout drift in upstream APIs

Documented timeouts are not contractual. An API that documents a 45-second window may return in 35 seconds or 55 seconds depending on backend load. If your own timeout is set to the documented value, you will fail requests that would have succeeded.

**Fix:** set your timeout above the documented maximum, log the actual response time distribution, and alert on drift. Treat the documented timeout as a lower bound, not a target.

### Scheduled triggers and timezone skew

Cron expressions evaluated in UTC will fire at the wrong local time for any team not in UTC, and the error is silent until someone notices that the nightly job ran during peak hours. This is a configuration bug, not a platform bug, but it is common enough to be worth checking.

**Fix:** set the timezone explicitly in the scheduler configuration, and write a test that asserts the next fire time is what you expect.

## A decision checklist

Use the following to choose a pattern. Answer each question for your own workload.

| Question | Always-on | On-demand | Hybrid |
|---|---|---|---|
| Is sustained utilization above ~20% of a single instance's capacity? | Yes | No | Yes |
| Is P99.9 end-to-end latency budget under 1 second? | Yes | No | Yes |
| Is there a hard upstream timeout you must fit inside? | Risky | Risky | Yes |
| Is the workload bursty with a predictable peak? | No | Yes | Yes |
| Can the team operate a durable buffer? | N/A | N/A | Required |
| Is the workload under ~5k jobs/month? | Yes | Yes | Overkill |
| Does correctness require mutual exclusion? | Depends on lock service | Depends on lock service | Depends on lock service |

The pattern that wins is the one whose failure modes you can operate. An always-on fleet fails by saturating; an on-demand fleet fails by timing out; a hybrid fails by losing or duplicating messages in the buffer. Pick the failure you are best equipped to detect and fix.

## What to do next

Within the next 30 minutes, measure your idle fraction. For an always-on agent, sample CPU utilization every minute for an hour and count the minutes below 10%:

```bash
# Example: sample container CPU every 60s for an hour
for i in $(seq 1 60); do
  date -u +%FT%TZ
  docker stats --no-stream --format '{{.CPUPerc}}' your-agent-container
  sleep 60
done
```

For an on-demand agent, count invocations that returned no work over the same window:

```bash
aws cloudwatch get-metric-statistics \
  --namespace AWS/Lambda \
  --metric-name Invocations \
  --dimensions Name=FunctionName,Value=your-agent \
  --start-time "$(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%SZ)" \
  --end-time "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --period 300 \
  --statistics Sum \
  --output table
```

If your always-on agent is below 10% CPU for more than 80% of samples, or your on-demand agent is invoked more than 20% of the time with an empty queue, the pattern you chose is mismatched to the workload. The next step is to compute the crossover point from your own utilization curve, not from a blog post's numbers.
