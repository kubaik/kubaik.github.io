# Spot evictions: the retry logic that holds up

## Why naive retry logic fails on Spot fleets

Running background workers on Spot Instances is a well-documented cost lever. The failure mode that catches teams off guard is not the eviction itself — it is what the fleet does immediately afterwards.

A first-pass implementation usually looks like this: a worker pulls a job from a queue, the instance is reclaimed, and the worker restarts and retries. If the retry policy is a fixed backoff (say, 5 minutes) or a plain exponential backoff with no jitter, every evicted worker computes roughly the same delay and fires at roughly the same moment. A hundred workers that vanished together come back together. The downstream API sees a step-function spike in traffic, returns 429s or 503s, and the retries compound.

This is a coordination problem, not a capacity problem. The fix is to make retries depend on the job's attempt count and the downstream system's health, not on the lifecycle of the instance that happened to be running it.

Three properties matter:

1. **Early detection.** Spot gives a short termination notice (documented at up to two minutes for EC2 Spot). That window is enough to persist state, but only if something is watching for it.
2. **Durable decoupling.** The retry record must survive the death of the instance that created it. If the retry timer lives in the worker process, it dies with the process.
3. **Backpressure-aware delay.** The delay should widen when the downstream is already unhealthy, so retries do not amplify an existing incident.

## A reference architecture

The pattern below is not the only workable one, but it separates the concerns cleanly and is easy to reason about during an incident.

- **Compute:** Spot Instances (or a Spot-backed managed compute service such as AWS Batch, GKE Autopilot with Spot node pools, or a self-managed autoscaling group).
- **Eviction signal:** the cloud provider's instance-metadata termination notice, or the equivalent event stream.
- **Buffer:** a durable, ordered log — Redis Streams, Kafka, Kinesis, or a database table with a status column.
- **Controller:** a small long-running service that reads the buffer, computes a delay from the attempt count and downstream health, and re-enqueues the job.
- **Job queue:** whatever the workers already consume. The controller writes back into it after the delay.
- **Observability:** metrics for eviction rate, retry delay distribution, and downstream error rate.

The important structural decision is that the retry timer lives in the controller, not in the worker. When an instance is reclaimed, the worker's only job is to publish an eviction event. Everything after that is handled by a process that is not co-located with the ephemeral fleet.

## Step 1: detect the eviction and publish an event

The EC2 instance metadata service exposes a termination notice at a well-known path. Polling it is cheap and does not require API credentials.

```python
# spot_monitor.py — runs as a sidecar on each Spot instance
import json
import os
import time
import urllib.request

import redis

REDIS_URL = os.environ["REDIS_URL"]
STREAM = "agent_evictions"
NOTICE_URL = "http://169.254.169.254/latest/meta-data/spot/instance-action"

r = redis.from_url(REDIS_URL, decode_responses=True)


def check_notice() -> dict | None:
    try:
        with urllib.request.urlopen(NOTICE_URL, timeout=2) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None  # no notice pending
        raise


def publish(action: dict) -> None:
    payload = {
        "instance_id": os.environ["EC2_INSTANCE_ID"],
        "job_id": os.environ["JOB_ID"],
        "attempt": int(os.environ.get("ATTEMPT", "1")),
        "action_time": action.get("time"),
        "detected_at": int(time.time()),
    }
    r.xadd(STREAM, {"message": json.dumps(payload)})


if __name__ == "__main__":
    while True:
        action = check_notice()
        if action:
            publish(action)
            # Give the sidecar a moment to flush before the instance dies.
            time.sleep(2)
            break
        time.sleep(5)
```

Two details are worth calling out.

First, the metadata endpoint returns HTTP 404 when no notice is pending. Treating that as an error will spam logs; treat it as the normal case.

Second, the notice does not guarantee a fixed grace period across all instance types and capacity sources. The documented maximum is two minutes, but actual time can be shorter. Publish the eviction event immediately on detection rather than waiting for a countdown.

## Step 2: compute a backoff that respects downstream health

A backoff function that only looks at the attempt count will still produce synchronized retries if many jobs share the same attempt number. Add jitter, and gate the base delay on the downstream error rate.

```go
// backoff.go
package retry

import (
	"math"
	"math/rand"
	"time"
)

// Config holds tunables. Defaults are illustrative starting points,
// not measured optima — tune them against your own downstream limits.
type Config struct {
	BaseDelay        time.Duration // e.g. 8 * time.Second
	MaxDelay         time.Duration // e.g. 15 * time.Minute
	JitterFraction   float64       // e.g. 0.5 for full jitter up to 50%
	ErrorMultiplier  float64       // e.g. 2.0 when downstream is unhealthy
	ErrorThreshold   int           // e.g. 3 recent 5xx/429 responses
}

func Backoff(attempt int, recentDownstreamErrors int, cfg Config) time.Duration {
	if attempt < 1 {
		attempt = 1
	}

	// Exponential component: base * 2^(attempt-1), capped.
	exp := float64(cfg.BaseDelay) * math.Pow(2, float64(attempt-1))
	if exp > float64(cfg.MaxDelay) {
		exp = float64(cfg.MaxDelay)
	}

	// Widen the window when the downstream is already struggling.
	if recentDownstreamErrors >= cfg.ErrorThreshold {
		exp *= cfg.ErrorMultiplier
		if exp > float64(cfg.MaxDelay) {
			exp = float64(cfg.MaxDelay)
		}
	}

	// Full jitter: uniform in [0, exp]. This is the part that breaks
	// the synchronization between workers that were evicted together.
	jitter := rand.Float64() * exp * cfg.JitterFraction

	return time.Duration(exp + jitter)
}
```

The `recentDownstreamErrors` value should come from a metric store, not from the worker's own experience. A worker that just got a 429 has only one data point; the controller can see the fleet-wide rate.

## Step 3: a controller that owns the retry timer

The controller reads eviction events, computes a delay, and re-enqueues the job after that delay. Because the controller is a separate process, it survives the death of the Spot fleet.

```go
// controller.go
package main

import (
	"context"
	"encoding/json"
	"log"
	"time"

	"github.com/redis/go-redis/v9"
)

type EvictionEvent struct {
	InstanceID string `json:"instance_id"`
	JobID      string `json:"job_id"`
	Attempt    int    `json:"attempt"`
}

type Enqueuer interface {
	Enqueue(ctx context.Context, jobID string, attempt int, delay time.Duration) error
}

type ErrorRateSource interface {
	RecentErrors(ctx context.Context, jobID string) (int, error)
}

func Run(ctx context.Context, rdb *redis.Client, q Enqueuer, errs ErrorRateSource) error {
	stream := "agent_evictions"
	group := "retry-controller"
	consumer := "controller-1"

	// Create the consumer group if it does not exist.
	_ = rdb.XGroupCreateMkStream(ctx, stream, group, "0").Err()

	for {
		res, err := rdb.XReadGroup(ctx, &redis.XReadGroupArgs{
			Group:    group,
			Consumer: consumer,
			Streams:  []string{stream, ">"},
			Count:    16,
			Block:    5 * time.Second,
		}).Result()
		if err != nil {
			if err == redis.Nil {
				continue
			}
			log.Printf("XReadGroup: %v", err)
			time.Sleep(time.Second)
			continue
		}

		for _, s := range res {
			for _, msg := range s.Messages {
				var ev EvictionEvent
				raw, _ := msg.Values["message"].(string)
				if err := json.Unmarshal([]byte(raw), &ev); err != nil {
					log.Printf("bad event %s: %v", msg.ID, err)
					rdb.XAck(ctx, stream, group, msg.ID)
					continue
				}

				recent, err := errs.RecentErrors(ctx, ev.JobID)
				if err != nil {
					// Fail open with zero errors, but do not ack — retry later.
					log.Printf("error rate lookup failed: %v", err)
					continue
				}

				delay := Backoff(ev.Attempt, recent, DefaultConfig)
				if err := q.Enqueue(ctx, ev.JobID, ev.Attempt+1, delay); err != nil {
					log.Printf("enqueue failed: %v", err)
					continue // do not ack; the event will be redelivered
				}
				rdb.XAck(ctx, stream, group, msg.ID)
			}
		}
	}
}
```

Two correctness notes on the Redis Streams usage:

- `XReadGroup` with `>` delivers only messages that have not been delivered to any consumer in the group. Acknowledging with `XAck` after a successful enqueue is what prevents duplicate retries.
- If the controller crashes between `Enqueue` and `XAck`, the message will be redelivered and the job will be enqueued twice. Make the enqueue idempotent — for example, by keying the retry record on `(job_id, attempt)` so a second write is a no-op.

## Step 4: make the retry queue idempotent

Whichever store holds the pending retries (S3, DynamoDB, a database table, a queue with deduplication), the write must be safe to repeat. The simplest approach is a deterministic key:

```
retry/{job_id}/{attempt}
```

If the same key is written twice with the same payload, the second write is harmless. If the payload differs — say, because the computed delay changed — you have a race between two controller instances. Either use a conditional write (write only if the key does not exist) or accept last-write-wins and make the consumer tolerant of an early or late retry.

## How to measure whether this is working

There is no universal benchmark for this pattern; the numbers depend on your fleet size, downstream rate limits, and job duration. What you can do is instrument the specific behaviours that indicate the pattern is doing its job.

**Eviction rate.** Count eviction events per hour. On EC2, the instance metadata notice is the authoritative signal for a given instance. Aggregate across the fleet to get a rate. If the rate is much higher than you expect, check whether you are running in a capacity-constrained instance family or availability zone.

**Retry delay distribution.** Record the computed delay for every retry as a histogram. A healthy distribution is wide — that width is the jitter doing its work. A distribution that clusters at a single value means jitter is not being applied, or the attempt counts are all identical.

**Downstream error rate during retry windows.** This is the metric that matters most. Correlate downstream 5xx and 429 rates with spikes in retry volume. If retries and downstream errors rise together, the backoff is not providing enough separation.

**Queue depth and age.** For the buffer (Redis Stream, Kafka topic, etc.), track the consumer lag and the age of the oldest unprocessed message. A growing lag during an eviction wave means the controller cannot keep up.

A simple way to start: emit one counter per retry and one gauge for the delay, then plot them against your downstream error rate. You do not need a full observability stack to see the shape of the problem.

## Failure modes to plan for

### Consumer lag when the buffer's primary goes down

If the buffer is a single-primary Redis instance and that primary is in an availability zone that becomes unavailable, the consumer stops until failover completes. During that window, eviction events accumulate and retries are delayed for every job in the queue.

Mitigation: run the buffer with multi-AZ failover, and alert on consumer lag rather than only on buffer availability. A lag alert fires earlier and is more actionable.

### Duplicate retries from at-least-once delivery

Redis Streams, Kafka, and SQS all deliver at least once. Any of them can redeliver a message after a consumer crash. If the enqueue step is not idempotent, the job runs twice.

Mitigation: deterministic keys as described above, plus a consumer-side check that the job has not already completed. For jobs with side effects (writes to external systems), keep an idempotency key with the downstream as well.

### Clock skew between the eviction notice and the retry timer

The eviction notice carries a timestamp from the provider. The controller's delay is computed from its own clock. If the two clocks differ, the effective delay differs from the intended one. In practice, the skew is small, but it is worth monitoring if you have strict ordering requirements.

Mitigation: run NTP or chrony on all hosts, and monitor offset. If you need strict ordering, use a logical clock (the attempt counter) rather than wall-clock time to sequence retries.

### Backoff that grows without bound

If the downstream stays unhealthy, an uncapped exponential backoff will eventually schedule retries hours or days out. Jobs appear to vanish.

Mitigation: cap the delay at a value your SLA can tolerate, and route jobs that exceed a maximum attempt count to a dead-letter queue with an alert. The dead-letter queue is the signal that something needs human attention; without it, the failure is silent.

## When not to use this pattern

This approach trades operational complexity for cost savings. It is a poor fit when:

- **The workload is latency-sensitive.** If a job must complete within seconds, a retry delay measured in tens of seconds violates the SLA. Use On-Demand capacity or a reserved pool.
- **The downstream has very tight rate limits.** If the downstream allows only a small number of requests per minute per tenant, retries from a large fleet will exhaust the budget regardless of jitter. Fix the rate limiting at the caller side first.
- **You cannot tolerate any data loss.** Spot eviction can kill a job mid-execution. If the job is not checkpointed, its work is lost. For non-idempotent jobs, that is a correctness problem, not a cost problem.
- **You do not have the operational capacity to run the buffer and controller.** A managed queue with built-in retry and dead-letter support may be a better fit if you do not want to operate Redis or Kafka.

## FAQ

**What is the smallest sensible base delay?**
It depends on the downstream's rate limit and the size of the fleet. As a starting point, pick a base delay that is at least as long as the time it takes the downstream to recover from a burst — often tens of seconds. A base delay in the low single-digit seconds is usually too short for a large fleet, because even with jitter the retries arrive close together.

**Should the retry live in the worker or in a separate controller?**
In a separate controller. A retry timer inside the worker dies with the worker, which is exactly the failure mode you are trying to avoid. The worker should only publish the eviction event and exit.

**How do I test this without real evictions?**
Publish synthetic eviction events to the stream with a controlled job ID and attempt count, and point the error-rate source at a mock that returns a fixed value. Then assert on the computed delay and on the number of enqueues. Because the delay computation is a pure function, it can be unit-tested directly without any infrastructure.

**Does this work with Kubernetes instead of a batch service?**
Yes. The controller can be a Kubernetes controller that watches for node termination events and creates Jobs with a delay, or it can be a standalone service that writes to whatever queue the Jobs consume. The key property — decoupling the retry timer from the evicted pod — is the same.

## What to do in the next 30 minutes

Pick one Spot-backed workload and add a single counter that increments every time a worker starts a retry. Emit it with the attempt number as a label. Run the workload for a day, then plot retries per minute against your downstream error rate.

If the two lines move together, your retry policy is amplifying downstream problems and is worth fixing. If they are independent, the current policy is providing enough separation and you can leave it alone. This measurement takes less time than implementing the full pattern and tells you whether the pattern is needed at all.
