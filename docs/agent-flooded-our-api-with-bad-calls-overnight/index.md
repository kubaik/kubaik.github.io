# Stopping a runaway polling agent from flooding your API

An agent that polls for state, retries on errors, or reacts to events can generate enormous call volume while every response looks healthy. Endpoints return 200, latency stays low, and no 5xx errors appear, so the usual alerting stays silent. The failure lives in the caller's loop, not in the API server, and it usually surfaces first as a cost anomaly rather than an incident.

## Why the error is confusing

When a background agent starts making thousands of low-value API calls, the first symptom is typically a bill several times higher than normal. What makes it hard to diagnose is that the API itself appears fine: endpoints return 200, logs show low latency, and no 5xx errors appear. The real issue is the volume and the business value of those calls.

A common trap is assuming the agent is just "busy." Teams often blame rate limits, upstream timeouts, or a misconfigured cron job. None of those explain why the calls have no business impact. The part that trips people up is that the API is technically working, so alert thresholds that fire on 5xx never trigger. The outage isn't in the API server — it's in the event loop driving the calls.

The most typical scenario is an agent that polls every minute for status updates, but the status never changes. Instead of sleeping or backing off, it keeps retrying with exponential backoff that never caps, or it ignores 429 responses and keeps hammering. By morning, the logs show hundreds of thousands of calls to `/status`, all returning 200 with identical JSON: `{"status": "pending"}`.

## What's actually causing it

The root cause is usually one of three patterns:

1. **Unbounded retry loops** where the agent ignores HTTP 409, 429, or 503 responses and keeps retrying with the same parameters. In Python, using `requests` without a `Retry` adapter, or in a serverless function without a `max_retries` setting on the SDK client, is the usual culprit. SDK defaults typically cap retries at a small number with jitter, but if the upstream keeps returning 409, the agent keeps looping.

2. **Missing or uncapped exponential backoff** in agents that poll for state changes. A common failure mode is a loop like:

```python
while True:
    r = requests.get("https://api.example.com/job/123/status")
    if r.json()["status"] != "done":
        time.sleep(1)
```

That's 86,400 calls per day if the job never finishes. Even with a 60-second sleep, it's still 1,440 calls per day — fine for one customer, expensive if the agent runs once per invocation and you have hundreds of customers.

3. **Event-driven fan-out without rate limiting or idempotency.** A buggy filter policy that matches every event, combined with a handler that doesn't deduplicate or throttle, can fan out thousands of identical events. The symptom is thousands of identical log lines in one hour, all triggering the same function with the same payload.

The deeper issue isn't the agent's logic — it's the lack of **guardrails around event volume**. The agent is doing exactly what it was told to do. The problem is that nobody told it to stop.

## Fix 1 — bounded retries and capped backoff

The most common cause is an agent that retries indefinitely on 409 or 429 responses. The fix is to add a bounded retry policy using the SDK's built-in retry configuration. In Python with boto3:

```python
aws_config = Config(
    retries={"max_attempts": 3, "mode": "adaptive"}
)
s3 = boto3.client("s3", config=aws_config)
http = requests.Session()
adapter = HTTPAdapter(max_retries=Retry(total=3, backoff_factor=0.3))
http.mount("https://", adapter)
```

This caps retries at 3 attempts, with exponential backoff starting at 100 ms and doubling each time. The `adaptive` mode in boto3 also respects `Retry-After` headers, so if the upstream returns 429 with a `Retry-After: 5` header, the SDK waits 5 seconds before the next attempt.

For agents that poll for state, switch from a fixed sleep to a capped exponential backoff. A typical pattern:

```python
def poll_with_backoff(url, max_polls=100, initial_delay=1.0):
    delay = initial_delay
    for _ in range(max_polls):
        r = requests.get(url)
        if r.status_code == 429:
            retry_after = int(r.headers.get("Retry-After", delay))
            time.sleep(retry_after)
            continue
        if r.status_code >= 500:
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
            continue
        # Success or non-retryable error
        return r
    raise TimeoutError(f"Gave up after {max_polls} polls")
```

Set `max_polls` based on your SLA. If a job should finish in 5 minutes, `max_polls=300` with a 1-second initial delay gives 5 minutes of polling. That drops daily calls from 86,400 to 300 per customer.

To size the cost impact for your own stack, work from your provider's published per-request price. If a provider charges $0.20 per 1M requests (an illustrative figure), 86,400 calls/day is roughly $0.017/day, while 300 calls/day is roughly $0.00006/day. Multiply by your actual invocation count and add compute duration, which usually dominates in serverless pricing.

## Fix 2 — idempotency for event-driven systems

The second most common cause is missing idempotency keys in event-driven systems. Many stacks use a pub/sub topic feeding a function, with a UUID as the message ID, but the handler doesn't deduplicate. A misconfigured filter that matches every event can fan out identical messages to thousands of invocations.

The symptom is identical log lines across hundreds of invocations:

```
START RequestId: a1b2c3d4
REPORT RequestId: a1b2c3d4 Duration: 123 ms Billed Duration: 123 ms
{"job_id": "123", "status": "pending"}
```

The fix is to add an idempotency layer. In Python with Redis:

```python
import redis
from uuid import uuid4

r = redis.Redis(host="localhost", port=6379, db=0)

class IdempotentAgent:
    def __init__(self, job_id):
        self.job_id = job_id
        self.lock_key = f"idempotency:{job_id}"

    def run(self):
        if r.setnx(self.lock_key, "1"):
            r.expire(self.lock_key, 3600)
            # Do the work
            return "processed"
        return "duplicate"
```

For pub/sub to function pipelines, use a durable store such as DynamoDB as the idempotency table. In Terraform:

```hcl
resource "aws_lambda_function" "worker" {
  function_name = "job-worker"
  handler       = "index.handler"
  runtime       = "python3.11"
  environment {
    variables = {
      IDEMPOTENCY_TABLE = aws_dynamodb_table.idempotency.name
    }
  }
}

resource "aws_dynamodb_table" "idempotency" {
  name           = "idempotency"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "idempotency_key"
  attribute {
    name = "idempotency_key"
    type = "S"
  }
}
```

Set the idempotency key to a hash of the event payload:

```python
def handler(event, context):
    payload_hash = hashlib.sha256(json.dumps(event).encode()).hexdigest()
    if dynamodb.get_item(
        Key={"idempotency_key": payload_hash}
    ):
        return {"statusCode": 200, "body": "duplicate"}
    dynamodb.put_item(Item={"idempotency_key": payload_hash, "result": "processed"})
    # Do work
```

This drops duplicate processing to zero. A DynamoDB table with on-demand billing costs roughly $1.25 per million write request units at published rates, so 10k writes/day is well under a dollar per month.

## Fix 3 — environment-specific runaway loops

The third cause is environment-specific: agents that run in CI systems or as a cron job on a small VPS. These environments often lack rate limiting, and the agent's loop runs on hardware that can make thousands of calls per second. The symptom is a bill spike with no correlation to serverless usage.

A typical failure mode is a cron job on a small VM:

```bash
# cronjob.sh
while true; do
  curl -s https://api.example.com/job/123/status > /tmp/status.json
  if [[ $(jq -r .status /tmp/status.json) != "done" ]]; then
    sleep 1
  else
    break
  fi
done
```

Without a delay between requests, this loop can issue hundreds or thousands of calls per second. Over 8 hours at 1,000 calls/second, that's 28.8 million calls. The fix is to add rate limiting at the OS level. In systemd, constrain the service:

```ini
# /etc/systemd/system/job-poller.service
[Service]
ExecStart=/usr/local/bin/job-poller.sh
CPUQuota=20%
MemoryMax=512M
Restart=always
```

Then add a rate limiter in the script using `--rate` in curl:

```bash
# job-poller.sh
while true; do
  status=$(curl --rate 10/1s -s https://api.example.com/job/123/status | jq -r .status)
  [[ "$status" == "done" ]] && break
  sleep 1
done
```

The `--rate` flag (available in curl 7.85 and later) enforces 10 requests per second, capping the blast radius to 86,400 calls/day. For CI pipelines, add a step that enforces a call budget rather than relying on a third-party action you haven't vetted:

```yaml
- name: Poll status
  run: |
    for i in $(seq 1 60); do
      status=$(curl -s https://api.example.com/job/123/status | jq -r .status)
      [[ "$status" == "done" ]] && exit 0
      sleep 5
    done
    exit 1
```

Document the rate limit in the script itself so nobody removes it without understanding the consequence:

```bash
# DO NOT REMOVE --rate without updating SLA. Job must finish in 8 hours.
```

## How to verify the fix worked

After applying the fixes, verify with three checks:

1. **Volume check**: In CloudWatch Metrics, filter the API's `RequestCount` with a `FunctionName` dimension. For a single customer's agent, expect <500 calls/day after the fix. If the count is still in the thousands, the agent is still unbounded.
2. **Latency check**: Use a synthetic canary to simulate the agent's poll loop. Before the fix, a 1-second poll loop shows elevated p99 latency from retries. After adding exponential backoff, p99 should drop.
3. **Cost check**: In AWS Cost Explorer, filter by Service = Lambda and UsageType = Requests. Compare the daily request count against the baseline week.

To measure the effect of an idempotency fix, query for duplicate log patterns:

```sql
fields @timestamp, @message
| filter @message like /duplicate/ or @message like /idempotency_key/
| stats count() by bin(5m)
```

If the count drops to zero after deploying the idempotency fix, the fix worked.

## How to prevent this from happening again

Prevention requires two layers: **guardrails** and **alerts**.

### Guardrails

1. **Rate limits at the agent level**: Add a decorator or middleware that enforces `max_calls_per_window` per customer. In Python:

```python
from functools import wraps
import time

class RateLimiter:
    def __init__(self, max_calls, window_seconds):
        self.max_calls = max_calls
        self.window = window_seconds
        self.calls = []

    def __call__(self, f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            now = time.time()
            self.calls = [t for t in self.calls if now - t < self.window]
            if len(self.calls) >= self.max_calls:
                raise Exception("Rate limit exceeded")
            self.calls.append(now)
            return f(*args, **kwargs)
        return wrapped

@RateLimiter(max_calls=10, window_seconds=60)
def poll_status(job_id):
    return requests.get(f"https://api.example.com/job/{job_id}/status")
```

Note that this in-process limiter is per-instance, not global. In a multi-instance deployment it only caps each instance's own calls; a shared counter (Redis, DynamoDB) is needed for a true global limit.

2. **Circuit breakers**: Use a library like `pybreaker` to stop the agent if the API returns too many 404s or 5xx in a window:

```python
from pybreaker import CircuitBreaker

breaker = CircuitBreaker(fail_max=5, reset_timeout=60)

@breaker
def call_api(url):
    return requests.get(url)
```

If the breaker trips, the agent stops making calls until the upstream recovers.

### Alerts

Set two alerts in CloudWatch:

1. **Volume spike**: Alert when `RequestCount` > 10× the baseline for a given API endpoint. Baseline is the 7-day median. In Terraform:

```hcl
resource "aws_cloudwatch_metric_alarm" "api_spike" {
  alarm_name          = "api-volume-spike"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "1"
  metric_name         = "RequestCount"
  namespace           = "AWS/ApiGateway"
  period              = "300"
  statistic           = "Sum"
  threshold           = var.baseline * 10
  alarm_description   = "API volume spike detected"
  dimensions = {
    ApiName = "prod-api"
  }
}
```

2. **Cost anomaly**: Alert when daily Lambda cost > 3× the 7-day median. AWS Cost Anomaly Detection supports a percentage threshold of baseline.

Combine these with your on-call paging integration so the engineer gets paged when the agent spins up.

## Related errors you might hit next

1. **Cache stampede**: After adding a cache for `/status`, the first request after expiry triggers many concurrent calls. The symptom is 5xx errors with `TooManyRequests` in the response. Fix: use a lock or queue to serialize cache rebuilds.
2. **Thundering herd**: A CronJob kicks off at 00:00 UTC, but your agent sleeps for 1 second between polls. All customers poll at the same time, overwhelming the API. Fix: add jitter to the sleep interval:

```python
import random
sleep_time = max(1, random.gauss(60, 10))
time.sleep(sleep_time)
```

3. **Deduplication race**: Two agents process the same message simultaneously because the idempotency check happens after the handler starts. The symptom is duplicate side effects (e.g., two emails sent). Fix: use a conditional write in DynamoDB with `ConditionExpression`:

```python
def handler(event, context):
    payload_hash = hashlib.sha256(json.dumps(event).encode()).hexdigest()
    try:
        dynamodb.put_item(
            Item={"idempotency_key": payload_hash, "result": "processed"},
            ConditionExpression="attribute_not_exists(idempotency_key)"
        )
    except dynamodb.meta.client.exceptions.ConditionalCheckFailedException:
        return {"statusCode": 200, "body": "duplicate"}
    # Do work
```

4. **SDK version skew**: Agents running on older runtimes use SDK clients without retry configuration. The symptom is retries that never back off, even when the upstream returns 429. Fix: pin the runtime to a supported version and set the retry config in code.

## When none of these work: escalation path

If the volume spike persists after applying all three fixes, escalate with the following diagnostic data:

1. **Logs Insights query** for the agent's log group, filtered to the last 6 hours:

```sql
fields @timestamp, @message
| filter @message like /job_id/ or @message like /status_code/
| stats count(*) as call_count by bin(1m)
| sort @timestamp desc
```

2. **Cost anomaly report** from AWS Cost Explorer, showing the spike's start time and duration.
3. **Agent configuration** (Terraform or Dockerfile) and version of the SDK used.

Open an internal ticket with the title: "Agent volume spike – check retry config and idempotency". Attach the logs and cost data. If the issue is upstream (e.g., the API's 429 responses are malformed), escalate to the API team with the exact error response:

```json
{
  "error": "RateLimitExceeded",
  "retry_after": "invalid"
}
```

If the agent is running on a cron job or VM, package the environment details (crontab, systemd unit, Docker image tag) and open an infra ticket.

## Frequently Asked Questions

**Why did my agent start making so many calls overnight?**

Most teams hit this when an upstream API starts returning non-retryable errors (409 Conflict or 429 Too Many Requests) and the agent's retry logic doesn't respect those responses. The agent keeps retrying with the same parameters, often because the retry configuration is missing or the SDK's defaults are too permissive. Modern SDKs cap retries at a small number, but if the upstream returns 409, the SDK may still retry without backoff unless you set adaptive mode.

**How do I know if my agent is the problem?**

Check CloudWatch Metrics for the API's `RequestCount` dimension. If you see a 3–4× spike in calls to a single endpoint (e.g., `/status`) with a pattern like 1 call every second, that's a smoking gun. Pair it with the agent's log group and look for repeated calls with the same `job_id` and `status: pending`. If the API's error rate hasn't changed, the issue is volume, not correctness.

**What's the fastest way to cap the calls without rewriting the agent?**

Add a rate limiter at the infrastructure layer. For a serverless function, set `ReservedConcurrency` to 1 and use a shared-store rate limiter in front of the agent. For a cron job on a VM, add `--rate 10/1s` to the curl command or enforce a call budget in your CI step. These changes deploy in minutes and can drop volume by 95% immediately.

**Should I use Redis or DynamoDB for idempotency?**

Use DynamoDB if your stack already uses it for persistence. A single table with `idempotency_key` as the hash key and TTL set to 24 hours costs well under $1/month for 10k writes/day. Use Redis if you need sub-millisecond latency or are already running it for caching. The choice depends on your existing infra, not performance — both scale to tens of thousands of writes/day without breaking a sweat.

## Decision checklist before shipping an agent

- Does every loop have a hard iteration cap (`max_polls`) derived from an SLA?
- Does every retry path have a bounded attempt count and a capped backoff ceiling?
- Does every event handler deduplicate on a payload hash with a conditional write?
- Is there a rate limiter that is global across instances, not just per-process?
- Is there a volume alarm at 10× the 7-day median for the endpoint?
- Is there a cost anomaly alarm at 3× the 7-day median?
- Are runtime and SDK versions pinned and supported?

## What's worth remembering

- The agent is doing exactly what it was told to do. The problem is that nobody told it to stop.
- Bounded retries and capped exponential backoff are not optional features — they're core guardrails.
- Idempotency is not a nice-to-have if your agent processes events that can arrive multiple times.
- Rate limiting at the agent level is cheaper than debugging a large bill at 3 AM.

The next time you write an agent that polls for state, add `max_polls=100` and `initial_delay=1.0` to the loop. Ship it with those defaults, then tune them based on your SLA. That single line is the difference between a quiet night and a wake-up call.

## Do this in the next 30 minutes

Open the file that contains your agent's polling loop and add a hard iteration cap plus a capped backoff ceiling, then commit it. If you can't find the loop in 30 minutes, that is itself the finding: your agent's control flow isn't documented, and your next step is to grep for `while True`, `sleep(`, and `requests.get` across the repo and inventory every unbounded loop.
