# Agent flooded our API with bad calls overnight

Most recovered after guides assume a clean environment and a patient timeline. The gap between the demo and the incident report is where this actually lives. Here's what actually worked, and why.

## The error and why it's confusing

When a background agent started making thousands of low-value API calls overnight, the first symptom you usually see is a bill that’s 3–4× higher than normal. In 2026, with AWS Lambda at $0.20 per 1M requests and most indie SaaS stacks spending under $50/month on compute, a sudden jump to $300–$400 overnight is impossible to miss. What’s confusing is that the API itself appears fine: endpoints return 200, logs show low latency, and no 5xx errors appear. The real issue isn’t the API’s health—it’s the volume and the business value of those calls.

A common trap here is assuming the agent is just “busy.” Teams usually blame rate limits, upstream timeouts, or a misconfigured cron job. None of those explain why the calls have no business impact. The part that trips people up is that the API is technically working, so the alert thresholds that fire on 5xx don’t trigger. The outage isn’t in the API server—it’s in the event loop that drove the calls.

The most typical scenario is an agent that polls every minute for status updates, but the status never changes. Instead of sleeping or backing off, it keeps retrying with exponential backoff that never caps, or it ignores 429 responses and keeps hammering. By 6 AM, the logs show 400k calls to `/status`, all returning 200 with identical JSON: `{"status": "pending"}`.

## What's actually causing it (the real reason, not the surface symptom)

The root cause is usually one of three patterns:

1. **Unbounded retry loops** where the agent ignores HTTP 409, 429, or 503 responses and keeps retrying with the same parameters. In Python, using `requests` without a `Retry` adapter or in AWS Lambda without a `max_retries` in the SDK client is the usual culprit. The SDK’s default retry behavior in 2026 (boto3 1.34, AWS SDK for JavaScript 3.500) retries 3 times with jitter, but if the upstream keeps returning 409, the agent keeps looping.

2. **Missing exponential backoff** in agents that poll for state changes. A common failure mode is a loop like:

```python
while True:
    r = requests.get("https://api.example.com/job/123/status")
    if r.json()["status"] != "done":
        time.sleep(1)
```

That’s 86,400 calls per day if the job never finishes. Even if the sleep is 60 seconds, it’s still 1,440 calls per day—fine for one customer, disastrous if the agent runs once per Lambda invocation and you have 300 customers.

3. **Event-driven fan-out without rate limiting or idempotency.** In 2026, many indie stacks use SNS → Lambda for async tasks. A buggy filter policy that matches every event, combined with a Lambda that doesn’t deduplicate or throttle, can fan out thousands of identical events. The symptom is 10k identical events in CloudWatch Logs Insights in one hour, all triggering the same Lambda with the same payload.

The deeper issue isn’t the agent’s logic—it’s the lack of **guardrails around event volume**. The agent is doing exactly what you told it to do. The problem is that you didn’t tell it to stop.

## Fix 1 — the most common cause

Most teams hit this because their agent retries indefinitely on 409 or 429 responses. The fix is to add a bounded retry policy using the SDK’s built-in retry configuration. In Python with boto3 1.34:

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

Set `max_polls` based on your SLA. If a job should finish in 5 minutes, `max_polls=300` with a 1-second initial delay gives 5 minutes of polling. This drops daily calls from 86,400 to 300 per customer.

Cost impact: A single customer’s agent going from 86k calls/day to 300 calls/day drops Lambda cost from ~$1.70/day to ~$0.006/day. For 300 customers, that’s $510/month saved.

## Fix 2 — the less obvious cause

The second most common cause is missing idempotency keys in event-driven systems. In 2026, many stacks use SNS → Lambda with a UUID as the message ID, but the Lambda doesn’t deduplicate. A misconfigured SNS topic with a filter that matches every event can fan out identical messages to thousands of Lambdas.

The symptom is identical log lines across hundreds of Lambda invocations:

```
START RequestId: a1b2c3d4
REPORT RequestId: a1b2c3d4 Duration: 123 ms Billed Duration: 123 ms
{"job_id": "123", "status": "pending"}
```

The fix is to add an idempotency layer. In Python with Redis 7.2:

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

For SNS → Lambda, use Lambda’s built-in idempotency support with DynamoDB as the store. In Terraform:

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

This drops duplicate calls to zero. The DynamoDB table costs ~$1/month for 10k writes/day.

## Fix 3 — the environment-specific cause

The third cause is environment-specific: agents that run in GitHub Actions, CircleCI, or a cron job on a $5/month VPS. These environments often lack rate limiting, and the agent’s loop runs on hardware that can make thousands of calls per second. The symptom is a bill spike with no correlation to Lambda usage.

A typical failure mode is a cron job on a Ubuntu 22.04 VM with 2 vCPUs and 4 GB RAM:

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

That loop runs at ~1,000 calls/second on a $5 VM. Over 8 hours, it makes 28.8 million calls. The fix is to add rate limiting at the OS level. In systemd, create a service with CPU and IO limits:

```ini
# /etc/systemd/system/job-poller.service
[Service]
ExecStart=/usr/local/bin/job-poller.sh
CPUQuota=20%
MemoryMax=512M
Restart=always
```

Then add a rate limiter in the script using `rate` in curl:

```bash
# job-poller.sh
while true; do
  curl --rate 10/1s -s https://api.example.com/job/123/status | jq -r .status
  [[ "$status" == "done" ]] && break
  sleep 1
done
```

The `--rate` flag in curl 7.85+ enforces 10 requests per second, capping the VM’s blast radius to 86,400 calls/day. For GitHub Actions, add a step with `rate-limiting-action`:

```yaml
- name: Poll status
  uses: your-org/rate-limiting-action@v1
  with:
    url: "https://api.example.com/job/123/status"
    max-calls: 10
    interval: 1
```

This is a hard-to-reverse decision: once you ship a cron job with `--rate`, you can’t easily remove it without breaking the job’s SLA. Document it in the README and add a comment in the cron file:

```bash
# DO NOT REMOVE --rate without updating SLA. Job must finish in 8 hours.
```

## How to verify the fix worked

After applying the fixes, verify with three checks:

1. **Volume check**: In CloudWatch Metrics, filter the API’s `RequestCount` with a `FunctionName` dimension. For a single customer’s agent, expect <500 calls/day after the fix. If the count is still in the thousands, the agent is still unbounded.
2. **Latency check**: Use CloudWatch Synthetics to simulate the agent’s poll loop. Before the fix, a 1-second poll loop shows p99 latency of 1,200 ms (due to retries). After adding exponential backoff, p99 drops to 300 ms.
3. **Cost check**: In AWS Cost Explorer, filter by Service = Lambda and UsageType = Requests. A single customer’s agent going from 86k calls/day to 300 calls/day drops cost from ~1.70/day to ~0.006/day. For 300 customers, that’s $510/month saved.

Use CloudWatch Logs Insights to query for duplicate log patterns:

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

Set `max_calls` to the SLA’s expected rate. For a job that should finish in 10 minutes, 10 calls/minute is safe.

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

2. **Cost anomaly**: Alert when daily Lambda cost > 3× the 7-day median. In AWS Cost Anomaly Detection, set the threshold to 300% of baseline.

Combine these with a PagerDuty integration so the on-call engineer gets a call at 3 AM if the agent spins up.

## Related errors you might hit next

1. **Cache stampede**: After adding a cache for `/status`, the first request after expiry triggers 100 concurrent calls. The symptom is 5xx errors with `TooManyRequests` in the response. Fix: use a lock or queue to serialize cache rebuilds.
2. **Thundering herd**: A CronJob kicks off at 00:00 UTC, but your agent sleeps for 1 second between polls. All 1,000 customers poll at the same time, overwhelming the API. Fix: add jitter to the sleep interval:

```python
import random
sleep_time = max(1, random.gauss(60, 10))
time.sleep(sleep_time)
```

3. **Deduplication race**: Two agents process the same SNS message simultaneously because the DynamoDB idempotency check happens after the Lambda starts. The symptom is duplicate side effects (e.g., two emails sent). Fix: use a conditional write in DynamoDB with `ConditionExpression`:

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

4. **SDK version skew**: Agents running on older Lambda runtimes (Node.js 14, Python 3.8) use SDK clients without retry configuration. The symptom is retries that never back off, even when the upstream returns 429. Fix: pin the Lambda runtime to Python 3.11 or Node.js 20 LTS and set the retry config in code.

## When none of these work: escalation path

If the volume spike persists after applying all three fixes, escalate with the following diagnostic data:

1. **CloudWatch Logs Insights query** for the agent’s log group, filtered to the last 6 hours:

```sql
fields @timestamp, @message
| filter @message like /job_id/ or @message like /status_code/
| stats count(*) as call_count, avg(@message like /200/) as success_rate by bin(1m)
| sort @timestamp desc
```

2. **Cost anomaly report** from AWS Cost Explorer, showing the spike’s start time and duration.
3. **Agent configuration** (Terraform or Dockerfile) and version of the SDK used.

Open an internal ticket with the title: "Agent volume spike – check retry config and idempotency". Attach the logs and cost data. If the issue is upstream (e.g., the API’s 429 responses are malformed), escalate to the API team with the exact error response:

```json
{
  "error": "RateLimitExceeded",
  "retry_after": "invalid"
}
```

If the agent is running on a cron job or VM, package the environment details (crontab, systemd unit, Docker image tag) and open an infra ticket.

## Frequently Asked Questions

**Why did my agent start making so many calls overnight?**

Most teams hit this when an upstream API starts returning non-retryable errors (409 Conflict or 429 Too Many Requests) and the agent’s retry logic doesn’t respect those responses. The agent keeps retrying with the same parameters, often because the retry configuration is missing or the SDK’s defaults are too permissive. In 2026, the default retry behavior in boto3 1.34 and AWS SDK for JavaScript 3.500 caps at 3 retries, but if the upstream returns 409, the SDK may still retry without backoff unless you set `mode: "adaptive"`.

**How do I know if my agent is the problem?**

Check CloudWatch Metrics for the API’s `RequestCount` dimension. If you see a 3–4× spike in calls to a single endpoint (e.g., `/status`) with a pattern like 1 call every second, that’s a smoking gun. Pair it with the agent’s log group and look for repeated calls with the same `job_id` and `status: pending`. If the API’s error rate hasn’t changed, the issue is volume, not correctness.

**What’s the fastest way to cap the calls without rewriting the agent?**

Add a rate limiter at the infrastructure layer. For Lambda, set `ReservedConcurrency` to 1 and use a DynamoDB-backed rate limiter in front of the agent. For a cron job on a VM, add `--rate 10/1s` to the curl command or use `rate-limiting-action` in GitHub Actions. These changes take 5–10 minutes to deploy and can drop volume by 95% immediately.

**Should I use Redis or DynamoDB for idempotency?**

Use DynamoDB if your stack already uses it for persistence. A single table with `idempotency_key` as the hash key and TTL set to 24 hours costs ~$1/month for 10k writes/day. Use Redis 7.2 if you need sub-millisecond latency or are already running Redis for caching. The choice depends on your existing infra, not performance—both scale to tens of thousands of writes/day without breaking a sweat.

## Tools and versions mentioned

| Tool | Purpose | Version/Config | Docs Link |
|---|---|---|---|
| Python | Agent runtime | 3.11 | [docs.python.org/3.11](https://docs.python.org/3.11/) |
| boto3 | AWS SDK | 1.34 | [boto3.amazonaws.com/1.34](https://boto3.amazonaws.com/v1/3.34.0/) |
| AWS Lambda | Compute | arm64, 1024 MB | [aws.amazon.com/lambda](https://aws.amazon.com/lambda/) |
| Redis | Idempotency store | 7.2 | [redis.io/7.2](https://redis.io/docs/release-notes/7.2/) |
| DynamoDB | Idempotency store | PAY_PER_REQUEST | [aws.amazon.com/dynamodb](https://aws.amazon.com/dynamodb/) |
| systemd | Service manager | 252 | [freedesktop.org/software/systemd](https://systemd.io/) |
| curl | CLI request tool | 7.85 | [curl.se/docs](https://curl.se/docs/) |
| CloudWatch | Monitoring | 2026-01-01 | [aws.amazon.com/cloudwatch](https://aws.amazon.com/cloudwatch/) |

## Cost snapshot

| Scenario | Calls/day | Lambda cost (USD) | Notes |
|---|---|---|---|
| Unbounded retry loop | 86,400 | ~$1.70/day | 300 customers × $1.70 = $510/month |
| Exponential backoff | 300 | ~$0.006/day | 300 customers × $0.006 = $1.80/month |
| Idempotency + retry | 300 | ~$0.006/day | DynamoDB cost: $1/month for 10k writes |
| VM cron job | 28.8M | ~$150/day | $5 VM, but 28.8M calls cost ~$150 in Lambda |

The difference between the first and third rows is $508/month for 300 customers—enough to fund a part-time dev or a marketing experiment.

## What’s worth remembering

- The agent is doing exactly what you told it to do. The problem is that you didn’t tell it to stop.
- Bounded retries and exponential backoff are not optional features—they’re core guardrails.
- Idempotency is not a nice-to-have if your agent processes events that can arrive multiple times.
- Rate limiting at the agent level is cheaper than debugging a $500 bill at 3 AM.

The next time you write an agent that polls for state, add `max_polls=100` and `initial_delay=1.0` to the loop. Ship it with those defaults, then tune them based on your SLA. That single line is the difference between a quiet night and a wake-up call.


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

**Last generated:** August 05, 2026
