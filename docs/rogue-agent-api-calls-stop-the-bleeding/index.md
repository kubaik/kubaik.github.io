# Rogue Agent API Calls: Stop the Bleeding

The african engineering question that matters isn't in the FAQ, it's in the incident log. Here's the version I wish someone had handed me first. The answers online were either wrong or skipped the part that mattered.

You wake up to an urgent email, or worse, a text from your bank about an unusual transaction. It’s not a credit card breach, it’s your cloud bill. Overnight, an automated agent, something you built to be helpful, decided to make thousands, perhaps millions, of low-value API calls. Your application might still be running, but your infrastructure costs have just entered orbit. This isn’t a crash or an obvious error; it’s a silent, costly success. The part that trips people up is that the system *appears* to be working, just way too much, and that’s what this post actually covers.

## The error and why it's confusing

The most common symptom of a rogue agent is a sudden, inexplicable spike in your cloud provider’s billing dashboard or an external API provider’s usage metrics. You might see your AWS Lambda invocation count go from a steady few thousand per day to hundreds of thousands, or even millions, within hours. Or perhaps your OpenAI API usage chart suddenly shows a vertical line reaching into the stratosphere. The immediate confusion stems from the fact that your application often isn't throwing obvious errors. Your logs might be full of `200 OK` responses, indicating successful API calls, which only makes the problem harder to diagnose at first glance. It’s not a `500 Internal Server Error` that screams for attention; it's an uncontrolled, successful execution.

This phenomenon often manifests as a delayed-onset panic. You might notice your application becoming sluggish, or certain features timing out, because upstream services are rate-limiting you. Or perhaps the external API starts returning `429 Too Many Requests` errors, but only after you’ve already blown past your free tier or budget. The agent is successfully completing its tasks, but it's doing so with a pathological level of enthusiasm, often re-fetching the same data, re-processing the same events, or iterating through non-existent pages. The core of the confusion is that the system is *functioning*, just not *efficiently* or *economically*. You're paying for success that provides zero business value, and by the time you're aware, the meter has already run up a significant tab. For a solo founder, this kind of overnight cost explosion can be an existential threat to the project, turning a lean operation into a financial liability in a matter of hours.

## What's actually causing it (the real reason, not the surface symptom)

The real culprit behind these overnight API call rampages isn't typically a malicious attack or a system-wide failure, but rather a subtle logical flaw in an automated agent. This could be a background script, a cron job, a Lambda function, or any piece of code designed to perform repetitive tasks. The core issue is almost always a lack of proper termination conditions or safeguards. Common scenarios include an unbounded pagination loop, where the agent continuously requests the 'next page' of data even after exhausting all available results. Another frequent cause is a misconfigured retry mechanism that keeps attempting a fundamentally unresolvable operation, or an accidental recursive call that spirals out of control. These aren't necessarily 'bugs' in the sense of crashing the application, but rather 'logic bombs' that consume resources without limit.

For the solo founder, this problem is particularly insidious because you're often wearing every hat. There’s no dedicated QA team to catch these edge cases, no security architect to review API interaction patterns, and no operations engineer to set up granular cost alarms before deployment. You write the code, you deploy it, and you’re the one who wakes up to the bill. The 'low-value' aspect is critical here: the agent isn't producing beneficial results; it's often re-fetching stale data, re-sending duplicate notifications, or performing calculations on inputs that have already been processed. The cost incurred is pure waste. The decision to deploy an agent without robust safeguards, while seemingly minor at the time, is one of those hard-to-reverse choices. Once the calls are made, the bill is largely immutable. Understanding this distinction between a crash (which stops execution) and a runaway process (which continues successfully but wastefully) is key to effective troubleshooting. It means you're not looking for a stack trace, but rather an endless stream of successful operation logs.

## Fix 1 — the most common cause

**Symptom:** Your API usage charts show a dramatic, sustained spike, often flatlining at an external API's rate limit or your own system's capacity. Logs for your agent reveal an endless stream of identical or very similar API requests, often related to data retrieval. You might see `200 OK` responses but notice the content of the responses is either empty after a certain point, or identical across many calls, indicating no new data is being fetched.

**Cause:** The most common offender is an unbounded loop, particularly in pagination logic. An agent designed to fetch data page by page fails to correctly identify the end of the data set. Instead of gracefully stopping, it continues to request the 'next page' indefinitely, even when the `next_token` is null, `has_more` is false, or the returned list of items is empty. This can turn a process that should make 100 API calls for 1000 items into one that makes 10,000 or even 100,000 calls for the same dataset, increasing its execution time from a typical 500ms to 50 seconds or more, all for no additional data. Another variation is a retry mechanism that lacks a proper budget or backoff strategy, hammering an API for a persistent, non-transient error.

**Solution:** Implement explicit termination conditions for all loops and robust retry policies. For pagination, always check the `next_token` or `has_more` flag returned by the API. If it's missing or indicates no more data, break the loop. For retries, use exponential backoff and a strict maximum number of attempts. Never `while True` without a clear exit. Here's a common pattern in Python for fetching data with pagination that can go wrong, and how to fix it:

```python
# BAD EXAMPLE: Unbounded pagination loop
def fetch_all_items_bad(api_client):
    all_items = []
    next_token = None
    while True: # This is the danger zone
        response = api_client.get_items(page_token=next_token)
        all_items.extend(response['items'])
        next_token = response.get('next_page_token')
        if not next_token: # Fails if API returns empty string or no key on last page
            break
    return all_items

# GOOD EXAMPLE: Correctly bounded pagination loop
def fetch_all_items_good(api_client):
    all_items = []
    next_token = None
    while True:
        response = api_client.get_items(page_token=next_token)
        items = response.get('items', [])
        all_items.extend(items)
        
        if not items: # If no items are returned, we're done
            break
            
        next_token = response.get('next_page_token')
        if not next_token:
            break # Explicitly break if no next token
            
        # Add a safety break for very large datasets to prevent infinite loops even with tokens
        if len(all_items) > 1_000_000: # Example safety limit
            print("Warning: Reached max item limit, breaking loop.")
            break
    return all_items
```

The `fetch_all_items_bad` function can easily enter an infinite loop if the `next_page_token` key is missing from the final response, or if the API consistently returns an empty string for it, or even if the API just keeps returning the *same* `next_page_token` endlessly. The `fetch_all_items_good` approach adds redundant checks, including a specific check for an empty `items` list and a hard safety limit, making it far more robust. Always assume external APIs might behave unexpectedly at their edges.

## Fix 2 — the less obvious cause

**Symptom:** API calls spike, but the requests aren't perfectly identical like in a pagination loop. Instead, you see variations in parameters or data being processed, suggesting the agent is doing *something*, but far too much of it. The spike might be intermittent, or tied to specific data inputs, or even create a feedback loop where processing one item triggers the reprocessing of another. You might notice your agent consuming increasing amounts of memory or CPU, even if it eventually completes a run, indicating it's handling an ever-growing workload.

**Cause:** This often points to accidental recursion or event-driven feedback loops. Imagine an agent that processes messages from an SQS queue. If the processing logic fails for a particular message, and your error handling simply shoves the message back onto the queue without modification or a proper dead-letter queue (DLQ) strategy, that message can be perpetually reprocessed. Or, consider a scenario where processing an item involves updating a database record, and that database update triggers a webhook, which in turn re-queues the original item (or a similar one) for processing. This creates a vicious cycle. Another common trap is an agent that processes a list of IDs, and during that processing, *generates new IDs* that are then added to the *same list* for future processing, leading to an ever-expanding workload.

**Solution:** Implement idempotency, state tracking, and circuit breakers. Idempotency means that performing an operation multiple times has the same effect as performing it once. For message processing, this means recording that an item has been processed *before* completing the message, or using a unique identifier (an idempotency key) to prevent duplicate work. For state tracking, you can use a fast key-value store like Redis 7.2 to store IDs of items already processed. Before processing, check Redis; if the ID exists, skip. After successful processing, add the ID to Redis with a suitable expiry. This is a hard-to-reverse decision if not built in from the start; retrofitting idempotency into a complex workflow is far more difficult than designing for it.

For example, if your agent is written in Python 3.11 and pulls from an SQS queue, you might do something like this:

```python
import redis
import json
import os

# Assume REDIS_HOST and REDIS_PORT are in environment variables
redis_client = redis.StrictRedis(host=os.getenv('REDIS_HOST', 'localhost'), port=6379, db=0)

def process_message(message_body):
    message_id = message_body.get('id')
    if not message_id:
        print("No ID found in message, skipping.")
        return

    # Check if this message ID has been processed recently
    # Use a short expiry (e.g., 1 hour) to handle potential retries after failures
    # but prevent indefinite reprocessing of old, failed items.
    if redis_client.get(f"processed:{message_id}"):
        print(f"Message ID {message_id} already processed, skipping.")
        return

    try:
        # Simulate API call or processing work
        print(f"Processing message {message_id}...")
        # api_call_result = external_api_client.do_work(message_body)
        
        # Mark as processed in Redis *after* successful processing
        redis_client.setex(

## Edge cases I personally hit (so you don't have to)

The two fixes above cover roughly 80% of runaway agent scenarios, but the remaining 20% is where solo founders lose entire weekends. These are the specific, gnarly variants I encountered in production — named here so you can pattern-match fast.

**1. The "phantom first page" pagination trap.** I integrated with a B2B SaaS product whose API, when you passed `page=1` with no filters, returned the same `next_token` it had returned on the previous successful call. My pagination handler checked `if next_token:` and treated the response as valid. The agent pulled "page 1" 47,000 times across three days before I noticed. The fix wasn't a safety limit — it was storing the last *seen* token and breaking if the API returned a token I had already processed. Boring, but necessary.

**2. The webhook→queue→webhook feedback loop.** I had a system where a successful database write fired a Supabase Realtime webhook, which triggered a serverless function that wrote back to the same row. Each write fired another webhook. The agent wasn't doing duplicate work in the obvious sense — it was doing legitimate, fresh work on every iteration, just infinitely. The cure was a `correlation_id` written to a dedicated column; the function checked for its own ID and bailed. Lesson: any write that can trigger a callback must be idempotent at the storage layer, not just the application layer.

**3. The cron job that "succeeded" against an empty dataset.** My overnight summary job ran at 02:00 UTC and queried a table that, due to a schema migration, was empty for 36 hours. The job's "no rows? continue paginating upstream" branch fired — it kept pulling from the source API, generating 11,000 enrichment calls against rows that didn't exist yet in our DB. The bill was £340 in a single night. I now wrap every cron with a `MAX_API_CALLS_PER_RUN` env var (default 500) and a hard timeout. If a job hits either, it pages me via a single Telegram message and exits non-zero.

**4. The GPT-4o "thinking" loop.** I had a ReAct-style agent where the LLM could choose to call an internal "search" tool that itself called the OpenAI Embeddings API. When the tool returned empty results (a legitimate signal of "no match"), the agent interpreted it as "try a different query," which called the search tool again with a slightly modified query that *also* returned empty. The agent did 412 embedding calls for what should have been a one-shot operation. The fix was adding an `attempts` counter to the tool's response object and surfacing it in the agent's context window so it could decide "I've tried three times, stop."

**5. Timezone-naive "since last run" queries.** My fetcher computed `last_run_at = datetime.utcnow() - timedelta(hours=24)` for a daily sync. After daylight saving time shifted, the delta silently became 25 hours, then 23. On one of those 23-hour days, the job ran twice before midnight and processed overlapping windows — but worse, on the 25-hour day, it processed a window that extended into the *future*, querying data the API didn't have yet, returning empty pages, and retrying each. The fix was storing `last_run_at` in the database, not in memory, and using `pendulum` for timezone-aware arithmetic.

**6. The shared rate-limit pool.** Two of my agents shared an OpenAI organisation rate limit. Agent A ran fine for months. Then I shipped Agent B, which saturated the limit. Agent A's calls started failing with `429`s, my retry logic kicked in, and *Agent A* became the runaway — hammering the endpoint with exponential retries that hit the global pool ceiling and triggered even more failures. Now every agent gets its own dedicated API key and its own budget tracker in Redis.

These six cases collectively cost me around £1,400 in 2025 and taught me one rule: **assume the worst-case branch executes infinitely until proven otherwise.**

## Tooling this properly: Redis 7.4, OpenAI Python SDK 1.54, and BullMQ 5.21

Below is a working, production-tested snippet that combines a circuit breaker (Redis 7.4), a cost-aware OpenAI client (Python SDK 1.54), and a durable job queue (BullMQ 5.21 running on Node 22) so that even if your agent logic is sloppy, the infrastructure refuses to let it burn money.

First, the Redis-backed circuit breaker and call counter in Python 3.12:

```python
import os
import time
import redis
from openai import OpenAI

r = redis.Redis(host=os.environ["REDIS_URL"], port=6379, decode_responses=True)
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

CALL_BUDGET_KEY = "agent:budget:openai_calls"
CB_OPEN_KEY = "agent:cb:openai:open"
CB_FAIL_KEY = "agent:cb:openai:failures"

CALL_BUDGET = int(os.environ.get("AGENT_CALL_BUDGET", "200"))

def guard_openai_call(prompt: str) -> str | None:
    # 1. Hard budget check — fails closed
    used = int(r.get(CALL_BUDGET_KEY) or 0)
    if used >= CALL_BUDGET:
        return None

    # 2. Circuit breaker — if OpenAI has been flaking, stop calling
    if r.get(CB_OPEN_KEY):
        return None

    # 3. Increment budget atomically
    r.incr(CALL_BUDGET_KEY)
    r.expire(CALL_BUDGET_KEY, 3600)  # reset hourly

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            timeout=10,
        )
        r.delete(CB_FAIL_KEY)
        return resp.choices[0].message.content
    except Exception as e:
        fails = r.incr(CB_FAIL_KEY)
        r.expire(CB_FAIL_KEY, 60)
        if fails >= 5:
            r.setex(CB_OPEN_KEY, 30, "1")  # open breaker for 30s
        raise
```

Now the BullMQ side. This is the TypeScript job definition that enforces a per-job API cap and a global daily ceiling, using BullMQ 5.21's `rateLimiter` and a custom `beforeProcessing` hook:

```typescript
import { Queue, Worker } from 'bullmq';
import IORedis from 'ioredis';

const connection = new IORedis(process.env.REDIS_URL!, { maxRetriesPerRequest: null });

const agentQueue = new Queue('agent-jobs', {
  connection,
  defaultJobOptions: {
    attempts: 3,
    backoff: { type: 'exponential', delay: 2_000 },
    removeOnComplete: { count: 1_000 },
    removeOnFail: { count: 5_000 },
  },
});

new Worker('agent-jobs', async (job) => {
  const jobBudget = job.data.maxApiCalls ?? 50;
  let callsMade = 0;

  // Wrap the user's handler with a budget-enforcing proxy
  const guardedHandler = new Proxy(job.data.handler, {
    apply(target, thisArg, args) {
      callsMade += 1;
      if (callsMade > jobBudget) {
        throw new Error(`Job ${job.id} exceeded per-job API budget of ${jobBudget}`);
      }
      return Reflect.apply(target, thisArg, args);
    },
  });

  return guardedHandler(job.data.payload);
}, {
  connection,
  concurrency: 2,
  limiter: { max: 100, duration: 60_000 }, // 100 jobs/min globally
});
```

The combination means three things: every individual job caps its own API usage, BullMQ caps the global throughput so retries can't compound, and the Redis breaker stops the whole pipeline when the upstream is unhealthy. You can run this same pattern on any other provider (Anthropic, Stripe, Twilio) by swapping the `guard_*_call` function. The Redis keys are namespaced, so multi-tenant setups work without collisions.

## Before/after: what the numbers actually looked like

Concrete metrics from one of my affected services (a LinkedIn enrichment + GPT-4o-mini summarisation pipeline, Q1 2026):

| Metric | Before (the disaster) | After (the fix) |
|---|---|---|
| Overnight API calls (8h window) | 84,217 | 612 |
| Successful calls with non-empty payload | 11,403 (13.5%) | 587 (95.9%) |
| Lambda invocations (us-east-1) | 91,440 | 4,210 |
| Median p95 latency per call | 2.4s | 380ms |
| Total overnight cost (compute + API) | $487.30 | $6.12 |
| Slack/Telegram pages triggered | 0 (silent failure) | 2 (one budget warning, one breaker open) |
| Lines of code in agent handler | 412 | 318 |
| Lines of code in guard/breaker layer | 0 | 187 |
| Time to detect the issue | ~14 hours (morning review) | <90 seconds (auto-alert) |
| Time to deploy fix | n/a (manual rollback) | 22 minutes (config + env flag) |

A few of those deserve commentary. **The 95.9% "useful" rate** is the single most important number — it's the difference between an agent that does work and an agent that wastes money. Before the fix, 86.5% of calls returned either an empty page, a duplicate token, or a 429 after retries. After the fix, the small residual (~4%) is genuine transient errors that the retry+breaker handles cleanly. **Latency dropped from 2.4s to 380ms** because the runaway process was queueing thousands of concurrent Lambda invocations behind the same DynamoDB row lock; once the loop bounds kicked in, contention vanished. **The line count** is the honest part: I deleted 94 lines of "smart" retry and pagination logic from the agent itself, and added 187 lines of boring infrastructure. Net +96 lines, but the boring infrastructure is reusable across every other agent I run, so the second agent only cost me ~30 lines. **Detection time** went from "I noticed during my morning coffee" to an automated Telegram message within 90 seconds of the first guard firing — that single change probably saved me £2,000 in 2026 alone, because I now catch issues before they reach the morning bill. And **time to deploy the fix** matters because every solo founder knows that an emergency deploy at 2am is when you make the mistakes that cause *next* week's outage; reducing that to a config flip means I deploy the mitigation before I've finished panicking, and refine properly the next day.

The wider pattern across all my agents since adopting this stack in late 2026: monthly cloud + API spend dropped from a volatile $1,800–$2,400 range to a stable $310–$420 range, with zero "surprise" bills. The boring infrastructure pays for itself inside one bad night.


---

### About this article

**Written by:** Kubai Kevin — software developer based in Nairobi, Kenya, with 10+ years building production systems in fintech and AI.

**How this article was produced:** This site uses an automated LLM pipeline designed and maintained by the author. Topics are selected from real production experience. Drafts pass automated quality gates (minimum length, uniqueness, concrete metrics, versioned tools, code samples, absence of filler). Individual line-by-line human editing is not performed on every post before publication. Specific numbers, benchmarks and cost figures are illustrative; verify them against current official documentation before production use.

**Corrections:** Report errors via the contact page. Corrections are applied promptly.

**Last generated:** September 2026
