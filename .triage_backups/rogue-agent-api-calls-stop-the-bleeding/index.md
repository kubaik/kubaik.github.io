# Rogue Agent API Calls: Stop the Bleeding

## What a runaway agent actually looks like

A runaway agent does not usually announce itself with a stack trace. It announces itself on a billing dashboard. The characteristic signature is a sudden, sustained spike in invocation counts or external API usage while the application itself keeps returning success. Logs fill with `200 OK`. Nothing pages anyone. The system is working — just far too much, for no additional business value.

Two failure shapes dominate:

- **A crash** stops execution. You get an exception, a failed health check, an alert.
- **A runaway process** continues executing successfully. It re-fetches the same page, re-processes the same event, or retries an operation that will never succeed. The cost accrues silently until a quota is hit or someone reads the bill.

That distinction drives the whole diagnostic approach. You are not hunting a stack trace; you are hunting an endless stream of successful operations. The first observable symptoms are usually indirect:

- Upstream services begin rate-limiting you, so unrelated features get `429` responses.
- Latency climbs because thousands of concurrent invocations contend for the same database row or lock.
- External API usage charts go vertical, often flatlining exactly at a rate limit — which is itself a clue that the agent is not self-limiting.
- Response payloads become empty or identical across many calls, indicating no new data is being retrieved.

A useful early signal is the ratio of *successful calls with non-empty payloads* to *total calls*. If that ratio collapses while total volume rises, you have a loop, not a load increase.

## Root causes, ranked by how often they appear

Most incidents trace back to one of a small number of logic patterns. None of them are exotic.

**Unbounded pagination.** An agent fetches data page by page and never correctly identifies the end of the dataset. It requests the next page even when `next_token` is null, `has_more` is false, or the returned item list is empty. A job that should make 100 calls for 1,000 items makes 10,000 calls for the same 1,000 items.

**Retry storms.** A retry policy without a budget, a maximum attempt count, or exponential backoff will hammer an endpoint for a condition that will never resolve — a malformed request, a deleted resource, an expired credential. Worse, retries against a shared rate-limit pool can starve healthy agents and cause *them* to retry, compounding the problem.

**Feedback loops.** An agent processes a message, writes to a database, and that write triggers a webhook that re-queues the same message. Or a failed message is returned to the queue unmodified, so it is reprocessed forever. Each iteration is legitimate, fresh work — which is exactly why it is invisible.

**Self-expanding worklists.** The agent iterates a list of IDs and, during processing, generates new IDs appended to the same list. The workload grows without bound.

**Empty-dataset continuation.** A scheduled job queries a table that is empty — because of a migration, a timezone bug, or an upstream outage — and its "no rows? keep pulling from the source" branch fires repeatedly against data that does not exist yet.

## Fix 1 — bound every loop and retry

**Symptom:** Sustained spike in API usage, often pinned at a rate limit. Logs show a long stream of near-identical requests. Responses become empty or duplicated after a point.

**Cause:** Pagination that never terminates, or retries without a ceiling.

**Solution:** Make termination explicit and redundant. Check the API's own end-of-data signal, check for an empty result set, and impose a hard safety limit regardless. For retries, use exponential backoff with a strict maximum attempt count and a total time budget.

```python
# BAD: trusts a single end-of-data signal
def fetch_all_items_bad(api_client):
    all_items = []
    next_token = None
    while True:
        response = api_client.get_items(page_token=next_token)
        all_items.extend(response["items"])
        next_token = response.get("next_page_token")
        if not next_token:
            break
    return all_items


# GOOD: redundant termination checks plus a hard ceiling
MAX_ITEMS = 1_000_000

def fetch_all_items_good(api_client):
    all_items = []
    seen_tokens = set()
    next_token = None

    while True:
        response = api_client.get_items(page_token=next_token)
        items = response.get("items", [])
        all_items.extend(items)

        if not items:
            break

        next_token = response.get("next_page_token")
        if not next_token:
            break

        # Guard against an API that returns the same token forever
        if next_token in seen_tokens:
            break
        seen_tokens.add(next_token)

        if len(all_items) >= MAX_ITEMS:
            break

    return all_items
```

The `seen_tokens` set matters more than it looks. Some APIs return a non-null token indefinitely — including, in documented cases, the same token repeatedly. A check for `if next_token:` alone will not catch that.

## Fix 2 — idempotency and circuit breakers

**Symptom:** Calls are not identical. Parameters vary. The spike is intermittent or tied to specific inputs. Memory or CPU grows across a run even if it eventually completes.

**Cause:** Accidental recursion or an event-driven feedback loop. A message that fails is requeued unmodified. A write triggers a callback that triggers the write.

**Solution:** Make processing idempotent and track state externally. Record that an item has been processed *before* acknowledging it, and check that record before doing work. A fast key-value store is the usual mechanism.

```python
import os
import redis

redis_client = redis.StrictRedis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=6379,
    db=0,
)

PROCESSED_TTL_SECONDS = 3600

def process_message(message_body):
    message_id = message_body.get("id")
    if not message_id:
        return

    key = f"processed:{message_id}"
    # SET NX is atomic: only the first caller gets True.
    if not redis_client.set(key, "1", nx=True, ex=PROCESSED_TTL_SECONDS):
        return

    try:
        do_work(message_body)
    except Exception:
        # Release the claim so a genuine retry can happen.
        redis_client.delete(key)
        raise
```

Two details are easy to get wrong. First, use an atomic set-if-absent operation (`SET NX`), not a `GET` followed by a `SET` — the read-then-write version has a race window. Second, the claim must be released on failure, or a transient error permanently marks the message as done.

For feedback loops where a write triggers a callback, the guard belongs at the storage layer: write a correlation identifier alongside the row, and have the callback check whether it is reacting to its own write.

## Worked example: sizing a budget from first principles

Guards are only useful if the numbers are chosen deliberately. Suppose a nightly enrichment job is expected to process 5,000 records. Each record needs at most 2 external API calls, and the API client retries up to 3 times per call.

- Expected calls: 5,000 × 2 = 10,000
- Worst-case calls with full retries: 10,000 × 3 = 30,000

A per-run budget of 30,000 is therefore the *legitimate* ceiling. Set the hard limit at, say, 40,000 — enough headroom for one unexpected retry wave, small enough that a runaway loop trips it within minutes rather than hours. If the job runs hourly instead of nightly, divide accordingly.

The same arithmetic applies to cost. If the per-call price is known, multiply the ceiling by that price to get the worst-case bill for one run. That number is the one to compare against your alerting threshold. A budget that cannot be expressed as a worst-case dollar figure is not a budget.

## Guardrails that hold when the agent logic is wrong

The most reliable posture is to assume the agent code is flawed and put the limits in infrastructure it cannot bypass. Three layers cover most cases:

1. **Per-run call budget.** A counter incremented atomically on every outbound call, compared against a configured ceiling. When exceeded, abort the run and exit non-zero.
2. **Circuit breaker.** Track consecutive failures against a dependency. After a threshold, stop calling entirely for a cooldown period rather than retrying into an unhealthy service.
3. **Queue-level rate limiting.** Cap global throughput so that retries from one agent cannot saturate a shared provider quota and starve others.

```python
import os
import redis
from openai import OpenAI

r = redis.Redis.from_url(os.environ["REDIS_URL"], decode_responses=True)
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

BUDGET_KEY = "agent:budget:calls"
BREAKER_OPEN_KEY = "agent:breaker:open"
BREAKER_FAIL_KEY = "agent:breaker:failures"

CALL_BUDGET = int(os.environ.get("AGENT_CALL_BUDGET", "200"))
FAILURE_THRESHOLD = 5
BREAKER_COOLDOWN_SECONDS = 30


def guarded_call(prompt: str) -> str | None:
    # Fail closed: no budget left, no call.
    used = int(r.get(BUDGET_KEY) or 0)
    if used >= CALL_BUDGET:
        return None

    if r.get(BREAKER_OPEN_KEY):
        return None

    r.incr(BUDGET_KEY)
    r.expire(BUDGET_KEY, 3600)

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            timeout=10,
        )
        r.delete(BREAKER_FAIL_KEY)
        return resp.choices[0].message.content
    except Exception:
        fails = r.incr(BREAKER_FAIL_KEY)
        r.expire(BREAKER_FAIL_KEY, 60)
        if fails >= FAILURE_THRESHOLD:
            r.setex(BREAKER_OPEN_KEY, BREAKER_COOLDOWN_SECONDS, "1")
        raise
```

The `expire` on the budget key is what makes it a *rate* budget rather than a lifetime one. Without it, the first hour of operation permanently exhausts the allowance.

## How to measure whether the fix worked

Do not rely on the monthly bill to tell you. Instrument the following and compare before and after:

- **Total outbound calls per run.** Tag by agent name and run ID.
- **Useful-call ratio.** Successful calls whose response payload was non-empty and not a duplicate of a recent response. This is the single most diagnostic metric.
- **Calls per unit of work.** Calls divided by records actually processed. If this rises, a loop is forming.
- **Retry count distribution.** A long tail of high-retry calls points at a non-transient failure being retried.
- **Time to detection.** How long between the first anomalous call and the first alert. If this is measured in hours, the guardrail is incomplete.
- **Worst-case run cost.** Budget ceiling multiplied by per-call price, compared against your alerting threshold.

A practical check: run the agent against a deliberately empty dataset and confirm it exits promptly instead of continuing to call upstream. Run it against a dataset that triggers a persistent upstream error and confirm the breaker opens rather than retrying indefinitely.

## Decision checklist before deploying an agent

- Does every loop have a termination condition that does not depend on a single external signal?
- Is there a hard maximum on iterations, items, and outbound calls per run?
- Are retries bounded by count *and* time, with exponential backoff?
- Is processing idempotent, enforced by an atomic claim rather than a check-then-act?
- Can any write trigger a callback that re-enters this same code path? If so, is there a correlation guard?
- Does each agent have its own credential and quota, so one cannot starve another?
- Is there a circuit breaker on every external dependency?
- Does the run exit non-zero and alert when a budget is hit?
- Can the limits be changed by configuration without a code deploy?
- Has the empty-dataset case been tested explicitly?

## FAQ

**Doesn't a hard call limit break legitimate large jobs?**
It breaks jobs whose limits were set without arithmetic. Derive the ceiling from records × calls-per-record × max-retries, as shown above, and the limit sits above any legitimate run.

**Is a dead-letter queue enough to stop reprocessing loops?**
Only if failed messages actually land there. A queue that requeues on failure without a delivery-count check will loop regardless of whether a DLQ exists downstream.

**Why use an external store for idempotency instead of an in-memory set?**
In-memory state dies with the process. Serverless functions and restarted workers lose it, so the same message is processed again after every cold start.

**Should the budget key reset on a schedule or per run?**
Per run is more predictable for batch jobs; a rolling time window suits continuous agents. Either way, the reset must be automatic — a manual reset is a guardrail that will be forgotten.

**What if the provider itself is returning duplicate pages?**
Store the tokens you have already seen and stop when one repeats. This is a documented behavior in some paginated APIs and is not something a retry policy can fix.

## Do this in the next 30 minutes

Pick your highest-volume agent, add a per-run outbound call counter incremented atomically before each external request, and set the ceiling to the arithmetic worst case for one run. Make it exit non-zero and log the run ID when the ceiling is hit. That single change converts an unbounded spend into a bounded one, and it takes less time than reading the rest of the incident.
