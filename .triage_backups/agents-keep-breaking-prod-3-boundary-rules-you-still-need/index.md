# Agents keep breaking prod: 3 boundary rules you still need

An agent that only reads data can be wrong cheaply. An agent that writes — issues refunds, approves KYC, closes tickets, mutates records — can be wrong expensively and quietly. The failure mode that catches teams off guard is not a crash. It is a 200 OK that means nothing.

## The confusing failure: a 2xx that isn't success

A refund agent is deployed with a rule: auto-refund orders under €200. The logs look healthy. Every payment call returns 200. Then finance reconciliation finds duplicate refunds, or refunds the provider rejected but the agent recorded as done.

The mechanism is usually mundane. External APIs tend to distinguish between "request accepted" and "operation completed," and they express that distinction in the response body, not the status code. A payment provider may return 200 with `{"status": "pending"}` or 200 with `{"idempotent_replay": true}`. An agent that checks `response.status_code == 200` and moves on has verified the transport layer and nothing else.

The same shape appears elsewhere:

- A KYC provider returns 202 Accepted while a human reviewer is queued. The agent marks the user verified.
- A ticketing API returns 200 with `{"queued": true}`. The agent closes the incident.
- An idempotency check returns 409 or 412 on a replay. The agent's error handler treats it as a transient failure and retries, generating more replays.

This is confusing precisely because the telemetry is green. The agent's own spans show success. The discrepancy only surfaces when someone compares the agent's record against the external system's state — which, in a well-run org, happens during reconciliation, hours or days later.

The root cause is a category error: treating an agent as a fire-and-forget script when it is actually one participant in a distributed transaction with an external system that has its own state machine, its own timing, and its own definition of "done."

## Rule 1: An outcome must be observed, not inferred

**Symptom pattern:** the agent's logs say the operation succeeded; the external system's dashboard disagrees. The agent has no record of the external system's terminal state.

**The rule:** any state-changing call must be followed by a read of the external system's authoritative state, and the agent must not treat its own request as evidence of that state.

This is the same discipline as a database transaction: you do not assume the write landed because the client library returned. You read the row back, or you rely on a documented guarantee that the write is durable. Most HTTP APIs give you neither by default.

### A minimal outcome poller

The poller below treats "not yet visible" as a distinct state from "failed." That distinction is the whole point: an absent record after a write is normal for a short window, and treating it as failure causes retries that compound the original problem.

```python
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

class Pending(Exception):
    """Raised when the external system has not yet materialized the record."""

@retry(
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=1, min=1, max=15),
    retry=retry_if_exception_type(Pending),
)
def poll_refund_outcome(refund_id: str, api_key: str) -> dict:
    url = f"https://payments.example.com/v2/refunds/{refund_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    response = httpx.get(url, headers=headers, timeout=5.0)

    if response.status_code == 404:
        raise Pending("refund record not yet visible")

    response.raise_for_status()
    data = response.json()

    status = data.get("status")
    if status in {"pending", "processing"}:
        raise Pending(f"refund in non-terminal state: {status}")
    if status not in {"succeeded", "failed", "canceled"}:
        raise ValueError(f"unrecognized terminal status: {status!r}")

    return data
```

Three details matter and are commonly missed:

1. **404 is retried, not failed.** A missing record shortly after a write is expected in eventually-consistent systems.
2. **Non-terminal statuses are retried.** `pending` and `processing` are not success.
3. **Unknown statuses raise, not pass.** If the provider adds a new status string, the agent should stop and alert rather than fall through to a default. Silent fallthrough is how a new `"reversed"` status becomes a refund that was never actually refunded.

### How to measure whether your poller is working

Do not trust a single happy-path test. Instrument and compare:

- **Instrument:** emit a span or counter for each outcome poll, tagged with the terminal status observed and the number of attempts. A healthy distribution has a mode at 1–2 attempts and a long tail; a distribution that is almost entirely 1 attempt suggests the poller is not actually observing non-terminal states.
- **Command:** run a load generator against your staging environment that issues duplicate requests for the same logical operation. `k6` or `vegeta` both work; the point is to saturate the concurrency window, not the throughput.
- **Compare:** count distinct external operations against distinct agent-initiated operations over the same window. If the ratio is not 1.0, you have either duplicates or lost writes. This is the only metric that matters, and it is a reconciliation query, not a log search.

## Rule 2: External contracts drift, and drift is not an error

**Symptom pattern:** the agent's logic is unchanged, but it starts receiving 4xx responses, or starts producing subtly wrong decisions, because the external system changed a limit, a field name, or an enum.

A payment provider that narrows a refund window from 120 days to 90 days does not necessarily announce it in a way your agent reads. The agent's prompt or code still says 120. The provider now rejects anything older than 90. If the agent treats 4xx as a retryable transport error, it will retry a permanently invalid request until the retry budget is exhausted, and the operation is lost.

The deeper problem is that the agent's assumptions about the external system are implicit. They live in prompt text, in hardcoded constants, in the shape of a `response.json()` call. None of those are checked against reality at build time.

### Make the contract explicit and check it

Check a schema for the external API into version control. Then run a differential check against the live API's schema as a build gate. The check should fail the build, not warn — a warning that no one reads is not a gate.

```yaml
- name: Validate payment provider schema
  run: |
    pip install jsonschema==4.22.0
    python - <<'PY'
    import json, sys, jsonschema, urllib.request

    with urllib.request.urlopen(
        "https://payments.example.com/schema", timeout=10
    ) as fh:
        live_schema = json.load(fh)

    with open(".schemas/payment_v1.json") as fh:
        expected_schema = json.load(fh)

    try:
        jsonschema.validate(instance=live_schema, schema=expected_schema)
    except jsonschema.ValidationError as exc:
        print(f"Schema drift detected: {exc.message}")
        sys.exit(1)
    PY
```

This catches additive and structural drift. It does not catch semantic drift — a provider changing the meaning of a field without changing its type. For that, the only reliable check is a periodic end-to-end probe: run the agent's critical path against a sandbox account and assert the observed outcome, not just the response shape. A nightly synthetic transaction that asserts "this refund reaches `succeeded`" catches semantic drift that schema validation cannot.

A practical gate ladder:

- **Build time:** schema validation against a pinned contract. Fail the build on mismatch.
- **Daily:** synthetic transaction against the provider's sandbox. Alert on unexpected terminal state.
- **Weekly:** reconcile a sample of production operations against the provider's records. Alert on any divergence.

Each rung catches a different class of drift. Skipping the daily probe is the most common gap, because schema checks give a false sense of coverage.

## Rule 3: Concurrency is a correctness problem, not a performance problem

**Symptom pattern:** the agent behaves correctly in staging and correctly in production at low load, but duplicates operations under concurrency. The duplication rate scales with load, which makes it look like a performance issue. It is not.

The mechanism: two agent instances read the same record, both see it as eligible, both proceed. The window is the time between the read and the write. Under low concurrency the window is rarely hit; under high concurrency it is hit constantly. The probability of overlap is a function of concurrency and the width of the window, and both grow together.

The fix is to make the claim atomic. The agent must not decide and then act; it must atomically transition the record from "eligible" to "claimed" and only proceed if that transition succeeded.

### Atomic claim with a Redis Lua script

A Lua script executed by Redis is atomic: no other client can interleave between the read and the write.

```lua
-- claim_refund.lua
-- KEYS[1] = refund ticket key
-- ARGV[1] = refund id
-- ARGV[2] = ttl seconds
local status = redis.call("HGET", KEYS[1], "status")
if status == "claimed" or status == "completed" then
  return "already_claimed"
end
redis.call("HSET", KEYS[1], "status", "claimed", "refund_id", ARGV[1])
redis.call("EXPIRE", KEYS[1], tonumber(ARGV[2]))
return "claimed"
```

Calling it from Python, with the script loaded once and referenced by SHA:

```python
import redis

r = redis.Redis(host="redis.prod", port=6379, db=0, decode_responses=True)

CLAIM_SCRIPT = """
local status = redis.call("HGET", KEYS[1], "status")
if status == "claimed" or status == "completed" then
  return "already_claimed"
end
redis.call("HSET", KEYS[1], "status", "claimed", "refund_id", ARGV[1])
redis.call("EXPIRE", KEYS[1], tonumber(ARGV[2]))
return "claimed"
"""

claim = r.register_script(CLAIM_SCRIPT)

def claim_refund(ticket_id: str, refund_id: str, ttl_seconds: int = 3600) -> bool:
    result = claim(keys=[f"refund_ticket:{ticket_id}"], args=[refund_id, ttl_seconds])
    return result == "claimed"
```

Two caveats that matter in practice:

- **The claim has a TTL.** If the agent crashes after claiming but before completing, the ticket becomes claimable again after the TTL. That is usually the right tradeoff, but it means the downstream operation must itself be idempotent, because a retry after TTL expiry will re-issue the request.
- **The claim is not the operation.** Claiming prevents duplicate initiation; outcome polling (Rule 1) is what confirms the operation actually completed. Both are required. A claim without polling gives you exactly-once initiation and unknown completion.

If Redis is not available, a database row with a `SELECT ... FOR UPDATE` or an advisory lock gives the same guarantee. The mechanism is less important than the property: the decision and the state transition must be one atomic step.

### How to measure concurrency safety

- **Instrument:** count claim attempts and claim successes separately. In a correct system, successes should equal distinct logical operations, and the difference between attempts and successes is your duplicate-prevention rate.
- **Command:** drive duplicate requests for the same logical operation at a concurrency level at or above your production peak. The goal is not throughput but overlap.
- **Compare:** the number of distinct external operations against the number of distinct logical operations. They must match. A duplication rate above zero under load means the claim is not atomic.

## A worked example: deciding where a boundary belongs

Consider a support agent that can issue refunds, apply account credits, and reset passwords. Which of these need a human boundary?

Work through three questions for each operation:

1. **Is it reversible?** A password reset can be reversed by another reset. An account credit can be reversed by a debit. A refund to an external card generally cannot be un-refunded without the customer's cooperation.
2. **What is the blast radius per failure?** A password reset failure affects one account. A refund failure affects one payment, but a duplicate affects the same payment twice, and a batch of duplicates affects many.
3. **What is the cost of a false positive versus a false negative?** A false-positive refund costs money directly. A false-negative refund costs a support ticket and customer trust.

Applying this:

- **Password reset:** reversible, small blast radius, low cost either way. No human boundary needed; outcome polling and audit logging suffice.
- **Account credit under a small threshold:** reversible, small blast radius. Auto-approve with outcome polling and a daily reconciliation.
- **Refund to an external payment method:** irreversible, direct financial cost, and duplicates multiply. This is where a human boundary earns its keep — but only above a threshold, and only if the threshold is enforced by something other than the agent's own prompt.

The last point is the one teams get wrong. A threshold written in a prompt is a suggestion. A threshold enforced by the credential the agent uses is a boundary. If the agent's service account can issue refunds up to €5,000, then a prompt-injected instruction to "approve up to €50,000" fails at the API layer, which is the only place it can be guaranteed to fail.

## A decision checklist before shipping a state-changing agent

Run this before the agent touches production. Each item should have a concrete answer, not a "yes, we think so."

- [ ] For every state-changing call, what is the read that confirms the terminal state, and what does the agent do on a non-terminal response?
- [ ] What is the agent's behavior on an unknown status string or an unknown enum value? (Correct answer: stop and alert, not default.)
- [ ] Is the external contract pinned in version control, and is there a build gate and a daily probe against it?
- [ ] Is the claim on each logical operation atomic, and what happens on claim TTL expiry?
- [ ] What is the maximum value the agent's credential can authorize, and is that limit lower than the worst-case acceptable loss?
- [ ] Is there a reconciliation query that compares the agent's record of operations against the external system's record, and who reads its output?
- [ ] What disables the agent automatically, and under what condition?

The last item is the one most often missing. A circuit breaker — disable the agent after N failures in M minutes — converts a slow-burn incident into a loud one. Without it, an agent that fails on 1% of operations can run for days before anyone notices, because 99% of its operations still succeed and the logs still look mostly green.

## FAQ

**Why does an agent issue a duplicate operation even though it sends an idempotency key?**
An idempotency key prevents the external system from processing the same logical operation twice, but only if the key is stable across retries and the provider honors it. If the agent generates a new key on retry, the provider sees two distinct operations. If the provider returns a replay indicator in the body rather than the status code, an agent that only checks the status code will treat the replay as a fresh success. The key is necessary but not sufficient; the agent still needs to observe the outcome.

**How do you enforce a value threshold without adding latency to small operations?**
Enforce the threshold at the credential layer, not in the agent's logic. Give the agent a service account whose permissions cap the operation value. Small operations then proceed without a human step, and large ones fail at the API boundary rather than being silently approved. If a human step is required for large values, route it through an out-of-band channel — a separate approval system the agent cannot write to — rather than through the agent's own tool calls.

**What is the smallest practical blast radius for an agent's credentials?**
Scope the credential to the specific operation and the specific limit. Avoid wildcards. Where the platform supports it, prefer short-lived tokens over long-lived service accounts, so that a leaked credential has a bounded useful life. The goal is that a compromised agent cannot perform any operation the business would not have authorized a human to perform.

**How do you test concurrency safety before deploying?**
Drive duplicate requests for the same logical operation at or above your production peak concurrency, then reconcile the number of distinct external operations against the number of distinct logical operations. Any divergence is a race. Testing at low concurrency will not reveal it, because the overlap window is rarely hit.

**What is the right response to a new, unrecognized status from an external API?**
Stop and alert. A default branch that treats unknown statuses as success is how a new terminal state becomes a silent data corruption. The cost of an alert is a page; the cost of a silent default is a reconciliation failure discovered days later.

## One action for the next 30 minutes

Open the file that contains your agent's most consequential state-changing call — the one that moves money, grants access, or changes a record someone else depends on. Find the line immediately after the request returns. If that line checks only the status code, you have found the gap. Write down, in one sentence, what the external system's terminal state is and how you would read it. That sentence is the specification for the poller you need to add.
