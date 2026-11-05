# Audit trails for AI agents: what breaks first

Most agent audit trails fail not because they log too little, but because they log the wrong shape. A JSON blob of the final answer plus a timestamp satisfies nobody: compliance cannot reconstruct *why* a decision was made, and on-call engineers cannot tell whether a failed tool call was a transient timeout or a policy violation. The audit trail has to be a first-class data product, not a log line appended at the end.

The constraint that reshapes everything: in Nigeria, Ghana, and East Africa, the agent may be running on a phone with intermittent 3G, the payment rail may be M-Pesa or Paystack with its own idempotency semantics, and the audit record may be the only evidence that a transaction was authorised. If the audit write is fire-and-forget, a dropped connection means the decision is unprovable. That is the real problem this article addresses: designing an audit trail that survives the network, satisfies a regulator, and still tells an engineer what went wrong at 2 a.m.

## The compliance-debugging gap is a schema problem

Documented behaviour for most logging libraries is straightforward: emit structured JSON, ship it to a sink, retain it. Production reality diverges in three places that matter for agents.

First, agents make *decisions*, not just requests. A decision has inputs (prompt, retrieved context, tool results), a policy in force at the time, and an outcome. Most logging setups capture the request and response but not the policy version. When a compliance reviewer asks "was this transaction allowed under the rules on the date it happened?", the answer is unrecoverable.

Second, tool calls are side effects. A payment initiation through Paystack or a mobile-money transfer via M-Pesa is not idempotent by default; the audit record must carry the idempotency key the agent used so that a retry can be reconciled against the original decision. Without it, the audit trail and the payment rail disagree, and finance spends a week matching rows.

Third, debugging needs the *negative* space. An engineer investigating a wrong answer wants to know which tools were *not* called, which documents were retrieved but discarded, and which policy branch was skipped. Standard application logs rarely record non-events; audit trails for agents must.

The gap, then, is not volume. It is that the record is shaped for a log aggregator, while the consumer is a regulator or a debugger who needs a causal chain.

## The mechanism: an append-only decision record with a policy snapshot

The recommended mechanism is a decision record: an append-only, immutable row per agent decision, written *before* the side effect and updated with the outcome after. It carries five things:

1. **Decision ID** — a UUID generated at the start, used as the idempotency key downstream.
2. **Inputs hash** — SHA-256 of the canonicalised prompt + retrieved context, so you can prove what the model saw without storing the full text (important for data-minimisation rules).
3. **Policy snapshot** — the exact rule set version evaluated, not a pointer to "current".
4. **Tool calls** — each with name, arguments hash, result status, and the external provider's reference (Paystack reference, M-Pesa receipt).
5. **Outcome and actor** — approved/denied/escalated, plus the human or service identity that triggered the run.

The append-only property is what makes it auditable. Use a database table with no UPDATE or DELETE grants for the application role; corrections are new rows that reference the original. In Postgres, a trigger that raises an exception on UPDATE or DELETE is a cheap guard:

```sql
CREATE TABLE agent_decisions (
    decision_id      UUID PRIMARY KEY,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    actor_id         TEXT NOT NULL,
    policy_version   TEXT NOT NULL,
    inputs_sha256    BYTEA NOT NULL,
    outcome          TEXT NOT NULL CHECK (outcome IN ('pending','approved','denied','escalated','failed')),
    outcome_at       TIMESTAMPTZ,
    metadata         JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE OR REPLACE FUNCTION reject_mutation() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'agent_decisions is append-only';
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER no_update BEFORE UPDATE OR DELETE ON agent_decisions
FOR EACH ROW EXECUTE FUNCTION reject_mutation();
```

That trigger is blunt: you cannot transition `pending` to `approved` in place. The right pattern is an event table (`agent_decision_events`) that references `decision_id`, and a view that folds events into current state. The decision row is inserted once; every subsequent change is an event. This is the same shape as event sourcing, and it is the shape that survives an audit.

For the tool-call linkage, store the provider reference. Paystack returns a `reference` on transaction initialisation; M-Pesa returns a `CheckoutRequestID` and later a `MpesaReceiptNumber`. If those are not in the audit record, reconciliation is guesswork.

## Implementation: writing the record before the side effect

The ordering matters more than the storage engine. A common failure is to call the payment API, then write the audit row. If the process dies between the two, you have a payment with no record. The fix is a write-ahead pattern: insert the decision with `outcome = 'pending'`, then call the provider, then append an event with the result.

```python
import hashlib, json, uuid
from datetime import datetime, timezone

def canonical_hash(payload: dict) -> bytes:
    # sort_keys ensures the same logical input hashes identically
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).digest()

def begin_decision(conn, actor_id: str, policy_version: str, prompt: str, context: list[str]) -> str:
    decision_id = str(uuid.uuid4())
    payload = {"prompt": prompt, "context": context}
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO agent_decisions
               (decision_id, actor_id, policy_version, inputs_sha256, outcome)
               VALUES (%s, %s, %s, %s, 'pending')""",
            (decision_id, actor_id, policy_version, canonical_hash(payload)),
        )
    conn.commit()
    return decision_id

def record_tool_call(conn, decision_id: str, tool: str, args: dict, provider_ref: str | None):
    event = {
        "type": "tool_call",
        "tool": tool,
        "args_sha256": canonical_hash(args).hex(),
        "provider_ref": provider_ref,
        "at": datetime.now(timezone.utc).isoformat(),
    }
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO agent_decision_events (decision_id, event) VALUES (%s, %s)",
            (decision_id, json.dumps(event)),
        )
    conn.commit()
```

Two details are load-bearing. The `canonical_hash` sorts keys so that the same logical input always hashes the same; without it, debugging "did the model see the same context twice?" becomes impossible. And the tool-call event stores the *hash* of arguments, not the arguments, which keeps sensitive data (account numbers, phone numbers) out of the audit store while still letting you prove which call was made if you have the original request elsewhere.

For low-bandwidth clients, the write-ahead record must be durable locally before the network call. On a mobile client, that means SQLite or an equivalent embedded store; the audit event is queued and flushed when connectivity returns. The server-side decision row is written on receipt, but the client keeps its own append-only queue so that a dropped connection does not lose the decision. This is the same reasoning behind offline-first sync: the record of intent must survive the transport.

One subtlety in the code above: `begin_decision` commits before the provider call, which is correct, but it means the insert must be fast and must not depend on anything downstream. If the insert fails, the agent must not proceed to the side effect. Treat the commit as a gate, not a notification.

## Measuring whether the trail actually satisfies anyone

Audit trails are like backups: the only meaningful test is a restore. For audit, the restore is a *reconstruction drill*. Once a quarter, pick a random decision from the previous month and ask two people to reconstruct it from the audit store alone: one playing compliance reviewer, one playing on-call engineer. If either cannot answer their questions, the trail is incomplete.

The compliance reviewer's questions are predictable: who authorised this, under which policy version, what data was used, and was the outcome recorded with a provider reference? The engineer's questions are different: what did the model see, which tools ran in what order, what was retried, and what was skipped?

A useful metric is *reconstruction time*: how long it takes to answer a standard question set from the audit store. Track it as a trend, not an absolute. If it is rising, the schema is drifting or the events are being written inconsistently.

A second metric is *orphan rate*: decisions with `outcome = 'pending'` older than the provider's maximum settlement window. A non-zero orphan rate means the write-ahead pattern is leaking — the process died between insert and outcome, and no reconciliation job closed the loop. For M-Pesa, settlement windows are documented per product; for Paystack, the transaction verify endpoint is the source of truth. A scheduled job that queries pending decisions and calls the provider's verify API is not optional; it is part of the audit trail.

A third is *hash stability*: sample inputs re-hashed at reconstruction time should match the stored hash. If they do not, your canonicalisation changed, and historical records are no longer verifiable. Version the canonicalisation function alongside the policy version.

To measure orphan rate without inventing a dashboard, run a query directly:

```sql
-- orphan rate: pending decisions older than 24h, as a fraction of all decisions in the window
SELECT
  count(*) FILTER (WHERE outcome = 'pending' AND created_at < now() - interval '24 hours')::numeric
    / NULLIF(count(*), 0) AS orphan_rate
FROM agent_decisions
WHERE created_at >= now() - interval '7 days';
```

The 24-hour threshold is a placeholder; set it to the longest documented settlement window among the providers you call. The point of the query is that it is cheap, it runs against the audit store itself, and it produces a number you can trend week over week.

## Failure modes: where audit trails break in the field

**The silent truncation.** Log pipelines have size limits. A prompt plus retrieved context can exceed them, and the pipeline truncates rather than rejects. The audit record then looks complete but is not. Store the hash and a pointer to the full payload in object storage with a retention policy, not the payload inline.

**The clock skew.** Mobile clients with intermittent connectivity often have wrong clocks. If `created_at` comes from the client, ordering is unreliable. Use server receipt time as the authoritative timestamp and keep client time as a separate field for debugging.

**The idempotency mismatch.** The agent generates a decision ID, but the payment provider generates its own reference. If the mapping is not stored, a retry after a timeout creates a second payment. The audit record must hold both, and the provider call must use the decision ID as the idempotency key where the provider supports it (Paystack accepts a `reference` you supply; M-Pesa's semantics differ by product and must be checked against the specific API).

**The policy drift.** The policy version is recorded as "v3", but v3 was edited in place. Now the record is a lie. Policies must be immutable and content-addressed: store the hash of the policy document, not just a label.

**The PII leak.** Debugging wants full prompts; data protection wants minimisation. The resolution is tiered storage: hashes and metadata in the hot audit store, full payloads encrypted in cold storage with access logging. The audit trail itself then records who accessed the cold store, which is often a compliance requirement in its own right.

**The dropped event on 3G.** The client queues events locally, but the queue is in memory. A process kill loses them. Persist the queue to disk before acknowledging to the agent loop.

The counterintuitive part is that the most common failure is not missing data — it is *inconsistent* data. Teams add fields over time, and six months later the audit store has three generations of schema with no version marker. Every event needs a `schema_version` field from day one.

### A worked example: tracing one payment decision

Consider a concrete sequence to see how the pieces interact. An agent is asked to pay a supplier invoice. Assume the following, purely illustrative, sequence:

1. Agent loop calls `begin_decision` with the prompt and retrieved invoice context. Row inserted with `outcome = 'pending'`, `policy_version = 'sha256:9f2c…'`, `inputs_sha256 = 0x4a1b…`. Commit returns.
2. Agent calls Paystack's initialise endpoint, supplying `reference = decision_id`. Provider returns its own reference and a pending status.
3. Agent appends a `tool_call` event with `provider_ref` set to the provider reference and `args_sha256` of the amount and recipient.
4. Webhook arrives; agent appends an `outcome` event with `approved` and the provider's final status.
5. A read view folds the two events: current state is `approved`, with both identifiers present.

Now break it deliberately. If step 2 times out, the reconciliation job queries for `pending` decisions older than the settlement window and calls the verify endpoint using the stored `reference` (which equals the decision ID). Because the provider was given the decision ID as its reference, the verify call is unambiguous: either the transaction exists under that reference or it does not. If it exists, the job appends the outcome event; if not, it appends a `failed` event. Either way the decision is closed and the orphan rate stays near zero.

If instead the agent had generated a fresh reference per retry, the verify call could not tell which attempt succeeded, and the audit trail would contain two pending decisions for one payment. That is the failure the idempotency key prevents, and it is worth stating plainly: the decision ID is not just a label, it is the join key between the audit store and the payment rail.

## When a full decision record is the wrong choice

Not every agent needs this. If the agent is a read-only assistant answering questions from public documentation, a standard structured log with request ID and latency is sufficient; the compliance surface is near zero and the debugging need is served by tracing. Building an append-only decision store for a chatbot is over-engineering that adds write latency and operational burden for no benefit.

The decision record earns its cost when at least one of these is true: the agent triggers a financial transaction, the agent's output is used in a regulated decision (credit, insurance, hiring), the agent acts on behalf of a user without per-action human review, or the agent operates across jurisdictions where you must prove which rules applied. In those cases the record is not optional, and retrofitting it after an incident is far more expensive than the write-ahead insert.

A middle path exists for agents that are *sometimes* consequential: record the full decision only when the outcome is `approved` or `escalated`, and record a lightweight hash-only row when the agent declines early. This keeps volume down while preserving the trail for the cases that matter. The risk is that you cannot debug the declines, so only take this path if declines are genuinely low-stakes.

A short decision checklist before committing to the full design:

- Does the agent move money or change state outside your system? If yes, full record.
- Is the output used in a decision a regulator supervises? If yes, full record.
- Does the agent act without a human approving each action? If yes, full record.
- Do you need to prove which policy applied on a past date? If yes, content-addressed policy snapshots are mandatory.
- Is the agent read-only over public data? A structured log with a request ID is enough.

## Frequently Asked Questions

**How long should agent audit records be retained?**
Retention is set by the regulation that applies to the decision, not by the logging stack. Financial records in most jurisdictions have multi-year retention requirements; the audit store should be designed so that retention is a policy on the cold tier, not a deletion job on the hot tier. Keep the hash and metadata hot, the encrypted payload cold, and make the retention period a configuration value with an audit trail of its own.

**Can I use my existing observability stack for agent audit trails?**
You can use it for the debugging half, but not for the compliance half. Observability backends are optimised for query speed and sampling; audit stores need immutability, no sampling, and access logging. A common pattern is to emit traces to the observability stack and decision records to a separate append-only store, linked by decision ID. Sampling must be off for the audit path.

**What is the minimum viable audit record for an agent that calls Paystack?**
At minimum: decision ID, actor, policy version hash, inputs hash, tool name, Paystack reference, outcome, and server timestamp. Without the Paystack reference you cannot reconcile; without the policy hash you cannot prove the rules; without the inputs hash you cannot prove what the model saw. Everything else is enrichment.

**How do I handle audit writes when the network is down?**
Persist the event to a local append-only queue before attempting the network write, and flush with exponential backoff. The server should accept out-of-order events and order by server receipt time, not client time. The client queue must survive process restarts, which means disk-backed storage, not memory.

**Should the audit store be the same database as the application?**
Not necessarily, but the write-ahead insert and the side effect must share a failure domain in one direction: the side effect must not happen unless the insert committed. If they are separate systems, you need an outbox or equivalent to guarantee the ordering. Keeping the decision table in the same database as the agent's operational state is the simplest way to get that guarantee, with the append-only trigger as the immutability guard.

## The next 30 minutes

Open your agent's main decision path and find the line where the side effect happens — the payment call, the ticket creation, the email send. Check whether an audit row is written *before* that line with a `pending` outcome. If it is not, add the write-ahead insert and the reconciliation job that closes pending decisions against the provider's verify endpoint. That single change converts your audit trail from a log of what happened into evidence that a decision was made, under a known policy, with a reference you can reconcile.
