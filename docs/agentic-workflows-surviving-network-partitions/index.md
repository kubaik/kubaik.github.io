# Agentic workflows: surviving network partitions

Most agentic workflows incidents trace back to a default nobody remembers choosing. Here's what actually worked, and why. The workaround gets copy-pasted forward long after the original reason is forgotten.

## The one-paragraph version (read this first)

Network partitions are not a rare edge case in Africa — they are the default operating condition for a large share of mobile and fixed-line traffic. An agentic workflow that assumes a stable, low-latency link to an LLM API will stall, duplicate side effects, or corrupt state the moment the link drops for 30 seconds. The patterns that survive are the ones that treat every tool call as a message that may need to be replayed, every LLM response as a candidate that may need to be re-validated, and every side effect as something that must be idempotent. This post walks through the mental model, a concrete worked example, the misconceptions that cause most of the damage, and the advanced patterns that only become necessary once the basics are solid. The part that trips people up is not the retry logic itself — it is the interaction between retries, LLM non-determinism, and non-idempotent tools, and that is what this post actually covers.

## Why this concept confuses people

Most agentic frameworks are built and tested in environments where a 200ms round trip to `api.openai.com` or `api.anthropic.com` is considered normal. In that world, a failed tool call is an exception. You catch it, you retry, you move on. The framework's default retry decorator handles it. The mental model is "network is reliable, errors are rare."

In much of Africa, the network is not reliable in that sense. A developer in Lagos on a typical mobile connection may see round-trip times to a US-East LLM endpoint that swing between 180ms and 4,000ms, with packet loss that causes TCP retransmits and occasional connection resets. A developer in Nairobi on a fiber link may have excellent latency to Europe but a saturated uplink during peak hours that causes 5–15% packet loss for minutes at a time. A field team in a rural area may be on a 3G link that drops entirely for 30–90 seconds when a tower hands off.

None of this is exotic. It is the normal shape of the network for a large fraction of the world's developers. But it breaks assumptions that agentic frameworks bake in silently:

- **Assumption 1: A tool call either succeeds or fails cleanly.** In reality, a tool call can time out on the client while succeeding on the server. The agent sees a failure; the side effect happened.
- **Assumption 2: The LLM response is deterministic enough to re-request.** It is not. Re-asking the same prompt can produce a different tool call, a different argument, or a different plan.
- **Assumption 3: The agent's local state is the source of truth.** If the agent is running on a device that loses connectivity, the local state and the server state diverge, and there is no automatic reconciliation.

A common failure mode here is an agent that calls a payment API, times out, retries, and charges the customer twice. The agent's logs show two failures and one success. The customer's bank shows two charges. The developer sees the logs and concludes the retry logic is broken. The retry logic is fine. The tool is not idempotent, and the agent had no way to know whether the first call actually landed.

This is the confusion: people treat "network partition" as a synonym for "slow network" and reach for timeout tuning. Timeout tuning helps with latency. It does nothing for the duplicate-side-effect problem, which is the one that actually costs money and trust.

## The mental model that makes it click

Think of an agentic workflow like a postal system, not like a phone call. A phone call is a live session: if the line drops, both parties know immediately and the conversation is over. A postal system is a message-passing system: you hand a letter to a courier, and you have no idea whether it arrived until you get a reply. If you send the same letter twice because you did not get a reply, the recipient gets two letters unless the letter itself carries a unique ID that lets them deduplicate.

That is the core shift. An agentic workflow over a flaky link is a distributed system with at-most-once or at-least-once delivery semantics, and you have to choose which one you are building. You cannot get exactly-once delivery for free. You get it by making the receiver idempotent and giving every message a stable identity.

The practical consequences:

1. **Every tool call needs an idempotency key.** Not a random UUID generated at call time — a key derived from the semantic intent, so that a retry of the same intent produces the same key. `sha256(agent_id + step_id + tool_name + canonical_args)` is the usual shape.
2. **Every LLM response needs to be treated as a proposal, not a fact.** The agent should validate the proposed tool call against the current state before executing it. If the state has moved (because a previous attempt actually succeeded), the proposal may be stale.
3. **Every workflow needs a reconciliation pass.** When connectivity returns, the agent should ask the server "what actually happened?" and reconcile its local view, rather than assuming its local view is correct.

This is not new distributed-systems theory. It is the same set of ideas behind idempotent HTTP methods, message deduplication in queues, and the outbox pattern. The novelty is that the "message" here is an LLM-generated tool call, which is non-deterministic and may be semantically equivalent to a previous call without being byte-identical.

## A concrete worked example

Consider a field-data collection agent. A survey worker in a rural area uses a mobile app that runs a small agent locally. The agent's job is to take a photo of a form, extract structured data using a vision model, validate the extraction against a schema, and submit the record to a central API. The app must work offline and sync when connectivity returns.

A naive implementation calls the vision model and the submission API directly from the agent loop. When the link drops mid-submission, the agent retries. The API receives two submissions. The backend has no deduplication, so the record appears twice. Downstream reporting double-counts.

The fix has three parts.

**Part 1: Local-first state with an outbox.** The agent writes every intended action to a local SQLite database before attempting it. The outbox row has a stable `action_id` derived from the record's content hash. The agent processes the outbox in order, marking rows as `pending`, `in_flight`, or `done`.

```python
import sqlite3, hashlib, json

def enqueue_action(conn, record: dict, tool: str, args: dict) -> str:
    # Stable ID: same record + same tool + same args => same action_id
    canonical = json.dumps({"record": record, "tool": tool, "args": args}, sort_keys=True)
    action_id = hashlib.sha256(canonical.encode()).hexdigest()[:32]
    conn.execute(
        "INSERT OR IGNORE INTO outbox (action_id, tool, args, state, attempts) "
        "VALUES (?, ?, ?, 'pending', 0)",
        (action_id, tool, json.dumps(args), ),
    )
    conn.commit()
    return action_id
```

The `INSERT OR IGNORE` is doing real work: if the same action is enqueued twice, the second insert is a no-op. That is the first layer of deduplication, and it happens before any network call.

**Part 2: Idempotent submission with a server-side key.** The submission API accepts an `Idempotency-Key` header. The server stores the key alongside the result for 24 hours. A retry with the same key returns the original result instead of creating a new record.

```javascript
async function submitRecord(record, actionId, baseUrl, token) {
  const res = await fetch(`${baseUrl}/records`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`,
      'Idempotency-Key': actionId,
    },
    body: JSON.stringify(record),
    // 8s is generous for a flaky link; the server-side key makes retries safe
    signal: AbortSignal.timeout(8000),
  });
  if (res.status === 409) {
    // Server already has this key; treat as success and fetch canonical state
    return { status: 'duplicate', record: await res.json() };
  }
  if (!res.ok) throw new Error(`submit failed: ${res.status}`);
  return { status: 'created', record: await res.json() };
}
```

The `409` branch matters. Some APIs return the original result with `200`; others return `409 Conflict` to signal "I have seen this key before." Either is workable, but the client has to handle it explicitly. A common bug is treating `409` as a fatal error and abandoning the action, which leaves the outbox row stuck in `in_flight` forever.

**Part 3: Reconciliation on reconnect.** When connectivity returns, the agent does not just flush the outbox. It first calls a `GET /records?since=<last_sync>` endpoint to pull the server's view, then reconciles. If a record exists on the server with the same `action_id`, the local outbox row is marked `done` without re-submitting. This handles the case where the original submission succeeded but the response was lost.

The reconciliation pass is the part people skip. It is also the part that saves the most duplicate records. In a typical field deployment, a meaningful fraction of "failed" submissions actually succeeded on the server; without reconciliation, those get re-submitted and duplicated.

## How this connects to things you already know

If you have worked with message queues, you already know most of this. RabbitMQ and SQS both have at-least-once delivery, and both push the deduplication problem to the consumer. The outbox pattern is standard in event-driven systems. The idempotency-key pattern is what Stripe has used for years, and what most payment APIs now require.

What is different in agentic workflows is the non-determinism of the "message producer." In a normal event-driven system, the producer emits a well-defined event. In an agentic system, the producer is an LLM that may emit a semantically equivalent but byte-different tool call on retry. That means you cannot rely on byte-identical deduplication keys. You need to canonicalize the tool call's arguments before hashing — sort keys, normalize whitespace, drop fields that do not affect the outcome (like a `timestamp` the model hallucinated).

The other difference is that the agent may not know the full set of side effects a tool call will have. A tool like `send_email` has one obvious effect. A tool like `run_sql` may have many. The safe default is to treat every tool as potentially non-idempotent and require an explicit idempotency key, rather than assuming the tool is safe to retry.

If you have worked with mobile sync (CouchDB, PouchDB, or any of the local-first frameworks), the reconciliation pattern is familiar. The agent's local outbox is a local replica; the server is the authoritative replica; sync is a reconciliation between them. The agentic twist is that the "document" being synced is a tool call, not a user edit.

## Common misconceptions, corrected

**Misconception 1: "A longer timeout fixes flaky networks."** It helps with slow networks, not flaky ones. If the link drops entirely, a 60-second timeout just means you wait 60 seconds before retrying. Worse, a long timeout increases the window during which the client does not know whether the server received the request. The fix is not a longer timeout; it is an idempotency key plus a reconciliation pass.

**Misconception 2: "The LLM will produce the same tool call if I re-prompt."** It usually will not, especially with `temperature > 0`. Even at `temperature=0`, providers do not guarantee bit-identical outputs across requests, and any change to the system prompt or tool schema shifts the distribution. Treat every LLM response as a fresh proposal and validate it against current state.

**Misconception 3: "Retries are safe if the tool is a GET."** Mostly true, but not always. A `GET` that triggers a side effect (a webhook, a cache warm, a downstream write) is not safe to retry blindly. The HTTP method is a hint, not a guarantee. If you control the tool, make side effects explicit and require an idempotency key for anything that writes.

**Misconception 4: "I can just run the agent on the server and avoid the problem."** This moves the partition to the client-server link, which is exactly where the problem already is. Running the agent server-side helps if the client is a thin UI, but the agent still has to call the LLM API and any external tools over the same flaky link. The partition does not disappear; it relocates.

**Misconception 5: "Exactly-once delivery is achievable with enough retries."** No. Exactly-once delivery is not achievable in a system that can lose messages. What you can achieve is exactly-once *effect*, by making the receiver idempotent. That is the only version of the guarantee that is actually available.

## The advanced version (once the basics are solid)

Once the outbox, idempotency keys, and reconciliation are in place, the next problems are subtler.

**Semantic deduplication.** Two tool calls may be semantically equivalent without being byte-identical. An LLM might call `create_ticket(title="Login broken", priority="high")` on the first attempt and `create_ticket(priority="high", title="Login broken")` on the retry. Canonicalize arguments (sort keys, normalize case, strip trailing whitespace) before hashing. For tools with free-text arguments, consider a semantic hash: embed the argument, and if the cosine similarity to a recent call exceeds a threshold (0.95 is a common starting point), treat it as a duplicate.

**Partition-aware planning.** An agent that knows it is offline can plan differently. Instead of a plan that requires three sequential API calls, it can produce a plan that batches local work and defers network calls. This is a planning-time decision, not a retry-time one, and it requires the agent to have a model of its own connectivity. A simple version: expose a `connectivity` tool that returns `online`, `degraded`, or `offline`, and include it in the system prompt so the model can condition its plan on it.

**Conflict resolution for concurrent edits.** If two agents (or an agent and a human) edit the same record while partitioned, you need a merge strategy. Last-write-wins is the simplest and the most likely to lose data. A better default for structured records is field-level merge with a vector clock or a per-field `updated_at`. For free-text fields, you need either a CRDT or a human review step. Do not pretend this is solved by timestamps alone; clock skew across devices is real and can be minutes.

**Backpressure and outbox growth.** An agent that is offline for hours can accumulate a large outbox. If the outbox grows unbounded, the device runs out of storage. Cap the outbox size, and when the cap is hit, either drop the oldest low-priority actions or refuse new ones. The right choice depends on the workflow; a survey app should refuse new submissions rather than drop old ones.

**Observability for partitioned agents.** Standard APM tools assume a stable connection to the collector. For partitioned agents, buffer telemetry locally and ship it on reconnect, with the same idempotency discipline as business actions. A common trap is a telemetry pipeline that floods the network on reconnect and starves the business outbox. Prioritize business actions over telemetry.

## Quick reference

| Pattern | What it solves | When it is worth the complexity |
|---|---|---|
| Outbox with stable action IDs | Duplicate actions from retries | Any workflow with non-idempotent tools |
| Idempotency-Key header | Duplicate side effects on the server | Any write API you do not fully control |
| Reconciliation on reconnect | Lost responses that actually succeeded | Any workflow that can lose a response |
| Argument canonicalization | Byte-different but semantically identical retries | Any workflow where the LLM generates arguments |
| Semantic dedup (embedding threshold) | Free-text argument drift | Workflows with long free-text tool arguments |
| Connectivity-aware planning | Plans that assume a stable link | Agents that run on mobile or field devices |
| Field-level merge + vector clock | Concurrent edits during partition | Multi-writer workflows |
| Outbox size cap with priority | Storage exhaustion on long partitions | Devices with limited storage |
| Local telemetry buffering | Telemetry that starves business traffic | Any agent with verbose logging |

## Frequently Asked Questions

**Why does my agent duplicate actions when the network drops?**
Because the agent cannot distinguish "the request never arrived" from "the request arrived but the response was lost." Both look like a failure to the client. The only reliable fix is an idempotency key that the server uses to deduplicate, plus a reconciliation pass that checks the server's actual state on reconnect. Retry logic alone cannot solve this; it can only make it worse by increasing the number of duplicate attempts.

**How do I make an LLM tool call idempotent when the LLM is non-deterministic?**
You make the *effect* idempotent, not the call. Derive a stable key from the semantic intent (record ID, tool name, canonicalized arguments) and pass it to the tool. The LLM can produce a different byte sequence on retry; as long as the canonicalized key matches, the server deduplicates. For free-text arguments, add a semantic dedup layer with an embedding similarity threshold, typically around 0.95.

**What is the difference between at-least-once and exactly-once delivery for agents?**
At-least-once means every action is attempted until acknowledged, so duplicates are possible. Exactly-once *delivery* is not achievable over a lossy link. Exactly-once *effect* is achievable by making the receiver idempotent. In practice, you build at-least-once delivery plus idempotent receivers, and the combination gives you exactly-once effects for the actions that matter.

**When should I run the agent on the server vs. on the device?**
Run it on the device when the workflow must continue during a partition (field data collection, offline-first apps). Run it on the server when the device is a thin client and can tolerate being offline (a chat UI that can queue messages). The partition does not disappear either way; it just moves. Server-side agents still call the LLM API over the same link, so they need the same idempotency discipline for their tool calls.

## Further reading worth your time

The outbox pattern is documented well in the microservices literature; search for "transactional outbox" and you will find implementations in most languages. Stripe's idempotency documentation is the canonical reference for the `Idempotency-Key` header and the `409 Conflict` behavior. For local-first sync, the CouchDB replication protocol and the newer local-first frameworks (Automerge, Yjs) are worth reading even if you do not use them, because they make the reconciliation model concrete. For the distributed-systems theory underneath, the original Gilbert and Lynch proof of the CAP theorem is short and worth the hour.

One thing to do in the next 30 minutes: open your agent's tool-call code and find every tool that writes to an external system. For each one, check whether it accepts an idempotency key. If it does not, add one — start with the payment or submission tool, since that is where duplicates cost the most. If you cannot change the tool, wrap it in a local deduplication layer keyed on `sha256(tool_name + canonical_args)` and log every call with that key. That single change will not solve partitions, but it will make the next duplicate visible instead of silent.


---

### About this article

**Written by:** [Kubai Kevin](/about/) — software developer based in Nairobi, Kenya, with 10+ years building production systems in fintech and AI.

**How this article was produced:** This site uses an automated LLM pipeline designed and maintained by the author. Topics are selected from real production experience. Drafts pass automated quality gates (minimum length, uniqueness, concrete metrics, versioned tools, code samples, absence of filler). Individual line-by-line human editing is not performed on every post before publication. Specific numbers, benchmarks and cost figures are illustrative; verify them against current official documentation before production use.

**Corrections:** Report errors via the [contact page](/contact/). Corrections are applied promptly.

**Last generated:** October 2026
