# APIs must die in 2026

## The conventional wisdom and where it stops holding

The standard guidance for HTTP APIs has been stable for a decade: keep handlers stateless, make writes idempotent, version the contract explicitly, and return synchronous responses. That guidance produced a large amount of working software, and it is still the right default for many systems.

It is also scoped to an environment that many teams do not actually operate in. The advice implicitly assumes a client that can hold a connection open, retry a failed request cheaply, and receive a response within a timeout window measured in single-digit seconds. When any of those assumptions break, the failure is rarely graceful. A single misconfigured keep-alive timeout can consume days of debugging because the symptom appears as an application-level error while the cause lives in the transport layer.

Two forces push against the classic model at once:

- **Network conditions.** Mobile connections in many markets combine high round-trip latency, meaningful packet loss during peak hours, and intermittent connectivity. Under those conditions, synchronous request/response with client-side retries produces duplicate work, hung requests, and timeouts that are indistinguishable from server errors.
- **Regulatory requirements.** Payment regulators increasingly mandate end-to-end encryption, auditable per-call records, and asynchronous notification of failed transactions. Asynchronous notification is fundamentally incompatible with a design that assumes the client is still connected when the outcome is known.

The important point is not that REST is wrong. It is that a synchronous, connection-coupled contract encodes assumptions that a regulated, high-latency environment violates by default. The rest of this article describes a design posture that survives those conditions, when it is worth the cost, and how to measure whether you need it.

## Failure modes that appear when you follow the standard advice

The following are recurring failure patterns, not incidents. Each one has a diagnosis path and a design response.

### 1. Success responses that carry no useful body

A load balancer or proxy resets a TLS session mid-response. The client receives a `200 OK` with an empty or truncated body. If the client only retries on `5xx`, the request hangs until a timeout fires, and the caller has no way to distinguish "succeeded, response lost" from "never processed."

The naive fix — retry on empty `200` — immediately introduces duplicate writes, because the server may have committed the transaction before the response was lost. This is the core argument for idempotency keys: without one, the client cannot safely retry, and with retries disabled, the client cannot recover.

**Diagnosis:** instrument the response body length and the time between request send and response completion. A cluster of short bodies with long completion times points at the proxy or load balancer, not the application.

### 2. Synchronous contracts versus asynchronous mandates

When a regulator requires that failed transactions be reported via callback, the outcome of a request may not be known at the time the HTTP response is written. A synchronous handler has to either block until the outcome is known (holding a connection for an unbounded time) or return a provisional status and reconcile later.

Teams that keep the synchronous shape typically bolt on a saga or orchestration layer with compensating transactions. That works, but the orchestrator becomes a new single point of failure, and every compensating step must itself be idempotent. The honest summary: a synchronous contract plus asynchronous requirements means you build the asynchronous machinery anyway, but you build it in the least convenient place.

### 3. Mutual TLS versus connection pooling

Mutual TLS is a common requirement for end-to-end encryption. It also interacts badly with naive connection pooling, because each new TLS session requires a full handshake. If the pool is too small or the idle timeout is too short, the client pays a handshake cost on most requests.

The relevant defaults are worth knowing. In Python's `urllib3`, `HTTPConnectionPool` defaults to `maxsize=1` and `block=False`, meaning excess connections are discarded rather than queued; the `Retry` object defaults to `total=10` and `backoff_factor=0`, so retries happen with no delay. Those defaults are tuned for low-latency, low-error environments. Raising `maxsize` and setting a non-zero `backoff_factor` is usually the first corrective step, and it costs memory per connection.

**How to measure this:** record the TLS handshake duration separately from the request duration. If handshake time is a meaningful fraction of total latency, the pool is the problem, not the network.

### 4. Callback loss under load

Dead-letter queues exist precisely because callbacks fail. A queue that survives process restarts and retries with exponential backoff converts "lost callback" into "delayed callback," which is a fundamentally different and much cheaper failure. The design question is not whether to queue callbacks but how long to retain them and how to alert when the dead-letter depth grows.

## A durable-by-default mental model

The posture that survives the conditions above can be summarized as: **assume the network drops, and make every operation safely repeatable.**

That yields four design rules:

1. **Every mutating call carries a client-supplied idempotency key.** The server stores the key alongside the result. A retry with the same key returns the stored result rather than re-executing. Keys should be scoped to the operation and the client, and stored with a TTL long enough to cover the client's retry window.
2. **Every response carries a correlation id and an explicit retry hint.** A server-generated correlation id lets you trace a request across services and logs without relying on client-supplied identifiers. If the server is shedding load, say so with `Retry-After` rather than returning a generic error.
3. **Long-running work returns `202 Accepted` plus a status resource.** The client polls a `GET` endpoint (or subscribes to a callback) to learn the outcome. This decouples the client's connection lifetime from the operation's duration.
4. **Callbacks are queued and retried, not fired once.** A durable queue with exponential backoff and a dead-letter destination converts transient failures into delays.

The trade-off is real: this model adds storage for idempotency keys, a queue, and a status resource. It removes the class of bugs where a client cannot tell whether its request took effect.

### A worked example: idempotent payment initiation

Suppose a client initiates a payment. The goal is that retrying the same logical request never produces two payments.

**Request:**

```http
POST /v1/payments HTTP/1.1
Idempotency-Key: 6f1c2b7e-3a4d-4f0e-9c1a-2b8d5e6f7a90
Content-Type: application/json

{ "amount_minor": 150000, "currency": "NGN", "destination": "acct_123" }
```

**Server logic, in order:**

1. Look up the idempotency key in the store. If a record exists and is complete, return the stored response with the original status code.
2. If a record exists but is in progress, return `409 Conflict` with `Retry-After`.
3. If no record exists, insert a record in the `in_progress` state, then execute the operation.
4. On success, update the record to `complete` with the response body and status, then return it.
5. On failure that is safe to retry, update the record to `failed_retryable` and return `503` with `Retry-After`.

The critical detail is step 3: the record must be inserted *before* the side effect, and the insert must be atomic. A common bug is to check-then-insert without a uniqueness constraint, which loses the race under concurrent retries. A unique index on the idempotency key plus an insert that fails on conflict closes that gap.

**Response:**

```http
HTTP/1.1 202 Accepted
Location: /v1/payments/pay_8f3a2c
Retry-After: 2

{ "id": "pay_8f3a2c", "status": "pending" }
```

The client then polls `GET /v1/payments/pay_8f3a2c` until the status is terminal. If the client retries the original `POST` with the same idempotency key, it receives the same `202` and the same payment id — no duplicate.

**Storage sizing, worked from stated assumptions.** Suppose the service handles 500 payment initiations per second at peak, and idempotency records are retained for 24 hours. That is:

```
500 requests/second × 86,400 seconds = 43,200,000 records
```

At roughly 200 bytes per record (key, status, response reference, timestamps), that is about 8.6 GB before index overhead. This is illustrative arithmetic, not a benchmark — substitute your own peak rate and retention window. The point is that idempotency storage is a capacity-planning input, not a free abstraction.

## When the classic model is the right choice

Durable-by-default is not universally better. It adds a queue, a key store, and a status resource, and those have operational costs. The classic synchronous model remains the better choice when:

- Clients are on reliable, low-latency links where a dropped connection is genuinely exceptional.
- Operations are short and the outcome is known before the response is written.
- No regulator requires asynchronous notification or per-call audit records.
- The team does not have the operational capacity to run a durable queue and monitor its depth.

Adding idempotency keys and a status resource to a staff-facing internal tool used over a wired network is usually unjustified complexity. The correct move is to match the contract to the environment, not to adopt the most resilient pattern everywhere.

## A decision checklist

Use the following to decide which posture fits. Treat it as a checklist, not a score.

**Adopt durable-by-default if any of these are true:**

- A regulator requires asynchronous notification of failed transactions.
- A regulator requires per-call audit records that survive client disconnection.
- Measured packet loss on your client population exceeds roughly 5% during peak hours.
- Measured round-trip latency exceeds roughly 300 ms for a meaningful share of clients.
- Duplicate transactions have a direct financial cost.
- Clients are low-end devices that may be killed mid-request.

**Stay with the classic synchronous model if all of these are true:**

- Clients are on reliable links with low latency.
- Operations complete quickly and synchronously.
- No asynchronous notification mandate applies.
- Duplicate writes are cheap or impossible.
- You cannot operate a durable queue reliably.

**Hybrid, which is often correct:**

- Keep synchronous reads.
- Make writes idempotent and return `202` for anything slow.
- Queue callbacks.
- Skip the full saga machinery unless a regulator explicitly requires compensating transactions.

## How to measure whether you need this

Do not adopt this posture on intuition. Instrument the following, then decide:

1. **Client-side latency distribution.** Record p50, p95, and p99 round-trip time from real clients, segmented by network type. A p99 that is an order of magnitude above p50 indicates tail latency that synchronous timeouts will handle badly.
2. **Retry rate and duplicate rate.** Count requests that are retried and, separately, count operations that executed more than once for the same logical action. If duplicates are non-zero, idempotency keys are already overdue.
3. **Callback failure rate.** Log every callback attempt and its outcome. A failure rate above a fraction of a percent justifies a durable queue.
4. **TLS handshake share of latency.** Measure handshake duration as a fraction of total request duration. A large share points at connection pooling, not the network.
5. **Dead-letter depth over time.** If you already have a queue, alert on its depth. A growing dead-letter queue is the earliest signal that your retry policy is wrong.

A simple way to reproduce adverse conditions locally is a network emulator that can inject latency, jitter, and packet loss between your client and server. Run your integration tests through it and assert that retries produce exactly one side effect. That single test catches most of the failure modes described above.

## Common objections

**"This adds too much complexity."** It relocates complexity rather than removing it. The alternative is debugging duplicate payouts and lost callbacks in production, which is more expensive than storing idempotency keys. The question is where you want the complexity to live.

**"Our clients cannot send idempotency keys."** Most mature mobile and server SDKs can attach a header. If a client genuinely cannot, that is a constraint worth documenting explicitly, along with the duplicate risk it creates.

**"The regulations will change."** They will. But idempotency, auditable correlation ids, and asynchronous notification are requirements that tend to persist because they address fraud and reconciliation, not fashion. A design built on unreliable-network constraints is robust to regulatory churn.

**"HTTP/3 will solve it."** A newer transport reduces head-of-line blocking and can improve latency, but it does not make a non-idempotent write safe to retry. Transport improvements and application-level idempotency solve different problems.

## Summary

- The classic REST model assumes a reliable, low-latency client and a synchronous outcome. Those assumptions fail under high packet loss and under regulations that mandate asynchronous notification.
- The recurring failure modes are lost success responses, synchronous contracts against asynchronous mandates, mutual TLS versus connection pooling, and callback loss under load.
- A durable-by-default posture — idempotency keys, correlation ids, `202` plus a status resource, and queued callbacks — addresses all four.
- The cost is real: a key store, a queue, and a status resource. Adopt it when the checklist says so, not by default.
- Measure before you migrate: latency distribution, duplicate rate, callback failure rate, handshake share, and dead-letter depth.

## FAQ

**Where should idempotency records live?**
Any store with a unique constraint on the key and atomic insert. A relational table with a unique index is the simplest correct choice. A key-value store works if it supports conditional writes. The requirement is atomicity, not a specific product.

**How long should idempotency records be retained?**
Long enough to cover the client's maximum retry window, plus a margin. If clients retry for up to an hour, retaining records for 24 hours is comfortable. Retaining forever is a capacity problem; retaining too briefly reintroduces duplicate risk.

**Should reads be idempotent too?**
Reads are naturally idempotent. The rule applies to mutating operations. That said, caching reads aggressively is often the cheapest latency win available, and it is independent of the write-side design.

**How do I handle rate limiting alongside idempotency?**
Rate limit before the idempotency lookup, so a retry of an already-accepted request does not consume new quota. Return `429` with `Retry-After`. A sliding-window counter in a shared store is sufficient for most services.

**What if the queue itself fails?**
That is why the queue must be durable and why dead-letter depth must be alerted on. A queue that loses messages is worse than no queue, because it hides the failure. Persist before acknowledging.

## Do this in the next 30 minutes

Pick your highest-value mutating endpoint and add a unique constraint on a client-supplied idempotency key, with an atomic insert before the side effect. Write one integration test that sends the same request twice through a network emulator with injected packet loss, and assert that exactly one side effect occurred. That test is the smallest possible proof that your write path is safe to retry.
