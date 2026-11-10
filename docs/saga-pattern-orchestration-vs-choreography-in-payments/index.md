# Saga pattern: orchestration vs choreography in payments

Payment flows are distributed transactions by nature. A card is charged, inventory is reserved, an order record is written, and a warehouse is notified — each step in a different service, each with its own database. The Saga pattern is the standard answer: break the transaction into a sequence of local transactions, each paired with a compensating action that undoes it if a later step fails. The confusion starts when teams pick *how* to coordinate those steps. Orchestration (a central coordinator tells each service what to do) and choreography (services react to each other's events) are both valid, but they fail in different ways under the specific constraints of money movement — idempotency, reversal timing, and the fact that a refund is not the inverse of a charge.

This article walks through what each style does at the message level, where each breaks in a payment context, and how to choose based on the number of participants, the reversibility of each step, and the team's tolerance for implicit flow logic. It ends with a concrete audit that can be completed in about thirty minutes.

## The real question: who owns the state of an unfinished payment?

The orchestration-versus-choreography debate is usually framed as centralization versus decoupling. That framing is too abstract for payments. The concrete question is: when a payment is halfway done — card authorized, inventory reserved, order not yet confirmed — which component knows that this payment exists in a partial state, and how does it find out when something goes wrong?

In orchestration, the answer is a single saga orchestrator. It holds a state machine for each payment attempt. It calls the payment service, waits for a response, then calls inventory, then calls order. If inventory fails, the orchestrator calls the payment service's void or refund endpoint. The orchestrator is the source of truth for in-flight payments.

In choreography, there is no such component. The payment service publishes `PaymentAuthorized`. The inventory service consumes it, reserves stock, and publishes `InventoryReserved`. The order service consumes that and creates the order. If inventory reservation fails, inventory publishes `InventoryReservationFailed`, and the payment service — which must be listening — issues a void. The state of the partial payment is distributed across the event log and the local state of each service. No single component can answer "what is the status of payment X?" without querying multiple services or replaying events.

That difference is the whole decision. Everything else — coupling, latency, operational complexity — follows from it.

## The underlying model: local transactions plus compensations

Both styles implement the same theoretical model. A saga is a sequence of local transactions T1, T2, ... Tn. Each Ti has a compensating transaction Ci that semantically undoes it. If Tk fails, the saga executes Ck-1, Ck-2, ... C1 in reverse order. The model assumes compensations are always possible and always succeed. In payments, both assumptions are shaky.

A card authorization can be voided only before capture, and only within the authorization window the processor allows. A captured charge cannot be voided; it must be refunded, which is a new transaction with its own settlement timeline. A refund can fail if the card is closed. A refund can succeed at the processor but fail to record in the ledger if the service crashes between the API call and the database write. These are not edge cases; they are the normal failure surface of money movement.

So the model forces a design decision before choosing orchestration or choreography: which steps are *reversible* (void an auth), which are *compensable* (refund a capture), and which are *irreversible* (send a payout to a third-party bank account). Irreversible steps must be placed at the end of the saga, after all reversible steps have succeeded, or they must be guarded by a separate confirmation mechanism. This ordering constraint is independent of coordination style, but choreography makes it harder to enforce because no single component sees the whole sequence.

A useful analogy: orchestration is a conductor with a score; choreography is a jazz combo listening to each other. Jazz works when the players know the tune and the changes are simple. Payment flows have too many key changes — partial captures, multi-currency, chargebacks arriving days later — for most teams to keep the whole arrangement in their heads.

## What each style looks like in code

A minimal orchestration loop in TypeScript, using a state machine persisted per payment:

```typescript
// Orchestrator: one instance per payment attempt, state persisted in Postgres.
type Step = 'authorize' | 'reserve' | 'capture' | 'confirm';
type SagaState = { paymentId: string; step: Step; completed: Step[] };

async function runSaga(state: SagaState): Promise<SagaState> {
  try {
    if (state.step === 'authorize') {
      await payments.authorize(state.paymentId);
      return { ...state, step: 'reserve', completed: [...state.completed, 'authorize'] };
    }
    if (state.step === 'reserve') {
      await inventory.reserve(state.paymentId);
      return { ...state, step: 'capture', completed: [...state.completed, 'reserve'] };
    }
    if (state.step === 'capture') {
      await payments.capture(state.paymentId);
      return { ...state, step: 'confirm', completed: [...state.completed, 'capture'] };
    }
    if (state.step === 'confirm') {
      await orders.confirm(state.paymentId);
      return { ...state, step: 'confirm', completed: [...state.completed, 'confirm'] };
    }
    return state;
  } catch (err) {
    return await compensate(state);
  }
}

async function compensate(state: SagaState): Promise<SagaState> {
  for (const step of [...state.completed].reverse()) {
    if (step === 'capture') await payments.refund(state.paymentId);
    if (step === 'reserve') await inventory.release(state.paymentId);
    if (step === 'authorize') await payments.void(state.paymentId);
  }
  return { ...state, step: 'authorize', completed: [] };
}
```

This is deliberately naive: it does not handle crashes between the API call and the state write, retries, or idempotency keys. Those are the hard parts, and they are the same in both styles. The point here is structural: the orchestrator can inspect `state.completed` and know exactly what to compensate. That list is the source of truth.

A choreographed version has no such list. The payment service must listen for `InventoryReservationFailed` and decide whether to void. The inventory service must listen for `PaymentVoided` and release the reservation if it already reserved. Each service encodes a fragment of the flow. Adding a step — say, a fraud check between authorize and reserve — means changing the event subscriptions of at least two services and reasoning about the new interleavings. In orchestration, it means adding one branch to the orchestrator.

## Where choreography breaks in payment flows

The failure modes are specific and predictable.

**Lost compensation.** The inventory service reserves stock, then crashes before publishing `InventoryReserved`. The payment service, waiting for that event to proceed, times out and voids the authorization. But inventory has already reserved. The reservation leaks until a reconciliation job finds it. In orchestration, the orchestrator would have called inventory and received a timeout; it would then retry or compensate explicitly, and the inventory service's idempotency key would prevent double reservation. The orchestrator knows the call was attempted; in choreography, the event is the only record, and it was never written.

**Out-of-order events.** Event buses do not guarantee global ordering. `PaymentVoided` can arrive at inventory before `InventoryReserved` if they are published to different partitions or topics. Inventory must handle a void for a reservation it has not yet made, which means it must buffer or reject, and the payment service must handle the resulting inconsistency. This is solvable with per-payment partition keys and careful consumer design, but it is work that orchestration does not require.

**The refund-is-not-a-void problem.** A compensation in choreography is just another event. If `PaymentCaptured` has already been published and consumed by the ledger service, a later `PaymentRefunded` event must be consumed by the same services in the right order. If the ledger service processes the refund before the capture (possible under at-least-once delivery with redelivery), the ledger shows a negative balance for a payment it never recorded as captured. Orchestration serializes these calls through one component, which can enforce ordering by awaiting each step.

**Observability.** When a customer asks "where is my refund?", choreography requires correlating events across topics. Orchestration requires querying one table. Most teams underestimate how much operational value that single table provides until their first payment incident.

**Partial visibility during incidents.** During an incident, an operator needs to answer "which payments are stuck and at what step?" In a choreographed system this requires a correlation ID threaded through every event and a query across every topic and consumer group. In an orchestrated system the same question is a single indexed query against the saga state table. The difference is not cosmetic: it determines whether incident triage takes minutes or hours.

## Where orchestration hurts

The orchestrator is a single point of failure and a coupling magnet. If it goes down, in-flight payments stall. If it holds too much business logic, it becomes a distributed monolith: every new payment method or regional rule requires a change to the orchestrator, and every change risks the whole flow. Teams that centralize too aggressively end up with an orchestrator that knows about card networks, fraud vendors, and tax rules — a component that is hard to test and harder to deploy.

The mitigation is to keep the orchestrator thin: it sequences calls and persists state, but the business rules live in the services it calls. The orchestrator should not know what a capture is; it should know that step three is called `capture` and that its compensation is `refund`. That separation is what makes orchestration maintainable.

A second cost is synchronous coupling. If the orchestrator calls services over HTTP, its availability is the product of all downstream availabilities. A common pattern is to make the orchestrator durable — persist state after each step, resume on restart — and to use asynchronous messaging between the orchestrator and services rather than direct HTTP. That combination gives the state visibility of orchestration with the resilience of messaging. It is more infrastructure than a simple loop, but less than debugging lost compensations in a choreographed system.

A third cost is version skew during deploys. If the orchestrator is being redeployed while sagas are in flight, those sagas must resume on the new code without misinterpreting persisted state. The practical rule is to make the persisted state a versioned, append-only record of completed steps rather than an enum that changes meaning between releases. A saga that was mid-`capture` when the deploy started must resume with the same definition of what `capture` means.

## A decision framework for payment sagas

Use this as a starting point, not a law. The thresholds are illustrative, based on the structural properties above, not on measured data.

| Factor | Favors orchestration | Favors choreography |
|---|---|---|
| Number of steps in the saga | 3 or more | 2, or simple fan-out |
| Steps include irreversible actions (payouts) | Yes — need explicit ordering | Rarely suitable |
| Compensation logic is non-trivial (refund vs void) | Yes | No |
| Team size and ownership boundaries | One team owns the flow | Many teams, each owns a service |
| Need for a single status query | Yes | No |
| Event infrastructure already mature | Not required | Required |
| Frequency of flow changes | High — orchestrator centralizes change | Low — changes ripple across services |

The strongest signal is the presence of irreversible steps. If the saga ends with a payout or a capture, it needs a component that can guarantee the preceding steps succeeded before it fires. Choreography can approximate this with a "confirm" event that every service must acknowledge, but at that point an orchestrator has been rebuilt with more moving parts.

The second strongest signal is the compensation semantics. If compensations are simple (delete a row, release a lock), choreography is tractable. If they involve calling a payment processor's refund API, handling partial refunds, and reconciling settlement reports days later, the logic is too stateful to distribute safely.

Choreography is a good fit for the *post-saga* parts of a payment system: emitting `PaymentCaptured` so that analytics, notifications, and loyalty services can react independently. Those consumers do not participate in the transaction; they observe it. That is the right use of events. Using events to coordinate the transaction itself is where teams get into trouble.

## Common misreadings

**"Choreography is more scalable."** Scalability is rarely the bottleneck in payment flows. Payment volume is bounded by customer behavior, not by the coordination mechanism. The orchestrator can be horizontally scaled by sharding on payment ID. The real cost of choreography is operational, not computational.

**"Orchestration means a single point of failure."** Only if the orchestrator is stateless and synchronous. A durable orchestrator that persists state and resumes after restart is not a single point of failure in the availability sense; it is a single point of *coordination*, which is exactly what a transaction needs. The failure mode to design for is the orchestrator crashing mid-saga, which durable state solves.

**"A team can start with choreography and move to orchestration later."** Migration is possible but expensive: an orchestrator must be introduced alongside the existing event flows, state dual-written, and compensations verified to still fire correctly. Teams that anticipate needing orchestration should start there. The reverse migration — orchestration to choreography — is rarer and usually driven by organizational boundaries, not technical need.

**"The saga pattern replaces distributed transactions."** It replaces *atomic* distributed transactions with *eventually consistent* ones. That means the system must tolerate intermediate states: an authorized-but-not-captured payment, a reserved-but-not-confirmed order. If the product cannot show a "pending" state to users or reconcile it in reporting, no coordination style will save it.

## A worked failure trace

Consider a four-step saga — authorize, reserve, capture, confirm — where the capture call succeeds at the processor but the orchestrator crashes before persisting the new state. On restart, the orchestrator reads `step: 'capture'` and calls capture again. Without an idempotency key, the customer is charged twice. With one, the processor returns the original result and the saga proceeds. The same crash in a choreographed design means the `PaymentCaptured` event was never published; inventory holds a reservation that nothing will ever release, and the payment service has no record that capture was attempted. The orchestrator's persisted state is what makes the retry safe; the idempotency key is what makes it correct. Neither is optional, and both are easier to reason about when one component owns the sequence.

## How to measure whether the choice is working

Rather than relying on intuition, instrument the flow and compare over a defined window, for example two weeks before and two weeks after a change:

- **Saga completion rate**: completed sagas divided by started sagas, per flow. Query the saga state table (or reconstruct from events) grouped by final state.
- **Stuck-saga age**: the maximum and 95th-percentile age of sagas not in a terminal state. A rising p95 indicates compensation or retry logic is failing.
- **Compensation rate**: compensations executed divided by sagas started. A sudden rise usually points to a downstream dependency, not the coordination style.
- **Reconciliation exceptions**: rows a daily reconciliation job cannot match between the ledger and the processor. Compare counts and dollar totals per day.
- **Time to answer "where is payment X?"**: sample ten real support tickets and time how long an engineer needs to produce a definitive answer. This is the most direct measure of the observability difference between the two styles.

Run the same queries against both designs if a migration is under consideration; the comparison, not the architecture diagram, is the evidence.

## Frequently Asked Questions

**How do I choose between orchestration and choreography for a payment saga?**
Start with the number of steps and the reversibility of each. With three or more steps, any irreversible step, or compensation logic that calls external APIs like refunds, use orchestration with a durable state machine. With two steps and simple compensations, choreography can work. The presence of an irreversible step is the clearest signal: the flow needs a component that can order the sequence and confirm prior success.

**Can choreography work for payments with a reliable event bus?**
A reliable bus with at-least-once delivery and per-payment ordering helps, but it does not solve the lost-compensation problem: if a service crashes before publishing its event, the bus has nothing to deliver. An outbox pattern or a coordinator that knows the call was attempted is still required. Choreography shifts the coordination burden into each service's consumer logic, which is harder to test and observe than a central state machine.

**What is the difference between a saga compensation and a rollback?**
A rollback is an atomic undo within a single database transaction. A saga compensation is a new local transaction that semantically reverses a prior step, often by calling an external system. A refund is not the inverse of a charge: it has its own settlement timeline, can fail independently, and may be partial. Saga design in payments must therefore treat compensations as first-class operations with their own failure handling, not as automatic undos.

**Do I need a workflow engine to implement orchestration?**
No. A durable state machine persisted in the existing database, with a worker that resumes sagas after crashes, is sufficient for most payment flows. Workflow engines add visibility and retry semantics out of the box, but they also add operational surface. Start with a table of saga state and a worker loop; move to an engine if scheduling, timers, and history are being rebuilt from scratch.

## Next step: audit one payment flow's compensations

Open the service that handles the most common payment path and list every step between "customer clicks pay" and "order confirmed." For each step, write down its compensation and classify it as reversible (void), compensable (refund), or irreversible (payout). If any irreversible step is not last, or if any compensation calls an external API without an idempotency key, that is the first thing to fix — regardless of whether orchestration or choreography is in use. That list is also the input to the orchestration decision: if it has more than two entries, a durable orchestrator is the safer default.
