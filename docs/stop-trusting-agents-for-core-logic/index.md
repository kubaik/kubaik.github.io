# Stop trusting agents for core logic

## The problem with treating agents as glue

Orchestration frameworks make it easy to wire together many small services or agents and let a central scheduler coordinate them. The pitch is appealing: a single orchestrator spins up a workflow, pushes a config change, and rolls back a failure without touching the underlying services.

The failure mode appears later. Teams end up with opaque dependency graphs, reviewers who cannot see the whole picture, and latency spikes that only surface under load. The root cause is usually the same assumption: that agents can be treated as black-box glue rather than as components that own business logic.

This article covers what that assumption breaks, how to restructure around it, and how to measure whether the restructuring paid off.

## The conventional wisdom, and where it falls short

A common narrative holds that agent-orchestrated architectures are a straightforward path to scaling. Break the monolith into small agents, let a central scheduler coordinate them, and get elasticity. Code review becomes a formality because each agent is supposedly isolated, and architectural decisions move into the orchestrator's config files.

That story leaves out three things.

**Isolation is only as good as the contracts you enforce.** Without a disciplined review process, contracts drift. An agent that "just" reads a config map and writes a multiplier is still coupled to the shape of that config map, the timing of other writers, and the failure semantics of the store.

**The orchestrator becomes a single point of failure** unless it is treated as a first-class service with its own SLO, capacity plan, and on-call rotation. Many teams discover this only when the scheduler's control loop saturates.

**Agents often embed business rules, caching strategies, and retry logic** that belong in the core domain. Once that happens, the orchestrator's YAML becomes the only place where decisions are visible, while the actual behavior lives in dozens of repositories that are rarely reviewed together.

A change in one agent can then silently break another, and review degenerates into a series of disjointed checklists rather than a holistic architectural conversation.

## What actually happens when you follow the standard advice

Three pain points show up reliably.

**Hidden coupling.** A typical failure mode is a missing dependency entry in the orchestrator's manifest. The orchestrator throws a generic error such as `Failed to resolve dependency graph` and the pipeline aborts. Because agents deploy independently, the missing dependency is not caught until runtime. The symptom is a latency bump on the critical path and an error rate that rises under peak traffic.

**Review fatigue.** Reviewers skim dozens of small PRs that each change one line of YAML. Cognitive load is high, and architectural concerns such as data consistency or transaction boundaries get missed. The measurable signal is review turnaround time per PR and the ratio of architectural comments to syntax comments over time.

**Orchestrator overload.** The orchestrator starts handling health checks, circuit breaking, and ad-hoc data transformations. Its CPU usage climbs, and its per-invocation cost becomes non-trivial for a small team. The measurable signals are the orchestrator's CPU and memory saturation, its p99 latency, and the fraction of its code that is not scheduling or policy.

None of these are inherent to orchestration. They are consequences of treating agents as black boxes.

## A different mental model: agents as micro-domains

Treat agents as micro-domains that own a slice of business logic. Treat the orchestrator as a policy engine rather than a logic engine. Code review then focuses on domain contracts and policy correctness, not YAML syntax.

Three shifts make this work.

**Contract-first design.** Define protobuf or OpenAPI contracts in a shared repository and version them with semantic tags. Every agent implements the contract, and reviewers verify that the implementation matches the spec. A contract mismatch becomes a compile-time or CI-time failure rather than a runtime surprise.

**Policy as code.** Express orchestration rules in a policy language such as Rego (Open Policy Agent) or Cedar, or in a typed configuration schema if you prefer. Policy changes go through the same PR pipeline as service code, so reviewers see the impact on the whole system.

**Ownership boundaries.** Assign a small cross-functional team ownership of each micro-domain. That team reviews the agent code and the policy that wires it to other domains. Ownership is what makes the contract real; without it, contracts decay.

The orchestrator stays lightweight, handling scheduling and policy enforcement, while the heavy lifting stays inside well-defined services.

## Worked example: a race-free config update

Consider a pricing feature that must not touch the core pricing service. A team adds a surge agent that reads traffic data and writes a multiplier to a shared config store. The orchestrator's manifest gains a dependency on the traffic collector, and the agent deploys via the same chart pipeline as everything else.

Under a load test, the price-lookup endpoint shows elevated latency. The root cause is a race: the surge agent updates the config without a distributed lock, producing intermittent config update conflicts. The orchestrator logs `Error: failed to apply config` but continues scheduling, which masks the problem.

The fix is a distributed lock around the read-modify-write:

```python
import redis
import json
import uuid

r = redis.Redis(host='redis-prod', port=6379, db=0)
lock = r.lock('surge_config_lock', timeout=5)

def update_surge(multiplier: float):
    acquired = lock.acquire(blocking=True, blocking_timeout=5)
    if not acquired:
        raise TimeoutError("could not acquire surge_config_lock")
    try:
        cfg = json.loads(r.get('surge_cfg') or '{}')
        cfg['multiplier'] = multiplier
        cfg['version'] = str(uuid.uuid4())
        r.set('surge_cfg', json.dumps(cfg))
    finally:
        lock.release()
```

Note the added `blocking_timeout` and the explicit failure path. A reviewer looking at only the YAML diff would never see this bug. A reviewer looking at the contract — "the config store must be updated atomically and versioned" — would flag it immediately.

The corresponding policy, expressed in Rego, keeps the authorization rule next to the agent:

```rego
package orchestrator.policy

allow {
    input.agent == "surge-agent"
    input.action == "update"
    input.user.role == "platform-admin"
}
```

Embedding the policy in the same repository as the agent forces a joint review. A change to the agent that requires a broader permission cannot merge without a matching policy change, and vice versa.

## How to measure whether this helps

The claims above are structural, not numeric. To decide for your system, instrument the following and compare before and after.

**Contract mismatch rate.** Emit a counter such as `contract_mismatch_total` at every contract boundary. Alert when the rate exceeds a threshold you choose. Compare the count of mismatches caught in CI versus caught in production.

**Review turnaround.** Track time from PR open to merge, split by PR type (contract change, policy change, agent implementation, YAML wiring). The interesting number is the ratio of architectural comments to syntax comments per PR.

**Orchestrator saturation.** Record CPU, memory, and p99 latency for the orchestrator process. If the orchestrator is doing work that is not scheduling or policy evaluation, that work should move into a domain service.

**Critical-path latency and error rate.** Measure p50, p95, and p99 latency for the endpoints that cross domain boundaries, plus the error rate during peak. Compare against a baseline captured before the change.

**Cost per million orchestration invocations.** If your orchestrator runs on a serverless platform, pull the invocation count and billed duration from the provider's metrics and divide. If it runs on a VM or pod, divide the monthly cost by the number of orchestration events. Either way, the number is arithmetic from your own bill and your own counters, not a vendor benchmark.

Run each measurement for at least one full traffic cycle before drawing conclusions. A latency improvement that only appears during off-peak hours is not an improvement.

## When the glue-first model is the right choice

The glue-first model is not always wrong. It is appropriate when the following hold.

- The team is small enough that everyone can hold the whole system in their head.
- The workload is batch-oriented, with no request-time interaction between agents.
- Agents are stateless, idempotent, and do not share mutable state.
- There is no strict versioned contract requirement from external consumers.

For a nightly data pipeline that runs on a managed batch service, a YAML-driven orchestrator can reduce time-to-market with little downside. The hidden-coupling risk is negligible when agents never interact at request time.

The model starts to crumble the moment you introduce shared caches, cross-domain transactions, or real-time user flows. Those are the conditions under which a missing lock, a drifted schema, or an unversioned config change becomes a production incident.

## Decision checklist

Work through these questions before choosing an approach.

| Question | If yes | If no |
|---|---|---|
| Do agents share mutable state at request time? | Contract-first | Glue-first may suffice |
| Are there cross-domain transactions? | Contract-first | Glue-first may suffice |
| Do external consumers depend on versioned schemas? | Contract-first | Glue-first may suffice |
| Is latency on a cross-domain path a competitive differentiator? | Contract-first | Glue-first may suffice |
| Can one engineer hold the full dependency graph in their head? | Glue-first is defensible | Contract-first |

If two or more rows point to contract-first, adopt it. Otherwise, a lightweight glue approach can be justified — but write down the conditions under which you would revisit the decision, and set a reminder to check them.

## Common objections

**"Contracts add bureaucracy and slow us down."** Contracts replace ad-hoc communication. The measurable version of this claim is the number of messages or tickets about data shape per week before and after introducing a contract. Track it; if it does not fall, the contract is not doing its job.

**"Policy languages are another thing to learn."** A declarative policy language is a small surface area compared to the systems it governs. It can be formatted and unit-tested with the tooling that ships with it (`opa fmt`, `opa test` for Rego). The payoff is a single source of truth for orchestration rules that can be tested in CI.

**"We cannot afford another service."** A contract repository is a git repository. If you already have a version control system and a CI runner, the marginal cost is the CI minutes spent validating contracts. Compare that to the cost of one production incident caused by a schema drift, which you can estimate from your own incident history.

**"Adding policy checks will break our pipeline."** Policy checks are another step in the pipeline. Measure the added wall-clock time on a representative PR. If it is more than a few seconds, cache the policy bundle or run the check only on paths that touch policy files.

**"Our agents are already isolated."** Isolation is a property of the contracts, not the deployment topology. Verify it by attempting to change one agent's output shape without updating its consumers and confirming that CI fails.

## A starting sequence for a new system

If you are building from scratch, the following order avoids most of the pain described above.

1. **Write the contracts first.** A shared OpenAPI or protobuf repository with semantic version tags, before any agent code exists.
2. **Keep agents as pure functions where possible.** Push retries, caching, and circuit breaking into a shared library or into the caller, not into the agent.
3. **Define orchestration rules as policy.** Keep them in the same repository as the agents they govern, and run policy tests in CI.
4. **Define infrastructure as code.** Keep the orchestrator definition in version-controlled modules alongside the contracts, so a change to the orchestrator is reviewed with the same rigor as a change to a service.
5. **Instrument every contract boundary.** Export a mismatch counter and set an alert threshold. Without this, you will not know whether the contracts are holding.
6. **Set an orchestrator budget.** Define the maximum CPU and memory the orchestrator may consume, and treat any work beyond scheduling and policy evaluation as a candidate for extraction.

These steps create a feedback loop in which reviewers see the full impact of a change and the orchestrator stays lean.

## Summary

Treating agents as black-box glue hides coupling in YAML, makes review shallow, and lets the orchestrator absorb responsibilities it was never designed to hold. Treating agents as micro-domains — with contract-first design, policy as code, and clear ownership — restores architectural depth to the review process and keeps the orchestrator focused on scheduling and policy.

The trade-off is a modest increase in upfront scaffolding. Whether it pays off is an empirical question, and the measurements listed above are how you answer it for your own system rather than borrowing someone else's numbers.

**Next step:** open your orchestrator's manifest, pick one agent, and write down the contract it implicitly depends on — the shape of the data it reads, the timing assumptions it makes, and the failure semantics it expects from its dependencies. If you cannot write it down in a paragraph, that is the first contract to make explicit.
