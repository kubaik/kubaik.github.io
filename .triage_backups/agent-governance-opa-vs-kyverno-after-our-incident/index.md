# Agent Governance: OPA vs Kyverno in Kubernetes

## The governance gap in multi-agent systems

A recurring failure mode in production multi-agent systems is a governance layer that covers the API surface but not the agent runtime. The agents run in their own namespace, talk to their own controllers, and can take actions — exporting data, mutating their own configuration, calling external services — that never pass through the policy engine the team already trusts. The result is a control plane that reports "all policies enforced" while the agent path is unguarded.

The symptom is rarely a single dramatic bug. It is a class of actions that are technically permitted because nothing in the request path evaluates them:

- An agent exports records containing regulated fields without a recorded approval reference.
- An agent rewrites its own ConfigMap or runtime flags, changing its own constraints.
- An agent makes outbound calls with no correlation ID, so the audit trail cannot be reconstructed after the fact.

This article compares the two policy engines most teams reach for when they need to close that gap: Open Policy Agent (OPA) and Kyverno. Both can validate, mutate, and audit. They solve the problem in structurally different ways, and the difference determines where each one fits.

## How OPA works

OPA is a general-purpose policy engine that evaluates policies against JSON input. It runs as a sidecar, a daemon, or a library, receives a JSON document, and returns a decision. Policies are written in Rego, a declarative query language with Datalog-style semantics.

A minimal Rego rule for PII export validation looks like this:

```rego
package agent.governance

violation[msg] {
  input.action == "export_data"
  contains(input.fields, "ssn")
  not input.approval.ref
  msg := sprintf("PII export without approval: %v", [input.request_id])
}
```

The engine returns a JSON decision:

```json
{
  "decision_id": "4b345c...",
  "result": "deny",
  "reason": "PII export without approval"
}
```

OPA is a good fit when you need:

- Logic that spans multiple inputs — agent state, request metadata, and external context in one evaluation.
- Fine-grained traversal of nested JSON structures.
- Enforcement outside Kubernetes: Lambda, ECS, bare-metal services, or edge nodes.

The costs are structural, not incidental. Rego has a steep learning curve. OPA runs as a separate service, so you own its availability, its API endpoint security, and its monitoring. And because the decision is a network round trip from the caller to OPA and back, every enforcement point pays that latency.

## How Kyverno works

Kyverno is a policy engine that runs inside the Kubernetes admission control path. Policies are Kubernetes Custom Resources written in YAML, so there is no separate query language to learn. Kyverno validates and mutates resources as they are created or updated.

A policy requiring an approval label on agent pods:

```yaml
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: agent-pii-approved
spec:
  validationFailureAction: enforce
  rules:
  - name: require-pii-approval-label
    match:
      resources:
      - kind: Pod
        selector:
          matchLabels:
            app: agent
    validate:
      message: "Agent must have pii-approved label"
      pattern:
        metadata:
          labels:
            governance/pii-approved: "true"
```

Kyverno is a good fit when:

- Everything you need to govern is a Kubernetes resource.
- Policies are expressible as label, annotation, and field constraints.
- You want audit output without standing up a separate logging pipeline — violations surface as Kubernetes events.

The limits are equally structural. Kyverno only sees what the API server sees. If your agent state lives in object storage, in a queue message, or in a Lambda event, Kyverno cannot evaluate it. Policies that need external data require an admission webhook that calls out to something else, which reintroduces the network hop and the operational surface you were trying to avoid.

## Comparing the two engines

| Dimension | OPA | Kyverno |
|---|---|---|
| Policy language | Rego (declarative query) | YAML (declarative rules) |
| Enforcement point | Anywhere you can call it with JSON | Kubernetes admission control |
| Input scope | Arbitrary JSON | Kubernetes resources |
| External data | Native (HTTP calls from Rego) | Requires an admission webhook |
| Decision path | Network round trip to the engine | In-process with the API server |
| Audit output | Decision log API, needs a pipeline | Kubernetes events, queryable with kubectl |
| Non-Kubernetes runtimes | Supported | Not supported |

The decision path difference is the one that most often decides the choice. Kyverno evaluates during admission, so there is no extra hop. OPA evaluates wherever you call it, which is more flexible but adds a round trip at every enforcement point.

## How to measure the tradeoff for your workload

Published benchmark numbers are not a substitute for measuring your own policies, because latency depends on policy complexity, input size, and where the engine sits relative to the caller. Here is what to instrument.

**Decision latency.** For Kyverno, admission latency shows up in the API server's admission duration metrics and in the audit events. For OPA, wrap the client call and record the elapsed time around it. Compare the median and the 95th percentile, not just the mean — the tail is what breaks user-facing timeouts.

**Policy evaluation cost as a function of complexity.** Write the same rule in both engines — for example, "deny an export request whose payload contains a regulated field and whose metadata lacks an approval reference." Then add nesting depth and branch count in steps and re-measure. This tells you where each engine's cost curve bends for your actual rule shapes.

**Resource footprint.** Use `kubectl top pod` and `kubectl top node` for steady-state memory and CPU. Run the comparison at the replica count you would actually deploy for availability, not a single instance.

**Audit completeness.** Attempt a policy violation deliberately and confirm it appears in your audit path. For Kyverno, that is `kubectl get events --sort-by=.metadata.creationTimestamp`. For OPA, confirm the decision log reaches your collector and that the decision ID is correlatable with the request ID your agent emitted.

**Failure behavior.** Kill the policy engine under load and observe what the caller does. A fail-open policy engine is a governance gap that only appears during an incident, which is the worst time to discover it.

## A worked example: enforcing one rule end to end

Consider a rule that is common in agent governance: an agent may not export records containing a regulated field unless the request carries an approval reference that resolves to a recorded approval.

**Step 1 — define the decision precisely.** The rule takes three inputs: the action type, the set of fields in the payload, and the approval reference. It returns allow or deny. Writing this down before choosing an engine matters, because the shape of the inputs determines which engine can express the rule directly.

**Step 2 — express it in Kyverno.** If the export is a Kubernetes resource, the rule maps cleanly onto a validate pattern: match the resource kind, require the approval annotation, and constrain the field list. The approval reference is checked for presence, not resolved against an external store. If you need resolution, you add a webhook — and at that point you have an external call in the admission path, with the latency and availability implications that follow.

**Step 3 — express it in OPA.** The same rule is a Rego violation rule. The approval reference can be resolved by an HTTP call from within the policy, or the caller can resolve it and pass the result in the input document. The second option keeps evaluation local and moves the external dependency into the caller, which is usually the better design.

**Step 4 — decide where enforcement lives.** If the export happens through the Kubernetes API, Kyverno enforces it without any new moving parts. If the export happens from a Lambda function, a queue consumer, or a long-running service outside the cluster, Kyverno never sees it, and OPA is the only one of the two that can enforce the rule at all.

The worked example exposes the real decision: it is not "which engine is faster" but "where does the action happen, and what can see it." Latency and ergonomics are tiebreakers among engines that can actually observe the action.

## Failure modes to design against

**Split-brain governance.** API traffic goes through OPA, agent traffic goes through Kyverno, and no one owns the union. A rule added to one engine silently does not apply to the other path. The fix is a single inventory of enforcement points and a test that asserts each rule is present at each point where it is required.

**Fail-open admission.** If the policy engine is unavailable, does the API server admit the resource or reject it? The answer is a configuration choice, and it should be a deliberate one. A fail-open default means an attacker who can degrade the policy engine can bypass it.

**Uncorrelated audit trails.** Both engines can produce a decision record. Neither can correlate it with the agent action unless the agent propagates a request ID into the input. Without that ID, post-incident reconstruction is guesswork.

**Policy drift between environments.** A rule enforced in staging but not production, or vice versa, produces exactly the class of incident this governance layer exists to prevent. Version policies alongside application code and diff them across environments.

**Rules that only check presence.** Requiring an annotation to exist is not the same as requiring it to be valid. A policy that checks for `governance/pii-approved: "true"` passes any resource that sets the label, including one that sets it without authorization. Where the check is meaningful, resolve the reference.

## Decision checklist

Work through these in order. The first question that resolves to a "no" usually determines the answer.

1. **Does every action you need to govern pass through the Kubernetes API server?** If no, Kyverno cannot enforce it, and OPA is the candidate.
2. **Do your rules need data from outside the cluster — an approval store, a risk service, a database?** If yes, either engine needs an external call. OPA can make it natively; Kyverno needs a webhook. Decide which of those you would rather operate.
3. **How complex is your most complex rule?** Enumerate the branches and nesting. If the rules are label and annotation constraints, Kyverno expresses them directly. If they traverse nested payloads and combine multiple inputs, Rego is the more natural fit.
4. **What is your latency budget at the enforcement point?** Measure the round trip for your candidate design, at the 95th percentile, under load. Compare it to the timeout the calling agent already has.
5. **What does your team already operate?** A policy engine you cannot debug at 3 a.m. is worse than a simpler one you can. Rego fluency is a real prerequisite, not a nice-to-have.
6. **What does your auditor require?** If they need a queryable record of every decision with the inputs that produced it, confirm the engine can produce that before you commit.

## Recommendation

For agents that run entirely inside Kubernetes and whose rules are expressible as resource constraints, Kyverno is the lower-overhead choice: no separate service to operate, no network hop in the decision path, and audit output through the Kubernetes event stream.

For agents that run outside Kubernetes, or rules that must evaluate arbitrary JSON or resolve external data, OPA is the engine that can actually see the action. Accept the operational cost of running it and the learning curve of Rego as the price of coverage.

If a system spans both — Kubernetes-resident agents and external ones — the common mistake is to pick one engine and assume it covers everything. It will not. Either run both with a single rule inventory and tests that assert coverage at every enforcement point, or route all agent decisions through one engine that can see all the inputs, even if that means an extra hop for the in-cluster path.

The choice that fails is not OPA and it is not Kyverno. It is assuming the policy engine you already run covers a path it cannot observe.

## Do this in the next 30 minutes

Pick one agent action that currently has no policy in its path — an export, a configuration change, an outbound call — and write down its inputs, its decision rule, and where the action physically executes. Then check whether that execution point passes through your existing policy engine. If it does not, you have found the gap this article is about, and you have the three facts you need to choose between the two engines above.
