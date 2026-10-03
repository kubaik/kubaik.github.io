# LLM risks underestimated in OWASP Top 10

## Why the checklist feels complete but isn't

The OWASP LLM Top 10 is a taxonomy of failure categories, not a control set. It names prompt injection, insecure output handling, and excessive agency, but it does not tell you what to instrument, what threshold to alert on, or what the mitigation costs in latency and complexity. Teams that treat the list as a build order tend to ship the input filter, the output filter, and the rate limiter, then declare the system safe. The failures that follow rarely look like the categories on the list.

The deeper problem is a mental-model gap. A prompt is not just text; it is an instruction stream that can invoke tools, write files, and mutate state outside the model. When a checklist treats the LLM as a passive text box, it under-weights everything that happens after the model decides to act. Sanitizing inputs controls what goes in. It says almost nothing about what the system is permitted to do on the way out.

This article covers the second-order risks: cascading tool failures, context pollution, latency budgets consumed by validators, and supply-chain exposure through tool dependencies. For each, it gives a way to measure the risk in your own system rather than a number to trust from someone else's.

## Risk 1: Cascading tool failures with silent fallbacks

### The failure mode

An agent calls tool A, uses the result to decide whether to call tool B, and tool A fails. If the orchestration layer does not distinguish "tool returned an error" from "tool returned a value," the model may proceed with a default, a stale value, or a hallucinated one. The user sees a confident answer built on a failed dependency.

A representative pattern: a model fetches a temperature reading, then decides whether to activate heating. The weather API returns HTTP 500. The model's context contains no error signal, so it substitutes a plausible default and calls the heating tool anyway. Nothing in the input or output filters fires, because no injection occurred and the output is well-formed. The defect is in the control flow between tools.

### How to measure it

Instrument every tool invocation with three fields: tool name, outcome (success, error, timeout), and whether the downstream step consumed the result. Then compute, over a fixed window:

- **Fallback rate** = tool calls that returned an error but whose result was still consumed downstream, divided by total tool calls.
- **Blast radius** = number of downstream actions taken after a failed upstream call.

A useful alerting rule is any non-zero blast radius, because a consumed failure is almost always a bug rather than a design choice. Log the raw tool response alongside the model's next action so you can reconstruct the decision.

### Mitigation

Make tool errors first-class in the context. Return a structured error object the model is instructed to surface, and gate dependent tools on an explicit success flag rather than on the presence of a value. Where the model must proceed, require it to state the assumption it is making so the fallback is visible in the output.

## Risk 2: Context pollution by generated artifacts

### The failure mode

When a system stores model output and later feeds it back as input, the context window accumulates machine-generated text. That text can contain internal endpoints, identifiers, or credentials that were valid in the session that produced them. Over successive turns, the model can reproduce fragments of earlier generated artifacts in unrelated responses.

This is a data-flow problem, not a model problem. The model has no notion of which tokens in its context are sensitive; it treats all context as available to condition on. If your pipeline round-trips generated code, logs, or documents through the same context, you have effectively built a slow leak.

### How to measure it

Tag every message in the context with its provenance: user, system, tool result, or model-generated. Then sample stored prompts and count the fraction containing substrings that match your secret patterns (API key prefixes, internal hostnames, connection strings). A simple regex sweep over a rolling sample is enough to get a rate. Track it over time; a rising rate means generated content is being recycled.

### Mitigation

Separate the write path from the read path. Generated artifacts should be stored in a system of record and referenced by identifier, not pasted into future contexts. Apply redaction at the boundary where content enters the context, not only where it leaves. Treat any context assembled from mixed provenance as untrusted input to the model.

## Risk 3: Validation latency and the error budget

### The failure mode

"Validate outputs" is easy to say and expensive to run. A second model call, a schema check, or a classifier adds fixed latency to every request. That cost is invisible until a cold start or a traffic spike, when the validator becomes the slowest component and the error budget drains faster than the feature budget.

### How to measure it

Instrument the validator as its own span in your tracing. Record p50, p95, and p99 for the validator separately from the model call. Then compute the fraction of your latency SLO consumed by validation:

```
validation_share = p95(validator) / p95(total_request)
```

If validation accounts for a large share of the tail, it is a candidate for caching, batching, or moving off the critical path. Also measure cold-start frequency for the validator's backing model; a validator that reloads on scale-out will spike exactly when you need it most.

### Mitigation

Cache validator decisions keyed on the content being validated when the decision is deterministic. For non-deterministic checks, consider running them asynchronously and surfacing a provisional response, or batching them across requests. Provision capacity for the validator independently of the primary model so a cold start in one does not stall the other.

## Risk 4: Input sanitizers that don't understand prompts

### The failure mode

HTML sanitizers like DOMPurify are built to neutralize markup, not to constrain model instructions. A payload that is inert as HTML can be active as a prompt. Structured control tokens, XML-like tags, and JSON fragments can pass through a sanitizer unchanged and still steer the model.

The same applies to naive regex filters. A filter that blocks a literal tool name is defeated by casing, whitespace, or synonym substitution. The filter gives a false sense of coverage because it catches the examples in the test suite.

### How to measure it

Build a corpus of adversarial inputs from your own logs and from public prompt-injection collections. Run each through your sanitizer and then through the model, and record whether the model's behavior changed. The metric that matters is behavioral: did the output or tool call deviate from the safe baseline. A sanitizer that passes adversarial inputs unchanged is not a control.

### Mitigation

Do not rely on a single sanitizer layer. Combine structural constraints (the model can only emit a validated schema), capability limits (the tools it can call), and behavioral monitoring (detecting deviations from expected tool-call patterns). Treat sanitization as defense in depth, not as a gate.

## Risk 5: Supply-chain exposure through tool dependencies

### The failure mode

An agent is allowed to call a restricted set of internal tools. Those tools import third-party packages. A malicious or compromised dependency runs with the tool's privileges, which may include access to the model's context, credentials, or the host filesystem. The agent's capability boundary is only as strong as the weakest dependency behind it.

This risk is easy to miss because the tool interface looks narrow. The interface is not the attack surface; the transitive dependency graph is.

### How to measure it

For each tool the agent can call, enumerate its dependency tree and its transitive dependencies. Flag any package that is unmaintained, recently changed maintainer, or not pinned to a hash. Record which tools run with elevated privileges or network access. The output is a per-tool risk score you can compare across the fleet.

### Mitigation

Pin dependencies to hashes and verify them at install time. Run tools in sandboxes with the minimum privileges and network access they need. For tools that must import code at runtime, restrict the import path to an allowlist. Review the dependency graph on the same cadence as the tool code itself.

## Instrumenting the second-order risks

The categories above share a common measurement pattern: instrument the boundary, log the decision, and alert on deviations from the safe baseline. A minimal instrumentation set:

- **Tool calls:** name, outcome, whether the result was consumed, downstream actions taken.
- **Context:** provenance tag per message, secret-pattern hits per assembled context.
- **Validation:** latency span, cache hit rate, cold-start count.
- **Sanitization:** adversarial corpus pass rate, behavioral deviation rate.
- **Dependencies:** per-tool dependency count, unpinned count, privilege level.

None of these require a new platform. They require deciding what "normal" looks like and writing it down, so an alert has something to compare against.

## A decision checklist

Before shipping an LLM feature that can call tools or mutate state, confirm:

1. Every tool call is logged with an outcome, and a failed call cannot be silently consumed downstream.
2. Generated content is stored by reference, not pasted into future contexts.
3. Validator latency is measured separately and its share of the tail is known.
4. Sanitization is tested behaviorally against adversarial inputs, not just against known patterns.
5. Every tool's dependency graph is enumerated, pinned, and sandboxed.
6. There is a defined safe baseline for tool-call patterns, and deviations alert.

If any of these is missing, the feature is not production-ready regardless of how many OWASP categories it addresses.

## The takeaway

The OWASP LLM Top 10 is a useful map of failure categories. It is not a control set, and treating it as one leaves the second-order risks unmeasured. The work that matters is instrumentation: knowing what your tools do when they fail, what your context carries, what your validators cost, and what your dependencies expose. That work starts after the checklist is done.

## Do this in the next 30 minutes

Pick one tool your agent can call and answer three questions in writing: what happens when it returns an error, what its transitive dependencies are, and what privileges it runs with. If you cannot answer all three from your current logs and manifests, that tool is your first instrumentation target.
