# Track AI agent costs in fractal workflows

## Why flat cost tracking fails on nested agent workflows

Most agentic FinOps guidance stops where the difficulty begins. The standard advice—attach a cost agent to every pod, tag every resource, run weekly reports, alert on spend spikes—was designed for static services: an application server, a database cluster, a background worker queue. Those workloads have predictable, additive, traceable costs. An agentic workflow does not.

A typical agentic request spawns sub-agents, loops, calls external APIs billed by token or duration, retries on failure, and waits on human approval. The cost of one top-level prompt is not a fixed rate per request; it is the sum of a tree of nested executions, each with its own billing entity, region, and lifecycle. Tagging only the top-level pod captures the root of that tree and nothing else.

A common failure mode illustrates the gap. A deploy adds a safety layer whose policy conflicts with the existing retry policy. The agent begins retrying every refund request indefinitely. The orchestrator scales out, each replica bills by the minute, and the cost report shows a single spike labeled with the top-level service name. The actual spend is distributed across hundreds of small line items under the model provider, the vector store, and the cache—none of which carry the top-level service tag. By the time anyone correlates the two, the incident has been running for hours.

The standard playbook assumes costs are additive and traceable. Agentic systems break both assumptions: costs are multiplicative across nesting depth, and the billing entity is frequently not the entity that triggered the spend.

## A worked example of cost moving out of view

Consider a customer-service agent running as a serverless function (Node 20 LTS) behind a managed LLM endpoint, with a tagging convention of `team:cx`, `project:agent`, `env:prod` and a daily spend alarm.

In week one, spend stays inside budget. Then the team upgrades to a model with stronger reasoning. Same prompt, same session shape. Two things change: function duration rises because the model emits longer reasoning chains, and the agent now calls the internal vector search three times per session instead of once. The function-level metric shows duration climbing. The FinOps dashboard still shows the top-level service at its old per-request figure, because the cost driver moved to a different service in a different account.

The agentic workflow invoked a sub-agent that provisioned a cache cluster in a second region. The cluster ran for seventeen hours because the idle timeout was set to 3600 seconds and the agent never explicitly closed the connection. The tag propagated to the function, not to the cache cluster, so the spend surfaced under the agent's name even though the real driver was the cache tier.

This pattern—a cost spike attributed to the wrong entity, followed by an internal argument about whether the cause was a model change or a misconfigured cache—is common enough to be a category. The dashboard showed the agent's cost up by a double-digit percentage; the actual driver was cross-region bandwidth on a cluster nobody had tagged.

## A dependency-graph mental model

Treat agentic FinOps as a dependency-graph problem rather than a resource-tracking problem.

Every agentic workflow is a tree or DAG of tasks. Each task has three attributes:

- **Cost driver** — tokens, wall-clock duration, bandwidth, memory, or a fixed per-call fee.
- **Billing entity** — the account, region, and service that actually issues the charge.
- **Lifecycle** — start, idle, retry, cancel, timeout.

The total cost of a top-level prompt is the sum of the cost of every leaf task plus orchestration overhead. Overhead is not negligible and includes:

- **Orchestration latency** — time the orchestrator spends waiting on sub-agent responses while its own runtime bills.
- **Retry storms** — a failing sub-agent triggering a cascade of re-executions.
- **Idle time** — a sub-agent holding resources while blocked on human approval or a slow API.
- **Cross-region transfer** — data moving between a vector store in one region and an agent runtime in another.

This model forces a specific question: what does "cost" mean when a single prompt triggers dozens of sub-agents, each billing by token count, duration, and bandwidth? Tagging the top-level pod is insufficient. Each sub-agent's lifecycle must be traced and mapped to the billing entity that issues the charge.

In practice that means:

- Instrument every agent spawn, not just the top-level function.
- Capture billing labels from every downstream service.
- Treat idle time as a cost driver, not as free waiting.
- Include orchestration latency in the cost equation.

## Instrumenting spawns and propagating billing labels

The core primitive is a context manager wrapped around every agent spawn that records spawn time, parent task ID, downstream services invoked, billing labels, and lifecycle events. Billing labels propagate through OpenTelemetry baggage so downstream services inherit them.

```python
from opentelemetry import baggage, context

def agent_spawn(billing_labels):
    ctx = baggage.set_baggage("billing.labels", ",".join(billing_labels))
    token = context.attach(ctx)
    try:
        # spawn agent
        ...
    finally:
        context.detach(token)
```

Labels should identify the billing entity, not the logical service. `anthropic:claude-3.7-sonnet`, `region:eu-central-1`, `service:vector-search` are useful; `team:cx` alone is not, because it does not map to an invoice line.

With this data, a single prompt's cost can be reconstructed as a sum over its task tree. Using illustrative per-task figures to show the shape of the arithmetic:

- Top-level agent: 0.042 USD
- Vector search sub-agent: 0.118 USD
- Cache cluster in a second region: 0.003 USD
- Orchestration overhead: 0.008 USD
- **Total: 0.171 USD**

The standard dashboard would show only the 0.042 USD attributed to the top-level agent—roughly a quarter of the real cost. The exact figures depend entirely on the model, region, and workload; the point is the ratio between what a flat dashboard captures and what the graph model captures.

## Three failure modes to instrument for

**Retry storms.** A missing dependency constraint in a retry policy can cause an agent to re-execute a non-idempotent sub-task many times. If a single attempt costs 0.042 USD and the agent retries twelve times, the retry overhead alone is 12 × 0.042 = 0.504 USD—an order of magnitude more than the intended single execution. The dashboard shows a spike in the top-level service; the root cause is in the retry policy, which is not a billed resource and therefore invisible to resource-based tracking.

**Idle time.** An agent blocked for twelve seconds on human approval still holds whatever runtime and connection it acquired. If the runtime bills by duration, that idle window is a cost. Idle time is invisible in request-count metrics because the request has not completed.

**Cross-region transfer.** A vector store in one region serving an agent runtime in another incurs egress charges on every query. Moving four megabytes per session at typical egress rates produces a small per-session cost that becomes material at volume. Regional affinity constraints at the orchestrator level prevent this, but only if the constraint is enforced rather than documented.

## When the standard FinOps stack is sufficient

The fractal model is not always warranted, and applying it everywhere adds instrumentation overhead that may exceed the visibility it buys. The standard playbook works when all of the following hold:

- The agentic system is stateless and idempotent.
- There is one agent per request, with no sub-agents.
- All downstream services are in the same region and account.
- Retry policies are deterministic and bounded.
- No external API bills by token or duration.
- No agent spawns another agent.

A chatbot that calls a single LLM endpoint and returns a response fits this model. Cost is additive: prompt tokens plus completion tokens plus latency. Tagging the pod and setting a spend alarm is enough.

A batch agent that runs hourly, processes a fixed number of items, and writes to a single bucket also fits. Cost is predictable and traceable.

In both cases, instrumenting every sub-agent would cost more engineering time than it saves.

## Decision checklist

Ask three questions about the workflow in front of you.

1. **Does the agent spawn sub-agents?** If yes, the fractal model is required. A refund agent that calls a fraud-check sub-agent qualifies; a chatbot calling one LLM does not.
2. **Are downstream services billed by token, duration, or bandwidth?** If yes, the fractal model is required. An embedding API billed by token qualifies; object storage billed by request count may not.
3. **Is there cross-account or cross-region data transfer?** If yes, the fractal model is mandatory, regardless of the other answers.

| Sub-agents? | Token/duration billing? | Cross-region/account? | Recommended model |
|---|---|---|---|
| No | No | No | Standard FinOps |
| Yes | No | No | Fractal (light): track sub-agent lifecycle |
| No | Yes | No | Fractal (light): track token/duration metrics |
| Yes | Yes | No | Fractal (full): instrument every spawn |
| Any | Any | Yes | Fractal (full): mandatory |

For the last two rows, implement the spawn-tracing context manager above. For the first three, standard FinOps with a few additional labels may suffice.

## Common objections

**"This is too much instrumentation when OpenTelemetry already exists."**
Traces and metrics are not billing labels. The fractal model needs each sub-agent mapped to the entity that issues the charge. Baggage can carry those labels, but they must be set explicitly. Without them, traces show latency and not cost.

**"Agentic systems are still rare."**
They are increasingly common in SaaS products, but the more useful point is that the failure mode—a cost spike attributed to the wrong entity—appears as soon as a workflow has sub-agents, retries, or cross-region calls. The threshold for needing the model is structural, not statistical.

**"Let the burn happen, then fix it."**
A retry storm is cheaper to prevent than to diagnose. Adding spawn tracing on day one is a few hours of work; reconstructing which of hundreds of micro-transactions caused a spike after the fact is days of correlation work.

**"Our agents are stateless, so there is no lifecycle to track."**
Stateless agents still have a lifecycle: spawn, execute, return. That lifecycle includes waiting on external APIs, idle loops, and retries. These are cost drivers regardless of whether the agent holds state.

## Implementation order

**Instrument before architecting.** Write the spawn-tracing context manager before the first agent. Run it in development. The first thing it usually reveals is a resource provisioned per session that should be shared—a cache cluster, a connection pool, a vector index handle. The idle cost of a per-session resource is easy to miss and easy to fix once visible.

**Make orchestration billing-aware.** A scheduler that knows the estimated cost of each task can prefer cheaper models for non-critical paths and defer cross-region work to off-peak windows. The estimate does not need to be precise; relative ordering is enough.

**Model idle time explicitly.** Give every agent an idle timeout. When the timeout is exceeded, emit a cost event. This surfaces agents blocked on human approval or slow APIs, which are otherwise invisible until the invoice arrives.

**Enforce regional affinity.** Require every agent to declare where its dependencies live and constrain it to run there. Documented affinity is not enforced affinity; a Kubernetes topology constraint or a Terraform region pin is.

**Add a cost simulation to CI.** Before deploying a new model or prompt, run a fixed number of simulated sessions and compute the graph cost. Fail the build if the average exceeds a threshold. This catches reasoning-chain regressions that increase token spend before they reach production.

## Summary

Agentic FinOps is not pod CPU or function duration tracking. It is tracing the cost of nested, ephemeral, cross-region executions and mapping each to the entity that bills for it. The standard playbook underreports agentic cost because it ignores sub-agent lifecycles, idle time, retry storms, and cross-region transfer.

The fractal model asks what cost means when a single prompt triggers dozens of sub-agents, each billing by token, duration, and bandwidth. The answer is not in a resource dashboard; it is in the dependency graph of the workflow.

Start by instrumenting every agent spawn with a context manager that records spawn time, parent task ID, downstream services, and billing labels. Then model idle time, retries, and cross-region transfer as first-class cost drivers.

## FAQ

**How do I know whether my agentic system is spawning sub-agents?**
Search orchestrator logs for the spawn event emitted by your framework. If more than one spawn occurs per top-level prompt, or if tool calls invoke other agents, you have nested sub-agents. Counting spawns per request is the fastest diagnostic.

**What is the simplest way to add billing labels?**
Use OpenTelemetry baggage with the context manager shown above. Pass labels that identify the billing entity—model identifier, region, service name—rather than only the owning team.

**Isn't this over-engineering for simple agents?**
Simplicity is defined by structure, not by intent. A single-LLM chatbot is simple. A refund agent that calls fraud check, a payment gateway, and vector search is not, even if it looks like one function. The moment retry logic, idle loops, or cross-service calls appear, the flat model starts underreporting.

**How do I enforce regional affinity?**
Set region environment variables in the agent config, add a Kubernetes topology constraint matching the dependency region, and pin the region in your infrastructure definition. Then verify with a test that asserts the agent's runtime region equals its dependency's region.

**What should I measure first?**
Spawn count per top-level request, and the ratio of top-level cost to total graph cost for a representative session. If that ratio is below roughly one half, the flat dashboard is missing enough spend to justify the instrumentation.

## Do this in the next 30 minutes

Pick one production agent workflow and add a single log line at every spawn point that emits the parent task ID, the spawned task ID, and the billing entity for that task. Run one representative request, collect the lines, and sum the distinct billing entities. Compare that count to the number of entities your current cost dashboard attributes to the workflow. If the dashboard shows fewer, you have found the gap the graph model closes.
