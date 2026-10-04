# Agent architecture: the new model trap

Benchmarks that report capability without describing failure conditions tell you very little about production behavior. Production supplies neither a clean environment nor a patient timeline, and the gap between a demo and a deployed agent is where most incidents live.

## The conventional wisdom (and why it's incomplete)

The conventional wisdom around recent frontier models goes something like this: models are now capable enough that the old scaffolding is dead weight. Stop building state machines. Stop writing retry logic. Stop wrapping the model in five layers of validation. Give it tools, give it memory, and let it reason its way through the task. The model *is* the agent.

This advice is seductive because it is partly true. The jump from earlier generations to current frontier models is not incremental in the ways that matter for agent design. Tool-calling reliability improved to the point where a single-shot tool invocation with a well-described schema succeeds most of the time, whereas earlier generations needed multiple retries and aggressive prompt engineering to get the same result. Long-context handling became usable rather than a demo trick. Instruction-following under multi-step constraints stopped collapsing after the third hop.

So the reasoning goes: if the model can plan, call tools, and recover from errors on its own, then the elaborate orchestration layer that teams built around earlier models is technical debt. Delete the DAG. Delete the hand-rolled retry loop. Delete the state machine. Ship the raw agent loop.

That conclusion is wrong, and it is wrong for a reason that has nothing to do with model capability. The models got better at *reasoning*. They did not get better at being *accountable*. In any system that touches production data, money, or a user's account, accountability is the entire problem. The trap is that the new models make the demo so easy that teams delete the wrong layer — they remove the deterministic boundary instead of removing the redundant prompt scaffolding.

## What actually happens when you follow the standard advice

The standard advice produces a predictable failure mode, and it tends to appear weeks into a production deployment rather than on day one.

A typical trajectory: a team builds a customer-support agent on a current frontier model. They wire up five tools — `lookup_order`, `issue_refund`, `send_email`, `update_shipping_address`, and `escalate_to_human`. They write a clean agent loop, perhaps 120 lines of Python. It works well in staging. Latency is acceptable for a simple lookup and higher but tolerable for a multi-tool task.

Then it hits production. A user asks to "cancel my order and refund me." The model, reasoning correctly, calls `lookup_order`, sees the order is already shipped, and decides the right move is to issue a partial refund and send a follow-up email explaining the situation. That is a *reasonable* decision. It is also a decision that the finance team, the refund policy, and an auditor may all disagree with, because partial refunds on shipped orders require manual approval above a threshold.

The model did not hallucinate. It did not call a wrong tool. It made a judgment call inside a policy boundary that existed only in a PDF, not in the code. And because the orchestration layer was deleted, there is nothing between the model's decision and the `issue_refund` API call.

A related failure mode is the runaway loop. A common pattern is to give the agent a `search` tool and let it iterate. Under a malformed query, a capable model will sometimes retry with variations. Without a hard iteration cap enforced in code, a single user request can trigger dozens of tool calls before the context window forces a stop. At typical token pricing, that is a large cost multiplier on an interaction that should have cost fractions of a cent. The model is not malfunctioning; it is doing what it was told, and it was not told when to stop in a way it cannot override.

The third common failure is the audit gap. When a regulator or an internal auditor asks "why did this agent issue this refund on this date," the answer needs to be a reconstructable trace: which tool was called, with which arguments, under which policy version. A raw agent loop that logs the final message and the tool calls is not sufficient, because the *decision rationale* lives in the model's reasoning, which is non-deterministic and, depending on provider configuration, may not even be retained. A non-deterministic model call cannot be replayed to produce the same answer. Only the deterministic boundary around it can be replayed.

## A different mental model

The mental model worth adopting: **the model is a planner, not an executor.** Current frontier models are genuinely good at deciding *what* should happen next. They should not be the thing that decides *whether* that action is permitted, *how many times* it may be attempted, or *what record* is kept of it.

Concretely, the orchestration layer does not disappear. It shrinks and changes shape. In earlier architectures the orchestration layer did two jobs: (1) compensating for weak reasoning by breaking tasks into small, tightly-scoped prompts, and (2) enforcing policy, retries, and audit. Newer models let teams delete job (1) almost entirely. Job (2) becomes *more* important, not less, because the model is now making higher-stakes decisions more autonomously.

So the architecture shifts from "many small prompts, thin policy layer" to "one capable planner, thick policy layer." The thick policy layer is deterministic code. It is the thing that says: this tool call is allowed for this user role; this refund amount exceeds the threshold and must be escalated; this loop has run eight times and must terminate; this action must be written to an append-only audit log before it executes, not after.

This is not a return to the old DAG. A DAG hard-codes the sequence of steps. A policy layer does not care about sequence; it cares about *constraints on any sequence the planner chooses*. That is a genuinely different thing, and it is what newer models make possible. The planner can be creative because the policy layer is the guardrail, not the prompt.

## A worked example: payments-adjacent agent

Consider how this plays out in a payments-adjacent agent. The planner is a frontier model. The tools are real. The policy layer is a small Python module that wraps every consequential tool call.

```python
from dataclasses import dataclass
from datetime import datetime, timezone

MAX_REFUND_EUR = 50.00
MAX_TOOL_CALLS = 12

@dataclass
class PolicyDecision:
    allowed: bool
    reason: str

def evaluate_refund(user_role: str, amount_eur: float, order_shipped: bool) -> PolicyDecision:
    if amount_eur > MAX_REFUND_EUR and user_role != "support_lead":
        return PolicyDecision(False, "refund_above_threshold_requires_lead")
    if order_shipped and amount_eur > 0:
        return PolicyDecision(False, "shipped_order_refund_requires_manual_review")
    return PolicyDecision(True, "ok")

def guarded_tool_call(tool_name, args, state):
    state["tool_calls"] += 1
    if state["tool_calls"] > MAX_TOOL_CALLS:
        raise RuntimeError("tool_call_budget_exceeded")

    if tool_name == "issue_refund":
        decision = evaluate_refund(
            user_role=state["user_role"],
            amount_eur=args["amount_eur"],
            order_shipped=args["order_shipped"],
        )
        if not decision.allowed:
            audit_log.write({
                "ts": datetime.now(timezone.utc).isoformat(),
                "event": "refund_blocked",
                "reason": decision.reason,
                "args": args,
            })
            return {"error": decision.reason, "escalate": True}

    audit_log.write({
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": "tool_call",
        "tool": tool_name,
        "args": args,
    })
    return TOOL_REGISTRY[tool_name](**args)
```

This is roughly 40 lines. It is not a DAG. It does not tell the planner what to do. It tells the planner what it *cannot* do, and it writes a record before the action, not after. The planner can still be creative; it just cannot exceed a 50 EUR refund without a lead role, and it cannot issue a refund on a shipped order at all without a human in the loop.

The second pattern worth naming is the budget cap. A common trap with capable models is that they are good enough to keep trying, which means a single ambiguous request can burn a surprising amount of money. A hard cap enforced outside the model is the only reliable fix.

```javascript
// Node 20 LTS, wrapping the Anthropic SDK call
const MAX_ITERATIONS = 10;
const MAX_WALL_CLOCK_MS = 30_000;

async function runAgent(userMessage, tools, policy) {
  const started = Date.now();
  let iterations = 0;
  const trace = [];

  while (iterations < MAX_ITERATIONS) {
    if (Date.now() - started > MAX_WALL_CLOCK_MS) {
      return { status: "timeout", trace };
    }
    iterations += 1;

    const response = await client.messages.create({
      model: "claude-sonnet-4-5",
      max_tokens: 1024,
      messages: buildMessages(userMessage, trace),
      tools,
    });

    const toolUse = response.content.find((b) => b.type === "tool_use");
    if (!toolUse) {
      return { status: "done", trace, final: response.content };
    }

    const decision = policy.check(toolUse.name, toolUse.input);
    if (!decision.allowed) {
      trace.push({ blocked: toolUse.name, reason: decision.reason });
      return { status: "blocked", trace, reason: decision.reason };
    }

    const result = await tools[toolUse.name](toolUse.input);
    trace.push({ tool: toolUse.name, input: toolUse.input, result });
  }

  return { status: "iteration_cap", trace };
}
```

Two numbers are worth internalizing here. First, a 10-iteration cap with a 30-second wall clock is a reasonable starting point for interactive agents; most successful tasks complete in a handful of iterations, so the cap rarely fires in normal operation. Second, without a cap, an ambiguous request that should cost a fraction of a cent can cost many times that when the model retries repeatedly. That multiplier compounds across thousands of daily requests.

To measure the actual multiplier in your own system, instrument three things: the iteration count per request, the total input and output tokens per request, and the wall-clock duration per request. Emit them as structured logs, then compute the p95 and p99 of each. The gap between the median and the p99 iteration count is the tail you are paying for. If p99 is more than three times the median, a cap will pay for itself quickly.

The third pattern is audit. If you serve EU users, GDPR Article 5(2) establishes the accountability principle, and Article 22 restricts solely automated decision-making with legal or similarly significant effects. The engineering conclusion follows directly: if an agent can issue a refund, suspend an account, or reject an application without human review, you need a reconstructable record of *why*, and a documented path to a human. A raw agent loop that logs only the final output does not give you that. A policy layer that writes a structured event before every consequential tool call does.

## The cases where the conventional wisdom IS right

The "delete the orchestration layer" position is correct in more cases than the previous section implies, and it is worth steelmanning.

If the agent is read-only, the policy layer is mostly overhead. A research agent that searches a corpus and summarizes findings has no side effects. The worst outcome of a bad decision is a bad answer, which the user will notice and correct. A thin loop with a tool-call budget is genuinely sufficient, and adding a policy engine is ceremony.

If the agent is internal-only and low-stakes, the same logic applies. An agent that drafts tickets, summarizes incidents, or answers questions from an internal wiki does not need a pre-execution audit log. The cost of a wrong action is low and reversible.

If the agent's tools are themselves idempotent and bounded, the policy layer's job shrinks. A `search` tool that cannot mutate state does not need a pre-call permission check. A `fetch_document` tool with a fixed scope does not need a refund threshold.

The conventional wisdom is also right that *prompt-level* scaffolding should go. The old pattern of chaining six prompts with hand-written JSON parsers between them is largely obsolete. Newer models handle multi-step reasoning in a single context window, and cutting that chaining removes a large class of parsing bugs. When someone says "delete the orchestration layer," they are often right about the *prompt chaining* layer and wrong about the *policy* layer. Those are different layers that got conflated because they were often implemented in the same place.

| Layer | Old role | Current role | Delete it? |
|---|---|---|---|
| Prompt chaining | Compensate for weak reasoning | Mostly unnecessary | Yes, largely |
| Tool schema design | Basic function calling | Critical for reliability | No, more important |
| Policy / permission checks | Minimal | Central | No, expand |
| Retry logic | Model-level retries | Budget and loop caps | No, reframe |
| Audit logging | Optional | Required for consequential actions | No, expand |
| State machine / DAG | Enforce sequence | Rarely needed | Usually yes |

## How to decide which approach fits your situation

The decision rule is blunt: **does any tool the agent can call have a side effect that is hard to reverse, costs money, or affects a user's rights?**

If the answer is no across every tool, run the thin loop. Give the planner tools, cap iterations at 10, cap wall clock at 30 seconds, log the final trace, and move on. Shipping is faster and the model handles the rest.

If the answer is yes for even one tool, a policy layer is needed, and it is needed before the tool executes, not after. The policy layer does not have to be a framework. It can be a single function per consequential tool, as in the Python example above. The point is that the check is deterministic code the model cannot reason its way around.

A useful intermediate heuristic: count the number of tools that can mutate state. If that count is zero, thin loop. If it is one or two, wrap those two tools and leave the rest unwrapped. If it is more than two, or if any tool affects money or user rights, build the policy layer as a first-class module with its own tests and its own audit sink.

The other axis is regulatory exposure. If the system serves EU users and the agent makes decisions with legal or similarly significant effects, GDPR Article 22 imposes a concrete obligation: either keep a human in the loop, or be able to explain the decision and offer a path to contest it. That obligation lands on the policy layer, not on the model. The model cannot be audited; the code around it can.

## Common objections, and responses

**"The model is smart enough to follow the policy if I put it in the system prompt."** Sometimes, most of the time, until it is not. Prompt-level policy is a suggestion. A model under pressure from a persuasive user, an ambiguous request, or a long context will occasionally violate a prompt instruction. A code-level check cannot be talked out of a decision. For low-stakes tools, prompt policy is fine. For refunds and account actions, it is not.

**"A policy layer adds latency."** A permission check and an audit write are sub-millisecond operations. The model call dominates latency. The policy layer is noise in the latency budget. The real latency cost of newer models is longer reasoning traces, not guardrails.

**"This is just bringing back the DAG."** No. A DAG dictates sequence. A policy layer constrains choices. The planner can call tools in any order it wants; it just cannot exceed a budget, violate a permission, or skip the audit write. That is a fundamentally different and much less brittle structure than a hard-coded graph.

**"Compliance has not asked for any of this."** They will, and the ask will arrive with a deadline. GDPR Article 5(2) already establishes accountability as a principle, and Article 22 already restricts solely automated decisions. Building the audit sink now costs a day. Retrofitting it into a live agent loop that has been issuing refunds for months costs considerably more, because it means reconstructing history that was never recorded.

## What the alternative approach would change

If teams adopted the planner-plus-policy model instead of the raw-loop model, three things would change in practice.

First, incident response would get faster. When an agent does something wrong, the question "what did it decide and why" has a deterministic answer in the audit log, not a probabilistic one in a re-run. Teams would stop trying to reproduce non-deterministic model behavior and start reading structured events.

Second, cost predictability would improve. A hard iteration and wall-clock cap turns a variable-cost agent into a bounded-cost one. Teams running high-volume agents would stop seeing the long tail of expensive interactions that comes from a model that keeps trying.

Third, the compliance conversation would move from "we think it is fine" to "here is the trace." That is a materially different posture when a regulator or an enterprise customer asks how the system behaves.

The tradeoff is real: the policy layer is code that must be written, tested, and maintained. It is not free. But it is a bounded, well-understood cost, and it replaces an unbounded, poorly-understood risk. For any agent with consequential tools, that trade is worth making.

## Frequently Asked Questions

**How do I stop an AI agent from calling tools in an infinite loop?**
Enforce an iteration cap and a wall-clock cap in the code that runs the agent loop, not in the prompt. A typical default is 10 iterations and 30 seconds per user request. The model cannot override a counter that lives in your runtime. Log every iteration so you can see when the cap fires and tune it.

**Why does my agent ignore the policy in the system prompt?**
Because a system prompt is an instruction, not a constraint. Under ambiguous input, long context, or adversarial user pressure, models occasionally deviate from prompt-level rules. Move any rule that protects money, user rights, or irreversible state into deterministic code that runs before the tool executes.

**What does GDPR Article 22 require for AI agents?**
Article 22 restricts solely automated decisions that produce legal or similarly significant effects, and gives data subjects rights around those decisions. In practice that means either keeping a human in the loop for consequential actions or being able to explain the decision and offer a path to contest it. Both require a structured audit trail, which is a code-level concern.

**When is a raw agent loop actually fine?**
When every tool the agent can call is read-only or trivially reversible. Research agents, internal Q&A agents, and summarization agents fit this pattern. If no tool mutates state, costs money, or affects a user's rights, a thin loop with a tool-call budget is sufficient and a policy engine is overhead.

## Summary

Recent frontier models did not make orchestration obsolete. They made one kind of orchestration obsolete — the prompt-chaining that existed to compensate for weak reasoning — and made another kind more important: the deterministic policy layer that decides what the planner is allowed to do. Teams that delete the wrong layer ship fast and then discover the failure mode in production, usually weeks in, usually in the form of an action the model was never authorized to take.

The correct architecture for any agent with consequential tools is a capable planner wrapped in a thin, deterministic policy layer that enforces permissions, caps iterations, and writes an audit record before each action. That layer is tens of lines of code, not a framework. It is the difference between an agent that can be explained to an auditor and one that cannot.

Here is the specific next step: open the module where your agent's tool calls are dispatched, and add a single counter that increments on every call and raises an exception past a fixed limit — start with 10. That one change bounds your worst-case cost and gives you a place to put the rest of the policy layer.
