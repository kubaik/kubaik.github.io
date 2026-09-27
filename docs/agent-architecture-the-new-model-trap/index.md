# Agent architecture: the new model trap

Benchmarks for claude gpt5 that don't mention their failure conditions aren't worth much. This walks through the fix and the reasoning, not just the patch. Production gives you neither a clean environment nor a patient timeline.

## The conventional wisdom (and why it's incomplete)

The conventional wisdom around the Claude 4 / GPT-5 generation of models goes something like this: models are now so capable that the old scaffolding is dead weight. Stop building state machines. Stop writing retry logic. Stop wrapping the model in five layers of validation. Just give it tools, give it memory, and let it reason its way through the task. The model *is* the agent.

This advice is seductive because it is partly true. The jump from GPT-4-class models to the Claude 4 / GPT-5 generation is not incremental. Tool-calling reliability improved to the point where a single-shot tool invocation with a well-described schema succeeds the vast majority of the time, whereas earlier generations needed multiple retries and aggressive prompt engineering to get the same result. Long-context handling became genuinely usable rather than a demo trick. Instruction-following under multi-step constraints stopped collapsing after the third hop.

So the reasoning goes: if the model can plan, call tools, and recover from errors on its own, then the elaborate orchestration layer that teams built in 2026 and 2026 is technical debt. Delete the DAG. Delete the hand-rolled retry loop. Delete the state machine. Ship the raw agent loop.

I think this is the wrong conclusion, and it is wrong for a specific reason that has nothing to do with model capability. The models got better at *reasoning*. They did not get better at being *accountable*. And in any system that touches production data, money, or a user's account, accountability is the entire problem. The part that trips people up is that the new models make the demo so easy that teams delete the wrong layer — they remove the deterministic boundary instead of removing the redundant prompt scaffolding, and that's what this post actually covers.

## What actually happens when you follow the standard advice

The standard advice produces a very predictable failure mode, and it shows up around week three of a production deployment, not on day one.

Here is the typical trajectory. A team builds a customer-support agent on Claude 4 or GPT-5. They wire up five tools: `lookup_order`, `issue_refund`, `send_email`, `update_shipping_address`, and `escalate_to_human`. They write a clean agent loop, maybe 120 lines of Python. It works beautifully in staging. Latency is around 1.8 seconds for a simple lookup, 4 to 6 seconds for a multi-tool task. Great.

Then it hits production. A user asks to "cancel my order and refund me." The model, reasoning correctly, calls `lookup_order`, sees the order is already shipped, and decides the right move is to issue a partial refund and send a follow-up email explaining the situation. That is a *reasonable* decision. It is also a decision that your finance team, your refund policy, and your auditor all disagree with, because partial refunds on shipped orders require manual approval above a threshold.

The model did not hallucinate. It did not call a wrong tool. It made a judgment call inside a policy boundary that existed only in a PDF, not in the code. And because you deleted the orchestration layer, there is nothing between the model's decision and the `issue_refund` API call.

A related failure mode is the runaway loop. With the new models, a common pattern is to give the agent a `search` tool and let it iterate. Under a malformed query, a model in the Claude 4 / GPT-5 class will sometimes retry with variations. Without a hard iteration cap enforced in code, a single user request can trigger 40 to 60 tool calls before the context window forces a stop. At typical token pricing, that is a 15x cost multiplier on an interaction that should have cost fractions of a cent. The model is not malfunctioning; it is doing what you told it to do, and you did not tell it when to stop in a way it cannot override.

The third common failure is the audit gap. When a regulator or an internal auditor asks "why did this agent issue this refund on this date," the answer needs to be a reconstructable trace: which tool was called, with which arguments, under which policy version. A raw agent loop that logs the final message and the tool calls is not sufficient, because the *decision rationale* lives in the model's reasoning, which is non-deterministic and, depending on your provider configuration, may not even be retained. You cannot replay a non-deterministic model call and get the same answer. You can only replay the deterministic boundary around it.

## A different mental model

The mental model I would argue for is this: **the model is a planner, not an executor.** The new generation is genuinely excellent at deciding *what* should happen next. It is not, and should not be, the thing that decides *whether* that action is permitted, *how many times* it may be attempted, or *what record* is kept of it.

Concretely, this means the orchestration layer does not disappear. It shrinks and it changes shape. In 2026, the orchestration layer was doing two jobs: (1) compensating for weak reasoning by breaking tasks into small, tightly-scoped prompts, and (2) enforcing policy, retries, and audit. The new models let you delete job (1) almost entirely. Job (2) becomes *more* important, not less, because the model is now making higher-stakes decisions more autonomously.

So the architecture shifts from "many small prompts, thin policy layer" to "one capable planner, thick policy layer." The thick policy layer is deterministic code. It is the thing that says: this tool call is allowed for this user role; this refund amount exceeds the threshold and must be escalated; this loop has run 8 times and must terminate; this action must be written to an append-only audit log before it executes, not after.

This is not a return to the 2026 DAG. A DAG hard-codes the sequence of steps. A policy layer does not care about sequence; it cares about *constraints on any sequence the planner chooses*. That is a genuinely different thing, and it is the thing the new models make possible. You can let the planner be creative because the policy layer is the guardrail, not the prompt.

## Evidence and examples from real systems

Consider how this plays out in a payments-adjacent agent. The planner is a Claude 4 or GPT-5 class model. The tools are real. The policy layer is a small Python module that wraps every tool call.

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

The second pattern worth naming is the budget cap. A common trap with the new models is that they are good enough to keep trying, which means a single ambiguous request can burn a surprising amount of money. A hard cap enforced outside the model is the only reliable fix.

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

Two numbers worth internalizing here. First, a 10-iteration cap with a 30-second wall clock is a reasonable default for interactive agents; typical successful tasks complete in 2 to 4 iterations, so the cap rarely fires in normal operation. Second, without a cap, the same ambiguous request that should cost roughly 0.4 cents can cost 6 to 10 cents when the model retries 15 to 25 times. That is a 15x to 25x multiplier on a single interaction, and it compounds across thousands of daily requests.

The third evidence point is audit. If you are serving EU users, GDPR Article 5(2) establishes the accountability principle, and Article 22 restricts solely automated decision-making with legal or similarly significant effects. You do not need to be a lawyer to draw the engineering conclusion: if your agent can issue a refund, suspend an account, or reject an application without human review, you need a reconstructable record of *why*, and you need a documented path to a human. A raw agent loop that logs only the final output does not give you that. A policy layer that writes a structured event before every consequential tool call does.

## The cases where the conventional wisdom IS right

I want to steelman the "delete the orchestration layer" position, because it is correct in more cases than the previous section implies.

If your agent is read-only, the policy layer is mostly overhead. A research agent that searches a corpus and summarizes findings has no side effects. The worst outcome of a bad decision is a bad answer, which the user will notice and correct. In that case, a thin loop with a tool-call budget is genuinely sufficient, and adding a policy engine is ceremony.

If your agent is internal-only and low-stakes, the same logic applies. An agent that drafts tickets, summarizes incidents, or answers questions from an internal wiki does not need a pre-execution audit log. The cost of a wrong action is low and reversible.

If your agent's tools are themselves idempotent and bounded, the policy layer's job shrinks. A `search` tool that cannot mutate state does not need a pre-call permission check. A `fetch_document` tool with a fixed scope does not need a refund threshold.

The conventional wisdom is also right that the *prompt-level* scaffolding should go. The old pattern of chaining six prompts with hand-written JSON parsers between them is genuinely obsolete. The new models handle multi-step reasoning in a single context window, and cutting that chaining removes a large class of parsing bugs. So when someone says "delete the orchestration layer," they are often right about the *prompt chaining* layer and wrong about the *policy* layer. Those are different layers that got conflated in 2026 because they were implemented in the same place.

| Layer | 2026 role | 2026 role | Delete it? |
|---|---|---|---|
| Prompt chaining | Compensate for weak reasoning | Mostly unnecessary | Yes, largely |
| Tool schema design | Basic function calling | Critical for reliability | No, more important |
| Policy / permission checks | Minimal | Central | No, expand |
| Retry logic | Model-level retries | Budget and loop caps | No, reframe |
| Audit logging | Optional | Required for consequential actions | No, expand |
| State machine / DAG | Enforce sequence | Rarely needed | Usually yes |

## How to decide which approach fits your situation

The decision rule I would use is blunt: **does any tool your agent can call have a side effect that is hard to reverse, costs money, or affects a user's rights?**

If the answer is no across every tool, run the thin loop. Give the planner tools, cap iterations at 10, cap wall clock at 30 seconds, log the final trace, and move on. You will ship faster and the model will handle the rest.

If the answer is yes for even one tool, you need a policy layer, and you need it before the tool executes, not after. The policy layer does not have to be a framework. It can be a single function per consequential tool, as in the Python example above. The point is that the check is deterministic code that the model cannot reason its way around.

A useful intermediate heuristic: count the number of tools that can mutate state. If that count is zero, thin loop. If it is one or two, wrap those two tools and leave the rest unwrapped. If it is more than two, or if any tool affects money or user rights, build the policy layer as a first-class module with its own tests and its own audit sink.

The other axis is regulatory exposure. If you serve EU users and your agent makes decisions with legal or similarly significant effects, GDPR Article 22 gives you a concrete obligation: either keep a human in the loop, or be able to explain the decision and offer a path to contest it. That obligation lands on the policy layer, not on the model. The model cannot be audited; the code around it can.

## Common objections, and responses

**"The model is smart enough to follow the policy if I put it in the system prompt."** Sometimes, most of the time, until it is not. Prompt-level policy is a suggestion. A model under pressure from a persuasive user, an ambiguous request, or a long context will occasionally violate a prompt instruction. A code-level check cannot be talked out of a decision. For low-stakes tools, prompt policy is fine. For refunds and account actions, it is not.

**"A policy layer adds latency."** A permission check and an audit write are sub-millisecond operations. The model call dominates latency at 800ms to 4 seconds depending on task complexity. The policy layer is noise in the latency budget. The real latency cost of the new generation is the longer reasoning traces, not the guardrails.

**"This is just bringing back the DAG."** No. A DAG dictates sequence. A policy layer constrains choices. The planner can call tools in any order it wants; it just cannot exceed a budget, violate a permission, or skip the audit write. That is a fundamentally different and much less brittle structure than a hard-coded graph.

**"Our compliance team has not asked for any of this."** They will, and the ask will arrive with a deadline. GDPR Article 5(2) already establishes accountability as a principle, and Article 22 already restricts solely automated decisions. Building the audit sink now costs a day. Retrofitting it into a live agent loop that has been issuing refunds for six months costs considerably more, because you will be reconstructing history you did not record.

## What the alternative approach would change

If teams adopted the planner-plus-policy model instead of the raw-loop model, three things would change in practice.

First, incident response would get faster. When an agent does something wrong, the question "what did it decide and why" has a deterministic answer in the audit log, not a probabilistic one in a re-run. You would stop trying to reproduce non-deterministic model behavior and start reading structured events.

Second, cost predictability would improve. A hard iteration and wall-clock cap turns a variable-cost agent into a bounded-cost one. Teams running high-volume agents would stop seeing the long tail of expensive interactions that comes from a model that keeps trying.

Third, the compliance conversation would move from "we think it is fine" to "here is the trace." That is a materially different posture when a regulator or an enterprise customer asks how the system behaves.

The tradeoff is real: the policy layer is code you have to write, test, and maintain. It is not free. But it is a bounded, well-understood cost, and it replaces an unbounded, poorly-understood risk. For any agent with consequential tools, that trade is worth making.

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

The Claude 4 / GPT-5 generation did not make orchestration obsolete. It made one kind of orchestration obsolete — the prompt-chaining that existed to compensate for weak reasoning — and made another kind more important: the deterministic policy layer that decides what the planner is allowed to do. The teams that delete the wrong layer ship fast and then discover the failure mode in production, usually around week three, usually in the form of an action the model was never authorized to take.

The correct architecture for any agent with consequential tools is a capable planner wrapped in a thin, deterministic policy layer that enforces permissions, caps iterations, and writes an audit record before each action. That layer is 40 to 80 lines of code, not a framework. It is the difference between an agent you can explain to an auditor and one you cannot.

Here is the specific next step: open the module where your agent's tool calls are dispatched, and add a single counter that increments on every call and raises an exception past a fixed limit — start with 10. That one change bounds your worst-case cost and gives you a place to put the rest of the policy layer.


---

### About this article

**Written by:** [Kubai Kevin](/about/) — software developer based in Nairobi, Kenya, with 10+ years building production systems in fintech and AI.

**How this article was produced:** This site uses an automated LLM pipeline designed and maintained by the author. Topics are selected from real production experience. Drafts pass automated quality gates (minimum length, uniqueness, concrete metrics, versioned tools, code samples, absence of filler). Individual line-by-line human editing is not performed on every post before publication. Specific numbers, benchmarks and cost figures are illustrative; verify them against current official documentation before production use.

**Corrections:** Report errors via the [contact page](/contact/). Corrections are applied promptly.

**Last generated:** September 2026
