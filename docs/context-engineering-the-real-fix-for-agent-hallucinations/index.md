# Context engineering: the real fix for agent hallucinations

Context engineering is easy to demo and hard to keep honest at scale. Workarounds get copy-pasted forward long after the original reason for them is forgotten. This article covers the root cause of a specific class of agent failure, not just the symptom.

Most teams treat hallucinations as a model problem. They upgrade the model, tweak the temperature, add a system prompt that says "do not make things up." Then the agent still invents a function name that doesn't exist, or claims it already sent the email when it only drafted it. Sometimes the model is the issue. Often the context is.

Long-running agents — ones that loop over many steps, call tools, and maintain state — fail in a specific way: the context window becomes a junk drawer. Old tool outputs, stale observations, contradictory instructions, and half-finished reasoning pile up. The model, forced to attend to all of it, does the statistically plausible thing: it confabulates a coherent story that ignores what actually happened. The fix is not always a better model. It is deliberate context engineering: deciding what goes in, what stays out, and how it is structured.

This article is about the patterns that reduce that failure mode. It assumes you already know how to call an LLM API and have built at least one agent loop. It is not about prompt tricks. It is about the architecture of the context you feed the model on every turn.

## The gap between the demo and the 40th step

Every agent tutorial shows a three-step loop. The model gets a task, calls a tool, gets a result, and answers. That works because the context is small and fresh. The gap opens when the loop runs for dozens or hundreds of steps. A typical failure report reads like this: the agent worked in testing, then in a longer session started referencing tools it never called or facts it never retrieved.

The documented behaviour of transformer models is that they attend to the entire provided context, with learned positional biases. Nothing in the API contract says the model will ignore irrelevant tokens. In practice, irrelevant tokens compete for attention. A tool output from step 3 that is no longer relevant still occupies space and can be retrieved by the model as if it were current. This is not a bug in the model; it is the expected behaviour of a system that treats all context as equally available.

The production reality is that context is a scarce, expensive resource that must be curated. The model has no concept of "this observation is stale." You have to encode that.

## Why stale observations cause confabulation

The mechanism is straightforward. An agent's context typically contains:

- The system prompt (instructions, tool definitions)
- The conversation history (user messages, assistant messages)
- Tool call records (the call and its result)
- Scratchpad or reasoning traces

As the loop progresses, the conversation history grows. If you simply append every tool result, the context becomes a chronological log. The model, when asked to decide the next action, must infer relevance from position and wording. A tool result from twenty steps ago that says "user email: alice@example.com" may be treated as current even if the user has since changed their email. Worse, if the agent previously hallucinated a tool result (because a tool failed and the model filled in a plausible output), that hallucination is now part of the context and will be treated as ground truth on subsequent turns.

The counterintuitive part is that adding more context often makes hallucinations worse, not better. Teams sometimes try to fix a confused agent by dumping the entire history back in. That increases the attention budget spent on irrelevant tokens and makes the model more likely to latch onto a plausible but wrong detail. The right move is usually to remove context, not add it.

A worked example makes the compounding effect concrete. Suppose a `read_file` tool fails at step 4 with a timeout. The model, seeing only an error string and no structured status, writes "the file contains Q3 figures" as a plausible placeholder. That sentence enters the history. At step 12, the agent is asked to compute a total from the file. It retrieves "the file contains Q3 figures," treats it as an observation, and proceeds to invent numbers. At step 20, those numbers have been restated twice and now look like established facts. One unstructured error at step 4 has produced three downstream fabrications. Structured status fields and a separate facts store break this chain at the first link.

## Patterns that keep the context honest

These patterns are not mutually exclusive. Most production agents use a combination. The goal is to ensure that at any given step, the model sees a context that is accurate, relevant, and unambiguous about what is known versus what is assumed.

### 1. Structured tool results with explicit status

A tool result should not be a bare string. It should be a structured object that includes a status and a timestamp or sequence number. This gives the model a signal about reliability.

For example, instead of returning `"File not found"` as a string, return:

```json
{
  "tool": "read_file",
  "status": "error",
  "error": "ENOENT: no such file or directory",
  "path": "/tmp/report.csv",
  "sequence": 17
}
```

The model can then reason about the error rather than inventing a file content. The `sequence` field also helps with deduplication and ordering.

### 2. Summarise and prune on a schedule

Do not let the raw history grow unbounded. After every N steps (or when token count crosses a threshold), replace the oldest K messages with a summary that preserves decisions and state changes, not verbatim text. The summary should be generated by the model itself with a prompt like: "Summarise the conversation so far, focusing on facts established, decisions made, and unresolved questions. Do not include tool call details unless they changed the state."

A common mistake is to summarise too aggressively and lose critical constraints. A better approach is to maintain a separate "state" object — a key-value store of established facts — and include that in every prompt, while summarising the narrative history. The state object is authoritative; the summary is just for continuity.

### 3. Explicitly mark assumptions and unknowns

When the model makes an assumption, it should be labelled. A pattern that works well is to require the model to output a structured plan with fields like `known_facts`, `assumptions`, and `next_action`. You then inject only `known_facts` and `next_action` into the context for the next step, discarding the assumptions unless they are later validated. This prevents an assumption from silently becoming a fact.

### 4. Tool definitions are context too

Every tool you expose consumes context. If you have 30 tools, the model must attend to all of them. Teams often add tools liberally and then wonder why the agent calls the wrong one. A pattern that reduces errors is to dynamically select a subset of tools based on the current step's intent, and only include those in the context. This is sometimes called "tool routing" or "tool retrieval." It is not perfect, but it reduces the chance of the model picking a tool that exists but is irrelevant.

### 5. Use a scratchpad that is not part of the main context

Chain-of-thought reasoning is useful, but the raw reasoning trace should not persist in the main context forever. A pattern is to have the model write its reasoning to a separate "scratchpad" channel that is not included in subsequent prompts, except perhaps the final conclusion. This keeps the main context clean. Some API providers support separate "reasoning" fields; if yours does not, you can simulate it by instructing the model to output reasoning inside a specific XML tag and then stripping that tag before the next turn.

### 6. Negative constraints and explicit "do not" instructions

Models are bad at negative instructions in general, but they are worse when the context is noisy. If you must say "do not invent tool names," put it close to the tool definitions and make it concrete: "Only use the tools listed below. If a tool is not listed, respond with 'I don't have that tool.'" This is more effective than a generic "don't hallucinate."

## A minimal implementation: context manager with state and pruning

Below is a self-contained Python example of a context manager that implements state tracking and pruning. It is not tied to any specific LLM API; it produces a list of messages you can send to your model.

```python
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class AgentState:
    facts: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, str]] = field(default_factory=list)
    step: int = 0

    def add_fact(self, key: str, value: Any):
        self.facts[key] = value

    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        self.step += 1

    def prune(self, max_messages: int = 10):
        """Keep only the most recent messages, but always keep the system prompt."""
        if len(self.history) > max_messages:
            # Assume first message is system prompt; preserve it.
            system = self.history[0] if self.history and self.history[0]["role"] == "system" else None
            recent = self.history[-max_messages:]
            # Drop a leading system message from the recent slice to avoid duplication.
            if system and recent and recent[0]["role"] == "system":
                recent = recent[1:]
            self.history = ([system] if system else []) + recent

    def build_context(self, system_prompt: str, tool_definitions: List[dict]) -> List[dict]:
        """Construct the message list for the next LLM call."""
        # Start with system prompt that includes tool definitions and current facts.
        facts_str = json.dumps(self.facts, indent=2) if self.facts else "{}"
        full_system = f"{system_prompt}\n\nCurrent known facts:\n{facts_str}\n\nAvailable tools:\n{json.dumps(tool_definitions, indent=2)}"
        # Replace or insert system message.
        messages = [{"role": "system", "content": full_system}]
        # Append history excluding any previous system message.
        messages.extend([m for m in self.history if m["role"] != "system"])
        return messages

# Example usage:
state = AgentState()
state.add_fact("user_name", "Alice")
state.add_message("user", "Find the latest sales report and summarise it.")
state.add_message("assistant", "I will use the search tool.")
state.add_message("tool", json.dumps({"tool": "search", "status": "ok", "result": "report.csv"}))
state.prune(max_messages=5)
context = state.build_context(
    system_prompt="You are a helpful assistant. Only use the tools provided.",
    tool_definitions=[{"name": "search", "description": "Search for files"}]
)
print(json.dumps(context, indent=2))
```

This example is minimal but demonstrates the key ideas: state is separated from history, pruning keeps the context bounded, and the system prompt is rebuilt each time with current facts and tools. In a real agent, you would also add summarisation of pruned messages and validation of tool results.

Two caveats on the code above. First, `prune` truncates rather than summarises, so information in dropped messages is lost outright; pair it with a summarisation step that writes durable conclusions into `facts` before truncation. Second, `build_context` re-serialises the full facts and tool list on every call, which is fine for a handful of facts but should be capped if the store grows large.

## Measuring whether your context engineering works

You cannot improve what you do not measure. The most useful metric for context engineering is not accuracy on a benchmark; it is the rate of unsupported claims per step. An unsupported claim is any assertion the agent makes that is not directly backed by a tool result or a user message in the current context.

To measure this, log every assistant message and every tool call, together with the exact context sent at that step. Then, for each assistant message, run a lightweight check: extract entities and facts, and verify that each appears in the context that was available at that step. A simple script that compares the assistant's output against the `facts` dictionary and the tool results is enough to start. A manual review of 50 steps will usually reveal patterns: for example, the agent may invent file paths when a tool returns an error, or it may assume a user's preference that was never stated.

Another useful metric is context churn: how many tokens of the context change between steps. High churn (large portions of the context being replaced) can indicate that the model is losing continuity. Low churn with high hallucination rate suggests the context is stale or contradictory.

A third metric is tool call error rate. If the agent frequently calls tools with invalid arguments, it may be because the tool definitions are buried or because the context contains outdated tool signatures. Reducing the number of tools in context often lowers this rate.

A concrete way to instrument this: wrap your model call in a function that writes one JSON line per step to a log file, containing the step number, the full message list, the tool calls made, and the raw assistant output. Then compute unsupported-claim rate offline by diffing asserted entities against the logged context. Compare two runs — one with pruning and a facts store, one without — on the same task set. The difference in unsupported claims per step is your signal; absolute numbers from someone else's system are not.

## Failure modes that context engineering cannot fix

Context engineering is not a silver bullet. Some hallucinations come from the model's training data or from ambiguous instructions. If the task itself is underspecified, no amount of context curation will make the model correct. Similarly, if the model is fundamentally too small for the reasoning required, better context will not close the gap.

Another failure mode is over-pruning. If you summarise too aggressively, you may drop a critical constraint that was mentioned early on. The agent then violates that constraint and appears to hallucinate, but the information was simply removed. This is why the state object should be authoritative and never pruned. Summaries are lossy; state is not.

A third failure mode is tool result spoofing. If a tool returns untrusted input (e.g., a web page), that input becomes part of the context and can contain instructions that hijack the agent. This is prompt injection, and context engineering alone does not solve it. You need input sanitisation and privilege separation.

Finally, context engineering adds latency and cost. Summarisation calls and state management require extra tokens and compute. For short-lived agents (fewer than 10 steps), the overhead may not be worth it. The patterns in this article are for agents that run long enough for context to become a liability.

A fourth failure mode is worth naming: contradictory facts in the state store. If two tool results disagree and both are written into `facts`, the model sees a contradiction it cannot resolve and may pick either one. Facts need a write policy — last-write-wins with a timestamp, or a source-of-truth rule per key — not just a dictionary.

## When you should not bother

If your agent completes its task in under five steps, or if the total context never exceeds a few thousand tokens, you can probably skip most of this. The failure modes described here emerge when the context grows large and the loop runs long. For simple retrieval-augmented generation (RAG) or single-turn classification, standard prompt engineering is sufficient.

Also, if you are using a model with a very large context window and strong long-context performance, some of these patterns are less critical. But even then, attention is not uniform, and stale information can still mislead. The cost of a state object is low; it is worth adding even for medium-length agents.

A short decision checklist:

- Loop length under five steps and context under a few thousand tokens? Skip this; use plain prompt engineering.
- Loop runs tens of steps and appends raw tool output? Add structured tool results first — it is the cheapest change.
- Agent repeats facts that were later contradicted? Add a facts store with a write policy and inject it every turn.
- Context is large but hallucination is low? Measure unsupported claims before adding summarisation; you may be paying for nothing.
- Tools are numerous and frequently mis-called? Try tool routing before rewriting prompts.
- Untrusted tool input in context? Fix injection and privilege separation first; context curation will not help.

## Frequently Asked Questions

**How does context engineering differ from prompt engineering?**
Prompt engineering focuses on the wording of instructions. Context engineering focuses on what information is present in the model's context at each step, how it is structured, and what is removed. Prompt engineering is a subset; context engineering includes state management, pruning, and tool routing.

**Why do agents hallucinate more after many steps?**
Because the context accumulates stale, irrelevant, or contradictory information. The model attends to all of it and may retrieve an outdated fact as if it were current. Without explicit state tracking and pruning, the context becomes a noisy log that encourages confabulation.

**Can I just use a larger context window to avoid this?**
A larger window delays the problem but does not solve it. More context means more tokens competing for attention, and the model still lacks a notion of recency or reliability unless you encode it. State tracking and pruning remain necessary for long-running agents.

**What is the simplest change I can make today?**
Add a `facts` dictionary to your agent state and include it in every system prompt. When the model asserts something new, require it to cite the tool result or user message that supports it. If it cannot, treat the assertion as an assumption and do not add it to `facts`.

## The next 30 minutes

Open your agent's main loop file and find where you append tool results to the message history. Add a `state` object (a simple dict is fine) and a `prune` function that keeps only the last N messages plus the system prompt. Then, in your system prompt template, inject the current `facts` from the state. Run one long session and log every assistant message that contains a factual claim, together with the context that was sent at that step. Check whether each claim is backed by a tool result or a user message in that context. That log will tell you more about your agent's reliability than any benchmark.
