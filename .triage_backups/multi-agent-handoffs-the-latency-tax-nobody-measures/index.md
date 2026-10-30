# Multi-agent handoffs: the latency tax nobody measures

Multi-agent systems get pitched on capability: split a complex task across specialised agents, let each do what it is good at, orchestrate the results. The demos look impressive. The standard advice, however, almost always stops at what the system can do and says little about what it spends to do it. That gap is where the latency tax hides.

## The conventional wisdom and where it stops

When teams discuss multi-agent performance, the conversation usually turns to token counts and model choice: use a smaller model for routing, cache the system prompt, trim the tool list. All of that is useful. But the dominant cost in many multi-agent pipelines is not inference. It is the handoff — the serialisation, transport, deserialisation, re-prompting and re-validation that happens every time control passes from one agent to another.

The overhead is invisible in most observability setups. An APM trace shows a 2.4-second request. It does not break out the portion that was handoff tax. Without that breakdown, optimisation effort goes to the wrong place: teams tune prompts and swap models while the structural cost sits untouched.

## What a handoff actually costs

A handoff is a remote procedure call with a model behind it. Four cost centres accumulate, and most of them are not instrumented by default.

**Serialisation.** Every handoff converts an in-memory object (typically a Pydantic model or a dict) to JSON, sends it, and parses it back. The cost scales with payload size and with the encoder used. `orjson` is meaningfully faster than the standard library `json` module on large payloads, but the difference is small next to what the payload size itself does to the total.

**Transport.** A same-region HTTP round trip has a floor set by network and TLS behaviour. Loopback and Unix sockets are cheaper than HTTP over TCP, which is cheaper than a cross-region call. This floor cannot be optimised away, only avoided by not making the call.

**Re-prompting.** Each agent needs to know what came before. If full history is passed, the final agent in a pipeline receives far more tokens than it needs. Longer prompts mean longer prefill, which means longer time-to-first-token. This is the largest of the four costs in most pipelines, and the one most often overlooked because it shows up as "model latency" rather than "handoff latency".

**Validation.** Well-built systems validate outputs between agents: a JSON schema check, sometimes a retry loop, sometimes a separate grading call. Each validation adds time, and each retry is a full agent invocation.

A fifth contributor is the orchestration layer itself — graph traversal, state management, callback hooks. It is usually small per step but compounds across a long pipeline.

The failure mode to watch for: a pipeline tested on a handful of sequential requests looks acceptable, then degrades under concurrent load. Handoff overhead scales with queue depth and context size, so p95 latency under real traffic can be substantially worse than the sequential test suggested. Demos rarely run at concurrency, which is why the problem surfaces after launch.

## A better mental model

Stop thinking of agents as workers passing a baton. Think of them as functions in a pipeline where every boundary is a network call with a cost. An agent handoff is an RPC in a hot path, and it deserves the same suspicion any RPC in a hot path would get. Nobody calls a microservice five times inside a request handler without asking whether those calls could be batched or removed.

This reframing changes the questions worth asking:

- Can two agents be merged into one prompt with structured output?
- Can the handoff carry a summary instead of full history?
- Can validation happen in-process rather than as a separate call?
- Can independent agents run in parallel instead of serially?

There is an uncomfortable truth underneath the architecture choice. Multi-agent designs are often selected for developer ergonomics, not runtime performance. Four small prompts are easier to reason about, test and iterate on than one large one. That is a legitimate reason to start there. It is not a legitimate reason to stay there once the cost has been measured.

## A worked example

Consider a document-analysis pipeline: extract entities, classify sentiment, summarise, fact-check. Four agents, serial, full history passed between each. The numbers below are illustrative, chosen to make the arithmetic visible; substitute your own measurements.

Assume a 4,000-token history. A rough rule of thumb is that one token is about four bytes of JSON-encoded text, so the payload is roughly 16KB. Assume a same-region HTTP round trip of 25ms and a prefill rate that adds measurable time per 1,000 input tokens. Four handoffs at 25ms each is 100ms of pure transport. Now add prefill: if each successive agent receives the accumulated history, the last agent sees roughly four times the context of the first, and the prefill cost grows with it. On a serial pipeline where each agent also generates output, the handoff-attributable share of end-to-end latency commonly lands in the range that makes the pipeline feel sluggish relative to a single well-structured prompt.

The fix that shows up repeatedly is to pass a summary plus the current task, not the full history.

```python
import orjson
import httpx
from pydantic import BaseModel


class SlimState(BaseModel):
    task: str
    summary: str          # keep this bounded, e.g. a few hundred tokens
    current_output: str


async def slim_handoff(state: SlimState, next_agent_url: str) -> SlimState:
    payload = orjson.dumps(state.model_dump())
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(next_agent_url, content=payload)
    return SlimState(**orjson.loads(resp.content))
```

The payload shrinks by roughly an order of magnitude, transport time falls proportionally to payload size, and prefill time falls because the prompt is shorter. The mechanism is not clever. It is the removal of work that was never necessary.

The same pattern applies to parallelism. If entity extraction and sentiment classification do not depend on each other, `asyncio.gather` runs them concurrently and the pipeline pays the latency of the slower branch rather than the sum of both.

```python
import asyncio


async def analyse(document: str) -> dict:
    entities, sentiment = await asyncio.gather(
        extract_entities(document),
        classify_sentiment(document),
    )
    return {"entities": entities, "sentiment": sentiment}
```

Both changes preserve task quality because neither removes information the downstream agent actually needs. They remove information it was carrying by default.

## How to measure the tax

The reason this cost stays hidden is that it is not a metric any framework surfaces by default. Instrument it directly.

**Log payload size per handoff.** In the function that builds each handoff payload, record `len(payload)` in bytes alongside the agent name and step index. This one line turns an invisible cost into a number.

**Time the boundary, not the agent.** Wrap the serialisation and the HTTP call separately. Serialisation time and transport time have different fixes: serialisation responds to smaller payloads and a faster encoder, transport responds to co-location or fewer calls.

**Compare prompt sizes across steps.** Log input token counts at each agent. If the count grows monotonically down the pipeline, full history is being carried and summarisation is the highest-leverage change available.

**Measure at concurrency.** Run the benchmark with realistic parallel request counts, not sequentially. Report p50 and p95. The gap between them is where the handoff tax lives.

**Establish a baseline before changing anything.** Record end-to-end latency, per-step latency, payload sizes and token counts on the current pipeline. Without the baseline, no improvement claim is verifiable.

The resulting number — handoff time as a share of total request latency — is the optimisation budget. If it is small, the pipeline has a different problem and effort belongs elsewhere.

## When multi-agent is the right call

Multi-agent architectures are not wrong. They are the right call in specific situations, and treating them as always-wrong would be as lazy as treating them as always-right.

**Genuinely different tool access.** If a research agent has web search and a writer agent has a sandboxed code interpreter, keeping them separate is a security boundary, not just an architectural preference. Merging them means the writer inherits permissions it should not have. That isolation is worth the handoff cost.

**Independent auditability.** In regulated environments, a separate critic agent whose only job is to validate output is a compliance feature. The handoff needs to be logged, attributable and immutable. The latency is the price of the audit trail.

**Different infrastructure.** If a planner runs on a GPU host and a tool executor runs on a sandboxed VM, the handoff is a real network boundary that cannot be eliminated, only made cheaper.

**Genuinely long-running tasks.** For a job that takes tens of seconds end to end, handoff overhead is a rounding error. The tax only matters when the baseline is short and concurrency is high.

The mistake is applying multi-agent architecture by default to tasks that are short, serial and share tool access. That is where the tax dominates and the benefits do not materialise.

## Decision checklist

| Signal | Separate agents | Merge or single agent |
|---|---|---|
| Tool access differs per step | Yes — security boundary | No — merge prompts |
| Audit or compliance requirement | Yes — separate traces | No — one trace is fine |
| Infrastructure differs per step | Yes — real network boundary | No — same process |
| Steps are independent | Run in parallel | Merge into one prompt |
| Context carried per handoff | Bounded summary | Full history |
| Latency budget | Generous | Tight |
| Concurrency | Low | High |

A practical rule: if you cannot name the specific reason two agents are separate — security, audit, infrastructure, or a genuinely different model — merge them. The default should be one agent with structured output, not a graph of specialists.

## Common objections

**"Merging agents makes prompts too long and hurts quality."** Sometimes true. But the usual fix is structured output with clear sections, not more agents. A single prompt with explicit task, context and output-format sections often matches or beats a three-agent pipeline on quality while costing less. Test it rather than assuming.

**"Summarisation loses information the next agent needs."** It can. The mitigation is to summarise and keep a retrieval path: store the full history, let the next agent fetch specific spans on demand. That is a tool call, not a handoff, and it is cheaper because it is paid only when needed.

**"The framework makes multi-agent easy, so why fight it?"** Because the framework optimises for developer experience, not runtime. Easy to build is not the same as cheap to run. Graph-based orchestration libraries are genuinely useful for prototyping; they are less useful as a production runtime if the handoff cost has never been measured.

**"Parallelism changes semantics — agents might race."** Only if they share mutable state. Design agents as pure functions over immutable inputs and parallelism is safe. If they share state, that is a separate problem the multi-agent design did not cause.

**"We already shipped; ripping out agents is too expensive."** Nothing has to be ripped out. Start by changing what the handoff carries — summary instead of full history. In most codebases that is a small change in one function, and it is where the largest single win usually lives.

## What better defaults would look like

If the RPC mental model were the default, three things would change.

Architecture diagrams would show latency budgets on every edge the way they show data flow. A handoff labelled with a duration and a payload size invites a different conversation than an unlabelled arrow.

Framework defaults would change. The sensible default for a handoff payload is a summary plus the current task, not full history. Frameworks that default to full history are optimising for debuggability at the cost of runtime — a defensible tradeoff during development, a poor one in production.

Observability tools would surface handoff tax as a first-class metric. Today it has to be instrumented by hand, which is precisely why it stays hidden.

The broader point is that multi-agent systems are a real architectural pattern with real costs. Treating them as free — which is what most tutorials implicitly do — produces systems that work in demos and disappoint in production. The fix is not to abandon the pattern. It is to measure the boundaries.

## Next step

Open the function that builds the payload for each handoff in your pipeline. Add one line that logs the serialised size in bytes and the agent name. Run your existing latency benchmark, sequentially and at realistic concurrency, and record the numbers. If any payload is larger than a few kilobytes, replace the full history with a bounded summary and re-run the same benchmark. That single change is where the largest win usually hides, and it takes less than thirty minutes to find out whether it applies to you.
