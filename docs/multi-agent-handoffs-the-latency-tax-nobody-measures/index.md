# Multi-agent handoffs: the latency tax nobody measures

Teams often end up shipping hidden latency twice — the second time because the first version failed quietly. This walks through the fix and the reasoning, not just the patch. Most write-ups stop exactly where the interesting part starts.

## The conventional wisdom (and why it's incomplete)

Multi-agent systems are sold as the next step in AI architecture: split a complex task across specialised agents, let each one do what it's good at, and orchestrate the results. The pitch is clean, and the demos look impressive. But the standard advice almost always stops at *capability* — what the system can do — and says almost nothing about *cost* — what the system spends to do it. That gap is where the latency tax hides.

When people talk about multi-agent performance, they usually reach for token counts and model choice. "Use a smaller model for routing," they say. "Cache the system prompt." All true, all useful. But the dominant cost in a multi-agent pipeline is not inference. It's the handoff: the serialisation, transport, deserialisation, re-prompting, and re-validation that happens every time control passes from one agent to another. A typical handoff adds 40–120ms of pure overhead before the next model even starts thinking. Chain five agents and you've added 200–600ms of dead time to every request, and that's before you count the extra tokens each agent burns re-reading context it didn't need.

The part that trips people up is that this overhead is invisible in most observability setups. Your APM shows a 2.4s request. It doesn't show that 580ms of that was handoff tax. And that's what this post actually covers.

## What actually happens when you follow the standard advice

The standard advice for building a multi-agent system goes something like this: define a planner agent, a researcher agent, a writer agent, and a critic agent. Wire them together with a framework like LangGraph 0.2 or CrewAI 0.30. Pass the full conversation history between them so each agent has context. Log everything to LangSmith or Langfuse for debugging.

This works. It also quietly accumulates latency in four places that nobody instruments.

**First, the serialisation cost.** Every handoff means converting a Python object (usually a Pydantic model or a dict) to JSON, sending it over HTTP or a message queue, and parsing it back. For a 4,000-token context, that's roughly 16KB of JSON. Serialise, transport, deserialise: 15–40ms depending on payload size and whether you're using `orjson` or the standard library. Multiply by four handoffs and you've spent 60–160ms doing nothing useful.

**Second, the re-prompting cost.** Each agent needs to know what came before. If you're passing full history, a 4-agent pipeline can easily send 12,000 tokens to the final agent when only 2,000 were relevant. At typical input pricing, that's not just latency — it's money. More importantly, longer prompts mean longer time-to-first-token. A 12k-token prompt can add 200–400ms of prefill time versus a 2k-token prompt on the same model.

**Third, the validation cost.** Good multi-agent systems validate outputs between agents. That's a JSON schema check, sometimes a retry loop, sometimes a separate LLM call to grade the output. Each validation is 5–50ms, and each retry is a full agent invocation.

**Fourth, the orchestration cost.** The framework itself — the graph traversal, the state management, the callback hooks — adds overhead. It's usually small (5–15ms per step) but it compounds.

A common failure mode here: a team builds a 5-agent pipeline, tests it on 10 requests, sees 1.8s average latency, and ships it. Under production load with concurrent requests, the same pipeline shows 3.2s p95 because the handoff overhead scales with queue depth and context size. The demo never caught it because the demo never ran at concurrency.

## A different mental model

Stop thinking of agents as workers passing a baton. Start thinking of them as functions in a pipeline where every boundary is a network call with a cost.

The mental model that actually helps: **an agent handoff is a remote procedure call, and you should treat it with the same suspicion you'd treat any RPC in a hot path.** You wouldn't call a microservice five times in a request handler without asking whether those calls could be batched or eliminated. The same discipline applies here.

This reframing changes what you optimise. Instead of asking "which model should each agent use?", you ask:

- Can two agents be merged into one prompt with structured output?
- Can the handoff carry a summary instead of full history?
- Can validation happen in-process instead of as a separate call?
- Can the orchestrator run independent agents in parallel instead of serially?

In most systems I've seen described publicly, the answer to at least two of those is yes, and the savings are larger than any model swap.

Here's the uncomfortable part: the multi-agent architecture is often chosen for *developer* ergonomics, not runtime performance. It's easier to reason about four small prompts than one large one. That's a legitimate reason to start there. It's not a legitimate reason to stay there once you've measured the cost.

## Evidence and examples from real systems

Consider a document-analysis pipeline: extract entities, classify sentiment, summarise, and fact-check. Four agents, serial, full history passed between each.

Here's a simplified version of what the handoff layer looks like in a typical Python implementation:

```python
import orjson
import httpx
from pydantic import BaseModel

class AgentState(BaseModel):
    history: list[dict]
    current_output: str

async def handoff(state: AgentState, next_agent_url: str) -> AgentState:
    # Serialise full history — this is the tax
    payload = orjson.dumps(state.model_dump())
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(next_agent_url, content=payload)
    return AgentState(**orjson.loads(resp.content))
```

At 4,000 tokens of history, `payload` is roughly 16KB. The round trip on a same-region call is 20–35ms. With four handoffs, that's 80–140ms of pure transport. Add prefill time for the growing prompt and you're at 400–700ms of overhead per request before any agent has produced a useful token.

The fix that consistently shows up: pass a *summary* plus the *current task*, not the full history.

```python
class SlimState(BaseModel):
    task: str
    summary: str          # 200 tokens max
    current_output: str

async def slim_handoff(state: SlimState, next_agent_url: str) -> SlimState:
    payload = orjson.dumps(state.model_dump())
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(next_agent_url, content=payload)
    return SlimState(**orjson.loads(resp.content))
```

The payload drops to roughly 1KB. Transport time drops to 8–15ms. Prefill time drops because the prompt is shorter. In a typical 4-agent pipeline, this change alone cuts 180–320ms of the total, which on a 1.5s baseline is a 12–21% reduction — before you touch parallelism or model choice.

Now add parallelism. If entity extraction and sentiment classification don't depend on each other, run them concurrently with `asyncio.gather`. On a pipeline where two of four steps are independent, that saves the full latency of the shorter branch — often 300–500ms.

Combine the two and you're looking at 40–60% total latency reduction on the same hardware, same models, same task quality. The mechanism is not magic; it's removing work that was never necessary.

## The cases where the conventional wisdom IS right

I'm not arguing that multi-agent architectures are wrong. They're the right call in specific situations, and pretending otherwise would be dishonest.

**When agents have genuinely different tool access.** If your researcher agent has web search and your writer agent has a sandboxed code interpreter, keeping them separate is a security boundary, not just an architectural choice. Merging them means the writer inherits search permissions it shouldn't have. That's worth 50ms.

**When you need independent auditability.** In regulated environments, having a separate critic agent whose only job is to validate output is a compliance feature. You want that handoff logged, signed, and immutable. The latency is the cost of the audit trail.

**When agents run on different infrastructure.** If your planner runs on a GPU box and your tool-executor runs on a sandboxed VM, the handoff is a real network boundary you can't eliminate. You can only make it cheaper.

**When the task is genuinely long-running.** For a job that takes 30 seconds end-to-end, 400ms of handoff overhead is 1.3%. Not worth optimising. The tax only matters when your baseline is short and your concurrency is high.

The mistake is applying multi-agent architecture *by default* to tasks that are short, serial, and share tool access. That's where the tax dominates and the benefits don't materialise.

## How to decide which approach fits your situation

Use this table as a first pass. It won't capture every nuance, but it forces the question.

| Signal | Multi-agent is worth it | Single-agent or merged is better |
|---|---|---|
| Tool access differs per step | Yes — security boundary | No — merge prompts |
| Total pipeline latency budget | > 5s | < 2s |
| Steps are independent | Parallelise agents | Merge into one prompt |
| Context size per handoff | < 1,000 tokens | > 4,000 tokens |
| Audit/compliance requirement | Yes — separate agents | No — single trace is fine |
| Concurrency | < 10 req/s | > 50 req/s |
| Model choice differs per step | Yes — specialised models | No — one model handles all |

A practical rule: if you can't name the specific reason two agents are separate — security, audit, infrastructure, or genuinely different models — merge them. The default should be one agent with structured output, not a graph of specialists.

To measure where you actually stand, instrument the handoff layer directly. Add a timer around your serialisation and transport calls, and log the payload size. In a typical pipeline, you'll find that handoffs account for 20–40% of total latency. That number is your optimisation budget. If it's under 10%, stop reading and go work on something else.

## Common objections, and responses

**"Merging agents makes prompts too long and hurts quality."** Sometimes true. But the fix is usually structured output with clear sections, not more agents. A single prompt with explicit `## Task`, `## Context`, `## Output format` sections often outperforms a 3-agent pipeline on both quality and latency. Test it before assuming it won't work.

**"Summarisation loses information the next agent needs."** It can. The mitigation is to summarise *and* keep a retrieval path — store the full history, let the next agent fetch specific spans if needed. That's a tool call, not a handoff, and it's cheaper because it's on-demand.

**"Our framework makes multi-agent easy, so why fight it?"** Because the framework optimises for developer experience, not runtime. Easy to build is not the same as cheap to run. Frameworks like LangGraph 0.2 and CrewAI 0.30 are genuinely useful for prototyping; they're less useful as a production runtime if you haven't measured the handoff cost.

**"Parallelism changes semantics — agents might race."** Only if they share mutable state. Design agents as pure functions over immutable inputs and parallelism is safe. If they share state, you have a different problem that multi-agent didn't cause.

**"We already shipped; ripping out agents is too expensive."** You don't have to rip anything out. Start by changing what the handoff carries — summary instead of full history. That's a one-line change in most codebases and it's where the biggest single win usually lives.

## What the alternative approach would change

If teams adopted the RPC mental model by default, three things would shift.

First, architecture diagrams would show latency budgets on every edge, the same way they show data flow. A handoff labeled "40ms, 16KB" invites a different conversation than an unlabeled arrow.

Second, framework defaults would change. The sensible default for a handoff payload is a summary plus the current task, not full history. Frameworks that default to full history are optimising for debuggability at the cost of runtime — a defensible tradeoff for development, a bad one for production.

Third, observability tools would surface handoff tax as a first-class metric. Right now you have to instrument it yourself. That's a gap in the tooling, and it's why the tax stays hidden.

The broader point: multi-agent systems are a real architectural pattern with real costs. Treating them as free — which is what most tutorials implicitly do — leads to systems that work in demos and disappoint in production. The fix isn't to abandon the pattern. It's to measure the boundaries.

## Summary

Multi-agent handoffs carry a latency tax that most observability setups don't surface: serialisation, transport, re-prompting, and validation overhead that adds 200–600ms to a typical 4-agent pipeline. The conventional advice focuses on model choice and token counts, which misses where the time actually goes. The better mental model is to treat every handoff as an RPC in a hot path — something to eliminate, batch, or slim down before you optimise anything else. Passing summaries instead of full history and running independent agents in parallel are the two changes that consistently deliver 40–60% latency reduction on the same hardware and models. Multi-agent is still the right call when agents have different tool access, different infrastructure, or a compliance requirement for separate audit trails — but it should be a deliberate choice, not a default.

Here's your next step: open your agent orchestration file, find the function that builds the payload for each handoff, and log its serialised size in bytes. If it's over 4KB, replace the full history with a 200-token summary and re-run your latency benchmark. That single change is where the biggest win usually hides.

## Frequently Asked Questions

**How do I measure handoff latency in a multi-agent system?**
Wrap your serialisation and HTTP call in a timer and log both the duration and the payload size in bytes. Do this for every handoff boundary, not just the first one. In most pipelines, you'll find handoffs account for 20–40% of total request latency. Once you have that number, you know whether optimisation is worth the effort.

**Why is my multi-agent pipeline slower than a single agent?**
Because each handoff adds serialisation, transport, and prefill overhead that a single agent doesn't pay. A 4-agent pipeline can easily add 400–700ms of pure overhead before any useful work happens. If your steps share tool access and don't need separate audit trails, merging them into one prompt with structured output is usually faster and cheaper.

**What is the best way to pass context between agents?**
Pass a summary plus the current task, not the full conversation history. A 200-token summary instead of a 4,000-token history cuts payload size by roughly 90% and reduces prefill time significantly. If the next agent needs specific details, give it a retrieval tool so it can fetch them on demand rather than carrying everything upfront.

**When should I use multiple agents instead of one?**
Use multiple agents when they have genuinely different tool access, run on different infrastructure, need separate audit trails for compliance, or require different models for quality reasons. If none of those apply, start with one agent and structured output. You can always split later if a specific boundary proves necessary — but you can't easily un-split a pipeline once it's in production.


---

### About this article

**Written by:** [Kubai Kevin](/about/) — software developer based in Nairobi, Kenya, with 10+ years building production systems in fintech and AI.

**How this article was produced:** This site uses an automated LLM pipeline designed and maintained by the author. Topics are selected from real production experience. Drafts pass automated quality gates (minimum length, uniqueness, concrete metrics, versioned tools, code samples, absence of filler). Individual line-by-line human editing is not performed on every post before publication. Specific numbers, benchmarks and cost figures are illustrative; verify them against current official documentation before production use.

**Corrections:** Report errors via the [contact page](/contact/). Corrections are applied promptly.

**Last generated:** September 2026
