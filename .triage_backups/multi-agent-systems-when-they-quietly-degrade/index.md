# Multi-agent systems: when they quietly degrade

A multi-agent system can pass every smoke test on launch day and still be materially worse three months later, without a single deploy that looks risky. The failure is rarely one bug. It is the accumulation of small, individually reasonable changes: prompts that grow, retries that multiply, context windows that fill with stale agent chatter, and no per-agent visibility into latency or cost.

Vendor documentation typically covers setup and stops there. The failure modes below are the part you have to reason about yourself.

## Why multi-agent degradation is hard to see

Engineers usually arrive from monolithic services or request/response APIs. In those worlds a regression produces a clear signal: p99 latency jumps, error rates spike, a dashboard turns red. Multi-agent systems do not behave that way. They degrade inside what looks like normal variance.

A research agent that answered in 2.1 seconds now takes 3.4 seconds. Still inside the SLO, so nobody pages. Cost per query creeps from $0.04 to $0.11 over six weeks, but the monthly bill moves by a few hundred dollars and gets absorbed. Output quality slips: citations get less precise, summaries get vaguer, but the system still returns something plausible.

The confusion deepens because agent frameworks encourage abstraction. Defining agents, tools, and orchestration in a few dozen lines is the selling point. That abstraction hides the execution graph. When you invoke the compiled graph, you do not see how many model calls fired, how many tokens each consumed, or which agent retried three times because a tool returned malformed JSON. The system is working — until it isn't.

A common failure mode is the extra-verification step added "just to be safe." One additional LLM call per query, at 10,000 queries per day, is 10,000 extra calls. If each consumes roughly 1,000 tokens, that is 10 million extra input tokens per day. At a hypothetical $2.50 per million input tokens, that is about $25 per day, or roughly $750 per month, for a step that may not improve output at all. Nobody notices because the change was one line in a prompt.

## The mental model: a road network, not a highway

A highway has a clear capacity limit. Exceed it and traffic jams immediately. A city road network absorbs extra cars for a while — side streets fill, intersections slow, but traffic still flows. Then one blocked intersection causes gridlock across the whole city.

A multi-agent system behaves the same way. Each agent is an intersection. Each tool call is a road segment. The system carries slack, so small inefficiencies hide in that slack until they don't.

Degradation is a function of three things:

- **Token growth.** Prompts accumulate examples, instructions, and few-shot demonstrations over time.
- **Call multiplication.** Retries, fallbacks, and verification loops add extra model invocations.
- **Context staleness.** Agents pass along conversation history that is no longer relevant, forcing the model to attend to noise.

Consider a typical research pipeline: a planner breaks a question into sub-questions, a retriever fetches documents, a summarizer condenses them, a critic reviews the summary.

Day one, illustrative numbers: planner prompt 400 tokens, retriever returns 3 documents of 500 tokens each, critic prompt 300 tokens. Total input per query is roughly 2,200 tokens.

After three months of "small improvements" — two few-shot examples added to the planner, retrieved documents raised from 3 to 5, a "be thorough" instruction added to the critic — the same query consumes about 4,800 input tokens. That is a 118% increase.

At 10,000 queries per day, daily input goes from 22 million to 48 million tokens. At $2.50 per million input tokens, that is an extra $65 per day, roughly $1,950 per month. Every number here is arithmetic on stated assumptions; substitute your own prices and volumes.

## A worked example of drift

A team builds a research assistant with four agents: Planner, Retriever, Synthesizer, Critic, behind an HTTP wrapper with a database for caching retrieved documents. Initial metrics look fine: median latency 2.8 seconds, p95 4.5 seconds, cost per query $0.038.

Over three months they make incremental changes:

- Add a fallback model for primary-model timeouts.
- Raise the retriever's top-k from 3 to 5.
- Add a self-critique loop where the Critic can send the Synthesizer back for revision, up to two times.
- Expand the prompt with a detailed quality rubric.

None looks risky in isolation. By month three, median latency is 4.9 seconds, p95 is 11.2 seconds, and cost per query is $0.094.

The p95 spike is driven by the self-critique loop: when the Critic rejects a synthesis, the Synthesizer runs again, adding 1,800–2,500 tokens and 1.5–2 seconds. If the loop triggers on roughly 18% of queries, that alone accounts for a large share of the tail.

The fallback adds a second-order effect. When the primary model times out — say on 3% of calls — the system switches to a smaller, faster model that produces lower-quality output. The Critic then rejects more often, which increases latency and cost further. That is a feedback loop, not a linear cost.

Here is a simplified orchestration graph that hides all of it:

```python
# Simplified state-graph orchestration
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

class ResearchState(TypedDict):
    question: str
    sub_questions: List[str]
    documents: List[str]
    synthesis: str
    critique: str
    revision_count: int

def planner_node(state: ResearchState):
    # Prompt has grown to 800 tokens over time
    sub_qs = call_llm(PLANNER_PROMPT, state["question"])
    return {"sub_questions": sub_qs}

def retriever_node(state: ResearchState):
    docs = []
    for q in state["sub_questions"]:
        docs.extend(vector_search(q, top_k=5))  # was top_k=3
    return {"documents": docs}

def synthesizer_node(state: ResearchState):
    synthesis = call_llm(SYNTH_PROMPT, state["documents"])
    return {"synthesis": synthesis}

def critic_node(state: ResearchState):
    critique = call_llm(CRITIC_PROMPT, state["synthesis"])
    return {"critique": critique}

def should_revise(state: ResearchState):
    if "reject" in state["critique"].lower() and state["revision_count"] < 2:
        return "synthesizer"
    return END

graph = StateGraph(ResearchState)
graph.add_node("planner", planner_node)
graph.add_node("retriever", retriever_node)
graph.add_node("synthesizer", synthesizer_node)
graph.add_node("critic", critic_node)
graph.set_entry_point("planner")
graph.add_edge("planner", "retriever")
graph.add_edge("retriever", "synthesizer")
graph.add_edge("synthesizer", "critic")
graph.add_conditional_edges("critic", should_revise)
app = graph.compile()
```

Note that `revision_count` is declared in the state and read in `should_revise`, but nothing in this sketch ever increments it. In a real graph, the Synthesizer node must return `{"revision_count": state["revision_count"] + 1}` on a revision pass, or the loop will never terminate at the intended bound.

The code exposes no token counts, call counts, or per-node latency. To diagnose drift, instrument each call:

```python
import time
import logging
from functools import wraps

logger = logging.getLogger("agent_metrics")

def instrument_llm(func):
    @wraps(func)
    def wrapper(prompt, input_text, *args, **kwargs):
        start = time.perf_counter()
        response = func(prompt, input_text, *args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info({
            "event": "llm_call",
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "latency_ms": round(elapsed_ms, 2),
            "model": response.model,
            "node": kwargs.get("node_name", "unknown"),
        })
        return response
    return wrapper

@instrument_llm
def call_llm(prompt, input_text, node_name=None):
    return openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": prompt},
                  {"role": "user", "content": input_text}],
    )
```

With this in place, aggregate logs by `node` and sort by summed tokens. If one node accounts for a disproportionate share — the Critic is a frequent culprit because it runs on every query and again after each revision — that is the node to attack first.

## How to measure drift without inventing numbers

No vendor will tell you your token growth rate. You have to derive it from your own traffic. The instrumentation above gives you the raw material. Then:

1. **Log per call, not per query.** Emit one structured record per LLM invocation: node name, model, prompt tokens, completion tokens, latency, timestamp, and a query ID so you can group.
2. **Compute per-query aggregates.** For each query ID, sum input tokens, sum calls, and take the max node latency. Store these as a time series.
3. **Compare against a rolling baseline.** A 7-day rolling median per node is usually enough. Alert when a metric exceeds the baseline by a threshold you choose — 15% is a reasonable starting point, not a law.
4. **Attribute the change.** When a metric moves, check the diff since the baseline window. Prompt edits, top-k changes, retry counts, and model version bumps are the usual suspects.
5. **Check provider-side variation.** Model latency and throughput change over time on the provider's side. Comparing your node latency to your own baseline, not to an absolute target, is what makes this detectable.

If you want a single command to start with, log to stdout as JSON and pipe through a tool that can group and aggregate — anything that can do `group by node, date` over your log stream will work.

## Misconceptions worth correcting

**"If it works, it's fine."** A multi-agent system can work while degrading. Output stays plausible, so nobody complains, while cost and latency trend up. Track tokens per query, calls per query, and p95 latency per node — not just success rate.

**"More agents means better results."** A Critic can improve quality, but it adds at least one call per query. If it rejects 20% of syntheses and triggers a revision, that is roughly 1.2 extra calls per query on average. At 10,000 queries per day and 1,500 tokens per call, that is 18 million extra tokens daily. Whether the quality gain justifies it is an empirical question, and you can only answer it with an eval set.

**"Caching will solve it."** Caching helps for repeated queries; research queries are often unique. A semantic cache can help, but introduces stale answers, embedding drift, and invalidation complexity. Do not assume a high hit rate — measure it.

**"A bigger model will fix it."** If token count has doubled, moving to a model with a higher per-token price multiplies the already-inflated cost. The problem is token growth, not model capability.

## Budgets and enforcement

Once per-node instrumentation exists, treat the system as a cost and latency budget. Define, per query: a maximum input-token count, a maximum number of model calls, and a p95 latency target. Then enforce it. If the planner wants another sub-question, it has to fit. If the critic wants a revision, it has to account for the extra tokens.

A practical enforcement mechanism is to count tokens before each call using the model's own tokenizer, then truncate context or skip the call when the remaining budget is exhausted. This is analogous to a database statement timeout: you set a ceiling and let the system degrade gracefully rather than blow through it.

A second technique is tiered models: a smaller, cheaper model for routing, classification, and critique; a larger model for synthesis. Whether this preserves quality is measurable with a small eval set — run it before and after, and compare on the same inputs.

A third is a scheduled drift check: a daily job that compares today's per-node token and latency metrics against a rolling baseline and alerts on a threshold breach. This catches the slow slide before it becomes a rewrite.

## Quick reference

The ranges below are illustrative starting points, not vendor-documented limits. Calibrate them against your own baseline.

| Metric | What to measure | Illustrative range | Warning sign |
|--------|----------------|--------------------|--------------|
| Tokens per query | Input + output tokens summed per query | 2,000–4,000 | >6,000 or 20% week-over-week increase |
| LLM calls per query | Model invocations per query | 2–4 | >6, or increase without quality gain |
| p95 latency per node | 95th percentile per agent node | <2s per node | Any node >3s or 30% increase |
| Cost per query | Total token cost per query | <$0.05 | >$0.10 or 25% increase |
| Revision rate | Share of queries triggering a revision | <10% | >20% |
| Cache hit rate | Share of queries served from cache | >20% | <10% |
| Error rate per node | Share of calls that fail or time out | <1% | >3% |

## FAQ

**How do I know if my multi-agent system is degrading?**
Look at trends, not absolute values. Track tokens per query, calls per query, and p95 latency per node over time. If any rises by more than a threshold you have chosen — 15% week-over-week is a common starting point — without a corresponding quality improvement, you are degrading. One dashboard showing those three metrics per node catches most cases.

**Why does it get slower when nothing changed?**
Something usually did change: a prompt, a retry count, a top-k value, a model version. These land in separate pull requests and look harmless alone. Providers also change model serving over time, so latency can move without any change on your side. Per-node latency compared against a rolling baseline is what makes this visible.

**What is the biggest cost driver?**
Input tokens from accumulated context. Agents often pass full conversation history or every retrieved document to every call. Summarizing or truncating context between agents — passing a short summary of each document instead of the whole document — is a common and effective reduction.

**Should some agents use a smaller model?**
If the task is classification, routing, or simple extraction, often yes. The trade-off is measurable: run a small eval set through both configurations and compare quality and cost. Do not assume the cheaper model is good enough; verify it.

**Does adding a critic agent pay for itself?**
Only if the quality gain exceeds the added calls and tokens. Instrument the critic's rejection rate and the tokens consumed by revisions, then compare against an eval score with and without it.

## What to do in the next 30 minutes

Open your orchestration code and wrap your model call function so it logs `prompt_tokens`, `completion_tokens`, `latency_ms`, `model`, and a `node_name` tag per invocation. Run it against an hour of traffic or a replay of recent requests, group the logs by node, and sort by summed tokens. The node at the top of that list is where your next hour of work should go.
