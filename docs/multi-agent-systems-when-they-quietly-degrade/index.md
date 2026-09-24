# Multi-agent systems: when they quietly degrade

The same our beautiful mistake shows up across production codebases often enough to be a pattern, not bad luck. Here's the root cause, not just the symptom. The vendor docs cover the setup and go quiet on the failure modes.

## The one-paragraph version (read this first)

A multi-agent research system that looks healthy on day one can drift into a slow, expensive mess over three months. The usual cause is not a single bug — it's the accumulation of small, unmeasured changes: prompts that grow by a few tokens, retries that multiply, context windows that fill with stale agent chatter, and a lack of observability into per-agent latency and cost. This post explains why degradation is gradual, what to instrument [first, and how](/multi-agent-systems-break-first/) to build a mental model that catches the slide before it becomes a rewrite. The part that trips people up is treating a multi-agent system like a monolith — you can't fix what you can't see per agent, and that's what this post actually covers.

## Why this concept confuses people

Most engineers come to multi-agent systems with experience in monolithic services or simple request/response APIs. In those worlds, a performance regression usually shows up as a clear signal: p99 latency jumps, error rates spike, a dashboard goes red. Multi-agent systems don't behave that way. They degrade in ways that look like normal variance. A research agent that used to answer in 2.1 seconds now takes 3.4 seconds — still within the SLO, so nobody pages. The cost per query creeps from $0.04 to $0.11 over six weeks, but the monthly bill only jumps by a few hundred dollars, so it gets absorbed. The output quality drops subtly: citations get less precise, summaries get vaguer, but the system still returns something plausible.

The confusion deepens because multi-agent frameworks encourage abstraction. LangGraph 0.2, CrewAI 0.30, and AutoGen 0.2 all let you define agents, tools, and orchestration in a few dozen lines. That abstraction hides the actual execution graph. When you call `graph.invoke(state)`, you don't see how many LLM calls fired, how many tokens each one consumed, or which agent retried three times because a tool returned malformed JSON. The system is working — until it isn't.

A common failure mode here is the 'helpful agent' anti-pattern: a planner agent that decides to add an extra verification step 'just to be safe.' That step adds one LLM call per query. At 10,000 queries per day, that's 10,000 extra calls, each consuming 800–1,200 tokens. Multiply by a model like GPT-4o at $2.50 per million input tokens, and you've added roughly $25–$30 per day — about $750–$900 per month — for a step that may not improve output at all. Nobody notices because the change was one line in a prompt.

## The mental model that makes it click

Think of [a multi-agent system](/built-a-multi-agent-system-without-langgraph/) like a city's road network, not a single highway. A highway has a clear capacity limit: exceed it and traffic jams immediately. A city network absorbs extra cars for a while — side streets fill up, intersections get slower, but traffic still flows. Then one day a single blocked intersection causes gridlock across the whole city. Multi-agent systems behave the same way. Each agent is an intersection. Each tool call is a road segment. The system has slack, so small inefficiencies hide in that slack until they don't.

The key insight is that degradation is a function of three things: token growth, call multiplication, and context staleness. Token growth happens when prompts accumulate examples, instructions, or few-shot demonstrations over time. Call multiplication happens when retries, fallbacks, or verification loops add extra LLM invocations. Context staleness happens when agents pass along conversation history that's no longer relevant, forcing the model to attend to noise.

To make this concrete, consider a typical research pipeline: a planner agent breaks a question into sub-questions, a retriever agent fetches documents, a summarizer agent condenses them, and a critic agent reviews the summary. On day one, the planner prompt is 400 tokens, the retriever returns 3 documents of 500 tokens each, and the critic prompt is 300 tokens. Total input tokens per query: roughly 2,200. After three months of 'small improvements' — adding two few-shot examples to the planner, increasing retrieved documents to 5, adding a 'be thorough' instruction to the critic — the same query consumes 4,800 input tokens. That's a 118% increase. At 10,000 queries per day, you've gone from 22 million to 48 million input tokens daily. On a model priced at $2.50 per million input tokens, that's an extra $65 per day, or about $1,950 per month.

## A concrete worked example

Let's walk through a realistic scenario. A team builds a research assistant using LangGraph 0.2 and OpenAI's GPT-4o. The system has four agents: Planner, Retriever, Synthesizer, and Critic. They deploy it with a simple FastAPI wrapper and a Postgres 16 database for caching retrieved documents. The initial metrics look good: median latency 2.8 seconds, p95 4.5 seconds, cost per query $0.038.

Over three months, the team makes incremental changes. They add a fallback model (GPT-4o-mini) for when the primary model times out. They increase the retriever's top-k from 3 to 5. They add a 'self-critique' loop where the Critic can send the Synthesizer back for revision up to two times. They also upgrade the prompt to include a detailed rubric for quality. None of these changes seem risky in isolation.

By month three, the metrics tell a different story. Median latency is 4.9 seconds, p95 is 11.2 seconds, and cost per query is $0.094. The p95 spike is driven by the self-critique loop: when the Critic rejects a synthesis, the Synthesizer runs again, adding 1,800–2,500 tokens and 1.5–2 seconds. On about 18% of queries, the loop triggers at least once. The fallback model adds another wrinkle: when GPT-4o times out (which happens on 3% of calls), the system switches to GPT-4o-mini, which is faster but produces lower-quality output, causing the Critic to reject more often — a feedback loop that increases both latency and cost.

Here's a simplified version of the orchestration code that hides these issues:

```python
# LangGraph 0.2 orchestration — simplified
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

The code doesn't show token counts, LLM call counts, or per-node latency. To diagnose degradation, you need to instrument each node. A minimal approach is to wrap `call_llm` with a decorator that logs tokens, latency, and model used:

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
            "node": kwargs.get("node_name", "unknown")
        })
        return response
    return wrapper

@instrument_llm
def call_llm(prompt, input_text, node_name=None):
    return openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": prompt},
                  {"role": "user", "content": input_text}]
    )
```

With this instrumentation, you'd see that the Critic node accounts for 42% of total tokens and 55% of p95 latency. That's the signal you need to decide whether the self-critique loop is worth its cost.

## How this connects to things you already know

If you've tuned a database, you already understand the pattern. A query that runs in 5 ms on day one can degrade to 200 ms after six months because of missing indexes, table bloat, or parameter sniffing. The fix isn't to rewrite the query — it's to look at the execution plan. Multi-agent systems need the same discipline: an execution plan per query, not just an aggregate dashboard.

Similarly, if you've managed a Node.js service, you know that memory leaks rarely announce themselves. They show up as a slow climb in RSS until the process hits the heap limit and crashes. Multi-agent token growth is the same shape: a slow climb in tokens per query until the cost or latency crosses a threshold. The difference is that Node.js has `process.memoryUsage()` and heap snapshots. Multi-agent frameworks often don't expose per-node token counts out of the box. You have to build that visibility.

Another parallel is connection pooling. In Postgres, a pool of 10 connections can handle 1,000 queries per second if each query takes 10 ms. But if a query takes 100 ms, the same pool only handles 100 queries per second. Multi-agent systems have a similar concurrency limit: each agent node is a 'connection' to the LLM provider. If the provider rate-limits you (e.g., 500 requests per minute on a standard tier), a self-critique loop that adds 20% more calls can push you over the limit, causing retries and timeouts that further degrade latency.

## Common misconceptions, corrected

**Misconception 1: 'If it works, it's fine.'** Multi-agent systems can work while degrading. The output is still plausible, so nobody complains. But the cost and latency curves are trending up. You need to track tokens per query, LLM calls per query, and p95 latency per node, not just overall success rate.

**Misconception 2: 'More agents means better results.'** Adding a Critic agent can improve quality, but it also adds at least one LLM call per query. If the Critic rejects 20% of syntheses and triggers a revision, that's 1.2 extra calls per query on average. At 10,000 queries per day, that's 12,000 extra calls — and if each call consumes 1,500 tokens, that's 18 million extra tokens daily. On GPT-4o, that's $45 per day, or $1,350 per month. The quality gain needs to justify that cost.

**Misconception 3: 'Caching will solve it.'** Caching helps for repeated queries, but research queries are often unique. A semantic cache (e.g., using embeddings to find similar past queries) can help, but it introduces its own failure modes: stale answers, embedding drift, and cache invalidation complexity. In practice, a cache hit rate above 30% for research queries is optimistic.

**Misconception 4: 'We can fix it with a bigger model.'** Upgrading from GPT-4o-mini to GPT-4o increases cost per token by roughly 16x (from $0.15 to $2.50 per million input tokens). If your token count has already grown 2x, you're now paying 32x the original cost. The problem is token growth, not model capability.

## The advanced version (once the basics are solid)

Once you have per-node instrumentation, the next step is to treat the multi-agent system as a cost and latency budget. Define a budget per query: e.g., max 5,000 input tokens, max 3 LLM calls, max 6 seconds p95 latency. Then enforce it. If the Planner wants to add a sub-question, it must fit within the budget. If the Critic wants to trigger a revision, it must account for the extra tokens.

A practical way to enforce budgets is to use a token-counting library like `tiktoken` (for OpenAI models) or the model's own tokenizer. Before each LLM call, estimate the token count. If it exceeds the remaining budget, either truncate the context or skip the call. This is similar to how you'd enforce a query timeout in Postgres: `SET statement_timeout = '5s'`.

Another advanced technique is to use a smaller model for routing and a larger model for synthesis. For example, use GPT-4o-mini for the Planner and Critic (which are classification-like tasks) and GPT-4o for the Synthesizer (which needs high-quality generation). This can cut costs by 40–60% without hurting quality, because the Planner and Critic don't need deep reasoning.

Finally, consider adding a 'degradation detector' — a cron job that runs daily and compares today's token-per-query and latency-per-node metrics against a 7-day rolling baseline. If any metric increases by more than 15%, alert. This catches the slow slide before it becomes a crisis.

## Quick reference

| Metric | What to measure | Typical healthy range | Warning sign |
|--------|----------------|----------------------|--------------|
| Tokens per query | Input + output tokens per LLM call, summed per query | 2,000–4,000 | >6,000 or 20% week-over-week increase |
| LLM calls per query | Count of model invocations per query | 2–4 | >6 or increase without quality gain |
| p95 latency per node | 95th percentile latency for each agent node | <2s per node | Any node >3s or 30% increase |
| Cost per query | Total token cost per query | <$0.05 | >$0.10 or 25% increase |
| Revision rate | % of queries triggering a revision loop | <10% | >20% |
| Cache hit rate | % of queries served from cache | >20% | <10% |
| Error rate per node | % of calls that fail or timeout | <1% | >3% |

## Frequently Asked Questions

**How do I know if my multi-agent system is degrading?**
Look for trends, not absolute values. Track tokens per query, LLM calls per query, and p95 latency per node over time. If any of these increase by more than 15% week-over-week without a corresponding quality improvement, you're degrading. A single dashboard that shows these three metrics per node is enough to catch most cases.

**Why does my multi-agent system get slower over time even though I didn't change anything?**
You probably did change something — a prompt, a retry count, a top-k value, or a model version. These changes are often made in separate PRs and don't look risky in isolation. Also, LLM providers update models silently; a model that was 10% faster last month may be 5% slower this month due to infrastructure changes. Instrument per-node latency and compare against a rolling baseline.

**What is the biggest cost driver in multi-agent research systems?**
Input tokens from accumulated context. Agents often pass full conversation history or all retrieved documents to every LLM call. A common fix is to summarize or truncate context between agents. For example, instead of passing all 5 retrieved documents to the Synthesizer, pass a 200-token summary of each. This can cut input tokens by 50–70%.

**Should I use a smaller model for some agents?**
Yes, if the task is classification, routing, or simple extraction. GPT-4o-mini costs about $0.15 per million input tokens versus $2.50 for GPT-4o — a 16x difference. Using the smaller model for the Planner and Critic while keeping the larger model for the Synthesizer often preserves quality and cuts costs significantly. Measure quality with a small eval set before and after.

## Further reading worth your time

- LangGraph documentation on persistence and streaming: https://langchain-ai.github.io/langgraph/
- OpenAI's tokenizer library, tiktoken: https://github.com/openai/tiktoken
- The original ReAct paper (Yao et al., 2026), which describes the reasoning-acting loop that many agent systems use: https://arxiv.org/abs/2210.03629
- Postgres `pg_stat_statements` documentation, for a masterclass in per-query instrumentation: https://www.postgresql.org/docs/current/pgstatstatements.html

**Your next step:** Open your multi-agent orchestration code and add a decorator that logs `prompt_tokens`, `completion_tokens`, and `latency_ms` for every LLM call, tagged with the node name. Run it for one hour of production traffic, then sort the logs by node and token count. You'll likely find that one node accounts for more than 40% of your tokens — that's the one to fix first.


---

### About this article

**Written by:** [Kubai Kevin](/about/) — software developer based in Nairobi, Kenya, with 10+ years building production systems in fintech and AI.

**How this article was produced:** This site uses an automated LLM pipeline designed and maintained by the author. Topics are selected from real production experience. Drafts pass automated quality gates (minimum length, uniqueness, concrete metrics, versioned tools, code samples, absence of filler). Individual line-by-line human editing is not performed on every post before publication. Specific numbers, benchmarks and cost figures are illustrative; verify them against current official documentation before production use.

**Corrections:** Report errors via the [contact page](/contact/). Corrections are applied promptly.

**Last generated:** September 2026
