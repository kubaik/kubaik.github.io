# Multi-agent DX: LangGraph vs CrewAI gaps

## Why the debugging gap is structural, not a missing feature

Multi-agent systems introduce a new failure mode: non-determinism at the orchestration layer. A single LLM call is already stochastic; chaining several agents with tool access multiplies the state space. Traditional debugging assumes you can reproduce a bug by replaying inputs. With agents, the same input can produce different tool calls, different intermediate reasoning, and different final outputs — even at temperature 0, because model providers generally do not guarantee bitwise determinism across requests.

That means the developer experience challenge is less about adding a `print()` statement and more about capturing enough context to reconstruct *why* a particular branch was taken. Both LangGraph and CrewAI provide tracing hooks, but they capture different things by default. LangGraph, built around an explicit state graph, records node transitions and state deltas when you attach callbacks. CrewAI, built around role-based agents and tasks, records task execution and agent delegation when verbose logging or callbacks are enabled. Neither gives you a full causal chain out of the box; you have to instrument the boundaries yourself.

The gap matters because teams commonly discover it only after deploying to staging. A typical failure mode: an agent silently retries a failed tool call three times, the retries are logged at DEBUG level, and the final output looks plausible but is wrong. Without a trace that links the retry to the final answer, the bug is invisible until a user reports it.

## LangGraph: explicit state, implicit observability

LangGraph's core abstraction is a state graph: nodes are functions that transform a shared state object, edges define control flow, and conditional edges allow branching. This explicitness is a double-edged sword for debugging. On one hand, you can inspect the graph structure statically — you know which nodes *can* run. On the other hand, the framework does not automatically log state transitions unless you add callbacks.

A minimal LangGraph example illustrates the instrumentation burden:

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    next_step: str

def planner(state: AgentState):
    # In a real app, this calls an LLM
    return {"next_step": "research"}

def researcher(state: AgentState):
    # Simulate a tool call
    return {"messages": ["research result"]}

graph = StateGraph(AgentState)
graph.add_node("planner", planner)
graph.add_node("researcher", researcher)
graph.set_entry_point("planner")
graph.add_conditional_edges("planner", lambda s: s["next_step"], {"research": "researcher"})
graph.add_edge("researcher", END)
app = graph.compile()
```

To debug this, you would typically add a callback handler that logs each node's input and output. LangGraph supports callbacks via the `callbacks` parameter in the run config, but the documentation focuses on integration with external tracing tools (LangSmith, OpenTelemetry) rather than built-in local debugging. That is a gap: if you are not using a hosted tracing service, you are writing your own logger. The framework gives you the graph, but not the flight recorder.

One subtlety worth knowing: the `Annotated[list, operator.add]` reducer means the `messages` key accumulates across nodes rather than being overwritten. That is convenient, and it is also how unbounded growth sneaks into long runs. When you inspect state snapshots, check the length of accumulated keys, not just their contents.

## CrewAI: role-based simplicity, opaque execution

CrewAI models agents as roles with goals and backstories, and tasks as units of work assigned to agents. The abstraction is higher-level than LangGraph's: you define a crew, give it a process (sequential or hierarchical), and let it run. This makes the initial setup faster but the debugging experience more opaque. Because the framework decides which agent handles which task (especially in hierarchical mode), the execution path is less predictable.

A minimal CrewAI example:

```python
from crewai import Agent, Task, Crew

researcher = Agent(
    role="Researcher",
    goal="Find recent papers on multi-agent systems",
    backstory="You are a diligent academic researcher.",
    verbose=True
)

writer = Agent(
    role="Writer",
    goal="Summarize research findings",
    backstory="You are a technical writer.",
    verbose=True
)

task1 = Task(description="Search for papers", agent=researcher)
task2 = Task(description="Write summary", agent=writer)

crew = Crew(agents=[researcher, writer], tasks=[task1, task2], verbose=True)
result = crew.kickoff()
```

Setting `verbose=True` prints agent thoughts and actions to stdout, which is helpful for local development but not for production. CrewAI also integrates with external tracing tools, but the built-in logging is unstructured text. There is no built-in way to query "show me all tasks where the researcher agent used the search tool and got an empty result." You would have to parse logs or add custom callbacks.

## Head-to-head: where each framework leaves you guessing

| Axis | LangGraph | CrewAI |
|------|-----------|--------|
| **State visibility** | Explicit state object; you can snapshot and diff it between nodes. | Implicit state; agent context is managed internally. Harder to inspect mid-run. |
| **Control flow tracing** | Graph structure is static and inspectable; conditional edges are visible. | Process (sequential/hierarchical) is configured, but runtime delegation is opaque. |
| **Error propagation** | Node failures raise exceptions unless caught; you can add retry logic per node. | Task failures may be retried by the agent or crew; default behavior is less documented. |
| **Local debugging** | Requires custom callbacks or external tracing; no built-in step-through debugger. | `verbose=True` gives stdout logs; no structured event stream. |
| **Reproducibility** | State graph can be replayed if you capture state snapshots; LLM calls still non-deterministic. | Harder to replay because agent delegation decisions are made at runtime. |
| **Integration with observability** | Callback hooks that map onto OpenTelemetry and LangSmith. | External tracing supported, with fewer granular hooks. |

Both frameworks share a common gap: neither provides a built-in, queryable event log of agent decisions. You get logs, but not a database you can ask questions of. That is the developer experience gap that remains.

## The missing piece: a causal trace of agent decisions

What developers actually need is a trace that answers: *Why did the system do X instead of Y?* For a single LLM call, that is the prompt and the response. For a multi-agent system, it is the chain of delegations, tool calls, and state changes that led to the final output. Neither LangGraph nor CrewAI gives you that out of the box.

A workaround is to instrument your agents with a custom callback that emits structured events to a local store (for example, SQLite or a JSONL file). Here is a minimal example for LangGraph using a callback handler:

```python
from langchain_core.callbacks import BaseCallbackHandler
import json, time

class JsonlLogger(BaseCallbackHandler):
    def __init__(self, path="agent_trace.jsonl"):
        self.path = path
        self.f = open(path, "a")

    def on_chain_start(self, serialized, inputs, **kwargs):
        event = {"ts": time.time(), "event": "chain_start",
                 "name": (serialized or {}).get("name"), "inputs": inputs}
        self.f.write(json.dumps(event, default=str) + "\n")

    def on_chain_end(self, outputs, **kwargs):
        event = {"ts": time.time(), "event": "chain_end", "outputs": outputs}
        self.f.write(json.dumps(event, default=str) + "\n")

# Attach to graph invocation
# app.invoke({"messages": [], "next_step": ""},
#            config={"callbacks": [JsonlLogger()]})
```

Note the `default=str` in the serialiser: callback payloads routinely contain objects that `json.dumps` cannot handle, and a logging callback that throws will either crash the run or be swallowed, depending on the framework. Also note the commented invocation — the callback only sees events for runs it is attached to, so attach it at the outermost invocation, not per node, or you will get nested duplicates.

This is not a complete solution — it does not capture LLM prompts unless you also implement `on_llm_start` — but it demonstrates the kind of plumbing teams end up writing. The gap is that this plumbing should be part of the framework's default developer experience, not an afterthought.

## The edge cases that eat your weekend

The happy-path examples above cover the majority of what gets shipped. The remainder is where multi-agent systems get genuinely nasty, and it is worth naming the specific failures because they recur across teams with almost boring regularity. None of these are exotic; they are the default behavior of LLMs plus tools plus concurrency.

**The infinite tool loop with a clean final answer.** An agent calls a search tool, gets a result it deems insufficient, reformulates the query, calls again, and does this six times before finally synthesizing an answer. The final output looks fine. The token bill does not. This happens because the agent's stop condition is "I have enough information," which is a judgment call the model makes, not a constraint the framework enforces. LangGraph will not stop it unless you add a recursion limit or a counter in your state; CrewAI's `max_iter` on an agent caps iterations, but the default is generous and the failure is silent. The fix is a hard step budget in the state object and a check at every conditional edge.

**Tool schema drift between agents.** Agent A produces a JSON payload that Agent B's tool expects, but the model generating A's output decides to add a field, rename a key, or wrap the whole thing in an extra object. This is one of the most common integration failures in crews, and it is maddening because each agent works in isolation. The failure surfaces only in the handoff. Structured outputs (function calling with a strict schema) reduce it but do not eliminate it — models still occasionally emit `{"result": {...}}` instead of `{...}`. Validate at every boundary, not just at the tool call.

**Context window collapse in long-running crews.** A hierarchical crew with ten tasks accumulates message history across every agent. By task seven, the manager agent is summarizing a history that no longer fits the window, and the framework silently truncates — usually from the front, which is exactly where the original task definition lives. The agent then "forgets" the goal and produces plausible-but-off-target output. This is a structural problem with the role-based model: shared context is convenient until it is not. LangGraph's explicit state lets you prune deliberately; CrewAI's implicit context makes it harder to see what got dropped.

**Partial failure in parallel branches.** LangGraph supports fan-out to parallel nodes, which is good for latency and bad for error handling. One branch raises, the others keep running, and the join node receives a half-populated state. Depending on how you wrote the reducer, this either raises a confusing `KeyError` three nodes downstream or, worse, silently produces a result from incomplete data. You need per-branch error capture and an explicit merge policy — fail-fast, best-effort, or quorum — and you need to choose it before production, not after.

**Retry storms against rate-limited APIs.** An agent hits a 429, the framework's retry logic backs off, but so do four other agents in the same crew, and they all retry in lockstep. Now a transient rate limit has become a sustained one. Neither framework coordinates retries across agents by default. You need a shared token bucket or a queue, and you need to instrument retry counts per tool so you can see the storm forming before it exhausts your quota.

**The stale-tool-cache trap.** Tools that cache results (search, database lookups, HTTP fetches) return stale data to an agent that has already reasoned about the previous version. The agent does not know the data is stale, so it treats a cached empty result as ground truth and loops trying to "find" something that exists. Cache TTLs tuned for human use are wrong for agent use; agents hammer tools in seconds, not minutes. Log cache hits and misses alongside tool calls or you will spend a day chasing a bug that is not in your code.

## Worked example: sizing a step budget

The loop problem above has a simple fix, but choosing the number requires arithmetic rather than intuition. Consider a three-agent pipeline (planner, retriever, writer) where the retriever may re-query up to a bounded number of times.

State the assumptions explicitly:

- Each retriever iteration costs one LLM call for query reformulation plus one tool call.
- The writer runs exactly once, at the end.
- The planner runs exactly once, at the start.
- A per-run hard cap of `N` retriever iterations is enforced in state.

Then the maximum number of LLM calls per run is `1 (planner) + N (retriever reformulations) + 1 (writer) = N + 2`. If the observed average is 1.2 retriever iterations and the worst legitimate case is 3, then `N = 3` gives a maximum of 5 LLM calls per run. Setting `N = 10` does not make the system more capable; it only raises the ceiling on the damage a misbehaving run can do.

To measure rather than guess, instrument the retry counter per run and emit it as an attribute on the run's root span. Then query the trace backend for the 99th percentile of that counter over a week of traffic. If p99 is 2 and the cap is 3, the cap is doing its job without truncating legitimate work. If p99 equals the cap, you are clipping real behavior and need to investigate why the retriever is unsatisfied — usually a tool returning empty results, which brings you back to the stale-cache failure mode.

The same arithmetic applies to token cost. If the average run costs `C` tokens and a loop multiplies that by `k`, the monthly bill scales linearly with the loop rate. Capping `k` caps the tail; it does not reduce the mean unless you also fix the cause. The cap is insurance, not optimization.

## Wiring it into tools you already run

The frameworks do not exist in a vacuum, and the real developer experience question is how they behave once you bolt on the observability and orchestration stack your team already runs. Three integration categories matter most: OpenTelemetry for vendor-neutral tracing, an LLM-observability backend for prompt-and-completion-level views, and a durable execution engine for long-running crews. The pattern below uses LangGraph with an LLM-observability callback handler plus an OTLP span, which is a common combination because it gives you both a queryable trace UI and a vendor-neutral export.

```python
import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from langgraph.graph import StateGraph, END

# OTLP: point at your collector (Tempo, Jaeger, Grafana Agent, etc.)
provider = TracerProvider()
provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint=os.environ["OTLP_ENDPOINT"]))
)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("agents.orchestrator")

def instrumented_node(name, fn):
    def wrapped(state):
        with tracer.start_as_current_span(f"node.{name}") as span:
            span.set_attribute("agent.node", name)
            span.set_attribute("agent.step", state.get("step", 0))
            out = fn(state)
            span.set_attribute("agent.next_step", out.get("next_step", "end"))
            return {**out, "step": state.get("step", 0) + 1}
    return wrapped

# ... build graph as before, wrapping nodes ...
# app.invoke({"messages": [], "next_step": "", "step": 0},
#            config={"callbacks": [your_llm_observability_handler]})
```

Two things to notice. First, the OTLP span and an LLM-observability callback are complementary, not redundant: OTLP gives you infrastructure-grade tracing that slots into whatever backend your platform team runs, while an LLM-aware backend gives you prompt, completion, and token-cost views that generic tracing backends render poorly. Second, wrapping nodes this way gives you the step counter you need for the loop-budget fix from the previous section — the instrumentation and the guardrail are the same code. For CrewAI, the equivalent is passing an observability handler into the crew's callback list (check the version you are on; callback support has changed across releases) and using a step callback to emit OTLP spans per agent step. A durable execution engine can wrap the whole `crew.kickoff()` or `app.invoke()` in a flow so a crashed run resumes from the last completed task instead of re-paying for every LLM call — which, at production token volumes, is the difference between a bad afternoon and a bad quarter.

## How to measure whether instrumentation paid off

Claims about tracing improving diagnosis time are easy to make and hard to verify, so measure them on your own system. Four metrics cover most of the value:

1. **Mean time to root cause.** Log the timestamp of the first alert or user report and the timestamp when the faulty node or tool call is identified. Compare a sample of incidents before and after instrumentation. The comparison is only fair if the incidents are of similar type, so bucket them.
2. **Token cost distribution, not just the mean.** Export per-run token counts as a histogram. The mean hides the tail, and the tail is where loops live. A cap on retries shows up as a hard cutoff in the histogram's right edge; if you see a spike just below the cap, the cap is being hit routinely and is masking a real problem.
3. **Retry counts per tool.** Emit a counter per tool call and per retry. A retry rate climbing over time is an early warning of a rate limit or a schema drift, often days before it produces a bad output.
4. **Replay success rate.** For a sample of runs, attempt to replay the orchestration from captured state. Count how many reproduce the same control-flow path. This is the honest measure of reproducibility: LLM outputs will differ, but the graph should not.

Each of these is a query against the trace store, not a manual exercise. If your instrumentation cannot answer them, it is logging, not observability.

## Who should pick which (and when to pick neither)

Choose LangGraph if you need explicit control over the orchestration graph and plan to integrate with existing observability tooling. The state graph gives you a mental model that maps well to debugging: you can reason about which nodes executed and in what order. The trade-off is that you will write more instrumentation code upfront.

Choose CrewAI if you want faster prototyping and your agents' roles are well-defined. The higher-level abstraction means less boilerplate, but the debugging experience is more opaque. For simple sequential crews, `verbose=True` may suffice; for hierarchical crews with dynamic delegation, you will likely need custom logging.

Pick neither if your system has strict reproducibility requirements or if you need to debug complex failure modes without external services. In that case, consider building a thin orchestration layer on top of raw LLM APIs, where you control every state transition and can log exactly what you need. The frameworks save time on the happy path but can cost more time on the unhappy path.

A decision checklist:

- **Prototype due within days, well-defined roles** → CrewAI. Accept the debugging debt explicitly and schedule the instrumentation work.
- **Debugging a production incident now** → LangGraph, because the explicit graph gives you a place to start.
- **Every decision must be auditable** → neither; build an event log you control.
- **Already using LangChain** → LangGraph integrates more naturally.
- **Team new to agents** → CrewAI's role metaphor is easier to grasp.
- **Long-running jobs that must survive restarts** → either framework, wrapped in a durable execution engine.

## The next concrete step

Open your agent's main orchestration file and add a single structured log line at the entry and exit of each agent or node, including a correlation ID. In LangGraph, wrap your node functions with a decorator that logs `node_name`, `input_state_keys`, and `timestamp` to a JSONL file. In CrewAI, use the step callback to emit similar events. Run your test suite and inspect the log for any node that executes more than once — that is your first clue to a retry loop or a misconfigured edge. This takes about 30 minutes and will surface at least one hidden behavior in your multi-agent system.
===
