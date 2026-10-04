# Debug AI agent failures differently

## Why deterministic runbooks fail on agents

Most incident runbooks assume a system with known inputs and predictable outputs. That assumption breaks when an LLM sits in the call stack. A deterministic service either returns a value or raises an error; an agent can return a plausible, well-formed, entirely fabricated answer with a 200 OK status and normal latency. Nothing in a standard log line distinguishes that from a correct response.

The practical consequence is that debugging effort shifts. With a microservice, the question is usually "which component failed?" With an agent, the question is often "why did this component choose this output?" Those are different investigative problems, and they need different instrumentation.

This article compares two approaches:

- **Structured log analysis with enriched metadata** — treat the agent like any other service, emit structured events, query them.
- **Agent behavior traces with causal attribution** — capture the decision path: thoughts, tool calls, tool outputs, confidence deltas, and final answers.

They are not mutually exclusive, and the recommendation at the end is a hybrid. But the tradeoffs are real, and choosing badly wastes either money or incident time.

| Dimension | Structured log analysis | Agent behavior traces |
|---|---|---|
| Primary use case | Debugging known error classes | Debugging unknown or behavioral failures |
| Data captured | Logs, metrics, spans, status codes | Prompts, thoughts, tool I/O, confidence deltas |
| Typical root causes found | Timeouts, rate limits, 5xx, bad deploys | Hallucination, prompt drift, tool-output masking |
| Instrumentation effort | Low (middleware + correlation IDs) | Moderate to high (trace schema + replay env) |
| Query model | Log search / SQL-like | Trace tree navigation + replay |
| Main limitation | Cannot see internal reasoning | Higher storage cost, schema upkeep |

## Option A: structured logs with enriched metadata

Structured logging treats the agent as an ordinary service. Every prompt, tool call, and response becomes a structured event with consistent fields, shipped to whatever log store the team already runs.

A workable field set:

- `trace_id`, `prompt_id`, `agent_id`, `session_id`, `user_id`
- `model`, `temperature`, `max_tokens`
- `tool_calls` (name, arguments, duration), `tool_outputs` (status, payload size)
- `confidence_score` (if the agent emits one), `latency_ms`, `cost_usd`, `status`

The correlation ID is the load-bearing field. Without it, reconstructing one agent run means stitching log lines by timestamp and guessing. With it, a single filter returns the whole run.

A minimal FastAPI wrapper that attaches a trace ID and emits spans:

```python
import uuid
from fastapi import FastAPI, Request
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

app = FastAPI()

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("ai-agent")

@app.post("/agent")
async def agent_endpoint(request: Request, payload: dict):
    trace_id = str(uuid.uuid4())
    with tracer.start_as_current_span("agent_call") as span:
        span.set_attribute("agent.trace_id", trace_id)
        span.set_attribute("agent.model", payload.get("model", "unknown"))
        result = await run_agent(payload)
        span.set_attribute("agent.status", result.get("status", "unknown"))
        return {"trace_id": trace_id, **result}
```

Note that the OTLP exporter is used rather than a vendor-specific exporter, so the same instrumentation works against any OpenTelemetry-compatible backend. Vendor SDKs exist, but pinning to one makes migration expensive later.

With this in place, the queries that become possible are the ones a log store is good at. Representative examples, phrased as filters rather than natural language:

- All runs where `confidence_score < 0.7` and `latency_ms > 2000`.
- All sessions where the called tool name does not appear in the expected tool set for that agent.
- Count of runs per day where `status = "error"` grouped by `tool_name`.

### Where structured logs win

Structured logs are excellent when the failure is deterministic and external:

- **Timeouts and rate limits.** The downstream API returned 429 or the request exceeded the deadline. The log line says so directly.
- **Tool failures.** A tool returned a non-2xx status or malformed payload. The `tool_outputs` field records it.
- **Latency spikes.** `latency_ms` percentiles by model and endpoint show whether the regression is in the agent or the provider.
- **Deploy correlation.** Because the events carry timestamps and version tags, comparing error rates before and after a release is a standard query.

The setup cost is low, the query language is familiar, and the dashboards usually already exist. For an agent whose behavior is largely rule-bound — call tool A, then tool B, then format the result — this is often sufficient.

### Where structured logs fail

The gap is behavioral failure. If a downstream API returns 200 with a valid payload and the agent then ignores that payload and produces a fabricated answer, the logs record a successful call with normal latency. There is no error to find. The failure exists only in the agent's internal reasoning, which structured logs do not capture.

Two failure classes fall entirely outside log coverage:

- **Hallucination.** The agent produces a fluent, plausible, wrong output. Status codes are clean.
- **Prompt drift.** The agent's behavior changes gradually as inputs shift away from the distribution the prompt was tuned for, without any single error event.

A third class is partially covered but easily misread: **tool-output masking**, where the tool returns something the agent treats as unusable and silently substitutes its own answer. The log shows a successful tool call; only the reasoning trace shows that the output was discarded.

## Option B: agent behavior traces with causal attribution

A behavior trace records the decision path, not just the outcomes. The structure is a tree (or DAG) where nodes are reasoning steps, tool calls, or final answers, and edges carry the state that led from one node to the next.

What a useful trace captures per step:

- The prompt or message state at that point
- The model's output, including any reasoning or tool-selection text
- Any confidence or log-probability signal the model exposes
- Tool name, arguments, raw output, and duration
- The transition decision: which node was chosen next and why

A minimal wrapper using a trace-collecting client:

```python
import uuid
from langchain_core.runnables import RunnableConfig
from langchain_core.tracers import LangChainTracer
from langchain_core.callbacks import StdOutCallbackHandler
from langsmith import Client

client = Client()

def build_traced_config(config: RunnableConfig) -> RunnableConfig:
    tracer = LangChainTracer(
        project_name="agent-support-triage",
        client=client,
        example_id=str(uuid.uuid4()),
    )
    config = dict(config or {})
    config["callbacks"] = [tracer, StdOutCallbackHandler()]
    return config

async def run_agent(prompt: str, config: RunnableConfig):
    traced_config = build_traced_config(config)
    return await agent_chain.ainvoke(prompt, config=traced_config)
```

The exact client and tracer class names depend on the framework version; the pattern is what matters. Any tracer that records inputs, outputs, and parent-child relationships per step gives the same investigative capability.

### Reading a trace

When an agent fails, the trace is opened as a tree and walked from the root. The useful signals are:

1. **Confidence deltas at edges.** If confidence drops sharply after a tool call, that edge is the pivot point. Either the tool returned something unexpected, or the agent misread a valid output.
2. **Tool call arguments versus tool output.** Comparing what was asked for against what came back separates "the tool is broken" from "the agent asked the wrong question."
3. **Branch points.** Where the agent chose between two paths, the trace shows the reasoning that drove the choice. A typo in a tool name, for example, produces a valid-looking plan that never executes the intended call — visible only in the reasoning text, not in the tool log.

### Replay

The distinguishing capability of traces is replay: rerunning the exact sequence with a modified prompt, temperature, or model and comparing the resulting tree. This turns a debugging session into a controlled experiment.

Replay has a hard prerequisite: the environment must be reproducible. Pin the runtime, the library versions, the model version, and any retrieval index snapshot. If those drift, a fraction of traces will fail to replay and the comparison becomes meaningless. Building replay images in CI on every agent release is the standard mitigation.

### Where traces cost more

Trace storage and processing are heavier than log lines. A single agent run may produce dozens of nodes with full prompt and output text. Two costs follow:

- **Storage and ingestion.** Trace payloads are large because they contain model I/O, not just metadata.
- **Operational overhead.** Trace schemas need versioning, replay environments need maintenance, and confidence thresholds need tuning.

## How to measure the difference instead of trusting a benchmark

Claims that traces detect failures "200x faster" or that logs "miss 80% of AI failures" are not portable. Detection performance depends on the failure mix, the agent's architecture, and how confidence is instrumented. The honest approach is to measure on your own traffic.

A workable measurement procedure:

1. **Build a labeled failure set.** Collect real agent runs, or synthesize them, and label each with its failure mode: hallucination, tool failure, latency spike, prompt drift, or none. Aim for at least a few hundred examples per class you care about. Keep the labels out of the instrumentation so the test is not circular.
2. **Define detection per mode.** For each mode, decide what counts as "detected" in each system. For logs, that might be a query returning the offending run. For traces, it might be a human opening the trace and identifying the pivot node. Record the definition, because it drives the result.
3. **Measure time to detect.** For each labeled failure, record wall-clock time from failure occurrence to correct identification. This is the number that matters operationally, and it is dominated by human investigation time, not query speed.
4. **Measure time to root cause.** Separately record time from detection to a correct causal explanation. Traces should win here; logs may not even be able to reach an answer.
5. **Record false positives.** Count investigations that were opened and closed without finding a real failure. This is where log-based alerting on confidence thresholds tends to look worse than expected.
6. **Compare cost per thousand runs.** Instrument ingestion volume, not list price. Trace payloads are larger, so the relevant figure is bytes stored and processed per run.

Instrument what you need to run this: emit both a structured log line and a trace for every run, tag each with a `run_id` shared across both systems, and record the timestamps of failure occurrence, detection, and root cause in a small table. After a few weeks, the comparison is empirical rather than asserted.

## Decision checklist

Work through these before committing to a stack.

**Blast radius**
- Does a wrong answer reach a customer, a payment, a legal document, or a medical decision? If yes, traces are effectively mandatory.
- If the agent is internal and a wrong answer is caught by a human before it matters, logs may be enough to start.

**Behavioral determinism**
- Does the agent follow a fixed tool sequence with no free-form generation? Logs cover most failures.
- Does the agent generate free text, choose among tools, or apply confidence thresholds? Traces are needed to see the choice.

**Failure history**
- Has the team had at least one incident where logs looked clean and the output was still wrong? That is the signature of a behavioral failure and the strongest argument for traces.

**Reproducibility budget**
- Can the team pin model versions, library versions, and retrieval snapshots? If not, replay will be unreliable and the trace investment is partly wasted.

**Volume**
- At very high call volumes, trace storage and replay costs can dominate. Consider sampling: trace 100% of failures flagged by cheap heuristics, and a small random sample of successes for baseline comparison.

## A hybrid setup

The practical configuration most teams converge on:

- **Structured logs for the infrastructure layer.** Timeouts, rate limits, non-2xx tool responses, latency percentiles, cost per run. These are cheap, familiar, and alert well.
- **Traces for the behavioral layer.** Reasoning steps, tool I/O, confidence deltas, and replay for any run that a log-based rule flags or that a user reports.

Routing logic can be explicit. The following sketch sends infrastructure errors to the logger and everything else to the trace client:

```python
import logging
import langsmith

LOGGER = logging.getLogger("ai_agent")
TRACER = langsmith.Client()

INFRA_ERRORS = {"timeout", "rate_limit", "5xx"}

def record_failure(error, error_type: str):
    if error_type in INFRA_ERRORS:
        LOGGER.error(
            "infrastructure error",
            extra={
                "error_type": error_type,
                "run_id": getattr(error, "run_id", None),
            },
        )
        return

    TRACER.create_run(
        name=error_type,
        run_type="chain",
        inputs=getattr(error, "inputs", {}),
        outputs=getattr(error, "outputs", {}),
        error=str(error),
    )
```

Two caveats on the hybrid model:

- **Cold-start hallucinations.** If the agent's first reasoning step is wrong because the prompt template itself is wrong, no per-run trace will flag it as anomalous — every run will look consistently wrong. Catching this requires prompt regression tests: a fixed set of inputs with expected properties, run against every prompt change.
- **Tool-output masking.** The trace will show the tool call and its output, but deciding whether the agent should have used that output requires domain knowledge. No tooling automates this judgment today; it remains a human review task.

## Common failure modes in trace instrumentation

- **Missing correlation between logs and traces.** If the two systems do not share a run identifier, cross-referencing during an incident is manual and slow.
- **Unversioned trace schemas.** When the agent's step structure changes, old traces become unreadable. Version the schema alongside the agent.
- **Confidence thresholds with no baseline.** Alerting on "confidence below 70%" requires knowing the distribution of confidence on successful runs. Without that baseline, the threshold is arbitrary and will produce false positives.
- **Replay without pinning.** Rerunning a trace against a different model version produces a different tree for reasons unrelated to the change being tested.
- **Tracing everything at full fidelity.** At scale, storing complete prompts and outputs for every run is expensive. Sampling strategies are usually necessary.

## Recommendation

Adopt a hybrid stack: structured logs for infrastructure failures, behavior traces for behavioral failures, with a shared run identifier across both. This is the configuration that covers the failure classes each system misses on its own.

Choose it when any of the following holds:
- The agent's output reaches customers, money, or legal or medical content.
- The agent generates free text or selects among tools based on model reasoning.
- The team has already had an incident where the logs looked clean and the output was still wrong.

Skip traces, at least initially, when all of the following hold:
- The agent is internal, and a wrong answer is caught before it has consequences.
- The agent's behavior is a fixed tool sequence with no free-form generation.
- The team cannot yet pin model and library versions, so replay would be unreliable.

The last point is worth emphasizing: traces without reproducibility give you a picture of what happened but not the ability to test a fix. If pinning is not in place, that is the first thing to build.

## Action for the next 30 minutes

Open the wrapper that calls your agent and add one trace export line plus one structured log line, both carrying the same generated `run_id`. If the agent runs behind FastAPI, the OpenTelemetry snippet in Option A gives you the log side; wrapping the agent call with a tracer gives you the trace side. Then write down, in the same file, the three fields you would want to see first when a user reports a wrong answer — typically the input prompt, the tool outputs, and the final response. If those three are not already captured, add them now. That single change is what makes the next incident debuggable instead of mysterious.
===END===
