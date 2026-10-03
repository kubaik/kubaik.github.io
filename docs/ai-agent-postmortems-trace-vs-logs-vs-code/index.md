# AI agent postmortems: trace vs logs vs code

AI agent failures rarely look like ordinary incidents. A latency spike in a microservice shows up in logs and metrics. An agent that starts recommending out-of-stock products after a prompt or model change often does not. The system is up, latency is normal, error rates are flat, and yet conversion drops or support tickets climb. The observable signals that worked for request/response services are necessary but not sufficient here.

This article compares three diagnostic surfaces teams use for AI agent postmortems: distributed tracing, evaluation logs, and code-level debugging. Each one exposes a different class of failure. Choosing the wrong one is a common way to lose days chasing a symptom that lives somewhere else.

## Why agent postmortems differ from service postmortems

A conventional postmortem asks: what broke, when, and why? For an agent, "broke" is ambiguous. There are at least four distinct failure classes, and they surface in different places:

- **Availability failures** — the agent errors, times out, or the upstream model API returns 429/5xx. Visible in logs and metrics.
- **Data failures** — retrieval returns stale, empty, or wrong documents. Visible in traces, invisible in aggregate metrics.
- **Reasoning failures** — the model produces a plausible but incorrect or unfaithful answer given correct inputs. Visible only through output comparison or evaluation.
- **Behavioral drift** — outputs change gradually after a model version bump, prompt edit, or index rebuild. Visible only when you compare outputs over time on a fixed input set.

Logs answer the first class well. Traces answer the second. Evaluation logs answer the third and fourth. Code debugging answers the question of which of your own branches produced the bad input in the first place. A postmortem that only inspects logs will systematically miss reasoning and drift failures.

The practical consequence: before opening any tool, classify the failure. The rest of this article describes what each surface can and cannot show.

## Tracing: causal chains across agent steps

Tracing instruments the agent lifecycle as a tree of spans. For a retrieval-augmented agent, a single request typically produces spans for: input parsing, embedding, vector search, reranking, prompt assembly, the model call, any tool invocations, and output post-processing. Each span carries attributes — model name, token counts, retrieval hit counts, similarity scores, latency per step.

The strength of tracing is causal localization. If retrieval returns zero documents and the model then answers from parametric memory, the trace shows an empty retrieval span immediately upstream of a model span with no retrieved context in its input. That is a five-second diagnosis in a trace and a multi-hour hunt in logs, because the final output looks like a normal answer.

Tracing also supports cross-request comparison. If the same query produces different answers at different times, comparing the two traces shows whether the retrieval results, the prompt, or the model version differed. That comparison is what makes drift attributable rather than merely observable.

The cost of tracing is proportional to request volume and span cardinality. Two failure modes are common:

- **Span explosion.** A loop that retries a failed export, or a tool that emits one span per item in a large list, can multiply span count by orders of magnitude. A retrying exporter that never succeeds will buffer and re-send indefinitely.
- **Sampling blind spots.** To control volume, teams sample, often at 10%. Sampling is fine for latency percentiles and coarse error rates. It is bad for rare reasoning failures, because the request that failed may not have been sampled.

A minimal OpenTelemetry setup for an agent step looks like this. Note that the exporter endpoint and headers depend on your backend; the values below are placeholders.

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

resource = Resource.create({"service.name": "recommendation-agent"})
provider = TracerProvider(resource=resource)
trace.set_tracer_provider(provider)

# Point this at your collector or vendor endpoint.
exporter = OTLPSpanExporter(
    endpoint="https://your-collector.example/v1/traces",
    headers={"authorization": "Bearer YOUR_TOKEN"},
)
provider.add_span_processor(BatchSpanProcessor(exporter))

tracer = trace.get_tracer(__name__)

def recommend(user_query: str):
    with tracer.start_as_current_span("recommend_products") as root:
        with tracer.start_as_current_span("retrieve_inventory") as span:
            products = get_inventory_from_api(user_query)
            span.set_attribute("retrieval.items", len(products))
            span.set_attribute("retrieval.backend", "vector")
        with tracer.start_as_current_span("llm_generate") as span:
            response = call_llm(user_query, products)
            span.set_attribute("llm.output.length", len(response))
            span.set_attribute("llm.model", "your-model-id")
        root.set_attribute("agent.recommendation.count", len(response))
```

Two notes on correctness. First, set attributes on the span object returned by `start_as_current_span`; calling `trace.get_current_span()` inside a nested block can return an unexpected span if the context has been re-entered. Second, avoid high-cardinality attributes such as raw user queries or full outputs on every span unless your backend is priced and indexed for it — these are the most common cause of cost surprises.

### How to measure tracing overhead before committing

Do not estimate overhead from blog posts. Measure it:

1. Instrument a single agent path behind a feature flag.
2. Run a fixed load (for example, 500 requests at steady state) with the flag off, then on.
3. Compare p50, p95, and p99 latency, and CPU utilization, between the two runs.
4. Record span count per request and bytes per span from your collector's own metrics.
5. Multiply span count by your backend's per-span ingest price to get a real monthly figure at your actual volume.

The numbers that matter are yours: span cardinality per request, sampling rate, retention window, and your vendor's ingest pricing. Any table of dollar figures that does not come from those four inputs is not a forecast.

## Evaluation logs: reproducible failure detection

Evaluation logs are structured records of running the agent against a fixed set of inputs with expected properties. A typical record contains: test case ID, input, expected output or expected properties, actual output, metric scores, model version, prompt version, retrieval configuration, and timestamp.

The strength of evaluation logs is reproducibility and regression detection. If a prompt change degrades faithfulness, a test suite run before deployment catches it. This is fundamentally different from tracing: tracing tells you what happened to real users; evaluation tells you what happens to a controlled input set, on demand, before users are affected.

A second strength is correlation. Because each run records the model version, prompt version, and retrieval configuration, you can join evaluation results to business metrics by deployment window. That join is what turns "conversion dropped 12%" into "conversion dropped 12% after prompt v2 shipped, and prompt v2 also dropped faithfulness from 0.92 to 0.65 on the regression suite."

The weakness is coverage. Evaluation logs only catch what your test cases exercise. A suite of 50 cases on a product catalog with thousands of SKUs will miss rare categories, ambiguous queries, and adversarial inputs. Coverage gaps are invisible: a green suite is not evidence of correctness, only of correctness on the cases you wrote.

Two metric categories are worth distinguishing, because they fail differently:

- **Reference-based metrics** compare the output to a known-good answer. They are precise but require labeled data and do not generalize to open-ended generation.
- **Reference-free metrics** (relevance, faithfulness to retrieved context, format compliance) use a model or heuristic to score the output. They scale to open-ended tasks but are themselves probabilistic and can be gamed by verbose or hedged answers.

A minimal evaluation harness using a common open-source evaluation library:

```python
import json
from dataclasses import dataclass, asdict

@dataclass
class Case:
    case_id: str
    input: str
    expected_properties: dict

CASES = [
    Case("oos-1", "Recommend a gaming laptop under $1500",
         {"in_stock": True, "price_max": 1500}),
    Case("amb-1", "Show me something good",
         {"asks_clarifying_question": True}),
]

def check(output: str, props: dict) -> dict:
    """Replace with your own checks or an eval library's metrics."""
    results = {}
    if "in_stock" in props:
        results["in_stock"] = "out of stock" not in output.lower()
    if "price_max" in props:
        results["price_max"] = True  # parse the price and compare
    if "asks_clarifying_question" in props:
        results["asks_clarifying_question"] = output.strip().endswith("?")
    return results

def run(agent, path="eval_results.jsonl"):
    with open(path, "w") as f:
        for case in CASES:
            output = agent(case.input)
            scores = check(output, case.expected_properties)
            record = {
                "case_id": case.case_id,
                "input": case.input,
                "output": output,
                "scores": scores,
                "passed": all(scores.values()),
                "model_version": agent.model_version,
                "prompt_version": agent.prompt_version,
            }
            f.write(json.dumps(record) + "\n")
```

Log the model and prompt versions on every record. Without them, a regression cannot be attributed to a change, and the log becomes a historical curiosity rather than a diagnostic instrument.

### How to measure evaluation coverage

Coverage is measurable, and it is worth measuring before trusting a green suite:

1. Sample a week of real production inputs.
2. Cluster them by intent or category (manually or with embeddings).
3. For each cluster, check whether at least one test case exercises it.
4. Track the percentage of production traffic covered by at least one test case.

If a cluster representing 15% of traffic has no test case, your suite has a 15% blind spot regardless of how many total tests it contains. Total test count is a vanity metric; traffic-weighted coverage is the useful one.

## Code-level debugging: when the bug is yours

Code debugging is the right tool when the failure originates in your own logic rather than the model's reasoning or the retrieval data. Classic examples:

- A prompt template that silently drops a variable because of a formatting bug, so the model never sees the retrieved context.
- A retry wrapper that catches an exception and returns an empty list, which the model interprets as "no products available."
- A caching layer with a TTL that serves stale inventory, so the model reasons correctly over wrong data.
- A token-truncation step that cuts the retrieved context before the model call, so the answer is unfaithful to documents the trace shows were retrieved.

These failures are invisible in evaluation logs if the test cases do not trigger the code path, and they look like model problems in traces unless you read the span attributes carefully. The diagnostic move is to compare what the trace says was retrieved with what the prompt actually contained. If those differ, the bug is in your code, not the model.

## Choosing between them

The three surfaces are complementary, not competing. The decision is about which to build first and where to invest depth.

| Surface | Best at | Blind to | Overhead profile |
|---|---|---|---|
| Tracing | Causal localization, cross-request comparison, production ground truth | Reasoning quality, rare failures under sampling | Per-request; scales with span count and retention |
| Evaluation logs | Regression detection, version attribution, pre-deploy gating | Anything outside the test set | Zero at request time; cost is test maintenance |
| Code debugging | Bugs in prompt assembly, caching, retries, truncation | Model reasoning, data quality | None; effort is human time |

A practical ordering for most teams:

1. **Start with evaluation logs** if the agent is changing frequently. The setup cost is low and the payoff — catching regressions before deployment — is immediate.
2. **Add tracing** when you need production ground truth: failures that only occur with real inputs, or drift that only appears over time.
3. **Keep code debugging in the loop** for every postmortem. Before blaming the model, verify that the prompt the model received matches the data the trace says was retrieved.

Signals that you have chosen wrong:

- You are reading traces to debug a reasoning failure. Traces show inputs and outputs, not why the model chose an output. Use evaluation.
- You are reading evaluation logs to debug a production-only failure. If the failure does not reproduce on your test set, the test set is the problem, not the log.
- You are adding spans to fix a bug in your prompt assembly. That bug is in code; a debugger or a print of the assembled prompt is faster.

## A worked example

Suppose a recommendation agent's conversion rate drops 12% over three days. Walk the surfaces in order of cost.

**Step 1 — check evaluation logs.** Did any regression suite run in the window? If prompt v2 shipped on day one and the suite shows faithfulness dropping from 0.92 to 0.65, the cause is likely the prompt. If no suite ran, you have no pre-deploy signal and must go to production data.

**Step 2 — check traces.** Filter to the deployment window. Compare the distribution of retrieval hit counts and model versions before and after. If retrieval hit counts are unchanged and the model version is unchanged, the change is in the prompt or in the data itself.

**Step 3 — check the data.** If the retrieval index was rebuilt in the window, compare a sample of retrieved documents before and after. A stale or partially rebuilt index produces correct-looking traces with wrong content.

**Step 4 — check the code.** Diff the prompt assembly path. If a template variable was renamed and the formatter silently dropped it, the model received a prompt without retrieved context. The trace shows the retrieval succeeded; only the assembled prompt reveals the bug.

The point of the ordering is cost: evaluation logs are cheapest to check, traces are next, and reading code is the most expensive in human time. Walk them in that order rather than starting with the most thorough-looking tool.

## FAQ

**Do I need all three?** No. Most teams need evaluation logs plus one of tracing or code debugging, depending on whether failures are reproducible offline. Add the third when you have a specific failure class the first two cannot see.

**How much should I sample traces?** Sample as much as your budget allows for the failure you are chasing, and no more. For latency and error-rate monitoring, low single-digit percentages are usually sufficient. For rare reasoning failures, sampling is the wrong tool — use evaluation logs instead.

**Can evaluation metrics be trusted?** Reference-based metrics can, within the scope of their labeled data. Reference-free metrics are themselves model outputs and should be treated as noisy signals, not ground truth. Calibrate them against a human-labeled sample before gating deployments on them.

**What should a trace span always carry?** Model version, prompt version, retrieval configuration, and token counts. Without version attributes, a trace cannot attribute a change to a cause, which is the whole point of tracing in a postmortem.

**What is the most common instrumentation mistake?** Putting high-cardinality data — raw queries, full outputs — on every span, then discovering the cost only after volume grows. Start with low-cardinality attributes and add detail only where you have a specific diagnostic need.

## Next step

Pick one agent you own and write ten test cases covering its known edge cases: ambiguous inputs, empty retrieval results, out-of-stock items, and inputs in a second language if you support one. Run them against the current agent, log each result as JSON with the model version and prompt version, and note which cases fail. That gives you a baseline regression suite, a measured coverage number, and a concrete list of failures to classify as data, reasoning, or code — which is the input the rest of this article's decision framework needs.
