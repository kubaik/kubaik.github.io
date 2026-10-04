# Ship evals before your agent ships

## Why demo tests stop matching reality

Agent documentation typically ends with a demo that looks like this:

```python
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage

workflow = StateGraph(...)
workflow.add_node("agent", your_agent)
workflow.set_entry_point("agent")
app = workflow.compile()
response = app.invoke({"messages": [HumanMessage(content="Fix this broken API call")]})
print(response)
```

That prints something that looks good. Then the docs say: "Now run your agent in production."

What they don't say is what happens when:

- The agent retries the same failing API 50 times because the retry policy never checked for idempotency.
- The prompt works for "fix this API" but breaks when the ticket says "fix this API with the order_id that has a space in it."
- The cost estimator in the prompt uses a stale price, but the live catalog changed.

Teams hit these walls and call it "agent drift." The more precise term is *evaluation drift*: the agent still passes the demo tests, but the demo tests stopped matching reality.

A typical failure timeline for a team that ships on demo tests alone:

- Week 0: Demo runs, prompt feels good, the team ships it.
- Week 1: First spike in "still running" tickets. The agent is stuck in retry loops on a small share of tickets.
- Week 2: Someone notices the prompt token count jumped because the LLM started adding debugging commentary.
- Week 3: The team writes a quick eval that checks "does the agent close the ticket?" It passes because most tickets are simple, but misses the tail that is now costing real money in retries.

The demo tests never covered the tail. The new eval only checks the head. That is the gap.

What's missing is a *production-grade eval* that:

1. Runs against *live data* (not curated examples).
2. Measures *end-to-end outcomes* (ticket closed, cost, latency), not just prompt accuracy.
3. Fails the build when the agent regresses, not when the on-call page arrives.

That loop replaces "vibe testing" with a mechanical process.

## The four-part evaluation loop

The loop has four parts. Each part has a sharp edge where teams usually cut corners.

### 1. Instrument everything that matters

Most teams instrument the agent call but skip the downstream systems. That misses the failure mode where the agent succeeds but the ticket system doesn't update.

What to instrument:

- Agent input/output (prompt tokens, completion tokens, latency).
- Downstream API calls (success, rate limit, idempotency key, retry count).
- State changes in the ticket system (open → in_progress → resolved).
- Cost per ticket (LLM tokens + API calls + downstream retries).

A common instrumentation stack: OpenTelemetry for traces and metrics, Prometheus as the scrape endpoint, Grafana for dashboards, and an agent-specific tracing tool for spans. The category of "agent tracing tool" matters more than the specific vendor — what you need is per-run spans that carry the prompt, the tool calls, and the downstream HTTP responses.

### 2. Build a golden dataset from production traffic

Golden datasets are not curated examples. They are a snapshot of the last N tickets that met a quality bar. The quality bar is usually: closed within a short window and no downstream error.

How to build it without polluting your prod DB:

- Use feature flags to duplicate a percentage of traffic to a shadow agent.
- Write the shadow agent's outputs to a sidecar table with a flag `is_shadow: true`.
- After 24–48 hours, run a query to promote tickets where `is_shadow: true` and `status = resolved` and `downstream_errors = 0`.
- Export that set as your golden dataset.

A typical size is a few hundred tickets. Enough to catch most regressions, small enough to label manually in a day.

### 3. Write evals that fail the build

A common trap here is writing evals that are too narrow. Example:

```python
@evaluate(name="prompt_accuracy")
def check_prompt_accuracy(run, example):
    return {"score": 1 if "fix" in run.output.lower() else 0}
```

This passes if the word "fix" appears, but misses the case where the agent returns a 500-word essay that includes "fix" 12 times but never calls the API.

A production-grade eval must check:

- End-to-end outcome (ticket closed within the target window).
- Downstream success (API call succeeded within the retry budget).
- Idempotency (second run with same ticket returns immediately).
- Cost delta (LLM tokens + API calls within budget).

The eval should return a score between 0 and 1, and the CI pipeline should fail the build if the score drops below a threshold you set.

### 4. Run evals in CI, not just nightly

If evals only run nightly, a regression introduced at 3 PM ships at 8 PM and the fix arrives at 2 AM. The loop that replaces vibe testing runs evals on every PR:

- CI (GitHub Actions, GitLab CI, or similar) triggers on every push.
- The eval job pulls the golden dataset and the PR's agent code.
- It runs the eval against a local runner or a throwaway agent instance.
- If the score drops below threshold, the job fails the build.

Runtime for a few hundred tickets is typically a few minutes. Acceptable for most PRs.

The sharp edge is cost. A naive eval that runs hundreds of agent calls on every PR adds up. For a team with many PRs per day, that is still usually cheaper than a 2 AM page — but the arithmetic is worth doing before you commit to per-PR runs.

## Worked example: sizing the eval cost

Assume the following, all illustrative:

- Golden dataset: 800 tickets.
- Average tokens per call: 200 in + 200 out = 400 tokens.
- Model price: $5 per million input tokens, $15 per million output tokens.
- PRs per day: 10.
- Cache hit rate on eval runs: 70%.

Step 1 — per-run token cost:

- Input: 800 × 200 = 160,000 tokens = 0.16M × $5 = $0.80.
- Output: 800 × 200 = 160,000 tokens = 0.16M × $15 = $2.40.
- Per full run: $3.20.

Step 2 — per-PR cost with caching at 70%:

- Only 30% of calls hit the model: 800 × 0.30 = 240 calls.
- Scale the $3.20 by 0.30: $0.96 per PR.

Step 3 — daily cost:

- 10 PRs × $0.96 = $9.60/day.

Step 4 — compare against sampling:

- Sample 10% for PRs: 80 tickets. Scale $0.96 by 0.10 = $0.096 per PR.
- 10 PRs × $0.096 = $0.96/day, plus one full nightly run at $3.20.
- Total: $4.16/day.

The point of the worked example is not the specific numbers. It is that you should pick a dataset size, a cache, and a sampling rate that keep the daily eval cost below the cost of one bad incident. If a single bad ticket costs more than a day of evals, run the full set on every PR. If not, sample.

## Step-by-step implementation

Below is a minimal but production-grade implementation using Python 3.11, OpenTelemetry, a tracing/eval platform, and Redis. It covers the four parts of the loop.

### Prerequisites

- Python 3.11
- Redis 7.x (for caching agent outputs and rate limiting)
- An OpenTelemetry collector or OTLP endpoint
- A tracing/eval platform that supports custom evaluators
- pytest
- A CI runner (GitHub Actions, GitLab CI, etc.)

### 1. Instrumentation with OpenTelemetry

Add this to your agent entry point:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Initialize tracing
provider = TracerProvider()
trace.set_tracer_provider(provider)
exporter = OTLPSpanExporter(endpoint="https://otlp.example.com:4317", insecure=True)
provider.add_span_processor(BatchSpanProcessor(exporter))

# Your agent code
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage

def your_agent(state):
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("agent_call"):
        # ... your agent logic ...
        return {"messages": [HumanMessage(content="ok")]}

workflow = StateGraph(...)
app = workflow.compile()
```

Measure the per-span overhead on your own system before assuming it is acceptable. Export a histogram of span duration and compare a traced run against an untraced run on the same input set.

### 2. Shadow traffic and golden dataset

Add a feature flag to duplicate traffic:

```python
import os
from typing import Optional
import uuid

SHADOW_PERCENT = int(os.getenv("SHADOW_PERCENT", "5"))

class ShadowRouter:
    def __call__(self, request: dict) -> bool:
        # Deterministic sampling for reproducibility
        h = hash(request.get("ticket_id", ""))
        return (h % 100) < SHADOW_PERCENT

router = ShadowRouter()

# In your webhook handler
if router(request):
    # Shadow agent
    shadow_output = call_agent(request)
    # Store in sidecar table
    store_shadow_output(request["ticket_id"], shadow_output, is_shadow=True)
else:
    # Prod agent
    prod_output = call_agent(request)
```

After 48 hours, run this SQL to build the golden dataset:

```sql
-- PostgreSQL
INSERT INTO golden_tickets (ticket_id, input, expected_output, metadata)
SELECT 
    ticket_id,
    input,
    shadow_output AS expected_output,
    jsonb_build_object(
        'closed_in_minutes', extract(epoch from (closed_at - created_at))/60,
        'downstream_errors', (SELECT count(*) FROM downstream_errors WHERE ticket_id = t.ticket_id)
    ) AS metadata
FROM shadow_outputs s
JOIN tickets t ON s.ticket_id = t.ticket_id
WHERE s.is_shadow = true
  AND t.status = 'resolved'
  AND t.closed_at > t.created_at
  AND (t.closed_at - t.created_at) < interval '5 minutes'
  AND (SELECT count(*) FROM downstream_errors WHERE ticket_id = t.ticket_id) = 0;
```

The dataset size depends on your traffic. With `SHADOW_PERCENT=5` and a few thousand tickets per day, a few hundred promoted rows after 48 hours is a reasonable expectation.

### 3. Production-grade evals

Define a custom evaluator that checks end-to-end outcomes:

```python
from langsmith import EvaluationResult, evaluator
from typing import Dict, Any
import time

@evaluator(run_type="chain")
def agent_eval(run: Dict[str, Any], example: Dict[str, Any]) -> EvaluationResult:
    # Extract outputs
    output = run.output
    # Check for end-to-end outcome
    expected_output = example["expected_output"]
    # Simple string match for demo; in prod use semantic similarity or API call checks
    score = 1.0 if expected_output in output else 0.0
    
    # Add metrics
    metrics = {
        "latency_ms": run.metrics.get("latency_ms", 0),
        "tokens_in": run.metrics.get("prompt_tokens", 0),
        "tokens_out": run.metrics.get("completion_tokens", 0),
    }
    
    return EvaluationResult(
        key="agent_outcome",
        score=score,
        comment=f"Output matches expected: {score == 1.0}",
        metadata=metrics,
    )
```

Then run the eval in CI:

```yaml
# .github/workflows/evals.yml
name: Agent Evals
on: [push]
jobs:
  evals:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install langsmith pytest
      - run: pytest tests/evals/test_agent_evals.py
        env:
          LANGSMITH_API_KEY: ${{ secrets.LANGSMITH_API_KEY }}
          LANGSMITH_PROJECT: "my-agent-prod"
```

### 4. Cache and rate limit to avoid stampedes

Add Redis caching to prevent the "cache stampede" where every eval call triggers a live agent run:

```python
import redis.asyncio as redis
import json

r = redis.Redis(
    host="redis.internal",
    port=6379,
    db=0,
    decode_responses=True,
)

async def cached_agent(input: dict, ttl: int = 300):
    key = f"agent:{hash(str(input))}"
    cached = await r.get(key)
    if cached:
        return json.loads(cached)
    output = await call_agent(input)
    await r.setex(key, ttl, json.dumps(output))
    return output
```

Measure your own cache hit rate by counting `r.get` hits and misses over a fixed window. The hit rate depends entirely on how much your eval inputs repeat across runs.

## How to measure the loop in your own system

Do not trust a benchmark table from someone else's deployment. Instrument these and read them yourself:

- **Eval pass rate per PR.** Emit a metric from the CI job. Plot it over time. A downward trend before a user-visible incident is the whole point of the loop.
- **Eval runtime.** Time the eval job end to end. If it crosses your PR latency budget, switch to sampling.
- **Eval cost per run.** Multiply measured token counts by your model's current price. Do this monthly; prices change.
- **Cache hit rate.** Count hits and misses in Redis. If it is low, either your inputs are too varied or your TTL is too short.
- **Downstream error rate.** Count non-2xx responses from the APIs your agent calls. This is the metric that catches the failures the agent itself cannot see.
- **Prompt token drift.** Track `prompt_tokens` per call as a time series. A slow upward drift is usually a prompt that keeps growing.

The last two are the ones that catch the failure modes described below.

## The failure modes nobody warns you about

### 1. The golden dataset becomes stale

Golden datasets decay. The evals still pass, but the agent no longer matches reality.

Typical symptom: the eval score stays high, but on-call pages spike.

Fix: rotate a fraction of the golden dataset on a fixed cadence. Use a background job that pulls the last 24 hours of closed tickets that passed the quality bar.

### 2. The eval is too slow for CI

Naive evals that run hundreds of agent calls on every PR can take 15+ minutes. Teams disable them.

Fix: use caching and sampling. Sample a fraction of the golden dataset for PRs, run the full set nightly. Or use a smaller golden set for PRs and the full set only on main.

### 3. The downstream API changes but the eval doesn't notice

The eval checks "ticket closed," but the downstream API changed its idempotency key format. The agent still closes tickets, but retries explode.

Fix: add a downstream API health check to the eval. Example:

```python
@evaluator(run_type="chain")
def downstream_health_eval(run: Dict[str, Any], example: Dict[str, Any]) -> EvaluationResult:
    # Extract the last downstream call from traces
    trace = run.get("traces", [{}])[0]
    downstream_calls = trace.get("downstream_calls", [])
    if not downstream_calls:
        return EvaluationResult(key="downstream_health", score=0.0, comment="No downstream calls found")
    last_call = downstream_calls[-1]
    if last_call.get("status") != "success":
        return EvaluationResult(key="downstream_health", score=0.0, comment=f"Downstream failed: {last_call.get('status')}")
    # Check idempotency key format
    key = last_call.get("idempotency_key", "")
    if not key.startswith("ord_"):
        return EvaluationResult(key="downstream_health", score=0.5, comment="Idempotency key format changed")
    return EvaluationResult(key="downstream_health", score=1.0, comment="Downstream healthy")
```

### 4. The eval metric is gamed

Teams write evals that are too narrow, so the agent learns to game them. Example: the eval only checks if the word "resolved" appears in the output. The agent starts returning "Status: resolved. Ticket closed. Status: resolved."

Fix: use multiple evals with different scorers. Example weighting:

| Eval name | Scorer | Weight |
|---|---|---|
| Outcome match | String match vs expected output | 0.4 |
| Downstream success | API call status = success | 0.3 |
| Idempotency | Second run returns immediately | 0.2 |
| Cost budget | tokens_in + tokens_out ≤ budget | 0.1 |

If any single scorer drops below a threshold you set, the overall score fails.

## Decision checklist before you build this

Answer these before writing eval code:

- **Is there a measurable end-to-end outcome?** If the agent's job is "brainstorm ideas" or "write a blog post," there is no outcome to measure. Vibe testing is the only option.
- **Does anyone read what the agent writes?** If the agent writes to a DB but nobody reads it, you cannot measure success. Use prompt regression tests instead (prompt length, token count, readability).
- **Is the eval cost less than the failure cost?** If a bad run costs cents and evals cost dollars per run, the loop is not worth it. This usually happens with low-traffic agents.
- **Will someone own the golden dataset?** Golden datasets require labeling. If nobody will do it, skip evals and instrument downstream health checks instead.
- **Can you fail a build?** If your team cannot block a merge on a failing eval, the loop will be ignored. Get that agreement first.

## Tooling categories worth evaluating

| Category | What it does | Trade-off |
|---|---|---|
| Tracing backend | Stores spans for agent runs and downstream calls | Vendor lock-in on span format |
| Eval runner | Executes scorers against a dataset and reports scores | Proprietary eval formats make migration costly |
| Cache | Deduplicates repeated eval calls | Requires key stability across runs |
| CI runner | Triggers evals on PRs and blocks merges | Minute costs scale with dataset size |
| Dataset store | Holds golden tickets and metadata | Needs a rotation job or it goes stale |

The sharp edge across all of these is lock-in. Eval formats in particular tend to be vendor-specific. If you expect to switch later, keep your scorers as plain Python functions and treat the platform as a runner, not as the source of truth.

## What to do in the next 30 minutes

Open your agent's repo and run this command:

```bash
grep -r "def your_agent" src/ | head -1
```

If you find an agent function, check two things:

1. Does it have OpenTelemetry tracing? If not, add the instrumentation shown above.
2. Does your CI run evals on every PR? If not, create a minimal eval that checks one end-to-end outcome — for example, "ticket closed within the target window."

If either is missing, add tracing or a failing eval today. The goal is not to ship perfect evals in one day. The goal is to turn one vague failure ("agent feels slow today") into a mechanical process ("eval failed at 3:42 PM, here's the trace").

## FAQ

**How do I build a golden dataset if my agent hasn't been in production yet?**

Start with synthetic data that matches your expected production traffic. Generate realistic tickets with a library like Faker, or use a synthetic dataset generator in your eval platform. Pair each synthetic ticket with an expected output (the correct API call or resolution text). Once you have 24–48 hours of real shadow traffic, replace the synthetic set with real golden tickets.

**What if my agent's output is unstructured (natural language summaries)?**

Combine string matching with semantic similarity. For example, pair a keyword check (e.g., "resolved" must appear) with a cosine similarity scorer using a sentence-embedding model. Weight the semantic scorer higher and the keyword scorer lower to avoid gaming. Add a cost metric to penalize verbose outputs.

**How do I handle evals that take too long for CI?**

Three levers: sample, cache, and split. Sample a fraction of the golden dataset for PRs and run the full set nightly. Cache agent outputs in Redis so repeated eval calls reuse results. Split evals into two jobs: a fast smoke test (a handful of tickets) that must pass, and a full regression suite that runs only on main.

**What's a realistic eval budget?**

Work it out from your own numbers using the worked example above. The key inputs are dataset size, tokens per call, current model prices, PR volume, and cache hit rate. If your agent is low-traffic, run evals nightly instead of per-PR to reduce cost.
