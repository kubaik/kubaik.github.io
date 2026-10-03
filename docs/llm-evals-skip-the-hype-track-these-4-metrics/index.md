# LLM evals: skip the hype, track these 4 metrics

Most LLM evaluation documentation is written for research: perplexity, BLEU, ROUGE, embedding cosine similarity. Those metrics optimize for language similarity. Production systems need to optimize for two different things: whether users notice a regression, and whether the cost per request stays sustainable. The gap between those two goals is where most evaluation pipelines quietly fail.

## Why research metrics miss production failures

A typical failure mode: a team spends weeks tuning a prompt, benchmarks against a public coding eval, and buys a third-party "AI quality score." Users still complain, because the assistant invents policy details that contradict the jurisdiction it operates in. The external score can be high while task accuracy is low, because the score is measuring fluency, not correctness against the business's own ground truth.

The root issue is not measurement technique. It is scope. If an evaluation pipeline does not include:

- a latency SLO tied to user drop-off,
- a cost-per-request ceiling,
- and a ground-truth alignment test built from your own business data,

then it is optimizing for noise. Two common traps:

1. Over-indexing on automatic similarity metrics (BLEU, ROUGE, BERTScore). These correlate weakly with human preference on open-ended generation tasks; the correlation is task-dependent and should be measured on your own data before you trust it.
2. Running expensive human evaluation monthly instead of lightweight continuous checks that can alert within minutes.

The useful distinction is between metrics that tell you something actionable now and metrics that tell you something interesting in a month.

## The four metric classes

Production LLM evaluation is a feedback loop: generate candidate outputs, score them with cheap automated checks, and feed results back into prompt iteration or fine-tuning. The scoring layer trades statistical rigor for speed and cost. Four classes of checks cover most production needs.

**1. Deterministic validators.** Regex and schema checks for IDs, phone formats, currency symbols, date ranges, and numeric bounds. These are O(1), cheap, and catch the mechanically wrong outputs. They will not catch subtle misalignment, but they catch a large share of the errors users actually report.

**2. Statistical validators.** Embedding similarity against a ground-truth corpus, or a small domain-specific classifier. Useful for detecting paraphrase-level drift, but prone to false positives when the ground truth is narrow. Measure the false-positive rate on a labeled sample before trusting the threshold.

**3. Policy validators.** A JSON schema or rule set that encodes hard constraints (regulatory caps, required disclosures, forbidden claims). A response that violates a policy fails immediately, no human needed. These are the highest-value checks in regulated domains because they are auditable.

**4. User impact.** A/B rollouts with a small traffic split, watching task completion rate, time-to-decision, and support ticket volume. This is the only class that measures what users actually experience.

No single class is sufficient. A portfolio is required, because each class fails in a different direction.

## A minimal loop in code

The examples below are illustrative and use placeholder model and endpoint names. Substitute your own inference server, embedding model, and moderation endpoint.

### 1. Candidate generation

```python
import asyncio
from vllm import AsyncLLMEngine
from vllm.sampling_params import SamplingParams

MODEL_ID = "your-model-id"
ENGINE = AsyncLLMEngine.from_engine_args(
    engine_args={
        "model": MODEL_ID,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.90,
        "max_model_len": 4096,
    }
)

async def generate(prompt: str, max_tokens: int = 512) -> str:
    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.3)
    result = await ENGINE.generate(prompt, sampling_params)
    return result.outputs[0].text
```

### 2. Automated scoring

Run scorers concurrently so total scoring latency is bounded by the slowest one.

```python
import re
from sentence_transformers import SentenceTransformer

# Load once at startup
FACT_CHECKER = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

RULES = {
    "id_format": r"^KEN[A-Z0-9]{6}$",
    "currency": r"KES\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?",
}

RATE_CAP = 0.24  # illustrative policy cap

async def score_response(prompt: str, response: str) -> dict:
    rule_score = 1.0
    for name, pattern in RULES.items():
        if not re.search(pattern, response):
            rule_score = 0.0

    rate_match = re.search(r"(\d{1,3}(?:\.\d{2})?)%", response)
    if rate_match:
        rate = float(rate_match.group(1)) / 100
        if rate > RATE_CAP:
            rule_score = 0.0

    embeddings = FACT_CHECKER.encode([prompt, response], convert_to_tensor=True)
    semantic_score = float(embeddings[0] @ embeddings[1].T)

    return {
        "rule_score": rule_score,
        "semantic_score": semantic_score,
    }
```

### 3. Metric aggregation

Expose a Prometheus endpoint and write per-request scores to a shared store for dashboards.

```python
from prometheus_client import start_http_server, Counter, Gauge
import redis.asyncio as redis

REDIS = redis.Redis(host="redis", port=6379, decode_responses=True)

SCORE_COUNTER = Counter("llm_eval_score_total", "Total LLM eval scores", ["metric"])
FAILURE_GAUGE = Gauge("llm_eval_failures", "Current failure rate")

async def record_metrics(request_id: str, scores: dict):
    await REDIS.hset(f"scores:{request_id}", mapping=scores)
    for k, v in scores.items():
        SCORE_COUNTER.labels(metric=k).inc(v)
    FAILURE_GAUGE.set(scores["rule_score"] < 1.0)

start_http_server(8000)
```

### 4. Canary rollout

A time-based canary shifts a small percentage of traffic, waits, then promotes or rolls back. The exact percentage and bake time depend on your traffic volume; the point is that rollback must be automatic and fast.

```yaml
# appspec.yml (illustrative)
version: 0.0
Resources:
  - TargetService:
      Type: AWS::Lambda::Function
      Properties:
        Name: llm-eval-candidate
        TrafficRouting:
          Type: TimeBasedCanary
          TimeBasedCanary:
            StepPercentage: 1
            BakeTimeMins: 15
Hooks:
  - AfterAllowTraffic: LambdaValidationFunction
```

### 5. Fine-tuning trigger

When a rolling average of the semantic score drops below a threshold, trigger a fine-tuning job. Keep the threshold configurable; a fixed threshold will not survive model or data drift.

```python
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "your-base-model"

def train_lora(train_data: list[dict], output_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # ... training loop ...
    model.save_pretrained(output_dir)
```

## How to measure the four metrics

Do not copy benchmark numbers from another team. Instrument your own system and compare against your own baseline. What to record:

- **Deterministic validator pass rate.** Log every rule name and pass/fail per request. Compare the daily rate against a 7-day rolling median.
- **Semantic score distribution.** Log the raw score, not just the mean. Watch the 5th percentile; a mean that holds steady while the tail degrades is a common failure pattern.
- **Policy violation count.** Count violations by rule. Any nonzero count on a hard policy rule is a page-worthy event.
- **User impact.** Run an A/B split, then compare task completion rate, time-to-decision, and support ticket volume between arms. Use a fixed observation window and compute the difference relative to the control arm's variance, not to a fixed percentage.

A worked example with stated assumptions: suppose a service handles 100,000 requests per day. If a scorer adds 50 ms of CPU work at a cost of $0.000001 per request-millisecond, that is 100,000 × 50 × $0.000001 = $5 per day, or roughly $150 per month. At 5,000,000 requests per day, the same scorer costs roughly $7,500 per month. The point is not the specific figures, which are illustrative, but the arithmetic: cost scales linearly with traffic, so a scorer that is negligible at low volume can dominate the bill at high volume. Recompute after every traffic milestone.

## Failure modes worth planning for

**Prompt drift under version control.** Prompts do not behave like code. Adding a single line to a system prompt can shift the semantic score materially on the same input set. Require a semantic diff between prompt versions and a mandatory bake time before promotion. Without it, a prompt change can silently degrade output for hours before anyone notices.

**Cold-start scoring latency.** The first call to an embedding model in a fresh container can take seconds. If your SLO is under 500 ms p95, account for this explicitly: pre-warm the model, use provisioned concurrency, or move scoring off the request path.

**Cost feedback loop inflation.** Adding a scorer increases cost per request, which may push you to reduce sampling, which reduces signal quality, which forces you to add more expensive checks. Track cost per metric, not just cost per request.

**Metric poisoning by edge cases.** A validator that matches the wrong region's ID format will pass while users receive wrong identifiers. Test validators against real user inputs, not synthetic ones. A regex that looks correct on paper often fails on production data.

**Alert fatigue from noisy baselines.** A fixed threshold below a rolling median will fire on normal weekday/weekend variation. Exclude non-working days from the baseline or use a seasonal decomposition before alerting.

## Tool categories and what to check before adopting

Rather than naming specific products, evaluate tools by category:

| Category | What it must do | What to check |
|---|---|---|
| Inference server | Batch and stream generation | p95 latency under your load; does it expose Prometheus stats |
| Prompt/version store | Diff prompts and scorers | Semantic diff support; rollback speed |
| Metrics backend | Store per-request scores | Cardinality limits; retention cost |
| Cache | Store ground truths and embeddings | p99 lookup latency; async client support |
| Fine-tuning platform | LoRA or full fine-tune | Checkpoint behavior on spot instances |
| Test framework | Test validators and scorers | Async test support; fixture reuse |
| Moderation endpoint | Policy and toxicity checks | Rate limits; whether it logs inputs |
| Deployment tooling | Canary and rollback | Rollback time; whether rollback is automatic |

A custom scorer is rarely worth building from scratch unless the domain is highly regulated. Start with a fine-tuned open model before training from scratch.

## When this approach is the wrong choice

Skip the full loop if:

1. **Users are internal and volume is low** (roughly under 1,000 requests per day). A prompt template and manual review are faster and cheaper.
2. **The LLM is a research prototype** with no business SLA. Academic metrics are sufficient.
3. **The model changes daily.** The maintenance overhead of validators and scorers outweighs the benefit.
4. **There is no ground truth.** If a correct response cannot be defined, automated scoring is meaningless.
5. **Regulation requires human sign-off.** Build a human-in-the-loop process with clear escalation paths instead.

A common mistake is adding an evaluation loop to a system that already meets its quality bar. If accuracy is acceptable and latency is the binding constraint, an added scorer can increase p95 latency without improving user-visible outcomes. Measure before adding.

## Treat scorers as production code

A single misplaced character in a regex can cause a measurable jump in false negatives. Scorers should live in the same repository as the prompt, have test coverage, and go through pull request review. They are infrastructure, not a side project.

The bigger lesson is about process. Manual prompt review cycles measured in weeks do not scale. A canary with automated rollback can cut iteration time from weeks to hours, at the cost of maintaining the scorer infrastructure. That trade is usually worth making once traffic is high enough to justify it.

Users do not care about your metrics. They care about task completion time and accuracy. The most reliable improvements come from reducing time-to-decision and eliminating wrong answers, not from moving a similarity score from 0.89 to 0.93.

## FAQ

**How much data is needed to start scoring?**
For rule-based validators, a few dozen labeled examples are enough to bootstrap. For a semantic scorer or classifier, start with at least 100 labeled examples and expand as traffic accumulates.

**Why not use human evaluation for everything?**
Human evaluation is slow and expensive per label. It is best used to calibrate automated scorers periodically, not for continuous monitoring.

**What latency budget should the scorer have?**
Budget the scorer so total p95 stays within your user-facing SLO. If the SLO is 500 ms, keep scorer overhead well under that, and measure cold-start behavior separately.

**How do you handle model updates without breaking validators?**
Pin scorer model versions, run a semantic diff between old and new scorers on a fixed input set, and pause the canary if the score delta exceeds a threshold you define.

## One action to take in the next 30 minutes

Open your prompt file and list the rules that would catch the most common errors your users report. Then write a test for each one:

```bash
pip install pytest pytest-asyncio
python -m pytest tests/test_validators.py -v
```

If any validator fails, fix it before merging. That single step catches a large share of downstream incidents before they reach users.
