# LLM metrics that stop wasting time

## Why static benchmarks stop being useful after launch

Model cards and public benchmarks answer a narrow question: which model performed better on a frozen test set at a point in time. MMLU, HumanEval and similar suites are useful for that. They are not designed to answer the question that matters six months into production: is the system answering correctly *right now*, for *these* users, with *this* prompt and *this* retrieval corpus.

The gap is not about model capability. It is about operational context. A model can score well on a public benchmark and still fail in a specific deployment because:

- The prompt template includes domain data (a product catalog, a discount engine) that the benchmark never saw.
- The retrieval corpus changes weekly, so the notion of "supported by context" changes with it.
- Users write in languages and registers the benchmark does not cover.
- Cost and latency constraints mean the model you *can* serve is not the model you would pick on accuracy alone.

A useful framing is that there are three distinct evaluation problems, and most tooling only addresses the first:

1. **Model selection** — which base model to fine-tune or serve.
2. **Prompt selection** — which prompt template to ship.
3. **Runtime selection** — which model, context and parameters to use for a given request.

Problems 2 and 3 are continuous. Every prompt edit, catalog update and traffic shift changes the answer to "is this good?". Metrics that only exist as a one-off benchmark run cannot track that.

## Define quality as a weighted objective before you measure anything

Before instrumenting anything, write down what "good" means for the system as a number you can compute per request. A typical weighted objective looks like this:

```
score = w1 * (1 - hallucination_rate)
      + w2 * latency_sla_pass        # 1 if p95 under target, else 0
      + w3 * (1 - normalized_cost)
      + w4 * (1 - safety_violation_rate)
```

The weights are a policy decision, not a research result. They should come from whoever owns the product trade-off — usually a combination of product, finance and engineering. Two properties matter more than the exact numbers:

- **The weights are written down and versioned.** If you cannot say which weights produced a given alert, you cannot debug it.
- **The objective is per-request, not per-batch.** A daily average hides the tail where the failures live.

A worked example, with arithmetic shown. Suppose you decide the following weights and targets:

- w1 = 0.4, w2 = 0.3, w3 = 0.2, w4 = 0.1
- Latency SLA: p95 under 500 ms
- Cost target: $0.01 per 1,000 output tokens
- Safety violations: any violation counts as a full miss for that request

For one request with hallucination_rate = 0.25, latency 620 ms, cost $0.008 per 1k tokens, no safety violation:

```
0.4 * (1 - 0.25)      = 0.300
0.3 * 0                 = 0.000   # latency SLA missed
0.2 * (1 - 0.008/0.01)  = 0.040
0.1 * (1 - 0)           = 0.100
total                   = 0.440
```

The same request with latency 420 ms scores 0.740. That difference is what makes the objective useful: it lets you compare a fast, slightly hallucinating answer against a slow, accurate one on a single axis. The weights are arbitrary in the sense that no external authority blesses them, but they are not arbitrary in the sense that they are explicit, reviewable and tunable against labelled data.

## Metric 1: grounding (a.k.a. hallucination rate)

Grounding asks: what fraction of the factual claims in the output can be traced back to the retrieved context? This is the metric that catches the failure mode where a model confidently cites policies, prices or SKUs that do not exist.

### How to measure it

The cheapest useful implementation is a two-step heuristic:

1. **Claim extraction** — split the answer into atomic factual claims.
2. **Grounding check** — for each claim, test whether it is supported by the retrieved context.

Both steps can be done with small local models. The code below uses `transformers` and `sentence-transformers`; pin the versions you test with, because embedding behavior changes between releases.

```python
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np

claim_extractor = pipeline(
    "text2text-generation",
    model="facebook/bart-large-cnn",
    device="cpu",
)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

def extract_claims(text: str) -> list[str]:
    """Extract atomic factual claims, one per line."""
    prompt = (
        "Extract the factual claims in this text. "
        "Return a list of claims only, one per line:\n" + text
    )
    output = claim_extractor(prompt, max_length=512, num_beams=4, early_stopping=True)
    return [c.strip() for c in output[0]["generated_text"].split("\n") if c.strip()]

def is_grounded(claim: str, context_embeddings, threshold: float = 0.85) -> bool:
    claim_embedding = embedding_model.encode(claim, convert_to_tensor=True)
    similarities = np.dot(context_embeddings, claim_embedding)
    return float(similarities.max()) >= threshold

def compute_hallucination_score(output: str, context: list[str]) -> float:
    """0 = every claim grounded, 1 = nothing grounded."""
    claims = extract_claims(output)
    if not claims:
        return 1.0
    context_embeddings = embedding_model.encode(context, convert_to_tensor=True)
    grounded = sum(1 for c in claims if is_grounded(c, context_embeddings))
    return 1.0 - (grounded / len(claims))
```

Two notes on correctness and cost:

- **Cache context embeddings.** The code above embeds the context once per call. If you evaluate many claims against the same context, embed the context once and reuse it. The naive version re-embeds the context on every request, which is the dominant cost.
- **The threshold is a tunable, not a truth.** A cosine threshold of 0.85 is a starting point. Calibrate it against a labelled sample: for each candidate threshold, compute precision and recall of "grounded" against human labels, and pick the point that matches your tolerance for false positives.

### What this metric misses

Embedding similarity measures *lexical and semantic closeness*, not entailment. A claim can be close to a context sentence and still be contradicted by it. This is the single most common source of false confidence in grounding metrics. Two mitigations:

- Add a second, stricter check (an entailment model or a small LLM judge) for claims near the threshold.
- Track the metric's *agreement with human labels* over time. If agreement drops, the metric has drifted, not the model.

## Metric 2: drift detection

Drift is the failure mode where your metric improves while user complaints rise. It happens because the inputs to the metric changed, not the model quality. Three drift channels matter:

- **Corpus drift** — the retrieval index was rebuilt with new documents. "Grounded in context" now means something different.
- **Prompt drift** — the template changed, tokens shifted, output length shifted.
- **Distribution drift** — user traffic moved to a new language, product line or intent.

### How to measure it

Instrument every request with four version identifiers and log them alongside the metric:

```
session_id, model_version, prompt_version, corpus_version, metric_value, timestamp
```

Then compute the metric *per version combination*, not globally. A dashboard that shows one hallucination line for the whole system will hide a regression in a single prompt version. A dashboard that shows one line per `(prompt_version, corpus_version)` will not.

For corpus drift specifically, a useful check is to re-run a fixed "golden set" of questions against the new corpus and compare the grounding score to the previous run. The golden set should be small (dozens of questions), stable, and manually verified. If the score moves by more than your noise floor, investigate before shipping the corpus update.

For prompt drift, track two derived quantities:

- **Output/input token ratio** per prompt version. A small preamble can change output length materially; measure it rather than assuming.
- **Claim count per answer.** If the model starts emitting more claims, grounding scores get noisier and safety classifiers see more surface area.

## Metric 3: cost and latency, measured per request

Cost and latency are not second-class citizens. In many deployments they are the metrics that determine whether the system survives. The mistake is to measure them as monthly aggregates rather than per request with version tags.

### What to instrument

- **Input tokens, output tokens, cached tokens** per request.
- **Time to first token** and **total latency** per request.
- **Retrieval latency** as a separate span from generation latency. Retries and timeouts in retrieval dominate tail latency more often than generation does.
- **Cache hit rate** for prompt prefixes and retrieval results.

### How to convert tokens to money

The arithmetic is straightforward once you have per-request token counts:

```
monthly_cost = sum_over_requests(output_tokens) * price_per_output_token
             + sum_over_requests(input_tokens)  * price_per_input_token
```

Do this per model version. If you serve two models behind a router, an aggregate cost number is meaningless. The useful comparison is cost per *successful* request, where "successful" is defined by the weighted objective above. A cheap model that fails 20% of requests is not cheap.

### A note on retry storms

A failure mode worth instrumenting explicitly: when retrieval latency spikes, the model server may time out and retry, which multiplies both cost and hallucination rate (retries often use degraded context). Track retries per request and alert on the retry rate, not just on latency. A circuit breaker that sheds load when a dependency's p95 exceeds a threshold is a standard mitigation and is cheaper than tuning the model.

## Metric 4: safety, including the false-positive side

Safety scanning is usually framed as catching harmful outputs. In practice, two failure modes matter equally:

- **False negatives** — harmful output passes the classifier, often because the classifier itself degrades under load.
- **False positives** — legitimate output is blocked, often because a domain-specific pattern (a discount code, an ID format) matches a harmful pattern.

Both should be measured. The false-positive rate is easy to ignore because it does not generate an incident, but it generates user frustration and support tickets.

### How to measure it

Run the classifier on a fixed labelled set of benign and harmful examples on a schedule, not just on live traffic. This gives you a stable estimate of classifier behavior independent of traffic mix. Log classifier latency alongside the verdict; if latency climbs, treat the classifier as degraded and consider a fallback path.

A practical mitigation for false positives is a post-processing allowlist for known-safe patterns in your domain, applied *after* the classifier. This is a targeted fix, not a general solution, and it should be reviewed whenever the domain patterns change.

## A minimal reference architecture

The components below are the ones that consistently earn their keep. Substitute equivalent managed services where convenient; the shape matters more than the vendor.

- **Prompt engine** — renders prompts from a versioned template. Every render is tagged with a `prompt_version`.
- **Model server** — serves the model with prefix caching enabled where supported. Emits per-request metrics.
- **Retrieval** — hybrid search over a versioned index. Emits retrieval latency and the `corpus_version` used.
- **Metric computation** — a lightweight per-request grounding score plus token and latency counters, computed close to the request.
- **Metric store** — a time-series store keyed by session, model, prompt and corpus versions.
- **Dashboard** — per-version views, not global aggregates.
- **Human review** — a small labelled sample per week, used to calibrate the automated metrics, not to replace them.

The single most important design choice is that **metrics are computed per request and stored with version tags**. Everything else is downstream of that.

## Human-in-the-loop: what it is for

Automated metrics drift. The fix is not to trust them more; it is to periodically check them against human labels on a small, stable sample. Two rules make this practical:

- **Sample deliberately, not randomly.** Include borderline cases (grounding score near the threshold), cases from each prompt version, and cases flagged by users. Random sampling wastes annotation budget on easy cases.
- **Use the labels to calibrate, not to score.** The output of annotation is a correction to the automated metric — a threshold, a prompt fix, or a note that the metric is no longer valid — not a headline number.

Active learning is worth trying once you have a working pipeline: send only borderline cases to annotators and keep the easy cases automated. This reduces annotation volume without changing the metric's calibration, provided you re-check the calibration periodically on a random sample.

## Failure modes and what they look like

**Latent metric drift.** The grounding score improves because the corpus changed, not because the model improved. *Symptom:* metric improves, user complaints rise. *Mitigation:* version the corpus, run a golden set on every corpus update, and alert on metric-vs-label agreement.

**Prompt drift.** A template change alters output length or claim density. *Symptom:* token ratio shifts, grounding noise increases. *Mitigation:* track output/input token ratio and claim count per prompt version; gate prompt changes on these.

**Cost explosion from retries.** A dependency slows down, the model server retries, cost and hallucination rate both rise. *Symptom:* latency p95 up, retry count up, cost up. *Mitigation:* circuit breaker on dependency latency; alert on retry rate.

**Safety false positives.** A domain pattern matches a harmful pattern. *Symptom:* legitimate outputs blocked; support tickets. *Mitigation:* post-classifier allowlist; measure false-positive rate on a labelled benign set.

**Traceability black hole.** You cannot answer "why did this answer change?". *Symptom:* rollbacks without a clear cause. *Mitigation:* propagate `session_id`, `model_version`, `prompt_version` and `corpus_version` through every span, and store them with every metric.

## When this approach is the wrong choice

- **Low volume.** Below roughly a thousand sessions a day, the operational cost of a time-series store, a streaming pipeline and annotation may exceed the benefit. A structured log and a weekly manual review are often enough.
- **Static prompts and static corpus.** If neither changes, a one-off evaluation is sufficient and continuous metrics add little.
- **No annotation budget.** Without any human labels, automated metrics cannot be calibrated and will drift silently. If this is the constraint, at minimum keep a small golden set that is manually verified once and re-run on every change.
- **Regulated environments.** Auditability requirements may exceed what general-purpose tracing provides. Plan for that explicitly rather than retrofitting.
- **Research and comparison only.** If you are comparing models in a notebook, use the standard benchmarks. The production machinery described here is for systems that serve users.

## A decision checklist

Before adding a metric, ask:

1. What user-visible failure does this metric predict?
2. What is the action if it crosses a threshold?
3. How is it calibrated against human labels, and how often?
4. Which version identifiers does it carry?
5. What is the cost of computing it per request, and does that cost scale with traffic?
6. What is the false-positive rate, and who is paged when it fires?

If you cannot answer 2 and 3, the metric will become noise.

## Do this in the next 30 minutes

Pick one production endpoint and add four fields to its structured log line: `prompt_version`, `model_version`, `corpus_version` and `output_token_count`. Deploy it. You now have the minimum data needed to attribute any future quality regression to a specific change, and every metric described above becomes computable from logs you already have.
