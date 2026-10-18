# Route, fine-tune, or scale: pick right

Most production LLM incidents trace back to a default nobody remembers choosing. Tutorials show the happy path; this article covers what comes after it.

## The problem this solves

A production LLM feature works, mostly. But latency creeps up, costs climb, and a subset of queries keeps producing garbage. Three pieces of advice circulate: fine-tune a smaller model, route queries to different models based on complexity, or pay for a bigger model and stop thinking about it. Each has a vocal fan club. Each also has a failure mode that shows up two weeks after shipping.

These are not competing strategies. They are layers in a decision tree, and the order of evaluation determines whether a team saves money or burns a quarter on a fine-tune that underperforms a well-routed prompt. This article lays out a practical framework for solo founders and small teams who make the call themselves and live with the consequences — no ML platform team, no dedicated infra engineer.

The running example is a support-ticket classifier that routes incoming messages to one of eight categories. It is a common shape — classification with a long tail of edge cases — and it exposes every tradeoff cleanly. By the end there is a working routing layer, a fine-tune decision checklist, and a cost model to adapt to any workload.

## Prerequisites and what gets built

Requirements: Python 3.11 or later, an API key for a hosted LLM provider, and roughly 200 labeled examples to start. No GPU. No PyTorch. If a fine-tuning plan involves renting an A100, the wrong problem is being solved first.

Three artifacts:

1. A routing classifier that sends easy queries to a small model and hard ones to a large model.
2. A fine-tuning pipeline for the small model using 500–2,000 labeled examples.
3. An evaluation harness that reports, with numbers rather than vibes, whether the fine-tune actually beat the routed baseline.

The stack: the `openai` Python SDK (1.x), `pydantic` 2.x for response validation, `pytest` 8.x for tests, and `scikit-learn` 1.x for the routing classifier. All boring, all stable. That is deliberate — the interesting decisions here are architectural, not tooling.

A note on cost baselines. Published list prices for hosted models vary by provider and change frequently. Rather than quoting figures that go stale, treat pricing as a parameter: pull current per-million-token rates from the provider's pricing page, and plug them into the cost model below. Fine-tuning typically carries an upfront training charge plus an inference rate above the base model's. Those two numbers drive every decision that follows.

## Step 1 — set up the environment

Before writing any model code, build the evaluation set. This is the step everyone skips and the reason most fine-tunes fail. A decision between routing and fine-tuning requires a held-out set of at least 100 examples that neither approach has seen.

```python
# eval_setup.py
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

LABELS = ["billing", "technical", "account", "feature_request",
          "bug_report", "cancellation", "integration", "other"]

def load_and_split(path: str = "tickets.jsonl", seed: int = 42):
    records = [json.loads(line) for line in Path(path).read_text().splitlines()]
    # Stratify so rare classes (cancellation, integration) appear in both splits
    train, eval_ = train_test_split(
        records, test_size=0.2, random_state=seed,
        stratify=[r["label"] for r in records]
    )
    return train, eval_

if __name__ == "__main__":
    train, eval_ = load_and_split()
    print(f"train={len(train)} eval={len(eval_)}")
    # Typical output for a 500-example set: train=400 eval=100
```

Stratification matters. With 30 cancellation tickets out of 500, a random split can leave zero in the eval set. The accuracy number is then a lie. A common trap: teams report 94% accuracy on an eval set that never contained the class they actually care about.

Install dependencies:

```bash
pip install 'openai>=1.30' 'pydantic>=2.7' 'scikit-learn>=1.5' 'pytest>=8.2'
export OPENAI_API_KEY="sk-..."
```

Pin versions. The `openai` SDK had breaking changes between 0.x and 1.x that silently altered retry behavior. Anything below 1.0 should be upgraded before proceeding.

## Step 2 — core implementation

Build the routing layer first, because it is cheap and it reveals whether a fine-tune is needed at all. The idea: a lightweight classifier (logistic regression on embeddings, or a small model call) assigns each query a difficulty score. Easy queries go to the small model; hard ones go to the frontier model.

```python
# router.py
import numpy as np
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()

class RouteDecision(BaseModel):
    tier: str  # "small" | "large"
    confidence: float

# Embedding-based difficulty heuristic: distance from decision boundary
# in a pre-trained classifier trained on your labeled examples.
def route(query: str, clf, embedder) -> RouteDecision:
    vec = embedder.encode([query])
    proba = clf.predict_proba(vec)[0]
    top = float(np.max(proba))
    if top >= 0.85:
        return RouteDecision(tier="small", confidence=top)
    return RouteDecision(tier="large", confidence=top)

SMALL = "gpt-4o-mini"      # cheap tier
LARGE = "gpt-4o"           # frontier tier

def classify(query: str, clf, embedder) -> str:
    decision = route(query, clf, embedder)
    model = SMALL if decision.tier == "small" else LARGE
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Classify into one of 8 categories. Reply with the label only."},
            {"role": "user", "content": query},
        ],
        temperature=0.0,
        max_tokens=8,
    )
    return resp.choices[0].message.content.strip().lower()
```

The threshold (0.85 here) is the single most important knob. Set it too low and everything routes to the small model, including the queries it gets wrong. Set it too high and frontier prices are paid for easy work. Tune it against the eval set: sweep 0.70 to 0.95 and plot accuracy against cost.

What to expect from a setup like this on a support-classification dataset: routing typically sends the majority of traffic to the small model, cuts blended cost substantially, and loses a small number of accuracy points versus sending everything to the large model. Whether that trade is worth it depends on which classes lose accuracy. If cancellation detection is the class that degrades, that is the signal to fine-tune.

## Step 3 — handle edge cases and errors

Three failure modes show up reliably in production routing systems.

**The out-of-distribution query.** A user pastes a 4,000-token log dump into the ticket field. The embedding classifier has never seen anything like it, returns low confidence, and routes to the large model — which then blows the context budget. Guard with a token-count check before routing:

```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

def safe_route(query: str, clf, embedder, max_tokens: int = 2000):
    n = len(enc.encode(query))
    if n > max_tokens:
        # Truncate, or route to a summarization step first
        query = enc.decode(enc.encode(query)[:max_tokens])
    return route(query, clf, embedder)
```

**The label drift.** Categories change. Someone adds "refund" as a ninth class. The fine-tuned model now confidently mislabels every refund ticket as "billing" because it has never seen the label. This is the most expensive failure mode because it is silent. Mitigation: log every low-confidence prediction to a review queue and re-evaluate weekly. A fine-tuned model needs retraining when labels change; a routed prompt just needs a new system message.

**The retry storm.** The `openai` SDK retries on 429 and 5xx by default. If the routing layer also retries, request volume can multiply during an outage. Set `max_retries=2` explicitly and handle `RateLimitError` at the routing layer, not just the call layer. The SDK's default retry uses exponential backoff with jitter, but with many concurrent workers, they all back off to roughly the same schedule and hammer the API together.

A comparison of the three strategies across the dimensions that matter:

| Dimension | Route (small + large) | Fine-tune small | Just use large |
|---|---|---|---|
| Upfront cost | $0 | training charge + 1–2 days | $0 |
| Inference cost | blended rate | base rate + fine-tune premium | frontier rate |
| Accuracy vs large baseline | small loss | roughly on par or better | baseline |
| Latency p50 | between the two tiers | lowest | highest |
| Label changes | edit prompt | retrain + redeploy | edit prompt |
| Ops burden | low | medium | none |
| Best when | moderate-to-high traffic | labels stable, high volume | low traffic |

The table makes the decision legible: routing wins on flexibility, fine-tuning wins on unit cost at volume, and the large model wins on time-to-ship. Fine-tuning's accuracy advantage over routing is often small — under a couple of points. That is why the order matters: route first, measure, then fine-tune only the tier that routing cannot handle well.

## Step 4 — add observability and tests

Nothing can be tuned that is not measured. Log four things per request: the routed tier, the confidence score, the model used, and whether the output matched the expected label (when a label exists).

```python
# observability.py
import logging, time
from dataclasses import dataclass, asdict

log = logging.getLogger("router")

@dataclass
class RouteEvent:
    query_hash: str
    tier: str
    confidence: float
    model: str
    latency_ms: int
    cost_usd: float
    correct: bool | None

def log_route(event: RouteEvent):
    log.info("route", extra={"event": asdict(event)})
    # Ship to your metrics backend. Datadog, Grafana Cloud, whatever you use.
```

Then write the test that actually matters — not a unit test of the router, but a regression test against the eval set:

```python
# test_routing.py
import pytest
from eval_setup import load_and_split
from router import classify

def test_accuracy_above_threshold():
    _, eval_ = load_and_split()
    correct = 0
    for row in eval_:
        pred = classify(row["text"], clf, embedder)
        correct += int(pred == row["label"])
    acc = correct / len(eval_)
    # Set the threshold to your measured baseline minus a small tolerance
    assert acc >= 0.88, f"accuracy dropped to {acc:.3f}"
```

Run this in CI on every prompt change. Prompts are code; treat them like code. A one-word change to the system message can move accuracy by several points, and nobody notices without a regression test.

For the fine-tune path, the evaluation is the same harness with a different model behind `classify()`. That is the point of building the harness first — it makes the fine-tune decision falsifiable. If the fine-tuned model scores 0.91 and the routed baseline scores 0.90, $300 and two days were spent for one point. If it scores 0.95, there is a real case.

## Cost math, worked through

The table below is illustrative, not measured. It assumes 200,000 requests per month, an average query of 150 input tokens and 20 output tokens, and these hypothetical per-million-token rates: small model $0.15 input / $0.60 output, large model $2.50 input / $10.00 output, fine-tuned small model $0.30 input / $1.20 output (the fine-tune premium).

Per request, input tokens cost: 150 / 1,000,000 × rate. Output tokens cost: 20 / 1,000,000 × rate.

- Large model per request: 150e-6 × $2.50 = $0.000375 input, plus 20e-6 × $10.00 = $0.0002 output. Total ≈ $0.000575. At 200,000 requests: **≈ $115/month**.
- Small model per request: 150e-6 × $0.15 = $0.0000225, plus 20e-6 × $0.60 = $0.000012. Total ≈ $0.0000345. At 200,000 requests: **≈ $6.90/month**.
- Fine-tuned small model per request: 150e-6 × $0.30 = $0.000045, plus 20e-6 × $1.20 = $0.000024. Total ≈ $0.000069. At 200,000 requests: **≈ $13.80/month**.
- Routed 70/30 split (70% small, 30% large): 0.7 × $6.90 + 0.3 × $115 ≈ $4.83 + $34.50 = **≈ $39.30/month**.

The numbers are small because the token counts are small. The point of the exercise is the ratio: routing cuts cost by roughly two-thirds versus all-large, and fine-tuning cuts it further. Whether that matters depends on the absolute volume. At 200,000 requests/month, the difference between all-large and routed is about $76/month — likely less than an hour of engineering time. At 20 million requests/month, the same ratio is about $7,600/month, which changes the calculus.

Run this arithmetic with your own token counts and your provider's current rates before committing to any architecture. The decision is a function of volume, not of strategy preference.

## Common questions and variations

### Can a small model be fine-tuned without a GPU?

Yes, for small models via hosted APIs. Several providers offer fine-tuning as a managed service — upload a JSONL file, they train, you get a model ID back. No local hardware required. Hosted fine-tuning of a 7B–8B model typically costs tens to low hundreds of dollars and finishes in hours. Local training with a GPU only becomes necessary for models under roughly 3B parameters for on-device inference, or when data residency rules forbid sending training data to a third party.

### How many examples are needed to fine-tune?

More than intuition suggests, fewer than the papers imply. For classification with 8 classes, a few hundred examples per class is a reasonable starting point. Below roughly 100 per class, few-shot prompting on a large model is usually the better bet. Label quality matters more than count: 500 clean examples beat 5,000 noisy ones. If 100 clean examples per class cannot be assembled, that is a signal the taxonomy is too granular.

### When should the biggest model just be used directly?

When monthly volume is under roughly 50,000 requests, or when the task is genuinely open-ended (summarization, code generation, multi-step reasoning). Routing and fine-tuning both add operational surface area — a classifier to maintain, a training pipeline, a retraining cadence. Below that threshold, the cost savings from routing are typically small relative to engineering time. Adding routing later is easy; it is not a one-way door.

### Does routing add meaningful latency?

The classifier call adds 20–60 ms with a local embedding model, or 150–300 ms with an embedding API. Against a 400 ms baseline, that is a 5–15% latency increase — usually invisible to users. If the SLA is tight, run the embedding model locally; a small sentence-transformer model is a ~90 MB download and runs on CPU in under 30 ms for short queries.

## Decision checklist

Before committing to any architecture, answer these in order:

1. **What is the monthly request volume?** Under ~50k, improve the prompt and use the large model. Over ~100k, routing is worth building.
2. **What is the baseline accuracy on a stratified eval set?** Without this number, no subsequent decision is falsifiable.
3. **Which classes lose accuracy under routing?** If the loss is concentrated in a class that matters, fine-tune. If it is spread thin, accept it.
4. **Are the labels consistent?** Check inter-annotator agreement. A Cohen's kappa below 0.7 means fixing labels first, because no fine-tune can beat a well-prompted large model when the training data encodes disagreement.
5. **How often do labels change?** Frequently changing labels favor routing; a stable taxonomy favors fine-tuning.
6. **What is the retraining cadence?** A fine-tuned model is a deployment artifact to maintain, retrain, and eventually retire. Budget for that.

## Where to go from here

The framework, compressed: build the eval set first, route second, fine-tune third, and only if the numbers justify it. Every step is reversible except the fine-tune — a trained model is a deployment artifact with a maintenance cost. Routing is a config change. Prompts are strings. Treat the irreversible step with the caution it deserves.

The next 30 minutes: open production logs and compute two numbers — monthly request volume and p50 latency. If volume is under 50,000, stop reading about fine-tuning and improve the prompt. If it is over 100,000, write the stratified train/eval split from Step 1 against 200 real queries, run the current prompt against the eval set, and record the baseline accuracy. That single number is what every subsequent decision gets measured against.
