# Route, fine-tune, or scale: pick right

Most finetune route incidents trace back to a default nobody remembers choosing. The tutorials all show the happy path. This post covers what comes after the happy path.

## The problem this solves

You have a production LLM feature. It works, mostly. But latency is creeping up, costs are climbing, and a subset of queries keeps producing garbage. You've heard three pieces of advice: [fine-tune a smaller model,](/big-model-api-vs-fine-tune-the-2026-math/) route queries to different models based on complexity, or just pay for a bigger model and stop thinking about it. Each has a vocal fan club. Each also has a failure mode that shows up two weeks after you ship it.

The part that trips people up is that these aren't competing strategies — they're layers in a decision tree, and the order you evaluate them in determines whether you save money or burn a quarter on a fine-tune that underperforms a well-routed prompt. This post is a practical framework for solo founders and small teams who have to make the call themselves and live with the consequences. No ML platform team, no dedicated infra engineer. You, a terminal, and a budget.

I'll walk through a concrete setup: a support-ticket classifier that routes incoming messages to one of eight categories. It's a common shape — classification with a long tail of edge cases — and it exposes every tradeoff cleanly. By the end you'll have a working routing layer, a fine-tune decision checklist, and a cost model you can adapt to your own workload.

## Prerequisites and what you'll build

You need Python 3.11 or later, an OpenAI API key (or Anthropic — the pattern is portable), and roughly 200 labeled examples to start. You do not need a GPU. You do not need to know PyTorch. If your fine-tuning plan involves renting an A100, you're solving the wrong problem first.

We'll build three things:

1. A routing classifier that sends easy queries to a small model and hard ones to a large model.
2. A fine-tuning pipeline for the small model using 500–2,000 labeled examples.
3. An evaluation harness that tells you — with numbers, not vibes — whether the fine-tune actually beat the routed baseline.

The stack: `openai` Python SDK 1.30+, `pydantic` 2.7 for response validation, `pytest` 8.2 for tests, and `scikit-learn` 1.5 for the routing classifier. All boring, all stable, all installable in under two minutes. That's deliberate. The interesting decisions here are architectural, not tooling.

A note on cost baselines: as of 2026, a small model (roughly 8B parameters, API-hosted) typically runs $0.15–$0.60 per million input tokens, a mid-tier model $1–$3, and a frontier model $5–$15. Fine-tuning a small model usually costs $50–$400 depending on dataset size, plus inference at 1.5–2x the base rate. Keep those ranges in mind — they drive every decision below.

## Step 1 — set up the environment

Before writing any model code, build the evaluation set. This is the step everyone skips and the reason most fine-tunes fail. You cannot decide between routing and fine-tuning without a held-out set of at least 100 examples that neither approach has seen.

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

Why stratify? If you have 30 cancellation tickets out of 500, a random split can leave you with zero in the eval set. Then your accuracy number is a lie. This is a common trap — teams report 94% accuracy on an eval set that never contained the class they actually care about.

Install dependencies:

```bash
pip install 'openai>=1.30' 'pydantic>=2.7' 'scikit-learn>=1.5' 'pytest>=8.2'
export OPENAI_API_KEY="sk-..."
```

Pin your versions. The `openai` SDK had breaking changes between 0.x and 1.x that silently changed retry behavior. If you're on a version older than 1.0, upgrade before doing anything else.

## Step 2 — core implementation

The routing layer is the first thing to build, because it's cheap and it tells you whether you need a fine-tune at all. The idea: a lightweight classifier (logistic regression on embeddings, or a small model call) assigns each query a difficulty score. Easy queries go to the small model. Hard ones go to the frontier model.

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

SMALL = "gpt-4o-mini"      # ~$0.15/M input tokens
LARGE = "gpt-4o"           # ~$2.50/M input tokens

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

The threshold (0.85 here) is the single most important knob. Set it too low and you route everything to the small model, including the queries it gets wrong. Set it too high and you're paying frontier prices for easy work. Tune it against your eval set: sweep 0.70 to 0.95 and plot accuracy against cost.

Typical numbers from a setup like this on a 500-example support dataset: routing sends 60–75% of traffic to the small model, cuts blended cost by 55–65%, and loses 1–3 percentage points of accuracy versus sending everything to the large model. That trade is usually worth it. If it isn't — if that 2% is your cancellation-detection accuracy — that's your signal to fine-tune.

## Step 3 — handle edge cases and errors

Three failure modes show up reliably in production routing systems.

**The out-of-distribution query.** A user pastes a 4,000-token log dump into the ticket field. Your embedding classifier has never seen anything like it, returns low confidence, and routes to the large model — which then blows your context budget. Guard with a token-count check before routing:

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

**The label drift.** Your categories change. Someone adds "refund" as a ninth class. Your fine-tuned model now confidently mislabels every refund ticket as "billing" because it has never seen the label. This is the most expensive failure mode because it's silent. Mitigation: log every low-confidence prediction to a review queue, and re-evaluate weekly. A fine-tuned model needs retraining when labels change; a routed prompt just needs a new system message.

**The retry storm.** The `openai` SDK retries on 429 and 5xx by default. If your routing layer also retries, you can multiply your request volume by 6x during an outage. Set `max_retries=2` explicitly and handle `RateLimitError` at the routing layer, not just the call layer. A common trap here is that the SDK's default retry uses exponential backoff with jitter, but if you're running 20 concurrent workers, they all back off to roughly the same schedule and hammer the API together.

Here's a comparison of the three strategies across the dimensions that matter:

| Dimension | Route (small + large) | Fine-tune small | Just use large |
|---|---|---|---|
| Upfront cost | $0 | $50–$400 + 1–2 days | $0 |
| Inference cost / 1M tokens | ~$1.10 blended | ~$0.30 | ~$2.50 |
| Accuracy vs large baseline | −1 to −3 pts | −0.5 to +1 pt | baseline |
| Latency p50 | 400–700 ms | 250–450 ms | 600–900 ms |
| Label changes | edit prompt | retrain + redeploy | edit prompt |
| Ops burden | low | medium | none |
| Best when | traffic > 100k/mo | labels stable, > 500k/mo | traffic < 50k/mo |

The table makes the decision legible: routing wins on flexibility, fine-tuning wins on unit cost at volume, and the large model wins on time-to-ship. Notice that fine-tuning's accuracy advantage over routing is small — often under 2 points. That's why the order matters: route first, measure, then fine-tune only the tier that routing can't handle well.

## Step 4 — add observability and tests

You cannot tune what you don't measure. Log four things per request: the routed tier, the confidence score, the model used, and whether the output matched the expected label (when you have a label).

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

Then write the test that actually matters — not a unit test of the router, but a regression test against your eval set:

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
    # Typical routed baseline lands 0.88–0.93 on this dataset shape
    assert acc >= 0.88, f"accuracy dropped to {acc:.3f}"
```

Run this in CI on every prompt change. Prompts are code; treat them like code. A one-word change to the system message can move accuracy by 5 points, and you'll never notice without a regression test.

For the fine-tune path, the evaluation is the same harness with a different model behind `classify()`. That's the point of building the harness first — it makes the fine-tune decision falsifiable. If the fine-tuned model scores 0.91 and the routed baseline scores 0.90, you've paid $300 and two days for one point. If it scores 0.95, you have a real case.

## Real results from running this

Here's what the numbers typically look like on a support-classification workload of 200,000 requests per month, with an average query of 150 tokens and 20 output tokens. These are representative figures for this workload shape, not measurements from a specific deployment.

| Approach | Monthly cost | p50 latency | Accuracy |
|---|---|---|---|
| All large model | ~$780 | 720 ms | 0.92 |
| Routed (70/30 split) | ~$310 | 480 ms | 0.90 |
| Fine-tuned small | ~$95 | 310 ms | 0.91 |
| Fine-tuned + routed tail | ~$140 | 340 ms | 0.92 |

The interesting row is the last one. Fine-tuning the small model and routing only the genuinely ambiguous 15% to the large model gets you the same accuracy as the all-large baseline at roughly 18% of the cost. That's the configuration most teams should aim for — but you can only find it by building routing first, measuring, and then fine-tuning the tier that's actually failing.

The failure mode to watch: teams fine-tune first because it feels like the "real" engineering work, then discover their labels are inconsistent. If two annotators disagree on 15% of tickets, no fine-tune will beat a well-prompted large model, because the model is learning to reproduce the disagreement. Check inter-annotator agreement before you spend a dollar on training. A Cohen's kappa below 0.7 means fix your labels first.

## Common questions and variations

### Can I fine-tune without a GPU?

Yes, for small models via hosted APIs. OpenAI, Anthropic, and several open providers offer fine-tuning as a managed service — you upload a JSONL file, they train, you get a model ID back. You do not need local hardware. If you're fine-tuning a 7B–8B model, the hosted route costs $50–$400 and finishes in hours. Only reach for local training (and a GPU) if you need a model under 3B parameters for on-device inference, or if data residency rules forbid sending training data to a third party.

### How many examples do I need to fine-tune?

The honest answer is "more than you think, fewer than the papers suggest." For classification with 8 classes, 300–500 examples per class is a reasonable starting point — so 2,400–4,000 total. Below 100 per class, you're usually better off with few-shot prompting on a large model. The quality of labels matters more than the count: 500 clean examples beat 5,000 noisy ones every time. If you can't get 100 clean examples per class, that's a signal your taxonomy is too granular.

### When should I just use the biggest model?

When your monthly volume is under 50,000 requests, or when the task is genuinely open-ended (summarization, code generation, multi-step reasoning). Routing and fine-tuning both add operational surface area — a classifier to maintain, a training pipeline, a retraining cadence. Below 50k requests/month, the cost savings from routing are typically $200–$400/month. If that's less than two hours of your time, just use the large model and ship the feature. You can always add routing later; it's not a one-way door.

### Does routing add meaningful latency?

The classifier call itself adds 20–60 ms if you're using a local embedding model, or 150–300 ms if you're calling an embedding API. For a 400 ms baseline, that's a 5–15% latency increase — usually invisible to users. If your SLA is tight, run the embedding model locally with `sentence-transformers` 3.0; it's a 90 MB download and runs on CPU in under 30 ms for short queries.

## Where to go from here

The framework, compressed: build the eval set first, route second, fine-tune third, and only if the numbers justify it. Every step is reversible except the fine-tune — a trained model is a deployment artifact you have to maintain, retrain, and eventually retire. Routing is a config change. Prompts are strings. Treat the irreversible step with the caution it deserves.

Your next 30 minutes: open your production logs and compute two numbers — your monthly request volume and your p50 latency. If volume is under 50,000, stop reading about fine-tuning and go improve your prompt. If it's over 100,000, write the stratified train/eval split from Step 1 against 200 of your real queries, run your current prompt against the eval set, and record the baseline accuracy. That single number is what every subsequent decision gets measured against, and you can have it before your coffee goes cold.


---

### About this article

**Written by:** [Kubai Kevin](/about/) — software developer based in Nairobi, Kenya, with 10+ years building production systems in fintech and AI.

**How this article was produced:** This site uses an automated LLM pipeline designed and maintained by the author. Topics are selected from real production experience. Drafts pass automated quality gates (minimum length, uniqueness, concrete metrics, versioned tools, code samples, absence of filler). Individual line-by-line human editing is not performed on every post before publication. Specific numbers, benchmarks and cost figures are illustrative; verify them against current official documentation before production use.

**Corrections:** Report errors via the [contact page](/contact/). Corrections are applied promptly.

**Last generated:** September 2026
