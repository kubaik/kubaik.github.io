# AI observability: the metrics that matter

## Why traditional monitoring is not enough for model-backed services

Standard application monitoring treats a service as a black box with a well-defined contract: a request arrives, work happens, and either a response or an error comes back. Metrics like request rate, error rate, and latency percentiles describe that contract well. Logs and distributed traces explain why a particular request failed.

Model-backed services break that assumption in three specific ways.

First, **correctness is not binary**. An endpoint can return HTTP 200 with a fluent, well-formatted answer that is wrong, off-policy, or subtly degraded. Latency and error rate stay flat while answer quality falls. Nothing in a traditional dashboard moves.

Second, **the input distribution is not stationary**. Users change behavior, upstream systems change formats, and prompt templates get edited. A model trained on one distribution can silently start receiving another. This is input drift, and it is invisible to counters and gauges unless something explicitly measures it.

Third, **a single logical request fans out**. One user prompt can trigger an embedding call, a retrieval step, a reranker, a generation call, and a post-processing pass. Each stage has its own latency, cost, and failure characteristics. A single end-to-end p99 hides which stage is responsible.

None of this means replacing traditional monitoring. Red metrics, structured logs, and distributed traces remain the foundation. AI observability is an additional layer that measures the model's inputs, outputs, and internal signals.

## The three pillars of AI observability

Beyond metrics, logs, and traces, model-backed systems benefit from three additional signal families.

**Drift vectors.** A numeric measure of how far the current input distribution has moved from a reference distribution captured at training or validation time. The most common implementation embeds inputs with the same embedding model used downstream, then computes a distance between the current window's centroid and the reference centroid.

**Confidence distributions.** Summaries of the model's own probability estimates over generated tokens. For open-weight models, these are available directly from the softmax. For hosted APIs, they may be exposed as logprobs or not at all. When unavailable, substitute proxy signals: output length distribution, refusal rate, retry rate, or a small classifier scoring outputs.

**Feature attribution.** A per-input-token estimate of how much each token influenced the output. This is the most expensive signal and the most prone to being misread. Treat it as a diagnostic tool for offline investigation, not a hot-path metric.

The useful framing is that traditional monitoring describes external behavior, and these three describe the relationship between inputs, outputs, and the model's internal state.

## Instrumenting drift, confidence, and attribution

The following is a minimal, framework-agnostic implementation. It assumes an embeddings interface and an LLM interface that can return per-token probabilities. Adapt the interfaces to your stack.

```bash
pip install numpy scikit-learn prometheus-client
```

The drift observer keeps a sliding window of recent embeddings and compares the window centroid to a reference set.

```python
from typing import List
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


class DriftObserver:
    """Tracks centroid distance between a sliding window and a reference set."""

    def __init__(self, ref_embeddings: np.ndarray, window_size: int = 100):
        if ref_embeddings.ndim != 2:
            raise ValueError("ref_embeddings must be 2-D (n_samples, dim)")
        self.ref_centroid = ref_embeddings.mean(axis=0, keepdims=True)
        # Distance from each reference point to the reference centroid gives
        # a scale for what "normal" spread looks like.
        self.ref_spread = float(
            np.mean(euclidean_distances(ref_embeddings, self.ref_centroid))
        )
        self.window: List[np.ndarray] = []
        self.window_size = window_size

    def observe(self, embedding: np.ndarray) -> float:
        """Return normalized drift: window-centroid distance / reference spread."""
        self.window.append(embedding)
        if len(self.window) > self.window_size:
            self.window.pop(0)
        if len(self.window) < 2 or self.ref_spread == 0.0:
            return 0.0
        current_centroid = np.mean(self.window, axis=0, keepdims=True)
        dist = float(euclidean_distances(current_centroid, self.ref_centroid)[0][0])
        return dist / self.ref_spread
```

Two details matter. The normalization by reference spread makes the threshold portable across embedding models: a value near 1.0 means the current window is as far from the reference centroid as a typical reference point is, which is a reasonable "something changed" signal. And the window is a sliding buffer, not a reservoir, so memory is bounded.

Wrapping an embeddings client is straightforward.

```python
from typing import List
import numpy as np


class ObservedEmbeddings:
    def __init__(self, base, drift_observer: DriftObserver):
        self.base = base
        self.drift_observer = drift_observer

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vectors = self.base.embed_documents(texts)
        for v in vectors:
            self.drift_observer.observe(np.asarray(v, dtype=np.float32))
        return vectors

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]
```

The confidence observer accumulates per-token probabilities and exposes summary statistics rather than the raw histogram. This is the single most important design decision for keeping cardinality under control.

```python
from collections import defaultdict
from typing import Dict, Iterable, List
import numpy as np


class ConfidenceObserver:
    def __init__(self, max_tokens_tracked: int = 512):
        self.probs: List[float] = []
        self.max_tokens_tracked = max_tokens_tracked

    def observe_token(self, probability: float) -> None:
        if len(self.probs) < self.max_tokens_tracked:
            self.probs.append(float(probability))

    def reset(self) -> None:
        self.probs = []

    def summary(self) -> Dict[str, float]:
        if not self.probs:
            return {"p10": 0.0, "p50": 0.0, "avg": 0.0, "min": 0.0}
        arr = np.asarray(self.probs, dtype=np.float64)
        return {
            "p10": float(np.percentile(arr, 10)),
            "p50": float(np.percentile(arr, 50)),
            "avg": float(arr.mean()),
            "min": float(arr.min()),
        }
```

The low percentile (p10) is often more informative than the average. A model that is confident on most tokens but occasionally very uncertain on a few tends to show a stable average and a falling p10. That pattern is a useful early warning.

For attribution, the practical approach is to compute it out of band. Raw input gradients are dominated by positional and token-frequency effects and are not a reliable attribution signal on their own. Integrated gradients, which integrate gradients along a path from a baseline input to the real input, give a more faithful estimate, at the cost of many forward and backward passes per example.

```python
def integrated_gradients(model_fn, input_ids, baseline_ids, steps: int = 32):
    """
    model_fn: callable that returns a scalar score for a batch of token id tensors.
    Returns per-token attribution scores, shape matching input_ids.
    """
    import torch

    input_ids = torch.as_tensor(input_ids)
    baseline_ids = torch.as_tensor(baseline_ids)
    if input_ids.shape != baseline_ids.shape:
        raise ValueError("baseline must match input shape")

    total_grad = torch.zeros_like(input_ids, dtype=torch.float32)
    for step in range(1, steps + 1):
        alpha = step / steps
        # Interpolate in embedding space in a real implementation; here we
        # interpolate token ids only as a stand-in for a differentiable input.
        interp = baseline_ids + alpha * (input_ids - baseline_ids)
        interp = interp.clone().float().requires_grad_(True)
        score = model_fn(interp)
        grad = torch.autograd.grad(score, interp)[0]
        total_grad += grad

    avg_grad = total_grad / steps
    return (input_ids - baseline_ids).float() * avg_grad
```

Two caveats. First, interpolating token IDs is not meaningful for a real transformer; the interpolation must happen in embedding space, and the baseline should be a token such as a padding token with a zero embedding. Second, integrated gradients on a long prompt against a large model is expensive. Compute it offline on sampled traffic, not on the request path.

## What to measure, and how to know it is working

A useful instrumentation plan answers three questions: what changes when the system degrades, how quickly can that be detected, and what is the false positive rate.

For drift, the instrumentation is the embedding call plus a centroid computation. To validate it, hold out a slice of known-good production traffic and a slice of deliberately shifted traffic (for example, inputs from a different locale or a different upstream service). Measure the distribution of the drift statistic on each slice and pick a threshold that separates them with an acceptable false positive rate.

For confidence, the instrumentation is the per-token probability stream. To validate it, correlate low-confidence outputs with a downstream quality label: human review, a task-specific evaluator, or a business outcome such as a follow-up contact rate. If low confidence does not predict low quality, the confidence signal is not useful for your task and should be dropped.

For attribution, the instrumentation is an offline job. To validate it, use a small set of prompts where the relevant input tokens are known by construction, and check whether the top-attributed tokens match. If they do not, the attribution method is not working for your model and should not be trusted.

The general principle: every new signal should be validated against an independent ground truth before it is put behind an alert.

## Cardinality, cost, and the failure modes that actually bite

The dominant operational risk in AI observability is metric cardinality. A per-request metric with a `model_version` label multiplies by the number of deployed variants. Add a `prompt_template_id` label and it multiplies again. Add a per-token dimension and it explodes.

Concrete guidance:

- **Never label a metric with a token ID, a prompt hash, or a user ID.** These belong in logs or a columnar event store, not in a time-series database.
- **Emit summaries, not distributions, as metrics.** p10, p50, and avg confidence per request are three time series. A full histogram with 100 buckets is 100 series per label combination.
- **Keep raw events in an event store.** A columnar store such as ClickHouse handles high-cardinality event data far better than a metrics system. Metrics are for alerting; events are for investigation.
- **Cap label cardinality at the collector.** Most metric pipelines allow dropping or hashing high-cardinality labels before ingestion. Use that.

The second failure mode is **false positives from a bad reference distribution**. If the reference embeddings are drawn from a noisy or unrepresentative dataset, the drift statistic will fire constantly and the team will learn to ignore it. Rebuild the reference set periodically from recent known-good traffic, and keep a record of when it was last rebuilt.

The third failure mode is **signal that does not predict anything**. Confidence and attribution are easy to compute and easy to misinterpret. Before adding an alert, verify that the signal correlates with an outcome the team cares about. A signal that fires but never precedes a real problem is worse than no signal, because it consumes attention.

The fourth failure mode is **hot-path cost**. Drift computation is cheap (a centroid and a distance). Confidence summarization is cheap. Attribution is not. Keep attribution offline. If a signal cannot fit in the latency budget, it should not be on the request path.

The fifth failure mode is **version skew in labels**. When multiple model variants serve traffic, every event must carry the exact variant identifier, not just a semantic version. Two variants that share a version string but differ in quantization or decoding parameters will produce merged metrics that describe neither.

The sixth failure mode is **privacy leakage through attribution or raw prompt logging**. Attribution scores and prompt text can both contain user data. Apply the same data handling rules to observability data as to application data: redaction, retention limits, and access controls.

## A decision checklist

Before adding any AI observability signal, answer these questions.

1. **What failure does this signal detect that current monitoring misses?** If there is no concrete failure, do not add the signal.
2. **What is the ground truth for this signal?** If there is no independent way to verify it, it cannot be trusted.
3. **What is the cardinality cost?** Count the label combinations and multiply by the number of series per label set.
4. **What is the hot-path latency cost?** Measure it, do not estimate it.
5. **Who acts on this signal, and what do they do?** A signal with no runbook is noise.
6. **When is the reference or threshold rebuilt?** Stale references cause false positives.
7. **What is the retention and privacy policy for the underlying data?**

If a proposed signal fails questions 1, 2, or 5, it should not be built.

## When not to build this

AI observability is not free, and it is not always worth the cost.

**Static prompts, frozen models, low-stakes outputs.** If the prompt template has not changed in a year and the model is pinned, drift is unlikely and the cost of detection exceeds the benefit. Plain request-level metrics and sampled output review are sufficient.

**Sub-100 ms latency budgets.** Drift and confidence instrumentation can add milliseconds, but attribution and any per-token work will not fit. Restrict to request-level metrics and sample drift computation on a small fraction of traffic.

**Teams without event-store operational experience.** Running a columnar event store and a message bus is real operational work. A single metrics system plus structured logs is easier to maintain and catches hard failures. Add the event store when the investigation workflow actually demands it.

**Quarterly model updates.** Drift accumulates between updates. If the model is retrained quarterly and the input distribution is stable, a periodic offline evaluation is usually enough.

The decision is not "traditional versus AI observability." It is which additional signals, at what cost, answer a question the team actually has.

## What to do in the next 30 minutes

Pick one production endpoint that calls a model. Add a single counter for the number of requests whose output confidence summary falls below a threshold you choose from a sample of recent traffic, and log the request ID and the confidence summary for those requests. Run it for a day, then compare the flagged requests against a small human review sample. If low confidence predicts low quality, the signal is worth expanding. If it does not, you have saved yourself the cost of building a pipeline around a signal that does not work for your task.

## FAQ

**How do I add drift tracking to an existing service without a rewrite?**

Wrap the embedding client with a thin observer that computes a centroid distance over a sliding window, as shown above. This is additive: no changes to the model, the prompt, or the request path beyond the wrapper. Emit the drift statistic as a single gauge and log the window contents only when the statistic crosses a threshold.

**Why are raw input gradients a poor attribution signal?**

Raw gradients at the input layer are sensitive to positional encodings, token frequency, and the specific point in input space where they are evaluated. They are not a faithful measure of how much each token contributed to the output. Integrated gradients, which average gradients along a path from a baseline, are more reliable, at the cost of many forward and backward passes.

**What is the smallest useful observability setup for a single-GPU development environment?**

A metrics endpoint exposing request count, error count, latency, and a confidence summary, plus structured logs containing the prompt hash, model variant, and output length. Add drift only when you have a reference embedding set and a stable input distribution to compare against. Skip attribution entirely until you have a specific question it can answer.

**When should a team move from traditional monitoring to adding AI observability signals?**

When a model-backed feature is affecting a business outcome and the current dashboards cannot explain a change in that outcome. Common triggers: model updates more often than monthly, a dynamic prompt library, or a quality regression that was not visible in latency or error rate.
