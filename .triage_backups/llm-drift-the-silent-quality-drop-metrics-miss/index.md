# LLM drift: the silent quality drop metrics miss

## The failure mode: flat dashboards, falling quality

A common and frustrating pattern in production LLM systems is this: latency is unchanged, cost per token is unchanged, error rate is zero, and yet users are complaining that answers are worse. Shorter summaries. Less relevant retrieval. Instructions followed less reliably.

This is silent quality degradation. It is not a model outage. Nothing throws an exception. The system is behaving exactly as configured — the configuration just changed underneath it.

The reason standard observability misses this is structural. Dashboards are built to catch high-frequency, low-cardinality events: request counts, p50/p99 latency, HTTP status codes, token spend. Silent degradation is the opposite: a one-time config mutation (a prompt template edit, an SDK bump that swaps the tokenizer, a context-assembly change) whose effect is diffuse and only visible in the *semantics* of outputs. It does not spike p99 latency. It does not increment an error counter. So no alerting rule fires.

The fix is to treat the prompt as a first-class telemetry object and to measure semantic stability directly, rather than inferring quality from operational metrics that were never designed to carry that signal.

## Three sources of silent degradation

**Prompt drift.** A prompt template changes subtly — a suffix added on a staging branch that leaks into production, a variable that now renders empty, a system message edited without a corresponding eval run. Prompt changes are often deployed with less ceremony than code, so they escape review. Without versioning, the change is invisible.

**Context pollution.** The context window fills with stale or irrelevant turns. The model does not error; it simply has less attention budget for the actual task and produces worse output. Long-running conversations and naive "append everything" history strategies are the usual culprits.

**Tokenizer skew.** A tokenizer update changes the mapping from text to token IDs. The model still runs. But every prompt is now tokenized differently, which shifts the input distribution away from what the model was trained and tuned on. This is the sneakiest of the three because it can arrive as a transitive dependency change: upgrading an SDK version can silently change which tokenizer is used for encoding, even when the model name in your config is unchanged.

## What to log

The core move is to log the prompt as data, not just send it. For every request, capture:

- raw prompt text (or a hash plus a sampled copy, depending on privacy requirements)
- prompt template ID and version
- tokenizer name and version, or a fingerprint of the vocabulary
- context window occupancy, as a fraction of the model's limit
- an embedding vector of the prompt

The embedding is what lets you detect change without needing labels. You compare each new prompt's embedding against a reference corpus of prompts captured while the system was known-good.

## Two nightly checks

**Embedding similarity against a golden corpus.** Embed the day's prompts, find each one's nearest neighbor in the golden set, and record the cosine distance. Alert when the distribution of distances shifts, not just when a single outlier appears. A per-prompt threshold produces constant noise; a shift in the median or p95 is the signal.

**Context occupancy.** Track the fraction of the context window used. When occupancy crosses a chosen ceiling, truncate or summarize the history rather than letting it grow. A ceiling in the 70–80% range is a reasonable starting point, since it leaves headroom for the model's own output and for tool results.

## Choosing a distance threshold

There is no universal threshold. Cosine distance depends on the embedding model, the domain, and the prompt length. The honest way to pick one is to calibrate it against your own task:

1. Take a sample of known-good prompts.
2. Inject controlled perturbations: a single typo, an extra whitespace token, a reordered clause, a dropped clause, a changed instruction word.
3. Embed both the original and the perturbed version, and record the distance.
4. Run both through the model and have a human (or a careful rubric-based judge) rate the outputs.
5. Plot distance against quality drop. Choose the threshold at the point where quality loss becomes unacceptable.

This gives a threshold grounded in your data rather than a borrowed number. The important property is that the calibration is repeatable: re-run it whenever the embedding model changes, because distances are not comparable across embedding models.

## A worked failure analysis

Consider a plausible and common sequence of events, with the reasoning shown.

**Setup.** A summarization service sends prompts through an SDK. The prompt template is stable. The model name in configuration is stable. The golden corpus was built during a period when output quality was verified as good.

**The change.** A dependency upgrade bumps the SDK by two minor versions to resolve a transitive conflict elsewhere in the tree. The SDK now defaults to a different tokenizer than the one the service was implicitly using before. No application code changes. No prompt template changes. The deployment looks routine.

**What breaks.** The tokenizer change means every prompt is now encoded differently. For many inputs the difference is small; for some — text with unusual whitespace, code blocks, non-Latin characters — the token boundaries shift substantially. The model receives a different token sequence than it was tuned on.

**What the dashboards show.** Latency: flat. Error rate: zero. Cost per token: roughly flat, maybe a slight change from different token counts, easily dismissed as traffic variance. Nothing alerts.

**What the drift check shows.** The nightly job embeds the day's prompts and compares each to its nearest neighbor in the golden corpus. Because the golden embeddings were computed from the *previous* tokenization of the same underlying text, distances rise. The median distance moves from its baseline band to a clearly higher band. The alert fires on the shift, not on an individual prompt.

**Slicing.** Group the distance distribution by cohort, route, endpoint, or tenant. If the shift is concentrated in one slice, that slice is where the tokenizer path differs — which is itself the diagnosis. If the shift is uniform across all slices, suspect a global change: a prompt template edit, a system-wide dependency bump, or a model version change.

**Confirming impact.** Pull a sample of outputs from before and after the change and compare them on the dimension that matters — length, factual consistency, instruction adherence, whatever your task rewards. A rubric-based judge or a small human review panel is enough to confirm whether the distance shift corresponds to a real quality change.

**Remediation.** Pin the dependency to the version whose tokenizer matches the golden corpus, redeploy, and watch the distance distribution return to its baseline band. Then add a tokenizer fingerprint to the telemetry so the next such change alerts immediately, regardless of embedding distance.

The general lesson: the drift check does not tell you *why* quality changed. It tells you *that* the input distribution changed and *where* to look. The diagnosis still requires slicing and human judgment.

## A minimal detector

The following Python 3.11 sketch implements the nightly check. It assumes you have a prompts table, a set of precomputed golden embeddings, and an embedding function.

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

# rows: list of (prompt_id, prompt_text) fetched from your store,
# ordered by recency.
rows = fetch_recent_prompts(limit=1000)

new_embeddings = embed([text for _, text in rows])       # shape (n, d)
golden_embeddings = np.load("golden_embeddings.npy")      # shape (m, d)

distances = cosine_distances(new_embeddings, golden_embeddings)
nearest = distances.min(axis=1)

median = float(np.median(nearest))
p95 = float(np.percentile(nearest, 95))

# Compare against a baseline band recorded during a known-good period,
# not a fixed constant. Alert on a shift in the distribution.
if median > BASELINE_MEDIAN + MEDIAN_TOLERANCE or p95 > BASELINE_P95 + P95_TOLERANCE:
    alert(
        f"Prompt embedding drift: median={median:.3f} p95={p95:.3f} "
        f"(baseline median={BASELINE_MEDIAN:.3f})"
    )
```

Two design notes. First, `embed` should be a function you control, so the embedding model and its version are explicit and logged alongside the numbers — otherwise you cannot compare today's distances to yesterday's. Second, compare against a baseline band derived from your own history rather than a hardcoded constant. The constant will be wrong for your domain, and it will silently become wrong again if the embedding model is ever swapped.

To measure whether the job is fast enough, instrument wall-clock time around `embed` and around the numpy comparison separately, and log both. The embedding call dominates; the comparison is negligible for corpora in the thousands. If the job is too slow, reduce the sample size before changing the embedding model, since changing the model invalidates your baseline.

## A minimal tokenizer fingerprint check

The embedding check catches drift after it has affected prompts. A fingerprint check catches tokenizer changes directly and immediately, which is cheaper and more precise for that specific failure mode.

```python
import hashlib

def tokenizer_fingerprint(vocab_bytes: bytes) -> str:
    return hashlib.sha256(vocab_bytes).hexdigest()

# At startup, and again on any dependency change:
fp = tokenizer_fingerprint(load_vocab_bytes())
log_telemetry("tokenizer_fingerprint", fp)

if fp != EXPECTED_FINGERPRINT:
    alert("Tokenizer vocabulary changed; prompt tokenization may have shifted.")
```

The value here is that it fires on the *cause* rather than the *symptom*, and it fires the moment the process starts rather than the next morning. Store the expected fingerprint in configuration and update it deliberately, with an eval run, when you intend to change tokenizers.

## Common misconceptions

**"Golden answers are enough."** Exact-match golden tests are brittle. A tokenization change can make every golden answer look wrong even when the model's behavior is fine, and a legitimate model improvement can make them look wrong too. Embedding similarity against a prompt corpus is more robust to surface changes, though it measures input stability rather than output quality — the two are related but not identical.

**"Context overflow will throw an error."** Many APIs truncate or reject silently depending on configuration, and the failure mode is a shorter or less complete answer, not an exception. Track occupancy explicitly.

**"Monitoring the model endpoint is sufficient."** The endpoint is the last link in the chain. The failure modes live in prompt templates, tokenizer versions, context assembly, and the dependencies that supply them.

**"A small cosine distance is harmless."** The distance number is meaningless without calibration. A small distance on a short prompt can correspond to a meaningful semantic change; a larger distance on a long prompt can be noise. Calibrate against your task.

## Decision checklist

Before adding a drift detector, answer these:

- Do you version prompt templates, and can you map any production request back to a template version?
- Do you log the tokenizer name and version, or a vocabulary fingerprint, per request?
- Do you know your context window occupancy distribution, not just its maximum?
- Do you have a golden corpus, and is it tagged with the model and tokenizer version it was built under?
- Have you calibrated a distance threshold against controlled perturbations on your own task?
- Do you have a rollback path for prompt templates that is as fast as your feature-flag rollback?
- Who is paged when drift fires, and what is the first diagnostic step they are expected to take?

If the answer to any of the first three is no, fix that before building the detector. Logging is the prerequisite; the detector is just a query over logs you already have.

## One next step

In the next 30 minutes, add a single field to your prompt logging: the tokenizer name and version, or a SHA-256 fingerprint of the tokenizer vocabulary, whichever your stack makes easier. Then write down the current value somewhere durable — a config file, a dashboard annotation, a comment in the deployment manifest.

That one field turns an invisible class of failure into a diffable one. When it changes, you will know immediately, and you will know it was a tokenizer change rather than a prompt edit or a model update. Everything else in this article — embeddings, thresholds, nightly jobs — is an elaboration on having that baseline.
