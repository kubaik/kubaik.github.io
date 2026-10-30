# LLM eval drift that users notice first

## Why a green eval suite proves less than it seems

An LLM feature can pass every synthetic evaluation, keep latency graphs flat, and still generate a rising tide of support tickets. The complaints rarely name an error code. They say "the summary is wrong," "it repeats itself," or "it ignores the second half of my document." Nothing in the application logs or the model provider dashboard flags a change, because in many cases nothing about the model changed.

The gap is between the distribution the evaluation suite was built on and the distribution production actually serves. A static test set is a snapshot. Live traffic is a moving target: new UI affordances, new user cohorts, seasonal shifts, and new document types all push prompts into regions the test set never sampled. That is drift, and it is usually a pipeline problem, not a model problem. Retraining is rarely the first fix; instrumenting the pipeline so drift surfaces before users do is.

Three patterns account for a large share of user-visible drift even when the eval suite looks healthy:

- **Prompt distribution drift** — live prompts diverge from the curated test set.
- **Token budget drift** — live inputs exceed the context budget and get truncated.
- **Cache invalidation drift** — downstream caches serve completions from a previous model version.

Each has a distinct symptom, a distinct measurement, and a distinct fix. Treating them as one "the model got worse" problem is what makes them expensive.

## A mental model: three assumptions that expire

Every LLM pipeline encodes assumptions made at build time. Drift is what happens when those assumptions stop holding in production.

| Assumption at build time | Production reality | Observable symptom |
|---|---|---|
| User prompts follow the test-set distribution | Prompts shift with UI changes, seasons, and new cohorts | Specific failure modes appear that the test set never covered |
| Inputs fit inside the context window | Users paste long documents; the app injects metadata | Summaries drop the tail of the input; QA answers miss facts |
| The cache key identifies the model that produced the value | Model version changes; key does not include it | Same prompt returns two different completions across servers |

The rest of this article takes each row in turn: how to recognize it, how to measure it, and how to fix it. The measurements matter more than the fixes, because a fix you cannot verify is a guess.

## Fix 1 — prompt distribution drift

**Symptom pattern.** Users report failures on prompt shapes that never appeared in the curated test set. A support assistant that handled "What are my lab results?" starts failing on "Summarize my last three lab results for my doctor." The eval suite still reports high accuracy because it never contained that prompt shape.

**Cause.** The test set was balanced across prompt types. Live traffic skews toward one type after a UI change or an event that changes what users ask. The model's weak spot is a function of the input distribution, so the same weights produce different quality on a shifted distribution.

**Measurement.** Log every raw prompt with a timestamp and a normalized hash. Compare a rolling histogram of live prompts against the training-set histogram using Jensen-Shannon divergence (a symmetric, bounded measure of how different two probability distributions are; it is 0 for identical distributions and grows as they diverge).

A minimal FastAPI middleware in Python:

```python
from collections import defaultdict, deque
from hashlib import sha256
import math
import time

# Captured once from the evaluation/training prompt set.
TRAINING_HASHES = {
    sha256(b"What are my lab results?").hexdigest()[:16],
    sha256(b"What does this blood test mean?").hexdigest()[:16],
    sha256(b"Can you explain my diagnosis?").hexdigest()[:16],
}

# Rolling window of live prompt hashes, bucketed by 5-minute intervals.
WINDOW_BUCKETS = 7 * 24 * 12  # 7 days at 12 buckets/hour
PROMPT_HISTOGRAM = defaultdict(lambda: deque(maxlen=WINDOW_BUCKETS))


def js_divergence(p: dict, q: dict) -> float:
    """Jensen-Shannon divergence between two normalized dicts."""
    keys = set(p) | set(q)
    m = {k: 0.5 * (p.get(k, 0.0) + q.get(k, 0.0)) for k in keys}
    total = 0.0
    for k in keys:
        pk, qk, mk = p.get(k, 0.0), q.get(k, 0.0), m[k]
        if mk > 0:
            if pk > 0:
                total += 0.5 * pk * math.log2(pk / mk)
            if qk > 0:
                total += 0.5 * qk * math.log2(qk / mk)
    return total


def record_and_check(prompt: str) -> float:
    prompt_hash = sha256(prompt.encode()).hexdigest()[:16]
    bucket = int(time.time() / 300)
    PROMPT_HISTOGRAM[bucket].append(prompt_hash)

    live_counts = defaultdict(int)
    for hashes in PROMPT_HISTOGRAM.values():
        for h in hashes:
            live_counts[h] += 1

    total = sum(live_counts.values())
    if total == 0:
        return 0.0

    live_p = {h: c / total for h, c in live_counts.items()}
    train_p = {h: 1.0 / len(TRAINING_HASHES) for h in TRAINING_HASHES}
    return js_divergence(train_p, live_p)
```

Two details are worth calling out because they are easy to get wrong.

First, the training distribution must be normalized over the same key space as the live distribution. If the training set has three prompt shapes and live traffic has fifty, the divergence will be dominated by the unmatched keys, which is the intended signal but also means the absolute value depends on how you bucket prompts. Deciding on a bucketing scheme — exact hash, embedding cluster, or intent label — is a design choice, not a detail.

Second, the threshold is not universal. A threshold of 0.15 is a starting point, not a law. The correct way to set it is to replay historical traffic through the detector and find the value that would have fired before the last few user-visible regressions, without firing on ordinary weekday/weekend variation. That is a calibration exercise against your own data.

**Fix.** When divergence crosses the threshold, the response is not to retrain. It is to add the new prompt shapes to the evaluation set, rebalance the weights, and re-run the suite. If the suite now fails, you have found a real quality gap and can decide whether to fix the prompt template, add retrieval, or fine-tune. If it passes, you have closed the coverage gap and the alert will stop firing on that shape.

## Fix 2 — token budget drift

**Symptom pattern.** Users report the model "stopped working" after pasting a long document or after a UI change that added metadata. Logs show either a hard context-length error or, worse, no error at all with a degraded answer.

**Cause.** Inputs grew past the context window. Two mechanisms are common: users paste long documents, and the application injects metadata (system preamble, retrieved context, tool schemas, user profile) that was not present when the budget was sized. Some APIs return a hard error when the request exceeds the limit; others truncate silently depending on the provider and the request shape. The silent case is the dangerous one, because the model answers confidently about a partial input.

**Measurement.** Count tokens before the request, not after. Use the tokenizer that matches the model family. A minimal Node example:

```javascript
const express = require('express');
const { encoding_for_model } = require('tiktoken');

const app = express();
app.use(express.json());

// Adjust to the tokenizer that matches your model family.
const enc = encoding_for_model('gpt-4o');

function countTokens(text) {
  return enc.encode(text).length;
}

const CONTEXT_LIMIT = 128000;   // documented model limit
const RESPONSE_RESERVE = 8192;  // tokens reserved for the completion
const ALERT_THRESHOLD = CONTEXT_LIMIT - RESPONSE_RESERVE;

app.post('/chat', async (req, res) => {
  const { prompt, metadata = '' } = req.body;

  const promptTokens = countTokens(prompt);
  const metadataTokens = countTokens(metadata);
  const totalTokens = promptTokens + metadataTokens;

  if (totalTokens > ALERT_THRESHOLD) {
    console.warn(
      `token budget alert: ${totalTokens} tokens ` +
      `(prompt=${promptTokens}, metadata=${metadataTokens})`
    );
    // Decide policy here: summarize, chunk, reject early, or downgrade.
  }

  // ... call the model with the full input ...
  res.json({ ok: true, totalTokens });
});
```

The arithmetic is the point: the budget for the input is the context limit minus the response reserve. If the documented limit is 128,000 tokens and you reserve 8,192 for the completion, the input budget is 119,808 tokens. Whether you alert at that number or at a lower one depends on how much variance you see in metadata size.

**Failure-mode analysis.** The dangerous case is not the request that errors out; it is the request that silently truncates the tail. This produces a specific class of bug: the answer is coherent but wrong about the last third of the input. A useful detection trick is to append a sentinel instruction to the end of long inputs — for example, "End of document. If you can read this line, reply with the word RECEIVED before answering." If the sentinel is missing from the response on long inputs, truncation is happening. This is a cheap, high-signal check that works across providers.

**Fix.** Three options, in order of preference:

1. **Reduce the input.** Summarize or chunk long documents before they reach the model. This is the only fix that preserves answer quality.
2. **Reserve explicitly.** Compute the budget as context limit minus response reserve and enforce it before the call, rejecting or downgrading early with a clear user-facing message.
3. **Change the model.** A larger context window buys headroom but not correctness; a model that answers well on a 10k-token input may degrade on a 100k-token input even if it accepts it.

## Fix 3 — cache invalidation drift

**Symptom pattern.** The same prompt returns different completions depending on which server or cache served the request. Support tickets include screenshots of two different answers to the same question. Nothing in the model logs shows a problem.

**Cause.** The application caches completions, but the cache key does not include the model version (or the prompt template version, or the retrieval index version). After a model update, the old key still resolves to the old completion.

A broken key looks like this:

```python
import hashlib
from fastapi import FastAPI, Request
from redis import Redis

app = FastAPI()
redis = Redis(host="localhost", port=6379, db=0)

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    prompt = data["prompt"]
    user_id = data["user_id"]

    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
    cache_key = f"user:{user_id}:prompt_hash:{prompt_hash}"  # missing model version

    cached = redis.get(cache_key)
    if cached:
        return {"response": cached.decode()}

    response = generate(prompt)
    redis.setex(cache_key, 3600, response)
    return {"response": response}
```

The key identifies the user and the prompt but not the thing that produced the answer. Any change to the producer — model version, system prompt, temperature, retrieval corpus — leaves stale values reachable.

The fix is to make the key a function of everything that can change the output:

```python
MODEL_VERSION = "gpt-4o-2024-08-06"   # bump on every rollout
PROMPT_TEMPLATE_VERSION = "v7"        # bump on every template change
RETRIEVAL_INDEX_VERSION = "2024-11-02"  # bump on every reindex

def build_cache_key(user_id: str, prompt: str) -> str:
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
    return (
        f"user:{user_id}"
        f":model:{MODEL_VERSION}"
        f":template:{PROMPT_TEMPLATE_VERSION}"
        f":index:{RETRIEVAL_INDEX_VERSION}"
        f":prompt:{prompt_hash}"
    )
```

The performance cost is negligible; the operational cost is real. Every rollout must bump the version identifiers, and that requires a single source of truth for what is currently deployed. Teams without a model registry tend to discover this the hard way: the fix is not the cache key, it is the deployment process that guarantees the key changes.

**Verification.** Simulate a rollout by changing `MODEL_VERSION`, then issue the same prompt twice. The first request should miss the cache and generate a fresh completion; the second should hit the cache and return the same new completion. If the second request returns the old value, the key is still missing a version component. This takes two minutes and catches the bug before users do.

## How to verify the fixes with metrics

Each fix has one primary metric. Instrument all three and put them on the same dashboard as latency and error rate, because drift and performance regressions often arrive together.

| Metric | What it measures | How to compute it | Alert shape |
|---|---|---|---|
| `prompt_js_divergence` | Shift in live prompt distribution vs. the evaluation set | Jensen-Shannon divergence over hashed prompt buckets | Sustained rise above a calibrated threshold |
| `input_tokens_p99` | Tail of input size per request | Tokenizer count before the call, p99 over a rolling window | p99 approaching the input budget |
| `cache_version_mismatch` | Completions served from a stale producer | Fraction of cache hits whose stored version tag differs from the deployed version | Any nonzero value after a rollout |

Two verification procedures are worth building into CI:

- **Prompt drift replay.** Replay the last N days of logged prompts through the detector and confirm it fires on the interval that preceded a known regression. This calibrates the threshold against your own traffic rather than a borrowed number.
- **Token budget load test.** Generate synthetic requests with a long document and confirm the `input_tokens_p99` alert fires before the request would exceed the budget. This proves the instrumentation works before it is needed.

For cache drift, the two-request check described above is sufficient and should run as part of the rollout checklist.

## A worked example of the reasoning

Suppose a team ships a support assistant. Two weeks later, tickets rise. The eval suite is still green. Walk through the three causes in order:

1. **Check prompt distribution.** Compute the divergence between the last 7 days of live prompts and the evaluation set. If it has risen sharply, the coverage gap is the likely cause. Add the new prompt shapes to the eval set and re-run. If the suite now fails, the quality gap is real and the fix is a prompt or retrieval change. If it passes, the alert was coverage noise.
2. **Check token budget.** Look at the p99 of input tokens over the same window. If it has risen toward the budget, long inputs are the likely cause. Add the sentinel instruction to long inputs and check whether it appears in responses. If it is missing, truncation is happening and the fix is summarization or chunking.
3. **Check cache versioning.** Diff the deployed model version against the version embedded in cache keys. If they differ, stale completions are being served and the fix is to include the version in the key and invalidate.

The order matters because the causes have different costs. Prompt distribution drift is the most common and the cheapest to check. Token budget drift is the most likely to cause silent wrong answers. Cache invalidation drift is the most likely to cause visible inconsistency and the easiest to verify.

## Deployment checklist

Run this before every rollout that changes the model, the prompt template, or the retrieval index.

- [ ] Evaluation set includes prompt shapes observed in the last 7 days of live traffic.
- [ ] Prompt divergence threshold is calibrated against historical regressions, not copied from a blog post.
- [ ] Input token budget is enforced before the model call, with a response reserve subtracted.
- [ ] Long-input truncation is detected via a sentinel check, not assumed absent.
- [ ] Cache keys include model version, prompt template version, and retrieval index version.
- [ ] Rollout plan includes an observation window with the three drift metrics on a shared dashboard.
- [ ] Rollback procedure is tested, not just documented.

## Escalation path when the three fixes do not resolve it

If tickets persist after all three causes have been ruled out, the problem is likely below the pipeline layer.

1. **Re-run the evaluation suite against the exact deployed artifact.** If it fails, the artifact is not the one that was evaluated.
2. **Log the prompt string before and after preprocessing.** If preprocessing changes the prompt materially, the evaluation set is measuring a different input than production sends.
3. **Check downstream systems.** Cache TTLs shorter than the rollout cadence, hash collisions in truncated prompt hashes, and embedding-model version mismatches between the vector store and the LLM are all common.
4. **Escalate to the provider with a reproducible case.** A minimal prompt, the expected completion, and the observed completion is a far stronger report than a support ticket describing symptoms.

A pattern worth watching for: a preprocessing step that injects a character the tokenizer handles unusually. Currency symbols, zero-width characters, and unusual whitespace can all change tokenization in ways that alter model behavior without changing the visible prompt. When a regression survives all three drift checks, diff the byte-level prompt, not the string-level prompt.

## FAQ

**How do I tell prompt drift from a model bug?**

Start with the divergence metric. If live prompts have shifted materially from the evaluation set, fix the coverage gap first. If divergence is flat and complaints are rising, the input distribution is not the cause and the token budget or cache versioning is the next place to look. A model bug that survives a flat distribution and a correct token budget is rare, but it is the case where a minimal reproducible prompt sent to the provider is the right escalation.

**What is a realistic token budget threshold?**

The threshold is the documented context limit minus the response reserve you need. If the limit is 128,000 tokens and you want up to 8,192 tokens of response, reserve the difference. The exact number depends on your model and your response length distribution; measure the p99 of response length and reserve above it.

**How often should the evaluation set be updated?**

Weekly is a reasonable default for a product with active UI development. The trigger, though, should be a divergence alert, not a calendar. A UI change can shift the prompt distribution within days, and a weekly cadence alone will miss it.

**Does adding version components to cache keys hurt performance?**

The key grows by a few dozen bytes and the lookup cost is unchanged. The real cost is operational: every rollout must bump the version, which requires a single source of truth for the deployed model, template, and index versions. That source of truth is the actual deliverable.

## Do this in the next 30 minutes

Pick one endpoint that calls an LLM and add token counting in front of the model call. Log the input token count with a timestamp. Do not add an alert yet — just collect data for a day. Tomorrow, compute the p99 for that endpoint and compare it to the documented context limit minus your response reserve. If the p99 is within 20% of the budget, you have found a latent truncation risk before a user did.
