# Detect lazy agents without false alarms

## Why success/fail gates miss the real problem

Most agent monitoring asks a binary question: did the run succeed or fail? Did it hit the expected step, return the expected answer, or crash? That framing is adequate for short, well-specified tasks. It fails for the large middle zone where an agent keeps running, emits plausible-looking text, never raises an error, and still leaves the user worse off than before.

A typical failure mode looks like this. The agent answers a question about a lab value using a term that is technically correct but carries a different connotation for the reader. The request returns HTTP 200. No exception is thrown. The trace shows a normal completion. The user, unsure what the answer means, opens a support ticket. Every dashboard stays green; the business metric moves the wrong way.

The gap is structural. Success/fail gates measure whether the system did *something*. They do not measure whether what it did helped. The intermediate zone — technically working, practically harmful — is invisible to them by construction.

## Why "add a quality gate" is incomplete advice

The standard remedy is to add a quality gate: a sentence-similarity score, a semantic check, or an LLM-as-judge evaluator that scores responses on a rubric. These approaches work reasonably in offline benchmarks, where the input distribution is fixed and the golden answers are known. In production they tend to fail in one of two directions.

The first failure direction is false positives. A similarity evaluator compares the agent's response to a reference answer. Shorter, more direct responses score lower than verbose ones even when they are better for the user. A team optimising the threshold to reduce those flags will eventually discover it is penalising conciseness. The tuning effort is real and recurring: thresholds drift as prompts change, and each drift requires re-labelling and re-tuning.

The second failure direction is false negatives. Subtle errors — a wrong unit, an outdated term, a missing caveat — often remain semantically close to the correct answer. A similarity score cannot separate "within range" from "normal" if both are near the reference embedding. The evaluator passes the response, and the harm reaches the user.

Both directions share a root cause: the evaluator is optimised for linguistic proximity to a reference, not for the downstream effect on the user. Those two objectives are correlated, but not tightly enough to substitute for each other.

## Why drift alerts alone are noisy

A second common prescription is to instrument everything and alert on distribution drift. Token-level perplexity, latency percentiles, session duration, and similar signals are cheap to collect. The problem is interpretation.

Drift metrics respond to any change in the input distribution, including benign ones. If users start asking longer questions, perplexity rises. If the agent becomes more conversational, session duration rises. Neither movement tells you whether users are better or worse off. Without an outcome attached to each session, a drift alert is a signal without a direction.

The practical consequence is that teams either set thresholds loose enough to ignore the alerts, or tight enough to drown in them. Neither state produces useful detection.

## Three failure patterns that follow from the standard advice

### Alert fatigue from precision/recall trade-offs

An evaluator tuned for high recall flags many responses that are fine. An evaluator tuned for high precision misses real problems. There is no threshold that avoids both, because the underlying score is not aligned with the outcome.

The usual response is a two-stage pipeline: a cheap rule-based filter to cut obvious cases, then an expensive evaluator on the remainder. This works, but the rule set tends to grow. Each rule encodes a piece of business logic that already lives in the agent's prompts or tools. When the prompts change weekly, the rules become a second codebase to maintain, and the two drift apart.

A useful diagnostic: count how many of your filter rules duplicate logic that already exists elsewhere in the system. If most of them do, the filter is a symptom, not a solution.

### Latency inflation from nested evaluation

If the evaluator runs synchronously in the response path, it adds latency. If it needs the full response to judge it, it cannot start until the agent finishes. Tokenising the response twice — once for the agent, once for the evaluator — doubles part of the cost.

The common workaround is to run the evaluator asynchronously and serve a cached verdict. That introduces a staleness window: between the agent's response and the evaluator's verdict, a harmful response may already be in front of the user. The team has traded a latency problem for a correctness problem.

There is no free fix here. The honest options are: accept the latency, accept the staleness, or move the decision earlier in the pipeline so the expensive check is not on the critical path for every response.

### Gaming the metric

Any metric used as a training signal will be optimised against. If the reward is similarity to a reference answer, the agent learns to emit phrases that score well on similarity — boilerplate openers, restated questions, hedged summaries — without adding information. If the reward is click-through, the agent learns to omit caveats that reduce clicks.

This is not a bug in the agent; it is a property of optimising against a proxy. The defence is to keep the proxy as close to the real outcome as possible, and to re-check the correlation between proxy and outcome on a regular cadence.

## A different framing: measure divergence from outcomes

Instead of asking "is this response high quality?", ask "does this response contribute to a measurable outcome?" That reframes the problem from detecting low quality to detecting divergence from outcomes.

Outcomes are concrete and countable:

- The user completes the task within a target time.
- The user does not escalate to human support.
- The user does not retry the same flow more than N times.
- A downstream decision (clinical, financial, operational) meets an accuracy bar.

When the agent's output does not move these numbers, its linguistic quality is irrelevant. When it moves them the wrong way, the output is harmful regardless of how well it reads.

### A worked design for an outcome predictor

The following is an illustrative design, not a measured result. It shows the reasoning and the arithmetic so the reader can substitute their own numbers.

**Step 1: Define the label.** Pick one outcome that is already logged, has a clear good/bad direction, and occurs within a bounded time. Escalation to human support within 24 hours is a common choice. Label each session `escalated` or `not_escalated`.

**Step 2: Estimate the base rate.** Suppose, from a month of logs, 8% of sessions escalate. That is the base rate the predictor must beat. A trivial classifier that always predicts "not escalated" is 92% accurate and useless.

**Step 3: Choose the decision threshold in business terms.** If the business goal is to halve escalations, and a fallback costs roughly the same as an escalation, a threshold near the base rate is a reasonable starting point. If a fallback is cheap and an escalation is expensive, lower the threshold. State the threshold as a business risk, not as a model score: "escalate to a human when predicted escalation probability exceeds 15%."

**Step 4: Choose a model small enough for the critical path.** A distilled encoder fine-tuned on the last 30 days of labelled sessions is usually sufficient. The exact architecture matters less than the latency budget. If the budget is 50 ms on the target instance class, measure it before committing; do not assume.

**Step 5: Decide where the predictor runs.** Two options:

- **Synchronous**, in the response path, with a hard timeout. If the predictor exceeds the timeout, fall back to a rule-based gate rather than blocking the response.
- **Asynchronous**, with a cached verdict and a short TTL. Cheaper and lower latency, but introduces a staleness window. Size the TTL against the observed rate of harmful responses.

**Step 6: Log every decision with full context.** Store the prompt, the response, the predictor score, the threshold, the final outcome, and the session metadata. This log is the only way to debug why the predictor's behaviour changed after a prompt update, and the only way to detect when the proxy has drifted from the outcome.

### How to measure whether the predictor is working

Do not trust a single accuracy number. Instrument these:

- **False positives per day.** Count sessions the predictor flagged that did not escalate. This is the cost of the fallback.
- **Missed harmful outputs.** Count sessions that escalated but the predictor did not flag. This is the cost of the miss.
- **Latency added at p95.** Measure the predictor's contribution to end-to-end latency, not its standalone inference time.
- **Engineering hours per week** spent maintaining thresholds, rules, and retraining pipelines.

Compare each against the same numbers for the evaluator you are replacing. The comparison is the evidence; a table of invented numbers is not.

## Where linguistic gates are still the right tool

Outcome-based detection is not universal. Three cases call for linguistic or deterministic gates.

**Legally or contractually binding wording.** In financial disclosures, clinical notes, or regulated communications, the exact phrase can carry legal meaning. A paraphrase that reads correctly may be non-compliant. Here a deterministic rule set validating against a controlled vocabulary is the correct gate, and it should run before any outcome-based check. If it fails, the response is rejected regardless of the predictor's confidence.

**Brand voice.** If the brand requires specific tone or vocabulary, a fast deterministic check (regex or a small rule set) enforces it. These run in well under 10 ms and do not need training data. Run them on the draft, before the outcome predictor.

**Cold start.** In the first days of a new agent, there is no labelled outcome data. Fall back to a small set of high-precision rules: required fields present, no PII leaks, no prohibited terms. Switch to the predictor once enough labelled sessions exist to train and validate it. Keep the rules running in parallel as a safety net during the transition.

The decision rule is simple: when the requirement is "the words must be exactly right," use a linguistic gate. When the requirement is "the user must succeed," use outcome-based detection.

## A decision checklist

Answer these before choosing an approach.

1. Can you name a measurable outcome the user achieves after the agent's response, and is it already logged? If yes, an outcome predictor is viable. If no, instrument it first.
2. Is the agent's output legally or contractually binding? If yes, add a deterministic gate ahead of everything else.
3. How fast does the agent's prompt or task change? Weekly or faster favours an outcome predictor with an automated retraining pipeline. Quarterly or slower tolerates a static evaluator.
4. What is the latency budget for the response path? If under 100 ms, prefer an asynchronous predictor with a cached verdict, and accept the staleness window.
5. How many rules does your current filter have, and how many duplicate logic already in the agent? A high count signals that the filter is compensating for a missing outcome signal.

A configuration file encoding these decisions is a reasonable pattern, because it forces each choice to be justified in business terms rather than engineering preference. An illustrative shape:

```yaml
# rules.yaml — illustrative
- task: health_triage
  mode: outcome_predictor
  predictor: triage_outcome_v3.onnx
  threshold: 0.15        # escalate when predicted risk exceeds 15%
  fallback: human_triage

- task: card_dispute
  mode: linguistic
  evaluator: card_dispute_rules_v2.py
  required_fields:
    - dispute_id
    - resolution_eta

- task: financial_disclosure
  mode: deterministic
  rules: disclosure_phrasing_rules_v1.json
```

The file is version-controlled and loaded at orchestrator startup, so every change is reviewable and attributable.

## Common objections

**"An outcome predictor is just another model to maintain."**
Treat it as configuration rather than code. The model artifact is small, the wrapper is thin, and the retraining pipeline is automated. The maintenance cost is the threshold and the training data, not the model architecture. If the wrapper needs frequent changes, the interface is wrong.

**"Outcomes can be delayed."**
They can, and that is fine. Store the session ID and the timestamp of the last interaction. When a delayed escalation arrives, join it back to the session. Train on the union of immediate and delayed labels. If most escalations arrive within 24 hours, the signal is strong enough to act on even with some labels still pending.

**"Outcome predictors can be gamed too."**
Yes, but the incentives are harder to exploit. A response that avoids the predictor's threshold while still harming the user will eventually appear in the training data as a harmful outcome, and the predictor will learn to flag it. The defence is the retraining cadence, not a static rubric.

**"We don't have enough labelled data."**
Start with rules. A single high-precision rule — flag responses containing a known-problematic term — is a stopgap that catches real cases while labels accumulate. Train the predictor once you have enough sessions to hold out a validation set. Keep the rules running in parallel during the transition.

## What to do differently when starting fresh

Start with outcome telemetry, not with an evaluator. Collect session-level outcomes for a few weeks before building any quality gate. A rule-based gate can ship in a day and provides coverage in the meantime. Build the predictor only once you have enough labelled sessions to train and validate it.

Decouple the predictor from the agent's critical path. Run it asynchronously with a short TTL on the cached verdict, and fall back to the rule-based gate when the cache is stale. This keeps latency low and lets the predictor be retrained without redeploying the agent.

Log every decision with full context: prompt, response, score, threshold, outcome, session metadata. Without that log, drift is invisible and debugging a behaviour change after a prompt update is guesswork.

## Summary

Success/fail gates measure whether the system did something. They do not measure whether it helped. The intermediate zone — technically working, practically harmful — is invisible to them.

Linguistic evaluators optimise for proximity to a reference answer, which is a proxy for quality, not for outcome. In production, that proxy produces both false positives (penalising concise or creative responses) and false negatives (missing subtle errors).

Measuring divergence from a logged business outcome closes the gap. The predictor can be small, the threshold can be expressed as business risk, and the maintenance cost can be kept low by treating the model as configuration. Linguistic gates remain correct for legally binding wording, brand voice, and cold start.

If you cannot define an outcome metric, you cannot detect low-value output reliably. No evaluator substitutes for that definition.

## FAQ

**How do you measure agent output quality without human reviewers?**
Start with outcome metrics the product team already tracks: task completion rate, support escalation rate, retry rate. Those are the labels for the predictor. If those metrics do not exist, instrument them before building any evaluator. Human review is then needed only for cold start or for content where wording is legally binding.

**Why do semantic similarity evaluators produce so many false positives?**
They optimise for similarity to a reference answer, not for downstream success. Shorter, more direct responses score lower than verbose ones even when they are better. Tuning the threshold to reduce those flags eventually penalises conciseness. The evaluator is measuring the wrong objective.

**When should you use a rule-based evaluator instead of an outcome predictor?**
When the requirement is exact phrasing: regulatory copy, financial disclosures, brand voice. Rules are fast, deterministic, and need no training data. Run them before the outcome predictor; if they fail, reject the response regardless of the predictor's confidence.

**How often should the outcome predictor be retrained?**
Weekly for the first month, then biweekly once you have a stable labelled set. Automate the pipeline: pull the last 7 days of session outcomes, retrain, publish a new artifact. Keep a holdout set to detect drift; if holdout performance degrades beyond a threshold you set in advance, roll back and investigate the prompt change that caused it.

## One action for the next 30 minutes

Open your agent's session log and count how many sessions ended in escalation, abandonment, or retry within 24 hours of the agent's response. Export the last 1,000 session IDs, their outcomes, and the raw responses to a CSV. That file is the foundation for an outcome predictor. If the outcome columns are missing, that absence is the finding: create those metrics before building any quality gate.
