# Model evals: the gaps that let drift slip through

A model is not a function. Deploy a new checkpoint, a new system prompt, a new decoding parameter, or let a vendor swap the model behind an API alias, and the outputs change. Not catastrophically — no stack traces, no 500s, no alert. The change is subtle: a classification boundary shifts by a few percent, a summarizer starts copying a different phrase, a tool-calling agent picks a different argument order. Dashboards stay green because they measure latency and error rate, not behavior.

Evaluation is supposed to catch this. In practice, most evaluation setups are built to answer "is the model good?" — a one-time question — when the production question is "did the model change in a way that matters?" Those are different tests. The first rewards a high average. The second rewards sensitivity to small, directional differences. The gap between them is where subtle behavior changes live.

This article is about that gap: what it looks like mechanically, why common eval designs miss it, and how to close it without building a full research lab. The concrete problem: teams routinely ship model and prompt changes that pass their eval suite but shift behavior on a slice of inputs nobody measured, and they find out from users, not from CI.

## The confusion: "eval passed" is not "behavior unchanged"

When an eval suite reports a score, people read it as a property of the model. It is actually a property of the model *on the specific inputs in the suite*, aggregated by a specific metric. Change either the inputs or the metric and the number moves. That is obvious in the abstract and routinely forgotten in practice.

Consider a support-ticket classifier with an eval set of 500 labeled examples. The model scores 0.91 macro-F1. A new checkpoint scores 0.91 macro-F1. The suite passes. But macro-F1 averages over classes, and the new checkpoint could have moved 15% of one rare class's predictions while compensating on another class. The average hides the swap. If that rare class is "billing dispute," the business impact is real even though the metric is flat.

The confusion is treating the aggregate as a fingerprint. An aggregate is a summary; a fingerprint is a set of per-example decisions. Subtle changes live in the per-example decisions, and aggregates are designed to smooth exactly those out.

## Why aggregates hide directional change

Averages and single-number metrics are low-pass filters. They are built to be stable under small perturbations, which is precisely the wrong property when you want to detect small perturbations.

Three mechanisms cause this:

1. **Compensation.** A model can get better on some examples and worse on others, netting zero. Accuracy, F1, BLEU, pass@k — all are invariant to this kind of swap. The metric says "no change"; the behavior says "different model."

2. **Class and slice dilution.** If 90% of your eval set is one easy slice, a change that affects only the other 10% barely moves the aggregate. A 20-point drop on a 5% slice moves overall accuracy by one point — inside most teams' noise threshold.

3. **Threshold effects.** Many evals use a pass/fail judge or a similarity threshold. A model that was just above the threshold and is now just below flips a verdict, but the distance moved is tiny. Aggregated over a suite, these flips look like noise; per-example, they are a coherent shift.

A useful analogy: measuring a model with a single accuracy number is like monitoring a server with only CPU utilization. It is a real signal, but it will not tell you that one endpoint started returning wrong data. You need per-endpoint error rates. For models, the "endpoints" are slices and individual examples.

## The eval stack most teams actually have

Most production eval setups, when examined closely, consist of three layers with different failure modes:

| Layer | Typical implementation | What it catches | What it misses |
|---|---|---|---|
| Unit evals | A fixed labeled set, run in CI | Regressions on known-hard examples | Anything not in the set; slice-level shifts |
| Online metrics | Business KPIs, thumbs up/down | Large, user-visible degradation | Subtle shifts users don't report; delayed feedback |
| Human review | Spot checks, ad hoc | Surprising failures someone happens to notice | Systematic drift; rare slices; anything not sampled |

None of these layers is wrong. The problem is they share a blind spot: they all depend on someone having thought to look at the right slice with the right comparison. Subtle changes are, by definition, the ones nobody thought to look for.

The failure mode is not "no evals exist." It is "the evals are shaped like a smoke test and used as a regression test." A smoke test answers "does it work at all?" A regression test answers "did this specific thing change?" Only the second catches subtle drift, and it requires per-example comparison against a baseline, not a fresh aggregate.

## Worked example: a prompt change that passes every aggregate

Suppose a team maintains a retrieval-augmented question-answering system. The eval set has 300 questions with reference answers. The metric is exact-match against a normalized reference, plus a judge model that scores helpfulness 1–5. Current scores: exact-match 0.62, mean helpfulness 4.1.

A developer tightens the system prompt: "Answer only using the provided context. If the context does not contain the answer, say you don't know." The intent is to reduce hallucination. The new scores: exact-match 0.63, mean helpfulness 4.0. The suite passes — exact-match went up, helpfulness dipped within noise.

Now look per-example. Of the 300 questions, 41 changed their answer. Of those:

- 22 were previously wrong and are now correct (the prompt helped).
- 19 were previously correct and are now refusals ("I don't know") because the context was retrieved but the answer required a small inference the prompt now forbids.

The aggregate shows a net gain of 3 correct answers. The behavior change is 41 examples, 19 of them regressions, concentrated on multi-hop questions. If multi-hop is 10% of real traffic, the team just shipped a prompt that improves the eval set and degrades a real user segment — and the aggregate said "ship it."

This is not a contrived scenario. It is the standard shape of prompt-tuning regressions: the change is directional (more refusals), the eval set is dominated by single-hop questions, and the metric rewards the majority.

The fix is not a better aggregate. It is a per-example diff with slice labels, reviewed before merge. A minimal version:

```python
# compare.py — run two model configs on the same eval set and diff per example
import json
from collections import defaultdict

def run(model_fn, examples):
    results = {}
    for ex in examples:
        results[ex["id"]] = model_fn(ex["input"])
    return results

def diff(baseline, candidate, examples):
    by_slice = defaultdict(lambda: {"changed": 0, "regressed": 0, "improved": 0})
    for ex in examples:
        b, c = baseline[ex["id"]], candidate[ex["id"]]
        if b == c:
            continue
        s = ex.get("slice", "default")
        by_slice[s]["changed"] += 1
        if ex["expected"] == b and ex["expected"] != c:
            by_slice[s]["regressed"] += 1
        elif ex["expected"] != b and ex["expected"] == c:
            by_slice[s]["improved"] += 1
    return dict(by_slice)

if __name__ == "__main__":
    examples = [json.loads(l) for l in open("eval_set.jsonl")]
    baseline = run(lambda x: "...", examples)   # replace with real call
    candidate = run(lambda x: "...", examples)  # replace with real call
    print(json.dumps(diff(baseline, candidate, examples), indent=2))
```

The output is a per-slice table of changed/regressed/improved counts. On the example above it would show `{"multi_hop": {"changed": 19, "regressed": 19, "improved": 0}}` — visible, reviewable, and gateable. The aggregate hid it; the diff surfaces it.

One caveat on this script: it counts a change as "regressed" only when the baseline matched the reference. Changes where neither baseline nor candidate matches the reference are counted as "changed" but neither improved nor regressed, so review those manually — they often reveal that the reference label itself is wrong or that the example needs a slice label.

## Common misreadings of eval results

Several patterns recur when teams interpret eval output. Each is a specific way subtle change slips through.

**"The score went up, so the change is good."** A score can go up while a slice goes down. Always ask which examples moved. A net gain of 3 with 19 regressions is a different change than a net gain of 3 with 3 improvements and 0 regressions, even though the aggregate is identical.

**"The score is within noise, so nothing changed."** Noise bands are usually estimated from run-to-run variance of the same model. They say nothing about whether a *different* model produces different outputs. A change can be perfectly reproducible and still inside the noise band of the aggregate — because the aggregate is insensitive, not because the behavior is stable.

**"We use a judge model, so we don't need labels."** Judge models are useful for open-ended outputs, but they introduce their own sensitivity profile. A judge asked "is this helpful?" will not notice that the answer now cites a different source, or refuses more often, or changes format. Judges answer the question you ask; subtle changes often live in dimensions you did not ask about.

**"Our online metrics would catch it."** Online metrics catch what users report or what moves a business KPI. Subtle behavior changes often produce neither — users adapt, or the segment is small, or the KPI is too coarse. Online metrics are a backstop, not a detector.

**"We tested the new model against the old one and it was better on average."** Better on average on your eval set is a statement about your eval set. It is not a statement about the production distribution, and it says nothing about per-example stability. Vendor model updates in particular can change behavior on inputs your eval set does not represent.

## Designing evals that are sensitive to small changes

The goal is not a bigger eval set. It is an eval set and metric pair that are *sensitive* to the changes you care about and *specific* enough to act on. Four design choices do most of the work.

**1. Compare against a pinned baseline, per example.** Store the baseline model's outputs for every eval example, not just its score. On every candidate run, diff outputs and report changed/regressed/improved per slice. This is the single highest-value change and it costs one extra file per eval set.

**2. Label slices, and report per slice.** Every eval example should carry metadata: task type, difficulty, language, input length bucket, whatever dimension the product cares about. Report metrics per slice, not just overall. A slice with 20 examples and a 30% regression is a real signal even if the overall number is flat.

**3. Use metrics that are sensitive at the example level.** Exact-match is brittle but sensitive; a judge score is smooth but insensitive. Pair them: exact-match (or a strict structural check) to catch directional shifts, a judge to catch quality changes the strict check cannot see. Report both, and look at the disagreement.

**4. Freeze the eval set and version it.** An eval set that changes between runs cannot detect change. Treat it like a test suite: versioned, reviewed, with additions for every production incident. When a subtle change slips through, the fix is a new example in the eval set, not a new dashboard.

A useful framing: the eval suite should be able to answer "what changed?" not just "is it good?" The first question requires per-example comparison and slice labels. The second requires only an aggregate. Most suites answer the second and are then trusted with the first.

## How to measure aggregate sensitivity before trusting it

Before relying on a metric to detect drift, quantify how much it moves when behavior actually changes. The procedure is mechanical and takes an afternoon:

1. Take the current production configuration and run the eval set three to five times unchanged. Record the per-run aggregate score. The spread is the run-to-run noise floor for that metric on that set.
2. Construct a deliberately perturbed configuration — for example, a prompt that adds one extra sentence of instruction, or a decoding temperature shift. Run the same eval set three to five times.
3. Compare the two distributions. If the mean shift is small relative to the noise floor, the aggregate cannot detect that class of change, regardless of how many examples you add.
4. Repeat step 3 for the per-slice regression counts. If the slice counts separate cleanly while the aggregate does not, the slice-level gate is the one to build.

Instrument per-example outputs and slice labels for this; without them you can only compare aggregates, which is the measurement you are trying to validate. The output of this exercise is a decision: which metric, at what granularity, with what threshold, is sensitive enough to gate on. That is a far stronger basis for a CI gate than a threshold picked by intuition.

## The vendor-model problem

A large share of subtle behavior change does not come from application code. It comes from a model called through an API whose underlying checkpoint changes. Providers update models; aliases like a "latest" or "stable" name can point at different weights over time. The prompt, the code, and the eval scores can all be unchanged while behavior shifts.

The practical defense is to pin a specific model version where the provider offers one, and to run the eval suite against the pinned version on a schedule, not only on deploy. If the provider does not offer version pinning, record the model identifier and a timestamp with every eval run so a behavior change can be correlated with a provider update after the fact. Treat a model version bump like a dependency upgrade: it gets its own eval run and its own review, not an incidental ride along with a feature branch.

This is also where per-example baselines earn their keep. If baseline outputs for the pinned version are stored, a provider update shows up as a diff that can be inspected, not a mysterious KPI wobble three weeks later.

## A minimal gate to add this week

No new platform is required. Three artifacts and a CI step suffice:

1. **A versioned eval set** with slice labels, stored as JSONL, one example per line.
2. **A baseline output file** produced by the current production configuration, committed alongside the eval set.
3. **A diff script** (like the one above) that fails CI when regressions in any slice exceed a threshold chosen deliberately, and prints the changed examples for review.

The threshold is a judgment call, and it should come from the sensitivity measurement above rather than intuition. A reasonable starting point: fail if any slice has more than a handful of regressions, or if the total number of changed examples exceeds a small fraction of the set. The point is not the exact number — it is that the gate looks at per-example change, not just the aggregate.

One caveat: if the model is non-deterministic (temperature > 0, or a provider that does not guarantee determinism), baseline diffs will contain noise. Either run evals at temperature 0 where the provider supports it, or run each configuration multiple times and compare distributions rather than single outputs. Non-determinism is a reason to be careful about thresholds, not a reason to skip per-example comparison.

## Frequently Asked Questions

**How many examples are needed to detect a subtle behavior change?**
It depends on the size of the slice that matters, not the overall set. To see a 20% regression on a slice, the slice needs enough examples that the regression is visible above noise — often a few dozen. The practical move is to size slices deliberately: if a behavior matters, give it at least a few dozen labeled examples and report it separately.

**Why not just use a stronger judge model?**
A stronger judge helps with quality assessment but does not solve sensitivity. A judge answers the question posed; subtle changes often occur on dimensions that were not asked about (format, source selection, refusal rate). Judges are a useful complement to per-example diffs, not a replacement.

**What about online A/B tests?**
A/B tests are the right tool for measuring user impact, but they are slow, need traffic, and are insensitive to changes that affect a small segment. They are a backstop after shipping, not a gate before it. Use them to confirm, not to detect.

**Does this apply to non-LLM models?**
Yes. Classifiers, rankers, and embedding models have the same failure mode: aggregate metrics hide per-example swaps and slice-level shifts. The mechanics are identical; only the metric names change.

## The next concrete step

Open the eval script and check whether it stores per-example outputs for the current production configuration. If it does not, add that now: write a JSONL file mapping example ID to model output, commit it as the baseline, and make the next eval run diff against it. That single change turns the eval from a score into a fingerprint, and a fingerprint is what catches subtle behavior change before users do.
