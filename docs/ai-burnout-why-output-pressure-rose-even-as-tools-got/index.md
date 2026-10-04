# AI burnout: why output pressure rose even as tools got

It works in the simple case, and breaks in a specific way under load. Here is the fuller picture.

## The one-paragraph version

AI coding assistants are often pitched as a way to cut coding time substantially, yet teams that adopt them frequently report higher output pressure rather than lower. The mechanism is not mysterious: faster generation produces more changes, more changes produce more review, more review produces more merge decisions, and more merges produce more deployments and more incidents. The tool did not create the workload. It moved the constraint downstream and made the new constraint visible to everyone at once.

## Why the concept confuses people

A common mistake is to assume a productivity tool reduces workload. Tools that increase throughput expose new bottlenecks. Code review queues, CI pipelines, QA capacity and incident response all get stressed when the rate of change rises. Teams see "more features shipped" and attribute it to the tool, ignoring how their own processes adapted to the new speed.

A second confusion is conflating "lines of code" with "work done." A 420-line pull request is not automatically three times more valuable than a 140-line one. It may simply mean the reviewer now has to parse three times as many context switches. An assistant configured to add tests for every function can balloon PR size in a single session, and that ballooning does not reliably correlate with better coverage or fewer defects.

A third confusion is over-indexing on immediate tool metrics. Autocomplete acceptance rate and completion latency are easy to collect and feel meaningful. They measure keystrokes, not outcomes. A completion that returns in a couple of hundred milliseconds feels instant, but if acting on it triggers a context switch into review, validation and merge, the real cost lands in attention rather than typing time.

## The mental model

Think of an engineering organization as a pipe. Adding AI assistants widens the pipe: more code flows per second. If the valves downstream — review, QA, incident response — keep their original settings, the system backs up at those points and pressure spikes upstream. Widening the pipe does not reduce pressure. It redistributes it to the weakest joints.

The traffic analogy is the same shape. Adding lanes to a highway without adding capacity on the bridge moves congestion from the on-ramp to the bridge. In engineering terms, faster generation shifts the bottleneck from writing code to reviewing, validating and deploying it. Metrics need to reflect that shift, not just the throughput increase.

The useful mental shift is to treat an AI assistant as a velocity amplifier rather than a productivity booster. Amplifiers do not reduce effort. They reveal where effort was already hiding. If median time-to-first-review has not changed in six months while merge rate has doubled, the assistant is not the problem. Review capacity is.

## A worked example

Consider a hypothetical team of six engineers that enables an assistant with a workspace rule suggesting unit tests for every public function. The numbers below are illustrative, chosen to make the arithmetic visible. They are not a benchmark.

Assume the team merges 8 PRs per day before adoption, at a median 140 changed lines and 3 review comments each. After six weeks, the same team merges 22 PRs per day at a median 420 changed lines and 8 review comments each.

| Metric | Before | After | Change |
|---|---|---|---|
| PRs merged per day | 8 | 22 | +175% |
| Median PR size (lines) | 140 | 420 | +200% |
| Median time to first review | 2 days | 2 hours | -96% |
| Review comments per PR | 3 | 8 | +167% |
| Prod defects per 1k PRs | 2.3 | 3.1 | +35% |

The arithmetic behind the review load is the part people skip. Before adoption: 8 PRs × 3 comments = 24 review comments per day. After: 22 PRs × 8 comments = 176 review comments per day. That is roughly 7.3× the review work, produced by the same six reviewers, in the same working hours. Merge rate rose 2.75×. Review workload rose faster than merge rate because both PR count and comments per PR increased.

The trigger is often a single configuration change. A rule that auto-suggests tests and scaffolding for every public function can turn a 10-line change into a 100-line change without any human deciding to write more code:

```python
# Before: a small utility function
async def process_payment(payment_id: str) -> bool:
    # ... payment logic ...
    return True

# After: assistant-suggested validation and test scaffolding
async def process_payment(payment_id: str) -> bool:
    """Process a payment and return success status."""
    if not payment_id:
        raise ValueError("payment_id cannot be empty")
    if not is_valid_uuid(payment_id):
        raise ValueError("payment_id must be a UUID")
    # ... payment logic ...
    return True
```

The docstring and validation are improvements. The cost is that the reviewer now validates four new branches instead of one, and the diff is larger than the logic change it contains. Note also that the illustrative snippet above shows calls to test functions rather than real tests; in a real repository those belong in a separate test module, not in the production file, and a reviewer should treat that placement as a defect.

The defect rate rise in the table is the predictable consequence. Reviewers facing 176 comments per day triage rather than review. Subtle issues survive. The assistant caught more edge cases in the code it generated, but humans had less capacity to validate the suggestions consistently.

## How this connects to patterns you already know

This mirrors what happened when CI/CD pipelines became standard. Teams that added automated testing without adjusting release cadence ended up with flaky suites and alert fatigue. The pipeline exposed the gap between deployment speed and test reliability.

It also mirrors the microservices era. Teams that decomposed monoliths without updating monitoring and on-call playbooks ended up with alert storms and unreliable systems. The decomposition was not wrong; the operational model had not caught up.

The same shape appears when a team enables strict mode in a typed language without updating lint rules and migration scripts. Strict mode catches more defects and also produces more compile-time errors, shifting pressure from runtime debugging to build-time triage. Assistants shift pressure from writing to reviewing. The pattern is identical; only the stage changes.

## Misconceptions, corrected

**AI reduces cognitive load.** It reduces keystrokes. Reviewing a 420-line PR with 8 comments is more mental work than reviewing a 140-line PR with 3 comments, even when a machine wrote most of the lines. Validation cost scales with unknowns: hidden dependencies, unhandled edge cases, and defects that only surface after merge.

**Bigger PRs mean better code.** Bigger PRs often mean more hidden complexity. A large diff dominated by generated scaffolding can obscure the handful of lines that carry the actual logic change, which makes review harder and merge mistakes more likely.

**Faster feedback loops reduce stress.** Faster loops increase stress when downstream capacity has not scaled. If a fast completion triggers a context switch into review and merge, the cost is attention fragmentation, not latency. The relevant measurement is not completion latency but switches per hour and time-to-first-review.

**Blaming the tool is productive.** The tool did not create the pressure. It exposed the gap between generation velocity and review capacity. Removing it restores the old bottleneck without addressing the underlying constraint.

## Instrumenting net time saved

Acceptance rate is the wrong metric. The useful metric is net time saved per suggestion: raw time saved minus review time, minus rework time, minus incident time attributable to the change. None of that is available from the assistant's own telemetry. It has to be assembled from the systems of record.

The measurement approach is straightforward even though the tooling is not:

1. Log suggestion events with a timestamp, a stable identifier, and the file and line range they touched. If the vendor does not expose this, emit a local event from an editor hook.
2. Log PR open, first-review, approval and merge timestamps, plus changed-line counts and comment counts, from the source host's API.
3. Join suggestions to PRs by file path and time window. This join is approximate; treat it as an estimate and state the window you used.
4. Attribute review cost per PR, for example as review comments multiplied by an assumed minutes-per-comment figure you derive from your own timestamps.
5. Subtract review and rework cost from the raw saving to get net saving, and report the distribution, not the mean. A mean hides the large PRs that dominate cost.

A minimal join looks like this. It is deliberately simple, and the assumptions are stated inline so they can be challenged:

```python
import pandas as pd

# suggestions: one row per accepted suggestion
# columns: suggestion_id, accepted_at, file_path, lines_added
# prs: one row per merged PR
# columns: pr_number, opened_at, merged_at, file_path, additions, review_comments

# Assumptions, all of which should be replaced with measured values:
# - a review comment costs 4 minutes of reviewer attention
# - a suggestion saves 0.5 hours of authoring time
# - a suggestion belongs to a PR if it was accepted within 30 minutes of the PR opening

MINUTES_PER_COMMENT = 4
HOURS_SAVED_PER_SUGGESTION = 0.5
JOIN_WINDOW = pd.Timedelta(minutes=30)

joined = suggestions.merge(
    prs,
    on="file_path",
    how="inner",
    suffixes=("_sug", "_pr"),
)

joined = joined[
    (joined["accepted_at"] >= joined["opened_at"] - JOIN_WINDOW)
    & (joined["accepted_at"] <= joined["merged_at"])
]

joined["gross_saved_hours"] = HOURS_SAVED_PER_SUGGESTION
joined["review_cost_hours"] = joined["review_comments"] * MINUTES_PER_COMMENT / 60
joined["net_saved_hours"] = joined["gross_saved_hours"] - joined["review_cost_hours"]

print(joined["net_saved_hours"].describe())
print(joined.groupby("file_path")["net_saved_hours"].sum().sort_values())
```

Two cautions on this. First, the join by file path and time window will over-attribute when a file is touched by many PRs; a stricter join needs commit-level data. Second, the per-suggestion saving figure is an assumption, not a measurement. The honest way to calibrate it is to time a small sample of comparable tasks with and without the assistant, and to report the sample size alongside the estimate. If that sample does not exist, the output is a hypothesis, not a result.

The pattern that tends to emerge from this kind of analysis is that a meaningful share of accepted suggestions land in tests, docs and scaffolding. Those suggestions save authoring time in the short term and add maintenance surface in the long term. Whether that trade is good depends on the codebase. In a young service with thin coverage, generated tests may be clearly positive. In a mature service with a large suite, generated tests that duplicate existing coverage add cost without adding signal.

## A decision checklist

Before enabling an aggressive suggestion rule, work through these questions:

- What is the current median time-to-first-review, and what will it be if PR volume doubles?
- How many review comments per day can the current reviewer pool absorb without triage behavior?
- Do generated tests add coverage the suite does not already have, or duplicate it?
- Are generated changes routed differently from human changes, or does everything share one queue?
- Is there a size threshold above which a PR is split, and is that threshold enforced automatically?
- When a defect traces back to a generated change, is that recorded, or does it disappear into the general defect count?
- Who owns the configuration that decides how much the assistant suggests, and how often is it reviewed?

If several of these have no answer, the assistant is not the thing to change first. The review process is.

## Practical mitigations

Tiered review service levels are the most common fix. A workable scheme routes small human-authored PRs to a fast lane, large generated PRs to a slower lane, and requires an explicit split for anything above a size threshold. The point is not to punish generated code. The point is to stop large diffs from competing for the same reviewer attention as small ones.

Routing is enforceable with standard repository tooling. Ownership rules can assign reviewers by path, and CI checks can fail or label a PR when the changed-line count exceeds a threshold. A label plus a dedicated queue is usually enough to keep the fast lane fast.

Suggestion rules deserve the same care as lint rules. Turning on test generation for every function is a policy decision with a review cost, and it should be reviewed on a schedule like any other policy. Rules that generate scaffolding in production files, such as the test calls shown earlier, should be disabled outright.

Finally, measure the distribution. Median and p90 time-to-first-review, review comments per PR, and PR size are the four numbers that reveal whether the pipeline is absorbing the new velocity or merely passing the pressure along.

## What to do in the next 30 minutes

Open your repository's settings and record the current median time-to-first-review and the median changed-line count per PR for the last 30 days. Write both numbers down with today's date. Everything else in this article is a response to those two figures, and without them any change to your assistant configuration is guesswork.
