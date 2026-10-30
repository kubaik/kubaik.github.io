# Rebalance pay when AI fills half your role

Compensation conversations after an AI rollout tend to fail for a structural reason: the job description, the salary band, and the engineer's own pitch all describe work that no longer exists in the same proportions. The tutorials cover the happy path of adopting an assistant. This article covers what comes after — measuring the shift, arguing from it, and locking in the result.

## The problem: your band was priced for a different job

When a company adopts an AI coding assistant broadly, three things move at once, and usually not in sync:

1. **Task mix changes.** Boilerplate, CRUD endpoints, test scaffolding, and migration scripts shrink as a share of the work. Debugging, system design, incident response, and cross-team judgment grow.
2. **Bands lag.** Published compensation data reflects what companies paid for a role over the previous hiring cycle, not what the role looks like this quarter.
3. **The pitch stays stale.** Most engineers negotiate on tenure, cost of living, or "what I delivered last year" — all backward-looking and all easy to deflect.

The failure mode is predictable: an engineer accepts a modest cost-of-living adjustment, the AI absorbs another tranche of their original tasks over the following two quarters, and when the next budget cycle arrives there is no new argument to make. The raise was real but the leverage was gone.

The fix is to negotiate on a *forward* definition of the role, backed by numbers the manager can independently verify. That requires three things: a defensible measurement of what AI already does in your repo, a small set of metrics that translate into business terms, and a written artifact the manager can forward upward.

## What "AI coverage" actually means, and how to measure it

There is no vendor-neutral, industry-standard metric called "AI coverage." Anyone quoting a single percentage is either using a specific tool's definition or making it up. What you can do is define the metric precisely for your repository and state your definition up front. A manager will accept a clearly-scoped metric far more readily than a vague claim.

Three measurable quantities are worth the effort:

- **AI-attributed change share** — the fraction of changed lines (added + removed) in a window that come from commits attributable to AI-assisted work.
- **Rework ratio** — how often AI-attributed output is subsequently modified by a human in a follow-up commit.
- **Domain uniqueness** — the fraction of actively-changed files with no AI attribution in the window. This is a rough proxy for the parts of the system where your judgment, not the model's, is doing the work.

Attribution is the hard part. There is no reliable way to detect "written by AI" from source text alone, and you should not pretend otherwise. What you *can* rely on is commit metadata your team already controls:

- A `Co-authored-by:` trailer naming the assistant, if your team's workflow adds one.
- A bot author suffix such as `[bot]`.
- A branch or PR label convention (`ai-assisted`, `copilot`, etc.).

If your team has no such convention, that is itself a finding worth raising — you cannot manage what you do not label. Propose the convention first, collect data for a few weeks, then negotiate.

### A worked example (illustrative numbers)

Suppose over a 90-day window your `git log` shows 12,400 changed lines across non-test source files. Of those, 7,700 carry an AI attribution trailer. Then:

- AI-attributed change share = 7,700 / 12,400 = **0.62** (62%)
- Of the 7,700 AI-attributed lines, 1,700 are touched again by a human commit within 14 days. Rework ratio = 1,700 / 7,700 = **0.22** (22%)
- Of the 340 files changed in the window, 51 have zero AI attribution. Domain uniqueness = 51 / 340 = **0.15** (15%)

These numbers are illustrative, not benchmarks. The point is that they are arithmetic on data you can pull from your own repository, and every step is auditable.

Here is a minimal CLI that computes them. It is deliberately small — a script you can read in one sitting is more persuasive in a meeting than a dashboard nobody can explain.

```python
# ai_audit/cli.py
import argparse
import json
import subprocess
from collections import defaultdict

AI_TRAILER = "Co-authored-by: github-actions"
BOT_SUFFIX = "[bot]"


def git_log(days):
    cmd = [
        "git", "log",
        "--pretty=format:__COMMIT__%H|%an|%ai|%s|%b",
        f"--since={days}.days",
        "--numstat",
        "--", ".",
        ":!tests/*",
    ]
    return subprocess.run(
        cmd, check=True, capture_output=True, text=True
    ).stdout.splitlines()


def parse(lines):
    stats = defaultdict(lambda: {"human": 0, "ai": 0, "rework": 0})
    current = None
    for line in lines:
        if line.startswith("__COMMIT__"):
            current = line[len("__COMMIT__"):]
            continue
        parts = line.split("\t")
        if len(parts) < 3 or current is None:
            continue
        added, removed, path = parts[0], parts[1], parts[2]
        added = int(added) if added.isdigit() else 0
        removed = int(removed) if removed.isdigit() else 0
        churn = added + removed
        is_ai = AI_TRAILER in current or BOT_SUFFIX in current
        stats[path]["ai" if is_ai else "human"] += churn
        if is_ai and ("fix" in current.lower() or "rework" in current.lower()):
            stats[path]["rework"] += 1
    return stats


def summarise(stats):
    human = sum(v["human"] for v in stats.values())
    ai = sum(v["ai"] for v in stats.values())
    rework = sum(v["rework"] for v in stats.values())
    changed_files = len(stats)
    unique_files = sum(1 for v in stats.values() if v["ai"] == 0)
    total = human + ai
    return {
        "human_pct": round(human / total, 3) if total else 0.0,
        "ai_pct": round(ai / total, 3) if total else 0.0,
        "rework_ratio": round(rework / ai, 3) if ai else 0.0,
        "domain_uniqueness": round(unique_files / changed_files, 3) if changed_files else 0.0,
        "files_analysed": changed_files,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90)
    args = parser.parse_args()
    stats = parse(git_log(args.days))
    print(json.dumps(summarise(stats), indent=2))


if __name__ == "__main__":
    main()
```

Run it and capture the output:

```bash
python -m ai_audit.cli --days 90 > audit.json
```

A result might look like:

```json
{
  "human_pct": 0.38,
  "ai_pct": 0.62,
  "rework_ratio": 0.22,
  "domain_uniqueness": 0.15,
  "files_analysed": 340
}
```

Note the bug that was in the naive version of this script: the original computed `human_pct` by dividing human churn by *AI* churn, which produces a meaningless ratio. Every percentage must be divided by the same total. That is exactly the kind of arithmetic slip a skeptical manager will catch, so audit your own formula before you present it.

### Extending to multiple languages

The `--numstat` output includes every changed file regardless of language, so the script already handles polyglot repos. What it does *not* do is separate TypeScript from Python from Go. If you want per-language breakdowns, filter on the path suffix:

```python
def by_language(stats):
    buckets = defaultdict(lambda: {"human": 0, "ai": 0})
    for path, v in stats.items():
        lang = path.rsplit(".", 1)[-1] if "." in path else "other"
        buckets[lang]["human"] += v["human"]
        buckets[lang]["ai"] += v["ai"]
    return dict(buckets)
```

This matters when you argue: if AI handles 80% of your TypeScript but only 30% of your Python, your Python work is the differentiated part and should be described that way.

## Turning the numbers into a negotiation position

Raw metrics do not win raises. Interpretations do. Here is a decision checklist for reading your own `audit.json`.

| Pattern | What it suggests | Argument to make |
|---|---|---|
| High AI share, low rework | AI is genuinely doing the routine work; you are supervising it | Argue for scope expansion and a level review, not a "quality premium" |
| High AI share, high rework | AI output is cheap but noisy; you are absorbing the cleanup | Argue for a defined quality-ownership role with a title change |
| Low AI share, high domain uniqueness | You work in territory the model cannot yet reach | Argue for a domain-expert premium and a written scope document |
| Low AI share, low uniqueness | The metric is probably misconfigured, or the window is too short | Fix the instrumentation before negotiating |

Two rules apply regardless of pattern:

1. **Never present a metric you cannot regenerate on demand.** If the manager asks "how did you get 62%?", you should be able to run the command in the room. Bring the script, not just the number.
2. **Tie every metric to a business outcome.** "22% rework ratio" is meaningless to a VP. "22% of AI-generated changes required a follow-up fix commit, which is why the on-call rotation absorbed three additional pages last month" is a cost argument.

## Handling the objections you will actually hear

### "The AI stats are just noise"

This is the most common pushback and it is partly fair — attribution via commit trailers is imperfect. The response is not to defend the number but to reframe it: the metric is a *lower bound* on AI contribution, because it only counts changes that were explicitly labeled. Any unlabeled AI-assisted work makes the real figure higher, not lower. If anything, the measurement understates the shift.

Then pivot to the verifiable part: the list of files and systems where AI attribution is zero. That list does not depend on attribution accuracy at all — it is simply the set of files with no AI trailer in the window. If those files are the ones that cause incidents, that is your argument.

### "Your band is already at market"

Published band data is a starting anchor, not an answer. Two things to check before accepting it:

- **What job is the band priced for?** A band labeled "Backend Engineer L4" was calibrated against a task mix that may predate your team's AI rollout. Ask what the band assumes about AI tooling.
- **Where in the band are you?** If you are at the midpoint, the question is what would move you to the upper quartile, and the answer should be a specific, time-bound deliverable — not tenure.

A useful move is to propose a 90-day trial with a measurable exit criterion: "If I reduce the rework ratio on the payments service from 0.22 to 0.15 within 90 days, we revisit the band." This converts a stalled negotiation into a scoped project with a defined payoff.

### "Should I take equity instead of cash?"

Equity compensates for future upside; salary compensates for present scope. If the concern driving the negotiation is that AI is absorbing your current tasks, cash addresses that concern directly and equity does not. A hybrid structure — part cash now, part equity with a short cliff — can work, but only if you understand the vesting schedule and the company's funding position. Do not accept a long cliff as a substitute for a raise you needed this cycle.

### "What if I have no AI tooling at work?"

The same measurement approach works on any artifact stream: design docs, RFCs, runbooks, migration scripts. Count what changed, count what was generated by an internal tool, and compute the same three ratios. If there is genuinely no automation, then the negotiation is about the *absence* of tooling — you can propose to build it and ask for scope and compensation in exchange. That is a forward-looking pitch, which is the whole point.

## Making it durable: instrument once, review on a schedule

A one-off audit is a talking point. A recurring audit is leverage. The difference is automation.

A nightly job that regenerates `audit.json` and stores it as a build artifact gives you a time series. When calibration season arrives, you can show a trend rather than a snapshot, and a trend is much harder to dismiss.

```yaml
# .github/workflows/ai-audit.yml
name: AI Audit
on:
  schedule:
    - cron: '0 2 * * *'
  workflow_dispatch:

jobs:
  audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: python -m ai_audit.cli --days 90 > audit.json
      - uses: actions/upload-artifact@v4
        with:
          name: ai-audit-${{ github.run_number }}
          path: audit.json
```

Two details matter here. First, `fetch-depth: 0` is required — a shallow clone truncates the history and your 90-day window will silently return partial data. Second, uploading an artifact rather than pushing to an external metrics endpoint keeps the data inside your existing access controls, which removes a plausible objection before it is raised.

If your organization already runs a metrics stack, exporting the same three numbers as gauges is straightforward with any client library, but it is optional. The artifact alone is sufficient for a negotiation; the dashboard is a convenience.

A smoke test guards against silent breakage:

```python
# tests/test_cli.py
import json
import subprocess


def test_audit_output_is_well_formed():
    out = subprocess.run(
        ["python", "-m", "ai_audit.cli", "--days", "7"],
        check=True, capture_output=True, text=True,
    ).stdout
    data = json.loads(out)
    assert 0.0 <= data["ai_pct"] <= 1.0
    assert 0.0 <= data["domain_uniqueness"] <= 1.0
    assert data["rework_ratio"] >= 0.0
```

The test asserts bounds, not specific values — it catches a broken pipeline, not a changed codebase.

## Writing the memo

The deliverable is a one-page document, not a spreadsheet. Managers forward documents upward; they do not forward spreadsheets. Keep it to three sections:

1. **What the tooling does now.** State the AI-attributed change share and the exact command that produced it. Name the window and the attribution convention.
2. **What that changes about the role.** List the tasks that have shrunk and the tasks that have grown. Be specific: "I now spend roughly a day a week reviewing AI-generated migrations" is concrete; "my role has evolved" is not.
3. **What you are asking for.** Scope, level, or compensation — pick one primary ask and one fallback. Attach the 90-day exit criterion if the answer is "not now."

Commit the memo to a repository you control and share the link. A dated, versioned document is harder to wave away than a verbal request, and it gives your manager something concrete to take to their own manager.

## Common questions

**Is commit-trailer attribution accurate enough to base a raise on?**
It is a lower bound, not a precise measure. Present it as such. The defensible part of the argument is the domain-uniqueness list, which does not depend on attribution at all.

**How long a window should I use?**
Ninety days is a reasonable default — long enough to smooth out a quiet sprint, short enough to reflect the current tooling. Anything under 30 days is too noisy to be persuasive.

**What if my team does not label AI-assisted commits?**
Propose the convention first. A two-line addition to your PR template and a `Co-authored-by:` trailer in the commit hook is usually enough. Collect data for a few weeks before you negotiate.

**Does this work for non-coding roles?**
Yes, with substitution. Measure the artifact stream that defines your role — documents, tickets, incident reports — and apply the same three ratios.

## Your next 30 minutes

Add an AI-attribution convention to your repository if one does not exist, then run the audit script above against your last 90 days of history and save the output as `negotiation_audit.json`. Do not interpret it yet. Just get the number on disk — the interpretation is a separate, calmer exercise, and having the raw data in hand is what makes the rest of this process possible.
===END===
