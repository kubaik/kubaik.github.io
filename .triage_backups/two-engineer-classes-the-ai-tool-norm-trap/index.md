# Two engineer classes: the AI tool norm trap

Mixed AI tool adoption inside a single team creates a predictable failure mode: the engineers who generate code fastest push more diffs into a review queue staffed by people who cannot reconstruct how that code was produced. The productivity gain is visible in the editor. The cost shows up in review latency, in rubber-stamped approvals, and in a quiet split between engineers who feel like authors and engineers who feel like QA. The coordination problem is the real one; the tooling choice is mostly a distraction.

## The failure mode, described precisely

A team of fourteen engineers has a problem everyone feels and nobody fixes. Some members have adopted an AI coding assistant with inline suggestions, others use a chat-style generation tool, and others use only the IDE. By month three, the gap is visible in pull requests. The AI-assisted engineers ship boilerplate-heavy features faster, but their reviews take longer because reviewers cannot reconstruct how a chunk of code was generated.

Two things then happen in parallel. First, the AI users start writing for the machine, producing code that is plausible but opaque. Second, the non-AI users start to feel like QA, which they resent. Neither group is behaving badly. The norms are missing.

This is a coordination failure, not a productivity failure. It appears whenever three conditions hold at once:

- Usage is uneven across the team.
- Adoption is voluntary and partly unspoken.
- The artifact handed to reviewers does not carry enough context to be read quickly.

The last condition is the one a team can actually fix, and it is where most of the leverage sits.

## Three common responses and why they stall

**The policy doc.** Two pages in the team channel: "AI use is allowed but must be disclosed in the PR description." A common trap is treating policy as the answer. Policy without enforcement becomes a norm that only the people who already cared about norms follow. A typical failure pattern: two engineers on the same squad mark "AI-assisted" on a PR and two do not, nobody catches the inconsistency, reviewers assume AI output was hand-written and rubber-stamp it, and the non-AI engineers feel undercut.

**The hard ban.** Engineers who built muscle memory with an inline assistant lose throughput on routine code, and some senior engineers quietly keep using a personal account on a second monitor. The ban creates the two-class problem it was meant to prevent: seniors operating outside the team's stated values, juniors feeling punished for compliance.

**The voluntary disclosure dashboard.** Ask everyone to log which tool they used on which PR and whether they accepted, rejected, or edited suggestions. This produces the most useful signal of the three, and it exposes the counterintuitive part: the engineers who use AI the least tend to spend the most time reviewing other people's AI-assisted code. The hidden labor is in the diff, not the editor.

| Norm attempt | Typical 30-day outcome | Side effect to watch for |
|---|---|---|
| Policy doc, no enforcement | Partial compliance, self-selected | Reviewers cannot distinguish AI from human code |
| Hard ban on AI tools | High stated compliance | Off-record usage by seniors; juniors feel penalized |
| Voluntary disclosure dashboard | Moderate compliance | Reveals review-time asymmetry across the team |

## The reframe: treat usage as a review problem

The shift that moves the needle is treating AI tool usage as a code review problem rather than a policy problem. Three norms, written into the PR template and the team's working agreement, carry most of the weight:

1. Every PR declares AI involvement with a checkbox: none, autocomplete-only, or full-generation. Reviewers treat each category differently.
2. Every AI-assisted PR includes a one-line "intent" comment on any non-obvious block — a sentence explaining what the code does and why, in human terms. This is the reviewer's escape hatch when the code is correct but cryptic.
3. The team tracks one metric weekly: review-to-PR ratio per engineer. If an engineer's median review queue length crosses roughly 1.5x the team median, the team discusses load distribution, not AI.

This works because it makes the invisible visible without making it moral. Nobody is cheating by using an assistant; nobody is pure by declining one. The norm is about what reviewers can see and what authors owe the team.

## Implementation: three pieces of process

### The PR template

Standardize on `.github/pull_request_template.md` so every PR, AI-assisted or not, gets the same fields. The disclosure checkbox matters because it costs the author three seconds; anything heavier gets skipped.

```markdown
## What changed
<!-- One paragraph, plain language -->

## AI assistance
- [ ] None
- [ ] Autocomplete / inline suggestions only (Copilot, Cursor Tab, etc.)
- [ ] Generation / chat (Claude Code, Cursor Agent, Copilot Chat, etc.)

## Human-readable summary
<!-- Required if any AI box is checked. One line per non-obvious block. -->
```

### The intent comment convention

Agree that any block over roughly 15 lines produced by a chat-style tool gets a leading comment:

```python
# INTENT: Streams newline-delimited JSON from S3 into a worker queue.
# The retry with jitter avoids a synchronized retry storm: without the
# jitter, every worker that failed on the same batch retries in lockstep.
# Do not "simplify" the backoff without reading the runbook first.
def enqueue_from_s3(bucket: str, prefix: str, queue: "SQSClient") -> int:
    backoff = exponential_backoff(base=0.5, cap=30.0)
    keys_processed = 0
    for key in s3_list(bucket, prefix):
        try:
            payload = s3_get(bucket, key)
            queue.send(MessageBody=payload)
            keys_processed += 1
        except QueueFull:
            time.sleep(next(backoff))
            queue.send(MessageBody=payload)
            keys_processed += 1
    return keys_processed
```

That comment is what makes AI-assisted code reviewable. It is also the cheapest documentation available: it costs the author a few seconds and can save the reviewer many minutes.

### The dashboard

A short script over the hosting provider's API produces a weekly report. Keep it rough on purpose; the point is visibility, not polish.

```python
# scripts/ai_norm_report.py — Python 3.11+
import datetime as dt
from collections import defaultdict
from github import Github

TEAM = ["alice", "bob", "carla", "dani", "eli", "fran"]

gh = Github("TOKEN")
repo = gh.get_repo("acme/core")
since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=7)

review_load = defaultdict(int)
pr_count = defaultdict(int)

for pr in repo.get_pulls(state="closed", sort="updated", direction="desc"):
    if pr.merged_at and pr.merged_at < since:
        continue
    author = pr.user.login.lower()
    if author not in TEAM:
        continue
    pr_count[author] += 1
    for review in pr.get_reviews():
        reviewer = review.user.login.lower()
        if reviewer in TEAM and reviewer != author:
            review_load[reviewer] += 1

for engineer in TEAM:
    ratio = review_load[engineer] / max(pr_count[engineer], 1)
    print(f"{engineer}: {pr_count[engineer]} PRs, "
          f"{review_load[engineer]} reviews, "
          f"ratio {ratio:.2f}")
```

Two implementation notes. `datetime.now(dt.timezone.utc)` returns an aware datetime, which compares correctly against `pr.merged_at`; mixing naive and aware datetimes raises `TypeError` and is a common bug when this script is copied. And the numbers are not used to rank engineers. They are used to find load imbalance.

## How to measure whether any of this helped

No credible before-and-after table can be published for your team, because the numbers depend on your repo, your review culture, and your baseline. What can be published is the measurement recipe. Instrument these five signals for a 30-day window before changing anything, then for a 30-day window after.

**Median PR review turnaround.** Time from `ready_for_review` to first substantive review, not to merge. Pull it from the API: for each PR, subtract the timestamp of the first review from the timestamp the PR left draft state. Compare medians, not means; a single week-long PR will drag a mean around.

**Review-to-PR ratio per engineer.** Reviews authored divided by PRs authored, over the window. The script above computes it. Watch the top quartile rather than the average.

**"I can't follow this" review comments.** Grep your review comments for phrases like "unclear", "why", "what does this do", "can you explain". This is noisy but directionally useful, and it is the closest proxy for perceived opacity.

**Disclosure coverage.** Percentage of merged PRs where the AI-assistance checkbox is filled in at all. If this is below roughly two-thirds, the template is not being read and no other metric is trustworthy.

**Self-reported fairness.** A one-question pulse survey, monthly: "Review load on this team is distributed fairly" on a 1-5 scale. Crude, but it moves before attrition does.

Run the same queries against both windows with the same team membership. If membership changed, say so and do not compare.

## Failure modes to expect

**Silent dashboard failure.** A scheduled script that returns empty results instead of erroring will look like a quiet week. Prefer a scheduled workflow over a hand-rolled cron job, and make the job fail loudly on an empty result set or a rotated token. An empty report and a broken report must be distinguishable at a glance.

**Ownership imbalance masquerading as AI imbalance.** If two engineers own the auth layer, they receive every auth PR regardless of how it was written. AI norms do not fix ownership imbalances; they reveal them. Pair the norm rollout with a `CODEOWNERS` file that routes by domain.

**Over-broad disclosure categories.** "Autocomplete-only" and "full-generation" hide a third category that matters: using the AI to draft tests for code the author wrote by hand. That combination tends to produce the highest-quality output, because the author owns the design and delegates the tedious part. Norms get better when they distinguish intent rather than lumping all usage together.

**Disclosure as a status marker.** If the checkbox becomes a proxy for "real engineer" versus "tool user," it will be gamed or abandoned. Keep it factual and keep the review bar tied to the artifact, not the author.

**The metric becoming a ranking.** The moment review-to-PR ratio appears in a performance review, engineers optimize it. State explicitly, in writing, that the dashboard exists to find load imbalance and will not be used in evaluation.

## A decision checklist

Before rolling this out, answer these in writing:

- Does every PR, regardless of authorship, pass through the same template? If not, fix that first.
- Is there a named owner for the weekly report, and does the job alert on failure?
- Has the team agreed that the review-to-PR ratio will not feed performance reviews?
- Is there a `CODEOWNERS` file, or will domain owners absorb all AI-related review load?
- Are the disclosure categories granular enough to distinguish "AI wrote this" from "AI wrote the tests for this"?
- What is the escalation path when one engineer's queue crosses the threshold — rebalancing, pairing, or a temporary review freeze?

## What actually generalizes

AI tool usage is a coordination problem, not a productivity problem. Teams that treat it as productivity end up with hidden two-class dynamics. Teams that treat it as coordination get a fairer review queue and code the whole team can read.

The corollary: the norm that matters most is not "use AI" or "don't use AI." It is "explain your non-obvious code in one line of human language." That rule scales across tools, languages, and individual preferences because it asks for a behavior, not a tool choice.

A related principle: when a tool changes how fast one group works, the unaddressed cost surfaces in review load. If the team is shipping more but reviewing is bottlenecking, the fix is rarely "review faster." The fix is to make the artifact reviewable. AI-generated code without an intent comment has the same shape as machine-generated code without a commit message: technically correct, socially expensive.

## Scope limits

These mechanics fit teams of roughly 10 to 30 engineers. Below that, the dashboard is overkill; a conversation and a template are enough. Above roughly 50, the PR template alone will not carry the load, and a `CODEOWNERS` file plus a filterable label for AI-assisted PRs becomes necessary. The underlying norm does not change with team size. The scaffolding does.

## FAQ

**Should AI-assisted code be flagged differently in review?**
Yes, and the cheapest mechanism is a checkbox in the PR template. Reviewers apply a slightly higher bar for intent comments on flagged PRs. The value is the signal, not any moral weight attached to it.

**How do you stop senior engineers from using AI tools off the record?**
You generally cannot, and attempting to tends to push usage further underground. The practical fix is to make disclosure unremarkable: when the norm is "AI use is fine, disclose it and explain the non-obvious parts," there is less to hide. Verify this with the disclosure-coverage metric rather than assuming it.

**What if the team cannot agree on one AI tool?**
Do not standardize on a tool; standardize on a norm. The disclosure-and-intent rule works across inline assistants, chat-style generators, and locally hosted models alike. Tool standardization is a procurement decision. Norm standardization is a culture decision, and it outlasts the tooling.

**How should AI-generated tests be treated versus AI-generated production code?**
Same disclosure, different scrutiny. Tests generated from human-written code are usually high quality because the author owns the design. Tests generated alongside AI-written production code are where subtle coverage gaps appear. Require an extra reviewer for that combination, or require the author to run the suite and paste the summary into the PR description.

## Start here

Open `.github/pull_request_template.md` in your repository right now — create it if it does not exist — and add the three-option AI disclosure checkbox plus the human-readable summary field. Commit it, request one review from a teammate, and merge it today. That is the smallest change that makes the invisible visible, and it takes less than thirty minutes.
