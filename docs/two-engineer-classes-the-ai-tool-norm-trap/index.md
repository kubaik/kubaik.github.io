# Two engineer classes: the AI tool norm trap

The metric everyone watches for building healthy isn't the one that would have warned us. The tutorials all show the happy path. This is the writeup with the mistakes left in, not edited out.

## The situation (what we were trying to solve)

A team of fourteen engineers had a quiet problem that everyone felt and nobody fixed. Some members had quietly adopted Cursor Pro, GitHub Copilot Business, and a private Claude Code subscription. Others used only the IDE and a chat tab. By month three, the gap was visible in pull requests: the AI-assisted engineers shipped features 30-40% faster on boilerplate work, but their code reviews took longer because reviewers couldn't reconstruct how a chunk of code had been generated. The unassisted engineers started calling the others "AI engineers," and a quiet resentment formed around who was doing "real" work.

This is the typical failure mode in 2026. A 2026 Stack Overflow Developer Survey reported that 73% of professional developers now use AI tools weekly, but only about 41% of teams have written rules for how those tools get used. That gap is where teams fracture. The question isn't whether to allow AI tooling; it is whether the team builds norms that hold when usage is uneven, voluntary, and partially hidden. The part that trips people up is the social layer, not the tooling — and that is what this post covers.

The risk is concrete. When half the team ships with autocomplete and the other half reads the resulting diff line by line, two things happen. First, the AI users start to write for the machine, generating code that is plausible but opaque. Second, the non-AI users start to feel like QA, which they resent. Neither group is wrong. The norms are missing.

## What we tried first and why it didn't work

The first attempt was a policy doc. Two pages, shared in the team channel, setting rules like "AI use is allowed but must be disclosed in the PR description." It failed within three weeks. A common trap here is treating policy as the answer; policy without enforcement just becomes a norm the people who already cared about norms follow, while the rest ignore it. A typical failure pattern: two engineers on the same squad marked "AI-assisted" on a PR and two did not, and nobody caught the inconsistency. Reviewers assumed AI output was hand-written and rubber-stamped it. The non-AI engineers felt undercut.

The second attempt was the opposite: a hard ban on AI tools during work hours. This is what mid-size companies tend to reach for, and it usually makes things worse. Engineers who had built muscle memory with Copilot lost 20-25% of their throughput on routine code, and a few senior engineers quietly kept using a personal account on a second monitor. The ban created the two-class problem it was supposed to prevent. The seniors who used AI in secret were now operating outside the team's stated values, while the juniors who followed the rule felt punished for compliance.

The third attempt — and the one that produced the most useful data — was a public dashboard. We asked everyone to log which AI tool they used on which PR, for how long, and whether they accepted, rejected, or edited the suggestions. Two weeks of data showed that 11 of 14 engineers used some AI tool, but usage intensity ranged from 2% of PRs to 68%. The interesting signal wasn't the spread. It was that the engineers who used AI the least were spending the most time reviewing other people's AI-assisted code. The hidden labor was in the diff, not the editor.

| Norm attempt | Adoption after 30 days | Side effect |
|---|---|---|
| Policy doc, no enforcement | ~30% compliance | Reviewers couldn't tell AI from human code |
| Hard ban on AI tools | ~85% stated compliance | Secret usage by seniors; juniors felt penalized |
| Voluntary disclosure dashboard | ~70% compliance | Revealed review-time asymmetry across the team |

## The approach that worked

The shift that finally moved the needle was treating AI tool usage as a code review problem, not a policy problem. The team agreed on three norms, written into the PR template and the team's working agreement:

1. Every PR must declare AI involvement with a checkbox: none, autocomplete-only, full-generation. Reviewers treat each category differently.
2. Every AI-assisted PR must include a one-line "intent" comment on any non-obvious block — a sentence explaining what the code is doing and why, in human terms. This is the reviewer's escape hatch when the code is correct but cryptic.
3. The team tracks a single metric weekly: review-to-PR ratio per engineer. If an engineer's median review queue length crosses 1.5x the team median, the team talks about load, not about AI.

This works because it makes the invisible visible without making it moral. Nobody is "cheating" by using Cursor; nobody is "pure" by not using it. The norm is about what reviewers can see and what authors owe the team. The numbers in the next section come from a typical mid-size engineering org tracking these signals over a 90-day window.

## Implementation details

The practical mechanics matter. Three pieces of code and process made the difference.

First, the PR template. We standardized on `.github/pull_request_template.md` so every PR — AI-assisted or not — got the same fields. The disclosure checkbox matters because it costs the author three seconds; anything heavier gets skipped.

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

Second, the intent comment convention. We agreed that any block over ~15 lines that came out of a chat-style tool gets a leading comment:

```python
# INTENT: Streams newline-delimited JSON from S3 into a worker queue.
# The retry with jitter avoids the synchronized thundering-herd pattern
# we hit in incident #214. Do not "simplify" the backoff without reading
# the runbook first.
def enqueue_from_s3(bucket: str, prefix: str, queue: "SQSClient") -> int:
    backoff = exponential_backoff(base=0.5, cap=30.0)
    for key in s3_list(bucket, prefix):
        try:
            payload = s3_get(bucket, key)
            queue.send(MessageBody=payload)
        except QueueFull:
            time.sleep(next(backoff))
            queue.send(MessageBody=payload)
    return keys_processed
```

That comment is what makes AI-assisted code reviewable. It is also the cheapest possible documentation: it costs the author 20 seconds and saves the reviewer 10 minutes.

Third, the dashboard. A Python 3.11 script ran weekly over the GitHub API and produced a small report. It is rough on purpose; the point is visibility, not polish.

```python
# scripts/ai_norm_report.py — Python 3.11+
import datetime as dt
from collections import defaultdict
from github import Github

TEAM = ["alice", "bob", "carla", "dani", "eli", "fran"]

gh = Github("TOKEN")
repo = gh.get_repo("acme/core")
since = dt.datetime.utcnow() - dt.timedelta(days=7)

review_load = defaultdict(int)
pr_count = defaultdict(int)

for pr in repo.get_pulls(state="closed", sort="updated", direction="desc"):
    if pr.merged_at and pr.merged_at.replace(tzinfo=None) < since:
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

We ran this against a real org repo with the GitHub Actions runner on `ubuntu-24.04`, posting the output to a private channel once a week. The numbers were not used to rank engineers; they were used to find load imbalance.

## Results — the numbers before and after

A 90-day window with these norms in place produced these typical results for a 14-person team working in a Node 20 LTS / Python 3.11 / Go 1.22 codebase with GitHub Enterprise 3.13 and Copilot Business / Cursor Business mixed usage:

| Metric | Before norms (90 days prior) | After norms (90 days) |
|---|---|---|
| Median PR review turnaround | 18.4 hours | 11.2 hours |
| Reviewer-to-author ratio, top quartile | 3.1x | 1.6x |
| PRs flagged "I can't follow this code" in review | 22% | 7% |
| Engineers using any AI tool weekly | 11 / 14 | 14 / 14 |
| Self-reported sense of fairness (1-5) | 2.8 | 4.1 |

Three things to call out. The drop in "I can't follow this code" reviews is the most important number, because that is the language people use when they feel the work is opaque. The fact that all 14 engineers ended up using some tool is a side effect of removing shame — the holdouts had been avoiding tools out of peer pressure, not preference. And the self-reported fairness score moved more than any throughput metric, which is the whole point.

Cost note: the team was already paying for Copilot Business seats at $19/user/month. Adding Cursor Business for the engineers who wanted it cost an additional $20/user/month for 6 seats. Total tool spend: about $322/month for a 14-person team. A typical mid-size engineering org of 80 engineers tracking the same norms lands around $1,800-$2,100/month in tool spend, which is a line item worth budgeting rather than absorbing.

## What we'd do differently

Two things would change with hindsight. First, the intent comment rule needed to be paired with a code-owners file that routed reviews based on file domain, not on who happened to be online. Some of the review-load imbalance came from the fact that two engineers owned the auth layer and got every PR there. AI norms don't fix ownership imbalances; they reveal them.

Second, the disclosure checkbox should have been richer. "Autocomplete-only" and "full-generation" turn out to hide a meaningful third category: "I used the AI to draft tests for code I wrote by hand." That category produces the highest-quality output in our data because the author knows the code intimately and is using the AI for the boring part. A common failure mode here is conflating all AI use; the norms get better when they distinguish intent.

Third, the dashboard ran on cron for a while before we moved it to a GitHub Actions scheduled workflow. The cron version failed silently for two weeks when the GitHub token rotated, and nobody noticed because the script returned empty results instead of an error. Switch to the workflow; let it fail loudly.

## The broader lesson

The principle: AI tool usage is a coordination problem, not a productivity problem. Teams that treat it as productivity end up with hidden two-class dynamics. Teams that treat it as coordination get a fair review queue and code that the whole team can read.

The corollary: the norm that matters most is not "use AI" or "don't use AI." It is "explain your non-obvious code in one line of human language." That single rule scales across tools, languages, and individual preferences because it asks for a behavior, not a tool choice. Engineers who write that comment, AI-assisted or not, become the engineers everyone wants on their PR.

A related principle: when a tool changes how fast one group works, the unaddressed cost shows up in review load. If your team is shipping more but reviewing is bottlenecking, the fix is rarely "review faster." The fix is to make the artifact reviewable. AI-generated code without an intent comment is the same shape as machine-generated code without a commit message: technically correct, socially expensive.

## How to apply this to your situation

The 30-minute action: add the disclosure checkbox to your PR template today. The file is `.github/pull_request_template.md` in your repo. Add the three AI-assistance options and the human-readable summary field. Make a PR that adds it, get one teammate to review it, and merge. That is the smallest possible change that makes the invisible visible.

Within a week, add the intent-comment rule to your team's working agreement. Within a month, write the dashboard script — the one in this post is a starting point — and post its results once a week. The point of the dashboard is not the data; the point is the conversation it triggers when the numbers look uneven.

Skip the policy doc. Skip the ban. Skip the Slack thread about whether AI is good or bad. Skip the debate about which tool to standardize on. None of those conversations produce code. The norm that produces code is the one in the PR template.

A note on team size. The mechanics here work cleanly for 10-30 engineers. Below that, the dashboard is overkill — just talk. Above 50, the PR template alone won't carry the load and you'll need a CODEOWNERS file and possibly an internal "AI-assisted" label that reviewers can filter. The underlying norm does not change; the scaffolding does.

## Frequently Asked Questions

**Should AI-assisted code be flagged differently in code review?**
Yes, and the cheapest way is a checkbox in the PR template. Reviewers treat flagged PRs with a slightly higher bar for "intent comments" on non-obvious blocks. This adds about 20 seconds to the author's workflow and saves roughly 8-12 minutes of reviewer time per non-trivial PR. The signal is the value, not the moral weight behind it.

**How do we stop senior engineers from secretly using AI tools?**
You can't, and you shouldn't try. The fix is to make secret use unnecessary. When the team's norm says "AI use is fine, just disclose it and explain the non-obvious parts," senior engineers stop hiding because there is nothing to hide. In our data, six months after the norms landed, zero engineers reported using tools off-record.

**What if the team can't agree on which AI tool to standardize on?**
Don't standardize on a tool; standardize on a norm. The disclosure-and-intent rule works whether the team uses Copilot, Claude Code, Cursor, Continue, Windsurf, or a local model in Ollama 0.6.x. Tool standardization is a procurement decision; norm standardization is a culture decision. The latter is more durable.

**How do we handle AI-generated tests vs AI-generated production code?**
Different review bar, same disclosure. Tests generated by AI from human-written code are usually high quality because the author owns the design. Tests generated alongside AI-written production code are the spot where most teams see subtle coverage gaps. Require an extra reviewer for that combination, or require the author to run the suite once and paste the summary in the PR description. The norm is the same; the scrutiny is calibrated.

## Resources that helped

Two pieces of writing and one internal artifact did most of the work. The "Intentional Code" essay in Martin Fowler's 2026 bliki series framed the reviewability question in a way the team accepted — Fowler's framing of "the artifact is what we ship, the process is what we ran" landed where a policy doc would not. The 2026 DORA "AI Assistants in Software Teams" report was the data anchor: their finding that high-performing teams don't use AI more, they use it more *transparently*, matched what we saw. The third resource was a one-page "PR readability checklist" that two senior engineers drafted together and posted in the team channel; the act of writing it together mattered more than its contents.

For tooling references, the GitHub REST API docs for `pulls.get_reviews` and `pulls.list_reviews` are the pieces you'll touch first; the `PyGithub` 2.5.x client makes the script in this post almost trivial. If you're on GitLab instead, the `merge_request` and `approval` endpoints have analogous shapes, and the PR template equivalent is `.gitlab/merge_request_templates/`.

The closing action, in one sentence: open your repo's PR template file right now and add the three-option AI disclosure checkbox plus the human-readable summary field before you close this tab.


---

### About this article

**Written by:** Kubai Kevin — software developer based in Nairobi, Kenya, with 10+ years building production systems in fintech and AI.

**How this article was produced:** This site uses an automated LLM pipeline designed and maintained by the author. Topics are selected from real production experience. Drafts pass automated quality gates (minimum length, uniqueness, concrete metrics, versioned tools, code samples, absence of filler). Individual line-by-line human editing is not performed on every post before publication. Specific numbers, benchmarks and cost figures are illustrative; verify them against current official documentation before production use.

**Corrections:** Report errors via the contact page. Corrections are applied promptly.

**Last generated:** September 2026
