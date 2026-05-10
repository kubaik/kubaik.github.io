# Claude reviews my code — here’s what works

I ran into this while migrating a production service under a hard deadline. The official docs covered the happy path well. This post covers everything else.

## The one-paragraph version (read this first)

I’ve tried every AI coding assistant since GitHub Copilot’s 2021 beta, and for non-trivial changes, Anthropic’s Claude is the only one that consistently catches real bugs instead of just writing plausible boilerplate. I use it to review my own pull requests before I hit ‘merge’: it catches logic errors, inconsistent error handling, and even subtle performance issues. But it fails when the code relies on undocumented tribal knowledge, when the problem isn’t code at all, or when the review requires domain expertise I haven’t given it. In 12 months and 472 reviews, it saved me an estimated 14 hours of manual review time, but it also wasted 3 hours reviewing code that relied on a legacy cron job no one documented. Below is the exact workflow I use, why it works, and where it breaks.


## Why this concept confuses people

Most developers think of AI code review as a replacement for human reviewers: drop in a PR, let the AI leave comments, and call it done. That’s wrong. The confusion starts with the word *review*. In human terms, a code review is a conversation about design trade-offs, business alignment, and long-term maintainability. AI doesn’t have a business stake, so it can’t weigh those trade-offs. What it *can* do is act like a senior engineer who has perfect memory of every line you’ve ever written — but zero context about why the business chose feature X over feature Y. Many teams burn weeks trying to automate away human reviewers, only to realize the AI comments are noise because it doesn’t understand the hidden constraints (compliance rules, vendor SLAs, or the fact that the ‘legacy cron’ is actually a fragile cron that breaks every quarter).


## The mental model that makes it click

Think of Claude as a *pair programmer with infinite patience and no ego*, but also no ability to ask clarifying questions unless you prompt it to. The key insight is that it excels at *pattern matching against known anti-patterns* and *structural consistency*, not at understanding intent. I model my interactions like this:

1. *Intent layer*: I explain the PR’s goal in plain English at the top of the prompt — e.g., ‘This PR refactors the payment retry logic to use exponential backoff and adds observability hooks. It must not break the 99.9% success rate we promised our payment provider.’
2. *Pattern layer*: I ask it to flag known issues — SQL injection risks, missing null checks, inconsistent logging format, or race conditions in async code.
3. *Cost layer*: I ask it to estimate runtime cost changes — e.g., ‘Will this new loop add 20ms per request under peak load?’

Claude doesn’t reason about intent, but if you give it the intent, it can simulate what a senior engineer *might* do if they had perfect memory of your codebase. That’s powerful, but it’s also fragile: if your intent is wrong or incomplete, the review will optimize for the wrong thing.

It also helps to think of it like a *syntax-aware diff tool with a grudge*. It spots things humans miss — like a missing `break` in a switch-case that’s been there since 2019 — but it can’t tell you whether that missing `break` is actually a bug or intentional obfuscation for a future hackathon demo.


## A concrete worked example

Here’s a real PR I opened in June 2024. The change was small: add a 30-second cache for user profile lookups to reduce database load. The PR diff is 34 lines. Below is the exact prompt I used and the review it produced.

**Prompt (sent as a comment on the PR):**
```
PR context:
- Goal: Add 30-second cache for user profile lookups to reduce DB load.
- Must preserve existing error behavior: if profile lookup fails, return 500 to caller.
- Cache key format: `profile:{user_id}`
- Cache store: Redis with 30s TTL.
- Observability: Add `cache_hit` metric, increment on hit; log warning if Redis is down.
- No breaking changes to public API.

Review this diff for:
1. Logic errors (missing null checks, race conditions).
2. Error propagation (does Redis failure trigger 500 or log and continue?).
3. Cache stampede risk (multiple requests for same user_id within 30s).
4. Performance regression (added Redis call per request could slow down 200ms requests by >5%).
5. Logging format consistency (ensure all logs use JSON and include request_id).
```

**Claude’s review (verbatim, trimmed for brevity):**
```
1. Logic error: Missing null check on user_profile after deserialization (line 42). If Redis returns null, the code will crash with a TypeError when accessing `user_profile['email']`.

2. Race condition: If two requests for the same user_id arrive within the same millisecond, both will miss the cache, trigger Redis writes, and potentially overload Redis. Add a distributed lock or use a short random delay before write.

3. Error propagation: If Redis is down, the code catches the exception and returns a 500 error, which matches the requirement. Good.

4. Performance regression: The added Redis call adds ~4ms per request on production hardware, which is within the 5% budget. No regression expected.

5. Logging: Missing `request_id` in the warning log when Redis is down (line 56). Inconsistent with the rest of the codebase.
```

**What I fixed:**
- Added null check for `user_profile` (4 lines).
- Added distributed lock using Redlock with 100ms TTL (12 lines).
- Added `request_id` to the Redis down warning log.

**What I ignored:**
- The race condition fix added 12 lines of code, which increases latency by ~1ms per request. Given the 4ms baseline, the total is still under 5ms, so I accepted the trade-off.

**Surprise:**
I assumed the cache stampede was handled by the TTL, but I’d forgotten that two fast requests could both miss the cache and hammer Redis. The lock fix added complexity, but it cut peak Redis QPS from 1,200 to 450 during load tests — a 62% reduction.


## How this connects to things you already know

If you’ve ever used `git blame` to trace a bug back to a junior engineer’s Friday commit, you’re already doing a manual version of what Claude does at scale. The difference is speed and coverage: Claude can scan 10,000 lines of code in 3 seconds and flag every instance of a pattern it recognizes as risky. It’s like having a senior engineer who remembers every line of code you’ve ever written, but who also has the attention span of a goldfish — it forgets the context the moment the prompt ends.

It also resembles a *static analyzer with a chat interface*. Tools like SonarQube or ESLint catch syntax-level issues, but they don’t understand intent. Claude, when given intent, can simulate what a senior engineer *would* flag if they had perfect memory of your codebase. But unlike static analyzers, it can explain its reasoning in plain English — which is why it’s useful for teams that don’t have senior engineers on every team.


## Common misconceptions, corrected

1. *Myth: ‘Claude can replace a human reviewer.’*
   No. It can’t weigh trade-offs like ‘we’re shipping next week so we’ll merge this even though it adds 200ms latency.’ It also can’t negotiate with stakeholders or explain why a change breaks a compliance rule. In my experience, teams that try to replace human reviewers with AI end up with lower code quality because the AI doesn’t understand the hidden constraints.

2. *Myth: ‘Claude catches all bugs.’*
   It catches *structural* bugs — missing null checks, race conditions, inconsistent logging — but it misses *semantic* bugs. For example, if you change a function’s behavior but don’t update the docstring, Claude won’t flag it unless you explicitly ask. It also won’t catch bugs that require domain knowledge, like ‘this financial calculation off-by-one error only matters on leap years.’

3. *Myth: ‘Claude is free.’*
   The API is cheap for small repos, but costs scale linearly with codebase size and review complexity. I measured my 2024 spend: $187 for 472 reviews, averaging 1.3 API calls per review and 1,200 tokens per call. At 2024 pricing ($0.000008 per token for input, $0.000024 for output), that’s ~$1.10 per 1,000 tokens. The hidden cost is time: writing good prompts takes longer than writing a quick code comment for a human reviewer.

4. *Myth: ‘Claude works out of the box.’*
   It does not. I spent two weeks tweaking prompts, adding examples, and building a small library of helper functions to standardize logging, error handling, and metric naming. Without those, the reviews were noisy and inconsistent.


| Misconception | Reality | My fix |
|----------------|---------|--------|
| Replaces human reviewers | It can’t weigh trade-offs or understand hidden constraints | Use it to catch structural issues, not design decisions |
| Catches all bugs | Misses semantic and domain-specific bugs | Pair it with targeted tests and human review |
| Free to use at scale | Costs scale linearly with codebase size | Budget 2–3% of dev time for prompt engineering and review tuning |
| Works out of the box | Requires prompt engineering and helper libraries | Build a prompt template library and CI integration |


## The advanced version (once the basics are solid)

Once you’re comfortable with basic reviews, you can push Claude further by giving it *staged prompts* and *automated rollback triggers*.

**Staged prompts:**
Instead of one monolithic prompt, break the review into stages:
1. *Pattern scan*: Ask for high-risk patterns (SQL injection, missing auth checks).
2. *Behavioral diff*: Ask how the change affects runtime behavior (latency, memory, error rates).
3. *Regression risk*: Ask for likely regressions given past incidents.

I use a Python script to generate these prompts dynamically based on the files changed. For example, if the PR touches any SQL files, I add a stage that asks for SQL injection risks. If it touches async code, I add a stage for race conditions.

**Automated rollback triggers:**
I use a GitHub Action that runs Claude in review mode on every PR, then posts a summary comment. If the summary contains the word *‘rollback’*, the action automatically adds a label `rollback-candidate` and pings the on-call engineer. In 6 months, this flagged 8 PRs that later caused incidents — a 15% reduction in post-deploy rollbacks.

**Cost optimization:**
I switched from Anthropic’s default model (Claude 3 Opus) to Sonnet for reviews because it’s 3x cheaper and only 2% less accurate for my use case. The trade-off is slightly higher latency (5–7 seconds vs 2–3 seconds for Opus), but it’s acceptable for non-blocking reviews.

**Surprise:**
The biggest win wasn’t catching bugs — it was *standardizing error handling*. Before Claude, my team had 7 different ways to handle Redis failures. After enforcing a standard pattern (catch, log, return 500, increment metric), on-call incidents dropped by 30% because engineers knew exactly what to expect.


## Quick reference

- **Prompt template:**
  - Start with PR context (goal, constraints, non-goals).
  - List specific risks to check (logic errors, race conditions, error propagation).
  - Ask for performance impact estimates and logging consistency.
  - End with a call to action: ‘List top 3 issues by severity.’

- **Tools I use:**
  - Prompt library: `claude-review-prompts` (private repo, 24 templates).
  - CI integration: GitHub Action `claude-review-action` (uses Sonnet by default).
  - Cost control: daily budget alert at $5/day; auto-cancel if exceeded.
  - Logging: all reviews logged to Elasticsearch for trend analysis.

- **Metrics to track:**
  - Review time: 3–7 minutes per PR (vs 15–30 minutes for human review).
  - False positives: 12% (flagged issues that weren’t real bugs).
  - False negatives: 3% (missed real bugs that caused incidents).
  - Cost per review: $0.04 average (Sonnet model, 1,200 tokens).

- **Common failure modes:**
  - Prompt too vague: leads to noisy reviews.
  - Model too old: misses new language features (e.g., Python 3.12 f-strings).
  - No rollback trigger: teams ignore the reviews.

- **When to avoid:**
  - PRs that rely on undocumented tribal knowledge.
  - Changes that require domain expertise (e.g., tax calculation updates).
  - Reviews that need stakeholder negotiation (e.g., ‘should we ship this?’).


## Further reading worth your time

- [Anthropic’s prompt engineering guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering) — Skip the intro; go straight to the section on ‘Contextual examples’.
- [Martin Fowler on code review patterns](https://martinfowler.com/articles/code-review.html) — Read the ‘Checklist’ section and compare it to the prompts I use.
- [Google’s Engineering Practices documentation](https://google.github.io/eng-practices/) — Focus on the ‘Small CLs’ and ‘Self Reviews’ sections.
- [Redis performance tuning guide](https://redis.io/docs/management/optimization/benchmarks/) — Useful when Claude flags Redis bottlenecks.
- [Redlock algorithm deep dive](https://redis.io/docs/reference/patterns/distributed-locks/) — The paper I skimmed before adding locks to my cache logic.


## Frequently Asked Questions

**How do I stop Claude from flagging style issues as bugs?**
Use a two-stage prompt. First, ask it to ignore style issues (indentation, line length) unless they directly affect functionality. Second, provide a style guide excerpt as context — e.g., ‘Follow the team’s Python style guide: no line breaks inside parentheses, 4-space indents.’ I maintain a small YAML file with style rules and inject it into the prompt for Python PRs.


**Can I use Claude to review Terraform or SQL files?**
Yes, but with caveats. For Terraform, I ask it to flag hardcoded secrets, missing tags, and inconsistent naming. For SQL, I ask it to check for SQL injection risks, missing indexes, and transaction boundaries. The key is to provide schema context — e.g., ‘This query runs against a table with 10M rows; flag missing indexes.’ Without schema context, the reviews are noisy.


**What’s the biggest mistake teams make when adopting this?**
They treat it as a drop-in replacement for human reviewers. The worst case I’ve seen: a team replaced all human reviews with AI, then shipped a change that broke a vendor API because the AI didn’t understand the SLA constraints. The fix was to re-introduce human review for any change that touches external APIs or compliance rules.


**How do I handle false positives from Claude?**
Log them. I use a simple Google Sheet with columns: PR link, false positive description, root cause, fix. After 50 false positives, I tweak the prompt to exclude that pattern. For example, I initially flagged all `try/except` blocks as risky, but later realized many were intentional (e.g., parsing user input). After logging 14 false positives, I added an exception to the prompt: ‘Ignore try/except unless the exception type is specific (e.g., `ValueError`).’


## Where it fails — and what to do instead

Claude fails in three scenarios:

1. **Undocumented tribal knowledge.** Example: A legacy cron job runs every night and updates a materialized view. The PR changes the cron schedule, but the AI doesn’t know the cron job is the only thing that prevents a daily analytics report from breaking. Human reviewer catches it; AI misses it.

2. **Domain-specific logic.** Example: A financial calculation change that off-by-one errors only on leap years. The AI doesn’t know leap years matter; human reviewer with finance background catches it.

3. **Negotiation-heavy changes.** Example: A PR that refactors an API to use a new vendor, but the vendor’s SLA is worse. The AI can’t weigh the trade-off between ‘faster time-to-market’ and ‘higher vendor risk.’ Human reviewer with product and legal context catches it.

**What to do instead:**
- For tribal knowledge: Add a ‘Known Constraints’ section to your PR template. Ask reviewers to flag any undocumented assumptions.
- For domain logic: Pair the AI review with a domain expert review. For example, if the PR touches finance code, ask a finance engineer to sign off.
- For negotiation-heavy changes: Keep human review for any change that affects SLAs, compliance, or vendor contracts.


## The one metric that matters

I track *post-deploy rollback rate* before and after adopting Claude reviews. In the 6 months before Claude, my team had 12 rollbacks (2.1% of deploys). In the 6 months after, we had 7 rollbacks (1.2% of deploys). The difference isn’t huge, but it’s consistent — and the rollbacks we avoided were the ones that would have taken 2–3 hours to debug.


## Your next step

Pick one PR this week, write a two-sentence prompt explaining the goal and the risks you care about, and run Claude on it. Don’t try to automate it yet — just see if it catches something you missed. If it does, save the prompt and reuse it. If it doesn’t, tweak the prompt and try again. The goal isn’t to replace human review; it’s to make human review faster and more consistent by offloading the easy, structural checks to an AI that never gets tired.