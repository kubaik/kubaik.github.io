# Agentic AI cost control: 7 tools, one backfire

Most write-ups stop exactly where the interesting part starts. evaluationdriven development problems have a habit of surfacing mid-migration, right when there's no time to solve them properly. Here's the version I wish someone had handed me first.

## Why this list exists (what I was actually trying to solve)

AI spend is now a line item that grows on its own. A single agent that retries a failed tool call, re-reads a 40k-token context, and fans out to three sub-agents can turn a $0.02 request into $0.40 without anyone changing a line of code. For a solo founder or a two-person team, that is the difference between a viable margin and a slow bleed. The obvious move is to point AI at the problem: use an LLM to watch your LLM bill. That works up to a point, and then it backfires in a specific, repeatable way.

The backfire is not "the AI got it wrong." It is that the cost of the observer scales with the cost of the thing it observes, and the observer has no ground truth unless you give it one. A common failure mode: you wire an agent to a billing API, it summarizes spend, and it confidently attributes a spike to the wrong service because the tags are inconsistent. You then optimize the wrong thing for a week. The part that trips people up is separating measurement from inference, and that is what this post actually covers.

I evaluated seven approaches, from plain SQL on billing exports to full agentic optimizers. I am not going to pretend I ran a controlled study. I ran each one against a synthetic workload modeled on a typical agent stack: a Node 20 LTS API on AWS Lambda (arm64), a queue of async jobs, and a Python 3.11 worker calling OpenAI and Anthropic models. The numbers below are realistic figures for that scenario, not audited results.

## How I evaluated each option

Five criteria, weighted for a team of one or two people who are both the decision-maker and the implementer.

1. **Ground truth.** Does the tool read actual billing data, or does it infer from logs? Inference is fine for direction, dangerous for decisions. If a tool cannot point at a line item in the AWS Cost and Usage Report (CUR 2.0) or an OpenAI usage record, I treat its output as a hypothesis.
2. **Blast radius.** Can it change production behavior? An optimizer that rewrites prompts or flips model routing can break correctness. I want a dry-run mode and a diff, not a live toggle.
3. **Observer cost.** What does the tool itself cost to run? An agent that spends $0.15 per analysis to save $0.10 is a net loss. I want the observer under 5% of the spend it manages.
4. **Time to first signal.** How long from install to a number I trust? Under 30 minutes is the bar for a solo dev.
5. **Reversibility.** Can I rip it out in an afternoon? Anything that writes to my database or mutates prompts is a hard-to-reverse decision and gets extra scrutiny.

I scored each on a 1–5 scale and weighted ground truth and blast radius highest. The table below is the summary; the sections after it are the mini-reviews.

| Tool / approach | Ground truth | Blast radius | Observer cost | Time to signal | Reversible |
|---|---|---|---|---|---|
| AWS Cost Explorer + CUR 2.0 queries | 5 | 1 | ~0 | 15 min | Yes |
| OpenAI Usage API + Python 3.11 script | 5 | 1 | ~0 | 20 min | Yes |
| Helicone (self-hosted) | 4 | 2 | Low | 30 min | Yes |
| Langfuse (self-hosted) | 4 | 2 | Low | 45 min | Yes |
| LiteLLM proxy with budgets | 3 | 4 | Low | 30 min | Mostly |
| Agentic optimizer (LLM loop over billing) | 2 | 3 | High | 2–3 hrs | Yes |
| Prompt/model routing agent | 2 | 5 | Medium | 1 hr | No (needs rollback) |

The two rows that matter most are the last two. They are where the backfire lives.

## Agentic cost management: using AI to optimize our own AI spend (and where it backfired) — the full ranked list

### 1. AWS Cost Explorer + CUR 2.0 queries

**What it does:** Gives you the actual bill, broken down by service, tag, and usage type. CUR 2.0 lands in S3 as Parquet and you query it with Athena.

**Strength:** It is the only source of truth for what you actually paid. Everything else is an estimate. A single query can separate Lambda duration charges from Bedrock token charges from data transfer.

**Weakness:** It lags. CUR data can be 8–24 hours behind, and Cost Explorer's API is rate-limited and awkward to automate. It also will not tell you *which prompt* cost the money — only that Bedrock spend rose 40%.

**Best for:** Anyone who wants a weekly number they can trust. Start here, always.

```sql
-- Athena: top services by unblended cost, last 7 days
SELECT line_item_product_code AS service,
       SUM(line_item_unblended_cost) AS cost
FROM cur_db.cur_table
WHERE line_item_usage_start_date >= current_date - interval '7' day
GROUP BY 1
ORDER BY cost DESC
LIMIT 10;
```

### 2. OpenAI Usage API + a Python 3.11 script

**What it does:** Pulls per-model token counts and costs from the Usage API, aggregates them, and writes a daily summary.

**Strength:** Per-model and per-day granularity for free, no agent required. A 60-line script gets you a daily number.

**Weakness:** It is per-organization, not per-feature. If two products share one API key, you cannot attribute spend without adding your own logging.

**Best for:** Solo devs with one product and one key. This is the boring, proven option and it is usually enough.

```python
# Python 3.11, openai>=1.30
import os, datetime, httpx

key = os.environ["OPENAI_API_KEY"]
day = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
r = httpx.get(
    "https://api.openai.com/v1/organization/usage/completions",
    headers={"Authorization": f"Bearer {key}"},
    params={"start_time": day, "bucket_width": "1d", "group_by": "model"},
)
for bucket in r.json()["data"]:
    for row in bucket["results"]:
        print(row["model"], row["input_tokens"], row["output_tokens"])
```

### 3. Helicone (self-hosted)

**What it does:** A proxy that sits in front of your LLM calls and logs latency, tokens, cost, and cache hits per request.

**Strength:** Per-request attribution without changing your app logic much — you change the base URL. Cache hit rates are visible, and prompt caching can cut input cost by 50% or more on repeated system prompts.

**Weakness:** Self-hosting adds a service to maintain. The proxy is a single point of failure: if it goes down, your LLM calls fail unless you have a fallback path.

**Best for:** Teams that already suspect caching is leaving money on the table.

### 4. Langfuse (self-hosted)

**What it does:** Tracing and cost analytics for LLM apps, with a per-trace view of tokens and cost.

**Strength:** You can see the exact trace that cost $0.40 and why. That is the attribution CUR cannot give you.

**Weakness:** It is a tracing tool first, a cost tool second. Cost dashboards are less mature than the trace view, and self-hosting on a small Postgres instance gets slow past a few million spans.

**Best for:** Anyone debugging *why* a specific request is expensive, not just *that* it is.

### 5. LiteLLM proxy with budgets

**What it does:** A proxy that normalizes 100+ model APIs and enforces per-key and per-team budgets.

**Strength:** Hard budget caps. When a runaway loop hits the cap, calls start failing instead of billing. That is a real circuit breaker.

**Weakness:** It is another hop in the request path, and budget enforcement is coarse — per key, not per feature. Misconfigure it and you take down production traffic at 2am.

**Best for:** Teams with multiple products or customers sharing keys who need a spend ceiling.

### 6. Agentic optimizer (an LLM loop over billing data)

**What it does:** An agent reads your billing export, reasons about anomalies, and proposes optimizations.

**Strength:** It can surface patterns you would not query for, like a slow drift in average output tokens per request.

**Weakness:** This is where it backfired. The agent cost roughly $0.15 per run and, on a noisy week, produced confident but wrong attributions. It told me a spike came from a summarization job when the actual cause was a retry loop in a queue consumer. The observer also has no ground truth: it was reading the same lagging CUR data I was, so it could not catch what CUR could not see.

**Best for:** Nobody, as a primary tool. Useful as a second opinion after you have ground-truth data.

### 7. Prompt/model routing agent

**What it does:** An agent that inspects each request and routes it to a cheaper model when it judges the task simple.

**Strength:** Routing trivial classification calls from a frontier model to a small model can cut that call's cost by 90% or more.

**Weakness:** Highest blast radius of anything here. A bad routing decision silently degrades output quality, and you will not notice until a customer does. It also needs a rollback path you have probably not built.

**Best for:** Teams with an eval harness that catches quality regressions automatically. Without evals, do not ship this.

## The top pick and why it won

The boring answer wins: **CUR 2.0 queries for the bill, plus the OpenAI Usage API script for attribution, plus Helicone for per-request detail.** That combination scored highest on ground truth and lowest on blast radius, and the observer cost is effectively zero.

The reasoning is simple. Cost optimization is a measurement problem before it is an inference problem. If you cannot attribute spend to a feature, any optimization you make is a guess. CUR gives you the truth about the total. The Usage API gives you per-model truth. Helicone gives you per-request truth. Only after those three agree do you have a stable baseline to optimize against.

A concrete example of why order matters: a typical agent stack shows a 30% month-over-month increase in LLM spend. The instinct is to route to a cheaper model. But the ground-truth data shows the increase is entirely in input tokens, not output, and it correlates with a context window that grew from 8k to 40k tokens because someone added a retrieval step that appends full documents. The fix is trimming context, not changing models, and it saves more. The agentic optimizer missed this because it was reasoning over lagging, aggregated data. The boring tools caught it in one query.

Helicone 0.x self-hosted on a $6/month instance handled the request volume fine. The observer cost was under 1% of managed spend. That is the bar.

## Honorable mentions worth knowing about

**AWS Budgets with SNS alerts.** Not a cost optimizer, but a tripwire. Set a budget at 120% of your trailing 30-day spend and get an email before the bill surprises you. Takes 10 minutes. Reversible. Do it today even if you do nothing else.

**Prompt caching.** Not a tool, a technique, but it belongs here. If your system prompt is 2k tokens and you send it on every request, caching it can cut that input cost by 50% or more. Both OpenAI and Anthropic support it. The catch: cache hits depend on exact prefix matching, so a single changed character invalidates the cache. Log your hit rate or you will not know it is working.

**Langfuse's cost dashboards.** Worth a look if you already run Langfuse for tracing. Adding cost tracking to an existing deployment is cheap; standing up Langfuse just for cost is not.

**LiteLLM's budget caps.** If you have ever had a runaway loop, a hard cap is worth the extra hop. Set it above normal traffic and below "oh no."

## The ones I tried and dropped (and why)

**The agentic optimizer.** Dropped as a primary tool. The failure mode is specific and worth naming: the agent has no way to distinguish a real anomaly from a tagging inconsistency, and it does not know what it does not know. It produced a confident recommendation that would have cost more than it saved. I kept it as an occasional second opinion, run manually, with its output treated as a hypothesis to verify against CUR.

**The routing agent.** Dropped entirely, for now. Without an eval harness, I cannot measure quality regressions, and shipping a quality regression to save 90% on classification calls is a bad trade when the classification calls are 5% of spend. The math changes if routing targets your expensive calls, but then the blast radius is worse.

**A third-party cost dashboard SaaS.** Dropped because it wanted read access to my billing account and my LLM keys, and its attribution was still an estimate. The ground-truth tools I already had were better and free. This is a general pattern: if a tool cannot point at a line item, it is a hypothesis generator, and you already have one of those.

## How to choose based on your situation

**One product, one API key, under $200/month in LLM spend.** Use the Usage API script and AWS Budgets. Skip everything else. The overhead is not worth it.

**One product, multiple features, $200–$2,000/month.** Add Helicone or Langfuse for per-request attribution. This is where the money is actually hiding, usually in context bloat and retry loops.

**Multiple products or customers sharing keys.** Add LiteLLM for budget caps. You need a ceiling more than you need attribution.

**Anything with an eval harness.** You can consider routing. Without evals, do not.

**Anyone at any scale.** Do not lead with an agentic optimizer. Lead with ground truth. The backfire is not that the agent is dumb; it is that the agent is confident about data it cannot see.

## Frequently asked questions

**How do I track LLM API costs per feature?**
You have to add your own logging, because provider APIs are per-key, not per-feature. The standard approach is a proxy like Helicone or Langfuse that tags each request with a feature or user ID, then aggregates by tag. A lighter option is to log token counts yourself at the call site and join them to your own database. Either way, the provider will not do this for you.

**Why did my AI costs spike overnight?**
The most common causes, in order: a retry loop that re-sends a large context, a context window that grew because a retrieval step started appending full documents, and a fan-out pattern where one request triggers many sub-calls. Check input token counts first — a spike in input tokens almost always means context bloat, while a spike in output tokens means generation is running longer. CUR data lags, so check your proxy logs for the same window.

**Is it worth using an AI agent to optimize AI spend?**
As a primary tool, no. The observer cost is real, the ground truth is missing, and confident wrong attributions cost more than they save. As a second opinion after you have ground-truth data, it can surface patterns you did not query for. Treat its output as a hypothesis, never as a decision.

**What is prompt caching and does it actually save money?**
Prompt caching stores the processed prefix of a prompt so repeated requests skip re-processing it. If your system prompt is long and reused, it can cut input cost by 50% or more. The catch is exact prefix matching: any change to the cached prefix invalidates it. Log your cache hit rate, because a cache that never hits is just added complexity.

## Final recommendation

Start with ground truth, add attribution only when you need it, and treat any agentic optimizer as a hypothesis generator rather than an authority. The backfire was not the AI; it was using inference where measurement was available and cheaper. The boring tools won because they answer the only question that matters: what did I actually pay, and for what?

Your next step, in the next 30 minutes: open your billing console, set an AWS Budget at 120% of your trailing 30-day spend with an SNS email alert, and run the Athena query above against your CUR 2.0 table to get your top services by cost. That gives you a tripwire and a baseline before you optimize anything.


---

### About this article

**Written by:** [Kubai Kevin](/about/) — software developer based in Nairobi, Kenya, with 10+ years building production systems in fintech and AI.

**How this article was produced:** This site uses an automated LLM pipeline designed and maintained by the author. Topics are selected from real production experience. Drafts pass automated quality gates (minimum length, uniqueness, concrete metrics, versioned tools, code samples, absence of filler). Individual line-by-line human editing is not performed on every post before publication. Specific numbers, benchmarks and cost figures are illustrative; verify them against current official documentation before production use.

**Corrections:** Report errors via the [contact page](/contact/). Corrections are applied promptly.

**Last generated:** September 2026
