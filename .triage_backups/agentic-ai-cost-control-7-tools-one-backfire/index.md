# Agentic AI cost control: 7 tools, one backfire

LLM spend grows on its own. A single agent that retries a failed tool call, re-reads a 40k-token context, and fans out to three sub-agents can multiply the cost of one logical request without anyone changing a line of code. For a small team, that is the difference between a viable margin and a slow bleed.

The obvious move is to point AI at the problem: use an LLM to watch your LLM bill. That works up to a point, and then it backfires in a specific, repeatable way. The backfire is not "the AI got it wrong." It is that the cost of the observer scales with the cost of the thing it observes, and the observer has no ground truth unless you give it one.

A common failure mode: an agent is wired to a billing API, it summarizes spend, and it confidently attributes a spike to the wrong service because the tags are inconsistent. The team then optimizes the wrong thing for a week. The part that trips people up is separating measurement from inference, and that is what this article covers.

## The evaluation criteria

Five criteria, weighted for a team of one or two people who are both the decision-maker and the implementer.

1. **Ground truth.** Does the tool read actual billing data, or does it infer from logs? Inference is fine for direction, dangerous for decisions. If a tool cannot point at a line item in the AWS Cost and Usage Report (CUR 2.0) or a provider usage record, treat its output as a hypothesis.
2. **Blast radius.** Can it change production behavior? An optimizer that rewrites prompts or flips model routing can break correctness. A dry-run mode and a diff are preferable to a live toggle.
3. **Observer cost.** What does the tool itself cost to run? An agent that spends $0.15 per analysis to save $0.10 is a net loss. A reasonable target is keeping the observer under 5% of the spend it manages.
4. **Time to first signal.** How long from install to a number you trust? Under 30 minutes is a workable bar for a solo developer.
5. **Reversibility.** Can it be removed in an afternoon? Anything that writes to a database or mutates prompts is a hard-to-reverse decision and deserves extra scrutiny.

Score each option on a 1–5 scale and weight ground truth and blast radius highest. The table below is a summary; the sections after it explain each row.

| Tool / approach | Ground truth | Blast radius | Observer cost | Time to signal | Reversible |
|---|---|---|---|---|---|
| AWS Cost Explorer + CUR 2.0 queries | 5 | 1 | ~0 | 15 min | Yes |
| OpenAI Usage API + Python script | 5 | 1 | ~0 | 20 min | Yes |
| LLM proxy with request logging (self-hosted) | 4 | 2 | Low | 30 min | Yes |
| LLM tracing platform (self-hosted) | 4 | 2 | Low | 45 min | Yes |
| LLM gateway with budget enforcement | 3 | 4 | Low | 30 min | Mostly |
| Agentic optimizer (LLM loop over billing) | 2 | 3 | High | 2–3 hrs | Yes |
| Prompt/model routing agent | 2 | 5 | Medium | 1 hr | No (needs rollback) |

The two rows that matter most are the last two. They are where the backfire lives.

## 1. AWS Cost Explorer + CUR 2.0 queries

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

## 2. OpenAI Usage API + a Python script

**What it does:** Pulls per-model token counts and costs from the Usage API, aggregates them, and writes a daily summary.

**Strength:** Per-model and per-day granularity for free, no agent required. A short script gets you a daily number.

**Weakness:** It is per-organization, not per-feature. If two products share one API key, you cannot attribute spend without adding your own logging.

**Best for:** Solo developers with one product and one key. This is the boring, proven option and it is usually enough.

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

## 3. LLM proxy with request logging (self-hosted)

**What it does:** A proxy that sits in front of your LLM calls and logs latency, tokens, cost, and cache hits per request.

**Strength:** Per-request attribution without changing your app logic much — you change the base URL. Cache hit rates become visible, and prompt caching can cut input cost substantially on repeated system prompts.

**Weakness:** Self-hosting adds a service to maintain. The proxy is a single point of failure: if it goes down, your LLM calls fail unless you have a fallback path.

**Best for:** Teams that already suspect caching is leaving money on the table.

## 4. LLM tracing platform (self-hosted)

**What it does:** Tracing and cost analytics for LLM apps, with a per-trace view of tokens and cost.

**Strength:** You can see the exact trace that cost $0.40 and why. That is the attribution CUR cannot give you.

**Weakness:** It is a tracing tool first, a cost tool second. Cost dashboards are often less mature than the trace view, and self-hosting on a small Postgres instance gets slow past a few million spans.

**Best for:** Anyone debugging *why* a specific request is expensive, not just *that* it is.

## 5. LLM gateway with budget enforcement

**What it does:** A gateway that normalizes many model APIs behind one interface and enforces per-key and per-team budgets.

**Strength:** Hard budget caps. When a runaway loop hits the cap, calls start failing instead of billing. That is a real circuit breaker.

**Weakness:** It is another hop in the request path, and budget enforcement is coarse — per key, not per feature. Misconfigure it and you take down production traffic at 2am.

**Best for:** Teams with multiple products or customers sharing keys who need a spend ceiling.

## 6. Agentic optimizer (an LLM loop over billing data)

**What it does:** An agent reads your billing export, reasons about anomalies, and proposes optimizations.

**Strength:** It can surface patterns you would not query for, like a slow drift in average output tokens per request.

**Weakness:** This is where the backfire lives. The observer has no ground truth: it reads the same lagging, aggregated CUR data you do, so it cannot catch what CUR cannot see. On a noisy week it will produce confident but wrong attributions — for example, blaming a summarization job for a spike that actually came from a retry loop in a queue consumer. And the observer is not free: each analysis run consumes tokens, so a tool that costs a meaningful fraction of the spend it manages is a net loss.

**Best for:** Nobody, as a primary tool. Useful as a second opinion after you have ground-truth data.

## 7. Prompt/model routing agent

**What it does:** An agent that inspects each request and routes it to a cheaper model when it judges the task simple.

**Strength:** Routing trivial classification calls from a frontier model to a small model can cut that call's cost dramatically.

**Weakness:** Highest blast radius of anything here. A bad routing decision silently degrades output quality, and you will not notice until a customer does. It also needs a rollback path you have probably not built.

**Best for:** Teams with an eval harness that catches quality regressions automatically. Without evals, do not ship this.

## The top pick and why it wins

The boring answer wins: **CUR 2.0 queries for the bill, plus the OpenAI Usage API script for attribution, plus a request-logging proxy for per-request detail.** That combination scores highest on ground truth and lowest on blast radius, and the observer cost is effectively zero.

The reasoning is simple. Cost optimization is a measurement problem before it is an inference problem. If you cannot attribute spend to a feature, any optimization you make is a guess. CUR gives you the truth about the total. The Usage API gives you per-model truth. The proxy gives you per-request truth. Only after those three agree do you have a stable baseline to optimize against.

A worked example of why order matters. Suppose a typical agent stack shows a 30% month-over-month increase in LLM spend. The instinct is to route to a cheaper model. But the ground-truth data shows the increase is entirely in input tokens, not output, and it correlates with a context window that grew from 8k to 40k tokens because a retrieval step started appending full documents.

Do the arithmetic on a single call. At an illustrative input price of $3 per million tokens, an 8k-token prompt costs 8,000 ÷ 1,000,000 × $3 = $0.024. A 40k-token prompt costs 40,000 ÷ 1,000,000 × $3 = $0.12. That is a 5x increase in input cost per call from a change nobody flagged as a cost change. Multiply by request volume and the "spike" is fully explained. The fix is trimming context, not changing models, and it saves more than routing would. The agentic optimizer missed this because it was reasoning over lagging, aggregated data. The boring tools caught it in one query.

That is the bar: the observer should cost a small fraction of the spend it manages, and it should be able to point at a line item.

## Honorable mentions worth knowing about

**Cloud budget alerts (for example, AWS Budgets with SNS).** Not a cost optimizer, but a tripwire. Set a budget at 120% of your trailing 30-day spend and get an email before the bill surprises you. Takes 10 minutes. Reversible. Do it even if you do nothing else.

**Prompt caching.** Not a tool, a technique, but it belongs here. If your system prompt is long and you send it on every request, caching it can cut that input cost substantially. Both OpenAI and Anthropic document prompt caching. The catch: cache hits depend on exact prefix matching, so a single changed character invalidates the cache. Log your hit rate or you will not know it is working.

**Cost dashboards inside an existing tracing deployment.** Worth a look if you already run a tracing platform. Adding cost tracking to an existing deployment is cheap; standing up a tracing platform just for cost is not.

**Gateway budget caps.** If you have ever had a runaway loop, a hard cap is worth the extra hop. Set it above normal traffic and below "oh no."

## The approaches that get dropped, and why

**The agentic optimizer.** Dropped as a primary tool. The failure mode is specific and worth naming: the agent has no way to distinguish a real anomaly from a tagging inconsistency, and it does not know what it does not know. It can produce a confident recommendation that costs more than it saves. Keep it, if at all, as an occasional second opinion run manually, with its output treated as a hypothesis to verify against CUR.

**The routing agent.** Dropped entirely, for now. Without an eval harness, quality regressions cannot be measured, and shipping a quality regression to save money on classification calls is a bad trade when the classification calls are 5% of spend. The math changes if routing targets your expensive calls, but then the blast radius is worse.

**A third-party cost dashboard SaaS.** Dropped because it wanted read access to the billing account and the LLM keys, and its attribution was still an estimate. The ground-truth tools were better and free. This is a general pattern: if a tool cannot point at a line item, it is a hypothesis generator, and you already have one of those.

## How to choose based on your situation

**One product, one API key, under $200/month in LLM spend.** Use the Usage API script and cloud budget alerts. Skip everything else. The overhead is not worth it.

**One product, multiple features, $200–$2,000/month.** Add a request-logging proxy or tracing platform for per-request attribution. This is where the money is actually hiding, usually in context bloat and retry loops.

**Multiple products or customers sharing keys.** Add a gateway with budget caps. You need a ceiling more than you need attribution.

**Anything with an eval harness.** You can consider routing. Without evals, do not.

**Anyone at any scale.** Do not lead with an agentic optimizer. Lead with ground truth. The backfire is not that the agent is dumb; it is that the agent is confident about data it cannot see.

## How to measure this yourself

If you want numbers instead of a table, instrument these four things and compare them week over week.

1. **Total spend from the bill.** Query CUR 2.0 for unblended cost grouped by service and by your cost-allocation tags. This is the denominator for everything else.
2. **Input tokens vs. output tokens per feature.** Log token counts at the call site, tagged with a feature ID. A spike in input tokens almost always means context bloat; a spike in output tokens means generation is running longer.
3. **Retries and fan-out per logical request.** Count tool-call retries and sub-agent invocations. A retry that re-sends a 40k-token context is a full re-bill, not a rounding error.
4. **Cache hit rate.** If you use prompt caching, log hits and misses. A cache that never hits is added complexity with no payoff.

The comparison that matters: does the sum of your per-feature token logs reconcile with the provider's usage record, and does that reconcile with the CUR line item? If the three disagree, fix the measurement before you optimize anything.

## Frequently asked questions

**How do I track LLM API costs per feature?**
You have to add your own logging, because provider APIs are per-key, not per-feature. The standard approach is a proxy or tracing layer that tags each request with a feature or user ID, then aggregates by tag. A lighter option is to log token counts yourself at the call site and join them to your own database. Either way, the provider will not do this for you.

**Why did my AI costs spike overnight?**
The most common causes, in order: a retry loop that re-sends a large context, a context window that grew because a retrieval step started appending full documents, and a fan-out pattern where one request triggers many sub-calls. Check input token counts first — a spike in input tokens almost always means context bloat, while a spike in output tokens means generation is running longer. CUR data lags, so check your proxy logs for the same window.

**Is it worth using an AI agent to optimize AI spend?**
As a primary tool, no. The observer cost is real, the ground truth is missing, and confident wrong attributions cost more than they save. As a second opinion after you have ground-truth data, it can surface patterns you did not query for. Treat its output as a hypothesis, never as a decision.

**What is prompt caching and does it actually save money?**
Prompt caching stores the processed prefix of a prompt so repeated requests skip re-processing it. If your system prompt is long and reused, it can cut input cost substantially. The catch is exact prefix matching: any change to the cached prefix invalidates it. Log your cache hit rate, because a cache that never hits is just added complexity.

## Final recommendation

Start with ground truth, add attribution only when you need it, and treat any agentic optimizer as a hypothesis generator rather than an authority. The backfire is not the AI; it is using inference where measurement was available and cheaper. The boring tools win because they answer the only question that matters: what did you actually pay, and for what?

Your next step, in the next 30 minutes: open your billing console, set a budget at 120% of your trailing 30-day spend with an email alert, and run the Athena query above against your CUR 2.0 table to get your top services by cost. That gives you a tripwire and a baseline before you optimize anything.
