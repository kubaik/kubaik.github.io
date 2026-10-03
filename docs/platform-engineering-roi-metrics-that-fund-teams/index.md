# Platform engineering ROI: metrics that fund teams

## Why platform ROI conversations go wrong

Platform engineering teams live or die by a budget line item that most executives don't understand. The team ships internal developer platforms, CI/CD pipelines, observability stacks, and golden paths. The output is not a customer-facing feature. It's a reduction in friction for other engineers. That makes the ROI conversation hard.

A common failure mode is that platform teams present activity metrics — number of pipelines migrated, tickets closed, dashboards built — and leadership nods politely while sharpening the axe. Activity is not outcome. A finance director does not care that 47 services were migrated to a new CI system. They care that the migration reduced the cost of a deploy or the time a product team spends waiting on infrastructure.

The problem this article addresses is not whether platform engineering is valuable. It's how to prove it with numbers that survive a budget review. The part that trips people up is choosing metrics that connect platform work to business outcomes.

## The trap of measuring what is easy

A typical first attempt is a dashboard. A team builds a Grafana board with 22 panels: pipeline duration, queue wait time, cache hit rate, runner utilization, deploy frequency, change failure rate, mean time to recovery, and a dozen more. They present it in a quarterly business review. The feedback is often brutal but fair: "This shows me that things are happening. It does not show me that things are better."

The core mistake is treating platform metrics as self-evident. They are not. A 12% improvement in pipeline cache hit rate means nothing to a VP of Engineering unless it translates to a developer waiting less time or a deploy failing less often. Teams fall into the trap of measuring what is easy to measure rather than what matters.

A second common mistake is attributing all developer productivity changes to the platform. When a product team's cycle time improves, the platform team claims credit. When it worsens, they blame the product team's process. That asymmetry destroys credibility. A platform team that only takes credit for wins is a platform team that gets defunded.

A third failure is tooling. Homegrown scripts that pull data from CI systems, issue trackers, and incident tools into a CSV, then manually build charts in a spreadsheet, break every time an API changes. They take hours per month to maintain, and the data is often two weeks stale by the time it reaches leadership. Stale data in a budget conversation is worse than no data, because it invites the question "what have you done lately?"

## A three-layer metric chain

The approach that tends to work is shifting from activity metrics to a small set of flow metrics with a clear causal chain. The chain has three layers:

1. **Platform health metrics** — things the platform team directly controls. Examples: pipeline queue time, runner cold-start latency, artifact pull time, deploy success rate.
2. **Developer flow metrics** — things product teams experience. Examples: lead time for changes (commit to production), deployment frequency, change failure rate, mean time to restore (MTTR).
3. **Business proxy metrics** — things leadership already tracks. Examples: number of customer-impacting incidents per month, time to ship a compliance fix, cost per deploy.

Do not claim that platform changes caused all movements in layers 2 and 3. Instead, present correlations with confidence intervals and explicitly note when other factors are at play. That honesty makes the numbers more credible, not less.

The key insight is to pick a single north-star metric for the platform team: **lead time for changes**. This is the time from a commit being pushed to that change running in production. It is a DORA metric, it is well understood, and it is directly affected by platform quality. If lead time drops, developers ship faster. If it rises, something in the platform is slowing them down.

Instrument lead time using CI workflow events and deployment markers in an observability tool. Break it down by team, by service, and by change type (feature, bugfix, config). That breakdown enables specific conversations: "Team A's lead time is 4 hours because their test suite takes 90 minutes. Team B's is 40 minutes because they have a fast test suite and a warm runner pool." That specificity turns the platform conversation from abstract to actionable.

## Instrumenting lead time

A lightweight data pipeline can be built with Python and the GitHub GraphQL API. It can run every 15 minutes via a scheduled CI workflow, pull the last 24 hours of workflow runs and deployments, and write aggregated metrics to a PostgreSQL database. SQLAlchemy handles the ORM and Alembic handles migrations. The entire pipeline can be a few hundred lines of Python, excluding tests.

A simplified version of the lead time calculation looks like this:

```python
from datetime import datetime
from github import Github

# Initialize with a token that has read access to actions and deployments
g = Github(access_token)
repo = g.get_repo("org/repo")

# Get the latest deployment to production
deployments = repo.get_deployments(environment="production")
latest = deployments[0]

# Find the commit that triggered it
commit_sha = latest.sha
commit = repo.get_commit(commit_sha)
commit_time = commit.commit.author.date

deploy_time = latest.created_at
lead_time = (deploy_time - commit_time).total_seconds() / 60  # minutes
print(f"Lead time for {commit_sha[:7]}: {lead_time:.1f} minutes")
```

Also instrument the CI pipeline itself to capture queue time and execution time separately. That distinction matters because queue time is often a large fraction of total pipeline duration during peak hours, and it points to a runner capacity problem rather than a slow test problem.

On the observability side, OpenTelemetry with a managed backend can be used. Add a custom span for "deploy" that records the deployment ID, the service name, and the environment. That allows correlation of deploy events with error rate spikes and latency changes. The correlation is not perfect, but it is good enough to answer questions like "did the last deploy cause the p99 latency increase?"

Store metrics in a table with columns: `timestamp`, `team`, `service`, `metric_name`, `value`, `unit`. That schema is simple enough to query with plain SQL and flexible enough to add new metrics without migrations.

Present the data in two places: a dashboard for the platform team and a weekly email digest for leadership. The email digest should have exactly three numbers: median lead time for changes, deploy success rate, and number of customer-impacting incidents. Each number gets a trend arrow and a one-sentence explanation of what changed. That format forces conciseness and outcome focus.

## How to measure your own baseline

The table below is not a benchmark. It is a measurement plan. For each metric, it tells you what to instrument and what to compare. Run the commands or queries against your own systems to produce your own numbers.

| Metric | What to instrument | What to compare |
|--------|-------------------|-----------------|
| Median lead time for changes | Timestamp of commit push vs. timestamp of production deploy marker | Current quarter vs. previous quarter, split by team and service |
| 95th percentile lead time | Same events, percentile calculation over all deploys in the period | Same period comparison; investigate outliers individually |
| Deploy success rate | CI workflow conclusion for deploy jobs (success/failure) | Rolling 30-day rate vs. prior 30-day rate |
| Mean time to restore (MTTR) | Incident start and resolution timestamps from your incident tool | Median and p95 across incidents in the period |
| Pipeline queue time | Time between job queued and job started, from CI API | Median and p95, segmented by runner pool |
| Manual interventions per deploy | Count of approval steps, retries, and manual fixes logged per deploy | Percentage of deploys requiring any intervention |
| Customer-impacting incidents | Incidents tagged as customer-facing in your incident tool | Count per month, with severity breakdown |

The point of this table is that every number is reproducible from data you already have. There is no magic benchmark to chase. The trend in your own organization is the only number that matters for a budget conversation.

## A worked example of the arithmetic

Suppose a platform team supports 100 developers. Suppose the median deploy wait time (queue plus pipeline) is 30 minutes, and each developer triggers 2 deploys per day. That is 100 × 2 × 30 minutes = 6,000 developer-minutes per day spent waiting, or 100 developer-hours per day. Over a 20-working-day month, that is 2,000 developer-hours per month.

Now suppose a runner capacity change cuts median wait to 10 minutes. The new figure is 100 × 2 × 10 = 2,000 developer-minutes per day, or 33.3 developer-hours per day. Over a month, that is 667 developer-hours. The difference is 2,000 − 667 = 1,333 developer-hours per month recovered.

If the fully loaded cost of a developer hour is $90 (an illustrative figure — substitute your own), the recovered time is worth 1,333 × $90 = $119,970 per month. If the runner capacity change costs an extra $800 per month in compute, the net is $119,170 per month. That is the arithmetic that belongs in a budget review, with every assumption stated so leadership can challenge it.

The same arithmetic applies to manual deploy interventions. If 22% of deploys require a human to approve, retry, or fix something, and there are 200 deploys per month, that is 44 interventions. If each intervention costs 30 minutes of engineer time, that is 22 engineer-hours per month. Reducing the intervention rate to 6% gives 12 interventions, or 6 engineer-hours — a recovery of 16 engineer-hours per month. At $90 per hour, that is $1,440 per month. Not huge, but easy to explain and easy to verify.

## Failure modes to watch for

**The stale data failure.** If the dashboard is two weeks behind, leadership will ask what has been done lately. Automate the pipeline and the digest so the numbers are always current. A cron job that fails silently is worse than no automation.

**The attribution failure.** If the platform team claims credit for every improvement, they will eventually be caught. When a product team's lead time improves because they hired more engineers, say so. That honesty makes the times when you do claim credit more believable.

**The vanity metric failure.** Cache hit rate, runner utilization, and dashboard count are inputs. They don't tell anyone whether the platform is working. If a metric cannot be connected to a developer or business outcome in one sentence, it does not belong in the leadership digest.

**The single-number failure.** A single metric can be gamed. Lead time can be reduced by deploying smaller, less tested changes. Pair lead time with change failure rate and MTTR so that speed cannot be bought with instability.

**The stale-baseline failure.** Comparing against a baseline from two years ago is meaningless if the organization has changed. Re-baseline every two quarters and state the period explicitly.

## A decision checklist

Before presenting platform metrics to leadership, verify each of the following:

- The metric connects to a business outcome in one sentence.
- The data is no more than 24 hours old.
- The baseline period is stated and recent.
- The attribution is honest — other factors are named.
- The metric is paired with a counter-metric (speed with stability, cost with quality).
- The trend, not just the absolute number, is shown.
- The specific platform change that moved the metric is named.
- A dollar estimate is provided with all assumptions stated.
- The target was set before the period, not after.
- The digest is consistent — same three numbers every week.

## Building the narrative

A dashboard shows the numbers. A narrative explains why they moved and what was done to move them. Without the narrative, leadership sees random fluctuations. With it, they see a team that understands its impact and can steer it.

The narrative should answer three questions: What changed? Why did it change? What is next? For example: "Median lead time dropped from 4 hours to 2 hours. The cause was moving from a shared runner pool to autoscaling runners with a warm pool, which cut queue time. Next quarter we are targeting test suite duration, which is now the dominant component."

That narrative turns the budget conversation from "keep funding us" to "fund this specific improvement." It also makes the platform team accountable for a specific change rather than a vague promise of value.

## Frequently asked questions

**How do you measure platform engineering ROI?**

Measure ROI by connecting platform changes to business-relevant outcomes. Start with flow metrics like lead time for changes, deployment frequency, and change failure rate. Then correlate those with business metrics like incident cost or time to ship compliance fixes. Avoid measuring activity like number of pipelines or tickets closed. The key is to show a causal chain, not just a correlation.

**What metrics convince leadership to fund platform teams?**

Leadership typically cares about risk reduction and cost avoidance. Metrics that show a drop in customer-impacting incidents, a reduction in manual deploy interventions, or a decrease in cost per deploy are persuasive. Lead time for changes is also effective because it directly affects how fast the business can respond to market changes. Present these with a clear before-and-after and a dollar estimate where possible.

**Why do platform engineering metrics fail in budget reviews?**

They fail when they are disconnected from business outcomes. A 20% improvement in cache hit rate means nothing to a CFO. They also fail when the data is stale or the platform team claims credit for all improvements without acknowledging other factors. Credibility is fragile; once lost, it's hard to regain. Keep the metrics few, relevant, and honest.

**What is a good lead time for changes benchmark?**

There is no universal benchmark. The DORA research program publishes distributions of lead time across performance bands, but the right target depends on your deployment model, your test suite, and your compliance requirements. A reasonable approach is to measure your current median and aim for a meaningful reduction over two quarters. The absolute number matters less than the trend and the stability of the counter-metrics.

## What to do in the next 30 minutes

Write a single Python script that queries your CI system for the last 100 workflow runs and calculates the median time from commit to deploy. Run it now. You will have a baseline before the end of the day. That baseline is the first step toward a credible ROI story — and it is the number you can defend in the next budget review.
===END===
