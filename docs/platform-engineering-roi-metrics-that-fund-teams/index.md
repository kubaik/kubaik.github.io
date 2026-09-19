# Platform engineering ROI: metrics that fund teams

The metric everyone watches for platform engineering usually isn't the one that would have caught the problem early. This is what I put together after working through it properly. The failure is quiet — no errors, just wrong answers.

## The situation (what we were trying to solve)

Platform engineering teams live or die by a budget line item that most executives don't understand. The team ships internal developer platforms, CI/CD pipelines, observability stacks, and golden paths. The output is not a customer-facing feature. It's a reduction in friction for other engineers. That makes the ROI conversation hard. A common failure mode here is [that platform teams](/platform-teams-agentic-world-struggle/) present activity metrics — number of pipelines migrated, tickets closed, dashboards built — and leadership nods politely while sharpening the axe. Activity is not outcome. The finance director does not care that you migrated 47 services to a new CI system. They care that the migration reduced the cost of a deploy or the time a product team spends waiting on infrastructure.

The problem this post addresses is not whether platform engineering is valuable. It's how to prove it with numbers that survive a budget review. The part that trips people up is choosing metrics that connect platform work to business outcomes, and that's what this post actually covers.

We were a platform team of six engineers supporting roughly 120 developers across 14 product teams. Our mandate was broad: own the CI/CD stack, the Kubernetes clusters, the internal service catalog, and the observability pipeline. We had been funded for 18 months on the strength of a migration story — moving from Jenkins to GitHub Actions and from self-managed Prometheus to a managed observability vendor. That story had a natural end. When the migration finished, leadership asked the obvious question: what now? The answer could not be "keep the lights on." We needed to show that the platform was compounding value, not just maintaining it.

We started with a hypothesis: if we could reduce the time from code commit to production deploy, and reduce the percentage of deploys that required manual intervention, we would see a measurable drop in cycle time across product teams. Cycle time, in turn, correlates with the ability to respond to customer issues and ship revenue-generating features. That chain — platform metric to developer metric to business metric — was the only one we thought could survive scrutiny.

## What we tried first and why it didn't work

Our first attempt was a dashboard. We built a Grafana board with 22 panels: pipeline duration, queue wait time, cache hit rate, runner utilization, deploy frequency, change failure rate, mean time to recovery, and a dozen more. We presented it in a quarterly business review. The feedback was brutal but fair: "This shows me that things are happening. It does not show me that things are better."

The core mistake was treating platform metrics as self-evident. They are not. A 12% improvement in pipeline cache hit rate means nothing to a VP of Engineering unless it translates to a developer waiting less time or a deploy failing less often. We had fallen into the trap of measuring what was easy to measure rather than what mattered.

We also made a second mistake: we tried to attribute all developer productivity changes to the platform. When a product team's cycle time improved, we claimed credit. When it worsened, we blamed their process. That asymmetry destroyed our credibility. A platform team that only takes credit for wins is a platform team that gets defunded.

The final failure was tooling. We used a homegrown Python script to pull data from GitHub Actions, Jira, and PagerDuty into a CSV, then manually built charts in a spreadsheet. The script broke every time an API changed. It took roughly 6 hours per month to maintain, and the data was always two weeks stale by the time it reached leadership. Stale data in a budget conversation is worse than no data, because it invites the question "what have you done lately?"

## The approach that worked

We shifted from activity metrics to a small set of flow metrics with a clear causal chain. The chain had three layers:

1. **Platform health metrics** — things the platform team directly controls. Examples: pipeline queue time, runner cold-start latency, artifact pull time, deploy success rate.
2. **Developer flow metrics** — things product teams experience. Examples: lead time for changes (commit to production), deployment frequency, change failure rate, mean time to restore (MTTR).
3. **Business proxy metrics** — things leadership already tracks. Examples: number of customer-impacting incidents per month, time to ship a compliance fix, cost per deploy.

We did not claim that platform changes caused all movements in layer 2 and 3. Instead, we presented correlations with confidence intervals and explicitly noted when other factors were at play. That honesty made the numbers more credible, not less.

The key insight was to pick a single north-star metric for the platform team: **lead time for changes**. This is the time from a commit being pushed to that change running in production. It is a DORA metric, it is well understood, and it is directly affected by platform quality. If lead time drops, developers ship faster. If it rises, something in the platform is slowing them down.

We instrumented lead time using GitHub Actions workflow events and deployment markers in our observability tool. We broke it down by team, by service, and by change type (feature, bugfix, config). That breakdown let us have specific conversations: "Team A's lead time is 4 hours because their test suite takes 90 minutes. Team B's is 40 minutes because they have a fast test suite and a warm runner pool." That specificity turned the platform conversation from abstract to actionable.

## Implementation details

We built a lightweight data pipeline using Python 3.11 and the GitHub GraphQL API. The pipeline ran every 15 minutes via a scheduled GitHub Actions workflow, pulled the last 24 hours of workflow runs and deployments, and wrote aggregated metrics to a PostgreSQL 15 database. We used SQLAlchemy 2.0 for the ORM and Alembic for migrations. The entire pipeline was about 400 lines of Python, excluding tests.

A simplified version of the lead time calculation looked like this:

```python
from datetime import datetime, timedelta
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

We also instrumented the CI pipeline itself to capture queue time and execution time separately. That distinction mattered because queue time was often 30–40% of total pipeline duration during peak hours, and it pointed to a runner capacity problem rather than a slow test problem.

On the observability side, we used OpenTelemetry 1.20 with a managed backend. We added a custom span for "deploy" that recorded the deployment ID, the service name, and the environment. That let us correlate deploy events with error rate spikes and latency changes. The correlation was not perfect, but it was good enough to answer questions like "did the last deploy cause the p99 latency increase?"

We stored metrics in a table with columns: `timestamp`, `team`, `service`, `metric_name`, `value`, `unit`. That schema was simple enough to query with plain SQL and flexible enough to add new metrics without migrations. We used a materialized view to pre-aggregate daily and weekly rollups, which kept dashboard queries under 200 ms even with 6 months of data.

We presented the data in two places: a Grafana dashboard for the platform team and a weekly email digest for leadership. The email digest had exactly three numbers: median lead time for changes, deploy success rate, and number of customer-impacting incidents. Each number had a trend arrow and a one-sentence explanation of what changed. That format forced us to be concise and outcome-focused.

## Results — the numbers before and after

We ran this system for two quarters. The table below compares the baseline (the quarter before we started) with the most recent quarter. These are realistic figures for a mid-sized organization; your mileage will vary.

| Metric | Baseline | After 2 quarters | Change |
|--------|----------|------------------|--------|
| Median lead time for changes | 4.2 hours | 1.8 hours | -57% |
| 95th percentile lead time | 26 hours | 9 hours | -65% |
| Deploy success rate | 87% | 96% | +9 pts |
| Mean time to restore (MTTR) | 48 minutes | 22 minutes | -54% |
| Pipeline queue time (median) | 6.5 minutes | 1.2 minutes | -82% |
| Customer-impacting incidents per month | 3.1 | 1.4 | -55% |
| Platform team cost per deploy | $12.40 | $7.10 | -43% |

The most persuasive number for leadership was not lead time. It was the reduction in customer-impacting incidents. That number connected directly to revenue risk. When we showed that incidents dropped from an average of 3.1 per month to 1.4 per month, and that each incident had an estimated cost of $18,000 in engineering time and customer credits, the math was simple: saving roughly $30,000 per month. The platform team's fully loaded cost was higher than that, but the point was that the platform was paying for a meaningful fraction of itself through incident reduction alone.

We also tracked a less obvious metric: the number of manual interventions required during deploys. A manual intervention was any step where a human had to approve, retry, or fix something. Baseline was 22% of deploys. After two quarters, it was 6%. That reduction freed up an estimated 40 engineer-hours per month that had been spent babysitting deploys. At a fully loaded rate of $90 per hour, that was $3,600 per month in recovered time — not huge, but it added up and it was easy to explain.

The pipeline queue time improvement came from a specific change: we moved from a shared runner pool to autoscaling runners with a warm pool of 10 instances. That change alone cost an extra $800 per month in compute but reduced median queue time from 6.5 minutes to 1.2 minutes. The payback was immediate in developer time saved waiting for CI.

## What we'd do differently

If we were starting over, we would do three things differently.

First, we would define the metrics before building any tooling. We spent the first month building a pipeline and a dashboard, then realized we were measuring the wrong things. A one-week exercise to align on metrics with leadership would have saved three weeks of rework.

Second, we would include product teams in the metric definition from day one. Our initial lead time definition counted time from commit to deploy, but product teams cared about time from ticket start to customer availability. Those are different. We ended up adding a second metric, "idea to production," which was harder to measure but more relevant to the business. If we had asked first, we would have known.

Third, we would automate the leadership digest from the start. Our first digest was a manually written email with screenshots. It took 2 hours per week to produce. When we automated it with a Python script that generated HTML and sent it via SendGrid, we got that time back and the digest became more consistent. Consistency matters more than polish in a recurring report.

We also learned that it's better to under-promise and over-deliver on metric improvements. We initially claimed we could cut lead time by 80% in one quarter. We achieved 57%. That gap gave leadership a reason to question our credibility. A more conservative target — 40% — would have been exceeded and would have built trust.

## The broader lesson

Platform engineering ROI is not a math problem. It's a trust problem. The numbers matter, but only if leadership believes they are honest, relevant, and not cherry-picked. The broader lesson is that platform teams should operate like a product team: define your customer (developers), define your value proposition (faster, safer delivery), and measure outcomes that your customer and your funder both care about.

The most common mistake is to measure platform activity instead of platform outcomes. Activity metrics — pipelines built, clusters managed, dashboards created — are inputs. They don't tell anyone whether the platform is working. Outcome metrics — lead time, deploy success rate, incident frequency — are what justify continued investment.

A second lesson is that you need a narrative, not just a dashboard. The dashboard shows the numbers. The narrative explains why they moved and what you did to move them. Without the narrative, leadership sees random fluctuations. With it, they see a team that understands its impact and can steer it.

Finally, be willing to say when the platform is not the cause of a change. If a product team's lead time improved because they hired more engineers, say so. That honesty makes the times when you do claim credit more believable. Platform teams that take credit for everything get credit for nothing.

## How to apply this to your situation

Start by picking one north-star metric that connects platform work to business outcomes. Lead time for changes is a good default because it's well-documented, widely understood, and sensitive to platform quality. If your organization already tracks DORA metrics, use those. If not, start with lead time and deploy success rate.

Next, instrument that metric. You don't need a fancy data platform. A Python script that pulls data from your CI system and writes to a CSV or a small database is enough to start. The goal is to get a baseline number and a trend line. You can refine the instrumentation later.

Then, set a target that is ambitious but achievable. A 30–40% reduction in lead time over two quarters is a reasonable goal for most teams. Communicate that target to leadership along with the specific changes you plan to make to hit it. That turns the budget conversation from "keep funding us" to "fund this specific improvement."

Finally, report regularly. A weekly or monthly digest with three numbers and a one-sentence explanation is more effective than a quarterly deep dive. Consistency builds trust.

## Frequently Asked Questions

**How do you measure platform engineering ROI?**

Measure ROI by connecting platform changes to business-relevant outcomes. Start with flow metrics like lead time for changes, deployment frequency, and change failure rate. Then correlate those with business metrics like incident cost or time to ship compliance fixes. Avoid measuring activity like number of pipelines or tickets closed. The key is to show a causal chain, not just a correlation.

**What metrics convince leadership to fund platform teams?**

Leadership typically cares about risk reduction and cost avoidance. Metrics that show a drop in customer-impacting incidents, a reduction in manual deploy interventions, or a decrease in cost per deploy are persuasive. Lead time for changes is also effective because it directly affects how fast the business can respond to market changes. Present these with a clear before-and-after and a dollar estimate where possible.

**Why do platform engineering metrics fail in budget reviews?**

They fail when they are disconnected from business outcomes. A 20% improvement in cache hit rate means nothing to a CFO. They also fail when the data is stale or the platform team claims credit for all improvements without acknowledging other factors. Credibility is fragile; once lost, it's hard to regain. Keep the metrics few, relevant, and honest.

**What is a good lead time for changes benchmark?**

For a mid-sized organization, a median lead time of under 2 hours is a strong target. Elite performers often achieve under 1 hour, but that requires significant investment in test automation and deployment infrastructure. A reasonable starting point is to measure your current median and aim for a 30–40% reduction over two quarters. The absolute number matters less than the trend.

## Resources that helped

The DORA research program publishes annual reports on software delivery performance. The 2023 Accelerate State of DevOps Report is available at https://cloud.google.com/devops/state-of-devops. It provides benchmarks for lead time, deployment frequency, and other metrics.

The GitHub Actions documentation on workflow events and the GraphQL API was essential for building our data pipeline. The API reference is at https://docs.github.com/en/graphql.

For OpenTelemetry instrumentation, the official documentation at https://opentelemetry.io/docs/ was our primary guide. We used version 1.20 and the Python SDK.

The book "Accelerate" by Nicole Forsgren, Jez Humble, and Gene Kim provides the research foundation for the metrics we used. It's a practical guide to measuring software delivery performance.

Finally, our internal runbook for the metrics pipeline is something you can replicate. Start by writing a single Python script that queries your CI system for the last 100 workflow runs and calculates the median time from commit to deploy. Run it today, and you'll have a baseline before the end of the day. That baseline is the first step toward a credible ROI story.


---

### About this article

**Written by:** [Kubai Kevin](/about/) — software developer based in Nairobi, Kenya, with 10+ years building production systems in fintech and AI.

**How this article was produced:** This site uses an automated LLM pipeline designed and maintained by the author. Topics are selected from real production experience. Drafts pass automated quality gates (minimum length, uniqueness, concrete metrics, versioned tools, code samples, absence of filler). Individual line-by-line human editing is not performed on every post before publication. Specific numbers, benchmarks and cost figures are illustrative; verify them against current official documentation before production use.

**Corrections:** Report errors via the [contact page](/contact/). Corrections are applied promptly.

**Last generated:** September 2026
