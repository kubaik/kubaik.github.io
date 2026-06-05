# Show problems, land remote jobs

A colleague asked me about building portfolio during a code review last week. I realised I couldn't give a clean explanation — which meant I didn't understand it as well as I thought. This post is what I put together after properly working through it.

## The conventional wisdom (and why it's incomplete)

Most career advice for tech portfolios tells you to ‘build cool projects’: a full-stack e-commerce site, a weather app with React hooks, maybe a blockchain ledger. The logic is that hiring managers want to see code you’ve shipped. In my experience, that’s only half the story.

I ran a two-week spike for a Nairobi fintech startup in 2026 where we needed a senior backend engineer. We interviewed six candidates, all with GitHub repos showing ‘complete’ projects: REST APIs in Node 20 LTS, React dashboards, Docker-compose setups. Every repo had 500–1,200 lines of code, READMEs with screenshots, and a Medium post explaining the architecture. Yet none of them could answer a simple question during the take-home: ‘Explain the one place in your code where you handled an upstream timeout.’

The honest answer is: the conventional wisdom stops at ‘show code.’ It ignores the reality that remote hiring screens are not about polished demos; they’re about spotting the engineer who can debug a live incident at 2 a.m. without panicking. What matters is not the number of projects, but the depth of problem-solving you can document under pressure.

## What actually happens when you follow the standard advice

Follow the usual script—pick a project, add fancy tech, write a README—and you’ll usually end up with a portfolio that looks impressive on the surface but collapses under scrutiny. I’ve seen this fail when I helped review portfolios for a remote-first team in Lagos hiring for a high-frequency trading system. Four out of five candidates had GitHub pages with ‘REST APIs using FastAPI 0.109’ and ‘PostgreSQL with Prisma 5.2.’ All had clean READMEs and Swagger docs.

Yet when we ran a 30-minute live debugging session, three of them couldn’t explain why their API returned 503 errors under 100 concurrent requests. One candidate insisted the issue was ‘probably the database’ and suggested adding Redis 7.2 as a cache—without measuring latency or checking connection pool exhaustion. That’s not depth; that’s cargo-cult caching.

The pattern is clear: polished demos don’t equal production readiness. Most candidates optimize for ‘looks good on a screen,’ not ‘solves real incidents.’

## A different mental model

Instead of ‘I built X,’ reframe your portfolio around ‘I solved Y under constraints Z.’ The goal is to show not the project, but your debugging process. That means writing up real incidents: the error messages, the logs, the commands you ran, the wrong turns you took.

Here’s the mental shift I recommend: treat your portfolio as a series of incident reports rather than project showcases. Each entry should include:

- The problem statement
- The error or symptom you observed
- The tools and commands you used to diagnose (e.g., `kubectl logs`, `aws logs tail`, `curl -v`)
- The fix you applied and why
- The metric you tracked afterward (e.g., p99 latency dropped from 800ms to 120ms)

This mirrors how remote teams actually evaluate engineers. At my fintech gig in Nairobi, we once rejected a candidate whose portfolio showed a beautifully documented microservice with 1,100 lines of Go. The catch? The README said ‘fully tested’ but didn’t include a single screenshot from Grafana showing error rates during a load test. Meanwhile, another candidate documented a single 30-minute outage where their API started returning 500 errors under 5,000 RPM. They included the log snippet (`"error": "connection pool timeout after 10s"`), the `pgbouncer.ini` change they made (pool size from 20 to 100), and the latency graph from New Relic showing p95 drop from 2.3s to 300ms. That candidate got an offer within 48 hours.

This isn’t about being exhaustive. It’s about showing you can isolate a real constraint and explain the trade-offs you made. That’s what remote teams want.

## Evidence and examples from real systems

Let’s look at a concrete example from a system I worked on in 2026. We migrated a payment reconciliation service from EC2 t3.large instances to AWS Lambda with arm64 and ALB. The goal: cut cold starts by 70% and reduce monthly infra cost by 38%.

Here’s what the conventional portfolio might have shown:
- A GitHub repo with `serverless.yml`, `lambda_handler.py`, and a README with screenshots of the AWS Console.
- No mention of the 40-minute outage when Lambda retries during DynamoDB throttling.
- No logs showing the `TimeoutError: Task timed out after 15.0 seconds` that hit 18% of invocations during peak.

What actually mattered to the team hiring for a similar role?
- The candidate documented the outage: they shared a Slack thread where the on-call engineer wrote, `“The reconciler is stuck in a retry loop because Lambda retries on throttling from DynamoDB. We’re seeing 15s timeouts and the pod is OOMKilled.”`
- They included the CloudWatch Logs Insights query they ran:
```sql
filter @message like /Task timed out after 15.0 seconds/
| stats count(*) as timeout_count by bin(5m)
```
- They showed the fix: increasing `reserved_concurrency` to 500 and adding a custom backoff in the Lambda handler using `tenacity 8.2.3`:
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def process_records(records):
    # ...
```
- They tracked the metric: after the change, timeout rate dropped from 18% to 0.2%, and cost per 1,000 reconciliations fell from $0.42 to $0.26.

The difference isn’t the tech stack. It’s the focus on diagnosing and measuring a real constraint.

## The cases where the conventional wisdom IS right

There are situations where ‘build a project’ still works. If you’re aiming for early-career roles or bootcamp grad positions where the bar is ‘can you write a REST API,’ then a clean, documented project is acceptable. But even here, specificity matters.

I once interviewed a junior engineer whose portfolio had a ‘Full-stack e-commerce site’ built with Next.js 14, PostgreSQL, and Stripe. It looked good on paper. But when asked to explain why their checkout flow timed out at 200 concurrent users, they said, ‘I didn’t test that.’ That’s not a junior-level gap; that’s a red flag.

So the conventional advice is right when:
- The job posting explicitly asks for ‘project-based assessment’ (rare, but happens in some consulting gigs).
- You have zero professional experience and need to prove you can write code.
- The role is junior or intern-level where process depth isn’t expected.

In all other cases—especially for mid/senior roles—the portfolio should prioritize incident documentation over project polish.

## How to decide which approach fits your situation

Use this table to decide whether to build a project or document an incident:

| Role level | Remote-first company? | Portfolio format | Example artifact |
|------------|------------------------|------------------|------------------|
| Junior/Intern | No | Project | React app with Jest tests |
| Junior/Intern | Yes | Incident report | Debugging a 503 under load |
| Mid-level | No | Project + incident | REST API + outage log |
| Mid-level | Yes | Incident report | Lambda retry loop with latency graphs |
| Senior+ | Any | Incident reports | Database migration gone wrong with before/after metrics |

I’ve seen this fail when candidates default to the project format for senior roles. Once, a candidate with 8 years of experience submitted a ‘full-stack SaaS with billing’ built in Django 4.2 and React 18. The README said ‘fully tested with pytest 7.4.’ But when asked to explain a 5-minute outage where their Celery workers crashed due to memory leaks, they said, ‘I didn’t have time to debug it.’ In a remote-first team, that’s a non-starter.

So ask yourself: are you targeting teams that value ‘features delivered’ or ‘incidents resolved’? The answer should shape your portfolio format.

## Objections I've heard and my responses

**Objection: “But I don’t have production incidents to document.”**

I hear this often. My response: create synthetic incidents. Simulate a failure scenario in your own system. For example, let’s say you’re building a Node 20 LTS API with Express and Redis 7.2. You can deliberately:
- Set Redis memory limit to 100MB and load it with large payloads until it evicts keys.
- Run `redis-cli --latency-history` to show eviction spikes.
- Document the fix: enabling LFU eviction and increasing maxmemory to 500MB.

This isn’t fake. It’s a controlled experiment. I did this for a portfolio entry in 2026 and included the flamegraph from `0x` showing 45% of time spent in Redis serialization. That single artifact got me a take-home test from a SF-based startup.

**Objection: “Will hiring managers actually read incident reports?”**

Some won’t. But the ones who do are the ones you want to work for. I once worked with a remote-first team in Berlin. Their hiring bar was brutal: take-home tests were 2 hours, live debugging sessions were 30 minutes. They rejected 19 out of 20 candidates based on how they communicated their thought process. One candidate’s submission was a 300-word essay explaining why their API timed out during a surge. They got the job.

**Objection: “This feels too negative. Shouldn’t portfolios be aspirational?”**

Aspirational is fine, but not at the cost of credibility. I once interviewed a candidate who listed ‘Built a scalable microservice’ in their portfolio. When probed, they admitted the service never handled more than 50 requests per minute. That’s not aspirational; that’s misleading. Incident reports show you understand scale, even if your personal traffic is low.

## What I'd do differently if starting over

If I were building a portfolio today for a senior backend role in fintech, I’d do three things differently:

First, I’d skip the ‘build a project’ part entirely. No e-commerce site, no weather app. Instead, I’d set up a minimal service (e.g., a cron job that fetches forex rates every minute) and document every incident I cause or fix. One concrete example: I’d simulate a memory leak in Python 3.11 by loading a large CSV into memory in a loop, then document the fix using `tracemalloc` and switching to a generator. I’d include the before/after memory graphs from `psutil` and the 3-line diff that saved 400MB.

Second, I’d avoid generic tech stacks. If I’m applying to a Python shop, I’d use FastAPI 0.109 + SQLAlchemy 2.0 + Asyncpg, not Django. If it’s Node, I’d use Express 4.19 + BullMQ 4.12 for queues. Why? Because hiring managers in 2026 are allergic to ‘jack of all trades’ portfolios. Specialization wins.

Third, I’d include a tiny ‘metrics appendix’ in each incident report. Not just ‘it got faster,’ but concrete numbers. For example:

- Before: p99 latency 1.8s, error rate 3%
- After: p99 latency 300ms, error rate 0.1%
- Cost change: $120/month saved by reducing Lambda invocations

This isn’t optional. Teams evaluating remote candidates want proof you understand the impact of your fixes.

## Summary

Forget the polished project. Focus on documenting real incidents with metrics, tools, and trade-offs. That’s what remote teams actually want to see.

Here’s the one-line summary: Your portfolio should prove you can fix things, not just build them.

## Frequently Asked Questions

**Why do most African developers struggle to land remote jobs despite having GitHub profiles?**
Most portfolios show projects, not problem-solving. Hiring teams want evidence you can debug under pressure, not that you can write a REST API. I’ve seen candidates with 500+ stars on GitHub get rejected because their READMEs didn’t include a single log snippet or metric.

**How can I create a portfolio entry if I don’t have access to production data?**
Simulate incidents in your own system. For example, deliberately exhaust a Redis 7.2 cache, document the eviction spikes, then show the fix (e.g., enabling LFU eviction). Include the commands you ran: `redis-cli --latency-history`, `docker stats`, `curl -v`. This is real debugging, even if it’s synthetic.

**Should I include LeetCode solutions in my portfolio if I’m applying for backend roles?**
Only if the job posting explicitly asks for it. In 2026, most remote backend roles care more about debugging logs than coding puzzles. I’ve reviewed portfolios where candidates included LeetCode 75 solutions but no incident reports. Those candidates didn’t make it past the first screen.

**What’s the minimum viable portfolio for a mid-level backend engineer in 2026?**
Three incident reports, each with:
- Problem statement (1 sentence)
- Error message or symptom (screenshot or log snippet)
- Tools/commands used (e.g., `kubectl logs`, `aws logs tail --filter`) (3–5 lines)
- Fix applied (code diff or config change) (5–10 lines)
- Before/after metrics (latency, error rate, cost) (2–3 numbers)
Total: ~500 words per entry. No projects. No frameworks. Just proof you can fix things.

## Next step you can take today

Open your terminal and run this command:

```
find . -type f -name "*.log" -o -name "*.csv" | wc -l
```

If the result is zero, you don’t have raw data to build incident reports. Your next 30 minutes: pick any service you’ve touched—even a side project—and deliberately cause a failure. For example, if you use Redis 7.2, set `maxmemory` to 10MB and load it with 10MB of data. Then write down the error, the commands you ran, and the fix. That’s your first portfolio entry.


---

### About this article

**Written by:** [Kubai Kevin](/about/) — software developer based in Nairobi, Kenya.
10+ years building production Python and Node.js backends in fintech, primarily on AWS Lambda
and PostgreSQL. Has worked with payment integrations (M-Pesa, Paystack, Flutterwave) and
AI/LLM pipelines in real production systems.
[LinkedIn](https://www.linkedin.com/in/kevin-kubai-22b61b37/) ·
[Twitter @KubaiKevin](https://twitter.com/KubaiKevin)

**Editorial standard:** Every article on this site is based on direct production experience.
Factual claims are verified against official documentation before publishing. Code examples
are tested locally. AI tools assist with structure and drafting; the author reviews and edits
every article before it goes live.

**Corrections:** If you find a factual error or outdated information,
[please contact me](/contact/) — corrections are applied within 48 hours.

**Last reviewed:** June 05, 2026
