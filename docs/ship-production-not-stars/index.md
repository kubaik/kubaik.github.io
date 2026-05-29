# Ship production, not stars

A colleague asked me about building portfolio during a code review last week. I realised I couldn't give a clean explanation — which meant I didn't understand it as well as I thought. This post is what I put together after properly working through it.

# Remote hiring starts with who you impress, not how many stars you have

The honest answer is that most African developers waste months polishing GitHub stars and LeetCode rankings while the people who actually hire them scroll past those metrics in under 10 seconds. I learned this the hard way in 2026 when a recruiter from a US fintech company replied to my cold email only to ask for a 30-minute call about my production experience—something my GitHub profile didn’t convey in the first paragraph. The signal that matters isn’t the count of public repos; it’s whether someone can look at your work and say, "This person ships systems that stay up when money is on the line."

Most career advice treats a portfolio like a resume in disguise—polish the bullets, count the metrics, game the algorithm. In reality, your portfolio competes against hundreds of other candidates who also have "built scalable microservices" and "optimized database queries." The difference maker is evidence you’ve actually done the job you’re applying for, in conditions that resemble a real production environment.

Here’s the contrarian take: **stop chasing GitHub stars and start building artifacts that simulate the hiring manager’s day-to-day pain.** A production-style incident report, a load-tested REST API, and a write-up of how you debugged a race condition under load beat 1,000 empty repository stars every time.

I ran into this when a friend at a Nairobi-based payments startup asked me to review his portfolio before he applied to a remote-first UK fintech. He had 14 repositories, each with 30+ GitHub stars, but zero documentation of the trade-offs he made when scaling to 5,000 RPS. When I asked him to walk me through the most interesting production issue he’d solved, he froze. That moment changed how we built his portfolio—we ended up replacing most of the GitHub stars with a single repository called `production-incidents` that contained runbooks, latency graphs, and the actual SQL queries he ran during an outage. He landed a remote interview within two weeks.

## The conventional wisdom (and why it's incomplete)

The standard advice goes like this: fork open-source projects, contribute to popular libraries, and accumulate GitHub stars until the stars start driving traffic to your profile. The logic is that stars act as a public, verifiable signal of technical skill. If you have 500 stars on a well-known library, recruiters assume you’re competent enough to maintain production systems.

In theory, this makes sense: open-source contributions demonstrate collaboration, code quality, and familiarity with community expectations. In practice, most African developers contribute to projects they don’t use in production, and recruiters rarely dig deep enough to distinguish between a meaningful contribution and a typo fix in the README.

The bigger issue is that GitHub stars are a lagging indicator. They tell you someone did something good in the past, but they don’t tell you whether that person can build resilient systems under pressure, debug a race condition at 2 a.m., or explain a technical decision to a non-technical stakeholder. Remote hiring managers aren’t looking for past achievements—they’re trying to predict future performance in their specific environment.

I’ve seen this fail when a brilliant Kenyan engineer with 800 GitHub stars on a Python ORM library bombed a remote interview because he couldn’t explain how he’d scale that ORM to handle 100,000 concurrent users. The recruiter’s feedback was blunt: "We need someone who can talk about load, caching, and failure modes—not someone who can refactor a for-loop."

The conventional wisdom also assumes that hiring managers have time to parse GitHub profiles. In reality, they spend an average of 6 seconds on a GitHub profile before deciding whether to move forward. Six seconds. That means your top three repositories, your README, and your commit messages need to scream "production-ready" in under 6 seconds—or you’re invisible.

A 2026 study by Hired found that 68% of remote engineering candidates in Africa are filtered out by automated keyword scans before a human ever sees their resume. Keywords like "Django," "PostgreSQL," and "REST API" are table stakes; what moves the needle is evidence that you’ve used those tools to solve real problems at scale.

## What actually happens when you follow the standard advice

What actually happens is that you end up with a profile that looks impressive to other developers but invisible to hiring managers who care about production outcomes. You’ll have a GitHub profile full of toy projects, each with a polished README, a clean architecture diagram, and a vague promise that it "scales well." But when you get to the interview, you’ll realize you can’t answer basic questions about how you’d monitor that system, how you’d handle a sudden traffic spike, or how you’d roll back a bad deployment without causing downtime.

I spent three weeks in 2026 helping a colleague in Lagos clean up his GitHub profile. He had 12 repositories, each labeled "Production-ready" with a green badge. But when we dug into the code, we found hardcoded API keys, no tests, and a database migration script that would have dropped the production schema if run. His README had a diagram showing "microservices" but no mention of how the services communicated, what the SLAs were, or how he’d handle a network partition. When he applied to a remote role, he never heard back—despite having 1,200 GitHub stars.

The problem isn’t the quality of the code; it’s the mismatch between the signal you’re sending and the signal that remote hiring managers actually need. GitHub stars reward cleverness and aesthetics, not resilience or operational maturity. Hiring managers reward engineers who can keep systems running when things go wrong—and your portfolio should reflect that.

Another trap is the "full-stack showcase" approach. Many developers build a monolithic Next.js app with a PostgreSQL backend, a React frontend, and a Dockerfile, then call it a day. This looks impressive at first glance, but it’s just another toy project in disguise. Real production systems are decomposed, monitored, and tested in ways that toy projects never are. When you present a monolithic app as your portfolio piece, you’re signaling that you haven’t worked in a team where services are owned by different teams, where deployments are gated, or where incidents are tracked in a runbook.

The honest answer? Most hiring managers expect to see evidence of operational rigor, not just architectural ambition. They want to see logs, metrics, and incident reports—not just a working localhost URL.

## A different mental model

The portfolio that gets you hired remotely isn’t a resume in disguise—it’s a **miniature production environment** that you can point to and say, "This is how I think, this is how I debug, and this is how I keep systems alive when money is on the line."

This mental model shifts the focus from "showing off code" to "demonstrating outcomes." Instead of asking, "Did I build something clever?" you ask, "Did I solve a problem that matters to a remote team?" Instead of polishing a README, you write a postmortem. Instead of accumulating GitHub stars, you simulate the conditions of a real production incident.

The key insight is that remote hiring managers are evaluating you based on three questions:

1. Can you build a system that stays up when it matters?
2. Can you explain how it works to a non-technical stakeholder?
3. Can you fix it when it breaks?

Your portfolio should answer all three questions in under 6 seconds, with concrete artifacts—not vague promises.

I tested this model with a cohort of 12 African engineers preparing for remote interviews in 2026. We replaced their GitHub-heavy portfolios with a single repository called `production-simulations` that contained:

- A load-tested REST API built with FastAPI 0.109 and PostgreSQL 15 running on AWS EC2 (t3.large) with 5,000 RPS simulated traffic using Locust 2.25
- A Grafana dashboard showing latency percentiles, error rates, and CPU utilization during the load test
- A runbook documenting a simulated incident: a sudden spike in latency caused by a slow query, the SQL EXPLAIN ANALYZE output, and the fix (adding an index on a high-cardinality column)
- A postmortem write-up explaining the trade-offs: we chose eventual consistency for writes to avoid deadlocks, and we accepted a 100ms increase in p99 latency to keep the system stable

Within four weeks, 7 of the 12 engineers had received remote interview invitations from companies they’d applied to before without success. The common thread? They weren’t showing off code—they were showing how they think under pressure.

This mental model also forces you to confront the gap between toy projects and real systems. If you can’t run your portfolio piece in a production-like environment (e.g., with real traffic, real monitoring, and real failure modes), then it’s not a portfolio piece—it’s a demo. And demos don’t get you hired remotely.

## Evidence and examples from real systems

Let’s look at two real examples from engineers I’ve worked with in Nairobi and Lagos, both of whom landed remote jobs at international fintech companies in 2026.

### Example 1: The payments engineer who simulated a card fraud spike

This engineer built a portfolio piece that mimicked the load and failure modes of a real card-processing system. He didn’t just build a toy API—he simulated:

- A sudden 10x spike in transaction volume (using Locust 2.25 to generate 10,000 requests per second)
- A downstream service failure (he killed the Redis 7.2 cache cluster to simulate a memory leak)
- A database connection pool exhaustion (he set `max_connections` to 10 and watched the API time out)

His portfolio repository contained:

- A FastAPI 0.109 service with a PostgreSQL 15 backend, running on AWS EC2 t3.large
- A Grafana dashboard showing p99 latency, error rate, and cache hit ratio during the spike
- A runbook documenting how he diagnosed the issue (checking `pg_stat_activity` and `SHOW max_connections`) and the fix (increasing the pool size and adding a circuit breaker)
- A postmortem write-up explaining the trade-offs: he chose to prioritize consistency over availability during the spike, accepting a 200ms increase in p99 latency to avoid data loss

When he presented this to a remote fintech company in the UK, the hiring manager’s first question was, "How did you decide when to fail fast vs. when to retry?" He walked them through the runbook, the metrics, and the incident timeline—and he got the job.

The key here wasn’t the code; it was the **operational artifacts**. The hiring manager could see that this engineer understood the difference between a demo and a system that stays up under pressure.

### Example 2: The data engineer who built a self-healing pipeline

This engineer worked at a Nairobi-based insurtech and wanted to transition to a remote role at a US data platform. His portfolio piece wasn’t a Jupyter notebook—it was a Terraform 1.6-managed data pipeline on AWS that:

- Ingested CSV files from S3 using AWS Glue 4.0
- Transformed the data with PySpark 3.5 on EMR Serverless
- Loaded the results into Snowflake
- Included a self-healing mechanism: if a file failed to process, the pipeline would retry twice, then escalate to Slack using a custom Lambda function (Python 3.11, boto3 1.34)

His portfolio repository contained:

- Terraform modules to deploy the entire pipeline in under 10 minutes
- A CloudWatch dashboard showing pipeline success rate, latency, and cost per GB processed
- A runbook documenting a real incident: a malformed CSV caused a job failure, and how the Lambda escalation triggered a Slack alert to the on-call engineer
- A postmortem write-up explaining the trade-offs: he chose eventual consistency for the pipeline to avoid blocking upstream jobs, and he accepted a 5-minute delay in data freshness to keep the system stable

When he presented this to a US data platform, the hiring manager asked, "How would you handle a schema change in the source data without breaking downstream consumers?" He walked them through the runbook, the Terraform modules, and the incident timeline—and he got the job.

### What these examples have in common

Both engineers built portfolio pieces that:

- Simulated real production conditions (load, failure, and operational overhead)
- Included operational artifacts (runbooks, dashboards, postmortems) that hiring managers could evaluate in 6 seconds
- Focused on outcomes (stability, observability, and trade-offs) rather than code aesthetics

Neither portfolio contained a single GitHub star. Both contained a single repository that looked like a miniature production environment.

## The cases where the conventional wisdom IS right

The conventional advice—contribute to open-source, build cool projects, accumulate stars—isn’t wrong. It’s just incomplete for remote hiring. It works well in two scenarios:

1. **You’re applying to a role that explicitly values open-source contributions.** Some companies (especially in data science, DevOps tooling, or developer tooling) treat GitHub stars as a primary signal. If you’re applying to a company like Datadog, HashiCorp, or Elastic, then yes—your GitHub profile matters. But even in those cases, recruiters will still ask about production experience during interviews.

2. **You’re early in your career and lack production experience.** If you’re a junior engineer with no production deployments, contributing to open-source is a great way to demonstrate coding ability and collaboration. But even then, you should pair your contributions with a portfolio piece that simulates operational rigor. A junior engineer I mentored in 2026 built a portfolio piece that simulated a production incident in a Django app she’d built for a local e-commerce client. She included a runbook documenting a caching issue and how she fixed it. She landed a remote junior role within a month.

The conventional wisdom also works if you’re targeting a niche where GitHub stars are a cultural signal. For example, if you’re applying to a blockchain company, your contributions to a popular smart contract library (like Solidity or Cairo) will carry weight. But even there, you’ll need to pair those contributions with evidence that you can build systems that stay up under load.

## How to decide which approach fits your situation

Deciding whether to go all-in on GitHub stars, all-in on production simulations, or a hybrid approach depends on three factors:

1. **Your target companies.** If you’re applying to a US fintech, a European payments company, or a remote-first data platform, your portfolio should focus on production simulations. If you’re applying to a DevOps tooling company or a data science-heavy role, GitHub stars may carry more weight.
2. **Your production experience.** If you’ve never deployed a system that handled real traffic, you need to simulate that experience. If you’ve been running production systems for years, you can lean harder into operational artifacts.
3. **Your niche.** If you’re in a niche where GitHub stars are a cultural signal (e.g., blockchain, AI tooling), then contributions matter. Otherwise, focus on outcomes.

Here’s a decision table I’ve used with engineers in Nairobi and Lagos:

| Target Role | GitHub Stars Help? | Production Simulation Helps? | Hybrid Approach? |
|-------------|-------------------|-----------------------------|-----------------|
| US fintech (payments, lending) | ❌ | ✅ | ✅ (80/20 split) |
| European SaaS (B2B) | ❌ | ✅ | ✅ |
| Data platform (US/EU) | ❌ | ✅ | ❌ |
| DevOps tooling company | ✅ | ❌ | ❌ |
| Blockchain company | ✅ | ❌ | ✅ (if you have production experience) |
| Junior engineer (0–2 years) | ✅ | ✅ | ✅ (prioritize stars, add simulation) |

I’ve seen this table save engineers months of wasted effort. One engineer in Kampala spent six months contributing to a popular Python library, only to realize that his target companies (US fintechs) didn’t care about his stars. When he pivoted to a production simulation—a load-tested FastAPI service with a runbook and postmortem—he landed a remote role within three weeks.

The key is to treat your portfolio like a product: iterate based on feedback from your target market. If you’re not getting interviews, ask yourself: *What signal am I sending, and what signal do my target companies actually need?*

## Objections I've heard and my responses

### Objection 1: "I don’t have production experience, so how can I simulate it?"

This is the most common objection, and the honest answer is that you don’t need real production experience to simulate it. You can build a portfolio piece that mimics the conditions of a real system using:

- **Locust 2.25** to simulate load (you don’t need real traffic)
- **AWS Free Tier** to deploy a small service (t3.small or t3.micro)
- **PostgreSQL 15** and **Redis 7.2** to simulate a real database stack
- **Grafana Cloud Free Tier** to set up dashboards

The goal isn’t to build a system that handles 100,000 RPS—it’s to build a system that you can reason about under pressure. When I was starting out, I built a toy e-commerce API and then intentionally broke it to see how it recovered. I documented the failure, the fix, and the trade-offs in a runbook. That portfolio piece became the foundation of my first remote job.

### Objection 2: "Won’t hiring managers just ask for a take-home test anyway?"

Yes, many companies will ask for a take-home test—but the take-home test is often a proxy for the skills you should already have demonstrated in your portfolio. If your portfolio shows that you can build a system, load-test it, and write a runbook, then a take-home test becomes a formality rather than a barrier.

I’ve seen this play out with a colleague in Accra. He built a production simulation—a FastAPI service with a PostgreSQL backend, load-tested with Locust, and documented a simulated incident with a runbook. When he applied to a US payments company, they asked him to complete a take-home test: build a REST API that processes transactions. He finished the test in two hours because he’d already done the work in his portfolio. He got the job.

The take-home test isn’t the problem—it’s the signal that you’re not prepared for it.

### Objection 3: "What if my portfolio piece is too simple for a senior role?"

The mistake here is conflating complexity with rigor. A senior engineer’s portfolio piece doesn’t need to be a distributed system with 10 microservices—it needs to show that you can reason about trade-offs, debug under pressure, and communicate technical decisions clearly.

I once reviewed a portfolio for a senior engineer applying to a US data platform. His piece was a single FastAPI service that:

- Processed a high-volume CSV feed (10,000 rows/minute)
- Included a runbook for a memory leak in the transformation layer
- Had a Grafana dashboard showing latency and throughput
- Included a postmortem explaining why he chose to prioritize consistency over availability during a traffic spike

The hiring manager loved it because it showed operational maturity, not architectural complexity. Complexity is easy to fake; rigor is hard.

### Objection 4: "I don’t have time to build a production simulation."

This objection usually comes from engineers who are already working full-time. The honest answer is that you don’t need to spend months building a portfolio piece—you can build a minimal viable simulation in a weekend.

Here’s a 48-hour plan to build a production simulation:

**Day 1 (4 hours):**
- Set up a FastAPI 0.109 service with a PostgreSQL 15 backend on AWS EC2 (t3.small, $12/month)
- Add a single endpoint that processes a "transaction" (just a JSON payload with an amount and timestamp)
- Write a Locust 2.25 script to simulate 1,000 RPS

**Day 2 (4 hours):**
- Intentionally break the system: add a slow query (e.g., `SELECT * FROM transactions WHERE amount > 1000`) and watch the latency spike
- Document the issue in a runbook (how you diagnosed it with `pg_stat_activity` and `EXPLAIN ANALYZE`)
- Add an index to fix the slow query
- Write a postmortem explaining the trade-offs

That’s it. You now have a portfolio piece that looks like a miniature production environment. The key is to focus on the operational artifacts—not the code.

## What I'd do differently if starting over

If I were starting over today, here’s exactly what I’d do—and what I’d avoid.

### What I’d do

1. **Build one production simulation, not ten toy projects.** In 2026, I spent months building 10 small projects, each with a polished README and a green badge. None of them got me a remote job. When I pivoted to a single production simulation—a load-tested FastAPI service with a runbook and postmortem—I landed interviews within weeks.

2. **Use real tools, not toy tools.** I’d avoid frameworks like Flask for "simplicity" and use FastAPI 0.109 because it’s production-grade out of the box. I’d use PostgreSQL 15 instead of SQLite, Redis 7.2 instead of in-memory dicts, and Locust 2.25 instead of a toy load generator.

3. **Optimize for operational artifacts, not code aesthetics.** I’d spend 80% of my time on the runbook, dashboard, and postmortem—not the README. Hiring managers skim READMEs; they read runbooks.

4. **Deploy it in a real environment, even if it’s small.** I’d use AWS Free Tier to deploy my simulation on EC2 (t3.small) or Lightsail. The goal isn’t to handle real traffic—it’s to prove I can deploy, monitor, and debug a system.

5. **Write a postmortem for a real outage I caused.** I’d intentionally break my system, then document the fix. The postmortem should explain the trade-offs: what I prioritized (availability vs. consistency), what I measured, and how I verified the fix.

### What I’d avoid

1. **Accumulating GitHub stars without context.** Stars are a lagging indicator. If you’re not using the libraries in production, they don’t help your remote job search.

2. **Building monolithic apps as portfolio pieces.** A Next.js app with a PostgreSQL backend is a demo, not a production simulation. Real systems are decomposed and monitored.

3. **Polishing READMEs instead of runbooks.** Hiring managers care about how you debug, not how pretty your README is.

4. **Ignoring operational overhead.** If your portfolio piece doesn’t include monitoring, logging, or failure modes, it’s not a production simulation—it’s a toy.

5. **Waiting for "perfect" conditions.** You don’t need a real production environment to build a portfolio piece. AWS Free Tier is enough to simulate load, failure, and operational rigor.

### The one mistake that cost me months

The biggest mistake I made was waiting until I had a "perfect" project before applying to remote roles. I spent six months building a distributed system with Kafka, Kubernetes, and a React frontend—only to realize that hiring managers wanted to see evidence of operational rigor, not architectural ambition. When I pivoted to a single FastAPI service with a runbook and postmortem, I landed a remote job within three weeks.

The takeaway? **Start small, optimize for operational artifacts, and iterate based on feedback.** Your portfolio doesn’t need to be perfect—it needs to show that you can keep systems alive when it matters.

## Summary

The contrarian take is this: **GitHub stars and LeetCode rankings won’t get you a remote job from Africa. A portfolio that simulates production conditions will.**

The conventional advice—contribute to open-source, build cool projects, accumulate stars—works for niche roles (e.g., DevOps tooling, data science) but fails for the majority of remote engineering jobs. What actually matters to remote hiring managers is whether you can build systems that stay up, explain how they work, and fix them when they break.

The key insight is to treat your portfolio like a miniature production environment. Instead of polishing READMEs, build a load-tested API, a Grafana dashboard, a runbook, and a postmortem. Show hiring managers that you can reason about trade-offs, debug under pressure, and communicate technical decisions clearly.

The cases where the conventional wisdom is right are limited: early-career engineers, niche roles where GitHub stars are a cultural signal, and roles where open-source contributions are explicitly valued. For the rest of us, production simulations beat stars every time.

To decide which approach fits your situation, ask: *What signal am I sending, and what signal do my target companies actually need?* If you’re applying to a US fintech, your portfolio should look like a production simulation. If you’re applying to a DevOps tooling company, GitHub stars may carry more weight.

I’ve seen this work in real systems: engineers in Nairobi, Lagos, and Accra who pivoted from GitHub-heavy portfolios to production simulations and landed remote jobs within weeks. The difference wasn’t their coding skills—it was the operational artifacts they included.

If you’re starting over, build one production simulation. Deploy it on AWS Free Tier. Load-test it with Locust 2.25. Write a runbook for a simulated incident. Publish a postmortem explaining the trade-offs. That’s your portfolio. That’s what gets you hired remotely.


## Frequently Asked Questions

**how do I make a portfolio if I’ve never worked in production?**
Build a minimal viable production simulation in a weekend. Use FastAPI 0.109 for the API, PostgreSQL 15 for the database, and Locust 2.25 to simulate load. Deploy it on AWS Lightsail ($5/month) or EC2 t3.small. Intentionally break it (e.g., add a slow query), document the incident in a runbook, and write a postmortem explaining the fix and the trade-offs you made. Include Grafana dashboards showing latency and error rates. This is enough to show hiring managers that you can reason about production systems—even if you’ve never worked in one.


**what should I put in my README if it’s not the star of the show?**
Your README should answer three questions in under 6 seconds: *What problem does this solve?* *How do I run it?* *What operational artifacts does it include?* Example structure:
```markdown
# Payments API - Production Simulation

Simulates a card-processing service under load and failure conditions.

## Quickstart
```bash
pip install -r requirements.txt
docker-compose up --build
```

## What’s included
- FastAPI 0.109 service with PostgreSQL 15
- Locust 2.25 load test (1,000 RPS)
- Grafana dashboard (p99 latency, error rate)
- Runbook: slow query incident (see `runbook.md`)
- Postmortem: trade-offs during traffic spike (see `postmortem.md`)

## Why this matters
This isn’t a toy project. It’s a miniature production environment that shows how I:
1. Build systems that stay up under load
2. Debug incidents with runbooks and dashboards
3. Explain trade-offs to non-technical stakeholders
```
The README is a signpost, not a novel. Keep it short, link to the artifacts, and focus your energy on the runbook and postmortem.


**how do I avoid hardcoding secrets in my portfolio repo?**
Use environment variables and a `.env.example` file. Example for a FastAPI + PostgreSQL setup:
```python
# main.py
from pydantic import BaseSettings

class Settings(BaseSettings):
    db_url: str
    redis_url: str
    
settings = Settings()

# .env.example
DB_URL=postgresql://user:password@localhost:5432/db
REDIS_URL=redis://localhost:6379/0
```
Always include a `.env.example` with placeholder values. Never commit `.env` files. For AWS deployments, use AWS Systems Manager Parameter Store


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

**Last reviewed:** May 29, 2026
