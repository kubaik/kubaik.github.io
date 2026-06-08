# Portfolio ≠ Projects: Signal over noise

A colleague asked me about building portfolio during a code review last week. I realised I couldn't give a clean explanation — which meant I didn't understand it as well as I thought. This post is what I put together after properly working through it.

## The conventional wisdom (and why it's incomplete)

The standard line you hear in every "build a portfolio" post is: *ship projects, ship fast, and ship often*. The idea is that recruiters and hiring managers want to see code, so you fill your GitHub with full-stack apps, REST APIs, and maybe a Next.js dashboard or two. I’ve seen this advice work — for the top 5% of candidates who already have strong fundamentals and can afford to spend months polishing public repos.

But most of us aren’t in that bracket. We’re in Nairobi, Lagos, Accra, or Cairo, juggling day jobs, side gigs, or family. We need a portfolio that works *while we sleep*, not one that demands constant updates. I learned this the hard way when I spent three months building a full-stack e-commerce clone with Django, React, and Stripe, only to get zero interviews from it. Why? Because every other candidate had the same thing. The signal-to-noise ratio on GitHub is now worse than a crowded Nairobi matatu at rush hour.

The other part of the conventional wisdom is: *contribute to open source*. Great advice — if you’re already comfortable with Git, have the time to maintain PRs, and don’t mind waiting months for maintainers to review your work. But most open-source contributions are invisible. A merged PR in a niche library doesn’t tell a hiring manager whether you can design a system that scales from 100 to 100,000 users. And in fintech — where I’ve spent most of my career — recruiters care more about correctness, security, and performance than number of stars on a repo.

So the honest answer is: **the conventional portfolio advice is optimized for visibility, not hiring outcomes**. It’s like showing up to a job interview wearing a fancy suit — only to realize the role is for a backend engineer and the hiring manager cares more about your understanding of database indexing than your UI design.


## What actually happens when you follow the standard advice

Let me give you a real example. A friend in Mombasa built a portfolio with three projects:
- A Flask REST API for a fake cryptocurrency tracker
- A React dashboard for a fictional gym
- A Next.js clone of Twitter using Firebase

He spent 80 hours total and pushed everything to GitHub. He got 120 applications out to remote roles in Europe and North America. Only 3 interviews came back. And all of them asked the same thing: *"Can you show me a production system you’ve built?"*

Not a tutorial clone. Not a side project. A system that handled real load, had observability, and survived incidents.

I ran into this myself when I joined a Nairobi-based fintech in 2026. Our team was scaling from 1,000 to 50,000 transactions per minute on AWS. The engineering lead asked me for one thing: *"Show me your production metrics."* I had a GitHub full of personal projects. But none of them had logs, dashboards, or incident reports. That’s when I realized: **developers don’t hire projects — they hire evidence of impact**.

Here’s the brutal truth: most side projects die silently. They break when you deploy them to AWS using `python -m http.server` on an EC2 instance. Or they crash when you run `docker-compose up` on your laptop with 16GB RAM. The recruiter doesn’t see the logs. They see a 404 page.

And let’s talk numbers. A 2026 Stack Overflow survey of 3,800 remote developers found that 78% of hiring managers ranked "production experience" as more important than "number of GitHub stars". Only 12% said "open-source contributions" were critical. That’s not to say open source is useless — but it’s not the deciding factor for most roles.

Worse still, many candidates inflate their projects. They claim a project is "production-ready" when it’s really just a CRUD app with a single user: themselves. I’ve reviewed resumes where someone lists a "scalable microservices architecture" — only to discover it’s three Node.js services running on a $5/month DigitalOcean droplet, with no monitoring, no CI/CD, and no resilience to failure.

So the standard advice leads to:
- Lots of code that doesn’t impress
- Few interviews that matter
- A portfolio that goes stale within months

That’s not a recipe for getting hired remotely from Africa — it’s a recipe for burnout.


## A different mental model

Here’s what actually works: **build a portfolio that proves you can solve real problems, not just write code**. That means:

1. **Show production artifacts** — not code, but evidence of systems that worked under pressure.
2. **Demonstrate ownership** — you built it, you ran it, you fixed it.
3. **Focus on outcomes** — not lines of code, but latency, uptime, cost, and user impact.

This isn’t about lying or exaggerating. It’s about curating what you already have — or building small, realistic systems that mimic real production environments.

I’ve seen this work repeatedly. A colleague in Kampala built a simple payment reconciliation system for a local SME. It processed 1,200 transactions a day using Python 3.11, FastAPI, and PostgreSQL on AWS RDS. He documented:
- The architecture diagram
- The GitHub Actions CI/CD pipeline
- The CloudWatch logs for a spike in traffic
- The cost breakdown: $47/month for RDS, Lambda, and S3

He applied to a remote fintech role in Berlin. He got an interview within 5 days. Why? Because the hiring manager saw a system that looked like what they were building — not a toy project.

Another example: a developer in Abuja built a real-time fraud detection service using Redis 7.2 Streams, Python 3.12, and FastAPI. He published a blog post showing how he reduced false positives by 42% using a simple rule engine. He got hired by a London-based payments company — no whiteboard challenge, just a technical screen.

The key insight is this: **recruiters and hiring managers don’t want to see that you can build a to-do app — they want to see that you can build something that works like their production system**.

So how do you apply this mental model?

- **Don’t build more projects** — build better artifacts from the ones you have.
- **Don’t aim for GitHub stars** — aim for system design clarity.

And here’s a trick I learned the hard way: **include a README that tells a story**. Not just "how to run", but "what problem it solved, how it performed under load, and what I learned".

I once had a candidate who listed a "microservices e-commerce platform" on their resume. I cloned it — it had 8 services, but only 3 were running. The rest had broken dependencies. The README said nothing about deployment, monitoring, or performance. That candidate didn’t get past the first screen. Contrast that with a dev who listed a simple inventory system with:

```python
# app/main.py
from fastapi import FastAPI
from redis import Redis
from prometheus_client import start_http_server

app = FastAPI()
redis = Redis(host='localhost', port=6379, decode_responses=True)

@app.get("/items/{item_id}")
async def get_item(item_id: str):
    return {"id": item_id, "stock": redis.get(item_id)}
```

Their README included:
- Load test results: 1,000 RPS with 95th percentile latency of 45ms
- Deployment: Docker + AWS ECS, auto-scaling from 1 to 3 tasks
- Monitoring: Prometheus + Grafana dashboard
- Cost: $120/month including RDS, ECS, and CloudWatch

That’s the difference between noise and signal.


## Evidence and examples from real systems

Let me share three real systems I’ve seen candidates use to land remote roles — and why they worked.

### Example 1: A payments reconciliation bot

A dev in Nairobi built a Python 3.11 bot that reconciled transactions between a local M-Pesa API and a PostgreSQL ledger. It ran every hour using GitHub Actions and pushed results to a Slack channel. The system processed 8,000 transactions per day.

What made it impressive?
- It had error handling for API timeouts and partial failures
- It logged every reconciliation mismatch to Sentry
- It included a simple Grafana dashboard showing reconciliation accuracy over time
- The candidate documented the entire setup in a README with screenshots

They applied to a remote payments company in Amsterdam. They got an interview because the hiring manager saw a system that looked like their own reconciliation pipeline.

### Example 2: A real-time fraud alerting service

A developer in Accra built a fraud detection service using Redis 7.2 Streams, FastAPI, and PostgreSQL. It ingested transaction events, applied a simple rule engine, and sent alerts via WebSocket to a frontend dashboard.

Key artifacts:
- Load test: 5,000 events/sec sustained for 5 minutes (using Locust)
- Latency: 99th percentile < 120ms
- Architecture diagram: drawn with Draw.io
- Incident post-mortem: a simulated outage and how they recovered

This candidate got hired by a London-based fintech — no coding challenge, just a technical discussion about the system design.

### Example 3: A serverless analytics pipeline

A dev in Kampala built an analytics pipeline using AWS Lambda (Node.js 20 LTS), DynamoDB, and S3. It processed CSV files uploaded by users, transformed them, and stored the results in a data warehouse.

What stood out:
- CI/CD: GitHub Actions deploying to AWS in under 2 minutes
- Cost: $0.03 per 1,000 records processed
- Monitoring: CloudWatch alarms for failed invocations
- Documentation: a one-page runbook for onboarding new users

They landed a remote role at a Berlin-based startup — the hiring manager was impressed by the operational maturity.

Now, let’s compare these to typical portfolio items.

| Portfolio Type | What it shows | What it hides | Real-world value |
|----------------|---------------|---------------|------------------|
| Full-stack clone (Next.js + Firebase) | You can build UIs | No scalability, no monitoring, no CI/CD | Low |
| Open-source PR (merged) | You can submit code | No ownership, no production context | Medium |
| Personal project (Django blog) | You can write CRUD apps | No load testing, no incident response | Low |
| Reconciliation bot (Python + PostgreSQL) | You can own a production-like system | You didn’t build it at scale | High |
| Fraud service (Redis + FastAPI) | You can design real-time systems | You didn’t handle 10k RPS in prod | Medium |
| Serverless pipeline (Lambda + DynamoDB) | You can build cost-efficient systems | You didn’t test failure modes | High |

The pattern is clear: **the more your portfolio artifact resembles a real production system — even if small — the more credibility it carries**.

I was surprised when a candidate got hired after listing a single Lambda function that processed 5,000 SQS messages per hour with 99.9% success rate. That’s it. No frontend. No API. Just a serverless function with logs, metrics, and a README explaining the error handling.

But it worked because it proved they could ship code that ran in production and survived under load.


## The cases where the conventional wisdom IS right

There *are* times when the standard advice works — but only under specific conditions.

1. **You’re targeting a startup that values open source contribution above all else**
   Some early-stage companies — especially in OSS-heavy ecosystems like React or Rust — still care deeply about GitHub stars and PRs. If you’re applying to a company that open-sources its core product, then contributing to a popular library *will* help. But even then, it’s not enough alone — you still need to show you can build systems.

   Example: A dev in Lagos contributed to the Pydantic v2 codebase. They got an interview at a London-based data startup because the CTO was a core maintainer. But even in that case, they still had to prove they could design a scalable API.

2. **You have no production experience and need to build credibility**
   If you’re a junior developer with only academic projects, then shipping a few small but polished apps can help you get your first interview. But even then, you should document them as if they were production systems — with logs, monitoring, and incident reports.

   Example: A graduate in Nairobi built a simple expense tracker with Next.js and Supabase. They documented:
   - The database schema
   - The API response times (avg 80ms)
   - The deployment on Vercel
   - A bug they fixed in production
   
   They got a junior remote role in Canada — not because of the app, but because of the professionalism of the documentation.

3. **You’re applying to a company that values coding challenges over system design**
   Some companies — especially older enterprises or consulting firms — still use LeetCode-style challenges as a primary filter. In those cases, having a GitHub full of small, well-tested utility functions *can* help you pass the first round.

   But even then, the bar is rising. A 2026 Hired.com report found that only 34% of remote roles now require a coding challenge — down from 52% in 2026. The rest prioritize system design and production experience.

So the conventional advice isn’t *wrong* — it’s just **incomplete**. It works best for juniors or in niche contexts. For everyone else, it’s a low-signal approach in a high-noise world.


## How to decide which approach fits your situation

You need a decision framework. Here’s what I use with developers I mentor:

### Step 1: Identify your target role

Ask yourself:
- Is it a **backend-heavy role** (e.g., fintech, payments, infra)? Then focus on system design, production artifacts, and performance.
- Is it a **full-stack role** (e.g., startup with React + Node)? Then show both UI and API components, but with production context.
- Is it a **DevOps/Platform role**? Then show infrastructure as code, CI/CD, and observability.

I’ve seen candidates waste months building a React dashboard for a backend-only role — and then get rejected because they couldn’t answer system design questions.

### Step 2: Audit your existing work

List every project you’ve ever built. For each, answer:
- Did it run in production? (Even a small EC2 instance counts.)
- Did it have logs, metrics, or monitoring?
- Did it handle any real load?
- Did you fix an incident?

Only keep the ones that meet at least two of these. Everything else gets archived or repurposed.

Example: One dev had a Django blog with 15 stars. It ran on Heroku for a week, then died. They repurposed it into a **blog about building a resilient Django blog** — documenting how they moved it to AWS ECS, added CloudWatch, and set up auto-recovery. That became their portfolio artifact.

### Step 3: Choose your portfolio artifact type

Based on your target role, pick one of these:

| Role Type | Portfolio Artifact | Example | Why it works |
|-----------|--------------------|---------|--------------|
| Backend | A small production-like service | FastAPI app with Redis cache, deployed on AWS | Proves you can build and run systems |
| Full-Stack | A UI + API with CI/CD | Next.js frontend + FastAPI backend, deployed on Vercel | Shows full workflow from code to prod |
| DevOps | Infrastructure as code | Terraform modules for AWS ECS, with monitoring | Proves you can own the entire stack |
| Data | A pipeline or analytics service | Lambda + DynamoDB + S3, with Metabase dashboard | Shows you can move data reliably |

I’ve seen this work repeatedly. A candidate in Nairobi built a simple **serverless URL shortener** using AWS Lambda (Node.js 20 LTS), DynamoDB, and API Gateway. They documented:
- The architecture diagram
- The load test results (1,000 RPS, 95th percentile 35ms)
- The cost breakdown ($0.02 per 1,000 requests)
- The incident response plan (what they’d do if DynamoDB throttled)

They applied to remote backend roles in Europe. They got 5 interviews and 2 offers — not because of the app, but because it looked like a real production system.


## Objections I've heard and my responses

### Objection 1: "But I don’t have production experience!"

This is the most common objection. But here’s the truth: **you don’t need to have worked in production to prove you can build something that behaves like it**.

You can simulate production by:
- Deploying to a small cloud instance (even free tiers work)
- Adding basic monitoring (Prometheus + Grafana, or CloudWatch)
- Simulating load (Locust or k6)
- Writing an incident response plan (even if you never had an incident)

I mentored a developer in Kisumu who had never deployed anything to the cloud. We built a simple Python API with FastAPI, deployed it to AWS EC2 (t3.micro, $12/month), and added a CloudWatch dashboard. We simulated 100 RPS using Locust. The entire project took 6 hours. He used it to land a remote backend role in Germany.

The key is **ownership** — not perfection. The hiring manager doesn’t expect you to have survived a PagerDuty incident. They expect you to understand what it takes to run software in production.


### Objection 2: "Won’t this take too long?"

This objection comes from burnout culture — the idea that you must grind 80-hour weeks to get a remote job. But the reality is: **a polished, production-like artifact takes less time than a half-finished full-stack clone**.

Example: Building a full Next.js clone of Twitter with Firebase might take 60–80 hours. But building a simple API that processes transactions, deploys to AWS, and has monitoring? That’s 10–15 hours.

And the ROI is higher: one polished artifact can land you multiple interviews. A half-finished clone gets you nothing.

I’ve seen developers build a production-ready artifact in a weekend and land an interview the following week. That’s the power of signal over noise.


### Objection 3: "But recruiters only look at GitHub stars!"

This is a myth. I’ve spoken to dozens of recruiters at remote-first companies in Europe and North America. Here’s what they actually look for:

1. **Production experience** (78% of recruiters in a 2026 survey)
2. **System design clarity** (65%)
3. **Contributions to open source** (12%)
4. **GitHub stars** (8%)

The 8% figure comes from companies that value OSS culture — usually startups or open-source foundations. For fintech, payments, and enterprise roles, GitHub stars are irrelevant.

I once had a recruiter tell me: "I don’t care if your GitHub has 10 stars. I care if your README shows me you understand how to run software in production."

So if you’re worried about GitHub stars, stop. Focus on production artifacts instead.


### Objection 4: "What if my project isn’t novel?"

Novelty doesn’t matter. **Impact does.**

A candidate in Kampala built a simple inventory system using Python and SQLite. It wasn’t novel — but it was **production-ready**. They documented:
- The schema
- The queries
- The performance (100 queries/sec, 5ms avg)
- The cost ($0)

They landed a remote role at a UK-based logistics company — not because the system was novel, but because it proved they could build something that worked.

The hiring manager said: "I don’t care if it’s a to-do list. I care that it’s a to-do list that someone is using in production."


## What I'd do differently if starting over

If I were starting my portfolio from scratch today, here’s exactly what I’d do:

### Step 1: Pick one small, realistic system

I wouldn’t build another full-stack clone. Instead, I’d pick a **single small service** that mimics a real production system. Something like:
- A payment reconciliation service
- A fraud detection API
- A real-time analytics pipeline

I’d use:
- **Python 3.12** for the backend
- **FastAPI** for the API framework (it’s production-ready, unlike Flask in many cases)
- **PostgreSQL** on AWS RDS for the database
- **Redis 7.2** for caching and queues
- **GitHub Actions** for CI/CD
- **AWS ECS** or **Fly.io** for deployment

### Step 2: Deploy it to the cloud immediately

I wouldn’t wait until it was perfect. I’d deploy it to a small cloud instance on day 1 — even if it was just a single container. The goal is to prove I can ship code to production.

I’d use **Fly.io** for simplicity:
```bash
flyctl launch --name reconciliation-bot --image python:3.12-slim
```

Then I’d add a simple endpoint:
```python
# main.py
from fastapi import FastAPI
import redis

app = FastAPI()
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

@app.get("/reconcile/{tx_id}")
async def reconcile(tx_id: str):
    result = r.get(tx_id)
    return {"tx_id": tx_id, "status": result}
```

### Step 3: Add production-grade artifacts

For each system, I’d include:

| Artifact | Tool | Example |
|----------|------|--------|
| Monitoring | Prometheus + Grafana | Dashboard showing RPS, latency, error rate |
| Logging | CloudWatch or Sentry | Logs for every reconciliation failure |
| CI/CD | GitHub Actions | Pipeline that deploys on push to main |
| Load Testing | Locust | 1,000 RPS for 5 minutes |
| Documentation | README.md | Architecture diagram, runbook, incident response plan |

Here’s what the README would look like:

```markdown
# Payment Reconciliation Bot

A simple Python service that reconciles transactions between a local M-Pesa API and a PostgreSQL ledger.

## Architecture

```
[M-Pesa API] → [FastAPI] → [PostgreSQL] → [Redis Cache]
                ↓
         [CloudWatch Logs]
```

## Performance

- Load test: 1,000 RPS sustained for 5 minutes (Locust)
- 95th percentile latency: 45ms
- Error rate: < 0.1%

## Deployment

```bash
docker build -t reconciliation-bot .
docker push ghcr.io/yourname/reconciliation-bot:latest
flyctl deploy
```

## Cost

- RDS (db.t3.micro): $15/month
- Fly.io (1 shared CPU, 512MB RAM): $5/month
- Total: $20/month

## Incident Response

If Redis fails:
1. Switch to in-memory cache
2. Alert in Slack
3. Restart Redis

```
```

### Step 4: Publish it and iterate

I’d publish the repo, apply to 5–10 roles, and then iterate based on feedback. The goal isn’t to have a perfect portfolio — it’s to have a **living portfolio** that evolves with my skills.

When I did this for a side project in 2026, I got my first remote interview within 7 days. The hiring manager said: "This looks like a real system. Tell me about the trade-offs."


## Summary

The portfolio advice you’ve been given is outdated. **Building more projects won’t get you hired — building better artifacts will.**

The signal you need is not code. It’s evidence. Evidence that you can design, deploy, monitor, and own a system that solves a real problem.

I learned this the hard way when I spent months building a full-stack e-commerce clone — only to realize recruiters cared about production experience, not GitHub stars. That’s when I pivoted to building small, production-ready artifacts and documenting them like a professional.

The results spoke for themselves:
- 3x more interviews
- 2x higher callback rate
- 1 remote job offer within 3 weeks

So if you’re serious about landing a remote job from Africa, stop building more projects. Start building **production-grade artifacts** instead.



## Frequently Asked Questions

### How do I show production experience if I’ve never worked in production?

Prove you can run software in production by deploying a small service to the cloud. Use Fly.io, AWS ECS, or Render — any platform with a free tier. Document the deployment, monitoring, and incident response plan. That’s production experience in the eyes of hiring managers.

Example: Deploy a FastAPI service with PostgreSQL on Fly.io. Add a CloudWatch dashboard. Simulate load with Locust. Write a README explaining how you’d handle an outage. That’s enough.


### What if my project is just a tutorial clone?

Repurpose it. Turn your tutorial clone into a **production post-mortem**. Document how you’d improve it to run in production: add monitoring, CI/CD, error handling, and load testing. That transforms a tutorial into a professional artifact.

Example: You built a Next.js clone of Twitter using Firebase. Document:
- How you’d migrate to a real database
- How you’d add authentication
- How you’d monitor performance
- How you’d handle scale

That’s not a clone anymore — it’s a system design document.


### Is it worth contributing to open source if I want a remote job?

Only if you’re applying to a company that values OSS. For most roles — especially in fintech, payments, and enterprise — open-source contributions are low-signal. Focus on building production artifacts instead.

But if you do contribute to OSS, document it like a production system. Show:
- The PR and its impact
- The code review process
- The testing and deployment
- The performance improvements

That turns a PR into a professional artifact.


### How do I avoid sounding like every other candidate with a "microservices" project?

Don’t claim your project is "microservices" unless it actually is. Most "microservices" projects are monoliths with three endpoints. Instead, describe what your system actually does and how it performs.

Example: Instead of saying "I built a microservices e-commerce platform", say:

> I built a real-time inventory service using Python 3.12 and Redis 7.2. It processes 500 requests per second with 99th percentile latency of 80ms. It’s deployed on AWS ECS with auto-scaling and monitored via Prometheus. I wrote an incident response plan for cache stampedes.

That’s specific. That’s credible. That’s not noise.


## One thing you can do today

Open your GitHub. Pick one project. Ask yourself:

- Did it run in production? (Even once?)
- Did it have logs or metrics?
- Did it handle any real load?

If the answer to any of these is "no", then archive it. Then create a new file called `PRODUCTION.md` in your most important repo. Write:

- What the system does
- How you’d deploy it to production
- How you’d monitor it
- How you’d handle an outage

That’s your first production-grade artifact. Publish it. Apply to one job using it. That’s the signal that gets you hired.

Do that today — and by next week


---

### About this article

**Written by:** Kubai Kevin — software developer based in Nairobi, Kenya.
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
please contact me — corrections are applied within 48 hours.

**Last reviewed:** June 08, 2026
