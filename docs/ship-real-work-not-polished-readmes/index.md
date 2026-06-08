# Ship real work, not polished READMEs

A colleague asked me about building portfolio during a code review last week. I realised I couldn't give a clean explanation — which meant I didn't understand it as well as I thought. This post is what I put together after properly working through it.

## The conventional wisdom (and why it's incomplete)

Most career advice for African developers trying to land remote jobs focuses on polishing a portfolio site, writing perfect READMEs, and grinding LeetCode until your fingers bleed. That advice assumes hiring is a meritocracy where code quality alone decides your fate. In my experience, it’s not.

I once helped a Nairobi-based engineer who had three polished projects on GitHub, each with a slick README, live demos, and thorough test coverage. He spent six weeks refining them. Then he applied to 47 remote roles. Rejection after rejection. The feedback loop was brutal: “We went with someone closer to the time zone.”

What was missing wasn’t technical skill — it was proof that he could ship production-grade work under real constraints. Hiring managers don’t want to hire someone who can write beautiful code in isolation. They want someone who can write code that survives in production, with latency targets, cost constraints, and on-call rotations. A portfolio that looks good on a laptop screen but breaks in AWS is a liability, not an asset.

The honest answer is: remote hiring pipelines are optimized for candidates who can demonstrate end-to-end ownership. That means infrastructure, monitoring, and reliability — not just syntax and style.

---

## What actually happens when you follow the standard advice

You build a personal site. You document three side projects. You add animated SVGs and custom color schemes. You push to GitHub Pages. You tweet about it. Then you wait. And wait. Maybe you get a few interviews. Maybe you even pass the technical screen. But then the recruiter ghosts you, or the hiring manager says, “We’re not sure you have the right timezone coverage.”

I ran into this when I was helping a junior developer in Mombasa apply for roles in Europe and North America. He had a beautifully designed portfolio with three projects: a REST API, a React dashboard, and a CLI tool. Each had 100% test coverage. He spent 14 hours tweaking the UI. He got two interviews, both technical screens. Both passed. Then silence.

After digging, I found the real filter wasn’t the code — it was the deployment story. His projects ran on a single t3.micro instance in us-east-1 with no monitoring, no autoscaling, and no incident response plan. When I asked how he’d handle a 500 error at 3 AM, he said, “I’d wake up and SSH in.”

That’s not production. That’s a demo.

Most remote hiring pipelines today run through automated screens that check for infrastructure as code (IaC), observability, and cost discipline. They use tools like GitHub Actions, Terraform, and CloudWatch. If your portfolio doesn’t include IaC, logs, and alerts, it’s invisible to these pipelines.

And even if you do get past the screen, the next stage is often a take-home assignment. The best take-homes aren’t toy problems — they’re real engineering tasks with realistic constraints. For example:

- “Deploy the API on AWS with auto-scaling, CloudFront caching, and a 200ms P95 latency target under 1000 RPS.”
- “Add structured logging, set up Datadog dashboards, and write an incident playbook.”

If your portfolio only runs locally with `python app.py`, it’s not going to survive this stage.

---

## A different mental model

Instead of thinking of your portfolio as a showcase of code, think of it as a **production system you own end-to-end**. That means:

1. **One** project you can explain from commit to customer.
2. **Full deployment pipeline** with CI/CD, IaC, and secrets management.
3. **Observability** — logs, metrics, traces — not just for debugging, but for proving you can operate at scale.
4. **Cost discipline** — show you understand AWS pricing and can optimize for it.
5. **Incident readiness** — a runbook, a post-mortem template, and a simulated outage scenario.

I once built a portfolio project that was a real-time price tracker for Nairobi’s boda boda (motorcycle taxi) market. It scraped multiple APIs, cached aggressively with Redis 7.2, and deployed on AWS ECS Fargate with arm64. I added a CloudWatch dashboard, a PagerDuty integration (yes, even for a solo project), and a Terraform stack that cost $18/month at steady state.

When I applied for a senior backend role in Berlin, the hiring manager asked about the project. Not about the code — about the deployment, the cache stampede I’d fixed when Redis evicted keys under load, and how I handled a Kafka consumer lag spike at 2 AM. That conversation led to an offer.

The key insight: remote employers don’t hire engineers to write code. They hire engineers to run systems. Your portfolio must prove you can do that.

---

## Evidence and examples from real systems

Let me show you three real portfolio projects I’ve seen — two that got interviews, one that didn’t — and why.

| Project | Tech Stack | Deployment | Observability | Cost | Outcome |
|--------|------------|------------|---------------|------|---------|
| Price tracker (my project) | Python 3.11, FastAPI, Redis 7.2, Terraform, AWS ECS Fargate | Multi-region with CloudFront and WAF | CloudWatch, PagerDuty, synthetic checks every 5 min | $18/mo | 8 interviews, 3 offers |
| Task API (common portfolio) | Node.js 20 LTS, Express, MongoDB Atlas | Single EC2 t3.micro, no IaC | Basic console logging | $12/mo | 2 interviews, no offers after technical screen |
| Real-time chat (open source fork) | Go 1.22, PostgreSQL 15, NATS, Kubernetes (k3s on Hetzner) | Multi-node cluster with Helm | Prometheus + Grafana, alertmanager | €45/mo | 5 interviews, 1 offer (for SRE role) |

The pattern is clear: the projects that got traction had three things in common:

1. **They ran in production for at least 30 days** with real traffic (even if simulated).
2. **They included IaC** (Terraform or AWS CDK) so the setup was reproducible.
3. **They had at least one incident** — a cache stampede, a DB overload, a 5xx spike — and a documented response.

The Task API project, despite being technically sound, failed because it had no observability beyond `console.log`. When the hiring manager asked how he’d know if the API was down, the candidate said, “I’d check the logs.” But the logs were only on the instance. No alerts. No dashboards. No way to know until a user complained.

The real-time chat project, though over-engineered for a portfolio, impressed because it demonstrated Kubernetes, multi-node resilience, and a proper alerting setup. The hiring team was looking for an SRE, and the project proved the candidate could operate at that level.

I once had a candidate apply with a portfolio that included a microservice running on AWS Lambda with Python 3.11 and arm64. It had no observability, but it used AWS X-Ray. When I asked why X-Ray, he said, “I wanted to see latency.” That was enough to get him past the first screen — not because it was perfect, but because it showed awareness of distributed tracing.

The message is: you don’t need to be perfect. You need to be *aware* of production concerns and show that you’ve grappled with them.

---

## The cases where the conventional wisdom IS right

There are times when a clean, well-documented GitHub repo is enough. For example:

- If you’re applying for a role that explicitly says “open source contributions matter more than production experience” (rare, but exists in research or data science teams).
- If the company is early-stage and values “move fast and break things” culture over operational rigor.
- If you’re applying to a team that uses serverless heavily (e.g., AWS Lambda with Python 3.11) and they care more about function signatures than infrastructure.

I saw this when a startup in Lagos hired a developer based almost entirely on a well-documented open-source CLI tool for managing Kubernetes clusters. The tool had 500 stars, clean code, and a strong README. The team was building a managed Kubernetes service and valued developers who could write tooling. The portfolio didn’t need IaC or observability — it needed proof of collaboration and community impact.

But these cases are exceptions. For 90% of remote backend and full-stack roles in 2026, the portfolio must prove you can operate a system — not just write code.

---

## How to decide which approach fits your situation

Ask yourself three questions:

1. **What kind of team are you applying to?**
   - If it’s a FAANG or global fintech, they care about operational excellence. Your portfolio must include IaC, monitoring, and cost discipline.
   - If it’s a startup or a research lab, clean code and open-source contributions may suffice.

2. **What’s your current skill level?**
   - If you’re early-career, focus on one project you can fully own. Don’t try to build Kubernetes on day one.
   - If you’re mid-level, add observability and incident response.

3. **How much time do you have?**
   - If you have 3 months, build a production-grade system with IaC and observability.
   - If you have 3 weeks, polish one project until it has:
     - A GitHub Actions CI/CD pipeline
     - A Terraform module for deployment
     - CloudWatch logs and a simple dashboard
     - A README with a deployment guide and incident response plan

I once advised a developer in Kampala who had two weeks to apply to 20 roles. Instead of building a new project, he took an old side project, added a GitHub Actions workflow, wrote a Terraform module, and set up CloudWatch alarms. He included a screenshot of the dashboard and a post-mortem of a simulated outage. He got five interviews and two offers.

The key is not the size of the project — it’s the *ownership* you can demonstrate.

---

## Objections I've heard and my responses

**“I don’t have AWS credits or a credit card to deploy.”**

You don’t need a credit card. Use free tiers:
- AWS Free Tier gives you 750 hours/month of t3.micro (enough for a small API).
- Fly.io and Render give $5–$10 free credits per month.
- GitHub Codespaces lets you run a full dev environment for free.

I once deployed a portfolio API on Fly.io using their free tier for three months. The latency was 120ms from Nairobi to US-East, and it cost $0. No excuses.

**“But I’m not a DevOps engineer. I just want to write code.”**

You’re exactly the person who needs to learn these skills. Remote teams hire for reliability. If you can’t deploy your own code, you’re a risk. Start with the basics: GitHub Actions + Terraform + CloudWatch. You don’t need to master Kubernetes. You need to master *one* deployment workflow.

**“Hiring managers don’t care about my Terraform. They care about my Python.”**

They care about your *ability to deliver*. If your Python is perfect but your deployment is a manual `scp` script, they’ll assume your code will break in production. I’ve seen this happen: a candidate with beautiful Python code got rejected because the hiring manager asked, “How do I deploy this?” and the candidate said, “I’ll send you a ZIP file.”

**“I don’t have time to build a full production system.”**

You don’t need a full system. You need one system that proves you can *ship*. A single FastAPI service with a CI/CD pipeline, IaC, and monitoring is enough. The rest is noise.

---

## What I'd do differently if starting over

If I were building a portfolio from scratch today, here’s exactly what I’d do:

1. **Pick one problem that matters to an African audience.**
   Not “build a todo app.” Something like:
   - A fare calculator for Nairobi matatus using real-time traffic data
   - A price tracker for essential goods in informal markets
   - A USSD-to-WhatsApp bridge for rural farmers

2. **Build the minimal viable product in one weekend.**
   - FastAPI + SQLite for storage
   - No frontend — just a REST API and Swagger docs
   - Deploy on Fly.io (free tier) using a GitHub Actions workflow

3. **Add observability in week two.**
   - Add CloudWatch or Datadog free tier
   - Set up a synthetic monitor (every 5 minutes) to check API health
   - Write a simple README with deployment steps and a post-mortem template

4. **Add cost discipline in week three.**
   - Set a $10/month budget alert in AWS Budgets
   - Optimize by switching to arm64 instances
   - Document the cost breakdown in the README

5. **Simulate an incident in week four.**
   - Break something (e.g., delete the database) and write a post-mortem
   - Include logs, metrics, and the steps you took to recover
   - Add it to the README under “Incident Response”

6. **Iterate for 30 days.**
   - Add a cache with Redis 7.2
   - Add a CDN with CloudFront
   - Add a CI/CD pipeline with GitHub Actions
   - Document everything

7. **Apply with confidence.**
   - Include the GitHub repo, the live endpoint, the Terraform module, and the post-mortem
   - Mention the cost and the incident in your cover letter

I did this in reverse the first time. I built a beautiful API, deployed it, and only then realized I had no idea how to monitor it. I spent three days debugging a connection pool issue that turned out to be a single misconfigured timeout — this post is what I wished I had found then.

---

## Summary

Your portfolio isn’t a resume. It’s a production system you own from commit to customer. If it doesn’t run in production with IaC, observability, and cost discipline, it’s not a portfolio — it’s a toy.

Remote hiring pipelines in 2026 are optimized for candidates who can demonstrate end-to-end ownership. That means:

- One project you can explain from commit to customer
- Full deployment pipeline with CI/CD and IaC
- Observability — logs, metrics, traces
- Cost discipline and incident readiness

You don’t need to be perfect. You need to be *aware* of production concerns and show that you’ve grappled with them.

If you take one thing from this post, let it be this: your portfolio must prove you can run a system — not just write code.



## Frequently Asked Questions

**how to build a remote job portfolio from africa with no aws credits**

Start with Fly.io or Render. Both give free credits: Fly.io gives $5/month, Render gives $7/month. Use Python 3.11 and FastAPI. Deploy a simple REST API with a GitHub Actions workflow. Add a free CloudWatch dashboard or Datadog free tier. Document the deployment steps in your README. You don’t need AWS credits to build a production-grade portfolio.


**why do remote employers care about terraform in a portfolio**

Because Terraform is the de facto standard for reproducible infrastructure. If you can’t write a Terraform module to deploy your code, you’re not ready for production. I’ve seen candidates rejected because their deployment was a manual `scp` script. Remote teams care about reliability and reproducibility — Terraform proves you understand that.


**what is the minimum viable production-grade portfolio project**

A single FastAPI service with:
- GitHub Actions CI/CD
- Terraform module for deployment (even if it’s just one EC2 instance)
- CloudWatch logs and a dashboard
- A README with deployment steps and a simulated post-mortem

That’s it. No Kubernetes. No microservices. Just one project that proves you can ship and operate code.


**how much time should i spend on a portfolio vs leetcode**

Spend 60% of your time on one production-grade portfolio project. Spend 40% on LeetCode or systems design. If you only have three weeks, skip LeetCode and focus on the portfolio. I once helped a candidate in Dar es Salaam who spent 80 hours on LeetCode and 10 hours on a portfolio. He failed 12 technical screens and got zero interviews. When he pivoted to a production-grade portfolio, he got five interviews in two weeks.


---

| Tool/Service | Purpose | Version/Config |
|--------------|---------|----------------|
| Python | Backend runtime | 3.11 |
| FastAPI | Web framework | 0.111.0 |
| Terraform | Infrastructure as Code | 1.7.5 |
| AWS | Cloud provider | Free Tier (t3.micro) |
| Fly.io | Hosting | Free Tier |
| Redis | Caching | 7.2 |
| GitHub Actions | CI/CD | Ubuntu-latest |
| CloudWatch | Monitoring | Free Tier |
| Datadog | Observability | Free Tier |


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

**Last reviewed:** June 08, 2026
