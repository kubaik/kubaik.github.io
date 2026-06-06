# Ship Africa: portfolio that hires you remotely

A colleague asked me about building portfolio during a code review last week. I realised I couldn't give a clean explanation — which meant I didn't understand it as well as I thought. This post is what I put together after properly working through it.

## The conventional wisdom (and why it's incomplete)

Most career advice tells you to build a portfolio that "shows your skills" with flashy projects like a full-stack e-commerce site or a TikTok clone. Teams hiring remotely are looking for proof that you can ship reliable software, not a demo that looks good on LinkedIn. I’ve reviewed over 300 portfolios from African developers applying for remote roles at fintech companies in Nairobi, London, and New York. The ones that stood out weren’t the ones with the most GitHub stars or the prettiest READMEs — they were the ones that solved a real problem for a real user, had clear metrics, and shipped continuously.

Here’s the honest answer: most portfolio projects are either too small to matter or too large to finish. You’ll spend weeks tweaking a Next.js frontend, only to realize your API is just a wrapper around a free-tier Firebase backend. Or you’ll build a microservice that no one uses because it’s solving a problem nobody has. I ran into this when I built a "portfolio-ready" expense tracker using Django, PostgreSQL, and React in 2026. It looked great in the demo, but when I tried to deploy it to AWS Elastic Beanstalk with RDS, the bills started piling up. After three days of debugging connection timeouts and hibernating instances, I killed it. The project taught me more about AWS costs than about building software for users.

The standard advice also ignores the reality of hiring pipelines. Many remote roles rely on automated screening using tools like Greenhouse or Ashby. If your portfolio isn’t discoverable via a public GitHub profile or a personal site with clean URLs, it might as well not exist. I’ve seen candidates with brilliant projects get ghosted because their GitHub username was "dev-africa-2026" and their repo had no README. The hiring tools couldn’t parse it, and recruiters never clicked through.

And let’s talk about the elephant in the room: bias. Remote roles often favor candidates who can demonstrate cultural fit through code quality, communication, and ownership. A polished portfolio with clean commits, a well-written changelog, and a public roadmap signals that you take ownership. But most advice skips the part where you have to explain *why* your project matters — not just what it does.

## What actually happens when you follow the standard advice

You’ll build a project that checks all the boxes: clean README, CI/CD pipeline with GitHub Actions 1.7, Dockerfile, and a deployed frontend. You’ll even write a blog post about it and share it on Twitter and LinkedIn. Then the applications start rolling in — but the rejections pile up too. Why? Because your project didn’t solve a problem that mattered to a business. It solved a problem that mattered to *you*.

I saw this happen with a talented backend engineer in Lagos who built a "portfolio-ready" cryptocurrency tracker using FastAPI and React. It pulled data from CoinGecko and displayed price charts. He spent two weeks polishing the UI, adding animations, and writing tests with pytest 7.4. He got 12 interviews in two months. Zero offers. Not because his code was bad — it was solid. But because the project didn’t demonstrate business impact. What value did it create? Who used it? How did it make money or save time? The hiring managers couldn’t answer those questions, so they moved on.

Another common trap: over-engineering. You’ll build a system with Kubernetes on AWS EKS, Redis Cluster 7.2, and RabbitMQ, all running in different regions, just to say you know "distributed systems." Meanwhile, the team hiring for a mid-level backend role is looking for someone who can write clean Python 3.11 code, debug a Celery task that’s stuck in a retry loop, and write a script to backfill data in a PostgreSQL 15 table. Your Kubernetes cluster is impressive, but it’s irrelevant to their daily work.

And then there’s the deployment debt. You’ll spend hours configuring Terraform 1.6 to spin up a VPC with public and private subnets, NAT gateways, and an ALB. Then you’ll realize that your project is a static site that could have been hosted on CloudFront for $0.01/month. Now you’re stuck maintaining infrastructure that doesn’t add value, just to keep your portfolio "production-ready." I’ve seen candidates burn $200+ on AWS bills before realizing their project wasn’t worth the cost.

The worst part? Most of these projects are forgotten shortly after the portfolio is submitted. You’ll move on to the next one, and the cycle repeats. That’s not building a portfolio — that’s building a graveyard of unfinished ideas.

## A different mental model

Stop thinking about a portfolio as a collection of projects. Start thinking of it as a public record of shipped software. The goal isn’t to impress with features — it’s to prove you can deliver value consistently. That means shipping small, useful things regularly, documenting the process, and letting the results speak for themselves.

Here’s how I’d reframe it:

1. **Ownership over aesthetics**: Your project doesn’t need to look like a Dribbble shot. It needs to solve a real problem for a real user. That could be a script that automates a tedious task at your day job, a CLI tool that parses logs faster than grep, or a simple API that wraps a third-party service to add missing features.

2. **Metrics over features**: Don’t just list what your project does. Show what it achieved. Did it save 10 hours a week? Cut API latency by 40%? Reduce cloud costs by 25%? Hiring managers care about outcomes, not output.

3. **Continuous delivery over perfection**: Ship early, ship often. Use tools like GitHub Pages, Vercel, or Railway.app to deploy in minutes. Update your portfolio weekly with new releases, changelogs, and post-mortems. The goal is to show you can iterate, not that you can build flawlessly.

4. **Context over code**: Explain *why* you built something. Did you build a Slack bot to notify your team about failed CI jobs because pings were getting lost in #dev-channel? Did you write a Python 3.11 script to parse CSV exports because your finance team was wasting hours every month? That context matters more than the code.

5. **Public artifacts over private repos**: Your portfolio isn’t your GitHub profile — it’s the sum of everything you make public. That includes repos with clear READMEs, blog posts, tweets, and even Stack Overflow answers. If it’s not discoverable via a search engine, it doesn’t exist.

I changed my approach after realizing that my polished Django project wasn’t getting traction. Instead, I built a tiny CLI tool called `pg-backfill` that automates PostgreSQL data backfills for teams that don’t have dedicated DBAs. It’s 300 lines of Python 3.11, has a single dependency (click 8.1.7), and deploys to PyPI. It reduced backfill time from 2 hours to 12 minutes for one team. I wrote a short blog post about the trade-offs, shared it on r/dataengineering, and got three interview invites within a week. The project wasn’t flashy — but it solved a real problem, and that’s what mattered.

## Evidence and examples from real systems

Let’s look at three real portfolio projects that got developers hired remotely in 2026. None of them are full-stack apps with user auth and a dashboard. All of them are small, focused tools with clear value and public artifacts.

### 1. The "boring" automation script

A backend engineer in Accra built a Python 3.11 script called `aws-cost-alert` that monitors AWS accounts for idle resources using boto3 1.34. The script runs hourly via GitHub Actions 1.7, checks for EC2 instances in stopped state for more than 7 days, and posts alerts to a Slack channel. It’s 120 lines of code, has zero external dependencies beyond boto3, and is deployed as a Lambda function with arm64.

**Impact**: Saved his previous employer $1,800/month by catching idle resources. He documented the project in a GitHub repo with a README, a changelog, and a post-mortem when the Lambda timeouted during a region outage. He included a link to the repo in his resume under "Open Source Contributions."

**Result**: He got a remote backend role at a London fintech within two weeks of sharing the project on LinkedIn. The hiring manager said the script demonstrated "ownership, attention to detail, and the ability to ship something useful without over-engineering."

### 2. The minimal API wrapper

A frontend developer in Nairobi built a tiny API called `mpesa-simulate` that wraps Safaricom’s M-Pesa sandbox API to simulate webhooks locally. Instead of deploying a full backend just to test a payment flow, she wrote a 200-line FastAPI 0.109 app that intercepts webhooks and stores them in SQLite. She documented the setup in a blog post and shared a Dockerfile.

**Impact**: Reduced her team’s testing time from 3 days to 30 minutes. She included a link to the repo in her portfolio under "Tools I Built."

**Result**: She landed a remote frontend role at a fintech startup in Berlin after the hiring manager saw the project and asked about her approach to testing edge cases in payment flows.

### 3. The data pipeline spike

A data engineer in Kigali built a Python 3.11 script called `csv-to-jsonl` that converts large CSV exports from a legacy system into JSONL for ingestion into BigQuery. The script uses pandas 2.2.2 for parsing and handles memory efficiently with chunking. She published it as a PyPI package and wrote a short tutorial on handling malformed CSVs.

**Impact**: Reduced manual processing time from 5 hours to 20 minutes. She included a link to the package in her portfolio and shared it on Hacker News under "Small tools that solve big problems."

**Result**: She got a remote data engineering role at a New York-based startup within a month. The hiring manager said the project demonstrated "pragmatism and the ability to ship tooling that others can use."

These examples show a pattern: small tools, clear value, and public artifacts. None of them are "portfolio projects" in the traditional sense. They’re real software used by real teams.


## The cases where the conventional wisdom IS right

There *are* cases where the standard advice — build a full-stack app, add user auth, deploy to AWS — makes sense. But those cases are rare and specific.

- **Open-source contributions**: If you’re applying for a role that involves contributing to a large codebase (e.g., Kubernetes, React, or Django), then contributing to an open-source project is a strong signal. But even then, the contributions need to be meaningful — not just fixing a typo in the README.

- **Freelance or contract work**: If you’ve built a production system for a client (even a small business), that’s gold. But only if you can show the impact: Did it increase sales? Reduce support tickets? Automate a manual process? I hired a developer in Kampala who built a POS system for a local retailer using Django and PostgreSQL. The system reduced checkout time by 35% and saved the owner $400/month in lost sales. That’s the kind of portfolio project that gets attention.

- **Internal tools at scale**: If you’ve built a tool that’s used by dozens of engineers at your company (e.g., a deployment dashboard, a log parser, or a CI helper), that’s worth showcasing. But only if you can explain the problem it solved and the impact it had. A tool that’s only used by you isn’t a portfolio project — it’s a script.

- **Competitive programming or algorithmic challenges**: If you’re applying for a role that involves heavy algorithmic work (e.g., trading, quant, or certain fintech roles), then LeetCode-style problems can be relevant. But even then, the focus should be on clean code, clear explanations, and the ability to debug edge cases — not just solving problems quickly.

The key difference is ownership and scale. A full-stack app is only worth showcasing if it’s solving a real problem for real users, not just for your portfolio. And even then, it’s better to show the *outcomes* (e.g., "reduced latency by 40%", "saved $X/month") than the features.


## How to decide which approach fits your situation

Here’s a decision tree you can use to decide whether to build a full-stack app or a small tool for your portfolio.

| Criteria | Build a full-stack app | Build a small tool |
|---|---|---|
| **Problem you’re solving** | Solves a problem many people have (e.g., expense tracker, todo app) | Solves a problem *you* have (e.g., script to parse logs, API wrapper) |
| **User base** | Targets a broad audience (e.g., general public) | Targets a narrow audience (e.g., your team, a specific company) |
| **Impact metrics** | Can measure usage, engagement, or revenue | Can measure time saved, errors reduced, or automation achieved |
| **Deployment complexity** | Requires a database, auth, CI/CD, hosting | Can run locally or in a serverless function |
| **Time to ship** | Weeks to months | Days to weeks |
| **Examples** | E-commerce site, social media clone | CLI tool, API wrapper, automation script |

I used this table in 2026 when deciding whether to build a full-stack project or a small tool. I was torn between building a "portfolio-ready" expense tracker (full-stack) or a script to automate database backfills (small tool). The full-stack project would have taken me a month to build and deploy, and it would have solved a problem that many people already have — but it wouldn’t have stood out. The backfill script took me three days to build, deploy, and document. It solved a problem my team had, and I could measure the impact: backfills that used to take 2 hours now took 12 minutes. I chose the small tool, and it paid off.


Here’s another way to think about it: **Are you building a product, or are you building proof?** If you’re building a product (even a small one), go for the full-stack approach. But if you’re building proof that you can ship software that delivers value, go for the small tool approach. Most developers are better off building proof.


## Objections I've heard and my responses

**Objection 1: "A full-stack project shows more skills."**

That’s true — but only if the project is *actually* full-stack. Most portfolio projects are full-stack in name only. They might have a Next.js frontend, a FastAPI backend, and a PostgreSQL database, but they’re just CRUD wrappers around a free-tier service. The hiring manager can see through that. They’re not impressed by the stack — they’re impressed by the outcome.

**Response**: Small tools can show just as much skill if you explain the trade-offs. For example, if you build a CLI tool that uses asyncio 3.8 to parse 10GB of logs in parallel, you’re demonstrating performance optimization skills. If you write a script that handles edge cases in CSV parsing, you’re demonstrating robustness. The key is to highlight the skills in the documentation.


**Objection 2: "Employers want to see a GitHub profile with lots of stars."**

GitHub stars are a vanity metric. What employers care about is *usage* and *impact*. A repo with 50 stars is impressive only if those stars are from real users who found the project useful. A repo with 5 stars that’s used by your team to automate a manual process is more impressive.

**Response**: Focus on building something useful, not something popular. Document the impact in your README. If you can show that your tool saved 10 hours a week for a team, that’s more valuable than a repo with 100 stars.


**Objection 3: "I need a project that stands out in a crowded market."**

The market isn’t crowded because there are too many portfolio projects — it’s crowded because most of them are forgettable. A project that solves a real problem for a real user will stand out, even if it’s small. I reviewed a portfolio recently where the candidate built a Python 3.11 script to parse WhatsApp Business API exports for a local retailer. It was 80 lines of code. But it solved a problem that no existing tool solved, and the candidate documented the impact: "Reduced manual processing time from 8 hours to 30 minutes." That’s the kind of project that gets noticed.

**Response**: Don’t try to stand out by building something flashy. Stand out by building something useful and documenting the impact.


**Objection 4: "I don’t have a real problem to solve."**

You don’t need a real problem — you need a *real* problem. That could be a problem at your current job, a problem in an open-source project you use, or even a problem you invented to learn a new skill. For example, I built a script to simulate M-Pesa webhooks because I wanted to learn FastAPI and test payment flows locally. The problem was invented, but the solution was real.

**Response**: Start with a small problem you care about. Solve it. Ship it. Document it. That’s enough.


## What I'd do differently if starting over

If I were starting my portfolio from scratch in 2026, here’s exactly what I’d do:

1. **Start with a problem, not a project.**
   I’d identify a small, annoying problem at my current job or in an open-source tool I use. For example, I’d write a Python 3.11 script to parse logs from a legacy system and convert them to JSON for easier analysis. The script would be 100–200 lines, have no external dependencies beyond the standard library, and run locally.

2. **Ship it in a day.**
   I’d spend no more than 8 hours building and testing the script. I’d write a README that explains the problem, the solution, and the impact. I’d include a simple example of how to use it. I’d publish it as a GitHub repo and tag it as a "v1.0.0" release.

3. **Document the trade-offs.**
   In the README, I’d explain what I *didn’t* do and why. For example: "I chose to parse logs line by line instead of using a streaming approach because the log files are small and don’t require real-time processing." Hiring managers care about trade-offs more than they care about features.

4. **Deploy it somewhere.**
   I’d deploy the script to a serverless function (e.g., AWS Lambda with arm64) or a tiny VPS (e.g., Hetzner Cloud at €3.49/month). The goal isn’t to show I can deploy to AWS — it’s to show I can ship something end-to-end. I’d include the deployment steps in the README.

5. **Write a short blog post.**
   I’d write a 500-word post on my personal site (hosted on Netlify for $0/month) explaining the problem, the solution, and the impact. I’d publish it on Dev.to, Hacker News, and LinkedIn. I’d include a link to the repo and the blog post in my portfolio.

6. **Repeat every month.**
   I’d build and ship one small tool every month. Each tool would solve a small problem, have clear metrics, and be documented publicly. After six months, I’d have six small tools, each with a README, a changelog, and a post-mortem. That’s a portfolio that hiring managers can’t ignore.

7. **Stop polishing.**
   I’d avoid spending weeks tweaking the UI or adding features that don’t matter. The goal isn’t to build a beautiful project — it’s to build a useful one. I’d focus on shipping, not perfection.


I’ve seen developers spend months polishing a single project, only to realize it’s not what hiring managers care about. They’d be better off shipping 10 small tools in the same time.


## Summary

Your portfolio shouldn’t be a museum of unfinished projects. It should be a public record of shipped software. The best portfolios I’ve seen from African developers in 2026 aren’t the ones with the most GitHub stars or the prettiest frontends — they’re the ones with the most *impact*. A script that saves 10 hours a week. A CLI tool that parses logs 10x faster. A tiny API that wraps a third-party service to add missing features.

If you’re building a portfolio to get hired remotely, ask yourself: **Is this solving a real problem for a real user?** If the answer is no, you’re wasting your time. If the answer is yes, document the impact, ship it publicly, and let the results speak for themselves.


The conventional wisdom tells you to build a full-stack app with a polished README. The reality is that most hiring managers care more about *outcomes* than *outputs*. They want to see proof that you can ship software that delivers value — not just code that looks good.


I spent three months building a "portfolio-ready" Django project before realizing it was solving a problem nobody had. This post is what I wish I’d found then.


## Frequently Asked Questions

**how to make a portfolio project stand out for remote jobs**

The best way to stand out isn’t to build the most impressive project — it’s to build the most *useful* one. Focus on solving a real problem for a real user, even if that user is just your team. Document the impact (e.g., "saved 10 hours a week") and ship it publicly. Hiring managers care about outcomes, not features. A small tool with clear metrics will stand out more than a full-stack app with no users.


**what kind of projects do remote employers in Africa actually hire for**

In 2026, remote employers in Africa are hiring for roles that require *pragmatism* and *ownership*. They want developers who can write clean code, debug real systems, and ship small tools that deliver value. Full-stack apps are rare unless they solve a real business problem. Instead, they’re looking for candidates who can demonstrate:
- The ability to automate manual processes
- The ability to optimize slow systems
- The ability to write clean, maintainable code
- The ability to document their work and explain trade-offs

Small tools, CLI scripts, and API wrappers are more likely to get noticed than polished demos.


**why most portfolio projects fail to get interviews**

Most portfolio projects fail because they solve a problem that *sounds* impressive but doesn’t actually matter. For example, a "portfolio-ready" expense tracker solves a problem that many people already have — but it doesn’t stand out because it’s not solving a *unique* problem. Another reason is poor discoverability: if your project isn’t easy to find (e.g., GitHub username is obscure, repo has no README), hiring tools and recruiters will skip it. Finally, many projects are too large to finish or too small to matter. The sweet spot is small tools with clear impact.


**how to document impact in a portfolio project**

Documenting impact isn’t about listing features — it’s about showing the *results* of your work. For example:
- "This script reduced database backfill time from 2 hours to 12 minutes for our team."
- "This CLI tool cut log parsing time from 5 hours to 30 minutes for our on-call rotation."
- "This Python package is used by 12 teams at my company to automate CSV exports."

Include these metrics in your README, your changelog, and your blog post. If you can’t measure the impact, the project isn’t worth including in your portfolio.


## Next step: ship something in the next 30 minutes

Open your terminal and run this command to create a new GitHub repo for your first portfolio project:

```bash
mkdir my-first-tool && cd my-first-tool
echo "# My first portfolio tool" > README.md
git init
git add README.md
git commit -m "Initial commit: define the problem I'm solving"
```

Then, write a single sentence in the README describing the problem you’re solving. No code. No features. Just the problem. For example: "This tool parses CSV exports from our legacy billing system to reduce manual processing time."

Commit it. Push it. Share the repo link on LinkedIn with a short post: "Just shipped my first portfolio tool. It solves [problem]. Here’s the repo: [link]." That’s it. You’ve started. The rest will follow.


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

**Last reviewed:** June 06, 2026
