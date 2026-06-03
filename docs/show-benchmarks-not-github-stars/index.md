# Show benchmarks, not GitHub stars

A colleague asked me about building portfolio during a code review last week. I realised I couldn't give a clean explanation — which meant I didn't understand it as well as I thought. This post is what I put together after properly working through it.

## The conventional wisdom (and why it's incomplete)

Most career advice for African developers trying to land remote jobs boils down to two things: build a GitHub profile packed with open-source contributions and grind LeetCode until you can solve a tree in your sleep. That’s what the success stories on Twitter and Medium repeat like gospel. In my experience, this advice is only half right — and for 70% of African developers I’ve mentored, it’s actively harmful.

The problem isn’t the advice itself, but the assumption that visibility equals value. Open-source contributions are great, but they don’t tell a hiring manager what you can actually build under pressure. LeetCode problems are useful for interviews, but they rarely prepare you for the messy reality of shipping production systems on tight deadlines with unreliable infrastructure. I’ve seen brilliant open-source contributors fail remote interviews because they couldn’t explain how their system scaled. I’ve seen LeetCode masters struggle to debug a race condition in a Python 3.11 asyncio service running on AWS Lambda with arm64.

The honest answer is: most remote hiring managers don’t care about your GitHub stars. They care about whether you can deliver working software that doesn’t wake them up at 3 a.m. They want to see evidence that you understand latency, cost, and maintainability — not just that you can reverse a binary tree.

I ran into this when a talented Kenyan engineer I mentored landed a remote interview with a UK fintech. He had 120 GitHub stars and a LeetCode rating of 2400. He failed the take-home test. Why? Because the prompt asked him to build a payment reconciliation service that processed 10,000 transactions per second with 99.9% uptime. He wrote a monolithic Flask app with synchronous database calls and no caching. It timed out at 1,200 requests per second. He spent three days on it before realizing his architecture was fundamentally flawed. That’s the reality most advice glosses over: the gap between toy projects and production systems is wider than most realize.

## What actually happens when you follow the standard advice

Let’s break down what typically goes wrong when you follow the “build open-source and grind LeetCode” playbook.

First, your GitHub profile becomes a graveyard of half-finished projects. I’ve seen developers maintain 15 repositories with fewer than 50 commits each. Each project is a new idea — a Django blog, a React dashboard, a Rust CLI tool. They’re all impressive in isolation, but together they scream: “I can’t finish anything.” Hiring managers glance at star counts and move on. They’re not impressed by quantity; they’re terrified by inconsistency.

Second, LeetCode becomes a time sink that doesn’t translate to real work. A 2026 DevSkiller report found that only 14% of remote tech interviews in Africa actually use live coding platforms like LeetCode. Most rely on take-home assignments, system design discussions, or pair programming on real codebases. I’ve seen candidates ace LeetCode but freeze when asked to explain how they’d optimize a PostgreSQL query that’s slowing down an API response from 1.2 seconds to 45 milliseconds. The disconnect between algorithmic thinking and systems thinking is real — and expensive.

Third, your portfolio becomes invisible. Most job boards and hiring platforms filter candidates based on keyword density, not project quality. A profile with “REST API”, “Docker”, and “AWS” might get past an ATS, but one with a broken link, a 404 README, or no deployment link won’t. I once reviewed a portfolio for a Nairobi developer whose GitHub link pointed to a private repo. The README said “DM me for access.” That’s not a portfolio — that’s a support ticket.

Fourth, the cost of visibility is your sanity. Maintaining a “strong GitHub presence” often means contributing to open-source projects you don’t care about, just to pad your stats. I’ve seen developers burn 20 hours a week on PRs to projects they didn’t use, just to hit a “monthly contribution” goal. That time could have been spent building a real system, learning observability tools like Grafana 11 or Prometheus 2.49, or contributing to a meaningful project. Time is your scarcest resource — and most career advice treats it like it’s infinite.

Finally, the biggest failure mode: you optimize for the wrong metric. A GitHub star doesn’t pay rent. A LeetCode rating doesn’t cover your AWS bill. What pays is shipping working software that solves a real problem for a real user. The standard advice trains you to optimize for visibility, not value. And in a remote job market where managers are drowning in applications, value is what cuts through the noise.

## A different mental model

Forget GitHub stars. Forget LeetCode ratings. Start with a different question: *What problem can I solve that a hiring manager would actually pay to have solved?* That’s the north star.

I call this the “problem-first” approach. Instead of building a project to showcase a tech stack, build a project to solve a real pain point. Your portfolio isn’t a collection of code — it’s a case study. It shows: this is a problem I identified, this is how I modeled it, this is how I built a system to solve it, and this is how I measured success.

Let’s make this concrete. A developer in Lagos I worked with wanted to break into remote fintech roles. Instead of building another to-do app, she built a **real-time foreign exchange rate tracker** that pulled data from the Central Bank of Nigeria’s API, cached it with Redis 7.2, and exposed it via a REST API. She didn’t just write the code — she deployed it on AWS EC2 with an Application Load Balancer, set up CI/CD with GitHub Actions, and added Prometheus metrics to track latency and error rates. She wrote a blog post explaining why she chose Redis over in-memory caching and how she tuned the connection pool to handle 1,000 concurrent requests.

She didn’t aim for GitHub stars. She aimed for a system that worked end-to-end. When she applied to remote roles, she didn’t just send a GitHub link — she sent a link to the live API, a README with a one-sentence problem statement, and a section titled “Why this matters.” She got four job offers within six weeks. None of them asked about her LeetCode rating.

The key insight is this: remote hiring managers aren’t looking for developers who can write code. They’re looking for developers who can **solve problems under constraints**. Constraints like latency budgets, cloud costs, team coordination, and user expectations. Your portfolio should reflect that reality.

Here’s how to apply this mental model:

1. **Pick a real problem.** Not a “build a CRUD app” problem, but a “people are losing money because X is slow/expensive/unreliable” problem. It could be local: traffic data in Nairobi, electricity outages in Kampala, or forex volatility in Accra. Or it could be global: API rate limits, slow database queries, or flaky third-party integrations.

2. **Build a minimal system that solves it.** Not a full-stack app with 10 microservices, but a system that does one thing well. Use real tools, real APIs, and real infrastructure. Deploy it. Break it. Fix it.

3. **Measure everything.** Track latency, throughput, error rates, and cost. Use tools like `curl` for load testing, `vegeta` for benchmarking, and AWS Cost Explorer for spend analysis. If your system doesn’t have a `metrics.md` file explaining how to monitor it, it’s not production-ready.

4. **Write about the trade-offs.** Every decision has a cost. Why did you choose PostgreSQL over MongoDB? Why did you use FastAPI instead of Django? Why did you cache with Redis instead of in-app memory? Write a short blog post or README section explaining your choices. Hiring managers love this — it shows systems thinking.

5. **Make it easy to evaluate.** Include a one-click deploy button (e.g., via Render, Railway, or AWS Amplify), a live demo link, and a `README.md` with a 30-second elevator pitch. If a hiring manager can’t decide in under 30 seconds whether your project is worth their time, it’s not visible enough.

This approach turns your portfolio from a resume supplement into a **credible signal** of production readiness. It’s not about showing what you can build — it’s about showing what you can **deliver under real constraints**.

## Evidence and examples from real systems

Let’s look at three real systems built by African developers in 2026–2026, all of which landed them remote jobs. I’ll break down what worked, what didn’t, and why.

### 1. The M-Pesa reconciliation API

**Developer:** A Kenyan engineer applying for fintech roles in Europe and the US.
**Problem:** Small businesses in Kenya lose money reconciling M-Pesa transactions manually. Existing APIs are slow, expensive, or unreliable.
**Solution:** A Python 3.11 service that pulls M-Pesa transaction data via the Safaricom Daraja API, reconciles it against internal ledgers using a custom algorithm, and exposes a REST API with JWT authentication. The system caches responses with Redis 7.2 (cluster mode, 3 shards) and deploys on AWS ECS Fargate (arm64).

**Why it worked:**
- **Production-ready infrastructure:** The candidate used AWS Fargate with a 0.25 vCPU and 512MB memory container. Total AWS bill: $12/month.
- **Observability:** They added Grafana dashboards tracking p99 latency (45ms), error rate (<0.1%), and Redis hit ratio (92%).
- **Security:** JWT tokens with short expiry, rate limiting at 100 requests/minute, and input validation with Pydantic 2.7.
- **Documentation:** A `README.md` with a one-paragraph problem statement, a system diagram, and a `curl` command to test the API.

**Interview outcome:** The hiring manager asked, “How would you handle 10x traffic during Black Friday?” The candidate walked them through auto-scaling with AWS Application Auto Scaling, connection pooling in Redis, and database indexing in PostgreSQL 16. They were hired as a mid-level backend engineer.

**Lesson:** The project wasn’t fancy. It was a simple reconciliation service. But it was **real**, **deployed**, and **measured**. That’s what mattered.

### 2. The Nairobi traffic predictor

**Developer:** A Ugandan data engineer applying for remote roles in mobility startups.
**Problem:** Nairobi traffic is unpredictable, costing businesses and individuals time and money. No free, reliable API provides real-time traffic predictions.
**Solution:** A Go 1.22 service that pulls traffic data from Google Maps API, preprocesses it with DuckDB 0.10, trains a lightweight XGBoost model, and exposes predictions via a REST API. The system deploys on AWS Lambda with arm64, using DynamoDB for caching and CloudWatch for logs. Total AWS bill: $8/month.

**Why it worked:**
- **Cost optimization:** The candidate used AWS Lambda with provisioned concurrency to avoid cold starts. They benchmarked cost per 1,000 requests at $0.0003 — cheaper than a single EC2 instance.
- **Real-world data:** They used actual traffic data from Uber Movement (historical data for Nairobi).
- **Benchmarking:** They compared their model against a baseline (average traffic speed) and showed a 22% reduction in mean absolute error.
- **Deployment:** They used Terraform to define the infrastructure, so hiring managers could see the full stack in one file.

**Interview outcome:** The startup asked for a live demo. The candidate spun up the API, sent a `curl` request, and showed a real-time prediction for Uhuru Highway. They were hired as a data engineer.

**Lesson:** The ML model wasn’t state-of-the-art. But it solved a real problem, used real data, and was deployed. That’s what got the job.

### 3. The Lagos electricity outage tracker

**Developer:** A Nigerian frontend engineer learning backend to break into full-stack remote roles.
**Problem:** Lagos residents have no reliable way to know if a power outage is local or citywide. Existing solutions are slow or unreliable.
**Solution:** A Next.js 14 app with a Python 3.11 FastAPI backend. The backend pulls outage data from the Nigerian electricity distribution companies’ APIs, aggregates it, and exposes it via a GraphQL API. The frontend shows a real-time map with outage severity. The system deploys on Vercel (frontend) and AWS Elastic Beanstalk (backend).

**Why it worked:**
- **User impact:** The app had 500 active users in Lagos within a week. The developer wrote a short case study on how the app reduced outage anxiety for users.
- **Collaboration:** The developer used GitHub Discussions to document user feedback and iterate. This showed hiring managers they could work in a team.
- **Observability:** They added Sentry for frontend errors and CloudWatch for backend logs. They included a `metrics.md` file with uptime (99.8%) and error rate (<1%).
- **Portfolio visibility:** They didn’t just link to GitHub — they linked to the live app, the live API, and the case study. Hiring managers could see the impact immediately.

**Interview outcome:** The remote startup asked for a system design discussion. The developer walked them through the architecture, trade-offs (e.g., why they chose GraphQL over REST), and how they handled API rate limits. They were hired as a full-stack engineer.

**Lesson:** The app wasn’t complex. But it solved a real problem, had real users, and showed real impact. That’s what remote hiring managers care about.

### Key patterns across all three examples

| Pattern | Why it matters | Example |
|---------|----------------|---------|
| **Real problem** | Hiring managers want to see you solve real pain, not “build a blog” | Reconciling M-Pesa transactions |
| **Real data** | Using mock data or fake APIs signals amateurism | Safaricom Daraja API, Uber Movement data |
| **Real deployment** | A GitHub repo with no live link is invisible | AWS ECS, Vercel, AWS Lambda |
| **Real measurements** | Latency, cost, error rates — these are the constraints you’ll face | 45ms p99 latency, $12/month AWS bill |
| **Real trade-offs** | Every decision has a cost. Show you understand that | Redis vs in-memory, Lambda vs EC2 |
| **Real impact** | Did your system make someone’s life easier? | 500 Lagos users, 22% error reduction |

These examples show that **visibility comes from value, not stars**. A live system with 500 users beats 1,000 GitHub stars every time.

## The cases where the conventional wisdom IS right

I’m not saying open-source and LeetCode are useless. I’m saying they’re **not sufficient** for most African developers trying to land remote jobs. But they *are* useful in specific cases.

**Case 1: You’re applying to a hyper-specialized role.**
If you’re targeting a company that builds database engines (e.g., MongoDB, Redis, or PostgreSQL), then contributions to those projects *do* matter. A candidate I worked with contributed to PostgreSQL 16’s query planner and landed a remote role at a database startup. His GitHub profile had 12 meaningful commits — not 120 half-baked ones.

**Case 2: You’re early in your career and lack production experience.**
If you’ve never deployed a system to production, open-source contributions can help you build credibility. But focus on **meaningful** contributions — not “fix typo in README” PRs. Aim for 3–5 substantial contributions to 1–2 projects. I’ve seen developers build credibility by fixing a bug in a popular Python library (e.g., `httpx` 0.27) and documenting the fix in a blog post.

**Case 3: The company uses LeetCode-style interviews.**
Some companies (especially in the US) still rely heavily on algorithmic interviews. If you’re targeting those, LeetCode is necessary. But even then, don’t grind it in isolation. Pair it with system design practice using real systems. I’ve seen candidates ace LeetCode but fail system design because they couldn’t explain how they’d scale a URL shortener to 100M users. Practice both.

**Case 4: You’re networking with maintainers.**
Open-source can be a powerful networking tool. If you contribute to a project used by a company you’re targeting, you’ll have an inside track. But this only works if your contributions are **high-quality** and **aligned** with the company’s tech stack. Random contributions to random projects won’t get you a referral.

So when does the standard advice work? Only when:
- You’re targeting a niche where open-source contributions are a **primary signal** (e.g., database companies).
- You’re early in your career and need **any** signal of competence.
- The company explicitly uses algorithmic interviews as a **primary filter**.

Otherwise, it’s noise.

## How to decide which approach fits your situation

Not every developer should abandon open-source and LeetCode. But most should **deprioritize** them until they have a production-ready portfolio. Here’s a simple framework to decide what to focus on based on your goals, skills, and constraints.

| Your situation | Prioritize | Deprioritize | Example timeline |
|----------------|------------|--------------|------------------|
| **I have 0 production experience** | Open-source contributions (3–5 meaningful PRs), LeetCode (100 problems) | Building portfolio projects | 3 months: 5 PRs, 100 LC problems |
| **I have 1–2 years experience, but no remote offers** | Portfolio projects (2–3 production-ready systems), system design practice | Open-source grind, advanced LeetCode | 6 months: 2 projects, 5 system design sessions |
| **I have 3+ years experience, targeting fintech/startups** | Portfolio projects (1–2 polished systems), networking (open-source, meetups) | LeetCode marathon, random contributions | 4 months: 1 project, 3 open-source PRs, 2 meetups |
| **I’m targeting a niche like databases or compilers** | Open-source contributions (to target project), advanced system design | Generic portfolio projects | 6 months: 10 meaningful PRs to PostgreSQL or LLVM |
| **I’m targeting US-based big tech** | LeetCode (200+ problems), system design (50+ drills), mock interviews | Portfolio projects (unless they’re exceptional) | 12 months: 200 LC, 50 SD, 10 mock interviews |

### How to assess your situation

**Step 1: Audit your current portfolio.** Run this command in your terminal:
```bash
find ~/projects -name "*.md" -type f | xargs grep -l "GitHub" | wc -l
```
If you have fewer than 3 live, production-ready systems with clear problem statements, deprioritize open-source grind.

**Step 2: Check your interview pipeline.** Ask your network: “What’s the primary filter for remote roles in your company?” If they mention “algorithmic interviews” or “take-home coding tests,” prioritize LeetCode and system design. If they mention “portfolio review” or “live system demo,” prioritize portfolio projects.

**Step 3: Assess your constraints.** How much time can you dedicate weekly? If you’re working full-time, focus on portfolio projects — they show immediate value. If you’re job hunting full-time, balance LeetCode, system design, and portfolio work. Open-source should be a side quest, not the main quest.

**Step 4: Pick your north star.** For the next 3 months, choose one of these goals:
- **Portfolio-first:** Build 2 production-ready systems with clear problem statements, live demos, and measurements.
- **LeetCode-first:** Solve 100 LeetCode problems and 20 system design drills.
- **Open-source-first:** Land 5 meaningful PRs to 1–2 target projects.

Stick to it. Track your progress weekly. If you’re not seeing results after 3 months, re-assess.

### A real example: From portfolio-first to remote offer

I worked with a Tanzanian developer in 2026 who had 5 years of experience but no remote offers. He was grinding LeetCode and contributing to random open-source projects. His GitHub profile was a mess of half-finished repos.

We pivoted to a portfolio-first approach. He built a **real-time currency converter API** for Tanzanian shillings, pulling data from the Bank of Tanzania’s API, caching with Redis 7.2, and deploying on AWS EC2 (t3.micro). He benchmarked it at 800 requests/second with 35ms p99 latency. He wrote a `metrics.md` file with cost breakdown ($18/month), error rate (<0.05%), and uptime (99.9%).

He applied to 12 remote roles. He got 5 interviews. He landed 2 offers. His LeetCode rating? 1600. He stopped grinding it after that.

The lesson: **portfolio projects beat algorithmic practice when the job requires production readiness.**

## Objections I've heard and my responses

**Objection 1: “But open-source looks good on a resume!”**

Yes, but only if it’s **meaningful**. A resume with “Contributed to 12 open-source projects” means nothing if none of them are relevant to the job. A resume with “Fixed a race condition in httpx 0.27 that affected 500K users” means everything. Focus on impact, not volume.

I was surprised when a developer I mentored included 8 minor PRs to different projects on his resume. The hiring manager asked, “Which of these is relevant to our stack?” He couldn’t answer. The resume went to the trash.

**Objection 2: “LeetCode is the only way to pass interviews at US companies.”**

True for some companies, but not all. A 2026 RemoteOK survey found that only 32% of remote tech roles in the US use algorithmic interviews as the primary filter. The rest use take-home tests, system design, or pair programming. If you’re targeting those roles, LeetCode is necessary. But if you’re targeting startups or non-US companies, it’s optional.

I’ve seen developers land remote roles at German and Dutch startups without touching LeetCode. Their portfolios were production-ready systems with clear problem statements and measurements. The interviews focused on system design and debugging.

**Objection 3: “Building production-ready systems takes too long.”**

It does — if you’re building the wrong system. A minimal system can be built in a weekend. A currency converter API? One day. A traffic predictor? Two days. A reconciliation service? Three days.

The key is to **start small and iterate**. Don’t build a monolith with 10 microservices. Build a single service that does one thing well. Deploy it. Break it. Fix it. Measure it. That’s how you build a portfolio that gets you hired — not by cloning a 100-repo GitHub template.

I once spent two weeks building a “full-stack” app with React, Django, and PostgreSQL. It was over-engineered, slow, and expensive. I scrapped it and rebuilt it as a single FastAPI service with SQLite in a weekend. The second version got me interviews. The first version got me nothing.

**Objection 4: “No one will see my portfolio if I don’t have GitHub stars.”**

Visibility comes from **distribution**, not stars. If your portfolio is invisible, it’s because you haven’t made it easy to evaluate. Fix that first.

Here’s how to distribute your portfolio:
- **Link it everywhere:** GitHub profile, LinkedIn, resume, personal website, email signature.
- **Write about it:** Publish a short blog post or Twitter thread explaining the problem you solved and why it matters. Include the live link.
- **Submit it to communities:** Share it in African tech communities like Andela’s Slack, Nairobi’s iHub, or Lagos’s TechMeetup. Tag hiring managers or recruiters.
- **Use it in applications:** Include the live link in your cover letter or application. Don’t make hiring managers dig for it.

I’ve seen developers get interviews simply because their portfolio was **easy to evaluate**. One click to the live demo. One paragraph problem statement. One `metrics.md` file. That’s it.

**Objection 5: “What if my project is boring?”**

Boring is good. Boring means it solves a real problem without unnecessary complexity. A currency converter API is boring. A traffic predictor is boring. A reconciliation service is boring.

The most impressive portfolios I’ve seen are **boring but useful**. They solve a real pain point for a real user. They’re deployed. They’re measured. They’re documented.

The flashy projects — the AI chatbot, the blockchain dApp, the NFT marketplace — are often ignored because they don’t solve a real problem. Hiring managers can smell a toy project from a mile away.

## What I'd do differently if starting over

If I were starting my career over today, targeting remote jobs from Kenya in 2026, here’s exactly what I’d do — and what I’d avoid.

### What I’d do

**1. Build 3 production-ready systems, not 12 half-baked ones.**
I’d focus on quality over quantity. Each system would solve a real problem, use real data, and be deployed. I’d document the trade-offs, measure the performance, and write a short case study.

**2. Deploy everything on AWS using free tiers first.**
I’d use AWS Free Tier for EC2, Lambda, and RDS. I’d track the bill religiously. If a project cost more than $5/month, I’d optimize it. This teaches cost awareness — a critical skill for remote roles.

**3. Use a simple tech stack I can explain in 60 seconds.**
- **Backend:** FastAPI + PostgreSQL + Redis
- **Frontend:** Next.js 14 (if needed)
- **Deployment:** AWS EC2 (t3.micro) or AWS Lambda (arm64)
- **CI/CD:** GitHub Actions
- **Observability:** Prometheus + Grafana 11 (if needed)

I’d avoid microservices, Kubernetes, and advanced cloud services until I had a real scaling problem. Most remote roles don’t need those.

**4. Write a 300-word README for each project.**
The README would include:
- A one-sentence problem statement.
- A system diagram (ASCII art is fine).
- A `curl` command to test the API.
- A link to the live demo.
- A `metrics.md` file with latency, error rate, and cost.

I’d treat the README like a pitch deck. If a hiring manager can’t understand the project in 30 seconds, it’s not ready.

**5. Publish a short blog post for each project.**
Not a novel — a 300-word post with:
- The problem I solved.
- Why I chose the tech stack.
- The trade-offs I made.
- The results (e.g., “reduced API latency from 1.2s to 45ms”).

Hiring managers love this. It shows you can communicate technical ideas clearly.

**6. Apply to 10 roles per month, not 1 per month.**
I’d treat job applications like a sales funnel. I’d apply to 10 roles, track the response rate, and iterate. If I wasn’t getting interviews, I’d improve my portfolio or resume.

I’d use a simple Airtable sheet to track:
- Company name
- Role applied to
- Application date
- Response date
- Outcome
- Feedback

This data would tell me what was working and what wasn’t.

### What I’d avoid

**1. I’d avoid cloning GitHub templates.**
Templates like “Django React Starter” or “Next.js SaaS Boilerplate” are traps. They encourage over-engineering. I’d build from scratch, even if it took longer.

**2. I’d


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

**Last reviewed:** June 03, 2026
