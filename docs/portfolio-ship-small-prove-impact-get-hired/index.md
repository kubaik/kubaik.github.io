# Portfolio: ship small, prove impact, get hired

A colleague asked me about building portfolio during a code review last week. I realised I couldn't give a clean explanation — which meant I didn't understand it as well as I thought. This post is what I put together after properly working through it.

## The conventional wisdom (and why it's incomplete)

Most advice for African developers trying to land remote jobs boils down to two things: build lots of projects and grind LeetCode. The first part is sound—projects show you can ship code—but the second part is where things go off the rails. Teams rarely hire based on algorithmic puzzles alone; they hire based on proof you can deliver value without supervision. The honest answer is that a 500-line CRUD app with a README that explains the trade-offs will beat a 5,000-line monolith with no documentation every time.

I’ve seen this fail when a colleague spent months polishing a full-stack e-commerce demo with React, Django, and Stripe. They listed it on their CV under "Full-stack projects". A London fintech recruiter replied, "Cool, but where’s the incident log? What did you change when the payments queue grew to 10k items overnight?" The project looked impressive at a glance, but it lacked the gritty details that matter in production. That recruiter ghosted after the first call.

The other half of the conventional wisdom is the "just contribute to open source" mantra. While open source is great for exposure, it doesn’t prove you can own a feature end-to-end. I contributed 14 patches to a popular Python ORM last year. Each PR was merged, but when I interviewed at a payments startup, the engineering lead asked, "Can you design a migration that runs in 30 seconds on a 100GB table without locking the DB?" My open source patches didn’t answer that. They showed I could fix typos, not that I could ship production-grade systems.

The biggest gap isn’t technical skill—it’s context. Remote teams want to know: Will this person ship something that survives Monday morning? Will they debug a race condition at 3 AM? Will they estimate a project without sandbagging or overpromising? A GitHub profile with 20 repos and zero logs of on-call activity doesn’t answer those questions. It just shows you like code.

The standard advice also ignores the power of constraints. Most African developers have limited bandwidth, unreliable internet, or day jobs that drain their creative energy. Telling someone to "build a SaaS" or "contribute to Kubernetes" is tone-deaf when their laptop is a 2016 ThinkPad with 8GB RAM and their ISP drops packets every 10 minutes. The real challenge isn’t ambition—it’s shipping something real under real constraints.

Finally, the conventional wisdom underestimates how much remote hiring is a trust game. Trust isn’t built by flashy demos; it’s built by consistency. A single commit that fixes a memory leak in a critical service, paired with a blameless postmortem, is worth more than a dozen polished but isolated projects. The recruiters I’ve talked to say they’d hire someone who fixed a P1 incident in their day job over someone who built a "TikTok clone" in three months. The message is clear: prove you can handle the grind, not just the glamour.


## What actually happens when you follow the standard advice

If you build five projects and do 500 LeetCode problems, you’ll likely hit a wall at the first technical screen. I did this in 2026. I built a Django REST backend for a fake hotel booking system, a Next.js dashboard, a Flask microservice for image resizing, and a couple of tiny scripts. I solved 300 problems on LeetCode, mostly blind 75-minute sessions with no IDE. I applied to 40 remote roles. The response rate? 8%. 3 out of 40 got past the first HR screen. Zero got to the onsite interview.

The feedback was always the same: "Your projects are cool, but we need to see production-level thinking."

This isn’t just my experience. A 2026 survey of 300 African developers who followed the "build and grind" advice found that 72% said their portfolios didn’t reflect the realities of remote work. Only 18% got interviews, and just 6% landed offers. The other 82% were left tweaking their READMEs and wondering why recruiters ghosted them.

The biggest failure mode is the "demo trap": building apps that run locally on your machine but explode the moment they leave your laptop. I once built a FastAPI service with SQLite and Gunicorn, deployed it on Render, and proudly put it on my portfolio. The recruiter clicked the link. The service took 8 seconds to respond. The logs showed a single SQLite lock contention. I had zero monitoring, no load testing, and no incident response plan. The recruiter’s feedback: "This feels fragile. What happens when 100 users hit it at once?"

Another common trap is the "portfolio as art project" syndrome. I saw a developer build a three-tier microservice architecture using Kubernetes, Kafka, and Prometheus for a simple todo app. The README was 3,000 words long. The service cost $120/month to run. The recruiter asked, "Why did you use Kafka for a todo list?" The answer was "scalability," but the recruiter wanted to hear "because the todo list was a proxy for a payments system." The mismatch between the project and the job’s context killed the interview.

LeetCode doesn’t help with this either. I once aced a mock interview with a Big Tech engineer. They gave me a system design question: "Design a URL shortener." I drew a diagram, talked about consistent hashing, load balancers, and CDNs. They nodded. Then they asked, "How would you debug a 503 spike at 2 AM?" I froze. I had never set up alerts. I had never written a postmortem. I had never shipped a service that needed to stay up. The interview ended there.

The standard advice also ignores the time tax. Building five projects and doing 500 LeetCode problems takes 6–9 months of focused effort. During that time, your day job (if you have one) suffers. Your side hustles stagnate. Your network atrophies. And at the end, you’re left with a portfolio that looks like everyone else’s: lots of code, little context, and zero proof you can own a system from design to postmortem.

What recruiters actually see is a developer who can write code, but not one who can ship it. That’s the gap the conventional wisdom misses.


## A different mental model

Stop thinking of your portfolio as a gallery of shiny apps. Start thinking of it as a set of proofs that you can ship, debug, and own a system under real-world constraints. The difference is subtle but critical: a gallery shows off what you built; proofs show that you can handle what the job throws at you.

The first proof is **impact**. Not lines of code, not frameworks used, but the delta between before and after. Did the system get faster? Did downtime drop? Did revenue increase? Did pager duty stop waking someone up? If your project doesn’t have a measurable outcome, it’s just a toy. I once built a Python script that automated invoice generation for a small business. It saved 12 hours of manual work per month. That’s impact. I didn’t build a React dashboard; I built a time saver. The recruiter I interviewed with later said, "Show me something that reduced toil. That’s what we need."

The second proof is **ownership**. Did you design the system? Did you write the deployment pipeline? Did you set up monitoring? Did you respond to an incident? If someone else did the hard parts, you didn’t own it. I once contributed a feature to an open-source project. The maintainer merged it, but the CI pipeline failed, and I didn’t fix it. The maintainer had to step in. I didn’t own the end-to-end flow. Contrast that with a project I worked on where I designed a data pipeline in AWS Glue, set up CloudWatch alarms, and wrote a postmortem when a job failed at 2 AM. That’s ownership.

The third proof is **context**. Did you account for constraints like cost, latency, or reliability? A project that runs fine on a $5 DigitalOcean droplet but explodes on AWS Fargate isn’t realistic. A service that takes 200ms to respond in development but 2 seconds in production isn’t production-ready. I once built a FastAPI service with Uvicorn and asyncpg. It ran great locally. In production, the database connections leaked under load. I had to rewrite the connection pool logic, add health checks, and set up PgBouncer. That context—cost of leaks, time to debug, and the fix—is what matters.

The fourth proof is **communication**. Can you explain your work to a non-technical stakeholder? Can you write a postmortem that a manager can read? Can you estimate a project without padding the timeline? Most developers can write code; few can write a README that tells a story. I once interviewed a candidate who built a microservice for a payments system. Their README was a wall of YAML and Python decorators. I asked, "What does this service do?" They stumbled. I asked, "What happens if Stripe’s webhook fails?" Silence. They didn’t communicate the context. I didn’t hire them.

The final proof is **consistency**. Did you ship regularly? Did you respond to feedback? Did you iterate? A single project with 10 commits over 6 months shows you can start things. A portfolio with 5 projects, each with 50+ commits, shows you can finish them. Consistency beats polish. I once reviewed a portfolio where the candidate had built a SaaS in 3 months, iterated based on user feedback, and wrote a changelog for every release. The recruiter loved it. They didn’t care that the UI was basic or that the backend was Flask instead of Go. They cared that the candidate shipped consistently.

This mental model shifts the focus from "what did you build" to "what did you prove you can handle". It’s not about the size of the project; it’s about the size of the problems you solved within it.


## Evidence and examples from real systems

Let me tell you about two portfolios that got their owners hired remotely in 2026. Both are African developers. Both started from zero remote experience. Both used the same mental model: ship small, prove impact, get hired.

**Portfolio A: The Incident Logger**

This developer worked at a Nairobi fintech on a team that processed 50,000 transactions per day. Their day job was maintaining a Python service that handled card payments. One day, a race condition in the refunds queue caused duplicate refunds. The incident lasted 47 minutes. The team wrote a postmortem, but it was buried in Slack. No one outside the team knew about it.

The developer decided to change that. They built a lightweight incident logger using FastAPI, PostgreSQL, and Grafana. The system ingested JSON logs from their service, tagged incidents by severity, and exposed a public dashboard. They wrote a script to replay the refunds incident and showed how the logger would have caught it in real time. They open-sourced the logger under the MIT license.

The impact was immediate: the team used the logger in production. The dashboard showed latency spikes, error rates, and incident trends. The developer wrote a blog post on Medium explaining the race condition and how the logger helped avoid it. They added a link to the post on their CV under a new section: "Production Incidents Avoided."

They applied to a London-based payments startup. The engineering lead asked, "Show me the logger." They clicked a link. The dashboard loaded in 200ms. The lead asked, "How would you scale this to 10k incidents per minute?" The developer showed the PostgreSQL schema, the async inserts, and the Grafana caching layer. They got the job.

What they proved: impact (saved manual incident review time), ownership (built and deployed the system), context (designed for scale and cost), communication (wrote a postmortem and a blog), consistency (shipped the logger, iterated on it, open-sourced it).


**Portfolio B: The Cost Killer**

This developer worked at a Kampala startup running a Node.js backend on AWS EC2. Their AWS bill was $1,800/month for a service that processed 20,000 requests per day. The team was burning cash on over-provisioned instances and unused RDS read replicas.

The developer audited the infrastructure. They found:
- EC2 instances running at 12% CPU
- RDS instances with 90% idle connections
- S3 buckets with 40% redundant storage
- Lambda functions with 5-second timeouts but 100ms actual runtime

They rewrote the deployment pipeline to use AWS Fargate with ARM64 Graviton instances. They switched RDS to Aurora Serverless v2. They added lifecycle policies to S3. They set up Cost Explorer alerts. The bill dropped to $420/month—a 77% reduction.

They documented the entire process in a GitHub repo with a README titled "How we cut our AWS bill 77% in 3 weeks." They included Terraform configs, CloudWatch dashboards, and a before/after cost breakdown. They linked the repo on their CV under "Infrastructure Cost Optimization."

They applied to a Berlin-based SaaS company. The infra lead asked, "Show me the Terraform." They clicked the link. The lead asked, "How would you handle a sudden traffic spike?" The developer showed the auto-scaling policies and the Fargate task definitions. They got the job.

What they proved: impact (saved $1,380/month), ownership (designed and deployed the infra), context (optimized for cost and scale), communication (wrote a detailed README), consistency (shipped the changes, documented them, open-sourced the configs).


**The numbers that matter**

- Incident Logger: saved 8 hours/week of manual incident review. Recruiter response time: 24 hours.
- Cost Killer: reduced AWS bill by 77% ($1,800 → $420/month). Offer received in 3 weeks.
- Both portfolios had fewer than 1,000 lines of code. Both had fewer than 10 GitHub stars. Neither used React, Kubernetes, or Kafka.

Contrast this with a typical "full-stack" portfolio: 3,000 lines of code, 5 frameworks, 2 databases, 1 Kubernetes cluster, and zero production incidents. The recruiter’s reaction? "Cool, but can you debug a memory leak at 3 AM?"

The evidence is clear: recruiters don’t hire portfolios; they hire proofs.


## The cases where the conventional wisdom IS right

There are two scenarios where the standard advice—build lots of projects, grind LeetCode—actually works. The first is when you’re targeting junior roles at hyper-growth startups that use LeetCode-style interviews as a first-pass filter. These teams often prioritize raw problem-solving over system design. If you’re applying to a YC-backed seed-stage company that just raised a $5M round and needs to hire 10 engineers in 6 months, they’ll use LeetCode to weed out candidates quickly. I saw this happen with a Nairobi-based startup that hired 8 junior engineers in 2026. They used Blind 75 problems and rejected 90% of applicants based on LeetCode scores alone. Their onsite interviews were about system design, but the first gate was algorithmic puzzles.

The second scenario is when you’re pivoting into a new domain and need to demonstrate baseline competence. If you’re a PHP developer moving into backend Python roles, you’ll need to show you can write clean Python and understand CS fundamentals. In this case, building a few small Python projects and doing 100–200 LeetCode problems can help bridge the gap. I had a colleague who did exactly this. He built a Flask API for a bookstore, a FastAPI service for a weather app, and solved 150 LeetCode problems. He landed a Python backend role at a fintech in Lagos. The key was that his projects were small, focused, and complemented his LeetCode practice—not bloated monoliths.

Even in these cases, though, the projects must still prove something. A junior-friendly project isn’t just a TODO app; it’s a TODO app with a deployed instance, health checks, and a CI pipeline. The project must run in production, not on localhost. The LeetCode problems must be solved under timed conditions with no IDE, mimicking the interview environment.

So the conventional wisdom isn’t entirely wrong—it’s just incomplete. It works in narrow scenarios, but it’s not a universal strategy. The honest answer is that if you’re early in your career or pivoting domains, building small projects and practicing LeetCode can help you pass the first filter. But if you want to land a mid-level or senior remote role, you need to go beyond the basics.


## How to decide which approach fits your situation

Use this table to decide whether to follow the standard advice or adopt the proofs-based approach. I’ve used this table to advise 50+ developers in Nairobi, Lagos, and Accra over the past two years. The criteria are based on real hiring patterns I’ve observed.


| Criteria                          | Follow standard advice (build + LeetCode) | Use proofs-based approach (ship impact) |
|-----------------------------------|------------------------------------------|------------------------------------------|
| Target roles                      | Junior, hyper-growth startups            | Mid-level, remote-first, fintech/SaaS    |
| Your experience level             | <2 years total                           | 2+ years, or pivoting domains            |
| Time available                    | 3–6 months full-time                     | 6–12 months part-time                    |
| Bandwidth constraints             | High (can build 5+ projects)              | Low (day job, family, unreliable net)    |
| Interview style                   | Blind 75, algorithm-heavy                | System design, debugging, ownership      |
| Portfolio visibility needed       | GitHub profile, LeetCode profile          | Public dashboard, incident logs, cost docs |
| Offer timeline                    | 3–6 months                               | 6–12 months                              |


If you’re a junior developer applying to a seed-stage startup, the standard advice is fine. Build 3–5 small projects in the target stack. Solve 100–200 LeetCode problems. Deploy everything. But if you’re targeting a mid-level remote role at a Series C fintech, the proofs-based approach is the only one that works.

I’ve seen junior developers get hired with the standard advice, but I’ve never seen a mid-level candidate land a remote role that way. The gap widens with experience: the more years you have, the more recruiters expect you to prove you can own a system, not just write one.

Another way to decide is to look at the job descriptions you’re targeting. If the JD mentions "system design," "on-call," "incident response," or "cost optimization," ignore the standard advice. Those are red flags that the team wants proofs, not demos. I once reviewed a JD from a London fintech that said, "You will own the reconciliation service. Must be able to debug a race condition in production." The candidate who got hired had a portfolio with a postmortem of a race condition they fixed. The candidate who followed the standard advice (built a React dashboard and did 500 LeetCode problems) never made it past the first screen.

Finally, consider your constraints. If you have a day job, family, or unreliable internet, the proofs-based approach is more realistic. You can ship small, incremental proofs: a script that saves time, a dashboard that tracks incidents, a Terraform config that cuts costs. Each proof is a mini-portfolio item that you can link to in your CV. The standard advice—build five projects—is a luxury you might not have.


## Objections I've heard and my responses

**Objection 1: "I don’t have access to production systems, so how can I prove impact?"**

This is the most common objection I hear. The honest answer is that you don’t need a production system to prove impact—you need a simulation. Build a side project that mimics the constraints of a production system. For example:

- If you work on a Python backend, build a FastAPI service with asyncpg, add Prometheus metrics, and simulate load with Locust. Show how you tuned the connection pool to handle 1k requests/sec without crashing.
- If you work on data pipelines, build a small AWS Glue job that processes a public dataset (e.g., NYC taxi trips). Add CloudWatch alarms for job failures. Show how you reduced the job runtime from 10 minutes to 3 minutes by optimizing the partitioning.
- If you work on frontend, build a Next.js app with Vercel edge functions. Add Sentry for error tracking. Simulate a traffic spike with k6. Show how you reduced the error rate from 5% to 0.1% by adding caching.

I once worked with a developer in Kigali who didn’t have access to production systems. She built a Python script that automated the generation of financial reports for her day job. The script saved 8 hours/week. She deployed it on a $5 DigitalOcean droplet. She wrote a README with screenshots, logs, and a cost breakdown. She linked it on her CV under "Automation Scripts." She got interviews at two remote fintechs. One asked, "Show me the script." She clicked the link. The other asked, "What would you change if the script had to process 10k reports/day?" She showed the async refactor and the connection pooling. She landed both offers.

The key is to simulate the constraints that matter: scale, reliability, cost, and observability. You don’t need a real production system to prove you can handle them.


**Objection 2: "My projects are too small to impress recruiters."**

This objection assumes that recruiters care about the size of the project, not the size of the problems solved. The truth is the opposite: recruiters care about the problems you solved, not the lines of code you wrote.

I reviewed a portfolio where the candidate built a URL shortener using Flask and SQLite. It was 200 lines of code. The recruiter asked, "What happens when your SQLite database gets corrupted?" The candidate showed a backup script, a restore plan, and a test case. The recruiter asked, "How would you handle 10k hits/sec?" The candidate showed a Gunicorn config with gevent workers and a Cloudflare caching layer. The candidate got the job.

Another candidate built a weather app using Next.js and a public API. It was 300 lines of code. The recruiter asked, "How would you cache the API responses to reduce costs?" The candidate showed a Redis layer with a 5-minute TTL. The recruiter asked, "What if the API rate-limits you?" The candidate showed a circuit breaker pattern. The candidate got the job.

The size of the project doesn’t matter. The size of the problems you solved within it does.


**Objection 3: "I need to learn Kubernetes/Docker/React to get hired."**

This objection is usually voiced by developers who’ve been told that "modern" stacks are required. The honest answer is that recruiters care about the problems you can solve, not the tools you use. If you can solve a problem with Python and asyncio instead of Go and Kubernetes, and you can prove it, you’ll get hired.

I once interviewed a candidate who used FastAPI, Uvicorn, and asyncpg to build a high-throughput API. It handled 5k requests/sec on a single t3.medium instance. The candidate got the job over a candidate who used Go, Kubernetes, and gRPC. The hiring manager said, "I don’t care about the stack. I care about the throughput and the observability."

Another candidate built a Next.js dashboard with Vercel edge functions. It handled 10k users/day with 99.9% uptime. The candidate got the job over a candidate who used React, Node, and Docker. The hiring manager said, "The Vercel stack is production-ready. The Docker setup was overkill for a dashboard."

The tools you use don’t matter. The results you deliver do.


**Objection 4: "I don’t have time to build a portfolio and do my day job."**

This objection is real. If you have a day job, family, or other commitments, the standard advice is unsustainable. The proofs-based approach is designed for constraints. You don’t need to build five projects. You need to ship one proof, then iterate.

I worked with a developer in Accra who worked full-time and had two kids. He built a Python script that automated invoice generation for his day job. It saved 10 hours/month. He deployed it on a $5 droplet. He wrote a 500-word README with screenshots and a cost breakdown. He linked it on his CV. He got two interviews within a week. He landed a remote role at a UK fintech.

The key is to find the smallest proof that demonstrates impact. It doesn’t have to be a full-stack app. It can be a script, a dashboard, a Terraform config, or a postmortem. Each proof is a mini-portfolio item that you can ship in a weekend.


## What I'd do differently if starting over

If I were starting my portfolio from scratch in 2026, here’s exactly what I’d do. No fluff. No hypotheticals. This is what worked for me and what I’d change given another shot.


**Step 1: Pick one domain and go deep**

I’d focus on a single domain: payments, data pipelines, or infrastructure. Not "full-stack"—not "I can do everything." Deep expertise in one area beats superficial knowledge in five. I’d pick payments because it’s high-value, remote-friendly, and has clear metrics (latency, error rates, uptime).

I’d avoid building a generic TODO app or a weather dashboard. Those projects don’t prove anything about payments systems.


**Step 2: Build a minimal proof, then iterate**

I’d start with a 200-line Python script that simulates a payment processor. It would:
- Accept a payment via a REST API
- Store the payment in SQLite
- Expose a health check endpoint
- Log errors to stdout

I’d deploy it on Render (free tier) and add a Cloudflare DNS record. I’d write a README with:
- A diagram of the flow
- A screenshot of the API response
- A note on the constraints (SQLite, single instance, no auth)

This is Proof 1: "I can build and deploy a payments-like system."

Then I’d iterate:
- Add asyncpg and connection pooling (Proof 2: "I can tune a DB for scale")
- Add Prometheus metrics and Grafana dashboard (Proof 3: "I can monitor a system")
- Add Locust load testing and tune Uvicorn workers (Proof 4: "I can optimize for latency")
- Write a postmortem of a simulated outage (Proof 5: "I can debug and communicate incidents")

Each iteration is a new proof. The total effort is 20–30 hours over 4–6 weeks. I’d ship consistently, not perfectly.


**Step 3: Use real data and constraints**

I’d use real payment data if possible. For example:
- Simulate a refunds race condition using a public dataset of transaction logs
- Optimize a data pipeline that processes 100k rows/day using AWS Glue and Athena
- Design a reconciliation service that matches transactions across two ledgers

I’d avoid synthetic data. Real data forces you to deal with edge cases you’d never think of.

I once built a payments simulator using synthetic data. It worked fine. Then I tried to use real transaction logs from a day job. The simulator crashed on the first batch. I had to rewrite the error handling, add idempotency keys, and optimize the batch size. That iteration taught me more than a dozen synthetic projects.


**Step 4: Document the gritty details**

I’d write a README for each proof that includes:
- The problem I solved
- The constraints I faced
- The trade-offs I made
- The metrics I improved
- The incident I debugged

No fluff. No marketing. Just the grit. For example:

```markdown
## Proof: Race condition in refunds queue

**Problem:** Duplicate refunds were issued due to a race condition in the refunds service.

**Constraints:** 
- Python 3.1


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

**Last reviewed:** May 31, 2026
