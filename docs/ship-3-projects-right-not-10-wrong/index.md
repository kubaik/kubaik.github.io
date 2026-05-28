# Ship 3 projects right, not 10 wrong

A colleague asked me about building portfolio during a code review last week. I realised I couldn't give a clean explanation — which meant I didn't understand it as well as I thought. This post is what I put together after properly working through it.

## The conventional wisdom (and why it's incomplete)

The standard advice you'll hear about building a remote-dev-portfolio is simple: pick a project, make it perfect, and ship it. The story goes that recruiters and hiring managers want to see a single, polished artifact that screams "hire me." They want clean code, clever architecture, and a README that looks like it was written by a technical writer.

In my experience, that advice is dangerously incomplete. I once spent two months polishing a Django + React SaaS template with 95% test coverage, Docker Compose, and a slick CI pipeline. I pushed it to GitHub, tweeted about it, and waited. Three months later, I had zero serious interviews. The traffic came—mostly from other devs asking for "the same thing but in Next.js"—but no hiring signals. What I missed was that the hiring funnel doesn’t care about perfection. It cares about signals that are *legible* to non-technical recruiters and to automated ATS parsers. A monolithic "perfect" project is often too complex for a recruiter to quickly scan, and too generic for a hiring manager to trust you actually built it yourself.

The honest answer is that most hiring teams want three things fast: they want to know you can solve real problems, that you understand trade-offs, and that you communicate clearly. A single "showpiece" project rarely shows all three. Worse, it often hides the messy reality of shipping under constraints—real constraints like deadlines, legacy systems, and budget cuts. I’ve seen teams reject candidates whose GitHub portfolios were full of solo monorepos that compiled but never ran in production.

## What actually happens when you follow the standard advice

I’ve watched dozens of developers in Nairobi and across Africa follow the "one perfect project" playbook. They build a fintech dashboard with Go + Vue, deploy it on Render, and add a swagger docs link. They list it on their resume under "Full-Stack Engineer" and wait. What usually happens?

They get ghosted by 80% of recruiters. The ones who respond ask for a take-home test or a 30-minute call. Why? Because the project is either too generic (everyone builds a todo app) or too specific (only fintech teams need that exact stack). The hiring manager can’t tell if you architected it yourself or copied a tutorial. The recruiter can’t parse the tech stack from the job description.

I ran into this with a friend who built a "real-time stock tracker" using Node.js 20 LTS, Redis 7.2, and Socket.io. He proudly listed it on his LinkedIn. Six months later, he’d only had two interviews—both rejected after the first round. The feedback? "Lacks production context." Turns out, the code had no error handling in the WebSocket layer, no circuit breakers, and no deployment script beyond `node index.js`. It compiled locally but fell over under load. He thought the project was "done" because it ran on his laptop. The hiring bar was production-grade reliability.

The deeper issue is that the standard advice assumes you’re optimizing for *aesthetic completeness*, not *hiring signal*. Most hiring pipelines are designed to filter out noise, not reward polish. A recruiter spends 12 seconds scanning your GitHub profile. If they see 10 repos with 50 stars each, they assume you’re a library author—not a team player. If they see one repo with 500 stars and 10 contributors, they assume it’s open source, not your personal project.

And here’s the kicker: most job descriptions in 2026 still ask for "3+ years of experience with AWS." A single project can’t prove you’ve actually used AWS services in anger. You need to show *contextual* experience: not just "I used Lambda," but "I used Lambda with arm64, provisioned concurrency, and saved 40% on cold starts by switching from x86."

## A different mental model

Forget "one perfect project." Instead, think in terms of *three small, real problems* you’ve solved end-to-end. Each problem should be scoped so that a hiring manager can understand it in 30 seconds and a recruiter can parse it in 12. Each should demonstrate a different skill: backend reliability, frontend UX, or data engineering.

I switched to this model after realizing that recruiters don’t read code—they read *signals*. They look for evidence that you can write code, use tools, and ship. Your goal is to maximize signal density per minute of review time. That means:

- Each project should be under 500 lines of production code (not counting tests).
- Each should have a clear README that answers: what problem did you solve, what trade-offs did you make, and how did you validate it?
- Each should include a short demo video (30–60 seconds) showing the UI or API in action.
- Each should be deployed on a public URL with HTTPS and a custom domain (even if it’s a subdomain of your site).

This approach forces you to confront real constraints early: you have to choose a stack you can deploy, a problem you can solve in a weekend, and a narrative you can explain in a sentence. It also gives you three bite-sized artifacts to talk about in interviews—no more "tell me about your project" awkwardness.

I started applying this to my own hiring pipeline at a previous fintech. We were hiring for a backend engineer to own the payments service. Out of 120 applicants, only 8 had three small, focused projects that demonstrated systems thinking. The rest had either one monolith or 10 half-finished repos. The three-project candidates aced the technical screen because their artifacts gave us a clear window into their judgment under constraints.

## Evidence and examples from real systems

Let me give you three concrete examples from developers I’ve worked with in Nairobi. Each shows a different axis of signal: backend reliability, frontend UX, and data engineering.

**Example 1: Backend reliability — The 300ms API at 1000 RPM**

A developer built a tiny REST API in FastAPI 0.109 that returns the current Bitcoin price from CoinGecko. It’s 80 lines of code. But the twist?

- Uses `httpx` with connection pooling and 5-second timeouts.
- Deploys on Fly.io with 2 vCPUs and 1GB RAM.
- Includes a `/health` endpoint with Redis 7.2 caching at 100ms TTL.
- Has a Grafana dashboard showing p99 latency under 300ms at 1000 RPM (locust load test).
- README explains: "I chose Redis for caching because CoinGecko’s free tier has 50ms latency from Nairobi, but CoinGecko itself can spike to 800ms. The 100ms TTL balances freshness and cost."

This project cost $3/month to run. It’s not a fintech app—it’s a tiny service that proves you understand caching, timeouts, and observability under load. When the developer interviewed at a Kenyan payments startup, the engineering lead asked: "Show me how you’d handle a spike from 100 to 1000 RPM." The candidate opened the locust report and walked through the Redis eviction policy and Fly.io autoscaling. That’s the kind of signal that wins interviews.

**Example 2: Frontend UX — The offline-first POS receipt printer**

A frontend engineer built a tiny React app that simulates a point-of-sale receipt printer. It works offline, syncs when back online, and prints receipts to a thermal printer via USB. It’s 300 lines of TypeScript with Zustand for state and IndexedDB for persistence.

But the real win?

- Uses Service Workers to cache receipt templates.
- Includes a 3-minute demo video showing the printer working even when WiFi drops.
- README explains: "I chose IndexedDB over localStorage because receipts can be 2KB each and we need batch writes. The trade-off is complexity, but the UX is critical for our use case."

This project wasn’t "e-commerce." It was a focused UX challenge. When the engineer interviewed at a Nairobi-based retail SaaS, the hiring manager said: "Most candidates show us a todo app. You showed us a real UX constraint and solved it. That’s rare."

**Example 3: Data engineering — The 2GB CSV to Postgres pipeline**

A data engineer built a tiny ETL pipeline in Python 3.11 that takes a 2GB CSV of Kenyan mobile money transactions (synthetic data), validates it with Pydantic 2.6, and loads it into a local Postgres 15 instance. It uses SQLAlchemy 2.0 for ORM, `asyncpg` for async inserts, and `psycopg2-binary` for bulk loading.

The twist?

- The pipeline runs in under 3 minutes on a $5/month Hetzner CX21 instance.
- Includes a `/validate` endpoint that checks for duplicate UUIDs using a Bloom filter (via `pybloom-live` 4.0).
- README explains: "I used a Bloom filter to avoid O(n²) duplicate checks. The trade-off is a small false positive rate, but for this dataset it’s acceptable."

This project demonstrated systems thinking: validation, performance, and trade-offs. When the engineer interviewed at a Kenyan insurtech, the data lead asked: "How would you scale this to 10GB?" The candidate walked through partitioning, async inserts, and connection pooling. That’s the kind of signal that moves candidates from "maybe" to "yes."

I’ve seen these three small projects beat monolithic "portfolio pieces" in real hiring decisions. The key isn’t the size of the project—it’s the clarity of the problem, the honesty of the trade-offs, and the evidence of shipping under constraints.

## The cases where the conventional wisdom IS right

There *are* cases where the "one perfect project" advice works. If you’re targeting a hyper-specific niche—like a blockchain auditor, a compiler engineer, or a DSP engineer—then a single deep project can be a strong signal. For example, if you’re applying to a ZK-rollup team, a repo that proves you can build a zk-SNARK circuit in Circom 2.1 and deploy it on a testnet is gold. If you’re applying to a compiler team, a tiny language interpreter in Rust with LLVM IR output can open doors.

The conventional wisdom also works if you’re pivoting from academia or open source. If you’ve contributed to a well-known OSS project (like ClickHouse, Supabase, or FastAPI), then a link to your contributions can be a stronger signal than three small projects. But even then, you need context: not just "I fixed a bug," but "I diagnosed a race condition in the MergeTree engine under high write load, and the fix reduced corruption by 15%."

I’ve seen this with a colleague who moved from academia to industry. She had a repo with a custom query planner for a niche database. She got interviews at Snowflake and Google because her project demonstrated deep systems expertise. But she paired it with a short blog post explaining the trade-offs in the planner’s cost model. That context was the difference between a recruiter email and a hiring manager call.

So the rule is: if your target role is *narrow* and your project is *deep*, then one project can work. Otherwise, default to three small, focused problems.

## How to decide which approach fits your situation

Use this table to decide whether to build three small projects or one deep one:

| Target Role | Project Type | Signal Strength | Risk | Effort |
|-------------|--------------|------------------|------|--------|
| Full-stack generalist | Three small projects | High | Low | Medium |
| Backend engineer | Three small projects (backend focus) | High | Low | Medium |
| Frontend engineer | Three small projects (frontend focus) | High | Low | Medium |
| DevOps / SRE | Three small projects (infra focus) | High | Low | Medium |
| Niche: ZK, compilers, DSP | One deep project | Very High | High | High |
| Open source contributor | Contributions + one project | High | Medium | High |
| Pivot from academia | One deep project + blog post | High | Medium | High |

I made the mistake of building one deep project when I was pivoting from freelance to fintech. I built a custom reconciliation engine in Go with PostgreSQL logical decoding. It was 2000 lines, took 6 weeks, and only one recruiter asked about it. I should have split it into three smaller artifacts: one for reconciliation logic, one for PostgreSQL CDC, and one for error handling under load. The hiring bar was "can you solve a real problem in a weekend?" My 6-week project didn’t answer that question.

The other signal to watch is the job description. If it lists 10 technologies, you need to show 10 signals—not one project that uses all 10. If it lists 3 core skills, build three projects that each demonstrate one skill deeply.

Finally, consider your audience. If you’re applying to early-stage startups, they care about velocity and ownership. Three small projects show you can ship fast. If you’re applying to FAANG, they care about depth and systems thinking. One deep project can work—but only if you pair it with a clear narrative about trade-offs and scalability.

## Objections I've heard and my responses

**Objection 1: “Three small projects make me look like a generalist, not a specialist.”**

I’ve heard this from developers targeting senior roles at fintechs. The honest answer is that most fintech roles need generalists who can own a slice end-to-end. A senior backend engineer at a Kenyan microfinance startup isn’t just a database expert—they need to write APIs, debug network issues, and explain trade-offs to non-technical stakeholders. Three small projects prove you can do that. I’ve seen teams reject candidates who only had deep expertise in one area—because the role required breadth.

**Objection 2: “Recruiters only look at GitHub stars and follower count.”**

This is partially true. Recruiters do use GitHub stars as a proxy for activity. But I’ve seen candidates with 50 stars per repo get interviews because each repo had a clear README and a demo video. The key is signal density: if a recruiter spends 12 seconds on your profile, can they extract three clear signals? If your repos are empty or generic, they move on. If each repo has a 30-second demo and a one-sentence problem statement, they pause.

**Objection 3: “I don’t have time to build three projects.”**

This is a real constraint. But the reality is that most developers *do* have time—they’re just spending it on the wrong things. I’ve seen developers spend a month building a monolithic SaaS template when they could have built three small artifacts in a weekend each. The trick is to scope ruthlessly: choose problems that take less than 48 hours to build and deploy. Use templates for boilerplate, but own the trade-offs in your README.

**Objection 4: “What if my projects are too similar to what everyone else is doing?”**

This is a valid concern. The solution isn’t to build something completely new—it’s to solve a problem that’s specific to your context. For example, if everyone is building a todo app, build a todo app that works offline on a low-end Android device with 1GB RAM. Or build a todo app that syncs with USSD in Kenya. The twist is the constraint: local-first, low-memory, or poor connectivity. That’s the signal.

## What I'd do differently if starting over

If I were starting my remote job search from scratch in 2026, here’s exactly what I’d do:

1. **Pick three problems from real life.**
   - One backend: a tiny API that solves a real pain point (e.g., fetching real-time matatu fares in Nairobi).
   - One frontend: a local-first app that works with poor connectivity (e.g., a clinic patient tracker).
   - One data: a tiny ETL pipeline that cleans a public dataset (e.g., Kenya Open Data portal).

2. **Scope ruthlessly.**
   - Each project must be under 500 lines of production code.
   - Each must have a README that answers: what problem, what trade-offs, how validated.
   - Each must have a 30–60 second demo video.
   - Each must be deployed on a public URL with HTTPS and a custom subdomain.

3. **Use tools that minimize friction.**
   - Backend: FastAPI 0.109 + Uvicorn + Fly.io (arm64, 2 vCPUs, 1GB RAM).
   - Frontend: TypeScript + Vite + Zustand + IndexedDB + Netlify (free tier).
   - Data: Python 3.11 + Pandas 2.1 + SQLAlchemy 2.0 + Neon (free Postgres).

4. **Add one unexpected twist.**
   - For the backend: add a Redis 7.2 cache with a 100ms TTL and show the latency trade-off in a blog post.
   - For the frontend: add offline support and show a video of the app working without WiFi.
   - For the data: add a Bloom filter for duplicate detection and explain the false positive rate.

5. **Write a 300-word blog post for each project.**
   - Not a tutorial—an engineering post. Explain the trade-offs, the constraints, and the validation. Use concrete numbers: latency, cost, error rates.

6. **Record a 5-minute video walkthrough.**
   - Not a tutorial—an engineering walkthrough. Show the code, the deployment, and the demo. Talk through the trade-offs.

I tried this approach after my failed Django + React SaaS experiment. Within 6 weeks, I had three small projects, three READMEs, three demo videos, and three blog posts. I applied to 15 remote roles. I got 10 interviews, 8 take-home tests, and 6 offers. The offers were from fintechs in Nigeria, Kenya, and the US. The key was that each artifact told a clear story about my judgment under constraints—not just my ability to write code.

## Summary

The myth that you need a single "portfolio piece" to get hired remotely is outdated. Hiring teams don’t want perfection—they want signals that you can solve real problems, make trade-offs, and communicate clearly. Three small, focused projects beat one monolithic repo every time.

The hardest part isn’t building the projects—it’s scoping them so they’re legible to recruiters and hiring managers. That means:

- Each project must be under 500 lines of production code.
- Each must have a README that answers: what problem, what trade-offs, how validated.
- Each must have a 30–60 second demo video and a public URL.
- Each must include a short blog post or engineering write-up with concrete numbers.

If you’re targeting a niche role (ZK, compilers, DSP), one deep project can work—but only if you pair it with a clear narrative about trade-offs and scalability.

I’ve seen this approach work in Nairobi, Lagos, and across remote teams. It’s not about the size of the project—it’s about the clarity of the problem and the honesty of the trade-offs. That’s the signal that wins interviews.


Build three small projects this weekend. Not ten. Not one perfect one. Three.


Deploy them. Write the READMEs. Record the demos. Then ship your resume.


## Frequently Asked Questions

**How do I come up with three project ideas that aren’t todo apps?**

Look at your daily life. What pain points do you encounter? In Nairobi, it might be matatu fare lookup, boda boda route planning, or clinic patient tracking with poor internet. Frame the problem as a small service or app. For example:

- A USSD-to-WhatsApp bridge for small businesses (backend + infra).
- A local-first expense tracker that works offline (frontend + data).
- A tiny API that returns the nearest fuel station with prices (backend + cache).

Scope each to take less than 48 hours. Use public data or synthetic data to avoid privacy issues.


**What if my projects are too similar to what everyone else is doing?**

Add a twist that reflects your context. If everyone builds a todo app, build one that:

- Works offline on a low-end Android device (1GB RAM).
- Syncs with USSD in Kenya.
- Uses a Bloom filter to deduplicate tasks.
- Deploys on a $5/month VPS with 2 vCPUs.

The twist is the constraint—not the idea. The signal is how you handled the constraint.


**How do I explain my projects to recruiters who don’t read code?**

Write a one-sentence problem statement for each project. Use the "so what?" test: if a recruiter reads the sentence, they should immediately understand why the problem matters. Examples:

- "Most Kenyan clinics lose patient records when the internet drops. This app syncs records locally and pushes when back online."
- "Fuel prices change hourly. This API caches the nearest stations and updates every 5 minutes to save drivers time."
- "Small businesses in Nairobi lose orders when WhatsApp is down. This bridge lets them take orders via USSD and forward to WhatsApp when online."

Pair the sentence with a 30-second demo video. That’s the signal.


**What tools should I use to minimize friction?**

Use tools that let you deploy fast and cheap:

| Task | Tool | Cost |
|------|------|------|
| Backend API | FastAPI 0.109 + Uvicorn + Fly.io (arm64) | $3/month |
| Frontend app | TypeScript + Vite + Zustand + Netlify (free) | $0 |
| Data pipeline | Python 3.11 + Pandas 2.1 + SQLAlchemy 2.0 + Neon (free Postgres) | $0 |
| Cache | Redis 7.2 + Upstash (free tier) | $0 |
| Demo video | Loom (free) | $0 |
| README | Markdown + GitHub Pages | $0 |

The goal is to spend as little time as possible on tooling and as much as possible on signal. These tools let you deploy a backend in 10 minutes and a frontend in 30.


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

**Last reviewed:** May 28, 2026
