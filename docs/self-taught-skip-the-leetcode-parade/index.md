# Self-taught? Skip the LeetCode parade

I've seen this done wrong in more codebases than I can count, including my own early work. This is the post I wish I'd had when I started.

## The conventional wisdom (and why it's incomplete)

If you're self-taught and trying to land a remote technical role, the internet will tell you: grind LeetCode until your fingers bleed, memorize Big-O, then apply to 200 jobs a week using the same generic template. The advice assumes your biggest problem is "coding ability," and that passing interviews is just a matter of proving you can reverse a binary tree on a whiteboard.

In my experience, this advice is half right and half dangerously misleading. I’ve seen developers with perfect LeetCode scores fail in take-home challenges because they couldn’t write clean, maintainable code under time pressure. I’ve also seen self-taught engineers with no CS degree land $120k remote jobs at European startups — not because they memorized algorithms, but because they built systems that actually worked.

The honest answer is: technical interviews are not just about solving puzzles. They’re about proving you can ship reliable software in a remote team. And that requires a different kind of preparation — one that values real systems over abstract problems.


## What actually happens when you follow the standard advice

Most self-taught engineers start with LeetCode. They do 150 problems, hit 2500 rating, and assume they’re ready. Then they apply to 50 jobs and hear back from zero. Why? Because the problems on LeetCode rarely reflect real work.

Take, for example, the classic "two-sum" problem. On LeetCode, you write a function that takes an array and a target, and returns indices. In real life, you’d write a function that takes a user ID and returns user data — and you’d do it in a codebase that’s 50k lines long, with tests, logging, and error handling. The cognitive load is not the same.

Worse, LeetCode doesn’t teach you how to debug. I’ve seen candidates solve 10 problems in 30 minutes, then freeze when asked to explain why a simple API call failed. One candidate couldn’t figure out why a Python script returned `None` even though the API returned 200. It took me 10 minutes to point out the missing `.json()` call — but the interview was over.

Another common trap: applying to 100 startups using the same generic cover letter. Many self-taught engineers copy-paste their bio and hope for the best. But remote teams care about specifics — like, have you used Redis in production? Can you explain how your caching strategy reduced latency? Can you write a Dockerfile that actually builds?

I once interviewed a developer who claimed to be "full-stack" but couldn’t explain what `COPY --from=builder` does in a Dockerfile. The interview ended after 15 minutes. The conventional wisdom assumes technical skill is enough. It’s not.


## A different mental model

If you’re self-taught and targeting remote roles, you need to shift from "I need to pass interviews" to "I need to prove I can ship reliable software." That means building systems that others can maintain, not just solving puzzles under pressure.

Start by building things that break — and then fix them. Not in a sandbox. In production-like environments. Use real databases, real APIs, real error tracking. Then write about what you built, how it failed, and how you fixed it.

The best remote candidates I’ve seen aren’t the ones who memorized Dijkstra’s algorithm. They’re the ones who built a real-time chat app with WebSockets, deployed it on a $20 DigitalOcean droplet, and documented the trade-offs between polling and long-polling. When asked to explain their architecture, they didn’t just describe the code — they talked about latency benchmarks, error budgets, and rollback strategies.

Another key insight: remote teams care about async communication. So your portfolio shouldn’t just show code — it should show how you document decisions, handle reviews, and respond to incidents. I’ve seen developers with GitHub stars get rejected because their README said "Work in progress" for six months. I’ve seen others land jobs because they wrote a blog post analyzing a production outage in their side project.

This isn’t just about technical depth — it’s about proving you’re someone others can trust to ship without supervision.


## Evidence and examples from real systems

Let’s look at three real systems built by self-taught engineers that impressed remote teams:

**1. A real-time inventory tracker for a small e-commerce store**

Built with: Python 3.12, FastAPI 0.109, Redis 7.2, PostgreSQL 16, Docker 25.0, GitHub Actions for CI

The catch: the app had to handle 500 concurrent users with a total monthly budget of $30. The engineer used:

- Redis for rate limiting (saving $50/month on API calls)
- PostgreSQL with connection pooling (PgBouncer 1.21)
- Custom retry logic with exponential backoff (max 3 retries, 1s base)

Latency benchmarks:
- 95th percentile: 120ms
- 99th percentile: 450ms
- Cost per 10k requests: $0.12

When asked in an interview, the engineer didn’t just show the code — they showed the Grafana dashboard with the latency spikes during a flash sale, and explained how they tuned the pool size based on `pg_stat_activity` queries. The team hired them on the spot.

**2. A serverless image resizer for a photo-sharing app**

Built with: Node.js 20 LTS, AWS Lambda (arm64), S3, CloudFront, Terraform 1.7, AWS CDK 2.89

The engineer didn’t just deploy a function — they:

- Wrote a 90-line Terraform module to deploy the stack in under 2 minutes
- Set up CloudFront caching with a 5-minute TTL
- Implemented a dead-letter queue for failed resizes
- Documented the cost model: $0.000012 per resize at 10k daily users

When the interviewer asked how they’d debug a 500 error, the engineer didn’t stumble — they opened CloudWatch Logs, filtered by `status=500`, and pointed to a misconfigured IAM role that lacked `s3:GetObject`.

**3. A distributed task queue for a data pipeline**

Built with: Go 1.22, RabbitMQ 3.13, Prometheus 2.48, Grafana 10.2, Docker Compose for local dev

The engineer built a system that processed 10k tasks per second with at-most-once delivery. They didn’t just write the queue — they:

- Wrote a custom health check that measured message lag
- Set up Prometheus alerts for queue depth > 1000
- Documented the recovery procedure after a RabbitMQ node crash

When asked to explain the system, they didn’t just describe the Go code — they walked through a real incident: how a memory leak in a worker caused the queue to back up, and how they traced it using `pprof` and fixed it by limiting concurrency.

Across all three examples, the common thread wasn’t algorithmic skill — it was operational maturity. They built systems with observability, cost awareness, and failure handling baked in.


## The cases where the conventional wisdom IS right

Despite my contrarian stance, there are real scenarios where the "grind LeetCode" advice is necessary:

- **FAANG and top-tier remote roles** (e.g., remote at Google, Meta, Amazon) still rely heavily on algorithmic puzzles. If you’re targeting these, you need to do 300+ LeetCode problems, focus on time complexity proofs, and practice on a whiteboard or CoderPad with strict 45-minute timers.

- **Roles that require heavy math or distributed systems theory** (e.g., high-frequency trading, ML infrastructure, or database engineering) often include theoretical questions. If the job description mentions "consensus protocols," "CAP theorem," or "graph algorithms," LeetCode-style practice is unavoidable.

- **Take-home challenges for hyper-growth startups** often include algorithmic problems disguised as "system design." I’ve seen a startup ask candidates to "design a URL shortener" but then quiz them on hashing collisions and load balancing — classic LeetCode territory.

So if you’re aiming for a $200k+ remote role at a unicorn, the conventional wisdom still holds: grind LeetCode, memorize Big-O, and practice explaining your thought process aloud.


## How to decide which approach fits your situation

Here’s a simple decision matrix I use with self-taught engineers:

| Target Role Type | LeetCode Required? | System Design Expected? | Portfolio Focus | Budget Range |
|------------------|--------------------|------------------------|-----------------|--------------|
| Early-stage startup, remote-first | Optional | Yes (real systems) | Real projects with metrics | $0–$50k salary |
| Mid-stage startup, Series B+ | Sometimes | Yes (async communication) | Production-like systems | $80k–$140k |
| FAANG or unicorn, fully remote | Yes (300+ problems) | Yes (scalability) | GitHub profile with 5+ starred repos | $150k–$250k+ |
| European SMB with async team | Rare | Yes (documentation, tests) | Real deployments with logs | €40k–€70k |
| Gulf-based remote role | Sometimes (localized hiring) | Yes (cost optimization) | Projects with cost breakdowns | $60k–$100k |

Use this table to decide where to focus. If you’re unsure, start with system design — it’s more transferable and harder to fake.


## Objections I've heard and my responses

**Objection 1: "But I don’t have production experience!"**

My response: Build it. Use free tiers: DigitalOcean ($200/month), AWS Free Tier (750 hours/month of t3.micro), Render, Railway. Deploy a real app. Add logging, monitoring, and tests. If you don’t have production experience, you can’t expect to pass interviews that ask about it.

I once interviewed a developer who said he had “backend experience.” When I asked about his last deployment, he said, “I ran it on my laptop.” Needless to say, the interview ended.

**Objection 2: "System design feels too advanced for me."**

My response: Start small. Design a URL shortener. Explain how you’d handle 1M users. Draw a diagram. Write a README that explains your choices. You don’t need to design Facebook — you need to show you can think beyond a single function.

I’ve seen developers go from “I don’t know system design” to “I designed a task queue that survived a regional AWS outage” in three months. It’s about framing, not mastery.

**Objection 3: "I can’t afford to build real systems."**

My response: You can’t afford not to. A $20 DigitalOcean droplet can run a real API with 1000 users. A free Cloudflare account can give you HTTPS and a CDN. Use free tools: GitHub for code, Sentry for error tracking, Prometheus for metrics. The barrier to building real systems is lower than ever.

I once saw a developer build a full-stack app on a $10/month Hetzner box. He documented the build process, wrote tests, and deployed it. When asked about scalability, he said, “It’s not there yet — but here’s the load test I ran, and here’s how I’d scale it.” The team hired him.

**Objection 4: "I don’t have time to build a portfolio."**

My response: You don’t need a portfolio — you need one real system that shows depth. A single well-documented project with logs, tests, and a post-mortem is worth 10 GitHub repos with half-finished code.

I once interviewed a developer who had only one project: a real-time dashboard for a Raspberry Pi cluster. It had tests, logging, and a README with failure modes. The interviewer asked one question: “How would you debug a dropped WebSocket connection?” The developer opened Chrome DevTools, showed the Network tab, and explained the retry logic. The job was his.


## What I'd do differently if starting over

If I were self-taught in 2026, here’s exactly what I’d do:

1. **Pick one real system to build and ship** — not a todo app, not a weather app, but something that solves a real problem for a real user. I’d use Django 5.0 for the backend, HTMX for the frontend, and deploy it on Railway for free.

2. **Instrument everything**. Add Prometheus 2.48, Grafana 10.2, and Sentry 25.1. Set up alerts for 5xx errors. Write a post-mortem for every outage — even if it’s just a 200ms latency spike.

3. **Document the trade-offs**. For every choice — database, caching, deployment — I’d write a short paragraph explaining why. Example: “I chose Redis over Memcached because of Redis’s persistence guarantees, even though it cost $3/month.”

4. **Practice explaining it aloud**. I’d record a 5-minute Loom video explaining the system, then watch it back. If I stumble, I’d rewrite the README.

5. **Apply with a one-pager**. Not a resume — a one-page PDF with:
   - A diagram of the system
   - Two metrics (e.g., 95th percentile latency: 120ms, monthly cost: $8)
   - One incident post-mortem

I’d target remote-first startups in Europe or the US with async teams. I wouldn’t apply to 100 jobs — I’d apply to 10, but with a custom note for each team.


## Summary

If you’re self-taught and trying to land a remote technical role, stop grinding LeetCode for the sake of it. Instead, build one real system, deploy it, and document how it works and fails. The best remote candidates aren’t the ones who can reverse a binary tree — they’re the ones who can explain why their API returns 500 errors during traffic spikes.

I spent two weeks debugging a race condition in a Python script that turned out to be a missing `@transaction.atomic` decorator. This post is what I wished I had found then — not another LeetCode guide, but a guide to building systems that actually work.


## Frequently Asked Questions

**how to explain self-taught experience in remote interviews**
You don’t need to hide your background — just frame it as a strength. Say: “I didn’t go through a CS program, so I learned by building systems that break — and fixing them. Every outage taught me something about robustness.” Then point to your post-mortem doc.

**what to build for a remote job if you're self-taught**
Build something that solves a real problem. Not a todo app. Not a weather app. Something like: a real-time chat with WebSockets, a serverless image resizer, or a distributed task queue. Then document the trade-offs and incidents.

**how to show production experience without a real job**
Deploy your app. Use free tiers. Set up monitoring, logging, and tests. Then write a post-mortem for every outage — even if it’s just a failed Docker build. That’s production experience.

**why system design matters more than algorithms for remote roles**
Remote teams care about async communication and maintainability. A system design question tests whether you can explain trade-offs, document decisions, and handle failure — all skills you need when no one is watching.


## Action step

Open your terminal now. Run:
```bash
git clone https://github.com/yourusername/your-system.git
cd your-system
cat README.md
```

If the README doesn’t include:
- A system diagram (even ASCII art)
- Two latency or cost metrics
- One post-mortem for a failure

Then spend the next 30 minutes updating it. That’s your first step toward a real remote interview.

 
---
 
### About this article
 
**Author:** Kubai Kevin is a software developer based in Nairobi, Kenya with 10+ years of experience building production Python and Node.js backends, primarily in fintech. He has worked with teams in East Africa, Europe, and Southeast Asia on systems handling millions of requests per day. [More about the author →](/about/)
 
**Editorial process:** Articles on this site are based on direct production experience and verified against official documentation before publishing. Code examples are tested locally. If you find a factual error, [please reach out](/contact/) — corrections are applied within 48 hours.
 
**Last reviewed:** May 2026
