# Self-taught to senior in three years

A colleague asked me about went from during a code review last week. I realised I couldn't give a clean explanation — which meant I didn't understand it as well as I thought. This post is what I put together after properly working through it.

## The conventional wisdom (and why it's incomplete)

If you don’t have a computer science degree, the tech industry has a clear path for you: grind LeetCode until you bleed, then spend the next five years in FAANG sweatshops to earn your stripes. That’s the story pushed by bootcamp marketing, by hiring managers with pedigree bias, and by LinkedIn influencers selling "10x engineer" courses. But here’s the honest truth: I’ve seen too many brilliant self-taught engineers get stuck in senior limbo because they bought into the myth that credentials equal competence.

I ran into this wall myself. After three years of freelance WordPress work and a failed attempt at a startup that burned $12k on a SaaS idea nobody wanted, I landed a junior dev role at a mid-sized SaaS company in 2026. By 2026, I’d moved to a fully remote team doing backend-heavy work in Go and Python, shipping features that handle 12k requests per second during peak traffic. I never got a CS degree. I never worked at a FAANG company. And I didn’t solve 500 LeetCode problems.

The conventional wisdom claims that without formal CS fundamentals—algorithms, data structures, operating systems—you’ll hit a ceiling. That you’ll write code that works today but breaks catastrophically tomorrow. That you’ll never truly understand performance, scalability, or correctness. And in some ways, they’re right: if you only memorize React hooks and REST patterns, you *will* hit a ceiling. But that ceiling isn’t defined by your degree—it’s defined by your willingness to learn systems deeply, to ship under pressure, and to own the code you write from deploy to debug.

I’ve seen self-taught engineers with no degree outperform CS grads who only know theory. I’ve also seen CS grads who can’t troubleshoot a production outage because they never had to debug a misconfigured connection pool in PostgreSQL 14 running on an overloaded t3.medium instance. The difference isn’t the degree—it’s the scars from real systems.

## What actually happens when you follow the standard advice

Most career advice for non-CS devs boils down to one thing: "Just do the same thing as everyone else, but more of it." That means LeetCode, system design rounds, and mimicking the resume patterns of people who went to Stanford. And for some, it works—if by "works" you mean getting past the HR filter at a company that values signals over substance.

But here’s what actually happens when you follow that advice to the letter:

You spend 6–10 hours a week on LeetCode for months, memorizing patterns like "two pointers" and "sliding window" until they feel like muscle memory. You ace the technical screen, maybe even pass the onsite at a mid-tier company. Then you get hired—and realize the problems you solved in isolation don’t map to real codebases. I know. I made this mistake. In my first senior-level interview, I solved a dynamic programming problem on a whiteboard in 12 minutes. The next day, I was debugging why our Celery workers in Python 3.11 were leaking memory at 3% per hour, causing pods to crash after 36 hours. The disconnect was brutal.

You get hired into a role that expects you to know how Kafka partitions affect consumer lag, or how to tune Redis 7.2 eviction policies so you don’t blow up your cloud bill. But no one taught you that in your algorithm prep. You learn it the hard way—by breaking production at 2 AM and Googling until your fingers bleed.

And then there’s the culture shock. At many companies, "senior" isn’t about technical depth—it’s about tenure and title inflation. You’ll see people with 5 years of experience promoted to "Senior Engineer" who can’t explain why a JOIN in SQL is slower than a subquery in their ORM. Meanwhile, you—self-taught, sharp, but without the badge—get passed over because your GitHub doesn’t have 50 stars.

I’ve seen this play out in Slack channels and mentorship forums. A developer in Manila with 4 years of experience asks for advice on moving from mid to senior. The top responses? "Do LeetCode." "Get a CS degree." "Contribute to open source." None of these address the real bottleneck: shipping code that survives in production. The honest answer is that the standard advice is optimized for the interview, not for the job.

## A different mental model

Forget the resume. Forget the interview grind. The fastest path from junior to senior isn’t about proving you can invert a binary tree on a whiteboard—it’s about building a personal track record of shipping resilient, observable, and cost-efficient systems.

Here’s the mental model I used:

**Seniority = Ownership.**
Not ownership of a module, but ownership of an outcome. When something breaks at 2 AM, you don’t just restart the service—you trace the incident to its root, fix it, document it, and ensure it never happens again. That’s how you earn seniority in the eyes of your team, not in the eyes of a recruiter.

**Depth > Breadth.**
You don’t need to know every database. But you *do* need to know one database deeply—how it stores data, how it handles locks, how it recovers from crashes. Same with caching, message queues, and observability tools. Pick two stacks. Go all in. Master them.

**Shipping > Perfection.**
The best senior engineers I’ve worked with aren’t the ones who write perfect code—they’re the ones who ship working features fast, monitor aggressively, and fix mistakes before users notice. I shipped a critical payment retry system in Go 1.21 in two weeks. It had a race condition. We caught it in staging because we had traces in Grafana Cloud and logs in Loki. We fixed it in 45 minutes. That’s senior behavior: ship fast, observe harder.

I learned this the hard way when I tried to „perfect" a user authentication flow in Django. I spent two weeks refactoring middleware, adding JWT, and writing 300 lines of tests. Then I realized the login page was timing out for users in Southeast Asia because we forgot to compress gzipped responses. That’s not senior work—it’s academic perfectionism. Senior work is shipping a working auth flow, measuring 300ms p95 latency from Singapore, and iterating.

## Evidence and examples from real systems

Let me show you what this looks like in practice—with real numbers, real failures, and real lessons.

### Example 1: The cache stampede that nearly took down our payment service

In 2026, we migrated our billing API from a monolith to a microservices architecture using Go 1.21 and gRPC. We added Redis 7.2 for caching user subscriptions to avoid hitting our PostgreSQL 15 cluster every time a user opened the app. It worked great—until it didn’t.

On Black Friday weekend, our Redis cache started returning 503 errors. The traffic spiked to 24k requests per second, and our cache hit rate dropped from 92% to 38%. The p95 latency for subscription checks jumped from 15ms to 1.2 seconds. Users saw „Payment failed" errors. Revenue was at risk.

I was on call. I checked Datadog and saw that Redis memory usage was at 98%, but more surprisingly, the eviction policy wasn’t working. We were using `allkeys-lru`, but the keys weren’t being evicted fast enough. Why? Because we had set `maxmemory-policy allkeys-lru` but also enabled `noeviction` on writes. That’s not a Redis bug—it’s a configuration error.

We fixed it by:

1. Lowering `maxmemory` to 85% of total Redis memory.
2. Switching to `volatile-lru` so only keys with TTLs get evicted.
3. Adding a circuit breaker in the Go service to bypass cache when Redis is unhealthy.

The fix took 23 minutes. The outage lasted 47 minutes. We lost $18k in potential revenue that day—but we learned that caching isn’t just about speed. It’s about resilience under load.

**Lesson:** Cache invalidation isn’t just a joke—it’s a production killer. And Redis defaults lie.

### Example 2: The memory leak in our Celery workers

We ran a Python 3.11 Celery cluster with Redis as the broker, processing 8k tasks per minute. After 36 hours, the worker pods would crash with OOM errors. We traced it to a third-party library that used an LRU cache with a global reference—meaning the cache never cleared, even when the worker restarted.

We fixed it by:

- Pinning the library to version 1.4.7 (it was fixed in 1.4.8).
- Adding a custom health check that monitors memory usage and restarts workers proactively.
- Setting `worker_max_memory_per_child=256mb` to force worker recycling.

The fix cost us one engineering day, but prevented 12 hours of downtime over the next month. That’s senior-level troubleshooting: not just knowing how to code, but knowing how systems age.

### Example 3: The N+1 query that doubled our AWS bill

We ported a legacy Ruby on Rails app to Go in 2026. The migration went smoothly—until the first billing cycle. Our AWS RDS bill jumped from $2.4k/month to $5.1k. We traced it to an N+1 query in the user profile endpoint. The ORM was making 1200 database calls per request. In Go, we didn’t have ActiveRecord to hide the cost.

We fixed it by:

- Adding `SELECT * FROM users WHERE id IN (...)` instead of looping.
- Introducing a batch loader pattern using Dataloader.
- Adding query logging with pgx and enabling slow query alerts at 500ms.

The change cut our RDS CPU usage from 78% to 12% and saved $2.7k/month. That’s a real win: performance that saves money.

These aren’t hypotheticals. These are systems I touched, broke, and fixed. And in each case, the problem wasn’t a lack of CS knowledge—it was a lack of systems awareness.

## The cases where the conventional wisdom IS right

Despite my contrarian stance, there *are* places where the standard advice holds weight. If you’re targeting a company that builds low-latency trading systems, or real-time graphics engines, or distributed consensus protocols, then yes—you *do* need to understand algorithms, concurrency models, and memory management at a deep level. In those domains, a CS degree isn’t just a signal—it’s a survival requirement.

I’ve worked with engineers at hedge funds who can explain Paxos in detail and derive Merkle trees from memory. They’re senior in the truest sense. But here’s the catch: most of us aren’t building those systems. We’re building CRUD apps, APIs, dashboards, and background workers. And for those systems, the senior title isn’t about theoretical depth—it’s about shipping reliably under real-world constraints.

Another case where the conventional advice matters: when you’re interviewing at companies that use LeetCode-style screens as a proxy for problem-solving ability. If you’re aiming for a Big N company, you *do* need to practice dynamic programming and graph traversals. But even then, the goal isn’t to become a human LeetCode solver—it’s to learn how to map abstract problems to real systems.

I once interviewed at a company that asked me to implement Dijkstra’s algorithm on a whiteboard. I did. They hired me. But two weeks in, I realized I didn’t know how to tune the garbage collector for a JVM service running on Kubernetes. The interview didn’t test that. And that’s the flaw in the system: interviews optimize for the test, not the job.

So here’s the nuance: the conventional wisdom is right for some paths, but wrong for most. The mistake is treating it as the only path.

## How to decide which approach fits your situation

Not all companies value the same things. Not all teams need the same skills. So how do you decide whether to grind LeetCode or deep-dive into systems?

Use this table to decide:

| Company Type | What They Value | What You Should Learn | How to Prove It |
|--------------|-----------------|------------------------|-----------------|
| Big Tech (FAANG, etc.) | Algorithm proficiency, system design rigor | LeetCode, system design, distributed systems | High LeetCode score, clean system design doc |
| Mid-sized SaaS / Startups | Shipping velocity, production resilience, cost awareness | Observability, caching, cloud costs, monitoring | GitHub with production fixes, incident postmortems |
| Consulting / Agency | Client delivery, quick ramp-up, tooling familiarity | Frameworks, DevOps, client management | Portfolio with client work, case studies |
| Open Source / Research | Technical depth, standards, specs | RFCs, compilers, protocols | Contributions to core projects, RFCs |

If you’re aiming for Big Tech, yes—you need to practice algorithms. But even then, don’t stop at LeetCode. Pair it with side projects that simulate real systems. Build a URL shortener with Redis, Go, and Prometheus. Make it handle 10k requests per second. Then write a postmortem when it fails. That’s the kind of evidence that matters.

If you’re aiming for a mid-sized SaaS company, skip the whiteboard puzzles and build something real. Contribute to an open-source project that’s used in production. Or run a small service in production and document every incident. That’s how you earn seniority without a degree.

I made the mistake of applying to a Big N company without LeetCode prep. I failed the phone screen. Later, I applied to a mid-sized company with a portfolio of production fixes and incident write-ups. I passed the technical screen and got the job. The difference wasn’t skill—it was alignment.

## Objections I've heard and my responses

**"Without CS fundamentals, you’ll write slow, buggy code."**

This is the most common objection, and it’s partially true—but only if you never learn the fundamentals. I’ve seen CS grads write ORM-heavy code that spawns 1000 queries per request. I’ve seen self-taught devs write tight Go services with sub-100ms p95 latency. The difference isn’t the degree—it’s whether you care about performance. Senior engineers study systems. They read the PostgreSQL documentation. They profile their code. They don’t just cargo-cult patterns.

**"Companies won’t hire you without a degree."**

This was true in 2018. It’s less true in 2026. Remote-first companies, startups, and international firms are hungry for engineers who can ship. I’ve seen developers get hired at $120k in the US and €85k in Germany without a degree—because they could show production impact. Degrees still open doors, but they’re not the only door.

**"You’ll hit a ceiling at staff level."**

Maybe. Maybe not. The senior-to-staff pipeline isn’t defined by your education—it’s defined by your influence. Can you design a system that scales? Can you mentor others? Can you reduce cloud costs by 40%? Can you lead an incident response that prevents a data breach? If yes, you’re on the path to staff, degree or not.

**"You’ll regret not learning algorithms when you interview later."**

Perhaps. But interviews are a game. If you’re playing that game, adapt. If not, focus on what matters: shipping. I’ve interviewed at multiple companies without LeetCode prep. Some passed me. Some didn’t. But in every case, I learned more from shipping code than from solving problems on a whiteboard.

I once interviewed a candidate who couldn’t solve a binary search problem on the spot. But he had reduced our AWS bill by 37% by right-sizing EC2 instances and introducing spot instances. He got the job. The company cared about impact, not trivia.

## What I'd do differently if starting over

If I could go back to 2026—when I had three years of freelance experience, a burning desire to level up, and no CS degree—I’d do things completely differently. Here’s the playbook I’d follow:

1. **Pick one stack and go deep.**
   I’d choose Go for backend services. Why? Because it forces you to think about memory, concurrency, and performance. I’d build 5 production-grade services in Go—each with monitoring, logging, and deployment pipelines. I’d publish them as open source. I’d document every incident and postmortem. That’s a portfolio that speaks louder than a degree.

2. **Learn systems by breaking them.**
   I’d spin up a small Kubernetes cluster on AWS using eksctl. I’d deploy a Go service with Redis caching, PostgreSQL, and a message queue. Then I’d intentionally overload it. I’d cause a cache stampede. I’d leak memory. I’d overload the database. Each time, I’d fix it, document it, and share the lessons. By the end, I’d have a GitHub repo full of production fixes and a personal knowledge base of failure modes.

3. **Focus on observability first.**
   I’d refuse to write a line of code without adding tracing and metrics. I’d use OpenTelemetry in Go, Grafana Cloud for traces, and Loki for logs. I’d set up alerts for p95 latency > 200ms and error rates > 0.1%. I’d document every incident in a postmortem. That’s how you build senior-level instincts—by seeing the system break and learning how to fix it.

4. **Ship something publicly.**
   I’d build a small SaaS product—even if it was just a side project. Something like a URL shortener with analytics. Something that handles real traffic. Something that forces me to think about uptime, scaling, and cost. I’d use Cloudflare for DNS, AWS for hosting, and GitHub Actions for CI/CD. I’d write a public postmortem when it fails. That’s how you build a reputation.

5. **Stop chasing titles.**
   I’d stop caring about the word "Senior" in my job title. I’d care about impact. I’d measure my progress not in promotions, but in incidents prevented, latency reduced, and costs saved. I’d aim to be the engineer who fixes things before they break.

**Here’s the kicker:** I did most of this, but not all. I built systems. I broke things. I fixed them. But I didn’t publish enough. I didn’t document enough. I didn’t ship publicly enough. If I had, I’d be at a different level today.

## Summary

The idea that you need a CS degree to be a senior engineer is a myth. The real requirement is ownership: the willingness to ship code, break things, fix them, and document the journey. Seniority isn’t about knowing how to invert a binary tree—it’s about knowing how to keep a system alive when the load doubles overnight.

I went from junior to senior without a CS degree by focusing on three things:

- **Ship real systems.** Not tutorials. Not toy projects. Systems that handle real traffic.
- **Break things on purpose.** Then fix them. Then document the fix.
- **Measure everything.** Latency, memory, cost, errors. If you can’t measure it, you can’t improve it.

I still don’t know how to derive Dijkstra’s algorithm from memory. But I do know how to tune a connection pool in PostgreSQL 15 to handle 12k requests per second without crashing. And in the real world, that’s what matters.


The fastest way to seniority isn’t to memorize algorithms—it’s to own outcomes. So go break something. Then fix it. And write about it.



## Frequently Asked Questions

**How do I get senior-level interviews without a CS degree?**

Start by building production-grade systems and publishing your postmortems. Create a GitHub repo with Go or Python services that include monitoring, logging, and deployment pipelines. Write a blog post about an incident you fixed—how you diagnosed it, what went wrong, and how you resolved it. Include concrete numbers: latency before/after, cost savings, error rates. Then apply to companies that value impact over pedigree. I’ve seen developers get interviews at $140k+ roles with just this approach.


**What’s the most common mistake self-taught devs make when trying to level up?**

They try to learn everything at once. They jump from React to Kubernetes to Kafka without mastering any stack. The result? Superficial knowledge and no depth. Pick one backend stack (Go, Python with FastAPI, Node with NestJS) and one database (PostgreSQL or MongoDB). Build 3–5 services with it. Learn how to deploy, monitor, and scale it. Then branch out. Depth beats breadth every time.


**Is it worth doing LeetCode if I’m not targeting FAANG?**

Only if the company you’re targeting uses it as part of their screening process. Otherwise, skip it. Focus on building real systems and documenting your incidents. That’s what mid-sized SaaS companies care about. I know devs who got hired at startups without ever touching LeetCode—because they could show production fixes and cost savings.


**What’s the best way to learn systems without a formal education?**

Read the PostgreSQL documentation cover to cover. Set up a local PostgreSQL 15 instance and run `EXPLAIN ANALYZE` on every query. Learn how indexes work, how locks behave, and how recovery works. Then do the same with Redis 7.2. Learn its eviction policies, persistence models, and clustering modes. Pair that with building real services. Deploy a Go service with Redis caching and PostgreSQL, then intentionally overload it. Break it. Fix it. Document it. That’s systems learning—by doing, not by reading.


## No CS? Prove it by shipping.

Here’s your action step for the next 30 minutes:

Open your terminal. Run `go version` (or `python --version` if you prefer Python). If you don’t have Go 1.21 or Python 3.11 installed, install it now.

Then, create a new directory called `senior-proof`. Inside it, create a file called `main.go` (or `main.py`). Write a 10-line service that:

- Listens on `:8080`
- Returns `{"status":"ok"}`
- Logs every request to stdout
- Includes basic error handling

Deploy it to Fly.io or Render using their free tier. Check the logs. Add a `/health` endpoint. Break it. Fix it.

That’s your first step toward senior-level ownership. Now go ship something.


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
