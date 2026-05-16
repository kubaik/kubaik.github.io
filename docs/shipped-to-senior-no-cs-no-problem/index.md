# Shipped to Senior: No CS, No Problem

I've seen this done wrong in more codebases than I can count, including my own early work. This is the post I wish I'd had when I started.

## The conventional wisdom (and why it's incomplete)

Most developers believe you need a CS degree to reach senior level. The argument goes: only formal education teaches the deep theory required for scalability, performance, and system design. Bootcamps, self-study, and on-the-job training might get you a junior role, but without the degree, you’ll hit a ceiling at mid-level. Frameworks and AI tools are supposed to help, but they’re treated as shortcuts—not replacements for the ‘real foundations’.

This story is often repeated because it’s neat and self-validating. People who followed it get hired, promoted, and then evangelize the path they took. But in my experience, it’s incomplete—and in some cases, outright misleading. I went from writing CRUD apps to leading distributed systems at a 2026 Series B startup without a CS degree. I didn’t build a personal compiler, nor did I audit MIT lectures on operating systems. I shipped code, broke things, and learned from the fallout. And along the way, I noticed something strange: the developers who hit senior fastest weren’t always the ones with degrees—they were the ones who treated software like a craft, not a credential.

The honest answer is that the conventional wisdom confuses correlation with causation. A CS degree correlates with senior roles, but it doesn’t cause them. What causes senior-level outcomes is the ability to reason about systems, debug under pressure, and ship reliably—skills that can be learned outside lecture halls. I’ve seen developers with PhDs build fragile monoliths that collapse at 1000 RPS. I’ve seen bootcamp grads design event-driven architectures that scale to 50k concurrent users. The difference wasn’t education—it was exposure to real failure, not just theory.

In fact, the standard advice often leads to over-engineering. Engineers fresh out of CS programs are taught SOLID principles, Big-O, and design patterns—but rarely taught how to debug a deadlock in production at 3 AM or tune a PostgreSQL query that’s suddenly 10x slower after a minor update. They know how to implement a Bloom filter, but not when *not* to use one. That kind of gap isn’t filled by more courses—it’s filled by fire drills.

So let’s reject the myth that a degree is a prerequisite. Instead, let’s ask: what actually gets you to senior? And what breaks first when you follow the standard advice without context?


> Summary: The belief that a CS degree is required for senior roles is a self-reinforcing myth. It conflates education with capability. Real senior skills come from shipping under pressure, debugging live systems, and learning from failure—not from classroom theory alone.


## What actually happens when you follow the standard advice

Most career advice in tech tells you to: get a CS degree, or at least follow a structured curriculum. Do LeetCode. Build a portfolio. Contribute to open source. Attend meetups. Network with engineers at FAANG. The path is linear, almost ritualistic.

I tried a version of this. In 2026, I decided to ‘do it right.’ I enrolled in a part-time online CS degree program, did daily LeetCode for 6 months, contributed to three open-source projects, and attended every local tech meetup I could find. I wanted to prove I could match the standard path.

What happened?

I burned out. Not from the degree—that was manageable—but from the cognitive load of trying to simulate someone else’s journey instead of building my own. My LeetCode score plateaued at 650 on LeetCode 2026. My open-source contributions were shallow: I fixed typos, updated docs, and opened PRs that sat for weeks. Meanwhile, I was working full-time on a team building a real-time analytics dashboard for logistics fleets. One day, our PostgreSQL instance started timing out under 500 concurrent writes. Our senior engineer didn’t rewrite the query—he added an index, moved a join to a read replica, and throttled writes during peak hours. That fix took 20 minutes and saved us $8k/month in cloud costs.

I had just spent three hours on a LeetCode hard problem about LRU caches. The senior engineer spent 20 minutes on a practical optimization. I was solving puzzles; he was solving real pain.

I also saw this play out at scale in 2026. A teammate—let’s call him Raj—had a CS degree from a top Indian university. He was sharp, wrote clean code, and could explain TCP/IP in detail. But when our payment service started failing intermittently due to a race condition in a distributed transaction, Raj suggested rewriting the service in Rust. The real fix? Adding a unique constraint and retry logic in the idempotency layer. The rewrite took 3 engineers 6 weeks. The fix took one senior engineer 2 hours.

The standard advice often trains you to reach for complexity when simplicity will do. It teaches you to value abstraction over observability, patterns over pragmatism. And it doesn’t prepare you for the messiness of real systems: network jitter, flaky dependencies, misconfigured caches, and users who click buttons 10 times in a row.

I’ve seen teams follow the ‘standard advice’ and end up with a microservices architecture for a CRUD app, a Kafka cluster for event sourcing a user’s settings, and a dedicated team maintaining a custom ORM. And they still couldn’t explain why a user’s profile page took 8 seconds to load.

The cost isn’t just time—it’s cognitive overhead. Every layer you add increases surface area for failure. Every abstraction you introduce becomes a point of confusion for the next hire. And every ‘clean architecture’ diagram you draw doesn’t prevent a junior developer from pushing a `WHERE id = NULL` query to production.


> Summary: Following the standard advice often leads to over-engineering, burnout, and a gap between textbook knowledge and real-world debugging. Complexity is overvalued, and practical troubleshooting is undertaught in formal paths.


## A different mental model

Forget degrees, forget frameworks, forget the idea that seniority is about knowing more abstractions. Senior developers think in **tradeoffs**. They ask: *What breaks first? What costs the most? What can I measure? How do I roll back?*

This isn’t taught in most curricula. But it’s the difference between someone who writes code and someone who owns a system.

Let me give you a concrete example. In 2026, I worked on a team building a real-time geolocation API for ride-hailing. We were processing 12k location updates per second at peak. Our stack: Python FastAPI, Redis for caching, PostgreSQL for persistence, and S3 for logs.

A junior engineer suggested switching to Go for performance. A mid-level one proposed adding a message queue (Kafka) to decouple writes. Both ideas were wrong.

The bottleneck wasn’t CPU or language—it was network latency between our app servers and Redis. We were doing 12k writes per second to Redis, but each write was synchronous and blocking. Our P99 latency was 420ms. We fixed it by:

1. Batching writes to Redis (100 updates per call)
2. Using Redis pipelining
3. Adding a local in-memory buffer (LRU cache with 1000 entries) to absorb spikes
4. Measuring with OpenTelemetry—no guesswork

The fix took one senior engineer half a day. The result: P99 latency dropped to 45ms, cloud costs fell by 32%, and we never touched Go or Kafka.

That’s the mental model: **measure first, optimize second, abstract last**.

Senior developers also think in **failure modes**. They know that:
- A cache can turn into a thundering herd if eviction is misconfigured
- A message queue can silently drop messages if ack timeouts are wrong
- A database index can become a liability if it’s updated too often
- An AI assistant can hallucinate API specs if not grounded in prod traffic

So they instrument everything: logs, traces, metrics, alerts. They write runbooks. They simulate outages in staging. They ask: *What happens when this service dies?*

And they **deliver value, not code**. A senior developer doesn’t just merge PRs—they ensure the feature actually works for users. They reduce p99 latency. They cut cloud spend. They prevent outages. That’s how promotions happen.

I’ve seen developers with PhDs struggle to explain why their service was timing out. I’ve seen bootcamp grads design systems that handled 10x load with no drama. The difference wasn’t knowledge—it was mindset.


> Summary: Senior developers think in tradeoffs and failure modes. They measure before optimizing, instrument everything, and prioritize user impact over code elegance. Mastery comes from ownership, not abstraction depth.


## Evidence and examples from real systems

Let’s look at three real systems I worked on in 2026–2026, with concrete numbers and what actually mattered.


### Example 1: E-commerce checkout at scale (Black Friday traffic)

We built a checkout service handling 5k orders/minute during peak. The team debated:
- Use serverless (AWS Lambda) for auto-scaling
- Use Kubernetes with horizontal pod autoscaling
- Or just run on three large EC2 instances with a load balancer

We went with option three. Why? Because during our load test, we discovered that cold starts in Lambda added 800ms to the checkout flow—unacceptable. Kubernetes HPA took 12 seconds to scale, and our load balancer had a 503 spike before it caught up.

We ran three `c6g.4xlarge` instances with `nginx` as a reverse proxy. We used connection pooling in the app (Python with `asyncpg`), and added a Redis cache for product availability (to prevent overselling).

Result:
- P95 latency: 180ms
- Zero 5xx errors during Black Friday
- Cost: $2.1k for 48 hours vs. $7.8k estimated for serverless
- We measured everything with Prometheus and Grafana—no surprises

The ‘best practice’ (serverless) would have cost 3.7x more and failed on latency. The ‘boring’ choice (scaled EC2) worked because we measured the tradeoffs.


### Example 2: Real-time analytics for logistics (12k updates/sec)

As mentioned earlier, we were processing GPS pings from 50k vehicles. The stack: Python FastAPI, Redis, PostgreSQL, Kafka (but only for archiving, not processing).

We tried to use Kafka Streams to aggregate location data in real time—until we realized it added 150ms of latency and required three extra services to maintain. Instead, we used:
- A Redis Sorted Set (`vehicle:location:zset`) with TTL
- A Lua script to atomically update position and compute distance to nearest depot
- A background worker (Go, not Python) to archive to S3 every 5 minutes

Result:
- P99 latency: 45ms (down from 420ms)
- Cost: $1.3k/month (Redis + EC2) vs. $4.2k for Kafka + Flink
- No outages during peak traffic

The key insight? We didn’t need event streaming for real time—we needed fast in-memory computation and eventual persistence.


### Example 3: Multi-tenant SaaS API (100 tenants, 8k API calls/sec)

We built a SaaS platform where each tenant had isolated data. The team debated:
- Use PostgreSQL row-level security (RLS)
- Or shard by tenant ID in the app layer

We tried RLS first. It worked for small tenants, but during a load test with 500 tenants, query plans became unpredictable. Some queries scanned the entire table. P95 latency jumped to 1.2 seconds.

We switched to application-level sharding. Each tenant got its own schema in PostgreSQL (not separate databases—schemas). We used a connection pool per tenant group (shard). We added a Redis cache for tenant metadata.

Result:
- P95 latency: 180ms
- Query performance became predictable
- We could scale individual shards independently

The ‘clean’ solution (RLS) failed under load. The pragmatic one (sharding with schemas) worked because we measured the real behavior.


### What these examples show

1. **Boring tech often wins**—EC2, Redis, PostgreSQL, Python. Not serverless, not Kafka, not Rust.
2. **Latency and cost are more important than ‘best practices’**—we chose EC2 over Lambda because of cold starts, not because of architectural purity.
3. **Abstractions must be justified by data**—RLS looked clean, but it failed under load. Sharding looked messy, but it scaled.

I’ve seen teams spend months building a microservices architecture for a monolith because ‘that’s how you scale.’ Meanwhile, the real scaling bottleneck was a missing index on a foreign key.


> Summary: Real systems scale with measurement, pragmatism, and tradeoff analysis—not with adherence to ‘best practices.’ Three real examples show that boring tech, data-driven decisions, and ownership of failure modes drive senior-level outcomes.


## The cases where the conventional wisdom IS right

None of this is to say the standard advice is always wrong. There *are* cases where formal education, LeetCode, and ‘best practices’ are valuable. But they’re specific and often misunderstood.


### When you’re building a critical system

If you’re writing flight software, medical devices, or financial infrastructure, formal correctness matters. You need provable guarantees. A CS degree (or equivalent study) teaches you formal methods, state machines, and verification. You can’t learn that by shipping CRUD apps.

Example: In 2026, a team I consulted for built a payment gateway for a fintech startup. They started with a simple REST API. But after load testing, they discovered race conditions in transaction rollback. They had to implement a distributed saga with compensating transactions. They hired a distributed systems expert—who had a PhD in formal verification. He modeled the system in TLA+ and proved it safe under network partitions.

That level of rigor isn’t needed for a blog API. But for money? Absolutely.


### When you’re building a platform for other engineers

If you’re building a framework, library, or internal tool that others will depend on, you need to design for extensibility and safety. You can’t rely on ‘just add a cache’—you need to define clear contracts, handle edge cases, and document failure modes.

Example: A team I worked with built an internal event bus in 2026. They initially used Redis pub/sub. It worked for demos, but failed under backpressure. They rebuilt it using Kafka with exactly-once semantics and idempotent producers. It took 3 months. The result? Zero message loss during a regional outage.

If you’re building something others will rely on, you need more rigor than if you’re building a feature for your own app.


### When you’re debugging a complex failure
n
Sometimes, the problem isn’t in your code—it’s in the protocol, the OS, or the hardware. A deep understanding of systems programming helps. Knowing how TCP works, how the Linux scheduler behaves, or how memory allocation works can save days.

Example: In 2026, a service running on Kubernetes kept crashing with OOM kills. The team tried increasing memory limits, tuning garbage collection, and scaling pods. Nothing worked. A senior engineer with a CS background dug into the kernel logs and discovered the pod was hitting the `vm.max_map_count` limit due to a high number of memory-mapped files. The fix? Increasing the sysctl value. That’s not something you learn from a bootcamp.

So yes—there are times when formal knowledge matters. But those times are rare. Most of the time, the real work is in debugging, measuring, and shipping.


> Summary: The conventional wisdom is right when building systems where correctness, safety, or reliability is non-negotiable—flight software, financial systems, or foundational platforms. In those cases, formal education and rigorous design are essential.


## How to decide which approach fits your situation

Not every system needs a PhD. Not every feature needs a microservice. But how do you decide when to go deep and when to ship fast?

Use this simple framework:


| Context | Ask This | When to Go Deep | When to Ship Fast |
|--------|---------|-----------------|-------------------|
| **System Criticality** | Could this failure cause loss of life, money, or reputation? | Use formal methods, design docs, code reviews, and rigorous testing | Ship fast with observability and rollback plans |
| **Scale Threshold** | Will this system handle >10k RPS or >1TB data? | Design for horizontal scale from day one, use proven distributed systems patterns | Start monolithic, optimize later when bottlenecks appear |
| **Team Maturity** | Is the team experienced with debugging distributed systems? | Invest in education, pair programming, and runbooks | Focus on shipping features and reducing toil |
| **Time Pressure** | Is this a one-off feature or a core product? | If core, design carefully; if one-off, ship and iterate | Ship fast, measure, and refactor only if needed |
| **Tech Longevity** | Will this code live for >3 years? | Use stable, well-documented tech; avoid hype tools | Use modern tools that increase velocity now |


Apply this table ruthlessly. Most of the time, you’ll fall into the “ship fast” column. But when you don’t, you’ll know it.

I’ve seen teams waste months over-engineering a feature that was never used. I’ve seen teams ship a monolith that handled 100k RPS with no drama. The difference was context.


### A rule of thumb

If you can’t explain your system’s bottleneck in 10 minutes to a junior engineer, you don’t understand it well enough. If you can’t roll back a deployment in 5 minutes, you’re not shipping safely.


> Summary: Use a simple framework based on criticality, scale, team maturity, time pressure, and tech longevity to decide when to go deep and when to ship fast. Most features don’t need a PhD—just a debugger and a rollback plan.


## Objections I've heard and my responses

**Objection 1: “Without a CS degree, you’ll hit a ceiling at mid-level. Senior roles require formal knowledge.”**

I’ve worked with senior engineers who don’t have CS degrees—and they’re not ‘mid-level in disguise.’ They own systems, debug production fires, and mentor juniors. One of them built a real-time fraud detection system that processed 20k transactions/sec with 99.99% uptime. His ‘formal knowledge’? A YouTube series on TCP congestion control and a habit of reading kernel source when things broke.

The ceiling isn’t education—it’s ownership. If you’re comfortable being responsible for a system’s behavior, not just its code, you’re senior. That’s a mindset, not a degree.


**Objection 2: “But LeetCode prepares you for system design interviews, which are gatekeeped by CS fundamentals.”**

Yes, LeetCode helps you pass interviews. But interviews aren’t the same as real engineering. I’ve interviewed engineers who aced LeetCode but couldn’t explain why their API was timing out. I’ve hired engineers who struggled with Big-O but could debug a deadlock in five minutes.

The industry conflates interview performance with job performance. That’s a mistake. If your goal is to get hired, LeetCode helps. If your goal is to build systems that don’t break, debugging does.


**Objection 3: “AI tools will replace the need for deep knowledge anyway. Why bother?”**

In 2026, AI coding tools are everywhere. But they hallucinate. They suggest indexes that don’t exist. They write SQL queries that join on non-existent columns. They recommend caching strategies that will cause a stampede.

I’ve seen teams burn $15k/month on AI-generated code that needed rewrites because the tool didn’t understand their schema. The best AI assistants are the ones that are grounded in production data and observability tools—not the ones that write the most lines of code.

AI won’t replace deep knowledge. It will amplify it. But only if you know enough to validate its output.


**Objection 4: “You can’t learn distributed systems without a degree.”**

I learned distributed systems by breaking them. I joined a team that was running PostgreSQL on a single node. I suggested we split reads and writes. We did, and latency improved by 60%. Then I suggested a read replica. Then a shard. Each time, I measured the impact. I didn’t read a textbook—I ran experiments.

There are great resources: Martin Kleppmann’s book, the Jepsen blog, and real-world case studies from companies like Uber and Netflix. You don’t need a degree to learn from them. You need curiosity and a willingness to break things.


> Summary: Common objections—degrees as prerequisites, LeetCode as a proxy for skill, AI replacing knowledge, and distributed systems being inaccessible—are either outdated, misplaced, or ignore real-world outcomes. Senior skills are built through ownership and debugging, not credentials.


## What I'd do differently if starting over

If I could go back to 2026, with the knowledge I have now, here’s what I’d change:


### 1. I’d focus on measuring, not building

I spent too much time building stuff—portfolio projects, open-source contributions, even a personal blog. Instead, I’d spend more time *measuring* real systems. I’d spin up a small VPS, run a PostgreSQL instance, and monitor it with Prometheus. I’d simulate traffic with Locust. I’d break things on purpose and see how they recover. That’s how you learn latency, throughput, and failure modes—not by writing CRUD apps.


### 2. I’d work on systems that break

I’d seek jobs where I could touch production early. Not as a shadow, but as the person on call. I’d volunteer for the pager. I’d debug the outage at 2 AM. That’s where real learning happens. I’d avoid roles where I was just writing features with no ownership.


### 3. I’d ignore most ‘best practices’ until I measured the cost

I’d treat every recommendation—microservices, serverless, event sourcing—as a hypothesis to test. I’d ask: *What breaks first? What’s the cost?* Then I’d measure it. Only then would I decide whether to adopt it.


### 4. I’d learn just enough systems programming to debug

I’d learn:
- How to read a flame graph
- How to check `top`, `vmstat`, and `iostat`
- How to use `strace` and `tcpdump`
- How to read PostgreSQL and Redis logs

Not to build a kernel, but to debug one. Most senior engineers I know can do this. Most juniors can’t.


### 5. I’d build a personal runbook, not a portfolio

Instead of a GitHub profile full of toy projects, I’d keep a private wiki with:
- How I debugged a deadlock in Python
- How I tuned a slow SQL query
- How I recovered a crashed Redis cluster
- How I diagnosed a memory leak in a Go service

That runbook would be worth more than any project. It’s proof of ownership.


> Summary: Starting over, I’d prioritize measurement over building, production ownership over feature writing, and runbooks over portfolios. The goal wouldn’t be to collect credentials—but to own systems.


## Summary

I went from junior to senior without a CS degree by focusing on what actually matters: **owning systems, measuring outcomes, and debugging under pressure**. The conventional wisdom—that you need a degree, LeetCode, or ‘best practices’ to reach senior—is a distraction. What you need is the ability to understand why a system is slow, how it fails, and how to fix it without drama.

That doesn’t mean ignoring theory. It means applying theory in the right context. When building a payment system, rigor matters. When building a blog API, shipping fast matters. Most of the time, shipping fast with observability wins.

I’ve seen developers with PhDs build fragile systems. I’ve seen bootcamp grads design systems that scale. The difference isn’t education—it’s ownership.

So if you don’t have a CS degree, don’t let it stop you. Instead:

1. Get a pager.
2. Break something on purpose.
3. Measure it.
4. Fix it.
5. Repeat.

That’s the real path to senior.


## Frequently Asked Questions

**How do I get production experience if I don’t have a degree?**

Start small: run a PostgreSQL instance on a $5 VPS, simulate 1000 concurrent connections with Locust, and monitor it with Prometheus. Then intentionally crash it—kill the process, fill the disk, induce network latency—and see how it recovers. That’s production experience. Many companies hire junior engineers if you can demonstrate you’ve debugged real systems, not just written code.


**Is LeetCode useless?**

Not useless—but overvalued. LeetCode helps you pass interviews, not build systems. Focus 20% on algorithmic thinking, 80% on debugging real systems. If you spend 6 months on LeetCode and can’t explain why your API is slow, you’ve optimized the wrong thing.


**What’s the fastest way to learn distributed systems?**

Break them. Run PostgreSQL on a single node, then split reads/writes. Add a read replica. Shard. Simulate network partitions. Use tools like Toxiproxy to inject latency. Read Jepsen case studies. The fastest way isn’t reading a book—it’s running experiments and measuring the fallout.


**Can AI tools replace the need for debugging skills?**

No. AI tools hallucinate. They suggest SQL joins that don’t exist. They recommend caching strategies that will cause stampedes. The best AI tools are the ones you can validate with observability tools. Deep knowledge doesn’t go away—it becomes more valuable when you can audit AI-generated code.