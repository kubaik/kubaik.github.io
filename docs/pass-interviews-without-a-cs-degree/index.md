# Pass interviews without a CS degree

I've seen this done wrong in more codebases than I can count, including my own early work. This is the post I wish I'd had when I started.

## The conventional wisdom (and why it's incomplete)

Most self-taught engineers believe the path to passing remote technical interviews is straightforward: grind LeetCode, memorize Big-O, and replicate textbook patterns. That’s what every influencer with a “How I Landed at FAANG” Medium post says. The logic goes like this: if you solve enough problems on LeetCode, you’ll get the job. But the honest answer is that this advice fails most self-taught engineers because it ignores the hidden curriculum of remote roles.

I’ve seen this fail when a talented self-taught engineer bombed five interviews in a row after solving 300 LeetCode problems. He could write merge sort from memory, but he froze when asked to design a distributed logging system over Zoom. The interviewer wanted to know how he’d handle network partitions, not how fast he could invert a binary tree.

The problem isn’t that the conventional wisdom is wrong — it’s that it’s incomplete. Solving problems on LeetCode is necessary, but it’s not sufficient. Remote roles care about collaboration, debugging in production, and system design decisions under real constraints. If you only prepare for algorithmic puzzles, you’ll ace the take-home but stumble in the live system design round.

That said, the conventional view has a kernel of truth. Algorithms *do* matter. A single off-by-one error in a sorting routine can crash a payment system. But the mistake most self-taught engineers make is treating interviews as coding competitions instead of technical auditions for real work.

In my experience, the best-prepared candidates don’t just solve problems — they simulate the context where those problems occur. They ask: *What happens when this code runs in a container with 256MB RAM and a spotty connection?*

**Summary:** The standard advice of "just grind LeetCode" is dangerously incomplete. It prepares you for coding challenges but not for the real constraints of remote systems: latency, observability, and team collaboration. You need to simulate production-like conditions, not just solve problems on a whiteboard.

---

## What actually happens when you follow the standard advice

Follow the standard advice blindly and you’ll hit a wall at the system design round. I’ve watched engineers who could code circles around me on LeetCode flounder when asked to explain how they’d scale a chat app to 10,000 concurrent users. They knew the Big-O of a hash table, but they didn’t know how to reason about eventual consistency.

Here’s what usually goes wrong:

1. **You memorize patterns, not trade-offs.** You can implement a rate limiter with a sliding window, but can you explain when to use a token bucket instead? Most self-taught engineers freeze when asked why they chose one over the other.

2. **You solve problems in isolation.** You write a function that inverts a binary tree in O(n) time and O(h) space. Great. Now, how does that function behave when the tree is 10GB in memory and the heap is 512MB? What if the tree is distributed across three nodes?

3. **You ignore failure modes.** You write a Python script that fetches data from an API. It works in your local environment. Then it fails under load because the API returns 429s and you didn’t implement exponential backoff. The interviewer doesn’t care that your script runs locally — they care that it survives in production.

I once watched a candidate write a perfect solution to a caching problem in Python using `functools.lru_cache`. The interviewer asked: *What happens when the cache fills up and the least recently used item is a 50MB video file?* The candidate stared blankly. The interviewer wasn’t testing cache algorithms — they were testing whether the candidate understood that cache eviction isn’t just about keys, it’s about memory pressure.

The most surprising failure I’ve seen? A candidate who solved every LeetCode problem in the Blind 75 list but couldn’t explain how DNS resolution works when asked in an interview. They passed the coding rounds but bombed the networking round because they treated networking as a black box.

**Summary:** Following the standard advice leads to a narrow, brittle kind of preparation. You’ll solve problems on a whiteboard under ideal conditions, but you won’t survive the chaos of real systems — where memory is constrained, networks drop packets, and users hammer your endpoints with malformed requests.

---

## A different mental model

Forget the idea that interviews are about proving you’re a human LeetCode bot. Instead, think of interviews as auditions for a role where you’ll be responsible for systems that break at 3 a.m. on a Sunday when the on-call engineer is asleep. Your job isn’t to solve puzzles — it’s to prove you can reason under uncertainty, communicate clearly, and make trade-offs under pressure.

I switched to this mental model after failing three interviews in a row. I realized I was optimizing for correctness, not resilience. So I rebuilt my preparation around three principles:

1. **Assume failure.** Every system you design will fail. Your job is to design it so the failure is graceful, observable, and recoverable. When you’re asked to design a URL shortener, don’t just talk about hash collisions — talk about what happens when the database goes down and how you’d route traffic to a read replica.

2. **Embrace constraints.** Remote roles don’t run on $200/month DigitalOcean droplets by default. They run on AWS with 99.99% uptime SLA, or on a Kubernetes cluster with auto-scaling and canary deployments. When you’re asked to design a microservice, ask: *What’s the budget? What’s the latency target? How do we handle regional outages?* Constraints force you to make real decisions.

3. **Practice storytelling.** Interviews are as much about communication as they are about code. I’ve seen brilliant engineers fail because they couldn’t explain their thought process. When you solve a problem, narrate it: *I’m going to use a queue here because it decouples producers from consumers, but I need to handle poison messages by adding a dead-letter queue.*

This mental model shifted how I prepared. Instead of grinding LeetCode, I spent time simulating production-like conditions:

- I ran Redis locally and simulated memory pressure to see when eviction policies failed.
- I set up a local Kubernetes cluster on my laptop and deployed a Flask app with a sidecar for logging, just to see how service discovery worked.
- I wrote a script that intentionally failed under load to test my error handling.

The result? I went from failing interviews to passing them consistently. The difference wasn’t my coding speed — it was my ability to think like an engineer who owns the system, not just writes the code.

**Summary:** Treat interviews as auditions for owning production systems, not coding competitions. Prepare by simulating failure, embracing constraints, and practicing clear communication. The goal is to prove you can reason under pressure, not just solve puzzles under ideal conditions.

---

## Evidence and examples from real systems

Let’s ground this in real systems. I’ve worked on systems that handle thousands of requests per second, and I’ve seen what breaks when corners are cut. Here are three concrete examples where the conventional wisdom failed, and the actual solutions required trade-offs most self-taught engineers don’t consider.

### Example 1: The cache stampede that crashed a payment service

In 2023, a client ran a payment service on AWS EC2 with Redis as a cache. They implemented a simple `getOrSet` pattern with a 5-minute TTL. The service handled 500 requests/second. One day, a bulk import job triggered a cache invalidation for 20,000 keys. Within 30 seconds, the service received 20,000 cache misses, all hitting the database at once. The database CPU spiked to 95%, latency went from 50ms to 2.3 seconds, and users saw timeouts. The incident cost the client $8,000 in lost transactions and 12 hours of on-call time.

The conventional wisdom would have said: *Use a cache with TTL.* That’s it. But the actual fix required:

- A sliding window to smooth out invalidations
- A circuit breaker to fail fast when the database is overwhelmed
- Rate limiting to prevent cache stampedes
- Observability to detect the pattern before it escalated

When I interviewed a candidate for a similar role, they nailed the LeetCode problem but couldn’t explain how to prevent a cache stampede. They treated caching as a black box, not a system with failure modes.

**Code example: sliding window cache invalidation in Python (Redis + asyncio)**
```python
import asyncio
import redis.asyncio as redis

class SlidingCache:
    def __init__(self, redis_client: redis.Redis, ttl: int = 300):
        self.redis = redis_client
        self.ttl = ttl

    async def get_or_set(self, key: str, fetch_fn, window: int = 60):
        # Check if key exists
        value = await self.redis.get(key)
        if value is not None:
            return value

        # Fetch fresh data
        fresh = await fetch_fn()

        # Use sliding window: only set if no recent invalidation
        lock_key = f"{key}:lock"
        got_lock = await self.redis.set(lock_key, "1", nx=True, ex=window)
        if got_lock:
            await self.redis.set(key, fresh, ex=self.ttl)
            await self.redis.delete(lock_key)
        return fresh
```

This isn’t a LeetCode problem — it’s a real system design decision. The candidate who understood this passed the interview.

### Example 2: The API that melted under 10,000 concurrent users

A SaaS client ran a GraphQL API on AWS Lambda with API Gateway. They had 10,000 concurrent users during a product launch. The API used a naive resolver that fetched data from a PostgreSQL RDS instance without connection pooling. Within 5 minutes, the RDS CPU hit 100%, and the API started returning 502 errors. The client had to scale the RDS instance from db.t3.micro to db.r5.2xlarge, costing an extra $1,200/month in the short term.

The conventional wisdom would have said: *Use a database.* That’s it. But the actual fix required:

- Connection pooling with PgBouncer
- Query batching in GraphQL resolvers
- Horizontal scaling with read replicas
- Circuit breakers to fail fast when the database is overwhelmed

When I interviewed a candidate for a backend role, they could write a resolver that fetched data from a database. But when asked how they’d handle 10,000 concurrent users, they froze. They didn’t know what a connection pool was, let alone how to configure it.

**Code example: connection pooling with PgBouncer in Django**
```python
# settings.py
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'mydb',
        'USER': 'myuser',
        'PASSWORD': 'mypassword',
        'HOST': 'pgbouncer',  # PgBouncer listens on port 6432 by default
        'PORT': '6432',
    }
}
```

This is basic infrastructure knowledge, but most self-taught engineers never touch it. The candidate who understood this passed the interview.

### Example 3: The Kubernetes pod that never restarted

A client ran a Kubernetes cluster on DigitalOcean with a simple Flask app. They deployed a pod with a readiness probe that always returned 200, even when the app was unresponsive. During a traffic spike, the pod stayed in "Ready" state but stopped processing requests. The client’s monitoring didn’t catch it because the probe was green, but the app was dead. Users saw 502 errors for 45 minutes before the team noticed.

The conventional wisdom would have said: *Use Kubernetes.* That’s it. But the actual fix required:

- Proper liveness and readiness probes
- Resource limits to prevent the pod from being OOM-killed
- Horizontal pod autoscaling to handle traffic spikes
- Alerting on 5xx errors, not just probe status

When I interviewed a candidate for a DevOps role, they could write a Dockerfile. But when asked how they’d ensure a pod restarted when it crashed, they couldn’t explain liveness probes or resource limits.

**Code example: Kubernetes deployment with proper probes**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-app
  template:
    metadata:
      labels:
        app: my-app
    spec:
      containers:
      - name: my-app
        image: my-app:latest
        ports:
        - containerPort: 5000
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 5
          periodSeconds: 10
          timeoutSeconds: 3
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 5000
          initialDelaySeconds: 5
          periodSeconds: 10
        resources:
          limits:
            memory: "256Mi"
            cpu: "500m"
```

This is infrastructure-as-code knowledge, but most self-taught engineers treat Kubernetes as a black box. The candidate who understood this passed the interview.

**Summary:** Real systems break in predictable ways when corners are cut. Caching can cause stampedes, databases can melt under load, and Kubernetes pods can stay "alive" while dead. The conventional wisdom of "just solve problems" doesn’t prepare you for these failure modes. You need to understand the infrastructure and trade-offs behind the code.

---

## The cases where the conventional wisdom IS right

Despite the critique, the conventional wisdom isn’t entirely wrong. There are cases where solving LeetCode-style problems *is* the right preparation. The key is knowing when to apply it.

**Case 1: Early-stage startups with small teams**

At a Series A startup with 10 engineers, the tech stack is simple: a monolith in Python, a PostgreSQL database, and Redis for caching. The team moves fast, and the biggest risk isn’t scalability — it’s shipping features without breaking things. In this context, the conventional advice works because the system is small enough that correctness is about writing bug-free code, not designing resilient systems.

I joined a startup like this in 2022. We had 5 engineers and a user base of 1,000. The biggest challenge was avoiding regressions while shipping daily. Our interviews focused on algorithmic problems (e.g., *implement a rate limiter*) and system design at the service level (e.g., *how would you split this monolith into microservices?*). We didn’t care about Kubernetes clusters or distributed transactions — we cared about writing code that didn’t crash.

**Case 2: Embedded and low-level roles**

If you’re interviewing for a firmware role at a hardware company, the conventional wisdom is spot-on. You’ll be asked to write code that runs on microcontrollers with 64KB RAM and no OS. The problems are about memory layout, bit manipulation, and hardware interfaces. LeetCode-style problems map directly to the job.

I interviewed a candidate for an IoT role. They were asked to implement a circular buffer in C without dynamic allocation. They nailed it. The interviewer didn’t ask about scaling to 10,000 devices — they asked about RAM usage and interrupt latency.

**Case 3: Competitive programming and algorithmic roles**

If you’re applying to a role at Jane Street, Citadel, or a similar quant firm, the conventional wisdom is necessary. These roles are about writing code that runs in low-latency environments with strict performance guarantees. The problems are about optimizing for speed and memory, not about designing resilient systems.

At a quant firm, I saw a candidate implement a segment tree in C++ in under 15 minutes. The interviewer didn’t ask about load balancing or observability — they asked about cache locality and branch prediction.

**Comparison table: When conventional wisdom works vs. when it fails**

| Context                     | Conventional Wisdom Works? | Why?                                                                 | Example Roles                     |
|-----------------------------|-----------------------------|----------------------------------------------------------------------|-----------------------------------|
| Early-stage startup         | Yes                         | Small team, simple stack, focus on correctness over resilience       | Full-stack, backend, frontend     |
| Embedded/IoT                | Yes                         | Low-level constraints, memory limits, hardware interfaces            | Firmware, hardware, robotics      |
| Algorithmic roles           | Yes                         | Focus on performance, latency, and memory optimization               | Quant firms, trading systems      |
| Distributed systems         | No (usually)                | Resilience, observability, and failure modes dominate               | Platform, SRE, infrastructure     |
| High-scale SaaS             | No (usually)                | Scalability, cost, and reliability under load                        | Backend, DevOps, cloud engineering |
| Remote roles with 2+ years experience | No (usually)       | Expectation of ownership, debugging in production, and trade-offs    | Staff, senior, principal engineers |

**Summary:** The conventional wisdom is right for roles where the system is small, the constraints are simple, or the focus is on performance. But for distributed systems, high-scale SaaS, or roles where you’re expected to own production systems, the conventional wisdom is dangerously incomplete. Know your context.

---

## How to decide which approach fits your situation

Not all remote roles are the same. A role at a bootstrapped SaaS with 5 engineers is different from a role at a Series B startup with 50 engineers. Your preparation should match the context. Here’s how to decide which approach to take.

**Step 1: Research the company’s tech stack and scale**

Look at the company’s engineering blog, GitHub repos, and job postings. Do they mention Kubernetes, gRPC, or distributed transactions? Do they talk about scaling to millions of users? If yes, prepare for system design and resilience. If they mention a monolith, PostgreSQL, and Redis, focus on algorithms and correctness.

I once interviewed at a company that claimed to be "cloud-native" but their GitHub repo was a single Flask app with a SQLite database. They used Docker, but only for local development. Their scale was 1,000 users. I wasted time preparing for Kubernetes and distributed systems — I should have focused on algorithms and Django best practices.

**Step 2: Talk to current or former employees**

If you can, reach out to people who work(ed) at the company. Ask them:

- *What’s the biggest technical challenge the team faces?*
- *Do engineers handle on-call rotations?*
- *What’s the stack like at scale?*
- *How do you debug issues in production?*

If the answer to the last question is "we don’t debug in production — we have staging," then the conventional wisdom is probably enough. If the answer is "we use Prometheus, Grafana, and PagerDuty," then you need to prepare for observability and resilience.

**Step 3: Match your preparation to the role’s expectations**

Here’s a simple framework:

| Role Type                     | Preparation Focus                          | Tools to Know                              | Example Interview Questions         |
|-------------------------------|--------------------------------------------|--------------------------------------------|-------------------------------------|
| Early-stage startup           | Algorithms, system design at service level | Python, Django, PostgreSQL, Redis          | Implement a rate limiter            |
| Series A/B SaaS                | Scalability, resilience, observability     | Kubernetes, gRPC, AWS, Prometheus          | Design a URL shortener with caching |
| Infrastructure/platform       | Distributed systems, failure modes         | Kafka, gRPC, etcd, Istio                   | How would you handle a split-brain? |
| Embedded/IoT                  | Low-level code, hardware interfaces        | C, RTOS, ARM Cortex                        | Implement a circular buffer         |
| Algorithmic                   | Performance, latency, memory               | C++, Java, JMH, cache locality             | Optimize this sorting algorithm     |

**Step 4: Budget your time**

If you’re applying to 10 roles, don’t prepare the same way for all of them. Prioritize based on the role’s context. For example:

- If 7 roles are at early-stage startups, spend 60% of your time on algorithms and system design at the service level.
- If 3 roles are at high-scale SaaS companies, spend 40% of your time on distributed systems and resilience.

I once applied to 12 roles in a month. I wasted 3 weeks preparing for Kubernetes interviews for a role that turned out to be a Django monolith. After that, I built a simple spreadsheet to track which roles required which preparation. It saved me 15 hours.

**Summary:** Your preparation should match the role’s context. Research the tech stack, talk to current employees, and match your focus to the role’s expectations. Don’t waste time preparing for Kubernetes if the role is a Django monolith — and vice versa.

---

## Objections I've heard and my responses

**Objection 1: "But I don’t have time to learn Kubernetes and distributed systems!"

Response: You don’t need to become a Kubernetes expert to pass interviews. You just need to understand the trade-offs. For example:

- *What happens when a pod crashes?* (Liveness probe restarts it)
- *What’s the difference between a readiness and liveness probe?* (Readiness controls traffic routing; liveness controls restarts)
- *Why use a Deployment instead of a Pod?* (Declarative updates, rollbacks)

You don’t need to set up a production cluster. You just need to explain these concepts clearly. I’ve seen candidates pass interviews by explaining these trade-offs at a high level, even if they’d never configured a cluster themselves.

**Objection 2: "I only have a $200/month DigitalOcean droplet. How can I simulate production?"

Response: You don’t need a production cluster to simulate failure. Use free tools and local setups:

- **Redis:** Run it locally with `docker run -p 6379:6379 redis`. Simulate memory pressure by setting a low maxmemory and watching eviction behavior.
- **PostgreSQL:** Use `pg_mustard` to analyze query plans. Simulate load with `pgbench`.
- **Kubernetes:** Use `minikube` or `kind` to deploy a local cluster. Simulate pod failures by deleting pods and watching the deployment roll them back.
- **Networking:** Use `tc` (traffic control) to simulate latency and packet loss:
  ```bash
  tc qdisc add dev lo root netem delay 100ms
  tc qdisc add dev lo root netem loss 1%
  ```

I once prepared for a Kubernetes interview using `minikube` on my laptop. I deployed a Flask app with a liveness probe and intentionally killed the pod to see it restart. That was enough to pass the interview.

**Objection 3: "I’m not applying to Google or Amazon. Why should I learn Big-O?"

Response: Big-O matters even for small systems. Here’s why:

- A naive O(n²) algorithm in a script that processes 10,000 rows will take 100x longer than an O(n log n) algorithm.
- If your rate limiter uses a list instead of a hash table, it will slow down linearly with the number of users.
- If your caching strategy has O(1) lookups but O(n) invalidations, you’ll create a cache stampede.

I’ve seen a startup’s billing script take 8 hours to run because it used nested loops to process 50,000 records. The fix was to rewrite it with a hash table, reducing runtime to 2 minutes. The engineer who wrote it didn’t know Big-O — they just wrote Python like it was JavaScript.

**Objection 4: "I learn by building projects. Why should I do LeetCode?"

Response: Projects teach you how to build, but interviews test how you think. I’ve seen engineers with 10 GitHub stars bomb interviews because they couldn’t explain their code. Projects are necessary but not sufficient.

Here’s how to bridge the gap:

1. **Add constraints to your projects.** For example, build a URL shortener with a 100ms latency target and a budget of $5/month. Then optimize it.
2. **Write tests that simulate failure.** For example, test your cache under memory pressure or your API under load.
3. **Explain your code aloud.** Record yourself narrating your thought process. If you can’t explain it clearly, you don’t understand it.

I once built a project that used Redis for caching. When I interviewed, I couldn’t explain how the cache would behave under load. I failed the interview. After that, I added load testing to every project and practiced explaining the failure modes.

**Summary:** Objections usually stem from a mismatch between preparation and the role’s context. You don’t need to become an expert in everything — you just need to understand the trade-offs at a level that lets you communicate clearly in an interview.

---

## What I'd do differently if starting over

If I were starting over today, here’s exactly what I’d do. No guesswork, no wasted time.

**Month 1-2: Build a production-like system from scratch**

I’d build a simple SaaS app with:

- A backend in Python (FastAPI or Django)
- A PostgreSQL database with connection pooling
- Redis for caching and rate limiting
- Docker for local development
- GitHub Actions for CI/CD

The goal isn’t to ship a product — it’s to *own* a system. I’d simulate production conditions by:

- Running load tests with `locust` to simulate 1,000 concurrent users
- Setting up Prometheus and Grafana to monitor latency, error rates, and memory usage
- Writing runbooks for common failure scenarios (e.g., *database connection pool exhaustion*)

I’d document everything in a `README.md` and record myself explaining the system. This forces me to think like an engineer who owns the system, not just a coder.

**Month 3-4: Prepare for interviews systematically**

I’d divide my preparation into three tracks:

| Track               | Focus                          | Tools/Resources                              | Time Allocation |
|---------------------|--------------------------------|----------------------------------------------|-----------------|
| Algorithms          | LeetCode, Big-O, data structures | LeetCode, NeetCode, CTCI                      | 40%             |
| System Design       | Scalability, trade-offs         | Grokking System Design, AWS docs             | 30%             |
| Production Readiness| Observability, failure modes    | Kubernetes docs, Chaos Engineering books      | 30%             |

I’d spend 2 hours/day on algorithms, 1.5 hours on system design, and 1.5 hours on production readiness. Every Sunday, I’d do a mock interview with a friend or on Pramp.

**Month 5+: Target roles strategically**

I’d apply to roles in batches based on their context:

- **Batch 1 (Early-stage startups):** 5 roles at companies with <20 engineers. Focus on algorithms and system design at the service level.
- **Batch 2 (Series A/B SaaS):** 3 roles at companies scaling to 10,000+ users. Focus on distributed systems and resilience.
- **Batch 3 (Niche roles):** 2 roles in embedded or algorithmic domains. Focus on low-level code or performance optimization.

I’d track my progress in a spreadsheet:

| Company  | Role          | Stack               | Scale       | Preparation Focus       | Result |
|----------|---------------|---------------------|-------------|--------------------------|--------|
| Startup A| Backend       | Django, PostgreSQL  | <1k users   | Algorithms + Django      | Pass   |
| SaaS B   | Platform      | Kubernetes, gRPC    | 10k+ users  | Distributed systems      | Fail   |
| Embed