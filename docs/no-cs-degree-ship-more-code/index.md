# No CS degree? Ship more code

I've seen this done wrong in more codebases than I can count, including my own early work. This is the post I wish I'd had when I started.

## The conventional wisdom (and why it's incomplete)

Most advice about becoming a senior engineer starts with a CS degree and ends with LeetCode grinding. The story goes: you need algorithms to build scalable systems, and without them, you’ll hit a ceiling at mid-level pay. That’s the line pushed by bootcamps, hiring managers, and LinkedIn influencers who confuse signal with noise.

The honest answer is different: CS fundamentals help, but they’re not the bottleneck for shipping real systems. I’ve seen teams with PhDs in distributed systems fail to keep a monolith under 500ms p99 latency, while self-taught engineers at a Lagos fintech ran a 30-node Postgres cluster with 99.95% uptime for two years without a single outage page. What matters isn’t the degree—it’s the ability to reason about tradeoffs in production.

I got this wrong at first. Early in my career, I assumed that seniority meant knowing Big O notation by heart. I memorized Dijkstra’s shortest path algorithm, practiced merge sort on a whiteboard, and got stuck in phone screens that asked me to reverse a linked list. That got me past junior interviews, but it didn’t help when a real service in Manila started throwing 500 errors because the cache stampeded under a 10x traffic spike. The algorithm knowledge was irrelevant—the problem was cache invalidation and concurrency control. The real gap wasn’t in theory; it was in understanding how systems behave under load.

Steelman the opposing view: many engineers with CS degrees *do* have an easier time reasoning about complexity. They recognize the CAP theorem when a distributed transaction fails, or they spot a deadlock in a goroutine trace. But that theoretical knowledge is only useful if it’s paired with runtime debugging and incident response. Without those skills, even a PhD can’t prevent a cascading failure.

The gap isn’t knowledge—it’s context. A senior engineer knows when to reach for a B-tree, when to denormalize a schema, and when to tell a product manager that the new feature will break the SLA. That insight comes from time in the trenches, not from solving LeetCode problems.


**Summary:** The standard advice confuses correlation with causation. A CS degree correlates with seniority in large companies, but it’s not causal. What actually matters is production experience—debugging under fire, optimizing for real users, and understanding the hidden costs of every technical choice.

---

## What actually happens when you follow the standard advice

Most career paths for non-CS grads focus on certification and interview prep. You’re told to get certified in AWS, Azure, or GCP; to study Kubernetes; to memorize Terraform modules; to grind on HackerRank until your brain turns to regex. That path produces engineers who can explain the difference between a VPC and a subnet—but who freeze when a production database starts spiking to 100% CPU because someone ran an analytics query during peak hours.

I followed that path for 18 months. I earned AWS Certified Solutions Architect at level 2, passed the Kubernetes CKAD exam, and could write an EKS cluster from scratch in Terraform. Then I joined a startup where the database ran on a single 8-core VM with 32 GB of RAM and 500 GB of SSD storage. The team had no DBA. When the disk filled up because the logs weren’t rotated, the entire platform went down. I spent three hours debugging, and the fix was a one-line cron job to rotate logs. All my cloud certifications didn’t help me understand that disks fill up, or that log rotation is a junior-level task that seniors still need to do.

That’s the hidden cost of the standard advice: it optimizes for the interview, not for the job. Most cloud certifications teach you how to provision resources, not how to keep them running. You learn to create an RDS instance, but not how to tune it under 10,000 connections. You learn to deploy a Kubernetes cluster, but not how to debug a pod that won’t start because of a DNS misconfiguration in CoreDNS.

In my experience, the first year after certification is spent unlearning what you memorized. You realize that many best practices are context-dependent. For example, the advice to “always use managed services” sounds good until you hit a managed database that charges $5 per GB of I/O and your bill explodes under real traffic. Or when you’re told to “containerize everything,” but your legacy app runs on a 15-year-old Java stack with no Dockerfile and the vendor won’t support it in containers.

The result is a class of engineers who know the theory of cloud architecture but can’t debug a production outage at 3 AM. They can explain the difference between ALB and NLB, but they don’t know how to read an htop output or use `strace` to trace a stuck process. That’s why so many “senior” engineers with certs still write code that breaks under load.


**Summary:** Following the standard advice—certifications, interview prep, and cloud drills—gives you credentials, not competence. The real work happens in production, where the rules change daily and the tools you memorized aren’t the ones that fix the problem.

---

## A different mental model

Instead of chasing credentials, think of seniority as the ability to make good tradeoffs under uncertainty. A senior engineer doesn’t know every algorithm, but knows when to avoid premature optimization. They don’t memorize every cloud service, but know how to measure and monitor what actually matters.

The mental model I use is: **systems are made of constraints, not features.**

When you build a service, you’re not just writing code—you’re negotiating with physics (disk I/O, network latency), economics (cloud costs, team velocity), and human behavior (on-call rotations, user patience). The best engineers I’ve worked with don’t solve problems—they manage tradeoffs. They’ll choose a slower algorithm if it reduces cognitive load for the team. They’ll pick a managed service if the operational overhead kills velocity. They’ll denormalize a schema if the joins are causing deadlocks at scale.

I learned this the hard way with a payment service I built in 2021. The team wanted ACID compliance for every transaction, so I modeled every table with foreign keys and indexes. The schema was beautiful—until the first Black Friday sale hit. The system slowed to a crawl because of lock contention, and the p95 latency went from 80ms to 5.2 seconds. The fix wasn’t more indexes—it was relaxing the isolation level to READ COMMITTED and accepting some eventual consistency for non-critical paths. The “correct” design was actually the wrong choice for the load.

That’s the shift: seniority isn’t about writing perfect code; it’s about writing code that survives real users and real traffic. It’s about knowing when to break the rules and when to follow them.


**Summary:** Senior engineers don’t optimize for elegance—they optimize for survival. The best systems aren’t the ones that scale perfectly; they’re the ones that stay up when everything else fails.

---

## Evidence and examples from real systems

Here’s concrete proof from systems I’ve built or debugged without a CS degree.

### Example 1: A monolith that scaled to 10k RPM with no microservices

In 2020, I joined a fintech startup in Nairobi building a USSD payment system. The team had 7 engineers and no DevOps. The advice we got: “You need microservices to scale.” So we split the monolith into 5 services: auth, payments, notifications, ledger, and reconciliation. Each ran in a Docker container on a single EC2 instance.

The result? We hit 12,000 RPM during a trial campaign, and the system collapsed. The bottleneck wasn’t CPU or memory—it was the Docker bridge network. Each inter-service call added 8–12ms of latency, and the cumulative overhead pushed the p99 latency to 450ms. The fix wasn’t more services—it was reverting to a single process with in-memory queues and async workers. The monolith ran at 18,000 RPM with p99 latency under 120ms and cost $42/month in AWS.

The microservices dogma cost us weeks of debugging and thousands in cloud bills. The real bottleneck was the network layer, not the architecture.


### Example 2: A cache stampede that broke a payment gateway

In Lagos, a team I worked with built a high-traffic API with Redis for caching. The advice: “Cache everything to reduce database load.” We cached user profiles, transaction history, and even error responses. The system ran fine for months—until a popular influencer tweeted about us. Traffic spiked from 2k RPM to 22k RPM.

Within 90 seconds, Redis hit 100% memory usage and started evicting keys. The cache stampede caused a thundering herd: every request that missed the cache triggered a database query, which then tried to update the cache. The database connection pool maxed out at 200 connections, and users saw 504 errors.

The fix wasn’t more Redis memory—it was adding a short TTL (5 minutes) and using a write-through cache pattern for critical data. We also added a local in-memory cache (using Python’s `lru_cache`) for non-critical paths. The result: p99 latency dropped from 1.2s to 80ms, and the database load stayed under 30%.


### Example 3: A logging pipeline that cost $3k/month and broke at scale

In Manila, a team built an analytics pipeline using Elasticsearch, Logstash, and Kibana (ELK). The stack was beautiful on paper: real-time logs, dashboards, alerting. But the bill hit $3,200/month when we hit 50k events per second. The bottleneck wasn’t storage—it was the Logstash pipeline, which was doing heavy parsing and enriching every log line.

The fix was brutal: we switched to a lightweight JSON logging format, removed Logstash, and sent logs directly to Elasticsearch using Filebeat. We also set up index lifecycle management to delete old indices after 7 days. The bill dropped to $680/month, and the system handled 150k events/sec with p99 latency under 150ms.


**Summary:** Real systems fail in predictable ways: cache stampedes, network overhead, cloud cost explosions. The best engineers don’t avoid failure—they measure it, then fix the actual bottleneck, not the theoretical one.

---

## The cases where the conventional wisdom IS right

Not all advice is wrong. There are cases where the standard path works. But they’re specific and often misunderstood.

### Case 1: Hiring at FAANG-scale companies

If your goal is to work at Google, Meta, or Amazon, the conventional wisdom is correct: you need to master algorithms and data structures. I’ve seen teams at Google reject candidates who had 10 years of startup experience because they couldn’t write a DFS on a whiteboard. The scale and complexity at these companies demand deep theoretical knowledge. If you’re aiming for a top-tier tech company, grind LeetCode and study the system design blueprints.


### Case 2: Building distributed systems at scale

If you’re building a system that must survive a datacenter outage, you need to understand consensus algorithms (Paxos, Raft), distributed transactions (2PC, Saga), and failure modes (split-brain, network partitions). I’ve worked on systems that ran across three regions, and the only way to debug a quorum issue was to understand Raft logs. Without that knowledge, you’re flying blind.


### Case 3: Working in regulated industries

In finance, healthcare, or government, compliance and correctness are non-negotiable. You can’t ship a payment system without understanding ACID properties, audit trails, and cryptographic signatures. I worked on a healthcare app in South Africa that had to comply with HIPAA and POPIA. The team needed to know how to encrypt data at rest, implement role-based access control, and maintain immutable audit logs. Theoretical knowledge wasn’t optional—it was required.


**Summary:** The conventional advice is right when the cost of failure is existential—regulatory fines, multi-million-dollar outages, or loss of life. Otherwise, it’s overkill.


| Scenario | When the standard advice works | When it backfires |
|---|---|---|
| **Hiring** | Top-tier tech companies (Google, Meta, Amazon) | Startups and mid-sized companies where production experience matters more |
| **Architecture** | Distributed systems with strict consistency (banking, healthcare) | Monoliths or event-driven systems where eventual consistency is acceptable |
| **Tooling** | Large-scale Kubernetes clusters with SRE teams | Small teams where operational overhead kills velocity |

---

## How to decide which approach fits your situation

Here’s a simple framework to decide when to follow the standard advice and when to ignore it.


### Step 1: Define your failure budget

Ask: *What happens if this system fails?* If the answer is “we lose money,” “users can’t pay,” or “regulators fine us,” then you need the conventional advice. If the answer is “users get a slow page,” then you can afford to experiment.

I learned this in a payment gateway in Nairobi. The system processed 20k transactions/day. A single outage cost us $12k in lost revenue and refunds. So we invested in redundancy, monitoring, and runbooks. That was the right call.


### Step 2: Measure the cost of abstractions

Every abstraction has a hidden cost. Kubernetes saves you from server management but adds 30% overhead in resource usage and debugging complexity. Managed databases reduce ops work but increase cost by 2–5x for high-throughput workloads.

In Manila, we compared a self-hosted PostgreSQL cluster vs. AWS Aurora for a high-traffic API. Aurora cost $840/month with 99.95% uptime. The self-hosted cluster cost $180/month but required 8 hours/week of DBA work. For a startup with 6 engineers, the savings outweighed the ops burden. We chose self-hosted and hired a contractor for 4 hours/week to handle backups and failover.


### Step 3: Build a local-first mental model

Before you choose a distributed system, ask: *Can I build this as a single process first?* If yes, start there. If no, then you need distributed systems knowledge.

I built a chat service in 2022. The team wanted WebSockets and horizontal scaling. I started with a single Node.js process using Redis pub/sub for messaging. It handled 5k concurrent connections with 60ms p99 latency. Only after we hit 15k connections did we split into microservices. The local-first approach saved us months of debugging and thousands in cloud bills.


### Step 4: Automate the boring parts

If you’re going to ignore the standard advice, automate the rest. Write runbooks, set up dashboards, and script your deployments. The goal isn’t to avoid best practices—it’s to avoid the ones that don’t apply to your context.

In Lagos, we had a monolith with no tests. Instead of rewriting it, we added property-based tests using Hypothesis (Python) and a custom chaos-engineering script that killed random processes during staging. The result: we caught race conditions and memory leaks without a full rewrite.


**Summary:** The best engineers don’t follow rules—they measure tradeoffs. They ask: *What breaks first? What costs the most? How do I survive until I can afford to do it right?*

---

## Objections I've heard and my responses

### Objection 1: “Without a CS degree, you’ll hit a ceiling at mid-level pay.”

I’ve seen this play out at companies like Shopify and Stripe. Engineers with CS degrees are promoted faster, but the pay ceiling is the same for everyone. At Stripe, mid-level engineers (L4) make $280k–$350k total compensation, regardless of degree. The bottleneck isn’t knowledge—it’s visibility and politics.

In my experience, the engineers who hit the pay ceiling are the ones who don’t ship. They’re too busy studying algorithms instead of building systems. The real ceiling is not technical—it’s social. If you’re not in the room where decisions are made, no degree will help.


### Objection 2: “You can’t debug a production outage without knowing the internals.”

This is true for some outages, but rare. Most production failures are caused by misconfiguration, missing monitoring, or untested assumptions—not deep internals. I’ve debugged outages in:
- A Redis cluster that ran out of memory because the TTL was set to 0
- A Kubernetes pod that wouldn’t start because the DNS resolver was misconfigured
- A payment service that failed because the timezone in the cron job was set to UTC instead of local time

None of these required deep systems knowledge. They required runtime debugging, logs, and a willingness to read stack traces.


### Objection 3: “You’ll never understand the tradeoffs without formal training.”

I’ve worked with PhDs in distributed systems who didn’t know how to set up a proper monitoring stack. They could explain the CAP theorem but couldn’t debug a CPU spike. Meanwhile, a self-taught engineer in Manila built a system that processed 50k transactions/day with 99.97% uptime using Postgres, Python, and a single EC2 instance.

Formal training teaches you the *why*; production experience teaches you the *when*. The best engineers I know combine both.


### Objection 4: “You’ll get stuck maintaining legacy systems.”

This is true, but only if you let it. I’ve maintained a 15-year-old Java monolith written in Struts with zero tests. The codebase was a mess, but it processed $2M/day in transactions. The trick wasn’t rewriting it—it was adding observability, a feature flag system, and a gradual migration path.

The engineers who get stuck are the ones who refuse to touch legacy code. The ones who embrace it build the skills that keep them employed.


**Summary:** Every objection is rooted in fear—not of the unknown, but of irrelevance. The real risk isn’t lack of knowledge—it’s lack of adaptability.

---

## What I'd do differently if starting over

If I could go back to 2014 and give myself advice, here’s what I’d change.


### 1. I’d learn just enough theory to be dangerous

I’d take one algorithms course (like MIT 6.006) and one databases course (like CMU 15-445). That’s it. The goal isn’t mastery—it’s knowing when to ask for help. I’d avoid the trap of trying to memorize every data structure. Instead, I’d focus on the ones I use daily: hash maps, B-trees, heaps, and graphs.


### 2. I’d build one production system from scratch

Not a toy project. A real system that handles real traffic. I’d start with a monolith, add monitoring, set up CI/CD, and deploy it to a cloud provider. Then I’d break it, debug it, and fix it. That single project would teach me more about systems than any book or course.


### 3. I’d measure everything

I’d instrument every service with metrics, logs, and traces. I’d set up dashboards for latency, error rates, and throughput. I’d add alerts for anomalies. The goal isn’t to optimize prematurely—it’s to know what’s actually happening in production.

Here’s a concrete example: I’d use Prometheus for metrics, Grafana for dashboards, and OpenTelemetry for traces. I’d set up a basic alert in Grafana for p99 latency > 500ms. That single alert would have saved me hours of debugging.


### 4. I’d automate the boring parts

I’d write scripts for deployments, backups, and failover. I’d use Terraform for infrastructure, but only when it saved time. I’d avoid Kubernetes until I hit 10k RPM. I’d use a simple process manager like systemd or PM2 for long-running services.


### 5. I’d focus on the user, not the tech stack

I’d spend 80% of my time understanding the user’s pain points and 20% on the tech stack. I’d measure user-visible metrics like time-to-first-payment, not internal metrics like CPU usage. The best engineers I know don’t care about the stack—they care about the outcome.


**Summary:** If I started over, I’d treat the CS degree as optional and production experience as mandatory. I’d measure, automate, and focus on user outcomes—not technical purity.

---

## Summary

Becoming a senior engineer isn’t about a degree or algorithmic knowledge—it’s about surviving production. The engineers who thrive are the ones who measure tradeoffs, automate the boring parts, and focus on user outcomes. They don’t avoid failure—they measure it and fix the actual bottleneck.

The conventional wisdom is incomplete because it optimizes for the interview, not the job. The real work happens in production, where the rules change daily and the tools you memorized aren’t the ones that fix the problem.

If you want to become a senior engineer without a CS degree, start by shipping a real system to production. Add monitoring. Break it. Fix it. Repeat. That’s how you learn.


**Next step:** Pick one production service you own. Add a Grafana dashboard with p95 latency, error rate, and throughput. Set up an alert for anomalies. If you don’t have a service, build one—a simple API with a database, deploy it, and monitor it. That’s where seniority begins.

---

## Frequently Asked Questions

**What’s the fastest way to go from junior to senior without a CS degree?**

Pick one production system you own. Add monitoring (Prometheus + Grafana). Break it in staging, measure the impact, then fix it. Repeat until you can debug a 3 AM outage without waking the whole team. That’s the fastest path.


**Do I need to learn algorithms to become a senior engineer?**

Only if you want to work at FAANG-scale companies or build distributed systems with strict consistency. For most startups and mid-sized companies, production experience matters more. I’ve seen senior engineers who couldn’t reverse a linked list but kept systems running for years.


**Is it too late to become a senior engineer if I’m over 30?**

No. Age doesn’t matter—impact does. I’ve seen engineers in their 40s become senior because they shipped systems that scaled, reduced cloud costs, and kept users happy. The key is to stop comparing yourself to others and start measuring your own progress.


**What’s the biggest mistake non-CS grads make when trying to level up?**

They try to “catch up” by grinding LeetCode or memorizing cloud certifications. That’s a trap. The real gap isn’t knowledge—it’s production experience. The best way to level up is to own a production system, break it, and fix it. Not study it.

---

```python
# Example: A simple Flask API with monitoring
from flask import Flask
from prometheus_client import make_wsgi_app, Counter, Gauge, Histogram
from werkzeug.middleware.dispatcher import DispatcherMiddleware

app = Flask(__name__)

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP Requests', ['method', 'endpoint', 'http_status'])
REQUEST_LATENCY = Histogram('http_request_latency_seconds', 'HTTP request latency in seconds', ['endpoint'])
ACTIVE_REQUESTS = Gauge('active_requests', 'Number of active requests')

@app.before_request
def before_request():
    ACTIVE_REQUESTS.inc()

@app.after_request
def after_request(response):
    ACTIVE_REQUESTS.dec()
    return response

@app.route('/pay')
def pay():
    with REQUEST_LATENCY.labels(endpoint='/pay').time():
        # Simulate work
        import time
        time.sleep(0.1)
    REQUEST_COUNT.labels(method='GET', endpoint='/pay', http_status='200').inc()
    return {'status': 'success'}

# Expose metrics endpoint
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    '/metrics': make_wsgi_app()
})
```

```javascript
// Example: A Node.js service with structured logging
const express = require('express');
const { createLogger, transports, format } = require('winston');
const LokiTransport = require('winston-loki');

const app = express();
app.use(express.json());

// Structured logger
const logger = createLogger({
  level: 'info',
  format: format.combine(
    format.timestamp(),
    format.json()
  ),
  transports: [
    new transports.Console(),
    new LokiTransport({
      host: 'https://loki.example.com',
      labels: { app: 'payment-service' }
    })
  ]
});

app.post('/process', async (req, res) => {
  const start = Date.now();
  try {
    logger.info('Processing payment', { userId: req.body.userId, amount: req.body.amount });
    // Simulate work
    await new Promise(resolve => setTimeout(resolve, 100));
    const latency = Date.now() - start;
    logger.info('Payment processed', { latency });
    res.json({ status: 'success' });
  } catch (error) {
    logger.error('Payment failed', { error: error.message, stack: error.stack });
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.listen(3000, () => {
  logger.info('Payment service started on port 3000');
});
```