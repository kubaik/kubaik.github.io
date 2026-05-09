# Ace remote interviews without a CS degree

I've seen this done wrong in more codebases than I can count, including my own early work. This is the post I wish I'd had when I started.

## The conventional wisdom (and why it's incomplete)

Most self-taught engineers are told to "just practice LeetCode" until they bleed. The advice comes in three flavors: grind hundreds of problems, memorize patterns, or watch NeetCode videos at 2x speed. That’s what got me my first remote job at a Berlin startup in 2021 — but only after I realized the advice is incomplete.

Here’s the hole: the same engineers who recommend LeetCode never tell you what happens after you solve the problem. They skip the part where you explain your solution to a room of people who don’t care about your Big-O notation — they care if your code will run in production at 3 AM when the on-call pager screams.

I’ve seen this fail when the candidate nailed the LeetCode round but froze during the system design round. The honest answer is that most self-taught engineers optimize for the wrong thing: solving problems quickly instead of designing systems that don’t wake them up.

The opposing view is seductive: "If you can solve hard problems on paper, you can solve anything in real life." That’s true only if you equate "solving" with "writing correct code under time pressure." Real systems fail in ways whiteboard problems never model: race conditions in distributed caches, memory leaks in long-running services, or N+1 queries that melt your database at 2 AM. Those don’t appear in LeetCode.

The truth is that technical interviews test two skills: problem-solving under constraints and clear communication. The first is measurable; the second is not. Most advice focuses on the first and ignores the second — until the interview is over and the feedback says "communication issues."

I learned this the hard way when I interviewed at a Gulf startup in 2022. I solved the LeetCode problem in 12 minutes, but the interviewer cut me off after my third sentence and said, "You’re not explaining what’s happening in memory." I passed that round but failed the next because my system design lacked trade-offs. The pattern repeats: technical correctness ≠ interview success.

The gap isn’t just in difficulty — it’s in context. LeetCode problems run in a single process with infinite memory. Real systems run on machines with 4 GB RAM, shared databases, and users who hammer refresh. The disconnect explains why many self-taught engineers pass interviews but struggle in their first remote job.

So the conventional wisdom is half-right: practice problems. But ignore the second half — how you talk about those problems — and you’ll fail interviews anyway.


## What actually happens when you follow the standard advice

I spent six months in 2020 doing nothing but LeetCode problems. I reached 400 problems on LeetCode, 150 on CodeSignal, and 50 on AlgoExpert. I timed myself, tracked accuracy, and even built a spreadsheet with problem types. I thought I was ready.

The first interview at a US SaaS company was a disaster. I solved the problem in 8 minutes, but the interviewer interrupted me after 4 minutes and said, "Stop. How much memory does your solution use?" I froze. I had never measured memory usage of a Python function because LeetCode doesn’t care.

I bombed three interviews that week. The feedback was consistent: "Good algorithm, but you didn’t discuss trade-offs." I realized I had optimized for solving problems, not for explaining decisions.

Then I tried system design. I watched videos, drew boxes, and memorized CAP theorem definitions. I thought I was ready until a Gulf fintech company asked me to design a payment system. I drew a diagram with three services and said, "This is scalable." The interviewer asked, "What happens if the payment service crashes mid-transaction?" I said, "Retry." He asked, "Where? How? What’s the timeout?" I had no answer.

I failed that round too.

The pattern became clear: the standard advice trains you to solve problems in isolation, but interviews test whether you can reason about systems in failure. LeetCode won’t teach you that you need to discuss durability, latency, and cost when you design a service.

I eventually passed interviews by changing my focus: I stopped measuring progress by problem count and started measuring by the clarity of my explanations. That meant recording myself explaining solutions and reviewing the recordings. It meant studying real system failures instead of textbook definitions.

The honest answer is that the standard advice works only if you already know how to communicate technical decisions. If you don’t, you’ll keep failing after you solve the problem.


## A different mental model

Forget the idea that interviews test pure coding ability. They test whether you can think like an engineer who owns a system, not a coder who writes functions.

Here’s a better mental model: imagine you’re on-call for a service that just went down. Your job is to explain what happened, why it failed, and how you’ll prevent it next time — all while the page is still ringing. The interviewer is playing the role of the teammate who needs to understand your reasoning before they can hand you the pager.

This model explains why many self-taught engineers pass algorithm rounds but fail system design. The algorithm round is like debugging a single endpoint: you isolate the problem, fix it, and move on. The system design round is like debugging a distributed system: you need to explain which services failed, how data flows, and what trade-offs you made.

I first used this model during an interview at a European e-commerce company in 2023. The interviewer asked me to design a shopping cart that scales to 10,000 concurrent users. Instead of drawing boxes, I asked: "What’s the worst-case failure?" He said: "Database overload." I then explained how I’d add a caching layer, shard the cart table, and implement a circuit breaker. I didn’t just describe the design — I explained the failure scenarios and recovery steps.

I passed that round.

The model also explains why many engineers struggle with behavioral rounds. The interviewer isn’t asking about Scrum or Jira — they’re asking whether you can own a system end-to-end. That means discussing incidents, on-call rotations, and post-mortems. Those stories reveal whether you think like an engineer or like a coder.

So the new mental model is: interviews test ownership, not problem-solving. Ownership means you can explain not just what you built, but why you built it that way, what breaks, and how you’ll fix it.


## Evidence and examples from real systems

Let me show you three real systems where the mental model matters more than the algorithm.

**Example 1: Caching layer in a payments service**

At a Gulf payment startup in 2021, we ran a Python service that processed 5,000 transactions per second. The service called a third-party fraud API for each transaction. The API had a 200ms latency and a rate limit of 100 calls per second. Our database was PostgreSQL on a $100/month DigitalOcean droplet.

The first version of the service called the fraud API synchronously for every transaction. The 95th percentile latency was 300ms. The database query queue grew during traffic spikes, and the service crashed when the API returned 429 errors.

I redesigned the service to cache fraud decisions for 5 minutes. The cache used Redis on a $15/month DigitalOcean droplet. The 95th percentile latency dropped to 80ms. The database load dropped by 80%. The API rate limit errors disappeared.

But the real lesson wasn’t the latency improvement — it was the trade-offs. We had to handle cache invalidation when user data changed, deal with Redis memory limits, and decide what to do if Redis crashed. Those trade-offs became system design questions in interviews.

I’ve seen this exact scenario in interviews at European companies. They ask: "How would you design a fraud detection cache?" The correct answer isn’t just "Use Redis" — it’s explaining the eviction policy, the fallback strategy, and the monitoring setup.

**Example 2: Memory leak in a long-running Node.js service**

At a US SaaS company in 2022, we ran a Node.js service that aggregated logs from 1,000 servers. The service used Winston for logging and ran in a 1 GB container. After three days, the service memory usage climbed to 900 MB and Node.js crashed.

The root cause was a circular reference in the log metadata object. The garbage collector couldn’t reclaim the memory because the objects referenced each other. The fix was to use `circular-json` to strip circular references before logging.

The real lesson wasn’t the fix — it was the observability. We added a memory gauge to Prometheus, set an alert at 800 MB, and implemented a health check that restarted the service if memory spiked. Those details became behavioral questions in interviews: "Tell me about a time your service crashed. How did you debug it?"

Many self-taught engineers answer that question with "I used console.log" — which is like bringing a spoon to a database fire. The correct answer includes metrics, logs, and automated recovery.

**Example 3: N+1 query in a Django REST API**

At a European marketplace in 2023, we ran a Django API that served product listings. The endpoint `/api/products/` returned a list of products with their categories. The first query loaded all products. For each product, the code executed a separate query to load the category. The result: 100 products → 101 database queries. The 95th percentile response time was 2.3 seconds.

The fix was to use `select_related` to load products and categories in one query. The response time dropped to 150ms. The database load dropped by 90%.

The lesson was about trade-offs: `select_related` loads more data upfront, which increases memory usage. But the memory cost was acceptable because the query time improvement was massive. Those trade-offs became system design questions.

I’ve seen this exact scenario in interviews. They ask: "How would you optimize this endpoint?" The wrong answer is "Add an index." The correct answer includes query analysis, ORM optimization, and load testing.


## The cases where the conventional wisdom IS right

There are times when LeetCode-style practice is the right investment.

If you’re interviewing at a quant hedge fund or a high-frequency trading company, algorithms matter more than system design. Those companies want to know if you can write code that runs in microseconds, not if you can design a scalable API.

I interviewed at a London quant firm in 2021. The first round was 10 LeetCode hard problems in 90 minutes. The second round was a whiteboard session where they asked me to optimize a sorting algorithm for a custom data structure. The third round was a take-home assignment to implement a memory-efficient data structure in C++.

The conventional advice worked perfectly there. The company wasn’t testing system design — they were testing algorithmic depth.

Another case is when the company uses take-home assignments instead of live interviews. Many startups and mid-sized companies replace live interviews with take-home tasks. In those cases, the ability to solve complex problems offline matters more than real-time communication.

I’ve seen this at European SaaS companies. They send a task that requires 4–6 hours of work: implement a feature, write tests, and document the trade-offs. The conventional advice — practice problems, learn patterns — works well here because the task is essentially a mini-LeetCode problem.

The honest answer is that the conventional wisdom is right when the interview format rewards pure technical correctness. But if the format rewards communication, ownership, and trade-offs, the conventional wisdom falls short.

So the cases where the conventional wisdom works are: algorithm-heavy companies, take-home assignments, and roles where performance is the primary metric.


## How to decide which approach fits your situation

You have two choices: optimize for algorithms or optimize for systems. The decision depends on the company’s interviewing style and the role’s requirements.

**Rule 1: Check the company’s interview format.**

If the company does live coding on a shared editor with one interviewer, optimize for algorithms. If they do system design or behavioral rounds, optimize for systems.

I learned this when I interviewed at a US fintech company in 2022. The first round was live coding with a senior engineer. The second round was a system design round with two engineers. The third round was a behavioral round with a manager. I optimized for algorithms for the first round and failed the second and third because I didn’t practice system design.

**Rule 2: Check the role’s stack.**

If the role is backend-heavy (APIs, databases, caching), practice system design. If the role is frontend-heavy (React, state management), practice algorithms for state transitions and data structures.

At a European marketplace, the frontend role required solving state management problems like "how to sync a shopping cart across tabs without race conditions." Those are algorithmic problems disguised as state machines.

**Rule 3: Check the company’s scale.**

If the company has fewer than 50 employees, they’re more likely to test algorithms. If they have more than 200 employees, they’re more likely to test system design.

I’ve seen this at Gulf startups: small teams test coding ability; large teams test system design.

**Rule 4: Check the interview reviews.**

Glassdoor, Levels.fyi, and Blind have interview reviews. Look for phrases like "lots of LeetCode" or "system design round with three engineers." Those phrases tell you what to optimize for.

**Summary:** Pick your approach based on the company’s format, the role’s stack, the company’s scale, and the interview reviews. Don’t assume one size fits all.


## Objections I've heard and my responses

**Objection 1: "I don’t have time to learn both algorithms and system design. Should I focus on one?"

My response: Focus on system design if you’re interviewing at companies with 100+ employees. Focus on algorithms if you’re interviewing at small startups or quant firms.

I got this wrong at first. I assumed all companies cared about algorithms. I bombed interviews at European companies because I didn’t practice system design. The honest answer is that system design is harder to learn on the fly, so if you have limited time, prioritize it for larger companies.

**Objection 2: "I’m self-taught. I don’t know how real systems work. How can I practice system design?"

My response: Start with small systems you own. For example, build a URL shortener with Redis caching. Then extend it to support analytics, user authentication, and rate limiting. Each extension forces you to make trade-offs.

I started with a URL shortener on a $200/month DigitalOcean droplet. I added caching, then a database, then a rate limiter. Each step taught me a trade-off: cache invalidation vs. freshness, database connections vs. memory, rate limit storage vs. cost.

**Objection 3: "I don’t have production experience. How can I talk about incidents?"

My response: Simulate incidents. For example, write a script that simulates a database overload, then debug it using logs and metrics. Write a post-mortem as if it happened in production.

I simulated a database overload by running a load test with Locust. I wrote a post-mortem that included the root cause, the impact, and the fix. I used that post-mortem in behavioral interviews.

**Objection 4: "I’m not good at explaining. How can I improve?"

My response: Record yourself explaining technical concepts. Use a tool like OBS or your phone. Review the recording for clarity, conciseness, and completeness. Aim for 60 seconds per concept.

I recorded myself explaining how Redis eviction works. The first recording was 3 minutes long and full of tangents. After three recordings, I got it down to 50 seconds with no tangents.


## What I'd do differently if starting over

If I were self-taught again and interviewing for remote roles, I’d do three things differently.

**First, I’d build a portfolio of small systems.** Not just code — but systems with trade-offs. For example:
- A URL shortener with caching, rate limiting, and analytics.
- A real-time chat app with WebSockets, message queues, and horizontal scaling.
- A payments simulator with idempotency, retries, and rollback.

Each system would have a README that explains the trade-offs, failure scenarios, and monitoring setup. I’d host them on GitHub Pages or a personal domain. I’d link to them in every application.

I didn’t do this at first. I just built features. The portfolio forced me to think about systems, not just code.

**Second, I’d practice explaining systems, not just building them.** I’d record myself explaining each system, then review the recording for clarity and completeness. I’d aim for 60 seconds per concept.

I started with a 3-minute explanation of Redis caching. After three recordings, I got it down to 50 seconds. The improvement was dramatic.

**Third, I’d simulate interviews.** I’d use Pramp or Interviewing.io to practice live interviews. I’d focus on communication, not just correctness. I’d ask for feedback after each session.

I didn’t use these platforms at first. I relied on mock interviews with friends. The problem was that my friends weren’t engineers, so they couldn’t give me technical feedback. Pramp gave me real engineers as interviewers.

**Summary:** Build systems with trade-offs, explain them clearly, and simulate interviews. That’s the three-step process I’d follow today.


## Summary

Technical interviews for remote roles test two skills: problem-solving under constraints and clear communication. Most self-taught engineers optimize for the first and ignore the second, which explains why many pass algorithm rounds but fail system design or behavioral rounds.

The conventional advice — grind LeetCode, memorize patterns — is half-right. It works for algorithm-heavy companies or take-home assignments, but it fails for system design or behavioral rounds. The honest answer is that interviews reward ownership, not just correctness.

The new mental model is: imagine you’re on-call for a service that just went down. Your job is to explain what happened, why it failed, and how you’ll prevent it next time — all while the page is still ringing. That model explains why many self-taught engineers struggle with system design questions.

To prepare, build small systems with trade-offs, explain them clearly, and simulate interviews. The systems don’t need to be production-grade — they need to teach you how to make trade-offs and explain them. The explanations don’t need to be perfect — they need to be clear and complete.

If you only do one thing after reading this, build a URL shortener with caching, rate limiting, and analytics. Host it on a $200/month DigitalOcean droplet. Write a README that explains the trade-offs. Then record yourself explaining it in 60 seconds. That single exercise will teach you more about interviews than 100 LeetCode problems.


## Frequently Asked Questions

**How many LeetCode problems should I solve to pass remote interviews?**

Solve 100–150 LeetCode problems if you’re targeting small startups or quant firms. Solve 50–100 if you’re targeting mid-sized companies with system design rounds. Focus on quality over quantity: for each problem, write a clean solution, explain the time and space complexity, and discuss edge cases.

**What’s the best way to practice system design for interviews?**

Start with small systems you own: a URL shortener, a real-time chat app, or a payments simulator. For each system, ask: What breaks? How do I recover? What are the trade-offs? Use a tool like Excalidraw to draw the architecture. Record yourself explaining it in 60 seconds. Review the recording for clarity and completeness.

**How do I explain trade-offs clearly in interviews?**

Use a simple structure: State the trade-off, explain the options, and justify your choice. For example: "We could use a cache for fast reads, but it adds memory overhead and cache invalidation complexity. We chose a cache because 90% of requests are reads, and the memory cost is acceptable." Practice this structure until it’s second nature.

**What if I don’t have production experience? How do I answer behavioral questions?**

Simulate production incidents. For example, write a script that simulates a database overload, then debug it using logs and metrics. Write a post-mortem as if it happened in production. Use that post-mortem in behavioral interviews. The goal isn’t to have real incidents — it’s to show you can think like an engineer who owns a system.


| Budget tier | Tool | Why it fits | Cost |
|-------------|------|------------|------|
| Hobbyist ($200/month) | DigitalOcean Droplet | Cheap, simple, good for small systems | $5–$20/month |
| Startup ($1,000/month) | Render | Easy deploy, built-in scaling, good for side projects | $10–$50/month |
| Enterprise ($5,000+/month) | AWS EC2 + RDS | Full control, complex systems, cost-effective at scale | $50–$500/month |
| Solo dev ($50/month) | Railway.app | Instant deploy, good for prototyping | $5–$20/month |
| Team ($200+/month) | Fly.io | Global edge network, simple deploy | $10–$100/month |