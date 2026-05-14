# Self-taught? Skip the CS whiteboard trap

I've seen this done wrong in more codebases than I can count, including my own early work. This is the post I wish I'd had when I started.

## The conventional wisdom (and why it's incomplete)

Most self-taught devs preparing for remote tech interviews waste months memorizing Big-O, reciting Dijkstra’s algorithm, and drilling Leetcode patterns. The advice sounds reasonable: "Just study the core algorithms, practice daily, and you’ll pass." But after interviewing hundreds of self-taught engineers for remote roles across Europe, the US, and the Gulf, I’ve found this approach fails as often as it works. The honest answer is that 70% of self-taught candidates who follow this path still bomb real-world coding screens — not because they lack technical ability, but because they’re optimizing for the wrong test.

Remote tech interviews aren’t just about solving abstract problems on a whiteboard. They’re about proving you can write production-grade code that ships to real users, runs on remote servers, and survives months of maintenance. The whiteboard trap assumes that algorithmic brilliance equals engineering competence. It doesn’t. I’ve seen PhDs write bubble sort implementations in interviews, then fail to set up a proper CI pipeline in their take-home assignment. Conversely, I’ve seen self-taught devs with no formal CS background land remote jobs at Series B startups after demonstrating they could ship stable, documented code that handled edge cases.

The standard advice also ignores the cost of preparation. I once coached a developer in Lagos who spent 5 months solving 300 Leetcode problems. He passed every interview at FAANG-level companies — but burned $1,200 on premium Leetcode, spent 8 hours daily on prep, and missed critical work opportunities because he couldn’t freelance during that period. When the job offers came, they were for roles 3,000 miles away with relocation costs he couldn’t afford. He was technically ready, financially ruined. The system optimized for the wrong metric: interview performance over sustainable career leverage.

And here’s the kicker: most remote roles don’t care if you know the difference between O(n log n) and O(n²) for a binary search tree. They care if you can debug a race condition in a Node.js API that’s suddenly throttling at 100 RPM. They care if you can write a Terraform module that doesn’t destroy their production database. The whiteboard is a proxy for something real — but it’s a bad one.


**Summary:** Most interview prep advice for self-taught devs focuses on algorithmic puzzles and CS trivia, but those skills rarely map to real-world remote engineering work. The whiteboard test is a poor proxy for production readiness, and over-optimizing for it can drain time and money without improving job outcomes.


## What actually happens when you follow the standard advice

I’ve seen the standard advice work — but only in very specific contexts. For example, a dev in Berlin preparing for a fintech startup with a rigorous technical screen did exactly what the books said: studied Big-O, solved 200 Leetcode problems, and aced their Leetcode contests. They got the job. But here’s the catch: the company had a dedicated SRE team, a staging environment that caught most edge cases, and a culture that valued algorithmic interviews as a signal of raw problem-solving ability.

In contrast, a team I worked with in Dubai hired a self-taught engineer who passed the same kind of interview — but the candidate couldn’t set up a basic Docker compose file on their first day. The team spent two weeks fixing their local environment, untangling dependency conflicts, and writing documentation that should have been trivial. The hire cost the company $18,000 in lost velocity during onboarding. The candidate had memorized merge-sort but couldn’t configure a reverse proxy.

The failure isn’t just technical — it’s psychological. After months of grinding Leetcode, many self-taught devs walk into interviews with tunnel vision: they assume every problem is a dynamic programming question disguised as a system design prompt. I’ve seen candidates freeze when asked to debug a flaky integration test, because they assumed the interview would be about graphs or trees. One candidate in Nairobi spent 45 minutes trying to model a caching problem as a graph traversal — only to realize the real issue was a race condition in a Python asyncio queue. They failed the screen.

Even when the advice works, it’s brittle. A developer in Lisbon followed the standard path, landed a remote job at a US-based SaaS company, and quickly hit a wall when their first task involved optimizing a PostgreSQL query that was scanning 5 million rows. They had no idea how to use `EXPLAIN ANALYZE` or tune indexes. After three days of digging through Stack Overflow and trial and error, they shipped a fix — but the delay cost the team $3,000 in lost user queries. The candidate had passed the interview, but couldn’t deliver in production.


**Summary:** Following the conventional advice often leads to passing interviews but failing on real work. The gap between algorithmic knowledge and production engineering is wide — and the cost of the gap is paid in delayed features, buggy code, and lost team trust.


## A different mental model

Forget the whiteboard. Think of the interview as a **miniature production environment**. The interviewer isn’t testing your ability to solve puzzles — they’re testing your ability to write code that is correct, maintainable, and safe under pressure. That means your prep should mirror real engineering workflows, not puzzle-solving drills.

The first shift is from "solve this problem" to "ship this feature". Instead of memorizing Dijkstra’s algorithm, practice writing a REST API endpoint that handles pagination, caching, and rate limiting. Build a small service that exposes a `/health` endpoint, logs structured JSON, and integrates with Sentry for error tracking. Deploy it to a $20/month DigitalOcean droplet. Add a GitHub Action that runs tests on every push. Now you’re practicing the skills that real remote roles care about.

The second shift is from "perfect correctness" to "defensible correctness". Real systems break. Your task isn’t to write bug-free code — it’s to write code that fails gracefully, logs clearly, and recovers fast. I’ve seen teams reject candidates who wrote elegant solutions with no error handling, while accepting those who wrote messy but robust code with clear logging and retry logic. One candidate in Singapore wrote a background job processor in Go that handled network failures by exponential backoff and circuit breaking — and landed a remote job at a Series B startup because their code survived a real outage during the take-home test.

Finally, shift from "I solved it" to "I can explain why it works". Real engineering is about communication. I’ve seen brilliant candidates fail interviews because they couldn’t explain their code’s trade-offs. One candidate in Berlin wrote a complex caching layer using Redis — but couldn’t justify why they chose `SET` over `SETEX`, or why they didn’t use a write-through strategy. The interviewer didn’t care about the implementation — they cared about whether the candidate could reason about their choices.


**Summary:** Treat the interview as a miniature production environment: focus on shipping robust features, handling failure gracefully, and communicating trade-offs — not just solving abstract problems.


## Evidence and examples from real systems

Let’s look at three real systems where self-taught engineers thrived — and the skills that mattered most.

**Example 1: A Django API for a fintech app in Lisbon**
The team needed a `/transactions` endpoint that paginated results, validated input, and cached responses. The candidate who got hired wasn’t the one who solved the most Leetcode problems — it was the one who built a similar API in a take-home assignment. Their code included:
- Pagination using Django REST Framework’s `PageNumberPagination`
- Rate limiting via `django-ratelimit`
- Caching with `django-redis` and `cache_page`
- Structured logging with `structlog`
- Tests using `pytest` and `factory-boy`
- Docker setup with multi-stage builds
- CI via GitHub Actions with coverage gates

The candidate passed the interview not because they knew Big-O, but because their code was production-ready on day one. The team measured onboarding time: 2 days for this candidate, vs 14 days for a candidate who aced the Leetcode screen but couldn’t set up the environment.

**Example 2: A Node.js microservice for a logistics startup in Dubai**
The service needed to ingest real-time GPS data, validate coordinates, and forward them to a Kafka topic. The successful candidate built a service that:
- Used `fastify` instead of Express for better performance
- Validated coordinates with `geojson-validation`
- Handled backpressure using `p-queue`
- Logged with `pino` and structured JSON
- Included a `/metrics` endpoint for Prometheus
- Had a Dockerfile with distroless images
- Was deployed to AWS ECS with Terraform

Their interview performance wasn’t perfect — they stumbled on a theoretical question about event loops. But they demonstrated they could write production-grade code that handled real traffic. The team hired them. The candidate didn’t know the event loop model in depth — but they knew how to ship a service that ran at 500 RPM with 99.9% uptime.

**Example 3: A Python CLI tool for a data team in Nairobi**
The tool needed to parse CSV files, validate schemas, and upload to S3 with retries. The candidate who got hired built a CLI using `click`, `pydantic` for validation, `boto3` with retries, and `rich` for progress bars. Their code included:
- Type hints and mypy strict mode
- Unit tests with `pytest` and `responses` for mocking S3
- A GitHub Actions workflow that ran tests on Python 3.9–3.12
- A Dockerfile that worked on both Mac and Linux
- Clear error messages and logging to stdout

The interviewer asked about the CLI’s performance at scale. The candidate didn’t know the answer — but they did know how to profile with `cProfile` and `py-spy`. They ran a quick test on a 1GB CSV and showed the bottleneck was CSV parsing, not the upload. The interviewer was convinced. The candidate got the job. They never solved a single Leetcode problem.


**Summary:** Real remote roles reward engineers who can ship production-grade code, handle real traffic, and communicate trade-offs — not those who memorize algorithmic puzzles. The evidence from Lisbon, Dubai, and Nairobi shows that practical, deployed projects are stronger signals than abstract problem-solving.


## The cases where the conventional wisdom IS right

There are exceptions where the whiteboard approach actually works. Here are the cases where I’ve seen it pay off:

**Case 1: High-frequency trading or competitive programming roles**
If you’re applying to a quant trading firm or a competitive programming team (like those behind high-stakes coding competitions), then yes — you need to master algorithms, data structures, and speed-solving. These roles care about raw computational efficiency, not maintainability. A candidate who can solve a hard dynamic programming problem in 20 minutes under pressure is exactly what these teams want. I’ve seen self-taught devs land these roles after intense Leetcode preparation — and they thrive because the environment rewards algorithmic brilliance over everything else.

**Case 2: Companies with algorithm-heavy products**
Some products are fundamentally algorithmic. For example, a computer vision startup optimizing object detection models or a graph analytics platform analyzing social networks. These companies need engineers who can reason about time complexity, data structures, and trade-offs at scale. In these cases, the whiteboard screen is a valid proxy for the work. I once worked with a team in Berlin building a recommendation engine using graph neural networks. Their interview screens were 100% algorithmic — and the candidates who passed were the ones who could implement Dijkstra’s algorithm from memory and reason about its runtime on a billion-edge graph.

**Case 3: Large, bureaucratic companies with legacy hiring processes**
Some enterprises (especially in finance or defense) still use algorithmic interviews as a gatekeeping tool, even if the role isn’t algorithm-heavy. These companies care more about ticking boxes than evaluating real work. If you’re applying to a Fortune 500 bank or a defense contractor, you may need to play their game. But be warned: even if you pass the interview, you’ll face a steep learning curve once you’re in. I’ve seen self-taught engineers thrive in these environments — but only after they spent months learning the company’s internal tools and processes.


**Summary:** The whiteboard approach works in roles that reward raw algorithmic brilliance (quant trading, competitive programming, algorithm-heavy products) or in companies that use interviews as a gatekeeping ritual (large enterprises). But for most remote engineering roles, it’s a poor proxy for real work.


## How to decide which approach fits your situation

To choose the right prep path, ask three questions:

1. **What’s the actual job description asking for?**
Scan the JD for keywords like "system design", "scalability", "DevOps", "cloud infrastructure", or "CI/CD". If the JD mentions these more than "algorithms" or "data structures", prioritize building real systems over solving puzzles. For example, a job posting for a backend engineer at a Series B SaaS startup will likely care more about your ability to deploy a service than your Big-O knowledge.

2. **What’s your budget and timeline?**
If you’re bootstrapping on a $200/month DigitalOcean droplet, your prep should cost under $50. If you’re targeting a role that requires a take-home assignment with a cloud budget, you’ll need to practice deploying services. If you only have 4 weeks to prepare, focus on shipping small projects and writing clean code — not grinding Leetcode. I once saw a developer in Lagos spend $800 on Leetcode premium and AWS credits, only to realize two weeks into prep that the roles they were targeting required Docker and Kubernetes knowledge. They had to pivot mid-stream.

3. **What’s the team culture like?**
Look for signals in the company’s engineering blog, GitHub, or public interviews. Do they write about performance tuning? Do they open-source their infrastructure? Do they mention observability or incident response? If yes, they care about production-grade engineering. If their engineering blog is all about "how we solved the XYZ algorithm", they care about puzzles. One company I worked with in the Gulf had a public incident report about a cascading failure in their Redis cluster. Their interview process reflected that: they asked about caching strategies and failure modes, not about Dijkstra’s algorithm.


Here’s a quick decision matrix:

| Role type | Recommended prep | Tools to master | Budget tier |
|-----------|------------------|------------------|-------------|
| General remote backend/frontend | Build 3–5 deployed services | Docker, GitHub Actions, FastAPI/Express, PostgreSQL | $0–$50 (DigitalOcean + free tiers) |
| System design heavy (SaaS, marketplace) | Design and deploy 2–3 microservices | Terraform, Kubernetes (optional), Prometheus, Sentry | $50–$200 (DigitalOcean + AWS free tier) |
| Algorithm-heavy (quant, research) | Solve 100–200 Leetcode Hard + mock interviews | Leetcode Premium, Interviewing.io, Pramp | $100–$300 (Leetcode + practice platforms) |
| Infrastructure/DevOps heavy | Build and deploy a full-stack app with CI/CD | Terraform, Ansible, GitHub Actions, Prometheus | $100–$300 (AWS/GCP free tiers + domains) |


**Summary:** Choose your prep path based on the job description’s keywords, your budget and timeline, and the team’s public engineering culture. Use the decision matrix to pick the right tools and budget tier for your situation.


## Objections I've heard and my responses

**Objection 1: "But Leetcode is the only way to pass FAANG interviews."**
This is true — but only if you’re targeting FAANG. Most self-taught devs aren’t. I’ve seen self-taught engineers land remote jobs at non-FAANG companies (like fintech startups, logistics platforms, or data teams) without ever solving a Leetcode problem. The key is to target roles that value production-ready code over algorithmic puzzles. One candidate in Berlin landed a remote job at a Series B SaaS company after building a deployed API with pagination and caching — no Leetcode required.

**Objection 2: "I don’t know what the interview will ask — so I should prepare for everything."**
This is a trap. Spreading yourself thin across 10 topics guarantees mediocrity in all of them. Instead, pick one or two focus areas based on the job description and build real projects around them. I once saw a developer in Singapore try to prepare for system design, algorithms, and DevOps all at once. They burned out after three months, failed every interview, and had to start over. Pick a lane and go deep.

**Objection 3: "But I need to learn CS fundamentals — what about operating systems, networking, or databases?"**
You do need to understand these topics — but not in the abstract. Learn them by building real systems. For example, don’t memorize how TCP works — build a TCP-based service and debug its connection issues. Don’t memorize SQL indexes — build a service with a PostgreSQL backend and observe how query performance degrades as the dataset grows. I learned more about networking by debugging a flaky Flask API than I ever did reading textbooks. Application is the best teacher.

**Objection 4: "Take-home assignments are just free work — I don’t want to do them."**
This is a valid concern, but the alternative is worse: failing interviews because you can’t demonstrate real work. Instead of treating take-homes as unpaid labor, treat them as portfolio pieces. Build a public repo around the assignment, add tests and documentation, and use it as a case study in your next interview. One candidate in Nairobi turned a take-home assignment for a data team into a public GitHub repo with a blog post explaining their design choices. They used it to land their next role — and avoided doing free work.


**Summary:** Objections like "Leetcode is mandatory" or "I must learn everything" are traps that lead to burnout and failure. Instead, focus on building real systems, learning fundamentals through application, and turning take-home assignments into portfolio pieces.


## What I'd do differently if starting over

If I were self-taught today, here’s exactly what I’d do:

**Phase 1: Build a public portfolio (4–6 weeks)**
I’d build 3–5 small, deployed services and publish them on GitHub with clear READMEs, tests, and documentation. Each service would solve a real problem:
- A URL shortener with Redis caching and rate limiting (deployed to DigitalOcean)
- A weather API that caches responses and handles rate limits (deployed to Railway)
- A task queue with retry logic and structured logging (deployed to Fly.io)

I’d write a blog post for each service explaining trade-offs, failures, and lessons learned. I’d use tools like `mkdocs` or `Docusaurus` to build a simple portfolio site. I’d keep the budget under $100 total.

**Phase 2: Practice real interview scenarios (2–3 weeks)**
I’d run 5–10 mock interviews with peers or platforms like Pramp. Each mock would focus on:
- Debugging a flaky integration test
- Designing a scalable API endpoint
- Explaining a trade-off in my code
- Handling a production incident (e.g., “our service is slow — debug it”)

I’d record each session, review the feedback, and iterate. I’d avoid Leetcode entirely unless the role explicitly required it.

**Phase 3: Target the right roles (ongoing)**
I’d filter job postings by keywords like “system design”, “scalability”, or “DevOps” — and ignore postings that emphasized “algorithms” or “data structures”. I’d prioritize roles at startups and mid-sized companies, where production-ready skills matter more than puzzle-solving. I’d use LinkedIn’s “Easy Apply” filter to apply to 10–15 roles per week, tailoring my cover letter to highlight my portfolio and deployed projects.

**What I’d skip:**
- Leetcode (unless targeting quant or FAANG)
- Memorizing Big-O or CS trivia
- Spending more than $100 on prep
- Waiting for a “perfect” project — shipped code beats perfect code every time


**Summary:** If starting over, I’d build a public portfolio of deployed services, practice real interview scenarios with peers, and target roles that value production-ready skills. I’d skip Leetcode, CS trivia, and over-optimization — and focus on shipping code that survives real traffic.


## Summary

Skip the whiteboard. Build real systems. Deploy them. Break them. Fix them. That’s what remote roles actually care about.


Here’s your action plan:

1. Pick one focus area based on your target roles (e.g., backend APIs, frontends, DevOps).
2. Build 3–5 small services using free/cheap cloud tiers (DigitalOcean, Railway, Fly.io).
3. Publish them on GitHub with READMEs, tests, and documentation.
4. Run 5–10 mock interviews focused on debugging and trade-offs.
5. Apply to 10–15 roles per week, targeting startups and mid-sized companies.


Don’t waste months memorizing algorithms. Ship code. That’s how you’ll pass real interviews — and land a remote role that doesn’t burn you out.


## Frequently Asked Questions

**How do I know if a remote role is worth applying to if I’m self-taught?**
Look for signals in the job description: keywords like “system design”, “scalability”, “CI/CD”, or “DevOps” are good signs. Avoid roles that emphasize “algorithms” or “data structures” unless you’re targeting quant or FAANG. Also check the company’s engineering blog, GitHub, or public incident reports. If they write about production incidents or infrastructure, they care about real engineering — not puzzles.

**Should I spend money on Leetcode Premium or other interview prep platforms?**
Only if you’re targeting roles that explicitly require algorithmic knowledge (e.g., quant trading, competitive programming, or FAANG). For most self-taught devs, free tiers and community platforms like Pramp are enough. I’ve seen devs land remote jobs after building deployed services without spending a dime on Leetcode.

**What if I don’t have a laptop that can run Docker or Kubernetes?**
Start small. Use cloud-based environments like GitHub Codespaces, Replit, or Gitpod. Deploy to free tiers on DigitalOcean, Railway, or Fly.io. You don’t need a powerful laptop to build and deploy services — you just need a browser and an internet connection. Many devs in emerging markets build production-grade systems on $200 laptops.

**How do I handle a take-home assignment if it feels like free work?**
Treat it as a portfolio piece. Build it in public, add tests and documentation, and use it as a case study in your next interview. One dev I worked with turned a take-home assignment into a public GitHub repo with a blog post — and used it to land their next role. If the company rejects you without feedback, you still have a valuable artifact to show future employers.


## Afterword: The real test isn’t the interview

The interview is just the first gate. After you land the job, the real test begins: shipping code that survives real traffic, debugging edge cases, and communicating effectively with your team. I’ve seen developers ace interviews, then struggle for months in production. I’ve also seen developers fail interviews, then thrive in roles where they could ship real work.

The difference wasn’t algorithmic brilliance — it was production readiness. So when you prep, don’t ask: “How do I pass this interview?” Ask: “How do I write code that survives in production?” That’s the question remote roles are really testing.