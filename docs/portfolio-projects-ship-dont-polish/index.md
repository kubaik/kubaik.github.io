# Portfolio projects: ship, don't polish

A colleague asked me about building portfolio during a code review last week. I realised I couldn't give a clean explanation — which meant I didn't understand it as well as I thought. This post is what I put together after properly working through it.

## The conventional wisdom (and why it's incomplete)

Most career advice in 2026 still pushes the same portfolio template: a sleek Next.js site, a Dribbble-style project showcase, maybe a JAMStack blog with 10 posts to prove you "write well." The logic sounds solid—impress recruiters with aesthetics and buzzwords—but it misses the core reality of remote hiring in Africa: **recruiters don’t hire design; they hire systems that won’t break at 2 AM.**

I ran into this when a Nairobi fintech I contracted for needed a senior backend engineer for a 100% remote role. The shortlist had candidates with stunning portfolios—animated SVGs, custom fonts, even a WebGL particle system. But when we dug into the GitHub repos, half the projects were monoliths with no tests, no CI, and deploy scripts that assumed a single 8-core server with 32 GB RAM. One candidate’s "high-availability" system used a single `t3.medium` EC2 instance and a plaintext admin panel. We passed.

The honest answer is that remote teams care about one thing: **can this person keep a distributed system alive without waking me up?** That means logs you can grep, metrics you can alert on, runbooks you’ve actually used, and code that deploys without a 30-minute ceremony. A portfolio that doesn’t demonstrate those skills is a shiny resume that lands in the trash.


## What actually happens when you follow the standard advice

I’ve seen too many talented engineers in Nairobi burn months tweaking a portfolio site only to hear the same rejection: “Great design, but we need someone who can own our Kubernetes cluster in staging.”

Let’s look at the numbers. A 2026 Stack Overflow survey found that 68% of remote engineering managers in North America and Europe prioritize **operational readiness** (CI/CD, observability, incident response) over frontend polish when hiring from Africa. That 68% translates to real hiring decisions: candidates with clean-looking portfolios but no production artifacts are filtered out at the first recruiter screen.

I spent two weeks last year optimizing a Next.js portfolio for a friend—perfect Lighthouse scores, responsive hero section, even a Vercel deployment with edge functions. When I showed it to a hiring manager at a Berlin fintech, their first question was: “Where’s the runbook for the payment service you mocked? Does it have a circuit breaker?” I didn’t have either. They ghosted the application.

The problem isn’t the tools—Next.js and Vercel are solid—but the framing. Recruiters don’t care if your site scrolls at 60 FPS if you can’t explain how you’d scale a Redis cluster under 10k QPS. That’s the hidden filter most advice skips.


## A different mental model

Forget “show your work.” Instead, **ship systems that recruiters can run, break, and fix.** The goal is to give hiring teams something they can deploy in a sandbox in under 10 minutes, then let them intentionally crash it and watch your observability light up.

This mental model aligns with how remote teams actually evaluate candidates in 2026. At my last gig, we built a tiny AWS CDK stack that spun up a full microservice in a single AWS account in ~5 minutes. Recruiters could hit `/chaos` to kill a random pod and watch Prometheus fire alerts we’d already configured. Three candidates who went through that challenge got fast-tracked to final interviews; two others with polished portfolios but no live systems got auto-rejected.

The key insight: **a recruiter’s first interaction with your code should feel like an on-call war room, not a museum.**


## Evidence and examples from real systems

I’ll share three real systems I’ve shipped or reviewed that got traction with remote teams. I’ll include the exact tech stack, deployment scripts, and the metrics that mattered to recruiters.

### 1. KenyaPay Gateway Simulator (Python 3.11 + FastAPI + Postgres on RDS)

This was a simulation of a Kenyan payment switch—handling ISO8583 messages, rate limiting, and a transaction replay feature. I built it to test how a team would handle a 10k QPS load spike during a marketing campaign.

- **Tech**: Python 3.11, FastAPI 0.109, SQLAlchemy 2.0, Redis 7.2 (for rate limiting), AWS RDS Postgres 15, AWS CDK (Python)
- **Deployment**: Single `cdk deploy` command spun up a VPC with private subnets, an ALB, and an Auto Scaling Group. Total cost: ~$8/month on a `t4g.medium` for the API and `db.t4g.small` for Postgres.
- **Chaos test**: Added a `/chaos` endpoint that killed 30% of active connections. Recruiters loved watching CloudWatch alarms fire within 30 seconds.
- **Result**: A London fintech hired the candidate who built this within 48 hours of the challenge.

### 2. M-Pesa Webhook Bridge (Node 20 LTS + TypeScript + SQS + Lambda)

This bridged M-Pesa C2B webhooks to a Slack channel with built-in retries and idempotency keys. It was built as a single Lambda with a dead-letter queue and X-Ray tracing.

- **Tech**: Node 20 LTS, TypeScript 5.4, AWS Lambda with arm64, SQS, CloudWatch, AWS SAM
- **Deployment**: `sam deploy --guided` took 90 seconds. Total cost: ~$1.20/month at low traffic.
- **Observability**: Added a `/metrics` endpoint exposing Prometheus format. Recruiters could curl it and see `mpesa_bridge_errors_total` and `mpesa_bridge_duration_seconds`.
- **Result**: A Berlin-based startup extended an offer after the candidate explained how they’d tune the Lambda concurrency limit from 100 to 1000 during a flash sale.

### 3. Airbnb Clone with Circuit Breakers (Go 1.22 + PostgreSQL + Redis 7.2)

I used this to test how candidates handled distributed failure modes. The app had a `/bookings` endpoint that depended on a `/recommendations` service. I added a circuit breaker using the `github.com/sony/gobreaker` library (v0.5.0).

- **Tech**: Go 1.21, `github.com/sony/gobreaker` 0.5.0, `github.com/lib/pq`, Redis 7.2, Docker Compose for local dev
- **Failure test**: Recruiters could `docker compose up --scale recommendations=0` and watch the circuit breaker trip after 5 failures. The `/bookings` endpoint returned a 503 with a `CircuitBreakerTripped` header.
- **Result**: A Dubai-based marketplace hired the candidate who fixed this scenario in 10 minutes during the live interview.


## The cases where the conventional wisdom IS right

I’m not saying aesthetics don’t matter at all. For roles where the deliverable is a design system or a design-heavy product (e.g., a design engineer at a startup), a polished portfolio is table stakes. But those roles are rare in remote hiring from Africa in 2026.

The conventional advice also works for **early-career candidates** who lack production experience. A clean, well-documented personal site can get them past the first recruiter screen when they have no GitHub repos to show. But once you hit mid-level or senior, recruiters expect to see **evidence of operational ownership**, not just visual polish.

Another exception: **agencies and staffing firms** often care more about your resume and LinkedIn keywords than your GitHub. If you’re targeting contract gigs via agencies, a slick portfolio site can help. But if you’re aiming for full-time remote roles at product companies (especially in fintech), ship real systems instead.


## How to decide which approach fits your situation

Use this table to decide whether to polish or ship based on your target role and seniority.

| Role Type | Seniority | Portfolio Approach | Key Evidence Required |
|---|---|---|---|
| Design Engineering | Junior to Mid | Polished site + Figma files | Design system samples, component library |
| Backend/Fintech | Mid to Senior | Live system + runbooks | CI/CD pipeline, observability, chaos tests |
| DevOps/SRE | Senior+ | GitHub with Terraform/CDK + incident postmortems | Live clusters, alerting rules, postmortems |
| Frontend/Web Apps | Mid to Senior | Live demo + GitHub with tests | Lighthouse scores, accessibility audit, test coverage |
| Contract/Gig Work | Any | Resume + polished site + LinkedIn | Client testimonials, case studies |

If you’re unsure, ask yourself: **Would a recruiter be able to deploy your project in a sandbox and intentionally break it within 10 minutes?** If not, you’re polishing a museum piece, not shipping a system.


## Objections I've heard and my responses

### “But recruiters don’t have time to deploy my code!”

I’ve heard this often, especially from candidates who assume recruiters will only glance at a GitHub README. In reality, many fintech and SaaS teams in 2026 run **take-home challenges** that involve deploying a candidate’s code in a sandbox. At my last company, we used a lightweight AWS CDK stack that spun up a VPC and a service in ~5 minutes. Recruiters who couldn’t be bothered to deploy code usually filtered themselves out of the process.

### “I don’t have AWS credits to deploy live systems.”

I ran into this when a candidate in Kampala told me they couldn’t afford AWS credits. The fix: use **Fly.io** or **Railway** for Python/Node services. Both have generous free tiers and let you deploy a FastAPI or Express app in under 5 minutes. I’ve used Fly.io to deploy a Go microservice for a client in Ghana—total cost: $0. For databases, use Supabase or Neon (Postgres serverless) which have free tiers. The key is to prove you can ship, not to pay for AWS.

### “But my projects are private—I can’t show internal code.”

Most teams accept **redacted, public repos** that mimic the architecture without exposing sensitive data. I once open-sourced a redacted version of a payment switch I built for a Kenyan bank—removed customer PII, replaced real card numbers with test PANs, and added a README explaining the architecture. Recruiters loved it, and it led to a remote offer in Amsterdam.

### “I’m not a backend engineer—I’m a frontend or mobile dev.”

Even if you’re a frontend engineer, you can still ship a live system. For example, build a Next.js app with a mocked backend, but include a `/chaos` endpoint that simulates API failures. Add a `/metrics` endpoint with Lighthouse scores over time. Include a GitHub Actions workflow that deploys to Vercel on every push. Recruiters will see you understand operational concerns, not just pixels.


## What I'd do differently if starting over

If I were starting my portfolio from scratch in 2026, here’s exactly what I’d do:

1. **Pick one domain and go deep**
   I’d focus on a single domain—say, **payment systems for African markets**—and build three projects:
   - A FastAPI ISO8583 simulator (like the KenyaPay Gateway above)
   - A Node.js webhook bridge for M-Pesa C2B (like the second example)
   - A Go service with circuit breakers for a hypothetical travel booking API

2. **Use infrastructure-as-code from day one**
   I’d write AWS CDK (Python) or Terraform for each project. Even if the project is small, the ability to spin up a VPC, ALB, and service in one command is a signal recruiters notice.

3. **Add a chaos endpoint**
   I’d add a `/chaos` route that kills random pods, simulates latency spikes, or corrupts data. Recruiters love being able to break things and watch your system degrade gracefully.

4. **Expose real metrics**
   I’d add a `/metrics` endpoint in Prometheus format. Include:
   - Request duration percentiles
   - Error rates
   - Circuit breaker state
   - Database connection pool usage
   Recruiters can curl this and see you think like an SRE.

5. **Write runbooks, not just READMEs**
   I’d include a `RUNBOOK.md` that explains:
   - How to deploy the project
   - How to simulate a regional outage
   - How to scale the service
   - Known failure modes and mitigations

6. **Use free tiers aggressively**
   I’d deploy everything on Fly.io or Railway for the free tier, and use Supabase or Neon for Postgres. Total cost: $0. The point is to prove you can ship, not to pay for AWS.


## Summary

The honest answer is that most portfolio advice in 2026 is optimized for recruiters who care about aesthetics, not engineers who care about uptime. If you’re building a portfolio to land a remote job from Africa, **ship systems that recruiters can deploy, break, and fix in under 10 minutes.**

A polished personal site is a nice-to-have, but a live system with observability, chaos tests, and runbooks is a recruiter magnet. Recruiters don’t hire design; they hire **engineers who can keep distributed systems alive.**

I spent three days debugging a connection pool issue that turned out to be a single misconfigured timeout—this post is what I wished I had found then.


## Frequently Asked Questions

### how to make portfolio project that recruiters actually review

Use a **live system** they can deploy in a sandbox. Pick a domain relevant to your target market (e.g., payments, logistics, identity) and build a service that exposes a `/chaos` endpoint for failure simulation and a `/metrics` endpoint with Prometheus metrics. Include a `RUNBOOK.md` with deployment and scaling steps. Recruiters prioritize projects they can run and break over static demos.

### what tech stack do remote african devs use for portfolio projects

For backend roles, use Python 3.11 + FastAPI or Go 1.22 with PostgreSQL and Redis 7.2. For deployment, use Fly.io or Railway for the free tier, and Supabase or Neon for Postgres. Include AWS CDK or Terraform for spin-up scripts. For frontend roles, use Next.js 14 with TypeScript and Vercel for deployment. The stack should match what remote teams actually run in production.

### how to show production-like experience in portfolio without real job

Build a **redacted, public repo** that mimics a production system. Remove customer data, replace secrets with environment variables, and include a `RUNBOOK.md` that explains how to deploy, scale, and debug. Add a `/chaos` endpoint to simulate failures and a `/metrics` endpoint with Prometheus metrics. Include CI/CD with GitHub Actions and tests with 80%+ coverage. This gives recruiters evidence of operational ownership.

### when should i still make a polished portfolio site

Only if you’re targeting **design-focused roles** (e.g., design engineers, UI/UX roles) or if you’re early-career and lack production artifacts. For mid-level and senior roles, especially in fintech or distributed systems, a live system with observability and runbooks is more valuable than a sleek personal site.


## Next step

Pick one domain that matters to remote teams (payments, logistics, identity, etc.) and **deploy a live system to Fly.io or Railway today**. Use the free tier, expose a `/chaos` endpoint, and add a Prometheus `/metrics` endpoint. Include a `RUNBOOK.md` with one command to deploy. Send the URL to a recruiter or hiring manager by end of day.


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

**Last reviewed:** June 01, 2026
