# Portfolio beats pedigree for remote roles

I've seen this done wrong in more codebases than I can count, including my own early work. This is the post I wish I'd had when I started.

## The conventional wisdom (and why it's incomplete)

Most career advice for African developers chasing remote jobs boils down to one mantra: *build projects that match what foreign companies want*. Tutorials push you to clone Stripe, build a TikTok scraper, or replicate Notion’s API. Bootcamps advertise “hands-on, industry-relevant projects” that promise to make your GitHub profile sparkle. The logic sounds airtight: *if you build what they’re hiring for, you’ll get hired.*

I’ve seen this fail when the project is technically impressive but emotionally hollow. A teammate in Lagos spent six weeks building a full-stack expense tracker with React, NestJS, and PostgreSQL—complete with role-based access, Stripe billing, and real-time charts. It looked like a classic fintech product. He applied to 80+ remote roles and heard back from exactly zero. The code was clean, the README was polished, but the projects didn’t tell a story about *him*. When I asked one recruiter why she passed, she said: “It’s well-built, but it feels like a template. Where’s the *you* in this?”

The honest answer is that most advice confuses *market fit* with *personal fit*. Remote employers care about two things: **can you do the job** and **will you fit in**? A cookie-cutter clone doesn’t prove you can do the job—it proves you can follow a tutorial. And it doesn’t prove you’ll fit in—it proves you’ll blend into the background.

I ran into this when I tried to hire a backend engineer for a Nairobi-based payments startup in 2026. Out of 120 applications, 40 had impressive GitHub profiles filled with cloned APIs. Only three had anything real to say about payments, fraud, or compliance—things we actually needed. The rest felt like cargo cult engineering. My takeaway: the portfolio that gets you hired isn’t the one that looks the most like a Y Combinator demo day project. It’s the one that screams *this person understands real problems in real systems* and *this person solves them in a way that feels authentically theirs*.

## What actually happens when you follow the standard advice

If you follow the mainstream playbook, here’s the typical outcome: you build a SaaS clone, write a glowing README, ship a live demo on Vercel or Railway, and blast your profile on LinkedIn with hashtags like #RemoteOK and #HireMe. You get a few emoji reactions and maybe a recruiter DM asking if you’re open to contract gigs at $15/hr. You optimize for GitHub stars and LinkedIn engagement, not for the kind of signals that make a hiring manager pause.

In my experience, this approach optimizes for vanity metrics, not actual hiring outcomes. I mentored a group of junior developers in Mombasa last year who collectively built five e-commerce clones. They polished each project for weeks: custom domain, CI/CD pipelines using GitHub Actions, even domain-specific Terraform modules. They shared the repos in tech communities and got hundreds of stars. When they applied to 200 remote jobs, only one interview led to an offer—and even that one fell through when the company realized their “production” stack was running on a single $5 DigitalOcean droplet with no redundancy.

The core problem isn’t the projects—it’s the *assumption* that building what foreign companies want is the same as building what proves you’re hireable. Most remote job postings in 2026 still look for candidates who can “build scalable microservices,” “optimize for low latency,” and “write clean, maintainable code.” But the *real* signal they’re searching for is: *will this person solve our problems faster than they cause new ones?*

I was surprised that most applicants who got past the first screen had something unusual: they didn’t just clone— they *extended*. One candidate built a React dashboard for a fictional bank, but then added a simulated fraud detection engine using scikit-learn. Another built a NestJS API, but included a custom middleware to log and compress GraphQL responses for a 30% latency drop. Neither project was a real product, but both proved *domain understanding* and *engineering judgment*—the two things hiring managers actually value.

The standard advice also ignores a brutal reality: **cultural fit is often decoded from personal artifacts, not technical ones**. A hiring manager in Berlin or San Francisco doesn’t just read your README—they look for clues in your commit messages, your project structure, your error handling philosophy. If your codebase is littered with `// TODO: fix later` comments and your commit log reads like a bot (“update package.json”), they’ll assume you’ll ship the same way.

And there’s the cost. Many developers burn 3–6 months building elaborate demos, only to discover that the market doesn’t reward perfection. I saw a talented engineer in Accra spend $1,200 on AWS credits building a real-time chat app with WebSockets, Redis Streams, and a React frontend—only to realize that no remote employer was hiring for WebSocket engineers in 2026. The project looked great on a resume, but it didn’t align with any actual job openings.

## A different mental model

Forget about building what foreign companies *say* they want. Instead, build what makes *you* stand out and what proves you can solve *their* problems. The mental model I now use is this: **your portfolio should be a proof-of-work system, not a proof-of-skill system.**

A proof-of-skill system answers: *can this person write code?* That’s table stakes. A proof-of-work system answers: *has this person already solved a problem similar to the one we’re hiring for, in a way that feels uniquely theirs?*

Here’s how to think about it:

- **Domain alignment**: Pick an industry or problem space that overlaps with real job postings. In 2026, fintech, healthcare APIs, and developer tooling are still hot. I’ve seen more success with candidates who built a small but real payments processor than with those who built a “TodoMVC with Docker.”
- **Authentic artifacts**: Include not just code, but traces of your thinking. Add a `/docs/decision-log.md` file where you explain why you chose PostgreSQL over MongoDB, or why you used gRPC instead of REST. Hiring managers read these like detective novels.
- **Failure stories**: Document a bug you fixed, a performance regression you diagnosed, or a security hole you closed. Include the metrics: “reduced p99 latency from 850ms to 210ms by adding a Redis cache layer with smart eviction.” Numbers impress more than adjectives.

I once hired a backend engineer in Uganda because her GitHub profile had a single repo: a lightweight GraphQL gateway for a local NGO’s health records system. It wasn’t a polished SaaS—it was a messy, real system with error logs and a hand-rolled auth layer. But it proved she understood real-world constraints: flaky networks, low budgets, and the need for audit trails. She got the job over candidates with 10x more GitHub stars.

The key insight: **remote employers care more about your ability to ship under constraints than your ability to write clean abstractions.** They want to know: *can this person build something that works in the real world, not just in a tutorial?*

This mental model also explains why most “portfolio projects” fail: they’re built in a vacuum. They don’t connect to a real user, a real pain point, or a real constraint. The best portfolios feel like diary entries from a developer who’s already solving problems similar to the ones the company faces.

## Evidence and examples from real systems

Let me show you what this looks like in practice. I’ll compare three portfolios from candidates I interviewed in 2026–2026 for remote backend roles in fintech and developer tools.

| Candidate | Project | GitHub Stars | Interview Outcome | Signal Sent |
|---------|---------|--------------|-------------------|-------------|
| A | Cloned Stripe API with NestJS, React, PostgreSQL. README: “Production-ready payments backend” | 842 | Rejected after first screen | “Looks good, but no domain depth” |
| B | Built a lightweight GraphQL gateway for a local NGO’s health records system. Added audit logs, rate limiting, and a custom retry mechanism with exponential backoff | 117 | Advanced to onsite | “Shows real constraints and trade-offs” |
| C | Created a CLI tool that generates OpenAPI specs from Django codebases, used by 3 internal teams. Includes a VS Code extension and a GitHub Action | 429 | Hired as Staff Engineer | “Proves impact and integration depth” |

What jumps out? Candidate B’s project wasn’t “production-ready”—it was *constraint-ready*. It had to run on a $20/month VPS, survive 3G networks, and log every access for compliance. Candidate A’s project was technically solid, but it didn’t prove she understood payments—she just copied Stripe’s endpoints.

I made a similar mistake early in my career. In 2019, I built a “real-time” stock trading simulator using WebSockets and Node.js. I polished it for months, added a React dashboard, and deployed it on a $50/month AWS EC2 instance. I proudly listed it on my resume. When I interviewed at a fintech startup in London, the CTO asked: “Your WebSocket server drops connections every 10 minutes during peak load. What’s your retry policy?” I didn’t have one. I had never tested it under load. I failed that interview not because my code was bad, but because I hadn’t faced the real constraints of the domain.

The lesson: **a portfolio project must include the scars of real-world usage.** That means logs, metrics, error traces, and constraints. It doesn’t have to be perfect—it has to be honest.

Another data point: in 2026, the average time-to-hire for remote backend roles in Africa is 47 days for candidates with domain-specific portfolios, versus 92 days for those with generic ones (source: RemoteTech Salary Report 2026). The difference isn’t technical skill—it’s signal clarity.

I’ve also seen candidates use open-source contributions as portfolio pieces. One hire in Rwanda contributed a bug fix to the `fastapi-users` library. He added a custom OAuth2 endpoint, wrote a migration guide, and documented the edge cases. When he applied to a startup building a developer platform, the hiring manager recognized the library—and the candidate—from previous PRs. That’s leverage.

But here’s the catch: **open source only works if it’s relevant.** Contributing to a random Python library won’t help if you’re applying to a Node.js shop. The signal has to align with the job.

## The cases where the conventional wisdom IS right

Despite my contrarian stance, there *are* scenarios where cloning a popular API or building a SaaS clone is the right move. If you’re targeting hyper-competitive roles like Full-Stack Engineer at a Series A startup, or if you’re early in your career with no professional experience, a polished clone can serve as a *training ground*.

Here’s when it makes sense:

- You’re applying to junior or mid-level roles where the bar is “can you write clean code?” rather than “can you solve domain problems?”
- The role is for a generalist position (e.g., “Full-Stack Engineer”) where domain depth isn’t critical.
- You’re pivoting into a new stack and need to prove you can write idiomatic code in that language.

I’ve seen this work for candidates targeting startups in Southeast Asia or Eastern Europe, where the hiring bar is lower and the focus is on fundamentals. One developer in Nairobi built a Trello clone with Next.js and Firebase and landed a $65k/year remote role with a Bangkok-based startup. The project wasn’t innovative, but it proved he could ship a full-stack app end-to-end.

But even then, the clone must be *flawlessly executed*. I once reviewed a candidate’s “Airbnb clone” and found 12 unhandled promise rejections, a hardcoded API key in the frontend, and a database schema that allowed SQL injection. The project looked impressive at a glance, but a 10-minute code review exposed deep gaps. That candidate never made it past the first round.

The other exception: if you’re targeting roles that explicitly require experience with a specific stack (e.g., “Must have experience with Django + Celery”), then a clone built in that stack can serve as a proxy for experience. But even then, it’s not the clone itself that matters—it’s the *evidence* that you can write production-grade code in that stack.

In short: **clones are fine for proving fundamentals, but they’re terrible for proving domain depth or engineering judgment.** Use them sparingly and only when the job posting signals that fundamentals are the primary requirement.

## How to decide which approach fits your situation

To decide whether to build a clone or a domain-driven project, ask yourself three questions:

1. **What’s the job posting asking for?**
   - If it mentions specific domains (fintech, healthcare, developer tools), prioritize domain alignment.
   - If it’s a general “Full-Stack Engineer” role, a clone might be sufficient.
   - Use tools like [RemoteOK](https://remoteok.com) or [We Work Remotely](https://weworkremotely.com) to analyze 20 recent postings for the role you want. Count how many mention domain-specific terms (e.g., “payments,” “compliance,” “real-time updates”).

2. **What’s your current signal gap?**
   - If your resume lacks any evidence of shipping to production, a clone can help—but only if it’s *real* (i.e., deployed, used, and iterated).
   - If your resume already has production experience, focus on *depth* over breadth. Add a post-mortem, a performance benchmark, or a failure story.

3. **What’s the hiring context?**
   - Startups care more about speed and adaptability. They’ll prefer a messy but real project over a polished but abstract one.
   - Established companies care more about reliability and maintainability. They’ll prefer a well-documented, constraint-aware project.

I’ve seen developers waste months building clones for roles that cared about fintech experience. One candidate built a “Robinhood clone” and applied to 120 fintech roles—only to realize that none of them wanted another trading simulator. He pivoted to building a small but real compliance tool for a local microfinance institution, added a benchmark showing a 40% reduction in API latency, and landed a $95k/year remote role within six weeks.

The other mistake is the opposite: building something too niche. I once reviewed a candidate’s portfolio with a repo called `django-pesa-api-middleware`. It was a Django middleware for a Kenyan mobile money API. It had great documentation and tests—but no one outside Kenya had ever heard of the API. The project was technically solid, but it didn’t help him get hired. The signal was too local.

So the rule is: **build for the intersection of your strengths and the market’s needs.** If the market wants fintech experience and you have a fintech problem you’ve solved in real life, build that. If the market wants generalists and you’re early in your career, build a polished clone—but deploy it, use it, and iterate on it.

## Objections I've heard and my responses

**Objection 1: “But won’t a polished clone get more GitHub stars and LinkedIn engagement?”**

I’ve seen this backfire. A candidate in Lagos built a “Notion clone” with Next.js, Supabase, and Tailwind. He got 2,100 GitHub stars and 500 LinkedIn shares. He applied to 150 remote jobs and got one interview—from a recruiter who wanted to hire him for a $12/hr contract gig. The stars meant nothing because the project didn’t prove domain depth or engineering judgment. The engagement came from developers admiring the aesthetics, not from hiring managers assessing competence.

**Response:** Stars and shares are vanity metrics. Hiring managers care about *signals*, not vanity. If your project gets stars but doesn’t align with a job posting, it’s noise.

**Objection 2: “I don’t have a real domain problem to solve. Should I just build a SaaS clone anyway?”**

This is where most advice fails. If you don’t have a real problem, don’t fake one. Instead, **find a real constraint and solve it**. For example:

- Build a CLI tool that automates a repetitive task you do every day (e.g., renaming files, scraping data).
- Build a browser extension that solves a small but annoying UX problem.
- Contribute to an open-source project in a domain you care about.

I once advised a developer in Kisumu who wanted to break into remote jobs. He didn’t have a fintech problem to solve, so he built a browser extension that auto-filled PDF forms for local SACCOs. He documented the constraints (flaky PDF libraries, inconsistent form layouts) and included a benchmark: “Reduced form-filling time from 3 minutes to 45 seconds.” He landed a $75k/year role with a European SaaS company that needed someone who understood real-world UX pain points.

**Response:** Fake problems lead to fake portfolios. Find a real constraint, even if it’s small.

**Objection 3: “Won’t hiring managers just look at my LeetCode score?”**

This depends on the company. For FAANG or high-growth startups, LeetCode is often a gatekeeper. But for most remote roles in 2026, it’s a secondary signal. I hired a backend engineer in Nairobi without a single LeetCode problem solved—she had a portfolio that proved she could debug production systems under pressure. She aced the system design round because she’d documented her own system’s failures.

**Response:** LeetCode matters more in gatekeeper rounds. But portfolio depth can override it if the signal is strong enough.

**Objection 4: “But I need to show I can write clean code. Isn’t a clone the easiest way to prove that?”**

Not necessarily. You can show clean code in a messy project. The key is *intentionality*. A candidate I worked with built a “chat app with WebRTC” that was full of race conditions and memory leaks. But she included a `/docs/errors.md` file that listed every race condition she found and fixed, along with the test cases she added. The hiring manager was impressed by the *debugging process*, not the code quality alone.

**Response:** Clean code is a hygiene factor. What matters more is *how you handle mess*—because that’s what you’ll do on the job.

## What I'd do differently if starting over

If I were starting my portfolio from scratch today, here’s exactly what I’d do:

1. **Pick a domain I actually care about**—not one I think will impress recruiters. In 2026, fintech, healthcare, and developer tools are still hiring hotspots. I’d pick one and go deep.

2. **Solve a real constraint for a real user**—even if the user is just myself or a local community. Last year, I built a small CLI tool that auto-generates Terraform modules for AWS Lambda functions. I used it daily for my own side projects. When I applied to a cloud infrastructure role, the hiring manager recognized the tool from a GitHub repo and invited me to interview.

3. **Document the trade-offs, not just the wins**—include a `/docs/post-mortems/` folder with real incidents. For example:
   ```markdown
   **Incident: Cache stampede on user profile endpoint**
   - Date: 2025-11-03
   - Impact: 95% of requests timed out during peak load
   - Root cause: Redis TTL misconfigured to 1 second
   - Fix: Added lock-based cache warming with exponential backoff
   - Metrics: p99 latency dropped from 1.2s to 210ms
   ```

4. **Deploy it and break it in public**—use a free tier on Fly.io or Railway, then intentionally load-test it. Include the error logs and the fixes. Hiring managers trust artifacts that survived real usage.

5. **Add a “How I’d improve this” section**—even if it’s aspirational. It shows humility and a growth mindset. One candidate I worked with added a section titled “What I’d do if I had a team” where he outlined how he’d split the project into microservices. The hiring manager loved it because it signaled he thought like an engineer, not just a coder.

I made a mistake early on by trying to build a “perfect” project. I spent three weeks refining a Next.js dashboard for a fake e-commerce store, optimizing every query, and polishing every UI edge case. When I interviewed at a SaaS company, the CTO asked: “What’s the most painful trade-off you made in this project?” I froze. I hadn’t documented any trade-offs—only wins. That interview ended poorly. The lesson: **perfection is the enemy of signal.**

## Summary

Your portfolio isn’t a showcase of your coding skills—it’s a proof-of-work system that answers two questions for remote employers: *Can this person solve our problems?* and *Will this person fit in?*

The conventional advice—clone a popular API, polish it, and blast it on LinkedIn—optimizes for vanity, not hiring outcomes. It leads to 0.5% conversion rates and a lot of frustration.

The better approach is to build something real, even if it’s small, that proves you understand constraints, trade-offs, and domain-specific challenges. Include artifacts that show your thinking: post-mortems, benchmarks, and failure stories. Make the hiring manager feel like they’re interviewing someone who’s already solved problems similar to theirs.

This works because remote employers care more about *evidence of impact* than *evidence of skill*. They want to know: *has this person shipped something that survived real usage?* Not: *can this person write clean abstractions?*

So if you’re building a portfolio today, start with a real constraint, not a tutorial. Deploy it. Break it. Document it. And then apply. 

That’s how you get hired—not by building what looks impressive, but by building what feels real.

## Frequently Asked Questions

**how to build a portfolio for remote jobs if I have no experience?**

Start with a tool or automation that solves a repetitive task you deal with daily. Build a CLI, a browser extension, or a small API wrapper. Deploy it for free on Fly.io or Railway. Include a README with a real problem, your solution, and the trade-offs you made. Even if it’s tiny, it proves you can ship something end-to-end. I’ve seen candidates with no professional experience land $60k/year roles with this approach.

**what makes a GitHub profile stand out to remote recruiters?**

Recruiters don’t care about stars or forks. They care about signals: clean code, intentional architecture, and evidence of real usage. A repo with 100 stars but no README, no tests, and no deployment instructions is noise. A repo with 20 stars, a `/docs/decision-log.md`, a `/post-mortems/` folder, and a deployed demo is a signal. Include a `CONTRIBUTING.md` file that explains how someone else could run your project—it shows you think like a maintainer.

**why do most portfolio projects fail to get interviews?**

Because they’re built in a vacuum. They don’t connect to a real user, a real pain point, or a real constraint. A polished TodoMVC clone doesn’t prove you can write production-grade code—it proves you can follow a tutorial. Remote employers want to see *domain depth* and *engineering judgment*, not *code cleanliness*. The best portfolio projects feel like diary entries from a developer who’s already solving the kinds of problems the company faces.

**what’s the most underrated part of a portfolio?**

The `/docs/` folder. Most candidates skip it. A well-written `README.md` and a `post-mortems/` directory tell a hiring manager more about you than 1,000 lines of clean code. Include your thought process, the trade-offs you made, and the incidents you survived. One candidate I worked with added a `docs/onboarding.md` that walked through how to deploy his project—and the hiring manager hired him on the spot because it showed he thought like a teammate.

## Next step: audit your current portfolio today

Open your main portfolio repository. Count how many of your projects include:
- A deployed demo (even if it’s on a free tier)
- A post-mortem or failure story
- A benchmark or metric (e.g., latency, error rate, cost)
- A `/docs/` folder with decision logs

If fewer than 2 out of 5 projects meet these criteria, delete or archive the ones that don’t. Then pick one project to improve today: add a post-mortem, deploy it, or write a decision log. That single change will make your portfolio 10x more hireable.

 
---
 
### About this article
 
**Author:** Kubai Kevin is a software developer based in Nairobi, Kenya with 10+ years of experience building production Python and Node.js backends, primarily in fintech. He has worked with teams in East Africa, Europe, and Southeast Asia on systems handling millions of requests per day. [More about the author →](/about/)
 
**Editorial process:** Articles on this site are based on direct production experience and verified against official documentation before publishing. Code examples are tested locally. If you find a factual error, [please reach out](/contact/) — corrections are applied within 48 hours.
 
**Last reviewed:** May 2026
