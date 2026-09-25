# Portfolio projects hiring managers skip

After enough code that touches actually gets gets reviewed, the same failure pattern keeps showing up. This is the writeup with the mistakes left in, not edited out. The failure is quiet — no errors, just wrong answers.

## The conventional wisdom (and why it's incomplete)

Ask ten developers what gets a portfolio project noticed and you'll get the same list: build something real, deploy it, write tests, add a README, use a modern stack. This advice is not wrong. It's just incomplete in a way that matters enormously in 2026, because the bar it sets is now the floor, not the ceiling.

The standard advice assumes the bottleneck is technical competence. It isn't. For most junior and mid-level candidates, the bottleneck is signal-to-noise. A hiring manager reviewing 80 applications for one backend role does not have 40 minutes per candidate. They have roughly 90 seconds for the first pass, and the portfolio project is usually the second thing they look at, after the resume's most recent job title. If your project doesn't answer a specific question in that window, it gets filed under "probably fine" and never revisited.

The conventional wisdom also quietly assumes that complexity equals credibility. Build a microservices app, add Kafka, add Kubernetes, add a GraphQL gateway. This produces projects that look impressive in a screenshot and terrifying in a code review, because the author had to learn six new systems simultaneously and the seams show. A common failure mode here is a project with a beautiful architecture diagram and a `docker-compose.yml` that only works on the author's machine because three services were never actually wired together.

So the conventional advice isn't wrong, it's just answering the wrong question. The question isn't "is this project technically impressive?" The question is "does this project let a stranger trust my judgment in under two minutes?" Those are different questions, and they have different answers. The part that trips people up is that the second question is about legibility, not capability, and that's what this post actually covers.

## What actually happens when you follow the standard advice

You build the thing. You deploy it to a free tier. You write a README that says "A full-stack app for tracking X." You link it on your resume. Then you wait.

What the reviewer actually does is this: they click the GitHub link. They see the repo name and the description. They scroll the file tree for about four seconds. They click into one or two files, usually the entry point or the largest file. They check the commit history. They look for a live demo link. If any of those four checks produce friction, they leave.

That friction is where most projects die. The repo has a `src` folder with 47 files and no obvious entry point. The largest file is 1,200 lines because everything got dumped into `main.py`. The commit history is one commit called "initial commit" or, worse, 200 commits all called "fix" or "update." The live demo link is dead because the free tier spun down, or it's behind a login the reviewer doesn't have credentials for.

None of this reflects on the developer's actual ability. A perfectly competent engineer can produce a repo that fails all four of these checks. But the reviewer cannot distinguish "competent engineer who didn't think about legibility" from "incompetent engineer." They have 90 seconds. They will assume the worst, because the cost of a bad hire is asymmetric.

There's a second failure mode that's more subtle. The project is technically excellent but solves a problem the reviewer doesn't recognize. A distributed task queue with custom backpressure handling is genuinely hard to build. But if the README opens with "A distributed task queue," the reviewer has to do work to understand why it matters. If it opens with "A task queue that survives worker crashes without dropping jobs, built because our cron-based retry system lost 0.3% of jobs per week," the reviewer immediately understands the problem and the solution. Same code, different legibility.

The standard advice produces projects that are technically real but narratively invisible. That's the gap.

## A different mental model

Treat the portfolio project as a compressed case study, not a demo. A case study has four parts: a specific problem, a specific constraint, a specific decision, and a measurable outcome. A demo has one part: the thing works.

This reframe changes what you build and how you present it. Under the case study model, a 400-line project with a clear problem statement beats a 4,000-line project with no narrative. Under the demo model, the opposite is true, because more code looks like more work.

The case study model also forces you to answer questions you'd otherwise dodge. What was the actual constraint? Was it latency? Cost? A flaky third-party API? A specific compliance requirement? Constraints are what make engineering interesting, and constraints are what reviewers remember. "I built a REST API" is forgettable. "I built a REST API that had to stay under 200ms p99 on a $5/month hosting budget, which meant choosing SQLite over Postgres and caching aggressively" is memorable, because it shows a tradeoff and a reason.

This is also where the reviewer's actual job enters the picture. Hiring managers are not evaluating your ability to write code in the abstract. They are trying to predict whether you will make good decisions on their team, with their constraints, under their deadlines. A project that shows one well-reasoned tradeoff is more predictive than a project that shows ten technologies used competently. The tradeoff reveals judgment. The technology list reveals only exposure.

So the mental model is: build one thing, constrain it honestly, make one or two decisions visible, and measure something. Everything else is decoration.

## Evidence and examples from real systems

Consider two projects that a reviewer might see in the same afternoon. Both are deployed, both have tests, both use reasonable stacks. The difference is entirely in legibility.

Project A is a URL shortener built with FastAPI 0.115, Redis 7.2, and Postgres 16. It has 2,400 lines across 30 files. The README is four paragraphs describing the architecture. The live demo is up. The commit history is 60 commits with messages like "add endpoint" and "fix bug."

Project B is also a URL shortener, built with the same stack. It has 900 lines across 12 files. The README opens with: "URL shortener optimized for the read-heavy case: 95% of requests are redirects, 5% are creates. Chose Redis as the primary store with Postgres as a write-behind log, which trades durability on the last 5 seconds of writes for a 10x reduction in p99 redirect latency (from ~40ms to ~4ms in local benchmarks)." The commit history has 25 commits, each scoped to one change, with messages like "swap primary store to Redis, keep Postgres as write-behind."

Project B gets the interview. Not because it's better code, but because it demonstrates three things in the first paragraph: the developer understood the workload, made a non-obvious decision, and measured the result. Project A might be better code. The reviewer will never find out.

Here's a concrete example of the kind of decision that reads well, expressed in code. This is the kind of thing that belongs in a README or a short design note, not buried in a commit:

```python
# cache.py — read-through cache with explicit staleness bound
# Decision: accept up to 5s of staleness on reads to keep p99 under 5ms.
# Tradeoff: a redirect may point to the old target for up to 5s after an update.
# Measured: p99 redirect latency dropped from 38ms (Postgres) to 4ms (Redis).

import redis
import json
from datetime import datetime, timedelta

r = redis.Redis(host="localhost", port=6379, decode_responses=True)
STALENESS_BOUND = timedelta(seconds=5)

def get_target(slug: str) -> str | None:
    raw = r.get(f"url:{slug}")
    if raw:
        entry = json.loads(raw)
        age = datetime.utcnow() - datetime.fromisoformat(entry["cached_at"])
        if age < STALENESS_BOUND:
            return entry["target"]
    # fall through to Postgres on miss or stale entry
    row = db.fetchone("SELECT target FROM urls WHERE slug = %s", (slug,))
    if row:
        r.setex(f"url:{slug}", 300, json.dumps({
            "target": row["target"],
            "cached_at": datetime.utcnow().isoformat(),
        }))
        return row["target"]
    return None
```

The code is not the point. The comment is the point. The comment tells the reviewer that the developer thought about staleness, picked a bound, and measured the effect. That's the signal.

The same principle applies to failure handling. A project that demonstrates awareness of a specific failure mode is more credible than one that doesn't. For example, a common trap in any service that calls a third-party API is that the third party will eventually return a 429 or a 503, and naive retry logic will make the problem worse. A project that shows a bounded retry with jitter and a circuit breaker is demonstrating operational awareness, not just coding ability.

```javascript
// retry.js — bounded retry with jitter and circuit breaker
// Decision: cap retries at 3, add full jitter, open the circuit after 5
// consecutive failures for 30s. Prevents thundering herd on a degraded dep.
// Observed in testing: naive retry produced 12x the request volume during
// a simulated 503 storm; this version produced 1.4x.

const failures = { count: 0, openedAt: null };
const CIRCUIT_THRESHOLD = 5;
const CIRCUIT_COOLDOWN_MS = 30_000;

async function callWithRetry(fn, maxAttempts = 3) {
  if (failures.count >= CIRCUIT_THRESHOLD) {
    const elapsed = Date.now() - failures.openedAt;
    if (elapsed < CIRCUIT_COOLDOWN_MS) {
      throw new Error("circuit open");
    }
    failures.count = 0;
  }
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      const result = await fn();
      failures.count = 0;
      return result;
    } catch (err) {
      if (attempt === maxAttempts) {
        failures.count += 1;
        if (failures.count >= CIRCUIT_THRESHOLD) failures.openedAt = Date.now();
        throw err;
      }
      const backoff = Math.min(1000 * 2 ** attempt, 8000);
      const jitter = Math.random() * backoff;
      await new Promise((res) => setTimeout(res, jitter));
    }
  }
}
```

Again, the code is ordinary. The comment and the observed number are the signal. A reviewer reading this learns that the developer understands that retries can amplify load, and that they tested the behavior rather than assuming it. That is a hiring signal.

## The cases where the conventional wisdom IS right

There are real situations where the standard advice is exactly correct, and it's worth being honest about them.

First, if you are applying for a role where the team's stack is highly specific and the employer is explicitly screening for it, matching that stack matters. A fintech team hiring a Go developer will look more favorably on a Go project than a Rust one, even if the Rust project is better. This is not irrational. It reduces onboarding cost, and onboarding cost is real. In these cases, build in the stack they use, and don't overthink it.

Second, if you are early enough in your career that you have no production experience at all, the conventional advice is doing important work: it's forcing you to finish something and deploy it. Finishing and deploying are non-trivial skills that many candidates lack. A deployed, working project is a genuine signal, even if the narrative is weak. The case study framing is an improvement on top of that, not a replacement for it.

Third, some roles genuinely reward breadth. A platform or DevOps role may value a project that touches Kubernetes, Terraform, and CI/CD, because the job is about integrating systems. In those cases, the complexity is the point, and the reviewer is looking for evidence that you can hold multiple systems in your head at once.

Fourth, and this is the one people miss: if your project is being reviewed by a technical screener rather than a hiring manager, the calculus shifts. A senior engineer doing a code review will read more of your code and care more about structure, tests, and edge cases than about narrative. For that audience, the conventional advice is closer to correct. The case study framing still helps, but it's not the deciding factor.

The honest position is that the conventional advice is right about half the time, and the case study framing is right for the other half. The skill is knowing which situation you're in.

## How to decide which approach fits your situation

Ask three questions about the role you're targeting.

Who screens first? If it's a recruiter or a hiring manager without deep technical background, legibility wins. If it's a senior engineer doing a technical screen, depth wins. Most companies do both, in that order, which means legibility gets you to the technical screen and depth gets you through it. You need both, but the order matters.

What is the team's biggest pain point right now? If the job posting mentions reliability, show a project that handles a specific failure mode. If it mentions performance, show a project with a measured latency number. If it mentions scale, show a project that had to make a tradeoff because of scale. The project should mirror the pain.

How much time do you have? A full case-study treatment takes maybe 4 to 6 hours on top of the project itself: writing the README, cleaning the commit history, adding the design note, recording a short demo. If you have a weekend, do it. If you have an evening, at minimum fix the README and the commit history, because those are the two highest-leverage changes per minute spent.

Here's a comparison of the two approaches across the dimensions that matter:

| Dimension | Demo approach | Case study approach |
|---|---|---|
| Time to build | 20–40 hours | 20–40 hours (same project) |
| Time to present | ~30 minutes | 4–6 hours |
| Lines of code | Often 2,000+ | Often under 1,000 |
| Reviewer time to "get it" | 3–5 minutes | 30–60 seconds |
| Signals judgment | Weakly | Strongly |
| Signals raw capability | Strongly | Strongly, if code is clean |
| Works for recruiter screen | Poorly | Well |
| Works for technical screen | Well | Well, if depth is real |
| Risk of overclaiming | Low | Medium (if you exaggerate) |

The last row is the one to watch. The case study framing rewards specificity, and specificity is easy to fake badly. If you claim a 10x latency improvement, a technical reviewer will ask how you measured it, and "it felt faster" ends the conversation. Only claim numbers you actually produced, and be ready to explain the method.

## Common objections, and responses

"This sounds like marketing, not engineering." It is partly marketing, and that's fine. A portfolio project is a communication artifact. The code is the evidence; the README is the argument. Pretending the argument doesn't matter is how good engineers get passed over for worse ones who communicate better. You can be both rigorous and legible.

"I don't have time to write a design note." You don't need a design note. You need three sentences at the top of the README: the problem, the constraint, the decision. That's 10 minutes. If you can't spare 10 minutes, the project wasn't going to get noticed anyway.

"My project doesn't have interesting tradeoffs." Every project has tradeoffs. You chose a database. You chose a deployment target. You chose to cache or not to cache. The tradeoff exists; you just haven't articulated it. Articulating it is the exercise.

"Won't this make my project look simple?" Yes, and that's usually an improvement. A reviewer who understands your project in 30 seconds is more likely to dig into the code than one who is still confused after 3 minutes. Simplicity is legible. Legibility is the goal.

"What if the reviewer doesn't read the README?" Some won't. But the ones who do are the ones whose opinion matters, and the README is also what gets pasted into Slack when a hiring manager asks a senior engineer "is this person worth a call?" The README travels further than you think.

## What the alternative approach would change

The practical changes are smaller than the philosophical ones. You'd stop adding technologies and start adding reasoning. You'd stop measuring lines of code and start measuring one thing: the time it takes a stranger to understand what you built and why.

You'd also change what you build next. Instead of "what's a cool project?" the question becomes "what's a project where I can make one interesting decision and measure its effect?" That's a much easier question to answer, and it produces better projects. A small tool that solves a specific annoyance with a measured improvement is a better portfolio piece than a sprawling app with no thesis.

You'd change your commit history. Instead of one commit at the end, you'd commit in scoped chunks with messages that describe the change and the reason. This is not just for the reviewer; it's how you'll actually work on a team, and the habit is worth building now.

You'd change your README from a description to an argument. Not longer, just more structured: problem, constraint, decision, result, how to run it. Five sections, each a few sentences. That structure alone puts you ahead of most applicants.

And you'd change how you evaluate your own work. The question stops being "is this impressive?" and becomes "is this legible?" Those are different questions, and the second one is the one that gets you the interview.

## Frequently Asked Questions

**How many projects should I have on my portfolio?**
Two or three well-presented projects beat ten shallow ones. A reviewer will look at the top one or two and stop. If those are strong, they'll assume the rest are similar. If those are weak, they'll assume the same. Curate ruthlessly. If a project isn't something you'd defend in a technical interview, either improve it or remove it from the list.

**Should I include a live demo link?**
Yes, but only if it actually works when the reviewer clicks it. A dead demo link is worse than no link, because it signals that you didn't check. If you're on a free tier that spins down, either pay for a cheap always-on instance or add a note explaining the cold start. Even better: record a 60-second screen capture and link that alongside the live demo, so the reviewer can see it working even if the demo is asleep.

**What if my project is a tutorial follow-along?**
Rebuild it with one meaningful change and document why. Tutorial projects are recognizable and reviewers discount them heavily. But a tutorial project that you extended, broke, and fixed is a real project. The extension is the signal. Write down what you changed and what happened when you changed it.

**How do I handle a project that's incomplete?**
Be explicit. "This is a work in progress. The auth flow is stubbed; the core data pipeline is complete and tested." Reviewers respect honesty about scope far more than they respect a project that pretends to be finished. An incomplete project with a clear boundary is legible. An incomplete project with no boundary is a red flag.

## Summary

The conventional advice [about portfolio projects](/portfolio-projects-that-hire-remote-senior-devs/) is not wrong, it's just aimed at the wrong target. Building something real and deploying it is necessary but no longer sufficient, because the reviewer's bottleneck is attention, not technical evaluation. The projects that get noticed are the ones that make a specific decision visible and measurable in the first minute of reading. That means a README that argues rather than describes, a commit history that shows scoped work, and one or two honest numbers that you can defend.

The alternative approach doesn't require more work. It requires different work: less time adding technologies, more time articulating tradeoffs. The code is the evidence; the narrative is what makes the evidence findable.

Your next step, in the next 30 minutes: open the README of your most recent project and rewrite the first paragraph to state the problem, the constraint, and one decision you made, in three sentences. If you can't, that's the signal to pick a different project or to make a decision explicit. Then commit the change with a message that says what you changed and why. That single edit is the highest-leverage 30 minutes you can spend on your portfolio this week.


---

### About this article

**Written by:** [Kubai Kevin](/about/) — software developer based in Nairobi, Kenya, with 10+ years building production systems in fintech and AI.

**How this article was produced:** This site uses an automated LLM pipeline designed and maintained by the author. Topics are selected from real production experience. Drafts pass automated quality gates (minimum length, uniqueness, concrete metrics, versioned tools, code samples, absence of filler). Individual line-by-line human editing is not performed on every post before publication. Specific numbers, benchmarks and cost figures are illustrative; verify them against current official documentation before production use.

**Corrections:** Report errors via the [contact page](/contact/). Corrections are applied promptly.

**Last generated:** September 2026
