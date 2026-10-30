# Promotions in the AI era: the mechanism

Code generation is cheap now. Judgment is not. That asymmetry has quietly broken the signals engineering ladders have used for years to decide who gets promoted, and most teams have not adjusted their evidence-gathering to match.

This is a mechanism writeup, not a motivational one. It covers why the old signals decayed, which adaptation patterns tend to fail, what a workable operating pattern looks like, and how to tell whether it is working.

## The pattern behind the promotions

Engineering promotions have always been a proxy for something else: evidence that a person can be trusted with larger ambiguity. The AI era did not change that. It changed where the ambiguity lives.

Before code assistants became normal, a large share of senior-level signal came from *production of artifacts*: the service, the migration, the runbook. Throughput was legible. A staff engineer could look at a year of commits and see scope.

Now the artifact layer is cheap. A junior engineer with a good model can produce a service skeleton, a test file, and a Dockerfile in an afternoon. That does not make them senior. It makes the *artifact* weak as a promotion signal. What replaced it is judgment about the non-artifact parts: what the artifact should not do, what it will cost, what happens when it fails at 3am, and who has to be told.

The engineers who advance faster in this environment are not the fastest prompt operators. They are the ones whose work creates *reviewable decisions* — decisions a manager can point at in a calibration meeting and say "this person owned this tradeoff." Engineers who stall are usually the ones whose output looks like a lot of code and very little ownership.

## Why the old promotion signals decayed

Engineering ladders historically rewarded three things: scope of code owned, complexity of systems touched, and cross-team influence. Code volume was never officially a criterion, but it leaked in everywhere — in perf review templates, in "impact" narratives, in the sheer visibility of someone who ships a lot.

When generation gets fast, three things break at once.

**First, the diff stops being evidence of understanding.** A 400-line PR generated from a clear prompt and a 400-line PR written by hand look identical in the review UI. Reviewers cannot tell, and more importantly, the *author* cannot always tell whether they understand the code or just recognize it. This is a well-documented failure mode: a developer accepts a suggestion, it passes tests, and six weeks later nobody on the team can explain why a particular retry loop exists. The code is correct until it isn't, and the person who "wrote" it has no model of it.

**Second, the bottleneck moves.** When generation is cheap, the constraint becomes review capacity, deployment safety, and the cost of being wrong. A typical failure mode: throughput goes up, incident rate goes up shortly after, and the on-call rotation becomes the real limit on how fast the org can move. The person who understands this and acts on it — by adding guardrails, by deliberately slowing the risky path — looks less productive in a commit graph and more valuable in a postmortem.

**Third, the definition of "hard" shifts.** Writing a distributed lock is now a prompt. Deciding whether you need a distributed lock, versus an idempotency key, versus accepting the duplicate, is not. The second kind of decision is where seniority lives, and it was always where seniority lived — it just used to be hidden behind the volume of the first kind.

A useful way to see it: the ladder did not change, the *visible surface* of the ladder changed. Engineers who kept optimizing for the old visible surface — lines, tickets, PR count — are now optimizing for a metric that no longer correlates with the thing being measured.

## The approaches that commonly fail

Three patterns recur when teams try to adapt, and all three tend to hurt the people who adopt them most enthusiastically.

### Treating AI output as a throughput multiplier

The most common failure is measuring the wrong thing. If a team starts tracking "PRs per engineer" or "features per sprint" after adopting assistants, the number goes up and the meaning goes down. Engineers who internalize this metric start shipping more surface area with less review, less testing, and less documentation. For a quarter it looks like a promotion case. Then a production incident traces back to a generated migration that dropped a column with no backfill, or a generated retry policy that turned a transient dependency blip into a thundering herd.

The mechanism is straightforward: generation optimizes for *plausible*, and production punishes *plausible but wrong*. The gap between those two is exactly the gap senior engineers are paid to close. A throughput metric rewards closing it less.

### Becoming the person who "just uses the tool well"

A subtler trap: engineers who become known primarily as effective tool users — fast prompters, good at getting the model to produce working code — can plateau. Tool fluency is real and useful, but it is a *skill*, not a *scope*. Ladders promote scope. If the entire visible contribution is "I can make the assistant produce the thing," that contribution is competing with everyone else who can also do it, and the skill commoditizes fast.

The engineers who avoid this trap use the tool to buy time, then spend that time on the parts of the problem that don't fit in a prompt: negotiating an interface with another team, writing the design doc that prevents a bad decision, instrumenting the system so the next incident is diagnosable in five minutes instead of five hours.

### Assuming the ladder will be rewritten for you

Some engineers wait for their org to publish new criteria — "AI-assisted development" competencies, updated impact rubrics. Ladders move slowly, and the people running calibration meetings are usually evaluating against the same dimensions they always used: scope, judgment, influence, and results. What changed is the *evidence* they can see. Producing evidence in the old shape (volume) feeds them a signal that no longer differentiates. Producing evidence in the new shape (decisions, tradeoffs, prevented failures) feeds them the signal they were always looking for.

## The approach that works: decision artifacts

The pattern that correlates with faster advancement in this environment is producing what can be called *decision artifacts* — durable, reviewable records of a judgment call, with the reasoning and the rejected alternatives visible.

A decision artifact is not a design doc in the ceremonial sense. It can be a short RFC, a well-written PR description, an ADR (architecture decision record), a postmortem action item, or a comment thread that gets linked in a review. What matters is that it contains four things:

1. **The decision** — one sentence, unambiguous.
2. **The alternatives considered** — including the one you rejected and why.
3. **The failure mode you are accepting** — what breaks if you're wrong, and how you'll know.
4. **The cost** — in money, latency, operational burden, or team time.

ADRs have been around for a long time and are well-documented. What changed is that in a world where the *code* is cheap, the artifact is the only durable evidence of seniority. A generated service with no decision artifact is a liability with a nice README. A generated service with a decision artifact is a promotion case.

### A worked example

Suppose a team needs to add a caching layer to a service that reads a product catalog. The junior-shaped output is: pick Redis, write the wrapper, ship it. The senior-shaped output is a short note with reasoning attached.

Here is the reasoning, shown step by step:

1. **Enumerate the candidates.** In-process cache (a map with a TTL), a managed cache service, or a self-hosted Redis cluster. Three options, not one.
2. **Estimate the latency win.** Suppose the catalog query takes 40 ms at p50 and the service handles 2,000 requests per second, and 90% of reads hit the same 5,000 keys. An in-process cache would remove most of that 40 ms for hot keys. A network cache would remove most of it too, but adds roughly 1 ms of network round trip. The latency difference between the two is small relative to the 40 ms being saved.
3. **Weigh the operational cost.** An in-process cache means N instances each hold their own copy. Invalidation requires either a pub/sub channel or a TTL short enough that staleness is tolerable. With 12 instances and a 60-second TTL, a catalog update is visible everywhere within 60 seconds at worst. If the product team needs sub-second propagation, that constraint kills the in-process option.
4. **Weigh the failure mode.** A network cache adds a dependency on the hot path. If it goes down, either the service falls back to the database (a thundering herd against a database sized for 10% of traffic) or it fails. That is the failure mode being accepted, and it needs a mitigation: a circuit breaker plus a request-coalescing layer so concurrent misses on the same key result in one database query, not 2,000.
5. **Write down the decision and the tripwire.** "We chose in-process caching with a 60-second TTL because propagation latency under one minute is acceptable and it avoids a new hot-path dependency. If the product team needs faster propagation, we revisit. We will know we were wrong if catalog-staleness complaints exceed a handful per week, or if the database load from cold starts after deploys spikes above baseline."

Same code in the end. Different evidence. One version gets discussed in calibration; the other gets skimmed in review.

Crucially, the decision artifact is *cheap to produce* when the reasoning already exists — and the reasoning should exist, because that is the job. The engineers who advance are the ones who write it down instead of keeping it in their head.

### How to measure whether it is working

The claim that decision artifacts help is testable at the team level, without any vendor benchmark. Instrument these:

- **Review latency by PR type.** Compare time-to-first-substantive-comment for PRs with a decision-style description versus those without. If reviewers engage differently, the difference shows up here.
- **Comment depth.** Count comments that question the approach rather than the syntax. A shift from "nit: rename this" to "why not an idempotency key?" is the signal.
- **Rework rate.** Track the fraction of merged changes that get reverted or substantially rewritten within 30 days. Decision artifacts should reduce this, because the alternatives were considered before merge rather than after.
- **Incident attribution.** In postmortems, tag whether the root cause was a decision that was never written down. A falling count over two quarters is weak but real evidence.

Run these for a quarter before drawing conclusions. The sample sizes are small and the confounders are many; treat the numbers as directional, not decisive.

## Implementation details

A workable pattern, in order of how much it costs to adopt:

**Start with PR descriptions.** The single highest-leverage change is writing PR descriptions that state the decision and the rejected alternative. Not a summary of the diff — the diff is right there. A description that says "we chose to do X instead of Y because Z, and if this is wrong the symptom will be W" turns a code review into a judgment review. Reviewers engage differently. Managers who read PRs (and they do, especially before calibration) see something they can cite.

**Add ADRs for anything cross-team.** An architecture decision record is a numbered file in a `docs/adr/` directory, with context, decision, status, and consequences. They are cheap to write and they accumulate. A year of ADRs is a portfolio. Most engineers never produce one because nobody asked; the ones who do become the people whose names are attached to how the system works.

**Instrument the decision.** A decision artifact that says "we'll know we were wrong if p99 exceeds X" is only credible if the metric exists. Adding a single dashboard panel or alert that ties back to a decision is a small amount of work and it converts a claim into a commitment. When the metric fires and you respond, that is an incident writeup — another artifact.

**Keep a running log.** Not a brag doc in the LinkedIn sense — a plain file with dated entries: decision, context, outcome. This is the raw material for a promotion packet, and it prevents the common failure where an engineer did senior work all year and cannot remember any of it in review season.

A minimal ADR template, since the format matters less than the discipline:

```markdown
# ADR-014: Use idempotency keys for payment retries

## Status
Accepted (2026-03-11)

## Context
Retries on the payment path can double-charge when the upstream
acknowledges but the response is lost. Current retry logic is
unconditional.

## Decision
Require a client-supplied idempotency key on all mutating payment
calls. Server stores key -> result for 24h.

## Alternatives considered
- At-most-once delivery (rejected: drops legitimate retries)
- Client-side dedup (rejected: cannot be enforced server-side)

## Consequences
- Adds a storage dependency on the hot path
- Key collision handling must be defined (see ADR-015)
- If wrong, symptom is elevated 409s on retry; alert on rate > 1%
```

That is roughly ten minutes of writing. It is also the kind of thing that gets quoted in a promotion discussion.

## What outcomes to expect, and their limits

It would be dishonest to claim this pattern reliably produces a promotion in a fixed timeframe. It does not, and the variance is mostly organizational. What it does reliably is change what a manager *can say about you* in a calibration meeting. That is the actual bottleneck for most engineers — not that they haven't done senior work, but that the senior work isn't legible.

A few honest limits:

- **It doesn't fix a bad manager or a frozen ladder.** If your org has no headcount or is in a hiring freeze, no artifact changes that. The artifacts still help at the next org.
- **It doesn't substitute for results.** A decision artifact about a decision that didn't work is still useful if you wrote down how you'd know and then responded — but a string of decisions that all failed is a different conversation.
- **It can read as performative if overdone.** One ADR per significant decision is signal. An ADR for every variable rename is noise, and it will be read as noise.
- **It competes for time with shipping.** The tradeoff is real. The argument here is that the time is better spent on artifacts than on additional generated surface area, not that artifacts are free.

A reasonable expectation, framed as a range rather than a promise: engineers who adopt this pattern consistently tend to find that review conversations change within a quarter, and that promotion discussions become easier to have because there is something concrete to discuss. Whether that translates into a promotion depends on the org.

## What to watch out for

**Artifact theater.** The failure mode of this pattern is producing decision documents that don't reflect real decisions — post-hoc rationalizations of what was already built. Reviewers can tell. The artifact has to precede or accompany the decision, not follow it.

**Using AI to write the artifacts.** There's a temptation to generate the ADR the same way you generate the code. The result is usually fluent and empty: it lists alternatives nobody considered and consequences nobody checked. The value of the artifact is that it encodes *your* judgment. If the judgment isn't yours, the artifact doesn't help you and may actively hurt you when someone asks a follow-up question in review.

**Confusing visibility with influence.** Writing things down makes you visible. It does not automatically make you influential. Influence comes from the decisions being good and from other people adopting them. The artifact is the medium, not the message.

**Neglecting the unglamorous work.** On-call, incident response, and dependency upgrades are still where a lot of trust is built, and they are exactly the areas where generated code tends to create latent problems. An engineer who is visibly good at cleaning up after fast shipping is more promotable than one who is only visibly good at fast shipping.

## The broader lesson

The AI era didn't change what senior engineering is. It removed the camouflage. When code production was expensive, the volume of code was a rough proxy for the amount of judgment that had gone into it. Now that production is cheap, the proxy is gone, and the judgment has to be visible on its own.

This is uncomfortable for engineers who built their identity around output, and it is an advantage for engineers who were always doing the judgment work but didn't know how to make it legible. The promotion mechanism was never really about code. It was about being trusted with ambiguity. AI made ambiguity the only thing left to demonstrate.

Engineers who stall are, in most cases, not stalling because they used AI. They are stalling because their visible contribution is now something the org can get from a tool plus a junior engineer, and they haven't moved up the stack to the part the tool can't do. That's a harsh framing, but it matches what tends to happen in calibration rooms.

The engineers who advance faster are the ones who noticed that the artifact layer got cheap and moved their effort to the decision layer, where it's still expensive.

## Frequently Asked Questions

**Does using AI coding assistants hurt your chances of promotion?**
No, and framing it that way misses the mechanism. The assistants are not the variable; the visibility of your judgment is. Engineers who use assistants to free up time for decision work advance faster. Engineers who use them to increase output volume without increasing judgment tend to plateau, because output volume stopped being a differentiator.

**How do you demonstrate senior impact when AI writes most of the code?**
By owning the decisions around the code: what to build, what not to build, what failure modes are acceptable, what the cost is, and how you'll know if you were wrong. These are the things a model cannot be accountable for, and they are exactly what promotion committees evaluate. Write them down in a form a manager can cite.

**What is an architecture decision record and why does it matter for promotions?**
An ADR is a short, numbered document recording a significant technical decision, its context, the alternatives considered, and the consequences. The format is well-established and lightweight. It matters for promotions because it converts your reasoning into durable, reviewable evidence that survives long after the code has been rewritten.

**When should you not write a decision artifact?**
When the decision is trivial and reversible — a formatting choice, a variable name, a dependency patch version. Artifacts are signal when they mark real tradeoffs. Writing one for every small change dilutes the signal and reads as performative, which is worse than not writing them at all.

## Do this next

Pick one decision you made in the last two weeks — a library choice, a schema change, a retry policy, anything with a real tradeoff — and write it up as a short ADR in a file called `docs/adr/0001-<slug>.md` in your repo. Include the alternatives you rejected and the failure mode you're accepting. Commit it. If you don't have a repo handy, write it in a notes file and paste it into your next PR description. The pattern only works as a habit, and habits start with a single instance.
