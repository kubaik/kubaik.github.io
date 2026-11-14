# Your platform team is too big

Most platform engineering advice assumes an organisation that does not exist in most of the world: a headcount of hundreds, a cloud budget with slack in it, and a dedicated devops hire per product group. Small engineering organisations — a five-person engineering team in Nairobi, a fifteen-person NGO product team in Accra, a fintech in Lagos with three squads and one person who actually understands Kubernetes — read the same blog posts and conclude they need a "platform team." They then either hire for a role they cannot scope, or pull their best backend engineer off delivery and call it platform work. Both fail for the same reason: the platform group is staffed as a *team* before it is scoped as a *product*.

The position taken here is blunt: for most engineering organisations under roughly fifty engineers, a standing platform team is the wrong shape. What is needed is a platform *function* — one named owner, a small set of paved roads, and a hard rule that the platform group does not grow until the platform has users who would complain if it disappeared. The real problem is the staffing decision: who is on the platform group, what it owns, and what it is explicitly forbidden from owning, before the group becomes a cost centre that ships internal tooling nobody asked for.

## The full-team model assumes a company you don't work at

The dominant advice — Team Topologies, the platform-as-a-product literature, the conference talks — comes out of organisations where the platform group has a dozen or more engineers, a product manager, a designer, and a roadmap that competes with product roadmaps for the same quarterly planning. That model is correct *at that scale*. The underlying book (Skelton and Pais, *Team Topologies*) is careful about this: it describes platform teams as a team type that reduces cognitive load on stream-aligned teams, and it is explicit that the platform must be treated as a product with users. What gets lost in the retelling is the precondition — you need enough stream-aligned teams that the cognitive-load reduction is worth a dedicated group.

A useful arithmetic check, with stated assumptions: assume each stream-aligned team of five to eight engineers can absorb roughly one platform-adjacent responsibility per person without dropping delivery throughput (this is an assumption, not a measured figure — teams vary). A platform group of four people costs about twenty engineer-days a week. If it serves three squads of six, that is eighteen engineers, and the platform group is roughly eighteen percent of engineering. That ratio is defensible *only* if the platform removes more than twenty engineer-days a week of toil from those squads. In small orgs, the toil is real but it is not twenty days a week — it is a few hours a week of "who rotates the TLS certs" and "who fixes the deploy script." A four-person platform group against that workload is over-staffed by an order of magnitude, and the surplus capacity gets spent building things nobody asked for.

The failure mode is predictable: the platform group, with no genuine backlog of user complaints, starts building an internal developer portal. The portal has a service catalogue. The service catalogue has a scorecard. The scorecard has a maturity model. Eighteen months later the platform group has four engineers, one product, zero external users, and the stream-aligned teams are still deploying with a shell script one of them wrote in a hurry. The platform group has become a product team whose customer is itself.

## Why "hire a platform engineer" is the wrong first move

The common advice, when a small org asks how to start platform engineering, is to hire a platform engineer. This is almost always wrong, for three reasons that are specific to constrained environments.

First, the role is under-specified. "Platform engineer" in a job ad can mean anything from "writes Terraform and owns the cloud account" to "builds an internal developer portal" to "is the SRE." In a market where you are competing for a small pool of engineers against remote-first companies paying in hard currency, you cannot afford to hire against a vague spec and discover six months in that you hired an infrastructure generalist when you needed a build-and-release specialist.

Second, the hire creates a single point of failure that is worse than the problem it was meant to solve. If one person owns the platform and that person leaves — and in markets with high mobility, they leave — you have concentrated every deployment, every secret, and every piece of institutional knowledge about your cloud account in one resignation letter. The platform function must be *documented* before it is *staffed*, or the staffing is a liability.

Third, and most important: the person may already be on staff. In most orgs under fifty engineers, there is one backend engineer who already knows how the deploy works, already holds the cloud credentials, and is already the person everyone asks. That person is your platform function. The question is not whether to hire a platform engineer; it is whether to *name* the one you have and *scope* what they own, so the work stops being invisible and starts being planned.

## A better model: one owner, three paved roads, a written boundary

The mental model to hold is this: the platform group is not a team, it is a *contract*. The contract has one owner (a named engineer, not a committee), a small number of paved roads (the things the platform guarantees work), and a written boundary (the things the platform explicitly does not do). Everything else is negotiation.

### The single owner

Name one engineer as the platform owner. Not a manager, not a committee, not "the infrastructure squad." One person, with a job title that reflects the work ("Platform Lead" or "Infrastructure Owner" — the title matters less than the naming), and an explicit allocation. In small orgs this is usually a partial allocation: sixty percent platform, forty percent product, or the reverse. The allocation must be written down and visible to the whole engineering org, because the most common failure mode is that the platform owner is *nominally* allocated but *actually* pulled into every product fire, and the platform work silently stops.

The owner's job is not to build everything. It is to own the paved roads, triage requests against them, and say no. A platform owner who cannot say no is a ticket queue.

### Three paved roads, not thirty

A paved road is a thing the platform guarantees: "if you use this path, it works, it is supported, and we will not break it without warning." The number of paved roads should be small enough that the owner can actually maintain them. Three is a reasonable starting ceiling for a small org. Typical candidates:

- **A deploy path.** One way to get code from a branch to production, documented, with the secrets handled by the platform, not by each team.
- **A cloud account structure.** One way to get a new environment, with the IAM, networking, and billing boundaries already set.
- **An observability baseline.** One way to get logs, metrics, and alerts for a new service, with the retention and access rules already decided.

Everything not on the list is *not* a paved road. Teams can still do it, but they own it, and they cannot file a ticket expecting the platform to fix it. This is the boundary that keeps the platform group from becoming a general-purpose ops team.

### The written boundary

Write the boundary down. A one-page document — a `PLATFORM.md` in the repo, or a page in whatever wiki you actually use — that lists what the platform owns, what it does not own, and how to request a change to the list. The boundary is the single most valuable artefact the platform function produces, because it is the thing that lets the owner say no without a meeting.

Here is a minimal template you can adapt. It is not code, but it is the most useful thing in this article:

```markdown
# Platform ownership

## Owned (paved roads)
- Deploy: the CI pipeline in .github/workflows/deploy.yml and the credentials it uses.
- Environments: the Terraform in infra/ and the AWS accounts it provisions.
- Observability: the log/metric/alert defaults in infra/observability/.

## Not owned (team-owned)
- Application code, including Dockerfiles.
- Database schema and migrations.
- Business-logic alerts and dashboards.
- Third-party SaaS integrations.

## How to change this list
Open a PR against this file. The platform owner reviews within 5 working days.
```

That document is worth more than a headcount. It is also the thing that survives the owner leaving.

## Where the full-team model is still right

This is not an argument that platform teams are a bad idea. It is an argument that they are a *late* idea for small orgs and an *early* idea for large ones, and the threshold is roughly where the arithmetic above crosses over: when the platform function is removing more engineer-days of toil per week than it costs, and when there are enough stream-aligned teams that the cognitive-load reduction is the binding constraint rather than a nice-to-have.

At that point — call it fifteen to twenty engineers across four or more squads, with a cloud footprint that genuinely needs governance — the full Team Topologies model starts to pay off, and the advice in the conference talks becomes correct. The platform group gets a product manager. It gets a roadmap. It gets a service catalogue, and the catalogue is justified because there are enough services that discovery is a real problem. The mistake is adopting the *end state* before you have the *preconditions*. The model is not wrong; it is just downstream of a scale most readers have not reached.

There is one piece of the full-team model that is right at every scale, and it is the piece most often skipped: the platform must be treated as a product with users. Even a one-person platform function should have a way to hear from its users. A monthly fifteen-minute review of "what broke, what was slow, what did you have to work around" is enough. The moment the platform owner stops hearing from users, the platform starts drifting toward internal tooling nobody asked for.

## The decision rule

Here is the rule to apply, stated as a single test you can run in a meeting:

> **If you cannot name three stream-aligned teams that would each lose more than a day a week if the platform function disappeared tomorrow, you do not have a platform team. You have a platform owner.**

Run the test honestly. "Would lose more than a day a week" means measurable toil: deploys that take hours, environments that take days to provision, incidents that take a full day to diagnose because there are no baselines. If the answer is yes for three or more teams, staff the platform group. If the answer is yes for one team, that team has a platform problem, not a platform team. If the answer is no, you have a platform owner and a documentation problem, and the correct next hire is probably another product engineer, not a platform engineer.

The rule has a corollary that matters for constrained environments: **the platform group's headcount should never exceed the number of paved roads it owns, divided by two, rounded up.** Three paved roads, two people maximum. This is a heuristic, not a law, but it captures the intuition that a platform group with more people than roads will invent roads to justify the headcount. If you find yourself with four platform engineers and three paved roads, you have a staffing problem disguised as a scope problem.

### How to measure whether the rule is satisfied

The test above is only useful if "a day a week" can be checked rather than asserted. Two measurements make it concrete, and neither requires new tooling.

**Deploy lead time.** For each squad, record the timestamp of the last commit on a merged change and the timestamp when it reached production, for every deploy over four weeks. Most CI systems already store both; if not, a short script against the CI API will produce it. Compare the median and the spread per squad. A squad whose median is under an hour and whose slowest deploys are routine is not losing a day a week to deploys. A squad whose deploys regularly stall for a day waiting on credentials, manual approval, or an environment that does not exist is a candidate for the deploy paved road.

**Environment provisioning time.** For each new environment or service requested over the last quarter, record the time from request to first successful deploy. If that number is measured in days and the delay is waiting on one person, that person is the platform function whether or not the title exists. If the number is measured in hours, the paved road is already in place and the question is whether it is documented.

Run both measurements for one quarter before making a hiring decision. The output is a list of toil categories with hours attached, which is a far better input to a staffing conversation than an article's opinion.

## What this looks like when the constraint is real

A few concrete implications for teams operating under the constraints common in the region — no corporate credit card for cloud signup, users on feature phones, unreliable power during deploy windows, no dedicated devops hire.

**The paved road for deploys must be offline-tolerant.** If your CI runs on a cloud provider you cannot reliably reach during a power cut, the paved road is not paved. The deploy path should be runnable from a laptop with a local checkout and a documented credential refresh, even if the primary path is a hosted CI runner. Document the fallback. Test it quarterly. A platform that only works when the network is up is a platform that fails exactly when you need it.

**The paved road for environments must be cost-bounded.** A Terraform module that provisions a staging environment is only a paved road if it also provisions a budget alarm and a teardown path. In orgs without a dedicated finance function watching cloud spend, the platform owner is the only person who will notice a runaway environment, and only if the platform makes it visible by default. Bake the budget alarm into the module; do not make it optional.

**The paved road for observability must be cheap by default.** Retention policies and log volumes are where small orgs get surprised. The platform's default should be short retention, sampled metrics, and alerts only on symptoms users feel (error rate, latency, saturation), not on every internal metric. The platform owner who sets a thirty-day retention default on every log stream has just spent the quarter's budget on logs nobody reads.

These are not exotic requirements. They are the requirements that the full-team model, written for orgs with a finance team and reliable power, quietly assumes away. The platform function in a constrained environment is smaller and more opinionated, not less rigorous.

## A worked example: deciding for a fifteen-engineer org

Consider an illustrative organisation with fifteen engineers across three squads: a payments squad of six, a customer-facing app squad of five, and a data squad of four. One backend engineer, call her the operations-minded engineer, holds the cloud credentials and fixes the deploy script when it breaks. The org is debating whether to hire two platform engineers.

Step one is to apply the three-teams test with measurements rather than impressions. Suppose the deploy lead-time review shows the payments squad waiting a median of two days for production deploys because each deploy requires a manual credential step that only one person can perform, and that person is often in meetings. The app squad waits a median of half a day for the same reason. The data squad deploys weekly and waits a few hours. Only one squad is clearly losing more than a day a week, and the cause is a single manual step, not a missing platform.

Step two is to price the fix against the hire. Two platform engineers cost ten engineer-days a week. The measured toil across all three squads is roughly one and a half days a week of waiting, concentrated in one manual credential step. Automating that step is a week of work for the operations-minded engineer, after which the measured toil drops. Hiring two people to remove one and a half days of toil a week is a negative return by the arithmetic, and the surplus capacity is exactly what produces the service catalogue nobody asked for.

Step three is to write the boundary. The operations-minded engineer is named platform owner at forty percent allocation. The deploy path and the cloud account structure become paved roads; observability stays team-owned for now because no squad has complained about it. The `PLATFORM.md` records the allocation and the paved roads. Six months later, if the measurements show two or three squads each losing more than a day a week, the case for a second person — and eventually a team — is made by data rather than by fashion.

The same reasoning applies in reverse. If the measurements show three squads each losing two days a week, the toil is six engineer-days weekly and a single owner at forty percent cannot clear it. That is the point at which the first dedicated platform hire is justified, and the written boundary tells the new hire what they inherited.

## Frequently Asked Questions

**How many engineers do you need for a platform team?**

For most orgs under fifty engineers, you need one named owner, not a team. The platform group becomes a team when the arithmetic crosses over: when the toil removed from stream-aligned teams exceeds the cost of the group. Run the three-teams test above before hiring. If you cannot name three teams that would each lose more than a day a week without the platform, you have a platform owner, not a platform team.

**What is the difference between a platform team and a devops team?**

A devops team is typically a ticket queue that operates infrastructure on behalf of product teams. A platform team builds paved roads that product teams use themselves. The distinction matters for staffing: a devops team grows with the number of tickets, while a platform team grows with the number of paved roads. If your "platform team" is mostly responding to tickets, you have a devops team with a new name, and the staffing model is different.

**Should the platform team own the cloud account?**

Yes, at the account-structure level: IAM boundaries, networking, billing alarms, and the Terraform that provisions environments. No, at the application level: teams should not need the platform owner to deploy a service or read their own logs. The boundary is between *the shape of the account* (platform-owned) and *what runs in it* (team-owned). Write that boundary down before the first incident, not after.

**When should a small org hire its first platform engineer?**

After the platform function is documented and has an owner, not before. Hiring a platform engineer into an undocumented, unscoped platform is how you get a single point of failure with a salary attached. The first hire should be the person who *inherits* a written boundary and three paved roads, not the person who invents them under pressure.

**Does a partial allocation actually work in practice?**

It works when the allocation is written down and the owner is protected from ad hoc pulls. The failure mode is a nominal sixty-forty split that becomes a hundred percent product work during a busy quarter, with platform work resuming never. The protection is procedural: platform work appears on the same planning board as product work, and the owner's manager treats a platform commitment as a commitment. If the org cannot protect the allocation, a partial owner is worse than no owner, because it creates the appearance of a platform function without the substance.

## The next thirty minutes

Open your repository and create a file called `PLATFORM.md` at the root. Write three headings: "Owned (paved roads)," "Not owned (team-owned)," and "How to change this list." Fill in the first heading with the three things your team would actually lose a day a week without. If you cannot fill in three, leave it at one and write down why — that gap is the real staffing question. Commit it. The commit is the first artefact of your platform function, and it costs nothing but the honesty to write it down.
