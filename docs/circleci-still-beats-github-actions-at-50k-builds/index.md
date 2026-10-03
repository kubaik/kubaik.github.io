# When CircleCI's Concurrency Model Beats GitHub Actions

## The conventional wisdom (and why it's incomplete)

GitHub Actions is widely treated as the default CI platform: it is free for public repositories, tightly integrated with the code host, and requires almost no onboarding effort if the repository already lives on GitHub. CircleCI is often framed as the choice for legacy shops, large monorepos, or teams willing to pay for features that appear free elsewhere.

That framing ignores three realities that only surface at volume:

1. **Free-tier math changes at scale.** A generous monthly minute grant covers small teams, but at tens of thousands of builds per month every additional minute, concurrent job, and artifact retention day is billed. CircleCI's pricing is organised around parallelism and credits rather than raw minutes, which matters when builds are short but frequent.

2. **Artifacts and caching are where cost leaks.** GitHub Actions bills artifact storage beyond the included allowance. CircleCI's model bills storage beyond an included retention window on a per-GB-per-day basis. Both can surprise you; neither is automatically cheaper.

3. **Concurrency, not minutes, is often the bottleneck.** GitHub Actions enforces per-account concurrency limits that vary by plan and are documented in GitHub's billing and limits pages. When a team hits that ceiling, jobs queue. CircleCI's paid plans advertise explicit concurrency allowances, so the ceiling is a known number you can budget against.

The conventional wisdom is incomplete because it treats CI as a checkbox rather than a system with real constraints: concurrency, artifact growth, queue latency, and minutes per build.

## A cost model you can actually run

Any comparison has to start from stated assumptions. The following is an **illustrative** workload, not a measured one:

- 50,000 builds per month
- Average build time: 1 minute 45 seconds (1.75 minutes)
- Artifacts: 1.5 GB per month retained for 30 days
- Peak concurrency: 200 parallel jobs

**Minutes consumed:** 50,000 × 1.75 = 87,500 minutes.

**GitHub Actions minutes.** GitHub publishes included minutes per plan and a per-minute rate for additional usage on hosted runners. The exact rate depends on runner size and OS. If the additional-minute rate is $0.008/min (a figure you must confirm against the current pricing page for your runner type), then:

- Extra minutes: (87,500 − included grant) × $0.008
- With a 2,000-minute grant: 85,500 × $0.008 = **$684/month**

**Artifact storage.** If storage is billed at $0.023/GB/day after an included allowance:

- 1.5 GB × $0.023 × 30 days = **$1.04/month**

That number is small; artifact storage rarely dominates unless retention is long or artifacts are large.

**Concurrency.** This is where the two platforms diverge structurally. GitHub Actions concurrency is tied to your plan and account type; raising it may require a plan upgrade or an enterprise agreement. CircleCI sells concurrency explicitly as part of a plan tier. The correct move is to price the *plan tier that gives you the concurrency you need*, not the per-minute rate alone.

**Queue time.** Queue time is real cost, but it is not billed as minutes on either platform. It is paid in developer waiting. To quantify it:

- Instrument queue time per job (both platforms expose this in their APIs and job metadata).
- Compute `sum(queue_seconds) / 3600` for a month to get queue-hours.
- Multiply by a loaded hourly engineering rate to get an internal cost figure.

Do not add queue time to the vendor invoice. It belongs in a separate "cost of latency" column, because it is an opportunity cost, not a line item.

## Why queue time is the hidden variable

A CI platform is a distributed system: a pool of runners executing jobs on demand. The metrics that matter are throughput (jobs per hour), latency (queue time plus build time), cost (minutes, storage, concurrency), and reliability (uptime, error rates).

GitHub Actions is optimised for onboarding velocity. It uses your existing identity, requires a single workflow file, and feels free until you scale. Its hosted runners are ephemeral and disk-constrained, and jobs compete for a shared pool. That is fine at low volume; at high volume the queue becomes the constraint.

CircleCI is optimised for explicit, purchasable parallelism. Concurrency is a plan feature, runners can be sized with `resource_class`, and pricing is tied to credits and parallelism rather than pure minutes. The trade-off is integration friction: API tokens, contexts, orbs, and a UI that many developers find dated.

A common failure mode is choosing a platform for onboarding convenience, then discovering the concurrency ceiling months later when the queue has already become a daily complaint. The fix is almost always a plan-tier change, not a migration — but by then the team has usually already decided to migrate, which is the expensive path.

## A worked comparison with two variables

The table below is **illustrative arithmetic** from the assumptions above. Every rate must be re-checked against current vendor pricing before you rely on it.

| Item | GitHub Actions (illustrative) | CircleCI (illustrative) |
|---|---|---|
| Minutes consumed | 87,500 | 87,500 |
| Included minutes | 2,000 | Plan-dependent |
| Extra-minute cost | 85,500 × $0.008 = $684 | Depends on credit rate |
| Artifact storage | 1.5 GB × $0.023 × 30 = $1.04 | Same formula if rate matches |
| Concurrency tier | Plan upgrade required | Explicit plan tier |
| Queue time | Not billed, but real | Not billed, but real |

The honest conclusion is not "CircleCI is 75% cheaper." The honest conclusion is: **the cheaper platform is the one whose concurrency tier matches your peak, priced at the per-minute rate you actually pay.** If your peak concurrency is below GitHub's included limit, GitHub Actions is usually cheaper because the integration is free. If your peak exceeds it and the upgrade path is expensive, CircleCI's explicit concurrency pricing can win.

## How to measure this yourself in 30 days

Do not trust either vendor's marketing or any blog post's numbers, including this one. Run a pilot.

**What to instrument:**

1. **Queue time per job.** Pull it from the CI API. On GitHub, the workflow run object exposes `created_at` and `run_started_at`; the difference is queue time. On CircleCI, the job object exposes `queued_at` and `started_at`.
2. **Minutes consumed per month.** Both platforms report this in billing. Record it weekly, not monthly, so you catch spikes.
3. **Artifact storage growth.** Query the artifact API per project and sum bytes. Track the 30-day delta.
4. **Peak concurrency.** Count concurrently running jobs at 1-minute resolution over a week. The maximum is your real peak, not your average.

**What to compare:**

- Total vendor invoice for the month.
- Total queue-hours (queue time summed across all jobs).
- Peak concurrency versus plan limit.
- Build time distribution (p50 and p95), not just the mean.

**A concrete pilot:**

Mirror one high-traffic repository onto the other platform. Keep the same test commands. Run both for 30 days. Compare the four numbers above. A migration is justified only if the total invoice plus the queue-hour cost is materially lower *and* the p95 build time does not regress.

## Decision checklist

Use this before choosing a platform at high volume:

- [ ] Is the repository public? If yes, GitHub Actions' free minutes usually dominate.
- [ ] What is the p95 build time, not the mean? Long-tail builds drive cost.
- [ ] What is the peak concurrent job count? Compare it to each platform's included limit.
- [ ] What does the plan tier that covers that peak cost, all-in?
- [ ] What is the 30-day artifact storage growth, and what retention is actually required?
- [ ] Is there an SLA requirement for queue time? If yes, only plans with a documented SLA qualify.
- [ ] How much engineering time will integration cost? Count secrets, webhooks, status checks, and artifact stores.
- [ ] Can the team accept the UI and workflow differences?

## Configuration notes

`resource_class` lets you request a larger runner on CircleCI. The available classes and their CPU/memory figures are documented in CircleCI's configuration reference; verify the exact sizes before relying on them.

```yaml
# .circleci/config.yml
jobs:
  test:
    docker:
      - image: cimg/python:3.11
    resource_class: medium+
    steps:
      - checkout
      - run: pip install -r requirements.txt
      - run: pytest
```

On GitHub Actions, cap runaway jobs and cancel superseded runs, both of which reduce billed minutes:

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    concurrency:
      group: ${{ github.ref }}
      cancel-in-progress: true
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: pytest
```

The `concurrency` block cancels in-progress runs for the same ref, which is the single cheapest way to cut both queue pressure and billed minutes on a busy repository.

## Where GitHub Actions is the right answer

- **Public repositories and open-source-heavy teams.** Free minutes cover most OSS workloads.
- **Teams already deep in GitHub workflows.** Status checks, secrets, and PR integration come for free; the cognitive load of a second platform is a real cost.
- **Light artifact needs.** If artifacts stay under the included allowance, storage is a non-issue.
- **Low peak concurrency.** If peak stays below the included limit, queue time is not a problem.
- **Teams that value simplicity over predictability.** One YAML file versus contexts, orbs, and API tokens.

## Where CircleCI is the right answer

- **Peak concurrency above GitHub's included limit**, where the upgrade path is expensive or slow.
- **A documented SLA requirement** for queue time or uptime.
- **Larger runner needs**, where `resource_class` gives finer control than the default hosted runner sizes.
- **Multi-cloud or multi-repo estates** where the CI platform should not be tied to one code host.
- **Budget predictability**, because concurrency is a purchasable, fixed number rather than a variable.

## Common objections

**"The CircleCI UI is slow and dated."** Often true. The UI is separate from runner performance, and the REST API plus CLI can cover day-to-day work. Build a small dashboard that surfaces only queue time, build time, and artifact size.

**"GitHub Actions integrates better with our GitHub workflows."** At low volume, yes. At high volume, weigh the integration friction against the concurrency and latency savings. Status checks can be posted back via the CI provider's GitHub integration, but it is not native and costs configuration effort.

**"CircleCI is more expensive for small teams."** Usually true. Below roughly ten engineers, GitHub Actions is typically cheaper. The crossover depends entirely on peak concurrency and plan-tier pricing, so compute it rather than assuming.

**"We'll move to self-hosted runners."** Self-hosted runners shift cost from minutes to operations: maintenance, autoscaling, and security patching. Evaluate that path on its own merits; do not use it as a reason to defer a platform decision.

## Frequently asked questions

**How do you estimate GitHub Actions cost at 50,000 builds per month?**

Multiply builds by average build minutes to get total minutes consumed. Subtract the included grant for your plan. Multiply the remainder by the per-minute rate for your runner type. Add artifact storage at the per-GB-per-day rate beyond the included allowance. Then add the plan tier required to reach your peak concurrency. Confirm every rate on the current pricing page, because rates vary by runner size and OS.

**Why can CircleCI cost less than GitHub Actions at scale?**

Not because minutes are cheaper — per-minute rates are often similar — but because concurrency is sold explicitly as a plan feature. If GitHub's included concurrency is below your peak and the upgrade path is expensive, CircleCI's fixed concurrency tier can be cheaper overall. The saving comes from the concurrency line, not the minutes line.

**What is the fastest CI setup for a mixed Python and Node stack?**

The dominant factor is runner size and disk, not the vendor. Larger runners with more disk avoid the ephemeral-disk thrashing that makes `npm install` and dependency resolution slow. On CircleCI, that means a larger `resource_class`. On GitHub Actions, it means a larger hosted runner. Benchmark p95 build time on both before deciding.

**How do you reduce CI cost at low volume?**

Stay on the free tier where possible. Cap job duration with `timeout-minutes`, cancel superseded runs with `concurrency`, cache dependencies, and shorten artifact retention. At around 1,000 builds per month, free grants usually cover the workload entirely.

## Action for the next 30 minutes

Open your CI billing dashboard and your CI provider's API, and record three numbers for the last seven days: peak concurrent jobs (sampled at one-minute resolution), total queue-hours (sum of `started_at − queued_at` across all jobs, divided by 3600), and 30-day artifact storage growth in GB. If peak concurrency is within 20% of your plan's limit, or total queue-hours exceed 10 hours per week, price the next plan tier on both platforms today. That single comparison will tell you more than any migration decision made on convenience.
