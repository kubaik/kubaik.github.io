# Deployment platforms: senior vs new hire

A deployment platform serves two users with conflicting needs. A senior engineer wants to bypass it when it is wrong. A new hire needs it to be right by default. A platform that serves only one of them fails in a predictable way: the senior engineer routes around it with a direct `kubectl apply`, and the new hire's mental model of how deployments work becomes wrong. Or the platform locks everything down so tightly that adding a migration job takes days of platform-team negotiation, and the new hire never learns what is actually happening.

Teams commonly solve this badly by picking a tool that optimises for one persona and telling the other to cope. This article is about the patterns that let both coexist, and how to evaluate them.

## What the problem actually is

The core tension is **escape hatches versus guardrails**. Senior engineers need to break glass when the platform is wrong. New hires need the platform to be right by default. Optimise only for escape hatches and you get snowflake deployments nobody can reproduce. Optimise only for guardrails and you get a platform team that becomes a ticket queue.

The deployment platform itself is rarely the problem. The problem is the **interface** between the platform and the people using it. That interface is usually a CLI, a config file, or a Git repository layout. Evaluate tools by how well their interface handles both personas.

## Evaluation criteria

Score each option on five things:

1. **Escape hatch cost** — how many steps to do something the platform did not anticipate? A good platform makes the escape hatch explicit and logged, not hidden.
2. **New-hire time-to-first-deploy** — can someone with zero context ship a change on day one without reading a long runbook?
3. **Blast radius control** — if a new hire runs the wrong command, what breaks? The platform should make the dangerous path harder than the safe path.
4. **Observability of intent** — can you look at a deployment and tell *why* it happened, not just *what* changed? Git history is the usual answer, but only if the platform reads from Git.
5. **Local reproducibility** — can a new hire run the same deployment locally without cloud credentials? This is where most platforms quietly fail.

## Options compared

### 1. Kubernetes with a thin internal CLI

**What it does:** You run a real Kubernetes cluster. You write a small CLI (often a wrapper around `kubectl`) that enforces naming conventions, injects environment variables, and defaults to safe rollout strategies. Senior engineers can still run `kubectl` directly; the CLI is a convenience, not a cage.

**Strength:** The escape hatch is free. `kubectl` is always there. A senior engineer who needs to debug a stuck pod can do so without asking permission. The CLI handles the common case (deploy a service, run a migration, tail logs) and gets out of the way otherwise.

**Weakness:** You are now maintaining a CLI. That CLI is a product with users, and if it drifts from the cluster's actual state, new hires get confusing errors. Also, `kubectl` access means a new hire can accidentally delete a production namespace if RBAC is not tight. You need to invest in RBAC and admission controllers, which is real work.

**Best for:** Teams with at least one person who has run Kubernetes in production and can own the cluster's upgrade path. If nobody on the team has done that, this option will consume more time than it saves.

A minimal wrapper that shows the pattern:

```bash
#!/usr/bin/env bash
# deploy.sh — thin wrapper around kubectl
set -euo pipefail

SERVICE="$1"
ENV="${2:-staging}"

# Guardrail: refuse to deploy to prod without an explicit flag
if [[ "$ENV" == "prod" && "${ALLOW_PROD:-}" != "yes" ]]; then
  echo "Refusing prod deploy. Set ALLOW_PROD=yes to proceed." >&2
  exit 1
fi

# Convention: manifests live in deploy/<service>/<env>.yaml
MANIFEST="deploy/${SERVICE}/${ENV}.yaml"
if [[ ! -f "$MANIFEST" ]]; then
  echo "No manifest at $MANIFEST" >&2
  exit 1
fi

kubectl apply -f "$MANIFEST"
kubectl rollout status "deployment/${SERVICE}" -n "${ENV}" --timeout=120s
```

This is boring, and boring is the point. The guardrail (prod requires a flag) is visible in the script. A new hire reading it learns the convention. A senior engineer can bypass it by running `kubectl` directly, which is logged by the API server.

### 2. GitOps with a reconciliation loop

**What it does:** A controller (Argo CD and Flux are the two widely documented options) watches a Git repository and reconciles the cluster to match. Deployments happen by merging a pull request. There is no `deploy` command.

**Strength:** Intent is fully observable. Every change has a commit, an author, and a review. A new hire can look at the Git history and understand what changed and why. Senior engineers get a clean rollback story: revert the commit, the controller reconciles.

**Weakness:** The reconciliation loop introduces a delay between merge and effect, which frustrates senior engineers debugging an incident. Worse, if the controller is down, merges pile up silently. And the escape hatch — manually editing a resource — gets reverted by the controller, which is correct but surprising the first time. New hires often do not realise their manual fix will disappear.

**Best for:** Teams where auditability matters more than deploy latency, and where at least one person understands the controller's sync policies. If you have compliance requirements, this is the strongest default.

### 3. A managed PaaS with a buildpack model

**What it does:** You push code, the platform detects the language, builds it, and runs it. Heroku popularised this model; Render, Railway, and Fly.io offer variations. There is no cluster to manage.

**Strength:** New-hire time-to-first-deploy is measured in minutes. There is no YAML to learn. The platform handles TLS, scaling, and health checks. For a small team, this removes an entire category of work.

**Weakness:** The escape hatch is expensive. If you need a custom binary, a sidecar, or a specific kernel parameter, you are either paying for a higher tier or moving off the platform. Senior engineers hit this ceiling quickly, and the migration off a PaaS is a project, not a task. Also, local reproducibility is imperfect: the buildpack that runs in production may not match your local environment.

**Best for:** Teams of roughly one to five engineers with a standard web application and no unusual infrastructure needs. Pre-product-market-fit, this is the right default. Do not over-engineer before you have users.

### 4. Serverless with infrastructure-as-code

**What it does:** You define functions and their triggers in a config file (AWS SAM, Serverless Framework, or Terraform), and a deploy command pushes them. There is no server to patch.

**Strength:** The scaling model is automatic, and the cost model is pay-per-invocation, which is friendly to early-stage products. New hires can deploy a function without understanding networking. The config file is the documentation.

**Weakness:** Local reproducibility is the hardest of any option here. Emulators exist (for example, `sam local` for AWS Lambda), but they diverge from production in ways that matter: IAM permissions, cold starts, and timeout behaviour. Debugging the differences between local and deployed is a common source of multi-hour sessions. Also, the escape hatch is limited: you cannot SSH into a function.

**Best for:** Event-driven workloads and teams comfortable with a single cloud provider's ecosystem. If your application is a long-running process with heavy state, this is the wrong shape.

### 5. Nomad or a similar scheduler with a job spec

**What it does:** You write a job spec (HCL for Nomad), and the scheduler places it. It is simpler than Kubernetes for non-container workloads and supports both containers and raw executables.

**Strength:** The job spec is readable by a new hire in an afternoon. Nomad's operational surface is smaller than Kubernetes, which means fewer things to break. Senior engineers appreciate that it runs non-containerised binaries, which is useful for legacy components.

**Weakness:** The ecosystem is smaller. Fewer managed offerings, fewer Stack Overflow answers, fewer people who have run it. If you hit a bug, you are more likely to be on your own. The escape hatch exists (you can run a job in dev mode), but the community around it is thinner.

**Best for:** Teams with a mix of containerised and non-containerised workloads, and at least one person willing to own the scheduler. Not a good first choice if nobody has operated it before.

### 6. A CI/CD pipeline that treats deployment as a pipeline stage

**What it does:** GitHub Actions, GitLab CI, or Jenkins runs a pipeline that builds, tests, and deploys. The deployment is one stage among many.

**Strength:** The pipeline is the documentation. A new hire reads the YAML and sees exactly what happens. Senior engineers can add a manual approval gate for production, which is a clean escape hatch: the pipeline pauses, a human clicks, the deploy continues.

**Weakness:** Pipelines rot. A pipeline that worked six months ago may fail because a base image was updated or a credential expired. New hires often cannot debug pipeline failures because the error messages are opaque. Also, the pipeline is not a platform: it does not enforce runtime conventions, only build-time ones.

**Best for:** Teams that already have CI and want to avoid introducing a separate deployment tool. This is the lowest-friction starting point, but it is not a substitute for a runtime platform.

### 7. A platform built on a service catalog

**What it does:** You define services in a catalog (Backstage is the widely documented open-source option), and the catalog drives scaffolding, documentation, and deployment links. New services are created from templates.

**Strength:** New-hire onboarding is the explicit design goal. A new hire opens the catalog, sees every service, its owner, its runbook, and its deploy button. Senior engineers get a single place to find things, which reduces the "who owns this?" problem.

**Weakness:** The catalog is a layer on top of your actual platform, not a platform itself. It does not deploy anything; it links to whatever does. If the underlying platform is inconsistent, the catalog exposes that inconsistency rather than hiding it. Also, maintaining a catalog is ongoing work, and it is easy to let it drift.

**Best for:** Organisations with enough services that discoverability is a real problem. For a team of five, this is overkill.

## The strongest default, and why

For most small teams, the strongest default is **a managed PaaS with a documented escape hatch to raw infrastructure**. That sounds like a compromise, but it is a deliberate sequencing decision.

The reasoning: a new hire's first week should be about learning the product, not the platform. A managed PaaS gets them to a deployed change on day one. The escape hatch matters for the senior engineer, but it should be **explicit and rare**, not the default path. If the senior engineer is bypassing the platform weekly, the platform is wrong and should be fixed, not worked around.

This is the opposite of the advice that teams often hear, which is to build a platform early because you will need it later. That advice assumes you know what you need later. You usually do not. A managed PaaS buys you time to learn what your actual deployment patterns are. Once you know, you can migrate to something more custom with evidence.

The exception is if you have a hard compliance requirement or an unusual workload (GPU scheduling, persistent state, custom networking). In that case, start with Kubernetes and a thin CLI, and accept the operational cost.

## Why the default is a managed PaaS: a worked example

Suppose a five-person team adopts Kubernetes with a thin CLI on day one. The reasoning is that they will need it eventually, so they might as well start there.

- Building the CLI, RBAC policies, and admission controllers: an estimate of two to four engineer-weeks. This is illustrative, not measured; the actual figure depends on the team's existing Kubernetes experience.
- Ongoing maintenance: cluster upgrades, controller updates, and RBAC reviews. A reasonable planning assumption is a fraction of one engineer's time each month, which is a recurring cost.
- Time for a new hire to ship their first change: dependent on how well the CLI documents itself, but plausibly a day or more of reading and pairing.

Now suppose the same team starts on a managed PaaS and migrates later, once they know their deployment patterns.

- Time to first deploy for a new hire: a push and a build, measured in minutes.
- Cost of migrating off the PaaS later: a project, not a task. It involves rewriting deployment config, re-establishing secrets, and re-testing the pipeline.
- The information gained: which services need custom networking, which need persistent state, and which are fine on a standard buildpack. That information is what makes the later platform decision correct rather than speculative.

The comparison is not "PaaS is faster" in the abstract. It is that the PaaS defers a large, speculative build until you have evidence about what you actually need. If the team's workload turns out to be standard, they may never need the custom platform. If it turns out to be unusual, they now know exactly which parts need custom handling.

The failure mode of this sequencing is a team that stays on the PaaS past the point where it fits, because migrating is unpleasant. The signal to watch for is the senior engineer spending more time working around the platform than using it. When that becomes the norm, the migration is overdue.

## How to measure whether your platform is working

Do not rely on impressions. Instrument three things:

1. **Time-to-first-deploy for a new hire.** Record the timestamp when a new engineer starts and the timestamp of their first merged production change. Track this per hire, not as an average, because the distribution is what matters.
2. **Escape-hatch frequency.** Count direct `kubectl` applies, manual console changes, and any other change that bypasses the platform. Kubernetes audit logs and cloud provider audit trails are the source. A rising count means the platform is missing a capability.
3. **Failed deploys and their cause.** Classify each failure as platform-related (the tooling was wrong or confusing) or change-related (the code was wrong). Platform-related failures are the ones the platform team should fix.

To compare options before committing, run a time-boxed spike: have one engineer deploy the same trivial service through two candidate platforms and record the steps, the commands, and the points where they had to consult documentation. This is cheap and produces evidence rather than opinion.

## Honorable mentions worth knowing about

**Docker Compose for local development.** Not a deployment platform, but it is one of the best tools for local reproducibility. If your deployment platform cannot be approximated locally with Compose, new hires will struggle. Worth the investment.

**Terraform or OpenTofu for infrastructure.** These define the platform itself, not the deployments. They are how you make the platform reproducible. The weakness is state management: a corrupted state file is a bad day. The strength is that the entire platform is reviewable in a pull request.

**A shared Makefile.** Unglamorous, but a `Makefile` with targets like `make deploy-staging` and `make logs` is often the most legible interface for a new hire. It has no magic, and it is easy to read. Senior engineers can ignore it and run the underlying commands.

## Options that look appealing but fail in practice

**A fully custom internal platform built before you have users.** This is a classic failure mode. A team spends months building a deployment platform, then discovers the product needs to change shape. The platform is now a liability. Build the platform when the pain of not having it exceeds the cost of building it.

**A platform with no escape hatch.** If the only way to deploy is through a UI, senior engineers will find a back door, and that back door will be undocumented. An explicit escape hatch is better than a hidden one.

**A platform that requires a ticket to deploy.** This optimises for control and destroys velocity. It also teaches new hires that deployment is someone else's job, which is the opposite of what you want.

**A platform with no local story.** If a new hire cannot run the deployment locally, they cannot debug it. They will become dependent on the senior engineer, which defeats the purpose.

## How to choose based on your situation

| Situation | Recommended option | Why |
|---|---|---|
| 1–5 engineers, standard web app | Managed PaaS | Fastest onboarding, lowest operational cost |
| Compliance or audit requirements | GitOps | Every change is a reviewed commit |
| Unusual workloads (GPU, stateful) | Kubernetes + thin CLI | Escape hatch is built in |
| Event-driven, single cloud | Serverless + IaC | Scaling and cost model fit |
| Mixed container and legacy binaries | Nomad | Runs both without extra layers |
| Already have CI, no platform | CI/CD pipeline stages | Lowest friction to start |
| Many services, discoverability problem | Service catalog | Onboarding is the design goal |

The decision is rarely permanent. The important thing is to pick something that gets a new hire deploying on day one and gives a senior engineer a documented way out. If both of those are true, you can migrate later without drama.

## Frequently Asked Questions

**How do I stop senior engineers from bypassing the deployment platform?**

You do not stop them; you make bypassing it visible and rare. The practical approach is to log direct changes (Kubernetes audit logs, cloud provider audit trails) and review them regularly. If bypasses are frequent, the platform is missing something, and the fix is to add that capability, not to tighten access. Treat bypasses as product feedback.

**What is the fastest way to onboard a new hire to deployments?**

Have them deploy a trivial change (a log line, a version bump) on their first day, using the same path everyone else uses. Do not give them a special onboarding environment, because that teaches a workflow they will never use again. Pair them with someone for the first production deploy, then let them do the second one alone.

**Should I build an internal deployment platform or use an existing tool?**

Use an existing tool until you have a specific, documented reason not to. Internal platforms are expensive to build and maintain, and the maintenance cost is ongoing. The exception is when your deployment needs are genuinely unusual (custom hardware, strict data residency, a workload no managed tool supports). Even then, start with the closest existing tool and extend it.

**Why does local reproducibility matter for deployments?**

Because debugging is the main activity of a new hire, and you cannot debug what you cannot run. If the deployment only works in production, every bug becomes a production investigation. Tools like Docker Compose, `sam local`, and `kind` (Kubernetes in Docker) exist to close this gap. They are imperfect, but imperfect local reproducibility is better than none.

## Final recommendation

Start with a managed PaaS and a `Makefile` that wraps its CLI. Write down the escape hatch — the exact commands a senior engineer would run to bypass the platform — in the repository's README. Then, in the next 30 minutes, open your current deployment documentation and check one thing: can a new hire deploy a change without asking anyone a question? If the answer is no, that is the gap to close first.
