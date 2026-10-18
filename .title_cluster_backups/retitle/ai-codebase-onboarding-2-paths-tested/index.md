# AI codebase onboarding: 2 paths tested

## The problem this comparison addresses

A repository where a large share of code was produced by an LLM fails the standard onboarding checklist. A README, a CI pipeline and a pull-request template assume that the code in the repo was written by people who are still around to explain it. Generated code breaks that assumption in specific, predictable ways:

- **Stale API usage.** The model was trained on a snapshot and reproduces methods that the team has since deprecated or replaced.
- **Non-deterministic diffs.** Re-running the same prompt produces different output, so "regenerate it and see" is not a reliable debugging step.
- **Locally green, CI red.** Generated tests often mock the wrong boundary, so they pass on a laptop and fail in a pipeline with real dependencies.
- **Policy violations that compile.** Imports of banned libraries, unbounded retries, hardcoded timeouts, and functions that exceed the repo's length limit all pass a type checker.

Two onboarding strategies are commonly used to get a new engineer productive in this environment:

1. **Shadow-mode onboarding.** The new hire reads merged PRs, runs the test suite locally, and traces how the AI-generated code fits together before writing anything. Code ownership comes later.
2. **Guided generation.** Tooling scaffolds the ticket, the prompt, the tests, and the PR description from a repo-aware template, so the new hire directs the model rather than reading its output.

The choice between them is not about which is "better." It is about which failure mode a team can absorb: slow ramp-up (shadow-mode) or prompt drift and tool dependency (guided generation).

## Option A: shadow-mode onboarding

Shadow-mode treats generated code the way a careful reviewer treats any unfamiliar contributor: observe first, act later.

A typical playbook:

1. Clone the repo and run the bootstrap target (for a Go service, something like `make bootstrap` that pins the toolchain, linter, and cloud CLI).
2. Filter the merged PR queue for files carrying generation markers, such as a `// generated` header or a commit trailer naming the tool.
3. For each such file, read the diff, run the test suite with race detection and caching disabled (`go test ./... -race -count=1`), and ask in the team channel why the model chose a given pattern.
4. Once the new hire can predict what the model will produce for a given ticket, they open their first PR.

### Where shadow-mode shines

- **Auditability.** Every change, generated or not, passes through the same review queue. A reviewer can grep for a banned import before it reaches the default branch.
- **Durable knowledge.** The new hire builds a mental model of the model's habits. A common one: the assistant reaches for `context.Background()` where the service requires `context.WithTimeout`, which silently breaks a request deadline.
- **Environment parity.** A pinned bootstrap script and a devcontainer mean every engineer has the same compiler, linter, and extensions.

### Where shadow-mode fails

- **Slow first contribution.** Weeks can pass before the new hire writes code that ships.
- **Attrition risk.** Engineers who joined to build things may read this as being benched.
- **Unmeasured cost.** The idle time is real but rarely appears on any dashboard, so it is easy to under- or over-estimate.

## Option B: guided generation

Guided generation inverts the order: the new hire acts first, and the tooling constrains what the model can produce.

A typical flow:

1. A scaffolding CLI inspects `package.json`, `tsconfig.json`, and the existing PR templates.
2. It emits a prompt file with placeholders for the function name, error-handling strategy, and logging conventions.
3. The engineer fills in the placeholders and runs the generator, which writes the source files and a reviewable PR.
4. The PR description is populated from the prompt plus a generated diff summary.

### Where guided generation shines

- **Fast feedback.** The prompt is iterated before code exists, so the first PR is closer to mergeable.
- **Policy enforcement at generation time.** A template can require a schema-validation library, refuse to emit functions over a length limit, and reject imports on a deny list — before a human ever reviews the diff.
- **Editor continuity.** A shared settings file can point the in-editor assistant at the same lint configuration the repo uses, reducing context switching.

### Where guided generation fails

- **Prompt drift.** After a few weeks, engineers bypass the scaffold and type raw prompts into a chat panel, producing inconsistent formatting and structure.
- **Second source of truth.** The scaffold's configuration file slowly diverges from the README and from the actual conventions in the code.
- **Release-cadence inheritance.** A third-party generator updates on its own schedule. A template change can break a naming convention across the repo overnight, and the team has to pin the version and wait.

## How to measure the trade-off in your own repo

None of the numbers below are portable between organizations. What is portable is the instrumentation. Measure these four things before choosing, and re-measure after.

### 1. Time to first merged non-trivial PR

**What to instrument:** the merge timestamp of each new engineer's first PR that touches more than a trivial change (say, more than 20 changed lines and at least one test file).

**How to compute it:** from your git host's API, take the PR creation date minus the engineer's start date, filtered to the first qualifying PR. Report the median and the maximum, not the mean — the tail is where the interesting failures live.

**Why the maximum matters:** a long tail usually means one specific class of generated mistake (a wrong mock, a stale SDK call) trapped an engineer for days. That is a fixable tooling problem, and the tail tells you where to look.

### 2. CI failure rate on AI-touched PRs

**What to instrument:** the fraction of PRs containing at least one generated file that fail CI at least once before merging. Tag PRs at creation time using a commit trailer or a file marker so the classification is automatic.

**How to compute it:** failed-first-run PRs divided by total AI-touched PRs, per week. Break it down by failure category (lint, test, build, policy) rather than reporting a single number.

### 3. Queue depth at a fixed time of day

**What to instrument:** the number of queued or running pipeline jobs at a fixed hour, sampled daily.

**How to compute it:** query your CI provider's API on a schedule and store the result. A rising trend with a flat PR volume indicates that individual failures are getting more expensive to diagnose, not that the team is shipping more.

### 4. Setup friction

**What to instrument:** the wall-clock time from a fresh machine to a green local test run, measured by a script rather than self-reported.

**How to compute it:** a container that clones the repo, runs the documented bootstrap, and runs the test suite, timed end to end. Run it weekly in CI. If this number grows, every new hire pays the increase.

### A worked example of the arithmetic

Suppose a team hires four engineers a year, and the fully loaded cost of an engineer is $150,000 per year. That is roughly $72 per working hour (150,000 ÷ 2,080 hours).

If shadow-mode onboarding adds 40 hours of pre-contribution reading per hire, the cost is 4 × 40 × $72 = $11,520 per year. If guided generation cuts that to 12 hours per hire, the cost is 4 × 12 × $72 = $3,456, a difference of $8,064.

These figures are illustrative. Substitute your own loaded hourly rate and your own measured ramp-up hours. The point of the arithmetic is not the total; it is that the idle-time cost is usually larger than the tooling subscription, which means the decision should hinge on measured ramp-up hours, not on license price.

## A head-to-head comparison

| Dimension | Shadow-mode | Guided generation |
|---|---|---|
| Time to first PR | Longer | Shorter |
| Review burden | Same queue, more reading per PR | Fewer, more uniform PRs |
| Policy enforcement | At review time | At generation time |
| Toolchain dependency | Repo bootstrap only | Generator CLI plus editor extension |
| Main failure mode | Slow ramp, attrition | Prompt drift, config divergence |
| Best fit | Regulated code, heterogeneous editors | Mid-size repos, uniform editor stack |

## Failure-mode analysis

### Failure mode 1: the model's stale API

A generated file imports a client library method the team replaced months ago. In shadow-mode, a reviewer catches it. In guided generation, a deny-list rule catches it before the file is written. In both cases the fix is the same: encode the current API surface in a lint rule or a generator constraint, so the correction does not depend on a human remembering.

### Failure mode 2: the wrong mock

Generated tests mock a service that the code does not actually call, so the test passes while the integration is broken. This is the single most expensive onboarding trap, because the new hire's local run is green. The countermeasure is a small number of real integration tests that run in CI against a containerized dependency, so a wrong mock fails fast.

### Failure mode 3: prompt drift

After the novelty wears off, engineers stop using the scaffold and type raw prompts. The output becomes inconsistent. The countermeasure is a fingerprint comment in each generated file that records the prompt template version, plus a lint rule that flags files without it. This is a tax on every change, so weigh it against the consistency it buys.

### Failure mode 4: the scaffold as second source of truth

The generator's config file describes conventions that no longer match the code. The countermeasure is to generate the documentation from the config, or to fail CI when the two disagree. Never maintain both by hand.

## A decision checklist

Answer these before committing to a path.

1. **What is the measured median and maximum time to first merged PR today?** If you do not know, measure it for one cohort before changing anything.
2. **What fraction of CI failures on AI-touched PRs are policy violations versus logic errors?** Policy violations are cheap to automate away; logic errors are not.
3. **Does the team share one editor stack?** Guided generation's tooling assumes it does. A heterogeneous team pays a friction tax on every generator update.
4. **Can the repo enforce its conventions mechanically?** If the answer is no, neither approach will help, because the conventions live only in reviewers' heads.
5. **Who owns the generator configuration?** If nobody does, it will drift.
6. **What is the blast radius of a bad generator release?** If the answer is "the whole repo," pin the version and review upgrades deliberately.

## Practical recommendation

Guided generation tends to pay off when the repository is large, most engineers share an editor stack, and the team can afford to own a generator configuration. Shadow-mode tends to be the safer default when the code is security-critical, the editor stack is mixed, or the team has no capacity to maintain scaffolding.

Two situations where shadow-mode remains the better choice even in a modern toolchain:

- **Regulated code.** Where every generated line must be diff-reviewed by a human before merge, the review queue is the control, and a generator that bypasses it adds risk rather than removing it.
- **Unstable generated output.** In a codebase translated wholesale from another language, the model's output may be inconsistent enough that a scaffold becomes a crutch. Reading first is the only way to build a reliable mental model.

Neither approach is a silver bullet. Guided generation inherits the release cadence of its tooling, and shadow-mode inherits the patience of its hires.

## Frequently asked questions

**How do I know whether my repo is ready for guided generation?**

You do not need a threshold; you need three measurements. Time to first merged PR, CI failure rate on AI-touched PRs broken down by category, and setup time from a fresh clone to a green test run. If policy violations dominate the failure categories, guided generation will help. If logic errors dominate, the problem is test design, not onboarding.

**What is the most common mistake with shadow-mode?**

Letting the new hire write code too early. The value of shadow-mode comes from building a predictive model of the assistant's behavior. If the new hire opens a PR in the first week, they reproduce the same mistakes the model already made, and the team learns nothing.

**Does guided generation work with editors other than VS Code?**

It can, but the friction is in the generator's runtime and configuration, not the editor. If the team cannot standardize on a runtime for the scaffold, the maintenance cost usually outweighs the speed gain, and shadow-mode is the better default.

**Should the two approaches be combined?**

Yes, and this is often the practical answer. Use guided generation for the first small, well-scoped ticket so the new hire ships something in week one, then switch to shadow-mode reading for the next two weeks while they build context. The first PR provides early feedback; the reading period prevents the drift that comes from relying on the generator alone.

## One action for the next 30 minutes

Pick the largest generated file in your repository. Run `git log --format='%ae' -- <path> | sort | uniq -c | sort -rn` on it to see who touched it most, then open one of those commits and read the diff alongside the current file. If you cannot explain why the code looks the way it does, that gap is exactly what a new hire will hit on day one — write it down as the first entry in an onboarding note.
