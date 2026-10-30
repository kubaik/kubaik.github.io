# Keeping AI-Generated Code from Rotting Your Repository

## The failure mode: AI code that passes review but rots the repo

A common pattern in codebases that have adopted AI assistants heavily: a large fraction of lines arrive from a code completion tool, pass a quick human glance, and merge. Individually each change looks reasonable. Collectively they produce a repository with three properties that are expensive to reverse:

1. **Inconsistent idioms.** A hand-written service layer sits next to generated blob functions that duplicate business logic, use different naming conventions, and re-implement utilities that already exist in the repo.
2. **Plausible-looking dead weight.** Unused imports, docstrings that restate the function name, `except: pass` blocks inserted to make the code appear robust, and configuration keys that reference services that were never provisioned.
3. **Non-actionable tests.** Tests that assert a function returns something, or that mock the exact thing under test, giving green builds and zero confidence.

None of these are caught by a compiler. Most are not caught by a standard linter either, because they are syntactically valid and stylistically unremarkable. The cost shows up later: new contributors spend their first days working out which parts of the codebase are load-bearing, and on-call engineers page on silent failures that were swallowed by a bare except.

The rest of this article is about the two broad approaches to containing that debt — a lightweight rule-based linter enforced at the PR gate, and a managed governance platform that scores diffs — and how to decide between them without buying either sight-unseen.

## Approach A: rule-based linting at the pull-request gate

The cheapest intervention is to treat AI-generated code exactly like hand-written code, and add a small set of rules that target the specific smells generated code tends to produce. This is not a product category you need to buy; it is a configuration you can build on top of an existing linter.

The mechanics:

- A linter runs locally (fast, no network) and in CI.
- A config file selects both standard correctness rules and a handful of custom or plugin rules for AI-specific patterns.
- The CI job fails the pull request on any new violation, so the rule set is enforced rather than advisory.

A representative configuration, using `ruff` (a Python linter and formatter) with its standard rule prefixes plus custom rule codes:

```toml
[tool.ruff]
line-length = 120
select = [
    "F401",   # imported but unused
    "E713",   # test for membership should be 'not in'
    "S110",   # try-except-pass detected
    "TRY302", # useless try-except that just re-raises
]
```

The rule codes above are standard `ruff` selectors; `S110` and `TRY302` come from the `flake8-bandit` and `tryceratops` rule families that `ruff` implements. If you want rules that are not shipped by the linter, the usual pattern is a small custom plugin or a grep-based CI step, not a new tool.

What this catches well:

- Unused imports and variables left behind after generated code is edited.
- Bare `except: pass` and other silent-failure constructs.
- Docstrings and comments that are pure restatement.
- Import cycles and obviously unreachable branches.

What it does not catch:

- Hallucinated API endpoints or SDK methods that do not exist.
- Database queries missing an index, or joins that return the wrong cardinality.
- Business logic that is internally consistent but wrong.
- License conflicts from copied snippets.

Those require tests, type checking against real interfaces, and human review. A linter is a floor, not a ceiling.

## Approach B: a managed governance platform

A managed governance platform (the category sometimes marketed as an "AI governance layer") ingests pull requests, runs the diff through a hosted model or classifier, and returns a score indicating how likely the code is to be machine-generated, often with similar public snippets and their license headers attached.

Typical architecture:

- A lightweight pre-commit or CI hook that uploads the diff or file contents.
- A cloud service that classifies the code and stores results.
- A dashboard and an audit log, with policy controls for blocking merges.

What this category genuinely provides that a local linter does not:

- **An audit trail.** If a regulator, donor, or customer requires evidence that generated code was reviewed under a documented policy, a hosted system with retention produces that artifact. A local linter does not.
- **License and provenance surfacing.** Comparing submitted code against public repositories can surface copied snippets and their license obligations faster than manual review.
- **Central policy.** One place to define and change rules across many repositories.

What it costs, structurally — not as a specific price, but as the shape of the cost:

- Per-seat or per-node licensing, plus per-file or per-analysis compute.
- A network dependency on the critical path of code review. If the service is unreachable, the merge is blocked or the check is skipped, and both outcomes have consequences.
- Onboarding and process change: contributors must wait for a bot before merging, which changes the review loop.

The network dependency is the part teams most often underestimate. A review gate that requires a cloud round trip degrades badly on unreliable connections, and a gate that silently passes when the service is down provides no governance at all.

## How to measure both approaches before committing

Do not accept vendor claims or blog benchmarks. Instrument your own repository. The measurements below are what actually determine whether either approach pays for itself.

**1. Establish the baseline size of the problem.**

Count how much of the codebase is machine-generated. There is no reliable automated way to do this, but a useful proxy is to count commits by author identity or co-author trailer. Many AI coding tools add a `Co-authored-by:` trailer to commits; you can count those:

```bash
git log --since="6 months ago" --pretty=%B | grep -c "Co-authored-by:.*\(Copilot\|Cursor\|Claude\)"
```

This undercounts (not every tool adds a trailer, and not every developer keeps it), so treat the result as a lower bound and cross-check by sampling 20 files manually.

**2. Measure the linter's false positive rate.**

Run the candidate rule set across the whole repository without failing the build, and record every violation. Classify each as a true positive (worth fixing) or a false positive (correct code the rule flagged). The ratio determines how much `# noqa` noise the team will tolerate. A rule set that produces more than a handful of false positives per hundred files will be disabled by frustrated contributors within weeks.

**3. Measure time-to-first-feedback.**

For the local linter, this is the wall-clock time of the lint command on a cold cache:

```bash
time ruff check .
```

For a hosted platform, this is the time from pushing a commit to receiving a result, measured across a full working day so you capture queueing. The number that matters is the median, not the best case.

**4. Measure whether the tool actually prevents defects.**

This is the hard one and the one most teams skip. Take a sample of merged pull requests and, for each, record whether the tool flagged anything that a human reviewer subsequently changed. If the tool's findings never influence the merged code, it is not preventing defects regardless of its accuracy. A simple way to approximate this: search the repository history for commits whose message references the tool, and read a sample.

**5. Compute cost against a stated assumption.**

Any cost comparison requires you to state your own inputs, because seat prices, cloud rates, and engineer salaries vary. The arithmetic is simple; the inputs are yours:

- Annual license cost = seats × monthly price × 12.
- Annual compute cost = files analyzed per month × per-file price × 12.
- Annual maintenance cost = hours per month spent tuning rules and suppressing false positives × loaded hourly rate × 12.

For a five-person team on a platform priced at $49 per seat per month, the license line alone is 5 × 49 × 12 = $2,940 per year. Whether that is expensive depends entirely on what it prevents, which is why step 4 matters more than the price list.

## A worked decision example

Suppose a 60,000-line Python service, six contributors, roughly 40 pull requests per month, deployed on infrastructure where the engineering team already has a CI pipeline that runs tests in about four minutes.

**Option A (local linter):**

- Setup: add a config file and one CI step. Realistically half a day of one engineer's time.
- Ongoing: the false positive rate measured in step 2 determines maintenance. If the team suppresses two violations per month at five minutes each, that is 10 minutes per month.
- Effect: catches unused imports, bare excepts, and dead code at the PR gate, before a reviewer sees them.

**Option B (hosted platform):**

- Setup: account provisioning, CI integration, and a policy discussion. Realistically two to three days across the team, plus a training session.
- Ongoing: every pull request now waits on a network round trip. If that adds even 30 seconds to the median review start and the team merges 40 PRs a month, that is 20 minutes of aggregate waiting per month — small, but it is waiting on a third party, and it fails when the network does.
- Effect: adds provenance and license checks, and produces an audit log.

The decision hinges on whether the provenance and audit capability is a requirement. If a regulator or customer contract demands a documented, retained review trail, Option B is the only one that produces it, and the cost is a cost of doing business. If no such requirement exists, Option A captures most of the defect-prevention value at a fraction of the operational complexity.

A frequent mistake is adopting the platform first and defining the policy later. The policy — which rules block a merge, who reviews a flagged diff, what happens when the service is down — is the actual governance. The tool is just where the policy is enforced.

## Failure modes to watch for

**The gate that does not gate.** A linter configured to warn rather than fail will be ignored. If a rule matters, it must fail the build, and the team must agree on how to fix violations rather than suppress them.

**The suppression spiral.** Every false positive that is silenced with an inline ignore comment is a small piece of the rule set dying. Track the number of suppressions as a metric; a rising count means the rules no longer match the codebase.

**The network-dependent gate.** Any check that requires a remote service will eventually run when that service is unreachable. Decide in advance whether the build fails closed (blocking merges during an outage) or open (merging unchecked). Both are defensible; leaving it undecided means it fails unpredictably.

**Measuring adoption instead of outcomes.** Counting how many pull requests the tool touched says nothing about whether the code improved. Count defects caught, false positives tolerated, and reviewer time saved — or admit you are not measuring.

**Treating the linter as a substitute for tests.** A linter cannot tell you that a query returns the wrong rows. If generated code touches data access or business rules, the coverage has to come from tests that assert on real behavior, not from a static rule.

## Decision checklist

Work through these in order; the first "yes" that cannot be satisfied by a local linter points you toward a hosted platform.

- Is there a contractual, regulatory, or donor requirement for a retained audit trail of code review? If yes, a hosted system with retention is likely necessary.
- Does the codebase contain substantial third-party or open-source code where license provenance is a real risk? If yes, provenance checking has value beyond linting.
- Is the team's network reliable enough that a cloud round trip will not block merges during working hours? If no, prefer local tooling.
- Have you measured the false positive rate of your candidate rule set on your own repository? Do that before choosing anything.
- Can you state, in one sentence, what defect class the tool is expected to prevent? If not, you are buying a feeling, not a control.
- Is the annual cost, computed from your own seat count and analysis volume, less than the estimated cost of the defects it prevents? If you cannot estimate the latter, run a one-sprint pilot on a single repository first.

## What to do in the next 30 minutes

Pick one repository that has received a lot of AI-assisted commits. Run your existing linter across the whole tree with a rule set that includes unused imports and bare-except detection, and count the violations without failing anything:

```bash
ruff check . --select F401,S110 --statistics
```

The output gives you two numbers: how many violations exist, and how they cluster by rule. That is your baseline. If the count is large and concentrated in a few rules, a local linter enforced at the PR gate will pay for itself immediately. If the count is small, your problem is not lintable smells — it is semantic correctness, and the next step is test coverage on the generated code paths, not a governance platform.
===END===
