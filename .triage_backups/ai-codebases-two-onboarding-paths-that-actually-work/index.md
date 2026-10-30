# AI codebases: two onboarding paths that actually work

## The problem: AI code passes tests and still breaks production

A recurring failure mode in repositories that contain AI-generated code has nothing to do with syntax or unit tests. The generated file compiles, the linter is quiet, the test suite is green, and the code still breaks in staging or production because the assumptions baked into the generation prompt did not match the surrounding system.

The mismatch is usually semantic, not syntactic. A generated endpoint returns a paginated object while the frontend expects a flat list. A generated migration assumes an empty table. A generated cron job assumes a dry-run mode that does not exist. A generated config file hardcodes a connection string that was meant as a placeholder.

The reason this is hard to catch: tests are written against the same mental model that produced the code. If the prompt said "return user data," the test asserts that user data comes back. Neither the prompt nor the test encodes the contract that the frontend, the CI pipeline, the secrets manager, or the observability layer actually enforces. That contract lives in the rest of the repo, and it is exactly what generated code tends to miss.

This article compares two onboarding and governance models for repositories that already contain AI-generated code. Both are real patterns teams use. Neither is universally correct. The goal is to give enough detail to pick one deliberately rather than by default.

## Two models, one axis

The two approaches split on a single question: **is AI-generated code treated as an opaque input you test around, or as a versioned artifact you reproduce?**

- **Option A — AI-as-dependency.** Generated files live wherever they were committed. A scanning step in CI flags likely AI-generated content and surfaces it for review. Nothing about the onboarding flow changes.
- **Option B — AI-as-artifact.** Generated files are segregated, pinned to a prompt and model version, and regenerated deterministically as part of the build. Onboarding gains a step that reproduces those artifacts.

Everything below follows from that split.

## Option A: AI-as-dependency

In this model, the repository is unchanged. New contributors clone it, install dependencies, and run the existing test command. The only addition is a CI job that inspects diffs for likely AI-generated content and posts a summary.

A typical implementation uses a small scanner service plus a CI workflow. The scanner does not block merges; it annotates the pull request with a confidence score and, where available, the prompt that produced the file.

A minimal workflow looks like this:

```yaml
# .github/workflows/ai-scan.yml
name: ai-scan
on: [pull_request]

jobs:
  scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r tools/ai-scan/requirements.txt
      - run: |
          python -m ai_scan \
            --base "${{ github.event.pull_request.base.sha }}" \
            --head "${{ github.event.pull_request.head.sha }}" \
            --output ai-audit.json
      - uses: actions/upload-artifact@v4
        with:
          name: ai-audit
          path: ai-audit.json
```

The scanner itself is usually a thin wrapper around pattern rules. A common rule catches string interpolation inside SQL execution, which is a genuine injection risk:

```python
# Flagged: interpolated SQL
cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")

# Not flagged: parameterized query
cursor.execute("SELECT * FROM users WHERE id = ?", [user_id])
```

The second form is safe because the driver binds the parameter. The rule is deliberately narrow: it flags f-strings and `%` formatting inside `execute` calls, not the presence of SQL.

### Where Option A works well

- **No migration cost.** Existing onboarding docs, scripts, and CI config stay as they are.
- **Low operational surface.** The scanner is a stateless job. There is no artifact store, no build step, no new dependency for contributors to install.
- **Familiar tooling.** The workflow is ordinary CI. Contributors do not need to learn a new convention.

### Failure mode

Option A catches local, pattern-shaped mistakes. It does not catch semantic drift, because semantic drift is not visible in the diff. A generated handler that returns the wrong shape, a generated query that assumes the wrong index, a generated retry policy that conflicts with an upstream timeout — none of these trip a regex.

A representative incident: a generated FastAPI route returns a paginated envelope, the frontend expects a flat array, and the mismatch surfaces only when a staging environment exercises the real frontend. The scanner saw nothing. The unit tests passed because they were written against the same assumption.

Option A therefore depends on test coverage being strong enough to encode the real contracts. If critical paths are not covered, the scanner provides a false sense of safety.

## Option B: AI-as-artifact

In this model, generated code is treated like a third-party dependency. It is segregated, versioned, and reproduced.

The conventions vary, but the shape is consistent:

- Generated files live under a reserved path, commonly `gen/`.
- A lockfile records the exact model identifier, temperature, seed, and prompt hash for each artifact.
- A build step regenerates the artifacts from those inputs and fails if the output differs from what is committed.
- A dedicated test suite under `tests/ai/` asserts the invariants the generated code is supposed to satisfy.

A verification workflow looks like this:

```yaml
# .github/workflows/gen-verify.yml
name: gen-verify
on: [push, pull_request]

jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r gen/requirements-ai.txt
      - run: python -m gen_toolkit regen --config .gen-config.yaml --check
      - run: pytest tests/ai/ -v
```

The `--check` flag is the important part: it regenerates and diffs rather than overwriting, so a mismatch fails the build instead of silently changing the tree.

The config that makes this reproducible must pin every input that affects output:

```yaml
# .gen-config.yaml
model:
  name: <pinned-model-identifier>   # never a floating "latest" alias
  temperature: 0.0
  seed: 42
prompts:
  root: prompts/
  hash_algorithm: sha256
limits:
  max_file_size_kb: 100
```

### Where Option B works well

- **Reproducibility.** Given the pinned inputs, the same artifact can be regenerated later and diffed against what is deployed.
- **Semantic auditing.** The `tests/ai/` suite can assert invariants that pattern rules cannot express, such as "every paginated endpoint includes a `next_cursor` field."
- **Rollback by version pin.** Reverting a bad artifact is a change to the lockfile, not a revert of a merge commit and a full CI cycle.

### Failure mode

Option B adds a build step, a lockfile, and a convention that contributors must learn. More importantly, it only delivers its benefits if the inputs are genuinely pinned. A floating model alias, a null seed, or an unpinned prompt root silently converts the deterministic build into a non-deterministic one, and the failure appears as flaky CI rather than as a clear error.

A representative incident: a config uses a floating model alias. The provider updates the model behind that alias. The regenerated output differs in field ordering, the diff check fails intermittently depending on which backend serves the request, and the team spends days attributing the flakiness to the CI runner before finding the alias.

## Measuring the tradeoff yourself

Published benchmark numbers for this comparison are not meaningful, because the result depends entirely on repository size, test coverage, model, and infrastructure. The honest approach is to measure in your own repo. The instrumentation is straightforward.

**Onboarding time.** Time a fresh clone through first passing test run, on a clean machine, for three new contributors. Record wall-clock time and the step where each one stalls. Do not measure on a machine with a warm cache.

**Scanner or build overhead.** In CI, record the duration of the scan or regeneration step across at least 50 runs. Report the median and the 95th percentile, not the mean — the tail is what developers notice.

**Storage growth.** Run `git count-objects -vH` before and after a month of normal commits. For Git LFS, `git lfs ls-files | wc -l` and the total size reported by your host give the real figure.

**Incident recovery time.** For each incident traced to generated code, record the time from detection to the point where the fix is deployed. Classify by recovery method: revert, pin, or patch. This is the metric that most often decides the choice, and it is the one teams rarely instrument.

**False positive and false negative rates.** For the scanner, sample 100 flagged files and 100 unflagged generated files, and have a reviewer classify each. The unflagged sample is the important one; it estimates what the scanner misses.

Once these five numbers exist for your repository, the decision usually becomes obvious without reference to anyone else's data.

## Comparing the two models

| Dimension | Option A (dependency) | Option B (artifact) |
|---|---|---|
| Onboarding change | None | Adds a regeneration step |
| Reproducibility | Not provided | Provided, if inputs are pinned |
| Semantic drift detection | Relies on existing tests | Dedicated invariant suite |
| Rollback mechanism | Revert commit, re-run CI | Change version pin |
| New conventions | None | Reserved path, lockfile, test dir |
| Failure mode | Silent semantic drift | Non-determinism from unpinned inputs |
| Prerequisite | Strong test coverage | Pinned model, seed, prompts |

The table is a decision aid, not a benchmark. The actual numbers depend on your repository.

## A worked example of the cost math

Assume a team of 20 developers, each cloning the repository twice per day, and a regeneration step that adds 2 seconds per clone. That is:

```
20 developers × 2 clones/day × 2 s = 80 s/day
80 s/day × 20 working days = 1,600 s/month
1,600 s ÷ 3,600 = ~0.44 developer-hours/month
```

Under half a developer-hour per month. That is almost never the deciding factor.

Now assume the same team has one AI-related incident per month, and the recovery time differs by 25 minutes between the two models:

```
1 incident/month × 25 min saved = 25 min/month
```

Still small in isolation. The decision only becomes significant when the incident rate is higher, when the recovery difference is larger, or when the incident has customer-visible impact. The point of doing this arithmetic is to avoid choosing a model based on a latency difference that does not matter, while ignoring the recovery difference that might.

## Decision checklist

Work through these in order. The first question that resolves to a clear answer usually settles the choice.

1. **Can you pin every input that affects generation?** Model identifier, temperature, seed, and prompt content. If any of these cannot be pinned — for example, the provider only offers a floating alias — Option B's core benefit is unavailable and Option A is the better fit.
2. **Are your critical paths covered by tests that encode real contracts?** If coverage is thin, Option A's scanner will not compensate, and the repository is exposed either way. Fix coverage first.
3. **Do you have more than roughly a hundred generated files in production?** Below that, the convention overhead of Option B rarely pays for itself. Above it, manual review stops scaling.
4. **Is rollback speed a real constraint?** If a bad generated artifact can cause customer-visible damage within minutes, version pinning is worth the setup cost. If generated code only touches internal tooling, it usually is not.
5. **Will your infrastructure support the artifact store?** If Git LFS or an equivalent is unavailable or unsupported, Option B is not viable regardless of the other answers.
6. **Are you in a regulated environment that requires reproducible builds?** If so, Option B is effectively mandatory for the artifacts in scope, even at small scale.

## Integration points that matter

Regardless of which model you choose, three integration points determine whether governance actually works.

**Secrets scanning.** Pattern-based scanners that look for injection risks typically do not look for credentials. Add a dedicated secrets scan over generated files, and treat a hit as a release blocker rather than a warning. Generated code is more likely than hand-written code to contain placeholder credentials copied from a prompt.

**Error tracking with artifact provenance.** Attach the artifact identifier, model identifier, and prompt hash to error reports for generated files. When an incident occurs, this turns "some generated code broke" into "this prompt, this model version, this seed." The instrumentation is a few lines at the point where the artifact is loaded, and it is the single highest-value addition for reducing recovery time.

**Prompt versioning.** Prompts are inputs. If they are edited in place, the artifact lockfile no longer describes what is deployed. Store prompts in version control under a dedicated directory and hash them as part of the build. A prompt change should be a reviewable diff, not an untracked edit.

## Common failure modes to guard against

**Floating model aliases.** A config that names a model alias rather than a pinned version will produce different output when the provider updates the alias. Always pin.

**Null or absent seeds.** A missing seed makes generation non-deterministic, which turns the diff check into a source of flaky failures. Validate the config at build time and fail loudly if the seed is unset.

**Unbounded artifact growth.** Generated artifacts accumulate. Set a size limit per file and a retention policy per artifact, and prune on a schedule rather than when the storage quota is reached.

**Scanner rules that do not match the real risk.** A rule that flags every use of string formatting in a logging call will be ignored within a week. Rules should target specific, demonstrable risks, and each rule should have a test case in both directions.

**Treating generated code as exempt from review.** The provenance of a file does not change its behavior in production. Generated code should pass the same review bar as hand-written code, with the added check that its inputs are pinned.

## Where the two models converge

In practice, mature repositories end up with elements of both. A lightweight scanner runs on every pull request to catch pattern-shaped mistakes quickly and cheaply. A smaller set of high-risk artifacts — anything touching payments, authentication, data deletion, or external contracts — is versioned and regenerated deterministically. The scanner handles breadth; the artifact pipeline handles the cases where a silent semantic error is expensive.

That combination is usually the right target. The question is not which model to adopt permanently, but which to adopt first given the current state of the repository.

## Next 30 minutes

Open your repository and check two things. First, list every file in the last 90 days of commits that was produced by a code generation tool, and count them. Second, open the config or prompt that produced the highest-risk one and check whether the model identifier, temperature, and seed are all explicitly pinned.

If the count is above roughly a hundred and any of those three inputs is unpinned, pin them now — that single change converts an unreproducible artifact into a reproducible one and is the prerequisite for everything else in this article.
