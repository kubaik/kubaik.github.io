# Context-aware security gates that cut CI alert noise

## Why CI security scanning drowns in noise

Enabling security scanning on a repository is usually a one-way door: the tool starts reporting, and the volume of findings per pull request tends to grow faster than the team's capacity to triage it. The common failure mode is not a bad scanner. It is that scanners evaluate the whole repository on every run, while a pull request only changes a handful of files. The result is a long list of findings that are technically true but irrelevant to the change under review.

Typical symptoms:

- A dependency scanner flags a vulnerable package in `devDependencies` on a PR that only edited a CSS file.
- A container scanner reports an exposed port that is an intentional health endpoint.
- A static analyzer reports a SQL-injection pattern in a file that was not touched, and in a framework that uses parameterized queries by default.
- A bot opens dependency-bump PRs that compete for review attention with feature work.

When the gate blocks merges on all of this, teams respond predictably. They disable rules, comment out steps, or add broad ignore files. The gate then stops providing value, and the noise is replaced by a false sense of coverage.

The fix is not to scan less. It is to filter findings by the intent of the change: what files did this PR touch, and is this finding relevant to those files? This article describes a policy-as-code layer that does exactly that, sitting between the scanners and the merge gate.

## The core idea: filter by diff, not by confidence

Most scanners already let you exclude paths and rules globally. Global exclusions do not scale, because the same finding can be legitimate in one context and noise in another. A port mapping in a local test container is noise; the same port mapping in a production Dockerfile is a real finding. A global exclusion cannot express that distinction.

A context-aware policy expresses it as a condition on the PR diff:

- If the PR changed `Dockerfile`, evaluate container findings normally.
- If the PR did not change `Dockerfile`, suppress container findings about that file.
- If the PR changed `package.json` or a lockfile, evaluate dependency findings.
- If it did not, suppress dependency findings, since the dependency graph did not change.

This is the whole trick. Findings are attached to files or dependency manifests; the PR diff tells you which of those changed; anything outside the changed set is, for the purposes of this PR, out of scope.

Two properties make this work well:

1. **It is conservative.** You are not deleting rules. You are scoping them to the change. A finding that survives the filter is one a reviewer should look at.
2. **It is auditable.** Every suppression carries a reason string in a versioned file, so a reviewer can see why a finding was dropped and challenge it.

## A policy file format

The policy layer reads a single YAML file per repository. A minimal format looks like this:

```yaml
# security-policy.yml
rules:
  - id: trivy-exposed-port-8080
    when: changed('Dockerfile', 'docker-compose.yml')
    action: ignore
    reason: "8080 is the health endpoint; only relevant when container config changes"

  - id: snyk-dev-dependency
    when: not changed('package.json', 'package-lock.json', 'yarn.lock')
    action: ignore
    reason: "devDependency CVEs only matter when the dependency graph changes"

  - id: sql-string-interpolation
    when: changed('src/**/*.py') and uses_framework('django')
    action: require_review
    reason: "Django ORM uses bound parameters; review only when query code changes"

scanners:
  - name: container-scanner
    findings: trivy.json
  - name: dependency-scanner
    findings: snyk.json
  - name: static-analyzer
    findings: sonar.json
```

Three actions are enough for most teams:

- `ignore` — drop the finding entirely for this PR.
- `require_review` — keep the finding, surface it in the PR comment, but do not block the merge.
- `block` — fail the gate.

The `when` expression is evaluated against a context object built from the PR diff. That context needs, at minimum:

- the set of changed file paths,
- the set of changed dependency manifests,
- a per-file language or framework hint (derived from extension and a small lookup table).

## Implementing the gate

The gate itself is a small program with one job: read the policy, read the scanner outputs, read the diff, and emit a filtered result plus an exit code. It does not need to be a service. A single script that runs as a CI step is easier to test and debug than a long-running process.

```python
# policy_gate/filter.py
import fnmatch
from dataclasses import dataclass
from typing import Iterable

@dataclass
class Finding:
    scanner: str
    rule_id: str
    path: str | None
    severity: str

@dataclass
class Rule:
    id: str
    when: str
    action: str
    reason: str

def changed(ctx, *patterns: str) -> bool:
    """True if any changed path matches any glob pattern."""
    return any(
        fnmatch.fnmatch(path, pattern)
        for path in ctx.changed_paths
        for pattern in patterns
    )

def not_changed(ctx, *patterns: str) -> bool:
    return not changed(ctx, *patterns)

def uses_framework(ctx, name: str) -> bool:
    return name in ctx.frameworks

def evaluate(expr: str, ctx) -> bool:
    # Only expose the helpers above; never use bare eval on untrusted input.
    allowed = {"changed": changed, "not_changed": not_changed,
               "uses_framework": uses_framework, "ctx": ctx}
    return bool(eval(expr, {"__builtins__": {}}, allowed))

def apply_policy(rules: Iterable[Rule], ctx, findings: Iterable[Finding]):
    kept, suppressed, review = [], [], []
    for finding in findings:
        match = next((r for r in rules if r.id == finding.rule_id), None)
        if match is None:
            kept.append(finding)
            continue
        if not evaluate(match.when, ctx):
            kept.append(finding)
        elif match.action == "ignore":
            suppressed.append((finding, match.reason))
        elif match.action == "require_review":
            review.append((finding, match.reason))
        else:
            kept.append(finding)
    return kept, suppressed, review
```

Two things to note in this code, both of which matter in practice:

- The `eval` call runs only expressions from a file committed to the repository and reviewed like any other code. Even so, it is scoped to a whitelist of helper functions with `__builtins__` removed. If your policy file can ever come from an untrusted source, replace `eval` with an explicit parser.
- Rules are matched by `rule_id`. That means each scanner's output must be normalized into the `Finding` shape before filtering. Normalization is where most of the integration work lives, not in the filter itself.

## Wiring it into CI

The workflow below runs the scanners, normalizes their JSON, and then runs the gate. Note the `|| true` on scanner steps: a scanner that exits non-zero because it found something should not abort the job before the gate has a chance to filter.

```yaml
# .github/workflows/security-gate.yml
name: security-gate

on:
  pull_request:

jobs:
  gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Compute changed files
        run: |
          git diff --name-only \
            "origin/${{ github.base_ref }}...HEAD" > changed.txt

      - name: Run scanners
        run: |
          trivy fs --format json --output trivy.json . || true
          snyk test --json > snyk.json || true
          sonar-scanner -Dproject.settings=sonar-project.properties || true

      - name: Normalize scanner output
        run: python -m policy_gate.normalize trivy.json snyk.json sonar.json > findings.json

      - name: Apply policy
        id: gate
        run: |
          python -m policy_gate.filter \
            --policy security-policy.yml \
            --changed changed.txt \
            --findings findings.json \
            --output filtered.json

      - name: Fail on blocking findings
        if: steps.gate.outputs.exit_code == '1'
        run: exit 1
```

Two details that are easy to get wrong:

- **Diff base.** `git diff --name-only origin/<base>...HEAD` uses the merge base, so it lists only files this branch changed. Using `HEAD~1` instead will miss files in multi-commit PRs and produce confusing suppressions.
- **Exit codes.** Keep the gate's exit codes distinct: `0` for clean, `1` for blocking findings, `2` for a policy or input error. A `2` should fail the job loudly, because it means the gate did not run, and a gate that silently passes when it is broken is worse than no gate.

## Failure modes to design against

**Suppression drift.** A rule written for one PR context can quietly suppress real findings later. Mitigation: require every rule to carry a `reason` string, and add a scheduled job that runs the gate against the default branch with an empty diff. Any rule that would suppress a finding on the default branch should be reviewed.

**Over-broad globs.** A pattern like `src/**` will match nearly everything and turn the gate into a rubber stamp. Keep patterns narrow and test them. A small unit test per rule, using a fixture diff and a fixture finding, catches most mistakes.

**Scanner output schema changes.** Normalization code that assumes a specific JSON shape breaks silently when a scanner updates. Pin scanner versions in CI, and add a schema check in the normalize step that fails loudly on unexpected keys.

**The gate becomes the bottleneck.** If filtering is correct but the scanners still take longer than the review cycle, the gate is not the problem. Measure the scanner runtime separately from the filter runtime before optimizing the wrong component.

**Review fatigue from `require_review`.** If everything is `require_review`, the PR comment becomes the new noise. Reserve it for findings that are real but not fixable in the current PR, and route them to a tracker.

## How to measure whether it is working

There is no universal number to quote here, because the baseline depends on your stack. What matters is that you measure before and after on the same repositories. Instrument the following:

- **Findings per PR, split by scanner and by action taken.** Emit a small JSON artifact per run containing counts of kept, suppressed, and review findings. Aggregate over a week.
- **Gate runtime.** Time the normalize and filter steps separately from the scanner steps. The filter should be well under a second for typical finding counts; if it is not, the bottleneck is elsewhere.
- **Merge time with and without the gate.** Query your forge's API for PR open-to-merge duration, and segment by whether the security check ran. This tells you the real cost of the gate.
- **Suppression rate per rule.** A rule that suppresses almost everything it matches is either very effective or too broad. Sample a few of its suppressions by hand each month.

A simple sanity check you can run on day one: take the last twenty merged PRs, replay their diffs against the current scanner outputs, and count how many findings the policy would have suppressed. That gives you a rough estimate of the noise reduction without waiting for new PRs.

## Migrating from global exclusions

Most teams already have a pile of exclusions scattered across tool configs. The migration is mechanical:

1. List every exclusion, with the reason it was added.
2. For each one, ask whether it is unconditional or tied to a file change. Unconditional exclusions (vendored code, generated files) stay as scanner-level path excludes. Conditional ones become policy rules.
3. Add the rule with a `when` condition and a `reason`.
4. Delete the old exclusion.
5. Run the gate in report-only mode for a week: log what it would suppress and what it would block, but do not fail the build. Compare against the old behavior before switching to enforcement.

Do this in small batches. A migration that changes the gate's behavior in fifty places at once is impossible to debug when a real finding slips through.

## Decision checklist

Before adopting diff-scoped filtering, confirm:

- [ ] Scanner output can be normalized into a common finding shape with a stable rule identifier and a file path.
- [ ] The CI system can compute the merge-base diff for a PR.
- [ ] Policy files will be reviewed like code, with an owner.
- [ ] There is a place to send `require_review` findings (a tracker, not a PR comment that scrolls away).
- [ ] Someone will look at suppression rates monthly.

If any of these is missing, fix that first. The filtering logic is the easy part.

## FAQ

**Does this replace the scanners' own configuration?**
No. It complements it. Path-level excludes for vendored or generated code still belong in the scanner config. The policy layer handles the conditional cases that scanner configs cannot express.

**What if a finding has no file path, such as a repository-level configuration issue?**
Treat findings without a path as always in scope. They are rare, and suppressing them by default is how real misconfigurations get missed.

**Can the same policy file serve a monorepo with several languages?**
Yes, because the conditions are about paths and manifests, not languages. The scanners are language-specific; the policy is not.

**Should secrets scanning go through the same filter?**
Generally no. A leaked credential is a real finding regardless of which file changed, and the cost of a false positive is much lower than the cost of a miss. Keep secrets scanning unconditional.

**How long should the raw scanner output be retained?**
Long enough to replay the gate against a past PR when a finding reappears. Thirty days of raw JSON in object storage is a reasonable default for most teams.

## Next step

Pick one repository, list its existing scanner exclusions, and rewrite the five most frequently triggered ones as conditional rules in a `security-policy.yml`. Run the gate in report-only mode on the next ten pull requests and count how many findings it would have suppressed. That count is your evidence for whether to roll it out further.
