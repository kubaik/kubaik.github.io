# AI Static Analysis vs Rule-Based SAST: A Practical Comparison

## What this comparison actually covers

Static analysis tooling has split into two families, and the split is not just marketing. On one side are **rule-based (often called legacy) SAST engines**: Semgrep, Bandit, dependency scanners, and their many commercial equivalents. They match code against explicit patterns and known-vulnerability databases. On the other side are **AI-assisted analyzers**: tools that build a model of a codebase's data and control flow, generate or rank findings with a learned component, and integrate with hosted CI platforms.

The two families fail differently, cost differently, and are operated differently. This article compares them category-by-category, shows where each is genuinely strong, and gives you a way to measure the difference on your own repositories rather than trusting someone else's numbers. It also covers the failure modes that show up after adoption, which is where most comparisons stop.

Two representative stacks are used throughout:

- **AI-assisted stack**: a hosted code-scanning service with an ML-ranked rule engine, ML-based secret detection, and ML-prioritized dependency advisories, wired into pull-request checks.
- **Rule-based stack**: Semgrep with community and custom rules, Bandit for Python-specific anti-patterns, and a dependency scanner for known CVEs, wired into a CI job.

Specific vendors are named only where the behavior described is documented and stable. Where behavior varies by vendor, the category is described instead.

## Why the comparison matters now

Three things changed the economics of static analysis.

**Merge windows shrank.** When a team merges tens of pull requests a day, a scan that takes minutes per PR becomes a queue. A scan that takes seconds does not. Latency is no longer a nice-to-have; it determines whether the gate is enforced or bypassed.

**Modern code patterns are harder to match by regex.** Async/await misuse, template injection in Jinja2, GraphQL resolver authorization gaps, and JWT validation bypasses are control-flow and data-flow problems. A pattern matcher can encode some of them, but the rules are brittle and lag behind language and framework changes.

**Compliance language moved from "scanning" to "detecting."** Controls that ask for automated testing of common weakness classes, with evidence of remediation timelines, push teams toward tools that produce a defensible finding record. Whether a rule-based engine satisfies that depends on how well its rule set covers the weakness classes in scope.

None of this means rule-based SAST is obsolete. It means the decision is now about failure modes, not feature checklists.

## How the AI-assisted stack works

A hosted AI-assisted scanner typically combines three components.

**1. A semantic code engine with an ML layer.** The engine builds a database of the codebase (functions, calls, taint sources and sinks) and runs queries over it. The ML layer does not replace the queries; it ranks and filters results, and in some products it drafts new queries from a natural-language description. This matters: the deterministic part still produces the finding, and the learned part decides what you see first.

**2. ML-assisted secret detection.** Regex-based secret scanning catches known formats. A classifier can additionally catch encoded or split secrets, at the cost of a new false-positive profile that has to be tuned.

**3. ML-prioritized dependency advisories.** Instead of ranking by CVSS alone, the tool ranks by whether the vulnerable function is reachable from your code paths. This is the single highest-leverage change for most teams, because unreachable advisories are the bulk of dependency noise.

A minimal GitHub Actions workflow for a hosted scanner looks like this:

```yaml
name: AI Static Analysis
on: [push, pull_request]

jobs:
  code-scan:
    runs-on: ubuntu-latest
    permissions:
      security-events: write
      contents: read
    steps:
      - uses: actions/checkout@v4
      - uses: github/codeql-action/init@v3
        with:
          languages: python
          queries: security-and-quality
      - uses: github/codeql-action/analyze@v3
        with:
          category: "/language:python"
```

Two notes on this workflow. First, `queries: security-and-quality` is broader than `security-extended` and will produce more findings, including maintainability ones; pick deliberately. Second, the `ai: true` input that appears in some drafts of this workflow is not a documented input of the action. Do not add inputs you cannot find in the action's own documentation; an unrecognized input is either ignored or fails the step depending on the action's implementation, and neither outcome is what you want in a security gate.

## How the rule-based stack works

The rule-based stack is three tools with three different jobs, glued together by CI.

**Semgrep** is an AST-based pattern matcher. Rules are written in YAML, either from a community registry, a vendor-supplied pack, or in-house. Because the rules are explicit, you can read a finding and know exactly which pattern matched. That property is worth a lot during incident review.

**Bandit** is a Python-specific linter for security anti-patterns: `eval`, `pickle`, hardcoded passwords, weak hashing, and similar constructs. It is fast and its findings are easy to explain. It does not model data flow, so it cannot tell you whether a user-controlled value reaches a dangerous sink.

**A dependency scanner** reads manifest and lock files and matches versions against advisory databases. This is not code analysis, but it is usually run in the same job and reported through the same gate, which is why teams lump it under "SAST."

A realistic CI job for this stack:

```yaml
name: Rule-Based SAST
on: [push, pull_request]

jobs:
  scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install scanners
        run: pip install bandit semgrep
      - name: Bandit
        run: bandit -r . -f json -o /tmp/bandit.json || true
      - name: Semgrep
        run: semgrep --config=auto --json --output /tmp/semgrep.json || true
      - name: Fail on findings
        run: |
          python - <<'PY'
          import json, sys
          findings = 0
          for path in ('/tmp/bandit.json', '/tmp/semgrep.json'):
              try:
                  with open(path) as fp:
                      data = json.load(fp)
              except FileNotFoundError:
                  continue
              results = data.get('results', [])
              findings += len(results)
          if findings:
              print(f"{findings} finding(s)")
              sys.exit(1)
          PY
```

The `|| true` on each scanner step is intentional: the scanners exit non-zero when they find something, and the aggregation step is what decides the build result. Note also that the aggregation script above fails on *any* finding, including low-severity ones. That is a policy choice, and it is the most common cause of gate abandonment. A better policy filters by severity and by rule confidence before failing the build.

A custom Semgrep rule for a house anti-pattern looks like this:

```yaml
rules:
  - id: hardcoded-secret-assignment
    patterns:
      - pattern: $NAME = "..."
      - metavariable-regex:
          metavariable: $NAME
          regex: (?i)(secret|token|password|api_key)
    message: "Possible hardcoded credential assigned to $NAME"
    languages: [python]
    severity: ERROR
```

## Where each approach genuinely shines

**AI-assisted scanning wins on:**

- **Latency on large repositories.** The scan is incremental and diff-aware; a full scan of a large repo is slower than a diff scan but still typically faster than running three separate rule-based tools over the same tree.
- **Weakness classes that require data flow.** Taint tracking from an HTTP handler to a log call, or from a request parameter to a template, is exactly what a semantic engine is built for and what a pattern matcher approximates badly.
- **Dependency noise reduction.** Reachability-based prioritization is the highest-return feature for most teams, because it turns a weekly pile of advisories into a short, defensible list.
- **Onboarding.** Enabling a hosted scanner is a configuration change, not a rule-writing project.

**Rule-based scanning wins on:**

- **Explainability.** Every finding maps to a rule you can read. When a finding is wrong, you know why, and you can fix the rule in a pull request.
- **Offline and air-gapped operation.** Semgrep and Bandit run entirely in your CI with no egress. For environments where no build artifact or source file may leave the network, this is decisive.
- **Cost predictability.** The marginal cost of another scan is runner time. There is no per-seat license and no metered model inference.
- **Tunability for domain-specific anti-patterns.** If your organization has a house rule ("never construct SQL with f-strings in this service"), a five-line rule enforces it deterministically. A learned analyzer may or may not learn it, and you cannot inspect why.

## Failure modes after adoption

Comparisons usually stop at feature lists. These are the failure modes that show up months in.

**Rule-based: the ignored gate.** The gate fails on too many low-severity findings, so someone adds `continue-on-error: true` or a `|| true`, and the gate silently stops blocking. The tool is still installed and still reporting; nobody is reading. Detection: check whether the CI job can actually fail the build. A job that cannot fail is documentation, not a control.

**Rule-based: rule lag on new weakness classes.** A weakness class enters the CWE catalog, the community writes a rule months later, and in the meantime the class is uncovered. This is not a bug; it is the operating model. Mitigation is to track which weakness classes your compliance scope names and verify coverage explicitly rather than assuming the default rule pack covers them.

**Rule-based: dependency alert fatigue.** Weekly advisories accumulate faster than they are triaged. Teams either disable the scanner or stop reading it. Mitigation is reachability filtering if available, or a triage SLA with an explicit "accepted risk" record so the backlog is visible rather than invisible.

**AI-assisted: opaque false positives.** A learned ranking or generated rule can flag a broad pattern (for example, treating every f-string as a potential injection) and it is not obvious from the finding why. Mitigation is to require that every suppressed finding carries a written justification, and to review suppressions monthly. If the tool cannot explain a finding, the suppression record has to.

**AI-assisted: false negatives reported as low severity.** Some real findings arrive tagged informational and are never triaged. This is the mirror image of the rule-based failure mode and it is harder to detect, because the finding is present. Mitigation is to sample informational findings periodically and check whether any were misclassified.

**AI-assisted: hosted-service dependency.** Model downloads, license checks, and API calls mean the scanner is unavailable when the network or the vendor is. For a merge-blocking gate, that means either a documented bypass path or a fallback scanner. Decide which before the first outage, not during it.

**Both: gate time dominated by aggregation, not scanning.** In the workflow above, the scanners finish and then a Python script parses JSON and decides. On a busy repository with several bots pushing, that aggregation step is where races and flaky failures live. Keep the decision logic small and deterministic.

## How to measure this on your own repositories

Do not adopt either stack on the strength of someone else's benchmark. Repositories differ enough that the ranking can invert. Measure these five things.

**1. Wall-clock scan time, per change size.** Instrument the CI job to print timestamps around each scanner invocation and around the aggregation step. Record `git diff --stat` for the commit under test so you can bucket by change size. Compare medians, not means; a single 200k-line full scan will drag a mean into uselessness.

**2. Gate time.** Time from "scanner starts" to "required check reports a result." This is what developers actually experience. It includes queue time, which is why a hosted check can feel faster even when the scan itself is comparable.

**3. Findings, split by disposition.** For each tool, record counts of true positive, false positive, and "needs review." Assign dispositions by sampling: take a random sample of N findings per tool per week, have a reviewer classify them blind to which tool produced them, and report the sample proportions with the sample size attached. Do not report a precision number without the sample size; a precision estimate from twelve findings is noise.

**4. Coverage against your named weakness classes.** List the weakness classes your compliance scope or threat model names. For each tool, check whether it has a rule or query for that class and whether it fires on a deliberately planted test case. This is a coverage matrix, not a benchmark, and it is the only measurement that directly answers "are we compliant."

**5. Engineering hours to operate.** Track setup, rule tuning, suppression review, and triage separately. These are the costs that dominate after the first quarter, and they are the ones that never appear in a vendor comparison.

A worked example of the arithmetic, with illustrative inputs:

- Assume a repository with 40 pull requests per day and a 30-second median scan for the AI-assisted tool versus a 150-second median for the rule-based stack. The difference is 120 seconds per PR, or 4,800 seconds (80 minutes) per day of CI wall-clock time. At a fully loaded CI cost of $0.008 per minute on hosted runners, that is roughly $0.64 per day, or about $230 per year. Latency savings alone rarely justify a per-seat license.
- Now assume the rule-based stack produces 30 dependency advisories per week and 2 are reachable, while the AI-assisted tool surfaces those 2 directly. If triaging an advisory takes 6 minutes, the difference is 28 advisories × 6 minutes = 168 minutes per week, about 14.5 hours per month. That is the number that usually justifies the license, and it is a number you can measure before buying.

The point of the example is the method, not the figures. Substitute your own PR volume, scan times, and triage minutes.

## A decision checklist

Work through these in order. The first "yes" that applies usually decides it.

1. **Is the environment air-gapped or subject to no-egress rules?** If yes, the rule-based stack is the only option unless you self-host the entire AI-assisted toolchain, which is a different project.
2. **Does your compliance scope name specific weakness classes?** If yes, build the coverage matrix first. The tool that covers the named classes wins, regardless of latency.
3. **Is dependency alert volume the dominant pain?** If yes, prioritize reachability-based prioritization over scan speed.
4. **Is the median PR scan over two minutes and the merge cadence over twenty PRs a day?** If yes, latency is a real cost and a hosted incremental scanner earns its place.
5. **Do you have someone who will own rule quality?** If yes, the rule-based stack can be tuned to beat a general model on your specific domain. If no, a managed analyzer with a hosted rule set is the lower-maintenance choice.
6. **Can the gate actually fail the build today?** If not, fix that before changing tools. A gate that cannot fail is the most expensive thing in this article.

## Where the comparison is heading

The two families are converging. Rule-based engines are adding data-flow queries and ML ranking; AI-assisted scanners are exposing query languages and letting teams write deterministic rules alongside learned ones. The useful mental model is not "old versus new" but "explicit rules versus learned ranking," and most mature setups end up with both: deterministic rules for the anti-patterns you can name, and a semantic engine for the ones you cannot.

What will not change is the operating cost. A scanner that produces findings nobody triages is worse than no scanner, because it consumes attention and produces a false sense of coverage. Whichever stack you pick, the measurement that matters is the same: of the findings you saw, how many were real, and of the real ones, how many did you see.

## Do this in the next 30 minutes

Open your CI configuration and find the security scan job. Answer one question: **can this job fail the build?** Look for `continue-on-error: true`, `|| true` on the scanner step, or an aggregation script that always exits zero. If the job cannot fail, you have no gate regardless of which scanner runs inside it. Fix that first, then run one scan on a deliberately vulnerable test file (a hardcoded credential and an obvious injection sink are enough) and confirm the job goes red. That single check tells you more about your current coverage than any tool comparison.
