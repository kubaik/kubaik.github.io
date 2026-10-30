# Stop false positives in CI scanning now

Most automated scanning guides assume a clean environment and a patient timeline. Production CI gives you neither. This article describes a triage layer that sits between vulnerability scanners and pull request comments, and why static ignore files tend to fail at scale.

## The problem: noise, not detection

Security scanning is a checkbox for most engineering teams, and the checkboxes are easy to satisfy. A typical setup runs two or three scanners on every pull request: a dependency scanner such as a Snyk CLI release, an image scanner such as Trivy, and a static analysis engine. Each tool has its own ignore syntax, its own severity model, and its own idea of what counts as a finding.

The failure mode is not that scanners miss things. It is that they report too much, and the reports are often wrong for your context. A dependency flagged as vulnerable may be present only in a build-time tool that never ships. A container finding may apply to a base image layer that is replaced at deploy time. A static analysis rule may fire on test fixtures. Multiply that by every pull request in every repository and the security channel becomes noise.

Teams commonly respond by adding ignore comments. The comments accumulate. Then a dependency is upgraded, an old ignore rule stops matching, and previously suppressed alerts reappear as a burst. Engineers learn that the scanner is unpredictable and start ignoring it wholesale. That is the real cost: not the alerts themselves, but the loss of trust in the signal.

## Why static ignore rules rot

### Repository-local ignore files

The first instinct is to commit ignore files per repository. Each scanner supports some version of this: a dotfile for the dependency scanner, a `.trivyignore` for image scanning, a configuration file for static analysis. This works until it does not, and the reasons are structural rather than accidental.

**Staleness.** An ignore rule is usually written against a specific version. When the dependency is upgraded, the rule no longer matches the new version, and the alert returns. The rule was correct when written and is wrong now, but nothing in the system notices the transition.

**Drift.** Teams add dependencies without updating ignore files. A transitive dependency bump can introduce a version that no existing rule covers.

**Fragmentation.** With N repositories you get N ignore strategies. Two services with the same dependency graph can have different suppression policies, and no one can say which is correct.

### Centralized ignore lists

The second instinct is a single source of truth: one ignore file in one repository, referenced by every CI job. This fixes fragmentation and creates two new problems.

**Propagation delay.** If the ignore list is consumed at job start, a newly added rule does not take effect until the next job runs. If it is baked into a runner image or a cached artifact, the delay can be hours. During that window, engineers see alerts for rules that were already decided. That erodes trust faster than the alerts themselves.

**Over-broad scope.** A rule written for one service can suppress a real finding in another. This is the dangerous failure mode. A suppression justified by "this service runs an old runtime where the vulnerable code path is unreachable" is not justified for a service on a newer runtime where the path is reachable. Centralized rules without service scoping silently convert a false positive into a false negative, and false negatives are invisible until something else surfaces them.

### Severity-only filtering

The third instinct is the simplest: raise the severity threshold. Report only high and critical, drop medium and low. This reduces volume immediately and introduces a specific class of miss.

Severity is a property of the vulnerability, not of your exposure. A medium-severity prototype pollution issue in a dependency that parses untrusted input may be more exploitable in your system than a high-severity issue in code you never call. Filtering purely on severity discards the context that would tell you which is which. The documented behavior of most scanners is to classify by CVSS or an equivalent score; none of them know your call graph.

## The approach: a triage layer

The shift that works is to stop configuring scanners and start configuring the pipeline. False positives are not primarily a scanner configuration problem. They are a pipeline design problem: the scanner has no access to the context that would let it decide relevance, and the pipeline does not add that context.

A triage layer sits between scan output and notification. It has three stages:

1. **Scan.** Run the scanners as usual and emit machine-readable output.
2. **Triage.** Parse the findings and apply rules derived from live context.
3. **Notify.** Post only the findings that survive triage.

The triage rules draw on three kinds of data, all of which change as the codebase changes:

- **Dependency graph.** What versions are actually resolved in this repository, including transitives.
- **Runtime context.** Which environment the service runs in, and which dependencies are reachable at runtime versus build time.
- **Maintenance state.** When the dependency was last updated, and whether an upgrade is already in flight.

The important property is that none of these are hand-maintained suppression lists. They are derived from artifacts the pipeline already produces.

### Dynamic ignore rules from the dependency graph

Instead of hardcoding suppressions, generate them from the dependency graph and the update PRs that modify it. A dependency automation tool such as Renovate maintains an up-to-date graph and opens upgrade PRs. When an upgrade PR moves a package from one version to another, that PR is a machine-readable statement that the old version is being retired in this repository.

A triage rule derived from that event looks like this:

```yaml
# triage-rules.yml
ignore_rules:
  lodash:
    - version: "4.17.21"
      reason: "Superseded by 4.17.30 in dependency update PR"
      valid_until: "2026-12-31"
      repo: "frontend-service"
```

Two properties matter here. First, the rule has an expiry, so it cannot silently outlive its justification. Second, the rule is scoped to a repository, so it cannot suppress a finding elsewhere. The `reason` field is not decoration; it is what makes the rule auditable six months later when someone asks why an alert stopped firing.

### Environment-aware filtering

A finding is only actionable if the vulnerable code is reachable in an environment that matters. Declaring that context explicitly per service makes the triage decision reproducible:

```yaml
# security-context.yml
service: "user-service"
environments:
  - name: "production"
    allowed_severities: ["critical", "high"]
  - name: "staging"
    allowed_severities: ["high", "medium"]
  - name: "development"
    allowed_severities: ["medium", "low"]
```

During triage, the layer checks which environments the artifact is deployed to and applies the corresponding severity floor. A medium-severity finding in a development-only dependency does not page anyone. The same finding in a production dependency does.

Note the direction of the default. The permissive environments are the ones where alerts are cheap; the production environment keeps the strictest floor. Inverting this — strict floors in development, relaxed in production — is a common configuration mistake and produces exactly the wrong tradeoff.

### Maintenance-window filtering

Dependencies that have not been touched in a long time are usually either stable or abandoned, and the distinction is hard to automate. A maintenance-window filter encodes a conservative policy: for dependencies older than a threshold, only critical findings alert.

```python
# maintenance_filter.py
import datetime
import yaml

class MaintenanceFilter:
    def __init__(self, last_updated_file: str):
        with open(last_updated_file) as f:
            self.last_updated = yaml.safe_load(f)

    def should_alert(self, vuln, repo):
        last_updated = self.last_updated.get(repo, {}).get(
            vuln.package, datetime.datetime.min
        )
        age_days = (datetime.datetime.now() - last_updated).days
        if age_days > 90 and vuln.severity != "critical":
            return False
        return True
```

The 90-day threshold is a policy choice, not a derived constant. Pick it from your own upgrade cadence: if your dependency automation opens upgrade PRs weekly, a dependency untouched for 90 days is genuinely anomalous. If your cadence is quarterly, 90 days is meaningless and the filter will suppress everything.

### Wiring the triage layer into CI

The triage step runs after scanning and before commenting. In GitHub Actions, a workflow triggered on completion of the scanning workflow keeps the two concerns separate:

```yaml
# .github/workflows/security-triage.yml
name: Security Triage
on:
  workflow_run:
    workflows: ["Security Scanning"]
    types: [completed]

jobs:
  triage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run triage
        run: python security/triage.py
        env:
          SCAN_RESULTS: ${{ github.event.workflow_run.artifacts_url }}
          CONTEXT_FILE: security/security-context.yml
          IGNORE_RULES_FILE: security/triage-rules.yml
```

The `workflow_run` trigger is the part worth understanding. It runs in the context of the default branch, which means the triage logic itself is not modifiable by the pull request being triaged. If the triage step ran inside the PR workflow with the PR's own checkout, a contributor could weaken the filter in the same commit that introduces the finding. That is a real threat model for any repository that accepts external contributions.

## Measuring whether it works

Any claim about reduced noise should be backed by numbers you collect yourself. The instrumentation is straightforward.

**What to record, per pull request:** total findings emitted by each scanner, findings after triage, findings dismissed by a human, and findings that were eventually confirmed as real. The last two are the only ones that matter for correctness; the first two measure volume.

**How to collect it.** Have each scanner emit JSON, have the triage layer emit JSON, and append both to a table keyed by PR number and commit SHA. A small script that joins the two on finding ID is sufficient. Do not rely on the PR comment count as a proxy; comments are deduplicated and edited, and the numbers will not reconcile.

**What to compare.** The ratio of confirmed-real to total-alerted is your precision. Track it weekly. A triage change that reduces alert volume while holding precision steady is a win. A change that reduces volume and precision together is suppressing real findings, and the only way to see that is to keep a sample of suppressed findings and review them.

**A worked example of the arithmetic.** Suppose a repository averages 120 findings per pull request across all scanners. A human reviewer needs roughly one minute per finding to triage, so 120 findings is about two hours of review per PR. If triage removes 100 of those and the 20 remaining take one minute each, review drops to about 20 minutes. That is arithmetic from stated assumptions, not a measurement — substitute your own numbers. The point is that the value of triage scales with findings-per-PR times review cost, and if either is small, triage is not worth building.

**On false negatives.** Suppression is only safe if you can detect when it was wrong. Keep a sample of suppressed findings — even a fixed percentage — and route them to a periodic review. The cadence should be tied to your release frequency, not to a calendar quarter. A suppression that has been in place across twenty releases without review is not a policy; it is an unexamined assumption.

## Failure modes to design against

**Suppression outliving its justification.** Every rule needs an expiry or a linked upgrade. Rules without either become permanent by default. The `valid_until` field in the rule schema is the cheapest mitigation.

**Cross-service rule leakage.** Scope every rule to a repository or a service name. A rule that applies everywhere is a rule that will eventually suppress something real somewhere.

**Triage logic editable by the PR.** Run triage from the default branch, not from the PR checkout. Otherwise the filter is part of the attack surface.

**Silent triage failures.** If the triage layer errors, the failure mode should be "post everything," not "post nothing." A triage layer that fails closed converts an outage in the filter into an outage in detection. Make the default permissive and alert on triage errors separately.

**Thresholds treated as constants.** The 90-day maintenance window and the severity floors are policy. They should live in version-controlled config with a comment explaining the reasoning, and they should be revisited when the upgrade cadence changes.

## A decision checklist

Before building a triage layer, answer these:

- What is the current findings-per-PR count, measured, not estimated?
- What fraction of those findings are dismissed by a human without action?
- How long does a human spend per finding?
- Which environments does each service actually deploy to, and is that recorded anywhere machine-readable?
- Does dependency automation already run, and does it produce machine-readable upgrade events?
- Who reviews suppressed findings, and how often?
- What happens when the triage layer itself fails?

If the first three answers are small, do not build this. The engineering cost of a triage layer is real, and it is only justified when alert volume times review cost exceeds the cost of maintaining the filter.

## Applying this incrementally

**Start with one scanner.** Pick the one producing the most findings. Run it unfiltered for a week and collect baseline numbers. Running three scanners from day one makes it impossible to attribute a change in noise to a change in triage.

**Build the simplest useful filter.** Environment filtering, if services have distinct deployment targets, or maintenance-window filtering if the codebase has old dependencies. One rule, measured.

**Add dependency automation before dynamic rules.** Dynamic ignore rules need a machine-readable dependency graph and a stream of upgrade events. Without that, the rules are hand-maintained and you are back where you started.

**Add filters one at a time and measure each.** Each filter should show a measurable reduction in findings-per-PR at stable precision. A filter that reduces volume without a precision measurement is an unverified assumption.

**Document the pipeline.** A short `SECURITY.md` in each repository explaining what is filtered, why, and how to appeal a suppression. The appeal path matters: without one, engineers who disagree with a suppression have no recourse except disabling the scanner.

## Frequently asked questions

### How do I avoid false negatives?

Layer the filters so that critical findings are never suppressed by volume-reduction rules. Keep maintenance-window and severity filters from applying to critical findings. Give every suppression an expiry. Sample suppressed findings and review them on a cadence tied to releases. The combination means a wrong suppression is caught within a bounded number of releases rather than never.

### Can this work without dependency automation?

Yes, but the dynamic rules become manual. You can derive the dependency graph from a lockfile and record upgrade events by hand when a dependency is bumped. That is workable for a small number of repositories and becomes unmanageable as the count grows. Dependency automation is the part that makes the approach scale, not the part that makes it possible.

### Does this apply to container and static analysis scanning?

Yes. The principle is the same: filter on reachability, not on the finding alone. For container scanning, a finding in a layer that is replaced at deploy time is not reachable. For static analysis, a finding in test code that never ships is not reachable. Both require you to record, per service, what actually gets deployed — which is the environment context file described above.

### How do I get buy-in?

Run the pilot on one repository and publish the before-and-after numbers alongside the precision measurement. Volume reduction alone is not a persuasive argument, because a filter that suppresses everything also reduces volume. Pairing volume with a confirmed-real rate shows that the reduction came from removing noise rather than from removing signal.

### What about findings with no available fix?

Treat them as a separate track from triage. Record the finding, assess exploitability in your environment, apply a mitigation at the runtime layer if one exists, and track it to resolution. Do not fold unfixable findings into the suppression rules; they will be forgotten there. The triage layer's job is to route them to a human, not to hide them.

## One thing to do in the next 30 minutes

Pick one repository, run its dependency scanner once against the current default branch, and dump the output to JSON. Count the findings. That count, multiplied by your team's realistic per-finding review time, is the number that tells you whether a triage layer is worth building — and you will have it before you write any configuration.
