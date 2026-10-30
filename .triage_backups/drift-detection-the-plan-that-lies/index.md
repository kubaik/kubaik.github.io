# Drift detection: the plan that lies

Terraform's `plan` is often treated as the source of truth for infrastructure state. The documentation describes it as a way to preview changes, and in a clean environment it does exactly that. But in any estate that has been running for more than a few weeks, `plan` becomes a negotiation between three unreliable narrators: the state file, the provider's read logic, and the actual cloud API. Drift detection is the discipline of finding out which one is lying before an `apply` does something irreversible.

The problem this article addresses: teams assume `terraform plan` will surface drift, but a plan only shows drift for attributes the provider actually reads back and compares. Everything else — tags applied by a policy engine, security group rules added by an incident responder, IAM roles modified by a break-glass script — stays invisible until the next apply silently reverts it, or worse, fails halfway through because a resource was deleted out-of-band. What follows covers how to build drift detection that catches the invisible cases, how to remediate without causing an outage, and how to recognize when remediation is the wrong move.

## How drift detection actually works under the hood

Terraform and OpenTofu share the same core model: a state file maps resource addresses to real infrastructure IDs, and each provider implements a `Read` function that fetches the current attributes of a resource. During `plan`, the core calls `Read` for every resource in state, compares the result to the configuration, and produces a diff. Drift is any difference between the refreshed state and the configuration that was not caused by a change in the configuration itself.

Three properties of that model matter for detection design:

1. **Drift is provider-defined.** A provider only reports drift for attributes it chooses to read. The AWS provider, for example, reads back most resource attributes, but computed fields and fields with `computed: true` in the schema are not compared against configuration. If a tag is added outside Terraform, whether it shows up in `plan` depends on whether that provider version includes tags in its read path and whether the configuration manages them.

2. **Refresh can be skipped or stale.** `terraform plan -refresh=false` uses the cached state without calling the provider. That is faster, but it cannot detect drift. Conversely, `-refresh-only` performs the refresh and shows the drift as a diff without proposing configuration changes — this is the closest thing to a pure drift report that the core offers.

3. **The state file is a cache, not a ledger.** If a resource is deleted outside Terraform, the next refresh removes it from state and the plan proposes to recreate it. That is drift, but it is also a data-loss event if the resource held state (a database, a bucket with objects). Detection and remediation have to be separated for exactly this reason.

OpenTofu inherits this model from the Terraform codebase it forked from. The provider protocol is compatible, so a drift detection strategy built around `plan -refresh-only` and provider read behavior works on both. The operational differences are in the surrounding tooling — state encryption, registry, and the pace of provider releases — not in the drift semantics.

One more mechanical detail worth internalizing: the refresh step mutates the state file's stored attribute values before the diff is computed. That means a refresh-only plan is not a read-only operation with respect to state. If the refresh discovers that a resource no longer exists, the refreshed state records its absence, and whether that absence is persisted depends on how the plan is applied or discarded. Treat the state file as something the detection pipeline writes to, not just reads from, and give the pipeline its own state access controls.

## A detection architecture that separates observe from act

A common failure mode is a scheduled `terraform apply` that is supposed to "fix drift." That conflates detection with remediation and gives the pipeline permission to make changes nobody reviewed. A better architecture has three stages:

- **Observe:** run `plan -refresh-only -detailed-exitcode` on a schedule, capture the output, and alert on non-zero exit codes. This never changes infrastructure.
- **Classify:** parse the plan output and decide whether each drift is benign (a tag added by a cost tool), dangerous (a security group rule opened to 0.0.0.0/0), or ambiguous (a resource deleted out-of-band).
- **Remediate:** only for classified-benign drift, run a targeted `apply` with the exact resource addresses. Everything else goes to a human queue.

`-detailed-exitcode` is the key primitive: it returns 0 for no changes, 1 for an error, and 2 for a non-empty diff. That makes it usable as a CI gate without parsing human-readable output.

```bash
#!/usr/bin/env bash
set -euo pipefail

# Run from a directory with an initialized backend.
set +e
terraform plan -refresh-only -detailed-exitcode -out=drift.tfplan > plan.log 2>&1
status=$?
set -e

case "$status" in
  0) echo "No drift detected" ;;
  1) echo "Plan failed — check credentials and backend"; cat plan.log; exit 1 ;;
  2) echo "Drift detected"; terraform show -json drift.tfplan > drift.json ;;
  *) echo "Unexpected exit code: $status"; exit 1 ;;
esac
```

The JSON output is the important artifact. It contains `resource_changes` with an `actions` array (`["no-op"]`, `["update"]`, `["delete"]`, `["create"]`, `["read"]`). Filtering for `update` where the `before` and `after` differ outside of known-noise attributes is how signal gets separated from the constant churn of computed timestamps and provider-generated IDs.

Note that `-refresh-only` with `-out` still writes a plan file. That plan, if applied, would commit the refreshed values to state. The observe stage should discard it after extracting JSON — never hand it to an apply step. If the pipeline archives plan files for audit, archive them in a location the remediation job cannot read.

## Classifying drift without drowning in noise

A raw `plan -refresh-only` on a mature estate produces a lot of diffs that are not actionable. The AWS provider, for example, may report changes to attributes like `tags_all` when a default tag is applied by an organizational policy, or to fields that are computed from the API and change between reads. Without a classification layer, the alert channel becomes noise and people stop reading it.

A workable classification framework has four buckets:

| Bucket | Signal | Action |
|---|---|---|
| Cosmetic | Tags, descriptions, non-security metadata | Auto-remediate with targeted apply, log to audit |
| Security-relevant | IAM policies, security group rules, public access blocks, encryption settings | Page on-call, never auto-remediate |
| Structural | Resource deleted out-of-band, replaced with different ID | Human review, check for data loss before recreate |
| Provider noise | Computed attributes, timestamps, provider-generated IDs | Suppress via ignore_changes or post-processing filter |

The `ignore_changes` lifecycle argument is the documented mechanism for telling Terraform not to compare specific attributes. It is appropriate for attributes that are intentionally managed outside Terraform — for example, a tag applied by a cloud governance tool. It is not appropriate as a blanket fix for noise, because it also hides real drift on those attributes.

```hcl
resource "aws_s3_bucket" "logs" {
  bucket = "example-logs-bucket"

  lifecycle {
    # Managed by the central tagging policy, not by this module.
    ignore_changes = [tags["CostCenter"]]
  }
}
```

A subtler trap: `ignore_changes` accepts attribute paths, and for nested blocks the path syntax matters. Getting it wrong silently does nothing, which is worse than not having it — the drift keeps appearing and the team assumes the mechanism is broken. The way to verify a suppression actually took effect is to run a refresh-only plan after adding it and confirm the specific attribute no longer appears in `resource_changes` for that address. If the diff persists, the path is wrong.

Classification itself can live in a few places. A small script that reads `drift.json` and matches on resource type plus attribute path is the lowest-dependency option. Policy-as-code engines evaluate the same JSON against declarative rules, which is easier to review and version. Whatever the mechanism, the rules should be stored in the same repository as the infrastructure code so that a change to what counts as "benign" goes through review.

## Remediation patterns and when each is safe

There are three remediation patterns, and they have very different risk profiles.

**Targeted apply.** `terraform apply -target=aws_s3_bucket.logs` applies only the resources matching the address. This is the safest automated remediation because the blast radius is explicit. The caveat, documented by HashiCorp, is that `-target` is intended for exceptional situations, not routine use — it can leave the rest of the state inconsistent with configuration. For drift remediation of a single cosmetic resource, that trade-off is usually acceptable, but it should be logged and reviewed.

**Full apply from a clean plan.** Running `terraform apply` on a saved plan file that was generated after a refresh. This is the normal path, but it remediates *all* drift at once, including the security-relevant changes that were supposed to wait for a human. It is only safe when the classification step has confirmed the plan contains nothing but benign drift.

**Config-as-truth reconciliation.** Instead of applying Terraform to match the drifted reality, update the configuration to match reality (if the out-of-band change was legitimate) and then apply. This is often the right answer for emergency changes that were made during an incident and should be kept. It requires a human, which is the point.

The counterintuitive part is that automated remediation is most valuable for the *least* important drift. Cosmetic tag drift is high-volume, low-risk, and tedious to fix manually — perfect for automation. Security drift is low-volume, high-risk, and exactly the kind of change that should interrupt someone. Teams that automate both end up with an auto-apply pipeline that can silently revert a security fix applied during an incident, which is a much worse outcome than a noisy alert.

A worked example makes the trade-off concrete. Suppose a scan reports three drifted resources: a bucket missing a `CostCenter` tag, a security group with an added ingress rule on port 22 from a troubleshooting session, and a queue whose retention setting was changed by hand. Under the framework above, the tag is cosmetic and can be targeted-applied automatically. The security group rule is security-relevant and pages a human, who decides whether the rule was a legitimate temporary exception (update config) or a mistake (revert). The queue change is ambiguous: it is not security-relevant, but it may have been made deliberately to handle a backlog, so auto-reverting it could cause message loss. The classification rule that separates the queue from the tag is not "is it metadata" but "would reverting this destroy data or state." Encoding that question as a resource-type allowlist for auto-remediation is a small amount of work that prevents most of the painful incidents.

## Measuring drift: what to track and what to ignore

The metrics that matter for a drift program are not the number of drifted resources — that number is noisy and depends on provider version. The useful metrics are:

- **Time-to-detect:** the interval between an out-of-band change and the first alert. This is bounded by scan frequency. A daily scan means up to 24 hours of undetected drift.
- **Time-to-remediate by bucket:** cosmetic drift should close within one scan cycle; security drift should close within the incident SLA. If security drift is closing faster than cosmetic drift, the automation is probably misconfigured.
- **False positive rate:** the proportion of alerts that a human marks as provider noise. If a large share of alerts are dismissed as noise, the classification layer needs work before the program will be trusted.
- **Recurrence:** the same resource address drifting repeatedly is a signal that something outside Terraform owns it and the configuration should be changed to reflect that ownership, not repeatedly reverted.

These are relative metrics; there is no published benchmark for what a "good" drift rate is, because it depends entirely on how much of the estate is under management and how many controllers touch it. Track the trend, not the absolute number.

To measure them, instrument the pipeline at three points. Record a timestamp when the scan starts and when the alert fires — the difference is time-to-detect, and it will be dominated by scan interval, so compare it against the interval rather than treating it as an independent number. Record the classification bucket and the timestamp of the closing event (auto-remediation applied, or human resolved) to compute time-to-remediate per bucket. For false positives, add a one-field disposition to the alert — "noise," "real," "accepted" — and count the ratio over a rolling window. All three can be derived from the JSON artifact plus a small sidecar record; no external telemetry system is required.

## The failure modes nobody warns you about

**The refresh that deletes state.** If a resource is deleted out-of-band and the provider's read returns not-found, the refresh removes it from state. A subsequent apply recreates it — possibly with a new ID, new ARN, and broken references. For stateful resources (databases, buckets, KMS keys), this is a data-loss path. Detection pipelines should flag `delete` actions in a refresh-only plan as critical, not routine.

**The provider upgrade that changes read behavior.** A provider version bump can start reading attributes it previously ignored, producing a wave of "drift" that is actually a change in the detection tool. Pin provider versions and treat upgrades as a change that requires a baseline scan before and after. The comparison is straightforward: run the scan on the pinned version, upgrade, run it again, and diff the two `drift.json` artifacts. Any resource address that appears only in the second run is a candidate for a read-behavior change rather than real drift.

**The state lock that blocks detection.** If a long-running apply holds the state lock, scheduled drift scans fail with a lock error. If the pipeline treats any non-zero exit as drift, you get false alerts. Distinguish exit code 1 (error) from exit code 2 (diff) — the `-detailed-exitcode` script above does this, but many homegrown wrappers do not.

**The module that manages the same resource twice.** Two workspaces or two modules managing overlapping resources will each see the other's changes as drift and fight. This is not a detection problem; it is an ownership problem that detection will surface loudly and repeatedly until someone fixes the boundary.

**The compliance trap.** Where personal data is in scope, drift on resources that store or process it is not cosmetic. A security group rule that opens a database to the internet, or an S3 bucket policy that removes a public access block, is a potential personal data breach. Detection pipelines for these resources should feed into the incident response process, not the auto-remediation queue. The audit trail of who changed what and when is itself a requirement under accountability principles — which means the drift log needs to be retained and tamper-evident, not just printed to CI output.

## Tools and approaches worth considering

There is no single tool that solves drift detection end-to-end, but several categories are well-established:

- **Native plan-based detection:** `terraform plan -refresh-only -detailed-exitcode` plus JSON parsing. This is the lowest-dependency option and works identically on OpenTofu.
- **Policy-as-code engines:** engines such as OPA/Conftest can evaluate the plan JSON against rules — for example, "no security group may allow 0.0.0.0/0 on port 22" — which turns drift classification into a policy problem rather than a custom script.
- **Cloud-native config recorders:** AWS Config, Azure Policy, and GCP's asset inventory can detect out-of-band changes independently of Terraform state. They are useful as a second opinion because they do not depend on the state file being accurate.
- **Commercial drift platforms:** several vendors offer continuous drift detection with dashboards. They are worth evaluating if the estate is large enough that building the classification layer in-house is a full-time job, but the underlying mechanism is still refresh-and-compare, so the same failure modes apply.

A minimal, dependency-free approach that many teams find sufficient: a scheduled CI job that runs the script above, uploads `drift.json` as an artifact, and posts a summary to the team channel with the resource addresses grouped by bucket. No vendor required, and the audit trail is the CI job history.

## When this approach is the wrong choice

Drift detection is not free, and there are estates where the cost outweighs the benefit.

If the infrastructure is entirely ephemeral — created and destroyed by the same pipeline that owns it, with no manual changes permitted — drift is a bug in the pipeline, not an operational condition. Fix the pipeline instead of building a detector.

If the estate is small enough that a single engineer can hold the whole thing in their head and changes are infrequent, a weekly manual `plan` review may be cheaper than automation. The break-even point is roughly when the number of resources or the number of people touching the estate exceeds what one person can track, but that is a judgment call, not a number that can be given in advance.

And if the organization has no process for acting on drift alerts, detection will produce a backlog that nobody owns. The tooling is the easy part; the remediation queue and the on-call rotation are the hard part. Build the process before the pipeline.

## Common production pitfalls and what they cost

- **Auto-apply on any drift.** Cost: silent reversion of incident fixes, occasionally an outage. Mitigation: classify before remediating.
- **Scanning too infrequently.** Cost: a security group opened on Friday is not detected until Monday. Mitigation: scan at least daily for security-relevant resources, more often if the threat model warrants it.
- **Ignoring provider noise.** Cost: alert fatigue, then ignored alerts, then a real drift that nobody sees. Mitigation: suppress known-noise attributes explicitly and review the suppression list quarterly.
- **No audit retention.** Cost: inability to answer "who changed this and when" during a compliance review. Mitigation: retain drift plan artifacts for the period the regulator requires.
- **Treating OpenTofu and Terraform as interchangeable in CI.** Cost: subtle differences in provider resolution and state encryption cause failures that look like drift. Mitigation: run detection with the same binary and version that applies changes.

## What to do next

The single most useful action in the next 30 minutes: open a terminal in a directory with an initialized Terraform or OpenTofu backend, and run `terraform plan -refresh-only -detailed-exitcode`. Note the exit code. If it is 2, save the JSON with `terraform show -json` and read through `resource_changes` — that gives a real drift report for the workspace and a baseline for everything that follows.
