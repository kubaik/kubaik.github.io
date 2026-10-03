# Runbooks vs AI Runbooks: A Decision Framework

## Why this comparison matters

Runbooks rot. A procedure that was correct when written can silently become wrong after a config rename, a chart upgrade, or a service split. The failure mode is consistent and expensive: during an incident, an on-call engineer follows a documented step that no longer matches the system, loses minutes on a dead end, and has to reconstruct the correct action under pressure.

AI-assisted runbook tooling claims to address this by generating procedures on demand from live infrastructure state, detecting anomalies against learned baselines instead of static thresholds, and updating itself as the environment changes. The claim is plausible. It is also the kind of claim that hides new failure modes: fabricated steps, non-idempotent remediation, inference latency on the paging path, and audit gaps.

This article is a trade-off sheet. It covers how each approach works, where each breaks, how to measure the difference honestly, and how to decide for a specific team.

## Option A — static runbooks

Static runbooks are versioned documents, usually Markdown, one per alert or per service, stored in a repository and rendered somewhere the on-call engineer can find them fast.

A common stack:

- Git for storage and review
- A static site generator for rendering
- The incident or alerting tool for embedding a link
- A linter in CI that enforces required fields

A typical runbook:

```markdown
## Alert: PostgreSQL connection exhaustion
Severity: P1
Owner: @platform-team

### Symptoms
- `pg_stat_database.connections > 95%` for 3 minutes
- connection error rate above the alert threshold

### Triage
```bash
kubectl exec -n postgres -it pg-primary-0 -- psql -c "select count(*) from pg_stat_activity;"
```

### Remediation
1. Raise the connection limit in `values.yaml`:
   ```yaml
   postgres:
     max_connections: 500
   ```
2. Apply:
   ```bash
   helm upgrade pg bitnami/postgresql -f ./values.yaml
   ```
3. Verify:
   ```bash
   kubectl rollout status deployment pg-primary
   ```

### Post-incident
- Lower the dashboard threshold to catch the condition earlier
```

The runbook lives next to the code, gets reviewed in pull requests, and can be linted in CI for required fields such as severity, owner, at least one code block, and a post-incident action.

### Where static runbooks work well

- Infrastructure changes slowly, on the order of a few times a month.
- The team has a review culture that treats documentation with the same rigor as code.
- Human-readable artifacts are required for compliance (SOC 2, ISO 27001).
- Ownership of a runbook is a gate for service ownership.

### Where static runbooks break

- New services ship without runbooks and get one only after their first incident.
- Engineers copy steps without understanding them, so a subtly wrong step propagates.
- Multi-alert incidents require merging several runbooks by hand. Each runbook assumes a different system state, and reconciling them under time pressure is where minutes disappear.
- Maintenance decays. Without a forcing function, stale runbooks accumulate faster than they are retired.

The maintenance tax is the real cost. It is not the writing; it is the periodic verification that the steps still match production. A useful forcing function is a scheduled job that flags files not touched in a set number of days and opens a review task, plus a rule that any change to a service's deploy or config requires a corresponding runbook update in the same pull request.

## Option B — AI-assisted runbooks

AI-assisted runbook systems replace or augment static documents with a pipeline of components. The categories are stable even though specific vendors change:

1. **Anomaly detection.** A model ingests metrics from your monitoring system and flags deviations from a learned baseline rather than crossing a fixed threshold. This is the part that can catch seasonal or workload-dependent behavior that a static number misses.
2. **Runbook generation.** When an anomaly crosses a severity boundary, the system queries your inventory or configuration source and generates a step-by-step procedure in whatever form your stack consumes: shell, Terraform, Ansible, or Kubernetes manifests.
3. **Verification loop.** After each step, the system runs a lightweight check and either proceeds or rolls back. This is what prevents a generated script from restarting a pod forever.
4. **Post-incident capture.** The generated procedure is stored as an artifact and linked to the incident record. Over time, the system can be retrained on the incident corpus.

### Where AI runbooks work well

- Infrastructure changes weekly or faster, so static documents are stale by design.
- Alert volume is high enough that triage consistency across time zones matters.
- The team accepts adaptive procedures in exchange for losing deterministic scripts.
- Someone owns the model, the guardrails, and the retraining loop.

### Where AI runbooks break

- **Fabricated steps.** A generated procedure can include a plausible but wrong command. The classic example is a destructive command that matches the shape of the alert but not its cause. Any generated step that mutates state needs a guardrail and an undo.
- **Latency on the paging path.** An inference call adds time before a human sees guidance. If your alerting SLA is tight, that delay is a real cost.
- **Cost.** Hosted inference is metered. Self-hosting shifts cost from a bill to operational burden: model serving, drift monitoring, and retraining.
- **Auditability.** Compliance reviewers want evidence of what changed during an incident. Generated diffs, confidence signals, approvals, and rollback commands must all be exported and retained. If that export is not automated, it becomes a manual scramble.

## How to measure the difference honestly

Published comparisons of runbook strategies tend to quote MTTR improvements without stating the test conditions. Do not trust those numbers; produce your own. The measurements below are the ones that actually drive the decision, and each is cheap to instrument.

**What to instrument**

- **MTTD:** timestamp of the first alert or detection event minus the timestamp the condition actually began, taken from the metric series, not from human memory.
- **MTTR:** timestamp the incident is declared resolved minus the timestamp the condition began. Record the resolver and the steps taken in a structured field, not free text.
- **Recurrence:** count incidents of the same class within a fixed window, say 30 days, per service.
- **False positives:** pages that required no action, counted per week.
- **Cost per incident:** engineer minutes multiplied by a loaded hourly rate, plus any metered inference attributed to that incident.

**How to run the comparison**

1. Pick one service and one incident class, for example Redis memory eviction on a single cluster.
2. Run the existing static runbook for a set number of incidents and record the metrics above.
3. Enable the AI system in shadow mode: it generates procedures and logs them, but no step executes automatically and the on-call engineer still follows the static runbook.
4. Diff each generated procedure against the static one and record every discrepancy, especially any step that would have mutated state incorrectly.
5. Only after the shadow period, allow the generated procedure to be used with human approval on every destructive step, and re-measure.

The shadow-mode diff is the most valuable output of the whole exercise. It tells you whether the model understands your environment before it is allowed to touch it.

## A worked cost comparison

Numbers here are illustrative, with the arithmetic shown so you can substitute your own.

Assume a team with 40 incidents per year. Assume the loaded cost of an on-call engineer is $120 per hour, and that the static-runbook path averages 60 minutes of engineer time per incident while an AI-assisted path averages 25 minutes. That is a 35-minute saving per incident.

- Static: 40 × 1.0 h × $120 = $4,800 per year in incident time.
- AI-assisted: 40 × (25/60) h × $120 = $2,000 per year in incident time.
- Saving: $2,800 per year.

Now subtract the AI costs. If hosted inference and licensing run $1,800 per month, that is $21,600 per year, and the AI path is far more expensive. If a self-hosted model runs on one instance at $500 per month, that is $6,000 per year, still more than the saving. Break-even on incident time alone occurs when the annual saving exceeds the annual AI cost:

- Required incidents per year = annual AI cost ÷ saving per incident.
- At $6,000 per year and $70 saved per incident (35 minutes at $120/hour), break-even is about 86 incidents per year.

The point of the arithmetic is not the number. It is that incident-time savings rarely justify the tooling cost on their own at low incident volume. The justification has to come from something else: reduced recurrence, less stale documentation, or consistency across a distributed on-call rotation. If those do not apply, the math does not work.

## The guardrail problem

The single most important design decision in an AI-assisted runbook system is what the model is allowed to do without a human in the loop.

A workable three-layer model:

1. **Deterministic policy check before execution.** A rule engine evaluates every generated command against a blocklist and an allowlist. Blocklist patterns include recursive deletion, namespace deletion, and infrastructure destroy operations. The policy engine, not the model, makes the final call.
2. **Human approval for state-mutating steps.** Read-only triage commands can run automatically. Anything that changes configuration, deletes a resource, or restarts a workload requires an explicit approval action.
3. **Mandatory rollback.** Every generated procedure includes a tested undo command. The rollback path is exercised in staging on a schedule, not only during incidents.

A guardrail that is only a prompt instruction is not a guardrail. The model can be asked to be careful; it cannot be relied on to be careful. Enforcement belongs in code that the model cannot modify.

## Decision checklist

Work through these in order. The first "no" usually settles it.

- **Alert volume.** How many pages per day? Under ten, static runbooks plus a staleness bot are almost always the better investment.
- **Rate of change.** Does infrastructure change more than twice a month? If not, static documents can stay accurate with modest effort.
- **Incident volume.** How many incidents per year? Below roughly 80, the incident-time saving rarely covers tooling cost on its own.
- **Ownership.** Is there a team of at least four engineers who can own model serving, guardrails, and retraining? If not, a hosted option only makes sense if the vendor owns the operational burden.
- **Compliance.** What does your auditor require as evidence of changes made during an incident? If the answer is "a human-readable document," confirm that generated artifacts can be exported and retained before committing.
- **Trust budget.** Has the system been run in shadow mode long enough to diff its output against reality? One destructive false step can cost weeks of engineer trust.

## Common mistakes

- **Adopting AI to avoid writing runbooks.** The system still needs a source of truth about your environment. If your inventory and configuration data are unreliable, generated procedures will be too.
- **Skipping shadow mode.** Letting generated procedures execute before diffing them against known-good ones converts a cheap experiment into an incident.
- **Ignoring the paging path.** Inference latency sits between the alert and the human. Measure it under load, not on an idle system.
- **Treating compliance as an afterthought.** Exporting generated diffs, approvals, and rollbacks needs to be automated on day one, not reconstructed during an audit.
- **Assuming cost scales with compute.** In practice, licensing and the operational overhead of self-hosting dominate. Model the full line items before committing.

## FAQ

**How do I know if alert volume justifies AI runbooks?**

Measure pages per week for two weeks using your alerting tool's own reporting. If the number is consistently above roughly 30 per week and triage quality varies by responder, the consistency argument for AI-assisted triage gets stronger. Below ten pages per day, the tooling cost usually outweighs the benefit.

**Which model should be used for self-hosted runbook generation?**

Model choice changes quickly, so evaluate against your own incident corpus rather than a leaderboard. The relevant axes are latency on the paging path, quality on your stack's command vocabulary, and serving cost per thousand prompts. Run the shadow-mode diff described above with two or three candidates and compare discrepancy rates. Fine-tuning on incident logs helps, but only if the logs are clean.

**How do I prevent a generated procedure from running destructive commands?**

Do not rely on prompt instructions. Enforce a policy check in code before execution, require human approval for any state-mutating step, and require a tested rollback command in every generated procedure. Test the blocklist regularly with known-bad commands to confirm it still fires.

**What is the hidden cost teams miss?**

Audit and export. Every generated procedure used during an incident becomes evidence. It must be captured, linked to the incident record, and retained in a form the auditor accepts. If that pipeline is not automated, someone rebuilds it manually under deadline.

## Next step

Open your alerting tool's reporting view and count the pages your team received in the last seven days. Write the number down. If it is above 30, pick one service and one incident class and start a shadow-mode evaluation this week, recording MTTD, MTTR, and every discrepancy between the generated procedure and your static runbook. If it is below 30, add a scheduled job that flags runbook files not updated in 90 days and opens a review task for the owner. Either way, you will have a measured baseline instead of an assumption.
