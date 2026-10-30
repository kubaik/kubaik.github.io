# Self-healing agents vs rule-based deployment rollback

## The comparison, stated honestly

Deployment pipelines recover from bad releases in one of two ways. Either a human writes explicit rules — "if error rate exceeds 1% for 30 seconds, roll back" — or a controller observes metrics and decides what to do, including actions nobody wrote down in advance.

The first approach is deterministic and auditable. The second can react faster and handle situations the rule author never anticipated, but it also can act on reasoning that exists only inside the controller's memory. This article compares the two as engineering options: what each is good at, how each fails, what to measure before choosing, and where the boundary between them actually sits.

Nothing here depends on a specific vendor product. The rule-based side is represented by the pattern most teams already run: a GitOps controller plus a progressive-delivery analysis step. The agent side is represented by a Kubernetes controller that reconciles metrics and writes back to the cluster. Both are described by their behavior, not by a product name, because the behavior is what determines the trade-off.

## Option A: rule-based rollback

A rule-based pipeline is a set of conditionals. You write them in YAML or HCL, they live in Git, and the system executes them deterministically. A typical shape:

1. A CI workflow triggers on push to the main branch.
2. A container image is built and pushed to a registry.
3. A GitOps controller syncs the new image to a staging cluster and runs a canary analysis.
4. If the analysis detects an SLO breach, the controller rolls back automatically.
5. Alert rules page a human for anything the analysis did not cover.

The canary analysis step is where the interesting logic lives. A progressive-delivery controller queries a metrics backend at a fixed interval and compares each result against a threshold range. A representative analysis block looks like this:

```yaml
analysis:
  metrics:
    - name: api-error-rate
      thresholdRange:
        min: 0
        max: 1
      interval: 30s
    - name: api-p99-latency
      thresholdRange:
        min: 0
        max: 200
      interval: 30s
  webhooks:
    - name: load-test-gate
      url: http://load-test-service.default.svc:8080/run
      timeout: 5m
```

The `load-test-gate` webhook runs a load generator in the same namespace and exits non-zero if any metric breaches its threshold. This is the key structural point: the gate is a separate process with a binary exit code, so its verdict is as auditable as the threshold it enforces.

**Where this shines.** Every decision is in Git. `git blame` on the rollback policy tells you who changed the threshold and when. The same pipeline deploys to dev, staging, and production without modification. The resource cost is negligible — a controller and a metrics backend you are probably already running.

**Where it fails.** The failure mode is not slowness, it is coverage. A rule-based pipeline promotes a bad image whenever the badness falls outside the metrics you thought to encode. A memory leak that only manifests under a traffic pattern you do not test, a DNS resolution spike every few minutes, a slow query that degrades p99 without breaching the error-rate threshold — none of these trip a rule that was never written.

The second failure mode is threshold drift. A p99 latency ceiling tuned in January is wrong by June if traffic composition changes. The rule does not know this. It either fires too late (threshold too loose) or rolls back healthy builds (threshold too tight). Both are silent until someone investigates.

## Option B: agent-driven rollback

An agent-driven pipeline replaces the fixed decision with a controller that watches the same metrics and chooses among several actions. The controller runs inside the cluster and reconciles on an interval. A typical decision loop:

1. A new image is pushed and synced to the cluster.
2. The controller notices the new revision and begins collecting metrics.
3. After a warm-up window it computes a rolling statistic — a z-score, a ratio, a distance from a learned baseline.
4. If the statistic crosses a boundary, it takes one of several actions: roll back, patch resource requests, or temporarily scale the workload.
5. It records the decision in a custom resource and emits a counter metric so the branch taken can be counted.

A simplified reconciliation loop in Go:

```go
func (r *AutoHealReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
    dep := &appsv1.Deployment{}
    if err := r.Get(ctx, req.NamespacedName, dep); err != nil {
        return ctrl.Result{}, client.IgnoreNotFound(err)
    }

    latency, err := r.promClient.Query(ctx, "rate(http_request_duration_seconds_sum[5m])", time.Now())
    if err != nil {
        return ctrl.Result{}, err
    }

    if latency > thresholdHigh {
        r.recordHealAction("rollback", dep)
        return ctrl.Result{}, r.rollbackDeployment(ctx, dep)
    } else if memoryPressure > thresholdMemory {
        r.recordHealAction("patch_memory", dep)
        return ctrl.Result{}, r.patchDeployment(ctx, dep, map[string]interface{}{
            "spec": map[string]interface{}{
                "template": map[string]interface{}{
                    "spec": map[string]interface{}{
                        "containers": []map[string]interface{}{
                            {
                                "name": dep.Spec.Template.Spec.Containers[0].Name,
                                "resources": map[string]interface{}{
                                    "limits": map[string]interface{}{
                                        "memory": "2Gi",
                                    },
                                },
                            },
                        },
                    },
                },
            },
        })
    }
    return ctrl.Result{RequeueAfter: 10 * time.Second}, nil
}
```

Note what is missing: the thresholds. In a full implementation they are loaded from a ConfigMap that the controller updates itself, typically by running a clustering job over the last N days of metrics. This is the central design decision. The brittleness does not disappear; it moves from the pipeline YAML into the threshold-generation job. A clustering job that produces a p99 latency ceiling 20 ms too low for one day's traffic will roll back a perfectly healthy build, and the rollback log may not name the metric that triggered it.

**Where this shines.** Adaptability is real. During a noisy-neighbor incident, a controller that notices memory saturation causing GC pauses can raise a memory limit and avoid a restart — an action no one wrote a rule for. Agents also surface latent problems: an endpoint leaking memory only under high-percentile load, or a periodic DNS latency spike from a misconfigured cache, are exactly the kind of slow-burn issues that threshold rules miss because nobody knew to look.

**Where it fails.** Opacity, and its downstream cost. When the controller patches instead of rolling back, the reason lives in a decision tree that grew organically. Reproducing a patch applied during a database failover can take days, especially if the trigger was a derived metric that exists only in the controller's logs and never in the metrics backend.

## A worked example: the derived-metric trap

Consider a controller that decides pod health by computing restarts divided by uptime and flagging any value above 0.1 as unhealthy. Walk through the arithmetic. A pod that has restarted twice in 18 hours has an uptime of 18 hours, so the ratio is 2/18 ≈ 0.111 — above the 0.1 boundary. The controller blocks the rollout.

The metric is not wrong in isolation. It is wrong because it was never defined, never reviewed, and never bounded. A pod that has run for 18 hours with two restarts is not obviously unhealthy; a pod that restarted twice in ten minutes is. The ratio conflates the two because uptime is in the denominator and the boundary was chosen without reference to expected restart frequency.

The failure is not that an agent invented a metric. The failure is that the metric was not registered, versioned, or alerted on. Three mitigations follow directly:

- **Register derived metrics in the same system as raw ones.** If the controller computes a ratio, export it to the metrics backend with a name and a help string. Then it appears in dashboards and can be alerted on like anything else.
- **Bound every derived metric.** A ratio of restarts to uptime should have a documented expected range and a maximum plausible value. A value that exceeds the maximum should escalate to a human rather than trigger an action.
- **Make the decision reason a first-class field.** The custom resource the controller writes should name the metric, the query, the observed value, and the threshold. If any of those is missing, the decision is not auditable.

## What to measure before choosing

Benchmarks comparing the two approaches are only meaningful if they are measured on your system. What follows is what to instrument, not a table of results.

**Pipeline latency.** Compare the time from image push to a completed canary decision. Instrument the CI job duration and the analysis step duration separately; the agent's advantage, if any, is in the analysis step, not the build. Compare medians and p95, not means — rollback latency is tail-dominated.

**Mean time to recovery.** Define the start point precisely. "SLO breach detected" is not the same as "first bad metric observed," and the two produce very different numbers. Instrument the timestamp of the first metric that would have breached the rule-based threshold, the timestamp of the rollback action, and the timestamp the service returned to baseline. The difference between the first and second is the decision latency; between the second and third is the recovery latency.

**False-positive rate.** Count rollbacks of builds that were, on later inspection, healthy. This requires a definition of "healthy" that a human applies after the fact. Do not let the controller grade its own homework by using its own thresholds.

**Escalation rate.** Count how often the automated path hands off to a human. A controller that handles more incidents but escalates the hard ones is not obviously better than a rule that pages a human for everything — the human's context switch is the cost.

**Audit time.** Measure how long it takes an engineer unfamiliar with the system to explain why a specific rollback happened. This is the metric that most consistently favors rules, and it is the one teams most often forget to measure.

## The decision checklist

Work through these in order. The first "no" is usually decisive.

1. **Can you afford a non-deterministic rollback?** If a compliance regime requires that every automated action be traceable to a written policy, an agent that generates its own thresholds is out of scope. This is a hard constraint, not a preference.
2. **Is your outage cost high enough that speed matters?** If a bad release costs minutes of degraded service and no revenue, the agent's latency advantage is worth little. If it costs money per minute, it is worth a lot.
3. **Is your traffic stable enough to tune thresholds once?** Stable traffic means rule-based thresholds stay correct for long periods, which removes the agent's main advantage.
4. **Do you have the instrumentation to audit a decision?** If you cannot answer "which metric, which query, which threshold, which value" for every automated action, do not deploy an agent. You will not be able to debug it.
5. **Can you run both?** A parallel pilot on a non-critical service, with identical metrics and both paths logging decisions, gives you a real comparison in weeks. This is almost always cheaper than an argument.

A useful heuristic: if your current mean time to recovery is above roughly 15 minutes and dominated by human acknowledgement time, an agent will likely halve it. If it is already under about 5 minutes, the speed gain rarely justifies the audit burden.

## Failure modes to design against

**The agent that blocks its own rollout.** A controller that decides a new revision is unhealthy based on a metric it computed at runtime will hold the pipeline. Mitigation: a timeout on any hold. If the controller has not reached a decision within a bounded window, it should escalate rather than continue to hold.

**The threshold job that drifts.** If thresholds are regenerated from recent data, a traffic shift changes them. A shift toward lower latency makes the ceiling tighter and can roll back healthy builds. Mitigation: clamp regenerated thresholds to a documented range, and require human review when the new value moves more than a set fraction from the previous one.

**The crash that loses state.** A controller that keeps its decision tree in memory loses it on eviction. If the tree determines future actions, the replacement starts from a different baseline. Mitigation: persist the state to a versioned object and reload it on startup; make the reconciliation loop idempotent so a restart delays decisions rather than duplicating them.

**The annotation escape hatch.** Any controller that can act on a workload needs a per-workload opt-out, expressed as an annotation:

```yaml
metadata:
  annotations:
    autoheal.ai/disable: "true"
```

When present, the controller should restrict itself to non-destructive actions. This matters for canary services that intentionally run with elevated error rates and for synthetic load-test pods that should not be interfered with.

## Auditing an agent's decision

If you run an agent, the audit trail is the product. Every decision should be written to a queryable object with the reason, the query, and the observed value. Listing them:

```sh
kubectl get healactions -A -o wide
```

When the reason names a derived metric, the raw values may only exist in the controller's logs:

```sh
kubectl logs -l app.kubernetes.io/name=autoheal -n autoheal --tail=1000
```

If that second command is routinely necessary, the instrumentation is insufficient. The target state is that the first command answers the question without the second.

## Recommendation

Default to rule-based rollback. It is deterministic, cheap, and auditable, and it covers the majority of failure modes that teams actually hit. Add agent-driven actions only where you can name the specific gap rules leave open — typically slow-burn resource degradation that no threshold captures — and only with the instrumentation to audit every decision.

If you do adopt an agent, do it in parallel. Run it alongside the existing rules on a non-critical service, with both paths logging decisions to the same place. Compare decision latency, false positives, escalation rate, and audit time over a fixed window. The comparison is the deliverable; the agent is just one of the two things being compared.

## Do this in the next 30 minutes

Pick one service and add a decision-reason field to whatever currently triggers its rollback — even if that is a human. Record the metric name, the query, the observed value, and the threshold in the same place the rollback is logged. Then ask one engineer who did not write the policy to read the last five rollback records and explain each one. If they cannot, you have found the gap that either approach has to close before it is worth automating further.
===END===
