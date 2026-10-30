# Self-healing pipelines: Argo vs Crossplane in 2026

## What "self-healing" actually means in each tool

Self-healing is a marketing word covering at least three different mechanisms. Keeping them separate prevents most bad comparisons.

**Continuous reconciliation.** A controller reads a desired state, reads the observed state, and issues API calls to close the gap. This is the Kubernetes controller pattern, and both tools use it. The difference is *what* they reconcile: Argo CD reconciles Kubernetes manifests into a cluster; Crossplane reconciles cloud provider APIs through Kubernetes custom resources.

**Health-driven rollback.** A controller watches a workload's health signal and, when the signal degrades, reverts to a previously known-good revision. This is a Git operation in Argo CD's case: the desired state is a commit, and reverting means pointing the Application back at an earlier revision.

**Drift correction.** Something outside the pipeline changes a resource — a console edit, another automation, a provider-side mutation. The controller notices the live state no longer matches the desired state and re-applies the desired state.

Argo CD is strongest at the first and third mechanisms applied to Kubernetes objects, plus health-driven sync status. Crossplane is strongest at the first and third mechanisms applied to cloud resources. Neither predicts failures. Neither repairs a resource whose provider API refuses the change. Any claim that one is "proactive" and the other "reactive" is really a statement about *polling interval and health signal quality*, not about a fundamentally different control model.

## Argo CD: reconciliation over Kubernetes state

Argo CD stores desired state in Git and runs a controller inside the cluster. The controller compares the rendered manifests against live objects and reports a sync status (`Synced` / `OutOfSync`) and a health status (`Healthy` / `Progressing` / `Degraded` / `Missing` / `Unknown`).

Two settings matter for self-healing:

- `syncPolicy.automated.selfHeal: true` makes the controller re-apply desired state when live state drifts.
- `syncPolicy.automated.prune: true` makes the controller delete resources that are no longer in Git.

Health status is not computed by Argo CD from first principles. It comes from health checks that Argo CD ships for built-in Kubernetes kinds, plus Lua health checks you can supply for custom resources. If a CRD has no health check, Argo CD reports `Unknown` and will not treat the resource as degraded. This is the single most common reason an Argo CD "self-healing" setup silently does nothing: the resource that fails is not one Argo CD knows how to evaluate.

A minimal ApplicationSet that syncs one app to every registered cluster:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: ApplicationSet
metadata:
  name: multi-cluster-apps
spec:
  generators:
  - clusters:
      selector:
        matchLabels:
          argocd.argoproj.io/secret-type: cluster
  template:
    metadata:
      name: '{{.name}}-{{.application}}'
    spec:
      project: default
      source:
        repoURL: https://github.com/example-org/gitops.git
        targetRevision: HEAD
        path: apps/{{.application}}
        helm:
          releaseName: {{.application}}
      destination:
        server: '{{.server}}'
        namespace: {{.application}}
      syncPolicy:
        automated:
          prune: true
          selfHeal: true
          allowEmpty: false
        retry:
          limit: 5
          backoff:
            duration: 5s
            factor: 2
            maxDuration: 3m
```

Notes on this manifest, because small mistakes here produce confusing behavior:

- `allowEmpty: false` prevents an empty render from pruning everything. Keep it.
- The retry block governs *sync* retries, not health evaluation. A resource stuck in `Progressing` will not be retried by this block; it will simply stay `Progressing`.
- `selfHeal` corrects drift in objects Argo CD manages. It does not correct drift in objects created by an operator that Argo CD does not track.

**Failure mode to design around:** a health check that returns healthy for a broken workload. A liveness probe that only checks that the process is listening will report healthy while the process returns errors for every request. Argo CD will see `Healthy`, will not roll back, and the outage continues. The fix is at the application layer — health endpoints that exercise real dependencies — not in Argo CD's configuration.

**Second failure mode:** a CRD with no health check. The resource is `Unknown` forever. Argo CD never considers the app degraded. Teams typically discover this only during an incident. Audit which kinds in your manifests have health checks before you rely on rollback.

## Crossplane: reconciliation over cloud APIs

Crossplane installs providers that expose cloud resources as Kubernetes custom resources. A `VPC`, a `Cluster`, an `RDSInstance` — each is a Kubernetes object with a `spec.forProvider` block that mirrors the provider's API. A composite resource (XR) groups several of these, and a Composition describes how to build them from the XR's fields.

The control loop is the same pattern as any Kubernetes controller: read desired, read observed, act. Crossplane's provider controllers watch their resources and re-issue create/update/delete calls when the observed state diverges from the spec.

A Composition that builds a VPC and an EKS cluster from a composite resource:

```yaml
apiVersion: apiextensions.crossplane.io/v1
kind: Composition
metadata:
  name: xeks.aws.example.org
  labels:
    provider: aws
    guide: quickstart
    vpcNetwork: "true"
spec:
  compositeTypeRef:
    apiVersion: example.org/v1alpha1
    kind: XEKS
  resources:
    - name: vpc
      base:
        apiVersion: ec2.aws.upbound.io/v1beta1
        kind: VPC
        spec:
          forProvider:
            region: us-west-2
            cidrBlock: 10.0.0.0/16
            enableDnsSupport: true
            enableDnsHostnames: true
      patches:
        - fromFieldPath: "metadata.uid"
          toFieldPath: "spec.writeConnectionSecretToRef.name"
          transforms:
            - type: string
              string:
                fmt: "%s-vpc"
    - name: eks-cluster
      base:
        apiVersion: eks.aws.upbound.io/v1beta1
        kind: Cluster
        spec:
          forProvider:
            region: us-west-2
            version: "1.28"
            roleArnSelector:
              matchControllerRef: true
            vpcConfig:
              - subnetIdSelector:
                  matchLabels:
                    access: public
              - securityGroupIdSelector:
                  matchControllerRef: true
          writeConnectionSecretToRef:
            namespace: crossplane-system
      patches:
        - fromFieldPath: "metadata.uid"
          toFieldPath: "spec.writeConnectionSecretToRef.name"
          transforms:
            - type: string
              string:
                fmt: "%s-eks"
      connectionDetails:
        - fromConnectionSecretKey: kubeconfig
      readinessChecks:
        - type: MatchString
          fieldPath: "status.atProvider.status"
          matchString: "ACTIVE"
```

Corrections worth flagging, because they appear in a lot of copied examples:

- `healthPolicy` is not a field on a Composition resource entry. Readiness for a composed resource is expressed with `readinessChecks` (Crossplane's own readiness evaluation) or, for provider resources, by the provider's own `status.conditions`. The version above uses `readinessChecks` with a field path into the observed state.
- `matchControllerRef: true` in a selector means "match the resource created by the same composite." It is correct here, but note that it only resolves after the referenced resource exists, so ordering matters.
- The `patches` blocks derive a connection secret name from the composite's UID. Without a stable, unique name, two composites can collide on the same secret.

**Failure mode to design around:** provider-side immutability. Many cloud fields cannot be updated in place. Changing an RDS engine version, an EKS cluster version, or a VPC CIDR may require replacement. Crossplane will attempt the change, the provider API will reject it or the resource will enter a failed state, and no amount of reconciliation will fix it. The controller is doing exactly what it should; the desired state is simply not achievable by update. Teams that model infrastructure declaratively often assume "the reconciler will handle it," and this is the assumption that breaks.

**Second failure mode:** drift that the provider API does not report. If a console change is invisible in the resource's observed state, Crossplane has nothing to compare against. Reconciliation is only as good as the provider's read API.

## How to measure detection and recovery yourself

Benchmarks comparing these tools are almost always unverifiable, because results depend on probe timings, provider latency, and which resource drifted. The useful thing is a procedure you can run on your own stack.

**Instrument these four timestamps for every simulated failure:**

1. `t_drift` — the moment you mutate the resource.
2. `t_detect` — the first time the controller reports the resource as not-ready or out-of-sync.
3. `t_act` — the first API call the controller makes to correct it.
4. `t_healthy` — the moment the health signal returns to healthy.

Detection latency is `t_detect - t_drift`. Recovery latency is `t_healthy - t_detect`. Total impact window is `t_healthy - t_drift`.

**What to watch:**

- For Argo CD: `argocd app get <app> -o yaml` shows `status.sync.status` and `status.health.status`. The controller's own metrics are exposed for Prometheus; the reconciliation and health-check counters tell you how often it is evaluating. Watch the Application's `.status.conditions` for the reason a health check is `Unknown`.
- For Crossplane: `kubectl get <kind> <name> -o yaml` and read `status.conditions` (`Synced`, `Ready`). Provider controller logs show the reconciliation attempts and the API errors. The `Synced` condition going `False` with reason `ReconcileError` is the signal that the desired state is not achievable.

**A minimal, reproducible experiment.** Pick one non-production resource. Change one mutable field out of band — for example, alter a security group rule or a bucket policy through the provider console. Then record the four timestamps above using a script that polls the status field every few seconds. Repeat five times and note the spread, not just the median. The spread is what determines whether your rollback window is predictable.

**Why the numbers vary so much between setups.** Detection latency for Argo CD is bounded below by the controller's reconciliation interval *and* by the workload's own health-probe configuration. A pod with a 60-second liveness probe and a failure threshold of 3 takes at least 180 seconds to be marked unhealthy, regardless of how often Argo CD polls. Detection latency for Crossplane is bounded by the provider controller's reconcile interval and by the provider API's own eventual consistency. Neither number is a property of the tool alone.

## Where each approach breaks down

| Situation | Argo CD behavior | Crossplane behavior |
|---|---|---|
| Custom resource with no health check | Reports `Unknown`; no rollback triggered | Not applicable; Crossplane uses provider readiness conditions |
| Workload healthy at the probe but failing requests | No rollback; health check is the weak link | No rollback; same weak link |
| Cloud field that cannot be updated in place | Not managed by Argo CD | Reconciliation fails; manual replacement needed |
| Drift invisible in the provider's read API | Not managed by Argo CD | Not detected |
| Desired state deleted from Git | Prunes managed resources (if `prune: true`) | Prunes composed resources when the XR is deleted |
| Provider API rate limiting | Not applicable | Reconciliation backs off and retries; expect delays |
| Change made directly in the cluster to a managed object | Re-applied on next sync | Not applicable |

The pattern in this table is that both tools fail at the same two places: an unreliable health signal, and a provider that will not accept the correction. Choosing between them does not remove those failure modes; it changes which layer you are debugging when they occur.

## A decision checklist

Work through these in order. The first question that produces a clear answer usually decides it.

1. **What is the resource that fails most often?** If it is a Kubernetes workload, Argo CD's model matches the problem. If it is a cloud resource — a database, a network, an IAM policy — Crossplane's model matches.
2. **Do you already have Terraform or another IaC tool that works?** Migrating existing modules to Compositions is a rewrite, not a port. If the existing tooling is not causing pain, the migration cost is unlikely to be repaid by faster reconciliation.
3. **Can your team write and debug Go?** Composition functions are the extension point for nontrivial logic. Teams without Go experience tend to hit a wall the first time a Composition needs conditional behavior.
4. **How good are your health signals today?** If your liveness probes only check that a process is listening, neither tool will roll back correctly. Fix the probes first; the tool choice is secondary.
5. **What is your actual recovery-time objective, and is it met by a manual rollback?** A Git revert plus a sync is often fast enough. If it is, the marginal gain from infrastructure-layer reconciliation may not justify the setup cost.
6. **How many distinct cloud resources do you manage?** At small counts, per-resource controllers add overhead without much benefit. At large counts, a single control plane reduces the number of places drift can hide.
7. **Can you test the failure path?** If you cannot simulate a drift event in staging and observe the rollback, you do not know whether self-healing works. This is a prerequisite, not a nice-to-have.

## Combining the two

These tools are not mutually exclusive, and the combination is often more honest than either alone.

A common arrangement: Argo CD manages the Kubernetes-side resources, including the Crossplane installation itself and the composite resources. Crossplane manages the cloud resources those composites describe. Argo CD's Git history then provides the audit trail for infrastructure changes, and Crossplane's controllers provide the reconciliation for resources that do not live in the cluster.

The tradeoff is that a failure can now originate in two control planes. When debugging, the first question becomes "which controller last touched this object?" — and the answer is in the `managedFields` of the object and in the `status.conditions` of the composite. Teams that adopt both should decide up front which system owns which class of resource, and write that boundary down. Ambiguous ownership is the failure mode that the combination introduces.

## FAQ

**Does Argo CD roll back automatically when health degrades?**
Not by default. `selfHeal: true` re-applies desired state; it does not revert to an earlier revision. Automatic rollback requires either a separate controller that watches health and rewrites the desired revision, or an Argo Rollouts-style progressive delivery setup. This distinction is the most common misunderstanding about Argo CD self-healing.

**How often does Crossplane reconcile?**
The interval is configurable per controller and per provider. There is no single default that applies to every resource. Check the provider's documentation and the controller's flags for the version you are running rather than assuming a fixed number.

**Can Argo CD manage cloud resources directly?**
It can apply any Kubernetes manifest, including Crossplane custom resources. That is the combination described above: Argo CD delivers the manifest, and Crossplane's controllers do the cloud API work. Argo CD itself does not call cloud provider APIs.

**What happens when the desired state is genuinely impossible?**
Both controllers will retry, log errors, and eventually back off. The resource stays in a failed condition. Someone has to change the desired state. Self-healing does not mean "always converges" — it means "converges when the correction is expressible as an API call the provider accepts."

**Is one of these faster?**
Detection latency depends on the controller's reconcile interval, the workload's health probe configuration, and the provider API's consistency guarantees. It is not a fixed property of either tool. Measure it on your own stack with the four-timestamp procedure above.

## Do this in the next 30 minutes

Pick one non-production resource that a controller currently manages. Note its current health or readiness condition. Change one mutable field out of band — through the provider console or `kubectl edit` — and start a timer. Poll the condition every five seconds and write down when it flips to not-ready and when it flips back. That single measurement tells you more about your pipeline's real detection latency than any comparison table, and it takes less time than reading one more evaluation.
