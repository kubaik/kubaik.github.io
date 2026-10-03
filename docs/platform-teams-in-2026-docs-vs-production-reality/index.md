# Platform teams in 2026: Docs vs. production reality

## Why documentation and production diverge

Platform documentation describes the happy path: a cluster comes up, a workload schedules, a pipeline goes green. Production adds time, drift, partial failures, and humans under pressure. The gap between the two is where most platform toil lives.

The divergence is structural, not a documentation defect. Docs are written against a clean reference environment with a single owner and no legacy. Production has clusters that were upgraded across several Kubernetes minor versions, teams that joined after the platform was built, and defaults that were correct when chosen and are now expensive. None of that shows up in a getting-started guide, because a getting-started guide cannot assume any of it.

Three categories of gap recur:

- **Default drift.** A default that was reasonable at cluster creation (storage class, garbage collection retention, node drain behavior) becomes a cost or reliability problem as usage grows.
- **Process gaps behind tooling.** A tool that supports a safe workflow does not force the workflow. A ConfigMap edit without a rollout restart is a process failure wearing a tool's clothing.
- **Ownership ambiguity.** When a platform is a side project that graduated, nobody owns the upgrade path until an upgrade breaks thirty services.

The rest of this article works through each category with concrete mechanisms, then builds a minimal internal developer platform, then covers the failure modes that survive good tooling.

## Cluster defaults that quietly cost money and stability

### Control-plane retention

Managed Kubernetes control planes expose limited tuning. On self-managed clusters, etcd retention is a common source of unbounded growth. The `--auto-compaction-retention` flag controls how much history etcd retains; leaving it at a long window while a metrics or logging workload writes heavily means the store grows with write volume, not with useful data.

The documented behavior is that etcd compacts revisions according to that flag and the configured compaction mode. The operational consequence is that a cluster ingesting high-cardinality series can accumulate a large etcd footprint with no corresponding query benefit. The fix is to set retention to a value matched to your backup and recovery window (commonly 8h–24h for high-churn clusters) and to verify the effect rather than assume it.

How to measure it, rather than trust a number:

```bash
# etcd database size on a control-plane node (self-managed)
sudo ETCDCTL_API=3 etcdctl \
  --endpoints=https://127.0.0.1:2379 \
  --cacert=/etc/kubernetes/pki/etcd/ca.crt \
  --cert=/etc/kubernetes/pki/etcd/server.crt \
  --key=/etc/kubernetes/pki/etcd/server.key \
  endpoint status --write-out=table
```

Run this before and after changing retention, and record the `DB SIZE` column. On managed offerings, the equivalent signal is the control-plane metrics endpoint or the provider's cluster-health view. Instrument it as a time series so you can see growth rate, not just absolute size.

### Storage class defaults

New clusters commonly receive a default StorageClass that is not the one you would choose today. The practical consequences are cost per GB-month and IOPS behavior. The check is one command:

```bash
kubectl get storageclass
```

Look for the `(default)` annotation. If the default is a legacy class and your workloads do not pin a class explicitly, every PersistentVolumeClaim without `storageClassName` inherits it. The remedy is to create the class you want, annotate it as default, and remove the default annotation from the legacy class — then audit existing PVCs for the class they actually bound to.

### Resource requests and limits

Requests drive scheduling; limits drive throttling and OOM behavior. A cluster where most containers have requests but no limits is vulnerable to noisy neighbors. A cluster where limits are set far below steady-state usage produces CPU throttling that looks like application latency.

Instrument three things before changing anything: container CPU throttling rate, container memory working set versus limit, and the ratio of scheduled-to-pending pods. The throttling metric is exposed by the kubelet and is the single best signal that a limit is too low.

## Internal developer platforms by company size

The shape of an internal developer platform (IDP) tracks organizational size far more than it tracks technology preference. The same primitives appear at every size; what changes is how much of the platform is enforced versus suggested.

### Small teams (roughly 1–50 engineers)

The platform is a thin opinionated layer over a managed control plane. The goal is cognitive load reduction: one manifest per service, one deployment path, no per-team variation.

Typical composition: a managed Kubernetes service, a GitOps controller watching a single repository, and a templated manifest that defines deployment, service, ingress, and autoscaling together. A common early failure is an image-pull permission gap: nodes assume a role that lacks registry read permissions, and pods fail with `ImagePullBackOff` while the application logs show nothing at all. The fix belongs in the bootstrap template, not in a runbook, because the next cluster will hit it too.

The measurement that matters at this size is time from repository creation to first successful deploy. Instrument it by timestamping the first commit and the first healthy rollout.

### Mid-size teams (roughly 50–500 engineers)

The platform becomes a negotiation between autonomy and control. Composition typically includes:

- One cluster per environment (dev, staging, production)
- A declarative provisioning layer for managed services (databases, caches, object storage)
- A developer portal with a service catalog
- Policy as code for admission control
- In-cluster workflow execution for CI/CD steps that need cluster access

The dominant failure here is manifest sprawl. When every team may customize its deployment, the platform accumulates one variant per team: some Helm, some Kustomize overlays, some raw manifests. Debugging requires reading each variant. The countermeasure is a single shared chart with enforced defaults and an explicit escape hatch that requires review.

### Large organizations (roughly 500+ engineers)

The platform becomes a shared-responsibility model with hard boundaries: multiple clusters segmented by business unit or compliance region, mutual TLS enforced by a service mesh, centralized secrets management, cost allocation via tagging, and admission policies for security controls.

The complexity at this size is organizational. When many teams deploy to one cluster without coordination, the symptoms are resource starvation at peak, noisy-neighbor latency, and cloud spend that cannot be attributed to a team. The technical fix — namespace quotas and limit ranges — is straightforward. The durable fix is topology: separate clusters or separate node pools per team or business unit, so that one team's burst cannot consume another's headroom.

A useful diagnostic at this scale is a cost-allocation coverage check: what fraction of resources carry a tag that maps to an owning team? Anything untagged is unattributable, and unattributable spend cannot be optimized.

## Building a minimal platform: provisioning, deployment, policy

The example below targets a mid-size organization. The goal is a consistent, auditable path from "new service" to "running in staging" without a ticket to the platform team. Three capabilities: cluster bootstrap, a standardized service chart, and admission policy.

### Step 1: Bootstrap the cluster with opinionated defaults

The following Terraform uses the community EKS module. Pin versions and verify the module's current major version before applying; module interfaces change between majors.

```hcl
# main.tf
terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.25"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
  }
}

provider "aws" {
  region = "us-west-2"
}

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 19.16"

  cluster_name    = "dev-cluster"
  cluster_version = "1.30"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  eks_managed_node_groups = {
    default = {
      min_size     = 3
      max_size     = 10
      desired_size = 3

      instance_types = ["t4g.large"] # ARM64 Graviton
      capacity_type  = "SPOT"
      labels = {
        "node-group" = "default"
      }
      taints = []
    }
  }

  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
      configuration_values = jsonencode({
        enableNetworkPolicy = "true"
      })
    }
    aws-ebs-csi-driver = {
      most_recent = true
      configuration_values = jsonencode({
        storage = {
          gp3 = {
            fsType = "ext4"
            type   = "gp3"
          }
        }
      })
    }
  }
}
```

Design decisions worth stating explicitly:

- **ARM64 node types.** Graviton instances generally offer better price/performance than comparable x86 instances for containerized workloads. The trade-off is image compatibility: every image must have an ARM64 variant, or the workload must run under emulation, which is slower and often not worth it.
- **Spot capacity.** Spot reduces compute cost substantially but introduces interruption. Only use it for workloads that tolerate eviction — stateless services with more than one replica and a disruption budget. Never use Spot for a single-replica stateful workload.
- **EBS CSI driver with gp3.** Installing the driver and configuring gp3 gives you a modern default storage class instead of inheriting the legacy default.
- **VPC CNI with network policy enabled.** Enabling network policy support at bootstrap means policies are enforceable later without a CNI change.

Verify the result before building on it:

```bash
kubectl get nodes -o wide
kubectl get storageclass
```

The storage class output should show your intended class as `(default)`. If it does not, fix it now — this is the cheapest moment.

### Step 2: A single shared Helm chart

Create the chart skeleton:

```bash
helm create platform-service
cd platform-service
rm -rf templates/*
```

```yaml
# Chart.yaml
apiVersion: v2
name: platform-service
description: "Standardized service deployment for internal platform"
version: 0.1.0
type: application
appVersion: "1.0"
```

```yaml
# values.yaml
replicaCount: 2

image:
  repository: "public.ecr.aws/my-org/default-service"
  tag: "latest"
  pullPolicy: IfNotPresent

resources:
  requests:
    cpu: "100m"
    memory: "256Mi"
  limits:
    cpu: "500m"
    memory: "512Mi"

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 5
  targetCPUUtilizationPercentage: 70

env:
  - name: LOG_LEVEL
    value: "info"
```

```yaml
# templates/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "platform-service.fullname" . }}
  labels:
    {{- include "platform-service.labels" . | nindent 4 }}
spec:
  replicas: {{ .Values.replicaCount }}
  selector:
    matchLabels:
      {{- include "platform-service.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      labels:
        {{- include "platform-service.selectorLabels" . | nindent 8 }}
    spec:
      containers:
        - name: {{ .Chart.Name }}
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          ports:
            - name: http
              containerPort: 8080
              protocol: TCP
          resources:
            {{- toYaml .Values.resources | nindent 12 }}
          env:
            {{- toYaml .Values.env | nindent 12 }}
          livenessProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /ready
              port: http
            initialDelaySeconds: 5
            periodSeconds: 5
      nodeSelector:
        kubernetes.io/os: linux
```

Two things to note. First, the probe paths and ports are fixed by the chart, which means every service must expose `/health` and `/ready` on the named `http` port. That is a real constraint and it should be documented as such — teams will push back, and the counterargument is that uniform probes make rollout behavior predictable across the fleet. Second, the default `image.tag: "latest"` is convenient for a first deploy and dangerous for production. Override it with an immutable tag (a commit SHA or digest) in the environment-specific values file.

### Step 3: Enforce the standard with admission policy

A chart that teams *may* use is a suggestion. A policy that rejects non-conforming manifests is a standard. Kyverno is one option for admission-time validation without writing a custom controller.

```bash
helm repo add kyverno https://kyverno.github.io/kyverno
helm repo update
helm install kyverno kyverno/kyverno -n kyverno --create-namespace
```

```yaml
# policies/require-resource-limits.yaml
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: require-resource-limits
  annotations:
    policies.kyverno.io/title: "Require resource limits"
    policies.kyverno.io/severity: "medium"
spec:
  validationFailureAction: enforce
  background: true
  rules:
    - name: check-resource-limits
      match:
        any:
          - resources:
              kinds:
                - Deployment
      validate:
        message: "Resource limits are required."
        pattern:
          spec:
            template:
              spec:
                containers:
                  - name: "*"
                    resources:
                      limits:
                        memory: "?*"
                        cpu: "?*"
```

```bash
kubectl apply -f policies/require-resource-limits.yaml
```

A non-conforming manifest is then rejected at admission:

```yaml
# bad-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bad-service
spec:
  replicas: 1
  selector:
    matchLabels:
      app: bad-service
  template:
    metadata:
      labels:
        app: bad-service
    spec:
      containers:
        - name: bad-service
          image: nginx:latest
          resources:
            requests:
              cpu: "100m"
              memory: "128Mi"
            # no limits
```

```bash
kubectl apply -f bad-deployment.yaml
# Error from server: admission webhook "validate.kyverno.svc-fail" denied the request:
# resource Deployment/default/bad-service is disallowed for the following reason:
# Resource limits are required.
```

The important operational detail is that admission webhooks sit in the request path. If the policy engine is unavailable and the webhook is configured to fail closed, deployments stop. Decide deliberately whether each policy fails open or closed, and monitor the webhook's latency and error rate as a first-class platform metric.

## How to measure whether the platform is working

Published platform benchmarks are rarely transferable, because they depend on workload mix, cluster size, and team behavior. Measure your own system. The table below lists the signals worth instrumenting and where each comes from.

| Signal | Where it comes from | What a bad reading looks like |
|---|---|---|
| Deployment success rate | CI/CD system records, per environment | Failures clustered in one team or one chart version |
| Time to first successful deploy | Timestamp of repo creation vs. first healthy rollout | Long tail driven by onboarding steps, not build time |
| Lead time for change | Version control to production | Growing while deploy count is flat |
| CPU throttling rate | Container metrics from the kubelet | Non-zero under normal load means limits are too low |
| Pending pod duration | Scheduler metrics | Spikes at peak indicate capacity or request inflation |
| Admission webhook latency | Policy engine metrics | p99 rising toward the API server timeout |
| Cost per namespace | Cloud billing joined to resource tags | Untagged resources with no owner |
| Developer satisfaction | Short recurring survey, one or two questions | Falling after a platform change, not before |

Two cautions on interpreting these. First, deployment success rate improves trivially if you make the pipeline accept anything — pair it with a policy-violation count so the two cannot both be gamed. Second, a latency increase after adding a service mesh is expected; the question is whether the error-rate reduction justifies it, and that is a per-organization judgment, not a universal rule.

## Failure modes that survive good tooling

### The platform as a side project

A platform often begins as one engineer's chart in a repository. It works, so it spreads, until most services depend on it. Nobody owns the upgrade path. When a Kubernetes minor version removes a deprecated API field, every dependent service breaks at once.

The countermeasure is treating the platform as a product with a versioning policy: semantic versions for charts, a documented deprecation window, and a CI job that renders every consumer's manifests against the next cluster version before you upgrade. That render check is the single highest-value test a platform team can add.

### The golden path that becomes a bottleneck

A golden path is the recommended route. When it cannot express a legitimate need, teams route around it, and the platform loses the standardization it was built for. The signal is the number of distinct deployment mechanisms in use. If it is growing, the golden path is missing a capability, not being disrespected.

The fix is to make the golden path the only supported path *and* to give it a fast, documented extension mechanism. A design review for every exception is a bottleneck; a parameterized chart with a reviewed escape hatch is not.

### Convenience with a hidden bill

Managed observability is convenient and its costs are opaque. Log ingestion, log retention, and cross-AZ or cross-region data transfer are the usual culprits. A logging pipeline that ships every application log at debug level can generate substantial egress and storage cost with no corresponding debugging value.

Instrument ingestion volume per service before optimizing anything. Then apply retention tiers: hot storage for a short window, object storage for the rest. The same discipline applies to metrics — high-cardinality labels are the most common cause of metrics cost growth.

### Cultural debt

Technical debt is visible in a repository. Cultural debt is invisible until adoption stalls. A platform built without input from the engineers who must use it tends to be adopted only where mandated. The measurable symptom is the gap between the number of services that *could* use a platform feature and the number that do.

The countermeasure is to treat adoption as a product metric and to run recurring intake sessions where engineers bring their actual friction points. Features that come from those sessions get adopted; features that come from platform-team intuition often do not.

### The compliance illusion

Passing an audit is not the same as enforcing a control. A secrets-rotation policy that exists in a document but not in the pipeline does not rotate secrets. The check is to trace one control end to end: for a given secret type, what enforces rotation, where is the evidence, and what happens when the enforcement fails?

The common gap is the CI/CD path. If secrets are injected as static environment variables at build time, rotation is manual by construction. Integrating a secrets manager with short-lived credentials issued per pipeline run removes the class of problem rather than detecting it.

## A tooling checklist by capability

Rather than a list of products, here is the capability set a platform needs, with the question to ask of any candidate:

| Capability | Question to ask | Common failure |
|---|---|---|
| Infrastructure provisioning | Can a new environment be created from version-controlled input alone? | Manual console steps that are undocumented |
| GitOps reconciliation | Does the cluster converge to the repository state without human action? | Drift that only surfaces during an incident |
| Managed service provisioning | Can a database be requested declaratively with the same review process as code? | Ticket queues for routine provisioning |
| Policy enforcement | Are violations rejected at admission, and is the webhook's availability monitored? | Policies that exist but are not enforced |
| Secrets management | Are credentials short-lived and issued per workload identity? | Long-lived static secrets in CI variables |
| Service mesh | Is mTLS enforced, and is the added latency measured against the error-rate benefit? | Mesh installed but not enforced |
| Linting and scanning | Do manifest checks run in CI and block merge? | Scanners available but optional |
| Cost attribution | Does every resource carry an owner tag? | Untagged spend with no accountable team |
| Developer portal | Do engineers use it voluntarily? | Portal that duplicates information available elsewhere |

## FAQ

**Should a small team build an IDP at all?**
Only enough to remove repeated manual steps. For a small team, that is usually a cluster bootstrap template, one shared chart, and a GitOps controller. Anything more is a maintenance liability until the team grows.

**How do you introduce policy enforcement without blocking delivery?**
Start in audit mode. Run the policy with `validationFailureAction: audit`, collect violations for a period, fix the templates that cause most of them, then switch to enforce. Switching to enforce on day one converts a standards problem into an outage.

**Is a service mesh required for mTLS?**
No. Some stacks provide workload identity and encryption at a lower layer. A mesh adds traffic management and observability alongside mTLS, at the cost of an extra hop and a new control plane to operate. Decide based on whether you need the traffic features, not on mTLS alone.

**How do you choose between one cluster and many?**
The deciding factors are blast radius, compliance boundaries, and cost attribution. Many small clusters give cleaner isolation and simpler quotas; one large cluster gives better utilization and simpler networking. Most organizations end up with a small number of clusters segmented by environment and compliance domain, not one per team.

**What is the first metric to instrument?**
Deployment success rate per environment, paired with policy-violation count. Together they show whether delivery is getting better or whether the pipeline is simply accepting more.

## Next 30 minutes: run one audit

Pick your production cluster and run these three commands. Record the output somewhere durable.

```bash
# 1. Which storage class will untyped PVCs inherit?
kubectl get storageclass

# 2. Which workloads have no CPU or memory limits?
kubectl get pods -A -o json | \
  jq -r '.items[] |
    select(.spec.containers[]?.resources.limits == null) |
    "\(.metadata.namespace)/\(.metadata.name)"'

# 3. Are there namespaces with no resource quota?
kubectl get ns -o json | \
  jq -r '.items[].metadata.name' | \
  while read ns; do
    kubectl get resourcequota -n "$ns" --no-headers 2>/dev/null | \
      grep -q . || echo "no quota: $ns"
  done
```

The output is your starting backlog. The storage class tells you whether new volumes are silently inheriting a legacy default. The unlabeled-limits list tells you which workloads can become noisy neighbors. The quota list tells you which namespaces can consume unbounded cluster resources. Fix the storage class default first — it is a one-line change with the widest blast radius.
