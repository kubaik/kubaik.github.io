# HPA vs KEDA vs Karpenter: autoscaling traps

## The problem this solves

A Kubernetes cluster usually hosts three workload shapes at once. Steady HTTP APIs. Queue consumers that sit idle until a burst of jobs arrives. Batch jobs that need a GPU for twenty minutes and then vanish. Three autoscalers are commonly available — HorizontalPodAutoscaler (HPA), KEDA, and Karpenter — and they overlap in ways that are not obvious from their names. HPA scales pods based on CPU, memory, or custom metrics. KEDA scales pods based on external events such as queue depth, Kafka lag, or cron schedules. Karpenter scales *nodes* by provisioning cloud instances directly, bypassing the traditional Cluster Autoscaler and node-group model.

The trap is assuming the three are interchangeable, or that stacking all three without thought produces a system that scales correctly. A typical failure mode is a system that scales too late, too aggressively, or not at all, and the failure is invisible until traffic arrives. This article is a decision framework plus working configuration for each tool, with the specific edge cases that cause incidents: HPA's missing-metrics behavior, KEDA's scale-to-zero cold-start cost, and Karpenter's interaction with pod disruption budgets and spot interruptions. By the end, a reader should be able to look at a workload and know which autoscaler owns it.

## The three tools, and what each actually controls

Before any YAML, get the mental model right. These tools operate at different layers.

**HPA** is a built-in Kubernetes controller. It reads metrics from the metrics API (served by metrics-server for resource metrics, or by an adapter for custom and external metrics) and adjusts the `replicas` field on a Deployment, ReplicaSet, or StatefulSet. It does not create nodes. If the cluster has no schedulable capacity, HPA will scale the replica count up and the new pods will sit `Pending`.

**KEDA** is a Kubernetes Operator that installs a `ScaledObject` custom resource. KEDA does not replace HPA — it creates and manages an HPA object under the hood, feeding it metrics from external sources via the Kubernetes external metrics API. When a `ScaledObject` targets zero replicas, KEDA activates the workload from zero and then hands control to the HPA it manages. KEDA supports scaling to zero, which a plain HPA cannot do.

**Karpenter** is a node autoscaler. It watches for unschedulable pods and provisions cloud instances that match the pod's resource requests, node selectors, and tolerations. It also consolidates underutilized nodes. Karpenter does not scale pods; it makes sure pods have somewhere to land.

The layers are therefore: KEDA or HPA decide *how many pods*. Karpenter decides *how many nodes and of what shape*. A common mistake is treating Karpenter as a replacement for HPA — it is not. A Deployment with a fixed replica count of 3 will stay at 3 pods no matter how many nodes Karpenter provisions.

| Tool | Scales | Metric source | Scale to zero | Node awareness |
|---|---|---|---|---|
| HPA | Pod replicas | CPU, memory, custom/external metrics | No (minReplicas ≥ 1) | No |
| KEDA | Pod replicas (via managed HPA) | Scaler-specific: SQS, Kafka, Prometheus, cron, and others | Yes | No |
| Karpenter | Nodes (cloud instances) | Pod resource requests, node selectors, taints | N/A | Yes |

## Prerequisites and what to build

A working cluster is assumed, with a Kubernetes version supported by the Karpenter release in use — check the Karpenter compatibility matrix for the exact supported versions rather than assuming. `kubectl` and `helm` are required. For KEDA, the KEDA Helm chart. For Karpenter, the Karpenter Helm chart plus IAM permissions for the controller to create instances — the exact policy differs by cloud provider, so follow the official Karpenter getting-started guide for that provider rather than copying a policy from a blog post.

Three workloads represent the three shapes:

1. A stateless HTTP API scaled by CPU with HPA — the boring case, done correctly.
2. A queue consumer scaled by SQS depth with KEDA, including scale-to-zero.
3. A batch job that requests a GPU, scheduled onto a node that Karpenter provisions on demand.

## Step 1 — Set up the environment and verify the metrics pipeline

HPA and KEDA both depend on a working metrics pipeline. If `kubectl top pods` returns an error, HPA will not scale and the reason will be buried in events. Verify first.

```bash
# metrics-server is required for CPU/memory HPA
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
kubectl top nodes
kubectl top pods -A | head
```

If `kubectl top` fails, check that metrics-server pods are running and that the API service `v1beta1.metrics.k8s.io` is available:

```bash
kubectl get apiservice v1beta1.metrics.k8s.io -o yaml
```

A common failure is metrics-server unable to scrape kubelets because of TLS certificate issues — the fix is provider-specific (adding `--kubelet-insecure-tls` is a workaround, not a production fix). Do not proceed to HPA until `kubectl top pods` works.

For KEDA, install the operator:

```bash
helm repo add kedacore https://kedacore.github.io/charts
helm repo update
helm install keda kedacore/keda --namespace keda --create-namespace
kubectl get pods -n keda
```

For Karpenter, install the controller per the official guide for the provider. The controller needs a node IAM role and a controller IAM role; the exact names and policies are documented per provider. Do not guess them.

## Step 2 — HPA for steady HTTP workloads

HPA is the right tool when traffic correlates with a resource that can be measured — CPU, memory, or a custom metric the app exports. The default behavior scales on average CPU utilization across pods.

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api
  minReplicas: 2
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 60
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
        - type: Percent
          value: 100
          periodSeconds: 15
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Percent
          value: 25
          periodSeconds: 60
```

The `behavior` block is where most production HPA configs go wrong. By default, HPA scales down after a 5-minute stabilization window, which is reasonable. Scale-up can be sluggish if the default 0-second window is combined with a low percentage policy — the controller may only add one pod per evaluation cycle. The config above allows doubling every 15 seconds on the way up and removes at most 25% of pods per minute on the way down. The asymmetry is deliberate: scale up fast, scale down slow, because a premature scale-down followed by a traffic spike causes a thundering herd on the remaining pods.

**Gotcha: HPA and missing metrics.** When the metrics API returns no data for a pod — for example, a pod that just started and has not been scraped yet — HPA treats that pod's utilization as unknown. If enough pods are unknown, HPA cannot compute a ratio and does not scale. The event on the HPA object reports that it failed to get CPU utilization. The fix is to set resource *requests* on every container. HPA computes utilization as a percentage of the request, not of the limit. A container with no CPU request is invisible to HPA.

**Gotcha: HPA with memory.** Memory-based HPA is a trap for most workloads. Memory usage rarely drops when load drops, because runtimes hold onto heap. An HPA targeting 70% memory will scale up and then never scale down. Use CPU or a custom request-rate metric for HTTP services.

For a custom metric, expose something like requests-per-second via Prometheus and use the Prometheus Adapter to register it as a custom metric. That is more work but produces scaling that tracks traffic rather than CPU, which is often a poor proxy for a service that is I/O-bound.

**How to measure whether the HPA is doing its job.** Run a fixed load generator at a known request rate against the service, record the pod count every 15 seconds with `kubectl get deploy api -o jsonpath='{.status.replicas}'`, and plot pod count against request rate. A healthy HPA produces a roughly linear relationship between load and replicas once above `minReplicas`. A flat line means the metric is missing or the request is not set; a sawtooth means the scale-down policy is too aggressive relative to the traffic variance.

## Step 3 — KEDA for event-driven and scale-to-zero workloads

KEDA's value is twofold: it can read metrics HPA cannot (queue depth, Kafka consumer lag, cron schedules), and it can scale a Deployment to zero replicas when there is no work. That second property is what makes it appropriate for queue consumers that are idle most of the day.

Here is a `ScaledObject` for an SQS consumer:

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: sqs-consumer
spec:
  scaleTargetRef:
    name: sqs-consumer
  minReplicaCount: 0
  maxReplicaCount: 30
  pollingInterval: 15
  cooldownPeriod: 120
  triggers:
    - type: aws-sqs-queue
      metadata:
        queueURL: https://sqs.us-east-1.amazonaws.com/123456789012/jobs
        queueLength: "10"
        awsRegion: us-east-1
        identityOwner: pod
```

`queueLength: "10"` means KEDA targets one replica per 10 messages in the queue. With 100 messages visible, it asks for 10 replicas. `pollingInterval: 15` means it checks SQS every 15 seconds. `cooldownPeriod: 120` means after the last message is consumed, it waits two minutes before scaling to zero.

Those three numbers are where the tuning happens. A short polling interval reacts faster but costs more SQS API calls and more load on the scaler. A long cooldown avoids flapping when the queue drains and refills, but keeps idle pods running longer. For a queue that bursts every few minutes, a cooldown of 120–300 seconds is a reasonable starting point. For a queue that gets one message per hour, a shorter cooldown saves money.

**Gotcha: scale-to-zero cold start.** When KEDA scales a Deployment from 0 to 1, the new pod must schedule, pull its image, start the process, and connect to the queue. That is not instant. If the queue receives a burst and the consumer takes 30 seconds to become ready, messages sit in the queue for 30 seconds. For latency-sensitive queues, set `minReplicaCount: 1` and accept the cost of one warm pod. Scale-to-zero suits workloads where a cold start is acceptable — batch processing, nightly jobs, low-priority consumers.

**Gotcha: KEDA and HPA conflict.** KEDA creates an HPA object named `keda-hpa-<scaledobject-name>`. If a separate HPA is also created targeting the same Deployment, the two fight. Do not mix them on the same workload. Pick one.

KEDA also supports cron triggers, which is useful for predictable traffic:

```yaml
  triggers:
    - type: cron
      metadata:
        timezone: UTC
        start: "0 8 * * *"
        end: "0 20 * * *"
        desiredReplicas: "5"
```

This scales to 5 replicas during business hours and back down outside them, regardless of load. It is a blunt instrument, but for a service with a known daily pattern it is more reliable than waiting for CPU to rise.

**Worked example: sizing the queue consumer.** Suppose messages arrive in bursts of 2,000, each message takes 250 ms to process, and the target is to drain a burst in under 2 minutes. One replica processes 4 messages per second, so 2,000 messages need 500 replica-seconds. Draining in 120 seconds requires about 4.2 replicas, so 5 replicas. With `queueLength: "10"`, KEDA asks for 2,000 / 10 = 200 replicas, which exceeds `maxReplicaCount: 30`. The trigger is therefore not the constraint — the ceiling is. Either raise `maxReplicaCount`, raise `queueLength` so the trigger asks for fewer replicas, or accept a longer drain time. Running the arithmetic before deployment makes the trade-off explicit.

## Step 4 — Karpenter for node provisioning

Karpenter's job is to make sure pods have nodes to run on, and to remove nodes that are no longer needed. It does this by watching for unschedulable pods and provisioning instances that satisfy their requirements.

A minimal `NodePool` and `EC2NodeClass` (AWS-specific; other providers have equivalents):

```yaml
apiVersion: karpenter.sh/v1
kind: NodePool
metadata:
  name: default
spec:
  template:
    spec:
      requirements:
        - key: kubernetes.io/arch
          operator: In
          values: ["amd64"]
        - key: karpenter.sh/capacity-type
          operator: In
          values: ["on-demand", "spot"]
        - key: node.kubernetes.io/instance-type
          operator: In
          values: ["m5.large", "m5.xlarge", "m5.2xlarge"]
  limits:
    cpu: "1000"
  disruption:
    consolidationPolicy: WhenEmptyOrUnderutilized
    consolidateAfter: 30s
---
apiVersion: karpenter.k8s.aws/v1
kind: EC2NodeClass
metadata:
  name: default
spec:
  amiFamily: AL2
  role: KarpenterNodeRole
  subnetSelectorTerms:
    - tags:
        karpenter.sh/discovery: my-cluster
  securityGroupSelectorTerms:
    - tags:
        karpenter.sh/discovery: my-cluster
```

The `requirements` block constrains what Karpenter is allowed to provision. Allowing both `on-demand` and `spot` lets Karpenter choose spot when available, which is cheaper but interruptible. The `consolidationPolicy` tells Karpenter to remove nodes that are empty or underutilized — this is how nodes that HPA scaled down stop costing money.

**Gotcha: Karpenter and pod disruption budgets.** When Karpenter consolidates a node, it evicts pods. If a pod's PodDisruptionBudget (PDB) does not allow eviction, the node stays. A common pattern is a PDB with `minAvailable: 1` on a Deployment that Karpenter wants to consolidate from 2 pods to 1 — the eviction is blocked and the node lingers. Check `kubectl get pdb -A` and the Karpenter logs when consolidation seems stuck.

**Gotcha: Karpenter and spot interruptions.** Spot instances can be reclaimed with a short warning. Karpenter handles the interruption notice by cordoning and draining the node, but pods need to tolerate that. For stateful workloads or long-running batch jobs, spot is a poor fit unless the job can checkpoint. For stateless services with a PDB and multiple replicas, spot is usually fine.

**Gotcha: Karpenter does not scale pods.** This is worth repeating because it is the most common misconception. Deploying a GPU batch job with `replicas: 1` causes Karpenter to provision one GPU node. Ten parallel batch workers require either a fixed replica count of 10 or an HPA/KEDA scaling on a queue. Karpenter only responds to what the scheduler cannot place.

**How to measure node provisioning latency.** Record the time a pod first appears as `Pending` and the time it becomes `Running`, using `kubectl get events --sort-by=.lastTimestamp` or a pod-startup-latency metric from the scheduler. Repeat during a controlled scale-up. That interval — not the HPA reaction time — is the floor on how fast the cluster can absorb a spike that exceeds current capacity.

## How the three interact in practice

A typical production setup uses all three, each at its layer:

- HPA scales the HTTP API between 2 and 20 pods based on CPU.
- KEDA scales the queue consumer between 0 and 30 pods based on SQS depth.
- Karpenter provisions nodes when either of the above creates pods that do not fit, and consolidates when they scale down.

The interaction to watch is timing. HPA reacts in tens of seconds. KEDA's polling interval adds up to 15 seconds of lag. Karpenter's node provisioning takes minutes — instance launch, kubelet registration, image pull. So the end-to-end time from traffic spike to a new pod serving requests is dominated by Karpenter, not HPA. If HPA scales to 20 pods but the cluster only has capacity for 5, the other 15 sit `Pending` until Karpenter catches up. That is why `minReplicas` and node headroom matter: keep enough spare capacity to absorb the first minute of a spike, and let Karpenter handle the sustained load.

A useful pattern is to set HPA `minReplicas` to a value that fits on the existing nodes, and let Karpenter handle anything above that. If HPA's minimum is 2 and the nodes can hold 6 pods, a spike to 6 is instant; a spike to 20 triggers Karpenter.

## Decision checklist: which tool owns the workload

Use this to classify a workload before writing any YAML.

1. **Is the load measurable as CPU, memory, or an in-process custom metric, and does the workload need at least one replica at all times?** Use HPA alone.
2. **Does the workload need to reach zero replicas when idle, or does the trigger live outside the app (queue, stream, cron)?** Use KEDA. If it also needs CPU-based scaling, add a CPU trigger to the same `ScaledObject` rather than a second HPA.
3. **Does the workload need hardware the default node pool does not offer (GPU, ARM, large memory, spot)?** That is a Karpenter `NodePool` requirement, independent of the pod scaler.
4. **Is the workload a batch job that runs to completion?** Consider a Kubernetes Job instead of a Deployment; HPA does not scale Jobs, and KEDA's `ScaledJob` covers queue-driven batch work.
5. **Is a cold start acceptable?** If not, `minReplicaCount: 1` and keep warm capacity; Karpenter cannot make an image pull faster.
6. **Can the workload tolerate eviction?** If not, it does not belong on spot, and its PDB will block consolidation.

## Verification checklist

After configuring all three, verify each layer independently:

1. **Metrics:** `kubectl top pods` returns data for every pod expected to scale.
2. **HPA:** `kubectl describe hpa api-hpa` shows current metrics and no `failed to get` events.
3. **KEDA:** `kubectl get scaledobject` shows `READY=True`, and `kubectl get hpa` shows a `keda-hpa-*` object for each ScaledObject.
4. **Karpenter:** `kubectl get nodepool` shows `READY=True`, and the Karpenter controller logs show no provisioning errors.
5. **End-to-end:** generate load against the API and watch `kubectl get pods -w` — HPA should add pods and Karpenter should add nodes if needed. Send a burst of messages to the queue and watch the consumer scale from zero.

If HPA does not scale, check events on the HPA object first. If KEDA does not scale, check the ScaledObject status and the KEDA operator logs. If Karpenter does not provision, check for unschedulable pods with `kubectl get pods -A --field-selector=status.phase=Pending` and read the events.

## Frequently Asked Questions

**Can HPA and KEDA be used together on the same Deployment?**
Not as two separate objects. KEDA creates and manages its own HPA object for each ScaledObject. If a standalone HPA also targets the same Deployment, the two controllers fight over the replica count. Pick one. If both CPU-based and queue-based scaling are needed, use KEDA with multiple triggers, since KEDA supports a CPU trigger alongside external triggers.

**Does Karpenter replace the Cluster Autoscaler?**
Yes, in the sense that one or the other is run, not both. Karpenter provisions nodes directly rather than through Auto Scaling Groups, which lets it choose instance types based on pod requirements rather than predefined node groups. Running both against the same cluster causes conflicting node lifecycle decisions.

**Why does an HPA show "unknown" for current metrics?**
The most common cause is missing CPU requests on the container. HPA computes utilization as a percentage of the request, so a container without a request has no denominator. The second most common cause is metrics-server not scraping the pod yet — new pods take a scrape interval to appear. Check `kubectl describe hpa` for the specific event.

**How do you scale a GPU batch job that runs occasionally?**
Use KEDA with a queue trigger (SQS, RabbitMQ, or a cron trigger) to scale the job Deployment from zero, and Karpenter with a NodePool that allows GPU instance types. The pod template must request `nvidia.com/gpu: 1` so Karpenter knows to provision a GPU node. Set `minReplicaCount: 0` so the job costs nothing when idle, and accept the cold-start time for the node and image pull.

**Why does a node stay around after load drops?**
Check PDBs first: an eviction that violates a PDB blocks consolidation. Second, check whether pods on the node have local storage or `emptyDir` data that makes them non-evictable by Karpenter's policy. Third, confirm `consolidationPolicy` is set to `WhenEmptyOrUnderutilized` rather than `WhenEmpty` if underutilized nodes should be removed.

## Where to go from here

Start with the workload that is least understood: list the HPA objects in the cluster and confirm that every container in the target Deployments has a CPU request set. That single check catches the most common autoscaling failure before it reaches production. Run `kubectl get hpa -A -o yaml | grep -A2 'target:'`, then cross-check each target Deployment's pod template for a `requests.cpu` value on every container — if one is missing, add it and watch the HPA's behavior change on the next load test.
