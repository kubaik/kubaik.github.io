# CI/CD pipelines stall with GPU pods: 2026 benchmarks

## Why GPU pods stall while CPU pods finish

A GPU job that runs in minutes on a CPU node can sit `Pending` for half an hour on a GPU node pool, then get killed by a job deadline. The pipeline UI often points at the image or the application logs, so engineers debug the container instead of the scheduler. The container is usually fine. The pod never reached a node.

Scheduling failures are quiet by design. The scheduler records a `FailedScheduling` event and retries, and retries produce more of the same event. If nobody reads events, the only visible symptom is elapsed time.

This article covers the four scheduling-layer causes that account for most GPU pipeline stalls, how to confirm each one with a command, and how to keep them from recurring.

## How Kubernetes actually schedules a GPU pod

A GPU is not a normal extended resource. The kubelet does not know how many GPUs a node has. A device plugin DaemonSet runs on each node, discovers the hardware, and registers the resource name and count with the kubelet through the Device Plugin API. Only then does the node advertise capacity, and only then can the scheduler place a pod that requests that resource.

This produces three distinct failure surfaces:

1. **The resource name in the manifest does not match the name the plugin advertises.** The node reports zero of the requested resource, so the pod never fits.
2. **The resource name matches, but the node has no free GPUs.** The pod waits for capacity or for a new node.
3. **The pod fits the resource request but is excluded by a selector, affinity rule, or taint.** Capacity is irrelevant because the pod is not eligible for any node.

A fourth surface sits outside the scheduler: a namespace `ResourceQuota` can reject the pod at admission, before scheduling is attempted at all. That failure looks different in events, which is useful—it tells you to stop looking at nodes.

### Reading the events correctly

```sh
kubectl describe pod <pod-name> -n <namespace>
```

Read the `Events` section at the bottom. The `Reason` field separates the surfaces above:

| Event reason | What it means | Where to look next |
|---|---|---|
| `FailedScheduling` with "Insufficient nvidia.com/gpu" or equivalent | No node advertises enough of that resource | Device plugin health, resource name |
| `FailedScheduling` with "node(s) didn't match node selector" | Pod is ineligible for every candidate node | `nodeSelector`, affinity |
| `FailedScheduling` with "node(s) had untolerated taint" | Node taints are not matched by pod tolerations | Taints and tolerations |
| `FailedScheduling` with "exceeded quota" | Namespace quota blocks the request | `ResourceQuota`, `LimitRange` |
| No events at all, pod stuck `Pending` | Often a scheduler backlog or a webhook | Scheduler logs, admission webhooks |

Run a cluster-wide sweep when you do not know which pod is at fault:

```sh
kubectl get events -A --field-selector reason=FailedScheduling \
  --sort-by=.lastTimestamp
```

The most recent entries are at the bottom.

## Cause 1: resource name mismatch between manifest and device plugin

Extended resource names are arbitrary strings. The device plugin chooses the name it advertises, and the pod must request that exact string. When the two disagree, the node advertises zero of the requested resource, and the scheduler reports insufficient capacity even though idle GPUs are sitting on the node.

Common sources of mismatch:

- The device plugin was upgraded or replaced and now advertises a different name than the one in your manifests.
- A cloud provider's managed GPU add-on advertises a vendor-specific name while your Helm chart still requests the upstream default.
- A node pool was rebuilt with a different AMI that ships a different plugin version.

Confirm what a node actually advertises:

```sh
kubectl get node <gpu-node> -o jsonpath='{.status.capacity}' | jq
```

Compare the GPU keys in that output against the `resources.requests` and `resources.limits` keys in your pod spec. They must match character for character.

A second, subtler mismatch is the plugin reporting not-ready. Check the DaemonSet:

```sh
kubectl get pods -n kube-system -l name=nvidia-device-plugin-ds -o wide
kubectl logs -n kube-system <device-plugin-pod> --tail=100
```

A plugin that fails to start—because the driver is missing, the runtime hook is absent, or the node cannot reach its metadata service—registers nothing. The node appears to have no GPUs.

### Validating the fix

A minimal smoke job isolates scheduling from application code:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: gpu-smoke
spec:
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: smoke
        image: nvidia/cuda:12.4.1-base-ubuntu22.04
        command: ["nvidia-smi"]
        resources:
          limits:
            nvidia.com/gpu: "1"
```

Replace the resource key with whatever your node advertises. Apply it, then watch:

```sh
kubectl get pod -l job-name=gpu-smoke -w
```

If the pod reaches `Running` and `nvidia-smi` prints a device table, the plugin, the resource name, and the runtime hook are all healthy. If it stays `Pending`, the events tell you which of the remaining causes applies.

## Cause 2: selectors, affinity, and tolerations that exclude GPU nodes

A pod can request a GPU and still be ineligible for every GPU node. The scheduler evaluates eligibility before capacity, so a pod excluded by a selector will never be placed, no matter how many idle GPUs exist.

The usual culprits:

- A `nodeSelector` pinning the pod to a CPU instance family, often left over from an earlier version of the workload.
- A toleration for a CPU-only taint, with no toleration for the GPU node pool's taint.
- A `requiredDuringSchedulingIgnoredDuringExecution` node affinity rule that names a node pool which no longer exists or is scaled to zero.
- Pod anti-affinity that prevents two replicas from sharing a node, combined with a GPU node pool smaller than the replica count.

Inspect the node's taints and labels, then compare them to the pod spec:

```sh
kubectl get node <gpu-node> -o jsonpath='{.spec.taints}'
kubectl get node <gpu-node> --show-labels
```

A GPU node pool is typically tainted so that only GPU workloads land on expensive hardware. A pod that does not tolerate that taint is filtered out. The fix is to add the matching toleration, not to remove the taint—removing the taint lets unrelated workloads consume GPU nodes.

```yaml
tolerations:
- key: "sku"
  operator: "Equal"
  value: "gpu"
  effect: "NoSchedule"
```

For affinity rules, prefer `preferredDuringSchedulingIgnoredDuringExecution` over the required form unless the placement is genuinely mandatory. A required rule turns a soft preference into a hard scheduling failure the moment the target pool is unavailable.

### A worked example

Suppose a deployment requests one GPU and carries this affinity:

```yaml
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
      - matchExpressions:
        - key: eks.amazonaws.com/nodegroup
          operator: In
          values: ["gpu-pool-a"]
```

The `gpu-pool-a` node group is scaled to zero overnight and back up in the morning. While it is at zero, every replica is unschedulable. The event reads "node(s) didn't match node affinity," which is accurate but does not say the pool is empty.

Two changes resolve it. First, relax the rule to the preferred form so the pod can land elsewhere if the pool is unavailable. Second, if the pool must be used, ensure the cluster autoscaler is configured to scale it from zero—a node group with a minimum size of zero needs the autoscaler to recognize the pending pod as a scaling trigger, which requires the pod's resource requests to be representable by the node group's instance types.

## Cause 3: namespace quota and LimitRange objects

A `ResourceQuota` rejects pods at admission. The pod is never created, or it is created and immediately fails, depending on the quota type. Either way, no scheduling occurs.

Check the quota and its live usage:

```sh
kubectl get resourcequota -n <namespace> -o yaml
kubectl describe resourcequota -n <namespace>
```

The `Status` block shows `used` and `hard` for each constrained resource. If `used` equals `hard` for the GPU resource, new pods cannot be admitted until something is deleted or the quota is raised.

A `LimitRange` is the quieter problem. A `LimitRange` can set a default request for a resource, or enforce a maximum. If a `LimitRange` sets a default GPU request of zero, or caps GPU requests below what the workload needs, the pod is rejected or silently modified. Inspect it:

```sh
kubectl get limitrange -n <namespace> -o yaml
```

Quota changes propagate through the quota controller, so allow a short interval before retrying. If a pod is rejected for quota and the quota has since been raised, delete and recreate the pod rather than waiting for the existing one to succeed.

## Cause 4: capacity, autoscaling, and instance availability

When the events say insufficient capacity and the resource name and selectors are correct, the cluster genuinely has no free GPUs and no way to add them. Two things can be true here, and they need different responses:

- **The autoscaler cannot add nodes.** The node group may be at its maximum size, the account may lack quota for the instance family, or the instance type may be unavailable in the availability zone.
- **The autoscaler can add nodes but has not yet.** GPU nodes take longer to join than CPU nodes because the driver and device plugin must initialize. A pod that waits several minutes is not necessarily broken.

Check the autoscaler's view of the pending pod:

```sh
kubectl logs -n kube-system deployment/cluster-autoscaler --tail=200 | grep -i "gpu\|scale"
```

Look for messages naming the node group, the reason it was not scaled, and any quota or instance-availability error. If the autoscaler reports that the node group is at maximum size, the fix is a quota increase or a larger maximum, not a manifest change.

To distinguish slow scaling from blocked scaling, watch node creation directly:

```sh
kubectl get nodes -w
```

If a new node appears and the pod schedules shortly after, the delay was provisioning. If no node appears and the autoscaler logs a refusal, the delay is a hard limit.

## How to measure whether GPU is even the right choice

Before investing in GPU scheduling fixes, confirm that the GPU is actually faster for the workload. The measurement is straightforward and does not require a benchmark suite.

1. Run the job on a CPU node and record wall-clock duration and node cost per hour.
2. Run the same job on a GPU node and record the same two numbers.
3. Compute cost per run as `duration_seconds / 3600 * hourly_rate` for each.
4. Compute the speedup as `cpu_duration / gpu_duration`.

The GPU is cheaper only when `gpu_cost_per_run < cpu_cost_per_run`. Because GPU instance rates are higher, this requires the speedup to exceed the ratio of the two hourly rates. As an illustrative example, if a GPU node costs four times as much per hour as a CPU node, the GPU must be more than four times faster for the cost per run to fall. If it is only twice as fast, the CPU run is cheaper despite taking longer.

The crossover also depends on whether the job is dominated by startup or by compute. A job that spends most of its time pulling an image, installing dependencies, or loading a model will not benefit much from a faster processor, and the GPU premium is wasted. Instrument the job with timestamps around each phase to see where the time goes before assuming the accelerator is the bottleneck.

For scheduling specifically, the metric worth tracking is time from pod creation to `Running`. Export it from the cluster and alert when the p95 exceeds a threshold you choose. A rising p95 is the earliest signal that quota, capacity, or a device plugin regression is starting to bite.

## Prevention checklist

- **Pin the device plugin version** in your manifest repository and update it deliberately. An unpinned DaemonSet can change the advertised resource name or fail to support a new instance type after an automatic upgrade.
- **Assert the resource name in CI.** A repository-wide grep for the expected GPU resource key catches manifests that still use a name the cluster no longer advertises:

```sh
if grep -rn "nvidia.com/gpu" . --include="*.yaml" --include="*.yml"; then
  echo "Found a GPU resource key; confirm it matches the device plugin's advertised name"
  exit 1
fi
```

Adjust the pattern to your environment's actual resource name. The point is to make the name explicit and reviewed rather than inherited.

- **Check quota before the pipeline starts.** A short preflight step that reads the namespace quota and fails early gives a clear error instead of a timeout.
- **Keep a smoke job in the repository.** When scheduling breaks, the smoke job tells you in under a minute whether the problem is infrastructure or application.
- **Alert on scheduling latency, not just failures.** A pod that eventually runs is still a problem if it waited twenty minutes.

## FAQ

**Why does the same manifest work in one cluster and not another?**
The advertised GPU resource name, node taints, and quota objects are cluster-scoped. Two clusters running the same workload can differ in all three. Compare `kubectl get node <node> -o jsonpath='{.status.capacity}'` and the namespace quota between the two clusters before changing the manifest.

**The pod is Pending with no events. What does that mean?**
No events usually means the scheduler has not attempted to place the pod, or the events have aged out of the API server's retention window. Check the scheduler's own logs and any admission webhooks in the path. Also confirm the pod is not blocked by a `PodDisruptionBudget` or a `PriorityClass` preemption that is waiting on a higher-priority pod.

**How do I set GPU memory limits correctly?**
GPU memory is not a schedulable resource in Kubernetes. `resources.limits.memory` constrains container RAM, not device memory. For device memory, the relevant controls live in the framework or runtime—for example, a PyTorch allocator configuration environment variable, or a framework-specific memory fraction setting. Right-size container RAM by watching actual usage during a representative run and setting the limit above the observed peak with headroom, then verify with a load test.

**Should I request GPUs with `limits` only, or both `requests` and `limits`?**
For extended resources, Kubernetes requires `requests` and `limits` to be equal if both are set, and a `limits`-only entry is treated as the request. Setting only `limits` is the common and valid form.

**What is the fastest way to tell whether a stall is quota or capacity?**
Read the event reason. A quota rejection names the quota object. An insufficient-capacity event names the resource. If the resource name is correct and a node advertises free capacity of that resource, the problem is eligibility—selectors, affinity, or taints—not capacity.

## Do this in the next 30 minutes

Pick the namespace that runs your GPU jobs and run:

```sh
kubectl describe resourcequota -n <namespace>
kubectl get events -n <namespace> --field-selector reason=FailedScheduling \
  --sort-by=.lastTimestamp | tail -20
```

If the quota shows zero or missing GPU capacity, or the events show a resource name that does not match what your nodes advertise, you have found the cause. Apply the minimal smoke job from this article with the correct resource name and confirm it reaches `Running`. That single check separates a scheduling problem from an application problem before the next pipeline run.
