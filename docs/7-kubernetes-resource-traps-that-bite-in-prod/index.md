# Kubernetes Requests and Limits: Seven Production Traps

Kubernetes resource requests and limits look like two simple numbers per container. In practice they encode scheduling, eviction, CPU throttling, memory OOM kills, and the difference between a rollout that finishes and one that wedges. The failure mode is rarely a crash on apply. It is a pod that schedules onto a node with no headroom, a Java process throttled to a crawl because its limit is half a core, or a node that starts evicting a payment service because a batch job's request was set to a token value.

The concrete problem: teams set requests and limits once, never revisit them, and then spend days debugging p99 latency and OOMKilled pods that trace back to a single mis-sized number in a Deployment manifest. This article covers seven traps that surface when a workload moves from a laptop to production under load, and gives a detection method for each.

## Why requests and limits are two different contracts

A **request** is what the scheduler uses to decide which node has room. It is a promise to the kubelet that this container will get at least that much. A **limit** is what the kernel enforces at runtime: exceed the CPU limit and the container is throttled; exceed the memory limit and it is OOM-killed.

They are not symmetric. CPU is compressible — throttling slows a workload down but does not kill it. Memory is incompressible — going over the limit terminates the process. That asymmetry drives most of the traps below.

If a request is set but no limit, the container can burst to use idle CPU on the node, and memory is unbounded until the node itself runs out and the eviction manager starts killing pods by QoS class. If a limit is set but no request, the scheduler assumes zero and will pack the pod onto a node that is already full — then throttle it to the limit the moment it tries to work.

The Kubernetes documentation on managing resources for containers is the canonical reference for the mechanics; the traps below are about the judgement calls the mechanics leave open.

## The seven traps

### 1. The CPU limit that turns a fast service into a slow one

**What it does:** A CPU limit caps how many CPU-seconds a container can consume per period (typically 100ms on the CFS quota). Hit the cap and the kernel throttles the container until the next period.

**Strength:** Prevents a runaway process — a hot loop, a badly tuned GC — from starving every other pod on the node.

**Weakness:** Throttling is invisible in most dashboards unless `container_cpu_cfs_throttled_seconds_total` is being watched. A service with a 500m limit and a bursty request pattern will show p99 latency spikes that look like network problems but are actually CFS quota exhaustion. Go and Java services with large thread pools are especially prone because they wake many threads at once, and a single 100ms period can be exhausted before the work completes.

**Best for:** Noisy-neighbour protection on shared nodes where the workload genuinely cannot be trusted. For latency-sensitive services, many teams drop the CPU limit entirely and rely on requests plus node autoscaling. That is a defensible default — but only if requests are set high enough that the scheduler does not overpack nodes.

**Check:** Query `rate(container_cpu_cfs_throttled_seconds_total[5m])` per pod. If it is non-zero for a service that is not supposed to be CPU-bound, the limit is too tight or the request is too low.

### 2. The memory limit set from an idle observation

**What it does:** Memory limits are enforced by the cgroup OOM killer. Exceed them and the kernel sends SIGKILL to the container's main process. Kubernetes reports `OOMKilled` in the pod status.

**Strength:** Hard ceiling. One leaking pod cannot take down the node.

**Weakness:** Peak memory in a container is not the same as steady-state memory. A Go service that holds a large in-memory cache, a JVM with a heap sized as a fraction of the limit, or a Python process that spikes during a batch import will all exceed a limit sized from a quiet-period observation. Worse, the JVM historically reads the cgroup limit to size its default heap — but only if the runtime is new enough to be cgroup-aware. Older JVMs size the heap from the node's total memory, blow past the container limit, and get OOM-killed on startup.

**Best for:** Every container. But size the limit from an observed peak under realistic load, not from an idle `kubectl top`. Give headroom — a limit equal to the observed peak is a limit that will eventually be hit.

**Check:** `kubectl describe pod <name>` and look for `Last State: Terminated, Reason: OOMKilled`. If it is there, the limit is the problem, not the node.

### 3. Requests so low the scheduler overpacks the node

**What it does:** The scheduler sums requests across all pods on a node and refuses to place a pod if the sum would exceed allocatable capacity. Requests are the scheduler's only signal.

**Strength:** Predictable placement. If every pod declares honest requests, no node is oversubscribed and the eviction manager has little to do.

**Weakness:** Teams often set requests to a token value — 10m CPU, 32Mi memory — because "it schedules fine." It does schedule fine. Then the node fills with pods that each want ten times their request, the kubelet's eviction manager starts ranking pods by QoS class, and a BestEffort batch job gets killed while a Guaranteed database pod survives. The batch job's owner sees a mysterious eviction and blames the platform team.

A worked example makes the arithmetic concrete. Suppose a node has 4 CPU and 16Gi allocatable, and it runs eight pods that each declare 100m CPU and 256Mi memory but actually consume 400m and 1Gi at peak. Declared requests total 800m CPU and 2Gi memory — comfortably inside the node. Actual usage is 3.2 CPU and 8Gi. The node looks 20% scheduled and is running at 80% CPU. Any ninth pod scheduled onto it will be squeezed, and a memory spike in one pod will trigger eviction. The scheduler did nothing wrong; it was told the wrong numbers.

**Best for:** All workloads. The request should approximate what the container needs to run well, not the minimum that lets it start.

**Check:** Compare `sum(kube_pod_container_resource_requests)` per node against `kube_node_status_allocatable`. If requests are far below actual usage, there is a scheduling debt that will surface as evictions.

### 4. The QoS class nobody chose

**What it does:** Kubernetes assigns each pod a QoS class based on how requests and limits are set: **Guaranteed** (requests equal limits for every container and every resource), **Burstable** (some requests set, not equal to limits), **BestEffort** (no requests or limits at all).

**Strength:** QoS class determines eviction order under node pressure. Guaranteed pods are evicted last; BestEffort pods first.

**Weakness:** QoS class is a side effect of how the manifest was written, not a field that is set. A pod that sets only limits is Burstable; a pod that sets nothing is BestEffort. A critical payment service with no requests is BestEffort and will be the first thing killed when a node runs low on memory — even though it is the most important workload on the node.

**Best for:** Anything that matters should be at least Burstable with honest requests; anything that cannot be lost should be Guaranteed. The cost of Guaranteed is that the pod cannot burst, so reserve it for services with predictable load.

**Check:** `kubectl get pod <name> -o jsonpath='{.status.qosClass}'`. Audit the production namespace and count how many critical services are BestEffort.

### 5. The LimitRange default that silently rewrites the manifest

**What it does:** A `LimitRange` in a namespace injects default requests and limits into pods that do not specify them, and can enforce min/max bounds.

**Strength:** Prevents the BestEffort-by-accident problem. A cluster admin can set sane defaults so a developer who forgets to size a container still gets something reasonable.

**Weakness:** The default is invisible in the Deployment manifest. A developer reads the YAML, sees no resources block, and assumes the pod is unbounded. It is not — it is whatever the LimitRange says, which might be 100m CPU and 128Mi memory. The pod gets throttled or OOM-killed and the manifest offers no clue. This is especially painful when workloads move between namespaces with different LimitRanges.

**Best for:** Platform teams enforcing a floor. But document the defaults and surface them in the developer portal or a `kubectl get limitrange -o yaml` snippet in onboarding docs.

**Check:** `kubectl get limitrange -n <namespace> -o yaml`. If the contents are unknown, the actual runtime resources of the pods are unknown too.

### 6. The HPA that scales on CPU when CPU is capped

**What it does:** The HorizontalPodAutoscaler scales replicas based on a metric — commonly CPU utilisation as a percentage of request.

**Strength:** Automatic capacity for variable load. Works well when the metric tracks the actual bottleneck.

**Weakness:** HPA compares observed CPU against the **request**, not the limit. With a CPU limit but a low request, the HPA sees high utilisation against the request and scales out aggressively — even though each pod is throttled at the limit and adding replicas may not help if the bottleneck is inside each pod. Conversely, with a high request and a low limit, the HPA may never scale because utilisation against the request stays low while every pod is throttled. The HPA and the limit are fighting each other.

**Best for:** Services where CPU is genuinely the bottleneck and requests are honest. For I/O-bound services, scale on a custom metric (queue depth, in-flight requests) instead of CPU.

**Check:** `kubectl describe hpa <name>` and look at the current vs target utilisation. Cross-reference with the throttling metric from trap 1. If both are high, the request/limit ratio is wrong.

### 7. The sidecar with no resources at all

**What it does:** Service meshes, log shippers, and metric exporters run as sidecars or DaemonSets alongside the application container.

**Strength:** They provide observability, mTLS, and traffic policy without changing application code.

**Weakness:** Sidecars are containers. They consume CPU and memory. A service mesh sidecar that proxies every request can add measurable latency and memory per pod. If the sidecar has no requests, the scheduler does not account for it; if it has no limits, a sidecar leak can OOM the pod. Teams commonly size the application container carefully and leave the sidecar at defaults, then wonder why the pod's total memory footprint is higher than the sum of its parts.

**Best for:** Every sidecar should have requests and limits set, even if the mesh's injection template does not do it by default. Check the injector configuration.

**Check:** `kubectl get pod <name> -o jsonpath='{.spec.containers[*].name}'` to list containers, then inspect each one's resources. An empty resources block on any container is a gap.

## How to prioritise which trap to fix first

Not all seven matter equally. A practical order, based on blast radius:

1. **OOMKilled pods in production.** Memory limits that are too low cause hard outages. Fix these first. Look for `OOMKilled` in pod status across the production namespace.
2. **BestEffort workloads that matter.** Any service in the critical path with no requests is one node-pressure event away from eviction. Set requests at minimum.
3. **CPU throttling on latency-sensitive services.** This shows up as p99 latency that does not correlate with traffic. Check the throttling metric before blaming the network.
4. **HPA and limit mismatch.** If autoscaling is not helping under load, the request/limit ratio is the likely cause.
5. **LimitRange defaults that are undocumented.** Lower urgency, but the root cause of many "why is this pod throttled" tickets.
6. **Sidecar resources.** Important for cost and stability, but usually a smaller blast radius than the application container.
7. **Requests set too low across the board.** A scheduling hygiene issue that compounds over time. Fix it with a policy (admission controller or LimitRange) rather than a one-off audit.

The principle: fix what causes outages first, then what causes latency, then what causes confusion.

## A minimal manifest that avoids all seven

Here is a Deployment fragment that sets requests equal to limits for a latency-sensitive service (Guaranteed QoS), includes a sidecar with its own resources, and leaves room for the HPA to scale on a request that reflects reality. The numbers are illustrative — replace them with values derived from your own load tests.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: payments-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: payments-api
  template:
    metadata:
      labels:
        app: payments-api
    spec:
      containers:
        - name: api
          image: registry.example.com/payments-api:1.4.2
          resources:
            requests:
              cpu: "500m"
              memory: "512Mi"
            limits:
              cpu: "500m"
              memory: "512Mi"
          ports:
            - containerPort: 8080
        - name: log-shipper
          image: registry.example.com/log-shipper:2.1.0
          resources:
            requests:
              cpu: "50m"
              memory: "64Mi"
            limits:
              cpu: "100m"
              memory: "128Mi"
```

Two things to notice. First, the application container has requests equal to limits, so the pod is Guaranteed and will be evicted last. Second, the sidecar has its own resources — it is not invisible to the scheduler. If the service is bursty and Burstable QoS is preferable, drop the CPU limit on the application container but keep the request honest; the memory limit should stay to prevent node-level OOM.

For a batch job that can tolerate eviction, BestEffort is sometimes acceptable — but only if it is labelled as such and the team owning the node knows it. The trap is accidental BestEffort, not deliberate.

## How to measure whether a change helped

Resizing resources is an experiment, and it needs a before-and-after comparison rather than a feeling that things got faster. Three signals cover most cases:

- **Throttling.** Compare `rate(container_cpu_cfs_throttled_seconds_total[5m])` per pod over a window that includes a representative peak, before and after the change. The target is zero for latency-sensitive services, or a small, steady value if the workload is deliberately capped.
- **OOM kills.** Count `kube_pod_container_status_last_terminated_reason{reason="OOMKilled"}` per pod over the same window. This should go to zero after a limit increase; if it does not, the memory growth is a leak, not a sizing problem, and a bigger limit only delays the failure.
- **Latency against traffic.** Plot p99 latency against request rate. If throttling was the cause, the latency curve should flatten at the traffic level where it previously spiked. If latency still climbs with traffic but throttling is zero, the bottleneck is downstream and the resource change was not the fix.

Run each change against one workload at a time. Changing requests, limits, and HPA targets together makes it impossible to attribute the result.

## Frequently Asked Questions

### What happens if I set a CPU limit but no CPU request?

The scheduler treats the request as zero, so it will place the pod on any node with free capacity — including one that is already heavily loaded. At runtime the container is capped at the limit, but the scheduler had no way to reserve capacity for it. The pod may be throttled immediately under load, and the node may be oversubscribed. Always set a request; the limit is optional depending on latency tolerance.

### Why does my pod get OOMKilled when memory usage looks fine in kubectl top?

`kubectl top` shows a point-in-time sample, not the peak. A container can spike between samples — during a GC pause, a large request, or a batch import — and hit the cgroup limit before the next sample. The kernel kills the process immediately. Check `kubectl describe pod` for `Last State: Terminated, Reason: OOMKilled` and compare the limit against the observed peak under load, not against the average.

### Should I always set requests equal to limits?

No. Equal requests and limits give Guaranteed QoS, which means no bursting and last-priority eviction. That is right for services with predictable load that cannot be lost. For services with spiky traffic, a lower request and a higher limit (Burstable) lets the pod use idle node capacity while still reserving a floor. The wrong answer is leaving both unset and getting BestEffort by accident.

### How do I know if my CPU limit is causing latency?

Look at `container_cpu_cfs_throttled_seconds_total` for the pod. If it is increasing while latency spikes, the limit is the cause. A second signal is that latency improves when the CPU limit is removed but the request is kept — that confirms the limit was the bottleneck, not the request.

## The next 30 minutes

Run this against the production namespace and look for containers with no requests or no limits:

```bash
kubectl get pods -n production -o json | \
  jq -r '.items[] | .metadata.name as $n | .spec.containers[] | select(.resources.requests == null or .resources.limits == null) | "\($n) / \(.name)"'
```

Every line in the output is a container that is either BestEffort (no requests) or unbounded (no limits). Pick the one in the most critical path, check its actual usage with `kubectl top pod`, and set a request based on that number. That single change is the highest-leverage fix available today.
