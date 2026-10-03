# Kubernetes or Nomad: the real experiment cost split

## Why the scheduler choice shows up in your GPU bill

Two clusters can run the same training image on the same GPU model and produce very different cost-per-experiment numbers. The difference is rarely the GPU itself. It is how long a GPU sits idle between the moment a job is submitted and the moment it starts burning FLOPs, plus how much control-plane and networking overhead you pay to keep the cluster alive.

This article compares two common ways to schedule GPU training workloads:

- **Kubernetes**, using the device plugin framework to advertise GPUs as schedulable resources.
- **Nomad**, which models GPUs as a device resource on the client and binds them directly to a task.

The goal is not to declare a universal winner. It is to give you the instrumentation, the arithmetic, and the failure modes you need to decide for your own workload profile.

## Option A: Kubernetes with GPU device plugins

Kubernetes is the default choice in many AI platforms because it gives you three things that are hard to replicate elsewhere: fine-grained GPU sharing, pod-level autoscaling, and a large ecosystem of Helm charts, operators, and ingress controllers.

Under the hood, GPU support comes from the **device plugin framework**. A vendor-supplied daemon runs on each node, discovers the GPUs, and registers them with the kubelet as an extended resource such as `nvidia.com/gpu`. Pods then request that resource exactly like CPU or memory:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: pytorch-mnist
spec:
  containers:
  - name: pytorch
    image: pytorch/pytorch:2.3.0-cuda12.1
    command: ["python", "train.py"]
    resources:
      limits:
        nvidia.com/gpu: 1
        cpu: "8"
        memory: "32Gi"
      requests:
        nvidia.com/gpu: 1
        cpu: "8"
        memory: "32Gi"
  nodeSelector:
    accelerator: "nvidia-tesla-a100"
```

The scheduler places the pod on a node that reports an available GPU, the kubelet mounts the device, and the container starts with the GPU visible.

What this buys you:

- **GPU sharing.** Multi-Instance GPU (MIG) partitions a physical GPU into isolated slices. A device plugin can advertise those slices as separate resources, so multiple pods share one card with hardware-level isolation.
- **Pod-level autoscaling.** The Horizontal Pod Autoscaler scales replicas from metrics; KEDA extends this to event-driven scaling. Both are useful for inference services.
- **Fast resource release.** When a pod exits, its GPU is returned to the pool immediately. You do not have to drain a whole node.

The cost is scheduling and setup latency. A GPU pod typically passes through more steps than a plain CPU pod: the scheduler must find a node whose device plugin reports a free GPU, the kubelet must set up the device, and the container runtime must mount it. On clusters that span availability zones or run several CNI and admission layers, that chain can add seconds to every job start. For a long training run this is noise. For a sweep of thousands of short jobs, it is the dominant cost.

### A failure mode worth knowing

A common symptom is a pod stuck in `ContainerCreating` while GPUs sit idle. The usual causes:

1. The image's CUDA user-space libraries do not match the node driver. The container runtime reports a driver initialization error, and the pod never starts.
2. The device plugin daemon is unhealthy on one node, so that node advertises zero GPUs even though the hardware is present.
3. A node selector or taint excludes every node that actually has a free GPU.

The diagnostic sequence is: `kubectl describe pod <name>` for events, then the kubelet logs on the target node, then the device plugin pod's logs in its namespace. Budget time for this; it is the tax you pay for the abstraction.

## Option B: Nomad device scheduling

Nomad takes a different approach. GPUs are declared as a device resource in the job spec, and the Nomad client on the target node binds the device directly to the task. There is no separate device plugin daemon and no scheduler-to-kubelet handshake for the device itself.

```hcl
job "pytorch-mnist" {
  datacenters = ["dc1"]
  type        = "batch"

  group "train" {
    count = 1

    task "pytorch" {
      driver = "docker"

      config {
        image   = "pytorch/pytorch:2.3.0-cuda12.1"
        command = "python train.py"
      }

      resources {
        cpu    = 8
        memory = 32000

        device "nvidia/gpu" {
          count = 1
        }
      }
    }
  }
}
```

Because device binding is local to the client, task start latency is typically lower and more predictable than in a Kubernetes cluster with a layered control plane. That matters most for short jobs.

What Nomad does **not** give you out of the box:

- **GPU sharing.** There is no native MIG partitioning. You can approximate sharing by setting `CUDA_VISIBLE_DEVICES` to a MIG slice from a wrapper script, but you own the isolation, the accounting, and the failure handling.
- **Pod-level autoscaling.** Nomad scales job counts, not individual pods driven by arbitrary metrics. Event-driven autoscaling requires custom work.
- **Namespace-grade isolation.** Nomad has ACLs and namespaces, but the isolation boundary is coarser than Kubernetes namespaces with network policies.

## How to measure this yourself instead of trusting a table

Published benchmark tables are almost never reproducible on your hardware. Measure your own cluster. Here is what to instrument.

**Job start latency.** Timestamp the moment the job is submitted and the moment the training process logs its first step. On Kubernetes, compare `kubectl get pod -w` creation time against the first application log line. On Nomad, compare the `Submitted` and `Started` events in `nomad job status`. Record the median and the 95th and 99th percentiles, not the mean. Tail latency is what starves a sweep.

**GPU idle time.** Scrape GPU utilization at a fixed interval (for example, once per second) and compute the fraction of samples below a threshold while a job is queued or starting. This is the number that converts directly into wasted money.

**Cost per experiment.** Define it explicitly:

```
cost_per_experiment =
    (job_wall_clock_seconds / 3600) * price_per_node_hour / gpus_per_node
  + (queue_and_start_seconds / 3600) * price_per_node_hour / gpus_per_node
```

The first term is useful work; the second is the latency tax. If the second term is a large fraction of the first, your scheduler is the problem, not your model.

**Cluster idle cost.** Sum the hourly price of every node that is running but not executing a job. Divide by the number of experiments completed in the same window. This exposes over-provisioning that a per-job metric hides.

### A worked example

Assume a cluster of 8 nodes, each with 4 GPUs, at an illustrative on-demand price of $3.00 per node-hour. That is $24.00 per hour for 32 GPUs, or $0.75 per GPU-hour.

Now consider a sweep of 1,000 jobs, each requesting one full GPU and running for 10 minutes of actual compute.

Useful GPU time per job: 10 minutes = 0.1667 GPU-hours.
Useful cost per job: 0.1667 × $0.75 = **$0.125**.

Add start latency. Suppose the median start-to-first-step time is 20 seconds on one scheduler and 3 seconds on the other.

- 20 seconds = 0.00556 GPU-hours → 0.00556 × $0.75 = **$0.0042** per job.
- 3 seconds = 0.00083 GPU-hours → 0.00083 × $0.75 = **$0.00063** per job.

Over 1,000 jobs the difference is $4.17 minus $0.63, or about **$3.54**. That is small against $125 of useful compute — a 2.8% overhead.

Now change the workload: jobs that run for 30 seconds of compute instead of 10 minutes.

Useful cost per job: (30/3600) × $0.75 = **$0.00625**.
Latency cost at 20 seconds: **$0.0042** — that is 67% of the useful cost.
Latency cost at 3 seconds: **$0.00063** — about 10%.

The lesson is arithmetic, not vendor preference: **the shorter your jobs, the more the scheduler's start latency dominates your bill.** Long training runs can absorb almost any scheduling overhead. Short sweeps cannot.

## Head-to-head comparison

| Dimension | Kubernetes with GPU device plugin | Nomad device scheduling |
|---|---|---|
| GPU advertised as | Extended resource via device plugin daemon | Device resource on the client |
| Native GPU sharing | Yes, via MIG slices | No, requires a custom wrapper |
| Pod/task-level autoscaling | Yes, HPA and KEDA | Job-count scaling; custom work for metrics-driven |
| Multi-tenant isolation | Namespaces, network policies, RBAC | ACLs and namespaces; coarser boundary |
| Typical start latency | Higher, more moving parts | Lower, local device binding |
| Ecosystem | Large: Helm, operators, ingress, workflow engines | Smaller: single binary, fewer integrations |
| Operational surface | etcd, control plane, CNI, CSI, observability stack | Single binary plus clients |
| Best fit | Sharing, isolation, inference autoscaling | Full-GPU batch training, short jobs, simple ops |

## Operational cost beyond the GPU

Hardware is only part of the bill. When you compare the two, include:

- **Control plane.** Kubernetes runs etcd and API server components that consume nodes or managed-service fees. Nomad's server is a single binary.
- **Networking.** A CNI plugin adds configuration and, depending on the plugin, per-packet overhead. Nomad defaults to host networking.
- **Storage.** CSI drivers and distributed storage systems add components to operate. Nomad can use host volumes for small datasets.
- **Observability.** Kubernetes deployments commonly add a metrics stack and dashboards. Nomad exposes metrics and a built-in UI.
- **Engineering time.** This is the largest and least visible line item. Track it as tickets resolved per month and hours per incident, not as a feeling.

The only scenario where Kubernetes reliably saves money on GPU spend is **sharing**. If four small inference workloads can share one GPU instead of occupying four, you cut GPU hours substantially. For training jobs that need a full GPU, sharing does not apply, and the extra platform cost buys you nothing.

## A decision checklist

Work through these in order. Stop at the first "yes" that forces a choice.

1. **Do you need GPU sharing or MIG partitioning?** Yes → Kubernetes. No → continue.
2. **Do you need namespace-grade multi-tenant isolation?** Yes → Kubernetes. No → continue.
3. **Do your jobs run for less than a few minutes each?** Yes → Nomad, because start latency dominates. No → continue.
4. **Do you already operate Kubernetes for other workloads?** Yes → Kubernetes, to avoid a second platform. No → continue.
5. **Is minimizing platform operational surface your priority?** Yes → Nomad. No → Kubernetes if you expect to need its ecosystem later.

The most expensive mistake is choosing Kubernetes for short batch jobs without measuring start latency first, then discovering that the scheduler overhead is a large fraction of each job. The second most expensive is choosing Nomad for a workload that will need GPU sharing in six months, then building that sharing by hand.

## FAQ

**Why is Nomad typically faster at starting GPU jobs?**
Nomad's client binds the device to the task locally. Kubernetes routes the placement decision through the scheduler and then has the kubelet set up the device, which adds steps to every job start.

**How do I enable GPU sharing on Nomad?**
There is no native MIG integration. You expose MIG slices and set `CUDA_VISIBLE_DEVICES` in a wrapper. You own the isolation, accounting, and error handling. Kubernetes device plugins handle this more directly.

**How do I compare cost fairly between the two?**
Measure cost per completed experiment, including queue and start time, not cost per GPU-hour. Use the formula in the measurement section and run it on your own hardware with your own job durations.

**When should I avoid Nomad for AI workloads?**
When you need GPU sharing, namespace-grade isolation, or metrics-driven autoscaling of inference services. Nomad fits full-GPU batch training with simple operations.

## Do this in the next 30 minutes

Pick one representative training job. Record its submit timestamp and the timestamp of its first training step, then compute the start latency. Run it ten times and take the median. Multiply that median by your GPU-hour price and by the number of jobs in a typical sweep. If that product is more than roughly 10% of your useful compute cost, your scheduler latency is worth fixing before you change anything else.
