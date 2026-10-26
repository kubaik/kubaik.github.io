# ARM in production: Graviton4 vs Ampere in 2026

Most ARM migration tutorials stop at "switch the base image." That step is rarely where migrations fail. The failures tend to cluster around timing, caching, and observability: image pull deadlines, NUMA placement on high-core-count parts, and probing behaviour that was tuned for a different core count. This article is a checklist for the parts that come after the base image.

The ARM64 vs x86 decision is no longer primarily a price question. AWS Graviton4 and Ampere Altra (used by Oracle Cloud and several European providers) are both mature targets, but neither is a drop-in replacement for an existing x86 deployment. Vendor performance claims are workload-specific, and the only reliable way to decide is to measure your own service. This guide covers two representative paths: Graviton4 on EKS for most teams, and Ampere Altra for high-core-count workloads.

## What you will build and measure

Two target paths:

1. AWS Graviton4 (arm64) on EKS, Node 20 LTS and Python 3.12
2. Ampere Altra A1 instances (arm64) on Oracle Cloud, Terraform and Redis 7.2

Assumptions: a service behind an ALB/NLB at roughly 50–500 QPS, state in a managed database or cache, and a staging environment that mirrors production. If staging does not exist, build a disposable one first. Migrating directly into production is the most common cause of avoidable incidents.

What to measure, and how:

- Image pull latency: instrument with `kubectl describe pod` events (`Pulling`, `Pulled` timestamps) or scrape `kubelet_image_puller_duration_seconds` from the kubelet metrics endpoint. Compare arm64 and amd64 images of the same service.
- Cold start latency: for Lambda, use the `Init Duration` field in CloudWatch Logs for both architectures; for Fargate, use the task start-to-ready time from ECS events.
- CPU efficiency: run a fixed benchmark inside both instance types and record throughput per vCPU and per dollar. Do not trust vendor tables; run your own binary.
- Error rate under load: compare p50, p95, p99 and error rate before and after the switch at the same request rate.

You will need: an AWS account with enough budget to run several nodes for a few hours, `kubectl` v1.29+, `helm` 3.14+, `eksctl` (current release), Python 3.12, `pytest` 8.1+, and Locust 2.24. For Ampere on Oracle Cloud, Terraform 1.8+.

The same approach applies on GCP or Azure: substitute the instance family and the node bootstrap mechanism, but keep the measurement plan identical.

## Step 1 — set up a disposable staging cluster

Create a fresh staging cluster on Graviton4 before touching production. The first cluster you migrate should be one you can delete without consequence.

1. Install `eksctl` and `kubectl`. Use the current release rather than pinning to a specific patch, since ARM AMI and instance-type support is added over time.

```bash
curl --silent --location "https://github.com/eksctl-io/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
sudo mv /tmp/eksctl /usr/local/bin
curl -LO "https://dl.k8s.io/release/v1.29.0/bin/linux/amd64/kubectl"
chmod +x kubectl && sudo mv kubectl /usr/local/bin
```

2. Create the cluster. `m7g.2xlarge` (8 vCPU, 32 GB) is a reasonable starting size for reproducing NUMA and scheduling effects without a large bill.

```bash
eksctl create cluster \
  --name arm-migration-staging \
  --region us-east-1 \
  --version 1.29 \
  --nodegroup-name g4g20xlarge \
  --nodes 3 \
  --nodes-min 1 \
  --nodes-max 10 \
  --instance-types m7g.2xlarge
```

Note: `eksctl` selects the correct AMI based on the instance type. Do not pass an `--arm64` flag that does not exist in the CLI; verify the resulting nodes instead.

3. Point `kubectl` at the cluster and confirm the CNI is running.

```bash
eksctl utils write-kubeconfig --cluster arm-migration-staging
aws eks update-kubeconfig --name arm-migration-staging --region us-east-1
kubectl get nodes -o wide
```

You should see nodes whose `INSTANCE-TYPE` begins with `m7g` and whose `KUBELET-VERSION` matches the cluster version. If any node is `NotReady`, check its AMI:

```bash
aws ec2 describe-instances \
  --instance-ids $(kubectl get nodes -o jsonpath='{.items[*].spec.providerID}' | sed 's/.*\(i-[a-f0-9]*\))/\1/') \
  --query 'Reservations[*].Instances[*].ImageId' --output text
```

The AMI ID should correspond to an arm64 EKS-optimised image. If it does not, the nodegroup was created with the wrong instance type or an outdated `eksctl`.

4. Install a node autoscaler. A managed node group is sufficient for a first migration; if you use a cluster autoscaler or a node-provisioning controller, configure it to select arm64 instance families only, so that x86 and arm64 nodes are not mixed in a single workload's node pool.

```yaml
apiVersion: karpenter.sh/v1
kind: NodePool
metadata:
  name: default-arm
spec:
  template:
    spec:
      requirements:
        - key: kubernetes.io/arch
          operator: In
          values: [arm64]
        - key: karpenter.sh/capacity-type
          operator: In
          values: [on-demand]
      nodeClassRef:
        name: default
  limits:
    cpu: 1000
  disruption:
    consolidationPolicy: WhenEmpty
    consolidateAfter: 30s
```

The exact API version and fields depend on the provisioner you run; check its documentation for the current schema. The important part is the `kubernetes.io/arch: arm64` requirement, which prevents mixed-architecture scheduling.

## Step 2 — migrate one stateless service

Start with a stateless API. Stateful services such as Redis or Postgres require additional planning around persistence and failover.

1. Build a multi-arch image.

```Dockerfile
# syntax=docker/dockerfile:1.5
FROM --platform=$BUILDPLATFORM python:3.12-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM python:3.12-slim
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
CMD ["gunicorn", "app:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker"]
```

Build and push explicitly for arm64:

```bash
docker buildx build --platform linux/arm64 -t yourrepo/api:1.2.0-arm --push .
```

The `--platform linux/arm64` flag is required. Without it, `buildx` produces an image for the builder's default architecture, which is often amd64 even when running on an ARM host. Confirm the result:

```bash
docker buildx imagetools inspect yourrepo/api:1.2.0-arm
```

The output should list `linux/arm64` as the platform.

2. Deploy with an explicit architecture constraint and a resource request you can justify.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api
  template:
    metadata:
      labels:
        app: api
    spec:
      nodeSelector:
        kubernetes.io/arch: arm64
      containers:
      - name: api
        image: yourrepo/api:1.2.0-arm
        resources:
          requests:
            cpu: "500m"
            memory: "512Mi"
          limits:
            cpu: "1000m"
            memory: "1024Mi"
        ports:
        - containerPort: 8000
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 2
          periodSeconds: 5
          timeoutSeconds: 2
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
          timeoutSeconds: 3
```

Do not copy x86 CPU requests across unchanged and assume they are correct. Measure the actual CPU usage of the service under load on both architectures, then set requests from that data. A common mistake is to leave requests unchanged and then interpret the resulting throttling as an ARM performance problem.

3. Roll out and watch the events.

```bash
kubectl apply -f deployment.yaml
kubectl rollout status deployment/api --timeout=300s
kubectl get events --sort-by=.lastTimestamp
```

If pods are killed during startup, check whether the image pull is exceeding the kubelet's `imagePullProgressDeadline` (see Step 3) before assuming a runtime problem.

4. Benchmark before and after. A minimal Locust file:

```python
from locust import HttpUser, task, between

class ApiUser(HttpUser):
    wait_time = between(0.5, 2.5)

    @task
    def get_items(self):
        self.client.get("/items")
```

```bash
locust -f locustfile.py --host http://<your-lb-dns> --users 1000 --spawn-rate 100
```

Record p50, p95, p99, error rate, and CPU usage for both architectures at the same request rate. If latency does not improve for a CPU-bound service, check whether your Python dependencies are shipping arm64 wheels; a source build that silently falls back to a slower path is a common cause.

## Step 3 — handle the failure modes that only appear under load

The failures below are the ones that tend to survive a happy-path tutorial.

1. Image pull deadline exceeded. Larger or less-cached images can exceed the kubelet's default `imagePullProgressDeadline`. The default is documented as 1 minute; some clusters run with a lower value. If pulls are timing out, raise it explicitly:

```yaml
kind: KubeletConfiguration
apiVersion: kubelet.config.k8s.io/v1beta1
imagePullProgressDeadline: 2m
```

Apply it via your node bootstrap mechanism (user data, launch template, or a managed node group configuration). Do not `kubectl apply` a `KubeletConfiguration` to a running cluster; it is not a cluster-scoped object. Verify the setting is active after a node replacement.

2. NUMA placement on high-core-count instances. On multi-socket or multi-NUMA parts such as Ampere Altra, a process whose threads migrate between NUMA nodes will see memory-access latency rise under load. The symptom is usually a p99 regression that does not appear at low utilisation. Measure it with `numactl --hardware` on the node and `numastat -p <pid>` for the process. Topology Manager and CPU Manager, configured via the kubelet, are the standard mechanisms for aligning pod CPU and memory to a single NUMA node. Configure them in the kubelet config and validate with a workload that pins a known number of CPUs:

```yaml
kind: KubeletConfiguration
apiVersion: kubelet.config.k8s.io/v1beta1
topologyManagerPolicy: single-numa-node
cpuManagerPolicy: static
```

These policies require a kubelet restart and a pod restart to take effect, and they only apply to pods in the Guaranteed QoS class (requests equal limits). Confirm with `kubectl get pod <name> -o jsonpath='{.status.qosClass}'`.

3. Thread-affinity assumptions. Code that calls `pthread_setaffinity_np` or relies on a specific core count may behave differently on a 48- or 80-core part. Run the test suite on the target hardware, not just in a container on your laptop:

```bash
python -m pytest tests/ -q
```

If you suspect a race that only appears on many cores, run the suite under ThreadSanitizer where the toolchain supports it, and increase the core count on the test host to reproduce.

4. Storage latency mistaken for CPU latency. A faster CPU does not make network-attached storage faster. If p99 rises after migration, check the volume type and IOPS before blaming the architecture. gp3 has a baseline of 3000 IOPS and 125 MiB/s independent of volume size; gp2 scales IOPS with volume size. Confirm the volume type in use:

```bash
aws ec2 describe-volumes --volume-ids <id> --query 'Volumes[*].VolumeType'
```

5. Lambda cold starts. arm64 Lambda functions use the same pricing model as x86, priced per GB-second. The relevant comparison is your function's measured duration and memory configuration on each architecture. Measure both:

```bash
aws lambda invoke --function-name my-arm-func --payload '{}' response-arm.json
aws lambda invoke --function-name my-x86-func --payload '{}' response-x86.json
```

Compare the `Init Duration` reported in CloudWatch Logs, and check that any layers you depend on are published for arm64. A layer built for x86 will fail to load or fall back to a slower path.

## Step 4 — observability and safe rollout

You cannot debug a latency regression you did not measure. Add these before declaring the migration complete.

1. Metrics. Install a Prometheus stack and scrape kubelet and node-exporter metrics. The metrics that matter for this migration are `kubelet_image_puller_duration_seconds`, `container_cpu_cfs_throttled_seconds_total`, `node_cpu_seconds_total` broken down by mode, and your application's own latency histograms. A per-NUMA-node view requires a node-level exporter that reports NUMA statistics; check that the exporter you choose actually emits them before relying on the dashboard.

2. Progressive rollout. Use a canary or progressive delivery controller to shift a percentage of traffic to the new architecture and roll back automatically on error-rate or latency thresholds. A minimal canary definition (the exact API version depends on the controller you use):

```yaml
apiVersion: flagger.app/v1beta1
kind: Canary
metadata:
  name: api-canary
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api
  service:
    port: 9898
  analysis:
    interval: 1m
    threshold: 5
    maxWeight: 50
    stepWeight: 10
    metrics:
    - name: request-success-rate
      thresholdRange:
        min: 99
      interval: 1m
    - name: request-duration
      thresholdRange:
        max: 500
      interval: 30s
```

The threshold values above are illustrative. Set them from your own baseline: a canary that rolls back at 500 ms p95 is useless if your service already runs at 480 ms.

3. CI check for the arm64 build. Fail the pipeline if the arm64 image does not build or does not pass tests:

```yaml
- name: Test ARM build
  run: |
    docker buildx build --platform linux/arm64 -t test-arm .
    docker run --rm test-arm python -m pytest tests/
```

4. Image size tracking. Compare arm64 and amd64 image sizes to catch unexpected bloat:

```python
import boto3

def compare_image_sizes(repo_name):
    ecr = boto3.client('ecr')
    images = ecr.describe_images(repositoryName=repo_name)['imageDetails']
    arm_sizes = [img['imageSizeInBytes'] for img in images if 'arm64' in img.get('imageTags', [])]
    x86_sizes = [img['imageSizeInBytes'] for img in images if 'amd64' in img.get('imageTags', [])]
    if arm_sizes and x86_sizes:
        print(f"ARM image larger by {(arm_sizes[0] - x86_sizes[0]) / 1e6:.1f} MB")
```

## How to decide between Graviton4 and Ampere Altra

The two families differ in core count, memory per instance, and NUMA topology. The table below lists properties that are documented by the vendors; treat the price column as illustrative and verify current pricing in your region.

| Property | AWS Graviton4 (m7g.2xlarge) | Ampere Altra A1 (48-core shape) |
|---|---|---|
| vCPU | 8 | 48 |
| Memory | 32 GB | 96 GB |
| NUMA nodes | 1 | 2 (per vendor documentation) |
| Typical use | General-purpose services | High-core-count, throughput-oriented |
| Migration risk | Low | NUMA placement under load |

Decision checklist:

- If your service is under 16 vCPU and single-NUMA, Graviton4 is the lower-risk choice. Start there.
- If you need 32+ vCPU per instance and your workload is throughput-oriented (encoding, batch processing, scientific), Ampere Altra is worth testing, but budget time for NUMA configuration and validation.
- If your workload is latency-sensitive at high core counts, measure p99 with and without Topology Manager before committing.
- If you depend on prebuilt binaries from a vendor that does not publish arm64 builds, resolve that dependency before scheduling the migration.

## Worked example: estimating whether a migration is worth it

Suppose a service runs on 6 x c7i.2xlarge instances at an illustrative on-demand price of $0.15/hour each.

- Current cost: 6 × $0.15 × 730 hours = $657/month.
- After migration to 6 x m7g.2xlarge at an illustrative $0.09/hour: 6 × $0.09 × 730 = $394/month.
- Illustrative saving: $263/month, or about 40%.

That number is arithmetic on stated assumptions, not a measured result. Before acting on it, replace the prices with the current rates for your region and your commitment (on-demand, savings plan, reserved), and confirm that the migrated service actually needs the same number of instances. If the arm64 build is slower for your workload, the instance count may need to increase, which can erase the saving entirely. Measure throughput per instance on both architectures before you resize.

## FAQ

**Why did my arm64 Lambda cost more even though it was faster?**

Lambda is billed per GB-second. If you increased the memory configuration during the migration, the per-invocation cost can rise even when the duration falls. Compare the product of configured memory and measured duration for both architectures, not duration alone. If the function is short enough that init duration dominates, the memory setting matters more than the architecture.

**How do I check whether a Python dependency has an arm64 wheel?**

```bash
pip download -r requirements.txt --platform manylinux2014_aarch64 --only-binary=:all: -d /tmp/wheels
```

If the command fails, at least one dependency has no arm64 wheel for that platform tag. Inspect the downloaded wheels:

```bash
unzip -l /tmp/wheels/*.whl | grep '\.so'
```

A compiled extension without `aarch64` in its filename is a sign the wheel is not built for your target. In that case, build from source in the image or find a pure-Python alternative.

**My Go service is slower on arm64. What should I check?**

Rebuild for the target architecture rather than relying on a cross-compiled binary built with assumptions from another platform. Check `GOARCH=arm64` is set, and verify that CGO is either disabled or that any C libraries are compiled for arm64. A binary that links an x86-only C library will not run, and one that falls back to a generic path may be measurably slower. Compare binaries built with the same flags on both architectures before drawing conclusions.

**Do I need Topology Manager for Graviton4?**

Graviton4 instances in the sizes most teams use are single-NUMA, so Topology Manager has little to do. It becomes relevant on high-core-count parts with multiple NUMA nodes, such as Ampere Altra. Enable it where the hardware has more than one NUMA node and your workload is latency-sensitive.

## Do this in the next 30 minutes

Pick one staging cluster and check the kubelet's image pull deadline and its current Topology Manager policy:

```bash
kubectl get --raw "/api/v1/nodes/$(kubectl get nodes -o jsonpath='{.items[0].metadata.name}')/proxy/configz" | python -m json.tool | grep -E 'imagePullProgressDeadline|topologyManagerPolicy|cpuManagerPolicy'
```

Record the values. If `imagePullProgressDeadline` is at or below the default and your images are large, raise it before your next rollout. If the node has multiple NUMA nodes and `topologyManagerPolicy` is `none`, that is the first configuration change to test under load.
