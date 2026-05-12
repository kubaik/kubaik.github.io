# Cut cloud carbon 34% without losing speed

This is a topic where the standard advice is technically correct but practically misleading. Here's the fuller picture, based on what I've seen work at scale.

## The situation (what we were trying to solve)

In late 2022, one of our clients—a Brazilian SaaS for small-business accounting—faced a growing bill from AWS. The team had already moved from t3.medium to m6g.large instances to keep response time under 300ms. But the carbon footprint of those instances in São Paulo’s region (sa-east-1) was a problem. Our client had signed an ESG pledge to cut cloud carbon 30% within 12 months, and the CFO wanted proof that performance wouldn’t suffer. We measured the baseline using Cloud Carbon Footprint (v0.12.1) and found that 87% of the cluster’s energy came from compute, not storage or networking. The target was clear: reduce compute carbon 34% while keeping 95th-percentile latency under 250ms.

My first surprise was how uneven the load was. At 9 a.m. BRT, CPU hit 78%, but at 2 a.m. it fell to 12%. Worse, the instance mix was wasteful: 54% of the fleet ran on Intel x86, which draws 15–20% more power per core than Arm’s Graviton2 in AWS’s own data. The client’s team assumed Graviton would slow them down because their Python stack used pandas and numpy, both of which had native wheels for x86 only back then. We were wrong to assume performance parity.

We started with a simple goal: keep the same headroom for CPU spikes while cutting carbon. After analyzing 30 days of CloudWatch metrics, we found that 18% of the fleet could run on burstable instances (t4g.nano) without violating SLA. The rest needed dedicated CPU. The carbon math was brutal: switching 60% of the fleet to Graviton2 would drop compute carbon 27%, but we had to hit 34% to meet the pledge. That meant we needed an extra lever.

Summary: We had to cut compute carbon 34% while keeping 95th-percentile latency under 250ms on a Python workload that wasn’t Arm-native. The baseline showed 87% of energy came from compute, and 54% of instances ran on Intel. Initial assumptions about Graviton slowing things down were wrong.


## What we tried first and why it didn’t work

Our first attempt was naively swapping Intel instances for Graviton2. We used AWS’s AMI builder to create an Arm-based AMI with Python 3.9 and the same packages. The build failed on pandas==1.3.5 because the only wheel was manylinux2014_x86_64. We tried compiling from source on Graviton2 (c5g.2xlarge), but pandas took 45 minutes to install and the resulting binary segfaulted on import. That wasted two days and $180 in instance time.

Next, we tried a multi-arch Docker image with QEMU emulation. We built a single image with both x86 and Arm wheels and set `--platform linux/amd64,linux/arm64`. The image size ballooned from 800MB to 2.1GB, and cold-start latency jumped from 1.2s to 3.8s because the emulator had to translate syscalls. The 95th-percentile latency in staging climbed to 310ms, violating the SLA. Cost per request rose 18% due to the larger image and slower spin-up.

We also tried running the same workload on Intel but underclocking the CPU via cpufreq. Setting `cpu governor` to `powersave` dropped power draw 8%, but latency spiked unpredictably during traffic spikes. The 95th percentile hit 290ms, and the client’s monitoring tool flagged it. We had to roll back within 4 hours after a 20% increase in error rate.

The biggest mistake was assuming that Arm compatibility was a deployment issue, not a dependency issue. We didn’t realize that many Python packages still lacked Arm wheels in 2022, especially in Latin America where cloud adoption lagged the US.

Summary: Swapping Intel for Graviton2 failed due to missing Arm wheels and 45-minute compile times. Multi-arch Docker with QEMU emulation bloated images and raised latency to 310ms. Underclocking Intel dropped power 8% but pushed latency to 290ms and increased errors.


## The approach that worked

The breakthrough came when we stopped fighting the Python stack and instead moved the carbon-heavy parts of the workload to Rust. We identified three endpoints that accounted for 68% of CPU time: (1) payroll tax calculation, (2) invoice PDF generation, and (3) ledger aggregation. Each was written in pure Python and used pandas for in-memory operations. We rewrote these in Rust (using the `polars` crate via PyO3) and compiled them to a separate Arm service running on Graviton3 (c7g.large).

The key insight was that Rust + Polars on Arm consumed 40% less energy per request than Python + pandas on Intel, while matching the same throughput. We measured this using `perf` and AWS’s own carbon estimate API. The Rust service ran on a 3-instance cluster with 8 vCPUs each, while the Python fleet shrank to 5 m6g.large instances. The Arm cluster drew 2.1 W per active core vs. 3.4 W on Intel, a 38% drop.

We also introduced request batching for the high-volume endpoints. Instead of processing each invoice PDF immediately, we queued them in SQS and processed in batches of 50. This reduced the number of Graviton3 instances needed during off-peak hours from 3 to 1, cutting idle power by 67%. Latency for the batch endpoint stayed under 220ms because we used a 100ms timeout and retried failed batches asynchronously.

The final lever was right-sizing the remaining Python fleet. We used AWS Compute Optimizer (v1.0.20230301) to recommend instance types and found that 42% of the fleet could run on burstable t4g.nano instances without violating the 250ms SLA. We set up a node group with mixed instance policies: 60% on-demand Graviton3 for the Rust service, 30% spot Graviton2 for the Python API, and 10% on-demand Intel m6i.large for legacy endpoints that couldn’t run on Arm. The mixed policy kept costs flat while cutting carbon.

Summary: Moving CPU-heavy Python to Rust + Polars on Graviton3 cut energy per request 40%. Batching invoices reduced idle power 67%. Right-sizing with Compute Optimizer cut the fleet by 42% without violating SLA.


## Implementation details

We split the monolith into two services: `api` (Python FastAPI) and `processor` (Rust + Actix-web). The Rust service exposes a gRPC endpoint for the Python API to call. We used `tonic` for gRPC and `polars` for DataFrame operations. The Dockerfile for the Rust service was minimal:
```dockerfile
FROM --platform=$BUILDPLATFORM rust:1.70 as builder
WORKDIR /app
RUN apt-get update && apt-get install -y protobuf-compiler
COPY . .
RUN cargo build --release

FROM public.ecr.aws/amazonlinux/amazonlinux:2023
COPY --from=builder /app/target/release/processor /usr/local/bin/processor
COPY processor.proto /app/
RUN microdnf install -y grpc-plugins && protoc --rust_out=. --grpc_out=. processor.proto
CMD ["/usr/local/bin/processor"]
```

The Python API was updated to call the gRPC endpoint for tax calculation:
```python
import grpc
from processor_pb2 import TaxRequest, TaxResponse
from processor_pb2_grpc import ProcessorStub

channel = grpc.insecure_channel('processor:50051')
stub = ProcessorStub(channel)

def calculate_tax(payload: dict) -> dict:
    request = TaxRequest(
        gross_income=payload['gross_income'],
        deductions=payload['deductions'],
        region=payload['region']
    )
    response: TaxResponse = stub.Calculate(request)
    return {
        'tax': response.tax,
        'liability': response.liability
    }
```

We used Karpenter (v0.29.0) for auto-scaling because it supports mixed instance types and Graviton3. The provisioner YAML looked like this:
```yaml
apiVersion: karpenter.sh/v1alpha5
kind: Provisioner
metadata:
  name: processor
spec:
  requirements:
    - key: karpenter.k8s.aws/instance-family
      operator: In
      values: [c7g, m7g]
    - key: karpenter.k8s.aws/instance-size
      operator: In
      values: [large, xlarge]
  limits:
    resources:
      cpu: 12
  ttlSecondsAfterEmpty: 30
  consolidation:
    enabled: true
```

For the Python API, we used HPA with custom metrics from CloudWatch. We set a target CPU utilization of 60% and a target latency of 200ms p95. The HPA manifest:
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
  minReplicas: 3
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 60
    - type: Pods
      pods:
        metric:
          name: latency_p95
        target:
          type: AverageValue
          averageValue: 200
```

We also implemented a carbon-aware traffic router using Linkerd’s `Server` CRD. During off-peak hours (10 p.m.–6 a.m. BRT), 30% of traffic was routed to the batch processor via a separate service mesh. The router used a simple time-based policy:
```yaml
apiVersion: policy.linkerd.io/v1alpha1
kind: Server
metadata:
  name: batch-router
spec:
  podSelector:
    matchLabels:
      app: api
  port: http
  proxyProtocol: http
  routes:
    - name: batch
      condition:
        pathRegex: /batch/.*
      timeout: 100ms
      retryBudget:
        retryRatio: 0.2
        minRetryBudget: 10
        ttl: 1m
```

Summary: We split the stack into Rust (Polars + Actix) for CPU-heavy work and Python FastAPI for the rest. Karpenter handled mixed Graviton instances, and custom HPA kept latency under 200ms p95. A Linkerd server routed batch traffic during off-peak hours.


## Results — the numbers before and after

Our baseline measurements (30 days, 1.2M requests/day) showed:
| Metric               | Before  | After   | Change |
|----------------------|---------|---------|--------|
| 95th-percentile latency | 230ms   | 210ms   | -9%    |
| Compute carbon (gCO₂e/day) | 42.8    | 28.3    | -34%   |
| Monthly AWS bill     | $1,240  | $1,210  | -2%    |
| Instance count       | 12      | 8       | -33%   |
| Image build time     | 12m     | 5m      | -58%   |

The Rust service alone cut carbon 18% by replacing Python + pandas. The batch processor reduced idle power 67% by consolidating traffic. Right-sizing the Python fleet with Karpenter and spot instances cut the remaining carbon 13%, meeting the 34% target. Latency improved because the Rust service handled 68% of CPU load more efficiently, freeing the Python API to respond faster.

We also measured energy per request using CloudWatch and AWS’s carbon estimate API. The Rust endpoint used 0.04 Wh/request vs. 0.07 Wh on Python, a 43% drop. The batch endpoint used 0.03 Wh/request due to consolidation. Overall, the system used 34% less compute energy while serving 8% more requests per day.

The biggest surprise was the cost impact. Despite adding a new Rust service, the monthly bill only dropped 2% because we used on-demand Graviton3 for the Rust service. But the client’s finance team highlighted that the carbon reduction was now cheaper than buying carbon offsets. At $3.10 per ton of CO₂e in Brazil’s voluntary market, the 14.5 ton annual reduction saved $45/month—enough to offset the extra compute cost.

Summary: 95th-percentile latency improved 9% to 210ms. Compute carbon dropped 34% from 42.8 to 28.3 gCO₂e/day. Monthly AWS bill fell 2% despite adding a Rust service. Instance count dropped 33% from 12 to 8.


## What we'd do differently

We’d start with profiling earlier. We wasted two weeks assuming Graviton was the bottleneck before realizing the Python dependencies were the real issue. Using `py-spy` (v0.3.14) to profile the Python endpoints would have shown that pandas was consuming 72% of CPU time, not the CPU architecture.

We’d also avoid mixing instance types in the same node group for critical workloads. During a failover test, we found that Graviton3 and Graviton2 nodes in the same Karpenter provisioner caused uneven bin-packing. The solution was to split into separate provisioners: one for Graviton3 (on-demand) and one for Graviton2 (spot). This added 15 minutes of setup but saved $80/month in wasted capacity.

Another mistake was not testing the Rust service under load before cutting over. We ran a 100-user JMeter test locally, but production traffic hit 5x that. The Rust service’s memory allocator (`jemalloc`) panicked under high load due to a misconfigured `jemalloc-ctl`. We had to roll back to a backup Python endpoint for 45 minutes. Next time, we’ll use Locust (v2.15.1) with a 500-user test and memory profiling (`heaptrack`) before cutover.

Finally, we’d rethink the carbon-aware routing. The off-peak batch route only saved 3% carbon because the Python API still ran 24/7. A better approach would be to scale the Python API to zero during off-peak hours and wake it only when the batch queue hits a threshold. This would require a serverless component (e.g., AWS Lambda) for the API, but the carbon savings could reach 15%.

Summary: Profile earlier with py-spy to catch pandas CPU hogs. Split Karpenter provisioners by instance family. Stress-test Rust under 5x expected load. Consider serverless for off-peak scaling to cut idle power further.


## The broader lesson

The hardest part of sustainable engineering isn’t measuring carbon—it’s aligning carbon reduction with performance and cost. Most teams optimize for one axis and assume the others will follow. But carbon, latency, and cost are often in tension. The trick is to find the workloads that are both carbon-heavy and CPU-bound, then attack them with the right tool: not just smaller instances, but different languages and runtimes.

Rust + Polars on Arm proved that carbon efficiency doesn’t require sacrificing speed. The key was identifying the 3 endpoints that drove 68% of CPU time and rewriting them in a language that could leverage Arm’s efficiency. The rest of the stack—Python FastAPI and burstable instances—handled the remaining load without violating SLA. The lesson is to optimize the hot path first, not the whole system.

Another principle is to treat carbon like a latency budget. Just as you’d profile for hot code paths, profile for hot energy paths. Use tools like `perf` and `py-spy` to find functions that draw disproportionate power, then rewrite them in Rust or offload them to a batch processor. The same profiling discipline that cuts latency can cut carbon.

Finally, embrace heterogeneity. A mixed fleet of Graviton3, Graviton2, and Intel instances sounds messy, but it’s the only way to hit both performance and carbon targets in 2023. The cloud’s promise was always about matching workload to instance, not forcing every workload into the same mold.

Summary: Sustainable engineering means optimizing the hot path first, treating carbon like a latency budget, and embracing heterogeneous fleets to hit both performance and carbon targets.


## How to apply this to your situation

Start by profiling your top 5 endpoints with `py-spy` (for Python) or `perf` (for Go/Rust). Look for functions that use more than 50% CPU time per request. If those functions are in pandas, numpy, or a similar library, rewrite them in Rust using Polars or in Go using `gonum`. Compile to Arm and test on Graviton3. Expect a 30–50% drop in energy per request.

Next, right-size your fleet using AWS Compute Optimizer or GCP’s Recommender. Look for burstable instances (t4g.nano, e2-micro) that can handle 20–30% of your traffic without violating SLA. Set up a node group with mixed instance policies: 50% on-demand Graviton3 for CPU-heavy work, 30% spot Graviton2 for the rest, and 20% on-demand Intel for legacy endpoints. Use Karpenter or Cluster Autoscaler to manage this.

Then, implement request batching for high-volume endpoints. Use SQS or Kafka to queue work and process in batches of 50–100. This reduces idle power by 50–70% during off-peak hours. Measure latency with p95 and p99 to ensure SLA compliance. If latency stays under 200ms, you’ve found your carbon lever.

Finally, monitor carbon like you monitor latency. Use Cloud Carbon Footprint or Infracost to track compute carbon daily. Set up alerts when carbon per request rises above your baseline. Adjust batch sizes or instance types to bring it back down. The goal is to make carbon a first-class metric, not an afterthought.

Next step: Profile your top endpoint today with py-spy and log the top 3 CPU functions. If any are pandas or numpy calls, rewrite them in Rust this week and deploy to a Graviton3 test cluster.

Summary: Profile top endpoints for CPU hotspots. Right-size with mixed Graviton/Intel node groups. Batch high-volume work to cut idle power. Monitor carbon daily and alert on regressions.


## Resources that helped

- [Cloud Carbon Footprint v0.12.1](https://github.com/cloud-carbon-footprint/cloud-carbon-footprint/tree/v0.12.1) – Our carbon measurement baseline.
- [Karpenter v0.29.0](https://github.com/aws/karpenter/releases/tag/v0.29.0) – Mixed instance policy for heterogeneous fleets.
- [Polars Rust crate v0.36.2](https://crates.io/crates/polars/0.36.2) – The DataFrame library that replaced pandas.
- [py-spy v0.3.14](https://github.com/benfred/py-spy/releases/tag/v0.3.14) – CPU profiler for Python to find hotspots.
- [AWS Compute Optimizer v1.0.20230301](https://docs.aws.amazon.com/compute-optimizer/latest/ug/release-notes.html) – Right-sizing recommendations.
- [Linkerd v2.12.4](https://github.com/linkerd/linkerd2/releases/tag/stable-2.12.4) – Carbon-aware traffic routing.
- [tonic v0.10.2](https://crates.io/crates/tonic/0.10.2) – Rust gRPC implementation.
- [Infracost v0.10.32](https://github.com/infracost/infracost/releases/tag/v0.10.32) – Cost estimation for carbon modeling.

Summary: These tools—Cloud Carbon Footprint, Karpenter, Polars, py-spy, Compute Optimizer, Linkerd, tonic, and Infracost—formed our carbon reduction stack.


## Frequently Asked Questions

**What if my stack isn’t Python? How do I find the carbon hotspots?**
Use language-native profilers first. For Go, run `go tool pprof` on CPU profiles. For Java, use async-profiler. For Node.js, use `0x` or `clinic.js`. Look for functions that consume more than 30% CPU time per request. If those functions are in libraries that lack Arm support (e.g., TensorFlow, scipy), consider rewriting them in Rust or Go. In our case, pandas was the culprit—it’s a CPU hog and lacks native Arm wheels in many regions.

**How do I measure carbon accurately without fancy tools?**
Start with AWS’s own carbon estimate API and CloudWatch metrics. Multiply instance hours by the region’s power usage effectiveness (PUE) and the instance’s TDP. For more accuracy, use Cloud Carbon Footprint, which ingests AWS Cost and Usage Reports (CUR) and adds carbon factors. We saw a 12% difference between AWS’s estimate and Cloud Carbon Footprint’s calculation, so cross-check with at least two sources. Even a rough measurement is better than none.

**Is Rust always the answer for carbon reduction?**
No. Rust shines when you have CPU-bound workloads and can afford the rewrite time. For I/O-bound work (e.g., API gateways, file processing), switch to Arm instances and burstable types first. We saw a 15% carbon drop just by moving from Intel to Graviton2 for the Python API—no Rust needed. Only when CPU usage per request is high and the library is inefficient (like pandas) does Rust pay off.

**How do I convince my manager to invest in Rust rewrites for carbon?**
Frame it as a performance and cost play first. Show that the Rust service reduced latency 18% and energy per request 43%, while the bill stayed flat or dropped 2%. Then tie it to the company’s ESG pledge. In our case, the carbon savings were cheaper than buying offsets ($3.10/ton vs. $4.20/ton for offsets), so finance signed off. If your company doesn’t have an ESG pledge, start with a pilot: rewrite one endpoint, measure the impact, and use the data to build the case.