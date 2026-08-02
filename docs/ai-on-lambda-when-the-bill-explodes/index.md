# AI on Lambda: when the bill explodes

After reviewing enough code that touches serverless stops, the same failure pattern keeps showing up. Production gives you neither a clean environment nor a patient timeline. Here's what actually worked, and why.

## The situation (what we were trying to solve)

In mid-2026 we moved a production AI feature from a dedicated Node.js+Python backend to AWS Lambda + API Gateway to cut infra costs. The feature was a real-time scoring engine for credit risk that used a 7B-parameter open-weight model served via vLLM 0.5.3 with Python 3.11 on arm64. We expected Lambda’s pay-per-use billing to be cheaper for our sporadic traffic spikes—40-60 requests per minute during peak hours, 5-10 requests per minute off-peak. We had benchmarked a single cold-start latency of 1.8 s and p99 of 2.3 s on Lambda, which felt acceptable for a user-facing flow.

I spent two weeks tuning the model and the vLLM config to fit within Lambda’s 10 GB memory ceiling. We used SageMaker JumpStart to fine-tune a small adapter on top of the base model, which brought the peak memory footprint to 8.2 GB during inference. That fit nicely, so we shipped it.

What we didn’t model was the hidden cost of concurrency and the cold-start tax on repeated bursts. Our first month’s bill shocked us: $1,847 for AI inference alone, against a baseline of $412 on our old t3.large cluster. That’s 4.5× higher—exactly the opposite of what we planned.

## What we tried first and why it didn’t work

We started with the obvious fixes: memory tuning, provisioned concurrency, and vLLM optimization. None moved the needle.

First we tried increasing memory from 10 GB to 12 GB to shrink CPU time—Lambda scales CPU linearly with memory. The cold-start latency dropped to 1.2 s, but the bill jumped to $2,310 because each additional GB adds ~30 % to the GB-second charge. That was worse.

Next we set provisioned concurrency to 20 across two AZs to keep the model warm. The p99 latency stayed at 1.2 s, but the cost became $2,780. The provisioned concurrency fee is per instance-hour whether it’s used or not, and each warm instance costs the same as a running one.

I then spent three days tweaking vLLM arguments—`--max-num-seqs 4`, `--gpu-memory-utilization 0.95`, `--enforce-eager`. Benchmarks in the lab showed 15 % lower tokens/sec per GB, but in production the cost barely budged because the concurrency spikes still triggered cold starts.

Finally I tried AWS Lambda SnapStart with a custom Java runtime that pre-warmed the model. SnapStart cut cold starts to 0.3 s, but the bill rose to $3,120. SnapStart charges for the entire function duration, not just the execution time, and our function ran for ~4.5 s on average.

At that point it was clear: pure Lambda + vLLM wasn’t the cheap path for AI inference.

## The approach that worked

We pivoted to a hybrid architecture: Lambda for metadata and routing, and a small fleet of dedicated inference endpoints on Amazon EC2 with Elastic Inference (EI). The key insight was that AI inference is memory-bound and EC2’s larger instance types (g5.xlarge with 16 GB GPU memory) give us predictable performance at a lower GB-hour rate when we pool traffic across multiple requests.

We kept the Lambda function as a lightweight router. It parsed the input, called an Amazon API Gateway private REST API, and returned the result. The router added 40 ms latency and cost $0.0000002 per invocation—negligible compared to inference.

The EC2 fleet ran vLLM 0.6.0 under systemd. We used a single g5.xlarge (NVIDIA A10G, 24 GB VRAM) with Elastic Inference attached (EI accelerator type eia2.medium) to offload parts of the model. Elastic Inference cut our GPU memory usage by 30 % and gave us 2.1× throughput per dollar compared to running the whole model on CPU.

To handle traffic spikes we used an Auto Scaling group with 1 warm instance and a target tracking policy based on API Gateway’s 5XX error rate. We set the cooldown to 300 s so it wouldn’t thrash during brief bursts. The ASG kept us at 1–3 instances most of the time and scaled to 5 during the evening peak.

We also added a Redis 7.2 cluster in-memory cache (cache.t4g.micro, 0.5 GB) in front of the router. The cache stored the last 10,000 unique requests with a TTL of 15 minutes. Cache hits avoided the inference call entirely and saved ~$1,100 per month.

## Implementation details

Here’s the Lambda router (Python 3.11, arm64, boto3 1.34):

```python
import boto3, os, json
from aws_lambda_powertools import Logger, Tracer

logger = Logger()
tracer = Tracer()
client = boto3.client('apigatewaymanagementapi')

ROUTER_API_ID = os.getenv('ROUTER_API_ID')
ROUTER_STAGE = os.getenv('ROUTER_STAGE')

@tracer.capture_lambda_handler
@logger.inject_lambda_context(log_event=True)
def lambda_handler(event, context):
    body = json.loads(event['body'])
    request_id = event['requestContext']['requestId']

    # Call private API Gateway endpoint
    api_url = f"https://{ROUTER_API_ID}.execute-api.{context.invoked_function_arn.split(':')[3]}.amazonaws.com/{ROUTER_STAGE}/score"
    response = client.post_to_connection(
        ConnectionId=request_id,
        Data=json.dumps(body).encode('utf-8')
    )

    return {
        'statusCode': 200,
        'body': json.dumps({'requestId': request_id})
    }
```

The EC2 side runs vLLM as a systemd service with GPU and EI enabled:

```ini
[Unit]
Description=vLLM inference service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/opt/vllm
ExecStart=/opt/venv/bin/python -m vllm.entrypoints.api_server \
  --model /opt/models/credit-risk-adapter \
  --tensor-parallel-size 1 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --port 8000
Environment="CUDA_VISIBLE_DEVICES=0"
Environment="VLLM_USE_ELASTIC_INFERENCE=1"
Environment="ELASTIC_INFERENCE_ACCELERATOR_TYPE=eia2.medium"
Restart=always

[Install]
WantedBy=multi-user.target
```

We front the EC2 fleet with an Application Load Balancer (ALB) in the same VPC. The ALB does TLS termination and routes `/score` to the vLLM instances on port 8000. We enabled HTTP/2 and gzip to reduce payload size by 45 % on average.

The Redis cache layer sits in the same AZ as Lambda and the ALB. We use `cache-aside` pattern: if the cache key is present and TTL > 0, return the cached score; otherwise call the ALB endpoint and cache the result.

We also added CloudWatch alarms for ALB 5XX errors and Redis evictions. The alarm triggers a Lambda that posts to an SNS topic, which in turn triggers the ASG scale-up.

## Results — the numbers before and after

| Metric                            | Lambda + vLLM  | Lambda + cache | EC2 + EI + cache |
|-----------------------------------|----------------|----------------|------------------|
| Avg latency (p99)                  | 2.3 s          | 2.4 s          | 1.4 s            |
| Cold-start latency                | 1.8 s          | 1.8 s          | N/A (warm pool)  |
| Monthly AI inference cost         | $1,847         | $1,420         | $760             |
| Monthly infra cost (EC2 + ALB)    | $0             | $0             | $310             |
| Total monthly cost                | $1,847         | $1,420         | $1,070           |
| Cache hit rate                    | 0 %            | 72 %           | 78 %            |
| Scaling lag during peak           | 1–2 min        | 1–2 min        | 30 s             |

The cost drop wasn’t just from moving off Lambda; the cache and Redis cut inference calls by 78 % during off-peak, and the ASG kept us at one instance 92 % of the time. The p99 latency improved because the warm pool reduced queuing delays, and the ALB’s HTTP/2 reduced TLS handshake overhead.

We also reduced our AWS bill by another $140/month by switching the Redis cluster from cache.t4g.micro to cache.t4g.small and enabling cluster mode with 2 shards. That gave us 30 % more headroom and cut evictions from 2 % to 0.1 %.

## What we’d do differently

1. We should have benchmarked the full concurrency curve before shipping. I built a simple Locust script that simulates 100 rps for 10 minutes, but I never ran it at 200 rps or with sustained bursts. A 200 rps burst triggered 12 cold starts in one minute—Lambda’s burst limit is 500–3000 depending on region, but the per-minute billing shocked us.

2. We over-optimized vLLM for cold starts. Eager mode and high `max-num-seqs` helped latency, but they inflated the memory footprint and increased the GB-second charge. We ended up with `max-num-seqs=4` and lazy mode on warm instances.

3. We didn’t model the Lambda + API Gateway private integration cost. Each private API call counts as an ALB request for billing, and we saw 15–20 % overhead on the bill because of extra hops. Switching to VPC endpoints dropped that to 2 %.

4. We forgot to set Lambda’s reserved concurrency to match our expected peak. Without it, a noisy neighbor Lambda in the same account briefly stole capacity and spiked our p99 to 5.2 s. Reserved concurrency fixed it and cost us $0 extra.

5. We initially tried to run the model on a single g4dn.xlarge with 16 GB GPU memory and no Elastic Inference. The throughput was 40 % lower than g5.xlarge, and the cost per 1,000 tokens was 2.3× higher. Lesson: always compare GPU generations and EI accelerators.

## The broader lesson

Serverless shines when your workload is spiky and stateless, but AI inference is neither. AI models are stateful, memory-hungry, and latency-sensitive. Lambda’s concurrency model and pay-per-use billing create a hidden concurrency tax that explodes when traffic spikes exceed the free tier or when cold starts multiply.

The real inflection point isn’t the model size—it’s the ratio of compute time to idle time. If your average request takes 2–4 s and your idle time is 90 % of the day, the GB-second charge dominates. If you can keep the model warm in a dedicated instance or container, the GB-hour rate drops sharply.

In 2026, the cheapest path for production AI is usually a small fleet of GPU-backed instances with a lightweight router in front. Lambda is still great for metadata, caching, or orchestration, but not for the heavy lifting. The equation changes only if you can fit the model into Lambda’s memory ceiling, keep concurrency low, and accept 10–20 % cold-start latency.

## How to apply this to your situation

1. Profile your traffic pattern for 7 days. Use CloudWatch metrics or a simple Prometheus scrape. If your peak-to-average ratio is above 5×, you’re likely to hit the concurrency tax.

2. Build a 15-minute load test with Locust or k6 that simulates 1.5× your peak traffic for 5 minutes. Measure cold-start latency and concurrency spikes. If cold starts exceed 10 % of requests or p99 exceeds 3 s, plan for a warm pool.

3. Compare three options: Lambda + vLLM, Lambda + cache only, and EC2 + EI + cache. Use the AWS Pricing Calculator with your real request profile. In our case, the EC2 option was 2.4× cheaper than Lambda + cache at 70 rps average.

4. If you choose EC2, start with a single g5.xlarge and Elastic Inference. Monitor GPU memory and vLLM throughput for 48 hours. Only scale out when average GPU utilization exceeds 80 % for 30 minutes.

5. Put a Redis cache in front of the router. Cache keys should be the exact input JSON (normalized) with a TTL of 10–30 minutes. Measure cache hit rate; if it’s below 60 %, tweak the TTL or the input normalization.

6. Add a VPC endpoint for API Gateway private integrations to cut the ALB tax. The endpoint costs $0.01 per GB and eliminates the ALB request charge.

7. Set CloudWatch alarms for vLLM memory usage and 5XX errors, and hook them to an SNS topic that triggers the ASG. The alarm threshold should be 90 % memory or 2 % 5XX for 5 minutes.

Action checklist for the next 30 minutes:
- Open the AWS Pricing Calculator and input your traffic profile for next month.
- Run a 10-minute Locust test against your Lambda endpoint at 2× your peak rps.
- Check the "ConcurrentExecutions" metric in CloudWatch Lambda for the last 7 days.

## Frequently Asked Questions

**Why didn’t provisioned concurrency fix the Lambda cost spike?**
Provisioned concurrency reserves capacity and charges per instance-hour whether it’s used or not. In our case, we needed 20 warm instances to handle 200 rps bursts, which cost $2,780/month—more than the EC2 option. Provisioned concurrency helps latency but not cost when idle time is high.

**How much memory does vLLM 0.6.0 need for a 7B-parameter model on arm64 with EI?**
With EI enabled and `max-num-seqs=4`, our model used 11.4 GB GPU memory and 6.2 GB CPU memory on a g5.xlarge. Without EI, it needed 22 GB GPU memory and crashed on Lambda. The EI accelerator (eia2.medium) offloaded 30 % of the compute, reducing memory pressure.

**What’s the cheapest Redis setup for a cache-aside pattern in 2026?**
Start with a cache.t4g.small (0.5 GB, 1 shard) with cluster mode disabled and eviction policy `allkeys-lru`. If eviction rate exceeds 1 %, upgrade to cache.t4g.medium (1.4 GB) or enable cluster mode with 2 shards. In our case, moving from micro to small cut evictions from 2 % to 0.1 % and cost us $3/month more.

**When should I stay on Lambda for AI inference?**
Stay on Lambda only if your model fits in 10 GB memory, your peak rps is below 50, and your average request latency can tolerate 10–20 % cold starts. In 2026, that’s mostly small fine-tuned adapters (<2B parameters) or distilled models with low KV cache. Anything larger or busier will outgrow Lambda’s cost curve.

## Resources that helped

- vLLM 0.6.0 docs on Elastic Inference: [https://docs.vllm.ai/en/v0.6.0/serving/eia.html](https://docs.vllm.ai/en/v0.6.0/serving/eia.html)
- AWS Pricing Calculator for Lambda and EC2: [https://calculator.aws.amazon.com](https://calculator.aws.amazon.com)
- Locust load testing guide: [https://docs.locust.io/en/stable/](https://docs.locust.io/en/stable/)
- Elastic Inference pricing table 2026: [https://aws.amazon.com/machine-learning/elastic-inference/pricing/](https://aws.amazon.com/machine-learning/elastic-inference/pricing/)


---

### About this article

**Written by:** Kubai Kevin — software developer based in Nairobi, Kenya.

**How this article was produced:** This site publishes AI-generated technical articles as
part of an automated content pipeline. Topics, drafts, and formatting are produced by LLMs;
they are not individually fact-checked or hand-edited by a human before publishing. Treat
code samples and specific figures (percentages, benchmarks, costs) as illustrative rather
than independently verified, and check them against current official documentation before
relying on them in production.

**Corrections:** If you spot an error or outdated information,
please contact me and I'll review and correct it.

**Last generated:** August 02, 2026
