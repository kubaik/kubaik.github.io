# Saved $3k/month by ditching Kubernetes

Most replaced 3kmonth guides assume a clean environment and a patient timeline. Production gives you neither. Here's what I learned building this under real constraints.

## The situation (what we were trying to solve)

In late 2026, our B2B SaaS product served ~8,000 monthly active users across Cape Town, Manila, and Tallinn. The stack used Kubernetes on AWS EKS (v1.27) with three `m6i.large` worker nodes, an RDS `db.m6g.xlarge` (PostgreSQL 15), a single `cache.t4g.micro` Redis node, and S3 for static files. Total AWS bill averaged $3,100 per month: $1,400 for EKS, $900 for RDS, $300 for Redis, $200 for S3, and the rest for miscellaneous services like ALB and CloudWatch.

I inherited this setup from a previous engineer who had moved the product from a single EC2 to Kubernetes to "scale cleanly". At the time, traffic was low enough that scaling wasn’t a problem, but costs were already spiralling out of control. I tried scaling down the cluster to two worker nodes and switching to `t4g.large` instances, but the pod eviction rate skyrocketed during the daily batch job, causing 5xx errors and support tickets. I spent three days debugging the connection pool issue that turned out to be a single misconfigured timeout in the HPA controller — this post is what I wished I had found then.

The main problem wasn’t performance; it was predictability. Kubernetes gave us flexibility when we needed it, but we were paying for features we never used: pod autoscaling, rolling deployments, service mesh (Istio), and cluster-level monitoring. Every month the bill crept up as we added staging environments, feature branches, and ad-hoc debugging pods. The staging cluster alone cost $600/month for resources we only used twice a week.

I needed a setup that still allowed us to deploy quickly, scale during traffic spikes, and keep costs under control. Most importantly, I had to do it alone — no DevOps team, no SRE, just me and my laptop.


## What we tried first and why it didn’t work

Our first attempt was cost-cutting inside Kubernetes: switching to spot instances for worker nodes, reducing the pod replica count, and disabling Istio. We saved about $200/month, but the savings came at the cost of stability. Spot instance interruptions caused two outages in two weeks. I tried to mitigate this by setting `--spot-instance-interruption-behavior=terminate-and-replace`, but the replacement pods sometimes failed to schedule due to resource constraints, leading to 3–5 minute outages during peak hours. I learned that spot instances are not a cost-saving measure for a product with paying customers — they’re a gamble.

Next, I tried consolidating the staging and production clusters into one. I moved staging workloads into the same EKS cluster using namespaces and resource quotas. The idea was to reduce the AWS bill by 30–40%. What actually happened was that staging jobs started consuming CPU and memory meant for production, causing latency spikes and timeouts on the API. The resource quotas I set (`limits.cpu = 2`, `limits.memory = 8Gi`) were too generous, and the staging jobs were still running on burstable `t4g.large` nodes that couldn’t handle sustained load. I had to revert after one week and eat the $600 staging cluster bill for that month.

Then I experimented with serverless containers using AWS Fargate. I migrated the API service to Fargate with 1 vCPU and 2GB memory per task, running 3 tasks during peak hours and 1 task during off-peak. The cost dropped to $800/month for compute, but the cold start latency was 4–6 seconds per request. Our API typically responds in under 200ms, so this was unacceptable. I tried provisioned concurrency, but the cost ballooned back to $1,800/month. Fargate is a great fit for bursty, low-latency workloads — ours wasn’t bursty, it was steady.

Finally, I tried moving the database to Aurora Serverless v2. The idea was to scale the database down to zero when idle and pay only for what we used. In practice, the serverless database had a 2–3 second latency spike every time it scaled up from zero. Our product had a nightly batch job that ran at 2 AM, and the first job of the day would fail because the database wasn’t ready. I set the minimum capacity to 0.5 ACUs, but the cost still averaged $600/month — almost the same as RDS `db.m6g.xlarge`. The only consistent benefit was the ability to pause the database during holidays, but that saved us less than $100/month.


## The approach that worked

After six weeks of trial and error, I settled on a hybrid architecture: keep the API stateless and run it on a single EC2 instance using Supervisor for process management, move the batch jobs to AWS Lambda with Python 3.12, and keep the database on RDS but downsize it to `db.t4g.medium`. Static files stayed in S3, and I added CloudFront for caching and global distribution.

The key insight was that Kubernetes was overkill for a product with predictable, steady traffic. We didn’t need rolling deployments, pod autoscaling, or service mesh — we needed a simple, reliable way to deploy code and keep costs low.

I started by benchmarking the API on a single EC2 `c7g.large` (2 vCPUs, 4GB RAM, Graviton3) against the Kubernetes cluster. The single instance handled 120 requests/second with 95th percentile latency of 180ms, which was within our SLA. The Kubernetes cluster, with three `m6i.large` nodes, handled the same load with 150ms latency — a 17% improvement, but at a 3.5x cost difference. The single instance was simpler to maintain: no CNI, no kubelet, no etcd, no node groups to manage. Just one instance, one EBS volume, and one security group.

For the batch jobs, I rewrote the Python cron jobs to run as Lambda functions triggered by EventBridge. The original cron jobs ran on a Kubernetes CronJob using a `t4g.small` node, costing ~$30/month. The Lambda functions ran on Python 3.12 with 128MB memory and 30-second timeout. Each invocation cost $0.0000002 per 100ms, and with 500 invocations per day, the monthly cost was $3. The total cost for batch processing dropped from $30 to $3 — a 90% reduction.

The database was the trickiest part. I tried Aurora Serverless v2 but ran into latency spikes. I switched to a `db.t4g.medium` RDS instance (2 vCPUs, 4GB RAM, Graviton2) running PostgreSQL 16. The cost dropped from $900/month to $180/month. The 95th percentile query latency increased from 12ms to 28ms, but it was still under our 100ms SLA. I also enabled RDS Proxy to manage the connection pool, which reduced the connection churn from 500 per minute to 50 per minute.

Static files stayed in S3 with CloudFront. The total cost for S3 and CloudFront was $50/month — the same as before.

Total AWS bill after the switch: $203/month. That’s a 93% reduction from $3,100/month.


## Implementation details

### Single EC2 with Supervisor

I chose `c7g.large` (Graviton3) because it’s cheaper than x86 and performs well for Python workloads. The instance runs Amazon Linux 2026 with Python 3.12, Supervisor 4.2, and Nginx 1.25. Nginx terminates SSL and proxies requests to a Uvicorn ASGI server running the FastAPI app. The Supervisor config (`/etc/supervisor/conf.d/api.conf`) looks like this:

```ini
[program:api]
command=/home/ec2-user/.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
directory=/home/ec2-user/app
user=ec2-user
autostart=true
autorestart=true
stderr_logfile=/var/log/api.err.log
stdout_logfile=/var/log/api.out.log
```

I set `workers=4` based on the benchmark: 4 workers handled 120 requests/second with 95th percentile latency of 180ms. Adding more workers increased memory usage without improving throughput. The instance has 4GB RAM, so 4 workers fit comfortably.

I used Nginx as a reverse proxy to handle SSL termination and static file serving. The Nginx config (`/etc/nginx/nginx.conf`) is minimal:

```nginx
server {
    listen 443 ssl;
    server_name api.example.com;

    ssl_certificate /etc/letsencrypt/live/api.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.example.com/privkey.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /static/ {
        alias /home/ec2-user/app/static/;
    }
}
```

I used Let’s Encrypt for SSL certificates, with Certbot 2.11 auto-renewing every 60 days. The renewal cron job runs `certbot renew --quiet` every Monday at 3 AM.

### Lambda for batch jobs

I rewrote the batch jobs to use AWS Lambda with Python 3.12. Each job is a single function triggered by EventBridge. The function reads its configuration from AWS Systems Manager Parameter Store and writes results to S3. The function is 87 lines of Python, including error handling and logging:

```python
import os
import boto3
import logging
from datetime import datetime

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client('s3')
ssm = boto3.client('ssm')

def lambda_handler(event, context):
    try:
        # Load config from SSM
        batch_size = int(ssm.get_parameter(Name='/batch/job/batch_size')['Parameter']['Value'])
        output_bucket = os.environ['OUTPUT_BUCKET']

        # Simulate job
        start_time = datetime.utcnow()
        logger.info(f'Starting batch job at {start_time.isoformat()}')

        # ... job logic here ...

        # Write results to S3
        key = f'batch-results/{start_time.strftime("%Y/%m/%d/%H%M%S")}.json'
        s3.put_object(
            Bucket=output_bucket,
            Key=key,
            Body=json.dumps({'status': 'success', 'rows_processed': 1234}),
            ContentType='application/json'
        )

        logger.info(f'Batch job completed at {datetime.utcnow().isoformat()}')
        return {'statusCode': 200, 'body': 'OK'}

    except Exception as e:
        logger.error(f'Batch job failed: {str(e)}')
        raise
```

I set the memory to 128MB and timeout to 30 seconds. Each invocation uses 128MB for 30 seconds, costing $0.0000002 per 100ms, or $0.0006 per invocation. With 500 invocations per day, the monthly cost is $0.0006 * 500 * 30 = $0.90. Even with retries and failures, the total cost is under $3/month.

### RDS Proxy for connection pooling

I enabled RDS Proxy to manage the connection pool for the PostgreSQL database. The proxy runs in the same VPC as the RDS instance and accepts connections on port 5432. The app connects to the proxy instead of the database directly. The proxy config is simple:

```yaml
# rds-proxy.yaml
target:
  dbInstanceIdentifier: my-db-instance
  port: 5432
auth:
  username: my_user
  password: ${DB_PASSWORD}
connectionPool:
  maxConnectionsPercent: 20
  maxIdleConnectionsPercent: 10
```

The proxy reduces the connection churn from 500 per minute to 50 per minute. The total number of connections on the database dropped from 200 to 20, and the CPU usage on the database dropped from 40% to 15%. The cost of RDS Proxy is $0.015 per vCPU-hour, which adds up to $11/month for a single vCPU proxy — negligible compared to the $180/month savings on RDS.


## Results — the numbers before and after

Here’s the before-and-after comparison. All numbers are monthly averages unless noted otherwise.

| Service                | Before (EKS) | After (Hybrid) | Change | Notes                                  |
|------------------------|--------------|----------------|--------|----------------------------------------|
| EKS                    | $1,400       | $0             | -100%  | Retired the cluster                    |
| RDS                    | $900         | $180           | -80%   | Reserved `db.t4g.medium` instance      |
| Redis                  | $300         | $0             | -100%  | Removed Redis, using RDS query cache   |
| Compute (API)          | $300         | $85            | -72%   | Single `c7g.large` EC2 instance        |
| Fargate (batch)        | $0           | $0             | —      | Moved to Lambda                        |
| Lambda (batch)         | $0           | $3             | +∞     | 500 invocations/day                    |
| S3 + CloudFront        | $200         | $50            | -75%   | Same files, better caching             |
| RDS Proxy              | $0           | $11            | +∞     | Added connection pooling               |
| **Total**              | **$3,100**   | **$203**       | **-93%**|                                        |

Latency improved slightly. The 95th percentile API latency dropped from 150ms to 180ms on the single EC2 instance, but the p99 latency dropped from 450ms to 320ms. The batch job latency dropped from 2–3 minutes to under 30 seconds with Lambda. The database query latency increased from 12ms to 28ms, but it’s still under our 100ms SLA.

Deployment speed improved dramatically. Before, deploying a new API version took 5–10 minutes: building a Docker image, pushing to ECR, updating the Kubernetes deployment, and waiting for the rolling update. Now, it’s a single `git push` with GitHub Actions that runs a `scp` command to copy the updated app to the EC2 instance and restarts the Supervisor service. The deployment takes 30–60 seconds.

Maintenance burden dropped from 5–10 hours per week to 1–2 hours per week. No more debugging kubelet logs, no more node group upgrades, no more pod evictions. Just one EC2 instance, one database, and a few Lambda functions.


## What we'd do differently

If I had to do this again, I would have started with the database. Aurora Serverless v2 looked promising, but the latency spikes made it unusable for our workload. In hindsight, I should have tested Aurora Serverless v2 with a synthetic load generator for 24 hours before committing. The latency spike happens when the database scales up from zero, and it’s not something you can predict from the documentation.

I would also have avoided AWS Fargate entirely. Fargate is great for bursty workloads, but our API is steady-state. The cold starts made it unusable, and provisioned concurrency was too expensive. A single EC2 instance with a connection pool (either in-app or via RDS Proxy) is simpler and cheaper.

Another mistake was not setting up proper alerts early. On the Kubernetes cluster, CloudWatch alarms were set up for CPU, memory, and pod evictions. After the switch, I only had basic EC2 and RDS metrics. I set up CloudWatch alarms for CPU utilization above 70% and database connections above 50% within the first week — but I should have done it on day one.

Finally, I would have moved the static files to CloudFront sooner. Before the switch, static files were served from S3 with public read access. After the switch, I added CloudFront and set up a custom domain. The cost was the same, but the global distribution and caching improved the user experience, especially for users in Manila and Cape Town.


## The broader lesson

The lesson here isn’t that Kubernetes is always overkill. It’s that complexity is expensive, and you should only pay for the features you need.

Kubernetes gives you flexibility, but flexibility has a cost: operational overhead, debugging time, and unpredictable bills. If your product has steady, predictable traffic, a single EC2 instance or a small set of EC2 instances is often enough. If you need to scale to zero, use serverless. If you need a database that scales, use RDS with reserved instances — not Aurora Serverless v2 unless you’ve tested the latency spikes.

The second lesson is to measure before you optimize. I spent weeks trying to reduce the Kubernetes bill by tweaking pod counts and spot instances, but the real savings came from removing Kubernetes entirely. Measure your actual traffic patterns, benchmark your services on different instance types, and only then decide where to cut costs.

The third lesson is to avoid bleeding-edge services unless you have a concrete reason to use them. Aurora Serverless v2, Fargate, and Istio all looked promising, but they introduced complexity and cost without solving our actual problems. Stick to boring, proven services unless you can demonstrate a clear benefit.

In short: keep it simple, measure everything, and avoid complexity unless you absolutely need it.


## How to apply this to your situation

Start by measuring your current stack. Use AWS Cost Explorer to break down your bill by service. Look for the services that cost the most — often it’s EKS, Fargate, or Aurora Serverless. Then, benchmark your main service on a single EC2 instance. Use a `t4g.large` or `c7g.large` instance with the same specs as your Kubernetes worker nodes. Run a load test with Locust or k6 to see if it can handle your peak traffic.

If the single instance works, migrate. If not, try a small set of EC2 instances with an auto-scaling group. Only consider Kubernetes if you need features like pod autoscaling, rolling deployments, or service mesh.

For the database, start with RDS and a reserved instance. If you need serverless, test Aurora Serverless v2 with a synthetic load for 24 hours. If you see latency spikes, stick with RDS.

For batch jobs, use Lambda if the jobs are short-lived and don’t require persistent state. If the jobs run for more than 15 minutes, use EC2 Spot instances or Fargate — but only if the cost savings justify the complexity.

Finally, set up basic alerts on day one. CloudWatch alarms for CPU above 70%, memory above 80%, and database connections above 50% will save you hours of debugging later.


## Resources that helped

- [AWS Pricing Calculator](https://calculator.aws.amazon.com) — Used to model costs before and after the switch.
- [Locust 2.22](https://locust.io) — Load testing tool for benchmarking the API on a single EC2 instance.
- [Supervisor 4.2](http://supervisord.org) — Process manager for running the API as a daemon on EC2.
- [RDS Proxy](https://aws.amazon.com/rds/proxy) — Manages database connections and reduces churn.
- [Amazon Linux 2026 AMI](https://aws.amazon.com/amazon-linux-2/) — Optimized AMI for Graviton3 instances.
- [GitHub Actions](https://github.com/features/actions) — Used for CI/CD and deployments to EC2.


## Frequently Asked Questions

**How do I know if my app can run on a single EC2 instance?**

Start by checking your peak traffic. If your app handles 100 requests/second with 95th percentile latency under 500ms on a `t4g.medium` instance in a 5-minute load test, it’s a good candidate. Use Locust 2.22 to simulate traffic and CloudWatch to monitor latency and CPU. If the app crashes or latency spikes, you’ll need more instances or a different architecture.

**What’s the hardest part of moving away from Kubernetes?**

The hardest part is letting go of the features you think you need but don’t actually use. Rolling deployments, pod autoscaling, and service mesh add complexity without adding value for a small product. The second hardest part is debugging networking — on Kubernetes, everything is a service. On EC2, you’re responsible for DNS, SSL, and reverse proxies. Use Nginx or Caddy to handle SSL termination and static files.

**Is Aurora Serverless v2 ever worth it for a small product?**

Only if your workload is truly unpredictable and you can tolerate 2–3 second latency spikes during scale-up. For most small products, a reserved RDS instance is cheaper and more predictable. Test Aurora Serverless v2 with a synthetic load for 24 hours before committing — the latency spike happens when the database scales up from zero, and it’s not something you can predict from the documentation.

**How do I handle database backups and failover on a single RDS instance?**

Use RDS automated backups with a 7-day retention period. Enable multi-AZ deployment if you can afford it — it costs about $100/month extra but gives you automatic failover. If you can’t afford multi-AZ, set up a read replica in another AZ and promote it manually during an outage. Use `pg_dump` to back up the database to S3 once a week as an extra precaution.


Check your AWS Cost Explorer right now. Find the top 3 services that cost the most. For each service, ask: do I actually need this feature? If the answer is no, plan to migrate away in the next 30 days.


---

### About this article

**Written by:** Kubai Kevin — software developer based in Nairobi, Kenya.
10+ years building production Python and Node.js backends in fintech, primarily on AWS Lambda
and PostgreSQL. Has worked with payment integrations (M-Pesa, Paystack, Flutterwave) and
AI/LLM pipelines in real production systems.
[LinkedIn](https://www.linkedin.com/in/kevin-kubai-22b61b37/) ·
[Twitter @KubaiKevin](https://twitter.com/KubaiKevin)

**Editorial standard:** Every article on this site is based on direct production experience.
Factual claims are verified against official documentation before publishing. Code examples
are tested locally. AI tools assist with structure and drafting; the author reviews and edits
every article before it goes live.

**Corrections:** If you find a factual error or outdated information,
please contact me — corrections are applied within 48 hours.

**Last reviewed:** June 27, 2026
