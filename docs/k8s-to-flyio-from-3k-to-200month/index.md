# K8s to Fly.io: from $3k to $200/month

Most replaced 3kmonth guides assume a clean environment and a patient timeline. Production gives you neither. Here's what I learned building this under real constraints.

## The situation (what we were trying to solve)

In late 2026, our SaaS product started getting traction. We were running a managed Kubernetes cluster on AWS EKS with 3 medium-sized nodes (m6i.large). The bill hit $3,180 in November 2026. For context, we were a team of two full-stack engineers and one designer. Our product served around 8,000 daily active users, mostly from Southeast Asia and Europe. The API was written in Python 3.11 with FastAPI 0.111.0, and the frontend was a Next.js 14 app hosted on Vercel.

We didn’t need Kubernetes. We needed a system that would scale when traffic spiked—like when a marketing campaign went live—and stay cheap when it was quiet. Our EKS cluster was overkill for a team that didn’t have a dedicated DevOps person. I spent a week setting up ArgoCD, Prometheus, and Grafana, only to realize we were maintaining infrastructure instead of building features. The cluster was stable, but the bill was not. I ran into this when I got an email from AWS saying our November bill was 30% higher than October. I dug in and found that our cluster autoscaler was spinning up extra nodes for no good reason—just because the cluster felt lonely at 3 a.m.

We tried reducing the node count, but then the latency spiked during peak hours. We switched from gp3 to cheaper sc1 storage, which saved $200/month but introduced 500ms extra latency on cold starts. Our users noticed. I spent three days tweaking the cluster autoscaler settings, only to realize the real issue was that we were running 4 replicas of every service when 2 would have been enough. We were paying for redundancy we didn’t need.

The big question was: how do we keep the uptime high without paying for idle resources?

## What we tried first and why it didn’t work

Our first attempt was to downsize the cluster. We switched from 3 m6i.large nodes to 2 m6i.xlarge nodes. The bill dropped to $2,400/month, but the latency during traffic spikes went from 120ms to 350ms. Our users in Manila and Cape Town complained about slow load times. I dug into the metrics and found that the CPU throttling on the smaller nodes was causing the API to queue requests. We tried increasing the CPU limits, but that just doubled the bill again. 

Then we tried spot instances. We configured the cluster autoscaler to use spot instances for non-critical workloads. The bill dropped to $1,800/month, but the cluster became unstable. Two days later, AWS terminated our spot nodes during a peak, and we had a 15-minute outage. Our SLA was 99.9%, and we missed it. I spent a day debugging why Prometheus wasn’t alerting on the spot instance termination—turns out the alert rule for `kube_node_unreachable` was set to 5 minutes, and AWS had given us only 2 minutes of notice.

We also tried using AWS Fargate to reduce node management. The bill dropped to $1,600/month, but the cold start latency was unbearable. Requests that used to take 80ms now took 2.1 seconds on Fargate. Our frontend team had to add skeleton loaders everywhere, which hurt the user experience. I spent a week tweaking the Fargate profiles, only to realize that the problem wasn’t the cold starts—it was that Fargate was charging us for every second the task was running, even if it was idle.

None of these approaches worked. We were still paying too much, and we were sacrificing performance or reliability. It was time to look beyond AWS.

## The approach that worked

I started by listing our non-negotiables:
- Sub-second latency for API responses
- 99.9% uptime
- Automatic scaling during traffic spikes
- Cheaper than $3k/month
- No DevOps overhead

I looked at Fly.io, Render, and Railway. Fly.io’s pricing model stood out: you pay for the resources you actually use, not for idle capacity. They also offered Postgres, Redis, and static hosting in one place. The pricing model was simple: $5 for 1 shared-CPU VM with 256MB RAM, and $16 for a dedicated VM with 2 vCPUs and 4GB RAM. No cluster management, no node autoscaling, just deploy and forget.

I signed up for Fly.io in December 2026 and ran a pilot with our staging environment. The first surprise was how fast it was. Deploying a new version took 15 seconds, compared to 5 minutes on EKS. The second surprise was the cost. Our staging environment, which was running 2 m6i.large nodes on EKS, cost $420/month. On Fly.io, it cost $40/month. The performance was identical, but the cost was 10x lower.

I was skeptical. How could a platform that didn’t use Kubernetes be this stable? I dug into their architecture and found that they use Firecracker microVMs for isolation, which gives them the security and performance of a VM without the overhead of a full Kubernetes node. They also handle networking, load balancing, and scaling automatically. No more tweaking the cluster autoscaler or debugging Prometheus alerts.

The only catch was the database. Our main Postgres database was running on AWS RDS, costing $500/month. Fly.io offered managed Postgres for $25/month, but I wasn’t sure if it would handle our write-heavy workload. I tested it with a copy of our production database. The benchmark showed that writes were 15% slower, but reads were 20% faster. Given that our read-to-write ratio was 8:1, the trade-off was worth it.

By January 2026, we had a plan: move the API and frontend to Fly.io, move the database to Fly.io Postgres, and retire the EKS cluster. The only hard decision was whether to keep Redis in-memory or switch to Fly.io’s Redis offering. I ran a quick test: our Redis cache was handling 120,000 requests per minute, and the latency was 1.2ms. Fly.io’s Redis (version 7.2) had a latency of 2.3ms. I wasn’t willing to sacrifice that much performance, so we kept Redis on AWS ElastiCache for $120/month.

## Implementation details

The migration took 5 days. Here’s how we did it:

### Step 1: Containerize the API

Our API was already in a Docker container, but we had to tweak the Dockerfile to work with Fly.io. Fly.io uses a `fly.toml` file for configuration, similar to `docker-compose.yml`. Here’s the Dockerfile we ended up with:

```dockerfile
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

The key was setting `PYTHONUNBUFFERED=1` to avoid buffering logs, which Fly.io expects. We also had to pin the Python version to 3.11 to avoid surprises when Fly.io updated their runtime.

### Step 2: Configure Fly.io

We created a `fly.toml` file:

```toml
app = "our-api"
primary_region = "sin"

[build]
  dockerfile = "Dockerfile"

[http_service]
  internal_port = 8080
  force_https = true
  auto_stop_machines = true
  auto_start_machines = true
  min_machines_running = 2
  processes = ["app"]

[[vm]]
  memory = "1gb"
  cpu_kind = "shared"
  cpus = 2
```

A few things worth noting:
- `auto_stop_machines` and `auto_start_machines` are Fly.io’s way of handling scaling. When traffic drops, they stop the VMs to save cost. When traffic spikes, they start new VMs automatically. This is the opposite of Kubernetes’ approach, where nodes are always running.
- `min_machines_running = 2` ensures we always have at least 2 VMs running, which gives us a basic level of redundancy.
- We set the region to Singapore (`sin`) because most of our users are in Southeast Asia. Fly.io has regions in Cape Town and other locations, but Singapore was the closest to our user base.

### Step 3: Migrate the database

We used Fly.io’s managed Postgres service. The migration was straightforward:

1. Create a new Postgres cluster on Fly.io:
   ```bash
   fly postgres create --name our-db --region sin
   ```

2. Take a snapshot of our AWS RDS instance:
   ```bash
   aws rds create-db-snapshot --db-instance-identifier our-db --db-snapshot-identifier snapshot-2026-01-15
   ```

3. Restore the snapshot to Fly.io Postgres. Fly.io provides a CLI tool for this:
   ```bash
   fly postgres restore --app our-db --from-snapshot snapshot-2026-01-15
   ```

4. Update the API’s database connection string to point to the new Postgres cluster.

The migration took 2 hours, including testing. The only hiccup was that Fly.io’s Postgres runs on a shared VM by default, which caused some performance issues during the restore. We upgraded to a dedicated VM ($25/month) and the restore completed in 45 minutes.

### Step 4: Migrate Redis

We kept Redis on AWS ElastiCache for two reasons:
- Latency: ElastiCache had a latency of 1.2ms, while Fly.io’s Redis was 2.3ms.
- Cost: ElastiCache was $120/month, which was cheaper than upgrading to Fly.io’s Redis ($50/month) and dealing with the latency trade-off.

We updated the API’s Redis connection string to point to ElastiCache, and the migration was complete.

### Step 5: Deploy the frontend

Our frontend was a Next.js 14 app hosted on Vercel. We didn’t want to move it off Vercel because of their excellent edge network. Instead, we configured the frontend to call the API on Fly.io. This was a network hop, but the latency was still under 50ms for users in Southeast Asia.

We also set up a CDN on Cloudflare to cache static assets, which reduced the load on the frontend and improved performance.

### Step 6: Monitor and optimize

Fly.io provides a dashboard with metrics like CPU usage, memory, and request latency. We set up alerts for high latency and low memory. The default alert for latency was set to 500ms, which we lowered to 200ms after a few days of testing.

We also enabled Fly.io’s built-in logging, which streams logs to Papertrail. This was a lifesaver when debugging a memory leak in one of our API endpoints. The logs showed that the endpoint was leaking memory at a rate of 1MB per 100 requests. We fixed the leak by adding a `gc.collect()` call in Python, which reduced the memory usage from 500MB to 80MB per VM.

## Results — the numbers before and after

Here’s a breakdown of the costs and performance before and after the migration:

| Metric                | Before (EKS + RDS + ElastiCache) | After (Fly.io + Fly.io Postgres + ElastiCache) | Change          |
|-----------------------|----------------------------------|------------------------------------------------|-----------------|
| API hosting           | $3,180/month                    | $160/month                                    | -95%            |
| Database              | $500/month                      | $25/month                                     | -95%            |
| Redis                 | $120/month                      | $120/month                                    | 0%              |
| Frontend              | $50/month (Vercel)              | $50/month (Vercel)                            | 0%              |
| **Total**             | **$3,850/month**                | **$355/month**                                | **-91%**        |
| API latency (p95)     | 120ms                           | 95ms                                          | -21%            |
| API latency (p99)     | 350ms                           | 180ms                                         | -49%            |
| Database writes/sec   | 1,200                           | 1,100                                         | -8%             |
| Database reads/sec    | 9,600                           | 10,200                                        | +6%             |
| 99.9% uptime          | 99.8% (missed SLA once)         | 100%                                          | +0.2%           |
| Deployment time       | 5 minutes                       | 15 seconds                                    | -95%            |
| DevOps time per week  | 8 hours                         | 1 hour                                        | -87%            |

The most surprising result was the latency improvement. Even though we moved from Kubernetes to a shared VM, the p99 latency dropped from 350ms to 180ms. The reason was that Fly.io’s networking stack is optimized for low-latency traffic, and we no longer had the overhead of Kubernetes’ iptables rules and CNI plugins.

The cost savings were even more dramatic. We went from $3,850/month to $355/month, a 91% reduction. The savings allowed us to hire a part-time designer, which improved our conversion rate by 12%.

The only downside was the database migration. Fly.io’s Postgres was slower for writes, but since our workload was read-heavy, the impact was minimal. If we had a write-heavy workload, we might have stuck with AWS RDS.

## What we’d do differently

If we had to do the migration again, we would have made a few changes:

1. **Test Fly.io’s Postgres with a write-heavy workload first.** We assumed our workload was read-heavy because of our API’s endpoints, but our analytics showed that 30% of our database traffic was writes. A quick benchmark with `pgbench` would have shown the performance difference before we migrated.

2. **Use Fly.io’s Redis from the start.** We kept Redis on ElastiCache because of the latency difference, but Fly.io’s Redis is now on dedicated VMs and the latency is comparable. The cost would have been $50/month instead of $120/month.

3. **Set up a staging environment on Fly.io earlier.** We tested the migration on a copy of our production data, but we didn’t have a staging environment that mirrored our production setup. This led to a few surprises when we deployed to production, like the memory leak we mentioned earlier.

4. **Monitor the `auto_stop_machines` setting more closely.** Fly.io stops VMs when traffic drops to save cost, but if you have a background job that runs every hour, it might wake up a VM that was stopped, causing a cold start. We had to adjust the `min_machines_running` setting to 2 to avoid this.

5. **Use Fly.io’s built-in secrets management from the start.** We initially stored secrets in environment variables in the `fly.toml` file, which is not secure. Fly.io recommends using their secrets management tool:
   ```bash
   flyctl secrets set API_KEY=secret_value
   ```
   This encrypts the secrets at rest and in transit.

The biggest lesson was that we over-optimized for Kubernetes when we didn’t need it. We spent weeks tweaking the cluster autoscaler, only to realize that Fly.io’s built-in scaling was simpler and cheaper. The trade-off was giving up some control, but for a small team, that control wasn’t worth the cost.

## The broader lesson

The lesson here isn’t that Kubernetes is bad. It’s that you should only use Kubernetes if you need its features. For most small teams, the overhead of managing a cluster isn’t worth the cost savings from fine-grained control. Kubernetes is a tool for teams that have dedicated DevOps engineers and need to scale to millions of users. For everyone else, a platform like Fly.io, Render, or Railway is a better fit.

The second lesson is that you should measure before you migrate. We assumed that Fly.io’s Postgres would be slower for writes, but we didn’t benchmark it. If we had run a quick `pgbench` test with realistic write load, we might have avoided the surprise. Always test your assumptions with data.

The third lesson is that latency isn’t just about your code. It’s about the entire stack, from the database to the CDN to the user’s ISP. Moving to Fly.io improved our latency not because our code got faster, but because Fly.io’s networking stack is optimized for low-latency traffic.

Finally, the cost of DevOps isn’t just the salary of a DevOps engineer. It’s the time you spend debugging infrastructure instead of building features. We saved 7 hours per week by moving off Kubernetes, which added up to 364 hours per year. At $100/hour (our blended cost), that’s $36,400 per year in saved time. That’s a hard number to ignore.

## How to apply this to your situation

If you’re running a SaaS product and your infrastructure bill is creeping up, here’s a checklist to see if you can move off Kubernetes:

1. **Check your traffic pattern.** If you have predictable traffic, you don’t need Kubernetes’ autoscaling. A platform like Fly.io or Render will work fine.
2. **Measure your latency requirements.** If you need sub-100ms latency, test Fly.io’s networking stack with a real workload. Don’t assume it’s slower.
3. **List your non-negotiables.** If you need a specific database feature (like AWS Aurora’s parallel query), you might be stuck with Kubernetes or a managed database service.
4. **Calculate the real cost of DevOps.** Add up the time you spend on infrastructure per week, and multiply by your hourly rate. Compare that to the cost of a managed platform like Fly.io.
5. **Start with a non-critical service.** Migrate a staging environment or a secondary service first. If it works, migrate the rest.

If your workload is read-heavy, like ours, Fly.io is a great fit. If you have a write-heavy workload, test Fly.io’s Postgres with realistic traffic before committing. If you need GPU instances or specific hardware, Kubernetes might still be the best option.

For most small teams, the decision comes down to this: do you want to spend time building features, or do you want to spend time tweaking YAML files? If it’s the former, move to a managed platform.

## Resources that helped

- [Fly.io documentation](https://fly.io/docs/) — The official docs are well-written and to the point. The `fly.toml` reference was especially helpful.
- [Fly.io Postgres benchmarks](https://fly.io/docs/postgres/benchmarks/) — These benchmarks helped us understand the performance trade-offs of Fly.io’s Postgres.
- [pgbench](https://www.postgresql.org/docs/current/pgbench.html) — We used this to benchmark our database workload before migrating.
- [FastAPI + Fly.io tutorial](https://fly.io/docs/python/start-python/) — A step-by-step guide to deploying a FastAPI app on Fly.io.
- [Docker multi-stage builds](https://docs.docker.com/build/building/multi-stage/) — We used this to optimize our Docker image size and build time.
- [Cloudflare CDN](https://www.cloudflare.com/cdn/) — We used Cloudflare to cache static assets and reduce the load on our frontend.

## Frequently Asked Questions

**How do I know if my workload is a good fit for Fly.io?**

Fly.io works well for stateless services like APIs, frontends, and background workers. If your workload is stateful (like a database), you’ll need to use Fly.io’s managed services or keep your database elsewhere. A good rule of thumb is: if you can containerize it, Fly.io can run it. If you need persistent storage that isn’t managed by Fly.io, you might need to stick with Kubernetes or a managed database service.

**What’s the catch with Fly.io’s auto-stop feature?**

Fly.io stops VMs when traffic drops to save cost, which can cause cold starts. If you have a background job that runs every hour, it might wake up a stopped VM, causing a latency spike. To avoid this, set `min_machines_running` in your `fly.toml` file to a value higher than 0. For example, `min_machines_running = 2` ensures you always have 2 VMs running, even if traffic is low.

**How does Fly.io compare to Render or Railway for cost?**

Fly.io is cheaper for compute-heavy workloads because they charge per VM instead of per vCPU. Render and Railway charge per vCPU-hour, which can add up if you have idle resources. For example, a 2 vCPU, 4GB RAM VM on Fly.io costs $16/month, while the same VM on Render costs $25/month. If you have 10 VMs, that’s a $110/month difference. However, Render and Railway have simpler pricing models and better support for databases, so they might be a better fit if you need managed services.

**What happens if Fly.io has an outage?**

Fly.io has a 99.95% SLA for their control plane, but they don’t guarantee uptime for your VMs. If Fly.io has an outage, your VMs might restart or become unavailable. To mitigate this, you can:
- Deploy your app to multiple regions (Fly.io supports this, but it’s more expensive).
- Use a health check endpoint that Fly.io will call to restart your VM if it’s unhealthy.
- Keep a backup of your app running on another platform (like AWS Lightsail or DigitalOcean) as a fallback.

Fly.io’s status page is [status.fly.io](https://status.fly.io), and they’re pretty transparent about outages. In the 6 months we’ve been using Fly.io, they’ve had one major outage (lasting 2 hours) and a few minor ones.

**Can I use Fly.io with AWS or GCP services?**

Yes. You can connect Fly.io to AWS RDS, GCP Cloud SQL, or any other external service. We kept our Redis on AWS ElastiCache and our Postgres on Fly.io Postgres, and it worked fine. The only caveat is that you’ll pay for the data transfer between Fly.io and the external service, which can add up if you have a lot of traffic. For example, if your API on Fly.io makes 100,000 requests to AWS RDS per minute, you’ll pay for the egress traffic from AWS to Fly.io.

**How do I debug a memory leak on Fly.io?**

Fly.io provides built-in logging via Papertrail. If you suspect a memory leak, check the logs for errors like `MemoryError` or `Out of memory`. You can also SSH into your VM to debug:
```bash
flyctl ssh console
```
Once inside, use tools like `top`, `htop`, or `ps` to check memory usage. If you find a leak, fix it in your code and redeploy. Fly.io also provides metrics for memory usage in their dashboard, which can help you identify leaks early.

**What’s the hardest part of migrating to Fly.io?**

The hardest part is letting go of control. With Kubernetes, you can tweak every knob—from the CNI plugin to the pod disruption budget. With Fly.io, you give up some of that control in exchange for simplicity. For example, you can’t customize the kernel parameters, and you can’t choose your CNI plugin. If you need that level of control, Fly.io isn’t for you. But for most small teams, it’s a worthwhile trade-off.

## Next step

Open your `docker-compose.yml` or `Dockerfile` and check the base image. If it’s Alpine Linux, switch to a slim Debian or Ubuntu image. Alpine’s package manager can cause unexpected issues with Python packages, and Debian’s larger base image rarely causes problems. Then, run `docker build --no-cache` and check the size. If it’s over 500MB, you’re likely using a bloated image. Switch to a multi-stage build to reduce the size. This will take 15 minutes and could save you hours of debugging later.


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

**Last reviewed:** June 26, 2026
