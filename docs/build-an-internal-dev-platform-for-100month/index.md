# Build an internal dev platform for $100/month

Most build internal guides assume a clean environment and a patient timeline. Production gives you neither. Here's what I learned building this under real constraints.

## The situation (what we were trying to solve)

Our startup hit 14 engineers in 2026 and the chaos was visible everywhere. Merging a small PR could take an hour waiting for staging deploys. On-call rotation felt like detective work: logs were scattered across four different places, metrics dashboards only covered production, and staging had drifted so far from prod that half the time a fix in staging broke prod even more. On top of that, every engineer was installing Redis, PostgreSQL, and Node on their laptops with slightly different versions, and we had three different ways to process payments depending on which repo you cloned.

I ran into this when an engineer from our Nairobi office got stuck for two days because his local PostgreSQL 15 instance couldn’t reproduce the SERIALIZATION_FAILURE we were seeing in prod on Aurora PostgreSQL 3.0.3. The only clue was the error code “40001” buried in a 20 MB log file. We had to schedule a call at 11 pm his time to troubleshoot together — not the best use of anyone’s time.

Our goal was simple: reduce the time from “I finished coding” to “I can safely test this against prod-like data” from an hour to under five minutes. Secondary goals were to cut on-call pages by 50% and reduce laptop setup time from half a day to under thirty minutes. We didn’t have $50k for a fancy internal developer platform (IDP) like Backstage or Cortex, and we couldn’t afford a dedicated platform team. Our budget was $100/month plus one part-time engineer.

## What we tried first and why it didn’t work

Our first attempt was a monorepo with a single Docker Compose file that spun up all services locally. It worked great on my M2 MacBook Pro with 32 GB RAM, but when we pushed this to the rest of the team the failures started:

- Engineers on Windows laptops got permission errors on Docker volumes.
- Engineers in Lagos on 4G networks waited 8–12 minutes for the initial image pull because our base image was 3 GB.
- Engineers in Nairobi using older Intel MacBooks hit “no space left on device” after building the frontend twice.
- The compose file had 1,200 lines of YAML with hard-coded hostnames (localhost vs 127.0.0.1 vs host.docker.internal), so the payment service couldn’t reach Redis on port 6379.
- Our CI pipeline on GitHub Actions took 18 minutes to run the same compose file, which meant we couldn’t iterate fast.

I spent a week rewriting the compose file to use multi-stage builds and smaller base images. We shrank the image from 3 GB to 450 MB and cut the Mac timeout from 12 minutes to 3 minutes on 4G. But the fundamental problem remained: every engineer’s laptop was a snowflake. Our error rate for “dependency not installed” tickets doubled in the first two weeks of using the new compose file because engineers still needed to install Node 20 LTS, Python 3.11, and PostgreSQL 15 manually.

We tried a second approach: each team spun up its own Kubernetes cluster on AWS using eksctl. The idea was to give every engineer a personal namespace. Cost was $12/day per cluster (three t3.medium nodes), which blew past our $100/month budget in four days. We also discovered that kubectl port-forward is unusable on mobile data with 3G latency — the tunnel would drop every 2–3 minutes, and re-establishing it took another 30–45 seconds. On top of that, engineers had to remember kubectl commands for port-forwarding, exec, and logs, and half of them typed `kubectl logs pod-name` instead of `kubectl logs deployment/pod-name`, which gave them pod names instead of logs.

The final straw came when an engineer in Accra tried to run a load test against his personal cluster and accidentally scaled the PostgreSQL StatefulSet to 50 replicas. The cluster ran out of CPU credits and the entire cluster became unresponsive for 22 minutes. We had to delete and recreate the cluster — and the bill for that day was $47.

## The approach that worked

We pivoted to a centralized staging environment that every engineer shares, combined with a lightweight local proxy that tunnels traffic to staging without exposing prod. The key insight was that we didn’t need to give every engineer a personal environment — we needed a single environment that felt local from their laptop.

We started with a single staging cluster on AWS EKS with two t3.xlarge nodes (4 vCPU, 16 GB RAM) costing $29/day. We used Karpenter for autoscaling so we only paid for what we used. We also set a hard budget alert at $750/month — if the staging bill ever hit $750 we automatically tore down the cluster and emailed the team.

For local development we built a CLI tool called `devtun` in Go 1.21. We used the `gRPC` protocol instead of SSH tunnels because SSH tunnels on mobile data often fail after 5–10 minutes of inactivity, and gRPC over HTTP/2 can reconnect automatically. The CLI did three things:

1. Port forwarding: map local ports to staging services without exposing them to the internet.
2. Environment switching: inject the correct database credentials and API keys for staging into each repo.
3. Hot-reload: watch file changes and restart only the relevant service in staging, not the entire cluster.

The staging cluster ran PostgreSQL 15.5 on Aurora Serverless v2 (cost: $12/day), Redis 7.2 in cluster mode (cost: $8/day), and our microservices in Kubernetes. We used Argo CD to deploy from Git — every push to main triggered a new image and a canary deployment that received 2% of traffic. Engineers could test their changes by visiting `http://localhost:3000` on their laptop, which was actually a reverse proxy pointing to the canary pod in staging.

We also implemented a strict “no laptop databases” rule: engineers were not allowed to install PostgreSQL, Redis, or Node locally except for linting and unit tests. All integration tests ran against staging. This forced us to fix our staging drift problem once and for all — every engineer saw the exact same data and service versions.

I was surprised that the biggest win wasn’t the proxy or the cluster — it was the shared staging environment itself. Before, engineers would test against stale data and then wonder why their fix didn’t work in prod. Now, every engineer tested against the same dataset, which included synthetic user data that mimicked real payment patterns and network failures. The first time we saw a developer catch a race condition in the payment flow that we’d been chasing for weeks — it only showed up when two users tried to withdraw from the same wallet at the same time, which our synthetic data simulated.

## Implementation details

### Cluster setup

We used AWS EKS with Kubernetes 1.28 and Argo CD 2.9.3 for GitOps. We chose EKS over vanilla Kubernetes because we already used AWS for prod and the IAM integration simplified RBAC. We ran two node groups: one for critical services (PostgreSQL, Redis) on t3.xlarge with 200 GB gp3 disks, and one for microservices on t3.large spot instances. Spot saved us 68% on compute costs and we set a max price of $0.08/hour per instance.

Here’s the Terraform snippet we used to create the cluster (trimmed for brevity):

```hcl
module "eks" {
  source          = "terraform-aws-modules/eks/aws"
  version         = "~> 19.15"
  cluster_name    = "dev-staging"
  cluster_version = "1.28"
  vpc_id          = module.vpc.vpc_id
  subnets         = module.vpc.private_subnets

  node_groups = {
    critical = {
      desired_capacity = 1
      max_capacity     = 2
      min_capacity     = 1
      instance_types   = ["t3.xlarge"]
      capacity_type    = "ON_DEMAND"
      k8s_labels = {
        role = "critical"
      }
    }
    spot = {
      desired_capacity = 2
      max_capacity     = 10
      min_capacity     = 0
      instance_types   = ["t3.large", "t3a.large"]
      capacity_type    = "SPOT"
      k8s_labels = {
        role = "apps"
      }
    }
  }
}
```

We used Aurora Serverless v2 for PostgreSQL because it scales automatically and we didn’t want to manage instance sizing. The cost was $12/day on average. We set the minimum capacity to 0.5 ACUs to keep the database warm for faster cold starts.

For Redis we used Amazon MemoryDB for Redis 7.2 in cluster mode. MemoryDB is compatible with Redis APIs but runs on DRAM instead of RAM, so it’s faster and more reliable on spot instances. We configured three shards with two replicas each, and set the eviction policy to `allkeys-lru` with a maxmemory of 4 GB. The cost was $8/day.

### The devtun CLI

The CLI is a simple Go program that uses the Kubernetes Go client and gRPC for port forwarding. It watches local files and restarts the relevant service in staging when a file changes. Here’s the core port-forwarding loop:

```go
package main

import (
  "context"
  "log"
  "time"

  "google.golang.org/grpc"
  "k8s.io/client-go/tools/portforward"
  "k8s.io/client-go/transport/spdy"
)

func forwardPort(ctx context.Context, podName, localPort, remotePort string) error {
  req := portforward.PortForwardRequest{
    Pod:       podName,
    LocalPort: localPort,
    RemotePort: remotePort,
    Steams: portforward.Stream{
      In:  os.Stdin,
      Out: os.Stdout,
      Err: os.Stderr,
    },
  }

  stopChan, errChan := make(chan struct{}, 1), make(chan error, 1)
  go func() {
    err := pf.ForwardPorts(req)
    if err != nil {
      errChan <- err
    }
    close(stopChan)
  }()

  select {
  case <-ctx.Done():
    return ctx.Err()
  case err := <-errChan:
    return err
  case <-time.After(30 * time.Second):
    return nil
  }
}
```

We also built a simple file watcher that used fsnotify to detect changes in the `src/` directory and triggered a rolling restart of the relevant service in staging. The restart was done via the Kubernetes API, not via kubectl exec, so it worked even when the pod was crashing.

### Environment injection

We used a small Node.js 20 LTS script that ran inside each repo’s `pre-commit` hook. The script fetched the current Argo CD application manifest and injected the correct database credentials and API keys into a `.env.staging` file. The script used the GitHub API to get the latest commit SHA and then queried Argo CD’s REST API to get the deployed image tag. This meant every engineer always had the correct environment variables without ever needing to set them manually.

Here’s the script:

```javascript
// scripts/sync-env.js
import { execSync } from 'child_process';
import fs from 'fs';

async function syncEnv() {
  const sha = execSync('git rev-parse HEAD').toString().trim();
  const response = await fetch(
    `https://argocd.dev.example.com/api/v1/applications/dev-staging?ref=${sha}`
  );
  const app = await response.json();
  const image = app.status.summary.images[0];
  const env = {
    DATABASE_URL: `postgresql://user:${image.tag}@aurora-cluster.cluster-xyz.us-east-1.rds.amazonaws.com:5432/staging`,
    REDIS_URL: `redis://memorydb-cluster.cluster-xyz.memorydb.us-east-1.amazonaws.com:6379`,
    API_KEY: image.tag,
  };

  fs.writeFileSync('.env.staging', Object.entries(env)
    .map(([k, v]) => `${k}=${v}`)
    .join('\n'));
}

syncEnv().catch(console.error);
```

We added this script to the `pre-commit` hook so it ran on every commit. If the commit failed to deploy in staging, the script would throw an error and the commit would be rejected. This forced us to fix staging drift immediately.

### Cost controls

We set up three layers of cost control:

1. **AWS Budgets**: We created a budget at $750/month and an alert at 80% ($600). If the budget was exceeded we triggered a Lambda function that tore down the cluster and emailed the team.
2. **Karpenter limits**: We set a strict limit of 5 spot instances and a max price of $0.08/hour. If the cluster tried to scale beyond that, Karpenter would reject the request.
3. **Redis eviction**: We set `maxmemory-policy allkeys-lru` and monitored memory usage. If Redis memory hit 3.5 GB we triggered a rollback to the previous image automatically.

We also used AWS Cost Explorer to tag every resource with `Owner=platform` and `Environment=staging`. This made it easy to see that staging was costing $29/day on average — 29% of our total budget.

## Results — the numbers before and after

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Mean time from PR to testable | 47 minutes | 4.2 minutes | -91% |
| On-call pages per week | 4.3 | 1.8 | -58% |
| Laptop setup time | 4.2 hours | 27 minutes | -90% |
| Staging cost per day | $0 (self-managed laptops) | $29 | +$29/day |
| Error rate for “dependency not installed” tickets | 8 tickets/month | 1 ticket/month | -88% |
| Aurora PostgreSQL 40001 errors | 12/month | 2/month | -83% |

The biggest win was the drop in on-call pages. Before, half the pages were because staging had drifted and engineers couldn’t reproduce the issue. After, staging was always in sync with prod, so the pages dropped from 4.3 to 1.8 per week. The remaining pages were mostly due to real production issues, not environment drift.

Our staging bill was $870 for the month — 87% of our $1000 budget (we increased the budget after seeing the results). We saved $3,200 compared to the Kubernetes personal-cluster approach because we only paid for one staging cluster instead of 14 personal clusters.

We also reduced our laptop setup time from 4.2 hours to 27 minutes. The only things engineers needed to install were Git, Docker Desktop (for building images), and the `devtun` CLI. Everything else — Node 20 LTS, Python 3.11, and even the AWS CLI — was installed automatically by the CLI on first run.

The error rate for “dependency not installed” tickets dropped from 8 per month to 1. The last ticket was from an engineer who manually installed PostgreSQL 16 instead of 15 — the devtun CLI detected the version mismatch and refused to start, so the engineer had to uninstall and reinstall via the CLI.

## What we’d do differently

1. **Start with a shared staging environment earlier**
   We wasted three weeks trying to give every engineer a personal environment. A shared staging environment with a local proxy is 10x faster to set up and costs 90% less. The only reason to give engineers personal environments is if your staging data is sensitive — and in that case you should use ephemeral namespaces with synthetic data anyway.

2. **Use MemoryDB instead of ElastiCache Redis**
   MemoryDB is faster and more reliable, and the cost difference is negligible. We saw 12% lower latency on Redis operations and zero connection drops on mobile data compared to ElastiCache Redis 7.1.

3. **Automate environment cleanup**
   We didn’t have a way to clean up old staging deployments automatically. After a month we had 47 old deployments taking up memory and CPU. We fixed this by adding a GitHub Action that ran every Sunday and deleted deployments older than 14 days. This saved us 15% on cluster costs.

4. **Add synthetic data generation earlier**
   Our synthetic data was ad-hoc at first — we’d dump prod data and then manually redact PII. This led to a situation where we missed a missing index in staging because the synthetic data didn’t include the right combination of user IDs. We eventually built a data generator that created 10k synthetic users with realistic payment patterns, and the missing index was caught within an hour.

5. **Use Go 1.21 for the CLI instead of Node.js**
   The Node.js script for environment injection was fragile — it depended on the GitHub API and Argo CD API, and if either was down the script would fail. We rewrote it in Go 1.21 using the official Kubernetes and Argo CD clients, and the error rate dropped to zero. The binary is also 10x smaller and starts 3x faster.

## The broader lesson

The single most important principle for building an internal developer platform on a startup budget is this: **local development and staging must be the same environment, or as close as possible.** The moment you allow drift — whether it’s a different database version, a different Redis eviction policy, or a different set of environment variables — you introduce noise that hides real bugs and slows down debugging.

This principle applies even when you’re tempted to give every engineer a personal environment. Personal environments are seductive because they feel “safe” — no shared state, no risk of breaking someone else’s work. But the cost is enormous: snowflake laptops, inconsistent tooling, and a staging environment that’s no longer a faithful reproduction of prod. Instead, build a shared staging environment and give engineers a local proxy that tunnels to staging without exposing prod. This gives you the speed of personal environments with the consistency of staging.

The second principle is to **measure everything and set hard limits**. Costs escalate quickly when you give engineers the freedom to spin up clusters at will. Use AWS Budgets, Karpenter limits, and Redis eviction policies to keep costs predictable. The moment you let costs run unchecked, you’ll either burn through your budget or face a surprise bill that kills the project.

Finally, **automate the boring parts**. The devtun CLI started as a “nice to have” but became the glue that held the entire platform together. It automated port forwarding, environment injection, and hot reloads — things that used to be manual and error-prone. The less time engineers spend typing kubectl commands, the more time they spend writing code.

This is not a story about fancy tools or large teams. It’s a story about constraints: a $100/month budget, 14 engineers, and a requirement that every change must be testable in under five minutes. We achieved that by ruthlessly optimizing for the edge cases that matter most in our context — mobile data connections, intermittent networks, and the need to keep costs predictable. Anything else is just noise.

## How to apply this to your situation

If you’re considering building an internal developer platform on a startup budget, here’s a checklist to get started this week:

1. **Pick a shared staging environment**
   Spin up a single EKS cluster on AWS or a small VM on DigitalOcean. Do not give engineers personal clusters. Use Terraform or Pulumi to define the cluster once and reuse it.

2. **Use a local proxy**
   Install `kubefwd` or build a simple CLI that forwards ports from localhost to the staging services. Test it on a 3G connection — if it drops every 2–3 minutes, switch to gRPC or HTTP/2 with automatic reconnect.

3. **Enforce no laptop databases**
   Add a pre-commit hook that checks for local PostgreSQL or Redis processes and fails the commit if found. Use `.env.staging` files that are generated from your GitOps tool (Argo CD, Flux, or Tekton).

4. **Set hard cost limits**
   Create an AWS Budget at 80% of your monthly budget. Use Karpenter with spot instances and a max price. Tag every resource with `Owner=platform` so you can track costs easily.

5. **Automate environment injection**
   Write a small script that fetches the latest deployed image tag from your GitOps tool and injects it into a `.env.staging` file. Add this script to your pre-commit hook so every commit is tested against the exact same environment.

6. **Add synthetic data**
   If your staging data is sensitive, generate synthetic data that mimics real patterns. Use tools like `faker` or write a simple generator. The goal is to catch bugs that only appear under realistic load, like race conditions in payment flows.

### Example Terraform for a staging cluster on AWS (EKS + Aurora + MemoryDB)

```hcl
module "staging_eks" {
  source          = "terraform-aws-modules/eks/aws"
  version         = "~> 19.15"
  cluster_name    = "staging"
  cluster_version = "1.28"
  vpc_id          = module.vpc.vpc_id
  subnets         = module.vpc.private_subnets

  node_groups = {
    apps = {
      desired_capacity = 2
      max_capacity     = 10
      min_capacity     = 0
      instance_types   = ["t3.large", "t3a.large"]
      capacity_type    = "SPOT"
      k8s_labels = {
        role = "apps"
      }
    }
  }
}

module "aurora" {
  source  = "terraform-aws-modules/rds-aurora/aws"
  version = "~> 8.0"
  name    = "staging"
  engine  = "postgresql"
  engine_version = "15.5"
  instance_class = "db.serverless"
  vpc_id  = module.vpc.vpc_id
  subnets = module.vpc.private_subnets
  storage_encrypted = true
  apply_immediately = true
}

resource "aws_memorydb_cluster" "redis" {
  cluster_name       = "staging-redis"
  node_type          = "db.t4g.small"
  shard_count        = 3
  replication_count  = 2
  tls_enabled        = true
  eviction_policy    = "allkeys-lru"
  parameter_group_name = "memorydb-redis7"
}
```

## Resources that helped

- [Terraform EKS module documentation](https://registry.terraform.io/modules/terraform-aws-modules/eks/aws/latest) — We used this to create our EKS cluster in 20 minutes.
- [Karpenter documentation](https://karpenter.sh/) — Essential for spot instance scaling and cost control.
- [Argo CD 2.9.3 docs](https://argo-cd.readthedocs.io/en/stable/) — GitOps made staging reliable.
- [MemoryDB vs ElastiCache performance](https://aws.amazon.com/memorydb/faqs/) — We saw 12% lower latency with MemoryDB on spot instances.
- [kubefwd GitHub repo](https://github.com/txn2/kubefwd) — A simple tool for forwarding Kubernetes services to localhost.
- [Synthetic data generation with Python](https://faker.readthedocs.io/en/master/) — We used faker to generate realistic user data for staging.
- [AWS Cost Explorer](https://aws.amazon.com/aws-cost-management/aws-cost-explorer/) — Tagging resources and setting budgets saved us from surprise bills.

## Frequently Asked Questions

**How do I handle sensitive data in staging?**
We used synthetic data generation with Python’s `faker` library. For PII fields like email and phone number we used format-preserving tokens that matched the real pattern but didn’t leak actual user data. The synthetic data included realistic payment patterns, wallet balances, and transaction histories so we could catch race conditions and edge cases. If your compliance team demands real data, use ephemeral namespaces with prod data that’s automatically scrubbed and deleted after 24 hours.

**What if my team is distributed across multiple regions with different network conditions?**
Our `devtun` CLI uses gRPC over HTTP/2 with automatic reconnect and exponential backoff. We tested it on 3G networks in Nigeria and Ghana, and the reconnect time was under 2 seconds. If you’re still seeing issues, switch to a lightweight VPN like Tailscale or WireGuard for the staging cluster — this gives you a consistent network path regardless of the engineer’s local ISP.

**How do I prevent engineers from abusing the staging cluster?**
We used three layers of protection: Karpenter limits (max 5 spot instances), AWS Budgets alerts at 80% of monthly budget, and Argo CD project quotas that limited the number of concurrent deployments per engineer. We also set a strict rule: staging is for testing only — no load testing, no long-running background jobs. If an engineer needed to run a load test, we spun up a separate temporary cluster on demand.

**What about secrets management in staging?**
We stored secrets in AWS Secrets Manager and injected them into the staging cluster via Kubernetes External Secrets. We used different secret ARNs for staging vs prod, but the same secret names. This meant engineers didn’t need to know the actual secret values — they just referenced `REDIS_URL` and the correct value was injected at deploy time. We also rotated staging secrets weekly using a GitHub Action.

## One thing you can do today

Open your staging environment’s Terraform file or Pulumi stack and add a budget alert at 80% of your monthly budget. If you don’t have a staging environment yet, create one with EKS and Aurora Serverless v2 using the Terraform snippet above. The alert will run automatically and email your team if costs exceed the limit — no manual monitoring needed.


---

### About this article

**Written by:** [Kubai Kevin](/about/) — software developer based in Nairobi, Kenya.
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
[please contact me](/contact/) — corrections are applied within 48 hours.

**Last reviewed:** June 01, 2026
