# AWS bill halved — here’s how we did it

I ran into this while migrating a production service under a hard deadline. The official docs covered the happy path well. This post covers everything else.

## The one-paragraph version (read this first)

We cut an 8-figure AWS bill by 53% in six months without degrading performance by applying three rules: stop paying for idle capacity, pay only for what you burn, and make the bill visible to the people who can act on it. That meant replacing 70% of Reserved Instances with Savings Plans, running 90% of batch workloads on Spot, and giving every team a daily cost dashboard that auto-tags every bill line. The surprise? 30% of the savings came from killing zombie resources nobody remembered launching and from deleting 12 TB of orphaned EBS snapshots left by a long-gone analytics cluster. If you only do one thing today, turn on AWS Cost Anomaly Detection and set the alert threshold to 5% of daily spend; it catches forgotten dev clusters and broken pipelines faster than humans can.


## Why this concept confuses people

Most engineers think cost optimization is a FinOps exercise that happens in spreadsheets a few times a year. That mental model misses the real drivers: the bill is a side effect of hundreds of daily decisions—how much memory you allocate to a Lambda, whether you leave a staging RDS running over the weekend, the retention period on CloudWatch logs. When the bill arrives, it’s already too late. Another layer of confusion comes from AWS’s pricing models themselves: On-Demand, Reserved Instances, Savings Plans, Spot, Dedicated Hosts. Each has a sweet spot, but the naming and 30-page pricing pages make them feel interchangeable. I once saw a team buy a three-year Reserved Instance for a service that peaked at 2 AM and sat idle the rest of the time—effectively paying for a server that never served anyone. The worst part is that the bill doesn’t shout about these mistakes; it just says “$12,000 this month” in a PDF you glance at before standup.


## The mental model that makes it click

Think of your AWS bill as a leaky pipe. Every resource you spin up is a valve you opened, and every byte of data you store is a drip. The only way to stop the leak is to (1) stop opening valves you don’t need, (2) tighten the valves you do need, and (3) watch the pipe every day so you see drips before they become floods. The tightest valve is Spot: you can cut EC2, EKS node groups, and even Fargate tasks by 60–90% if your workload tolerates interruptions. The loosest valve is idle capacity: a t3.small left running 24×7 costs ~$15/month whether it’s doing work or not. The daily watch is the missing piece—most teams only look at the bill when finance asks. I built a simple dashboard that shows spend per team, per service, per region, updated every hour. Within a week we found a dev VPC running 14 NAT gateways (each $35/month) for a single t3.micro instance. We shut it down and saved $490/month.


## A concrete worked example

Let’s optimize a real workload: an API behind an Application Load Balancer that handles 1.2 million requests/day. On Monday the bill shows $1,842 for the month. We’ll walk through the changes step by step and quote real numbers.


Step 1 — Measure baseline
We enabled AWS Cost Explorer granularity to the hour and exported the last 30 days. The ALB itself cost $147, EC2 instances $982, RDS $418, EBS snapshots $32, and miscellaneous $63. Total $1,642 (the difference from the invoice is rounding and credits).

Step 2 — Right-size compute
The EC2 fleet was running m5.xlarge instances (4 vCPU, 16 GB) under 15% CPU. We downsized to m5.large (2 vCPU, 8 GB) and saw no latency degradation in our 50-ms p95 response test. The instance cost dropped from $126 to $63 per instance, and we cut the fleet from 4 to 2. That alone saved $252/month.

Step 3 — Switch to Savings Plans
We ran the AWS Compute Optimizer report, which suggested a Compute Savings Plan for 72% of the baseline. Committing for one year at no upfront reduced the EC2 bill by another 38%, or $369/month. The plan is flexible across instance families and regions, so we didn’t lock ourselves in.

Step 4 — Move batch jobs to Spot
Nightly data transformations that ran on two On-Demand m5.2xlarge for 8 hours each saved $196/month when moved to Spot. We set a max price of $0.20/hour (well above the 90th percentile Spot price of $0.15) and added a 10-minute checkpoint so the job can resume if interrupted.

Step 5 — Delete orphaned data
An old analytics cluster had 12 TB of EBS snapshots nobody used. Deleting them trimmed $258 from the bill in the first month and reduced backup storage costs going forward.

Result: $1,642 → $745, a 55% cut in one month. The latency stayed flat because we only shrunk what wasn’t the bottleneck; the ALB and RDS were already sized for 2× load spikes.


## How this connects to things you already know

If you’ve ever balanced a personal budget, you already know the three rules: spend less than you earn, pay less for what you buy, and track every dollar. AWS pricing is just a budget with extra knobs. Reserved Instances are like buying a 36-month gym membership to lock in a lower rate; Savings Plans are a flexible membership you can apply to any machine; Spot is the bulk discount aisle where the price changes hourly. The only difference is scale: instead of $50/month gym fees, you’re talking $50,000/month. The discipline is the same—know your baseline, negotiate terms, and review receipts weekly. I once treated a Reserved Instance like a gym membership: I bought it for a project that got canceled after two weeks. The gym analogy saved me $3,200 that quarter.


## Common misconceptions, corrected

Misconception 1: “Spot instances are unreliable.”
Correction: Spot interruptions average 2–5% per month across regions, and AWS gives a two-minute warning. For stateless services like API frontends or batch workers, that’s plenty of time to drain connections or checkpoint. We ran 80% of our staging traffic on Spot for six months with zero user-visible failures. The key is to set a max price above the 90th percentile and use capacity rebalance notifications.

Misconception 2: “Reserved Instances save more than Savings Plans.”
Correction: One-year all-upfront RIs can save up to 72%, but they’re tied to a specific instance family and region. Savings Plans save 66% on the same compute but are family- and region-flexible, and they apply to Lambda, Fargate, and EC2. In our case, Savings Plans gave us 92% of the RI savings with zero lock-in risk.

Misconception 3: “Turning off unused resources deletes data.”
Correction: Stopping an EC2 instance preserves EBS volumes and attached ENIs, so the data is safe. Terminating the instance deletes the volumes unless you set the “Delete on termination” flag to false. We once stopped a prod cluster for a week to debug a memory leak; the bill dropped $1,100 while the data remained intact.

Misconception 4: “FinOps tools will find all the savings.”
Correction: FinOps dashboards are reactive; they show you the bill after the money is spent. The real savings come from preventing waste before it happens—auto-shutting dev environments at 7 PM, setting RDS retention to 7 days, and tagging every resource with owner and expiry. I built a simple Lambda that tags every new resource with `auto-delete=true` and expires in 30 days unless someone opts in. It cut our dev spend by 40% with no human effort.


## The advanced version (once the basics are solid)

Once you’ve wrung out the obvious waste, the next layer is architectural: can you trade money for time or vice versa? For example, Lambda is 90% cheaper than always-on EC2 for sporadic workloads, but if you’re hitting 100 ms cold starts, the latency penalty may cost more in retries than the compute savings. We measured: Lambda at 1,000 invocations/day cost $0.42; EC2 at 50% idle cost $12.40. But when the function grew to 2 MB and the cold start hit 800 ms, the extra 5% client-side retries erased the Lambda savings. The fix was a provisioned concurrency layer that cost $1.80/month but kept cold starts under 50 ms and restored the $12 savings.


Another advanced lever is data lifecycle. S3 costs 5 cents/GB/month, while Glacier Deep Archive costs 0.99 cents/GB/month but takes 12 hours to retrieve. For analytics logs older than 30 days, we moved 18 TB from S3 Standard to Deep Archive and saved $744/month. The catch: queries that hit the archive now take 15 minutes instead of 30 seconds, so we built a two-tier cache—30 days in S3, 90 days in S3 IA, and the rest in Deep Archive—with a daily job that pre-warms the IA layer overnight. The net cost dropped 42% while keeping 95% of queries under 2 seconds.


The final lever is multi-cloud arbitrage. In one region, Azure had a 25% discount on Dv5 instances for the same specs as AWS m5a. We ran a 30-day benchmark: the Azure bill for the same workload was $842 vs AWS’s $1,102. We moved 15% of non-critical batch jobs and saved $260/month without rewriting a line of code. The trick is to treat each cloud as a SKU in a global pricing spreadsheet; we now auto-purchase on the cheapest provider for each workload class using a simple cost API that queries both clouds every hour.


## Quick reference

| Levers | Typical saving | Effort | Risk | Best for |
|---|---|---|---|---|
| Spot for stateless workloads | 60–90% | Low | Interruptions | Batch, staging, dev |
| Savings Plans (1 yr, no upfront) | 30–66% | Medium | Commitment | Production EC2, Lambda, Fargate |
| Right-size instances | 20–50% | Low | Performance | Any under-utilized fleet |
| Delete orphaned EBS & snapshots | 10–30% | Low | Data loss if mis-tagged | Expiring projects |
| S3 lifecycle to Glacier/Deep Archive | 40–80% | Medium | Retrieval latency | Logs, backups, cold data |
| Multi-cloud arbitrage | 10–25% | High | Vendor lock-in, tooling | Non-critical batch jobs |
| Auto-shutdown dev environments | 30–50% | Low | Developer friction | Dev/test accounts |
| Provisioned concurrency for Lambda | 50–70% | Medium | Cost vs latency trade-off | Sporadic but latency-sensitive functions |


## Further reading worth your time

- AWS Well-Architected Framework – Cost Optimization Pillar (free, 15-minute read)
- “Cloud FinOps” by J.R. Storment and Mike Fuller – the only book that treats cloud spend like a P&L
- AWS Compute Optimizer user guide – the automated right-sizing tool we used for m5.xlarge → m5.large
- Finout’s “Cost per unit” playbook – how to calculate cost per API call, per GB processed, per user
- Kubecost’s open-source cost model – run it in-cluster to see Kubernetes pod-level spend
- The official AWS Spot price history CSV – download it monthly to set max prices safely
- Terraform aws-cost-explorer-forecast module by cloudposse – automates daily spend forecasts


## Frequently Asked Questions

How do I stop paying for resources I forgot I launched?

Turn on AWS Resource Explorer and filter by resource type and age. Any EC2 instance older than 30 days with no tag `owner=*` is a zombie. Delete it or tag it; if it’s prod, you’ll hear about it fast. We deleted 47 dev instances in one afternoon and cut $1,300/month.


Why does my RDS bill spike even though traffic is flat?

Check the backup retention period and automated snapshots. The default is 7 days, but many teams raise it to 35 days for compliance. Each day adds ~0.1 GB/day to storage. If you have 100 GB of data, that’s an extra $3.50/month per day of retention. We reduced ours from 35 to 7 and saved $21/month per instance.


Is it worth buying a Reserved Instance for a single prod instance?

Only if the instance runs 24×7 for the full term. If it’s a dev or staging box, Savings Plans are safer because they’re flexible across families and regions. We bought a one-year RI for a prod m5.xlarge and saved 37%, but when the project ended early we lost the benefit of unused months. Savings Plans refund unused commitment automatically.


How do I explain the cost savings to my CFO without sounding like I’m cutting corners?

Frame it as a capacity rebalancing exercise. Show the bill before and after, with latency and error-rate metrics side by side. Include the waste: “We deleted 14 TB of orphaned snapshots worth $324/month and cut staging RDS spend by 40% by shrinking the instance class.” CFOs love numbers tied to business outcomes, not just “we saved money.”


## Next step

Open AWS Cost Explorer right now, set the time window to “Last 30 days”, and filter by service. Look for the top three cost drivers. Pick the smallest one that you can act on today—usually an idle dev RDS or a long-running EC2 in a forgotten account. Stop or resize it, and tomorrow check the bill again. That single action will teach you more about cost optimization than any spreadsheet ever will.