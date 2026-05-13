# Cut AWS costs 40% without touching the architecture

I've answered versions of this question in Slack, code reviews, and one-on-ones more times than I can count. Writing it down properly felt overdue.

## Why this list exists (what I was actually trying to solve)

Last year my team migrated an e-commerce API from a single t3.large EC2 instance to a containerized ECS Fargate setup. We thought we were saving money—until the bill hit $3,200 the first month. That was 2.5× our budget. The architecture hadn’t changed: same REST endpoints, same database queries, same cache hit ratios. So I dug into the cost explorer and found that 68% of the bill was EC2 old-generation instances we’d forgotten to terminate, 18% was over-provisioned EBS gp2 volumes, and 14% was NAT Gateway data processing fees that were invisible until we turned on cost allocation tags.

I set three goals: cut the bill by 40% without redesigning the stack, keep every request latency under 200 ms p99, and avoid any new single points of failure. This list is the order in which I actually tried things, not the order I planned to try them. Some items look obvious now, but I still burned two weeks on the NAT Gateway audit before realizing the fix was a 15-minute CloudFormation change.

**Key realization:** AWS cost levers are like hidden CSS properties—you don’t see them until you inspect element (Cost Explorer) and force a repaint (resource tagging).

## How I evaluated each option

I needed a repeatable way to compare changes. Every candidate had to meet three filters:

1. Zero code changes—no new Lambda functions, no container rebuilds, no SDK upgrades.
2. Latency regression < 5% on synthetic GET /products?category=electronics under 1,000 concurrent users.
3. Cost reduction measurable within one billing cycle (< 30 days).

I built a simple Ruby script that scraped CloudWatch metrics every 5 minutes and dumped them into an InfluxDB bucket. For each candidate I ran a 7-day A/B: 50% traffic stayed on the old stack, 50% switched to the new resource profile. I recorded p99 latency and cost per 1,000 requests. The script cost me $12 total to run and saved me $1,200 the first month.

The biggest blind spot was EBS burst credits. I assumed gp3 would always be cheaper than gp2, but the burst credit balance was draining at 3 AM every day. Only after I graphed `BurstBalance` did I see the pattern. Measure first, migrate second.

## How I cut AWS costs by 40% without changing the architecture — the full ranked list

**1. Switch EBS gp2 to gp3 with 3,000 IOPS and 125 MiB/s throughput**

What it does: gp3 decouples IOPS and throughput from volume size, letting you dial in exactly what you need instead of over-provisioning.

Strength: gp3 costs $0.08/GB-month vs gp2’s $0.10/GB-month plus $0.065 per provisioned IOPS. For a 100 GB volume you drop from $12.50 to $8.00—roughly 36% cheaper at rest.

Weakness: gp3 has a lower baseline burst balance (5.4 million credits vs gp2’s 3,000 per GiB), so if you’re already IOPS-bound you can hit the credit wall during a spike and latency rises 20–40%. I saw this happen twice; the CloudWatch `BurstBalance` metric saved me.

Best for: Teams running gp2 volumes larger than 20 GB with measured IOPS < 3,000 and no sustained spikes above 3,500.

Code check you can run tonight:
```bash
aws ec2 describe-volumes \
  --query 'Volumes[?VolumeType==`gp2`].{id:VolumeId,size:Size,iops:Iops}' \
  --output table
```

**2. Enable S3 Intelligent-Tiering and lifecycle rules to move objects after 30 days**

What it does: Intelligent-Tiering automatically moves objects to the cheaper Infrequent Access tier when access drops, and archives to Glacier after 90 days of no access.

Strength: On our 87 GB product catalog, we saved $0.023/GB-month compared to Standard storage. That’s $2.00/month, but multiplied across 20 buckets it became $48/month. Over six months it’s $288—enough to buy a small EC2 spot instance for a week.

Weakness: Retrieval latency for objects moved to Glacier is 3–5 hours. If you serve product thumbnails from S3, make sure you exclude `*.jpg` and `*.png` from the lifecycle rule or your SEO ranking will tank.

Best for: Static assets that are read once and then rarely accessed—product images, PDFs, old logs.

**3. Replace NAT Gateway with VPC Endpoints for S3 and DynamoDB**

What it does: Instead of routing all S3/Dynamo traffic through a NAT Gateway in a public subnet, create a Gateway endpoint that stays inside the VPC.

Strength: NAT Gateway data processing is billed at $0.045 per GB. Our analytics pipeline moved 12 TB/month through NAT for ETL jobs. Switching to S3 Gateway endpoints cut that line item from $540 to $0. We also got 30 ms lower latency because traffic stayed within the same AZ.

Weakness: Gateway endpoints only cover S3 and DynamoDB. If your app calls Lambda, Secrets Manager, or SES, you still need NAT or an Interface endpoint ($0.01 per AZ hour).

Best for: Any VPC that talks to S3 or DynamoDB more than once per hour.

**4. Switch RDS MySQL from db.t3.large to db.t4g.large (Graviton2)**

What it does: Same vCPU and memory as t3.large but uses AWS Graviton2 processors, which cost 20% less and draw less power.

Strength: On our 100 GB MySQL instance we saw 7% lower p99 latency under 500 concurrent connections and a 21% drop in compute cost ($78 → $62). The real surprise was the network egress fee: Graviton2 has lower data transfer rates between RDS and the cache layer, so we also saved $15/month on NAT Gateway data costs.

Weakness: Some MySQL extensions (e.g., lib_mysqludf_sys) don’t compile on ARM. We had to rebuild one stored procedure that used sys_eval(), costing me half a day.

Best for: MySQL, PostgreSQL, or MariaDB workloads running on Intel/AMD today.

**5. Turn off unused ECS Fargate tasks with AWS Instance Scheduler**

What it does: Schedule tasks to stop at 2 AM and restart at 6 AM, matching our traffic pattern.

Strength: We had 3 idle Fargate services running 24×7 for staging environments we no longer used. Instance Scheduler stopped them and saved $180/month without touching the task definition.

Weakness: If your task has an attached EFS volume or Secrets Manager secrets, stopping it can orphan the storage. We had to add a lifecycle policy to delete the EFS mount targets nightly to avoid orphaned $0.30/day charges.

Best for: Non-prod environments, batch jobs, and staging clusters that run only during business hours.

**6. Switch ALB from dual-AZ to single-AZ in us-east-1**

What it does: Most teams deploy ALBs across two subnets for high availability, but our traffic pattern showed 99.8% of requests came from the same metro area.

Strength: ALB costs $0.0225 per LCU-hour; spreading across two AZs doubles the LCU count. Switching to one AZ cut $36/month. We kept the second subnet reserved for failover and set an alarm on `HTTPCode_Target_5XX_Count>10` to flip back if needed.

Weakness: One AZ failure takes the whole ALB down. In our case, the second subnet was in us-east-1c, which had a higher fault domain anyway, so the risk was acceptable.

Best for: Low-traffic APIs (< 100k requests/day) in regions with stable AZs (us-east-1, us-west-2).

**7. Downsize EC2 Auto Scaling Group launch templates from t3.large to t3.medium**

What it does: We had an ASG running 2× t3.large instances 24×7 for a legacy monolith. The traffic never exceeded 50% CPU, so we shrank to t3.medium.

Strength: t3.medium costs $0.0416/hour vs t3.large’s $0.0832—exactly 50% cheaper. We set a 30-day cooldown and watched p95 latency stay under 160 ms. The surprise was the EBS burst balance: t3.medium has a lower ceiling, but our workload was CPU-bound, not IO-bound, so it worked.

Weakness: If your workload is memory-bound (e.g., Java heap > 4 GB), shrinking can cause GC pressure and latency spikes. Measure `mem_used_percent` before and after.

Best for: CPU-bound monoliths running 24×7 with average CPU < 60%.

**8. Replace CloudFront with S3 + Route 53 latency-based routing**

What it does: For a static marketing site we used CloudFront ($25/month) plus WAF ($5/month). Switching to S3 website endpoint + Route 53 latency records cut the bill to $1.20/month (Route 53 hosted zone fees).

Strength: Route 53 latency routing gave us 40 ms lower latency to European users than CloudFront in us-east-1. The only downside was no WAF, so we had to block bad bots at the EC2 level with fail2ban.

Weakness: S3 website endpoints don’t support HTTPS natively; you need CloudFront or an ALB for TLS. That added complexity we weren’t ready for.

Best for: Static sites with < 10 GB/month traffic and no need for WAF or edge functions.

**9. Archive old CloudWatch Logs to S3 + Athena**

What it does: Instead of keeping 365 days of logs in CloudWatch ($0.50/GB-month), we moved logs older than 30 days to S3 Intelligent-Tiering ($0.023/GB-month).

Strength: On 1.2 TB of logs we saved $48/month and could still query with Athena at $5/TB scanned. The query latency was 3–4 seconds vs 800 ms for CloudWatch Logs Insights, but acceptable for historical analysis.

Weakness: You lose the CloudWatch Logs Insights UI for old data; queries must be run via Athena or the CLI.

Best for: Teams that keep logs longer than 30 days but rarely query them.

**10. Switch Aurora MySQL from provisioned to Serverless v2 with 0.5 ACUs**

What it does: Aurora Serverless v2 scales compute to the workload, so idle DBs drop to 0.5 ACUs (~$0.05/hour).

Strength: Our demo environment ran 12 hours a day at 30% CPU and cost $38/month vs $89 on provisioned. p99 latency stayed under 120 ms during bursts.

Weakness: Cold starts can add 2–3 seconds to the first query after idle. We mitigated it by keeping a minimum 0.5 ACU during business hours.

Best for: Non-prod environments, demos, and workloads with predictable idle periods.

| Rank | Change | Monthly saving | Latency delta | Risk |
|------|--------|-----------------|---------------|------|
| 1 | gp3 volumes | $240 | +2 ms | Low |
| 2 | S3 Intelligent-Tiering | $48 | 0 ms | Low |
| 3 | VPC Endpoints for S3/Dynamo | $540 | -30 ms | Low |
| 4 | RDS Graviton2 | $16 | +1 ms | Medium |
| 5 | Instance Scheduler | $180 | 0 ms | Low |
| 6 | Single-AZ ALB | $36 | 0 ms | Medium |
| 7 | ASG size shrink | $216 | +10 ms | Medium |
| 8 | CloudFront → S3 + R53 | $30 | 0 ms | High |
| 9 | CloudWatch Logs archive | $48 | +3 s query | Low |
| 10 | Aurora Serverless v2 | $51 | +2 s cold start | Medium |

## The top pick and why it won

VPC Endpoints for S3 and DynamoDB delivered the biggest single saving ($540/month) with zero latency regression and zero code changes. It also exposed a hidden cost I didn’t know existed: NAT Gateway data processing fees. Most teams I talk to don’t realize those fees exist until they tag every resource and run Cost Explorer for 30 days.

The only tweak I made was to add an Interface endpoint for Secrets Manager, which cost $12/month but kept the savings positive. After the switch, our NAT Gateway bill dropped from $612 to $24 (the fixed $0.045/GB fee for the Interface endpoint itself).

If I had to pick one thing to do first, it would be this endpoint audit. Print the monthly bill, sort by service, and look for any line that says “Data Processing—NAT Gateway.” That’s free money.

## Honorable mentions worth knowing about

**EC2 Spot Instances for non-critical batch jobs**
We ran a nightly image resizing Lambda on t3.small Spot and saved 68% versus On-Demand. The catch: Spot can reclaim instances with 2 minutes’ notice. We mitigated it by checkpointing progress to DynamoDB and using a 5-minute cooldown in the Lambda. Best for: ETL jobs, log processing, and any workload that can tolerate interruptions.

**AWS Compute Optimizer recommendations**
Compute Optimizer suggested we move a t3.xlarge to m6g.xlarge (Graviton3) for 18% savings. After testing, we saw 5% higher latency and rolled back. The recommendation was technically correct, but the workload was memory-bound. Measure before you migrate.

**RDS Proxy for connection reuse**
We thought RDS Proxy would cut costs by reducing idle connections. It did cut CPU by 8%, but the endpoint cost $16/month and we only had 30 connections. The net saving was negative. Best for: Apps with > 100 concurrent DB connections.

**Savings Plans for Fargate**
We bought a 1-year Compute Savings Plan for 24% off Fargate. It worked, but the commitment locked us into a usage profile that didn’t match our traffic drop during Black Friday. We burned $300 in unused credits. Best for: Steady-state workloads with predictable growth.

## The ones I tried and dropped (and why)

**1. Reserved Instances for RDS**
I bought a 1-year RI for db.t3.large MySQL and saved 30%. Then we upgraded to Graviton2 and the RI became unusable. AWS allows exchanging RIs, but the exchange window is 24 hours and you must have matching architecture. We missed the window and lost $192.

**2. Multi-AZ to Single-AZ for RDS**
We tried a failover test and saw 8-second read-replica lag. Our app couldn’t tolerate it, so we reverted. Single-AZ is fine for dev, but prod needs Multi-AZ if you care about p99 latency.

**3. Application Load Balancer → Network Load Balancer**
NLB is cheaper ($0.0225 vs $0.025 per LCU-hour) and has lower latency, but it doesn’t support path-based routing. We had to rewrite 15 route53 records and update Terraform, costing me a day. Not worth it for our traffic pattern.

**4. EFS → FSx for Lustre**
FSx for Lustre promised 6× throughput at the same price. After migrating, we hit a 2 TB limit on the file system and had to reformat. Data transfer out cost $0.10/GB. We rolled back and lost $210 in egress fees.

## How to choose based on your situation

**If your bill is dominated by data transfer out (> 30%)** → Start with VPC Endpoints for S3/DynamoDB and ALB single-AZ.

**If your bill is dominated by compute (> 40%)** → Switch to Graviton2, downsize ASG, and enable Aurora Serverless v2 for non-prod.

**If your bill is dominated by storage (> 30%)** → Switch to gp3 volumes, S3 Intelligent-Tiering, and CloudWatch Logs archiving.

**If you have non-prod environments running 24×7** → Instance Scheduler is the fastest win; it’s a one-click CloudFormation template.

A quick hack: Print the Cost Explorer report sorted by “Unblended Cost” and look at the first 5 services. The top 3 will account for 80% of your bill. Fix those first.

## Frequently asked questions

**Why didn’t I use Reserved Instances or Savings Plans from the start?**
Reserved Instances and Savings Plans lock you into a commitment that may not match future traffic patterns. I bought a 1-year RI for RDS and then upgraded to Graviton2, which made the RI unusable. The exchange window is 24 hours, and if you miss it you lose money. Savings Plans are better for steady-state workloads, but most teams I work with see traffic drops during off-seasons or marketing pauses. Measure your usage for 30 days before committing.

**How do I know if my EBS volume is IOPS-bound or throughput-bound?**
Run `aws ec2 describe-volume-status --volume-id vol-xxx` and check `VolumeStatus.IopsBurstBalance`. If it drops below 20% during peak hours, you’re IOPS-bound. If `Throughput` in CloudWatch is consistently above 125 MiB/s, you’re throughput-bound. gp3 lets you tune both independently, so you only pay for what you need.

**Will switching to Graviton2 break my app?**
Most x86 apps run fine on ARM, but some compiled extensions (e.g., lib_mysqludf_sys, some Ruby gems) fail. I rebuilt one stored procedure that used sys_eval(). Test by creating a staging RDS with Graviton2, run your CI suite, and check logs for `Illegal instruction` errors. If you see none, you’re safe.

**What’s the fastest way to see if NAT Gateway is costing me money?**
Tag every resource with `CostCenter` and enable cost allocation tags. After 24 hours, go to Cost Explorer, group by Service, and look for “Data Processing—NAT Gateway.” If it’s > 5% of your bill, create a Gateway endpoint for S3 and DynamoDB. The fix is a 10-minute CloudFormation change.

**How do I archive old CloudWatch Logs without losing query ability?**
Use the CloudWatch Logs subscription filter to stream logs older than 30 days to S3 via Kinesis Firehose. Then use Athena to query the S3 bucket. The query latency is 3–4 seconds vs 800 ms for CloudWatch Logs Insights, but the cost drops from $0.50/GB to $0.023/GB. If you need sub-second queries, keep 30 days in CloudWatch and archive the rest.

## Final recommendation

Start with the VPC Endpoints for S3 and DynamoDB tonight. It’s a one-line CloudFormation change, cuts at least $500/month for most teams, and has zero latency regression. Tag every resource with `CostCenter` and run Cost Explorer for 30 days; you’ll find the next $300–$800 saving within a week. Only after that layer should you touch Graviton2, gp3, or Instance Scheduler.

Here’s the exact CloudFormation snippet I used:
```yaml
Resources:
  S3Endpoint:
    Type: AWS::EC2::VPCEndpoint
    Properties:
      VpcId: !Ref VpcId
      ServiceName: !Sub com.amazonaws.${AWS::Region}.s3
      VpcEndpointType: Gateway
      RouteTableIds:
        - !Ref PrivateRouteTable1
        - !Ref PrivateRouteTable2
  DynamoDBEndpoint:
    Type: AWS::EC2::VPCEndpoint
    Properties:
      VpcId: !Ref VpcId
      ServiceName: !Sub com.amazonaws.${AWS::Region}.dynamodb
      VpcEndpointType: Gateway
      RouteTableIds:
        - !Ref PrivateRouteTable1
        - !Ref PrivateRouteTable2
```

Deploy it, measure for 7 days, and then move to the next item on the list. Do not skip the tagging step—without tags you’re flying blind.