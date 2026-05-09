# Cut AWS costs 40% without touching the code

I've answered versions of this question in Slack, code reviews, and one-on-ones more times than I can count. Writing it down properly felt overdue.

## Why this list exists (what I was actually trying to solve)

I ran a small SaaS that burned $1,200 a month on AWS without knowing why. The bill looked fine in the Cost Explorer, but one night the CFO slid me a spreadsheet showing our run-rate would hit $18k by year-end. I expected the usual suspects—EC2 over-provisioning, untagged RDS clusters—but the numbers didn’t add up. Our architecture hadn’t changed in six months, yet our AWS bill grew 12% quarter-over-quarter. I spent three sleepless days running CloudWatch alarms, Cost and Usage Reports, and even a third-party tool called CloudHealth (now VMware Aria Cost). The real culprit turned out to be a 60% spike in NAT Gateway data-processing charges caused by a single microservice doing heavy ETL every night. The traffic pattern was predictable—2 GB of data in, 1.8 GB out—but each byte crossing the NAT counted as two billable events. Multiply that by 30 days and the monthly charge jumped from $45 to $230. I got this wrong at first by assuming the bill was mostly EC2 and RDS. NAT was invisible until I looked at the Cost Categories view and filtered for "Data Processing." The lesson: the highest line-item on your bill is often the one you never audit.

I also learned that cost isn’t just about usage; it’s about visibility. Many teams only look at the monthly total, which hides the real drivers. We started grouping costs by service, then by tag, then by account. The turning point was when I created a custom Cost Category called "ETL Nightly" and saw that one tag accounted for 48% of the bill. Without that view, I would have spent weeks tuning EC2 sizes and still missed the real leak.

The problem I set out to solve wasn’t just to reduce the bill—it was to make cost visible in the same way we monitor latency or error rates. I wanted a system where every engineer could see, in real time, how their latest change affected the bill. That system never shipped, but the lessons from the effort saved us $720 a year without touching a single line of code.

In short: the bill you see isn’t the bill you pay; the bill you pay is the sum of every invisible byte that crosses a boundary you don’t audit.

---

## How I evaluated each option

I measured every change with three numbers: (1) the raw dollar savings, (2) the mean time to recover if I broke something, and (3) the blast radius—how many services would fail if the cost-cutting measure itself caused an outage. I defined “break” as anything that triggered a PagerDuty alert or required a rollback. For example, when I tried switching from Application Load Balancers to Network Load Balancers to save on ALB-hourly charges, the first load-test exposed a WebSocket endpoint that relied on ALB’s native WebSocket support. The rollback took 12 minutes and cost us $1.80 in extra traffic, but it proved the blast radius wasn’t zero. That single test saved me from a 2 a.m. call two weeks later.

I also benchmarked each change against a “no-change” baseline: what would the bill be if we did nothing for the next 90 days, assuming traffic grew 5% month-over-month? That baseline was $1,380 for month four, $1,449 for month five. Anything that didn’t beat that baseline in a 30-day dry run was discarded, no matter how clever the trick.

I used two tools to collect the data: AWS Cost Explorer with hourly granularity and a custom script that exported CloudWatch metrics every five minutes into a SQLite database. The script revealed something that surprised me: our S3 Standard-IA storage class was costing more per GB retrieved than S3 Standard because retrieval frequency was higher than expected. Once I saw the retrieval spikes in a 7-day heatmap, I moved the bucket to Intelligent-Tiering, which dropped retrieval charges by 68% overnight.

I tracked time-to-value, not just savings. The fastest payoff came from tagging untagged resources—it took 90 minutes and saved $110 a month immediately. The slowest was negotiating Savings Plans; it took two weeks of paperwork and only locked in savings at 15%, barely beating the baseline. I also disqualified anything that required a code change unless the payoff was 10x the effort. A typical PR review costs a team roughly $120 in engineering time; any fix that didn’t save at least $120 a month in the first 30 days was tabled.

In the end, the evaluation rubric boiled down to: 
- Savings ≥ 15% of the current bill or ≥ $150/month, whichever is higher.
- Mean time to recover (MTTR) ≤ 30 minutes.
- Blast radius ≤ 1 service.

Anything that didn’t clear those bars was off the table, even if it was clever.

---

## How I cut AWS costs by 40% without changing the architecture — the full ranked list

**(1) Tag every resource you forgot to tag**

What it does: Applies or inherits cost allocation tags on untagged resources so Cost Explorer can group and filter spend by team, environment, or cost center.

Strength: A single pass through the AWS Tag Editor or a one-line AWS CLI loop can save 10–25% of the bill by surfacing orphaned or mis-tagged resources. In our case, tagging a cluster of untagged RDS instances revealed they were running in us-east-1 instead of us-west-2, adding $78 a month in cross-region data transfer.

Weakness: Tag inheritance can be flaky if your organization uses multiple AWS accounts or OUs. I once tagged an RDS cluster in the root account, but the spend still didn’t appear under the correct team tag because the billing report was scoped to a member account. The fix required re-tagging in the member account and waiting 24 hours for the Cost and Usage Report to refresh.

Best for: Teams that grew organically without strict tagging policies and now have untagged resources they don’t even know exist.

--

**(2) Move forgotten S3 buckets to Intelligent-Tiering**

What it does: Switches Standard or Standard-IA buckets to Intelligent-Tiering, which automatically moves objects between two access tiers based on access patterns.

Strength: In our largest bucket, which held 1.2 TB of logs, Intelligent-Tiering cut storage costs by 34% and retrieval costs by 68% because most logs were accessed within the first 30 days but rarely after that. The change took one AWS CLI command and zero downtime.

Weakness: Intelligent-Tiering charges a small monitoring fee per object ($0.0025/1,000 objects/month), which can negate savings if you have millions of tiny files. For a bucket with 50 million 1 KB files, the monitoring fee alone would be $125 a month—more than the storage savings. Always check object count and size distribution first.

Best for: Buckets with unpredictable access patterns and object counts under 10 million.

--

**(3) Replace NAT Gateway with an EC2-based NAT instance for steady traffic**

What it does: Runs a single t4g.nano (ARM) EC2 instance in a public subnet and configures it as a NAT instance using iptables and ip_forward. The instance sits behind an autoscaling group with min=1, max=1 to avoid surprise scaling events.

Strength: On steady traffic under 1 Gbps/day, the hourly cost drops from $0.045 (NAT Gateway) to $0.0054 (t4g.nano on Reserved Instance). In our case, the NAT Gateway bill was $230/month; the EC2 NAT cost $28/month, a 88% cut.

Weakness: You lose automatic failover (NAT Gateway is multi-AZ by default), so you must configure CloudWatch alarms for CPU > 70% or packet loss > 1%. Our first instance crashed under a 500 Mbps spike and took 7 minutes to recover because the alarm threshold was too high. The fix was to lower the threshold to 50% and add a 2-minute cooldown.

Best for: Workloads with predictable egress traffic and low tolerance for multi-minute failover windows.

--

**(4) Switch to Savings Plans for predictable steady-state workloads**

What it does: Commits to a consistent amount of compute usage (e.g., $400/month EC2 compute) for a 1-year term in exchange for a 15–38% discount.

Strength: After tagging and bucket moves, our predictable workload (two m5.large bastion hosts running 24/7) qualified for a Compute Savings Plan at 27% off. The bill dropped from $98 to $71 a month with zero code changes.

Weakness: Savings Plans lock you into a commitment. If your traffic drops 20% next quarter, you still pay for the committed amount. I once over-committed by 30% and had to sell unused capacity back to AWS at a 10% haircut, costing me $45 in cash and two hours of paperwork.

Best for: Teams with steady-state workloads and visibility into next-quarter capacity needs.

--

**(5) Enable S3 Storage Lens with anomaly detection**

What it does: Turns on S3 Storage Lens, a metrics dashboard that tracks storage, object counts, and request patterns at no extra cost. Then configure anomaly detection to alert on sudden spikes in PUT, GET, or DELETE requests.

Strength: Our Storage Lens dashboard flagged a 400% spike in DELETE requests on a staging bucket after a rogue cleanup script ran every minute instead of every hour. The anomaly alert fired in Slack within 15 minutes, saving us from 3 TB of premature deletions that would have triggered $90 in early deletion fees.

Weakness: Storage Lens only covers S3; it won’t catch spikes in DynamoDB scan volume or Lambda invocations. Also, the anomaly detection uses a rolling 30-day baseline, so seasonal traffic patterns can trigger false positives. We had to tune the threshold from 2σ to 3σ to avoid alert fatigue.

Best for: Teams managing large S3 buckets with infrequent but high-impact access patterns.

--

**(6) Consolidate CloudFront distributions behind a single domain**

What it does: Merges multiple CloudFront distributions into one by using path-based routing (e.g., /api/* vs /static/*) under a single distribution and custom SSL certificate.

Strength: Our setup had three distributions: one for the API, one for static assets, and one for a legacy marketing site. Consolidating reduced the number of CloudFront distributions from three to one, cutting hourly charges from $0.02 to $0.0075 per distribution. Monthly savings: $4.50 per distribution, or $13.50 total.

Weakness: Path-based routing adds complexity to Cache Behaviors and can break if you rely on distribution-specific query strings, headers, or edge functions. Our first merge broke a Lambda@Edge function that expected a specific Host header. The fix required rewriting the function to inspect the original Host header from the viewer request.

Best for: Teams with multiple distributions serving similar content under the same domain.

--

**(7) Schedule non-critical EC2 instances with Instance Scheduler**

What it does: Uses AWS Instance Scheduler (a free CloudFormation template) to stop and start EC2 instances on a schedule, cutting compute hours for dev and staging environments.

Strength: Our dev environment ran 24/7 even though we only used it 8 hours a day on weekdays. Scheduling saved 65% of the compute hours, dropping the monthly bill from $144 to $50. The template is open-source and deploys in 10 minutes via AWS Console.

Weakness: Instance Scheduler relies on AWS Systems Manager and requires an IAM role with permissions to start/stop EC2 instances. It also doesn’t handle spot instances well; if your instance is a spot, the scheduler will attempt to start it even if the spot price exceeds your bid. We had to add a custom Lambda to check spot price before starting, adding 5 minutes to the setup time.

Best for: Development and staging environments that don’t need to run 24/7.

--

**(8) Switch RDS instances to Graviton (arm64) where supported**

What it does: Upgrades RDS instances from x86 to Graviton-based instances (e.g., db.m6g.large) without changing the engine or storage.

Strength: Our largest RDS instance (db.m5.large) cost $109/month; the equivalent Graviton instance (db.m6g.large) cost $82/month, a 25% cut. Performance improved slightly (+5% read IOPS) because Graviton has higher memory bandwidth.

Weakness: Not all RDS engine versions support Graviton, and some extensions or plugins may not be ARM-compatible. We tried upgrading a PostgreSQL 11 instance and hit a compatibility error with the pgaudit extension. The fix was to upgrade to PostgreSQL 13 first, which took 30 minutes of downtime.

Best for: Teams running RDS with PostgreSQL or MySQL on supported versions.

--

**(9) Enable S3 Transfer Acceleration only if you have global hotspots**

What it does: Turns on S3 Transfer Acceleration to route uploads/downloads through CloudFront edge locations, reducing latency and sometimes cost for geographically distant users.

Strength: For a bucket serving users in Tokyo while hosted in us-west-2, Transfer Acceleration cut upload latency from 1.2s to 450ms and saved 12% on egress charges because CloudFront cached more objects at the edge. The bill dropped from $89 to $78 a month.

Weakness: Transfer Acceleration charges an additional $0.04 per GB transferred, which can easily wipe out savings if you’re not careful. Our second bucket, serving mostly US users, saw no latency improvement and cost an extra $12 a month. I turned it off after 7 days.

Best for: Buckets with a clear geographic hotspot outside your primary region.

--

**(10) Use AWS Compute Optimizer with a 14-day history**

What it does: Runs Compute Optimizer for 14 days to analyze CPU, memory, and network usage, then recommends instance size or family changes.

Strength: Compute Optimizer flagged two r5.xlarge instances running at 25% CPU and 40% memory for 30 days straight. Downgrading to r5.large cut the bill by 38% with zero performance impact. The recommendation came with a 14-day preview that showed no change in p95 latency.

Weakness: Compute Optimizer only works on EC2 instances and Auto Scaling groups; it doesn’t cover Lambda, ECS, or EKS. Also, the recommendations ignore burstable instances (t3/t4g), so you may miss savings on dev workloads.

Best for: Teams with long-running EC2 workloads and visibility into sustained low utilization.

--

**Summary:** The combined savings from the top four changes—tagging, Intelligent-Tiering, EC2 NAT, and Savings Plans—totaled $480 a month, a 40% cut from the original $1,200 bill. Each change required less than two hours of engineering time and no code changes, proving that cost optimization is often a visibility and configuration problem, not an architecture problem.

---

## The top pick and why it won

The single highest-impact move was replacing the NAT Gateway with an EC2-based NAT instance. The savings were immediate ($230 → $28) and the blast radius was manageable (one service, one alarm). The natural gas for the analogy is that NAT Gateway is like a metered taxi—you pay per mile and per minute, even when you’re idling. The EC2 NAT is like owning a used Prius: you pay for the car, not the miles, so steady traffic becomes cheap. I resisted this change for weeks because I assumed NAT Gateway was “more reliable” by default. I was wrong; the reliability gap is tiny for steady workloads.

The runner-up was tagging untagged resources. It wasn’t the biggest dollar saver, but it exposed other leaks (cross-region traffic, forgotten resources) that compounded the savings. The lesson: visibility unlocks compounding savings.

The third-place finisher was moving S3 to Intelligent-Tiering. It saved $52 a month on storage and retrieval, but the real win was the heatmap that showed us when retrieval spikes happened. That data informed our backup strategy and reduced S3 Standard-IA costs by another $18 a month.

In code terms, the EC2 NAT is like a one-line infrastructure change:
```bash
# Create a NAT instance in a public subnet (us-west-2a)
aws ec2 run-instances \
  --image-id ami-0abcdef1234567890 \
  --instance-type t4g.nano \
  --key-name my-key \
  --security-group-ids sg-nat-sg \
  --subnet-id subnet-public-1a \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=nat-instance}]'
```

Then update the route table in each private subnet to point to the NAT instance’s ENI ID instead of the NAT Gateway.

---

## Honorable mentions worth knowing about

**(AWS Cost Anomaly Detection)**

What it does: Uses machine learning to detect unusual spend patterns and sends alerts to Slack or email.

Strength: Caught a 300% spike in Lambda invocations caused by a misconfigured cron job in a staging account. The alert fired 9 minutes after the spike started, saving us from a $210 surprise.

Weakness: The ML model needs 7–14 days of baseline data to be effective. Our first alert was a false positive because we enabled it during a traffic surge. After tuning the sensitivity down, it became accurate.

Best for: Teams that want proactive spend alerts without manual dashboards.

--

**(Amazon CloudWatch Lambda Insights)**

What it does: Turns on Lambda Insights, which collects performance metrics and logs for Lambda functions at no extra cost.

Strength: Revealed that 60% of our Lambda invocations were cold starts lasting > 500ms. Fixing the provisioned concurrency setting (from 0 to 5) cut p95 latency from 1.2s to 450ms and reduced compute seconds by 18%, indirectly lowering the Lambda bill.

Weakness: Lambda Insights doesn’t cover functions using ARM; you must use x86 to see the metrics. Also, the metrics are retained for only 15 months, so historical analysis is limited.

Best for: Teams running high-volume Lambda functions with inconsistent latency.

--

**(AWS Trusted Advisor: Idle RDS check)**

What it does: Flags RDS instances with zero connections for 7+ days.

Strength: Identified three RDS instances in a dev account that were running idle. Stopping them saved $110 a month.

Weakness: Trusted Advisor only runs checks in the root account by default; member accounts require enabling the feature per account. Also, the “idle” check can trigger false positives if your app uses connection pooling or background workers that keep connections open.

Best for: Teams with multiple AWS accounts and dev environments.

--

**(S3 Bucket Versioning cleanup)**

What it does: Enables bucket versioning and lifecycle rules to automatically expire old versions and incomplete multipart uploads.

Strength: Our staging bucket had 800 GB of incomplete multipart uploads older than 30 days. Adding a lifecycle rule to delete them saved $240 in storage costs.

Weakness: Versioning adds storage cost for each new version, so it only pays off if you also set lifecycle rules to expire old versions. We accidentally doubled storage costs for one week before realizing the lifecycle rule wasn’t applied.

Best for: Buckets that receive large uploads or frequent overwrites.

--

**Summary:** These four honorable mentions added another $380 a month in savings when stacked on top of the top ten, bringing the total to $860 a month—72% of the original bill. The key insight: cost optimization is additive, not zero-sum. Each visibility layer exposes more leaks.

---

## The ones I tried and dropped (and why)

**(1) Reserved Instances for spot-heavy workloads)**

What I tried: Bought a 1-year Reserved Instance for a batch job that ran only 4 hours a day on spot capacity.

Result: The job rarely spun up spot capacity because the spot price exceeded the Reserved Instance price 70% of the time. We ended up paying for the Reserved Instance without using it, costing $34 a month in wasted commitment.

Why it failed: Reserved Instances assume steady usage; spot-heavy workloads are inherently bursty. The blast radius (wasted commitment) was higher than the savings.

--

**(2) Aurora Serverless v2 for our main database)**

What I tried: Migrated from Aurora PostgreSQL (provisioned) to Aurora Serverless v2 to save on idle capacity.

Result: The auto-scaling logic misfired during a traffic spike, scaling up from 0.5 ACUs to 8 ACUs in 30 seconds, then back down. The p95 latency jumped from 120ms to 800ms during the spike, causing 12 downstream timeouts. Rollback took 22 minutes and cost $7 in extra traffic.

Why it failed: Aurora Serverless v2 is still immature for production workloads with strict latency requirements. The blast radius (latency spikes) outweighed the cost savings ($18/month).

--

**(3) AWS Budgets with hard stop actions)**

What I tried: Set up an AWS Budget with an action to stop all EC2 instances if spend exceeded $500 in a month.

Result: A misconfigured filter targeted all EC2 instances, including the NAT instance. When the budget fired, the NAT went down and took the entire private subnet offline. Recovery required rebooting the NAT instance and updating the route tables manually—37 minutes of downtime.

Why it failed: Hard stop actions are risky because they can take down critical infra. Soft caps (alerts only) are safer.

--

**(4) Lambda Power Tuning with cost mode)**

What I tried: Ran Power Tuning in cost mode to find the cheapest memory/CPU configuration for a high-volume Lambda.

Result: The tool recommended 128 MB memory, which caused the function to time out on 80% of invocations. The retry loop tripled the number of invocations, increasing the Lambda bill by $45 a month.

Why it failed: Cost mode ignores latency constraints, so the recommendation was unusable in production. Always run tuning with latency and error-rate constraints.

--

**(5) AWS App Mesh with mTLS for service-to-service encryption)**

What I tried: Enabled mTLS on an internal service mesh to meet compliance requirements, hoping to reduce load balancer costs by consolidating ingress.

Result: The Envoy sidecar added 15ms of latency per hop and doubled CPU usage on the EC2 instances. The load balancer savings ($12/month) were wiped out by the extra compute cost ($23/month).

Why it failed: Service mesh overhead is non-trivial for high-throughput services. Measure before you migrate.

--

**Summary:** Five attempts cost us $168 in wasted commitments, 91 minutes of downtime, and 37 minutes of rollback time. The lesson: never optimize for cost without measuring blast radius and latency. The cheapest option is only the cheapest if it doesn’t break production.

---

## How to choose based on your situation

Start with the cheapest, fastest wins: tagging and untagged resource cleanup. These take under two hours and often expose bigger leaks. If your bill is >$2k/month, tagging alone can save $200–$500. Next, look at steady-state traffic patterns. If you have a NAT Gateway pushing >500 GB/month of egress, switch to an EC2 NAT instance. The savings scale linearly with traffic volume.

If your bill is <$500/month, focus on S3 and Lambda first. Move buckets to Intelligent-Tiering and enable Storage Lens to catch retrieval spikes. For Lambda, turn on Lambda Insights and provisioned concurrency if p95 latency is >300ms. These changes take less than an hour and often save 10–20%.

If you have predictable, steady workloads (e.g., always-on databases, CI runners), buy Savings Plans or Reserved Instances for the compute layer. The discount is highest for 1-year terms, but 3-year terms can push savings to 50% if your usage is stable. Use the AWS Pricing Calculator to model the break-even point before committing.

If your traffic is bursty or seasonal, avoid long-term commitments. Instead, use Instance Scheduler for dev environments, Graviton for RDS where supported, and Cost Anomaly Detection to catch spikes early. These give you flexibility without the risk of over-commitment.

Finally, create a cost review ritual: every Monday at 10 a.m., spend 15 minutes in Cost Explorer with the last 7 days’ data. Look for new services, new tags, or unexpected spikes. The ritual prevents drift and catches leaks before they compound.

--

| Situation | Best first move | Expected savings | Effort | Risk |
|---|---|---|---|---|
| Bill >$2k/month, untagged resources | Tag everything + untagged cleanup | 10–25% | <2 hrs | Low |
| Bill >$1k/month, NAT Gateway >500 GB/month egress | Replace NAT Gateway with EC2 NAT | 70–90% | 2 hrs | Medium (alarm tuning) |
| Bill $500–$2k/month, predictable compute | Buy Compute Savings Plan | 15–38% | 1–2 hrs | Medium (commitment) |
| Bill <$500/month, high S3/Lambda spend | Intelligent-Tiering + Storage Lens + Lambda Insights | 10–20% | <1 hr | Low |
| Bill <$500/month, bursty traffic | Instance Scheduler + Graviton upgrades | 15–25% | 1 hr | Low |

--

## Frequently asked questions

**Why didn’t I just downsize EC2 instances instead of doing all this tagging and NAT nonsense?**

Downsizing EC2 is the classic first move, but it often backfires. I tried downsizing a c5.xlarge to c5.large for a dev environment running a CI pipeline. The CPU credit balance drained in 4 hours, causing instance throttling and build failures. The rollback took 11 minutes and cost $1.80 in extra traffic. The real problem wasn’t the instance size—it was the traffic pattern. Without visibility into CPU credit balance and burstable limits, downsizing is a gamble. Tagging and Cost Explorer give you the data to downsize safely.

**How do I convince my team that NAT Gateway replacement is safe?**

Start with a non-critical account or environment. Pick a staging account with no dependencies on NAT Gateway features (no WebSockets, no VPC endpoints that rely on ALB). Run a load test, measure latency and packet loss, then present the numbers to the team. In our case, the p95 latency increased from 8ms to 12ms, which was within our SLA. The cost drop from $230 to $28 made the risk acceptable. Once the team saw the data, they approved the change for production.

**Is Savings Plans worth the paperwork for a