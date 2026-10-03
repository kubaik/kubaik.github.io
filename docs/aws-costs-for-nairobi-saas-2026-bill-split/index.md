# AWS costs for Nairobi SaaS: 2026 bill split

Most AWS cost tutorials stop at "here is the pricing page." The harder problem is attribution: knowing which line item belongs to which product decision, and which decisions to revisit first. This article walks through a representative stack for a SaaS product serving users in Kenya and East Africa, the cost drivers that dominate it, and how to measure each one instead of guessing.

All currency figures below are in USD. Where a number is an AWS list price or a documented default, it is labelled as such. Where a number is arithmetic, the assumptions are stated so you can substitute your own. Nothing here is a measured bill from a real account.

## The stack, and why each piece exists

A common architecture for a low-latency SaaS serving East African users:

- Application servers on EC2 (or a container service) in the closest available region.
- PostgreSQL on a managed relational database service, with a standby for failover.
- A Redis-compatible cache for sessions, rate limiting, and hot reads.
- Object storage for user uploads, fronted by a CDN.
- DNS with latency-based routing if more than one region is in play.
- Cost and usage data exported to a queryable store for attribution.
- Metrics, logs, and traces in a hosted observability tool.

The geography matters more than the component list. Nairobi has no AWS region. The nearest options are `af-south-1` (Cape Town) and `me-south-1` (Bahrain), with `eu-west-1` and `eu-south-1` also viable depending on traffic patterns. That choice cascades into every line item: instance pricing, inter-region transfer, CDN origin fetch latency, and the cost of replicating data for disaster recovery.

## Cost drivers, in rough order of impact

For most small SaaS deployments, four categories dominate:

1. **Compute** — application servers plus any NAT gateways.
2. **Database** — instance hours, storage, backups, and I/O.
3. **Data transfer** — egress to the internet, inter-region replication, and CDN origin fetches.
4. **Observability** — metrics, logs, and traces, which are frequently underestimated because they grow with traffic.

Everything else — DNS, secrets, IAM, small caches — is usually a rounding error at this scale, until it isn't.

### Compute

EC2 pricing varies by region. Graviton (`t4g`, `m6g`, `r6g`) instances are typically cheaper per hour than equivalent x86 instances, and the difference is usually large enough to justify the porting effort for interpreted or well-behaved compiled workloads. To find out what your workload actually costs, the only reliable method is to look at your own usage:

```bash
aws ce get-cost-and-usage \
  --time-period Start=2026-01-01,End=2026-01-08 \
  --granularity DAILY \
  --metrics UnblendedCost UsageQuantity \
  --group-by Type=DIMENSION,Key=INSTANCE_TYPE \
  --filter '{"Dimensions":{"Key":"SERVICE","Values":["Amazon Elastic Compute Cloud - Compute"]}}'
```

That gives you cost per instance type per day. Compare it against your CloudWatch CPU and memory metrics over the same window. An instance averaging under 10% CPU over a week is a candidate for downsizing or consolidation.

NAT gateways deserve separate attention. They are billed per hour plus per gigabyte processed, and they are easy to accumulate — one per availability zone is the default in many infrastructure-as-code templates. A workload doing heavy outbound traffic through NAT can spend more on NAT than on the instances it serves. The mitigation is VPC endpoints for AWS services (S3, ECR, Secrets Manager, and so on), which bypass NAT entirely.

```bash
aws ce get-cost-and-usage \
  --time-period Start=2026-01-01,End=2026-01-08 \
  --granularity DAILY \
  --metrics UnblendedCost \
  --filter '{"Dimensions":{"Key":"USAGE_TYPE","Values":["NatGateway-Hours","NatGateway-Bytes"]}}'
```

### Database

Managed PostgreSQL cost has three components that scale independently:

- **Instance hours**, multiplied by the number of instances. A primary plus a standby is roughly double the instance cost of a primary alone.
- **Storage**, billed per gigabyte-month, plus provisioned IOPS if you use them.
- **Backups**, billed for storage beyond the free retention window. Automated backups within the retention period are typically free up to the size of the database; manual snapshots are billed at standard storage rates indefinitely.

The failure mode here is snapshot accumulation. A team that takes a manual snapshot before every deploy, and never deletes them, will eventually pay more for snapshots than for the live database. The fix is a retention policy enforced in code, not a reminder in a wiki.

To see what you are actually paying for:

```bash
aws rds describe-db-clusters \
  --query 'DBClusters[*].[DBClusterIdentifier,BackupRetentionPeriod,AllocatedStorage]'

aws rds describe-db-snapshots \
  --snapshot-type manual \
  --query 'DBSnapshots[*].[DBSnapshotIdentifier,SnapshotCreateTime,AllocatedStorage]' \
  --output table
```

Sort the snapshot list by age. Anything older than your stated retention policy is a candidate for deletion, subject to whatever compliance regime applies.

For read-heavy workloads, read replicas can be cheaper than scaling the primary vertically, but they add their own instance hours and can lag. Measure replica lag before committing.

### Data transfer

This is where region choice bites hardest. Data transfer out to the internet is billed per gigabyte, and rates differ by region. Inter-region transfer is also billed, in both directions in some configurations. A cross-region replication setup that looks cheap on paper can become the largest line item once you account for daily replication volume.

The way to measure this is to break out usage types:

```bash
aws ce get-cost-and-usage \
  --time-period Start=2026-01-01,End=2026-01-08 \
  --granularity DAILY \
  --metrics UnblendedCost UsageQuantity \
  --filter '{"Dimensions":{"Key":"USAGE_TYPE_GROUP","Values":["EC2: Data Transfer - Internet (Out)","EC2: Data Transfer - Regional"]}}'
```

If inter-region transfer is a meaningful fraction of your bill, the question to ask is whether the replication is serving reads or serving disaster recovery. Read-serving replication can often be replaced by a CDN with a well-configured cache. DR replication cannot, but its volume can sometimes be reduced by replicating snapshots rather than continuous streams.

### CDN and caching

A CDN in front of object storage usually reduces both latency and egress cost, but only if the cache hit ratio is high. A distribution configured to cache nothing is pure overhead: you pay for the CDN request and the origin fetch.

Two configuration mistakes are common:

- **Caching signed URLs.** If your application generates short-lived signed URLs for private objects, and the CDN caches them, the cache key includes the signature. Every request is a cache miss, and worse, a cached response may outlive the URL's validity. Set the minimum, default, and maximum TTL to zero for these behaviours, or use signed cookies with a separate cache policy.
- **Caching methods that should not be cached.** POST, PUT, and DELETE requests should not be cached. Verify the allowed and cached method sets on each behaviour.

To measure cache effectiveness, use the CloudFront console's cache statistics, or query the access logs:

```
# After enabling CloudFront standard logging to S3 and querying with Athena:
SELECT
  count(*) AS requests,
  sum(CASE WHEN sc_status = 200 AND x_edge_result_type = 'Hit' THEN 1 ELSE 0 END) AS hits,
  sum(CASE WHEN x_edge_result_type = 'Miss' THEN 1 ELSE 0 END) AS misses
FROM cloudfront_logs
WHERE date = '2026-01-01'
```

A hit ratio below roughly 80% for static assets usually indicates a cache key that is too specific — often caused by forwarding query strings, headers, or cookies that do not affect the response.

### Observability

Hosted observability platforms bill by ingested volume: metrics, log bytes, trace spans, or some combination. Cost grows with traffic, and it grows fastest during incidents — exactly when you least want to be thinking about sampling.

Two controls matter:

- **Sampling.** Trace sampling at the SDK level, not the collector level, reduces both ingest cost and application overhead.
- **Log retention.** Logs are usually the largest volume. Set retention per log group rather than keeping everything forever.

```bash
aws logs describe-log-groups \
  --query 'logGroups[*].[logGroupName,retentionInDays,storedBytes]' \
  --output table
```

Any log group with `retentionInDays` set to null is retaining logs indefinitely. That is a default, not a decision.

## Building the attribution pipeline

Cost Explorer's console is fine for a rough look, but attribution by product area, customer tier, or environment requires tags and a queryable export.

The mechanism is the Cost and Usage Report (CUR), delivered to S3 and queryable via Athena. The setup is:

1. Enable CUR in the Billing console, with resource IDs included.
2. Point it at an S3 bucket with a lifecycle policy.
3. Create an Athena table over the Parquet output (AWS publishes a CloudFormation template for this).
4. Query by tag.

Tagging is the part that requires discipline. A tagging policy that covers `Environment`, `Service`, `Owner`, and `CostCenter` is enough to answer most questions. Untagged resources appear as a single line item, which is itself a useful signal — if the untagged total is more than a few percent, the tagging policy is not being enforced.

An illustrative query to split cost by service tag for a given month:

```sql
SELECT
  line_item_usage_account_id,
  resource_tags_user_service AS service,
  line_item_product_code AS product,
  sum(line_item_unblended_cost) AS cost
FROM cur_table
WHERE month = '1'
  AND year = '2026'
GROUP BY 1, 2, 3
ORDER BY cost DESC
LIMIT 50
```

Run this weekly. The point is not the report; it is the habit of looking at the trend before it becomes a surprise.

## A worked sizing example

The following is arithmetic from stated assumptions, not a measured bill. Assume:

- 500 monthly active users.
- Two application instances, each `t4g.medium`, running continuously.
- One managed PostgreSQL primary plus one standby, `db.r6g.large`, 200 GB storage.
- One Redis-compatible cache node, `cache.m6g.large`.
- 200 GB object storage, 1 TB monthly CDN egress.
- One NAT gateway.

Using published on-demand list prices for `af-south-1` at the time of writing (verify current prices before relying on them):

- Application instances: 2 × $0.0336/hr × 730 hr ≈ $49.06
- Database instances: 2 × $0.2268/hr × 730 hr ≈ $331.13
- Database storage: 200 GB × $0.138/GB-month ≈ $27.60
- Cache node: 1 × $0.166/hr × 730 hr ≈ $121.18
- Object storage: 200 GB × $0.0255/GB-month ≈ $5.10
- CDN egress: 1,000 GB × $0.085/GB ≈ $85.00
- NAT gateway: $0.062/hr × 730 hr ≈ $45.26, plus per-GB processing
- DNS, secrets, and observability: variable, often $50–$200 at this scale

The sum of the fixed components above is roughly $664 before data processing, observability, and any inter-region transfer. At 500 users that is about $1.33 per user per month for infrastructure alone — before the database backup storage, before NAT data processing, and before any DR replication.

The useful takeaway is not the total. It is the shape: the database instances are the largest single line, followed by the cache, followed by the CDN. That ordering tells you where optimization effort pays off first.

## Decision checklist

Before optimizing anything, answer these in writing:

- **Which region, and why?** Latency to the nearest user population, data residency requirements, and service availability all constrain this. Document the choice and the alternatives considered.
- **What is the failover target?** Single-AZ with fast restore, multi-AZ in one region, or multi-region. Each step roughly doubles the database cost.
- **What is the retention policy for backups, snapshots, and logs?** State it in days. Enforce it in infrastructure-as-code, not in a runbook.
- **What is the cache hit ratio target?** If it is not measured, it is not a target.
- **Which resources are untagged?** Anything untagged cannot be attributed.
- **What is the traffic growth assumption?** A stack sized for 500 users and a stack sized for 5,000 differ mainly in the database and observability lines.

## Common failure modes

**Snapshot accumulation.** Manual snapshots taken "just in case" and never deleted. Detect by listing snapshots older than the retention policy. Fix by setting retention in code.

**NAT gateway sprawl.** One per availability zone, plus no VPC endpoints, plus a service that talks to S3 constantly. Detect by filtering cost by `NatGateway-Bytes`. Fix by adding gateway endpoints for S3 and DynamoDB, and interface endpoints for other AWS services.

**Cache misses on signed URLs.** A CDN behaviour that caches signed requests produces both cost and correctness problems. Detect by comparing request count to hit count. Fix by setting TTLs to zero for signed behaviours and using signed cookies where caching is desired.

**Indefinite log retention.** Log groups created without a retention setting default to never expiring. Detect with `describe-log-groups`. Fix by setting retention at creation time in your infrastructure code.

**Observability ingest growth during incidents.** A retry storm or a verbose log level increases both the bill and the noise. Detect by graphing ingest volume against request volume. Fix with SDK-level sampling and per-service log levels.

## FAQ

**Should the application run in `af-south-1` or a European region?**

It depends on where the users are and what the application does. `af-south-1` reduces round-trip latency for users in southern and eastern Africa, but service availability and instance selection are narrower than in older regions. The way to decide is to measure: deploy a small test endpoint in each candidate region and record round-trip time from representative user locations. Do not choose based on a blog post, including this one.

**Is a managed cache worth it over running Redis on an instance?**

A managed cache removes operational work — patching, failover, backups — and adds cost. The break-even depends on how much an hour of engineering time is worth to your team and how much downtime costs. For small teams, the managed option is usually the right default; for teams with existing operational maturity and a strong reason to control the deployment, self-managed can be cheaper.

**How do I bill customers in local currency when AWS bills in USD?**

Track the exchange rate you use, the date you applied it, and the margin you added. Round consistently. The important thing for finance is that the rate and the margin are documented and stable, not that they are optimal.

**What is the cheapest way to get a cost breakdown by customer?**

Tag resources per customer where possible (rarely feasible), or attribute shared costs by a documented driver — request count, storage bytes, or seat count. The driver should be one you can measure per customer, not one you estimate.

## One action for the next 30 minutes

Pick the largest line item in your last full month of AWS spend, and find out what it actually is. Run:

```bash
aws ce get-cost-and-usage \
  --time-period Start=$(date -d '1 month ago' +%Y-%m-01),End=$(date +%Y-%m-01) \
  --granularity MONTHLY \
  --metrics UnblendedCost \
  --group-by Type=DIMENSION,Key=SERVICE \
  --output table
```

Then drill into the top service with `--group-by Type=DIMENSION,Key=USAGE_TYPE`. Write down the top three usage types and what resource each one corresponds to. If you cannot name the resource, that is the finding — an unattributable cost is the one most likely to grow without anyone noticing.
