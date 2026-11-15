# Multi-region active-active: the hidden costs

Multi-region active-active sounds like the ultimate resilience play: deploy to several regions, route users to the nearest one, and let the cloud providers handle the rest. The marketing pages for AWS, Azure, and Google Cloud all promise global load balancing, multi-region replication, and automatic failover. The gap between those documented features and a working active-active system is wide, and it is filled with costs that do not show up until a team is already committed.

The core problem is that active-active is not a product you buy; it is a distributed systems problem you build. The cloud providers give you primitives—global load balancers, cross-region replication for storage and databases, and multi-region networking—but they do not give you a turnkey active-active architecture. Teams have to stitch those primitives together in a way that handles latency, consistency, and failure modes, and they have to do it without breaking the budget or the on-call rotation.

This article covers the technical, operational, and financial costs of designing active-active across AWS, Azure, and Google Cloud. It is written for teams that have outgrown single-region deployments and are considering multi-region for latency or resilience, and it assumes familiarity with VPCs, load balancers, and managed databases. By the end you should have a framework for deciding whether active-active is worth it, and, if so, how to avoid the most common traps.

## The gap between the docs and your p99

Cloud providers document their global load balancers as routing users to the nearest healthy region. In practice, routing is based on latency measurements that can be stale, and health checks are often too coarse to detect partial failures. A typical failure mode is a global load balancer that keeps sending traffic to a region that is degraded but not completely down—say, a database responding slowly because of a hot shard. The health check might only verify that the load balancer can reach an instance, not that the application is actually serving requests correctly.

The result is a p99 latency spike with no failover, because the health check passes. Teams commonly discover this pattern during incident reviews rather than during testing. The fix is to implement application-level health checks that exercise the full request path, including dependencies, and to configure the load balancer to use those checks. That is only the beginning.

Another gap is data replication. AWS, Azure, and Google Cloud all offer cross-region replication for object storage and managed databases, but the consistency models vary. Asynchronous replication means a failover can lose recent writes, and the size of that window is a property of the specific service, not a constant. Teams that assume replication is instantaneous are surprised when a regional failover silently discards the last few seconds or minutes of writes.

The practical lesson: treat every replication guarantee as a number you must look up for your exact service, configuration, and region pair, then design your recovery point objective (RPO) around it. The provider's documentation is the source of truth; a design review should cite the specific document and configuration, not a general impression that "it replicates."

## The mechanism: why active-active is hard

Active-active means that multiple regions serve traffic simultaneously. That is different from active-passive, where one region handles traffic and another stands by. The moment a system goes active-active, it introduces the possibility of conflicting writes, and it needs a strategy for conflict resolution. It also needs a way to route users to the closest region without introducing excessive latency, and it needs to handle the failure of any region without impacting the others.

The fundamental challenge is data consistency. If a user in Europe writes to a European region and a user in the US writes to a US region, and those writes affect the same data, there is a conflict. There are several approaches:

1. **Single-writer per partition**: Shard data so that each partition has a single writer region. For example, user data is partitioned by user ID, and each user is assigned a home region. All writes for that user go to the home region. Reads can be served from any region, but they may be stale. This is the simplest model and works well for many applications.

2. **Conflict-free replicated data types (CRDTs)**: Use data structures that can be merged automatically without conflicts. This works for certain types of data (counters, sets, and similar) but is not a general solution.

3. **Last-write-wins (LWW)**: Use timestamps to resolve conflicts. This is simple but can lose data if clocks are skewed or if writes are concurrent.

4. **Consensus protocols**: Use a distributed consensus protocol such as Raft or Paxos to coordinate writes across regions. This is complex and adds latency.

Most teams end up with a combination: single-writer per partition for most data, and CRDTs or LWW for specific cases. The key is to design the data model with multi-region in mind from the start. Retrofitting active-active onto a single-region schema is painful.

### A worked conflict example

Consider an inventory counter: `stock = 10`. A US region sells 3 units and a European region sells 4 units at nearly the same moment, each writing to its local replica.

- **Last-write-wins** keeps one write and discards the other. If the US write lands last, the stored value is `10 - 3 = 7`, but 7 units were actually sold, so the system has oversold by 4. The arithmetic here is illustrative, but the failure is real: LWW loses one of two concurrent decrements.
- **CRDT counter (a grow-only or positive-negative counter)** stores each region's delta separately and merges by summing: `10 - 3 - 4 = 3`. No sale is lost, and the merge is commutative, so the order of arrival does not matter. The cost is that the counter's representation is larger and reads must aggregate deltas.
- **Single-writer per partition** routes both sales to the partition's home region, which serializes them: `10 - 3 = 7`, then `7 - 4 = 3`. Correct, but the non-home region's write now pays a cross-region round trip.

The choice is not abstract: it determines whether the inventory system can oversell under partition. Decide per data type, and write the decision down.

## Implementing active-active on AWS, Azure, and Google Cloud

Each cloud provider offers a different set of primitives for building active-active. The table below compares the categories that matter; verify current limits and behavior against provider documentation before committing, since these services evolve.

| Capability | AWS | Azure | Google Cloud |
|---|---|---|---|
| Global traffic routing | Route 53 (DNS, latency/geo routing, health checks); Global Accelerator (anycast) | Azure Front Door (layer 7); Traffic Manager (DNS) | Cloud Load Balancing (global anycast); Cloud DNS |
| Multi-active database | DynamoDB Global Tables (LWW conflict resolution) | Cosmos DB (configurable consistency, conflict resolution policies) | Cloud Spanner (externally consistent, multi-region configs) |
| Relational active-passive | Aurora Global Database (one writer region, read replicas) | Azure SQL active geo-replication (one writer, readable secondaries) | Cloud SQL (regional; read replicas) |
| Object storage replication | S3 Cross-Region Replication (asynchronous) | Blob Storage GRS/RA-GRS (asynchronous) | Cloud Storage multi-region/dual-region (synchronous within the region pair) |

### AWS

AWS provides several building blocks:

- **Route 53**: Global DNS with latency-based routing, geolocation routing, and health checks. It can route users to the nearest region, but DNS caching can delay failover, and health checks are limited.
- **Global Accelerator**: Anycast IP addresses that route traffic to the nearest AWS edge location, then over the AWS global network to the application. This provides faster failover than DNS and avoids caching issues, at an additional cost.
- **Application Load Balancer (ALB)**: A regional load balancer; one is needed per region.
- **Aurora Global Database**: A primary region for writes and secondary regions for reads. Replication is asynchronous, so failover to a secondary can lose recent writes, and promotion is a deliberate operation rather than an instant switch.
- **DynamoDB Global Tables**: Multi-region, multi-active database with last-writer-wins conflict resolution. This is one of the few truly active-active databases on AWS.
- **S3 Cross-Region Replication**: Asynchronous replication of objects.

A typical AWS active-active architecture uses Route 53 or Global Accelerator for traffic routing, ALBs in each region, and DynamoDB Global Tables for data. For relational data, Aurora Global Database with a single writer region is common, but that is active-passive for writes.

### Azure

Azure's active-active primitives include:

- **Azure Front Door**: Global HTTP load balancer with latency-based routing, health probes, and failover. It operates at layer 7.
- **Azure Traffic Manager**: DNS-based global load balancer, similar in role to Route 53.
- **Azure Load Balancer**: Regional layer 4 load balancer.
- **Azure Cosmos DB**: Multi-region, multi-active database with configurable consistency levels and conflict resolution policies (last-write-wins or custom). This is Azure's flagship active-active database.
- **Azure SQL Database**: Geo-replication for active-passive. Active geo-replication allows readable secondaries, but writes go to the primary.
- **Azure Blob Storage**: Geo-redundant storage (GRS) replicates asynchronously; read-access geo-redundant storage (RA-GRS) allows reads from the secondary region.

Azure's active-active story is strongest with Cosmos DB. For relational data, the model is active-passive.

### Google Cloud

Google Cloud offers:

- **Cloud Load Balancing**: Global load balancer with anycast IPs, latency-based routing, and health checks. A single global anycast IP simplifies routing.
- **Cloud DNS**: Global DNS with latency-based routing.
- **Cloud Spanner**: Globally distributed, strongly consistent database with multi-region configurations. Spanner provides external consistency across regions, which means active-active writes without application-level conflict resolution. The trade-off is higher write latency, because of the consensus protocol, and cost.
- **Cloud SQL**: Regional database with read replicas; not multi-region active-active.
- **Cloud Storage**: Multi-region and dual-region buckets with synchronous replication within the region pair.

Google Cloud's active-active story is strongest with Spanner. For many teams, Spanner's consistency guarantees are worth the cost and latency, but it is not a drop-in replacement for MySQL or PostgreSQL.

## A concrete implementation: active-active with DynamoDB Global Tables

A minimal active-active setup on AWS using DynamoDB Global Tables is one of the simpler paths because DynamoDB handles replication and conflict resolution.

**Architecture:**

- Two regions, for example us-east-1 and eu-west-1.
- Route 53 latency-based routing to ALBs in each region.
- Each region has an ALB and an Auto Scaling group of EC2 instances running the application.
- A DynamoDB Global Table replicated between the two regions.
- The application reads and writes to the local DynamoDB endpoint.

**Code example (Python with boto3):**

```python
import boto3
from botocore.config import Config

# Configure the DynamoDB client to use the local region's endpoint.
# In practice, set the region via environment variable or instance metadata.
region = 'us-east-1'  # or 'eu-west-1'
dynamodb = boto3.resource('dynamodb', region_name=region, config=Config(retries={'max_attempts': 3}))

table = dynamodb.Table('MyGlobalTable')

# Write to the local region
table.put_item(Item={'PK': 'user#123', 'SK': 'profile', 'name': 'Alice', 'updated_at': '2025-01-01T00:00:00Z'})

# Read from the local region
response = table.get_item(Key={'PK': 'user#123', 'SK': 'profile'})
print(response['Item'])
```

This code writes to and reads from the local region's DynamoDB endpoint. DynamoDB Global Tables replicates the write to the other region asynchronously. Conflicts are resolved using last-writer-wins, so if two regions write to the same item concurrently, the write with the later timestamp wins and the earlier one is lost. For some applications that is acceptable; for others it is not.

**Important:** DynamoDB Global Tables require that the table have streams enabled and that replication be configured. The application should also handle the case where a region is unavailable by catching exceptions and retrying against the other region's endpoint—but note that a blind retry to another region can itself create a conflict, so retries should be idempotent or gated on the failure being regional rather than request-specific.

## Measuring what matters: latency, replication lag, and failover time

You cannot manage what you do not measure. For active-active, the key metrics are:

- **User-facing latency**: p50, p95, p99 by region. Use synthetic monitoring from multiple locations to measure the latency users actually experience. AWS CloudWatch Synthetics, Azure Application Insights availability tests, and Google Cloud Monitoring uptime checks can do this.
- **Replication lag**: For databases, measure the time between a write in one region and its visibility in another. For DynamoDB Global Tables, use the `ReplicationLatency` CloudWatch metric. For Aurora Global Database, use `AuroraGlobalDBReplicationLag`. For Spanner, use its replication lag metric. Confirm the exact metric name in current documentation for your service version.
- **Failover time**: The time from a region failure to traffic being served by another region. This includes health check detection time, DNS TTL, and load balancer failover. Test it regularly with chaos engineering tools.
- **Error rates**: By region and by dependency. A spike in errors in one region might indicate a partial failure that health checks miss.

A useful practice is a dashboard that shows these metrics side by side for all regions, so degradation is visible immediately rather than after a customer complaint.

### How to measure replication lag without a benchmark table

Rather than trusting a vendor's "typically under a second" figure, measure it in your own environment:

1. Instrument the write path to emit a monotonic write timestamp (from the database's own clock, not the application host) with each record.
2. In the secondary region, read the record and compute `now - write_timestamp`. Record it as a histogram, not an average.
3. Run this continuously at a low rate (for example, one probe write every few seconds) and alert on the p99, not the mean, because the tail is what determines your RPO.
4. Compare the p99 against the RPO in your design. If the p99 exceeds the RPO, either tighten the RPO's assumptions or change the replication strategy.

The same approach works for failover time: record the timestamp when a region's health check first fails and the timestamp when the first request is served by the standby, and treat the difference as a measured distribution.

## Failure modes that active-active doesn't fix

Active-active is not a silver bullet. It does not protect against certain failure modes, and in some cases it makes them worse.

- **Correlated failures**: If a bad configuration change is deployed to all regions simultaneously, active-active does not help. It can make the outage worse because all regions fail at once. Canary deployments and fast rollback are the mitigations.
- **Data corruption**: If a bug writes corrupt data, active-active replicates it to all regions. Backups and point-in-time recovery are the mitigations.
- **Dependency failures**: If a shared dependency (such as an external API) fails, active-active does not help. Circuit breakers and fallbacks are the mitigations.
- **Split-brain**: In a partitioned network, regions might not be able to communicate. With a consensus protocol, the minority partition becomes unavailable. With last-writer-wins, conflicting writes can be silently resolved. The team must decide which trade-off is acceptable, in advance.
- **Cost**: Active-active roughly multiplies infrastructure cost by the number of active regions. A team pays for capacity in each region to handle failover, plus cross-region data transfer. For many teams, the cost outweighs the benefits.

## When active-active is the wrong choice

Active-active is not for everyone. It is the wrong choice when:

- **Users are concentrated in one geography**: If the vast majority of users are in one country, a single region in that country with a disaster recovery region might be sufficient. Active-active adds complexity and cost without much benefit.
- **The application is stateful and hard to partition**: A monolithic database with complex transactions makes active-active very difficult. A single-region primary and a warm standby may be better.
- **The team lacks operational maturity**: Active-active requires robust monitoring, automated failover, and a team that can handle the complexity. If deployments are still manual, active-active will be a nightmare.
- **The budget is tight**: Active-active is expensive. For a small startup, active-active on a major cloud is usually not the right first investment. Even at a growth-stage company, the cost can be significant.

A common pattern is to start with active-passive (a single region with a standby) and only move to active-active when the business case is clear. For many applications, a multi-AZ single-region deployment with good backups and a disaster recovery plan is sufficient.

### Decision checklist

Answer these before committing:

1. What is the measured cost of one hour of downtime (revenue, support load, contractual penalties)?
2. What are the RTO and RPO requirements, and where do they come from—business, contract, or habit?
3. Which data sets can tolerate LWW, which need CRDTs, and which need single-writer partitioning?
4. Can the team detect a partial failure (degraded but alive) within minutes, and does the load balancer act on it?
5. Has failover been measured, or only designed?
6. What is the estimated monthly cost of the second and third active regions, including cross-region data transfer?
7. Who is on call for a regional failover, and has that runbook been exercised?

If questions 4 through 6 cannot be answered with numbers, the team is not yet ready for active-active.

## Frequently Asked Questions

**How do I choose between active-active and active-passive?**

Start by quantifying the cost of downtime. If the application can tolerate a few hours of downtime per year, active-passive is likely sufficient and much simpler. If downtime costs thousands of dollars per minute, active-active may be justified. Also consider latency requirements: if users are globally distributed and latency is a competitive differentiator, active-active can help. Be honest about the operational complexity—active-active requires a dedicated effort to build and maintain.

**What is the cheapest way to do multi-region on AWS?**

The cheapest approach is typically a single-region deployment with cross-region backups and a disaster recovery plan. If multi-region is needed for latency, a read-only replica in a second region with reads routed there, while writes stay in one region, is active-passive for writes but can improve read latency. For true active-active, DynamoDB Global Tables and S3 Cross-Region Replication are relatively low-cost compared to running full compute in multiple regions.

**How do I handle data conflicts in active-active?**

The best approach is to avoid conflicts by design: partition data so that each piece has a single writer region. If that is not possible, choose a conflict resolution strategy that fits the data. Last-writer-wins is simple but can lose data. CRDTs are safe but limited in the data types they support. Consensus protocols are strong but add latency. Many teams use a combination: single-writer for critical data, LWW for non-critical data, and CRDTs for specific use cases such as counters.

**Can I use Kubernetes for multi-region active-active?**

Yes, but Kubernetes alone does not solve the data consistency problem. Kubernetes clusters can run in multiple regions with a global load balancer routing traffic. For data, a multi-region database or a replication strategy is still required. Tools such as Karmada or Cluster API can help manage multiple clusters, but they add complexity. For many teams, managed services such as DynamoDB or Cosmos DB are simpler than running stateful workloads across regions.

## Next step: audit your architecture for multi-region readiness in 30 minutes

Open your infrastructure-as-code repository and search for hardcoded region references, single-region database endpoints, and stateful components that would be difficult to replicate. List the top three dependencies that would need to be multi-region aware. Then, using the cloud provider's pricing calculator, estimate the monthly cost of running the same compute, storage, and cross-region data transfer in a second region. Write the three numbers—estimated second-region cost, current RPO, and measured or assumed failover time—on one page. If any of them is unknown, that unknown is the first thing to resolve before designing anything.
