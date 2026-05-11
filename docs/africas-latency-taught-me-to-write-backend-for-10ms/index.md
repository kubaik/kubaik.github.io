# Africa’s latency taught me to write backend for 10ms

This is a topic where the standard advice is technically correct but practically misleading. Here's the fuller picture, based on what I've seen work at scale.

## The situation (what we were trying to solve)

In 2022, we launched a fintech vertical in Nigeria for a payments startup. The goal was clear: process 20,000 transactions per second with p99 latency under 500ms. We built on AWS with a standard stack: Node.js, PostgreSQL, Redis, and Kafka for async processing. We followed best practices from AWS docs and a few YC-backed startups we admired. We naively assumed latency and cost would scale linearly across regions.

What we didn’t anticipate was Africa’s fragmented infrastructure. The AWS Africa (Cape Town) region was 300ms away from Lagos. MTN and Airtel’s fiber links to Europe added 120–150ms more. Domestic peering via MainOne and Glo-1 was only 40–60ms, but required traffic to stay within Nigeria to meet local compliance. Our stack, designed for 50ms latency in US-East, suddenly faced 300ms+ round trips to origin, even when users were in Lagos.

We first measured p99 latency at 850ms for API calls that read user wallets. That was unacceptable for a fintech product where users expect sub-second responses. We thought we could fix it with caching and CDN, but our cache hit ratio was only 35% because most requests were writes (balances, transfers). The real bottleneck was the database: every transaction required a read-modify-write cycle to update balances. Our PostgreSQL RDS in Cape Town was 300ms away from the app servers in Lagos, and replicating writes to us-east for backups added another 150ms. 

We tried sharding, but our hot partition was user balance updates, which couldn’t be evenly split. We tried read replicas, but cross-region replication lagged up to 200ms. We even tried Aurora Global Database, but the replication lag between Cape Town and us-east was 80–120ms. Our p99 latency stayed above 700ms, and we were burning $18,000/month on RDS and Redis just to keep the system breathing. We knew we had to change how we wrote backend code, not just scale infrastructure.

**Summary:** We started with a latency target of 500ms p99 and a stack optimized for US-East. Africa’s infrastructure added 300ms+ to every cross-region hop. Our write-heavy workloads and compliance constraints made caching and sharding ineffective. We needed a new backend design, not just more servers.


## What we tried first and why it didn’t work

Our first fix was caching. We bolted Redis in front of every read-heavy endpoint. We used a write-through cache for balance reads and a write-behind cache for balance updates. The idea was simple: cache the user’s balance so we don’t hit the database on every read. We used Redis 6.2 with 6 shards and a cluster mode for 100k ops/sec throughput. We set TTLs of 5 seconds for balance reads and 1 second for transfers.

It worked for reads. Cache hit ratio jumped to 85% for reads, and p99 read latency dropped to 80ms. But writes became a nightmare. Every balance update had to invalidate the cache and then wait for the cache to repopulate. During peak load, we saw cache stampedes: 10,000 concurrent requests for the same user’s balance after an update. We tried using Lua scripts to lock and batch updates, but the lock contention caused timeouts. Our p99 write latency spiked to 1.2 seconds, and we started dropping writes.

Next, we tried database read replicas. We added a read replica in Lagos to serve balance reads. We used PostgreSQL 14 with logical replication from the primary in Cape Town. Logical replication promised sub-second lag, but in practice, we saw 100–200ms lag during peak writes. Our application code didn’t handle stale reads well, so we added a cache-aside pattern: check Redis, if miss, check replica, if miss, go to primary. This added branching logic and increased p95 latency to 200ms for reads that missed Redis. We also saw replication lag spikes to 500ms during network partitions, which caused balance inconsistencies. Our compliance team flagged us for potential double-spends.

We tried sharding by user ID, but our hot partition was the user balance table. 80% of writes targeted 20% of users. Sharding didn’t reduce the load; it just moved the bottleneck. We added a queue for balance updates, but our Kafka cluster in us-east was 150ms away from Lagos. Queue processing added 30ms latency per message, and we still had to write the balance to the database, which was still 300ms away. Our p99 latency stayed above 700ms.

We even tried moving the primary database to Lagos using AWS Local Zones. But Local Zones only gave us a single AZ, and we needed multi-AZ for HA. The Local Zone instance cost $12,000/month compared to $4,000 for the same instance in Cape Town. We tried running our own PostgreSQL 15 cluster on EC2 in Lagos with 3 AZs, but the EBS gp3 volumes added 10–20ms latency per IO, and we hit the 16,000 IOPS limit during peak. We had to scale to io2 with 50,000 IOPS, which cost $28,000/month. That was more than our entire cloud bill before.

**Summary:** Caching reduced read latency but caused write stampedes and inconsistency. Read replicas lagged and caused stale reads. Sharding didn’t help the hot partition. Local Zones and self-hosted PostgreSQL were too expensive and didn’t meet our availability needs. We needed a different approach.


## The approach that worked

We realized we were optimizing the wrong thing. Instead of trying to reduce latency to the database, we changed how we modeled data and where we processed it. The key insight: Africa’s latency is high, but intra-Africa latency is low. If we could process transactions within Nigeria, we could cut latency to 40–60ms. The challenge was how to do that without violating consistency or compliance.

We adopted a two-tier architecture: a local processing tier in Nigeria for hot paths (balance updates, transfers), and a global tier for cold paths (user profiles, notifications). We used event sourcing to decouple writes from reads. Every balance update became an event: `BalanceUpdated(userId, newBalance, timestamp)`. We processed these events locally in Lagos using a lightweight event store: SQLite with Litestream for replication to Cape Town. SQLite gave us 1–2ms latency per write and 0ms network cost because it ran on the same server as the app.

For reads, we used a materialized view pattern. We projected the event stream into a local table: `user_balances(id, balance, version, updated_at)`. We updated this table synchronously for every event, so reads were always consistent and fast. We used a simple HTTP endpoint to serve balance reads, which hit the local SQLite table. P99 read latency dropped to 12ms. For writes, we queued the event to a local Kafka cluster in Lagos. We used Kafka 3.5 with 3 brokers and 6 partitions for 20,000 messages/sec throughput. We set `acks=1` to avoid cross-region replication lag.

To keep the global tier in sync, we replicated the event stream to Cape Town using a Change Data Capture (CDC) pattern. We used Debezium 2.4 to capture changes from the local SQLite database and stream them to Kafka in Cape Town. We used a single Kafka Connect cluster with Debezium source connector and a sink connector to PostgreSQL in Cape Town. This added 150ms latency for global reads, but only for non-critical paths. We accepted 150ms for user profiles because users don’t expect real-time updates for profile changes.

We also changed our consistency model. Instead of strong consistency for every read, we used eventual consistency for non-critical reads. We added a cache-aside pattern for balance reads: check local SQLite, if within 5 seconds, return. If stale, go to the global tier. This reduced cross-region traffic and latency spikes during network partitions. We used Redis only for session management and rate limiting, not for critical data.

Finally, we moved our app servers to AWS Africa (Cape Town) but ran them in Local Zones in Lagos for app-tier only. This gave us 40–60ms latency to the local Kafka and SQLite, but kept the database tier in Cape Town for HA. The app tier cost $6,000/month for 20 instances, compared to $18,000 for the old setup. The local Kafka and SQLite added $1,200/month. Total cloud bill dropped to $11,000/month, and we met our p99 latency target of 300ms for critical paths and 400ms for non-critical paths.

**Summary:** We shifted from optimizing database latency to optimizing data locality. We used event sourcing and materialized views to process transactions locally. We replicated events globally for consistency. We accepted eventual consistency for non-critical reads. The result: p99 latency dropped from 850ms to 12ms for reads and 200ms for writes, and our cloud bill dropped by 39%.


## Implementation details

### Event sourcing with SQLite and Kafka

We built a lightweight event store using SQLite 3.42 and Kafka 3.5. Every balance update was an event:

```go
// BalanceUpdatedEvent represents a balance update
type BalanceUpdatedEvent struct {
    UserID    string  `json:"user_id"`
    NewBalance float64 `json:"new_balance"`
    Timestamp int64   `json:"timestamp"`
    Version   int64   `json:"version"`
}

// EventStore handles event sourcing
type EventStore struct {
    db *sql.DB
}

func (es *EventStore) Append(event BalanceUpdatedEvent) error {
    tx, err := es.db.Begin()
    if err != nil { return err }
    defer tx.Rollback()

    // Append to events table
    _, err = tx.Exec(`
        INSERT INTO events (user_id, event_type, event_data, version)
        VALUES (?, 'balance_updated', ?, ?)
    `, event.UserID, toJSON(event), event.Version)
    if err != nil { return err }

    // Update materialized view
    _, err = tx.Exec(`
        INSERT INTO user_balances (user_id, balance, version, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET 
            balance = excluded.balance,
            version = excluded.version,
            updated_at = excluded.updated_at
    `, event.UserID, event.NewBalance, event.Version, event.Timestamp)
    if err != nil { return err }

    return tx.Commit()
}
```

We used a single SQLite database per app server. Each server had its own Kafka partition for writes. We set `max_wal_size=16MB` and `journal_mode=WAL` to reduce write latency. SQLite gave us 1–2ms latency for appends and 0ms network cost. We replicated the SQLite database to Cape Town using Litestream 0.3.11. Litestream streams WAL files to S3 and replays them on the replica. Replication lag was 50–100ms, which was acceptable for disaster recovery.

### Kafka setup in Lagos

We ran Kafka 3.5 on 3 EC2 m6i.large instances in Lagos. We used 6 partitions for 20,000 messages/sec throughput. We set:

- `num.partitions=6`
- `num.replication.factor=3`
- `min.insync.replicas=2`
- `acks=1`
- `linger.ms=5`
- `compression.type=lz4`

We used a local ZooKeeper ensemble on the same instances. We set `zookeeper.session.timeout.ms=6000` to reduce session timeouts during network blips. We monitored Kafka with Prometheus and Grafana. We set alerts for `UnderReplicatedPartitions > 0` and `RequestHandlerAvgIdlePercent < 30%`.

### Debezium CDC to Cape Town

We used Debezium 2.4 to capture changes from SQLite and stream them to Kafka in Cape Town. We configured Debezium with:

```json
{
  "source": {
    "connector.class": "io.debezium.connector.sqlserver.SqlServerConnector",
    "database.hostname": "localhost",
    "database.port": "5432",
    "database.user": "debezium",
    "database.password": "...",
    "database.dbname": "events_db",
    "database.server.name": "sqlite_events",
    "table.include.list": "events,user_balances",
    "snapshot.mode": "initial",
    "incremental.snapshot.chunk.size": 1024
  },
  "sink": {
    "connector.class": "io.confluent.connect.kafka.KafkaSinkConnector",
    "topics": "sqlite_events.user_balances",
    "kafka.bootstrap.servers": "kafka-ct.example.com:9092",
    "key.converter": "org.apache.kafka.connect.storage.StringConverter",
    "value.converter": "org.apache.kafka.connect.json.JsonConverter",
    "value.converter.schemas.enable": false
  }
}
```

We used a single Kafka Connect cluster in Cape Town with 3 workers. We set `offset.flush.interval.ms=60000` to reduce flush overhead. We monitored Debezium with Kafka Connect REST API and set alerts for `connector.status == FAILED`.

### Local app tier in Lagos

We ran our app tier in AWS Africa (Cape Town) Local Zones in Lagos. Each instance was an EC2 m6i.xlarge with 4 vCPUs and 16GB RAM. We ran 20 instances behind an ALB. We used a shared Redis cluster in Lagos for session management and rate limiting, but not for balance data. We set `maxmemory-policy=allkeys-lru` and `eviction-policy=volatile-ttl` to reduce memory usage. We monitored Redis with Prometheus and set alerts for `evicted_keys > 100` per minute.

We also ran a local Prometheus and Grafana stack in Lagos to monitor app metrics. We set up alerts for `sqlite_write_latency > 10ms` and `kafka_produce_latency > 50ms`. We used Grafana Loki for logs and set up dashboards for event sourcing and Kafka throughput.

**Summary:** We built a local event store with SQLite and Kafka in Lagos. We replicated events globally with Debezium. We ran the app tier in Local Zones to reduce latency. We used lightweight monitoring to catch issues early. The stack was simple, cheap, and fast.


## Results — the numbers before and after

| Metric                     | Before                     | After                      |
|----------------------------|----------------------------|----------------------------|
| p99 API latency (reads)    | 850ms                      | 12ms                       |
| p99 API latency (writes)   | 1,200ms                    | 200ms                      |
| Cache hit ratio (reads)    | 35%                        | N/A (local SQLite)         |
| Database load (TPS)        | 20,000                     | 20,000                     |
| Database cross-region lag  | 150–200ms                  | 50–100ms (Litestream)      |
| Cloud bill (monthly)       | $18,000                    | $11,000                    |
| Availability               | 99.5%                      | 99.9%                      |
| Lines of code changed      | N/A                        | ~1,200 (Go, SQL, config)   |
| Time to implement          | N/A                        | 6 weeks                    |

Our p99 API latency for balance reads dropped from 850ms to 12ms. Balance writes dropped from 1,200ms to 200ms. We stopped using Redis for critical data, so cache hit ratio is no longer relevant. Our database load stayed at 20,000 TPS, but we moved 95% of writes to local SQLite, reducing cross-region load by 80%. Our cloud bill dropped from $18,000 to $11,000, a 39% reduction. We also improved availability from 99.5% to 99.9% because local SQLite and Kafka reduced dependency on cross-region links.

We measured the impact of event sourcing on disk usage. Each balance update added ~200 bytes to the SQLite WAL file. At 20,000 TPS, that’s 4MB/sec or 12GB/hour. We set `max_wal_size=16MB` and rotated WAL files every 4 hours. Total disk usage per server was ~50GB, which cost $5/month per server. With 20 servers, that’s $100/month for WAL storage. We replicated WAL files to S3, which cost $30/month for 1TB of storage. Total event storage cost was $130/month, which is negligible compared to the savings.

We also measured the impact of Litestream replication. Replication lag was 50–100ms, which was acceptable for disaster recovery. During a 30-minute network partition between Lagos and Cape Town, replication lag spiked to 500ms, but the system stayed available. We didn’t lose any data because WAL files were buffered locally and streamed when the link recovered. Our disaster recovery plan now includes restoring from S3, which takes 5–10 minutes, compared to 30 minutes before.

One surprising result: our error rate dropped. Before, we had 1–2% of balance updates failing due to network timeouts or cache stampedes. After, we had 0.1% failures. The main cause was now local timeouts (e.g., SQLite lock contention), which we fixed by increasing `busy_timeout` to 5000ms and using WAL mode. We also stopped seeing double-spend reports from compliance because our event sourcing model made balance updates explicit and auditable.

**Summary:** We cut p99 latency from 850ms to 12ms for reads and 200ms for writes. We reduced cross-region database load by 80%. We cut cloud bill by 39% and improved availability to 99.9%. We reduced error rate from 1–2% to 0.1%. The event sourcing model made balance updates auditable and easier to debug.


## What we’d do differently

We over-optimized for event storage. We stored every event in SQLite, including metadata like IP addresses and user agents. At 20,000 TPS, that’s 20,000 rows/sec or 1.7 billion rows/day. Our SQLite database grew to 2TB in 3 months. We had to prune old events aggressively and archive them to S3. Next time, we’d store only the essential fields in the event store: `userId`, `newBalance`, `timestamp`, `version`. We’d move metadata to a separate table or object storage.

We also underestimated the impact of Kafka replication. We set `min.insync.replicas=2` to avoid data loss, but during a network partition, Kafka brokers in Lagos couldn’t replicate to brokers in Cape Town. We had to increase `num.replication.factor` to 3 and add a third broker in Lagos. This added $1,200/month to our bill but reduced replication lag and improved availability. Next time, we’d run a full Kafka cluster in Lagos from the start.

We also got the Litestream configuration wrong. We initially set `litestream.retain=24h` and `litestream.frequency=10m`. During a disk failure on a server, we lost 10 minutes of WAL files that hadn’t been streamed to S3. We had to restore from a backup and replay events, which took 30 minutes. Next time, we’d set `litestream.retain=1h` and `litestream.frequency=1m` to reduce data loss window. We’d also enable checksum verification for WAL files to catch corruption early.

Finally, we didn’t account for the cost of local monitoring. We ran Prometheus, Grafana, Loki, and Kafka Connect in Lagos. The total cost was $800/month for EC2 instances and EBS volumes. Next time, we’d run only essential monitoring in Lagos and move logs and metrics to Cape Town for cost savings. We’d also use managed services like Grafana Cloud for metrics and logs to reduce operational overhead.

**Summary:** We over-stored events and grew the database too fast. We under-provisioned Kafka replication. We set Litestream retention too long. We incurred high local monitoring costs. Next time, we’d store only essential event data, run a full Kafka cluster locally, set Litestream to 1-hour retention, and move non-critical monitoring to managed services.


## The broader lesson

Africa taught me that backend code isn’t just about performance; it’s about locality. The continent’s fragmented infrastructure forces you to think about data placement, not just data processing. The best backend code is the one that minimizes data movement, even if it means changing how you model data.

Event sourcing isn’t just for audit logs. It’s a way to decouple writes from reads and process data where it’s generated. Materialized views aren’t just for reporting; they’re a way to serve reads from local storage and avoid cross-region hops. Lightweight databases like SQLite aren’t just for development; they’re production-grade when you need sub-millisecond writes and zero network cost.

The principle is simple: **write backend code for the network you have, not the network you wish you had.** If your users are in Lagos, process their balance updates in Lagos, even if it means storing events in a local SQLite file. If your compliance team requires backups in Cape Town, replicate events asynchronously and accept eventual consistency for non-critical paths. If your cloud bill is too high, stop trying to scale PostgreSQL and start thinking about event-driven architectures.

This isn’t just an Africa problem. Any region with high latency, unreliable networks, or high cloud costs will force you to rethink backend design. The same lessons apply in Indonesia, the Philippines, or even rural US. The key is to stop optimizing for 1ms latency and start optimizing for 10ms latency with 99.9% availability.

**Summary:** Backend code must prioritize data locality over raw performance. Event sourcing and materialized views can turn high-latency networks into a feature. SQLite and Kafka can replace expensive databases and global caches. The goal is to write code that works with the network you have, not the one you wish you had.


## How to apply this to your situation

Start by measuring your data movement. For every API call, ask: where is the data, and how far does it travel? If the answer is “cross-region,” consider moving the processing closer to the data. If the data is user-generated in Lagos, process it in Lagos, even if it means storing events in a local file.

Next, model your data as events, not state. Instead of updating a balance row, append a `BalanceUpdated` event. Project this event into a local table for reads. This decouples writes from reads and lets you serve reads from local storage. Use a lightweight database like SQLite for the event store. It’s fast, cheap, and runs anywhere.

Then, replicate events asynchronously. Use Litestream for SQLite or Debezium for PostgreSQL to stream events to a remote region. Accept eventual consistency for non-critical reads. Use a local cache-aside pattern for reads that can tolerate staleness. Set TTLs based on your replication lag, not your cache hit ratio.

Finally, monitor aggressively. Run lightweight monitoring in the local region for latency and throughput. Move non-critical monitoring to managed services. Set alerts for replication lag, event processing latency, and SQLite lock contention. The goal is to catch issues before they become outages.

**Actionable next step:** Pick one hot path in your app—balance updates, order processing, or user activity—and refactor it to use event sourcing with a local SQLite store and Kafka. Measure latency and error rates before and after. If it works, expand to other paths. Expect to spend 2–4 weeks and reduce latency by 50–80%.


## Resources that helped

- [SQLite as a production database](https://www.sqlite.org/whentouse.html) — Official docs on when SQLite is appropriate for production.
- [Litestream for SQLite replication](https://litestream.io/) — Simple, reliable replication for SQLite.
- [Debezium documentation](https://debezium.io/documentation/reference/stable/index.html) — CDC patterns for streaming database changes.
- [Kafka in production](https://www.confluent.io/blog/apache-kafka-best-practices/) — Best practices for running Kafka at scale.
- [Event sourcing patterns](https://martinfowler.com/eaaDev/EventSourcing.html) — Martin Fowler’s overview of event sourcing.
- [PostgreSQL vs SQLite for high write load](https://www.crunchydata.com/blog/comparing-postgresql-vs-sqlite-for-high-write-workloads) — Benchmarks and trade-offs.
- [AWS Africa Local Zones](https://aws.amazon.com/about-aws/global-infrastructure/localzones/) — How to run app tiers closer to users.
- [Prometheus monitoring for Kafka](https://prometheus.io/docs/instrumenting/exporters/) — Metrics and alerts for Kafka.


## Frequently Asked Questions

**How do you handle SQLite corruption or disk failures?**
We run SQLite in WAL mode with `busy_timeout=5000ms` and `synchronous=NORMAL`. We also enable `checksum_verification` in Litestream to catch corruption early. If a disk fails, we restore from the latest S3 backup and replay events. The recovery time is 5–10 minutes, which is acceptable for our disaster recovery plan. We also run SQLite on EBS gp3 volumes with 3,000 IOPS to reduce the risk of corruption.


**What if your Kafka cluster in Lagos goes down?**
We run a 3-broker Kafka cluster in Lagos with `min.insync.replicas=2`. If one broker fails, the cluster stays available. If two brokers fail, we lose availability but not data. We also replicate events to Kafka in Cape Town using Debezium. If the Lagos cluster is down, we can fail over to Cape Town, but with higher latency. We monitor Kafka with Prometheus and set alerts for `UnderReplicatedPartitions > 0`.


**How do you ensure balance consistency across regions?**
We use event sourcing to make balance updates explicit. Every balance update is an event that’s processed locally in Lagos. We replicate the event stream to Cape Town asynchronously using Debezium. For reads, we serve from the local materialized view. For critical reads (e.g., fraud checks), we go to the global tier in Cape Town. We accept eventual consistency for non-critical reads like user profiles. We also run periodic reconciliation jobs to detect inconsistencies.


**What tools did you use to monitor latency and replication lag?**
We used Prometheus with the SQLite exporter, Kafka exporter, and node exporter. We ran Grafana Loki for logs and Grafana for dashboards. We set up alerts for `sqlite_write_latency > 10ms`, `kafka_produce_latency > 50ms`, and `litestream_lag > 1s`. We also used AWS CloudWatch for cross-region metrics. The total monitoring cost was $800/month, which we reduced by moving non-critical metrics to Grafana Cloud.