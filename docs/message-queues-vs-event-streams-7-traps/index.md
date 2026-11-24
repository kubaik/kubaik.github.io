# Message queues vs event streams: 7 traps

The first time you wire a service to a queue and a service to a stream, they look interchangeable: both accept messages from a producer, both deliver them to a consumer, both promise not to lose them. Then the consumer that reads from the stream replays a week of history by accident, or the queue consumer blocks for 30 seconds on a slow job and the backlog doubles, and the distinction stops being academic. The real problem this post solves is that the two systems have different contracts for *what a message means after it has been delivered* — and most of the pain comes from picking the one whose contract does not match the work.

Below are seven traps, framed as the specific behaviors that force the choice. Each one names the mechanism that causes it, the situation where it applies, and the situation where it breaks. I have a bias: I think most teams reach for a log-based stream (Kafka, Kinesis, Pulsar) when a plain queue (SQS, RabbitMQ, a managed queue service) would be simpler and cheaper. But there are several traps where the queue is the wrong tool and the stream is the only sane option, and those are called out too.

## 1. The replay trap: a queue deletes, a stream retains

**What it does.** A message queue is a delivery mechanism. Once a consumer acknowledges a message, the broker removes it. RabbitMQ, SQS, and most managed queue services work this way. An event stream is a durable log. Kafka, Kinesis, and Pulsar keep records for a configured retention period regardless of who has consumed them. Consumers track their own offset.

**Strength (queue).** The mental model is simple: one message, one consumer, gone. You never have to think about retention or offset management.

**Weakness (queue).** You cannot replay. If a consumer has a bug that silently drops records, the data is gone unless you built an archive on the producer side. Adding a second consumer to an existing queue means changing the producer to fan out to two destinations.

**Strength (stream).** A new consumer can start from the beginning of the retention window and rebuild state. This is the single most useful property of a log. A common pattern is to run a new version of a service against the same topic for a day, compare its output to the old version, then cut over.

**Weakness (stream).** Retention costs money and creates a second problem: consumers that fall behind retention get an error. In Kafka, the consumer offset is reset according to `auto.offset.reset` (commonly `latest` or `earliest`); in Kinesis, a shard iterator expires after 5 minutes, and a consumer that stops for longer has to resume from a checkpoint. If you have never seen a consumer silently skip a day of data because its offset was out of range, you have not run a stream in production long enough.

**Who it is for.** Use a queue when the message is a command — "send this email", "charge this card" — and the work is done when it is done. Use a stream when the message is a fact — "order 123 was placed" — and more than one system may care, now or later.

## 2. The consumer-group trap: a queue fans out by hand, a stream fans out by config

**What it does.** In a queue, if you want two independent services to react to the same event, the producer publishes to two queues (or to an exchange with two bindings in RabbitMQ, or to an SNS topic with two SQS subscriptions on AWS). In a stream, you create two consumer groups on the same topic and each gets the full stream.

**Strength (stream).** Adding a consumer is a configuration change on the consumer side. The producer does not know or care. This is the property that makes event streams the default backbone for event-driven architectures: the producer publishes once, and the set of interested parties can change without touching it.

**Weakness (stream).** Consumer groups are a shared namespace with subtle semantics. In Kafka, two consumers in the same group split partitions; two consumers in different groups each get everything. Teams regularly create a second consumer with the same `group.id` by copy-paste and then wonder why each one sees half the messages. The error is silent — no exception, just half the data.

**Strength (queue).** The fan-out is explicit. You can look at the topology and see who receives what. For a small number of consumers, this is easier to reason about.

**Weakness (queue).** Every new consumer is a producer change and a deploy. At three consumers this is fine; at fifteen it is a coordination problem.

**Who it is for.** If you expect the number of consumers to grow, or if consumers are owned by different teams, the stream's decoupling is worth the operational cost. If you have one producer and one consumer and expect that to stay true, a queue is less machinery.

## 3. The backpressure trap: a queue buffers, a stream partitions

**What it does.** Queues absorb bursts by holding messages until a consumer is free. Streams absorb bursts by appending to a partitioned log, and the consumer's progress is measured by lag per partition.

**Strength (queue).** A queue with a visibility timeout is a natural fit for work that takes variable time. A consumer receives a message, processes it, and deletes it. If it crashes, the message becomes visible again after the timeout and another consumer picks it up. This is the at-least-once retry model, and it is exactly what you want for a job queue.

**Weakness (queue).** Ordering is not guaranteed in a standard queue. SQS standard queues explicitly provide best-effort ordering; FIFO queues provide ordering within a message group but cap throughput. If your work requires strict order, you have to design around it.

**Strength (stream).** Ordering is guaranteed within a partition. Kafka preserves order per partition; Kinesis preserves order per shard. This is the property that makes streams suitable for change data capture and for anything where "state after event N" must be reconstructable.

**Weakness (stream).** A slow consumer does not slow the producer, but it does accumulate lag, and lag is not free: it is the gap between the head of the log and your offset. If the consumer's throughput is below the producer's, lag grows without bound until retention deletes data the consumer has not read. The fix is to increase partition count and consumer parallelism, but partition count is not freely changeable in every system, and increasing it changes the key-to-partition mapping, which breaks per-key ordering for in-flight data.

**Who it is for.** Use a queue when the unit of work is independent and retryable. Use a stream when the order of events matters or when you need to reconstruct state by replaying.

## 4. The delivery-semantics trap: both are at-least-once, neither is exactly-once for free

**What it does.** Almost every queue and stream offers at-least-once delivery by default. Exactly-once is either a transactional feature with caveats or a property you build by making the consumer idempotent.

**Strength (both).** At-least-once is honest: you will not lose the message, but you may see it twice. That is a contract you can build on if you design for it.

**Weakness (both).** Teams read "exactly-once" in a feature list and assume it applies end to end. It usually applies within the broker's boundary: Kafka transactions can make a produce-and-commit atomic across Kafka topics, but they do not make your database write atomic with the Kafka offset commit. The classic bug is: consumer writes to the database, then crashes before committing the offset; on restart it reprocesses and writes a duplicate row.

**The mechanism that actually works.** Make the consumer idempotent. The simplest version is a deduplication key with a unique constraint:

```sql
CREATE TABLE processed_events (
  event_id     UUID PRIMARY KEY,
  processed_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

The consumer inserts the event ID in the same transaction as the business write. A duplicate insert fails on the primary key, and the consumer treats that as success (the work is already done). This costs one extra table and one extra write per event, and it makes the delivery semantics a property of your code rather than a promise from the broker.

**Who it is for.** Everyone. If you are choosing between a queue and a stream on the basis of exactly-once, you are choosing on a feature that will not save you from the duplicate-write bug.

## 5. The operations trap: a queue is a service, a stream is a cluster

**What it does.** A managed queue is usually a URL and a set of permissions. A stream is usually a topic with partitions, a retention policy, a replication factor, and a set of consumer groups with committed offsets.

**Strength (queue).** The operational surface is small. There is no partition count to choose, no retention to tune, no rebalance to observe. For a team without a dedicated platform engineer, this matters.

**Weakness (queue).** You give up control. Throughput limits, message size limits, and visibility timeout maximums are set by the provider. Hitting a limit is a support ticket, not a config change.

**Strength (stream).** You control partitioning, retention, and replication. You can size the cluster to the workload.

**Weakness (stream).** You now own a distributed system. Consumer rebalances stop the world for the consumer group; a broker going down triggers leader election; disk fills up if retention outpaces disk. A managed stream service removes some of this, but not the partition-count decision or the consumer-lag alerting.

**Who it is for.** A queue if your team is small and the workload is request/response-shaped. A stream if you have the operational capacity or are paying a managed provider to have it for you, and the workload is genuinely event-shaped.

## 6. The schema trap: a queue carries a payload, a stream carries a contract

**What it does.** A queue message is usually a blob. A stream record is usually a blob too, but because it is retained and replayed, it becomes a de facto API between producers and consumers that may be deployed weeks apart.

**Strength (queue).** You can change the payload freely as long as the producer and consumer deploy together. For internal job queues this is a feature.

**Weakness (queue).** The moment a second consumer exists, the payload is a contract, and you have no tooling for it.

**Strength (stream).** The retention window forces you to think about compatibility. Schema registries (Confluent Schema Registry, AWS Glue Schema Registry) exist to enforce backward and forward compatibility rules at produce time.

**Weakness (stream).** A schema registry is another service to run, and compatibility rules are easy to set to the permissive option and forget. A common failure is a producer adding a required field, which is backward-incompatible, and a consumer that was replayed from old data failing on the new records.

**Who it is for.** If records are retained and replayed, treat the schema as an API from day one. If records are consumed once and deleted, a schema registry is overhead you can skip until you have a second consumer.

## 7. The cost trap: you pay for retention whether or not you replay

**What it does.** Queue pricing is usually per request plus data transfer. Stream pricing is usually per shard-hour or per partition plus storage for retention plus data transfer.

**Strength (queue).** For low, bursty volume, a queue is cheap because you pay per message and idle costs nothing.

**Weakness (queue).** At sustained high volume, per-request pricing can exceed a provisioned stream. This is arithmetic, not a benchmark: assume 10,000 messages/s, which is 864 million messages/day (10,000 × 86,400). At a per-million-request price, that is a large daily bill; a provisioned stream sized for the same throughput may be cheaper, depending on the provider's tier. The point is to do the multiplication with your own numbers before choosing.

**Strength (stream).** Provisioned capacity is predictable. You pay for shards or partitions, not per message.

**Weakness (stream).** You pay for idle capacity. A stream sized for peak runs at peak cost all month. Retention multiplies storage cost, and a long retention window chosen "just in case" is a recurring line item.

**Who it is for.** Do the arithmetic for your actual volume, including retention, before you pick. If you cannot estimate volume within an order of magnitude, start with a queue and migrate when the shape of the traffic is clear.

## How to prioritise these traps

The traps are not equally weighted. The first three — replay, fan-out, and backpressure — determine whether the system can do the job at all. The last four — delivery semantics, operations, schema, and cost — determine whether it can do the job sustainably.

A workable order:

1. **Write down the consumption pattern.** One consumer or many? Will a new consumer need history? If yes to either, the stream is the stronger candidate.
2. **Write down the ordering requirement.** If order per key matters, a partitioned stream is the natural fit; a FIFO queue can work but constrains throughput.
3. **Write down the retry model.** If the unit of work is independent and retryable, a queue's visibility timeout is simpler than offset management.
4. **Do the cost arithmetic with your own volume.** Include retention for the stream and request volume for the queue.
5. **Check the operational capacity.** A stream you cannot operate is worse than a queue you can.

A useful rule of thumb: if the message is a **command** ("do this"), a queue. If the message is a **fact** ("this happened"), a stream. Commands are consumed once and are done; facts are retained because more than one party may care, now or later.

## Frequently Asked Questions

**Can I use Kafka as a work queue?**

Yes, and many teams do, but you inherit the stream's operational model for a queue-shaped workload. The main mismatch is retry: a queue's visibility timeout gives you per-message retry for free, while a Kafka consumer that fails on one record either blocks the partition or needs a separate retry topic. If the workload is genuinely a job queue, a queue service is usually less machinery.

**What is the difference between a message queue and an event stream in one sentence?**

A queue delivers a message to one consumer and then forgets it; a stream appends a record to a durable log that many consumers can read at their own pace, and retains it for a configured period regardless of who has read it.

**Do I need exactly-once delivery?**

Usually not, and you usually cannot get it end to end from the broker alone. What you need is idempotent consumers: a deduplication key with a unique constraint, written in the same transaction as the business change. That makes at-least-once delivery safe, and it works the same whether the source is a queue or a stream.

**When should I migrate from a queue to a stream?**

When a second consumer needs the same events, when a new consumer needs history to rebuild state, or when ordering per key becomes a requirement. Migrating earlier than that adds operational cost for a property you are not using.

## The next 30 minutes

Open the producer that publishes your busiest message and write down, in one sentence, whether that message is a command or a fact. Then list every consumer of it. If the answer is "one consumer, and the work is done when it is done", you are on a queue and should stay there; if the answer is "more than one, or one that might need history", you have a stream-shaped problem and the queue is the trap. Do this for the single busiest message first — it is the one where the wrong choice costs the most.

## Official documentation

- [Apache Kafka documentation](https://kafka.apache.org/documentation/)
- [Stripe: idempotent requests](https://docs.stripe.com/api/idempotent_requests)
- [Google SRE Book: Monitoring Distributed Systems](https://sre.google/sre-book/monitoring-distributed-systems/)

---

### About this article

**Written by:** [Kubai Kevin](/about/) — software developer based in Nairobi, Kenya.

**How this article was produced:** This site uses an automated LLM pipeline designed and maintained by the author. Before publication, drafts are screened automatically for minimum length, near-duplicate content, invented or unsourced claims and citations, unverifiable product names, and broken internal links. Individual line-by-line human editing is not performed on every post. Figures, benchmarks and scenarios are illustrative unless a source is linked; verify them against official documentation before relying on them in production. See the [AI content policy](/ai-content-policy/).

**Corrections:** Report errors via the [contact page](/contact/). Corrections are applied promptly.

**Last generated:** November 2026
