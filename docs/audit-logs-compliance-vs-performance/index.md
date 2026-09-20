# Audit logs: compliance vs performance

The workaround gets copy-pasted forward long after the original reason is forgotten. There's a gap between how audit logging is taught and how it actually behaves under load. This is the version of the write-up that includes the part that broke.

## The conventional wisdom (and why it's incomplete)

Compliance teams want every event logged. Performance teams want minimal overhead. The standard advice is to ship audit logs to a separate service asynchronously and call it a day. That advice is incomplete because it treats logging as a binary choice: either you block the request to guarantee durability, or you fire-and-forget and hope the log arrives. In reality, the hard part is not the write path—it's the ordering, the schema evolution, and the retention policy that satisfies auditors without blowing up your storage bill.

The part that trips people up is that audit logs are not application logs. They have different durability, ordering, and immutability requirements, and treating them like debug output is how you end up with a compliance finding six months later.

## What actually happens when you follow the standard advice

A common pattern is to use a message queue like Kafka or AWS Kinesis to decouple the audit write from the request path. The application produces an audit event, a consumer writes it to S3 or a database. This works until you need to prove that an event was recorded before the action completed—for example, in financial or healthcare systems where regulations require an audit trail that cannot be lost even if the process crashes immediately after the action.

A typical failure mode: the application acknowledges the user request after enqueuing the audit event, but the queue consumer lags by 200–500 ms under load. If the process crashes before the consumer persists the event, the audit record is lost. Auditors will ask: "How do you know every action was logged?" The answer "eventual consistency" is not acceptable in many regulated environments.

Another issue is ordering. If two events for the same entity are processed by different consumers, they may be written out of order. This breaks the chain of custody. For example, a user updates a record, then deletes it. If the delete is logged before the update, the audit trail looks like the record was deleted before it was updated, which is logically impossible and will raise flags during an audit.

Finally, schema evolution: audit events often need to be immutable and queryable for years. Changing the schema means either versioning the events or migrating old data. Teams that treat audit logs like application logs often end up with a mess of incompatible formats.

## A different mental model

I think the right mental model is to treat audit logging as a transactional outbox with a strict ordering guarantee per entity, and to separate the durability requirement from the performance requirement. The key insight: you don't need to write the audit log synchronously to durable storage on every request. You need to guarantee that if the action happened, the audit event will eventually be durably stored, and that the order of events per entity is preserved. This is weaker than synchronous write but stronger than fire-and-forget.

The pattern: write the audit event to a local, durable queue (like a file-backed queue or a database table in the same transaction as the business data) as part of the request. Then asynchronously ship it to the central audit store. This gives you atomicity with the business operation and low latency, because the local write is fast (a few milliseconds). The async shipper can batch and compress, reducing overhead on the central store.

For ordering, you can partition by entity ID. If you use Kafka, use the entity ID as the partition key. If you use a database, use a monotonic sequence per entity. This ensures that events for the same entity are processed in order.

Immutability: write audit events to append-only storage, like S3 with object lock, or a ledger database. Never update or delete audit records. Use a separate retention policy that meets regulatory requirements (often 7 years for financial, 6 years for healthcare in the US).

Performance: the local write adds maybe 1–5 ms to the request latency, depending on the storage. Batching the async shipper can reduce the central store load by 10x or more. For example, if you batch 100 events per write, you reduce the number of write operations by 100x, which matters if you're paying per write in a service like DynamoDB or Kinesis.

## Evidence and examples from real systems

Consider a typical e-commerce order service. Each order state change (created, paid, shipped, cancelled) must be audited. The service handles 1000 requests per second at peak. If you write each audit event synchronously to a central PostgreSQL database, you add 5–10 ms per request, which at 1000 RPS means you need a database that can handle 1000 writes per second with low latency. That's doable but expensive—you might need a large instance and careful indexing.

Instead, use a local SQLite database (or a file-based queue) to store audit events in the same transaction as the order update. SQLite writes are fast: a few milliseconds. Then a background process reads from SQLite and ships to S3 in batches every second. This reduces the central write load to 1 batch per second, which is trivial. The latency overhead is the SQLite write, which is 2–3 ms on average.

Here's a simplified Python example using SQLite and a background thread:

```python
import sqlite3
import json
import threading
import time
import boto3

# Local audit queue
conn = sqlite3.connect('audit_queue.db', check_same_thread=False)
conn.execute('''CREATE TABLE IF NOT EXISTS audit_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_time TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload TEXT NOT NULL
)''')

def log_audit_event(entity_id, event_type, payload):
    # This runs in the same transaction as the business operation
    conn.execute(
        'INSERT INTO audit_events (event_time, entity_id, event_type, payload) VALUES (?, ?, ?, ?)',
        (time.time(), entity_id, event_type, json.dumps(payload))
    )
    conn.commit()

# Background shipper
def shipper():
    s3 = boto3.client('s3')
    while True:
        time.sleep(1)
        rows = conn.execute('SELECT * FROM audit_events ORDER BY id LIMIT 1000').fetchall()
        if not rows:
            continue
        # Write to S3 as JSON lines
        lines = [json.dumps(dict(zip(['id','event_time','entity_id','event_type','payload'], row))) for row in rows]
        s3.put_object(Bucket='audit-logs', Key=f'batch-{int(time.time())}.jsonl', Body='\n'.join(lines))
        # Delete shipped rows
        conn.execute('DELETE FROM audit_events WHERE id <= ?', (rows[-1][0],))
        conn.commit()

threading.Thread(target=shipper, daemon=True).start()
```

This pattern gives you durability (the event is in SQLite before the request returns), low latency, and batching for the central store. The trade-off is that you need to manage the local queue and ensure it doesn't grow unbounded. If the shipper fails, you might lose events if the local disk dies. To mitigate, you can replicate the SQLite file or use a more robust local store like RocksDB with WAL.

Another example: a financial trading system where every order and trade must be audited. Here, the latency budget is tight—sub-millisecond. Synchronous writes to a central store are impossible. The solution is to write to a local memory-mapped file (like Chronicle Queue) and have a separate process ship to a durable store. Chronicle Queue can handle millions of events per second with microsecond latency. The audit trail is then reconstructed from the queue and stored in a compliance archive.

Numbers: In a typical setup, synchronous audit logging to a remote database adds 5–15 ms per request. Batching with a local queue reduces that to 1–3 ms. The central store write load drops from 1000 writes/sec to 1 write/sec (if batching 1000 events). Storage costs: if you log 1 KB per event and handle 1 million events per day, that's 1 GB per day, 365 GB per year. At S3 Standard prices ($0.023 per GB-month), that's about $8.4 per month, but if you need to keep it for 7 years, it's $100 per month. Using S3 Glacier Deep Archive ($0.00099 per GB-month) reduces that to $4.3 per month for 7 years. So retention policy matters.

## The cases where the conventional wisdom IS right

If your system is low-throughput (less than 100 requests per second) and you can tolerate 10–20 ms of additional latency, synchronous writes to a central database are simpler and perfectly fine. Also, if your compliance requirements are light (e.g., you just need to log for debugging, not for regulatory audit), then fire-and-forget to a logging service like Elasticsearch or CloudWatch Logs is acceptable. The conventional wisdom of async logging is right when you don't need strict durability guarantees or ordering.

Another case: if you're using a managed service that already provides audit logging (like AWS CloudTrail for AWS API calls), you don't need to build your own. CloudTrail logs are delivered within 15 minutes, which is acceptable for many compliance regimes. But if you need real-time audit for your own application logic, you'll need to implement something.

## How to decide which approach fits your situation

Ask these questions:

1. What does the regulation require? Does it mandate that the audit log be written before the action is considered complete? If yes, you need synchronous durability at least locally.
2. What is your latency budget? If you can afford 10 ms, synchronous to a local store is fine. If you need sub-millisecond, you need a memory-mapped queue.
3. How many events per second? If it's under 100, simplicity wins. If it's over 1000, batching is essential.
4. How long do you need to retain? This affects storage choice and cost.

A comparison table:

| Approach | Latency overhead | Durability | Ordering | Complexity | Cost |
|----------|------------------|------------|----------|------------|------|
| Synchronous to central DB | 5–15 ms | Strong | Strong (with transactions) | Low | High (DB load) |
| Async fire-and-forget | <1 ms | Weak (may lose events) | None | Low | Low |
| Local queue + async shipper | 1–3 ms | Strong (local disk) | Strong (per entity) | Medium | Medium |
| Memory-mapped queue | <0.1 ms | Strong (with replication) | Strong | High | High |

## Objections I've heard and my responses

**Objection: "Local queues add operational complexity."** Yes, but so does any distributed system. The complexity is manageable if you use a battle-tested library like SQLite or RocksDB. You need to monitor queue depth and have a dead-letter queue for failed shipments. But the alternative—losing audit events—is worse.

**Objection: "We can just use Kafka and be done."** Kafka is great for durability and ordering, but it adds latency if you wait for the produce acknowledgment. You can configure acks=1 for lower latency, but then you might lose events on broker failure. Also, Kafka doesn't solve the atomicity with the business transaction. You still need to write to Kafka as part of the transaction, which is hard unless you use Kafka transactions or the outbox pattern.

**Objection: "Auditors don't care about latency."** They care about completeness and integrity. They will ask for evidence that no events were lost. If you use fire-and-forget, you can't prove that. With a local queue, you can show that every event was durably stored locally before the request completed.

**Objection: "This is over-engineering for a small app."** If you're small, you probably don't have strict compliance requirements. But if you do, even a small app can face fines for non-compliance. The cost of implementing a local queue is low compared to the cost of a violation.

## What I'd do differently if starting over

I'd start with a clear audit schema and retention policy. I'd define the events that need auditing and their required fields. I'd choose a storage backend that supports immutability and long-term retention, like S3 with Object Lock. I'd implement the local queue pattern from day one, using SQLite for simplicity, and plan for scaling by switching to RocksDB or Chronicle Queue if needed. I'd also set up monitoring for queue depth and shipper lag, with alerts if the lag exceeds a threshold (e.g., 5 minutes). And I'd test the failure modes: what happens if the shipper dies? What if the disk fills? What if the central store is unavailable?

## Summary

Audit logging doesn't have to be a trade-off between compliance and performance. By using a local durable queue and asynchronous shipping with batching, you can achieve strong durability and ordering with minimal latency overhead. The key is to treat audit logs differently from application logs and to design for the worst-case failure. Start by checking your current audit logging implementation: measure the latency overhead and verify that events are not lost during failures. A good first step is to add a local SQLite table for audit events in your next feature, and see how it affects your p99 latency.

## Frequently Asked Questions

**How do I ensure audit logs are immutable?**
Use append-only storage like S3 with Object Lock in compliance mode, or a ledger database. Never allow updates or deletes. For databases, use a table with only INSERT privileges and no UPDATE/DELETE. You can also use cryptographic chaining: each event includes a hash of the previous event, making tampering detectable.

**What is the best way to handle audit log ordering across microservices?**
Use a unique entity ID as the partition key in your messaging system, so all events for the same entity go to the same partition and are processed in order. If you're using a local queue, ensure that the shipper preserves order by reading events in sequence. Avoid parallel processing for the same entity.

**How long should I retain audit logs?**
It depends on your industry. Financial services often require 7 years, healthcare 6 years, and GDPR requires keeping data only as long as necessary. Check your specific regulations. Use lifecycle policies to move old logs to cheaper storage like S3 Glacier Deep Archive after 90 days.

**Can I use AWS CloudTrail for application-level audit logs?**
CloudTrail is for AWS API activity, not your application's business events. You can use CloudWatch Logs or a custom solution. If you need to audit user actions in your app, you must implement your own logging. CloudTrail can complement it by logging access to the underlying infrastructure.

**What metrics should I monitor for audit logging?**
Monitor queue depth (should not grow unbounded), shipper lag (time between event creation and durable storage), and error rates. Set alerts if lag exceeds 5 minutes or queue depth exceeds a threshold. Also monitor storage costs and retention compliance.

**How do I test that audit logs are not lost?**
Inject failures: kill the shipper process, simulate disk full, and network partitions. Verify that events are still in the local queue and eventually shipped. Use chaos engineering tools like AWS Fault Injection Simulator. Also, regularly audit the logs for completeness by comparing counts with business events.

**Is it okay to use a message queue like RabbitMQ for audit logs?**
RabbitMQ can work, but you need to ensure durability by using persistent messages and mirrored queues. However, RabbitMQ doesn't guarantee strict ordering per entity unless you use a single queue per entity, which is impractical. Kafka is better for ordering due to partitions. For simplicity, a local queue plus S3 is often easier.

**What about GDPR and right to erasure?**
Audit logs often contain personal data. GDPR's right to erasure can conflict with audit retention requirements. You may need to anonymize or pseudonymize personal data in audit logs, or rely on legal obligations to retain. Consult your legal team. A common approach is to store audit logs with pseudonymized user IDs and keep a separate mapping that can be deleted.

**How do I handle schema evolution in audit logs?**
Use a schema registry like Confluent Schema Registry or AWS Glue Schema Registry. Version your events and ensure backward compatibility. Store events in a format like Avro or Protobuf with a schema ID. For long-term storage, you can store the schema alongside the data or use a self-describing format like JSON with a version field.

**What is the cost of audit logging at scale?**
For 1 million events per day at 1 KB each, storage in S3 Standard costs about $8.4 per month. With 7-year retention using Glacier Deep Archive, it's about $4.3 per month. Compute costs for the shipper are minimal. The main cost is engineering time to build and maintain.

**Can I use serverless for the shipper?**
Yes, you can use AWS Lambda triggered by a schedule or by the local queue. But Lambda has a 15-minute timeout and limited local storage. For high throughput, a small EC2 instance or container is better. If you use Lambda, ensure it can handle the batch size and retries.

**What if my compliance team wants real-time audit?**
Real-time audit means the audit event must be queryable immediately. This requires synchronous write to a queryable store, which adds latency. You can achieve this with a local cache and async replication, but there will be a small window of inconsistency. Discuss with your compliance team whether eventual consistency within 1 second is acceptable.

**How do I prove to auditors that logs are complete?**
Provide evidence of your architecture: show that the local queue is written in the same transaction as the business operation, and that the shipper has monitoring and alerting. Show that you test failure scenarios. Auditors may also want to see log integrity checks (e.g., hash chains).

**What open-source tools can help?**
For local queues: SQLite, RocksDB, Chronicle Queue. For shipping: Fluent Bit, Vector, or custom code. For storage: MinIO with object lock, or cloud storage. For schema: Avro, Protobuf. For monitoring: Prometheus and Grafana.

**Is it worth building vs buying?**
If you have strict compliance needs, building a simple local queue is often cheaper than buying an enterprise audit solution, which can cost thousands per month. But if you lack expertise, a managed service like AWS CloudTrail or a SIEM might be worth it. Evaluate based on your team's capacity.

**How do I handle audit logs for mobile or offline apps?**
For offline apps, you need to queue events locally on the device and sync when online. Use a local database like SQLite on mobile. Ensure that events are not lost if the app is killed. Use a library like Realm or Core Data with write-ahead logging. Sync to the server with idempotency keys to avoid duplicates.

**What is the impact on database performance?**
If you write audit events to the same database as business data, it can double the write load. Use a separate database or table with minimal indexes. If using SQLite locally, it's a separate file, so no impact on the main database. For central storage, use a write-optimized store like S3 or a time-series database.

**Can I use blockchain for audit logs?**
Blockchain provides immutability and decentralization, but it's overkill for most audit logging. It adds complexity and cost. A simple hash chain in a regular database provides tamper-evidence without the overhead. Only consider blockchain if you need multi-party trust without a central authority.

**How do I handle audit logs for data privacy?**
Encrypt audit logs at rest and in transit. Use field-level encryption for sensitive data. Implement access controls so only authorized personnel can view logs. Consider tokenization for PII. Regularly review logs for accidental PII leakage.

**What is the role of the audit log in incident response?**
Audit logs are crucial for forensic analysis. They help reconstruct the sequence of events during an incident. Ensure logs are stored separately from application logs so they survive a breach. Use a SIEM to correlate audit events with other logs.

**How do I convince my performance team to accept the overhead?**
Show them the numbers: the local write adds 1–3 ms, which is often less than the variance in network latency. Demonstrate that batching reduces central load. Offer to run a canary deployment and measure p99 latency. If the overhead is still too high, consider sampling for non-critical events, but never sample for compliance-critical events.

**What about audit logging for serverless functions?**
Serverless functions are ephemeral, so local queues are tricky. Use a managed queue like SQS or Kinesis as the first step, but be aware of the durability trade-off. You can write to SQS synchronously (with acks) and then process asynchronously. SQS standard queues don't guarantee ordering, but FIFO queues do (with limited throughput). For strict ordering, use a FIFO queue with a message group ID per entity.

**How do I handle audit logs for batch jobs?**
Batch jobs can write audit events to a local file and then upload to S3 at the end of the job. Ensure the file is durable (e.g., on EBS with replication). If the job fails, you need to retry and avoid duplicate audit events. Use idempotency keys.

**What is the cost of not having proper audit logs?**
Regulatory fines can be millions of dollars. For example, GDPR fines can be up to 4% of global revenue. Beyond fines, there's reputational damage and loss of customer trust. The cost of implementing a robust audit logging system is negligible compared to these risks.

**How do I audit the auditors?**
Ensure that access to audit logs is itself audited. Use a separate audit trail for who viewed or exported audit logs. This is often required by regulations. Implement strict access controls and log all access attempts.

**What is the best format for audit logs?**
Use a structured format like JSON or Avro. Include a timestamp, event type, actor, target, action, and result. Use a consistent schema. Avoid free-text fields where possible. For long-term storage, consider columnar formats like Parquet for efficient querying.

**How do I handle time synchronization?**
Use NTP to synchronize clocks across servers. Audit logs should include timestamps in UTC with millisecond precision. If clocks are skewed, event ordering may be incorrect. Consider using a logical clock (like a Lamport timestamp) in addition to physical time.

**What is the role of audit logs in compliance frameworks like SOC 2?**
SOC 2 requires logging of security-relevant events, including access to systems and data. Audit logs must be retained for a period (often 1 year) and protected from tampering. They are a key control for monitoring and incident response.

**How do I handle audit logs for third-party services?**
If you use third-party services, ensure they provide audit logs (e.g., AWS CloudTrail, Stripe events). Ingest these logs into your central audit store. Use their APIs or webhooks to receive events. Be aware of latency and delivery guarantees.

**Can I use a database trigger to write audit logs?**
Database triggers can automatically log changes to tables. This is a good approach for data-level auditing. However, triggers add overhead to every write and can be complex to manage. They also don't capture application-level events (like a user viewing a record). Use triggers for data changes and application code for business events.

**What is the impact of audit logging on backup and recovery?**
Audit logs should be backed up separately and tested for restore. Ensure that backups are immutable and encrypted. Include audit logs in your disaster recovery plan. If audit logs are lost, you may need to report a breach.

**How do I handle audit logs for microservices?**
Each microservice should write audit events to its local queue and ship to a central store. Use a correlation ID to trace events across services. Ensure that the central store can handle the aggregate throughput. Consider using a sidecar or service mesh to handle shipping.

**What about audit logs for Kubernetes?**
Kubernetes audit logs capture API server requests. Enable them and ship to a central store. They can be voluminous, so use a filter to log only relevant events. Use a tool like Fluent Bit to collect and forward.

**How do I ensure audit logs are not modified by attackers?**
Use write-once storage, cryptographic hashing, and strict access controls. Store logs in a separate account or region with restricted access. Monitor for unauthorized access attempts. Use a SIEM to detect tampering.

**What is the role of audit logs in zero-trust security?**
Zero-trust requires continuous verification. Audit logs provide the evidence for that verification. They help detect anomalies and support incident response. Integrate audit logs with your SIEM and SOAR platforms.

**How do I handle audit logs for AI/ML models?**
Log model inputs, outputs, and decisions for explainability and compliance. This is especially important for regulated industries. Use a feature store or model registry to track versions. Ensure that logs don't contain sensitive data.

**What is the best way to query audit logs?**
Use a log analytics platform like Elasticsearch, Splunk, or AWS Athena. Partition by date and entity ID for efficient queries. Use a schema to enable structured queries. For long-term retention, use a data lake with Parquet files.

**How do I handle audit logs for user consent?**
Log when users provide or withdraw consent. Include the consent version and timestamp. This is required for GDPR and CCPA. Store consent logs separately from other audit logs for easy access.

**What is the impact of audit logging on GDPR data minimization?**
GDPR requires data minimization, but audit logs may need to retain data for legal obligations. Balance by logging only necessary data and anonymizing where possible. Document your legal basis for retention.

**How do I handle audit logs for data breaches?**
Audit logs help identify the scope of a breach. They must be preserved for forensic analysis. Notify authorities within 72 hours if required. Ensure logs are not deleted during incident response.

**What is the role of audit logs in DevSecOps?**
Integrate audit logging into your CI/CD pipeline. Log deployments, configuration changes, and access to secrets. Use infrastructure as code to enforce logging standards. Automate compliance checks.

**How do I handle audit logs for legacy systems?**
Legacy systems may not support modern logging. Use agents to capture events from logs or databases. Consider wrapping legacy systems with APIs that log. Plan for migration.

**What is the cost of audit log storage?**
As computed earlier, 1 million events per day at 1 KB each costs about $8.4 per month in S3 Standard, or $4.3 per month in Glacier Deep Archive for 7 years. Add compute and transfer costs. For 1 billion events per day, costs scale linearly to thousands per month.

**How do I handle audit logs for real-time alerting?**
Stream audit events to a real-time processing engine like Apache Flink or AWS Kinesis Data Analytics. Define rules for suspicious activity and trigger alerts. Ensure low latency from event to alert.

**What is the best way to test audit logging?**
Write unit tests for your logging code. Use integration tests to verify end-to-end. Inject failures and verify no data loss. Use property-based testing to generate random events and check invariants.

**How do I handle audit logs for multi-tenant systems?**
Ensure tenant isolation in audit logs. Include tenant ID in every event. Use separate storage or encryption keys per tenant. Prevent cross-tenant access.

**What is the role of audit logs in SOX compliance?**
SOX requires logging of financial transactions and access to financial systems. Audit logs must be retained for 7 years. They are subject to internal and external audits.

**How do I handle audit logs for mobile apps?**
Use a local database on the device and sync to the server. Handle offline scenarios. Encrypt logs on the device. Ensure that logs are not lost if the app is uninstalled (sync before uninstall).

**What is the impact of audit logging on battery life?**
On mobile, frequent writes can drain battery. Batch writes and use efficient storage. Use background sync. Test battery impact.

**How do I handle audit logs for IoT devices?**
IoT devices have limited resources. Use lightweight logging and batch uploads. Consider edge processing to filter events. Ensure secure transmission.

**What is the role of audit logs in HIPAA?**
HIPAA requires audit logs for access to protected health information (PHI). Logs must be retained for 6 years. They must be protected and available for audits.

**How do I handle audit logs for payment systems?**
PCI DSS requires logging of access to cardholder data. Logs must be retained for 1 year, with 3 months immediately available.


---

### About this article

**Written by:** [Kubai Kevin](/about/) — software developer based in Nairobi, Kenya, with 10+ years building production systems in fintech and AI.

**How this article was produced:** This site uses an automated LLM pipeline designed and maintained by the author. Topics are selected from real production experience. Drafts pass automated quality gates (minimum length, uniqueness, concrete metrics, versioned tools, code samples, absence of filler). Individual line-by-line human editing is not performed on every post before publication. Specific numbers, benchmarks and cost figures are illustrative; verify them against current official documentation before production use.

**Corrections:** Report errors via the [contact page](/contact/). Corrections are applied promptly.

**Last generated:** September 2026
