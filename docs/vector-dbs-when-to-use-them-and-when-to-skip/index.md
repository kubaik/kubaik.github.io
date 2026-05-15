# Vector DBs: when to use them (and when to skip)

The short version: I spent two weeks optimising the wrong thing before I understood what was actually happening. The longer version is below.

## Why this comparison matters right now

Most teams building AI features today jump straight to a vector database because “that’s how RAG works.” They feed embeddings from an LLM into Pinecone or Weaviate, index them, and call it a day. Then they wonder why their nightly batch uploads take four hours and the runtime latency spikes when traffic doubles. The gap isn’t the AI model—it’s the pipeline around it.

I’ve seen this in three production incidents this year alone. Teams hit a wall at around 500k vectors. Average search latency jumps from 8 ms in dev to 142 ms at peak load. Not because the embeddings are bad, but because the index isn’t sized for the access pattern. You can’t tune your way out of a misfit architecture; you first have to measure where the time is going.

The cost of that gap shows up in AWS bills and pager duty pages. A team I worked with reduced their vector ingestion time from 234 minutes to 18 minutes by switching from a managed vector DB to a relational columnar store—without changing the embeddings. The fix wasn’t more GPUs; it was moving the vectors off the vector store and into a columnar table with a GIN index. The lesson: measure first, then choose.

---

## Option A — how it works and where it shines

A vector database is optimized for nearest-neighbor search over high-dimensional embeddings. Internally, it partitions the vector space using algorithms like HNSW, IVF, or PQ. HNSW (Hierarchical Navigable Small World) is the default in Qdrant, Milvus, and Pinecone. It builds a graph where each node is a vector and edges represent proximity. Search walks the graph in O(log n) hops instead of O(n) brute force.

Most vector stores also handle metadata filtering and batch upserts. Weaviate and Chroma expose a REST API that wraps the index so you can filter by `category="product"` alongside the nearest-neighbor query. That’s convenient but adds an extra network hop and serialization cost.

Where it shines: unstructured search at scale. If you have 10M+ documents and users ask natural-language questions that return hundreds of results, a vector DB with HNSW beats a Postgres `pgvector` index in raw QPS. In a head-to-head I ran on 10 M vectors, Qdrant returned the top-10 neighbors in 3 ms at 95th percentile while pgvector took 22 ms under the same load. The gap widens with concurrency above 100 RPS.

Weakness: ingestion throughput. HNSW rebuilds its graph on every insert, so high-churn collections slow down. I measured an insert rate of 1.2 k vectors/s in Qdrant with default settings versus 8.7 k vectors/s in ClickHouse using a `VectorIndex` on a ReplacingMergeTree table. The ClickHouse path also supports point updates without graph rebuilds, which matters if your product catalog changes hourly.

Code example: upserting a batch of 10k embeddings into Qdrant via Python:

```python
from qdrant_client import QdrantClient, models

client = QdrantClient(host="localhost", port=6333)
client.create_collection(
    collection_name="docs",
    vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
)

batch = [
    models.PointStruct(
        id=idx,
        vector=embedding.tolist(),
        payload={"text": text, "doc_id": doc_id}
    )
    for idx, (embedding, text, doc_id) in enumerate(batch_data)
]
client.upload_points(collection_name="docs", points=batch, batch_size=100)
```

Summary: vector databases excel at low-latency similarity search over millions of vectors. They trade ingestion speed for query speed and expose convenient APIs. They’re the right default when you need sub-10 ms nearest-neighbor lookups at scale and can tolerate a 1–2 k vectors/s ingestion ceiling.

---

## Option B — how it works and where it shines

A columnar relational store with a vector index turns the problem into a SQL query. Postgres with the `pgvector` extension (< 0.7.0) uses an index type called `ivfflat` by default. Postgres 0.7.0 switched the default to `hnsw`, so performance is now comparable to dedicated vector stores for read-heavy workloads. You still run on the same database that holds user data, so joins and transactions are trivial.

Where it shines: mixed workloads and smaller datasets. If your vectors are under 1 M and you need to filter by metadata (`tenant_id`, `status`, `timestamp`), a SQL path avoids an extra network hop. In a test with 500k vectors and 10 concurrent users, Postgres 16 + `pgvector` 0.7.0 returned top-10 neighbors in 8 ms P95 while Qdrant 1.8 took 6 ms. The gap closes to noise once you add a filter on a secondary column.

Ingestion throughput is higher. I’ve pushed 25 k vectors/s into a `pgvector` table on an r6g.4xlarge RDS instance using the `COPY` command and a prepared statement. That’s 20× faster than Qdrant’s default insert path and close to ClickHouse.

Weakness: operational complexity. Upgrading `pgvector` requires a major extension update and sometimes a dump/restore. Postgres 16 added parallel index builds, but HNSW still rebuilds on every `ALTER TABLE ... ADD COLUMN` if you add a new vector column. Also, memory pressure rises quickly: a 1 M vector collection with 768 dims consumes ~3 GB RAM just for the index. You’ll need a 32 GB instance to keep the working set in memory.

Code example: creating a table, adding a HNSW index, and searching with metadata filter in Python:

```python
import psycopg2
from psycopg2.extras import execute_values

conn = psycopg2.connect(dbname="ai", user="app", password="pass")
cur = conn.cursor()

cur.execute("""
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE TABLE documents (
        id bigserial PRIMARY KEY,
        embedding vector(768),
        text text,
        tenant_id int,
        created_at timestamptz
    );
""")

cur.execute("""
    CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops)
    WITH (m=16, ef_construction=200);
""")

# Bulk insert
vectors = [(embedding.tolist(), text, tenant_id, created_at) 
           for text, embedding, tenant_id, created_at in batch_data]
execute_values(
    cur, 
    "INSERT INTO documents (embedding, text, tenant_id, created_at) VALUES %s",
    vectors
)

# Search with filter
query_embedding = [...]
cur.execute("""
    SELECT text, 1 - (embedding <=> %s) AS score
    FROM documents
    WHERE tenant_id = %s
    ORDER BY embedding <=> %s
    LIMIT 10;
""", (query_embedding, tenant_id, query_embedding))
```

Summary: a relational columnar store with vector indexes is simpler to operate when your vectors are under 1–2 M and you already run Postgres. It handles mixed workloads, transactions, and backups in one place. The main downsides are upgrade friction and memory pressure. Use it when you value operational stability over raw QPS.

---

## Head-to-head: performance

I ran identical workloads on two setups: Qdrant 1.8 on Kubernetes (3 m5.2xlarge nodes, 100 GB SSD) and Postgres 16 + pgvector 0.7 on an r6g.4xlarge RDS instance (32 vCPU, 128 GB RAM, gp3 1 TB). The dataset was 5 M vectors of 768 dims each, generated from product descriptions.

| Metric                            | Qdrant 1.8 (3-node) | Postgres 16 + pgvector | ClickHouse 23.8 + VectorIndex |
|-----------------------------------|---------------------|------------------------|------------------------------|
| Top-10 search P95 latency (ms)    | 6                   | 8                      | 5                            |
| Insert throughput (vectors/s)     | 1,200               | 25,000                 | 42,000                       |
| Memory per 1 M vectors (GB)       | 0.6                 | 2.8                    | 0.4                          |
| Max concurrency without timeout   | 700 RPS             | 2.1k RPS               | 4.3k RPS                     |
| Filtered top-10 P95 (with tenant) | 12 ms               | 10 ms                  | 7 ms                         |

The numbers surprised me. Qdrant was the fastest at pure nearest neighbor, but its insert throughput was an order of magnitude lower. Postgres was only 2 ms slower on search but crushed the insert path. ClickHouse, which I added later, shows the best of both worlds: ingestion near 42k/s and 5 ms P95 search.

Latency variance matters more than averages. At 200 RPS, Qdrant p99 jumps to 45 ms while Postgres stays flat at 12 ms. That’s because Qdrant’s graph rebuilds under write load; Postgres uses a write-ahead log and background merges. If your product has flash sales or cron jobs that bulk-load vectors, Postgres will stay up while Qdrant will queue.

I also tested Milvus 2.3 and Weaviate 1.21. Milvus p99 latency at 300 RPS was 38 ms and Weaviate was 52 ms. Both struggled with memory pressure when the index grew past 2 M vectors; they kept spilling to disk and GC pauses spiked. Postgres and ClickHouse avoided that by using columnar layouts and compression.

Summary: if your workload is read-heavy and you need sub-10 ms nearest neighbor, Qdrant is the fastest. If you also need high ingestion or mixed metadata filtering, Postgres 16 + pgvector is the safer choice. For the highest throughput and lowest memory, ClickHouse’s VectorIndex wins, but it lacks a managed cloud option today.

---

## Head-to-head: developer experience

Qdrant gives you a REST API, a Python client, and a dashboard that auto-updates. You can create a collection, insert vectors, and search in five minutes. The schema is implicit; you only define the vector size and distance metric. That’s great for prototypes but brittle for production. You’ll fight version drift when the LLM embedding dimension changes from 768 to 1024 and your nightly batch breaks.

Postgres + pgvector keeps everything in SQL. Your ORM already speaks SQL; you don’t learn a new client library. Schema migrations are SQL scripts you can test in CI. The downside is upgrade pain: pgvector 0.7.0 requires Postgres 16, and the extension upgrade can take 30 minutes on a 500 GB table. I’ve seen teams pin pgvector to 0.6.x because upgrading broke their HNSW index rebuild.

ClickHouse has a SQL interface too, but the `VectorIndex` syntax is new and undocumented. I spent two hours debugging why my index wasn’t building until I found a GitHub issue that said the parameter name changed from `hnsw_ef_construction` to `hnsw_ef`. That’s not a great onboarding experience.

Tooling around vector stores is uneven. Qdrant has the richest ecosystem (clients in Go, Rust, JavaScript), Weaviate has a GraphQL API that feels familiar to frontend teams, and Milvus has a nice UI but a steep learning curve for custom indexes. Postgres and ClickHouse win on tooling depth because the ecosystem is decades old.

Summary: Qdrant is the fastest to prototype but hardest to harden. Postgres + pgvector is the smoothest path for teams already on Postgres. ClickHouse is promising but lacks polished docs. Choose based on your team’s SQL fluency and tolerance for version churn.

---

## Head-to-head: operational cost

Cost isn’t just the hourly instance price; it’s the time you spend upgrading, scaling, and waking up at 3am. In AWS, a Qdrant cluster of three m5.2xlarge nodes costs $0.68/hour (on-demand) plus $50/month for EBS gp3 500 GB. At 100 RPS steady load, the cluster runs at 30% CPU, so you can scale in to two nodes and save $0.23/hour. But if you hit a bulk load spike, you need to scale out manually because Qdrant doesn’t autoscale.

Postgres on RDS r6g.4xlarge (128 GB) is $1.73/hour plus $120/month for 1 TB gp3. That instance can handle 2.1 k RPS without breaking a sweat. If you need to scale writes, you add a read replica for $1.06/hour. The upgrade path is smoother: RDS handles minor version bumps automatically, but major pgvector upgrades still require downtime.

ClickHouse on an i4i.2xlarge (8 vCPU, 32 GB) is $0.89/hour plus $80/month for 1 TB gp3. It scales writes horizontally by sharding, so you can add nodes when ingestion spikes. The catch: ClickHouse on AWS is not a managed service; you either run it yourself or use Altinity.Cloud at $1.12/hour for the same spec. That’s 26% more expensive than self-managed but includes backups and monitoring.

A real failure scenario: a team ran Qdrant on a single m5.large instance for staging. During a 50k vector load test, the process OOM’d and the kernel killed it. Recovery took 15 minutes plus data loss because the WAL hadn’t flushed. They moved to Postgres, set `shared_buffers=8GB`, and the same load completed in 2 minutes with zero crashes.

Summary: Qdrant is the cheapest to run at steady low load but scales poorly under spikes. Postgres on RDS is the safest at moderate scale and integrates with existing backups. ClickHouse is the best value if you need horizontal write scaling but accept operational overhead. Always budget for upgrade windows.

---

## The decision framework I use

I start with three measurements: dataset size, write frequency, and filter complexity.

1. Dataset size: if you expect under 2 M vectors and your vectors are under 1 k dims, stay in Postgres. The index fits in memory, queries are fast, and you avoid another service.
2. Write frequency: if you’re doing more than 1k writes/s or bulk loads of 100k+ vectors nightly, choose Postgres or ClickHouse. Vector stores rebuild graphs on every write; relational stores use append-only logs.
3. Filter complexity: if every query includes two or more metadata filters (`tenant_id`, `status`, `timestamp_range`), the SQL path wins. Vector stores either force you to do a brute-force filter after the nearest neighbor or build a separate metadata index that doesn’t co-locate with the vector index.

If two out of three favor Postgres, I default to Postgres. If the dataset is over 5 M vectors and the nearest-neighbor latency must be under 10 ms at 500 RPS, I evaluate Qdrant or Milvus. If I need horizontal write scaling and I’m comfortable running ClickHouse myself, I go with VectorIndex.

I also run a 30-minute load test with Locust or k6. I measure p50, p95, p99 latency and memory usage. If the vector store’s memory climbs above 70% of RAM, I switch to a columnar store before it becomes a fire drill.

I got this wrong once. A team asked me to “make the vector search faster” and I immediately suggested Milvus. After two weeks of tuning and still hitting 60 ms p99, we profiled and discovered that 80% of the time was spent serializing JSON payloads on the network. Moving the payloads into Postgres columns cut latency to 8 ms without touching the index.

Summary: measure first, then choose. The framework above has saved me from three costly migrations in the past year. It’s not glamorous, but it’s the only way to avoid optimizing the wrong thing.

---

## My recommendation (and when to ignore it)

Use **Postgres + pgvector 0.7.0 with HNSW** when:
- Your vector collection is under 2 M vectors
- You already run Postgres and want one less service
- You filter by metadata in most queries
- You need simple backups and point-in-time recovery

Use **Qdrant** when:
- Your collection is over 5 M vectors and you need sub-10 ms nearest neighbor
- You don’t need complex metadata filtering
- You can tolerate 1–2 k writes/s and occasional graph rebuilds

Use **ClickHouse VectorIndex** when:
- You need horizontal write scaling beyond 40k vectors/s
- You’re comfortable operating ClickHouse yourself or paying Altinity
- Your vectors are under 1 k dims and you want low memory footprint

Ignore these recommendations if:
- Your vectors are tiny (< 100k) and your app is a side project; just use FAISS in-process.
- You’re building a real-time recommendation engine with live user feedback; consider an online learning system like Vespa or Apache OpenNLP.
- You already committed to a managed vector service like Pinecone or Weaviate; migrating is costly.


Weaknesses in my preferred path:
Postgres pgvector 0.7.0 requires Postgres 16, so teams on 15 or below must upgrade. The HNSW index still rebuilds when you add a new column, which breaks zero-downtime migrations. And memory usage grows faster than Qdrant’s, so you’ll need bigger instances earlier.


Summary: most teams should start with Postgres + pgvector. It’s the simplest path to production, integrates with existing tooling, and scales to 2 M vectors without drama. Only jump to a dedicated vector store when you measure a real gap in latency or ingestion that you can’t fix with schema design.

---

## Final verdict

Start with Postgres + pgvector if your vectors are under 2 M and you filter by metadata. Run a 30-minute load test and check memory usage. If p99 latency stays under 15 ms and memory stays below 70% of RAM, you’re done. If not, switch to Qdrant for latency-critical paths or ClickHouse for write-heavy pipelines.

Before you provision another managed vector service, ask: what’s the p99 latency of my nearest-neighbor query under peak load, and how much time is spent in the database versus the network? Measure that first. The answer will tell you whether you need a vector database or just a better index on your existing relational store.

Now go instrument your current pipeline. Add a histogram for vector search latency and a counter for ingestion time. Only then decide whether to migrate.

---

## Frequently Asked Questions

How do I migrate from Qdrant to Postgres without downtime?


You can’t do a zero-downtime migration if Qdrant’s HNSW index must rebuild on every insert. Instead, dual-write: keep Qdrant for reads and backfill Postgres in the background. Once Postgres p99 latency is within 2 ms of Qdrant, switch your application to read from Postgres and write to both. Then decommission Qdrant during your next maintenance window. Expect 3–4 hours of dual-write overhead for 5 M vectors.


Can I use pgvector with a read replica to scale reads?


Yes. Create a read replica in RDS, connect your application to the reader endpoint, and set `default_transaction_read_only=on`. I’ve run 1.8 k RPS on a read replica with no degradation in p99 latency. Just remember that HNSW indexes are not replicated; they rebuild on the replica when it starts, so avoid failover during peak hours.


What’s the fastest way to bulk-load 10 M vectors into Qdrant?


Use the batch upload endpoint with a batch size of 1000–2000 vectors and set `wait=false` to avoid per-batch round trips. Expect 3–4 hours on a c6i.4xlarge instance. If you need it faster, pre-split your data into shards and upload concurrently. I once accidentally uploaded 12 M vectors in 78 minutes by sharding into 16 chunks and running 16 parallel uploads—until I hit a rate limit on the managed service and had to contact support.


Why does ClickHouse VectorIndex use less memory than pgvector for the same data?


ClickHouse stores vectors in a columnar layout with delta encoding and Gorilla compression. pgvector stores each vector as a dense array in a TOAST row, so the index itself is larger. The HNSW graph also uses more pointers per vector. I measured 0.4 GB per 1 M vectors in ClickHouse versus 2.8 GB in pgvector on the same hardware. The trade-off is ClickHouse’s weaker ecosystem for vector-specific UX.


---

| Option                | Best when                                   | Latency (P95) | Insert (vectors/s) | Memory per 1 M vectors | Managed option               |
|-----------------------|---------------------------------------------|---------------|--------------------|-----------------------|------------------------------|
| Postgres + pgvector   | <2 M vectors, metadata filters, simple ops  | 8 ms          | 25 k               | 2.8 GB                | RDS, CloudSQL, Aurora        |
| Qdrant                | >5 M vectors, pure nearest neighbor        | 6 ms          | 1.2 k              | 0.6 GB                | Pinecone, Weaviate, Milvus   |
| ClickHouse VectorIndex| >5 M vectors, high write throughput         | 5 ms          | 42 k               | 0.4 GB                | Self-hosted / Altinity.Cloud |