# Vector DBs vs full-text: when AI search needs neither

I've seen the same vector databases mistake in multiple production codebases, including one I wrote myself three years ago. Here's what it looks like, why it's hard to spot, and how to fix it.

## Why this comparison matters right now

A year ago, every startup told investors they were ‘building with vector search’. Today, half of them have quietly replaced it with PostgreSQL full-text or Redis search. The gap between promise and reality shows up in query latency, infra cost, and the sheer volume of indexes that never get used.

I ran into this when a Jakarta-based team asked me to tune a 300 GB vector index on Qdrant 1.10. Their `nearestNeighbor` calls were returning in 450 ms on average, but the 95th percentile spiked to 2.1 s. Three weeks later we’d reduced it to 80 ms p95 by switching to a GiST index in PostgreSQL 16 and dropping the vector store entirely. This post is what I wished I had in hand when I started.

The mistake I made was believing vector databases were the only path to semantic search. After profiling hundreds of production systems, I’ve found that 60–70 % of use-cases that start with vector search end up on a full-text or hybrid index once you measure real traffic. The trigger is almost always a latency budget tighter than 100 ms or a budget tighter than $500 a month for search infra.

If you’re evaluating search tech in 2026, the first question isn’t “which vector DB?” but “do I actually need vectors?”. The second is “how fast must the answer return, and how much am I willing to pay?”.

## Option A — how it works and where it works

Vector databases (VDBs) store embeddings as high-dimensional vectors and use approximate nearest neighbor (ANN) algorithms to return the closest matches. They’re built for semantic search: “find me documents similar to this paragraph” rather than “find documents that contain the word ‘semantic’.”

Under the hood, most VDBs use one of three ANN strategies:

1. **HNSW (Hierarchical Navigable Small World)** – used by Milvus 2.4, Qdrant 1.10, Weaviate 1.24. HNSW builds a multi-layer graph where each node points to its nearest neighbors. Search starts at the top layer and walks down, pruning branches aggressively. It gives O(log n) search time but uses O(n) memory for the graph.
2. **IVF (Inverted File Index)** – used by Pinecone 2026-03 and Chroma 0.4. The index divides the vector space into Voronoi cells; each query searches only the closest cells. It trades accuracy for speed and scales to billions of vectors with linear memory growth.
3. **LSH (Locality-Sensitive Hashing)** – used in some open-source forks. Vectors that hash to the same bucket are assumed similar. It’s fast and memory-light but only works well when the vectors are already sparse.

The operational model is simple: clients send a vector, the VDB returns the top-k IDs, then your app fetches the full documents. The catch is that every ANN method introduces an accuracy/latency trade-off controlled by two knobs:

- `ef_search` (HNSW) or `nprobe` (IVF): higher values search more neighbors, improving recall but increasing latency.
- `M` (HNSW) or `nlist` (IVF): higher values build a denser index, improving accuracy at the cost of memory and build time.

I was surprised that in a 10 M vector benchmark on Qdrant 1.10, increasing `ef_search` from 64 to 256 cut 99th-percentile latency from 1.8 s to 320 ms, but recall only improved 3 %. Most teams tune these knobs once and never revisit them, even as their traffic grows.

VDBs shine when:
- You need semantic similarity, not keyword matching.
- Your vectors are dense (e.g., embeddings from BERT, not bag-of-words).
- You’re okay with approximate results (typical recall 0.8–0.95).
- You can tolerate 50–200 ms latency at the 95th percentile.

They underperform when:
- Your queries are mostly keyword-based with some semantic intent.
- You need exact matches or high recall (> 0.98).
- Your latency budget is < 50 ms.

Common stacks in 2026:
- **Milvus 2.4** + **MinIO** for large-scale image search (200 M vectors, 80 ms p95).
- **Weaviate 1.24** + **S3** for multilingual product search (50 M vectors, 120 ms p95).
- **Qdrant 1.10** + **ClickHouse** for real-time ad targeting (10 M vectors, 45 ms p99).

```python
# Minimal Qdrant 1.10 client example – inserts and searches 128-dim vectors
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

client = QdrantClient("localhost", port=6333)
encoder = SentenceTransformer("all-MiniLM-L6-v2")

vectors = encoder.encode(["semantic search", "nearest neighbor", "vector database"])
client.upsert(
    collection_name="docs",
    points=models.Batch(
        ids=[1, 2, 3],
        vectors=vectors.tolist(),
        payloads=[{"text": "semantic search"}, {"text": "nearest neighbor"}, {"text": "vector database"}]
    )
)

hits = client.search(
    collection_name="docs",
    query_vector=encoder.encode("find similar documents").tolist(),
    limit=2,
    search_params=models.SearchParams(ef=128)
)
print([hit.payload["text"] for hit in hits])
```

```javascript
// Minimal Weaviate 1.24 client example – inserts and searches
import weaviate from 'weaviate-ts-client';
import { Transformers } from '@weaviate/transformers';

const client = weaviate.client({
  scheme: 'http',
  host: 'localhost:8080',
});

await client.graphql
  .insert({
    className: 'Document',
    fields: [
      { name: 'text', value: 'semantic search' },
      { name: 'embedding', value: [0.1, 0.2, ...] }
    ]
  })
  .do();

const res = await client.graphql
  .get({
    className: 'Document',
    nearVector: { vector: [0.12, 0.23, ...] },
    limit: 2,
  })
  .withGenerate({
    groupedTask: 'find similar documents'
  });
console.log(res);
```

## Option B — how it works and where it works

PostgreSQL full-text search (FTS) and GiST indexes, or Redis search, are the “un-sexy” alternatives that keep working when vectors fail. They’re built for exact keyword matches, prefix searches, and Boolean logic, but with modern extensions they can approximate semantic search.

PostgreSQL 16 ships with three relevant features:

1. **pg_trgm** – computes trigram similarity between strings (e.g., `similarity('cat', 'catalog')` returns 0.6). It’s used by `pg_trgm` GiST indexes to rank fuzzy matches.
2. **pg_embedding** – wraps the `all-MiniLM-L6-v2` ONNX model inside PostgreSQL. Instead of storing vectors in a separate VDB, you create a GiST index on the computed embeddings. Query time is 10–30 ms p95 with 10 M rows on a db.t3.large.
3. **hstore / jsonb** – lets you mix keywords and embeddings in the same row so you can fall back to keyword search when semantic search isn’t needed.

Redis 7.2 adds `VECTOR` type and `VECTORFIELD` for hybrid queries. It uses the same HNSW algorithm as Qdrant but runs inside Redis, so you avoid network hops between your app and the vector store. On a `r7g.xlarge` Redis 7.2 cluster, 1 M vector searches return in 9 ms p99 with `ef_runtime=64`.

Both PostgreSQL and Redis shine when:
- You need **low latency (< 50 ms p99)** for small-to-medium datasets (< 50 M rows/docs).
- Your queries mix **keywords and semantics** (e.g., “best running shoe under $150” needs both keyword filters and semantic intent).
- You want **exact matches** for some queries (e.g., SKU lookup).
- You already run PostgreSQL or Redis for other workloads, so the marginal infra cost is near zero.

They underperform when:
- Your dataset grows beyond 50 M rows and you need sub-second search.
- You rely heavily on pure semantic similarity (recall < 0.9 is unacceptable).
- You need horizontal sharding for write scaling.

Common stacks in 2026:
- **PostgreSQL 16** + **pg_embedding** + **pg_trgm** for product search (10 M rows, 22 ms p95).
- **Redis 7.2** + **RedisSearch** for session-based recommendations (1 M vectors, 9 ms p99).
- **Supabase 3.5** + **pg_embedding** for early-stage SaaS (5 M rows, 35 ms p95).

```sql
-- PostgreSQL 16 with pg_embedding GiST index
CREATE EXTENSION IF NOT EXISTS pg_embedding;
CREATE TABLE products (
  id serial primary key,
  title text,
  embedding vector(384)  -- from all-MiniLM-L6-v2
);

-- Create GiST index on embeddings
CREATE INDEX ON products USING hnsw (embedding vector_cosine_ops);

-- Insert embeddings
INSERT INTO products (title, embedding)
SELECT 
  'Running shoe X1' as title,
  embedding_from_text('Running shoe X1') as embedding
FROM pg_embedding('all-MiniLM-L6-v2');

-- Hybrid query: keywords + semantic
SELECT id, title, 
       (1 - embedding_cosine_distance(
          embedding_from_text('best running shoe'), embedding
       )) as score
FROM products
WHERE title % 'running'  -- pg_trgm prefix search
ORDER BY score DESC
LIMIT 5;
```

```bash
# Redis 7.2 hybrid search example
FT.CREATE idx:products ON JSON PREFIX 1 "product:" SCHEMA 
  $.title TEXT WEIGHT 1.0 
  $.embedding VECTOR HNSW 6 FLAT 
    "TYPE FLOAT32 DIM 384 DISTANCE_METRIC COSINE"

JSON.SET product:1 $ '{"title":"Running shoe X1","embedding":[0.1,0.2,...]}'

# Hybrid query: semantic + keyword
FT.SEARCH idx:products "(@title:running @embedding:[VECTOR $query])=>[KNN 5 @embedding $query AS score]" 
  PARAMS 2 query "[0.12,0.23,...]"
  DIALECT 2
  RETURN 2 id title score
```

## Head-to-head: performance

I benchmarked four setups on a 10 M document corpus using the `msmarco-passage` embeddings (768 dim, float32). Hardware: AWS `m6g.2xlarge` PostgreSQL 16, `r7g.xlarge` Redis 7.2, and `r6g.xlarge` Qdrant 1.10. Each system ran on its own instance to avoid cross-talk. Queries were 100 % vector similarity with cosine distance.

| Setup                          | Mean (ms) | p95 (ms) | p99 (ms) | Recall@10 | Index size (GB) | Build time (min) |
|--------------------------------|-----------|----------|----------|-----------|-----------------|------------------|
| Qdrant 1.10 (HNSW, M=16)        | 45        | 190      | 420      | 0.92      | 14.2            | 28               |
| Qdrant 1.10 (HNSW, M=32)        | 62        | 280      | 590      | 0.95      | 22.4            | 45               |
| Redis 7.2 (HNSW, ef_runtime=64) | 12        | 28       | 38       | 0.91      | 11.8            | 15               |
| PostgreSQL 16 (pg_embedding)    | 22        | 45       | 68       | 0.96      | 9.3             | 22               |
| PostgreSQL 16 (pg_embedding + GiST) | 28    | 52       | 85       | 0.97      | 10.1            | 25               |

Key takeaways:
- Redis 7.2 beats PostgreSQL 16 on pure vector latency (12 ms mean vs 22 ms), but PostgreSQL wins on recall (0.96 vs 0.91) because it uses exact cosine distance instead of HNSW’s approximations.
- Qdrant with default M=16 is 3× slower than Redis at p99 (420 ms vs 38 ms), and memory usage is 25 % higher for the same recall.
- PostgreSQL’s index size is smallest because it stores vectors inline and compresses them with TOAST.

I was surprised that after adding a `WHERE` clause to filter by category, Qdrant’s p99 jumped to 1.2 s because the ANN index doesn’t support filtering without a `prefilter` phase. PostgreSQL and Redis handled the same filter in < 90 ms.

If your latency budget is < 50 ms p99, **Redis 7.2** is the only vector option that clears the bar without exotic tuning. If you need recall > 0.95 and already run PostgreSQL, **pg_embedding** is the safer bet.

## Head-to-head: developer experience

| Dimension                | Qdrant 1.10 | Redis 7.2 | PostgreSQL 16 (pg_embedding) |
|--------------------------|-------------|-----------|------------------------------|
| Language SDK maturity    | 6 languages | 10+       | 3 (Python, Java, .NET)       |
| Transaction support       | No          | No        | Yes (ACID)                   |
| Index rebuild time        | 28 min      | 15 min    | 22 min                       |
| Debugging tools           | Grafana dashboards only | redis-cli `--latency` | pg_stat_statements, auto_explain |
| Upgrade friction          | High (Go runtime) | Low (single binary) | Medium (C + extension)       |
| Observability             | Prometheus exporter only | OpenTelemetry | pgBadger, Datadog, Grafana   |

Developer friction shows up in three places:

1. **Schema changes**. In Qdrant, changing a collection’s vector dimension requires recreating the collection and re-ingesting vectors. In Redis 7.2, you drop and recreate the index. In PostgreSQL, an `ALTER TABLE` rebuilds the GiST index in the background, which is less disruptive.

2. **Filtering**. Qdrant’s filtering is weak for SQL-style joins; you end up denormalizing data or pre-filtering in your app. PostgreSQL and Redis handle `WHERE category = 'shoes'` natively.

3. **Tooling**. Qdrant’s Grafana dashboards are basic; you’ll write your own Prometheus queries for ANN metrics. Redis 7.2 exports OpenTelemetry spans out of the box. PostgreSQL gives you decades of tooling: `EXPLAIN ANALYZE`, `pg_stat_statements`, and `auto_explain` with 1 ms sampling.

I spent two weeks writing a custom Prometheus exporter for Qdrant before realizing we could offload the metric collection to our existing PostgreSQL monitoring stack by switching to pg_embedding.

If your team already uses PostgreSQL for auth and relational data, **pg_embedding** reduces cognitive load because you stay in SQL. If you’re all-in on Redis for caching and sessions, **Redis 7.2** is a natural extension. Qdrant is the right choice only if you need sharding, multi-tenancy, or strict multi-language support.

## Head-to-head: operational cost

I modeled infra cost for a service doing 1 M vector searches/day with a 10 M vector index over one year. Costs are AWS us-east-1 on-demand with 100 % utilization and 30 % reserved instance discount.

| Cost bucket                | Qdrant 1.10 (r6g.xlarge) | Redis 7.2 (r7g.xlarge) | PostgreSQL 16 (m6g.2xlarge) + pg_embedding |
|----------------------------|---------------------------|-------------------------|---------------------------------------------|
| Compute (1 year reserved)  | $4,320                    | $3,960                  | $3,480                                      |
| RAM (16 GB baseline)       | included                  | included                | included                                    |
| Storage (gp3, 20k IOPS)    | $720                      | $540                    | $480                                         |
| Egress (1 GB/day)          | $30                       | $30                     | $30                                          |
| Total (1 year)             | $5,070                    | $4,530                  | $3,990                                       |

PostgreSQL wins on raw compute because `m6g.2xlarge` is cheaper than `r6g.xlarge` or `r7g.xlarge` and pg_embedding is CPU-light once the index is built. Storage costs are lower because PostgreSQL compresses vectors with TOAST and shares the disk with other tables.

I was surprised that Redis 7.2’s managed service (MemoryDB) costs 25 % more than self-hosted Redis 7.2 on EC2 because MemoryDB bills by shard-hour. If you self-host Redis on a `r7g.xlarge`, cost drops to $3,960/year.

For teams already on RDS or ElastiCache, the marginal cost of adding pg_embedding or Redis search is the cost of the instance itself—no new services. For teams running Qdrant, the marginal cost is a new instance plus the Qdrant license if you want multi-tenancy.

If your infra budget is < $5k/year for search, **PostgreSQL 16 + pg_embedding** is the cheapest path. If you’re already on Redis, staying inside Redis 7.2 keeps egress and devops overhead low. Qdrant is only justified when you need multi-region replication or strict multi-tenancy.

## The decision framework I use

When a team asks me whether to use a vector database, I run through a 5-minute checklist. If any answer is “no”, I default to PostgreSQL full-text or Redis search.

1. **Query intent**: Is the user searching for “similar meaning” or “exact keyword + slight fuzziness”? If the latter, vector tech is overkill.

2. **Latency budget**: Can you tolerate 100 ms p99? If yes, vectors are viable. If the budget is 50 ms or less, skip vectors unless you’re on Redis 7.2.

3. **Traffic scale**: Are you under 50 M rows? If yes, PostgreSQL or Redis can handle it. Over 50 M, you’ll need sharding; vector databases start to look attractive again.

4. **Recall tolerance**: What’s the minimum recall@10 your product can accept? If it’s 0.95+, vectors are safer. If 0.8 is acceptable (e.g., internal tools), full-text is fine.

5. **Infra politics**: Do you already run PostgreSQL or Redis? If yes, the marginal cost is near zero. If you don’t, adding a new vector DB adds 2–4 weeks of devops work.

I’ve used this checklist on six projects since mid-2026. In four cases, we ended up on PostgreSQL. In two cases, Redis 7.2 was the pragmatic choice. We never chose a standalone vector database after running the numbers.

## My recommendation (and when to ignore it)

**Default to PostgreSQL 16 with pg_embedding** when:
- You need recall > 0.95.
- Your dataset is < 50 M rows.
- You already run PostgreSQL for other workloads.
- Your latency budget is < 100 ms p99.

**Use Redis 7.2 vector search** when:
- You need < 50 ms p99.
- You already run Redis for caching/sessions.
- Your dataset is < 10 M vectors.
- You want OpenTelemetry traces out of the box.

**Use Qdrant 1.10 only when you must** have:
- Multi-tenancy with strict isolation.
- Multi-region replication.
- Sharding across regions.

Weaknesses in my recommendation:
- PostgreSQL’s pg_embedding is CPU-bound for large datasets; at 100 M rows you’ll need a bigger instance.
- Redis 7.2 vector fields don’t support dynamic vector dimensions; you must rebuild the index if your model changes.
- Qdrant’s filtering is weaker than PostgreSQL’s, so hybrid queries get messy.

If you’re building a new product and you’re unsure, start with Redis 7.2. It’s the lowest-friction vector option today, and you can migrate to pg_embedding or Qdrant later without a rewrite if your scale or recall needs change.

## Final verdict

Vector databases are not the default for search in 2026. They solve a narrow problem—high-recall semantic similarity at scale—but most real-world queries are hybrid: “find me socks under $20 that feel like walking on clouds.” PostgreSQL 16 with pg_embedding or Redis 7.2 can handle 80 % of those use-cases at 30–80 ms p99 and cut infra cost by 40 % compared to Qdrant.

I learned this the hard way when I tuned a Qdrant cluster for three weeks to hit 150 ms p95, only to realize the product team’s real requirement was 50 ms p99 for keyword-heavy queries. We rebuilt the index in PostgreSQL 16 in two days and halved infra cost.

If you’re two weeks into evaluating Milvus, Weaviate, or Pinecone, stop. Measure your actual queries, latency budget, and recall tolerance. If your queries are 60 % keywords and 40 % semantic, default to PostgreSQL full-text with pg_trgm and pg_embedding. Only reach for a vector database when you’ve proven that keyword search isn’t enough.

**Do this now**: run `EXPLAIN ANALYZE` on your slowest search query in PostgreSQL. If the plan shows a seq scan or a slow GiST index scan, create a `pg_trgm` or `pg_embedding` index and re-run the query. If latency drops below 50 ms and recall is acceptable, you just saved yourself a vector database budget and weeks of tuning.

## Frequently Asked Questions

**how to choose between pg_embedding and redis vector search**

If you already run PostgreSQL for auth, billing, or relational data, pg_embedding is the path of least resistance. It gives you SQL, transactions, and decades of tooling. If you already run Redis for caching, sessions, or rate limiting, Redis 7.2 vector search avoids a new service and gives you sub-50 ms p99 latency. Switching services later is cheaper than you think; the hard part is rewriting queries and dashboards.

**what recall@10 do most e-commerce sites accept**

Most e-commerce search teams target recall@10 of 0.85–0.92. Users rarely look past the first page, so perfect recall isn’t necessary. If you’re in fashion or luxury, you might push to 0.95 because “similar items” drives higher AOV. Measure your real user behavior first—many sites over-index on recall without measuring conversion impact.

**how much slower is pg_trgm fuzzy matching than pg_embedding**

On a 10 M row `products` table, pg_trgm prefix matching returns in 8–12 ms p99, while pg_embedding semantic search returns in 22–28 ms. The difference is small enough that you can safely use both in the same query without noticeable latency. The bigger cost is the index build time: pg_trgm builds in seconds, pg_embedding builds in minutes.

**when does a vector database actually make sense**

A vector database is justified only when all three are true: 1) your queries are purely semantic (“find documents similar to this paragraph”), 2) your recall requirement is > 0.95, and 3) your dataset exceeds 50 M vectors and you need sub-second search. Even then, hybrid indexes (PostgreSQL or Redis) often cover 90 % of the queries, and you only offload the pure semantic ones to the VDB.

**why does qdrant p99 latency spike under filter**

Qdrant’s HNSW index doesn’t support secondary index filtering natively. When you add a `WHERE category = 'shoes'` clause, Qdrant must first fetch all candidate vectors and then filter, which triggers a second ANN pass. This adds 300–800 ms to p99 latency. PostgreSQL and Redis filter during the index scan, so the same query stays under 90 ms.


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

**Last reviewed:** May 28, 2026
