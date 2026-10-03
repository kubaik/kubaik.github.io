# RAG on legacy systems: vector DB vs. full-text SQL

## The real problem: retrofitting retrieval onto systems that predate JSON

Adding retrieval-augmented generation to an existing enterprise system is usually framed as an AI problem. It is not. It is a data plumbing problem, and the plumbing was installed long before anyone expected a chat completion endpoint to be attached to it.

A typical legacy stack looks like this: a Java or .NET application server, an Oracle or SQL Server database, a message queue for internal integration, and a strict change-control process around the database schema. None of these components were designed to store or query high-dimensional vectors, and none of them speak the HTTP/JSON conventions that most AI services assume. The team's job is to make retrieval work without breaking the SLA that the existing system already meets.

Two architectural patterns dominate this retrofit:

- **Dedicated vector database**: keep the legacy store unchanged, publish document chunks and their embeddings to a separate vector service, and query it over the network.
- **In-database vector search**: add vector columns and an approximate-nearest-neighbour (ANN) index to the existing relational database, so retrieval happens inside the same transaction boundary the application already uses.

The rest of this article compares them on latency, cost, operational burden, and developer experience, and ends with a decision checklist and a 30-minute experiment you can run to get your own numbers.

## Option A: a dedicated vector database

In this pattern the legacy application never stores embeddings. A separate ingestion job chunks documents, calls an embedding model, and writes the vectors to a vector service. At query time the application sends the user's text to an embedding endpoint, sends the resulting vector to the vector service, and receives a list of document IDs. The legacy database is only consulted afterwards, to fetch the full text of the matched documents.

The appeal is isolation. The vector service owns sharding, replication, and ANN index tuning. The legacy application only needs an HTTP client. This matters when the database is under a change freeze or when the application runs on a platform whose toolchain cannot be recompiled on demand.

A minimal retrieval service looks like this:

```python
# requirements.txt
sentence-transformers==2.7.0
fastapi==0.110.2

from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")

@app.post("/retrieve")
def retrieve(query: str, limit: int = 3):
    vector = model.encode(query, convert_to_numpy=True)
    results = vector_store.search(vector.tolist(), limit=limit)
    return [r["document_id"] for r in results]
```

The `vector_store` object is whatever client your chosen service provides. The important property is that it is a network call, with all the latency and failure characteristics of one.

### Where the isolation hurts

Every network hop adds latency. A request that traverses the application, an API gateway, the embedding service, and the vector service accumulates round-trip time at each step. On a well-provisioned cluster inside one availability zone this might be a few milliseconds per hop; across availability zones it is routinely tens of milliseconds. For a chat interface a 100 ms floor is tolerable. For an agent-facing search box with a 150 ms p95 target it is not.

Cost has a similar shape. A managed vector service is typically priced per node per hour, and production availability requires more than one node. Embedding inference is priced per token. Both scale with query volume and document volume rather than with the number of users, which is unfamiliar to teams used to sizing for concurrent sessions.

The isolation also creates a security seam. Legacy systems often authenticate with a message-queue credential or a database user. A vector service authenticates with an API key or an OAuth2 token. Bridging the two usually means writing a gateway service, which is one more component to deploy, monitor, and patch.

### Where it genuinely wins

- The legacy schema never changes, so no DBA approval is required for the retrieval feature itself.
- The ANN index can be tuned or replaced without touching the application database.
- Regional deployment is straightforward: run a vector service in each region and keep the legacy database where it is.
- If the embedding model changes frequently, only the ingestion job and the retrieval service change.

## Option B: vector search inside the existing database

In this pattern the embeddings live in the same database as the rest of the application data. PostgreSQL with the `pgvector` extension is the most common choice because it is open source and widely available on managed platforms. Other relational databases have added native vector types and distance operators, and the SQL below is representative of the general shape rather than specific to one vendor.

```sql
-- Enable the extension (requires privileges)
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE document_chunks (
    id bigserial PRIMARY KEY,
    document_id varchar(64) NOT NULL,
    chunk_text text NOT NULL,
    embedding vector(384) NOT NULL
);

-- Approximate nearest-neighbour index
CREATE INDEX ON document_chunks
    USING hnsw (embedding vector_l2_ops);

CREATE OR REPLACE FUNCTION semantic_search(query text, match_count int DEFAULT 3)
RETURNS TABLE (document_id varchar(64), chunk_text text, score float) AS $$
DECLARE
    query_embedding vector(384);
BEGIN
    query_embedding := embed_text(query);
    RETURN QUERY
    SELECT
        dc.document_id,
        dc.chunk_text,
        1 - (dc.embedding <=> query_embedding) AS score
    FROM document_chunks dc
    ORDER BY dc.embedding <=> query_embedding
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;
```

`embed_text` is a placeholder for whatever mechanism you use to turn a query string into a vector. It might be a call to an external embedding API, a call into a local model served over HTTP, or a database function that shells out to an inference runtime. The database does not embed text by itself.

### Where the consolidation helps

The retrieval query and the follow-up query that fetches document metadata run in the same connection, often in the same transaction. There is no extra network hop, no second authentication scheme, and no second backup policy. Latency is dominated by the ANN index scan and the embedding call rather than by round trips.

Operationally, this is the smallest possible change. Existing connection pools, monitoring, and failover procedures apply unchanged. The main new requirement is memory: an HNSW index is held largely in memory, so the database instance must be sized for the index plus the working set.

### Where it hurts

- Schema changes require the same approval process as any other migration. In regulated environments, adding an extension can be a multi-week review.
- Index builds and rebuilds consume memory and I/O. On a database that is already near its memory limit, a rebuild can push it over.
- Logical replication does not automatically carry vector columns in all configurations; cross-region topologies need explicit handling.
- The database now has a new failure mode that the operations team may not have seen before.

## Comparing latency honestly

Published latency comparisons between vector databases and in-database search are rarely useful, because the result depends on index type, dimensionality, dataset size, hardware, and network topology. Two systems with identical software can differ by an order of magnitude based on whether the query crosses an availability zone.

What you can do is measure your own. The instrumentation that matters is:

- **End-to-end p50/p95/p99** at the client, not at the service. Client-side percentiles capture the network path that users actually experience.
- **Per-hop timing**: time spent in the embedding call, in the vector search, and in the follow-up metadata fetch. Without this breakdown, a slow embedding service is easily mistaken for a slow vector index.
- **Recall@k** against an exact brute-force search on a sample of queries. ANN indexes trade recall for speed, and a latency win that drops recall is not a win.

A simple load test with `wrk` or `k6` against your retrieval endpoint, run for at least ten minutes to reach steady state, will produce numbers you can act on. Compare the two architectures under the same concurrency, the same document set, and the same embedding model. Anything else is a comparison of two different workloads.

The one structural difference that measurement will consistently reveal is the number of network hops. A dedicated vector service adds at least one hop that in-database search does not. Whether that hop costs 2 ms or 40 ms depends entirely on your topology, and it is the single largest source of variance between teams.

## Comparing cost honestly

Cost comparisons are similarly easy to get wrong. The components that recur in both patterns are:

- **Embedding inference**, priced per token. This is usually the dominant line item and is identical in both architectures if the same model is used.
- **Storage and compute for the index**, priced per node-hour or per provisioned capacity unit.
- **Network egress**, which is zero for intra-zone traffic and non-zero for cross-zone or cross-region traffic.
- **Operational labour**, which is real but hard to quantify and should be tracked as engineer-hours rather than dollars.

The structural difference is that the dedicated vector service adds a second compute cluster and, in most designs, a second network path. The in-database pattern adds memory to an existing instance instead.

To estimate your own costs, start with three inputs: average tokens per query, queries per day, and the current price per thousand tokens for your embedding model. Multiply them to get a daily inference cost, then multiply by 30 for a monthly figure. Do the same arithmetic for the vector service's node count times node-hour price times 730. These are arithmetic, not benchmarks; the point is to make the assumptions explicit so they can be challenged.

A worked example with illustrative numbers: assume 200,000 queries per day, 400 tokens per query (prompt plus document text sent to the model), and a price of $0.0004 per thousand tokens. Daily tokens are 200,000 × 400 = 80,000,000. At $0.0004 per thousand tokens that is 80,000 × $0.0004 = $32 per day, or roughly $960 per month. If the same workload requires a three-node vector cluster at $0.50 per node-hour, that adds 3 × $0.50 × 730 = $1,095 per month. The inference cost and the cluster cost are then comparable, which is worth knowing before choosing an architecture. Substitute your own figures; the arithmetic is the same.

## Operational and developer experience

The difference that teams notice after deployment is rarely latency. It is the number of systems that must be understood to debug a problem.

With a dedicated vector service, a retrieval bug can originate in the application, the gateway, the embedding service, the vector service, or the network between them. Each has its own logs, metrics, and deployment cadence. Tracing a slow query requires correlation IDs across all of them. This is a well-understood problem, but it is work that did not exist before.

With in-database search, the retrieval query appears in the same slow-query log as everything else. A developer who already knows the schema can read the query plan. The cost is that the database team now owns a new class of index, and the application team must learn the distance operators and index parameters.

On the developer side, the in-database pattern usually wins on familiarity. Application developers already write SQL. Adding one more query to an existing data-access layer is a smaller conceptual step than introducing a new service client, a new authentication scheme, and a new deployment target. Teams that have been burned by microservice sprawl tend to prefer it for that reason alone.

The trade-off is testability. Tests that exercise vector search need a database with the extension installed and the index built, which slows down local and CI runs. Tests against a vector service need a running service or a mock, which is a different kind of friction.

## Decision checklist

Work through these in order. The first question that produces a clear answer usually decides the architecture.

1. **Can the schema be changed?** If the database is under a change freeze, or if adding an extension requires a review that will not complete in the project's timeframe, the dedicated vector service is the only option.
2. **Does the database support vector types and ANN indexes?** Older releases of most relational databases do not. If not, the choice is between upgrading the database and using a dedicated service.
3. **Is the p95 latency budget tight?** If the target is under roughly 200 ms including the language model call, the extra network hop of a dedicated service is a significant fraction of the budget. Measure it before committing.
4. **How often will the embedding model change?** Frequent changes favour a dedicated service, because re-embedding and rebuilding an index inside a production database is disruptive.
5. **What is the operations team's appetite?** A team with no experience running a vector index inside a relational database will need to learn it. A team with no experience running a stateful service will need to learn that instead. Neither is free.
6. **Is cross-region replication required?** Check whether your database's replication mechanism carries vector columns. If it does not, plan for a custom mechanism or prefer the dedicated service.

## Common failure modes

**Mismatched vector dimensions.** The column type must match the embedding model's output dimension exactly. A model that produces 384-dimensional vectors cannot be stored in a column declared as 1536. Choosing a smaller model where quality permits reduces index size and build time proportionally, but the dimension must be consistent across ingestion and query.

**Index build during peak hours.** Building or rebuilding an ANN index is memory- and I/O-intensive. Schedule it outside peak traffic and monitor the database's memory headroom during the build. On a database already near its memory limit, a rebuild can cause query failures.

**Assuming the database embeds text.** Relational vector extensions store and compare vectors. They do not call embedding models. The embedding step is always a separate component, whether it is an external API, a locally served model, or an inference runtime, and it must be sized and monitored like any other dependency.

**Treating recall as a free parameter.** ANN indexes are approximate. Increasing the search breadth improves recall at the cost of latency. If retrieval quality drops after switching index parameters, measure recall against an exact search before assuming the model is at fault.

## FAQ

**Can vector search be added to a database that does not support it natively?**
Not without changing the database. The usual approach is to keep the legacy database as the system of record and run a separate retrieval service alongside it, with the application calling both. The integration layer between them is the main engineering effort.

**How much memory does an in-database ANN index need?**
It scales with the number of vectors and their dimensionality. The index is largely resident in memory, so the instance must be sized for the index plus the existing working set plus headroom for maintenance operations. Measure the index size after a representative load rather than estimating from documentation.

**Does the embedding model have to run on a GPU?**
Smaller models run acceptably on CPU for low query volumes. Larger models and higher throughput generally require a GPU. This cost is identical in both architectures and should be evaluated separately from the choice of vector store.

**What happens to retrieval quality when the document set grows?**
Recall at a fixed index parameter setting tends to degrade as the dataset grows, because the approximate search has more candidates to miss. Re-evaluate recall after significant growth and adjust index parameters if needed.

## Your next 30 minutes

Open a connection to a copy of your target database and run `CREATE EXTENSION IF NOT EXISTS vector;`. If it succeeds, you have confirmed that in-database vector search is at least technically possible in your environment, and you can proceed to size the instance and measure latency on a representative sample. If it fails, you have confirmed that the dedicated vector service is the path, and you can start scoping the integration layer. Either result is more useful than any benchmark you will read.
