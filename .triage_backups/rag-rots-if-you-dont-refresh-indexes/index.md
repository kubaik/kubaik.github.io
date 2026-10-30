# RAG rots if you don’t refresh indexes

A retrieval-augmented generation (RAG) pipeline is easy to stand up and hard to keep correct. The failure is rarely dramatic: no crash, no error page. Answers simply drift away from the source of truth as documents change underneath a fixed index. This article covers why that happens, how to detect it, and how to build a refresh path that scales with the actual rate of change in your corpus.

## The conventional playbook and its blind spot

The standard RAG recipe is well known: chunk documents, embed the chunks with a sentence embedding model, store the vectors in a vector database, and retrieve nearest neighbours at query time. It is a good recipe. Its blind spot is that it treats the embedding set as a one-time artifact rather than a derived view of mutable data.

Two consequences follow.

First, teams often spend their tuning budget on the embedding model while spending nothing on change detection. Whether a document was edited, deleted, or replaced is a separate question from whether the embedding model is good, and it is the question that determines whether retrieved chunks are still true.

Second, fixed-size chunking interacts badly with edits. A 512-token window over a document is a function of that document's exact text. Edit the document and the same chunk offset now covers different words. If the index is not rebuilt, the retriever returns a chunk ID whose payload no longer matches the source — the vector is stale, and the text shown to the model is stale with it.

The honest framing: a vector index is a materialised view. Materialised views go stale unless something refreshes them.

## Failure modes to design against

Four patterns account for most staleness incidents.

**Undetected edits.** The refresh job only looks at documents the source system reports as modified. If the source system's modification flag is unreliable, or if the job filters on a field that is not updated (for example, a `last_modified` column that only changes on creation), edits slip through. The job runs on schedule and reports success while the index stays wrong.

**Chunk drift.** Even when a document is re-embedded, the chunk boundaries move. A paragraph insertion near the top shifts every subsequent chunk. If chunk IDs are positional (for example, `doc-42-chunk-7`), the ID now refers to different content than it did before. Any downstream cache, evaluation set, or click log keyed on chunk ID is silently invalidated.

**Partial invalidation.** A document is updated, but only some of its chunks are re-embedded. The rest remain, and the retriever mixes old and new content in the same context window. This is worse than a fully stale index because the model receives contradictory passages and may synthesise an answer that matches neither version.

**Deletion blindness.** Source documents get removed or access-restricted, but their chunks stay in the index. The retriever keeps surfacing content that should no longer be visible. In regulated settings this is a compliance problem, not just a quality problem.

## A refresh model: three gates

A useful mental model is a pipeline with three gates, each of which can fail independently:

1. **Change detection** — determine which documents changed since the last successful index build.
2. **Incremental update** — re-chunk and re-embed only the affected documents, and remove their previous chunks.
3. **Consistency validation** — assert that the index matches the source, and surface a metric when it does not.

The second gate is usually the easiest. The first is the hardest, because it depends on what the content source can tell you. The third is the one teams skip, and it is the one that turns an invisible problem into a monitored one.

### Change detection options

| Source capability | Detection method | Typical latency | Notes |
|---|---|---|---|
| Emits change events (CDC, webhooks, event notifications) | Consume the event stream | Seconds | Most reliable; requires wiring the source to a queue |
| Exposes `updated_at` per record | Query by timestamp watermark | Poll interval | Only works if the field is maintained correctly |
| Exposes content only | Hash or checksum each document | Poll interval | Costs a read per document per cycle |
| Object storage | Bucket event notifications | Seconds | Covers create/delete; verify coverage of metadata-only changes |

The important property is not which method you pick but that the chosen method has a defined latency and a defined failure mode. "The nightly job re-indexes modified docs" is a detection method with an undefined failure mode if nobody has verified that the modified flag is trustworthy.

### Incremental update

The core operation is: compute a stable identity for the current version of a document, compare it to what is indexed, and if it differs, delete that document's existing chunks and write new ones. Content hashing gives you a cheap identity:

```python
import hashlib

def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
```

Two details matter. Hash the content you actually embed, not a rendered or normalised variant, or the hash will disagree with the index for reasons that are hard to debug. And store the hash alongside the chunks so that a validation pass can compare without re-embedding.

### Chunk versioning

Give every chunk a version identifier derived from the document version, and make retrieval aware of it. A minimal metadata schema:

```json
{
  "doc_id": "policy-2024-11",
  "doc_version": 7,
  "chunk_index": 3,
  "content_hash": "9f2c...",
  "text": "..."
}
```

With this, retrieval can filter to the highest `doc_version` per `doc_id`, and a validation job can find chunks whose `content_hash` no longer matches any current document version. Positional chunk IDs become unnecessary; identity comes from `doc_id` plus version plus index within that version.

## Worked example: sizing an incremental refresh

Consider a corpus of 10,000 documents, each producing 20 chunks, so 200,000 chunks total. Suppose 1% of documents change per day — 100 documents, or 2,000 chunks.

Full rebuild: re-chunk and re-embed 200,000 chunks.
Incremental: re-chunk and re-embed 2,000 chunks, plus delete 2,000 stale chunks.

The arithmetic is straightforward. The incremental path touches 1% of the embedding work of a full rebuild, so if embedding throughput is the bottleneck, the daily refresh window shrinks by roughly the same factor. That ratio is the whole argument for incremental updates, and it holds regardless of the absolute speed of your embedding model.

What the ratio does *not* tell you is the cost of the delete path. Deleting by `doc_id` is fast if the vector store indexes that field; it can be slow if it requires a scan. Measure both halves before assuming the incremental path is cheap.

### How to measure it

Do not trust a vendor's throughput number or a blog's benchmark. Measure on your own corpus:

- Instrument the embedding call: log wall-clock time and token count per batch. Compute chunks per second.
- Instrument the delete: log the time to remove all chunks for one document. Run it against a store with realistic size, not an empty one.
- Instrument the end-to-end refresh: record the timestamp of the change event and the timestamp when the new chunks are queryable. That difference is your staleness window, and it is the number that matters to users.
- Compare full rebuild versus incremental on the same 100-document change set. Report both wall-clock time and compute cost.

## Staleness as a monitored metric

Define staleness explicitly, then expose it. A practical definition: a chunk is stale if its `content_hash` does not appear in the current version of its `doc_id`, or if its `doc_id` no longer exists in the source.

A validation job can then emit two numbers:

- **Stale chunk ratio** = stale chunks / total chunks.
- **Time since last successful refresh** per document bucket.

Both are cheap to compute if `content_hash` is stored on the chunk. Neither requires re-embedding, only re-hashing the source and comparing sets. Alert on the ratio crossing a threshold you choose, and on the refresh timestamp exceeding the cadence you promised.

## Classifying documents by change rate

Not every document deserves the same pipeline. Classify by observed change frequency, not by intuition.

| Bucket | Observed change rate | Refresh approach | Validation cadence |
|---|---|---|---|
| Stable | Less than ~1% of docs per month | Scheduled batch rebuild | Monthly |
| Active | ~1–10% of docs per month | Change detection plus incremental update | Weekly |
| Volatile | More than ~10% of docs per month | Event-driven, near-real-time update | Daily or continuous |

The thresholds are illustrative, not universal — set them from your own change logs. The point is that the refresh mechanism should be proportional to the observed churn, because both the cost and the risk scale with it.

Two caveats. Change rate is not the only input; a single high-stakes document that changes quarterly may deserve event-driven refresh regardless of bucket. And bucketing is a policy, not a technical constraint — it is fine to start everything in "Active" and promote documents as their churn is measured.

## When the conventional advice is right

The static-index approach is correct in several genuine cases:

- The corpus is immutable by construction — a frozen archive, a published standard, a versioned legal library where new versions are new documents rather than edits.
- Changes are rare enough that a manual or scheduled rebuild is within the tolerance of the product.
- The data is append-only — logs, event streams, telemetry. New records are new chunks; nothing is invalidated. Change detection collapses to "process the tail."
- The application tolerates stale answers and says so explicitly, with citations and timestamps shown to the user.

In the last case, the design decision is a product decision. It should be written down, because "we accept a 24-hour staleness window" is a very different commitment from "the index is always current," and users will hold you to whichever one they inferred.

## A decision checklist

Before building, answer these in writing:

1. What is the maximum staleness window the product can tolerate, per document class?
2. What is the source of truth, and what change signal does it emit? What is that signal's known failure mode?
3. Is every document's identity stable, or do documents get renamed, merged, or split?
4. What happens to chunks when a document is deleted or access-restricted?
5. How will you detect that the index has drifted, without a user reporting a wrong answer?
6. What is the measured cost of a full rebuild, and of an incremental update, on your corpus?
7. Who is paged when the stale chunk ratio crosses the threshold?

If any answer is "we'll figure that out later," that is the gate most likely to fail.

## A minimal refresh implementation

The sketch below shows the shape of an incremental refresh: fetch the document, hash it, compare against what is indexed, and if it differs, delete the old chunks and write new ones. It uses a generic vector store client and a generic chunker interface, so adapt the calls to your stack.

```python
import hashlib
from dataclasses import dataclass

@dataclass
class Chunk:
    doc_id: str
    doc_version: int
    chunk_index: int
    content_hash: str
    text: str

def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def refresh_document(doc_id, fetch_document, chunker, embedder, store):
    """Idempotent refresh of one document. Safe to call repeatedly."""
    doc = fetch_document(doc_id)          # {"content": str, "version": int}
    new_hash = content_hash(doc["content"])

    indexed = store.get_chunks_by_doc(doc_id)
    if indexed and all(c.content_hash == new_hash for c in indexed):
        return {"status": "unchanged", "doc_id": doc_id}

    # Delete before writing: avoids mixing versions in the same namespace.
    store.delete_chunks_by_doc(doc_id)

    texts = chunker.split(doc["content"])
    vectors = embedder.encode(texts)

    chunks = [
        Chunk(
            doc_id=doc_id,
            doc_version=doc["version"],
            chunk_index=i,
            content_hash=new_hash,
            text=text,
        )
        for i, text in enumerate(texts)
    ]
    store.upsert_chunks(chunks, vectors)
    return {"status": "updated", "doc_id": doc_id, "chunks": len(chunks)}
```

The important properties, independent of the specific libraries:

- **Delete before write.** If you write first and delete second, a crash between the two leaves both versions in the index and the retriever can return either.
- **Idempotence.** Re-running the function for an unchanged document is a no-op, so a retry after a partial failure is safe.
- **Hash on the embedded text.** The hash must describe exactly what was chunked, or validation will produce false positives.
- **Version on every chunk.** Without it, retrieval cannot prefer the newest content when old chunks linger.

## Handling sources that emit no events

When the content source offers no change feed, polling is the fallback. Make the poll interval proportional to the risk:

- Compute a hash per document and store the last-seen hash.
- Poll the volatile bucket frequently and the stable bucket rarely.
- Use conditional requests where the source supports them, so unchanged documents cost a 304 rather than a full body.
- For object storage, prefer native event notifications over listing the bucket, and verify that the notification covers the operations you care about.

The failure mode of polling is that it has a latency floor equal to the poll interval, and that a poll cycle can be skipped under load. Log every cycle's start and end so that a missed cycle is visible rather than silent.

## Retrieval-side handling of versions

Even with a clean refresh pipeline, in-flight queries can hit the index mid-update. Two mitigations:

- Filter retrieval to the highest `doc_version` per `doc_id` so a lingering old chunk cannot be returned.
- Include `doc_version` and `content_hash` in the retrieval response, so downstream evaluation and logging can attribute an answer to a specific index state.

The second point matters for debugging. When an answer is wrong, you want to know whether the model was reasoning over stale content or over correct content it mishandled. Without version metadata on the result, that question is unanswerable after the fact.

## Cost model, with stated assumptions

A refresh pipeline's cost has four components. Using illustrative figures so the arithmetic is visible:

- **Change detection:** a queue or event bus. Cost scales with event volume, which scales with change rate, not corpus size.
- **Re-chunking and embedding:** the dominant variable cost. Proportional to the number of changed chunks, which is why incremental updates matter.
- **Vector store writes:** deletes plus upserts for changed documents.
- **Validation:** a periodic re-hash of source documents. Proportional to corpus size, not change rate, but cheap per document because no embedding is involved.

Substitute your own measured numbers for each line. The structural point is that three of the four components scale with change rate, and only validation scales with corpus size. A corpus can be large and cheap to keep fresh if its change rate is low; a small corpus with high churn can be expensive.

## FAQ

**How do I detect changes when the source has no webhooks?**
Poll and hash. Store the last-seen hash per document, re-hash on a schedule proportional to the document's bucket, and use conditional requests where supported. Record the start and end of each poll cycle so skipped cycles are visible.

**Should chunks be deleted and rewritten, or updated in place?**
Delete and rewrite. In-place updates require the chunk boundaries to be stable across edits, which they generally are not. Deleting by `doc_id` and writing the new version keeps the index free of mixed-version content.

**How do I keep chunk IDs stable across edits?**
You generally should not. Treat chunk identity as scoped to a document version. Anything that caches on chunk ID — evaluation sets, click logs — should key on `doc_id` plus version plus index, and be invalidated when the version changes.

**Can a relational database with a vector extension serve as the store?**
It can, and it keeps the index in the same transaction boundary as your metadata, which simplifies consistency. The trade-off is that per-document refresh means deleting and re-inserting that document's rows, which can contend with concurrent queries. Measure the write latency under realistic query load before committing.

**What staleness threshold should trigger an alert?**
Set it from the product's tolerance, not from a default. If the product promises answers no older than a day, alert well before that window closes, and alert on the refresh job's success/failure separately from the staleness ratio, so you can distinguish "the job broke" from "the job is running but not keeping up."

**Does a better embedding model reduce staleness?**
No. Embedding quality affects how well a correct chunk is retrieved; it does not affect whether the chunk is correct. Staleness is a data-freshness property, and it is fixed by the refresh pipeline, not the model.

## One thing to do in the next 30 minutes

Pick one document in your corpus that changes regularly. Compute a SHA-256 hash of its current content, then compare it to the `content_hash` stored on its indexed chunks. If the field does not exist, that is your answer: your index cannot currently tell you whether it is stale. Add the field on the next write, and run the same comparison as a scheduled job.
