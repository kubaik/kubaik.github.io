# Pagination for large datasets: offset vs cursor

Most pagination tutorials show the happy path: a `LIMIT`/`OFFSET` query, a page number in the URL, a total count in the response. That pattern works until the table grows, the client pages deep, and rows start shifting underneath them. This article covers offset, keyset (also called cursor or seek) pagination, where each one breaks, and how to implement keyset pagination on a real schema — including the security and API-contract details that are easy to get wrong when every row is addressable by a client-supplied token.

## Why offset pagination falls over at scale

A list endpoint typically starts like this:

```sql
SELECT * FROM orders ORDER BY created_at DESC LIMIT 20 OFFSET 40;
```

It is easy to reason about, and the client can jump to page 7 without any server-side state. Then the table grows past a few million rows and the endpoint that responded in milliseconds takes seconds. The database is doing exactly what was asked: counting and discarding `OFFSET` rows before returning the ones requested.

The deeper problem is correctness, not just speed. Offset pagination is defined relative to the *current* ordering of the table. If a row is inserted or deleted between page 1 and page 2, the window shifts: a row can be returned twice, or skipped entirely. For an admin console that is annoying. For a payments ledger, a medical record list, or an audit feed, a skipped row is a data-integrity bug that surfaces as "the export is missing transactions" weeks later.

The rest of this article covers offset, cursor, and keyset pagination, where each breaks, and how to ship keyset pagination on a real schema.

## Prerequisites: what to settle before choosing a scheme

Pagination is a query-design problem disguised as an API-design problem. Three things need to be nailed down first.

1. **A total ordering.** Keyset pagination requires a column (or tuple of columns) that is unique and immutable per row. `created_at` alone is not enough — two rows can share a timestamp. Pair it with the primary key: `(created_at, id)`. A table without a stable unique key needs that fixed first.

2. **An index that matches the sort.** For keyset pagination on `(created_at DESC, id DESC)`, an index on `(created_at, id)` is the natural fit. Postgres, MySQL, and SQL Server can all use a composite index for a tuple comparison, but only if the index column order matches the `ORDER BY` direction. A missing index turns keyset pagination into a full scan with extra steps.

3. **A contract for what the client sees.** Will the client receive an opaque cursor string, or the raw `(created_at, id)` values? Opaque is better: it lets the sort key change later without breaking clients, and it stops clients from constructing cursors for rows they should not see.

There is also a decision that drives most of the edge cases later: is the list *stable* (rows don't change) or *mutable* (rows are inserted, updated, deleted while a client is paging)?

## Offset pagination: when it's fine and when it isn't

Offset pagination is not wrong. It is the correct choice when:

- The dataset is small enough that `OFFSET` stays cheap (a few thousand rows, or a filtered subset that is always small).
- The client needs random access — "jump to page 12" — and that cannot be expressed as a cursor.
- The data is append-only and the client tolerates a shifting window (for example, a "load more" feed where duplicates are acceptable).

The cost is linear in `OFFSET`. A query with `OFFSET 100000 LIMIT 20` makes the database walk 100,020 index entries and discard 100,000. On Postgres, `EXPLAIN ANALYZE` will show an `Index Scan` with a high `rows removed by filter` count, or a `Limit` node sitting above a large scan. The fix is not a bigger instance; it is a different access pattern.

There is a second, subtler cost: **offset pagination leaks row counts**. If the endpoint returns `total_count`, an unauthenticated or low-privilege client can infer how many records exist in a table — useful reconnaissance for an attacker probing a multi-tenant system. Even without `total_count`, timing differences between `OFFSET 0` and `OFFSET 1000000` can confirm the existence of a large dataset.

Here is the offset query most teams start with, and the index that makes it survivable:

```sql
-- Schema
CREATE TABLE orders (
  id          BIGSERIAL PRIMARY KEY,
  tenant_id   BIGINT NOT NULL,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  status      TEXT NOT NULL
);

CREATE INDEX idx_orders_tenant_created
  ON orders (tenant_id, created_at DESC, id DESC);

-- Offset query
SELECT id, tenant_id, created_at, status
FROM orders
WHERE tenant_id = $1
ORDER BY created_at DESC, id DESC
LIMIT 20 OFFSET $2;
```

The index makes the scan ordered, but it does not remove the `OFFSET` walk. At `OFFSET 500000` the query still pays for 500,000 skipped entries.

## Keyset pagination: the mechanics

Keyset pagination (also called seek pagination) replaces `OFFSET` with a `WHERE` clause that starts *after* the last row the client saw. Instead of asking the database to skip N rows, it asks for the first row whose sort key is greater than (or, for descending order, less than) the last one returned.

For a descending sort on `(created_at, id)`, the predicate is a tuple comparison:

```sql
SELECT id, tenant_id, created_at, status
FROM orders
WHERE tenant_id = $1
  AND (created_at, id) < ($2, $3)
ORDER BY created_at DESC, id DESC
LIMIT 20;
```

`($2, $3)` is the `(created_at, id)` of the last row on the previous page. The database seeks directly to that position in the index and reads 20 rows. Cost is roughly constant per page, regardless of how deep the client is — the index seek replaces the linear walk.

Postgres, MySQL 8+, and SQL Server all support row-value comparison like this. Where the database does not, expand it manually:

```sql
WHERE tenant_id = $1
  AND (
    created_at < $2
    OR (created_at = $2 AND id < $3)
  )
```

That expansion is equivalent, but the tuple form is easier to read and lets the planner use the composite index more directly. Check the `EXPLAIN` output: a `Filter` node instead of an `Index Cond` means the index is not being used for the seek.

### The cursor is a contract, so treat it like one

The client should not see `created_at=2026-01-14T09:00:00Z&id=88213`. That leaks internal identifiers, lets clients forge cursors, and couples the public API to the schema. Encode the tuple into an opaque token:

```python
import base64, json, hmac, hashlib
from datetime import datetime, timezone

CURSOR_SECRET = b"...load from secret manager..."

def encode_cursor(created_at: datetime, row_id: int) -> str:
    payload = json.dumps({
        "t": created_at.astimezone(timezone.utc).isoformat(),
        "i": row_id,
    }, separators=(",", ":")).encode()
    sig = hmac.new(CURSOR_SECRET, payload, hashlib.sha256).digest()[:16]
    return base64.urlsafe_b64encode(payload + sig).decode().rstrip("=")

def decode_cursor(token: str) -> tuple[datetime, int]:
    padded = token + "=" * (-len(token) % 4)
    raw = base64.urlsafe_b64decode(padded)
    payload, sig = raw[:-16], raw[-16:]
    expected = hmac.new(CURSOR_SECRET, payload, hashlib.sha256).digest()[:16]
    if not hmac.compare_digest(sig, expected):
        raise ValueError("invalid cursor")
    data = json.loads(payload)
    return datetime.fromisoformat(data["t"]), int(data["i"])
```

Two things matter here. First, the HMAC stops clients from tampering with the cursor to jump to arbitrary rows — a common way IDOR bugs get introduced into paginated APIs. Second, signing the cursor allows detection of stale tokens after a schema change and a clear `400 invalid_cursor` response instead of a confusing empty page.

**Gotcha:** do not put secrets in the cursor itself. A signed cursor is fine; an *encrypted* cursor is usually overkill and adds key-rotation pain. Sign for integrity, not confidentiality, and keep the payload free of anything sensitive.

## Cursor pagination on mutable data: the hard part

Keyset pagination is exact when the sort key is immutable. When rows are inserted or updated between pages, the cursor can still produce surprising results — but the failure modes are different from offset, and usually less severe.

Consider a feed sorted by `created_at DESC`. A new row inserted at the top does not affect a client paging downward: the cursor points at a fixed position, and new rows simply appear on the next refresh. That is the desired behavior for most feeds. With offset, a single insert at the top shifts every subsequent page by one and causes a duplicate.

Updates are trickier. If `created_at` is updated (it usually should not be — treat it as immutable), a row can move behind the cursor and disappear from the client's view. The fix is to sort on a column that never changes, or to snapshot the query at the start of the pagination session.

Deletes are the classic edge case. If the row referenced by the cursor is deleted between page 1 and page 2, the next query `WHERE (created_at, id) < (cursor)` still works — the predicate is a position, not a row lookup. The client just sees the next surviving row. That is one of the reasons keyset is more robust than offset for mutable tables.

### When a stable snapshot is required

Some use cases — exports, compliance reports, "download all my data" — require the client to see a consistent snapshot even as writes continue. Keyset pagination alone does not give that. Two practical options:

1. **Snapshot the result set** into a temporary table or materialized view at the start of the export, then page over it with keyset. This is what most "export to CSV" endpoints should do.
2. **Bound the query by time**: `WHERE created_at <= $snapshot_time`, where `$snapshot_time` is captured when the export begins. Rows inserted after that timestamp are excluded, giving a stable view without a temp table.

The second is cheaper but only works if the sort key is time-based and monotonic. If the sort is by `updated_at` on a table that receives backdated writes, the snapshot approach is needed.

### A worked example: what a shifting window actually does

Take a table with rows carrying `(created_at, id)` values of `(10:00, 1)`, `(10:01, 2)`, `(10:02, 3)`, `(10:03, 4)`, sorted descending. Page size is 2.

Offset, page 1: `ORDER BY created_at DESC LIMIT 2 OFFSET 0` returns `(10:03, 4)` and `(10:02, 3)`. Now a new row `(10:04, 5)` is inserted. Page 2 asks for `LIMIT 2 OFFSET 2` against the *new* ordering, which is `5, 4, 3, 2, 1`. Positions 2 and 3 are `(10:02, 3)` and `(10:01, 2)`. Row `3` has now been returned twice, and the client never sees it flagged.

Keyset, page 1 returns the same two rows, and the cursor is `(10:02, 3)`. After the insert, page 2 asks for `WHERE (created_at, id) < (10:02, 3)`, which returns `(10:01, 2)` and `(10:00, 1)`. No duplicates, no skips. The new row at `(10:04, 5)` is simply not part of this traversal — which is the expected behavior for a descending feed.

## Multi-tenant and permission edge cases

Pagination bugs and authorization bugs love each other. The most common failure mode in a multi-tenant API is a cursor that encodes only the sort position, not the tenant scope. A client can take a valid cursor from tenant A and replay it against tenant B's endpoint — the signature is valid, the position is valid, and if the query does not also filter by `tenant_id`, the response includes another tenant's rows.

The fix is to bind the cursor to the scope it was issued for. Include the tenant ID (or a hash of it) in the signed payload, and verify it on decode:

```python
def encode_cursor(tenant_id: int, created_at: datetime, row_id: int) -> str:
    payload = json.dumps({
        "s": tenant_id,
        "t": created_at.astimezone(timezone.utc).isoformat(),
        "i": row_id,
    }, separators=(",", ":")).encode()
    sig = hmac.new(CURSOR_SECRET, payload, hashlib.sha256).digest()[:16]
    return base64.urlsafe_b64encode(payload + sig).decode().rstrip("=")

def decode_cursor(token: str, expected_tenant: int) -> tuple[datetime, int]:
    padded = token + "=" * (-len(token) % 4)
    raw = base64.urlsafe_b64decode(padded)
    payload, sig = raw[:-16], raw[-16:]
    expected = hmac.new(CURSOR_SECRET, payload, hashlib.sha256).digest()[:16]
    if not hmac.compare_digest(sig, expected):
        raise ValueError("invalid cursor")
    data = json.loads(payload)
    if data["s"] != expected_tenant:
        raise ValueError("cursor scope mismatch")
    return datetime.fromisoformat(data["t"]), int(data["i"])
```

This is the pagination equivalent of an indirect object reference check. The cursor is an object reference; if the caller's right to use it is not validated, that is an IDOR. The same rule applies to filters: if the cursor was issued for `status=paid`, and the client replays it with `status=refunded`, either reject the cursor or re-derive the position — do not silently mix scopes.

A second edge case: **empty result sets**. If the first page returns zero rows, there is no cursor to issue. Clients that assume `next_cursor` is always present will break. Make the contract explicit: `next_cursor` is `null` when there are no more rows, and the client must handle that. Do not return an empty string, which some clients will happily send back as a cursor.

### Choosing between the two

| Property | Offset | Keyset |
| --- | --- | --- |
| Cost at depth | Grows linearly with `OFFSET` | Roughly constant per page |
| Random access ("page 12") | Native | Not supported without extra work |
| Behavior under inserts/deletes | Duplicates or skips possible | Position-stable; no duplicates or skips |
| Total count available | Yes, if computed | Not natively; requires a separate query |
| Client-visible state | Page number and page size | Opaque cursor |
| Index requirement | Sort-matching index helps, does not remove the walk | Sort-matching composite index required |

## Verification: what to check before shipping

Pagination bugs are hard to see in a demo and easy to see in production. Before shipping a new list endpoint, run through this list:

- **`EXPLAIN` the query at depth.** Run the keyset query with a cursor from page 1,000 and confirm the plan shows an index seek, not a scan. On Postgres, look for `Index Cond` rather than `Filter`. To quantify the difference, record `EXPLAIN (ANALYZE, BUFFERS)` output for both the offset and keyset forms at the same depth and compare `Execution Time` and `Buffers: shared hit/read`.
- **Test with a duplicate sort key.** Insert two rows with the same `created_at` and confirm both appear exactly once across pages. If the cursor only encodes `created_at`, this test will fail.
- **Test with a delete between pages.** Delete the row referenced by the cursor and confirm the next page returns the correct next row, not an error or a skip.
- **Test cross-tenant replay.** Take a cursor from tenant A and send it to tenant B's endpoint. It must fail with `400`, not return data.
- **Test tampered cursor.** Flip a byte in the token and confirm the HMAC check rejects it.
- **Check the response contract.** Confirm `next_cursor` is `null` on the last page, and that the client can distinguish "no more rows" from "empty page due to filter".

A useful load-test pattern: page through 10,000 rows in a loop and assert that the set of returned IDs has no duplicates and no missing IDs (relative to a snapshot). This catches off-by-one errors in the tuple comparison that unit tests miss. If the traversal is done while a writer inserts rows concurrently, the assertion should be "no duplicates and no skips of rows that existed at the start of the traversal" — new rows are allowed to be absent.

## Frequently Asked Questions

**Why is offset pagination slow on large tables?**
Because the database must count and discard `OFFSET` rows before returning the requested page. The cost grows linearly with the offset value, so page 10,000 is dramatically slower than page 1 even though both return the same number of rows. An index helps the scan stay ordered, but it does not eliminate the walk.

**Can keyset pagination use a non-unique sort column?**
Not safely on its own. If two rows share the same `created_at`, the cursor cannot distinguish between them, and rows can be skipped or duplicated. Always pair the sort column with a unique tiebreaker such as the primary key, and build the index on the full tuple.

**How is "jump to page N" handled with keyset pagination?**
Generally it cannot be, and that is a deliberate trade-off. Keyset pagination is designed for sequential traversal. If the UI genuinely needs random access, keep offset pagination for shallow pages and switch to keyset for deep traversal, or expose a search/filter interface instead of page numbers.

**Should the cursor be encrypted or just signed?**
Sign it. Encryption adds key-management overhead without much benefit, because the payload (a timestamp and a row ID) is not sensitive on its own. Signing with an HMAC prevents tampering and allows the cursor to be bound to a tenant or filter scope, which is the security property that actually matters.

## Next step: audit one endpoint today

Pick the list endpoint with the highest `OFFSET` values in the logs — the one where clients are paging deep — and run `EXPLAIN ANALYZE` on it with a representative offset. If the plan shows a large scan before the `Limit` node, that endpoint is a candidate for keyset pagination. The 30-minute version: add the composite index on `(tenant_id, created_at, id)`, rewrite the query to use the tuple comparison, and return a signed `next_cursor` instead of a page number. Ship it behind a new API version so existing clients keep working, and watch the p99 latency on that endpoint for a week.
