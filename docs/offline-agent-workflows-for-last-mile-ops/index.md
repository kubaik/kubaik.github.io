# Offline agent workflows for last-mile ops

An offline-capable agent client is easy to demo and hard to keep honest at scale. Most write-ups stop exactly where the interesting part starts — at the point where the queue has to survive a process kill, a battery pull, and an OS upgrade without losing a single record.

## The problem this solves

Last-mile delivery in Lagos, Nairobi, or Accra rarely happens on a stable 4G connection. A field agent scans a parcel at a depot gate, walks 200 metres into a market, and the app shows a spinner. The parcel moves, the scan doesn't, and by end of day the reconciliation report has a hole in it that finance has to chase. This is not a coverage problem you can fix with a bigger antenna — it's a data integrity problem, and it's solvable in the client.

The standard failure mode is a workflow that assumes the network is a reliable dependency rather than an intermittent one. Teams typically build the happy path first (scan → POST → 200 OK) and bolt on retry logic later, which produces one of two bad outcomes: either the agent sees a blocking error and stops working, or the app queues writes in memory and loses them when Android kills the background process at low battery. A common trap is using `navigator.onLine` as the connectivity signal — it returns `true` on a captive portal or a 2G connection that can't complete a TLS handshake, so the retry loop fires into a dead socket and the queue never drains.

The part that trips people up is that offline-first is not "store and forward." It's a conflict-resolution problem with an audit requirement attached. Every mutation needs a stable client-generated identity, a monotonic ordering, and enough metadata to prove later that the agent was where they said they were. In a European context this maps onto GDPR Article 5(1)(f) (integrity and confidentiality) and, for logistics, Article 30 records of processing. In an African deployment you're usually also reconciling with local data protection acts — Nigeria's NDPA 2023, Kenya's DPA 2019 — which carry similar lawful-basis and breach-notification obligations. The architecture below handles both without a round trip to the server on every write.

## Prerequisites and what you'll build

You'll build a small offline-capable agent client in TypeScript on Node 20 LTS, using IndexedDB (via `idb` 8.0) for durable local storage, a background sync worker, and a reconciliation endpoint. The server side is a compact HTTP handler that accepts idempotent batch uploads. Total surface area is under 600 lines of application code.

What you need installed:

- Node 20 LTS (the `structuredClone` and `Array.prototype.toSorted` built-ins matter here)
- `idb` 8.0 — thin promise wrapper over IndexedDB, avoids the callback pyramid
- A Node HTTP framework with schema-based body parsing; the examples below use plain handler signatures so they port to Express, Hono, or a managed runtime
- `zod` 3.23 for payload validation on both sides
- A device with Chrome for Android 120+ or a WebView-based shell (Capacitor 6 works)

What you're building: an agent scans a parcel, the scan is written to IndexedDB with a client-generated UUIDv7 and a device timestamp, a service worker flushes the queue when connectivity returns, and the server deduplicates on the UUID. The server never trusts the client clock — it stores both `device_ts` and `server_ts` and flags drift over 5 minutes for review.

The design constraint that drives everything: a field agent may be offline for 4–6 hours across a shift, accumulate 300–800 scans, and the app must not lose a single one across a process kill, a battery pull, or an OS upgrade. That rules out in-memory queues, `localStorage` (5 MB cap, synchronous, no transactions), and anything that writes to a single blob.

## Step 1 — set up the environment

Why IndexedDB and not SQLite-in-WASM: for a scan volume under roughly 5,000 rows per shift, IndexedDB's transactional guarantees and native async are enough, and you avoid shipping a WASM binary over a metered connection. If you're already running Capacitor, a native SQLite plugin is often the better choice because you get real SQL and the file survives WebView cache clears. For a pure PWA, stay with IndexedDB.

Create the project and pin versions:

```bash
npm init -y
npm install idb@8.0 zod@3.23
npm install -D typescript@5.4 vitest@1.6 @types/node@20
```

Define the schema first. Every record carries three things that make reconciliation possible: a client UUID, a device monotonic counter, and a captured location fix with its own accuracy radius. The counter is what lets you order scans within a shift even if the device clock jumps — Android devices without network time can drift tens of seconds per day.

```typescript
// db.ts
import { openDB, type DBSchema } from 'idb';

interface ScanRecord {
  id: string;            // UUIDv7, client-generated
  seq: number;           // monotonic per device
  parcelId: string;
  agentId: string;
  status: 'picked_up' | 'in_transit' | 'delivered' | 'failed';
  deviceTs: number;      // epoch ms, untrusted
  lat: number;
  lng: number;
  accuracyM: number;
  synced: 0 | 1;
}

interface AgentDB extends DBSchema {
  scans: { key: string; value: ScanRecord; indexes: { 'by-synced': number } };
  meta: { key: string; value: { lastSeq: number } };
}

export const db = await openDB<AgentDB>('agent', 1, {
  upgrade(d) {
    const store = d.createObjectStore('scans', { keyPath: 'id' });
    store.createIndex('by-synced', 'synced');
    d.createObjectStore('meta');
  },
});
```

The `by-synced` index is what makes the flush cheap. Without it, draining 800 records means reading all 800 and filtering in JS; with it, the store returns only the pending set. The exact gap depends on the device, but the shape is consistent: a full-store scan scales linearly with total rows, while an index lookup scales with pending rows. That matters when the flush runs on a 2G connection and you're racing the OS background-execution timeout.

## Step 2 — core implementation

The write path must be synchronous from the agent's point of view. The agent taps "delivered," the UI updates, and the durable write happens behind it. If the write fails, the UI rolls back — but it almost never does, because IndexedDB commits are local.

```typescript
// scan.ts
import { db } from './db';
import { v7 as uuidv7 } from 'uuid';

export async function recordScan(input: Omit<ScanRecord, 'id' | 'seq' | 'synced' | 'deviceTs'>) {
  const tx = db.transaction(['scans', 'meta'], 'readwrite');
  const meta = await tx.objectStore('meta').get('seq');
  const seq = (meta?.lastSeq ?? 0) + 1;

  const rec: ScanRecord = {
    ...input,
    id: uuidv7(),
    seq,
    deviceTs: Date.now(),
    synced: 0,
  };

  await tx.objectStore('scans').put(rec);
  await tx.objectStore('meta').put({ lastSeq: seq }, 'seq');
  await tx.done;
  return rec;
}
```

UUIDv7 matters because it's lexicographically sortable by timestamp. When two agents' batches arrive at the server out of order, you can sort by `id` and get a stable, clock-independent ordering without trusting `deviceTs`. UUIDv4 would force you to fall back on the device clock, which is exactly what you're trying to avoid.

The flush runs in a service worker with the Background Sync API. Note the retry backoff — a naive `setInterval(flush, 5000)` will hammer a dead radio and drain battery. On a small-battery device with a bad signal, a 5-second retry loop keeps the radio awake for the entire shift, which is a measurable and avoidable battery cost.

```typescript
// sw.ts
const BACKOFF = [1_000, 5_000, 15_000, 60_000, 300_000];

self.addEventListener('sync', (event: SyncEvent) => {
  if (event.tag === 'flush-scans') event.waitUntil(flush());
});

async function flush(attempt = 0) {
  const pending = await db.getAllFromIndex('scans', 'by-synced', 0);
  if (!pending.length) return;

  const res = await fetch('/api/scans/batch', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ scans: pending.slice(0, 200) }),
  }).catch(() => null);

  if (!res?.ok) {
    const delay = BACKOFF[Math.min(attempt, BACKOFF.length - 1)];
    setTimeout(() => flush(attempt + 1), delay);
    return;
  }

  const { accepted } = await res.json();
  const tx = db.transaction('scans', 'readwrite');
  for (const id of accepted) {
    const rec = pending.find(p => p.id === id);
    if (rec) await tx.store.put({ ...rec, synced: 1 });
  }
  await tx.done;

  if (pending.length === 200) flush(0);
}
```

The batch cap of 200 is a deliberate compromise. Larger batches reduce round trips but increase the blast radius of a mid-upload failure and push against the request-body limits some mobile carriers' transparent proxies enforce. Those proxies can silently truncate oversized POSTs on some networks; capping the batch keeps each request comfortably inside typical limits and makes a failed batch cheap to retry.

## Step 3 — handle edge cases and errors

The three failure modes that actually bite:

**Duplicate delivery.** The agent's phone uploads a batch, the server commits, the response is lost on the way back (common on 2G — the request succeeds, the response doesn't). The client retries. Without server-side dedup on `id`, you get two "delivered" records for one parcel. The fix is a unique constraint on `scans.id` and an `ON CONFLICT DO NOTHING`, returning the accepted IDs.

**Clock drift.** A device that's been offline for two days can have `deviceTs` off by minutes. If you sort by `deviceTs` on the server, a parcel can appear "delivered" before it was "picked up." Sort by `seq` within an agent's session, and by `id` (UUIDv7) across sessions.

**Partial batch loss.** The client marks a record `synced: 1` before the server has actually committed. This happens when the client optimistically updates after a 200 that was actually a proxy's cached response. The fix: only mark synced after parsing the server's `accepted` array, never on status code alone.

The server handler enforces all three:

```javascript
// server.js
import { z } from 'zod';

const Scan = z.object({
  id: z.string().uuid(),
  seq: z.number().int().positive(),
  parcelId: z.string(),
  agentId: z.string(),
  status: z.enum(['picked_up','in_transit','delivered','failed']),
  deviceTs: z.number(),
  lat: z.number(), lng: z.number(), accuracyM: z.number(),
});

export async function postScansBatch(req, reply) {
  const { scans } = z.object({ scans: z.array(Scan).max(200) }).parse(req.body);
  const accepted = [];
  for (const s of scans) {
    const drift = Math.abs(Date.now() - s.deviceTs);
    if (drift > 300_000) req.log.warn({ id: s.id, drift }, 'clock drift');
    const r = await pool.query(
      `INSERT INTO scans (id, seq, parcel_id, agent_id, status, device_ts, server_ts, lat, lng, accuracy_m)
       VALUES ($1,$2,$3,$4,$5,$6,now(),$7,$8,$9)
       ON CONFLICT (id) DO NOTHING RETURNING id`,
      [s.id, s.seq, s.parcelId, s.agentId, s.status, s.deviceTs, s.lat, s.lng, s.accuracyM]
    );
    if (r.rowCount) accepted.push(s.id);
  }
  return reply.send({ accepted });
}
```

The 5-minute drift threshold is a starting point. For agents in areas without network time, tighten to 2 minutes and route flagged records to a review queue rather than rejecting them — the scan is still valid, the timestamp just needs a human to confirm.

## Step 4 — add observability and tests

You can't debug offline behaviour from a server log. The client needs to emit its own telemetry, buffered locally and flushed with the same queue. Track four metrics: queue depth, oldest-unsynced age, flush success rate, and clock drift. A queue depth that grows past 500 for more than 2 hours is the signal that something is wrong — either the endpoint is down or the device has been offline longer than the shift.

```typescript
// metrics.ts
export async function snapshot() {
  const pending = await db.getAllFromIndex('scans', 'by-synced', 0);
  const oldest = pending.length
    ? Math.min(...pending.map(p => p.deviceTs))
    : Date.now();
  return {
    depth: pending.length,
    oldestAgeMs: Date.now() - oldest,
    driftMs: Math.abs(Date.now() - (await getServerTime())),
  };
}
```

To measure the flush cost rather than guess at it, instrument the client directly:

- Wrap `flush()` in `performance.mark`/`performance.measure` and record the duration of each batch.
- Log `pending.length` before and after each flush, plus the byte size of the request body.
- Run the same script on the slowest device you actually support, throttled to a realistic profile in Chrome DevTools ("Slow 3G" plus 4x CPU slowdown is a reasonable proxy for a mid-range Android on 2G).
- Compare the indexed lookup against a full-store scan by temporarily removing the `by-synced` index and re-running.

For tests, use Vitest 1.6 with `fake-indexeddb` 6.0 to run the DB layer in Node. The test that catches the most bugs is the kill-and-resume test: write 500 records, close the DB mid-transaction, reopen, assert all 500 are present and none are marked synced. A common trap is testing only the happy path with a mocked fetch — that passes while the real device loses data on process kill.

| Approach | Durability | Sync latency | Battery cost | Best for |
|---|---|---|---|---|
| In-memory queue | None | ~0 ms | Low | Never in field ops |
| localStorage | 5 MB cap | ~2 ms | Low | Config only, not scans |
| IndexedDB (idb 8.0) | Transactional | Low ms | Low | PWA, < 5k rows/shift |
| Native SQLite plugin | Full ACID | Low ms | Medium | Capacitor, > 5k rows |
| Server-authoritative | N/A | Network-bound | High | Wi-Fi-only depots |

The latency column is deliberately qualitative. Any concrete millisecond figure depends on device, storage driver, and row count, so measure it on your own hardware using the instrumentation above rather than trusting a table.

## What to expect when you run this

The numbers below are illustrative, derived from the stated assumptions, not measured benchmarks. Treat them as a model to check against your own instrumentation.

Assume a batch of 200 scans, each record roughly 250 bytes of JSON, so a batch body is about 50 KB. On a 3G link with ~200 kbps effective throughput, 50 KB takes about 2 seconds to upload; four batches of 200 for an 800-scan queue is therefore roughly 8 seconds of pure transfer, plus per-request round-trip time. The naive alternative — one request per scan — pays the round-trip cost 800 times instead of 4, which is why a one-record-per-request design is so much slower on a lossy link.

For duplicates: if retries are not idempotent, every lost response produces a duplicate. On a link where responses are lost even a few percent of the time, that fraction of scans is duplicated. Idempotent inserts on `id` reduce that to zero by construction, because the second insert is a no-op.

For battery, the driver is radio wake time, not the number of requests. A fixed 5-second retry loop wakes the radio 720 times per hour; the backoff schedule above wakes it 12 times in the first minute and then once every 5 minutes. The exact drain depends on the radio and the OS, which is why the right move is to measure it with the device's own battery stats over a shift rather than quote a percentage.

The reconciliation report is where the value shows. With `device_ts`, `server_ts`, and `seq` all stored, a finance team can reconstruct exactly when a parcel changed state and whether the delay was network or human. That's the audit trail GDPR Article 30 and NDPA 2023 both expect you to be able to produce, and it's the difference between a disputed delivery and a resolved one.

## Common questions and variations

**What about multi-agent handoffs?** When a parcel passes from one agent to another, the receiving agent's scan needs the sender's `id` as a `parentId`. This creates a chain you can validate server-side. Without it, two agents can both mark "delivered" and you have no way to tell which is correct.

**Should the client encrypt the queue?** Yes, if scans contain personal data — a recipient's name and address is personal data under both GDPR and NDPA. Use the Web Crypto API with a key derived from the agent's login, and clear the key on logout. This prevents a stolen device from leaking the queue.

**How do you handle a device that's offline for a week?** The queue will grow to thousands of records. Cap the local store at 10,000 records and, past that, refuse new scans with a clear message rather than silently dropping old ones. Silent drops are the single worst failure mode in field ops.

**What if the server rejects a record permanently?** Return it in a `rejected` array with a reason, mark it locally, and surface it in a "needs attention" list. Never silently discard — the agent needs to know a scan didn't land.

## Decision checklist before you ship

- Every mutation has a client-generated stable ID (UUIDv7 preferred) before it touches the network.
- The durable write happens before the UI confirms success, not after.
- The sync worker uses backoff, not a fixed interval.
- The server enforces a unique constraint on the client ID and returns the accepted set.
- The client marks synced only from the accepted set, never from a status code.
- Ordering uses a monotonic sequence and UUIDv7, not the device clock.
- `device_ts` and `server_ts` are both stored, and drift is flagged.
- Queue depth and oldest-unsynced age are exported as metrics.
- There is a kill-and-resume test that asserts zero loss.
- There is a hard cap on local storage with a visible refusal, not a silent drop.

## Where to go from here

Open your current client's storage layer and count how many writes go straight to `fetch()` without a local durable write first. Every one of those is a scan you'll lose the next time an agent walks into a market with no signal. In the next 30 minutes, pick the single highest-frequency write in your app — usually the status update — and route it through IndexedDB with a UUIDv7 and a `synced` flag before you touch anything else. That one change is where the data integrity actually comes from.
