# Designing Offline-First Apps for Unreliable Networks

Offline-first guides often assume a clean environment and a patient timeline. Production gives you neither. This article covers the failure modes that show up when an app must work on metered data, intermittent signal, and low-end hardware — and the architecture that survives them.

## The problem with online-first assumptions

Consider a personal finance app for users in Nigeria, Kenya, and Ghana, where mobile data is metered and signal drops are routine. The core feature is expense tracking with cloud sync — a classic online-first architecture. When this pattern ships into Lagos or Nairobi, a familiar failure appears: sessions fail on network timeouts even though the API responds quickly when reachable. The consequence is that a portion of the audience cannot log expenses on payday, when budgets matter most.

The common mistake is assuming connectivity will be "good enough." On 2G and 3G networks, users often see latency bursts measured in seconds, with packet loss climbing during peak hours. An app whose only retry strategy is a single exponential backoff will lose users even after the network recovers, because they have already abandoned the task.

Offline-first does not mean "store everything and sync later." It means the app must be fully functional without a network, degrade gracefully when offline, and stay secure when data eventually leaves the device. In emerging markets this also means handling SIM-switched phones, low-storage devices, and users who disable mobile data to save credit.

A typical first attempt uses a thin wrapper around a browser database and a single sync endpoint. That is a few hundred lines of code, but it fails on two counts: it does not protect user data if the device is lost, and it assumes sync will complete in one attempt. Storage, encryption, conflict resolution, and sync scheduling all need separate design work.

## What teams try first, and why it breaks

The first common move is to bolt a local relational store onto the app — often SQLite inside a WebView wrapper on Android and iOS — plus a background sync service scheduled through the platform's background task APIs. SQLite is a natural pick: it is lightweight and supports ACID transactions, which matters when two users share one device. It feels like this should solve everything. It does not.

**Failure mode 1: silent sync failure after network changes.** Background jobs are often scheduled with a network constraint and then assumed to succeed. When the device's network changes — a SIM swap, a Wi-Fi handoff, a captive portal — the job can be cancelled or deferred and simply never retried with the same urgency. The fix is not a bigger retry count; it is persisting the queue durably and treating each sync attempt as resumable.

**Failure mode 2: unencrypted local data.** A relational store is not encrypted by default. If a device is stolen, an attacker with filesystem access can read the raw records. Encryption at rest is a requirement, not a hardening step, for anything financial.

**Failure mode 3: wrong assumptions about user behavior.** Teams assume users open the app daily. Many open it twice a month. If the sync window is short and the queue is capped, expenses pile up and the request eventually fails with a payload-too-large error once the queue exceeds the server's limit. The queue must be chunked and the limit must be known.

**Failure mode 4: identity bound to the device.** Tokens tied to a hardware identifier break when the SIM or device changes. Tokens then have to be revoked manually, and users see "login required" without understanding why.

A browser-only approach using a service worker for precaching and background sync hits the same wall from a different direction. On feature-phone platforms and some low-end Android builds, background execution is unavailable or heavily restricted. The sync never fires, and users lose data when they close the browser. That is the moment it becomes clear offline-first is not about the happy path — it is about every device, every connection state, and every user behavior.

## The architecture that holds up

The durable approach rests on three pillars: local-first storage, secure sync, and explicit conflict resolution.

**Local-first storage.** Keep a relational store for structured data, but put a reactive layer on top so the UI reads from local state and never blocks on a network call. Observable queries mean a write updates the screen immediately, and the sync layer reconciles in the background. Batching and partial sync are handled by the layer, not by ad-hoc code in each screen.

**Encryption.** Encrypt the database with a per-user key derived from a user secret plus a hardware-backed keystore when one is available. Store the device secret in the Android Keystore or iOS Keychain, never in plaintext. Key derivation adds latency on first unlock; subsequent unlocks use cached keys.

**Sync scheduling.** Design a two-phase protocol: local writes are queued immediately and durably, and sync attempts happen in the background with exponential backoff capped at a bounded interval (24 hours is a reasonable ceiling). Use the platform's background scheduler with a network constraint, but never depend on it firing on time — the queue is the source of truth.

**Identity.** Move from hardware-bound tokens to short-lived, revocable credentials tied to the user session. A short access-token lifetime plus a refresh token stored in the secure enclave allows revocation when a device is lost, without forcing a re-login on every network change. A "sync on demand" button gives users control and reduces support contacts.

**Conflict resolution.** This is the hardest part. Last-write-wins for numeric fields and timestamps is simple but loses data when two devices edit the same record offline. A hybrid works better: last-write-wins for amounts and timestamps, manual merge prompts for free-text fields such as descriptions. A small metadata header on every sync payload makes it possible to detect partial syncs and resume without data loss.

**Storage budgeting.** Add a low-storage mode that prunes old sync logs and compacts the database when free space drops below a threshold. Run compaction when the app launches, not mid-interaction.

## Implementation details

The core sync loop below uses a reactive local store, an encrypted database, and a background worker that respects device constraints.

**Local storage schema (reactive ORM):**
```javascript
import { tableSchema } from '@nozbe/watermelondb';

const expenseSchema = tableSchema({
  name: 'expenses',
  columns: [
    { name: 'description', type: 'string' },
    { name: 'amount', type: 'number' },
    { name: 'date', type: 'number' },
    { name: 'sync_status', type: 'string' },
    { name: 'sync_version', type: 'number' },
  ],
});
```

Add a `pending_sync` table for operations that failed and an `expense_sync_versions` table to track per-record versions. Each record carries a `sync_version` that increments on every local change and is compared with the server during sync.

**Background sync worker (Android):**
```kotlin
val constraints = Constraints.Builder()
    .setRequiredNetworkType(NetworkType.CONNECTED)
    .setRequiresBatteryNotLow(true)
    .build()

val syncWork = PeriodicWorkRequestBuilder<SyncWorker>(
    15, TimeUnit.MINUTES, // minimum interval
    5, TimeUnit.MINUTES  // flex interval
).setConstraints(constraints)
 .build()

WorkManager.getInstance(context).enqueueUniquePeriodicWork(
    "expenseSync",
    ExistingPeriodicWorkPolicy.KEEP,
    syncWork
)
```

On iOS, the equivalent background task API is stricter about when work runs; schedule with a longer interval and add a "sync now" button that triggers an immediate attempt with a short expiration window. The 15-minute minimum above is the documented floor for periodic work on Android; anything shorter is silently coalesced.

**Encryption layer:**
```sql
-- Open database with per-user key derived from password + device secret
PRAGMA key = 'x\' || hex(sha256(user_password || device_id)) || '\'';
PRAGMA cipher_page_size = 4096;
PRAGMA cipher_plaintext_header_size = 32;
```

Note that this pseudo-SQL illustrates the shape of the key derivation; in practice, use the encryption library's supported key API rather than string-concatenating secrets into SQL. Store the device secret in the platform keystore, never in plaintext.

**Conflict resolver (hybrid strategy):**
```typescript
async function resolveConflicts(local: Expense, remote: Expense) {
  if (local.sync_version > remote.sync_version) {
    // Local is newer; prefer local unless the remote amount differs materially
    if (Math.abs(local.amount - remote.amount) > 0.01) {
      return askUserToMerge(local, remote);
    }
    return local;
  }
  return remote;
}
```

Drive the UI from reactive state so conflicts surface as a banner ("2 expenses merged. Tap to review.") rather than a silent overwrite. Log conflict events to your error tracker to find edge cases.

**Storage budgeting:**
```sql
-- Prune old sync logs when free disk space is low
SELECT COUNT(*) FROM expenses WHERE date < date('now', '-90 days');
DELETE FROM expenses WHERE date < date('now', '-90 days');
VACUUM;
```

Run this on launch and when free space drops below your threshold. `VACUUM` rewrites the database file, so run it when the app is idle rather than during a write.

## How to measure whether any of this worked

Do not trust a before/after table from someone else's deployment. Instrument these on your own build and compare over the same window.

| What to measure | How to instrument it | What "good" looks like |
|---|---|---|
| Sync failure rate | Count sync attempts that end in error or timeout, divided by total attempts, tagged by error class | Failures are rare and each one is attributable to a known cause |
| Queue depth | Record the number of pending operations at each app foreground event | Depth returns to zero after connectivity returns, without a manual retry |
| Session abandonment offline | Log a session start with no network and no subsequent write within N minutes | Abandonment drops after the local-first rewrite |
| Write latency | Timestamp the local write call on a low-end reference device | Stays within your interaction budget (a few hundred ms) |
| Payload size | Log request body size for each sync call | Delta sync keeps it roughly proportional to changed records, not to total records |
| Storage growth | Sample the database file size weekly per device tier | Growth is bounded and compaction reclaims space |

A simple network probe that logs latency and packet loss every few minutes tells you what fraction of sessions exceed your latency budget and what fraction exceed your loss budget. Without that data, it is easy to optimize for the wrong problem.

To estimate data cost for a user, work from your own measurements. If an online-first client sends roughly 8 KB per sync attempt and a delta-based client sends roughly 1.4 KB, and a user syncs 40 times a month, the delta approach moves about 6.6 KB × 40 ≈ 264 KB less per month. Multiply by the carrier's per-MB rate in your target market to get a per-user figure. Treat any such number as illustrative until you have measured it on your own traffic.

## What to do differently next time

**Do not use long-lived tokens for offline identity.** Long-lived tokens are hard to revoke when a device is lost. Prefer short-lived, sender-constrained access tokens (for example, DPoP-bound tokens) with refresh tokens stored in the device's secure enclave. That allows revocation without user interaction.

**Model network identity changes explicitly.** When a user switches SIMs the device's IP changes. A session that is not designed to survive that change ends up orphaned, and users report "login required" while believing they are logged in. A silent re-authentication flow on network change fixes this.

**Test encryption cost on the devices your users actually own.** Encryption is not free on low-end hardware. Measure write latency and memory on a 1 GB RAM reference device before committing to a page size or cipher configuration. If writes exceed your interaction budget, reduce the page size or move heavy writes off the main thread.

**Build a data-saver mode early.** A flag that skips image uploads and reduces sync frequency on metered networks is cheap to implement and directly visible to users. Surface it in settings rather than hiding it behind a heuristic.

## The broader lesson

Offline-first is not a feature; it is a constraint. Every decision — storage, identity, conflict resolution — must work when the network is unreachable, the device is low on power, and the connection is metered. The hardest part is not the code; it is the assumptions. Teams assume users open the app daily, that SIM switches are rare, that background execution will fire, and that users will trust auto-sync. None of those hold reliably.

The principle to follow is: **design for the worst connection, not the average.** If your app feels sluggish on a fast network, it will be unusable on a slow one. If your sync fails silently, users will blame the app, not the network.

Security and offline-first are not opposites. Encrypting local data is not optional; revoking tokens when a device is lost is not optional. Without both, offline-first becomes offline-only, which is not acceptable for a finance app. The key is to encrypt without crippling performance and to revoke without blocking the user.

Finally, users want control over their data. Give them a "sync now" button, a low-data mode, and a clear explanation of when and why data leaves their device. Transparency builds trust, and trust is the foundation of any financial app.

## How to apply this to your situation

Start by mapping your users' connection realities. What is the median latency in your target market? What is the packet loss? How often do users switch SIMs? If you do not have data, run a lightweight telemetry probe in your current app for two weeks. A probe that logs latency and packet loss every five minutes tells you what fraction of sessions exceed your latency and loss budgets.

Next, audit your data model. What happens if a user loses their device tomorrow? Can you recover their local data? For many teams the answer is no until they add encryption with a per-user key. Then ask: what conflicts can occur? If two users edit the same record offline, how is it resolved? Last-write-wins is simple but loses data; operational transforms are robust but complex. Pick the simplest strategy that meets your users' needs.

Then implement a minimal offline-first stack: a local relational store with a reactive layer, encryption at rest, and a background worker with a durable queue. For web apps, use a service worker with a local database and the Background Sync API where it is available, and a manual sync fallback where it is not. Avoid heavy frameworks; offline-first rewards simplicity. Add a "sync on demand" button and a storage budgeting policy — these two features alone address a large share of user pain.

Test on low-end devices. Use an Android Go emulator or a cheap physical device. Measure memory usage, disk usage, and battery impact. If writes take longer than your interaction budget, optimize queries or move to a lighter storage layer.

Finally, measure what matters: sync failure rate, session abandonment, and data usage. Do not optimize for "feels fast." Optimize for "works when the network doesn't."

## FAQ

**How do I handle sync conflicts when two users edit the same record offline?**
Start with last-write-wins for numeric fields and timestamps, but prompt the user to merge free-text fields such as descriptions. Use operational transforms only if the app is genuinely collaborative. For finance apps, manual merge prompts are safer and simpler. Log conflicts to your error tracker to spot edge cases early.

**What is the best way to encrypt local data without killing performance?**
Use a database encryption layer with a per-user key derived from a user secret plus a hardware-backed keystore. Store the device secret in the Android Keystore or iOS Keychain. Measure write latency on a low-end reference device before choosing a page size; smaller pages reduce memory pressure at some cost to throughput.

**How do I prevent data loss when a user switches SIMs or loses their device?**
Encrypt local data with a per-user key. For recovery, add an export flow that produces an encrypted backup the user controls. Do not assume cloud backup is available — many users disable it to save data. If you do use cloud backups, encrypt them with a user-provided key and warn users that the backup is unrecoverable without it.

**What background execution limits should I expect on low-end devices?**
Some feature-phone platforms do not support service workers or background sync at all. On Android Go, periodic work is restricted to a documented minimum interval. Test on real devices early. Where background execution is unavailable, implement a manual "sync now" button and a local reminder to open the app periodically.

**How do I reduce data usage for users on metered connections?**
Add a low-data mode that disables image uploads, reduces sync frequency, and compresses payloads. Measure data usage per user and offer to enable the mode automatically when usage crosses a threshold the user sets.

**What is the smallest offline-first stack I can start with?**
A local relational store with a reactive layer, encryption at rest, and a background worker with a durable queue. Add a "sync on demand" button and a storage budgeting policy that prunes old data when free space is low. That is a few hundred lines of code beyond your existing data layer and adds a small amount to your app size.

## The one thing to do in the next 30 minutes

Open your sync endpoint's request logging and record the average request body size over the last 24 hours. If it is above 2 KB, you are almost certainly sending full records instead of deltas — add a `since` parameter and a `changed_fields` field to your sync API, then re-measure.
