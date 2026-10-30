# Database restore drills that mean something

## The backup you never tested is a rumour

Every team has a backup job. Far fewer have a restore they can prove. The gap between the two is where disasters live: a `pg_dump` that has been failing silently for weeks because the cron user's password expired, a WAL archive that stopped shipping after a disk filled, an object-storage lifecycle rule that quietly transitioned backups to an archive tier and made a restore take hours longer than the RTO on the slide. None of these show up in a backup dashboard. They show up in a restore drill.

The problem this article addresses is that most "DR tests" are theatre — a quarterly checkbox where someone restores a small table to a staging box, sees rows, and declares victory. A drill that means something measures the same things a real incident would: how long until the database is usable, how much data is missing, and whether the application can actually run against the restored copy. What follows is a repeatable drill harness for PostgreSQL (the mechanics generalise to MySQL and managed equivalents) that produces a number you can put in front of a CTO, plus the failure modes that make drills lie.

## What a drill has to measure (and what a backup dashboard won't tell you)

Three quantities matter, and none of them is "backup succeeded":

- **RTO (recovery time objective)** — wall-clock time from "we decide to restore" to "the application serves reads and writes against the restored database." This includes provisioning, data transfer, replay, and application smoke tests. A dump that restores quickly from a local disk can take far longer when it has to come back from cold object storage over a constrained link. The only way to know which case applies is to measure the real retrieval path.
- **RPO (recovery point objective)** — the timestamp of the last committed transaction present in the restored copy. Nightly logical backups imply an RPO of up to 24 hours, no matter what the slide says. Continuous WAL shipping implies an RPO equal to the gap between the last archived segment and the failure.
- **Integrity** — does the restored database pass the checks the application depends on: foreign keys valid, row counts within expected bounds, sequences ahead of `max(id)`, extensions present, roles and grants intact.

A drill that only checks "the restore command exited 0" measures none of these. A drill that only checks row counts measures integrity but not RTO. The harness below records all three per run so they can be trended.

## Prerequisites: a restore target that is not production

Three things are needed before the first drill is meaningful:

1. **An isolated restore environment.** A separate VPC or at minimum a separate database instance with no route to production. A common drill accident is a `pg_restore` pointed at the wrong `PGHOST` because someone exported the variable in the wrong shell. Use a dedicated hostname like `drill-db.internal` and refuse to run if `PGHOST` resolves to a production address.
2. **A backup artefact fetched the same way an incident would fetch it.** If backups live in object storage with lifecycle rules, the drill must go through the same retrieval path — including any restore-from-archive-tier step. Testing against a local copy of the dump tests nothing about retrieval.
3. **A smoke-test script that exercises the application's real queries.** Not `SELECT 1`. The queries the app runs on its hottest path, plus any query that touches recently added columns or indexes.

For PostgreSQL, the two backup shapes worth drilling are logical (`pg_dump` / `pg_dumpall`, custom format with `-Fc`) and physical (base backup plus WAL archive, via `pg_basebackup` and continuous archiving, or a managed service's point-in-time restore). Drill both if both are relied on; they fail differently. Logical backups miss things like large objects depending on flags, and exclude roles unless `pg_dumpall --globals-only` is used alongside; physical backups depend on a WAL archive that can silently stop.

## Step 1 — Record the restore start and the backup's own timestamp

Before restoring anything, capture the metadata that makes the run comparable across weeks. The dump file's mtime is not the backup time; for logical dumps, the meaningful timestamp is when the dump transaction started. For custom-format dumps, `pg_restore -l` lists the archive's table of contents and header, which includes creation info. For physical backups, the `backup_label` inside the base backup records the start and stop WAL positions and time.

```bash
#!/usr/bin/env bash
set -euo pipefail

DRILL_ID="$(date -u +%Y%m%dT%H%M%SZ)"
DUMP_URI="s3://example-backups/pg/prod-$(date -u +%Y-%m-%d).dump"
WORKDIR="/var/tmp/drill-${DRILL_ID}"
mkdir -p "$WORKDIR"

# Refuse to run against anything that looks like production.
case "${PGHOST:-}" in
  *prod*) echo "refusing: PGHOST looks like production" >&2; exit 1 ;;
esac

START_EPOCH=$(date +%s)
aws s3 cp "$DUMP_URI" "$WORKDIR/prod.dump"
FETCH_EPOCH=$(date +%s)
echo "{\"drill_id\":\"$DRILL_ID\",\"fetch_seconds\":$((FETCH_EPOCH-START_EPOCH))}" >> "$WORKDIR/metrics.jsonl"
```

Why this matters: RTO is dominated by fetch time more often than teams expect, especially when backups sit in infrequent-access or archive storage tiers. Recording fetch separately from restore tells you whether to fix the storage class or the database.

## Step 2 — Restore into a clean database and time it

For a custom-format logical dump, restore with `pg_restore` into a freshly created database. Use `--exit-on-error` so a partial restore fails loudly instead of leaving a half-populated schema that passes a row-count check.

```bash
DRILL_DB="drill_${DRILL_ID}"
createdb "$DRILL_DB"
RESTORE_START=$(date +%s)
pg_restore \
  --dbname="$DRILL_DB" \
  --jobs=4 \
  --exit-on-error \
  --no-owner \
  "$WORKDIR/prod.dump"
RESTORE_END=$(date +%s)
echo "{\"drill_id\":\"$DRILL_ID\",\"restore_seconds\":$((RESTORE_END-RESTORE_START))}" >> "$WORKDIR/metrics.jsonl"
```

Two gotchas here are documented and still bite teams every quarter:

- **`--jobs` with `--single-transaction` is invalid.** Parallel restore and single-transaction are mutually exclusive in `pg_restore`; combining them errors out. For atomicity, drop `--jobs` and accept the slower restore, or accept a partial restore and rely on `--exit-on-error` plus dropping the database on failure.
- **`--no-owner` changes semantics.** Restoring without ownership means objects end up owned by the restoring role. That is usually what a drill wants (the production owner role may not exist on the target), but it means a drill that passes with `--no-owner` does not prove that a restore *with* ownership will succeed. If the runbook restores with owners, drill that path.

For a physical backup with WAL replay, the drill is a `pg_basebackup` restore into a data directory plus `restore_command` pointing at the WAL archive, then starting the server and waiting for recovery to reach the target. The measurement runs from "start recovery" to "database accepts connections and `pg_is_in_recovery()` returns false." Managed services expose this as point-in-time restore; the drill is the same shape, but the timing is whatever the console or API takes — which is precisely why it should be measured rather than trusted from an SLA page.

## Step 3 — Smoke-test the restored database like the application would

Row counts are necessary, not sufficient. The restored database has to answer the application's queries. Write the smoke test as SQL the app actually issues, with assertions.

```sql
-- smoke.sql: run with psql -v ON_ERROR_STOP=1 -f smoke.sql
DO $$
DECLARE
  n bigint;
  seq_last bigint;
  max_id bigint;
BEGIN
  -- Hot-path query the app runs on every page load.
  SELECT count(*) INTO n
  FROM orders o
  JOIN order_items oi ON oi.order_id = o.id
  WHERE o.created_at > now() - interval '1 day';
  IF n = 0 THEN
    RAISE EXCEPTION 'hot-path join returned zero rows; data may be stale or missing';
  END IF;

  -- Sequences must be ahead of max(id) or the first insert after failover collides.
  SELECT last_value INTO seq_last FROM orders_id_seq;
  SELECT max(id) INTO max_id FROM orders;
  IF seq_last < max_id THEN
    RAISE EXCEPTION 'sequence orders_id_seq (%) behind max(id) (%)', seq_last, max_id;
  END IF;
END $$;
```

The sequence check catches a real and common failure: logical restores recreate sequences at their dump-time value, but if any data was loaded or replayed after the dump (or if the restore is partial), the sequence can lag. The first write after cutover then throws a duplicate key error — during an incident, when it is least welcome. The `--no-owner` path can also leave sequences owned by the wrong role, which breaks `nextval` for the application role.

For physical restores, sequences are part of the data directory and generally consistent, but the smoke test still earns its keep: it proves the WAL replay reached a point where the application's schema and data are coherent.

A zero-row failure on the hot-path query deserves a note of caution: if the application legitimately has quiet periods, a strict `n = 0` assertion produces false alarms. Compare against a stored baseline instead — record the hot-path row count from production at the same hour on a previous day, and fail only if the restored count is below some fraction of it. The fraction is a judgement call; the point is that the threshold is explicit and reviewable rather than implicit in a human's glance at the output.

## Step 4 — Measure RPO by finding the last committed transaction

RPO is the quantity teams most often assert and least often measure. For logical dumps, the honest RPO is "time of dump start" — everything committed after that is gone. For WAL-based recovery, RPO is the gap between the last archived WAL segment and the failure time.

A practical drill technique: write a heartbeat row to a known table on a schedule (say every minute) from the application or a sidecar, with a timestamp. After restore, query the max heartbeat timestamp. The difference between that and the drill start time is the measured RPO for that run.

```sql
SELECT max(beat_at) AS last_heartbeat,
       now() - max(beat_at) AS measured_rpo
FROM drill_heartbeat;
```

This works for both logical and physical restores and turns RPO from a slide into a number. If the heartbeat table doesn't exist in production, add it — it is a single-row-per-minute table and among the cheapest observability available.

A worked example makes the arithmetic concrete. Suppose the heartbeat runs every 60 seconds, the last archived WAL segment covers transactions up to 14:00:00, and the simulated failure is declared at 14:04:30. The measured RPO is 4 minutes 30 seconds, because everything committed between 14:00:00 and the failure is not in the restored copy. If the target RPO is 5 minutes, the drill passes; if it is 2 minutes, the drill fails and the fix is either more frequent WAL archiving or `archive_timeout` set low enough to force segment switches. Note that `archive_timeout` forces a segment switch on a timer, so a low value trades archive volume for a tighter RPO bound — that trade-off is the actual engineering decision the drill surfaces.

## Step 5 — Verify grants, extensions, and the things dumps forget

A restore that passes row counts and smoke tests can still fail at cutover because of what the dump did not include. The recurring list:

- **Roles and grants.** `pg_dump` of a single database does not include roles; `pg_dumpall --globals-only` does. If the runbook restores only the database dump, the application role may not exist on the target, and `--no-owner` will have handed ownership to the restoring superuser. Drill the globals restore too if production relies on it.
- **Extensions.** `CREATE EXTENSION` statements are in the dump, but the extension binaries must be installed on the target. A drill on a host without the required extensions fails at restore, not at query time — better to find out in the drill.
- **Large objects.** `pg_dump` includes large objects by default, but `--no-blobs` excludes them. If any part of the app stores files as large objects, drill with blobs included.
- **Foreign keys and constraints.** Restoring with `--disable-triggers` (which requires superuser) or restoring data before constraints can leave invalid state that only shows up under write load. Prefer the default order and let `pg_restore` handle it; if reordering is necessary for speed, add a post-restore `ALTER TABLE ... VALIDATE CONSTRAINT` pass.

A verification query that catches the most common of these:

```sql
SELECT rolname FROM pg_roles WHERE rolname = 'app_rw';
SELECT extname FROM pg_extension;
SELECT count(*) FROM pg_largeobject_metadata;
```

If the first returns zero rows, the application role is missing and the restore is not usable regardless of data.

## Edge cases and the traps that make drills lie

- **The dump that has been failing silently.** A cron job that writes to a local path and then uploads can fail at either step. Check the artefact's age as part of the drill: if the newest dump is older than the backup interval, the drill should fail before restoring. Listing the object store and comparing dates is enough.
- **Storage class transitions.** If lifecycle rules move backups to a colder tier, the fetch step in Step 1 will be slow or, for archive tiers, require an explicit restore request that takes time. Drill the real retrieval path, not a cached copy.
- **WAL archive gaps.** For physical backups, a gap in the WAL archive means recovery stops at the gap. The drill should assert that recovery reached the target LSN or timestamp, not merely that the server started. `pg_waldump` on the archive, or the recovery target settings in `postgresql.conf` (such as `recovery_target_time`), make this explicit.
- **The `--jobs` / `--single-transaction` combination** mentioned earlier: it errors, and teams sometimes "fix" it by removing `--exit-on-error`, which is worse.
- **Restoring into a database that already has objects.** `pg_restore` into a non-empty database produces a stream of "already exists" errors that `--exit-on-error` will halt on. Always drop and recreate, or restore into a fresh name per drill.
- **Time zone and clock skew.** RPO measurement via heartbeat assumes clocks are synced. If the application host and database host drift, the measured RPO is wrong. Use the database's `now()` for both the heartbeat write and the post-restore query.

A failure-mode walkthrough shows how these interact. Consider a team that runs a quarterly drill against a dump copied to a workstation. The drill passes in 15 minutes every quarter. During a real incident, the restore from object storage takes 70 minutes because the objects were in an archive tier, the application role is missing because only the database dump was restored, and the sequence is behind `max(id)` because the dump predates the most recent data load. All three failures were invisible to the drill because the drill tested a different artefact, a different retrieval path, and a different set of checks. Each of the six traps above maps to one of those three substitutions.

## Verification: turning drills into a trend, not an anecdote

The output of each drill is a line of JSON. Keep them. After a few runs there is a trend of fetch seconds, restore seconds, measured RPO, and pass/fail per check. That trend is what to show when someone asks whether the DR plan works — not a slide, but a series of dated runs with numbers.

A minimal pass/fail gate suitable for CI or a scheduled job:

```bash
#!/usr/bin/env bash
set -euo pipefail
# Fail the drill if any check regressed beyond stated thresholds.
MAX_RESTORE_SECONDS=1800   # illustrative threshold; set from your RTO
MAX_RPO_SECONDS=300        # illustrative; set from your RPO

restore_seconds=$(jq -s 'map(.restore_seconds // 0) | max' "$WORKDIR/metrics.jsonl")
rpo_seconds=$(psql -Atc "SELECT extract(epoch FROM now() - max(beat_at))::int FROM drill_heartbeat")

if (( restore_seconds > MAX_RESTORE_SECONDS )); then
  echo "FAIL: restore took ${restore_seconds}s > ${MAX_RESTORE_SECONDS}s" >&2
  exit 1
fi
if (( rpo_seconds > MAX_RPO_SECONDS )); then
  echo "FAIL: measured RPO ${rpo_seconds}s > ${MAX_RPO_SECONDS}s" >&2
  exit 1
fi
echo "PASS: restore ${restore_seconds}s, RPO ${rpo_seconds}s"
```

The thresholds are illustrative; derive them from the RTO and RPO the team has committed to, and treat a breach as a real incident ticket, not a note in a wiki. The point of the drill is to find the breach on a Tuesday afternoon, not during a failover.

## Frequently Asked Questions

**How often should database restore drills run?**
Frequency should follow how fast the schema and backup configuration change, not a calendar. A team shipping schema migrations weekly and changing backup tooling quarterly gets more value from a weekly automated drill than a quarterly manual one. The practical rule: automate the drill so it runs on a schedule, and have a human review the trend monthly. If a drill fails, that is a ticket, not a rerun.

**What is the difference between RTO and RPO in a restore drill?**
RTO is how long until the database is usable again; RPO is how much data was lost, expressed as time. A drill that measures only restore duration measures RTO and ignores RPO. Measure both: RTO from restore start to passing smoke test, RPO from the newest heartbeat row in the restored copy versus the drill start time.

**Can restores be tested against a replica instead of a separate environment?**
A replica tests replication, not restore. Restoring from a dump or a base backup exercises a different code path — file retrieval, archive replay, role and extension setup — and that is where the failures are. Use a replica for failover drills and a separate restore target for restore drills; they answer different questions.

**Why does a logical restore pass but the application fail after cutover?**
The usual causes are missing roles (single-database dumps exclude globals), sequences behind `max(id)`, or extensions not installed on the target. All three pass a row-count check and fail the first real write. Add the role, sequence, and extension checks from Step 5 to the smoke test.

## The next 30 minutes

Open the backup job's configuration and find the exact command that produces the artefact — then run `pg_restore -l` (or the equivalent listing command for the backup tool) against the newest artefact and confirm it is younger than the backup interval and contains the tables expected. If it is not, the failure has been found before an incident did; if it is, the drill harness above has its input.
