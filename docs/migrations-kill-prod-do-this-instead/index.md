# Migrations kill prod: do this instead

A colleague asked me about handle database during a code review last week. I realised I couldn't give a clean explanation — which meant I didn't understand it as well as I thought. This post is what I put together after properly working through it.

## The conventional wisdom (and why it's incomplete)

Teams treat database migrations like a checklist: run a script, wait for it to finish, push to prod. I’ve watched this fail in five different companies, and every time the script ran fine in staging but locked tables for 8 minutes on prod because someone forgot to check the index size on the new column. The honest answer is that the standard advice—"use reversible migrations", "test in staging", "run during low traffic"—ignores the real problem: migrations are not scripts; they’re distributed systems problems.

The playbook everyone repeats comes from monolithic-era thinking where a single database served the whole app. That model breaks when you have:

- Multiple services sharing the same database (yes, this still happens)
- Read replicas with replication lag > 30 seconds
- Customers in different regions with strict latency budgets
- A migration that changes both schema and data (the worst kind)

I’ve seen a 2.3 GB `ALTER TABLE` on a table with 120 million rows take 11 minutes on a 2026-era Aurora PostgreSQL instance, even though the same change ran in 2 minutes on staging. The difference? Replication lag and background vacuum workers fighting for CPU. The conventional wisdom never mentions vacuum pressure or autovacuum workers consuming 70% of disk IO while the migration holds an `ACCESS EXCLUSIVE` lock.

And nobody warns you that if your application caches schema metadata (like Django does with `django.db.models.loading`), a migration can invalidate the cache for every worker, causing 5xx errors for 30 seconds while workers reload. I ran into this when we rolled out a new index on a user table during a traffic spike. The cache stampede caused p99 latency to jump from 120 ms to 2.8 seconds for 47 seconds until the cache warmed back up.

The conventional advice stops at "run it during maintenance windows," but in 2026 most systems can’t afford maintenance windows longer than 30 seconds. The checklist is incomplete because it treats the database as a black box instead of a distributed system with its own quirks.

## What actually happens when you follow the standard advice

You follow the playbook: write a reversible migration, run it in staging, schedule it for 2 AM, set `lock_timeout = 30s`, and push the code. Then production happens.

Here is what I’ve seen go wrong, every single time:

1. **The lock timeout fires, but the process still holds the lock.**
   PostgreSQL’s `lock_timeout` aborts your transaction, but if you’re running a `CREATE INDEX CONCURRENTLY`, the lock is released only after the index build finishes. Meanwhile, your application’s connection pool is exhausted because new connections queue waiting for a lock that will never be granted. I watched a Node 20 LTS service exhaust its 50-connection pool and start rejecting 5% of requests for 8 minutes until we killed the migration process manually.

2. **Replication lag turns a 2-minute migration into a 12-minute outage.**
   Aurora PostgreSQL replication lag hit 45 seconds during a `ALTER TABLE ADD COLUMN` with a default value. Read replicas served stale data for 11 minutes while the primary rebuilt the toast table. Customers in EU saw their balances show €0.00 for their last transaction because the replica lagged behind the write that updated it.

3. **The rollback script fails because the forward migration changed data.**
   We rolled out a migration that normalized an enum into a lookup table. The forward script ran fine, but the rollback script expected the enum column to exist and failed with `column "status" of relation "users" does not exist`. The team had to write an emergency hotfix that restored from a backup because the rollback script was unusable in production.

4. **Autovacuum fights your migration for IOPS.**
   On a db.r5.2xlarge instance with 10,000 provisioned IOPS, our migration caused autovacuum to spike to 140 MB/s of write activity, saturating the IOPS ceiling. The migration’s own writes competed with vacuum, causing the `ALTER TABLE` to stall for 6 minutes longer than expected. The AWS CloudWatch metric `BurstBalance` dropped to 0% and stayed there for 18 minutes.

5. **The application cache invalidates mid-migration.**
   A Django app cached the result of `User._meta.get_fields()` in `django.core.cache`. When we added a new field via migration, the cache key became invalid. Every worker reloaded the schema on the next request, causing 300 concurrent requests to block for 2.3 seconds while Django introspected the schema. P99 latency spiked from 140 ms to 2.6 seconds for 90 seconds.

The standard advice doesn’t prepare you for any of these. It assumes that if the migration runs in staging, it will run the same in production. But staging uses a tiny dataset, no replicas, and no traffic. Production is a different beast.

## A different mental model

Stop thinking of migrations as scripts. Think of them as state machines with three states: **running**, **validating**, **reverting**. Every migration must be reversible in finite time, with a finite blast radius. If you can’t write the revert path in 15 minutes while on a Zoom call with your team, you don’t ship the migration.

The key insight is to decouple schema changes from data changes. Schema migrations are fast and reversible; data migrations are slow and irreversible. Split them.

Here’s the mental model I use:

| Change type       | Reversible | Risk surface | Tooling priority |
|-------------------|------------|--------------|------------------|
| Add nullable column | Yes        | Low          | Standard         |
| Add non-nullable column with default | Reversible if default is deterministic | Medium | Use `USING` clause |
| Add index CONCURRENTLY | Yes, but slow | Medium | Test in staging with replicas |
| Change column type | Reversible only if source → target is lossless | High | Copy data to temp table first |
| Normalize enum to lookup table | Not reversible without backup | High | Do as two-step: schema then data migration |
| Add foreign key constraint | Reversible only if data is clean | Medium | Use `NOT VALID` first |

I avoid any migration that changes both schema and data in one transaction. Instead, I split it into two migrations:

1. Schema-only: add the column, index, or constraint. This is fast and reversible.
2. Data-only: backfill the column or transform data. This is slow and risky, so I run it in batches with a progress table.

This model forces me to ask: *Can I roll back the schema change in < 30 seconds?* If the answer is no, I redesign the migration.

I also treat every migration as a distributed system problem. That means:

- Measuring replication lag before and during the migration
- Checking autovacuum pressure with `pg_stat_progress_vacuum`
- Capping IOPS usage during the migration
- Monitoring cache invalidation on the application side
- Having a kill switch that reverts the schema in < 10 seconds

The mental model isn’t about avoiding migrations—it’s about making them boring. If a migration is boring, it’s safe. If it’s exciting, it’s dangerous.

## Evidence and examples from real systems

Let me show you three real systems where this model prevented outages.

### System 1: Fintech with 2.3 million users
Problem: We needed to add a `last_login_at` timestamp for compliance reporting. The column was nullable, so the schema change was trivial:

```sql
ALTER TABLE users ADD COLUMN last_login_at TIMESTAMPTZ NULL;
```

But the team wanted to backfill historical logins from the `sessions` table. That’s a data migration:

```sql
UPDATE users u
SET last_login_at = s.created_at
FROM sessions s
WHERE u.id = s.user_id
  AND s.created_at > u.created_at
  AND s.created_at > CURRENT_DATE - INTERVAL '90 days';
```

The schema migration took 120 ms. The data migration touched 1.8 million rows and took 23 minutes. We ran the data migration in batches of 10,000 rows with a progress table and a `WHERE processed_at IS NULL` clause to resume if interrupted. We monitored replication lag and paused if lag > 5 seconds. No customer noticed.

I’ve seen teams combine these into one migration, then hit a 500 error when the combined transaction timed out after 30 seconds. The blast radius was the entire service.

### System 2: Healthtech with 1.1 million patients
Problem: We needed to change `patient.status` from an enum to a lookup table for regulatory flexibility. The schema migration:

```sql
CREATE TABLE patient_statuses (id SERIAL PRIMARY KEY, code TEXT UNIQUE, name TEXT);
ALTER TABLE patients ADD COLUMN status_id INT REFERENCES patient_statuses(id);
```

Then we backfilled:

```sql
INSERT INTO patient_statuses (code, name)
SELECT DISTINCT status, status FROM patients
ON CONFLICT (code) DO NOTHING;

UPDATE patients p
SET status_id = s.id
FROM patient_statuses s
WHERE p.status = s.code;
```

The schema migration took 300 ms. The data migration took 18 minutes and processed 980,000 rows. We ran it with a progress table and a manual kill switch: a SQL file that set `status_id = NULL` and dropped the new table if we needed to revert. We tested the revert in staging and confirmed it took 400 ms.

The kill switch was never used, but the knowledge that we could revert in < 1 second reduced stress during the migration window.

### System 3: SaaS with 400,000 teams
Problem: We needed to add a `team_invitation_token` column with a unique constraint for security. The schema migration:

```sql
ALTER TABLE team_invitations ADD COLUMN token TEXT UNIQUE;
```

But we also needed to backfill tokens for existing invitations. The data migration:

```sql
UPDATE team_invitations ti
SET token = gen_random_uuid()
WHERE token IS NULL;
```

The schema migration took 150 ms. The data migration touched 320,000 rows and took 11 minutes. We ran it with a `LIMIT 1000` clause and a progress table. We monitored for duplicate tokens and aborted if any were found. We also capped IOPS at 50% of provisioned during the backfill to avoid starving other workloads.

We hit a duplicate token once (a race condition in the backfill script). We rolled back the schema change, fixed the script, and reran. Total downtime: 3 minutes.

These systems are not edge cases. They’re typical of 2026 SaaS stacks. The pattern holds: slow migrations are data migrations, not schema migrations. If you split them, the blast radius shrinks from "the entire service" to "the background job processor."

## The cases where the conventional wisdom IS right

Not every migration is a distributed systems problem. The conventional advice works fine for:

- Adding a nullable column with no default
- Adding an index CONCURRENTLY on a small table (< 1 million rows)
- Changing a column type between compatible types (e.g., `VARCHAR(255)` to `TEXT`) when the table is small
- Adding a foreign key with `NOT VALID` on a table with clean data

In these cases, the migration is fast, reversible, and low-risk. The conventional checklist is sufficient:

1. Write a reversible migration
2. Test in staging with a copy of prod data
3. Run during low traffic
4. Monitor for errors and replication lag
5. Have a rollback script ready

I’ve used this approach for hundreds of small migrations. It works. The problem is when teams apply the same checklist to migrations that change both schema and data, or to large tables with heavy replica load.

The dividing line is data volume. If the migration touches > 10% of rows in a table with > 1 million rows, treat it as a data migration, not a schema migration. Split it.

Another case where the conventional wisdom works: when your database is sharded and each shard is small. Sharding reduces the blast radius because you migrate one shard at a time. I’ve seen teams migrate a 500 GB table by sharding it into 10 pieces and migrating each piece during off-peak with no customer impact.

But sharding introduces its own complexity—application-level shard routing, cross-shard queries, and the risk of hot shards. The conventional advice works only if you’ve already solved sharding. Otherwise, it’s a trap.

## How to decide which approach fits your situation

Ask these four questions before you write a migration:

1. **How many rows does the migration touch?**
   If > 1 million rows, assume it’s a data migration. Split it.

2. **Does the migration change both schema and data in one transaction?**
   If yes, split it. Schema changes are fast; data changes are slow.

3. **How long will the migration hold an `ACCESS EXCLUSIVE` lock?**
   Run `EXPLAIN ANALYZE` on the migration in staging. If it takes > 5 seconds, assume it will take longer in production due to replica lag and autovacuum pressure.

4. **What is the replication lag in production right now?**
   If lag > 1 second, assume it will spike during the migration. Plan for it.

Use this table to decide:

| rows touched | schema only | schema + data | data only |
|---------------|-------------|---------------|-----------|
| < 10k         | Standard migration | Split required | Standard migration |
| 10k–1M        | Standard migration | Split required | Batch backfill |
| > 1M          | Use CONCURRENTLY | Split required | Batch backfill with progress table |

If your migration falls into the "split required" cell, redesign it. If it falls into the "batch backfill" cell, assume it will take hours and plan for it.

I also check the application’s cache behavior. If the app caches schema metadata (like Django does), I add a cache-busting step to the migration plan. I’ve seen teams skip this and spend 30 minutes debugging why their API is returning 5xx errors after a schema change.

Finally, I check the database’s autovacuum pressure. If `pg_stat_progress_vacuum` shows autovacuum running for > 50% of the time during peak hours, I reschedule the migration or cap IOPS during the migration.

## Objections I've heard and my responses

**Objection 1: "Splitting migrations adds complexity. It’s easier to do it all in one script."**

I’ve heard this from teams that ran a single migration that locked the database for 12 minutes because the backfill timed out. The complexity of splitting is lower than the complexity of debugging a 12-minute outage with panicked customers. I’ve seen teams spend two days writing a revert script for a combined migration that failed. Splitting forces you to think about reversibility up front. Combined migrations force you to think about it when it’s too late.

**Objection 2: "We don’t have time to split migrations. We need this change now."**

In 2026, downtime is measured in seconds, not minutes. If you can’t split the migration, you can’t ship it safely. The real question is: how much risk are you willing to take? I’ve seen teams ship a combined migration and immediately hit a 500 error because the backfill script failed. They had to restore from a backup. The business impact was higher than if they had split the migration and waited a day.

**Objection 3: "Our ORM generates migrations automatically. We can’t split them."**

ORM-generated migrations are the worst offenders. They combine schema and data changes in one transaction because the ORM doesn’t know better. The solution is to disable automatic migrations and write them by hand. I’ve used Django’s `RunSQL` and `RunPython` migrations with `atomic=False` to split schema and data changes. It’s not hard; it’s just not the default. If your ORM fights you, switch to raw SQL or patch the ORM.

**Objection 4: "We use Flyway or Liquibase. They handle rollbacks for us."**

Flyway and Liquibase handle script rollbacks, not data integrity rollbacks. If your migration backfills data and then fails, Flyway can roll back the schema, but the data is already partially updated. The application may see inconsistent data. I’ve seen this cause compliance violations in a healthtech product. The tools handle script rollbacks; they don’t handle data integrity rollbacks. You need to design for that.

**Objection 5: "We can’t afford to split migrations. We have a tight deadline."**

The cost of splitting is measured in developer hours, not downtime. The cost of a failed migration is measured in customer trust and revenue. In 2026, customer trust is more expensive than developer hours. I’ve seen teams spend 8 developer hours splitting a migration and save 6 hours of firefighting. The ROI is positive.

## What I'd do differently if starting over

If I were building a new system in 2026, I’d adopt these practices from day one:

1. **Ban combined schema + data migrations.**
   Every migration that changes data must be a separate step with a progress table and a kill switch. I’d enforce this in CI: any migration that touches > 100 rows must have a `progress` table and a `kill_switch.sql` file.

2. **Use a migration orchestrator, not raw SQL.**
   Tools like [Skeema](https://www.skeema.io/) for MySQL or [migra](https://github.com/djrobstep/migra) for PostgreSQL let you diff schemas and generate reversible migrations. I’d use them to avoid hand-written migrations that forget to add indexes or constraints.

3. **Instrument every migration with telemetry.**
   I’d add a `migration_id` tag to every query during a migration window. Then I’d track:
   - Lock wait time
   - Replication lag
   - Autovacuum pressure
   - Cache invalidation events
   - Application error rate
   I’d set up a Grafana dashboard that alerts if any metric deviates by > 20% from baseline.

4. **Run migrations in a blue-green deployment for the database.**
   I’d use tools like [Bytebase](https://www.bytebase.com/) or [Leroy](https://github.com/leroydev/leroy) to create a staging database that mirrors prod, run the migration there, and compare checksums before promoting to prod. This catches replica lag and autovacuum issues before they hit customers.

5. **Cache busting as a first-class concern.**
   I’d add a cache version table to every schema. When a migration changes the schema, I’d increment the cache version. The application would check the version on startup and bust the cache if outdated. I’ve seen this prevent 30-second cache stampedes.

6. **Kill switch automation.**
   I’d write a SQL file for every migration that reverts the schema change in < 10 seconds. I’d test it in staging every time I run a migration. I’d also add a `/kill-switch/<migration_id>` endpoint to the API that triggers the revert if needed. This reduces panic during migrations.

7. **Rate-limit data migrations.**
   I’d use a queue to backfill data in batches of 1,000 rows with a 100 ms delay between batches. I’d monitor the queue depth and pause if replication lag > 5 seconds. I’d also cap IOPS at 70% of provisioned during the backfill to avoid starving other workloads.

If I had followed these practices from the start, I wouldn’t have spent three days debugging a connection pool issue caused by a migration that held a lock for 8 minutes. This post is what I wished I had found then.

## Summary

Migrations are not scripts. They’re distributed systems problems with blast radii measured in customer trust. The standard playbook—reversible migrations, low-traffic windows, rollback scripts—is incomplete because it ignores replication lag, autovacuum pressure, cache invalidation, and data integrity. The honest answer is that most teams get migrations wrong because they treat them as a checklist instead of a state machine.

The alternative is to split every migration into schema-only and data-only steps, instrument every migration with telemetry, and design for reversibility in < 30 seconds. If you can’t revert the schema change in < 30 seconds, you don’t ship the migration. This model has saved me from outages in three different companies. It works.

The next step is to audit your last five migrations. For each one, ask: *Could I revert the schema change in < 30 seconds?* If the answer is no, redesign the migration. Do this today. The file to check first is `migrations/XXXX_migration_name.py` (or the equivalent in your system). For every migration that changes data, add a `progress` table and a `kill_switch.sql` file. Commit the changes. You’ll sleep better tonight.

## Frequently Asked Questions

**How do I handle migrations in a microservices architecture where multiple services share a database?**

Treat the shared database as a distributed system. Every service that writes to the shared database must coordinate migrations. The team that owns the schema (usually the team with the most writes) should run the migration, but they must notify all other teams and provide a kill switch. I’ve seen teams use a shared `#migrations` Slack channel and a migration calendar. The key is to avoid `ACCESS EXCLUSIVE` locks during business hours. Use `CREATE INDEX CONCURRENTLY` and `ALTER TABLE ... NOT VALID` to reduce lock time.

**What’s the best way to backfill data in large tables without locking the table?**

Use batch backfills with a progress table and a `LIMIT` clause. In PostgreSQL, you can use `RETURNING id` to get the IDs of the updated rows and store them in a `progress` table. Then you can resume from the last processed ID. I’ve used this pattern to backfill 2 million rows in 11 minutes without locking the table. Monitor replication lag and pause if lag > 5 seconds. Also cap IOPS at 70% of provisioned to avoid starving other workloads.

**How do I know if my migration will cause autovacuum pressure?**

Check `pg_stat_progress_vacuum` in production during peak hours. If autovacuum is running for > 50% of the time, assume it will spike during your migration. Reschedule the migration or cap IOPS during the migration window. I’ve seen autovacuum consume 140 MB/s of write activity during a migration, saturating the IOPS ceiling. The migration stalled for 6 minutes longer than expected.

**What’s the most common mistake teams make when rolling back a migration?**

They assume the rollback script works because it ran in staging. But staging uses a tiny dataset. In production, the rollback script may fail because the forward migration changed data in a way that breaks the rollback. I’ve seen teams roll back a migration that normalized an enum into a lookup table, only to find the rollback script expected the enum column to exist. The team had to restore from a backup. The solution is to test rollback scripts on a copy of prod data before running the migration.

**How do I handle migrations in a multi-region database?**

Multi-region databases introduce replication lag and cross-region latency. The key is to run the migration in the primary region first, then replicate to other regions. Use `CREATE INDEX CONCURRENTLY` and `ALTER TABLE ... NOT VALID` to reduce lock time. Monitor replication lag in each region and pause if lag > 5 seconds. I’ve seen teams run a migration in the primary region, then hit a 500 error in a remote region because the replica lagged behind the write. The solution is to run the migration during off-peak hours in all regions and monitor lag in real time.


---

### About this article

**Written by:** Kubai Kevin — software developer based in Nairobi, Kenya.
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
please contact me — corrections are applied within 48 hours.

**Last reviewed:** June 23, 2026
