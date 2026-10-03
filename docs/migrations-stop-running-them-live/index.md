# Migrations? Stop running them live

## The conventional wisdom (and why it's incomplete)

The standard playbook for schema changes in a live system looks like this: treat the database schema as a deployable artifact, run migrations in production while the application is live, use a migration tool that tracks applied changes, wrap statements in transactional DDL where the engine supports it, add a retry loop, test in staging, and ship the migration alongside the next release. This is the textbook approach taught in most deployment guides.

The problem is that this playbook assumes the schema is a deterministic artifact you can patch live. In practice, schema changes leak state across deployments. A column added in v3.2 is still there when v4.5 deploys. A default value set in a migration can be overridden by application code. A foreign key added in a background job can block a foreground query during peak traffic. Live schema migrations are not "eventually consistent" with the application's expectations — they are eventually *inconsistent* unless the application and schema are coordinated.

The root cause is rarely the migration tool. It is that the migration script and the application code are versioned independently. A migration can succeed at the database layer while failing at the application layer, because the deployed code expects a different schema shape than the one that now exists.

## What actually happens when you follow the standard advice

Consider a migration that adds a `NOT NULL` column to a `users` table:

```sql
ALTER TABLE users ADD COLUMN mfa_required BOOLEAN NOT NULL DEFAULT FALSE;
```

On a small table this may finish in milliseconds. On a large table, the same statement can take much longer depending on the engine, the version, and whether the default requires a table rewrite. If the migration runs during peak traffic and acquires a lock that blocks authentication queries, p95 latency on `/login` can climb from tens of milliseconds to seconds. Users on high-latency connections time out first.

Rolling back is not always simple. If the `ALTER` already committed, the rollback is another migration that drops the column. But if the application code in the new release already references `user.mfa_required`, the rollback breaks the new code path. The result is a partial state: schema reverted, code not reverted, or vice versa.

Even when the migration is reversible, the state it creates lingers. A soft-deleted column is not truly deleted; it is hidden behind a view or a feature flag. That hidden state becomes a landmine when you try to change the schema again. Schema-diffing tools can detect drift, but they do not solve the coordination problem between the migration and the runtime behavior.

## A different mental model: expand and contract

Instead of treating the migration as a patch, treat it as a synchronization step between two independently evolving artifacts: the application code and the database schema. The key insight is that the schema must always be a valid superset of what every deployed version of the application expects. That means you cannot add a `NOT NULL` column and fill it with data in one step if older clients still read the table.

The standard pattern is often called **expand and contract**:

1. **Expand:** add the new column as nullable, or add the new table, without changing existing behavior.
2. **Migrate data:** backfill the new column or table in batches, independent of the schema change.
3. **Dual-write / dual-read:** deploy code that writes to both old and new structures and can read from either.
4. **Contract:** once all clients use the new structure, drop the old column or table and remove the dual-write code.

Each step is independently deployable and independently reversible. No single step requires the application and the schema to change atomically.

A useful way to enforce this is to model the database as a versioned artifact alongside the application. A release bundle can include the application artifact, the migration script, and a schema version manifest. The CI pipeline runs the migration against a snapshot of production, then deploys the application only after the schema version matches the expected version. A lock record — in object storage, a configuration table, or a dedicated lock service — records the current schema version. If the migration fails, the lock is not updated, so the next deploy still sees the previous version.

A minimal lock implementation in Go might look like this:

```go
package main

import (
	"context"
	"log"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/s3"
)

type SchemaLock struct {
	Version  int
	Checksum string
	LockedAt string
}

func Acquire(ctx context.Context, client *s3.Client, bucket, key, body string) error {
	_, err := client.PutObject(ctx, &s3.PutObjectInput{
		Bucket: aws.String(bucket),
		Key:    aws.String(key),
		Body:   []byte(body),
	})
	if err != nil {
		return err
	}
	log.Println("Schema lock updated")
	return nil
}

func main() {
	ctx := context.Background()
	cfg, err := config.LoadDefaultConfig(ctx)
	if err != nil {
		log.Fatal(err)
	}
	client := s3.NewFromConfig(cfg)
	if err := Acquire(ctx, client, "deploy-locks", "schema.json", `{"version":43,"checksum":"sha256:..."}`); err != nil {
		log.Fatal(err)
	}
}
```

A background reconciliation job can compare the actual schema against the expected version and emit an alert if drift is detected. Such a job should be read-only: it reports drift, it does not commit changes. Running it every 30 seconds is cheap and catches accidental manual changes.

## How to measure whether this is working

Claims about migration safety are only meaningful if you instrument them. Before adopting any change, record a baseline:

- **Migration duration:** log start and end timestamps per migration. Compare the p50 and p99 across releases.
- **Lock wait time:** on PostgreSQL, query `pg_locks` and `pg_stat_activity` during migrations, or enable `log_lock_waits` and set `deadlock_timeout` to a low value in staging. On MySQL, check `performance_schema.data_lock_waits`.
- **Deployment rollback rate:** count how many releases require a rollback, and classify the cause (schema, code, configuration).
- **Application error rate during deploy:** compare error rates in the 10 minutes before and after each deploy, segmented by endpoint.
- **Replication lag:** on replicas, monitor `pg_last_wal_replay_lsn` or the equivalent. A migration that causes multi-second lag can break downstream consumers.

None of these require a special tool. They require that migration steps are logged with a correlation ID that ties them to a release, and that the logs are queryable after the fact.

## The cases where live migrations are reasonable

There are scenarios where live migrations are acceptable, even optimal.

For read-heavy systems with low write concurrency, adding a nullable column with a default value is usually safe, provided the engine does not rewrite the table. A column that is only written by a background job and read by a new UI version does not affect existing queries.

When you control both the application and the database versions tightly — for example, a backend and client that release in lockstep — you can sometimes ship a schema migration that assumes all clients are updated. This works less well in regulated industries where audit trails require backward compatibility, or where clients cannot be forced to upgrade.

For analytics workloads where the database is read-only from the application layer, schema migrations can often run live without risk. Even here, a schema version file helps downstream BI tools and dashboards detect breaking changes such as a renamed table.

## A decision checklist

Use this checklist to decide whether a given migration can run live or needs the expand/contract treatment. It is based on the risk surface, not the size of the change.

| Factor | Question | Higher risk when |
|---|---|---|
| Write concurrency | How many writes per second hit the affected table at peak? | Thousands of writes/sec |
| Data volume | How large is the table, and does the engine rewrite it? | Large tables with rewrites |
| Regulatory scope | Are audit trails or PII involved? | Health, finance, PII |
| Client diversity | How many client versions read this table? | Mobile + web + legacy APIs |
| Reversibility | Can the change be undone in under five minutes? | No, or unknown |
| Lock behavior | Does the statement acquire a table lock? | Yes, during peak traffic |

If several factors are high, use expand/contract. If most are low, a live migration may be fine. The checklist is a prompt for discussion, not a scoring formula — the weights depend on your system.

## Objections and responses

**"Versioned schemas slow us down."**

Not if the diff step is automated. A CI job that compares the repository's expected schema against a read replica of production can run in parallel with unit tests. The cost is one job; the benefit is catching drift before it reaches production.

**"We don't have time to write a schema version file."**

If you can write a migration script, you can write a schema version file. It can be a single JSON line: `{"version":43,"checksum":"sha256:..."}`. Store it in the repository alongside the migration. The setup cost is small; the value is that the deploy pipeline has a single source of truth for the expected schema.

**"Our database is Postgres; we can do everything in transactions."**

PostgreSQL supports transactional DDL for many statements, but not all. `ALTER TABLE ADD COLUMN` is transactional in modern versions, but `CREATE INDEX CONCURRENTLY` cannot run inside a transaction block. Even within transactional DDL, long-running transactions bloat the WAL and can cause replication lag. Keeping migrations short and validation separate avoids this.

**"We use Flyway (or Liquibase) and it works fine."**

Migration tools are good at applying repeatable, ordered changes. They do not, by themselves, coordinate the migration's success with the application's expectations. A migration can succeed in the tool and still fail in production because the deployed code expects a different schema. The versioned model decouples the two. The tool can still be used for repeatable migrations; the versioned manifest wraps the deployment pipeline around it.

## Practical steps and one action for the next 30 minutes

If you are starting fresh, three principles are worth adopting:

1. Every schema change should be reversible in under five minutes, or explicitly marked irreversible.
2. The database schema should be versioned and diffed before every deploy.
3. Application code should handle schema versions older than itself, at least during the expand phase.

To implement this, you can use a schema-diffing tool that supports your engine, a CI job that runs it against a read replica, a lock record in a highly available store, and a read-only reconciliation job. Add a test suite that spins up a temporary database container, applies the migration, and runs queries simulating old and new client behavior. Enforce a policy that no migration runs longer than a fixed threshold in production; if it does, split it into smaller steps and move data manipulation into a separate job.

**Action for the next 30 minutes:** create a file named `schema-version.json` in your repository containing a single field, `version`, set to the current schema version. Add a CI step that fails the build if the file's version does not match the version recorded in your migration tool's history table. This does not require any new infrastructure, and it gives you a single place to detect schema drift before it reaches production.

## FAQ

**How do you safely add a NOT NULL column without downtime?**

Add the column as nullable first. Deploy that change. Backfill existing rows in batches with a background job. Once the backfill is complete and all clients write a value, add the `NOT NULL` constraint in a separate migration. On PostgreSQL, adding a `NOT NULL` constraint with a default can avoid a full table rewrite in recent versions; verify the behavior for your engine and version before relying on it.

**What is the safest way to rename a column in production?**

Use a three-step process: add the new column, backfill it, update the application to write to both and read from both, then drop the old column once no code references it. During the transition, the application should tolerate either column being authoritative. This eliminates the race where a query hits the new column before it is populated.

**How do you roll back a failed migration in a zero-downtime system?**

If the migration already committed, the rollback is another migration that reverses the change. If the application code has already deployed, you also need to disable the new code path, typically with a feature flag. The safest rollback is to deploy the previous application version and run the reverse migration, in that order or the reverse depending on which direction is safe for your schema.

**Why do online schema change tools still cause issues?**

Tools that create a shadow table and swap it in can still block queries during the swap phase. In high-concurrency systems, the swap can take seconds, during which locks accumulate and tail latency spikes. The expand/contract model avoids this by decoupling schema changes from runtime behavior, so no single operation needs to swap a table under load.
