# Double-entry ledger: concurrent write errors

A double-entry ledger has one job: every transaction's debits equal its credits, and the sum of all account balances stays constant. That invariant holds trivially in a single-threaded test and fails in production the moment two requests touch the same account at the same time. This article covers why the invariant breaks, how to enforce it at the database level, how to measure whether your fix actually works, and how to keep it from regressing.

## The error and why it's confusing

A ledger schema typically looks like this: a `transactions` table, an `entries` table with one row per debit or credit, and an `accounts` table with a balance column. Each transaction inserts two or more entries summing to zero and updates the affected account balances.

Under normal load this works. Under concurrency, teams see one of several failure modes:

- `ERROR: could not serialize access due to concurrent update`
- `ERROR: deadlock detected`
- `Duplicate key value violates unique constraint "entries_transaction_id_account_id_key"`
- No error at all, but the trial balance drifts by small amounts over thousands of transactions.

The last case is the dangerous one. There is no exception to alert on, no failed request to retry, and no obvious moment when the books went wrong. The drift is discovered weeks later during reconciliation, by which point the audit trail is cold.

The reason these errors don't appear in development is that development rarely reproduces the interleaving that causes them. Two requests must overlap in a specific window: both read state, both decide their write is valid, both commit. A single-threaded test or a low-traffic staging environment almost never produces that window.

The root cause is that double-entry accounting requires atomicity across multiple rows, and most databases do not provide that automatically. You must enforce the invariant at the database level, not just in application code. The specific trap is that the default isolation level in PostgreSQL, `READ COMMITTED`, does not prevent write skew or lost updates on aggregate reads.

## What's actually causing it

The surface symptom is a failed transaction or a drifted balance. The underlying reason is that a multi-row invariant is being treated as if it were a single-row update.

The invariant "sum of debits equals sum of credits for this transaction" spans multiple rows, often across different accounts. Under concurrent writes, two transactions can interleave in ways that break the invariant even though each transaction individually preserves it. This is write skew.

A concrete example:

1. Transaction A reads account X (balance 100) and account Y (balance 100).
2. Transaction B reads account X (balance 100) and account Y (balance 100).
3. A inserts a debit of 50 to X and a credit of 50 to Y, then updates balances.
4. B inserts a debit of 50 to Y and a credit of 50 to X, then updates balances.
5. Both commit.

Under `READ COMMITTED`, each transaction saw a consistent snapshot at read time and each write is individually valid. The final state depends on which updates landed last, and the aggregate may not reflect either intended transfer. If a cached balance column is also being updated, you get lost updates on top of write skew.

Three recurring causes:

**Cause 1: relying on application-level checks under `READ COMMITTED`.** Two transactions both read the same balance, both decide it is sufficient, and both proceed. The result is an overdraft or a double-spend that no constraint catches.

**Cause 2: misusing `SELECT ... FOR UPDATE`.** Locking account rows in a different order in different transactions produces deadlocks. Not locking them at all produces lost updates. Locking them but reading the balance from a cache produces inconsistency anyway.

**Cause 3: derived tables updated outside the main transaction.** Triggers or application hooks that update a monthly summary or running balance outside the transaction that wrote the entries will drift under concurrency, because the summary update can commit or fail independently.

The fix in every case is to make the database enforce the invariant, either through isolation level, explicit locking, or constraints.

## Fix 1: SERIALIZABLE isolation with retries

**Symptom:** PostgreSQL logs show `could not serialize access due to concurrent update` or `deadlock detected`. Errors spike under peak traffic. Application-level retries sometimes succeed and sometimes don't.

**Cause:** You are using `READ COMMITTED` (the PostgreSQL default) and relying on application-level checks to maintain the ledger invariant.

**Fix:** Use `SERIALIZABLE` isolation for transactions that touch the ledger, and retry the whole transaction on serialization failure.

```sql
BEGIN ISOLATION LEVEL SERIALIZABLE;
-- ledger operations here
COMMIT;
```

In application code, catch SQLSTATE `40001` and retry the entire transaction:

```python
import psycopg2
from psycopg2 import errors
import time

def transfer(conn, from_account, to_account, amount, transaction_id):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with conn.cursor() as cur:
                cur.execute("BEGIN ISOLATION LEVEL SERIALIZABLE")
                cur.execute("SELECT balance FROM accounts WHERE id = %s", (from_account,))
                balance = cur.fetchone()[0]
                if balance < amount:
                    raise ValueError("Insufficient funds")
                cur.execute(
                    "INSERT INTO entries (transaction_id, account_id, amount) VALUES (%s, %s, %s)",
                    (transaction_id, from_account, -amount),
                )
                cur.execute(
                    "INSERT INTO entries (transaction_id, account_id, amount) VALUES (%s, %s, %s)",
                    (transaction_id, to_account, amount),
                )
                cur.execute(
                    "UPDATE accounts SET balance = balance - %s WHERE id = %s",
                    (amount, from_account),
                )
                cur.execute(
                    "UPDATE accounts SET balance = balance + %s WHERE id = %s",
                    (amount, to_account),
                )
                conn.commit()
                return
        except errors.SerializationFailure:
            conn.rollback()
            if attempt == max_retries - 1:
                raise
            time.sleep(0.1 * (2 ** attempt))
```

Note that the retry wraps the entire transaction, including the balance read. Retrying only the failed statement is not sufficient: the read that informed the decision must be re-executed against a fresh snapshot.

`SERIALIZABLE` in PostgreSQL 14 and later uses Serializable Snapshot Isolation (SSI), which detects write skew and other anomalies. It is not free. Expect throughput overhead and a nonzero serialization failure rate under contention. The overhead is workload-dependent, so measure it rather than assuming a number.

**How to measure the retry rate.** Instrument your transfer function to count attempts and failures, and export the ratio as a metric. Alternatively, query `pg_stat_database` for `xact_rollback` before and after a load test and compute rollbacks as a fraction of total transactions. A small failure rate is normal; a sustained high rate means a hot account is serializing the workload, and the fix is to reduce contention rather than to increase retries.

## Fix 2: deferred constraint triggers

**Symptom:** No database errors, but the trial balance drifts over time. The discrepancy is small and hard to reproduce, sometimes appearing only after weeks of running.

**Cause:** Entries are written in separate transactions, or derived tables are updated outside the main transaction, or a retry partially succeeded and left the ledger in an unbalanced state.

**Fix:** Ensure all writes that affect the invariant happen in one transaction, and add a deferred constraint trigger that verifies the per-transaction balance at commit time.

```sql
CREATE OR REPLACE FUNCTION check_ledger_balance() RETURNS TRIGGER AS $$
BEGIN
    IF (SELECT SUM(amount) FROM entries WHERE transaction_id = NEW.transaction_id) != 0 THEN
        RAISE EXCEPTION 'Ledger imbalance for transaction %', NEW.transaction_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE CONSTRAINT TRIGGER ledger_balance_check
AFTER INSERT ON entries
DEFERRABLE INITIALLY DEFERRED
FOR EACH ROW
EXECUTE FUNCTION check_ledger_balance();
```

The `DEFERRABLE INITIALLY DEFERRED` clause is what makes this correct. A regular `AFTER INSERT` trigger fires per row and would see only the first entry of a two-entry transaction, which by definition does not sum to zero. A deferred trigger fires at commit time, when all entries for the transaction are visible.

There is a performance cost: the trigger runs a `SUM` over the entries for the transaction at commit. For a two-entry transaction this is cheap, but it is a per-transaction query, so measure it under your own load rather than assuming it is negligible.

Critically, this trigger does not prevent write skew across transactions. It only guarantees each transaction is internally balanced. For cross-transaction consistency you still need `SERIALIZABLE` isolation or explicit locking. The two mechanisms are complementary: the trigger catches application bugs, and the isolation level catches concurrency anomalies.

## Fix 3: connection pooling and lock scope

**Symptom:** Everything works in staging, but production shows intermittent `deadlock detected`, `Lock wait timeout exceeded` (MySQL), or `canceling statement due to lock timeout` (PostgreSQL). Errors become more frequent as you scale horizontally.

**Cause:** Three environment-specific issues are common.

First, a connection pooler running in transaction mode may reuse a session for a different transaction. Session-level advisory locks (`pg_advisory_lock`) then leak across transactions or fail to protect what you intended.

Second, managed database services may ship with different default isolation levels than a local install. Do not assume the default matches your staging environment; verify it.

Third, multiple application instances each maintaining an in-memory cache of account balances will serve stale reads, and any decision based on a stale balance is wrong regardless of the database's isolation level.

**Fix:** If you use a connection pooler in transaction mode, use transaction-scoped advisory locks (`pg_advisory_xact_lock`) rather than session-scoped locks. Transaction-scoped locks are released at commit or rollback, so they cannot leak into the next transaction on the same pooled session.

```sql
BEGIN;
SELECT pg_advisory_xact_lock(hashtext('account_' || account_id));
UPDATE accounts SET balance = balance - 100 WHERE id = account_id;
COMMIT;
```

For in-memory caches, either remove them for balances on the critical path or read the balance directly from the database inside the transaction. A distributed cache with strong consistency adds its own failure modes and does not remove the need for a correct transaction boundary.

## Choosing an approach

| Approach | Isolation | Concurrency behavior | Complexity | Fits |
|---|---|---|---|---|
| SERIALIZABLE + retry | Serializable | Retries on conflict | Low | General ledgers, moderate contention |
| SELECT ... FOR UPDATE | Read Committed | Blocking; deadlock risk if lock order varies | Medium | Explicit control over hot rows |
| Advisory locks | Read Committed | Blocking; scoped to transaction | Medium | Serializing a specific resource |
| Optimistic version column | Read Committed | Retries on version mismatch | High | High contention, distributed writers |

Two practical notes. First, any blocking approach requires a consistent lock order, usually by sorting account IDs before locking, or you trade lost updates for deadlocks. Second, retry-based approaches require idempotency, because a retry after a partial commit can duplicate an entry.

## Idempotency: required by every retry strategy

Every approach above retries transactions. A retry is only safe if applying the same transaction twice has the same effect as applying it once. The standard pattern is a `processed_transactions` table with a unique constraint on the transaction ID:

```sql
CREATE TABLE processed_transactions (
    transaction_id UUID PRIMARY KEY,
    processed_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

Insert into this table as the first statement of the transaction. If the insert fails with a duplicate key error, the transaction has already been applied; skip it and return success. Because the insert and the ledger writes share a transaction, a rolled-back attempt leaves no trace and can be retried safely.

Without this, a serialization failure that occurs after the entries are written but before commit is fine (the rollback removes them), but a network failure after commit and before the client receives the acknowledgement is not: the client retries and the entries are applied twice.

## How to verify the fix under load

You cannot verify a concurrency fix by reading the code. You have to run concurrent writers and check the invariant afterward.

**Step 1: build a concurrency test.** Spawn many clients performing random transfers between a fixed set of accounts. `pgbench` with a custom script works, as does a script using a thread pool. The essential property is that multiple clients hit the same accounts simultaneously, because disjoint accounts never contend.

**Step 2: instrument the failure modes.** Count serialization failures and deadlocks per attempt. In PostgreSQL, `pg_stat_database.xact_rollback` gives a rollback count; compare it before and after the test run and divide by total transactions to get a rollback rate. Export retry attempts from the application as a counter.

**Step 3: check the invariant after the run.** Both of these queries should return zero rows:

```sql
-- Imbalanced transactions
SELECT transaction_id, SUM(amount) AS total
FROM entries
GROUP BY transaction_id
HAVING SUM(amount) != 0;
```

```sql
-- Total balance should equal the starting total
SELECT SUM(balance) FROM accounts;
```

**Step 4: test partial failure.** Kill a connection mid-transaction and confirm the transaction rolls back cleanly and the ledger remains balanced. A proxy that can inject network failures makes this reproducible.

**Step 5: measure throughput.** Record transactions per second and the retry rate at the same time. A fix that eliminates drift but drops throughput by an order of magnitude may not be acceptable, and the only way to know the trade-off is to measure both numbers under the same load.

## Preventing regression

- **Enforce the invariant in the database, not only in code.** The deferred trigger above catches unbalanced transactions at commit. Application-level assertions can be bypassed by a second service, a migration script, or a manual fix.
- **Make retries idempotent.** The `processed_transactions` table is a small cost for eliminating double-application.
- **Lock in a deterministic order.** Sort account IDs before any `SELECT ... FOR UPDATE` or advisory lock acquisition. Inconsistent ordering is the usual cause of deadlocks in ledger code.
- **Run a periodic reconciliation.** Compare the sum of entries against the sum of balances on a schedule and alert on any nonzero difference. This catches drift before it compounds.
- **Verify isolation level in every environment.** Managed services and poolers can change what your transaction actually sees. Query the current setting at startup and log it.

## Related errors

- `deadlock detected` — usually inconsistent lock ordering. Lock accounts in a deterministic order.
- `could not serialize access due to read/write dependencies among transactions` — a serialization failure variant. Retry the whole transaction.
- `duplicate key value violates unique constraint` — expected if you retry a partially applied transaction. Use an idempotency key.
- `canceling statement due to lock timeout` — reduce contention or raise `lock_timeout` deliberately.
- `Ledger imbalance` from your own application — add the deferred trigger to catch it at commit rather than at reconciliation.

## When the standard fixes don't work

If `SERIALIZABLE`, explicit locking, and deferred triggers are all in place and inconsistencies remain, look further out:

1. Check for storage or network faults. Silent corruption is rare but real; use your database's integrity-checking tools.
2. Audit for non-transactional writes. If multiple services write to the ledger without a shared transaction, no isolation level will save you. A saga with compensating actions, or a single writer per account, is the structural fix.
3. Reconsider the storage engine. A ledger-specific database that handles the invariant internally is a legitimate option for new systems, though migrating an existing one is a separate project.
4. Bring in a specialist. Complex concurrency bugs are worth an outside pair of eyes before they become an audit finding.

## FAQ

**What isolation level should a double-entry ledger use in PostgreSQL?**
`SERIALIZABLE` for transactions that modify the ledger. `READ COMMITTED` allows two transactions to read the same balance and both proceed, which is exactly the write-skew pattern that breaks the invariant. `SERIALIZABLE` can produce retryable failures, which the application must handle.

**How do I retry without double-spending?**
Use an idempotency key. Insert the transaction ID into a table with a unique constraint as the first statement of the transaction. A duplicate-key error means the work is already done; return success without reapplying.

**Can triggers enforce the double-entry invariant?**
Yes, if they are deferred constraint triggers. A per-row `AFTER INSERT` trigger sees only one entry and cannot check the sum. A deferred trigger runs at commit, when all entries for the transaction are visible. It catches application bugs but does not prevent cross-transaction write skew.

**Why do deadlocks still occur under `SERIALIZABLE`?**
`SERIALIZABLE` prevents write skew; it does not prevent deadlocks. Deadlocks come from inconsistent lock ordering. Sort the rows you lock and acquire locks in that order.

**How should I test ledger consistency under concurrency?**
Run many concurrent clients performing random transfers between a shared set of accounts, then verify that every transaction's entries sum to zero and that the total of all balances equals the starting total. Measure throughput and retry rate in the same run.

## Your next 30 minutes

Run the imbalance query from the verification section against your current database:

```sql
SELECT transaction_id, SUM(amount) AS total
FROM entries
GROUP BY transaction_id
HAVING SUM(amount) != 0;
```

If it returns rows, you have a live consistency bug and the deferred constraint trigger is the fastest mitigation. If it returns nothing, check `pg_stat_database.xact_rollback` and your application logs for serialization failures and deadlocks. Either result tells you which fix to apply first.
