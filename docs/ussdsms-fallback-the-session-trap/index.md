# USSD/SMS fallback: the session trap

USSD and SMS fallback flows are the part of a payments stack that looks trivial until it isn't. A common misconception is to treat USSD as "a form that works on any phone" and SMS as "a notification channel," then discover in production that USSD is a stateful, timeout-bound, per-session protocol with no delivery guarantee, and SMS is a store-and-forward channel whose only reliable signal is a delivery receipt that may never arrive. The result is duplicated debits, orphaned sessions, and users who see "Transaction failed" while the money has already left their account. This article explains the underlying model, shows a worked fallback state machine, and ends with one file to check in the next 30 minutes.

## The misconception: USSD is not a form, SMS is not a channel

The mental model most engineers arrive with is wrong in two directions.

USSD (Unstructured Supplementary Service Data, specified in the GSM 02.90 family of standards) is not a stateless request/response like HTTP. When a user dials a short code such as `*123#`, the handset opens a session to the mobile network operator's USSD gateway. That gateway holds the session open while your application — usually reached over a gateway-specific HTTP or SMPP bridge — sends menus and receives responses. The session has a lifetime. On most networks it is short: tens of seconds between user inputs is the practical ceiling, and the operator terminates the session if the user does not respond. When the session ends, so does every variable held in process memory for that user. There is no cookie, no session ID you control, no reconnect.

SMS, by contrast, is store-and-forward. Your application hands a message to an SMSC (Short Message Service Centre) via SMPP or an HTTP API. The SMSC queues it, tries to deliver it, and may retry for hours or days. You get a `submit_sm_resp` acknowledging that the SMSC accepted the message — not that the handset received it. A delivery receipt (`deliver_sm` with an `esm_class` indicating a receipt) may arrive, may arrive late, or may never arrive. Treating "accepted by SMSC" as "delivered" is the root of a large class of support tickets.

The practical consequence: a fallback flow that says "if USSD fails, send an SMS" is under-specified. Fails how? At which step? With what state? If the USSD session dropped after the user entered a PIN but before the debit was confirmed, an SMS saying "your transaction failed, try again" invites a second debit. If the debit succeeded but the confirmation SMS was lost, the user calls support convinced they were robbed. Both are the same bug: the fallback path has no shared source of truth with the primary path.

## The underlying model: sessions, transactions, and messages are three different things

Separate three lifecycles that teams routinely conflate:

1. **The USSD session.** Owned by the operator gateway. Short-lived. Identified by a session ID the gateway assigns (in SMPP-based USSD, the `session_id` in the `ussd_request`/`ussd_response` payloads). Every inbound USSD request must be treated as potentially the first, middle, or last of a conversation, and state keyed on that session ID must be persisted outside process memory.
2. **The financial transaction.** Owned by your ledger. Long-lived. Must be idempotent and must survive the death of any USSD session. This is the record the user actually cares about.
3. **The message.** Owned by the SMSC. Fire-and-forget with an unreliable receipt. Message delivery must never be used as a proxy for transaction state.

The correct architecture makes the transaction the source of truth and treats both USSD and SMS as unreliable transports that read from and write to it. A USSD menu step is a command against the transaction state machine. An SMS is a projection of the transaction state at send time, with a receipt that is advisory only.

A second model point: USSD is synchronous from the user's perspective but asynchronous from the backend's. The user waits, staring at a menu, while the backend does whatever it does. If the backend takes longer than the operator's gateway timeout — commonly a few seconds; the exact value is operator-specific and should be confirmed in the gateway's integration documentation — the session drops and the user sees a generic error. Long-running work such as a core-banking call, a fraud check, or a third-party wallet debit must not block the USSD response. The pattern is: acknowledge the step, persist intent, respond to the user with "processing," and complete asynchronously.

## A worked fallback state machine

The following is a minimal, self-contained example of the pattern. It is illustrative Python using only the standard library, not a production integration — the point is the state model, not the transport.

```python
import sqlite3
import uuid
from contextlib import contextmanager

# States for a single transfer attempt.
# INITIATED -> CONFIRMED -> SETTLED  (happy path)
# INITIATED -> FAILED                 (terminal failure)
# INITIATED -> PENDING_RECONCILE      (we lost the session mid-flight)

@contextmanager
def db(path="tx.db"):
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def init():
    with db() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS tx (
                id TEXT PRIMARY KEY,
                msisdn TEXT NOT NULL,
                amount_cents INTEGER NOT NULL,
                state TEXT NOT NULL,
                ussd_session TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS idem (
                key TEXT PRIMARY KEY,
                tx_id TEXT NOT NULL
            )
        """)

def start_transfer(msisdn, amount_cents, idem_key):
    """Idempotent on idem_key. Returns tx_id."""
    with db() as c:
        row = c.execute("SELECT tx_id FROM idem WHERE key=?", (idem_key,)).fetchone()
        if row:
            return row[0]
        tx_id = str(uuid.uuid4())
        c.execute(
            "INSERT INTO tx (id, msisdn, amount_cents, state) VALUES (?,?,?,?)",
            (tx_id, msisdn, amount_cents, "INITIATED"),
        )
        c.execute("INSERT INTO idem (key, tx_id) VALUES (?,?)", (idem_key, tx_id))
        return tx_id

def bind_session(tx_id, session_id):
    with db() as c:
        c.execute("UPDATE tx SET ussd_session=? WHERE id=?", (session_id, tx_id))

def confirm(tx_id):
    with db() as c:
        cur = c.execute(
            "UPDATE tx SET state='CONFIRMED' WHERE id=? AND state='INITIATED'",
            (tx_id,),
        )
        # rowcount == 0 means it was already confirmed or failed: safe to ignore
        return cur.rowcount == 1

def mark_pending_reconcile(tx_id):
    """Call when a USSD session drops after INITIATED but before CONFIRMED."""
    with db() as c:
        c.execute(
            "UPDATE tx SET state='PENDING_RECONCILE' WHERE id=? AND state='INITIATED'",
            (tx_id,),
        )

def reconcile(tx_id, settled):
    """Called by a background job that queries the core ledger."""
    with db() as c:
        target = "SETTLED" if settled else "FAILED"
        c.execute(
            "UPDATE tx SET state=? WHERE id=? AND state IN ('PENDING_RECONCILE','CONFIRMED')",
            (target, tx_id),
        )
```

Two properties matter here. First, `confirm` is guarded by `WHERE state='INITIATED'`, so a duplicate USSD callback (operators do retry) cannot double-confirm. Second, `PENDING_RECONCILE` is a first-class state, not an error. It exists precisely because USSD sessions die mid-flight and the only honest answer is "we don't know yet, go ask the ledger."

The fallback SMS is then a projection of `tx.state`, sent by a worker that reads the table — not by the USSD handler inline. If the SMS is lost, the state is still correct. If the user dials back in, the USSD handler looks up their most recent `PENDING_RECONCILE` transaction and can say "we are checking your last transfer" instead of starting a fresh one.

### Walking through a mid-session drop

Consider a concrete sequence, with the reasoning shown at each step:

1. The user dials `*123#` and selects "send money." The handler calls `start_transfer(msisdn, 5000, idem_key)` where `idem_key` is derived from the MSISDN and a coarse time bucket, or supplied by the gateway if it offers one. State is `INITIATED`, `tx_id` is a fresh UUID.
2. The handler calls `bind_session(tx_id, session_id)` before showing the PIN prompt. The session is now tied to the transaction *before* any credential is entered.
3. The user enters the PIN. The handler validates it, calls `confirm(tx_id)`, and returns "processing" to the gateway immediately. The core-banking debit is dispatched to a queue.
4. The session drops — the user's phone lost signal, or the gateway timed out the next menu. A gateway callback arrives with the session ID and no further input. The handler looks up the transaction by session ID, sees `CONFIRMED`, and does nothing: the debit is already in flight.
5. Alternatively, the session drops *between* step 2 and step 3. The handler sees `INITIATED` and calls `mark_pending_reconcile(tx_id)`. The user sees "session ended"; the transaction is not lost.
6. A scheduled reconcile job picks up every `PENDING_RECONCILE` row older than a few minutes, queries the core ledger by `tx_id` (which must be passed as the ledger's idempotency key), and calls `reconcile(tx_id, settled=<ledger says debited>)`.
7. Once the transaction reaches `SETTLED` or `FAILED`, the SMS worker sends a projection of that terminal state. If the SMS is lost, nothing breaks: the next USSD dial-in or app query reads the same row.

The critical detail in step 6 is that the `tx_id` must be the idempotency key on the *ledger* side too. If the ledger assigns its own reference and your reconcile job matches on amount and MSISDN, two transfers of the same amount to the same number in the same minute will be ambiguous. Pass your UUID through.

### A failure-mode analysis of the same flow

| Failure | What breaks without the model | What the model does |
|---|---|---|
| Session drops after PIN, before confirm | Handler has no record of which transaction the PIN belonged to | `bind_session` ran first; handler finds the row and marks `PENDING_RECONCILE` |
| Gateway retries the confirm callback | Second debit or double-confirm | `WHERE state='INITIATED'` guard makes the second call a no-op |
| Backend slower than gateway timeout | User sees error, backend still debits | Handler responds "processing" before dispatch; state is `CONFIRMED`, not lost |
| SMS confirmation lost | User believes transfer failed, retries | State is authoritative; retry path refuses a new transfer while `PENDING_RECONCILE` exists |
| Reconcile job matches on amount + MSISDN | Wrong transaction settled | Reconcile matches on `tx_id`, which is also the ledger idempotency key |

## Where SMS fallback actually earns its place

SMS fallback is not a substitute for USSD. It is useful in three narrow cases, and teams over-apply it to a fourth where it does harm.

| Fallback scenario | Appropriate? | Why |
|---|---|---|
| USSD session drops after debit; notify user of outcome | Yes | The transaction state is authoritative; the SMS is a projection |
| Smartphone user prefers an app link over USSD | Yes, with caution | Deep-link SMS is a real pattern; the link must not carry a bearer token |
| Feature phone user has no data; send a one-time code for a call-centre flow | Yes | Short, single-purpose, no state carried in the message |
| "If USSD fails, send an SMS asking the user to retry the transfer" | No | Invites duplicate debits; the retry must re-enter the state machine, not start a new one |

The fourth row is the trap. An SMS that says "your transfer failed, dial *123# to retry" is only safe if the retry path first looks up the prior transaction by MSISDN and refuses to start a new one while a `PENDING_RECONCILE` exists for that user. Without that guard, the user dials again, a second `INITIATED` transaction is created, and if the first one later settles there is a double debit and a very unhappy customer.

A related trap: putting sensitive data in the SMS body. SMS is not encrypted end-to-end. It transits SMSCs and aggregators. One-time codes and reference numbers are fine; account balances, full names paired with account numbers, and anything that could be used for social engineering are not. Regulators in most markets treat SMS as an insecure channel, and treating it otherwise is a compliance finding waiting to happen.

## Common misreadings that cause real incidents

**"The SMSC accepted it, so the user got it."** The `submit_sm_resp` is an acknowledgement of receipt by the SMSC, nothing more. Delivery receipts are optional in SMPP and frequently dropped by aggregators. Build the product so that a missing SMS is a degraded experience, not a broken one — the user can always dial back in and check state.

**"USSD menus are stateless, so the PIN can go in step 3."** The PIN is fine in step 3 as long as step 3 is idempotent and the session ID is bound to the transaction before the PIN is requested. The failure mode is binding after: user enters PIN, session drops, the handler never learns which transaction the PIN was for, and there is a credential in the logs with no context. Bind the session to the transaction at the first menu, not the last.

**"A session timeout is an error."** It is a normal outcome. Model it as `PENDING_RECONCILE` and reconcile against the ledger. Treating it as an error leads teams to retry the debit, which is the double-debit bug.

**"The USSD session ID can be the transaction ID."** Session IDs are operator-assigned and may be reused or collide across operators. Use your own UUID and store the session ID as a foreign key, as in the example above.

**"SMS delivery receipts close the loop on confirmation."** They indicate that the message reached the handset, not that the user read it or acted on it. For anything that matters financially, the loop closes when the user re-enters USSD or the app and the state is queried, not when a receipt arrives.

## Measuring whether the fallback actually works

Claims about fallback reliability are only credible if they are instrumented. The following are the metrics worth emitting, and how to compare them before and after a change:

- **Reconcile backlog age.** Count `PENDING_RECONCILE` rows by age bucket (under 1 minute, 1–5, 5–30, over 30). A healthy system drains the under-5 buckets continuously; a growing over-30 bucket means the reconcile job is broken or the ledger query is timing out.
- **Reconcile outcome ratio.** Of transactions that entered `PENDING_RECONCILE` in a window, what fraction settled versus failed versus are still open after the reconcile window? An "open after window" count above zero is a bug, not a statistic.
- **Duplicate-attempt rate.** Count `start_transfer` calls that returned an existing `tx_id` because the idempotency key matched. A rising rate suggests users are retrying because they are not getting confirmation — a UX signal hiding in the data.
- **SMS projection lag.** Time between a transaction reaching a terminal state and its SMS being submitted to the SMSC. This is a queue-depth metric, not a delivery metric.
- **Delivery-receipt arrival rate**, tracked only as an advisory number. Because receipts are optional, a drop here is not proof of a delivery failure; correlate with support contacts before drawing conclusions.

To measure these without a metrics stack, add a single query to the reconcile job that logs the age buckets to stdout once per run, and diff the output across deploys. To measure them properly, emit counters from the reconcile job and the SMS worker and chart them on the same dashboard. Either way, the comparison is before-versus-after a change, not against an industry figure.

## What to check first: the reconcile job

The single highest-leverage thing to verify in the next 30 minutes is whether the system has a reconcile job for `PENDING_RECONCILE` (or whatever the equivalent is called) transactions, and whether it runs on a schedule that can be named. Open the file or job definition that owns transaction state transitions — commonly a worker module, a cron entry, or a scheduled workflow — and confirm three things: it queries the core ledger (not the USSD gateway) for ground truth; it transitions `PENDING_RECONCILE` to a terminal state; and it emits a metric for how many transactions are older than, say, ten minutes in that state. If that metric does not exist, add it before touching anything else. A payments system without a reconcile job is one dropped USSD session away from a support queue it cannot clear.

## Frequently Asked Questions

**How long does a USSD session stay open?**
It varies by operator and gateway, but the practical window between user inputs is short — on the order of tens of seconds, and the session terminates if the user does not respond. The exact timeout is operator-specific and should be confirmed in the gateway integration documentation rather than assumed. Design so that no step depends on the session surviving a slow backend call.

**Why does a USSD flow show "Transaction failed" but the money left the account?**
Because the USSD response and the ledger debit are not atomic. The debit may succeed after the session has already dropped, or the gateway may time out the response while the backend continues processing. The fix is to make the transaction state authoritative, record `PENDING_RECONCILE` when the session dies, and have a reconcile job query the ledger. The user-facing message should say "we are checking," not "failed."

**Can SMS be the primary channel instead of USSD?**
For some flows, yes — one-time codes and simple notifications are fine over SMS. But SMS is fire-and-forget with unreliable delivery receipts, so it cannot carry a multi-step interactive flow. USSD is the right primary channel for feature-phone users because it is interactive and session-bound; SMS is the fallback for outcomes, not for steps.

**What is the difference between `submit_sm_resp` and a delivery receipt in SMPP?**
`submit_sm_resp` is the SMSC acknowledging that it received your `submit_sm` request — it says nothing about the handset. A delivery receipt is a separate `deliver_sm` message, sent later and optionally, indicating the final status at the handset. Treating the first as the second is a common cause of "we told the user, but they never got it" incidents.

**Should the USSD handler ever call the core ledger synchronously?**
Only if the operator's gateway timeout comfortably exceeds the ledger's worst-case latency, which is rarely the case and is not something to assume. The safer default is to persist intent, respond immediately, and let a worker complete the call. If a synchronous call is unavoidable for a specific step, measure the ledger's latency distribution first and set an explicit timeout below the gateway's documented limit.

## Action for the next 30 minutes

Open the file that owns transaction state transitions and add (or verify) one query: the count of transactions stuck in the pending-reconcile state for longer than ten minutes, logged once per reconcile run. If the state or the job does not exist, that absence is the finding — create the state and a scheduled job that queries the ledger by your own transaction ID before anything else ships.
