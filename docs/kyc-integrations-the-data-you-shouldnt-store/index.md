# KYC integrations: the data you shouldn't store

Most KYC integrations are built like any other third-party API call: the backend sends data to the vendor, receives a verdict, and stores the response. That instinct is the source of much avoidable compliance and security burden. For European deployments, the default architecture — your backend calling the vendor directly and persisting the full identity payload — is usually the wrong one.

A more defensible pattern routes verification through a component that never lets raw identity documents or full identity profiles settle in your primary datastore, and treats the vendor's verdict as a derived, time-bounded fact rather than a durable record. If the current integration stores passport scans, selfie images, or full name-plus-address-plus-date-of-birth tuples in your own database "just in case", it is carrying GDPR Article 9-adjacent risk and audit burden that is largely unnecessary — often because the vendor SDK made that the path of least resistance.

## The stance: verification is a data-minimisation problem before it is an API problem

KYC vendors sell identity verification. They do not sell data custody advice, and their SDKs are optimised for successful verification, not for minimising what you keep. A typical integration looks like this: the frontend posts a document image to your API, your API forwards it to the vendor, the vendor returns a JSON blob with extracted fields, confidence scores, document images, and a decision, and your API writes that blob to a `kyc_verifications` table. That table then becomes a permanent, queryable copy of identity documents and personal data, subject to subject access requests, erasure requests, breach notification, and retention limits you now have to enforce yourself.

The more useful question is not "which KYC vendor has the best pass rate" but "what is the minimum set of facts my system needs to retain after the vendor has made a decision". For most products the answer is: a boolean outcome, a timestamp, an expiry, a reference to the vendor's case ID, and the level of assurance achieved. Everything else — the document image, the extracted MRZ, the address history, the liveness score — can stay with the vendor, who is contractually and operationally set up to hold it. Your database should hold a pointer, not a copy.

This is not a theoretical purity argument. It changes the architecture: the verification flow becomes a short-lived, isolated pipeline with its own storage rules, not a feature bolted onto the main user table. The main application database never sees a passport number. And the audit trail is built from events and references, not from retained personal data.

## Why the "store everything for audit" advice falls short

The common advice is that identity data must be retained for audit and AML record-keeping, so the full vendor response is stored. That advice is half right and dangerously incomplete. AML record-keeping obligations are real: under the EU Anti-Money Laundering Directive framework, obliged entities must retain customer due diligence records for a defined period after the end of the business relationship — commonly five years, with member-state variation. An obliged entity cannot simply delete everything after verification.

But here is the distinction that gets lost: the obligation to be able to demonstrate that CDD was performed is not the same as the obligation to store the raw evidence in the primary operational database. The vendor is typically the record-holder for the underlying evidence, and the obligation can be satisfied by retaining a verifiable reference plus the decision metadata, with contractual guarantees that the vendor retains the underlying records for the required period and can produce them on request. Where copies genuinely must be retained, that retention belongs in a separate, access-controlled store with its own retention schedule and deletion jobs — not in the `users` table where every analytics query and every support tool can reach it.

The second half of the common advice — "just encrypt it" — also falls short. Encryption at rest protects against some breach scenarios but does nothing for subject access requests, erasure requests, or the operational fact that a support engineer with database read access can see identity documents. GDPR Article 5(1)(c) data minimisation and Article 5(1)(e) storage limitation are about not holding the data in the first place, not about holding it securely. A field-level encrypted passport number in the main database is still personal data that has to be found, exported, and deleted on request.

## The better mental model: a verification sidecar with a narrow contract

The architecture worth arguing for is a verification sidecar: a small service (or a well-isolated module with its own database schema and credentials) whose only job is to talk to the KYC vendor and emit a minimal, signed result. The main application never calls the vendor directly and never receives raw identity payloads. The sidecar has a narrow contract with the rest of the system: given a user reference and a verification session, return an outcome record containing only what downstream systems need.

A minimal outcome record looks like this:

```json
{
  "user_ref": "u_8f3a...",
  "vendor": "vendor_name",
  "vendor_case_id": "case_123...",
  "outcome": "verified",
  "assurance_level": "document_and_liveness",
  "decided_at": "2026-02-11T09:14:00Z",
  "expires_at": "2027-02-11T00:00:00Z",
  "evidence_retained_by_vendor": true,
  "schema_version": 1
}
```

Note what is absent: name, date of birth, address, document number, document image, extracted fields. If the product genuinely needs the verified legal name — for example to print it on a contract — that is a separate, justified data flow with its own lawful basis and retention rule, and it should be modelled as such rather than smuggled in as a side effect of KYC. The sidecar's database holds the outcome records and an append-only event log of decisions; it does not hold the evidence.

The sidecar pattern also gives you a clean place to enforce the things that are easy to forget when KYC is just another endpoint in a monolith:

- **Network egress control.** Only the sidecar can reach the vendor's API. The main application's egress rules and service credentials do not include the vendor.
- **Retention and deletion.** The sidecar owns the retention schedule for outcome records and the deletion job that runs when a user exercises erasure. Deleting an outcome record does not require touching the vendor's evidence store if the contract says the vendor handles erasure propagation.
- **Audit events.** Decisions are emitted as events (user reference, outcome, timestamp, vendor case reference) to the audit log. The audit log contains no special-category data, so it can be retained and queried more freely.
- **Vendor swap.** Because the contract is a small outcome schema, replacing the vendor is a sidecar change, not a schema migration across the product.

### Where the raw data actually goes

In a web-based flow, the cleanest arrangement is for the document capture and liveness steps to happen directly between the user's browser and the vendor's hosted flow, so the raw images never transit your servers at all. Many vendors support this with a hosted session or a tokenised SDK flow. If the capture must be proxied through your backend — for example because the vendor's flow does not support your region or your CSP — then the proxying component should be the sidecar, streaming the payload to the vendor without writing it to disk, and the sidecar should have no persistent volume for request bodies.

This is where a lot of integrations quietly go wrong: the sidecar is written as a normal web service, request logging is left on, and document images end up in an access log or an error-tracking payload. If one thing sticks from this piece, it should be this: turn off request body logging on the verification path, and check the error tracker's scrubbing rules before shipping.

## Where the common advice is still right

This is not an argument that identity data should never be stored. There are cases where the sidecar-plus-reference model is not sufficient:

- **You are the obliged entity and your regulator expects you to hold the evidence.** In that case, hold it — but in a dedicated evidence store with its own access controls, encryption keys, retention schedule, and deletion tooling, not in the product database. The sidecar pattern still applies; it just gains an evidence store behind it.
- **Offline or low-connectivity verification.** If verification happens where the vendor cannot be reached in real time and the decision must be made locally, data will be held locally. Isolate it and delete it as soon as the decision is recorded and synced.
- **Re-verification and periodic review.** AML frameworks require ongoing monitoring and periodic refresh of CDD. This is an argument for keeping a durable reference and a re-verification schedule, not for keeping the original document image forever.

And the common advice to use a reputable vendor rather than building verification in-house is right. Document forensics, liveness detection, and sanctions screening are specialist problems. The architectural argument here is about what you keep, not about whether to use a vendor.

## A decision rule for what to persist

When a new field arrives from the vendor, the default should be "do not persist". Persist a field only if all three questions can be answered affirmatively:

1. **Does a downstream system need this field to function?** Not "might be useful for support" — actually needs it, and cannot fetch it from the vendor on demand.
2. **Is there a documented legal or contractual basis to retain it, with a defined retention period?** "Audit" is not an answer unless you can name the obligation and the period.
3. **Can you delete it on request without breaking that downstream system?** If deletion would break a process, the field is load-bearing and you need a design that does not depend on it.

A useful corollary: any field that fails these questions but that you still want for analytics should be aggregated or hashed at ingest, not stored raw. Counts of verification outcomes by day and country are useful; a table of verified names is a liability.

| Data item | Default persistence | Where it belongs |
|---|---|---|
| Outcome (verified / rejected / pending) | Yes | Sidecar outcome table |
| Vendor case ID | Yes | Sidecar outcome table |
| Decision timestamp and expiry | Yes | Sidecar outcome table |
| Assurance level | Yes | Sidecar outcome table |
| Verified legal name | Only if a product flow needs it | Separate justified store |
| Document number | No | Vendor only |
| Document image / selfie | No | Vendor only |
| Extracted address | No unless required | Separate justified store |
| Liveness / confidence scores | No | Vendor only |

This table is a starting position, not a universal answer; the point is that every "Yes" should be defensible in writing.

## The integration details that bite

A few concrete failure modes show up repeatedly when teams wire KYC into a European stack.

**Webhook payloads carry more than you think.** Vendor webhooks often include the full decision payload, not just the outcome. If the webhook handler writes the raw body to the event bus or a `webhook_events` table for replay, that recreates the data store the design was trying to avoid. Parse the payload, extract the minimal outcome, and discard the rest — or route the webhook to the sidecar only.

**Idempotency keys and replay.** Verification decisions can be delivered more than once. The sidecar should treat the vendor case ID as the idempotency key and make outcome writes upserts, so a replayed webhook does not create duplicate records or, worse, a second copy of the payload in a dead-letter queue.

**Retention jobs that never run.** A retention policy that exists only in a document is not a control. The sidecar should have a scheduled job that deletes outcome records past their retention period and emits a count to your metrics. If that count is zero for a month, either nothing is expiring (suspicious) or the job is broken.

**Erasure requests that stop at your database.** When a user exercises the right to erasure, the deletion job must cover the sidecar outcome table and must trigger the vendor's erasure process for the evidence they hold. This needs to be in the vendor contract and in the runbook, not discovered during the request.

**Region and transfer.** If the vendor processes data outside the EEA, the transfer mechanism (for example Standard Contractual Clauses with a transfer impact assessment) is part of the integration, and the data protection impact assessment should cover the verification flow specifically. A DPIA that says "we use a KYC vendor" without describing what data leaves, where it goes, and what you retain is not doing the job.

**Logging and error tracking.** As above: request bodies, response bodies, and error contexts on the verification path must be scrubbed. This is the single most common way raw identity data leaks into systems that were never designed to hold it.

### A minimal sidecar handler

The following is a self-contained illustration of the shape, not a production service. It shows the contract: accept a minimal start request, call the vendor, persist only the outcome, and return only the outcome.

```python
# Illustrative only. Assumes a vendor client and a database session.
from datetime import datetime, timedelta, timezone

OUTCOME_FIELDS = {"user_ref", "vendor", "vendor_case_id", "outcome",
                  "assurance_level", "decided_at", "expires_at", "schema_version"}

def record_outcome(db, user_ref: str, vendor_payload: dict) -> dict:
    # Extract only the fields we are allowed to keep. Everything else is dropped.
    decided_at = datetime.now(timezone.utc)
    outcome = {
        "user_ref": user_ref,
        "vendor": "vendor_name",
        "vendor_case_id": vendor_payload["case_id"],
        "outcome": vendor_payload["decision"],
        "assurance_level": vendor_payload.get("assurance", "unknown"),
        "decided_at": decided_at,
        "expires_at": decided_at + timedelta(days=365),
        "schema_version": 1,
    }
    # Upsert on vendor_case_id so replays are harmless.
    db.execute(
        """INSERT INTO kyc_outcomes (user_ref, vendor, vendor_case_id, outcome,
               assurance_level, decided_at, expires_at, schema_version)
           VALUES (:user_ref, :vendor, :vendor_case_id, :outcome,
               :assurance_level, :decided_at, :expires_at, :schema_version)
           ON CONFLICT (vendor_case_id) DO UPDATE SET
               outcome = EXCLUDED.outcome,
               assurance_level = EXCLUDED.assurance_level,
               decided_at = EXCLUDED.decided_at,
               expires_at = EXCLUDED.expires_at""",
        outcome,
    )
    # Emit an audit event with no special-category data.
    db.execute(
        "INSERT INTO audit_events (event_type, subject_ref, occurred_at, detail)"
        " VALUES ('kyc.decision', :user_ref, :decided_at, :outcome)",
        {"user_ref": user_ref, "decided_at": outcome["decided_at"],
         "outcome": outcome["outcome"]},
    )
    db.commit()
    return {k: outcome[k] for k in OUTCOME_FIELDS}
```

The important line is the one that builds `outcome` from a small allow-list. If a new field appears in the vendor payload, it is ignored until someone deliberately adds it and records why. Note also that `decided_at` is computed once and reused, so the stored timestamp and the expiry are consistent, and the audit event uses the same value as the outcome row.

### Worked example: sizing the outcome table

A small amount of arithmetic shows why the outcome-only design is cheap to operate. Assume a product with 50,000 verification attempts per year (an illustrative figure, not a measurement). Each outcome row holds a short user reference, a vendor case ID, an outcome string, an assurance level, two timestamps, and a schema version — on the order of 250 bytes including row overhead. The outcome table therefore grows by roughly:

- 50,000 rows × 250 bytes = 12,500,000 bytes ≈ 12.5 MB per year before indexes.
- With indexes on `user_ref` and `vendor_case_id`, a reasonable planning multiplier is 2–3×, so budget on the order of 25–40 MB per year.

Retaining those rows for the full CDD period (commonly five years) leaves a table in the low hundreds of megabytes at this volume — trivial for any relational database, and easy to back up, replicate, and audit. The same 50,000 attempts, if each stored a document image of even 500 KB, would be 25 GB per year of special-category data before any extracted fields, all of it in scope for access requests, erasure, and breach notification. The arithmetic is the argument: the minimal record is operationally negligible, and the full payload is not.

## How to verify the design holds

Claims about data minimisation are only worth what you can measure. There are a few concrete checks that turn the design into something testable.

**Audit the schema, not the intent.** Query the information schema for every table in the application and sidecar databases and list columns whose names or types suggest identity data (document, passport, selfie, image, address, date_of_birth, name). Compare that list against the allow-list in the sidecar. Any column that is not on the allow-list is either a bug or an undocumented exception.

**Instrument the payload path.** Add a counter or log line that records the size and field names (not values) of vendor responses at each point they are handled. If a response body is written anywhere other than the allow-list extraction, the field-name list will show it. In staging, run a verification with a synthetic identity and then grep the application logs, error tracker, and database for the synthetic document number. If it appears anywhere outside the vendor's own systems, the path is leaking.

**Test deletion end to end.** Create a synthetic user, run a verification, issue an erasure request through the normal process, and then check every store — application database, sidecar outcome table, audit log, backups where feasible, and the vendor's own records via their API or support process. The test is only passed when the outcome record is gone or anonymised and the vendor confirms erasure of the evidence.

**Watch the retention job's output.** As noted above, the deletion job should emit a count of records removed. Alert if the count stays at zero beyond the expected cadence, and alert if it spikes, which can indicate a misconfigured retention period.

**Compare against the contract.** Read the vendor agreement's data retention, deletion, and sub-processor clauses and check that they match what the architecture assumes. If the design assumes the vendor retains evidence for five years but the contract says ninety days, the design is wrong.

## What to check in the next 30 minutes

Open the verification integration's webhook handler and the sidecar's request logging configuration, and search the codebase for every place the vendor's response body is written to a table, a log, or an error tracker. If there is a `raw_response` column or a `logger.info(response.json())` on the verification path, that is the first fix: replace it with an allow-list extraction like the one above, and add a scheduled deletion job for any historical rows that already contain raw identity data. That single change moves the system from "identity documents are stored and we hope" to "a decision and a pointer are stored", which is the architectural position every other compliance control depends on.
