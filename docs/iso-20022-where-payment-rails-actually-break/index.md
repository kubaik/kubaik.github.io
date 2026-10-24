# ISO 20022: where payment rails actually break

The failure is quiet — no errors, just wrong answers. A one-line change buried in a typical ISO 20022 mapping is not flagged until it is expensive. This is the version that skips the happy-path tutorial.

## The problem, in general terms

ISO 20022 is not a new SWIFT message format. It is a syntax and dictionary standard — XML (and now also a JSON variant) built on a shared data model — that replaces a set of fixed-format messages that grew up separately in different clearing systems. The migration is usually described as "MT to MX", which makes it sound like a serialisation change. It is closer to a schema migration with a governance layer bolted on: every field that used to be positional now has a name, a type, a cardinality, and a set of allowed code values, and those code values are maintained by an external body on its own release cycle.

The real problem teams hit is not parsing XML. It is that ISO 20022 makes implicit contracts explicit, and most payment rails have decades of implicit behaviour that nobody wrote down. A field that was "always populated by the sending bank" becomes an optional element with a defined meaning. A free-text remittance line becomes structured `RemittanceInformation` with a maximum of 140 characters per `Unstructured` entry. A charge bearer that was inferred from the message type becomes an explicit `ChrgBr` code. When a rail moves to the new format, every one of those assumptions has to be either encoded correctly or consciously dropped — and dropping them silently is how reconciliation breaks three days later.

This piece walks through the pattern teams commonly hit when they build or extend a payment rail against ISO 20022, why the obvious approaches fail, and what a workable implementation looks like. It is written from the standard and from publicly documented behaviour of the major clearing systems, not from a single deployment.

## The approaches that commonly fail, and why

**Treating it as a serialiser swap.** The first instinct is to write an MT103-to-pacs.008 mapper and call it done. This fails because the two formats do not have a clean bijection. MT103 has a free-text field 70 (`Remittance Information`) that can carry up to four lines of 35 characters. The ISO equivalent, `RemittanceInformation`, splits into `Unstructured` (max 140 characters) and `Structured` (a `CreditorReferenceInformation` block with a `Type` code and a `Reference`). A mapper that concatenates field 70 into `Unstructured` will pass schema validation and lose the ability to reconcile against structured invoice references. A mapper that tries to parse field 70 into structured form will mis-parse free text that was never structured in the first place. There is no correct automatic answer; the mapping has to be a policy decision per counterparty.

**Validating against the XSD and stopping there.** The ISO 20022 XSDs enforce structure — element order, types, cardinality. They do not enforce the business rules that the clearing system actually applies. A pacs.008 can be schema-valid and still be rejected because `IntrBkSttlmAmt` has more decimal places than the currency allows, or because `ChrgBr` is `DEBT` on a message where the scheme requires `SLEV`, or because a `Cdtr` address is missing a country code that the scheme mandates. Teams that treat XSD validation as the finish line discover the rest of the rules in production rejections.

**Hard-coding the external code lists.** ISO 20022 code lists (the `ExternalCodeSets` — `ExternalPurpose1Code`, `ExternalBankTransactionCode`, etc.) are published and versioned by the registration authority. They change. A `Purp` code that was valid in one release can be deprecated or removed in the next. Teams that inline the code list as an enum in application code end up with a deploy required every time the standard moves, and — worse — with a validation layer that rejects messages from counterparties who have already moved to a newer release. The code list is reference data, not source code.

**Assuming the character set is ASCII.** ISO 20022 messages are XML, and the allowed character set for text fields is constrained by the scheme, not just by XML. The common failure is a name or address containing an ampersand, a non-Latin script, or a character outside the scheme's permitted set. XML escaping handles the ampersand; it does not handle a scheme that restricts `Nm` to a specific Unicode range. This surfaces as a rejection at the clearing system, not at your parser, which makes it hard to catch in unit tests.

## The approach that works in practice

The pattern that holds up is to treat ISO 20022 as three separate layers and keep them separate in code:

1. **A generated model** — classes or types generated from the published XSDs, regenerated when the standard version changes. Never hand-edit these. They exist to guarantee structural correctness and to give you a typed object to work with.
2. **A validation layer** — schema validation plus a separate set of business rules expressed as data, not code. The business rules cover things the XSD cannot: currency-specific decimal limits, scheme-mandated field presence, code-list membership against a versioned reference table.
3. **A mapping layer** — explicit, per-counterparty mapping between your internal domain model and the ISO message. This is where the policy decisions live (how field 70 becomes `RemittanceInformation`, which `ChrgBr` applies, how addresses are normalised).

The key discipline is that layers 1 and 2 are mechanical and versioned; layer 3 is where humans make judgement calls, and it should be visible and reviewable rather than buried in a serialiser.

A worked example makes the layering concrete. Suppose an internal payment record has a free-text `reference` field and a separate `invoice_number`. The mapping to `RemittanceInformation` should be an explicit function with a documented policy:

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class InternalPayment:
    reference: str
    invoice_number: Optional[str]
    amount: str  # decimal string, already currency-validated
    currency: str

def to_remittance_info(p: InternalPayment) -> dict:
    """Map internal reference fields to ISO 20022 RemittanceInformation.

    Policy: if an invoice number exists, emit Structured with a
    CreditorReferenceInformation using the SCOR type code. Otherwise
    emit Unstructured, truncated to the 140-character scheme limit.
    """
    if p.invoice_number:
        return {
            "Structured": {
                "CdtrRefInf": {
                    "Tp": {"CdOrPrtry": {"Cd": "SCOR"}},
                    "Ref": p.invoice_number[:35],
                }
            }
        }
    return {"Unstructured": p.reference[:140]}
```

That function is deliberately boring. The point is that the two policy decisions — prefer structured when possible, truncate at the documented limit — are visible in one place and can be reviewed by someone who understands the reconciliation requirements, not just the XML schema. The truncation is also a decision that should be logged, because silent truncation of a reference is a reconciliation defect waiting to happen.

## A worked mapping decision, with the reasoning shown

Consider a payment where the internal record has `reference = "INV-2024-0042 / PO 8831"` and `invoice_number = None`. The function above emits `Unstructured` with the whole string, because there is no invoice number to build a structured reference from. That is the correct outcome only if the counterparty can reconcile against free text. If the counterparty requires structured references, the payment will be accepted and then sit unmatched.

Now consider the same record with `invoice_number = "PO-8831-2024-000123456789"`, which is 26 characters — under the 35-character limit for `Ref`. The function emits `Structured` with `SCOR` as the type code. But `SCOR` is a specific creditor-reference scheme; using it for a purchase-order number that is not a SCOR reference is a semantic error that no schema validator will catch. The receiving party may parse the reference according to the SCOR rules and fail to match it.

The reasoning to apply is: the choice of `Cd` in `CdOrPrtry` is itself a policy decision, and it must be driven by what the reference actually is, not by what field it happens to live in. A purchase-order number is not a SCOR reference. A more honest mapping either uses a type code that matches the reference's real scheme, or falls back to `Unstructured` and flags the record for manual review. The code change is small; the decision requires someone who knows the commercial context.

The same shape appears with `ChrgBr`. A message type may permit `DEBT`, `CRED`, `SHAR`, `SLEV` and `NONE`, but the commercial agreement with a given counterparty usually permits exactly one. Encoding the agreement as a per-counterparty configuration value — validated against the scheme's allowed set for that message type — keeps the decision visible. Encoding it as a default in the serialiser hides it.

## Implementation details

The details that separate a working rail from one that generates support tickets:

**Version the XSDs and the code lists together.** ISO 20022 releases are named by year and version (for example, the 2019 version, the 2022 version). The XSD and the external code lists move together, but not always on the same day in every clearing system. Store both as versioned artefacts — a directory per release, checksummed — and record which release each outbound message was validated against. When a counterparty reports a rejection, the first question is which release they are on, and you need to be able to answer it from your own records.

**Model the amount as a decimal string, not a float.** `IntrBkSttlmAmt` carries a currency attribute and a decimal value. Floating-point representation will produce values that are schema-valid but wrong at the last decimal place for currencies with non-trivial minor units. Keep amounts as decimal strings or a decimal type end to end, and validate the number of fraction digits against the currency's documented minor-unit count before serialisation.

**Validate character sets at the boundary, not at the parser.** Build a validation rule that checks text fields against the scheme's permitted character set and reports the offending field and character. This is the rule that catches the name with an unusual script or a stray control character before it becomes a clearing rejection. It is cheap to implement and it removes a whole class of production incidents.

**Treat `ChrgBr` and similar enums as business rules, not defaults.** The charge bearer is a commercial term. Defaulting it in code means the default silently applies to counterparties who negotiated something else. Make it a field on the counterparty configuration, validated against the scheme's allowed values for that message type.

**Keep an audit record of the exact bytes sent.** When a message is rejected, the clearing system's reference points to what they received. If your application re-serialises on retry, the bytes can differ — element ordering, optional empty elements, namespace prefixes. Persist the serialised message (or a hash of it) alongside the internal record so that a rejection can be reproduced exactly. This is the single most useful operational habit in the whole migration.

**Test against real counterparty samples, not just the XSD.** The XSD tells you what is structurally valid. Counterparty sample messages tell you what they actually send — which optional fields they populate, which code values they use, how they format addresses. Build a corpus of samples per counterparty and run it through your inbound validation on every change. This catches the case where your validator is stricter than the scheme and rejects valid traffic.

## How to measure whether the rail is actually improving

Claims about rejection rates or straight-through processing are only meaningful if they are measured on your own traffic. The instrumentation is straightforward:

- **Count rejections by layer.** Tag each rejection with the layer that produced it: schema validation, business-rule validation, clearing-system rejection, or downstream reconciliation failure. A migration that is going well moves defects from the first two categories into the last two, where they are visible to the business.
- **Count silent truncations.** Every `[:N]` in a mapping function should increment a counter keyed by field and counterparty. A truncation rate that changes without a corresponding change in input data is a signal that something upstream has shifted.
- **Compare bytes sent against bytes acknowledged.** Store a hash of the serialised message and the clearing system's acknowledgement reference. If a retry produces a different hash for the same logical payment, the serialiser is not deterministic and the audit trail is unreliable.
- **Track unmatched references as a separate metric from rejected messages.** These are different failure modes with different owners. A rejected message is a technical defect; an unmatched reference is a data-quality or policy defect.

Run these as a dashboard before changing anything, so there is a baseline. Without a baseline, any improvement claim is anecdotal.

## What outcomes to expect, and their limits

It is tempting to promise that ISO 20022 reduces rejection rates or improves straight-through processing. The honest position is that the standard enables those outcomes but does not deliver them on its own. Richer structured data — structured remittance, purpose codes, legal entity identifiers — gives reconciliation systems more to match on, but only if the counterparties populate it, and many still send unstructured remittance because their upstream systems cannot produce structured references. A rail that assumes structured remittance will arrive will fail on exactly the counterparties it most needs to serve.

The realistic expectation is a shift in where defects appear. Before the migration, defects concentrate in parsing and field extraction. After a well-executed migration, they concentrate in mapping policy and in counterparty data quality — which is a better place for them to be, because they are visible and reviewable rather than buried in a parser. Teams commonly find that the volume of schema-level rejections falls once validation is layered properly, while the volume of business-level exceptions (unmatched references, missing purpose codes) stays roughly constant until counterparty data improves. That is not a failure of the standard; it is the standard doing its job by making the data quality problem visible.

A second limit: the standard is a moving target. Code lists and message definitions are versioned and revised. A rail built once and not maintained will drift out of conformance. The maintenance cost is real and should be budgeted — not as a one-off migration but as an ongoing reference-data update process.

## What to watch out for

- **Silent truncation.** Any `[:N]` in a mapping function is a decision. Log it, count it, and alert if the rate changes.
- **Validator drift.** If your inbound validator is stricter than the scheme, you will reject valid messages and blame the counterparty. Keep a sample corpus and a conformance test that runs against it.
- **Release skew.** Counterparties and clearing systems do not all move to a new release on the same day. Your validation needs to know which release a message claims to conform to, and apply the matching rules.
- **Namespace and prefix changes.** XML namespace prefixes are semantically irrelevant but operationally significant when you are diffing messages or hashing them. Normalise before hashing, or hash the canonical form.
- **Character set assumptions in downstream systems.** Even if your ISO layer is correct, a downstream ledger or reporting system may not handle the full permitted character set. The rejection then happens after the payment has left, which is worse.
- **Type-code misuse.** Using a reference type code that does not match the reference's real scheme produces a message that validates and cannot be reconciled. This is the most expensive failure mode because nothing flags it.

## The broader lesson

The principle worth taking away is that format migrations are really contract migrations, and contracts have three parts: syntax, semantics, and governance. The syntax is the easy part and the part tooling solves well — generated models, schema validation, serialisers. The semantics are where the judgement lives, and they belong in explicit, reviewable mapping code rather than in a clever serialiser. The governance is the part teams most often forget: the code lists and message definitions are external reference data with their own release cycle, and a system that treats them as compile-time constants will require a code deploy every time the outside world changes.

This is not unique to ISO 20022. The same shape appears in any standards migration — EDI to API, fixed-width to delimited, proprietary to open. The teams that come through it well are the ones that separate the mechanical layers from the judgement layers and version the external dependencies explicitly. The teams that struggle are the ones that try to make the serialiser smart.

## How to apply this to your situation

Start by auditing where your current implementation puts the judgement. If your MT-to-ISO mapping is a single function or a serialiser configuration, that is the risk. The next 30 minutes: open the file that builds your outbound ISO 20022 message and list every place it makes a policy decision — truncation, default code values, field mapping choices, reference type codes. If any of those decisions are not visible in one place with a comment explaining why, move them into an explicit mapping function and add a log line for each truncation. That single change turns invisible reconciliation risk into something you can measure.

## Frequently Asked Questions

**What is the difference between ISO 20022 and SWIFT MT?**
SWIFT MT is a set of fixed-format messages with positional fields, while ISO 20022 is a syntax and data dictionary standard with named, typed elements and externally maintained code lists. ISO 20022 messages are typically XML (with a JSON variant now defined), and the same data model is used across multiple clearing systems rather than being specific to one. The migration is a change of data model, not just a change of encoding.

**Does ISO 20022 replace SWIFT MT entirely?**
The major clearing systems have been migrating to ISO 20022, and the MT formats are being retired for cross-border payments on a published timetable. The exact end dates differ by scheme and by message type, so the authoritative source is the scheme's own migration documentation rather than a general summary. Teams should track the timetable for each scheme they connect to separately.

**How do I validate an ISO 20022 message beyond the XSD?**
Layer the validation: XSD for structure, then a separate rule set for business constraints that the XSD cannot express — currency-specific decimal limits, scheme-mandated field presence, code-list membership against a versioned reference table, and permitted character sets. Keep the business rules as data so they can be updated when the scheme's rules change without a code deploy.

**Why does my ISO 20022 message pass validation but get rejected by the clearing system?**
Because the clearing system applies business rules that the XSD does not encode. Common causes are a charge bearer code that is not allowed for that message type, an amount with too many fraction digits for the currency, a missing country code in an address, or a code value from a deprecated code-list release. The fix is to add those rules to your own validation layer so the rejection happens before the message leaves.

**Why does a message pass validation and still fail to reconcile?**
Because schema validity says nothing about semantic correctness. A reference type code that does not match the reference's real scheme, a truncated reference, or an unstructured remittance sent to a counterparty that requires structured data will all validate and all fail to match. These are mapping-policy defects, not parsing defects, and they are caught by reconciliation metrics rather than by validators.

## Further reading

The ISO 20022 registration authority publishes the message definitions, the external code lists, and the release schedule; those documents are the authoritative source for anything version-specific. Each clearing system publishes its own usage guidelines, which narrow the standard for that scheme and are where the business rules actually live. Read the usage guideline for the scheme you connect to alongside the standard, not instead of it — the gap between the two is where most production rejections come from.
