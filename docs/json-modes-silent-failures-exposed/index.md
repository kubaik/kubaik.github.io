# JSON mode’s silent failures exposed

## Why JSON mode fails silently

JSON mode is widely misunderstood as a validation feature. It is not. It constrains the model's output to a JSON-shaped grammar, which reduces the chance of a syntactically broken string. It does not guarantee that the returned object satisfies your schema, and it does not distinguish between "the model deliberately omitted this field" and "the model failed to produce this field."

The documented behavior of JSON mode is that the model is constrained to emit valid JSON syntax. Everything above that — required fields, types, null handling, casing — is your responsibility. When the model is uncertain, a common failure mode is that it omits fields rather than emitting a placeholder. The result is valid JSON that violates your contract. Downstream consumers that assume the contract holds will silently drop rows, truncate values, or write nulls into columns that were declared NOT NULL.

Two failure classes are worth separating:

- **Syntactic failure**: the output is not parseable as JSON. This is loud. Your parser throws, your job fails, you get an alert.
- **Semantic failure**: the output parses but violates the schema. This is quiet. It propagates into storage and surfaces days later as missing data.

JSON mode addresses the first class. The second class is where most production incidents live.

The root cause is a mismatch between what the model optimizes for and what the downstream system requires. The model produces plausible text. The downstream system requires a typed contract. A grammar flag does not close that gap.

## A minimal failure demonstration

Consider a schema that requires `id`, `status`, and `email`. A model given an ambiguous input may return:

```json
{ "id": 123 }
```

This is valid JSON. It parses cleanly. It fails schema validation. If your pipeline only checks `json.loads()`, it will accept this payload and write a row with a missing `status`.

A second common case is a JSON-encoded string nested inside a JSON document:

```json
{ "payload": "{\"id\": 123, \"status\": \"pending\"}" }
```

Again, valid JSON. Again, wrong shape.

A third case: the model returns the correct fields but with a type mismatch — `"id": "123"` instead of `"id": 123`. Whether this breaks depends entirely on how strict your downstream consumer is. Typed schemas catch it; text-based consumers do not.

## The envelope protocol

The fix is not a better prompt or a higher temperature setting. The fix is to treat structured output as a protocol with four explicit layers:

1. **Schema**: a machine-checkable definition of valid output.
2. **Envelope**: a wrapper that carries metadata about the payload, including whether it was validated, repaired, or flagged.
3. **Validation and repair**: a deterministic pipeline that either fixes known violations or quarantines the payload.
4. **Metrics**: instrumentation that makes failure visible before it reaches storage.

The envelope is the key idea. It gives you a place to record the outcome of validation, so downstream systems can act on the payload's provenance rather than guessing.

A minimal envelope schema in Pydantic:

```python
from pydantic import BaseModel, EmailStr, field_validator

class ExtractionResult(BaseModel):
    id: int
    status: str
    email: EmailStr | None = None
    metadata: dict[str, str] = {}

    @field_validator("status")
    @classmethod
    def status_must_be_valid(cls, v: str) -> str:
        allowed = {"pending", "processed", "failed"}
        if v not in allowed:
            raise ValueError(f"status must be one of {sorted(allowed)}")
        return v

class Envelope(BaseModel):
    version: str
    timestamp: str
    guardrails: str
    payload: ExtractionResult
```

The `guardrails` field is a string enum in practice: `"validated"`, `"repaired"`, or `"flagged"`. Downstream consumers can branch on it. A `"flagged"` payload is quarantined, not written.

## Prompting the envelope

The system prompt must describe the envelope, not just the payload. It must also tell the model what to do when it is uncertain.

```python
SYSTEM_PROMPT = """
You are an extraction agent. Return ONLY a JSON payload wrapped in a text envelope.

Envelope format:
{
  "version": "1",
  "timestamp": "<ISO-8601 UTC>",
  "guardrails": "validated",
  "payload": {
    "id": 123,
    "status": "pending",
    "email": "user@example.com"
  }
}

Rules:
- Never return malformed JSON.
- If a required field cannot be determined, set it to null and set guardrails to "flagged".
- Do not invent values. Omission is an error; explicit null is acceptable.
"""
```

The instruction "omission is an error; explicit null is acceptable" is doing real work. Without it, models tend to drop uncertain fields. With it, uncertain fields become visible as nulls, which the validation layer can handle deterministically.

## Validation and repair

The validation step parses the envelope, validates the payload against the schema, and attempts deterministic repair for a known set of violations. Repair rules should be static and reviewable — dynamic schema inference in production is a source of runtime surprises.

```python
import json
from pydantic import ValidationError

REPAIR_RULES = {
    "missing_status": lambda payload: {**payload, "status": "pending"},
    "invalid_email": lambda payload: {**payload, "email": None},
    "string_id": lambda payload: {**payload, "id": int(payload["id"])},
}

def classify_error(error: ValidationError) -> str:
    for err in error.errors():
        loc = ".".join(str(p) for p in err["loc"])
        if "status" in loc and err["type"] == "missing":
            return "missing_status"
        if "email" in loc:
            return "invalid_email"
        if "id" in loc and err["type"] in {"int_parsing", "int_type"}:
            return "string_id"
    return "unrepairable"

def validate_and_repair(envelope_json: str) -> dict:
    try:
        raw = json.loads(envelope_json)
    except json.JSONDecodeError as e:
        return {"valid": False, "guardrails": "flagged", "reason": f"parse_error: {e}"}

    try:
        envelope = Envelope.model_validate(raw)
        return {"valid": True, "guardrails": "validated", "payload": envelope.model_dump()}
    except ValidationError as e:
        rule_name = classify_error(e)
        rule = REPAIR_RULES.get(rule_name)
        if rule is None:
            return {"valid": False, "guardrails": "flagged", "reason": str(e)}

        candidate = dict(raw.get("payload", {}))
        repaired_payload = rule(candidate)
        try:
            envelope = Envelope.model_validate(
                {**raw, "guardrails": "repaired", "payload": repaired_payload}
            )
            return {"valid": True, "guardrails": "repaired", "payload": envelope.model_dump()}
        except ValidationError as e2:
            return {"valid": False, "guardrails": "flagged", "reason": str(e2)}
```

Two properties matter here. First, repair is bounded: only known rule names are applied, and the repaired payload must re-validate. Second, unrepairable payloads are quarantined with a reason string, not dropped. Quarantine preserves the raw input so the failure can be investigated.

## Instrumenting the pipeline

The metrics that matter are the ones that tell you whether the contract is holding:

- `envelope_parse_errors` — count of payloads that failed JSON parsing entirely.
- `schema_validation_failures` — count of payloads that parsed but failed validation, broken down by rule name.
- `repairs_applied` — count of successful deterministic repairs, broken down by rule name.
- `quarantined_payloads` — count of payloads routed to the quarantine queue.
- `guardrail_ratio` — the fraction of payloads with `guardrails != "validated"`.

The `guardrail_ratio` is the single most useful number. A sudden rise in `repaired` events with a specific rule name usually indicates a change in the input distribution — a new document template, a new upstream source, a new file format — rather than a change in the model. This is why the reason field matters: it points you at the input, not the model.

## How to measure the cost and latency of a repair pipeline

Published cost figures for a specific pipeline are not portable, because they depend on your payload size, your concurrency profile, and your runtime. What is portable is the method.

To measure repair cost:

1. Instrument the repair function to log the number of invocations and the wall-clock duration per invocation.
2. Multiply invocations by the per-invocation price of your compute unit (for a serverless function, use the documented GB-second and request pricing; for a container, use the instance-hour price divided by measured utilization).
3. Compare the total against the model inference cost for the same window. If repair exceeds inference, the repair path is doing too much work.

To measure the latency impact:

1. Record a timestamp before the model call and after validation completes.
2. Emit the difference as a histogram, not an average. Averages hide the tail.
3. Compare the p50 and p95 of the validation step against your downstream ingestion SLA.

A worked illustrative example. Suppose a repair function runs 10,000 times per night, each invocation takes 50 ms, and the function is allocated 512 MB. The GB-seconds per invocation are `0.512 GB * 0.05 s = 0.0256 GB-s`. At a hypothetical price of $0.0000166667 per GB-s (a round number chosen for arithmetic, not a quote), that is `0.0256 * 0.0000166667 ≈ $0.000000427` per invocation, or about `$0.00427` for 10,000 invocations. The point of the exercise is not the number — it is that the number is small and knowable, and that you should compute it from your own measurements rather than trusting a figure from an article.

## Failure modes to plan for

**The model returns a dict instead of a string.** Some client libraries deserialize the response content before you see it. Code that assumes `response.choices[0].message.content` is a string will raise `AttributeError`. Guard with an explicit type check and normalize to a string before parsing.

**The model wraps the envelope in prose.** Despite instructions, some outputs begin with "Here is the JSON:" or end with a trailing sentence. A strict `json.loads` will fail. Either strip known prefixes before parsing, or treat the payload as a parse error and quarantine it. Quarantining is safer than heuristic stripping, because heuristic stripping can silently mangle valid content.

**The envelope itself is truncated.** Long payloads can hit output token limits mid-object. The result is a parse error. This is loud and should be counted separately from schema failures, because the remediation is different: raise the token budget or reduce the payload size.

**A repair rule masks a real bug.** If `missing_status` always repairs to `"pending"`, and the upstream extractor stops returning status entirely, the repair path will hide the regression. Track repair rates per rule and alert on sustained increases.

**Schema drift between services.** If two services validate against different versions of the schema, one will accept what the other rejects. Version the schema explicitly and include the version in the envelope.

## Decision checklist

Before shipping a structured output pipeline, confirm:

- [ ] The output schema is defined in code and exported to JSON Schema.
- [ ] The model is instructed to return an envelope with a `guardrails` field.
- [ ] Validation distinguishes parse errors from schema errors.
- [ ] Repair rules are static, named, and re-validate after application.
- [ ] Unrepairable payloads are quarantined with a reason, not dropped.
- [ ] Metrics are emitted for parse errors, schema failures, repairs, and quarantines.
- [ ] The schema is versioned and the version is carried in the envelope.
- [ ] The repair path has been tested against synthetic malformed payloads.

## FAQ

**Is JSON mode useless?**
No. It reduces syntactic failures, which are the loudest and most annoying class. It just does not replace schema validation. Use it, and validate anyway.

**Should repair ever modify data?**
Only when the modification is deterministic and defensible — for example, coercing a numeric string to an integer, or replacing an invalid email with null. Never use repair to guess at values the model failed to produce.

**What about provider-side structured output features?**
Some providers offer schema-constrained decoding, which is stronger than JSON mode. It is still worth wrapping the output in an envelope, because constrained decoding does not tell you whether the model was confident — only that the output matched the grammar. The envelope is where you record that distinction.

**Does the envelope add token cost?**
Yes. The envelope fields add tokens to every request and response. Measure the increase on a representative sample before committing to a format, and keep the envelope as small as the metadata requires.

**Can this work with any model provider?**
The protocol is provider-agnostic. The prompt format and the client library differ; the schema, envelope, validation, and repair layers do not.

## What to do in the next 30 minutes

Open your current structured output code path and add a single check: after parsing the model's response, validate it against your schema and log the validation result with a rule name for each failure. Do not add repair yet. Run it for one batch and count how many payloads fail validation but parse successfully. That number is the size of your silent failure problem, and it is the number you should be optimizing against.

Stop treating JSON mode as a contract. Define the contract yourself.
