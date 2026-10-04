# JSON mode's hidden costs revealed

Most structured-output guides assume a clean environment and a patient timeline. Production gives you neither. The failure mode described below is common enough to have a name in postmortems: the pipeline that emitted valid JSON and still corrupted the accounting ledger.

## The situation

Consider a real-time expense categorization service. Raw transaction text such as `Safeway on 01/14/2026 $47.12` goes into an LLM, and structured fields — merchant, date, amount — come out. Services like this run at tens of thousands of transactions per minute at peak, often across multiple regions.

The natural starting point is the provider's built-in JSON mode. It is documented to guarantee syntactically valid JSON. A first version is therefore a short function that calls the chat completion endpoint with a JSON response format. The first few dozen calls work. Then the errors appear: malformed JSON in a meaningful share of responses, missing fields, and array fields occasionally rendered as strings.

The problem is not JSON mode. The problem is the assumption that "structured output" means "usable data." What downstream systems need is reliable parsing, not syntactically correct blobs. An expense system requires merchant names normalized (`7-Eleven` vs `Seven Eleven`), dates in ISO format, and amounts as decimals. JSON mode guarantees none of that.

Over time, teams that stay on this path find a growing share of compute budget going to validation and retry logic, monthly model bills climbing well past what the traffic alone would justify, and on-call rotations fielding alerts about misclassified expenses. Something has to change.

## Why the obvious fixes fall short

**Attempt 1: JSON mode plus regex validation.** A validation pass using the standard `json` module and regular expressions catches most malformed responses but adds tens of milliseconds per call. The accounting team notices the latency before it notices the improvement. It is a wall: accept the latency penalty or let bad data through.

**Attempt 2: JSON mode with a retry loop.** Exponential backoff with three retries per call reduces bad data substantially but multiplies model invocations. At 15,000 calls per minute, three retries means up to 45,000 extra calls per minute in the worst case. The bill scales with the failure rate, which is exactly backwards: the more the model struggles, the more you pay.

**Attempt 3: Fine-tuning the base model.** Fine-tuning on a few thousand labeled examples can reduce format errors, but it changes model behavior in ways that are hard to anticipate. A fine-tuned model may hallucinate categories for unfamiliar merchants, and the artifact is large enough that serving it requires meaningfully more GPU capacity than the base model. The worst outcome is that it still produces `7-Eleven` and `Seven Eleven` inconsistently, because consistency was never a property fine-tuning optimized for.

**Attempt 4: Schema validation with a typed model.** A typed model with strict validation looks clean on paper:

```python
from pydantic import BaseModel, field_validator
import json

class Transaction(BaseModel):
    merchant: str
    date: str
    amount: float

    @field_validator('merchant')
    @classmethod
    def normalize_merchant(cls, v: str) -> str:
        return v.strip().lower().replace("  ", " ")

raw_output = llm_output.strip()
try:
    data = json.loads(raw_output)
    transaction = Transaction(**data)
except Exception as e:
    # log error and retry
    ...
```

This catches most errors but requires a schema per data type, and each schema becomes a small maintenance surface. Across a fleet of microservices the schemas multiply faster than the teams that own them. Validation also adds per-call overhead on top of the model call.

All four attempts fail the same way: they treat the LLM as the source of truth and try to validate its output, rather than treating it as a noisy, probabilistic extractor whose output needs correction.

## The approach that works

Stop trying to make the LLM produce perfect structured data. Design a two-stage pipeline: extraction plus correction. LLMs are strong at pulling entities out of messy text and weak at consistent formatting. Split the job accordingly.

1. **Extraction stage:** use the LLM to pull out raw entities (merchant, date, amount) without enforcing strict structure.
2. **Correction stage:** apply deterministic validation, normalization, and deduplication to the raw entities.

The extraction prompt becomes:

```text
Extract the following fields from the transaction text:
- merchant: the store or service name
- date: the transaction date in MM/DD/YYYY format
- amount: the transaction amount

Transaction text: {transaction_text}

Return ONLY the extracted values, one per line, no explanations.
```

The correction stage uses:

- Regex patterns for date and amount parsing.
- A merchant normalization table mapping known variants to canonical names.
- A deduplication step that groups similar merchant names.

Why this works: the model is asked to do the one thing it is reliably good at — locating entities in messy text. Every property your downstream system actually depends on (format, canonical names, numeric types) is enforced by code that is deterministic and testable. The model's variance is confined to a single step whose failure modes you can enumerate.

## Implementation

A representative stack:

- **LLM provider:** a hosted chat-completions API with a JSON-capable model.
- **Runtime:** FastAPI with Python 3.11 on a current LTS Linux distribution.
- **Message queue:** Redis Streams for request buffering.
- **Validation:** Python `dataclasses` with custom validation.
- **Merchant normalization:** SQLite with a merchant-variant table.

The extraction endpoint:

```python
from anthropic import Anthropic
from dataclasses import dataclass
import re
from typing import Optional
import logging

client = Anthropic()

@dataclass
class RawTransaction:
    merchant: Optional[str] = None
    date: Optional[str] = None
    amount: Optional[str] = None

def extract_entities(text: str) -> RawTransaction:
    prompt = f"""
    Extract the following fields from the transaction text:
    - merchant: the store or service name
    - date: the transaction date in MM/DD/YYYY format
    - amount: the transaction amount

    Transaction text: {text}

    Return ONLY the extracted values, one per line, no explanations.
    """

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=64,
        temperature=0.1,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_output = response.content[0].text.strip()
    return parse_raw_output(raw_output)

def parse_raw_output(raw: str) -> RawTransaction:
    tx = RawTransaction()
    lines = [line.strip() for line in raw.split("\n") if line.strip()]

    for line in lines:
        if re.match(r'\d{2}/\d{2}/\d{4}', line):
            tx.date = line
        elif re.match(r'\$?\d+\.\d{2}', line):
            tx.amount = line
        else:
            tx.merchant = line

    return tx
```

The correction endpoint:

```python
from datetime import datetime
import re
import sqlite3

NORMALIZATION_DB = "merchant_normalization.db"

class TransactionCorrector:
    def __init__(self):
        self.conn = sqlite3.connect(NORMALIZATION_DB)
        self.conn.execute("PRAGMA journal_mode=WAL")

    def normalize_merchant(self, merchant: str) -> str:
        # Strip and normalize whitespace
        merchant = re.sub(r'\s+', ' ', merchant.strip().lower())

        # Check against known variants
        cursor = self.conn.execute(
            "SELECT canonical FROM merchants WHERE variant = ?",
            (merchant,)
        )
        result = cursor.fetchone()
        return result[0] if result else merchant

    def parse_date(self, date_str: str) -> str:
        try:
            return datetime.strptime(date_str, "%m/%d/%Y").isoformat()
        except ValueError:
            return None

    def parse_amount(self, amount_str: str) -> float:
        try:
            cleaned = re.sub(r'[^0-9.]', '', amount_str)
            return float(cleaned)
        except ValueError:
            return None
```

A few notes on the code above, because the details matter more than the shape:

- `parse_raw_output` assigns the first non-date, non-amount line to `merchant`. If the model emits multiple lines that are neither, the last one wins. For a single merchant per transaction that is fine; for multi-merchant receipts you need a different contract, such as one entity per line with a leading key.
- `normalize_merchant` lowercases before lookup, so the `merchants` table must store lowercase variants. Mixing cases in the table produces silent misses.
- `parse_date` returns `None` on failure rather than raising. The caller must decide whether a missing date is a rejection or a deferred item. Do not silently default to today's date.

The pipeline runs with Redis Streams as the message queue. Each consumer processes a bounded number of messages per second with a worker pool sized to the model's throughput. The FastAPI endpoint uses a circuit breaker to prevent cascading failures when the model provider degrades.

The merchant normalization table is updated on a schedule from an internal merchant catalog. SQLite handles concurrent reads well on a single small instance; the file is small enough to ship with the service or mount as a read-only volume.

## Measuring the pipeline honestly

The numbers below are illustrative, not measured from a production system. The point is the method: instrument these five quantities before and after the change.

| Metric | What to instrument | How to compare |
|---|---|---|
| Valid structured outputs | Count rows rejected by the correction stage | Ratio of accepted to total, per hour |
| p95 latency | Histogram of end-to-end request duration | Compare extraction-only vs extraction+correction |
| Model cost | Provider usage API, tokens in and out | Multiply by published per-token price |
| Downstream error rate | Tickets or exceptions from the accounting system | Weekly count, normalized by transaction volume |
| Maintenance hours | Time logged against the pipeline's repos | Per sprint, per engineer |

To produce a real table, run a labeled sample of at least a few hundred transactions through both pipelines and diff the outputs. Any figure you cannot reproduce from a script is not a result; it is an anecdote.

## What to do differently

**1. Start with extraction, not structure.** Ask "What can the model reliably extract?" rather than "How do we validate its output?" The first question leads to a design; the second leads to an arms race.

**2. Avoid fine-tuning for formatting.** Fine-tuning changes model behavior in unpredictable ways. The base model is already good at entity extraction; formatting belongs in code. Fine-tuning also couples you to a specific model version, which makes upgrades expensive.

**3. Invest in merchant normalization early.** Normalization tables grow organically. Seed them from an existing merchant catalog on day one. A well-maintained table reduces both model calls and downstream inconsistency.

**4. Use a queue from day one.** Direct API calls are fine for prototypes. In production, buffering smooths model latency spikes and decouples your request rate from the provider's throughput.

**5. Monitor at the extraction stage, not the output stage.** JSON validity is a downstream concern. The signal that matters is entity extraction accuracy: did the pipeline get merchant, date, and amount on the first pass? Track that per field, not per response.

The core lesson: do not let the LLM do your validation work. It is expensive, inconsistent, and fragile. Use it for what it is good at — extracting entities from messy text — and handle the rest with deterministic code.

## The broader pattern

Structured outputs from LLMs are a workflow problem, not a feature. JSON mode, guardrails, and fine-tuning are all attempts to push the model into a validation role it was not designed for. The alternative is to treat the model as an extraction engine.

The pattern generalizes:

- **Resume parsing:** extract skills and roles, then map them to a controlled vocabulary.
- **Clinical notes:** extract symptoms and medications, then normalize to standard codes.
- **Support tickets:** extract intent and entities, then route deterministically.

The principle is the same everywhere: **LLMs for extraction, deterministic code for correction.**

## Failure modes to watch for

- **Model emits a plausible but wrong entity.** Extraction accuracy is a separate metric from format validity. Sample and label regularly.
- **Normalization table drifts.** New merchants appear constantly. Schedule a review of unmatched merchant strings and promote the frequent ones into the table.
- **Correction stage becomes a second model.** If the correction logic starts making fuzzy judgments, it will inherit the same inconsistency you removed from the extraction stage. Keep it deterministic.
- **Silent defaults.** Returning `None` and then defaulting to a plausible value downstream hides failures. Surface them at the boundary.
- **Queue backlog.** Buffering absorbs spikes, but a sustained backlog means the extraction stage cannot keep up. Alert on queue depth, not just on errors.

## Applying this to your workflow

Start by answering three questions about your current pipeline:

1. **What entities does the model reliably extract?** Run a labeled sample of at least 100 inputs through the current model. Measure per-field accuracy. If a field is missed more than occasionally, the prompt or model is wrong, not the validation.
2. **Where does the output fail your downstream systems?** List the specific rules the consumer enforces — formats, allowed values, required fields. Write them down. Most teams discover these rules only when data breaks.
3. **What deterministic correction handles the edge cases?** Identify the normalization, deduplication, and validation steps that belong in code.

Then implement:

- **Extraction:** a minimal prompt that asks for raw entities without structure.
- **Correction:** strict validation and normalization in code.
- **Queue:** buffer requests to absorb model latency spikes.
- **Monitor:** track per-field extraction accuracy, not JSON validity.

## FAQ

**Why does JSON mode still produce malformed output?**
JSON mode is documented to guarantee syntactically valid JSON, not semantically correct data. The model can emit `{"merchant": null, "date": "invalid", "amount": "$47.12"}` — valid JSON, useless to the consumer. JSON mode constrains structure, not field values.

**How much latency does extraction plus correction add?**
The extraction call dominates; the correction stage is a regex pass, a dictionary lookup, and a number parse, all of which are sub-millisecond to low-millisecond operations. The apparent latency win over JSON mode with validation comes from removing retries and heavy schema validation, not from the correction stage being free. Measure end-to-end, not stage by stage.

**How should merchant variants like `7-Eleven` and `Seven Eleven` be handled?**
Build a normalization table of known variants. Seed it from your existing merchant catalog, then supplement with common misspellings. Use fuzzy matching to surface new candidates for review, but keep the mapping itself exact. Storing the canonical name and mapping every variant to it scales better than trying to train the model to normalize names.

**When is fine-tuning the right call?**
Fine-tune when the model consistently fails to extract entities you need, not when it extracts them and formats them inconsistently. Formatting is a code problem. Fine-tuning is expensive, slow, and fragile across model updates.

## One action for the next 30 minutes

Write a script that runs 100 representative inputs through your current pipeline and records, per field, whether the extracted value was correct. Print the per-field accuracy. That single number tells you whether your problem is extraction or correction — and which of the two stages to fix first.
