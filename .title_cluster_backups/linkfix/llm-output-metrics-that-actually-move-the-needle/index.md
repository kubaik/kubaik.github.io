# LLM output metrics that actually move the needle

Most LLM evaluation documentation focuses on offline metrics: perplexity, ROUGE, BLEU. Those are useful for model selection in research settings, but they say very little about whether a deployed system is producing correct answers, staying within latency budgets, or holding on to users. This article describes a layered measurement approach that teams commonly adopt once an LLM feature has real traffic, and it explains the failure modes that show up when only one layer is instrumented.

## Why offline metrics are not enough in production

A typical failure mode looks like this. A summarization feature is demoed and judged on fluency. The offline ROUGE-L score is stable across releases. Weeks later, support tickets accumulate because summaries omit specific facts — a loan amount, a repayment date, an account identifier. ROUGE measures n-gram overlap with a reference, not factual correctness. A model can score well while dropping the one number the user needed.

The practical consequence is that production quality needs a different yardstick than research quality. The four signals that tend to matter most are:

1. **Fact accuracy** — whether extracted or generated values match ground truth.
2. **Latency at p95** — because tail latency, not mean latency, determines whether users abandon a feature.
3. **Cost per request** — because token-based pricing scales linearly with prompt size and traffic.
4. **Retention or adoption delta** — because a feature that is fast, cheap, and wrong still loses users.

Each of these is measurable, but each requires instrumentation that offline evaluation does not provide.

## The three layers of production LLM quality

Treat quality as a layered system rather than a single number.

**Layer 1 — Token-level quality.** Perplexity, ROUGE, and BLEU are appropriate for comparing candidate models on a fixed dataset. They are not appropriate as runtime gates. Embedding similarity between a prompt and its response can catch semantic drift, but it cannot confirm that a specific field is correct.

**Layer 2 — Structured correctness.** For domains such as finance or healthcare, the useful signal is whether extracted entities (dates, amounts, IDs) match a schema and match a source of truth. This is where schema validation and business rules belong.

**Layer 3 — User impact.** Latency, cost, and retention translate model behavior into product outcomes. A model change that improves Layer 1 and Layer 2 can still fail Layer 3.

The key insight is that a metric not tied to a business rule is decoration. If a validator cannot state what "correct" means in code, the metric cannot drive a rollback decision.

## Implementing structured extraction with Pydantic

Pydantic v2 provides field-level constraints and custom validators that map raw model output to a typed object. The example below defines a schema for a bank statement summary.

```python
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional

class StatementSummary(BaseModel):
    account_number: str = Field(..., pattern=r'^\d{10}$')
    statement_date: str = Field(..., pattern=r'^\d{4}-\d{2}-\d{2}$')
    total_balance: float = Field(..., gt=-1_000_000, lt=1_000_000)
    transactions_count: int = Field(..., ge=0)
    high_risk_transactions: Optional[List[str]] = []

    @field_validator('total_balance')
    @classmethod
    def check_balance_range(cls, v: float) -> float:
        if abs(v) > 1_000_000:
            raise ValueError('Balance out of expected range')
        return v
```

Two notes on correctness. In Pydantic v2 the field constraint keyword is `pattern`, not `regex`, and validators use `@field_validator` with `@classmethod` rather than the v1 `@validator` decorator. The validator here is redundant with the field constraint, which is fine for illustration but should be removed in production to avoid duplicated logic.

This schema rejects malformed account numbers, dates outside the expected format, and balances outside a plausible range before the value reaches a user. The business rule — a balance range for the target institution — is what makes the validator meaningful.

## Measuring latency and cost

Latency should be measured from the client's perspective, not from inside the model call. Server-side timing misses queueing, cold starts, and network transit. The metric to track is p95 end-to-end latency, because the mean hides the tail that users actually experience.

Cost should be computed from token counts, not estimated from request counts. Given a prompt of 10,000 tokens at $0.03 per 1,000 tokens, the prompt alone costs $0.30 per call. At 100,000 calls per day that is $30,000 per month. This arithmetic is illustrative; substitute the actual per-token price for the model in use. The point is that prompt size is a first-class cost variable, and it should be capped.

A token budget works as follows: cap the prompt at a fixed size, cap the response with a `max_tokens` parameter, and reject responses that exceed the cap at the gateway. If the average call uses 2,000 tokens at $0.005 per 1,000 tokens, the average cost is $0.01 per call. At 100,000 daily calls that is $1,000 per month. Doubling the prompt without a cap doubles the bill.

To measure cost daily, query the response table for token counts over the previous 24 hours and multiply by the per-token rate. The query below is illustrative and assumes a DynamoDB table with `prompt_tokens` and `completion_tokens` attributes:

```bash
# Run daily. Replace the timestamp with yesterday's date.
aws dynamodb scan \
  --table-name LLMResponses \
  --filter-expression "#ts >= :since" \
  --expression-attribute-names '{"#ts": "timestamp"}' \
  --expression-attribute-values '{":since": {"S": "2026-06-01T00:00:00Z"}}' \
  --projection-expression "prompt_tokens, completion_tokens" \
  --output json \
| jq '[.Items[] | (.prompt_tokens.N | tonumber) + (.completion_tokens.N | tonumber)] | add'
```

Note that `scan` with a filter reads the full table and is expensive at scale; a production implementation should use a query on a partition key and a secondary index on the timestamp. The output is a total token count, which is then multiplied by the per-token price to get the daily cost.

## Measuring retention delta

Retention measurement requires logging a custom event each time the feature is used, tagged with the model version. At the end of a comparison window, count distinct users per model version and compute the percentage change. The pseudocode below illustrates the shape of the check:

```python
import pandas as pd

# df has columns: distinct_id, model_version, event_date
retention = df.groupby('model_version')['distinct_id'].nunique()
delta = retention.pct_change()

if delta.iloc[-1] < -0.05:
    trigger_rollback(model_version=delta.index[-1])
```

Two cautions apply. First, retention is noisy; a 5% week-over-week threshold will fire on seasonality as well as regressions. Normalize by total feature opens and account for known seasonal patterns before acting. Second, retention is a lagging indicator. By the time it moves, users have already had a bad experience. It is a backstop, not a primary signal.

## How to measure fact accuracy

Fact accuracy is the metric most teams skip because it requires ground truth. Three approaches are practical:

- **Schema validation** — confirm that extracted values are well-formed and within plausible ranges. This catches format errors but not wrong values.
- **Cross-checking against a source of truth** — for a loan amount, compare the extracted value to the value in the loan table. This catches wrong values but requires a join.
- **Sampled human review** — label a random sample of outputs weekly and track the error rate over time. This is the only method that catches errors the other two miss, such as a plausible but incorrect date.

A useful instrument is to log both the raw model output and the validated object, then run a daily job that joins validated values against the source table and counts mismatches. The mismatch rate is the fact accuracy metric.

## Failure modes to plan for

**Schema drift across model upgrades.** Model versions differ in output formatting. A date that was emitted as `YYYY-MM-DD` may appear as `DD/MM/YYYY` after an upgrade, breaking a strict validator. Mitigation: include a `model_version` field in the schema and allow format variants conditionally, or normalize formats before validation.

**Prompt injection.** User-supplied text can instruct the model to ignore extraction rules. Mitigation: treat all user input as untrusted, separate instructions from data in the prompt structure, and validate outputs against the schema regardless of what the prompt requested. A web application firewall may reduce known patterns but does not replace output validation.

**Cold-start latency in serverless deployments.** A cold start can add hundreds of milliseconds to the first request after an idle period. If a meaningful fraction of requests are cold starts, p95 latency reflects the cold path, not the model. Mitigation: measure cold-start rate separately, keep functions warm, or use a runtime feature that reduces initialization time.

**Cost creep from prompt edits.** Adding a paragraph to a prompt adds tokens to every request. A 150-token addition at 50,000 daily calls and $0.005 per 1,000 tokens costs about $37.50 per month. That figure is illustrative; the point is that prompt edits have a recurring cost and should be reviewed like code.

**Metric drift from user behavior.** Retention and adoption metrics move with seasonality and marketing, not only with model quality. Mitigation: normalize by total feature usage and compare against a control group where possible.

## A decision checklist

Before adopting a layered measurement system, check the following:

- Is there a code-expressible definition of "correct" for this feature? If not, structured validation is not possible.
- Is the request volume high enough that the instrumentation cost is justified? Below roughly 1,000 requests per day, manual sampling may be sufficient.
- Is the use case factual or creative? Creative outputs should be evaluated on user ratings, not schema adherence.
- Can the model be rolled back quickly? Metrics without a rollback path do not reduce risk.
- Are there regulatory approval constraints on model changes? If so, the rollback mechanism must fit the approval process.

If the answer to the first question is no, or the answer to the fourth is no, a simpler approach — latency, cost, and periodic manual review — is the better choice.

## Tools by category

| Category | What to look for | Why it matters |
|---|---|---|
| Schema validation | Field constraints, custom validators, typed output | Catches malformed and out-of-range values before they reach users |
| Tracing and metrics | OpenTelemetry-compatible exporter, p95 aggregation | Measures end-to-end latency including queueing and cold starts |
| Cost accounting | Per-request token counts, daily aggregation | Turns token usage into a reviewable cost figure |
| Product analytics | Custom events tagged with model version | Ties model changes to retention and adoption |
| Entity extraction | Multilingual tokenization, custom rules | Reduces false positives when text mixes languages |
| Rollback control | Versioned deploys, feature flags | Converts a metric breach into a reversible action |

## When this approach is the wrong choice

The layered approach adds operational complexity. It is not appropriate when:

1. Request volume is very low. The overhead of validators, dashboards, and rollback pipelines exceeds the benefit.
2. The use case is purely creative. Users judge novelty, not factual accuracy; ratings are the better signal.
3. No business rule can be expressed in code. Without a definition of correctness, structured validation has nothing to check.
4. Model changes require lengthy approval. If rollback takes weeks, real-time metrics cannot act on their findings.

In those cases, measure latency and cost, sample outputs manually, and defer the rest until volume or risk justifies it.

## What to do in the next 30 minutes

Open the table or log store where LLM responses are recorded. If token counts are not stored per request, add two columns — `prompt_tokens` and `completion_tokens` — and populate them from the API response for new requests. Then run a query over the last 24 hours that sums those columns and multiplies by the per-token price for the model in use. Compare the result to the expected daily budget. If the figure is higher than expected, the prompt size is the first thing to inspect, because prompt tokens are paid on every call and are the easiest cost to reduce.
