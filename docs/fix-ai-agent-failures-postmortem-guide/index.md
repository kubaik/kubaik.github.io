# Postmortems for AI agents: drift, cost, replay

## Why agent postmortems look different

An agent rarely crashes. It degrades. A prompt template gets edited, a retrieval index goes stale, a model version is swapped upstream, and the agent keeps returning HTTP 200 while quietly producing worse decisions. There is no stack trace for that. The incident is discovered later, through a billing spike, a support queue, or an audit.

That changes what a postmortem has to contain. Traditional incident review asks: what broke, when did it start, how many requests failed, how do we prevent recurrence. For an agent, the useful questions are:

- What changed in the input distribution, the prompt, the retrieval corpus, or the model version?
- When did output quality move, and how is "quality" defined for this task?
- What did the incident cost in tokens, retries, and downstream actions?
- Can a specific bad session be replayed and reproduced?

If your postmortem template cannot answer those four, it will produce a document that says "the model behaved unexpectedly" and nothing actionable.

Four failure modes are specific enough to plan for:

1. **Output drift.** Behavior changes gradually because inputs, prompts, or the model itself changed. Prompt templates are code, and they rot like code.
2. **Non-determinism.** The same input can produce different outputs. Replay is not free; it requires capturing enough state to constrain the run.
3. **Compute amplification.** One bad routing decision fans out into many model calls, or a retry loop multiplies calls for a subset of sessions.
4. **Confident fabrication.** The agent cites a regulation, function, or citation that does not exist. This is usually caught by a human downstream, not by monitoring.

## The minimum telemetry an agent postmortem needs

Most teams already log something. The gap is usually in what is captured per call. A postmortem is only as good as the fields you recorded before the incident. At minimum, record one structured event per model call with:

- `request_id` and `session_id`
- `timestamp`
- `model` and `model_version` (the exact snapshot identifier, not a family name)
- `prompt_template_id` and `prompt_template_hash`
- `input_tokens`, `output_tokens`, and the computed cost for that call
- `latency_ms`
- `retrieval_doc_ids` if the agent uses retrieval
- `tool_calls` made and their outcomes
- `finish_reason` and any error field
- a hash of the user input, and the raw input if policy allows storing it

Two fields are commonly omitted and are the ones you will miss most: the prompt template hash and the retrieval document IDs. Without the first, you cannot correlate a quality shift with a template edit. Without the second, you cannot tell whether the model changed or the context it was given changed.

Store these as structured events, not free-text log lines. JSON to stdout is fine if a collector parses it into a queryable store. The important property is that you can group by session and filter by template hash and model version after the fact.

## Measuring drift without a vendor

Drift detection does not require a specialized platform. It requires a definition of "normal" and a way to compare today's traffic to it.

**Step 1: pick a signal.** For unstructured text, common signals are output length distribution, refusal rate, and embedding distance from a reference set. For structured output, use schema-validity rate and field-level value distributions. For task-specific agents, use a labeled evaluation set and track pass rate.

**Step 2: build a reference window.** Take a period you consider healthy, for example the first week after a known-good deployment. Compute the signal's distribution over that window and store it.

**Step 3: compare on a schedule.** Recompute the signal over a rolling window, for example the last 24 hours, and compare it to the reference. For a scalar like mean output length, a simple z-score or a fixed percentage band works. For distributions, use a divergence measure such as population stability index or KL divergence over bucketed values.

**Step 4: alert on the comparison, not the raw value.** A mean output length of 400 tokens is meaningless on its own. A mean output length that moved 30 percent from the reference window is a signal.

A worked example makes the arithmetic concrete. Suppose the reference window has a mean output length of 320 tokens with a standard deviation of 40 tokens, computed over 5,000 calls. Today's window has 5,000 calls with a mean of 380 tokens. The standard error of the mean for each window is `40 / sqrt(5000)`, which is about 0.57 tokens. The difference in means is 60 tokens, which is roughly 105 standard errors. That is not noise; something changed. The same calculation on a 50-call window would have a standard error near 5.7 tokens, so a 60-token move would still be large but far less certain. This is why small windows generate false alarms.

The same structure applies to cost. Track cost per session, not cost per call. A session that fans out into 40 calls is the failure mode you care about, and per-call averages hide it.

## Tracing a real failure: a worked example

Consider an agent that drafts replies to inbound support tickets. On a Tuesday morning the support lead notices replies getting longer and more hedged. No alerts fired. Here is how the postmortem proceeds.

**Establish the timeline.** Query per-call events grouped by hour for the last seven days, filtering on `model_version` and `prompt_template_hash`. The output shows the template hash changed at 09:14 on Monday. Output length begins rising within the same hour. The model version did not change. That narrows the cause to the template or to the inputs.

**Check the inputs.** Group by a coarse input category, for example the ticket's product area. The length increase appears across all categories, which argues against an input shift and points at the template.

**Diff the template.** The change added an instruction to "consider all relevant policies before responding." The model interpreted this as an instruction to enumerate policies, inflating output length and adding hedging language.

**Quantify the cost.** Multiply the mean output token increase by the number of calls in the affected window and by the per-token output price. If the increase is 60 tokens per call, the window has 12,000 calls, and output costs $10 per million tokens, the added cost is `60 * 12000 / 1_000_000 * 10`, which is $7.20. That is small. Now suppose the same template change also increased the chance of a second model call for policy lookup on 15 percent of sessions, at 800 input tokens each and $2.50 per million input tokens. That adds `0.15 * 12000 * 800 / 1_000_000 * 2.50`, which is $3.60. The point of the arithmetic is not the total; it is that you can defend the number in the review instead of guessing.

**Write the action items.** Roll back the template. Add a regression check that runs the evaluation set against any template change before deploy. Add an alert on output length relative to the reference window. Add the template hash to the deploy record so the next timeline query is faster.

This example is illustrative, but the method is not: timeline by version and template hash, then narrow by input grouping, then quantify with per-token arithmetic.

## Choosing a pipeline: log-based or instrumented

Two broad approaches exist. The log-based approach uses your existing observability stack: structured logs, a log store, a query layer, and scheduled jobs you write yourself. The instrumented approach uses SDKs or sidecars that capture model calls and compute metrics for you, either self-hosted or as a managed service.

The decision is not about which is better in the abstract. It is about which constraints bind.

| Constraint | Log-based pipeline | Instrumented pipeline |
|---|---|---|
| Where the logic lives | Queries and jobs you write and maintain | SDK or sidecar plus configuration |
| Time to first useful signal | Fast if logs already exist | Depends on instrumentation coverage |
| Coverage | Only what you logged | Only what the SDK captures |
| Data residency | Whatever your log store supports | Depends on the vendor or your deployment |
| Cost model | Storage and query cost | Storage plus per-event or per-seat pricing |
| Custom metrics | Fully flexible | Limited to what the platform exposes |
| Failure mode | You forget a field and cannot recover it retroactively | You forget to instrument a call path and it is invisible |

The last row is the one that causes the most pain in both directions. In a log-based pipeline, the fields you did not log are gone. In an instrumented pipeline, the call paths you did not wrap are invisible, and that is often worse because the dashboard looks complete.

A practical hybrid is common: use an instrumented pipeline for standard metrics such as latency, token counts, and cost, and keep a log-based path for the fields specific to your domain, such as retrieval document IDs or tool call arguments. This avoids paying per-event for high-cardinality fields a vendor will not index usefully anyway.

## Instrumentation sketch

The following is a minimal pattern for capturing the fields a postmortem needs, independent of vendor. It assumes a single function wraps every model call.

```python
import hashlib
import json
import time
from dataclasses import dataclass, asdict

@dataclass
class CallEvent:
    request_id: str
    session_id: str
    timestamp: float
    model: str
    model_version: str
    prompt_template_id: str
    prompt_template_hash: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    finish_reason: str
    retrieval_doc_ids: list

def template_hash(template: str) -> str:
    return hashlib.sha256(template.encode("utf-8")).hexdigest()[:16]

def call_model(client, session_id, request_id, template, variables, model, model_version, price_in, price_out):
    prompt = template.format(**variables)
    start = time.monotonic()
    response = client.complete(model=model, prompt=prompt)
    latency_ms = (time.monotonic() - start) * 1000

    usage = response.usage
    cost = (usage.input_tokens / 1_000_000) * price_in + \
           (usage.output_tokens / 1_000_000) * price_out

    event = CallEvent(
        request_id=request_id,
        session_id=session_id,
        timestamp=time.time(),
        model=model,
        model_version=model_version,
        prompt_template_id=template_id_of(template),
        prompt_template_hash=template_hash(template),
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cost_usd=cost,
        latency_ms=latency_ms,
        finish_reason=response.finish_reason,
        retrieval_doc_ids=response.retrieval_doc_ids or [],
    )
    emit(json.dumps(asdict(event)))
    return response
```

Two details matter here. First, `model_version` must be the exact snapshot identifier the provider returns, not a family name, or the timeline query will not separate one deployment from another. Second, the price constants must be updated deliberately; a stale price makes every cost figure in the postmortem wrong in a way nobody notices.

## Replay and non-determinism

Full replay of a model call is not generally possible. Providers do not guarantee that the same input returns the same output, and sampling parameters may not be exposed or stable across versions. What you can do is capture enough to make a bad session reproducible in the sense that matters: you can rerun the same template with the same variables and the same retrieval documents and observe whether the failure recurs.

That requires storing, per call:

- the rendered prompt, or the template plus the exact variable values
- the retrieval document IDs and their contents at the time of the call
- the model version identifier
- the sampling parameters, if the provider exposes them

If the failure does not reproduce with those held fixed, the cause is likely the model or an upstream change you cannot control, which is itself a finding worth recording.

## A postmortem template for agent incidents

Use these sections. They map to the failure modes above and force the numbers to be shown.

**Summary.** One paragraph: what the agent did wrong, over what window, and what the user-visible impact was.

**Detection.** How the incident was found, and how long after onset. If detection was manual, that is an action item.

**Timeline.** Grouped by `model_version` and `prompt_template_hash` changes, with timestamps.

**Impact.** Sessions affected, extra model calls, added token cost computed from per-token prices, and downstream actions taken incorrectly.

**Root cause.** Which of the four failure modes, with the evidence that distinguishes it from the others.

**What made it hard to see.** The missing field, the unlogged call path, the alert that did not exist.

**Action items.** Each with an owner and a verification method. "Add an alert" is not an action item; "alert when mean output length over a 24-hour window deviates more than 20 percent from the reference window, verified by replaying last week's traffic" is.

## Decision checklist

Before the next incident, answer these. If any answer is "we do not know," that is the gap to close.

- Can you list every model call path in the agent, including retries and tool calls?
- Is the exact model version recorded per call?
- Is the prompt template hash recorded per call?
- Can you compute cost per session, not just per call?
- Do you have a reference window for at least one quality signal?
- Can you group calls by session and reconstruct the sequence?
- Are retrieval document IDs recorded?
- Is there a labeled evaluation set that runs before deploy?
- Does the alert fire on a comparison to a reference, not on a raw threshold?
- Is there a documented rollback for a prompt template change?

## Do this in the next 30 minutes

Pick one agent in production. Query your logs for the last 24 hours and count distinct `session_id` values and the number of model calls per session. If the distribution has a tail where a small number of sessions account for a disproportionate share of calls, you have found a compute amplification risk before it became an incident. Write down the query, save it, and add the two fields most likely missing from your events: the prompt template hash and the retrieval document IDs.
