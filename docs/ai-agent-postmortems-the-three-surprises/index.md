# AI agent postmortems: the three surprises

## Why agent incidents break normal playbooks

A conventional service fails loudly. A request returns a 5xx, a health check goes red, a queue depth climbs, and an alert fires within a minute. An AI agent can fail quietly. It returns HTTP 200, the schema validates, the latency is nominal, and the answer is wrong in a way no type checker can see. The output satisfies the API contract but violates the business rule.

This is the reason a postmortem template written for uptime incidents is insufficient for agents. Uptime metrics answer "did the request complete?" They do not answer "was the completion correct?" For an agent, the interesting failure surface is semantic: prompt drift, retrieval noise, tool-call misalignment, and confidence miscalibration. A postmortem that only records CPU, memory, and p99 latency will document a healthy system that happened to approve a fraudulent claim.

A useful framing is to separate three failure classes:

- **Hard failure.** The agent throws, times out, or returns malformed output. Standard playbooks apply.
- **Contract failure.** The output parses but breaks a declared constraint, such as an enum value outside the allowed set. Detectable with validation.
- **Semantic failure.** The output is well-formed and in-range but wrong relative to ground truth. This is the class that requires new instrumentation.

Most agent postmortems that go badly are semantic failures retrofitted into a hard-failure template. The timeline ends up empty because nothing "broke." The action items are vague because the team never captured what the agent actually decided and why.

## Two detection strategies worth comparing

Two approaches cover most production needs. They are not mutually exclusive, but they have very different cost and latency profiles, so it is worth understanding each before choosing a default.

**Structured semantic logging** wraps each agent step in a typed event and ships it to a central sink. Every retrieval call, tool call, and final decision becomes a span with named attributes. The value is forensic: after an incident, you can reconstruct exactly which retrieval chunk drove a decision. The limitation is that it is retrospective. It tells you what happened after it happened.

**Live shadowing against a golden dataset** duplicates live requests to a parallel agent instance that does not serve traffic, and compares the shadow output to labeled ground truth in real time. The value is prospective: drift is detected while the primary agent is still serving the bad behavior. The limitation is cost and setup effort.

A note on terminology: "golden dataset" here means a set of inputs paired with the correct output class, not a set of ideal text strings. The comparison is usually a class match or an embedding-distance match, not string equality.

## Structured semantic logging: how it works

The core idea is to make every decision point observable with a typed attribute rather than a free-text log line. A workable span schema for an agent step includes:

- `llm.prompt` — the prompt text, truncated to a bounded length to control storage
- `llm.completion.token_count` — output token count, useful for detecting runaway generation
- `retrieval.query`, `retrieval.hit_count`, `retrieval.miss_ratio` — retrieval health
- `decision.output_class` — an enum such as `Approve`, `Reject`, `ManualReview`
- `decision.confidence` — a float in `[0, 1]` if the model or a wrapper produces one

The output class enum is the important part. It converts a free-form decision into a value you can aggregate and alert on. Without it, every dashboard is a text search.

A representative query pattern is to surface spans where the agent made a confident terminal decision and the retrieval was weak:

```sql
SELECT
  span_id,
  decision_output_class,
  decision_confidence,
  retrieval_miss_ratio
FROM agent_spans
WHERE decision_output_class IN ('Approve', 'Reject')
  AND decision_confidence < 0.75
  AND retrieval_miss_ratio > 0.5
ORDER BY ts DESC
LIMIT 100;
```

The exact column names depend on the sink. The shape of the query is what matters: you are looking for the intersection of confidence and retrieval weakness, because that is where semantic failures cluster.

A second query pattern checks drift over time by comparing the distribution of output classes across two windows:

```sql
SELECT
  decision_output_class,
  countIf(ts >= now() - INTERVAL 1 DAY) AS last_24h,
  countIf(ts >= now() - INTERVAL 8 DAY AND ts < now() - INTERVAL 1 DAY) AS prior_week
FROM agent_spans
GROUP BY decision_output_class;
```

A shift in the ratio between `Approve` and `ManualReview` across those windows is a leading indicator, even before any user reports a problem.

Where this approach fits well: the agent's output space is small and enumerable, the team already runs a log pipeline, and the primary goal is fast root-cause analysis after an incident. It is also the cheapest option to start, because it adds no parallel inference.

Where it falls short: it cannot detect a novel failure class that has never produced a span before. If the agent invents a new output category, the enum will not have a bucket for it, and the drift will appear as an unexpected value or a validation error rather than a clean alert.

## Live shadowing: how it works

Shadowing runs a second agent instance on mirrored input and compares its output to labeled ground truth. The live agent's response is unaffected; the shadow agent's response is only used for measurement.

The comparison metrics that matter:

- **Exact match rate (EM).** Fraction of shadow outputs identical to the golden label. Useful only when outputs are short and normalized; capitalization and whitespace differences will inflate the miss rate.
- **Semantic match rate (SM).** Fraction of shadow outputs within an embedding-distance threshold of the golden label. This is the metric to alert on, because it tolerates phrasing variation.
- **Latency delta.** Shadow latency minus live latency. This is a cost signal, not a correctness signal, but it tells you whether the shadow path is keeping up.

A shadowing pipeline has four moving parts: a request duplicator, a shadow agent, a comparator, and a golden dataset. The comparator is where most of the engineering effort goes, because the threshold choice determines the false-positive rate.

Where this approach fits well: the output space is large or open-ended, the cost of a wrong decision is high, and the team can invest in curating labeled examples. It is the only approach that catches a drift within seconds of the first affected request, rather than after a user complaint.

Where it falls short: it roughly doubles inference cost for the shadowed traffic, it requires a golden dataset that covers edge cases, and it introduces a new mental model for developers. Semantic equivalence is not the same as syntactic equivalence, and a team that has only ever written exact-match assertions will need calibration time.

## A worked example: choosing a threshold

Threshold selection is where shadowing projects most often go wrong, so it is worth working through the arithmetic rather than picking a number by intuition.

Suppose you have a labeled calibration set of 100 known-correct pairs and 100 known-incorrect pairs. For each candidate threshold `t`, you compute the confusion matrix at that threshold and pick the `t` that maximizes F1.

Illustrative numbers, chosen to show the method rather than to report a real measurement:

| Threshold (cosine distance) | True positives | False positives | False negatives | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| 0.05 | 62 | 4 | 38 | 0.94 | 0.62 | 0.75 |
| 0.10 | 81 | 9 | 19 | 0.90 | 0.81 | 0.85 |
| 0.15 | 93 | 17 | 7 | 0.85 | 0.93 | 0.89 |
| 0.20 | 98 | 31 | 2 | 0.76 | 0.98 | 0.86 |

Read the table as a method, not a result. At `t = 0.05`, the comparator is strict: it rarely calls a wrong answer correct, but it misses many correct answers, so recall is low. At `t = 0.20`, it is lenient: it catches almost every correct answer but generates many false alarms, so precision drops. The F1 peak in this illustrative set is at `0.15`.

The important consequence is that the threshold is a property of your embedding model, your label distribution, and your tolerance for false alarms. It is not a universal constant. Re-run the calibration whenever you change the embedding model or materially change the label distribution.

If you cannot afford a labeled calibration set, a cheaper proxy is to sample 200 recent production decisions, have a human label them as correct or incorrect, and run the same sweep. Two hundred labels is usually enough to see the shape of the curve, though not enough to trust the third decimal place.

## Measuring cost and overhead honestly

Published cost tables for agent observability are usually not reproducible, because they depend on traffic shape, retention, and instance pricing that change constantly. A more durable approach is to measure the two quantities that matter on your own workload.

**Instrumentation overhead.** Measure the CPU and memory of the agent process with the tracing SDK disabled, then enabled, on the same traffic. The delta is your overhead. The measurement command depends on your runtime; for a containerized service, comparing `container_cpu_usage_seconds_total` and `container_memory_working_set_bytes` across the two configurations over a fixed window is sufficient. Expect the overhead to scale with span count, not with request count, so an agent that emits ten spans per request will pay more than one that emits two.

**Shadow compute cost.** Shadow cost is approximately the cost of running the same inference twice for the shadowed fraction of traffic. If you shadow 100% of traffic, double your inference bill for the shadowed model. If you shadow 5%, the added cost is 5% of the inference bill, at the price of slower detection. The tradeoff is explicit: shadowing a fraction of traffic is a sampling decision, and the detection latency scales inversely with the sampling rate.

A useful practice is to record both numbers in the postmortem itself. A postmortem that says "we added shadowing" is less useful than one that says "shadowing added N milliseconds of latency and M dollars per month at current traffic, and detected the drift within K requests."

## Failure modes to watch for

These are the failure modes that most often undermine an otherwise sound detection setup.

**Enum drift.** The agent emits an output class that is not in the enum. Depending on your validation, this either throws a hard error or, worse, gets coerced to a default. If the default is `Approve`, a schema violation silently becomes an approval. Validate enums strictly and alert on unknown values.

**Confidence miscalibration.** The model's self-reported confidence is not a probability. A model can report `0.95` while being wrong. Do not set alert thresholds on raw confidence without validating that it correlates with accuracy on your labeled set. If it does not correlate, drop the confidence attribute from alerts and rely on semantic match instead.

**Retrieval staleness.** The retrieval index is refreshed less often than the source data changes. The agent retrieves an outdated chunk and produces a decision that was correct last week. This is invisible in latency and error metrics. Track the age of the retrieved documents as an attribute and alert when the median age exceeds the expected refresh interval.

**Prompt version skew.** A prompt change is deployed to one code path but not another, or a cached prompt is served after an update. The result is two agents behaving differently under the same version label. Always include a prompt hash in the span attributes so postmortems can distinguish "the model drifted" from "the prompt did not propagate."

**Comparator false positives.** The shadow comparator flags a difference that is semantically irrelevant, such as capitalization. This erodes trust in the alert. Normalize outputs before comparison and calibrate the threshold as described above.

**Golden dataset rot.** The golden dataset was labeled months ago and no longer reflects current business rules. Shadowing then measures drift against an obsolete standard. Treat the golden dataset as a versioned artifact with an owner and a review cadence.

## A decision checklist

Before choosing an approach, answer these questions explicitly and record the answers in the postmortem template.

1. **What is the blast radius of a wrong decision?** A wrong summary costs reviewer time. A wrong approval costs money, health, or safety. Higher blast radius favors shadowing.
2. **How fast does the agent's behavior change?** Weekly changes favor shadowing, because detection latency matters. Monthly changes are usually served by logging.
3. **Is the output space enumerable?** If the agent produces one of five classes, logging plus strict validation covers most cases. If it produces free text, shadowing with semantic match is more robust.
4. **What is the cost of a false positive versus a false negative?** If false positives are expensive, favor shadowing and tune the threshold for precision. If false negatives are expensive, tune for recall.
5. **Does the team have labeled data and the capacity to maintain it?** Shadowing without a maintained golden dataset degrades into noise.
6. **What is the budget for duplicate inference?** Shadowing costs roughly the inference cost of the shadowed traffic fraction. Decide the fraction deliberately.

A reasonable default for teams new to agent observability is to start with structured semantic logging, add strict enum validation, and introduce shadowing when either of two conditions is met: semantic drift exceeds a threshold the team has agreed on, or the blast radius of a wrong decision exceeds an amount the team has agreed on. Both conditions should be stated as numbers in the postmortem template, not left implicit.

## What to record in the postmortem

A semantic postmortem should capture the following, in this order:

- **The decision timeline.** Which requests were affected, and what the agent decided versus what it should have decided.
- **The detection path.** How the failure was found: user report, log query, shadow alert, or audit. Record the time from first affected request to detection. This is the metric to improve.
- **The prompt hash and model version** in effect at the time of the first affected request.
- **The retrieval state.** Which index version was live, and the age of the retrieved documents.
- **The confidence distribution** for affected decisions versus unaffected decisions in the same window.
- **The action items**, each with an owner and a measurable acceptance criterion.

The last point is the one most often skipped. "Improve prompt" is not an action item. "Add a validation rule that rejects `Approve` when `medical_history` is null, verified against the last 30 days of labeled decisions" is.

## FAQ

**Can logging alone catch semantic drift?**

It can catch drift that manifests as a change in the distribution of output classes or confidence values. It cannot catch a novel failure class that has not appeared before, because there is nothing to compare against. Logging is best understood as a forensic tool, not a predictive one.

**How large does a golden dataset need to be?**

Large enough to cover the edge cases that matter. Two hundred to three hundred well-chosen labels are often enough to calibrate a threshold and detect gross drift. The limiting factor is coverage, not count. A dataset of 10,000 labels that all come from the common case will miss the rare failure that matters most.

**What if the agent calls a private model behind an API?**

Shadowing still works, but the shadow instance must call the same model with the same parameters. Mirror the model, temperature, and max token settings exactly. If the model provider does not offer version pinning, record the model identifier returned in the response and alert when it changes.

**Should shadowing run on every request?**

No. Shadowing a fraction of traffic is a deliberate tradeoff between detection latency and cost. A common pattern is to shadow a small percentage continuously and shadow 100% for a bounded window after a prompt or model change.

**What is the single most useful metric to alert on?**

Semantic match rate, if a golden dataset exists. Without one, the most useful alert is on the ratio of terminal decisions made with weak retrieval, which is a leading indicator of semantic failure.

## Do this in the next 30 minutes

Open the file that holds your agent's system prompt and record its hash in a comment or a version field. Then query your structured logs for the last 100 terminal decisions and count how many were made with a retrieval miss ratio above 0.5. If that count is greater than zero, you have a candidate incident to investigate, and you have just established the baseline that a postmortem template needs in order to say anything useful.
