# Prompts are code: how to measure their value

Most guidance on measuring prompt quality either skips the hard part or measures the wrong thing. The failure mode is rarely a missing dashboard; it is a metric that looked reasonable and answered a question nobody needed answered. This article covers how to instrument prompts and agent graphs so that a change can be classified as an improvement or a regression, without building a second product to do it.

## The problem, in general terms

When a growing share of a product's behaviour lives in prompts, tool schemas, and agent graphs rather than in functions and classes, standard engineering metrics stop telling you much. Lines of code, cyclomatic complexity, and test coverage do not map cleanly onto a system where the logic is a paragraph of English and the control flow is a graph of model calls. A reviewer can see the diff, but cannot easily answer the question that matters: did this change make the product better, or just different?

The temptation is to reach for metrics that already exist. Count tokens. Count prompt versions. Count agent steps. These are easy to instrument and almost useless on their own, because they measure activity rather than value. A prompt that takes twice as many tokens but resolves a support ticket on the first turn is not worse than a terse prompt that loops the user through three clarifications. An agent graph with twelve nodes is not better than one with four — it is just harder to debug at 2am.

The underlying difficulty is that prompts and agent graphs have no natural unit of correctness. A function either returns the right value or it does not. A prompt returns a distribution of outputs, and "right" depends on context the model cannot see. Any measurement system therefore has to start by deciding what counts as a good outcome, then work backwards to the prompt. The rest of this article is about building that loop without turning it into a second full-time job.

## The approaches that commonly fail, and why

The first failure mode is measuring the model instead of the product. Teams commonly set up an eval harness that scores outputs against a golden dataset, watch the score go up, and ship. The score is real, but it is a proxy for a proxy. If the golden dataset was written by the same person who wrote the prompt, it encodes the same blind spots. A prompt can score well on a self-authored eval and still fail the first real user who phrases the request in a way the author never imagined.

The second failure mode is measuring cost in isolation. Token spend is easy to attribute and easy to report, so it becomes the default metric. A prompt that costs half as much per call looks like a win until it starts triggering retries, or producing outputs that need human cleanup. Cost per call is a real number, but it is only meaningful next to a completion rate. Optimising cost without holding completion constant is how teams end up with a cheaper system that costs more to operate.

The third failure mode is treating the agent graph as architecture rather than as a hypothesis. A graph with a planner node, a retrieval node, a critic node, and a synthesis node feels rigorous. It is also four places for the model to hallucinate, four prompts to maintain, and four chances for a timeout to cascade. The graph is a hypothesis about how the task decomposes. If the hypothesis cannot be stated in one sentence, the graph is probably compensating for a prompt that is doing too much or too little.

The fourth failure mode is versioning prompts without versioning the inputs they depend on. A prompt that says "summarise the document" behaves differently when the retrieval layer starts returning three chunks instead of one. The prompt did not change, but its behaviour did. Teams that only track prompt versions miss this class of regression entirely.

A fifth failure mode is less discussed: measuring at the wrong boundary. If instrumentation is added inside the agent graph — per node, per retry, per tool call — the resulting metrics describe the internals of a system that the user never sees. A retry loop that eventually succeeds looks like a failure at the node level and a success at the user level. Both are true, but only one of them is the number that should drive a shipping decision.

## The approach that works in practice

The approach that holds up is to define a small number of outcome metrics tied to user-visible behaviour, instrument them at the boundaries of the system, and treat prompts and graphs as implementation details behind those boundaries. This is not novel — it is normal product instrumentation applied to a new kind of component.

Start by naming the outcomes. For a support agent, that might be: resolved without human handoff, resolved within one turn, and user did not rephrase. For a code assistant, it might be: suggestion accepted, suggestion accepted without edit, and suggestion led to a passing test. For a research agent, it might be: answer cited a source that exists, answer matched the user's question, and answer was delivered within a latency budget. Three to five outcomes is usually enough. More than that and nobody looks at the dashboard.

Then instrument the boundaries. The boundary is wherever code hands control to the model and gets something back. That is the only place that needs logging, because it is the only place where both the input and the outcome are observable. Everything inside the agent graph — the planner, the critic, the retry loop — is an implementation detail. Logging at the boundary allows outcomes to be attributed to the prompt version and the graph version that produced them without instrumenting every node.

Finally, make the prompt and graph versions part of the log record. Not in a comment, not in a git tag, but as a field in the structured log line that also carries the outcome. This is the single change that makes the whole system work, because it lets the question "did the prompt change help?" be answered with a query instead of a memory.

## Implementation details

A minimal pattern in Python follows. It assumes some function that calls a model and returns a result, and some way to record whether the result was good. The point is not the specific library — it is the shape of the record.

```python
import json
import time
import uuid
from dataclasses import dataclass, asdict

@dataclass
class BoundaryRecord:
    trace_id: str
    prompt_id: str
    graph_id: str
    outcome: str          # "resolved", "handoff", "rephrased"
    latency_ms: int
    input_tokens: int
    output_tokens: int
    model: str

def call_with_trace(prompt_id, graph_id, model, messages, outcome_fn):
    trace_id = str(uuid.uuid4())
    start = time.monotonic()

    # This is the boundary. Everything inside is opaque to the metric.
    result = model.invoke(messages)

    latency_ms = int((time.monotonic() - start) * 1000)
    outcome = outcome_fn(result)  # returns a string label

    record = BoundaryRecord(
        trace_id=trace_id,
        prompt_id=prompt_id,
        graph_id=graph_id,
        outcome=outcome,
        latency_ms=latency_ms,
        input_tokens=result.usage.input_tokens,
        output_tokens=result.usage.output_tokens,
        model=model.name,
    )
    log_line = json.dumps(asdict(record))
    # Write to whatever sink you already have: stdout, a file, a table.
    print(log_line)
    return result
```

The `outcome_fn` is the interesting part. It decides what counts as a good result. In practice it is often a small classifier, a regex, or a human review queue. It does not have to be perfect. It has to be consistent, so that changes in the outcome rate reflect changes in the system rather than changes in the judge.

Once these records exist, the queries that matter are simple. For each `prompt_id`, what is the outcome distribution? For each `graph_id`, what is the latency distribution? When `prompt_id` changed from A to B, did the outcome distribution shift? These are the same queries that would be run on any A/B test, because that is what this is.

The second implementation detail is a prompt registry. A service is not required. A directory of files with a naming convention is enough.

```
prompts/
  support_classify/
    v1.txt
    v2.txt
    current -> v2.txt
  support_reply/
    v1.txt
    current -> v1.txt
```

The `current` symlink is what the application reads. To change a prompt, add a new version and repoint the symlink. The `prompt_id` in the log record is the directory name plus the version, so a record can always be mapped back to the exact text that produced it. This is boring, and boring is the point. A prompt registry that requires a deploy to change is a prompt registry nobody uses.

The third detail is a graph registry, the same idea applied to the agent graph. A graph is a set of nodes and edges, and the version is a hash of the graph definition. If a framework serialises graphs, hash the serialised form. If the graph is hand-rolled, hash the node names and edges. The hash goes in the log record as `graph_id`. When the graph changes, the hash changes, and outcomes can be compared across hashes.

## A worked example

Consider a support agent that classifies incoming tickets into one of six categories and then drafts a reply. Suppose the team notices that handoff rate — the share of tickets escalated to a human — has crept up. The instrumentation described above makes it possible to localise the cause.

The relevant log records carry `prompt_id` and `graph_id`. A first query groups by `prompt_id` and shows the outcome distribution. Suppose `support_classify/v2` is producing handoffs at twice the rate of `support_classify/v1`. That is a strong signal, but it is not yet a conclusion, because the graph may also have changed between the two periods.

A second query groups by `graph_id` and shows that the graph hash is unchanged. Now the prompt is the only variable that moved. The next step is to read the v1 and v2 texts side by side and look for a wording difference that could plausibly affect classification. A common culprit is a change from an explicit list of categories to a fuzzier instruction like "use your judgement." The model, given latitude, hedges toward the handoff category when it is uncertain, and handoff rate rises.

To confirm, the team can revert the symlink to v1, wait for the outcome distribution to move back, and then reapply v2 with the explicit list restored. This is a two-week experiment in a low-volume system, and it is only possible because the `prompt_id` field was present in the logs before the change was made. Without it, the same investigation would have been a memory exercise across several people.

The arithmetic here is worth stating explicitly. If the agent handles 400 tickets per week and the handoff rate moves from 8% to 16%, that is an increase of 32 handoffs per week, or roughly 4.6 additional human handoffs per day. At 10 minutes per handoff, that is 46 minutes of human time per day, or about 3.8 hours per week. These figures are illustrative — substitute the actual ticket volume and handling time for the system in question. The point is that a small percentage shift in an outcome metric translates into a concrete operational cost, which is what makes it worth tracking.

## What outcomes to expect, and their limits

With this instrumentation in place, a typical deployment of a small support agent will start to show patterns within a few hundred traces. The most common pattern is that a small number of prompt versions account for most of the variance in outcomes. This is useful because it tells you where to spend your time. The second most common pattern is that latency and outcome are correlated in one direction: faster responses tend to have higher completion rates, up to a point. This is a real effect, but it is easy to over-read. A fast wrong answer is still wrong.

The limits are worth stating plainly. First, outcome labels are noisy. If the `outcome_fn` is a human reviewer, there is a labelling bottleneck. If it is a classifier, there is a classifier that can drift. Either way, the signal is weaker than a unit test. Treat outcome rates as trends, not as pass/fail gates.

Second, the instrumentation only covers the boundary that was chosen. If a failure happens inside the agent graph — a retry loop that eventually succeeds but costs three times as much — the boundary record will show a success. A separate cost metric is needed if that matters, and it usually does. The boundary record should carry token counts for this reason.

Third, prompt and graph versions are not independent. Changing the graph changes the effective prompt, because the same prompt text behaves differently when it is called by a planner versus a critic. This means an outcome shift cannot be cleanly attributed to one or the other. The practical answer is to change one thing at a time when possible, and to accept that when it is not, the measurement describes the system, not the component.

## How to measure this on your own system

The instrumentation described here is only useful if it produces numbers a team can act on. A few concrete measurement recipes:

To measure the effect of a prompt change, compare the outcome distribution for the old `prompt_id` against the new one over a matched time window. If the graph hash is unchanged, the comparison is close to an A/B test. If the graph hash changed too, the comparison is confounded and should be treated as descriptive only.

To measure the cost impact of a prompt change, sum `input_tokens` and `output_tokens` per `prompt_id` and divide by the number of traces. Report this alongside the completion rate, never on its own. The useful derived number is tokens per resolved outcome, not tokens per call.

To measure latency regression, compute the p50 and p95 of `latency_ms` per `graph_id`. A graph change that improves outcomes but doubles p95 latency is a trade-off, not a win, and the trade-off should be visible in the dashboard rather than discovered by users.

To measure outcome-label drift, periodically sample traces and have a human label them independently. Compare the human labels against the `outcome_fn` output. A rising disagreement rate means the outcome signal is decaying and needs to be re-grounded.

None of these require a platform. A log sink, a query tool, and a weekly review are sufficient.

## What to watch out for

The first trap is building the measurement system before there is a product worth measuring. A small team with ten users does not need a prompt registry. They need to read the ten conversations and fix the obvious problems. The instrumentation described here pays off when volume is high enough that reading every trace is impractical, which is usually somewhere in the hundreds to low thousands of interactions per week. Before that, manual review is faster and more accurate.

The second trap is letting the outcome metric become the goal. If the metric is "resolved without handoff", the system will learn to avoid handoff even when handoff is correct. This is the same dynamic as any other metric, and the same defence applies: keep a second metric that measures the thing you are afraid of losing. For a support agent, that might be "user did not reopen the ticket within seven days". For a code assistant, "suggestion accepted without edit" paired with "suggestion led to a passing test".

The third trap is versioning prompts in a way that makes rollback hard. If the prompt registry is a directory of files, rollback is repointing a symlink. If the prompt registry is a database table edited through an admin UI, rollback is a migration. Prefer the file-based approach until there is a reason not to. The reason not to is usually that non-engineers need to edit prompts, and that is a real reason — but it is a reason to add a UI on top of the files, not to replace them.

The fourth trap is assuming that a better model will fix a bad prompt. It sometimes will, but the improvement is not predictable, and it does not change the fact that there is no way to tell whether the new model helped. The instrumentation is what lets that question be answered. Without it, a model upgrade is a leap of faith.

A fifth trap is over-instrumenting. Every field added to the log record is a field that must be maintained, migrated, and explained to new team members. The record shown earlier has eight fields, and that is close to the practical minimum for the queries described here. Resist adding fields until a specific query requires them.

## A decision checklist

Before committing to the full instrumentation described here, it is worth checking the following:

- Can three to five user-visible outcomes be named for this product? If not, the problem is product definition, not measurement.
- Is there a single place in the code where the model call returns and the outcome is knowable? If not, the boundary needs to be identified before anything else.
- Is there a log sink that already exists? If yes, add fields to it. If no, a single JSON line to stdout is enough to start.
- Is the volume high enough that reading every trace is impractical? If not, manual review is the better investment.
- Is there a person who will look at the numbers weekly? If not, the dashboard is decoration.

If the answer to the last question is no, fix that first. Instrumentation without a reader is just storage cost.

## The broader lesson

The broader lesson is that prompts and agent graphs are code, and like any code they need a way to tell good changes from bad. The unit of measurement is not the prompt or the graph — it is the outcome at the boundary where the system meets the user. Everything else is implementation.

This is a reversal of the usual instinct. The usual instinct is to measure what is easy: tokens, steps, latency, cost. Those are real and worth watching, but they are inputs, not outcomes. A system that uses more tokens and produces a better result is a better system. A system that uses fewer tokens and produces a worse result is a worse system. The only way to know which one is in front of you is to measure the result.

The second part of the lesson is that the measurement does not have to be sophisticated to be useful. A structured log line with a prompt version and an outcome label, written to a file, queried with a script, is enough to answer most of the questions a small team actually has. A platform is not required. A consistent record and a willingness to look at it are.

## How to apply this to your situation

Start by writing down three outcomes that matter for the product. Not metrics — outcomes. "The user got an answer they did not have to rephrase." "The suggestion compiled." "The ticket did not come back." Then find the one place in the code where the model call returns and the outcome is knowable, and add a log line there with the prompt version and the graph version.

If logs already exist, add two fields to the existing record: `prompt_id` and `graph_id`. If they do not, add a single `print` statement that emits JSON. The goal is to be able to run one query — "what is the outcome distribution for prompt version X versus version Y?" — by the end of the week.

## Frequently Asked Questions

### How do I measure prompt quality without a golden dataset?

A golden dataset is not required to start. A consistent outcome label is. The simplest version is a human review queue: every hundredth trace gets read by a team member, and the outcome is recorded as good or bad. This gives a noisy but unbiased signal. A golden dataset is useful later, when automated evals need to run in CI, but it is not the first step. The first step is deciding what "good" means and writing it down.

### What is the difference between a prompt version and a graph version?

A prompt version is the text of a single prompt. A graph version is the structure of the agent — which nodes exist, how they are connected, and what each node does. They are separate because they change for different reasons. A prompt is changed to fix a wording problem. A graph is changed to fix a decomposition problem. Logging both lets a team tell which kind of change caused an outcome shift, which is the whole point.

### Why not just use an LLM to grade every output?

This is possible, and many teams do it. The problem is that the grader is itself a model with its own failure modes, and it can drift as the model behind it is updated. If an LLM grader is used, pin the grader version the same way prompt versions are pinned, and periodically spot-check its labels against human review. Treat it as a noisy sensor, not as ground truth.

### When should I stop measuring and just ship?

When the cost of measurement exceeds the cost of being wrong. For a pre-product team, that is almost immediately. For a product with paying users and a support burden, measurement pays for itself quickly. The signal to start is when the question "did that change help?" can no longer be answered from memory. That is the point where a log line is cheaper than a guess.

## One thing to do in the next 30 minutes

Open the file where the application makes its main model call, and add two fields to whatever is logged there: a `prompt_id` string and a `graph_id` string. They can be hardcoded for now. The point is to have the fields in place before the next prompt change, so that when the question "did that help?" comes up, there is something to query.
