# AI code traps: Copilot vs Cursor smackdown

## The failure mode that matters most

AI coding assistants are good at producing code that satisfies the happy path. They are much less reliable at producing code that survives real input. The characteristic failure looks like this: a completion parses a field as an integer because the surrounding code and the prompt both imply integers, and nothing in the visible context says the field can be empty, carry a leading zero, or arrive with a `+` sign. The suggestion compiles, passes a unit test written against the same assumption, and fails the first time production sees a value the author never imagined.

This is not a defect of one vendor. It is a property of next-token prediction over a context window: the model optimizes for the most plausible continuation of what it can see, and edge cases are by definition the implausible continuations. Any evaluation of an assistant that does not test edge-case behavior is measuring autocomplete speed and calling it quality.

The rest of this article covers what to measure, how to measure it, and how to turn the measurements into a decision. It avoids vendor scorecards because those go stale and because the right answer depends on constraints that vary enormously between teams.

## Categories of assistant, not brands

It helps to reason about deployment shapes rather than product names, because the tradeoffs follow the shape.

**Hosted completion service.** The editor sends your context to a vendor endpoint and receives a completion. Latency is dominated by network round-trip plus model inference; on a decent connection, single-line completions typically return in well under a second, and multi-line generations take longer. The vendor handles model hosting, so client hardware requirements are low. The costs are a per-seat subscription, a dependency on outbound HTTPS, and whatever data-processing agreement the vendor offers. Data-residency and compliance review usually happens here.

**Local model runner.** The editor runs a quantized model on the developer's machine. There is no per-request cost and no network dependency, which matters for intermittent connectivity. The costs are RAM, CPU or GPU load, battery drain, and latency that is one to two orders of magnitude worse than a hosted service on the same task. A quantized 7B-parameter model on a laptop without a discrete GPU is usable for short completions and painful for whole-file generation.

**Hybrid.** The editor routes some requests locally and some to a hosted endpoint, often with a setting to force offline mode. This is the most flexible shape and the hardest to reason about, because behavior changes with connectivity and the user may not notice which path a given completion took.

When comparing two products, first establish which shape each one is in for your configuration. A local-mode comparison against a hosted service is a comparison of deployment shapes, not of model quality.

## What to measure, and how

### Edge-case suggestion quality

This is the measurement most teams skip and the one that predicts production incidents.

Build a small evaluation set from your own bug history. For each historical bug caused by unhandled input, write a prompt that a developer would plausibly type, and record whether the assistant's first suggestion handles the edge case. Ten to twenty prompts is enough to see a pattern. Score each suggestion as correct, correct-but-fragile, or wrong, and record the specific failure.

Concrete prompt families worth including:

- Parsing a field that may be empty, signed, or zero-padded.
- Handling a filename that may contain non-ASCII characters or characters illegal on the target filesystem.
- Processing a delimited file with embedded delimiters, quoted fields, or a byte-order mark.
- Consuming a callback endpoint that may receive duplicate or out-of-order deliveries.
- Retrying a network call where the first attempt may have succeeded server-side.

The output of this exercise is a table of prompt, suggestion verdict, and failure description. It is not a benchmark score, and it should not be reported as one. It is a locally valid signal about whether the assistant tends to reach for `int(value)` or for a parse with an explicit error path.

### Latency

Latency is measurable and worth measuring on your own hardware, because published figures rarely match a specific machine.

Instrument it by timestamping the moment you trigger a completion and the moment text appears. A stopwatch over twenty repetitions gives a usable median and a rough sense of the tail. If you want percentiles rather than a median, log the timestamps to a file and compute them. Record separately for single-line completion, multi-line function generation, and whole-file or multi-file operations, because the ratios between these differ sharply between hosted and local modes.

Measure on the weakest machine your team actually uses, not the strongest. A completion that takes 200 ms on a workstation can take several seconds on an older laptop, and that difference changes whether developers keep the tool enabled.

### Resource footprint

On Linux, `ps` or `top` sampled during an editing session gives resident memory and CPU for the editor process and any model-runner process. On macOS, Activity Monitor serves the same purpose. Sample at rest, during typing, and during a long generation. Note whether the machine's fans spin up and whether battery life changes noticeably over a working day.

The relevant question is not the absolute number but whether the tool leaves enough headroom for the rest of the toolchain — a container runtime, a database, a browser with many tabs. On an 8 GB machine, a local model runner that peaks above roughly 2 GB competes directly with everything else.

### Cost

Cost has three components that are easy to conflate:

- Seat or subscription cost, per developer per month, multiplied by headcount and months.
- Inference cost, if the assistant is metered rather than included in the seat.
- Hardware cost, if local inference requires a RAM or storage upgrade, amortized over the machine's useful life.

Write the arithmetic out explicitly for your own numbers. For example, a team of four developers on a subscription at a stated monthly rate, over twelve months, is four multiplied by the monthly rate multiplied by twelve. If a local-runner option requires a memory upgrade costing a known amount on two of the four machines, add that once, not monthly. Presenting the total as a single figure without showing the multiplication hides which input dominates, and the dominant input is usually seats, not inference.

Compliance review has a cost too, in engineering and legal time, and it is often the deciding factor regardless of the other numbers.

## A worked example: the zero-padded identifier

Consider a service that receives records from an upstream system and looks them up by identifier. The upstream system emits identifiers as zero-padded strings of fixed width, for example `0001234`. A plausible prompt to an assistant is: "write a function that takes a record dict and returns the matching row from the database."

A common suggestion is to read the identifier and pass it directly to a query. A variant is to coerce it with `int()` before querying. Both look reasonable. The failure appears when the upstream system changes its padding, when a record arrives with an identifier that is not numeric at all, or when the database column is a text type and the coercion changes the value's representation.

A more robust suggestion keeps the identifier as a string, validates its shape, and fails loudly on mismatch:

```python
import re

ID_PATTERN = re.compile(r"^[0-9]{7}$")

class InvalidIdentifier(ValueError):
    pass

def extract_identifier(record: dict) -> str:
    raw = record.get("id")
    if not isinstance(raw, str):
        raise InvalidIdentifier(f"id must be a string, got {type(raw).__name__}")
    candidate = raw.strip()
    if not ID_PATTERN.match(candidate):
        raise InvalidIdentifier(f"id {candidate!r} does not match expected format")
    return candidate
```

Note what this does and does not do. It does not strip leading zeros, because the upstream format defines them as significant. It rejects non-string input rather than coercing, because a silent coercion is exactly the failure mode being guarded against. It raises a specific exception type so callers can distinguish a format problem from a database miss.

The evaluation question for an assistant is not whether it can produce this function when asked directly. It is whether it produces something like this when asked the vague version, without being told that padding, emptiness, and type are concerns. That is the difference that shows up in production.

## Failure modes to watch for

**Silent coercion.** `int()`, `float()`, `bool()`, or a truthiness check applied to a value whose type is not guaranteed. The code does not raise; it produces a wrong answer.

**Broad exception handling.** A `try` block wrapped around a large region with a bare `except` that logs and continues. This converts a loud failure into a quiet one and makes the eventual incident harder to diagnose. Assistants suggest this pattern frequently because it makes the surrounding code appear to work.

**Assumed nullability.** Code that treats a missing key and a key with a null value as the same thing, or that assumes a field is always present because it is present in the examples the model saw.

**Encoding assumptions.** Reading or writing files with an implicit encoding, or assuming that a path is representable in the current locale. This fails first on non-ASCII filenames and on systems with a non-UTF-8 default.

**Resource leaks in loops.** Appending to a list that is never cleared, opening connections without closing them, or accumulating results inside a loop that runs many times per request.

**Concurrency assumptions.** A fix that is correct single-threaded and wrong under concurrent access, typically because it introduces a check-then-act sequence.

The practical defense is a small set of lint rules or a review checklist that flags these patterns in generated code before merge. Teams that adopt assistants without adding such a gate tend to see the same class of bug repeatedly.

## A decision checklist

Work through these in order. The early items are usually decisive.

1. **Compliance.** Does the work involve data that cannot leave a jurisdiction or that falls under a specific regulatory regime? If yes, the assistant must be one whose data-processing terms cover that case, and this constrains everything else.
2. **Connectivity.** Do developers work for extended periods without reliable internet? If yes, a local or hybrid shape is required, and latency expectations must be set accordingly.
3. **Hardware.** What is the weakest machine in active use, and how much memory can the assistant consume without competing with the rest of the toolchain? If the answer is under roughly 2 GB of headroom, local inference is likely to be unpleasant.
4. **Language mix.** Does the repository mix languages with different conventions for null, errors, and types? If yes, weight edge-case evaluation heavily, because generic suggestions travel badly across languages.
5. **Repository size.** How long does it take a developer to find the relevant file today? If cross-file navigation is already a bottleneck, a tool with fast symbol search has outsized value.
6. **Evaluation result.** Run the edge-case prompt set from the section above. This is the only item that measures the thing you actually care about.
7. **Cost arithmetic.** Write out seats, inference, and hardware for your own headcount and timeline.
8. **Pilot.** Run both candidates on real tasks for two weeks with the same developers, and record which suggestions were accepted, rejected, and later reverted. Reverts are the most informative signal.

The checklist does not produce a universal winner. It produces a defensible choice for a specific team, which is the only kind of choice available.

## When the answer is neither

There are configurations where no assistant is the right call, or where the assistant should be restricted to a narrow role.

If the project is greenfield and small, the fastest path is often to write the input validation and the tests first, then let an assistant fill in the mechanical parts. The assistant is being used for what it is good at, and the parts where it is weak are already covered.

If the budget cannot absorb a per-seat subscription and the hardware cannot run a local model, the honest answer is that the tool is not affordable yet. Adopting it anyway and relying on an individual developer's personal license creates a compliance and continuity problem.

If outbound API access is blocked or unreliable, a hosted service is not viable regardless of its quality, and the evaluation should be limited to local options.

## A 30-minute first step

Open the repository you work in most. Pick the three most recent bugs whose root cause was unhandled input, and for each one write a single-sentence prompt that a developer might have typed just before introducing the bug. Then, with your current assistant enabled, type each prompt into a scratch file and record the first suggestion verbatim, without correcting it.

You now have a three-row table: prompt, suggestion, and your verdict on whether it handles the edge case. That table is a real measurement of the thing that determines whether the assistant helps or hurts this codebase. Save it, add to it whenever a new input-related bug appears, and use it the next time you evaluate a change of tooling.
