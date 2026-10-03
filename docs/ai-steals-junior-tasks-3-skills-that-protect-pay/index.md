# AI steals junior tasks: 3 skills that protect pay

## The economic problem hiding behind faster code generation

AI coding assistants are genuinely good at a specific class of work: CRUD endpoints, signup flows, form validation, basic dashboards, and glue code between services. A developer who once spent two days on a settings page can often get a working draft in an afternoon. The code compiles, the tests pass, and the deploy pipeline stays green.

The confusion starts there. If the code works and the pipeline is green, why does the business not feel proportionally better off?

The answer is opportunity cost. Engineering time is a fixed budget. Every hour spent generating and reviewing scaffolding is an hour not spent deciding what to build, talking to users, pricing the product, or fixing the one workflow that causes churn. AI makes you faster at the wrong things when the wrong things were never the bottleneck.

This is not a technical failure. It is an allocation failure. The metric that matters is not lines of code per hour or commits per week. It is validated customer value per engineering hour. AI changes the cost of the numerator (code) while leaving the denominator (your attention) fixed.

## What actually causes the trap

Three mechanisms explain most of the pattern.

**First, a mismatch of optimization targets.** Code generation tools optimize for syntactically correct, functionally adequate code that satisfies the prompt. They do not optimize for whether the feature should exist, how it should be priced, or whether users will return. Those decisions remain human work, and they are the work that protects a salary or a runway.

**Second, tool evaluation overhead.** The market for AI coding tools is crowded and changes quickly. A common failure mode is spending several hours a week reading launch posts, testing new assistants, and migrating configuration. That time is invisible in Git history but very visible in a calendar. A useful discipline is to evaluate tools on a fixed schedule rather than continuously, and to require a measurable time saving before adopting one.

**Third, the illusion of progress.** When an assistant writes your API routes, the commit graph fills up quickly. Twenty commits for one feature feels like momentum. But commit count measures activity, not outcomes. The real bottleneck becomes your ability to tell whether the feature moved a metric.

A fourth, quieter cause is environmental. If your editor, indexing, or model round-trip is slow, you experience the assistant as "bad" when the problem is local latency. That leads to tool churn instead of fixing the actual constraint.

## Fix 1: Gate code generation behind a written hypothesis

The most common cause of wasted AI-assisted work is treating the assistant as a junior developer rather than as a multiplier for a senior one. A junior developer needs direction. An assistant with no direction will happily produce a large, well-formed answer to the wrong question.

The concrete rule: before generating code for a feature, write one sentence of the form:

> We believe that [change] will [measurable effect] for [segment] within [time window].

For example: "We believe that adding a usage-based pricing page will increase free-to-paid conversion by 15% within 30 days." If that sentence cannot be written, the feature is not ready to build, regardless of how fast the assistant can produce it.

This habit is cheap and it kills work early. A Stripe integration can be generated quickly and still produce zero conversion change. The hypothesis is what tells you, in advance, what would count as success.

Two supporting practices:

- **Audit the tool stack on a schedule, not continuously.** Quarterly is usually enough. For each tool, ask: does it measurably reduce time on a task I actually do? Is it cheaper than the time it saves? If neither answer is clearly yes, drop it.
- **Cap generated code as a share of your work.** Pick a ceiling, such as 20% of weekly commits, and treat exceeding it as a prompt to review what you are building. The exact number is a judgment call; the discipline of measuring it is the point.

A commit-counting command is a crude but useful instrument. Many assistants record themselves as a co-author, so you can count matching trailers:

```bash
# Count commits in the last week whose message or trailers mention an AI assistant
git log --since="1 week ago" --pretty=format:"%B" \
  | grep -ciE "co-authored-by:.*(copilot|cursor|claude|codex)"

# Total commits in the same window, for the ratio
git log --since="1 week ago" --oneline | wc -l
```

This only works if the assistant actually writes co-author trailers. If it does not, the number will read zero and tell you nothing. A more reliable signal, if you want one, is a lightweight manual tag: add a label to commits you consider AI-dominated and count those. The measurement method matters less than having one you trust.

## Fix 2: Optimize for leverage instead of velocity

Velocity is how fast code gets written. Leverage is how much customer value is produced per hour of attention. They diverge precisely when code generation gets cheap.

The fix is to invert the default workflow: start with customer discovery, and write code only for problems that survive contact with users.

A lightweight loop that fits a solo schedule:

1. Interview two or three users or churned users for fifteen minutes each, weekly.
2. Record their top three pain points verbatim.
3. Rank by frequency and by stated willingness to pay.
4. Build only the top-ranked pain point that the assistant cannot already solve by pattern-matching existing solutions.

The underlying reason is that AI is strong at replicating known solution shapes and weak at choosing which problem is worth solving. Leverage comes from problem selection.

Three levers tend to beat feature work:

**Pricing.** A pricing change can be tested before any new code exists. A common design is a short experiment with a new tier or a changed price point, measured against a conversion or revenue-per-user metric over a fixed window. The important part is the pre-registered metric and window, not the specific price.

**Churn reduction.** Much churn is caused by friction rather than missing features. Small workflow changes — a clearer cancellation flow, a pause option, better notifications — are often cheap to build and can be measured directly against monthly churn rate. The reason to prioritize them is that they address an existing, observed problem rather than a hypothetical one.

**Automating discovery itself.** Survey and session-recording tools let you collect customer signal without writing code. This is leverage: it increases the rate at which you learn, which is the input to every other decision.

## Fix 3: Fix the environment before blaming the assistant

Local latency is a real and frequently misattributed cause of "the AI feels slow." If suggestions take several seconds to appear, the constraint may be your machine, your editor's indexing, or the network round-trip to a hosted model — not the model's quality.

Practical levers, in rough order of impact:

- **RAM and disk.** Large-file indexing and multi-file context are memory- and I/O-heavy. Insufficient RAM or a slow disk shows up as lag during indexing, not just during generation.
- **Local versus hosted inference.** Hosted models add network round-trip time to every suggestion. Local models remove that round-trip but trade it for hardware limits. Which is faster depends on your machine and the model size, so measure rather than assume.
- **Editor configuration.** Project-wide context features are useful but expensive. Excluding build directories, dependency folders, and generated files from indexing often produces a larger speedup than any hardware change.

To measure suggestion latency rather than guess at it, instrument the round trip. For a local model server, a direct request timing is the cleanest signal:

```bash
# Time a single generation request against a local model server
time curl -s http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "write a function that reverses a string",
  "stream": false
}' > /dev/null
```

Run it several times and compare the median, not a single sample. For a hosted assistant, measure the time from cursor placement to suggestion appearance with a stopwatch over ten trials; the median is what you care about. Then change one variable at a time — disable indexing on a large directory, switch to a local model, or add RAM — and re-measure.

A worked example of the reasoning, with illustrative numbers: suppose suggestions currently take a median of 900 ms and you make 200 suggestions a day. That is 180 seconds, or three minutes, of pure waiting. If a change cuts the median to 300 ms, you save two minutes a day. That is real but small. Now suppose the change also removes a two-second pause each time you switch files, and you switch files 60 times a day; that is two minutes saved, on top of the suggestion time. The lesson is that context-switch and indexing costs often dominate generation latency, which is why measuring both matters.

```python
import time
import subprocess

# Illustrative measurement: time to launch the editor, not time to first keystroke.
start = time.time()
subprocess.run(["cursor", "--version"], check=False)
end = time.time()
print(f"Editor launch time: {end - start:.2f}s")
```

Launch time is a rough proxy at best. For a real context-switch measurement, record the timestamp when you leave a task and when you begin editing the next file, and log both.

## How to verify any of this worked

The verification problem is the same as the original problem: activity metrics are easy and outcome metrics are hard. Track a small set, weekly, and define them before you start.

| Metric | What it measures | How to instrument it |
|---|---|---|
| AI-assisted commit share | Fraction of work that is generated scaffolding | Count commits tagged or co-authored by an assistant, divide by total commits |
| Feature validation rate | Fraction of shipped features that moved a pre-registered metric | Keep a log: feature, hypothesis, metric, window, result |
| Customer interviews per week | Rate of learning from users | Calendar count |
| Hypothesis win rate | Fraction of hypotheses confirmed within their window | Same log as above |
| Suggestion latency (median) | Local environment health | Time ten suggestion round trips, take the median |
| Revenue per engineering hour | The outcome that matters | Monthly revenue divided by tracked engineering hours |

Two cautions. First, correlation between a falling AI-commit share and a rising validation rate is not proof of causation; both may be driven by a change in what you chose to work on. Second, small samples are noisy. A single month of feature results will not distinguish a real improvement from luck. Treat these as directional signals and keep the log long enough to see a trend.

## Prevention: make the discipline structural

Habits decay. Structure does not. Three lightweight mechanisms:

**A hypothesis gate in the pull request template.** Every PR includes a one-sentence hypothesis and a validation plan. This forces the question before the code exists, when it is still cheap to say no.

```markdown
## Hypothesis
Adding a one-click pause option will reduce monthly churn by at least 10% within 30 days.

## Validation plan
Track monthly churn for 30 days after release. Compare against the trailing 90-day baseline.
```

**A scheduled tool audit.** Once a quarter, list every AI tool you pay for. For each, write the task it speeds up, the measured time saving, and the cost. Drop anything that cannot answer all three.

**A monthly interview block.** Two hours on the calendar, two or three conversations, notes in one place. The goal is not to build anything; it is to keep the input to every other decision fresh.

## Related failure modes

- **The over-optimization trap.** Endless environment tuning with no measured improvement. Fix: timebox any optimization to two hours and require a before/after measurement.
- **Assistant drift.** An assistant keeps suggesting patterns your project has moved away from. Fix: keep a project rules file that states the conventions explicitly, and review it when the stack changes.
- **Interview burnout.** Discovery stops because it feels repetitive. Fix: rotate across new, churned, and power users, and keep sessions to fifteen minutes.
- **Experiment fatigue.** Too many concurrent pricing or onboarding tests confuse users and muddy results. Fix: one experiment at a time, with a pre-registered metric.

```text
# Example project rules file (place at the repository root, name per your editor's convention)
# This project uses GraphQL for client-server communication.
# Prefer GraphQL mutations and queries.
# Avoid introducing REST endpoints unless explicitly requested.
```

## When the fixes are not enough

If the metrics still stagnate after several months, the problem is probably upstream of engineering:

1. **Re-examine pricing.** A price that is too low caps revenue regardless of feature quality. Test willingness to pay directly rather than inferring it.
2. **Test retention levers.** Add a short survey to the cancellation flow. Friction fixes often outperform new features.
3. **Audit the funnel.** Instrument onboarding and look for drop-off points. Fixing a drop-off can raise conversion without new code.
4. **Consider a scope change.** A narrower product aimed at a specific audience may be more defensible than a broad one. This is a market decision, not an engineering one, and AI does not change it.

## FAQ

**Why can AI make me faster without making the business more profitable?**
Because speed at code generation only helps if code generation was the constraint. When the constraint is choosing what to build, faster generation increases output of the wrong thing. Writing a hypothesis before generating code is the cheapest way to check which constraint you are under.

**How do I tell whether an AI tool is actually saving time?**
Pick one recurring task, measure how long it takes with and without the tool over several instances, and compare medians. If the saving is not visible in that measurement, the tool is probably costing more in evaluation and context-switching than it returns.

**What is the fastest way to start customer-driven development?**
Book two or three fifteen-minute conversations this month with users or churned users. Ask open-ended questions about difficulty and willingness to pay. Record answers in one document and look for repeated themes. This costs less time than building a single feature.

**Is a hardware upgrade worth it for AI latency?**
Only if you have measured the latency and identified the hardware as the cause. Measure suggestion latency and context-switch time first. If indexing or disk I/O dominates, configuration changes may help more than a new machine.

**Do commit-count metrics actually work?**
They are a rough proxy, and they fail entirely if your assistant does not record co-author trailers. Use them as a directional signal, not a target, and prefer a manual tag if you need reliability.

## Next 30 minutes

Open your repository and add a hypothesis section to the pull request template, then write the hypothesis for the next feature you were about to generate code for. If you cannot write a sentence that names a metric, a segment, and a time window, do not start the feature — schedule a fifteen-minute user conversation instead.
