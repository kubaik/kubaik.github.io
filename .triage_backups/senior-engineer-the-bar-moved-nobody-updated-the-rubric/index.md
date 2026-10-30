# Senior engineer: the bar moved, nobody updated the rubric

Hiring rubrics are lagging indicators. They get written after a role stabilises, then ossify for years while the actual job drifts underneath them. The common version of "senior engineer" has drifted further than most rubrics admit, and the drift is largely downstream of AI coding tools becoming a normal part of the development loop.

The uncomfortable part isn't that AI writes code. It's that the specific skills hiring panels used as proxies for seniority — producing a large volume of correct code, recalling API surface, debugging by reading a stack trace line by line — are the exact skills AI has commoditised hardest. Meanwhile the skills that now separate a senior from a mid-level engineer are ones most interview loops don't test for, because until recently they were hard to test.

This article covers what actually changed in the job, why the common rubric responses ("add an AI section", "ban AI in interviews") mostly fail, and how to rewrite a rubric so it measures the work that now exists.

## The proxy skills collapsed

For most of the last fifteen years, a senior engineer interview was a proxy for a set of capabilities that correlated with on-the-job performance:

- **Recall** — knowing the standard library, the framework idioms, the common algorithms.
- **Production** — writing syntactically correct, reasonably idiomatic code under time pressure.
- **Local debugging** — reading a stack trace, forming a hypothesis, bisecting.
- **Translation** — turning a vague ticket into a concrete implementation.

Every one of those is now partly or largely offloaded. An engineer with a good editor and a capable model gets recall for free, gets a first draft for free, gets a plausible hypothesis for a stack trace for free, and gets a scaffolded implementation from a ticket description for free.

That doesn't make the skills worthless. It makes them non-diagnostic. When everyone can produce a working first draft, "produced a working first draft" stops separating candidates. The rubric keeps scoring it because the rubric was written when it did separate them.

The counterintuitive part is that this makes the interview *harder* to design, not easier. When a signal is scarce, it's easy to measure. When a signal is abundant, you have to measure the thing that's actually scarce now, and that thing is much less visible in a 45-minute coding exercise.

## What actually got scarce

The scarce capabilities cluster around judgement and verification, not production. Concretely, in roughly the order they matter:

1. **Specifying the problem well enough to delegate it.** A model given a vague instruction produces a plausible-looking answer to the wrong question. The engineer's job is increasingly to write the specification — constraints, invariants, edge cases, what "done" means — precisely enough that the output is checkable. This is a writing skill as much as a technical one.

2. **Reviewing generated code for the failure modes generators don't see.** Models are good at the common path and bad at the specific. They will happily produce code that passes tests and violates a data-residency constraint, or that introduces a dependency with a licence you can't ship, or that quietly logs a field you're not allowed to log.

3. **Knowing what you don't know.** The dangerous output isn't the obviously broken one; it's the confident, fluent, subtly wrong one. Detecting that requires holding the domain model in your head well enough to notice when the code contradicts it.

4. **Ownership of the outcome, not the artefact.** When code is cheap to produce, the value shifts to whoever is accountable for it working in production, under load, within regulation, at 3am.

5. **Taste about what to build at all.** The cheapest time to delete a feature is before it's generated. Engineers who can say "we shouldn't build this" with a defensible reason are worth more than ones who can build anything quickly.

Notice that four of those five are about *not* producing code. A rubric that scores candidates on how much code they produce is measuring the thing that got cheap.

## Why "add an AI section" fails

The most common rubric patch is to bolt on a section: "uses AI tools effectively." It fails for the same reason "uses Git effectively" would fail — it's not a discriminator, it's a baseline. Everyone uses it. The interesting question isn't *whether* a candidate uses a model, it's what they do when the model is wrong, and that's a behaviour, not a checkbox.

A second common patch is to ban AI in interviews and test raw ability. This has a real appeal — it's easy to administer and produces comparable scores. The problem is construct validity: you're measuring a skill the job no longer requires in isolation. You'll select for engineers who are good at an increasingly rare mode of work, and reject engineers who are excellent at the common mode. That's a defensible choice only if a team genuinely believes the job will revert, which is a hard bet to justify.

A third patch is to make the exercise harder — bigger scope, more constraints, tighter time. This mostly measures throughput under pressure, which is the axis AI already flattened. It selects for the same people the old rubric selected for, just more so.

## A rubric that measures the new work

The useful reframe: stop scoring the artefact, start scoring the *process around the artefact*. Below is a structure that has become common in teams that have actually adjusted. It's not the only valid one, but it's concrete enough to argue with.

| Old signal | Why it broke | Replacement signal |
|---|---|---|
| Wrote correct code from scratch | First drafts are now cheap | Can specify constraints precisely enough to delegate |
| Recalled API surface | Retrieval is free | Can spot when generated code contradicts the domain model |
| Debugged a stack trace unaided | Hypothesis generation is cheap | Can design a test that falsifies a wrong hypothesis |
| Translated a ticket into code | Scaffolding is free | Can identify which tickets shouldn't be built |
| Knew the framework idioms | Idioms are in the model | Can review for licence, privacy, and residency constraints |

Each replacement signal is harder to assess in an hour. That's the cost of the change, and pretending otherwise is how rubrics end up measuring the wrong thing while feeling rigorous.

## Making the signals observable

You can't score "judgement" directly. You have to construct a task where judgement leaves a visible trace. A pattern that works is the **constrained review exercise**: give the candidate a small, plausible, AI-generated change and ask them to review it.

Here is a minimal example. Assume a service that stores customer records, and a change that adds a "notes" field to an API response.

```python
# generated_change.py
# Reviewer's task: decide whether this is safe to merge.

from fastapi import FastAPI
from pydantic import BaseModel
import logging

app = FastAPI()
logger = logging.getLogger(__name__)

class CustomerNote(BaseModel):
    customer_id: str
    note: str
    author_email: str

def fetch_note(customer_id: str) -> CustomerNote:
    # In the real service this hits the primary datastore.
    row = _load_row(customer_id)
    return CustomerNote(
        customer_id=row["id"],
        note=row["note"],
        author_email=row["author_email"],
    )

@app.get("/customers/{customer_id}/note")
def get_note(customer_id: str):
    note = fetch_note(customer_id)
    logger.info("served note for %s by %s", note.customer_id, note.author_email)
    return note
```

The code runs. It passes a naive test. It is also a problem for a service handling European users' data, for at least three reasons a senior reviewer should catch:

1. **`author_email` is personal data** and is now returned in an API response and written to logs. Logging it is almost certainly unnecessary for the stated purpose and creates a processing record that now has to be justified and, on request, erased.
2. **The log line includes a customer identifier** alongside the email, which links the two — that's a stronger data linkage than either field alone.
3. **There's no access control shown** — the endpoint takes a `customer_id` and returns a note. Whether that's an authorisation bug depends on the surrounding code, but the reviewer should *ask*, not assume.

A candidate who says "looks fine, add a test" is not senior under the new rubric, regardless of how fast they can write a binary search. A candidate who says "why is the author email in the response at all, and why are we logging it" is demonstrating exactly the scarce skill.

What makes this exercise useful is that it's *short*. It can run in twenty minutes, it's objectively scorable (did they flag the personal data, did they flag the logging, did they ask about authorisation), and it's hard to game by having memorised a pattern, because the specific fields and the specific obligation are the point.

## Scoring without inventing a number

It's tempting to want a single "AI fluency score." Resist it. The useful output of a review exercise is a set of binary observations, not a composite number. For the example above, the observable facts are:

- Did the candidate notice the email is returned to the client?
- Did the candidate notice the email is logged?
- Did the candidate connect either to a data-protection obligation?
- Did the candidate ask about authorisation rather than assume it?
- Did the candidate propose a concrete fix (drop the field, omit it from logs, add a retention note)?

Five yes/no observations give you more signal than a 1–5 scale, because the scale invites disagreement about what a "4" means while the observations are just facts about what was said. This is the same reason structured interviews outperform unstructured ones — you're reducing the surface area for interviewer bias.

## The failure modes nobody warns you about

Three patterns show up repeatedly when teams try to modernise a rubric, and all three are avoidable.

**The fluency trap.** Interviewers who are themselves impressed by fast, confident AI output will over-score candidates who produce it. The fix is to score the review, not the generation. If your exercise lets a candidate generate code and you only look at the result, you've rebuilt the old rubric with extra steps.

**The regulation blind spot.** Teams outside the EU often treat data-protection review as a specialist concern that "legal handles." For a service handling European users' data, it isn't — it's an engineering concern that shows up in code review, and a senior engineer who can't spot a personal-data field in a response body is a liability regardless of how good their algorithms are. If your rubric doesn't test for this, you're hiring for a job that doesn't exist in your jurisdiction.

**The seniority inflation.** Because the new signals are harder to assess, there's a temptation to lower the bar and call more people senior. That's backwards. The bar didn't move down; it moved sideways. A candidate who can't demonstrate the new signals isn't a senior under the old rubric either — they were just good at the proxy, and the proxy was always a proxy.

## What this means for the interview loop

A loop that reflects the change usually looks like this, and it's shorter than the old one, not longer:

1. **A scoped build task** (unchanged in spirit, but explicitly allowed to use AI tools). Score the specification the candidate writes before generating, and the tests they write after. The code itself is a means, not the artefact.
2. **A review exercise** like the one above. This is the highest-signal part of the loop and the part most teams are missing.
3. **A design conversation** focused on constraints — what would you *not* build, what would you refuse to log, what would you escalate. This replaces the traditional "design a URL shortener" prompt, which mostly tests recall of a pattern.
4. **A debugging exercise** where the candidate is given a failing test and a codebase they can query with AI assistance. Score how they form and falsify hypotheses, not how fast they find the bug.

Notice that none of these require the candidate to pretend AI doesn't exist, and none of them reward using it uncritically. That's the balance the old rubric couldn't strike because the tools didn't exist when it was written.

## Where this doesn't apply

There are roles where the old rubric is still close to correct. Embedded and firmware work, safety-critical systems, and anything where the code is small, verified, and the cost of a subtle error is catastrophic — in those contexts, the ability to reason about the code without a generator in the loop is still the core skill, and the review exercise above is less relevant than raw correctness. The mistake is assuming that because the old rubric is right for those roles, it's right for the web-service roles that make up most of the market.

Similarly, if a team's bottleneck is genuinely throughput — more well-specified work than engineers — then raw production speed still matters, and a rubric that scores it is defensible. But that's a specific situation, not the default, and it's worth checking whether it's actually true before optimising for it.

## Frequently Asked Questions

**Should candidates be allowed to use AI in interviews?**
Yes, but score what they do with it, not whether they use it. The useful exercise is to give them a task where a naive AI-generated answer is subtly wrong and see whether they catch it. Banning AI tests a skill the job no longer requires in isolation.

**How do you test for regulatory awareness without making it a legal quiz?**
Don't test for the regulation, test for the behaviour. Put a personal-data field in a review exercise and see whether the candidate questions it. You're checking whether they *notice*, not whether they can recite an article number.

**Doesn't this just reward people who are good at prompting?**
No — prompting is the cheap part. The exercise above doesn't reward a good prompt; it rewards noticing that the output logs an email address. That's domain judgement, not prompt craft, and it's the thing that's actually scarce.

**The loop is already long. How does this fit in?**
Replace, don't add. The review exercise can take the slot of one of the existing coding rounds, and it's higher-signal than a second algorithm question. If a loop can't afford twenty minutes for the highest-signal exercise, the loop is the problem.

## What to do in the next 30 minutes

Open your current interview rubric and find the line that scores "writes correct code" or "solves the problem." Rewrite that one line so it scores whether the candidate can spot a problem in code that already runs. That single edit is the whole change in miniature — and it's the one most rubrics are missing.
