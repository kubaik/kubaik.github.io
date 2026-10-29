# AI answers first, judgment later

AI assistants have changed the order of operations in junior engineering work. The first answer to most technical questions now arrives in seconds, and it usually compiles. Mentoring models that assume the junior's first move is to ask a human are built on an assumption that no longer holds. The useful question is not how to restore the old order, but how to restructure mentoring so that AI-generated answers become the start of a review conversation rather than the end of one.

## The stance: AI-first answers are fine, AI-first judgment is not

The problem is not that junior engineers consult an AI assistant before asking a senior. The problem is that they stop after the first answer. Assistants are good at producing plausible, syntactically correct, context-free code. They are bad at knowing which of several plausible answers applies to a specific system, with its particular failure modes, on-call rotation, and regulatory constraints.

Mentoring therefore shifts from being the source of answers to being the source of judgment: the person who asks "what happens when this times out?" and "who gets paged when it does?"

The economics explain why policy alone cannot reverse the order. An AI answer costs the junior ten seconds and no social capital. Asking a senior costs a context switch, a thread in chat, and the low-grade anxiety of looking slow. No written rule out-competes that. What can change is what happens after the AI answer arrives.

A second consideration: the senior's answer is not always better. For a self-contained algorithm question, a well-prompted model can produce a cleaner explanation than a rushed senior late on a Friday. The senior's advantage is not raw knowledge. It is knowledge of this codebase, this team's risk tolerance, and this incident history. Mentoring that competes on raw knowledge competes on the wrong axis.

The most damaging failure mode is prohibition. If AI use is forbidden, juniors use it anyway and hide it, and visibility into the wrong answers they are internalizing disappears. The failure is not "junior used AI." It is "junior used AI, got a confident wrong answer, and shipped it because nobody saw the conversation."

## Why "just tell them to ask me first" fails

The common advice — tell juniors to ask a human before asking AI — sounds sensible and is nearly unenforceable. It also misreads what the junior is optimizing for, as described above. Three distinct failures follow from it.

**It loses the prompt.** The question a junior types into an assistant is the clearest available signal of their mental model. A policy that routes questions away from AI discards that signal rather than capturing it.

**It assumes the human is always the better source.** For public, stable, verifiable facts — a date-format string, a standard-library signature — the assistant is often faster and equally correct, and a senior's time is better spent elsewhere.

**It drives use underground.** Hidden use means hidden errors. A junior who gets a confident wrong answer and has no sanctioned way to surface the conversation has no path to correction short of production.

## The mental model: a fast, confident intern with no context

A useful framing is to treat the assistant as a very fast, very confident intern who has read every public blog post but has never been on call for the service in question. Three properties follow.

1. **Fluent, not grounded.** It produces code that looks like it belongs in the repository, without knowing whether the service uses synchronous HTTP or an event bus, whether the datastore is relational or key-value, or whether the team has a rule against new dependencies.
2. **Confident under uncertainty.** The model does not reliably signal "I am guessing" in a way a junior can detect. A junior who has not yet built a mental model of what correct looks like in the domain cannot distinguish a textbook answer from a production answer.
3. **Optimizes for the question asked, not the question that should have been asked.** Ask "how do I retry this HTTP call?" and you get a retry loop. You do not get "should this call be retried at all, or is it non-idempotent, so retrying will double-charge the customer?"

Under this model, the mentor supplies the context the model lacks and teaches the junior to ask for it. That is a different skill from explaining how a retry loop works.

## The shift: from answer-giver to answer-reviewer

The practical change is to make the AI conversation an artifact that gets reviewed rather than a private transaction. Three practices, in increasing order of ceremony.

**Ask for the prompt, not just the answer.** When a junior brings a solution, the first question is "what did you ask it?" A prompt like "write a function to process orders" produces generic code. A prompt like "write a Python function that processes orders from a queue, assumes at-least-once delivery, and must be idempotent on order_id" produces something reviewable. Teaching juniors to write the second prompt is teaching them to think about idempotency, delivery semantics, and failure modes. The prompt is the design document.

**Require a "what I checked" note.** Before AI-assisted code merges, ask for three lines in the pull request description: what the AI produced, what the junior changed, and what they are unsure about. The third line is the most valuable. "I am not sure this handles a malformed queue message" identifies exactly where the mental model has a hole, and the habit of locating uncertainty rather than suppressing it is the transferable skill.

**Review the diff against the requirement, not against the AI's output.** The AI's output is not the source of truth; the requirement is. A useful review question: "if the AI had never been involved, what would you have written, and why is this different?" Sometimes the answer is "it is the same, I just typed it faster." Sometimes it is "I would not have thought to handle the partial-failure case." Both are informative.

## A decision rule for when to answer and when to send them to the AI

Not every question deserves the same response. The tradeoff is genuinely side-by-side:

| Situation | Who answers | Why |
|---|---|---|
| Self-contained syntax or API question | AI | The answer is public, stable, and cheap to verify by running it. Senior time here teaches little. |
| Question about this codebase's conventions | Human | The AI does not know the conventions and will guess plausibly. |
| Tradeoff with production consequences | Human, with AI as input | The AI can enumerate options; only a human knows the idempotency story and the on-call pain. |
| Question the junior has not tried to formulate | Neither — send them back to write a prompt | The act of writing the prompt is the learning. Answering directly skips it. |
| Debugging a specific error message | AI first, human if unresolved | Error messages are well represented in training data; the assistant is often right and fast. |

The rule of thumb: **if the answer depends on facts that are not in the public corpus, a human must be in the loop.** Schema, service-level objectives, incident history, and compliance constraints are not in the model's training data, and the model will not announce that it is guessing.

## Where human-first is still right

The argument is not that juniors should route everything through AI. Several categories keep the human-first rule, and pretending otherwise is the failure mode on the other side.

**Anything with a security or compliance dimension.** An assistant will readily generate code that logs a full request body, including auth tokens, because the prompt did not say not to. A junior who does not yet know what counts as sensitive data cannot evaluate that output. Security review has to be human, and the earlier a junior learns to flag it, the better.

**Anything where the junior cannot yet tell a good answer from a bad one.** This is the core skill gap. A junior with two years of experience can evaluate a retry loop. A junior in the first month cannot evaluate a database migration strategy, because they have never seen one fail. For those questions, the human answers first and explains why the AI's likely answer would have been wrong. That comparison is the lesson.

**Anything where the social cost of asking is the point.** Part of mentoring is teaching that asking a question is not an admission of incompetence. If every question routes through a machine, the junior never builds the habit of saying "I do not know this yet" to a person. That habit matters on incident calls, in design reviews, and in performance conversations. Protect it deliberately.

## What this looks like in practice: a reviewable prompt

A concrete example makes the shift tangible. Suppose a junior is asked to add retry logic to an outbound HTTP call. The AI-first version of the question is "how do I retry an HTTP request in Python?" The mentored version is a prompt that encodes the constraints:

```python
# The junior writes this prompt, not the AI:
# "Write a Python function using the requests library that calls an
#  external payment API. It must retry on 5xx and connection errors,
#  but NOT on 4xx (those are client errors and retrying wastes quota).
#  It must use exponential backoff with jitter, cap total attempts at 3,
#  and respect a 2-second per-attempt timeout. The call is NOT idempotent
#  on the server side unless we pass an Idempotency-Key header, so the
#  function must generate and reuse one across retries. Show the code
#  and list the assumptions you made."
```

The prompt itself is the design review. It forces the junior to decide which errors are retryable, whether the call is idempotent, how long to wait, and what to do about duplicate charges. The AI will produce code; the junior will already have done the thinking. When the code comes back, the mentor's review is not "does this retry loop work?" but "did you get the idempotency key right, and what happens if the third attempt also fails?" That is a conversation about failure modes, which is what mentoring is for.

A useful exercise is to have the junior paste the prompt and the AI's response into the pull request description. The reviewer can then see exactly what context was given. If the prompt omits the idempotency requirement, the reviewer catches it before the code is written rather than after.

## The failure mode nobody warns about: silent wrong answers

The dangerous case is not the answer that is obviously wrong. It is the one that is subtly wrong in a way that passes tests and code review. A retry loop that retries on 4xx passes a happy-path test and fails in production by burning quota. A query that looks correct but does not use the index passes in staging with 100 rows and times out in production with millions. A junior who has not yet internalized "what does this look like at scale?" cannot catch these, and the assistant will not volunteer the concern.

The mentoring fix is to make the junior responsible for naming the assumption. After every AI-assisted change, the junior writes one sentence in the pull request: "This assumes X." If X is "the payment API is idempotent," the reviewer knows exactly where to push. If the junior cannot name an assumption, that is the signal that they do not yet understand the code well enough to ship it.

## A worked example: the retry loop that retries the wrong thing

Consider a typical AI-generated retry function. The junior asks for "retry with backoff" and receives something like:

```python
import time
import random
import requests

def call_with_retry(url, payload, max_attempts=3):
    for attempt in range(max_attempts):
        try:
            resp = requests.post(url, json=payload, timeout=2)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError:
            if attempt == max_attempts - 1:
                raise
            time.sleep((2 ** attempt) + random.uniform(0, 1))
```

This code is plausible and wrong for most payment APIs. `raise_for_status()` raises on 4xx as well as 5xx, so a 400 (bad request) is retried three times, wasting quota and delaying the error. There is no idempotency key, so a retry after a timeout can double-charge. And `timeout=2` is a connect-and-read timeout, not a total deadline, so a slow-drip response can exceed it.

A mentored junior would have written the prompt with those constraints, and the resulting code would look different. The more important outcome is that the junior now knows why the naive version is wrong. That knowledge transfers to the next API they integrate, with or without AI.

### Verifying the fix by measurement

Rather than trusting a review, the corrected version can be checked. The point of the exercise is to show a junior how to turn "I think this is right" into evidence.

Instrument the function with a counter for attempts per error class, and run it against a local stub server that returns 400, 500, and a slow response. Compare the attempt counts:

- A 400 response should produce exactly one attempt. If the counter reads three, the retry predicate is wrong.
- A 500 response should produce at most three attempts, with gaps that grow and include jitter.
- A response that trickles bytes should be cut off at the total deadline, not merely at each socket read.

The commands are ordinary: start the stub locally, point the function's URL at it, and read the counters from the log. The value is not the specific numbers but the habit — every assumption in the prompt should have a test that would fail if the assumption were false.

## What to do in the next 30 minutes

Open the team's pull request template — commonly `.github/pull_request_template.md` or `.gitlab/merge_request_templates/Default.md` — and add three lines under a heading such as "AI-assisted changes": what the AI produced, what you changed, and what you are unsure about. Then tell the next junior you pair with that the third line is the one you care about most. That single change turns a private AI conversation into a reviewable artifact, and it costs one commit.

## Frequently Asked Questions

**Should AI assistants be banned for junior engineers?**
No. A ban is hard to enforce and drives use underground, which is worse than visible use. The better move is to require that AI-assisted changes are labelled and reviewed with the prompt attached, so wrong answers are caught before merge. Bans also waste the genuine speedup on self-contained questions where the assistant is reliably correct.

**How can a mentor tell if a junior is over-relying on AI?**
Look for the absence of named assumptions. A junior who ships AI-assisted code without being able to say what it assumes is over-relying. A junior who writes "this assumes the queue delivers at least once" is using the tool well. The signal is in the pull request description, not in the code.

**Does using AI stop juniors from learning?**
Only if the AI is used to skip the thinking. Used well, it compresses the lookup phase and leaves more time for the judgment phase, which is where the learning happens. The risk is that juniors skip both. The mentoring fix is to make the judgment phase visible and reviewed.

**What is the single best question to ask in an AI-assisted code review?**
"What did you ask it, and what did you change?" That question surfaces the prompt, which reveals the junior's mental model; the diff, which reveals what they noticed; and the gap between them. It is more informative than reviewing the final code alone, because it shows the reasoning that produced it.
