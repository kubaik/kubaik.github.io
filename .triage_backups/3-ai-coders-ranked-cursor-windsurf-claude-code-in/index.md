# 3 AI coders ranked: Cursor, Windsurf, Claude Code in

AI coding assistants are usually selected on the strength of a demo: a fast edit cycle, a large context window, a polished diff view. Those demos are real, but they are measured on someone else's repository, with someone else's latency budget, and under someone else's data-handling policy. When the same tool is pointed at a production monorepo with row-level security, a caching layer, and a compliance regime, the interesting failures tend to be about context selection and prompt construction rather than raw model quality.

This article describes a framework for choosing among three widely used categories of assistant — an AI-native fork of an editor, a plugin-based assistant, and a terminal-first agent — by measuring four properties that actually predict whether a tool survives contact with a real codebase: edit latency, context retention, audit trail coverage, and cost. It also covers the failure modes that show up only after weeks of daily use, and the guardrails that make a tool safe enough to keep.

## Why a ranking is the wrong unit of analysis

A ranked list implies a total order, and a total order only exists if every team weights the same properties the same way. They do not. A frontend team working in a single application with a handful of contributors has almost nothing in common with a platform team maintaining a multi-repo service mesh under a data-residency policy.

The properties that vary most between teams:

- **Latency tolerance.** Inline completion is judged against typing speed. Anything above roughly 300 ms of perceived delay breaks the flow of writing code and causes developers to stop waiting for suggestions. Chat-style and agent-style interactions tolerate seconds, because the developer has already stopped to think.
- **Context shape.** A single-repo application can often be understood from the open file plus its imports. A monorepo with generated clients, shared schema packages, and multiple languages cannot. Tools differ enormously in how they decide what to send.
- **Data classification.** If the codebase contains personal data, credentials, connection strings, or anything covered by a residency requirement, then every prompt is a potential disclosure. The relevant question is not "does the tool send code to a model" — almost all of them do — but "what exactly does it send, and can you constrain it."
- **Auditability.** Regulated or safety-critical teams increasingly need to answer, after the fact, which lines were machine-suggested. That is a tooling property, not a model property.

Because these weights differ, the useful output of an evaluation is not a winner. It is a table of measured values for your repository, plus a decision rule you write for yourself before you start measuring.

## How to evaluate an assistant on your own repository

Give each candidate its own branch and its own week, and measure the same four things each time. The point is not precision; it is comparability. A rough number measured consistently across three tools beats a precise number measured once.

### Latency per edit

Latency is the wall-clock time between the end of a keystroke and the first rendered suggestion. Instrument it in two places, because they measure different things:

1. **Editor-side perceived latency.** Wrap the completion request in a timer in the editor's extension host or renderer process, or use the browser/Electron performance API if the editor exposes one. Log the delta between request dispatch and first paint of the ghost text. This is the number a developer actually feels.
2. **Process-level startup and index time.** For any tool that builds a symbol index, the cold-start cost dominates the first minutes of a session. Measure it with a shell timer:

```bash
# Cold-start: time to first usable index on your largest repository
hyperfine --warmup 3 --runs 10 'code /path/to/large-repo'
```

`hyperfine` reports mean and standard deviation across runs, which matters because cold-start latency is noisy. Run it on a warm filesystem cache and again after dropping caches if your platform allows it; the difference tells you how much of the cost is disk I/O versus indexing.

Report latency as a median and a 95th percentile, not a mean. The tail is what makes an editor feel unpredictable.

### Context retention

Context retention is the proportion of edits that the tool completes correctly without the developer manually re-supplying information it should already have — an import path, a type definition, a function signature from another module.

There is no clean automated metric for this. The workable approach is a manual log kept over a fixed number of edits:

- Keep a counter of total accepted suggestions.
- Keep a second counter of suggestions that were wrong specifically because the tool lacked context (wrong import, wrong overload, invented symbol).
- The ratio of the second to the first is your context-failure rate.

A tool with a large advertised context window can still score badly here, because a large window does not mean the tool selects the right content to put in it. Selection quality and window size are different properties, and the first matters more.

### Audit trail coverage

Audit trail coverage is the fraction of committed machine-suggested lines that are traceable to a tool and, ideally, to a session. Most tools attach a trailer or a hash to the commit message when a suggestion is accepted through the tool's own commit path; almost none do so when the developer edits the suggestion by hand before committing.

Measure it by sampling commits, not by trusting the tool's dashboard:

```bash
# Count commits in the last 30 days that carry an AI-attribution trailer
git log --since='30 days ago' --pretty=%H --grep='Co-authored-by' | wc -l

# Total commits in the same window, for the denominator
git log --since='30 days ago' --oneline | wc -l
```

Then sample twenty of the commits that carry no trailer and read the diffs. In practice a meaningful share of them will contain machine-suggested code that lost its attribution during manual editing. That gap is the honest coverage number, and it is usually well below whatever the tool reports.

### Cost

Cost has three components and they are easy to conflate:

- **Seat cost.** A flat per-developer subscription. Predictable, independent of usage.
- **Token cost.** Metered model usage. Unpredictable, and strongly correlated with how much context the tool decides to send — which is exactly the property you cannot easily control.
- **Local compute cost.** Electricity and hardware amortisation for locally hosted models. Small per hour, non-trivial per year, and it does not disappear when usage drops.

For token cost, read the tool's own usage export rather than estimating. If the tool does not expose per-request token counts, that absence is itself a finding: you cannot budget for a cost you cannot observe. A worked estimate is worth doing once so you understand the sensitivity:

> Suppose a tool sends an average of 40,000 input tokens per accepted suggestion, and a developer accepts 60 suggestions per working day over 20 working days. That is 48 million input tokens per developer per month. Multiply by your provider's published input price per token to get the monthly figure. Now repeat the arithmetic with 15,000 input tokens per suggestion — the same developer, the same acceptance rate. The result is roughly a third of the cost. The variable that moved was context selection, not model choice.

That is why cost is best treated as a consequence of the other three measurements rather than an independent axis.

## Failure modes that only appear after weeks of use

These are the problems that do not show up in a one-day trial, and they are the reason a tool that looks excellent in a demo can be retired a month later.

### Over-broad context selection and inadvertent disclosure

The most common serious failure is not a malicious one. It is a tool deciding, reasonably by its own heuristics, that the most relevant context for the current edit includes a neighbouring file, a configuration file, a fixture, or a log. If any of those contain a personal identifier, a connection string, a token, or customer data, that content is now in a prompt sent to a third-party model.

The signature of this failure is a diff that is syntactically correct and semantically wrong: the assistant proposes a change that references a symbol from a repository or package it should not have been reading. A related signature is a suggestion that swaps one resource for another — for example, substituting a read-only replica connection string for a primary one — because both appeared in the retrieved context and the model had no way to know which was authoritative.

Mitigations, in order of strength:

1. **Constrain retrieval at the source.** Exclude repositories, directories, and generated files from indexing. Most tools support an ignore file or an index-scope setting; check whether it is honoured by the retrieval path and not only by the editor's file watcher.
2. **Post-process the prompt.** If the tool exposes a hook that runs before the request is sent, apply a redaction pass. A minimal email redaction in a preprocessing script looks like this:

```python
# redact.py — strip common identifiers from a prompt before it leaves the machine
import re
import sys

EMAIL = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
REDIS_URL = re.compile(r'redis://[^:]+:[^@]+@[^:]+:\d+')

def redact(text: str) -> str:
    text = EMAIL.sub('<REDACTED_EMAIL>', text)
    text = REDIS_URL.sub('redis://<REDACTED_HOST>:<REDACTED_PORT>', text)
    return text

if __name__ == '__main__':
    sys.stdout.write(redact(sys.stdin.read()))
```

3. **Scan the diff before it is committed.** A pre-commit hook that runs a secret scanner over staged changes catches the case where redaction missed something or the tool bypassed the hook entirely:

```bash
#!/bin/bash
# .git/hooks/pre-commit — fail the commit if staged changes contain secrets
if ! ggshield secret scan pre-commit; then
  echo "Secret scanner rejected staged changes."
  exit 1
fi
```

4. **Verify empirically.** Do not assume redaction works because the configuration says it does. Capture outbound traffic from the editor process for one session and inspect the request bodies. This is the only way to confirm what is actually transmitted, and it is worth repeating after every tool upgrade, because prompt construction is not a stable interface.

### Context bloat and silent truncation

The opposite failure: a tool that includes so much context that the request approaches or exceeds the model's limit. The visible symptom is a slow, expensive session. The invisible symptom is truncation — the tool silently drops the middle of the context, and the model then reasons over an incomplete picture. A classic result is an assistant that fills its window with thousands of lines of test output or build logs and loses the actual source it was supposed to be editing.

Watch for these signals:

- Response latency creeping up over a session without any change in task complexity.
- Suggestions that reference symbols from early in the conversation but not from the file currently open.
- Cost per accepted suggestion rising over the course of a day.

The fix is to cap the context the tool may attach, and to exclude log directories, build output, and test fixtures from retrieval. Where the tool does not offer a cap, treat that as a capability gap.

### Symbol collision across packages

In polyglot or multi-package repositories, two modules frequently export the same name — a `User` type from an authentication package and a `User` model from a legacy data layer, for example. A retrieval-based assistant may resolve the name to whichever definition it retrieved, producing code that compiles against the wrong type or fails at runtime.

The reliable fix is disambiguation at the source, not in the prompt:

```python
from auth.types import User as AuthUser
from legacy.models import User as LegacyUser
```

This is a few lines of boilerplate per affected file, and it is worth it, because prompt-level disambiguation is fragile: it depends on the tool retrieving the instruction along with the code.

### Timezone- and locale-dependent context

Tools that attach timestamps, log excerpts, or generated identifiers to the context can produce diffs whose contents depend on the machine's local settings. On a distributed team this makes review confusing and, worse, makes a suggestion non-reproducible: the same edit request on two machines yields different context. The mitigation is to normalise anything that enters the context — timestamps in UTC, identifiers stripped, locale-independent formatting — at the point where the context is assembled.

## A decision checklist

Work through this before committing to a tool. It is ordered so that cheap disqualifiers come first.

1. **Data classification.** Does the tool's retrieval path touch repositories containing personal data or secrets? If yes, can retrieval be scoped and can the prompt be post-processed? If neither, stop here.
2. **Outbound verification.** Can you capture and inspect outbound requests? If the tool obscures its traffic in a way you cannot audit, you cannot make a compliance claim about it.
3. **Latency on your repository.** Measure median and p95 edit latency on your largest repo, not a sample project. Compare against your team's tolerance, which for inline completion is usually in the low hundreds of milliseconds.
4. **Context failure rate.** Log context-caused errors over at least 200 accepted suggestions. A tool that is fast but wrong about imports is slower in practice than a slower, more accurate one.
5. **Audit coverage.** Sample commits and compute the real attribution rate. Decide whether the number you measured satisfies your review requirements.
6. **Cost observability.** Can you export per-request token counts? If not, you cannot forecast spend, and you should assume the cost is unbounded until proven otherwise.
7. **Exit cost.** How much of your configuration — ignore files, prompt templates, hooks — is portable to another tool? Prefer configurations that live in your repository and are tool-agnostic.

## A worked comparison of the three categories

The table below compares the three shapes of tool, not three specific products. The values are illustrative — they are the kind of numbers a team typically measures on a mid-sized monorepo, and they are included to show how the axes interact, not as findings about any vendor.

| Property | Editor-native assistant | Plugin-based assistant | Terminal-first agent |
|---|---|---|---|
| Interaction model | Inline completion + inline chat | Inline completion inside an existing editor | Conversational, operates on the working tree |
| Typical latency tolerance | Tight; sub-300 ms expected | Tight; sub-300 ms expected | Loose; seconds acceptable |
| Context selection | Editor index plus open buffers | Host editor's index plus repo search | Explicit file/command scope you specify |
| Auditability | Depends on commit integration | Depends on commit integration | Often high, since the agent runs commands you can log |
| Data-handling control | Varies; check retrieval scope | Varies; inherits host editor's surface | Highest, because scope is explicit |
| Cost profile | Seat, sometimes plus tokens | Seat, sometimes plus tokens | Usually metered tokens |
| Main failure mode | Over-broad retrieval from the index | Over-broad retrieval from repo search | Context bloat from verbose command output |

The pattern worth noticing: the terminal-first shape trades latency for control. You pay in seconds per interaction and you get an explicit, loggable scope. The editor-native and plugin shapes optimise for flow, and the price of that optimisation is that retrieval decisions happen without you.

## Choosing by situation

**If your codebase is a single application and your main concern is speed of writing code**, an editor-native or plugin-based assistant is the natural fit. Measure latency and context failure rate, and keep the retrieval scope narrow. Audit coverage will be whatever your commit integration gives you; if that is not enough, add a pre-commit attribution check rather than switching tools.

**If you maintain a multi-package repository with strict review requirements**, prefer the terminal-first shape or any tool that lets you state scope explicitly. The ability to say "read these files" is worth more than a large automatic context window, because it makes the request reproducible and the disclosure surface knowable. Budget for metered tokens and instrument token counts per session so the cost does not drift.

**If you have data-residency or personal-data constraints**, the deciding question is not which model is best but which tool gives you a verifiable boundary. Choose the one whose outbound requests you can capture and whose retrieval you can scope from a file in your own repository. Then verify, and re-verify after every upgrade.

**If you cannot measure outbound traffic at all**, treat the tool as unsuitable for repositories containing regulated data, regardless of what its documentation claims.

## FAQ

**Does a larger context window mean better context retention?**
No. Window size is a capacity limit; retention depends on what the tool chooses to put in the window. A tool with a modest window and good retrieval frequently outperforms one with a huge window that fills it with logs and fixtures.

**How do you measure latency without instrumenting the editor?**
Use a process-level timer for startup and index time (`hyperfine` works well), and for per-edit latency either instrument the extension host or accept a coarser proxy: record the time from keystroke to the suggestion appearing in a screen capture at a known frame rate. The proxy is imprecise but consistent, which is enough for comparing tools.

**Is local model inference automatically safer?**
It removes the network disclosure path, which is a real benefit. It does not remove the need to scope retrieval, and it introduces its own failure modes: smaller local models tend to produce more incorrect symbol references, and the latency is usually higher than a hosted model on the same hardware.

**Why is audit coverage usually lower than the tool reports?**
Because attribution is attached when a suggestion is accepted through the tool's commit path. Developers routinely edit suggestions before committing, and that edit often drops the trailer. Sampling commits and reading diffs gives a truer number than any dashboard.

**Can prompt post-processing break suggestions?**
Yes. Aggressive redaction can remove identifiers the model genuinely needs, producing suggestions that reference placeholder text. Test redaction rules against a corpus of real diffs and check the context failure rate before and after, rather than assuming redaction is free.

## Do this in the next 30 minutes

Pick your largest repository and measure its cold-start cost with a tool that builds an index:

```bash
hyperfine --warmup 3 --runs 10 '<your-editor-command> /path/to/your/largest-repo'
```

Record the mean and the standard deviation. Then open the repository in each candidate assistant and time, by hand, ten inline suggestions from keystroke to first render. Write all three numbers down next to your team's latency tolerance. If any tool's p95 exceeds that tolerance on your actual repository, you have eliminated it without spending a week on a trial — and you have the first row of the comparison table you will fill in for the rest.
