# Commit ownership in AI pair coding

AI coding assistants are usually adopted for speed, and the documentation that ships with them focuses on speed: suggestions, completions, refactors. What the documentation rarely covers is what happens after the generated diff lands in a repository. The unresolved question is ownership. When an AI-assisted commit fails a linter in CI, misses an edge case in a unit test, or pages the on-call engineer, the commit's `author` field still points at a human. But the logic originated from a prompt that may have been reviewed lightly or not at all.

That gap is not a tooling bug. It is a missing contract between the agent, the developer, and the CI pipeline. This article describes what that contract looks like in practice, how to implement it with a small amount of code, where it breaks, and when it is the wrong choice.

## The gap between what the docs say and what production needs

Most vendor documentation for AI coding assistants describes the generation step: how to prompt, how to accept a suggestion, how to run an agent over a repository. It says much less about the verification step. The result is that teams inherit a workflow where the artifact (a diff) and the accountability (a person) are only loosely coupled.

A typical failure mode looks like this. An AI suggests a regular expression. It passes the developer's local pre-commit hook. It fails in CI because the CI runner uses a different language runtime version, and the regex depends on Unicode behavior that changed between those versions. The commit author is the developer. The logic came from the model. The reviewer skimmed the diff because it was small and looked plausible. Nobody in that chain can say with confidence which requirement the regex was meant to satisfy.

That ambiguity compounds. The next developer who touches the file inherits a black box: no clear intent, no test that pins the behavior, and no way to tell whether the regex was hand-written or generated. The cost is not paid at generation time. It is paid at debugging time, usually weeks later, usually under pressure.

The constraint is not model capability. It is the absence of a lightweight, enforceable boundary that makes the AI's contribution visible and testable before it reaches the main branch.

## Treating the agent as a constrained subprocess

The framing that works is to stop treating the assistant as a silent pair programmer and start treating it as a subprocess with explicit inputs, outputs, and gates. Concretely, that means three things:

1. The agent receives a prompt that includes acceptance criteria and an environment signature, not just a ticket title.
2. The agent's output must pass the same test suite that human code must pass, in a container that matches CI.
3. The commit records which prompt version produced the diff, so the review can be traced back to its inputs.

The agent never writes directly to the main branch. It writes a diff to a staging location. A human reviews that diff and decides whether to commit it. The commit message states plainly that the change is AI-generated and names the reviewer.

Three lightweight mechanisms implement this:

**Prompt guardrails.** The prompt template is hashed, and the hash is stored alongside the commit (a Git note works; so does a row in an audit table). If the template changes, the hash changes, and prior reviews are marked stale. This addresses prompt drift, the situation where the same ticket produces different output because the template evolved underneath it.

**A test regression gate.** Before a diff is accepted, it runs through the project's test suite inside a container pinned to the same runtime as CI. If the suite fails, the diff does not reach the developer's editor. This catches the common case where generated code is correct for the developer's machine but not for the deployment target.

**Ownership tagging.** Lines or hunks in the diff carry a marker linking them to a requirement or a test case, for example a comment referencing a ticket ID. When something fails later, the trace points at a requirement, not at "some AI change." Rollbacks become surgical rather than all-or-nothing.

None of this requires a cluster, a feature-flag service, or a dedicated on-call rotation. It runs on a single small VM or even locally, depending on team size.

## A worked example: offline-first form caching

Consider a ticket for a progressive web app: "Add offline-first caching for form submissions using localStorage." The acceptance criteria are that form data survives an app restart, that submissions queue while offline, and that no data is lost when the network recovers.

### Step 1: The prompt template

Store the template in the repository so it is versioned alongside the code.

```text
You are an experienced frontend engineer working on a progressive web app.

Ticket: {ticket_description}

Acceptance criteria:
- Use localStorage to cache form data
- Support offline submission when network recovers
- No data loss on app restart

Environment signature:
- Node 20 LTS
- React 18.2
- TypeScript 5.4
- Jest 29.7
- Playwright 1.40

Write a diff that passes the acceptance tests below.

If you cannot meet the criteria, output nothing.
```

The environment signature matters. Without it, the model has no way to know which runtime behaviors to assume, and it will default to whatever is most common in its training data.

### Step 2: The pre-commit gate

The gate runs the test suite in a container that matches CI. The script below is illustrative and assumes the project installs with `npm ci` and tests with `npm test`.

```python
#!/usr/bin/env python3.11
import subprocess
import sys
import os

def run_tests():
    # Use the same container image as CI
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{os.getcwd()}:/app",
        "-w", "/app",
        "node:20-slim",
        "sh", "-c",
        "npm ci && npm test"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stdout, result.stderr

def main():
    passed, stdout, stderr = run_tests()
    if not passed:
        print("AI diff failed tests:")
        print(stdout)
        print(stderr)
        sys.exit(1)
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

Two details are easy to get wrong. First, `-w /app` sets the working directory inside the container; without it, `npm ci` runs in the wrong place. Second, the image tag should be pinned to a specific patch release rather than the floating `node:20-slim` tag, because slim images occasionally change their base layer and break a dependency that relied on a specific glibc version.

### Step 3: Prompt hashing

The hash is written to a file and attached to the commit as a Git note. This is what makes the review traceable.

```python
#!/usr/bin/env python3.11
import hashlib
import os

PROMPT_PATH = ".ai_prompt_hash.txt"

def hash_prompt():
    with open("ai_prompt_template.txt", "r") as f:
        prompt = f.read()
    return hashlib.sha256(prompt.encode()).hexdigest()

if os.path.exists(PROMPT_PATH):
    with open(PROMPT_PATH, "r") as f:
        old_hash = f.read().strip()
    new_hash = hash_prompt()
    if old_hash != new_hash:
        print("Prompt changed. Invalidating previous reviews.")
        os.remove(PROMPT_PATH)

with open(PROMPT_PATH, "w") as f:
    f.write(hash_prompt())
```

Attach the hash to the commit:

```bash
git notes add -m "ai_prompt: $(cat .ai_prompt_hash.txt)" HEAD
```

When reviewing an older commit, `git notes show HEAD` reveals which prompt version produced the diff.

### Step 4: Developer workflow

1. The developer opens a ticket.
2. A script fetches the ticket description, injects it into the template, calls the model, and writes the resulting diff to a staging file.
3. The developer reviews the diff in an editor with a diff viewer.
4. The pre-commit gate runs the test suite in the pinned container.
5. If tests pass, the developer commits with a message that records the AI's contribution and the reviewer:

```
feat(cache): offline-first localStorage for form submissions

AI-generated, reviewed by @dev-name
Test: jest --testPathPattern=offline-cache
Req: #1234
```

6. CI runs the same test suite on the same runtime.
7. Only if all checks pass does the change merge.

The developer retains ownership. The AI is a subprocess whose output is gated by the same checks as human code.

## How to measure whether this actually helps

Claims about productivity gains from AI guardrails are easy to make and hard to verify. The honest position is that the effect depends on the codebase. What can be measured is a set of proxy metrics that any team can instrument.

Track these over a baseline period before introducing the guardrails, then again afterward:

- **CI pass rate on first push.** Count PRs where the first CI run succeeds. This is the most direct signal that the local gate matches CI.
- **Median PR size in changed lines.** Available from your Git host's API. Smaller, focused diffs are easier to review and easier to revert.
- **Median time from PR open to first review.** Also available from the Git host.
- **Reverts and hotfixes per week.** Count commits that revert a prior commit or are labeled as hotfixes.
- **Failures attributable to environment mismatch.** Grep CI logs for errors that do not reproduce locally.

The comparison that matters is the same repository before and after, not one repository against another. A worked assumption: if first-push CI pass rate moves from 80% to 90% on a team that opens 40 PRs per week, that is four fewer failed CI cycles per week. Whether that is worth the maintenance cost of a pinned container is a judgment call the team has to make with its own numbers.

## Failure modes worth knowing about

Even with guardrails in place, several failure modes recur.

**Prompt drift.** A team adds a constraint to the template, for example support for a new locale. The hash changes. Old commits still reference the old hash, and reviewers comparing a new diff to an old one may not realize the inputs differ. The fix is to keep the Git note attached to each commit so the prompt version is always visible, and to treat a hash change as a reason to re-review rather than to assume equivalence.

**Container drift.** The pinned image is rebuilt upstream, or a base layer changes. A test that passed last week fails this week for reasons unrelated to the diff. The fix is to pin to a digest rather than a tag, and to rebuild the image on a schedule so drift is caught in CI rather than during a commit.

**False ownership attribution.** Once a diff carries an `#ai:` tag, reviewers may assume the tagged lines are the model's and skim the rest, or vice versa. The fix is to require a short human-written summary in the PR body that states what the model proposed and what the reviewer verified. Without that summary, a commit can look reviewed while containing untested assumptions.

**Empty error handling.** A model asked to add a network call may wrap it in a try/catch and leave the catch block empty. Tests pass if the mocked call never throws. A lint rule that forbids empty catch blocks, enforced in the pre-commit gate, catches this class of bug cheaply. The lesson is that ownership is partly a function of which lint rules are enforced on generated output.

## Tools and their roles

| Role | Example category | What to look for |
|---|---|---|
| Diff review | Editor extension with inline diff view | Line-level attribution, PR integration |
| Pre-commit hooks | Hook manager (e.g., `pre-commit`) | Runs arbitrary checks before commit |
| Runtime matching | Container runtime | Pinnable image digests, CI parity |
| Prompt hash cache | Key-value store | Avoids redundant model calls on unchanged prompts |
| CI | Hosted runner | Same runtime as the local gate |
| AI agent | Editor or CLI agent with prompt templates | Writes diffs, not direct commits |

The most underrated item in that table is the hook manager. Most teams treat pre-commit hooks as a linting convenience. Used as a boundary enforcer, they shift the contract from "trust the model" to "the model's output passes the same gates as human code."

Avoid any agent feature that auto-commits generated changes. Configure the agent to write diffs to a staging location and let the developer decide what to commit.

## When this approach is the wrong choice

The guardrail pattern assumes a few preconditions. Where they are missing, it adds overhead without adding safety.

**No test suite.** If the project has no meaningful tests, the gate has nothing to validate. The prompt hash still runs, but it is only a linter. Teams in this position should invest in test coverage first. Adopting the guardrails before that produces a false sense of safety.

**High dependency churn.** Research prototypes that update dependencies weekly will fight the pinned container. Every update invalidates the image and the prompt hash. In that setting, skip the container gate and rely on strict lockfile-based installs in CI instead.

**Regulated environments.** Git notes are mutable and not designed as an audit trail. Financial and healthcare systems may need the prompt, the diff, and a cryptographic hash stored in an append-only log with retention guarantees. A database table with appropriate access controls is a better fit than Git notes in that context.

**Very small teams.** With two or three developers, the maintenance cost of a pinned container and a prompt template can exceed the benefit. The overhead is real: image rebuilds, hash invalidation, and review summaries all take time. A team should measure its own CI failure rate before deciding whether the gate pays for itself.

## A decision checklist

Before adopting the guardrail pattern, answer these questions:

- Does the project have a test suite that runs in under five minutes?
- Can the test suite run in a container that matches CI?
- Is there a place to record the prompt hash that reviewers will actually check?
- Is there a lint rule set that catches the error-handling patterns the model tends to produce?
- Does the team have a convention for marking AI-generated commits in the message?
- Is there a designated reviewer for AI-generated diffs, or does review fall to whoever is available?

If more than two answers are "no," fix those first. The guardrails are only as good as the practices underneath them.

## What to do in the next 30 minutes

Open the repository and create a pre-commit hook that runs the project's test suite inside a container pinned to the same runtime as CI. Make it executable, then run it against the most recent AI-generated diff in the history. If the hook fails, the failure tells you which part of the contract is missing: the tests, the container, or the prompt. That single step is enough to expose the gap between the current workflow and what production actually requires.

## FAQ

**How do I keep AI commits from adding whitespace noise to history?**
Run the formatter in the pre-commit hook before the test gate, and stage the formatted output as a separate commit. That keeps the generated diff focused on logic and avoids noise in blame output.

**Does using feature flags instead of feature branches solve the ownership problem?**
No. Flags change when code is exposed, not who is accountable for it. If a flagged feature fails, the flag owner is still responsible for rollback. Keep the same gate: the diff must pass the test suite in a production-like container before it merges, regardless of how it is released.

**How do I apply this in a monorepo with multiple languages?**
Pin a base image that includes the required runtimes, install language-specific toolchains in the Dockerfile, and have the hook run each language's test command in sequence. Exit non-zero if any suite fails.

**When is it reasonable to disable the gate?**
During exploratory prototyping on a branch that will never merge directly to main. Use a separate prompt template for that branch, tag the branch as experimental, and require the gate before any merge.

**Why does the containerized hook fail with install errors?**
Usually a mismatch between the lockfile in the repository and the one the container resolves. Reproduce locally with the same command the hook runs, then update the lockfile if needed.

**What is the minimum team size for this to be worth it?**
It depends on the CI failure rate and the cost of a production incident. Teams of three or four can share a single gate and template. Smaller teams should measure first: if CI failures are rare and incidents are cheap, the maintenance overhead may not be justified.
</parameter>
