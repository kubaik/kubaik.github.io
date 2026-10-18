# Claude code review: wins and blind spots

LLM code review works in the simple case and fails in a specific, predictable way. This article covers where the approach earns its keep, where it quietly breaks down, and how to measure both for your own repository instead of trusting someone else's numbers.

## The one-paragraph version

An LLM reviewer applied to a pull request diff can catch surface-level problems quickly: missing docstrings, unused imports, absent timeouts on new async functions, endpoints that skip required metadata. It will also hallucinate import paths, invent APIs that do not exist, and miss race conditions, because it does not execute code or simulate concurrent load. The practical posture is to treat it as a narrow, prompt-driven linter rather than a reviewer: scope it to the diff, ask it specific questions, require structured output, and never let it commit or block a merge on its own authority.

## Why the concept confuses people

Two forces push expectations in opposite directions. Vendor marketing frames these tools as autonomous collaborators, which sets a bar no current model meets. Meanwhile, developers who try them once on a whole file, get a wall of irrelevant comments, and conclude the whole category is useless.

The reality is narrower and more useful than either position. A language model reviewing code is doing pattern completion over tokens. It is genuinely good at local, syntactic, rule-shaped concerns. It is bad at anything requiring execution, runtime state, or knowledge of your domain invariants. Most disappointment comes from asking it to do the second kind of work.

A typical failure mode: an LLM flags a missing docstring on a helper nobody will read, while staying silent on a new pagination parameter that can return duplicate rows under concurrent writes. The docstring comment is not wrong. It is just irrelevant to the risk that actually ships.

## The mental model that makes it click

Think of the model as a probabilistic grep combined with a junior developer who has read a very large amount of public code. It does not hold your repository in its head. It holds a compressed statistical sense of what code usually looks like.

That gives it strength in a specific band:

- Type hints that disagree with usage within a function
- Imports that are unused, or names that do not resolve
- Argument counts that do not match the call site
- Docstring presence and rough shape
- Style drift relative to the surrounding file
- Blocking calls sitting inside an async function

That band is real but narrow. It is a small fraction of what a code review actually needs to cover. The rest — correctness under concurrency, domain invariants, migration safety, authorization boundaries — requires either execution or knowledge the model does not have.

The operational consequence: run it on the diff, not the whole file. Ask narrow questions. A prompt that says "review this PR" produces noise. A prompt that says "flag any new async function without an explicit timeout" produces a short list you can act on.

## A worked example

Consider a PR that adds an endpoint `/api/v2/users/{id}/orders` returning paginated orders. The diff adds roughly 150 lines: one async route, one ORM query using offset/limit pagination, three unit tests, and one OpenAPI schema file.

### Step 1: Write a constrained prompt

```
You are a senior Python code reviewer. Review only the diff below. Check:
1. Every new async function has an explicit timeout.
2. Every new endpoint has OpenAPI tags matching /api/v2/*.
3. Every new public function has a Google-style docstring.
4. No new global variables.
5. All new imports are used.
6. No new blocking calls inside async functions.
7. All new SQL queries use LIMIT and OFFSET safely.

Return a JSON array with keys: issue_type, line, message, severity (low/medium/high).
Return [] if there are no issues in these categories.
```

Two details matter here. The enumerated list bounds the model's attention, and the explicit "return []" instruction prevents it from manufacturing findings to seem useful. Models asked for issues will usually produce issues; giving them a legal empty answer reduces that pressure.

### Step 2: Pipe the diff in

```python
import json
import subprocess

prompt = """You are a senior Python code reviewer. Review only the diff below. Check:
1. Every new async function has an explicit timeout.
2. Every new endpoint has OpenAPI tags matching /api/v2/*.
3. Every new public function has a Google-style docstring.
4. No new global variables.
5. All new imports are used.
6. No new blocking calls inside async functions.
7. All new SQL queries use LIMIT and OFFSET safely.

Return a JSON array with keys: issue_type, line, message, severity (low/medium/high).
Return [] if there are no issues in these categories."""

diff = subprocess.check_output(
    ["git", "diff", "--unified=0", "main...HEAD"], text=True
)

result = subprocess.run(
    ["claude", "--prompt", prompt, "--input", diff],
    capture_output=True, text=True, check=True,
)

try:
    issues = json.loads(result.stdout)
except json.JSONDecodeError:
    issues = []
    print("Model returned non-JSON output; skipping this diff.")
    print(result.stdout[:500])

print(f"Found {len(issues)} issues")
for issue in issues:
    print(issue["severity"], issue["line"], issue["message"])
```

Note the `try`/`except`. Structured output is a request, not a guarantee. Any pipeline that assumes valid JSON on the first try will eventually break in CI at the worst possible moment.

### Step 3: Triage the output

A representative result on a diff like this might be four findings: missing OpenAPI tags on the new endpoint, an async function without a timeout, a missing docstring, and a blocking call inside an async function. Of those, the timeout and the blocking call are the ones with real consequences. The docstring is style. The OpenAPI tag matters only if something downstream depends on it.

The triage ratio is the number that determines whether the tool is worth running. If three of four findings are noise, you are paying a context-switch tax for one useful signal. That can still be worth it — but only if you measure it.

### Step 4: Human review of what the model cannot see

The model will not tell you that the new offset/limit pagination can return duplicate rows when a concurrent insert shifts the window between page requests. It will not tell you that the ORM query is safe from injection only because the ORM parameterizes it, and would not be if someone switched to a raw string later. It will not validate that the OpenAPI schema matches the actual response shape.

Those checks remain human work. The model narrows the surface you have to inspect manually; it does not remove it.

## How this connects to tools you already use

If you run a linter in CI, you already have a static analyzer. An LLM reviewer is a static analyzer with a natural-language interface and a much larger, much fuzzier rule set. The difference is important: a linter enforces rules you wrote down, so its false-positive rate is bounded by your configuration. An LLM enforces rules it infers from training data plus your prompt, so its false-positive rate is bounded by nothing in particular.

The same comparison holds against fuzz testing. A fuzzer throws inputs at your code to find crashes. An LLM throws critiques at your diff to find inconsistencies. Both are probabilistic. Both miss deep logic errors. Both are assistants, not oracles.

This is why replacing an existing linter with an LLM is usually a mistake. A linter's rules are deterministic, cheap, and reproducible. Keep the linter for what it does well and add the LLM for the categories the linter cannot express — "every new endpoint has tags," "no blocking call in async context." Those are rules a linter could technically encode, but writing and maintaining a custom AST rule for each one is often more work than a prompt.

## Common misconceptions

**"It can review whole files, not just diffs."**

It can, but the output degrades badly. On a large file with a small change, the model has no signal about what is new, so it comments on everything. The result is a long list dominated by observations about code that has not changed in years. Diff-scoping is not a limitation to work around; it is the mechanism that makes the tool usable.

**"It understands types and imports."**

Partially. It reliably spots unused imports and internal type mismatches. It also invents import paths that look plausible and do not exist. A suggestion like `from app.models.user import UserModel` when the real path is `from app.models import User` is a one-token difference that breaks the build. Never apply an import fix without running the test suite.

**"It catches security issues."**

Sometimes, and unpredictably. It may flag a hardcoded credential in a config file because that pattern is heavily represented in training data. It is much less reliable on injection risks where the tainted value is assembled across several functions, because detecting that requires tracking data flow, not recognizing a shape. Keep parameterized queries, secret scanning, and human review for security.

**"It is cheaper than a human reviewer."**

The API cost is usually small. The real cost is the context switch. Every false positive forces a developer to stop, read the comment, evaluate it, and dismiss it. If that takes a few minutes each and you get several per PR, the accumulated interruption can exceed the value of the true positives. This is the number to measure before scaling up, and it is the reason prompt precision matters more than model choice.

## Measuring it on your own repository

Do not adopt published accuracy figures. They are measured on other codebases with other conventions. Run a small blind comparison instead.

1. Collect 30 to 50 recently merged PRs where a human reviewer left comments.
2. Run the LLM reviewer on each diff with your production prompt. Save the raw output.
3. For each PR, list the issues the human reviewer raised.
4. Classify every LLM finding as true positive (a real issue a human would also raise), false positive (noise), or novel (a real issue the human missed).
5. Compute precision as true positives divided by total findings, and recall as true positives divided by human-raised issues.

The two numbers tell you different things. Low precision means your prompt is too broad and you are generating interruption cost. Low recall means the tool is missing categories you care about — usually a sign you need to enumerate those categories explicitly in the prompt, or accept that they are out of scope.

What to instrument in production, once it is running:

- Findings per PR, split by severity
- Fraction of findings a human marks as actionable
- Time from comment posted to comment resolved or dismissed
- Number of merges blocked by an LLM finding that turned out to be wrong

A pipeline whose dismissal rate climbs over time is drifting. That usually means the prompt has not kept up with the codebase, or the model has started pattern-matching on your own past comments rather than the code.

## The advanced version, once the basics hold

**Custom rule sets from your own bug history.** Take your last twenty postmortems, extract the code shape that caused each incident, and add a line to the prompt for each one. "Flag any `requests.get` inside an async function" is a rule derived from a real outage, and it is far more valuable than a generic instruction to "find bugs."

**Pre-commit hooks.** Run the reviewer on staged changes before the test suite. Keep it advisory at first — print findings, do not fail the build. Once precision is high enough that developers trust it, you can exit non-zero on high-severity findings only.

```python
#!/usr/bin/env python3
import json
import subprocess
import sys

PROMPT = """Review this git diff for these anti-patterns only:
- Any synchronous HTTP call inside async context
- Any raw SQL without parameterization
- Any function longer than 50 lines added or changed
- Any new public method without a docstring

Return a JSON array with keys: issue, severity, line.
Return [] if none apply."""

def review_diff(diff: str) -> list:
    result = subprocess.run(
        ["claude", "--prompt", PROMPT, "--input", diff],
        capture_output=True, text=True, check=True,
    )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        print("Non-JSON response from reviewer; treating as no findings.")
        return []

if __name__ == "__main__":
    diff = subprocess.check_output(["git", "diff", "--cached"], text=True)
    if not diff.strip():
        sys.exit(0)
    issues = review_diff(diff)
    if issues:
        print(json.dumps(issues, indent=2))
        sys.exit(1)
```

**Blind benchmarking against your team.** The methodology above is the only honest way to know whether the tool is helping. Run it quarterly. Codebases change, prompts drift, and a configuration that was net-positive six months ago may not be today.

**Cost guardrails.** Set a hard daily ceiling on API spend and alert when you approach it. The failure mode is not a single expensive call; it is a hook that fires on every commit in a monorepo and quietly multiplies. Log token counts per invocation so you can see which diffs are expensive and why.

## Where it fails, and what to do instead

The failure modes cluster tightly:

- **Concurrency bugs.** Race conditions, deadlocks, and async starvation require simulating interleavings. A model reading a diff cannot do this. Cover these with stress tests and targeted integration tests.
- **Domain logic.** The model does not know that orders must be unique per user under concurrent writes. That invariant lives in your team's head and your schema. Write it down as a test.
- **Legacy drift.** In a codebase with years of accumulated convention, the model's priors about how code "usually" looks will be wrong. It will suggest imports and signatures from a generic Python project rather than yours.
- **Security.** Subtle injection risks and secrets in unusual locations are missed. Keep dedicated tooling for this category.

For all four, the answer is the same as it was before LLMs existed: tests, human review, and deliberate failure injection. The model reduces the volume of mechanical review work. It does not reduce the need for the parts of review that require understanding.

## A decision checklist

Before wiring an LLM reviewer into your workflow, answer these:

- Is the review scoped to the diff, not the file?
- Does the prompt enumerate specific, checkable rules rather than asking for general review?
- Does the prompt explicitly permit an empty result?
- Is the output parsed defensively, with a fallback when it is not valid JSON?
- Have you measured precision and recall on at least 30 of your own PRs?
- Do you know your false-positive cost in developer minutes per week?
- Is the tool advisory rather than merge-blocking until precision justifies otherwise?
- Is there a daily spend cap with alerting?
- Are concurrency, domain-logic, and security reviews still assigned to humans?

Any "no" is a gap worth closing before scaling.

## FAQ

**What prompt structure works best for Python diffs?**

An enumerated list of checkable rules, an explicit output schema, and an explicit empty-result option. Vague instructions like "find bugs" produce vague findings. Rules like "every new async function has an explicit timeout" produce findings you can verify in seconds.

**How do I keep false positives down?**

Three levers, in order of impact: narrow the prompt to specific rules, scope the input to the diff, and pin the model version so behavior does not shift under you. If precision is still poor, your rules are probably broader than your codebase's actual conventions.

**Can it replace a linter?**

No. A linter is deterministic and cheap; an LLM is probabilistic and costs tokens per run. Use the linter for anything expressible as a rule, and the LLM for the categories you would otherwise review by eye.

**Should it block merges?**

Not initially. Run it in advisory mode, measure precision, and only promote high-severity findings to blocking once the false-positive rate is low enough that developers do not route around it.

## Do this in the next 30 minutes

Pick one merged PR from the past week that a human reviewed. Run your diff through the prompt above, then classify each finding as true positive, false positive, or novel. If more than half are noise, rewrite the prompt to be narrower before running it on anything else.
