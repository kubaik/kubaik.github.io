# Agent code review: 7 traps that slip through

Agent-generated code arrives with a confidence human code rarely has. It compiles, it passes the tests the agent wrote for itself, and it reads as though the author has internalized every style guide. The complication is that the same model also reproduces patterns from outdated tutorials, answers that were correct until a library changed its API, and conventions that predate current best practice. At team scale, the review habit built around "does this person understand what they're doing?" quietly stops working, because the agent always sounds like it understands.

The seven traps below are recurring patterns in agent output, each with a distinct fix. Most are not new mistakes. They are old mistakes with a new author, which is exactly why they slip through: the reviewer recognizes the shape of the code as familiar and stops looking at the details.

## What this list actually solves

Agent output changes the economics of code review. A human contributor who writes a `setTimeout`-based retry loop usually has a reason, even a bad one, and a reviewer can ask about it. An agent that writes the same loop is pattern-matching from training data, so the reviewer's usual signal — "this person seems to know what they're doing" — becomes actively misleading. The agent always seems to know what it's doing.

The review process therefore has to shift from judging intent to verifying behavior. Each trap below is a place where the code looks intentional but the behavior is wrong, and each comes with a concrete check a reviewer or a script can run.

## Evaluation criteria

A trap earns a place on this list if it meets three conditions:

1. **It passes the agent's own tests.** If the agent wrote a test that would catch the mistake, it usually does — the failures live in the seams between tests.
2. **It looks correct to a reviewer skimming for intent.** The code reads as deliberate, not sloppy.
3. **It fails in a way that is expensive to discover late.** Usually at integration, under load, or after a dependency upgrade.

That filter removes a lot of common complaints (formatting, naming, comment density) and leaves the ones that actually cost teams time.

## 1. The retry loop with no jitter and no cap

**What it does:** The agent generates a retry wrapper around an HTTP call or a database operation:

```python
import time
import requests

def fetch_with_retry(url, max_retries=5):
    for attempt in range(max_retries):
        try:
            return requests.get(url, timeout=10)
        except requests.RequestException:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"failed after {max_retries} attempts")
```

**Strength:** Exponential backoff is the right idea, and the agent gets the shape right almost every time. It is genuinely better than the naive `while True: try: ... except: pass` found in older code.

**Weakness:** There is no jitter, so every client hitting the same failing endpoint retries in lockstep. There is no cap on the backoff: with `max_retries=5` the sleeps are 1, 2, 4, 8 and 16 seconds, and raising the cap to 10 would put the last sleep at 512 seconds. There is also no distinction between retryable and non-retryable errors — a 400 gets retried like a 503. The pattern comes from tutorials that predate the widespread adoption of jittered backoff, and the agent reproduces it faithfully.

**Check:** Every agent-generated retry loop should be reviewed against three questions: is there jitter, is there a cap, and does it distinguish error classes? The documented AWS guidance on exponential backoff and jitter describes the full pattern; the agent usually implements the first third of it. A corrected version:

```python
import random
import time
import requests

RETRYABLE_STATUS = {429, 500, 502, 503, 504}

def fetch_with_retry(url, max_retries=5, base=1.0, cap=30.0):
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code in RETRYABLE_STATUS:
                raise requests.HTTPError(f"retryable status {resp.status_code}")
            resp.raise_for_status()
            return resp
        except requests.RequestException:
            if attempt == max_retries - 1:
                raise
            # full jitter: sleep uniformly between 0 and the capped backoff
            backoff = min(cap, base * (2 ** attempt))
            time.sleep(random.uniform(0, backoff))
```

## 2. The test that asserts the implementation, not the behavior

**What it does:** The agent writes tests alongside the code, which is good. But the tests tend to assert on internal calls rather than observable outcomes:

```python
from unittest.mock import patch

def test_process_order():
    with patch("orders.db.save") as mock_save:
        process_order({"id": 1, "total": 100})
        mock_save.assert_called_once_with({"id": 1, "total": 100, "status": "pending"})
```

**Strength:** The test exists, it runs, and it exercises the code path. Coverage tools report the line as covered.

**Weakness:** The test is coupled to the exact call signature of an internal function. The next refactor that changes `save` to take a model object instead of a dict breaks this test, and the fix is to update the mock — after which the test verifies nothing about behavior. Worse, the test passes even if `save` silently drops the data, because the mock never saves anything. The agent has produced the appearance of testing without the substance.

**Check:** Every mock assertion should be paired with at least one assertion on a real observable outcome — a return value, a database row, an HTTP response. A behavior-focused version against a real (or in-memory) database:

```python
def test_process_order_persists_pending_order(db):
    process_order({"id": 1, "total": 100})
    row = db.query("SELECT id, total, status FROM orders WHERE id = 1").one()
    assert row == (1, 100, "pending")
```

If the test can't express an observable outcome, the code under test is probably too coupled to its collaborators.

## 3. The dependency pinned to a version that no longer exists

**What it does:** The agent adds a dependency to `package.json`, `requirements.txt`, or `go.mod`, and the version it picks is whatever was current in its training data. Sometimes that version has been yanked. Sometimes the API has changed. Sometimes the package has been renamed.

**Strength:** The dependency is usually real and the version was usually valid when it was written into the model's weights. The agent is not hallucinating a package from nothing — it is reproducing a real historical state.

**Weakness:** The lockfile resolves, the install succeeds, and then a transitive dependency has a CVE that was patched two minor versions later. Or the agent uses an API removed in a major version bump, and the code fails at runtime with a `TypeError` the test suite doesn't catch because the suite was written against the same stale API. This is where the outdated-tutorial origin is most visible: the code targets a version of the library that no longer exists.

**Check:** Require that every new dependency in an agent PR comes with a lockfile diff and a note on which version the agent targeted. Compare that target against the current release, and if it is more than one major version behind, treat the PR as needing a manual upgrade pass before merge. In CI, run the dependency audit tool for your ecosystem against the lockfile and fail on known advisories rather than reviewing versions by eye.

## 4. The async function that is secretly synchronous

**What it does:** The agent writes `async def` functions that call blocking I/O inside them — `requests.get`, `time.sleep`, a synchronous database driver — under the assumption that `async` makes code concurrent:

```python
import asyncio
import requests

async def fetch_all(urls):
    results = []
    for url in urls:
        results.append(requests.get(url, timeout=10).text)
    return results
```

**Strength:** The function signature is correct, the call site is correct, and the code runs. In a single-request test it behaves exactly as expected.

**Weakness:** The event loop is blocked for the duration of every request. Under concurrency this is worse than plain synchronous code, because the caller believes it has concurrency and allocates resources accordingly. The failure mode is a p99 latency spike that looks like a downstream problem until someone profiles the event loop. The pattern appears because a large amount of tutorial code uses `requests` inside `async def` without explanation.

**Check:** Search the diff for `requests.`, `time.sleep`, and synchronous database drivers inside any `async def`, and require an async-native equivalent. A corrected version:

```python
import asyncio
import httpx

async def fetch_all(urls):
    async with httpx.AsyncClient(timeout=10) as client:
        responses = await asyncio.gather(*(client.get(u) for u in urls))
        return [r.text for r in responses]
```

## 5. The error handler that swallows the stack trace

**What it does:** The agent wraps a block in `try/except` and logs a message without the exception object, or catches a broad exception class and continues:

```python
try:
    result = risky_operation()
except Exception as e:
    logger.error(f"operation failed: {e}")
    return None
```

**Strength:** The error is handled, the process doesn't crash, and there is a log line. For a prototype, this is reasonable.

**Weakness:** `str(e)` for many exception types is empty or uninformative. The stack trace — the part that says where the failure originated — is discarded. Catching `Exception` sweeps up programming errors alongside operational ones, so a `TypeError` from a typo is logged as "operation failed" and the function returns `None`, which then propagates as silent data loss. The pattern comes from tutorials that prioritized "don't crash" over "fail loudly."

**Check:** Every `except` block must either re-raise, log with `logger.exception(...)` (which captures the traceback), or return a typed error the caller is forced to handle. A bare `return None` in an exception handler is a red flag in review:

```python
try:
    result = risky_operation()
except SomeOperationalError:
    logger.exception("operation failed")
    raise
```

## 6. The workflow that assumes idempotency it doesn't have

**What it does:** The agent generates a multi-step workflow — charge a card, send an email, update a record — and wires it together without considering what happens when step 2 fails after step 1 succeeded. Retries at the workflow level then double-charge or double-send.

**Strength:** The happy path is correct and readable. Each step is individually well written.

**Weakness:** The workflow has no idempotency key, no compensation step, and no dead-letter path. When the retry logic from trap #1 kicks in, the first step runs again. This is the trap most expensive to discover in production, because the failure is a business-logic failure (a duplicate charge) rather than a technical one.

**Check:** For every step that causes an external side effect, ask what happens if the step before it retries. If the answer is "it runs twice," the workflow needs an idempotency key or an outbox pattern before merge. A worked example, reasoning shown:

- Step 1: `POST /charges` with body `{"amount": 100}`.
- Step 2: `POST /emails` with the receipt.
- Step 2 times out and the workflow retries from step 1.
- The charge endpoint has no idempotency key, so it creates a second charge.

The fix is to generate one idempotency key per logical workflow run and send it with every attempt of step 1. The payment provider then returns the original charge on the second attempt instead of creating a new one. The key must be generated once per run and persisted, not regenerated inside the retry loop — regenerating it reproduces the bug.

## 7. The configuration that hardcodes a value the environment should provide

**What it does:** The agent writes a connection string, a timeout, a feature flag, or a model name directly into the code, often with a comment like `# adjust as needed`.

**Strength:** The code runs immediately in the environment the agent was targeting. No setup required.

**Weakness:** The value is now wrong in every other environment. Worse, if the value is a credential or an endpoint, it may be a real one reproduced verbatim from a tutorial. The `# adjust as needed` comment is the tell: the agent knows the value is environment-specific but has no mechanism to express that. This is the oldest trap on the list, and it persists because it is the one that most reliably produces a working demo.

**Check:** Grep the diff for hardcoded URLs, ports, model names, and timeout literals. Each one should be a named constant sourced from environment configuration, with a documented default:

```python
import os

PAYMENTS_BASE_URL = os.environ.get("PAYMENTS_BASE_URL", "http://localhost:8080")
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "10"))
```

## The strongest default, and why

If a team can adopt only one practice, it should be the **behavioral review checklist** — a short, mechanical set of greps and questions applied to every agent PR before a human reads it for intent. Six of the seven traps above are detectable without understanding the code's purpose. You don't need to know what the service does to notice a missing jitter, a `requests.get` inside `async def`, or a bare `return None` in an `except` block.

A minimal version, runnable as a CI step:

```bash
#!/usr/bin/env bash
# agent-review-checks.sh — run on the diff before human review
set -euo pipefail

diff=$(git diff origin/main...HEAD)

# Trap 4: blocking calls inside async functions
if echo "$diff" | grep -qE '^\+.*async def' && echo "$diff" | grep -qE '^\+.*(requests\.|time\.sleep)'; then
  echo "WARN: possible blocking call inside async function"
fi

# Trap 5: swallowed exceptions
if echo "$diff" | grep -qE '^\+.*except Exception' && echo "$diff" | grep -qE '^\+.*return None'; then
  echo "WARN: exception handler may return None"
fi

# Trap 7: hardcoded endpoints and credentials
if echo "$diff" | grep -E '^\+.*https?://' | grep -vqE 'example\.com|localhost'; then
  echo "WARN: hardcoded URL in diff"
fi
```

This is illustrative, not exhaustive. It catches three of the seven traps and produces false positives, and it is line-based, so it will miss multi-line constructs. But it shifts the review conversation from "does this look right?" to "why did the check fire?", which is a much better use of a senior reviewer's time.

## How to measure whether the checks are working

Do not trust anecdotes about how many bad PRs the checks caught. Instrument the process instead:

- **Warning rate per trap.** Count how many times each check fires per week. A check that never fires is either unnecessary or broken; a check that fires on most PRs is too broad.
- **Precision.** For each warning, record whether a reviewer confirmed it as a real defect. Precision is confirmed defects divided by warnings. Track it per check, not in aggregate — the aggregate hides which rule is noisy.
- **Escape rate.** Count defects that reached production and would have been caught by a check that exists. This is the number that justifies the process; it is also the slowest to move.
- **Review time.** Time from PR open to first substantive human comment. A useful mechanical pass should reduce this, because the reviewer no longer spends the first ten minutes on missing jitter and hardcoded URLs.

A simple way to start: add a label to agent-authored PRs, log every check warning and its resolution in the PR thread, and review the counts monthly. Compare a few weeks before and after adding a check rather than assuming it helped.

## Honorable mentions worth knowing about

A few more patterns that don't quite make the top seven but show up often enough to recognize:

- **The `datetime.utcnow()` call.** Documented as deprecated in modern Python in favor of timezone-aware `datetime.now(timezone.utc)`, but present in a large amount of tutorial code. The failure is subtle: comparing naive and aware datetimes raises `TypeError`, usually in a test path that wasn't exercised.
- **The `SELECT *` in an ORM query.** Fine for a demo, expensive when a column is added to a wide table. Agents generate it because it is shorter and reads as simpler.
- **The `console.log` left in a production path.** Not a correctness bug, but a signal that the agent was iterating without a debugger and the reviewer should check what else was left behind.

## Options that look appealing but fail in practice

Two approaches to agent code review are popular and don't hold up at team scale:

**"Let the agent review the agent."** Running a second model over the first model's output catches some surface issues but shares the same training-data biases. If the first agent learned the un-jittered retry pattern, the second is likely to consider it correct. It is useful as a first pass for style and obvious bugs, not as a substitute for the mechanical checks above.

**"Require 100% test coverage for agent PRs."** Coverage measures which lines executed, not which behaviors were verified. Trap #2 — the mock-heavy test — produces high coverage and verifies almost nothing. A coverage gate on agent PRs tends to produce more tests of the same shape, not better ones.

## How to choose based on your situation

| Situation | First practice to adopt |
|---|---|
| Small team, agent PRs are a minority | The mechanical grep checklist in CI |
| Agent PRs are the majority of changes | Behavioral review checklist plus a rule that every mock assertion has a real-outcome counterpart |
| Workflows touch external side effects | Idempotency review before merge, no exceptions |
| Multiple deployment environments | Hardcoded-value grep plus a config-schema test |
| Heavy reliance on generated tests | Replace coverage gates with mutation testing on critical paths |

The common thread: pick the practice that matches the failure mode you are most likely to hit, not the one that is easiest to automate. The mechanical checks are cheap and catch the most traps; the behavioral rules are more expensive but catch the ones that cause incidents.

## Frequently Asked Questions

**How do I review agent-generated code without reading every line?**

You can't skip reading, but you can front-load the mechanical checks so the reading is focused. Run the grep-based checks on the diff first, resolve the warnings, then read the code for intent. The mechanical pass removes the low-value review comments (missing jitter, hardcoded URLs) and leaves the reviewer free to think about whether the workflow is correct.

**Why does agent code pass tests but fail in production?**

The tests were written by the same agent that wrote the code, against the same stale assumptions. If the agent believes `requests.get` inside `async def` is fine, it writes a test that calls the function once and asserts the result — which passes. The failure only appears under concurrency, which the test never exercises. Require at least one test per agent PR that exercises the code under the conditions it will actually face: concurrent calls, retries, or a failing dependency.

**What is the most common mistake in agent-generated retry logic?**

Missing jitter. Exponential backoff without jitter causes synchronized retries across clients, which turns a transient downstream failure into a thundering herd. The second most common is retrying non-retryable errors — a 400 or a validation failure gets the same treatment as a 503. Both are correct in the tutorial the agent learned from and wrong in production.

**Should agent-generated code be held to a different review standard than human code?**

Yes, but not a higher one — a more mechanical one. Human code carries intent a reviewer can interrogate. Agent code carries pattern-matching a reviewer can't interrogate, because the agent will confidently explain a wrong choice. The standard should emphasize verifiable behavior over apparent intent: does the retry have jitter, does the exception handler preserve the traceback, does the workflow tolerate a retry. Those questions have objective answers, which is exactly what you want when the author's confidence is not a signal.

## Final recommendation

Start with the mechanical checks. Open the file that defines your CI pipeline (`.github/workflows/`, `.gitlab-ci.yml`, `Jenkinsfile`) and add a step that runs the grep checks against the diff of any PR labeled as agent-generated. It will produce false positives and miss traps, and that's fine: the goal is to move the first pass of review from a human reading for intent to a script checking for behavior. Once that step is in place, human review can focus on the two traps no grep can catch — the mock-heavy test and the non-idempotent workflow — because those are the ones that cost the most to discover late.

**Do this in the next 30 minutes:** add the three grep checks above to your CI pipeline as a non-blocking warning step, open a test PR, and confirm each warning fires on a deliberately bad snippet and stays silent on a clean one.
