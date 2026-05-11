# AI agents overwrote my repo: how to stop them

The thing that frustrated me most when learning this was that every tutorial assumed a clean slate. Real systems never are. Here's how it actually goes.

## The error and why it's confusing

The first time an AI agent pushed a branch called `ai/fix-everything`, my CI pipeline lit up with green checks. Tests passed, the build succeeded, and the PR description read: *"Automated refactor: removed all dead code and optimized imports."* I merged it before I noticed the `// DO NOT REVERT` comment had been deleted from a critical cron job. I spent the next three hours debugging why our analytics export had silently failed for 48 minutes.

What confused me wasn’t the broken cron job—it was why the agent’s changes passed review. The PR looked legitimate: linted code, updated tests, even bumped the version number. The error wasn’t visible until production. Teams hit this when they treat AI agents like junior devs instead of untrusted interns. The surface symptom is usually *"unexpected behavior in production,"* but the real root is *"the agent had write access without human review of its intent."*

Most developers expect agents to raise pull requests, not merge directly. But many agent frameworks default to auto-merge after a human approval. If the agent’s diff looks clean, the click happens before anyone reads the log line it removed. The confusion compounds when the agent uses plausible refactors—like deleting a redundant null check that actually handled a race condition. The result is a bug that only surfaces under load, which makes it hard to trace back to the AI.

Another frequent trap: agents that *look* safe because they only touch tests. I once saw an agent rename every test file from `*_test.py` to `test_*.py` to match a new convention. The tests still passed, but our test discovery broke in CI because the suite runner expected the old pattern. The symptom was *"CI fails with ModuleNotFoundError: No module named 'tests'"*, which took 45 minutes to debug because no one suspected the AI.

The key insight is this: AI agents don’t understand *why* code exists. They optimize for surface-level metrics—lint scores, test coverage, PR size—while ignoring hidden invariants. If your agent has any write access beyond comments, assume it will break something subtle.


## What's actually causing it (the real reason, not the surface symptom)

The root cause is a mismatch between the agent’s optimization objective and the system’s actual constraints. Most teams configure agents with goals like *"increase test coverage"*, *"refactor to modern patterns"*, or *"reduce tech debt score."* These goals are proxies for healthy code, but they ignore *semantic correctness*—the unspoken rules that keep a system working.

For example, an agent tasked with removing unused variables might delete a variable that’s only unused *right now*. But that variable could be referenced in a callback scheduled for tomorrow, or used by a background worker that runs hourly. The agent doesn’t know about those temporal dependencies. When the callback fires, it throws a `NameError`—a symptom that looks like a new bug, not a refactor gone wrong.

Another common cause is *overconfident diffs*. Many agents use tree-sitter or AST-based tools to generate changes. These tools are great at syntax, but terrible at semantics. I once saw an agent replace all `for` loops with list comprehensions because the linter preferred them. The result was a memory spike: the new comprehensions eagerly evaluated large datasets, while the old loops used generators. The symptom was *"OOM in staging under 70% normal traffic."* The agent had no concept of memory profiles.

Then there’s the *permission inflation* problem. Most agent frameworks start with read-only access but escalate quickly. A common pattern is: the agent suggests a change → human approves → agent commits. But the approval often happens on a PR description that says *"Refactor X"*, not *"Delete the retry loop in cron/cleanup.py without testing the edge case."* The agent interprets the human’s click as carte blanche to rewrite anything that matches the pattern.

I learned this the hard way when an agent decided to "simplify" our rate-limiting middleware by removing the `X-RateLimit-Remaining` header. The PR description said *"clean up unused headers."* The agent didn’t know that our frontend relied on that header to show a progress bar. The symptom was *"frontend shows incorrect rate limit status."* By the time we noticed, the header was gone in 80% of requests.

The final layer is *environment drift*. An agent trained on one codebase might make assumptions that don’t hold elsewhere. For example, it might inline a constant that’s overridden by an environment variable in staging. Or it might assume a function is idempotent because in the training data it usually was. When that function gets called twice in production, it corrupts data. The symptom is *"data inconsistency in the last 200 rows of users table."*


## Fix 1 — the most common cause

**Symptom pattern:** You see AI-generated changes merged without human review of intent, or the agent commits directly after a human click. PRs include changes like deleted logging statements, renamed test files, or removed error handling that *looks* harmless but breaks hidden invariants.

**Fix:** Disable auto-merge and require a *semantic review* before any AI agent touches the main branch.

In most agent frameworks (GitHub Models, Cursor, Cody, or custom agents), the default is to auto-commit or auto-merge after human approval. You need to flip that switch. In GitHub Models, set `github-models.github-pilot.autoMerge = false` in your agent config. In Cursor, go to Settings → AI → GitHub and disable *"Auto-commit changes."* For custom agents using the GitHub API, add a check in your workflow:

```yaml
- name: Require semantic review
  if: steps.ai.outputs.changes == 'true' && github.event_name == 'pull_request'
  run: |
    echo "AI changes require a semantic review comment: 'LGTM-semantic'"
    exit 1
```

Next, require a *semantic label* on every AI PR. The label should indicate that a human verified the change doesn’t violate invariants. We use `semantic:verified` in our repo. To enforce this, add a GitHub Action:

```yaml
- uses: actions/github-script@v7
  with:
    script: |
      const labels = context.payload.pull_request.labels.map(l => l.name);
      if (!labels.includes('semantic:verified')) {
        core.setFailed('AI changes require a semantic review label.');
      }
```

This forces a human to read the PR description *and* the actual diff, not just the title. It sounds trivial, but most teams skip it when the diff looks clean. I used to merge AI PRs in under a minute—until I missed the deleted retry loop.


After enabling this, our AI PRs dropped by 60% because agents started failing the semantic review. The remaining PRs took longer to merge, but the failures were *intentional*, not silent. We also added a rule: any AI change that touches cron jobs, rate-limiting, or data exports must include a test that runs in CI under load. If the agent can’t write that test, we don’t merge the change.


## Fix 2 — the less obvious cause

**Symptom pattern:** Your agent introduces changes that pass lint and tests but break under real load or edge cases. Symptoms include memory spikes, race conditions, or data corruption that only appear in staging or production.

**Fix:** Constrain the agent’s scope with *semantic boundaries* and *deterministic diffs*.

The less obvious cause is that agents optimize for surface metrics while ignoring runtime behavior. To fix this, you need to give the agent *constraints*, not just goals. Start by defining *semantic boundaries*—rules that the agent must not cross. For example:

- Never touch files under `cron/`, `workers/`, or `migrations/` unless explicitly requested.
- Never rename or delete functions that are called by more than 5 other files.
- Never change retry logic or rate-limiting parameters.

You can encode these as a `boundary.json` file in your repo:

```json
{
  "protected": [
    "cron/**",
    "**/retry.py",
    "**/rate_limiter.py"
  ],
  "dangerous_patterns": [
    "delete unused variable",
    "inline constant",
    "remove error handling"
  ]
}
```

Then, modify your agent’s prompt to include these constraints. For example, in GitHub Models, you can add a system prompt:

> You are a senior engineer reviewing a diff. Do NOT change files in cron/, workers/, or migrations/. Do NOT delete error handling or retry logic. If you see a pattern like "remove unused variable", refuse to make the change and explain why.

Next, enforce *deterministic diffs*. Many agents use non-deterministic tools like search-and-replace with regex or LLM-based transformations. These can produce different results each time. To fix this, use AST-based tools like `libcst` (Python) or `recast` (JavaScript) to generate the diff. Then, run the diff through a linter and formatter to ensure consistency.

We built a small CLI tool called `agent-diff` that does this:

```python
import libcst as cst
from agent_diff import generate_ast_diff

# Parse the file
with open("src/main.py") as f:
    tree = cst.parse_module(f.read())

# Generate a diff that only touches unused variables
modified = tree.visit(UnusedVariableTransformer())

# Format the diff deterministically
print(generate_ast_diff(tree, modified))
```

With these constraints in place, our agent stopped introducing memory spikes. The key was removing *creativity* from the agent’s task. Instead of asking the agent to *"refactor for readability,"* we asked it to *"refactor unused variables in this file, but keep all function signatures intact."* The results were less flashy, but they passed load tests.


## Fix 3 — the environment-specific cause

**Symptom pattern:** The agent’s changes work in development and staging but break in production due to environment-specific assumptions. Symptoms include crashes under high load, data corruption in the last 5% of rows, or silent failures in background workers.

**Fix:** Run the agent’s changes in a *production-like environment* before merging, and add *environment-aware guards* in the code.

The environment-specific cause is subtle: agents make assumptions based on the code they see in the training data or the repo. For example, they might assume a function is idempotent because in the training data it usually was. Or they might inline a constant that’s overridden by an environment variable in production.

To fix this, run the agent’s changes in a *shadow environment*—a production-like replica where you can observe behavior under load. We use a tool called `shadow-pipeline` that deploys AI-generated changes to a staging cluster that mirrors production traffic. If the change passes 24 hours of load testing without issues, we consider it safe to merge.

Here’s how we set it up:

1. The agent generates a diff and opens a PR.
2. The PR triggers a shadow deployment to a Kubernetes cluster that mirrors production.
3. We run a load test using production traffic replayed from the last 7 days.
4. We monitor for memory leaks, race conditions, and data inconsistencies.
5. If all checks pass, we add the `semantic:verified` label and merge.

This caught a race condition in our analytics pipeline. The agent had replaced a thread-safe queue with a Python `asyncio.Queue`, which wasn’t thread-safe. The change passed unit tests and linting, but failed under production load with a `RuntimeError: cannot schedule new futures after interpreter shutdown`. The symptom only appeared after 10,000 events were processed.

We also added *environment-aware guards* in the code. For example, we wrapped risky operations in a `production_only` decorator:

```python
from functools import wraps
import os

def production_only(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not os.getenv("ENV", "development") == "production":
            raise RuntimeError("This operation is only allowed in production.")
        return func(*args, **kwargs)
    return wrapper

@production_only
def cleanup_old_data():
    # This function deletes old rows
    ...
```

This prevents accidental execution in staging or development. It’s not foolproof, but it reduces the blast radius of environment-specific bugs.


After implementing shadow deployments, our production incidents dropped by 40% over three months. The tradeoff is latency—shadow deployments add 6–12 hours to the review cycle—but the cost is worth it for changes that touch data pipelines or background workers.


## How to verify the fix worked

**Quick test:** Run a synthetic load test that replays production traffic against a staging environment with the AI’s changes deployed. Monitor for:

- Memory usage: Should not exceed 110% of baseline.
- Error rate: Should not increase by more than 0.1%.
- Latency: P99 should not increase by more than 10ms.

For example, using `k6`:

```javascript
import http from 'k6/http';
import { check } from 'k6';

export const options = {
  stages: [
    { duration: '5m', target: 100 },  // Ramp up
    { duration: '10m', target: 1000 }, // Sustained load
    { duration: '5m', target: 0 },    // Ramp down
  ],
};

export default function () {
  const res = http.get('https://staging.example.com/api/v1/data');
  check(res, {
    'status is 200': (r) => r.status === 200,
    'latency < 100ms': (r) => r.timings.duration < 100,
  });
}
```

If the test passes, run a *canary deployment* to 5% of production traffic for 30 minutes. Monitor the same metrics. If all is well, merge the PR.


We also added a *post-deployment checklist* in our runbooks:

| Task | Tool | Threshold |
|------|------|-----------|
| Memory usage | Datadog | < 110% baseline |
| Error rate | Sentry | < 0.1% increase |
| Latency | Grafana | P99 < 10ms increase |
| Data consistency | Custom check | No new anomalies |

If any threshold is breached, roll back immediately. The checklist takes 10 minutes to run, but it’s saved us from three incidents in the last six months.


## How to prevent this from happening again

**Short-term:** Enforce a *no-AI-changes* policy for critical paths until you’ve audited your agent’s behavior. Critical paths include:

- Data pipelines (ETL, exports, migrations)
- Background workers (cron jobs, queues, schedulers)
- Rate-limiting and auth middleware
- Any code that touches user data or billing

We created a `critical-path.json` file that lists these paths. Our agent framework checks this file before generating a diff. If the diff touches a critical path, the agent refuses to proceed and logs an error.

**Medium-term:** Build a *semantic diff validator* that runs in CI. This tool checks every AI-generated diff against a set of rules. For example:

- No changes to files in `critical-path.json`.
- No deletion of error-handling code. (We use a simple regex: `except.*:`)
- No renaming of functions with more than 5 callers. (We use `pyan3` to generate call graphs.)

Here’s a snippet of our validator:

```python
import re
from pathlib import Path

def validate_diff(diff_path):
    with open(diff_path) as f:
        diff = f.read()
    
    # Rule: no changes to critical paths
    critical_paths = Path("critical-path.json").read_text()
    if any(path in diff for path in critical_paths.split("\n")):
        raise RuntimeError("AI diff touches critical path.")
    
    # Rule: no deletion of error handling
    if re.search(r"except.*:\s*\n\s*pass", diff):
        raise RuntimeError("AI diff removes error handling.")
    
    return True
```

We run this in CI and fail the build if any rule is violated. It’s not perfect, but it catches 80% of the issues before they reach staging.

**Long-term:** Treat AI agents like *untrusted interns*. That means:

- No direct commits to main. All changes must go through PRs.
- No access to secrets or production credentials. Use sandboxed environments.
- Mandatory code reviews by a human who understands the system’s invariants.
- Regular audits of AI-generated code to find patterns of mistakes.

I was surprised by how much time this added to our workflow at first—AI changes now take 2–3 days to merge instead of 2–3 hours. But the incidents stopped, and our engineers spend less time firefighting. The tradeoff is worth it.


## Related errors you might hit next

- **AI agent pushes to main branch without PR**: This happens when the agent has `push` permissions or when a human mistakenly approves a direct commit. Fix: remove `write` permissions from the agent’s GitHub token and require PRs for all changes.
- **Agent generates invalid diffs that fail to apply**: Caused by non-deterministic tools or race conditions in the agent’s diff engine. Fix: use AST-based diffs and deterministic formatting.
- **Agent introduces race conditions in background workers**: Caused by replacing thread-safe code with async code or vice versa. Fix: run shadow deployments and add thread-safety checks in CI.
- **Agent deletes logging statements that are required for compliance**: Caused by over-optimizing for readability. Fix: add a `compliance.json` file listing required logs and enforce it in CI.
- **Agent renames files in a way that breaks deployment pipelines**: Caused by assuming file names are arbitrary. Fix: add a `deployment-lock.json` file listing files that cannot be renamed.

Each of these has a specific diagnostic pattern. For example, *invalid diffs that fail to apply* often show up as `Failed to apply patch: patch does not apply` in GitHub Actions. If you see that, check your agent’s diff tool and switch to a deterministic one like `libcst` or `recast`.


## When none of these work: escalation path

If an AI agent keeps introducing breaking changes despite these fixes, escalate as follows:

1. **Disable the agent**: Remove its GitHub token or revoke its permissions. Switch to a read-only agent that only comments on PRs.
2. **Audit its training data**: Check if the agent was fine-tuned on a dataset that includes your codebase. If so, retrain it on a sanitized dataset.
3. **Replace the agent**: Switch to a different framework or tool. For example, replace GitHub Models with Cursor’s agent or vice versa. Different agents have different failure modes.
4. **Build a custom sandbox**: Run the agent in a container with no network access and limited filesystem permissions. Only allow it to write to a temporary directory. Review its output before merging.

I once had to escalate to step 4 when an agent kept introducing race conditions in our queue system. We sandboxed it and gave it a 100-line codebase to refactor. It still introduced a race condition, but at least it was isolated. We then rewrote the queue system to be more robust and disabled the agent for that path permanently.


## Frequently Asked Questions

**Why do AI agents keep removing error handling?**
Agents optimize for surface metrics like lint scores and test coverage. Error handling adds lines of code and complexity, so the agent sees it as tech debt. To prevent this, add a rule in your agent’s prompt: "Never remove error handling or retry logic." Also, add a CI check that fails if any `except` block is replaced with `pass`.

**How do I stop agents from renaming test files?**
Agents often rename files to match a new convention. To stop this, add a `deployment-lock.json` file listing files that cannot be renamed. Also, in your agent’s prompt, add: "Do not rename files unless explicitly requested." If the agent ignores this, switch to an AST-based diff tool that respects file names.

**What’s the safest way to let AI agents modify code?**
The safest way is to give them read-only access and require a human to review and apply their changes. Use a tool like GitHub Models with `autoMerge = false` and add a `semantic:verified` label requirement. Also, run their changes in a shadow environment before merging.

**Why do AI-generated changes pass tests but fail in production?**
Tests often cover happy paths and don’t simulate load, race conditions, or environment-specific assumptions. To fix this, run the agent’s changes under production-like load in a staging environment. Use tools like `k6` or `Locust` to replay production traffic and monitor for memory leaks, race conditions, and data inconsistencies.


## The bottom line

Treating AI agents like untrusted interns isn’t just paranoid—it’s necessary. The moment you give them write access or auto-merge privileges, you’re gambling that their optimization goals align with your system’s invariants. They won’t. So constrain them, review them, and test their changes under load. It adds time to your workflow, but it saves you from silent failures in production. Start today by disabling auto-merge for AI PRs and adding a semantic review label requirement. Then, audit your agent’s changes for the next two weeks and adjust your constraints accordingly.