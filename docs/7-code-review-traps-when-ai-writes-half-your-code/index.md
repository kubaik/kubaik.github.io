# 7 code review traps when AI writes half your code

AI coding assistants change the failure modes that code review is meant to catch. Reviewers who are good at spotting human mistakes — copy-paste errors, inconsistent naming, missing null checks — often miss the failure modes that language models produce instead. This article describes seven of those traps, how to detect each one, and a lightweight checklist that addresses all of them.

## Why AI-generated code fails differently

Human code tends to fail in ways that correlate with effort and attention: code written in a hurry is shallow, code written by a junior has gaps in edge-case handling, code written under deadline pressure skips tests. Reviewers learn to calibrate for these signals.

Language models fail differently. They produce plausible, well-formatted code that compiles and often passes the tests they also wrote. The failure modes cluster around four themes:

- **Environment assumptions.** The model has no knowledge of your runtime, your dependency pins, or your deployment target. It writes code for the "average" environment, which is not yours.
- **Golden path bias.** Generated code handles the documented success case well and under-specifies error paths, expiry, rate limits, and partial failures.
- **Dependency drift.** The model suggests imports and versions based on training data, which may be older or newer than what your project pins.
- **Prompt artifacts.** Anything you put in the prompt — API keys, internal hostnames, customer names — can end up echoed into code, comments, or log statements.

None of these are exotic. Each one has a cheap detection method. The mistake is treating AI-generated code as if it were human code and applying only the review habits you already have.

## Trap 1: The silent import and environment mismatch

A model asked to write JWT handling in Python will often reach for a library it saw frequently during training. If your project pins a different library, or your runtime lacks the native crypto module the model assumes, the code may import cleanly on the model's mental model of your system and fail at runtime.

Detection: after any AI-generated diff, run the import check in a clean environment, not your dev shell:

```
python -c "import your_module"
```

Better, run it inside the same container image or virtualenv your CI uses. A local shell with a dozen globally installed packages will hide the problem.

For Node projects, `npm ls <package>` confirms the resolved version, and `node -e "require('your-module')"` confirms it loads under your Node version. The point is to test the dependency resolution your production environment will actually perform, not the one your laptop has accumulated.

## Trap 2: Prompt leakage into code and logs

If a prompt contains a credential, an internal URL, or a customer identifier, the model may reproduce it in a docstring, a debug log line, a comment, or a test fixture. This is not the model being malicious; it is the model being helpful with the context it was given.

Once committed, the value lives in Git history. Rotating the credential fixes the immediate exposure but not the historical record.

Detection: scan the diff, not just the final file. A grep for common credential shapes over the added lines catches most cases:

```
git diff --cached -U0 | grep -E '^\+' | grep -Ei 'api[_-]?key|secret|token|password|bearer'
```

This is a coarse filter and will produce false positives on variable names. That is acceptable; the cost of a false positive is a few seconds of reading.

Prevention is more reliable than detection. Keep secrets out of prompts entirely. Reference environment variable names, not values. If your tooling supports it, redact known secret patterns before the prompt leaves your machine.

## Trap 3: Golden path bias

Ask for a token validation function and you will usually get one that validates a well-formed, unexpired, correctly signed token. Ask for tests and you will usually get tests that cover exactly that case.

The missing cases are predictable: expired tokens, tokens with a `nbf` (not before) claim in the future, tokens signed with the wrong algorithm, malformed base64, missing claims, clock skew, and revoked tokens. These are the cases that matter in production and the cases the model did not think to write.

Detection: enumerate the error branches in the generated code and check that a test exists for each. A quick way is to list the `raise` and `return` statements in the function and confirm each has a corresponding test:

```
grep -nE 'raise |return ' path/to/module.py
```

If a branch has no test, that is a review finding regardless of whether the code looks correct.

A useful exercise is to ask the model explicitly for the negative cases after it has produced the happy path. Models are generally capable of writing edge-case tests when asked directly; they simply do not volunteer them.

## Trap 4: Dependency drift

Generated code may reference a version of a library that differs from your pin. Sometimes it suggests an older API that has since been deprecated; sometimes it suggests a newer one that does not exist in your lockfile.

Detection: after each AI-generated change, inspect what the diff does to your dependency manifests.

```
git diff -- requirements.txt requirements/*.txt pyproject.toml package.json package-lock.json
```

If the diff adds or bumps a dependency, treat that as a separate review item. Read the changelog for the affected range. Check whether the bump crosses a major version. Check whether the project's own release notes flag behavior changes.

A subtle version of this trap is an upgrade that fixes a security issue but introduces a performance regression. The security fix is visible in the advisory; the performance change is not. If you have latency monitoring in CI, compare the p95 of the affected code path before and after. If you do not, this is a good reason to add it.

## Trap 5: Model drift across runs

The same prompt does not always produce the same code. Model versions change, sampling parameters change, and provider-side updates happen without notice. A prompt that produced acceptable code last month may produce subtly different code today.

Detection: snapshot the generated output and compare across runs. The mechanism does not need to be elaborate — store the generated file, and on the next generation of the same prompt, diff the two.

Two cautions. First, whitespace and formatting differences will dominate the diff unless you normalize them; strip trailing whitespace and normalize indentation before comparing. Second, the threshold for "meaningful change" is a judgment call. A useful default is to flag any change that alters control flow or the set of imported symbols, and ignore pure formatting.

This trap is easy to over-engineer. A simple `diff` run manually when you regenerate code catches most of it.

## Trap 6: Tests that only test what the model wrote

Generated tests tend to assert the behavior of the generated implementation, including its bugs. If the implementation has an off-by-one in a boundary check, the test will happily assert the off-by-one behavior and pass.

Detection: for each generated test, ask what it would take to make the test fail. If the answer is "the code would have to be deleted," the test is not testing anything useful.

A more systematic check: mutate the implementation deliberately — flip a comparison operator, change a constant, remove a bounds check — and confirm the test suite fails. If a mutation survives, the test suite has a gap. This is mutation testing, and it is a well-established technique, though running it on every PR is usually too slow. Running it periodically on the modules that AI generates most often is a reasonable compromise.

## Trap 7: Review comments generated by a model

Using a model to draft review comments is tempting when review capacity is the bottleneck. The failure mode is that the comments read as authoritative while being wrong. A model that suggests `os.environ['SECRET_KEY']` without mentioning key-length validation, or that recommends a library function with different semantics than the one already in use, produces a comment that a tired reviewer will approve.

Detection: treat model-generated review comments as suggestions requiring verification, never as findings. Any comment that asserts a fact about a library's behavior should be checked against that library's documentation before it is acted on.

A practical rule: model-generated comments may point at code, but they may not assert correctness. "This branch has no test" is verifiable by looking. "This function is safe against timing attacks" is not.

## A worked example

Consider a generated token validation function:

```python
import os
import jwt

SECRET_KEY = os.environ.get("SECRET_KEY")

def validate_token(token: str) -> dict:
    return jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
```

Walking the traps:

1. **Environment mismatch.** `jwt.decode` with a `None` key raises at call time, not import time. If `SECRET_KEY` is unset in staging, the failure appears only when a request arrives. Detection: assert at startup that required configuration is present, rather than deferring to first use.
2. **Prompt leakage.** Nothing here, but check the surrounding diff for log statements that print the token or the key.
3. **Golden path bias.** No expiry handling beyond what the library does by default, no handling of `nbf`, no clock-skew tolerance, no distinction between an expired token and a malformed one. Each of these is a separate test case that does not exist yet.
4. **Dependency drift.** Confirm the pinned version of the JWT library is the one this code was written against, and that `algorithms=["HS256"]` is the intended algorithm rather than the model's default.
5. **Model drift.** If this function is regenerated later, the algorithm list or the error handling may change silently.
6. **Test quality.** A test that asserts a valid token decodes will pass regardless of whether expiry is enforced. Add a test that asserts an expired token raises.
7. **Review comments.** If a model drafted the review, verify any claim it makes about the library's default behavior.

The function is not wrong. It is incomplete in ways that a human reviewer focused on syntax and structure will not notice.

## A checklist that fits in a PR template

The checklist below is deliberately short. Long checklists get skipped; short ones get used.

```markdown
- [ ] No secrets, keys, or internal hostnames in the diff (including comments and logs)
- [ ] Imports resolve in the CI environment, not just locally
- [ ] Every error branch in new code has a corresponding test
- [ ] Dependency changes are reviewed against the changelog
- [ ] Generated code regenerated since last review has been diffed
- [ ] Tests would fail if the implementation were mutated
- [ ] Model-generated review comments have been verified against documentation
```

Store it where reviewers will see it: a pull request template, a required status check, or a comment on the PR. The location matters less than the fact that it appears before approval.

## Choosing what to adopt first

Not every team needs all seven checks. A rough prioritization:

| Situation | Start with | Why |
|---|---|---|
| Handling credentials or PII | Prompt hygiene and diff scanning | A leak in Git history is expensive to undo |
| High-traffic service | Golden path and mutation checks | Missing error branches surface under load |
| Many contributors, fast merge cadence | PR-template checklist | Cheap, visible, catches the common cases |
| Long-lived codebase with pinned deps | Dependency diff review | Drift accumulates quietly |
| Frequent regeneration of the same code | Snapshot diffing | Detects model changes between runs |

The checklist is the cheapest starting point because it requires no tooling and no CI changes. The other checks are worth adding as the team's confidence in its review process grows.

## FAQ

**Does this mean AI-generated code is less safe than human code?**
No. It means it fails differently. Human code has its own predictable failure modes, and reviewers have learned to catch those. The failure modes above are simply less familiar.

**How much review time does the checklist add?**
For most diffs, the checklist adds a minute or two of reading. The expensive items are dependency review and mutation testing, and those only apply when the diff touches those areas.

**Can a model review its own output?**
It can produce useful comments, but the comments need verification. The failure mode is a confident but wrong assertion, which is worse than no comment because it consumes the reviewer's attention.

**What if the team has no time to add process?**
Start with the pull request template. It is a single file and requires no CI changes. Add the other checks when a specific failure mode actually bites.

## Do this in the next 30 minutes

Create `.github/pull_request_template.md` in your repository and paste the checklist above into it. Commit and push. The next pull request opened in that repository will show the checklist, and reviewers will see it before they approve anything. Measure how often a checklist item catches something over the next twenty pull requests; if an item never fires, remove it, and if a new failure mode appears that no item covers, add one.
