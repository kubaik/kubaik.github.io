# Who signs off on AI code?

AI coding assistants compress the distance between intent and diff. A prompt produces a patch; the patch passes CI; the patch merges. What that flow often omits is the step that matters most in high-stakes codebases: a named human approving code they did not write, cannot fully trace, and may not understand line by line. That is not a tooling gap. It is an ownership gap, and it widens as teams grow and as generated diffs get larger.

The documented behavior of most assistants — suggestion acceptance, inline completion, chat-driven edits — says nothing about who is responsible when a suggestion is wrong. The responsibility does not transfer to the model. It stays with the person who clicked merge.

## The gap between demo behavior and production requirements

Vendor material for AI coding assistants tends to describe suggestion acceptance and time saved. It rarely describes the moment a junior developer merges a 200-line generated function that passes tests but quietly changes a retry policy from exponential backoff to fixed one-second intervals. That change will not fail CI. It will fail at 02:00 when a downstream service starts returning 429s.

A typical failure mode is not "the AI wrote bad code." It is "the AI wrote plausible code that satisfied the tests but changed a behavioral contract nobody was watching." Tests encode what someone thought to assert. Review encodes what someone actually understands. Generated code can pass the first while failing the second.

Constraints make this harder in some environments than others. Teams working with intermittent power during deploy windows, limited per-seat tooling budgets, or few senior reviewers available cannot afford a dedicated platform engineer to audit every AI-assisted commit. They need a lightweight process that preserves accountability without adding half an hour to every pull request.

The hard question is not whether AI can write code. It is who is responsible when it writes the wrong code, and how to make that responsibility explicit before the merge button is pressed.

## The three mechanical guarantees

Accountability in an AI-augmented team rests on three properties: traceability, bounded trust, and explicit sign-off.

Traceability means every AI-generated hunk is marked in the diff. Bounded trust means the team agrees in advance on which file types and risk levels can accept AI suggestions without extra review. Explicit sign-off means a named human approves every merge, and that approval is recorded against a specific commit SHA.

Traceability is the easiest to implement. Git trailers work. A commit message line like `Assisted-by: copilot` or `Generated-by: cursor` is a convention, not a platform feature, and it survives rebases and merges. The harder part is enforcement. A pre-commit hook can reject commits that touch sensitive paths without a trailer, but that adds friction. A CI check that scans the diff for AI markers and fails the build when a high-risk file changed without one is usually less disruptive.

Bounded trust is a policy decision, not a technical one. A typical policy might say: AI suggestions are allowed without extra review in test files, documentation, and small utility functions; they require a second reviewer in authentication code, payment logic, and infrastructure-as-code; they are forbidden entirely in cryptographic key handling and database migration scripts. That policy lives in a `CODEOWNERS` file or a custom CI rule, not in a wiki page nobody reads.

Explicit sign-off is where most teams fail. A thumbs-up emoji on a pull request is not sign-off. Sign-off is a commit status check that names the reviewer, the timestamp, and the commit SHA. Branch protection can require this. The key property is that the approval is tied to the exact code being merged, not to a conversation that happened three days earlier and may not apply to the current diff.

Put together, the pipeline looks like this: developer prompts the assistant, the assistant generates a diff, the developer reviews and commits with a trailer, CI runs tests and policy checks, a human reviewer approves, branch protection enforces the approval, merge happens. Every step is logged. Every step is reversible. The AI is a tool inside the process, not a replacement for it.

## Implementing traceability with hooks and CI

Start with a pre-commit hook that flags AI-generated code in sensitive paths. The following Python script uses `pre-commit` and `gitpython`. It checks the staged diff for a trailer and compares the file path against a policy list.

```python
# .pre-commit-hooks/check_ai_trailer.py
import re
import sys
from git import Repo

SENSITIVE_PATHS = [
    r"^src/auth/",
    r"^src/payments/",
    r"^infra/terraform/",
]

AI_TRAILER = re.compile(r"^Assisted-by:\s*\S+", re.MULTILINE)

def main():
    repo = Repo(".")
    staged = repo.index.diff("HEAD")
    for diff in staged:
        path = diff.b_path or diff.a_path
        if any(re.match(p, path) for p in SENSITIVE_PATHS):
            blob = repo.git.show(f":{path}")
            if not AI_TRAILER.search(blob):
                print(f"Missing AI trailer in sensitive file: {path}")
                sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    main()
```

Note the limitation: this checks the file contents for the trailer string, not the commit message. If the trailer is intended to live in the commit message, the hook should inspect `repo.head.commit.message` instead of the blob. Either convention works as long as the team picks one and the CI check reads the same location.

Next, add a CI policy check. The following GitHub Actions step uses `actions/checkout` and a small Node script. It reads a `policy.json` file and enforces review requirements based on file paths and AI markers.

```javascript
// scripts/enforce-ai-policy.js
const fs = require('fs');
const { execSync } = require('child_process');

const policy = JSON.parse(fs.readFileSync('policy.json', 'utf8'));
const changedFiles = execSync('git diff --name-only HEAD~1 HEAD')
  .toString()
  .trim()
  .split('\n');

let requiresSecondReview = false;
for (const file of changedFiles) {
  for (const rule of policy.rules) {
    if (new RegExp(rule.path).test(file) && rule.requiresSecondReview) {
      requiresSecondReview = true;
    }
  }
}

if (requiresSecondReview) {
  const approvals = execSync('gh pr view --json reviews --jq ".reviews | length"')
    .toString()
    .trim();
  if (parseInt(approvals, 10) < 2) {
    console.error('This PR requires two approvals per policy.');
    process.exit(1);
  }
}
```

Two caveats worth stating plainly. First, `git diff --name-only HEAD~1 HEAD` only inspects the last commit; for a multi-commit PR, use the merge-base form, e.g. `git diff --name-only $(git merge-base HEAD origin/main) HEAD`. Second, counting reviews via the GitHub API counts review submissions, not distinct approving reviewers, so a single reviewer who submits twice can satisfy a threshold of two. If distinct approvals matter, deduplicate by reviewer login before comparing.

Finally, enforce sign-off with branch protection. In GitHub, enable "Require a pull request before merging" and "Require approvals" with a minimum of one. For high-risk repositories, set it to two. Enable "Require review from Code Owners" so that changes to critical paths require the right reviewer. This is a configuration change, not code.

## Measuring whether the policy works

Any claim about the effect of a review policy has to come from measurement on the team's own repository. The instrumentation is straightforward.

To measure review latency, record the timestamp of the first commit on a PR and the timestamp of the merge. Both are available from the Git history and the PR API. Compare the distribution before and after the policy change, not just the mean.

To measure escaped defects, tag incidents with the commit SHA that introduced the fault. That requires post-incident discipline, but it is the only way to attribute a production failure to a specific change. Without it, defect-rate comparisons are guesswork.

To measure AI-assisted commit share, count commits whose message contains the trailer. This is exact if the trailer is enforced and approximate if it is voluntary.

To measure reviewer load, count distinct reviewers per PR and the number of PRs each reviewer touches per day. A policy that doubles approvals but leaves the same three people approving everything has not distributed ownership; it has concentrated it.

A worked example, using illustrative numbers only. Suppose a team merges 40 PRs per week, and the policy adds one extra reviewer to 25% of them. That is 10 extra review events per week. If each takes 15 minutes of focused reading, the added cost is 2.5 person-hours per week. If the policy prevents one production incident per quarter, and each incident costs 4 person-hours of response plus downstream impact, the break-even is one prevented incident every five weeks. The arithmetic is simple; the inputs are the hard part, and the only honest source for them is the team's own incident log.

## Failure modes worth designing against

**The silent behavior change.** An assistant asked to "fix the retry logic" may replace exponential backoff with a fixed sleep. Tests pass because they mock the sleep. Production fails because the downstream API rate-limits after a burst. A useful control is to require that any change to retry, timeout, or backoff parameters carries a comment explaining the rationale and a reference to the downstream API's documented limits.

**The hallucinated import.** Generated code sometimes references packages or import paths that do not exist. The error surfaces at runtime as `ModuleNotFoundError`. A dependency check in CI that verifies every import resolves to an installed package catches this before merge. Most language ecosystems have a tool for this; the important part is running it in the pipeline rather than on developer machines.

**The security blind spot.** Models trained on public code reproduce public code's patterns, including insecure ones. A generated SQL query may use string concatenation instead of parameterized queries. A generated Terraform block may open a security group to `0.0.0.0/0`. Static analysis can catch some of these, but only if it is configured to fail the build on high-severity findings. Report-only mode is a common way to run a scanner that nobody ever reads.

**The ownership vacuum.** When an assistant generates a function and a developer merges it without understanding it, the developer is still the owner. If it breaks at 03:00, they are on the hook. This is a cultural failure, not a technical one. Making it explicit in the pull request template — a checkbox that says "I have read and understand every line of this change" — is a cheap forcing function. If the box cannot be checked honestly, the change is not ready.

## A decision checklist before adding policy

Not every team needs the full workflow. Use the following checklist to decide how much to add.

- Is the codebase handling money, personal data, or shared infrastructure? If no, light-touch traceability is probably enough.
- Are there at least two people who can meaningfully review the sensitive paths? If not, the bottleneck is staffing, not process.
- Does the team already track escaped defects by commit? If not, no policy change can be evaluated.
- Is the current policy narrow enough that it will not be routed around? Start with authentication, payments, and infrastructure, and expand only with data.
- Can the CI check run in a few seconds on a typical diff? If not, it will be disabled.

If the answer to the first question is no and the answer to the second is no, the right next step is test coverage and fast rollbacks, not review ceremony. If the answer to the first is yes, the workflow above is the minimum viable version.

## Frequently asked questions

**How should AI-generated code be marked in Git?**
Use a Git trailer in the commit message, such as `Assisted-by: <tool>`. This is a convention, not a standard, but it is widely understood and it survives rebases and merges. Enforce it in sensitive paths with a hook or a CI check that reads the same location the trailer is written to.

**Does the choice of AI assistant matter for review burden?**
Somewhat, but less than the policy around it. Assistants differ in context window, multi-file editing, and how idiomatic their output is, and those differences shift review time. The policy determines whether risky changes get a second reader regardless of which tool produced them. Pick a tool, measure the defect rate and review time, and adjust the policy rather than chasing tool swaps.

**How can AI-introduced security vulnerabilities be caught before merge?**
Run static analysis in CI and fail the build on high-severity findings. Require a second reviewer for changes to authentication, authorization, and data handling code. Because models reproduce patterns from public code, human review of security-relevant diffs remains necessary even when scanners are clean.

**Which metrics actually indicate the policy is working?**
Four are enough to start: AI-assisted commit count, escaped defects per thousand lines, median review time, and time to first review. If the defect rate falls and review time stays roughly flat, the policy is earning its cost. If review time doubles without a defect-rate change, the policy is too heavy for the risk it addresses.

## One action to take in the next 30 minutes

Open the repository's commit template — `.gitmessage` in the repo root, or the path configured via `git config commit.template` — and add a single line: `Assisted-by: <tool-name>`. If no template exists, create one and point Git at it with `git config commit.template .gitmessage`. Then make one commit and confirm the trailer appears in `git log -1 --format=%B`. That is the smallest possible step toward traceability, and it costs nothing. The hook and the CI policy can wait until the trailer is a habit.
