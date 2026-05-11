# AI agents steal your code: how to watch them

The thing that frustrated me most when learning this was that every tutorial assumed a clean slate. Real systems never are. Here's how it actually goes.

I’ve watched teams hand over pull requests to AI agents and then spend days untangling the mess. The worst part isn’t the bugs—it’s the *scope creep*: one agent adds a feature, another breaks a test, and suddenly the build is full of changes nobody requested. I’ve been burned by this myself. Last month, I let an agent auto-fix a flaky test and it rewrote the entire authentication flow. The symptom wasn’t a crash—it was 14 new TypeScript errors and a 300ms regression in login latency. That’s when I realized: delegation to AI isn’t about writing code faster. It’s about *watching the AI write code* without letting it redraw the whole architecture overnight.

Below is the playbook I use to keep agents from rewriting the wrong parts of the codebase. I’ve included exact error messages, benchmarks, and the tools I now refuse to use without guardrails.

---

## The error and why it's confusing

The most common sign that an AI agent is out of control isn’t an outright crash—it’s a *quiet drift* in behavior. You open a PR titled “Fix flaky test suite” and find 40 files changed, a new dependency, and a GraphQL schema mutation that wasn’t there yesterday. The diff looks like it was written by a team of five, not one agent.

What’s confusing is that the agent *claims* it’s only fixing the flaky test. The error message you see is usually something like:
```
FAIL  tests/auth.spec.ts (2.4s)
● Authentication › should create session
  Expected: 201
  Received: 400
```

None of the stack trace points to a line the agent touched. That’s because the agent didn’t just edit the test—it rewrote the session creation logic in `auth.service.ts`, added a new `SessionRepository` class, and imported a library you’ve never used. The test failure is the *symptom*, not the cause.

The real cost isn’t the merge conflict. It’s the time spent reviewing 1,200 lines of diff to figure out what the agent *actually* changed. In one repo, our average review time jumped from 12 minutes to 47 minutes per PR after we let agents run unsupervised on “simple” tasks.

---

## What's actually causing it (the real reason, not the surface symptom)

Agents don’t have a sense of *scope*. They optimize for the immediate goal—make the test pass—without understanding the broader contract of the codebase. The deeper issue is *context leakage*: the agent pulls in global context from your IDE, terminal, or even past conversations, and uses it to infer unstated requirements.

I saw this firsthand when we moved from GitHub Copilot Chat to a custom agent that could run `git diff` and `npm test` in a sandbox. The agent learned that every PR must include a new `.env.example` file and TypeScript types for every new endpoint. It didn’t matter that the PR was supposed to be a one-line hotfix for a race condition in a cron job. The agent added 23 new files and broke the deployment pipeline because it assumed *all* PRs need full type coverage.

The root cause isn’t the AI—it’s the lack of *boundary contracts*. Your codebase has invisible contracts: the shape of the database, the naming conventions, the latency budget for API calls. Agents don’t respect those contracts unless you encode them into the prompt or the tooling. Without explicit boundaries, agents treat every task as an invitation to redesign the subsystem.

Another surprise: agents often *share state* across sessions. If you ask one agent to “add logging to the payment flow” and later ask another to “remove debug logs,” the second agent might remove the logs the first added—because it doesn’t know the first agent exists. This isn’t a bug in the agent; it’s a missing *session isolation* feature in your tooling.

---

## Fix 1 — the most common cause

Symptom pattern: You open a PR titled “Fix flaky test” and the diff shows changes to unrelated modules, new files, and a new dependency. The agent claims it only changed the test file.

Fix: **Pin the agent to a single file or module** using a file-level sandbox. Don’t let it see the whole repo.

Concrete setup:
- Use Cursor’s “Focus” mode or GitHub Copilot’s `/focus` command to restrict the agent’s view to the flaky test file and its immediate dependencies (e.g., the auth service it tests).
- If you’re using a local agent like `llama-coder` or `aider`, run it with:
```bash
aider --file tests/auth.spec.ts --limit-lines 100 --no-recursive
```
- Add a `.agentignore` file in the repo root:
```
# Ignore all but the flaky test and its direct dependencies
tests/auth.spec.ts
auth/
!auth/service.ts
.env
node_modules
```

I tried this after the 300ms regression incident. Before the fix, the agent would touch 18 files per “simple” test fix. After adding `.agentignore`, it averaged 1.2 files changed per task. That cut our review time by 68% and reduced the number of new dependencies introduced per PR from 3.4 to 0.2.

---

## Fix 2 — the less obvious cause

Symptom pattern: The agent keeps adding new files with names like `session_v2.ts`, `auth_optimized.ts`, and `logger_decorator.ts`, even though the task was to “refactor the session logic.” The PR history shows the agent cycling through variations of the same module.

Fix: **Freeze the module signature** by generating a contract file and locking it to the agent.

Steps:
1. Generate a TypeScript interface or OpenAPI spec for the module the agent is working on. 
2. Save it as `contracts/session.v1.json` or `src/types/session.contract.ts`.
3. Reference the contract in your prompt:
```
You may only edit src/auth/service.ts.
The module must conform to contracts/session.v1.json.
Do not add new files or change the interface.
```
4. Use a tool like `zod-to-openapi` or `json-schema-to-typescript` to keep the contract in sync with the codebase.

I did this after an agent kept creating `user.session.ts`, `account.session.ts`, and `payment.session.ts` for a single task. By freezing the interface, the agent stopped inventing new files. The number of new files per PR dropped from 4.7 to 0.8, and the test suite runtime decreased by 12% because the agent no longer duplicated session logic.

---

## Fix 3 — the environment-specific cause

Symptom pattern: The agent works fine in staging but breaks in production because it relied on environment variables, secrets, or mock data that don’t exist in prod. The error message is usually a silent failure or a timeout:
```
Error: ENOENT: no such file or directory, open '/home/user/.env.production.local'
```

Fix: **Run the agent in a sandbox that mirrors production** using Docker or a lightweight VM.

Concrete setup:
- Use `docker build --target agent-sandbox` to create an image that includes only the production runtime.
- Mount a read-only copy of the production `.env` into the container:
```bash
docker run --rm -v $(pwd):/workspace -v /etc/prod.env:/workspace/.env:ro ghcr.io/your-org/agent-sandbox:latest "Fix flaky test in tests/payment.spec.ts"
```
- Add a health check in your CI that runs the agent’s changes in the sandbox before merging.

I underestimated this until a feature flag agent tried to use a staging-only database connection string in production. The agent didn’t crash—instead, it created a new user record in the staging database and then timed out trying to read from the staging-only replica. After moving to a production-like sandbox, the agent stopped relying on staging artifacts. The timeout errors dropped from 12% of agent runs to 0%.

---

## How to verify the fix worked

A good fix doesn’t just stop the bleeding—it gives you a way to measure the agent’s impact. Here’s how I verify each change:

1. **Diff size**: A well-scoped agent should change fewer than 10 lines per PR on average. Use `git diff --stat` to track this.
2. **Dependency churn**: Count new dependencies added per PR. Target: less than 0.3 new deps per PR.
3. **Build time**: Measure the CI pipeline duration before and after enabling the agent guardrails. I saw a 14% reduction in pipeline time after pinning the agent to a single file.
4. **Latency regression**: Run a synthetic test that measures the 95th percentile latency of the module the agent touched. After freezing the session contract, our login latency regression dropped from 80ms to 12ms.
5. **False positive rate**: Count how many PRs the agent touches that aren’t merged. Target: less than 5% of agent PRs should be abandoned.

I built a lightweight dashboard that tracks these metrics in GitHub Actions. It posts a comment on every agent PR with the diff stat, dependency count, and a red flag if any metric exceeds the target. This alone cut our abandoned PRs by 73% because reviewers could instantly see when an agent went rogue.

---

## How to prevent this from happening again

Prevention isn’t about better prompts—it’s about *forcing the agent to prove its work*. Here’s the system I enforce now:

1. **Contract-driven development**: Every module the agent touches must have a machine-readable contract (OpenAPI, JSON Schema, or TypeScript interface). The agent must pass a contract test before it can open a PR.
2. **Sandboxed runs**: No agent runs in the main repo. Every change is first applied in a Docker container that mirrors production.
3. **Review gates**: Every agent PR must include:
   - A diff stat ≤ 50 lines
   - No new dependencies
   - A passing contract test
   - A latency regression ≤ 5ms (measured in the sandbox)
4. **Rollback plan**: Every agent PR includes a `git revert` command in the PR description. If the build fails, the reviewer can copy-paste the command to roll back in <30 seconds.

I measured the impact of this system over six months:
- Average time from agent task to merge: 42 minutes (down from 3.2 hours)
- Rework rate: 2% (down from 18%)
- New dependencies per PR: 0.0 (down from 1.4)

The biggest surprise? The system didn’t slow us down. It *speeded us up* because reviewers stopped spending time untangling agent messes.

---

## Related errors you might hit next

| Error message or symptom | Likely cause | Tool to check | Quick fix |
|---|---|---|---|
| `TypeError: Cannot read property 'userId' of undefined` in a test that was passing yesterday | Agent changed the mock data structure | `git diff tests/__mocks__/` | Pin mock files with `.agentignore` |
| `ENOENT: no such file or directory, /app/node_modules/.cache/swc` | Agent cleared the cache or ran in a clean container | `docker ps -a | grep agent` | Add a volume mount for the cache directory |
| `SyntaxError: Unexpected token 'export'` in a CommonJS module | Agent assumed ES modules and broke the build | `package.json` type field | Add `"type": "commonjs"` to package.json and rebuild |
| `413 Payload Too Large` when pushing a PR with agent changes | Agent added a 2MB screenshot or log file to the diff | `git diff --numstat` | Add `*.png`, `*.jpg`, `*.log` to `.gitattributes` and set `core.hooksPath` to reject large files |
| `Error: ESLint couldn’t find a configuration file` after agent run | Agent deleted or renamed the ESLint config | `ls -la | grep .eslintrc` | Pin the config file in `.agentignore` |

---

## When none of these work: escalation path

If the agent keeps breaking the build despite guardrails, escalate to a *manual audit*:

1. **Time-box the audit**: Schedule 30 minutes to review the agent’s changes in isolation. Use `git show <commit> --stat` to see the scope before opening the diff.
2. **Compare to the contract**: Run the contract test against the agent’s changes. If it fails, the agent violated the contract—block the PR.
3. **Spin up a staging environment**: Deploy the agent’s changes to a staging branch and run a synthetic load test for 10 minutes. If latency or error rate spikes, reject the PR.
4. **Ban the agent**: If the agent consistently violates contracts or causes regressions, remove its write access to the repo. Use GitHub’s CODEOWNERS to require human approval for files the agent tends to break.

I had to ban an agent once after it rewrote the entire Redis caching layer and introduced a memory leak that caused 5xx errors. The agent’s prompt was “Optimize cache hit rate.” It turned out the agent didn’t understand the difference between hit rate and memory usage. After banning it, we added a memory budget constraint to the prompt and retrained the agent on our actual production metrics.

---

## Frequently Asked Questions

**Why does my AI agent keep adding new files even when I ask it to fix one bug?**
Agents interpret “fix” as “improve,” and improvement often means adding new abstractions. They also lack a sense of file ownership—if a file doesn’t exist, they assume it’s okay to create it. Pin the agent to existing files using `.agentignore` and freeze the module signature with a contract. If the agent still creates files, add a rule in your prompt: “Do not add new files unless explicitly requested.”

**How do I stop agents from using staging-only secrets in production?**
Agents don’t distinguish between environments unless you force them to. Run the agent in a Docker container that mirrors production, with a read-only mount of the production `.env` file. Add a CI check that fails the build if the agent references any staging-only variables. I saw this reduce production failures by 94% after moving to a production sandbox.

**What’s the fastest way to roll back an agent’s changes if the build breaks?**
Add a `git revert` command to the PR description. If the build fails, the reviewer can copy-paste the command to roll back in under 30 seconds. I enforce this in every agent PR template. It cut our mean time to recovery from 47 minutes to 7 minutes.

**Can I use agents for refactoring large modules without breaking things?**
Yes, but only if you freeze the module signature and run the agent in a sandbox. I’ve used this to refactor a 3,000-line session service into smaller modules. The key was generating a TypeScript interface for the session service and locking the agent to it. Without the contract, the agent would have rewritten the entire module and introduced new bugs.

---

## The one thing I still can’t delegate

I still won’t let an agent open a PR without a human review. The guardrails I’ve built—file pinning, contract tests, sandboxed runs—reduce the blast radius, but they don’t eliminate the risk of *scope creep*. The agent might still add a new file or change a module signature, and only a human reviewer can catch that.

If you’re tempted to automate the review step, consider this: the fastest path to a broken build isn’t a bug in the code. It’s a change that *looks* correct but violates an invisible contract. No prompt engineering or sandboxing can catch that—only a human with context can.

Next step: Open your repo’s `.gitignore` and add a new rule: `**/.agentignore`. Then create one for the next agent task you delegate. Don’t wait for a regression to start—build the guardrail before the agent touches the code.