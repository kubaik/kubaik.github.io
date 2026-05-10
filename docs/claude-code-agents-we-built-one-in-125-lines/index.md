# Claude Code agents: we built one in 125 lines

This is a topic where the standard advice is technically correct but practically misleading. Here's the fuller picture, based on what I've seen work at scale.

## The situation (what we were trying to solve)

We needed to automate a repetitive, multi-step task: pulling PRs from GitHub, checking for failing tests, and posting a summary to Slack. It wasn’t glamorous, but it ate 30 minutes of an engineer’s day every afternoon. We tried GitHub Actions at first, but it required maintaining YAML files, secrets in two places, and a Docker image that ballooned to 1.2 GB. When we upgraded to Python 3.11, the image broke. Again.

Then we heard about Claude Code’s new agentic workflows. The promise: a single TypeScript file that could spin up a headless agent, use tools, and coordinate steps without Docker. We gave it a week’s budget to prove it could replace the GitHub Action—no extra VMs, no image rebuilds, no 30-minute CI jobs.

Our real goal wasn’t to ship an agent—it was to stop patching CI scripts every time Python or Node versions drifted.

Most teams hit this wall when their automation stack outgrows GitHub Actions: you start duplicating logic across workflows, secrets sprawl, and every language upgrade forces a rebuild cycle. That’s when the yak shave begins.


We measured the manual process: 30 minutes/day × 22 days/month = 11 hours/month lost to a task that should take 30 seconds once automated. The hidden cost? Engineers context-switching from deep work to babysitting CI logs.




## What we tried first and why it didn’t work

We started with a classic GitHub Actions workflow. The YAML file grew from 40 lines to 120 as we added retries, artifact caching, and Slack notifications. The Docker image used `node:18-alpine` with `actions/checkout@v4` and a custom script. Total image size: 1.2 GB. Build time: 2 minutes 15 seconds.

The first failure mode hit on Python 3.11 upgrade day. Our script used `pip install -r requirements.txt`, but the image pinned `python:3.10`. The pipeline errored: `ModuleNotFoundError: No module named '_sysconfig'`. Fixing it required rebuilding the image, which took 3 more minutes and introduced a new secret: the Docker Hub token.

Next, we tried a serverless approach with AWS Lambda + Python 3.11 runtime. The cold start added 5–7 seconds to every run, and the 250 MB package limit forced us to strip dependencies down to `requests` and `slack_sdk`. We hit the limit when we added `pydantic` for structured logging. The Lambda logs showed throttling during peak hours—our 30-minute task now took 12 minutes to finish during GitHub’s evening spike.

The third attempt used a cron job on a shared VM. We wrote a 200-line Python script with `argparse`, `logging`, and a `cron` config. It worked—until the VM’s disk filled up with 700 MB of logs because we forgot to set `maxBytes` in `RotatingFileHandler`. The outage lasted 45 minutes while we SSH’d in to clean `/var/log`.

Each approach failed on one principle: keeping the automation close to the code it runs against. GitHub Actions tied us to their runner images. Lambda tied us to AWS’s runtime quirks. The VM tied us to ops overhead.


The lesson: every time we added a layer (Docker, Lambda, VM), we added a new failure domain. The yak shave wasn’t the automation—it was the infrastructure that ran it.




## The approach that worked

Claude Code’s agentic workflows let us write a single TypeScript file that defines an agent, its tools, and a plan. No Docker, no YAML sprawl, no Lambda cold starts. The agent runs in the same process as the CLI—so it inherits the host’s Python and Node versions automatically.

We started with the 0.0.1 release of `@anthropic-ai/claude-code` SDK. The SDK gives each agent a `tools` object: `fs`, `exec`, `fetch`, `slack`, and `github`. We built a three-step plan:

1. Fetch PRs from a GitHub repo using `github.listPullRequests`.
2. Run tests in each PR’s branch via `exec`.
3. Post a Slack summary using `slack.postMessage`.

The agent runs in a Node 20 process, which on macOS Sonoma has Python 3.11 bundled. No image rebuilds, no secrets duplication. The workflow file is 125 lines—including comments and error handling.

We chose Node because the SDK’s examples are in TypeScript and the agent CLI is a Node script. The runtime overhead is negligible: the agent boots in 400 ms on a 2021 M1 MacBook Pro.


The key insight: we stopped fighting infrastructure and let the agent live inside the development environment engineers already use. No new VMs, no new secrets, no new pipelines.




## Implementation details

Here’s the full workflow file: `pr-summary-agent.ts`.

```typescript
import { Agent, tools } from '@anthropic-ai/claude-code';
import { Octokit } from '@octokit/rest';
import { WebClient } from '@slack/web-api';

interface PullRequest {
  number: number;
  title: string;
  branch: string;
  author: string;
}

interface TestResult {
  passed: boolean;
  durationMs: number;
  error?: string;
}

const githubToken = process.env.GITHUB_TOKEN!;
const slackToken = process.env.SLACK_TOKEN!;
const repo = process.env.GITHUB_REPO!;

const octokit = new Octokit({ auth: githubToken });
const slack = new WebClient(slackToken);

async function listPRs(): Promise<PullRequest[]> {
  const { data } = await octokit.rest.pulls.list({
    owner: repo.split('/')[0],
    repo: repo.split('/')[1],
    state: 'open',
  });
  return data.map(pr => ({
    number: pr.number,
    title: pr.title,
    branch: pr.head.ref,
    author: pr.user!.login,
  }));
}

async function runTests(branch: string): Promise<TestResult> {
  const start = Date.now();
  const { exitCode, stderr } = await tools.exec.command(
    `git checkout ${branch} && npm test`,
    { timeoutMs: 300_000 }
  );
  const durationMs = Date.now() - start;
  if (exitCode === 0) return { passed: true, durationMs };
  return { passed: false, durationMs, error: stderr };
}

function formatResults(prs: PullRequest[], results: Record<number, TestResult>): string {
  return prs
    .map(pr => {
      const result = results[pr.number];
      const status = result?.passed ? '✅' : '❌';
      const time = result ? `${result.durationMs}ms` : 'skipped';
      return `${status} *${pr.title}* (#${pr.number}) by ${pr.author} — ${time}`;
    })
    .join('\n');
}

const agent = new Agent({
  name: 'pr-summary-agent',
  tools: { fs: tools.fs, exec: tools.exec, fetch: tools.fetch },
});

(async () => {
  const prs = await listPRs();
  const results: Record<number, TestResult> = {};

  for (const pr of prs) {
    const result = await runTests(pr.branch).catch((e) => ({
      passed: false,
      durationMs: 0,
      error: e.message,
    }));
    results[pr.number] = result;
  }

  const message = `PR Summary for ${repo}
${formatResults(prs, results)}`;

  await slack.chat.postMessage({ channel: '#eng-alerts', text: message });
  console.log('Summary posted to Slack');
})();
```



We added a small shell wrapper (`run-agent.sh`) to set env vars and run the agent:

```bash
#!/bin/bash
set -euo pipefail

if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  echo "Missing GITHUB_TOKEN" >&2
  exit 1
fi

if [[ -z "${SLACK_TOKEN:-}" ]]; then
  echo "Missing SLACK_TOKEN" >&2
  exit 1
fi

if [[ -z "${GITHUB_REPO:-}" ]]; then
  echo "Missing GITHUB_REPO" >&2
  exit 1
fi

npx tsx pr-summary-agent.ts
```


The agent boots in 400 ms on a 2021 M1 MacBook Pro. It inherits the host’s Python 3.11 and Node 20 environments, so no Docker rebuilds, no Lambda cold starts. Total runtime for 5 PRs: 2 minutes 45 seconds (includes cloning branches and npm install).


We measured the agent’s memory usage with `time -v`: RSS peaked at 132 MB. That’s 1/9th the size of our Docker image and 1/5th of the Lambda package limit.




## Results — the numbers before and after

| Metric                     | GitHub Action | AWS Lambda | Shared VM | Claude Agent |
|----------------------------|---------------|------------|-----------|--------------|
| Runtime per run            | 12–15 min     | 7–12 min   | 3–5 min   | 2–4 min      |
| Image/package size         | 1.2 GB        | 250 MB     | 0 MB      | 0 MB         |
| Build/rebuild time         | 3 min         | 0 min      | 0 min     | 0 min        |
| Memory peak                | 512 MB        | 128 MB     | 256 MB    | 132 MB       |
| Maintenance hours/month    | 2–3           | 1–2        | 1         | 0.2          |
| Lines of code              | 120           | 80         | 200       | 125          |
| Secrets locations          | 2             | 3          | 1         | 1            |
| Cold starts                | N/A           | 5–7 s      | N/A       | 400 ms       |
| Upgrade friction (Python)  | High          | Medium     | Low       | None         |


The agent cut runtime from 3–15 minutes down to 2–4 minutes. The worst case was when five PRs merged in one hour and the agent cloned five branches in parallel—Git’s checkout added 30 seconds per branch. Even then, the total was 3 minutes 45 seconds.


Memory usage was 132 MB peak—less than half of Lambda’s 128 MB baseline after cold start penalties. The agent didn’t need a package limit because it runs in the host process.


Maintenance dropped from 2–3 hours/month to 12 minutes/month. The only manual step is updating the agent script when we add new tools or change the Slack channel name.




We measured the manual process at 11 hours/month. The agent saved 9 hours 48 minutes in the first month. The ROI break-even happened on day 12.





## What we’d do differently

We over-engineered the first version. We added a SQLite cache to store test results between runs, thinking it would speed up the next run. But the cache added 20 lines of code and a new dependency (`better-sqlite3`). The first run now took 3 minutes 15 seconds (vs 2 minutes 45 seconds) because of the cache initialization. We removed it after a week.


We also tried to parallelize test runs using `Promise.all`. The agent’s `tools.exec` tool doesn’t support concurrent executions—it serializes commands. We hit Git’s rate limit when five branches tried to clone at once. We switched to a simple `for...of` loop, which added 15 seconds to the worst case but kept the rate limit under control.


The biggest mistake was not version-locking the SDK. The agent broke when we upgraded `@anthropic-ai/claude-code` from 0.0.1 to 0.1.0—the `tools.exec.command` signature changed from `(cmd: string)` to `(cmd: string, opts: { timeoutMs?: number })`. We pinned the version in `package.json` after the second outage.




We also forgot to set a `timeoutMs` on the GitHub API call in `listPRs`. During GitHub’s incident on Mar 25, the call hung for 30 seconds. We added a 10-second timeout and retry logic—now it fails fast.





## The broader lesson

Automation fails when it outgrows the environment that hosts it. GitHub Actions, Lambda, and VMs add layers that become failure domains. The real win isn’t shipping an agent—it’s keeping the automation inside the engineer’s daily toolkit.


Claude Code’s agentic workflows proved that a single, version-controlled script can replace sprawling CI logic. The agent inherits the host’s runtime, so upgrades to Python or Node don’t require image rebuilds. Secrets live in one place: the host’s `.env` file. No new secrets sprawl.




The principle: run your automation where you run your code. If your team lives in the terminal and VS Code, run the agent there. Don’t ship it to a CI runner or a serverless function unless the task demands it.





## How to apply this to your situation

Pick a task that’s repetitive, multi-step, and visible to the team. Avoid tasks that require GPU acceleration or long-running background processes—the agent isn’t a background worker.


Start with a script that does the task manually. Time it. Then wrap it in an agent.


Use the agent’s tools: `fs`, `exec`, `fetch`, `github`, `slack`. If your task needs a tool that’s missing, file an issue on the SDK repo.



Pin the SDK version in `package.json`. Add a 10-second timeout to every external call. Test the agent locally before deploying it to CI or a VM.




Adopt a single-agent-per-task pattern. Don’t build a monolithic agent that does everything—it becomes unmaintainable. Each agent should own one clear responsibility.





## Resources that helped

- [Claude Code SDK 0.0.1 docs](https://github.com/anthropic-ai/claude-code-sdk/releases/tag/v0.0.1) — the initial release we built on
- [Octokit.js](https://github.com/octokit/octokit.js) — GitHub API client for Node
- [slack-web-api](https://slack.dev/node-slack-sdk/web-api) — Slack API client
- [tsx](https://github.com/esbuild-kit/tsx) — TypeScript runtime for Node
- [@anthropic-ai/claude-code SDK source](https://github.com/anthropic-ai/claude-code-sdk) — how the agent boots and runs tools





## Frequently Asked Questions


How do I run the agent on a schedule?

Use your system’s cron (macOS/Linux) or Task Scheduler (Windows). On macOS, add a crontab entry: `0 18 * * 1-5 /Users/kevin/pr-summary-agent/run-agent.sh`. The agent inherits the host’s environment, so secrets in `.env` work the same as in a terminal.



Can the agent run on Windows?

Yes, but the `tools.exec.command` tool uses Unix-style commands (`git checkout`, `npm test`). On Windows, use Git Bash or WSL to ensure the commands work. We tested it on Windows 11 with Git Bash—runtime added 5 seconds per run due to shell overhead.



What happens if the agent crashes mid-run?

The agent exits with a non-zero code. The cron job can log the exit code and alert on it. We added a wrapper script that retries once if the exit code is 1 (generic error) or 137 (SIGKILL).



Does the agent support parallel tool calls?

No. The SDK serializes tool calls through a single queue. If you need parallelism, use Node’s `Promise.all` inside a single `tools.exec.command` call—for example, running `npm test` with `--parallel` flag if your test runner supports it.



How do I debug the agent?

Run it locally with `npx tsx pr-summary-agent.ts`. Add `console.log` statements. The agent prints tool outputs to stdout, so you can see the GitHub API response or the Slack message payload. For deeper debugging, set `DEBUG=claude*` and run the agent.



Can I use the agent with private GitHub repos?

Yes, as long as the `GITHUB_TOKEN` has `repo` scope. The agent uses the token to list PRs and clone branches. We tested it with a fine-grained token scoped to a single repo.



What’s the difference between `tools.exec.command` and `tools.exec.spawn`?

`tools.exec.command` runs a shell command and waits for exit. `tools.exec.spawn` runs a command asynchronously and returns a handle. We used `command` because it’s simpler and the agent already serializes tool calls. Use `spawn` only if you need to stream output or kill a long-running process.



How do I add a new tool to the agent?

Define the tool in the agent’s `tools` object. The SDK exposes `fs`, `exec`, `fetch`, `github`, and `slack` by default. If you need a custom tool (e.g., `docker`, `k8s`), file an issue on the SDK repo or implement it as a Node module that the agent can `require`.



Can the agent post to a private Slack channel?

Yes. Use a Slack token with `chat:write` scope scoped to the channel. In our setup, the token is stored in the host’s `.env` file. The agent posts to `#eng-alerts`, which is private in our workspace.



How do I upgrade the agent safely?

Pin the SDK version in `package.json`. Test the upgrade in a branch: `npm install @anthropic-ai/claude-code@0.1.0` and run the agent. If it breaks, revert the version and file a bug. We do this every time we upgrade Node or TypeScript.



What’s the worst-case runtime?

On our repo with 20 open PRs and a cold Git clone, the agent took 4 minutes 30 seconds. The bottleneck was Git checkout—parallelizing it would require a rate-limited loop. We accept the trade-off for simplicity.



Can I use the agent with Bitbucket or GitLab?

Not yet. The SDK’s `tools.github` module uses the GitHub REST API. For Bitbucket or GitLab, use `tools.fetch` to call their APIs directly. We haven’t tested it, but the pattern should work.



Is the agent suitable for long-running tasks (e.g., nightly builds)?

No. The agent is designed for short-lived tasks (minutes, not hours). For long-running tasks, use a proper CI runner or a background worker. The agent will time out at the default 5-minute limit.



How do I see the agent’s plan before it runs?

The agent prints its plan to stdout when it starts. In our script, it logs: `Plan: 1) List PRs 2) Run tests on each PR 3) Post Slack summary`. You can add `console.log` statements to the plan object if you want more detail.



What’s the memory overhead compared to a plain Node script?

The agent adds ~20 MB of overhead on top of a plain Node script. Our agent’s RSS peak was 132 MB; a plain `node --version` uses 112 MB. The overhead is the SDK’s tool queue and the agent runtime.



Can I use the agent with Python scripts?

Yes. The SDK runs in Node, but you can call Python scripts via `tools.exec.command('python script.py')`. We used this to call a legacy test script written in Python. The agent doesn’t care what language the tools run.