# Claude Code after a year: daily wins

Agentic coding assistants are most useful when they are scoped to a narrow, verifiable task. "Bump this dependency and prove the tests still pass" is exactly that kind of task: the success condition is mechanical, the blast radius is small, and a failing run costs nothing if the agent rolls back cleanly. This article walks through the design of a test-gated dependency patcher, the failure modes that show up in practice, and how to instrument it so you can tell whether it is actually working.

## The problem with ungated automation

The common failure mode for dependency automation is not "the agent wrote bad code." It is "the agent opened a pull request before anything verified the change." A version bump that compiles locally can still break a container build, a lockfile check, or a test that depends on the old behavior. When the gate is missing, the cost is not the patch itself — it is the review queue, the failed CI minutes, and the on-call attention spent closing PRs nobody asked for.

The design principle that follows is simple: **the agent may propose, but only a passing test suite may publish.** Everything below is in service of that.

A second, subtler failure mode is environment drift. A patch validated on one architecture or base image can fail on another. If your CI runs on `arm64` and production runs on `x86_64`, a native module that builds fine in one place can crash in the other. The containerized test step is what catches this, provided the container image matches the target environment.

## Prerequisites

The instructions below assume a Unix-like host (macOS or a recent Ubuntu LTS), Node.js 20 LTS, and Python 3.11. You will need:

- An agentic coding CLI that can read a repository and emit a unified diff.
- Docker with BuildKit enabled, so test containers start quickly.
- A GitHub repository with at least one service that runs tests in CI.
- A GitHub token with `repo` scope (or a fine-grained token with contents and pull-request write access).

The patcher is intentionally small. Its value is in the seams: where the agent's output meets your test suite, your container runtime, and your version control.

## Step 1 — environment setup

Install the CLI using whatever distribution channel your vendor documents, then confirm the version:

```bash
claude --version
```

Create a virtual environment and install the Python dependencies:

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install docker pydantic requests
```

Install the GitHub CLI and authenticate it:

```bash
brew install gh          # macOS
sudo apt install gh -y   # Ubuntu
gh auth login
```

If you are behind a corporate proxy, export the standard variables before starting any process that needs network access:

```bash
export HTTPS_PROXY=http://proxy.example.com:8080
export HTTP_PROXY=http://proxy.example.com:8080
```

One operational note worth checking on your own machine: CLIs of this class often cache credentials in a dotfile under your home directory. Inspect the permissions on that file and, on shared hosts, either move it to an encrypted volume or restrict it to your user. Do not commit it.

## Step 2 — core implementation

Create a directory `.claude-agent` and add `patcher.py`.

```python
from pathlib import Path
import asyncio
import subprocess
import json
from typing import List

import docker
from pydantic import BaseModel

class Alert(BaseModel):
    dependency: str
    current_version: str
    new_version: str
    ecosystem: str
    manifest_path: str
    pr_url: str

class PatchResult(BaseModel):
    success: bool
    log: List[str]
    new_pr_url: str | None = None

def run_tests(service_dir: str) -> bool:
    """Run the project's test command inside a container."""
    client = docker.from_env()
    try:
        container = client.containers.run(
            image="node:20-alpine",
            command=["npm", "test"],
            volumes=[f"{service_dir}:/app"],
            working_dir="/app",
            remove=True,
            detach=True,
        )
        for chunk in container.logs(stream=True, follow=True):
            print(chunk.decode("utf-8").strip())
        return container.wait()["StatusCode"] == 0
    except Exception as e:
        print(f"Test container failed: {e}")
        return False

async def claude_plan(alert: Alert) -> PatchResult:
    """Ask the agent for a minimal patch, apply it, and gate on tests."""
    prompt = f"""
You are an expert Node.js developer. The repo at {alert.manifest_path}
has a Dependabot alert upgrading {alert.dependency} from {alert.current_version}
to {alert.new_version}.

Write a minimal patch that bumps the dependency and nothing else.
Do not edit any other files. Output the patch as a unified diff.
"""

    cmd = [
        "claude", "execute",
        "--max-turns", "3",
        "--input", prompt,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    diff = result.stdout.strip()

    diff_path = Path("patches") / f"{alert.dependency}-{alert.new_version}.patch"
    diff_path.parent.mkdir(exist_ok=True)
    diff_path.write_text(diff)

    subprocess.run(["git", "apply", str(diff_path)], check=True)
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(
        ["git", "commit", "-m",
         f"chore(deps): bump {alert.dependency} to {alert.new_version}"],
        check=True,
    )

    service_dir = str(Path(alert.manifest_path).parent)
    success = run_tests(service_dir)

    if not success:
        subprocess.run(["git", "reset", "--hard", "HEAD~1"], check=True)
        return PatchResult(success=False, log=[diff, "Tests failed."])

    branch = f"auto/dep-{alert.dependency}-{alert.new_version}"
    subprocess.run(["git", "checkout", "-b", branch], check=True)
    subprocess.run(["git", "push", "origin", branch], check=True)
    pr_url = subprocess.run(
        ["gh", "pr", "create",
         "--title", f"Bump {alert.dependency} to {alert.new_version}",
         "--body", "Auto-generated by the dependency patcher."],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return PatchResult(success=True, log=[diff], new_pr_url=pr_url)

if __name__ == "__main__":
    alert = Alert(
        dependency="express",
        current_version="4.18.2",
        new_version="4.19.0",
        ecosystem="npm",
        manifest_path="packages/api/package.json",
        pr_url="https://github.com/example/repo/issues/1",
    )
    asyncio.run(claude_plan(alert))
```

Two details in that script matter more than the rest. First, the branch is created **after** the tests pass, so a failed run never leaves a remote branch behind. Second, the rollback is `git reset --hard HEAD~1`, which discards the agent's commit entirely. If your repository uses a different default branch or a rebase workflow, adjust the reset target accordingly — `HEAD~1` assumes exactly one commit was made.

Run it once against a fixture repository to confirm the plumbing:

```bash
python .claude-agent/patcher.py
```

Expected behavior on success: a patch file under `.claude-agent/patches/`, a new local branch, a pushed branch, and a PR URL on stdout. On failure: a hard reset and a `PatchResult` with `success=False`.

## Step 3 — retries, rate limits, and circuit breaking

The naive version treats any Docker error as a test failure. In practice, transient daemon errors are common enough that a single retry is worth adding.

```python
import asyncio

MAX_RETRIES = 3
RETRY_DELAY = 2

async def run_tests_with_retry(service_dir: str) -> bool:
    client = docker.from_env()
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            container = client.containers.run(
                image="node:20-alpine",
                command=["npm", "test"],
                volumes=[f"{service_dir}:/app"],
                working_dir="/app",
                remove=True,
                detach=True,
            )
            if container.wait()["StatusCode"] == 0:
                return True
            return False
        except docker.errors.APIError as e:
            if attempt < MAX_RETRIES:
                print(f"Docker API error, retrying in {RETRY_DELAY}s "
                      f"(attempt {attempt}/{MAX_RETRIES}): {e}")
                await asyncio.sleep(RETRY_DELAY)
                continue
            raise
    return False
```

Note that only `APIError` is retried. A non-zero exit code from the test command is a real failure and should not be retried — retrying a deterministic failure just burns time.

The GitHub API has documented rate limits, and a burst of Dependabot alerts can exhaust them. Handle the 403 response explicitly:

```python
from github import Github, GithubException

g = Github(os.environ["GITHUB_TOKEN"])
repo = g.get_repo("your-org/your-repo")

try:
    pr = repo.create_pull(
        title=f"Bump {alert.dependency} to {alert.new_version}",
        body="Auto-generated patch.",
        head=branch,
        base="main",
    )
except GithubException as e:
    if e.status == 403 and "rate limit" in str(e).lower():
        await asyncio.sleep(60)
        pr = repo.create_pull(
            title=f"Bump {alert.dependency} to {alert.new_version}",
            body="Auto-generated patch.",
            head=branch,
            base="main",
        )
    else:
        raise
```

Add a circuit breaker so the agent stops calling Docker when the daemon is unresponsive rather than queueing work that will all fail:

```python
import time

class DockerCircuitBreaker:
    def __init__(self, max_failures: int = 3, reset_timeout: int = 30):
        self.max_failures = max_failures
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.last_failure = 0.0

    def allowed(self) -> bool:
        if self.failures >= self.max_failures:
            if time.time() - self.last_failure < self.reset_timeout:
                return False
            self.failures = 0
        return True

    def record_failure(self) -> None:
        self.failures += 1
        self.last_failure = time.time()
```

The failure mode this addresses is not "Docker is down" — that is obvious. It is "Docker is partially degraded": the daemon accepts connections but hangs on container start, usually because the host has run out of disk or the layer cache is corrupted. Without a breaker, every queued alert waits for the same timeout. With one, the agent fails fast and pages a human.

## Step 4 — observability and tests

Structured logs and a small metrics surface make the difference between "it seems to work" and "here is what it did."

```python
import logging
from fastapi import FastAPI
import prometheus_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("dep-patcher")

app = FastAPI()

PATCHES_TOTAL = prometheus_client.Counter(
    "patches_total", "Total patch operations", ["success"],
)
PATCH_LATENCY = prometheus_client.Histogram(
    "patch_latency_seconds", "Latency of a patch operation",
)

@app.post("/patch")
async def patch_alert(alert: Alert):
    with PATCH_LATENCY.time():
        result = await claude_plan(alert)
    PATCHES_TOTAL.labels(success="true" if result.success else "false").inc()
    logger.info(
        "patch_result dependency=%s success=%s",
        alert.dependency, result.success,
    )
    return result

@app.get("/health")
def health():
    return {"status": "ok"}
```

Avoid reading internal histogram attributes to compute durations — read the value from the observed timer or log the wall-clock delta yourself. The `_metrics` attribute is a private implementation detail and its shape is not part of the client library's public contract.

A test suite for the patcher itself should cover both branches:

```python
import pytest
from pathlib import Path
from .patcher import Alert, claude_plan

@pytest.mark.asyncio
async def test_patch_success(monkeypatch):
    monkeypatch.setattr("patcher.run_tests", lambda _: True)
    alert = Alert(
        dependency="express",
        current_version="4.18.2",
        new_version="4.19.0",
        ecosystem="npm",
        manifest_path="tests/fixtures/package.json",
        pr_url="https://github.com/example/repo/issues/1",
    )
    result = await claude_plan(alert)
    assert result.success is True

@pytest.mark.asyncio
async def test_patch_failure_rolls_back(monkeypatch):
    monkeypatch.setattr("patcher.run_tests", lambda _: False)
    alert = Alert(
        dependency="express",
        current_version="4.18.2",
        new_version="4.19.0",
        ecosystem="npm",
        manifest_path="tests/fixtures/package.json",
        pr_url="https://github.com/example/repo/issues/1",
    )
    result = await claude_plan(alert)
    assert result.success is False
    assert result.new_pr_url is None
```

Run them:

```bash
pip install pytest pytest-asyncio
pytest .claude-agent/patcher_test.py -v
```

Stubbing `run_tests` is deliberate. The unit test is verifying the control flow — that a failure produces a rollback and no PR — not the container runtime. Container behavior belongs in a separate integration test.

## Measuring whether it works

Any claim about this design improving a workflow has to be measured on your own repository. The instrumentation above gives you the raw material. What to collect:

- **Gate accuracy.** Count patches where the test suite passed but the change later broke something (a false negative), and patches where the suite failed but the change was fine (a false positive). The first is a coverage problem in your tests; the second is usually an environment mismatch.
- **Time to merge.** Compare the timestamp on the Dependabot alert to the merge timestamp on the resulting PR, before and after enabling the patcher.
- **Cost per scan.** If you run the test container locally, cost is wall-clock time. If you run it in a cloud function, cost is `memory_GB × duration_seconds × the provider's GB-second rate`. Record the memory limit and the observed duration for each run; the arithmetic is then trivial and the rate is on the provider's pricing page.
- **Rollback rate.** The fraction of runs that end in `git reset --hard`. A rising rollback rate is the earliest signal that your test suite or your container image has drifted.

Compare two equal-length windows — for example, four weeks before and four weeks after — and report the deltas with the sample sizes. A change from 12 failed runs out of 100 to 1 out of 100 is meaningful; a change from 3 out of 25 to 2 out of 25 is noise.

## Failure modes to expect

**Scope creep.** Agents asked to bump a version will sometimes reformat the whole file or "tidy" adjacent code. The mitigation is a prompt that states the constraint explicitly and a post-apply check that rejects diffs touching more than one file:

```python
changed = subprocess.run(
    ["git", "diff", "--name-only", "HEAD~1"],
    capture_output=True, text=True, check=True,
).stdout.split()
if len(changed) > 1:
    subprocess.run(["git", "reset", "--hard", "HEAD~1"], check=True)
    return PatchResult(success=False, log=["Patch touched multiple files."])
```

**Architecture mismatch.** If your CI and production run on different CPU architectures or base images, a containerized test on the wrong image gives false confidence. Pin the test image to match production and assert the architecture at runtime.

**Lockfile drift.** A dependency bump that updates `package.json` but not the lockfile will fail under `npm ci`. This is a *good* failure — it is the gate doing its job — but it means the agent's prompt should explicitly mention the lockfile, and the test command should be the same one CI runs.

**Private registry authentication.** The test container does not inherit your host's `~/.npmrc`. Mount it:

```python
volumes=[
    f"{service_dir}:/app",
    f"{Path.home()}/.npmrc:/root/.npmrc:ro",
],
```

The `:ro` suffix matters; the container has no reason to write to your credentials file.

**Credential expiry mid-run.** If the token used by the CLI is short-lived, a rotation failure will surface as an authentication error deep in the patch flow. Check token validity before starting work rather than after the diff has been applied.

## Choosing your gate

| Gate | Catches | Misses | Cost |
|---|---|---|---|
| Lint only | Syntax and style | Runtime breakage | Seconds |
| Unit tests | Logic regressions in covered code | Integration and environment issues | Seconds to minutes |
| Containerized test suite | Environment, native modules, lockfile | Untested code paths | Minutes |
| Containerized suite + smoke test | The above plus startup failures | Subtle behavioral changes | Minutes |

Most teams should start at the containerized test suite and add a smoke test once the patcher runs reliably. Lint-only gating is not a gate at all for this purpose.

## FAQ

**Can this run outside GitHub?**
Yes. The publish step is the only GitHub-specific part. Substitute the equivalent CLI or API call for your host — for example, a merge-request creation call — and keep the branch-created-after-tests-pass ordering.

**What if the agent generates a patch that does not apply cleanly?**
`git apply` will fail and raise. Catch that before committing, log the diff, and skip the alert. Do not fall back to `patch --fuzz`; a fuzzy apply is exactly the kind of silent change the gate exists to prevent.

**How do you audit decisions?**
Write one JSON record per run containing the alert, the diff path, the test exit code, the duration, and the resulting PR URL. Ship those records to append-only storage. The diff file plus the exit code is sufficient to reconstruct any decision.

**Should the agent ever merge?**
Not automatically. The gate proves the tests pass; it does not prove the change is desirable. Keep a human in the loop on the merge.

## Do this next

Pick one real Dependabot alert in a repository you own, set the `alert` object at the bottom of `patcher.py` to match it, and run the script once. Watch which step fails first — the diff apply, the container start, or the test command. That single run tells you more about your repository's readiness for agentic patching than any amount of reading, and it takes about thirty minutes including the container pull.
