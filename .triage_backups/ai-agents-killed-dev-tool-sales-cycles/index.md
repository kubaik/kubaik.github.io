# AI agents killed dev tool sales cycles

Developer tool adoption is increasingly mediated by automation: CI pipelines, coding assistants, and scripted setup routines that evaluate a tool by trying to run it. A tool that cannot be installed and invoked programmatically within a short window often never reaches a human evaluator at all. The practical consequence is that integration design, documentation format, and pricing structure now carry as much weight as the core functionality.

## Why the evaluation step changed

Traditional evaluation assumed a human would read documentation, run a benchmark, and compare pricing. In agent-mediated workflows, that sequence is compressed. An agent typically does something like this:

1. Discover the install command or action reference.
2. Attempt installation in a sandbox or CI runner.
3. Invoke the tool on a sample input.
4. Parse the output for structured results.
5. Decide whether to keep or discard the dependency.

Each step is a potential failure point. A required secret that must be created manually, a multi-step authentication flow, or an output format designed only for human reading will cause the agent to abandon the tool. The human never sees the failure.

This is not a claim about a specific product or a measured conversion rate. It is a structural observation: when the first evaluator is a program, the cost of friction is paid before any human judgment is applied.

## Common failure modes in integration design

### Failure mode: multi-secret setup

A workflow that requires three separate secrets, a custom environment variable file, and a manual approval step will fail agent evaluation. Each secret is a step the agent cannot complete without human intervention. The fix is to reduce required configuration to a single credential, or to support anonymous operation for evaluation.

### Failure mode: human-readable-only output

If the tool prints a formatted table to stdout and nothing else, an agent cannot reliably parse results. Structured output — JSON, JSONL, or a documented schema — allows programmatic consumption. Human-readable formatting can be layered on top, but the machine-readable form should exist first.

### Failure mode: heavy installation footprint

A tool that requires downloading a large runtime, compiling native dependencies, or installing a language toolchain will time out in many sandboxes. Prebuilt binaries, container images, or single-file scripts reduce this risk. The relevant measurement is not "how long does it take on a fast laptop" but "how long does it take in a cold CI runner."

### Failure mode: interactive prompts

Any command that waits for stdin input will hang an agent. Flags for non-interactive mode, environment variables for defaults, and `--yes` style options are necessary for automated use.

## Patterns that reduce integration friction

### Single-file CI configuration

A GitHub Action or CI job that references a published action and one secret is the lowest-friction starting point. The action itself should handle dependency installation internally so the consuming workflow stays short.

```yaml
name: Lint
on: [push]
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run linter
        uses: example-org/lint-action@v1
        with:
          api_key: ${{ secrets.LINT_API_KEY }}
```

The `api_key` is the only required secret. Everything else is handled inside the action. This pattern is not specific to any vendor; it is a general shape that works for any tool that can be packaged as a container or Node action.

### Config import from existing tools

Teams already have configuration for other linters, formatters, or test runners. A command that reads those files and produces the tool's native config removes a manual translation step.

```bash
# Convert an existing config to the tool's format
lint-tool import-config --source .existingrc --output .lint-tool.yml
```

The value is not the conversion itself but the elimination of a human editing step. Agents can run this command; they cannot meaningfully edit a config file with comments and conditional logic.

### Machine-readable rule metadata

Documentation written for agents should be structured. A JSON schema per rule is more useful than a markdown page because an agent can parse it without natural language understanding.

```json
{
  "rule_id": "LINT001",
  "description": "Mutable default arguments are shared across calls.",
  "severity": "high",
  "fix": {
    "type": "replace",
    "pattern": "def func(arg=[]):",
    "replacement": "def func(arg=None):\n    if arg is None:\n        arg = []"
  }
}
```

The same data can be rendered as HTML for humans. The point is that the structured form is the source of truth, not an afterthought.

### Cached, low-latency endpoints

Agents have timeouts. An endpoint that takes several seconds to respond may be abandoned even if it would eventually succeed. Caching results by repository and commit SHA is a standard approach.

```python
from fastapi import FastAPI
from fastapi_cache import caches
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache

app = FastAPI()

redis = RedisBackend("redis://redis-master:6379", pool_size=20, timeout=500)
caches.set("default", redis)

@app.post("/lint")
@cache(expire=300)
async def lint(payload: LintRequest):
    cache_key = f"lint:{payload.repo_id}:{payload.sha}"
    cached = await redis.get(cache_key)
    if cached:
        return cached
    result = await run_linter(payload)
    await redis.set(cache_key, result, expire=300)
    return result
```

Note that the `@cache` decorator and the explicit `redis.get`/`redis.set` calls overlap; in a real implementation you would use one or the other, not both. The decorator handles the cache lookup and storage, so the manual calls are redundant. Choose the decorator for simplicity or the manual calls if you need custom cache keys or conditional caching.

To measure the effect of caching, instrument the endpoint with a counter for cache hits and misses, and log response times. A simple approach is to emit a structured log line per request with `cache_hit`, `duration_ms`, and `endpoint`, then aggregate with your existing log tooling. Compare p50 and p95 latency before and after enabling the cache.

### Offline-tolerant clients

Editors and CLIs used in environments with unreliable connectivity should queue work locally and sync when a connection is available. This is a general pattern, not specific to any region.

```typescript
const offlineQueue = new PersistentQueue('offline-lint');

workspace.onDidChangeTextDocument(async (event) => {
  if (!navigator.onLine) {
    await offlineQueue.add(event.document.uri.fsPath);
    return;
  }
  const result = await lintDocument(event.document);
  displayResults(result);
});

window.addEventListener('online', async () => {
  while (offlineQueue.size > 0) {
    const file = offlineQueue.pop();
    await lintDocument(Uri.file(file));
  }
});
```

The queue persists across editor restarts, which matters when a session ends before connectivity returns.

## Designing for agent use without neglecting humans

The goal is not to replace human-facing design but to ensure the machine path exists. A useful ordering is:

1. CLI with non-interactive flags and structured output.
2. CI integration that installs and runs in one step.
3. Machine-readable documentation (JSON schemas, OpenAPI specs).
4. Editor extension or UI for humans who want it.

Building the UI first and the CLI second inverts the dependency order. Teams that start with the CLI tend to have a smaller surface area to maintain and a clearer contract for automation.

## Pricing models that align with automated use

Seat-based pricing assumes a human logs in. When usage is driven by CI jobs and agents, seat counts do not reflect value. Two alternatives are common:

- **Usage-based**: charge per invocation, per fix applied, or per repository scanned. This aligns cost with consumption but requires metering and may need caps to avoid surprise bills.
- **Outcome-based**: charge per accepted change, per merged pull request, or per resolved issue. This aligns cost with value but requires a reliable signal of acceptance.

A hybrid is often practical: a free tier for evaluation with generous limits, then usage-based pricing above that. The free tier must be usable by an agent without a credit card or manual approval, or it will not serve its purpose.

## Measuring whether your integration passes the test

The "30-second test" is a heuristic. To make it concrete, measure the following in a cold environment:

- **Time from `git clone` to first successful tool invocation.** Instrument this by running the documented setup steps in a fresh container and timing each step.
- **Number of required secrets or manual steps.** Count them; each one is a potential failure point for automation.
- **Output parseability.** Attempt to parse the tool's output with a JSON parser. If it fails, the output is not machine-readable.
- **Non-interactive behavior.** Run the tool with stdin closed and no TTY. If it hangs or prompts, it is not agent-compatible.

These measurements can be collected with a shell script that runs in CI and reports the results. The script itself becomes a regression test for integration friction.

## A worked example: reducing setup steps

Consider a hypothetical tool that currently requires:

1. Install a CLI via a package manager.
2. Run `tool init` which prompts for a project name and API key.
3. Edit a generated config file to set the rule set.
4. Add a CI step that calls the CLI.

An agent attempting this will fail at step 2 because of the interactive prompt. The fix is to add flags:

```bash
tool init --non-interactive --project "$PROJECT" --api-key "$API_KEY" --ruleset default
```

Now step 2 is automatable. Step 3 can be eliminated by making the default ruleset sufficient for evaluation. Step 4 can be replaced by a published CI action that wraps the CLI. The result is a two-step setup: install, then run with flags.

The measurement to confirm the improvement: time the setup in a fresh container before and after the change. If the before time is dominated by waiting for human input, the after time will be dramatically lower even if the underlying work is identical.

## Decision checklist

Before shipping a developer tool, check the following:

- Can the tool be installed and invoked in a single CI step with one secret?
- Does it produce structured output by default or via a flag?
- Does it run without interactive prompts when stdin is closed?
- Is there a machine-readable schema for rules, config, or API?
- Does the free tier work without manual approval?
- Is there a documented way to measure setup time and output parseability?

If any answer is no, that is the next thing to fix.

## FAQ

**Does this mean human-facing documentation is unnecessary?**
No. Humans still read documentation, especially for troubleshooting and advanced configuration. The point is that the structured form should exist alongside the human-readable form, not instead of it.

**How do I know if agents are actually using my tool?**
Look for programmatic usage patterns: CI job runs, API calls without a browser user agent, invocations with non-interactive flags. Structured logging with a `client_type` field makes this measurable.

**What if my tool genuinely requires human judgment?**
Provide a default automated path for the common case and a human review step for exceptions. The automated path handles evaluation and routine use; the human path handles edge cases.

**Is usage-based pricing always better?**
No. It is better when usage correlates with value and when metering is reliable. If usage is sporadic or hard to meter, a flat fee with generous limits may be simpler.

## Action for the next 30 minutes

Run your tool's documented setup in a fresh container with stdin closed and no TTY, and time it. If it hangs, prompts, or takes more than a minute, identify the first blocking step and add a non-interactive flag or a default that removes it. That single change is the highest-leverage integration improvement you can make today.
