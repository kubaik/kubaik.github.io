# Tech debt: when half your stack is unfamiliar

A stack assembled by several contributors over time tends to expose the difference between code that works and code that can be trusted. That difference only shows up under the exact conditions nobody tests for.

## The problem: decision-making under unfamiliarity

Consider a common situation. The frontend is React, the backend is Django, and infrastructure is split between AWS Lambda and ECS Fargate. A contractor adds a Rust service for image processing. Another contributor introduces Terraform for a side project. A third swaps out Redis for Valkey because of licensing. Half the stack is now software nobody on the team has written a line of code in — and the same one or two people still approve pull requests, debug production incidents, and explain the architecture to a non-technical co-founder.

The confusion is not primarily about syntax. It is about not knowing what "normal" looks like in those tools. When something breaks at 2 AM, it is hard to tell whether an error message reflects a real bug or a misconfiguration, and whether a proposed fix is idiomatic or a hack.

This is a useful definition of technical debt: not the code itself, but the decision-making overhead it creates. The reassuring part is that expertise in every tool is not required. What is required is a repeatable process for evaluating, debugging, and governing tools you don't fully understand.

## The real cause: missing observability, not missing knowledge

The root cause is rarely a lack of knowledge. It is a lack of *observability into the tool's behavior*. When a tool is well understood, there are mental models for its failure modes. A Django `OperationalError: FATAL: too many connections` points to connection pooling, not more RAM. A Node.js `MaxListenersExceededWarning` usually signals a leaked event listener, not a memory leak. With unfamiliar tools, those heuristics are absent, and the only fallback is documentation often written for people who already know the context.

The surface symptom is "I don't know this tool." The underlying problem is "I have no way to verify whether this tool is behaving correctly." That is a process problem, not a knowledge problem, and it compounds when there is no senior engineer to ask and no reviewer who has seen the tool in production.

A typical failure mode is the cargo-cult configuration: a config is copied from a blog post or GitHub issue, it works in staging, and then it fails in production because the post assumed a different version, a different cloud provider, or different defaults. Without knowing the tool's actual defaults, it is impossible to tell the difference.

## Fix 1 — establish a baseline before diagnosing anything

**Symptom pattern:** intermittent failures that don't correlate with load, vague error messages, logs that are either too verbose or too quiet. Suspicion falls on the tool itself, but there is no way to prove it.

**Cause:** there is no baseline. Without knowing what normal behavior looks like, anomalies are invisible. This is the most common source of decision paralysis with unfamiliar tools: it is impossible to tell whether the tool is misbehaving or being misused.

**Fix:** run the tool in isolation with a known input and record the distribution of results. For an inherited Rust image-processing service, a small harness that feeds it a 1 MB JPEG and measures time and memory, repeated ten times, produces a distribution. A 200 ms processing time is then recognizably normal and a 2-second time is recognizably an anomaly. The same numbers can be compared against documented expectations: if the docs claim 100 requests per second and the measurement shows 10, something is wrong.

```python
import requests
import time
import statistics

url = "http://localhost:8080/process"
times = []
for i in range(10):
    with open("test.jpg", "rb") as f:
        files = {"image": f}
        start = time.perf_counter()
        resp = requests.post(url, files=files)
        elapsed = (time.perf_counter() - start) * 1000  # ms
    assert resp.status_code == 200, f"Request failed: {resp.text}"
    times.append(elapsed)

print(f"Median: {statistics.median(times):.1f} ms")
print(f"p95: {statistics.quantiles(times, n=20)[18]:.1f} ms")
print(f"Max: {max(times):.1f} ms")
```

Run this against staging. A median of 150 ms with a p95 of 180 ms is a tight baseline; a p95 of 800 ms is a problem. Either way, the output is a concrete number to compare against when someone proposes a change, and a way to verify a fix: if the p95 drops to 200 ms after a configuration change, the change did something measurable. Without a baseline, every decision is a guess.

## Fix 2 — write down the tool's contract

**Symptom pattern:** the tool works in development but fails in production. Errors mention environment variables, permissions, or network timeouts. The configuration looks correct on inspection.

**Cause:** a missing *contract* between the tool and the rest of the system. Unfamiliar tools often carry implicit assumptions: a specific shared library version, the presence of a file, a maximum acceptable network latency. When those assumptions are violated, the failure messages often don't mention the assumption at all.

**Fix:** document the contract explicitly — environment variables, file paths, network access, CPU and memory limits — then verify each item in the production environment.

A classic example is glibc. A Rust binary built on Ubuntu and deployed to Alpine Linux fails at runtime with `version 'GLIBC_2.34' not found`, because Alpine uses musl rather than glibc. Building inside a container that matches the runtime image eliminates the whole class of failure:

```dockerfile
FROM rust:1.75-slim-bookworm AS builder
WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates
COPY --from=builder /app/target/release/my-service /usr/local/bin/
CMD ["my-service"]
```

The broader point: treat unfamiliar tools like any other dependency. They have a contract, and that contract needs to be tested in production. A common mistake is assuming that because a tool is written in a familiar language, its runtime behavior will be familiar. Rust, Go, and Python have different deployment characteristics, and those differences must be verified rather than assumed.

## Fix 3 — measure the environment, not just the tool

**Symptom pattern:** the tool works in isolation but fails when integrated with other services. Errors are inconsistent — sometimes a timeout, sometimes a 500, sometimes a silent failure — and behavior changes across regions or cloud providers.

**Cause:** sensitivity to environmental factors that were never accounted for: network latency, DNS resolution, clock skew, or the behavior of a managed service. A tool might assume a cache call completes in under 5 ms; if that cache lives in another availability zone, a 20 ms round trip can trigger timeouts.

**Fix:** measure the environment and compare it to the tool's assumptions. A short script measures round-trip time to a cache instance:

```python
import redis
import time

r = redis.Redis(host='cache.internal.example', port=6379)
times = []
for _ in range(100):
    start = time.perf_counter()
    r.ping()
    times.append((time.perf_counter() - start) * 1000)

times.sort()
print(f"Median RTT: {times[50]:.2f} ms")
print(f"p95 RTT: {times[95]:.2f} ms")
```

A median RTT of 1 ms indicates the same availability zone; 15 ms suggests a cross-AZ hop. That 15 ms may be fine for most tools, but a tool with a 10 ms internal timeout will fail. The fix might be to move the service closer, raise the timeout, or add a local cache. The point is that the environment's behavior has to be visible before it can be reasoned about.

The table below lists environmental factors worth measuring, with the commands that produce numbers. The impact column is illustrative — actual values depend on your topology and provider, so measure rather than assume.

| Factor | Illustrative impact | How to measure |
|--------|--------------------|----------------|
| Cross-AZ latency | Sub-millisecond to a few ms added per call | `ping` or `traceroute` between hosts |
| Cross-region latency | Tens to hundreds of ms added per call | `ping` between regions |
| DNS resolution | A few ms first lookup, then cached | `dig` with timing, or `resolvectl statistics` |
| Clock skew | Drift that grows with uptime | `chronyc tracking` or `ntpq -p` |
| Disk I/O | Sub-ms to tens of ms per read | `fio` with a workload resembling yours |

Prioritize by blast radius: cross-region latency and disk I/O usually dominate for data-intensive tools.

## How to verify a fix actually worked

Verification is where solo engineers most often cut corners. A fix is applied, the symptom disappears, and everyone moves on. Without verification, there is no way to know whether the root cause was addressed or merely masked, and the symptom may return under different conditions.

To verify properly, reproduce the original failure and confirm it is gone. If the failure was a timeout, write a test that sends a request with a tight timeout and asserts success. If it was a memory leak, run the tool for a sustained period and assert that memory usage stays flat. Run the test before and after the fix: the before run should fail, the after run should pass. That gives both confidence and a regression test.

A common mistake is verifying only in development, where environments are more forgiving — lower latency, more memory, fewer concurrent requests. Verify in an environment that resembles production as closely as possible, and compare the same metrics before and after: latency percentiles, error rate, memory usage, CPU usage. If the numbers don't improve, the fix didn't work, regardless of whether the symptom disappeared. Symptoms are intermittent; measurements are not.

## Preventing the next unfamiliar tool from becoming a liability

Prevention means building a system that makes unfamiliar tools less risky.

**Limit adoption.** Every new tool adds a maintenance burden. A useful heuristic is to adopt a new tool only when it replaces two existing tools or solves a problem the current stack has already failed to solve. That forces an explicit justification for the added complexity.

**Create a tool card for every tool in the stack.** A tool card is a one-page document answering: what does this tool do, what version are we on, what are its dependencies, what are its failure modes, how is it monitored, where is its documentation. This matters most for tools nobody on the team chose. Requiring a tool card as part of the pull request that introduces a tool forces the author to think about maintenance and leaves a reference for the next incident.

**Standardize observability.** If every tool emits logs in a different format, events can't be correlated. Structured logging libraries — `structlog` for Python, `winston` for Node.js — plus a central log store such as Grafana Loki or AWS CloudWatch make it possible to see what else was happening when a tool failed. Scattered logs make request tracing effectively impossible.

**Schedule regular tool audits.** Once a quarter, review each tool: is it still maintained, are there security updates, is there a newer major version with breaking changes. This prevents the slow accumulation of outdated dependencies that eventually become impossible to upgrade in one step.

## Related errors that tend to cluster

Unfamiliar tools produce errors that fall into a few categories, and recognizing the category usually points to the fix.

- `Connection refused` — often a service starting before its dependency is ready. Add a readiness check or retry with exponential backoff.
- `Out of memory: Killed process` — a memory leak or a limit set too low. Check the tool's documented memory guidance and measure actual usage under load.
- `Permission denied` on a file or socket — often a user or group mismatch, or a container with a read-only filesystem. Check UIDs, GIDs, and mount options.
- `SSL certificate problem: unable to get local issuer certificate` — the tool doesn't trust an internal CA. Add the CA certificate to the tool's trust store.

The categories are connectivity, permissions, resources, and configuration. Identifying the category narrows the fix considerably; not every error is a new mystery.

## When none of this works: an escalation path

Sometimes the baseline is verified, the contract is documented, the environment is measured, and the problem persists. At that point, escalate deliberately.

1. **Isolate the tool.** Run it in a minimal environment with no other services. If it works there, the problem is integration. If it fails there, the problem is the tool.
2. **Search the issue tracker** for the exact error message. If it's a known bug, there may be a workaround or patch. If not, open an issue with a minimal reproduction: version, environment, exact steps.
3. **Consider replacing the tool.** This is a last resort, but if a tool is poorly maintained or has fundamental design flaws, replacement is the right call. Use the tool card to evaluate alternatives.
4. **Ask for help.** Post in a community forum or chat channel, or bring in a consultant for a few hours. A fresh perspective often spots the problem quickly.

Set a time limit. If no progress has been made in four hours, escalate. Time spent grinding on one issue is time not spent elsewhere.

## A decision checklist for unfamiliar tools

- Is there a baseline measurement for this tool's normal behavior, and is it stored somewhere the team can find it?
- Is the tool's contract written down, and has each item been verified in production?
- Are the environmental factors the tool depends on (latency, DNS, clock, disk) measured rather than assumed?
- Does a test exist that would fail if the original problem returned?
- Does the tool have a card covering version, dependencies, failure modes, monitoring, and documentation?
- Is the tool's log output structured and shipped to the same place as everything else?
- When was this tool last audited for maintenance status and security updates?
- If this tool failed right now, is there a documented escalation path, and a time limit on how long to debug before escalating?

## Frequently asked questions

**How do I decide which unfamiliar tool to learn first?**
Prioritize by blast radius. The tool whose failure takes down the whole application comes first — usually the database, the web server, or the message queue. Background jobs and analytics tooling can wait. Start with failure modes: what happens when it runs out of memory, when the network partitions, when the disk fills. Those are the scenarios that appear in production.

**What's the best way to document a tool I don't understand?**
Use a tool card. Record the version, dependencies, configuration, and failure modes, plus a link to official documentation and relevant issues. Add a "known gotchas" section for anything that was surprising. Keep it in the repository next to the code and update it whenever something new is learned.

**How can I tell whether a tool is misbehaving or I'm misusing it?**
Compare behavior to a baseline. Run the tool with a known input and measure the output. If the output deviates from documented or expected behavior, the tool is misbehaving; if it matches, the usage is likely wrong. Check the logs too: most tools emit warnings when used incorrectly, and silent failure is itself a distinct problem worth investigating.

**When should I replace a tool instead of learning it?**
Replace it when the cost of learning exceeds the cost of switching. Clear signals include: no longer maintained, critical unpatched vulnerabilities, no support for your platform, or resource requirements beyond what you can afford. Track how much time maintenance actually consumes; if it exceeds what a migration would cost, switch. Be careful, though — switching has its own cost, and the new tool needs to be genuinely better, not merely different.

## The one thing to do in the next 30 minutes

Open your repository and find the tool you understand least. Create a `TOOL_CARD.md` in the root directory containing the version, the command that checks its health, and the three most likely failure modes you can name or imagine. Commit it. The next time that tool breaks, the card will cut the diagnosis time substantially — and writing it forces a clear separation between what you actually know and what you are guessing.
