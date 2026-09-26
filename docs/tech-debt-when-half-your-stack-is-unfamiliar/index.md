# Tech debt: when half your stack is unfamiliar

webassembly server tends to expose the difference between working and being trustworthy. This post covers what comes after the happy path. It only shows up under the exact conditions nobody tests for.

## The error and why it's confusing

You're the solo technical lead. Your frontend is React 18, your backend is Django 4.2, and your infrastructure is a mix of AWS Lambda and ECS Fargate. Then a contractor adds a Rust service for image processing. Another contributor introduces Terraform 1.7 for a side project. A third person swaps out your Redis 7.2 cache for Valkey 7.2 because of licensing. Suddenly, half your stack is tools you've never written a line of code in. You're still the one who has to approve pull requests, debug production incidents, and explain the architecture to a non-technical co-founder. The confusion isn't just about syntax. It's about not knowing what "normal" looks like in those tools. When something breaks at 2 AM, you can't tell if the error message is a real bug or a misconfiguration. You can't tell if the proposed fix is idiomatic or a hack. This is the real cost of technical debt: not the code itself, but the decision-making overhead. The part that trips people up is that you don't need to become an expert in every tool. You need a repeatable process for evaluating, debugging, and governing tools you don't fully understand. That's what this post covers.

## What's actually causing it (the real reason, not the surface symptom)

The root cause isn't a lack of knowledge. It's a lack of *observability into the tool's behavior*. When you understand a tool, you have mental models for its failure modes. You know that a Django `OperationalError: FATAL: too many connections` means you need PgBouncer, not more RAM. You know that a Node.js `MaxListenersExceededWarning` is usually a sign of a leaked event listener, not a memory leak. With unfamiliar tools, you lose those heuristics. You're forced to rely on the tool's own documentation, which is often written for experts who already know the context. The surface symptom is "I don't know this tool." The real reason is "I don't have a way to verify whether this tool is behaving correctly or not." That's a process problem, not a knowledge problem. And it gets worse when you're the only engineer. There's no senior person to ask. There's no code review from someone who's seen the tool in production. You're making architectural decisions based on incomplete information, and the cost of a wrong decision is paid in production incidents. A common failure mode here is the "cargo cult" trap: you copy a configuration from a blog post or a GitHub issue, it works in staging, and then it fails in production because the blog post assumed a different version or a different cloud provider. You can't tell the difference because you don't know what the tool's defaults actually are.

## Fix 1 — the most common cause

**Symptom pattern:** You're seeing intermittent failures that don't correlate with load. The error messages are vague. The tool's logs are either too verbose or too quiet. You suspect the tool itself is buggy, but you can't prove it.

**Cause:** You're missing a baseline. You don't know what the tool's normal behavior looks like, so you can't spot anomalies. This is the most common cause of decision paralysis with unfamiliar tools. You're not sure if the tool is misbehaving or if you're misusing it.

**Fix:** Establish a baseline by running the tool in isolation with a known input. For example, if you've inherited a Rust service that processes images, write a small test harness that feeds it a 1 MB JPEG and measures the time and memory usage. Do this 10 times. You'll get a distribution. Now you know that a 200 ms processing time is normal and a 2-second processing time is an anomaly. You can also use this to compare against the tool's documentation. If the docs say it should handle 100 requests per second and you're getting 10, you know something is wrong. Here's a Python example of a baseline harness for a hypothetical image processing service:

```python
import requests
import time
import statistics

url = "http://localhost:8080/process"
with open("test.jpg", "rb") as f:
    files = {"image": f}
    times = []
    for i in range(10):
        start = time.perf_counter()
        resp = requests.post(url, files=files)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        assert resp.status_code == 200, f"Request failed: {resp.text}"
        times.append(elapsed)

print(f"Median: {statistics.median(times):.1f} ms")
print(f"p95: {statistics.quantiles(times, n=20)[18]:.1f} ms")
print(f"Max: {max(times):.1f} ms")
```

Run this against your staging environment. If the median is 150 ms and the p95 is 180 ms, you have a tight baseline. If the p95 is 800 ms, you have a problem. This gives you a concrete number to compare against when someone proposes a change. It also gives you a way to verify a fix: if the p95 drops to 200 ms after a configuration change, you know it worked. Without a baseline, you're guessing. With a baseline, you're making an engineering decision.

## Fix 2 — the less obvious cause

**Symptom pattern:** The tool works fine in development but fails in production. The error messages mention environment variables, permissions, or network timeouts. You've checked the configuration and it looks correct.

**Cause:** You're missing a *contract* between the tool and the rest of your system. Unfamiliar tools often have implicit assumptions about their environment: the version of a shared library, the presence of a specific file, the latency of a network call. When those assumptions are violated, the tool fails in ways that are hard to diagnose because the error messages don't mention the assumption.

**Fix:** Document the tool's contract explicitly. Write down what the tool needs to function: environment variables, file paths, network access, CPU/memory limits. Then verify each one in your production environment. For example, a common trap with Rust services is that they're compiled with a specific glibc version. If you build on Ubuntu 22.04 and deploy to Alpine Linux, you'll get a `version 'GLIBC_2.34' not found` error at runtime. The fix is to build in a container that matches the production environment. Here's a Dockerfile snippet that does this:

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

This ensures the binary is built against the same glibc as the runtime. It's a small change, but it eliminates a whole class of "works on my machine" failures. The broader point is that you need to treat unfamiliar tools like any other dependency: they have a contract, and you need to test that contract in production. A common mistake is to assume that because a tool is written in a language you know, its runtime behavior will be familiar. Rust, Go, and even Python have different deployment characteristics. You need to verify them.

## Fix 3 — the environment-specific cause

**Symptom pattern:** The tool works in isolation but fails when integrated with other services. The errors are inconsistent: sometimes it's a timeout, sometimes it's a 500, sometimes it's a silent failure. The behavior changes when you deploy to a different region or a different cloud provider.

**Cause:** The tool is sensitive to environmental factors that you haven't accounted for: network latency, DNS resolution, clock skew, or the behavior of a managed service. This is especially common with tools that rely on external services, like a message queue or a database. For example, a tool might assume that a Redis call will complete in under 5 ms. If your Redis instance is in a different availability zone, the latency could be 20 ms, causing timeouts.

**Fix:** Measure the environment's characteristics and compare them to the tool's assumptions. Use a simple network latency test between your services. Here's a Python script that measures the round-trip time to a Redis instance:

```python
import redis
import time

r = redis.Redis(host='my-redis.example.com', port=6379)
times = []
for _ in range(100):
    start = time.perf_counter()
    r.ping()
    times.append((time.perf_counter() - start) * 1000)

print(f"Median RTT: {sorted(times)[50]:.2f} ms")
print(f"p95 RTT: {sorted(times)[95]:.2f} ms")
```

If the median RTT is 1 ms, you're in the same AZ. If it's 15 ms, you're crossing AZs. That 15 ms might be fine for most tools, but if your tool has a 10 ms timeout, it will fail. The fix could be to move the service closer, increase the timeout, or add a local cache. The key is to make the environment's behavior visible. Once you have numbers, you can make an informed decision. This is also where a comparison table helps. Here's a table of common environmental factors and their typical impact:

| Factor | Typical impact | How to measure |
|--------|----------------|----------------|
| Cross-AZ latency | 1–2 ms added per call | `ping` or `traceroute` |
| Cross-region latency | 50–200 ms added per call | `ping` between regions |
| DNS resolution | 5–20 ms first lookup, then cached | `dig` with timing |
| Clock skew | 1–100 ms drift per hour | `ntpq -p` or `chronyc tracking` |
| Disk I/O | 0.1–10 ms per read | `fio` benchmark |

Use this table to prioritize which factors to investigate first. Cross-region latency is usually the biggest culprit, followed by disk I/O for data-intensive tools.

## How to verify the fix worked

Verification is where most solo engineers cut corners. You apply a fix, the symptom disappears, and you move on. But without verification, you don't know if the fix addressed the root cause or just masked it. The symptom might return under different conditions. To verify properly, you need to reproduce the original failure and confirm it's gone. Start by writing a test that triggers the failure. If the failure was a timeout, write a test that sends a request with a tight timeout and asserts that it succeeds. If the failure was a memory leak, write a test that runs the tool for 10 minutes and checks that memory usage stays flat. Run this test before and after the fix. The before run should fail; the after run should pass. This gives you confidence that the fix is real. It also gives you a regression test for the future. A common mistake is to verify only in development. Development environments are often more forgiving: lower latency, more memory, fewer concurrent requests. You need to verify in an environment that resembles production. If you can't reproduce production exactly, use a staging environment that mirrors it as closely as possible. Measure the key metrics before and after: latency, error rate, memory usage, CPU usage. A typical verification might show that p95 latency dropped from 800 ms to 200 ms, and error rate dropped from 5% to 0.1%. Those are the numbers you want to see. If the numbers don't improve, the fix didn't work. Don't assume it did because the symptom disappeared. Symptoms can be intermittent. Numbers are objective.

## How to prevent this from happening again

Prevention is about building a system that makes unfamiliar tools less risky. The first step is to limit the number of tools you adopt. Every new tool adds a maintenance burden. A common heuristic is to adopt a new tool only when it replaces two existing tools or solves a problem you've already tried to solve with your current stack. This forces you to justify the complexity. The second step is to create a "tool card" for each tool in your stack. A tool card is a one-page document that answers: What does this tool do? What version are we on? What are its dependencies? What are its failure modes? How do we monitor it? Where is the documentation? This is especially important for tools you didn't choose yourself. When a contributor adds a new tool, require them to fill out a tool card as part of the pull request. This forces them to think about maintenance, and it gives you a reference when something breaks. The third step is to standardize your observability. If every tool emits logs in a different format, you can't correlate events. Use a structured logging library like `structlog` for Python or `winston` for Node.js. Send all logs to a central system like Grafana Loki or AWS CloudWatch. This way, when a tool fails, you can see what else was happening at the same time. A common failure mode is to have logs scattered across different services, making it impossible to trace a request. Centralized logging fixes that. Finally, schedule regular "tool audits." Once a quarter, review each tool in your stack. Is it still maintained? Are there security updates? Is there a newer version with breaking changes? This prevents the slow accumulation of outdated dependencies that become impossible to upgrade. For a solo engineer, this is 2–3 hours per quarter, but it saves days of firefighting later.

## Related errors you might hit next

When you're dealing with unfamiliar tools, certain errors tend to cluster. If you've just fixed a timeout issue, you might next encounter a `Connection refused` error. This often happens when a service starts before its dependency is ready. The fix is to add a health check or a retry with exponential backoff. Another related error is `Out of memory: Killed process`. This can happen when a tool has a memory leak or when its memory limit is too low. Check the tool's documentation for recommended memory settings. A third common error is `Permission denied` when accessing a file or a socket. This is often due to running the tool as a different user or in a container with a read-only filesystem. Check the user and group IDs. A fourth error is `SSL certificate problem: unable to get local issuer certificate`. This happens when a tool doesn't trust your internal CA. You need to add the CA certificate to the tool's trust store. Each of these errors has a specific cause and a specific fix. The key is to recognize the pattern and apply the right diagnostic. Don't treat every error as a new mystery. Most errors fall into a few categories: connectivity, permissions, resources, and configuration. Once you identify the category, the fix is usually straightforward.

## When none of these work: escalation path

Sometimes you've tried everything and the tool still fails. You've verified the baseline, documented the contract, measured the environment, and the problem persists. At this point, you need to escalate. The first step is to isolate the tool completely. Run it in a minimal environment with no other services. If it works there, the problem is integration. If it fails there, the problem is the tool itself. The second step is to check the tool's issue tracker. Search for your exact error message. If it's a known bug, there may be a workaround or a patch. If it's not, open a new issue with a minimal reproduction. Include the version, the environment, and the exact steps to reproduce. The third step is to consider replacing the tool. This is a last resort, but sometimes it's the right call. If a tool is poorly maintained or has fundamental design flaws, you're better off with a different tool. Use the tool card to evaluate alternatives. The fourth step is to ask for help. If you're part of a community, post in a forum or a Slack channel. If you have a budget, hire a consultant for a few hours. Sometimes a fresh pair of eyes can spot the problem in minutes. The key is to not spend days on a single issue. Set a time limit: if you haven't made progress in 4 hours, escalate. Your time is better spent on other things. As a solo engineer, you're the bottleneck. You need to protect your time.

## Frequently Asked Questions

**How do I decide which unfamiliar tool to learn first?**
Prioritize by blast radius. The tool that, if it fails, takes down your entire application is the one to learn first. For most solo engineers, that's the database, the web server, or the message queue. Tools that are used for background jobs or analytics can wait. Start by learning the tool's failure modes: what happens when it runs out of memory, when the network partitions, when the disk is full. Those are the scenarios you'll face in production. Once you know how to diagnose those, you can handle most incidents.

**What's the best way to document a tool I don't understand?**
Use a tool card. Write down the version, the dependencies, the configuration, and the failure modes. Include a link to the official documentation and any relevant GitHub issues. Add a section for "known gotchas" where you record the things that surprised you. This document becomes your reference during incidents. It also helps when you onboard a new contributor. Keep it in your repository, next to the code, so it's easy to find. Update it whenever you learn something new.

**How can I tell if a tool is misbehaving or if I'm misusing it?**
Compare its behavior to a baseline. Run the tool with a known input and measure the output. If the output deviates from the documentation or from your expectations, it's misbehaving. If it matches, you're probably misusing it. Another approach is to check the tool's logs. Most tools log warnings or errors when they're used incorrectly. If you see warnings, read them carefully. They often tell you exactly what's wrong. If you don't see any warnings, the tool might be silently failing, which is a different problem.

**When should I replace a tool instead of learning it?**
Replace it when the cost of learning exceeds the cost of switching. This is a judgment call, but some signals are clear: the tool is no longer maintained, it has critical security vulnerabilities, it doesn't support your platform, or it requires more resources than you can afford. If you're spending more than 4 hours a week on maintenance, it's probably time to switch. For a solo engineer, the switching cost is usually lower than the ongoing maintenance cost, because you don't have to coordinate with a team. But be careful: switching tools also has a cost. Make sure the new tool is actually better, not just different.

## The one thing to do in the next 30 minutes

Open your repository and find the tool you understand the least. Write a `TOOL_CARD.md` file for it in the root directory. Include the version, the command to check its health, and the three most common failure modes you've seen or can imagine. Commit it. This single file will save you hours the next time that tool breaks, and it forces you to articulate what you actually know versus what you're guessing. Do it now, before the next incident.


---

### About this article

**Written by:** [Kubai Kevin](/about/) — software developer based in Nairobi, Kenya, with 10+ years building production systems in fintech and AI.

**How this article was produced:** This site uses an automated LLM pipeline designed and maintained by the author. Topics are selected from real production experience. Drafts pass automated quality gates (minimum length, uniqueness, concrete metrics, versioned tools, code samples, absence of filler). Individual line-by-line human editing is not performed on every post before publication. Specific numbers, benchmarks and cost figures are illustrative; verify them against current official documentation before production use.

**Corrections:** Report errors via the [contact page](/contact/). Corrections are applied promptly.

**Last generated:** September 2026
