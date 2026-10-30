# AI fixes legacy code nobody owns

## Why legacy systems drift away from their documentation

The official documentation for a legacy service is often the last accurate artifact anyone produced. What it rarely covers is what happens years into production, after emergency hotfixes, silent patches, and one-off config edits have accumulated. This article addresses that gap: how to use AI as a triage lens over code that no longer has a clear owner.

A legacy codebase resembles an untended garden. When nobody prunes it, the weeds take over, the paths disappear, and what used to be a simple flower bed becomes a tangle of dead branches. Teams commonly avoid touching anything older than five years, and the teams that do touch it rarely have time to refactor — they need the thing to keep running until the next planning cycle. AI can help, but not in the way vendor marketing usually frames it.

A typical failure mode is a mismatch between the documented stack and the actual runtime. Docs might claim a service runs on a specific framework version, while the deployed binaries carry patches from a later minor release, a custom data-access layer that bypasses the ORM, and a repository abstraction that no longer matches the code. A common piece of folklore in such teams is "don't touch the DAOs — they break every time." That folklore is usually a signal that the documented architecture and the production reality have diverged.

The second gap is organizational. Nobody wants to own legacy code. When developer tenure is short, the original authors leave, tribal knowledge evaporates, and the system is held together by habit. AI tools that promise to "automate refactoring" often assume there is someone available to review the changes. In practice, many teams treat legacy code like a rental car: they don't want to modify it, they just want it to run until they can trade it in.

The third gap is measurement. Technical debt is often tracked as a generic percentage or a ticket count rather than as an observable property of the running system. A useful reframing is to ask: what is the redeploy time, the deployment failure rate, the p95 latency of the slowest path, the memory headroom under peak load? These are measurable, and they are what AI triage should be anchored to. A profiler run during a production-like load test will often reveal that an apparent "legacy Java slowness" is actually a single thread-pool starvation issue in a library that nobody has looked at in years. The docs say nothing about thread pools, because the docs were written before the thread pools became the bottleneck.

AI cannot replace the judgment of a developer who understands the business domain, but it can bridge the gap between the idealized system and the messy reality. The useful framing is AI as a lens, not a replacement. For example, an AI-assisted dependency graph over a mid-sized Python codebase can surface a circular dependency between two modules that has been silently patched for years. The original developers may have known only that updating either module broke the other. The AI does not fix the cycle; it makes the cycle visible, which is the precondition for deciding whether to break it.

The real win is not automating away the legacy. It is making the legacy's quirks visible to the people who have to live with it.

## How AI-assisted triage actually works

Most AI tooling for legacy code falls into two categories: code generation and code analysis. Generation tools write new code or refactor existing code. Analysis tools scan a codebase and produce insights. Neither works well alone for a legacy system, because generation without context produces plausible-looking changes that break undocumented behavior, and analysis without runtime data produces a list of issues that may not correspond to anything users experience.

The productive pattern is to use AI as a forcing function for documentation. Legacy systems accumulate undocumented behavior because nobody has time to write it down. AI can reverse-engineer some of that behavior from code, logs, and traces, but only if it is given the right context. The inputs that matter are: the code, the runtime traces, the error logs, and the deployment history.

A workflow that holds up in practice:

1. **Static analysis first.** Run a static analysis tool over the codebase. Focus on security hotspots, performance anti-patterns, and unused dependencies. A common finding is a class using a deprecated API that was patched for a symptom but never migrated off the deprecated call. The deprecation warning persists across years of patches.
2. **Runtime traces next.** Use a lightweight profiler — Java Flight Recorder for JVM services, `py-spy` for Python, `perf` for native binaries. Capture traces under a production-like load. Generate a flame graph and look for hot paths that are not documented. A frequent discovery is a service that is "supposed to be fast" but has a multi-hundred-millisecond delay caused by a nested loop introduced in a hotfix and never removed.
3. **AI contextualization.** Feed the static analysis results and runtime traces into an LLM with a structured prompt: given these findings, what are the top three risks, what behaviors are likely to break in production, and what undocumented assumptions exist? Long-context models handle this well when the input is curated. The model is not authoritative, but it surfaces patterns a human reviewer might miss — for example, a single SQL query executed inside a loop across a dozen files, where the query itself is fast but the loop is the real problem.
4. **Documentation generation.** Use the model to draft markdown that describes the behaviors it found. Treat the output as a first draft, not as truth. A draft that describes undocumented retry logic in a Node.js service is useful even if every sentence needs verification, because it gives reviewers something concrete to correct.
5. **Human review and prioritization.** AI can surface risks; it cannot rank them by business impact. A risk-scoring pass — high (production outage likely), medium (performance degradation likely), low (minor) — should be adjusted by someone who knows which endpoints matter. A `SimpleDateFormat` usage in a rarely accessed admin endpoint is not the same risk as the same usage in a payment path.

The combination is what produces value. Static analysis finds structural issues. Runtime traces find performance issues. AI contextualization connects them and surfaces undocumented behavior. The result is not a refactored system — it is a system whose quirks are visible, which is the first step toward taming it.

A concrete example of the combination working: static analysis flags a SQL injection vulnerability in a commit from several years ago. It misses that the same endpoint is also called from a background job under a different user context. When the runtime traces are fed to the model, the discrepancy is surfaced. The static analysis was correct but incomplete; the AI filled the gap.

The other surprise for newcomers is how much context the model needs. A naive prompt like "analyze this codebase" produces useless output. Feeding the model the deployment topology changes its conclusions. A memory leak in a Python service looks different when the model knows the service runs in a Kubernetes pod with a 512 MB memory limit — the leak is still real, but the eviction policy becomes the more urgent concern. Context changes priority, not just accuracy.

## A worked example: Django monolith triage

The workflow below uses a hypothetical Python/Django monolith as the subject. The numbers are illustrative, chosen to make the arithmetic explicit; they are not measurements from a specific system. The commands are real and runnable.

Assume the system is ~67,000 lines of code, has no tests, and has a reputation for being "unstable" despite running in production for years without a major incident. The stack is Python 3.8, Django 2.2, Celery 4.4, PostgreSQL 12, Redis 6.2, deployed on a single EC2 instance.

### Step 1: Static analysis

Run Semgrep with the auto and security-audit rulesets:

```bash
pip install semgrep
semgrep --config=auto --config=p/security-audit --error --json --output=semgrep-results.json .
```

This produces a JSON report. Typical findings in a codebase of this age:

- A Django view using `django.utils.timezone.now()` instead of `timezone.now()` (deprecated import path)
- A raw SQL query concatenating user input (SQL injection risk)
- A Celery task using `pickle.loads()` on data from a queue (arbitrary code execution risk if the queue is ever compromised)

For code-quality metrics such as duplication and complexity, a code-quality scanner in the Sonar-family category can be run in a container. The exact image tag and license terms change frequently, so pin whatever version your organization has approved rather than copying a tag from an article.

The combined output gives a baseline of structural issues. Structural issues are half the story; they do not tell you which paths are actually exercised in production.

### Step 2: Runtime profiling

Install `py-spy` and capture a CPU profile from a running worker:

```bash
pip install py-spy
py-spy top --pid <PID> --duration 30 --format speedscope > profile.json
```

The resulting flame graph typically reveals something like:

- A ~420 ms hot path in an API endpoint that should be under 100 ms
- A ~150 ms delay in a Celery task that was assumed to be fully asynchronous
- A ~200 ms delay in a database query using `SELECT *`

For distributed tracing, an OpenTelemetry-compatible collector or a managed APM can be used. The instrumentation pattern is the same regardless of vendor:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(provider)
```

The traces tell you which issues matter in production. The static analysis tells you which issues exist in the code. Together they produce a prioritized list.

### Step 3: AI contextualization

Feed the curated findings into a long-context model. A prompt template that works:

```
You are an experienced software engineer specializing in legacy systems.

Analyze the following findings from a codebase:
- Tech stack: Python 3.8, Django 2.2, Celery 4.4, PostgreSQL 12, Redis 6.2
- Environment: single EC2 instance, no containerization
- Business domain: B2B SaaS for logistics management
- Known pain points: slow API responses, flaky Celery tasks, intermittent outages

Static analysis results:
[Paste Semgrep output]

Runtime traces:
[Paste py-spy / tracing output]

Answer:
1. Top 3 risks, ranked by likelihood and impact.
2. Specific production scenarios likely to break.
3. Undocumented assumptions in the system.
4. Quick wins implementable within two weeks.
```

A representative response for this kind of input:

1. **Top risks**: the `SELECT *` query on the hot path (420 ms, high call volume); the Celery race condition where some tasks run synchronously and block the API; the SQL injection in the raw query (low frequency, high impact).
2. **Likely break scenarios**: latency spikes at peak hours; Celery task timeouts during bulk operations; connection pool exhaustion under N+1 query load.
3. **Undocumented assumptions**: the Redis cache is used only for session storage, not query results; the Celery queue is assumed FIFO, but Redis does not guarantee ordering; the PostgreSQL connection pool is sized for average load, not peak.
4. **Quick wins**: add indexes for the slow queries; replace `SELECT *` with explicit field lists; add caching for the hot endpoint.

The model's analysis overlaps with what a careful manual review would find, but it surfaces the undocumented assumptions faster. The assumption about Redis not being used for query caching is the kind of detail that typically lives only in a departed engineer's memory.

### Step 4: Documentation generation

Turn the model's output into a living document. A small script is enough:

```python
from pathlib import Path
import json

ai_output = json.loads(Path("ai-analysis.json").read_text())

doc = f"""# Legacy Django System: Known Quirks

## Overview
- Stack: Python 3.8, Django 2.2, Celery 4.4, PostgreSQL 12, Redis 6.2
- Environment: single EC2 instance, external PostgreSQL

## Top Risks

### 1. Slow endpoint `/api/v1/shipments/`
- Hot path: ~420 ms (target <100 ms)
- Root cause: `SELECT *` plus N+1 queries
- Fix: add covering indexes, replace `SELECT *`

### 2. Celery race condition
- Symptom: a fraction of tasks run synchronously and block the API
- Root cause: Redis queue is not FIFO; connection pool exhaustion
- Fix: task deduplication, larger queue, explicit retry policy

### 3. SQL injection in raw query
- Impact: high if the admin UI is reachable
- Fix: parameterized queries

## Undocumented Assumptions
- Redis is used only for sessions, not query results
- Celery queue ordering is assumed but not guaranteed
- PostgreSQL pool sized for average load, not peak
"""

Path("LEGACY_QUIRKS.md").write_text(doc)
```

This file is not authoritative. It is a starting point that the team can correct, and the corrections are themselves valuable because they capture knowledge that was previously oral.

### Step 5: Prioritization

Prioritize by impact and effort, with the impact grounded in the traces:

1. Replace `SELECT *` with explicit fields — high impact, low effort
2. Add caching for the hot endpoint — high impact, medium effort
3. Fix the Celery race condition — medium impact, high effort
4. Parameterize the raw query — low frequency, but cheap to fix

After the first two items, a plausible outcome is that the hot path drops from ~420 ms to ~80 ms. This is an illustrative figure, not a measurement. The point of the exercise is that the team's confidence in the system improves because the quirks are visible and the fixes are bounded, not because the system was rewritten.

## What to instrument, and how to measure

When this workflow is applied, the temptation is to report impressive-sounding numbers. Resist that. The honest approach is to define what to measure and how, then let the measurements speak.

For a service like the one above, the minimum instrumentation set is:

- **Latency**: p50, p95, p99 per endpoint, exported from the tracing layer. Compare before and after a change on the same load profile.
- **Error rate**: HTTP 5xx per endpoint, plus task failure rate for background workers.
- **Queue depth and task duration**: from the broker's metrics, plus per-task timing.
- **Database query counts and durations**: from `pg_stat_statements` or the equivalent for your database.
- **Memory and CPU**: from the container or host metrics, sampled at the same interval as the load test.
- **Deployment duration and failure rate**: from the CI/CD system, not from memory.

A load test should be run before and after each change, with the same request mix and the same concurrency. The comparison is only meaningful if the load profile is held constant. A useful command for a Python service is `py-spy record` over the duration of the load test, which produces a flame graph that can be diffed against the baseline.

The reason to insist on this discipline is that AI-generated analyses are easy to over-trust. A model can produce a confident-sounding narrative about performance that has no basis in the traces. The traces are the ground truth; the model is a lens over them.

## Failure modes and how to avoid them

AI is not a silver bullet for legacy systems. The following failure modes are common enough to plan for.

### Hallucinated dependencies

Models frequently invent dependency versions. A model may claim a service depends on a specific version of a library because it saw the import name and assumed the latest release. The fix is to cross-check any AI-generated dependency list against the actual manifest.

```bash
pip install pipdeptree
pipdeptree -p numpy
```

This prints the installed version, which is the only version that matters for the running system.

### Context window exhaustion

Large codebases do not fit in a context window, even a long one. Feeding the entire codebase produces truncation or hallucination. Curate the input: only the files modified in the last year, or only the files with the highest issue counts from static analysis, or only the files that appear in the runtime traces. Reducing a 100k-line codebase to the 15 files with the highest issue counts often improves the accuracy of the analysis rather than degrading it.

### Over-reliance on AI-generated fixes

Models generate plausible patches. A patch that replaces a legacy date formatter with a modern one may look correct but break a downstream consumer that depends on the old format. Never apply an AI-generated fix without running the existing tests, and if there are no tests, write a characterization test first that captures the current behavior.

```java
// Legacy
SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd");
String dateStr = sdf.format(new Date());

// Modern equivalent
DateTimeFormatter dtf = DateTimeFormatter.ofPattern("yyyy-MM-dd");
String dateStr = dtf.format(LocalDate.now());
```

The two are not equivalent in all cases. `SimpleDateFormat` is not thread-safe and uses the default time zone; `LocalDate.now()` also uses the default time zone but is immutable. A characterization test should pin the expected output for a fixed clock and time zone before the change is made.

### False positives in security scans

Static analysis and AI both produce false positives. A `pickle.loads()` call is a genuine risk only if the data source is untrusted. A second scanner in the same category can be used to cross-check, but the final decision is a human one. The output of the scan should be treated as a queue of items to investigate, not as a list of confirmed vulnerabilities.

```bash
pip install bandit
bandit -r .
```

### Behaviors that are not in the code

The hardest failure mode is behavior that exists only in the runtime environment: an nginx config that bypasses an auth check, a cron job that mutates state, a manual database patch. AI cannot find these because they are not in the repository. The mitigation is to audit the deployment surface explicitly: server configs, cron entries, systemd units, environment variables, and any manual change logs. This audit is part of the workflow, not an optional extra.

### Cost and data exposure

LLM API calls have a cost, and sending proprietary code to a third-party endpoint has compliance implications. A common pattern is to use a local model for the first pass over the codebase, then a hosted model for the final contextualization over a curated subset. The local pass is cheap and keeps the bulk of the code on-premises; the hosted pass is limited to the files that matter. Before sending any code to an external endpoint, confirm that the endpoint's data retention terms are acceptable for the code in question.

## Choosing tools

The tooling landscape changes quickly, so the useful output is a set of categories and selection criteria rather than a fixed list.

| Category | What it does | What to check before adopting |
| --- | --- | --- |
| Static analysis (security) | Finds injection, unsafe deserialization, deprecated APIs | Rule coverage for your language; false-positive rate on your codebase; CI integration |
| Static analysis (quality) | Duplication, complexity, code smells | Whether the metrics map to anything you act on; license terms |
| Profilers | CPU and memory flame graphs | Overhead under production load; ability to attach to a running process |
| Distributed tracing | Cross-service latency and error attribution | Sampling strategy; storage cost; vendor lock-in |
| Long-context LLMs | Contextualization over curated findings | Context window; data retention terms; cost per analysis |
| Local LLMs | First-pass analysis without data egress | Quality on your language and code style; hardware requirements |

The selection criterion that matters most is whether the tool produces output you will act on. A scanner that produces 500 issues nobody triages is worse than a scanner that produces 20 issues the team fixes.

## FAQ

**Can AI refactor a legacy system automatically?**
No. It can propose changes and surface risks, but the decision to change behavior in a system with undocumented dependencies requires human judgment. Treat AI output as a draft.

**How large a codebase can be analyzed?**
It depends on the model's context window and on how much you curate the input. Curating to the files that appear in traces or that have the highest issue counts is usually more effective than trying to fit everything.

**Do I need a paid model?**
Not necessarily. A local model is often sufficient for the first pass. A hosted long-context model helps for the contextualization step, where the input is a curated set of findings rather than the whole codebase.

**What if there are no tests?**
Write characterization tests first. They capture current behavior, including behavior that may be a bug, and they give you a safety net for any subsequent change. This is the highest-value work you can do before applying AI-suggested fixes.

**How do I know the AI's analysis is correct?**
Cross-check it against the traces and against the code. Any claim that is not supported by a trace, a log line, or a specific code location should be treated as a hypothesis, not a finding.

## Action for the next 30 minutes

Pick one legacy service you have access to, run `py-spy top --pid <PID> --duration 30` (or the equivalent profiler for your runtime) against it during a period of normal load, and save the output. Then open the three files that appear hottest in the profile and check whether any of them have been modified in the last year. That single pass will tell you whether the system's real hotspots match the parts of the codebase your team actually understands — which is the precondition for any further triage.
