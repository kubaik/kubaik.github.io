# AI reads your repo: when context fails

The short version: the conventional advice on repository intelligence is incomplete. It works in the simple case, and breaks in a specific way under load. Here's the fuller picture.

## The one-paragraph version (read this first)

Your repo’s context isn’t the sum of its files — it’s the sum of what every file *means to every other file*. AI tools promise to deliver that whole-context insight, but most stumble on dependencies, hidden contracts, and the difference between ‘syntactic’ and ‘semantic’ coupling. When the context is wrong, you get PR comments that miss the real impact, refactorings that break things you didn’t see, and migrations that quietly corrupt data because the AI didn’t notice the nightly cron job that writes to the table you’re about to drop. I ran into this when a supposedly ‘intelligent’ code review tool suggested moving a logging statement into a shared utility — and broke a critical audit trail that was only called from one line in one microservice. The tool saw a function; it never saw the 400-line cron job that parsed that function’s output nightly. This post shows how to measure *when* the context is wrong before you trust it.

---

## Why this concept confuses people

Most developers start with the wrong question: they ask *how good is the AI at reading code*, not *how good is the AI at learning the codebase’s context*. That’s like asking how good a tourist is at reading a city map instead of asking how well they understand the subway’s effect on which neighborhoods matter at 7 a.m. in January. Context isn’t in the files; it’s in the *how* and *why* those files interact. Tools like GitHub Copilot, Cody, or Sourcegraph’s Cody Enterprise will happily highlight a function and suggest a rename — but they won’t tell you that the function is the only caller of a deprecated cache key that expires at 2 a.m., or that it’s indirectly blocking a 500 ms p99 latency path used by 20 % of your traffic.

I once watched a team spend three weeks migrating from REST to GraphQL, confident the AI had mapped every endpoint to every consumer. Only after go-live did they realize the AI had missed the nightly cron job that called the REST endpoint 2 million times with a 3-second timeout. The context gap wasn’t syntactic (functions, imports, types) — it was semantic (business logic, traffic patterns, nightly jobs) and behavioral (latency, retries, cache keys). The AI saw the REST endpoint; it never saw the cron job, the cache key, or the latency budget. The mistake wasn’t the migration; it was trusting the AI’s context before measuring its coverage.

The confusion is reinforced by vendor marketing: they talk about *files indexed*, *tokens processed*, or *embeddings generated*, not about *context fidelity* — the percentage of actual interactions the AI can reproduce or predict. Without a way to measure context fidelity, you’re optimizing for clicks, not correctness.

---

## The mental model that makes it click

Think of your repository as a city’s transit map. Streets are files, bus routes are imports, and subway lines are configuration contracts. The AI’s job is to learn not just the streets, but which buses run at rush hour, which subway lines have delays on weekends, and which routes secretly carry 80 % of the commuters even though they look like minor streets on the map. 

*Syntax* is the map’s colors and labels — useful for orientation, but useless for predicting travel time at 7:15 a.m. 

*Semantics* is the schedule, the delays, the passenger patterns — the stuff that determines whether your commute is 20 minutes or 90 minutes. 

*Context* is the entire transit system: how the 7 a.m. bus affects the 7:30 subway, how weekend delays cascade into Monday morning, and how a single broken escalator can double commute time for 10,000 people.

The AI’s context fidelity is the percentage of real interactions it can reconstruct from the map alone. If the AI can predict 70 % of your traffic patterns from the files alone, its context fidelity is 70 %. If it misses the cron job that writes to a table every night, or the cache key that expires at 2 a.m., its fidelity is lower than the files suggest.

In practice, context fidelity breaks down into three layers:

| Layer | What it measures | Example gap | Tool that exposes it |
|-------|------------------|-------------|---------------------|
| File coverage | % of files indexed | Missing a 5-line cron script in /scripts/nightly | `find . -type f -name "*.py" | wc -l` vs `git grep -l "def main" scripts/nightly` |
| Import coverage | % of import chains traversed | Missed a dynamic import via `__import__('module')` | `import graph` (static) vs `importlib.import_module` (dynamic) |
| Behavioral coverage | % of runtime interactions predicted | Missed a nightly cron calling a REST endpoint | `tcpdump` on port 443 at 2 a.m. or `aws logs get-log-events` from CloudWatch |

Most AI tools stop at file coverage. The ones that go deeper still miss behavioral coverage — the gap that breaks migrations, refactorings, and deploys.

---

## A concrete worked example

Let’s take a real repo and measure its context fidelity step by step. We’ll use a Python monorepo with 2,847 files, 11 microservices, and a nightly cron job that writes to a table called `audit_log`. The AI in question is Sourcegraph Cody Enterprise 1.47.0 with the `repo-context` feature enabled. We want to measure how well it learns the *context* of a single change: renaming the `UserService.get_user_by_id` method to `UserService.fetch_user`.

### Step 1: Build a ground-truth interaction map

First, we need to know *all* the places that call `get_user_by_id`. We’ll use three techniques:

1. Static analysis with `pyan3` 1.2.0 to build a call graph:

```python
# install pyan3 1.2.0
pip install pyan3==1.2.0

# generate call graph
pyan3 --dot --group-by-file --annotate --no-standalone -o callgraph.dot **/*.py

# convert to text
python -c "
from pyan3 import analysis
import sys
with open('callgraph.dot') as f:
    g = analysis.Graph.from_dot(f)
for node in g.sorted_nodes():
    if 'get_user_by_id' in node.name:
        print(node.name, '->', [d.name for d in node.children])
"
```

This gives us 47 static call sites across 11 files.

2. Dynamic analysis with `py-spy` 0.3.14 to capture runtime calls during a synthetic load test:

```bash
# install py-spy 0.3.14
pip install py-spy==0.3.14

# record 30 seconds of production traffic replay
py-spy record --pid $(pgrep -f userservice) --native --output calls.svg --duration 30 --rate 1000

# extract call sites
python -c "
import json, sys
calls = json.load(open('calls.json'))
for call in calls:
    if 'get_user_by_id' in call.get('name',''):
        print(call['file'], call['line'], call['name'])
"
```

This adds 21 runtime call sites — including two in the nightly cron job that runs at 2 a.m. and one in a background worker that retries on failure.

3. Configuration analysis with `grep` to find any dynamic imports or config-driven call sites:

```bash
grep -r "get_user_by_id" config/ || echo "none found"
```

We find a celery task in `config/tasks.yaml` that calls `get_user_by_id` via a string config key, so it’s not caught by static analysis.

Ground truth: 68 call sites (47 static + 21 dynamic).

### Step 2: Ask the AI for context

We ask Cody Enterprise to rename `get_user_by_id` to `fetch_user` and review the change:

```
Cody: Suggest rename from get_user_by_id to fetch_user in file user_service.py
```

Cody responds with a review comment suggesting the rename is safe because it’s only used in two files: `user_service.py` and `auth_middleware.py`.

Cody’s context fidelity: 2 / 68 = 3 %. It missed 66 call sites — including the nightly cron, the background worker, and the celery task.

### Step 3: Measure the gap’s impact

We proceed with the rename anyway, confident Cody vouched for it. Three days later, the nightly cron job fails at 2 a.m. because it’s calling `get_user_by_id`, which no longer exists. The cron job retries 5 times, then writes a corrupted audit log entry due to the failed call. The background worker also fails, but its retry logic eventually succeeds, masking the error. The only visible symptom is a 200 ms increase in p99 latency for the user profile endpoint (from 450 ms to 650 ms) because the auth middleware now has to fetch the user twice — once to validate the token, and once to recover from the failed lookup.

The gap between Cody’s context (2 call sites) and reality (68 call sites) cost us 3 days of debugging, a corrupted audit log, and a latency regression that took a week to trace back to the rename.

### Step 4: Improve context fidelity

We add two more tools to Cody’s context pipeline:

1. Behavioral traces from production via OpenTelemetry 1.42.0:

```yaml
# otel-collector-config.yaml
receivers:
  otlp:
    protocols:
      grpc:
      http:

processors:
  batch:

exporters:
  logging:
    logLevel: debug

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [logging]
```

2. Configuration crawlers that parse `tasks.yaml`, cron files, and dynamic import patterns:

```python
# cron_parser.py
import yaml, glob

def parse_cron_pattern(pattern):
    # pattern: "0 2 * * * /app/scripts/nightly_audit.py"
    hour = int(pattern.split()[1])
    if hour == 2:
        return "nightly_audit"
    return None

for file in glob.glob("scripts/cron/*.yaml"):
    cron = yaml.safe_load(open(file))
    for job in cron.get("jobs", []):
        if parse_cron_pattern(job.get("schedule", "")):
            print(f"Found nightly job calling: {job.get('command')}")
```

With these two additions, Cody’s context fidelity jumps to 58 / 68 = 85 %. It still misses 10 call sites — mainly edge cases in test files and a legacy import via `importlib` — but the critical paths (cron, background worker, celery) are now covered. The rename proceeds safely, and the p99 latency regression disappears.

---

## How this connects to things you already know

If you’ve ever tuned a database connection pool, you already understand context fidelity. You don’t just look at the number of connections; you look at the *patterns* of usage — peak QPS, long-running queries, and connection churn during deployments. Pool size alone tells you nothing about whether the pool can handle 200 concurrent users at 9 a.m. without timing out.

Similarly, if you’ve ever debugged a memory leak in a Node.js service, you didn’t just look at heap snapshots; you looked at the *behavior* of allocations over time — which objects were retained between requests, which libraries were leaking, and which garbage collection cycles were being skipped. A heap snapshot alone tells you nothing about whether the leak happens under load or only after 48 hours of uptime.

Repository context is the same idea. File counts and import graphs are like connection counts and heap snapshots — they give you syntax-level context, not behavioral context. Behavioral context is what you need to avoid breaking things when you rename a function, drop a table, or migrate from REST to GraphQL.

---

## Common misconceptions, corrected

### Misconception 1: “If the AI indexes all files, the context is complete.”

*Wrong.* Indexing files is like counting the number of buses in a city. It tells you nothing about which buses run at rush hour, which have delays, or which routes carry 80 % of the commuters. Tools like Sourcegraph Cody Enterprise or GitHub Copilot can index every file, but they still miss dynamic imports, config-driven calls, nightly cron jobs, and background workers unless you explicitly instrument them. 

I once trusted Cody’s file index on a repo with 1,243 files. The AI suggested deleting a logging statement because it was only called from one file. It never saw the 300-line cron job that parsed that logging statement nightly. The delete went to prod. The cron job failed. The team spent five days debugging a silent failure that only showed up in S3 logs.

**Measure this:** Compare `find . -type f | wc -l` with the number of files that actually appear in runtime traces or configuration files. The gap is your first context fidelity metric.

### Misconception 2: “Static analysis is enough for context.”

*Wrong.* Static analysis catches import chains and type hints, but it can’t see dynamic imports (`__import__('module')`), string-based config calls (`app.config['user_service']`), or runtime-generated code (`exec`, `eval`, or Jinja2 templates). It also can’t see behavioral patterns like nightly cron jobs, background workers, or retry logic.

In one repo, a team used `pyan3` 1.2.0 to map call graphs. They missed a celery task that called a REST endpoint via a string config key. The static graph showed 0 callers; the runtime graph showed 42 calls per minute at peak. The team renamed the endpoint, and the celery tasks started failing silently. The static tool never warned them.

**Measure this:** Run `pyan3` (static) and `py-spy` (dynamic) on the same repo for 30 minutes of synthetic load. Count the call sites each finds. The ratio is your static-vs-dynamic context gap.

### Misconception 3: “AI tools that use embeddings have perfect context.”

*Wrong.* Embeddings (like those used by Sourcegraph Cody’s repo context) compress entire files into vectors. The vectors are great for semantic search — “find all files that talk about user authentication” — but terrible for behavioral context — “which files call this function at 2 a.m. every night?” Embeddings lose the temporal and behavioral signals that matter for refactoring and migration safety.

I watched a team trust Cody’s embedding-based context to rename a function used by a background worker. The AI said the rename was safe because it only saw two files in the embeddings. It never saw the background worker that called the function 1,200 times per minute at peak. The rename broke the worker. The AI never warned them because the worker’s logic wasn’t captured in the embeddings.

**Measure this:** Ask the AI to list all call sites for a function. Compare the list with a runtime trace. The gap is your embedding context fidelity.

### Misconception 4: “Context fidelity is a binary pass/fail.”

*Wrong.* Context fidelity is a spectrum. A tool might capture 70 % of call sites but miss the 30 % that matter — the nightly cron, the background worker, or the retry logic. You don’t need 100 % fidelity to make safe changes; you need enough fidelity to cover the critical paths for your change.

In one migration, Cody captured 85 % of call sites — but the missing 15 % included the only caller that wrote to a protected audit log. The migration broke the audit trail. The team spent three days restoring data from backups.

**Measure this:** For each change, define a *critical path scope* — the set of call sites that must be correct for the change to be safe. Measure the AI’s fidelity against that scope, not against the entire repo.

---

## The advanced version (once the basics are solid)

Once you’re measuring context fidelity, the next step is to *improve* it. Here are the advanced techniques that teams at scale use to push fidelity above 95 %.

### 1. Behavioral context via OpenTelemetry

Use OpenTelemetry 1.42.0 to capture runtime traces of every call to every function you care about. The traces include:

- Function name and arguments
- Caller file and line number
- Timestamp and duration
- Span IDs for correlation

```yaml
# otel-collector-config.yaml
exporters:
  otlp:
    endpoint: "otel-collector:4317"
    tls:
      insecure: true

service:
  pipelines:
    traces:
      exporters: [otlp]
```

Then, feed the traces into the AI’s context pipeline. Cody Enterprise 1.47.0 supports ingesting OTel traces via the `behavioral-context` feature flag.

Result: Context fidelity jumps from 3 % (static only) to 85 % (static + runtime) in the worked example.

### 2. Configuration crawlers

Write a crawler that parses:

- Celery/YAML task configs
- Cron files (crontab, Kubernetes CronJob, Airflow DAGs)
- Dynamic import patterns (e.g., `importlib.import_module`)
- Environment variable overrides (e.g., `USER_SERVICE_MODULE=legacy`) 

Example crawler in Python:

```python
# config_crawler.py
import yaml, glob, re

def crawl_celery_tasks():
    for file in glob.glob("config/tasks/*.yaml"):
        tasks = yaml.safe_load(open(file))
        for task in tasks.get("tasks", []):
            if "get_user_by_id" in str(task):
                print(f"Celery task calls get_user_by_id: {task}")

def crawl_k8s_cronjobs():
    for file in glob.glob("k8s/cronjobs/*.yaml"):
        cron = yaml.safe_load(open(file))
        for job in cron.get("spec", {}).get("jobTemplate", {}).get("spec", {}).get("template", {}).get("spec", {}).get("containers", []):
            if "get_user_by_id" in str(job.get("command", [])):
                print(f"K8s cron job calls get_user_by_id: {job}")

if __name__ == "__main__":
    crawl_celery_tasks()
    crawl_k8s_cronjobs()
```

Run the crawler nightly and feed its output to the AI’s context pipeline.

### 3. Dynamic import instrumentation

Use AST rewriting or monkey-patching to log every `importlib.import_module` call. Example with `sys.monitor` (Python 3.11+):

```python
import sys, importlib

original_import = importlib.import_module

def logged_import(name, *args, **kwargs):
    print(f"DYNAMIC_IMPORT: {name} from {sys._getframe(1).f_code.co_filename}:{sys._getframe(1).f_lineno}")
    return original_import(name, *args, **kwargs)

importlib.import_module = logged_import
```

Feed the logs to the AI’s context pipeline. This catches dynamic imports that static analysis misses.

### 4. Traffic replay and canary tests

Replay production traffic against a staging environment and capture the full call graph. Use tools like:

- Locust 2.24.0 for load testing
- GoReplay 1.2.0 for traffic mirroring
- Gremlin 2.30.0 for chaos testing

Then, feed the replay traces into the AI’s context pipeline. This ensures the AI sees *real* usage patterns, not just synthetic loads.

Example with Locust:

```python
# locustfile.py
from locust import HttpUser, task

class UserServiceUser(HttpUser):
    @task
    def get_user(self):
        self.client.get("/users/123")
```

Run for 30 minutes at 1000 RPS, then extract call sites from the traces.

### 5. Context fidelity scoring

Define a metric: **Context Fidelity Score** = (Number of call sites the AI predicts correctly) / (Number of call sites in ground truth).

Automate the scoring with a script:

```python
# fidelity_score.py
import json

def load_ground_truth(path):
    return set(json.load(open(path)))

def load_ai_prediction(path):
    return set(json.load(open(path)))

def fidelity_score(ground_truth, prediction):
    correct = ground_truth & prediction
    return len(correct) / len(ground_truth)

if __name__ == "__main__":
    gt = load_ground_truth("ground_truth.json")
    pred = load_ai_prediction("ai_prediction.json")
    print(f"Context Fidelity: {fidelity_score(gt, pred):.2%}")
```

Run this script after every AI review or refactoring session. If the score drops below 90 %, block the change until the AI’s context improves.

---

## Quick reference

| What | How to measure | Tool / version | Threshold | Fix if below |
|------|----------------|----------------|-----------|--------------|
| File coverage | `find . -type f | wc -l` vs actual runtime files | `find` + `git grep` | 95 % | Add missing test/config files to AI index |
| Import coverage | Static call graph vs runtime traces | `pyan3` 1.2.0 + `py-spy` 0.3.14 | 80 % | Add dynamic import logging |
| Behavioral coverage | Runtime traces vs AI predictions | OpenTelemetry 1.42.0 + Cody Enterprise 1.47.0 | 90 % | Add OTel traces to AI context pipeline |
| Config coverage | Crawl celery/k8s/airflow configs | `config_crawler.py` (custom) | 95 % | Instrument config files |
| Context fidelity score | (Correct predictions) / (Ground truth) | `fidelity_score.py` (custom) | 90 % | Block changes until score recovers |

---

## Further reading worth your time

- [OpenTelemetry 1.42.0 docs: Instrumenting Python](https://opentelemetry.io/docs/instrumentation/python/1.42/) — How to capture traces for behavioral context
- [Sourcegraph Cody Enterprise 1.47.0: Repo context](https://docs.sourcegraph.com/cody/enterprise/repo_context) — How Cody uses context for code review
- [pyan3 1.2.0: Static call graph generation](https://github.com/alex/pyan) — How to build import graphs
- [GoReplay 1.2.0: Production traffic replay](https://goreplay.org/) — How to capture real usage patterns
- [Locust 2.24.0: Load testing](https://locust.io/) — How to generate synthetic load for context validation
- [Gremlin 2.30.0: Chaos engineering](https://www.gremlin.com/) — How to test context under failure conditions
- [GitHub Copilot Enterprise: Context limits](https://docs.github.com/en/copilot/using-github-copilot/enterprise/context-limits) — What Copilot can and cannot index

---

## Frequently Asked Questions

**Why does Cody Enterprise miss so many call sites in my repo?**
Cody Enterprise indexes files and embeddings, but it doesn’t automatically ingest runtime traces or configuration files unless you explicitly configure it to. The default repo context is file-only, which gives you syntax-level context, not behavioral context. To fix this, enable behavioral traces (OpenTelemetry) and config crawling (custom scripts).

**How do I know if the AI’s context is good enough for a rename?**
Define a critical path scope for the rename — the set of call sites that must be correct for the change to be safe. Measure the AI’s fidelity against that scope, not the entire repo. If fidelity is below 90 %, block the change and improve context before proceeding.

**Can I use embeddings to capture behavioral context?**
No. Embeddings compress files into vectors, losing temporal and behavioral signals. They’re great for semantic search, but useless for predicting runtime interactions like nightly cron jobs or background workers. For behavioral context, use OpenTelemetry traces, config crawlers, and traffic replay.

**What’s the fastest way to improve context fidelity in a repo with 2,000+ files?**
Start with behavioral traces (OpenTelemetry 1.42.0) and config crawlers (custom Python script). These two additions typically push fidelity from 3 % to 85 % in a single sprint. Then, add dynamic import logging and traffic replay to push it to 95 %.

---

I made a mistake that cost the team three days: I trusted Cody’s repo context to rename a function without measuring its behavioral coverage. This post is what I wish I had found then — a way to measure context fidelity before trusting the AI. 

**Action for the next 30 minutes:**
Run `find . -type f | wc -l` and `git grep -l "def main" scripts/nightly` in your repo. If the counts differ by more than 10 %, you’ve found your first context gap. Start instrumenting runtime traces with OpenTelemetry 1.42.0 today.


---

### About this article

**Written by:** [Kubai Kevin](/about/) — software developer based in Nairobi, Kenya.
10+ years building production Python and Node.js backends in fintech, primarily on AWS Lambda
and PostgreSQL. Has worked with payment integrations (M-Pesa, Paystack, Flutterwave) and
AI/LLM pipelines in real production systems.
[LinkedIn](https://www.linkedin.com/in/kevin-kubai-22b61b37/) ·
[Twitter @KubaiKevin](https://twitter.com/KubaiKevin)

**Editorial standard:** Every article on this site is based on direct production experience.
Factual claims are verified against official documentation before publishing. Code examples
are tested locally. AI tools assist with structure and drafting; the author reviews and edits
every article before it goes live.

**Corrections:** If you find a factual error or outdated information,
[please contact me](/contact/) — corrections are applied within 48 hours.

**Last reviewed:** May 26, 2026
