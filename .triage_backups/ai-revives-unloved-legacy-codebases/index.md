# AI revives unloved legacy codebases

Legacy codebases rarely rot because the language is old or the patterns are dated. They rot because the context around them disappears: the original developers leave, product priorities shift, and the infrastructure that ran the tests gets decommissioned. What remains is a repo with a README that says "run `docker-compose up` and you're good," a pinned runtime nobody has installed locally, and a CI job that last ran eighteen months ago.

The gap is not primarily technical. It is cognitive. Maintenance teams inherit systems where the only shared understanding lives in someone's head or in a chat thread from two years ago. AI tools are often pitched as a way to bridge that gap, but most tutorials assume a clean repo, a recent runtime, and a maintainer who still believes in tests. The realistic starting point is a repo nobody has touched in six months, an outage that just happened, and a stakeholder asking about a missing invoice.

Treating a single AI assistant as a replacement for all that missing context rarely works. What works better is treating AI as a force multiplier for the genuinely scarce resource: human attention. Instead of asking a model to write a new feature, ask tooling to surface the parts of the system most likely to break, explain why they are risky, and suggest the smallest change that reduces risk. The goal is not to replace developers. It is to give them a map when the trail has been overgrown.

## What the stack actually consists of

"AI" in this context is shorthand for several tools that each do one thing and hand off to the next:

- A **static analyzer** that finds type errors, undefined variables, and risky patterns without requiring the code to be rewritten.
- A **security-focused analyzer** that looks for injection sinks, unsafe deserialization, and dynamic includes.
- A **characterization test generator** that observes existing behavior and records it, so future changes that alter behavior fail loudly.
- A **documentation extractor** that turns comments, TODOs, and function signatures into a rough spec a product owner can react to.

The glue between these is ordinary scripting. None of the individual pieces is novel; the value comes from running them together on every change and routing the output to a human who can triage it.

A key property makes this tractable: the tooling does not need to understand the entire codebase. It only needs to understand the slice of code that is changing. That is why the approach scales to repos that are far too large to reason about end to end.

## Step 1: Bootstrap a minimal CI pipeline

Most legacy repos either have no CI or have a single job whose tests are already broken. The first useful step is a pipeline that runs analysis and uploads results as artifacts, without gating merges yet.

```yaml
name: Legacy Maintenance

on:
  push:
    branches: [main]
  pull_request:

jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up PHP
        uses: shivammathur/setup-php@v2
        with:
          php-version: '7.2'
          coverage: none
      - name: Install static analyzer
        run: composer require --dev vimeo/psalm:^5.22 --with-all-dependencies
      - name: Run static analysis
        run: vendor/bin/psalm --output-format=json --no-cache > psalm.json
      - name: Initialize CodeQL
        uses: github/codeql-action/init@v3
        with:
          languages: php
      - name: Run CodeQL analysis
        uses: github/codeql-action/analyze@v3
      - name: Upload analysis results
        uses: actions/upload-artifact@v4
        with:
          name: psalm-report
          path: psalm.json
```

The important detail is pinning the runtime to match production. If the analyzer runs under a newer PHP than the application, it will parse syntax the application cannot execute and miss the errors that actually matter. This is also why a static analyzer that requires a modern runtime is often unusable on a legacy PHP codebase: the analyzer's own requirements become the blocker.

## Step 2: Generate characterization tests

The most valuable tests on a legacy system are not unit tests of intended behavior. They are characterization tests: tests that record what the code currently does, so that any change to that behavior is visible. Generated tests are rarely elegant, but they fail when behavior changes, which is exactly the signal a maintenance team needs.

A test generator walks the AST, extracts function signatures, generates inputs, runs each function in isolation, and records the output or exception. The core loop looks like this:

```python
import subprocess
import random
from pathlib import Path
from php_parser import Parser, NodeVisitor


class TestGenerator(NodeVisitor):
    def __init__(self):
        self.tests = []

    def visit_Function(self, node):
        if node.name.name == "__construct":
            return
        params = [self.generate_param(p) for p in node.params.params]
        self.tests.append({
            "name": node.name.name,
            "params": params,
            "source": str(node.loc),
        })

    def generate_param(self, param):
        param_type = param.type.name if param.type else "mixed"
        if param_type == "int":
            return random.randint(-1000, 1000)
        elif param_type == "string":
            return "test_" + str(random.randint(0, 999))
        elif param_type == "array":
            return []
        else:
            return None


def run_test(test):
    code = (
        "<?php\n"
        f"$result = {test['name']}("
        + ", ".join(repr(p) for p in test["params"])
        + ");\n"
    )
    result = subprocess.run(
        ["php", "-r", code],
        capture_output=True,
        text=True,
    )
    return {
        "input": test["params"],
        "output": result.stdout,
        "exception": result.stderr if result.returncode != 0 else None,
    }


if __name__ == "__main__":
    repo_path = Path(".")
    parser = Parser(repo_path)
    visitor = TestGenerator()
    parser.walk(visitor)
    for test in visitor.tests:
        result = run_test(test)
        if result["exception"]:
            print(f"FAIL: {test['name']} with {test['params']}")
            print(result["exception"])
```

Two things matter here. First, `repr()` is used when building the call so that strings are quoted correctly; naive string interpolation produces invalid PHP for any string parameter. Second, the generator runs application code, so it must never touch a real database, filesystem, or network. The failure-mode section below covers containment.

In practice, most generated cases are trivial, a minority fail because the function throws or returns an unexpected type, and those failures become the seed for real tests written by a human.

## Step 3: Turn comments into candidate specs

Legacy code is full of comments that encode requirements nobody wrote down: "should handle null input," "must validate email," "TODO: race condition here." Extracting them and asking a local model to restate each as a test case or an API fragment produces a rough spec that is useful for a conversation, not for direct execution.

```python
import subprocess

comments = subprocess.run(
    ["grep", "-rnE", "(should|must|TODO|FIXME)", "./src"],
    capture_output=True,
    text=True,
).stdout.splitlines()

prompt = """
You are a senior developer reviewing legacy PHP code.
The following comments were extracted from the codebase.
Convert each comment into a minimal OpenAPI 3.0 operation or a set of test cases.
If a comment is too vague to convert, say so instead of guessing.

--- Comments ---
""" + "\n".join(comments)

result = subprocess.run(
    ["ollama", "run", "llama3.2", prompt],
    capture_output=True,
    text=True,
)

print(result.stdout)
```

The output is noisy and should be treated as a draft. Its value is that it gives a product owner something concrete to correct, which is faster than asking them to describe the system from memory. The instruction to say "too vague" rather than guess is deliberate: models will otherwise invent a plausible requirement that nobody ever asked for.

## Step 4: Glue the outputs into a single summary

The point of the pipeline is to produce one short, opinionated summary per pull request rather than four separate noisy reports. A script reads each tool's output, filters to high-signal items, and posts a comment.

```yaml
- name: Generate PR summary
  run: |
    python scripts/generate_summary.py \
      psalm.json codeql-results.json tests.json comments.json > summary.md
    gh pr comment ${{ github.event.pull_request.number }} --body-file summary.md
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

The filtering rules matter more than the formatting. A useful default is to show only analyzer findings on lines the pull request actually changed, plus any characterization test that flipped from pass to fail. Everything else stays in the artifact for later.

## How to measure whether this is working

Claims about AI-assisted maintenance are easy to make and hard to verify. Rather than trusting a summary table, instrument the pipeline and compare against a baseline period.

What to record, per week:

- **Analyzer findings on changed lines.** Count only findings that touch code in the diff. Total repo findings will fall slowly and is a misleading metric.
- **Characterization tests that flipped.** A test going from pass to fail on an unchanged function is a signal worth investigating; a test that never runs is not.
- **Time from PR open to first human review.** This measures whether the summary is helping or adding noise.
- **Incidents by category.** Classify each production incident by the file or subsystem involved, then check whether any analyzer finding had already flagged that location. This is the only honest way to evaluate "prediction" claims.
- **CI wall-clock time for the fast job.** If the fast job exceeds a few minutes, developers will stop reading its output.

A concrete comparison method: pick a four-week baseline before enabling the pipeline, record the metrics above, then enable the pipeline and record the same metrics for four weeks. Compare medians, not totals, and note any confounders such as a release freeze or a staffing change. A pipeline that shortens review time but does not reduce incidents is still useful; a pipeline that does neither is overhead.

## Failure modes and how to contain them

### Hallucinated architecture

Asked to reverse-engineer a call graph from a partial codebase, a language model will produce a plausible diagram containing functions that do not exist and relationships that were never in the code. The output looks authoritative because it is well formatted.

Containment: use models for narrow, verifiable tasks such as drafting a test case or restating a comment. Do not use them to reconstruct system architecture. If a generated diagram is useful, verify each edge against a real call-site search before acting on it.

### Analyzer false confidence

Static analyzers infer types from docblocks and usage. A function documented as returning `array` but returning `array|string` in practice will be analyzed as if it always returns an array, producing false negatives on the union case. The tool is not lying; it is reasoning from the annotations it was given.

Containment: treat a clean analyzer run as "no findings under these assumptions," not "no bugs." Where a type is intentionally dynamic, add a suppression with a comment explaining why, so the suppression is itself documented.

### Destructive generated tests

Randomized inputs against real code can call functions that delete files, drop tables, or send email. A test that passes because the target directory happened to be empty is not a safe test.

Containment: run the generator in a container with a read-only root filesystem and an explicit writable mount for temporary output, and point it at a disposable database.

```bash
docker run --rm --read-only \
  -v "$(pwd)/tests:/tmp/tests" \
  -v "$(pwd):/app" \
  -w /app \
  python:3.11 python scripts/test_generator.py
```

Even with `--read-only`, any network access should be blocked, since a function that posts to a webhook will otherwise fire during test generation.

### The pipeline becomes the bottleneck

Running every analyzer on every pull request adds minutes to each build. When the added time is large, developers start ignoring the output, which defeats the purpose.

Containment: split into a fast job (static analysis, a couple of minutes) that posts a summary immediately, and a slow job (test generation) that runs asynchronously and publishes results as an artifact. Allow merges on the fast job alone.

### Stakeholders distrust the output

The most common objection to an AI-flagged issue is "the app works fine." That is usually true: the tool flags potential problems, not confirmed ones. A function calling `eval()` on admin-only, validated input is a real risk that a team may reasonably accept.

Containment: frame findings as questions, not verdicts. "This function evaluates user input; is the input validated upstream?" is actionable. "Critical vulnerability" is not, and it erodes trust when it turns out to be a false positive.

## Choosing tools

The categories matter more than specific products, because availability and version support change quickly. For each category, the selection criteria are:

- **Static analysis for legacy PHP.** The analyzer must run under a PHP version the codebase can actually parse, and must emit machine-readable output. Verify this before committing: an analyzer that requires a modern runtime cannot analyze code pinned to an old one.
- **Security analysis.** Look for one that integrates with the existing CI provider, supports the application's language, and allows custom queries for patterns specific to the codebase, such as a homegrown template engine.
- **Local model runtime.** A local runtime avoids per-token costs and keeps source code off third-party servers, which matters for regulated codebases. The tradeoff is that local models are weaker than hosted frontier models on complex reasoning.
- **AST parsing library.** Choose one that is actively maintained and handles the exact language version in the repo. Parser correctness determines whether generated tests are meaningful.
- **CI provider.** Any provider with artifact upload and pull-request comments will do.

A small model (roughly 3B parameters) is adequate for summarization and comment extraction. A mid-size model (roughly 8B) handles drafting test cases better. Neither is reliable for architectural reasoning, and neither should be given write access to production systems.

## When this approach is the wrong choice

AI tooling cannot rescue a codebase that is fundamentally unmaintainable, and adding it to one mostly adds a layer of indirection. Warning signs:

- The build process takes more than half an hour and fails most of the time. Fix the build first; no analyzer output is trustworthy until the code compiles reproducibly.
- There is no CI at all and no appetite to add one. The pipeline is the delivery mechanism for every finding; without it, nothing reaches a human.
- The team spends more time debating coding standards than fixing bugs. The bottleneck is social, not informational.
- The architecture is a ball of mud with no module boundaries. Analysis findings will be correct and useless, because every change touches everything.
- The team has already decided the system must be replaced. In that case, effort is better spent on a strangler-fig migration, where AI tooling can help generate the seam tests but cannot decide the boundaries.

The approach is also a poor fit when the codebase is small and healthy. If a repo has tests, a working build, and a maintainer who understands it, the overhead of an analysis pipeline exceeds the benefit.

For teams with limited infrastructure budgets, the stack is viable on commodity hardware: a small VPS running a local model runtime and a hosted CI account are sufficient. No GPU cluster is required for the summarization and comment-extraction tasks described here.

## A worked example of triage

Suppose the analyzer reports forty findings on a pull request that changed three files. A reasonable triage sequence:

1. **Filter to changed lines.** Of the forty, perhaps six touch the diff. The other thirty-four are pre-existing and belong in a separate backlog.
2. **Classify the six.** Two are type mismatches in a function whose return type is genuinely dynamic — suppress with a documented reason. One is an unused variable — fix in the same commit. Two are possible null dereferences on a value that comes from a database column defined `NOT NULL` — verify the schema, then either fix or suppress. One is a call to a function that writes a file whose path comes from user input — this is the one worth a conversation.
3. **Write one real test for the risky case.** Not a generated test: a hand-written test that asserts the path is validated. This test survives refactoring, unlike the generated characterization tests.
4. **Record the decision.** If the team accepts the risk, write down why and where. An accepted risk with a written rationale is a decision; the same risk unrecorded is a future incident.

This sequence takes perhaps twenty minutes for six findings. The value is not that the analyzer found six things; it is that it turned an open-ended question ("is this change safe?") into a bounded list.

## A note on cost and latency

Costs depend entirely on choices: a hosted CI provider's free tier for public repos, a local model runtime on existing hardware, and open-source analyzers can bring marginal cost close to zero. Hosted model APIs and larger runners add up quickly. Before adopting, estimate the per-pull-request cost of each component and multiply by the team's merge frequency.

The same applies to latency. Static analysis on a mid-sized codebase typically completes in under a minute; security analysis is slower on first run because it downloads and compiles query packs, then faster on subsequent runs with a warm cache. Test generation is the slowest component and should not block merges.

## What to do in the next 30 minutes

Pick one legacy repository you have been avoiding. Run a static analyzer against it in a container so you do not have to install anything locally, and capture the machine-readable output:

```bash
docker run --rm -v "$(pwd):/app" -w /app \
  ghcr.io/vimeo/psalm:5.22 \
  psalm --output-format=json --no-cache -m src/ > psalm.json
```

Then count the findings by file and open the file with the most. Fix the top three findings that are unambiguous — unused variables, undefined variables, obviously wrong types — and commit. Do not attempt to fix everything; the goal of the first session is to prove that the analyzer runs, that its output is readable, and that at least one finding was real. Everything else in this article builds on that first result.
===
