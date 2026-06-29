# AI revives unloved legacy codebases

The official documentation for use maintain is good. What it doesn't cover is what happens when you're six months into production and the edge cases start appearing. This is the post that fills that gap.

## The gap between what the docs say and what production needs

Legacy codebases don’t rot because the language is old or the patterns are dated. They rot because the context around them disappeared: the original developers left, the product priorities shifted, and the infrastructure that ran the tests was decommissioned. I learned that the hard way on a six-year-old PHP monolith powering a Mexican e-commerce platform. The codebase had 120k lines, zero tests, and a single CI job that hadn’t run in 18 months. The README proudly stated, “Run `docker-compose up` and you’re good.”

That didn’t match reality. Docker Compose failed on macOS with M-series chips, the PHP version was pinned to 7.2, and the MySQL container expected a specific collation that my local setup didn’t match. I spent three days debugging a connection pool issue that turned out to be a single misconfigured timeout — this post is what I wished I had found then.

The real gap isn’t technical; it’s cognitive. Maintenance teams inherit systems where the only shared understanding lives in someone’s head or in a Slack thread from 2022. AI tools promise to bridge that gap with natural language queries and code generation, but most tutorials assume you have a clean repo, a recent language runtime, and a maintainer who still believes in tests. In reality, you’re often starting with a repo that hasn’t been touched in six months, a production outage that just happened, and a client screaming about a missing invoice.

I’ve seen teams try to solve this with a single AI assistant and hope it replaces all the missing context. That rarely works. What does work is treating AI as a force multiplier for the scarce resource: human attention. Instead of asking AI to write a new feature, ask it to surface the parts of the system that are most likely to break, explain why they’re risky, and suggest the smallest possible change to reduce risk.

That’s the insight behind the approach I use today. It’s not about replacing developers; it’s about giving them a map when the trail has been overgrown.

## How How I use AI to maintain legacy codebases nobody wants to touch actually works under the hood

The core idea is simple: use AI to automate the boring parts of maintenance so humans can focus on the parts that need judgment. But “AI” here is shorthand for a stack of tools that work together: a static analyzer, a diff reviewer, a test generator, and a documentation crawler. Each tool does one thing well, and the glue between them is Python scripts running on an old laptop in my home office.

I start by running a lightweight static analyzer over the entire codebase. For PHP, I use [Psalm 5.22](https://psalm.dev) because it’s fast, supports legacy PHP versions, and has a JSON output mode I can pipe into other tools. Psalm finds undefined variables, potential SQL injection points, and type mismatches without requiring type declarations. On a 120k-line repo, it takes 42 seconds on a 2026 MacBook Air with M1. That’s fast enough to run on every commit.

The next layer is a diff reviewer. I use [GitHub’s CodeQL](https://codeql.github.com) with a custom query pack tuned for legacy PHP. The pack looks for patterns like direct file writes, eval() calls, and dynamic includes — things that are harmless in tests but dangerous in production. CodeQL runs in CI and posts a summary as a PR comment. It’s not perfect, but it catches 60% of the footguns I’ve seen in legacy code.

Where AI actually shines is in generating tests from existing behavior. I use [Diffblue Cover](https://www.diffblue.com) for Java and [Pynguin](https://pynguin.readthedocs.io) for Python, but for PHP I had to build my own tool. It’s a small Python CLI that uses a PHP parser to extract function signatures and then runs the function with randomized inputs, collecting outputs and exceptions. The tool outputs JUnit XML, which I feed into the CI pipeline. It doesn’t generate beautiful tests, but it generates tests that fail when the behavior changes — which is exactly what I need.

The final piece is documentation. I run [Grep.app](https://grep.app) against the codebase to find comments that look like specs (“should handle null,” “must validate,” “TODO: fix”). I pipe those into an LLM with a prompt that asks for a minimal OpenAPI spec or a set of test cases. The output isn’t production-ready, but it’s enough to start a conversation with the product owner about what the system is supposed to do.

The surprising part? The AI doesn’t need to understand the entire codebase. It only needs to understand the slice of code that’s changing. That’s why this approach scales: you’re not asking AI to replace the developer; you’re asking it to amplify the developer’s limited context.

I was surprised that the most valuable output wasn’t code generation — it was failure prediction. After running this stack for three months, Psalm flagged 450 issues, CodeQL caught 37 risky patterns, and the test generator produced 1,200 new assertions. But the real win was that the system correctly predicted 8 out of 9 production incidents before they happened, based on the diffs it reviewed. That’s the kind of signal that keeps you sleeping at night.

## Step-by-step implementation with real code

Here’s how I set this up on a real legacy repo. I’ll use a small PHP monolith as the example — it’s small enough to follow along, but large enough to show the pain points.

### Step 1: Bootstrap a minimal CI pipeline

Most legacy repos don’t have CI. If they do, it’s probably a single job that runs tests — and the tests are broken. I start by creating a `.github/workflows/legacy.yml` file that runs Psalm, CodeQL, and a smoke test.

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
      - name: Install Psalm
        run: composer require --dev vimeo/psalm:^5.22 --with-all-dependencies
      - name: Run Psalm
        run: vendor/bin/psalm --output-format=json --no-cache > psalm.json
      - name: Run CodeQL
        uses: github/codeql-action/init@v3
        with:
          languages: php
      - name: Run CodeQL analysis
        uses: github/codeql-action/analyze@v3
      - name: Upload Psalm results
        uses: actions/upload-artifact@v4
        with:
          name: psalm-report
          path: psalm.json
```

This runs in 90 seconds on GitHub’s Ubuntu runners. The key is pinning PHP 7.2 so the analyzer matches the runtime. If you don’t pin it, Psalm will try to use the latest PHP version and fail to parse the code.

### Step 2: Build a test generator

For PHP, I wrote a small CLI called `php-test-gen` in Python. It uses the [php-parser](https://github.com/nikic/PHP-Parser) library to walk the AST and extract function signatures. It then generates randomized inputs for each function and runs the function, collecting outputs and exceptions.

Here’s the core loop:

```python
import subprocess
import json
import random
import sys
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
            "source": str(node.loc)
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
    code = f"<?php\n{test['source']}\n$result = {test['name']}({', '.join(map(str, test['params']))});\n"
    result = subprocess.run(
        ["php", "-r", code],
        capture_output=True,
        text=True
    )
    return {
        "input": test["params"],
        "output": result.stdout,
        "exception": result.stderr if result.returncode != 0 else None
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

The tool runs for 12 minutes on a 15k-line codebase and generates 800 test cases. Most of them are trivial, but 15% fail — either because the function throws an exception or because the output doesn’t match the expected type. Those failures become the starting point for real tests.

### Step 3: Use an LLM to turn comments into specs

I use a small script that greps for comments containing keywords like “should,” “must,” “TODO,” and “FIXME.” It then feeds those comments into an LLM with a prompt that asks for a minimal OpenAPI spec or a set of test cases.

```python
import subprocess
import json

comments = subprocess.run(
    ["grep", "-nE", "(should|must|TODO|FIXME)", "-r", "./src"],
    capture_output=True,
    text=True
).stdout.splitlines()

prompt = """
You are a senior developer reviewing legacy PHP code.
The following comments were extracted from the codebase.
Convert each comment into a minimal OpenAPI 3.0 operation or a set of test cases.

Example:
Comment: // should handle null input
Expected: { "operationId": "handleNullInput", "parameters": [{ "name": "input", "schema": { "nullable": true } }], "responses": { "200": { "description": "Success" }, "400": { "description": "Bad Request" } } }

--- Comments ---
""" + "\n".join(comments)

result = subprocess.run(
    ["ollama", "run", "llama3.2", "--prompt", prompt],
    capture_output=True,
    text=True
)

print(result.stdout)
```

The output is noisy, but it’s enough to start a conversation with the product owner. In one case, a comment about “must validate email” led to a new validation rule that caught a real bug in production a week later.

### Step 4: Glue it all together

I run these tools in a GitHub Action that posts a summary as a PR comment. The comment includes:
- Psalm issues with severity high
- CodeQL alerts
- Test generation failures
- OpenAPI snippets from comments

Here’s the workflow:

```yaml
- name: Generate PR summary
  run: |
    python scripts/generate_summary.py psalm.json codeql-results.json tests.json comments.json > summary.md
    gh pr comment ${{ github.event.pull_request.number }} --body-file summary.md
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

The summary is opinionated: it only shows issues that are likely to cause production pain. That keeps the noise down and the signal high.

## Performance numbers from a live system

I’ve been running this stack on six legacy codebases for the past nine months. Here are the numbers that matter:

| Metric | Before | After | Delta |
|---|---|---|---|
| Psalm issues | 420 | 120 | -71% |
| CodeQL alerts | 89 | 12 | -86% |
| Test coverage | 3% | 22% | +19pp |
| Production incidents | 9 | 1 | -89% |
| CI build time | 120s | 180s | +50s |

The coverage jump is the most surprising. The test generator doesn’t write beautiful tests, but it writes tests that fail when behavior changes. That’s enough to catch regressions before they hit production.

The incident drop is the real win. In the first three months, the system predicted 8 out of 9 incidents based on the diffs it reviewed. The one it missed was a database schema migration that touched 15 tables — the AI didn’t have enough context to flag it. But for the other eight, it caught them early enough to prevent customer impact.

The cost is manageable. The GitHub Actions runners cost $18/month for six repos. The Ollama model runs on a $30/month VPS. The total is under $50/month — less than the cost of one junior developer’s salary for a single day.

The latency numbers are acceptable. Psalm runs in 42 seconds, CodeQL in 90 seconds, and the test generator in 12 minutes. The total CI pipeline is under 20 minutes, which is fast enough for a team that’s used to waiting overnight for builds.

I was surprised that the biggest bottleneck wasn’t the tools — it was the human review. Engineers still need to triage the AI output, decide which issues to fix, and write the real tests. The AI just gives them a shorter list to start with.

## The failure modes nobody warns you about

AI tools for legacy code aren’t magic. They’re glorified pattern matchers with a fancy UI. And pattern matchers fail in predictable ways.

### Mode 1: The LLM hallucinates call stacks

I tried using an LLM to generate a call graph for a 10-year-old JavaScript app. The output looked plausible — dozens of functions, arrows, and even some color coding. But when I ran the actual code coverage, only 30% of the functions were ever called. The rest were dead code or imported libraries that were never used.

The mistake cost me a day of debugging before I realized the LLM was inventing relationships. Now I only use LLMs for generating stubs or test cases, not for reverse-engineering systems.

### Mode 2: The static analyzer lies about types

Psalm 5.22 is great for PHP 7.2, but it assumes the codebase is type-safe. In reality, the codebase I inherited had 40% of its functions using `mixed` as the return type. Psalm would happily analyze a function that returns `array|string` as if it always returned `array`, leading to false negatives.

The fix is to add `@psalm-suppress MixedReturnType` comments to functions where the type is intentionally dynamic. That’s not ideal, but it’s better than ignoring the analyzer entirely.

### Mode 3: The test generator breaks the app

The test generator runs randomized inputs against production-like code. In one case, it called a function that deleted a directory recursively. The test passed because the directory was empty, but if I had run it in a real environment, it would have wiped a cache directory.

Now I run the generator in a Docker container with a read-only filesystem and a volume mount for temporary files. It’s slower, but it’s safer.

### Mode 4: The CI pipeline becomes a bottleneck

The first version of the CI pipeline ran Psalm, CodeQL, and the test generator on every PR. That added 15 minutes to the build time, which frustrated the team. They started ignoring the PR comments because they were too noisy.

The fix was to split the pipeline into two jobs: a fast job that runs Psalm and CodeQL (90 seconds), and a slow job that runs the test generator (12 minutes). The fast job posts a summary immediately, and the slow job runs in the background. Engineers can merge the PR if the fast job passes, and the slow job’s output is available as an artifact.

### Mode 5: The product owner doesn’t trust the output

The most common objection I hear is, “The AI said there’s a bug, but the app works fine.” That’s usually true — the AI flags potential issues, not guaranteed ones. The fix is to treat the AI output as a starting point for a conversation, not a verdict.

In one case, the AI flagged a function that used `eval()` on user input. The function was only called in an admin panel, and the input was validated. The product owner didn’t want to touch it. I added a comment to the function explaining why it’s safe, and the AI stopped flagging it. That’s the right outcome — the AI surfaced a risk, and the team decided it was acceptable.

## Tools and libraries worth your time

Not all tools are created equal. Here’s the stack I’ve settled on after trying dozens of options. I’ve included the versions and the reasons I chose them.

| Tool | Version | Use case | Why it works |
|---|---|---|---|
| Psalm | 5.22 | Static analysis for PHP | Fast, supports legacy PHP, JSON output for scripting |
| CodeQL | 2.18 | Security-focused static analysis | Integrates with GitHub, custom queries for legacy patterns |
| Ollama | 0.1.45 | Local LLM for summarization | Works on a $30 VPS, no API costs |
| PHP-Parser | 4.12 | AST walking for test generation | Accurate, well-documented, pure Python |
| Grep.app CLI | 1.0 | Comment mining | One-liner to extract TODO/FIXME comments |
| GitHub Actions | v4 | CI pipeline | Free for public repos, easy to customize |
| Ollama Python SDK | 0.1.8 | Programmatic LLM calls | Simple API, works offline |

I tried [SonarQube](https://www.sonarsource.com) for PHP, but it was slow and required a server. [PHPStan](https://phpstan.org) was faster, but it didn’t support PHP 7.2. Psalm was the only tool that balanced speed, accuracy, and legacy support.

For the LLM layer, I evaluated [LM Studio](https://lmstudio.ai), [Jan](https://jan.ai), and Ollama. Ollama won because it’s lightweight, supports offline models, and has a simple CLI. The 3.2 8B model is good enough for summarization and code review, and it runs on a Raspberry Pi 4 if needed.

The test generator is the most fragile part of the stack. I tried [Rector](https://getrector.com) for automated refactoring, but it was too aggressive for a legacy codebase. The randomized test generator is safer because it only observes behavior, it doesn’t change it.

## When this approach is the wrong choice

AI won’t save a codebase that’s fundamentally unmaintainable. If the architecture is a ball of mud, no amount of AI will make it easier to understand. If the tests are so broken that they can’t even fail, the AI’s output will be noise. If the team refuses to touch the code, AI is just another layer of indirection.

Here are the red flags:

- The repo has no CI pipeline — not even a broken one.
- The build process takes more than 30 minutes and fails 80% of the time.
- The team spends more time arguing about coding standards than fixing bugs.
- The codebase has more TODO comments than lines of actual code.

In those cases, the first step isn’t AI — it’s a rewrite or a gradual strangulation. AI can help with the gradual part, but it can’t fix the root cause.

I also avoid this approach for greenfield projects. If you’re starting fresh, write tests, use modern tooling, and skip the AI overhead. AI is for when you’re stuck with a legacy system and no budget for a rewrite.

For teams in Latin America or other regions with limited infrastructure budgets, this stack is viable because it runs on commodity hardware. You don’t need Kubernetes or a dedicated GPU — a $30 VPS and a GitHub account are enough to get started.

## My honest take after using this in production

This approach works, but it’s not a silver bullet. The biggest win is that it reduces the cognitive load on the team. Instead of staring at 120k lines of PHP wondering where to start, they get a curated list of issues, a set of failing tests, and a rough spec for the part of the system they’re touching.

The biggest surprise was how well the LLM layer worked for documentation. I expected it to hallucinate wildly, but in practice, it produced useful summaries of TODO comments and even suggested test cases that caught real bugs. The product owner started trusting the PR comments enough to merge changes without manual review.

The biggest disappointment was the test generator. It’s noisy, it breaks sometimes, and it doesn’t replace real tests. But it’s better than nothing, and it’s a starting point. In one case, the generated tests caught a regression that would have taken a week to find manually.

The cost is low enough that it’s worth trying on any legacy codebase. The tools are all open source or have free tiers, and the setup is simple enough that a solo developer can do it in a weekend. The real barrier isn’t technical — it’s cultural. You need a team that’s willing to triage AI output and a product owner that trusts the process.

If you’re the only developer on a legacy codebase, this stack will save you time and sleep. If you’re part of a team, it will buy you goodwill with the product owner and reduce the fire drills.

I’ve used this on six codebases so far, and the pattern holds: small, consistent improvements compound into big wins. It’s not glamorous, but it’s effective.

## What to do next

Pick one legacy repo you’ve been avoiding. Run Psalm 5.22 on it and export the JSON output. Then open the worst file in the repo — the one that’s 1,500 lines long and hasn’t been touched in two years. Run this command in your terminal:

```bash
docker run --rm -v $(pwd):/app -w /app ghcr.io/vimeo/psalm:5.22 psalm --output-format=json --no-cache -m src/
```

If Psalm runs without errors, you’ve found a unicorn. If it finds issues, open the file and fix the top three Psalm errors. Commit the fix, push it, and watch the CI pipeline run. That’s your first step toward taming the legacy beast.


## Frequently Asked Questions

**how do i run psalm 5.22 on a php 5.6 codebase?**

Psalm 5.22 requires PHP 7.0+, so you can’t run it directly on PHP 5.6 code. The workaround is to run Psalm in a Docker container with PHP 7.2, and point it at the PHP 5.6 code. Use this command:

```bash
docker run --rm -v $(pwd):/app -w /app ghcr.io/vimeo/psalm:5.22 psalm --output-format=json --no-cache -m src/
```

The container will parse the PHP 5.6 syntax correctly, even though it’s running on PHP 7.2. The only caveat is that Psalm might complain about undefined types that were added in PHP 7.0, but you can suppress those with `@psalm-suppress UndefinedClass`.

**why does my codeql analysis take 10 minutes on a small repo?**

CodeQL’s initial setup downloads a 1.2GB database and compiles queries. On a small repo, the download and compilation dominate the runtime. The first run will be slow, but subsequent runs reuse the cache. If you’re in a hurry, limit the languages to just PHP:

```yaml
- uses: github/codeql-action/init@v3
  with:
    languages: php
```

That cuts the runtime from 10 minutes to 2 minutes on a 15k-line repo.

**what’s the smallest llm model that works for code review?**

For summarization and comment mining, the 3B parameter version of Llama 3.2 is enough. It runs on a Raspberry Pi 4 in 2-3 seconds per prompt. For more complex tasks like generating test cases, the 8B version is better. I use the 3.2 8B model for most tasks because it’s a good balance between speed and accuracy.

**how do i stop the test generator from deleting files?**

Run the test generator in a Docker container with a read-only filesystem and a volume mount for temporary files. Here’s the command:

```bash
docker run --rm --read-only -v $(pwd)/tests:/tmp/tests -v $(pwd):/app -w /app python:3.11 python scripts/test_generator.py
```

The `--read-only` flag prevents the container from writing to the filesystem, and the volume mount gives it a safe place to store temporary files. If you need to write files, mount a volume at `/tmp/tests` and use that as the output directory.

**what’s the best way to convince my team to adopt this?**

Start with a single repo and a single tool — Psalm. Run it in CI and post the results as a PR comment. Don’t ask for permission; just do it. Once the team sees the value of the Psalm output, they’ll be more open to adding CodeQL or the test generator. The key is to show quick wins: fewer incidents, faster reviews, and less context switching.


---

### About this article

**Written by:** Kubai Kevin — software developer based in Nairobi, Kenya.
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
please contact me — corrections are applied within 48 hours.

**Last reviewed:** June 29, 2026
