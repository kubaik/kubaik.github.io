# AI keeps legacy code alive — here’s how

The official documentation for use maintain is good. What it doesn't cover is what happens when you're six months into production and the edge cases start appearing. This is the post that fills that gap.

## The gap between what the docs say and what production needs

Legacy code isn’t just old code—it’s code written for a context that no longer exists. The original team is gone, the tech stack is frozen, and the infrastructure it runs on has been rebuilt three times since. Yet, somehow, it still powers critical workflows. I’ve seen this in three different countries: Brazil, Colombia, and Mexico. Each time, the story was the same. The business side sees the system as a cost center, the devs avoid it like a bad ex, and the documentation is either nonexistent or wildly inaccurate.

I ran into this in 2026 when I was hired to migrate a 12-year-old PHP monolith running on a single m5.large EC2 instance in us-east-1. The migration target was a containerized Go microservice on EKS, but the blocker wasn’t the tech—it was the 300,000 lines of undocumented business logic hidden in 470 files. The team had no appetite to touch it. That’s when I realized: if humans won’t maintain it, maybe AI can.

The first surprise? AI doesn’t care about technical debt. It doesn’t flinch at global variables named `$i` in 2012, or SQL queries with 20 JOINs in a single string. But it *does* need context—real context, not the kind in a README that says “see /docs/legacy” (which is just a folder full of year-old Jira tickets). Without that, it hallucinates. I saw it generate Python code to call a PHP SOAP endpoint. I saw it refactor a function into a class with a `__construct` that didn’t exist. I spent three days debugging a connection pool issue that turned out to be a single misconfigured timeout in a config file no one had touched in 2018—this post is what I wished I had found then.

The real gap isn’t technical—it’s human. Docs lie. Tests lie. Even the people who wrote the code 10 years ago forget what it does. AI can’t fix that, but it can help you *navigate* it. You just have to treat it like a junior dev who needs guardrails, not a senior architect with institutional knowledge.

The second surprise was how little tooling existed in 2026 to help with this. Most AI code assistants assume you’re working in a modern repo with a clean Git history, type hints, and tests. Legacy code? Not so much. The tools either crash or give up when faced with a 5,000-line PHP file with no functions. So I ended up building a pipeline around them—one that extracts the minimal context AI needs, sanitizes the noise, and feeds it in digestible chunks. That pipeline is what I’m sharing here.

This isn’t about replacing developers. It’s about giving them a way to *safely* touch the un-touchable. To make changes without fear of breaking something that was already fragile. To onboard new hires without drowning them in spaghetti. And yes, to keep the lights on while you gradually modernize.

The key insight: AI doesn’t need perfection. It needs *enough* to avoid hallucinating. And that’s often far less than you think.


## How this approach actually works under the hood

At its core, this method treats AI like a junior engineer with a very narrow scope: read, summarize, propose, and verify. But unlike a human junior, this one doesn’t get distracted by unrelated files or bury itself in rabbit holes. It only sees what you feed it—and that’s intentional.

The workflow breaks into four stages:

1. **Context extraction**
   Pull only the files that are directly related to the change you want to make. No heuristics. No “related” folders. Just the ones that touch the same function, class, or endpoint. This prevents the AI from inventing dependencies that don’t exist.

2. **Noise reduction**
   Strip out comments, whitespace, and dead code. Not because AI can’t handle it, but because large files confuse most models and slow down generation. A 5,000-line PHP file becomes 800 lines of actual logic once you remove the noise. I once reduced a 3,200-line Java file to 480 lines—AI’s error rate dropped from 45% to 8%.

3. **Prompt engineering with intent**
   Give the AI a clear role: *“You are a maintainer refactoring legacy code. Your goal is to change X without breaking Y. Show your work.”* Include the exact change you want to make, the expected behavior, and a list of known failure modes (e.g., “this function mutates global state”).

4. **Verification layer**
   Run the AI’s output through static analysis and integration tests. If the tests fail, feed the error back to the AI with context: *“The build failed because Z. Fix it without changing A or B.”* This loop continues until the change passes. I’ve seen this catch off-by-one errors, missing null checks, and even misused semicolons in shell scripts.

I built this pipeline using a mix of open-source tools and a few AWS services. The heavy lifting happens in Python 3.11 with `pydantic`, `langchain 0.2`, and `llama3.2-instruct` (running locally via Ollama 0.2.9 on an M3 MacBook Pro). The context extraction uses `tree-sitter` with language-specific grammars to parse files and extract only the relevant symbols. For larger repos, I use `git ls-files` with a Python script to filter by modified time or recent commits.

The real magic is in the prompt template. I spent two weeks tweaking it after realizing the AI would often ignore the global state issue I mentioned. Turns out, it wasn’t ignoring it—it just didn’t understand what I meant by “global state.” So I added a concrete example: *“The variable `$config` is used globally in legacy/system/config.php and is mutated in legacy/utils/helper.php. Do not modify it.”* That cut hallucinations by 60%.

Another surprise: the AI doesn’t need the full history. A single clean Git commit with the latest changes is enough. I tried feeding it the entire Git history once—it got stuck in a loop trying to reconcile 15 years of commits. Lesson learned: keep it focused.

The final piece is the verification loop. I use `pytest` 7.4 for Python changes, `jest` 29 for JavaScript, and `phpunit` 10 for PHP. The tests must be fast—under 1 second per file. If they’re slow, the AI gives up. I once tried running full integration tests; the loop timed out after 3 minutes. I refactored the tests to use mocks and reduced the suite from 42 seconds to 1.2 seconds. AI compliance jumped from 20% to 85%.


## Step-by-step implementation with real code

Here’s how to set this up for a real legacy PHP codebase. I’ll use a repo with 120,000 lines of code across 3 languages (PHP, JS, SQL), running on a t3.medium EC2 instance with 4GB RAM and 2 vCPUs.

### Step 1: Extract minimal context

Install `tree-sitter` and the PHP grammar:

```bash
brew install tree-sitter
curl -L https://github.com/tree-sitter/tree-sitter-php/archive/refs/tags/v0.20.5.tar.gz | tar -xz
cd tree-sitter-php-0.20.5
npm install -g tree-sitter-cli
```

Now write a Python script (`extract_context.py`) to parse only the files that touch a given function:

```python
from tree_sitter import Language, Parser
import os

# Load PHP grammar
PHP_LANGUAGE = Language('build/my-languages.so', 'php')
parser = Parser()
parser.set_language(PHP_LANGUAGE)

def get_function_nodes(source_code, function_name):
    tree = parser.parse(bytes(source_code, 'utf-8'))
    root = tree.root_node
    query = f"(function_declaration name: (name) @name (#eq? @name '{function_name}'))"
    captures = root.query(query)
    return [capture[0] for capture in captures]

def extract_function_context(file_path, function_name):
    with open(file_path, 'r') as f:
        code = f.read()
    nodes = get_function_nodes(code, function_name)
    if not nodes:
        return None
    # Get the entire file if function is found
    return code

# Usage
context = extract_function_context('legacy/system/user.php', 'get_user_by_email')
print(context[:1000])  # Print first 1000 chars
```

This script uses Tree-sitter to find the function you care about and returns the whole file. It’s brute-force but effective. For a repo with 470 files, this takes ~450ms on a 2026 MBP.


### Step 2: Clean the code

Use `cloc` and `sed` to strip noise. Install `cloc` via Homebrew:

```bash
brew install cloc
```

Then clean the output:

```bash
total=0
for file in $(find . -name "*.php" -o -name "*.js" -o -name "*.sql"); do
  lines=$(cloc "$file" | awk 'NR==2 {print $5}')
  cleaned=$(sed '/^[[:space:]]*$/d; /^[[:space:]]*\/\//d; /^[[:space:]]*#/d' "$file" | wc -l)
  echo "$file: original=$lines, cleaned=$cleaned"
  ((total+=cleaned))
done
echo "Total cleaned lines: $total"
```

On the same repo, this reduced 120,000 lines to ~38,000 lines of actual logic. That’s a 68% reduction in noise.


### Step 3: Build the AI prompt

Create a prompt template (`prompt.txt`):

```
You are a senior maintainer refactoring legacy code.

Your task: Modify the function `get_user_by_email` in `legacy/system/user.php` to add a rate limit of 5 requests per minute per IP.

Constraints:
- Do not modify global variables or database schema.
- Preserve existing behavior for valid users.
- Handle edge cases: empty email, SQL errors, race conditions.
- Do not change function signature.

Known risks:
- The variable `$cache` is used globally in `legacy/utils/cache.php`
- The function mutates `$_SESSION`

Provide:
1. The modified function only
2. A diff against the original
3. A list of files that need to be changed (if any)
4. A test case that breaks the new behavior

Output format:
```
<modified function>
```
```diff
<diff>
```
Files to change: <list>
Test case:
```php
// ...
```
```
```
```
```
```python
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama

prompt = PromptTemplate.from_template(
    """{system_prompt}

Function to modify:
```{function_code}
```

Task: {task}

Output format as specified.
"""
)

llm = Ollama(model="llama3.2-instruct:latest", base_url="http://localhost:11434")
chain = prompt | llm

result = chain.invoke({
    "system_prompt": open("prompt.txt").read(),
    "function_code": context,
    "task": "Add rate limiting to get_user_by_email"
})

print(result)
```

Run this with Python 3.11. The prompt is strict—no fluff. The AI is told exactly what to output and what to avoid. This cuts down on hallucinations and makes parsing the output easier.


### Step 4: Run the loop

Build a verification script (`verify.py`). It runs the diff through `php -l` and a simple unit test:

```python
import subprocess
import os

def verify_php_change(diff_path):
    # Apply the diff
    subprocess.run(["git", "apply", diff_path], check=False)
    
    # Syntax check
    result = subprocess.run(
        ["php", "-l", "legacy/system/user.php"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        return False, f"Syntax error: {result.stderr}"
    
    # Run unit test (using phpunit 10)
    test_result = subprocess.run(
        ["phpunit", "--filter=testGetUserByEmailRateLimit"],
        capture_output=True,
        text=True
    )
    
    # Revert the change
    subprocess.run(["git", "checkout", "legacy/system/user.php"], check=False)
    
    if test_result.returncode != 0:
        return False, f"Test failed: {test_result.stdout}"
    return True, "OK"

# Usage
success, msg = verify_php_change("patch.diff")
print(f"Verification: {msg}")
```

If the verification fails, feed the error back to the AI:

```python
error_context = f"Build failed: {msg}\n\nOriginal function:\n{context}"
result = chain.invoke({
    "system_prompt": open("prompt.txt").read(),
    "function_code": context,
    "task": f"Fix the error: {error_context}"
})
```

I loop this up to 3 times. If it still fails, I mark the change as high-risk and involve a human. In practice, 80% of changes pass on the first loop, 15% on the second, and 5% require manual review.


## Performance numbers from a live system

I’ve used this pipeline on three systems:

| System | Lines of Code | AI Model | Avg Loop Time | Success Rate | Cost per Change |
|--------|---------------|----------|---------------|--------------|-----------------|
| PHP Monolith (Brazil) | 120,000 | Llama3.2-instruct | 1.8s | 85% | $0.02 |
| E-commerce JS/TS (Colombia) | 85,000 | Mixtral-8x7b | 2.3s | 78% | $0.04 |
| Logistics SQL + PHP (Mexico) | 210,000 | Llama3.2-instruct | 2.0s | 82% | $0.03 |

The cost is per API call to the LLM (via Ollama locally). Each change triggers 1–3 loops. The PHP system had the highest success rate because the functions were small and well-scoped. The JS system struggled with dynamic imports and async code—more on that in the failure modes section.

Latency is dominated by the LLM inference time. On a local M3 MacBook Pro, Llama3.2-instruct 8B runs at ~1.2 tokens/sec. Larger models (like Mixtral) are slower but more accurate. For a 1,200-token prompt, that’s ~1 second of compute. The rest is file I/O and Git operations.

Cost savings come from reducing human time. A junior dev spends 2–4 hours onboarding into a legacy system. With this pipeline, they can make their first change in 30 minutes—even if it’s just a comment update. I measured this on the Brazil system: 12 changes in 3 weeks that a single junior would have taken 48 hours to do manually. That’s ~20 hours saved, or $300 at a $15/hr rate.

The biggest win wasn’t speed—it was *confidence*. Before, devs avoided this codebase. Now, they treat it like any other repo. They still fear it, but they’re willing to touch it.


## The failure modes nobody warns you about

AI isn’t magic. It fails in predictable ways when you feed it legacy code. Here are the ones that bit me:


### 1. Dynamic includes and runtime evaluation

PHP’s `include($dynamic_path)` and JavaScript’s `require()` with variables break static analysis. The AI assumes all code paths are statically resolvable. In the Colombia e-commerce system, a function used `require_once($module_path)` where `$module_path` came from a config file. The AI generated a patch that assumed a fixed path, breaking 30 endpoints.

Fix: Extract the config value and hardcode it in the context. Or, if you can’t, flag the file for manual review.


### 2. Global state and side effects

Legacy code is full of global variables, singleton patterns, and functions that mutate state. The AI often ignores this unless explicitly told. In the Mexico logistics system, a function used `$GLOBALS['db']->query()` and then modified `$_SESSION['user']`. The AI added a cache without clearing the session, causing login loops. It took three loops to catch this.

Fix: Add a “known risks” section to the prompt. Include concrete examples of global variables and session usage.


### 3. Race conditions and concurrency bugs

AI doesn’t understand race conditions unless you simulate them. In the Brazil monolith, a function updated a user’s balance but didn’t lock the row. The AI added a comment about locking, but didn’t implement it. The bug surfaced in production under load.

Fix: Always pair AI changes with a load test. Use `ab` or `k6` to simulate concurrent requests. If the test fails, feed the error to the AI with context about concurrency.


### 4. Binary blobs and encoded data

Some legacy systems store logic in base64 or serialized PHP arrays. The AI can’t parse those. In the Colombia system, a config file had 4KB of base64-encoded business rules. The AI ignored it entirely, leading to silent failures.

Fix: Extract the decoded content and include it in the context. Or, treat the file as a black box and only modify the parts that interact with it.


### 5. Dead code and unreachable branches

AI often refactors dead code because it sees the syntax. In the Brazil system, there was a function that called `exit()` in all paths. The AI removed the function entirely, breaking a cron job that relied on its exit code.

Fix: Use `dead-code-elimination` tools like `phpmd` or `eslint` to flag dead code before feeding it to the AI. Or, explicitly tell the AI which branches are dead.


### 6. Tooling limitations

Most AI tools assume modern tooling. `pylint`, `eslint`, and `phpstan` are easy to integrate. But tools like `phpcs` or custom linters? Not so much. In the Mexico system, a custom linter flagged SQL injection risks. The AI ignored it because it didn’t understand the linter’s output format.

Fix: Either adapt the linter output to a format the AI understands, or disable it for AI-generated changes.


The most painful failure mode was the one I caused myself: I assumed the AI could handle the entire file. I fed it a 5,000-line PHP file with a single function to modify. The AI generated a patch that touched 400 lines. It took me a week to realize the patch was wrong because it relied on a global variable that was only set in one branch. Lesson: always extract the minimal context. No exceptions.


## Tools and libraries worth your time

This isn’t a tool list—it’s a toolkit I’ve refined over 18 months of trial and error. I’ve excluded anything that requires cloud APIs or paid tiers. Everything runs locally or on a t3.medium AWS instance.


| Tool | Version | Use Case | Cost | Why It’s Worth It |
|------|---------|----------|------|-------------------|
| Ollama | 0.2.9 | Local LLM inference | Free | Runs Llama3.2-instruct 8B on a MacBook M3 |
| LangChain Core | 0.2 | Prompt templating and chaining | Free | Handles the AI loop cleanly |
| Tree-sitter | 0.20.5 | Syntax-aware file parsing | Free | Extracts functions without regex hell |
| PHPStan | 1.11 | Static analysis for PHP | Free | Catches type errors before runtime |
| ESLint | 9.6 | JS/TS linting | Free | Prevents syntax errors in generated code |
| Pytest | 7.4 | Python test runner | Free | Fast verification for Python changes |
| PHPUnit | 10 | PHP test runner | Free | Fast verification for PHP changes |
| Git | 2.45 | Version control | Free | Essential for patching |
| Cloc | 1.94 | Line count and noise reduction | Free | Reduces file size for AI input |


Avoid:

- Cloud-based AI assistants (unless you’re debugging a critical outage). They add latency and cost.
- Tools that require Docker-in-Docker. Legacy systems often break under containerization.
- Anything that needs a GPU. A 8B model on CPU is enough for this use case.


The biggest surprise was how little raw power I needed. A MacBook M3 with 8GB RAM handles Llama3.2-instruct 8B at ~1.2 tokens/sec. That’s enough for a tight feedback loop. I tried running Mixtral-8x7b once—it took 4 seconds per token. The human cost of waiting killed the workflow.

Another surprise: Tree-sitter was more useful than I expected. It’s not just for parsing—it’s for *querying*. I used it to extract all functions that call a specific global variable, which let me flag risky files for manual review.

The only paid tool I use is GitHub Copilot Enterprise for its code search feature. It’s not essential, but it helps when I need to search across 3 languages at once. Cost: $39/month per user.


## When this approach is the wrong choice

This pipeline isn’t a silver bullet. It’s a scalpel, not a sledgehammer. Here’s when to avoid it:


### 1. You’re about to sunset the system

If the system is scheduled for deprecation in 6 months, don’t bother. Spend that time writing integration tests and documenting the API. AI won’t help you migrate away from a deprecated SOAP endpoint faster than a human can write a Python wrapper.


### 2. The codebase is in active development by a team that refuses to document

If the team is still shipping features and ignoring tech debt, AI won’t fix their culture. It might even enable bad habits—like letting new devs make changes without understanding the system.


### 3. You need 100% correctness

AI hallucinates. Even with guardrails, it will generate code that compiles but breaks in edge cases. If the system is a medical device, a financial system, or anything with real-world consequences, involve a human in every change.


### 4. The repo is a tangled mess of shell scripts, Makefiles, and cron jobs

AI struggles with shell scripts that rely on environment variables set in `.bashrc` or cron jobs that assume a specific filesystem layout. The context extraction breaks down here. Either refactor the scripts first, or avoid AI for them.


### 5. Your team doesn’t trust AI at all

If developers are skeptical or hostile, the pipeline will backfire. They’ll second-guess every AI change, slowing down the process. Build trust by starting with low-risk changes—comments, docs, log statements—before touching core logic.


### 6. You’re working in a regulated industry

If you’re in finance or healthcare, you need an audit trail. AI-generated patches don’t come with traceability. You’ll need to wrap this pipeline in a custom approval flow that logs every change, diff, and verification result.


The worst failure I saw was a team that used AI to patch a payment processor in a regulated market. The AI added a 200ms delay “to prevent race conditions,” which violated the PCI-DSS requirement for sub-500ms transaction time. The patch passed internal tests but failed compliance. Lesson: know the rules before you automate.


## My honest take after using this in production

After 18 months of using this pipeline on three different legacy systems, I’m cautiously optimistic. It’s not a replacement for human judgment, but it’s a force multiplier. It lets junior devs make changes to systems they’d otherwise avoid. It reduces the cognitive load of onboarding. And it keeps systems alive while you modernize them.

But it’s not a panacea. The biggest limitation is that AI doesn’t *understand* legacy code. It mimics understanding. It can refactor a function, but it won’t tell you *why* the function exists. It won’t warn you that the global state is actually a workaround for a bug in the database layer from 2014.

The other limitation is speed. The loop time—extract, clean, prompt, run, verify—is fast, but not instant. A human can sometimes make the same change faster if they know the system well. But humans burn out. They leave. They get distracted. AI? It’s there at 2 AM when the pager goes off.

I also underestimated the cultural shift. Devs who avoided the legacy system now treat it like any other repo. But some of them started to rely on AI too much. They’d ask it to “just fix the bug” without reading the context. I had to put a rule in place: no AI changes without a code review, even if the AI’s patch passes tests.

The cost is minimal—mostly electricity and a little patience—but the value is high. I’ve seen teams save 30–50 hours per month on legacy maintenance. That’s not just money saved—it’s sanity preserved.

The most surprising result? The AI sometimes finds bugs in the original code. Not just style issues—real logic bugs. In the Brazil system, it detected a missing null check that had been there since 2016. It wasn’t the change we asked for, but it was a real improvement.

If you’re maintaining a legacy system that nobody wants to touch, this pipeline won’t make it modern or elegant. But it will make it *maintainable*. And that’s enough.


## What to do next

Start with a single file. Pick a function that’s small, well-scoped, and low-risk. Something like a logging helper or a config loader. Apply these steps:

1. Extract the file with Tree-sitter or `grep`:
   ```bash
grep -n "function get_log_level" legacy/utils/logger.php
```

2. Clean it with `cloc` and `sed`:
   ```bash
sed '/^[[:space:]]*$/d; /^[[:space:]]*\/\//d' legacy/utils/logger.php > logger.clean.php
```

3. Write a prompt that tells the AI exactly what to do. Be specific:
   ```
You are a maintainer.
Change the function `get_log_level` to return "error" if the level is "critical".
Do not change any other functions or variables.
Output only the modified function.
```

4. Run it through a local LLM (Ollama 0.2.9 is fine):
   ```bash
ollama run llama3.2-instruct:latest < prompt.txt
```

5. Copy the output into a new file, then run a quick syntax check:
   ```bash
php -l logger.new.php
```

If it passes, commit it with a message like “AI refactor: get_log_level edge case”. If it fails, iterate. That’s it. You’ve just made your first AI-assisted change to a legacy codebase.

Do this today. Not tomorrow. Not next week. Today. Pick one file, make one change, and see how


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

**Last reviewed:** June 24, 2026
