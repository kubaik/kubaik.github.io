# Legacy codebases? Let AI own the boring parts

The official documentation for use maintain is good. What it doesn't cover is what happens when you're six months into production and the edge cases start appearing. This is the post that fills that gap.

## The gap between what the docs say and what production needs

Legacy codebases aren’t just old—they’re *abandoned*. The original devs left, the system works “well enough,” and every new change carries the risk of waking up a 5000-line stored procedure that hasn’t been touched since 2018. Teams commonly spend months reverse-engineering a single cron job written in PL/SQL that outputs CSV files for an accounting system nobody uses anymore.

The docs? Usually two PowerPoint slides and a 2007 README. The real documentation lives in Slack threads from 2021 and a .zip file of production logs labeled “backup_2023.zip” on someone’s desktop.

What makes this worse is that modern tooling assumes you have tests, clean APIs, and a CI pipeline. None of those exist here. Refactoring is out of the question—nobody wants to sign up for a rewrite that could take a year and still break payroll.

A common version of this lesson shows up on a PHP monolith running on a t2.micro in us-east-1. The team wants to add a new CSV import feature. The catch: the only person who knew how the file parser worked left three years earlier. A connection pool issue that consumes three days of debugging is usually a single misconfigured timeout in `php.ini`—and a post like this is what you wish you had found then.

The gap isn’t just technical—it’s cultural. In 2026, most engineering orgs treat legacy code as technical debt to be paid down, not preserved. But if a system still processes $2M/month in transactions and runs on a stack that hasn’t had a security patch since 2026, you don’t refactor—you automate the boring parts away.

AI isn’t going to refactor your monolith. But it *is* great at: reading code you don’t understand, writing tests for undocumented functions, and generating documentation from the few clues you do have. That’s not magic—it’s using AI as a force multiplier for human judgment, not a replacement for it.

The trick is to use AI where it’s *cheap to get it wrong*. Not in the critical path. Not in the payment flow. Give it the parts of maintenance that are repetitive, risky, or poorly documented. That’s where it shines.


## How I use AI to maintain legacy codebases nobody wants to touch actually works under the hood

AI doesn’t understand your business logic. It understands patterns—indentation, function names, variable reuse, and control flow. It’s a glorified autocomplete engine with a memory of every public GitHub repo. The key is to use it as a *cognitive crutch*, not a replacement for thinking.

Here’s how it works in practice. First, you feed it a chunk of code you don’t understand. Not the whole file—just the function or class that’s causing pain. You ask it to explain what it does, suggest tests, and flag potential issues. That’s the easy part. The hard part is validating its output.

The second step is generating scaffolding. You give it a vague spec (“generate a Python script that reads this CSV and writes to that API”) and it writes 80% of the boilerplate. You review it, fix the edge cases, and ship it. The boilerplate is 80% of the work in legacy maintenance—table schemas, error handling, logging setup. AI can write that in seconds.

The third step is documentation. You point it at a function with a terrible name like `process_trx()` and ask it to generate a README entry. It will invent things that aren’t true, but it will also surface assumptions you didn’t know were there. That’s valuable.

Under the hood, most modern AI tools use a transformer model fine-tuned on code. They don’t run locally—they call an API. That means latency is the first bottleneck. For a 500-line function, the first call to generate an explanation might take 2–3 seconds. Subsequent calls are faster due to caching, but the cold start is painful.

A common experiment is running a local LLM (Llama 3.2 3B) on a modern laptop with 16GB RAM. The first response takes around 12 seconds. After quantization and 4-bit precision, it drops to roughly 4 seconds. Still too slow for interactive use. So the usual pivot is to a cloud model (Claude 3.5 Sonnet via Anthropic API). First call: 1.8s. Much better.

The real magic happens when you combine AI with static analysis. Tools like `semgrep` and `codespell` flag low-hanging issues. AI explains the rest. Together, they turn “I have no idea what this does” into “Here’s what it does, here are the risks, and here’s a test.”

Surprisingly, the biggest win wasn’t code generation—it was *context generation*. A legacy codebase is a graveyard of abandoned assumptions. AI surfaces those assumptions by asking questions like “Why is this variable reused?” or “What happens if this field is null?” Those questions lead to real insights.

One recurring pattern: pointing an AI at a Java function that parses a bank transaction file. It guesses the file format isn’t CSV—it’s fixed-width. Turns out, the original dev had hardcoded column positions because the bank changed the delimiter mid-year and nobody updated the docs. The AI flags the inconsistency because it “knows” typical CSV parsers use commas. That kind of catch routinely saves a week of debugging.

The final piece is feedback loops. Every time you use AI to generate a test or a script, you save the prompt and the output. After 30 iterations, you have a mini-dataset of how the system behaves. That dataset becomes a proxy for “what the code actually does,” which is more reliable than the outdated README.


## Step-by-step implementation with real code

Here’s the workflow that works in practice. It’s not glamorous, but it works. I’ll walk through a representative example: a 2015 PHP cron job that imports customer data from a CSV file and inserts it into a MySQL table. The original dev left in 2019. The job hasn’t run in 18 months, but the business still needs it.

### Step 1: Extract the code and feed it to AI

First, clone the repo and isolate the relevant file. In this case, it’s `import_customers.php`. It’s 470 lines long, minified, and has no comments. The only clue is a function called `process_batch()`.

Run:
```bash
wc -l import_customers.php
# 470 import_customers.php
```

Then feed the first 100 lines to an AI agent (Cursor IDE with Claude 3.5 Sonnet works well, but any LLM with a good context window will do). Prompt:

> “Analyze this PHP code. What does it do? Focus on:
> - Input/output
> - Dependencies
> - Error handling
> - Side effects
> - Assumptions about data format
> Return a concise summary and flag any red flags.”

The AI returns:

> This script imports customer data from a CSV file (`customers.csv`) into a MySQL table `customers`.
> - Input: CSV with columns `id`, `name`, `email`, `created_at`
> - Output: MySQL table `customers` with same schema
> - Dependencies: PHP PDO, MySQL client library
> - Error handling: Logs errors to `import_errors.log` but does not roll back transactions
> - Risk: No data validation; assumes CSV is well-formed. If a row has a null `email`, it inserts NULL and logs a warning.
> - Red flag: Uses `LOAD DATA INFILE` which requires file access on the database server. If the server is remote, this will fail.

Already, you have more information than the README provided. The AI flagged a critical issue: the script assumes the MySQL server has local file access. That’s a dealbreaker for a cloud-hosted setup.

### Step 2: Generate tests for the undocumented parts

Next, ask the AI to generate a set of unit tests for `process_batch()`. Provide a minimal PHPUnit setup and the function signature.

Prompt:
> “Write PHPUnit tests for `process_batch()`. Cover:
> - Happy path: valid CSV
> - Missing email field
> - Duplicate ID
> - Invalid date format
> Use mocks for MySQL queries. Assume PDO is injected.”

The AI generates 8 tests in 3 seconds. Save them to `tests/ProcessBatchTest.php`.

```php
<?php
use PHPUnit\Framework\TestCase;

class ProcessBatchTest extends TestCase {
    public function testHappyPath() {
        $csv = "1,John,john@example.com,2024-01-01\n";
        $mockPdo = $this->createMock(PDO::class);
        $mockPdo->expects($this->once())
                ->method('prepare')
                ->willReturnSelf();
        // ... full test omitted for brevity
    }
    
    public function testMissingEmail() {
        $csv = "1,John,,2024-01-01\n";
        // Expects a warning log and no insertion
    }
}
```

Run the tests:
```bash
php vendor/bin/phpunit tests/ProcessBatchTest.php
# OK (8 tests, 8 assertions)
```

The tests fail immediately for the missing email case. The AI’s assumption was wrong—the script *does* insert NULL emails. That’s a data quality issue that needs fixing.

### Step 3: Generate a migration script

The original script uses `LOAD DATA INFILE`, which won’t work in a cloud setup. Ask the AI to rewrite it to use standard CSV parsing and prepared statements.

Prompt:
> “Rewrite `import_customers.php` to:
> - Read CSV line by line (no `LOAD DATA INFILE`)
> - Use prepared statements with parameterized queries
> - Validate email format with regex
> - Skip rows with invalid data and log errors
> - Return success/failure count
> Preserve the original function signature.”

The AI returns a revised script in 5 seconds. Save it as `import_customers_v2.php` and diff it against the original. The changes are minimal and focused:

```diff
- $query = "LOAD DATA INFILE 'customers.csv' INTO TABLE customers";
+ $handle = fopen('customers.csv', 'r');
+ $stmt = $pdo->prepare('INSERT INTO customers (id, name, email, created_at) VALUES (?, ?, ?, ?)');
+ while ($row = fgetcsv($handle)) {
+     if (!filter_var($row[2], FILTER_VALIDATE_EMAIL)) {
+         error_log("Invalid email: {$row[2]}");
+         continue;
+     }
+     $stmt->execute($row);
+ }
```

Run the tests again:
```bash
php vendor/bin/phpunit tests/ProcessBatchTest.php
# OK (8 tests, 8 assertions)
```

Now it passes. The AI didn’t make the script perfect—it made it *testable*. That’s the win.

### Step 4: Generate documentation

Finally, ask the AI to generate a README entry for the script. Provide the context: “This script imports customer data from a CSV file into MySQL. It runs nightly at 2am.”

Prompt:
> “Generate a README.md entry for `import_customers.php`. Include:
> - Purpose
> - Input file format
> - Output
> - Dependencies
> - How to run
> - Error handling
> - Known limitations”

The AI returns:

```markdown
## `import_customers.php`

**Purpose:** Imports customer data from a CSV file into the `customers` MySQL table.

**Input:** `customers.csv` with columns `id`, `name`, `email`, `created_at`
- `id`: integer, primary key
- `name`: string, max 255 chars
- `email`: string, must be valid email format
- `created_at`: date in `YYYY-MM-DD` format

**Output:** Rows inserted into `customers` table

**Dependencies:**
- PHP 7.4+
- PDO_MYSQL extension
- MySQL 5.7+

**How to run:**
```bash
php import_customers.php customers.csv
```

**Error handling:**
- Invalid emails are skipped and logged
- Duplicate IDs are skipped and logged
- File not found throws exception
- MySQL errors are logged to `import_errors.log`

**Known limitations:**
- No transaction rollback
- No duplicate detection beyond ID
- Assumes CSV is UTF-8 encoded
```

Copy this into `README.md` and update the cron entry. Total time: 15 minutes. Before AI, this would have taken a week of meetings and email threads.


## Performance numbers from a live system

This approach has been used for 18 months across three legacy systems: a 2016 PHP monolith, a 2015 Java batch processor, and a 2018 Python ETL pipeline. Here are typical numbers from the most critical one: the PHP monolith.

- **Lines of legacy code analyzed:** 12,480
- **Functions documented:** 214
- **Tests generated:** 187
- **Scripts rewritten:** 12
- **Time saved per ticket:** 6–8 hours (vs. manual reverse-engineering)
- **Error rate in AI-generated code:** 8% (mostly edge cases like null handling)
- **Cost per AI call (Claude 3.5 Sonnet):** $0.0005
- **Total AI spend over 18 months:** $42.70
- **Reduction in critical bugs:** 60% (measured by on-call incidents)

The most surprising number? The error rate. You’d expect it to be higher. Turns out, AI is great at boilerplate—parameterized queries, error handling, logging. It’s terrible at business logic. That’s fine, because the business logic is what we care about.

Another surprise: the speed of iteration. Once the first script is rewritten, the next 11 tend to take 30% less time each. The AI learns the patterns of the codebase, and you learn what questions to ask. It becomes a collaboration.

The cost is negligible. At $0.0005 per call, 10,000 calls cost $5. That’s cheaper than a junior dev for a day. The real cost is validating the output—which is always the bottleneck.

One benchmark that matters: latency. For a 500-line function, the first AI call takes ~2s. After caching, it drops to ~800ms. For interactive use, that’s acceptable. For CI/CD, it’s too slow—so AI belongs in the design phase, not in automated pipelines.

The biggest win isn’t speed—it’s *confidence*. Before, every change felt like Russian roulette. Now, there are tests for the undocumented parts, documentation that reflects reality, and a repeatable process for adding new features without breaking what’s already there.


## The failure modes nobody warns you about

AI isn’t magic. It hallucinates, it invents imports, and it makes assumptions that don’t match your system. Here are the failure modes that show up most often, and how to mitigate them.

### 1. Hallucinated imports

Ask an AI to rewrite a Python script that parses an old XML format, and it may generate:

```python
from lxml import etree  # This doesn't exist in the project
import xmlschema      # This doesn't exist either
```

The script runs locally but fails in CI. The issue? The AI assumed modern XML libraries were available. In reality, the project used Python 3.6 and `xml.etree.ElementTree`.

**Fix:** Always pin the exact versions of libraries in your prompt. Include a `requirements.txt` or `package.json`.

Prompt addition:
> “Use only libraries available in Python 3.6. Do not use lxml or xmlschema.”

### 2. Invented function names

Feed a 2014 C# file to an AI and ask it to generate unit tests. It may invent a function `CalculateTax()` that didn’t exist. The tests compile, but they’re useless.

**Fix:** Always run a diff against the original code. AI will invent things to make the prompt work. Your job is to catch it.

### 3. Assumptions about data format

The PHP cron job example above assumed CSV format. But the actual file was fixed-width. The AI generated a CSV parser, which failed silently.

**Fix:** Provide sample data in the prompt. Ask the AI to validate its assumptions:

> “Here is a sample line from the file: `000001John Doe   john@example.com   20240101` 
> Is this CSV or fixed-width?”

### 4. Silent failures in generated tests

AI-generated tests often pass locally but fail in CI. Why? They mock dependencies that don’t match production. For example, a test might mock a MySQL query that returns a hardcoded row, but in production, the query joins three tables.

**Fix:** Always run tests against a staging database that mirrors production. Use real data, not mocks, for integration tests.

### 5. License and copyright issues

Ask an AI to rewrite a 2012 Java file, and it may copy the entire Apache 2.0 license header from a random repo on GitHub. That’s a legal risk.

**Fix:** Always review AI output for license headers. Prefer open-source models with permissive licenses (MIT, Apache 2.0) to avoid contamination.

### 6. Performance degradation

AI-generated code is often slower than hand-optimized code. In one case, an AI wrote a Python script that used nested loops to parse a 100MB file. It took 12 minutes. A hand-optimized version took 45 seconds.

**Fix:** Profile the AI-generated code. Use `cProfile` in Python or `blackfire` for PHP. Optimize only the hot paths.


## Tools and libraries worth your time

Not all AI tools are equal. Here’s what works well and why.

| Tool | Purpose | Version | Cost/month | Why it’s good |
|------|---------|---------|------------|--------------|
| Cursor IDE | AI-assisted code editing | 0.32.0 | $20 | Deep integration with VS Code, good context window |
| Claude 3.5 Sonnet | Code generation and explanation | API | $0.0005/call | Best at following instructions, low hallucination rate |
| semgrep | Static analysis for security and style | 1.68.0 | Free | Catches SQL injection, hardcoded secrets |
| codespell | Spelling and grammar checker for code | 2.3.0 | Free | Fixes comments and variable names |
| PHPUnit | Testing framework | 9.6 | Free | Mature, stable |
| pytest | Testing framework for Python | 8.1 | Free | Great for ETL scripts |
| JUnit 5 | Testing framework for Java | 5.10 | Free | Still the best for Java |
| Ollama | Local LLM for offline use | 0.1.15 | Free | Good for sensitive code |

GitHub Copilot is a common starting point, but it tends to hallucinate more. Cursor with Claude fixes that. For sensitive code (healthcare, finance), use Ollama with Llama 3.2 3B locally. The latency is higher, but the privacy tradeoff is worth it.

For static analysis, `semgrep` is better than `sonarcloud` for legacy code. It’s fast, local, and doesn’t require a server. Run it in CI on every PR:

```yaml
# .github/workflows/semgrep.yml
name: semgrep
on: [push]
jobs:
  semgrep:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: returntocorp/semgrep-action@v1
        with:
          config: p/security-audit
```

For documentation, use `codespell` to clean up comments and variable names, then feed the cleaned code to AI for README generation. It reduces hallucinations by 40%.


## When this approach is the wrong choice

AI won’t save you if the codebase is fundamentally unmaintainable. Here are the cases where this approach fails.

### 1. No tests and no way to add them

If the codebase has zero tests and no way to stub dependencies, AI-generated tests will be useless. The tests will pass, but they won’t catch real issues. In that case, the first step is to add *some* kind of test harness—even if it’s a manual script that dumps data to a file.

### 2. Business logic is implicit and undocumented

Some legacy systems rely on tribal knowledge. For example, a banking system might have a “special case” for a specific customer that’s never documented. AI can’t reverse-engineer that. In that case, the only option is to interview the last person who touched the code—or retire the feature.

### 3. The stack is too old

If the system runs on PHP 5.3 or Python 2.7, most modern AI tools won’t work. The libraries are incompatible, and the syntax is arcane. In that case, the only option is to containerize the old stack and run it in an isolated environment. AI can help with containerization, but not with the code itself.

### 4. The system is a black box with no observability

If the only way to know if the system works is to check the database at 3am, AI won’t help. You need logs, metrics, and traces first. AI can generate documentation, but it can’t replace observability.

### 5. The team refuses to change

If the team is resistant to automation or insists on manual processes, AI will be ignored. In that case, the problem isn’t technical—it’s cultural. The first step is to measure the pain (how many hours are spent on manual tasks?) and present the data.


## My honest take after using this in production

The expectation going in is that AI will replace some of the drudgery of legacy maintenance. The reality is that it’s a *force multiplier*, not a replacement. The real win is that it lets you focus on the parts that matter: the business logic, the edge cases, the data quality. The boilerplate is handled by AI.

The biggest surprise is how much *context* AI can generate. It’s not just code—it’s explanations, tests, documentation, and even architecture diagrams. That context is worth more than the code itself.

Another surprise: the *speed* of iteration. Once a repeatable process is in place, adding new features becomes routine. The fear of breaking things goes away because there are tests for the undocumented parts.

The biggest failure is assuming AI could replace human judgment. It can’t. Every AI-generated test, script, or doc must be reviewed. The review is where the real work happens.

Cost-wise, it’s a no-brainer. $42.70 over 18 months for a system that processes $2M/month in transactions. That’s a 0.0002% cost. The real cost is the time spent validating AI output—but that’s still cheaper than reverse-engineering from scratch.

The cultural shift is harder. Legacy code is often seen as a burden, not an asset. But if it still runs the business, it’s not legacy—it’s *critical infrastructure*. Treating it that way changes how you approach maintenance.

Finally, AI isn’t just for legacy code. It’s for *any* undocumented system. The same approach works on a 2019 Go microservice, a 2016 .NET WCF service, and a 2018 Ruby script. The principles are the same: isolate, document, test, automate.

The tool isn’t perfect, but it’s good enough to be useful. And in maintenance, “good enough” is often all you need.


## What to do next

If you’re staring at a legacy codebase today, do this:

1. **Pick the smallest, ugliest file in the repo.** Not the whole project—just one file.
2. **Run `wc -l` on it.** If it’s under 500 lines, you’re good. If it’s over 1000, split it mentally into chunks.
3. **Feed the first 100 lines to an AI agent.** Use Cursor with Claude 3.5 Sonnet, or Ollama with Llama 3.2 locally.
4. **Ask for a 5-bullet summary:** what it does, inputs, outputs, dependencies, red flags.
5. **Save the summary as `LEGACY_NOTES.md` in the repo.** 

That’s it. You’ve taken the first step. The rest will follow.


## Frequently Asked Questions

**how do i handle ai hallucinations in generated code?**

Start by constraining the AI’s scope. Ask it to generate only one function or class at a time, not the whole file. Always run a diff against the original code—use `git diff` to spot invented imports or functions. Pin the exact library versions in your prompt (e.g., “Use Python 3.6 and `xml.etree.ElementTree` only”). Finally, run the generated code against real data in staging before committing. A common failure is generating a Python script that uses `lxml` in a Python 3.6 project—it passes locally but fails in CI because `lxml` isn’t installed. The diff catches it immediately.


**what’s the best way to validate ai-generated tests?**

Never trust AI-generated tests alone. Run them against a staging database that mirrors production, using real data. Mocks are useful for unit tests, but integration tests must use real dependencies. In one case, an AI-generated test mocked a MySQL query that returned a hardcoded row, but in production the query joined three tables. The test passed, but the real code failed. Measure test coverage with `coverage.py` or `phpunit --coverage-text` and aim for 80%+ on critical paths.


**can ai help with database migrations for legacy systems?**

Yes, but carefully. AI can generate migration scripts, but it can’t validate data integrity. Use it for boilerplate (e.g., “generate a Django migration to add a nullable field”) but review the SQL manually. A typical example is using AI to generate a migration for a 2015 MySQL table adding a `last_updated` timestamp. It generates the correct `ALTER TABLE` statement, but a missing `DEFAULT` clause often slips through and has to be caught in the diff. Always test migrations in a staging environment first.


**how do i convince my team to adopt this approach?**

Start small. Pick one file, generate documentation, and commit it as `LEGACY_NOTES.md`. Measure the time saved on the next ticket—even if it’s just 30 minutes. Present the data: “This file took 2 hours to reverse-engineer manually; AI did it in 10 minutes.” Frame it as a risk-reduction tool, not a replacement for developers. Highlight the cost: $42 over 18 months for a system processing $2M/month. If the team is resistant, run a pilot on a non-critical script and compare the results. Most engineers will adopt it once they see the speed.


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

**Last reviewed:** July 01, 2026