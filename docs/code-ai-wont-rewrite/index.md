# Code AI won’t rewrite

## Why legacy code resists ordinary maintenance

Legacy codebases are not simply old. They are codebases that have outlived the people who understood them. The business still depends on them, but nobody remembers why `retryFailedTransactions()` retries exactly three times, or why a cron job runs at `0 3 * * *` in a timezone where nobody is awake to watch it.

The documentation is usually missing, wrong, or written for a different stack. Pull requests sit for weeks because reviewers treat the legacy system as a black box. A common failure mode is a team rewriting an entire module only to discover that the new version broke something subtle — a nightly batch job that depended on a side effect in the old code, for example.

Worse, the failure is often silent. A connection pool can drain because threads are stuck waiting on a timeout that never fires, while the logs stay quiet and the metrics dashboard shows nothing. The symptom only becomes visible when you dump the pool state with a tool like VisualVM and see hundreds of threads holding connections. The lesson generalizes: legacy code does not just break, it hides.

Most teams respond with more meetings, more documentation, and more code review. Meetings do not run the batch job at 3 AM. Documentation does not catch a typo in an endpoint name that has survived since 2018. Code review cannot stop the next engineer from pasting a decade-old snippet into a critical path.

AI changes the economics here — not because it is smart, but because it is fast and repeatable. It will not replace the engineer, but it can shorten the time it takes to understand unfamiliar code.

## What AI-assisted legacy work actually looks like

The key is not to ask AI to maintain the code. It is to ask it to help you *understand* it.

A workable loop has four stages:

1. **Static analysis pass.** Run a linter or pattern matcher across the whole codebase to find candidate problems. Semgrep is one common choice; it accepts a custom ruleset and emits JSON. Typical legacy antipatterns worth flagging include `SimpleDateFormat`, `eval()`, and SQL built by string concatenation.
2. **Context assembly.** For each finding, collect the surrounding function, its callers, and any relevant deployment facts (timezone, container runtime, database engine).
3. **Model query.** Ask a model to explain the function, list the top risks of changing it, and propose a safer refactor with the tests that would be needed.
4. **Human review.** You pick the finding with the highest impact and lowest change risk, then write the test first.

A prompt that works well for step 3 looks like this:

```
You are a senior engineer reviewing legacy code.

Context: This is a monolith written in Java 8 and Python 2.7. It runs batch jobs
nightly and serves an API used in Colombia and Mexico. The team is remote,
timezone-diverse, and no one remembers why the code looks like this.

Task:
- Explain what this function does.
- List the top 3 risks if we change it.
- Suggest a safer refactor and the tests we need to add.
- Return JSON only. No markdown.
```

The context block matters more than the task block. A model that does not know the code runs in containers will not flag a dependency on a file lock in `/tmp`. A model that does know will flag it immediately. That is not intelligence; it is breadth, and you supply the breadth.

To feed findings into that prompt, a shell pipeline is enough:

```bash
semgrep --config=./legacy-rules.yaml --json \
  | jq -r '.results[] | "\(.path):\(.start.line)-\(.end.line)"' \
  > findings.txt
```

From there, a small script reads each finding, extracts the function body, and calls a local model. Running the model locally avoids shipping proprietary source code to a third party — a hard requirement in many regulated environments.

## Worked example: a timezone bug that no profiler shows

Consider a Java service that stores transaction timestamps as `java.util.Date` and runs a nightly report. Suppose the application runs in a region at UTC-5 and the database stores UTC. The report query groups by date:

```java
public void generateDailyReport() {
    Date start = new Date(); // uses the JVM default time zone
    ...
}
```

And the SQL contains something like `GROUP BY DATE(start)`.

Now reason through what happens at a daylight-saving transition, assuming the region observes DST and the transition shifts local time by one hour:

- Before the transition, local 02:00 maps to UTC 07:00.
- After the transition, local 02:00 maps to UTC 06:00.
- A job scheduled at local 02:00 therefore produces a UTC timestamp one hour earlier on the transition day.
- If the report groups by the database server's local date rather than UTC, one hour of transactions falls into the wrong bucket.

The arithmetic is simple: one hour of transactions is misclassified, and no exception is thrown. This is the class of bug that `jstack`, `jvisualvm`, and similar tools will never surface, because nothing is stuck and nothing is slow. The tooling is looking at the wrong layer.

A model given the full class and this prompt:

> Analyze this Java class for timezone-sensitive operations. Return a list of methods that may produce inconsistent results during daylight-saving transitions.

will typically flag the `new Date()` call and the date-grouping SQL. Candidate fixes include:

- Set the JVM default timezone explicitly at startup, e.g. `TimeZone.setDefault(TimeZone.getTimeZone("UTC"))`.
- Use `java.time.ZonedDateTime` or `Instant` in new code instead of `java.util.Date`.
- Backfill or reinterpret historical timestamps consistently, and make the database session timezone explicit.

Each of these has a cost. Setting the default timezone changes behavior for every other component that relied on the implicit local zone, so it must be verified against the whole application, not just the report. That verification is the actual work; the model's contribution is pointing at the right line.

## Failure-mode analysis: where this approach breaks

AI-assisted legacy work fails in predictable ways. Knowing them in advance saves time.

- **Hallucinated APIs.** A model may suggest a method that does not exist in the version of the library you use. Always check the suggestion against the actual dependency version before applying it.
- **Confident wrong fixes.** A model can propose a refactor that changes behavior in a way it did not notice. This is why the test comes before the change, not after.
- **Stale context.** If you feed the model a function without its callers, it will optimize the function in isolation. The bug may be in the caller's assumption, not the function.
- **Encoding and locale traps.** String handling in legacy code often depends on platform defaults. A model reading the source will not see the deployment's locale. You have to state it.
- **Silent scope creep.** A "safe refactor" can quietly change exception handling, ordering, or side effects. Diff the behavior, not just the text.

The mitigation for all of these is the same: write the test that captures current behavior first, then change the code, then run the test. If you cannot write that test, you do not yet understand the code well enough to change it.

## A worked floating-point example

Monetary arithmetic in floating point is a classic legacy hazard. Suppose a bonus function is written as:

```python
def calculate_bonus(base):
    return base * 1.15  # 15% bonus
```

And suppose the base salary arrives as a string from a legacy import, then gets converted to `float`. The conversion is where precision is lost.

Work through the arithmetic with a stated assumption. Take a base of `1000000.00`. Multiply by `1.15`:

- Exact decimal result: `1150000.00`.
- In binary floating point, `1.15` is not exactly representable, so the product may land at something like `1150000.0000000001` depending on the platform.

Now scale it. If each of 5,000 employees is off by roughly `0.0000000001` in the stored value, the individual error is negligible. But if the error is larger — say the string conversion itself introduces a cent-level drift — the aggregate can cross a reporting threshold. The point is not the specific magnitude; it is that the error is invisible in the source and only appears in aggregate.

The fix is to use exact decimal arithmetic:

```python
from decimal import Decimal

def calculate_bonus(base: Decimal) -> Decimal:
    return (base * Decimal("1.15")).quantize(Decimal("0.01"))
```

And a test that pins the behavior:

```python
def test_bonus_calculation():
    assert calculate_bonus(Decimal("1000000.00")) == Decimal("1150000.00")
```

A model can flag the `float` usage, but it cannot tell you the magnitude of the error without knowing the data distribution. That is your job, and it is a good reason to instrument the actual pipeline rather than trust a suggested number.

## A worked SQL injection example

Consider a query built by string concatenation:

```java
public User findByEmail(String email) {
    String hql = "from User u where u.email = '" + email + "'";
    return session.createQuery(hql).list().get(0);
}
```

A "sanitizer" that escapes quotes looks safe:

```java
String safeEmail = email.replace("'", "''");
```

But escaping by doubling quotes is not sufficient across all database engines. An input containing a backslash followed by a quote can defeat naive escaping, because the backslash may itself be treated as an escape character depending on the engine's string-literal rules. The result is a query that matches more rows than intended — effectively an injection.

The fix is to stop building queries from strings:

```java
public User findByEmail(String email) {
    return session
        .createQuery("from User u where u.email = :email", User.class)
        .setParameter("email", email)
        .uniqueResult();
}
```

And a test that proves the fix:

```java
@Test
public void testEmailSanitization() {
    String malicious = "admin@company.com\\'";
    User u = userDao.findByEmail(malicious);
    assertNull(u);
}
```

Note the parameterized query also changes behavior for the empty-result case: `uniqueResult()` returns `null` instead of throwing on an empty list. That is a behavior change you should confirm is acceptable before merging.

## Instrumenting the improvement

Claims about how much AI assistance helps are easy to make and hard to verify. If you want to know whether your own process is improving, measure it directly rather than trusting a table.

What to instrument:

- **Change lead time.** Time from first commit on a branch to merge. Most version-control hosts expose this per pull request.
- **Change failure rate.** Fraction of merges that require a follow-up fix within a stated window.
- **Mean time to repair.** Time from alert to resolution, taken from your incident tracker.
- **Test coverage on touched files.** Coverage on the specific files you changed, not the whole repository.

What to compare: the same metrics over a baseline period before you introduced the new workflow, and the same period after. Keep the measurement window identical in length and similar in workload, or the comparison is meaningless.

What to watch for: an improvement in lead time that comes with a rise in change failure rate is not an improvement. The two metrics must move together.

## A decision checklist before you touch a legacy module

Use this before starting any change:

- Can you write a test that captures the current behavior? If not, stop and build one.
- Do you know every caller of the function you are changing? Trace them, not just the definition.
- Does the function depend on environment state — timezone, locale, filesystem, environment variables, container lifecycle? List each one.
- Is there a scheduled job, batch process, or external integration that depends on a side effect? Check the cron table and the message queues.
- Does the change alter error handling, ordering, or null behavior? Confirm each change is intentional.
- Is the refactor reversible? Can you deploy the old version if the new one fails?
- Have you confirmed every API the model suggested actually exists in your dependency versions?

If you cannot answer all of these, the change is not ready.

## What AI does and does not do here

AI does not understand your business rules, your data distribution, or your deployment topology. It does not know that a cron job runs in a container with an ephemeral filesystem unless you tell it. It does not know that a report is audited, or that a rounding error crosses a legal threshold.

What it does is read code faster than you can, hold more of it in context than you can, and produce a first draft of an explanation, a test, or a refactor. That draft is a starting point, not a conclusion. The engineer's job shifts from writing every line to verifying every claim.

The teams that get value from this approach are the ones that treat the model as a fast, fallible research assistant and keep the test-first discipline intact. The teams that get burned are the ones that accept the output without checking it against the actual system.

## Do this in the next 30 minutes

Pick one function in your legacy codebase that you are afraid to change. Run a static analysis pass over just that file, extract the function and its direct callers, and ask a model to list the top three risks of changing it. Then, before you touch anything, write a single test that captures the function's current behavior — including the case you suspect is buggy. If the test passes, you have a safety net. If it fails, you have found the bug.
