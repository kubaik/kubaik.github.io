# Vibe coding: MVP gem, prod albatross

## What "vibe coding" actually optimizes for

Vibe coding is the practice of writing code by iterating quickly against immediate feedback — a REPL session, a notebook cell, a browser refresh — without a design step, tests, or type checking. It is genuinely effective at one thing: reducing the time between an idea and a running artifact.

That property is valuable. It is also narrowly scoped. The techniques that make prototyping fast (global state, copy-paste reuse, no schemas, no tests) are the same techniques that make a codebase expensive to change once it has users, multiple contributors, and a deployment pipeline.

The failure is not that vibe coding produces bad code. It is that the code keeps working long enough to acquire dependents. A prototype that gets one paying customer is now a production system with a prototype's architecture.

This article covers where the approach breaks, how to measure the breakage in your own repository rather than trusting anyone's numbers, which tooling categories reduce the cost, and how to migrate incrementally without a rewrite.

## The four failure modes

Almost every "the prototype became production" incident traces back to one of these.

### 1. Implicit state

Notebook cells and REPL sessions accumulate state that is not visible in the source file. A cell defines a variable; a later cell reads it. The notebook runs top-to-bottom in an interactive session and fails when run headlessly in CI, because the execution order or the pre-existing globals are gone.

The same pattern appears in application code as module-level mutable state: a cache dictionary that is populated at import time, a singleton client that is created on first call, a global config object mutated by whichever module loads first. It works until two code paths disagree about initialization order.

### 2. Untyped boundaries

When data crosses a boundary — HTTP request to handler, queue message to worker, environment variable to config — there is no check that the shape is what the consumer expects. In a dynamic language, a missing field surfaces as `undefined is not a function` deep inside a call stack, far from the request that caused it. The debugging cost is proportional to the distance between the boundary and the crash site.

### 3. No regression signal

Without tests that assert behavior, the only way to know a change is safe is to exercise the affected path manually. As the surface area grows, the fraction of paths a developer can manually check in a session shrinks toward zero. Changes become risky by default, and the team compensates with slow, careful releases rather than fast ones — which is the opposite of what vibe coding was supposed to buy.

### 4. Configuration drift

Infrastructure defined by ad-hoc commands (`terraform apply` against a state file nobody inspects, Kubernetes manifests with literal IPs, environment variables edited in a dashboard) diverges from what is in version control. The failure mode is a deploy that works from one machine and not another, or an environment that cannot be recreated after an incident.

None of these are language problems. They are all consequences of skipping a boundary, a check, or a record.

## How to measure whether your codebase has drifted

Rather than quoting error rates or onboarding times from someone else's project, instrument your own. Every measurement below is a command or a count you can run today.

**Untyped surface area.** If the project is TypeScript, count occurrences of `any` and of `@ts-ignore` / `@ts-expect-error`:

```
grep -rn --include='*.ts' --include='*.tsx' -E '\bany\b|@ts-(ignore|expect-error)' src | wc -l
```

Compare that number to the total line count of `src`. A ratio that grows over time means the type system is being routed around rather than used.

**Test signal quality.** Run the suite twice under identical conditions and compare results. A suite with order-dependent or timing-dependent tests will produce different outcomes. Then check coverage of the modules that handle money, auth, and persistence specifically — aggregate coverage numbers hide the fact that the risky modules are untested.

**Boundary validation.** For each external input (HTTP handler, queue consumer, cron entry point), check whether there is a schema or type assertion at the entry. The count of validated entry points divided by the total number of entry points is your boundary coverage. This is a small, countable number, not a survey.

**Time to reproduce a failure.** Take a recent production bug. Measure wall-clock time from "we know something is wrong" to "we have a failing test that reproduces it." If that number is measured in hours or days, the codebase lacks the observability and test scaffolding to localize faults.

**Onboarding.** Have someone unfamiliar with the repository attempt a small, well-specified change — add a field to an existing response, for example — and record where they get stuck. The blockers they hit are the actual documentation and structure gaps.

**Error rate under load.** If you want a load figure, generate it yourself. Put the service behind a load generator at your expected peak request rate, run it for a duration longer than your longest cache TTL and longer than your connection pool's idle timeout, and record the error rate and the p99 latency. The interesting failures — connection leaks, eviction storms, retry amplification — appear at the timescale of those timeouts, not in the first minute.

A note on one specific trap: a cache configured with an eviction policy that only evicts keys carrying a TTL will fail to evict keys that have none. When memory fills with non-expiring keys, writes start failing. The symptom is periodic, correlated with memory pressure rather than traffic. This is a configuration property you can check directly — list the eviction policy and count keys without TTL — rather than something to discover during an incident.

## The tooling that pays for itself

The categories below are not a ranked list of technologies. They are the four mechanisms that address the four failure modes above.

### Static types at the boundaries

A type system that runs before the code does converts a class of runtime failures into compile failures. The important part is not annotating every internal function; it is annotating the boundaries — request payloads, response shapes, config, and the return types of anything that touches the network or disk.

Runtime validation complements static types at the edges where data is genuinely untrusted. A schema library that parses an incoming payload and throws on mismatch turns a downstream `undefined` into a clear error at the entry point.

### Tests that assert behavior

The value of a test is the regression it catches, not the line it covers. A test that asserts a function returns the sum of its inputs catches nothing; a test that asserts a retry stops after N attempts, or that a timeout rejects with a specific error, catches a real change in behavior.

Write tests at the level where the behavior is specified. For a retry helper, that means testing the retry count and the timeout, not the internal timer implementation.

### Linting and formatting

Linting is cheap consistency enforcement. The rules that matter most are the ones that prevent whole categories of bug — floating promises, unchecked `any`, unused variables that mask a typo — rather than stylistic preferences. Formatting is best delegated entirely to a tool so it never appears in a code review.

### Reproducible configuration

Infrastructure and environment configuration should be reconstructable from version control. The test is simple: can a new environment be created from the repository alone, with no manual steps that exist only in someone's memory? If not, the configuration is documentation, not infrastructure.

## A worked example: making a retry helper safe

Consider a retry-with-timeout helper, the kind of utility that gets written quickly during a prototype and then relied on everywhere.

The prototype version typically looks like this:

```typescript
// prototype version — do not ship
export function withTimeout(fn: () => Promise<any>, ms: number, retries: number) {
  return new Promise((resolve, reject) => {
    setTimeout(() => reject(new Error('timeout')), ms);
    fn().then(resolve).catch((err) => {
      if (retries > 0) return withTimeout(fn, ms, retries - 1);
      reject(err);
    });
  });
}
```

Three defects are visible on inspection:

1. The timer is never cleared. On success, the promise resolves but the timer still fires later and calls `reject` on an already-settled promise — harmless in this case, but it keeps the event loop alive and, in a server process, holds a handle per call.
2. The timeout does not cancel the in-flight operation. `fn` keeps running after the caller has given up.
3. The retry recursion has no delay, so a failing dependency is hit as fast as the event loop allows — the classic retry-amplification pattern that turns a partial outage into a full one.

The corrected version separates configuration validation from the retry loop, clears the timer on every path, and adds a delay between attempts:

```typescript
// src/utils/timeout.ts
import { z } from 'zod';

const TimeoutConfig = z.object({
  timeoutMs: z.number().min(100).max(10000),
  retries: z.number().min(0).max(5),
  delayMs: z.number().min(0).max(5000).default(100),
});

type TimeoutConfig = z.infer<typeof TimeoutConfig>;

export function withTimeout<T>(
  fn: () => Promise<T>,
  config: TimeoutConfig
): Promise<T> {
  const { timeoutMs, retries, delayMs } = TimeoutConfig.parse(config);

  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(new Error(`Timeout after ${timeoutMs}ms`));
    }, timeoutMs);

    fn()
      .then((result) => {
        clearTimeout(timer);
        resolve(result);
      })
      .catch((err) => {
        clearTimeout(timer);
        if (retries > 0) {
          setTimeout(() => {
            withTimeout(fn, { timeoutMs, retries: retries - 1, delayMs })
              .then(resolve)
              .catch(reject);
          }, delayMs);
          return;
        }
        reject(err);
      });
  });
}
```

The tests target the behavior that matters — the timeout, the retry count, the delay, and the configuration bounds:

```typescript
// __tests__/timeout.test.ts
import { withTimeout } from '../src/utils/timeout';

describe('withTimeout', () => {
  it('rejects when the function exceeds the timeout', async () => {
    const slowFn = () =>
      new Promise((resolve) => setTimeout(resolve, 200));
    await expect(
      withTimeout(slowFn, { timeoutMs: 100, retries: 0, delayMs: 0 })
    ).rejects.toThrow('Timeout after 100ms');
  });

  it('retries until the function succeeds', async () => {
    let attempts = 0;
    const flakyFn = () => {
      attempts++;
      if (attempts < 3) return Promise.reject(new Error('flaky'));
      return Promise.resolve('ok');
    };
    const result = await withTimeout(flakyFn, {
      timeoutMs: 100,
      retries: 3,
      delayMs: 0,
    });
    expect(result).toBe('ok');
    expect(attempts).toBe(3);
  });

  it('stops retrying after the configured limit', async () => {
    let attempts = 0;
    const alwaysFails = () => {
      attempts++;
      return Promise.reject(new Error('always'));
    };
    await expect(
      withTimeout(alwaysFails, { timeoutMs: 100, retries: 2, delayMs: 0 })
    ).rejects.toThrow('always');
    expect(attempts).toBe(3); // initial attempt plus two retries
  });

  it('rejects invalid configuration before calling the function', async () => {
    const fn = jest.fn().mockResolvedValue('ok');
    await expect(
      // timeoutMs below the minimum of 100
      withTimeout(fn, { timeoutMs: 50, retries: 0, delayMs: 0 })
    ).rejects.toThrow();
    expect(fn).not.toHaveBeenCalled();
  });
});
```

The last test is the one that matters most in practice: it asserts that a misconfiguration fails at the boundary rather than after the operation has already been attempted. That is the difference between a five-minute fix and a three-day investigation.

## Configuration for the tooling

```json
// package.json
{
  "scripts": {
    "test": "jest",
    "lint": "eslint . --ext .ts,.tsx",
    "typecheck": "tsc --noEmit"
  },
  "devDependencies": {
    "@types/jest": "^29.5.12",
    "@typescript-eslint/eslint-plugin": "^7.11.0",
    "eslint": "^8.56.0",
    "eslint-config-prettier": "^9.1.0",
    "jest": "^29.7.0",
    "ts-jest": "^29.1.2",
    "typescript": "^5.4.5"
  }
}
```

```json
// .eslintrc.json
{
  "root": true,
  "parser": "@typescript-eslint/parser",
  "plugins": ["@typescript-eslint"],
  "extends": [
    "eslint:recommended",
    "plugin:@typescript-eslint/recommended",
    "plugin:@typescript-eslint/strict",
    "prettier"
  ],
  "rules": {
    "@typescript-eslint/no-explicit-any": "error",
    "@typescript-eslint/no-floating-promises": "error"
  }
}
```

Pin exact versions in your own project and update deliberately. Version ranges in a lockfile-less setup are a common source of "it worked yesterday" failures.

## Other tooling categories worth knowing

**Compiled languages with strict compilers.** Languages whose compilers reject unsafe memory access or unchecked errors move a class of defect from runtime to build time. The tradeoff is development velocity: the compiler rejects programs that a dynamic language would run, at least until they fail. This is a reasonable trade for services where a crash is expensive and the domain is stable, and a poor trade for exploratory work where the shape of the problem is still unknown.

**Languages with minimal feature sets and built-in tooling.** A small language surface makes code easier to read across a team and reduces the space of clever-but-wrong constructs. Error handling that is explicit and verbose is a cost, but it also makes failure paths visible in the source rather than hidden in exception propagation.

**Dynamically typed languages with optional static checking.** Adding a type checker to a dynamic language is a common middle path. The checker catches a subset of errors before runtime, and the annotations double as documentation. The cost is that the checker is opt-in per module, so coverage is uneven unless enforced in CI.

**Managed platforms for internal tools.** Drag-and-drop builders produce working internal dashboards quickly. The costs are that the generated behavior is opaque when it breaks, and that the tool's data model is not portable. For short-lived internal tools this is often the right trade; for anything with a long lifetime, the exit cost should be estimated up front.

**AI code completion.** Completion tools generate plausible code quickly. The relevant risk is that plausibility and correctness are different properties. Generated code that compiles and reads correctly can still be wrong in ways that only appear under load or at a boundary — a connection that is opened and never closed, a retry with no backoff, a check that is inverted for the empty case. The mitigation is not to avoid these tools but to apply the same review and test discipline to generated code as to hand-written code, with particular attention to resource lifecycle and error paths.

## When to stop vibe coding

The decision is not about code quality in the abstract. It is about whether the system has dependents whose failures are expensive.

Vibe coding remains appropriate when:

- The only user is the author.
- The artifact has a defined, short lifetime and will be discarded.
- The problem domain is still being explored and the shape of the solution is unknown.
- A full rewrite is acceptable and expected.

Move to structured tooling when any of these becomes true:

- Someone other than the author depends on the system working.
- The code will outlive the current sprint.
- More than one person edits it.
- It runs in an environment where failure is visible to users.
- A change can no longer be verified by hand in a single session.

The last condition is the practical trigger. Once manual verification stops being sufficient, the absence of automated verification becomes the dominant cost.

## A migration path that does not require a rewrite

Rewrites are expensive and usually unnecessary. The failure modes above can be addressed incrementally, in an order that front-loads the highest-value fixes.

1. **Turn on the compiler in non-strict mode and fix what it reports.** This is mechanical and produces immediate value at the boundaries.
2. **Add schema validation at every external entry point.** This is the highest-value change per line of code, because it converts distant crashes into local errors.
3. **Write tests for the paths that handle money, auth, and persistence.** Coverage of these modules matters more than aggregate coverage.
4. **Run the tests and the type checker in CI on every change.** A check that is not enforced will not be maintained.
5. **Turn on linting rules that prevent whole bug categories**, not stylistic rules.
6. **Move configuration into version control** and verify it by recreating an environment from scratch.
7. **Tighten the compiler**, one module at a time, as each is brought under test.

Each step is independently shippable. None requires pausing feature work.

## FAQ

**How do I tell whether a codebase is still in "vibe" state?**

Look for commented-out code nobody dares delete, tests that only cover the happy path, environment variables committed to the repository, a `utils` module whose contents are unfamiliar to the current maintainers, and a README that does not describe how to run the project. Any one of these indicates the codebase is being maintained by memory rather than by structure.

**What is the fastest way to make a prototype maintainable?**

Add validation at the boundaries first, then tests for the risky paths, then enforce both in CI. Types and linting follow. The ordering matters: boundary validation and tests catch the failures that actually page someone.

**Why does prototyping feel more productive than structured development?**

Because prototyping optimizes for the feedback loop, and the feedback is immediate and visible. Structured development's payoff is deferred and mostly invisible — it shows up as the bug that did not happen. Comparing the two by how they feel during a work session systematically favors the prototype.

**Can AI tools replace the discipline described here?**

No. They change the cost of producing code, not the cost of verifying it. Generated code still has to be reviewed, tested, and maintained, and it is subject to the same failure modes — resource leaks, missing backoff, incorrect edge-case handling — as code written by hand. The discipline is what makes the output of these tools safe to depend on.

## The next 30 minutes

Run the type checker and the linter over your current project and count the errors:

```
npx tsc --noEmit
npx eslint . --ext .ts,.tsx
```

If the project is not TypeScript, run the equivalent static check for your language. Record the two numbers. Then pick the single module that handles the most valuable data — payments, authentication, or persistence — and write one test that asserts a behavior you would be upset to lose. That test, not the error count, is the first piece of the safety net.
