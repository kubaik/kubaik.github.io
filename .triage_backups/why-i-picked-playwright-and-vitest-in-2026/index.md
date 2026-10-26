# Choosing a JavaScript Test Stack: Playwright and Vitest

Most advice about JavaScript test tooling either skips the parts that matter or repeats vendor marketing. What follows is a decision framework: the gates a test stack has to pass, how to measure them honestly, where each tool breaks down, and how to migrate without a rewrite.

## What a test stack actually has to solve

A typical modern frontend is a framework app (React, Vue, SvelteKit) with real-time transport such as WebSockets, a charting or canvas-heavy library, and a GraphQL or REST backend. Common constraints:

- Small teams with no dedicated QA function.
- CI minutes that are metered, so test suite runtime has a direct cost.
- Onboarding pressure: a new hire should be able to run the suite on day one.
- Race conditions in async UI code that unit tests with a simulated DOM will not catch.

The recurring failure mode is a suite that is green locally and red in CI. The usual causes are timing assumptions that hold on a fast local machine, selector strategies that break when components unmount and remount, and mocks that leak state between test files. Any tool choice should be judged against those failure modes rather than against a feature checklist.

## Define gates before you compare tools

Write down pass/fail criteria first, because tool comparisons without them turn into preference arguments. Three gates that work well:

1. **Flake rate.** What fraction of runs fail for reasons unrelated to a real defect? You cannot know this without measuring, so instrument it.
2. **Cost per run.** Total CI spend divided by number of runs, including retries. Retries are the hidden multiplier: a suite with a 5% flake rate and automatic retries pays for those reruns.
3. **Time to first green run for a new contributor.** Measure it on a clean machine, not on the machine of the person who wrote the tests.

### How to measure flake rate

Run the same commit N times (30 is a reasonable starting point) against unchanged code and count non-deterministic failures. Most runners can emit machine-readable results; parse those rather than eyeballing the UI. Record the failure signature (test name plus error class) so you can tell a genuine intermittent bug from a selector that depends on generated class names.

### How to measure cost per run

Multiply the billed minutes per run by your provider's per-minute rate, then add retries. For example, if a suite takes 6 minutes of wall clock on a 4-vCPU runner, and your provider bills in whole minutes, that is 6 billed minutes per run. At 3,000 runs per month that is 18,000 billed minutes. At a hypothetical $0.008 per minute that is $144 per month; at $0.016 per minute it is $288. Substitute your provider's real rate — the arithmetic is the point, not the numbers. The same calculation for a 4-minute suite at $0.008 is 12,000 minutes, or $96 per month. The saving comes from runtime, not from the tool's brand.

### How to measure onboarding time

Give a new contributor a clean checkout and a written task: run the full suite, then fix one intentionally broken test. Time both. If the second step requires reading source code of the runner itself, the stack is too clever.

## Playwright for end-to-end and component tests

Playwright drives Chromium, Firefox, and WebKit through one API from Node.js. It records traces, videos, and screenshots, and it can retry failed tests.

**Where it earns its place:** the trace artifact. A trace bundles DOM snapshots, network activity, and console output on a timeline, so a failure can be inspected after the fact instead of reproduced. For a WebSocket reconnect race — where the socket reconnects while a state update is in flight — a timeline showing attempt timestamps alongside the DOM state at each step is the difference between reading a timeout message and understanding the ordering. Instrument by enabling tracing on failure (the default in recent versions) and opening the trace with `npx playwright show-trace trace.zip`.

**Where it hurts:** the API surface is broad, and newcomers tend to write selectors that depend on implementation details. Prefer role- and label-based locators, which survive refactors and markup changes. If a test depends on a generated class name, it will break on the next dependency upgrade; that is a test defect, not a tool defect.

**Component testing** mounts a single component in an isolated context and asserts on it without a full page load. This is useful for SVG- or canvas-heavy components where a full E2E run is disproportionate. Be aware that the isolated context has its own timing characteristics; code that depends on wall-clock timers can behave differently than in a full page.

## Vitest for unit and integration tests

Vitest is a Vite-native runner that reuses your Vite config and provides a Jest-compatible API surface. It runs tests in worker threads, which usually means faster startup and watch-mode feedback than a Jest plus simulated-DOM setup.

**Where it earns its place:** watch mode. Editing a component and rerunning only the affected files in well under a second changes how often people actually run tests. Speed here is a behavioural intervention, not just a convenience.

**Where it hurts:** environment stubs. `localStorage`, `WebSocket`, `fetch`, and timers usually need small adapters written by hand. The ecosystem assumes familiarity with stubbing, so budget time for it. A minimal WebSocket stub is a class assigned to `globalThis.WebSocket` that records sent messages and lets the test dispatch `open`, `message`, and `close` events manually; keep it in one shared file so every suite uses the same semantics.

**Migration from Jest** is mostly mechanical: alias the runner's globals, swap the environment package, and replace `jest.useFakeTimers()` with `vi.useFakeTimers()`. The work that is not mechanical is mocks that depend on Jest's module registry internals; those need rewriting by hand. Migrate one directory at a time and keep both runners green until the old one has no files left.

## API mocking at the network layer

Intercepting `fetch` and XHR at the network level, rather than stubbing modules, keeps tests independent of how the application imports its HTTP client. The main risk is handler state leaking between tests, which produces failures that appear only when the whole suite runs — often only in CI, where file order and parallelism differ.

The fix is a global reset in an `afterEach` hook. Treat the reset as mandatory, not optional: a suite that passes in isolation but fails in a full run is almost always leaking handlers, timers, or module-level state.

For GraphQL, intercept the single endpoint and dispatch on the operation name in the request body. Subscription-style transports need separate handling because they are long-lived connections rather than request/response pairs; mock the transport, not the query.

## A worked comparison

The table below is a decision aid, not a benchmark. Every number in it depends on your codebase, so treat the columns as questions to answer for yourself.

| Question | What to record | Why it matters |
|---|---|---|
| Unit test wall time | Seconds for the full unit suite on a fixed runner | Directly drives CI cost |
| E2E wall time | Seconds, per browser project | Multiplies by browser count |
| Flake rate | Non-deterministic failures / total runs over 30 runs | Drives retries and trust |
| Debug time | Minutes from red build to root cause, sampled over 5 real failures | The metric most teams never track |
| Onboarding time | Minutes for a new contributor to run the suite and fix one seeded failure | Predicts long-term maintenance cost |

To make the cost comparison concrete with stated assumptions: suppose a unit suite runs in 4 seconds and an E2E suite in 3 minutes on a 2-vCPU runner, and the provider bills $0.008 per minute. A CI job that runs both takes about 3.1 minutes, or roughly $0.025 per run. At 3,000 runs per month that is about $74. Change the E2E suite to 6 minutes and the same math gives about $0.049 per run, or about $146 per month. These figures are illustrative; substitute your own timings and rates.

The debugging comparison is harder to tabulate but more important. A runner that emits a timeline artifact turns an intermittent failure into a reading exercise. A runner that emits only a timeout message turns it into a reproduction exercise, which is far more expensive.

## Where these tools break down

- **Simulated DOM for async UI.** A simulated DOM does not model layout, paint, or real network timing. Bugs in resize observers, scroll behaviour, and socket reconnection ordering will pass unit tests and fail in production. Cover those paths with a real browser.
- **Selector brittleness.** Any locator tied to generated class names or DOM structure will break on framework upgrades. Use accessible roles and labels.
- **Mock leakage.** Shared mutable state across test files produces order-dependent failures. Reset in `afterEach` and run the suite with randomised order at least once before trusting it.
- **Retry masking.** Automatic retries improve the signal-to-noise ratio but also hide genuine intermittent defects. Track which tests only pass on retry and treat that list as a bug queue.
- **Isolated component contexts.** Mounting a component outside a full page changes timer and layout behaviour. Verify anything timing-sensitive in a real page.

## When a legacy runner still makes sense

A team with a large existing suite in another runner should not migrate for its own sake. The migration cost is real and the payoff is mostly in debugging ergonomics and runtime. A reasonable rule: migrate when the existing suite's flake rate or runtime is actively blocking delivery, or when a framework upgrade has broken the runner's compatibility. Otherwise, contain the legacy suite, stop adding to it, and write new tests in the new stack.

## Decision checklist

- [ ] Gates written down: flake rate, cost per run, onboarding time.
- [ ] Flake rate measured over at least 30 runs of an unchanged commit.
- [ ] CI cost computed from billed minutes per run times your provider's rate, including retries.
- [ ] Traces or equivalent artifacts enabled on failure.
- [ ] Locators use roles and labels, not class names or DOM paths.
- [ ] Mock handlers reset in `afterEach`.
- [ ] Suite run once with randomised file order.
- [ ] Tests that pass only on retry tracked as defects.
- [ ] Timing-sensitive behaviour verified in a real browser.
- [ ] Migration scoped per directory, with both runners green during the transition.

## FAQ

**How do I migrate from Jest to Vitest without rewriting every mock?**
Alias the runner's globals to the new ones, swap the environment package, and rename timer APIs. Mocks that reach into the old runner's module registry internals must be rewritten by hand. Migrate one directory at a time.

**Why does a trace show a different DOM state than local dev tools?**
Traces are captured in a clean browser profile with no extensions, ad blockers, or cached assets. Local dev tools reflect your machine's state. Trust the trace for test failures.

**What is the fastest way to stub a WebSocket in a unit test?**
Assign a mock class to `globalThis.WebSocket` that records sent frames and lets the test dispatch `open`, `message`, and `close` events. Keep it in one shared file so all suites share semantics.

**How do I stop mock handlers from leaking between tests?**
Call the mock server's reset function in an `afterEach` hook. If failures appear only in full-suite runs, leaking handlers or timers are the first thing to check.

## Do this in the next 30 minutes

Pick your slowest or flakiest E2E test and enable trace capture on failure, then run that single test with `npx playwright test <path> --trace on`. Open the resulting trace and find the exact moment the assertion failed. If the trace does not make the cause obvious within five minutes, the test is asserting on something too indirect — rewrite its locator to target an accessible role or label and rerun. That single loop, applied to your worst test, tells you more about whether this stack fits your project than any comparison table.
