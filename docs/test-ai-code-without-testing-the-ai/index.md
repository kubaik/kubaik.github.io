# Test AI code without testing the AI

AI assistants generate code quickly, and they generate plausible-looking bugs just as quickly. The failure mode is consistent: the generated code assumes a happy path that does not exist in production. It assumes every HTTP call returns `200` with a JSON array. It assumes `user.id` is an integer. It assumes CSV rows always have values in every column. The code compiles, the example in the prompt works, and the bug ships.

The instinctive response is to test the AI. Write tests that assert the model returns a particular snippet for a particular prompt. This is a trap. Model output varies across versions, sampling parameters, context windows, and cache state. Tests written against that variability are flaky by construction, and they validate the wrong artifact. What matters is not whether the model is consistent; it is whether the code that was merged into the repository behaves correctly.

The strategy below treats AI output like any other external dependency: verify the inputs you accept, validate the outputs you produce, and measure behavior under conditions the generator never saw. None of these techniques require access to the model, and none of them change based on who or what wrote the code.

## The core principle: test the artifact, not the author

A useful litmus test for any test you write: if the AI were removed from the process tomorrow and a human wrote the same function, would the test still make sense? If yes, the test targets your code. If no — if the test only passes because a specific prompt produced a specific string — the test targets the model, and it will break for reasons unrelated to your system's correctness.

This distinction matters because AI-generated code has a characteristic defect profile. It tends to be locally clean and globally fragile:

- It handles the documented case and ignores the undocumented one.
- It assumes well-formed input.
- It assumes external calls succeed.
- It optimizes for readability over performance in hot paths.
- It silently narrows types (an ID that is sometimes a string becomes an integer).

Each of these is testable without knowing anything about the model. The techniques below are ordered roughly by how broadly they apply.

## 1. Property-based testing

Instead of asserting `output == 42`, define invariants that must hold for all valid inputs: "the sum of line items equals the order total," "a balance never goes negative," "parsing then serializing a record is idempotent." The test framework generates inputs, shrinks failures to a minimal counterexample, and reports it.

This is the highest-leverage technique for AI-assisted code because it directly attacks the "assumed happy path" defect. The generator does not know which edge cases the model overlooked, so it explores them systematically.

The cost is real: writing invariants requires more thought than writing examples. A common rule of thumb is that the first property suite for a module takes two to three times longer to author than an equivalent example-based suite. In exchange, it rarely needs updating when implementation details change, because it asserts behavior rather than structure.

```python
from hypothesis import given, strategies as st
from myapp.finance import calculate_balance

@given(
    st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=10000),   # deposit
            st.integers(min_value=0, max_value=10000)    # withdrawal
        ),
        min_size=1,
        max_size=100
    )
)
def test_balance_never_negative(transactions):
    total_deposit = sum(d for d, _ in transactions)
    total_withdrawal = sum(w for _, w in transactions)
    balance = calculate_balance(total_deposit, total_withdrawal)
    assert balance >= 0, f"Balance went negative: {balance}"
```

A representative failure this catches: a generated implementation computes `balance = deposit - withdrawal` for a single transaction instead of accumulating a running balance. An example-based test that exercises one deposit and one withdrawal passes. The property test fails on the second generated case, because the invariant is stated over the whole sequence rather than one instance.

Note the shape of the assertion: it says something about the relationship between inputs and output, not about a literal value. That is what makes it portable across implementations.

## 2. Contract testing

Contract tests pin down the shape, status codes, and error behavior of an external API, then verify that your adapter code handles each documented case — including the ones the model never saw. The point is not to test the remote service. It is to test your code's response to a 404, a 429, a malformed body, and a timeout.

Generated code frequently assumes the success path only. A contract test suite that includes rate limiting and error responses will fail immediately against such code, which is the desired outcome.

```javascript
// contracts/user-api.contract.js
const { expect } = require('@jest/globals');
const nock = require('nock');
const { fetchUser } = require('../src/services/user');

describe('User API contract', () => {
  afterEach(() => nock.cleanAll());

  it('returns 404 when user not found', async () => {
    nock('https://api.example.com')
      .get('/users/99999')
      .reply(404, { error: 'Not found' });

    const result = await fetchUser(99999);
    expect(result).toEqual({ error: 'Not found', status: 404 });
  });

  it('retries on 429 and succeeds on the third attempt', async () => {
    const scope = nock('https://api.example.com')
      .get('/users/1')
      .times(2)
      .reply(429, { error: 'Rate limit exceeded' })
      .get('/users/1')
      .reply(200, { id: 1, name: 'Alice' });

    const result = await fetchUser(1);
    expect(result).toEqual({ id: 1, name: 'Alice' });
    expect(scope.isDone()).toBe(true);
  });
});
```

The second test is the important one. It encodes two facts about your system: retries happen, and the retry budget is bounded. If a generated retry loop is unbounded, or retries on 404 as well as 429, the test fails. That is a real defect, not a style preference.

Contract tests require mocking or service virtualization, which some teams treat as overhead. The overhead is the point: it forces the network boundary into a small, explicit adapter rather than letting HTTP calls scatter through the codebase.

## 3. Fuzz testing

Fuzzing feeds random, malformed, or adversarial input to a function and asserts only that it does not crash, hang, or violate a basic safety property. For generated parsers and validators, this is often the fastest route to a real bug.

A full coverage-guided fuzzing harness is not always necessary. A bounded random loop catches a surprising amount:

```python
import random
from myapp.parser import parse_csv_row

def test_fuzz_csv_parsing():
    for _ in range(1000):
        fields = [str(random.randint(-1000, 1000))
                  for _ in range(random.randint(1, 10))]
        row = ','.join(fields)
        try:
            result = parse_csv_row(row)
            assert isinstance(result, list)
            assert all(isinstance(x, int) for x in result)
        except ValueError:
            pass  # rejected input is acceptable; crashing is not
        except Exception as e:
            raise AssertionError(f"Unexpected failure on {row!r}: {e}")
```

Two details make this useful rather than noisy. First, the loop allows a `ValueError` — a validator is permitted to reject input, but it is not permitted to raise an unexpected exception type. Second, the assertion message includes the offending input, so a failure is reproducible without a shrinking pass.

The classic defect this surfaces in generated code is an empty field. A parser written against examples where every column has a value will often index into an empty string or call `int("")` and raise `ValueError` where the caller expects a domain error. Fuzzing finds this in seconds.

## 4. Integration snapshots

Snapshot testing records a real response once and replays it in subsequent runs. When the upstream API changes, the snapshot diff makes the change visible instead of silently breaking production.

The value here is that snapshots capture real-world messiness — nullable fields, string-encoded numbers, inconsistent casing — that hand-written mocks tend to smooth over. A mock written by the same person who prompted the model is likely to share the model's assumptions. A recorded response does not.

```bash
# Record a real response once
curl -s -X POST https://api.example.com/v1/users \
  -H 'Content-Type: application/json' \
  -d '{"name":"Alice"}' > tests/snapshots/users.post.json
```

```python
from snapshottest import assert_match_file

def test_create_user_snapshot(client):
    response = client.post('/users', json={'name': 'Alice'})
    assert_match_file('tests/snapshots/users.post.json', response.json())
```

The failure mode to watch for is snapshot drift: a stale snapshot fails, someone regenerates it without reading the diff, and a real regression is laundered into the baseline. The mitigation is procedural — treat snapshot regeneration as a reviewed change, not a one-line command run reflexively.

## 5. Chaos testing in a staging environment

Chaos testing injects latency, timeouts, 5xx responses, and dependency failures into a staging environment and observes whether the system degrades gracefully. It targets the assumption that everything works, which is the single most common assumption in generated error handling.

```yaml
apiVersion: chaos-mesh.org/v1alpha1
kind: PodChaos
metadata:
  name: payment-service-pod-failure
spec:
  action: pod-failure
  mode: one
  duration: 30s
  selector:
    namespaces:
      - staging
    labelSelectors:
      app: payment-service
```

This requires real infrastructure and is not appropriate for every project. The decision rule is straightforward: if the code path can page someone at 3 a.m., it deserves a chaos experiment in staging. If it is a CLI tool that runs on a laptop, it does not.

## 6. Type-driven testing

A strict type system moves a class of defects from runtime to compile time. In TypeScript, enabling `strict` and avoiding `any` forces the generator to handle nullability and discriminated unions explicitly. In Rust, the type system enforces error handling through `Result`. In Go, interfaces and explicit error returns serve a similar role.

The limitation is that types verify shape, not values. A `number` type does not tell you the number is positive, and a `string` type does not tell you it is a valid email. Types are a complement to property tests, not a replacement.

```typescript
// types/order.ts
export type OrderItem = {
  productId: string;
  quantity: number;
  price: number;
};

export type Order = {
  id: string;
  total: number;
  items: OrderItem[];
  createdAt: Date;
};
```

```typescript
// tests/order.test.ts
import { validateOrder } from '../src/validators/order';
import type { Order } from '../types/order';

test('rejects an order whose total does not match its line items', () => {
  const order: Order = {
    id: '123',
    total: 100,
    items: [
      { productId: 'p1', quantity: 2, price: 50 },
      { productId: 'p2', quantity: 1, price: 1 },
    ],
    createdAt: new Date(),
  };
  expect(() => validateOrder(order)).toThrow('Total mismatch');
});
```

The type system guarantees the fields exist and have the right primitive types. The test guarantees the cross-field invariant holds. Neither subsumes the other.

## 7. Golden master testing

Golden master testing captures the output of a program for a fixed input and compares future runs against it. It is well suited to data transformations, report generators, and any code with deterministic output where the expected result is tedious to express as assertions.

```python
import json
import subprocess
from deepdiff import DeepDiff

def test_report_matches_golden_master():
    result = subprocess.run(
        ['python', 'src/report.py', '--input', 'data/input.json'],
        capture_output=True,
        text=True,
        check=True,
    )
    output = json.loads(result.stdout)

    with open('tests/golden/master.json') as f:
        golden = json.load(f)

    diff = DeepDiff(output, golden, ignore_order=True)
    assert not diff, f"Output differs from golden master:\n{diff}"
```

The classic defect this catches is a sort order that depends on locale or case sensitivity — a report where `"Alice"` sorts after `"bob"` because the comparison is byte-wise. That is not visible in a small example and is obvious in a golden diff.

The weakness is the same as snapshots: a golden master encodes current behavior, including current bugs. Regenerating it without inspecting the diff erases the signal.

## 8. Performance regression testing

Generated code frequently favors readability over efficiency in hot paths — an append inside a loop where a comprehension would do, or a repeated linear scan where a set lookup belongs. Functional tests will not catch this. A benchmark comparison will.

```yaml
# .github/workflows/perf.yml
name: Performance regression
on: [push]
jobs:
  perf:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: pip install pytest pytest-benchmark
      - run: pytest tests/perf/ --benchmark-only --benchmark-save=baseline
      - run: pytest tests/perf/ --benchmark-compare=baseline
```

Shared CI runners are noisy, so absolute thresholds produce flaky failures. Compare against a stored baseline on the same runner class, and set the regression threshold wide enough to absorb variance — a factor of two, not ten percent. The goal is to catch cliffs, not jitter.

### How to establish a baseline without inventing one

Rather than trusting any published number, measure your own. A minimal procedure:

1. Pick one hot function — a parser, a serializer, a request handler.
2. Write a benchmark that runs it on a fixed, representative input.
3. Run it on your CI runner class at least ten times and record the median and the spread.
4. Set the regression threshold at several times the observed spread.
5. Commit the baseline alongside the benchmark.

The threshold is now derived from your hardware and your workload rather than borrowed from someone else's environment.

## 9. Behavioral (end-to-end) testing

Behavioral tests drive the system the way a user does and assert on the observable outcome. They are slow, they need a running environment, and they catch the class of bug that only appears when real input meets real state.

```javascript
// tests/behavioral/checkout.test.js
import { test, expect } from '@playwright/test';

test('checkout rejects a quantity larger than available stock', async ({ page }) => {
  await page.goto('/products');
  await page.click('text=Add to cart');
  await page.click('text=Checkout');
  await page.fill('input[name="quantity"]', '100');
  await page.click('button[type="submit"]');
  await expect(page.locator('.error')).toHaveText('Out of stock');
});
```

The generated code that this catches typically validates the quantity is a positive integer and stops there. It never checks the quantity against inventory, because the example in the prompt used a quantity of one.

## Choosing a method

| Situation | Primary method | Secondary method | Effort |
|---|---|---|---|
| Business logic, financial calculations | Property-based | Type-driven | Medium |
| Code that calls external APIs | Contract | Integration snapshots | Medium–High |
| Parsers and validators for untrusted input | Fuzz | Property-based | Low–Medium |
| Data pipelines and report generators | Golden master | Property-based | Low |
| Performance-sensitive hot paths | Benchmark regression | Property-based | Medium |
| User-facing workflows | Behavioral | Contract | High |
| Services that page someone on failure | Chaos | Contract | High |

Start with the row that matches the code you are most afraid of. Do not adopt every row at once. A single well-chosen property suite on the payment path is worth more than five shallow suites spread across the codebase.

## Patterns that do not work

**Mocking the model layer.** Writing tests that stub a code-generation call and assert on its output couples the suite to model behavior. Output varies with version, sampling parameters, and context. The suite becomes flaky, and its failures carry no information about your system.

**Asserting on prompt-to-output pairs.** A test that asserts a given prompt yields a given snippet is testing the model, not the code. It will break on a model upgrade for reasons unrelated to correctness, and it will pass when the model produces the right string for the wrong reason.

**Generic static analysis as a substitute.** Linters and code-quality scanners catch surface-level smells — long functions, too many parameters. Generated code often passes these cleanly while containing a logic error. Static analysis is a useful gate; it is not a correctness check.

**Style rules aimed at "AI artifacts."** Rules that ban specific imports or comment lengths do not catch defects, because generated code is usually stylistically indistinguishable from human code. The signal is not in the formatting.

## A worked decision example

Suppose a service has a function that parses a currency string from a payment provider's webhook and returns an integer number of cents. The function was generated by an assistant from a prompt that included two examples: `"10.00"` and `"0.99"`.

The invariant is: for any valid decimal string with at most two fractional digits, the parsed integer equals the value in cents, and parsing is exact — no floating-point rounding.

That statement suggests three tests, none of which mention the model:

1. A property test generating random amounts with two decimal places and asserting `parse(format(n)) == n` for a range of `n`.
2. A fuzz test feeding malformed strings — empty, `"1.234"`, `"abc"`, `"-1.00"`, strings with thousands separators — and asserting a domain error rather than an unexpected exception.
3. A golden master over a fixture file of real webhook payloads, asserting the parsed output matches the recorded expectation.

The likely generated defect is a float round-trip: `int(float(s) * 100)`, which returns `999` for `"9.99"` on some inputs due to binary representation. Test one catches it immediately, because the property is stated in terms of exact equality, not approximate equality. No knowledge of the model is required to write or interpret the failure.

## FAQ

**How can I tell whether a test is testing the model or my code?**

Remove the model from the scenario. If the test still expresses a meaningful requirement about your system's behavior, it targets your code. If it only makes sense in terms of a specific generation event, it targets the model.

**Should AI-generated code be tested differently from human-written code?**

No. The techniques are the same. The emphasis shifts, because generated code more often assumes well-formed input and successful dependencies, so property and fuzz tests tend to pay off sooner. But the standard is identical.

**What if generated code changes on every commit?**

Use property-based or golden master tests. Both assert on behavior rather than implementation, so a rewrite that preserves behavior passes. A rewrite that changes behavior fails, which is the correct outcome.

**Should the model run inside CI?**

Test the code the model produced, not the model. If the model is used at build time to generate code, the generated code is reviewed and committed like any other change, and CI tests that artifact. Running the model in CI makes the build nondeterministic for no correctness benefit.

**How many property tests are enough?**

Start with the invariants that, if violated, would cause data loss or incorrect money movement. Typically three to five properties cover the core of a module. Add more when a specific bug class escapes.

## Start here

Pick one function in your codebase that handles money, identity, or user-supplied input — the kind of code where a silent wrong answer is expensive. Open it, and write down two invariants it must satisfy for *all* inputs, not just the examples in your head. Then add `hypothesis` to that project's test dependencies and write the first property test against the first invariant.

```bash
pip install hypothesis
```

That single test, written in the next thirty minutes, will tell you more about the real quality of your generated code than a week of asserting on model output.
