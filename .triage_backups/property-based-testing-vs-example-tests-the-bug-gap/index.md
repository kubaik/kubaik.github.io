# Property-based testing vs example tests: the bug gap

Example-based tests are the default in financial codebases: a developer writes a test for a known scenario, it passes, the code ships. Property-based testing flips the model — instead of asserting on specific inputs, you assert on invariants that must hold for all valid inputs, and a generator produces many cases looking for a counterexample. The comparison matters because financial systems fail in the gaps between examples: rounding drift, sign errors on negative amounts, boundary conditions in currency conversion, and state that becomes invalid after a sequence of operations. This article compares the two approaches on what they actually catch, how they behave under real constraints, and where each belongs. It ends with a concrete step you can take in the next 30 minutes.

## Why this comparison matters

Financial code has a property that makes example tests structurally weak: the valid input space is enormous and the failure modes are often arithmetic. A payment amount is not a single value; it is any decimal in a range, with currency-specific precision, sign, and rounding rules. An account balance is not a number; it is the result of a ledger that must satisfy conservation — debits equal credits, no money is created or destroyed by a transfer. Example tests sample this space. Property-based tests quantify over it, within the limits of the generator.

The second reason is operational. Financial systems are increasingly expected to demonstrate correctness beyond happy paths, and incidents often trace to edge cases that no one wrote a test for. Property-based testing does not replace example tests — it catches a different class of bug. The question is not which is better in the abstract, but which failure modes each one actually finds, and what it costs to run both.

## Option A — example-based tests: how they work and where they shine

An example test is a concrete assertion: given input X, expect output Y. In a financial context that might be: transferring 100.00 USD from account A to B leaves A with balance minus 100.00 and B with balance plus 100.00. The test is readable, fast, and serves as documentation. It is also the only kind of test that can assert on a specific business rule like "a transfer above the daily limit is rejected with code LIMIT_EXCEEDED."

Example tests shine in three places. First, regression: once a bug is found, an example test pins the exact input that broke, so it cannot silently return. Second, specification: business rules are often stated as examples, and encoding them directly reduces translation error. Third, debugging: when an example test fails, the input is right there in the test, so reproduction is trivial.

The weakness is coverage of the input space. A typical example suite for a transfer function might test positive amounts, zero, and one negative case. It will not test the interaction of rounding with a currency that has zero decimal places, or a sequence of transfers that leaves the ledger in a state where the sum of balances no longer equals the sum of transactions. Those are the bugs that reach production because no one thought to write the example.

A subtler weakness is what a passing example suite implies. Ten green tests look like evidence of correctness, but they are evidence about ten inputs. When the suite is used as a release gate, the gate measures the author's imagination, not the code's behavior over its input domain.

## Option B — property-based testing: how it works and where it shines

Property-based testing starts from invariants. For a ledger, the central invariant is conservation: for any sequence of valid transfers, the sum of all account balances plus the sum of fees must equal the initial total. A property-based framework generates random sequences of transfers — random amounts, random source and destination accounts, random currencies within a configured set — and checks the invariant after each step. When it fails, it shrinks the sequence to the smallest reproduction, which is often a two- or three-step sequence that a human can read.

The approach shines where the correctness condition is a mathematical or structural property rather than a specific output. Examples in financial systems include:

- Rounding: for any amount and any supported currency, converting to minor units and back must be idempotent within the currency's precision.
- Sign symmetry: a debit of X followed by a credit of X must restore the original balance exactly, for all X in the valid range.
- Order independence: applying a set of independent fees in any order must produce the same final balance.
- Idempotency: replaying the same idempotency key must not change state, for any request body that is otherwise valid.
- Monotonicity: increasing a transfer amount must never decrease the total fees charged, given a fixed fee schedule.

Where property-based testing is weaker: it cannot easily assert on specific error codes or human-readable messages, because the generator does not know which inputs should trigger which business rule. It also requires a generator that produces valid inputs — a generator that produces invalid inputs will fail on preconditions rather than on the property, which wastes cycles and can hide real bugs. And it is not a replacement for example tests that encode explicit regulatory rules.

## Head-to-head: what each approach catches

The most useful comparison is by failure mode. The table below is a qualitative comparison based on documented behavior of the two techniques, not a benchmark.

| Failure mode | Example tests | Property-based tests |
|---|---|---|
| Specific regression (known input) | Strong: pins the exact case | Weak unless the generator happens to produce it |
| Rounding and precision drift | Weak: only tested values | Strong: quantifies over amounts and currencies |
| Sign and boundary errors | Moderate: depends on cases written | Strong: generators include negatives and boundaries |
| State corruption after sequences | Weak: sequences rarely enumerated | Strong: generates sequences and shrinks failures |
| Business rule with specific code | Strong: asserts exact code | Weak: property usually checks state, not messages |
| Idempotency and replay | Moderate: one or two cases | Strong: generates arbitrary valid bodies |
| Performance and latency | Not applicable | Not applicable |

The pattern is clear: example tests are better at pinning known behavior; property-based tests are better at discovering unknown interactions. A financial system needs both, but the ratio matters. If the suite is all examples, the unknown interactions are untested. If the suite is all properties, the specific business rules are underspecified.

## Head-to-head: developer experience and workflow

Example tests are easy to write and read. A new engineer can open a test file and understand what is being asserted in seconds. Property-based tests have a steeper learning curve: the engineer must identify the invariant, write a generator that produces valid inputs, and interpret shrunk failures. The first property in a codebase often takes longer to write than ten example tests.

That cost is front-loaded. Once the generator and the invariant are in place, adding a new property is cheap, and the generator can be reused across the suite. The harder ongoing cost is maintenance: when the domain model changes, the generator must change with it, and a generator that silently produces invalid inputs will cause properties to pass vacuously. A common failure mode is a generator that produces only positive amounts because the random function was never configured to include negatives; the property then passes while the sign bug remains.

Debugging differs too. An example test failure gives you the input directly. A property-based failure gives you a counterexample, which the framework shrinks — but the shrunk case can still be abstract, and reproducing it requires running the same seed. Most frameworks print the seed and the shrunk input, so reproduction is possible, but it is an extra step compared to an example test.

In CI, property-based tests are slower because they run many cases. The number of cases is configurable; a common default in property-based frameworks is 100 cases per property, which is a documented default in several libraries. Teams often run a lower count on every commit and a higher count nightly. That is a workflow decision, not a property of the technique.

### Measuring the cost and the yield

Neither technique's cost is worth guessing at. Two measurements settle the question for a given codebase.

**Runtime.** Instrument the test runner to record per-test wall time and compare the total suite time with properties enabled and disabled, on the same machine and the same commit. For property tests, record the case count alongside the time so the per-case cost is visible. A property whose per-case cost is high usually points at an expensive generator or a slow model, not at the technique.

**Yield.** Track, per test file, how many defects each technique has found since it was introduced. A simple way is to tag commits that fix a bug with the test that would have caught it. Over a quarter, the count of bugs first caught by property tests versus example tests is the only honest evidence of which technique is paying for itself in that codebase.

## Head-to-head: cost model

The cost of example tests is roughly linear in the number of cases: each test is a small, fast unit. The cost of property-based tests is dominated by the number of generated cases and the complexity of the generator. For a financial ledger, a generator that produces valid transfer sequences must also produce valid account states, which can require a model of the domain. That model is code that must be maintained.

The trade-off is best expressed as a coverage-per-effort curve. Example tests give high coverage of known scenarios quickly, then flatten. Property-based tests have a higher fixed cost to set up the generator and invariant, then a lower marginal cost per additional property. The crossover depends on how many properties the domain has. A ledger with conservation, rounding, idempotency, and order independence has enough properties to justify the setup. A single currency conversion function with one rounding rule may not.

There is also a runtime cost. Generating and checking many cases takes longer than running a handful of examples. For a suite that must run in seconds on every commit, property-based tests are often scoped to a smaller case count in the fast path and a larger count in a nightly job. That is a configuration choice, not an inherent limitation.

A worked cost comparison makes the shape concrete. Suppose an example suite has 40 tests averaging 5 ms each: 200 ms total. A property suite has 6 properties at the default 100 cases, each case costing 2 ms including generation: 6 × 100 × 2 ms = 1,200 ms. (These figures are illustrative, not measured.) The property suite costs six times the runtime and covers an input space the example suite cannot enumerate. Reducing the per-commit case count to 25 brings the property suite to 6 × 25 × 2 ms = 300 ms, close to the example suite, with the full count reserved for nightly runs.

## A worked example: conservation in a ledger

The following Python example uses Hypothesis, a well-documented property-based testing library. It is self-contained and runnable. The property is conservation: for any sequence of transfers, the sum of balances equals the initial total.

```python
from hypothesis import given, strategies as st

class Ledger:
    def __init__(self, accounts):
        self.balances = {a: 0 for a in accounts}

    def transfer(self, src, dst, amount):
        if amount < 0:
            raise ValueError("negative amount")
        if self.balances[src] < amount:
            raise ValueError("insufficient funds")
        self.balances[src] -= amount
        self.balances[dst] += amount

    def total(self):
        return sum(self.balances.values())

accounts = ["a", "b", "c"]

@given(st.lists(
    st.tuples(
        st.sampled_from(accounts),
        st.sampled_from(accounts),
        st.integers(min_value=0, max_value=1000),
    ),
    min_size=1, max_size=20,
))
def test_conservation(transfers):
    ledger = Ledger(accounts)
    ledger.balances["a"] = 10_000
    initial = ledger.total()
    for src, dst, amount in transfers:
        if src == dst:
            continue
        try:
            ledger.transfer(src, dst, amount)
        except ValueError:
            pass
    assert ledger.total() == initial
```

This property will fail if the transfer function ever creates or destroys money. The generator produces sequences of up to 20 transfers, including zero amounts and same-account transfers, which the test skips. A typical bug it catches: a fee that is subtracted from the source but not added anywhere, breaking conservation. An example test would only catch that if someone wrote a case with a fee.

The example also shows a common pitfall: the test catches `ValueError` and continues, which means the property only checks conservation over successful transfers. That is intentional here, but it also means a bug that causes an invalid transfer to succeed would not be caught by this property alone. A second property should assert that invalid transfers raise.

### Reading a shrinking failure

Suppose the property fails. The framework reports a counterexample — after shrinking, something like `[(a, b, 0), (b, a, 0)]` — plus the seed that produced it. The reasoning to follow is:

1. **Reproduce first.** Re-run with the printed seed to confirm the failure is deterministic. If it does not reproduce, the generator is reading mutable state or the code under test is nondeterministic, and that is itself a finding.
2. **Read the shrunk input as a hypothesis.** A minimal counterexample usually names the mechanism. A sequence of zero-amount transfers that breaks conservation points at fee handling on zero amounts, not at arithmetic in general.
3. **Confirm with an example test.** Write the shrunk case as a plain example test. It becomes the permanent regression guard, runs in milliseconds, and documents the bug in a form any reviewer can read.
4. **Check the generator's blind spots.** If the counterexample was surprising, ask whether the generator could have produced it sooner — for instance, whether small amounts and same-account pairs are represented with realistic frequency.

Step 3 is the bridge between the two techniques: every property failure should leave behind an example test. The property keeps exploring; the example keeps the specific bug dead.

## A decision framework

Use this checklist to decide where each approach belongs. It is not a binary choice; most financial systems should have both.

1. If the rule is stated as a specific example in a regulation or product spec, write an example test. It pins the exact behavior and is easy to audit.
2. If the rule is a mathematical invariant (conservation, idempotency, rounding, order independence), write a property test. It will explore cases you did not think of.
3. If the domain has sequences of operations (transfers, settlements, reversals), write a property test over sequences. State corruption is the most common class of bug that example tests miss.
4. If the input space is large and continuous (amounts, rates, timestamps), write a property test with a generator that includes boundaries and negatives.
5. If the code is a thin wrapper over an external API, example tests with mocked responses are usually sufficient; property tests add little because the behavior is defined by the external system.
6. If the code is a pure function with a small input domain, example tests may be enough; property tests are cheap but the marginal value is lower.

A practical default: every financial domain module should have at least one property test for its central invariant, plus example tests for each business rule. That combination catches both known and unknown failures.

## Failure modes to watch for

Both techniques have characteristic ways of going wrong. Knowing them in advance saves a quarter of wasted effort.

**Vacuous properties.** A property that passes because the generator never produces inputs that exercise the code. The symptom is a property that passes in milliseconds with no apparent work, or one that passes even when the implementation is deliberately broken. The check: introduce a deliberate bug and confirm the property fails. If it does not, the generator is the problem.

**Precondition-heavy generators.** A property guarded by many `assume`-style filters that discard most generated cases. The suite runs slowly and the effective case count is far below the configured one. The fix is to build the generator so that valid inputs are the default, with filters used sparingly.

**Over-specified properties.** A property that restates the implementation rather than an invariant — for example, asserting that a function returns exactly the value the implementation computes, step by step. Such a property fails on every legitimate refactor and catches nothing. An invariant should be checkable without reading the implementation.

**Example suites that stop growing.** Once a bug is fixed and pinned, the example test stays green forever and the suite stops being extended. The symptom is a suite whose newest tests are years old while the module has changed substantially. A periodic review that asks "what input would break this module now?" restores the habit.

**Model drift.** A stateful property test that maintains a simplified model of the domain, where the model and the real implementation diverge. The property then checks the model against itself. Keep the model minimal — conservation, not a reimplementation of the ledger — so drift is visible.

## Recommendation and caveats

For financial systems, the recommendation is conditional. Use property-based testing as the primary technique for invariants and sequence behavior: ledger conservation, idempotency of payment endpoints, rounding and currency conversion, and state machines for transfers and reversals. Use example-based testing as the primary technique for specific business rules, error codes, and regulatory requirements, and as regression tests for every bug found in production. The two are complementary, and the failure mode of choosing only one is predictable: example-only suites miss interaction bugs; property-only suites miss explicit rule coverage.

The caveats are real. Property-based testing requires a generator that produces valid inputs, and a generator that is too permissive will cause properties to fail on preconditions rather than on real bugs. It also requires discipline to keep the number of cases manageable in CI. And it does not remove the need for example tests; it changes what they are for. Teams that adopt property-based testing without maintaining their example suite often find that specific business rules become underspecified.

A second caveat is tooling maturity. Property-based testing libraries exist for many languages, but the quality of shrinking and the ease of writing generators vary. In some ecosystems, the generator combinators are expressive; in others, writing a generator for a nested domain object is tedious. That cost should be weighed against the bug classes the property would catch.

## Frequently Asked Questions

**What is property-based testing in simple terms?**
Property-based testing is a technique where you assert on general properties that must hold for all valid inputs, and the testing framework generates many inputs to try to find a counterexample. Instead of writing one test per input, you write one property and let the framework explore the input space. When it finds a failure, it shrinks the input to a minimal reproduction.

**Can property-based testing replace example-based tests?**
No. Property-based tests are good at finding unknown interactions and invariants, but they cannot easily assert on specific error codes or business rules that are defined by examples. Example tests also serve as regression tests for known bugs. Most financial systems need both: properties for invariants, examples for rules and regressions.

**How many test cases should a property-based test run?**
The number of cases is configurable, and common defaults in property-based frameworks are around 100 cases per property. Teams often run a lower count on every commit and a higher count nightly. The right number depends on the cost of generation and the risk of the domain; there is no universal answer, but starting with the framework default and increasing it for critical invariants is a reasonable approach.

**Does property-based testing work for stateful systems like ledgers?**
Yes, and that is where it is most valuable. Stateful property-based testing generates sequences of operations and checks invariants after each step. For a ledger, the invariant is conservation: the sum of balances must equal the initial total. This catches bugs that example tests miss because they rarely enumerate sequences long enough to expose state corruption.

**What if a property test fails only intermittently?**
Intermittent failures usually mean one of three things: the generator depends on mutable global state, the code under test is nondeterministic (time, ordering, concurrency), or a shrinking step is being skipped so the reported counterexample is not minimal. Record the seed on every run so any failure can be replayed exactly, and treat non-reproducible failures as defects in the test harness before dismissing them as flakiness.

## Final verdict

The comparison is not close in one direction: for invariants and sequence behavior, property-based testing finds bugs that example tests structurally cannot, because it quantifies over inputs rather than sampling them. For specific business rules and regressions, example tests remain the right tool. The verdict is to use both, with a clear division of labor, and to treat the absence of any property test in a financial domain module as a gap worth closing.

The one step to take in the next 30 minutes: open the module that handles transfers or ledger updates, and write a single property test for conservation — that the sum of balances is unchanged after any sequence of valid transfers. Use the framework for your language, start with the default number of cases, and run it. If it passes, you have a baseline; if it fails, you have found a bug that example tests missed.
