# AI Code Generation: Failure Modes and Guardrails

## What AI code assistants actually change

The common framing is that AI assistants are faster autocomplete: they write boilerplate, fix typos, suggest a unit test. That framing holds for trivial tasks. It breaks down once the assistant touches anything that affects state, concurrency, interfaces, or external dependencies.

The reason is structural. A code assistant optimises for locally plausible code. It has no model of your connection pool size, your DST rules, your event-listener cleanup, or the downstream clients that depend on a field constraint. So the output is often correct in isolation and wrong in context — and it arrives fast enough that review becomes the bottleneck rather than the author.

The rest of this article covers the failure modes that recur, the code that produces them, and the checks that catch them. The examples are illustrative; the mechanisms are the point.

## Pattern 1: Silent state mutation

A typical failure mode is an assistant adding a cache without a bound or eviction policy, because the prompt said "add a cache" and nothing said "and bound it."

```javascript
// Illustrative: unbounded in-process cache
const cache = new Map();

function getUser(id) {
  if (cache.has(id)) return cache.get(id);
  const user = db.fetchUser(id);
  cache.set(id, user);
  return user;
}
```

Nothing here is syntactically wrong. The defect is that `cache` has no TTL, no size cap, and no metrics. Under steady traffic the map grows until the process hits its heap limit. The assistant did not warn about memory growth because unbounded growth is not a property of the code it wrote — it is a property of the workload.

**What to check.** Any assistant-generated cache needs four things stated explicitly: a maximum size, an eviction policy, a TTL, and an exported hit/miss counter. If the prompt does not specify them, the assistant will not invent them.

**How to measure whether it matters.** Instrument resident memory and cache entry count as gauges, then run a soak test at realistic request rate for at least the longest expected cache lifetime. Compare heap growth against a flat baseline. A cache that is working shows a plateau; an unbounded one shows a line with positive slope.

## Pattern 2: Assumed concurrency safety

Assistants reach for parallelism readily, because "make it faster" maps cleanly to `asyncio.gather` or a thread pool. The failure is that the parallelism is added at the call site while the limit lives somewhere else entirely.

```python
# Illustrative: parallelises I/O but ignores pool size
import asyncio

async def fetch_all(ids):
    return await asyncio.gather(*[db.query(i) for i in ids])
```

If the database connection pool is configured to 10 and `ids` has 500 entries, this opens 500 concurrent acquire attempts against a pool of 10. Depending on the driver, you get either a queue that times out or an error storm. The code is fine; the assumption that concurrency is free is not.

**What to check.** Before accepting any generated parallelism, answer three questions: what is the pool size, what is the downstream rate limit, and what happens when concurrency exceeds it? If any answer is "I don't know," the change is not ready.

**How to measure.** Load-test with a concurrency level above the pool size and watch the pool's wait-time metric. If wait time grows linearly with concurrency, you are queueing, not parallelising. Compare p99 latency at concurrency 10 and concurrency 100; a healthy system shows sublinear growth.

## Pattern 3: External dependency drift

Assistants are confident about standard-library contents and frequently wrong. A prompt that says "remove unnecessary dependencies" can produce a removal that is correct for the common case and wrong for the edge case.

```python
# Illustrative: stdlib replacement that drops edge-case handling
from datetime import datetime, timezone

def to_local(dt: datetime, tz_name: str) -> str:
    # zoneinfo handles historical and future offset transitions
    from zoneinfo import ZoneInfo
    return dt.astimezone(ZoneInfo(tz_name)).isoformat()
```

The general lesson: when an assistant proposes replacing a third-party library with a standard-library equivalent, the burden of proof is on the replacement. Timezone databases, date parsing, and Unicode normalisation are the usual places where "it's in the stdlib now" is only partly true.

**What to check.** For any dependency removal, diff the behaviour on the inputs your system actually sees, not on the happy path. Keep a fixture set of edge-case inputs — DST boundaries, leap seconds, malformed identifiers — and run both implementations against it before deleting the old import.

**How to measure.** Canary the change to a small percentage of traffic and compare error rate between canary and baseline. A rise in cold-start import errors is the signature of a bad removal.

## Pattern 4: Interface drift

This is the most expensive pattern because it escapes your service boundary. An assistant asked to "add a timestamp to the response" may also normalise the surrounding schema.

```python
# Illustrative: field added, constraint silently dropped
from pydantic import BaseModel, Field

class UserOut(BaseModel):
    id: int
    name: str = Field(max_length=100)
    created_at: str
```

A plausible generated variant drops `max_length` because it looks redundant next to the new field. Downstream clients that relied on the 100-character limit now send longer values and receive 400s from a different layer. The change is invisible in the diff unless you are reading for removals, not additions.

**What to check.** Review interface changes by diffing the generated schema against the previous one, not by reading the new code. Any removed constraint, renamed field, or changed optionality is a breaking change and needs a version bump or a compatibility shim.

**How to measure.** Contract tests that assert the exact serialised shape of responses catch this class of drift. Run them against the previous release's schema, not just the current code.

## A worked example: the CPF regex

This is a compact case that shows how a correct-looking generation fails a domain rule.

```python
import re

# Illustrative: format-only validation
cpf_pattern = re.compile(r'^(\d{3})\.?(\d{3})\.?(\d{3})-?(\d{2})$')

def is_valid_cpf(value: str) -> bool:
    return bool(cpf_pattern.match(value))
```

The regex accepts any eleven digits in the right shape. A Brazilian CPF also carries two check digits computed by a modulus-11 algorithm, so the regex accepts invalid identifiers and, depending on formatting assumptions, may reject valid ones. The assistant produced the pattern because the prompt asked for "CPF format validation" — format, not validity.

The fix is to add the checksum and keep the regex only as a preprocessing step:

```python
import re

CPF_PATTERN = re.compile(r'^(\d{3})\.?(\d{3})\.?(\d{3})-?(\d{2})$')

def _check_digit(digits: list[int]) -> int:
    total = sum(d * w for d, w in zip(digits, range(len(digits) + 1, 1, -1)))
    remainder = total % 11
    return 0 if remainder < 2 else 11 - remainder

def is_valid_cpf(value: str) -> bool:
    if not CPF_PATTERN.match(value):
        return False
    digits = [int(c) for c in re.sub(r'\D', '', value)]
    if len(set(digits)) == 1:
        return False  # reject repeated-digit placeholders
    first = _check_digit(digits[:9])
    second = _check_digit(digits[:9] + [first])
    return digits[9] == first and digits[10] == second
```

The reasoning to note: the assistant was not wrong about the format, it was wrong about the requirement. The requirement was never stated. This is the general shape of most AI-generated domain bugs — the model satisfies the literal prompt and the prompt omitted the constraint.

**How to measure.** Build a fixture set with known-valid and known-invalid identifiers, including repeated-digit placeholders and boundary check digits. Run it as a unit test. A regex-only implementation will fail a measurable fraction of the invalid cases; the checksum version should pass all of them.

## The async listener leak

A second compact example, this time in the cleanup path.

```javascript
// Illustrative: listener added per connection, never removed
socket.on('message', async (msg) => {
  const data = await processAsync(msg);
  socket.send(data);
});
```

Each connection registers a new listener on the shared emitter. Without a matching removal on disconnect, listeners accumulate and every message is processed once per accumulated listener. Memory grows with connection count, and throughput degrades quadratically.

The corrected version keeps the handler reference so it can be removed:

```javascript
function attach(socket) {
  const handler = async (msg) => {
    const data = await processAsync(msg);
    socket.send(data);
  };
  socket.on('message', handler);
  socket.on('close', () => socket.off('message', handler));
}
```

**How to measure.** Export a gauge for listener count per emitter, or use the runtime's built-in listener-count accessor. Run a connection churn test — connect and disconnect repeatedly — and assert the count returns to baseline. If it climbs monotonically, you have the leak.

## Guardrails that catch these patterns

The patterns above share a shape: plausible local code, violated global assumption. Guardrails should therefore test assumptions, not code.

**State the constraints in the prompt.** "Add a bounded cache with a 5-minute TTL, a 10,000-entry cap, and a hit-rate metric" produces different code from "add a cache." The assistant cannot infer limits you did not state.

**Diff interfaces, not implementations.** For any change that crosses a service boundary, compare the serialised schema before and after. Additions are usually safe; removals and type changes are not.

**Instrument the assumption.** Pool wait time, heap growth, listener count, cache entry count. These are the signals that fail first when an assistant has made an implicit assumption about scale.

**Canary dependency changes.** Any removal or replacement of a third-party library goes to a small traffic slice first, with error rate compared against baseline.

**Keep a domain fixture set.** The edge cases your business cares about — check digits, DST boundaries, locale formats — belong in tests, not in prompt text. Tests persist; prompts are per-session.

## Choosing where to apply review effort

Not every generated line deserves the same scrutiny. A rough triage:

| Change touches | Risk | Review approach |
|---|---|---|
| Comments, formatting, local variable names | Low | Skim |
| Pure functions with existing test coverage | Low | Let tests decide |
| New code with no test coverage | Medium | Write the test first, then review |
| Shared mutable state, caches, singletons | High | Instrument before merge |
| Concurrency, connection pools, rate limits | High | Load-test at production scale |
| Serialised interfaces, API schemas | High | Diff against previous schema |
| Dependency additions or removals | High | Canary and compare error rate |

The table is a heuristic, not a rule. The point is that review effort should track blast radius, and blast radius is a property of what the code touches, not how it was written.

## FAQ

**Does this mean AI-generated code is worse than human-written code?**
No. The failure modes are the same ones humans produce — unbounded caches, unremoved listeners, dropped constraints. The difference is rate. Assistants generate these faster than review can catch them, so the guardrails need to be automated rather than relying on reviewer attention.

**Do I need a multi-agent validation system?**
Not by default. Most of the patterns here are caught by instrumentation and schema diffs, which are cheaper and more reliable than a second model reviewing the first. Add model-based review only after you have baseline metrics and a fixture set to measure its false-positive rate against.

**How do I know if my assistant is making an implicit assumption?**
You generally don't, from the code alone. The signal is a change in a metric after deployment — pool wait time, heap growth, error rate. Instrument first, then attribute.

**Is it worth documenting domain quirks in prompts?**
It helps within a session but does not persist. The durable place for domain rules is a fixture set and a test. If a rule matters, it should fail a build when violated.

## What to do in the next 30 minutes

Pick one assistant-generated change currently in your codebase that touches shared state, concurrency, or a serialised interface. Diff it against the previous version and list every removed constraint, every new unbounded structure, and every added concurrent call. For each item, write down the assumption it depends on. If you cannot state the assumption, that change needs instrumentation before it ships.
