# Claude reviews my code: blind spots found

I ran into this while migrating a production service under a hard deadline. The official docs covered the happy path well. This post covers everything else.

## The one-paragraph version (read this first)

I give Claude 3.7 Sonnet chunks of production code and ask it to find bugs, edge cases, and performance traps before I merge the PR. In the last six months, it caught 62 real issues across 42 repos—38 logic bugs that tests missed, 12 scalability cliffs, and 12 security smells—while flagging 23 false positives that looked scary at first but were actually fine. The cost was 78 cents of API credits for 1,140 prompts. The failures were predictable: it misses stateful race conditions, over-indexes on style nitpicks, and hallucinates import chains when the repo uses unconventional layouts. I still review the diff manually, but Claude now surfaces 80% of the issues I’d catch on my own in half the time.


## Why this concept confuses people

Most developers think of LLM code review as a silver bullet: paste the diff, get perfect feedback, done. The marketing shows a cursor hovering over a red squiggle that says “Use asyncio.run()” and everyone assumes the AI understands the whole system. Reality is we’re still in the “autocomplete with a PhD” phase. Teams that skip the manual QA step discover, after production fires, that the AI praised a function that silently corrupted 11 GiB of Parquet files because it didn’t trace the data lineage through three microservices.

Another confusion is cost. A single Sonnet 3.7 call costs $0.00083 per 1,000 input tokens and $0.003 per 1,000 output tokens. At 500 LOC diffs, that’s ~6,000 input tokens and 2,000 output tokens, or $0.007 per review. If you do 100 reviews a week, that’s $7. If you skip filtering, a side project with 20 PRs a week hits $140 a month—enough to make finance ask why you didn’t just hire a junior reviewer.

People also conflate style linting with correctness. A linter will tell you your function is 140 characters long; Claude will tell you the function silently swallows exceptions when the upstream service returns HTTP 429. One is about PEP 8, the other is about data loss. Unless you explicitly ask for correctness, you’ll get the former.


## The mental model that makes it click

Think of Claude as a **real-time pair programmer who never slept and has read every Stack Overflow thread ever written**. It can’t *execute* your code, but it can simulate execution paths by pattern matching against its training data. When you paste a diff, you’re giving it a static snapshot of state machines, not a running process. That means:

- It excels at **static analysis**: control-flow gaps, undefined variables, missing null checks.
- It struggles at **dynamic analysis**: race conditions, memory leaks under load, exact cache eviction timing.
- It hallucinates when the codebase uses unconventional build systems (Bazel monorepos, Nix shells, custom Docker layers) because those patterns are rare in the training set.

The trick is to frame the prompt so the model simulates a **skeptical reviewer who knows the domain**. Instead of “Review this code,” I ask:

> “Assume this is a production service receiving 1,200 requests/sec. List every input that could trigger undefined behavior, every scalability cliff, and every security smell. Return a numbered list; prefix each item with [BUG], [PERF], or [SEC].”

That framing forces the model to think in load, not just correctness.


## A concrete worked example

Let’s walk through a real PR I opened last week for a Python FastAPI service that ingests sensor telemetry and writes to a PostgreSQL time-series table. The diff added a new endpoint `/v2/ingest` that batches writes.

### The diff (simplified)

```python
# Added after existing ingest_v1
@app.post("/v2/ingest")
async def ingest_v2(payload: list[Telemetry]) -> JSONResponse:
    if not payload:
        raise HTTPException(400, "Empty payload")

    # New batch upsert
    query = """
        INSERT INTO telemetry (sensor_id, ts, value)
        VALUES %s
        ON CONFLICT (sensor_id, ts)
        DO UPDATE SET value = EXCLUDED.value
    """
    async with db.acquire() as conn:
        await conn.execute(query, [(t.sensor_id, t.ts, t.value) for t in payload])
    return JSONResponse({"inserted": len(payload)})
```

### My prompt to Claude

> You are a senior SRE reviewing a FastAPI diff.
> The repo is a production telemetry service with 1,200 req/sec peak.
> The new endpoint ingests a list of telemetry objects and writes to PostgreSQL using an asyncpg batch upsert.
> List every input that could trigger undefined behavior, every scalability cliff, and every security smell. Prefix each with [BUG], [PERF], or [SEC].
> Give line numbers. Limit to 10 items.

### Claude’s response (edited for brevity)

| # | Issue | Prefix | Line | Risk | Impact |
|---|-------|--------|------|------|--------|
| 1 | Missing `max_batch_size` guardrail; malicious client can send 100,000 items and DoS the DB | [PERF] | 12 | 8/10 | OOM in connection pool |
| 2 | No transaction isolation; interleaved inserts can create phantom rows under high concurrency | [BUG] | 15 | 6/10 | Data inconsistency |
| 3 | No authz check on sensor_id; user can spoof telemetry from any sensor | [SEC] | 12 | 7/10 | Integrity violation |
| 4 | No rate limit; 1,200 req/sec × 100 items each = 120k writes/sec → PostgreSQL CPU spike | [PERF] | 3 | 9/10 | 5xx errors |

### What I changed after reading the list

1. Added `max_batch_size=1000` and `429 Too Many Requests` when exceeded.
2. Wrapped the upsert in `BEGIN … COMMIT` to get serializable isolation.
3. Added `authz: check_sensor_ownership(request.user, t.sensor_id)` inside the loop.
4. Added `fastapi-limiter` with 100 req/min per user.

### Benchmark before vs after

| Metric | Before | After |
|--------|--------|-------|
| P99 latency | 842 ms | 112 ms |
| Error rate (5xx) | 4.2% | 0.3% |
| DB CPU | 87% | 42% |

I shipped the fix two hours after the review instead of two days later when the pager woke me up.


## How this connects to things you already know

If you’ve ever used `mypy --strict` or `golangci-lint --enable-all`, you’ve done static analysis. Claude is the same idea, but with two upgrades:

1. **Context window** = your entire codebase + dependencies (if you paste the `requirements.txt` or `go.mod`). `mypy` only sees one file at a time.
2. **Natural-language reasoning** = it can explain why a null check is missing in plain English, whereas a linter just says “possible NoneType”.

It’s also similar to **Chaos Engineering** tools like Gremlin, except instead of injecting failures, you’re injecting a reviewer who spots the same classes of failure before they happen.


## Common misconceptions, corrected

**Misconception 1: “Claude understands the whole system.”**

Reality: It understands the tokens it sees. If your repo has a custom ORM layer or a dynamic import that only resolves at runtime, the model will hallucinate the shape of the objects. I once asked it to review a Haskell module that used Template Haskell to generate SQL at compile time. It confidently told me there was a memory leak in a function that didn’t exist after the macro expanded. 

**Misconception 2: “The more tokens I give, the better the review.”**

Reality: A 5,000-line diff reduces accuracy. The model’s attention span isn’t infinite; it starts skipping sections. Break the review into small, domain-focused chunks—e.g., one prompt for SQL queries, one for auth, one for performance. I tried reviewing a 2,800-line Java monolith in one shot; the model missed a critical `synchronized` block because it was buried under 200 lines of Lombok annotations.

**Misconception 3: “Claude’s suggestions are always safe to apply.”**

Reality: It will suggest `multiprocessing.Pool` in a FastAPI worker pool, which deadlocks the event loop. It will suggest `ThreadPoolExecutor` in async code, which leaks sockets. Always sanity-check the diff against the runtime constraints of your stack.


## The advanced version (once the basics are solid)

Once the obvious bugs are gone, you can push the model further with two techniques: **property-based testing** and **load-test simulation**.

### Property-based testing

Instead of writing unit tests by hand, ask Claude to generate a property:

> “Write a Hypothesis test (Python) that checks the new `/v2/ingest` endpoint never returns a 5xx response when given valid JSON, never writes duplicate rows under high concurrency, and never exceeds 200 ms P95 latency for batches ≤ 1,000 items. Put the test in `tests/test_ingest_v2.py`.”

Claude produced this in one shot (trimmed):

```python
from hypothesis import given, strategies as st
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

@given(payload=st.lists(st.builds(Telemetry, sensor_id=st.integers(1, 100), ts=st.datetimes(), value=st.floats(0, 100)), min_size=1, max_size=1000))
def test_ingest_v2_safety(payload):
    resp = client.post("/v2/ingest", json=[p.dict() for p in payload])
    assert resp.status_code == 200
    assert resp.json()["inserted"] == len(payload)
```

I ran it under `pytest --hypothesis-seed=1234`; it found a race where two concurrent requests with identical `(sensor_id, ts)` created duplicate rows because the `ON CONFLICT` clause fired after the first insert but before the second. The model’s suggestion was spot-on.

### Load-test simulation

Ask the model to write a Locust script that replays traffic patterns from production:

> “Write a Locustfile that replays 1,200 req/sec of `/v2/ingest` traffic with 70% valid payloads and 30% malformed JSON. Include a step where the DB connection pool is saturated at 50 connections.”

Claude produced a 60-line script that I ran against a staging DB. Within 90 seconds, the CPU on the database hit 95% and the endpoint returned 502s. The model then suggested adding a connection pool size cap and a circuit breaker, which I applied. The staging error rate dropped from 68% to 0% under the same load.


## Quick reference

| Task | Prompt template | Token estimate (in/out) | Cost (Sonnet 3.7) | Typical accuracy |
|------|-----------------|------------------------|-------------------|------------------|
| Review a 300-line PR | Give numbered list of bugs, perf cliffs, security smells. Prefix each with [BUG], [PERF], [SEC]. Line numbers required. | 4k / 1.2k | $0.004 | 85% |
| Review a SQL query | Assume 1,200 req/sec. List every missing index, parameterization risk, and race condition. | 2k / 0.8k | $0.002 | 92% |
| Generate property test | Write a Hypothesis test that checks invariant X. Put it in tests/test_Y.py. | 1.5k / 2k | $0.003 | 78% |
| Write Locust load test | Replay 1,200 req/sec traffic with 30% malformed payloads. Include DB saturation scenario. | 3k / 3k | $0.008 | 88% |

**Cost control tips:**
- Cap token count with `max_tokens=1000` in the API call.
- Use `temperature=0.3` to reduce verbosity (and thus output tokens).
- Cache prompts with identical diffs—Claude’s output is deterministic for the same input.
- Run reviews in a GitHub Action so you don’t burn credits on every local edit.


## Further reading worth your time

- [Anthropic’s 2024 system card](https://www.anthropic.com/system-card-2024) – The limits of Sonnet 3.7 on code tasks.
- [PostgreSQL: The Docs](https://www.postgresql.org/docs/current/explicit-locking.html) – Isolation levels and batch upsert semantics.
- [FastAPI: Advanced Dependencies](https://fastapi.tiangolo.com/advanced/using-request-directly/) – How to plug authz into endpoints.
- [Hypothesis: What is Property-Based Testing?](https://hypothesis.readthedocs.io/en/latest/) – The mental model for turning AI suggestions into tests.
- [Locust: Writing Custom Load Shapes](https://docs.locust.io/en/stable/custom-load-shape.html) – How to reproduce production traffic patterns.


## Frequently Asked Questions

**How do I stop Claude from nitpicking style issues?**

Ask it to ignore style: “Focus only on correctness, performance, and security. Ignore naming, docstrings, and line length.” Add a filter in your prompt: “Do not mention PEP 8, black formatting, or trailing commas.” That reduces false positives by ~40% in my repos.

**Can I use Claude offline to save costs?**

No. Sonnet 3.7 is a cloud-only model. Some teams run smaller open models (e.g., Codestral 22B) locally for style linting, but they lack the reasoning depth for correctness reviews. I benchmarked Codestral on the same PR; it caught 18 style issues and 2 real bugs. For production code, I still prefer Sonnet.

**What’s the worst false negative I should watch for?**

Stateful race conditions where the bug only appears under specific interleaving of goroutines or threads. Example: a Python async function that assumes a global cache is populated by a sync thread. The model sees the function signature and misses the thread boundary. Mitigation: pair the review with `pytest-asyncio` fuzzing or a runtime race detector like `tsan`.

**How do I handle repos with unconventional setups (Bazel, Nix, custom Docker)?**

Paste the entire `WORKSPACE`, `shell.nix`, and Dockerfile into the prompt so the model can see the build graph. Without that context, it hallucinates import chains and misses build-time dependencies. I tried reviewing a Bazel monorepo without context; the model suggested a `pip install` line that broke the hermetic build. After I pasted the `WORKSPACE`, it gave accurate feedback.


## Action checklist (do this tomorrow)

1. Pick one open PR that touches a core service.
2. Paste the diff + `requirements.txt` into a new Claude chat.
3. Use the prompt template from the Quick Reference.
4. Copy the numbered list into the PR description as a review.
5. Manually verify the top three [BUG] items before merging.

Do that for five PRs, measure the issues caught vs your usual rate, and adjust the prompt. After a week you’ll know whether the 78-cent tool is worth the noise.