# Hire for agents, not autocomplete

Most engineering hiring processes optimise for a skill that production does not reward: producing correct code quickly in a clean, well-lit environment. Take-home tests are CRUD apps with a REST API. Interviews are whiteboard algorithms. Onboarding is a two-week sprint through documentation. Then the new hire meets production, where the network drops packets, the payment gateway retries aggressively, the logs arrive late, and the incident channel is already on fire.

The mismatch is not about seniority. It is about what the hiring pipeline measures. A candidate can be excellent at writing clean code and still be ineffective on a system where the dominant constraint is partial information and degraded infrastructure. This article covers how teams reframe hiring and onboarding around that reality: what to test, how to simulate constraints, what code to use, and where the approach breaks down.

## The gap between what interviews measure and what production demands

A typical failure mode looks like this. A team hires strong engineers from companies with reliable fibre, fast IDEs, and predictable CI. The engineers pass a rigorous take-home test. Onboarding begins. Within weeks, their pull requests are stuck in review loops, their features pass unit tests but fail in staging, and their questions cluster around the same handful of production behaviours: why a webhook retries, why a request times out on a slow connection, why a refund gets processed twice.

None of those failures are visible in a CRUD take-home test. They are visible only when the environment is hostile. The skills that matter under those conditions are:

- Debugging with incomplete information, often over a high-latency SSH session.
- Recognising when retry logic is the bug, not the fix.
- Designing for idempotency and backpressure before writing the happy path.
- Shipping a partial fix under a deadline rather than a perfect fix after it.

These are agentic skills in the sense that they require autonomy and judgment, not just code generation. A candidate who can write a clean endpoint but cannot diagnose a retry storm is not yet ready for a production on-call rotation. The hiring process should surface that gap before the offer letter, not after.

## What constraint-aware hiring looks like in practice

The shift is from abstract problem-solving to problems that mirror real traffic. Two patterns recur across teams that have made this change.

The first is a simulated incident as the take-home. Instead of a CRUD app, candidates receive a broken service, a staging environment with throttled bandwidth, and a written scenario. They have a fixed window, often 60 to 90 minutes, to identify the root cause, patch it, and prove the fix. The proof matters: a passing local test is not evidence. A captured request that returns the correct status code under simulated latency is.

The second is a live debugging session. Candidates get a broken webhook handler, a script that injects packet loss and latency, and a requirement to ship a fix within a short window. Strong performers tend to do three things: fix the immediate bug, add a guard against recurrence (a circuit breaker, a deduplication key, a queue), and write a short post-mortem. Weak performers edit the code, run the local tests, and assume the fix works. Their pull request breaks in staging because they never exercised it under load.

Onboarding mirrors the same principle. A generic two-week sprint through documentation becomes a structured constraint bootcamp: debug a failing CI job under a tight timeout, optimise an endpoint so it returns within a budget on a slow connection, ship a feature on a staging environment with no fibre fallback. The goal is not to teach tools. It is to build muscle memory for shipping when the environment is unreliable.

Teams also replace generic onboarding checklists with failure-mode checklists. Instead of "read the docs," the list contains real production failures and their fixes: how to debug a webhook stuck in a retry loop, how to keep a mobile-money push from timing out on a slow network, how to recover a transaction when the user's session expires. These are written by engineers who have hit the failure, not by product managers, and the tone is direct.

## A worked example: the refund endpoint take-home

The following is an illustrative take-home problem. The numbers are chosen to be plausible and are labelled as such; they are not measured results.

The candidate receives a repository containing a FastAPI refund endpoint, retry logic that fires every two seconds without deduplication, a test suite that passes locally but fails under CI's timeout, and instructions: the staging environment simulates 3G latency and packet loss. Ship a fix that deduplicates refunds, returns 200 OK within a budget, logs the refund ID and timestamp, and passes CI.

The repository includes a script that wraps an HTTP client with latency and packet loss:

```python
# simulate_3g.py
import httpx
import random
import asyncio

async def slow_http_client():
    transport = httpx.AsyncHTTPTransport(retries=3)
    async with httpx.AsyncClient(
        transport=transport,
        timeout=httpx.Timeout(15.0),
    ) as client:
        response = await client.post(
            "http://localhost:8000/refunds/123",
            json={"amount": 100},
        )
        return response

async def simulate_3g():
    if random.random() < 0.05:
        raise httpx.ReadTimeout("Simulated timeout")
    await asyncio.sleep(random.uniform(0.3, 1.2))
    return await slow_http_client()
```

Note two corrections relative to naive drafts of this script. `httpx.AsyncHTTPTransport` does not accept `http2` or `network_backoff_factor` arguments; retry behaviour belongs in the transport's `retries` parameter or in an explicit retry policy, and HTTP/2 is negotiated by the server and client configuration rather than forced here. Keeping the script honest matters, because a candidate who trusts a broken simulator learns the wrong lesson.

The expected solution has four parts:

1. A deduplication layer keyed on the transaction ID, stored in Redis with a short TTL.
2. A circuit breaker around the outbound refund call so a failing gateway does not trigger unbounded retries.
3. Structured logging of the refund ID and timestamp.
4. A response that returns within the stated budget even when the downstream call is slow.

Grading is on observable behaviour, not on whether the candidate used a particular library. The questions to ask are: does the endpoint return the correct status under simulated latency, does a repeated request produce a single refund, and does the log line contain the fields needed to reconstruct what happened?

### How to measure the outcome rather than assert it

Rather than reporting a pass rate, instrument the test. Record, for each candidate:

- Whether the endpoint returned within the budget under `simulate_3g.py`, measured by the client's elapsed time.
- Whether two identical requests produced one refund or two, checked against the datastore.
- Whether the CI job completed within its timeout, read from the CI log.
- Whether the fix includes a guard against recurrence, identified by reading the diff.

Compare cohorts by these four booleans. If the constraint-aware version of the test filters more candidates than the previous version, that is a signal about the test, not proof that the previous cohort was better. The useful comparison is between candidates who passed the constraint test and their subsequent on-call performance, which requires tracking new hires past their first quarter.

## Simulating constraints honestly

The most common mistake in constraint-aware hiring is a simulator that does not resemble the constraint. Browser-based network throttling changes bandwidth and latency but does not reproduce jitter, packet loss, or connection resets. A candidate who passes under browser throttling may still fail on a real mobile network.

On Linux, `tc` (traffic control) with the `netem` queueing discipline can approximate a lossy, high-latency link:

```bash
# Apply 5% packet loss and 300ms latency with 100ms jitter to eth0.
# Requires root and the sch_netem kernel module.
tc qdisc add dev eth0 root netem loss 5% delay 300ms 100ms
```

To remove the rule afterwards:

```bash
tc qdisc del dev eth0 root
```

This is illustrative configuration, not a measured network profile. The loss and delay values should be chosen to match the worst realistic condition for the product's users, and the resulting behaviour should be verified with a client that reports elapsed time and error types.

For SSH-based exercises, connection options can approximate a slow link without any traffic shaping:

```bash
# Constrain SSH connect and keepalive behaviour to mimic a slow link.
ssh -o ConnectTimeout=30 -o ServerAliveInterval=10 -o ServerAliveCountMax=3 user@staging-host "tail -f /var/log/app.log"
```

The honest framing for all of these tools is that they are approximations. A simulator that is too gentle produces false positives. A simulator that is too harsh produces false negatives and burns out candidates. The way to calibrate is to run the same exercise on an engineer who already ships to the constrained environment and confirm the exercise is passable in the allotted time.

## A worked example: the webhook handler

A second illustrative exercise uses a broken Flutterwave-style webhook handler that times out on slow connections:

```javascript
// webhook.js (broken)
app.post('/webhook', async (req, res) => {
  const { transaction_id } = req.body;
  try {
    await refundTransaction(transaction_id);
    res.status(200).send('OK');
  } catch (err) {
    // Retry aggressively
    setTimeout(() => refundTransaction(transaction_id), 1000);
    res.status(500).send('Retrying');
  }
});
```

The failure modes are visible on inspection: the retry is unbounded, it is not deduplicated, and it responds 500 to the caller while the retry runs, which invites the caller to retry as well. Under a slow network, this produces a retry storm and duplicate refunds.

A candidate's fix should introduce a bounded failure path. One shape is a circuit breaker around the outbound call plus a queue for deferred work:

```javascript
// webhook.js (fixed)
import CircuitBreaker from 'opossum';
import { Queue } from 'bullmq';

const refundQueue = new Queue('refunds', { connection: redisConnection });
const circuit = new CircuitBreaker(refundTransaction, {
  timeout: 100,
  errorThresholdPercentage: 50,
  resetTimeout: 30000,
});

app.post('/webhook', async (req, res) => {
  const { transaction_id } = req.body;
  try {
    await circuit.fire(transaction_id);
    res.status(200).send('OK');
  } catch (err) {
    await refundQueue.add('refund', { transaction_id });
    res.status(202).send('Queued');
  }
});
```

This is not the only correct answer. A candidate who uses a different circuit breaker library, or who implements idempotency keys and an explicit retry budget, may be equally correct. The graded behaviours are: the caller is not told to retry while work is pending, the retry is bounded, and the refund is deduplicated. The queue worker must also be idempotent, or the deduplication problem simply moves downstream.

## Failure modes in the hiring process itself

**Assuming the simulation is accurate.** A simulator that uses browser throttling will pass candidates who cannot handle real packet loss. The correction is to validate the simulator against a known-constrained link and to include at least one exercise where the candidate must read logs over a high-latency connection.

**Over-optimising for the wrong constraint.** Some candidates will shave milliseconds off a response while leaving the retry logic untouched. Grade the failure path, not just the happy path. A useful prompt is: "what happens if the downstream call never returns?"

**Assuming staging reflects production.** If staging runs on reliable infrastructure and production does not, the bootcamp teaches the wrong environment. Either shape staging to match, or make the constraint explicit in the exercise.

**Ignoring the human cost.** Debugging under degraded conditions for a full day is exhausting. Limit constraint exercises to a few hours, and pair the candidate or new hire with a mentor for the rest of the time. A hiring process that burns out the people it selects for is not a process worth running.

**Assuming the process scales linearly.** Constraint bootcamps need mentors. A rough planning figure is one mentor per two participants; if that ratio cannot be met, reduce the cohort size rather than diluting the supervision.

## When this approach is the wrong choice

Constraint-aware hiring and onboarding is not universally correct. It is a poor fit when:

- The product's users are on reliable, high-bandwidth connections and the stack is homogeneous. Simulated packet loss will feel artificial and will not predict on-the-job performance.
- The team is very small. A structured bootcamp requires mentorship time that a team of a few engineers may not have. Pair programming and a short failure-mode checklist are better first steps.
- Hiring volume is low. Rewriting a take-home test and building a simulation environment is a fixed cost that is hard to justify for a handful of hires per year.
- The culture resists constraint-first thinking. If the team expects reliable infrastructure and treats degraded conditions as an anomaly, a full bootcamp will meet resistance. Start with a single 30-minute live debugging session and see whether it surfaces useful signal.

## Tools and libraries

The table below lists tools used in the exercises above, with the caveat that version numbers change and should be pinned to whatever the team's own environment supports.

| Tool | Category | Use in the exercise |
|---|---|---|
| FastAPI | Python web framework | Scaffold the refund endpoint |
| httpx | Async HTTP client | Simulate slow outbound calls |
| pytest | Test runner | Run the constraint-aware test suite |
| Redis | Key-value store | Deduplication, queues, short-lived state |
| opossum | Circuit breaker (Node.js) | Bound retries in the webhook handler |
| BullMQ | Redis-backed queue (Node.js) | Defer refund work off the request path |
| structlog | Structured logging (Python) | Log refund ID and timestamp |
| tc / netem | Linux traffic control | Approximate packet loss and latency |
| GitHub Actions or equivalent | CI | Enforce a timeout budget on the test suite |
| Vim or any terminal editor | Editor | Read logs over a slow SSH session |

The most commonly overlooked item is `tc`. It is available on most Linux hosts, requires no additional hardware, and can approximate a lossy mobile link closely enough to change candidate behaviour. The most commonly over-trusted item is browser-based throttling.

## A decision checklist

Before adopting constraint-aware hiring, answer these questions:

1. What is the worst realistic network condition for the product's users, and can it be described in numbers (latency, loss, bandwidth)?
2. Does the current take-home test exercise that condition at all?
3. Is there an engineer on the team who already ships under that condition and can calibrate the exercise?
4. What will be measured, and how will it be recorded? (Elapsed time, duplicate count, CI duration, presence of a recurrence guard.)
5. How many mentors are available, and what cohort size does that support?
6. How will the process be evaluated after the first cohort, and against what outcome?
7. What is the plan if the constraint-aware test filters candidates who would have succeeded?

If the answer to question 3 is no, the exercise cannot be calibrated and should not be used for decisions. If the answer to question 6 is "we'll see," the process will drift back toward what is easy to measure.

## What to do in the next 30 minutes

Pick one real failure mode from your own system, write it down as a single sentence, and add it to the next take-home test as a required behaviour. For example: "A repeated webhook delivery must produce exactly one refund." Then run the existing test suite under a constrained link using `tc` on a disposable host, and record the elapsed time and the duplicate count. That one measurement tells you whether your current test exercises the constraint at all, and it is the smallest step that produces real signal.
