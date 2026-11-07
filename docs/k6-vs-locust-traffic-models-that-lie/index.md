# k6 vs Locust: traffic models that lie

Load testing tools are easy to run and hard to trust. Two teams can point k6 and Locust at the same staging URL, get wildly different p95 numbers, and both be technically correct — because the traffic model, not the tool, determines whether the result means anything. If scripted virtual users (VUs) think in lockstep, the latency curve is a fiction. If the arrival rate is open-loop but think time is zero, the test measures queueing at the server, not user experience. The problem worth solving is how to design a traffic model that survives the gap between "the test passed" and "production fell over."

## The gap between virtual users and real arrivals

Most load tests start with a closed-loop model: N virtual users, each running a loop of requests with some think time between iterations. k6's default `vus` + `duration` execution is exactly this. Locust's default `User` class with `wait_time` is the same idea. The model is intuitive — "simulate 500 users" — but it hides a coupling that real traffic does not have: in a closed loop, a slow response makes the user wait, which reduces the request rate. The system self-throttles. In an open-loop model, requests arrive at a fixed rate regardless of how slow the server is, so a slow server builds a queue, and the queue is what the p99 actually measures.

This distinction is not academic. A closed-loop test against a service that starts queuing will show flat latency and declining throughput — fewer requests per second, which looks like saturation, but not the latency explosion a real user experiences when many mobile clients all retry at once. The closed-loop test is a lie of omission: it cannot represent a thundering herd because the herd is politely waiting its turn.

Open-loop models are the correct default for any user-facing endpoint behind a CDN or mobile network, because real arrivals do not care whether the server is ready. k6 exposes this through the `constant-arrival-rate` and `ramping-arrival-rate` executors. Locust exposes it through `constant_throughput` and `constant_pacing` wait-time strategies, plus `--processes` and distributed modes for generating enough load. The trap is that both tools let you write a closed-loop script in five lines and an open-loop one in fifteen, so the five-line version wins by default — and that is the version that gets committed.

## What a traffic model actually contains

A traffic model is four decisions, not one. Teams commonly collapse them into "how many users" and then wonder why the test does not predict production.

1. **Arrival process.** Poisson (memoryless, realistic for independent users), uniform (easier to reason about, harsher on queues), or bursty (diurnal peaks, retry storms, marketing blasts). k6's arrival-rate executors approximate a Poisson process by default when you set `rate` and `timeUnit`; Locust's `constant_throughput` gives you a fixed rate, not Poisson, unless you add jitter yourself.
2. **Session shape.** How many requests per user session, in what order, with what branching. A login-then-browse-then-checkout flow is not three independent endpoints; it is a state machine, and the state machine is where realistic tests diverge from `http.get('/')` in a loop.
3. **Think time.** The pause between requests. Zero think time is a stress test, not a user model. On mobile networks, think time is dominated by radio state transitions and screen reading, not by the server. A common planning assumption is 2–10 seconds between page views for a browsing session — labelled as an assumption, not a measurement.
4. **Data distribution.** Which product IDs, which user accounts, which payload sizes. If every virtual user hits the same cached object, the test exercises the cache, not the database. A realistic model samples from a distribution that matches access logs — and if there are no access logs, the distribution is a guess, which is fine as long as it is labelled as one.

The tool choice (k6 vs Locust) matters far less than whether these four decisions are made explicitly. k6 is a Go binary with a JavaScript scripting API; Locust is Python with a gevent-based runner. The performance characteristics of the load generator itself differ, but at the request rates most teams test (hundreds to low thousands of RPS), both will saturate the target before they saturate themselves. The real difference is in how each tool expresses the arrival model.

| Concern | k6 | Locust |
|---|---|---|
| Closed-loop default | `vus` + `duration` | `User` + `wait_time` |
| Open-loop arrival | `constant-arrival-rate`, `ramping-arrival-rate` executors | `constant_throughput`, `constant_pacing` wait strategies |
| Scripting language | JavaScript (ES modules) | Python |
| Distributed generation | k6 operator or multiple instances with `--out` | `--master` / `--worker` built in |
| Thresholds as code | `thresholds` in options, exit code on breach | Custom `@events.quitting` listener or external check |
| Reuse of app code | None (separate JS runtime) | Can import Python modules from the app |

The last row decides many teams: if the service is Python and the test wants to reuse serializers or auth helpers, Locust removes a class of drift between test and app. If the requirement is a single static binary in CI with thresholds that fail the build, k6 is the shorter path. Neither is universally better; the model is what is being tested.

## Worked example: an open-loop model in k6

The following k6 script models a browsing session with a fixed arrival rate, a think time, and a sampled product ID. It is self-contained and runnable against any HTTP endpoint that returns 200.

```javascript
import http from 'k6/http';
import { sleep } from 'k6';
import { randomIntBetween } from 'https://jslib.k6.io/k6-utils/1.4.0/index.js';

export const options = {
  scenarios: {
    browse: {
      executor: 'constant-arrival-rate',
      rate: 50,            // 50 iterations started per second
      timeUnit: '1s',
      duration: '5m',
      preAllocatedVUs: 100,
      maxVUs: 500,
    },
  },
  thresholds: {
    http_req_failed: ['rate<0.01'],
    http_req_duration: ['p(95)<800', 'p(99)<2000'],
  },
};

export default function () {
  const productId = randomIntBetween(1, 5000);
  const res = http.get(`https://example.test/api/products/${productId}`);
  if (res.status !== 200) {
    console.error(`unexpected status ${res.status} for product ${productId}`);
  }
  // Think time: 2-6 seconds, labelled as an assumption, not a measurement.
  sleep(randomIntBetween(2, 6));
}
```

Two details matter more than the syntax. First, `preAllocatedVUs` and `maxVUs` are not the load — they are the worker pool that services the arrival rate. If the target slows down, k6 needs more VUs to keep starting 50 iterations per second, and if it runs out, it reports dropped iterations. That dropped-iteration count is a signal, not a failure of the tool: it means the worker pool could not keep up with the open-loop arrivals, which is exactly the queueing behaviour the test is trying to expose. Second, the `sleep` is inside the iteration, so the arrival rate is the rate of *new sessions*, not the rate of requests. At 50 iterations per second with an average of 4 seconds of think time, the test holds roughly 200 concurrent sessions — the arithmetic is 50 × 4 = 200, illustrative, and depends entirely on the assumed think time.

## The same model in Locust

Locust's equivalent uses `constant_throughput` to fix the per-user request rate, but the cleanest way to get an open-loop arrival rate is to combine a fixed user count with `constant_pacing`, or to use `constant_throughput` and accept that the user count becomes the concurrency ceiling. The following is a minimal self-contained example.

```python
from locust import HttpUser, task, between
import random

class BrowseUser(HttpUser):
    # between(2, 6) means each user waits 2-6 seconds between tasks (closed-loop).
    wait_time = between(2, 6)

    @task
    def browse(self):
        product_id = random.randint(1, 5000)
        with self.client.get(f"/api/products/{product_id}", catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"unexpected status {resp.status_code}")
```

Run it with `locust -f locustfile.py --headless -u 200 -r 20 --run-time 5m --host https://example.test`. The `-u 200` is the concurrency ceiling, and `-r 20` is the spawn rate. Because `wait_time` is closed-loop, the actual request rate will fall if the server slows down — which is the closed-loop lie described earlier. To get open-loop behaviour in Locust, use `constant_throughput`:

```python
from locust import HttpUser, task, constant_throughput
import random

class BrowseUser(HttpUser):
    wait_time = constant_throughput(0.25)  # 0.25 tasks per second per user = 1 every 4s

    @task
    def browse(self):
        product_id = random.randint(1, 5000)
        with self.client.get(f"/api/products/{product_id}", catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"unexpected status {resp.status_code}")
```

With `-u 200` and `constant_throughput(0.25)`, the target rate is 200 × 0.25 = 50 task starts per second, matching the k6 example. The arithmetic is shown so it can be checked; the numbers are illustrative and should be replaced with your own assumptions. The difference from k6 is that Locust will *not* spin up extra workers to maintain the rate if the server slows — it will simply fall behind. That is a real limitation of the closed-loop user model, and it is why Locust users who care about open-loop behaviour often run multiple worker processes and monitor actual RPS against the target.

## Where both tools mislead you

**The coordinated omission problem.** If the load generator waits for a response before scheduling the next request, and the server is slow, the generator under-samples the slow period. This is the classic coordinated omission bug, and it affects any closed-loop model. The fix is to schedule requests on a fixed timeline independent of responses — which is what arrival-rate executors do. If a closed-loop script is in use, its p99 is optimistic by construction. Say so in the report.

**The cache warm-up illusion.** A test that runs for 60 seconds against a cold cache measures cache misses; a test that runs for 10 minutes measures the steady state. Neither is wrong, but the p95 of the first minute is not the p95 a user sees at 9am on a Monday. Teams commonly run short tests, see good numbers, and ship. A useful rule is to discard the first 20–30% of the run as warm-up and report the steady-state window separately — labelled as a methodology choice, not a measurement.

**The single-region blind spot.** k6 and Locust both run from wherever they are launched. If the generator runs from a CI runner in one region and users are on mobile networks in three countries, the latency numbers describe the data-centre path, not the user path. For a Nigerian or East African user base, the dominant latency is often the radio link and international transit, not the application. A load test without a network emulation layer (tc netem, or a proxy with added latency) will systematically under-report the user-perceived p95. The tool is not lying; the model is missing a hop.

**The retry storm you did not script.** Real clients retry. Mobile SDKs retry. Payment callbacks retry. If the traffic model has no retry behaviour, it tests a world where nothing goes wrong, which is the one world production is never in. A realistic model includes a retry fraction — even a small one, labelled as an assumption — because retries are what turn a 2x slowdown into a 10x queue.

## Reading the results without fooling yourself

A load test produces three numbers that matter: the arrival rate actually achieved (not the one configured), the latency distribution at that rate, and the error rate broken down by status code. The first is the one teams skip. If 50 iterations per second were configured and k6 reports dropped iterations, the achieved rate is lower, and the latency measured is at that lower rate. Reporting p95 without the achieved rate is like reporting a car's fuel efficiency without saying how fast it was going.

A useful habit is to plot latency against achieved RPS, not against time. The curve usually has a knee — a point where latency rises faster than throughput. That knee is the capacity number that actually matters, and it is invisible in a time-series chart that just shows a flat line then a spike. Both k6 and Locust can export raw results (k6 to JSON or CSV via `--out`, Locust to CSV via `--csv`), and the knee is easiest to see when a ramp is run and the results plotted as a scatter.

Finally, treat thresholds as hypotheses, not gates. A `p(95)<800` threshold that passes on a laptop against localhost says nothing about a 3G user in Accra. The threshold should be derived from a user-facing budget — for example, a checkout page that must render within 3 seconds on a median mobile connection — and then back-calculated into a server-side budget with the network hop subtracted. That subtraction is the step most teams skip, and it is why a passing load test and a failing product launch can coexist.

## Frequently Asked Questions

**How many virtual users should a load test run?**

There is no universal number; the right question is what arrival rate needs modelling and what concurrency that implies. If the assumed peak is 100 requests per second and an average session is 5 requests over 10 seconds, the concurrency is roughly 100 × 10 / 5 = 200 sessions — arithmetic shown so it can be substituted with other assumptions. Size the generator to that concurrency plus headroom, and watch for dropped iterations as the signal that the pool is too small.

**Is k6 or Locust better for open-loop load testing?**

k6's arrival-rate executors are a first-class part of the execution model, so open-loop is the path of least resistance. Locust can do open-loop via `constant_throughput`, but the user model is closed-loop by default and the generator will fall behind rather than spin up workers. If open-loop fidelity is the priority, k6 is the shorter path; if reusing Python application code is the priority, Locust wins on a different axis.

**Why does a load test show low latency but production is slow?**

The most common causes are coordinated omission (a closed-loop model under-samples slow periods), a missing network hop (the generator is closer to the server than users are), and a warm cache (the test ran long enough to cache everything). Check the achieved arrival rate first, then add a latency emulation layer, then re-run long enough to separate warm-up from steady state.

**Can k6 and Locust be used together?**

Yes, and some teams do: k6 for protocol-level arrival-rate tests that gate CI, Locust for scenario tests that reuse application code and model multi-step flows. The risk is divergence — two scripts that claim to model the same traffic but disagree on think time or data distribution. Keep a single written traffic model document and derive both scripts from it; the tool is an implementation detail.

## The one thing to change first

Open the existing load test script and find the line that sets concurrency — `vus` in k6, `-u` in Locust. Before changing it, add one line that records the achieved request rate over time (k6: `--out csv=results.csv` and inspect `http_reqs`; Locust: `--csv=results` and read `requests.csv`). Run the test for at least five minutes, discard the first minute as warm-up, and compare the achieved rate to the configured rate. If they diverge by more than a few percent, the model is closed-loop in disguise, and every latency number reported from it is optimistic. That single comparison is the highest-value 30 minutes available on load testing this week.
