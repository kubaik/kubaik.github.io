# Queueing theory for backend capacity

Capacity planning is rarely the hard part. Knowing when a service is about to fail is. The failure mode usually appears only under the exact conditions nobody tests for, and it appears suddenly, as a latency cliff rather than a gradual slope.

Avoiding the classic capacity planning mistake does not require a PhD in queueing theory. It requires one formula, a measurement habit, and the discipline to use both before the p99 latency page fires at 3am.

A familiar scenario: a service handles 200 requests per second at 40% CPU utilisation, with a p99 latency of 80ms. A new feature doubles traffic to 400 rps. The team provisions twice the CPU, expecting the same 80ms p99. Instead, p99 jumps to several hundred milliseconds, and the on-call engineer spends the night adding instances while the load balancer sheds connections. The root cause is not a bug in the new feature. It is that utilisation rose from 0.4 to 0.8, and at 0.8 utilisation the queueing delay is not twice as bad—it is several times worse. That nonlinearity is the entire point of queueing theory, and it is the part spreadsheets miss.

This article walks through the mechanism, shows the arithmetic with a worked example, and describes a monitoring and provisioning process that applies to any request-response service. The goal is not to turn anyone into a queueing theorist. It is to provide a mental model that turns capacity planning from guesswork into a calculation that can be defended in a design review.

## The utilisation cliff: why 80% is not 80%

The core result comes from the M/M/1 queue: the simplest model of a single server with random arrivals and random service times. The average time a request spends in the system (waiting plus being served) is:

W = 1 / (μ - λ)

where λ is the arrival rate (requests per second) and μ is the service rate (requests per second the server can handle). Utilisation is ρ = λ / μ. Rewriting in terms of ρ and service time S = 1/μ:

W = S / (1 - ρ)

When ρ = 0.4, W = S / 0.6 ≈ 1.67 S. When ρ = 0.8, W = S / 0.2 = 5 S. So doubling utilisation from 0.4 to 0.8 multiplies the average time in system by 3, not 2. And that is only the average; the tail is worse.

The p99 latency in an M/M/1 queue scales even more sharply. For an exponential service time distribution, the 99th percentile of time in system is approximately S · ln(100) / (1 - ρ). At ρ = 0.4, that is about 4.6 S / 0.6 ≈ 7.7 S. At ρ = 0.8, it is 4.6 S / 0.2 = 23 S. So p99 grows by a factor of 3 when utilisation doubles—the same factor as the mean in this model, but the absolute numbers are much larger relative to service time.

Real services are not M/M/1. Service times are not exponential; they usually have lower variance. Arrivals are not Poisson; they burst. But the qualitative shape—latency rising hyperbolically as utilisation approaches 1—holds for any queue with variability. The M/M/1 formula is a conservative approximation that gives the right intuition and a safety margin.

The practical implication: capacity cannot be planned by linear extrapolation. At 40% utilisation with traffic doubling, the service does not need 80% of current capacity; it needs enough headroom to keep utilisation below a threshold where latency is acceptable. For most interactive services, that threshold is somewhere between 60% and 70% for CPU, and lower for resources with higher variance such as database connections or thread pools.

## Where the capacity spreadsheet goes wrong

A typical capacity planning exercise looks like this: measure current CPU at peak, measure requests per second at peak, compute CPU per request, multiply by forecasted requests, add 20% headroom, file a ticket for more instances. This approach fails for three reasons.

First, it ignores the queueing nonlinearity described above. Adding 20% headroom on top of 80% utilisation gives 96% utilisation after the traffic increase—a latency disaster. The correct headroom calculation uses the latency target, not a fixed percentage.

Second, it treats CPU as the only bottleneck. In a typical web backend, the bottleneck is often not CPU but a downstream dependency: a database connection pool, a thread pool in an application server, a connection limit on a load balancer, or a rate limit on an external API. Each of these is a queue with its own service rate and its own utilisation curve. The one with the highest utilisation relative to its latency target is the bottleneck, and adding CPU will not help.

Third, it uses averages. Capacity planning based on average requests per second misses the peaks. A service that averages 200 rps but peaks at 600 rps every day at noon needs to be provisioned for the peak, not the average. And the peak itself has a distribution—the highest peak of the month is higher than the highest peak of the day.

The fix is to model each constrained resource as a queue, measure its utilisation at peak, and compute the latency at the forecasted utilisation using a queueing approximation. The resource with the worst projected latency is the scaling priority.

## A worked example: from 200 to 400 rps

Assume a service with the following measured characteristics at peak (these are illustrative numbers, not a benchmark):

- Arrival rate: 200 requests per second
- Average service time per request on one instance: 10ms of CPU time
- Instance count: 4
- CPU utilisation per instance: 0.5 (so each instance handles 50 rps, and 50 rps × 10ms = 0.5 CPU-seconds per second)

The service time S = 10ms. Utilisation ρ = 0.5. Using the M/M/1 approximation, average time in system W = S / (1 - ρ) = 10ms / 0.5 = 20ms. The p99 is roughly S · ln(100) / (1 - ρ) = 10ms × 4.6 / 0.5 = 92ms. That matches an observed p99 of ~90ms.

Now traffic doubles to 400 rps. With 4 instances, each handles 100 rps, so utilisation per instance becomes 1.0—the system is saturated and queues grow without bound. Doubling to 8 instances keeps utilisation at 0.5 and latency the same. What if only 6 instances are added? Utilisation becomes 400 / (6 × 100) = 0.667. Then W = 10ms / 0.333 = 30ms, and p99 ≈ 10ms × 4.6 / 0.333 = 138ms. That is a 50% increase in p99 for a 33% increase in capacity—not linear.

If the SLO is p99 < 150ms, 6 instances are acceptable. If the SLO is p99 < 100ms, 8 instances are needed (or service time must be reduced). The decision is now a calculation, not a guess.

The M/M/1 model assumes one server. In reality, a service has multiple instances behind a load balancer, which is an M/M/c queue. The formula for M/M/c is more complex, but the key difference is that a pool of servers is more efficient than a single server of equivalent total capacity. For c servers each with service rate μ, the utilisation per server is ρ = λ / (c μ), and the probability of queueing is lower than for a single server at the same ρ. The Erlang C formula gives the exact queueing probability, and the average wait time is:

W_q = C(c, ρ) · S / (c (1 - ρ))

where C(c, ρ) is the Erlang C probability. For planning purposes, a pool of c servers behaves like a single server with service rate c μ but with a correction factor that reduces the wait time. In practice, for c >= 4 and ρ <= 0.8, the M/M/1 approximation is conservative (it overestimates latency), which is fine for capacity planning—over-provisioning slightly is preferable to under-provisioning.

## The bottleneck you didn't measure: connection pools and thread pools

CPU is rarely the first resource to saturate in a typical backend. More often it is a fixed-size pool: a database connection pool with a small default maximum, a thread pool in a servlet container, or an HTTP client connection pool. These are queues with a hard limit on the number of servers (connections or threads), and when they saturate, requests wait indefinitely or fail fast.

The queueing model for a fixed pool is M/M/c/c—a system with c servers and no queue (requests that arrive when all servers are busy are dropped). The probability that an arriving request finds all servers busy is given by the Erlang B formula. For a pool of 10 connections, each holding a connection for an average of 5ms, the maximum throughput is 10 / 0.005 = 2000 requests per second. At 1500 rps, utilisation is 0.75, and the Erlang B blocking probability is roughly 0.1—about 10% of requests would be rejected if there were no queue. In practice, there is a queue (the request waits for a connection), so the wait time grows as utilisation approaches 1.

The key point: a connection pool of size 10 is a queue with 10 servers. If a service handles 200 rps and each request holds a connection for 5ms, the utilisation is 200 × 0.005 / 10 = 0.1—fine. If traffic grows to 1500 rps, utilisation becomes 0.75, and the wait for a connection becomes significant. Adding CPU instances will not help if the database cannot handle more connections; either the pool size must increase (if the database allows) or the connection hold time must drop.

A common failure mode is a connection pool sized for average load but not peak load. The pool saturates during a traffic spike, requests queue for connections, and latency spikes even though CPU is idle. The fix is to measure pool utilisation (active connections / max connections) at peak and apply the same queueing calculation: keep utilisation below 0.7 at peak to avoid queueing delay.

## Monitoring: the metrics that tell you you're near the cliff

You can't manage what you don't measure, and the metrics that matter for queueing are not the ones most dashboards show. CPU utilisation is necessary but not sufficient. Utilisation of every constrained resource must be measured, and latency must be measured at the tail, not just the average.

The metrics to track:

- **Utilisation per resource**: CPU per instance, database connection pool active/max, thread pool active/max, load balancer active connections / max, queue depth for any internal queues (for example, a work queue between a web tier and a worker tier).
- **Latency percentiles**: p50, p95, p99, and p99.9 for each service and each downstream dependency. The p99 is the early warning; the p99.9 is the canary for the cliff.
- **Arrival rate and service time**: requests per second and average time per request, measured at the same point. From these, utilisation can be computed and compared to the measured utilisation to validate the model.
- **Queue depth and wait time**: for any explicit queue (a message queue or a thread pool queue), measure the number of items waiting and the time they wait. A rising queue depth at constant arrival rate is a sign that service time is increasing—often due to a downstream bottleneck.

The pattern to watch for: as utilisation crosses 0.7, p99 starts to grow faster than p50. That is the queueing effect. If p99 grows while p50 is stable, the service is approaching the cliff. The fix is to add capacity or reduce service time before utilisation hits 0.8.

A useful dashboard is a scatter plot of utilisation versus p99 latency for each resource, with a fitted curve. The curve should follow the 1/(1-ρ) shape. If it deviates upward, there is a bottleneck that has not been modeled—perhaps lock contention or a garbage collection pause.

## A minimal queueing calculator in Python

The M/M/1 and M/M/c latency estimates can be computed with a few lines of Python. This is not a replacement for measurement, but it is useful for quick what-if analysis during design reviews.

```python
import math

def mm1_latency(service_time_ms, utilisation):
    """Average time in system for M/M/1, in ms."""
    if utilisation >= 1:
        return float('inf')
    return service_time_ms / (1 - utilisation)

def mm1_p99(service_time_ms, utilisation):
    """Approximate p99 for M/M/1 with exponential service times."""
    if utilisation >= 1:
        return float('inf')
    return service_time_ms * math.log(100) / (1 - utilisation)

def erlang_c(c, utilisation):
    """Erlang C probability for M/M/c queue."""
    if utilisation >= 1:
        return 1.0
    numerator = (c * utilisation) ** c / math.factorial(c) * (1 / (1 - utilisation))
    sum_terms = sum((c * utilisation) ** k / math.factorial(k) for k in range(c))
    return numerator / (sum_terms + numerator)

def mmc_latency(service_time_ms, c, arrival_rate):
    """Average time in system for M/M/c, in ms."""
    mu = 1 / (service_time_ms / 1000)  # service rate per server, per second
    rho = arrival_rate / (c * mu)
    if rho >= 1:
        return float('inf')
    wait = erlang_c(c, rho) * (service_time_ms / 1000) / (c * (1 - rho))
    return (wait + service_time_ms / 1000) * 1000

# Example: service time 10ms, 4 instances, arrival rate 200 rps
print(mmc_latency(10, 4, 200))  # ~20ms
print(mmc_latency(10, 4, 400))  # inf (saturated)
print(mmc_latency(10, 8, 400))  # ~20ms
print(mmc_latency(10, 6, 400))  # ~30ms
```

Run this with your own service time and instance count to see how latency scales with traffic. The numbers will be approximate because real service times are not exponential and arrivals are not Poisson, but the shape is right.

## How to validate the model against your own service

The model is only useful if it is checked against reality. A simple validation procedure:

1. Pick a representative peak window (for example, the busiest hour of the last week).
2. Record requests per second, average service time (or CPU seconds per request), and instance count at the same timestamps.
3. Compute predicted utilisation and compare it to the measured utilisation from your metrics. A large gap means the service time estimate is wrong—often because it excludes downstream waits.
4. Plot measured p99 against measured utilisation for each minute in the window. If the points cluster along the 1/(1-ρ) curve, the model is a reasonable fit. If they sit above it, look for a hidden queue: retries, lock waits, GC pauses, or a saturated downstream dependency.
5. Repeat the same plot for every constrained resource. The resource whose curve is steepest at current utilisation is the next bottleneck.

This procedure takes an afternoon and produces a defensible answer to "how much headroom do we have?"

## The broader lesson: capacity is a distribution, not a number

The deeper principle is that capacity is not a single number to provision to; it is a relationship between arrival rate, service time, and the tail of the latency distribution. Any capacity plan that says "we need N instances" without specifying the latency target and the utilisation threshold is incomplete. The right question is: "At what utilisation does our p99 exceed the SLO?" And the answer comes from measurement plus a queueing model, not from a spreadsheet that multiplies averages.

This also changes how autoscaling is designed. A reactive autoscaler that adds instances when CPU exceeds 70% is already too late—by the time CPU hits 70%, the service is on the steep part of the curve, and new instances take time to boot and warm up. A better approach is to scale on a leading indicator: request rate relative to capacity, queue depth, or p99 latency relative to SLO. And to provision a buffer so that normal traffic spikes do not cross the cliff.

The same reasoning applies to databases, message queues, and any shared resource. Each is a queue. Each has a utilisation curve. The one with the highest utilisation relative to its latency target is the constraint. Find it, measure it, and plan around it.

## Start here: compute your current utilisation and p99

The next step is concrete: pick one service you own, and for its busiest hour, pull the requests per second and the average service time (or CPU time per request) from your metrics. Compute utilisation per instance and compare it to the observed p99 latency. If p99 is more than 3x the average service time, utilisation is likely above 0.6 and capacity should be added or service time reduced before the next traffic increase. Do this for every constrained resource—CPU, connection pools, thread pools—and you will have a capacity model you can defend.
