# CPU-based autoscaling is mostly wrong

Most autoscaling policies are built on a metric that has almost nothing to do with the thing you actually care about. CPU utilization feels scientific — it's a number, it goes up and down, you can draw a threshold on it — so teams wire it to a scaling policy, ship it, and then spend the next six months wondering why the service falls over at 3 a.m. while the CPU graph sits comfortably at 40%. The contrarian claim is simple: for request-serving workloads, CPU-based autoscaling is the wrong default, and the metric you should be scaling on is usually a queue-depth or concurrency signal that the platform will hand you for free if you stop reaching for CPU first.

This isn't an argument that CPU is useless. It's an argument that CPU is a lagging, noisy proxy for the thing that actually determines whether a request succeeds: how long it waits before something picks it up. The problem this post solves is the specific one where your autoscaler is technically working — it's scaling, the graphs move, the alarms fire — and your users are still timing out.

## The stance: scale on the queue, not the thermometer

If a service handles requests, the question the autoscaler needs to answer is "is work arriving faster than it's being completed?" CPU utilization answers a different question: "how busy are the cores right now?" Those two questions diverge in exactly the situations where autoscaling matters most.

Consider a service that spends most of its request time waiting on a downstream call. Its CPU utilization can sit low while every worker thread is blocked, requests pile up, and latency climbs. A CPU-target policy sees idle cores and does nothing. Now consider a service doing heavy in-process computation: CPU pins at 100%, but that's the steady state, and adding replicas doesn't make each request faster — it just adds more workers competing for the same downstream resource. CPU-target scaling helps in neither case.

A queue-depth or concurrency metric — requests in flight, messages waiting, connections queued — measures the actual backlog. When backlog grows, work is arriving faster than it's draining, and that is precisely the condition under which another replica helps. This is the same reasoning behind the standard queueing result that latency grows non-linearly as utilization approaches saturation; the useful signal is how close to saturation you are, and concurrency measures that directly while CPU only approximates it.

## Why CPU targets feel right and fail quietly

The appeal of CPU is that it's universal. Every runtime exposes it, every container platform reports it, and it's the one metric you can set up without understanding your workload. That universality is the trap. A metric that works for every workload works well for almost none of them.

The failure is quiet because CPU-based scaling fails in the boring direction. It doesn't crash; it under-provisions. A typical failure mode: a service has a CPU target of 60%. Under a traffic spike, requests queue behind a fixed worker pool, latency rises, the load balancer starts returning 503s or the client times out — but CPU never crosses 60% because the work is blocked, not computing. The autoscaler never fires. The incident is real, the metric is green, and the postmortem says "autoscaling didn't trigger" without anyone asking why the trigger was the wrong signal.

The inverse failure is just as common: a service does bursty CPU work (JSON parsing, compression, image resizing) and CPU spikes on every request. A CPU target of 70% causes constant scale-out and scale-in, thrashing the replica count and paying cold-start cost on every fluctuation. The autoscaler is busy. It's just busy reacting to noise.

There's a third failure specific to how thresholds interact with averaging. If your policy uses an average CPU across replicas, one hot replica and nine idle ones average to a comfortable number while the hot one is saturated. If it uses a maximum, a single noisy replica drives scale-out for the whole fleet. Neither is wrong in the abstract; both are wrong when the underlying metric doesn't reflect the bottleneck.

## The mental model: separate the demand signal from the capacity signal

The useful reframe is to stop thinking about "a metric to scale on" and start thinking about two distinct signals.

The demand signal tells you how much work is waiting or arriving. This is what should drive scale-out. Concurrency, queue depth, in-flight requests, consumer lag — these measure demand directly. When demand rises, add capacity.

The capacity signal tells you how much headroom a single replica has. This is what should drive scale-in, or act as a guardrail. CPU, memory, and thread-pool saturation are capacity signals. They tell you a replica is close to its limit, not that you need more replicas.

When you conflate the two, you get the classic CPU-target policy: it uses a capacity signal to make a demand decision. It works only when demand and capacity happen to move together, which is true for simple CPU-bound request handlers and false for most real services.

The second half of the model is that scaling decisions are about rate, not level. A queue of 100 messages is not inherently a problem; a queue growing by 100 messages per second is. Policies that scale on the absolute level of a metric will over-react to steady-state backlog and under-react to a fast-growing one. Where the platform supports it, prefer metrics that express rate or that you can difference over a window.

A worked example makes the arithmetic concrete. Assume a consumer processes 50 messages per second per replica, and traffic arrives at 200 messages per second at steady state. You need four replicas just to keep up, with zero headroom. If you set a target of "queue depth under 100" and traffic steps to 300 messages per second, the queue grows at 100 messages per second until the autoscaler adds replicas; each new replica drains 50 per second, so you need two more replicas to restore balance, and during the scale-out window the queue grew by however many seconds the policy took to react times 100. If instead you scale on consumer lag with a target expressed as "seconds of backlog" — queue depth divided by drain rate — the same arithmetic tells you how much time you have before latency breaches, which is the number you actually care about. These figures are illustrative, but the structure is the point: the useful metric is the one whose units match the decision.

## What the platforms actually give you

This is where the argument stops being abstract. The major container orchestrators and managed platforms have converged on the same set of primitives, and they are not CPU.

Kubernetes' Horizontal Pod Autoscaler has supported custom and external metrics since early versions, and the autoscaling/v2 API exposes them alongside the original CPU and memory targets. The mechanism matters: the HPA doesn't read your metric directly. It reads it through the metrics API, which is fed by an adapter — commonly the Prometheus adapter for custom metrics, or a platform-specific external metrics provider. That indirection is why "just scale on queue depth" is a real engineering task and not a config flag: you need the metric to be exported, scraped, adapted, and registered before the HPA can see it. Teams that skip that work fall back to CPU because it's the one metric the HPA can read out of the box.

The HPA's behavior block, stable in autoscaling/v2, is the part most teams under-use. It lets you set separate scale-up and scale-down policies, a stabilization window that prevents flapping, and a tolerance that suppresses scaling when the metric is close to target. A CPU target with a default tolerance will still thrash on bursty workloads; the fix is often the behavior block, not the metric.

On the messaging side, the pattern is even more established. Queue consumers scale on backlog — the number of visible or unacknowledged messages — because that is the demand signal by definition. Managed queue services expose it directly, and the standard guidance for consumers is to scale on queue depth relative to consumer throughput rather than on consumer CPU.

For serverless request handlers, the platform scales on concurrency by default, and the documented concurrency limit per instance is the knob. That's the same principle: concurrency is the demand signal. When teams move a workload from a concurrency-scaled platform to a CPU-scaled one and see latency regress, the metric change is usually the reason.

## Where CPU is still the right answer

I'd be arguing against a strawman if I said CPU is never right. It's the correct signal in a specific and identifiable case: the workload is CPU-bound, request latency is dominated by computation, and replicas don't contend for a shared downstream resource. Batch image processing, video transcoding, compression, and CPU-heavy serialization all fit. In those cases CPU tracks demand closely because the work *is* the CPU, and a CPU target is a reasonable proxy for queueing.

CPU is also the right guardrail even when it isn't the scaling signal. A concurrency-target policy that ignores CPU can scale into a state where replicas are added faster than the downstream can absorb them, or where memory pressure causes OOM kills before concurrency ever rises. Using CPU and memory as maximum-value alarms — not as the target — catches that. The rule is about which metric *drives* the decision, not which metrics you're allowed to look at.

And there's a legitimate simplicity argument. A small service with a simple CPU-bound handler and low traffic volume is better served by a CPU target and a generous behavior block than by a custom-metrics pipeline nobody maintains. The cost of the better metric is real: an adapter, a scrape config, a metric that has to stay correct as the service changes. For a service where the difference doesn't show up in user-visible latency, the simpler policy wins.

## The decision rule

Here's the rule I'd hand to a team configuring autoscaling today, in order:

1. **Is the request latency dominated by CPU?** If yes, a CPU target is fine. If no, go to step 2.
2. **Is there a natural concurrency or backlog metric?** In-flight requests, queue depth, consumer lag, connections waiting. If yes, scale on it, with a target expressed in units that map to latency (requests per replica, seconds of backlog).
3. **Is the workload scaled by a platform that already exposes concurrency?** Use the platform's concurrency setting and stop fighting it.
4. **If you must use CPU**, set a behavior block with a stabilization window and a tolerance, and add a separate latency or error-rate alarm so a silent under-provision is visible.

| Signal | Measures | Best for | Failure mode if used as target |
|---|---|---|---|
| CPU utilization | Capacity consumed | CPU-bound handlers | Under-provisions when work is blocked; thrashes on bursty work |
| Memory utilization | Capacity consumed | Guardrail, not target | Sticky — memory rarely falls, so scale-in never happens |
| Concurrency / in-flight | Demand | Request handlers, RPC services | Needs a per-replica target tuned to actual throughput |
| Queue depth / lag | Backlog | Async consumers, workers | Absolute level over-reacts; prefer rate or seconds-of-backlog |
| Request rate | Arrival rate | Pre-scaling, capacity planning | Ignores per-request cost variation |
| Latency (p95/p99) | User experience | Alarm and guardrail | Noisy; slow to react as a scaling target |

The table is the whole argument in one place: the signals that measure demand are the ones that should drive scale-out, and the ones that measure capacity belong in guardrails.

## Wiring a concurrency signal into an HPA

If you're on Kubernetes and the service is a request handler, the practical path is to expose in-flight requests as a Prometheus metric and scale on it. The metric is cheap to maintain because it's a gauge incremented at request start and decremented at request end. Here's a minimal, correct handler in Go using the standard Prometheus client:

```go
package main

import (
	"net/http"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

var inFlight = prometheus.NewGauge(prometheus.GaugeOpts{
	Name: "http_in_flight_requests",
	Help: "Current number of in-flight HTTP requests.",
})

func init() {
	prometheus.MustRegister(inFlight)
}

func instrument(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		inFlight.Inc()
		defer inFlight.Dec()
		next.ServeHTTP(w, r)
	})
}

func main() {
	mux := http.NewServeMux()
	mux.Handle("/work", instrument(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// handler body
		w.WriteHeader(http.StatusOK)
	})))
	mux.Handle("/metrics", promhttp.Handler())
	http.ListenAndServe(":8080", mux)
}
```

That gauge is the demand signal. The HPA reads it through the Prometheus adapter as a Pods-type custom metric — the average in-flight count per pod — and scales to keep it at a target. Choosing the target is the part that requires measurement rather than guesswork: set it to roughly the concurrency at which your p95 latency starts to bend, which you can observe from a load test or from production latency data. A target that's too high lets queues build before scaling; too low and you pay for replicas you don't need.

The behavior block matters as much as the metric. A starting point for a service with a warm-up cost:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api
  minReplicas: 2
  maxReplicas: 20
  metrics:
    - type: Pods
      pods:
        metric:
          name: http_in_flight_requests
        target:
          type: AverageValue
          averageValue: "10"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
        - type: Percent
          value: 100
          periodSeconds: 30
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Percent
          value: 25
          periodSeconds: 60
```

The asymmetry is deliberate: scale up fast, scale down slowly. The 300-second scale-down window means a brief dip in traffic doesn't shrink the fleet just before the next spike. The values are illustrative starting points, not tuned recommendations — the point is the shape.

One caveat that bites teams: the HPA's custom-metrics path depends on the adapter being healthy and the metric being present for every pod. If a pod hasn't reported the gauge yet (during startup, or after a scrape failure), the HPA may treat the metric as unavailable and skip scaling. That's another reason to keep a CPU or memory metric alongside the custom one as a fallback, and to alarm on adapter scrape failures rather than discovering them during an incident.

## Frequently Asked Questions

**Why does my HPA not scale even though CPU is high?**

The most common causes are a missing or stale metrics pipeline, a tolerance that suppresses the change, or a target that's already satisfied on average across pods. Check that the metrics API returns a value for every pod before blaming the policy. A single pod without a metric can make the whole calculation fall back.

**Should I scale on p99 latency directly?**

Usually not as the primary target. Latency is noisy, reacts after the queue has already built, and a target that's too aggressive causes flapping. It's better as an alarm and as the metric you use to choose your concurrency target, not as the thing the autoscaler chases.

**What's a reasonable stabilization window for scale-down?**

Long enough to cover the gap between traffic dips that aren't real and the ones that are. Five minutes is a common starting point for request services with warm-up cost; shorter for stateless services that start instantly. Tune it against your actual traffic periodicity rather than copying a number.

**Does this apply to serverless?**

The principle does, the mechanics don't. Serverless request platforms already scale on concurrency, so the knob is the concurrency limit per instance, not a metric target. The mistake to avoid is moving a workload to a CPU-scaled platform and assuming the scaling behavior travels with it.

## The first thing to check

The next concrete step is small and specific: open the autoscaling policy for your highest-traffic service and look at which metric type is driving scale-out. If it's CPU, pull the p95 latency graph for the same window as the last traffic spike and check whether latency rose while CPU stayed under target — that's the silent under-provision, and it's the evidence you need to justify the migration. Then add the in-flight gauge to the handler, ship it, and confirm the metric appears in your metrics API before you touch the HPA. The policy change can wait until the signal is real.

## Official documentation

- [Kubernetes documentation](https://kubernetes.io/docs/home/)
- [Prometheus documentation](https://prometheus.io/docs/)

---

### About this article

**Written by:** [Kubai Kevin](/about/) — software developer based in Nairobi, Kenya.

**How this article was produced:** This site uses an automated LLM pipeline designed and maintained by the author. Before publication, drafts are screened automatically for minimum length, near-duplicate content, invented or unsourced claims and citations, unverifiable product names, and broken internal links. Individual line-by-line human editing is not performed on every post. Figures, benchmarks and scenarios are illustrative unless a source is linked; verify them against official documentation before relying on them in production. See the [AI content policy](/ai-content-policy/).

**Corrections:** Report errors via the [contact page](/contact/). Corrections are applied promptly.

**Last generated:** November 2026
