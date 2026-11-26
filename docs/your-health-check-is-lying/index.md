# Your health check is lying…

Kubernetes has a probe called `readinessProbe`. It answers one question: should this pod receive traffic right now? Most teams answer that question with a TCP connect or a `GET /healthz` that returns `200 OK` as long as the process is alive. That is not readiness. That is liveness wearing a readiness costume. The result is a deployment that looks healthy, a load balancer that sends traffic, and a service that returns 500s because its database connection pool is exhausted or its schema migration hasn't finished. The health check told the truth about the wrong thing.

The contrarian take: **your health check should fail more often than it does.** A readiness check that only fails when the process is dead is redundant — the orchestrator already knows the process is dead. A useful readiness check fails when the process is alive but cannot serve its purpose. That means it must be allowed to fail during startup, during dependency outages, and during graceful shutdown. If your readiness endpoint has never returned a non-200 in production, it is not a readiness check. It is a decoration.

This post is about designing checks that reflect real readiness: what to check, what not to check, how to avoid the failure modes that make health checks actively harmful, and how to test that they work before an incident does it for you.

## The common advice: check dependencies, keep it simple, fail fast

The standard guidance from platform teams and documentation is reasonable on its surface:

- A liveness probe should be cheap and only indicate whether the process is running.
- A readiness probe should indicate whether the instance can serve traffic.
- Readiness checks should verify critical dependencies like databases and caches.
- Keep checks fast and avoid heavy queries.
- Fail fast so the orchestrator can route around bad instances.

None of that is wrong. The problem is that teams implement the first two bullets and skip the third, or implement the third naively and create a cascading failure. The advice is directionally correct but underspecified. "Check dependencies" without a model of *which* dependencies, *how deep*, and *what the orchestrator will do with the answer* leads to checks that either never fail or fail in ways that amplify outages.

## Why `GET /healthz` returning 200 is not readiness

A health endpoint that returns 200 because the HTTP server is listening tells you almost nothing. The HTTP server can accept a connection while the application is unable to do anything useful. Common examples:

- The database connection pool is exhausted because of a connection leak. The process is alive, the HTTP listener is alive, but every request that needs the database will block until the pool timeout.
- A background worker that populates an in-memory cache has crashed. The HTTP server still serves requests, but they all hit a cold cache and time out downstream.
- A schema migration is still running. The process started, the HTTP server bound the port, but queries against the new schema fail.
- The instance lost its credentials or its network route to a dependency. The process is fine; its world is not.

In each case, a liveness-style check passes and traffic flows to an instance that cannot serve it. The load balancer has no signal to stop sending requests. Users see errors, and the orchestrator sees a healthy pod.

The fix is not to make the check heavy. The fix is to make it *representative*: check the things that must be true for a request to succeed, at a cost low enough that you can run the check frequently.

## The mental model: readiness is a contract, not a ping

Think of readiness as a contract between the instance and the router. The instance says: "I can serve requests of the kind you are about to send me, right now, within the latency budget you expect." The router says: "I will send you traffic only while that contract holds."

A contract has terms. For a typical HTTP service, the terms are:

1. **The process is running and the HTTP listener is bound.** This is necessary but not sufficient.
2. **The instance has completed its startup sequence.** Configuration loaded, migrations applied (or confirmed unnecessary), caches warmed to whatever degree the service requires.
3. **Critical dependencies are reachable and responsive within a bounded time.** Not every dependency — the *critical* ones, the ones whose absence makes every request fail.
4. **The instance is not in the process of shutting down.** During graceful shutdown, readiness should flip to false before the process exits, so the router stops sending new requests while in-flight requests drain.

A readiness check that verifies all four terms will fail during startup, during dependency outages, and during shutdown. That is correct behavior. A check that verifies only term 1 is a liveness check with a misleading name.

The contract model also clarifies what readiness should *not* do. It should not verify non-critical dependencies whose failure degrades but does not break the service. It should not run expensive queries that add load. It should not depend on the health of other instances or on the router itself. And it should not be used as a monitoring system — readiness is a routing signal, not an alerting signal.

## The dependency check that takes down your whole cluster

Here is the failure mode that makes teams abandon dependency checks entirely. A service checks its database in its readiness probe. The database has a brief hiccup — a failover, a network blip, a slow query that saturates connections. Every instance of the service fails its readiness check at the same time. The orchestrator, seeing all instances unready, stops routing traffic to all of them. The service is now completely down, even though the database recovered seconds later and most requests would have succeeded with a retry.

This is a cascading failure caused by a health check that was too strict and too correlated. The check was technically correct — the database *was* unavailable — but the response (remove all instances from rotation) was disproportionate.

Several mechanisms make this worse:

- **Correlated checks.** If every instance checks the same dependency in the same way, they all fail together. There is no diversity to absorb a transient blip.
- **No hysteresis.** A single failed check flips readiness to false immediately. A transient timeout becomes a routing decision.
- **Deep checks.** A check that runs a query joining several tables is more likely to time out under load, precisely when you need instances to stay in rotation.
- **Shared fate.** If the readiness check depends on a central service that is also struggling, the check adds load to the struggling service and removes capacity from the system at the same time.

The better pattern is to make dependency checks *shallow, bounded, and tolerant*. Check that a connection can be acquired and a trivial round-trip succeeds within a short timeout. Do not run application queries. Do not check every dependency. And consider whether a dependency failure should remove an instance from rotation at all, or whether it should be handled by retries, circuit breakers, and load shedding at the request level.

## A readiness handler that fails for the right reasons

The following Go example is self-contained and illustrates the contract model. It separates liveness from readiness, tracks startup completion, checks a critical dependency with a short timeout, and flips readiness to false during shutdown. It is illustrative, not production-hardened, but the structure is the point.

```go
package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"net/http"
	"sync/atomic"
	"time"

	_ "github.com/lib/pq"
)

type health struct {
	db        *sql.DB
	started   atomic.Bool
	shutting  atomic.Bool
}

func (h *health) live(w http.ResponseWriter, r *http.Request) {
	// Liveness: is the process able to respond at all?
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte("ok"))
}

func (h *health) ready(w http.ResponseWriter, r *http.Request) {
	if h.shutting.Load() {
		http.Error(w, "shutting down", http.StatusServiceUnavailable)
		return
	}
	if !h.started.Load() {
		http.Error(w, "starting", http.StatusServiceUnavailable)
		return
	}

	// Shallow dependency check with a hard timeout.
	ctx, cancel := context.WithTimeout(r.Context(), 250*time.Millisecond)
	defer cancel()

	if err := h.db.PingContext(ctx); err != nil {
		http.Error(w, "database unreachable", http.StatusServiceUnavailable)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(map[string]string{"status": "ready"})
}

func main() {
	db, err := sql.Open("postgres", "postgres://localhost/app?sslmode=disable")
	if err != nil {
		panic(err)
	}

	h := &health{db: db}

	mux := http.NewServeMux()
	mux.HandleFunc("/livez", h.live)
	mux.HandleFunc("/readyz", h.ready)

	srv := &http.Server{Addr: ":8080", Handler: mux}

	// Simulate startup work: migrations, cache warmup, etc.
	go func() {
		time.Sleep(2 * time.Second)
		h.started.Store(true)
	}()

	// On shutdown, flip readiness before draining.
	go func() {
		<-shutdownSignal()
		h.shutting.Store(true)
		time.Sleep(5 * time.Second)
		_ = srv.Shutdown(context.Background())
	}()

	_ = srv.ListenAndServe()
}

func shutdownSignal() <-chan struct{} {
	// Placeholder: wire to os/signal in real code.
	ch := make(chan struct{})
	return ch
}
```

Several details matter here. The liveness handler does no work — it cannot fail because of a dependency. The readiness handler checks startup state and shutdown state explicitly. The dependency check uses a 250 ms timeout, which is short enough to fail fast but long enough to absorb a normal round-trip. The check is a `PingContext`, not a query — it verifies connectivity, not schema correctness. And the shutdown path flips readiness before calling `srv.Shutdown`, so the router stops sending new requests while in-flight requests finish.

What is missing? Hysteresis. In this example, a single failed ping flips readiness to 503 immediately. In a real system, you might want to require two consecutive failures before marking unready, or to treat a timeout differently from a connection refused. That logic belongs in the handler or in a small state machine around it. The point is that the handler is the place to encode the contract, not a static `return 200`.

## When the common advice is right: keep liveness dumb

The advice to keep liveness checks cheap and dependency-free is correct and underappreciated. Liveness should answer exactly one question: is this process wedged in a way that a restart would fix? A deadlock, an infinite loop, a corrupted state — these are liveness problems. A database outage is not. If your liveness probe checks the database, a database blip will cause the orchestrator to restart every instance of your service, which is the opposite of helpful.

Similarly, the advice to avoid heavy queries in readiness checks is right. A readiness check runs frequently — every few seconds per instance — and its cost multiplies across the fleet. A check that runs a multi-table join adds load proportional to the number of instances times the check frequency. Under stress, that load is exactly what you don't need.

And the advice to fail fast is right, with a caveat: fail fast *within the check*, but do not necessarily propagate a single failure immediately to the router. The check should return quickly. The decision to remove an instance from rotation can be smoothed with hysteresis or with a short grace period.

## The decision rule: check what breaks every request

The rule I use: **a readiness check should verify exactly those conditions whose absence would cause every request to fail, and nothing else.**

Apply it as a set of questions:

1. **If this condition is false, will every request fail?** If yes, it belongs in readiness. If only some requests fail, or requests fail slowly, it probably does not.
2. **Can the condition be checked cheaply and with a bounded timeout?** If not, find a cheaper proxy or move the check to a background process that updates a flag.
3. **Is the check correlated across all instances?** If every instance checks the same central dependency in the same way, a blip removes all capacity. Consider whether the dependency failure should be handled at the request level instead.
4. **Does the check add load to a system that is already stressed?** If the dependency is struggling, a readiness check that hammers it makes things worse. Use a connection pool ping, not a query.
5. **Does the check flip during shutdown?** It must, or you will drop in-flight requests.

A practical application: for a typical CRUD service, the critical conditions are usually (a) startup complete, (b) database connection pool can acquire a connection, (c) not shutting down. Cache availability is often non-critical — a cache miss is slower but not fatal. Downstream service availability is often non-critical if you have retries and circuit breakers. Message broker connectivity is critical for consumers, not for HTTP request handlers.

This rule also tells you when *not* to add a check. If a dependency is non-critical, adding it to readiness turns a degradation into an outage. Resist the urge to check everything. The goal is not a comprehensive status page; the goal is a routing signal.

## Testing readiness before an incident does

A readiness check that has never failed in production is untested. You can test it deliberately:

- **Kill the dependency.** In a staging environment, stop the database or block the port. Verify that readiness flips to 503 and that the router stops sending traffic. Verify that liveness stays 200 so the orchestrator does not restart the pod.
- **Simulate slow startup.** Add an artificial delay before marking startup complete. Verify that readiness returns 503 during the delay and 200 after, and that the orchestrator does not send traffic prematurely.
- **Trigger shutdown.** Send SIGTERM and verify that readiness flips to 503 before the process exits, and that in-flight requests complete.
- **Check the timeout.** Introduce a dependency that responds slower than your check timeout. Verify that readiness fails within the expected time and does not hang.
- **Check the blast radius.** With multiple instances, verify that a single dependency blip does not remove all instances from rotation simultaneously. If it does, add hysteresis or reconsider the check.

These tests are cheap and can run in CI against a disposable environment. They catch the two most common failures: a readiness check that never fails (because it checks nothing) and a readiness check that fails too aggressively (because it checks a shared dependency without tolerance).

## What to do in the next 30 minutes

Open the file that defines your service's health endpoints. Find the readiness handler. If it returns 200 unconditionally, or if it only checks that the process is alive, replace it with a handler that checks three things: startup completion, a shallow ping to your most critical dependency with a timeout of 250 ms or less, and a shutdown flag. Then run one test: stop the dependency in a local or staging environment and confirm that `/readyz` returns 503 while `/livez` still returns 200. If it does not, you have found the bug before your users did.

## Frequently Asked Questions

**Should a readiness check hit the database?**
Yes, if the database is critical to every request, but only with a shallow check like a connection ping and a short timeout. Do not run application queries in the readiness handler. The goal is to verify connectivity, not correctness. If the database is not critical to every request, leave it out of readiness and handle its failures with retries and circuit breakers.

**What is the difference between liveness and readiness in Kubernetes?**
Liveness answers whether the process should be restarted. Readiness answers whether the pod should receive traffic. A liveness failure triggers a restart; a readiness failure removes the pod from service endpoints. They should not check the same things. Liveness should be cheap and dependency-free; readiness should reflect the ability to serve requests.

**Why does my pod get traffic before it is ready?**
Because the readiness probe is passing too early. Common causes: the probe checks only that the HTTP server is listening, the initial delay is too short, or the application marks itself ready before migrations or cache warmup complete. Add an explicit startup gate and verify that readiness returns 503 until it is lifted.

**Can a readiness check cause an outage?**
Yes. If every instance checks the same dependency and that dependency has a transient failure, all instances can fail readiness at once, removing all capacity. This is a correlated failure. Mitigate with shallow checks, timeouts, hysteresis, and by handling non-critical dependency failures at the request level rather than in readiness.

## Official documentation

- [Kubernetes documentation](https://kubernetes.io/docs/home/)
- [Google SRE Book: Monitoring Distributed Systems](https://sre.google/sre-book/monitoring-distributed-systems/)

---

### About this article

**Written by:** [Kubai Kevin](/about/) — software developer based in Nairobi, Kenya.

**How this article was produced:** This site uses an automated LLM pipeline designed and maintained by the author. Before publication, drafts are screened automatically for minimum length, near-duplicate content, invented or unsourced claims and citations, unverifiable product names, and broken internal links. Individual line-by-line human editing is not performed on every post. Figures, benchmarks and scenarios are illustrative unless a source is linked; verify them against official documentation before relying on them in production. See the [AI content policy](/ai-content-policy/).

**Corrections:** Report errors via the [contact page](/contact/). Corrections are applied promptly.

**Last generated:** November 2026
