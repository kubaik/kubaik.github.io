# Agent followed instructions too literally: the cron job trap

## The symptom: a job that never finishes and never fails

A recurring production pattern in agent-style workloads looks like this: an alert fires for a job stuck in a `Running` state, the application log contains no stack trace, and CPU usage sits near idle. The container may exit cleanly after a long interval with exit code 0 and no error message, or it may simply keep running. Teams typically start by reading pod logs looking for a panic or timeout, then check `kubectl top pod`, and find nothing obviously wrong.

The confusion comes from a specific mismatch. The agent was built to a ticket that said something like: "run every 5 minutes; if the task errors, retry up to 3 times with 1-minute backoff." The spec described retries but never described a timeout. An implementation that follows that spec literally has no upper bound on how long a single attempt may take, so a blocked I/O call never produces an error, never triggers a retry, and never hard-fails.

That mismatch between *process alive* and *work done* is the core trap. Kubernetes reports the pod as healthy because the process has not crashed. The liveness probe returns 200 because the health endpoint is served by a different code path than the blocked work. The surface symptom is "pods stay green but do not process anything," which reads like a monitoring gap rather than a bug in the agent code.

## Why the literal reading fails

The root cause is the conflation of two distinct failure modes in distributed systems: process exit and work completion. The retry spec treats retries as a process-level concern — restart or re-enter the loop on error — but the actual failure is a network-level hang that never raises an exception.

In Python, `requests` documents `timeout=None` as the default, meaning the underlying socket blocks until the server responds or the connection is reset. A hung TCP connection leaves the process blocked in a syscall, and from the process's point of view nothing has gone wrong. There is no exception to catch, so the retry branch is never entered.

A common failure mode is that teams rely on load balancer or ingress timeouts to bound request duration. That works when traffic traverses those layers. Kubernetes-native agents frequently call internal services directly over a service mesh or pod IPs, bypassing the layer that would have cut the connection. The bound that existed before no longer applies.

A second contributing factor is DNS. Resolution happens before the HTTP client's connect timeout starts, so a slow resolver adds latency that no HTTP-level timeout covers. In Kubernetes, CoreDNS can become slow under load, and a multi-second lookup followed by a slow TCP handshake looks like a single long hang attributed to the server.

## Fix 1 — set explicit socket-level timeouts

The most common cause is the absence of a timeout on the HTTP client. The fix is to pass a timeout on every outbound call. In `requests`, a tuple splits connect and read timeouts.

```python
import requests

# Set timeouts everywhere the agent calls external services.
response = requests.get(
    "https://internal-service/api/v1/tasks",
    timeout=(2.0, 10.0),  # (connect, read) in seconds
)
```

The connect timeout bounds how long the client waits for the TCP handshake. The read timeout bounds how long it waits for bytes after the connection is established. Both matter: a server that accepts the connection and then stalls will pass the connect check and hang on the read.

A common objection is "we already have a 30-second liveness probe." That probe only detects process crashes or a dead health endpoint. It does not detect a blocked worker thread, because the health endpoint is usually served by a separate thread or path.

### Worked example: why the retry loop never fires

Consider a worker that processes one task per invocation with this structure:

```python
for attempt in range(3):
    try:
        result = requests.get(url)  # no timeout
        return result
    except Exception:
        time.sleep(60)
```

If the server accepts the connection and then stalls, the call blocks in the read. No exception is raised, so `except` never runs, `sleep(60)` never runs, and the loop never advances. The job runs until the process is killed or the connection is reset by something upstream. The retry logic is present and correct in isolation; it simply never gets control.

Adding `timeout=(2.0, 10.0)` changes the outcome: after 10 seconds the call raises `requests.exceptions.ReadTimeout`, the `except` branch runs, and the loop proceeds. The retry behavior the ticket asked for now actually executes.

### How to measure whether this is your problem

Instrument the client, not the process. Add a histogram around outbound calls, for example `http_client_duration_seconds` with buckets at 0.5, 1, 2, 4, 8, 10, 30, 60. Then:

1. Compare the p99 against the intended timeout. If p99 exceeds the timeout you believe you set, the timeout is not being applied on that code path.
2. Check for a spike in the highest bucket. A cluster of samples at exactly 60 seconds usually means something upstream is cutting the connection, not your client.
3. Correlate with `http_client_errors_total` by exception type. If `ReadTimeout` never appears, the timeout is not firing.

During an active hang, inspect the socket state:

```bash
kubectl exec <pod> -- ss -tunap | grep <port>
```

Connections stuck in `SYN_SENT` indicate the handshake never completed (connect timeout territory). Connections in `ESTABLISHED` for longer than the read timeout indicate the read is not being interrupted.

## Fix 2 — retry the work, not the process

The less obvious cause is retry logic that treats retries as process restarts rather than work retries. When the retry loop wraps a blocking call, the backoff sleeps inside the same process while the socket stays open. The retry interval is consumed waiting for a connection that may never resolve.

The fix is to bound each attempt and let the retry loop run between bounded attempts. An async client with a total timeout makes this explicit.

```python
import asyncio
from aiohttp import ClientSession, ClientTimeout

async def fetch_task(session: ClientSession, url: str) -> dict:
    timeout = ClientTimeout(total=10)
    async with session.get(url, timeout=timeout) as response:
        return await response.json()

async def retry_task(url: str, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            async with ClientSession() as session:
                return await fetch_task(session, url)
        except (asyncio.TimeoutError, ConnectionError):
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)  # exponential backoff
```

Two details matter here. First, `ClientTimeout(total=10)` bounds the entire request, including connection setup and body read, so a stalled server cannot hold the coroutine open. Second, `asyncio.sleep` yields the event loop, so the backoff does not block other work.

If migrating to async is not feasible, the same shape can be achieved with a thread pool and a timeout:

```python
from concurrent.futures import ThreadPoolExecutor, TimeoutError

def call_with_timeout(fn, timeout_s):
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn)
        return future.result(timeout=timeout_s)
```

Be aware of the caveat: the thread is not cancelled when the timeout fires. It continues running in the background until the underlying call returns. This bounds the caller's wait but leaks the blocked thread, so it is a mitigation, not a full fix.

### Failure-mode analysis: async does not fix a blocking library

A frequent trap is wrapping a blocking library in an async function. If `requests.get` is called inside an `async def`, it blocks the event loop for the duration of the call. The coroutine timeout never gets a chance to fire because the loop cannot schedule the timeout callback while it is blocked. The symptom is identical to the original bug: the agent hangs with no error. The fix is to use an async-native client such as `aiohttp` or `httpx`, or to run the blocking call in an executor.

## Fix 3 — bound DNS resolution separately

DNS resolution happens before the HTTP client's connect timeout starts, so it needs its own bound. In Kubernetes, CoreDNS can slow down under load, and a slow lookup plus a slow handshake looks like one long server-side hang.

In Node.js, `dns.lookup` accepts a `timeout` option, and the default result order can be pinned:

```javascript
import { setDefaultResultOrder, lookup } from 'dns/promises';

setDefaultResultOrder('ipv4first');
try {
  await lookup('internal-service', { all: false, timeout: 2000 });
} catch (err) {
  console.log('DNS lookup failed or timed out after 2s');
  throw err;
}
```

In Python, DNS timeout is not exposed by `requests` directly. Options include setting a resolver-level timeout via the container's `resolv.conf` (`options timeout:2 attempts:2`), or using a client library that exposes resolver configuration. Checking from inside the pod is quick:

```bash
kubectl exec <pod> -- dig +time=2 +tries=1 internal-service.svc.cluster.local
```

If the lookup takes longer than the configured bound, the resolver is the bottleneck, and no HTTP client timeout will help.

## Timeout defaults across common clients

The table below lists documented defaults for widely used HTTP clients. Verify against the version you actually ship; defaults change between majors.

| Runtime / library | Setting | Documented default | Notes |
|---|---|---|---|
| Python `requests` | `timeout` | `None` (no timeout) | Tuple form sets connect and read separately |
| Python `urllib3` | `Timeout` | `None` | Underlying library for `requests` |
| Node.js `http` | `timeout` | `0` (no timeout) | Applies to socket inactivity, not total duration |
| Node.js `dns.lookup` | `timeout` | `-1` (OS default) | Resolution only, not TCP |
| Go `http.Client` | `Timeout` | `0` (no timeout) | Covers dial, TLS handshake, headers, body |
| Java `HttpClient` | `connectTimeout` | Implementation-defined | Set both connect and request timeouts explicitly |

The pattern across languages is consistent: the default is unbounded, and the burden is on the caller to set a bound.

## How to verify a fix actually took effect

A timeout change that is not exercised proves nothing. Verification requires forcing the failure.

1. **Synthetic delay.** Introduce a proxy that delays responses on the target endpoint by more than the configured timeout. A tool such as `toxiproxy` can inject a fixed latency. After the fix, the client should abort at the timeout and log a `ReadTimeout` (or the equivalent exception for your runtime).
2. **Metric check.** Confirm the histogram's upper buckets are empty and that the error counter increments by exception type. If the error counter does not move, the timeout path was never taken.
3. **Socket check during the hang.** Run `ss -tunap` inside the pod while the synthetic delay is active and confirm no connection stays in `ESTABLISHED` beyond the read timeout.
4. **Log check.** Confirm the exception is logged with the endpoint and the elapsed time. A timeout that fires but is swallowed by a broad `except Exception` is invisible and will not be caught in review.

## Prevention checklist

- Require an explicit timeout on every outbound call in code review. Treat a missing timeout as a blocker, not a nit.
- Add static analysis to CI. For Python, `bandit` can flag `requests` calls without a `timeout` argument. For Node.js, ESLint rules can flag `http.request` without a timeout. Configure the rule to fail the build.
- Document the timeout choice next to the call, ideally in the function docstring, so a future override is a deliberate decision rather than an accident.
- Keep a default client wrapper that applies a baseline timeout, and allow per-call overrides with a stated reason. This makes the default safe and the exceptions visible.
- Run periodic fault injection against the endpoints the agent depends on. A weekly or per-release experiment that injects latency and verifies fast failure catches timeout gaps before they reach production.

## Related failure modes and what they actually indicate

- **Task stuck in `Pending`.** Usually a scheduling or resource constraint, not a timeout issue. Check pod events and resource requests before touching client code.
- **Exit code 137.** The process was killed, typically by the OOM killer. Often caused by buffering a large response in memory. Streaming responses and a memory limit address this.
- **`ECONNRESET`.** The peer closed the connection. This can mask a client timeout misconfiguration because the error looks like a server problem. Check whether the server has its own idle timeout that fires first.
- **`ENOTFOUND`.** Resolution failed outright. Common when the agent runs outside the cluster and uses the host resolver. Pin the resolver or verify it from inside the pod.

## FAQ

**Why does the agent still hang after setting `timeout=10`?**
The timeout may not be applied on every code path. Check whether some calls use a session object with its own defaults, whether a wrapper overrides the value, and whether any call bypasses the HTTP client entirely (for example, a raw `socket.create_connection` without a timeout).

**How do I tell whether DNS is the cause?**
Run `dig +time=2 +tries=1 <host>` from inside the pod. If it takes longer than the bound, the resolver is slow. Check CoreDNS resource usage and consider scaling it or pinning the resolver in `resolv.conf`.

**The agent uses asyncio but still hangs. What is missing?**
A blocking library called inside an async function blocks the event loop, so the coroutine timeout cannot fire. Use an async-native HTTP client, or run the blocking call in an executor.

**Should every request use the same timeout?**
No. Set timeouts per endpoint based on observed latency distributions. A health check can tolerate a short bound; a bulk export may need a long one. Record the rationale next to the call.

**What is a reasonable starting default for internal services?**
A common starting point is a 2-second connect timeout and a read timeout set above the endpoint's observed p99. Measure first, then set the bound with headroom, and revisit when the service's latency profile changes.

## Action for the next 30 minutes

Open the module that wraps outbound HTTP calls in your agent — commonly `clients/http.py`, `utils/network.py`, or an equivalent. Grep for every call site that constructs a request without a timeout argument. For each one, add an explicit connect and read timeout, choosing the read value from the endpoint's observed p99 plus headroom. Then run one synthetic delay against a single endpoint and confirm the client aborts at the configured bound and logs the exception. That single verified abort proves the timeout path is wired correctly end to end.
