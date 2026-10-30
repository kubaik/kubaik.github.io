# MCP servers: the 3 hidden costs nobody quotes

Most MCP (Model Context Protocol) tutorials stop at a working localhost demo. The gap between that demo and a production deployment is where the operational costs live: memory that grows with concurrency, proxy timeouts that don't match long-lived streams, bandwidth overhead from protocol framing, and a security surface that expands the moment you leave localhost.

This article covers what actually changes when MCP moves into production, how to measure each cost yourself, and where the protocol is the wrong tool for the job.

## Why MCP behaves differently from a REST API

MCP is a JSON-RPC 2.0 protocol, commonly carried over WebSocket. A single connection can multiplex multiple independent tool calls, each with its own request/response lifecycle. The server is a stateful process: it holds tool manifests in memory, may cache resource contents, and may run background workers for long-running tasks.

That statefulness is the root of most production surprises. A REST endpoint can be treated as stateless and horizontally scaled with a load balancer. An MCP server holds per-connection state, which means:

- Memory scales with the number of concurrent streams and the size of payloads in flight.
- Long-lived connections interact badly with default proxy and load balancer timeouts.
- A single process's event loop becomes a shared bottleneck for all concurrent tool calls.

The practical consequence: an MCP server in production behaves more like a small application server or notebook kernel than like a stateless API endpoint. Plan capacity accordingly.

## Cost 1: Memory growth under concurrency

### Why it happens

In async Python servers, each inbound WebSocket frame is typically held in a queue or coroutine frame until its handler completes. If tool calls return large payloads — multi-megabyte resource files, for example — the working set grows faster than the garbage collector reclaims it. The result is RSS that climbs with concurrency and payload size rather than settling at a fixed baseline.

A second amplifier is resource caching. If the server caches every resource it reads and there is no eviction policy, the cache grows monotonically with the number of distinct resources accessed. Ten thousand distinct 1 MB resources is 10 GB of resident memory if nothing evicts them.

### How to measure it

Do not trust a single `docker stats` reading. Measure the relationship between concurrency and RSS:

1. Instrument the process to expose RSS. On Linux, read `/proc/self/statm` and multiply the resident pages by the page size, or use `psutil.Process().memory_info().rss`.
2. Drive load at increasing concurrency levels (for example 50, 100, 200, 400 streams) with a fixed payload size, holding each level for several minutes.
3. Record steady-state RSS at each level. If RSS grows roughly linearly with concurrency and does not plateau, you have a per-connection retention problem. If RSS grows with the number of distinct resources accessed, you have a cache eviction problem.

For a quick check on a running container:

```bash
docker stats --format "{{.Name}}\t{{.MemUsage}}\t{{.NetIO}}" --no-stream $(docker ps --format '{{.Names}}' | grep mcp)
```

Watch the memory column over several minutes of real traffic. A steady climb with no plateau is the signal to investigate.

### Mitigations

- Replace the standard `json` module with a faster serializer such as `orjson` if profiling shows serialization is a significant allocation source. Measure before and after; the gain depends on payload shape.
- Cap resource caching with an explicit LRU policy and a maximum entry count.
- Explicitly drop large buffers after use rather than relying on reference counting to do it promptly.
- Set a container memory limit and alert on RSS approaching it, so you find the ceiling before the kernel OOM killer does.

## Cost 2: Latency from serialization and the event loop

Each JSON-RPC round trip adds serialization and deserialization overhead on top of the tool's actual work. In a single-threaded async runtime, CPU-bound serialization competes with every other coroutine on the same event loop.

Two effects compound:

- **Serialization overhead per message.** Measured with a profiler, not assumed. JSON encoding of a multi-megabyte payload is real CPU time.
- **Event loop contention.** If a tool call blocks the event loop — synchronous I/O, CPU-heavy work, a slow library — every other in-flight request waits. Thread pool executors help only if the blocking work is actually dispatched to them and the pool is sized for the expected concurrency.

### How to measure it

Instrument three separate histograms rather than one end-to-end latency number:

- Time spent in serialization (encode and decode).
- Time spent inside the tool handler.
- Time a request waits in a queue before its handler starts.

If queue wait dominates, the problem is concurrency or pool sizing. If serialization dominates, the problem is payload size or the serializer. If handler time dominates, the problem is the tool itself. End-to-end p50 hides all three.

Also measure p99 and p99.9, not just p50 and p95. Spikes that last a few hundred milliseconds are exactly what a coarse sampling interval misses. If your metrics scrape interval is 10 seconds, a 500 ms spike may never appear in the data. Use a histogram with fine buckets and a scrape interval short enough to catch the tail you care about, or record the distribution in-process and export percentiles.

## Cost 3: Bandwidth and protocol framing overhead

Base64 encoding of binary payloads inflates size by a known, computable factor. Base64 represents 3 bytes as 4 characters, so encoded size is 4/3 of the original, a 33.3% increase. Add JSON envelope overhead and the total inflation depends on payload shape.

### Worked example (illustrative)

Assume a 500 KB binary resource and a JSON envelope of roughly 1 KB per message.

- Base64-encoded payload: 500 KB × 4/3 ≈ 667 KB.
- Plus envelope: ≈ 668 KB on the wire.
- Inflation versus the raw 500 KB: (668 − 500) / 500 ≈ 33.6%.

At 1,000 requests per day, that is roughly 668 MB/day versus 500 MB/day of raw payload — about 168 MB/day of framing overhead. Multiply by your egress rate to get the monthly cost. These figures are illustrative; substitute your own payload sizes and pricing.

### How to measure it

Compare bytes at three points: bytes of raw payload, bytes written to the socket by the server, and bytes billed by your cloud provider. The difference between the first two is protocol overhead; the difference between the last two is provider-side accounting you cannot control. Export a counter for bytes sent and received per connection, and reconcile it against the provider's billing report monthly.

## Cost 4: Security surface beyond localhost

The protocol itself does not enforce authentication or transport security. Those are deployment concerns, and the defaults in example code are usually tuned for demos, not production.

Common failure modes:

1. **Secrets embedded in tool manifests.** Some SDKs allow an `env` field on tool definitions. If a manifest is serialized into memory and logged at debug level, embedded credentials end up in logs and process memory. Treat tool manifests as untrusted input: strip credential-bearing fields before the server accepts them, and use workload identity or short-lived credentials instead of long-lived keys.

2. **Resource URI traversal.** A resource URI scheme like `mcp://resource/...` maps to a backing store. If the resolver does not normalize and constrain paths, a crafted URI can escape the intended root. Resolve URIs to absolute paths, reject relative segments (`..`), and enforce an allowlist of permitted roots.

3. **Ambient credentials via instance roles.** If the server runs with an instance profile or service account, every tool call that fetches a resource inherits those permissions. A single overly broad role can turn one MCP client into a broad data-access path. Scope the role to exactly the resources the server needs, and audit what the server can reach.

4. **Transport security.** Run TLS in production. If the SDK or example configuration disables TLS for convenience, that choice must not survive into a shared environment.

5. **Missing audit trail.** Tool arguments are not logged by default. If you need an audit trail, wrap tool handlers with a logging decorator that records the caller identity, tool name, and argument shape (with sensitive values redacted).

## Cost 5: Version drift and maintenance

Protocols and SDKs in this space evolve quickly. Breaking changes may land without a long deprecation window, and migration notes may live in changelogs or issue threads rather than a dedicated guide.

The maintenance burden shows up in three places:

- **Tool version mismatches.** If the server tags tool calls with a version and a client pins an older version, calls can be rejected with an opaque error. Automate version bumps in CI and keep client and server manifests in sync.
- **Runbook drift.** Operational runbooks that reference specific SDK behavior go stale. Version the runbook alongside the code.
- **Dependency churn.** Pin dependencies, test upgrades in staging, and treat SDK upgrades as you would any other dependency with breaking-change potential.

## A corrected minimal server

The code below is a minimal aiohttp-based WebSocket server. It is not a full MCP implementation; it shows the shape of the process and the production knobs that matter. Pin versions to whatever your environment has validated rather than copying the numbers here.

```bash
pip install aiohttp orjson uvloop
```

```python
import asyncio
import json
import logging
import os

from aiohttp import web

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("mcp")

# Cache with a hard bound. Without this, resident memory grows with the
# number of distinct resources accessed.
RESOURCE_CACHE_MAX = 200
_resource_cache: dict[str, bytes] = {}


def cache_get(uri: str) -> bytes | None:
    return _resource_cache.get(uri)


def cache_put(uri: str, value: bytes) -> None:
    if len(_resource_cache) >= RESOURCE_CACHE_MAX:
        # Simple eviction: drop the oldest inserted entry.
        _resource_cache.pop(next(iter(_resource_cache)))
    _resource_cache[uri] = value


async def read_resource(uri: str) -> bytes:
    cached = cache_get(uri)
    if cached is not None:
        return cached
    # Replace with a real fetch. Keep the read off the event loop if it
    # is blocking, e.g. via asyncio.to_thread.
    data = await asyncio.to_thread(_fetch_resource_sync, uri)
    cache_put(uri, data)
    return data


def _fetch_resource_sync(uri: str) -> bytes:
    raise NotImplementedError("wire up your backing store here")


def handle_request(payload: dict) -> dict:
    """Dispatch a single JSON-RPC style request."""
    method = payload.get("method")
    request_id = payload.get("id")
    if method == "read_resource":
        uri = payload.get("params", {}).get("uri", "")
        if not _uri_is_allowed(uri):
            return _error(request_id, -32602, "invalid resource uri")
        data = asyncio.get_event_loop().run_until_complete(read_resource(uri))
        return {"jsonrpc": "2.0", "id": request_id, "result": data.decode("utf-8", "replace")}
    return _error(request_id, -32601, "method not found")


def _uri_is_allowed(uri: str) -> bool:
    # Reject traversal and anything outside the permitted root.
    if ".." in uri.split("/"):
        return False
    return uri.startswith("mcp://resource/")


def _error(request_id, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(heartbeat=30.0)
    await ws.prepare(request)
    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    payload = json.loads(msg.data)
                except json.JSONDecodeError:
                    await ws.send_json(_error(None, -32700, "parse error"))
                    continue
                response = await asyncio.to_thread(handle_request, payload)
                await ws.send_json(response)
            elif msg.type == web.WSMsgType.ERROR:
                logger.error("websocket error: %s", ws.exception())
    finally:
        await ws.close()
    return ws


app = web.Application()
app.router.add_get("/mcp", websocket_handler)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    web.run_app(app, port=port, access_log=None)
```

Notes on the changes from a naive example:

- `heartbeat=30.0` sends WebSocket pings so idle connections are detected rather than sitting in `ESTABLISHED` until a proxy timeout fires.
- Blocking work is dispatched with `asyncio.to_thread` so it does not stall the event loop.
- The resource cache has a hard maximum. Without it, memory grows with the working set.
- URIs are validated before use.
- Logging goes to stderr, which works in containers without volume mounts.

## A reverse proxy configuration that matches long-lived streams

Default proxy timeouts are tuned for short request/response cycles. Long-lived WebSocket streams need explicit configuration.

```nginx
worker_processes auto;
worker_rlimit_nofile 65536;

events {
    worker_connections 1024;
}

http {
    upstream mcp_backend {
        server 127.0.0.1:8080;
        keepalive 64;
    }

    server {
        listen 80;
        server_name mcp.example.com;

        location /mcp {
            proxy_pass http://mcp_backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;

            # Match these to the longest expected tool call, not to a default.
            proxy_read_timeout 300s;
            proxy_send_timeout 300s;
            client_max_body_size 64M;

            # Streaming responses must not be buffered by the proxy.
            proxy_buffering off;
        }
    }
}
```

Key points:

- `proxy_read_timeout` and `proxy_send_timeout` must exceed the longest expected quiet period on the connection. Set them from measurement, not from a template.
- `proxy_buffering off` is required for streaming; buffering delays or truncates responses.
- `worker_rlimit_nofile` must be raised if you expect many concurrent connections, or the worker hits the file descriptor limit before the process hits its memory limit.
- `keepalive` in the upstream block should reflect realistic backend concurrency, not an arbitrary large number.

## Failure modes to design against

1. **Memory growth without a plateau.** Symptom: RSS climbs with concurrency and never settles. Cause: per-connection buffers not released, or an unbounded cache. Response: bound the cache, drop large buffers explicitly, and set a container memory limit with alerting.

2. **Proxy closes long-lived connections.** Symptom: `upstream prematurely closed connection` in proxy logs, or clients silently stop receiving data. Cause: proxy read timeout shorter than the quiet period between messages. Response: raise the timeout to match measured behavior and add application-level heartbeats.

3. **Truncated large payloads.** Symptom: clients report JSON parse errors on large responses. Cause: proxy buffer size or `client_max_body_size` too small. Response: raise both to match the largest expected message, and measure it rather than guessing.

4. **URI traversal.** Symptom: resource reads succeed for paths outside the intended root. Cause: no normalization or allowlist. Response: normalize to absolute paths, reject `..`, and enforce a root allowlist.

5. **Credential leakage via manifests or logs.** Symptom: secrets appear in debug logs or process memory. Cause: credential-bearing fields accepted in manifests, or debug logging enabled in production. Response: strip credential fields on input, disable debug logging in production, and use short-lived credentials.

6. **Version mismatch between client and server.** Symptom: opaque `invalid_method` or similar errors after an upgrade. Cause: client pinned to an older tool version. Response: automate version bumps in CI and keep manifests in sync.

7. **Dead connections consuming resources.** Symptom: connection count stays high after clients disconnect. Cause: no heartbeat or ping/pong, so the kernel holds sockets open until a timeout fires. Response: enable application-level heartbeats and set proxy timeouts consistently.

## Measuring the right things

Export at least these metrics, with fine-grained histograms rather than averages:

- Request duration, split into queue wait, handler time, and serialization time.
- Queue depth over time.
- Resident memory, sampled frequently enough to catch growth trends.
- Bytes sent and received per connection.
- Error counts by category (timeouts, parse errors, rejected URIs).
- Active connection count.

Scrape at an interval short enough to catch the tail latency you care about. If your scrape interval is longer than the spikes you are trying to find, you will not find them.

## When MCP is the wrong choice

MCP is a reasonable fit when you need bidirectional streaming between a client and long-running tools. It is a poor fit when:

- **Payloads are small and latency-sensitive.** JSON-RPC framing overhead is a fixed cost per message. For sub-kilobyte payloads with tight latency budgets, HTTP/2 or gRPC may be cheaper.
- **The host is memory-constrained.** An async Python runtime plus its dependencies has a non-trivial baseline footprint. On devices with a few hundred megabytes of RAM, the baseline plus working set can exceed available memory quickly. Measure the baseline before committing.
- **The team cannot maintain async Python.** The runtime is async-first. Blocking the event loop stalls every other request. If nobody on the team can debug that class of problem, choose a stack you can operate.
- **You need a strict audit trail out of the box.** Tool arguments are not logged by default. If compliance requires it, budget for the wrapper code.
- **Clients are browsers with strict CORS constraints.** Browser support is limited; most deployments end up with a local bridge process.

For fire-and-forget tasks, a plain HTTP API or a message queue is simpler. For data-heavy pipelines, a streaming RPC framework may be a better fit.

## Decision checklist

Before committing to MCP in production, answer these:

- What is the largest payload a tool can return, and have you measured the proxy and server behavior at that size?
- What is the longest quiet period on a connection, and do your proxy and load balancer timeouts exceed it?
- What is the maximum number of concurrent streams, and what is RSS at that level?
- Does the resource cache have a hard bound and an eviction policy?
- Are resource URIs normalized and constrained to an allowlist of roots?
- Are credential-bearing fields stripped from tool manifests on input?
- Is debug logging disabled in production?
- Is there an audit log of tool invocations, and does it redact sensitive arguments?
- How are client and server tool versions kept in sync?
- What is the alerting threshold on RSS, and what happens when it fires?

## FAQ

**How do I find a memory leak in an MCP server written in Python?**
Drive load at increasing concurrency with a fixed payload and record steady-state RSS at each level. If RSS grows linearly with concurrency and never plateaus, look at per-connection buffers and queues. If it grows with the number of distinct resources accessed, look at the resource cache. Use a memory profiler to attribute allocations to the code paths responsible.

**Why do WebSocket connections drop under load?**
The most common causes are a proxy read timeout shorter than the quiet period between messages, a client that never sends pings so idle sockets are reaped, and a blocking call stalling the event loop. Check the proxy error log for connection-closed messages, confirm heartbeat configuration on both sides, and profile the event loop for blocking calls.

**What is the cheapest way to run this at scale?**
That depends on your traffic shape, not on a fixed recipe. Measure bytes in and out per request, average connection duration, and peak concurrency. Then compare a small always-on instance against a serverless container that scales to zero, using your own measured numbers. Include the cost of the load balancer, which is often the dominant fixed cost at low traffic.

**Is base64 the only source of bandwidth overhead?**
No. Base64 accounts for a 33.3% increase on binary payloads by itself, and the JSON envelope adds more. Measure actual bytes on the wire against raw payload bytes to get the true factor for your workload.

## Action for the next 30 minutes

Pick one running MCP server and run the following against it for five minutes of real traffic:

```bash
docker stats --format "{{.Name}}\t{{.MemUsage}}\t{{.NetIO}}" --no-stream $(docker ps --format '{{.Names}}' | grep mcp)
```

Record the memory reading every 30 seconds. If it climbs steadily without plateauing, you have a retention problem to fix before the next deploy. If it plateaus, note the plateau value and compare it against your container memory limit — that gap is your safety margin.
