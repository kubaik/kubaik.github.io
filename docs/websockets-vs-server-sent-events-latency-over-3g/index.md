# WebSockets vs Server-Sent Events: latency over 3G

Real-time features on mobile networks fail in ways that staging environments rarely reproduce. A connection that works on office Wi-Fi can drop every 60 seconds on a train, and the cause is usually not the protocol itself but an idle timeout, a reconnect strategy, or a proxy that quietly buffers the stream. This article compares WebSockets and Server-Sent Events (SSE) for agent-style features — status feeds, live dashboards, chat — where the client is on intermittent mobile data and the backend sits behind a load balancer.

The goal is not to declare a universal winner. It is to lay out the documented behaviour of each protocol, the failure modes that matter on unreliable links, and a way to measure which one actually performs better for your workload.

## Why the comparison is not obvious

Both protocols replace polling with a single long-lived connection, so both cut the round trips, TLS handshakes, and header overhead that make REST polling feel sluggish. Beyond that they diverge:

- WebSockets are full-duplex and message-oriented. Either side can send at any time.
- SSE is unidirectional and text-oriented. The server streams events; the client sends commands over separate HTTP requests.

The common assumption is that WebSockets are always lower latency. That is true for client-to-server messaging, where SSE has no equivalent path. It is not automatically true for server-to-client streaming, where SSE can be lighter because it avoids WebSocket framing and carries no masking bytes.

The second common assumption is that SSE is always simpler to operate. That holds for client code, because browsers reconnect automatically. It does not hold for infrastructure, because a long-lived HTTP response is exactly the kind of connection that idle timeouts and buffering proxies are built to terminate.

Both assumptions need testing against your own traffic. The rest of this article covers what to test and what tends to break.

## WebSockets: mechanics and failure modes

A WebSocket connection starts as an HTTP/1.1 request with an `Upgrade: websocket` header. If the server accepts, the connection switches protocols and both peers exchange frames. RFC 6455 defines the frame format: a small header of 2 to 14 bytes depending on payload length, a masking bit on client-to-server frames, and a close-code registry (1000 normal closure, 1001 going away, 1006 abnormal closure) that lets clients distinguish a clean shutdown from a dropped link.

The small frame header is the main bandwidth argument for WebSockets over HTTP polling, where headers of 80 bytes or more are typical per request. Against SSE the comparison is closer, because SSE also avoids per-message request headers.

Where WebSockets are the only practical option:

- Client-to-server messages are frequent. A chat, a collaborative editor, or a telemetry upload path needs a persistent send channel.
- Both directions carry latency-sensitive traffic. A ride-hailing agent receiving driver GPS pings while pushing route updates is a two-way workload.
- You need binary payloads. SSE is text-only; binary data must be base64-encoded, which inflates it.

The failure modes that show up on mobile networks:

**Idle timeout disconnects.** A load balancer or proxy closes connections that carry no traffic for a configured period. The AWS Application Load Balancer default idle timeout is 60 seconds and is configurable up to 4000 seconds. A client that loses signal for longer than the timeout comes back to a dead socket. The next send fails, and the client must reconnect from scratch.

**Reconnect storms.** The browser `WebSocket` API does not reconnect automatically. Every client library implements its own backoff. If the backoff is a fixed delay, or has no jitter, a regional blip can bring thousands of clients back at the same instant. Exponential backoff with random jitter is the standard mitigation.

**Ordering across instances.** RFC 6455 guarantees ordering within a single connection. It says nothing about ordering across connections. If messages for one user can be served by more than one backend instance, you need a broker or a partition key to preserve order.

A minimal Node.js WebSocket server:

```javascript
import { WebSocketServer } from 'ws';
import { createServer } from 'http';

const server = createServer();
const wss = new WebSocketServer({ server });

wss.on('connection', (ws) => {
  console.log('agent connected');

  ws.on('message', (data) => {
    const t1 = Date.now();
    ws.send(JSON.stringify({
      type: 'agent_update',
      payload: JSON.parse(data),
      server_latency_ms: Date.now() - t1,
    }));
  });

  ws.on('close', () => console.log('agent disconnected'));
  ws.on('error', (e) => console.error('ws error', e));
});

server.listen(8080, () => console.log('ws server on 8080'));
```

This echoes each message back with a server-side processing timestamp. Combined with `performance.now()` in the browser, that gives you a round-trip measurement you can log and aggregate.

A client wrapper with capped exponential backoff and jitter:

```javascript
const MAX_DELAY = 15000; // 15s cap
let ws;
let retryCount = 0;

function connect() {
  ws = new WebSocket('wss://api.example.com/agent-ws');

  ws.onopen = () => {
    retryCount = 0;
  };

  ws.onmessage = (e) => {
    const t0 = performance.now();
    handleIncoming(JSON.parse(e.data));
    console.log('handler latency:', Math.round(performance.now() - t0));
  };

  ws.onclose = () => {
    const delay = Math.min(MAX_DELAY, 1000 * Math.pow(2, retryCount));
    setTimeout(connect, delay + Math.random() * 100);
    retryCount++;
  };
}

connect();
```

The cap prevents a client from waiting minutes after a brief outage. The jitter spreads reconnect attempts so that a shared network event does not synchronise every client. Without both, a reconnect storm is the predictable outcome.

## Server-Sent Events: mechanics and failure modes

SSE is a long-lived HTTP response with `Content-Type: text/event-stream`. The client opens a GET request, and the server writes events as `data:` lines separated by blank lines. The browser parses them and dispatches `message` events. There is no protocol upgrade, no framing layer, and no masking.

The parsing rules are defined in the HTML standard, and the format is deliberately simple:

```
data: {"agent_id":"123","status":"active"}

data: {"agent_id":"123","status":"idle"}

```

Where SSE is the better fit:

- The traffic is mostly server-to-client. Status feeds, notification streams, and live dashboards fit this shape.
- You want browser-managed reconnection. `EventSource` reconnects automatically after a drop.
- You want the stream to be inspectable with ordinary HTTP tooling. `curl -N` shows the raw event stream, which makes debugging on a constrained connection straightforward.
- You want to reuse existing HTTP authentication, caching, and observability infrastructure without a protocol upgrade path.

The failure modes:

**Idle timeout disconnects.** Same mechanism as WebSockets. An SSE stream that sends an event every few seconds is usually safe, but a stream that goes quiet for longer than the proxy timeout will be closed. Many deployments send a periodic comment line (`: keepalive`) as a heartbeat to keep the connection active. This is a workaround for the timeout, not a fix, and it costs bandwidth.

**Proxy buffering.** Some intermediaries buffer the response body before forwarding it, which defeats streaming entirely. The symptom is that events arrive in bursts rather than as they are produced. Disabling buffering is usually a response-header or proxy-configuration change, and it is the first thing to check when SSE latency looks wrong.

**No client-to-server channel.** Commands go over a separate HTTP request. That is an extra round trip per command, but for a dashboard where the user taps a button occasionally, it is not usually the bottleneck.

**Automatic reconnect timing is not configurable in the client.** `EventSource` retries on its own schedule; the server can suggest a delay with a `retry:` field, but the browser decides how to honour it. If you need backoff shaped to your network conditions, you wrap `EventSource` or use a fetch-based reader instead.

**Message loss on reconnect.** SSE has no delivery guarantee. Events in flight when the connection drops are gone. The server can send an `id:` field per event and the browser will send `Last-Event-ID` on reconnect, which lets you replay from a known point — but only if your backend stores recent events.

A minimal FastAPI SSE endpoint:

```python
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import asyncio
import json

app = FastAPI()

async def event_stream(agent_id: str, request: Request):
    try:
        while True:
            if await request.is_disconnected():
                break
            await asyncio.sleep(2)
            data = {
                "agent_id": agent_id,
                "status": "active",
                "latency_ms": 23,
            }
            yield f"data: {json.dumps(data)}\n\n"
    except asyncio.CancelledError:
        return

@app.get("/agent/{agent_id}/stream")
async def sse_stream(agent_id: str, request: Request):
    return StreamingResponse(
        event_stream(agent_id, request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

The `X-Accel-Buffering: no` header disables buffering in nginx-style reverse proxies. The disconnect check prevents the generator from running forever after the client has gone away.

Client side:

```javascript
const es = new EventSource('/agent/123/stream');
es.onmessage = (e) => {
  const t0 = performance.now();
  const data = JSON.parse(e.data);
  console.log('SSE handler latency:', Math.round(performance.now() - t0));
};
es.onerror = () => {
  console.warn('SSE stream error; browser will retry');
};
```

Note that `onerror` does not expose the HTTP status or the reason for the failure. If you need to distinguish an auth failure from a network drop, you have to probe separately:

```javascript
es.onerror = () => {
  fetch('/health', { method: 'HEAD' }).then((r) => {
    if (r.status === 401) showBanner('session expired');
    else if (r.status === 429) showBanner('rate limited');
    else showBanner('connection lost');
  }).catch(() => showBanner('connection lost'));
};
```

That extra request is the cost of SSE's minimal error surface.

## Comparison table

| Property | WebSocket | Server-Sent Events |
|---|---|---|
| Direction | Full duplex | Server to client only |
| Transport | HTTP upgrade to `ws`/`wss` | Plain HTTP/1.1 or HTTP/2 |
| Payload | Text or binary | Text only (UTF-8) |
| Browser auto-reconnect | No | Yes |
| Reconnect timing control | Full, in your client | Limited; server can hint via `retry:` |
| Delivery guarantee | None beyond TCP | None; `Last-Event-ID` supports replay |
| Frame overhead | 2–14 byte header plus masking | `data:` prefix and blank line |
| Debugging | Requires a WS-aware client | `curl -N` works |
| Idle-timeout sensitivity | Yes | Yes |
| Proxy compatibility | Generally good; some networks block upgrades | Generally good; watch for buffering |

Neither column is a verdict. The table describes behaviour; which row matters depends on your workload.

## How to measure the difference for your workload

Published benchmarks are close to useless here because the result depends on your network path, your proxy configuration, and your event rate. Measure it yourself. The setup below is a template, not a result.

**Define the workload.** Pick representative numbers for concurrent clients, messages per client per second in each direction, and payload size. A status feed might be one server-to-client event every 2 seconds and one client-to-server command every 60 seconds. A chat might be several messages per second in both directions.

**Instrument four things:**

1. **End-to-end latency per message.** Timestamp at the producer, timestamp at the consumer, subtract. Log the distribution, not just the mean. Report p50, p95, and p99.
2. **Reconnect rate.** Count connection establishments per minute across all clients. A healthy system shows a flat baseline; a spiking line means backoff or timeouts are wrong.
3. **Message loss.** Give every event a monotonic sequence number per stream. On the client, log gaps. A gap is either a dropped event or a reconnect that skipped replay.
4. **Bytes transferred per session.** Measure at the socket, not at your application layer. This is the number that matters for mobile data cost.

**Control the network.** You cannot reproduce a 3G hand-off on office Wi-Fi. Use a network-conditioning tool to inject latency, packet loss, and bandwidth limits, and a scheduled link drop to simulate a hand-off. Record the drop duration and compare how long each client takes to recover.

**Compare like with like.** Run both protocols against the same backend instance type, the same load balancer configuration, and the same event rate. Change one variable at a time. The most common measurement error is comparing a tuned WebSocket deployment against an untuned SSE one, or the reverse.

**Watch the configuration, not just the protocol.** The ALB idle timeout, the proxy buffering setting, and the client backoff parameters will dominate the result. If SSE reconnects every 60 seconds in your test, check the idle timeout before concluding anything about the protocol.

The metrics that tell you where the protocol actually breaks are reconnect rate, loss rate, and bytes per session. Latency alone will mislead you, because a protocol that reconnects cleanly can show worse p99 latency while delivering a better user experience than one that holds a stale connection open.

## Decision checklist

Work through these before choosing:

- **Is the client-to-server direction frequent?** If commands are more than occasional, WebSockets avoid a round trip per command. If commands are rare, SSE plus a REST endpoint is fine.
- **Is the payload binary?** SSE requires base64, which adds roughly 33% to the payload size. If you are streaming binary, use WebSockets.
- **Do you need replay after reconnect?** Both protocols need application-level support. SSE's `Last-Event-ID` gives you a hook; WebSockets give you nothing, so you build it yourself.
- **What is your idle timeout, end to end?** Check every hop: load balancer, reverse proxy, CDN, and any carrier-grade NAT in the path. The smallest value wins, and that is your effective maximum silence before a disconnect.
- **Can you configure buffering off?** If a proxy in the path buffers responses and you cannot change it, SSE streaming will not work as intended.
- **What does your client do on disconnect?** For SSE, the browser handles it. For WebSockets, you own it. Budget the implementation and testing time accordingly.
- **What is your mobile data budget per session?** Estimate bytes per event including framing and headers, multiply by events per session, and compare against your users' data plans. This often decides the question on its own.

## Common failure modes and how to diagnose them

**Disconnects at a suspiciously regular interval.** Almost always a timeout. Compare the interval to the idle timeouts of every hop in the path. A 60-second interval points at a default that was never changed.

**Events arrive in bursts.** Buffering. Check response headers and proxy configuration. `X-Accel-Buffering: no` and equivalent settings are the usual fix.

**Reconnect storm after an outage.** Backoff without jitter, or a fixed retry interval. Add jitter and cap the maximum delay.

**Latency grows over the life of a connection.** Often a head-of-line blocking problem on a shared connection, or a client that is falling behind and buffering. Check whether the consumer is keeping up.

**Messages arrive out of order.** For WebSockets, check whether messages for one logical stream can reach more than one backend instance. For SSE, check whether your replay logic can deliver an old event after a newer one.

**Works on Wi-Fi, fails on mobile.** Look for carrier-grade NAT idle timeouts and for networks that block the WebSocket upgrade. SSE over plain HTTPS traverses more networks without special handling.

## Recommendation

For read-heavy features — status feeds, dashboards, notification streams — SSE is usually the better default. It reuses HTTP infrastructure, the browser handles reconnection, and it is inspectable with standard tools. The costs are a separate path for client-to-server commands and a minimal error surface that you may need to work around.

For bidirectional or binary features — chat, collaboration, telemetry upload — WebSockets are the practical choice. Budget time for reconnect logic, backoff with jitter, and idle-timeout configuration, because the browser will not do any of it for you.

A hybrid is often the right answer: SSE for the status stream, WebSockets opened only for the tab or view that needs two-way messaging. That keeps the common case cheap and confines the complexity to the feature that requires it.

## What to do in the next 30 minutes

Open your load balancer configuration, find the idle timeout for the target group that serves your real-time endpoint, and write down the current value. Then find the same setting on every proxy between the load balancer and your application. If any value is lower than the longest silence your stream can produce, raise it and note the change. That single number, not the choice of protocol, is the most common cause of real-time features that work in staging and fail on a mobile network.
