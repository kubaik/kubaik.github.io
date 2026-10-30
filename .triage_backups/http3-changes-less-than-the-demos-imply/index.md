# HTTP/3 changes less than the demos imply

HTTP/3 runs over QUIC, which runs over UDP. That single sentence explains why migration tickets stay open for months: the protocol is real, browser support is real, and CDN edges have terminated it for years, yet most origins have no reason to care.

The useful question is not "should we adopt HTTP/3" but "what is the gap between what the edge terminates and what the origin actually sees?" A common failure mode: a team enables HTTP/3 at the load balancer or CDN, watches the graph go green, and finds the application behaving exactly as before — same head-of-line blocking on the backend connection pool, same retry storms, same 502s when one slow request pins a worker. The transport changed; the failure modes did not.

This article covers where HTTP/3 and QUIC genuinely change backend behavior, where they are purely an edge concern, and how to reason about the boundary without a dedicated network team.

## The mechanism, briefly and precisely

QUIC is a transport protocol built on UDP. It folds together what TCP, TLS 1.3, and part of HTTP/2 stream multiplexing used to do in separate layers. The parts that matter for backend work:

- **Streams are independent at the transport layer.** A lost packet on one stream does not block delivery on another, because QUIC tracks loss per stream rather than per connection. TCP has no concept of streams, so a lost segment stalls everything behind it at the kernel level — the head-of-line blocking HTTP/2 could not fix because it sat above TCP.
- **Connection establishment is shorter.** A first connection combines the transport handshake and the TLS 1.3 handshake, so it typically needs one round trip before application data. Repeat connections can use 0-RTT resumption. A TCP plus TLS 1.3 handshake needs more round trips before the first byte of application data.
- **Connection migration is built in.** A QUIC connection is identified by a connection ID, not by the four-tuple of source IP, source port, destination IP, and destination port. A client switching from Wi-Fi to cellular can keep the connection alive. TCP cannot do this without help from a higher layer.
- **Encryption is mandatory, not optional.** There is no cleartext QUIC in the deployed web. Header protection means middleboxes cannot inspect or rewrite transport headers the way they can with TCP.

That last point is the one that quietly breaks things. More on it below.

## What actually changes at the origin

In the architecture `client -> CDN/edge -> origin`, if the edge speaks HTTP/3 to clients while speaking HTTP/1.1 or HTTP/2 to the origin, then **nothing at the origin changes**. The edge terminates QUIC, reassembles streams, and forwards ordinary HTTP requests over a TCP connection. The application neither knows nor cares.

This is the common case, and it is why many "we enabled HTTP/3" announcements are, from the backend's perspective, a no-op. The user-perceived latency improvement comes from the client-to-edge leg, where round-trip savings are largest for high-latency, lossy mobile networks. The origin-to-edge leg is usually short and well provisioned.

HTTP/3 reaches the origin only when it is explicitly configured there: a load balancer or reverse proxy that terminates QUIC and forwards to the app, or, more rarely, an application server that speaks QUIC directly. When that happens, the differences are concrete:

| Concern | HTTP/2 over TCP | HTTP/3 over QUIC |
|---|---|---|
| Loss handling | One lost segment stalls the connection | Loss isolated to the affected stream |
| Connection identity | Four-tuple; breaks on NAT rebind or IP change | Connection ID; survives migration |
| Handshake | TCP plus TLS, multiple round trips | Combined, 1-RTT, 0-RTT on resume |
| Middlebox visibility | Full transport header visibility | Headers encrypted; limited inspection |
| CPU profile | Kernel TCP stack, often offloaded | Userspace crypto per packet, higher CPU |
| Operational tooling | Mature: `ss`, `tcpdump`, conntrack | Newer; `qlog`, different capture workflow |

The CPU row is not a footnote. QUIC moves encryption and congestion control into userspace. On a busy edge, that is more CPU per byte than kernel TCP with hardware offload. Teams terminating QUIC on general-purpose VMs commonly see the CPU cost before any latency benefit.

## Where the demos mislead

The typical HTTP/3 demo shows a page load on a simulated lossy mobile connection, with HTTP/3 winning on time to first byte and full load time. That result is real and well documented. It is also not the workload most backend teams optimize.

Three patterns where the demo does not transfer:

**1. Server-to-server traffic on a stable network.** Two services in the same region, on a low-loss path, gain little from QUIC's loss isolation. Handshake savings matter for cold connections; a pooled, long-lived connection amortizes the handshake to nothing. The CPU cost of userspace crypto is paid on every byte.

**2. Workloads dominated by a single large response.** Stream independence helps when many streams compete and one stalls. If a request is one big body on one stream, a lost packet stalls that stream regardless of transport. QUIC does not make a single stream faster than a single TCP connection on the same path.

**3. Anything behind a middlebox that drops UDP.** This failure mode produces the most confused incident reports. Corporate firewalls, some VPNs, and older network equipment block or rate-limit UDP/443. The client usually falls back to HTTP/2, but fallback is not guaranteed to be fast, and if HTTP/2 was disabled at the edge because "we're on HTTP/3 now," those users are broken outright.

The third point is the one to internalize: **HTTP/3 is additive, never a replacement at the edge.** Keep HTTP/2 and HTTP/1.1 listeners. The `Alt-Svc` header is how a server advertises HTTP/3 availability so clients can attempt it and fall back cleanly.

## A decision framework

Work through these in order and stop at the first "no."

1. **Is the client population primarily browsers or mobile apps on variable networks?** If yes, HTTP/3 at the edge is worth enabling. If traffic is service-to-service inside one cloud region, skip to step 5.
2. **Does the edge (CDN, load balancer) support HTTP/3 termination with automatic fallback?** If it does, enable it there and stop. Do not push QUIC to the origin without a measured reason.
3. **Is there observability that distinguishes HTTP/3 from HTTP/2 sessions?** If not, fix that first. Enabling a transport you cannot see is how a fallback bug becomes undiagnosable.
4. **Is there a measured client-side latency problem on lossy networks that HTTP/2 has not solved?** If yes, enabling at the edge is the cheap experiment. Measure time to first byte and connection error rate by protocol version, not in aggregate.
5. **For origin-side QUIC: are both ends under your control, and is the path lossy or mobile?** If not, the cost/benefit rarely clears. Origin-side QUIC is a specialist move for specific topologies, not a default.

For most small-team backends serving clients on variable mobile networks, the answer lands at step 2: turn it on at the CDN, keep HTTP/2 at the origin, and spend the saved effort on what actually breaks — connection pooling, timeouts, and retry behavior.

## Implementation details that bite

### Alt-Svc and fallback

The server advertises HTTP/3 with a response header like:

```http
Alt-Svc: h3=":443"; ma=86400
```

`ma` is the max-age in seconds the client may remember the advertisement. If the header is served and the UDP listener then breaks, clients keep trying HTTP/3 for up to `ma` seconds before falling back. A short `ma` during rollout is a cheap safety valve.

### Health checks must not assume UDP

Most load balancer health checks are TCP or HTTP over TCP. If a QUIC listener is added, the health check still passes while UDP/443 is silently blocked upstream. A separate check is needed — or at minimum an alert on the ratio of HTTP/3 to HTTP/2 sessions dropping to zero. A QUIC listener that accepts nothing looks healthy to a TCP check.

### Timeouts and retries

QUIC has its own idle timeout and its own loss detection. If the application layer also has aggressive timeouts, double-timeout behavior can result: the transport retransmits while the app has already given up, or the app retries a request the transport was still delivering. Keep application-level timeouts longer than the transport's, and make retries idempotent. This is not new advice, but HTTP/3 makes transport timing less visible because it is not in the kernel.

### Observability

QUIC exposes `qlog`, a structured JSON event log for connection-level events. It is the packet-capture equivalent at the protocol layer, and it is the tool to reach for when a connection behaves strangely. On the server side, make sure access logs record the negotiated protocol version per request — many default log formats do not, which makes it impossible to answer "did this user get HTTP/3 or fall back?" after the fact.

### The middlebox problem in reverse

Because QUIC encrypts transport headers, some enterprise networks treat it as unknown traffic and block it. If a meaningful share of users sits behind such networks, the HTTP/3 adoption rate will plateau, and the HTTP/2 fallback path must be as fast as it was before. Do not let the fallback path rot.

## What outcomes to expect, and their limits

Published results vary by network conditions, and a figure measured on a simulated lossy link does not transfer to a well-provisioned datacenter path. What can be said with confidence:

- The largest gains are on high-latency, lossy, mobile client networks, and they come mostly from fewer round trips at connection setup and per-stream loss isolation.
- On stable, low-loss paths, the gain is small and may be negative once userspace crypto CPU is accounted for.
- The benefit is front-loaded: the first connection and the first few hundred milliseconds benefit most. Long-lived connections amortize the handshake savings away.

Illustrative arithmetic, not a measurement: assume a client on a network with 100 ms round-trip time. A TCP plus TLS 1.3 first connection needs roughly three round trips before application data (TCP handshake plus TLS handshake). QUIC's combined handshake is roughly one. That is a difference of about 200 ms on the first byte, before any loss. On a 10 ms datacenter path the same arithmetic gives about 20 ms — real, but rarely the thing that determines whether a page feels fast. The point is the shape of the curve, not the specific milliseconds.

### How to measure it for your own traffic

- **Instrument the negotiated version.** Log the protocol version (for example, the `:protocol` pseudo-header or the negotiated ALPN on the server) on every request. Then break down p50 and p95 time to first byte and error rate by version. An aggregate latency graph hides the split entirely.
- **Capture connection setup.** Server-side, record the time between accepting a connection and the first request byte, tagged by version. This isolates the handshake saving from everything else.
- **Count fallbacks.** Alert on the ratio of HTTP/3 to HTTP/2 sessions. A sudden drop is the signature of blocked UDP or a broken `Alt-Svc` advertisement.
- **Watch CPU per request at the terminator.** Compare CPU seconds per thousand requests before and after enabling QUIC on the same hardware and traffic mix. This is where userspace crypto shows up.
- **Test the blocked-UDP path deliberately.** From a network that drops UDP/443, load the site and confirm fallback is fast and correct. Do this before rollout, not after.

## What to watch out for

- **Disabling HTTP/2 "because we have HTTP/3."** Never. Fallback is mandatory.
- **Assuming the origin benefits.** If the edge terminates QUIC, the origin's connection pool behavior is unchanged. Head-of-line blocking problems at the origin are still there.
- **Health checks that pass while UDP is blocked.** Add a protocol-aware check or an alert on session ratio.
- **Logging without protocol version.** You cannot debug what you cannot see.
- **CPU surprises at the edge.** Userspace crypto costs more per byte than kernel TCP with offload. Size accordingly.
- **Treating 0-RTT as free.** 0-RTT resumption has replay considerations; do not use it for non-idempotent requests without understanding the tradeoff.

## The broader lesson

Transport upgrades change the shape of latency, not the shape of the application. The failure modes that hurt backend teams — unbounded connection pools, retries without idempotency, timeouts that fight each other, health checks that test the wrong thing — live above the transport and survive every protocol migration. HTTP/3 is worth enabling where it is cheap and observable, and it is not worth restructuring an origin for without a measured reason. Treat it as one more edge setting with a fallback path, not as a platform migration.

## Frequently Asked Questions

**Does HTTP/3 improve origin-to-origin service calls?**
Rarely enough to justify the change. On a stable, low-loss path inside a cloud region, handshake savings are amortized by connection pooling and userspace crypto cost is paid on every byte. QUIC's advantages are largest on lossy, high-latency, mobile client networks. For service mesh traffic on a healthy network, the benefit is marginal.

**What happens if a client's network blocks UDP/443?**
The client should fall back to HTTP/2 over TCP, which is why keeping the HTTP/2 listener is mandatory. Fallback is usually fast but not guaranteed to be instant, and it depends on the client honoring the `Alt-Svc` advertisement and its max-age. If HTTP/2 was disabled, those users are broken outright, not degraded.

**Does application code need to change to support HTTP/3?**
If the edge terminates QUIC and forwards HTTP/1.1 or HTTP/2 to the origin, no. If QUIC is terminated at your own load balancer or reverse proxy, the application still typically sees ordinary HTTP requests; the changes are in proxy configuration, health checks, logging, and timeout tuning. Application-level code changes are rare.

**Is HTTP/3 more expensive to run at the edge?**
Generally yes, per byte, because encryption and congestion control move into userspace rather than the kernel TCP stack with hardware offload. The cost is real but manageable at typical edge volumes; it becomes a sizing problem on high-throughput termination points. Measure CPU per request before and after enabling it.

## Do this next

In the next 30 minutes: open your edge or load balancer configuration and check two things. First, whether HTTP/3 is enabled with HTTP/2 still available as fallback — if HTTP/2 is disabled, re-enable it now. Second, whether access logs record the negotiated protocol version per request. If they do not, add that field to the log format before changing anything else. An HTTP/3 rollout cannot be evaluated without measurement, and the fallback path is the part that breaks silently.
