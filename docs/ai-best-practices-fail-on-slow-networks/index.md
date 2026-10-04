# AI best practices fail on slow networks

Most AI engineering guidance is written for fast, cheap, unmetered connectivity. It assumes you can stream tokens from a remote model endpoint, retry aggressively on failure, keep a persistent connection open, and ship multi-megabyte payloads without anyone noticing. In markets where mobile data is metered and billed per megabyte, and where latency to the nearest hyperscaler region is measured in hundreds of milliseconds, those same patterns become silent cost increases and user-visible hangs.

The fix is not "use a smaller model." It is to treat the network as a scarce resource and design the AI feature around a request budget rather than around the model's capabilities. This article explains why the gap exists, gives a mental model for reasoning about it, and ends with a concrete thing to check in a codebase today.

## The one-paragraph version

Almost every "best practice" in LLM application development is conditional on a network assumption that is never stated. Streaming, aggressive retries, and connection reuse are genuinely better on a fast, unmetered link. On a metered 3G connection they can be actively harmful. Engineers then misattribute the resulting slowness to the model and swap models, when the time was actually spent in TCP handshakes, TLS negotiation, DNS resolution, and retransmits on a lossy link. Optimizing the model when the problem is the network is the most common misdiagnosis in this space.

## Why this concept confuses people

Take streaming responses. Streaming is better UX on a fast connection: the user sees tokens appear, perceived latency drops, and a 20-second generation feels responsive. Put the same feature on a connection where data is billed per megabyte and the user is on a prepaid bundle, and the stream is not free — every token rendered is bytes on the wire. A verbose model that streams 800 tokens of preamble before answering is not just slow; it is charging the user for the privilege of waiting.

The confusion comes from the fact that the guidance is not wrong in its original context. It is wrong *here*, and the people writing it usually have no reason to know that. A team reading "always stream, always retry with exponential backoff, always keep the connection warm" implements all three. Two of the three may be harmful in their deployment environment, and nothing in the source material flags it.

There is a second layer of confusion: engineers often attribute the problem to the model. A 12-second response is assumed to mean the model is slow, so the team switches to a smaller model. Sometimes that helps. Often the model was fine and the time was spent elsewhere in the request path. Optimizing the model when the problem is the network is the single most common misdiagnosis in this space.

## A mental model: the taxi meter

Think of the network like a taxi meter that runs on distance, not time — and the roads are bad. Every byte sent over the mobile network is distance on the meter. Latency is how long the trip takes, which depends on traffic. Packet loss is potholes: when you hit one, the taxi has to go back and redo part of the trip.

Three quantities matter, and they are independent:

1. **Round-trip time (RTT).** The time for one packet to reach the server and return. This sets the floor on any request-response interaction. If RTT to a model endpoint is 250 ms, then a naive chat flow that does five sequential round trips (auth check, session fetch, model call, tool call, response write) has already spent over a second before the model has produced a single token.
2. **Bandwidth and cost per byte.** How much data moves, and what it costs the user. On metered connections this is a budget, not a free resource.
3. **Loss rate.** On mobile links this is not zero. TCP treats loss as congestion and backs off, so a single lost packet can add a full RTT of delay even when bandwidth is fine.

The key insight: these three compound multiplicatively, not additively. High RTT plus loss means retries are expensive. High RTT plus large payloads means streaming is expensive. High cost per byte plus verbose models means every "helpful" preamble has a price tag. A multiplicative problem cannot be fixed with an additive solution like "make the model 20% faster."

A useful reframe: design the feature around a **request budget** — a fixed number of round trips and a fixed number of bytes — and then fit the model inside that budget, rather than picking a model and hoping the network cooperates.

## A concrete worked example

Suppose you are building an AI assistant for a fintech app. The user asks a question in a chat box. Here is the naive flow, the one that falls out of following generic best practices:

```
1. Client → API gateway (auth)          ~1 RTT
2. API → session store (fetch history)  ~1 RTT
3. API → model endpoint (stream)        ~1 RTT + generation time
4. Model → tool call → API → tool       ~2 RTT (if using function calling)
5. API → client (stream final answer)   ~1 RTT + stream duration
```

Assume, purely as an illustration:

- RTT to your backend: 80 ms (regional, reasonable)
- RTT from backend to model endpoint: 200 ms (cross-region, typical when the model is hosted far away)
- Model generation: 3 seconds for a 400-token answer
- Streaming overhead: the client receives ~400 tokens ≈ 1.6 KB of text, but with JSON framing and metadata, call it 3 KB

Naive total time before the first token reaches the user: roughly 80 + 80 + 200 + 200 + 200 = 760 ms of pure network, then generation starts. With streaming, the user waits about 760 ms plus time-to-first-token. That sounds fine. Now add packet loss: at 2% loss on a mobile link, TCP retransmits will occasionally add a full RTT here and there, and the tail latency (p95, p99) balloons. The median looks acceptable; the tail is where users churn.

Now the cost side. 3 KB per response. If the user asks 30 questions a day, that is 90 KB. Over a month, about 2.7 MB just for chat text — before counting the app's other traffic. On a metered prepaid bundle, that is a real line item for the user, and it competes with everything else they do on the phone.

Now the optimized flow:

```
1. Client → API (auth + session in one call, cached)  ~1 RTT
2. API → model (single non-streaming call, capped)    ~1 RTT + generation
3. API → client (compact response)                    ~1 RTT
```

Same feature, three round trips instead of five, and the output length can be capped so the payload is predictable. Notice what changed: not the model, but the number of network interactions and the size of the response. You can go further — cache the session client-side so step 1 becomes zero round trips on repeat questions, and use a local embedding or rules layer to answer trivial queries without hitting the model at all.

The arithmetic is illustrative, but the shape is real: in high-RTT environments, round-trip count dominates, and in metered environments, payload size dominates. Both are under your control. The model choice is often the least important lever.

## How to measure this in your own stack

The worked example above is a model, not a measurement. To replace it with real numbers, instrument each phase of the request path separately and log durations for:

- DNS resolution
- TCP connect
- TLS handshake
- Time to first byte from the model endpoint
- Total generation time
- Response transfer time

Most APM tools can break a request into these phases, or you can log timestamps at each step in the request handler. Then compare the distribution, not the average: p50, p95, and p99 for each phase. If time-to-first-byte is high but generation is fast, the problem is the network path, not the model. If generation dominates, a model change is worth considering.

To measure byte cost, log the `Content-Length` of every response the client receives, including framing and metadata, and multiply by the user's actual request volume. On a metered plan this is the number that matters, not the size of the model's text output alone.

To measure the effect of loss, compare request latency distributions across network conditions — for example, a stable Wi-Fi connection versus a throttled or lossy mobile profile. A packet-loss simulator or a mobile network emulator will show the tail behavior that a fast office connection hides.

## Common misconceptions, corrected

**"Just use a smaller model."** Sometimes right, often a distraction. A smaller model reduces generation time, which helps if generation dominates. But if p95 latency is dominated by round trips and retransmits, a smaller model changes almost nothing. Measure first.

**"Streaming is always better UX."** Streaming improves perceived latency on fast, unmetered connections. On metered connections it can increase total bytes and hold a connection open longer, making it more likely to drop. For short answers, a single compact response is often better. For long answers, streaming still wins — but cap the length.

**"Retries with exponential backoff make things robust."** Retries are correct for transient server errors. They are wrong as a response to network loss, because TCP already retries at the transport layer, and an application-level retry multiplies the cost. A retry storm on a lossy link is how one slow request becomes five slow requests and five times the data cost.

**"Keep the connection warm."** Persistent connections (HTTP keep-alive, websockets) help on stable networks. On mobile, the radio state machine means an idle connection is not actually free — the device may drop to a low-power state, and re-establishing costs a full RTT plus radio wake-up. Keep-alive helps for bursty traffic, hurts for sparse traffic.

**"The model is the bottleneck."** Usually it is not. Instrument the request path end-to-end before touching the model.

## The advanced version, once the basics are solid

Once the request budget is under control, the next levers are:

**Edge inference for small tasks.** Classification, intent detection, and short extractions can sometimes run on-device or at a nearby edge location, eliminating the cross-region RTT entirely for a fraction of requests. The tradeoff is model quality and the cost of shipping the model to the device — for anything beyond tiny models this is not yet practical on low-end phones, so be honest about which tasks qualify.

**Response caching keyed on normalized input.** Many user queries repeat. A cache layer that normalizes the query (lowercase, strip punctuation, canonicalize) can serve a fraction of requests with zero model calls and zero cross-region traffic. The hard part is cache invalidation and avoiding stale or wrong answers — scope the cache to safe categories.

**Speculative local answers with server verification.** Return an instant local answer, then reconcile with the server asynchronously. This is the pattern behind optimistic UI applied to AI. It works when wrong answers are cheap to correct and the user understands the answer may update.

**Compression at the application layer.** Text model responses compress well. Gzip or Brotli on the response body is table stakes and often forgotten on AI endpoints because the payloads "feel" small. They are not small once multiplied by request volume.

**Batching and debouncing on the client.** If the UI fires a model call on every keystroke, you are paying for every keystroke. Debounce, and batch related queries into one request where the model can handle it.

## Quick reference

| Lever | Reduces RTT cost | Reduces byte cost | Notes |
|---|---|---|---|
| Fewer round trips | Yes (large) | Neutral | Biggest single win on high-RTT links |
| Smaller/capped responses | Neutral | Yes (large) | Cap max tokens, strip preamble |
| Response caching | Yes (large) | Yes (large) | Scope carefully to safe categories |
| Edge/on-device inference | Yes (large) | Yes | Only for small tasks today |
| Compression (gzip/brotli) | Neutral | Yes (moderate) | Easy, often forgotten |
| Streaming | Neutral | Sometimes worse | Better perceived latency, more bytes |
| Aggressive retries | Worse | Worse | Let TCP handle transport loss |
| Smaller model | Sometimes | Neutral | Only if generation dominates |

## Decision checklist

1. Measure p50 and p95 latency for the AI endpoint, split by network phase (DNS, TLS, request, generation, response).
2. Count the round trips in the happy path. If it is more than three, consolidate.
3. Cap response size in tokens and in bytes.
4. Add response compression if it is missing.
5. Replace application-level retries on network errors with a single retry at most, and only on idempotent requests.
6. Cache aggressively where correctness allows.
7. Only then consider changing the model.

## Frequently asked questions

**How do I measure whether latency is network or model?**
Instrument each phase of the request separately: DNS lookup, TCP connect, TLS handshake, time to first byte from the model endpoint, and total generation time. Most APM tools can break this down, or you can log timestamps at each step. If time-to-first-byte is high but generation is fast, the problem is the network path, not the model.

**Why does p99 spike even though the median is fine?**
Tail latency on mobile networks is dominated by packet loss and radio state transitions. A single lost packet causes TCP to retransmit after a timeout, which can add hundreds of milliseconds. This affects a small fraction of requests but disproportionately affects the slowest ones. Reducing round trips reduces the number of opportunities for loss to bite.

**Is streaming ever the right choice on metered connections?**
Yes, for long-form answers where the alternative is a very long wait with no feedback. The user needs to see progress. But cap the output length, and consider whether the answer can be shorter. Streaming a very long essay to a user on a prepaid bundle is a cost decision, not just a UX one.

**Can I just host the model in-region to fix latency?**
If a suitable model is available in a region close to your users, yes, that eliminates a large chunk of RTT. The tradeoff is model quality and cost — in-region options are often smaller or more expensive per token. For many tasks that is the right trade. For others, the cross-region call is worth it, and round trips should be optimized instead.

## Deeper topics worth studying

The topics worth going deeper on are TCP behavior on lossy links (any good networking textbook covers retransmit and congestion control), HTTP/2 and HTTP/3 connection reuse, and the general literature on optimizing for low-bandwidth users, which predates LLMs by decades and still applies. The AI-specific part is new; the network discipline is not.

## Your next 30 minutes

Open your AI endpoint's request handler and count the network round trips in the happy path. Write them down in order. If the number is greater than three, pick the two you can merge or cache and do it today. That single change will do more for users on slow, metered connections than any model swap.
