# Which agent stack survives 3G?

## The failure mode this comparison is really about

Agent features tend to work fine in development and degrade in a specific, predictable way in the field. The initial round-trip is rarely the problem. The problem is what happens after a 30-second connectivity dropout, when every client that queued work during the outage reconnects within the same few seconds and replays its backlog at your origin simultaneously. That is a reconnection storm, and it is the single most useful lens for comparing an edge-deployed agent against a centralised one.

Mobile links in much of East Africa make this a routine event rather than an edge case. Connections commonly sit in the single-digit Mbps range downstream with round-trip times in the low hundreds of milliseconds, and packet loss spikes during peak hours. An agent feature designed against a stable fibre link will meet this environment eventually, whether or not it was designed for it.

Two broad architectures dominate:

- **Edge-deployed agent with local fallback.** Compute runs on points of presence close to the user, with per-session state held at the edge (for example, a durable-object style primitive or a small replicated store on a regional VM provider).
- **Centralised agent with client-side caching and request coalescing.** A conventional Node LTS service in one region, a Redis-compatible cache, and a service worker on the client that queues and replays requests.

Both work. Both fail in ways that are visible in advance if you know what to instrument.

## Option A: edge-deployed agent

Edge deployment places compute within roughly one ISP hop of the user by running on points of presence in cities such as Nairobi, Lagos, Johannesburg, and Mombasa. A durable per-session object at the edge lets an agent hold working memory without a round-trip to a central database.

The advantage is connection survival. When a phone drops from 4G to 3G, the edge node is often still reachable — the path is the same, just slower. A session object can hold the agent's working memory across a short disconnection and replay missed tokens when the client returns. Because the compute is local, first-token latency is typically dominated by the model call rather than the network hop, and the hop is short.

The hard constraint is state size. Edge session primitives are deliberately small. A durable object with a fixed per-object storage ceiling is the common shape; once you exceed it you either shard state across objects (reintroducing coordination latency) or push history to object storage or a central database (reintroducing the network call you moved to the edge to avoid). A ten-turn conversation with tool outputs is not small. Add retrieval context and the ceiling arrives sooner than teams expect.

Two more edge-specific costs are worth naming. Cold starts are usually near zero because the runtime keeps isolates warm, but that is a property of the platform, not a guarantee. And debugging is weaker than on a conventional server: you generally cannot attach an interactive debugger to a running edge worker, and log delivery is eventually consistent, so a tail-style log stream is your main tool rather than a debugger.

**Where Option A fits:** conversational features where the user is actively waiting, streaming responses, and workloads that benefit from buffering outbound tokens close to the client.

## Option B: centralised agent with client-side resilience

A centralised agent runs in one region and delegates disconnection handling to the client. The pattern is well known: a service worker caches recent agent responses, queues outgoing requests in IndexedDB, and replays them when the browser reports connectivity. On the server, a cache with a short coalescing window deduplicates the replay burst.

This is the stack most teams already operate: a Node LTS runtime, a web framework, Redis or a compatible cache for session state, and a relational database for persistence. The agent loop itself is unremarkable — call the model, parse tool calls, execute them, repeat. The interesting engineering is entirely in the client and the cache layer.

The strengths are predictability and operational familiarity. You control the runtime, you can scale vertically, and you have one region's worth of logs and metrics. Debugging is mature: Node's inspector protocol gives you full DevTools, and Redis exposes `MONITOR` and `SLOWLOG` for live inspection. A small team can run the whole stack locally in minutes.

The weakness is the storm. If several thousand clients in one city lose connectivity for 45 seconds during a fibre cut and then return at once, the origin sees a burst of thousands of requests in a couple of seconds. A cache helps only if the key structure actually collapses those requests into one. A naive per-user key with a 60-second TTL produces one key per user and therefore zero coalescing. You need a shared key per agent session, which implies session affinity or a central session store — a design decision that has to be made before the storm, not after.

**Where Option B fits:** batch and background processing, long-running tasks, large retrieval contexts, and any feature where the user is not actively waiting.

## How to measure the difference instead of guessing

Published latency figures for "an agent on 3G" are close to meaningless because they depend on model choice, prompt size, region, and carrier. Measure your own stack. Four instruments cover most of the ground.

**1. First-token latency, split by phase.** Record a timestamp when the client sends the request and another when the first token arrives. Subtract the network round-trip time, which you can estimate from a lightweight ping endpoint on the same origin. The remainder is your server-side time. If network time dominates, the fix is placement; if server time dominates, the fix is the model or the prompt.

**2. Reconnection storm size.** In your origin or load balancer logs, count requests arriving from a single client identity within a short window after a connectivity restoration event. If that count is small, a server-side coalescing window is sufficient. If it is large, you need buffering closer to the client or the edge. This is the number that decides the architecture, and it is a query against logs you already have.

**3. Replay cost.** Count how many times the same logical request is processed. If a client replays a request and the server re-runs the agent from scratch, you pay for the model call twice. Instrument a request idempotency key and count duplicates per key.

**4. Egress volume per session.** Log bytes sent per completed agent session, including retransmissions. Multiply by your provider's per-gigabyte egress rate. This is the line item that most often changes the answer, and it is invisible unless you measure it.

A useful way to run these: replay a recorded session trace against both stacks with an artificial delay and loss profile injected on the client side, and compare the four numbers. Reasoning from a controlled replay is far more reliable than reasoning from someone else's benchmark table.

## Worked example: sizing a coalescing window

Suppose a session trace shows 5 requests per user over a 40-second outage, and a plausible worst case is 5,000 users in one city reconnecting within 3 seconds. That is 25,000 requests arriving in 3 seconds if nothing is deduplicated.

Now apply a coalescing key of `sessionId + hash(normalisedRequest)` with a 5-second in-flight window. If the client sends retries of the *same* logical request, they collapse to one. If the client attaches a fresh nonce or timestamp to each retry — a common idempotency pattern — nothing collapses, and you are back to 25,000.

The arithmetic is the point: coalescing only works when the key is stable across retries. Normalise the request before hashing, or send the idempotency key as a separate header that the client reuses. Then re-run the trace and count how many distinct keys arrive. That count, not the raw request count, is what your origin has to absorb.

## Code: the same loop, two ways

An edge session handler that truncates history to stay inside a storage ceiling:

```javascript
// Edge session object: per-session state, history truncated to bound size
export class AgentSession {
  constructor(state, env) {
    this.state = state;
    this.env = env;
  }

  async fetch(request) {
    const { message } = await request.json();
    const history = (await this.state.storage.get("history")) || [];
    history.push({ role: "user", content: message });

    // Keep only the most recent turns to stay within the per-object limit.
    const truncated = history.slice(-4);

    const reply = await this.env.AI.run("@cf/meta/llama-3-8b-instruct", {
      messages: truncated,
    });

    truncated.push({ role: "assistant", content: reply.response });
    await this.state.storage.put("history", truncated);

    return new Response(JSON.stringify({ reply: reply.response }));
  }
}
```

A centralised equivalent with a Redis-compatible session cache:

```javascript
// Centralised agent: session history in a Redis-compatible cache
import { createClient } from "redis";
import express from "express";

const app = express();
app.use(express.json());

const redis = createClient({ url: process.env.REDIS_URL });
await redis.connect();

app.post("/agent", async (req, res) => {
  const { sessionId, message } = req.body;
  const key = `agent:${sessionId}:history`;

  const history = JSON.parse((await redis.get(key)) || "[]");
  history.push({ role: "user", content: message });

  const reply = await callLLM(history);
  history.push({ role: "assistant", content: reply });

  await redis.set(key, JSON.stringify(history), { EX: 3600 });
  res.json({ reply });
});

app.listen(3000);
```

The centralised version above has no storm protection at all. Adding in-flight coalescing is short but only correct if the key is stable:

```javascript
// In-flight coalescing. Correct only if the key is stable across retries.
const inflight = new Map();

async function processAgent(sessionId, message) {
  // ...load history, call the model, persist, return the reply
}

app.post("/agent", async (req, res) => {
  const { sessionId, message } = req.body;
  const key = `agent:${sessionId}:${hash(normalise(message))}`;

  if (inflight.has(key)) {
    return res.json(await inflight.get(key));
  }

  const promise = processAgent(sessionId, message);
  inflight.set(key, promise);

  try {
    res.json(await promise);
  } finally {
    inflight.delete(key);
  }
});
```

Two details decide whether this helps. First, `normalise(message)` must strip anything the client regenerates per attempt, or the keys never collide. Second, the `inflight` map is per-process; behind a load balancer with multiple instances, two retries can land on different processes and both run. A shared cache key with a short TTL, or sticky routing by session, closes that gap.

## Failure modes worth designing against

**State ceiling exceeded at the edge.** The symptom is a storage-limit error mid-conversation, typically after several turns with tool output. The fix is to bound what you keep, not to raise the ceiling — raising it usually means backing onto network storage, which erases the latency advantage.

**Coalescing that never fires.** The symptom is a storm that looks exactly like no protection. The cause is almost always an unstable key. Verify by logging distinct coalescing keys during a replay.

**Replay that re-runs the agent.** If the client replays a request and the server has no checkpoint, the model call is repeated. Checkpoint after each tool call so a replay resumes rather than restarts.

**Idle timeouts cutting long streams.** Gateways and load balancers close idle connections on their own schedule. If your agent can pause between tokens, raise the idle timeout above your worst-case pause and enable keepalive on both ends.

**Egress that scales with retransmission.** On a lossy link, the same bytes cross the network more than once. If your provider bills egress by volume, your effective cost per session is higher than the payload size suggests.

## Decision checklist

Answer these before committing to either stack.

1. Is the user actively waiting for tokens? Yes leans edge; no leans centralised.
2. How large is per-session state at your worst realistic turn count? If it approaches the edge ceiling, centralised is simpler.
3. What is your measured reconnection storm size? Small storms are a server-side coalescing problem; large storms are a placement problem.
4. What is your egress cost per session, including retransmissions? This frequently decides the answer.
5. Can your team debug a distributed edge runtime? If not, the latency gain may not be worth the operational cost.
6. Do you need long conversation history or large retrieval context? Both push toward centralised.

A hybrid is legitimate: serve the first few turns at the edge for responsiveness, then hand the session to a centralised worker once history grows past the edge ceiling. It is also the most complex option, and complexity is paid in bugs rather than invoices.

## FAQ

**Why does an agent connection drop after a fixed idle period?**

Gateways and load balancers close idle connections on their own schedule, and streaming agents can pause long enough between tokens to look idle. Raise the idle timeout above your worst-case inter-token gap and enable TCP keepalive on both client and server. On Node, `server.keepAliveTimeout` and `server.headersTimeout` are the two settings to align with that value.

**Should a client replay every queued request after an outage?**

No. If the agent is stateful and the messages are sequential, replaying all of them re-runs work the user no longer needs. Attach a stable idempotency key to each logical request, deduplicate server-side, and let the client send only what has not been acknowledged.

**Is edge deployment always faster?**

No. It reduces the network hop, but it caps session state and complicates debugging. If your bottleneck is model latency or prompt size, moving compute closer to the user changes little. Measure the network share of first-token latency before assuming placement is the lever.

**How do I know if my coalescing is working?**

Log the coalescing key for every incoming request during a replayed outage. If the number of distinct keys is close to the number of requests, coalescing is not firing. If it is close to the number of sessions, it is.

**Does compression help on a lossy link?**

It reduces payload size, which reduces the number of packets and therefore the chance of loss, but the benefit depends on your payload. Measure bytes on the wire before and after enabling it rather than assuming a ratio.

## Do this in the next 30 minutes

Open your origin or load balancer logs and run one query: for each client identity, count requests arriving within 5 seconds of a connectivity restoration event. If the maximum is above roughly 100, add an in-flight coalescing map to your agent endpoint with a 5-second window and a key built from a normalised request plus the session id. Then replay one recorded outage trace and confirm the distinct-key count is close to the session count rather than the request count. That single measurement tells you whether your remaining problem is coalescing or placement.
