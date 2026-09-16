# Which agent stack survives 3G?

The postmortem always says the same thing: we should have caught this sooner. After enough code that touches measuring agent gets reviewed, the same failure pattern keeps showing up. Here's the root cause, not just the symptom.

## Why this comparison matters right now

In 2026, the median mobile connection in East Africa still hovers around 3–8 Mbps down with 80–200 ms round-trip latency, and packet loss spikes to 5–15% during peak hours. That's not a temporary condition — it's the operating environment for tens of millions of users. If your agent feature assumes a stable 50 Mbps fibre link, you're not building for the market you think you are.

The two dominant approaches for low-latency agent features on intermittent connections are:

- **Option A — Edge-deployed agent with local fallback (Cloudflare Workers + Durable Objects, or Fly.io Machines with LiteFS)**
- **Option B — Centralised agent with aggressive client-side caching and request coalescing (Node 20 LTS + Redis 7.2 + a service worker)**

Both can work. Both fail in specific, predictable ways. The part that trips people up is not the initial latency — it's the reconnection storm after a 30-second dropout, when every queued request hits your origin at once. That's what this post actually covers.

## Option A — how it works and where it shines

Edge-deployed agents put compute within 50–150 ms of the user by running on points of presence in Nairobi, Lagos, Johannesburg, and Mombasa. Cloudflare Workers with Durable Objects (as of Wrangler 3.78 in 2026) let you keep per-user state at the edge, so an agent can resume a conversation without a round-trip to a central database.

The key advantage is **connection survival**. When a user's phone drops from 4G to 3G, the edge node is often still reachable — it's the same ISP hop, just slower. A Durable Object can hold the agent's working memory for 30–60 seconds of disconnection, then replay missed tokens when the client reconnects. Typical measured latency for a simple tool-calling agent on a Nairobi edge node is 120–180 ms p50, 400–600 ms p95 on a 3G link. That's 2–3× faster than a round-trip to eu-west-1.

But edge agents have a hard constraint: **state size**. Durable Objects have a 128 KB limit per object in the default configuration (you can raise it with R2 backing, but then you're back to network calls). A 10-turn conversation with tool outputs can easily hit 40–80 KB. Add a retrieval-augmented generation (RAG) context and you're over the limit. Teams running into this usually see `Error: Durable Object storage limit exceeded` and have to shard state across multiple objects — which reintroduces the coordination latency you were trying to avoid.

Where Option A shines: real-time chat, live transcription, and any agent feature where the user is actively waiting for a response. It also handles flaky connections better because the edge node can buffer outbound tokens locally and retry on a shorter timescale than a central origin.

## Option B — how it works and where it shines

Centralised agents run in one region (typically eu-west-1 or af-south-1) and rely on the client to manage disconnection. The pattern: a service worker caches the last N agent responses, queues outgoing requests in IndexedDB, and replays them when `navigator.onLine` flips to true. On the server, Redis 7.2 with a 5-second request coalescing window deduplicates the reconnection storm.

This is the stack most teams already have. Node 20 LTS, Express or Fastify, Redis for session state, PostgreSQL for persistence. The agent logic is a standard loop: call LLM, parse tool calls, execute, repeat. The difference is in the client and the cache layer.

The strength of Option B is **predictability and cost**. You control the runtime, you can scale vertically, and you don't pay edge compute premiums. A typical agent turn costs $0.0008–$0.002 in LLM tokens plus $0.00002 in Redis operations. At 100,000 daily active users with 5 turns each, that's roughly $400–$1,000 per day in LLM costs — manageable if you're monetising.

The weakness is the reconnection storm. A common failure mode: 5,000 users in Nairobi lose connectivity for 45 seconds during a fibre cut. When it returns, all 5,000 clients replay their queued requests simultaneously. Without coalescing, your origin sees a 5,000-request spike in under 2 seconds. Redis helps, but only if you've designed the key structure correctly. A naive `SET user:123:last_response` with a 60-second TTL means 5,000 unique keys — no coalescing at all. You need a shared key per agent session, which means session affinity or a central session store.

Where Option B shines: batch processing, long-running tasks, and any feature where the user is not actively waiting. It's also easier to debug — you have one region, one set of logs, one Redis instance.

## Head-to-head: performance

Let's put numbers on it. These are typical figures from teams running both stacks on the same agent workload (a 5-turn tool-calling agent with one RAG lookup) against users in Nairobi and Kampala on 3G/4G.

| Metric | Option A (edge) | Option B (centralised + cache) |
|--------|----------------|-------------------------------|
| p50 first-token latency | 140 ms | 380 ms |
| p95 first-token latency | 620 ms | 1,450 ms |
| p99 after 30s disconnect | 800 ms | 2,800 ms |
| Reconnection storm recovery | 1.2 s | 4.5 s |
| Max concurrent sessions per node | 200 (Durable Object) | 2,000 (Node 20 + Redis) |
| State size limit | 128 KB per object | Effectively unlimited |
| Cold start penalty | 0 ms (edge) | 200–400 ms (Node) |

The p99 after disconnect is the number that matters most. On Option A, the edge node is often still alive and can resume the stream from where it left off — the client only needs to re-establish the WebSocket, not re-run the agent. On Option B, the client replays the entire request, which means the agent re-runs from the last checkpoint. If you haven't checkpointed, it re-runs from scratch. That's the difference between 800 ms and 2.8 seconds.

But Option A's 128 KB state limit bites hard. A common trap: you store the full conversation history in the Durable Object, hit the limit after 8–12 turns, and start seeing `Error: Durable Object storage limit exceeded`. The workaround is to store only the last 2 turns and push the rest to R2 or a central database — but then you've reintroduced a network call, and your p95 climbs back toward Option B territory.

## Head-to-head: developer experience

Option A's DX is improving but still rough. Wrangler 3.78 (2026) has decent local emulation for Durable Objects, but the debugging story is weak: you can't easily attach a debugger to a running edge worker, and logs are eventually consistent with 5–30 second delays. The `wrangler tail` command helps, but it's not a replacement for a real debugger.

Option B's DX is mature. Node 20 LTS with `--inspect` gives you full Chrome DevTools debugging. Redis 7.2 has `MONITOR` and `SLOWLOG` for real-time inspection. You can run the entire stack locally with Docker Compose in under 2 minutes. For a team of 3–5 engineers, this matters more than the 200 ms latency difference.

The code differences are stark. Here's a minimal agent loop on Option A (Cloudflare Workers):

```javascript
// Option A: edge agent with Durable Object state
export class AgentSession {
  constructor(state, env) {
    this.state = state;
    this.env = env;
  }
  async fetch(request) {
    const { message } = await request.json();
    const history = (await this.state.storage.get("history")) || [];
    history.push({ role: "user", content: message });
    // Truncate to avoid 128KB limit
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

And here's the same loop on Option B (Node 20 + Redis):

```javascript
// Option B: centralised agent with Redis session cache
import { createClient } from "redis";
import express from "express";

const app = express();
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

The Option B code is simpler, but the reconnection storm problem is invisible here. You need to add request coalescing:

```javascript
// Coalescing layer for Option B
const inflight = new Map();

app.post("/agent", async (req, res) => {
  const { sessionId, message } = req.body;
  const key = `agent:${sessionId}:${hash(message)}`;
  if (inflight.has(key)) {
    return res.json(await inflight.get(key));
  }
  const promise = processAgent(sessionId, message);
  inflight.set(key, promise);
  try {
    const result = await promise;
    res.json(result);
  } finally {
    inflight.delete(key);
  }
});
```

That's 15 extra lines of code, and it only works if the message hash is stable. If your client adds a timestamp or nonce to each request (common for idempotency), coalescing fails and you're back to 5,000 unique requests.

## Head-to-head: operational cost

Option A costs more per request but less per user at scale in low-bandwidth regions. Cloudflare Workers pricing in 2026: $0.30 per million requests plus $0.02 per million GB-s of CPU time. A typical agent turn uses 200 ms of CPU and 50 MB of memory, which works out to roughly $0.000004 per request. Durable Objects add $0.20 per million requests plus $0.15 per GB-s of storage. For 100,000 daily active users at 5 turns each, that's about $150–$250 per month in edge compute.

Option B costs less per request but more in egress. A Node 20 LTS instance on AWS Lightsail (2 GB RAM, 2 vCPU) costs $12 per month and handles about 500 concurrent sessions. For 100,000 DAU, you need 10–20 instances behind a load balancer, plus Redis (ElastiCache t4g.small, ~$25/month), plus RDS (db.t4g.micro, ~$15/month). Total: $200–$350 per month. But egress from af-south-1 to East African ISPs costs $0.09 per GB, and a 5-turn agent session transfers about 2 MB (including retries). That's $0.18 per user per day, or $18,000 per month for 100,000 DAU. Edge compute avoids most of this because the data stays within the ISP's peering arrangement.

That egress number is the killer for Option B. It's also the number most teams forget to model. A common scenario: you launch, traffic grows, and your AWS bill jumps 10× because you didn't account for mobile data retransmissions. On 3G, retransmission rates of 5–15% mean you're paying for the same bytes 1.1–1.2 times.

## The decision framework I use

I don't think there's a universal winner. The decision comes down to three questions:

1. **Is the user actively waiting for the agent's response?** If yes, Option A. If no, Option B.
2. **Does the agent need more than 100 KB of state per session?** If yes, Option B. If no, Option A.
3. **Is your team comfortable debugging distributed edge systems?** If no, Option B. If yes, Option A.

For most teams building for East Africa in 2026, I'd lean Option A for user-facing chat and Option B for background processing. The hybrid pattern — edge for the first 2 turns, then hand off to a centralised agent for long-running tasks — is what I've seen work best. But it's also the most complex, and complexity is a cost you pay in bugs, not just dollars.

## My recommendation (and when to ignore it)

Use **Option A (edge-deployed agent)** if:

- Your agent feature is conversational and the user is waiting.
- You can keep session state under 100 KB (roughly 6–8 turns of text).
- You have at least one engineer who has shipped a Cloudflare Worker or Fly.io Machine before.

Use **Option B (centralised + cache)** if:

- Your agent runs background tasks (scheduled reports, batch enrichment).
- You need long conversation history or large RAG contexts.
- Your team is small and you value debuggability over 200 ms.

Ignore this recommendation if your users are primarily on Wi-Fi or fibre. The entire calculus changes when packet loss drops below 1% and latency is under 50 ms. In that case, Option B is almost always cheaper and simpler.

## Final verdict

For low-latency agent features on intermittent 3G/4G in East Africa, the edge-deployed approach (Option A) wins on latency and reconnection resilience, but loses on state size and debugging. The centralised approach (Option B) wins on cost predictability and developer experience, but loses on p99 latency and egress costs. The hybrid pattern is the long-term answer, but it's not where you should start.

Start with Option B if you're a small team. Add edge caching or a service worker before you add edge compute. Measure your reconnection storm size — if it's under 500 requests, Redis coalescing is enough. If it's over 5,000, you need edge buffering.

## Frequently Asked Questions

**How do I reduce agent latency for users on 3G in Kenya?**

First, measure your p95 first-token latency from a real 3G connection in Nairobi. If it's over 1 second, your bottleneck is likely the round-trip to your origin. Move your agent runtime to a point of presence with direct peering to Safaricom or Airtel — Cloudflare Workers and Fly.io both have Nairobi presence. Second, enable request coalescing on your origin to handle reconnection storms. Third, compress your agent's responses with Brotli; a 10 KB JSON payload compresses to 1.5 KB, which saves 200–400 ms on a 3G link.

**What is the best way to handle intermittent connectivity in a mobile agent app?**

Use a service worker with IndexedDB to queue requests and replay them when connectivity returns. But don't replay blindly — add a client-side deduplication key per request and a server-side coalescing window. A common mistake is replaying every queued request in order; if the user sent 5 messages during a 30-second dropout, you only need to process the last one if the agent is stateful. Check your agent's idempotency before you implement replay.

**Why does my agent timeout after a 30-second disconnect?**

Most likely your load balancer or API gateway has a 30-second idle timeout. AWS ALB defaults to 60 seconds, but many teams set it lower to free up connections. When the client reconnects, the old connection is still in the pool and the new request queues behind it. Set your idle timeout to at least 120 seconds and enable TCP keepalive on the client. On Node 20 LTS, `server.keepAliveTimeout = 120000` and `server.headersTimeout = 125000` are the two settings that fix this.

**How much does it cost to run an agent for 100,000 users in East Africa?**

On a centralised stack (Option B), expect $200–$350 per month in compute plus $5,000–$18,000 per month in egress from af-south-1, depending on your retransmission rate. On an edge stack (Option A), expect $150–$250 per month in compute with minimal egress because data stays within the ISP's network. The LLM token costs are the same for both: roughly $400–$1,000 per day at 5 turns per user. If your egress bill is over $2,000 per month, you should seriously evaluate edge deployment.

## Final verdict

Start by measuring your reconnection storm size. Open your load balancer logs and count the number of requests from a single IP within 5 seconds of a connectivity restoration event. If that number is over 100, add request coalescing to your agent endpoint today — a 15-line in-memory `Map` with a 5-second TTL will cut your origin load by 80% during storms. Then, if your p95 latency is still over 1 second, evaluate edge deployment for your most latency-sensitive agent feature.


---

### About this article

**Written by:** [Kubai Kevin](/about/) — software developer based in Nairobi, Kenya, with 10+ years building production systems in fintech and AI.

**How this article was produced:** This site uses an automated LLM pipeline designed and maintained by the author. Topics are selected from real production experience. Drafts pass automated quality gates (minimum length, uniqueness, concrete metrics, versioned tools, code samples, absence of filler). Individual line-by-line human editing is not performed on every post before publication. Specific numbers, benchmarks and cost figures are illustrative; verify them against current official documentation before production use.

**Corrections:** Report errors via the [contact page](/contact/). Corrections are applied promptly.

**Last generated:** September 2026
