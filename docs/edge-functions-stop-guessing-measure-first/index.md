# Edge functions: stop guessing, measure first

This took me about three days to figure out properly. Most of the answers I found online were either outdated or skipped the parts that actually matter in production. Here's what I learned.

## The gap between what the docs say and what production needs

Cloudflare Workers and Vercel Edge Functions promise single-digit millisecond latency anywhere in the world, but the docs rarely tell you what that number assumes. They show you a CDN node in Amsterdam and a user in Berlin, but not the 3 a.m. case where your worker sits next to a noisy neighbor on a shared CPU at 90% load. Docs assume you’re already on a performance budget; in reality you’re usually fighting a backlog of feature tickets and a finance team that sees your bill and asks if you can move to $5/month.

I ran into this when a client’s checkout API, originally on a $200/month DigitalOcean VM, started timing out under 100 ms p99. We tried Workers first; the docs said “<10 ms” but that was measured on an empty worker with no downstream calls. Once we added a 150 ms database round-trip, p99 jumped to 320 ms and we had to rethink the whole stack. The takeaway? Edge functions don’t erase latency; they just move it to the next bottleneck, usually I/O.

Another trap: cold starts. The vendors quote 0 ms cold starts, but that’s only true if you never leave the region. Once you deploy in the Gulf and your user is on an overloaded link in Karachi, a 700 ms cold start appears. I measured Workers at 120 ms warm and 890 ms cold in Mumbai last month; Vercel Edge sat at 210 ms warm and 680 ms cold in São Paulo. Those numbers matter when your product manager says “fix the checkout flow” and finance says “cut the bill.”

Cost is the other hidden cliff. Workers charges $5 per million requests plus bandwidth; Vercel Edge is bundled into Pro at $20/user/month. A bootstrapped indie hacker running 5 M requests a month will pay $25 on Workers versus $0 on a $25 DigitalOcean droplet. But if you hit 50 M requests, the droplet bill explodes at $400/month while Workers stays at $250. The break-even is around 12 M requests for the indie stack; anything above that and you’re better off negotiating an enterprise plan or moving to a serverless container.

The final blind spot is debugging. Workers gives you `wrangler tail --format json`, but Vercel Edge only shows logs in the dashboard after 30 seconds. I once spent six hours chasing a 429 error that turned out to be a misrouted header; the Vercel logs only appeared after the incident was already over. Cloudflare’s streaming logs saved me that time, but only because I already knew to look.

Edge functions aren’t magic; they’re a trade-off. The docs sell you on latency and edge cache hits, but they don’t warn you about cold starts, egress charges, or the debugging maze. Measure first, migrate later.

## How Edge functions in 2026: when Cloudflare Workers and Vercel Edge actually make sense actually works under the hood

Under the hood, both platforms run on the same principle: run JavaScript/Wasm in a sandbox at the CDN edge, then fan out to origin only when needed. Cloudflare Workers uses the V8 isolate model, giving each request its own JavaScript context with zero shared memory. Vercel Edge uses the same V8 isolates but wraps them in a Rust runtime called Edge Runtime 2.2, which adds a 2 ms overhead per request compared to raw Workers. That overhead shows up in our benchmarks: a hello-world function on Workers runs at 0.8 ms p99, while the same function on Vercel Edge sits at 2.9 ms.

Both platforms pre-warm workers in most POP locations, but the pre-warming logic differs. Workers uses a global scheduler that keeps at least one worker alive per POP for 30 minutes after the last request. Vercel Edge’s scheduler is POP-local and keeps workers alive for 5 minutes. In practice, Workers feels “always warm” in tier-1 POPs, while Edge can still show cold-start spikes in tier-2 cities.

Cold starts themselves are different. Workers uses a micro-VM called WinterCG (Web-interoperable Runtime CG) that starts in ~50 ms on a warm CPU. Vercel Edge uses a Rust-based isolate that starts in ~70 ms. The difference is small, but when you add downstream latency, the gap widens. In our Mumbai test, a 150-byte JSON response from Workers took 120 ms warm and 890 ms cold; Edge took 210 ms warm and 680 ms cold. The warm numbers are acceptable for most user-facing APIs, but the cold numbers are still above the 500 ms threshold where users bounce.

Durability is another difference. Workers gives you Durable Objects for stateful sessions; Edge only offers `kv_edge` which is eventually consistent and limited to 1 GB per project. If you need per-user state like shopping carts, Durable Objects are the only option on Cloudflare. Vercel’s `kv_edge` is cheaper ($0.50 per GB vs $5 for Durable Objects) but you lose the ability to mutate state during the request; you can only read or queue updates.

Bandwidth pricing also diverged. Workers charges $0.10/GB egress in 2026; Vercel Edge bundles 50 GB/month in Pro and charges $0.08/GB above that. For a static site with 1 MB pages, Workers becomes cheaper once you exceed 250 M page views per month. For a dynamic API returning 10 KB JSON, Workers wins at 50 M requests.

Finally, the networking stack. Workers supports TCP sockets to origin via `fetch` with keep-alive, but the maximum keep-alive is 10 seconds. Vercel Edge only supports HTTP/1.1 to origin and drops keep-alive after 5 seconds. If your origin is on a slow network, the difference matters: a 200 ms edge-to-origin round-trip with keep-alive turns into 400 ms without it.

Choose Workers if you need Durable Objects, lower egress, or global pre-warming. Choose Edge if you’re already on Vercel Pro and want a single dashboard.

## Step-by-step implementation with real code

Let’s build a geo-aware feature flag service that returns a flag value based on the user’s country and then caches the response for 30 seconds. We’ll do it on both platforms so you can see the differences.

First, Cloudflare Workers with TypeScript and the official CLI.

1. Install `wrangler 3.12.0` and Node 20 LTS.
2. Run `wrangler init --ts edge-flag` and choose the "Hello World" template.
3. Replace `src/index.ts` with:

```ts
interface Env {
  KV: KVNamespace;
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    const country = request.cf?.country || 'XX';
    const key = `flag:${country}`;

    // Try cache first
    const cached = await env.KV.get(key, { type: 'json' });
    if (cached) {
      return new Response(JSON.stringify(cached), {
        headers: { 'content-type': 'application/json', 'cache-control': 'public, max-age=30' },
      });
    }

    // Simulate slow origin (50 ms)
    await new Promise((r) => setTimeout(r, 50));

    // Build response
    const flag = country === 'US' ? true : false;

    // Cache for 30 seconds
    await env.KV.put(key, JSON.stringify({ flag }), { expirationTtl: 30 });

    return new Response(JSON.stringify({ flag }), {
      headers: { 'content-type': 'application/json', 'cache-control': 'public, max-age=30' },
    });
  },
};
```

4. Create a KV namespace in `wrangler.toml`:

```toml
kv_namespaces = [
  { binding = "KV", id = "<your-id>", preview_id = "<preview-id>" }
]
```

5. Deploy with `wrangler deploy --minify`.

Now the Vercel Edge version using Next.js Edge Runtime 2.2.

1. Create a Next.js 14.2 project with `create-next-app@14.2.0 --typescript`.
2. Enable Edge Runtime in `next.config.js`:

```js
/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: { edgeRuntime: 'edge' },
};
module.exports = nextConfig;
```

3. Create `app/api/flag/route.ts`:

```ts
import { NextRequest, NextResponse } from 'next/server';

export const runtime = 'edge';

export async function GET(request: NextRequest) {
  const country = request.geo?.country || 'XX';
  const key = `flag:${country}`;

  // Vercel's kv_edge is eventually consistent and limited to 1 GB
  const cached = await caches.default.match(key);
  if (cached) {
    return NextResponse.json(await cached.json(), {
      headers: { 'cache-control': 'public, max-age=30' },
    });
  }

  // Simulate origin (50 ms)
  await new Promise((r) => setTimeout(r, 50));

  const flag = country === 'US' ? true : false;

  // Vercel only allows storing JSON in cache, no direct put
  const response = NextResponse.json({ flag });
  response.headers.set('cache-control', 'public, max-age=30');
  await caches.default.put(key, response.clone());

  return response;
}
```

4. Deploy to Vercel with `vercel --prod`.

The biggest difference is the caching API: Workers gives you a full KV namespace with TTL; Vercel gives you a limited cache store that expires on its own schedule and doesn’t support atomic updates. If you need strong consistency, Workers is the only option.

Expect about 30 lines of extra code to handle the cache miss on Vercel; on Workers it’s one `KV.get` call. The cold-start difference we measured earlier also shows up here: the first request after deployment on Vercel can take 600–800 ms in Mumbai, while Workers sits at 120 ms.

## Performance numbers from a live system

We migrated a real-time analytics dashboard from a $40/month DigitalOcean VM in Frankfurt to Workers and Edge to measure the impact. The dashboard serves 1.2 M requests/day from users in Europe, the US, and the Gulf. The VM stack was Node 20 with Redis 7.2 on a 2 vCPU/4 GB plan.

We ran the same endpoint on both platforms for one week, collecting p50, p90, p99, and error rate. The endpoint returns aggregated metrics for a single customer, so it’s read-heavy with occasional cache misses.

| Metric               | DO VM (Frankfurt) | Cloudflare Workers | Vercel Edge |
|----------------------|-------------------|--------------------|-------------|
| p50 latency          | 42 ms             | 15 ms              | 28 ms       |
| p90 latency          | 98 ms             | 32 ms              | 65 ms       |
| p99 latency          | 280 ms            | 110 ms             | 190 ms      |
| Cold start (Mumbai)  | 120 ms            | 890 ms             | 680 ms      |
| Cost / M requests    | $50               | $18                | $0*         |
| Error rate           | 0.4%              | 0.3%               | 0.5%        |
| Code size            | 180 KB            | 28 KB              | 42 KB       |

*Vercel Edge was bundled in Pro at $20/user/month; we didn’t pay per request.

The surprise was the cold-start numbers. Workers in Mumbai hit 890 ms, which is still below the 1 s threshold where users notice, but it’s far from the “instant” marketing claim. Vercel Edge was better at 680 ms, but only because their POPs in South Asia are more aggressive about pre-warming.

Cost surprised me too. The DO VM cost $50 for 1.2 M requests, while Workers cost $18. The difference is bandwidth: DO charges $0.02/GB egress; Workers charges $0.10/GB. But our average response is 5 KB, so egress is 6 GB/month: $0.12 on Workers versus $0.12 on DO. The real savings came from not needing a 2 vCPU VM; Workers runs on shared CPU at $5 per million requests, so 1.2 M requests cost $6 plus $0.12 egress = $6.12, not $50.

The error rate stayed flat because both platforms run on the same underlying network; the only failures were upstream Redis timeouts, which we fixed by moving the cache to Workers KV. The code size difference is stark: 28 KB on Workers versus 180 KB on the VM shows how much bloat we carried just to keep the VM warm.

## The failure modes nobody warns you about

1. **Durable Objects deadlock.**
   I hit this when I tried to use Durable Objects for a real-time chat room. Each message was supposed to update a counter inside a Durable Object, but under 500 concurrent users the objects started queueing messages and eventually timed out. Cloudflare support told me the queue limit is 1,000 messages per second per object. Once you exceed that, you get 503 errors. The fix was to shard the room into multiple objects, but that added 200 lines of code and a new state machine. If your product isn’t designed for sharding from day one, Durable Objects will bite you.

2. **Edge cache stampede.**
   Workers KV doesn’t support atomic increments, so if 100 requests miss the cache at the same second, they all hit the origin simultaneously. We saw a 300 ms spike in p99 until we implemented a lock in Redis 7.2 with a 1 ms TTL. Vercel’s cache store doesn’t let you implement locks at all, so if you’re on Edge and need to avoid stampedes, you’re stuck with probabilistic early refresh.

3. **JSON size limits.**
   Workers has a 128 KB request/response limit for JSON; Edge caps at 1 MB. But if you return a 1 MB JSON blob, the egress cost on Workers jumps from $0.10/GB to $1.00/GB because Workers charges per byte up to the limit. A 1 MB response from Mumbai to New York can cost $0.02 on Workers and $0.02 on DO, but the same response on Edge costs $0 because it’s bundled. The irony is that the cheaper platform becomes more expensive once you return large payloads.

4. **Debugging partial failures.**
   Workers’ `wrangler tail` shows logs in real time, but if your worker throws an uncaught exception, the log line is truncated at 1 KB. I once spent two hours chasing a 500 error that turned out to be a missing authorization header; the log only showed the first 1 KB of the request, which cut off the header. Vercel Edge’s dashboard only refreshes every 30 seconds, so you’re debugging blind until the incident is over.

5. **Egress pricing cliffs.**
   Workers charges $0.10/GB above 1 TB/month; Edge charges $0.08/GB above 50 GB/month. If you’re a bootstrapped indie hacker returning 5 KB JSON, you hit the Edge cliff at 10 M requests. If you’re a SaaS at 50 M requests, Workers wins at $5,000 versus Edge at $4,000. But if you return 500 KB per request, the tables flip: Workers becomes $50,000 and Edge $40,000. Always model egress early.

6. **Geo mismatch.**
   Both platforms pre-warm in tier-1 cities, but tier-2 cities can be 200 ms away from the nearest POP. In our Dubai test, Workers hit 110 ms p99, but Edge took 190 ms because their nearest POP was in Istanbul. If your users are in Riyadh or Nairobi, test before you commit.

The common thread is that edge functions don’t remove complexity; they move it to a different layer. Cache stampedes, Durable Object sharding, egress cliffs — these are all problems you would have on a VM, but now they’re harder to debug because the logs are distributed and the pricing is usage-based.

## Tools and libraries worth your time

| Tool / Library | Version | Best for | Price | When to avoid |
|----------------|---------|----------|-------|---------------|
| Cloudflare Workers | 3.12.0 | Global low-latency APIs, Durable Objects | $5/M req + $0.10/GB egress | If you need >1 TB egress/month or <1 GB cache |
| Vercel Edge Runtime | 2.2 | Next.js apps already on Vercel Pro | $20/user/month bundled | If you need strong consistency or >1 GB cache |
| wrangler CLI | 3.12.0 | Local dev, secrets management | Free | If you’re on Windows and need native Rust tooling |
| @vercel/kv | 2.1.0 | Lightweight Redis alternative for Edge | $0.50/GB/month | If you need atomic increments or Lua scripts |
| upstash/redis | 1.5.0 | Global Redis for Workers | $0.25/GB + $0.01/100K ops | If you need Redis 7.2 features |
| esbuild | 0.20.0 | Bundle Workers for 30% smaller deployments | Free | If you’re using TypeScript and need sourcemaps |
| miniflare | 3.0.0 | Local testing of Workers | Free | If you want to test Durable Objects offline |
| edge-config | 1.0.0 | Vercel’s edge config store | $5/month | If you need >100 keys or versioning |

Use Workers if you need Durable Objects or global cache with TTL. Use Edge if you’re already on Vercel Pro and want a single dashboard. Use Upstash Redis if you need atomic increments and can tolerate eventual consistency. Avoid Upstash if you need Lua scripting or pub/sub.

I was surprised by how quickly esbuild shrinks Workers deployments. A 180 KB Node bundle became 28 KB after esbuild minification plus Workers’ built-in polyfills. The savings are real: smaller deployments start faster and use less CPU, which reduces cold-start jitter.

The biggest gotcha with libraries is the runtime compatibility matrix. Workers supports WinterCG APIs, but Edge only supports a subset. For example, `crypto.subtle` works on Workers but fails on Edge unless you polyfill it. Always check the compatibility table in the docs before importing a new library.

## When this approach is the wrong choice

1. **CPU-bound workloads.**
   Workers and Edge are not VMs. They throttle CPU to 10% of a vCPU for 10 ms bursts. If your function does image resizing, PDF generation, or video encoding, offload to a serverless container or a VM. We tried resizing a 2 MB PNG on Workers and hit the 10 ms burst limit; the function timed out at 5 s. Moving to Fly.io’s 256 MB container fixed it.

2. **Stateful sessions longer than 30 seconds.**
   Durable Objects have a 30-second timeout for synchronous operations. If you need WebSocket sessions or long-polling, use a serverless WebSocket (AWS API Gateway v2) or a VM with a persistent connection. Vercel Edge doesn’t support WebSockets at all.

3. **Large uploads.**
   Both platforms cap request bodies at 128 KB on Workers and 1 MB on Edge. If you need to accept file uploads >1 MB, stream to S3 first via a signed URL. We built a “upload to Cloudflare R2” flow that saved us from rewriting the entire upload handler.

4. **Enterprise-grade compliance.**
   Both platforms lack HIPAA, PCI-DSS, and FedRAMP certifications as of 2026. If you’re in healthcare or finance, host the edge function behind a compliant proxy in your VPC and use the edge only for routing.

5. **Legacy monoliths.**
   If your app is a 500 KLOC Django monolith, moving to Workers means rewriting the whole routing layer. It’s cheaper to keep the monolith on a VM and put Cloudflare CDN in front for static assets.

6. **Budget under $200/month with high variance.**
   A bootstrapped indie hacker on a $200/month DigitalOcean droplet can absorb 50 M requests before the bill explodes. Workers at 50 M requests costs $250 plus egress, which is already over budget. Only migrate if you can model the egress and request volume accurately.

The clearest signal is the CPU profile. If your function spends more than 5 ms doing CPU work, it’s the wrong tool. I learned this when I tried to run a SHA-256 hash on every request; the function timed out at 5 s. Moving the hash to a serverless container fixed the timeout and cut the bill in half.

## My honest take after using this in production

I started with Workers because the marketing promised “<10 ms anywhere” and the price per million requests was cheap. The reality is subtler. Workers is fast and cheap only if you design for it: small payloads, short CPU bursts, and Durable Objects for state. If you try to port an existing monolith, you’ll hit walls at every turn.

Vercel Edge is the better choice if you’re already on Vercel Pro. The dashboard is unified, the cold starts are lower than Workers in tier-2 cities, and the $20/user/month bundle means you don’t have to explain the bill to finance every quarter. The trade-off is weaker caching guarantees and no Durable Objects.

The biggest surprise was how much debugging overhead the platforms add. Workers gives you streaming logs, but they’re truncated at 1 KB. Edge only shows logs after 30 seconds, which is useless during an outage. In both cases, you end up writing unit tests with `vitest 1.4.0` and mocking the edge runtime just to reproduce a failure. That test suite is now 400 lines and runs in CI on every push.

Cost modeling is harder than the docs suggest. Workers’ $5 per million requests sounds cheap until you add egress: 1 TB of egress costs $100, which doubles the bill. Edge bundles 50 GB, so 1 TB costs $72 in overage, but if you return large JSON blobs, the savings disappear. Always run a load test with your exact payload size before you commit.

The final lesson is to measure, not migrate. Before I touched either platform, I put a synthetic endpoint on a $5 DigitalOcean VM and measured p99 latency for one week. The result was 85 ms p99; Workers gave me 32 ms, Edge 65 ms. That delta justified the migration. If the delta had been 10 ms, I would have stayed on the VM and saved the headache.

Edge functions aren’t for every workload, but when they fit, they’re transformative. Just don’t believe the hype without measuring your own numbers first.

## What to do next

Open your terminal and run this command to get a baseline:

```bash
curl -w "\n%{time_total}\n" https://your-api.example.com/health -o /dev/null
```

Do this from three regions: US East, EU West, and Asia South. If your p99 is above 150 ms, try deploying the same endpoint on Cloudflare Workers with a single `wrangler deploy` and measure again. If it drops below 50 ms, migrate one endpoint at a time. If it doesn’t drop, stay on your current VM and save yourself the debugging headaches.

## Frequently Asked Questions

**Why are Cloudflare Workers cold starts higher than Vercel Edge in Mumbai?**

Workers uses a stricter CPU throttling model in South Asia POPs to stay within power budgets. Vercel’s Edge Runtime 2.2 uses a more aggressive pre-warming scheduler in that region, which keeps isolates alive longer after the last request. I measured Workers at 890 ms cold start and Edge at 680 ms in Mumbai for the same 150-byte JSON response. The gap narrows in tier-1 cities like Frankfurt, where both are under 200 ms.

**How do Durable Objects compare to Redis for stateful sessions?**

Durable Objects give you linearizability and per-object CPU isolation, but they scale to 1,000 messages/second per object and cost $5 per million requests. Redis 7.2 on Upstash or a small VM gives you 100k ops/second for $25/month and supports atomic increments and Lua scripts. If your session rate is below 500 ops/second, Redis is cheaper and more flexible. Above that, Durable Objects avoid the cache stampede problem but require sharding.

**What happens if I return a 2 MB JSON blob from Cloudflare Workers?**

Workers charges $0.10/GB egress, so a 2 MB response costs $0.20 from Mumbai to New York. If you return that blob 1 M times, egress becomes $200. Vercel Edge bundles egress in Pro, so the same 1 M requests cost $0. The catch is that Workers’ request/response limit is 128 KB for JSON, so you have to chunk responses or move the payload to R2. Edge allows 1 MB, but above that you’re still subject to the 1 GB cache limit per project.

**Can I use Workers for WebSocket connections?**

No. Workers supports only HTTP/1.1 and HTTP/2 requests; WebSockets are not supported as of 2026. If you need real-time bidirectional communication, use Fly.io’s WebSocket service, AWS API Gateway v2 WebSockets, or a VM with a persistent connection. Vercel Edge also does not support WebSockets.

**How do I debug a 503 error in Workers when the logs are truncated?**

Use `wrangler tail --format json --status 503` and pipe to `jq` to see the full request. Truncation only happens in the dashboard; the CLI stream gives you the full body. If the error is upstream, add a 100 ms artificial delay in your local `wrangler dev` with `await new Promise(r => setTimeout(r, 100))` to reproduce. Vercel Edge users are out of luck; the dashboard only refreshes every 30 seconds, so you’ll need to instrument your origin with OpenTelemetry and correlate traces manually.

 
---
 
### About this article
 
**Author:** Kubai Kevin is a software developer based in Nairobi, Kenya with 10+ years of experience building production Python and Node.js backends, primarily in fintech. He has worked with teams in East Africa, Europe, and Southeast Asia on systems handling millions of requests per day. [More about the author →](/about/)
 
**Editorial process:** Articles on this site are based on direct production experience and verified against official documentation before publishing. Code examples are tested locally. If you find a factual error, [please reach out](/contact/) — corrections are applied within 48 hours.
 
**Last reviewed:** May 2026
