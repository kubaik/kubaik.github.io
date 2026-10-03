# Trim LLM bills: 3 FinOps moves that work

Most cost-control advice for LLM services is written for a clean environment and a patient timeline. It stops at "track tokens and set budget alerts." That advice is fine for a prototype. It falls apart once you have real traffic, because LLM traffic is not CRUD: it is iterative, unpredictable, and often wasteful. A single mis-routed prompt can fan out into several parallel tool calls, each burning tokens while the client waits.

Three levers move the needle in production: request shaping, cache placement, and queue discipline. Everything else is tuning.

## Why token dashboards are not a cost control

A token counter tells you what you already spent. It does not change what the next request costs. The documented behavior of most managed inference APIs is that you are billed for input tokens plus output tokens, and that long-context requests are the expensive ones. Nothing in the billing model rewards you for sending a well-formed prompt.

A typical failure mode looks like this: a request arrives with a compound question, the model generates a long internal reasoning chain, the client times out at 30 seconds, and the server keeps generating until it hits its own output limit. The client never sees the answer, but the tokens were produced and billed. Multiply that by a retry policy and the same user question can be paid for three times.

The first useful measurement is therefore not "tokens per day." It is **tokens billed per successful response delivered**. Instrument two counters at your gateway: tokens billed (from the provider's usage field in the response) and responses returned with a 2xx status to the client. The ratio is your waste indicator. If a request times out client-side but completes server-side, you are paying for output nobody reads.

## Lever 1: Request shaping

Request shaping means rewriting or splitting a prompt before it reaches the model. Two shapes are worth doing:

1. **Split compound questions.** "What is my balance and my recent transactions?" becomes two independent calls. Each call has a smaller context, so each is cheaper, and the two can run in parallel.
2. **Strip redundant context.** Conversation history that repeats the same system preamble on every turn is a common source of wasted input tokens. Keep a stable system prompt and send only the delta.

A shaping function does not need a model to do its job. A deterministic splitter is cheap, testable, and has no failure mode of its own. Here is a minimal Node implementation:

```javascript
// prompt-shaper.js
const COMPOUND = /\s+and\s+(?=(?:what|how|when|where|why|who|show|list|give)\b)/i;

export function shapePrompt(prompt) {
  const parts = prompt.split(COMPOUND).map(s => s.trim()).filter(Boolean);
  if (parts.length <= 1) return [prompt];
  return parts.map(p => `Answer only this question: ${p}`);
}

// shapePrompt("What is my balance and what are my recent transactions?")
// => ["Answer only this question: What is my balance",
//     "Answer only this question: what are my recent transactions?"]
```

Two caveats. First, splitting changes semantics: a question that depends on the answer to the first half must not be split. Gate the splitter on an explicit list of independent question types rather than on a general conjunction. Second, splitting multiplies request count, which increases per-request overhead (TLS, routing, model warm-up). Only split when the combined prompt is large enough that the context reduction outweighs the extra round trips.

**How to measure whether shaping helps:** log `input_tokens` and `output_tokens` per request before and after the change, grouped by a stable request-type label. Compare the median, not the mean, because a handful of large prompts will dominate the average. A split that reduces median input tokens but raises median latency is a trade you should make explicitly, not by accident.

## Lever 2: Cache placement

Caching model responses is where teams most often pick the wrong layer. A distributed cache with a short TTL looks fast on paper, but the dominant cost on a cache miss is not the network round trip. It is the model's time-to-first-token, which on a self-hosted model includes any GPU cold start or queue wait. A cache that saves 5 ms of network but still triggers a 300 ms model call has not saved much.

The practical pattern is two layers:

- **In-process cache** for the hottest keys, bounded by entry count and entry size. This avoids a network hop entirely.
- **Distributed cache** for the long tail of keys shared across instances.

The cache key must include everything that changes the answer: the shaped prompt, the model identifier, and the model version. Omitting the version means a model upgrade silently serves stale answers.

```javascript
// cache-layer.js
import { LRUCache } from 'lru-cache';
import { createClient } from 'redis';

const local = new LRUCache({ max: 500, maxSize: 8 * 1024 * 1024, sizeCalculation: (v) => v.length });
const redis = createClient({ url: process.env.REDIS_URL });
await redis.connect();

export async function getCached(key) {
  const hit = local.get(key);
  if (hit !== undefined) return hit;
  const remote = await redis.get(key);
  if (remote !== null) {
    local.set(key, remote);
    return remote;
  }
  return null;
}

export async function setCached(key, value, ttlSeconds = 300) {
  local.set(key, value);
  await redis.set(key, value, { EX: ttlSeconds });
}
```

The `lru-cache` option `maxSize` with `sizeCalculation` bounds total bytes, not just entry count, which matters when responses vary in length. A fixed entry count with unbounded value size is a memory leak waiting for a long response.

**How to measure cache value:** instrument three counters — local hits, remote hits, misses — and compute the hit ratio per layer. Then compute the cost avoided: `(local_hits + remote_hits) * mean_cost_per_model_call`. If the remote hit ratio is high but the local hit ratio is near zero, your hot set is larger than your local bound, or your instances are not seeing repeat traffic.

## Lever 3: Queue discipline

A first-in-first-out queue lets one large request delay every small request behind it. The failure is not throughput; it is tail latency and idle GPU time. While a 10,000-token summarization runs, a queue of one-line questions waits, and the user-visible latency for those small requests is dominated by the large one ahead of them.

Priority queuing fixes this by ordering work by estimated cost rather than arrival time. A simple version uses a visibility timeout as a delay: set a short visibility for low-priority messages so they become visible again later, and a long visibility for high-priority ones.

```python
# queue_worker.py
import boto3
from token_estimator import estimate_tokens  # your tokenizer wrapper

sqs = boto3.client("sqs")

def receive(queue_url):
    return sqs.receive_message(
        QueueUrl=queue_url,
        MaxNumberOfMessages=10,
        WaitTimeSeconds=1,
        MessageAttributeNames=["All"],
    ).get("Messages", [])

def classify(prompt):
    tokens = estimate_tokens(prompt)
    if tokens < 200:
        return "high", 0        # immediately visible
    if tokens < 2000:
        return "normal", 5      # visible after 5s
    return "low", 30            # visible after 30s

def enqueue(queue_url, prompt, body):
    tier, delay = classify(prompt)
    sqs.send_message(
        QueueUrl=queue_url,
        MessageBody=body,
        DelaySeconds=delay,
        MessageAttributes={"Tier": {"StringValue": tier, "DataType": "String"}},
    )
```

Two honest warnings about this pattern. First, `DelaySeconds` is capped at 900 seconds by the SQS API, and the delay is applied at send time, so a message already in flight cannot be re-prioritized without re-sending it. Second, starvation is real: if high-priority traffic never pauses, low-priority work never runs. The standard mitigation is aging — after a message has been deferred N times, promote it one tier. Cap N explicitly and log every promotion so you can see whether starvation is happening.

**How to measure queue discipline:** record queue wait time (time from enqueue to first token) per tier, and record GPU utilization separately from GPU busy time. If utilization is low while wait time is high, the scheduler is the problem, not capacity.

## Failure modes and how to detect them

**Cache stampede.** When a popular key expires, every concurrent request misses and calls the model at once. Detect it by charting misses per second against cache expiry events. Mitigate with a short random jitter on TTL, or by serving a slightly stale value while one request refreshes the key.

**Prompt drift.** Users rephrase over time, so a cached answer becomes subtly wrong. Detect it by sampling cache hits and comparing the stored prompt to the incoming one with a similarity threshold. If you add a similarity check, measure its latency cost separately; an embedding call per request can cost more than the cache saves.

**Queue starvation.** Detect it with the promotion counter described above. If promotions exceed a small fraction of total messages, your tier boundaries are wrong.

**Model version drift.** A prompt that works on one model version can behave differently on the next. Pin the version in the cache key and log every miss. This is cheap and prevents a class of silent correctness bugs.

**Retry amplification.** A client-side timeout that triggers a retry while the server is still generating doubles your cost for one user-visible answer. Detect it by comparing server-side completion counts to client-side success counts per request ID.

## Measuring cost honestly

Do not trust a single before/after table from someone else's system. Build your own with three steps.

1. **Establish a baseline over a fixed window.** Pick seven days of steady-state traffic. Export per-request records with: timestamp, model ID, input tokens, output tokens, cache layer hit (none/local/remote), queue wait, and total latency.
2. **Change one lever.** Deploy shaping only, or caching only. Changing all three at once makes attribution impossible.
3. **Compare distributions, not averages.** Report p50 and p95 for tokens and latency, plus the hit ratio per cache layer. Averages hide the expensive tail that usually drives the bill.

The only arithmetic worth doing is from your own numbers. For example, if your baseline shows 12,000 requests per day at a mean of 2,000 billed tokens, that is 24,000,000 tokens per day. If shaping reduces the median input by 30 percent on the half of requests that are compound, the expected daily reduction is `12,000 * 0.5 * 0.3 * (mean input tokens per request)` — plug in your own mean input figure, because it is not the same as total billed tokens.

## When this approach is wrong

Three-lever shaping, caching, and queuing pays off when you have repeated traffic, a small number of dominant prompt shapes, and control over the request path. It is the wrong choice when:

- **Every prompt is unique.** Creative writing, legal drafting, and one-off analysis have near-zero cache hit rates. A priority queue still helps, but caching does not, and the added complexity is pure overhead.
- **You are on a fully managed endpoint.** If the provider owns the queue and the cache, you can only shape and monitor on your side. Shaping adds a hop; measure whether it is worth it.
- **Traffic is tiny.** Below a few hundred requests a day, the operational cost of running a cache and a priority queue exceeds the token savings. Use a single queue and revisit later.
- **Latency is the only constraint.** Caching adds a lookup on every request. If your SLA is tight and your hit rate is low, the lookup is a tax with no rebate.

## A decision checklist

Before implementing any lever, answer these:

1. What fraction of requests repeat a previously seen prompt? (If under 5 percent, skip caching.)
2. What is your p95 input token count, and how much of it is repeated context? (If most, shaping pays.)
3. What is your queue wait time at p95, and is it correlated with prompt size? (If yes, priority queuing pays.)
4. Can you attribute cost per request ID end to end? (If not, fix observability first.)
5. What is your rollback plan if shaping changes answer quality? (Keep the splitter behind a flag.)

## The one thing to do in the next 30 minutes

Open your gateway logs for the last 24 hours and compute a single number: billed tokens divided by successful client responses. If that ratio is meaningfully above your mean tokens per prompt, you are paying for output nobody received. Find the top three request IDs by billed tokens where the client never got a 2xx response, and check whether a client timeout triggered a retry. That one query tells you whether your problem is shaping, caching, or simply retry policy — before you write any new infrastructure.
