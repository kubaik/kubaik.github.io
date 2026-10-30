# Senior engineer title inflation in 2026

## What changed in the senior-engineer job description

Job descriptions for senior engineers have quietly absorbed a requirement that used to belong to platform and compliance teams: the ability to constrain, audit, and escalate around AI-generated code. The written rubric still says "write clean code, pass code reviews, mentor juniors." The actual failure surface has moved.

The mismatch shows up in mundane places. A pull request that adds one decorator to route a prompt through a managed LLM endpoint can create a data-residency problem if that endpoint resolves to a region outside the one the data is permitted to leave. The infrastructure is unchanged; the compliance surface is wider. Teams that treat this as a platform-team concern often discover it during an audit rather than during review.

Three capabilities separate engineers who handle this well from those who don't:

1. Constraining what context reaches a model, so sensitive fields never leave the boundary.
2. Defining deterministic fallback behavior when a model returns something unusable — a hallucinated API version, a malformed tool call, a refusal.
3. Producing a reconstructable record of every prompt, response, and downstream mutation, so an auditor's question can be answered without guesswork.

None of these are model-specific. They are ordinary distributed-systems problems with an unfamiliar failure mode: the "remote dependency" is nondeterministic, priced per call, and subject to rules your code review does not enforce.

## The three layers that appear when you add an LLM call

Most production LLM integrations end up with three logical layers, whether or not anyone names them that way.

| Layer | Responsibility | Typical failure mode |
|---|---|---|
| Gateway | Routes prompts to a permitted endpoint; enforces residency and model allow-lists | Endpoint selected from configuration that drifted from policy; prompts leave the permitted region |
| Orchestrator | Owns retries, timeouts, circuit breaking, and fallback shape | Retry loop has no ceiling; a degraded dependency multiplies invocation count |
| Auditor | Writes an append-only record of prompts, responses, and mutations | Log volume grows unbounded; retention and deletion obligations are never implemented |

The gateway is where policy lives. The orchestrator is where cost and latency live. The auditor is where legal exposure lives. Conflating them is the most common structural mistake, because it makes each one harder to test in isolation.

A second common mistake is assuming the LLM call is stateless. The orchestrator usually carries state — attempt counters, fallback flags, cost accumulators — and that state must be persisted transactionally alongside the side effects it guards. A missing dependency ordering between the orchestrator and its cache can produce silent retries and duplicate writes long before anyone notices the bill.

## A worked example: residency-aware gateway with bounded retries

The following is a minimal pattern for routing a call through a managed LLM endpoint while enforcing a region allow-list and logging every exchange. It uses Node.js on a serverless function, a Redis-compatible cache, and a managed model endpoint. Substitute your own provider; the structure is what matters.

### Step 1: Gateway with residency enforcement

```javascript
// gateway.js
import { BedrockRuntimeClient, InvokeModelCommand } from '@aws-sdk/client-bedrock-runtime';
import { Redis } from 'ioredis';

const REDIS = new Redis(process.env.REDIS_URL);
const BEDROCK = new BedrockRuntimeClient({ region: process.env.RESIDENCY_REGION });

const ALLOWED_REGIONS = new Set(['eu-central-1', 'eu-west-1']);

export async function callLLM(prompt, residencyTag) {
  if (!ALLOWED_REGIONS.has(residencyTag)) {
    throw new Error(`Residency tag not permitted: ${residencyTag}`);
  }
  if (residencyTag !== process.env.RESIDENCY_REGION) {
    throw new Error(`Configured region ${process.env.RESIDENCY_REGION} does not match tag ${residencyTag}`);
  }

  const cacheKey = `llm:${residencyTag}:${await hashPrompt(prompt)}`;
  const cached = await REDIS.get(cacheKey);
  if (cached) {
    return JSON.parse(cached);
  }

  const command = new InvokeModelCommand({
    modelId: process.env.MODEL_ID,
    body: JSON.stringify({ prompt }),
  });
  const response = await BEDROCK.send(command);
  const parsed = JSON.parse(new TextDecoder().decode(response.body));

  await REDIS.setex(cacheKey, 3600, JSON.stringify(parsed));
  return parsed;
}
```

Points worth noting:

- Residency is validated at call time against both an allow-list and the client's configured region. Checking only the tag lets a misconfigured client pass validation while calling the wrong endpoint.
- The cache key includes the residency tag. Two regions must never share a cache entry, because the cached value is itself data that has crossed a boundary.
- Hashing the prompt keeps the cache key bounded in length and avoids writing raw prompt text into a key namespace that may be replicated or backed up.

### Step 2: Orchestrator with a bounded retry budget

```javascript
// orchestrator.js
import { callLLM } from './gateway.js';

const MAX_ATTEMPTS = 3;
const BASE_DELAY_MS = 1000;
const DEADLINE_MS = 5000;

export async function safeLLMCall(prompt, residencyTag) {
  const startedAt = Date.now();
  let lastError = null;

  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
    if (Date.now() - startedAt > DEADLINE_MS) {
      return { ok: false, reason: 'deadline_exceeded', error: lastError?.message };
    }
    try {
      const result = await callLLM(prompt, residencyTag);
      return { ok: true, result };
    } catch (err) {
      lastError = err;
      if (attempt < MAX_ATTEMPTS) {
        const jitter = Math.random() * 250;
        await new Promise(r => setTimeout(r, BASE_DELAY_MS * attempt + jitter));
      }
    }
  }

  return { ok: false, reason: 'attempts_exhausted', error: lastError?.message };
}
```

Two details make this meaningfully different from a naive retry loop. First, the total wall-clock deadline caps the work regardless of attempt count, so a slow dependency cannot stretch a request indefinitely. Second, the return shape is explicit and total: callers must handle `{ ok: false }`. Silent fallthrough to an undefined result is how retry storms become data corruption.

### Step 3: Append-only audit record

```python
# auditor.py
import hashlib, json, os
from datetime import datetime, timezone
import boto3

dynamodb = boto3.resource('dynamodb', region_name=os.environ['RESIDENCY_REGION'])
audit_table = dynamodb.Table(os.environ['AUDIT_TABLE'])

def _digest(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()

def log_exchange(prompt: str, response: dict, residency_tag: str, subject_id: str):
    audit_table.put_item(Item={
        'pk': f"subject#{subject_id}",
        'sk': f"{datetime.now(timezone.utc).isoformat()}#{os.getpid()}",
        'prompt_digest': _digest(prompt),
        'response_digest': _digest(json.dumps(response, sort_keys=True)),
        'residency_tag': residency_tag,
        'model_id': os.environ['MODEL_ID'],
    })
```

Notes:

- Use a real cryptographic digest, not Python's built-in `hash()`. The built-in is salted per process and is not stable across runs, so it cannot support a later lookup or deletion request.
- Partitioning by subject identifier means a deletion request can be satisfied by deleting one partition rather than scanning the table.
- Storing digests rather than raw content keeps the record useful for integrity checks without turning the audit table into a second copy of the sensitive data. If your retention policy requires the content itself, store it in a separate store with its own lifecycle rules.

## How to measure the cost and latency you actually added

Published numbers from someone else's system tell you almost nothing about yours. Measure these four quantities before and after the LLM layer, on the same traffic:

- **Invocation count per logical request.** Instrument the orchestrator to emit a counter incremented once per attempt, tagged with the outcome. Compare this to your request counter. A ratio above 1.0 is retry overhead; a ratio that climbs over time means the dependency is degrading.
- **Cache hit ratio.** Emit a counter on both the hit and miss path in the gateway, tagged by residency tag. A ratio that differs sharply between tags usually indicates a cache-key bug or a stampede on a newly introduced tag.
- **Added latency at p50 and p99.** Record the wall-clock time around the model call only, not the whole handler. This isolates the remote dependency from your own processing and tells you whether a timeout is even reachable.
- **Cost per successful request.** Multiply invocations by your provider's per-token price using token counts from the response metadata. Divide by successful requests, not total requests, or a failing dependency will look cheap.

The arithmetic is straightforward once you have the counters. If a service handles 1,000 requests per minute and the invocation ratio is 1.4, the model endpoint sees 1,400 calls per minute. At a hypothetical $0.003 per call, that is 1,400 × 0.003 = $4.20 per minute, or about $6,048 per day — illustrative figures, but the method is the point. Caching to a 70% hit ratio would cut the call volume to roughly 420 per minute and the cost proportionally.

## Failure modes worth designing against

**Cache stampede on a new residency tag.** When a new tag is introduced, every request for it misses the cache simultaneously. The result is a burst of model calls and, if the cache write path is slow, a queue of pending writes. Mitigations: pre-warm the cache for a new tag before routing production traffic to it, and stagger TTLs with jitter so entries do not all expire at the same instant.

**Header injection through routing metadata.** If the residency tag or routing header is derived from user input, a crafted value can select an unintended endpoint. Validate against a strict allow-list of exact values; never parse or normalize the value before comparison, and never interpolate it into a URL or region string.

**Unbounded audit growth.** Audit tables grow at a rate proportional to traffic, and a table with no lifecycle policy will eventually hit storage limits or retention rules that require deletion. Define the retention period first, then implement archival and deletion as scheduled jobs. If a deletion obligation exists, the partition key must be chosen so that deletion is a single-partition operation.

**Cost attribution without tags.** When the bill for the model endpoint appears, it is rarely separable by service unless every call carries a cost-center or service tag. Emit that tag at the orchestrator, not at the provider console, so it is present even if the provider's own tagging is incomplete.

**Nondeterministic output treated as deterministic.** A model may return a plausible but nonexistent API version, a tool call with wrong argument types, or a response that parses as JSON but violates the schema. Validate the response shape before applying any mutation, and treat validation failure as a normal error path, not an exception.

## When not to route through an LLM at all

The pattern above adds latency and per-call cost. It is the wrong choice when:

- The operation is on a strict sub-millisecond path, such as order matching or real-time bidding.
- The workload is high-volume batch processing, where per-record cost dominates and a small number of failures is tolerable.
- The data never touches personal or regulated information, so residency routing adds complexity without reducing risk.
- The surrounding system has no durable store or cache, so the audit and caching layers would have to be built from scratch.

In those cases, keep the model behind a feature flag and route only interactive paths through the constrained gateway. The flag is not just a rollout mechanism; it is the switch that lets you disable the dependency during an incident without a deploy.

## A decision checklist before merging an LLM integration

- Does the call carry an explicit residency tag, and is that tag validated against an allow-list at call time?
- Is the configured endpoint region checked against the tag, or only the tag itself?
- Does the cache key include every dimension that affects the response, including region and model version?
- Is there a total deadline on the retry path, and does the caller handle the failure shape explicitly?
- Are prompt and response digests recorded with a stable hash and a partition key that supports deletion?
- Is there a retention policy, and is it enforced by a scheduled job rather than by intention?
- Can you compute cost per successful request from your own metrics, without opening the provider's console?
- Can you disable the LLM path with a flag, and has that flag been tested in production?

## The action to take in the next 30 minutes

Open the file that defines your LLM gateway or the function that wraps your model call, and add a counter that increments once per attempt with a tag for the outcome, plus a counter on the cache hit path. Deploy it to staging, send the traffic you normally send, and read the ratio of attempts to requests. If it is above 1.0, you have retry overhead you did not know about; if the cache hit ratio is below your expectation, the cache key is probably missing a dimension. Both findings are cheap to act on now and expensive to discover from a bill or an audit later.
