# LLM agents: why monoliths win

Most LLM agent systems start as a single process and work fine. The trouble usually begins when a team reads that "production-grade" agents need separate services for prompting, tool calls, post-processing and safety checks, and rebuilds a working pipeline into a distributed one. This article covers what that migration actually costs, why the costs are hard to see in advance, and how to decide with measurements rather than architecture fashion.

## The conventional wisdom, and what it leaves out

The standard advice in agent tutorials tends to read like a checklist:

1. Decompose the task into atomic LLM calls.
2. Wrap each call in its own service, often a FastAPI container.
3. Communicate over a queue or message bus.
4. Persist intermediate results in a cache or database.
5. Scale each piece independently on a container platform.

The reasoning is not silly. Each component can be versioned, monitored and replaced in isolation. In a regulated environment, a dedicated audit service that writes immutable records to encrypted object storage looks like the safe choice.

What the checklist omits is that an LLM call is not a pure function. Modern models carry state across a request in ways that matter to your architecture:

- **Prompt caching and system-prompt persistence.** Providers cache prefixes server-side, so the cost and latency of a call depend on what was sent before, not just on the current request. Splitting a logically single request across processes can invalidate or fragment that cache.
- **Per-request identity.** Audit trails, rate limits, retries and tracing all key off a request ID. Every network hop is a place where that ID can be dropped, regenerated or mismatched.
- **Tail latency compounding.** If each hop has a p99, the end-to-end p99 is not the sum of the averages — it is closer to the sum of the tails, plus queueing.

None of these are visible in a whiteboard diagram. All of them are measurable.

## A worked failure mode

Consider a ticket-routing bot: a router receives a message, a classifier assigns a category, a handler produces a reply. Suppose each stage is a separate service, and each writes its own audit record.

Walk through a partial failure:

1. The router signs an audit record and forwards the request with a request ID.
2. The classifier succeeds but the downstream cache write fails, for example because the cache has no eviction policy configured and has hit its memory ceiling. The error surfaces as something like `OOM command not allowed when used memory > 'maxmemory'`.
3. The classifier returns an error to the router. The router marks the request failed.
4. The handler, already in flight, completes and writes its own audit record — but the request ID it carries was issued by the router and is now associated with a failed request.

The result is an audit trail with two records for one logical request, one of which is marked failed, and no record of the classifier's partial work. In a GDPR Article 30 context, that is a records-integrity problem, not a bug. The remediation is manual: reconcile the logs, determine what data was processed, and document it.

The same failure in a single process is a single exception. Either the whole request fails and one audit record is written, or it succeeds and one audit record is written. There is no window in which two processes disagree about what happened.

This is the core argument for monolith-first: **the unit of audit and the unit of retry should be the same unit as the unit of request.** Distribution breaks that alignment, and you then have to rebuild it explicitly with idempotency keys, outbox patterns and reconciliation jobs.

## What to measure before you decompose

The claim "micro-services are slower here" is only useful if you can check it on your own workload. Instrument the following.

**End-to-end latency, by percentile.** Use a load generator that reports p50, p95 and p99, not just averages. Averages hide the tail that your SLA is actually written against. Run the same prompt mix against a single-process prototype and against the distributed version.

**Per-hop latency.** Time each network call, including DNS, TLS handshake and serialization. A hop that "should" take 2 ms often costs 10–15 ms once you include connection setup, and connection setup repeats whenever a container is cold.

**Audit completeness.** Define a test that asserts exactly one audit record per logical request, keyed by request ID. Run it under injected failures: kill a downstream service mid-request, throttle the cache, expire credentials. Count the requests where the assertion fails. This number is the one that matters for compliance, and it is almost always higher in a distributed design.

**Cost per request.** Break the bill into model tokens, compute time, and inter-service traffic. Inter-service traffic is the line item people forget: every hop is bytes in and bytes out, and in a container platform you may also pay for the load balancer and NAT gateway.

**Cold-start behavior.** Measure the first request after a deploy or scale-out event separately from steady state. A monolith has one cold start; a pipeline has as many cold starts as it has services, and they can compound if each service waits on the next.

A useful rule is to write down the numbers before and after any split. If the split does not improve the metric you care about — usually p99 latency or cost per request — by more than the added operational burden is worth, do not keep it.

## A monolithic-first design

The alternative mental model is to treat the LLM call as the core of the system and keep everything that depends on request identity in the same process.

**Single entry point.** One application handles the full request-response cycle. All model calls happen in-process. There are no serialization boundaries inside a request.

**In-process caching for stable inputs.** System prompts and other stable strings can be memoized in the process. This avoids re-sending identical prefixes and avoids a network round trip to a cache service for data that never changes during the process lifetime.

**Unified audit.** Build the audit record once, in the request handler, and write it once. Sign it with a key you control, and include the request ID in every downstream payload so that any later processing can be correlated.

**Feature flags for optional stages.** Keep optional components — an external classifier, a retrieval step, a safety filter — behind configuration flags. The monolith calls them in-process by default; the flag exists so you can move a stage out later without rewriting the request path.

A minimal version of this in Python:

```python
import base64
import hashlib
import hmac
import json
import os
import time
from functools import lru_cache

import boto3
from openai import OpenAI

s3 = boto3.client("s3")
client = OpenAI()

@lru_cache(maxsize=256)
def get_system_prompt() -> str:
    return "You are a helpful banking assistant. Be concise and accurate."

def sign_audit(request_id: str, payload: dict) -> str:
    # Illustrative HMAC. Use a managed KMS key for real signing.
    secret = os.environ["AUDIT_SECRET"].encode()
    msg = json.dumps({"id": request_id, "payload": payload}, sort_keys=True).encode()
    return base64.b64encode(
        hmac.new(secret, msg, hashlib.sha256).digest()
    ).decode()

def handler(event, context):
    start = time.time()
    body = json.loads(event["body"]) if isinstance(event.get("body"), str) else event["body"]
    request_id = event.get("request_id") or body.get("request_id") or str(int(time.time() * 1000))
    user_msg = body["message"]

    response = client.chat.completions.create(
        model=os.environ["MODEL_ID"],
        messages=[
            {"role": "system", "content": get_system_prompt()},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )
    answer = response.choices[0].message.content

    audit_record = {
        "request_id": request_id,
        "timestamp": int(time.time()),
        "answer": answer,
    }
    audit_record["signature"] = sign_audit(request_id, audit_record)

    s3.put_object(
        Bucket=os.environ["AUDIT_BUCKET"],
        Key=f"{request_id}.json",
        Body=json.dumps(audit_record).encode(),
        ServerSideEncryption="aws:kms",
    )

    return {
        "statusCode": 200,
        "body": json.dumps({
            "answer": answer,
            "latency_ms": int((time.time() - start) * 1000),
        }),
    }
```

Two details matter more than the rest. First, the audit write happens after the model call and before the response is returned, so a request that produced output always has a record. Second, the request ID is generated or accepted at the boundary and never regenerated downstream.

If you later need an external component — say a dedicated classifier — keep it behind a flag and pass the request ID through explicitly:

```javascript
import { S3Client, PutObjectCommand } from "@aws-sdk/client-s3";

const s3 = new S3Client({ region: process.env.AWS_REGION });

export async function classify(message, requestId) {
  const resp = await fetch(process.env.CLASSIFIER_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: message, request_id: requestId }),
  });

  if (!resp.ok) {
    throw new Error(`Classifier failed: ${resp.status}`);
  }

  const { category } = await resp.json();

  await s3.send(new PutObjectCommand({
    Bucket: process.env.AUDIT_BUCKET,
    Key: `${requestId}.classifier.json`,
    Body: JSON.stringify({ category, request_id: requestId, ts: Date.now() }),
  }));

  return category;
}
```

The caller passes the same `requestId` it used for its own audit record. If the flag is off, the monolith never makes the call and never writes the second record. If the flag is on, the two records share a key and can be joined.

## Where the distributed design does earn its keep

Monolith-first is a default, not a religion. There are cases where splitting is the right call:

| Situation | Why splitting helps | What it costs |
|---|---|---|
| Multiple independent model providers with different failure modes | A provider outage affects one stage, not the whole path | Extra hop latency; per-stage audit records to reconcile |
| Hard per-tenant compute isolation required by contract or regulation | Container boundaries make isolation auditable | More infrastructure; higher fixed cost per tenant |
| Sustained load beyond a single runtime's limits | Independent scaling of the bottleneck stage | Capacity planning for each stage; more cold starts |
| Long-running background work alongside interactive requests | Interactive path stays fast while batch work runs elsewhere | Two code paths to keep consistent |

In each of these, the split is justified by a constraint, not by an aesthetic preference for small services. If you cannot name the constraint, the split is probably premature.

## A decision checklist

Work through these in order. Stop at the first one that applies.

1. **Is there a regulatory or contractual requirement for compute isolation?** If yes, split. If no, continue.
2. **Does the single-process prototype meet your p99 latency target?** Measure it. If yes, do not split.
3. **Is the bottleneck a specific stage that scales differently from the rest?** If yes, split that stage only, and keep everything else in-process.
4. **Can you keep one audit record per logical request across the split?** If you cannot, fix that before splitting, not after.
5. **Does the split reduce cost per request after including inter-service traffic and load balancer costs?** If not, do not split.

The order matters. Teams often start with step 3 because it is the most technically interesting, and discover steps 1, 2 and 4 later, when the audit trail is already fragmented.

## Common objections

**"We need to reuse the classifier across products."** Reuse is usually better served by a shared library than a shared service. A library keeps the call in-process, preserves request identity, and removes a network hop. A shared service adds a deployment dependency and a failure domain for every consumer.

**"We need to scale beyond what a single function can handle."** Serverless platforms document concurrency limits and allow limit increases; container platforms scale further but with more operational surface. The relevant question is not whether a single runtime can handle your peak, but whether the bottleneck stage is the one you are splitting. If the model call is the bottleneck, splitting the surrounding code does not help.

**"Audit logs must live in a separate system."** Storage and execution context are separate concerns. You can write to a dedicated bucket, a dedicated table, or a dedicated account from inside the monolith. What you should not do is split the *write* across processes, because that is what breaks the one-record-per-request invariant.

**"Security policy forbids running model calls alongside business logic."** This is a real constraint in some environments, and it is usually satisfiable without a network hop — for example, by isolating the model client in a separate module or layer within the same runtime, or by running the model call in a sidecar that shares the request context. Confirm what the policy actually requires before assuming it requires a separate service.

## What to do in the next 30 minutes

Open your agent's request handler and find every place where the request ID is generated, read or forwarded. If it is generated more than once, or if any downstream call does not carry the ID it received, fix that first — before any other architecture change. Then add a single test that asserts exactly one audit record exists per request ID after a successful request, and run it. That test is the cheapest guard you can put on the invariant that distributed designs tend to break.
===END===
