# AI features rot faster than you build them

## The problem is not the stack, it is the implicit contract

Most teams are told to ship AI features like any other feature: wrap a model call behind an HTTP endpoint, add caching, add a rate limiter, put the rollout behind a feature flag, and ship. On paper the path is clean. A typical stack might be an orchestration library in Python, a FastAPI app, Redis for hot model responses, and a sidecar for rate limiting. The runbook usually ends with a line about "monitoring token usage" and calls the feature production-ready.

The stack choice is rarely the problem. The problem is the hidden lifecycle. Model drift, API deprecations, and payment rail changes hit AI features harder than conventional ones because the surface area is larger. Prompts, token budgets, embeddings, vector store schemas, and downstream API contracts all age at different speeds. A prompt change can violate a validation rule in a payment provider's callback schema. A model version bump can change your embedding dimension overnight, turning a 500 MB cache into a multi-gigabyte problem that evicts everything else.

The reason these failures are hard to diagnose is that they do not present as infrastructure failures. They present as "our cache hit rate dropped" or "our LLM costs exploded." The contract between the caller and the model is implicit, so nothing in the code says what the caller actually requires.

The conventional advice — treat an AI feature like any other API — is incomplete for one reason: the model is a moving target that you do not control. A typical failure mode is hardcoding prompt templates or expected output shapes into validation logic. When the model's output shape shifts, validation starts rejecting valid responses, and the error surfaces as a generic schema error rather than as a model change.

## What actually happens when you follow the standard advice

Consider a plausible scenario: a logistics company ships an AI routing feature. The pipeline calls a hosted model through an aggregator API, Redis caches "last-mile" routing decisions, and a FastAPI backend serves requests. The runbook says: cache hot responses, limit concurrency, monitor token usage.

Within days of launch, the cache hit rate collapses. The cause is not a bug in the cache. The upstream API started returning a duration field in a different unit than before. Because the cache key is derived from the response payload, every cached entry is now unreachable, and the cache churns. Memory usage climbs, the cold path hits the model on nearly every request, and p99 latency rises sharply because the previously cached fast path is gone.

The standard advice also misses that model APIs change behavior without warning. Parameters get deprecated, defaults change, and token limits move. Teams that did not pin a model version see the failure cascade across environments simultaneously, because every environment resolves "latest" to the same new version. The fix requires a code change, a rebuild, and a redeploy.

Payment rails add another layer. A callback schema that gains a new required field will cause a strict request model to reject otherwise valid callbacks. The error looks like a validation failure in your service, but the root cause is upstream: an external contract changed and your code assumed it would not.

None of this is edge-case noise. It is the normal behavior of systems that depend on external, versioned-by-someone-else services. The teams that recover fastest share one property: they treat the AI feature as a distributed system with an explicit, versioned contract rather than as a thin wrapper around a model call.

## A different mental model: versioned contracts

Shipping an AI feature is not about wrapping a model call behind an endpoint. It is about shipping a contract that can evolve without breaking downstream systems. The path you ship is less important than the path you can change without a fire drill.

Think of the AI feature as a state machine whose states are versions of the prompt, the model, the response schema, and any external callback schema. Each state must be versionable, testable, and reversible. The contract between the AI system and the rest of the stack must be explicit and enforced at the boundary, not assumed in the middle.

A concrete example: a symptom-checker service defines its request contract as a JSON Schema document with a version field:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {
    "version": { "const": "1.2.0" },
    "symptoms": { "type": "array", "items": { "type": "string" } },
    "language": { "type": "string", "enum": ["en", "sw", "fr"] }
  },
  "required": ["version", "symptoms"]
}
```

The API accepts only requests declaring a known version. The backend uses that version to select the prompt template, the model version, and the response schema. When the prompt needs to change, the version is incremented, the new template is deployed alongside the old one, and traffic shifts gradually. The old version stays active for rollback. No downstream system breaks, because the contract was explicit and versioned.

This turns the AI feature into a set of versioned contracts rather than a single fragile endpoint. The tooling required is modest: a schema registry (even a directory in a repository works), a versioned prompt store, and a way to shift traffic between versions. In practice this can be done with a reverse proxy, a service mesh, or a small weighted router in your own application. A managed LLM gateway can also absorb some of this if you already run one.

The objection is that versioning adds complexity. It does. But the complexity does not disappear if you skip it; it moves to production, where every outage is a surprise and every fix is a scramble. The question is not whether to add complexity, but where to put it.

## Three approaches and what each one costs

### The naive wrapper

A pricing service exposes a single endpoint backed by an orchestration pipeline and a Redis cache keyed on user and product identifiers. The prompt template is hardcoded. When the upstream API changes token limits, the pipeline truncates responses silently. Users see prices but no explanations. The cache hit rate falls because new fields appear in responses and change the derived cache keys. Debugging takes days because the symptom (cache churn) is far from the cause (upstream output change).

The cost is not only engineering time. Once users notice degraded output, trust in the feature drops, and recovery requires a redeploy, a cache flush, and a gradual rollout behind flags. The blast radius is the whole feature.

### The versioned contract

A triage service defines its API contract in JSON Schema, pins the model version in request headers, and stores prompts in a versioned file store. The backend injects the correct prompt template and model version based on the version header.

When the prompt needs to change, the version is incremented, the new prompt is deployed alongside the old one, and a small percentage of traffic is routed to the new version. After validation, traffic shifts gradually. The old version remains active for a rollback window. Cache keys include the version, so invalidation is explicit and predictable.

The result is that model drift does not cause outages, because drift is detected at the boundary rather than discovered by users. Engineering time spent on AI maintenance falls because failures are localized and reversible.

### The hybrid approach

A hybrid wraps the AI feature in a REST endpoint but adds a validation step that checks responses against a versioned schema before caching. The schema enforces token limits, required fields, and expected response structures. When the upstream API changes behavior, validation rejects invalid responses immediately, preventing cache corruption.

The tradeoff is added latency and operational overhead: the validator must be updated whenever the schema changes, and it sits in the hot path. Running the validator as a separate process with an efficient internal protocol can reduce the latency cost, but it does not eliminate it.

### Comparing the three

| Approach | Failure detection | Recovery cost | Ongoing overhead | Blast radius |
|----------|-------------------|---------------|------------------|--------------|
| Naive wrapper | Users notice degraded output | Redeploy, cache flush, rollout | Low until it fails | Whole feature |
| Versioned contract | Boundary validation rejects mismatches | Shift traffic back to prior version | Moderate: schema, prompt store, router | One version |
| Hybrid | Validator rejects bad responses before cache | Update validator, redeploy | Moderate to high: validator in hot path | Validator and cache |

The pattern is that detection moves earlier and blast radius shrinks as you add explicit contracts. None of these numbers are universal; the shape of the tradeoff is what matters.

## How to measure whether you have this problem

Rather than trusting a benchmark, instrument your own system. The following measurements are cheap and will tell you whether your AI feature is fragile.

- **Cache hit rate over time.** Export a counter for cache hits and misses, and compute the ratio on a dashboard. A sudden drop that is not explained by traffic mix usually means the cache key changed because the response shape changed.
- **Schema validation failure rate.** Increment a counter every time a response fails validation. This is your earliest warning that the model's behavior changed.
- **Model version in use.** Emit the resolved model version as a label on every request metric. If it changes without a deploy, you have an unpinned dependency.
- **Token usage per request, as a distribution.** Track p50 and p99, not just the mean. Truncation and verbosity changes show up in the tail first.
- **p99 latency split by cache hit and miss.** If the miss path is dominating, the cache is no longer doing its job.
- **Cost per successful request.** Divide total model spend by successful requests. A rising ratio means you are paying for failures.

A useful exercise is to run a script that fetches the current schema or model metadata from your provider and diffs it against the version you have pinned. If the diff is non-empty and you have not deployed, your contract is implicit. This is the single highest-value check you can automate.

## When the conventional approach is fine

Not every AI feature needs a versioned contract. If the feature is low-risk, short-lived, or internal, a single endpoint with a hardcoded prompt is acceptable. An internal onboarding bot with a handful of users, no external dependencies, and no downstream consumers has a small blast radius and a low cost of failure.

A second case: a feature that uses a pinned model version with a fixed prompt and deterministic outputs. A tool that generates structured exports from structured data using a fixed prompt and a pinned model can be shipped as a simple API, because the contract surface is small and stable.

The litmus test is risk. Ask what happens if this feature breaks. If the answer is "a few people wait a bit," the conventional approach is fine. If the answer is "payments fail," "users lose data," or "we breach a compliance obligation," versioning is not optional.

Teams in regulated industries — healthcare, finance, anything touching payment rails — should assume high risk by default. A single outage in a fraud-detection system can trigger compliance findings, lost transactions, and reputational damage.

## A decision checklist

Work through these questions in order. The first "yes" tells you where to start.

1. Does the feature touch payment rails, user data, or a compliance obligation? If yes, version the contract from day one.
2. Do downstream systems parse the AI output? If yes, version the response schema and validate at the boundary.
3. Will the prompt, model, or schema change more than once a quarter? If yes, versioning is mandatory.
4. Can you roll back a model version change without a redeploy? If no, add version routing before your next model change.
5. Do you know which model version served a given request last week? If no, add version labels to your metrics.
6. Is the feature internal and low-traffic with no downstream consumers? If yes, start simple and revisit when usage grows.

## Worked example: adding versioning to an existing endpoint

Suppose you have a FastAPI endpoint that calls a model and caches the result. The current code looks roughly like this:

```python
import hashlib
import json
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class RouteRequest(BaseModel):
    origin: str
    destination: str

def cache_key(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

@app.post("/route")
def route(req: RouteRequest):
    payload = {"origin": req.origin, "destination": req.destination}
    key = cache_key(payload)
    cached = redis.get(key)
    if cached:
        return json.loads(cached)
    result = call_model(payload)
    redis.set(key, json.dumps(result), ex=3600)
    return result
```

The failure modes are visible in the code. The cache key is derived from the request only, so a response shape change does not invalidate it — you serve stale-shaped data until the TTL expires. The model version is not recorded anywhere. There is no validation of the model's output before caching.

A versioned version of the same endpoint:

```python
import hashlib
import json
import os
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, ValidationError

app = FastAPI()

MODEL_VERSION = os.environ["MODEL_VERSION"]  # pinned, e.g. "2026-01-15"
SUPPORTED_VERSIONS = {"1.2.0", "1.3.0"}

class RouteRequest(BaseModel):
    version: str
    origin: str
    destination: str

class RouteResponse(BaseModel):
    duration_minutes: int
    distance_km: float
    explanation: str

def cache_key(version: str, payload: dict) -> str:
    material = {"version": version, "model": MODEL_VERSION, **payload}
    return hashlib.sha256(
        json.dumps(material, sort_keys=True).encode()
    ).hexdigest()

@app.post("/route", response_model=RouteResponse)
def route(req: RouteRequest, x_request_id: str = Header(...)):
    if req.version not in SUPPORTED_VERSIONS:
        raise HTTPException(status_code=400, detail="unsupported version")
    payload = {"origin": req.origin, "destination": req.destination}
    key = cache_key(req.version, payload)
    cached = redis.get(key)
    if cached:
        return json.loads(cached)
    raw = call_model(payload, version=req.version, model=MODEL_VERSION)
    try:
        validated = RouteResponse(**raw)
    except ValidationError as exc:
        schema_validation_failures.inc()
        raise HTTPException(status_code=502, detail="invalid model output") from exc
    redis.set(key, validated.model_dump_json(), ex=3600)
    return validated
```

The differences matter:

- The cache key includes the contract version and the pinned model version, so a model change or a schema change produces new keys instead of serving stale data.
- The model output is validated before it is cached, so a bad response cannot poison the cache.
- Validation failures are counted, which gives you the early-warning metric described above.
- Unsupported versions are rejected at the boundary, so you can retire old versions deliberately rather than accidentally.

The cost is a few dozen lines and one new environment variable. The benefit is that the next model change becomes a routing decision rather than an outage.

## Objections and responses

**"Versioning adds too much complexity."**
The complexity does not disappear if you skip it; it moves from design time to on-call time. A versioned contract is more work upfront and far less work during an incident. The question is where you want to pay.

**"Our model is stable; we do not need versions."**
Model stability is an assumption, not a guarantee. Providers deprecate parameters, change defaults, and adjust limits. Pinning a version and validating output turns a silent change into a visible one. The cost of a version identifier is trivial compared to the cost of an outage.

**"Our team is small; we cannot afford this."**
Versioning does not require a platform team. A JSON Schema file, a version header, a pinned model identifier, and a validation call are within reach of a small team. The overhead is a few hundred lines and a small increase in deployment complexity. The alternative is spending days debugging cache invalidation or schema mismatches.

**"Our users do not care about versions."**
They do not, and they should not have to. A versioned contract is an implementation detail whose user-visible benefit is reliability. The goal is for the AI feature to be unremarkable: it works, and nobody thinks about it.

## What to do first

If you are starting a new AI feature, define the request contract in JSON Schema from day one and pin the version in a header. If you already have a feature in production, add the version to the cache key and validate the model output before caching. Both are small changes with a large effect on blast radius.

## Next 30 minutes

Open the codebase for your highest-traffic AI feature and find where the model is called. Add a single line that records the pinned model version and includes it in the cache key:

```python
MODEL_VERSION = os.environ["MODEL_VERSION"]  # pin this; do not default to "latest"
```

Then add `MODEL_VERSION` to your deployment configuration with an explicit value, and change the cache key to include it. Deploy to staging and confirm that a change to `MODEL_VERSION` produces a different cache key. That one change turns a silent model upgrade into a visible, reversible event.
