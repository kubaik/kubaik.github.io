# LLM costs: fine-tune vs route vs bigger model

The decision between spending more on a larger model, fine-tuning an existing one, or building a router that sends each prompt to the smallest model that can handle it looks straightforward in a design doc. It stops being straightforward once real traffic arrives. This article covers what comes after the happy path: how to measure each option, what breaks, and how to decide.

## The three options and what each actually costs

There are three common ways to improve quality or reduce cost on an LLM workload:

- **Scale up**: call a larger, more capable model for every request.
- **Fine-tune**: train the model you already use on task-specific data.
- **Route**: classify incoming prompts and send each to the smallest model that can handle it.

Each has a different cost structure. Scaling up raises per-token cost linearly with usage. Fine-tuning adds a large fixed cost in engineering and GPU time before it saves anything, and it couples the system to a specific model version. Routing adds a permanent operational component — a classifier, a fallback path, and the monitoring to keep both honest.

A typical failure mode is picking the option that is easiest to reason about rather than the one that matches the workload. Teams often scale up because it requires no new infrastructure, then discover the cost grows faster than revenue. Others fine-tune early because they read that it reduces token cost, then discover the fine-tuned model degrades when a new label or feature is introduced. A third group builds a router with a dozen micro-models and a prompt selector no one can debug.

The bottleneck is rarely raw compute. It is the time required to ship a change and the cost of keeping the system running when traffic doubles.

## Prerequisites and what you will build

The examples below assume a working Python environment and an AWS account with billing alarms configured. The stack is deliberately boring:

- Python 3.12
- FastAPI
- Hugging Face Transformers
- vLLM as the inference engine
- Redis for prompt caching and routing state
- AWS Lambda for the router
- A managed LLM gateway or hosted API for the large-model fallback

The application is a customer-support ticket tagger: it reads JSON, calls a model to classify the ticket (spam, billing, support, feature request), and writes the label to a database. The only thing that changes between approaches is how the model is called and what it costs per 10,000 prompts. Keeping the task fixed makes the comparison clean; the framework applies to any task that fits within the model's context window.

An illustrative setup: Redis on a small managed node, a DynamoDB table for results, and a container image for the inference service. Actual costs depend on region, instance type, and traffic; measure them rather than assuming.

## Step 1 — establish a baseline you can trust

Before comparing approaches, build a baseline that calls one model directly and logs everything needed to compare later.

Create a dataset split and export it in the prompt template your application uses:

```python
from datasets import load_dataset

ds = load_dataset("csv", data_files="tickets.csv")
ds = ds.train_test_split(test_size=0.2, stratify_by_column="label")
ds.save_to_disk("ticket_dataset")
```

Deploy the baseline API. A minimal Terraform module:

```hcl
module "api" {
  source      = "./modules/api"
  model_id    = "mistralai/Mistral-7B-Instruct-v0.3"
  memory_mb   = 8192
  timeout_sec = 30
}
```

Then measure. The baseline is only useful if the numbers are reproducible, so instrument the following:

- **Latency**: record p50 and p95 end-to-end request time, not just model time. Include the router, network hop, and any cache lookup.
- **Cost**: compute cost per 1,000 prompts from actual token counts multiplied by the current per-token price. Do not estimate from request counts.
- **Quality**: hold out a labelled evaluation set and compute accuracy or F1 per class. Track it on every deploy.

A simple way to capture latency and cost is to log token counts and timestamps per request, then aggregate with a script:

```bash
python scripts/aggregate.py --input logs/requests.jsonl --group-by model
```

The output should give you cost per 1,000 prompts and p50/p95 latency per model. Run this for at least a few thousand real or representative prompts before drawing conclusions.

A hard-to-reverse decision at this stage is the model family. If you pick a model whose weights or tokenizer are later removed from public distribution, fine-tuning work must be redone. Prefer model families with active, stable release channels.

## Step 2 — implement the router

The router is a state machine that maps prompt characteristics to model choices. A common implementation is a small Lambda function that inspects the prompt and returns a model identifier.

A minimal router:

```python
import json
from transformers import AutoTokenizer

def load_tokenizer(model_id: str):
    return AutoTokenizer.from_pretrained(model_id)

def route(prompt: str, tokenizer) -> str:
    if "spam" in prompt.lower():
        return "spam_model"
    tokens = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
    if tokens > 200:
        return "fine_tuned_model"
    return "base_model"

def lambda_handler(event, context):
    prompt = json.loads(event["body"])["prompt"]
    tokenizer = load_tokenizer("mistralai/Mistral-7B-Instruct-v0.3")
    model_id = route(prompt, tokenizer)
    return {
        "statusCode": 200,
        "body": json.dumps({"model": model_id})
    }
```

The FastAPI app calls the router first, then proxies to the chosen model:

```python
import os
import httpx

async def classify_ticket(ticket: str):
    router_url = os.getenv("ROUTER_URL")
    async with httpx.AsyncClient(timeout=5.0) as client:
        r = await client.post(router_url, json={"prompt": ticket})
        model_id = r.json()["model"]
        if model_id == "spam_model":
            resp = await client.post(
                "http://spam-model:8000/v1/chat/completions",
                json={"messages": [{"role": "user", "content": ticket}]},
                timeout=3.0
            )
            return resp.json()["choices"][0]["message"]["content"]
        # ... same pattern for the other models
```

The router adds latency. On a cold start, a Lambda-based router can add tens of milliseconds. If the workload is latency-sensitive, run the router in a long-lived container instead. The trade-off is cost: a container runs continuously and bills by the hour, while Lambda bills per invocation. Measure the router's p95 before committing.

## Step 3 — handle edge cases and errors

Three failure modes account for most routed-stack incidents.

**Prompt drift.** A new ticket style appears that the classifier was not trained on. Mitigate by adding a confidence threshold: if the chosen model returns a confidence below a threshold you set from your evaluation data, fall back to the large model. Log the fallback with a `model_used` field so you can review drift later.

```python
async def safe_classify(ticket: str):
    router_url = os.getenv("ROUTER_URL")
    async with httpx.AsyncClient(timeout=5.0) as client:
        r = await client.post(router_url, json={"prompt": ticket})
        model_id = r.json()["model"]
        try:
            if model_id == "spam_model":
                resp = await client.post(
                    "http://spam-model:8000/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": ticket}]},
                    timeout=3.0
                )
                if resp.json()["confidence"] < 0.65:
                    return await large_model_fallback(ticket)
                return resp.json()["label"]
        except Exception:
            return await large_model_fallback(ticket)
```

**Timeout cascade.** One slow model can block a queue. Set a per-model timeout in the inference engine and catch the exception in the router so a single prompt falls back rather than the whole batch.

```hcl
module "fine_tuned" {
  model_id = "mistralai/Mistral-7B-Instruct-v0.3"
  vllm_args = [
    "--max-model-len", "8192",
    "--timeout-seconds", "10",
  ]
}
```

**Cache stampede.** When many requests arrive for the same uncached prompt, they can all recompute it. A distributed lock in Redis prevents this. Set the lock TTL to cover the longest expected computation.

```python
import os
from redis.asyncio import Redis

redis = Redis.from_url(os.getenv("REDIS_URL"))

async def cached_route(prompt: str) -> str:
    cache_key = f"route:{hash(prompt)}"
    async with redis.pipeline() as pipe:
        pipe.watch(cache_key)
        cached = await pipe.get(cache_key)
        if cached:
            return cached
        pipe.multi()
        pipe.set(cache_key, "base_model", ex=3600)
        await pipe.execute()
    return "base_model"
```

The lock adds a small amount of latency on a cache hit and prevents the stampede. It is harder to change later if you move to a multi-region cache, so plan the key scheme before scaling beyond a single region.

## Step 4 — observability and tests

A routed stack cannot be debugged without three things:

- **Metrics** scraped from the API and inference engine.
- **Traces** that separate router latency from model latency.
- **Synthetic checks** that replay a fixed prompt set on a schedule and alert when p95 latency or error rate crosses a threshold.

Expose Prometheus metrics from FastAPI:

```python
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from fastapi import FastAPI, Response

REQUEST_COUNT = Counter("api_requests_total", "Total API requests", ["model"])

app = FastAPI()

@app.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
```

Build a dashboard that shows p95 latency and cost per 1,000 prompts split by model. A common failure mode is that the spam classifier lags when traffic shifts to late-night spam waves; the dashboard shows this before users complain.

For tests, cover the router logic, timeout and fallback behavior, and cache hit/miss scenarios. Mock the model calls so tests run without GPU or network access:

```python
import pytest

@pytest.mark.asyncio
async def test_spam_route():
    prompt = "win a free iphone click here"
    model = await route(prompt, tokenizer)
    assert model == "spam_model"
```

Run the suite on every push. If a test fails, have the pipeline post the exact error so no one has to reproduce it manually.

A hard-to-reverse decision here is the metric namespace. Use OpenTelemetry semantic conventions so dashboards remain portable when you migrate to a centralized telemetry stack.

## How to compare the approaches

Rather than trusting a benchmark table, measure the three approaches on your own workload. The table below shows what to instrument and what each approach tends to change.

| Approach | What it changes | What to measure | Reversibility |
|---|---|---|---|
| Scale up | Per-token cost rises with usage | Cost per 1,000 prompts, accuracy | Easy |
| Fine-tune | Large fixed cost, model version lock-in | GPU hours, accuracy on held-out set, drift over time | Hard |
| Route | Adds classifier, fallback, monitoring | Router p95, fallback rate, blended cost | Medium |

To compare, run the same evaluation set through each approach and record:

- **Cost per correct label**, not cost per prompt. A cheaper model that is wrong more often is not cheaper.
- **p95 latency** end-to-end, including the router.
- **Human review rate**: the fraction of outputs that need a person to check them.

The decision rule that follows from these measurements: keep routing while cost per correct label stays below your target and the human review rate stays low. Fine-tune only when the review rate stays elevated across multiple evaluation runs and fine-tuning demonstrably reduces errors on a held-out set. Scale up when the task is genuinely beyond the smaller models and the volume is low enough that per-token cost does not dominate.

## Worked example: reasoning through a decision

Suppose a support ticket classifier handles 100,000 tickets per month. A large model costs $0.03 per 1,000 input tokens and $0.15 per 1,000 output tokens. At an average of 150 input and 20 output tokens per ticket, each ticket costs:

- Input: 150 / 1,000 × $0.03 = $0.0045
- Output: 20 / 1,000 × $0.15 = $0.0030
- Total: $0.0075 per ticket

At 100,000 tickets per month, that is $750 per month on the large model alone.

A small self-hosted model on a single GPU instance might cost a fixed amount per month regardless of traffic, plus the engineering time to operate it. If the instance costs $400 per month and handles the volume, the saving is $350 per month — but only if accuracy is acceptable. If accuracy drops and 5% of tickets need human review at a cost of $2 per review, that adds 5,000 × $2 = $10,000 per month, which dwarfs the saving.

This is why cost per correct label matters. The large model at $0.0075 per ticket with 99% accuracy costs about $0.0076 per correct label. The small model at effectively $0.004 per ticket with 90% accuracy, plus review cost, may cost far more per correct label.

Routing sits between them: route the easy majority to the small model and the hard minority to the large model. If 80% of tickets are easy and handled correctly by the small model, and 20% go to the large model, the blended cost is:

- Small model: 80,000 × $0.004 = $320
- Large model: 20,000 × $0.0075 = $150
- Total: $470 per month

That is a 37% reduction versus the large model alone, before accounting for router overhead. Whether it is worth it depends on the router's own cost and the fallback rate. If the fallback rate is higher than assumed, the saving shrinks.

These figures are illustrative. Substitute your own token counts and prices.

## Common questions

**When should fine-tuning be preferred over routing?**
When routing cannot reach the required accuracy because no available model handles the task well, and when the task is stable enough that retraining is infrequent. Fine-tuning is a poor first step because it locks the system to a model version and adds a large fixed cost before any saving appears.

**Does routing work with function calling?**
It can, but the function schema must be supported by every model in the route table. Pin function-calling prompts to models that support tools, and route plain text to smaller models. Verify support per model rather than assuming it.

**Is a cache necessary?**
A cache helps when prompts repeat. In-memory caches in serverless functions are ephemeral and reset on cold start, so they only help within a single invocation. A shared cache persists across instances and deploys. Measure the hit rate before adding one; below a few thousand hits per month the operational cost may not be justified.

**Should I just use the largest model from the start?**
Only if the task is genuinely beyond smaller models or the volume is low. For high-volume classification, the per-token cost difference between a small self-hosted model and a large hosted model is large enough that routing or fine-tuning usually pays off — provided accuracy holds.

## One action to take in the next 30 minutes

Instrument cost per correct label for your current model. Add a log line that records the model name, input tokens, output tokens, and whether the output matched a known-correct label on a small evaluation set. Run it for a few hundred requests, then compute the cost per correct label. That single number is the baseline every routing, fine-tuning, or scaling decision should be measured against.
