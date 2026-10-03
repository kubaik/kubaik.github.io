# AI-native apps: forget the old stack

## Why the conventional stack is incomplete

The default architecture for an LLM-backed service tends to look the same: a Python API service, an agent framework, a vector store, and a JavaScript frontend. That stack demos well. It also produces a predictable set of production failures: latency well above what the notebook showed, costs that scale faster than traffic, and agent loops that behave badly under concurrent load.

The conventional advice — add a message queue, cache responses, scale horizontally — solves throughput. It does not address the underlying difference. Traditional web services are bound by CPU, memory, or database round-trips, and their retries are cheap. LLM-backed services are bound by tokens: the number of tokens in the request and response determines both latency and cost, and a retry costs the same as the original call. The old stack treats the LLM as just another API. It is not.

A reasonable counter-argument: "Add Redis and a queue, cache and retry, and you scale the same way you always have."

The problem is that caching tokens is not the same as caching HTTP responses. Token consumption depends on conversation state, retrieved context, and prompt templates, so a cache miss can trigger a full-priced call. That call can fail with rate limits or quota errors, and it can also be manipulated by prompt injection. A queue can back up when the provider rate-limits, and an autoscaler reacting to queue depth will spin up pods that enqueue more requests — increasing cost without increasing completed work. The old stack assumes failures are rare and retries are cheap. For LLM calls, retries are expensive and failures cascade.

## A minimal example and what breaks

A typical small service looks like this:

```python
from fastapi import FastAPI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import HuggingFaceEndpoint

app = FastAPI()

prompt = ChatPromptTemplate.from_template("Answer the user question: {question}")
llm = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/meta-llama/Llama-3-70b-instruct",
    huggingfacehub_api_token="hf_YOUR_TOKEN"
)

chain = {"question": RunnablePassthrough()} | prompt | llm

@app.post("/ask")
async def ask(question: str):
    return {"answer": chain.invoke(question)}
```

This is clean and it works in a demo. In production, several distinct problems appear:

1. **Cold start latency.** The first request after a pod restart pays the cost of initializing the endpoint client and any model warm-up. Depending on the runtime and the provider, that can be seconds rather than milliseconds.
2. **Token bloat from unbounded history.** If each call passes the full conversation history, a long chat consumes far more tokens than the current question requires. Token usage grows with conversation length, not with the size of the answer.
3. **Thundering herd.** When the provider rate-limits, requests pile up. An autoscaler that reacts to queue depth adds pods, which add more concurrent requests against the same quota. Cost rises while throughput stays flat.
4. **Prompt injection.** If user input is interpolated into a system prompt, a user can attempt to override instructions. If responses are cached, a poisoned response can be served to later users.

The root cause is usually a template that interpolates user input directly into the system prompt, combined with a cache that stores whatever came back without validation. The LLM is not a stateless API. It is stateful, expensive, and manipulable, and the architecture needs to account for that.

## Failure modes worth designing against

These are recurring patterns rather than one-off incidents. Each one is worth a mitigation.

**Token accounting drift.** Token counts depend on the exact tokenizer and on whitespace and formatting. A message with unusual spacing can tokenize differently than expected. The mitigation is to count tokens client-side with the same tokenizer the model uses, before sending the request, and to enforce a budget.

**Cache key collisions and misses.** Exact-string caching fails on paraphrases, typos, and multilingual input. Every variation becomes a fresh call. Semantic caching — keying on an embedding with a similarity threshold — reduces misses but introduces a new risk: a near-match can return a subtly wrong answer. The threshold is a tunable trade-off between cost and correctness, and it should be evaluated on real traffic.

**Cold starts and node recycling.** If model weights are pulled on demand, a new pod may spend a long time loading before it can serve. Pre-pulling images and keeping a warm replica help, but the underlying constraint is that large models are expensive to start. Serving infrastructure that bundles the model and runtime reduces this cost.

**Prompt injection, including non-English.** Input sanitizers that only match English phrases miss injections in other languages. A defense-in-depth approach combines instruction hierarchy, output validation, and a classifier for injection attempts. No single filter is sufficient.

**Context truncation and hallucination.** If the context window is nearly full, the model may silently drop input and produce a plausible but wrong answer. The mitigation is to compute the token budget explicitly: prompt template + retrieved context + history + reserved output tokens must fit within the model's limit. When it does not, the system should summarize, retrieve less, or refuse — not truncate silently.

## How to measure these problems

None of the numbers above should be taken on faith. Each is measurable with standard tooling.

- **Latency:** instrument the API with request-level timing and record p50, p95, and p99 for the full request and for the LLM call separately. Compare against a local notebook run to isolate the overhead introduced by the service layer.
- **Token usage:** log the token count of every request and response using the model's tokenizer. Aggregate by endpoint and by user. This is the single most useful metric for cost control.
- **Cache effectiveness:** log cache hits, misses, and the similarity score of the nearest match. A rising miss rate on semantically similar queries indicates the threshold is too strict.
- **Error rate and error types:** separate timeouts, rate-limit errors, validation failures, and injection blocks. These have different causes and different fixes.
- **Cost:** compute cost from logged token counts and the provider's published per-token price. Do not estimate from request counts alone.

A useful exercise is to run a fixed set of representative queries against the service and record these metrics before and after each change. That produces a real before/after comparison rather than an anecdote.

## Patterns that contain the problem

### Token budgeting before the call

Count tokens with the model's tokenizer before sending. Enforce a budget that includes the prompt template, retrieved context, conversation history, and reserved output tokens.

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4-turbo")

def check_token_budget(text: str, limit: int = 4000) -> int:
    tokens = len(enc.encode(text))
    if tokens > limit:
        raise ValueError(f"Token budget exceeded: {tokens} > {limit}")
    return tokens
```

The specific tokenizer must match the model. Using a different tokenizer produces counts that are close but not exact.

### Semantic caching with a threshold

Cache responses keyed by an embedding of the question, and return a cached answer only when similarity exceeds a threshold. Redis supports vector similarity search, which makes this practical without a separate vector database.

```python
from redis import Redis
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.indexDefinition import IndexDefinition

r = Redis(host="redis-vector", port=6379, decode_responses=True)
schema = (
    TextField("question"),
    TextField("answer"),
    VectorField(
        "question_embedding",
        "FLAT",
        {"TYPE": "FLOAT32", "DIM": 768, "DISTANCE_METRIC": "COSINE"},
    ),
)
r.ft("idx:qa").create_index(
    schema,
    definition=IndexDefinition(prefix=["qa:"]),
)

def cache_answer(question: str, answer: str, embedding: list[float]) -> None:
    check_token_budget(question)
    r.hset(f"qa:{question[:128]}", mapping={
        "question": question,
        "answer": answer,
        "question_embedding": str(embedding),
    })
    r.ft("idx:qa").add_document(
        f"qa:{question[:128]}",
        vectors={"question_embedding": embedding},
    )
```

Two caveats. First, the cache key truncation (`question[:128]`) can collide for long questions; use a hash instead. Second, semantic caching changes the correctness contract: two different questions may receive the same answer. The threshold should be chosen with that in mind, and high-stakes endpoints should not use semantic caching at all.

### Serving the model close to the request

Bundling the model, tokenizer, and runtime into a single deployable artifact reduces cold-start cost and enables batching and GPU sharing. A serving framework such as BentoML is one option; managed inference endpoints are another. The trade-off is control versus operational burden.

```python
import bentoml
from bentoml.io import JSON
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-70b-instruct",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-70b-instruct")

@bentoml.service(
    name="llama-70b",
    traffic={"timeout": 10.0, "max_concurrent": 50},
    resources={"gpu": 1},
)
class Llama70B:
    @bentoml.api(input=JSON(), output=JSON())
    def ask(self, payload: dict):
        messages = payload["messages"]
        input_ids = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to("cuda")
        outputs = model.generate(
            input_ids,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
        )
        return {"response": tokenizer.decode(outputs[0])}
```

The `max_concurrent` value must be tuned to the GPU's memory and the model's size. Setting it too high causes out-of-memory errors; too low wastes capacity.

### Runtime validation of inputs and outputs

A validation layer between the user and the model can enforce length limits, block known injection patterns, and check that the output conforms to an expected schema. This is the same role a web application firewall plays for HTTP. Managed guardrail services and open-source validation libraries both exist; the important property is that validation happens on every request, including cached ones, and that it covers multiple languages.

```python
from pydantic import BaseModel, Field

class Answer(BaseModel):
    text: str = Field(max_length=1000)
```

The specific validator library is less important than the placement: validation must sit on both the input path and the output path, and it must run before a response is written to the cache.

## A decision checklist

Before shipping an LLM-backed feature, confirm:

- Every request has an explicit token budget, computed with the model's tokenizer.
- Conversation history is bounded, summarized, or dropped by policy.
- Cache keys are semantic where appropriate, and the similarity threshold is documented.
- Cached responses are validated before being stored and before being served.
- Rate-limit errors are handled with backoff, not with autoscaling.
- The serving path has a defined cold-start strategy (warm pool, bundled artifact, or managed endpoint).
- Input validation covers more than one language.
- Cost is computed from logged token counts, not from request counts.
- There is a runbook for quota exhaustion, model deprecation, and provider outage.

## FAQ

**Is a message queue still useful?**
Yes, for smoothing bursts and decoupling ingestion from inference. It does not solve token cost or rate-limit cascades by itself, and it can hide them if queue depth is not monitored alongside provider errors.

**Should semantic caching be used for every endpoint?**
No. It trades correctness for cost. Endpoints where a near-match is acceptable — FAQ-style support, documentation search — are good candidates. Endpoints involving user-specific data or irreversible actions are not.

**How is prompt injection mitigated in practice?**
Layered defenses: instruction hierarchy in the prompt, input classification, output validation, and least-privilege access to any tools the model can call. No single filter is reliable, and filters should be tested against non-English inputs.

**When is a managed inference endpoint preferable to self-hosting?**
When the operational burden of GPU capacity planning, cold starts, and model updates outweighs the cost savings of self-hosting. The decision usually turns on steady-state utilization: high, predictable utilization favors self-hosting; spiky or low utilization favors managed endpoints.

## What to do in the next 30 minutes

Add token logging to your LLM endpoint. For every request, record the model name, the input token count from the model's tokenizer, the output token count, the cache status (hit, miss, or bypass), and the wall-clock latency of the LLM call. Write these to a structured log line. Once a day of traffic has accumulated, aggregate by endpoint and sort by total tokens. The top endpoint is where the next optimization should go.
