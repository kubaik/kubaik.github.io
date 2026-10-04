# Local LLM agents: vLLM vs Ollama for fintech

## Why the serving stack decides the architecture

A common failure mode in regulated fintech is that a team picks a model, writes the agent, and only then discovers that the serving layer cannot handle the concurrency the product needs. The model choice is rarely the problem. The concurrency model is.

Consider the constraint that drives most of these projects. A product team wants an LLM agent that reads transaction memos, drafts suspicious activity report narratives, or summarizes KYC documents. A compliance lead wants to know exactly where those bytes go. Sending customer PII to a hosted API endpoint is a non-starter for many regulated entities under Kenya's Data Protection Act 2019, Nigeria's NDPA 2023, and the GDPR's transfer rules where EU counterparties are involved. Open-weight models in the 7B–70B parameter range are good enough for a large share of back-office fintech tasks, and the tooling to serve them locally is mature. The question is no longer whether local inference is viable. It is which serving stack to standardize on.

Two options dominate architecture reviews: vLLM, a high-throughput inference server built around PagedAttention and continuous batching, and Ollama, a developer-friendly model runner that wraps llama.cpp and GGUF quantization. They are not the same kind of tool. Treating them as interchangeable is the most common mistake in this decision, because their operational profiles diverge sharply once an agent loop with tool calls, retrieval, and a queue sits in front of them.

## What each tool actually is

vLLM is a Python inference server. Its central design decision is PagedAttention, which manages the KV cache like virtual memory pages rather than one contiguous block per sequence. That is why a single GPU can hold many concurrent sequences without exhausting VRAM. It exposes an OpenAI-compatible HTTP API, supports tensor parallelism across GPUs, prefix caching for shared system prompts, and guided decoding for structured output.

Ollama is a Go binary that manages GGUF-quantized models. It runs on macOS with Metal, on Linux with CUDA or ROCm, and on plain CPU. It handles model lifecycle, pulls, and serving behind a simple CLI and an HTTP API on port 11434, with an OpenAI-compatible `/v1` endpoint.

The important asymmetry: vLLM is a serving tier you operate; Ollama is a model runner you install. The difference shows up the moment concurrency, structured output guarantees, or auditability enter the requirements.

## Concurrency: the axis that decides everything

An agent that makes three sequential LLM calls multiplies every per-call delay by three. That is why the metric that matters is not single-stream tokens per second. It is p95 end-to-end latency under the concurrency the product actually generates.

vLLM's continuous batching is the reason it scales. With a 14B model at 4-bit AWQ on a 24 GB A10G, a well-tuned server can hold dozens of concurrent sequences because the KV cache is paged and the scheduler admits new work as sequences finish. Ollama's scheduler is fundamentally different. The `OLLAMA_NUM_PARALLEL` environment variable, which defaults to 1 and is commonly raised to 4, splits a single loaded model into multiple context slots. Beyond that ceiling, requests queue, and queue latency grows faster than throughput.

The practical consequence is a cliff rather than a slope. Below the parallel slot count, both tools feel responsive. Above it, Ollama's p95 latency climbs steeply while vLLM degrades gracefully. For an internal tool with a handful of analysts, the cliff may never be reached. For a customer-facing agent with a five-second SLA, it is reached on the first busy morning.

## Structured output: guided decoding vs grammar constraints

When an agent must emit valid JSON for a downstream ledger or case-management system, malformed output is not a cosmetic problem. It is a failed transaction, and regex repair of model output is a fragile substitute for a guarantee.

vLLM supports guided decoding, including a `guided_json` parameter that constrains generation to a supplied JSON schema. This eliminates an entire class of agent bugs: no more parsing failures when a model decides to add a friendly preamble before its JSON. Ollama supports grammar constraints through the Modelfile, which is useful but more limited in expressiveness and harder to change per request.

If the downstream system cannot tolerate malformed output, this single feature often settles the decision before throughput is even discussed.

```python
# vLLM, OpenAI-compatible client
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-14B-Instruct-AWQ",
    messages=[
        {"role": "system", "content": "You extract structured data from transaction memos. Reply only with JSON."},
        {"role": "user", "content": "Memo: 'TRF/2026-04-11/FX SETTLE USD 48200 REF INV-9931'"},
    ],
    temperature=0.0,
    extra_body={"guided_json": {
        "type": "object",
        "properties": {
            "amount": {"type": "number"},
            "currency": {"type": "string"},
            "reference": {"type": "string"},
        },
        "required": ["amount", "currency", "reference"],
    }},
)
print(resp.choices[0].message.content)
```

The equivalent call against Ollama uses the native API, where `keep_alive` and `num_ctx` are the knobs worth setting explicitly:

```javascript
// Ollama native API
const res = await fetch('http://localhost:11434/api/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model: 'qwen2.5:14b-instruct-q4_K_M',
    messages: [
      { role: 'system', content: 'You are a KYC document summarizer. Cite page numbers.' },
      { role: 'user', content: 'Summarize the attached passport and utility bill.' },
    ],
    stream: false,
    keep_alive: '30m',
    options: { temperature: 0, num_ctx: 8192 },
  }),
});
const data = await res.json();
console.log(data.message.content);
```

## How to measure this yourself instead of trusting a table

Published benchmark tables are close to useless for this decision, because throughput depends on your prompt length, your batch composition, your quantization, and your GPU. The honest approach is to measure on your own hardware with your own prompt shape.

Instrument four things:

1. **Time to first token (TTFT)** at concurrency 1, 4, 10, and 20.
2. **p95 end-to-end latency** at each of those concurrency levels, not the mean. The tail is what breaks SLAs.
3. **Tokens per second per sequence** while the batch is warm.
4. **KV cache utilisation**, if the server exposes it. vLLM publishes Prometheus metrics including `vllm:num_requests_running`, `vllm:gpu_cache_usage_perc`, and a time-to-first-token histogram, which makes this straightforward.

A simple way to generate the load is a load generator pointed at the OpenAI-compatible endpoint. `hey`, `wrk`, or a small async Python script all work. The key is to send a prompt that matches production length, including the system prompt and tool schema, because a 1,500-token prompt behaves very differently from a 20-token one. Run the same load against both servers on the same GPU, and compare the p95 curve rather than a single number.

Two measurement traps are worth naming. First, cold-start latency is a one-time cost per process; do not mix it into steady-state numbers. Second, a warm-up phase matters, because both servers behave differently on the first few requests while caches and CUDA graphs settle.

## Cold start and memory behaviour

vLLM is slower to start than Ollama, typically on the order of tens of seconds, because it compiles CUDA graphs and allocates the KV cache pool up front. That cost is per process, not per request, and it disappears if the server stays warm. Ollama starts faster because it does less up front, but it also degrades faster under load.

vLLM holds VRAM aggressively and allocates against `gpu_memory_utilization`. Misconfiguring that value produces a `torch.cuda.OutOfMemoryError` at startup rather than at request time, which is at least a fast, obvious failure. Ollama's memory behaviour is more forgiving and its CPU fallback means a 7B model can genuinely run on a laptop with no GPU.

A second failure mode worth knowing: CUDA errors of the form `no kernel image is available for execution on the device` almost always mean the installed wheel was compiled for a different compute capability than the GPU. On older cards such as the T4 (compute capability 7.5), wheels built for newer architectures will not run. Check the wheel's target architecture against the GPU before blaming the model or the driver.

## VRAM budget: work it out from first principles

Rather than quoting a table, it is more useful to show the arithmetic, because the answer changes with context length and concurrency.

**Weights.** A 14B parameter model at 4-bit quantization stores roughly 4 bits per parameter, so:

- 14 × 10^9 parameters × 0.5 bytes/parameter ≈ 7 GB of raw weights.
- Quantization metadata, embedding layers, and non-quantized tensors typically add overhead, so budget 9–11 GB in practice for the weights alone.

**KV cache.** The KV cache size depends on layers, KV heads, head dimension, context length, concurrency, and the cache dtype. As an illustrative example, take a 14B model with 48 layers, 8 KV heads, and a head dimension of 128, storing the cache in FP16 (2 bytes per element). Per token, per sequence:

- 2 (key and value) × 48 layers × 8 KV heads × 128 dim × 2 bytes ≈ 196 KB per token.

At an 8,192-token context with 20 concurrent sequences:

- 196 KB × 8,192 × 20 ≈ 32 GB.

That figure exceeds a 24 GB card, which is exactly the point: the weights are the small part. In practice, `max_model_len` and concurrency are tuned together until the KV cache fits, and vLLM's paged allocation lets you raise concurrency without pre-allocating the worst case per sequence. On a 16 GB card the model fits but leaves little headroom for batching, which is why a T4-class GPU is usually an Ollama box rather than a vLLM box.

Treat those numbers as illustrative arithmetic, not measured results. Substitute your model's layer count, KV head count, head dimension, and cache dtype to get your own budget.

## Cost: a worked comparison

Assume AWS on-demand pricing for a `g5.xlarge` (A10G, 24 GB) at roughly $1.00 per hour and a `g4dn.xlarge` (T4, 16 GB) at roughly $0.53 per hour, both in us-east-1. These are illustrative figures; verify current rates before budgeting.

**Low volume.** Fifty requests per day, eight hours of uptime:

- `g4dn.xlarge`: $0.53 × 8 × 30 ≈ $127 per month.
- `g5.xlarge`: $1.00 × 8 × 30 ≈ $240 per month.

At this volume the software is free and you are paying for idle GPU time. Ollama on the cheaper instance is the rational choice, and the throughput ceiling is irrelevant because it is never approached.

**High volume.** Fifty thousand requests per day, 24/7 uptime:

- One `g5.2xlarge` running vLLM may serve the load.
- Four `g4dn.xlarge` instances running Ollama may be needed for equivalent throughput.

The arithmetic: four `g4dn.xlarge` at $0.53 × 24 × 30 ≈ $1,526 per month, versus one `g5.2xlarge` at roughly $2.00 × 24 × 30 ≈ $1,440 per month, with better tail latency from the batched server. The exact crossover depends on your prompt length and model, but the direction is consistent: batching efficiency means fewer GPUs for the same throughput once volume is high enough.

A rough planning rule, offered as a heuristic rather than a measured result: below a few thousand requests per day, Ollama's lower operational overhead usually wins; above roughly ten thousand per day, vLLM's throughput tends to win on cost per request.

## Operational cost that does not appear on the invoice

There is a second cost axis: audit surface. vLLM's configuration is explicit — model path, quantization, `max_model_len`, sampling parameters — which makes it easier to freeze and document for an auditor. Ollama's configuration is spread across a Modelfile and environment variables, which is workable but less obviously frozen.

Neither tool sends data to a vendor by default in a typical deployment, but that claim should be verified rather than assumed. Check egress rules at the network layer regardless of which server you choose, because the compliance argument rests on bytes staying inside your perimeter, not on a default setting.

## Decision checklist

Use Ollama when most of the following hold:

- The workload is a prototype or an internal tool.
- Volume is below a few thousand LLM calls per day.
- The agent is interactive: one analyst, one conversation.
- Developers need to run the same model on laptops for offline work.
- There is no one on the team comfortable with CUDA and containerised GPUs.

Use vLLM when most of the following hold:

- There is a batch pipeline: overnight memo classification or bulk document summarization.
- Downstream systems cannot tolerate malformed output, so guided JSON decoding is required.
- More than roughly ten users or agent workers hit the endpoint concurrently.
- Prometheus metrics are needed for latency and error-rate evidence.
- Prefix caching would meaningfully cut cost, because every request carries the same long tool schema.

## Migration asymmetry

The migration path is not symmetric, and this is the key practical insight.

Moving from Ollama to vLLM is mostly a `base_url` change plus a container. Client code written against the OpenAI-compatible interface does not care which server answers. Moving from vLLM back to Ollama means giving up prefix caching and guided decoding, which usually means rewriting prompts and adding output validation.

The implication for an unsure team is straightforward: prototype on Ollama, but design the client so the endpoint is a configuration value rather than a hardcoded string. That single discipline preserves the option to migrate, and it costs almost nothing to adopt.

## A recommended default, and its honest weakness

For a fintech workflow that must keep data on-premises or in-country, a reasonable default is vLLM serving a 14B 4-bit model on a 24 GB GPU, fronted by a small service that handles authentication, rate limiting, and audit logging. The reasons are concrete: guided JSON removes a class of agent failures, prefix caching reduces the cost of repeated tool schemas, and Prometheus metrics supply the evidence needed when compliance asks how reliability is demonstrated. A 24 GB card fits a 14B 4-bit model with room for a useful context window, which covers most memo-extraction and document-summarization tasks.

The honest weakness of that default is operational burden. vLLM is a heavier dependency, it breaks more often on driver upgrades, and it demands someone comfortable with CUDA and containerised GPUs. If that person does not exist on the team, Ollama is the correct choice even at higher volume, because a working Ollama beats a broken vLLM every time.

## Failure modes to watch

**Ollama behind a queue.** A single worker with `OLLAMA_NUM_PARALLEL=1` will serialise an agent that fans out tool calls. The symptom is a p95 latency that grows linearly with concurrency while throughput stays flat. Fix by raising the parallel slot count, or by moving to a batched server.

**vLLM OOM at startup.** Almost always `gpu_memory_utilization` set too high for the chosen `max_model_len` and concurrency. Lower the memory fraction or the context length and restart.

**CUDA kernel image mismatch.** A wheel built for a newer compute capability than the GPU produces a runtime error at load. Verify architecture compatibility before installing.

**Silent JSON drift.** Without guided decoding, a model may occasionally wrap JSON in prose. If the downstream parser is lenient, the bug hides until a specific memo triggers it. Constrain the output or validate strictly.

**Cold-start spikes.** Both servers are slow on the first request after a restart. Keep the process warm, or accept a health check that fails until the model is loaded.

## Final verdict

vLLM is the better fit for production fintech agent workloads above roughly ten thousand requests per day, for batch pipelines, and wherever structured output must be guaranteed. Ollama is the better fit for prototyping, low-volume internal tools, and laptop-first development. They are less competitors than different stages of the same journey, and teams that get this right treat the serving layer as a swappable component behind an OpenAI-compatible interface. The compliance argument for local inference is settled: open-weight 14B models at 4-bit quantization handle structured extraction and summarization well enough that sending PII abroad is a choice, not a requirement.

## FAQ

**Can Ollama serve concurrent requests for a fintech agent?**
Yes, with a low ceiling. Setting `OLLAMA_NUM_PARALLEL=4` lets one loaded model serve up to four concurrent sequences by splitting its context window. Beyond that, requests queue and p95 latency climbs sharply. For an internal tool with a handful of analysts that is acceptable; for a customer-facing agent with a five-second SLA it is not, and a batched server is the better fit.

**Does local inference settle data protection compliance?**
It removes the cross-border transfer problem, which is the largest hurdle for LLM features in fintech. It does not remove the rest. Access logging, retention limits, and the fact that prompts and outputs may contain personal data landing in your logs all remain. The serving layer is the easy part; the audit trail and retention policy around it is where teams underestimate the work.

**How much VRAM does a 14B model at 4-bit quantization need?**
The weights alone typically require roughly 9–11 GB once quantization metadata and non-quantized tensors are counted. The KV cache is the larger variable, and it scales with context length and concurrency. Working the arithmetic for a specific model — layers, KV heads, head dimension, cache dtype — is the only reliable way to size the GPU.

**Does vLLM support CPU-only inference?**
vLLM has an experimental CPU backend, but its throughput is low and its feature set lags the GPU path. For a regulated workflow without GPU budget, a smaller quantized model on Ollama running on a CPU instance is the more predictable choice for asynchronous batch work, accepting single-digit tokens per second.

**Is a 16 GB GPU enough?**
For a 14B 4-bit model, the weights fit but leave little headroom for batching, which limits concurrency. That configuration is usually better suited to Ollama than to a batched server. A 24 GB card is the more comfortable floor for serving a 14B model with a useful context window and meaningful concurrency.

## Your next 30 minutes

Pick the model you are actually considering and measure it on the hardware you actually have. Start the server, then drive it with a load generator that sends production-length prompts at increasing concurrency:

```
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-14B-Instruct-AWQ --max-model-len 8192
```

Then, from another shell, send a warm-up batch, followed by a measured run at concurrency 1, 4, 10, and 20, and record p95 latency at each level. Repeat against Ollama on the same GPU with the same prompt. The p95 curve under concurrency tells you more about which server belongs in your architecture than any published benchmark, including the arithmetic in this article.
===BODY===
(full markdown article)
