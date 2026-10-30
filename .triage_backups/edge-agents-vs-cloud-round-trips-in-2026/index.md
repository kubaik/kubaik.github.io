# Edge agents vs cloud round-trips in 2026

On-device "edge agents" run ML inference on the user's device or a nearby gateway instead of calling a remote API. The appeal is straightforward: no network round-trip, no dependency on connectivity, and no per-request egress. The catch is equally straightforward: device RAM, thermal limits, and OS process management become the bottleneck that cloud APIs used to absorb for you.

This article walks through building a hybrid agent: a local model that handles most requests, plus a cloud fallback that triggers only when the local model is uncertain or the input is too large. The code targets Python 3.11 with ONNX Runtime, but the architecture applies to any runtime.

## When edge inference actually pays off

The decision is not "edge vs cloud." It is "which requests can the device serve without hurting the user." A useful framing:

- **Latency-sensitive, small-model, short-input** requests are strong candidates for edge.
- **Long-context, high-accuracy, or multimodal** requests generally belong in the cloud today.
- **Offline-critical** requests must be edge, even if quality drops.

Two failure modes dominate real deployments:

1. **Memory kill.** The OS reclaims the process when resident memory exceeds the per-app limit. On Android this is enforced per-process; on iOS it is enforced more aggressively when the app is backgrounded. A model that fits at load time can still be killed mid-inference if the KV cache grows.
2. **Silent quality regression.** Quantized models lose accuracy in ways that are invisible in aggregate benchmarks but obvious on specific inputs (long prompts, rare tokens, non-Latin scripts). Without a confidence signal, you ship degraded answers and never notice.

Both are addressable, but only if you instrument for them from day one.

## Prerequisites and what you'll build

- Python 3.11
- ONNX Runtime for inference
- FastAPI for the local HTTP surface
- Redis for short-lived response caching
- A quantized causal LM exported to ONNX
- Android 13+ or iOS 17+ for device testing

You'll end up with an agent that:

- Serves text generation from a local ONNX model
- Caches identical prompts for 30 seconds
- Falls back to a cloud endpoint when a confidence heuristic drops below a threshold or the input exceeds the context window

## Step 1 — environment setup

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate  # Windows
```

```bash
pip install fastapi uvicorn onnxruntime redis transformers optimum sentencepiece httpx prometheus-client
```

Verify what you actually installed before trusting any tutorial (including this one):

```bash
python -c "import onnxruntime, fastapi, redis; print(onnxruntime.__version__, fastapi.__version__, redis.__version__)"
```

On Windows, ONNX Runtime requires the Microsoft Visual C++ Redistributable. Missing it produces DLL load errors that look like corrupt installs.

For Android, Termux from F-Droid is the practical way to run Python on-device. The Play Store build is stale. Inside Termux:

```bash
pkg update && pkg upgrade
pkg install python python-pip rust
pip install fastapi uvicorn onnxruntime redis
```

iOS does not permit arbitrary Python. The realistic path is embedding the model in a Swift app via CoreML (covered later).

## Step 2 — export and load a quantized model

The export step is offline and slow. Run it on a development machine, not the device.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM
import torch

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

ort_model = ORTModelForCausalLM.from_pretrained(
    model_name,
    export=True,
    quantization_config={"load_in_4bit": True},
)

ort_model.save_pretrained("tinyllama-1.1b-4bit")
tokenizer.save_pretrained("tinyllama-1.1b-4bit")
```

Loading on the device with bounded thread counts matters more than most guides admit. Unbounded threads cause thermal throttling on phones within seconds.

```python
import onnxruntime as ort
from pathlib import Path

MODEL_DIR = Path("tinyllama-1.1b-4bit")

sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 2
sess_options.inter_op_num_threads = 2
sess_options.enable_mem_pattern = False  # reduces peak allocation on constrained devices

sess = ort.InferenceSession(
    str(MODEL_DIR / "model.onnx"),
    sess_options,
    providers=["CPUExecutionProvider"],
)
```

## Step 3 — the FastAPI surface with Redis caching

Caching identical prompts is the cheapest latency win available. A 30-second TTL covers retry storms and repeated UI actions without serving stale answers for long.

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis.asyncio as redis
import json, hashlib
import numpy as np

app = FastAPI()
redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)

class PromptRequest(BaseModel):
    text: str
    max_tokens: int = 128
    temperature: float = 0.7

def cache_key_for(req: PromptRequest) -> str:
    digest = hashlib.sha256(req.text.encode()).hexdigest()[:16]
    return f"gen:{digest}:{req.max_tokens}:{req.temperature}"

@app.post("/generate")
async def generate(request: PromptRequest):
    key = cache_key_for(request)
    cached = await redis_client.get(key)
    if cached:
        return json.loads(cached)

    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    outputs = sess.run(
        None,
        {
            "input_ids": inputs["input_ids"].numpy().astype(np.int64),
            "attention_mask": inputs["attention_mask"].numpy().astype(np.int64),
        },
    )

    generated_tokens = outputs[0][0]
    response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    await redis_client.setex(key, 30, json.dumps({"text": response_text}))
    return {"text": response_text}
```

Two details worth calling out:

- `hash()` is not stable across processes in Python because of hash randomization. Use a real digest for cache keys, as above.
- ONNX Runtime expects int64 inputs on most builds but some ARM packages default differently. Casting explicitly avoids shape and dtype errors that surface as cryptic messages.

Run it:

```bash
uvicorn edge_agent:app --host 0.0.0.0 --port 8000
```

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"text":"What is edge AI?", "max_tokens": 64}'
```

## Step 4 — handle the three failure classes

### Memory pressure

The OS will kill the process before it returns an error. Preventative measures:

- Cap thread counts (done above).
- Cap `max_length` on the tokenizer.
- Set a lower `max_tokens` default for mobile.
- Watch RSS explicitly.

On Android, `termux-wake-lock` prevents the process from being frozen when the screen locks:

```bash
pkg install termux-api
termux-wake-lock -s edge_agent
```

### Cold starts

The first inference after boot is slow because weights are paged in and the runtime initializes kernels. Pre-warm at startup:

```python
@app.on_event("startup")
async def startup_event():
    dummy = tokenizer("Hello", return_tensors="pt", truncation=True, max_length=16)
    sess.run(
        None,
        {
            "input_ids": dummy["input_ids"].numpy().astype(np.int64),
            "attention_mask": dummy["attention_mask"].numpy().astype(np.int64),
        },
    )
    print("Model warmed up")
```

### Confidence and fallback

A confidence heuristic does not need to be perfect. It needs to correlate with the cases you care about. Average negative log-likelihood of the generated tokens is a reasonable starting point.

```python
def mean_neg_log_likelihood(logits, token_ids):
    # logits: (seq_len, vocab); token_ids: (seq_len,)
    # Returns average -log p(token_t | token_<t)
    import numpy as np
    log_probs = []
    for t in range(1, len(token_ids)):
        row = logits[t - 1]
        row = row - row.max()  # numerical stability
        log_softmax = row - np.log(np.exp(row).sum())
        log_probs.append(log_softmax[token_ids[t]])
    if not log_probs:
        return float("inf")
    return float(-np.mean(log_probs))
```

Wire the fallback:

```python
import httpx

FALLBACK_URL = "https://your-cloud-endpoint.example.com/generate"
CONFIDENCE_THRESHOLD = 2.5  # tune empirically; higher = more fallbacks

async def call_cloud_fallback(request: PromptRequest):
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.post(
            FALLBACK_URL,
            json={
                "prompt": request.text,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
            },
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail="Cloud fallback failed")
        return resp.json()
```

The threshold value is not universal. It depends on the model, the tokenizer, and the domain. Pick it by running a labeled set of ~100 prompts through the local model, computing the heuristic on each, and choosing the value that separates "acceptable" from "unacceptable" answers with the precision/recall trade-off you want.

### Keeping the process alive on Android

A foreground service is the supported way to survive backgrounding:

```xml
<!-- AndroidManifest.xml -->
<service
    android:name=".EdgeAgentService"
    android:foregroundServiceType="dataSync"
    android:exported="false" />
```

```java
// EdgeAgentService.java (inside onStartCommand)
NotificationManager manager = (NotificationManager) getSystemService(NOTIFICATION_SERVICE);
NotificationChannel channel = new NotificationChannel(
    "agent", "Edge Agent", NotificationManager.IMPORTANCE_LOW);
manager.createNotificationChannel(channel);
Notification notification = new Notification.Builder(this, "agent")
    .setContentTitle("Edge Agent Running")
    .setContentText("Processing requests locally")
    .build();
startForeground(1, notification);
```

Termux does not expose a proper foreground service API to Python processes. The workable pattern is to wrap the Python process with a small native service that owns the notification, or to use a Tasker-style automation that starts the process under a foreground context.

## Step 5 — observability and load testing

Instrument before you optimize. Three counters and one histogram cover most of what you need:

```python
from prometheus_client import Counter, Histogram

REQUEST_LATENCY = Histogram(
    "edge_agent_request_latency_seconds",
    "Latency of edge agent requests",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)
CACHE_HITS = Counter("edge_agent_cache_hits_total", "Total cache hits")
FALLBACK_TRIGGERS = Counter("edge_agent_fallbacks_total", "Total fallback triggers")
```

Expose `/metrics` from the same FastAPI app (the Prometheus client provides an ASGI app you can mount), then scrape it. Alert on two signals: fallback rate above your chosen budget, and p95 latency above your product's tolerance.

A minimal Locust test:

```python
from locust import HttpUser, task, between

class EdgeAgentUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def generate(self):
        self.client.post(
            "/generate",
            json={"text": "Explain edge AI in 50 words",
                  "max_tokens": 64, "temperature": 0.7},
        )
```

```bash
locust -f load_test.py
```

### How to measure honestly

Benchmark numbers from a single device are not portable. What you should record:

- **p50 and p95 latency** per request, per device, per model. Use the histogram above; do not quote a single average.
- **Peak RSS** during inference, not at load. On Android: `adb shell dumpsys meminfo <pid>`. On Linux: `/proc/<pid>/status` `VmHWM`.
- **Fallback rate** as a fraction of total requests, tracked over time. A rising rate is the earliest signal of memory pressure or quality drift.
- **Thermal state.** On Android, `adb shell dumpsys thermalservice`. Sustained throughput on a phone is bounded by thermal throttling, not by peak FLOPS.

Run each measurement long enough for the device to reach steady-state temperature. A 30-second benchmark will look far better than a 30-minute one.

## Quantization trade-offs

Quantization is where most edge projects quietly lose quality. The right way to choose a level is to measure on your own evaluation set, not to trust published perplexity numbers.

A workflow that works:

1. Build a small eval set of 100–300 prompts that represent your real traffic, including edge cases (long inputs, code, non-Latin scripts).
2. Score the FP16 model and each quantized variant on that set with a metric your product cares about (exact match, human rating, task success).
3. Pick the smallest variant whose score is within your tolerance.
4. Re-measure on-device, because runtime kernels differ from training-time numerics.

As a general pattern, 8-bit quantization typically preserves quality better than 4-bit at the cost of roughly double the memory for the weights. Whether that trade is worth it depends entirely on your tolerance and your device floor.

## Common questions

**Can this run on iOS?**
Not with Python. Convert the ONNX graph to CoreML and embed it in a Swift app. `coremltools` handles the conversion, but the resulting model needs to be validated on-device — CoreML's operator coverage and numerical behavior differ from ONNX Runtime.

**Does local inference mean the data is private?**
Local inference means the input never leaves the device for the edge path. If you use a cloud fallback, the request does leave. Whether that satisfies a given compliance regime depends on what data is sent, whether it is logged by the provider, and what consent the user gave. Treat the fallback path as you would any other network call.

**What about speech-to-text?**
The same architecture applies, but the model sizes are larger and the input is continuous. Whisper-family models in the tens-to-hundreds of MB range are the practical range for phones; larger variants hit memory limits. Quantize and test on your target hardware before committing.

**What about slow or unreliable networks?**
The edge path doesn't care. The fallback path does. Use bounded timeouts, exponential backoff, and queue state locally so a failed sync can resume. Compress payloads; keep them small.

## Decision checklist before shipping

- [ ] Peak RSS measured on the lowest-spec target device, not the dev machine
- [ ] p95 latency measured after thermal steady-state, not on first run
- [ ] Fallback rate measured over at least a day of realistic traffic
- [ ] Confidence threshold tuned on a labeled set, not guessed
- [ ] Cache keys stable across processes (real hash, not `hash()`)
- [ ] Foreground service or equivalent keeps the process alive on mobile
- [ ] Cold-start pre-warm runs on app launch
- [ ] Metrics exported and alerted on

## Your next 30 minutes

Pick the lowest-spec device you intend to support, install the agent there, and run this loop for 30 minutes while watching peak RSS and p95 latency:

```bash
while true; do
  curl -s -X POST http://localhost:8000/generate \
    -H "Content-Type: application/json" \
    -d '{"text":"Summarize the last message in one sentence.","max_tokens":64}' \
    -o /dev/null -w "%{time_total}\n"
  adb shell dumpsys meminfo com.termux | grep -i "TOTAL"
  sleep 2
done
```

If peak RSS stays comfortably below the OS limit and p95 latency stays within your product budget after the device warms up, you have a viable edge path. If either degrades over the run, the bottleneck is memory or thermals — reduce the context window, lower `max_tokens`, or move to a smaller quantized variant before adding features.
