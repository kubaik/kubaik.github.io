# Model router: keep it simple, keep it local

Most tutorials describe model routing as a diagram: a gateway, a few arrows, some YAML annotations, and a claim that the problem is solved. The distance between that diagram and a working system is usually measured in weeks of debugging, not minutes. The recurring failure pattern is not that routing is hard — it is that the first design treats models as stateless functions, when in production they are stateful resources with cold starts, regional behavior, quotas, and costs that shift over time.

This article describes a routing design that stays deliberately small: a rule-based decision engine in front of local and cloud endpoints, with complexity pushed into the endpoints and the infrastructure around them. It also covers the failure modes that show up once such a router meets real traffic.

## The gap between the diagram and production

A typical first implementation accepts a `model_id` parameter and forwards the request either to a local inference server or to a managed cloud endpoint. That works in a demo. In production, several things break at once:

- **Cold starts.** Managed inference endpoints commonly scale to zero or release capacity when idle. The first request after an idle period pays a startup cost that can range from hundreds of milliseconds to tens of seconds depending on instance size and model size. The exact figure is workload-specific; the point is that it is not zero and it is not visible in a diagram.
- **Process and context reuse.** A naive local server that spawns a new process per request — or reloads model weights per request — pays a fixed cost on every call. Reusing a long-lived process and its accelerator context is usually the single largest local optimization available.
- **Regional variance.** Latency to a managed endpoint depends on the network path between caller and region, not only on the endpoint's own compute. Two regions running identical hardware can differ substantially for a given user population.
- **Quotas.** Managed services enforce per-region invocation quotas. When a quota is exhausted, requests fail with a throttling error that often does not name the quota as the cause, which sends debugging in the wrong direction.
- **Observability.** Without per-model, per-endpoint metrics, a routing problem is indistinguishable from a model problem.

The design principle that follows from this list: the router should make a routing decision and nothing else. Retries, connection reuse, health verification, pre-warming, and quota awareness belong to the layers around it, not inside the decision path.

## The core design: a two-tier router

The router is a small HTTP service that evaluates ordered rules and returns a target endpoint. It does not pool connections to models, does not implement weighted balancing, and does not hold model state.

**Tier 1 — the decision engine.** A lightweight server reads a JSON config describing, per model, a local endpoint, a cloud endpoint, and a fallback endpoint. Selection order is fixed: local if healthy, then cloud, then fallback. A fixed order is easier to reason about under incident pressure than a weighted algorithm, and for most workloads the extra sophistication of least-connections or weighted round-robin adds latency and operational surface without changing outcomes.

**Tier 2 — the endpoints and their infrastructure.** The local model server runs as a long-lived process (for example, an ASGI app under a process manager) so the model and its accelerator context stay resident. Cloud endpoints are pre-warmed on a schedule. Connection reuse, retry policy with capped attempts and jitter, and quota monitoring live here.

The router's config is loaded once at startup and refreshed on a timer or via an explicit reload endpoint, so rule changes do not require a redeploy. Parsing and validating the config on every request is a common and avoidable source of per-request overhead; caching the parsed structure in memory removes it.

### A worked latency budget

Suppose a request path consists of the following stages, with illustrative numbers chosen to show the reasoning rather than to describe a measured system:

- Router config lookup and rule evaluation: 0.5 ms (in-memory, no I/O)
- Health state check: 0.2 ms (cached, refreshed on a timer)
- Local inference on a warm process: 40 ms
- Serialization and network hop to a cloud endpoint: 80 ms
- Cloud endpoint cold start, if not pre-warmed: +400 ms

With a warm local endpoint, total is roughly 41 ms. If the local endpoint is unhealthy and the cloud endpoint is warm, total is roughly 81 ms. If the cloud endpoint is cold, total is roughly 481 ms. The arithmetic shows where the leverage is: pre-warming changes the worst case by an order of magnitude more than any router-level micro-optimization. This is why the router should stay dumb — the interesting variance is elsewhere.

### How to measure it yourself

Do not trust published latency numbers, including the illustrative ones above. Instrument the following and compare against your own baseline:

1. **Per-stage timing.** Emit a histogram of time spent in rule evaluation, health check lookup, and upstream call, as separate metrics. A single end-to-end number cannot tell you which stage regressed.
2. **Cold-start frequency.** Count requests where the upstream call exceeded a threshold you set from your own warm-path distribution (for example, 5x the median). This approximates cold-start exposure without needing provider-side data.
3. **Regional comparison.** Send a fixed synthetic request to each candidate region on a schedule and record latency. Compare medians, not means; a single cold start will distort a mean.
4. **Quota headroom.** Poll the provider's quota or usage API on a schedule and export the remaining headroom per region as a gauge. Alert on a threshold, not on failure.

A simple load test against the router with a fixed payload, run before and after each change, will surface regressions in the decision path. Any change that moves the decision-path latency by more than a fraction of a millisecond deserves scrutiny.

## Implementation

The examples below use Go for the router and Python for the local model server. The principles transfer to other stacks.

### Step 1: Define routing rules

Create `router.json`:

```json
{
  "models": [
    {
      "id": "embedding-v3",
      "local": {
        "enabled": true,
        "endpoint": "http://localhost:8000/infer",
        "health_path": "/health"
      },
      "cloud": {
        "enabled": true,
        "endpoint": "https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/embedding-v3/invocations",
        "region": "us-east-1"
      },
      "fallback": {
        "enabled": true,
        "endpoint": "https://fallback.example.com/infer"
      }
    },
    {
      "id": "classifier-v1",
      "local": {
        "enabled": false
      },
      "cloud": {
        "enabled": true,
        "endpoint": "https://runtime.sagemaker.eu-central-1.amazonaws.com/endpoints/classifier-v1/invocations",
        "region": "eu-central-1"
      }
    }
  ]
}
```

Note the URL shapes: a managed endpoint hostname and a self-hosted service differ, and the config should not assume they are interchangeable beyond "an HTTP endpoint that accepts a JSON body."

### Step 2: Build the router

The router exposes `/infer`, `/health`, and `/config/reload`. The corrected core logic below fixes a request-forwarding bug in the common naive version: the original body is read but never attached to the outgoing request, and the incoming `Content-Length` header is copied, which will be wrong for the new request.

```go
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"sync"
	"time"
)

type ModelConfig struct {
	ID     string `json:"id"`
	Local  struct {
		Enabled    bool   `json:"enabled"`
		Endpoint   string `json:"endpoint"`
		HealthPath string `json:"health_path"`
	} `json:"local"`
	Cloud struct {
		Enabled  bool   `json:"enabled"`
		Endpoint string `json:"endpoint"`
		Region   string `json:"region"`
	} `json:"cloud"`
	Fallback struct {
		Enabled  bool   `json:"enabled"`
		Endpoint string `json:"endpoint"`
	} `json:"fallback"`
}

type RouterConfig struct {
	Models []ModelConfig `json:"models"`
}

type Router struct {
	config     RouterConfig
	configPath string
	mu         sync.RWMutex
	client     *http.Client
}

func (r *Router) loadConfig() error {
	data, err := os.ReadFile(r.configPath)
	if err != nil {
		return fmt.Errorf("failed to read config: %w", err)
	}
	var cfg RouterConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return fmt.Errorf("failed to parse config: %w", err)
	}
	r.mu.Lock()
	r.config = cfg
	r.mu.Unlock()
	log.Printf("reloaded config with %d models", len(cfg.Models))
	return nil
}

func (r *Router) selectEndpoint(modelID string) (string, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	var model *ModelConfig
	for i := range r.config.Models {
		if r.config.Models[i].ID == modelID {
			model = &r.config.Models[i]
			break
		}
	}
	if model == nil {
		return "", fmt.Errorf("model %s not found", modelID)
	}

	if model.Local.Enabled {
		resp, err := r.client.Get(model.Local.Endpoint + model.Local.HealthPath)
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return model.Local.Endpoint, nil
			}
		}
	}

	if model.Cloud.Enabled {
		return model.Cloud.Endpoint, nil
	}

	if model.Fallback.Enabled {
		return model.Fallback.Endpoint, nil
	}

	return "", fmt.Errorf("no healthy endpoint for model %s", modelID)
}

func (r *Router) handleInfer(w http.ResponseWriter, req *http.Request) {
	modelID := req.URL.Query().Get("model_id")
	if modelID == "" {
		http.Error(w, "model_id required", http.StatusBadRequest)
		return
	}

	target, err := r.selectEndpoint(modelID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusServiceUnavailable)
		return
	}

	body, err := io.ReadAll(req.Body)
	if err != nil {
		http.Error(w, "failed to read request body", http.StatusBadRequest)
		return
	}

	proxyReq, err := http.NewRequestWithContext(
		req.Context(), req.Method, target, bytes.NewReader(body),
	)
	if err != nil {
		http.Error(w, "failed to build upstream request", http.StatusInternalServerError)
		return
	}
	// Copy only end-to-end headers; hop-by-hop and length are set by the client.
	proxyReq.Header.Set("Content-Type", req.Header.Get("Content-Type"))

	resp, err := r.client.Do(proxyReq)
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to forward: %v", err), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	w.WriteHeader(resp.StatusCode)
	io.Copy(w, resp.Body)
}

func main() {
	router := &Router{
		configPath: "./router.json",
		client: &http.Client{
			Timeout: 30 * time.Second,
			Transport: &http.Transport{
				MaxIdleConns:        100,
				MaxIdleConnsPerHost: 20,
				IdleConnTimeout:     90 * time.Second,
			},
		},
	}
	if err := router.loadConfig(); err != nil {
		log.Fatal(err)
	}

	go func() {
		ticker := time.NewTicker(60 * time.Second)
		defer ticker.Stop()
		for range ticker.C {
			if err := router.loadConfig(); err != nil {
				log.Printf("failed to reload config: %v", err)
			}
		}
	}()

	http.HandleFunc("/infer", router.handleInfer)
	http.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	http.HandleFunc("/config/reload", func(w http.ResponseWriter, _ *http.Request) {
		if err := router.loadConfig(); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		w.WriteHeader(http.StatusOK)
	})

	log.Println("starting router on :8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}
```

Two details matter here. First, the health check runs synchronously inside `selectEndpoint` on every request, which is a latency and availability risk: a slow health endpoint will stall inference. Better practice is to maintain health state in a background goroutine and read the cached result in the decision path. Second, `http.Client` must be reused; creating a new client per request defeats connection pooling.

### Step 3: The local model server

```python
from fastapi import FastAPI
import torch
from pydantic import BaseModel

app = FastAPI()
model = torch.jit.load("embedding-v3.pt")
model.eval()

class InferRequest(BaseModel):
    text: str

@app.post("/infer")
def infer(req: InferRequest):
    with torch.no_grad():
        embedding = model.encode(req.text).tolist()
    return {"embedding": embedding}

@app.get("/health")
def health():
    return {"status": "ok"}
```

Run it as a long-lived process so the model and accelerator context stay resident:

```bash
uvicorn main:app --workers=4 --host=0.0.0.0 --port=8000
```

The worker count is a tuning parameter, not a fixed answer. Too few workers and requests queue; too many and the accelerator is oversubscribed and memory pressure grows. Measure throughput and tail latency as you vary it.

### Step 4: Pre-warm cloud endpoints

A scheduled job that invokes each managed endpoint with a small payload keeps capacity warm. The exact interval depends on the provider's idle timeout, which is documented per service; choose an interval comfortably below it.

```python
import json
import os
import boto3

client = boto3.client("runtime.sagemaker", region_name=os.environ["AWS_REGION"])
ENDPOINT = os.environ["ENDPOINT_NAME"]

def lambda_handler(event, context):
    try:
        client.invoke_endpoint(
            EndpointName=ENDPOINT,
            ContentType="application/json",
            Body=json.dumps({"inputs": ["warmup"]}),
        )
        return {"status": "warmed"}
    except Exception as exc:
        return {"error": str(exc)}
```

Schedule the function at an interval below the endpoint's idle timeout. The cost is a function of your invocation count and the provider's pricing, so compute it from your own schedule rather than assuming a figure.

### Step 5: Deploy the router

The router is small enough to run as a single modest instance or a small container with tight resource limits. Size it from measurement: run a load test, observe CPU and memory at your target request rate, and add headroom. Expose a `/metrics` endpoint with counters and histograms for routing decisions, upstream errors, and per-stage latency.

## Failure modes to plan for

**Health checks that lie.** A `/health` endpoint that returns `200 OK` unconditionally will keep a broken endpoint in rotation. A shallow check (process is up) and a deep check (a trivial inference succeeds and returns a plausible shape) catch different classes of failure. Deep checks cost real compute, so run them on a schedule rather than per request, and cache the result.

**Config drift across replicas.** If multiple router instances read config from a shared store on a timer, different instances can serve different rules for a window. Use a store that supports conditional writes or versioning, and have the router reject a config whose version is older than the one it already holds.

**Quota exhaustion.** Managed endpoints enforce per-region quotas. When a quota is hit, the error surfaced to the caller is often a generic throttling response that does not name the quota. Track headroom proactively and fail over to another region before the quota is reached, rather than reacting to errors.

**Model version skew.** If the router points at a model identifier but the deployed model has changed underneath it, requests can fail or, worse, return plausible but wrong results. Expose a version identifier from the model server and include the expected version in the router config; treat a mismatch as an unhealthy endpoint.

**Connection exhaustion on the local side.** A local server with a fixed worker count and a bounded file-descriptor limit will reject connections under burst load. This is a capacity-planning problem, not a routing problem, but it presents as routing errors. Monitor the local server's queue depth and rejection rate directly.

**Silent fallback masking degradation.** A fallback path that always succeeds can hide a persistently unhealthy primary. Alert on fallback activation rate, not only on error rate.

## When this design is the wrong choice

This pattern fits when models are either local (accelerator-backed) or managed, when a few tens of milliseconds of cloud latency is acceptable, when routing rules are simple enough to express as an ordered list, and when traffic justifies the operational cost of running a local inference path.

It is a poor fit when:

- Sub-10 ms latency is required end to end; a cloud fallback will violate that budget.
- The number of models is large enough that a flat config file becomes unmanageable. At that point, a database-backed config with a management interface is warranted, and the router should read from it rather than from a file.
- The router itself would run in a serverless function, where per-invocation startup overhead is added to every request. In that case, embed the routing decision in the inference function rather than putting a separate hop in front of it.
- Models require persistent session state, such as streaming or stateful inference; the ordered-endpoint model assumes each request is independent.

## A decision checklist

Before adding a routing layer, answer these:

1. What is the measured cold-start cost of each endpoint, and what interval keeps it warm?
2. What is the measured latency from your user population to each candidate region?
3. What is the quota headroom per region, and how is it monitored?
4. What does a deep health check cost, and how often can it run?
5. What is the fallback activation rate, and who is alerted when it rises?
6. How does a config change propagate, and how are stale replicas prevented from serving it?
7. What is the router's own latency contribution, measured separately from upstream latency?

If any of these cannot be answered with a metric, the routing layer is not ready to carry production traffic.

## Take action in the next 30 minutes

Instrument your current routing path before changing it. Add a timer around the upstream call and one around the routing decision itself, then run a fixed load test and record both distributions:

```bash
# Replace the URL with your own router endpoint.
hey -n 500 -c 10 -m POST \
  -H 'Content-Type: application/json' \
  -d '{"text":"hello"}' \
  'http://localhost:8080/infer?model_id=embedding-v3'
```

Compare the median and the 99th percentile. A large gap between them, or a bimodal distribution, indicates cold starts or connection churn rather than a slow model. That measurement tells you which layer to fix first — and it is the one piece of information no architecture diagram will give you.
