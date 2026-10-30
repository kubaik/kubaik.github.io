# Use feature flags to deploy AI models safely

## Why model rollouts need a control plane separate from deployment

Deploying a model and exposing it to traffic are two different decisions. Most tutorials collapse them into one: build an image, push it, and every user hits the new model the moment the pods become ready. That coupling is what makes AI rollouts expensive. A one-line prompt template change, a tokenizer version bump, or a quantization tweak can shift output quality in ways that only show up under real traffic, and without a control plane the only remedy is another deploy.

A feature flag layer decouples that. The model container ships to the cluster and sits idle until a flag routes traffic to it. That gives you four capabilities that matter for AI specifically:

- An instant kill switch when output quality degrades, without rebuilding or restarting anything.
- Canary splits by user ID, percentage, or attribute, so a new model sees a controlled slice of traffic.
- Shadow mode, where the new model scores requests and logs results but its output is discarded before the response is returned.
- Metric-gated promotion, where the flag value changes only after latency and error thresholds hold for a defined window.

The rest of this article builds that layer end to end: a flagged inference endpoint, a canary pipeline, automated rollback, and dashboards that show flag state next to model KPIs.

## Prerequisites and target architecture

The stack below assumes:

- A Kubernetes cluster, or a local minikube with at least 4 vCPUs and 8 GB RAM.
- Docker or Podman for image builds.
- Node 20 LTS and Python 3.11.
- A flag management system. Any system with an SDK, percentage rollout, and a local evaluation cache works; the examples use an open-source self-hosted flag service with a PostgreSQL backend.
- An inference service. The examples use a FastAPI app wrapping a small CPU-friendly model exported to OpenVINO IR format. The model choice matters less than the flag wiring.
- Prometheus and Grafana for metrics.

By the end you will have:

1. A flagged inference endpoint behind an ingress.
2. A canary pipeline sending a configurable percentage of traffic to the new model.
3. Automated rollback when error rate exceeds a threshold in a rolling window.
4. A Grafana dashboard showing flag state and model KPIs together.

## Step 1 — stand up the flag service and metrics stack

Install the base tooling:

```bash
curl -LO https://dl.k8s.io/release/v1.28.0/bin/linux/amd64/kubectl
chmod +x kubectl
sudo mv kubectl /usr/local/bin/

minikube start --driver=docker --cpus=4 --memory=8192

curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update
```

Deploy PostgreSQL for the flag service, then the flag service itself. The exact chart name and values depend on which flag system you choose; the shape is the same — a database, a web service, and an ingress.

```bash
kubectl create ns flags

helm install postgres bitnami/postgresql --namespace flags -f - <<EOF
primary:
  persistence:
    size: 20Gi
  resources:
    requests:
      cpu: 500m
      memory: 1Gi
auth:
  postgresPassword: "$(openssl rand -base64 16)"
EOF
```

Once the flag service is reachable, create a project and generate an SDK key for server-side evaluation. Store that key in a Kubernetes secret rather than in the deployment manifest:

```bash
kubectl create secret generic flag-sdk-key \
  --namespace ai \
  --from-literal=key="$FLAG_SDK_KEY"
```

Install Prometheus and Grafana:

```bash
kubectl create ns monitoring
helm install prometheus prometheus-community/prometheus --namespace monitoring
helm install grafana grafana/grafana --namespace monitoring
kubectl port-forward svc/grafana -n monitoring 3000:80 &
```

Add the Prometheus data source at `http://prometheus-server.monitoring.svc:80`. Change the Grafana admin password before exposing the port beyond localhost.

## Step 2 — build the flagged inference service

Project layout:

```
ai-features/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── models.py
│   ├── flags.py
│   └── metrics.py
├── Dockerfile
├── requirements.txt
└── k8s/
    ├── deployment.yaml
    └── service.yaml
```

Dependencies:

```
fastapi==0.109.0
uvicorn[standard]==0.27.0
prometheus-client==0.19.0
numpy==1.26.3
pydantic==2.6.1
```

Install your flag SDK alongside these, pinned to a known version.

The flag helper is the component that decides which model serves a request. The important design choice is the failure default: if the flag service is unreachable or the flag is missing, return the last known good version rather than raising. A flag outage must not become an inference outage.

```python
# app/flags.py
import os
import logging
from flag_sdk import Client  # replace with your SDK's import

logger = logging.getLogger(__name__)

FLAG_URL = os.getenv("FLAG_URL", "http://flags.flags.svc.cluster.local:8000")
FLAG_KEY = os.getenv("FLAG_KEY")

client = Client(environment_key=FLAG_KEY, api_url=FLAG_URL)

DEFAULT_VERSION = "v1"

def get_model_flag(user_id: str) -> str:
    """
    Returns 'v1' or 'v2'. Falls back to DEFAULT_VERSION on any error
    so that a flag-service outage never blocks inference.
    """
    try:
        state = client.get_feature_state("recommendation_model")
        if not state.is_enabled:
            return DEFAULT_VERSION
        return state.get_value(DEFAULT_VERSION)
    except Exception as exc:
        logger.warning("flag evaluation failed, defaulting to %s: %s",
                       DEFAULT_VERSION, exc)
        return DEFAULT_VERSION
```

Two details worth noting. First, `get_value` takes a default argument, so a flag that exists but has no value set returns the default rather than `None`. Second, catching broad exceptions here is deliberate: a slow or malformed flag response should degrade to the default, not surface as a 500 to the caller.

The model wrapper loads the compiled model once at process start, not per request:

```python
# app/models.py
from typing import List
import logging
import numpy as np
from openvino.runtime import Core

logger = logging.getLogger(__name__)

class Model:
    def __init__(self, model_path: str):
        core = Core()
        self.compiled_model = core.compile_model(model_path, "CPU")

    def predict(self, user_embedding: np.ndarray,
                product_embeddings: np.ndarray) -> List[float]:
        try:
            output_tensor = self.compiled_model.output(0)
            result = self.compiled_model(
                [user_embedding, product_embeddings]
            )[output_tensor]
            return result.tolist()
        except Exception as exc:
            logger.error("inference failed: %s", exc)
            raise
```

The FastAPI layer wires flag evaluation, inference, and metrics together:

```python
# app/main.py
import time
from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, Gauge

from .flags import get_model_flag
from .models import Model

app = FastAPI(title="Flagged Inference Service")

REQUEST_COUNT = Counter(
    "ai_requests_total",
    "Total inference requests",
    ["model_version", "http_status"],
)
REQUEST_LATENCY = Histogram(
    "ai_request_latency_seconds",
    "Inference latency in seconds",
    ["model_version"],
)
FLAG_EVAL_ERRORS = Counter(
    "ai_flag_eval_errors_total",
    "Flag evaluations that fell back to the default",
)

models = {
    "v1": Model("/models/v1/ir_model.xml"),
    "v2": Model("/models/v2/ir_model.xml"),
}

class RecommendationRequest(BaseModel):
    user_id: str
    product_ids: List[str]

@app.post("/recommend")
async def recommend(payload: RecommendationRequest):
    version = get_model_flag(payload.user_id)
    start = time.perf_counter()

    try:
        model = models.get(version)
        if model is None:
            raise ValueError(f"unknown model version: {version}")

        scores = model.predict(
            np.random.rand(256),
            np.random.rand(len(payload.product_ids), 256),
        )
        REQUEST_COUNT.labels(model_version=version, http_status="200").inc()
        REQUEST_LATENCY.labels(model_version=version).observe(
            time.perf_counter() - start
        )
        return {"scores": scores, "model_version": version}

    except Exception as exc:
        REQUEST_COUNT.labels(model_version=version, http_status="500").inc()
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/health")
async def health():
    return {"status": "ok"}
```

The `model_version` label on every metric is what makes per-variant dashboards possible. Without it, you can see that latency rose but not which model caused it.

Dockerfile:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Deployment manifest with the SDK key pulled from a secret and probes wired to `/health`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-recommendation
  namespace: ai
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-recommendation
  template:
    metadata:
      labels:
        app: ai-recommendation
    spec:
      containers:
        - name: ai
          image: ai-features:1.0.0
          ports:
            - containerPort: 8000
          env:
            - name: FLAG_URL
              value: http://flags.flags.svc.cluster.local:8000
            - name: FLAG_KEY
              valueFrom:
                secretKeyRef:
                  name: flag-sdk-key
                  key: key
          resources:
            requests:
              cpu: "500m"
              memory: 512Mi
            limits:
              cpu: "1"
              memory: 1Gi
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 15
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 5
```

## Step 3 — canary, shadow, and rollback

### Canary routing

A percentage rollout flag evaluates per user, not per request, so a given user consistently sees one model. That consistency matters: if a user flips between models mid-session, ranking output changes unpredictably and any quality comparison is confounded.

Configure the flag with a percentage split, start at 5%, and promote in steps. The decision rule should be a metric condition, not a calendar:

- Promote only if `error_rate_v2 < 1%` and `p95_latency_v2 < 350ms` over a rolling 30-minute window.
- Roll back automatically if either threshold is breached for two consecutive 5-minute windows.
- Never promote during a traffic trough, where the sample size is too small to be meaningful.

### Shadow mode

Shadow mode runs the candidate model on live requests but returns the incumbent's output. It gives you latency and error data under production traffic with zero user-visible risk.

The cost is real: shadow mode roughly doubles inference compute for the shadowed fraction. To estimate it, take your per-request CPU cost from the container's `container_cpu_usage_seconds_total` metric, multiply by the shadowed request rate, and compare against your node pricing. If shadowing 100% of traffic would double your inference spend, shadow 10% instead — the latency distribution converges quickly at that sample size.

A shadow implementation calls both models and discards the second result:

```python
import asyncio

async def shadow_predict(primary, shadow, user_emb, product_embs):
    primary_task = asyncio.create_task(primary.predict_async(user_emb, product_embs))
    shadow_task = asyncio.create_task(shadow.predict_async(user_emb, product_embs))

    result = await primary_task
    try:
        shadow_result = await shadow_task
        SHADOW_DIVERGENCE.observe(compare_rankings(result, shadow_result))
    except Exception as exc:
        SHADOW_ERRORS.inc()
        logger.warning("shadow model failed: %s", exc)
    return result
```

Note that the shadow result is awaited but never returned, and its failure is counted rather than raised. A shadow model that crashes must not affect the primary path.

### Timeouts and fallbacks

Inference should have a hard timeout. When it fires, serve a cached or degraded response rather than hanging:

```python
async def predict_with_timeout(model, user_emb, product_embs, timeout=0.3):
    try:
        return await asyncio.wait_for(
            model.predict_async(user_emb, product_embs), timeout=timeout
        )
    except asyncio.TimeoutError:
        INFERENCE_TIMEOUTS.labels(model_version=model.version).inc()
        cached = await cache.get(cache_key(user_emb))
        if cached is not None:
            return cached
        return [0.0] * len(product_embs)
```

Do not use `eval()` to deserialize cached values. Store them as JSON and parse with `json.loads`, or use a typed serialization format. The earlier pattern of `eval(cached)` is a code-execution vulnerability if anything else can write to that cache key.

### Flag propagation delay

Flag SDKs cache evaluations locally, and the cache TTL determines how long a rollback takes to reach every pod. If the SDK default is 30 seconds, a kill switch is really a 30-second kill switch. Two mitigations:

- Set the cache TTL to a value you can tolerate as your worst-case rollback time. Five seconds is a common choice for staging; production teams often accept 10–30 seconds in exchange for fewer flag-service round trips.
- Do not add a background job that polls the flag service per pod on a short interval unless you have measured the load. With N pods polling every T seconds, you generate N/T requests per second against the flag service, and that scales badly during an incident when every pod is retrying.

## Step 4 — observability and tests

Prometheus scrapes the `/metrics` endpoint that `prometheus-client` exposes automatically. Add a scrape annotation to the pod template:

```yaml
metadata:
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "8000"
    prometheus.io/path: "/metrics"
```

Useful queries to build the dashboard around:

- Request rate by model: `sum by (model_version) (rate(ai_requests_total[1m]))`
- P95 latency by model: `histogram_quantile(0.95, sum by (le, model_version) (rate(ai_request_latency_seconds_bucket[5m])))`
- Error ratio by model: `sum by (model_version) (rate(ai_requests_total{http_status="500"}[5m])) / sum by (model_version) (rate(ai_requests_total[5m]))`
- Flag fallback rate: `rate(ai_flag_eval_errors_total[5m])`

That last metric is the one teams forget. A rising fallback rate means flag evaluation is failing and silently pinning traffic to the default model — which looks like a successful rollout until someone checks which model actually served the requests.

Unit tests should cover the fallback path explicitly:

```python
import pytest
from unittest.mock import patch, MagicMock
from app.flags import get_model_flag

def test_defaults_to_v1_on_error():
    with patch("app.flags.client.get_feature_state") as mock_state:
        mock_state.side_effect = Exception("flag service unreachable")
        assert get_model_flag("user123") == "v1"

def test_v2_when_enabled():
    with patch("app.flags.client.get_feature_state") as mock_state:
        state = MagicMock()
        state.is_enabled = True
        state.get_value.return_value = "v2"
        mock_state.return_value = state
        assert get_model_flag("user124") == "v2"

def test_default_when_flag_disabled():
    with patch("app.flags.client.get_feature_state") as mock_state:
        state = MagicMock()
        state.is_enabled = False
        mock_state.return_value = state
        assert get_model_flag("user125") == "v1"
```

Load test with k6 to establish the latency baseline before any canary:

```javascript
import http from 'k6/http';
import { check } from 'k6';

export const options = {
  stages: [
    { duration: '2m', target: 100 },
    { duration: '5m', target: 500 },
    { duration: '2m', target: 0 },
  ],
  thresholds: {
    http_req_duration: ['p(95)<400'],
  },
};

export default function () {
  const payload = JSON.stringify({
    user_id: `user_${Math.floor(Math.random() * 10000)}`,
    product_ids: ['p1', 'p2', 'p3'],
  });
  const res = http.post(
    'http://ai-recommendation.ai.svc.cluster.local/recommend',
    payload,
    { headers: { 'Content-Type': 'application/json' } }
  );
  check(res, { 'status was 200': (r) => r.status === 200 });
}
```

## Measuring flag overhead honestly

A common claim is that flag evaluation adds "a few milliseconds." Whether that is true for your service depends on the SDK's caching behavior, not on the flag system's marketing. Measure it rather than assuming.

The method: run the same load profile twice against the same image, once with the flag check in the request path and once with the flag value read from a constant. Compare the `histogram_quantile` output for `ai_request_latency_seconds` at p50 and p95 in Grafana, and use the same k6 script both times so the request mix is identical. The difference is your flag overhead, and it will be dominated by whether the SDK evaluates locally from cache or makes a network call per request.

The design goal is local evaluation. With a local cache, the flag check is a dictionary lookup plus a TTL comparison, and the overhead is lost in the noise of inference. Without it, every request pays a network round trip, and the flag service becomes a hard dependency on your critical path.

## Failure modes to design for

**Flag service unavailable at startup.** If the SDK cannot fetch its initial configuration, the service should still start and serve the default model. Blocking startup on flag availability turns a flag outage into a full outage.

**Flag exists but has an unexpected value.** Validate the returned version against the set of loaded models before using it. An unrecognized value should log loudly and fall back, not raise a 500.

**Cache stampede after a flag change.** When a flag flips, every pod's cache expires at roughly the same time and they all refetch. With a small number of pods this is fine; with hundreds it can spike load on the flag service. Jitter the TTL by a random fraction of its value to spread the refetches.

**Shadow model starving the primary.** If the shadow model shares a CPU-bound node with the primary, its compute competes directly. Run shadow replicas on separate nodes or with strict CPU limits, and monitor the primary's latency during shadow periods to confirm it is unaffected.

**Metrics cardinality explosion.** Labelling by `user_id` or any high-cardinality field will overwhelm Prometheus. Label by `model_version` and `http_status` only; keep per-user detail in logs or traces.

## A worked promotion decision

Suppose the canary is at 5% and the dashboard shows, over the last 30 minutes:

- `v1`: 95,000 requests, 210 errors.
- `v2`: 5,000 requests, 9 errors.
- `v2` p95 latency: 340 ms against a 350 ms budget.

The error rates are 0.22% for v1 and 0.18% for v2 — both under the 1% threshold, so the error condition passes. Latency passes with 10 ms of headroom. But 5,000 requests is a thin sample for a 1% threshold: at that volume, one additional error moves the rate by 0.02 percentage points, and the confidence interval around 0.18% is wide enough to overlap 1%. The honest decision is to hold at 5% for another window rather than promote, because the sample cannot yet distinguish a good model from a marginal one. This is why promotion gates should specify a minimum request count alongside the rate thresholds.

## FAQ

**Does the flag check belong before or after authentication?**
After. The flag decision usually depends on a user identifier, and evaluating it before authentication means unauthenticated traffic can influence your rollout metrics.

**Can this pattern work with serverless inference?**
Yes, but cold starts change the caching calculus. Each new execution environment fetches the flag configuration once; if your concurrency is high and traffic is bursty, that is many fetches. Cache the flag state in a global scoped to the execution environment and set the TTL to something longer than your typical invocation duration.

**How do you roll back without redeploying?**
Set the flag to the previous model version. The running pods pick up the change on their next cache refresh, so worst-case rollback time equals the SDK cache TTL. Deploying the new model and exposing it are separate operations, which is the entire point.

**What if both models are needed simultaneously?**
That is what the percentage rollout gives you. During a canary, both models serve live traffic, and the per-`model_version` metrics tell you how each is performing.

## Take action now

Open your inference service's metrics endpoint and check whether request counters are labelled by model version. If they are not, add a `model_version` label to your request counter and latency histogram, deploy that change, and confirm in Prometheus that `sum by (model_version) (rate(ai_requests_total[5m]))` returns more than one series during a canary. Without that label, no promotion gate you write later will be able to tell you which model caused a regression.
