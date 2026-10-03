# Designing an AI Alert Triage Layer That Reduces Pages

Most alert-triage tutorials stop at the happy path: an alert arrives, a model classifies it, a decision is emitted. Production adds the parts that decide whether the system helps or hurts — duplicate storms, stale metrics, a vector store that vanishes, and a model endpoint that times out at the worst moment. This article covers the full design, including the failure modes that cause AI triage layers to increase noise instead of reducing it.

## The problem AI triage is supposed to solve

Alert fatigue is a routing problem before it is a machine-learning problem. A typical on-call rotation receives a long tail of alerts that are technically firing but operationally uninteresting: a single pod restarting repeatedly, a scrape target that flaps, a warning that self-resolves within two minutes. Each of those consumes attention, and attention is the scarce resource.

A triage layer sits between Alertmanager and the paging provider. It does not replace Alertmanager's grouping, inhibition, or silencing — those still run first. It adds three decisions on top:

1. **Auto-close** — the alert matches a known-benign pattern or a recent duplicate, so it is resolved without human contact.
2. **Queue** — the alert is real but not urgent; it is recorded and surfaced in a digest.
3. **Page** — a human is woken now.

The value of the layer is entirely determined by how well it distinguishes these three. A triage service that pages on everything is pure overhead. A triage service that drops real pages is worse than no triage at all. Everything below is in service of getting that boundary right and being able to prove where it sits.

## Prerequisites and what you will build

The reference implementation assumes:

- A Kubernetes cluster with Prometheus scraping pod metrics on a fixed interval (15s is a common choice).
- A vector database for storing incident fingerprints. Any store with approximate nearest-neighbour search works; the examples use Qdrant because its API is small and it runs as a single container.
- A Python 3.11+ runtime. The service uses `fastapi`, `pydantic`, `httpx`, `qdrant-client`, and `prometheus-client`.
- An LLM endpoint. This can be a hosted chat-completions API or a locally served model behind an OpenAI-compatible or Ollama-compatible HTTP interface. The code below treats the endpoint as a configurable URL and model name.
- A paging provider with an events or incidents API. PagerDuty and Opsgenie both expose REST APIs suitable for this; the example uses the PagerDuty Events API shape.

What you will build is a small FastAPI service that:

1. Receives Alertmanager webhook payloads at `/ingest`.
2. Computes a stable fingerprint per alert.
3. Looks up recent incidents with that fingerprint in the vector store.
4. Applies deterministic rules first (known-bad list, cooldown, duplicate window).
5. Falls back to a short LLM call only when the deterministic rules are inconclusive.
6. Routes the surviving alerts to the paging provider and logs every decision.

The goal is not to write new alert rules. It is to make the existing ones route better.

## Step 1 — environment setup

Create a namespace for the service:

```bash
kubectl create ns alert-router
```

Prometheus should already be scraping. If you run the Prometheus Operator, a minimal scrape configuration looks like this:

```yaml
apiVersion: monitoring.coreos.com/v1
kind: Prometheus
metadata:
  name: main
  namespace: monitoring
spec:
  scrapeInterval: 15s
  resources:
    requests:
      memory: 1Gi
    limits:
      memory: 2Gi
```

Deploy the vector store in the same namespace so the service can reach it over cluster DNS:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: qdrant
  namespace: alert-router
spec:
  replicas: 1
  selector:
    matchLabels:
      app: qdrant
  template:
    metadata:
      labels:
        app: qdrant
    spec:
      containers:
      - name: qdrant
        image: qdrant/qdrant:v1.9.0
        ports:
        - containerPort: 6333
        resources:
          limits:
            memory: 2Gi
            cpu: "1"
---
apiVersion: v1
kind: Service
metadata:
  name: qdrant
  namespace: alert-router
spec:
  selector:
    app: qdrant
  ports:
    - port: 6333
```

The Python dependencies:

```text
pydantic==2.7.0
fastapi==0.111.0
uvicorn[standard]==0.29.0
qdrant-client==1.9.0
httpx==0.27.0
prometheus-client==0.20.0
```

Run the service locally to test the fingerprinting logic before deploying:

```bash
uvicorn router:app --reload --port 8000
```

`GET /health` should return `{"status":"ok"}` and `POST /ingest` should accept Alertmanager-shaped payloads.

A common first failure: Alertmanager's webhook body has a specific structure, and a service that expects a different shape will silently drop every alert. Log the raw payload on the first deploy and confirm the fields you depend on are present.

## Step 2 — core implementation

The service is a reverse proxy between Alertmanager and the paging provider. Every alert posted to `/ingest` is fingerprinted, compared against recent incidents, and routed.

```python
import hashlib
import json
import os
import time
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient, models

app = FastAPI()

PROM_URL = os.getenv("PROM_URL", "http://prometheus-operated.monitoring.svc:9090")
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant.alert-router.svc.cluster.local")
PD_API_KEY = os.getenv("PD_API_KEY")
PD_ROUTING_KEY = os.getenv("PD_ROUTING_KEY")
LLM_URL = os.getenv("LLM_URL", "https://api.example-llm-provider.com/v1/chat/completions")
LLM_MODEL = os.getenv("LLM_MODEL", "small-instruct")
LLM_KEY = os.getenv("LLM_KEY")

qdrant = QdrantClient(host=QDRANT_HOST, port=6333)

COOLDOWN_SECONDS = 60

class Alert(BaseModel):
    receiver: str
    status: str
    alerts: list[dict[str, Any]]
    externalURL: str

def fingerprint(alert: dict[str, Any]) -> str:
    """Stable hash over the labels that identify this alert class."""
    labels = alert.get("labels", {})
    seed = json.dumps(
        {
            "alertname": labels.get("alertname"),
            "namespace": labels.get("namespace"),
            "pod": labels.get("pod"),
        },
        sort_keys=True,
    )
    return hashlib.sha256(seed.encode()).hexdigest()

def recent_hits(fp: str, limit: int = 10) -> int:
    """Count recent incidents with the same fingerprint."""
    results = qdrant.scroll(
        collection_name="alert_fingerprints",
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="fingerprint",
                    match=models.MatchValue(value=fp),
                ),
                models.FieldCondition(
                    key="ts",
                    range=models.Range(gte=time.time() - COOLDOWN_SECONDS),
                ),
            ]
        ),
        limit=limit,
    )
    return len(results[0])

@app.post("/ingest")
async def ingest(alert: Alert):
    handled = 0
    for a in alert.alerts:
        fp = fingerprint(a)
        hits = recent_hits(fp)

        # Deterministic rules first.
        if hits > 0 and a.get("status") == "firing":
            a.setdefault("annotations", {})["ai_decision"] = "auto_closed_duplicate"
            handled += 1
            continue

        decision = await llm_decide(a)
        a.setdefault("annotations", {})["ai_decision"] = decision

        if decision == "page":
            await send_to_pagerduty(a)
            handled += 1
        else:
            a["status"] = "resolved"
            handled += 1

        qdrant.upsert(
            collection_name="alert_fingerprints",
            points=[
                models.PointStruct(
                    id=hashlib.sha256(f"{fp}:{time.time()}".encode()).hexdigest(),
                    vector=[0.0] * 8,  # placeholder; real embeddings optional
                    payload={
                        "fingerprint": fp,
                        "labels": a.get("labels", {}),
                        "startsAt": a.get("startsAt"),
                        "ts": time.time(),
                    },
                )
            ],
        )
    return {"status": "ok", "handled": handled}

async def llm_decide(alert: dict[str, Any]) -> str:
    prompt = (
        "You are an on-call triage assistant. Given the alert below, "
        "decide whether a human should be paged now. "
        "Reply with exactly one word: page or queue.\n\n"
        f"Alert: {json.dumps(alert)}\n"
    )
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            LLM_URL,
            headers={"Authorization": f"Bearer {LLM_KEY}"},
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 5,
                "temperature": 0,
            },
            timeout=2.0,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip().lower()
        return "page" if text.startswith("page") else "queue"

async def send_to_pagerduty(alert: dict[str, Any]):
    dedup_key = alert.get("labels", {}).get("alertname", "unknown")
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://events.pagerduty.com/v2/enqueue",
            headers={"Content-Type": "application/json"},
            json={
                "routing_key": PD_ROUTING_KEY,
                "event_action": "trigger",
                "dedup_key": dedup_key,
                "payload": {
                    "summary": alert.get("labels", {}).get("alertname", "Alert"),
                    "source": alert.get("labels", {}).get("instance", "unknown"),
                    "severity": alert.get("labels", {}).get("severity", "warning"),
                    "custom_details": alert,
                },
            },
            timeout=3.0,
        )
        resp.raise_for_status()
```

Several design choices here are load-bearing:

- **Deterministic rules run before the model.** Duplicate suppression is exact and cheap. Sending every duplicate to an LLM wastes latency and money and introduces a chance of inconsistent decisions on identical inputs.
- **The fingerprint is a hash, not an embedding.** For alert triage, the identity of an alert class is its label set, not its semantic meaning. Hashing labels is reproducible and debuggable; embeddings are neither unless you also store the source text.
- **The vector store is used as a time-windowed key-value index.** The `vector` field is a placeholder. The lookup that matters is the payload filter on `fingerprint` and `ts`. This is deliberate: it keeps the store's role honest and avoids pretending semantic similarity is doing work it is not.
- **The LLM call has a hard timeout.** Two seconds is generous for a five-token completion; anything slower should be treated as an outage, not a slow path.

A frequent bug in this design: using Python's built-in `hash()` for fingerprints. It is salted per process, so the value changes across restarts and replicas. Use `hashlib.sha256` over a canonical JSON serialization instead.

## Step 3 — edge cases and fail-open behaviour

The triage layer is on the critical path between an alert firing and a human being notified. Every failure mode must resolve in the direction of paging, not silence.

| Failure | Detection signal | Correct action |
|---|---|---|
| Alertmanager retries the same alert | Same fingerprint within cooldown window | Drop duplicate, do not re-page |
| Prometheus scrape lag exceeds the alert's evaluation window | Scrape timestamp freshness metric | Fail open: page immediately |
| Vector store unreachable | Connection error on lookup | Fail open: page immediately |
| LLM endpoint errors or times out | Non-2xx response or timeout | Fail open: page immediately |
| Fingerprint collision across namespaces | Two distinct alerts hash identically | Include `namespace` in the fingerprint seed |
| Triage service itself is down | Health endpoint fails | Paging provider must be reachable directly as a fallback |

The collision case deserves detail. Two alerts with `alertname=PodRestarting` in `default` and `kube-system` have different operational meaning. If the fingerprint is seeded only on `alertname` and `pod`, they collide. Including `namespace` in the seed separates them. This is not a percentage improvement to be quoted — it is a correctness fix, and you can verify it by constructing two such alerts and asserting their fingerprints differ.

The fail-open rule is the single most important line in the service. Implement it explicitly:

```python
async def llm_decide(alert: dict[str, Any]) -> str:
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                LLM_URL,
                headers={"Authorization": f"Bearer {LLM_KEY}"},
                json={
                    "model": LLM_MODEL,
                    "messages": [{"role": "user", "content": build_prompt(alert)}],
                    "max_tokens": 5,
                    "temperature": 0,
                },
                timeout=2.0,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip().lower()
            return "page" if text.startswith("page") else "queue"
    except Exception:
        # Any failure to classify means a human decides.
        return "page"
```

A known-bad list handles alerts that should never page regardless of model output — `Watchdog`, `Info`-severity notifications, and anything your team has explicitly agreed is noise. Keep it in a ConfigMap and reload it periodically rather than baking it into the image:

```python
import json

BAD_ALERTS: set[str] = set()

def load_bad_alerts(path: str = "/etc/alert-router/bad_alerts.json") -> None:
    global BAD_ALERTS
    with open(path) as f:
        BAD_ALERTS = set(json.load(f))

def is_known_bad(alert: dict[str, Any]) -> bool:
    return alert.get("labels", {}).get("alertname") in BAD_ALERTS
```

Call `is_known_bad` before the LLM path. Without it, informational alerts will occasionally be classified as page-worthy, and the model has no way to know your team's conventions.

Finally, expose a health endpoint that actually checks the dependencies:

```python
@app.get("/health")
async def health():
    try:
        qdrant.get_collection("alert_fingerprints")
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {"status": "ok"}
```

If the vector store is gone and the health check is superficial, alerts can disappear silently. The health endpoint exists so that an external monitor notices before your on-call rotation does.

## Step 4 — observability and tests

Three signals matter:

1. **Decision mix.** A counter labelled by decision (`page`, `queue`, `auto_closed_duplicate`) tells you the ratio the layer is producing. If `page` is not meaningfully lower than the pre-triage rate, the layer is not doing its job.
2. **LLM latency and error rate.** A histogram of call duration and a counter of failures, both labelled by endpoint. This is where you discover that your timeout is too tight or your provider is degrading.
3. **False negatives.** The hardest and most important signal. There is no automatic ground truth, but a workable proxy is: alerts that were queued or auto-closed and then re-fired within a short window. That pattern usually means the first decision was wrong.

```python
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response

DECISIONS = Counter(
    "alert_router_decisions_total",
    "Triage decisions by outcome",
    ["decision"],
)
LLM_LATENCY = Histogram(
    "alert_router_llm_duration_seconds",
    "LLM classification latency",
)
LLM_ERRORS = Counter(
    "alert_router_llm_errors_total",
    "LLM classification failures",
)

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

Increment `DECISIONS.labels(decision=...)` at each branch, wrap the LLM call in `LLM_LATENCY.time()`, and increment `LLM_ERRORS` in the exception handler. These three are enough to answer "is the layer working" without a bespoke dashboard.

Tests should cover the deterministic paths exhaustively and the LLM path only for its contract:

```python
from unittest.mock import patch

from fastapi.testclient import TestClient

from router import app

client = TestClient(app)

def make_payload(alertname: str, namespace: str = "default") -> dict:
    return {
        "receiver": "alert-router",
        "status": "firing",
        "externalURL": "http://alertmanager",
        "alerts": [
            {
                "status": "firing",
                "labels": {
                    "alertname": alertname,
                    "namespace": namespace,
                    "pod": "nginx-0",
                },
                "annotations": {},
                "startsAt": "2026-01-01T00:00:00Z",
            }
        ],
    }

def test_duplicate_is_auto_closed():
    payload = make_payload("PodRestarting")
    first = client.post("/ingest", json=payload)
    second = client.post("/ingest", json=payload)
    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["handled"] == 1

def test_llm_failure_pages():
    with patch("router.llm_decide", side_effect=Exception("endpoint down")):
        resp = client.post("/ingest", json=make_payload("HighCPU"))
    assert resp.status_code == 200

def test_namespace_separates_fingerprints():
    from router import fingerprint
    a = make_payload("PodRestarting", "default")["alerts"][0]
    b = make_payload("PodRestarting", "kube-system")["alerts"][0]
    assert fingerprint(a) != fingerprint(b)
```

The third test is the one that catches the collision bug. It is worth writing before you deploy, because the failure is silent and only shows up as a missed page weeks later.

## Measuring whether it actually helps

The original claim — "cut pages 80%" — is the kind of number that should never be asserted without a measurement plan. Here is how to produce your own, honestly.

Define the baseline window before deployment. For at least two weeks, record:

- Pages delivered to humans, per week, from the paging provider's API.
- Alerts received by Alertmanager, per week.
- On-call acknowledgements and the time between page and acknowledgement.

After deployment, record the same three. The ratio of pages to alerts tells you whether the layer is filtering or merely relabelling. The acknowledgement time tells you whether the remaining pages are the ones that matter.

For false negatives, instrument the re-fire proxy described above and review the queued and auto-closed decisions weekly. A sample of twenty reviewed decisions per week is enough to estimate the error rate with useful precision; a sample of zero is how a triage layer quietly becomes a source of missed incidents.

Two illustrative calculations, with assumptions stated:

- If a team receives 500 alerts per week and 40 of them currently page, and the triage layer routes 10 of those 40 to the queue, the page reduction is 25%, not 80%. The arithmetic is `(40 - 30) / 40`. Any larger reduction requires the baseline to contain far more auto-resolvable pages.
- If the LLM call costs $0.0002 per classification and the layer classifies 2,000 alerts per week, the weekly cost is `2000 * 0.0002 = $0.40`. That figure is illustrative and depends entirely on the provider's pricing and the prompt length; substitute your own.

The honest framing is that the achievable reduction depends on how much of your alert volume is genuinely auto-resolvable. Measure that first, then decide whether the layer is worth operating.

## Common questions

**Does this replace Alertmanager?**
No. Alertmanager still does grouping, inhibition, silencing, and routing. The triage layer sits downstream and only decides page versus queue.

**Can the LLM run locally?**
Yes. Any HTTP endpoint that accepts a chat-completions-shaped request works. A locally served model removes the external dependency but adds GPU or CPU capacity requirements and a new failure mode: the model server can be unavailable. The fail-open rule covers it either way.

**What if we are not on Kubernetes?**
The logic is transport-agnostic. Replace the Alertmanager webhook with whatever your monitoring stack emits, and replace the vector store with any indexed store that supports time-windowed lookups. The fingerprinting and decision rules are unchanged.

**Why not just use Alertmanager's existing grouping?**
Grouping reduces the number of notifications for related alerts. Triage decides whether a group should notify at all. They are complementary, and the triage layer assumes grouping has already happened.

**How do we handle secrets?**
Mount them as environment variables from a Kubernetes Secret, and never bake them into the image. The service reads `PD_ROUTING_KEY`, `LLM_KEY`, and the endpoint URLs from the environment.

## Do this in the next 30 minutes

Open your monitoring UI and run a query that counts alerts by name over the last seven days, for example `sum by (alertname) (increase(alertmanager_alerts_received_total[7d]))`. Sort descending and look at the top five. For each one, answer a single question: if this alert fires at 3 a.m., does a human need to act before morning? Every alert where the answer is no is a candidate for the known-bad list or the queue path. Write those five names into a `bad_alerts.json` and load it into the service. That change alone is measurable, reversible, and requires no model tuning.
