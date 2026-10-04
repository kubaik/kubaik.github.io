# Zero-trust AI: skip the latency tax

## Why standard zero-trust playbooks strain under inference traffic

Zero-trust networking for AI services is often described as a solved problem. The pitch is clean: mutual TLS everywhere, short-lived certificates, per-request authorization, no implicit trust based on network location. Then a model goes behind it and p99 latency climbs from roughly 180 ms to 640 ms. Security review passes. Users notice.

The gap is not conceptual. Zero-trust is a reasonable default for anything that touches user data, model weights, or inference logs. The gap is that most zero-trust reference architectures were written for request/response APIs where a 20 ms handshake is noise. Inference services differ in three ways that break the standard playbook:

1. **Connection churn is higher.** Inference workloads scale up and down aggressively. A GPU-backed pod that handles 40 requests/second at peak may handle zero for six hours. Every cold start re-establishes mTLS, re-fetches tokens, and re-validates policy. That handshake cost is amortized in a steady-state web app and exposed in a bursty inference service.
2. **Payloads are large and serialized.** A single request can carry a 2 MB prompt with retrieved context. If a sidecar performs full-body inspection or re-encryption, it pays for bytes that did not need inspecting.
3. **The trust boundary is not where it appears.** A model server often calls a vector database, a feature store, and an external tool API. Each hop is a new identity, a new token, a new policy decision. A naive implementation adds multiple round trips before the first token is generated.

The part that trips people up is that zero-trust does not have to mean zero-trust-everywhere. It is possible to keep the security properties — no implicit network trust, per-request identity, least privilege — while moving the expensive parts out of the hot path.

A common failure mode: a team enables mTLS on every internal hop, deploys a service mesh sidecar, and then finds time-to-first-token (TTFT) has doubled. The mesh is not wrong. The mistake is treating every hop as equally sensitive and paying the full handshake cost on each one.

## Separating identity from enforcement

The core idea is to separate **identity** from **enforcement**. Identity is cheap if it is cached. Enforcement is expensive if applied per byte.

In a standard zero-trust setup, every request carries a signed identity (a JWT or a SPIFFE SVID), and every service validates that identity against a policy engine. The expensive parts are:

- **Cryptographic verification** of the token signature. RSA-2048 verify is roughly 0.1–0.3 ms per call on a modern x86 core; ECDSA P-256 is closer to 0.05 ms. These are order-of-magnitude figures that depend on CPU and library; measure on your own hardware before designing around them.
- **Policy evaluation** against a remote engine such as Open Policy Agent (OPA) or Cedar. A network round trip to a remote policy service is typically 2–8 ms. A local evaluation is typically 0.1–0.5 ms.
- **mTLS handshake** on a new connection. A full TLS 1.3 handshake with client certificates is typically 8–20 ms depending on cipher suite and CPU. Session resumption brings that to roughly 1–3 ms.

The strategy is to make the first two local and cacheable, and to make the third rare:

- Use a SPIFFE-compatible identity system to issue short-lived SVIDs. The SVID is a certificate, not a bearer token, so it can be used for mTLS directly. Rotation is automatic and the trust bundle is small.
- Run the policy engine as a sidecar or in-process library, not as a central service. The policy is compiled and loaded locally. Decision latency drops to sub-millisecond. Central policy distribution still happens through a bundle API.
- Enable **TLS session resumption** (session tickets) on internal listeners. This is a one-line config change in most ingress controllers and eliminates most of the handshake cost for repeat connections.
- Keep **connection pools warm** at the inference layer. A pool of 8–16 persistent connections per model server absorbs bursty traffic without re-handshaking.

The security properties are unchanged — every request is authenticated, every call is authorized, no implicit trust — but the per-request overhead drops from tens of milliseconds to low single-digit milliseconds.

A useful mental model: zero-trust is a property of the **decision**, not the **transport**. The decision can be made locally and cheaply as long as its inputs (identity, policy, trust bundle) are fresh and signed.

## Step-by-step implementation

The following builds a minimal setup: a Python inference service (FastAPI on Python 3.12) behind an Envoy sidecar, with a SPIFFE-compatible identity issuer and an in-process policy engine. Version numbers are omitted deliberately; the configuration surface changes between releases, and pinning to a specific minor version in an article ages badly. Verify against your installed version's documentation.

### Step 1: Issue workload identities

The identity issuer runs as a DaemonSet. Each workload gets an SVID mounted at a well-known path. The SVID is a cert/key pair plus a trust bundle.

```yaml
# identity-agent config snippet
agent {
  data_dir = "/run/identity/data"
  socket_path = "/run/identity/sockets/agent.sock"
  trust_bundle_path = "/run/identity/bundle/bundle.crt"
}

workload {
  spiffe_id = "spiffe://example.org/ns/inference/sa/model-server"
  parent_id = "spiffe://example.org/ns/inference/sa/identity-agent"
  selectors = [
    "k8s:ns:inference",
    "k8s:sa:model-server",
  ]
}
```

### Step 2: Terminate mTLS at the sidecar with session resumption

Envoy handles the handshake. The key settings are `require_client_certificate: true` and enabling session tickets.

```yaml
# envoy.yaml (excerpt)
static_resources:
  listeners:
  - name: inference_listener
    address:
      socket_address: { address: 0.0.0.0, port_value: 8443 }
    filter_chains:
    - transport_socket:
        name: envoy.transport_sockets.tls
        typed_config:
          "@type": type.googleapis.com/envoy.extensions.transport_sockets.tls.v3.DownstreamTlsContext
          require_client_certificate: true
          common_tls_context:
            tls_certificates:
            - certificate_chain: { filename: "/certs/server.crt" }
              private_key: { filename: "/certs/server.key" }
            validation_context:
              trusted_ca: { filename: "/certs/bundle.crt" }
          session_ticket_keys:
            keys:
            - filename: "/certs/ticket.key"
```

Session tickets are usually the highest-leverage change. In bursty inference workloads, resumption is commonly reported to cut handshake cost by 70–85% relative to full handshakes; confirm the figure for your traffic mix by comparing handshake counters before and after.

### Step 3: Evaluate policy in-process

Instead of calling a remote policy server, embed the policy as a library. Bindings exist for several languages, including Rust crates callable via FFI and WASM builds. The example below uses a Rust-backed Python binding.

```python
# policy.py
from fastapi import FastAPI, Request, HTTPException
import regorus

app = FastAPI()
engine = regorus.Engine()
engine.add_policy_from_file("policy.rego")
engine.set_rego_v0(True)

@app.middleware("http")
async def authorize(request: Request, call_next):
    identity = request.headers.get("x-spiffe-id")
    if not identity:
        raise HTTPException(status_code=401, detail="missing identity")
    engine.set_input_json({
        "identity": identity,
        "path": request.url.path,
        "method": request.method,
    })
    result = engine.eval_rule("data.authz.allow")
    if not result or not result[0]:
        raise HTTPException(status_code=403, detail="policy denied")
    return await call_next(request)
```

The Rego policy itself is small:

```rego
package authz

default allow = false

allow if {
    input.identity == "spiffe://example.org/ns/inference/sa/model-server"
    input.method == "POST"
    input.path == "/v1/generate"
}
```

This runs in well under a millisecond per request on a modest CPU. No network hop, no serialization, no remote policy server in the hot path.

### Step 4: Keep connections warm

At the client side, use a persistent HTTP/2 connection pool. In Python, `httpx` with `http2=True` and a shared `AsyncClient` does this. Set `limits=httpx.Limits(max_keepalive_connections=16, keepalive_expiry=300)`.

The combination of session resumption, local policy, and warm connections removes the latency tax. None of these weaken the security model. They stop the same decision from being paid for twice.

## How to measure the latency tax on your own system

Published benchmark tables for zero-trust inference overhead are rarely transferable. The numbers depend on CPU, cipher suite, policy complexity, and traffic shape. Measure instead of copying.

What to instrument:

- **TTFT p50 and p99** at the request handler, from request receipt to first generated token. Emit it as a histogram, not an average.
- **Handshake counters** from the sidecar — most sidecars expose `ssl.handshake` and `ssl.session_reused` style counters. The ratio tells you how often resumption is working.
- **Policy evaluation duration** as a histogram inside the service, tagged by decision outcome.
- **Connection pool saturation** — active versus idle connections, and how often a new connection is opened per unit time.

What to compare:

1. Baseline: no mTLS, no policy evaluation.
2. Naive zero-trust: mTLS on every hop, remote policy service, no resumption.
3. Local policy, no resumption.
4. Local policy plus session resumption.
5. Local policy, session resumption, warm connection pool.

Run each configuration against the same load profile — ideally a replay of production traffic or a generator that reproduces your prompt size distribution and burst pattern. Compare p50 and p99 separately. The tail is where zero-trust overhead usually hides, because remote policy calls queue under load and full handshakes compete for CPU with inference.

A useful arithmetic check: a full TLS 1.3 handshake with client certificates consumes roughly 8–20 ms of CPU time. At 100 new connections per second with no resumption, that is 0.8–2.0 CPU-seconds per second spent on handshakes — potentially a full core. Session resumption at 1–3 ms per handshake drops that to 0.1–0.3 CPU-seconds per second. These are illustrative figures derived from the stated per-handshake range, not measurements; substitute your own handshake cost and connection rate.

## Failure modes to plan for

**1. Certificate rotation storms.** Short-lived SVIDs rotate frequently — hourly is a common default. If the service caches the certificate in memory and does not reload on rotation, validation failures appear at exactly the rotation boundary. The fix is to watch the SVID file and reload on change, or use a library that does it. A common symptom is a burst of 401s at a regular interval that resolves on its own.

**2. Session ticket key rotation without coordination.** If the ticket key is rotated on one sidecar but not others, clients that resume against the wrong pod get a full handshake. This shows up as intermittent p99 spikes correlated with deploys. Rotate ticket keys in lockstep, or share a key across the fleet.

**3. Policy bundle staleness.** If policies are distributed via a bundle API and the bundle fails to load, the engine may fall back to a default decision. Depending on configuration, that default can be `allow`. Always set `default allow = false` and alert on bundle load failures. The failure mode is silent: requests succeed but are not being authorized.

**4. mTLS between sidecar and model server.** If the sidecar terminates mTLS but talks plaintext to the model server on localhost, there is a trust gap. The standard mitigation is to run the model server in the same pod and treat the pod as the trust boundary. If that is not possible, run a second mTLS hop but keep it on the loopback interface where handshake cost is lower.

**5. Clock skew.** Short-lived certificates and JWTs are sensitive to clock skew. A 30-second skew between nodes is enough to cause intermittent validation failures. Run NTP or chrony, and set validation leeway to 60 seconds. This is a boring failure mode that costs teams days.

**6. The vector database hop.** If the inference service calls a vector database, that hop needs its own identity and policy. Teams often forget it and end up with a zero-trust front door and a wide-open back door. Audit every outbound call.

## Choosing between a mesh and direct identity

| Approach | Identity source | mTLS handled by | Operational cost | Best fit |
|---|---|---|---|---|
| Service mesh (e.g. Linkerd, Istio) | Mesh-issued workload identity | Mesh micro-proxy or sidecar | Lower; mesh manages rotation and trust | Single Kubernetes cluster, teams wanting mTLS with minimal config |
| Identity issuer + Envoy directly | SPIFFE SVIDs from a dedicated issuer | Envoy sidecar with explicit TLS config | Higher; more moving parts to run and upgrade | Identities needed outside Kubernetes — bare metal, multi-cloud, on-prem |
| Application-level tokens | Signed JWTs | Application library | Lowest infrastructure, highest code surface | Single service, single client, or when mTLS is impractical |

If the entire system is one cluster, a service mesh is usually less work. The reason to reach for a dedicated identity issuer plus Envoy is when SPIFFE identities must work outside Kubernetes.

## When this approach is the wrong choice

Zero-trust adds operational complexity. It is the wrong choice when:

- **There is a single service and a single client.** If the entire system is one API and one frontend, mTLS between them is ceremony. Use a shared secret over TLS and move on.
- **The latency budget is already tight and the threat model is internal.** If all traffic is inside one VPC and every workload is controlled, network-level controls plus IAM may be sufficient. Zero-trust matters more with multi-tenant or multi-cluster traffic.
- **There is no capacity to operate an identity issuer.** It is not hard, but it is another stateful system to run, back up, and upgrade. A service mesh with built-in mTLS is often a better trade.
- **Inference is batch, not interactive.** For nightly batch inference, a 20 ms handshake is irrelevant. Optimize for throughput, not tail latency.

The honest framing: zero-trust is a spectrum. The question is not "zero-trust or not" but "which trust decisions must be made per request, and which can be made per connection or per workload?"

## A decision checklist

Before adding per-request enforcement to an inference path, answer these:

1. Which hops carry user data, model weights, or inference logs, and which carry only derived or non-sensitive metadata?
2. For each hop, is a per-request decision required, or is a per-connection decision sufficient?
3. Where does policy evaluation happen today — remote service, sidecar, or in-process? What is its measured p99 contribution?
4. Is session resumption enabled on every internal listener? What is the observed reuse ratio?
5. Are connection pools warm enough to absorb the burst pattern, or does traffic arrive as cold connections?
6. Which outbound calls (vector DB, feature store, tool APIs) carry identity, and which are currently unauthenticated?
7. What happens when the policy bundle fails to load? Is the default deny?
8. Who owns certificate and ticket-key rotation, and is it coordinated across the fleet?

## Frequently Asked Questions

**How much latency does mTLS add to an AI inference service?**
A full TLS 1.3 handshake with client certificates costs roughly 8–20 ms of CPU time depending on cipher suite and hardware. With session resumption, that drops to roughly 1–3 ms. The key is connection reuse; if every request opens a new connection, the full cost is paid every time. Measure the reuse ratio from sidecar counters rather than assuming.

**Can zero-trust networking be implemented without a service mesh?**
Yes. A dedicated identity issuer plus Envoy gives SPIFFE identities and mTLS without a full mesh. The tradeoff is more configuration surface and more moving parts to operate. A mesh handles identity and mTLS with less config but typically ties you to Kubernetes. If identities are needed outside Kubernetes, the direct approach is the better fit.

**Why does p99 spike at a regular interval with zero-trust?**
The most common cause is certificate rotation. Short-lived SVIDs rotate on a schedule, often hourly. If the service caches the certificate and does not reload on rotation, validation failures appear at the rotation boundary. The fix is to watch the SVID file and reload on change. A secondary cause is session ticket key rotation without fleet-wide coordination.

**Is a policy engine fast enough for per-request authorization?**
Yes, if it runs in-process. A local evaluation of a simple policy typically takes 0.1–0.5 ms. A remote call adds 2–8 ms of network round trip plus queueing under load. For AI services where p99 matters, local evaluation is the right default. Use a bundle API to distribute policies, but evaluate locally.

**What is the most common mistake with zero-trust for AI?**
Forgetting the outbound hops. Teams secure the front door with mTLS and policy, then let the model server call the vector database, feature store, and external tool APIs without identity. That is edge-only trust, not zero-trust. Audit every outbound call and give it its own identity and policy.

## What to do in the next 30 minutes

Open the handler in your inference service that receives the request before it reaches the model, and add a single histogram metric recording time from request receipt to first generated token. Deploy it to staging, run a five-minute load test at expected peak RPS, and look at the p99. If the gap between p50 and p99 exceeds 100 ms, the zero-trust overhead is in the tail. The first thing to check is whether policy evaluation is remote or local — that single measurement tells you which of the four optimizations above to apply first.
