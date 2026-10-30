# AI sales cycles: why dev tools now need 3 demos

Developer tool sales used to assume a patient buyer and a clean environment. Neither holds anymore. The people signing purchase orders are often platform leads, CTOs, and procurement staff who have evaluated many AI-adjacent tools in a short period. They have learned to distrust slides, recorded walkthroughs, and generic sandboxes. What they want is evidence produced inside their own infrastructure, on their own traffic, with their own network conditions.

That shift has a concrete consequence for how a tool is demonstrated. A single polished demo no longer carries a deal. A sequence of three live sessions — each answering a different objection — tends to survive procurement, because by the third session the buyer has already reproduced the result themselves.

## Why the old funnel broke

Historically, developer tools sold through blog posts, conference talks, webinars, and feature checklists. That worked when the buyer was an engineer evaluating a library on a laptop. The evaluation was cheap, reversible, and technical.

The current buyer is different in three ways:

- They are not the primary user. A CTO or platform lead may never run the tool, but they own the risk of adopting it.
- They have a procurement process. A purchase order needs a justification that survives review, which means a documented result, not a verbal claim.
- They have been burned. Many teams have adopted tools that worked in a vendor demo and failed against their own VPC, TLS configuration, or mobile network.

The failure mode is predictable. A recorded walkthrough is sent, the technical contact is impressed, and the deal stalls in procurement because nobody can point to the tool doing something useful against the buyer's own systems. Weeks pass. The budget cycle moves on.

A second failure mode is the generic sandbox. A hosted trial with a mock API and a sample dashboard reduces friction for individual developers, but it does not resemble the buyer's architecture. When the buyer tries to connect it to their own cluster, configuration drift appears: overlapping CIDR ranges, missing IAM roles, an egress proxy that blocks the sidecar. The trial collapses at exactly the moment trust was supposed to be established.

A third failure mode is the live screen-share. It works until the buyer's staging environment sits behind a VPN or a private network. Then a thirty-minute demo becomes a two-hour support call, and the buyer concludes the tool is fragile.

All three failures share a root cause: the demo assumed the buyer would adapt to the vendor's environment. The alternative is to run the demo inside the buyer's environment from the start.

## The three-demo structure

The structure below is not a marketing sequence. Each session has a distinct technical purpose, and a deal that reaches the third session has usually already been decided on evidence.

### Demo 1: Confirm the problem exists

The first session is diagnostic, not promotional. The goal is to show the buyer something about their own system that they did not already know.

Practically, this means deploying an observation-only agent into a staging cluster with a single command and letting it collect for a defined window — commonly 24 to 48 hours. The agent should be read-only with respect to application traffic. It records timings per hop, DNS resolution latency, TLS handshake duration, and connection anomalies.

What to instrument:

- Per-hop request duration, broken down by DNS, TCP connect, TLS handshake, time to first byte, and body transfer.
- Retransmission and timeout counters at the socket level.
- Resolver identity and response time, since resolver churn is a common source of intermittent latency on mobile networks.
- The negotiated TLS version and cipher suite per connection.

What to compare: the same metric before and after the agent is present, over a comparable traffic window. If the agent is observation-only, the two should be statistically indistinguishable. That comparison is itself useful, because it establishes that the tool does not perturb the system it measures.

The deliverable is a written finding, not a dashboard link. A short document that says "during this window, p95 time-to-first-byte was X, and Y percent of that was DNS resolution against a resolver that changed Z times" is something a procurement reviewer can read.

### Demo 2: Reproduce a specific failure

The second session targets a known or suspected problem the buyer already cares about. This is where the tool stops being an observer and starts being an instrument.

The buyer supplies a workload or a reproduction case. The tool runs against it and produces a trace that localizes the fault. The important property is falsifiability: the buyer should be able to say in advance what result would disprove the tool's usefulness.

A worked example makes the reasoning concrete. Suppose a staging API shows a p95 latency of 320 ms and the buyer believes the database is slow. The trace shows the breakdown as follows:

- DNS resolution: 120 ms
- TCP connect: 15 ms
- TLS handshake: 85 ms
- Time to first byte: 80 ms
- Body transfer: 20 ms

The database is not in the critical path at all. DNS and TLS together account for 205 ms, roughly 64 percent of the p95. That is arithmetic from the stated breakdown, and it redirects the investigation entirely. The next step is to check resolver configuration and session resumption, not query plans.

This is the moment a demo becomes evidence. The buyer did not learn that the tool is fast or well-designed. They learned something true about their own system that they can act on.

### Demo 3: Prove it works under their constraints

The third session runs under the conditions that actually matter to the buyer: their network path, their TLS policy, their egress rules, their traffic shape. This is the session that most often fails for competitors, and it is the one procurement remembers.

For a tool aimed at mobile-first or intermittent-connectivity environments, the constraints to exercise include:

- Path MTU along the real route, since fragmentation causes retransmissions that look like application latency.
- Resolver behavior when the egress IP changes, which is common on shared mobile data pools.
- Address family mismatch, where the service listens on IPv6 but test clients arrive over IPv4.
- TLS version and cipher negotiation against a load balancer that may not support the newest protocol.
- Burst behavior, since a buffer sized for steady traffic will drop metrics during a spike.

Each of these is a real failure class, and each has a standard mitigation. Documenting the mitigation in advance — an MTU override, a preferred resolver with a bounded timeout, dual-stack egress selection, a configurable minimum TLS version, an adaptive buffer — turns a potential deal-killer into a configuration note.

## Implementation notes

The deployment mechanism should be boring. A sidecar or an agent plus a mutating admission webhook is the common Kubernetes pattern: the operator watches for pods matching a label selector and injects the agent container, with configuration supplied through a custom resource.

```yaml
apiVersion: gateway.example.io/v1alpha1
kind: GatewayConfig
metadata:
  name: prod-api
spec:
  sidecar:
    image: registry.example.io/gateway:1.4.7
    resources:
      requests:
        cpu: "50m"
        memory: "128Mi"
      limits:
        cpu: "200m"
        memory: "512Mi"
  workloadSelector:
    matchLabels:
      app: api-server
```

For non-Kubernetes environments, a Compose file is enough to reproduce the topology locally:

```yaml
services:
  api-server:
    image: ${API_IMAGE:-ghcr.io/acme/api:latest}
    ports:
      - "8080:8080"
    labels:
      gateway-sidecar: "true"

  gateway:
    image: registry.example.io/gateway:1.4.7
    environment:
      - GATEWAY_CONFIG=/config/config.yaml
    volumes:
      - ./config:/config
    ports:
      - "9090:9090"
    depends_on:
      - api-server
```

Two design rules matter more than the specific tooling:

1. The agent must be removable with a single command and must not modify application state. A buyer who cannot cleanly uninstall the trial will not install it.
2. Configuration must be declarative and version-controlled. If reproducing the demo requires a support engineer, the demo is not self-service.

## Integrating with the buyer's existing stack

A self-service session still has to fit the buyer's workflow. Three integrations cover most cases.

**Infrastructure as code.** A Terraform module that deploys the agent into a managed Kubernetes cluster, handles IAM roles, and respects existing subnet and VPC configuration removes the most common source of setup failure. The module should be idempotent and should fail with a clear message when it detects a CIDR conflict rather than silently creating an overlapping range.

```hcl
module "gateway" {
  source = "github.com/example/terraform//modules/eks?ref=v2.3.1"

  cluster_name    = var.cluster_name
  region          = var.region
  vpc_id          = data.aws_vpc.selected.id
  subnet_ids      = data.aws_subnets.private.ids
  workload_labels = { app = "api-server" }
  sidecar_resources = {
    requests = { cpu = "50m", memory = "128Mi" }
    limits   = { cpu = "200m", memory = "512Mi" }
  }
}
```

**Issue tracking.** When the agent detects an anomaly, it should open a ticket in the buyer's tracker with the trace attached, rather than sending an email. A workflow that creates an issue from a webhook payload keeps the finding in the system the buyer already reviews.

```yaml
name: Open Latency Issue
on:
  workflow_dispatch:
    inputs:
      spike_id:
        required: true
      url:
        required: true

jobs:
  open_issue:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/github-script@v7
        with:
          script: |
            const { data: issue } = await github.rest.issues.create({
              owner: context.repo.owner,
              repo: context.repo.repo,
              title: `Latency spike detected: ${context.payload.inputs.spike_id}`,
              body: `Spike detected at ${context.payload.inputs.url}`,
              labels: ['latency', 'debug-session']
            });
            core.setOutput('issue_number', issue.number);
```

**Dashboards.** A dashboard deployed into the buyer's cluster, reading from their existing metrics backend, avoids the "another login" problem. A minimal panel that plots request duration over time is usually enough to start.

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: gateway-dashboard
data:
  dashboard.json: |-
    {
      "title": "Gateway - Latency Breakdown",
      "panels": [
        {
          "title": "HTTP Request Duration",
          "type": "timeseries",
          "targets": [
            {
              "expr": "rate(http_request_duration_seconds_sum[5m]) / rate(http_request_duration_seconds_count[5m])",
              "legendFormat": "mean"
            }
          ]
        }
      ]
    }
```

## How to measure whether the process works

Vendor-side claims about conversion rates are not useful to a reader, so the honest approach is to describe what to instrument and let each team produce its own numbers.

Track these per opportunity:

- Time from first contact to first successful deployment of the agent in the buyer's environment.
- Time from first deployment to a written finding.
- Number of support interactions required per session, which is the real measure of self-service.
- Whether the deal advanced after each session, and the stated reason if it did not.

On the infrastructure side, the costs are ordinary cloud costs and can be estimated from the buyer's own pricing: two small nodes for 48 hours, a metrics backend, and a small amount of CI time. Compute the figure from the buyer's region and instance types rather than quoting a vendor number.

For the technical result, compare the same metric over comparable windows before and after any change. A useful finding has three parts: the metric, the window, and the mechanism. "p95 time-to-first-byte fell from 320 ms to 210 ms over a 48-hour window after DNS caching was enabled" is a finding. "Latency improved by 34 percent" is a marketing claim.

## Decision checklist

Before committing to a three-demo sales process, confirm the following:

- The agent can be installed and removed with one command each, with no persistent state left behind.
- Configuration is declarative and can be reviewed in version control.
- The agent is observation-only by default; any traffic-affecting behavior is opt-in and documented.
- The tool degrades gracefully when the network path is unusual — high MTU, resolver churn, address family mismatch, older TLS.
- Findings are exported to the buyer's existing tools rather than a vendor-hosted portal.
- A session can run to completion without a vendor engineer present.
- Every claim in the final report is traceable to a metric, a window, and a mechanism.

If any of these is false, the third demo will fail, and it will fail in front of the person who controls the budget.

## The 30-minute action

Pick one staging service you can reach without a VPN. Deploy your agent or sidecar against it with a single command, let it run for one hour, then write down the per-hop latency breakdown for that hour — DNS, TCP connect, TLS handshake, time to first byte, body transfer — as a single table with the window stated. That table is the artifact your first demo produces, and it is the only part of the sales conversation that a skeptical CTO cannot dismiss.
