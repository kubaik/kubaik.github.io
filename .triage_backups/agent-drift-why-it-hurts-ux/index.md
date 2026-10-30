# Agent drift: why it hurts UX

A fleet of background agents — telemetry collectors, security scanners, feature-flag updaters — can silently diverge from the baseline build. The user-visible symptom is usually a latency or error-rate regression that correlates with a release whose notes show no breaking changes. The gap between "the pipeline reported success" and "the incident report says otherwise" is where drift lives.

## What drift looks like from the application layer

A representative log line:

```
[ERROR] AgentHealthCheckFailed: expected version 2.4.1, found 2.3.8 – aborting request
```

The message points at a version mismatch, but the underlying cause is frequently a configuration drift that disables a performance-critical path — a local metrics buffer, a connection pool, a batch processor. Two properties make this confusing:

- The error surfaces in the application layer while the root cause lives in the agent's host environment.
- The drift is often intermittent, affecting only nodes that received a delayed OS patch or that missed a rollout window.

A typical failure mode: a Kubernetes DaemonSet updates an OpenTelemetry Collector across most nodes, but a few nodes are stuck in `CrashLoopBackOff` during the rollout window and keep the old image. Those nodes report higher round-trip times, which drags the fleet-wide p99 upward. The application team sees a performance regression; the ops team sees a harmless version flag.

## The four classes of drift

Drift is rarely a single event. It is a cascade of small mismatches, and knowing the class narrows the search space immediately.

1. **Binary version drift** — different agent binaries across hosts, from staggered rollouts, manual hot-fixes, or nodes that skipped an update.
2. **Configuration drift** — YAML or JSON config edited in place, diverging from the source of truth in Git.
3. **Runtime dependency drift** — underlying libraries (for example, an OpenSSL minor-version bump delivered by an OS patch) changing TLS or DNS behavior under a binary that never changed.
4. **Environment drift** — environment variables, IAM role permissions, container runtime flags, or kernel versions that differ from the reference host.

The reason performance degrades is that one or more of these disables a tuned path. If a local metrics buffer falls back to a small default instead of a tuned size, the agent emits far more HTTP requests per second than intended, which increases both network contention and egress cost. The surface symptom — higher latency — is downstream of that.

### Worked example: buffer size and request amplification

Suppose a tuned agent flushes a 256 KB batch buffer and a drifted agent falls back to a 10 KB default. Holding event size constant, the drifted agent needs 256 KB / 10 KB = 25.6× as many flush requests to move the same volume of data. If the baseline is 40 requests/second, the drifted fleet issues roughly 1,024 requests/second. That 25× amplification is what shows up as network contention and, indirectly, as latency. The arithmetic is illustrative — substitute your own buffer sizes and flush rate — but the mechanism is the point: a config value that looks cosmetic can multiply request volume by an order of magnitude.

## Diagnosis workflow

Work the classes in order. Each step is cheap and eliminates a whole category.

1. **Enumerate agent versions across the fleet.** Query the orchestrator API rather than trusting the pipeline's success report.
2. **Checksum live config against the Git baseline.** Any file that differs is configuration drift by definition.
3. **Record library versions on a sample of hosts.** Compare OpenSSL, libc, and container runtime versions between a known-good host and a suspect host.
4. **Diff environment and IAM state.** Launch templates, environment variables, and role policies are common hiding places because they live outside the repo.

### Step 1: enumerate agent versions

```bash
# List agent image per pod
kubectl get pods -n monitoring -l app=otel-collector -o json \
  | jq -r '.items[] | "\(.metadata.name) \(.status.containerStatuses[0].image)"'
```

```python
# Verify version via an HTTP health endpoint
import requests

nodes = ['node-a', 'node-b', 'node-c']
expected = '2.4.1'
for n in nodes:
    try:
        r = requests.get(f'http://{n}:55679/healthz', timeout=2)
    except requests.RequestException as exc:
        print(f'{n}: unreachable ({exc})')
        continue
    if r.ok and expected in r.text:
        print(f'{n}: OK')
    else:
        print(f'{n}: version drift detected')
```

Note the `timeout` and the exception handling: a health check that hangs is itself a signal, and an unhandled exception would abort the audit halfway through.

### Step 2: checksum config against baseline

```bash
# Compare live config hash to the value stored at deploy time
live=$(sha256sum /etc/otel-collector-config.yaml | awk '{print $1}')
baseline=$(aws s3 cp s3://config-baselines/otel-collector.sha256 - | awk '{print $1}')
if [ "$live" != "$baseline" ]; then
  echo "config drift on $(hostname): live=$live baseline=$baseline"
fi
```

Run this from a cron job or a DaemonSet sidecar and export the result as a metric. A boolean `config_matches_baseline` is enough to alert on.

### Step 3: compare runtime libraries

```bash
# Capture library versions for comparison across hosts
openssl version
ldd --version | head -1
containerd --version
uname -r
```

The point is not to pin every library forever; it is to know which hosts differ so that when a TLS or DNS symptom appears, you can correlate it with a library change instead of guessing.

## Failure-mode analysis

**Symptom:** version-mismatch errors in logs plus a p99 latency spike, while the pipeline reports success.
**Likely cause:** a subset of nodes skipped the rollout because they were `NotReady` or crash-looping during the window.
**Fix:** force a rolling restart of the DaemonSet, then add a readiness probe that checks the agent's health endpoint for the expected version string. Without the probe, the next rollout will fail the same way.

**Symptom:** latency spikes persist after all agents report the correct binary version, with occasional `TLS handshake failed` warnings.
**Likely cause:** runtime dependency drift. An OS patch upgraded a TLS library, breaking a custom cipher suite in the agent's TLS config, and the agent silently fell back to a slower client.
**Fix:** pin the library version in the base image and use a cipher suite that tolerates minor upgrades.

```dockerfile
# Pin the TLS library version in the base image
FROM amazonlinux:2023
RUN yum install -y openssl-1.1.1k && \
    yum clean all
COPY otel-collector-config.yaml /etc/otel-collector-config.yaml
```

**Symptom:** intermittent health-check failures only on instances from one Auto Scaling Group.
**Likely cause:** environment drift. The launch template points at a stale endpoint that still resolves but returns errors, so the agent aborts processing. The CI pipeline never sees it because the value lives in the template, not the repo.
**Fix:** read the endpoint from a parameter store at boot rather than hardcoding it in the template.

```yaml
# Read the endpoint from SSM at instance boot
Resources:
  OtelCollectorLaunchTemplate:
    Type: AWS::EC2::LaunchTemplate
    Properties:
      LaunchTemplateData:
        UserData: !Base64 |
          #!/bin/bash
          export OTEL_EXPORTER_OTLP_ENDPOINT=$(aws ssm get-parameter \
            --name /prod/otel/endpoint --query Parameter.Value --output text)
```

## How to verify a fix actually worked

Verification should be quantitative and reproducible. Pick the metrics your stack already exposes; the names below are placeholders.

1. **Metric check.** Query the version-mismatch counter and the p99 latency for the last 15 minutes. The mismatch counter should be zero and p99 should return to the pre-drift baseline. Compare against a baseline window, not an absolute number.

```bash
# Query Prometheus for the mismatch counter
curl -s "http://prometheus.local/api/v1/query?query=agent_version_mismatch_total" \
  | jq '.data.result[]?.value[1]'
```

2. **Log audit.** Count occurrences of the exact error string, grouped by host, over the same window. A nonzero count on a single host points at a host that did not restart.

```
fields @message
| filter @message like /AgentHealthCheckFailed/
| stats count() by host
```

3. **Synthetic traffic.** Run a short load test against a representative endpoint and record the 95th percentile. Compare it to a baseline captured before the fix. Without a stored baseline, this step proves nothing.

If any check still shows anomalies, revisit the earlier steps — overlapping drift sources are common, and fixing the binary version will not fix a stale endpoint.

## Detection methods compared

| Method | Catches | Misses | Cost |
|---|---|---|---|
| Heartbeat version check | Binary version drift | Config, library, and environment drift | Low |
| Config checksum diff | Any config file change | Binary and library drift | Low; needs a stored baseline hash |
| Immutable infrastructure | Manual edits entirely | Drift introduced by the image build itself | Higher operational cost |
| CI drift-detection step | Version mismatches before deploy | Runtime state after deploy | Custom scripts to maintain |

No single row covers all four drift classes. Most teams need a version check plus a config checksum at minimum, because those two are cheap and cover the most common causes.

## Prevention checklist

- **Automate version consistency.** Add a CI job that reads the agent image tag from the deployment manifest and asserts it matches the version declared in the repo.
- **Store config hashes.** After each successful deploy, compute a hash of the agent config and store it. Compare live hashes against it on a schedule.
- **Enforce immutability where practical.** Use infrastructure-as-code drift detection on launch templates and IAM roles so that out-of-band edits surface as failures rather than surprises.
- **Alert on anomalies, not absolutes.** A version-mismatch counter greater than zero is a clean signal. A latency threshold works better as a percentage deviation from a moving average than as a fixed millisecond value.
- **Record host-level context.** Kernel version, container runtime version, and libc flavor belong in host telemetry. When a symptom appears, the correlation is otherwise invisible.

## Edge cases worth knowing about

**DNS resolution differences by libc.** Agents built on Alpine use musl, whose resolver caches differently from glibc and handles `NDOTS` configuration less flexibly. A service endpoint that changes IP while an agent process is running can leave that process resolving a stale address indefinitely, even though `dig` from inside the pod resolves correctly. The symptom is sporadic `connection refused` or name-resolution errors from a subset of healthy-looking agents. Mitigation: keep DNS cache TTLs low in the cluster resolver and restart agent pods after endpoint migrations.

**Kernel-level CPU throttling.** An agent can be correctly versioned and configured and still starve for CPU if the host kernel throttles low-priority cgroup workloads aggressively while `kubelet` and other system processes are busy. Latency rises, queue-full counters climb, and nothing in the agent's own configuration explains it. Mitigation: include kernel version and CPU throttling metrics in host telemetry, and keep worker node images uniform.

**File-handle leaks in the container runtime.** A runtime bug can leak handles for the stdout/stderr pipes of exited containers. Over time the agent's inotify watcher hits a limit relative to watched inodes even though the system-wide `ulimit` looks fine. The agent appears to stop processing logs with no errors at all. Mitigation: track open inotify watches and file descriptors as a time series, not just as a point-in-time check, and keep the runtime patched.

Each of these presents as "agent drift" but is actually drift in a layer beneath the agent. That is why host-level context in telemetry matters as much as agent-level metrics.

## Escalation path

If the three fix categories above do not resolve the symptom:

1. **On-call engineer.** Run the verification checks on a fresh host and capture raw agent logs. Compare against a host that is behaving correctly.
2. **Platform team.** Open a ticket with the service name, severity, observed latency, error count over a fixed window, and the config hash diff.
3. **Vendor support.** If the agent is a third-party binary, provide the exact error string, OS and kernel version, container runtime version, and a minimal reproduction using the commands above.
4. **Post-mortem.** Focus on which drift class was undetected and which check would have caught it. Then add that check.

## FAQ

**How do you detect agent drift in Kubernetes?**
Run a DaemonSet that periodically calls the agent's health endpoint and exports the reported version as a Prometheus metric. Pair it with a config checksum compared against a stored baseline, because a version check alone will not catch config or library drift.

**Why does agent drift increase latency?**
Drift often disables a performance optimization — batch processing, TLS session reuse, a tuned buffer — which raises per-request overhead. Because the effect is per-request, it compounds across the fleet and shows up as a p99 regression rather than a uniform slowdown.

**What automates drift detection?**
Infrastructure-as-code drift detection for launch templates and IAM roles, a scheduled job that hashes config files and compares against a baseline, and a CI step that asserts declared versions match deployed versions. The specific tools matter less than having all three layers covered.

**When should agent images be rebuilt?**
Whenever a base image receives a security patch that changes a runtime library, or when the agent's own dependencies change. Automate it with dependency-update pull requests that trigger a full rollout, so library drift never accumulates silently.

**How do you prevent config drift without manual checks?**
Keep the canonical config in Git, render it into a ConfigMap at deploy time, mount it read-only, and use an admission controller to reject pods that try to override it.

## Take action in the next 30 minutes

Run a version audit against one production namespace and record what you find. Adapt this to your orchestrator and label selector:

```bash
kubectl get pods -n monitoring -l app=otel-collector -o json \
  | jq -r '.items[] | "\(.metadata.name) \(.status.containerStatuses[0].image)"' \
  | sort -k2 \
  | tee /tmp/agent-versions.txt
```

Then count distinct images:

```bash
awk '{print $2}' /tmp/agent-versions.txt | sort -u | wc -l
```

If that number is greater than one, you have binary version drift right now, and every other detection step in this article is worth setting up before the next rollout.
