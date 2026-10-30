# Sandboxed computer-use agents with gVisor

Computer-use agents — models that drive a real browser or desktop session through screenshots, clicks, and keystrokes — are among the most dangerous components a team can deploy. Not because the models are malicious, but because the permission model around them is usually an afterthought. A typical setup gives the agent a headless browser, a service account with broad IAM rights, and a filesystem it can write to. That combination means a single prompt-injection payload on a page the agent visits can turn into an outbound request that reads from an object store or posts to an internal admin endpoint.

The failure mode is not hypothetical. The OWASP Top 10 for LLM Applications lists prompt injection as LLM01, and computer-use agents are the clearest case where injection crosses from "the model says something wrong" into "the model does something wrong on a real system." Vendors shipping computer-use features commonly warn that the agent should run in a container with no access to sensitive data — advice that production rollouts often quietly ignore because the agent needs credentials to be useful.

The part that trips teams up is that the obvious mitigations (a system prompt saying "don't visit untrusted sites," a regex filter on tool calls) don't hold under adversarial input, and the mitigations that do hold — kernel-level isolation, network egress allowlists, short-lived scoped credentials — require rebuilding the deployment shape rather than tuning the prompt. This article covers the architecture that works, the numbers to plan against, and the specific places teams get burned.

## The approaches that commonly fail, and why

Three patterns show up repeatedly in teams that get this wrong. All three look reasonable in a design doc and fail in production.

**Prompt-level guardrails.** The agent gets a system prompt like "only interact with domains on this allowlist." This fails because the model is processing untrusted text from the page it's driving. A page can contain text that reads as an instruction, and the model has no reliable way to distinguish "content I was asked to read" from "instruction I should follow." Simon Willison's framing of the lethal trifecta — private data access, untrusted content, and external communication — is the clearest summary: if an agent has all three, exfiltration is a matter of when, not if. Prompt filtering reduces the rate of accidental failures but does nothing against a determined page.

**Tool-call validation in the orchestrator.** This is better. The orchestrator inspects each tool call before executing it — block `curl`, block writes outside `/tmp`, block navigation to non-allowlisted hosts. The problem is that computer-use agents don't emit clean tool calls. They emit `click(x=430, y=612)` and `type("...")`. The semantic meaning of a click depends on what's rendered at those coordinates. A validation layer that tries to infer intent from pixel coordinates is a research project, not a deployment.

**Full VM per session with broad credentials.** Teams that do isolate correctly often over-provision the credentials because the agent needs to log into a real app. A common shape is a service account with `s3:*` on the bucket the agent's screenshots land in, plus a database read role, plus a session token that lives for 24 hours. When the agent is compromised, the blast radius is everything that token can reach for the next 24 hours. The isolation was real; the credential scope undid it.

| Approach | Injection resistance | Ops cost | Latency added | Common failure |
|---|---|---|---|---|
| Prompt guardrails only | Low | Very low | 0 ms | Page text overrides system prompt |
| Tool-call validation | Medium | Low | 5–20 ms | Coordinates have no semantic meaning |
| VM per session, broad creds | High on isolation | High | 800–1500 ms cold start | 24h token becomes the blast radius |
| Sandbox + egress allowlist + short creds | High | Medium | 120–300 ms warm | Complexity in the allowlist itself |

The pattern across all three failures is the same: teams treat the agent as a trusted component and try to make its inputs trustworthy. The approach that works inverts that — treat the agent as untrusted code, and make the environment safe regardless of what it decides to do.

## The approach that works in practice

Run the agent inside a sandboxed container with a kernel boundary, restrict its network egress to an explicit allowlist, and give it credentials that expire before the session ends. Three layers, each cheap on its own, and together they contain the failure modes above.

For the kernel boundary, gVisor is the pragmatic choice on Linux. It intercepts syscalls in userspace, so a container escape requires a gVisor bug rather than a kernel bug — and gVisor's syscall surface is much smaller than the host kernel's. The tradeoff is syscall overhead: filesystem-heavy workloads run roughly 2–5x slower under gVisor than under runc, and CPU-bound work is closer to 1.2–1.5x. For a browser agent that spends most of its time waiting on network and rendering, the overhead is usually acceptable. Firecracker microVMs are the alternative when near-native performance is required and a heavier cold start is acceptable (typically 125 ms for the microVM plus whatever guest init costs).

For egress, don't rely on the container's default network. Put the sandbox on a network with a default-deny egress policy and an explicit allowlist of hostnames the agent is permitted to reach. In Kubernetes this is a `NetworkPolicy` plus a DNS-resolving egress gateway; outside Kubernetes it's an iptables rule set or a sidecar proxy. The allowlist should be hostnames, not IPs, because CDN IPs rotate and you'll be chasing them forever.

For credentials, the rule is simple: the token must expire before the session does. If a session is capped at 10 minutes, the token lives 9. Use short-lived STS credentials, not long-lived IAM users. A typical setup issues a session token scoped to a single prefix in a single bucket, valid for 600 seconds, and the orchestrator never sees the long-lived credential.

## Implementation details

Here's a minimal pattern for launching a gVisor-sandboxed agent session on a Linux host with `runsc` installed. This is the orchestration layer, not the agent itself — the point is that the agent code never runs with host privileges.

```python
# sandbox_launcher.py — Python 3.11, runsc 20240617
import subprocess
import time
import boto3
from datetime import timedelta

SESSION_TIMEOUT_SECONDS = 600


def issue_scoped_credentials(session_id: str):
    sts = boto3.client("sts")
    # Assume a role whose trust policy requires an ExternalId tied to session_id.
    resp = sts.assume_role(
        RoleArn="arn:aws:iam::123456789012:role/agent-sandbox-readonly",
        RoleSessionName=f"agent-{session_id}",
        DurationSeconds=SESSION_TIMEOUT_SECONDS - 60,  # expire before session ends
        Policy="""{
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Action": ["s3:GetObject"],
                "Resource": "arn:aws:s3:::agent-artifacts/sessions/%s/*"
            }]
        }""" % session_id,
    )
    return resp["Credentials"]


def launch_sandbox(session_id: str, creds: dict) -> str:
    env = {
        "AWS_ACCESS_KEY_ID": creds["AccessKeyId"],
        "AWS_SECRET_ACCESS_KEY": creds["SecretAccessKey"],
        "AWS_SESSION_TOKEN": creds["SessionToken"],
        "SESSION_ID": session_id,
    }
    # runsc is the gVisor runtime; --network=none forces the agent
    # through the egress proxy sidecar rather than the host network.
    cmd = [
        "runsc", "--rootless", "--network=none",
        "run", "--detach", "--name", f"agent-{session_id}",
        "--env", ",".join(f"{k}={v}" for k, v in env.items()),
        "ghcr.io/example/agent-runtime:1.4.2",
        "/usr/local/bin/agent-entrypoint",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"sandbox launch failed: {result.stderr}")
    return f"agent-{session_id}"
```

The key detail is `--network=none`. With no network namespace access, the agent cannot reach anything except through the sidecar proxy that the orchestrator controls. That proxy is where the allowlist lives.

On the egress side, a small proxy that checks the SNI or the Host header against an allowlist is enough for most browser agents. Here's the shape in Node 20 LTS:

```javascript
// egress_proxy.js — Node 20 LTS, http-proxy 1.18.1
const httpProxy = require('http-proxy');
const http = require('http');

const ALLOWLIST = new Set([
  'example.com',
  'docs.example.com',
  'api.stripe.com',
]);

const proxy = httpProxy.createProxyServer({ changeOrigin: true });

const server = http.createServer((req, res) => {
  const host = (req.headers.host || '').split(':')[0];
  if (!ALLOWLIST.has(host)) {
    res.writeHead(403, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ error: 'egress_denied', host }));
    return;
  }
  proxy.web(req, res, { target: `https://${host}` }, (err) => {
    res.writeHead(502);
    res.end(JSON.stringify({ error: 'upstream_failed', detail: err.message }));
  });
});

server.listen(3128, '127.0.0.1');
```

Two things to watch. First, the allowlist must include the hosts the agent's OAuth flows redirect through, or you'll get 403s on login pages that look like agent bugs. Second, HTTP CONNECT for HTTPS requires handling the `connect` event separately — the snippet above handles plain HTTP and TLS-terminated proxying; for true CONNECT tunneling you need to inspect the SNI in the TLS hello, which is more code than fits here but is documented in the `http-proxy` README.

## Planning figures and how to measure your own

The ranges below are illustrative planning figures for a browser agent running 1–2 vCPU and 2–4 GB RAM per session, not measured results from a specific deployment. Benchmark on your own workload before committing to a runtime.

- **Cold start:** runc container ~150–300 ms; gVisor `runsc` ~400–800 ms; Firecracker microVM ~125 ms for the VM plus guest boot. Warm pools cut this to 50–100 ms for all three.
- **Syscall overhead:** gVisor adds roughly 2–5x on filesystem-heavy syscalls and 1.2–1.5x on CPU-bound work. For a browser agent, end-to-end session latency typically increases 10–25% versus runc.
- **Egress proxy hop:** 1–5 ms added per request. Negligible relative to page load.
- **Credential issuance:** `sts:AssumeRole` typically returns in 80–200 ms. Cache the credentials for the session, don't re-assume per tool call.
- **Cost:** a 10-minute session on a 2 vCPU instance at typical on-demand rates runs roughly $0.01–$0.03 in compute. The sandbox overhead is not the cost driver; the model inference is.

To replace these with your own numbers, measure three things. First, container start time: wrap `runsc run` in a timestamp diff and log the delta between process spawn and the first successful health check from inside the sandbox, repeated across 50 launches to get a distribution rather than a single sample. Second, syscall overhead: run a fixed filesystem workload (`fio` with a small block size and a known file count, or `sysbench fileio`) under both runc and `runsc` on the same host and compare IOPS and p99 latency. Third, end-to-end session latency: instrument the orchestrator to record the wall-clock time from session request to first agent action and to session teardown, and compare the same agent task with and without the proxy hop. Anything you can't measure this way is a guess.

The limit of the planning figures is that they assume the agent is doing browser work. If you're running a computer-use agent against a native GUI app with heavy file I/O, the gVisor filesystem penalty dominates and you should benchmark Firecracker instead. If your sessions are long (30+ minutes), the warm-pool math changes and you may be better off with a long-lived sandbox and rotating credentials inside it.

## What to watch out for

**The allowlist becomes the attack surface.** Every host you add is a host the agent can reach. Teams commonly add `*.googleapis.com` or `*.amazonaws.com` to unblock something, and that wildcard is now a data exfiltration channel. Prefer exact hostnames and revisit the list weekly.

**DNS rebinding.** If your proxy resolves hostnames to IPs and caches the result, an attacker-controlled DNS record can point at an internal IP after the check. Resolve to IP, verify the IP is not in RFC1918 space, then connect to the IP — not to the hostname. This is the same defense browsers use for their SSRF protections.

**Screenshot exfiltration.** The agent's screenshots go somewhere. If that somewhere is a bucket the agent can also write to with a broad token, a compromised agent can upload a screenshot of an internal dashboard and then read it back. Keep the artifact bucket write-only from the agent's perspective, and read-only from the orchestrator's.

**The "just this once" credential.** The most common regression is a team that hits a session timeout mid-task and extends the token lifetime to 24 hours to "fix" it. That single change undoes the containment. If sessions need to be longer, issue a new short-lived token when the old one expires, and re-verify the session is still doing what it's supposed to do.

**Logging what the agent did.** A record of every tool call, every navigation, and every credential issuance is needed, with the session ID as the correlation key. Without it, incident response on a compromised session is guesswork. Ship logs to a destination the agent's credentials cannot reach.

## The broader lesson

The principle here is not "sandbox your agents." It's that **the trust boundary must be enforced by something the untrusted component cannot influence.** A prompt is influenceable. A tool-call validator that reads pixel coordinates is influenceable. A kernel syscall filter and a network policy enforced by a separate process are not — the agent can't talk its way past them because they don't parse language.

This is the same reason untrusted code isn't run as root in the hope that it behaves. The agent is untrusted code. The fact that it's generated by a model rather than written by a person doesn't change the threat model; it only changes the input distribution. Treat the agent's decisions as adversarial, and design the environment so that the worst decision it can make is still contained.

The corollary is that the cost of containment should be measured against the value of the feature. For a read-only research agent that summarizes public pages, the sandbox is cheap insurance. For an agent that needs to write to a production database, the answer is usually no — not because the sandbox can't be built, but because the agent's failure modes are too varied to enumerate, and the allowlist will consume more engineering time than the feature is worth.

## How to apply this to your situation

Start by mapping what the agent can actually reach today. If it can read a secret, talk to the internet, and be influenced by content it reads, that's the lethal trifecta and should be treated as a P0. The fix order that works: cut the credential scope first (fastest, biggest impact), then add egress allowlisting (medium effort, blocks exfiltration), then move to gVisor or Firecracker (highest effort, contains the rest).

On Kubernetes, the egress policy is a `NetworkPolicy` plus a DNS-aware egress gateway such as Cilium's `toFQDNs`. On a single VM, it's `runsc` plus an iptables default-deny and a local proxy. Either way, the shape is the same: the agent runs in a place where the only way out is a path you control.

## Frequently Asked Questions

**How do I stop a computer-use agent from being prompt-injected?**

You can't stop it at the prompt layer; you contain it at the environment layer. Assume any page the agent reads may contain adversarial instructions, and design so that following those instructions still can't reach anything sensitive. That means short-lived scoped credentials, an egress allowlist, and a kernel-level sandbox. Prompt filtering reduces accidental failures but is not a security control against a determined page.

**What is the difference between gVisor and Firecracker for agent sandboxing?**

gVisor intercepts syscalls in userspace and runs as a normal process, so it starts faster and integrates with standard container tooling, but adds 2–5x overhead on filesystem-heavy syscalls. Firecracker is a KVM-based microVM with near-native performance and a ~125 ms cold start, but a heavier guest boot and more moving parts. For browser agents, gVisor is usually the better default; for GUI apps doing heavy file I/O, benchmark Firecracker.

**Why does my agent session fail with 403 egress_denied after login?**

OAuth and SSO flows redirect through hosts you didn't allowlist — often an identity provider's domain or a CDN the login page pulls assets from. The fix is to log the denied host from your proxy, add it to the allowlist if it's legitimate, and never add a wildcard. A common mistake is adding `*.googleusercontent.com` to unblock an avatar and accidentally opening a broad exfiltration path.

**How long should agent credentials live?**

Shorter than the session. If a session is capped at 10 minutes, issue credentials valid for 9 minutes with `sts:AssumeRole` and `DurationSeconds=540`. Never extend the token lifetime to fix a timeout — issue a fresh one and re-verify the session's task is still what you expect. A 24-hour token turns a single compromised session into a day-long incident.

## The next 30 minutes

Open the IAM role definition your agent assumes and check the `DurationSeconds` on its `sts:AssumeRole` call. If it's longer than your session timeout, change it to `SESSION_TIMEOUT - 60` right now — that single edit is the highest-impact containment change available today, and it takes less time than reading this sentence twice.
