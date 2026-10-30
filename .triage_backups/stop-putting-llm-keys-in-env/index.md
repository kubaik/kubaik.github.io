# Stop Putting LLM Keys in .env

A service that talks to many LLM providers has a habit of breaking in ways the monitoring wasn't watching for. The postmortem almost always says the same thing: this should have been caught sooner. This article covers the fix, the cost of not knowing sooner, and what to monitor.

## The conventional wisdom (and why it's incomplete)

Every secrets management guide starts the same way: don't commit secrets, use environment variables, rotate keys regularly. That advice was written for a world where a service had maybe three external integrations — a database, a payment processor, and an email provider. A mid-sized fintech or healthtech product can now routinely talk to dozens of LLM providers, embedding APIs, vector databases, and inference gateways. The standard advice doesn't scale, and worse, it gives teams a false sense of security.

The contrarian take: for teams with many LLM integrations, environment variables are the wrong abstraction. They're a flat namespace with no provenance, no scoping, and no audit trail. A `.env` file with 40 keys is not a secrets strategy — it's a spreadsheet with extra steps. The moment a second environment, a CI runner, or a contractor appears, that spreadsheet becomes a liability.

The conventional view deserves a fair hearing first, because it isn't stupid. Environment variables are simple, supported everywhere, and keep secrets out of source control. For a team of three shipping a single service, that is genuinely enough. The problem is that the threat model changes as integration count grows, and it changes again past a few dozen.

The part that trips people up is that the failure isn't usually a dramatic breach. It's the slow accumulation of keys nobody owns, in places nobody audits, with rotation schedules that exist only in someone's head. That's what this article actually covers.

## What happens when the standard advice is followed to the letter

A common trap is the "just add another env var" pattern. It works fine until it doesn't. Three failure modes tend to appear, and they compound.

First, the `.env` file becomes the de facto secret store. A large integration setup might have `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `COHERE_API_KEY`, `MISTRAL_API_KEY`, plus dozens more, each with a different naming convention, each rotated on a different cadence. As key count grows, the probability that at least one is stale, over-privileged, or shared across environments approaches certainty. Production systems exist where the staging key and the production key are the same string because someone copy-pasted during a launch crunch.

Second, the values leak into places nobody intended. A common failure mode is a crash handler that dumps `process.env` into a log aggregator. In Node 20 LTS, `console.error(process.env)` in a catch block is a one-line mistake that ships every secret to an observability vendor. The error message looks innocuous — `TypeError: Cannot read properties of undefined (reading 'completion')` — but the stack trace context includes the full environment dump. Datadog, Sentry, and similar tools will happily ingest it. This is one of the most common ways LLM keys end up in third-party systems.

Third, rotation becomes impossible. When a key is referenced in 14 places — the app, the CI pipeline, a Terraform module, a notebook, a cron job — rotating it means finding all 14. Teams typically rotate the keys they remember and leave the rest. Postmortems repeatedly describe the same shape of problem: a meaningful share of keys never rotated, and some still belonging to people who have left the company. The exact percentages vary by organization; the pattern does not.

## A different mental model

Stop thinking of secrets as values you store. Start thinking of them as capabilities you broker.

The shift is from "where do I put this string" to "who or what needs to call this API, for how long, and with what scope." That framing changes everything. A capability has an owner, a lifetime, a scope, and an audit trail. A string in a `.env` file has none of those things.

In practice, this means three moves:

1. **Centralize the store.** Use a real secrets manager — AWS Secrets Manager, GCP Secret Manager, or HashiCorp Vault. The specific tool matters less than the fact that there's one source of truth with versioning and access logs.
2. **Broker short-lived credentials.** Instead of handing your app a long-lived API key, hand it a token that expires quickly and is scoped to a single provider. AWS IAM Roles for Service Accounts (IRSA) does this for AWS resources; the same pattern applies to LLM gateways, which can hold the upstream keys and issue scoped virtual keys to your services.
3. **Route through a gateway.** A gateway gives you one place to enforce rate limits, log usage, redact PII, and rotate upstream keys without touching application code. For dozens of integrations, this is the single highest-leverage change available.

The reason this matters more for LLM integrations than for, say, a Postgres connection is that LLM keys are unusually dangerous. They're bearer tokens with no IP binding by default, they often have generous rate limits, and they're directly monetizable. A leaked LLM key can be drained in hours. A leaked Postgres credential requires the attacker to also reach your VPC.

## A worked gateway example

Consider a gateway sitting in front of many providers. The application only ever sees one credential — a virtual key issued by the gateway. Here's what that looks like in Python:

```python
import os
from openai import OpenAI

# The app holds ONE credential: a scoped virtual key from the gateway.
# Upstream provider keys live in the gateway, never in the app.
GATEWAY_KEY = os.environ["LLM_GATEWAY_KEY"]
GATEWAY_URL = "https://llm-gateway.internal/v1"

client = OpenAI(base_url=GATEWAY_URL, api_key=GATEWAY_KEY)

response = client.chat.completions.create(
    model="anthropic/claude-sonnet-4",
    messages=[{"role": "user", "content": "Summarize this claim note."}],
    extra_body={"metadata": {"user_id": "clinician-4412", "trace_id": "req-8f2a"}},
)
```

When a provider key needs rotation, you rotate it in the gateway's config — one place. The application doesn't redeploy. The audit log shows which service called which model, when, and with what token. That's a capability model, not a string model.

The failure mode this prevents is concrete. Teams that skip the gateway and instead inject every key into the app environment typically discover the leak the hard way: a `git log -p` on an old branch, a Sentry event with a full env dump, or an unexpected bill. To measure the exposure window, instrument two things: (1) the timestamp of the first anomalous provider call, and (2) the timestamp your billing alert fires. The gap between them is your detection latency. If that gap is measured in hours, an attacker has already had hours.

Here's the other pattern worth knowing: the CI leak. A common mistake is passing secrets as build args in a Dockerfile. Build args are visible in `docker history` and in image metadata. The fix is to never pass secrets at build time — inject them at runtime from the secrets manager:

```javascript
// Avoid: secrets baked into the image
// ARG OPENAI_API_KEY
// ENV OPENAI_API_KEY=$OPENAI_API_KEY

// Prefer: fetch at runtime from the secrets manager
import { SecretsManagerClient, GetSecretValueCommand }
  from "@aws-sdk/client-secrets-manager";

const client = new SecretsManagerClient({ region: "us-east-1" });

async function getProviderKey(provider) {
  const res = await client.send(new GetSecretValueCommand({
    SecretId: `llm/${provider}/api-key`,
  }));
  return JSON.parse(res.SecretString).api_key;
}
```

That's roughly a dozen lines per service, and it removes the entire class of "key in image layer" findings from a pentest. The tradeoff is a cold-start latency cost — typically tens of milliseconds on the first call, then cached. For most workloads that's invisible; for a p99-sensitive endpoint, cache it in memory with a short TTL.

## How to measure your own exposure

Rather than trusting anyone's numbers, instrument these directly:

- **Key inventory:** enumerate every secret path in your store and every `.env` reference in your repos. Diff the two sets. Anything in one but not the other is unmanaged.
- **Rotation age:** most secret managers expose a `LastChangedDate` per secret. Query it and sort descending. Anything older than your stated policy is a finding.
- **Reachability:** for each key, list every deployment, CI job, and notebook that references it. A key referenced in more than three places is a rotation hazard.
- **Exposure surface:** run a secret scanner over git history and container images, not just the working tree. `gitleaks detect --source . --no-git -v` covers the working directory; add the git-history and image-layer modes for the full picture.

The point of measuring is to replace "we probably rotate" with a list. A list can be worked through; a feeling cannot.

## The cases where the conventional wisdom IS right

Environment variables aren't wrong everywhere. They're correct when the blast radius is small and the team is small. With fewer than ten integrations, one deployment target, and no contractors, a well-managed `.env` file with encryption at rest and a documented rotation calendar is genuinely fine. A three-person team is better served by that plus a calendar reminder than by a half-implemented, misconfigured Vault.

The conventional advice is also right about the fundamentals: never commit secrets, scan your repos, and rotate. Those aren't in dispute. What's in dispute is the assumption that env vars are a sufficient *delivery mechanism* at scale. Tools like `gitleaks` and `trufflehog` catch committed secrets, and every team should run them in CI. But they only catch the leak *after* it's in git history, which means the key is already compromised. Prevention beats detection.

There's also a legitimate argument that gateways add a single point of failure and some latency. That's true. A gateway that goes down takes every integration with it. The mitigation is standard: run it as a stateless service behind a load balancer, with health checks and a fallback path. The latency cost is real but small — typically single-digit to low-double-digit milliseconds added per call — and it's almost always cheaper than the alternative.

## How to decide which approach fits your situation

Use this table as a rough decision guide. The thresholds are approximate and should be tuned to your own risk tolerance and audit requirements.

| Integrations | Team size | Recommended approach | Rotation cadence |
|---|---|---|---|
| 1–10 | 1–5 | Encrypted `.env`, documented rotation | Quarterly |
| 10–25 | 5–20 | Central secrets manager | Monthly |
| 25–40 | 20+ | Secrets manager + LLM gateway | Weekly |
| 40+ | 20+, multi-region | Gateway + short-lived virtual keys + audit | Daily/on-demand |

The key insight in that table is that the *mechanism* should change as the count grows, not just the discipline. Teams that try to solve a 40-integration problem with better `.env` hygiene are bringing a spreadsheet to a database problem. The tooling has to match the scale.

One more decision axis: regulatory exposure. A healthtech product handling PHI under HIPAA, or a fintech under PCI-DSS, will be pushed toward the gateway model regardless of team size. Auditors want to see access logs, scoped credentials, and rotation evidence. A `.env` file provides none of those artifacts.

## Common objections

**"A gateway is just another thing to operate."** True, and it's a real cost. But you're already operating dozens of integrations — the question is whether you want dozens of direct connections or one managed hop. Teams that adopt a gateway often spend *less* total operational effort because they stop debugging per-provider auth issues.

**"We can't put PHI through a gateway."** You don't have to. The gateway brokers credentials; it doesn't have to log payloads. Configure it to redact or drop request bodies, and keep the audit log to metadata only — model, timestamp, token ID, status code. That satisfies most auditors and keeps PHI out of the gateway's storage.

**"Short-lived keys break our long-running jobs."** This is the most legitimate objection. Batch jobs that run for hours need credentials that outlive a short token. The answer is token refresh: the job holds a refresh credential and exchanges it for short-lived tokens as needed. It's more code, but it's the same pattern every cloud SDK already uses.

**"We're too small for this."** Then you're probably not at dozens of integrations yet. The advice in this article is scoped to teams that are. If you're at 8, keep your `.env` and revisit when you hit 20.

## What to do from day one on a new service

If you're standing up a new service that you know will eventually talk to many LLM providers, do three things before writing any integration code.

First, put a gateway in front from the start. Not because it's needed at three integrations, but because retrofitting one at forty is a multi-week migration. Managed LLM gateways and open-source gateways both support the major providers, and the config is measured in dozens of lines, not thousands.

Second, adopt a naming convention for secrets that encodes owner, environment, and provider — something like `llm/prod/anthropic/api-key` — and enforce it in the secrets manager's IAM policy. This sounds bureaucratic until you're trying to answer "who owns this key" during an incident at 2 a.m.

Third, wire up secret scanning and rotation from the first commit. A scanner in a pre-commit hook, a weekly rotation job for non-critical keys, and an on-demand rotation runbook for the rest. The cost of doing this early is a few hours; the cost of doing it after a leak is measured in incident response, customer notification, and sometimes regulatory fines.

Most teams don't do this until they get burned. The teams that do it early usually have a security-conscious founder or a compliance requirement forcing their hand. You can choose which group you're in.

## Summary

Environment variables are fine for a handful of integrations and a small team. At dozens of LLM integrations, they're the wrong abstraction — a flat namespace with no provenance, no scoping, and no audit trail. The fix is to move from storing secrets to brokering capabilities: a central secrets manager, a gateway that holds upstream keys, and short-lived scoped tokens for your services. This costs a few milliseconds of latency and some operational effort, and it removes entire classes of leak — the env dump in a crash handler, the key baked into a Docker layer, the unrotated credential from a departed employee.

The conventional advice about not committing secrets and rotating regularly is still correct. It's just not sufficient. The mechanism has to match the scale.

If you take one action in the next 30 minutes: run `gitleaks detect --source . --no-git -v` against your current working directory and review the output. If it flags any LLM provider keys, you've just found the leak you didn't know you had — and you know exactly which file to fix first.

## Frequently Asked Questions

**How many LLM API keys is too many for a .env file?**
There's no hard line, but the practical threshold is around 15–20. Below that, an encrypted `.env` with a documented rotation calendar is manageable. Above it, the probability of a stale, shared, or over-privileged key climbs fast, and you lose the ability to answer "who owns this key" during an incident. At 40+, the `.env` approach is actively dangerous.

**Why route LLM calls through a gateway instead of calling providers directly?**
A gateway gives you one place to hold upstream keys, enforce rate limits, log usage, redact sensitive fields, and rotate credentials without redeploying application code. The latency cost is typically a few milliseconds per call, which is negligible for most workloads. The alternative — dozens of direct integrations with dozens of sets of credentials — makes rotation and auditing nearly impossible.

**What is the most common way LLM API keys leak in production?**
Two patterns dominate: crash handlers that dump `process.env` into a log aggregator, and secrets passed as Docker build args that end up in image layers visible via `docker history`. Both are one-line mistakes that expose every key in the environment. Runtime injection from a secrets manager eliminates both.

**Can I use short-lived credentials for long-running batch jobs?**
Yes, via token refresh. The job holds a refresh credential and exchanges it for short-lived tokens as needed — the same pattern cloud SDKs already use. It's slightly more code than a static key, but it means a leaked token is useless within minutes instead of months.
