# Scope agent permissions per task

## The one-paragraph version

Least privilege for AI agents means giving an agent only the permissions it needs to complete its current task, for only as long as it needs them. The concept is simple; the implementation is where teams get stuck. Developers rarely resist least privilege because they disagree with it — they resist it because the obvious implementations (static IAM roles, hardcoded API keys, broad OAuth scopes) add friction to every local test, every deploy, and every new tool the agent needs. The result is a familiar pattern: a well-intentioned security policy that gets bypassed within a couple of sprints. The permission model that survives contact with a real codebase is one where the agent's identity is scoped per task, credentials are short-lived and automatically rotated, and the local environment mirrors production closely enough that nobody needs an escape hatch. That is what this article covers.

## Why this concept confuses people

The confusion starts with the word "agent." In traditional software, a service account is a stable identity with a fixed set of permissions. It is created once, a policy is attached, and it runs until deleted. An AI agent is different: it decides at runtime which tools to call, in what order, and with what arguments. The permission surface is therefore not static — it is a function of the task the agent is currently executing.

A common failure mode: a team gives an agent a service account with `s3:GetObject` on `arn:aws:s3:::my-bucket/*`. The agent works fine for months. Then someone adds a new tool that needs to write a report to the same bucket. The quick fix is to add `s3:PutObject` to the same role. A year later the role has dozens of permissions, nobody knows which tool needs which, and the security team flags it in an audit. The agent's effective permissions are now indistinguishable from a human power user's.

Another source of confusion is the difference between *authentication* (who is this agent?) and *authorization* (what can it do right now?). Most teams solve authentication well — they use an API key or an OIDC token — and then treat authorization as an afterthought. But authorization is where least privilege actually lives, and it is the part that requires runtime decisions.

The third confusion is tooling. AWS IAM, GCP IAM, Azure RBAC, and OAuth 2.0 scopes all have different models for expressing fine-grained permissions. A developer who knows IAM policies well may still struggle with OAuth scopes, and vice versa. The mental model does not transfer cleanly, so teams end up with inconsistent patterns across services.

## The mental model that makes it click

Think of an AI agent like a contractor hired for a specific job. The contractor does not get a master key to every room in the building on day one. They get a key to the rooms they need for the job, and the key is taken back when the job is done. If they need a different room tomorrow, a new key is issued.

That is the core of least privilege for agents: **scoped, short-lived credentials issued per task, not per agent.**

In practice, this means three things:

1. **The agent has a stable identity** (so it can be audited), but that identity does not carry permissions directly.
2. **Permissions are attached to a task context** — a short-lived token, a session, or a role assumption that expires.
3. **The agent requests permissions at runtime** through a broker or token service, and the broker enforces policy.

A useful analogy is a hotel key card. The card identifies the guest, but it only opens the rooms that were paid for, and it stops working after checkout. The hotel does not rekey every door when the guest leaves; it just invalidates the card.

This model solves the friction problem because developers do not have to manage long-lived secrets. They configure the broker once, and the agent gets what it needs automatically. Local development uses a mock broker that issues test-scoped tokens, so the developer experience mirrors production.

The key insight: **least privilege is not about restricting the agent; it is about making the agent's permissions legible.** When permissions are scoped per task, the question "what could this agent have done?" can be answered by looking at one token's claims, not by tracing dozens of IAM policies.

## A concrete worked example

Consider a support agent that reads customer tickets from a helpdesk API, looks up order status in an internal API, and sends a response email through a transactional email provider. The agent runs as a containerized service on a managed container platform.

**The naive approach:** create an IAM role for the container task with permissions to read tickets, call the internal API, and send email. Store the helpdesk and email provider API keys in a secrets manager and inject them as environment variables. This works. It also means the agent has permanent access to all three services, and if the container is compromised, the attacker gets all three keys.

**The scoped approach:** the container task role has permission to call a token broker (for example, an internal service running on a serverless function). The broker authenticates the task using its IAM role, checks the task's current job (for example, `ticket_id=12345`), and issues a short-lived JWT with scopes like `zendesk:read:ticket:12345`, `orders:read:order:67890`, and `sendgrid:send:email`. The JWT expires in 15 minutes. The agent calls the downstream services with this JWT. Each service validates the JWT and enforces the scope.

Here is a simplified Python example of the broker's policy check:

```python
import jwt
from datetime import datetime, timedelta

# Simulated policy store: maps job type to required scopes
def get_required_scopes(job_type: str) -> list[str]:
    policies = {
        "ticket_response": ["zendesk:read:ticket", "orders:read:order", "sendgrid:send:email"],
        "order_lookup": ["orders:read:order"],
        "escalation": ["zendesk:read:ticket", "zendesk:write:ticket"],
    }
    return policies.get(job_type, [])

def issue_token(task_id: str, job_type: str, resource_ids: dict) -> str:
    scopes = get_required_scopes(job_type)
    if not scopes:
        raise ValueError(f"Unknown job type: {job_type}")

    # Build scoped claims: e.g., zendesk:read:ticket:12345
    scoped_claims = []
    for scope in scopes:
        resource_key = scope.split(":")[1]  # e.g., 'ticket' or 'order'
        resource_id = resource_ids.get(resource_key)
        if not resource_id:
            raise ValueError(f"Missing resource ID for {resource_key}")
        scoped_claims.append(f"{scope}:{resource_id}")

    payload = {
        "sub": task_id,
        "scopes": scoped_claims,
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(minutes=15),
    }
    return jwt.encode(payload, "secret-key", algorithm="HS256")
```

Note that the email scope is not resource-scoped in this example, because the email provider's send permission is not tied to a specific resource ID. In a real deployment, that scope would typically be narrowed by a recipient-domain or template allowlist enforced at the downstream service.

Here is how a downstream service (for example, the internal orders API) validates the token and enforces the scope:

```python
from fastapi import FastAPI, HTTPException, Header
import jwt

app = FastAPI()

@app.get("/orders/{order_id}")
def get_order(order_id: str, authorization: str = Header(...)):
    token = authorization.replace("Bearer ", "")
    try:
        payload = jwt.decode(token, "secret-key", algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

    required_scope = f"orders:read:order:{order_id}"
    if required_scope not in payload.get("scopes", []):
        raise HTTPException(status_code=403, detail=f"Missing scope: {required_scope}")

    # Fetch order from database
    return {"order_id": order_id, "status": "shipped"}
```

This pattern adds on the order of 40 lines of code to the broker and about 10 lines to each downstream service. JWT validation latency is typically a few milliseconds, which is small compared to the network calls the agent is already making. The security benefit is substantial: if the agent container is compromised, the attacker gets a token that expires in 15 minutes and only works for one ticket ID.

The friction for developers is minimal because the broker is a shared service. A developer writing a new tool for the agent adds the required scopes to the policy store and tests locally using a mock broker that issues tokens with the same claims. There is no need to request new IAM roles or wait for a security review for every change.

### Which resource IDs should be in scope?

The scoping granularity is a design decision, not a default. Three options are worth comparing:

- **Scope by resource ID** (`orders:read:order:67890`). Tightest blast radius, but the broker must know the ID before issuing the token. Works when the task's inputs are known upfront.
- **Scope by resource type plus a filter claim** (`orders:read:order`, with a `customer_id` claim the service checks). Looser, but usable when the agent discovers IDs at runtime.
- **Scope by resource type only** (`orders:read:order`). Easiest to implement, weakest isolation. Acceptable only when the underlying service already enforces per-user access.

A practical rule: scope by resource ID for write operations and for reads of sensitive records; scope by type plus filter for reads of records the task legitimately enumerates.

## How this connects to things you already know

If you have worked with AWS Lambda, this pattern resembles how Lambda execution roles work — except instead of one role per function, there is one role per task. The broker is a custom authorization layer that sits between the agent and the downstream services.

If you have used OAuth 2.0 for user-facing apps, the scoped token is like an access token with a narrow scope. The difference is that the "user" is the agent itself, and the scopes are tied to a specific resource ID, not just a resource type.

If you have used Kubernetes service accounts with RBAC, the broker is like a dynamic admission controller that issues per-pod credentials. The concept is the same: do not give the pod a cluster-admin role; give it a role scoped to the namespace and resource it needs.

The pattern also connects to zero-trust networking: never trust, always verify. The agent's identity is verified at the broker, and every downstream call is verified again. There is no implicit trust based on network location.

One important difference from traditional service accounts: the broker must be highly available. If the broker goes down, the agent cannot get tokens, and the task fails. A token TTL of 15 minutes means a brief broker outage does not immediately break running tasks — they can continue using their existing tokens until they expire. That property is worth designing for explicitly: size the TTL against the acceptable broker outage window, not just against task duration.

## Common misconceptions, corrected

**Misconception 1: "Least privilege means the agent can't do anything useful."**

Reality: the agent can do exactly what it needs for the current task. The scoped token for a ticket response includes read access to that ticket, read access to the related order, and send access to email. That is enough to complete the task. The agent does not need access to all tickets or all orders.

**Misconception 2: "We can just use IAM policies and be done."**

IAM policies are necessary but not sufficient. They define what the agent's identity can do, but they do not scope permissions to a specific task. An IAM policy can allow `s3:GetObject` on a specific prefix, but that permission cannot easily be made to expire after 15 minutes or be tied to a specific ticket ID. The broker layer adds that runtime scoping. In practice the two layers compose: IAM restricts the agent's identity to calling the broker, and the broker restricts what the task token can do.

**Misconception 3: "Short-lived tokens are too much overhead."**

The overhead is real but small. JWT validation adds a few milliseconds per request. Token issuance adds tens of milliseconds per task. For an agent that takes seconds to complete a task, that is a small fraction of total latency. The developer overhead is also small if the broker is well-designed: adding a new scope is a one-line change in the policy store.

**Misconception 4: "This is only for high-security environments."**

Least privilege is valuable for any agent that touches user data or external services. A common failure mode: an agent with broad permissions gets prompt-injected and starts exfiltrating data. With scoped tokens, the blast radius is limited to the current task's resources. The agent cannot read other users' tickets or send emails to arbitrary addresses.

**Misconception 5: "We need a full identity provider."**

A minimal broker can be built with a few hundred lines of code and a JWT library. A full IdP is not required unless human users are being managed too. Many teams start with a simple broker and migrate to a managed service later if needed.

## The advanced version (once the basics are solid)

Once the basic broker pattern works, several advanced techniques are worth considering.

**Dynamic scope elevation:** instead of issuing all scopes upfront, the agent can request additional scopes at runtime if it needs them. For example, a ticket response agent might start with read-only scopes and request `sendgrid:send:email` only when it is ready to send. This reduces the window of exposure. The broker can enforce additional checks (for example, human approval) before granting elevated scopes.

**Audit logging with correlation IDs:** every token issuance and every downstream call should be logged with a correlation ID that ties back to the task. This makes it possible to answer "what did this agent do?" by querying logs for a single ID. In practice, this means adding a `task_id` claim to the JWT and propagating it through all downstream calls.

**Rate limiting per agent:** scoped tokens prevent access to unauthorized resources, but they do not prevent an agent from making too many authorized calls. Add rate limiting at the broker level (for example, a cap on token issuances per minute per agent) and at the downstream service level (for example, a cap on requests per second per token). This prevents runaway agents from overwhelming services.

**Token revocation:** JWTs are stateless, so revoking a token before it expires requires a blocklist. For a 15-minute TTL, an in-memory blocklist with a 15-minute TTL is often sufficient. The broker checks the blocklist before issuing a new token, and downstream services check it before accepting a token. This adds a small amount of state but enables immediate revocation if an agent is compromised.

**Multi-cloud and hybrid:** if the agent calls services across AWS, GCP, and on-prem, the broker must issue tokens accepted by all of them. The simplest approach is to use a standard like OAuth 2.0 with JWT access tokens and configure each service to trust the broker's signing key. This avoids vendor-specific token formats.

**Performance considerations:** at scale, the broker can become a bottleneck. Throughput depends on the JWT library, the signing algorithm, and the hardware. Asymmetric signing (RS256) is slower than symmetric (HS256) but allows downstream services to validate tokens without sharing the secret. For most agent workloads, HS256 is sufficient and faster. To find the actual ceiling, benchmark the broker directly: issue tokens in a loop against a running instance and record p50/p99 latency and requests per second at increasing concurrency. Do this before assuming a single instance is enough.

## What to measure

The claims above about latency and overhead are only useful if they are verified against a specific deployment. The following are the concrete things to instrument, and how:

- **Token issuance rate.** Emit a counter at the broker on every successful issuance, labelled by agent ID and job type. Query it as a rate over one minute. A sudden rise means either a new workload or a loop.
- **Token TTL distribution.** Record the configured `exp - iat` as a histogram at issuance. This catches drift where someone quietly raises the TTL to "fix" a timeout.
- **Broker latency.** Time the issuance path end to end and record a histogram; alert on p99. Compare against the downstream service's own latency to confirm the broker is not the dominant cost.
- **Denied scope requests.** Emit a counter at each downstream service on every 403 caused by a missing scope, labelled by required scope. A steady trickle means policies are slightly behind the code; a spike means a deploy changed scope requirements without updating the policy store.
- **Token reuse across tasks.** Log `sub` (task ID) and the resource IDs in the scopes. If one token's scopes reference multiple unrelated task IDs, the broker is leaking scope across tasks.

A useful before/after comparison is a single request trace: run one agent task with tracing enabled, and compare the span durations for the broker call and the downstream calls. That tells you the real overhead fraction for your workload, rather than a generic estimate.

## Failure modes to design against

- **Broker outage blocks all tasks.** Mitigate with multiple instances and a TTL long enough to ride out a restart, and make the agent fail closed rather than fall back to a long-lived credential.
- **Policy store drift.** The policy store and the agent code evolve separately. Mitigate by generating the required-scope list from the tool definitions where possible, and by failing CI if a tool's declared scopes are not present in the store.
- **Scope string parsing bugs.** Splitting scope strings on `:` is fragile if any segment can contain a colon. Validate scope format at issuance and at validation, and reject malformed scopes rather than ignoring them.
- **Clock skew.** JWT validation depends on synchronized clocks. Allow a small leeway in the decoder and monitor for `ExpiredSignatureError` spikes that correlate with host clock drift.
- **Over-broad "read" scopes.** A scope like `orders:read:order` with no resource ID is effectively a type-wide grant. Audit the policy store for scopes that lack a resource ID and confirm each one is intentional.

## Quick reference

| Aspect | Naive approach | Scoped approach |
|--------|----------------|-----------------|
| Credential lifetime | Until manually rotated | A few minutes to an hour, chosen per task |
| Permission scope | All resources of a type | Specific resource ID, or type plus filter |
| Blast radius if compromised | Everything the agent can reach | One task's resources |
| Developer friction | Low initially, high over time | Low if broker is well-designed |
| Auditability | Requires tracing many policies | One token's claims |
| Latency overhead | None added | A few milliseconds per request |
| Implementation cost | None added | Roughly one broker plus a small hook per service |

**Key metrics to track:**
- Token issuance rate (per agent, per minute)
- Token TTL distribution (watch for drift)
- Broker latency (p99, compared against downstream latency)
- Denied scope requests (labelled by scope)

**Common error messages and what they mean:**
- `403 Forbidden: Missing scope: orders:read:order:12345` — the token does not include the required scope. Check the broker's policy for the job type and the resource ID mapping.
- `401 Unauthorized: Token expired` — the token TTL has passed. The agent should request a new token, or the TTL is too short for the task.
- `400 Bad Request: Unknown job type: ticket_response_v2` — the job type is not in the policy store. Add it, and add a CI check so this fails before deploy.

## Frequently Asked Questions

**How do I test least-privilege permissions locally without a full broker setup?**

Run a mock broker in the local development environment that issues tokens with the same claims as production. Use a JWT library to generate tokens with a test signing key, and configure downstream services to trust that key in development. This gives the same scoping behavior without deploying the real broker. In Python, a JWT library such as PyJWT can generate a token with the scopes you expect, and the service code will validate it identically to production. The differences are the signing key and the absence of a real policy store — so also add a test that asserts the mock broker and the real policy store produce the same scope list for each job type.

**What token TTL should I use for AI agents?**

Start from the task's expected duration plus a small buffer, then check the result against two constraints: how long a leaked token should remain usable, and how long the broker can be unavailable without breaking running tasks. A 15-minute TTL is a common starting point because most short agent tasks finish well within it. For long-running jobs, either size the TTL to the job or implement refresh so the agent can request a new token mid-task. Avoid TTLs longer than an hour unless there is a specific reason; the security benefit of short TTLs diminishes as the TTL grows.

**How do I handle agents that need permissions across multiple cloud providers?**

Use a standard token format like JWT and configure each provider's services to trust the broker's signing key. For AWS, an API Gateway custom authorizer can validate the JWT. For GCP, a gateway or endpoint layer can do the same. For on-prem services, validate the JWT directly in application code. The key is to avoid vendor-specific token formats and stick to a standard that all services can validate. This adds complexity to the broker (it must sign tokens all services trust) but simplifies the agent's code, which just passes the token through.

**What is the biggest mistake teams make when implementing least privilege for agents?**

Treating it as a one-time project instead of an ongoing practice. Teams often implement scoped tokens, then gradually add broad permissions back when a new tool needs them, and within a year they are back to a monolithic role. The fix is to make the policy store the single source of truth and require a review for any change that adds a new scope. Automate the check: if a pull request adds a scope to the policy store, require approval from the security team. This keeps the permission model legible and prevents drift.

**Should the agent ever fall back to a long-lived credential when the broker is unreachable?**

No. A fallback credential defeats the entire model, because it is exactly the credential an attacker would prefer. Fail the task instead, and make the failure visible: emit a counter for broker-unreachable events and alert on it. If the failure rate is unacceptable, fix availability at the broker rather than adding a fallback path.

## Your next step

Open the agent's IAM role or service account configuration and list every permission it has. For each permission, ask: "Does the agent need this for every task, or only for specific tasks?" If the answer is "specific tasks," you have found a candidate for scoping. Start with the highest-risk permission (for example, write access to an external service) and implement a scoped token for just that one: add a broker endpoint that issues a short-lived JWT, and update one downstream service to validate it. Then add a counter for denied scope requests at that service so you can see whether the policy store keeps up with the code.
