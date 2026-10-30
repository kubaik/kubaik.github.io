# Agents create incidents at 2am

## The gap in the standard playbook

The usual guidance for operating automation is sound as far as it goes: automate repetitive work, let agents absorb low-level noise, invest in observability, and page humans only when the blast radius justifies it. That advice holds when agents are well-scoped workers doing read-mostly tasks. It breaks down when an agent is given autonomy to mutate production state, open incident tickets, and page humans — for example, because a synthetic account verification step exceeded its SLA by a few milliseconds.

The problem is not observability or automation. It is the assumption that an agent can be promoted from background worker to first-class incident creator without changing the permission model that governs it. Most operational playbooks stop at the alert router. They describe how anomalies reach a human, but not what the agent is allowed to do once it is on the path to creating an incident.

The distinction that matters is the delta between *agent can detect* and *agent can mutate*. Everything below is about closing that delta.

## A worked failure: one latency spike, two pages

Consider a typical stack: a Node backend on a container platform, Python workers on a managed container service, a cloud metrics service for time-series data, and an incident-management SaaS for paging. An automation agent polls a payment provider's API every five minutes to verify subscription status. If a subscription is inactive, the agent opens a ticket in an issue tracker and applies a label such as `finops:cost-overrun` so the billing team can act.

This works until the webhook signature verification step starts timing out. Instead of a clean 40 ms response, p99 latency climbs to 1.2 s. The agent's retry policy — say, three attempts with exponential backoff — exhausts its 2 s timeout on the third attempt. The agent emits a `CRITICAL` metric, the alarm fires, and the on-call engineer is paged.

Then the agent does the things that turn one anomaly into an avalanche:

- It transitions the ticket from `Open` to `In Progress`.
- It appends a comment recording that the automated check triggered the incident.
- Because the billing team's paging service is subscribed to the `finops:cost-overrun` label, a second page fires moments later.

A single upstream latency spike produced two pages, woke two humans, and left a ticket in a state that needed manual cleanup the next morning. Nothing in the alerting stack malfunctioned. The agent's role had expanded — implicitly, through accumulated permission grants — from detector to mutator.

A common variant of this failure mode: an agent that runs with a role permitted to assume a service account with write access to the issue tracker. A concurrency spike exhausts that service account's API quota, retries fire, and the same ticket is mutated several times, generating duplicate pages. The triggering condition is mundane; the amplification comes entirely from write permissions.

## Three circles, not two

A more useful mental model is three concentric circles rather than two.

1. **Detector circle** — what the agent can observe: metrics, logs, traces, API responses.
2. **Router circle** — how the agent surfaces anomalies: alerts, dashboards, ticket labels, notifications.
3. **Mutator circle** — what the agent is allowed to change: production state, tickets, paging state, infrastructure.

The standard playbook covers circles one and two and omits circle three. Incident avalanches happen where the mutator circle overlaps the other two. The goal is not to eliminate the mutator circle — some agents legitimately remediate — but to make it explicit, small, and auditable.

Two traps recur.

**Role creep.** An agent built to watch a queue-depth metric is later granted write scope on the issue tracker so it can mark stale tickets. That single permission change converts a passive observer into an active participant in incident triage. Once the agent can open or mutate tickets, its blast radius is no longer bounded by the metric it watches.

**State leakage across contexts.** An agent running in a function with a role that can assume a cross-account service account may, under retry pressure, mutate the same ticket repeatedly. The permission itself is not the bug; the absence of an idempotency key and a bounded retry policy is.

The practical artifact that resolves both is a **write boundary**: an explicit, version-controlled definition of which resources an agent may mutate, under what rate and SLO constraints, and with what identity. If the boundary is not in infrastructure code, it is not a boundary.

## Where the real edge cases live

### Cyclic trust policies in multi-account setups

An agent in account A assumes a role in account B, and the role in account B is trusted to assume a role back in account A for access to a third-party API. This transitive loop is a misconfiguration, but it is easy to create accidentally when teams add cross-account access incrementally. Under a cold-start spike, many concurrent executions each initiate a token-exchange handshake. The STS endpoint throttles, the agent's exponential backoff compounds concurrency, and the degradation spreads to every agent in the region using that endpoint.

The mitigation is not a clever retry policy. It is: do not create cyclic trust, cap function concurrency at a level the downstream identity provider can absorb, and test the trust graph with a tool that can enumerate assume-role paths. A useful check is to simulate the principal's policy against `sts:AssumeRole` on the target roles and confirm the result is what you expect — not merely that it is allowed.

### Duplicate triggers during deployment windows

A log subscription filter that invokes a function which opens incidents is vulnerable during deployments. When the function is updated, there is a window in which the subscription can invoke both the old and new versions, or invoke the same event more than once. If each invocation independently calls the incident API, a single anomaly produces several incidents.

Two defenses, both cheap:

- **Idempotency keys.** Most incident APIs accept a deduplication key. Derive it deterministically from the anomaly identity — for example, a hash of the source, the metric name, and the anomaly's time bucket — not from a random UUID generated per invocation.
- **Synchronous invocation and a dedupe layer.** Make the trigger path synchronous where the platform allows it, and add a short-lived dedupe store (a cache with a TTL, or a conditional write to a key-value store) in front of the incident call.

Incident APIs commonly enforce per-key rate limits and return a retry-after header on 429. If every agent shares one API key and one agent enters a retry storm, the resulting 429s can black out incident creation for *all* agents using that key. Separate keys per agent class, and treat 429 as a signal to shed load rather than to retry immediately.

### Metadata endpoint trust in dual-stack networks

Container tasks commonly fetch configuration from a link-local metadata endpoint. If the network permits non-local sources to reach that address — for example, through a dual-stack misconfiguration or an overly permissive load balancer — a process in the same VPC can inject false environment variables. An agent that trusts the metadata endpoint implicitly may then read a corrupted credential and begin making authenticated calls with it, generating authentication failures that its own failure policy converts into tickets.

The fix is network-level: disable the unused address family on the task network interface, and add firewall rules that drop all traffic to the metadata address from non-local sources. Treat the metadata endpoint as trusted only from the local task.

## Enforcing a write boundary in code

### Idempotent incident creation

The single highest-leverage change is deterministic deduplication. The example below validates the payload with a schema and derives the dedupe key from the anomaly rather than generating a new one per call.

```python
# agent_incident.py
import hashlib
import os
from datetime import datetime, timezone

import requests
from pydantic import BaseModel, SecretStr


class IncidentPayload(BaseModel):
    routing_key: SecretStr
    event_action: str
    dedup_key: str
    payload: dict


def anomaly_dedup_key(source: str, metric: str, bucket_minutes: int = 15) -> str:
    """Deterministic key: same anomaly -> same key -> one incident."""
    now = datetime.now(timezone.utc)
    bucket = now.replace(
        minute=(now.minute // bucket_minutes) * bucket_minutes,
        second=0,
        microsecond=0,
    )
    raw = f"{source}:{metric}:{bucket.isoformat()}"
    return hashlib.sha256(raw.encode()).hexdigest()


def trigger_incident(summary: str, source: str, metric: str) -> str:
    payload = IncidentPayload(
        routing_key=os.environ["INCIDENT_ROUTING_KEY"],
        event_action="trigger",
        dedup_key=anomaly_dedup_key(source, metric),
        payload={
            "summary": summary,
            "source": source,
            "severity": "critical",
        },
    )
    resp = requests.post(
        "https://events.example-incident-api.com/v2/enqueue",
        json=payload.model_dump(exclude={"routing_key"}),
        headers={"Content-Type": "application/json"},
        timeout=5,
    )
    resp.raise_for_status()
    return resp.json()["dedup_key"]
```

Two properties matter here. First, the dedupe key is a pure function of the anomaly, so retries and duplicate invocations collapse into one incident. Second, the routing key comes from the environment, so each agent class can hold its own credential and its own rate-limit budget.

### A minimal permissions boundary

A permissions boundary caps the maximum permissions an identity can hold, regardless of what its attached policies grant. The Terraform below defines a boundary that allows only logging and read-only key-value access — no issue-tracker writes, no cross-account assume-role.

```hcl
resource "aws_iam_role" "agent_lambda_role" {
  name = "agent-lambda-write-boundary-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_boundary" "agent_boundary" {
  role   = aws_iam_role.agent_lambda_role.name
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action   = ["logs:PutLogEvents", "dynamodb:Query"]
      Effect   = "Allow"
      Resource = "*"
    }]
  })
}
```

Verify the boundary actually binds by simulating the principal against the action you are worried about:

```bash
aws iam simulate-principal-policy \
  --policy-input-file boundary.json \
  --action-names sts:AssumeRole \
  --resource-arns arn:aws:iam::123456789012:role/target-role
```

The simulation should return `implicitDeny` for the actions the boundary excludes. If it returns `allowed`, the boundary is not attached or not scoped as intended.

### Bounded retries and idempotent ticket updates

Retries are where read-only agents become write-amplifying agents. Bound them, and make every write check current state first.

```python
# agent_tickets.py
import os

from jira import JIRA
from tenacity import retry, stop_after_attempt, wait_exponential


class SafeTicketClient:
    def __init__(self) -> None:
        self.client = JIRA(
            server="https://your-domain.atlassian.net",
            basic_auth=(os.environ["JIRA_EMAIL"], os.environ["JIRA_API_TOKEN"]),
            timeout=5,
            max_retries=3,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def mark_ticket_stale(self, issue_key: str, comment: str):
        issue = self.client.issue(issue_key)
        if issue.fields.status.name == "Stale":
            return issue  # Already in target state; no write.
        return self.client.update_issue(
            issue_key,
            fields={"status": {"name": "Stale"}},
            update={"comment": [{"add": {"body": comment}}]},
        )
```

The read-before-write check is the important part. It converts a non-idempotent mutation into an idempotent one, so a retry storm cannot multiply the number of state changes.

## How to measure whether this is working

Do not trust a before/after table from someone else's environment. Instrument your own. Five signals are enough to start.

- **Pages attributable to agents.** Tag every incident created by an automated path with a distinct source field, then count incidents grouped by source. Compare the agent-triggered share against the human-triggered share over the same window.
- **Duplicate rate.** For each anomaly identity, count incidents created within one dedupe window. The target is one. Anything higher means the dedupe key is not deterministic or is not being passed through.
- **Write attempts per anomaly.** Count mutations (ticket transitions, comments, state changes) per anomaly identity. This number should be small and stable; growth here is the earliest sign of role creep.
- **Retry amplification.** For each agent, record retries per logical operation and the resulting concurrency. A retry policy that increases concurrency under load is the mechanism behind most regional degradations.
- **Identity-provider throttle events.** Count throttling responses from your token-exchange and third-party APIs, grouped by agent identity. These are leading indicators of a concurrency cap that is set too high.

A useful one-off comparison: pick a 24-hour window before and after the write boundary is enforced, and compute the same five signals over both. The interesting output is not the percentage change; it is which signal moved first.

## Decision checklist before granting an agent write access

Run through this before adding any mutating permission to an automated identity.

- **Is the write necessary?** Can the agent emit an event and let a separate, human-reviewed process act on it? Detection and mutation are separable concerns.
- **Is the write idempotent?** If the same call arrives twice, does state change once? If not, add a deterministic key or a read-before-write check.
- **Is the identity scoped to this agent alone?** Shared service accounts couple unrelated agents through a common rate limit and a common blast radius.
- **Is there a concurrency cap?** Every downstream API has a limit. Set the cap below it and test at the cap.
- **Is the boundary in version control?** If the permission set exists only in a console, it will drift.
- **Can you enumerate the trust graph?** If you cannot list every path by which this identity can obtain credentials, you cannot reason about its blast radius.
- **Is there a kill switch?** One flag that disables all mutating calls from this agent class, testable without a deployment.

If any answer is "no," the agent should stay in the detector and router circles.

## The 30-minute action

Pick one agent that currently holds a write permission, and run the IAM policy simulation shown above against the single action you would least want it to perform — typically `sts:AssumeRole` on a broad resource, or a write action on your incident or issue system. If the result is `allowed` rather than `implicitDeny`, you have found a write boundary that is not actually enforcing anything. Attach a permissions boundary that excludes that action, re-run the simulation, and confirm the result flips. That is thirty minutes, and it converts an assumption into a verified constraint.
