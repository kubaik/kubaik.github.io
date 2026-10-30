# AI-First vs Human-First Incident Response

## The core distinction

Two architectural patterns dominate AI-assisted incident response. They differ in exactly one place: who is allowed to take a mutating action without a human in the loop.

**Human-first ops with AI triage.** The AI reads alerts, correlates them, ranks probable causes, and drafts a timeline. A human acknowledges, decides, and executes every remediation. The AI never pages anyone, never rolls back a deployment, and never closes an incident.

**AI-first autonomous response.** An orchestrator ingests the alert, gathers context, selects a remediation from a policy set, executes it, verifies recovery, and closes the incident. Humans are paged only when no policy matches, when the action is classified irreversible, or when verification fails.

Everything else — the alert router, the chat ops layer, the observability stack — is largely shared between the two. The decision is not "which vendor" but "which action boundary."

That boundary determines your failure modes. A human-first system fails by being slow. An AI-first system fails by being confidently wrong at machine speed. Both are real, and they require different mitigations.

## Human-first architecture

A typical human-first stack:

- **Alert routing:** a paging service with on-call schedules and escalation policies.
- **Incident coordination:** a chat-ops bot that opens a channel, assigns roles, and appends events to a timeline.
- **Triage layer:** an AI assistant that consumes historical incident data and surfaces ranked hypotheses plus candidate runbooks.

The triage layer is the only novel component. It is usually a retrieval system over past incidents plus a summarization model, not an autonomous agent. It has read access to observability APIs and write access to one thing: the incident channel.

A minimal webhook handler that opens an incident channel looks like this:

```javascript
// Webhook handler: open an incident channel from a paging event
const axios = require('axios');
const INCIDENT_TOKEN = process.env.INCIDENT_TOKEN;

module.exports = async (req, res) => {
  const { incident_key, title, severity, service } = req.body;

  if (!incident_key || !severity) {
    return res.status(400).json({ error: 'missing incident_key or severity' });
  }

  try {
    const response = await axios.post(
      'https://api.example-incident-tool.com/v1/incidents',
      {
        name: title || `Incident ${incident_key}`,
        severity,
        service,
        status: 'investigating'
      },
      {
        headers: { Authorization: `Bearer ${INCIDENT_TOKEN}` },
        timeout: 5000
      }
    );
    return res.json({ id: response.data.id });
  } catch (err) {
    // Never fail closed on incident creation: log and let the pager retry.
    console.error('incident create failed', err.message);
    return res.status(502).json({ error: 'upstream failure' });
  }
};
```

Two details matter more than they look. First, the handler returns 502 rather than 200 on failure, so the pager's retry policy actually fires. Second, it validates input before calling upstream — a malformed payload that reaches the incident tool can create a duplicate channel that nobody closes.

### Where it works well

**Mixed-layer incidents.** When a symptom spans infrastructure and application layers, correlation is genuinely hard. A human who can see that a pod restart and a garbage-collection pause share a timestamp will beat a policy engine that only knows "restart → roll back."

**Noisy alert sources.** When one underlying fault produces dozens of alerts in minutes, grouping them into a single incident is the highest-value thing triage does. This is a clustering problem, and it is well within what a retrieval-plus-ranking system can do.

**Regulated or high-blast-radius environments.** If every production change requires a named human approver, an autonomous remediation path is not a shortcut — it is a compliance problem.

### Where it breaks

**Latency floor.** The system cannot resolve anything faster than a human can be woken, read context, and act. For a fault that compounds — a connection pool exhaustion, a retry storm — that floor is the whole cost.

**Playbook drift.** Static runbooks go stale when manifests, service names, or deploy tooling change. The AI will confidently suggest a remediation step that references a resource that no longer exists. Mitigation: version runbooks alongside the code they describe and fail the triage response if the referenced resource is absent.

**Unvalidated hypotheses.** A ranked list of probable causes is not the same as a correct one. If the list is long, it adds reading time rather than removing it. Track how often the top-ranked hypothesis is confirmed in the postmortem; if that number is low, the ranking is decoration.

## AI-first architecture

A typical AI-first stack:

- **Alert source:** paging service with routing rules that fan out to an orchestrator.
- **Orchestrator:** a service that holds remediation policies and executes them.
- **Remediation engine:** declarative deploy tooling (GitOps sync, infrastructure-as-code apply) plus a feature-flag or traffic-shifting API.
- **Verification:** a health check that decides whether the action worked.
- **Human gate:** a mandatory review window for actions classified as irreversible.

The orchestrator is event-driven. An alert triggers a function that fetches context, evaluates policy, and either executes or escalates.

```python
# Orchestrator: evaluate policy, execute, verify, escalate on failure
import os
import time
import requests

class RemediationAgent:
    IRREVERSIBLE = {"delete_database", "drop_table", "rotate_root_credentials"}

    def __init__(self):
        self.orchestrator_token = os.getenv("ORCHESTRATOR_TOKEN")
        self.deploy_token = os.getenv("DEPLOY_TOKEN")
        self.verify_url = os.getenv("VERIFY_URL")
        self.max_verify_seconds = int(os.getenv("MAX_VERIFY_SECONDS", "300"))

    def handle(self, alert):
        action = self.select_action(alert)
        if action is None:
            return self.escalate(alert, reason="no matching policy")

        if action["name"] in self.IRREVERSIBLE:
            return self.request_human_gate(alert, action)

        result = self.execute(action)
        if not result["ok"]:
            return self.escalate(alert, reason=result["error"])

        if self.verify():
            return self.close(alert, action)

        # Verification failed: do not retry the same action blindly.
        return self.escalate(alert, reason="verification failed")

    def select_action(self, alert):
        # Policies are keyed by alert signature, not free-text reasoning.
        policies = {
            "deployment_5xx_spike": {"name": "rollback_deployment"},
            "hpa_max_replicas": {"name": "raise_replica_ceiling"},
            "feature_flag_error_rate": {"name": "disable_flag"},
        }
        return policies.get(alert.get("signature"))

    def execute(self, action):
        try:
            r = requests.post(
                "https://deploy.example.com/api/v1/actions",
                json=action,
                headers={"Authorization": f"Bearer {self.deploy_token}"},
                timeout=30,
            )
            r.raise_for_status()
            return {"ok": True}
        except requests.RequestException as e:
            return {"ok": False, "error": str(e)}

    def verify(self):
        deadline = time.time() + self.max_verify_seconds
        while time.time() < deadline:
            try:
                r = requests.get(self.verify_url, timeout=5)
                if r.status_code == 200 and r.json().get("healthy"):
                    return True
            except requests.RequestException:
                pass
            time.sleep(15)
        return False

    def escalate(self, alert, reason):
        requests.post(
            "https://pager.example.com/v2/enqueue",
            json={"event_action": "trigger", "payload": {"summary": f"{alert['id']}: {reason}"}},
            timeout=10,
        )

    def request_human_gate(self, alert, action):
        self.escalate(alert, reason=f"human approval required for {action['name']}")

    def close(self, alert, action):
        requests.post(
            "https://orchestrator.example.com/v1/incidents/close",
            json={"alert_id": alert["id"], "resolved_by": action["name"]},
            headers={"Authorization": f"Bearer {self.orchestrator_token}"},
            timeout=10,
        )
```

Three design choices in that code carry most of the safety:

**Policy selection is keyed by alert signature, not by model reasoning.** A free-text "what should I do?" prompt is the single largest source of wrong actions. Map a known signature to a known action; escalate everything else.

**Irreversible actions are enumerated, not inferred.** The `IRREVERSIBLE` set is explicit and reviewed. A model that decides for itself whether an action is reversible will eventually be wrong.

**Verification failure escalates rather than retries.** Retrying the same action after a failed health check is how a single bad deploy becomes a rollback loop.

### Where it works well

**Time-critical, well-understood faults.** If the signature is known and the remediation is a rollback to a known-good revision, the machine is faster than any human, and speed is the entire value.

**Repetitive faults.** Cache stampedes, scheduled job timeouts, and autoscaling ceiling hits recur with the same shape. These are the highest-confidence automation candidates.

**Off-hours coverage.** The gap between "alert fires" and "human is awake" is where autonomous remediation earns its keep — provided the policy set is small and the verification is strict.

### Where it breaks

**App-layer faults are not rollback-shaped.** A race condition, a feature-flag misconfiguration, or a data-dependent bug does not resolve by reverting a deployment. An agent that treats every 5xx spike as a deploy problem will roll back healthy releases and make things worse.

**Silent misconfiguration.** If the orchestrator's credentials expire, or a policy references a renamed resource, the failure is quiet. The alert fires, the agent does nothing, and nobody notices until the next incident review.

**The escalation paradox.** Every irreversible action still requires a human. If most of your incidents touch irreversible paths, the net reduction in pages is small, and you have added a system to maintain.

**Automation-induced incidents.** An agent with broad write access is itself a production risk. A bug in the remediation path can cause the outage it was meant to fix.

## A worked failure analysis

Consider a service that starts returning 5xx after a config change. Walk the same scenario through both architectures.

**Human-first.** Alert fires. Triage groups the 5xx spike with a simultaneous latency increase and a config-change event, and ranks "recent config change" first. On-call confirms, reverts the config, verifies. Elapsed time: minutes to tens of minutes, dominated by human wake-up and confirmation.

**AI-first, policy keyed to the wrong signature.** The alert signature is `deployment_5xx_spike`, which maps to `rollback_deployment`. The agent rolls back the deployment. But the fault was the config change, which is not part of the deployment artifact. The rollback succeeds, the health check still fails, verification times out, and the agent escalates — now with a rollback that removed unrelated fixes and a confusing timeline.

The lesson is not "autonomy is bad." It is that **alert signature and root cause are different things**, and a policy engine that conflates them will take confident wrong actions. The mitigation is to make the verification step authoritative: if health does not recover, escalate immediately and do not attempt a second action.

## How to measure both stacks

Do not trust published benchmarks for this. Instrument your own system. For each incident record:

- **Time to acknowledge:** timestamp of first alert to timestamp of first human or agent acknowledgment.
- **Time to resolution:** first alert to health check passing.
- **Action attribution:** which action resolved it, and whether the postmortem confirmed that action was causal.
- **Escalation rate:** fraction of incidents that reached a human, and why.
- **False action rate:** actions taken that the postmortem judged unnecessary or harmful.
- **Repeat rate:** fraction of incidents matching a previously seen signature.

The two numbers that decide the architecture question are **repeat rate** and **false action rate**. High repeat rate plus low false action rate is the only combination that justifies autonomy. Low repeat rate means your policy set will rarely match, and you will pay the maintenance cost for nothing.

To measure false action rate you need postmortem discipline: every incident must record whether the automated action was causal. Without that, you are optimizing on speed alone.

## Cost modeling without invented numbers

Cost comparisons in this space are usually fabricated. Build yours from three inputs you can actually observe:

1. **Per-seat or per-incident SaaS pricing** — read your contract, not a blog post.
2. **Compute cost of the orchestration path** — count invocations per incident, multiply by your provider's published per-invocation price, and add the verification polling calls.
3. **Human time** — escalation rate × average minutes per escalation × your loaded hourly cost.

A worked illustration with stated assumptions: suppose 200 incidents per month, an escalation rate of 30%, and 45 minutes of senior engineer time per escalation at a loaded cost of $120/hour. Human escalation cost is 200 × 0.30 × 0.75 hours × $120 = $5,400 per month. If autonomy cuts the escalation rate to 15%, the saving is $2,700 per month — before subtracting the engineering time to build and maintain the policy layer. If that layer costs one engineer-day per month to maintain, the saving is smaller than it looks.

These figures are illustrative. Substitute your own incident count, escalation rate, and loaded hourly cost. The arithmetic is the point, not the numbers.

## Decision checklist

Work through these in order. The first "no" is usually decisive.

1. **Do you have at least 20 incidents with a recorded signature and a confirmed causal action?** Without this history, you cannot write reliable policies.
2. **Is the repeat rate high?** If fewer than roughly half of incidents match a previously seen signature, autonomy will rarely fire.
3. **Are the recurring incidents infrastructure-shaped?** Rollback-shaped faults automate well; race conditions and data bugs do not.
4. **Can you enumerate irreversible actions exhaustively?** If you cannot list them, you cannot gate them.
5. **Does a health check exist that reliably distinguishes recovered from not-recovered?** Weak verification turns autonomy into guesswork.
6. **Is there an owner for the policy layer?** Unmaintained policies drift and fire on stale assumptions.
7. **Does the team accept that the agent will sometimes be wrong in production?** If not, the human gate will be applied to everything, and the automation buys nothing.

If you answer yes to all seven, an AI-first design is defensible. If you answer no to any of the first five, build the human-first stack first and collect the incident history that would let you revisit the question.

## FAQ

**Does AI triage replace on-call?** No. It reduces the time to form a hypothesis. Someone still has to confirm and act.

**Can I run both patterns at once?** Yes, and it is common: autonomy for a small set of high-confidence signatures, human-first for everything else. The risk is that two paths create two timelines that disagree during an incident. Keep one system of record.

**What is the biggest cause of autonomous remediation failures?** Policy selection keyed to alert signature rather than root cause, combined with weak verification that cannot tell a fixed system from a still-broken one.

**How do I roll autonomy back safely?** Start with actions that are trivially reversible and observable, keep the human gate on everything else, and track false action rate from the first incident. If it rises, disable the policy rather than tuning it live during an outage.

## Do this in the next 30 minutes

Open your incident tracker, filter to the last 90 days, and tag each incident with two fields: a normalized signature (for example `deployment_5xx_spike`) and whether the resolving action was confirmed causal in the postmortem. Then compute the repeat rate: incidents whose signature has appeared more than once, divided by total incidents. If that number is below half, your next investment is incident history and runbook hygiene, not an autonomous agent.
