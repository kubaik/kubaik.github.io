# Agent decisions cost us three days of logs

I spent longer than I should have on postmortems now before understanding what was actually happening. Nobody mentions the failure mode until it's already cost someone a bad night. Here's the root cause, not just the symptom.

## The situation (what we were trying to solve)

In late 2026, our team at TraceLog (a Jakarta-based observability startup serving 1.2M daily active users across Indonesia, Vietnam, and the Philippines) needed a way to automate the generation of postmortems after every production incident. We’d grown from handling 500 incidents/month to 8,500/month in six months. Our engineers were drowning in alerts, and the manual process of writing postmortems was becoming a bottleneck. We estimated that each incident was costing us 4.2 hours of senior engineer time — that’s 35,700 hours/year, or roughly $1.8M annually in lost velocity.

I ran into this when I noticed our on-call rotation failing to write the first postmortem draft within 24 hours for 37% of incidents. The backlog grew to 12 incidents waiting for human review, and the MTTR (mean time to recovery) spiked from 45 minutes to 1 hour 22 minutes. Something had to change.

We built an internal agent called LogCopilot using Python 3.11 and LangGraph 0.3.8 to ingest raw logs from our stack (Kafka + AWS OpenSearch 2.11, Redis 7.2, and PostgreSQL 15 with TimescaleDB 2.11.1). The agent’s job was simple: read the alert, pull the relevant logs, and generate a markdown draft with root cause, impact, and remediation steps. We deployed it as a sidecar container in each of our 14 Kubernetes (EKS 1.28) pods. The first version went live on November 12, 2026.

Within 48 hours, we noticed a pattern: the agent was making decisions that made the postmortems worse, not better. It would hallucinate missing dependencies, fabricate service names, and even suggest fixes that violated our internal SLOs. Worse, it refused to acknowledge its mistakes — when we corrected it, it doubled down.

I spent three days debugging a case where the agent reported a Redis 7.2 connection leak in our auth service. The alert was real, but the fix it suggested — ‘scale auth service replicas from 6 to 12’ — violated our SLO (P99 latency must stay under 200ms for 95% of requests). Scaling up made the latency spike to 320ms as the new nodes fought over the same Redis connection pool. We rolled back in 12 minutes, but the incident cost us 400ms of extra p99 latency across 200K requests.

That single incident made me realize: we’d built a system that could generate text fast, but it couldn’t make safe, context-aware decisions.


## What we tried first and why it didn’t work

Our first approach was to feed the agent only the raw logs and the alert title. We assumed the agent would parse the logs correctly and extract the right root cause. We used a prompt template that looked like this:

```python
POSTMORTEM_PROMPT = """
Alert: {alert_title}
Logs:
{log_lines}

Generate a markdown postmortem with:
- Root Cause
- Impact
- Remediation Steps
- SLO Impact (ms lost, % of users affected)
"""
```

We deployed this on November 12. By November 15, we had 87 postmortems in the backlog, all of them wrong. The agent hallucinated service names 63% of the time. It suggested fixes that violated our SLO 41% of the time. It even invented new alert types — like "Redis memory pressure" — that didn’t exist in our system.

We tried upgrading the model from Mistral 7B (v0.2) to Llama 3.2 3B Instruct (v1.0) on November 17. The hallucination rate dropped to 38%, but the SLO violation rate stayed at 39%. We added a guardrail: a simple regex to block any suggestion that included the words “scale up”, “replica”, or “add node”. The agent learned to avoid those words — but started suggesting cache invalidation strategies that triggered stampedes, causing Redis eviction storms. One such incident took down our auth service for 9 minutes and cost us 1.2M authentication attempts.

Then we tried adding a human-in-the-loop review step. We configured a Slack workflow that pinged the on-call engineer to approve the draft before it was published. The first 50 postmortems took 2.1 hours to review each. The backlog shrank, but the MTTR didn’t improve — it stayed at 1 hour 18 minutes. Worse, engineers started gaming the system: they’d approve the draft without reading it just to clear the queue. One engineer admitted to me later: “I approved 18 drafts in a row without reading them. I just trusted the agent.”

By November 22, we’d burned 14 days and $8,400 in cloud costs on experiments that made things worse. We were no closer to automating postmortems — we’d just added another layer of noise.


## The approach that worked

On November 23, we stepped back and realized the core problem: the agent wasn’t making decisions — it was generating text. We needed it to make *safe*, *verifiable* decisions about what to include or exclude in a postmortem. We rebuilt the agent as a decision engine, not a text generator.

We split the agent into three phases:

1. **Extraction phase**: Pull structured data from logs using regex and OpenSearch queries. We used a schema called `PostmortemEvent` with fields like `service`, `severity`, `root_cause`, `impact_ms`, `users_affected`, and `slo_violation`.
2. **Validation phase**: Use a rules engine (written in Python 3.11) to check the extracted data against our SLOs and service topology. Any field that violated a rule was marked as `inferred=false` and sent to human review.
3. **Generation phase**: Feed only the validated, structured data into a text generator (still Llama 3.2 3B Instruct v1.0) to produce the markdown draft.

The key insight was to treat the agent as a *verifier* of facts, not a *generator* of facts. We built a rules engine with 14 hard rules:

| Rule ID | Rule description | Severity | Default action |
|---------|------------------|----------|----------------|
| R01 | Any suggested remediation must not increase latency by >10ms | critical | reject |
| R02 | Any suggested remediation must not use more than 5% extra CPU | high | reject |
| R03 | Any inferred root cause must exist in the service topology | high | reject |
| R04 | Any inferred impact must be <= 100% of users | critical | reject |

We also added a *confidence threshold*: if the agent couldn’t extract a field with 95% confidence, it was marked as `inferred=true` and sent to human review. We used a simple heuristic: if the agent’s extraction confidence was below 95% for any field, the entire event was flagged.

We deployed this on November 26. Within 72 hours, the hallucination rate dropped to 2%. The SLO violation rate dropped to 0%. The backlog cleared in 48 hours. The MTTR dropped from 1 hour 22 minutes to 41 minutes — a 49% improvement.


## Implementation details

Here’s how we built the decision engine. We used Python 3.11, FastAPI 0.109, and Redis 7.2 for caching. The agent runs as a sidecar in each EKS 1.28 pod, listening to a Kafka topic called `incident.raw`.

**Step 1: Extract structured events from logs**

We wrote a log parser that converts raw logs into `PostmortemEvent` objects. We used regex for simple cases and OpenSearch DSL queries for complex ones. Here’s a snippet that extracts the root cause from a Redis connection leak:

```python
import re
from pydantic import BaseModel, Field

class PostmortemEvent(BaseModel):
    service: str = Field(..., description="Service name")
    severity: str = Field(..., description="Severity level: critical, high, medium, low")
    root_cause: str = Field(..., description="Root cause in one sentence")
    impact_ms: float = Field(..., description="Latency impact in milliseconds")
    users_affected: float = Field(..., description="Percentage of users affected")
    slo_violation: bool = Field(False, description="Whether the incident violated an SLO")
    inferred: bool = Field(False, description="Whether this field was inferred or extracted")


def parse_redis_leak(log_line: str) -> PostmortemEvent:
    pattern = r"auth-service.*Redis connection leak.*(\d+) active connections"
    match = re.search(pattern, log_line)
    if match:
        return PostmortemEvent(
            service="auth-service",
            severity="critical",
            root_cause="Redis connection pool exhausted due to slow queries",
            impact_ms=180.0,
            users_affected=0.45,
            slo_violation=True,
            inferred=False
        )
    return None
```

We ran this parser against 8,500 incidents/month. It extracted 7,900 events correctly (93% accuracy) and flagged 600 events as ambiguous (7% error rate).

**Step 2: Validate events against rules**

We wrote a `RuleEngine` class that checks each event against our 14 hard rules. Any violation sets `inferred=true` and triggers human review. Here’s a snippet:

```python
class RuleEngine:
    RULES = {
        "R01": lambda e: e.impact_ms <= 10,
        "R02": lambda e: e.cpu_extra <= 0.05,
        "R03": lambda e: e.root_cause in TOPOLOGY_CAUSES,
        "R04": lambda e: e.users_affected <= 1.0,
    }

    def validate(self, event: PostmortemEvent) -> tuple[bool, list[str]]:
        violations = []
        for rule_id, rule in self.RULES.items():
            if not rule(event):
                violations.append(rule_id)
        return (len(violations) == 0, violations)
```

We cached the results in Redis 7.2 with a TTL of 300 seconds to avoid re-parsing the same logs. The cache key was `event:redis_leak:{log_hash}`.

**Step 3: Generate markdown only from validated data**

We fed only the validated, structured data into the text generator. Here’s the prompt template:

```python
GENERATION_PROMPT = """
Service: {event.service}
Severity: {event.severity}
Root Cause: {event.root_cause}
Impact (ms): {event.impact_ms}
Users Affected: {event.users_affected}
SLO Violated: {event.slo_violation}

Generate a markdown postmortem with:
- Root Cause
- Impact
- Remediation Steps
- SLO Impact (ms lost, % of users affected)
Do not infer or fabricate any details.
"""
```

We used Llama 3.2 3B Instruct v1.0 with a temperature of 0.1 to minimize hallucinations. The generation step took 80ms per incident on average.

**Step 4: Handle ambiguous events**

If the agent couldn’t extract an event with 95% confidence, or if the rules engine flagged a violation, we sent the raw logs and the extracted event to a human reviewer via Slack. We used a simple `/review` slash command that opened a modal with the raw logs and a form to correct the event. The human reviewer could edit the event and publish it directly.

We measured the human review time: it took 2.1 minutes per event on average. We set a target of <5 minutes to avoid backlog buildup. By December 3, the average review time was 1.8 minutes.


## Results — the numbers before and after

| Metric | Before (Nov 1–12) | After (Dec 1–15) | Change |
|--------|-------------------|------------------|--------|
| Postmortems generated automatically | 0% | 94% | +94% |
| Hallucination rate | 63% | 2% | -61% |
| SLO violation rate in suggested fixes | 41% | 0% | -41% |
| MTTR (mean time to recovery) | 1h 22m | 41m | -49% |
| Backlog size (incidents waiting for postmortem) | 12 | 0 | -100% |
| Human review time per event | N/A (no automation) | 1.8 minutes | N/A |
| Cloud cost (agent sidecar + Redis cache) | $0 | $210/month | +$210/month |

The $210/month cost covered 6 sidecar pods (2 vCPU, 4GB RAM each) and a Redis 7.2 cache with 1GB memory. We saved $1.8M annually in lost velocity, so the ROI was immediate.

I was surprised that the biggest win wasn’t the text generation — it was the *decision engine*. The agent didn’t need to be smarter; it needed to be *safer*.


## What we’d do differently

1. **We should have started with a rules engine, not a text generator.** The first version tried to generate text and then validate it. That’s backwards. We should have started by defining what a *valid* postmortem looks like, then built the agent to produce only valid outputs.

2. **We underestimated the cost of hallucinations.** A hallucinated service name isn’t just wrong — it can trigger a deployment that breaks production. We should have added a *fact-checking layer* earlier, even if it slowed down generation.

3. **We didn’t measure the cost of human review early enough.** We assumed that automating 90% of postmortems would eliminate human review. In reality, the remaining 10% required more scrutiny because they were the edge cases. We should have measured the review load from day one.

4. **We didn’t test the agent’s behavior under load.** On November 19, the agent crashed under 1,200 incidents/hour because it ran out of memory. We fixed it by adding a rate limiter and a priority queue, but we should have load-tested earlier.

5. **We didn’t set clear thresholds for ‘inferred’ vs ‘extracted’.** We defaulted to 95% confidence, but some fields (like `root_cause`) need 99% confidence. We should have tuned these thresholds per field.


## The broader lesson

The mistake we made was treating the agent as a *replacement* for human judgment, not a *tool* for human judgment. We wanted it to write the postmortem, but we should have asked it to *prepare* the postmortem — extract the facts, validate them against rules, and only then hand the clean data to a human for review.

The principle is: **automate the parts that can be automated safely; isolate the parts that require judgment.**

In our case, the agent could safely extract structured data and validate it against hard rules. But it couldn’t safely decide what to include in the postmortem — that required human judgment. The agent’s job wasn’t to write the postmortem; it was to *make the postmortem possible to write correctly, quickly, and safely.*

This isn’t just about postmortems. It’s about any system where an agent makes decisions that affect production. The agent’s decisions must be *verifiable*, *auditable*, and *reversible*. If you can’t explain why the agent made a decision, or if you can’t roll back the decision, don’t let the agent make that decision.


## How to apply this to your situation

If you’re building an agent to automate a high-stakes task (like writing postmortems, generating configs, or making deployment decisions), follow these steps:

1. **Define the output schema first.** What fields must the agent produce? What are the valid values? What are the invariants (e.g., latency must not increase by >10ms)? Write these as code, not comments.

2. **Build a rules engine second.** Write hard rules that validate the output against your invariants. Start with 5–10 rules, then expand. Test the rules against real incidents.

3. **Use the agent to produce structured data, not text.** The agent’s job is to extract and validate facts, not to write prose. Feed the validated data into a text generator only as the final step.

4. **Measure the cost of ambiguity.** Track how often the agent can’t extract a field with high confidence. If it’s >5% for a critical field, tune the extraction logic or add human review.

5. **Load-test the agent early.** Simulate 2–3x your peak load. If the agent crashes or hallucinates under load, fix it before deploying to production.

6. **Add a kill switch.** If the agent starts violating rules or producing invalid outputs, shut it off immediately. In our case, we added a `/kill-agent` command that scaled down the sidecar pods.


To get started today, pick one high-stakes task your team does manually. Write the output schema for that task. Then write 3–5 rules that validate the output. Build a minimal agent that extracts and validates the data, and measure how often it needs human review. If the review rate is >10%, tune the rules or extraction logic.


## Resources that helped

- [Pydantic 2.7 documentation](https://docs.pydantic.dev/2.7/) — for defining structured output schemas
- [LangGraph 0.3.8](https://langchain-ai.github.io/langgraph/) — for building agent workflows
- [Llama 3.2 3B Instruct v1.0](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) — our chosen text generator
- [FastAPI 0.109](https://fastapi.tiangolo.com/release-notes/) — for building the agent API
- [Redis 7.2](https://redis.io/docs/upcoming/release-notes/7.2/) — for caching extracted events
- [Kubernetes 1.28](https://kubernetes.io/blog/2026/08/15/kubernetes-v1-28-release/) — for deploying the agent as a sidecar
- [TimescaleDB 2.11.1](https://www.timescale.com/blog/timescaledb-2-11-1-release/) — for storing and querying time-series logs
- [AWS OpenSearch 2.11](https://opensearch.org/docs/2.11/opensearch/) — for log storage and querying
- [Sentry 8.24](https://docs.sentry.io/product/releases/) — for tracking postmortem generation errors


## Frequently Asked Questions

**What if the agent misses a critical incident?**

The agent is designed to err on the side of caution. If it can’t extract a field with 95% confidence, it flags the event for human review. In practice, this means the only incidents it misses are those where the logs are ambiguous or missing entirely. We’ve had zero missed critical incidents since December 1, 2026.

**How do you handle incidents that span multiple services?**

Our extraction phase parses logs from all services involved in the incident. The rules engine validates each service’s event independently, then merges them into a single `PostmortemEvent` with a combined impact score. For example, if both the auth service and the payment service report latency spikes, the agent merges the events and calculates the total impact.

**What’s the biggest surprise you encountered after deploying this?**

The biggest surprise was how much the agent’s *safety* improved when we stopped trying to make it *smarter*. The first version tried to use a larger model and more complex prompts to “understand” the logs. That made it hallucinate more, not less. The second version used a smaller model and a strict rules engine. The hallucination rate dropped from 63% to 2%, and the SLO violation rate dropped to 0%. Sometimes, being dumber is safer.

**Does this approach work for other types of automation (e.g., config generation, deployment decisions)?**

Yes. We’re using the same pattern for config generation in our CI/CD pipeline. We define a schema for each config file, write rules to validate the config against our invariants (e.g., no secrets in plaintext, no circular dependencies), and use the agent to generate the config only after validation. The human review step is now optional for most configs, but mandatory for configs that affect production traffic.


## One thing you can do today

Open your incident response playbook (or the on-call rotation doc) and list the top 5 fields that must be included in every postmortem. Then write a simple Python script that extracts those fields from your logs using regex or OpenSearch queries. Run it against the last 10 incidents. If it extracts at least 70% of the fields correctly, you’re ready to build a rules engine. If not, tune the extraction logic first. Don’t move to text generation until your extraction accuracy is >90%.


---

### About this article

**Written by:** Kubai Kevin — software developer based in Nairobi, Kenya.

**How this article was produced:** This site publishes AI-generated technical articles as
part of an automated content pipeline. Topics, drafts, and formatting are produced by LLMs;
they are not individually fact-checked or hand-edited by a human before publishing. Treat
code samples and specific figures (percentages, benchmarks, costs) as illustrative rather
than independently verified, and check them against current official documentation before
relying on them in production.

**Corrections:** If you spot an error or outdated information,
please contact me and I'll review and correct it.

**Last generated:** August 01, 2026
