# Agent versioning: the contract you forgot

The gap between the demo and the incident report is where this actually lives. After enough code that touches version evolve gets reviewed, the same failure pattern keeps showing up. Here's what actually worked, and why.

## The one-paragraph version (read this first)

When you ship an agent (a system that calls tools, chains LLM outputs, and acts on external state), changing its capabilities feels like changing code. It is not. It is changing a public API that other teams, scripts, and sometimes other agents depend on. Versioning an agent means treating its tool schemas, output contracts, and side effects as a stable interface — even though the underlying model or prompt might change weekly. The simplest accurate explanation: an agent is a function with side effects, and its signature is the set of tools it exposes plus the shape of what it returns. If you change that signature without a version boundary, every caller breaks silently. The part that trips people up is that the LLM layer makes it look like there is no interface at all — so teams skip versioning until an integration fails at 2am.

## Why this concept confuses people

Most software versioning is about code artifacts: you tag a release, you publish a package, you bump a major version when you break backward compatibility. Agents break that mental model in three ways.

First, the agent's behavior is not fully determined by its code. A prompt change, a model upgrade from `gpt-4o` to `gpt-4.1`, or a temperature tweak can alter tool-call sequences without any code diff. Your CI passes, your unit tests pass, and yet the [agent now calls](/rogue-agent-api-calls-stop-the-bleeding/) `search_flights` before `get_user_profile` instead of after — and a downstream service that expected the profile ID in the first call fails.

Second, tool schemas are often generated dynamically. A common pattern is to introspect Python functions with `inspect.signature()` and build the JSON schema at runtime. That means adding an optional parameter to a function silently changes the agent's public contract. No version bump, no changelog.

Third, the consumers of an agent are not always humans. They are other agents, cron jobs, webhook handlers, or a Zapier-style integration that a customer built. Those consumers do not read your release notes. They read the tool schema once, cache it, and assume it will not change.

A typical failure mode: a team adds a `priority` field to a `create_ticket` tool, defaulting to `"normal"`. The agent starts sending `"high"` for urgent issues. A downstream integration that only accepts `"low"`, `"medium"`, `"high"` rejects the new value with a 422. The agent retries three times, then gives up. The user sees "ticket creation failed" with no explanation. The fix is not in the agent code — it is in the version boundary that was never drawn.

## The mental model that makes it click

Think of an agent like a REST API that happens to be implemented with an LLM. The LLM is the business logic. The tools are the endpoints. The output schema is the response body.

Once you accept that analogy, versioning rules from API design apply directly:

- **Additive changes are safe.** Adding a new optional tool parameter, a new tool, or a new field in the output is backward compatible.
- **Removals and renames are breaking.** Removing a tool, renaming a parameter, or changing a type (string to integer) requires a new major version.
- **Behavioral changes are breaking even if the schema is identical.** If `search` used to return 10 results and now returns 5, that is a breaking change for any caller that paginates.

But there is a twist: agents are stateful in ways APIs are not. An agent might remember a conversation across turns. Changing the memory format — say from a list of messages to a summary string — breaks replay and debugging. That is why agent versioning needs two axes: the **interface version** (tools + output schema) and the **state version** (memory layout, checkpoint format).

A useful analogy: an agent is like a database with a schema and a migration history. You would not `ALTER TABLE` in production without a migration. You would not change a stored procedure's signature without versioning it. Agents deserve the same discipline.

## A concrete worked example

Consider a customer-support agent that exposes three tools: `get_order`, `issue_refund`, and `send_email`. It runs on a serverless platform (AWS Lambda with arm64, Python 3.11, `pydantic` 2.7 for schema validation). The agent's output is a JSON object with `action_taken`, `confidence`, and `next_steps`.

Version 1 of the agent is deployed. A partner integration calls it via HTTP and expects `action_taken` to be one of `"refunded"`, `"escalated"`, or `"no_action"`.

The team decides to improve refund logic. They add a new tool `check_refund_eligibility` and change `issue_refund` to require an `eligibility_token` parameter. They also change `action_taken` to include `"pending_review"` for cases needing manual approval.

They deploy. The partner integration receives `"pending_review"` and crashes because its enum does not include it. The agent's retry logic kicks in, but the partner's webhook returns a 400, so the agent marks the task as failed. Within an hour, 12% of refund requests are stuck.

The fix is to version the agent. Here is a minimal versioning scheme in Python using a version header and a schema registry:

```python
# agent_v2.py
from pydantic import BaseModel, Field
from typing import Literal, Optional

class AgentV2Output(BaseModel):
    action_taken: Literal["refunded", "escalated", "no_action", "pending_review"]
    confidence: float = Field(ge=0.0, le=1.0)
    next_steps: list[str]
    version: Literal["2.0"] = "2.0"

class AgentV1Output(BaseModel):
    action_taken: Literal["refunded", "escalated", "no_action"]
    confidence: float
    next_steps: list[str]
    version: Literal["1.0"] = "1.0"

# In the request handler, route based on the client's Accept-Version header
def handle_request(request):
    version = request.headers.get("Accept-Version", "1.0")
    if version == "1.0":
        return run_agent_v1(request)
    elif version == "2.0":
        return run_agent_v2(request)
    else:
        raise ValueError(f"Unsupported version: {version}")
```

The key point: both versions run side by side. New callers use `2.0`. Existing callers stay on `1.0` until they migrate. You can run both for months if needed.

On the tool side, you version the schema by including a version field in the tool definition and rejecting calls that do not match:

```javascript
// tool_schema.js — Node 20 LTS, using zod for validation
import { z } from 'zod';

const issueRefundV1 = z.object({
  orderId: z.string(),
  amount: z.number().positive(),
  reason: z.string().optional(),
});

const issueRefundV2 = z.object({
  orderId: z.string(),
  amount: z.number().positive(),
  reason: z.string().optional(),
  eligibilityToken: z.string().uuid(), // new required field
});

export function getToolSchema(version) {
  if (version === '1.0') return issueRefundV1;
  if (version === '2.0') return issueRefundV2;
  throw new Error(`Unknown tool version: ${version}`);
}
```

This pattern lets you add `eligibilityToken` without breaking v1 callers. The agent decides which tool version to use based on the interface version negotiated at request time.

## How this connects to things you already know

If you have ever worked with gRPC or protobuf, this will feel familiar. Protobuf enforces backward compatibility rules: you can add fields, but you cannot change field numbers or types. Agents need the same discipline, but without a compiler to enforce it. You have to build the enforcement yourself.

If you have worked with database migrations, the state version axis will make sense. Just as you would not change a column type without a migration, you should not change the agent's memory format without a migration path. In practice, that means storing a `state_version` alongside the memory blob and writing a transformer that upgrades old states when they are loaded.

If you have worked with API gateways, the routing-by-version pattern is identical. You can use AWS API Gateway with a custom header, or a simple reverse proxy that inspects `Accept-Version`. The infrastructure is not the hard part; the hard part is remembering to bump the version when the agent's behavior changes.

A common trap here is assuming that because the LLM is stochastic, versioning does not matter. The opposite is true: stochastic behavior makes versioning more important, because you cannot rely on tests to catch every regression. A version boundary gives you a stable reference point — you can always say "v1 did not call `check_refund_eligibility`" and reason about the delta.

## Common misconceptions, corrected

**Misconception 1: "The model is the version."** No. Changing from `gpt-4o` to `gpt-4.1` is a dependency upgrade, not an interface change. Your interface version should remain stable unless the tool schemas or output shape change. However, you should pin the model version in your deployment (e.g., `gpt-4.1-2025-04-14`) so that a silent model update does not alter behavior. If you must upgrade the model, treat it as a new minor version and run A/B tests.

**Misconception 2: "We can just tell users to update."** In practice, integrations are built by other teams or customers who have their own release cycles. A typical enterprise integration might take 3–6 months to update. If you break them, you lose trust. Versioning buys you time.

**Misconception 3: "Versioning is too much overhead for a prototype."** Start with a single version string in your output and a routing header. That is 10 lines of code. It costs almost nothing and saves you from a rewrite later. The overhead is in the discipline, not the code.

**Misconception 4: "Tools are internal, so they don't need versions."** Tools are the agent's public API. If another agent or a script calls your agent, it sees the tools. Even if tools are only used internally, other services in your own stack may depend on them. Version them.

## The advanced version (once the basics are solid)

Once you have basic versioning, you can add more sophisticated controls.

**Capability flags.** Instead of versioning the entire agent, you can version individual capabilities. For example, `refund_v2` is a capability flag that enables the new refund flow. Callers opt in by passing `capabilities: ["refund_v2"]`. This is more granular and lets you roll out features gradually. It also makes it easier to deprecate: you can turn off `refund_v1` for 90% of traffic, monitor, then turn it off entirely.

**Schema registry with compatibility checks.** Tools like Buf Schema Registry (for protobuf) or Confluent Schema Registry (for Avro) enforce compatibility rules. You can build a lightweight version for agents: a JSON file that maps tool names to versions and a CI check that fails if you remove a field or change a type without bumping the major version. This is a 50-line script in Python using `jsonschema`.

**State migration.** If you store agent memory in Redis 7.2 or a Postgres table, include a `state_version` column. When loading state, check the version and run a migration function if needed. For example, if you changed from storing a list of messages to a summary string, write a function that converts old states on the fly. This avoids a big-bang migration.

**Deprecation policy.** Define a clear timeline: when you release v2, v1 is supported for at least 6 months. After that, you can return a 410 Gone with a link to migration docs. Publish this policy so consumers can plan.

**Observability.** Tag every request with the interface version. In your logs (e.g., AWS CloudWatch), you can then see which versions are still in use. A common metric is `agent_requests_by_version`. If v1 traffic drops to 0.1% after 3 months, you can safely deprecate.

A comparison of versioning strategies:

| Strategy | Granularity | Overhead | Best for |
|----------|-------------|----------|----------|
| Full agent version (v1, v2) | Coarse | Low | Small teams, few integrations |
| Capability flags | Fine | Medium | Gradual rollouts, A/B tests |
| Schema registry + CI checks | Tool-level | High | Large orgs, many consumers |
| Model pinning only | None | Very low | Prototypes, internal tools |

## Quick reference

- **Interface version**: tools + output schema + side effects. Bump major when breaking.
- **State version**: memory format, checkpoint layout. Bump when changing storage.
- **Model version**: pin to a dated release (e.g., `gpt-4.1-2025-04-14`). Treat upgrades as minor versions.
- **Routing**: use `Accept-Version` header or a query param. Default to latest stable.
- **Deprecation**: support old versions for at least 6 months. Announce via changelog and email.
- **Enforcement**: CI check that compares tool schemas against a baseline. Fail on breaking changes without a version bump.
- **Observability**: log `version` with every request. Track adoption.

## Further reading worth your time

- The protobuf language guide on backward compatibility rules (historical, still relevant): https://protobuf.dev/programming-guides/proto3/#updating
- AWS Lambda versioning and aliases documentation: https://docs.aws.amazon.com/lambda/latest/dg/configuration-versions.html
- Pydantic 2.7 docs on schema generation: https://docs.pydantic.dev/2.7/concepts/json_schema/
- The Twelve-Factor App, especially the "Backing services" factor for state management: https://12factor.net/backing-services

## Frequently Asked Questions

**How do I version an agent's tools without breaking existing callers?**
Use a version header or capability flag. Run both tool versions side by side. The agent selects the tool version based on the negotiated interface version. Add new required parameters only in a new major version. Keep old versions for at least 6 months.

**Why does my agent's output schema change break integrations?**
Because integrations often validate against a fixed schema. Adding a new enum value like `pending_review` can cause a 422 if the consumer's validator rejects unknown values. Always treat output schema changes as breaking unless you are only adding optional fields that consumers ignore.

**What is the difference between agent versioning and model versioning?**
Model versioning is about pinning the LLM to a specific release (e.g., `gpt-4.1-2025-04-14`). Agent versioning is about the interface: tools, output shape, and state format. You can upgrade the model without changing the agent version if the interface remains compatible. But you should test model upgrades carefully because behavior can shift.

**When should I bump the major version of an agent?**
Bump major when you remove or rename a tool, change a parameter type, change the output schema in a non-additive way, or alter side effects (e.g., now sends an email when it did not before). If you are only adding optional fields or new tools, a minor version is enough.

## The one thing to do in the next 30 minutes

Open your agent's main handler file (e.g., `handler.py` or `agent.js`) and add a `version` field to the output schema. Then add a routing check at the top of the handler that reads an `Accept-Version` header and defaults to `"1.0"`. This is a 15-line change. Commit it. It will not fix everything, but it establishes the boundary. The next time you change a tool or output shape, you will have a place to put the new version — and a way to keep your existing users working.


---

### About this article

**Written by:** [Kubai Kevin](/about/) — software developer based in Nairobi, Kenya, with 10+ years building production systems in fintech and AI.

**How this article was produced:** This site uses an automated LLM pipeline designed and maintained by the author. Topics are selected from real production experience. Drafts pass automated quality gates (minimum length, uniqueness, concrete metrics, versioned tools, code samples, absence of filler). Individual line-by-line human editing is not performed on every post before publication. Specific numbers, benchmarks and cost figures are illustrative; verify them against current official documentation before production use.

**Corrections:** Report errors via the [contact page](/contact/). Corrections are applied promptly.

**Last generated:** September 2026
