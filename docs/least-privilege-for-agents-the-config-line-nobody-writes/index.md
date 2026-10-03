# Least privilege for agents: the config line nobody writes

Most guidance on least privilege for AI agents stops at "grant only what the agent needs." That advice is correct and almost useless on its own, because in most agent frameworks the default is the opposite: every agent in the process gets access to every tool in the registry. The missing piece is not a principle but a config line — the one that makes authorization a runtime decision instead of a manifest-wide default.

This article covers why permissive defaults fail at scale, what a context-aware policy layer looks like, a working implementation skeleton, and the failure modes that show up only after you ship it.

## Why permissive tool defaults fail past a dozen tools

Typical agent frameworks — LangChain, LlamaIndex, crewAI, AutoGen and similar — register tools into a shared pool and let the planner choose. Convenience wins in demos: list functions in a manifest, hand the manifest to the agent, done. Three things break as the tool set grows.

**Search space blows up the planning loop.** When an agent enumerates tool signatures at the start of each turn, the enumeration itself is a cost. If `list_tools()` is a network round-trip to a registry service, that latency is paid per turn and again per retry. A typical failure mode is an agent that exhausts its turn timeout budget scanning tool metadata rather than doing work. The fix is not a smarter planner; it is shrinking the candidate set the planner ever sees, which is exactly what a policy check does.

**Widening blast radius passes review silently.** A junior engineer adds a tool with a broad scope. The change is one line in a manifest. Nothing in the diff shows that the agent's effective capabilities just grew from "read analytics" to "read analytics and delete customer records." Policy engines that evaluate at call time surface this because the deny shows up in test traffic; manifests do not.

**Auditors ask a question manifests cannot answer.** SOC 2, ISO 27001 and PCI-DSS reviews expect explicit, documented authorization for sensitive actions. A YAML file granting `*_all` to every role is not that. What satisfies the question is a record of every denied call, tied to a policy version, queryable after the fact.

The deeper mismatch is that static manifests describe coarse-grained permissions, while agent behavior is dynamic. An agent may start a conversation in read-only mode and pivot to write operations after a user confirms an action. A static allow-list cannot express "read always, write only after confirmation," but a runtime policy can.

## What a runtime policy layer actually is

A policy layer sits between the agent's decision to call a tool and the tool's execution. It answers one question per call: given this agent's role, this tool's metadata, and the current conversation context, is this call allowed?

The design that holds up in practice borrows from two established systems: AWS IAM's action/resource model and Open Policy Agent (OPA) as the evaluation engine. Each tool carries metadata:

- `resource_type` — the category of thing being touched (e.g. `analytics_report`, `customer_record`)
- `actions` — what the tool can do (`read`, `export_csv`, `delete`)
- `sensitive_data` — whether the tool touches regulated or personal data
- `cost_tier` — rough cost class, so budget rules can gate expensive calls

The policy defines roles (`read_only`, `write`, `admin`) and conditions. At call time the evaluator receives the role, the tool metadata, and a context object containing things like `allow_reads`, `daily_budget_remaining`, and `high_risk`. If any condition fails, the call is denied before the tool's HTTP client fires. A hard 403 is better than a soft warning here: it stops retries, stops latency spend, and gives the agent a clear signal to replan.

Two properties matter more than the specific engine:

1. **Deny by default.** `default allow = false` is the only safe starting point.
2. **The context is scoped to the conversation.** Role and budget flags must not leak across sessions. This is the single most common correctness bug in these systems, and it is covered in the failure modes below.

## A worked example: gating an expensive export

Before writing code, it helps to trace one call through the system.

Assume an agent with role `write`. A user asks it to export a year of analytics data to CSV. The tool `analytics` has `resource_type: analytics_report`, `actions: [read, export_csv]`, `cost_tier: medium`, and `sensitive_data: true`. Context for this conversation is `{allow_reads: true, daily_budget_remaining: 320, high_risk: false}`.

The evaluator checks, in order:

1. Is `write` permitted to call `analytics` at all? The policy maps `write` to a set of resource types that includes `analytics_report`. Pass.
2. Is the requested action `export_csv` in the tool's declared actions? Yes. Pass.
3. Does the `export_csv` rule require budget? The rule states `daily_budget_remaining > 500`. Current value is 320. Fail.

Result: 403 with a machine-readable reason. The agent receives the denial, sees that budget is the constraint, and can tell the user the export is unavailable until the budget resets — rather than silently retrying four times and timing out.

Now change one input: `daily_budget_remaining: 750`. The same call passes. This is the behavior a static manifest cannot produce, because the decision depends on state that does not exist at manifest-write time.

The arithmetic here is worth stating plainly: if a single export costs an illustrative $0.40 in compute and API calls, a budget of 500 units at one unit per dollar permits roughly 1,250 exports before the gate closes. The exact numbers depend on your pricing; the point is that the gate is a number you choose and can audit, not a hope.

## Implementation skeleton

The following is a minimal working shape. It uses Python 3.11, FastAPI for the service layer, and OPA as the policy engine. Treat it as a scaffold to adapt, not a drop-in library.

### 1. Tool metadata

Each tool exports its own permissions. If a tool cannot describe what it does, it should not be callable by an agent.

```python
# tools/analytics.py
from typing import Dict, Any

def get_permissions() -> Dict[str, Any]:
    return {
        "resource_type": "analytics_report",
        "actions": ["read", "export_csv"],
        "sensitive_data": True,
        "cost_tier": "medium",
        "description": "Run analytics queries and export CSV",
    }
```

### 2. Tool registry

A registry maps names to handlers plus metadata. Lazy loading avoids import-time side effects.

```python
# registry.py
from typing import Dict, Callable, Any
from tools.analytics import analytics_tool

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}
        self._load_default_tools()

    def _load_default_tools(self):
        self.register(
            name="analytics",
            handler=analytics_tool,
            permissions=analytics_tool.get_permissions(),
        )

    def register(self, name: str, handler: Callable, permissions: Dict[str, Any]):
        self._tools[name] = {"handler": handler, "permissions": permissions}

    def get_tool(self, name: str) -> Dict[str, Any] | None:
        return self._tools.get(name)

registry = ToolRegistry()
```

### 3. Policy in Rego

The policy file defines roles and the conditions under which an action is allowed. Note the default deny.

```rego
package agent.perms

default allow = false

# Roles are derived from the agent's declared role.
role["read_only"] { input.agent.role == "read_only" }
role["write"]     { input.agent.role == "write" }
role["admin"]     { input.agent.role == "admin" }

# Read access to analytics reports.
allow {
    role[input.agent.role]
    input.resource_type == "analytics_report"
    input.action == "read"
    input.agent.context.allow_reads == true
}

# CSV export requires write role and remaining budget.
allow {
    role["write"]
    input.resource_type == "analytics_report"
    input.action == "export_csv"
    input.agent.context.daily_budget_remaining > 500
}
```

One subtlety: the `role` set is built from `input.agent.role` rather than being a fixed lookup, which means an unknown role string produces an empty set and every rule referencing it fails closed. That is the intended behavior.

### 4. The permission check

The check runs before the handler is invoked. It builds the policy input from tool metadata plus conversation context.

```python
# dependencies.py
from fastapi import HTTPException, Request
from typing import Dict, Any
from registry import registry

# Assume a client wrapper around the OPA HTTP API.
from opa_client import Client as OpaClient
opa = OpaClient(url="http://opa:8181")

async def check_tool_permission(
    request: Request,
    tool_name: str,
    agent_role: str,
    context: Dict[str, Any],
):
    tool = registry.get_tool(tool_name)
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")

    action = context.get("requested_action", "read")

    policy_input = {
        "agent": {"role": agent_role, "context": context},
        "resource_type": tool["permissions"]["resource_type"],
        "action": action,
        "sensitive_data": tool["permissions"].get("sensitive_data", False),
        "cost_tier": tool["permissions"].get("cost_tier", "low"),
    }

    result = await opa.check(policy_input)
    if not result.get("allow", False):
        raise HTTPException(
            status_code=403,
            detail=f"Tool call denied by policy: {tool_name}/{action}",
        )
    return tool["handler"]
```

The original version of this pattern hard-coded `"action": "_call"` for every tool, which makes the action-level rules in the policy unreachable. Passing the actual requested action is what makes per-action rules meaningful.

### 5. Wiring it into a route

The dependency returns the handler only if the policy allows the call.

```python
# routes.py
from fastapi import APIRouter, Depends
from dependencies import check_tool_permission
from typing import Any, Dict

router = APIRouter()

@router.post("/call_tool")
async def call_tool(
    tool_name: str,
    agent_role: str,
    context: Dict[str, Any],
    handler = Depends(check_tool_permission),
):
    return await handler()
```

Note that `context` is now a required argument rather than a mutable default. Mutable defaults in Python are shared across calls, which in a policy context means one conversation's flags can bleed into the next — a security bug, not a style issue.

### 6. Running OPA

OPA can run as a sidecar alongside the agent service, reachable over localhost, or in-process via the WASM target. The sidecar form is simpler to operate and keeps policy updates independent of application deploys.

```yaml
# docker-compose.yml (simplified)
services:
  agent:
    image: python:3.11-slim
    ports:
      - "8000:8000"
    depends_on:
      - opa
  opa:
    image: openpolicyagent/opa:0.60.0
    command: run --server --log-level error /policies
    ports:
      - "8181:8181"
    mem_limit: 256m
```

Set an explicit timeout on the OPA client so a slow evaluation fails closed rather than blocking the agent indefinitely. A timeout that returns "deny" is a safe default; a timeout that returns "allow" is not.

## How to measure whether this is working

Rather than quoting a benchmark, instrument the following and compare before/after on your own traffic:

- **Policy evaluation latency.** Emit a histogram from the OPA client. Watch p50 and p95. In-process evaluation is typically sub-millisecond; a sidecar over localhost adds a small constant. If p95 climbs into double-digit milliseconds, the policy has grown too complex or the bundle needs precompilation.
- **Denied-call rate and reasons.** Count 403s grouped by rule. A sudden spike in one rule usually means a role mapping changed, not that agents got more malicious.
- **Tool-call latency percentiles.** Compare p50 and p95 before and after. A well-scoped candidate set should reduce planning time, because the agent is choosing from fewer options.
- **Timeout incidents per thousand turns.** This is the metric that matters most. If the policy layer is doing its job, agents stop burning turns on calls that were never going to succeed.
- **Context leakage.** Add an assertion that the context object's conversation ID matches the request's conversation ID. Log any mismatch. This catches the failure mode below before it becomes an incident.

Run each of these for at least a week of representative traffic before drawing conclusions. Small samples on agent systems are dominated by which tasks happened to arrive that day.

## Failure modes that appear after launch

### Context leaking between conversations

If the policy context is stored in a module-level variable, a request-scoped cache, or a mutable default argument, flags from one conversation will be visible to another. A high-risk escalation in one session can grant elevated permissions to an unrelated read-only session. Fix: scope context to a conversation identifier, pass it explicitly through the call chain, and expire it after a defined idle period.

### Policy evaluation timeouts under growth

OPA has a default evaluation timeout, and large policies with nested set operations can approach it. The symptom is intermittent denies that disappear on retry. The fix is to precompile the policy into a bundle in CI (`opa build`) and load the bundle at startup, rather than evaluating raw Rego on every request. Splitting a monolithic policy into modules also helps, because evaluation cost scales with the rules that must be considered.

### Tools that cannot describe themselves

Third-party or legacy tools often expose no metadata. Wrapping them in a shim that hard-codes permissions works, but the shim becomes a maintenance liability and drifts from the tool's real behavior. The defensible rule: a tool that cannot declare its resource type, actions and sensitivity does not get registered. Refactor it or exclude it.

### Memory growth from bundle size

Policy bundle size drives the sidecar's memory footprint. A bundle that grows to several megabytes can push a small container past its limit and trigger OOM kills under load. Mitigations: exclude unused policies from the bundle, split by domain, and set the container's memory limit with headroom rather than at the observed steady-state value.

### Role proliferation

Creating a role per use case feels precise and becomes unmaintainable. Teams that start with a dozen product-specific roles often end up with dozens within a quarter, and the policy file becomes something nobody reviews carefully. Consolidate to a small number of capability tiers (read, write, admin) and model edge cases with context flags. Fewer roles, more conditions, easier review.

## When not to add a policy layer

This is real complexity. It is not worth it everywhere.

- **Fewer than about ten tools, all read-only.** The overhead of a policy engine outweighs the benefit. A simple allow-list in code is sufficient.
- **No appetite for policy-as-code.** Rego is a declarative language with a different debugging model than Python. If nobody on the team will own it, the policy will rot and become a false sense of security.
- **Very tight latency budgets.** The sidecar adds a small constant per call. If the system's median tool call is already near its budget, measure before adding anything.
- **Memory-constrained runtimes.** A sidecar needs its own memory allocation. If the platform caps sidecar memory below what the policy engine needs, use the in-process WASM target or skip the layer.

The threshold where the tradeoff flips is roughly when agents start performing write operations or touching sensitive data. At that point, an unenforced capability is a liability regardless of latency.

## FAQ

**How do I write policies for tools that don't expose metadata?**
Wrap the tool in an adapter that implements the metadata interface and delegates to the original implementation. Then treat the adapter as the tool. Never register a tool the policy engine cannot describe, because the engine cannot enforce what it cannot see.

**Can I use cloud IAM instead of a dedicated policy engine?**
Cloud IAM is coarse-grained and has no concept of conversation context or per-session budgets. It works for infrastructure-level roles. For per-call agent decisions that depend on runtime state, a policy engine that accepts arbitrary input is the better fit. Many systems use both: IAM for infrastructure, a policy engine for agent logic.

**What latency budget should the policy layer have?**
Target single-digit milliseconds at p95. If the policy check is a meaningful fraction of the tool call's total time, either precompile the policy or move evaluation in-process. Measure with a histogram, not an average — averages hide the tail that causes retries.

**How do I test policy changes without deploying?**
Run the policy engine in-process during unit tests, load the policy file, and assert allow/deny outcomes for a table of inputs. Include cases for unknown roles, missing context fields, and boundary values on numeric conditions like budget. Boundary tests catch the off-by-one errors that produce either over-permission or spurious denies.

## Do this in the next 30 minutes

List every tool currently registered to your agents and check which ones declare their resource type, actions, and whether they touch sensitive data:

```bash
grep -L "get_permissions" tools/**/*.py
```

Any tool that appears in the output is currently callable with no policy metadata, which means no policy engine can gate it. Either add a `get_permissions()` function to it, wrap it in an adapter, or remove it from the registry. Doing this first gives you the inventory you need before writing a single line of policy.
