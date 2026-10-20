# MCP vs A2A: pick the wrong layer

Agent-driven systems were never the hard part. Knowing when one was about to fail was. The gap between the demo and the incident report is where the real engineering lives.

Two protocols now appear in most agent architecture diagrams: Model Context Protocol (MCP) for connecting a model to tools and data, and Agent-to-Agent (A2A) for connecting agents to each other. They are frequently drawn as alternatives. They are not. They sit at different layers, and a common failure mode in design reviews is not picking the wrong one — it is using one to do the other's job, then spending weeks debugging the symptoms.

The concrete problem this article addresses: teams wire agents together with MCP, or wrap tool calls in A2A, and then hit a class of bugs that looks like a protocol bug but is actually a layer violation — missing task lifecycle, no streaming progress, tool schemas leaking across agent boundaries, and authorization checks that live in the wrong process.

## The two layers, stated precisely

MCP is a client-server protocol. A host application (an IDE, a chat client, an agent runtime) runs an MCP client that connects to one or more MCP servers. Each server exposes three primitives: **tools** (functions the model can call), **resources** (read-only context identified by URI), and **prompts** (reusable templates). The transport is a JSON-based request/response protocol over stdio or HTTP. The server is a capability provider; it does not initiate work, does not have a goal, and does not know about other servers.

A2A is an agent-to-agent protocol. An agent publishes an **Agent Card** at a well-known URL describing its skills, endpoints, and authentication requirements. A client agent sends a **task** (a message plus context), receives updates — including streaming updates — and the task moves through a lifecycle (submitted, working, input-required, completed, failed, canceled). The protocol is designed for opaque agents: the caller does not see the callee's internal tools, prompts, or model.

The distinction that matters operationally:

| Dimension | MCP | A2A |
|---|---|---|
| Relationship | Host ↔ capability provider | Agent ↔ agent |
| Unit of work | A tool call or resource read | A task with a lifecycle |
| Visibility of internals | Schemas are exposed to the caller's model | Agent is opaque; only skills are advertised |
| Who initiates | The host's model decides to call | A calling agent delegates a goal |
| Streaming | Progress notifications per request | Task status updates and artifact streaming |
| Discovery | Server config, often static per host | Agent Card fetched from a URL |
| Auth model | Server-defined; often local process trust | Declared in the Agent Card per scheme |

Read that table as a layering diagram, not a scorecard. A production system usually needs both: MCP inside an agent to reach its tools, A2A between agents that own different tools.

## The failure modes that look like protocol bugs

### Failure 1: MCP as an agent bus

The pattern: three agents each expose an MCP server; a router agent connects to all three and calls them as tools. It works in a demo. Then the first long-running job arrives. MCP tool calls are request/response with optional progress notifications — there is no task object, no `input-required` state, and no standard way for the callee to say "I need clarification before I can continue." The router has to invent that state machine itself, usually by encoding it in tool return strings. Now every agent pair has a bespoke, undocumented state machine. When one agent restarts mid-task, the caller has no protocol-level way to resume; it re-calls the tool and the callee re-does the work.

A2A exists precisely because that state machine needed to be standardized. If the thing on the other side has its own goals, its own tools, and takes minutes rather than milliseconds, you want a task, not a tool call.

### Failure 2: A2A for a single function

The inverse pattern: a team adopts A2A everywhere, including for a lookup that returns a row from a database. Now a 20 ms query is wrapped in an Agent Card fetch, a task submission, a status poll, and an artifact retrieval. The overhead is architectural, not just latency: you have introduced a network hop, an auth handshake, and a lifecycle for something that was a function call. A2A is explicitly scoped to agent-to-agent collaboration; using it as an RPC layer for tools discards MCP's schema exposure, which is the feature that lets the caller's model reason about arguments.

### Failure 3: leaking tool schemas across the boundary

When teams bridge MCP servers through an A2A agent by simply re-exporting the underlying tools, the callee's internal tool surface becomes visible to the caller's model. This breaks the opacity A2A is designed to provide and creates a coupling problem: the callee cannot change its tools without breaking callers. The correct bridge exposes **skills** — named capabilities with input/output descriptions — and keeps the MCP tool list private to the callee.

### Failure 4: authorization in the wrong process

MCP servers commonly run with the host's credentials or a server-specific token. A2A Agent Cards declare authentication schemes, and the calling agent is expected to present credentials for the callee. When an A2A agent internally calls an MCP server using a shared service account, the callee can no longer tell which upstream agent requested the action. Audit logs collapse. The fix is to propagate a caller identity through the task context and have the MCP server enforce per-caller policy — but that only works if you designed the task context to carry it.

## A decision framework

Ask four questions about the thing on the other side of the boundary. The first "yes" determines the layer.

1. **Does it have its own goal and decide its own steps?** If yes, it is an agent — use A2A.
2. **Does it need to ask clarifying questions or pause for input mid-work?** If yes, you need a task lifecycle — use A2A.
3. **Does the caller's model need to see the argument schema to decide how to call it?** If yes, it is a tool — use MCP.
4. **Is the expected latency closer to a function call than a job?** If yes, MCP; wrapping it in A2A adds lifecycle overhead for no benefit.

If both 1 and 3 are true, you have an agent that also exposes tools. That is normal: the agent uses MCP internally and A2A externally. The layers compose; they do not compete.

A quick sanity check on the framework: the questions are ordered so that the most expensive misclassification surfaces first. Treating a goal as a capability (question 1 answered "no" when it should be "yes") produces the bespoke-state-machine failure described above, which is costly to unwind. The reverse error — treating a capability as a goal — is cheaper to fix later, because you can wrap the tool in an agent without changing the tool. When in doubt, start with MCP and promote to A2A once the interaction acquires a lifecycle.

## A worked example: a research agent and a data agent

Assume a system with two agents. The research agent browses and summarizes. The data agent owns a warehouse and answers analytical questions. Both need to be callable by an orchestrator. (Figures below are illustrative, not measured.)

**Inside the research agent:** an MCP client connects to a search MCP server and a fetch MCP server. The model sees tool schemas and picks tools. This is MCP's job.

**Inside the data agent:** an MCP client connects to a SQL MCP server and a schema-resource MCP server. The model sees table schemas as resources and runs queries as tools. Still MCP.

**Between the orchestrator and each agent:** A2A. The orchestrator fetches each Agent Card, sees the skills ("summarize topic," "answer analytical question"), submits a task, and receives streaming status updates. The orchestrator never sees the SQL tool or the search tool. If the data agent needs clarification ("which quarter?"), it returns `input-required` and the orchestrator routes the question to the user.

What breaks if you flatten this? If the orchestrator calls the data agent's SQL tool directly via MCP, the orchestrator's model must understand the warehouse schema, and the data agent's guardrails — row limits, PII filters, query cost checks — live in a process the orchestrator bypasses. That is the layer violation in its most common form.

### Tracing one request through both layers

Following a single request end to end makes the division concrete. An orchestrator receives "What was Q3 revenue by region?" and decides to delegate.

1. The orchestrator reads the data agent's Agent Card from its cached copy (refreshed on a schedule or on version change) and confirms the skill `analytical-question` exists and accepts `text/plain`.
2. It submits an A2A task containing the question and the caller context. The task moves to `submitted`, then `working`.
3. Inside the data agent, the model reads the schema resource from the SQL MCP server, decides it needs the `orders` and `regions` tables, and issues tool calls through its own MCP client. Those calls never appear in the A2A stream.
4. If the question is ambiguous, the data agent transitions the task to `input-required` and emits a status update. The orchestrator surfaces the question and, when answered, submits the follow-up message against the same task ID.
5. On completion, the data agent attaches a table artifact and marks the task `completed`. The orchestrator retrieves the artifact by its identifier rather than parsing it out of a status message.

Every step above has a defined place in one of the two protocols. The value of the split is that each step is observable at a known boundary: the orchestrator can log task transitions without understanding SQL, and the data agent can log tool calls without knowing which upstream user asked.

## Implementation details that matter

### Keep the Agent Card honest

The Agent Card is a contract. If it advertises a skill the agent cannot reliably perform, callers will route work to it and fail. Version the card and treat changes as API changes. A minimal card (illustrative shape, not a full schema) looks like this:

```json
{
  "name": "data-agent",
  "description": "Answers analytical questions over the warehouse",
  "url": "https://data-agent.internal/a2a",
  "version": "1.2.0",
  "capabilities": { "streaming": true },
  "skills": [
    {
      "id": "analytical-question",
      "name": "Analytical question",
      "description": "Answer a question using warehouse data. Returns a table artifact.",
      "inputModes": ["text/plain"],
      "outputModes": ["text/plain", "application/json"]
    }
  ],
  "securitySchemes": { "bearer": { "type": "http", "scheme": "bearer" } },
  "security": [{ "bearer": [] }]
}
```

### Propagate identity through the task

When the data agent calls its SQL MCP server, it should pass the originating caller identity, not a shared service account. A small context object attached to the task is enough:

```python
# Illustrative: attach caller identity to an A2A task and forward it
# to the internal MCP call. Field names are simplified for clarity.
from dataclasses import dataclass

@dataclass
class CallerContext:
    subject: str          # authenticated upstream agent or user
    scopes: list[str]     # e.g. ["warehouse:read"]
    task_id: str

def handle_task(task, caller: CallerContext):
    # Enforce policy at the tool boundary, not at the agent boundary.
    if "warehouse:read" not in caller.scopes:
        return task.fail(reason="insufficient_scope")

    result = mcp_client.call_tool(
        server="sql",
        name="run_query",
        arguments={"sql": task.message.text},
        # Forward identity so the MCP server can audit and enforce.
        metadata={"subject": caller.subject, "task_id": caller.task_id},
    )
    return task.complete(artifact=result)
```

The important line is the `metadata` argument: the MCP server must be able to see who is ultimately responsible. Without it, the audit trail ends at the agent.

### Handle the lifecycle states you did not plan for

Most teams implement `completed` and `failed` and forget `input-required` and `canceled`. Both occur in production. `input-required` is how a well-behaved agent avoids guessing; `canceled` is how an orchestrator stops work when a user gives up. If your orchestrator does not handle them, tasks hang until a timeout fires, and the timeout looks like a protocol failure.

A useful exercise is to enumerate every transition your orchestrator handles and compare it against the full state set. Any state with no handler is a latent hang. The same applies to the agent side: if the callee never emits `working` updates, callers cannot distinguish a slow task from a dead one, and will cancel healthy work.

### Do not stream artifacts through status updates

A2A separates task status from artifacts. Teams sometimes stuff partial results into status messages to avoid implementing artifact retrieval. That works until the artifact is large or needs a content type. Use the artifact channel.

## What outcomes to expect, and their limits

Adopting the two-layer split does not make agents faster. It makes failures legible. Three effects follow from the protocol design:

- **Bounded blast radius.** When an agent misbehaves, the task lifecycle gives you a place to cancel and a record of which state it reached. With MCP-as-bus, a stuck call is just a hung request.
- **Stable interfaces.** Agent Cards and skills change less often than tool schemas, because skills describe outcomes rather than arguments. This reduces cross-team breakage.
- **Clearer authorization.** Identity propagation through tasks makes per-caller policy enforceable at the tool boundary.

The limits are real. A2A adds a discovery step and a lifecycle; for a single synchronous function it is the wrong tool. MCP's schema exposure is a feature only when the caller's model should reason about arguments; when it should not, exposing them is a leak. Neither protocol solves prompt injection, and neither replaces your existing service mesh or API gateway — they sit above them.

### Measuring whether the split helped

Because the benefits are about legibility, the useful metrics are operational rather than throughput. Instrument the following and compare a window before and after the change:

- **Task state distribution.** Count tasks that end in each state. A high share of `failed` or timeout-driven cancellations points at a contract problem, not a network one.
- **Time in `input-required`.** Long dwell times mean the orchestrator's question routing is slow or the agent is asking unnecessary questions.
- **Orphaned tasks.** Tasks still in `working` with no status update beyond a threshold. These are the ones a lifecycle makes visible and an improvised tool-call state machine hides.
- **Tool-call error rate inside the agent.** This should be unaffected by the A2A change; if it moves, the bridge is doing something unexpected.
- **Audit completeness.** Fraction of tool calls whose metadata carries a resolvable caller subject. This should be 100% after identity propagation is in place.

Collect these from your existing tracing and log aggregation rather than building new infrastructure; the task ID is the join key across both layers.

## What to watch out for

- **Agent Card sprawl.** Every agent publishing a card is a new discovery surface. Treat cards like public APIs: versioned, reviewed, and retired deliberately.
- **Skill inflation.** Advertising many narrow skills is easier to route than one broad skill, but it multiplies the contract surface. Prefer a small number of well-described skills.
- **MCP server trust.** An MCP server runs with whatever credentials the host gives it. A malicious or buggy server can exfiltrate context. Scope tokens per server and prefer local processes for sensitive resources.
- **Version skew.** Both protocols are evolving. Pin protocol versions in clients and test upgrades against a staging agent before rolling the card change.
- **The bridge nobody owns.** The code that translates an A2A task into MCP calls is where layer violations hide. Give it an owner and tests.

## The broader lesson

The recurring mistake is treating protocols as brands — picking one and applying it everywhere — instead of as layers with different contracts. MCP's contract is "here is a capability and its schema." A2A's contract is "here is a goal and its lifecycle." A system that respects both contracts gets composability; a system that collapses them gets a bespoke state machine in every pair of agents, which is exactly the cost the protocols were meant to remove. The principle: match the protocol to the **shape of the interaction** — capability or goal, call or task — not to the vendor or the framework you started with.

## How to apply this to your situation

List every boundary in your current design and label it either "capability" or "goal." Capabilities belong on MCP; goals belong on A2A. Any boundary labeled "goal" that is currently an MCP tool call is your highest-risk refactor, because it is the one where lifecycle state is being improvised in strings.

## Frequently Asked Questions

**Can I use MCP and A2A in the same agent?**

Yes, and most production agents do. The agent acts as an A2A server toward its callers and as an MCP client toward its tools. The two protocols run in different directions and do not conflict.

**Is A2A a replacement for MCP?**

No. They solve different problems. A2A connects agents that have their own goals; MCP connects a model to tools and resources. Replacing one with the other forces you to reimplement the missing layer's lifecycle or schema exposure yourself.

**When should an agent expose tools directly instead of skills?**

When the caller's model genuinely needs to choose arguments — for example, a code agent calling a compiler with specific flags. If the caller only cares about the outcome, expose a skill and keep the tools private.

**How do I secure an A2A endpoint?**

Declare an authentication scheme in the Agent Card, require it on every task submission, and propagate the authenticated identity into internal MCP calls so tool-level policy can enforce per-caller limits. Do not let the agent fall back to a shared service account.

---

**Your next 30 minutes:** open your agent's tool-registration file — the place where MCP servers are listed — and mark each entry as either a capability the model should reason about or a goal another agent should own. Any entry in the second category is a candidate for an A2A task boundary, and moving it is the highest-leverage change you can make this week.
