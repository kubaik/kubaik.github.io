# Multi-agent systems break first

## What the docs leave out

Framework documentation covers how to wire agents together. It rarely covers what happens months into production, when traffic is real and the edge cases start accumulating. The gap is not in the API surface; it is in the operational assumptions that tutorials quietly make.

The first assumption is that **agents fail like functions**. A function returns a value or raises. An agent can return a partial result, a plausible-but-wrong answer, an empty object, or nothing at all while remaining "healthy" from the scheduler's point of view. A typical failure mode is a downstream timeout cascade with no error in the agent's own logs, because the agent never raised — it just produced something the next agent could not use.

The second assumption is that **agents are deterministic**. They are not, by design. A prompt-plus-tool-call loop samples from a model, and the same input can produce different tool sequences on different runs. A test suite that passes repeatedly in staging tells you the happy path is reachable; it does not tell you the distribution of outputs under real load, with stale caches and partial tool failures.

The third assumption is that **agents are stateless**. Any agent that keeps conversation history, tool results, or retrieved documents is a stateful service. That state needs a store, a retention policy, an eviction policy, and a recovery path. Skipping those turns a memory store into an outage.

The fourth assumption is that **agent output is cheap**. A decision loop that looks trivial in a notebook can hold a container open, poll a queue, and call paid APIs. The cost is dominated by infrastructure and API calls, not by the agent's own compute.

## The architecture under the hood

A production multi-agent system is a stateful, event-driven graph. Nodes are processes that read messages, mutate state, and emit messages. Edges are queues or topics. The graph changes at runtime as agents scale, restart, or get replaced.

A concrete shape: a customer support escalation system with three roles.

- **RouterAgent** — assigns a ticket based on skill match, current load, and SLA urgency.
- **SupportAgent** — handles the conversation and calls tools such as a knowledge-base lookup or a payment API.
- **EscalationAgent** — watches for tickets that are stuck or breaching SLA and either reassigns them or hands off to a human.

Each agent runs in its own container with a minimal runtime. Agents communicate over a message broker with durable queues. State (conversation history, tool outputs, metadata) lives in a key-value store with TTLs. An orchestrator service manages lifecycle, health checks, and scaling.

The important property is that **the graph is not a pipeline**. When a SupportAgent calls a tool, the result can feed back into routing. When the EscalationAgent finds a stale ticket, it can override a routing decision. There is no single transaction spanning these steps, so partial failure is the normal case, not the exception.

### Worked example: a malformed message

Suppose the payment API times out. The SupportAgent emits a result with `status: "timeout"` and no `assignee` field. The RouterAgent's contract expects `ticket_id` and `assignee`.

Three possible policies, with different consequences:

1. **Reject and dead-letter.** The RouterAgent validates the message, fails validation, and routes it to a dead-letter queue. A human reviews it. Latency for that ticket increases, but no incorrect routing happens.
2. **Retry.** The RouterAgent re-queues the message. If the payment API is still down, this repeats until the retry budget is exhausted, then dead-letters. Retries help with transient faults and hurt with persistent ones.
3. **Best-effort default.** The RouterAgent assigns to a default handler. This keeps latency flat but can route a payment-failure ticket to an agent that will try the same failing call again.

The correct choice depends on the cost of a wrong assignment versus the cost of delay. The point is that the policy must be explicit and logged, because the framework will not choose it for you.

## State, memory, and the eviction trap

Agent memory is the most common source of production incidents in these systems.

A SupportAgent that keeps a sliding window of the last N messages will grow its state store steadily. If the store is configured with `noeviction`, writes fail once memory is full, and the agent starts dropping messages or crashing. If it is configured with an LRU policy, older keys are evicted under pressure, and an agent may silently lose the context it needs to answer correctly.

The two failure modes look different:

- **`noeviction`** produces loud failures: write errors, agent crashes, queue backups.
- **`allkeys-lru`** produces quiet failures: the agent answers with missing context and the output looks plausible.

Both are avoided the same way: set an explicit TTL per state key, cap the size of any per-agent structure, and monitor memory with `redis-cli info memory` (or the equivalent for your store). A sliding-window TTL — refreshed on each access, with a hard ceiling on retained entries — bounds memory without losing recent context.

A second trap is **latent state**. An agent waiting for user input can sit in that state indefinitely. Without a timeout, tickets never close. With a timeout, replies that arrive after it can land in a closed ticket and get a canned response that ignores the new content. The fix is a small explicit state machine with defined transitions and a rule for out-of-order input — for example, a reply after closure opens a new ticket linked to the old one rather than being appended to it.

## A minimal implementation

The following is a small but honest skeleton: agents communicate over durable queues, keep bounded state in a key-value store, and acknowledge or dead-letter every message. It is deliberately incomplete in the places noted, because those places are where production systems diverge.

Infrastructure:

```bash
docker run --name redis-agent -p 6379:6379 -d redis/redis-stack-server:7.2.0-v1

docker run --name rabbitmq-agent \
  -p 5672:5672 -p 15672:15672 \
  -e RABBITMQ_DEFAULT_USER=user \
  -e RABBITMQ_DEFAULT_PASS=pass \
  rabbitmq:3.13-management
```

Agent base class:

```python
# agent.py
import json
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import pika
import redis


@dataclass
class AgentMessage:
    message_id: str
    sender: str
    recipient: str
    content: Dict[str, Any]
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None


class BaseAgent:
    def __init__(self, agent_id: str, input_queue: str, output_queue: str):
        self.agent_id = agent_id
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.redis = redis.Redis(host="localhost", port=6379, db=0)
        self.state_key = f"agent:{agent_id}:state"

        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host="localhost",
                port=5672,
                credentials=pika.PlainCredentials("user", "pass"),
            )
        )
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue=input_queue, durable=True)
        self.channel.queue_declare(queue=output_queue, durable=True)
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(
            queue=input_queue, on_message_callback=self._process_message
        )

    def _process_message(self, ch, method, properties, body):
        try:
            msg = AgentMessage(**json.loads(body))
            self._update_state(msg)
            result = self.process(msg)
            if result:
                response = AgentMessage(
                    message_id=str(uuid.uuid4()),
                    sender=self.agent_id,
                    recipient=msg.sender,
                    content=result,
                    timestamp=time.time(),
                    metadata={"processed_by": self.agent_id},
                )
                self.channel.basic_publish(
                    exchange="",
                    routing_key=self.output_queue,
                    body=json.dumps(asdict(response)),
                    properties=pika.BasicProperties(delivery_mode=2),
                )
            ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as exc:
            print(f"Error processing message: {exc}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    def _update_state(self, msg: AgentMessage):
        key = f"agent:{self.agent_id}:messages"
        pipe = self.redis.pipeline()
        pipe.lpush(key, json.dumps(asdict(msg)))
        pipe.ltrim(key, 0, 9)
        pipe.expire(key, 60 * 60 * 24)
        pipe.execute()

    def process(self, msg: AgentMessage) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def start(self):
        print(f"Agent {self.agent_id} started. Listening on {self.input_queue}")
        self.channel.start_consuming()
```

Router agent:

```python
# router_agent.py
import random
import time

from agent import AgentMessage, BaseAgent

REGION_HANDLERS = {
    "us": "handler_us",
    "eu": "handler_eu",
    "asia": "handler_asia",
}

PRIORITY_WEIGHTS = {"high": 3, "medium": 2, "low": 1}


class OrderRouter(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_id="router",
            input_queue="orders",
            output_queue="router_output",
        )
        self.handler_load = {h: 0 for h in REGION_HANDLERS.values()}

    def process(self, msg: AgentMessage):
        order = msg.content.get("order", {})
        region = order.get("region")
        priority = order.get("priority", "medium")
        order_id = order.get("order_id")

        weights = {
            h: PRIORITY_WEIGHTS[priority] * (1 + self.handler_load[h] * 0.1)
            for h in self.handler_load
        }
        choice = random.choices(
            list(weights.keys()), weights=list(weights.values()), k=1
        )[0]
        self.handler_load[choice] += 1

        return {
            "action": "assign",
            "handler": choice,
            "order_id": order_id,
            "priority": priority,
            "region": region,
            "assigned_at": time.time(),
        }


if __name__ == "__main__":
    OrderRouter().start()
```

Handler agent:

```python
# handler_agent.py
import random
import time

from agent import AgentMessage, BaseAgent


class OrderHandler(BaseAgent):
    def __init__(self, agent_id: str):
        super().__init__(
            agent_id=agent_id,
            input_queue=f"{agent_id}_input",
            output_queue="escalation_input",
        )
        self.inventory = {"item1": 100, "item2": 50}

    def process(self, msg: AgentMessage):
        order = msg.content.get("order", {})
        order_id = order.get("order_id")
        item = order.get("item")
        quantity = order.get("quantity")

        if item not in self.inventory:
            return {
                "action": "error",
                "error": "item_not_found",
                "order_id": order_id,
                "handler": self.agent_id,
            }

        if self.inventory[item] < quantity:
            return {
                "action": "error",
                "error": "out_of_stock",
                "order_id": order_id,
                "handler": self.agent_id,
            }

        time.sleep(random.uniform(0.1, 0.5))
        self.inventory[item] -= quantity

        return {
            "action": "processed",
            "order_id": order_id,
            "handler": self.agent_id,
            "status": "completed",
            "notified": True,
        }


if __name__ == "__main__":
    for region in ["us", "eu", "asia"]:
        OrderHandler(agent_id=f"handler_{region}").start()
```

Escalation agent — note that this version blocks the consumer loop while waiting, which is a real defect in any system that must handle more than one ticket at a time:

```python
# escalation_agent.py
import time

from agent import AgentMessage, BaseAgent


class EscalationAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_id="escalation",
            input_queue="escalation_input",
            output_queue="human_review",
        )
        self.pending_orders = set()

    def process(self, msg: AgentMessage):
        content = msg.content
        if content.get("action") == "processed":
            self.pending_orders.discard(content["order_id"])
            return None

        if content.get("action") == "assign":
            order_id = content["order_id"]
            self.pending_orders.add(order_id)
            time.sleep(2)
            if order_id in self.pending_orders:
                return {
                    "action": "escalate",
                    "order_id": order_id,
                    "reason": "stuck",
                    "assigned_handler": content["handler"],
                }
        return None


if __name__ == "__main__":
    EscalationAgent().start()
```

Publishing a test order:

```python
# publish_test_order.py
import json

import pika

connection = pika.BlockingConnection(
    pika.ConnectionParameters(
        host="localhost",
        port=5672,
        credentials=pika.PlainCredentials("user", "pass"),
    )
)
channel = connection.channel()
channel.queue_declare(queue="orders", durable=True)

order = {
    "order_id": "order_123",
    "region": "us",
    "priority": "high",
    "item": "item1",
    "quantity": 10,
}

channel.basic_publish(
    exchange="",
    routing_key="orders",
    body=json.dumps({"order": order}),
    properties=pika.BasicProperties(delivery_mode=2),
)

print("Order published")
connection.close()
```

The skeleton shows the mechanics: durable queues, explicit acknowledgement, bounded per-agent state. It is not production-ready. The escalation agent blocks its consumer while sleeping; a real implementation uses a timer wheel or a scheduled check against a sorted set of deadlines. The inventory dict is process-local; a real implementation needs a shared store with atomic decrement. The router ignores handler health and queue depth; a real implementation weighs those.

## Measuring performance instead of guessing

Published benchmark tables for multi-agent systems are close to useless, because results depend on model choice, tool latency, prompt length, and queue configuration. What is useful is a measurement plan.

**Define the units first.** Per-message latency (time from publish to final acknowledgement) and per-task latency (time from first message to terminal state) are different numbers. Report both.

**Instrument these points:**

- Publish timestamp and handler-received timestamp on every message, so queue wait time is separable from processing time.
- Per-agent processing duration, emitted as a histogram rather than an average.
- Tool-call count and duration per task, tagged by tool name.
- State-store memory and key count, sampled on a fixed interval.
- Queue depth per queue, sampled on a fixed interval.
- Retry count and dead-letter count per queue.

**Compare configurations, not absolutes.** Run the same synthetic workload against two builds and compare P50 and P99 per-task latency, total tool calls, peak queue depth, and peak state-store memory. A change is worth keeping if it improves P99 without increasing dead-letter rate.

**Watch for the latency spikes that come from timeouts.** If a tool call has a fixed timeout and a retry, a tool that consistently exceeds the timeout produces a retry storm. The symptom is a bimodal latency histogram: most tasks fast, a tail clustered at multiples of the timeout. Instrumenting tool duration separately from agent duration is what makes this visible.

**Watch for uneven utilization.** In many systems the routing component is nearly idle while one processing role saturates. Per-agent CPU and per-agent queue depth reveal this; aggregate CPU does not.

## Failure modes and fixes

### Silent data loss

An agent returns `{}` or `null` when a tool times out. The next agent parses it, fails, retries, and eventually dead-letters. Nothing in the producing agent's logs indicates a problem.

**Fix:** Validate every agent output against a schema before publishing. Reject malformed output at the producer with an explicit error code, and count rejections as a first-class metric.

### State growth

Per-agent history stored without TTL grows until the store evicts or fails.

**Fix:** Set TTLs on all state keys, cap the number of retained entries per key, and alert on state-store memory growth rate rather than absolute level.

### Reprocessing after restart

A durable queue plus a restarted consumer can deliver the same message twice. If the consumer is not idempotent, it repeats side effects — re-escalating a resolved ticket, charging twice, sending a duplicate notification.

**Fix:** Deduplicate on a message or task identifier before performing side effects. A short-TTL key set with an atomic set-if-not-exists is sufficient for most cases.

### Prompt drift

Prompts are code but are often edited directly in production. A prompt change alters routing distribution, which can overload one downstream agent.

**Fix:** Version prompts alongside code, deploy them through the same pipeline, and roll out changes to a fraction of traffic first. Track the routing distribution as a metric so a shift is visible immediately.

### Tool-call explosion

Permissive tool definitions let an agent call a tool that triggers another agent that calls another tool. A single task can fan out into dozens of calls, inflating latency and cost.

**Fix:** Enforce a per-task tool-call budget and a per-task token budget. Log every call with its cost and duration, and fail the task when the budget is exhausted rather than continuing.

### Backpressure blindness

Queues are assumed to be infinite. When input rate exceeds processing rate, depth grows until the broker hits memory limits or consumers time out.

**Fix:** Monitor queue depth and consumer count. Apply backpressure at the entry point — reject new work with a 429 when depth exceeds a threshold — rather than letting the queue absorb the overflow.

### Observability gaps

Many frameworks do not expose agent state transitions by default. Without instrumentation, a stuck agent looks identical to a slow one.

**Fix:** Emit a structured event on every state transition, tool call, and message acknowledgement. Include the task identifier so events from different agents can be joined into a single trace. This is the single highest-value investment in these systems.

## Decision checklist

Before running a multi-agent system in production, confirm each of these:

- Every agent output is schema-validated before it is published.
- Every state key has a TTL and a size cap.
- Every consumer is idempotent with respect to message identifiers.
- Every tool call has a timeout, a retry budget, and a cost log.
- Queue depth and consumer count are monitored with alerts.
- Entry-point backpressure rejects work when queues are saturated.
- Prompts are versioned and deployed through a controlled rollout.
- State transitions, tool calls, and acknowledgements emit structured events joined by task identifier.
- A dead-letter queue exists and is reviewed, not just monitored.
- Per-task latency and per-task cost are reported as distributions, not averages.

## Take action in the next 30 minutes

Pick one agent in your system and add a structured log line at three points: message received, tool call started and finished, and message acknowledged — each including the task identifier and a monotonic timestamp. Restart the agent, send a handful of messages through it, and join the three events for a single task. If you cannot reconstruct the task's timeline from those logs alone, that gap is the first thing to fix, and it is the gap that makes every other failure mode harder to diagnose.
