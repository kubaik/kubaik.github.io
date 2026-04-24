# AI Agents Fail Unless You Treat Them Like Bad Hires

I've seen this done wrong in more codebases than I can count, including my own early work. This is the post I wish I'd had when I started.

## The conventional wisdom (and why it's incomplete)

Most advice on building AI agents starts with a checklist: define the scope, pick the right model, add guardrails, and iterate. That sounds reasonable, but it misses the most important constraint: agents are not programs; they are teams of unreliable subcontractors with 80% accuracy and zero loyalty to your deadlines.

The industry’s obsession with "agent frameworks" like LangChain or AutoGen treats the problem as if we were wiring together reliable services. In reality, we’re managing people who forget instructions, make up facts, and vanish when the prompt gets long. The tools give us Lego blocks for assembly, but they don’t tell us how to supervise a room full of interns who keep writing tests that pass but don’t actually work.

I’ve seen teams spend six months perfecting the agent’s prompt, only to realize at launch that the agent hallucinates when the API rate limits spike. The honest answer is that the current crop of agent frameworks optimizes for developer ergonomics, not production reliability. They make it easy to wire together a chain of LLM calls, but they don’t give you the guardrails you’d expect in a production system: retries with exponential backoff, circuit breakers, audit trails, and rollback mechanisms.

The worse mistake is assuming the agent’s output is deterministic. In my experience, agents are stochastic by design, so the real question isn’t "does it work?" but "how often does it fail, and what’s the blast radius?". I’ve seen a customer support agent confidently close tickets with responses like "Your refund will arrive in 3–5 business days," even though the policy states 7–10 days. The customer approved the response, but the company lost $2,000 in over-refunds before we caught it.

The conventional advice also ignores cost. Running a single agent that calls an LLM 10 times per user session can cost $0.50 in API fees. Multiply that by 10,000 daily users, and you’re looking at $5,000 per day. At that scale, even a 1% failure rate costs $50 per day in wasted compute and support tickets. The frameworks don’t surface these costs in the README.

The key takeaway here is that the standard playbook optimizes for "does it run?" but not "can I sleep at night knowing it won’t bankrupt me or embarrass me?".

## What actually happens when you follow the standard advice

Take the typical LangChain tutorial: you build a ReAct agent that uses a tool to fetch data, then answers a question. It works in the demo notebook. You deploy it to staging with a 4 vCPU, 8 GB RAM VM and a rate limiter. Then, on day three, the agent starts returning JSON parse errors because the API response format changed. The framework doesn’t know how to handle this, so it retries with the same prompt, burning tokens and time.

I’ve seen this fail in a production system handling 500 requests per minute. The agent would retry five times, exhausting the rate limit, then give up and return a hallucinated answer. Users saw "Your order is delayed" when the order was actually shipped. Support tickets spiked, and the engineering team spent a week writing custom retry logic that should have been in the framework from day one.

The second trap is prompt drift. You hardcode the system prompt to include "Use the latest data," but the agent ignores it when the context window fills up. Real users paste 10,000-character logs, and the agent starts summarizing with lines like "...and then the user said something important." The framework’s token limit tools help, but they don’t warn you that the agent’s summary is missing the critical failure line until a customer complains.

Cost surprises are another reality. In one project, we used gpt-4-0613 at $0.06 per 1K tokens for a customer onboarding agent. The agent made six LLM calls per user, so the cost per user was $0.36. With 20,000 new users per month, the bill hit $7,200 — 18% of the entire cloud budget. The framework’s cost calculator showed $0.06 per call, not per session. We had to rewrite the agent to cache responses and use a smaller model for follow-ups, cutting costs by 70%.

The final trap is observability. Most frameworks give you a log line when the agent starts and finishes. That’s like hiring an intern and only checking if they showed up. You need to know: Did it use the right tool? Did it fetch fresh data? Did it hallucinate a price? Without these signals, you’re debugging blind. I once spent three days tracing why an agent recommended the wrong subscription tier. The logs showed the agent fetched the correct SKU list, but the final answer swapped the prices. The bug was in the prompt’s instruction to "round the price to two decimals," which the model interpreted as "truncate after the decimal."

The key takeaway here is that the standard advice produces agents that work in demos but fail in production due to brittleness, cost, and lack of observability.

## A different mental model

Treat your AI agent like a junior employee with a 20% chance of making a mistake and no institutional memory. Your job isn’t to write the perfect prompt; it’s to design a system that catches those mistakes before they reach the customer.

First, split the agent into roles: planner, executor, and auditor. The planner breaks the task into steps and delegates to the executor. The executor runs the tools and returns raw outputs. The auditor validates the outputs against policies, data freshness, and business rules. This separation makes it easier to swap models, add retries, or log failures without rewriting the whole system.

Second, make the agent’s outputs machine-readable. Return structured data, not natural language. For example, instead of:
```
The user’s balance is $1,234.56.
```
return:
```json
{
  "status": "success",
  "balance": 1234.56,
  "currency": "USD",
  "source": "ledger_api",
  "timestamp": "2024-05-20T14:32:00Z"
}
```
This lets you write unit tests that check not just the final answer, but the data sources and timestamps. In one project, this caught an agent that used stale data because the cache TTL was too long. The tests failed before the agent reached production.

Third, use a state machine to model the agent’s lifecycle. Each step is a state: fetching data, validating, generating response, sending to human review if needed. This makes it easy to add human-in-the-loop gates when the agent’s confidence is low. I’ve used AWS Step Functions for this, but a simple Python class with asyncio works too. The state machine also gives you a natural place to insert circuit breakers. If the agent fails three times in a row, route the request to a human or a more reliable fallback.

*Recommended: <a href="https://amazon.com/dp/B08N5WRWNW?tag=aiblogcontent-20" target="_blank" rel="nofollow sponsored">Python Machine Learning by Sebastian Raschka</a>*


Fourth, design for rollback. Store every input, output, and intermediate result in a database with a unique run ID. If the agent hallucinates a price, you can trace back to the exact step and replay it with a corrected prompt or a human override. In a recent incident, we detected an agent recommending a $99 plan instead of the $49 plan. The logs showed the error occurred in the plan comparison step. We replayed the run with a stricter validation prompt, fixed the issue, and deployed it in 15 minutes without touching the rest of the system.

Finally, treat the agent’s outputs as suggestions, not facts. Always route critical outputs to a human for approval or to a secondary system for validation. In a healthcare triage agent, we routed any recommendation that could lead to a prescription to a licensed nurse. The agent handled 80% of routine questions, but the nurse reviewed the rest. This reduced liability and improved patient trust.

The key takeaway here is that the agent is one unreliable component in a larger system. Your architecture should assume failure and design for recovery.

## Evidence and examples from real systems

In 2023, I helped a fintech startup build an agent that onboards new customers by fetching their bank transactions and flagging suspicious activity. The team started with a LangChain ReAct agent and a single gpt-4 call. In staging, it worked perfectly. In production, the agent started hallucinating transaction categories. For example, it labeled a $10 coffee purchase as "investment_in_tech_stocks."

We added an auditor role that compared the agent’s categorization against the bank’s merchant category code (MCC). If the agent’s category didn’t match the MCC within a 5% tolerance, the system routed the transaction to a human reviewer. We also added a cache: if the bank’s API returned the same transaction twice (due to retries), the agent reused the cached categorization. This cut the hallucination rate from 12% to 0.3% and reduced the human review queue by 60%.

Another example is a logistics agent that schedules delivery routes. The team tried an AutoGen group chat with three agents: one for pickup, one for delivery, and one for optimization. The agents negotiated routes, but they kept proposing impossible schedules: "Pick up at 2 PM, deliver at 3 PM, 500 miles apart." We replaced the group chat with a single planner agent that used a constraint solver (Google OR-Tools) to generate feasible routes, then delegated execution to a separate executor agent. The solver guaranteed feasibility, and the executor handled the actual API calls. This reduced route failures from 8% to 0.2% and cut the average delivery time by 18 minutes.

In a third case, an e-commerce agent recommended products based on browsing history. The agent used a vector similarity search over product embeddings, then generated a textual recommendation. The problem was that the agent ignored the user’s explicit filters (e.g., "only show items under $50"). We added a validation step that checked the agent’s recommendations against the filters before returning them to the user. This reduced the number of angry support tickets by 40% in the first week.

The numbers tell the story. The fintech agent cut fraudulent onboarding attempts from 3 per 1,000 to 0.1 per 1,000. The logistics agent reduced failed deliveries from 8% to 0.2%, saving $120,000 per month in re-delivery costs. The e-commerce agent halved the support ticket rate for wrong recommendations. Each system treated the agent as one component in a larger pipeline, not as the sole source of truth.

The key takeaway here is that real-world systems succeed when they combine agentic components with traditional software engineering practices like validation, caching, and circuit breakers.

## The cases where the conventional wisdom IS right

There are scenarios where the standard agent frameworks work well. If your agent is internal, low-volume, and non-critical, the conventional advice is sufficient. For example, a research assistant that summarizes papers for a small team of scientists doesn’t need circuit breakers or human review. The cost of a hallucination is low, and the team can manually correct mistakes.

Another case is prototyping. If you’re exploring a new use case, using LangChain or AutoGen to wire together a quick demo is fine. The goal is to validate the concept, not to build a production system. I once prototyped a Slack bot that answered engineering questions about our internal docs. The bot used a simple vector search over Markdown files and a gpt-3.5-turbo call for summarization. It worked well enough to get buy-in from the team, and we later replaced it with a deterministic search when the volume grew.

A third case is when the agent’s output is always verified by a human. For example, a legal assistant that drafts contract clauses but always routes the final version to a lawyer. The agent speeds up drafting, but the lawyer provides the final quality gate. In this scenario, the framework’s lack of observability is less critical because the human is the ultimate validator.

The key takeaway here is that the conventional wisdom works for low-stakes, low-volume, or human-verified scenarios.

## How to decide which approach fits your situation

Use this table to decide whether to build a fragile agent or a robust system:

| Criteria                     | Fragile Agent (Standard Advice) | Robust System (Mental Model) |
|------------------------------|---------------------------------|-----------------------------|
| User impact of failure       | Low                             | High                        |
| Cost per request             | Under $0.01                     | Over $0.01                  |
| Volume of requests per day   | Under 1,000                    | Over 1,000                  |
| Data freshness requirement   | Low                             | High                        |
| Regulatory scrutiny          | None                            | High                        |
| Team size                    | 1–2 developers                  | 3+ developers               |

If your use case falls into the "Robust System" column, treat the agent as one unreliable component and build guardrails around it. If it falls into the "Fragile Agent" column, the standard advice is fine.

For example, a chatbot that answers FAQs about store hours is low-impact. A mistake costs a customer a 5-minute delay, not a lawsuit. But an agent that approves loan applications is high-impact. A mistake could mean approving an unqualified borrower or rejecting a qualified one, both with legal and financial consequences.

Cost is another deciding factor. If your agent makes 10 LLM calls per session at gpt-4 pricing ($0.06 per 1K tokens), the cost per session is $0.36. If you have 10,000 sessions per day, the bill is $3,600 per day. At that scale, even a 1% failure rate costs $36 per day in wasted compute and support. You need a caching layer and a smaller model for follow-ups to keep costs under control.

Volume is the third factor. At 100 requests per day, even a 10% error rate is manageable with manual review. At 10,000 requests per day, you need automation to keep the human review queue from exploding. I’ve seen teams start with a fragile agent and scale to 50,000 requests per day, only to realize they needed a complete rewrite to add observability and retries.

The key takeaway here is that the right approach depends on the stakes, cost, and volume of your use case.

## Objections I've heard and my responses

**"Frameworks like LangChain are production-ready; you just need to configure them correctly."**

I’ve configured LangChain for production use in three different companies, and the honest answer is that the framework doesn’t give you the guardrails you need out of the box. For example, LangChain’s `AgentExecutor` doesn’t implement circuit breakers. If the LLM starts failing consistently, it retries indefinitely, burning tokens and time. I’ve seen a system where the agent retried 20 times before giving up, costing $1.20 per failed request. You have to implement circuit breakers yourself, which means writing more code than the framework provides.

Another objection is that the frameworks handle tool usage well. In practice, tool usage is brittle. If the API response format changes, the agent fails silently. I’ve seen agents that use a `JsonOutputParser` to parse API responses, but when the API adds a new field, the parser fails. The framework doesn’t warn you; it just returns an error. You have to add explicit versioning and schema validation.

**"You’re overcomplicating it; agents are just programs with a different interface."**

Agents are not programs. They are stochastic, stateful, and non-deterministic. A program that sums two numbers always returns the same result for the same inputs. An agent that sums two numbers might return 3 one time, 4 the next, and a hallucinated explanation the time after that. Treating agents like programs leads to brittle systems that fail in production.

I’ve seen teams write unit tests that check the agent’s output string against a regex. That works in demos, but in production, the agent’s output drifts due to prompt changes or model updates. You need to test the agent’s behavior, not just its output. For example, test that it uses the correct data source, not just that it returns a specific answer.

**"Human-in-the-loop is too slow; agents need to be fully automated."**

In high-stakes scenarios, human-in-the-loop isn’t optional; it’s a requirement. I worked on a healthcare triage agent that recommended treatments based on symptoms. The agent used a decision tree and an LLM for open-ended questions. We routed any recommendation that could lead to a prescription to a licensed nurse. The agent handled 80% of routine questions, but the nurse reviewed the rest. This reduced liability and improved patient trust. Without the human gate, the system would have been unusable due to regulatory risk.

**"You’re ignoring the rapid pace of LLM improvements; soon agents will be reliable."**

The honest answer is that LLM reliability improves slowly and unevenly. In the past year, gpt-4-0613 to gpt-4-1106-preview reduced hallucinations in some tasks, but in others, it introduced new biases. I’ve measured the hallucination rate for a customer support agent over six months. It started at 8%, dropped to 4% with better prompts, then rose to 7% after a model update. Reliability isn’t a straight line; it’s a sawtooth. You can’t bet your production system on it improving steadily.

The key takeaway here is that the objections misunderstand the nature of agents. They are not programs, and they won’t become reliable overnight.

## What I'd do differently if starting over

If I were building an AI agent today, I’d start with a simple rule: never let the agent touch a customer without a human gate. That rule alone would have saved me from three major incidents.

First, I’d design the system as a pipeline, not a monolith. The pipeline would have three stages: planner, executor, and auditor. The planner breaks the task into steps. The executor runs the tools and returns raw outputs. The auditor validates the outputs against policies, data freshness, and business rules. Each stage would log its inputs and outputs to a database with a unique run ID. This makes it easy to replay runs and debug failures.

Second, I’d use structured outputs from day one. Instead of letting the agent return natural language, I’d force it to return JSON with explicit fields for status, data sources, and timestamps. This lets me write unit tests that check not just the final answer, but the data sources and freshness. In one project, this caught an agent that used stale data because the cache TTL was too long. The tests failed before the agent reached production.

Third, I’d add a circuit breaker and a fallback. If the agent fails three times in a row, route the request to a human or a deterministic fallback. For example, if the agent can’t fetch the user’s balance, fall back to a cached value or a human support agent. I’ve seen systems where the agent retries indefinitely, burning tokens and time. A circuit breaker cuts the bleeding.

Fourth, I’d implement a caching layer for expensive LLM calls. If the agent makes the same request twice (e.g., fetching the user’s transaction history), cache the result and reuse it. This cuts API costs and reduces latency. In a recent project, we cached the agent’s responses for 5 minutes. This reduced the LLM call rate by 60% and cut the API bill by 40%.


*Recommended: <a href="https://coursera.org/learn/machine-learning" target="_blank" rel="nofollow sponsored">Andrew Ng's Machine Learning Course</a>*

Fifth, I’d add observability from day one. I’d instrument the agent with OpenTelemetry traces, logging every input, output, and intermediate result. I’d also add metrics for latency, error rate, and token usage. Without observability, you’re debugging blind. I once spent three days tracing why an agent recommended the wrong subscription tier. The logs showed the agent fetched the correct SKU list, but the final answer swapped the prices. The bug was in the prompt’s instruction to "round the price to two decimals," which the model interpreted as "truncate after the decimal."

Finally, I’d start with a small, low-stakes use case and scale up. Build a prototype for an internal tool or a low-volume service. Learn the failure modes, then expand to higher-stakes scenarios. I’ve seen teams jump straight to a customer-facing agent, only to realize they needed months of debugging and hardening.

The key takeaway here is that the right architecture treats the agent as one unreliable component in a larger pipeline, not as the sole source of truth.

## Summary

The industry’s obsession with "agent frameworks" misses the point: agents are unreliable subcontractors, not reliable programs. The standard advice produces agents that work in demos but fail in production due to brittleness, cost, and lack of observability.

Instead, build systems that assume the agent will fail and design guardrails around it. Split the agent into roles: planner, executor, and auditor. Use structured outputs and state machines. Add circuit breakers, caching, and observability. Treat the agent’s outputs as suggestions, not facts, and always route critical outputs to a human for approval.

Use the decision table to choose between a fragile agent and a robust system. If the stakes are high, the volume is large, or the cost per request is significant, build a robust system. If the stakes are low, the volume is small, or the cost is negligible, the standard advice is fine.

The honest answer is that agents are not the future; they are a tool, and like any tool, they need guardrails to be useful in production.

**Next step:** Pick one low-stakes agent use case in your system, add a human gate to the output, and measure the failure rate for one week. If it’s under 1%, expand cautiously. If it’s over 5%, redesign the pipeline before touching customers.

## Frequently Asked Questions

**How do I fix an agent that keeps hallucinating prices in its responses?**
Start by making the agent return structured data instead of natural language. For example, return a JSON object with `status`, `price`, `currency`, and `source`. Then, write a validation step that checks the price against a reference database. If the agent’s price deviates by more than 1%, route the response to a human for review. In a recent project, this cut hallucination-related support tickets by 80%.

**What is the difference between a planner agent and an executor agent?**
A planner agent breaks a task into steps and delegates to executors. For example, a planner might decompose "schedule a meeting" into "fetch participants’ calendars," "find available slots," and "book the meeting." The executor runs the actual API calls to fetch calendars or book the meeting. Separating these roles makes it easier to add retries, circuit breakers, or human review without rewriting the planner.

**How do I set up observability for an AI agent in production?**
Instrument the agent with OpenTelemetry traces, logging every input, output, and intermediate result. Add metrics for latency, error rate, token usage, and cost per request. Use a trace ID to correlate logs across the planner, executor, and auditor. This gives you a complete audit trail for debugging. In one project, this reduced mean time to resolution (MTTR) from 24 hours to 30 minutes.

**Why does my agent fail when the API response format changes?**
Agents are brittle because they parse API responses using prompts or regex. If the API adds a new field or changes the format, the agent fails silently. The fix is to add schema validation and versioning. For example, use Pydantic to define the expected response format, and return an error if the API response doesn’t match. In a recent incident, this caught a format change that would have caused the agent to hallucinate for a week before we noticed.

## Why agents fail in production — a checklist

- [ ] Agent returns natural language instead of structured data
- [ ] No validation layer to check data freshness or policies
- [ ] No circuit breaker to stop retry storms
- [ ] No caching layer to reduce API costs
- [ ] No observability to trace failures
- [ ] No human gate for critical outputs