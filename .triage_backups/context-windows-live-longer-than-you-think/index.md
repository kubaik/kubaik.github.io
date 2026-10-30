# Context windows live longer than you think

The default configuration works right up until it doesn't. A context window that never gets pruned turns into an ever-growing liability: cost, latency, and hallucination risk all scale with history length. The practical fix is not a bigger model or a longer prompt — it is treating the context window as a managed store with a retention policy, an eviction rule, and a summarisation step.

## Why this concept confuses people

Most guidance about truncation, summarisation, and "keep it short" assumes sessions that end within a working day. That assumption breaks for background agents: workers that handle support tickets, monitor infrastructure, or run multi-day workflows. In those cases the context window behaves less like a frame you slide and more like a ledger that only closes when the process dies.

Three failure modes show up repeatedly:

**Unbounded growth.** Every turn appends a user message and an assistant reply. Without eviction, token count rises monotonically until the provider either truncates from the front or the request is rejected for exceeding the limit. Front-truncation is especially dangerous because the dropped tokens are the oldest and often the most load-bearing — the original problem statement, the customer's tier, the constraints agreed at the start.

**Stale context.** A model does not forget on its own. If a refunded order ID stays in the window, the model will keep referencing it. If last week's policy is still in the system prompt, the model will apply it. Staleness is not a memory problem; it is a retention-policy problem.

**Cost and latency creep.** Prompt tokens are billed per call and processed per call. A window that grows to tens of thousands of tokens makes every subsequent turn slower and more expensive, even when the extra tokens are irrelevant to the current question.

The word "window" invites the wrong mental model. A window suggests a fixed frame. In practice, the useful model is a table with three columns: **relevant**, **stale**, and **restricted** (PII, secrets, or content that must be redacted). The job is to keep **relevant** large while shrinking the other two as fast as possible.

## A concrete worked example

Consider a support agent handling tickets for a SaaS product. The agent is written in TypeScript, runs on Node 20 LTS, uses a chat-completions API with a 128k-token context window, and stores hot state in Redis with persistent storage in PostgreSQL. The numbers below are illustrative — substitute your own measured token counts.

### Step 1: the naive approach

```typescript
// naive-context.ts
import { OpenAI } from 'openai';

const openai = new OpenAI({ apiKey: process.env.OPENAI_KEY });

interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

class NaiveAgent {
  private messages: ChatMessage[] = [];

  async handleTicket(ticketId: string, userQuery: string) {
    this.messages.push({ role: 'user', content: userQuery });

    const systemPrompt = `You are a support agent for SaaS-X. Current ticket: ${ticketId}.`;

    const response = await openai.chat.completions.create({
      model: 'gpt-4o-mini-2024-07-18',
      messages: [{ role: 'system', content: systemPrompt }, ...this.messages],
      max_tokens: 4000,
    });

    this.messages.push({ role: 'assistant', content: response.choices[0].message.content! });
    return response.choices[0].message.content;
  }
}
```

The bug is structural: `this.messages` is never bounded. Every turn adds tokens, and the only ceiling is the model's context limit. When that ceiling is reached, the provider drops the oldest messages from the front of the array — including the system prompt and the original problem statement. The agent then behaves as if the ticket's constraints were never stated.

### Step 2: sliding window with summarisation

```typescript
// smart-context.ts
import { OpenAI } from 'openai';
import { Redis } from 'ioredis';
import { summarizeText } from './summarizer.js';

const openai = new OpenAI({ apiKey: process.env.OPENAI_KEY });
const redis = new Redis(process.env.REDIS_URL);

interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
  timestamp: number;
}

class SmartAgent {
  private messages: ChatMessage[] = [];
  private readonly MAX_TOKENS = 64000;
  private readonly SUMMARY_INTERVAL_MINUTES = 240;
  private readonly MIN_MESSAGES = 20;

  async handleTicket(ticketId: string, userQuery: string) {
    const now = Date.now();
    this.messages.push({ role: 'user', content: userQuery, timestamp: now });

    while (this.tokenCount() > this.MAX_TOKENS && this.messages.length > this.MIN_MESSAGES) {
      await this.pruneOldest(ticketId);
    }

    const lastSummary = await redis.get(`summary:${ticketId}`);
    if (!lastSummary || now - parseInt(lastSummary, 10) > this.SUMMARY_INTERVAL_MINUTES * 60 * 1000) {
      await this.summariseHistory(ticketId);
    }

    const systemPrompt = `You are a support agent for SaaS-X. Current ticket: ${ticketId}.`;
    const promptMessages = [{ role: 'system', content: systemPrompt }, ...this.messages];

    const response = await openai.chat.completions.create({
      model: 'gpt-4o-mini-2024-07-18',
      messages: promptMessages,
      max_tokens: 4000,
    });

    this.messages.push({
      role: 'assistant',
      content: response.choices[0].message.content!,
      timestamp: now,
    });
    return response.choices[0].message.content;
  }

  private tokenCount(): number {
    // Approximation only. Use a real tokenizer in production.
    return this.messages.reduce((sum, msg) => sum + Math.ceil(msg.content.length / 4), 0);
  }

  private async pruneOldest(ticketId: string): Promise<void> {
    const oldest = this.messages.shift();
    if (!oldest) return;
    await redis.lpush(`old-messages:${ticketId}`, JSON.stringify(oldest));
  }

  private async summariseHistory(ticketId: string): Promise<void> {
    const summary = await summarizeText(this.messages.map(m => m.content).join('\n'));
    this.messages = [{ role: 'system', content: `Previous conversation summary: ${summary}`, timestamp: Date.now() }];
    await redis.set(`summary:${ticketId}`, Date.now().toString(), 'EX', 86400 * 7);
  }
}
```

Three changes matter here, and each fixes a specific bug from the naive version:

1. **The eviction loop has a floor.** `MIN_MESSAGES` prevents the loop from emptying the window when a single message is enormous. In the naive version, a single 100k-token paste would exceed the budget and force the loop to strip everything.
2. **Pruned messages are persisted, not discarded.** `lpush` to a per-ticket Redis list keeps the audit trail recoverable. The naive version lost them permanently.
3. **The summary replaces history rather than appending to it.** This is the key distinction from truncation: truncation drops tokens silently; summarisation replaces them with a bounded, intentional representation.

Note that `tokenCount` uses a character-to-token ratio. That is a placeholder. In production, use a tokenizer that matches the model you are calling, because the ratio varies with language and content type. A Spanish-language support ticket and an English one with the same character count can differ meaningfully in token count.

### How to measure whether any of this is working

Do not trust intuition about token growth. Instrument it:

- **Token count per turn.** Record the exact prompt token count returned in the API response's usage field, not an estimate. Emit it as a histogram.
- **Truncation events.** Log a counter whenever the provider's response indicates the prompt was truncated, or whenever your own eviction loop fires. A rising eviction rate is the signal that your budget is too tight or your summarisation interval too long.
- **Answer quality proxy.** Track a per-turn metric that correlates with correctness for your domain. For support agents, "did the agent restate a fact that contradicts the ticket state" is a workable signal, scored by a small classifier or by human review of a sample.
- **Cost per resolved ticket.** Divide total inference spend by tickets closed. This normalises for traffic changes.

The comparison to make is between your old configuration and the new one, on the same traffic, over at least a full traffic cycle. Percentiles matter more than means: the 90th and 95th percentile token counts tell you how close the tail is to the limit.

## How this connects to things you already know

If you have debugged a service that leaks memory, the pattern is familiar. Every message is an allocation; nothing frees it; the process eventually dies. The difference is that a leaking microservice can be restarted. A conversation cannot be restarted without losing the customer's trust.

If you have used Redis as a cache, the structure maps directly. The sliding window is the hot cache. The per-ticket Redis list is the warm tier. The summary is a compacted representation of the cold tier. The vector store, discussed below, is a retrieval index over cold storage.

If you have managed table bloat in PostgreSQL, the retention policy maps too. Deleting rows outright loses information; summarising first and then archiving loses information gracefully, with a record of what was decided.

## Common misconceptions, corrected

**Myth: bigger models need bigger context windows.**
Model size and optimal window size are independent. A small model with a tight, well-curated window often outperforms a large model with a sprawling one, because irrelevant tokens act as noise. The right window size is determined by the task's information requirements, not the model's capacity.

**Myth: summarisation always loses critical information.**
Summarisation loses whatever the summary prompt does not ask for. A generic "summarise this conversation" prompt will drop structured fields like subscription tier, preferred language, and stated constraints. A summary prompt that explicitly enumerates those fields preserves them. The summary is a lossy filter, and you choose the filter's passband.

**Myth: system messages can be pruned like anything else.**
System messages carry the agent's operating instructions. Pruning them causes the model to improvise instructions, which is how agents start inventing policies. Keep system messages pinned outside the eviction pool, and version them so a policy change is auditable.

**Myth: an external store is too slow for this.**
A single Redis instance handles far more than the write rate of a typical support agent. The bottleneck in practice is the summarisation call, which is an extra model invocation. Measure the summarisation step separately from the main turn; if it dominates latency, move it off the critical path.

## The advanced version: retrieval over cold storage

Once the sliding window and summarisation are stable, the next step is to retrieve old context by relevance instead of keeping it in the window. Store each pruned message in a vector store, embed the current query, and inject the top-k most similar past messages into the prompt.

```python
# vector-store-context.py
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.load_local("support_faiss", embeddings, allow_dangerous_deserialization=True)
llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18")

class VectorAgent:
    def __init__(self, ticket_id: str):
        self.ticket_id = ticket_id
        self.window = []
        self.vector_store = vector_store

    async def handle_query(self, query: str) -> str:
        docs = self.vector_store.similarity_search(query, k=5)
        context = "\n".join([doc.page_content for doc in docs])

        prompt = f"""
        Ticket: {self.ticket_id}

        Relevant past messages:
        {context}

        Current conversation:
        {"\n".join(self.window)}

        User: {query}
        Assistant:
        """

        response = await llm.ainvoke(prompt)
        self.window.append(f"User: {query}\nAssistant: {response.content}")
        return response.content
```

The trade-off is a retrieval round-trip on every turn in exchange for a much smaller prompt. Whether that is a net win depends on your latency budget and how often the relevant context is actually outside the recent window. Measure both: retrieval latency added, and answer quality with and without retrieval, on a held-out set of tickets.

Two failure modes are worth naming. First, retrieval can surface a semantically similar but factually outdated message — for example, a refunded order that matches the query embedding. Mitigate by filtering retrieved documents on ticket state and timestamp before injection. Second, retrieval adds a new dependency to the request path; if the vector store is unavailable, decide in advance whether to degrade to window-only or to fail the turn.

## Retention policy by ticket state

A static policy is a starting point, not an endpoint. Ticket state is a useful signal for how aggressively to prune.

| Ticket state | Window size | Summarise interval | Vector retrieval | Rationale |
|--------------|-------------|--------------------|------------------|-----------|
| Active chat (recent activity) | Large | Long | Top 3 | Continuity matters most while the customer is present |
| Pending internal review | Small | Short | Top 5 | Agent is not responding live; compress aggressively |
| Closed | Minimal | Long | Top 1 | Only the resolution and any refund reference are needed |
| Escalated | Medium | Medium | Top 10 | Preserve the escalation trail for the next responder |

These are illustrative starting values. Tune them against the measurements described earlier: eviction rate, truncation events, and the quality proxy.

## Strategy comparison

| Strategy | When it fits | Prompt size | Added latency | Main risk |
|----------|--------------|-------------|---------------|-----------|
| Sliding window only | Short sessions | Low | None | Loses long-range context silently |
| Window + summarisation | Multi-hour sessions | Medium | Summariser call | Summary omits fields the prompt did not request |
| Retrieval over cold storage | Long histories with recurring topics | Low | Retrieval round-trip | Surfaces stale-but-similar context |
| Two-tier retention | Mixed workloads | Variable | Variable | Policy complexity; needs monitoring |
| Archive and delete | Compliance-driven retention limits | None | None | Irreversible data loss |

## Compliance and multi-agent considerations

Retention limits are a design input, not an afterthought. If your jurisdiction or policy requires deletion after a fixed period, store archived messages in object storage with a lifecycle rule rather than in a cache with per-key TTLs. Per-key TTLs are easy to get wrong, and a missed key is a compliance incident. Object-storage lifecycle rules delete by prefix and age, which is auditable.

For multiple agents working the same ticket, use a distributed lock keyed on the ticket ID. A Redis `SET ticket_id lock_id NX PX <ttl_ms>` acquires the lock atomically. Set the TTL generously relative to your worst-case turn latency, and renew it if a turn runs long. A lock that expires mid-turn is how two agents end up responding to the same message.

## FAQ

**How do I know when the context window is too large?**
Look at the distribution, not the mean. Pull the prompt token count from the API response's usage field for every turn, and compute the 90th and 95th percentiles per ticket. If the tail is approaching the model's limit, you are one unusually long turn away from truncation.

**Can a smaller model handle the summarisation step?**
Test it. Summarisation quality is measurable: check whether the summary preserves the fields your prompts depend on, such as tier, language, and open constraints. If a smaller model preserves them, use it. If not, the savings are not real.

**How should retention interact with deletion requests?**
Design for deletion from the start. Keep a per-ticket index of every storage location that holds that ticket's data — cache, archive, vector store — so a deletion request is a lookup plus a set of deletes rather than a search. Vector stores are the easy one to forget.

**What about multiple agents on one ticket?**
Lock on the ticket ID with a TTL longer than your worst-case turn. Log lock acquisition and release so you can see contention and expiry events in production.

## One thing to do in the next 30 minutes

Open your agent's code and find where the message array is appended to. If there is no eviction step between the append and the API call, that is the bug. Add a single counter that records the prompt token count from the API response's usage field, tagged by ticket ID, and emit it to whatever metrics system you already run. That one metric will tell you within a day whether your tail is approaching the limit — and whether the work in this article applies to you at all.
