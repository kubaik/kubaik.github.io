# Agents running hours? You're losing context fast

Most context-window guides describe the happy path: a clean environment, a short run, and a patient user. Production agents rarely get that. They run for hours or days, accumulate steps, and quietly grow their prompts until latency and cost become the dominant problem.

## The one-paragraph version

Agents that run for hours or days with large context windows waste compute, drift off-topic, and can exhaust memory. The fix is usually not a bigger model or a cleverer prompt, but chunked context with a sliding window that drops what is no longer needed and retains what still affects future decisions. This article covers how to keep agents on track without rediscovering the same facts at every step, using widely available tools: Python 3.11, PostgreSQL 16, and Redis 7.2. It includes a worked example and a comparison of five strategies that can be implemented in under a day.

## Why this concept confuses people

The wall most developers hit is that the agent's prompt grows linearly with each step. A common trap is believing that more context always produces better results. The symptoms of unmanaged growth look like this:

- Prompt tokens climbing from roughly 10k to roughly 100k over a long run.
- Latency rising sharply once the model's prompt cache stops hitting.
- More hallucinations as the agent re-derives the same facts across consecutive steps.

The real problem is not a hard memory limit. It is that context keeps accumulating while relevance decays. After many steps, only a small fraction of the tokens in the prompt still bear on the current goal, yet the model processes all of them. The trap is trying to keep everything, when what is actually needed is a way to forget selectively and retain only what affects future decisions.

## The mental model that makes it click

Think of the agent's context as a rolling notebook. Every page added should either:

1. Directly affect the next action, or
2. Be referenced again within the next N steps, whichever comes first.

If neither condition holds, the page belongs in an archive that is retrieved only when explicitly asked for. This is not about trimming fat. It is about preventing the notebook from becoming a warehouse that cannot be searched in real time.

A useful analogy is a restaurant's ticket rail:

- Tickets in the "next to cook" rail are the active context (the last few minutes of work).
- Tickets in the "pending" rail are recent but not immediately relevant (the last few hours).
- Tickets on the "archives" shelf are pulled only when someone asks about a specific order.

The rail has a fixed length. When it fills up, the oldest ticket drops into the pending rail. When pending fills, it drops to archives. Archives are searchable by order number, but they are fetched only on demand.

Apply the same idea to context. Keep a sliding window of the last M steps (the rail), plus a few key facts tagged as persistent (the shelf). Anything older than a threshold T goes into cold storage that can be queried on demand.

## A concrete worked example

Consider a supply-chain agent that runs continuously for 48 hours, coordinating orders, shipments, and customs paperwork. The agent starts with a 12k-token prompt that includes:

- Standard operating procedures (2k tokens)
- Current inventory snapshot (3k tokens)
- List of active suppliers (2k tokens)
- Last 24 hours of order history (5k tokens)

After 10 hours, the agent has processed 120 new orders and 80 shipment updates. Without context management, the prompt grows by roughly the size of those updates. With a sliding window of the last 120 steps, plus persistent facts, the prompt stays bounded.

The numbers below are illustrative, using a stated assumption of about 100 tokens per step:

- Active window: 120 steps × 100 tokens = 12,000 tokens
- Persistent facts (suppliers, SOPs): about 3,000 tokens
- Total prompt: about 15,000 tokens, regardless of how long the agent runs

```python
# context_manager.py
from typing import List, Dict, Any
import json
from datetime import datetime, timedelta, timezone

class ContextWindow:
    def __init__(self, window_size: int = 120, archive_threshold: int = 360):
        self.window_size = window_size          # active rail size
        self.archive_threshold = archive_threshold  # minutes to move to cold storage
        self.active_window: List[Dict[str, Any]] = []
        self.archive: Dict[str, Dict[str, Any]] = {}
        self.persistent_facts: Dict[str, Dict[str, Any]] = {}

    def add_step(self, step_data: Dict[str, Any]):
        # Enforce sliding window
        if len(self.active_window) >= self.window_size:
            oldest = self.active_window.pop(0)
            self._archive_if_old(oldest)
        self.active_window.append(step_data)

    def _archive_if_old(self, step: Dict[str, Any]):
        step_time = datetime.fromisoformat(step["timestamp"])
        # Use timezone-aware "now" so comparisons are unambiguous.
        now = datetime.now(timezone.utc)
        if step_time.tzinfo is None:
            step_time = step_time.replace(tzinfo=timezone.utc)
        if (now - step_time) > timedelta(minutes=self.archive_threshold):
            step_id = step["id"]
            self.archive[step_id] = step

    def get_current_context(self) -> str:
        # Build prompt from active window + persistent facts
        active_context = json.dumps(self.active_window, ensure_ascii=False)
        persistent_context = json.dumps(self.persistent_facts, ensure_ascii=False)
        return f"""Active context (last {self.window_size} steps):
{active_context}

Persistent facts:
{persistent_context}
"""

    def mark_persistent(self, fact_id: str, fact_data: Dict[str, Any]):
        self.persistent_facts[fact_id] = fact_data

    def retrieve_archive(self, step_id: str) -> Dict[str, Any]:
        return self.archive.get(step_id, {})
```

Usage:

```python
cm = ContextWindow(window_size=120, archive_threshold=360)

# Mark supplier list as persistent
suppliers = load_suppliers()
for sid, data in suppliers.items():
    cm.mark_persistent(f"supplier:{sid}", data)

# Add new step every 5 minutes
while agent_running:
    step = fetch_latest_step()
    cm.add_step(step)
    prompt = cm.get_current_context()
    next_action = llm_query(prompt, max_tokens=4096)
    process_next_action(next_action)
```

In this setup:

- The active window holds only the last 120 steps (about 12k tokens at 100 tokens per step).
- Persistent facts include supplier lists and standard procedures that rarely change.
- Anything older than the archive threshold is moved out of the active window and retrieved only on explicit query.

This keeps the prompt roughly flat even after 48 hours, while preserving the ability to recall older facts on demand.

## How this connects to things you already know

If you have used a database cursor or a Redis LRU cache, the sliding window should feel familiar. The difference is scale: a cursor fetches rows in batches, but an agent's context must be rebuilt into a single prompt every step. The techniques map like this:

| Concept you know | How it maps to context windows |
|------------------|-------------------------------|
| Database cursor fetch size | Sliding window size (M) |
| Redis maxmemory-policy | Archive threshold (T) |
| Materialized views | Persistent facts |
| Indexed columns | Tags for archive retrieval |

The gotcha is that you are not just paging memory. You are rewriting the entire prompt each time. That means every optimization must be measured in prompt token count, not just memory footprint.

To measure this yourself, instrument three things at each step: the serialized prompt length in tokens, the wall-clock time spent building the prompt, and the model's response latency. Log them with timestamps. Then compare a small window against a large one over the same workload. The prompt-building cost is usually linear in prompt size, so a 4× larger prompt tends to cost roughly 4× more to serialize and tokenize, before the model is even called.

## Common misconceptions, corrected

**Misconception: "More context always improves agent reliability."**
Correction: Beyond the active window, additional context does not help and can hurt by diluting the signal. The way to know for a given workload is to run an ablation: hold the task fixed, vary the prompt size, and measure task success and latency. If success plateaus while latency keeps rising, the extra context is not earning its cost.

**Misconception: "You can just trim the prompt with a summarizer."**
Correction: Summarizers can introduce factual errors, especially when compressing long histories. A safer default is to drop irrelevant steps entirely and keep the rest verbatim. If summarization is used, treat the summary as a lossy artifact and keep the original retrievable.

**Misconception: "Vector databases solve this."**
Correction: Vector search is useful for retrieval, but it does not by itself reduce prompt size. A retrieval layer that returns the top-k chunks still has to cap k and cap total tokens, or the prompt grows anyway. Use vector search to augment retrieval, not to replace context management.

**Misconception: "Compression like gzip on the prompt will save compute."**
Correction: Compression saves bandwidth, not model compute. The model still tokenizes the entire decompressed string, and decompression adds latency per step. Keeping the prompt small from the start is more effective.

## The advanced version (once the basics are solid)

Once the sliding window is working, layer in these refinements.

**1. Token-based eviction**
Instead of counting steps, count tokens and drop the oldest step once the active window exceeds a token budget. This prevents a single verbose step from clogging the rail.

```python
class TokenAwareContextWindow(ContextWindow):
    def __init__(self, max_tokens: int = 12000):
        super().__init__()
        self.max_tokens = max_tokens

    def add_step(self, step_data: Dict[str, Any]):
        step_tokens = estimate_tokens(step_data)
        while (self._current_token_count() + step_tokens) > self.max_tokens and self.active_window:
            oldest = self.active_window.pop(0)
            self._archive_if_old(oldest)
        self.active_window.append(step_data)

    def _current_token_count(self) -> int:
        # Approximate token count
        return sum(estimate_tokens(step) for step in self.active_window)
```

**2. Priority tags**
Tag each step with a priority: HIGH (affects the next action), MEDIUM (referenced within 10 steps), LOW (anything else). Evict LOW first, then MEDIUM, and HIGH only if absolutely necessary.

**3. Cold storage with partial rehydration**
When retrieving archived facts, pull only the fields the agent currently needs. A JSONB column in PostgreSQL with a GIN index supports this pattern.

```sql
-- archive_table.sql
CREATE TABLE agent_archive (
    step_id TEXT PRIMARY KEY,
    step_data JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    search_vector TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', step_data::text)) STORED
);

CREATE INDEX idx_agent_archive_search ON agent_archive USING GIN(search_vector);
```

**4. Periodic relevance audit**
At a fixed interval, run a lightweight LLM audit to mark facts as persistent or discardable. Persistent facts go into the persistent_facts map; the rest are dropped unless referenced again within a defined horizon.

```python
async def relevance_audit(archive: Dict[str, Dict[str, Any]]):
    prompt = f"""
    Given these facts, mark each as PERSISTENT or DISCARD.
    Only mark PERSISTENT if the fact affects future decisions.

    Facts:
    {json.dumps(archive, ensure_ascii=False)}
    """
    results = await llm_call(prompt, temperature=0.0)
    for step_id, verdict in parse_llm_results(results).items():
        if verdict == "PERSISTENT":
            archive_fact = archive[step_id]
            mark_persistent(step_id, archive_fact)
            del archive[step_id]
```

With these layers, a long-running agent can keep its prompt bounded while retaining the facts it actually needs. The exact retention rate depends on the workload and should be measured, not assumed.

## Choosing a strategy

| Strategy | When to use | Tools to implement | Token growth over a long run |
|----------|-------------|--------------------|------------------------------|
| Fixed sliding window | Simple agents, moderate volume | Python dict, Redis LRU | Bounded by window size |
| Token-based eviction | Verbose steps, uneven token counts | Python, a tokenizer library | Bounded by token budget |
| Priority tags | Multi-priority workflows | PostgreSQL 16, JSONB | Bounded, biased toward high-priority steps |
| Cold storage with partial rehydration | Long-running agents, large archives | PostgreSQL 16, JSONB + GIN index | Bounded, with on-demand retrieval |
| Full relevance audit | Agents with changing goals | An LLM client, Redis 7.2 | Bounded, with periodic reclassification |

Use the simplest strategy that meets your token budget. If the prompt stays under 15k tokens after 48 hours, a fixed sliding window is enough. If it creeps toward 50k, add token-based eviction. Once it exceeds 50k, layer in priority tags and cold storage.

## How to measure whether it is working

Pick a single metric and a single command. For example, log the token count of the serialized prompt at every step, then compute the 95th percentile over a 24-hour run. Compare it against the same run with the window disabled. If the p95 does not drop, the window is not the bottleneck; look at the persistent facts or the retrieval layer instead.

Also measure the archive hit rate: how often the agent has to fetch a fact outside the active window. A high hit rate suggests the window is too small or the persistence rules are too aggressive.

## Frequently asked questions

**How do you decide the window size for the active context?**
Start with a step count or a token budget, whichever is easier to reason about. Measure the agent's p95 latency after a full run. If it stays within your budget, the window is large enough. If it spikes, reduce the window by 20% and re-measure. The right value depends on the workload and should be derived from measurement, not assumed.

**What happens if the agent needs a fact outside the active window?**
The agent queries the archive by step_id or a semantic tag. The query adds some latency but prevents prompt bloat. The fraction of queries that hit the archive depends on the workload and is worth logging.

**Can you use vector search instead of a sliding window?**
Vector search is useful for retrieval but does not reduce prompt size on its own. Use it to augment retrieval, and still cap the total prompt length explicitly.

**What is the best way to estimate tokens in a step?**
Use a tokenizer that matches the model you are calling. For a mixed JSON step, measure the overhead empirically by tokenizing a representative sample. The overhead depends on the tokenizer and the data shape.

## Next step

Open your agent's context builder file and set a hard limit: either a maximum number of steps or a maximum token budget. Run the agent for one full cycle, then log the serialized prompt size at each step. If the p95 stays under your budget, you are done. If not, halve the window and rerun. This single constraint resolves most context bloat without touching the model or the prompt template.
