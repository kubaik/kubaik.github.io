# Context windows: agents that run for days

african engineering problems have a habit of surfacing mid-migration, right when there's no time to solve them properly. Here's what changed once we stopped guessing and started measuring. The gap between the demo and the incident report is where this actually lives.

## The one-paragraph version (read this first)

A long-running agent has a context window that fills up. The naive fix — just summarize old turns and keep going — works for a few hours and then fails in a way that's hard to debug: the agent forgets a constraint it was told at hour 2, or it re-reads a file it already modified, or it starts hallucinating a tool result that never happened. The real fix is to stop treating the context window as a log and start treating it as a working set. You keep a small, structured state object outside the prompt, you retrieve only the slices you need per step, and you let the raw transcript age out. That's the whole idea. The part that trips people up is that "summarize and continue" looks like it preserves state when it actually destroys the exact details agents depend on — and that's what this post actually covers.

## Why this concept confuses people

When you build a chatbot, the context window is basically a conversation. Turn 1, turn 2, turn 3, all appended. It grows, you hit the limit, you truncate the oldest turns, done. That mental model is correct for chat and wrong for agents, and the confusion comes from the fact that both use the same API shape.

An agent is not a conversation. It's a loop: observe, decide, act, observe. Each iteration appends a tool call and a tool result. A single `read_file` on a 2,000-line source file can burn more tokens than the entire preceding conversation. A `grep` that returns 400 matching lines can do the same. Over a multi-hour run, the transcript is dominated not by reasoning but by tool output — and tool output is the least useful thing to keep around once you've extracted what you needed from it.

Here's the trap. Teams hit the limit, reach for summarization, and the summary drops the things that matter. A common failure mode looks like this: an agent is told at step 4 that the staging database is read-only and it must not run migrations. By step 300, that constraint has been summarized into "working on staging database tasks." The agent then runs a migration. Nothing in the summary was wrong, exactly. It just wasn't load-bearing.

There's a second source of confusion: people conflate the model's context limit with the agent's memory. They're different. The context limit is a hard wall — GPT-4o at 128k tokens, Claude 3.5 Sonnet at 200k, Gemini 1.5 Pro at 1M. Memory is whatever you decide to persist outside that wall. If your only memory is the context window, you have no memory, you have a buffer.

## The mental model that makes it click

Think of it like a chef working a 12-hour service. The context window is the counter in front of them. It holds what they're actively working on: the ticket they're plating, the ingredients for that dish, the pan on the burner. It does not hold the entire walk-in fridge, and it should not — you can't cook on a counter that's also a warehouse.

The walk-in fridge is your external state. It's a database, a JSON file, a Redis key, a vector store, whatever. The chef walks over, grabs what they need for this ticket, comes back. The counter stays small.

The mistake most teams make is treating the counter as the fridge. They keep piling things on it because "the model might need it later." It won't need all of it. It needs the specific things relevant to the current step, and the skill is knowing which those are.

So the model is: **external state + retrieval + bounded working set.** Three pieces.

External state is a structured record of facts the agent has established: file paths it has touched, decisions it has made, constraints it was given, results it has committed. It's small. It's queryable. It survives process restarts, which matters when your agent runs for three days and the worker gets OOM-killed at hour 40.

Retrieval is how you pull slices of that state into the prompt. Sometimes it's a simple key lookup ("give me the constraints list"). Sometimes it's semantic search over past tool results. Sometimes it's just "the last 5 actions."

Bounded working set is the discipline of capping what actually goes into the prompt per step. You decide a budget — say 8k tokens for context, 4k for the current task — and you enforce it. When something doesn't fit, it gets summarized or dropped, and you decide which, explicitly, per category.

## A concrete worked example

Say you're building an agent that monitors a codebase and opens PRs for dependency updates. It runs continuously, checks for new releases every 30 minutes, and when it finds one it reads the changelog, checks the repo for usage, and either opens a PR or logs "skip."

Run that for a week and you'll process maybe 300 updates. If you append everything to context, you're looking at 300 changelogs plus 300 grep results plus 300 decisions. At an average of 1,500 tokens per cycle, that's 450k tokens — well past most models' limits, and even on a 1M-token model you're paying for tokens that are 99% irrelevant to the current decision.

The fix is to split state into three stores:

1. **Decisions log** — append-only, structured. `{package, version, action, reason, timestamp}`. One line per cycle. This is what you retrieve when the agent needs to know "have I already handled this?"
2. **Constraints** — a small, stable list. "Don't touch packages in the `@internal/*` scope." "Major version bumps require a human review flag." This never gets summarized; it's always injected verbatim, because it's the thing summarization destroys.
3. **Working context** — the current cycle only. Changelog for this package, grep results for this package, the relevant package.json lines. Discarded when the cycle ends.

Here's a minimal version of that in Python, using a plain SQLite file for state (no vector DB needed for this shape):

```python
import sqlite3
import json
from datetime import datetime, timezone

DB = "agent_state.db"

CONSTRAINTS = [
    "Never modify packages under @internal/* scope.",
    "Major version bumps must set requires_human_review=True.",
    "If changelog mentions a breaking change, skip and log.",
]

def init_db():
    conn = sqlite3.connect(DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS decisions (
            package TEXT, version TEXT, action TEXT,
            reason TEXT, ts TEXT,
            PRIMARY KEY (package, version)
        )
    """)
    conn.commit()
    return conn

def already_handled(conn, package, version) -> bool:
    row = conn.execute(
        "SELECT 1 FROM decisions WHERE package=? AND version=?",
        (package, version),
    ).fetchone()
    return row is not None

def build_prompt(conn, package, version, changelog, grep_hits):
    # Bounded working set: constraints always, recent decisions only.
    recent = conn.execute(
        "SELECT package, version, action FROM decisions "
        "ORDER BY ts DESC LIMIT 20"
    ).fetchall()

    return {
        "constraints": CONSTRAINTS,          # never summarized
        "recent_decisions": recent,          # last 20, hard cap
        "current_task": {                    # discarded after this cycle
            "package": package,
            "version": version,
            "changelog": changelog[:4000],   # truncate, don't summarize
            "grep_hits": grep_hits[:50],
        },
    }

def record(conn, package, version, action, reason):
    conn.execute(
        "INSERT OR REPLACE INTO decisions VALUES (?,?,?,?,?)",
        (package, version, action, reason,
         datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
```

Three things to notice. First, constraints are injected verbatim every cycle — no summarization, ever. Second, the working set is truncated by character count, not summarized. Truncation is lossy but predictable; summarization is lossy and unpredictable, and unpredictability is what kills long runs. Third, the decisions table is the actual memory, and it's tiny — a week of cycles is maybe 300 rows, a few hundred KB.

A common failure mode here is forgetting to cap `recent_decisions`. Six months in, that LIMIT 20 query is fine, but if someone "improves" it to "all decisions" to give the agent more history, you're back to the same problem. Cap everything that goes into the prompt.

## How this connects to things you already know

If you've done any backend work, this is a cache eviction problem. The context window is your L1 cache. External state is your database. Retrieval is your cache-miss handler. You already know that you don't put the whole database in L1, and you already know that LRU isn't always right.

The specific eviction policy matters, and it's where the analogy gets useful. For a long-running agent, you want something closer to **category-based eviction** than LRU. Constraints never evict. Decisions evict by recency. Tool results evict by relevance-to-current-task. Raw transcripts evict immediately after the step that produced them.

It's also the same shape as a stream processor's state management. If you've used Kafka Streams or Flink, you know that a long-running stream operator can't keep everything in memory — it keeps a bounded state store and checkpoints it. Agent state is the same: bounded, checkpointed, recoverable.

And it's the same as session management in a web app. You don't put the whole user session in a cookie. You put a session ID, and the server looks up the rest. Your agent's prompt should include a state pointer, not the state itself.

## Common misconceptions, corrected

**"Bigger context windows solve this."** They delay it. A 1M-token window at 300 cycles still fills in a few days. More importantly, models degrade on long contexts — the "lost in the middle" effect is well-documented, where information in the middle of a long prompt is retrieved less reliably than information at the start or end. Even if you fit, you lose. Anthropic's own docs on long-context prompting recommend putting key instructions at the beginning and end for exactly this reason.

**"Summarization is memory."** Summarization is compression, and compression is lossy in ways you can't predict. A summary of "the agent worked on authentication" loses "the agent must not touch the OAuth token refresh path." The constraint is what matters; the summary is what survives.

**"Vector search over the transcript is enough."** It helps, but it's a retrieval layer over a growing pile, not a replacement for bounded state. You still need to cap what goes in. A vector store that returns 50 chunks per query is just a slower way to blow your context budget.

**"The agent should just remember."** Models don't remember between calls. Every call is stateless. If you're not persisting state explicitly, it doesn't exist.

**"Tool results should be kept for audit."** Audit them to a log file, not to the prompt. The audit log and the working set are different artifacts with different lifetimes.

## The advanced version (once the basics are solid)

Once you have external state and bounded working sets, the next problems are retrieval quality and checkpointing.

Retrieval quality: a flat `LIMIT 20` on decisions works until it doesn't. When an agent handles 20 different packages and needs to know about package #3 specifically, recency is the wrong axis. The fix is hybrid retrieval — filter by entity ("decisions where package = lodash") first, then rank by recency within that. This is a SQL `WHERE` clause, not a vector search, for structured state. Vectors are for unstructured state like changelog text where you don't have a clean key.

Checkpointing: a 3-day run will survive process restarts, deploys, and OOM kills. If your state is only in memory, you lose it. SQLite (as above) is fine for single-writer agents. For concurrent agents, Redis 7.2 with AOF persistence or Postgres 16 both work; Redis is faster for the read-heavy pattern, Postgres is better if you want transactional guarantees across multiple state tables.

Here's a checkpoint pattern using Redis 7.2 that survives a worker restart mid-cycle:

```javascript
import { createClient } from 'redis';

const redis = createClient({ url: process.env.REDIS_URL });
await redis.connect();

const CYCLE_TTL_SECONDS = 60 * 60 * 6; // 6 hours, generous for one cycle

async function checkpoint(agentId, cycleState) {
  // Working set: TTL'd, gone after the cycle.
  await redis.set(
    `agent:${agentId}:cycle`,
    JSON.stringify(cycleState),
    { EX: CYCLE_TTL_SECONDS }
  );
  // Durable state: no TTL, survives restarts.
  await redis.hSet(`agent:${agentId}:decisions`, {
    [cycleState.package]: JSON.stringify({
      version: cycleState.version,
      action: cycleState.action,
      ts: Date.now(),
    }),
  });
}

async function resume(agentId) {
  const raw = await redis.get(`agent:${agentId}:cycle`);
  if (!raw) return null; // cycle completed or expired; start fresh
  return JSON.parse(raw);
}
```

The distinction between the two Redis keys is the whole point. The `cycle` key has a TTL because it's scratch space. The `decisions` hash has no TTL because it's the agent's actual memory. Getting this backwards — TTL on decisions, no TTL on cycle — produces an agent that forgets everything and a Redis instance that fills with garbage.

One more advanced concern: token accounting. You should log prompt token counts per cycle. A typical healthy long-running agent stays under 10k input tokens per cycle even at hour 200. If you see it creeping toward 40k, something is leaking into the working set. That's your alarm.

| Strategy | Context cost per cycle | Survives restart | Best for |
|---|---|---|---|
| Full transcript append | Grows ~1.5k/cycle | No | Short chat, <50 turns |
| Summarize-and-continue | Grows ~200/cycle | No | Chat with soft constraints |
| External state + retrieval | Flat ~8-10k | Yes | Multi-hour/day agents |
| External state + vector retrieval | Flat ~12-15k | Yes | Unstructured history search |
| Hierarchical (state + summary tiers) | Flat ~10k | Yes | Very long runs, 1000+ cycles |

## Quick reference

- **Context window is a working set, not a log.** Cap it explicitly per cycle.
- **Constraints go in verbatim, every cycle.** Never summarize them.
- **Decisions go in a structured store** (SQLite, Redis 7.2, Postgres 16). Query by entity, not just recency.
- **Tool results are scratch.** Truncate by character count, don't summarize.
- **Working set gets a TTL. Durable state doesn't.** Getting this backwards is the #1 bug.
- **Log prompt tokens per cycle.** >40k input tokens on a simple agent means a leak.
- **Models degrade on long contexts** (lost-in-the-middle). Fitting isn't the same as working.
- **Every model call is stateless.** If you didn't persist it, it's gone.

## Frequently Asked Questions

**How many tokens can a long-running agent actually use before it degrades?**

It's not a hard number, but empirically most agents show quality drops well before the model's stated limit. On a 128k model, degradation often shows up past 30-40k of mixed context, especially when key constraints are buried in the middle. The practical move is to keep per-cycle input under 10k tokens regardless of the model's ceiling. That gives you headroom and keeps you in the region where retrieval is reliable.

**Why does my agent forget instructions I gave it at the start?**

Because those instructions got summarized or evicted as the transcript grew. If you're using summarize-and-continue, the summarizer has no way to know which details are load-bearing. The fix is to move constraints out of the transcript entirely and inject them verbatim on every call. They're small — a few hundred tokens — and they're the one category that should never be compressed.

**Should I use a vector database for agent memory?**

Only for unstructured history where you don't have a clean key. If you can answer "what do I need?" with a SQL `WHERE` clause, use SQL. Vector search adds latency (typically 20-80ms per query plus embedding cost) and returns fuzzy results, which is the wrong tradeoff for structured state like decisions and constraints. Use vectors for "find past changelogs that mentioned breaking changes," not for "what's the status of package lodash."

**How do I handle an agent that runs for days and gets killed mid-cycle?**

Checkpoint the cycle state to a TTL'd key and the durable decisions to a non-TTL store, as in the Redis example above. On startup, try to resume the cycle; if the key expired, start fresh. The important part is idempotency: recording a decision twice should be a no-op (`INSERT OR REPLACE` or `HSET`), so a restart mid-write doesn't corrupt state. If your agent can't survive a `kill -9` at any point, it's not ready to run for days.

## Further reading worth your time

Anthropic's long-context prompting guidance is worth reading in full — the lost-in-the-middle finding is the single most useful thing to internalize before you build anything that runs past an hour. The original "Lost in the Middle" paper (Liu et al., 2026, arXiv:2307.03172) is the source. For state management patterns, the Kafka Streams documentation on state stores is a good mental model even if you never touch Kafka — it's the same bounded-state-with-checkpointing problem, solved at a different layer. And if you're running agents on Kubernetes, the Kubernetes documentation on liveness probes is a useful reminder that a process that's alive isn't necessarily healthy; your agent needs an equivalent signal for "my context is leaking."

**Do this in the next 30 minutes:** open your agent's main loop file and search for where you build the prompt. Count the categories of things you're putting in. If constraints and tool results are in the same list, split them — pull constraints into a module-level constant that gets injected verbatim, and cap tool results with a character slice. Then add one log line printing the input token count per cycle. That single number will tell you within a day whether you have a leak.


---

### About this article

**Written by:** [Kubai Kevin](/about/) — software developer based in Nairobi, Kenya, with 10+ years building production systems in fintech and AI.

**How this article was produced:** This site uses an automated LLM pipeline designed and maintained by the author. Topics are selected from real production experience. Drafts pass automated quality gates (minimum length, uniqueness, concrete metrics, versioned tools, code samples, absence of filler). Individual line-by-line human editing is not performed on every post before publication. Specific numbers, benchmarks and cost figures are illustrative; verify them against current official documentation before production use.

**Corrections:** Report errors via the [contact page](/contact/). Corrections are applied promptly.

**Last generated:** September 2026
