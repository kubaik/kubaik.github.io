# AI apps break when you treat them like regular software

## The conventional playbook and where it stops working

The standard advice for AI features is familiar: put embeddings in a vector store, call a model endpoint, return structured output, then scale horizontally and cache aggressively. That advice is not wrong so much as incomplete, because it was written for stateless request/response services and AI features are rarely stateless.

Three properties separate AI components from ordinary microservices:

- **Probabilistic output.** The same input can produce different outputs across calls. Cache keys built on request parameters alone will miss, or worse, serve a semantically different answer.
- **Accumulated state.** Embeddings, session history, retrieved context, and cached generations all persist and all go stale. State leaks into places a stateless design assumes are empty.
- **External dependency on a variable-latency service.** The model endpoint is often the slowest and least predictable hop in the request path, and its latency distribution shifts over time.

A typical failure mode is a system that passes staging and degrades under production traffic in ways that look like infrastructure problems but are actually design problems: connection pools sized for a stateless service, caches keyed on inputs that no longer determine outputs, and retrieval windows that quietly return month-old context.

## Failure mode 1: connection pool exhaustion behind a slow dependency

Consider a service that embeds a query, searches a vector index, and returns results. Under load, teams often scale the application tier first. If the bottleneck is the connection pool to the database backing the index, adding replicas makes things worse: each new replica opens its own pool, and the database's connection limit is reached sooner.

A concrete arithmetic example, with illustrative numbers:

- 12 application pods, each configured with a pool of 50 connections = 600 connections.
- The database allows 500 concurrent connections.
- Under load, 100 connection attempts queue. Each waits on a checkout timeout.
- If the checkout timeout is 5 seconds and the median query takes 6–8 seconds, connections are returned to the pool after the caller has already given up. The pool churns: connections are marked bad, dropped, and reopened, which costs more than the original query.

The visible symptom is latency spikes that do not correlate with CPU or memory. The actual cause is a timeout shorter than the work it guards.

**How to measure it.** Instrument pool checkout wait time as a separate histogram from query duration. In PostgreSQL, `pg_stat_activity` shows current connections by state; a growing count in `idle in transaction` or repeated connection resets in the server log points at pool churn. Compare p99 checkout wait against p99 query duration. If checkout wait is a meaningful fraction of query duration, the pool is the bottleneck, not the model.

**The fix is a sizing exercise, not a scaling exercise.** Set the checkout timeout above the p99 of the guarded operation, size the pool so that `pods × pool_size` stays below the database's connection limit, and consider a pooler that multiplexes many client connections onto fewer server connections. Adding replicas only helps once the per-connection cost is understood.

## Failure mode 2: state that accumulates and never gets evicted

Stateless designs assume memory is released when a request ends. AI features often break that assumption:

- Session context is appended to a conversation buffer and never trimmed, so the prompt grows until it hits the context limit or the cost per request becomes unacceptable.
- Embeddings are written for every document version, and old versions are never deleted, so the index grows and retrieval quality degrades as near-duplicates compete for the same top-k slots.
- Caches hold generated answers keyed on inputs that no longer map to the same output.

A common pattern is a service whose memory grows steadily over days of traffic, then fails when the process is restarted and the cache is cold. The restart itself is the incident.

**How to measure it.** Track resident memory per pod over a multi-day window, not a multi-hour one. Track index size and the ratio of distinct documents to total vectors — a ratio well below 1 means duplicates are accumulating. Track prompt token count per request as a distribution; a p99 that creeps upward over weeks is context growth, not traffic growth.

**Bound state explicitly.** Give every piece of accumulated state a maximum age and a maximum size:

- Conversation buffers: keep the last N turns plus a rolling summary, not the full history.
- Embeddings: version documents and delete superseded vectors on write, or run a compaction job on a schedule.
- Caches: set a TTL that reflects how long the underlying data is meaningful, not a default.

## Failure mode 3: cache hit rates that collapse under model variance

Caching is the first instinct for latency and cost, and it is the wrong first instinct for generative components when the cache key is the raw input.

Two problems compound:

1. **Semantic equivalence.** "How do I reset my password?" and "I forgot my password, how do I change it?" are the same intent but different strings. An exact-match cache misses both.
2. **Output variance.** Even with a stable key, the model may return different text on each call. A cache that stores one sample and serves it forever can serve an answer that no longer matches the model's current behavior.

A realistic sequence: a summarization service caches on input hash. The model provider ships an update that changes output length characteristics. Cache hit rate falls because more inputs are now producing outputs that differ from the cached version, and the service starts calling the model for requests it previously served from cache. The upstream model becomes the bottleneck, and p99 latency rises sharply.

**How to measure it.** Log cache hit, miss, and "stale-but-served" as three separate counters. Track the distribution of output lengths or token counts over time; a shift in that distribution is an early signal that cached outputs and fresh outputs have diverged. Track upstream request rate as a share of total request rate — if that share rises without a traffic increase, the cache is losing effectiveness.

**Design the cache around semantics, not strings.** Normalize inputs before keying (lowercase, strip punctuation, canonicalize intent where feasible), store the embedding alongside the cached entry, and use a similarity threshold for lookup. Set a TTL even on semantic hits, and version the cache by model identifier so a model change invalidates rather than silently serves stale output.

## Failure mode 4: staleness that is invisible to the user

Eventual consistency is a reasonable choice for recommendation, search, and support features. The failure is not staleness itself but unbounded, unobserved staleness.

A typical pipeline: user activity events land in a queue, a consumer updates a feature store, and a serving layer reads from it. If the consumer falls behind, the serving layer keeps returning recommendations based on old activity. There is no error, no failed health check, and no alert — the system is "up" and serving wrong answers.

**How to measure it.** The key metric is consumer lag, expressed in time rather than messages: how old is the oldest unprocessed event? Compare that against the staleness budget. Also measure the age of the data actually used to serve each response, and expose it. A response that reports its own data age is debuggable; one that does not is not.

**Bound staleness and degrade deliberately.** Define a staleness budget per feature — for example, personalized recommendations may use activity up to five minutes old. When lag exceeds the budget, serve a non-personalized fallback rather than stale personalized results. This converts a silent correctness failure into a visible, bounded quality reduction, and it keeps the system responsive while the consumer catches up.

The same principle applies to retrieval: if the index has not been refreshed within its window, either serve from a known-good snapshot or disclose the age of the results.

## Failure mode 5: model drift with no detection path

Models degrade as the world changes. A recommender trained on one interaction pattern returns less relevant results after a UI redesign changes how users behave. A classifier trained on one distribution of inputs sees different inputs in production.

Drift is not a single event, so detection should not be a single check. Practical signals, in rough order of how early they fire:

- **Input drift.** The distribution of incoming requests shifts — new languages, new document formats, longer inputs. Cheap to compute, fires early.
- **Embedding drift.** The distribution of embedding vectors shifts in norm or direction. Also cheap, and it does not require labels.
- **Output drift.** The distribution of model outputs changes — answer length, refusal rate, retrieval scores. Cheap, and often the first signal users would notice.
- **Outcome drift.** Business or quality metrics move — click-through, resolution rate, escalation rate. Expensive to collect and delayed, but the most meaningful.

**How to measure it.** Store a sample of production inputs and their embeddings. Compute a distribution distance between a reference window and the current window on a schedule. Alert on a threshold, not on a single observation. For outcome metrics, maintain a held-out evaluation set and re-run it on a schedule so you have a stable comparison point independent of live traffic.

**Close the loop.** Drift detection is only useful if it triggers something: a retraining job, a prompt or retrieval change, or a fallback to a more conservative model. A dashboard nobody acts on is not a drift strategy.

## A decision checklist

Before choosing between standard stateless design and an explicitly stateful, probabilistic design, answer these:

1. **What is the latency budget, and what fraction does the model call consume?** If the model call is most of the budget, caching and retrieval design matter more than the application tier.
2. **What state does a request depend on, and how old may it be?** Write down a staleness budget per state source. If you cannot state one, you cannot enforce one.
3. **What happens when the model is slow or unavailable?** Decide the fallback before launch: cached answer, degraded answer, or explicit error.
4. **How will you know the system is serving wrong answers?** Health checks measure liveness, not correctness. Name the metric that would move first.
5. **What is the eviction policy for every accumulating store?** Conversation buffers, embeddings, and caches all need one.
6. **How will you detect drift, and what will it trigger?** An alert with no action is not detection.

A short comparison of when each approach fits:

| Condition | Stateless design is fine | Explicit state and staleness design needed |
|---|---|---|
| Latency budget | Loose (seconds acceptable) | Tight, model call dominates |
| Traffic | Low and predictable | High or spiky |
| Request state | None beyond the request | Session, history, or live data |
| Output determinism | Deterministic or near-deterministic | Probabilistic, varies per call |
| Data freshness | Not user-visible | User-visible and must be bounded |
| Drift impact | Low, model rarely changes | High, distribution shifts over time |

## Where the conventional advice is genuinely correct

The standard playbook is not a straw man. It is the right answer in several common situations:

- **The AI component is a small feature in a larger system.** A single classification call inside an otherwise ordinary service does not justify a stateful architecture.
- **Traffic is low and predictable.** A single instance with a modest connection pool and a simple TTL cache will serve hundreds of requests per day comfortably.
- **The model is effectively deterministic.** A fixed-threshold model over stable features behaves like ordinary software; standard autoscaling and load balancing apply.
- **The workload is batch.** If results are generated on a schedule rather than on demand, latency budgets and cache design are far less constrained.
- **The team is small and operational maturity is low.** Complexity you cannot operate is worse than simplicity you can. Start with the standard design and add state management only where a measured failure demands it.

The mistake is not following the conventional playbook. The mistake is following it without checking whether its assumptions — statelessness, determinism, bounded dependencies — hold for the component in question.

## Common objections, examined

**"This architecture is too complex and expensive."**

Complexity here is not optional decoration; it is proportional to the problem. Stateful, probabilistic systems have state and variance, and those must be managed somewhere. The choice is whether you manage them explicitly or discover them during an incident. That said, complexity should be added in response to measured failures, not in anticipation of every possible one. Start with the simplest design that meets the latency and freshness budgets, and add bounded staleness, drift detection, and semantic caching when a specific metric shows they are needed.

**"We can just optimize the model instead."**

Model optimization and architecture optimization address different bottlenecks. Reducing model latency from 500 ms to 300 ms does not help if retrieval is doing a brute-force scan over millions of vectors, and shrinking a context window does not help if the system's real cost is re-fetching state on every request. Optimize the model, but measure where time is actually spent first — the two are complements, not substitutes.

**"Eventual consistency is too risky for our use case."**

For some domains — clinical decision support, transaction authorization — it genuinely is. For recommendation, search, and support, bounded staleness is usually acceptable, provided the bound is enforced and the age of the data is visible. The practical technique is to make freshness observable: expose the age of the context used to produce a response, and fall back to a non-personalized result when the bound is exceeded. Staleness that is disclosed and bounded is a design decision; staleness that is silent is a defect.

**"We don't have the data to detect drift."**

You do not need labels to start. Input and embedding distributions are computable from production traffic alone, and they fire earlier than outcome metrics. Begin with a reference window, a scheduled comparison, and a threshold. Add outcome-based detection when you have a held-out evaluation set or a reliable business metric.

## What to do first, in order

If you are building or repairing an AI feature, the highest-value work is usually observability before architecture:

1. Separate the latency histogram of the model call from the rest of the request path.
2. Instrument pool checkout wait as distinct from query duration.
3. Emit the age of every piece of state used to serve a response.
4. Track cache hit, miss, and stale-serve as three counters.
5. Record a sample of inputs and embeddings for later distribution comparison.

These five measurements will tell you which of the failure modes above you actually have, rather than which ones you might have.

## FAQ

**How do I know if my AI system needs explicit state and staleness handling?**

Check three things: whether a request depends on state beyond the request itself, whether the latency budget is dominated by a model call, and whether the age of the data used is visible to users. If any answer is yes, the standard stateless design will need modification. If all are no, start simple.

**What is the single most common mistake when scaling AI features?**

Treating the model endpoint and the vector store as ordinary stateless dependencies. That assumption leads to undersized connection pools, caches keyed on inputs that do not determine outputs, and state that accumulates without an eviction policy.

**Does an explicit state design cost more to run?**

Usually yes, because it involves more infrastructure and more instrumentation. The right way to evaluate it is to compare the cost of the added infrastructure against the cost of the failures it prevents — measured in error rate, latency SLO violations, and engineering time spent on incidents. Do not assume the added cost is justified; measure both sides.

**Can managed services remove the need for this work?**

Managed vector stores and managed model gateways remove some operational burden — sharding, index maintenance, and capacity management — but they do not remove the need to reason about staleness, cache validity, and drift. Those are properties of your data flow and your users' expectations, not of the hosting model.

**How do I set a staleness budget?**

Start from what the user would notice. If a recommendation reflects activity from five minutes ago, is that acceptable? If a support answer reflects a policy change from an hour ago, is that acceptable? Write the answer down per feature, then enforce it as a hard bound with a defined fallback when it is exceeded.

## One action for the next 30 minutes

Open the code path that serves your most latency-sensitive AI request and add a single metric: the age, in seconds, of the oldest piece of state used to produce the response — session history, retrieved document, cached generation, or feature value. Emit it as a histogram. If you cannot compute that number from the code as written, you have found the first thing to fix.
