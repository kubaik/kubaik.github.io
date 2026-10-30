# 9 Vibe coding traps that break production code

Vibe coding — prompting an assistant, pasting a snippet, and moving on when the happy path works — is genuinely effective for prototypes. It becomes expensive when the prototype is promoted to production without anyone examining the assumptions baked into the generated code. The traps below are not tool defects. They are recurring mismatches between what a snippet assumes and what a production runtime actually does.

## Why this list exists

The failure mode is consistent: code that passes a manual click-through test, ships, and then degrades under concurrency, cold starts, or schema change. Three representative shapes of that failure:

- A Node runtime on a serverless platform whose memory ceiling is exceeded because a module-level cache or listener accumulates across invocations. The symptom is not a crash but rising billed memory and creeping latency as the event loop backs up.
- A Python ASGI service that silently drops requests under load because the worker count was left at a default that assumes fewer CPU cores than the instance provides.
- A React app where effects run without dependency arrays, causing redundant re-renders on every route change and inflating interaction latency.

None of these require exotic tooling to detect. They require deciding, before you ship, what you will measure.

## How to evaluate any tool or pattern

Rather than trust a ranking, instrument the four numbers that predict production pain:

1. **Cold-start latency.** Deploy the smallest realistic handler and invoke it after a forced idle period. On AWS Lambda, compare `Init Duration` in CloudWatch Logs across memory settings and architectures. On container platforms, measure time from request receipt to first byte after a scale-to-zero event.
2. **Error rate under concurrency.** Use a load tool such as k6 and ramp to a concurrency level above your expected peak. Watch for non-2xx responses and, more importantly, for responses that succeed but return stale or partial data.
3. **Dependency and bundle cost.** For frontend code, run a bundle analyzer on the production build and record the parsed size of each chunk. For backend code, count transitive dependencies — a deep graph is a supply-chain and cold-start cost even when it is not a bundle cost.
4. **On-call surface.** Count how many distinct alerts a component can fire. A tool that generates five infrastructure stacks per deploy multiplies the number of things that can page you.

Record these before and after adopting a tool. A change that improves developer velocity but doubles cold-start latency is a trade, not a win, and you should be able to state which side you chose.

## The traps

### 1. Generated code with unexamined imports

**What it does:** An assistant generates a complete function, test, or infrastructure definition from a prompt, usually with idiomatic structure.

**Where it breaks:** Generated snippets frequently import an entire SDK or utility library for a single function. On the server this inflates cold-start time and dependency surface; in a browser bundle it ships code the user never executes. The code passes review because it reads well, not because anyone audited the import graph.

**What to do instead:** After accepting a generated file, run your bundle analyzer or dependency tree and confirm every import is used. Treat the import list as part of the review, not an afterthought.

### 2. Copy-pasted answers from stale threads

**What it does:** Provides a ready-made snippet for an unfamiliar error message or API.

**Where it breaks:** Highly ranked answers are often written for runtimes several major versions old. A pattern that was correct for a callback-based API can be subtly wrong for a promise-based one, and the failure appears only under concurrency.

**What to do instead:** Before adopting a snippet, check the answer date against your runtime version and read the current official documentation for that API. If the documentation contradicts the answer, the documentation wins.

### 3. Server components with client-side assumptions

**What it does:** Lets React components render on the server, reducing the JavaScript shipped to the browser.

**Where it breaks:** Streaming and suspense boundaries change when the client runtime becomes available. Third-party scripts that assume `document` or `window` exists at parse time fire too early, and analytics or tag managers silently lose events.

**What to do instead:** Load such scripts through the framework's script component with an explicit strategy, or move the logic into a server component. Verify by checking that your analytics tool records page views during a hard refresh, not just client-side navigation.

### 4. ORMs that move cost from SQL to the bundle

**What it does:** Generates a typed client from a database schema so queries can be written in the application language.

**Where it breaks:** Two distinct problems get conflated. First, a generated client that is imported into client-side code can pull a large amount of code into the browser bundle even when only one model is used. Second, query patterns that look cheap in application code can produce N+1 queries or hydration waterfalls that dominate time-to-interactive on slow connections.

**What to do instead:** Keep the ORM on the server. If client code needs data, expose a narrow endpoint. Measure the production bundle before and after, and log query counts per request in staging so N+1 patterns surface before they reach users.

### 5. Serverless databases with surprising branch semantics

**What it does:** Provides a managed database with branching, often used to create ephemeral environments for testing or previews.

**Where it breaks:** Branch behavior varies by product. Some branches are read-only; some replicate schema but not data. Tests that write to a branch then fail — sometimes silently, if the test asserts on a value that happened to be present.

**What to do instead:** Read the branch documentation for the specific product and write one test that performs a write and asserts on the result. If writes are not supported, restructure the test to seed a separate database.

### 6. Vector databases adopted for novelty

**What it does:** Provides managed similarity search over embeddings, typically for retrieval-augmented generation.

**Where it breaks:** Cost scales with stored vectors and query volume, and both grow silently. Increasing chunk count to improve retrieval quality multiplies storage; adding a reranking step multiplies queries. A prototype that costs little at a thousand vectors can become a recurring line item at a hundred thousand.

**What to do instead:** Instrument storage count and query volume from day one, and set a budget alert. Decide your chunking strategy before ingesting, because re-chunking means re-embedding and re-uploading the entire corpus.

### 7. Edge functions treated as free latency

**What it does:** Runs request handlers in data centers close to users.

**Where it breaks:** Edge runtimes are frequently more constrained than regional serverless runtimes. Cold starts can be longer, execution time limits are often shorter, and the runtime environment may lack APIs your code assumes. A handler that runs in 400 ms regionally can behave very differently at the edge.

**What to do instead:** Measure cold and warm latency from a location far from your origin before committing. Check the runtime's supported APIs against your dependencies. Reserve edge deployment for read-heavy, stateless handlers.

### 8. Caches with default eviction policies

**What it does:** Serves frequently read data from memory, typically for sessions, rate limits, or hot lookups.

**Where it breaks:** Default eviction policies are chosen for general safety, not for your access pattern. A policy that evicts by recency can remove a hot key that is expensive to recompute, and concurrent requests then all recompute it at once — a cache stampede. The result is a latency spike and a burst of load on the backing database.

**What to do instead:** Choose an eviction policy that matches your data. If keys have meaningful TTLs, a TTL-based policy avoids evicting keys that are still valid. Add jitter to TTLs so keys do not expire simultaneously, and consider request coalescing for expensive keys.

### 9. Infrastructure-as-code that generates more infrastructure than you expect

**What it does:** Lets you define cloud resources in a general-purpose language instead of a declarative template.

**Where it breaks:** Abstractions can expand into many underlying resources. Each deploy may create multiple stacks, each with its own roles and policies. Generated policies are often broader than necessary, widening the blast radius of a compromised function and making drift hard to reason about.

**What to do instead:** After the first deploy, inspect the generated resources and policies directly in the cloud console. Compare the permissions granted against the permissions the function actually uses. Narrow them, and treat the generated output as a starting point rather than a finished artifact.

## A worked example: diagnosing a rising-latency service

Suppose a service on a serverless platform shows p99 latency climbing over several hours while request volume is flat. The reasoning path:

1. **Rule out traffic.** Confirm request rate and payload size are unchanged. If they are, the change is internal.
2. **Check memory and duration together.** If billed memory is rising while CPU time per request is stable, something is accumulating between invocations — a module-level map, an unbounded listener, or a cache with no eviction.
3. **Look for event-loop delay.** A blocked event loop shows up as request duration growing while downstream call latency stays flat. Instrument the gap between when a request is received and when your handler begins.
4. **Bisect by reverting.** If the growth started at a specific deploy, revert it and confirm the metric recovers. That converts a hypothesis into a fact.
5. **Fix the root cause, not the symptom.** Raising the memory limit buys time and increases cost; it does not stop the accumulation. Add a bound to the structure that is growing, or move the state out of the process.

The same loop applies to the other traps: establish that the metric changed, isolate the component, and confirm the fix by measurement rather than intuition.

## How to choose based on your situation

| Situation | Approach | Main risk to watch |
|---|---|---|
| Solo prototype, weeks not months | Assistant-generated code, managed hosting | Unaudited imports and unbounded state |
| Small team, multi-month roadmap | Typed ORM on the server, managed database with branching | Bundle growth and N+1 queries |
| High-throughput API | Explicit SQL, tuned cache, measured worker counts | Index tuning effort and cache stampede |
| Global read-heavy app | Edge or CDN for static and cached content | Cold starts and runtime API gaps |
| Retrieval-augmented feature | Managed vector store with budget alerts | Silent growth in stored vectors and queries |

## FAQ

**Why do streaming server-rendered pages break analytics scripts?** The client runtime becomes available after the initial HTML is streamed. Scripts that expect `document` at parse time run before hydration and lose events. Load them with an explicit strategy or move them server-side.

**How do I estimate vector database cost before committing?** Multiply your document count by your average chunks per document to get stored vectors, then estimate queries per user session times sessions per day. Both numbers grow as you tune retrieval quality, so set an alert on each rather than a one-time estimate.

**Is raw SQL always cheaper than an ORM?** No. Raw SQL costs engineering time for schema changes and typos; an ORM costs bundle size and can hide inefficient query patterns. The right choice depends on whether your bottleneck is developer time or request latency.

**Why does a cache evict the key I use most?** Eviction policies operate on the metadata they are given, not on your intuition about importance. If a policy evicts by recency and your hot key is not recently written, it becomes a candidate. Match the policy to your access pattern.

## Final recommendation

Do not promote a prototype to production until you can state, for each external dependency, what happens under concurrency, cold start, and failure.

1. Pick the one component in your stack you understand least.
2. Write a load test that ramps above your expected peak and record error rate, p99 latency, and cold-start time.
3. Compare those numbers against the same test run with the component removed or replaced.

**Action for the next 30 minutes:** open your project, run your production build with a bundle analyzer enabled, and list every dependency contributing more than 50 KB to a chunk the user downloads on first load. For each one, decide whether it is used on the critical path. Remove or defer at least one.
