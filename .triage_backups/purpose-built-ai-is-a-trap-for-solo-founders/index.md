# Purpose-built AI is a trap for solo founders

Most write-ups stop exactly where the interesting part starts: what happens to an AI pipeline six months after it ships. The same framework-versus-SDK trade-off shows up across production codebases often enough to be a pattern rather than bad luck. This article works through that trade-off properly, with code and failure modes.

## The conventional wisdom (and why it's incomplete)

The conventional wisdom says: if you're building an AI product, use a purpose-built AI platform. Frameworks in this category promise to handle memory, tool calling, retries, and agent orchestration out of the box. The pitch is seductive: a pre-built abstraction layer so you can focus on your product, not plumbing.

That advice is usually written for teams with platform engineers. A solo founder or a two-person team is the platform engineer. Every abstraction you adopt is a dependency you now maintain, a version you must track, and a failure mode you must debug at 2 AM. Purpose-built AI platforms optimize for a different constraint than a small team's: they optimize for feature velocity in a large organisation, not for operational simplicity in a one-person shop.

The abstraction looks free until it breaks, and then you're debugging someone else's opinionated stack instead of your own code. That is the real subject of this article.

## What actually happens when you follow the standard advice

A common failure mode: a team starts with a framework's high-level RAG helper, wires up a retrieval pipeline, and ships. Six weeks later, they need to change how documents are chunked. The chunking logic is buried three layers deep in a text splitter that is instantiated inside an index creator that is called from a question-answering chain. Changing the chunk size is no longer a one-line edit; it requires understanding the framework's internal contracts.

This usually shows up when you try to do something the framework authors didn't anticipate. Maybe you want to cache embeddings in Redis instead of the default in-memory store. Maybe you want to stream partial results to the client. Maybe you want to add a fallback model when the primary API returns a 429. Each of these is a small change in a hand-rolled pipeline and a multi-day investigation in a framework.

Dependency count is the part you can actually verify on your own machine, and it is worth verifying. A framework-based pipeline commonly pulls in dozens of transitive dependencies. A minimal hand-rolled pipeline using the provider's Python SDK plus a vector database client pulls in a handful. That difference is not just install time — it is surface area for CVEs, version conflicts, and breaking changes. Major-version migrations in this ecosystem have historically required touching a long list of call sites. For a small team, that is a week not spent on the product.

The failure modes are worse than the dependency count. Framework abstractions tend to swallow errors. A tool call fails, the agent retries, the retry fails, and you get a generic output-parser exception with no indication of which tool failed or why. In a hand-rolled pipeline, you control the error handling. You log the exact request, the exact response, and the exact exception. Debugging takes minutes, not hours.

## A different mental model

Treat AI providers as infrastructure, not as frameworks. The provider (OpenAI, Anthropic, Google) gives you an API. That API is documented and versioned. The framework sits on top of that API and adds opinion. Opinion is what you want when you agree with it and what you fight when you don't.

For a small team, the default should be: use the provider SDK directly, write your own thin orchestration layer, and only adopt a framework when you have a specific, painful problem it solves better than a few hundred lines of your own code.

This doesn't mean reinventing everything. You should still use:

- A vector database (Pinecone, Qdrant, or pgvector on Postgres) for retrieval
- A job queue (Redis with RQ or Celery) for async processing
- A structured logging library (for example structlog in Python) for observability
- A tracing tool (Langfuse or OpenTelemetry) for LLM-specific spans

What you shouldn't use is a framework that owns your control flow. The distinction is: libraries you call, versus frameworks that call you. Libraries are safe. Frameworks are a commitment.

Here's what a minimal RAG pipeline looks like without a framework:

```python
# Python 3.11, openai SDK, qdrant-client
import openai
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")
openai_client = openai.OpenAI()

def embed(text: str) -> list[float]:
    resp = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return resp.data[0].embedding

def retrieve(query: str, top_k: int = 5) -> list[str]:
    vec = embed(query)
    hits = client.search(
        collection_name="docs",
        query_vector=vec,
        limit=top_k,
    )
    return [h.payload["text"] for h in hits]

def answer(query: str) -> str:
    context = "\n\n".join(retrieve(query))
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer using only the context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
    )
    return resp.choices[0].message.content
```

That's roughly 30 lines. It does the same job as a high-level retrieval chain. It's debuggable, testable, and you own every line. When you need to change the chunking, you change the chunking. When you need to add a cache, you add a cache.

The same logic applies in TypeScript. Here's the equivalent with the Vercel AI SDK and pgvector:

```typescript
// Node 20 LTS, ai SDK, @ai-sdk/openai
import { openai } from '@ai-sdk/openai';
import { embed, generateText } from 'ai';
import { sql } from './db';

export async function answer(query: string): Promise<string> {
  const { embedding } = await embed({
    model: openai.embedding('text-embedding-3-small'),
    value: query,
  });

  const rows = await sql`
    SELECT text FROM documents
    ORDER BY embedding <=> ${JSON.stringify(embedding)}::vector
    LIMIT 5
  `;

  const context = rows.map((r: { text: string }) => r.text).join('\n\n');

  const { text } = await generateText({
    model: openai('gpt-4o-mini'),
    system: 'Answer using only the context.',
    prompt: `Context:\n${context}\n\nQuestion: ${query}`,
  });

  return text;
}
```

Again, about 25 lines. No framework. No hidden control flow. The Vercel AI SDK is a library here — you call it, it doesn't call you.

## Failure modes you can reason about

The case for the thin approach comes from failure modes, not benchmarks. Purpose-built platforms fail in ways that are hard to diagnose because the failure is in the abstraction, not in your code. Three concrete examples follow.

**Memory and summarisation.** A framework's memory module stores conversation history in a format you don't control. After enough turns, the context window fills up. The framework's default summarisation kicks in, but it summarises in a way that loses details your users care about. Users complain that the assistant "forgot" something they said earlier. If the summarisation prompt is hardcoded in the framework's source, you either fork the framework or abandon it. In a hand-rolled system, you decide when to summarise and what to keep. A simple heuristic — keep the last N turns verbatim and summarise everything before that with a prompt you wrote and can tune — is a few dozen lines and full control.

**Rate limits.** The OpenAI API returns HTTP 429 with a `Retry-After` header when you exceed a limit. A framework might retry with exponential backoff, but it might also swallow the error and return a generic failure. You don't know whether your request failed because of rate limits, a bad API key, or a malformed prompt. In your own code, you handle the 429 explicitly:

```python
import time
import openai

client = openai.OpenAI()

def call_with_retry(messages, max_retries=5):
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
            )
        except openai.RateLimitError as e:
            wait = float(e.response.headers.get("retry-after", 2 ** attempt))
            time.sleep(wait)
        except openai.APIError:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("max retries exceeded")
```

That's about 15 lines. You know exactly what happens on each error type. You can log it, alert on it, and tune it. The framework version of this is opaque.

**Version churn.** When a framework makes a breaking change, every call site in your code is a candidate for migration. When a provider SDK makes a breaking change, the surface is smaller and the changelog is usually explicit about what moved. That asymmetry compounds over the life of a product.

## How to measure the trade-off for your own pipeline

Published numbers about framework overhead are rarely reproducible across machines and versions, so measure on your own workload instead. The instrumentation is cheap.

**Dependency surface.** Count transitive dependencies for each candidate approach:

```bash
pip install pipdeptree
pipdeptree --packages <your-ai-package> | wc -l
```

For Node, `npm ls --all | wc -l` gives a comparable count. Record the number before and after adding a framework.

**Cold start and import time.** Time the import in isolation:

```bash
python -X importtime -c "import your_module" 2>&1 | tail -n 20
```

This shows cumulative import time per module, which is where framework overhead usually hides.

**Resident memory.** Run your service and sample RSS after the first request completes, then again after 100 requests:

```bash
ps -o rss= -p <pid>
```

Compare the delta between the two approaches under the same load.

**Per-call latency.** Wrap your retrieval and generation calls in timers and log p50 and p95. The difference between approaches usually shows up in the tail, not the median, because abstraction layers add work on retries and error paths.

**Error-path visibility.** Deliberately trigger a rate limit (a low-quota test key works) and a malformed request. Note how long it takes to identify the cause from logs alone. This is the measurement that matters most in practice and the one least likely to appear in any comparison table.

Run these five measurements once, on your real workload, before committing to either approach. The result is far more useful than any general claim, including the ones in this article.

## The cases where the conventional wisdom is right

The conventional wisdom isn't wrong everywhere. There are cases where a purpose-built platform is the right call.

First, if you're building a prototype to validate an idea, a framework gets you to a demo faster. If you need to show a working agent to a potential customer tomorrow, a framework can get you there in hours. The mistake is treating the prototype as production. Prototypes are disposable. If the idea validates, expect to rewrite the orchestration layer.

Second, if your problem is genuinely complex — multi-agent collaboration with tool use, planning, and reflection — a framework encodes patterns you'd otherwise have to invent. If a framework's role-based agent model matches your problem, using it saves design time. But be honest: most "multi-agent" systems are single-agent systems with a few tools. Don't adopt a multi-agent framework because it sounds impressive.

Third, if you're in an enterprise with a platform team, the calculus changes. The platform team can own the framework, handle upgrades, and provide support. A solo founder has no such backup. The advice "use a framework" is often written by people who work at companies with platform teams, and they're not wrong for their context — they're just not writing for yours.

Fourth, if you need features that are genuinely hard to build yourself — a full evaluation harness with LLM-as-judge, or a tracing UI — a dedicated tool gives you those out of the box. Note the distinction: a tracing tool you integrate is a library, not a framework that owns your control flow. That distinction decides whether the tool adds maintenance burden or removes it.

## A decision checklist

Work through these in order. Stop at the first "yes".

1. **Is this a prototype?** Use whatever gets you to a demo fastest. Plan to rewrite before production.
2. **Can you describe the pipeline in ten lines of pseudocode?** If not, the framework is hiding control flow you'll need to understand eventually. Write the pseudocode first, then decide.
3. **Is the pain point a library-shaped problem?** Caching, tracing, structured logging, vector search — these are libraries. Adopt them without hesitation.
4. **Is the pain point orchestration-shaped?** Retries, tool loops, state machines. Try writing it yourself first. If it exceeds roughly 200 lines and you're still not done, look for a framework.
5. **If you adopt a framework, what is the exit cost?** Read the changelog for the last six months. Count breaking changes. If the release cadence is fast and breaking changes are frequent, budget maintenance time explicitly.
6. **Can you keep 80% of your code provider-agnostic?** If the framework forces provider-specific types through your business logic, the exit cost is high regardless of what the changelog says.

A useful heuristic underneath all of this: if you can't explain what your pipeline does in a ten-line pseudocode sketch, you don't understand it well enough to debug it at 2 AM. Frameworks often obscure this. Your own code rarely does.

## Objections and responses

**"You're reinventing the wheel."** No — you're choosing which wheels to buy. The provider SDK is the wheel. The framework is a car: it comes with an engine, a transmission, and a lot of opinions about how you should drive. Sometimes you want the car. Often you want the wheel.

**"Frameworks handle edge cases you haven't thought of."** True, and that's exactly the trade-off. They handle edge cases with their own opinions. When their opinion differs from yours, you're stuck. Handling the edge cases you know about and discovering the ones you don't is often the cheaper path.

**"You'll spend more time on plumbing."** Initially, yes. But the plumbing is simple: HTTP calls, JSON parsing, retries. What's hard is debugging a framework's internal state machine when it fails in production. The time saved upfront is often paid back with interest.

**"What about vendor lock-in?"** Using a provider SDK is less lock-in than using a framework, because the SDK is a thin wrapper over HTTP. Swapping one provider SDK for another is usually an afternoon of work. Swapping out a framework's abstractions is a multi-week project.

**"But the framework has a community and docs."** So does the provider SDK, and it's maintained by the people who run the API. When the provider changes something, they update their own SDK first. Framework maintainers have to catch up, and sometimes they don't.

## A worked example: adding a fallback model

To make the trade-off concrete, consider a requirement that appears in almost every production AI product: if the primary model returns a 429 or a 5xx, fall back to a secondary model.

In a thin pipeline, the change is local:

```python
PRIMARY = "gpt-4o-mini"
FALLBACK = "gpt-4o"

def generate(messages):
    try:
        return call_with_retry(messages, model=PRIMARY)
    except (openai.RateLimitError, openai.APIError) as primary_error:
        try:
            return call_with_retry(messages, model=FALLBACK)
        except Exception as fallback_error:
            raise RuntimeError(
                f"primary failed: {primary_error!r}; "
                f"fallback failed: {fallback_error!r}"
            ) from fallback_error
```

The reasoning is visible: try the cheap model, retry on transient errors, fall back to the expensive model, and if both fail, raise an error that names both causes. Two failure modes are handled explicitly, and the error message tells you which model failed and why.

In a framework, the same requirement typically means finding the right configuration hook, checking whether the framework's retry logic runs before or after the fallback, and confirming that the fallback path preserves your message format. If the framework's retry wrapper catches the exception before your fallback sees it, the fallback never fires — and the failure is silent. That is the class of bug that costs a weekend.

The pattern generalises: any requirement that touches control flow (fallbacks, timeouts, circuit breakers, partial streaming) is cheap in code you own and expensive in code you don't.

## Summary

Purpose-built AI platforms are optimized for teams with platform engineers, not for solo founders and small teams. The abstraction they provide looks free until it breaks, and then you're debugging someone else's opinionated stack. The alternative — provider SDKs plus a thin orchestration layer — is more code upfront but less pain in production. Use frameworks for prototypes and genuinely complex problems, but keep your production pipeline thin, debuggable, and yours.

## FAQ

**Should I use a high-level AI framework for a production app?**
Only if it solves a specific problem better than your own code, and you're prepared to own the upgrade path. For most small teams, a thin pipeline using the provider SDK is simpler and more debuggable. Framework abstractions add dependencies and obscure control flow, which makes production debugging harder.

**How do I handle rate limits without a framework?**
Catch the provider's rate limit exception, read the `Retry-After` header, and sleep for that duration. Add exponential backoff for other transient errors. This is 15–20 lines of code and gives you full visibility. Frameworks often hide these errors behind generic exceptions.

**What's a reasonable vector database choice for a small team?**
Postgres with pgvector is the boring, proven choice: you already have Postgres for your app data, so you avoid running a second database. A dedicated vector database is a reasonable option if you need more scale or specialised indexing. A managed service is easy to start with but adds cost and a vendor dependency.

**How do I know when to adopt a framework?**
When you hit a problem you can't solve in a week with your own code, and the framework has a clear, documented solution. Before adopting, read the changelog for the last six months. If there are frequent breaking changes, budget maintenance time explicitly. If the framework is stable and solves your problem, it may be worth it.

## Take this action in the next 30 minutes

Open your `requirements.txt` or `package.json` and count how many dependencies your AI pipeline pulls in. If the count is higher than you expected, identify which ones are framework-specific and whether direct provider SDK calls could replace them. Start with the one that causes the most debugging pain, and write down its exit cost: how many call sites would change if you removed it.
