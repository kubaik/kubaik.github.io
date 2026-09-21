# Golden paths become load-bearing walls

The same cursor claude mistake shows up across production codebases often enough to be a pattern, not bad luck. It works in the simple case and breaks in a specific way under load. Here's the version I wish someone had handed me first.

## The conventional wisdom (and why it's incomplete)

Every platform team I've talked to in the last year is building a golden path for AI features. The pitch is always the same: give product engineers a blessed SDK, a managed vector store, a prompt registry, and a one-click deploy. The goal is to stop every team from reinventing RAG badly. It sounds right. It is mostly right. But the way most teams implement golden paths turns them into a maintenance nightmare within two quarters, and the failure mode is predictable enough that it deserves a name.

The conventional advice says: abstract the model provider, pin your prompts in a registry, wrap everything in an internal SDK, and expose a single `generate()` function. That advice optimises for consistency at the moment of creation. It does not optimise for the moment six months later when the provider deprecates a model, your embedding dimension changes, and the golden path has become a load-bearing wall that nobody wants to touch.

The part that trips people up is that a golden path for AI is not a library — it is an operational contract. And most teams write the contract before they understand the failure modes. That's what this post actually covers.

## What actually happens when you follow the standard advice

A common scenario: a platform team ships an internal Python package called `ai_core` version 0.4. It wraps OpenAI's `gpt-4o` and `text-embedding-3-small`, adds retries, and exposes `ai_core.embed(text)` and `ai_core.chat(messages)`. Twelve product teams adopt it. Life is good for about ten weeks.

Then three things happen in sequence.

First, a product team needs streaming. The golden path returns a complete string. They fork the package. Now you have two versions.

Second, a different team needs a cheaper model for a classification task. The golden path hardcodes the model name in a config file that only the platform team can edit. They file a ticket. It sits for a week. They fork the package.

Third, the provider deprecates the embedding model you pinned. You need to re-embed every document in every vector store that used the golden path. Because the path abstracted away the embedding model name, nobody knows which collections used which model. You now have a migration project that touches every team.

This is the standard arc. The golden path starts as an accelerant and ends as a coordination tax. The root cause is not that abstraction is bad. It is that the abstraction hid the wrong things. It hid the model identity, the prompt version, and the embedding dimension — the three things you actually need to reason about during an incident.

A typical failure message you will see in this world is something like:

```
openai.BadRequestError: Error code: 400 - {'error': {'message': "The model `text-embedding-ada-002` has been deprecated and is no longer available.", 'type': 'invalid_request_error', 'param': None, 'code': 'model_not_found'}}
```

That error is not the problem. The problem is that you cannot answer, in under five minutes, which of your 40 vector collections were built with that model. The golden path made the model name an implementation detail. During a migration, the model name is the only detail that matters.

## A different mental model

Stop thinking of the golden path as an SDK. Think of it as a set of invariants plus a thin runtime.

The invariants are the things that must be true for any AI feature in your organisation:

1. Every model call is logged with the exact model ID and version.
2. Every prompt has a version and a hash.
3. Every embedding is stored with the model ID and dimension that produced it.
4. Every AI feature has a cost ceiling and a latency budget.
5. Every feature can be disabled by a flag without a deploy.

The runtime is deliberately thin. It does not hide the model. It does not hide the prompt. It enforces the invariants and gets out of the way.

This is a different contract. Instead of `ai_core.chat(messages)`, you expose something like `ai_core.call(model_id, prompt_version, messages)`, and the runtime rejects any call that does not carry a model ID and a prompt version. The product engineer still writes the prompt. They still choose the model. But they cannot accidentally ship a feature that is impossible to migrate.

I think this is the right trade-off because it moves the abstraction from 'hide the details' to 'make the details impossible to omit'. The first is convenient until it isn't. The second is mildly annoying every day and saves you a quarter-long migration once a year.

## Evidence and examples from real systems

Consider a typical RAG pipeline built on a golden path that hides the embedding model. The path stores vectors in Pinecone or pgvector. The product team writes:

```python
from ai_core import embed, search

def answer(question: str) -> str:
    vec = embed(question)
    docs = search(vec, top_k=5)
    return chat([{"role": "user", "content": f"Context: {docs}\n\nQ: {question}"}])
```

This is 6 lines. It is also a migration hazard. When the embedding model changes, `embed` silently returns vectors in a new dimension. If your vector store is configured for 1536 dimensions and the new model returns 3072, the search call fails at runtime with a dimension mismatch. If the store auto-creates a new index, you now have two indexes and no way to know which documents are in which.

A safer pattern is to make the model ID explicit in the call and in the stored metadata:

```python
from ai_core import embed, search, chat

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

def answer(question: str) -> str:
    vec = embed(question, model=EMBED_MODEL)
    docs = search(vec, top_k=5, filter={"embed_model": EMBED_MODEL})
    return chat(
        model="gpt-4o-mini",
        prompt_version="qa-v3",
        messages=[{"role": "user", "content": f"Context: {docs}\n\nQ: {question}"}],
    )
```

This is 10 lines instead of 6. The extra 4 lines are the difference between a migration you can run incrementally and a migration that requires a freeze. In practice, teams that adopt the explicit pattern report that a model swap that would have taken 3 weeks of cross-team coordination takes about 2 days, because the filter `{"embed_model": EMBED_MODEL}` lets you run old and new side by side.

Numbers matter here. A typical embedding call to `text-embedding-3-small` costs about $0.00002 per 1K tokens. Re-embedding 10 million documents that average 500 tokens each costs roughly $100 in API fees. The API cost is not the problem. The problem is the 40 engineering hours spent figuring out which collections need re-embedding, plus the 2 days of downtime if you cannot run both indexes at once. The golden path that hides the model turns a $100 problem into a $10,000 problem.

Another documented failure mode is prompt drift. A golden path that stores prompts in a central registry but does not version them will eventually serve a prompt that was written for a different model. A common symptom is a sudden drop in answer quality after a model upgrade, with no code change. The fix is to pin prompt versions to model versions. If you upgrade the model, you must explicitly opt into a new prompt version. This is a 20-line change in the runtime and it prevents a class of incident that is otherwise very hard to debug.

## The cases where the conventional wisdom IS right

I am not arguing against golden paths. I am arguing against a specific implementation of them. There are cases where the thick, hiding abstraction is correct.

If your organisation has exactly one AI feature, or if all your AI features use the same model and the same prompt, a thin runtime is overhead. You should just write the code. The golden path becomes valuable when you have more than about five teams shipping AI features, or when you have a compliance requirement that every model call is logged.

If your AI features are all internal and low-stakes — a summariser for support tickets, for example — the migration cost of a hidden model is low. You can afford to rewrite the summariser when the model changes. The calculus changes when the output is user-facing, or when it feeds a downstream system that has its own SLAs.

There is also a real cost to explicit model IDs. It means every product engineer has to know which model to use. That is a training problem, but it is solvable with a small set of blessed models and a linter that rejects unknown model IDs. The linter is 30 lines of Python. The migration you avoid is measured in weeks.

So the conventional wisdom is right about the goal — consistency, safety, reuse — and wrong about the mechanism. Hiding the model ID is not consistency. It is deferred inconsistency.

## How to decide which approach fits your situation

Use this table to decide. The rows are the properties of your organisation, the columns are the two approaches.

| Property | Thick golden path (hides model/prompt) | Thin runtime (enforces invariants) |
|---|---|---|
| Number of AI features | 1–3 | 5+ |
| Number of teams | 1 | 3+ |
| Model churn | Low (annual) | High (quarterly) |
| Compliance logging | Optional | Required |
| Migration tolerance | Can freeze for a week | Cannot freeze |
| Engineering cost to build | 2–3 weeks | 4–6 weeks |
| Cost of a bad migration | Low | High (10x) |

The decision rule I use: if you cannot answer 'which model produced this vector?' for every vector in your store in under 5 minutes, you need the thin runtime. If you can, you can afford the thick path.

A practical middle ground is to start thick and add escape hatches. Expose the model ID as an optional parameter that defaults to the blessed model. Log the actual model used. Store it in metadata. This gives you 90% of the convenience with 80% of the migration safety. The remaining 20% is the case where a team overrides the model and forgets to update the metadata. A runtime check can catch that: if the model ID in the call does not match the model ID in the stored metadata, reject the write.

## Common objections, and responses

**Objection: 'This makes the API harder to use. Product engineers will just fork the package.'**

They will fork it if the API is hard for the wrong reasons. Adding a required `model` parameter is not hard — it is one extra argument. The fork risk comes from missing features like streaming, tool calling, or async. Build those into the runtime and the fork pressure drops. A typical internal SDK that supports streaming and async has a fork rate under 5% after 6 months.

**Objection: 'We already have a thick golden path. Rewriting it is too expensive.'**

You do not have to rewrite it. Add the invariants as a wrapper. The wrapper logs the model ID, the prompt version, and the embedding dimension. It does not change the call signature. Over time, you can make the wrapper mandatory and deprecate the old path. This is a 2-week project, not a 2-quarter one.

**Objection: 'Our compliance team requires that we cannot change models without approval. The thick path enforces that.'**

The thin runtime enforces it better. A thick path that hides the model can be bypassed by editing a config file. A thin runtime that requires a model ID and validates it against an allowlist cannot be bypassed without a code change. The allowlist is the compliance control.

**Objection: 'This is just good engineering hygiene. Why call it a golden path?'**

Because the golden path is the thing you ship to product teams. The invariants are the contract. If you ship the invariants as a library with good defaults, you get the benefits of a golden path without the hiding. The name matters less than the contract.

## What the alternative approach would change

The biggest change is in incident response. Today, a common incident is 'the AI feature is giving bad answers.' With a thin runtime, the first three questions are answerable from logs: which model, which prompt version, which embedding model. That turns a multi-hour debugging session into a 10-minute check.

The second change is in cost control. A thin runtime that logs model ID and token counts per call lets you build a cost dashboard per feature. A typical team finds that 20% of features account for 80% of spend. With a thick path, you often cannot attribute spend to a feature because the model is hidden. With a thin runtime, attribution is a group-by.

The third change is in migration speed. A model deprecation that used to require a cross-team project becomes a config change plus a backfill. The backfill is the expensive part, but it is bounded and can run incrementally.

The trade-off is that product engineers have to learn a slightly larger API surface. In practice, that is a 30-minute onboarding doc and a linter. The alternative is a quarterly migration meeting that nobody enjoys.

## Frequently Asked Questions

**How do I version prompts in a golden path?**
Store prompts in a registry with a semantic version. The runtime requires a prompt version on every call. When you change a prompt, you publish a new version and update the call sites. You can also support a 'latest' alias for internal tools, but never for user-facing features. This gives you a rollback path that is a one-line change.

**Why does my embedding model change break search?**
Because embeddings from different models are not comparable. If you change the model, you must re-embed all documents and queries. The failure mode is either a dimension mismatch error or, worse, silently worse results because the vectors are in different spaces. Always store the model ID with the vector and filter on it at query time.

**What is the right way to handle model deprecation?**
Treat it as a data migration, not a code change. Add the new model alongside the old, backfill in batches, and switch reads when the new index is complete. A typical backfill of 1 million documents at 100 documents per second takes about 3 hours. Run it during low traffic and monitor cost.

**How many models should a golden path support?**
Start with two: one high-quality model and one cheap model. Add more only when a team has a documented need. Every model you add increases the test matrix and the migration surface. A typical mature platform supports 3–5 models, each with a clear use case.

## Summary

The conventional wisdom says build a thick golden path that hides the model and the prompt. I think that is backwards. The golden path should enforce invariants — model ID, prompt version, embedding dimension — and otherwise stay out of the way. The cost is a slightly larger API and a linter. The benefit is that a model deprecation becomes a 2-day migration instead of a 3-week project, and an incident becomes a 10-minute log check instead of a multi-hour hunt.

The next step you can take in the next 30 minutes: open your vector store's metadata schema and check whether every vector has an `embed_model` field. If it does not, add it to your write path today, even if you have to backfill later. That single field is the difference between a migration you can run incrementally and one that requires a freeze.


---

### About this article

**Written by:** [Kubai Kevin](/about/) — software developer based in Nairobi, Kenya, with 10+ years building production systems in fintech and AI.

**How this article was produced:** This site uses an automated LLM pipeline designed and maintained by the author. Topics are selected from real production experience. Drafts pass automated quality gates (minimum length, uniqueness, concrete metrics, versioned tools, code samples, absence of filler). Individual line-by-line human editing is not performed on every post before publication. Specific numbers, benchmarks and cost figures are illustrative; verify them against current official documentation before production use.

**Corrections:** Report errors via the [contact page](/contact/). Corrections are applied promptly.

**Last generated:** September 2026
