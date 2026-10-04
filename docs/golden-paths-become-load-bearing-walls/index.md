# Golden paths become load-bearing walls

## The pattern: convenience now, coordination tax later

A recurring failure mode in production AI systems is the internal SDK that works beautifully in the simple case and becomes an obstacle during a model or embedding migration. The code is not buggy. The design is. It optimises for the moment of creation and ignores the moment, months later, when a provider deprecates a model and nobody can determine which stored vectors were produced by it.

The conventional advice is well known: abstract the model provider, pin prompts in a registry, wrap everything in an internal SDK, expose a single `generate()` function. That advice optimises for consistency at authoring time. It does not optimise for the incident, the deprecation notice, or the embedding dimension change that forces a backfill across every team.

The argument here is that a golden path for AI features is not a library. It is an operational contract. Most teams write the contract before they understand the failure modes. This article covers the failure modes first, then the contract.

## What happens when the standard advice is followed literally

Consider a common scenario. A platform team ships an internal Python package that wraps a chat model and an embedding model, adds retries, and exposes `embed(text)` and `chat(messages)`. Several product teams adopt it. For a while, everything works.

Then three things happen, usually in sequence.

First, a product team needs streaming. The golden path returns a complete string. The team forks the package. Now there are two versions to maintain.

Second, a different team needs a cheaper model for a classification task. The golden path hardcodes the model name in a config file that only the platform team can edit. The team files a ticket. It sits in a queue. The team forks the package.

Third, the provider deprecates the pinned embedding model. Every document in every vector store built through the golden path must be re-embedded. Because the path abstracted away the embedding model name, nobody can say which collections used which model. What should have been a bounded migration becomes a cross-team project.

This is the standard arc. The golden path starts as an accelerant and ends as a coordination tax. The root cause is not abstraction itself. It is that the abstraction hid the wrong things: the model identity, the prompt version, and the embedding dimension — the three facts needed to reason about an incident.

A representative error from this world looks like:

```
openai.BadRequestError: Error code: 400 - {'error': {'message': "The model `text-embedding-ada-002` has been deprecated and is no longer available.", 'type': 'invalid_request_error', 'param': None, 'code': 'model_not_found'}}
```

The error is not the problem. The problem is that no one can answer, in under five minutes, which of dozens of vector collections were built with that model. The golden path made the model name an implementation detail. During a migration, the model name is the only detail that matters.

## A different mental model: invariants plus a thin runtime

Stop thinking of the golden path as an SDK. Think of it as a set of invariants plus a thin runtime.

The invariants are properties that must hold for any AI feature in the organisation:

1. Every model call is logged with the exact model ID and version.
2. Every prompt has a version and a hash.
3. Every embedding is stored with the model ID and dimension that produced it.
4. Every AI feature has a cost ceiling and a latency budget.
5. Every feature can be disabled by a flag without a deploy.

The runtime is deliberately thin. It does not hide the model. It does not hide the prompt. It enforces the invariants and gets out of the way.

This is a different contract. Instead of `chat(messages)`, the runtime exposes something like `call(model_id, prompt_version, messages)`, and it rejects any call that does not carry a model ID and a prompt version. The product engineer still writes the prompt. They still choose the model. But they cannot accidentally ship a feature that is impossible to migrate.

The trade-off is deliberate. The abstraction moves from "hide the details" to "make the details impossible to omit." The first is convenient until it is not. The second is mildly annoying every day and saves a long migration once a year.

## A worked example: the RAG pipeline that cannot be migrated

Take a typical retrieval-augmented generation pipeline built on a golden path that hides the embedding model. The path stores vectors in a vector database. The product team writes:

```python
from ai_core import embed, search, chat

def answer(question: str) -> str:
    vec = embed(question)
    docs = search(vec, top_k=5)
    return chat([{"role": "user", "content": f"Context: {docs}\n\nQ: {question}"}])
```

Six lines. Also a migration hazard. When the embedding model changes, `embed` silently returns vectors in a new dimension. If the vector store is configured for 1536 dimensions and the new model returns 3072, the search call fails at runtime with a dimension mismatch. If the store auto-creates a new index, there are now two indexes and no way to know which documents are in which.

A safer pattern makes the model ID explicit in the call and in the stored metadata:

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

Ten lines instead of six. The extra four lines are the difference between a migration that can run incrementally and one that requires a freeze. The filter `{"embed_model": EMBED_MODEL}` lets old and new indexes coexist, so reads can be switched once the backfill completes.

### Reasoning about the cost, with stated assumptions

The API cost of re-embedding is usually small compared to the coordination cost. The arithmetic, using illustrative assumptions:

- Assume an embedding price of $0.02 per 1M tokens (a plausible order of magnitude for small embedding models; check current provider pricing).
- Assume 10 million documents at 500 tokens each: 10,000,000 × 500 = 5,000,000,000 tokens.
- 5,000,000,000 / 1,000,000 = 5,000 units of 1M tokens.
- 5,000 × $0.02 = $100 in API fees.

The API cost is not the problem. The problem is the engineering time spent determining which collections need re-embedding, plus any downtime if old and new indexes cannot run side by side. A hidden model turns a small, bounded backfill into an open-ended project.

### How to measure this in your own system

Do not take the numbers above as a benchmark. Measure the properties that matter:

- **Attribution latency.** Time how long it takes to answer "which embedding model produced this vector?" for a random sample of vectors. If the answer requires reading application code rather than querying metadata, that is the finding.
- **Backfill throughput.** Instrument documents embedded per second during a small test backfill. Divide total document count by that rate to get wall-clock time.
- **Cost per feature.** Log model ID and token counts per call, then group by feature. This is the only reliable way to attribute spend.
- **Incident triage time.** Record the wall-clock time from "bad answers reported" to "model and prompt version identified." Track it over time.

## Prompt drift: the second failure mode

A golden path that stores prompts in a central registry but does not version them will eventually serve a prompt written for a different model. A common symptom is a sudden drop in answer quality after a model upgrade, with no code change.

The fix is to pin prompt versions to model versions. Upgrading the model requires explicitly opting into a new prompt version. This is a small change in the runtime and it prevents a class of incident that is otherwise very hard to debug, because the code diff is empty.

## When the thick abstraction is the right call

The argument is not against golden paths. It is against one specific implementation of them. There are cases where a thick, hiding abstraction is correct.

If the organisation has exactly one AI feature, or all AI features use the same model and the same prompt, a thin runtime is overhead. Write the code. A golden path becomes valuable when several teams ship AI features, or when a compliance requirement demands that every model call is logged.

If AI features are all internal and low-stakes — a support-ticket summariser, for example — the migration cost of a hidden model is low. The summariser can be rewritten when the model changes. The calculus changes when the output is user-facing, or when it feeds a downstream system with its own SLAs.

There is a real cost to explicit model IDs: every product engineer has to know which model to use. That is a training problem, solvable with a small set of blessed models and a linter that rejects unknown model IDs.

The conventional wisdom is right about the goal — consistency, safety, reuse — and wrong about the mechanism. Hiding the model ID is not consistency. It is deferred inconsistency.

## Decision checklist

| Property | Thick golden path (hides model/prompt) | Thin runtime (enforces invariants) |
|---|---|---|
| Number of AI features | 1–3 | 5+ |
| Number of teams | 1 | 3+ |
| Model churn | Low (annual) | High (quarterly) |
| Compliance logging | Optional | Required |
| Migration tolerance | Can freeze for a week | Cannot freeze |
| Engineering cost to build | Lower | Higher |
| Cost of a bad migration | Low | High |

A useful decision rule: if the answer to "which model produced this vector?" cannot be obtained for every vector in the store in under five minutes, the thin runtime is warranted. If it can, the thick path is affordable.

A practical middle ground is to start thick and add escape hatches. Expose the model ID as an optional parameter that defaults to the blessed model. Log the actual model used. Store it in metadata. This provides most of the convenience with most of the migration safety. The residual risk is a team overriding the model and forgetting to update metadata. A runtime check can catch that: if the model ID in the call does not match the model ID in the stored metadata, reject the write.

## Common objections

**"This makes the API harder to use. Product engineers will fork the package."**

Teams fork when the API is hard for the wrong reasons. Adding a required `model` parameter is one extra argument. Fork pressure comes from missing features such as streaming, tool calling, or async. Build those into the runtime and fork pressure drops.

**"A thick golden path already exists. Rewriting it is too expensive."**

It does not have to be rewritten. Add the invariants as a wrapper. The wrapper logs the model ID, the prompt version, and the embedding dimension without changing the call signature. Over time, make the wrapper mandatory and deprecate the old path. This is a short project, not a multi-quarter one.

**"Compliance requires that models cannot change without approval. The thick path enforces that."**

A thin runtime enforces it better. A thick path that hides the model can be bypassed by editing a config file. A thin runtime that requires a model ID and validates it against an allowlist cannot be bypassed without a code change. The allowlist is the compliance control.

**"This is just good engineering hygiene. Why call it a golden path?"**

Because the golden path is what gets shipped to product teams. The invariants are the contract. Shipping the invariants as a library with good defaults provides the benefits of a golden path without the hiding. The name matters less than the contract.

## What changes in practice

The biggest change is in incident response. A common incident is "the AI feature is giving bad answers." With a thin runtime, the first three questions are answerable from logs: which model, which prompt version, which embedding model. That turns a multi-hour debugging session into a short check.

The second change is cost control. A thin runtime that logs model ID and token counts per call allows a cost dashboard per feature. Attribution becomes a group-by instead of an investigation.

The third change is migration speed. A model deprecation that used to require a cross-team project becomes a config change plus a backfill. The backfill is the expensive part, but it is bounded and can run incrementally.

The trade-off is a slightly larger API surface for product engineers. In practice, that is a short onboarding doc and a linter.

## FAQ

**How should prompts be versioned in a golden path?**

Store prompts in a registry with a version identifier. The runtime requires a prompt version on every call. When a prompt changes, publish a new version and update the call sites. A "latest" alias can be supported for internal tools, but not for user-facing features. This provides a rollback path that is a one-line change.

**Why does an embedding model change break search?**

Embeddings from different models are not comparable. Changing the model requires re-embedding all documents and queries. The failure mode is either a dimension mismatch error or, worse, silently worse results because the vectors live in different spaces. Always store the model ID with the vector and filter on it at query time.

**What is the right way to handle model deprecation?**

Treat it as a data migration, not a code change. Add the new model alongside the old, backfill in batches, and switch reads when the new index is complete. Run the backfill during low traffic and monitor cost. The wall-clock time is total documents divided by measured backfill throughput.

**How many models should a golden path support?**

Start with two: one high-quality model and one cheap model. Add more only when a team has a documented need. Every model added increases the test matrix and the migration surface. A mature platform typically supports a small set of models, each with a clear use case.

## Summary

The conventional wisdom says build a thick golden path that hides the model and the prompt. The better contract is the inverse: enforce invariants — model ID, prompt version, embedding dimension — and otherwise stay out of the way. The cost is a slightly larger API and a linter. The benefit is that a model deprecation becomes a bounded migration instead of a cross-team project, and an incident becomes a short log check instead of a long hunt.

In the next 30 minutes: open the vector store's metadata schema and check whether every vector has an `embed_model` field. If it does not, add it to the write path today, even if existing vectors must be backfilled later. That single field is the difference between a migration that can run incrementally and one that requires a freeze.
