# 3 LLM context techniques: when each fails

Every team building on an LLM eventually hits the same wall: the model doesn't know something it needs to know. The instinct is to reach for the most powerful-sounding fix — fine-tuning, retrieval, or stuffing everything into a longer context window. Each is a real technique with a real cost curve, and each fails in a specific, predictable way. The confusion isn't that any of them is hard to understand in isolation. It's that they solve *different* problems, and the marketing around all three makes them sound interchangeable.

The framing that matters: fine-tuning changes *how* the model behaves, retrieval changes *what* the model can see at inference time, and long context changes *how much* it can see at once. Mix those up and you will spend weeks fine-tuning a model to teach it a fact it could have been handed in the prompt — and the fact will still be wrong the day after the next product launch.

What follows is three mini-reviews plus a decision framework. Each technique gets a concrete strength, a concrete failure mode, and the kind of problem it actually fits. The goal is to stop teams from picking the technique that sounds most advanced and start them picking the one whose failure mode they can live with.

## 1. Fine-tuning: the behavior shaper that forgets facts

Fine-tuning takes a base model and continues training it on your examples. The output is a new set of weights that has absorbed patterns from your data. What it does well is *style, format, and task adherence*: making a model consistently answer in a specific JSON schema, adopt a house tone, follow a multi-step reasoning pattern, or classify with your label taxonomy. If the problem is "the model can do this, but not reliably in the shape I need," fine-tuning is the right tool.

**Concrete strength:** Format compliance. A base model prompted to output strict JSON will occasionally wrap it in prose or add a trailing sentence. A model fine-tuned on a few hundred examples of clean JSON output does this far more consistently, and the regex to strip markdown fences goes away. The same applies to classification: a fine-tuned classifier can outperform a prompted one on a specific label set because the decision boundary is baked into the weights, not reconstructed from a prompt on every call.

**Concrete weakness:** Fine-tuning is a poor way to teach facts. Supervised fine-tuning does not reliably insert new knowledge into a model; it adjusts the distribution of outputs. Fine-tune on a document containing a pricing table and the model may reproduce that table when prompted in exactly the way the training examples were phrased, and produce a plausible variant otherwise. Worse, the knowledge is frozen at training time. The day pricing changes, the fine-tuned model is confidently wrong and must be retrained. There is also a real risk of catastrophic forgetting: fine-tune aggressively on a narrow task and the model can lose general capability it had before.

**When it fails:** Any problem where the answer changes faster than you can retrain, or where the model needs to cite a source. Fine-tuning gives no provenance. It cannot say *which document* an answer came from, because the answer is distributed across weights. For regulated domains where showing your work is mandatory, that is disqualifying.

**Best for:** Teams with a stable task definition, a few hundred to a few thousand high-quality labeled examples, and a need for consistent output shape or tone. Also the right call when latency or per-token cost matters and a smaller model can do a narrow job well.

A minimal fine-tuning job (generic trainer API shape — check the provider's docs for exact field names) looks like this:

```python
# Illustrative shape only — field names vary by provider.
import json

# Training data: one JSON object per line, prompt/completion pairs.
with open("train.jsonl", "w") as f:
    for pair in examples:
        f.write(json.dumps({
            "prompt": pair["input"],
            "completion": pair["output"],
        }) + "\n")

# The actual training call is provider-specific.
# job = client.fine_tuning.jobs.create(
#     training_file="train.jsonl",
#     model="<base-model-id>",
# )
```

The thing to notice is that the file contains *examples of behavior*, not a knowledge base. If `train.jsonl` is mostly facts, it is the wrong tool.

## 2. Retrieval-augmented generation: the fact supplier with a plumbing problem

RAG splits the problem in two: an offline step that chunks and indexes documents, and an online step that embeds the user's query, retrieves the top-k most similar chunks, and puts them in the prompt. The model then answers from what it was handed. What it does well is *freshness and provenance*: update the index and the model knows the new fact on the next call; every answer can cite the chunk it came from.

**Concrete strength:** The source can be pointed at. In a support or compliance context, "here is the passage this answer is based on" is often more valuable than the answer itself. RAG also allows retrieval to be scoped per user — filtered by tenant, permission, or document type — which is very hard to do with fine-tuning.

**Concrete weakness:** Retrieval quality is the whole ballgame, and it fails in ways that stay invisible until a user complains. The classic failure is the chunking boundary: a policy document says "Refunds are allowed within 30 days" on one page and "except for digital goods" on the next. Chunk them separately and the retriever may surface only the first, producing an answer that is confidently wrong and cites a real source. Embedding similarity is also not the same as relevance — a query about "cancellation policy" may retrieve a chunk about "cancellation of scheduled maintenance" because the words are close in embedding space. And multi-hop questions ("which of our vendors had a breach that affected the same region as our outage?") require retrieving across several documents and reasoning over them, which a single top-k pass often misses.

**When it fails:** When the answer requires synthesizing many documents, when the corpus is small enough to fit in context anyway, or when documents are so poorly structured that chunking destroys meaning. It also fails silently: a bad retriever returns *something*, so the model answers instead of saying "I don't know."

**Best for:** Teams with a large, changing corpus, a need for citations, and per-user access control. The canonical fit is internal documentation search, customer support over a knowledge base, or any domain where the source of truth is a set of documents already maintained.

A minimal retrieval loop, written to make the failure modes visible:

```python
# Illustrative — replace embed() and search() with your vector store's API.
def answer(question: str, index, llm) -> str:
    q_vec = embed(question)
    hits = index.search(q_vec, top_k=5)

    # The failure mode lives here: if hits is empty or irrelevant,
    # the model will still answer. Guard against it.
    if not hits or hits[0].score < 0.75:  # threshold is illustrative
        return "I don't have a source for that."

    context = "\n\n".join(
        f"[doc:{h.id}] {h.text}" for h in hits
    )
    prompt = (
        "Answer using only the sources below. "
        "Cite doc ids. If the sources don't contain the answer, say so.\n\n"
        f"Sources:\n{context}\n\nQuestion: {question}"
    )
    return llm(prompt)
```

The `score < 0.75` line is the most important one in the snippet. Most RAG failures are not retrieval failures — they are the absence of a refusal path. Without a threshold, the model treats weak hits as strong evidence.

## 3. Long context: the everything-at-once option with an attention bill

Modern models accept context windows measured in hundreds of thousands of tokens. The appeal is obvious: skip the chunking, skip the index, put the whole document (or the whole codebase) in the prompt. For a surprising number of problems, this is the correct and simplest answer.

**Concrete strength:** Zero infrastructure. No vector store, no embedding pipeline, no chunking heuristics. For a one-off analysis — "read this 200-page contract and list the termination clauses" — long context is strictly better than building a RAG pipeline that will be used once. It also handles global questions that retrieval struggles with: "summarize the themes across all of these tickets" requires seeing all of them, not the top 5.

**Concrete weakness:** Two things. First, cost and latency scale with input tokens on every call. A 100k-token prompt sent on every request is expensive in a way a 5k-token retrieved prompt is not, and the bill compounds with traffic. Second, and more subtly, models do not attend uniformly across a long context. The "lost in the middle" effect — information placed in the middle of a long prompt being recalled less reliably than information at the start or end — is a documented behavior of transformer attention, not a myth. If the critical fact is buried at token 60,000 of a 100k prompt, the answer may be worse than if just that passage had been retrieved.

**When it fails:** High-volume production traffic (cost), tasks with many documents where only a small relevant subset matters (attention dilution), and anything where the model must demonstrably have seen a specific passage (it can be proven, but you pay to send everything).

**Best for:** Low-volume, high-complexity analysis; prototyping before committing to an index; and cases where the corpus is small and stable enough that sending it whole is cheaper than maintaining retrieval infrastructure.

## The decision that actually matters: does the answer change?

The three techniques are usually presented as a menu. They are better understood as answers to three different questions, and the first question is not "which is best" but "how often does the ground truth change?"

| If the problem is... | Reach for... | Because... |
|---|---|---|
| Output format or tone is inconsistent | Fine-tuning | Behavior lives in weights, not prompt |
| Facts change weekly and need citations | Retrieval | Index updates without retraining |
| One-off analysis of a large document | Long context | No pipeline to build |
| High-volume Q&A over a big corpus | Retrieval | Per-call token cost stays low |
| Model must follow a strict schema at scale | Fine-tuning | Format compliance is what it does |
| "Summarize everything in these 500 tickets" | Long context | Global view beats top-k |
| Answer must cite a source passage | Retrieval | Weights can't cite |

A worked example makes the cost logic concrete. Assume a support assistant handles 100,000 requests per day and each request needs about 2,000 tokens of relevant policy context. With retrieval, roughly 2,000 context tokens go out per call. With long context, the whole 150,000-token policy corpus goes out per call. The ratio is 150,000 ÷ 2,000 = 75:1 in input tokens. Even at identical per-token pricing, retrieval costs one seventy-fifth as much on input for every request. That arithmetic — not a benchmark — is why long context is a prototyping tool and retrieval is a production tool for high-volume, document-grounded Q&A. (The 100,000 / 150,000 / 2,000 figures are illustrative; substitute your own.)

The second question is "can the model be wrong silently?" Fine-tuning and long context both tend to produce confident answers with no provenance. Retrieval at least gives a chunk to inspect. If the domain requires showing your work, that alone narrows the choice.

The third question is "how much labeled data is available?" Fine-tuning needs examples. With ten examples there is no fine-tuning dataset; there is a prompt. Retrieval needs documents, which most teams already have. Long context needs nothing but a big enough window and a budget.

A common pattern is that teams reach for fine-tuning too early and retrieval too late. Fine-tuning feels like the "real" ML move, so it attracts effort disproportionate to its fit. Retrieval feels like plumbing, so it gets deferred until the prompt is unmanageable. In practice, the order that works for most document-grounded problems is: start with long context to validate the task, move to retrieval when volume or corpus size makes long context expensive, and consider fine-tuning only when there is a stable behavioral problem that retrieval cannot fix.

## How to prioritise when two techniques could both work

When genuinely torn between retrieval and long context, run the smallest experiment that distinguishes them. Take 20 real user questions with known correct answers. Answer each twice: once with the full corpus in context, once with top-k retrieval. Compare accuracy *and* cost per correct answer. The technique that wins on cost per correct answer at expected volume is the one to build. This is a two-hour experiment, not a two-week one, and it produces a number that can be defended.

To make that comparison concrete, log four things per call: input tokens sent, output tokens returned, whether the answer matched the known-correct answer (human or rubric graded), and wall-clock latency. Cost per correct answer is then (sum of input and output token cost across all calls) ÷ (number of correct answers). A technique that is 10% more accurate but sends 75× the input tokens loses badly once volume is real. Instrument the token counts from the API response's usage field rather than estimating from character counts, since tokenizers differ.

When torn between fine-tuning and prompting, the test is different: prompt the base model with 10 to 20 carefully written examples of the desired behavior (few-shot). If accuracy is close to what is needed, fine-tuning isn't required — better examples are. Fine-tuning earns its cost only when few-shot prompting plateaus below the requirement and there are hundreds of examples to push past it. Measure the plateau by evaluating few-shot accuracy at 5, 10, and 20 examples; if the curve is flat, more examples in the prompt won't help.

A prioritisation checklist, in order:

1. Is the problem *behavior* (format, tone, classification) or *knowledge* (facts, documents)? Behavior points to fine-tuning; knowledge points to retrieval or long context.
2. Does the knowledge change faster than you can retrain? If yes, fine-tuning is out.
3. Do you need citations or per-user access control? If yes, retrieval is the only one that gives it cleanly.
4. Is volume high enough that per-call input tokens matter? If yes, long context is out for production.
5. Do you have hundreds of labeled examples? If no, fine-tuning is out for now.

## Failure modes worth designing against

Each technique has a silent-failure profile, and the design work is mostly about making failures loud.

**Fine-tuning: the frozen fact.** A model fine-tuned on a product catalog answers confidently about a discontinued product months after removal. Mitigation: keep facts out of training data entirely; use fine-tuning only for behavior, and route all factual answers through retrieval or a tool call. Add an evaluation set that includes questions whose answers changed recently; if the fine-tuned model regresses on those, facts have leaked into the weights.

**Retrieval: the confident wrong citation.** The retriever returns a real chunk that does not actually answer the question, and the model paraphrases it into an answer. Mitigation: a similarity threshold plus an explicit refusal instruction, and an evaluation set of questions deliberately absent from the corpus to confirm the system says "I don't know." Chunk with overlap and keep section headers attached to each chunk so the "except for digital goods" clause travels with the rule it modifies.

**Long context: the buried fact.** The relevant passage sits in the middle of a large prompt and gets under-weighted. Mitigation: place the most important material at the start or end of the prompt, and test recall by moving the same fact to different positions in the prompt and comparing answer accuracy. If accuracy drops when the fact moves to the middle, positional placement is a real variable in the design.

## Frequently Asked Questions

**Can fine-tuning and retrieval be combined?**

Yes, and it is often the right answer. Fine-tune the model for output format and task adherence, then feed it retrieved context at inference time. The two are complementary: fine-tuning handles *how* to answer, retrieval handles *what* to answer from. The mistake is expecting either one alone to cover both.

**Why does a RAG system hallucinate when the answer isn't in the documents?**

Because nothing in a standard RAG loop tells the model it is allowed to refuse. The retriever always returns top-k results, and the prompt usually says "answer the question" rather than "answer only if the sources support it." Add an explicit refusal instruction and a similarity threshold, and test with questions whose answers are deliberately absent from the corpus.

**Is long context going to replace retrieval?**

Not for high-volume production systems. The per-call token cost of sending an entire corpus on every request scales with traffic in a way retrieval does not, and attention dilution means a long prompt is not equivalent to a focused one. Long context is replacing retrieval for low-volume, one-off, or global-summary tasks, which is a real shift, but the cost arithmetic still favors retrieval at scale.

**How many examples are needed to fine-tune?**

There is no universal number, and anyone who gives one is guessing. The honest answer is: enough that the model sees the range of inputs and outputs it will encounter, which for a narrow format task can be a few hundred and for a broad behavioral change can be thousands. Start with few-shot prompting; if that plateaus, fine-tune with a few hundred high-quality examples and measure before scaling up.

## The 30-minute next step

Write down the 10 questions the system must answer correctly and, next to each, mark whether its correct answer is a *behavior* (a format or classification) or a *fact* that could change. If most are facts, open the retrieval code and find the line where the prompt is built — check whether it has a refusal path and a similarity threshold. If it does not, add them today; that single change fixes more hallucination than any amount of fine-tuning.
