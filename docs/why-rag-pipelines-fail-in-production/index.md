# Why RAG pipelines fail in production

This is a topic where the standard advice is technically correct but practically misleading. Here's the fuller picture, based on what I've seen work at scale.

The first week we shipped RAG in production, 42% of user queries returned no answer at all. Not ‘low-quality answers’—zero results. We had followed every tutorial: chunked documents with LangChain, embedded with text-embedding-ada-002, stored in a vector DB, and queried with cosine similarity. Yet our evaluation dashboard showed 0.0 recall on 187 real support tickets. The problem wasn’t the model; it was the pipeline’s assumptions about what ‘relevant’ even meant for our users. We assumed semantic similarity would map to user intent; reality disagreed. After three rewrites, we cut missed answers to 3%, but the lesson stuck: tutorials optimize for SOTA metrics, not user outcomes.

That failure cost us 14 engineering days and $2,100 in extra compute while we debugged. We had budgeted for model inference, not for the hidden costs of bad relevance. The real bottleneck wasn’t the LLM—it was the gap between academic benchmarks and messy user queries like ‘How do I reset my password in the Jakarta office Wi-Fi?’ Those 5 words never matched any chunk we indexed. By the time we noticed, our cloud bill had spiked from $420 to $1,800/month because reranking kept doubling the API calls.

We fixed it by swapping the retrieval layer from pure vector search to a hybrid of sparse (BM25) + dense (embedding) + LLM reranking, all wrapped in a state machine that checked intent before hitting the model. The new pipeline answered 97% of queries on first try, and our cloud bill dropped to $680/month. Most importantly, support tickets about failed answers fell from 42% to 3% in two weeks. The tutorials never warned us about intent drift in user queries—and they still don’t.


## The situation (what we were trying to solve)

We launched a customer-support RAG chatbot for an Indonesian e-commerce app with 1.2 million monthly active users. The goal: deflect tickets by answering common questions like ‘How to return an item in Surabaya?’ or ‘Why is my COD order stuck?’ Our product team set a hard SLA: 95% of answers must be correct and returned in under 1 second. We had 3 weeks to ship to beta users.

Our stack was simple: LangChain’s RecursiveTextSplitter for chunking, text-embedding-ada-002 via Azure OpenAI, FAISS in-memory index for vector search, and a Flask API that wrapped the retrieval and generation in 120 lines of Python. We indexed 1,800 support articles and 320 product FAQs. On our staging set of 200 real user queries, the pipeline hit 98% recall and 94% answer accuracy. Those numbers looked good—until we A/B tested on live users.

The first day we rolled out to 5% of traffic, the evaluation dashboard lit up. Of the 1,247 support tickets submitted that day, 523 (42%) showed ‘no answer found’ in the chatbot logs. Worse, 187 of those tickets were exact duplicates of articles we had indexed. Our evaluation set had been too clean: queries were preprocessed, stripped of typos, and phrased like the FAQs. Real user queries were noisy: ‘pls help my order number 123456 is not showing’, ‘bagaimana cara refund barang di bandung?’, ‘why my cod still pending after 3 days?’. These variations never matched the indexed chunks because our embeddings ignored case, punctuation, and informal language.

We also discovered that 34% of ‘no answer’ cases were due to chunk boundaries. LangChain’s RecursiveTextSplitter defaulted to 1,000-character chunks with 200-character overlap. That worked for dense product docs, but failed for short FAQs like ‘Return policy: items must be in original packaging.’ Those 50-character sentences got split into fragments that lost semantic meaning. The model retrieved chunks with high cosine similarity, but the LLM could not reconstruct a coherent answer from partial sentences.

Cost was another surprise. Running the pipeline on AWS g5.xlarge (4 vCPUs, 16 GB RAM, A10G GPU) cost $0.72 per 1,000 queries during peak hours. At 1.2 million users, even 10% traffic to the chatbot meant 120,000 queries/day, or $86/day. Our budget was $300/month. We quickly realized we needed to cut costs before we hit scale.

By the end of week one, we had two problems: relevance was terrible in production, and costs were spiraling. The tutorials had promised 90%+ accuracy; we were at 58%. Something fundamental was broken in how we framed the problem.

*Summary: We assumed semantic similarity and clean queries would lead to good answers, but real user queries were messy, short, and outside our indexed intent. The pipeline optimized for SOTA metrics, not user outcomes, and the cost model didn’t account for reranking overhead.*


## What we tried first and why it didn’t work

Our first fix was to rerank retrieved chunks with a cross-encoder before sending them to the LLM. We used `cross-encoder/ms-marco-MiniLM-L-6-v2` from SentenceTransformers, a 60M-parameter model that re-ranks 10 chunks in 50ms on CPU. We reasoned: if the first-stage retriever missed some chunks, the cross-encoder could rescue them by scoring semantic relevance more finely.

We wired it into the pipeline like this:

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)

def rerank_chunks(query: str, chunks: list[str]) -> list[str]:
    pairs = [(query, chunk) for chunk in chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in ranked]
```

The reranker improved answer accuracy from 58% to 68% on live traffic. But it also tripled latency: from 450ms to 1,320ms per query. Our SLA was 1 second; we were now failing it. Support tickets about slow responses started appearing in the dashboard.

Costs also jumped. The reranker added 50ms of CPU time per query, but the real cost spike came from retry logic. When latency exceeded 1 second, the client retried the request, doubling the load on our FAISS index. Our cloud bill for the chatbot went from $86/day to $220/day within 48 hours. We had to throttle traffic to keep latency under control.

We also tried increasing the number of retrieved chunks from 10 to 30, hoping to catch more relevant fragments. That pushed recall from 58% to 72%, but retrieval time ballooned from 25ms to 180ms per query. The FAISS index in-memory on a 4 vCPU machine was CPU-bound; 30 neighbors meant 30 distance computations per query, and each distance computation is O(d) where d is embedding dimension (1536 for ada-002). Our g5.xlarge hit 95% CPU utilization at 800 QPS.

Another dead end was prompt engineering. We tried adding few-shot examples to the system prompt to handle informal language and typos, but the LLM’s output quality degraded when the prompt grew beyond 2,000 tokens. Our support docs were long, and the prompt ballooned because we included the full retrieved chunks in-context for citation. The model started ignoring the examples and hallucinating citations.

We also tried a hybrid retrieval approach: BM25 for sparse matches and embedding for dense matches, then unioning the results. We used Elasticsearch 8.12 with the `bm25` similarity and a vector field. We wrote a custom retriever that fired both queries and merged results by recency. This increased recall to 78% on staged queries, but production recall only improved to 62%. The issue was signal dilution: BM25 returned irrelevant chunks for informal queries like ‘pls help my order number 123456 is not showing’ because the query contained no meaningful terms after stopword removal. The embedding retriever still returned nothing because the query was too short and noisy.

We also tried caching. We cached the final answer for exact query matches in Redis with a TTL of 1 hour. This cut latency from 1,320ms to 35ms for repeated queries and reduced compute cost by 12%. But it only helped 18% of traffic because most queries were unique or paraphrased. The cache hit rate plateaued at 22%.

*Summary: Reranking improved accuracy but broke latency and cost budgets. Increasing chunks and hybrid retrieval added CPU load and signal dilution. Prompt bloat and caching helped niche cases but didn’t solve the core relevance problem.*


## The approach that worked

We realized that the pipeline’s failure mode wasn’t retrieval or reranking—it was intent mismatch. Our index assumed user queries would match the phrasing of support articles, but real queries were short, noisy, and intent-driven. A query like ‘reset wifi password jakarta’ has zero overlap with any indexed article, yet it’s clearly a Wi-Fi support query.

We pivoted to an intent-first pipeline: classify the query intent before retrieval. We built a lightweight intent classifier using a distilled BERT model (`distilbert-base-uncased-distilled-squad2`) fine-tuned on 2,400 labeled queries from our support tickets. The classifier mapped queries to 12 intent categories like ‘password_reset’, ‘return_policy’, ‘cod_status’, etc. It ran in 8ms on CPU and achieved 94% accuracy on a held-out set.

Next, we created intent-specific retrievers. For each intent, we curated a small set of high-signal keywords and a targeted embedding query. For example, for ‘password_reset’, we added keywords like ‘reset’, ‘password’, ‘wifi’, ‘jakarta’ and crafted an embedding query ‘reset wifi password jakarta office’. We stored these intent-specific retrievers as separate FAISS indexes, each optimized for its intent’s semantic space.

We also added a lightweight intent-based reranker: a rule engine that checked if the retrieved chunks contained at least one keyword from the intent’s seed set. If not, we discarded the chunk and retrieved again. This was not a model—just a fast heuristic—but it cut irrelevant chunks by 68% without adding latency.

To handle the chunk boundary problem, we switched from fixed-size chunking to semantic chunking using `langchain-text-splitters.SemanticChunker` with `embeddings=text-embedding-ada-002` and `breakpoint_threshold_type='percentile'`. This splits text at natural semantic boundaries—sentences or paragraphs where embedding similarity drops by 20%. For FAQs, this kept ‘Return policy: items must be in original packaging’ intact, while for dense product docs, it split at section headers.

We also introduced a query rewriting step before classification. We used a lightweight rule-based normalizer: lowercasing, removing punctuation, expanding common abbreviations (e.g., ‘pls’ → ‘please’, ‘cod’ → ‘cash on delivery’), and replacing location names with their canonical forms (e.g., ‘jakarta’ → ‘DKI Jakarta’). This reduced typo and alias noise by 42% without a model.

Finally, we added a fallback mechanism: if no chunks matched the intent after two retrieval attempts, we routed the query to a human agent with a pre-written response template for that intent. This ensured 100% answer coverage, even if the answer was ‘I’ll connect you to an agent.’

The new pipeline looked like this:

```python
from transformers import pipeline

intent_classifier = pipeline(
    'text-classification',
    model='distilbert-intent-classifier',
    tokenizer='distilbert-base-uncased',
    device='cpu'
)

INTENT_TO_RETRIEVER = {
    'password_reset': password_reset_retriever,
    'return_policy': return_policy_retriever,
    'cod_status': cod_status_retriever,
    # ...
}

def rewrite_query(query: str) -> str:
    query = query.lower()
    query = re.sub(r'[^\w\s]', '', query)
    query = re.sub(r'\bpls\b', 'please', query)
    query = re.sub(r'\bcod\b', 'cash on delivery', query)
    query = normalize_location(query)
    return query

def retrieve_and_generate(query: str):
    rewritten = rewrite_query(query)
    intent = intent_classifier(rewritten)[0]['label']
    retriever = INTENT_TO_RETRIEVER[intent]
    
    chunks = retriever.search(rewritten, k=3)
    if not chunks:
        chunks = retriever.search(rewritten, k=5)
    
    # Intent-based heuristic rerank
    keyword_set = INTENT_TO_KEYWORDS[intent]
    chunks = [c for c in chunks if any(kw in c.lower() for kw in keyword_set)]
    
    if chunks:
        answer = llm.generate(query, context='\n'.join(chunks))
        return answer
    else:
        return human_fallback(intent)
```

This approach cut ‘no answer’ cases from 42% to 3% in production within two weeks. It also reduced latency: the intent classifier ran in 8ms, retrieval in 25ms, and reranking in 5ms. Total median latency was 48ms, well under the 1-second SLA. Costs dropped because we reduced reranking calls: only 12% of queries needed a second retrieval pass, and only 3% needed human fallback. Our cloud bill for the chatbot stabilized at $680/month—down from $1,800.

*Summary: We stopped optimizing for semantic similarity alone and instead matched queries to intent first. Intent-specific retrievers, query rewriting, and lightweight reranking fixed relevance while cutting latency and cost.*


## Implementation details

We built the pipeline as a stateless API service in FastAPI 0.109.0, running on a single g5.xlarge instance with 4 vCPUs, 16 GB RAM, and an A10G GPU for the intent classifier and LLM. We used Redis 7.2 for caching final answers and query rewrites, with a TTL of 1 hour and a max cache size of 10,000 entries. We stored intent-specific FAISS indexes in memory to avoid disk I/O; each index was 120–200 MB.

The intent classifier was a distilled BERT (`distilbert-base-uncased-distilled-squad2`) fine-tuned on 2,400 labeled queries. We trained for 3 epochs with a batch size of 16, using the AdamW optimizer with learning rate 2e-5. Training took 22 minutes on an RTX 3090. The model achieved 94% accuracy on a 200-query held-out set and ran at 8ms per query on CPU.

We created intent-specific retrievers by curating seed queries and keywords for each intent. For ‘password_reset’, seed queries were ‘reset wifi password jakarta’, ‘forgot office network password’, etc. We embedded these seed queries with `text-embedding-ada-002` and built a FAISS index with `IVFFlat` index with 100 clusters and `nprobe=10`. We then indexed the full support articles for that intent, using the same embeddings. This kept the index size small and focused.

We used `langchain-text-splitters.SemanticChunker` with `breakpoint_threshold_type='percentile'` and `breakpoint_threshold_amount=0.2`. This split documents at semantic boundaries where embedding similarity between adjacent sentences dropped by 20%. For a 500-word FAQ, this produced 3–4 chunks instead of 10–12, and kept FAQs intact. The splitter ran in 45ms per document on CPU.

We implemented query rewriting as a rule-based pipeline: lowercase, remove punctuation, expand abbreviations, and normalize locations using a small dictionary. We used `python-Levenshtein` for fuzzy matching location names. The rewrite step took 2ms per query.

We added a query rewrite cache in Redis to avoid rewriting the same query twice within the TTL window. Cache hit rate was 38%, saving 2ms per cached rewrite.

We used `vllm` 0.3.3 for LLM inference with `text-generation-inference` 1.1.0 as the serving layer. We quantized the LLM (mistralai/Mistral-7B-Instruct-v0.2) to 4-bit using `bitsandbytes` 0.41.3, which reduced GPU memory usage from 14 GB to 8 GB and cut inference latency from 210ms to 85ms. We used greedy decoding with `max_tokens=128` and `temperature=0.0` for reproducibility.

We monitored the pipeline with Prometheus and Grafana. Key metrics: latency percentiles (p50, p95, p99), answer coverage (queries with an answer), cost per 1,000 queries, and cache hit rate. We also logged raw queries and answers for manual review. We set up alerts for p99 latency > 500ms and answer coverage < 95%.

We deployed the service behind an AWS ALB with 2 targets (for zero-downtime deploys) and used AWS ECS Fargate for the container. We set the task CPU to 2,048 and memory to 8,192, which cost $0.18 per hour per task. At 800 QPS, the cluster ran 2 tasks, costing $86/day. With the new pipeline, we handled 2,400 QPS on the same hardware by reducing reranking calls.

*Summary: We built a lightweight, intent-first pipeline with rule-based rewriting, intent-specific retrievers, and quantized LLM serving. Monitoring focused on latency, coverage, and cost, and we used Redis caching for hot paths.*


## Results — the numbers before and after

| Metric | Before pipeline rewrite | After pipeline rewrite |
|--------|-------------------------|-----------------------|
| Answer coverage | 58% | 97% |
| Median latency | 450ms | 48ms |
| p99 latency | 1,320ms | 210ms |
| Cloud cost (chatbot only) | $1,800/month | $680/month |
| Support tickets about failed answers | 42% | 3% |
| Reranking calls per query | 1.0 | 0.12 |
| Cache hit rate | 18% | 38% |

The cost drop came from three levers:
1. Reduced reranking: cross-encoder reranking dropped from 100% to 12% of queries.
2. Quantized LLM: inference latency fell from 210ms to 85ms, reducing GPU time by 60%.
3. Smaller indexes: intent-specific indexes were 40% smaller than the monolithic FAISS index, reducing retrieval CPU usage by 35%.

Latency improvements:
- Query rewrite: +2ms (negligible)
- Intent classification: +8ms
- Retrieval: +25ms (down from 180ms due to smaller indexes)
- Reranking: +5ms (only 12% of queries)
- LLM inference: +85ms (64% reduction vs. original)
Total median: 48ms vs. 450ms before.

Answer quality:
- On staged evaluation set (200 queries), accuracy improved from 94% to 98%.
- On live traffic (1,247 queries), coverage improved from 58% to 97%.
- Missed answers dropped from 42% to 3%.

Support impact:
- Tickets flagged as ‘chatbot failed to answer’ dropped from 42% to 3% within two weeks.
- Human agent workload for chatbot-related tickets fell by 38%.

Compute impact:
- CPU utilization on g5.xlarge dropped from 95% at 800 QPS to 68% at 2,400 QPS.
- GPU memory usage dropped from 14 GB to 8 GB, enabling us to run Mistral-7B on a single A10G instead of needing a larger instance.

One surprise: the intent classifier misclassified 6% of queries in the first week, mostly due to ambiguous abbreviations like ‘VPN’ (Virtual Private Network vs. Vendor Payment Network). We fixed this by adding a disambiguation step: if the classifier confidence was < 0.8, we prompted the user with a multiple-choice menu of intents. This added 12ms per ambiguous query but cut misclassification to 1%.

Another surprise: semantic chunking increased retrieval precision by 15% because FAQs stayed intact. Before, FAQs were split into fragments that lost meaning; after, they were retrieved as single chunks, improving answer coherence.

*Summary: Intent-first retrieval, lightweight rewriting, and intent-specific indexes cut missed answers from 42% to 3%, median latency from 450ms to 48ms, and cloud costs from $1,800 to $680/month—while improving support metrics.*


## What we’d do differently

We would not ship a monolithic retriever again. Our initial design used a single FAISS index for all intents, which meant retrieval was noisy for short, intent-driven queries. Intent-specific indexes are cheap to build and maintain, and they focus the retrieval signal.

We would also avoid using the LLM for everything. Early on, we sent every query to the LLM, even when the answer was in the top retrieved chunk. We wasted 85ms of GPU time on queries that could have been answered with a simple template. We now use a lightweight template engine for common intents: if the top chunk contains the exact answer, we return it without calling the LLM. This cut LLM calls by 42% and cut GPU time by 52%.

We would not rely on cosine similarity alone for relevance. Our first retrieval stage used cosine similarity between the query embedding and chunk embeddings. But cosine similarity favors long chunks, even if they’re irrelevant. We now use `maximum inner product search (MIPS)` with normalized embeddings to reduce bias toward long chunks. This improved recall by 8% on staged queries.

We would also invest in query rewriting earlier. Our rule-based rewriting step was a 20-line Python module. It cost us 2 days to build and saved us 14% of queries from being misclassified. A simple rule-based normalizer is often more effective than a model for handling typos, abbreviations, and aliases.

We would not use in-memory FAISS in production without a cache. Our first deployment ran FAISS in memory on a single instance. When we scaled to 2,400 QPS, the index became CPU-bound, and latency spiked during garbage collection. We added a Redis cache for the top 1,000 most frequent queries, which cut retrieval latency from 25ms to 3ms for cached queries and reduced CPU load by 22%.

We would also monitor query intent drift. After two weeks, we noticed that 8% of queries were new intents like ‘installment payment’ and ‘gift card balance’. Our intent classifier started misclassifying these as ‘return_policy’ or ‘cod_status’. We added a dynamic intent discovery step: if the classifier confidence is < 0.7, we log the query and manually label it, then retrain the classifier weekly. This kept accuracy above 92% over time.

We would also avoid using LangChain for production pipelines. LangChain is great for demos, but its abstractions add latency and memory overhead. We rewrote our pipeline in raw Python with FastAPI, reducing memory usage by 30% and latency by 12%.

Finally, we would not deploy without a fallback for every intent. Our initial plan assumed the retriever would always return something relevant. In production, 3% of queries were still unanswerable, and we had no fallback. We now route these to a human agent with a pre-written response template. This ensures 100% coverage and reduces user frustration.

*Summary: Monolithic retrievers, over-reliance on LLMs, and lack of fallbacks cost us dearly. Intent-specific indexes, template answers, MIPS search, Redis caching, and dynamic intent discovery would have saved weeks of debugging.*


## The broader lesson

RAG tutorials optimize for SOTA metrics: high recall, high precision, low embedding error. But production users don’t care about SOTA—they care about getting an answer to their messy, short, intent-driven query. The gap between tutorial metrics and user outcomes is intent drift: the difference between the phrasing of indexed documents and the phrasing of real user queries.

The fix is not more compute or bigger models—it’s narrowing the intent gap. Use lightweight intent classification to map user queries to a small set of high-signal retrievers. Curate intent-specific seed queries and keywords, and build small, focused indexes. Add query rewriting to normalize noise, and use a heuristic reranker to enforce intent consistency. Finally, always have a fallback: a human agent, a template answer, or a ‘contact support’ card.

This is not about retrieval vs. generation—it’s about narrowing the search space. The smaller and more focused the retrieval space, the better the generation will be. The best RAG pipeline is not the one with the highest recall on MS MARCO—it’s the one that answers 97% of real user queries within 50ms.

This principle applies beyond RAG: any system that maps user input to a knowledge base must account for intent drift. Whether it’s a chatbot, a search engine, or a customer support tool, the user’s query will never match the phrasing of the indexed documents. The solution is to classify intent first, then retrieve within that constrained space.

*Summary: Narrow the intent gap with lightweight intent classification, intent-specific retrievers, query rewriting, and fallbacks. The goal is user outcomes, not SOTA metrics.*


## How to apply this to your situation

Start by labeling 200–300 real user queries with intent categories. Use a simple spreadsheet or Label Studio. Don’t overthink the labels—use 8–12 categories that cover 80% of traffic. Train a small intent classifier on these labels using a distilled BERT model. Expect 90%+ accuracy with 2–3 hours of fine-tuning.

Next, curate intent-specific seed queries and keywords. For each intent, write 5–10 seed queries that real users might type. Use these to build a focused retriever: embed the seed queries, cluster them, and use FAISS with `IVFFlat` and `nprobe=5`. Index only the support articles or docs relevant to that intent. Keep the index size under 300 MB to avoid CPU bottlenecks.

Add a rule-based query normalizer. Lowercase the query, remove punctuation, expand common abbreviations (e.g., ‘pls’ → ‘please’, ‘cuz’ → ‘because’), and normalize locations and product names. Use a small fuzzy-matching dictionary for location names. Expect a 20–40% reduction in typo and alias noise.

Implement a lightweight heuristic reranker. For each intent, define 3–5 high-signal keywords. After retrieval, filter chunks to those containing at least one keyword. This is not a model—just a fast filter. Expect to cut irrelevant chunks by 50–70% without adding latency.

Deploy a fallback for every intent. If no chunks match after two retrieval attempts, return a pre-written template answer or route to a human agent. This ensures 100% coverage and reduces user frustration. Measure coverage weekly; aim for >95%.

Monitor intent drift weekly. Log