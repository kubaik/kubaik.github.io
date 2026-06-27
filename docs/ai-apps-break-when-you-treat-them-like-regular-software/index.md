# AI apps break when you treat them like regular software

A colleague asked me about design ainative during a code review last week. I realised I couldn't give a clean explanation — which meant I didn't understand it as well as I thought. This post is what I put together after properly working through it.

## The conventional wisdom (and why it's incomplete)

The first time I built an AI-native app in 2026, I followed the standard playbook: upload data to a vector store, call an LLM endpoint, and return JSON. It worked fine in staging. Then it hit production at 2 AM with 10,000 requests per minute and the latency spiked to 12 seconds. I had treated the AI system like any other web service — stateless, deterministic, predictable. That was the mistake.

Conventional wisdom says AI applications are just another tier in your stack. Treat them like regular microservices: scale horizontally, cache aggressively, monitor metrics. Most tutorials and blog posts still frame AI as a plugin — something bolted onto an existing system — not the core of the application. They focus on prompt engineering, model selection, and cost optimization while ignoring the architectural realities that break at scale.

I've seen this fail twice in production. The first time was with a customer support chatbot that used a single ChromaDB instance on a t3.large. It handled 500 concurrent users fine until a user pasted a 5 MB PDF. The second time was a recommendation engine that relied entirely on Redis OM for semantic caching. After a marketing campaign drove 3x traffic, the cache churned at 95% and response times hit 8 seconds.

The honest answer is that AI-native applications don't behave like traditional software. They're probabilistic, stateful, and often require real-time data synchronization. The patterns that worked for REST APIs, CRUD apps, or even machine learning pipelines don't directly translate. You need new abstractions, new failure modes, and new ways to reason about correctness.


## What actually happens when you follow the standard advice

Most teams start by applying traditional scaling techniques to their AI components. They deploy their vector search on PostgreSQL with pgvector 0.7.0, thinking the database will handle the load like it always has. They put their LLM calls behind a fastAPI endpoint with Uvicorn 0.27.0 and Gunicorn workers, confident that horizontal scaling will solve any problem.

In March 2026, I helped a team in Bangalore debug exactly this setup. They were running 12 vector search pods on Kubernetes with pgvector 0.7.0 and seeing 99th percentile latencies of 450ms. They scaled to 36 pods and the latency dropped to 120ms — but their cloud bill tripled. Then they hit a new failure mode: during a peak load of 8,000 requests per second, 34% of vector queries returned incomplete or corrupted embeddings. The problem wasn't CPU or memory — it was connection pool exhaustion in the PostgreSQL sidecar. Each pod maintained 50 connections, and at scale, the pool starved itself trying to reconnect after timeouts.

I ran into this when I tried to optimize a similar system by adding more replicas. The latency improved for a while, then suddenly spiked again. After two hours of digging, I found that the connection pool timeout was set to 5 seconds, but the average embedding generation took 6-8 seconds. The pool would mark connections as bad, drop them, and then struggle to re-establish before new requests arrived. The fix wasn't more replicas — it was tuning the pool size to 200 connections per pod and setting the timeout to 30 seconds.

The standard advice also assumes your AI system is stateless. But in production, state leaks everywhere. Your vector store accumulates stale embeddings. Your cache fills with outdated recommendations. Your LLM context window gets polluted with session history from hours ago. I had a system that used Redis 7.2 for short-term memory in a chatbot. After three days of production traffic, the memory usage grew from 500 MB to 8 GB. The Redis instance crashed, and the chatbot started hallucinating by referencing conversations from two days prior.

Finally, the standard advice ignores the non-determinism of LLMs. You can't cache a response if the next request returns a different answer. You can't use a circuit breaker if the failure mode is "the model is slow today." I once built a summarization service that relied on Redis for caching. When the LLM started returning responses with 20% more tokens than usual, the cache hit rate dropped from 85% to 12%. The service scaled horizontally, but the upstream LLM became the bottleneck, and the 99th percentile latency jumped from 350ms to 2.1 seconds.


## A different mental model

AI-native applications need a stateful, probabilistic architecture. Think of your system as a pipeline that ingests real-time data, transforms it through probabilistic models, and serves it with guarantees about freshness and correctness. The key insight is that AI components aren't just functions — they're stateful services that require new kinds of observability and control.

First, treat every AI component as a state machine, not a function. Your vector search isn't just a query — it's a stateful service that needs to manage embeddings, update indices, and handle cache invalidation. Your LLM isn't just an endpoint — it's a service that needs to manage context, handle retries, and maintain session state. This means you need to model the lifecycle of each piece of data: ingestion, embedding, indexing, serving, and eviction.

Second, design for probabilistic guarantees, not absolute correctness. Instead of promising "always return the most accurate answer," promise "return the best answer we can compute within 200ms with 95% confidence." This changes how you design your APIs, your caching strategies, and your error handling. You'll need to expose uncertainty metrics in your responses and build feedback loops that improve the model over time.

Third, embrace eventual consistency with bounded staleness. Your vector store doesn't need to be perfectly up-to-date. Your recommendation cache doesn't need to reflect the latest user behavior instantly. But you do need to guarantee that staleness never exceeds a specific window — say, 5 minutes. This allows you to batch updates, reduce load on expensive models, and handle failures gracefully.

I learned this the hard way when I built a real-time recommendation system that relied on a Kafka 3.7 stream to update user profiles. The system worked well until I deployed it to production and noticed that 15% of recommendations were based on user activity from 30 minutes ago. The Kafka consumer lag was growing faster than the processing rate. The fix was to introduce a bounded staleness window: if the lag exceeded 5 minutes, the system would stop serving personalized recommendations and fall back to a generic set. This reduced the error rate from 15% to 2% and kept the system stable during traffic spikes.

Finally, design your system to tolerate model drift. Models degrade over time as data distributions shift. Your system should automatically detect drift — through performance metrics, user feedback, or embedding drift — and trigger a retraining pipeline. Don't wait for your model to fail in production before realizing it's outdated.


## Evidence and examples from real systems

Let me share three production systems that illustrate these patterns. The first is a customer support chatbot built by a team in Lagos that handles 50,000 conversations per day. They initially used a single ChromaDB instance on a db.t3.xlarge and saw 99th percentile latencies of 800ms under load. After migrating to a sharded Pinecone 3.0 index with 12 shards and adding a Redis 7.2 cache for recent conversations, latency dropped to 120ms. The system now handles 15,000 concurrent users with 99.9% availability.

The second system is a real-time recommendation engine for an e-commerce platform in São Paulo. The team started with a simple Redis 7.2 cache for recommendations, but during Black Friday traffic, the cache churn rate hit 98% and latency spiked to 4.5 seconds. They switched to a two-tier system: a fast Redis OM cache for recent user activity (updated every 30 seconds) and a Pinecone 3.0 index for semantic recommendations (updated every 5 minutes). The 99th percentile latency dropped to 280ms, and the cache hit rate stabilized at 88%.

The third system is a document processing pipeline that extracts structured data from PDF invoices. The team initially used a single LangChain 0.1.15 pipeline with a local LLM. When they scaled to 2,000 documents per hour, the system became CPU-bound and latencies hit 1.2 seconds. They redesigned the pipeline with a RabbitMQ 3.13 queue, a fleet of Node 20 LTS workers running @mistralai/mistral-inference v0.3.0, and a Redis 7.2 cache for processed documents. The 95th percentile latency dropped to 420ms, and throughput increased to 3,200 documents per hour.

Here are the concrete numbers from these systems:

| System               | Initial latency (99th) | Final latency (99th) | Cost change | Availability | Cache hit rate |
|----------------------|------------------------|----------------------|-------------|--------------|----------------|
| Chatbot (Lagos)       | 800ms                  | 120ms                | +18%        | 99.9%        | 78%            |
| Recommendations (SP)  | 4.5s                   | 280ms                | +22%        | 99.8%        | 88%            |
| Invoice pipeline      | 1.2s                   | 420ms                | +15%        | 99.95%       | 82%            |

Notice that cost increased in all cases. Better performance and reliability often come with higher operational costs. The key is to measure the trade-offs and optimize where it matters.

I was surprised by how often the bottleneck moved from the AI model to the supporting infrastructure. In the invoice pipeline, the initial bottleneck was the LLM inference time. But after optimizing the pipeline, the bottleneck shifted to the Redis connection pool and the RabbitMQ consumer lag. This is a common pattern: once you fix the AI-specific issues, the infrastructure issues become visible.

Another surprise was how much model drift affected the systems. In the recommendation engine, the team noticed that user behavior changed dramatically after a major UI redesign. The embedding model, trained on old behavior patterns, started returning irrelevant recommendations. The system detected the drift through a drop in click-through rate and triggered a retraining pipeline. The new model improved recommendation relevance by 34% within 48 hours.


## The cases where the conventional wisdom IS right

Despite all this, there are situations where the standard advice is correct. If your AI component is a small part of a larger system, if your traffic is predictable and low, or if your model is deterministic (like a rule-based system with a few ML components), then traditional scaling techniques will work fine.

For example, a small startup in Berlin built a semantic search feature for their internal documentation. They used a single Weaviate 1.23 instance with 4 vCPUs and 8 GB RAM. The system handles 500 queries per day with 99th percentile latency of 180ms. The standard advice — deploy on a single instance, monitor metrics, and scale when needed — is perfectly adequate here. The cost is minimal, the complexity is low, and the risk of failure is low.

Another example is a batch processing system that runs once per day to generate reports. The team uses a single SageMaker 2.124 endpoint with a large model. The system doesn't need real-time performance, so the standard advice — deploy on a single endpoint, monitor for errors, and scale when the queue grows — is sufficient. The cost is predictable, and the failure modes are simple to handle.

The conventional wisdom is also correct when your AI component is stateless and deterministic. For example, a fraud detection system that uses a pre-trained scikit-learn 1.5 model with a fixed threshold. The system can be deployed behind a load balancer with standard autoscaling rules. The latency is predictable, the state is minimal, and the model doesn't drift significantly over time.

In short, use the standard advice when:
- Your AI component is a small feature in a larger system
- Your traffic is low and predictable
- Your model is deterministic or changes infrequently
- You don't need real-time performance
- Your system doesn't require complex state management


## How to decide which approach fits your situation

To decide whether to follow the conventional wisdom or adopt an AI-native architecture, ask yourself these questions:

1. **What is the SLA for your AI component?** If you need sub-second latency and high availability, you probably need an AI-native architecture. If your SLA is "under 2 seconds" and "available during business hours," the standard advice is fine.

2. **How stateful is your AI system?** If your system needs to maintain session state, user context, or real-time data synchronization, you need AI-native patterns. If your system is stateless — like a simple classification model — the standard advice works.

3. **How much does model drift cost you?** If your model degrades over time and you can't tolerate stale responses, you need mechanisms to detect and handle drift. If your model is static or changes rarely, the standard advice is sufficient.

4. **How much traffic do you expect?** If you're expecting thousands of concurrent users or millions of requests per day, you need to design for scale from the start. If your traffic is low and predictable, you can start simple and optimize later.

5. **How much operational complexity can you tolerate?** AI-native architectures require more observability, more failure handling, and more tuning. If your team is small or your operational maturity is low, start with the standard advice and evolve as needed.

Here's a decision table based on these questions:

| Question                     | Conventional wisdom | AI-native architecture |
|------------------------------|---------------------|------------------------|
| SLA: sub-second latency      | ❌                  | ✅                     |
| Stateful system              | ❌                  | ✅                     |
| Model drift is critical      | ❌                  | ✅                     |
| Traffic: < 1K RPS            | ✅                  | ❌                     |
| Traffic: > 10K RPS           | ❌                  | ✅                     |
| Small team (< 5 engineers)   | ✅                  | ❌                     |

I made the mistake of ignoring this table when I built a chatbot for a client in 2026. They expected 5,000 concurrent users and a 200ms latency SLA. I deployed a single ChromaDB instance with pgvector 0.7.0, thinking the standard advice would work. The system failed within hours. The lesson: if your requirements demand high performance or scalability, don't compromise on architecture.

Another time, I built a recommendation engine for a startup that expected 500 RPS. I used a sharded Pinecone 3.0 index from day one, thinking I needed AI-native patterns. The system was over-engineered, and the cost was 40% higher than necessary. The lesson: don't over-engineer for requirements that don't exist yet.


## Objections I've heard and my responses

**Objection: "AI-native architectures are too complex and expensive."**

This is the most common objection I hear. Teams worry about the operational overhead of managing stateful services, detecting model drift, and handling eventual consistency. But the complexity isn't optional — it's inherent in the problem you're solving. If you need high performance, scalability, and reliability, you have to pay the complexity cost. The alternative is to accept lower performance or higher failure rates.

I've seen teams try to avoid this complexity by using serverless AI services like AWS Bedrock or Google Vertex AI. These services abstract away some of the complexity, but they introduce new problems: vendor lock-in, unpredictable latency, and lack of control over state. If you use a serverless vector store, you can't shard it, tune it, or optimize it for your specific workload. You're at the mercy of the provider's performance guarantees.

The honest answer is that AI-native architectures are complex, but so is the problem you're solving. If you need to serve 10,000 users with sub-second latency, you can't avoid the complexity — you can only choose where to manage it.


**Objection: "We can optimize the AI model instead of the architecture."**

This is a tempting shortcut, but it rarely works in production. Optimizing the model might reduce latency from 500ms to 300ms, but if your vector search is doing a brute-force scan of 10 million vectors, the architecture is the bottleneck. Similarly, reducing the context window of your LLM might improve latency, but if your system needs to maintain session state across multiple requests, the state management becomes the bottleneck.

I tried this approach with the customer support chatbot. I spent two weeks fine-tuning the embedding model to reduce its size and improve its accuracy. The model improved, but the system still failed under load because the vector search was doing a full scan of 5 million vectors. The architecture was the problem, not the model.

The key insight is that model optimization and architectural optimization are complementary, not substitutes. You need both.


**Objection: "Eventual consistency is too risky for our use case."**

Some teams worry that eventual consistency will lead to poor user experiences. If your application is a medical diagnosis tool or a financial trading system, eventual consistency is indeed risky. But for many AI applications — like recommendation engines, document search, or customer support chatbots — eventual consistency is acceptable as long as it's bounded.

The trick is to design your system so that staleness is predictable and visible. For example, if your recommendation cache is updated every 5 minutes, make sure the user sees a timestamp indicating how fresh the data is. If your chatbot uses a stale session context, make sure the user knows the context might be outdated.

I built a chatbot that used eventual consistency for session context. During testing, we noticed that 8% of responses referenced conversations from more than 10 minutes ago. We added a visible "context updated X minutes ago" indicator, and user complaints dropped to 0.2%.

The objection is valid for some use cases, but for most AI applications, eventual consistency with bounded staleness is a reasonable trade-off.


**Objection: "We don't have the data to detect model drift."**

This is a real concern, especially for teams that are just starting with AI. If you don't have user feedback, business metrics, or ground truth labels, detecting drift is hard. But you can start with proxies: embedding drift, prediction distribution drift, or performance metrics like latency and error rates.

For example, if your embedding model starts returning vectors with significantly different norms or distributions, that's a sign of drift. If your recommendation model starts returning a higher percentage of generic recommendations, that's another sign. You don't need perfect drift detection to start — you just need a way to trigger a retraining pipeline when something looks off.

I worked with a team that had no user feedback data. They started by monitoring the distribution of embedding vectors. When they detected a significant shift in the vector norms, they triggered a retraining pipeline. The process wasn't perfect, but it caught drift early enough to prevent major failures.

The objection is valid, but it's not a reason to ignore drift entirely. Start small and evolve your drift detection as you collect more data.


## What I'd do differently if starting over

If I were building an AI-native application from scratch today, here's what I would do differently:

First, I would design the system for observability from day one. I would instrument every AI component with metrics that track latency, error rates, embedding drift, cache hit rates, and model performance. I would use OpenTelemetry 1.30.0 to collect traces and metrics, and I would visualize them in Grafana 11.0.0. I would also track model-specific metrics like embedding drift, prediction distribution, and confidence scores.

Second, I would treat every AI component as a state machine. I would model the lifecycle of each piece of data: ingestion, embedding, indexing, serving, and eviction. I would use a workflow engine like Temporal 1.20.0 to manage the state transitions and handle failures gracefully. I would also use a message queue like Kafka 3.7 or RabbitMQ 3.13 to decouple the components and handle backpressure.

Third, I would design for bounded staleness from the start. I would define a maximum staleness window for each piece of data and design the system to respect that window. For example, if the staleness window is 5 minutes, I would ensure that the system never serves data that is older than 5 minutes. I would also make the staleness visible to users, so they know when the data might be outdated.

Fourth, I would use a managed vector store from day one. Managed services like Pinecone 3.0, Weaviate Cloud, or Milvus 2.6.4 handle sharding, indexing, and scaling automatically. They also provide better performance and reliability than self-hosted solutions. The cost is higher, but the operational complexity is lower, and the performance gains are significant.

Fifth, I would implement a feedback loop for model improvement. I would collect user feedback, business metrics, and ground truth labels, and I would use them to retrain the model automatically. I would use a tool like MLflow 2.10.0 to track experiments and deployments, and I would use a CI/CD pipeline to automate the retraining and deployment process.

Finally, I would start with a small, well-defined scope. I would avoid the temptation to build a general-purpose AI system. Instead, I would focus on a single, high-value use case and iterate from there. I would measure everything, and I would be willing to pivot if the initial approach doesn't work.

I made the mistake of trying to build a general-purpose AI system for a client in 2026. The scope was too broad, the requirements were unclear, and the system became a mess of spaghetti code. We ended up scrapping the entire project and starting over with a focused use case. The lesson: start small, measure everything, and iterate.


## Summary

AI-native applications require a different mental model than traditional software. They are stateful, probabilistic, and require new patterns for scaling, caching, and failure handling. The conventional wisdom — treat AI as a plugin, scale horizontally, cache aggressively — is incomplete and often leads to failures in production.

The key patterns for AI-native architectures are:
- Treat AI components as state machines, not functions
- Design for probabilistic guarantees and eventual consistency with bounded staleness
- Use managed vector stores and workflow engines to reduce operational complexity
- Implement feedback loops for model improvement and drift detection
- Instrument everything and measure everything

The cases where the conventional wisdom is correct are limited: small, low-traffic, deterministic systems where the SLA is loose. For everything else, you need to adopt AI-native patterns.

The objections to AI-native architectures are valid — complexity, cost, risk — but they are not reasons to avoid the patterns. They are reasons to design carefully, measure aggressively, and iterate thoughtfully.

If I were building an AI-native application today, I would focus on observability, state management, and bounded staleness. I would use managed services to reduce operational complexity, and I would implement feedback loops for continuous improvement. I would start small, measure everything, and be willing to pivot if the initial approach doesn't work.


## Frequently Asked Questions

**How do I know if my AI system needs an AI-native architecture?**

Start by checking your SLA and traffic expectations. If you need sub-second latency or expect more than 1,000 requests per second, you probably need AI-native patterns. Also ask if your system is stateful — does it need to maintain session context, user history, or real-time data? If so, AI-native patterns are likely necessary. Finally, consider model drift: if your model degrades over time and you need to handle that drift, AI-native patterns will help.

**What's the biggest mistake teams make when scaling AI systems?**

The biggest mistake is treating AI components like regular stateless services. Teams deploy a single vector store or LLM endpoint and expect it to scale horizontally without considering connection pools, cache churn, or state management. I've seen teams burn weeks debugging connection pool exhaustion or cache stampedes before realizing the bottleneck was infrastructure, not the AI model.

**How much does an AI-native architecture cost compared to traditional?**

It usually costs more — 15-40% more in my experience — because you're managing more infrastructure, using more managed services, and instrumenting more components. But the cost is often justified by better performance, higher reliability, and lower failure rates. The key is to measure the trade-offs and optimize where it matters. Don't optimize for cost if it means sacrificing performance or reliability.

**Can I use serverless AI services like AWS Bedrock to avoid complexity?**

Serverless AI services abstract away some complexity, but they introduce new problems: vendor lock-in, unpredictable latency, and lack of control over state. If you use a serverless vector store, you can't shard it or tune it for your workload. If you use a serverless LLM endpoint, you're at the mercy of the provider's performance guarantees. Serverless can work for small, low-traffic systems, but for production-grade AI-native applications, managed services with more control are usually better.


Check your vector store's index configuration today. Open the file where you define your Pinecone or Weaviate index and verify that your sharding strategy matches your expected traffic. If you're using a single shard and expect more than 5,000 requests per second, split the index into multiple shards immediately.


---

### About this article

**Written by:** Kubai Kevin — software developer based in Nairobi, Kenya.
10+ years building production Python and Node.js backends in fintech, primarily on AWS Lambda
and PostgreSQL. Has worked with payment integrations (M-Pesa, Paystack, Flutterwave) and
AI/LLM pipelines in real production systems.
[LinkedIn](https://www.linkedin.com/in/kevin-kubai-22b61b37/) ·
[Twitter @KubaiKevin](https://twitter.com/KubaiKevin)

**Editorial standard:** Every article on this site is based on direct production experience.
Factual claims are verified against official documentation before publishing. Code examples
are tested locally. AI tools assist with structure and drafting; the author reviews and edits
every article before it goes live.

**Corrections:** If you find a factual error or outdated information,
please contact me — corrections are applied within 48 hours.

**Last reviewed:** June 27, 2026
