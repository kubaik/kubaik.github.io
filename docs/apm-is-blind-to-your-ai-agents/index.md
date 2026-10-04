# APM is blind to your AI agents

## The conventional wisdom (and why it's incomplete)

Application performance monitoring was built on a simple premise: a request comes in, work happens, a response goes out. Trace it, time it, alert if it's slow or errors. That model served two decades of CRUD apps well. But AI systems — LLM calls, agent loops, RAG pipelines — don't behave like CRUD apps. They are non-deterministic, they spawn sub-calls, they can silently degrade, and they often fail *semantically* rather than technically. A 200 OK response from an LLM endpoint tells you nothing about whether the answer was hallucinated, whether the agent got stuck in a loop, or whether the retrieval step returned zero relevant chunks.

The conventional wisdom says: instrument your HTTP calls, set up latency and error-rate alerts, and you're covered. That advice is incomplete because it assumes failure means an exception or a timeout. In AI systems, the most expensive failures are the ones that return a perfectly valid HTTP 200 with a plausible-but-wrong answer. The APM dashboard stays green while users get garbage. Traditional APM has no concept of semantic correctness, token economics, or agent state — and that gap is what this article addresses.

## What happens when you follow the standard advice

A team deploys an AI agent that calls a hosted LLM to answer customer questions. They set up a standard APM tool, instrument the HTTP client, and configure alerts for p99 latency above some threshold and error rate above 1%. Everything looks fine for weeks. Then a support manager mentions that customers are complaining about wrong answers. The dashboards show normal latency, low error rate, no alerts. Digging into logs reveals that a meaningful share of responses contained hallucinated product IDs. The APM never noticed because the API returned 200 OK with a valid JSON body.

This is not hypothetical. A common failure mode in production RAG systems is the "empty retrieval" problem: the vector search returns zero documents, the LLM is prompted with no context, and it confidently makes something up. Traditional APM sees a successful database query (0 rows returned is not an error) and a successful LLM call. No red lights. But the user gets a wrong answer, and trust erodes.

Another gap: token usage and cost. A runaway agent loop can make dozens of LLM calls in a single user request, burning through far more tokens than expected. APM might show a latency spike, but it won't tell you that the request cost 50x the baseline. Token-level cost tracking is required, and most APM tools don't provide it out of the box.

## A different mental model

Instead of thinking about requests and responses, think about **intent, execution, and outcome**. For AI systems, you need to monitor:

1. **Intent accuracy** — did the system understand what the user wanted? This often requires a separate classifier or a human-in-the-loop eval.
2. **Execution traces** — not just HTTP spans, but the full chain of reasoning steps, tool calls, and retrieval results. OpenTelemetry's semantic conventions for GenAI attempt to standardize this, but adoption is uneven.
3. **Outcome quality** — was the final answer correct, relevant, and safe? This is where traditional APM completely drops the ball.

The key shift is from **infrastructure metrics** to **semantic metrics**. Latency and error rates are still necessary, but so are metrics like retrieval recall@k, answer faithfulness score, and token cost per request. These are not available from an APM vendor's default dashboard. They have to be built.

A practical approach: wrap LLM calls in a custom instrumentation layer that logs the prompt, the response, token counts, and a quality score (from a small evaluator model or heuristic). Then ship those logs to a system that can alert on quality degradation. This is more work than installing an APM agent, but it is the only way to catch the failures that matter.

## Worked example: catching empty retrieval

Consider a customer support agent with access to three tools: `search_knowledge_base`, `get_order_status`, and `escalate_to_human`. In a healthy run, the agent calls `search_knowledge_base`, gets three relevant chunks, and answers. In a failure mode, retrieval returns zero chunks (because the query was phrased oddly), and the agent either hallucinates or escalates unnecessarily.

Traditional APM would show: one LLM call, one tool call, 200 OK, sub-second latency. No error. But the user experience is broken. To catch this, log the retrieval result count and alert if it is zero for more than a chosen threshold of requests. Here is a minimal Python example using OpenTelemetry to add a custom span attribute:

```python
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

tracer = trace.get_tracer(__name__)

def search_knowledge_base(query: str):
    with tracer.start_as_current_span("retrieval.search") as span:
        results = vector_store.similarity_search(query, k=3)
        span.set_attribute("retrieval.num_results", len(results))
        if len(results) == 0:
            span.set_status(Status(StatusCode.ERROR, "empty retrieval"))
        return results
```

This adds a custom attribute and marks the span as error when retrieval is empty. But note: this only works if the APM backend supports custom span attributes and can alert on them. Some don't, or charge extra for custom metrics.

## Worked example: token cost tracking

A common trap is to assume that all LLM calls cost the same. In reality, input and output tokens are priced differently, and the ratio varies by workload. A single agent run might use 2,000 input tokens and 500 output tokens; if the agent loops ten times, the per-request token cost multiplies accordingly. Traditional APM won't show this; token counts must be aggregated per request and alerted on when cost per request exceeds a threshold.

Here is a JavaScript example for tracking token usage in a Node.js environment using the OpenAI SDK:

```javascript
import OpenAI from 'openai';
import { metrics } from '@opentelemetry/api';

const meter = metrics.getMeter('ai-agent');
const tokenCounter = meter.createCounter('llm.tokens.total', {
  description: 'Total tokens used by LLM calls',
});

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

export async function callLLM(prompt, model = 'gpt-4o') {
  const response = await openai.chat.completions.create({
    model,
    messages: [{ role: 'user', content: prompt }],
  });
  const usage = response.usage;
  tokenCounter.add(usage.total_tokens, {
    model,
    prompt_tokens: usage.prompt_tokens,
    completion_tokens: usage.completion_tokens,
  });
  return response.choices[0].message.content;
}
```

This records token usage as a metric. Alerts can then be created for when the average tokens per request exceeds a baseline. Without this, cost visibility is effectively zero.

## How to measure the gap you actually have

Rather than relying on a vendor's claim or a blog post's benchmark, measure the gap directly on a representative sample of production traffic. The procedure is straightforward:

1. **Instrument retrieval.** Emit a counter for `retrieval.num_results` on every retrieval call. Aggregate into a histogram per hour.
2. **Instrument tokens.** Emit `llm.tokens.total` per request, tagged by model and prompt/completion split.
3. **Instrument a quality proxy.** For a sample (say, every 100th request), run a small evaluator model that scores the answer against the retrieved context for faithfulness. Store the score.
4. **Compare to your alerts.** For one week, note how many requests had `retrieval.num_results == 0`, how many exceeded your token baseline, and how many scored below your faithfulness threshold. Cross-reference against the alerts your APM actually fired.

If the counts of semantic failures are non-trivial and the APM alert count is near zero, the gap is real and quantified. This procedure uses only instruments you control, and the numbers it produces are specific to your system rather than borrowed from someone else's benchmark.

## The cases where the conventional wisdom IS right

Traditional APM is not useless for AI systems. It still catches:

- **Infrastructure failures**: LLM API outages, network timeouts, rate limit errors. These are real and APM handles them well.
- **Latency regressions**: If p95 latency jumps from sub-second to several seconds, APM will alert. That could indicate a model change, a cold start, or a downstream dependency issue.
- **Basic error rates**: If code throws an exception because the LLM returned malformed JSON, APM catches it.

So the conventional wisdom is right for the *transport layer*. It's wrong for the *semantic layer*. Both are needed. The mistake is assuming that because APM covers the transport, it covers the whole system. It doesn't.

## How to decide which approach fits your situation

The right level of semantic monitoring depends on risk tolerance and the cost of a wrong answer. Use this table to decide:

| Scenario | Traditional APM sufficient? | Additional semantic monitoring needed |
|----------|----------------------------|---------------------------------------|
| Internal chatbot for FAQ | Yes, mostly | Low priority; occasional manual review |
| Customer-facing support agent | No | High: retrieval recall, hallucination detection |
| Code generation assistant | No | High: compilation success rate, unit test pass rate |
| Document summarization | No | Medium: factuality score, coverage of key points |
| Real-time translation | No | Medium: reference-based score, human spot-checks |
| AI-powered search | No | High: click-through rate, zero-result rate |

If you're in the "No" column, invest in custom instrumentation. Start with the highest-risk failure mode: for support agents, that's usually empty retrieval or hallucination. For code assistants, it's syntax errors or non-compiling code. Measure that, alert on it, and iterate.

## Common objections, and responses

**Objection: "We can't afford to build custom monitoring."**
A single hallucination that reaches a customer can cost more in trust and support time than the engineering hours to build a simple evaluator. Start small: log the retrieval result count and token usage. That's two metrics, perhaps 50 lines of code.

**Objection: "Our APM vendor says they support AI monitoring."**
Check what they actually measure. Many vendors have added "LLM observability" features that are HTTP tracing with a new label. They still don't measure semantic quality. Ask them: can you alert on retrieval recall? Can you track token cost per user session? If not, the semantic layer is still unmonitored.

**Objection: "We'll just use human evaluation."**
Human evaluation is great for calibration but too slow and expensive for real-time alerting. Automated metrics are needed to catch regressions within minutes, not days. Use humans to label a sample and train a small evaluator model, then run that evaluator on every request.

**Objection: "This is too complex; we'll just use a better model."**
A better model reduces but doesn't eliminate hallucinations. Even frontier models can hallucinate when given no context. The monitoring gap remains.

## What changes when semantic monitoring is in place

Incident response changes. Instead of waiting for a customer complaint, an alert fires: "Retrieval recall dropped from 95% to 70% in the last 15 minutes." The vector database can be checked — maybe an index rebuild failed, or a new document batch has a different embedding distribution. The issue is fixed before users notice.

Cost visibility improves. A dashboard showing token cost per request, per user, per model helps optimize prompts and choose cheaper models for simple tasks. For example, routing simple queries to a smaller, cheaper model while reserving the frontier model for complex ones can cut costs substantially without hurting quality — but only if the routing logic is measured.

Finally, a feedback loop forms. Every request logs its inputs, outputs, and quality score. Failures can be analyzed, prompts improved, and retrieval models retrained. This is how AI systems get better over time — not by hoping the model improves, but by systematically finding and fixing failure modes.

## Summary

Traditional APM misses the majority of AI-related incidents because it monitors transport, not semantics. To catch the failures that matter, instrument retrieval quality, token cost, and answer correctness. Start by adding two metrics to the AI pipeline: retrieval result count and total tokens per request. Then set alerts for when retrieval returns zero results or token usage exceeds a baseline. This won't cover everything, but it will catch the most common and expensive failures.

## Frequently Asked Questions

**How do I monitor LLM hallucinations in production?**
Direct detection requires ground truth, which is usually unavailable. Proxy metrics help: retrieval recall (did you retrieve relevant documents?), answer consistency (does the same question get different answers?), and factuality scores from a small evaluator model. Start by logging retrieval results and using a lightweight model to score answer relevance on a sample of requests. Alert if the average score drops below a threshold.

**Why does my APM show no errors but users complain about wrong answers?**
Because APM only sees HTTP status codes and latency. A wrong answer is still a 200 OK. Semantic monitoring is needed to catch it. Add custom metrics for retrieval quality and token usage, and consider periodic human evaluation to calibrate automated scores.

**What metrics should I track for a RAG pipeline?**
At minimum: retrieval recall@k (percentage of queries where at least one relevant document is retrieved), average number of retrieved documents, token usage per request, and answer latency. If ground truth is available, track answer accuracy. Also monitor embedding drift and index freshness.

**How much does it cost to add semantic monitoring?**
The engineering cost is typically tens of hours to build a basic pipeline that logs retrieval results and token usage, plus the cost of running an evaluator model on a sample of requests. Compare that to the cost of a single customer-facing hallucination. The right sample rate is a trade-off between evaluation cost and detection latency.

## Your next 30 minutes

Open the code path that handles a single AI request in your service. Add two log lines: one that records the number of documents returned by your retrieval step, and one that records the total token count from the LLM response. Deploy to a staging or low-traffic environment, let it run for an hour, then query the logs for the distribution of retrieval counts and token totals. Any zero-retrieval requests or token totals well above the median are the first incidents your APM missed.
