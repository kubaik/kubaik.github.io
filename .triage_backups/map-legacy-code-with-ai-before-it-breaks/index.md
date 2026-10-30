# Map legacy code with AI before it breaks

## Why legacy systems resist understanding

Legacy systems are not hard to read because the code is clever. They are hard to read because they were never designed as a whole. They grew. A caching layer was added one quarter and the configuration was never updated. An endpoint was added and the monitoring was never extended to cover it. A feature flag was turned on for a migration and never turned off. The documentation describes the system someone intended to build, not the system that is running.

That gap has a predictable shape. The documented behavior promises low latency and high availability; production delivers occasional multi-second stalls and a steady background error rate. Nobody is lying. The documentation was true once, or was written from an architecture diagram rather than from the running configuration.

AI is useful here, but not in the way marketing material suggests. It does not repair the system. It helps you build an accurate map of the system as it actually behaves, so that human judgment can be applied to the right places. The value is in reducing the cost of understanding, which is the real reason nobody wants to touch these codebases.

This article describes a workflow that combines static analysis, runtime instrumentation, and a language model as an analysis aid. It also covers where the approach breaks down, because the failure modes are more instructive than the success stories.

## Three layers of analysis, and what each one can and cannot see

Any serious attempt to map a legacy system needs three kinds of evidence, and each has blind spots.

**Static analysis** parses source files and builds a graph of what calls what. It is cheap, reproducible, and requires no running system. Its blind spot is that it sees possibility, not reality. Dead code, unreachable branches, and configuration values that differ between environments all look identical to a parser. A call graph will happily show you a path that no request has taken in two years.

**Dynamic inference** observes the running system: traces, metrics, and logs. It shows which code paths are actually exercised, which endpoints are slow, and which errors occur. Its blind spot is coverage. If a library is not instrumented, its calls are invisible. If a log line is truncated, the detail you need is gone. Absence of evidence in traces is not evidence of absence in the code.

**Contextual synthesis** is where a language model earns its place. Given the call graph and the runtime evidence, it can correlate the two and produce a readable account of a failure: this endpoint is slow, here is the chain of calls responsible, here are the configuration keys involved, here is a candidate fix. Its blind spot is that it will produce a confident answer whether or not the evidence supports one. It interpolates. That is the central risk of the whole workflow.

The practical consequence is that the model's output is a hypothesis, not a finding. Everything it produces must be checked against the code or against a measurement.

## Extracting a semantic map of the codebase

You do not need to parse every file before starting. A useful map can be built from a small entry point and expanded outward. The goal is a machine-readable structure that links functions to their callers and callees, endpoints to their handlers, and configuration keys to the code that reads them.

Parsing source with a real grammar is more reliable than regular expressions, especially for a language with dynamic features. Tree-sitter grammars exist for most widely used languages and expose a query interface for extracting nodes of interest.

```python
import tree_sitter_php as tsp
from tree_sitter import Language, Parser

# Load the PHP grammar
PHP_LANGUAGE = Language(tsp.language())
parser = Parser()
parser.set_language(PHP_LANGUAGE)

with open("legacy/Router.php", "r") as f:
    code = f.read()

tree = parser.parse(bytes(code, "utf-8"))

# Extract function declarations and call expressions
query = PHP_LANGUAGE.query("""
    (function_declaration
        name: (name) @func_name)
    (call_expression
        function: (name) @call_name)
""")

matches = query.matches(tree.root_node)
for match in matches:
    # A single match may capture either a declaration or a call;
    # inspect the capture name rather than assuming both are present.
    for capture_name, nodes in match[1].items():
        for node in nodes:
            print(capture_name, node.text.decode("utf-8"))
```

The output is a list of edges such as `Router::route -> UserController::getUser`. Accumulated across the files you care about, this becomes the call graph.

Two cautions. First, dynamic dispatch defeats static call graphs: a call through a variable, a service container, or a magic method will not appear as a direct edge. Second, the graph is a starting point for questions, not an answer. Its value is that it lets you ask "what reaches this database table" without reading the whole repository.

## Instrumenting a system that was never built for observability

Runtime evidence is what separates a guess from a diagnosis. OpenTelemetry is the pragmatic choice because it is vendor-neutral and has auto-instrumentation agents for several runtimes, including Java and PHP.

For a Java application, the agent attaches at startup:

```bash
java -javaagent:opentelemetry-javaagent.jar \
  -Dotel.service.name=legacy-java-app \
  -Dotel.traces.exporter=otlp \
  -Dotel.exporter.otlp.endpoint=http://otel-collector:4317 \
  -jar app.jar
```

Pin the agent version in your build rather than relying on whatever is current, and record the version alongside your traces so that a later change in instrumentation is distinguishable from a change in the system.

For a PHP application, configuration typically goes in `php.ini`:

```ini
; php.ini
opentelemetry.enable=1
opentelemetry.service_name=legacy-php-app
opentelemetry.traces_exporter=otlp
opentelemetry.metrics_exporter=otlp
opentelemetry.exporter_otlp_endpoint=http://otel-collector:4317
```

Auto-instrumentation covers common HTTP servers, database drivers, and cache clients. It does not cover everything. Custom HTTP clients, homegrown connection pools, and older database drivers are frequently missed, and a missed library is a blind spot that will distort every conclusion drawn from the traces. When you find one, wrap it manually:

```java
Span span = tracer.spanBuilder("SOAP Request").startSpan();
try (Scope scope = span.makeCurrent()) {
    span.setAttribute("soap.endpoint", endpoint);
    // perform the request
} catch (Exception e) {
    span.recordException(e);
    span.setStatus(StatusCode.ERROR);
} finally {
    span.end();
}
```

Before trusting any analysis, verify coverage. A simple check is to pick three endpoints you know are used and confirm that each produces a trace with a database span and an outbound-call span where you expect one. If a span is missing, fix instrumentation before drawing conclusions.

## Feeding the evidence to a model

The prompt matters less than the structure of the input. Give the model the call graph, a summary of latency and error rates per endpoint, and the configuration keys that appear in the affected code paths. Ask for specific, checkable claims.

A workable instruction set:

```
You are assisting with the analysis of a legacy application.
You will be given a call graph, per-endpoint latency and error summaries,
and a list of configuration keys referenced by the relevant code.

For each of the five slowest endpoints:
1. State the suspected root cause as a testable hypothesis.
2. Cite the specific file and function involved.
3. Name the configuration key or query that would confirm or refute it.
4. Describe the smallest change that would test the hypothesis.
5. State what evidence would falsify your explanation.

Do not speculate beyond the provided evidence. If the evidence is
insufficient to identify a cause, say so explicitly.
```

The last two instructions are the important ones. Asking for a falsification condition forces the model to commit to something you can check, and explicitly permitting "insufficient evidence" removes the pressure to produce a confident answer from thin data.

A useful output looks like a hypothesis with a test attached:

```json
{
  "endpoint": "/api/v1/users/{id}",
  "hypothesis": "Sequential cache lookups for user metadata and permissions dominate latency.",
  "evidence_cited": ["trace span 'redis.get' appears twice per request", "p95 of 420ms correlates with cache miss rate"],
  "confirming_measurement": "Compare p95 with cache warm vs cold; count redis.get spans per request.",
  "smallest_test": "Add key versioning and re-measure p95 for the same traffic profile.",
  "falsified_if": "Latency is unchanged when cache hit rate is 100%."
}
```

That structure is what makes the output usable. It is also what makes it auditable: when the hypothesis turns out to be wrong, you can see which piece of evidence misled the analysis.

## A worked example: the endpoint that was not talking to a database

Consider a common pattern. An endpoint shows a p95 of roughly 400ms. The team's assumption is that the database queries are inefficient, because that has been the cause before.

The call graph shows the endpoint reaching a service class, which calls a client library. The traces show that the client library produces no database span; instead there is an outbound HTTP span with a duration close to the total request time. The configuration file referenced by that client contains a URL pointing at a service that was replaced some time ago.

The reasoning chain is short but each step is checkable:

1. The endpoint's latency is dominated by one outbound call, not by database work. This is visible directly in the trace waterfall.
2. The outbound call targets a host recorded in a configuration file. This is visible in the span attributes and in the file itself.
3. The target is a legacy service. This is a fact about the environment, not something the model can know; a human confirms it.

The fix — repointing the configuration at the current service — is small. The reason it survived for years is that the static call graph looked healthy and the documentation described the current architecture. Only the runtime evidence showed the mismatch.

Note what the model contributed and what it did not. It did not discover that the service was decommissioned. It correlated a latency outlier with a specific span and a specific configuration key, which is exactly the kind of correlation that is tedious to do by hand across many endpoints. The environmental fact came from a person.

## How to measure whether any of this helped

Claims about latency and error-rate improvements are only meaningful if you can reproduce the measurement. Before changing anything, record:

- The exact traffic profile or a replay of it, if you have one.
- p50, p95, and p99 latency for the endpoint, over a window long enough to include normal variation.
- Error rate by status class, and the count of timeouts separately, since timeouts often surface as client-side errors rather than server 5xx.
- The instrumentation version and collector configuration.

After the change, re-measure under the same conditions. If you cannot reproduce the traffic profile, say so when reporting the result; a before-and-after comparison across different load is not evidence.

A useful discipline is to state the expected effect before making the change, in the form "p95 should fall below X because the redundant call is removed." If the measured effect does not match, the hypothesis was wrong even if the number improved, and something else changed.

## Failure modes and how to detect them early

**Incomplete instrumentation.** The most damaging failure, because it is silent. A missing span makes an entire code path invisible, and the model will reason confidently about the paths it can see. Detect it by checking span coverage for known-used endpoints before starting analysis.

**Truncated or unstructured logs.** If a log line is cut off at a fixed length, the detail you need — the query, the error cause, the identifier — is gone. A model given truncated logs will fill the gap, and the filled gap will look plausible. Use structured logging with explicit field limits rather than silent truncation, and treat any field that is frequently at its limit as suspect.

**Confident but wrong recommendations.** A model asked to optimize will optimize, whether or not optimization is warranted. Suggestions to swap one component for another are frequently based on general reputation rather than on the specifics of your workload. Treat any recommendation that involves replacing a component as requiring a benchmark on your own traffic before it is considered.

**Optimizing what does not matter.** A low-contention code path changed from one map implementation to another is a real change with no measurable effect. This is not harmful in itself, but it consumes review attention and adds risk. Require a stated measurement before accepting a change whose justification is performance.

**Analysis without authority.** The most common outcome of a technically successful investigation is a report nobody acts on. If the team that owns the system has no capacity or no mandate to change it, the analysis is wasted. Establish who will act on the findings before producing them.

## When this approach is the wrong choice

There are situations where the workflow above will not pay off, and recognizing them early saves effort.

If the runtime is old enough that no instrumentation agent exists and no maintained parser is available, the dynamic layer is unavailable and the static layer is unreliable. The realistic options are to place a proxy in front of the system and analyze the proxy's traffic, or to plan a replacement.

If the system has no logs, no metrics, and no tests, the model has nothing to reason from and will produce plausible fiction. Adding minimal structured logging is a prerequisite, not an optional step.

If the system is very large, a whole-system analysis produces a report too broad to act on. Scope the work to a single endpoint or a single module, and expand only when the first result has been verified and acted upon.

If the organization is not prepared to change the system, no amount of analysis will help. A small proof of concept on one endpoint, with a measured before-and-after, is usually the only argument that moves the conversation.

## A short checklist before you start

- Pick one endpoint with a known problem. Do not start with the whole system.
- Confirm that the endpoint produces a trace, and that the trace includes the spans you expect.
- Extract the call graph for the files that endpoint touches.
- Write down the current latency percentiles and error rate, with the measurement conditions.
- Ask the model for hypotheses with a confirming measurement and a falsification condition.
- Verify each hypothesis against the code or a measurement before acting.
- After the change, re-measure under the same conditions and compare against the stated expectation.

## What to do in the next 30 minutes

Choose one endpoint that users complain about, open its trace in your observability tool, and check whether the spans you expect are actually present. If a database call or an outbound request is missing from the trace, you have found your first problem, and it is an instrumentation problem rather than a performance one. Fix that before running any analysis, because every conclusion drawn from incomplete traces will be wrong in a way that looks convincing.
