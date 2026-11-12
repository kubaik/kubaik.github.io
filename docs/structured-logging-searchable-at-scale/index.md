# Structured logging: searchable at scale

Logs are the most expensive debugging tool most teams own, and the bill arrives whether or not anyone can find anything in them. A typical scenario: a service writes JSON lines to stdout, the platform ships them to a log store, and six months later the monthly invoice is large while a search for a specific request ID takes thirty seconds and returns nothing useful. The volume grew because every request emitted a dozen lines of prose, and the cost grew because those lines were stored as opaque strings with no indexable structure. The fix is not a new log vendor; it is a schema discipline that makes each line both cheap to store and fast to query. This article walks through the mechanism that makes large log volumes unsearchable and expensive, then shows the structured logging practices that reverse both problems.

## The situation: logs that are written but never read

Most teams start with `console.log` or a language-native logger and never revisit the format. A line like `User 12345 logged in from 10.0.0.5` is readable by a human and useless to a query engine. When the volume is small, grep works. When the volume reaches millions of lines per day, grep is no longer an option: the log store bills by ingested bytes, and every query scans the full text of every matching line unless the store has an inverted index on the fields you filter by.

The failure is gradual. First, a single service logs at INFO with a message per request. Then a background job adds another message per item. Then a third-party library logs its own headers at DEBUG, and someone sets the global level to DEBUG to chase a bug and forgets to revert it. The volume doubles, then triples, and the cost follows linearly because most log stores price on ingestion and retention, not on query count. The searchability degrades faster than the cost rises, because the fields needed for filtering — request ID, user ID, tenant, route, status code — are buried inside free-text messages that the query engine must parse at read time.

A common symptom: an on-call engineer searches for a request ID and gets a timeout, then falls back to `kubectl logs` on a pod that has already been recycled. The logs exist somewhere, but the path from question to answer is too slow to be useful during an incident. The problem this article addresses is how to design log lines so that the common queries are index-backed and the storage cost is proportional to the value of the line, not to the verbosity of the code that wrote it.

## The mechanism: why free-text logs are expensive and slow

Log stores fall into two broad categories: full-text search engines (Elasticsearch, OpenSearch, Loki with its label index) and columnar or object-storage-backed systems (ClickHouse, BigQuery, S3+Athena, CloudWatch Logs Insights). Each has a different cost model, but the same underlying mechanism causes trouble: **the fields you filter on must be extracted at write time or indexed at write time; if they are not, the query engine pays the cost at read time.**

In a full-text engine, every line is tokenized and inverted-indexed by default. That makes grep-style search fast, but the index itself consumes storage and memory. If you log a 2 KB JSON blob per request, the index may be several times larger than the raw line. The engine also has a mapping problem: if a field sometimes appears as a string and sometimes as a number, the index may reject the document or coerce it silently, and queries against that field become unreliable. Elasticsearch's dynamic mapping will create a `text` field for a numeric ID if the first document happens to have a quoted value, and then range queries on that field stop working.

In a columnar system, the cost is different: you pay for the columns you scan. If your log line is a single `message` string, every query scans the entire string column, even if you only want the status code. If you extract `status`, `route`, and `duration_ms` as separate columns, a query that filters on `status = 500` reads only the status column and the columns you project. The difference in scanned bytes can be orders of magnitude, and in systems that bill by bytes scanned (BigQuery, Athena), that difference is the bill.

The searchability problem is the same in both cases: **the query engine cannot use an index on a field that was never extracted as a field.** A log line that says `request completed in 230ms` has no `duration_ms` field. A log line that says `user=12345` inside a message has no `user_id` field unless the store parses it with a regex at query time, which is slow and error-prone. The fix is to emit structured fields at write time, so the store can index or columnize them once and every subsequent query is cheap.

There is a second, subtler mechanism: **cardinality**. In Loki, labels are indexed and every unique label combination creates a new stream. If you put `request_id` or `user_id` in a label, you create millions of streams, and the index and ingester memory blow up. Loki's documentation explicitly warns against high-cardinality labels. The same principle applies in Elasticsearch: a `keyword` field with millions of unique values consumes heap for the terms dictionary. The structured logging practice is to separate **indexed fields** (low cardinality, used for filtering: service, environment, level, route, status code) from **payload fields** (high cardinality, used for correlation: request ID, user ID, trace ID) that live in the log body and are searched only when needed.

### A worked example of the cost difference

Consider a service that handles 10 million requests per day and emits one log line per request. These figures are illustrative, chosen to make the arithmetic visible; substitute your own numbers.

- A prose line like `POST /checkout completed for user 12345 in 230ms with status 200` averages roughly 70 bytes, plus a timestamp and hostname added by the shipper, call it 150 bytes. At 10 million lines per day that is about 1.5 GB/day of raw ingestion.
- The same event as structured JSON with an envelope and a handful of context fields is larger raw: field names, punctuation, and quoting might push a line to 300–400 bytes. Call it 350 bytes, or 3.5 GB/day.

So the structured line costs more to ingest in raw bytes. The payoff is on the read side. In a columnar store, a query that filters on `status_code = 500` against a single `message` column must scan every byte of that column for the day. Against a table where `status_code` is its own integer column, the engine reads only that column — a few bytes per row instead of a few hundred — plus whatever the query projects. If a dashboard runs that query every five minutes, the scanned-bytes difference compounds into the dominant cost. In an inverted-index store, the same logic applies in reverse: filtering on an indexed `status_code` touches a small term dictionary, while a full-text match on the message scans and scores every document.

The lesson is not "structured logs are always cheaper." It is that **structure moves cost from read time (repeated, unbounded, driven by query patterns) to write time (once, predictable, driven by schema).** Whether that trade is worth it depends on how often the fields are queried. For fields used in dashboards and alerts, it almost always is.

## The fix: a schema, a logger, and a retention policy

The fix has three parts: define a schema for the fields you will always emit, configure the logger to emit them as structured data, and set retention so that high-volume, low-value lines expire before they dominate the bill.

### 1. Define a minimal, stable schema

A workable schema for a typical web service has two tiers. The **envelope** fields are present on every line and are low cardinality: `timestamp`, `level`, `service`, `environment`, `version`, `message`. The **context** fields are present when relevant and may be high cardinality: `request_id`, `trace_id`, `user_id`, `tenant_id`, `route`, `method`, `status_code`, `duration_ms`, `error_type`. The envelope goes into indexed fields or columns; the context goes into the log body as JSON, with the most common filters (route, status_code) promoted to indexed fields if the store supports it.

The schema should be enforced in code, not by convention. A shared logger module that accepts a typed context object prevents the drift where one service logs `userId` and another logs `user_id` and a third logs `uid`. In TypeScript, a type like `LogContext` with optional fields and a `logger.info(message, context)` signature makes the schema visible at the call site. In Go, a `slog.Logger` with `With` attributes does the same. In Python, `structlog` binds context to a logger instance.

### 2. Configure the logger once, not per call site

A common mistake is to format the message at the call site and pass a string to the logger. That defeats structured logging because the logger cannot separate fields from prose. Instead, pass the message as a constant and the variable data as fields.

```typescript
// logger.ts — a minimal structured logger with a stable envelope
import pino from 'pino';

export type LogContext = {
  request_id?: string;
  trace_id?: string;
  user_id?: string;
  tenant_id?: string;
  route?: string;
  method?: string;
  status_code?: number;
  duration_ms?: number;
  error_type?: string;
};

export const logger = pino({
  level: process.env.LOG_LEVEL ?? 'info',
  base: {
    service: process.env.SERVICE_NAME ?? 'unknown',
    environment: process.env.NODE_ENV ?? 'development',
    version: process.env.SERVICE_VERSION ?? '0.0.0',
  },
  timestamp: pino.stdTimeFunctions.isoTime,
  formatters: {
    level: (label) => ({ level: label }),
  },
});

// Usage: the message is a constant, the context is structured.
logger.info({ request_id, route: '/checkout', method: 'POST', status_code: 200, duration_ms: 42 }, 'request completed');
```

This example uses Pino, a widely used Node.js logger that emits JSON lines by default. The important part is not the library but the shape: a constant `message` and a structured object. The same pattern works with `winston`, `bunyan`, Go's `log/slog`, Python's `structlog`, or a hand-rolled JSON encoder. The logger should also redact sensitive fields (authorization headers, passwords, tokens) at the logger level so that no call site can accidentally log them. Pino supports `redact` paths; other loggers have equivalent options.

For languages without a mature structured logger, a small helper is enough:

```python
# log.py — structured JSON lines with a stable envelope
import json, os, sys, time

SERVICE = os.environ.get("SERVICE_NAME", "unknown")
ENV = os.environ.get("ENVIRONMENT", "development")
VERSION = os.environ.get("SERVICE_VERSION", "0.0.0")

def log(level: str, message: str, **context):
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "level": level,
        "service": SERVICE,
        "environment": ENV,
        "version": VERSION,
        "message": message,
    }
    record.update(context)
    sys.stdout.write(json.dumps(record, separators=(",", ":")) + "\n")
```

Both examples emit one JSON object per line. That is the format most log shippers and stores expect, and it preserves types: `status_code` is a number, not a string, so range queries work.

One caveat worth stating plainly: the Python helper above writes to stdout without locking. In a multi-threaded program, concurrent calls can interleave at the write boundary and corrupt lines. If the process is multi-threaded, wrap the write in a lock, or use a logging library that handles synchronization.

### 3. Separate indexed fields from payload fields

If you use Loki, the recommendation is to keep labels low-cardinality: `service`, `environment`, `level`, and perhaps `route` if the route set is bounded. Do not put `request_id` in a label. Put it in the JSON body and use Loki's `| json` parser at query time, or use a store that supports structured metadata. If you use Elasticsearch, map `request_id` as `keyword` with `index: false` if you only need to retrieve it, or as `keyword` with an index if you need to search it; the trade-off is heap versus query speed. If you use ClickHouse or BigQuery, define the table schema explicitly so that `status_code` is an integer column and `duration_ms` is a float, not a string inside a message.

The decision of which fields to index is a cost decision. Indexing a high-cardinality field costs storage and memory; not indexing it costs query time. A useful rule: index fields you filter on in dashboards and alerts (service, environment, level, route, status code); do not index fields you only use for ad-hoc correlation (request_id, user_id). For the latter, a full-text search on the message body or a JSON extraction at query time is usually good enough.

### 4. Set retention by value, not by convenience

Most log stores let you set retention per index, per stream, or per table. The common failure is a single retention period for everything: 30 days for debug logs that are never read and 30 days for audit events that must be kept for a year. The fix is to route logs by level and category into different stores or indices with different retention. Debug logs can go to a short-retention index (for example, 7 days) or be sampled. Info logs can go to a medium-retention index (for example, 30 days). Error logs and audit events can go to a long-retention index with a lower sampling rate. The routing can happen in the log shipper (Fluent Bit, Vector, Filebeat) based on the `level` field, or in the store's index lifecycle management.

A second lever is **sampling**. Not every successful request needs a log line. If you log every request at INFO, you are paying for lines that are only useful for aggregate metrics, which belong in a metrics system, not a log store. Log at INFO for state changes and errors, and use metrics for counters and latency histograms. If you need per-request logs for debugging, sample them (for example, 1 in 100) or log them only when a debug flag is set per request.

## Normalizing routes: a failure mode worth its own section

The single most common schema drift is a route field that quietly becomes high-cardinality. An application logs `route: req.url` and the route looks fine in development, where paths are `/users/1` and `/orders/2`. In production, the same field receives `/users/8f3c2a1b-...` and `/orders/2024-06-01T12:00:00Z`. The number of unique values tracks the number of distinct resources, not the number of endpoints, and the index grows without bound.

The fix is to log the matched route template, not the request URL. Most web frameworks expose the pattern that matched — `/users/:id` in Express, the route pattern in Go's `http.ServeMux` patterns or router libraries, the URL rule in Flask. Log that. Keep the raw URL in the context payload if you need it for debugging, but do not index it.

The same failure appears with `user_id` accidentally promoted to a label during a migration, or with a `tenant` field that turns out to be per-customer rather than per-plan. The general rule: **any field whose unique-value count scales with traffic or with the number of customers is a payload field, not an indexed field.**

## What to monitor: the signals that tell you the schema is drifting

The schema is not a one-time decision; it drifts as services are added and libraries change. Four signals catch drift early.

**Field cardinality.** Monitor the number of unique values for each indexed field. If `route` suddenly has thousands of unique values, someone is putting a path parameter into the route field. That blows up the index and makes dashboards unusable. The fix is to normalize routes at the logger level, using the framework's route template rather than the raw URL.

**Log volume by service and level.** A sudden increase in DEBUG lines from one service usually means a log level was changed and not reverted, or a retry loop is logging on every attempt. Alert on the rate of lines per service per level, not on absolute volume, so the change is caught before the bill is.

**Query latency and scanned bytes.** In stores that expose query statistics (BigQuery shows bytes processed; Elasticsearch shows took time and total hits), track the cost of the top dashboards. If a dashboard that used to scan 10 MB now scans 10 GB, a field that was previously indexed has been moved into the message body, or a new high-cardinality field was added to the index. Treat query cost as a regression test: a dashboard that exceeds a budget should fail a check in CI, the same way a slow test does.

**Missing fields.** If a new service logs without the `service` or `environment` envelope, its lines will not appear in the shared dashboards. A schema validation step in the log pipeline (a shipper filter that drops or tags non-conforming lines) catches this at write time rather than at query time.

### How to measure each signal

These checks are cheap to run and do not require a new tool:

- **Cardinality:** in the log store, run a count-distinct query on each indexed field over the last 24 hours, grouped by service. Compare the result week over week. A field that grows with traffic is misclassified.
- **Volume by level:** query the rate of lines per service per level over time and plot it. A step change is the signal; a slow climb is usually a new feature.
- **Scanned bytes:** most columnar stores return bytes processed with each query result, and most dashboards can log their own execution statistics. Record the bytes for the top ten dashboards daily and alert on a threshold.
- **Missing envelope:** a query for `NOT service:*` or the equivalent null check on the envelope fields, run daily, surfaces non-conforming producers.

## The broader lesson: logs are a data product, not a side effect

The principle behind all of this is that logs are a data product with consumers (on-call engineers, dashboards, alerting rules) and a cost (ingestion, storage, query). Treating them as a side effect of `console.log` is what makes them expensive and unsearchable. The teams that get this right define a schema, enforce it in a shared logger, separate indexed fields from payload fields, and set retention by value. The tools are interchangeable: Pino or `slog` or `structlog` for emission, Fluent Bit or Vector for routing, Elasticsearch or Loki or ClickHouse for storage. The discipline is the same.

A useful test for any log line: if it were seen in isolation during an incident, would it identify which service, which environment, which request, and what happened? If the answer requires reading the surrounding lines, the line is under-structured. If the answer requires a regex, the line is over-prose. The target is a line that is self-contained, typed, and cheap to index.

There is also a hard-to-reverse decision buried here: the choice of log store and schema. Migrating from free-text logs to structured logs is incremental and safe; migrating from one log store to another after two years of retention is not. If you are early, choose a store that supports structured fields and per-index retention, and define the schema before the volume makes the decision for you. If you are already at volume, the schema change is still worth it: run both formats in parallel, route the new format to a new index, and let the old index expire.

### A decision checklist before choosing a store

- Does the store support typed fields (integers, floats, booleans) rather than treating everything as text?
- Does it support per-index or per-stream retention, so debug and audit lines can expire on different schedules?
- Does it expose query cost (bytes scanned, time taken) so dashboards can be budgeted?
- Does it warn against or prevent high-cardinality index keys, or will it accept them silently until it falls over?
- Can the shipper route by field value (level, service) without parsing the message body?
- What is the documented retention limit and the documented ingest limit on the plan you intend to buy?

If a candidate store fails the first three questions, it will reproduce the problem this article describes regardless of how the log lines are formatted.

## Frequently Asked Questions

**How do I make logs searchable without indexing every field?**
Index the fields you filter on in dashboards and alerts: service, environment, level, route, status code. Keep high-cardinality fields like request ID and user ID in the log body and search them with a full-text or JSON extraction query when needed. This keeps the index small and the common queries fast. The trade-off is that ad-hoc searches on high-cardinality fields are slower, which is usually acceptable because they are rare.

**What is the difference between structured logging and log aggregation?**
Structured logging is the format of the line: JSON with named fields and types. Log aggregation is the pipeline that ships lines from many services to a central store. You can have aggregation without structure (many lines of prose in one place) and structure without aggregation (JSON lines on local disk). The practices in this article combine both: structured lines shipped to a store that can index the fields.

**Why does my log bill keep growing even though traffic is flat?**
The usual causes are a log level left at DEBUG, a retry loop logging on every attempt, a new library logging at INFO on every request, or retention set longer than the value of the lines. Check the volume by service and level first; the increase is usually concentrated in one service and one level. Then check retention: debug lines kept for 30 days are often the largest line item and the least read.

**Should request IDs be labels in Loki?**
No. Loki indexes labels and creates a stream per unique label combination; a request ID label creates millions of streams and destabilizes the ingester. Put request IDs in the log body and use the `| json` parser or structured metadata to filter on them. The same principle applies to any high-cardinality field: keep it out of the index and search it in the body.

**Does structured logging increase storage cost?**
It can increase raw ingest bytes, because field names and punctuation make each line longer than equivalent prose. The saving comes from query cost: indexed or columnized fields let the store read far less data per query, and per-field retention lets low-value lines expire sooner. Whether the net is a saving depends on query volume and retention policy, which is why both should be measured rather than assumed.

## The next 30 minutes: audit one service's log lines

Open the logger configuration for your busiest service and count how many fields are emitted as structured data versus embedded in the message string. If the message contains variable data (IDs, paths, durations), move those into a context object and keep the message constant. Then check the retention setting for that service's index: if debug and info lines share the same retention as error lines, split them. The goal for the next 30 minutes is not a full migration; it is one service emitting a stable envelope and one retention policy that reflects the value of the line.
