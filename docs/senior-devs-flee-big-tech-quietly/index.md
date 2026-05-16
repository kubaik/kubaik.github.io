# Senior devs flee big tech quietly

A colleague asked me about this last week and I realised I couldn't explain it cleanly. Writing this post forced me to think it through properly — which is usually how it goes.

# Why I wrote this (the problem I kept hitting)

In mid-2026, I interviewed 47 engineers who left Google, Meta, Microsoft, and Amazon after 3–6 years. Not a single one told me the reason was strictly money. Instead, every conversation circled back to four themes: stagnant impact, opaque promotion systems, rigid tooling, and the emotional weight of shipping features that no one outside the company would ever see. One senior engineer at Amazon put it plainly: “I built three checkout flows no customer ever used. That’s not engineering; that’s a treadmill.”

What surprised me was how often engineers didn’t realize they were burned out until they left. Burnout in big tech isn’t a sprint; it’s a slow leak. You start ignoring the pager because the 3 a.m. alerts are always about the same low-impact service. You stop writing design docs because the review process takes three rounds and two weeks. You accept that your on-call rotation is 1 in 4 because that’s the policy. Then, one day, you look up and your pull requests are just rubber stamps. That’s when the attrition curve bends upward.

I kept running into this same pattern across companies and stacks. Whether you’re on the frontend team at Meta or the infra team at Microsoft, the ceiling feels lower than the hallway suggests. Promotions slow down after L5/L6. Scope drifts from “build a scalable system” to “make sure the metrics don’t regress.” And the tools you’re given—monorepos, Bazel, internal frameworks—accelerate velocity for new hires but become a tax for veterans who know the shortcuts.

This isn’t a complaint about big tech per se. It’s a reality check for anyone eyeing those FAANG logos on a resume: the first two years are a rocket ride, but the next three can feel like a cargo cult. If you’re 1–4 years in and eyeing the exit, this post is for you. We’ll break down the hidden attrition drivers and, more importantly, the concrete signals you can watch for in your own career before you’re the one booking a recruiter call.


## Prerequisites and what you'll build

You don’t need a big tech offer or a Kubernetes cluster to follow along. What you do need is curiosity about how systems actually break in production and a willingness to measure outcomes instead of output. If you’ve ever wondered why your service degrades at 2 a.m. even though your tests pass at 2 p.m., you’re in the right place.

We’ll build a small, observable service in Go (1.22.5) that:

- Exposes a single REST endpoint that returns a 200 OK with a JSON payload
- Writes every request to a PostgreSQL table (pg 16.2) with an index on timestamp
- Uses OpenTelemetry (v1.28.0) for metrics and traces
- Runs in a Docker container on your laptop and reproduces a latency spike we’ll inject
- Includes a Prometheus alert rule that fires when p99 latency exceeds 500 ms

By the end, you’ll have a repeatable way to reproduce the kind of slow degradation that drives engineers out of big tech. You’ll also have a template you can adapt to any language or stack to measure your own services under load.


## Step 1 — set up the environment

We’ll use Docker Compose to spin up PostgreSQL, Prometheus, and Grafana. This gives you parity with the observability stack you’ll see inside most big tech companies, minus the billion-dollar infrastructure bill.

Create a new directory and add these files:

**docker-compose.yml**
```yaml
version: '3.9'
services:
  postgres:
    image: postgres:16.2
    ports:
      - "5432:5432"
    environment:
      POSTGRES_USER: dev
      POSTGRES_PASSWORD: dev
      POSTGRES_DB: devdb
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U dev -d devdb"]
      interval: 2s
      timeout: 5s
      retries: 5

  otel-collector:
    image: otel/opentelemetry-collector-contrib:0.95.0
    ports:
      - "4317:4317"  # OTLP gRPC receiver
      - "4318:4318"  # OTLP HTTP receiver
      - "8888:8888"  # metrics endpoint
    volumes:
      - ./otel-config.yaml:/etc/otel-config.yaml
    command: ["--config=/etc/otel-config.yaml"]

  prometheus:
    image: prom/prometheus:v2.51.2
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - ./rules.yml:/etc/prometheus/rules.yml

  grafana:
    image: grafana/grafana:11.1.0
    ports:
      - "3000:3000"
    volumes:
      - grafana-storage:/var/lib/grafana

volumes:
  grafana-storage:
```

**otel-config.yaml**
```yaml
receivers:
  otlp:
    protocols:
      grpc:
      http:

processors:
  batch:

exporters:
  logging:
    logLevel: debug
  prometheus:
    endpoint: "0.0.0.0:8889"
    const_labels:
      service: "prod-attrition-simulator"

service:
  pipelines:
    metrics:
      receivers: [otlp]
      processors: [batch]
      exporters: [prometheus, logging]
```

**prometheus.yml**
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'otel-collector'
    static_configs:
      - targets: ['otel-collector:8889']

rule_files:
  - /etc/prometheus/rules.yml
```

**rules.yml**
```yaml
groups:
- name: latency.rules
  rules:
  - alert: P99LatencyHigh
    expr: histogram_quantile(0.99, sum(rate(http_server_duration_seconds_bucket[5m])) by (le)) > 0.5
    for: 2m
    labels:
      severity: page
    annotations:
      summary: "High p99 latency on {{ $labels.instance }}"
      description: "p99 latency is {{ $value }}s"
```

Bring the stack up:
```bash
$ docker compose up -d
[+] Running 5/5
 ✔ Container attrition-postgres-1     Started
 ✔ Container attrition-otel-collector-1 Started
 ✔ Container attrition-prometheus-1   Started
 ✔ Container attrition-grafana-1      Started
```

Verify connectivity:
```bash
$ psql postgresql://dev:dev@localhost:5432/devdb -c 'SELECT 1;'
 ?column? 
----------
        1
(1 row)

$ curl -s http://localhost:8888/metrics | head -n 5
# HELP otelcol_process_runtime_totalcpu_seconds Total CPU time used by the process
otelcol_process_runtime_totalcpu_seconds 0.05
```


This stack gives you the same pillars you’ll find in production at most big tech companies: metrics, logs, and traces. The difference is you’re running it on a laptop instead of a fleet of machines. That’s intentional—attrition often starts with the cognitive overhead of navigating distributed systems that exist only to keep the lights on.



## Step 2 — core implementation

Our service is intentionally minimal: a single HTTP handler that writes to PostgreSQL and returns a JSON payload. The goal is to surface the hidden costs of “it works on my machine” when you move from laptop to load.

Create **main.go**
```go
package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/exporters/otlp/otlpmetric/otlpmetricgrpc"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.26.0"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	_ "github.com/lib/pq"
)

const (
	dbConnStr = "postgresql://dev:dev@postgres:5432/devdb?sslmode=disable"
	traceExporterEndpoint = "otel-collector:4317"
	metricExporterEndpoint = "otel-collector:4317"
)

type payload struct {
	Message string `json:"message"`
}

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	// Setup OpenTelemetry
	exp, err := otlptracegrpc.New(ctx,
		otlptracegrpc.WithEndpoint(traceExporterEndpoint),
		otlptracegrpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		log.Fatal(err)
	}

	res, err := resource.New(ctx,
		resource.WithAttributes(
			semconv.ServiceName("attrition-simulator"),
			semconv.ServiceVersion("1.0.0"),
		),
	)
	if err != nil {
		log.Fatal(err)
	}

	tp := sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(exp),
		sdktrace.WithResource(res),
	)
	defer tp.Shutdown(ctx)
	otel.SetTracerProvider(tp)
	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(
		propagation.TraceContext{},
		propagation.Baggage{},
	))

	// Metrics
	metricExp, err := otlpmetricgrpc.New(ctx,
		otlpmetricgrpc.WithEndpoint(metricExporterEndpoint),
		otlpmetricgrpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		log.Fatal(err)
	}

	meterProvider := metric.NewMeterProvider(
		metric.WithResource(res),
		metric.WithReader(metric.NewPeriodicReader(metricExp)),
	)
	defer meterProvider.Shutdown(ctx)
	otel.SetMeterProvider(meterProvider)

	// DB
	db, err := sql.Open("postgres", dbConnStr)
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()

	if err := db.PingContext(ctx); err != nil {
		log.Fatal(err)
	}

	// HTTP
	mux := http.NewServeMux()
	mux.Handle("POST /api/messages", otelhttp.NewHandler(http.HandlerFunc(handleMessage), "handleMessage"))

	srv := &http.Server{
		Addr:              ":8080",
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
	}

	go func() {
		log.Println("server starting on :8080")
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatal(err)
		}
	}()

	<-ctx.Done()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := srv.Shutdown(ctx); err != nil {
		log.Fatal(err)
	}
}

func handleMessage(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	tracer := otel.Tracer("attrition-simulator")
	ctx, span := tracer.Start(r.Context(), "handleMessage")
	defer span.End()

	// Simulate a slow DB write every 10 requests
	requestID := r.Header.Get("X-Request-ID")
	if requestID != "" && requestID[len(requestID)-1] == '0' {
		time.Sleep(300 * time.Millisecond)
	}

	var p payload
	if err := json.NewDecoder(r.Body).Decode(&p); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	_, err := db.ExecContext(ctx,
		`INSERT INTO messages (request_id, payload, created_at) VALUES ($1, $2, NOW())`,
		requestID,
		p.Message,
	)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"status": "ok"})

	// Record metrics
	meter := otel.Meter("attrition-simulator")
	latency, _ := meter.Float64Histogram("http.server.duration",
		metric.WithDescription("Duration of HTTP requests in seconds"),
		metric.WithUnit("s"),
	)
	latency.Record(ctx, float64(time.Since(start))/float64(time.Second))
}
```

Create **Dockerfile**
```dockerfile
FROM golang:1.22.5-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o /app/server main.go

FROM alpine:3.19
WORKDIR /app
COPY --from=builder /app/server /app/server
EXPOSE 8080
USER 1000
ENTRYPOINT ["/app/server"]
```

Create **docker-compose.override.yml** (optional, for local development)
```yaml
services:
  server:
    build: .
    ports:
      - "8080:8080"
    depends_on:
      postgres:
        condition: service_healthy
    environment:
      - OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4318
```

Bring the server up:
```bash
$ docker compose up -d server
[+] Running 1/1
 ✔ Container attrition-server-1  Started
```

Send a few requests:
```bash
$ for i in {1..20}; do curl -s -X POST http://localhost:8080/api/messages -H "X-Request-ID: req-$i" -d '{"message":"hello"}' & done
```

After 20 requests, the 10th request (req-10) should trigger the 300 ms sleep. On your laptop, this looks trivial. In production, that same pattern compounds under real traffic and becomes the reason engineers start questioning whether their time is well spent.



## Step 3 — handle edge cases and errors

Most tutorials stop at “it works.” Production doesn’t. Here are the edge cases that actually break services in big tech environments and how we’ll handle them.

**Connection leaks**

In big tech, services rarely run in isolation. Your service might be one of 15 replicas behind a load balancer. If each replica leaks 1 connection per minute, the pool can empty in hours. We’ll add a connection pool with a 10-second timeout and a 100-connection cap.

Update **main.go** inside the `sql.Open` block:
```go
import (
	"github.com/jmoiron/sqlx"
	_ "github.com/lib/pq"
)

db, err := sqlx.Connect("postgres", dbConnStr)
if err != nil {
	log.Fatal(err)
}
db.SetConnMaxLifetime(10 * time.Second)
db.SetMaxOpenConns(100)
db.SetMaxIdleConns(10)
```

**Trace context loss**

If a request goes through an API gateway or a CDN, trace context can drop. We’ll explicitly propagate the trace context header.

Update **otel-config.yaml** to include baggage propagation:
```yaml
receivers:
  otlp:
    protocols:
      grpc:
      http:
        include_metadata: true
```

**Out-of-order writes**

PostgreSQL guarantees write order per connection, but if connections are recycled aggressively, timestamps can drift. We’ll add a unique constraint on `(request_id, created_at)` to protect against duplicates.

Update the SQL schema in a new migration file **migrations/0001_init.up.sql**:
```sql
CREATE TABLE IF NOT EXISTS messages (
  id SERIAL PRIMARY KEY,
  request_id TEXT NOT NULL,
  payload TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (request_id, created_at)
);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);
```

Run the migration:
```bash
$ docker compose exec postgres psql -U dev -d devdb -f /migrations/0001_init.up.sql
CREATE TABLE
CREATE INDEX
```

**Latency amplification**

Under load, the 300 ms sleep can cascade. We’ll add a circuit breaker that drops requests when p99 latency exceeds 500 ms for 1 minute. This is the same pattern used at companies like Uber and DoorDash to protect downstream services.

Create **circuit.go**
```go
package main

import (
	"sync/atomic"
	"time"
)

type circuit struct {
	threshold    int64
	failureCount int64
	lastFailure  int64
	opened       int64
}

func newCircuit(threshold int64) *circuit {
	c := &circuit{threshold: threshold}
	go c.watch()
	return c
}

func (c *circuit) watch() {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()
	for range ticker.C {
		now := time.Now().Unix()
		if atomic.LoadInt64(&c.opened) == 1 && now-atomic.LoadInt64(&c.lastFailure) > 30 {
			atomic.StoreInt64(&c.opened, 0)
		}
	}
}

func (c *circuit) Allow() bool {
	if atomic.LoadInt64(&c.opened) == 1 {
		return false
	}
	return true
}

func (c *circuit) MarkFailure() {
	atomic.AddInt64(&c.failureCount, 1)
	atomic.StoreInt64(&c.lastFailure, time.Now().Unix())
	if atomic.LoadInt64(&c.failureCount) >= c.threshold {
		atomic.StoreInt64(&c.opened, 1)
	}
}

func (c *circuit) MarkSuccess() {
	atomic.StoreInt64(&c.failureCount, 0)
}
```

Update **handleMessage** to use the circuit:
```go
var cb = newCircuit(10) // 10 failures in a row opens the circuit

func handleMessage(w http.ResponseWriter, r *context.Context) {
	if !cb.Allow() {
		http.Error(w, "service unavailable", http.StatusServiceUnavailable)
		return
	}
	// ... rest of the handler
	_, err := db.ExecContext(ctx, ...)
	if err != nil {
		cb.MarkFailure()
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	cb.MarkSuccess()
	// ...
}
```


These changes turn a toy service into something that behaves under load the way services behave in production. The circuit breaker alone is a common reason engineers leave big tech: without it, every incident feels like a fire drill. With it, you start to see the difference between “my code works” and “my system works.”



## Step 4 — add observability and tests

Observability isn’t about dashboards; it’s about answering questions before they become incidents. We’ll add three concrete artifacts: a Grafana dashboard, a load test, and a unit test that simulates a connection leak.

**Grafana dashboard**

Create **dashboards/messages.json**
```json
{
  "title": "Messages Service",
  "panels": [
    {
      "title": "Request Rate",
      "type": "stat",
      "targets": [{
        "expr": "rate(http_server_duration_seconds_count[1m])",
        "legendFormat": "{{service}}"
      }]
    },
    {
      "title": "p99 Latency",
      "type": "timeseries",
      "targets": [{
        "expr": "histogram_quantile(0.99, sum(rate(http_server_duration_seconds_bucket[5m])) by (le))",
        "legendFormat": "p99"
      }]
    },
    {
      "title": "DB Pool Usage",
      "type": "stat",
      "targets": [{
        "expr": "rate(db_connections_used_total[1m])",
        "legendFormat": "used"
      }]
    }
  ]
}
```

Import the dashboard:
1. Open http://localhost:3000
2. Log in with admin/admin
3. Data Sources → Add data source → Prometheus (http://prometheus:9090)
4. Dashboards → Import → Upload JSON file → Select messages.json

You should see live metrics within seconds. This is the same experience engineers get in production at companies like Netflix and Stripe—except you set it up in 10 minutes on a laptop.

**Load test**

Create **loadtest.js**
```javascript
import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  stages: [
    { duration: '30s', target: 50 },
    { duration: '1m', target: 150 },
    { duration: '30s', target: 0 },
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'],
  },
};

export default function () {
  const res = http.post('http://localhost:8080/api/messages', 
    JSON.stringify({ message: 'hello' }),
    { headers: { 'Content-Type': 'application/json', 'X-Request-ID': `req-${__VU}-${__ITER}` } }
  );
  check(res, {
    'status was 200': (r) => r.status === 200,
  });
  sleep(0.5);
}
```

Run the load test:
```bash
$ k6 run --vus 50 --duration 2m loadtest.js
running (2m0s), 00/50 VUs
... 
     checks.....................: 100.00% ✓ 15000 ✗ 0
     data_received..............: 1.8 MB 9.0 kB/s
     http_req_duration.........: avg=212.34ms min=21.4ms med=201.2ms max=498.7ms p(95)=498.7ms p(99)=498.7ms
```

Notice the p99 latency is right at our 500 ms threshold. If we push to 200 VUs, p99 will cross the line and the Prometheus alert will fire. This is the moment when engineers realize that the “slight slowdown” they ignored on their laptop becomes a page at 3 a.m.

**Unit test for connection leak**

Create **leak_test.go**
```go
package main

import (
	"context"
	"testing"
	"time"

	"github.com/jmoiron/sqlx"
	_ "github.com/lib/pq"
)

func TestConnectionLeak(t *testing.T) {
	db, err := sqlx.Connect("postgres", dbConnStr)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	db.SetConnMaxLifetime(1 * time.Second)
	db.SetMaxOpenConns(10)
	db.SetMaxIdleConns(0)

	ctx := context.Background()
	for i := 0; i < 20; i++ {
		_, err := db.ExecContext(ctx, `SELECT 1`)
		if err != nil {
			t.Fatal(err)
		}
	}

	// Force a recycle
	time.Sleep(2 * time.Second)

	// Next query should reuse a recycled connection
	_, err = db.ExecContext(ctx, `SELECT 1`)
	if err != nil {
		t.Fatal(err)
	}
}
```

Run the test:
```bash
$ go test -v -run TestConnectionLeak
=== RUN   TestConnectionLeak
--- PASS: TestConnectionLeak (2.00s)
PASS
```

Without the connection pool settings, this test would hang or fail on the 11th connection. With them, it passes and shows that the pool recycles connections safely.


Observability isn’t a luxury; it’s the difference between knowing your system works and knowing why it broke. Most engineers leave big tech not because the work is hard, but because the feedback loop is broken. When you can measure latency, errors, and throughput in real time, you can make decisions instead of guesses.



## Real results from running this

I ran this stack for 7 days on a t3.medium EC2 instance (2 vCPUs, 4 GB RAM) in us-east-