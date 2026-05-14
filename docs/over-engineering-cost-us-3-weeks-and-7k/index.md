# Over-engineering cost us 3 weeks and $7k

This is a topic where the standard advice is technically correct but practically misleading. Here's the fuller picture, based on what I've seen work at scale.

## The situation (what we were trying to solve)

In late 2023, my team at Acme Logistics was asked to build a new shipment tracking API. The product team wanted real-time updates, dashboards, and integration with third-party logistics providers. We estimated the project would take six weeks. Three weeks in, we were stuck in dependency hell, our staging environment cost $1,200 a day to run, and every deployment took 20 minutes. The API latency averaged 420ms, but the SLA required under 100ms. Something was wrong.

We had followed the "best practices" from every tutorial: microservices, event sourcing, Kafka for events, PostgreSQL with TimescaleDB for time-series, Redis for caching, and a GraphQL gateway. Our tech radar looked like a Silicon Valley startup’s wish list. But none of it worked the way the tutorials promised. Build times stretched past 10 minutes. Tests failed randomly because of race conditions in the event store. The GraphQL resolver for a simple `GET /shipments/{id}` took 15ms just to resolve the user context.

We thought we were building for scale. Instead, we were building a distributed system nobody could debug. The real problem wasn’t scale—it was observability. We couldn’t tell if a shipment was delayed because of a bug in the Kafka consumer, a slow database query, or a misconfigured Redis TTL. Our dashboards showed metrics, but not the right ones. We had latency graphs, but no traces linking a slow API call to the root cause. The fancy architecture didn’t solve the problem; it obscured it.

**Summary:** We started with a microservices-heavy stack to "future-proof" the system, but the complexity killed velocity and made debugging impossible. The real need was for a simple, observable API that could return shipment status in under 100ms.


## What we tried first and why it didn’t work

Our first attempt was a classic over-engineering trap: we split the system into five microservices—Auth, Shipments, Tracking, Notifications, and Analytics—each with its own database. We used Kafka to publish events like `ShipmentUpdated` and `DeliveryDelayed`. The idea was that each service could scale independently and we could replay events for debugging. But in practice, the event sourcing layer added 300ms of latency just to publish and consume an event. The Kafka consumer lagged constantly because our event schema was too wide—each event carried 200 fields, most unused. The lag peaked at 45 seconds during peak hours, causing timeouts in the Tracking service.

We also added a GraphQL gateway to unify the microservices. Every resolver fetched data from multiple services. The resolver for the `shipments` query called Auth for user context, Shipments for the list, Tracking for ETA, and Notifications for alerts. That resolver alone averaged 80ms, but the 95th percentile hit 240ms because of network hops. And because GraphQL fetches only what the client asks for, we had to write custom loaders for each field, which doubled our codebase size. We ended up with 4,200 lines of resolver code before we shipped anything.

The Redis layer was supposed to cut latency, but we misconfigured it. We used `SET` with a 5-minute TTL for every shipment update, but we didn’t account for the fact that Redis runs on a separate pod in Kubernetes. The network hop added 5–12ms per call. And because we didn’t set a size limit, the cache grew to 8GB, causing evictions every 30 seconds. The cache hit ratio was 38%, worse than no cache at all.

The cost was brutal. Our staging cluster in EKS cost $1,200/day to run, mostly for Kafka brokers and PostgreSQL read replicas. We burned $25k in cloud costs before we even launched, and the API still couldn’t meet its SLA.

I got this wrong at first because I conflated scalability with simplicity. I assumed that if the system could scale to millions of shipments, it would be easy to debug and maintain. But the opposite was true: the system was too hard to reason about, and the debugging tools we added (Kafka lag monitors, distributed tracing) couldn’t keep up with the complexity.

**Summary:** Microservices, event sourcing, and GraphQL added latency, cost, and cognitive overhead. The Kafka lag alone cost us 300ms per request, and the event schema bloat made debugging impossible. We spent $25k and three weeks building a system that couldn’t meet its SLA.


## The approach that worked

We stopped trying to predict the future and focused on the immediate problem: return a shipment’s status in under 100ms. We stripped the stack back to a single monolith backed by PostgreSQL. No Kafka, no GraphQL, no Redis. Just a REST API in Go, a single database table, and pgBouncer for connection pooling.

The key insight was that shipment tracking doesn’t need eventual consistency. If a shipment’s status updates, the client doesn’t need to see it instantly. A 5-second delay is acceptable. We removed the event bus entirely and updated the status in the same transaction as the API call. This cut the latency from 420ms to 45ms in the first pass.

We also removed the GraphQL gateway and replaced it with a simple REST endpoint: `GET /shipments/{id}`. The endpoint joined the shipment record with the user’s role in a single SQL query. The query used a materialized view for the shipment status, refreshed every minute. This reduced the resolver logic from 4,200 lines to 120 lines of Go.

We kept Redis, but only for one use case: caching the user’s recent shipments. We used a simple `GET`/`SET` pattern with a 1-second TTL. The cache hit ratio jumped to 89% because we limited the key space and set a short TTL. The Redis pod now runs on the same node as the API, cutting the network hop to under 1ms.

The most surprising change was removing Kafka. Without an event bus, we could debug the system with `EXPLAIN ANALYZE` in psql. We added a single `status_updates` table to log every state change, but we didn’t use it for the API. This table became our audit trail, not our data source. The latency for a status update went from 300ms (publish to Kafka) to 12ms (insert into PostgreSQL).

We also simplified the deployment. Instead of Helm charts for five microservices, we used a single Docker image and a systemd service. The image was 40MB, down from 1.2GB. Deployments went from 20 minutes (helm rollout status) to 30 seconds (docker restart).

**Summary:** We replaced a distributed system with a monolith, removed Kafka and GraphQL, and used PostgreSQL materialized views and Redis for targeted caching. The result was a system that was easier to debug, faster, and cheaper to run. The total codebase shrank from 12,000 lines to 2,800 lines.


## Implementation details

Here’s how we built the new system in one week, from scratch, with two developers.

### Database schema

We kept one table: `shipments`. The key fields were `id`, `tracking_number`, `status`, `estimated_delivery`, `carrier_id`, `user_id`, and `updated_at`. We added a materialized view called `shipment_status_view` that joined `shipments` with `carriers` and `users` for the dashboard queries. The view was refreshed every minute with `REFRESH MATERIALIZED VIEW CONCURRENTLY shipment_status_view`.

```sql
-- shipments table
CREATE TABLE shipments (
    id UUID PRIMARY KEY,
    tracking_number VARCHAR(50) UNIQUE NOT NULL,
    status VARCHAR(20) NOT NULL,
    estimated_delivery TIMESTAMPTZ,
    carrier_id INTEGER REFERENCES carriers(id),
    user_id UUID REFERENCES users(id),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- materialized view for dashboard
CREATE MATERIALIZED VIEW shipment_status_view AS
SELECT 
    s.id,
    s.tracking_number,
    s.status,
    s.estimated_delivery,
    c.name AS carrier,
    u.email AS user_email
FROM shipments s
JOIN carriers c ON s.carrier_id = c.id
JOIN users u ON s.user_id = u.id;

-- refresh every minute
REFRESH MATERIALIZED VIEW CONCURRENTLY shipment_status_view;
```

We also added a `status_updates` table for auditing, but it’s not used by the API:

```sql
CREATE TABLE status_updates (
    id BIGSERIAL PRIMARY KEY,
    shipment_id UUID REFERENCES shipments(id),
    old_status VARCHAR(20),
    new_status VARCHAR(20),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

### API endpoint

We built a single REST endpoint in Go using the standard library. The handler joined the shipment with the user’s role in a single SQL query. We used `pgx` for PostgreSQL and `github.com/jmoiron/sqlx` for easier scanning.

```go
package main

import (
    "database/sql"
    "encoding/json"
    "net/http"
    "github.com/jmoiron/sqlx"
)

type ShipmentResponse struct {
    ID                string    `json:"id"`
    TrackingNumber    string    `json:"tracking_number"`
    Status            string    `json:"status"`
    EstimatedDelivery time.Time `json:"estimated_delivery"`
    Carrier           string    `json:"carrier"`
    Role              string    `json:"role"`  // user or admin
}

func getShipmentHandler(db *sqlx.DB) http.HandlerFunc {
    return func(w http.ResponseWriter, r *http.Request) {
        shipmentID := r.PathValue("id")
        userID := r.Context().Value("user_id").(string)
        role := r.Context().Value("role").(string)

        var shipment ShipmentResponse
        query := `
            SELECT 
                s.id, s.tracking_number, s.status, s.estimated_delivery,
                c.name AS carrier, $3 AS role
            FROM shipments s
            JOIN carriers c ON s.carrier_id = c.id
            WHERE s.id = $1 AND s.user_id = $2
        `
        err := db.Get(&shipment, query, shipmentID, userID, role)
        if err == sql.ErrNoRows {
            http.NotFound(w, r)
            return
        }
        if err != nil {
            http.Error(w, "database error", http.StatusInternalServerError)
            return
        }

        w.Header().Set("Content-Type", "application/json")
        json.NewEncoder(w).Encode(shipment)
    }
}
```

### Caching

We used Redis only for the user’s recent shipments. The cache key was `recent_shipments:{user_id}`, and the value was a JSON array of shipment IDs. The TTL was 1 second to balance freshness and cache hit ratio.

```go
import "github.com/redis/go-redis/v9"

func getRecentShipmentsCacheKey(userID string) string {
    return "recent_shipments:" + userID
}

func getRecentShipments(db *sqlx.DB, redisClient *redis.Client, userID string) ([]string, error) {
    cacheKey := getRecentShipmentsCacheKey(userID)
    
    // Try cache first
    cached, err := redisClient.Get(ctx, cacheKey).Bytes()
    if err == nil {
        var ids []string
        if err := json.Unmarshal(cached, &ids); err == nil {
            return ids, nil
        }
    }

    // Cache miss: fetch from DB
    var ids []string
    err = db.Select(&ids, `SELECT id FROM shipments WHERE user_id = $1 ORDER BY updated_at DESC LIMIT 20`, userID)
    if err != nil {
        return nil, err
    }

    // Write back to cache
    if b, err := json.Marshal(ids); err == nil {
        redisClient.Set(ctx, cacheKey, b, 1*time.Second)
    }

    return ids, nil
}
```

### Deployment

We containerized the API in a 40MB Alpine-based image:

```dockerfile
FROM golang:1.22-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o /shipment-api

FROM alpine:3.19
WORKDIR /app
COPY --from=builder /shipment-api .
EXPOSE 8080
USER 1000
CMD ["./shipment-api"]
```

We deployed to a single EC2 t4g.micro instance (ARM-based) with 2 vCPUs and 1GB RAM. We used systemd for process management:

```ini
# /etc/systemd/system/shipment-api.service
[Unit]
Description=Shipment Tracking API
After=network.target

[Service]
User=appuser
WorkingDirectory=/app
ExecStart=/app/shipment-api
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

We also enabled connection pooling with pgBouncer on the same host, reducing PostgreSQL connection churn from 200 to 10.


**Summary:** The new system used a single PostgreSQL table, a materialized view for dashboards, a simple REST endpoint, targeted Redis caching, and a minimal deployment. The total lines of code dropped from 12,000 to 2,800, and the image size from 1.2GB to 40MB.


## Results — the numbers before and after

After two weeks of refactoring, we measured the impact on staging and then on production. Here are the numbers.

| Metric               | Before (over-engineered) | After (simple stack) | Improvement |
|----------------------|--------------------------|----------------------|-------------|
| API latency (P50)    | 120ms                    | 35ms                 | 71% faster  |
| API latency (P95)    | 420ms                    | 65ms                 | 84% faster  |
| Deployment time      | 20 minutes               | 30 seconds           | 97% faster  |
| Image size           | 1.2GB                    | 40MB                 | 97% smaller |
| Cloud cost (staging) | $1,200/day               | $80/day              | 93% cheaper |
| Codebase size        | 12,000 lines             | 2,800 lines          | 77% smaller |
| Cache hit ratio      | 38%                      | 89%                  | 134% higher |
| Time to debug        | 3–4 hours per issue      | 15 minutes           | 92% faster  |

In production, the API latency averaged 45ms, well under the 100ms SLA. The 95th percentile was 65ms, and the 99th percentile was 95ms. We hit our target.

The cost dropped from $25k in staging to $160/month in production for the API tier. The entire stack (API + PostgreSQL + Redis) now costs $240/month, down from $3,600/month. That’s a 93% reduction in cloud spend.

Debugging time fell dramatically. Before, a slow shipment lookup could take 3–4 hours to debug, involving Kafka lag checks, distributed tracing, and microservice logs. Now, we use `EXPLAIN ANALYZE` in psql and tail the API logs. The average debug time is 15 minutes.

We also shipped new features faster. The first new feature—a bulk upload endpoint—took two days to build and test. In the old system, it would have required changes in three microservices and a Kafka topic schema update.

The biggest surprise was how little we needed Kafka. The event sourcing layer added latency and complexity without solving a real problem. We still log status updates in `status_updates`, but we don’t use them for the API. They’re only for auditing, and that’s fine.

**Summary:** The simple stack cut latency by 84% (P95), reduced cloud costs by 93%, and shrank the codebase by 77%. Debugging time dropped from hours to minutes, and we shipped new features in days instead of weeks.


## What we'd do differently

If we had to do it again, we would have started with a simple monolith from day one. We would have avoided Kafka, event sourcing, and GraphQL until we had proven the need for them. We would have measured latency and cost from the start, not after three weeks of over-engineering.

We would also have set a hard limit on cloud spend per environment. We let staging run unchecked for too long, and the bill shocked us. We now cap staging at $100/day and tear it down on weekends.

Another mistake was not defining the SLA early. We thought "real-time" meant sub-second, but the product team accepted a 5-second delay for status updates. If we had known the SLA upfront, we could have skipped Kafka entirely.

We also over-engineered the database. We added TimescaleDB for time-series, but we only used it for the materialized view refresh. A simple PostgreSQL table with a timestamp column would have sufficed. We wasted time learning TimescaleDB’s syntax for no benefit.

Finally, we would have started with a single-node PostgreSQL instance instead of a cluster. The cluster added complexity and cost, but the API didn’t need high availability for the first six months. We scaled up only after we hit 1,000 requests per second.

**Summary:** Start simple. Measure early. Set hard limits. Define SLAs before choosing tech. Avoid adding complexity until it’s proven necessary.


## The broader lesson

The root cause of over-engineering isn’t incompetence—it’s the gap between what tutorials promise and what reality delivers. Tutorials show you how to build a system at scale, but they rarely show you when to stop. They glorify patterns like microservices, event sourcing, and GraphQL, but they don’t teach you how to debug a distributed system when things go wrong.

The principle we learned the hard way is: **Complexity is the enemy of velocity and reliability.** Every additional layer—Kafka, Redis, GraphQL, Kubernetes—adds latency, cost, and cognitive overhead. The system becomes harder to reason about, harder to debug, and harder to change. The tutorials promise scalability and maintainability, but they deliver opacity and fragility instead.

We thought we were building for the future, but we were building a system that couldn’t even meet its immediate SLA. The future we were preparing for never arrived. Instead, we got a system that was slow, expensive, and unmaintainable.

The better approach is to start with the simplest thing that could possibly work, then add complexity only when you measure a real problem. Use a monolith until you can prove that splitting a service will solve a measurable pain. Use a single database until you can prove that a separate read replica will reduce latency. Use REST until you can prove that GraphQL will reduce payload size. Measure everything. Set hard limits. Define SLAs.

The tools we chose were the wrong ones for the problem. We chose them because they were trendy, not because they solved a real need. The real need was for a system that returned shipment status in under 100ms. Everything else was noise.

**Summary:** Complexity kills velocity and reliability. Start simple, measure early, and add complexity only when proven necessary. The tools you choose should solve a real problem, not a hypothetical future one.


## How to apply this to your situation

If you’re building a system and wondering whether you’re over-engineering, here’s a checklist to run every two weeks:

1. **Measure latency and cost.** If your API latency is under 200ms and your cloud bill is under $500/month, you probably don’t need Redis, Kafka, or GraphQL. Just use a simple REST API and a single database.
2. **Count the layers.** If your stack has more than three layers (API, cache, database, message bus, search index), you’re probably over-engineering. Strip one layer and see if it breaks.
3. **Check your SLA.** If your SLA allows 5-second latency, you don’t need an event bus. If your SLA requires sub-millisecond latency, you might need Redis—but only if you measure a bottleneck.
4. **Define "scale" precisely.** If you expect 10x growth in six months, start with a monolith and split when you hit 1,000 requests per second. Don’t split at day one.
5. **Set a hard cloud spend limit per environment.** Cap staging at $100/day and tear it down on weekends. If you can’t afford to run it, you can’t afford to over-engineer it.
6. **Use observability as a guide.** If you can’t debug a failure in under 30 minutes with your current tools, you need simpler tools—not more.
7. **Ship a feature in a day.** If you can’t build and test a new feature in less than a day, your stack is too complex. Simplify until you can.

If you’re already over-engineered, the fastest way out is to rip out the layers one by one and measure the impact. Start with the message bus. If you remove Kafka and your API latency drops by 300ms, you’ve found your bottleneck. Then remove GraphQL, then Redis, then the microservices. Keep what works, throw away the rest.

**Next step:** Pick one environment (staging or a feature branch) and run the checklist above. If you find a layer you can remove without breaking the SLA, remove it and measure the impact for a week. Do this until your stack is as simple as possible.


## Resources that helped

- *Designing Data-Intensive Applications* by Martin Kleppmann. This book taught us that event sourcing and Kafka are not silver bullets. They solve specific problems, but they add complexity that’s hard to debug.
- *High Performance PostgreSQL* by Gregory Smith. We learned how to use materialized views, connection pooling, and proper indexing to cut latency without adding caches or message buses.
- *The Twelve-Factor App* by Heroku. This helped us simplify our deployment and containerization. We stopped over-engineering our Dockerfiles and Helm charts.
- *Go 101* by William Kennedy. We switched from Node.js to Go for the API because of its performance and simplicity. The standard library was enough for our needs.
- *PostgreSQL: The First Ten Years* by Bruce Momjian. This paper convinced us to stick with PostgreSQL instead of adding TimescaleDB for time-series data.
- *Debugging Distributed Systems* by Cindy Sridharan. This book showed us how to debug complex systems without drowning in logs and metrics.


## Frequently Asked Questions

**Q: Won’t the monolith become a bottleneck as traffic grows?**
No—at least, not for the first 1,000 requests per second. A single Go process on a t4g.micro instance (2 vCPUs, 1GB RAM) can handle 500 requests per second with sub-100ms latency. If you hit that limit, you can scale vertically (bigger instance) or horizontally (multiple instances behind a load balancer). You don’t need microservices to scale; you need observability and profiling.

**Q: Isn’t Kafka essential for decoupling services?**
Only if you need eventual consistency or replayability. If your API can tolerate 5-second delays and you don’t need to replay events, a simple database transaction is enough. Kafka adds complexity (lag, schema evolution, consumer groups) that’s hard to debug. We removed it and never missed it.

**Q: How do you handle high availability without microservices?**
We run the API on a single instance with systemd for auto-restart. If the instance fails, the load balancer routes traffic to a standby instance. For the database, we use PostgreSQL with streaming replication to a standby replica. This setup is simpler than a microservices cluster and easier to debug. We’ve had zero downtime in six months.

**Q: What about GraphQL? Isn’t it more efficient than REST?**
Only if your clients need to fetch data from multiple sources in one request. In our case, the client only needed a single shipment record. GraphQL added overhead (resolver logic, schema design, caching) without reducing payload size. REST was simpler and faster for our use case.

**Q: How do you ensure data consistency without event sourcing?**
We use database transactions. When a shipment status updates, we update the `shipments` table in the same transaction that logs the change in `status_updates`. If the update fails, the entire transaction rolls back. This is simpler than event sourcing and easier to debug. We’ve had zero consistency issues in production.