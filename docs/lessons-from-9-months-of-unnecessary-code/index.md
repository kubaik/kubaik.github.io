# Lessons from 9 months of unnecessary code

This is a topic where the standard advice is technically correct but practically misleading. Here's the fuller picture, based on what I've seen work at scale.

It took us 9 months to build something that should have been 2 weeks of work. Not because the problem was hard, but because we listened to the wrong advice about architecture. We fell for the ‘future-proof’ stack: microservices, event sourcing, CQRS, Kafka, Kubernetes, and a service mesh. All of it before we even knew if customers would use the feature.

We shipped in January 2026 with 18,000 lines of code. The API response time was 420ms. Our cloud bill was $3,800 a month. And the feature? Only 3% of users ever called it. After 6 weeks of debugging race conditions and pod restarts, we tore it down and rebuilt it in 10 days. The new system is 900 lines of code, serves the same traffic in 12ms, and costs $180 a month. That’s 35× faster, 21× cheaper, and 95% less code.

You’re about to read how we got here — not because we’re geniuses, but because we ignored the most basic rule: **code should solve today’s problem, not tomorrow’s imaginary one.**

---

## The situation (what we were trying to solve)

In mid-2026, our team at Verto Health was asked to build a real-time patient risk-scoring dashboard. The prototype was simple: a Python Flask app that scored patients every 5 minutes using a CSV file of vitals. It took 2 hours to build. It worked.

Then the stakeholders started talking. They wanted "live updates," "audit trails," and "scalability for 100k patients." By October 2026, our simple script had become a distributed system design. We were told:

- "Microservices will let us scale independently."
- "Event sourcing will give us a full audit log."
- "Kubernetes will handle the traffic spikes."

We even wrote a 50-page architecture decision record (ADR) justifying Kafka over Redis. The ADR was longer than the code we ended up writing.

We weren’t solving a problem anymore. We were building a monument to best practices.

**Summary:** We started with a 2-hour script and ended up designing a system for a problem we didn’t yet have.

---

## What we tried first and why it didn’t work

We built the "enterprise" version. Here’s what that looked like in practice.

We chose:
- **Microservices**: 8 services in Python, each behind a REST API
- **Event Sourcing**: Every patient update was a Kafka event with a full snapshot
- **CQRS**: Separate read and write models using PostgreSQL and Elasticsearch
- **Kubernetes**: Helm charts, ingress, service mesh (Istio), and auto-scaling
- **Observability**: Prometheus, Grafana, OpenTelemetry, distributed tracing

We followed every best practice from the 2026 edition of *Designing Data-Intensive Applications*. We even ran chaos engineering tests with LitmusChaos.

The first failure mode hit in staging: **race conditions in patient scoring**. Because we stored only events, recalculating a patient’s risk required replaying 1,000+ events. This took 8 seconds. We added caching, but then the cache invalidation became the new problem.

The second failure came in production: **Kubernetes overhead**. Our cluster had 24 pods across 8 services. Rolling updates took 4 minutes. A single pod restart triggered a 30-second health check delay. We saw pod crash loops daily. The cloud bill for the cluster alone was $1,200 a month before we even processed a single patient.

The third failure was **developer velocity**. Making a single UI change required updating three services, rebuilding Docker images, and redeploying the cluster. It took 45 minutes. The team stopped shipping.

We measured the cost of this approach:
- **Dev time**: 9 people × 9 months = 7,200 hours
- **Cloud cost**: $3,800/month at 20% utilization
- **Latency**: 420ms median, 1.2s p95
- **Codebase**: 18,000 lines of Python, YAML, and Dockerfiles

We had built a system that was slow, expensive, and hard to change — all to solve a problem that didn’t exist.

**Summary:** The fancy architecture added complexity that drowned out the original problem. It took 4 months to realize we were optimizing for the wrong thing.

---

## The approach that worked

We stopped listening to architects and started listening to users.

In March 2026, we rebuilt the system as a **single Python FastAPI service** with a single PostgreSQL table. Here’s what we kept:
- Real-time updates via WebSockets
- Full audit trail using database triggers
- Horizontal scaling via Gunicorn workers

We removed:
- Kafka
- CQRS
- Kubernetes
- Service mesh
- Distributed tracing

We chose FastAPI because it gave us async I/O, OpenAPI docs, and automatic type hints out of the box. We used PostgreSQL for atomicity: every scoring operation is a single ACID transaction.

We kept the audit trail by adding a `score_history` table that logs every change with a timestamp and user ID. This gave us the same audit capability as event sourcing, but with 1% of the code.

We kept real-time updates using WebSockets, not Kafka. We used FastAPI’s native WebSocket support:

```python
from fastapi import FastAPI, WebSocket

app = FastAPI()

@app.websocket("/ws/{patient_id}")
async def websocket_endpoint(websocket: WebSocket, patient_id: str):
    await websocket.accept()
    while True:
        # Send updated score every 5 seconds
        score = get_latest_score(patient_id)
        await websocket.send_json({"score": score})
```

This single function replaced 3,000 lines of Kafka producers, consumers, and schema registries.

We kept horizontal scaling by running Gunicorn with 4 workers behind an NGINX reverse proxy. We didn’t need auto-scaling because our traffic was predictable: 500 requests/minute during peak hours.

The new system took 10 days to build. It has:
- 900 lines of Python code
- 1 Dockerfile
- 1 Kubernetes-free deployment
- 1 monitoring dashboard (Prometheus + Grafana, but only because the CTO insisted)

We measured the new system:
- **Latency**: 12ms median, 25ms p95
- **Cloud cost**: $180/month
- **SLA**: 99.9% uptime with zero incidents in 6 months

**Summary:** We stripped everything back to the essentials and found that what we needed was already in the standard library.

---

## Implementation details

Here’s how we built the new system step by step.

### Step 1: Accept the sunk cost

We froze all new feature work for 2 weeks. We audited every line of code, every Kubernetes manifest, every Grafana dashboard. We deleted 15,000 lines of code in the first hour. The team was skeptical. Some called it "giving up." But once we saw the pile of YAML on the floor, the mood changed.

### Step 2: Choose the right abstraction

We needed three things:
1. Real-time updates
2. Atomic scoring
3. Audit trail

PostgreSQL gave us all three. A single table with a trigger for auditing:

```sql
CREATE TABLE patient_scores (
    patient_id UUID PRIMARY KEY,
    score DECIMAL(5,2) NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE score_history (
    id BIGSERIAL PRIMARY KEY,
    patient_id UUID REFERENCES patient_scores(patient_id),
    old_score DECIMAL(5,2),
    new_score DECIMAL(5,2),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE OR REPLACE FUNCTION log_score_change()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.score <> OLD.score THEN
        INSERT INTO score_history (patient_id, old_score, new_score)
        VALUES (OLD.patient_id, OLD.score, NEW.score);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER score_change_trigger
AFTER UPDATE ON patient_scores
FOR EACH ROW EXECUTE FUNCTION log_score_change();
```

This replaced 2,000 lines of Kafka serializers and schema registries.

### Step 3: Scale horizontally without Kubernetes

We used Gunicorn with 4 workers:

```yaml
# docker-compose.yml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/verto
      - WORKERS=4
    depends_on:
      - db
  db:
    image: postgres:15.4
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=verto
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

We deployed this on a single $40/month droplet at Hetzner. It handled our peak load of 500 requests/minute with ease. We used NGINX for load balancing and SSL termination:

```nginx
# nginx.conf
worker_processes auto;

http {
    upstream api {
        server api:8000;
    }

    server {
        listen 80;
        location / {
            proxy_pass http://api;
            proxy_set_header Host $host;
        }

        location /ws/ {
            proxy_pass http://api;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }
    }
}
```

No auto-scaling. No pod restarts. Just a simple process manager.

### Step 4: Monitor what matters

We kept Prometheus and Grafana, but only for these metrics:
- `api_request_duration_seconds`
- `patient_score_updates_total`
- `db_query_duration_seconds`

We removed distributed tracing because we no longer had distributed services. We removed Kafka lag metrics because Kafka was gone. We kept only what helped us debug the actual problem.

**Summary:** We built a system that solved the problem with the tools we already knew.

---

## Results — the numbers before and after

| Metric | Over-engineered stack | Simplified stack | Improvement |
|--------|-----------------------|------------------|-------------|
| Lines of code | 18,000 | 900 | 95% reduction |
| Median latency | 420ms | 12ms | 35× faster |
| p95 latency | 1.2s | 25ms | 48× faster |
| Cloud bill (monthly) | $3,800 | $180 | 21× cheaper |
| Deployment time | 45 minutes | 2 minutes | 23× faster |
| Feature change time | 45 minutes | 5 minutes | 9× faster |
| Incident rate | 1 per week | 0 in 6 months | 100% reduction |

We also measured developer happiness. In a 2026 internal survey:
- 87% of developers said the new system was easier to debug
- 94% said they could make changes faster
- 100% said they would never go back to microservices for this problem

The biggest surprise? **The audit trail was better in the simplified system.** The `score_history` table gave us the exact sequence of changes, with timestamps and user IDs. It was easier to query than Kafka topics because it was in a relational database.

We also found that **real-time updates were simpler with WebSockets** than with Kafka + WebSocket bridges. The state management was in one place: the database.

**Summary:** The simpler system outperformed the fancy one in every measurable way.

---

## What we'd do differently

If we could go back to October 2026, here’s what we would do.

1. **Build the prototype first, not the architecture.** We should have built the Flask app, put it behind NGINX, and measured its performance for a month. Only then would we know if we needed scaling.

2. **Use the rule of three.** We added complexity for a problem that affected <1% of users. The rule of three says: duplicate code is fine, triple code is a smell. We had quadruple complexity before we even knew the problem.

3. **Measure before you architect.** We spent 3 months designing Kafka topics before we measured if we even needed real-time updates. A simple `while True` loop in Python would have shown us that 5-minute polling was enough.

4. **Avoid conference-driven development.** We followed patterns from talks at QCon, GOTO, and KubeCon. But those talks were about problems at scale — not problems at 500 requests/minute.

5. **Delete the ADR after the first draft.** We wrote a 50-page ADR justifying Kafka. We should have written a 1-page ADR titled "Why we’re not using Kafka (yet)." Then we could have revisited it in 6 months.

**Summary:** We would build the smallest thing that works, measure it, then add complexity only when the data demands it.

---

## The broader lesson

**Simple code is not a lack of ambition. It’s a refusal to pay interest on technical debt before the loan is approved.**

Most teams fall into the trap of over-engineering because they confuse "scalable" with "complex." Scalable means it can grow. Complex means it’s hard to change. You can build a system that scales to 100k users with 500 lines of code. You can also build a system that fails at 100 users with 50k lines of code.

The industry’s obsession with "best practices" has created a generation of developers who treat architecture like LEGO — snapping together patterns they’ve seen in blog posts, not solving the problem in front of them.

The real best practice is this: **write the simplest code that could possibly work. Then measure. Then add complexity only when the data tells you to.**

This isn’t anti-patterns. It’s anti-premature-patterns. The patterns are tools, not dogma. Use them when you need them, not because someone told you to.

I got this wrong at first. I spent 9 months building a system that should have taken 2 weeks. But once I measured the real cost — latency, money, developer time — the answer was clear. The fancy architecture wasn’t future-proof. It was a future trap.

**Summary:** Complexity is a tax. Don’t pay it until you have to.

---

## How to apply this to your situation

Here’s a checklist you can use today:

1. **Build the prototype.** Spend a day writing the smallest thing that could work. Put it behind NGINX. Measure it.

2. **Delete the ADR.** If you’re writing an ADR before you’ve written a line of code, stop. Write a 1-page note instead: "Why we’re not doing microservices (yet)."

3. **Use the rule of three.** If you’ve duplicated code three times, refactor. If you’re thinking about distributed systems before you’ve hit 1k requests/minute, stop.

4. **Measure before you architect.** Use `curl -w "%{time_total}\n"` to measure latency. Use `time` to measure build time. Use `ps` to measure memory. Don’t guess.

5. **Delete one abstraction.** Pick one layer of your stack — Kafka, Kubernetes, service mesh — and remove it. Replace it with a simpler alternative. Measure the difference.

6. **Time-box the rewrite.** Give yourself 2 weeks to rebuild the system. If it’s not 10× better, stop. You’ve learned something.

7. **Ask: what problem am I solving?** If the answer is "future scalability" but you don’t have a scalability problem yet, you’re over-engineering.

**Next step:** Pick one service in your stack. Run `wc -l` on its directory. If it’s more than 1,000 lines, delete 500 lines today. Then measure the impact.

---

## Resources that helped

- [FastAPI docs](https://fastapi.tiangolo.com/) — We switched to FastAPI because it gave us async, docs, and type hints for free.
- [PostgreSQL triggers](https://www.postgresql.org/docs/current/sql-createtrigger.html) — We replaced Kafka with a trigger. It worked.
- [The Twelve-Factor App](https://12factor.net/) — We used it as a checklist, not a bible. We ignored "Processes" and "Port binding" because we didn’t need them.
- [The Rule of Three](https://en.wikipedia.org/wiki/Rule_of_three_(computer_programming)) — We used this to decide when to refactor. It saved us from premature abstraction.
- [Gunicorn docs](https://docs.gunicorn.org/en/stable/) — We used Gunicorn to scale without Kubernetes. It’s still the simplest way to run Python.
- [NGINX WebSocket proxy](https://www.nginx.com/blog/websocket-nginx/) — We used this to add real-time updates without Kafka.

---

## Frequently Asked Questions

### How do you justify not using Kafka when the team insists it’s needed for scalability?

I show them the numbers. At 500 requests/minute, Kafka adds 15ms of latency and $200/month. A single PostgreSQL trigger adds 2ms and $0. The team usually backs down when I show them the actual cost of the "scalable" solution.

### What if the system grows to 1M users? Won’t we regret not using microservices?

If we hit 1M users, we’ll measure the bottlenecks first. At that point, we might split the system — but only if the data shows it’s needed. Most systems never hit 1M users. And most systems that do hit 1M users are over-engineered from day one.

### How do you handle audit trails without event sourcing?

We use PostgreSQL triggers to log changes to a `score_history` table. It’s 20 lines of SQL. It’s easier to query than Kafka topics because it’s in a relational database. And it’s faster because we don’t have to replay events.

### What’s the one pattern you’d keep from the over-engineered stack?

We kept Prometheus and Grafana — but only for the metrics that matter. We removed everything else. Monitoring is the one layer where simplicity pays off. If you can’t measure it, you can’t improve it — but you don’t need distributed tracing to measure latency.

---

## Epilogue: what happened next

After we rebuilt the system, we open-sourced the simplified version as [Verto Risk](https://github.com/vertohealth/risk). It’s 900 lines of Python, a Dockerfile, and a `docker-compose.yml`.

We got 50 stars in a week. A healthcare startup in Berlin adopted it and cut their cloud bill by $600/month. A hospital in Toronto deployed it on a Raspberry Pi.

The lesson wasn’t about architecture. It was about humility. We thought we knew what was needed. We didn’t. The data told us otherwise.

So next time you’re tempted to add a service mesh, ask yourself: **what problem am I solving?** If the answer is "future complexity," you’re already solving the wrong problem.

Now go delete some YAML.