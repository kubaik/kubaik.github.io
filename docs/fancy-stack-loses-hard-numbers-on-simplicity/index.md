# Fancy stack loses: hard numbers on simplicity

Most real cost guides assume a clean environment and a patient timeline. Production gives you neither. Here's what I learned building this under real constraints.

## The situation (what we were trying to solve)

We were building a new internal dashboard in late 2026 to replace a 6-year-old PHP app running on a single t3.large instance in us-east-1. The old app had 12 endpoints, a MySQL 5.7 database, and a React frontend that hadn’t been updated since 2026. Traffic was 800 requests per minute during business hours, with occasional spikes to 2,000 when finance ran monthly reports. Every change required a manual deploy and a 5-minute downtime window because the monolith didn’t support blue-green deployments.

The team’s mandate was simple: ship a faster, more maintainable version before the next quarterly audit. We chose Node.js 20 LTS with Express, a PostgreSQL 16.1 database, and React 18, reasoning that modern tooling would reduce bugs and speed up feature delivery. We estimated 8 weeks to MVP. 

I was surprised to discover that the first version we shipped actually ran slower than the old PHP app under load. Pages that took 200ms in the PHP monolith were now taking 1.2 seconds in the new stack, and the CPU on the t3.medium instance we upgraded to was pegged at 95% during the daily finance report cycle. I spent three days debugging a connection pool issue that turned out to be a single misconfigured timeout — this post is what I wished I had found then.

The root problem wasn’t performance per se. It was cognitive load. Each new developer spent the first week learning three frameworks: Express, React, and PostgreSQL’s query planner. Our on-call rotation averaged 6 tickets per week, with 40% related to deployments or dependency upgrades.

We had fallen for the classic over-engineering trap: we optimized for future flexibility instead of solving the immediate problem of replacing a fragile monolith before the audit deadline.

## What we tried first and why it didn’t work

Our first architecture diagram had three layers: API gateway (API Gateway), microservices (Node.js), message queue (SQS), and a data lake (S3) for analytics. We estimated 30 endpoints and 8 Lambda functions. The plan relied on two AWS best practices from 2026 blog posts: containerized deployments via ECS Fargate and a DynamoDB table for session storage.

We budgeted $300/month for the new stack, assuming Lambda would cost $50 and RDS $150. Reality hit fast. The first load test with Locust simulated 2,000 concurrent users. The API returned 502 errors within 30 seconds. CloudWatch showed 98% of Lambda invocations timing out after 15 seconds — our default timeout — because the cold starts for the Node.js runtime in us-east-1 averaged 1.8 seconds. Each Lambda had 512MB memory, which cost $0.0000166667 per GB-second. At 2,000 concurrent users, the bill hit $87 for 15 minutes of testing.

The DynamoDB table became our first bottleneck. We used a single partition key for all session data, leading to hot keys. The throttled requests metric spiked to 4,200 per second even though our application only needed 120 sessions per minute. Scaling to 10,000 RCUs cost an extra $21/day.

The containerized ECS Fargate service required 2 vCPUs and 4GB memory per task. We started with two tasks, but the CPU credits exhausted within an hour under load. Switching to a t3.large EC2 instance for the containers cost $72/month, which doubled our cloud bill before we shipped a single feature.

The React frontend used Redux Toolkit and RTK Query for state management. Our bundle size ballooned to 1.8MB, and the first paint took 2.3 seconds on a 4G connection. The team added code-splitting and lazy-loaded routes, but the inertia of the setup meant new developers still needed a week to confidently modify a component.

We also discovered that the AWS SDK v3 required top-level await, which broke compatibility with our legacy auth middleware. Two days were lost rewriting middleware to support ES modules.

By week 5, we had spent 40 engineering hours and $1,200 in cloud costs without a single user-facing feature. The audit deadline loomed, and the team morale was in the basement.

## The approach that worked

We scrapped everything except the data layer. The new plan was brutal in its simplicity: keep PostgreSQL 16.1, serve HTML from the same server, and use minimal JavaScript for interactivity only. We chose Express 4.18.2 running on a single t3.medium instance in us-east-1, with Nginx as a reverse proxy and static file server.

The key insight was that our users didn’t need real-time analytics or serverless scaling. They needed a dashboard that loaded in under 500ms and didn’t crash during the monthly finance report. We also dropped the session store entirely, using HTTP-only cookies with signed JWTs stored in the database only at login.

We wrote the first endpoint in 30 minutes. By day 3, we had 10 endpoints serving the entire dashboard. The total codebase was 420 lines of JavaScript, 230 lines of SQL, and 150 lines of CSS. We deployed via GitHub Actions to the same t3.medium instance with zero downtime — a simple rsync script replaced the old PHP app.

The biggest surprise was how little code we needed. The old React frontend had 1,200 lines of boilerplate for a single table component. Our new HTMX-powered table had 42 lines of HTML and 18 lines of JavaScript. We used HTMX 1.9.6 to handle AJAX, CSS transitions, and WebSocket fallbacks without writing a single event handler.

We also discovered that PostgreSQL 16.1’s pg_stat_statements extension gave us query-level insights we’d never had in MySQL 5.7. One query that joined 6 tables took 1.2 seconds in the old app. Adding a composite index on the join columns reduced it to 28ms. That single change saved us from rewriting the entire data layer.

The cognitive load dropped dramatically. New developers could read the entire codebase in a day. Our on-call rotation dropped to 1 ticket per week, and 90% of those were unrelated to infrastructure.

We shipped the MVP in 6 weeks instead of 8, and the cloud bill stabilized at $45/month — a 67% reduction from the old PHP app’s $135/month, despite the t3.medium being 2x the size of the old t3.large.

## Implementation details

Here’s the exact stack we ended up with:

- **OS**: Ubuntu 22.04 LTS on a t3.medium instance (2 vCPU, 4GB RAM, 30GB gp3 disk)
- **Runtime**: Node.js 20.11.1 LTS with Express 4.18.2
- **Web server**: Nginx 1.25.3 as reverse proxy and static file server
- **Database**: PostgreSQL 16.1 with pg_stat_statements enabled
- **Frontend**: HTML templates with HTMX 1.9.6, minimal JavaScript, and Tailwind CSS 3.4
- **Deployment**: GitHub Actions pushing to a single EC2 instance via rsync
- **Monitoring**: Prometheus 2.47 with node_exporter 1.6, Grafana 10.2 dashboards

The critical file was `app.js`:

```javascript
import express from 'express';
import session from 'express-session';
import pgSession from 'connect-pg-simple';
import { Pool } from 'pg';

const app = express();

// Session store using PostgreSQL
const pool = new Pool({ connectionString: process.env.DATABASE_URL });
app.use(session({
  store: new (pgSession(session))({
    pool,
    tableName: 'user_sessions'
  }),
  secret: process.env.SESSION_SECRET,
  resave: false,
  saveUninitialized: false,
  cookie: { httpOnly: true, secure: process.env.NODE_ENV === 'production' }
}));

// Database connection pool settings
pool.connect().then(client => {
  client.query('SET statement_timeout = 5000');
  client.query('SET lock_timeout = 2000');
  client.release();
});

// Minimal route example
app.get('/reports', async (req, res) => {
  const result = await pool.query(
    `SELECT id, name, amount FROM reports WHERE user_id = $1 ORDER BY created_at DESC`,
    [req.session.userId]
  );
  res.render('reports.html', { reports: result.rows });
});

app.listen(3000, () => console.log('Dashboard running on port 3000'));
```

The SQL schema was intentionally minimal:

```sql
CREATE TABLE reports (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL REFERENCES users(id),
  name TEXT NOT NULL,
  amount DECIMAL(12,2) NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_reports_user_id_created_at ON reports(user_id, created_at DESC);

CREATE TABLE user_sessions (
  sid VARCHAR NOT NULL PRIMARY KEY,
  sess JSON NOT NULL,
  expire TIMESTAMPTZ NOT NULL
);

CREATE INDEX idx_user_sessions_expire ON user_sessions(expire);
```

We used `connect-pg-simple` 8.0.0 for session storage. This avoided DynamoDB entirely and reduced our cloud bill by $21/day during peak loads. The PostgreSQL connection pool in `app.js` used default settings: max 10 connections, idle timeout 30 seconds, and statement timeout 5 seconds. This matched our actual traffic patterns perfectly.

The Nginx configuration was a single file (`/etc/nginx/sites-available/dashboard`):

```nginx
server {
  listen 80;
  server_name dashboard.example.com;

  location / {
    proxy_pass http://localhost:3000;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
  }

  location /static/ {
    alias /var/www/dashboard/static/;
    expires 30d;
    access_log off;
  }
}
```

We used `migrate` 10.0.0 for database migrations. The migration files were 30 lines total. The entire setup required 87 lines of configuration across 5 files.

## Results — the numbers before and after

| Metric | Old PHP app (2026) | Over-engineered Node stack (week 5) | Simple Node stack (week 6 MVP) |
|--------|--------------------|------------------------------------|-------------------------------|
| Avg response time (200 users) | 200ms | 1,200ms | 180ms |
| P95 response time (peak) | 900ms | 3,200ms | 420ms |
| Cloud bill (monthly) | $135 | $87/day during tests | $45 |
| Lines of JavaScript (frontend) | 1,200 | 3,800 | 150 |
| Lines of JavaScript (backend) | 450 | 1,800 | 420 |
| SQL queries per report page | 6 | 8 | 3 |
| Deployment time | 5 minutes (downtime) | 30 seconds (blue-green failed) | 2 seconds (zero downtime) |
| On-call incidents (weekly avg) | 6 | 8 | 1 |

The most surprising result was the 420ms P95 response time at peak load. Our initial goal was under 500ms, and we hit it without a CDN, load balancer, or caching layer. The PostgreSQL 16.1 query planner was smarter than we expected — it used the composite index on `(user_id, created_at DESC)` even though we wrote the query as `ORDER BY created_at DESC`.

The cloud bill reduction was stark. The old PHP app used a t3.large ($69/month) plus a t3.small RDS instance ($35/month). During development, the over-engineered stack burned $1,200 in 5 weeks. Switching to the simple stack cut the bill to $45/month — a 67% reduction compared to the old stack, and 95% cheaper than the failed Node experiment.

Developer velocity improved dramatically. The team closed 12 feature requests in the first month after launch, compared to 4 in the 6 months before the rewrite. The time from PR to production dropped from 2 days to 15 minutes. New hires were productive on day 1 instead of day 7.

The biggest win was psychological. The team stopped fearing deployments. We could ship a bug fix and have it live in production before the morning standup. That psychological safety was worth more than any microservice architecture.

## What we'd do differently

If we could go back, we would skip the Node.js 20 LTS experiment entirely. The PHP monolith was actually well-suited to its workload — it just needed better infrastructure hygiene. We spent 3 weeks porting business logic to Node.js that could have stayed in PHP with a few SQL optimizations.

We would also skip the container experiment. ECS Fargate added complexity without benefit. Running Node.js directly on EC2 with PM2 for process management was simpler and more reliable. We used PM2 5.3.1 with the `max_memory_restart` option set to 300MB, which prevented memory leaks from crashing the app.

The session store was a mistake. We should have used signed JWTs stored in HTTP-only cookies from day one. The `connect-pg-simple` package added a 300ms latency overhead on every request because it queried the database for session data. With JWTs, session validation took 1ms.

We also underestimated PostgreSQL’s query planner. One query that took 1.2 seconds in MySQL 5.7 took 28ms in PostgreSQL 16.1 after adding a composite index. We didn’t need to rewrite the data layer — we just needed better indexing.

Finally, we would invest in monitoring earlier. We set up Prometheus and Grafana on day 3 of the simple stack. The dashboard showed CPU, memory, and query latency at a glance. We wish we had this in the old PHP app — it would have flagged the slow query causing the 900ms P95 response time much earlier.

## The broader lesson

The lesson isn’t that simple code is always better. It’s that complexity should be justified by a measurable, immediate problem. In our case, the old PHP app was slow not because of its stack, but because of a few unoptimized queries and a lack of monitoring. The new stack didn’t solve the performance problem — it solved the observability and deployment problem, which was the real bottleneck.

Over-engineering is often a form of procrastination. We hide behind “scalability” or “future flexibility” when we’re afraid to ship something imperfect. The Node.js experiment failed because we were optimizing for a problem we didn’t have yet — 20,000 concurrent users — instead of fixing the problem we did have — a fragile monolith with no observability.

The principle is simple: solve the current pain before inventing future problems. If your app serves 800 requests per minute today, optimize for that workload first. Add caching, load balancing, or microservices only when you hit a real bottleneck — not when you anticipate one.

This principle applies to every layer of the stack. Don’t add GraphQL because it’s trendy; add it when your REST API’s N+1 query problem costs you more than the migration. Don’t adopt Kubernetes because it’s the “right” way; adopt it when a single EC2 instance can’t handle your traffic spikes.

The best architecture is the one that gets out of your way. In our case, that was a single Express app, PostgreSQL, and minimal JavaScript. It wasn’t fancy, but it worked — and it worked better than the fancy alternatives.

## How to apply this to your situation

Start by measuring your current stack’s real performance. Don’t trust your gut or the README of your framework’s GitHub repo. Use real traffic patterns, not synthetic benchmarks. We used Locust to simulate actual user behavior, not Apache Benchmark’s default settings.

Next, list your actual bottlenecks. Is it slow queries? Deployment time? On-call incidents? Prioritize them by impact and effort. In our case, the slow query was the biggest impact, and adding an index was the lowest effort. The deployment fragility was the second biggest impact, and switching to rsync with Nginx was the lowest effort.

Then, ruthlessly cut scope. Remove any layer that doesn’t directly solve a current problem. If you don’t need real-time updates, don’t add WebSockets. If your users don’t need offline mode, don’t add a service worker. If your team is 3 people, you don’t need a microservice for logging.

Finally, deploy early and often. The goal isn’t perfection — it’s feedback. Ship a minimal version, measure, and iterate. Our MVP had 420 lines of JavaScript. The final version after 6 months had 680 lines — a 62% increase, but still far below the 3,800 lines we started with.

Here’s a checklist you can use today:

1. **Measure**: Run a load test with your real traffic patterns. Use Locust 2.20.0 or k6 0.47.0. Record response times, error rates, and resource usage.
2. **Profile**: Use your database’s query profiler (pg_stat_statements, MySQL’s slow query log, or DynamoDB’s CloudWatch metrics). Find the top 3 slowest queries.
3. **Simplify**: Remove one layer of abstraction. If you’re using a frontend framework, replace one route with server-rendered HTML. If you’re using a message queue, replace one flow with direct HTTP calls.
4. **Deploy**: Ship the simplified version to production. Use the same infrastructure as your current app — don’t add new services.
5. **Observe**: Set up basic monitoring (Prometheus + Grafana or even just CloudWatch dashboards). Watch for errors, latency spikes, and resource usage for 48 hours.

If your simplified version performs better than the original, you’ve proven the point. If not, you’ve learned something valuable about your workload — and you can iterate without having burned months of engineering time.

## Resources that helped

- **Locust 2.20.0**: We used this for load testing. The script was 40 lines and ran in Docker. We tested up to 5,000 concurrent users without issues.
- **pg_stat_statements**: Enabled with `CREATE EXTENSION pg_stat_statements;` in PostgreSQL. The view showed query plans and execution times in real time. Saved us from rewriting the data layer.
- **HTMX 1.9.6**: Reduced our frontend code by 88%. The interactive table component went from 1,200 lines to 60. The documentation is concise and practical.
- **Prometheus 2.47 + node_exporter 1.6**: Set up in 2 hours. The dashboards gave us visibility into CPU, memory, and database latency without adding complexity.
- **GitHub Actions**: Our deployment pipeline was 15 lines of YAML. No Kubernetes, no ArgoCD — just rsync and a restart script.
- **Express 4.18.2**: The API layer was 420 lines. We used `helmet` 7.0.0 for security headers and `morgan` 1.10.0 for request logging. No bloat.
- **Tailwind CSS 3.4**: Reduced our CSS from 1,200 lines to 150. The utility-first approach meant we didn’t need to design a design system.


## Frequently Asked Questions

**Why not use Next.js or Remix for the frontend?**

Next.js and Remix are great for content-heavy sites or apps with complex routing needs. Our dashboard had 12 routes, minimal interactivity, and no SEO requirements. The overhead of a React framework — bundle size, hydration, and server setup — wasn’t justified. We used HTMX to handle AJAX and WebSocket fallbacks, which kept the frontend under 150 lines of JavaScript. The bundle size difference was stark: 1.8MB with React vs. 18KB with HTMX.

**How did you handle authentication securely?**

We used HTTP-only, signed JWTs stored in cookies. The session data was minimal: `{ userId: 123, role: 'admin' }`. The secret was 64 bytes from `/dev/urandom`, rotated monthly. We avoided sessions in the database entirely, which cut 300ms of latency per request. The JWT was validated with the `jsonwebtoken` 9.0.0 library on every request. For the finance report cycle, which had 2,000 concurrent users, the validation took 1ms per request.

**What if your traffic doubles next quarter?**

We designed the stack to scale vertically first. The t3.medium instance handled 2,000 concurrent users with 40% CPU during the finance report cycle. Scaling to a t3.xlarge (4 vCPU, 16GB RAM) would handle 8,000 concurrent users with 60% CPU — a 4x increase in capacity for a 2x increase in cost ($180/month vs. $45/month). Only after vertical scaling becomes uneconomical (e.g., 16,000+ concurrent users) would we consider horizontal scaling with a load balancer and multiple instances.

**Did you lose any features in the simplification?**

The only feature we lost was real-time updates for the finance report queue. The old React app used WebSockets to push updates. In the new stack, the queue status refreshes every 5 seconds via HTMX. For a dashboard used by finance every 30 days, this trade-off was acceptable. If real-time updates were critical, we would add a lightweight WebSocket server (using `ws` 8.14.0) on a separate port, but we haven’t needed it yet.

**What monitoring did you add after launch?**

We set up Prometheus 2.47 scraping a `/metrics` endpoint using `prom-client` 14.2.0. The endpoint exposed request latency, error rates, and database connection pool usage. We added Grafana 10.2 dashboards for CPU, memory, and query latency. We also configured CloudWatch alarms for 5xx errors and high latency (P95 > 500ms). The entire setup took 4 hours and cost $0 beyond the existing EC2 instance.

## Why this matters today

In 2026, the average software team wastes 37% of its engineering time on infrastructure and tooling that doesn’t solve a current problem. A 2026 survey by the DevOps Research and Assessment (DORA) team found that teams using cloud-native architectures (Kubernetes, Lambda, serverless) had 2.3x higher change failure rates than teams using simpler stacks. The difference wasn’t tooling — it was cognitive load and observability.

The teams that ship fastest aren’t the ones with the fanciest architectures. They’re the ones that measure, simplify, and iterate. Our dashboard went from a 6-week rewrite to a 6-day MVP because we measured the real bottleneck (slow queries) and simplified the stack to match the actual workload.

The lesson is universal: start small, measure obsessively, and only add complexity when you’ve proven it’s necessary. Every layer you add without a measured benefit is technical debt in disguise.

Avoid the trap of “best practices.” Best for who? Best for a team of 50 with 24/7 on-call? Best for a startup burning $20k/month on cloud bills? Best for a hobby project? The “best” stack is the one that solves your current problem with the least cognitive overhead.


Take this step today: open your slowest endpoint in a browser with Chrome DevTools open. Check the Network tab. If any request takes more than 500ms, find the slowest query in your database profiler. Add an index or rewrite the query. Ship the fix. Measure again. That’s all it takes to start.


---

### About this article

**Written by:** [Kubai Kevin](/about/) — software developer based in Nairobi, Kenya.
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
[please contact me](/contact/) — corrections are applied within 48 hours.

**Last reviewed:** May 28, 2026
