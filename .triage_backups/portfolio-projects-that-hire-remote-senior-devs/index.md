# Portfolio projects that hire remote senior devs

Generic tutorials produce portfolios that pass a junior screen and stall at a senior one. The gap is rarely the framework. It is the absence of evidence that you can design, operate, and debug a system when it misbehaves under real constraints.

This article walks through building a URL shortener that demonstrates that evidence: a REST API with a Redis cache, a Redis-backed rate limiter, structured logging, connection pooling, metrics, and a load test you can point at in an interview. The stack is deliberately unglamorous. The interesting part is what happens when it fails.

## What senior remote screens actually probe

A hiring manager reading a portfolio usually has one question: can this person own a system we would have to page someone about? That translates into a handful of concrete signals.

- **Observability.** Can you tell what the system is doing without attaching a debugger? Metrics, structured logs, and a health endpoint answer this.
- **Failure behavior.** What happens when the cache is cold, the database is saturated, or a dependency drops a connection mid-request?
- **Resource discipline.** Do you pool connections, bound retries, and know your memory and cost footprint?
- **Measurement.** Can you state a latency number and explain how you obtained it?

Most portfolio projects answer none of these because they were built to run once on a laptop. The sections below build each signal into the same small service.

## Prerequisites and the target system

You need a machine with Docker, a cloud account with billing alerts configured, Node.js 20 LTS or later, and a GitHub account. No prior cloud experience is required; the services used here are the small, cheap tiers.

The service is a URL shortener with these components:

- A REST API in Node.js with Express
- A PostgreSQL database for durable storage of short-code mappings
- A Redis cache in front of the database
- A Redis-backed rate limiter on the write path
- Structured JSON logging
- Prometheus metrics exposed on a `/metrics` endpoint
- A Jest test suite plus a Docker Compose integration setup

A URL shortener is a good teaching vehicle because the read path is cacheable, the write path is not, and the ratio between them is enormous. That asymmetry is where the interesting failure modes live.

```bash
git init url-shortener
cd url-shortener
npm init -y
npm install express redis pg ioredis rate-limiter-flexible cors helmet winston winston-daily-rotate-file dotenv
npm install --save-dev jest supertest typescript @types/node @types/express @types/jest nodemon
```

Use TypeScript for the type safety it gives you across the request and database boundary. A `tsconfig.json` targeting ES2022 with `strict` enabled is the baseline:

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "commonjs",
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true
  }
}
```

## Step 1 — the API skeleton

`src/index.ts` wires the middleware and routes. Keep this file thin; the interesting logic belongs in modules you can test in isolation.

```typescript
import express from 'express';
import helmet from 'helmet';
import cors from 'cors';
import { createShortUrl, getOriginalUrl } from './routes/shortener';

const app = express();
const PORT = process.env.PORT || 3000;

app.use(helmet());
app.use(cors());
app.use(express.json());

app.post('/shorten', createShortUrl);
app.get('/:shortCode', getOriginalUrl);

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
```

### Region placement is a design decision

Latency between your compute and your data stores is not a detail you fix later. A cache that sits in a different region from the API adds a round trip to every request, which can erase the benefit of caching entirely. A common failure mode is deploying the API and the cache in different regions and then wondering why p99 does not improve.

The fix is to co-locate everything: API, database, and cache in one region. Pick the region closest to your expected users, verify the round-trip latency with a simple timing test, and keep it consistent across environments.

Create the database and cache with the provider CLI. Substitute your own region and credentials:

```bash
aws rds create-db-instance \
  --db-instance-identifier url-shortener-db \
  --db-instance-class db.t4g.micro \
  --engine postgres \
  --engine-version 15.4 \
  --master-username admin \
  --master-user-password "$(openssl rand -base64 16)" \
  --allocated-storage 20 \
  --region <your-region>

aws elasticache create-cache-cluster \
  --cache-cluster-id url-shortener-cache \
  --cache-node-type cache.t4g.micro \
  --engine redis \
  --num-cache-nodes 1 \
  --region <your-region>
```

Both take several minutes to become available. Retrieve the endpoints when they are ready:

```bash
aws rds describe-db-instances --region <your-region>
aws elasticache describe-cache-clusters --region <your-region>
```

Put the endpoints in a `.env` file for local development. In production, read secrets from a managed secrets store rather than a file in the image.

```
NODE_ENV=development
DB_HOST=<rds-endpoint>
DB_PORT=5432
DB_USER=admin
DB_PASSWORD=<rds-password>
DB_NAME=urlshortener
REDIS_HOST=<cache-endpoint>
REDIS_PORT=6379
PORT=3000
RATE_LIMIT_WINDOW_MS=60000
RATE_LIMIT_MAX=100
```

## Step 2 — connection pooling and the cache layer

### Pool, do not connect per request

Opening a database connection per request is the single most common cause of "too many connections" errors in Node services under load. A pool bounds the number of live connections and reuses them.

`src/db.ts`:

```typescript
import { Pool } from 'pg';

const pool = new Pool({
  host: process.env.DB_HOST,
  port: parseInt(process.env.DB_PORT || '5432', 10),
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  database: process.env.DB_NAME,
  max: 20,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 5000,
});

export default pool;
```

The `max` value is a budget, not a suggestion. Every connection costs memory on both the client and the server. Size it to the concurrency your instance can actually sustain, and watch `pool.waitingCount` to see whether requests are queueing for a connection. If it is consistently above zero, you are either under-pooled or your queries are too slow.

### The read path with cache-aside

`src/routes/shortener.ts` implements the write path as a transaction and the read path as a cache-aside lookup.

```typescript
import { Request, Response } from 'express';
import crypto from 'crypto';
import Redis from 'ioredis';
import pool from '../db';
import logger from '../logger';

const redis = new Redis({
  host: process.env.REDIS_HOST,
  port: parseInt(process.env.REDIS_PORT || '6379', 10),
  retryStrategy: (times) => Math.min(times * 50, 2000),
});

const CACHE_TTL = 3600;

export const createShortUrl = async (req: Request, res: Response) => {
  const { url } = req.body;
  if (!url) {
    return res.status(400).json({ error: 'URL is required' });
  }

  const shortCode = crypto
    .createHash('sha256')
    .update(crypto.randomUUID())
    .digest('hex')
    .slice(0, 8);

  const client = await pool.connect();
  try {
    await client.query('BEGIN');
    await client.query(
      'INSERT INTO short_urls(short_code, original_url) VALUES($1, $2)',
      [shortCode, url]
    );
    await client.query('COMMIT');
    await redis.setex(shortCode, CACHE_TTL, url);
    res.json({ shortUrl: `${process.env.API_BASE_URL}/${shortCode}` });
  } catch (err) {
    await client.query('ROLLBACK');
    logger.error('Database error on insert', { error: (err as Error).message });
    res.status(500).json({ error: 'Failed to shorten URL' });
  } finally {
    client.release();
  }
};

export const getOriginalUrl = async (req: Request, res: Response) => {
  const { shortCode } = req.params;

  const cachedUrl = await redis.get(shortCode);
  if (cachedUrl) {
    return res.redirect(302, cachedUrl);
  }

  const client = await pool.connect();
  try {
    const result = await client.query(
      'SELECT original_url FROM short_urls WHERE short_code = $1',
      [shortCode]
    );

    if (result.rows.length === 0) {
      return res.status(404).send('URL not found');
    }

    const originalUrl = result.rows[0].original_url;
    await redis.setex(shortCode, CACHE_TTL, originalUrl);
    res.redirect(302, originalUrl);
  } catch (err) {
    logger.error('Database error on lookup', { error: (err as Error).message });
    res.status(500).send('Server error');
  } finally {
    client.release();
  }
};
```

Note the `finally` block on both paths. A connection that is not released is a connection that is permanently unavailable, and the pool will eventually starve.

## Step 3 — failure modes worth designing for

### Cache stampede

When a popular key expires, every concurrent request for it misses the cache and hits the database at the same time. Under high read volume this produces a latency spike that looks like a database problem but is really a cache-coordination problem.

The standard mitigation is a short-lived lock so that one request rebuilds the value while the others wait briefly.

`src/cache.ts`:

```typescript
import Redis from 'ioredis';

const redis = new Redis({
  host: process.env.REDIS_HOST,
  port: parseInt(process.env.REDIS_PORT || '6379', 10),
});

export const getWithCacheRebuild = async (
  key: string,
  ttl: number,
  fetchFn: () => Promise<string>
): Promise<string> => {
  const cached = await redis.get(key);
  if (cached) {
    return cached;
  }

  const lockKey = `${key}:lock`;
  const acquired = await redis.set(lockKey, '1', 'EX', 10, 'NX');
  if (!acquired) {
    await new Promise((resolve) => setTimeout(resolve, 50));
    const retry = await redis.get(key);
    if (retry) return retry;
    return fetchFn();
  }

  try {
    const value = await fetchFn();
    await redis.setex(key, ttl, value);
    return value;
  } finally {
    await redis.del(lockKey);
  }
};
```

Two details matter here. First, `SET ... NX EX` is atomic, so only one caller acquires the lock. Second, the lock has an expiry, so a process that dies while holding it cannot deadlock the key forever. The fallback `fetchFn()` call after the wait handles the case where the lock holder failed without writing a value.

### Connection leaks and the health endpoint

A health endpoint that reports pool statistics turns an invisible resource problem into a visible one.

`src/routes/health.ts`:

```typescript
import { Request, Response } from 'express';
import pool from '../db';

export const healthCheck = async (_req: Request, res: Response) => {
  const stats = await pool.query('SELECT count(*) FROM pg_stat_activity');
  res.json({
    status: 'ok',
    dbConnections: parseInt(stats.rows[0].count, 10),
    poolSize: pool.totalCount,
    available: pool.idleCount,
    waiting: pool.waitingCount,
  });
};
```

Register it with `app.get('/health', healthCheck);`. When `waiting` climbs during a load test, you have found your bottleneck without guessing.

### Redis reconnection

If the Redis client gives up after a failed connection, the API hangs or errors on every read. Configure bounded retries with backoff and a connect timeout so failures surface quickly instead of cascading.

```typescript
const redis = new Redis({
  host: process.env.REDIS_HOST,
  port: parseInt(process.env.REDIS_PORT || '6379', 10),
  retryStrategy: (times) => Math.min(times * 50, 2000),
  connectTimeout: 5000,
  maxRetriesPerRequest: 3,
});
```

The tradeoff is explicit: fail fast and let the caller see an error, rather than block a request thread waiting on a dependency that may not recover.

### Rate limiting on the write path

Writes are expensive and unbounded. A Redis-backed limiter survives process restarts and works across multiple instances, which an in-memory limiter does not.

`src/middleware/rateLimiter.ts`:

```typescript
import { RateLimiterRedis } from 'rate-limiter-flexible';
import Redis from 'ioredis';

const redisClient = new Redis({
  host: process.env.REDIS_HOST,
  port: parseInt(process.env.REDIS_PORT || '6379', 10),
});

const rateLimiter = new RateLimiterRedis({
  storeClient: redisClient,
  keyPrefix: 'rl_url_shortener',
  points: parseInt(process.env.RATE_LIMIT_MAX || '100', 10),
  duration: parseInt(process.env.RATE_LIMIT_WINDOW_MS || '60000', 10) / 1000,
  blockDuration: 60,
});

export const rateLimiterMiddleware = (req: any, res: any, next: any) => {
  rateLimiter
    .consume(req.ip)
    .then(() => next())
    .catch(() => res.status(429).json({ error: 'Too many requests' }));
};
```

Apply it only to the write route: `app.post('/shorten', rateLimiterMiddleware, createShortUrl);`. Reads are cheap and cacheable; limiting them would hurt legitimate traffic.

## Step 4 — observability and tests

### Structured logging

Unstructured logs are nearly useless once you have more than one service. Emit JSON with a timestamp so your log aggregator can index fields.

`src/logger.ts`:

```typescript
import winston from 'winston';
import DailyRotateFile from 'winston-daily-rotate-file';

const logger = winston.createLogger({
  level: 'info',
  format: winston.format.combine(winston.format.timestamp(), winston.format.json()),
  transports: [
    new winston.transports.Console(),
    new DailyRotateFile({
      filename: 'logs/url-shortener-%DATE%.log',
      datePattern: 'YYYY-MM-DD',
      maxSize: '5m',
      maxFiles: '7d',
    }),
  ],
});

export default logger;
```

### Metrics

`prom-client` gives you counters and histograms that a Prometheus server can scrape.

```typescript
import client from 'prom-client';

const register = new client.Registry();

const httpRequestDuration = new client.Histogram({
  name: 'http_request_duration_seconds',
  help: 'Duration of HTTP requests in seconds',
  labelNames: ['method', 'route', 'status_code'],
  buckets: [0.05, 0.1, 0.3, 0.5, 1, 2, 5],
});

const cacheHits = new client.Counter({
  name: 'cache_hits_total',
  help: 'Total number of cache hits',
  labelNames: ['route'],
});

const cacheMisses = new client.Counter({
  name: 'cache_misses_total',
  help: 'Total number of cache misses',
  labelNames: ['route'],
});

register.registerMetric(httpRequestDuration);
register.registerMetric(cacheHits);
register.registerMetric(cacheMisses);

export { register, httpRequestDuration, cacheHits, cacheMisses };
```

Wire the histogram into Express and expose the registry:

```typescript
import { register, httpRequestDuration } from './metrics';

app.use((req, res, next) => {
  const end = httpRequestDuration.startTimer();
  res.on('finish', () => {
    end({
      method: req.method,
      route: req.route?.path || req.path,
      status_code: res.statusCode,
    });
  });
  next();
});

app.get('/metrics', async (_req, res) => {
  res.set('Content-Type', register.contentType);
  res.end(await register.metrics());
});
```

The cache hit and miss counters are the ones that matter most for this service. Their ratio tells you directly whether the cache is earning its keep.

### Unit tests

Mock the external dependencies so tests run without infrastructure.

```typescript
import request from 'supertest';
import app from '../index';
import Redis from 'ioredis-mock';

jest.mock('ioredis', () => Redis);
jest.mock('pg', () => ({
  Pool: jest.fn(() => ({
    connect: jest.fn(() => ({
      query: jest.fn().mockResolvedValue({ rows: [{ original_url: 'https://example.com' }] }),
      release: jest.fn(),
    })),
    totalCount: 0,
    idleCount: 0,
    waitingCount: 0,
  })),
}));

describe('URL Shortener', () => {
  it('creates a short URL', async () => {
    const res = await request(app).post('/shorten').send({ url: 'https://example.com' });
    expect(res.statusCode).toEqual(200);
    expect(res.body).toHaveProperty('shortUrl');
  });

  it('redirects to the original URL', async () => {
    const res = await request(app).get('/abc123');
    expect(res.statusCode).toEqual(302);
    expect(res.headers.location).toEqual('https://example.com');
  });

  it('returns 404 for a missing URL', async () => {
    const res = await request(app).get('/missing');
    expect(res.statusCode).toEqual(404);
  });
});
```

Run them with `npx jest`. A passing suite is table stakes; the point is that mocking the database and cache correctly is itself a demonstration of understanding the boundaries.

### Integration tests with Docker Compose

Unit tests with mocks will not catch a broken query or a misconfigured pool. Run the same tests against real services.

```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=test
      - DB_HOST=db
      - DB_PORT=5432
      - DB_USER=test
      - DB_PASSWORD=test
      - DB_NAME=test
      - REDIS_HOST=redis
      - REDIS_PORT=6379
    depends_on:
      - db
      - redis
    command: npm run test:integration

  db:
    image: postgres:15.4
    environment:
      POSTGRES_USER: test
      POSTGRES_PASSWORD: test
      POSTGRES_DB: test
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U test -d test"]
      interval: 5s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7.2
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5
```

Run with `docker-compose up --build --exit-code-from app`. The `depends_on` plus healthchecks ensure the app starts only after the dependencies are ready, which eliminates a class of flaky test failures.

## How to produce your own numbers

Do not put a latency figure in a README unless you can reproduce it. Here is how to get one that means something.

Instrument the request duration histogram, then run a load test against a deployed instance. A k6 script that ramps virtual users gives you a distribution rather than a single number:

```javascript
import http from 'k6/http';
import { check } from 'k6';

export const options = {
  stages: [
    { duration: '30s', target: 50 },
    { duration: '1m', target: 200 },
    { duration: '30s', target: 500 },
    { duration: '30s', target: 0 },
  ],
  thresholds: {
    http_req_duration: ['p(95)<150'],
  },
};

export default function () {
  const res = http.post('http://<api-url>/shorten', { url: 'https://example.com' });
  check(res, { 'status is 200': (r) => r.status === 200 });
}
```

Run it from a host near your deployment region, not from your laptop, and record the p50, p95, and p99 from the summary. Then repeat the same test with the cache disabled. The difference between the two runs is the number worth quoting, because it demonstrates that you measured the effect of a specific design decision.

If you want a concrete target to design against, this is a reasonable one: a cache-aside read path should keep p95 well under the database-only p95 under the same load. If it does not, the cache is not being hit often enough, the TTL is too short, or the cache is too far from the API.

## What this project proves in an interview

| Signal | Where it appears in the project |
| --- | --- |
| Observability | `/metrics` endpoint, structured JSON logs, `/health` pool stats |
| Failure handling | Cache rebuild lock, bounded Redis retries, transaction rollback |
| Resource discipline | Bounded connection pool, released connections, explicit timeouts |
| Measurement | Load test script, histogram buckets, before/after cache comparison |
| Testing | Mocked unit tests plus real-service integration tests |

The interview conversation is the real deliverable. When asked about the project, describe a failure you designed around, explain the tradeoff you chose, and show the metric that confirmed the fix. That is what distinguishes a senior candidate from someone who has only written code that runs.

## In the next 30 minutes

Pick one project you already have, add a request-duration histogram and a `/metrics` endpoint to it, and run a short load test against it. Whatever number comes back is the first honest performance figure you can put in a README.
