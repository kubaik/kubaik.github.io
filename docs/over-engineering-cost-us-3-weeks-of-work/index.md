# Over-engineering cost us 3 weeks of work

This is a topic where the standard advice is technically correct but practically misleading. Here's the fuller picture, based on what I've seen work at scale.

We spent three weeks building a microservice with Kafka, gRPC, and a custom event schema for a simple feature. When we finally shipped, the feature worked—except half the users never got past the splash screen because the new service added 400ms of latency. We rolled it back the same day. That mistake taught me the brutal truth: most over-engineering isn’t about clever architecture—it’s about avoiding the boring work of understanding the real cost of complexity.

The feature in question was a real-time notification badge that showed unread messages. It had to update in under 500ms for 95% of requests. Our initial stack was Node.js, Express, and MongoDB with a REST endpoint. That handled the load fine for months. Then someone suggested “scalability” and we jumped to a Kafka cluster, three gRPC services, and a protobuf schema for event types. We spent weeks on message serialization, partition strategies, schema evolution, and dead letter queues. We even added a Redis cache to absorb writes. The new service passed all unit tests and load tests in isolation. But when we pointed production traffic at it, every request that touched the new pipeline added 400ms of latency—and worse, 12% of requests failed outright due to Kafka consumer lag spikes.

I’ll never forget the Slack message I sent at 2 AM: “The microservice works locally. The database works locally. But the latency is killing us.” We rolled back to the old monolith by 6 AM. The rollback script took 6 minutes. The lesson wasn’t that microservices are bad—it was that we measured the wrong things during design. We optimized for horizontal scale and message durability but ignored the actual user-perceived latency and failure rate. That’s the hidden cost of over-engineering: it turns invisible complexity into visible pain at 3 AM.


## The situation (what we were trying to solve)

The product team wanted a real-time notification badge that updated when a new message arrived. The requirement was simple: for 95% of requests, the badge should show the correct unread count within 500ms. The existing solution—a REST endpoint hitting MongoDB—handled 1,200 requests per second with a median latency of 80ms and p95 of 220ms. That was acceptable for months. But then we got a new product requirement: support 10,000 concurrent connections during peak chat hours. Someone argued that the REST endpoint would become a bottleneck under load, so we decided to “future-proof” the system before it broke.

I got this wrong at first. I assumed the bottleneck would be the database or the Node.js event loop. So I drew a diagram with three boxes: API service, event bus, and notification service. I labeled the arrows with “async”, “idempotent”, and “backpressure”. That diagram felt sophisticated, but it ignored the real constraint: the user’s screen. The badge update doesn’t care how many services we add—it only cares that the updated count reaches the browser in under 500ms. We were optimizing for a problem we hadn’t measured yet.

The real problem wasn’t scalability—it was observability. We didn’t have a single dashboard showing the time from message insertion to badge update in production. Without that baseline, we assumed the bottleneck was CPU or I/O, when in reality it was network hops and serialization overhead. We also fell for the “distributed systems are more reliable” myth. Adding Kafka and gRPC introduced new failure modes: consumer lag, schema incompatibility, and serialization timeouts. None of these existed in the monolith.


We learned later that the REST endpoint’s p99 latency under load was 320ms—still under the 500ms bar. The real issue was the client-side polling interval, which we had set to 1 second for “efficiency.” Changing that to 500ms alone cut the perceived latency by 50%. We never needed the fancy architecture.



## What we tried first and why it didn’t work

Our first attempt was to replace the monolith with a microservice architecture: an API gateway, a Kafka topic for message events, a gRPC notification service, and a Redis cache for badge counts. We used Protocol Buffers for schema versioning and added a dead letter queue to handle malformed messages. The architecture looked great in our internal docs. We wrote 4,200 lines of new code, added 5 new services, and configured 3 new infrastructure components: Kafka Connect, Schema Registry, and a Redis cluster.

We tested it with k6 at 5,000 RPS. The services handled the load fine in isolation, but when we pointed production traffic at the new service, the latency exploded. The median latency jumped from 80ms to 240ms, and the p95 jumped from 220ms to 620ms—well over our 500ms target. Worse, 12% of requests failed with a timeout because the Kafka consumer lag spiked when the producer outpaced the consumer during traffic bursts.

I traced the issue to the event flow: message → API service → Kafka → gRPC service → Redis cache → client. That’s five network hops and two serialization layers before the client even saw the update. Each hop added 30–80ms of latency. The Redis cache helped with read throughput, but the write path was now synchronous and blocking.


The failure wasn’t the tools—it was the pattern. We followed the “async everything” playbook from a 2018 microservices tutorial. That tutorial assumed we were building a system that would scale to millions of users. We were building a feature that needed to work reliably today. The pattern we chose optimized for future scale, not present simplicity.



We also hit a wall with schema evolution. We added a new field to the notification event, but the Schema Registry rejected old clients, breaking backward compatibility. We spent two days rolling back the schema change and rewriting the event type. That’s when I admitted we’d over-engineered for a feature that didn’t need it.



## The approach that worked

We rolled back to the monolith and made three small changes. First, we reduced the client polling interval from 1 second to 500ms. That alone cut the perceived latency by 50%. Second, we added a simple WebSocket connection for users who were actively chatting. The WebSocket reduced the update path to a single hop: message inserted → WebSocket push → client updates badge. Third, we added a Redis cache for badge counts, keyed by user ID, with a 1-second TTL. That reduced database reads by 80% and kept the median latency under 120ms even during peak load.

The new system handled 10,000 concurrent connections with a median latency of 110ms and p95 of 280ms—well under the 500ms requirement. Failure rate dropped to 0.1%. We shipped it in 3 days and spent the rest of the sprint on features that actually mattered.


The winning pattern wasn’t fancy—it was boring. We focused on the actual constraint (user-perceived latency) and optimized the path from message insertion to badge update. We avoided distributed systems where a single hop would do. We used caching not for scale, but for consistency under load. And we measured the right thing: the time from database write to client update, not the CPU usage of our services.



This surprised me: the WebSocket connection reduced latency more than the Redis cache did. The cache helped with read throughput, but the real latency killer was the polling interval. Switching to WebSocket shaved 100ms off the median response time because it eliminated the client’s wait for the next poll.



## Implementation details

We implemented the solution in three parts: the client, the server, and the cache.


### Client-side

We replaced the 1-second polling with a WebSocket connection that stays open for the duration of the chat session. The WebSocket is established when the user opens the chat screen and closes when they navigate away. The server pushes badge updates immediately when a new message arrives.

Here’s the client code (TypeScript):

```typescript
// client/chat.ts
const socket = new WebSocket(`wss://api.example.com/notifications/${userId}`);

socket.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'badge_update') {
    updateBadge(data.count);
  }
};

function updateBadge(count: number) {
  document.getElementById('badge').textContent = count > 0 ? count : '';
}
```


This reduced the update path to one network hop: server → client. No polling, no waiting. The client receives the update as soon as the server sends it.


### Server-side

The server listens for new messages in the same Node.js process that handles the REST API. When a message is inserted, it publishes a badge update to all active WebSocket connections for that user. We use the `ws` library for WebSocket handling.

Here’s the server code (Node.js):

```javascript
// server/notifications.js
const WebSocket = require('ws');
const wss = new WebSocket.Server({ noServer: true });

// Track active connections by user ID
const activeConnections = new Map();

// Handle WebSocket upgrade
server.on('upgrade', (request, socket, head) => {
  const userId = request.url.split('/').pop();
  wss.handleUpgrade(request, socket, head, (ws) => {
    wss.emit('connection', ws, request, userId);
  });
});

wss.on('connection', (ws, req, userId) => {
  if (!activeConnections.has(userId)) {
    activeConnections.set(userId, []);
  }
  activeConnections.get(userId).push(ws);

  ws.on('close', () => {
    const connections = activeConnections.get(userId);
    if (connections) {
      activeConnections.set(
        userId,
        connections.filter(conn => conn !== ws)
      );
    }
  });
});

// When a new message is inserted
function onMessageInserted(userId) {
  const connections = activeConnections.get(userId);
  if (connections) {
    connections.forEach(ws => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'badge_update', count: 1 }));
      }
    });
  }
}
```


This keeps the update path simple and synchronous within the same process. No Kafka, no gRPC, no schema registry. The message insertion and badge update happen in the same transaction.


### Cache layer

We added a Redis cache for badge counts to handle users who aren’t actively chatting (so no WebSocket connection). The cache key is `badge:{userId}` and the value is the unread count. We set a 1-second TTL to keep the counts fresh without hammering the database.

Here’s the cache code (Python):

```python
# server/cache.py
import redis

r = redis.Redis(host='redis', port=6379, db=0)

CACHE_TTL = 1  # second

def get_badge_count(user_id: str) -> int:
    cached = r.get(f"badge:{user_id}")
    if cached is not None:
        return int(cached)
    return fetch_from_db(user_id)

def update_badge_count(user_id: str, count: int):
    r.setex(f"badge:{user_id}", CACHE_TTL, count)
    # Also push to WebSocket if active
    push_to_websocket(user_id, count)
```


The cache reduced database reads by 80% during peak hours. It also kept the median latency under 120ms even when the database was under load.



## Results — the numbers before and after

| Metric | Before (monolith + REST) | After (monolith + WebSocket + Redis) |
|--------|--------------------------|--------------------------------------|
| Median latency | 80ms | 110ms |
| p95 latency | 220ms | 280ms |
| p99 latency | 320ms | 410ms |
| Failure rate | ~0% | 0.1% |
| Database reads/sec | 1,200 | 240 |
| Lines of new code | 0 | ~200 |
| Time to ship | N/A | 3 days |


The after numbers might look worse at first glance—especially the latency medians. But these are real user-perceived latencies, not synthetic benchmarks. The p95 of 280ms is still under the 500ms requirement. The key improvement is the failure rate, which dropped from 12% during the Kafka rollout to 0.1% in production. And we shipped the feature in 3 days instead of 3 weeks.



The biggest win wasn’t the numbers—it was the simplicity. The new system has one less infrastructure component, one less serialization layer, and one less failure mode. It’s easier to debug, easier to scale horizontally if we ever need to, and easier to maintain. We measured the right thing this time: the time from message insertion to badge update in production, not the CPU usage of our services.



We also saved money. The Kafka cluster cost us $180/month in AWS MSK fees. The Redis cluster cost $24/month. We shut down the Kafka cluster entirely, so we saved $156/month. Over a year, that’s $1,872—enough to buy a new monitor for every engineer on the team.



## What we’d do differently

If we had to build this feature again, we’d start with the boring solution and only add complexity if we measured a real bottleneck. Here’s the order we’d follow:


1. **Measure first**. Use OpenTelemetry to trace the path from message insertion to badge update in production. Find the real latency bottlenecks before changing anything.

2. **Optimize the path**. If the bottleneck is database reads, add a simple cache with a short TTL. If the bottleneck is network hops, reduce the number of services in the path.

3. **Add WebSocket only for active users**. Don’t enable WebSocket for every user—only for those who are actively chatting. This keeps the connection count manageable and reduces server load.

4. **Avoid distributed systems for simple features**. The microservice we built would have made sense for a system with 10x the traffic and multiple teams. For a single feature with one team, it added more complexity than value.



We also would have avoided the “future-proofing” trap. Future-proofing is a myth—it’s just over-engineering with a fancy name. The real future is unpredictable. The best we can do is make the system easy to change when the future arrives.



## The broader lesson

The lesson isn’t “microservices are bad” or “Kafka is useless.” The lesson is that complexity is a tax you pay every day, not just at scale. Every extra service, every extra hop, every extra serialization layer adds latency, increases failure modes, and slows down development. The tax is invisible until it explodes at 3 AM.



This principle applies beyond notification badges. Teams I’ve reviewed fall into the same trap with GraphQL, event sourcing, and CQRS. They start with a simple REST API, then add GraphQL “for flexibility,” then add Kafka “for scalability,” then add a read model “for performance.” By the end, the system is a distributed puzzle where no single engineer understands the full flow. The real cost isn’t the infrastructure bill—it’s the time spent debugging a failure that could have been prevented by a simpler design.



The principle is this: **design for the problem you have, not the problem you fear.** Measure the actual constraints in production. Optimize the path that matters to users. Add complexity only when you can measure the benefit and justify the cost. Simplicity isn’t the absence of features—it’s the absence of accidental complexity.



This surprised me: the teams that ship fastest aren’t the ones with the most advanced architectures. They’re the ones that measure the real constraints, optimize the boring path, and avoid the temptation of “scaling before it’s needed.” The best architecture is the one you don’t have to debug at 3 AM.



## How to apply this to your situation

Start by measuring the real constraint in your system. Don’t guess—measure. Use OpenTelemetry or your APM tool to trace the path from user action to system response. Find the median and p95 latency, the failure rate, and the resource usage. Then ask: where is the bottleneck, and what’s the simplest way to reduce it?



If you’re building a feature that needs to scale to millions of users, start with a simple design and only add complexity when you measure a real bottleneck. If you’re building a feature that needs to work reliably today, avoid distributed systems where a single hop will do. Use boring tools: REST, WebSocket, a simple cache. Avoid GraphQL if you don’t need its flexibility. Avoid event sourcing if you don’t need its audit trail.



Here’s a checklist to avoid over-engineering:

- **Measure first**: Trace the actual path in production. Find the real latency and failure modes.
- **Optimize the path**: Reduce network hops, serialization layers, and database queries.
- **Add caching for reads**: Use a simple TTL-based cache, not a distributed cache, unless you measure a bottleneck.
- **Use WebSocket for real-time**: Only for users who need real-time updates. Don’t enable it for everyone.
- **Avoid “future-proofing”**: Don’t add Kafka, gRPC, or event sourcing unless you measure a real need.
- **Ship small**: Start with the simplest solution that works. Refactor only when you measure a problem.



For example, if you’re building a real-time dashboard, start with a simple WebSocket connection from the frontend to the backend. Don’t add Kafka and three microservices unless you measure a bottleneck at 10,000 connections. Most dashboards never hit that scale—and if they do, you can refactor later.



## Resources that helped

- **OpenTelemetry**: We used the Node.js and Python SDKs to trace the latency path from message insertion to badge update. The traces showed us exactly where the 400ms latency was coming from.
- **Redis TTL caching**: The Redis documentation on TTL caching helped us set the right cache duration without hammering the database.
- **WebSocket in Node.js**: The `ws` library docs were clear and easy to integrate. We didn’t need a heavy WebSocket framework.
- **Latency numbers every programmer should know**: This classic post by Jeff Dean helped us set realistic expectations for network and serialization latency.



These resources taught us to measure before optimizing and to prefer boring tools over fancy ones. They’re the antidote to the “sophisticated architecture” trap.




## Frequently Asked Questions

**How do I know if my system is over-engineered?**

Start by measuring the latency from user action to system response in production. If the median latency is under your target and the p95 is under 2x the median, your system is likely fine. If you’re using more than three infrastructure components (databases, caches, message queues, etc.) for a single feature, you’re probably over-engineered. Ask: can I delete half the services and still meet the requirements? If the answer is no, reconsider.



**When should I use microservices instead of a monolith?**

Use microservices when you have multiple teams, distinct business domains, or measured bottlenecks that require horizontal scaling. If you’re a single team building a single feature, start with a monolith and split only when you measure a real need. Don’t split “for scalability” unless you can prove the monolith can’t handle the load.



**Is GraphQL always overkill for simple APIs?**

GraphQL is powerful but adds complexity. For a simple CRUD API with a few endpoints, REST is simpler and faster. Only use GraphQL if you need its flexibility (e.g., client-specific queries) or if you’re building a platform with many clients. Even then, measure the actual benefit before committing.



**How do I convince my team to simplify the architecture?**

Show them the latency traces and failure rates from production. Share the cost savings from reduced infrastructure. Point to the time saved in development and debugging. Frame simplicity as a competitive advantage: the team that ships features fastest wins. If they still resist, ask them to justify the complexity with data—not with “best practices” or “scalability.”




Go build the boring thing first. Measure the real cost. Then add complexity only if you must.