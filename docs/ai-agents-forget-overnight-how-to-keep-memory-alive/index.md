# AI agents forget overnight — how to keep memory alive

After reviewing a lot of code that touches memory systems, I keep seeing the same patterns that cause problems later. This post addresses the root cause rather than the symptom.

## The error and why it's confusing

Your AI agent remembers everything in the session, then forgets it all when you restart. Users complain that yesterday’s conversation context is gone, even though the agent’s memory system is enabled. The logs show no errors, just silent data loss. You assumed the agent’s "long-term memory" would persist — after all, the docs said it supported "session-to-session continuity."

I ran into this when a client’s customer support bot started failing compliance audits because it couldn’t recall prior ticket resolutions. The agent’s response time stayed under 200ms, so we blamed the retrieval layer. But the problem wasn’t latency — it was that the memory store was silently rolling back changes every hour.

This is the first symptom teams hit when they underestimate how AI memory systems handle persistence. The confusion comes from conflating three layers of memory:

- **Short-term memory** — the agent’s working context (tokens in RAM, discarded on disconnect)
- **Session memory** — kept in a stateful backend (Redis, PostgreSQL, or a managed vector store) but scoped to a single conversation thread
- **Long-term memory** — persists across sessions and users, often implemented as embeddings or structured logs, but rarely used for immediate recall in the same thread

The error looks like this in the agent’s trace:
```json
{
  "event": "session_end",
  "memory": {
    "stored": false,
    "reason": "session_ttl_exceeded"
  }
}
```

Most agents set a default TTL of 24 hours on session data. If your agent’s session isn’t flagged as "long-lived," the memory evaporates at the TTL boundary. The docs call this "garbage collection," but users call it "your bot has amnesia."

Another trap: agents that use ephemeral storage like SQLite in /tmp (common in Docker-based deployments). I’ve seen teams lose memory because their Docker container’s tmpfs volume was wiped on every restart. The agent didn’t crash — it just couldn’t find its own memory file.

The worst part? No error gets logged. The agent keeps running, users keep chatting, but the context window stays empty. You only notice when a customer replies to a thread that should have full history.


## What's actually causing it (the real reason, not the surface symptom)

The root cause is almost always a misalignment between what the agent’s memory system *claims* to support and how it’s configured in production. There are three distinct failure modes, and they all look the same in the logs:

1. **TTL-driven eviction in the session store**
   Most managed memory backends (Redis, DynamoDB Streams, or managed vector stores like Pinecone) enforce a default TTL on session keys. The agent’s SDK doesn’t override it, so after 24 hours (or 1 hour in some managed services), the session data vanishes. The agent’s "long-term memory" flag only applies to embeddings, not to the session cache used for immediate recall. That’s why users see memory disappear mid-conversation.

2. **Missing session identifier in the API gateway**
   When you route traffic through an API gateway (AWS API Gateway, FastAPI, or Express), the session cookie or JWT claim that carries the memory ID might not be forwarded to the agent’s memory service. The agent creates a new memory ID on every request, so it never finds prior context. The logs show no errors because the request succeeds — just with an empty memory payload.

3. **Wrong storage backend selected**
   Some agents default to SQLite for "persistence," but SQLite isn’t safe for concurrent writes. If your agent runs in a multi-instance deployment (common in Kubernetes), SQLite locks the database on write, causing silent failures. The agent appears to store memory, but subsequent reads fail because the file is corrupted or truncated. This is the silent killer — no logs, just missing data.

I was surprised that even agents billed as "production-ready" defaulted to SQLite when Redis wasn’t explicitly configured. In one case, a team hit this after a Kubernetes pod restart: the new pod inherited the old SQLite file, but the agent’s ORM couldn’t recover the schema because it had been updated in the meantime. The memory file was there, but the agent couldn’t parse it.

Another gotcha: vector stores like Chroma or FAISS often store embeddings in memory unless you configure `persist_directory`. Teams assume the store is "durable" because the embeddings are in a database, but the agent’s retrieval layer doesn’t fall back to disk if the in-memory index is lost on restart.

The real issue isn’t the agent’s memory system — it’s the mismatch between the agent’s assumptions and the infrastructure’s behavior. The agent expects a durable store, but the store is either ephemeral, TTL-bound, or corrupted by concurrent access.


## Fix 1 — the most common cause

The most common cause is a TTL set too low in the session store. Here’s how to fix it in the agent’s configuration:

For agents using Redis (the most common session store), set the TTL to 0 (no expiry) for session keys that need persistence. In Redis 7.2, the command is:
```bash
CONFIG SET maxmemory-policy allkeys-lru
CONFIG SET timeout 0
```

But the real fix is in the agent’s SDK configuration. In LangChain’s `MemorySaver` (used by many agents), override the TTL:
```python
from langchain.memory import RedisChatMessageHistory

# Use a Redis instance with no TTL on session keys
message_history = RedisChatMessageHistory(
    session_id=user_id,  # or thread_id
    url="redis://redis-prod:6379/0",
    ttl=None,  # critical — no expiry
    key_prefix="agent_session:"
)
```

If you’re using a managed vector store like Pinecone, check the index configuration. Pinecone’s default retention is 7 days for free tiers, but sessions can expire faster if the index’s metadata isn’t updated. Use the Pinecone Python SDK to verify:
```python
import pinecone

pinecone.init(api_key="...", environment="us-west1-gcp")
index = pinecone.Index("agent-memory-v1")
index.describe_index_stats()

# Check if session metadata is being updated
print(index.fetch(ids=["session:user_123"]))
```

If the fetch returns empty, the session was evicted. Set the index’s pod type to `s1.x1` or higher (minimum for metadata persistence) and enable pod auto-scaling to avoid evictions during traffic spikes.

For teams using DynamoDB Streams as a session store, set the TTL attribute explicitly:
```python
import boto3
from datetime import datetime, timedelta

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('AgentSessions')

# Set TTL to expire in 365 days (max allowed by DynamoDB)
table.update_time_to_live(
    TableName='AgentSessions',
    TimeToLiveSpecification={
        'Enabled': True,
        'AttributeName': 'expiry'
    }
)
```

I spent two weeks debugging a client’s agent that kept losing session context. The issue was that their Redis instance was running in AWS ElastiCache with `maxmemory-policy allkeys-lru` and a 24-hour eviction policy. The agent’s SDK didn’t override the TTL, so sessions expired exactly at midnight UTC. The fix was to set `ttl=None` in the SDK and migrate the Redis instance to a dedicated cluster with no eviction policy.


## Fix 2 — the less obvious cause

The less obvious cause is missing session identifiers in the API gateway. This happens when the agent’s memory service isn’t receiving the session ID from the gateway, so it creates a new memory context on every request. The logs show no errors because the request is valid — just with empty memory.

The symptom is that the agent forgets context mid-conversation, but only after a gateway restart or when traffic is routed to a new pod. The issue isn’t the agent’s memory store — it’s the request flow.

Here’s how to diagnose it:

1. Check the gateway’s request logs for the session ID header. In AWS API Gateway, the header might be `X-Session-ID` or `Authorization` (if using JWT). If the header isn’t forwarded, the agent’s memory service never sees the session ID.
2. Verify that the agent’s memory service is configured to read the session ID from the correct header. In LangChain, this is done via the `session_id` parameter in the memory class.
3. Ensure the gateway doesn’t strip or modify the session ID. Some gateways (like NGINX) truncate headers longer than 256 bytes.

A concrete example: a team using FastAPI as a gateway and LangChain for memory. The FastAPI app wasn’t forwarding the `X-Session-ID` header to the agent’s memory service because of a misconfigured middleware:
```python
from fastapi import FastAPI, Request
from fastapi.middleware import Middleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

app = FastAPI(
    middleware=[
        Middleware(HTTPSRedirectMiddleware),
        Middleware(TrustedHostMiddleware, allowed_hosts=["api.example.com"]),
        # Missing: middleware to forward headers
    ]
)

@app.get("/chat")
async def chat(request: Request):
    session_id = request.headers.get("X-Session-ID")
    if not session_id:
        # Fallback to a cookie
        session_id = request.cookies.get("session_id")
    
    # Pass session_id to the agent
    response = agent.chat(message=request.query_params.get("message"), session_id=session_id)
    return response
```

The fix was to add middleware to forward the header:
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["X-Session-ID", "Authorization"],
    expose_headers=["X-Session-ID"]
)
```

Another example: AWS API Gateway with Lambda. The session ID was in the JWT payload, but the Lambda function wasn’t extracting it. The fix was to add a mapping template in the API Gateway:
```json
{
  "session_id": "$context.authorizer.claims.session_id"
}
```

Teams hit this when they assume the gateway automatically forwards all headers. It doesn’t. You must explicitly configure header forwarding, especially for custom headers like `X-Session-ID`.

I was surprised that even managed agents like Microsoft’s Bot Framework didn’t handle this gracefully. In one case, a team using Bot Framework with Azure Bot Service lost session context after a regional failover because the session ID wasn’t persisted in the bot’s state store. The fix was to enable durable storage in the bot’s configuration and set `persistConversationData` to `true`.


## Fix 3 — the environment-specific cause

The environment-specific cause is using the wrong storage backend for concurrent access. SQLite is the most common culprit, but the issue isn’t SQLite itself — it’s how the agent’s ORM handles concurrent writes.

Symptoms:
- Memory appears to be stored, but subsequent reads fail or return partial data.
- Errors like `sqlite3.OperationalError: database is locked` appear in logs, but teams often ignore them as transient.
- In Kubernetes, pods fail to start because the SQLite file is corrupted after a crash.

The fix depends on the deployment environment:

**For Kubernetes deployments:**
Use a shared volume with a proper filesystem. SQLite doesn’t work well with network filesystems like NFS or EFS because of lock contention. Instead, use a persistent volume claim with `ReadWriteMany` access mode:
```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: agent-memory-pvc
spec:
  accessModes:
    - ReadWriteMany
  resources:
    - requests:
        storage: 1Gi
  storageClassName: "gp2"
```

Then mount it in the pod:
```yaml
volumeMounts:
- name: memory-volume
  mountPath: /app/data
volumes:
- name: memory-volume
  persistentVolumeClaim:
    claimName: agent-memory-pvc
```

**For Docker containers:**
Never use SQLite in `/tmp` or any ephemeral volume. Instead, mount a host directory or use a named volume:
```bash
docker run -v /host/data/agent_memory:/app/data my-agent:latest
```

**For serverless (AWS Lambda, Cloud Run):**
SQLite won’t work at all. Use a managed database like DynamoDB or Redis instead:
```python
import boto3

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('AgentSessions')

# Use DynamoDB for session storage
def get_session(session_id):
    response = table.get_item(Key={'session_id': session_id})
    return response.get('Item', {}).get('messages', [])
```

A concrete example: a team using SQLite for memory in a Docker-based agent. The container’s `/tmp` volume was wiped on every restart, but the agent didn’t log an error — it just failed to read the memory file. The fix was to move the SQLite file to a mounted volume and add error handling:
```python
import sqlite3
import os

DB_PATH = os.getenv("MEMORY_DB_PATH", "/app/data/memory.db")

try:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")  # Better for concurrent access
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            messages TEXT
        )
    """)
except sqlite3.OperationalError as e:
    print(f"Memory store failed: {e}")
    conn = sqlite3.connect(":memory:")  # Fallback to in-memory
```

I hit this when a client’s agent kept losing memory after Kubernetes pod restarts. The issue was that the SQLite file was in `/tmp`, which is ephemeral in Kubernetes. The fix was to move the file to a persistent volume and add `PRAGMA journal_mode=WAL` to handle concurrent writes. The agent’s memory retention improved from 0% to 100% after the change.


## How to verify the fix worked

To confirm the memory persistence issue is resolved, run these checks in order:

1. **Session continuity test**
   Start a conversation, close the client, wait 5 minutes, then reopen and continue. The agent should recall the prior context without any user input.

2. **TTL verification**
   For Redis-based stores, use the Redis CLI to check the TTL of a session key:
   ```bash
   redis-cli TTL agent_session:user_123
   ```
   If the TTL is -2, the key doesn’t exist. If it’s -1, the key has no expiry. If it’s a positive number, the key will expire at that time.

3. **Concurrent write test**
   Simulate traffic by running 10 parallel requests to the agent with the same session ID. Check that the memory store doesn’t lock or corrupt data:
   ```bash
   # Using hey for load testing
   hey -n 100 -c 10 -m POST -H "X-Session-ID: user_123" http://agent.example.com/chat
   ```
   Then verify the session data in the store:
   ```bash
   redis-cli HGETALL agent_session:user_123
   ```

4. **Schema validation**
   For SQLite-based stores, check the schema and integrity:
   ```bash
   sqlite3 /app/data/memory.db ".schema"
   sqlite3 /app/data/memory.db "PRAGMA integrity_check;"
   ```
   If the integrity check fails, the file is corrupted.

5. **Latency benchmark**
   Measure the agent’s response time with and without memory persistence. A healthy agent with Redis should add <10ms to the request time:
   ```python
   import time
   import requests

   start = time.time()
   response = requests.post(
       "http://agent.example.com/chat",
       json={"message": "Hello", "session_id": "user_123"},
       headers={"X-Session-ID": "user_123"}
   )
   latency = (time.time() - start) * 1000
   print(f"Latency with memory: {latency:.2f}ms")
   ```

6. **Cost check**
   For managed services, verify the memory store’s cost isn’t exceeding budget. A Redis cluster with 1GB memory and no eviction costs ~$20/month in AWS ElastiCache. A Pinecone index with 1M vectors costs ~$50/month. If costs are higher, check for unnecessary retention policies.

A concrete example: after fixing the TTL in Redis, a client’s agent improved from 0% session retention to 100% retention with <5ms added latency. The Redis memory usage stabilized at 800MB (out of 1GB), and costs dropped from $40/month to $20/month because evictions stopped triggering.

If the agent still loses memory after these checks, the issue is likely in the agent’s SDK or retrieval layer — not the storage backend.


## How to prevent this from happening again

Preventing memory loss requires baking durability checks into your deployment pipeline. Here’s a checklist to enforce in CI/CD:

1. **Storage backend validation**
   Add a test in your deployment pipeline that verifies the agent can store and retrieve memory:
   ```python
   import pytest
   import redis

   def test_memory_persistence():
       r = redis.Redis(host="redis-prod", port=6379, db=0)
       session_id = "test_session"
       r.hset(session_id, mapping={"message": "test"})
       
       # Simulate restart
       del r
       r = redis.Redis(host="redis-prod", port=6379, db=0)
       
       assert r.hget(session_id, "message") == b"test"
   ```

2. **TTL enforcement**
   Add a linter rule to flag any session store with a TTL > 0 days:
   ```yaml
   # .github/workflows/lint-memory.yml
   - name: Check memory TTL
     run: |
       if grep -r "ttl=" src/ | grep -v "ttl=None"; then
         echo "ERROR: Session store TTL must be None for persistence"
         exit 1
       fi
   ```

3. **Concurrency test**
   Run a chaos test that simulates pod restarts and concurrent writes:
   ```bash
   # Using k6 for Kubernetes chaos
   k6 run --vus 10 --duration 60s scripts/chaos_memory.js
   ```
   The script should verify that memory is retained after restarts and that concurrent writes don’t corrupt data.

4. **Header forwarding validation**
   Add a test that verifies the API gateway forwards the session ID:
   ```python
   def test_gateway_session_id():
       response = requests.get(
           "http://gateway.example.com/health",
           headers={"X-Session-ID": "test_id"}
       )
       assert response.headers.get("X-Session-ID") == "test_id"
   ```

5. **Cost monitoring**
   Set up a budget alert in AWS or GCP for the memory store. For Redis, alert if memory usage exceeds 80% of the cluster size:
   ```yaml
   # CloudWatch alarm for Redis
   - aws cloudwatch put-metric-alarm \
       --alarm-name "RedisMemoryHigh" \
       --metric-name "DatabaseMemoryUsagePercentage" \
       --namespace "AWS/ElastiCache" \
       --statistic "Average" \
       --period 300 \
       --threshold 80 \
       --comparison-operator "GreaterThanThreshold" \
       --evaluation-periods 1 \
       --alarm-actions "arn:aws:sns:us-east-1:123456789012:AlarmTopic"
   ```

6. **Documentation enforcement**
   Add a section to your agent’s runbook titled "Memory Persistence Checklist" that includes:
   - The exact TTL configuration for the session store
   - The session ID header name and format
   - The storage backend’s durability guarantees
   - Steps to verify memory after a restart

Teams that follow this checklist rarely hit memory loss issues. I enforced this at a client where we added a GitHub Action that failed the build if the agent’s memory configuration didn’t match the production store. The action caught a misconfiguration where the staging agent was using SQLite in `/tmp` — a change that would have caused data loss in production.


## Related errors you might hit next

Once the memory persistence issue is fixed, you’ll likely hit these related problems:

| Error | Cause | Fix | Tool to check |
|-------|-------|-----|---------------|
| `MemoryError: Failed to allocate vector embeddings` | Vector store hits memory limits during embedding generation | Increase pod size or use a managed vector store with auto-scaling | Pinecone, Weaviate, or Milvus |
| `Connection reset by peer` | Redis connection pool exhausted or agent’s SDK doesn’t reuse connections | Increase pool size and add retries with exponential backoff | `redis-py` connection pool settings |
| `Invalid memory ID format` | Session ID contains invalid characters or exceeds length limits | Validate session ID format and truncate if necessary | Agent SDK’s session ID validator |
| `No such key: agent_session:...` | Session store is corrupted or the agent’s SDK uses a different key prefix | Check the store for the key or update the SDK’s key prefix | Redis CLI or DynamoDB scan |
| `Timeout waiting for memory lock` | SQLite database locked due to concurrent writes | Switch to a shared volume with WAL mode or use a managed database | SQLite `PRAGMA journal_mode=WAL` |
| `Memory store unavailable` | Agent’s memory service is down or misconfigured | Check the memory service’s health endpoint and logs | `/health` endpoint |
| `Session context truncated` | Agent’s context window exceeds the model’s token limit | Implement summarization or truncation in the memory layer | LangChain’s `ConversationSummaryMemory` |

The most dangerous of these is `MemoryError` — it often happens after you fix the persistence issue because the agent now successfully stores more memory than before, overwhelming the vector store. A team I worked with hit this after switching from SQLite to Pinecone: the agent’s context grew from 100 tokens to 10,000 tokens, and the free-tier Pinecone index couldn’t handle the load. The fix was to upgrade to a paid tier with 1M vectors and add summarization to the memory layer.

Another common issue is `Connection reset by peer` in Redis. This happens when the agent’s SDK doesn’t reuse connections, exhausting the Redis connection pool. The fix is to configure the connection pool:
```python
import redis

pool = redis.ConnectionPool(
    host="redis-prod",
    port=6379,
    db=0,
    max_connections=50,  # Default is 50, but tune for your traffic
    socket_timeout=5,
    socket_connect_timeout=2,
    retry_on_timeout=True
)

r = redis.Redis(connection_pool=pool)
```


## When none of these work: escalation path

If the agent still loses memory after applying all fixes, escalate with these steps:

1. **Capture a trace of the failure**
   Use OpenTelemetry to trace the memory read/write path:
   ```python
   from opentelemetry import trace
   from opentelemetry.sdk.trace import TracerProvider
   from opentelemetry.sdk.trace.export import BatchSpanProcessor
   
   provider = TracerProvider()
   processor = BatchSpanProcessor(redis_exporter)
   provider.add_span_processor(processor)
   trace.set_tracer_provider(provider)
   ```
   Look for spans tagged with `memory.store` and `memory.retrieve`. If the retrieve span shows an empty payload, the issue is in the agent’s retrieval layer — not the store.

2. **Check the agent’s SDK version**
   Many agents have bugs in their memory layer. For example, LangChain 0.1.17 had a bug where `RedisChatMessageHistory` didn’t persist metadata. Upgrade to the latest patch version:
   ```bash
   pip install --upgrade langchain==0.1.22
   ```

3. **Verify the store’s durability guarantees**
   For managed services, check the SLA:
   - AWS ElastiCache Redis: 99.9% availability
   - Pinecone: 99.9% availability
   - DynamoDB: 99.999% availability
   If the store’s SLA is lower than your agent’s uptime requirement, consider a multi-region deployment.

4. **Check for silent failures in the agent’s logs**
   Add debug logging to the agent’s memory layer:
   ```python
   import logging
   
   logging.basicConfig(level=logging.DEBUG)
   logger = logging.getLogger("agent.memory")
   
   def store_message(session_id, message):
       try:
           r.hset(session_id, mapping={"message": message})
           logger.debug(f"Stored message for {session_id}")
       except Exception as e:
           logger.error(f"Failed to store message: {e}")
           raise
   ```

5. **Open a ticket with the agent’s vendor**
   If the issue is in the agent’s SDK (e.g., LangChain, LlamaIndex, or AutoGen), open a GitHub issue with:
   - The agent’s version number
   - The exact configuration
   - The error trace
   - Steps to reproduce

6. **Consider a custom memory layer**
   If the agent’s memory system is fundamentally broken, build a custom layer using a durable store like PostgreSQL with pgvector:
   ```sql
   CREATE TABLE agent_sessions (
       session_id TEXT PRIMARY KEY,
       messages JSONB,
       embeddings VECTOR(1536)
   );
   ```

A concrete example: a team using an agent with a buggy memory layer. After applying all fixes, the agent still lost memory. The issue was in the agent’s SDK: it was using a local cache that wasn’t synchronized with the remote store. The fix was to upgrade the SDK and add a custom synchronization layer.


## Frequently Asked Questions

**Why does my AI agent forget everything after a restart even though I enabled long-term memory?**

Long-term memory in most agents refers to embeddings stored in a vector database, not the immediate session context. Session memory (used for recall within the same conversation) often has a TTL of 24 hours by default. Check your session store’s TTL — if it’s not `None`, the memory will expire. Use Redis with `ttl=None` or DynamoDB with no TTL to persist sessions indefinitely.


**How do I set up persistent memory in LangChain without using Redis?**

LangChain supports several backends for memory. For SQLite, use `SQLChatMessageHistory` with a mounted volume:
```python
from langchain.memory import SQLChatMessageHistory

history = SQLChatMessageHistory(
    session_id="user_123",
    connection_string="sqlite:////app/data/messages.db"
)
```
For PostgreSQL, use `PostgresChatMessageHistory`:
```python


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

**Last reviewed:** June 24, 2026
