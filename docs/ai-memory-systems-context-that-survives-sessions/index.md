# AI memory systems: context that survives sessions

## The symptom: identical prompts, different answers

An agent answers a question sensibly, the process is restarted, and the same question now produces a different answer. No exception is raised, no error line appears in the logs, and the health check still reports healthy. The behavior looks like model flakiness, which is why debugging often starts in the wrong layer.

The confusion comes from the word "memory." In most tutorials, memory is a variable in the current process: a list of messages appended to the prompt. That is working memory, and it disappears when the process exits. Persistence is a separate concern that requires explicit writes to storage, explicit reads on startup, and a stable key that connects the two.

A typical failure mode is a system that writes memory correctly but reads it back without a scope. The write path stores entries tagged with a user or conversation, the read path queries the most recent entries regardless of owner, and after a restart the agent loads whatever is newest in the shared store. The result is not an error but a plausible-looking answer built from the wrong context.

This failure is easy to miss in development because the process rarely restarts, the dataset is small, and a single user owns every entry. In production the same code runs across restarts, deploys, and multiple concurrent users, and the missing scope becomes visible.

## Root cause 1: memory written without a session scope

The most common defect is an unscoped memory store. The write includes a session identifier, but the read does not filter on it, so the query returns the most recent entries in the collection rather than the current session's entries.

The fix is to make the session identifier a required part of both the write and the read. With a vector store that supports metadata filters, the write looks like this:

```python
import weaviate
from datetime import datetime, timezone

client = weaviate.Client("https://your-cluster.weaviate.network")

client.data_object.create(
    data_object={
        "content": "User asked about pricing",
        "session_id": "user_123_conversation_456",
        "created_at": datetime.now(timezone.utc).isoformat(),
    },
    class_name="AgentMemory",
    vector=embedding,  # your embedding vector
)
```

The matching read must carry the same filter:

```python
response = client.graphql_raw_query(f"""
{{
  Get {{
    AgentMemory(
      where: {{
        path: ["session_id"],
        operator: Equal,
        valueString: "user_123_conversation_456"
      }}
    ) {{
      content
      created_at
    }}
  }}
}}
""")
```

With a key-value store the same discipline applies through key naming. A key prefix that includes the session identifier makes the scope structural rather than optional:

```python
import redis

r = redis.Redis(host="your-redis.internal", port=6379, decode_responses=True)

r.json().set(
    "agent:memory:user_123_conversation_456:action_789",
    ".",
    {"content": "User reviewed plan", "timestamp": "2026-06-05T14:30:00Z"},
)

keys = r.keys("agent:memory:user_123_conversation_456:*")
memory = [r.json().get(key) for key in keys]
```

Two details matter here. First, `KEYS` performs a full scan and blocks the server on large keyspaces; in production use `SCAN` with a cursor or maintain an index set per session. Second, the session identifier must be stable. A random UUID generated at process start produces a new scope on every restart, so memory never accumulates. Derive the identifier from something durable, such as `user_id` plus a conversation identifier that outlives the process.

A useful design rule: the storage layer should have no API that reads memory without a session argument. If the interface makes an unscoped read impossible to express, the bug class disappears.

## Root cause 2: credentials that expire while the process runs

A second failure mode produces empty memory rather than wrong memory. The agent connects successfully at startup, writes and reads for a while, and then silently stops seeing its own entries. Logs show a successful connection because the connection was successful — at the time it was made.

The cause is a credential that expires mid-run: a short-lived API key, a rotating secret, or temporary cloud credentials issued to a role. The client library holds the old credential, the storage layer rejects requests, and depending on the client, the rejection surfaces as an empty result rather than an exception.

The fix has two parts. First, wrap storage calls so that authentication failures trigger a credential refresh and a bounded retry:

```python
import os
import time
from functools import wraps

def refresh_on_auth_failure(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        for attempt in range(3):
            try:
                return func(*args, **kwargs)
            except AuthError as exc:  # your client's auth exception type
                if attempt == 2:
                    raise
                reinit_client(os.environ["MEMORY_API_KEY"])
                time.sleep(2 ** attempt)
        raise RuntimeError("unreachable")
    return wrapper
```

Second, make credential refresh part of the process lifecycle rather than a one-time setup step. A common pattern is a sidecar that refreshes the credential and exposes it on a local endpoint, so the agent fetches a fresh value instead of holding a long-lived one. On managed cloud platforms, prefer workload identity or instance roles over static keys, since the platform refreshes those credentials for you.

The diagnostic signature to watch for is a time-correlated failure: memory works for the first minutes or hours after deploy and then degrades. Instrument the age of the credential used on each storage call and alert when it approaches its expiry.

## Root cause 3: storage that is not actually persistent

A third failure mode is storage that behaves like persistence in development and like a scratchpad in production. Container filesystems are ephemeral, and `/tmp` is cleared between serverless invocations. An agent that writes memory to `/tmp/agent_memory.json` appears to work until the first restart.

The same trap appears in Kubernetes when a volume is mounted at one path and the application writes to another. A pod spec that mounts an `emptyDir` at `/tmp` while the agent writes to `/app/tmp` will lose data on restart with no error, because both paths behave exactly as configured — one of them is simply ephemeral.

For a single-writer agent, a persistent volume claim mounted at the application's data path is sufficient:

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: ai-agent
spec:
  serviceName: "ai-agent"
  replicas: 1
  selector:
    matchLabels:
      app: ai-agent
  template:
    spec:
      containers:
        - name: agent
          image: your-ai-agent:latest
          volumeMounts:
            - name: memory-volume
              mountPath: /app/persistent-memory
      volumes:
        - name: memory-volume
          persistentVolumeClaim:
            claimName: agent-memory-pvc
```

The agent must be configured to write to `/app/persistent-memory`, not `/tmp`. For serverless runtimes, use a managed storage service instead of the local filesystem, and choose based on access pattern: object storage for durable archives, a network filesystem or managed cache for low-latency reads.

A related trap is configuration injected as an environment variable. Environment variables are read once at process start. If a secret is rotated and the pod is not restarted, the process keeps using the old value indefinitely. Either watch the secret and trigger a rollout on change, or have the process read the current value from a mounted volume or a local endpoint on each connection attempt.

## A worked example: tracing a silent memory loss

Consider an agent that stores conversation summaries in a vector collection. The write path is:

```python
def save_summary(session_id, summary, embedding):
    client.data_object.create(
        data_object={
            "content": summary,
            "session_id": session_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        class_name="AgentMemory",
        vector=embedding,
    )
```

The read path, written early in development, is:

```python
def load_recent(limit=10):
    return client.query.get("AgentMemory", ["content"]).with_limit(limit).do()
```

Now trace what happens. During development, one user, one session, no restarts: `load_recent` returns that user's entries and everything looks correct. In production, three things change. Multiple users write to the same collection. The process restarts on every deploy. And the collection accumulates entries from all sessions.

After a restart, `load_recent` returns the ten most recent entries across all users. If another user was active more recently, the agent loads that user's context. The answer it produces is fluent and wrong. Nothing in the logs indicates a problem, because every call succeeded.

The fix is not to add a filter at the call site but to remove the unscoped function entirely:

```python
def load_recent(session_id, limit=10):
    return (
        client.query.get("AgentMemory", ["content"])
        .with_where({
            "path": ["session_id"],
            "operator": "Equal",
            "valueString": session_id,
        })
        .with_limit(limit)
        .do()
    )
```

The reasoning generalizes: when a defect is caused by a missing constraint, the durable fix is to make the constraint unrepresentable to omit. Deleting the unscoped function is stronger than remembering to filter at each call site.

## How to measure whether memory actually persists

Verification should be automated, not manual. The core test is a restart cycle with a unique marker.

```python
import subprocess
import time
import uuid
import requests

def start_agent():
    return subprocess.Popen(
        ["python", "agent.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

def test_memory_persistence():
    marker = f"marker-{uuid.uuid4()}"
    session_id = "test-session-fixed"

    proc = start_agent()
    time.sleep(5)

    requests.post(
        "http://localhost:8000/remember",
        json={"session_id": session_id, "content": marker},
        timeout=10,
    )

    proc.kill()
    proc.wait()
    time.sleep(2)

    proc = start_agent()
    time.sleep(5)

    response = requests.post(
        "http://localhost:8000/recall",
        json={"session_id": session_id},
        timeout=10,
    )

    assert marker in response.text, "memory did not survive restart"
    proc.kill()

test_memory_persistence()
```

Use a unique marker per run so a passing test cannot be explained by stale data. Run this in CI against a real storage backend, not an in-memory fake, since the fake is exactly what hides the bug.

For production, a synthetic canary extends the same idea. Periodically write a marker under a dedicated session, wait longer than the credential lifetime, and read it back. If the read fails while the write succeeded, the storage layer or credential path is at fault. This catches expiry and rotation failures that application logs do not surface.

To inspect the storage layer directly, count entries scoped to a session before and after a restart:

```bash
redis-cli --scan --pattern "agent:memory:user_123_conversation_456:*" | wc -l
```

A decreasing count after restart means the data is not durable. A stable count with failing reads means the scope or credential is wrong. Separating those two cases is the fastest way to localize the fault.

## Choosing a storage layer

| Option | Access pattern | Persistence caveat | Operational cost |
|---|---|---|---|
| Managed vector store | Semantic search across sessions | Metadata filters must be indexed; auth tokens may expire | Managed, per-query pricing |
| Key-value store with JSON support | Low-latency per-session reads and writes | In-memory by default; requires persistence configuration | Self-managed or managed |
| Relational database | Structured queries, transactional writes | Requires schema and connection management | Well-understood, widely supported |
| Local files | Prototyping only | Ephemeral in containers and serverless | Lowest until it fails |

The choice follows the query pattern, not popularity. Semantic retrieval across sessions needs a vector index with a filtered search path. Session-scoped key-value access is served well by a cache with persistence enabled. Structured history with joins and transactions belongs in a relational database. Local files are acceptable only where the filesystem is known to be durable and single-writer.

One sizing note worth stating explicitly: any store that answers queries by scanning an ever-growing collection will slow down as the collection grows. Retention policy is part of the design. Keep a bounded window per session, or summarize older entries into a compact record, rather than appending indefinitely.

## Prevention checklist

- Define a memory interface with no unscoped reads: `save(session_id, content)` and `load(session_id)`.
- Derive session identifiers from durable context, never from a process-local random value.
- Store credentials in a form that refreshes, and instrument credential age on each storage call.
- Mount persistent storage at the exact path the application writes to, and verify the path in a staging restart test.
- Run a restart persistence test in CI against real storage.
- Run a production canary that writes a marker, waits past the credential lifetime, and reads it back.
- Set a retention window per session so the store does not grow without bound.
- Document where memory lives, how long credentials last, and what happens when storage is unavailable.

## Escalation path when memory still does not persist

1. Query the storage layer directly, outside the agent, before and after a restart. This separates "data is gone" from "agent cannot see data."
2. Confirm the identity in use. On AWS, `aws sts get-caller-identity`; on GCP, `gcloud auth list`. Verify the identity has read and write permissions on the specific resource.
3. Inspect pod events with `kubectl describe pod <pod-name>` for `FailedMount` or `Evicted`, which point at storage or resource problems rather than application logic.
4. Reduce the agent to a minimal save/load program. If the minimal version persists, the fault is in agent logic; if not, it is in the environment.
5. Check for in-flight writes lost during restart. A write that has not been acknowledged before the process exits is lost unless the client uses atomic or transactional operations.
6. Check storage quotas and collection limits. A store at its size limit can reject writes while returning empty reads.

## FAQ

**Why does an agent lose memory after restart even when a cache like Redis is in use?**

Redis is in-memory by default. Data survives a client restart but not a server restart unless persistence is configured. Verify that persistence is enabled on the instance the client actually connects to, and confirm the session keys are stable across restarts.

**How should session identifiers behave when users log in and out?**

Tie memory to a conversation identifier that outlives the login session. Store that identifier in the authentication token and reuse it on subsequent logins. Generating a new identifier per login resets the agent's history for a returning user.

**What is the smallest change that fixes most memory loss?**

Make the session identifier deterministic and make it required on every read. Replace process-local random identifiers with a value derived from user and conversation context, and remove any storage function that can read without a session scope.

**Does memory need to be summarized or can it grow indefinitely?**

Unbounded growth degrades query latency and increases cost. Apply a retention window per session, or periodically summarize older entries into a compact record, so the active set stays bounded.

## Do this in the next 30 minutes

Open the module that reads memory in your agent and search for any query that does not take a session identifier. If one exists, add the identifier to the function signature, add the filter to the query, and delete the unscoped version so it cannot be called again. Then write a single test that saves a unique marker under a fixed session, restarts the process, and asserts the marker is still readable. That test is the difference between memory that works in a demo and memory that survives a deploy.
