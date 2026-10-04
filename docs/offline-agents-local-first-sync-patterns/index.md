# Offline Agents: Local-First Sync Patterns

## The failure mode this article addresses

A recurring failure mode in offline-capable agent systems is quiet: no exceptions are raised, no alerts fire, and yet the data the agent sees is wrong. It usually traces back to an application built with a connected-first mindset, where offline support is retrofitted later. The retrofit introduces duplicate writes, lost updates, and inconsistent state between the device and the backend.

The harder problem is not making an API call idempotent. It is rethinking data flow, storage, and synchronization around a local-first model. When the data includes personal information — for example, records about EU residents under GDPR, or records collected in the field that later reside in EU data centers — data residency, the right to erasure, and audit trails must be part of the architecture from the start rather than bolted on.

This article covers the synchronization layer itself: a client-side operation log, a versioned conflict-resolution strategy, and an auditable sync endpoint. The examples use Python 3.11, FastAPI, and AWS services, but the patterns apply to any stack with a local database and a central service.

## Prerequisites and system shape

The reader should be comfortable with Python 3.11, asynchronous programming, and client-side database concepts. The worked example focuses on the backend sync logic; the mobile client is described conceptually.

The architecture has four parts:

1. **Client (agent device).** A mobile app writing to an encrypted local database (for example, SQLite with SQLCipher, or a mobile-native database). The database stores an append-only log of `operations` (`CREATE`, `UPDATE`, `DELETE`) rather than only the latest state. The operation log is what makes synchronization and conflict resolution tractable.
2. **Backend sync service.** A FastAPI application that receives operation batches, resolves conflicts, persists to a primary store, and writes audit records.
3. **Data store.** A transactional store such as DynamoDB for entity records and version history, and object storage such as S3 for large binaries (photos, scanned documents).
4. **Audit trail.** Application-level audit logging plus cloud provider audit services that capture data mutations and access attempts.

The goal: an agent may operate offline for days, and when it reconnects, its data is encrypted at rest, eventually consistent with the central system, and fully auditable. The relevant GDPR obligations are the processing principles in Article 5, records of processing activities in Article 30, security of processing in Article 32, and the right to erasure in Article 17. This article describes the engineering patterns; it is not legal advice, and the compliance mapping should be reviewed by counsel.

## Step 1 — Configure the environment

Configure services with security and data residency in mind, not just dependency installation.

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install fastapi uvicorn 'boto3>=1.34.0' 'pydantic>=2.0.0' 'pendulum>=2.1.2'
```

For workloads that must keep personal data within the EU, provision the primary data stores in an EU region, for example `eu-central-1` (Frankfurt) or `eu-west-1` (Dublin). Confirm this against the actual contractual and regulatory requirements for the workload; residency obligations vary by data category and by the legal basis for processing.

The DynamoDB table below stores entity records. The primary key is `entity_id`, and the sort key is `version`, which makes it possible to query the latest version of an entity and to retain prior versions for audit.

```python
# Conceptual setup; use AWS CLI, CloudFormation, or Terraform in practice.
import boto3

AWS_REGION = 'eu-central-1'

dynamodb = boto3.client('dynamodb', region_name=AWS_REGION)

def setup_aws_resources():
    try:
        dynamodb.create_table(
            TableName='AgentDataSync',
            KeySchema=[
                {'AttributeName': 'entity_id', 'KeyType': 'HASH'},
                {'AttributeName': 'version', 'KeyType': 'RANGE'}
            ],
            AttributeDefinitions=[
                {'AttributeName': 'entity_id', 'AttributeType': 'S'},
                {'AttributeName': 'version', 'AttributeType': 'N'}
            ],
            BillingMode='PAY_PER_REQUEST'
        )
        print("Table 'AgentDataSync' created.")
    except dynamodb.exceptions.ResourceInUseException:
        print("Table 'AgentDataSync' already exists.")

# For S3, enforce EU residency via bucket configuration and enable default
# encryption (SSE-S3 or SSE-KMS) at bucket creation time.
# setup_aws_resources()
```

**Failure mode:** `boto3` clients fall back to the region configured in the environment when `region_name` is omitted. A single client constructed without an explicit region can write data outside the intended jurisdiction. Construct every client with an explicit `region_name`, and add a configuration test that asserts the region on each client.

## Step 2 — Core implementation

The sync endpoint accepts a batch of operations from an agent. Each operation carries the entity it targets, the operation type, the payload, a client-side timestamp, and the client's version of the entity at the time the operation was created. The server applies operations, resolves conflicts, and returns a new sync timestamp plus any server-side changes the client must apply.

The conflict-resolution strategy here is last-write-wins (LWW) by version number: the server keeps a monotonically increasing `version` per entity, and an operation whose `client_version` is behind the server's current version is rejected and the server's record is returned to the client. LWW is sufficient for many field-operation workloads but loses concurrent edits. Workloads that require merging concurrent edits should use CRDTs or an explicit merge function rather than LWW.

```python
# app/main.py
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import boto3
import pendulum
from typing import List, Literal, Optional, Dict, Any

app = FastAPI()

AWS_REGION = 'eu-central-1'
dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)
table = dynamodb.Table('AgentDataSync')

class AgentOperation(BaseModel):
    operation_id: str = Field(..., description="Unique ID for this operation on the client")
    entity_id: str = Field(..., description="ID of the entity being operated on")
    entity_type: str = Field(..., description="Type of entity (e.g., 'user', 'delivery')")
    operation_type: Literal['CREATE', 'UPDATE', 'DELETE']
    payload: Optional[Dict[str, Any]] = None
    timestamp: pendulum.DateTime = Field(..., description="Timestamp of the operation on the client")
    client_version: int = Field(..., description="Version of the entity on the client at operation time")

class SyncRequest(BaseModel):
    agent_id: str
    last_sync_timestamp: pendulum.DateTime
    operations: List[AgentOperation]

class ServerUpdate(BaseModel):
    entity_id: str
    entity_type: str
    data: Optional[Dict[str, Any]] = None
    version: int
    deleted: Optional[bool] = False

class SyncResponse(BaseModel):
    new_last_sync_timestamp: pendulum.DateTime
    server_updates: List[ServerUpdate]
    server_deletions: List[Dict[str, str]] = []

@app.post("/sync", response_model=SyncResponse)
async def sync_data(request: SyncRequest):
    server_updates_to_client: List[ServerUpdate] = []
    server_deletions_to_client: List[Dict[str, str]] = []
    current_server_time = pendulum.now('UTC')

    for op in request.operations:
        # Audit log before processing, so rejected operations are still recorded.
        print(
            f"AUDIT_LOG: Agent {request.agent_id} performed {op.operation_type} "
            f"on {op.entity_type}/{op.entity_id} at {op.timestamp} "
            f"(client_version: {op.client_version})"
        )

        try:
            response = table.query(
                KeyConditionExpression='entity_id = :eid',
                ExpressionAttributeValues={':eid': op.entity_id},
                ScanIndexForward=False,  # latest version first
                Limit=1
            )
            latest_server_record = response['Items'][0] if response['Items'] else None
            server_version = latest_server_record.get('version', 0) if latest_server_record else 0
            server_data = latest_server_record.get('data', {}) if latest_server_record else {}

            if op.client_version < server_version:
                server_updates_to_client.append(ServerUpdate(
                    entity_id=op.entity_id,
                    entity_type=op.entity_type,
                    data=server_data,
                    version=server_version,
                    deleted=latest_server_record.get('deleted', False)
                ))
                print(
                    f"Conflict: client version {op.client_version} for {op.entity_id} "
                    f"is older than server version {server_version}; skipping client op."
                )
                continue

            new_version = server_version + 1
            item_to_put = {
                'entity_id': op.entity_id,
                'version': new_version,
                'updated_at_server': current_server_time.isoformat(),
                'updated_at_client': op.timestamp.isoformat(),
                'agent_id': request.agent_id,
                'operation_type': op.operation_type,
                'operation_id': op.operation_id,
            }

            if op.operation_type in ('CREATE', 'UPDATE'):
                merged_data = {**server_data, **(op.payload or {})}
                item_to_put['data'] = merged_data
            elif op.operation_type == 'DELETE':
                item_to_put['deleted'] = True
                item_to_put['data'] = {}

            table.put_item(Item=item_to_put)
            print(f"Processed {op.operation_type} for {op.entity_id}, new version {new_version}")

        except Exception as e:
            # Do not halt the batch on one bad operation; route it for later review.
            print(f"ERROR processing operation {op.operation_id} for {op.entity_id}: {e}")

    # Server-initiated deletions are looked up from a separate pending-deletions
    # store populated when an erasure request is received. Placeholder below.
    if pendulum.now('UTC').day == 15:
        server_deletions_to_client.append({'entity_id': 'user-123', 'entity_type': 'user'})

    return SyncResponse(
        new_last_sync_timestamp=current_server_time,
        server_updates=server_updates_to_client,
        server_deletions=server_deletions_to_client
    )

# Run with: uvicorn app.main:app --reload --port 8000
```

Two details matter for correctness. First, the audit log is written before conflict resolution, so rejected operations remain in the record. Second, the client's `timestamp` is preserved alongside the server's `updated_at_server`, because device clocks drift and the true ordering of events is often only recoverable from the server's timestamp.

### A worked conflict example

Suppose an entity `delivery-42` has server version 5. Two agents edit it while offline:

- Agent A reads version 5, edits the delivery address, and produces operation `op-A` with `client_version = 5`.
- Agent B reads version 5, edits the delivery notes, and produces operation `op-B` with `client_version = 5`.

Agent A syncs first. The server sees `client_version = 5` and `server_version = 5`, accepts the operation, and writes version 6 with the merged address. Agent B syncs next with `client_version = 5`. The server now has `server_version = 6`, so `5 < 6`, and the operation is rejected. The server returns its version 6 record to Agent B, and Agent B's client must surface the conflict to the user or re-apply the notes on top of version 6. The edit to the notes is not lost, but it is not automatically merged either. This is the LWW trade-off, and it is worth documenting explicitly for the client team so that the user experience around conflicts is deliberate.

## Step 3 — Edge cases: media, deletion, and network behavior

### Media references

Storing large binaries in DynamoDB is inefficient and expensive. Upload media to object storage and store only the object keys in the entity record. The client is responsible for uploading media before or alongside the operation that references it, and for retrying uploads when connectivity is poor.

```python
class AgentOperation(BaseModel):
    # ... existing fields ...
    media_references: Optional[List[str]] = None  # object storage keys
```

```python
if op.media_references:
    item_to_put['media_references'] = op.media_references
```

### Deletion propagation

The right to erasure requires that a deletion request propagates to all systems, including devices that may be offline for weeks. The `DELETE` operation handles client-initiated deletion; server-initiated deletion requires a separate mechanism. A workable pattern is a pending-deletions store that is populated when an erasure request is received, queried by the sync endpoint, and returned to the client as `server_deletions`. The client must then physically purge the entity and any associated media from local storage, and confirm the purge on the next sync.

The placeholder below simulates a pending deletion. In production, replace the day-of-month check with a query against the pending-deletions store, keyed by agent or by entity.

```python
if pendulum.now('UTC').day == 15:
    server_deletions_to_client.append({'entity_id': 'user-123', 'entity_type': 'user'})
```

Two failure modes are common here. First, the client deletes the record but not the media, leaving personal data in local object storage. Second, the client deletes the record but the server's audit log still contains the payload, which may conflict with the erasure obligation unless the audit log is designed to store references rather than personal data. Both should be addressed in the client purge routine and in the audit schema.

### Encryption and transport

Data at rest on the device should be encrypted (for example, SQLCipher for SQLite), and object storage should use server-side encryption. All transport should use TLS. These are baseline measures under Article 32, and they should be verified by configuration tests rather than assumed.

### Network behavior

Sync latency is dominated by payload size and connection quality. A batch of 50 operations with 5 KB payloads over a slow mobile connection will take noticeably longer than the same batch over Wi-Fi, and media uploads add several seconds per object. Rather than guessing at a budget, measure it: instrument the client to record the time from sync start to sync completion, the number of operations in the batch, the total payload bytes, and the number of media uploads. Compare the resulting distribution across connection types. This produces a real budget for the workload instead of an assumed one.

## Step 4 — Observability and testing

Offline-capable systems cannot rely solely on real-time client metrics, because the client is often unreachable. The backend should emit structured logs for every sync request, including the agent ID, the number of operations, the number of conflicts, and the number of server-initiated deletions. CloudWatch can ingest these logs and derive metrics such as sync batch size and conflict count.

On the client, a health-check routine that reports local database status, last successful sync time, and the count of un-synced operations is valuable. That report should be queued and sent on the next successful sync so that the backend has a complete picture.

For measurement, the key instrument is a counter per sync outcome: accepted, rejected as stale, and errored. Track the ratio of rejected-as-stale operations to accepted operations over time. A rising ratio indicates that agents are holding stale data for longer, which may point to longer offline periods, slower sync cadence, or a conflict-resolution strategy that is no longer appropriate for the workload.

### Testing strategy

1. **Unit tests** for the conflict-resolution logic, covering equal versions, client ahead, client behind, and delete-then-update sequences.
2. **Integration tests** against the `/sync` endpoint covering: first sync with empty client and empty server; client-only changes; server-only changes; concurrent updates where the client version is behind; and delete operations followed by a re-sync to confirm purge instructions are returned.
3. **Failure injection**: simulate network failures mid-batch and confirm the client retries the same operation IDs without duplicating writes. Idempotency on `operation_id` is what makes this safe.
4. **Clock skew tests**: run the client with a clock offset and confirm the server's timestamps remain authoritative for ordering.

## Decision checklist

Before shipping an offline-capable agent, confirm:

- The client stores an operation log, not just final state.
- Every operation has a stable `operation_id` and the server is idempotent on it.
- The conflict-resolution strategy is chosen deliberately (LWW, CRDT, or explicit merge) and its user-visible consequences are documented.
- Audit logging happens before conflict resolution, and the audit schema is compatible with erasure obligations.
- Deletion propagates to devices, including media, and is confirmed on the next sync.
- Data at rest and in transit is encrypted, and region configuration is asserted by tests.
- Sync outcome counters exist per outcome, and the stale-rejection ratio is monitored.

## Next 30 minutes

Open the sync endpoint and add a counter for rejected-as-stale operations, incremented each time `op.client_version < server_version` is true. Emit it as a structured log line alongside the agent ID and entity ID. That single counter turns the quiet failure mode — wrong answers with no errors — into something visible, and it is the cheapest instrumentation to add before the next field deployment.
