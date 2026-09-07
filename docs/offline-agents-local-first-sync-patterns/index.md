# Offline Agents: Local-First Sync Patterns

After reviewing enough code that touches offlinecapable agent, the same failure pattern keeps showing up. Here's the version I wish someone had handed me first. The failure is quiet — no errors, just wrong answers.

## Why I wrote this (the problem I kept hitting)

Working with teams serving European users, especially those expanding into regions like Africa for field operations or last-mile delivery, I've repeatedly seen a critical oversight: building an application with a 'connected-first' mindset, then scrambling to bolt on offline capabilities. This approach invariably leads to compliance headaches, data loss incidents, and a poor user experience for agents operating in areas with intermittent or non-existent internet access. The core issue isn't just about making an API call idempotent; it's about fundamentally rethinking data flow, storage, and synchronization from a local-first perspective. When you're dealing with personal data, especially for EU citizens under GDPR, or even data collected from African users that might eventually reside in EU data centers, compliance isn't an afterthought. Data residency, the right to be forgotten, and comprehensive audit trails must be baked into the architecture from day one. The part that trips people up is building a robust, compliant local-first data synchronization layer that handles conflicts and ensures data integrity, and that's what this post actually covers.

## Prerequisites and what you'll build

This tutorial assumes you're comfortable with Python 3.11, asynchronous programming, and have some familiarity with client-side database concepts. We'll build a simplified offline-capable agent workflow focusing on data collection and synchronization. Our field agents will use a mobile application (conceptualized, but we'll focus on the backend sync logic) that stores data locally using SQLite. This local data will then synchronize with a central backend service built with FastAPI, using AWS services for storage and processing. The critical components will be a robust client-side data model, a conflict-resolution strategy, and a secure, auditable synchronization mechanism.

The architecture will comprise:

1.  **Client-side (Agent Device)**: A conceptual mobile app saving data to an encrypted SQLite database. This database will store `operations` (e.g., `create`, `update`, `delete`) rather than just the final state, enabling robust synchronization.
2.  **Backend Sync Service**: A FastAPI application responsible for receiving data from agents, resolving conflicts, persisting to a primary database (e.g., AWS DynamoDB), and ensuring audit trails.
3.  **Data Store**: AWS DynamoDB for transactional data and AWS S3 for larger binary objects (e.g., photos).
4.  **Audit Trail**: AWS CloudTrail and custom logging to capture all data mutations and access attempts.

Our goal is to ensure that even if an agent operates offline for days, their data is secure, eventually consistent with the central system, and fully auditable, meeting GDPR's Article 5 (principles relating to processing of personal data), Article 30 (records of processing activities), and Article 32 (security of processing) requirements.

## Step 1 — set up the environment

Setting up a compliant environment means more than just installing dependencies; it means configuring services with security and data residency in mind. For our backend, we'll use Python 3.11, FastAPI 0.110.0, and `boto3` for AWS interactions. On the client side, while we won't implement the mobile app, understand that a robust local database like SQLite (with SQLCipher for encryption) or Realm (if using a mobile framework) is essential. Data at rest on the agent's device *must* be encrypted. GDPR Article 32 mandates appropriate technical and organisational measures to ensure a level of security appropriate to the risk, including the pseudonymisation and encryption of personal data.

First, set up your Python environment:

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install fastapi uvicorn 'boto3>=1.34.0' 'pydantic>=2.0.0' 'pendulum>=2.1.2'
```

Next, configure your AWS environment. For European users, data residency means your primary data stores should be in an EU region (e.g., `eu-central-1` in Frankfurt or `eu-west-1` in Dublin). This is a non-negotiable requirement for many European clients. Create a DynamoDB table and an S3 bucket in your chosen EU region. The DynamoDB table will store our agent data and a synchronization log. Let's call the table `AgentDataSync`. It needs a primary key, say `entity_id`, and a sort key `version` to manage concurrent updates.

```python
# This is conceptual; you'd use AWS CLI or CloudFormation/Terraform for actual setup
import boto3

# Ensure this region is an EU region for GDPR compliance
AWS_REGION = 'eu-central-1'

dynamodb = boto3.client('dynamodb', region_name=AWS_REGION)
s3 = boto3.client('s3', region_name=AWS_REGION)

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

    # For S3, ensure bucket policies enforce EU residency and encryption at rest
    # Example: s3.create_bucket(Bucket='your-agent-media-bucket', CreateBucketConfiguration={'LocationConstraint': AWS_REGION})
    # And enable default encryption (SSE-S3 or KMS)

# Call this once to set up
# setup_aws_resources()
```

**Gotcha**: A common trap here is to forget that `boto3`'s default region might not be your desired EU region. Always explicitly specify `region_name` for all AWS service clients to prevent accidental data storage outside your compliance zone. A single misconfigured bucket or database could lead to non-compliance and hefty fines under GDPR, which can reach up to €20 million or 4% of annual global turnover, whichever is higher, as of 2026.

## Step 2 — core implementation

The core of our offline-capable workflow lies in a synchronization endpoint that handles incoming data batches from agents. Each data record from the client won't just be the final state but a series of "operations" (e.g., `create_user`, `update_user_address`, `delete_user`). This approach is crucial for conflict resolution and auditability. The agent's device will maintain a `last_sync_timestamp` and send all operations that occurred after that timestamp. The server will then apply these operations, resolve any conflicts, and return a new `last_sync_timestamp` and any server-side changes to the client.

We'll use a simple versioning scheme for conflict resolution: the highest version number wins. More complex scenarios might require CRDTs (Conflict-free Replicated Data Types), but for many field operations, a last-write-wins (LWW) with versioning is sufficient. We also need to ensure every incoming operation is logged for audit purposes, regardless of whether it results in a data change.

```python
# app/main.py
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import boto3
import pendulum
from typing import List, Literal, Optional, Dict, Any

app = FastAPI()

# Ensure this region is an EU region for GDPR compliance
AWS_REGION = 'eu-central-1'
dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)
table = dynamodb.Table('AgentDataSync')

class AgentOperation(BaseModel):
    operation_id: str = Field(..., description="Unique ID for this operation on the client")
    entity_id: str = Field(..., description="ID of the entity being operated on")
    entity_type: str = Field(..., description="Type of entity (e.g., 'user', 'delivery')")
    operation_type: Literal['CREATE', 'UPDATE', 'DELETE']
    payload: Optional[Dict[str, Any]] = None # Data for create/update
    timestamp: pendulum.DateTime = Field(..., description="Timestamp of the operation on the client")
    client_version: int = Field(..., description="Version of the entity on the client at operation time")

class SyncRequest(BaseModel):
    agent_id: str
    last_sync_timestamp: pendulum.DateTime
    operations: List[AgentOperation]

class SyncResponse(BaseModel):
    new_last_sync_timestamp: pendulum.DateTime
    server_updates: List[Dict[str, Any]] # Server-side changes to push to client

@app.post("/sync", response_model=SyncResponse)
async def sync_data(request: SyncRequest):
    server_updates = []
    current_server_time = pendulum.now('UTC')

    for op in request.operations:
        # 1. Log the operation for audit trail (before processing)
        # This could go to CloudWatch Logs, a dedicated audit table, or both.
        # GDPR Article 30 requires records of processing activities.
        print(f"AUDIT_LOG: Agent {request.agent_id} performed {op.operation_type} on {op.entity_type}/{op.entity_id} at {op.timestamp} (client_version: {op.client_version})")

        try:
            # Get current server state for conflict resolution
            response = table.query(
                KeyConditionExpression='entity_id = :eid',
                ExpressionAttributeValues={':eid': op.entity_id},
                ScanIndexForward=False, # Get latest version first
                Limit=1
            )
            latest_server_record = response['Items'][0] if response['Items'] else None
            server_version = latest_server_record.get('version', 0) if latest_server_record else 0
            server_data = latest_server_record.get('data', {}) if latest_server_record else {}

            # Conflict Resolution: Last-Write-Wins based on version
            if op.client_version < server_version:
                # Server has a newer version, client's operation is outdated
                # We might push the server's version back to the client
                server_updates.append({
                    'entity_id': op.entity_id,
                    'entity_type': op.entity_type,
                    'data': server_data,
                    'version': server_version
                })
                print(f"Conflict: Client version {op.client_version} for {op.entity_id} is older than server version {server_version}. Skipping client op.")
                continue # Skip processing this client operation

            new_version = server_version + 1 # Increment server version for new write
            item_to_put = {
                'entity_id': op.entity_id,
                'version': new_version,
                'updated_at_server': current_server_time.isoformat(),
                'updated_at_client': op.timestamp.isoformat(),
                'agent_id': request.agent_id,
                'operation_type': op.operation_type,
                'operation_id': op.operation_id,
            }

            if op.operation_type == 'CREATE' or op.operation_type == 'UPDATE':
                # Merge client payload with existing server data, if any
                merged_data = {**server_data, **(op.payload or {})}
                item_to_put['data'] = merged_data
            elif op.operation_type == 'DELETE':
                item_to_put['deleted'] = True
                item_to_put['data'] = {} # Clear data on delete

            table.put_item(Item=item_to_put)
            print(f"Processed {op.operation_type} for {op.entity_id}, new version {new_version}")

        except Exception as e:
            # Log detailed error, but don't halt the entire sync if possible
            print(f"ERROR processing operation {op.operation_id} for {op.entity_id}: {e}")
            # Consider a dead-letter queue for failed operations for later review.

    # Fetch any server-side changes that occurred since client's last_sync_timestamp
    # This is a simplified example; a real system would query more intelligently.
    # For instance, using a global secondary index on 'updated_at_server'.
    # For this tutorial, we'll assume no external server changes for simplicity in this step.

    return SyncResponse(
        new_last_sync_timestamp=current_server_time,
        server_updates=server_updates # Send back conflicts or server-initiated changes
    )

# To run: uvicorn app.main:app --reload --port 8000
```

This `sync_data` endpoint is the heart of the system. It receives a batch of operations, logs them for audit, applies conflict resolution, and updates the central store. The `client_version` is critical for LWW. If the client tries to update an entity with version 3, but the server already has version 5, the client's update is rejected, and the server's version 5 is pushed back to the client. This prevents data loss from stale client data. The `timestamp` on `AgentOperation` is also vital for understanding the true sequence of events, especially if device clocks drift.

## Step 3 — handle edge cases and errors

Robust offline synchronization demands careful handling of edge cases beyond simple conflict resolution. Data integrity, security, and GDPR compliance are paramount. One significant edge case is managing large media files (e.g., photos, documents) collected by agents. Storing these directly in DynamoDB is inefficient and costly. Instead, they should be uploaded to AWS S3, with only references (S3 object keys) stored in DynamoDB. This also means robust client-side retry logic for S3 uploads, which can be flaky in low-bandwidth environments.

Another critical edge case is data deletion. GDPR's "right to be forgotten" (Article 17) means that when a user requests data deletion, it must propagate to all systems, including offline devices. Our `DELETE` operation type is a start, but a server-initiated data purge mechanism is also needed. When a deletion request comes in for a specific `entity_id`, the server must mark it for deletion, and during the next sync, instruct all connected agents to purge that data from their local storage. This is complex because an agent might be offline for an extended period.

Consider the following additions to our `SyncRequest` and `SyncResponse` models to handle server-initiated deletions and media references:

```python
# app/main.py (modifications)
# ... (existing imports and classes)

class AgentOperation(BaseModel):
    # ... (existing fields)
    media_references: Optional[List[str]] = None # S3 object keys for media

class ServerUpdate(BaseModel):
    entity_id: str
    entity_type: str
    data: Optional[Dict[str, Any]] = None
    version: int
    deleted: Optional[bool] = False # Indicates server-side deletion

class SyncResponse(BaseModel):
    new_last_sync_timestamp: pendulum.DateTime
    server_updates: List[ServerUpdate] # Use the new ServerUpdate model
    server_deletions: List[Dict[str, str]] = [] # Explicit instructions for client to delete

@app.post("/sync", response_model=SyncResponse)
async def sync_data(request: SyncRequest):
    server_updates_to_client = [] # Renamed for clarity
    server_deletions_to_client = []
    current_server_time = pendulum.now('UTC')

    for op in request.operations:
        # ... (audit logging as before)

        try:
            # ... (fetch latest_server_record as before)
            # ... (conflict resolution as before)

            if op.client_version < server_version:
                # Server has a newer version, client's operation is outdated
                server_updates_to_client.append(ServerUpdate(
                    entity_id=op.entity_id,
                    entity_type=op.entity_type,
                    data=server_data,
                    version=server_version,
                    deleted=latest_server_record.get('deleted', False)
                ))
                print(f"Conflict: Client version {op.client_version} for {op.entity_id} is older than server version {server_version}. Skipping client op.")
                continue

            # ... (new_version, item_to_put as before)

            if op.media_references:
                item_to_put['media_references'] = op.media_references

            table.put_item(Item=item_to_put)
            print(f"Processed {op.operation_type} for {op.entity_id}, new version {new_version}")

        except Exception as e:
            print(f"ERROR processing operation {op.operation_id} for {op.entity_id}: {e}")
            # In a real system, you'd push this to a dead-letter queue (e.g., SQS) for async processing and alerting.
            # This prevents a single bad operation from blocking the entire sync batch.

    # Server-initiated deletions: Query a separate table or index for pending deletions
    # For example, a 'PendingDeletions' DynamoDB table where records are added when a GDPR deletion request comes in.
    # This would require a background process to populate 'PendingDeletions' and then for sync_data to query it.
    # For this tutorial, we'll simulate a pending deletion:
    if pendulum.now('UTC').day == 15: # Simulate a deletion instruction on the 15th of every month
        server_deletions_to_client.append({'entity_id': 'user-123', 'entity_type': 'user'})

    return SyncResponse(
        new_last_sync_timestamp=current_server_time,
        server_updates=server_updates_to_client,
        server_deletions=server_deletions_to_client
    )
```

When an agent receives `server_deletions`, their client application *must* physically purge that data from its encrypted local SQLite database. This ensures compliance with deletion requests. The client should also confirm deletion back to the server. Failure to implement this can lead to serious GDPR violations. Furthermore, all data, including media files, must be encrypted at rest (e.g., S3's Server-Side Encryption) and in transit (TLS 1.2+ for all communications). The typical latency for a batch sync of 50 operations with 5KB average payload size over a 3G connection in a remote African region could easily be 500-800ms, not accounting for S3 uploads which could add several seconds for media-heavy payloads. Optimizing network calls and payload sizes is crucial.

## Step 4 — add observability and tests

Observability for offline-capable systems is more complex than for always-connected ones. You can't rely solely on real-time metrics from the client. You need to track sync success rates, conflict rates, and data staleness. AWS CloudWatch can ingest logs from our FastAPI service, allowing us to monitor `AUDIT_LOG` messages, `ERROR` messages, and custom metrics like `sync_batch_size` or `conflict_resolved_count`. For client-side observability, a 'health check' endpoint on the mobile app that reports local database status, last successful sync, and un-synced operations when connectivity is available is invaluable. This data, when synced, provides a holistic view of the system's health.

**Testing Strategy:**

1.  **Unit Tests**: Standard tests for individual functions (e.g., conflict resolution logic).
2.  **Integration Tests**: Test the `/sync` endpoint with various scenarios:
    *   First sync (empty client, empty server).
    *   Client-only changes (no server conflicts).
    *   Server-only changes (client pulls updates).
    *   Concurrent client updates (client version < server version).
    *   Concurrent client updates (client version > server version – should not happen with proper client logic, but test for robustness).
    *   Simulated network failures (client retries).
    *


---

### About this article

**Written by:** Kubai Kevin — software developer based in Nairobi, Kenya, with 10+ years building production systems in fintech and AI.

**How this article was produced:** This site uses an automated LLM pipeline designed and maintained by the author. Topics are selected from real production experience. Drafts pass automated quality gates (minimum length, uniqueness, concrete metrics, versioned tools, code samples, absence of filler). Individual line-by-line human editing is not performed on every post before publication. Specific numbers, benchmarks and cost figures are illustrative; verify them against current official documentation before production use.

**Corrections:** Report errors via the contact page. Corrections are applied promptly.

**Last generated:** September 2026
