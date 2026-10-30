# IDP for AI: 2026 stack evolution

A local guide usually assumes a clean environment and a patient timeline. The edge cases only appear once real traffic hits the system. This article is about one specific edge case: **keeping an internal developer platform (IDP) in sync with embedding models whose output shape changes**.

The failure mode is well documented in teams that ship AI features. A model update changes the vector dimension — say from 768 to 1024. The staging embeddings job runs, fails quietly because the index mapping does not auto-resize, and the next deploy ships a 768-dimension endpoint against a 1024-dimension index. The error surfaces as an index size mismatch deep in the retrieval path, long after the deploy went green. Teams fix it by hand, push a hot patch, and promise to automate it later. The next model update repeats the cycle.

The fix is not clever. It is to bake the embedding pipeline into the platform so that a model change triggers a rebuild through the same deploy path as everything else, with validation that fails loudly at the boundary. This article walks through that design, the code, and the failure modes it does and does not remove.

## The shape of the problem

Three properties of embedding models make them awkward platform citizens:

1. **They change shape, not just weights.** A dimension change is a schema change for every downstream store: vector indexes, caches, and any serialized vectors in object storage.
2. **Rebuilds are expensive and slow.** Regenerating embeddings for a large corpus is a batch job, not a request-time operation, and it competes with serving traffic.
3. **The failure is silent until it isn't.** A mismatched index often accepts writes and only fails at query time, or fails on a subset of documents.

A useful mental model: treat the embedding model like a database schema with a version number. Every artifact that stores vectors carries that version. Anything that reads or writes vectors asserts the version matches. When the version changes, a migration runs — and that migration is a first-class platform resource, not a shell script someone remembers to run.

## Reference stack

The stack below is one reasonable choice among several. Each component is described by category so you can substitute your own.

- Node.js 20 LTS — application runtime
- A Node HTTP framework (any will do; the routing code is trivial)
- Redis 7.2 — write-through vector cache and job coordination
- AWS Lambda on arm64 — background embedding and index rebuild compute
- Pulumi — infrastructure as code, so the rebuild job is a versioned resource
- OpenSearch with k-NN vector support — vector store

None of these are load-bearing for the design. The same shape works with pgvector instead of OpenSearch, a container job instead of Lambda, and Terraform instead of Pulumi.

## Prerequisites

You need a working IDP that already deploys a Node.js service to your cloud of choice. If you are starting from scratch, the minimal setup is:

1. One cloud account with a sandbox VPC and private subnets.
2. A repository with an infrastructure-as-code stack that creates a container service running Node.js 20 LTS.
3. IAM roles that let the pipeline push images and update the service.
4. A vector store with k-NN search enabled.

What you will build:

- An embedding endpoint (`POST /embeddings`) backed by a write-through cache.
- A background job that rebuilds the vector index when the model changes.
- A platform resource that publishes the new index version so the next deploy picks it up.
- Validation at three boundaries: request, index, and cache.

## Step 1 — model the vector index as a platform resource

The first change is conceptual. Instead of a script that creates an index, define a component that owns the index, its dimension, and its model version together.

```typescript
// vector-infra.ts
import *pulumi* as pulumi from "@pulumi/pulumi";
import * as aws from "@pulumi/aws";

export interface VectorInfraArgs {
  indexName: string;
  dimension: number;
  modelVersion: string;
}

export class VectorInfra extends pulumi.ComponentResource {
  public readonly cacheEndpoint: pulumi.Output<string>;
  public readonly refreshJobArn: pulumi.Output<string>;

  constructor(name: string, args: VectorInfraArgs, opts?: pulumi.ComponentResourceOptions) {
    super("custom:vector:infra", name, {}, opts);

    const redisSubnetGroup = new aws.elasticache.SubnetGroup("redisSubnetGroup", {
      subnetIds: pulumi.output(aws.ec2.getSubnetIds({})).apply(s => s.ids),
    }, { parent: this });

    const redis = new aws.elasticache.Cluster("embeddingCache", {
      engine: "redis",
      engineVersion: "7.2",
      nodeType: "cache.t3.micro",
      numCacheNodes: 2,
      subnetGroupName: redisSubnetGroup.name,
      securityGroupIds: [/* your security group */],
    }, { parent: this });

    const lambdaRole = new aws.iam.Role("lambdaRole", {
      assumeRolePolicy: aws.iam.assumeRolePolicyForPrincipal({ Service: "lambda.amazonaws.com" }),
    }, { parent: this });

    new aws.iam.RolePolicy("lambdaPolicy", {
      role: lambdaRole.id,
      policy: pulumi.output(aws.iam.getPolicyDocument({
        statements: [
          {
            actions: ["es:ESHttpPost", "es:ESHttpPut"],
            resources: ["arn:aws:es:*:*:domain/*"],
          },
          {
            actions: ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
            resources: ["*"],
          },
        ],
      }).then(doc => doc.json)),
    }, { parent: this });

    const refreshJob = new aws.lambda.Function("refreshIndex", {
      runtime: "python3.11",
      handler: "refresh_index.handler",
      role: lambdaRole.arn,
      code: new pulumi.asset.AssetArchive({
        "refresh_index.py": new pulumi.asset.StringAsset(`
import os
import boto3

def handler(event, context):
    client = boto3.client('opensearchserverless')
    index = os.environ['INDEX_NAME']
    dimension = int(os.environ['DIMENSION'])
    client.create_index(
        index=index,
        mapping={"properties": {"embedding": {"type": "knn_vector", "dimension": dimension}}},
    )
    return {"status": "ok", "index": index, "dimension": dimension}
`),
      }),
      memorySize: 512,
      timeout: 300,
      architectures: ["arm64"],
      environment: {
        variables: {
          INDEX_NAME: args.indexName,
          DIMENSION: String(args.dimension),
          MODEL_VERSION: args.modelVersion,
        },
      },
    }, { parent: this });

    this.cacheEndpoint = pulumi.interpolate`${redis.cacheNodes[0].address}:6379`;
    this.refreshJobArn = refreshJob.arn;

    this.registerOutputs({
      cacheEndpoint: this.cacheEndpoint,
      refreshJobArn: this.refreshJobArn,
    });
  }
}
```

The important detail is not the resource shapes; it is that `indexName`, `dimension`, and `modelVersion` are constructor arguments. Changing the model version in the stack definition changes the index name and dimension in one place, and the diff is reviewable like any other infrastructure change.

A note on the Lambda body: the real rebuild job will be larger than this. It needs to read the corpus, call the embedding model in batches, write to the new index, and then flip an alias. The skeleton above only creates the index. Keep the rebuild logic in the same repository as the stack so the job and the schema move together.

## Step 2 — the embedding service

The service has two responsibilities: serve embeddings with a cache, and expose a control endpoint that a model registry can call. Keep the control endpoint behind the same auth as your other internal endpoints; it is not a public API.

```bash
npm install redis
```

```typescript
// src/vector.ts
import { createClient } from 'redis';

const redis = createClient({
  url: process.env.REDIS_URL!,
  socket: { reconnectStrategy: (retries) => Math.min(retries * 100, 5000) },
});

export const MODEL_VERSION = process.env.MODEL_VERSION ?? 'v1';
export const MODEL_DIMENSION = Number(process.env.MODEL_DIMENSION ?? 768);

export async function embed(texts: string[]): Promise<number[][]> {
  const vectors = await Promise.all(texts.map(async (text) => {
    const cacheKey = `embeddings:${MODEL_VERSION}:${text}`;
    const cached = await redis.json.get(cacheKey);
    if (cached) return cached as number[];

    // Replace with a real call to your embedding model.
    const vector = Array(MODEL_DIMENSION).fill(0.1);
    validateDimension(vector);

    await redis.json.set(cacheKey, '$', vector);
    await redis.expire(cacheKey, 3600);
    return vector;
  }));
  return vectors;
}

export function validateDimension(vector: number[]): void {
  if (vector.length !== MODEL_DIMENSION) {
    throw new Error(
      `Dimension mismatch: expected ${MODEL_DIMENSION}, got ${vector.length}`
    );
  }
}
```

Two details matter more than they look:

- The cache key includes `MODEL_VERSION`. Without it, a model update silently serves vectors from the old model. Including the version makes old entries unreachable rather than wrong, and the TTL reclaims the space.
- `validateDimension` runs before the cache write and before the response. It converts a downstream index error into a request-time error with a clear message.

The HTTP layer is intentionally small:

```typescript
// src/server.ts
import fastify from 'fastify';
import { embed } from './vector';

const app = fastify({ logger: true });

app.post('/embeddings', async (req, reply) => {
  const { texts } = req.body as { texts: string[] };
  if (!Array.isArray(texts) || texts.length === 0) {
    return reply.code(400).send({ error: 'texts must be a non-empty array' });
  }
  const vectors = await embed(texts);
  reply.send({ vectors, modelVersion: process.env.MODEL_VERSION });
});

app.get('/healthz', async (_req, reply) => reply.send({ ok: true }));

app.listen({ port: 3000, host: '0.0.0.0' });
```

Note the response includes `modelVersion`. Clients that store vectors alongside metadata can record which model produced them, which makes later migrations auditable.

## Step 3 — validate the index, not just the request

Request-time validation catches the case where the model returns the wrong shape. It does not catch the case where the index was created with the wrong dimension. Add an explicit check at service startup and in the rebuild job.

```typescript
// src/opensearch.ts
import { Client } from '@opensearch-project/opensearch';

const client = new Client({ node: process.env.OPENSEARCH_ENDPOINT });

export async function ensureIndex(index: string, dim: number): Promise<void> {
  const exists = await client.indices.exists({ index });
  if (exists.body) {
    const mapping = await client.indices.getMapping({ index });
    const currentDim = mapping.body[index]?.mappings?.properties?.embedding?.dimension;
    if (currentDim !== dim) {
      throw new Error(
        `Index ${index} has dimension ${currentDim}, expected ${dim}. ` +
        `Create a new index and reindex instead of mutating this one.`
      );
    }
    return;
  }

  await client.indices.create({
    index,
    body: {
      settings: { index: { knn: true, 'knn.algo_param.ef_search': 100 } },
      mappings: {
        properties: {
          embedding: { type: 'knn_vector', dimension: dim },
        },
      },
    },
  });
}
```

The error message is deliberate. A dimension mismatch on an existing index is not something to fix in place — the correct response is to create a new index with a new name and reindex. Making the error say so removes the temptation to patch the mapping.

## Step 4 — make the rebuild idempotent

The most common production failure in this design is not a wrong dimension; it is two rebuild jobs running at once. A model registry that retries on timeout, combined with a job that takes minutes, produces duplicate indexes and, worse, partially written ones.

The fix is a lock with a TTL, held in the same Redis instance used for caching:

```python
# refresh_index.py
import os
import time
import boto3
import redis

LOCK_KEY = "vector:rebuild:lock"
LOCK_TTL_SECONDS = 900

def handler(event, context):
    r = redis.Redis.from_url(os.environ["REDIS_URL"])
    acquired = r.set(LOCK_KEY, context.aws_request_id, nx=True, ex=LOCK_TTL_SECONDS)
    if not acquired:
        return {"status": "skipped", "reason": "rebuild already in progress"}

    try:
        index = os.environ["INDEX_NAME"]
        dimension = int(os.environ["DIMENSION"])
        client = boto3.client("opensearchserverless")
        client.create_index(
            index=index,
            mapping={"properties": {"embedding": {"type": "knn_vector", "dimension": dimension}}},
        )
        return {"status": "ok", "index": index, "dimension": dimension}
    finally:
        r.delete(LOCK_KEY)
```

Two properties make this safe enough: the lock is acquired with `nx` and `ex`, so it is atomic and self-healing if the job dies; and the job deletes the lock in a `finally` block, so a successful run does not block the next one. The TTL is a backstop, not the primary mechanism.

The remaining gap: if the job runs longer than the TTL, a second job can start. Size the TTL comfortably above the worst-case rebuild time and alert when a rebuild exceeds half the TTL.

## Step 5 — observability that answers the right question

Three signals are worth instrumenting, and they are worth choosing carefully because each one answers a specific question:

| Signal | Question it answers | How to collect it |
| --- | --- | --- |
| Embedding request latency, labeled by `modelVersion` | Did a model change make serving slower? | Histogram in your metrics system, incremented in the request handler |
| Index dimension vs. configured dimension | Is the running service pointed at the wrong index? | Startup check that emits a gauge and exits non-zero on mismatch |
| Rebuild job duration and outcome | How long does a model migration take, and does it succeed? | Job start/end timestamps and a success/failure counter |

The labeling detail matters. Without `modelVersion` on the latency histogram, a p99 regression after a model update is unattributable. With it, the answer is a single query.

To measure the rebuild cost before committing to a design, instrument the job itself: log the number of documents processed, the wall-clock time, and the total tokens sent to the embedding model. Run it once against a staging copy of the corpus and compare the projected cost against your budget. This is a measurement you can do in an afternoon and it will tell you more than any published benchmark, because the numbers depend entirely on your corpus size, your batch size, and your model's per-token price.

## A worked example: the dimension change, step by step

To make the failure mode concrete, here is the sequence with a 768-to-1024 change, assuming the design above is in place.

1. The model registry updates the stack configuration: `dimension: 1024`, `modelVersion: v2`, `indexName: embeddings-v2`.
2. `pulumi preview` shows a new index resource and a new Lambda environment. A reviewer sees the dimension change in the diff.
3. The deploy creates the new index with 1024 dimensions. The old index is untouched.
4. The service restarts with `MODEL_DIMENSION=1024` and `MODEL_VERSION=v2`. Its startup check confirms the index it points at has 1024 dimensions.
5. The rebuild job runs, reading the corpus, calling the v2 model, and writing to `embeddings-v2`. The lock prevents a concurrent run.
6. Once the rebuild completes and a spot check passes, traffic is switched to the new index and the old one is retained for a rollback window.
7. Cache entries under `embeddings:v2:*` are written fresh; `embeddings:v1:*` entries expire on their own.

The property that makes this work is that the new index is created before the old one is removed, and the switch is a separate, reversible step. Compare that with mutating the existing index in place: there is no rollback, and the window between the mapping change and the reindex completion serves wrong results.

## Decision checklist

Before adopting this pattern, answer these questions. If any answer is unclear, that is the part of the design that needs work.

- **Can you rebuild the index without serving stale results?** If not, you need a dual-index or alias-based switch, not an in-place update.
- **Do you know your worst-case rebuild time?** If not, measure it against a staging corpus before you rely on a lock TTL.
- **Is the model version recorded with every stored vector?** If not, a future migration cannot tell which vectors need regenerating.
- **Does a dimension mismatch fail at startup or at query time?** Startup is strictly better; query-time failures are partial and hard to reproduce.
- **Is the rebuild job idempotent?** If running it twice creates two indexes, a retry will eventually do exactly that.
- **Can you roll back a model change without a data migration?** If the answer is no, the rollback plan is "restore from backup," which is a different conversation.

## Common questions

### What if the model weights change but the dimension stays the same?

No index rebuild is required. The vectors are the same shape, so the existing index accepts them. The stale-vector problem is real, though: cached and indexed vectors from the old weights are not comparable with new ones. The clean approach is to treat a weights change as a new `modelVersion` even when the dimension is unchanged, which gives you the same dual-index switch. If that is too expensive, at minimum version the cache key and re-embed on read.

### How do I handle several models with different dimensions?

One index per model, with the model version in the index name and in the cache key prefix. Route by path or by a header, and keep the routing table in configuration rather than code. The cost is one more index per model; the benefit is that a change to one model cannot break another.

### Can this run outside AWS?

Yes. Every component maps to a category: the container runtime, the cache, the background job, the vector store, and the infrastructure-as-code tool. Substituting pgvector for a dedicated vector store removes one moving part at the cost of index performance at large scale — a reasonable trade for a small corpus. The validation and versioning logic is unchanged.

### What is the simplest starting point?

A single Postgres instance with pgvector, the embedding service as a container, and a cron-triggered rebuild job. Skip the cache until you have measured that it is needed. The versioning and validation logic described above is what prevents incidents; the cache is a performance optimization and can be added later.

## Next step

Open your infrastructure-as-code stack and add the `modelVersion` and `dimension` values as explicit inputs to whatever resource creates your vector index. Then add a startup check in your embedding service that compares the configured dimension against the index mapping and exits non-zero on mismatch. That is a small change — likely under an hour — and it converts the most common silent failure in this pipeline into a loud one at deploy time.
