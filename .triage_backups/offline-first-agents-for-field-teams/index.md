# Offline-first agents for field teams

Most offline-capable agent guides assume a clean environment and a patient timeline. The problem they describe is easy to reproduce and hard to explain: a rider's device drops off the network mid-shift, the backend has no idea whether the rider is stuck or simply out of coverage, and the day's transactions end up reconstructed from chat messages and paper. This article lays out the tradeoffs that a tutorial usually skips.

## Why this problem keeps showing up

Offline-capable agents break most tutorials because those tutorials assume constant connectivity. Field teams in Nairobi, Accra, or Kampala move between areas with no signal, roam between towers, or hit deliberate network throttling. The real requirement is not just local caching. It is a state machine that can survive hours of disconnection, merge upstream changes when backhaul returns, and still give the rider a usable UI. Anything less and the team ships yesterday's failed deliveries today.

A typical failure mode: a team rolls out real-time tracking for a fleet of dispatch riders against a regionally distant SaaS. On a rainy-season afternoon, a large share of devices go offline for the better part of an hour. The central server cannot distinguish "stuck" from "no signal," so the dispatch queue freezes while staff rebuild it manually. A connection-pool issue that consumes three days of debugging is often a single misconfigured timeout in the WebSocket reconnect loop.

Teams commonly try three wrong paths:

- **Push everything to the edge device and call it done.** Devices fill up, battery drains, and drivers uninstall the app when it eats a large share of phone storage.
- **Assume the rider will remember to press "sync."** Human error erases transactions when the app finally reconnects.
- **Use a global SaaS with eventual consistency.** Data-residency rules can mean a rider's biometric data cannot be stored in another jurisdiction, so it has to stay on-device or in-country.

The shape that works is an offline-first agent that:

- runs a local state machine (no full database on the device)
- syncs in the background when connectivity returns
- keeps PII on-device or in-country
- presents a UI that never looks "offline"

## Prerequisites and what you'll build

You need Node.js 20 LTS and Python 3.11. The agent runs in a React Native shell for Android 13/14 devices, which remain common field hardware. The backend is a Lambda function (arm64, Node 20 runtime) behind an API Gateway, with S3 for media blobs in a region that satisfies your data-residency requirements.

The stack is intentionally minimal:

- SQLite with the bundled json1 extension for local state (no extra binaries)
- React Query 5.x for optimistic updates and background refetches
- AppSync for GraphQL subscriptions that survive reconnects
- Cognito with MFA and no SMS fallback (SMS costs are real and field teams dislike the flow)

What you'll have at the end:

- A rider-facing screen that shows "offline" status in the top bar but still lets them scan packages
- A background worker that queues network calls until connectivity returns
- A sync screen that shows progress and any conflicts
- A conflict-resolution UI that lets the rider choose which version to keep

Roughly 450 lines of TypeScript for the agent, plus about 200 lines for the Lambda resolver.

## Step 1 — set up the environment

### 1.1 Create the React Native shell

```bash
npx react-native init FieldAgent --version 0.72.6 --package-name "@acme/field-agent"
cd FieldAgent
```

Install the offline stack:

```bash
yarn add @react-navigation/native @react-navigation/stack react-query@5 sqlite3@5.1 react-native-sqlite-storage@6 aws-appsync@5 react-native-netinfo@11
```

Pin versions, because React Native libraries change frequently.

### 1.2 Configure SQLite on Android

Edit `android/app/build.gradle`:

```gradle
dependencies {
  implementation "com.facebook.react:react-native:+"
  implementation "net.sqlcipher:android-database-sqlcipher:4.5.3"
}
```

SQLCipher gives you 256-bit AES encryption so rider data isn't readable if the device is lost. An unencrypted SQLite bundle is too risky for PII.

### 1.3 Set up AWS resources

Deploy the backend once with CDK in TypeScript:

```bash
mkdir infra && cd infra
yarn init -y
yarn add aws-cdk-lib constructs
yarn add --dev ts-node
```

`bin/infra.ts`:

```typescript
import * as cdk from 'aws-cdk-lib';
import { FieldAgentStack } from '../lib/field-agent-stack';

const app = new cdk.App();
new FieldAgentStack(app, 'FieldAgentStack', {
  env: { region: 'eu-central-1' },
});
```

`lib/field-agent-stack.ts`:

```typescript
import * as cdk from 'aws-cdk-lib';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as apigateway from 'aws-cdk-lib/aws-apigateway';

export class FieldAgentStack extends cdk.Stack {
  constructor(scope: cdk.App, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    const handler = new lambda.Function(this, 'SyncHandler', {
      runtime: lambda.Runtime.NODEJS_20_X,
      code: lambda.Code.fromAsset('../backend/dist'),
      handler: 'sync.handler',
      memorySize: 512,
      timeout: cdk.Duration.seconds(25),
      environment: {
        BUCKET: 'field-agent-media',
        REGION: 'eu-central-1',
      },
    });

    new apigateway.RestApi(this, 'Api', {
      defaultCorsPreflightOptions: {
        allowOrigins: apigateway.Cors.ALL_ORIGINS,
        allowMethods: apigateway.Cors.ALL_METHODS,
      },
    }).root.addMethod('POST', new apigateway.LambdaIntegration(handler));
  }
}
```

Deploy:

```bash
cdk bootstrap
yarn cdk deploy --require-approval never
```

### 1.4 Add an AppSync subscription

A common dead end is trying to use MQTT over WebSockets with a custom broker, which can burn half a day before a team gives up. AppSync GraphQL subscriptions handle reconnect with exponential backoff and message batching, so use those instead. In `App.tsx`:

```typescript
import { ApolloClient, InMemoryCache, ApolloLink } from '@apollo/client';
import { createAuthLink } from 'aws-appsync-auth-link';
import { createSubscriptionHandshakeLink } from 'aws-appsync-subscription-link';
import { Auth } from 'aws-amplify';

const client = new ApolloClient({
  link: ApolloLink.from([
    createAuthLink({
      url: process.env.APPSYNC_ENDPOINT!,
      region: 'eu-central-1',
      auth: {
        type: 'AMAZON_COGNITO_USER_POOLS',
        jwtToken: async () => (await Auth.currentSession()).getIdToken().getJwtToken(),
      },
    }),
    createSubscriptionHandshakeLink({
      url: process.env.APPSYNC_ENDPOINT!,
      region: 'eu-central-1',
    }),
  ]),
  cache: new InMemoryCache(),
});
```

Gotcha: the subscription link reconnects automatically, but the first subscription message can arrive before the UI is ready. Handle that race with a `useEffect` that checks `NetInfo` before subscribing.

## Step 2 — core implementation

### 2.1 Local state machine

Create `src/state/machine.ts`:

```typescript
type State = 'online' | 'offline' | 'syncing';
type Event = 'GO_ONLINE' | 'GO_OFFLINE' | 'START_SYNC' | 'END_SYNC';

const machine = createMachine<{ state: State }>({
  id: 'connectivity',
  initial: 'offline',
  states: {
    offline: {
      on: { GO_ONLINE: 'online' },
    },
    online: {
      on: { GO_OFFLINE: 'offline', START_SYNC: 'syncing' },
    },
    syncing: {
      on: { END_SYNC: 'online' },
    },
  },
});
```

Drive this from React Query optimistic updates so the UI never blocks.

### 2.2 Background sync worker

Create `src/workers/syncWorker.ts`:

```typescript
export async function syncQueue() {
  const queue = await db.getPending();
  if (queue.length === 0) return;

  try {
    const { success, conflicts } = await callLambda('/sync', queue);
    if (success) {
      await db.deleteBatch(queue.map(q => q.id));
    } else {
      await db.markConflicts(conflicts);
    }
  } catch (err) {
    if (isNetworkError(err)) {
      await db.markRetry(queue.map(q => q.id));
      return;
    }
    throw err;
  }
}
```

**Sizing the batch.** Batch size is a latency-versus-overhead tradeoff, not a fixed number. To pick one, instrument three things: the serialized payload size per record, the round-trip time for one request on the slowest link you support, and the server-side processing time per record. Then batch size is roughly `(latency_budget − rtt) / per_record_server_cost`, capped by the payload size your backend accepts. For example, if a 2G link gives 900 ms round-trip, the server processes 10 records per millisecond, and your p99 budget is 1.2 s, you have 300 ms of server budget, so about 30 records per request. Confirm the number by running the sync through a network proxy that emulates 2G and recording p99 with real payloads — never trust a theoretical constant.

### 2.3 Conflict resolution UI

In `src/screens/SyncScreen.tsx`:

```typescript
const ConflictList = ({ conflicts }: { conflicts: Conflict[] }) => {
  const [choice, setChoice] = useState<Choice | null>(null);
  const mutation = useUpdatePackage();

  const handleResolve = () => {
    if (!choice) return;
    mutation.mutate({ id: choice.id, version: choice.version });
  };

  return (
    <FlatList
      data={conflicts}
      renderItem={({ item }) => (
        <ConflictCard
          item={item}
          onAccept={() => setChoice({ id: item.id, version: item.localVersion })}
          onReject={() => setChoice({ id: item.id, version: item.remoteVersion })}
        />
      )}
      ListFooterComponent={
        <Button onPress={handleResolve} disabled={!choice}>Resolve</Button>
      }
    />
  );
};
```

Keep the version vector in the SQLite row as a `BLOB` (8 bytes is enough for most field workloads) so you can merge correctly even when the device was offline for two days.

## Step 3 — handle edge cases and errors

### 3.1 Battery-aware sync

Riders commonly complain the app drains their phone in a few hours. Add a battery check:

```typescript
const batteryThreshold = 20; // percent
const level = await getBatteryLevel();
if (level < batteryThreshold) {
  await db.pauseSync();
  Notifications.post('Sync paused: battery low');
}
```

### 3.2 Storage pressure

SQLite can bloat if the rider scans hundreds of packages per day for a week. Add a prune job:

```typescript
const size = await db.size();
if (size > 100 * 1024 * 1024) { // 100 MB
  await db.pruneOld(7); // keep 7 days
}
```

### 3.3 Network detection traps

`NetInfo` can lie — it reports link state, not reachability. A health-check endpoint that returns a 204 within a short timeout is more reliable. If the endpoint misses three consecutive pings, go offline:

```typescript
const isHealthy = await fetchWithTimeout('/health', { timeout: 800 });
if (!isHealthy) {
  machine.send('GO_OFFLINE');
}
```

### 3.4 Error taxonomy

| Error type | Typical cause | Recovery | User impact |
|------------|---------------|----------|-------------|
| Lambda timeout | Batch too large | Retry with smaller batch | Spinner |
| Cognito token expiry | Long offline period | Refresh token | Login screen flash |
| SQLite disk full | Unpruned history | Prune | App crash |
| Slow-link timeout | 2G latency | Exponential backoff | "Syncing" indicator |

The "typical cause" column is the useful one: it tells you which knob to turn. Frequency numbers are workload-specific, so measure them in your own environment rather than borrowing them.

## Step 4 — add observability and tests

### 4.1 Logging without PII

Use a proxy Lambda that strips PII before forwarding to CloudWatch:

```typescript
exports.handler = async (event) => {
  const { userId, ...rest } = JSON.parse(event.body);
  await cloudWatch.putLogEvents({
    logGroupName: '/field-agent/anon',
    logStreamName: new Date().toISOString().slice(0, 10),
    logEvents: [{ message: JSON.stringify(rest) }],
  });
};
```

### 4.2 SQLite test doubles

Use an in-memory SQLite database to run tests in CI without touching disk:

```typescript
import Database from 'better-sqlite3';

describe('syncWorker', () => {
  it('retries on network error', async () => {
    const db = new Database(':memory:');
    db.exec('CREATE TABLE queue (id TEXT PRIMARY KEY, payload TEXT)');
    const worker = new SyncWorker(db);
    await expect(worker.sync()).resolves.toBeUndefined();
  });
});
```

### 4.3 Measuring battery life honestly

Battery claims are easy to invent and hard to reproduce. If you want a number, measure it: install the app on a representative device, disable charging, run a scripted workload (for example, a scan every 30 seconds plus a sync every 3 minutes), and log `BatteryManager` level and timestamp to a file. Run the same script before and after your optimization and compare the slopes. Report the device model, Android version, brightness, and whether the screen was on — otherwise the number is meaningless.

## What this pattern tends to deliver

Teams that run an offline-first agent with a persisted queue typically see fewer duplicate scans. Without the local state machine, riders can scan the same package twice when the UI freezes during a dropout; with it, the scan is queued locally and merged upstream, and the rider gets a success toast even when offline.

To know your own numbers, instrument these:

- **Sync p99 over 2G.** Run the sync worker through a proxy that emulates 2G, record request duration, and take the 99th percentile. Compare against a naive reconnect loop on the same proxy.
- **Conflict rate.** Count rows where the server rejects a stale version, divided by total rows synced. Anything above a few percent usually means the client is not sending version vectors.
- **Local read latency.** Time the SQLite query that backs the rider's main screen. If it exceeds a few hundred milliseconds, the UI will feel sluggish during sync.

## Common questions and variations

### How do I keep rider GPS data compliant?

Store GPS only when the rider presses "start route" and delete it after a short retention window. Use an encrypted blob column in SQLite; derive the key from the rider's Cognito sub so it can't be decrypted elsewhere.

### Can I use this with Flutter instead of React Native?

Yes. Replace SQLite bindings with a Flutter SQLite package and the worker with a background-task plugin. The conflict resolution UI is pure Dart, so porting is straightforward.

### What if the rider's phone is stolen?

The local database is encrypted with SQLCipher. The rider logs out via Cognito, which invalidates all tokens; the next login triggers a wipe of the local DB. No extra code needed.

### How do I scale this to many riders?

Use a stream-based ingestion path (for example, DynamoDB streams) with a Lambda that writes to an outbox table per rider. The agent still reads from SQLite, so the UI stays snappy.

## Frequently Asked Questions

### How do I test an offline-first agent without a real 2G network?

Use a network proxy such as `toxiproxy` or Android's built-in network throttling to simulate high latency, packet loss, and intermittent dropouts. Run your end-to-end suite against the proxy so the sync worker exercises real timeout paths rather than mocked ones. Pair that with a CI job that runs the in-memory SQLite tests for fast feedback, and reserve the proxy-based tests for nightly runs. The goal is to reproduce the reconnect race and the partial-batch failure, which are the two failure modes unit tests almost never catch.

### Does React Query handle offline mutations out of the box?

No. React Query retries failed mutations and can pause them when the network is offline, but it does not persist the mutation queue across app restarts. For a field agent that may be killed by the OS or the user, you need to persist the queue yourself — typically in the same SQLite database that holds your local state. React Query then becomes the UI-facing layer that reads from that queue, while your sync worker owns delivery and retry logic.

### How do I avoid conflicts when two devices edit the same record?

Use a version vector or a monotonically increasing version number stored alongside each row, and send it with every mutation. The server rejects writes whose version is stale and returns the current server version so the client can present a merge choice. Keeping the vector small (8 bytes is enough for most field workloads) means you can store it inline in SQLite without a separate table. For most logistics data, last-writer-wins is acceptable for status fields, but quantities and signatures should always go through explicit conflict resolution.

### Should the sync worker run in the foreground or as a background task?

Both, with different responsibilities. A foreground worker triggered by connectivity changes gives the rider immediate feedback and handles the common case. A background task (Android WorkManager, iOS BGTaskScheduler) catches up when the app is not open, but it is subject to OS scheduling limits and battery restrictions. Design the queue so it is idempotent and resumable, then let either worker drain it — that way a killed background task never loses data or double-applies a mutation.

## Where to go from here

Open `src/workers/syncWorker.ts` and add a counter that records the serialized payload size of each batch and the round-trip time of each `/sync` call. Run the sync through a 2G proxy for ten minutes and print the p99. Then use the formula above to pick a batch size and confirm the p99 fits your budget. That is the single change that turns a guess about offline performance into a measurement.
