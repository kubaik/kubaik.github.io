# AI freelance rates 2026: which stacks still pay

AI assistants are good at the visible part of a codebase: components, CRUD handlers, schema drafts, boilerplate. They are indifferent to the invisible part: idempotency, durable audit trails, row-level security, latency budgets under concurrency. A typical failure mode is not that the generated code is obviously wrong — it is that it compiles, passes local tests, and fails in a way that costs money weeks later.

This article covers four recurring failure patterns, the guardrails that catch them, and how to price the work honestly instead of absorbing the cost silently.

## Why generated code fails late rather than early

Generated code is optimized for the shape of the prompt, not for the shape of production. Three properties of production are systematically underrepresented:

- **Concurrency.** Local tests run one request at a time. Duplicate webhooks, retried jobs and simultaneous writes only appear under real traffic.
- **Durability.** Code that writes to a container's local filesystem, an ephemeral CI runner, or an in-memory cache looks correct in review and disappears on restart.
- **Identity.** Authorization logic that relies on application code passing the right filter breaks the moment a query is written by something else — including a future version of the same assistant.

None of these are exotic. They are the ordinary reasons production systems need review, and they are exactly the categories that a code generator has no way to verify.

## Failure pattern 1: non-idempotent payment handling

A common scenario: an assistant is asked for a checkout or billing integration and returns a working-looking handler built on a deprecated redirect method or a naive "charge on event" flow. Test keys generally do not enforce idempotency, so the code passes local testing. In production, webhook providers retry on timeouts and non-2xx responses, and the same event can be delivered more than once.

**What to instrument.** Log every inbound webhook with its provider event ID and a hash of the raw body. Count distinct event IDs versus total deliveries per day. A ratio above 1.0 means retries are happening in your environment and your handler must be idempotent.

**What to compare.** Run the handler twice with the same event payload against a test database and assert that the number of side effects (rows written, emails queued, ledger entries) is identical after both runs.

**The guardrail.** Persist a record of each processed event ID in a unique-indexed table and short-circuit on conflict. The database constraint, not application logic, is what makes this safe under concurrency.

## Failure pattern 2: hallucinated or deprecated imports

Assistants frequently produce import paths that look plausible and do not exist, or that existed in an older major version. Because package managers resolve namespaces loosely in some ecosystems, a wrong import can compile and fail only at runtime when the module is loaded.

**What to instrument.** A pre-merge check that extracts every import specifier from changed files and verifies it against the lockfile plus an explicit allow-list of internal aliases. Reject anything not present in both.

**What to compare.** Diff the set of imported packages against `dependencies` and `devDependencies` in the manifest. Every external import should have a matching entry; every entry with no import is a candidate for removal.

A minimal allow-list check in CI:

```bash
# fail the build if any import is not in the manifest or the allow-list
node - <<'EOF'
const fs = require('fs');
const path = require('path');
const manifest = require('./package.json');
const known = new Set([
  ...Object.keys(manifest.dependencies || {}),
  ...Object.keys(manifest.devDependencies || {}),
]);
const allowInternal = [/^@\//, /^\.\.?\//];
const files = process.argv.slice(2);
let bad = 0;
for (const f of files) {
  const src = fs.readFileSync(f, 'utf8');
  for (const m of src.matchAll(/from\s+['"]([^'"]+)['"]/g)) {
    const spec = m[1];
    if (allowInternal.some(r => r.test(spec))) continue;
    const pkg = spec.startsWith('@')
      ? spec.split('/').slice(0, 2).join('/')
      : spec.split('/')[0];
    if (!known.has(pkg)) {
      console.error(`unknown import ${pkg} in ${f}`);
      bad++;
    }
  }
}
process.exit(bad ? 1 : 0);
EOF
```

This is deliberately crude. It will not resolve conditional exports or workspace protocols, but it catches the common case — a package name that was invented or renamed — before it reaches a deploy.

## Failure pattern 3: audit evidence that is not durable

Compliance frameworks ask for evidence that is written somewhere it cannot be silently lost. Generated CI workflows often satisfy the *shape* of that requirement — a step named "export logs", an `echo` of a timestamp — while writing to the runner's ephemeral filesystem. The pipeline is green; the evidence does not exist an hour later.

**What to instrument.** After each pipeline run, assert that the expected artifact exists at a durable location and that its checksum matches the value recorded in the build log. A missing artifact should fail the pipeline, not a later audit.

**What to compare.** For a given change, confirm that the set of required artifacts (merged diff, dependency graph, deploy timestamp) is present in object storage with a retention policy that exceeds the compliance window.

**The guardrail.** Write evidence to object storage with an explicit retention rule and an immutable-object policy where the provider supports it. Treat any evidence step that writes only to the runner as a bug.

## Failure pattern 4: middleware that changes the latency profile

A middleware added for tracing, logging or header injection is usually cheap in a single-request test and expensive under concurrency if it performs synchronous I/O. The failure is a step change, not a gradual degradation: p95 looks fine in staging and collapses at production concurrency.

**What to instrument.** Measure p50, p95 and p99 separately for requests with and without the middleware. Record event-loop lag and the time spent in the middleware itself, not just total request duration.

**What to compare.** Run a load test at your expected peak and at twice that value. A component that is acceptable at 1× and unacceptable at 2× is a scheduling problem, not a capacity problem.

A quick loop-lag probe to keep in a staging environment:

```js
// logs event-loop delay; run during load tests, not in production
let last = process.hrtime.bigint();
setInterval(() => {
  const now = process.hrtime.bigint();
  const delayMs = Number(now - last - 1000n * 1000n * 1000n) / 1e6;
  if (delayMs > 50) console.warn(`event loop lag ${delayMs.toFixed(1)}ms`);
  last = now;
}, 1000).unref();
```

## A guardrail stack that fits most projects

The specific tools matter less than the categories. Any project that touches payments, regulated data, or a latency-sensitive endpoint needs four checks, and they can all run in the same pipeline.

| Category | What it detects | Where it runs |
|---|---|---|
| Import allow-list | Hallucinated or renamed packages | Pre-merge |
| Dependency and secret scan | Known-vulnerable versions, committed credentials | Pre-merge |
| Idempotency test | Duplicate side effects on replay | Pre-merge |
| Artifact durability check | Evidence written only to ephemeral storage | Post-deploy |
| Load test at 2× peak | Middleware that degrades non-linearly | Pre-release |

The pipeline below implements the first and last of these; the idempotency and durability checks are application-specific and belong in the test suite.

```yaml
# .github/workflows/guardrails.yml
name: Guardrails
on: [push, pull_request]

jobs:
  checks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
      - run: npm ci
      - name: Import allow-list
        run: node scripts/check-imports.js $(git diff --name-only origin/main...HEAD -- '*.ts' '*.tsx')
      - name: Dependency audit
        run: npm audit --audit-level=high
      - name: Duplicate-delivery test
        run: npm test -- --grep "webhook idempotency"
```

Two notes on this shape. First, the import check runs only against changed files, which keeps it fast enough that nobody argues about it. Second, the idempotency test is a real test with a real database, not a mock: mocks cannot reproduce unique-constraint behavior, and the unique constraint is the whole point.

## Idempotent webhook handling, concretely

The handler below is written for a framework with a Web Fetch-style request object and a Postgres-compatible database. The important properties are that the raw body is used for signature verification, and that the event ID is inserted into a unique-indexed table before any side effect occurs.

```ts
// app/api/webhooks/stripe/route.ts
import { NextResponse } from 'next/server';
import Stripe from 'stripe';
import { db } from '@/lib/db';

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY!, {
  apiVersion: '2024-10-15.acacia',
});

export async function POST(req: Request) {
  const body = await req.text();
  const signature = req.headers.get('stripe-signature');
  if (!signature) {
    return NextResponse.json({ error: 'missing signature' }, { status: 400 });
  }

  let event: Stripe.Event;
  try {
    event = stripe.webhooks.constructEvent(
      body,
      signature,
      process.env.STRIPE_WEBHOOK_SECRET!,
    );
  } catch (err) {
    // verification failure is a client error, not a server error
    return NextResponse.json({ error: 'invalid signature' }, { status: 400 });
  }

  // The unique constraint on event_id is what makes this idempotent.
  // If the insert conflicts, the event has already been handled.
  const inserted = await db.query(
    `INSERT INTO processed_events (event_id, received_at)
     VALUES ($1, now())
     ON CONFLICT (event_id) DO NOTHING
     RETURNING event_id`,
    [event.id],
  );

  if (inserted.rowCount === 0) {
    return NextResponse.json({ received: true, duplicate: true });
  }

  try {
    await handleEvent(event);
  } catch (err) {
    // Roll back the marker so the provider's retry can succeed.
    await db.query(`DELETE FROM processed_events WHERE event_id = $1`, [event.id]);
    throw err;
  }

  return NextResponse.json({ received: true });
}

async function handleEvent(event: Stripe.Event) {
  switch (event.type) {
    case 'checkout.session.completed':
      // business logic
      break;
    default:
      break;
  }
}
```

Three details are doing the work here, and each corresponds to a failure pattern above:

1. **Signature verification uses the raw body.** Parsing the JSON first and re-serializing it will break the signature check in ways that are hard to debug.
2. **The `ON CONFLICT DO NOTHING` insert is the idempotency mechanism.** Application-level "have I seen this?" checks race under concurrency; a unique constraint does not.
3. **The marker is deleted on handler failure.** Without this, a transient downstream error would permanently suppress a legitimate retry.

A corresponding migration:

```sql
CREATE TABLE processed_events (
  event_id TEXT PRIMARY KEY,
  received_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Retain markers at least as long as the provider's retry window.
CREATE INDEX processed_events_received_at_idx ON processed_events (received_at);
```

## Row-level security as a backstop, not a primary control

Application-level tenant filtering is a primary control: it is what you intend to happen. Row-level security is a backstop: it is what happens when the application is wrong. Both are worth having for multi-tenant data.

```sql
CREATE TABLE gyms (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  owner_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE
);

ALTER TABLE gyms ENABLE ROW LEVEL SECURITY;

CREATE POLICY gym_access_policy ON gyms
  USING (owner_id = current_setting('app.current_user_id', true)::UUID);
```

The session variable must be set by the connection pool on checkout, not by a trigger on the table. A trigger that sets the variable during `INSERT` or `UPDATE` does not protect `SELECT`, which is the query that leaks data. Set it per transaction:

```sql
-- executed at the start of each request's transaction
SELECT set_config('app.current_user_id', $1, true);
```

Two caveats worth stating plainly. First, `current_setting(..., true)` returns `NULL` rather than erroring when the variable is unset, and `NULL` comparisons fail closed only if the policy is written as an equality against a non-null column — verify this with a test that queries without setting the variable and asserts zero rows. Second, table owners and roles with `BYPASSRLS` are exempt by default, so the application role must not be the table owner.

## Pricing the invisible work

The honest way to price AI-assisted work is to separate the visible deliverable from the guardrails, and to make the guardrails a line item rather than a hidden cost.

A worked example, with all figures illustrative and assumptions stated:

- Assume a feature that an assistant drafts in **6 hours** of prompting and review.
- Assume the guardrails — import check, idempotency test, artifact durability check, 2× load test — take **5 hours** to write and wire into CI.
- Assume the guardrails catch two defects that would otherwise have surfaced in production, each costing **4 hours** of incident response plus a client-relationship cost that is not billable.

The visible work is 6 hours; the invisible work is 5 hours plus the avoided 8 hours. Quoting only the visible work means the 5 hours are absorbed and the 8 hours are a gamble. Quoting both means the client sees an 11-hour estimate for a 6-hour artifact, which is a conversation worth having explicitly rather than discovering at invoice time.

The same logic applies to infrastructure. A vector store, a tracing backend, or an audit-log bucket adds a recurring cost that did not exist before. Quote it as a separate line so it does not silently erode margin.

## A decision checklist before shipping generated code

Run through this before merging anything an assistant produced for a system that handles money, personal data, or a stated latency target.

- Does every external import resolve against the lockfile, and is there a CI check that enforces this?
- For every event-driven handler, is there a unique-indexed record of processed event IDs, and a test that replays the same event twice?
- Does every compliance artifact land in storage with a retention policy, and does the pipeline fail if it does not?
- Does any middleware perform synchronous I/O on the request path, and has the system been load-tested at twice expected peak?
- Are tenant-scoped queries protected by both application filtering and a database-level policy, and is the policy tested with the session variable unset?
- Is the recurring infrastructure cost of the guardrails quoted separately from the build cost?

If any answer is "we will check later", that is the item most likely to become the incident.

## What to do in the next 30 minutes

Pick one event-driven endpoint in a current project and write a test that sends the same event payload twice, then assert that the number of side effects after the second delivery equals the number after the first. If the test fails, the fix is a unique constraint on the event ID and an `ON CONFLICT DO NOTHING` insert — and you have just found, in half an hour, a class of bug that otherwise surfaces as duplicate charges.
