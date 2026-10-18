# AI wrote the docs — humans wrote the gaps

Auto-generated API documentation is good at the happy path. Given JSDoc comments, docstrings, or type signatures, a language model will produce clean, plausible reference pages: parameters, response shapes, example values, a friendly summary sentence. What it will not produce, unless a human writes it down first, is the set of facts that actually cause production incidents — the rate limit that is lower than anyone assumes, the field that is soft-deleted after a fixed window, the webhook queue that silently drops events past a threshold, the retry policy that a partner expects you to honor.

This article describes a pattern for separating those two kinds of documentation and keeping them mechanically in sync: let automation own the generated reference, let humans own a small, explicit file of edge cases, and add a CI check that fails when the two disagree with runtime behavior.

## The failure mode: plausible documentation that is wrong

The characteristic problem with AI-drafted reference docs is not that they are obviously broken. It is that they read correctly. A model trained on API documentation has seen thousands of endpoints named `/transfers` and will confidently describe one. It has no way to know that your `/transfers` endpoint rejects a currency mismatch with a 400, that your account records are soft-deleted after 30 days, or that your webhook delivery drops events when the queue exceeds a fixed size.

The failure mode has a recognizable shape:

1. A developer adds an endpoint and writes a docstring covering inputs and outputs.
2. A generation step expands that docstring into full reference documentation.
3. The documentation is published and looks complete.
4. A consumer integrates against it, hits an undocumented constraint, and files a bug.
5. The fix is a one-line addition to the docs, but the next endpoint repeats the cycle.

The cost is not the documentation itself. It is the integration work, support load, and debugging time that follows from a consumer believing a document that was never checked against the system.

The structural fix is to stop treating "the docs" as one artifact. Split it:

- **Generated reference** — endpoint paths, request and response schemas, parameter types. Derived from code. Safe to automate.
- **Human-maintained edges** — rate limits, retry policies, deprecation schedules, undocumented side effects, error conditions that are not visible in a type signature. Written by people who were present when the system broke.

The rest of this article is about the second category: how to write it, where to put it, and how to make CI enforce it.

## Prerequisites

- A service with at least a handful of public-facing endpoints or classes.
- Node.js 20 LTS or Python 3.11 (either is fine; both are shown below).
- An OpenAPI spec generation tool. Any bundler that reads annotations or framework metadata and emits an OpenAPI document will do.
- An OpenAPI viewer of your choice for rendering.
- A CI system that runs on pull requests.

The examples use a deliberately small banking API with three endpoints: `GET /accounts/{id}`, `POST /transfers`, and a webhook receiver. The size is intentional — the pattern matters more than the surface area.

## Step 1 — create the human edge file first

Before touching any generator, write the file that automation must not overwrite. Create `docs/edge-cases.md`:

```markdown
# Edge cases and hidden contracts
> Human-maintained. Do not auto-generate.

This file lists behaviors that cannot be inferred from types or signatures:
rate limits, retry policies, deprecated fields, and undocumented side effects.

## Accounts
- `GET /accounts/{id}`: returns 404 if the account was soft-deleted more than 30 days ago
- Rate limit: 100 requests per minute per API key

## Transfers
- `POST /transfers`: `source_currency` must match the account currency, else 400
- Retry policy: 3 attempts, exponential backoff starting at 2 seconds

## Webhooks
- Events are dropped if queue size exceeds 1000
- Signature expires 5 minutes after issuance
```

Two properties make this file useful. First, every line is a claim that can be tested — a status code, a numeric limit, a time window. Second, it is small enough that a reviewer will actually read a diff to it.

Commit this file before writing any generation script. The point is to make the human gaps visible before automation fills the rest of the documentation with plausible-sounding text.

## Step 2 — generate the reference from code

Keep the annotations minimal and factual. The generator will reproduce what you write and will not invent constraints you omit.

For a Node service, annotate the route:

```javascript
/**
 * @openapi
 * /accounts/{id}:
 *   get:
 *     summary: Retrieve account by ID
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema:
 *           type: string
 *           format: uuid
 *     responses:
 *       '200':
 *         description: OK
 *       '404':
 *         description: Account not found or soft-deleted more than 30 days ago
 */
app.get('/accounts/:id', (req, res) => { /* ... */ });
```

For a Python service using FastAPI, the framework derives the schema from type annotations, and the docstring carries the human-readable text:

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Account(BaseModel):
    id: str
    balance: float
    currency: str

@app.get("/accounts/{account_id}", response_model=Account)
async def get_account(account_id: str):
    """
    Retrieve account by ID.

    Returns 404 if the account was soft-deleted more than 30 days ago.
    """
    # ...
```

Note the asymmetry: the schema is generated, but the sentence about soft deletion is written by a human and copied verbatim. That sentence also appears in `edge-cases.md`. The duplication is deliberate — it is what the CI check will verify.

## Step 3 — merge generated reference with human edges

Write a build script that produces a single OpenAPI document from two inputs: the generated spec and the edge file. The script should also fail loudly if an edge cannot be matched to a path.

```javascript
import { readFileSync, writeFileSync } from 'fs';
import { execSync } from 'child_process';

// Load human edges, split on level-2 headings.
const edges = readFileSync('docs/edge-cases.md', 'utf8')
  .split('## ')
  .slice(1)
  .map(section => {
    const [title, ...lines] = section.split('\n');
    return { title: title.trim(), body: lines.join('\n') };
  });

// Generate the OpenAPI document from code annotations.
console.log('Generating OpenAPI spec...');
execSync('npx @redocly/cli bundle openapi.yaml -o dist/openapi.json', { stdio: 'inherit' });

// Attach the human edges as a vendor extension.
const spec = JSON.parse(readFileSync('dist/openapi.json', 'utf8'));
spec.components = spec.components || {};
spec.components['x-edge-notes'] = edges;

writeFileSync('dist/openapi.json', JSON.stringify(spec, null, 2));

// Fail the build if any edge section has no corresponding path in the spec.
const missing = edges.filter(edge => {
  const key = '/' + edge.title.toLowerCase().replace(/\s+/g, '');
  return !spec.paths[key];
});

if (missing.length) {
  console.error('Edges with no matching path:', missing.map(m => m.title));
  process.exit(1);
}
```

Two things to be aware of when wiring this up. First, most bundlers do not preserve arbitrary `x-` fields through every transform; verify that `x-edge-notes` survives your pipeline, and if it does not, patch the output after bundling rather than before. Second, the naive path-matching above assumes the edge file's section headings map to URL paths. That assumption breaks as soon as you have nested resources or versioned prefixes. A more robust approach is to require an explicit path in each edge line, or to maintain a small mapping table in the script.

## Step 4 — enforce the edges at runtime

Documentation that is not enforced will drift. Add the mechanisms that make each documented edge a real behavior.

### Rate limiting

```javascript
import rateLimit from 'express-rate-limit';

const accountsLimiter = rateLimit({
  windowMs: 60 * 1000,
  max: 100,
  keyGenerator: (req) => req.headers['x-api-key'],
  handler: (req, res) => {
    res.status(429).json({
      error: 'Too many requests',
      retryAfter: req.rateLimit.resetTime
    });
  }
});
```

### Soft-delete visibility

```javascript
app.use('/accounts/:id', async (req, res, next) => {
  const account = await db.getAccount(req.params.id);
  const SOFT_DELETE_WINDOW_MS = 30 * 24 * 60 * 60 * 1000; // 30 days
  if (account && account.deletedAt && Date.now() - account.deletedAt > SOFT_DELETE_WINDOW_MS) {
    return res.status(404).json({ error: 'Account not found' });
  }
  next();
});
```

The Python equivalent, using a limiter decorator and an explicit window check:

```python
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address

app = FastAPI()
limiter = Limiter(key_func=get_remote_address)

SOFT_DELETE_WINDOW = timedelta(days=30)

@app.get("/accounts/{account_id}")
@limiter.limit("100/minute")
async def get_account(account_id: str):
    account = await db.get_account(account_id)
    if not account:
        raise HTTPException(status_code=404, detail="Account not found")
    if account.deleted_at and datetime.now() - account.deleted_at > SOFT_DELETE_WINDOW:
        raise HTTPException(status_code=404, detail="Account not found")
    return account
```

### A single source of truth for limits

Numeric limits should live in one file that both the runtime and the documentation build read:

```yaml
accounts:
  get: 100/minute
  soft_delete_days: 30
transfers:
  post: 50/minute
  retries: 3
  retry_delay_start_ms: 2000
webhooks:
  queue_size: 1000
  signature_expiry_minutes: 5
```

Loading it is trivial in both runtimes:

```javascript
import yaml from 'js-yaml';
import { readFileSync } from 'fs';
const limits = yaml.load(readFileSync('config/limits.yaml', 'utf8'));
```

```python
import yaml
with open('config/limits.yaml') as f:
    limits = yaml.safe_load(f)
```

When the limit file is the only place a number appears, changing a rate limit is a one-line diff that propagates to the runtime and to the generated documentation in the same commit. This is the single highest-leverage part of the pattern.

## Step 5 — check documentation against runtime in CI

The build script verifies that the edge file and the spec agree with each other. It does not verify that either agrees with the running service. That requires an integration check.

A minimal CI step that starts the service and runs a test collection against it:

```yaml
- name: Validate docs against runtime
  run: |
    docker build -t edge-api:test .
    docker run -d -p 8000:8000 edge-api:test
    sleep 5
    npm install -g newman
    newman run tests/docs-validation.json
```

The collection mirrors the human edge file, one request per documented claim:

```json
{
  "item": [
    {
      "name": "GET /accounts/{soft_deleted_id} returns 404",
      "request": {
        "method": "GET",
        "url": "http://localhost:8000/accounts/123e4567-e89b-12d3-a456-426614174000"
      },
      "response": { "status": 404 }
    },
    {
      "name": "GET /accounts/{valid_id} returns 200",
      "request": {
        "method": "GET",
        "url": "http://localhost:8000/accounts/123e4567-e89b-12d3-a456-426614174001"
      },
      "response": { "status": 200 }
    },
    {
      "name": "POST /transfers with mismatched currency returns 400",
      "request": {
        "method": "POST",
        "url": "http://localhost:8000/transfers",
        "header": [{ "key": "Content-Type", "value": "application/json" }],
        "body": {
          "mode": "raw",
          "raw": "{\"source_currency\":\"EUR\",\"amount\":100}"
        }
      },
      "response": { "status": 400 }
    }
  ]
}
```

If any of these fail, the build fails, and the author has to reconcile the edge file, the code, or the test. That reconciliation is the entire point: it converts a documentation drift problem into a failing check that someone must resolve before merging.

The same check can be written as a native test in either runtime:

```javascript
import { test } from 'node:test';
import assert from 'node:assert';
import { execSync } from 'child_process';

test('documented edges match runtime behavior', () => {
  execSync('docker build -t edge-api:test .');
  execSync('docker run -d -p 8000:8000 edge-api:test');
  const output = execSync(
    'npm install -g newman && newman run tests/docs-validation.json --reporters cli',
    { encoding: 'utf8' }
  );
  assert(output.includes('failures'), 'newman should report a failure count');
  assert(!/failures\s+[1-9]/.test(output), 'no documented edge should fail');
});
```

```python
import pytest
import requests

@pytest.fixture(scope="module")
def api():
    import subprocess
    subprocess.run(['docker', 'build', '-t', 'edge-api:test', '.'], check=True)
    subprocess.run(['docker', 'run', '-d', '-p', '8000:8000', 'edge-api:test'], check=True)
    yield 'http://localhost:8000'

def test_soft_deleted_account_returns_404(api):
    r = requests.get(f'{api}/accounts/soft-deleted-fixture')
    assert r.status_code == 404

def test_currency_mismatch_returns_400(api):
    r = requests.post(f'{api}/transfers', json={
        'source_currency': 'EUR',
        'amount': 100
    })
    assert r.status_code == 400
```

## Measuring whether this is working

Claims about reduced onboarding time or fewer incidents are only meaningful if the underlying numbers are instrumented. If you want to know whether the pattern helps, measure these directly rather than relying on anecdote:

- **Documentation drift rate.** Count the number of CI failures attributable to the docs-vs-runtime check per week. A stable or declining count with a growing endpoint count is a good signal. Instrument this by tagging the CI job's failure reason.
- **Time from endpoint change to documentation update.** Track the commit timestamps of the endpoint code and the corresponding `edge-cases.md` line. The gap should shrink toward zero once the CI check is enforced.
- **Support tickets referencing undocumented behavior.** Tag tickets at intake with the endpoint and whether the reporter cited the docs. Compare the rate before and after enforcing the check.
- **Review time on generated documentation.** If reviewers are spending time on generated reference pages, that is wasted effort — the pages should be reviewed once when the annotation changes and never again. Measure the ratio of review comments on generated versus human-written sections.

Each of these is a query against data you already have, not a benchmark you need to trust someone else's word on.

## Common questions

### Why not let the model generate the edge cases too?

It can generate plausible ones, and that is the problem. Given a prompt like "list edge cases for a transfers endpoint," a model will produce a long list of rules that sound reasonable — retry policies for callbacks that do not exist, validation rules your service does not enforce, deprecated fields you never had. The failure is not that the output is empty; it is that reviewing it costs more than writing the rules from scratch, because every line has to be verified against the system. The edge file works precisely because its contents are claims a human already knows to be true.

### How do you keep the edge file from growing without bound?

Set an explicit size budget and enforce it in CI. When the file exceeds the budget, split it:

- `edge-cases.md` — enforced, tested, and reviewed on every change.
- `edge-appendix.md` — documented but not enforced; rare cases.
- `changelog.md` — deprecations and upcoming changes.

Anything not in the enforced file must be justified in the pull request. Without a budget, the file becomes a dumping ground and stops being read.

### What about GraphQL or gRPC?

The pattern is transport-agnostic. The generated artifact changes — a GraphQL schema or a `.proto` file instead of OpenAPI — but the structure is the same: a machine-derived contract plus a small human-maintained file of constraints, with a CI check that the two agree with runtime. For GraphQL, the human file can be a set of directives or a sidecar document listing deprecations and rate limits. For gRPC, comments on the service definition serve the same role as the docstring above.

### How are breaking changes handled?

Treat a breaking change as an edge with a date attached:

```markdown
## Breaking change: transfers v2
- On 2026-09-01, `POST /transfers` will reject `source_currency` mismatches with 400
- Clients must use `source_account_id` instead
```

Add a `deprecation_date` to the limits file, and add a CI check that fails if any client still uses the old field after that date. The changelog entry is generated from the same source, so the human and machine views cannot diverge.

## What to do in the next 30 minutes

Open your repository, create `docs/edge-cases.md`, and write the first three constraints you know are true but that are not in your generated documentation. Pick ones that are testable: a status code, a rate limit, a time window. Then add a single CI step that asserts one of them against a running instance of the service. The goal is not completeness — it is to make the first gap visible and enforced, so the next endpoint you add has a place to record the things a model will never guess.
===END===
