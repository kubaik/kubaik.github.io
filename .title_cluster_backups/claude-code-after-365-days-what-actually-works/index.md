# Claude Code after 365 days: what actually works

Launch posts tend to show the happy path: a prompt, a burst of generated files, a merged pull request. What they rarely show is the second pass — the human review, the missing environment variable, the retry that fires four times when you asked for three, the circuit breaker that reopens too early. This article walks through a repeatable template for agent-assisted development on a small but non-trivial service, and documents the failure modes that show up when you move past the demo.

The goal is not to argue for or against agents. It is to describe a workflow that produces code you can actually operate: pinned dependencies, tests that fail for the right reasons, observability wired in from the start, and a review step that catches what the agent cannot know about your environment.

## What agents are good at, and what they are not

A useful mental model is that an agent is a very fast boilerplate generator with no memory of your production incidents. It will produce a plausible GitHub client, a plausible retry wrapper, a plausible CI workflow. It will not know that your CI runner has no outbound network, that your token is injected by an OIDC provider, or that your team was paged twice last quarter by a specific upstream. Those are the things humans add.

The recurring pattern across teams that adopt this workflow:

- **Faster first drafts.** The agent produces a working skeleton in minutes rather than an afternoon.
- **A mandatory second pass.** Almost every generated file needs a human edit before it is safe to merge. The edit is usually small but non-optional.
- **Test suites that grow faster than they mature.** Coverage numbers rise, but the new tests often assert the happy path and skip the failure modes you actually care about.
- **Documentation that is technically correct but shallow.** Generated docs describe what the code does, not what it does when the upstream returns 429 at 3 a.m.

None of these are reasons to skip the tool. They are reasons to design the workflow around them.

## Prerequisites and what you will build

You will need:

- A GitHub, GitLab, or Bitbucket repository containing Node.js or Python code you are willing to restructure.
- Node 20 LTS or Python 3.11 on your PATH.
- Access to an agentic coding CLI or IDE extension. The specific product does not matter much; the workflow below assumes a CLI that reads a project instruction file and generates files into a target directory.
- A terminal, an editor, and roughly 45 minutes of uninterrupted time.

The example service is deliberately small: a REST endpoint that accepts an issue reference, validates it, and adds a label via the GitHub API. It must:

- Validate inputs with a schema library (Zod for Node, Pydantic for Python).
- Persist nothing by default — keep the service stateless so tests are fast and deterministic.
- Generate an OpenAPI document from the same schemas used for validation, so the two cannot drift.
- Include unit tests and property-based tests.
- Run lint, type-check, and tests in CI on every push.

The point of the exercise is the template, not the service. Once the template exists, it can be copied into any repository and the agent loop behaves predictably.

## Step 1 — Pin the environment before the agent touches it

Install dependencies with exact versions and commit the lockfile. Agents frequently suggest upgrades to "latest," and a version bump that lands mid-task can invalidate the code they just wrote. Pinning removes that variable.

```bash
mkdir agent-labeler && cd agent-labeler
git init
npm init -y
npm pkg set type="module"

npm install --save-exact \
  typescript@5.5.4 \
  zod@3.23.8 \
  fastify@4.26.1 \
  @fastify/type-provider-zod@2.0.0 \
  @octokit/rest@21.0.2 \
  p-retry@6.2.0 \
  opossum@8.4.0 \
  zod-to-json-schema@3.23.5

npm install --save-dev --save-exact \
  tsx@4.19.1 \
  eslint@9.6.0 \
  @typescript-eslint/parser@7.13.1 \
  prettier@3.3.2 \
  tap@18.7.1 \
  sinon@17.0.0 \
  @sinonjs/fake-timers@11.2.2 \
  fast-check@3.15.1
```

Two notes on version choice. Pin the GitHub client to a major version you have actually tested against the API; the Octokit API surface changes between majors. Pin the retry and circuit-breaker libraries for the same reason — their option names have changed across releases, and an agent trained on an older example will happily use the old names.

Add a minimal server so there is something for the agent to extend:

```typescript
// src/server.ts
import Fastify from 'fastify';
import { z } from 'zod';
import { serializerCompiler, validatorCompiler } from 'fastify-type-provider-zod';

const server = Fastify({ logger: true });
server.setValidatorCompiler(validatorCompiler);
server.setSerializerCompiler(serializerCompiler);

const IssueSchema = z.object({
  owner: z.string().min(1),
  repo: z.string().min(1),
  issue_number: z.number().int().positive(),
});

server.post('/issues', { schema: { body: IssueSchema } }, async (req, reply) => {
  const { owner, repo, issue_number } = req.body;
  reply.send({ ok: true, owner, repo, issue_number });
});

server.get('/health', async () => ({ status: 'ok' }));

const start = async () => {
  try {
    await server.listen({ port: 3000, host: '0.0.0.0' });
  } catch (err) {
    server.log.error(err);
    process.exit(1);
  }
};
start();
```

Sanity-check it before involving an agent:

```bash
npx tsx src/server.ts
curl -X POST http://localhost:3000/issues \
  -H 'Content-Type: application/json' \
  -d '{"owner":"octocat","repo":"Hello-World","issue_number":42}'
```

If the endpoint returns `{"ok":true,...}`, the baseline is sound. If it does not, fix it now — debugging a broken baseline through an agent's output is significantly harder than debugging it directly.

## Step 2 — Give the agent a project instruction file

Most agentic CLIs read a project-level instruction file. Keep it short and specific. Long instruction files get partially ignored, and vague instructions produce vague code.

```markdown
# Project instructions

- TypeScript strict mode is on. Do not disable it.
- Prefer functions over classes. No inheritance.
- Application code lives under `src/`. Tests live under `test/`.
- Tests use `tap`. Mocking uses `sinon`. Do not introduce Jest.
- Every network call must have a timeout.
- Secrets are read from `process.env` and never written to disk.
- Commit messages follow Conventional Commits.
- Do not add dependencies without listing them in the prompt first.
```

That last rule matters more than it looks. Without it, agents routinely import packages that are not installed, or that do not exist at all. Requiring the dependency list up front turns a silent hallucination into a visible line in the prompt.

## Step 3 — First generation: expect it to be incomplete

A reasonable first prompt:

```
Write the following, using only the dependencies already installed:

1. `src/clients/github.ts` — a thin wrapper around @octokit/rest that
   exposes addLabel(token, owner, repo, issueNumber).
2. `src/services/label.ts` — validates inputs, reads GITHUB_TOKEN from
   process.env, throws a descriptive error if missing, and calls the client.
3. `test/label.test.ts` — tap tests covering: success, missing token,
   invalid owner/repo characters, and a 429 response from the API.
4. `.github/workflows/ci.yml` — runs lint, type-check, and test on push.

Start with the failing tests, then the implementation.
```

The generated client is usually fine. The generated service typically looks like this:

```typescript
// src/services/label.ts
import { addLabel as clientAddLabel } from '../clients/github.js';

export async function addLabel(
  owner: string,
  repo: string,
  issueNumber: number,
) {
  const token = process.env.GITHUB_TOKEN;
  if (!token) {
    throw new Error('GITHUB_TOKEN environment variable is required');
  }
  return clientAddLabel(token, owner, repo, issueNumber);
}
```

That is close to correct, and the missing-token check is a good sign — it suggests the instruction file was read. What is usually missing:

- No retry on transient failures.
- No timeout on the underlying HTTP call.
- Validation of `owner` and `repo` against GitHub's actual naming rules.
- No handling for the case where the label already exists.

The review pass is where you add those. Budget for it.

## Step 4 — Add resilience, and verify the semantics

Retries and circuit breakers are where agents most often produce code that looks right and behaves wrong. Two specific traps:

**Off-by-one retries.** Most retry libraries treat `retries: 3` as three *additional* attempts after the first, for four total. Agents frequently write a test that asserts three total attempts, then "fix" the library call to match. Decide which semantics you want and state it explicitly.

**Circuit breakers that reopen too soon.** A breaker with a short `resetTimeout` will let traffic through while the upstream is still degraded, generating a second wave of failures. If your upstream outages typically last longer than the default reset window, raise the timeout and add jitter.

A retry wrapper with explicit semantics:

```typescript
// src/lib/retry.ts
import retry from 'p-retry';

export async function withRetry<T>(
  fn: () => Promise<T>,
  { retries = 3, minTimeout = 100, maxTimeout = 5_000 } = {},
): Promise<T> {
  return retry(fn, {
    retries,
    minTimeout,
    maxTimeout,
    factor: 2,
    onFailedAttempt: (err) => {
      console.warn(
        `Attempt ${err.attemptNumber} failed: ${err.message}. ` +
          `${err.retriesLeft} retries left.`,
      );
    },
  });
}
```

And a circuit breaker around the client call:

```typescript
// src/services/label.ts
import CircuitBreaker from 'opossum';
import { addLabel as clientAddLabel } from '../clients/github.js';
import { withRetry } from '../lib/retry.js';

const breaker = new CircuitBreaker(
  (token: string, owner: string, repo: string, n: number) =>
    withRetry(() => clientAddLabel(token, owner, repo, n)),
  {
    timeout: 3_000,
    errorThresholdPercentage: 50,
    resetTimeout: 60_000,
    volumeThreshold: 5,
  },
);

export async function addLabel(
  owner: string,
  repo: string,
  issueNumber: number,
) {
  const token = process.env.GITHUB_TOKEN;
  if (!token) throw new Error('GITHUB_TOKEN environment variable is required');
  return breaker.fire(token, owner, repo, issueNumber);
}
```

To verify this behaves as intended, do not trust a prose description of the outcome. Instrument it and measure. A minimal harness:

1. Wrap the GitHub client in a stub that sleeps for a configurable duration and fails with a configurable probability.
2. Run the service against the stub with the breaker disabled, then enabled.
3. Record p50 and p95 latency and the failure rate for each configuration.
4. Repeat with a stub that fails continuously for 30 seconds, and log every breaker state transition with a timestamp.

That last step is the one that catches a too-short `resetTimeout`. If the breaker closes while the stub is still failing, you will see it in the transition log immediately. The numbers you get will depend entirely on your stub parameters, so record the parameters alongside the results — a latency figure without the simulated upstream behavior is not reproducible.

## Step 5 — Tests that find real bugs

Unit tests generated by an agent tend to assert the happy path and the two or three obvious errors. Property-based tests are where the agent earns its keep, because they force it to enumerate invariants rather than examples.

A prompt that produces useful output:

```
Write property-based tests using fast-check that verify:

- addLabel rejects any owner or repo that does not match GitHub's
  naming rules (alphanumeric, hyphen, underscore; max 100 chars).
- withRetry makes exactly (retries + 1) attempts on a persistently
  failing function.
- withRetry succeeds on the first attempt when the function succeeds.
- The circuit breaker transitions to open after the configured
  error threshold is exceeded.
```

Two classes of bug commonly surface here. The first is a validation regex that is too permissive — often it checks length but not the character class, so `owner: "a/b"` passes. The second is a retry-count assertion that was written against the wrong convention, which the property test exposes as a mismatch between the stated invariant and the implementation.

Neither bug is exotic. Both are the kind of thing that reaches production when the test suite only covers examples the author thought of.

## Step 6 — Observability, because the agent will not add it

Agents rarely add metrics unless asked, and when asked they tend to add the easy ones. The minimum useful set for a service like this:

- HTTP request duration as a histogram, with explicit buckets. The default Prometheus buckets are tuned for sub-second latencies and will not resolve a multi-second upstream stall.
- Upstream call duration, labelled by outcome.
- Circuit breaker state, exported as a gauge (0 = closed, 1 = half-open, 2 = open).
- Retry attempts, as a counter labelled by attempt number.

The circuit breaker state gauge is the one that pays for itself. A breaker that has been open for ten minutes is a signal you want on a dashboard, not buried in logs.

Wire the metrics endpoint to the same Fastify instance, and add a CI check that the endpoint returns a non-empty body. That check catches the common failure where the metrics library is imported but never registered.

## A decision checklist before merging agent output

Run through this before the pull request goes up:

- **Are all dependencies pinned, and does the lockfile match?** Run `npm ls` or `pip check`. Any missing or duplicated entry is a red flag.
- **Does every network call have a timeout?** A missing timeout is the single most common cause of a hung request handler.
- **Are retry semantics stated explicitly, and does the test match?** Check the total attempt count, not just that a retry happened.
- **Is the circuit breaker reset window longer than a typical upstream outage?** If you do not know the typical outage length, measure it before choosing.
- **Do tests cover the failure modes, not just the happy path?** Missing token, invalid input, upstream 429, upstream timeout.
- **Are secrets read from the environment and never logged?** Grep the diff for the token variable name.
- **Is there a metric or log line for every failure path?** If a failure is silent, it is invisible.

If any answer is no, the second pass is not finished.

## FAQ

**Does this workflow depend on a specific agent product?**
No. It depends on three properties: the agent reads a project instruction file, it writes into a target directory you control, and you can review the diff before merging. Any tool with those properties supports the same loop.

**How do you stop the agent from inventing dependencies?**
Require the dependency list in the prompt, pin exact versions, and run a dependency check in CI. A missing or duplicated package should fail the build.

**What about languages other than TypeScript?**
The structure transfers. The specific libraries differ — for Python, a schema library, a retry helper, and a circuit-breaker package play the same roles — but the review checklist is identical. The main difference is that agent-generated Go tends to be more verbose and more likely to need manual restructuring, because the language has fewer idioms the model can lean on.

**Should the agent write the tests first?**
Yes. Tests written first constrain the implementation to something the agent can verify, and they make the review pass faster because you can see what the agent believed the contract was.

**How much human review time should be budgeted?**
Enough to run the checklist above. For a service of this size, that is typically a short review pass per generated file, plus longer for anything touching retries, timeouts, or secrets. Treat the review as part of the task, not as overhead on top of it.

## What to do in the next 30 minutes

Pick one repository you own. Create a project instruction file at the root with the rules from Step 2, adjusted for your stack. Then, before running any agent, list every dependency the task will need with exact versions and install them. Commit that state. You now have a baseline that is pinned, reviewable, and reproducible — which is the precondition for everything else in this article.
