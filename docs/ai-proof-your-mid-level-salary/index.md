# AI-proof your mid-level salary

Junior-heavy teams tend to have a pyramid shape. At the bottom sits repetitive, well-defined work that scales linearly with headcount: CRUD endpoints, mock-based unit tests, schema scaffolding. Large language models are genuinely good at that bottom layer. The top of the pyramid — high-uncertainty, high-impact decisions — still needs human judgment. The squeeze on mid-level salaries is less about AI replacing developers and more about AI absorbing the lowest-value slice of the development lifecycle while budgets for that slice disappear.

The practical consequence is a mismatch between job titles and actual scope. A developer titled "mid-level engineer" may still write CRUD endpoints, but is increasingly expected to own observability, cost control and architectural trade-offs. The differentiator is not typing speed. It is the ability to make decisions that are expensive to reverse and to measure whether those decisions worked.

If your work is still described as a series of tickets, you are competing directly with tooling that produces tickets at near-zero marginal cost. The rest of this article is about the alternative: instrumenting your own work in outcome terms, owning more than one layer, matching compute to workload, and writing down decisions so they survive you.

## Why output metrics fail you

Commits, pull requests and story points measure production, not effect. They are also the easiest things for an AI assistant to inflate. An engineer who merges forty small PRs generated from a scaffold looks productive on a velocity chart and may have changed nothing about latency, cost or reliability.

Outcome metrics are harder to game because they are measured at the system boundary, not at the editor:

- **Latency under load** — p50, p95 and p99 for your endpoints during peak traffic, not in a local benchmark.
- **Cost per request** — total infrastructure spend divided by requests served, over the same window.
- **Error rate** — failed requests per thousand, broken down by endpoint and status class.
- **Retention or conversion delta** — the product metric your change was supposed to move.

The point is not that these numbers are inherently fair. It is that they force you to state a hypothesis ("this change should reduce p99 because it removes a synchronous call"), make the change, and then either confirm or falsify it. That loop is what senior-level interviews and promotion packets are actually about, and it is the part an AI assistant cannot run on your behalf.

### A worked example of the reasoning

Suppose an endpoint serves 3 million requests per month. p99 is 1.4 s and the monthly bill attributable to that service is $600. Cost per request is therefore 600 / 3,000,000 = $0.0002. Two candidate fixes:

1. Add a cache in front of the slowest query. Estimated hit rate 80%, estimated p99 after the change 300 ms, added infrastructure cost $40/month.
2. Rewrite the query to use a covering index. Estimated p99 after the change 500 ms, no added infrastructure cost, two days of engineering.

Option 1 gives a better latency number but raises cost per request to 640 / 3,000,000 ≈ $0.000213. Option 2 keeps cost flat and improves latency substantially. If the product is latency-sensitive and margin is comfortable, option 1 may be correct. If the product is margin-sensitive, option 2 is correct. The decision depends on which constraint binds — and you cannot have that conversation credibly without the numbers.

Note that none of these figures come from a benchmark table. They come from your own billing export and your own metrics backend. That is the only kind of number worth quoting in a salary conversation.

### How to measure it

Instrument the boundary, not the internals. For an HTTP service, record request duration as a histogram with buckets chosen around your SLO, labelled by route and status code. Derive p99 from the histogram rather than logging individual timings, because percentile-of-averages is a common and silent error.

For cost, the reliable source is the cloud provider's billing export, grouped by service or tag. Divide by request count from the same period. A short script that pulls both and writes a single gauge is enough to start; the value is in the trend line, not the dashboard's visual polish.

For errors, count at the same boundary so that latency and error metrics share a denominator. If your error rate is measured on a different population than your latency, the two numbers will disagree during incidents and you will waste time reconciling them.

Set alerts on the outcome, not the cause: p99 above your SLO for two consecutive evaluation periods, cost per request above a threshold you have agreed with whoever owns the budget, error rate above a fraction of a percent. An alert that fires on CPU utilisation is a cause-based alert and will page you for conditions that do not affect users.

| Metric | What to instrument | Where the number comes from | Failure mode if unmeasured |
|---|---|---|---|
| p99 latency | Request duration histogram at the service boundary | Metrics backend query over a fixed window | You optimise averages and miss the tail users actually feel |
| Cost per request | Provider billing export divided by request count | Billing export grouped by service or tag | Cost regressions ship silently and surface at invoice time |
| Error rate | Counter of failed requests by route and status | Same boundary as latency | Incidents are discovered by users, not by you |
| Retention or conversion | Product analytics on the affected cohort | Product analytics tool | You cannot tell whether a technically correct change helped |

## Owning more than one layer

A backend engineer who waits for a frontend colleague to confirm that a change works has outsourced part of their own verification. A frontend engineer who assumes the backend will absorb a new request pattern has done the same. The integration between layers is exactly where AI-generated code tends to be weakest, because the model sees each side in isolation and rarely sees the contract drift between them.

Expanding scope by one adjacent layer is usually enough to change how you are perceived:

- Backend: own the API contract, the schema, and the integration tests that exercise both.
- Frontend: own the UI, the request orchestration, and the performance budget.
- Platform or DevOps: own the infrastructure and the application behaviour that runs on it.

The mechanism is not prestige. It is that you can now debug failures that cross the boundary, which is where the expensive incidents live. An end-to-end test that fails the build when the contract breaks is the cheapest possible version of that ownership.

```javascript
// Example: Playwright test covering API + frontend integration
import { test, expect } from '@playwright/test';

test('checkout flow', async ({ page }) => {
  await page.goto('/login');
  await page.fill('#email', 'user@example.com');
  await page.fill('#password', 'password');
  await page.click('button[type="submit"]');

  await page.waitForURL('/products');
  await page.click('.product:first-child');
  await page.click('text="Add to Cart"');
  await page.click('text="Checkout"');

  // Fails if the API returns a 5xx or the contract drifts
  await expect(page.locator('.order-confirmation')).toBeVisible();
});
```

Two caveats worth stating plainly. First, this test is only as good as its selectors; a test that asserts on a CSS class will break on a styling change and train the team to ignore failures. Prefer role- or test-id-based selectors. Second, an end-to-end test that runs against a shared staging environment will be flaky for reasons unrelated to your change. Run it against an ephemeral environment seeded per build where possible.

## Matching compute to workload

The compute model you choose determines both your cost curve and your latency floor. The trade-off is structural, not a matter of one service being cheaper than another:

- **Serverless functions** bill per invocation and per unit of duration, scale to zero, and pay a cold-start penalty on the first request after idle. They suit sporadic or bursty traffic where paying for idle capacity is wasteful.
- **Container services** bill for provisioned CPU and memory per second, have no cold start, and have a higher baseline cost because you pay while idle. They suit steady traffic.
- **Fixed instances** have the lowest per-unit cost at high, predictable utilisation and the highest operational burden, since you own scaling and patching.

The correct choice follows from your traffic shape, not from a general preference. A useful procedure: plot requests per second over a week, note the ratio of peak to median, and note how much of the day sits near zero. A peak-to-median ratio above roughly ten with long idle troughs favours serverless. A flat profile favours provisioned capacity.

The documented AWS Lambda pricing for arm64 x86-equivalent compute is billed in 1 ms increments with a per-request charge; the exact figures change and should be read from the current pricing page rather than quoted from memory. The important arithmetic is the comparison, and you can do it with your own numbers:

1. Measure your median and peak requests per second over a representative week.
2. Measure your median and peak request duration.
3. Compute provisioned capacity cost as (vCPU-seconds × price) + (GB-seconds × price) for the capacity needed at peak, times 730 hours.
4. Compute serverless cost as (invocations × per-request price) + (GB-seconds consumed × duration price).
5. Add the latency cost of cold starts if your SLO is tight.

Step 5 is the one teams skip. If your p99 SLO is 300 ms and cold starts add 400 ms, serverless is disqualified for that endpoint regardless of cost, unless you keep a minimum number of instances warm — which reintroduces the idle cost you were avoiding.

A common failure mode is a workload that is cheap on serverless at low traffic and becomes both expensive and slow at high traffic, because concurrency limits cause queuing. The symptom is a latency curve that is flat until a threshold and then vertical. If you see that shape, the fix is usually to move the hot path to provisioned capacity and leave the cold path serverless.

## Verifying that a change actually worked

The verification loop is the same for every change:

1. Record the metric before the change over a window long enough to include a normal traffic cycle.
2. State the expected effect and the threshold at which you would call the change a failure.
3. Deploy.
4. Compare the same window after the change, accounting for traffic volume differences.
5. Write down the result, including negative results.

Step 5 is the one that compounds. A repository of short decision records is the difference between an engineer who has five years of experience and one who has one year of experience five times. The template is deliberately small:

```markdown
# ADR-001: Cache user sessions in a key-value store

## Context
Session reads hit the primary database on every request. p99 latency for
authenticated endpoints was above the SLO during peak traffic.

## Decision
Store sessions in a key-value store with a one-hour TTL. The database remains
the source of truth for user records.

## Alternatives considered
- Increase the database connection pool: raises memory pressure and does not
  remove the round trip.
- Cache in process memory: no shared state across instances, so sessions are
  lost on deploy.

## Consequences
- p99 for authenticated endpoints should fall; verify against the same window.
- Adds one more stateful dependency to operate and to back up.
- Session invalidation now requires a write to the cache as well as the database.
```

The consequences section is where most ADRs fail. "Improves performance" is not a consequence. "Adds a stateful dependency that must be backed up, and makes logout require two writes" is a consequence, and it is the part a future reader needs.

Automating incident response is the natural extension. If the documented runbook for a known failure is "scale the service up," that can be an alarm action rather than a page. The value is not that the system heals itself perfectly; it is that the human is only woken for failures that have no known remedy.

```yaml
# Example: CloudFormation alarm that triggers a scaling action
Resources:
  HighLatencyAlarm:
    Type: AWS::CloudWatch::Alarm
    Properties:
      AlarmName: HighLatencyAlarm
      ComparisonOperator: GreaterThanThreshold
      EvaluationPeriods: 2
      MetricName: Latency
      Namespace: AWS/ApiGateway
      Period: 60
      ExtendedStatistic: p99
      Threshold: 500
      ActionsEnabled: true
      AlarmActions:
        - !GetAtt ScaleUpFunction.Arn

  ScaleUpFunction:
    Type: AWS::Lambda::Function
    Properties:
      Runtime: python3.12
      Handler: index.handler
      Code:
        ZipFile: |
          import boto3
          def handler(event, context):
              ecs = boto3.client('ecs')
              ecs.update_service(
                  cluster='my-cluster',
                  service='my-service',
                  desiredCount=2
              )
      Role: !GetAtt LambdaRole.Arn
```

Two honest caveats. `ExtendedStatistic: p99` requires the underlying metric to be published as an extended statistic, which API Gateway latency is not by default — you will typically need a custom metric or a metric math expression. And an auto-scaling action that only ever scales up will, over time, leave you running at maximum capacity. Pair it with a scale-down condition or a scheduled action.

## Common failure modes

**Local and production divergence.** Configuration, environment variables, dependency versions and data volumes differ. The fix is to make the local environment as close to production as is practical and to run the same container image in both. This is unglamorous and it removes an entire class of "it worked on my machine" incidents.

**Happy-path assumptions.** Tests written against the happy path pass while the system fails on the first malformed input from a real user. Property-based tests and a small amount of fuzzing on input boundaries catch more of these than additional example-based tests.

**Dependency rot.** A library stops receiving updates or a managed service is deprecated. The mitigation is not to avoid dependencies but to know which ones are load-bearing: keep an inventory of your direct dependencies, note the maintenance status of the critical ones, and have a migration path sketched before you need it.

**Deferred maintenance.** Cutting a corner to hit a deadline is sometimes correct. Doing it without recording the debt is not, because the cost is paid later by someone who does not know why the code looks the way it does. An ADR that says "we chose the simpler approach to ship by the deadline; revisit if traffic doubles" converts an invisible liability into a scheduled decision.

## When the problem is leverage, not skill

Improving your metrics and scope will make you more effective in your current role. It does not automatically change what that role pays, because compensation is bounded by the budget of the organisation and the market it hires in. Those are separate problems.

If your outcomes are strong and compensation is flat, the options are structural: move to a role with a larger scope of responsibility, work in a market that pays more for the same work, take equity in exchange for below-market cash, or build something you own. Each has a different risk profile, and the honest summary is that all of them take more than a weekend. The one thing that does not work is continuing to produce output metrics and hoping the budget changes.

## FAQ

**Does AI actually reduce junior hiring?**
The observable pattern is that the well-specified, easily verified tasks that used to be junior work are increasingly produced by tooling, so fewer people are needed to complete the same volume. The effect on any specific market depends on local labour costs, regulation and how much of the work is genuinely well-specified. Treat broad claims about percentages with suspicion; look at the job listings in your own market over time.

**How do I tell whether my work is automatable?**
Ask whether the task can be fully specified in writing, verified automatically, and completed without access to context that is not in the repository. If all three are true, assume it will be automated. If any is false, the task depends on judgment that is currently hard to replace.

**What if I cannot get access to cost or retention data?**
Start with what you can measure at the service boundary: latency, error rate, throughput. Those are usually available to any engineer with access to the metrics backend. For cost, a rough proxy is resource utilisation multiplied by published unit prices. For product metrics, ask the person who owns them; the request itself signals that you are thinking in outcome terms.

**Is this advice only for engineers in high-cost markets?**
No. The arithmetic is the same everywhere; only the absolute numbers differ. The relative gap between an engineer who can reason about latency, cost and reversibility and one who cannot is a market-independent effect.

## Do this in the next 30 minutes

Pick one endpoint you own. Query its p99 latency and its request count for the last seven days from your metrics backend, and its attributable cost for the same period from your billing export. Divide cost by requests to get cost per request. Write those three numbers in a file in your repository with today's date and a one-line note about what you expect to change. That file is the first entry in the record that will eventually make your impact legible to someone other than you.
