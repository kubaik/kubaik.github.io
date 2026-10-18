# Contract testing is not optional

## The problem contract testing solves

Services break each other most often not through network partitions or bad deploys, but through quiet shape changes: two teams agree on a JSON response in a chat thread, one team adds a field and renames another, and a consumer breaks days later in a part of the system nobody was watching. Contract testing is the discipline that makes the consumer's assumptions explicit and verifiable before the provider ships.

This article is for teams running multiple services that have already felt the sting of a consumer breaking after a provider deploy. It covers what the common advice gets wrong, the mental model that fixes it, when contract testing is the wrong tool, and how to start without turning CI into a second job.

## The conventional wisdom, and where it stops short

The standard advice comes in three layers, and each one helps a little while missing the same thing.

**End-to-end tests.** Spin up the whole stack in staging, hit the API, assert the response. If it passes, the reasoning goes, the services are integrated. This is the advice in most tutorials and most conference talks. It is also the advice from people who have never maintained a staging environment with eleven services and a database seeded by a script only one person understands.

**API versioning.** Use `/v2/`, deprecate `/v1/` on a schedule, publish a changelog. Better than nothing, but versioning is a coordination mechanism, not a verification mechanism. It signals that something *might* have changed. It does not tell anyone whether a consumer's assumptions still hold. A provider can add a required field to `/v2/` and break every consumer without touching `/v1/`. Versioning looks like a contract but is only a label.

**Schema validation.** Keep an OpenAPI spec or schema registry and validate against it. Closer, but a static spec checked at build time only proves the provider *intends* to return a shape. It does not prove the consumer *depends on* that shape. Remove a field nobody uses and the spec check may block a harmless change. Change a field's semantics without changing its type — `amount` switching from cents to dollars — and the spec check passes while the consumer computes wrong totals for a week.

The shared flaw: all three treat integration as a property of the provider. Integration is a property of the *relationship* between provider and consumer. Contract testing is the practice that makes that relationship explicit and testable.

## What happens when you rely on end-to-end tests alone

A typical failure mode: Team A owns the orders service, Team B owns notifications. Notifications calls orders to fetch order details before sending a receipt. The staging end-to-end test passes because staging runs a version of orders that returns `{ id, total, currency, customer_email }`. Team A renames `customer_email` to `email` for a new frontend. The staging test still passes, because staging's notifications build is older and hasn't been redeployed, and the test fixture mocks the orders response anyway. Production breaks; receipt emails go out with `undefined` where the greeting should be.

The general shape of this bug: the test that was supposed to catch the break was testing a *copy* of the integration. Staging is a copy. Mocks are copies. The only artifact that isn't a copy is the consumer's actual expectations, encoded somewhere the provider's CI can run against.

A second failure mode: the end-to-end suite grows slow and flaky, teams stop trusting it, and merges happen on red because "it's probably the flaky one." Integration breakage ships anyway, and the signal that would have caught it is gone.

A third: the provider team doesn't know who its consumers are. A change breaks something, the incident channel fills with people from a team the provider has never heard of, and there was no way to run the consumers' checks because the consumers never published them anywhere reachable.

## A different mental model: the contract is consumer-owned

The shift that makes this work: **the contract belongs to the consumer.** The provider publishes an API. Each consumer writes down the subset of that API it actually depends on — fields read, status codes handled, headers required — and that written-down subset becomes a test the provider must pass before shipping.

This inverts the usual direction. Instead of the provider publishing a spec and hoping consumers conform, consumers publish their expectations and the provider verifies against them. The provider's CI runs every consumer's contract; a change that would break a consumer turns the build red before the change ships, and the output names the consumer and the field.

This is the model behind consumer-driven contract testing tools. Pact is the best-known open-source implementation, and Pact Broker is the component that stores contracts and tracks which provider versions have verified which consumer versions. The model is tool-agnostic: a crude version can be built from JSON fixtures in a shared repository plus a CI job that replays them. The discipline matters more than the tool.

Four properties define the model:

1. **Contracts are narrow.** A consumer records only what it uses. If it reads `id` and `total` and ignores `currency`, the contract says nothing about `currency`, so the provider can change it freely. Broad contracts ("response matches this full schema") recreate the brittleness you were trying to escape.

2. **Contracts are versioned with consumer code.** When a consumer stops using a field, the contract changes in the same pull request. When it starts using a new field, the contract changes and the provider's next build verifies it. The contract is a test artifact, not a separate document that drifts.

3. **Verification runs in the provider's CI.** No spin-up, no seeding, no flaky network. The provider's test suite replays each consumer's recorded expectations against the provider's code, usually a test server rather than a full deployment.

4. **Breaking changes are visible before merge.** The provider sees "this change breaks notifications-service contract v3" in the pull request, not in a production incident three days later.

## What a contract looks like

Pact's documentation describes the flow: consumers generate pact files (JSON documents describing requests and expected responses), publish them to a broker, and providers run a verification step that replays those requests against the real provider and asserts the responses match. The broker tracks which provider versions have verified which consumer versions, so "is this provider version safe to deploy?" becomes a lookup rather than a guess.

A minimal consumer-side contract in Pact's JavaScript DSL looks roughly like this. The exact API varies by language and binding version, so check the docs for yours.

```javascript
const { PactV3, MatchersV3 } = require('@pact-foundation/pact');
const { like, integer } = MatchersV3;

const provider = new PactV3({
  consumer: 'notifications-service',
  provider: 'orders-service',
});

describe('orders contract', () => {
  it('returns order details for a receipt', async () => {
    provider
      .given('order 123 exists')
      .uponReceiving('a request for order 123')
      .withRequest({ method: 'GET', path: '/orders/123' })
      .willRespondWith({
        status: 200,
        body: {
          id: like('123'),
          total: integer(4500),
          email: like('a@b.com'),
        },
      });

    await provider.executeTest(async (mockServer) => {
      const res = await fetch(`${mockServer.url}/orders/123`);
      const body = await res.json();
      expect(body.email).toBeDefined();
      expect(body.total).toBeGreaterThan(0);
    });
  });
});
```

The matchers are the important part. `like` and `integer` say "this field exists and has this type," not "this field equals this exact value." A provider that changes `total` from `4500` to `4600` still passes. A provider that changes `total` to a string fails, correctly.

On the provider side, verification is typically a test that loads each consumer's pact file and replays the recorded requests against a running instance, often a test server rather than a full deployment. The provider wires this into CI as a required check. Failure output names the consumer and the mismatched field, which is what makes it actionable.

## A worked example: removing a field

Suppose orders-service wants to remove `email` because a new `contact` object replaces it. The provider opens a pull request. CI runs the notifications-service contract, which expects `email`. The build fails. The provider now has three options:

- **Keep `email` and add `contact` alongside it.** Backward compatible; no consumer changes needed. Cost: the field lives longer than the provider would like.
- **Coordinate a two-step deploy.** Notifications updates its contract to use `contact` first, publishes it, and then orders removes `email`. Cost: sequencing across teams.
- **Accept the break and version the endpoint.** Serve `/v2/` without `email`, keep `/v1/` until notifications migrates. Cost: two endpoints to maintain.

The reasoning to show explicitly: the contract did not make the change impossible. It moved the moment of discovery from a production incident to a pull request, where someone can still choose among the three options. That timing shift is the entire value proposition. Whether the provider ultimately removes the field matters less than the fact that the decision was made deliberately.

## Measuring whether it helps

Claims about "fewer incidents" are easy to make and hard to defend. What can be measured, and how:

- **Lead time to detect a breaking change.** Instrument your CI to record, for each provider pull request, whether verification ran and whether it failed. Compare the timestamp of the failure against the timestamp of the merge for the same change. The useful number is how often a breaking change reaches the default branch without a failed check.
- **End-to-end suite runtime and flake rate.** Record wall-clock duration and pass/fail per run over a fixed window (say, the last 200 runs). If contracts let you delete end-to-end cases, both numbers should move. If they don't, the contracts aren't covering what the end-to-end tests covered.
- **Incidents attributable to integration shape changes.** Tag incidents by cause in your tracker. Count those caused by a consumer misreading a provider response over a quarter before and after adoption. This requires disciplined tagging, which is worth doing anyway.
- **Time to add a new consumer.** Measure from "consumer team starts integrating" to "first contract verified in provider CI." This is the number that tells you whether the setup is sustainable or whether every new consumer is a project.

Run the comparison over a fixed window with the same services in scope. A before/after across a reorg or a major platform migration tells you nothing.

## When the conventional wisdom is right

Contract testing does not replace everything, and pretending otherwise would be dishonest.

**UI flows and cross-service user journeys.** Contracts verify that service A's response matches what service B expects. They do not verify that a user can log in, add an item to a cart, and check out. Keep a small number of end-to-end tests for critical paths. The mistake is having *only* end-to-end tests, or hundreds of them because the contracts aren't trusted.

**Infrastructure and configuration.** A missing environment variable, a misconfigured load balancer, or a migration that didn't run is invisible to a contract test. Those need post-deploy smoke tests, health checks, and canary analysis.

**Third-party APIs.** You can write consumer-side tests against a vendor's documented responses, but the vendor will not run them. The mitigation is to pin to documented API versions and monitor response shapes in production.

**Very small systems.** With two services owned by one team, a shared test suite that imports both and asserts the integration is simpler and catches the same bugs. Contract testing earns its keep when ownership is split and the provider can't easily run the consumer's code.

**Event-driven and async systems.** Message contracts are possible — Pact supports them — but they are more complex than request/response because ordering and delivery guarantees matter. Budget more setup time and expect rougher edges.

## A decision checklist

| Signal | Lean toward contract testing | Lean toward shared integration tests |
|---|---|---|
| Service ownership | 2+ teams, separate ownership | 1 team owns both sides |
| Provider change frequency | Weekly or more | Rarely |
| Cost of a production break | High (revenue, trust, compliance) | Low (internal tool, easy rollback) |
| Staging reliability | Flaky, slow, hard to seed | Fast, reliable, representative |
| Consumers per provider | 3 or more | 1 |
| CI maturity | Existing required checks | Nothing beyond running tests locally |

Three or more left-column checks make contract testing likely to pay for itself. Mostly right-column, start with a shared integration suite and revisit when ownership splits.

The cost side, honestly: expect a few days of setup for the first provider-consumer pair — installing tooling, wiring a broker or a shared repository, and getting provider verification into CI as a required check. After that, a new consumer is mostly one contract file and a publish step. The ongoing cost is keeping contracts narrow. Teams drift toward recording full responses because it is easier, and that drift recreates the brittleness the practice was meant to remove.

## Common objections

**"Our API is internal, we don't need contracts."** Internal APIs are where this matters most, because there is no external pressure to keep them stable. External APIs have customers who complain; internal APIs have consumers who silently break. The absence of a contract makes internal changes more dangerous, not less.

**"We use OpenAPI, isn't that a contract?"** An OpenAPI spec describes what the provider *can* return. A contract describes what the consumer *requires*. These are different sets, and the gap between them is where breakage lives. An OpenAPI spec is a reasonable starting point for generating a consumer contract, but the consumer still has to narrow it to what it actually uses.

**"This is just more tests to maintain."** It is more tests, but they fail for one reason: a real integration break. They don't flake on network, they don't need a database, and they run in seconds. The maintenance cost is usually lower than the end-to-end suite they let you shrink.

**"Our provider team won't run consumer contracts."** Then the practice won't work, and that is an organizational problem, not a tooling one. Contract testing requires the provider team to accept a CI check they don't control. If that is a non-starter, the system is back to staging and hope. The conversation about *why* a fast, deterministic check is unwelcome is more valuable than the tooling decision.

**"We tried a full contract-testing setup and it was too complex."** Broker, versioning, and webhooks are genuinely a lot for a small team. The crude version — JSON fixtures in a shared repository, a CI job that replays them — captures most of the value with a fraction of the setup. Start crude and upgrade when coordination overhead grows.

## What changes in practice

With end-to-end-only testing, a provider team's deploy conversation is "did staging pass?" — a question about a copy. With contract testing, it is "did every consumer's contract verify?" — a question about the actual relationships, answerable in the pull request by the person making the change.

It also changes what a breaking change costs. A team that renames a field finds out from a red CI check naming the consumer and the field, rather than from an incident. The change may still happen; it stops being a surprise.

And it changes the end-to-end suite. Once contracts cover the seams, end-to-end tests can shrink to the handful of user journeys that need a full stack. The intended outcome: contracts for the seams, end-to-end for the journeys, staging for what neither can catch.

## FAQ

**How is contract testing different from integration testing?** Integration testing typically spins up real services and tests them together, which is slow and environment-dependent. Contract testing records the consumer's expectations and replays them against the provider in isolation, so it runs fast and deterministically in CI. The tradeoff is coverage: contracts cover request/response shapes, not full system behavior.

**Do I need a broker?** No. A broker makes it easier to share contracts across teams and track which versions have verified, but a shared Git repository with JSON fixtures and a CI job that replays them captures most of the value for a small team. Add a broker when coordination overhead grows.

**What happens when a consumer's contract is wrong?** Provider verification fails, and either the consumer fixes the contract or the provider accommodates it. That is the point: a wrong contract is a wrong assumption about the API, and catching it before deploy is cheaper than catching it in production. Failure output should name the consumer so the right team gets pulled in.

**Can contract testing catch breaking changes in event-driven systems?** Yes, with more effort. Message contracts let a consumer record the message shape it expects and the provider verify it can produce that shape. Async contracts are harder to reason about than request/response because ordering and delivery guarantees matter.

## Summary

Contract testing makes the consumer's expectations explicit and verifies them in the provider's CI. It doesn't replace end-to-end tests; it lets you shrink them to the journeys that need a full stack. It doesn't replace API versioning; it makes versioning decisions informed by actual consumer dependencies. The setup cost is real — a few days for the first pair — and the failure mode it prevents, a silent break in a service nobody was watching, is among the most expensive kinds of integration bug.

Your next step, takeable in the next thirty minutes: open the CI config for your most-changed provider service and check whether any job verifies a consumer's expectations. If not, pick one consumer, write a single JSON fixture describing the fields it reads from one endpoint, and add a CI step that asserts the provider's response includes those fields. That is a crude contract. It will catch the next rename before it ships.
