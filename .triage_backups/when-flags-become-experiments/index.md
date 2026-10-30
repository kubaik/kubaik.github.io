# When flags become experiments

Most guidance on experimentation platforms assumes a data warehouse, a JavaScript-heavy frontend, and a monthly budget measured in thousands. That assumption breaks for teams whose users are on 2G feature phones, whose backend runs on a single small VPS, and whose only conversion channel is an inbound SMS reply. This article is about the platform category that actually fits that environment: self-hosted, offline-tolerant flag and experimentation systems that treat SMS replies as first-class conversion events.

## The real problem: flags are not experiments

Feature flags and experimentation are frequently conflated, and the confusion causes wasted work. A flag answers "should this user see variant B?" An experiment answers "did variant B change the outcome, and can we trust that answer?" A flag system needs low-latency evaluation and a cached state. An experiment system additionally needs exposure logging, a conversion signal, and a way to stop a harmful variant.

Teams commonly adopt a flag tool, then discover that the experiment half requires an event pipeline they don't have. The documented failure mode looks like this: a team ships a new SMS template behind a flag, sees replies drop, and has no baseline to compare against because exposures were never logged. The flag worked; the experiment never existed.

For SMS-driven systems there are four constraints that eliminate most of the market:

1. **Latency budget.** If flag evaluation blocks the request path, it delays message dispatch. A queue draining at one message per second tolerates tens of milliseconds per check; a synchronous evaluation that adds hundreds of milliseconds does not.
2. **Offline tolerance.** Power and connectivity interruptions are normal. A platform that requires a live call to a vendor API to evaluate a new variant cannot roll out anything during an outage.
3. **No warehouse.** Conversion data lives in PostgreSQL, Redis, or a spreadsheet. A platform that requires a cloud warehouse connector adds egress cost and a second system to operate.
4. **Non-pixel conversions.** The conversion event is an inbound SMS reply mapped to a user ID, not a browser event. The platform must accept arbitrary events, not just pageviews.

Everything below is organized around those four constraints.

## How to evaluate platforms against your own constraints

Published benchmark tables are close to useless here because latency and cost depend on your stack, your event volume, and your network. What follows is the measurement procedure, so the numbers you get are yours.

**Runtime latency.** Instrument the flag evaluation call, not the whole request. Wrap the SDK call in a timer and emit a histogram; if the SDK does not expose metrics, log start and end timestamps per evaluation and aggregate offline. Drive load with any HTTP load generator against a staging instance sized like production, and report p50, p95, and p99 — the tail matters more than the mean because a stalled evaluation blocks a queue. Compare the same endpoint with flags disabled to isolate the SDK's contribution.

**Deployment footprint.** Count the lines of configuration and code the SDK and server components add to your existing application. This is a proxy for maintenance burden and for how much of your team's attention the platform will consume during an incident.

**Cost at your event volume.** Multiply your actual daily event count by the vendor's per-event rate, then add storage, egress, and compute. Do this arithmetic before the trial, not after the invoice. For self-hosted options, cost is the instance size you actually need, which you can measure by watching memory during a load test.

**Offline behavior.** Disconnect the network for a period longer than the SDK's cache TTL and observe what happens to flag evaluation and to event capture. Document the cache TTL explicitly; it determines how long an outage can last before the system starts serving stale or default values.

**Conversion ingestion.** Send a test conversion from a non-browser source (a webhook, a queue consumer, a cron job) and confirm it appears in the experiment results. If the only documented ingestion path is a client-side SDK, the platform will not fit.

A worked example of the cost arithmetic, with illustrative numbers: suppose the platform charges per event, your system sends 10,000 SMS per day, and each message produces one exposure event plus a reply event for roughly 20% of recipients. That is 10,000 + 2,000 = 12,000 events per day, or about 360,000 per month. At a hypothetical $0.0005 per event, the monthly bill is 360,000 × $0.0005 = $180. The same arithmetic at 100,000 messages per day gives 1,200,000 events per month and $600. Substitute your vendor's real rate; the point is that per-event pricing scales with message volume, not with the number of experiments, which is the opposite of what a low-traffic team wants.

## The decision checklist

Before committing to a platform, answer these in writing:

- Does flag evaluation happen in-process or over the network? In-process evaluation with a cached ruleset is the only option that survives an outage.
- What is the cache TTL, and what does the SDK serve after it expires?
- Can conversion events arrive from a server-side source with no browser involved?
- Is the rollback mechanism automatic, or does it require a human to notice and act?
- What is the per-event cost at 10× your current volume?
- If the vendor disappears tomorrow, can you run the software yourself?

A platform that fails the first two questions is disqualified for offline-first SMS work regardless of how polished its dashboard is.

## Self-hosted options that fit

### OpenFeature with a custom backend

OpenFeature is a vendor-neutral specification for feature flagging, with SDKs in several languages. It defines the interface; you supply the evaluation backend. A common pattern is to store flag rules in Redis and evaluate them in a Lua script, so evaluation is a single in-process or loopback call with no external dependency.

The strength is control: no vendor API in the request path, no per-event billing, and the entire flag state can be backed up as a Redis dump. The cost is that you build the management UI and the experiment analysis yourself. A minimal admin page and a script that computes a two-proportion comparison are usually enough; a full statistical engine is not required to answer "did replies go up or down, and is the difference larger than noise?"

This option suits teams with a backend engineer and a hard constraint against recurring SaaS spend.

### Unleash

Unleash is an open-source feature flag server. It supports gradual rollouts, a proxy component that evaluates flags close to the application, and strategies that can be driven by user attributes. Self-hosted, the software cost is zero and the running cost is whatever instance you put it on.

The operational detail that matters most for this use case is the proxy's caching behavior: the proxy holds a local copy of the flag configuration and continues to serve it when the upstream server or the internet is unreachable, then reconciles when connectivity returns. That property is what makes gradual rollout possible during an outage. Confirm the current cache TTL in the documentation for your version before relying on it, and test the behavior by stopping the upstream server and observing evaluation.

The trade-off is resource consumption. The proxy and its database are not free in memory terms; a 1 vCPU, 1 GB instance is a realistic floor for a small deployment, and headroom shrinks as concurrent users grow. Measure memory under load rather than assuming.

### GrowthBook

GrowthBook is an open-source platform whose distinguishing feature is that experiment definitions and cohorts can be expressed as SQL against your own database. For teams already comfortable in SQL, this makes the definition of "active user" or "replied in the last 30 days" auditable and reproducible rather than buried in a dashboard.

The performance caveat is structural, not a bug: if cohort membership is computed by a query, that query has a cost, and running it on the request path will show up in latency. The standard remedy is to materialize cohort membership on a schedule into a table or cache and have the flag evaluation read the materialized result. Measure the query time separately from the evaluation time so you know which one to optimize.

### PostHog feature flags

PostHog combines product analytics with feature flags and experimentation, which means exposures and conversions land in the same system without an export step. For teams already using it for analytics, the marginal setup cost is low.

The consideration is pricing model. Usage-based pricing means cost tracks event volume, so a spike in traffic — or a verbose event schema — shows up on the bill. Model your own volume against the current published rates before committing, and check whether the free tier's limits match your message volume.

### Flagsmith

Flagsmith provides feature flags with trait-based segmentation, which is useful when the same binary must serve different audiences — for example, different SMS templates for different user roles. Traits are passed at evaluation time, so segmentation does not require separate deployments.

The limitation to verify is offline behavior: many SaaS flag SDKs fall back to a cached state when the network is unavailable, but cannot evaluate new rules until connectivity returns. If your rollout plan depends on changing rules during an outage, confirm this before adopting.

## Options that usually do not fit, and why

**Enterprise SaaS flag platforms.** These are polished, well-documented, and scale to very large user counts. They typically evaluate flags via a vendor API or a client SDK that phones home, and they price per event or per seat. For a team whose budget is a single small VPS, the arithmetic rarely works, and the connectivity requirement conflicts with offline operation. They remain the right choice for organizations with dedicated platform budgets and reliable networks.

**Marketing-oriented experimentation suites.** These are built around a visual editor and a browser SDK. The browser SDK's size is the problem on constrained networks: a few hundred kilobytes of JavaScript is a meaningful download on 2G, and it delays the page rather than the message. If the conversion happens over SMS rather than in a browser, the visual editor buys little.

**Any platform requiring a cloud data warehouse.** Warehouse connectors add egress cost and a second operational dependency. If your data is already in PostgreSQL, a platform that reads from PostgreSQL directly is simpler and cheaper.

## Failure modes to design against

**Stale flag state after a long outage.** If the cache TTL expires during an outage, the SDK serves defaults. Decide deliberately what the default is: for a new SMS template, the safe default is usually the old template, not the new one.

**Exposure logging that blocks the request path.** Writing an exposure event synchronously to a remote endpoint adds latency to every message. Buffer exposures locally and flush asynchronously; accept that a crash may lose a small number of exposures rather than slowing every send.

**Automatic rollback that fires on noise.** An error-rate threshold that is too tight will roll back a healthy variant during a transient spike. Set the threshold from observed baseline variance, require the condition to hold for a minimum window, and log every automatic rollback with the metric values that triggered it so the decision can be reviewed.

**Unbounded Redis growth.** Storing full user objects per flag evaluation bloats memory. Store a hashed user identifier and look up attributes only when a rule actually needs them. Measure memory per user under load and extrapolate before you hit swap.

**Conversion mapping errors.** Phone numbers are not stable identifiers: they get recycled, reformatted, and entered with country-code variations. Normalize to E.164 at ingestion, store the mapping separately from the flag state, and log unmapped replies rather than dropping them silently.

## A minimal working setup

The following deploys a self-hosted flag server with Docker, evaluates a flag from a Flask application, and ingests SMS replies as conversions through a Redis stream. It is a starting point, not a production configuration: add authentication, TLS, backups, and monitoring before relying on it.

```bash
# On a fresh Ubuntu 22.04 instance
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker "$USER"
newgrp docker

git clone https://github.com/Unleash/unleash.git
cd unleash/docker

# Configure the database connection and any required secrets in .env
# before starting the stack.
docker compose up -d

# Confirm the server is responding
curl -i http://localhost:4242/health
```

Then evaluate a flag from the application. Check the current SDK package name and version on the project's release page before pinning it; the import path has changed across major versions.

```python
# requirements.txt
# Pin versions after checking the current releases.
UnleashClient
redis

# app.py
from flask import Flask
from UnleashClient import UnleashClient

app = Flask(__name__)

unleash = UnleashClient(
    url="http://localhost:4242/api",
    app_name="sms-campaign",
    instance_id="your-instance-id",
)
unleash.initialize_client()

@app.route("/send-sms")
def send_sms():
    if unleash.is_enabled("new-sms-template"):
        return "Hi, your appointment is tomorrow at 9am. Reply STOP to opt out."
    return "Hi, your appointment is tomorrow. Reply STOP to opt out."
```

Finally, a collector that turns inbound replies into conversion events. This runs as a separate process; the flag server never sees the request path.

```python
# metrics_collector.py
import redis

r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

while True:
    # Block for new replies; "$" means only messages added after this call.
    replies = r.xread({"sms-replies": "$"}, count=100, block=5000)
    for _stream, messages in replies:
        for _message_id, data in messages:
            user_id = data["user_id"]
            conversion = data.get("reply", "").strip().lower() == "yes"
            r.xadd("conversions", {"user_id": user_id, "conversion": int(conversion)})
```

Two notes on correctness. First, using `$` as the stream position means the collector only sees messages that arrive after it starts; on restart it will skip anything sent while it was down. Track the last processed message ID persistently and resume from it. Second, the conversion definition here is a placeholder — decide what counts as success for your campaign and encode that explicitly, because a vague success criterion is the most common reason an experiment's result is uninterpretable.

## Measuring whether the experiment worked

You do not need a statistics package to get a defensible answer. Log, for each user, which variant they were exposed to and whether they converted. Then compute:

- Conversion rate per variant: conversions divided by exposures. Report the denominator explicitly; a rate without a denominator hides how many users were actually exposed.
- The absolute difference between rates, with a confidence interval. For two proportions, the standard error of the difference is the square root of `p1(1-p1)/n1 + p2(1-p2)/n2`, and an approximate 95% interval is the difference plus or minus 1.96 times that standard error. This is arithmetic you can do in a spreadsheet.
- The exposure count per variant. If one variant has far fewer exposures than the other, the split is broken; investigate the assignment logic before interpreting the result.

A difference whose confidence interval includes zero is not evidence of an effect, regardless of how large the point estimate looks. And a result measured over a period that includes a holiday, a network outage, or a template change is measuring those things too.

## Frequently asked questions

**How do I track conversions from SMS replies without a data warehouse?**

Map the sender's phone number to a user identifier at ingestion, write a conversion event to a queue or table, and have the experiment analysis read from there. A Redis stream or a PostgreSQL table both work; the requirement is that the conversion arrives server-side, not from a browser.

**What is the smallest instance that can run a self-hosted flag server?**

It depends on the server, the proxy, and the database. Measure memory under a load test that matches your expected concurrent users, and treat swap usage as the signal that you have gone too small. A 1 vCPU, 1 GB instance is a common starting point for small deployments, but verify rather than assume.

**Can I run experiments without a flag platform?**

Yes. A boolean column in a database plus a deterministic hash of the user ID into buckets is a complete assignment mechanism. What you give up is the operational layer: a management UI, gradual rollout controls, automatic rollback, and exposure logging. The failure mode that follows is manual error — a bad variant pushed to all users with no fast way to reverse it.

**How should the system behave when the internet drops?**

Flag evaluation should continue from cached state, and conversion events should be buffered locally and flushed when connectivity returns. Decide and document what happens when the cache expires during an outage, because that is the moment the system starts serving defaults.

**How long should an experiment run?**

Long enough to accumulate exposures on both variants and to cover at least one full weekly cycle, since reply behavior varies by day of week. Stopping as soon as the difference looks significant inflates the false-positive rate; fix the duration or the sample size in advance.

## The next 30 minutes

Bring up a self-hosted flag server on a scratch instance and verify the health endpoint returns a success status, then stop the server and confirm the SDK or proxy still evaluates a flag from its cache. That single test tells you whether the platform can survive an outage, which is the constraint that eliminates most of the alternatives. If it fails, you have learned the most important thing about the platform before writing any application code.
