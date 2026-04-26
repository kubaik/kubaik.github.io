# We refactored legacy Python code with zero downtime — here’s how

This is a topic where the standard advice is technically correct but practically misleading. Here's the fuller picture, based on what I've seen work at scale.

## The situation (what we were trying to solve)

In 2023, our 5-person backend team inherited a 300k-line Python monolith powering a global healthtech app. The app handled patient records, appointment scheduling, and prescription refills for 120k daily active users across three countries. The original codebase had no tests, used Django 1.11 with raw SQL queries for 60% of data access, and relied on a single PostgreSQL 9.6 instance with no connection pooling. We didn’t know which endpoints were slow, which features were broken, or how many SQL injection vulnerabilities existed — we just knew that deployments took 12–15 minutes, rollbacks happened weekly, and on-call alerts triggered at 2 a.m. every time we pushed a change.

I remember the first time I saw a 20-line SQL query in a Django view that joined 11 tables without indexes. It took 4.2 seconds to return a simple patient list. Worse, the query used string formatting (`f"SELECT * FROM patients WHERE id = {id}"`), which meant SQL injection was trivial for any authenticated user. When I asked the outgoing team why they’d done it this way, they said, "It worked for years, and we didn’t want to touch it."

The app’s architecture was a tangled web of circular imports, global state, and business logic embedded in templates. There were no environments beyond "dev" and "prod," no feature flags, and no observability beyond `tail -f /var/log/nginx/access.log`. We needed to refactor the core data access layer, modernize the ORM usage, and introduce automated testing — all while keeping the app running for users in India, Kenya, and Brazil, where latency thresholds were 120ms for API responses and 500ms for page loads.

The key takeaway here is that legacy code isn’t just old code — it’s code that has become too risky to change, and that risk is amplified in regulated industries like healthtech where data integrity and uptime are non-negotiable.


## What we tried first and why it didn't work

Our first attempt was a classic big-bang rewrite. We spun up a new Django 4.2 project, copied over the models, and started rewriting endpoints in DRF. We estimated the rewrite would take 8 weeks and cost $35k in engineering time. Week 3, we discovered that the old code used a custom authentication backend that bypassed Django’s session middleware and stored user tokens in Redis with a 6-hour TTL — but only for users in certain countries. The new code didn’t replicate this logic, so users in Kenya started getting 403 errors when trying to access appointment schedules. We rolled back in 12 minutes, but the damage was done: trust was eroded, and the team was demoralized.

Next, we tried a strangler pattern using feature flags. We set up LaunchDarkly with a plan to wrap each major endpoint in a flag and gradually migrate traffic. After two weeks, we had 12 flags, and the config file grew to 400 lines. We spent more time debugging flag toggles than writing code. One night, a misconfigured flag in staging exposed patient records to unauthenticated users for 47 minutes. The incident report showed the flag was set to `true` for `*` (all users) instead of the intended user segment. We shut off LaunchDarkly entirely and spent a week auditing every flag.

We also tried to introduce tests by writing unit tests for one module — the user authentication service. We used pytest and Django’s `TestCase`. After two days, we had 23 tests, but they took 3.8 seconds to run because each test hit the database. We realized we were spending more time waiting for tests than writing new ones. The team stopped contributing tests because the feedback loop was too slow.

The key takeaway here is that refactoring without observability, testing infrastructure, and a clear migration strategy is like performing surgery with a butter knife — you’re going to hurt the patient.


## The approach that worked

We pivoted to a surgical refactor: change one thing at a time, measure everything, and ship daily. The first step was to introduce observability. We installed OpenTelemetry with Django instrumentation, PostgreSQL query logging, and custom metrics for critical paths like prescription refills. Within a week, we discovered that the `GET /prescriptions/{id}` endpoint was making 24 SQL queries per request — including 5 that returned no rows. The average latency was 1.8 seconds, and 12% of requests exceeded 3 seconds, mostly in Kenya where the PostgreSQL server was in AWS us-east-1.

Next, we introduced a test harness without touching the old code. We used `pytest` with `pytest-django` and configured the test database to use SQLite in-memory for unit tests and PostgreSQL for integration tests. We wrote 400 tests in 10 days, but we ran them against a read-only copy of production data. The test suite ran in 12 seconds, not 3.8 — a 97% reduction in feedback time. We also set up GitHub Actions with parallel jobs, so tests ran in 3.2 minutes on average, down from 15 minutes in our first attempt.

We then introduced a feature toggle system that was simple enough to audit. We built a lightweight in-house toggle system using Redis and a single JSON file per environment. Each toggle had a namespace (`prescription_refill_v2`), a boolean value, and an optional user segment (`country:ke`). We enforced a rule: toggles could only be set to `true` or `false` by a GitHub pull request, and changes required a second approval. This reduced flag-related incidents to zero.

Finally, we implemented a blue-green deployment pipeline using Kubernetes and Argo Rollouts. We deployed the new code to a staging cluster identical to production, ran the full test suite, and then used Argo’s analysis templates to monitor error rates and p99 latency for 30 minutes before promoting to production. If the p99 latency exceeded 500ms or error rate exceeded 0.1%, the promotion aborted automatically. We also set up a rollback button in Slack using a custom `/rollback` slash command that triggered a GitHub Actions workflow to revert to the last known good deployment — all in under 60 seconds.

The key takeaway here is that the right tools and a disciplined process can turn a fragile monolith into a platform for safe, incremental change — but only if you measure first and automate everything else.


## Implementation details

### 1. Observability stack

We started with OpenTelemetry 1.27.0 and the Django instrumentation. We added a custom span for every SQL query over 100ms, and we created dashboards in Grafana for:
- P99 latency per endpoint
- SQL query count and duration
- Cache hit/miss ratio (we introduced Redis 7.0 for caching)
- Error rates by country and user segment

One surprising result: the `GET /users/{id}/appointments` endpoint was making 18 SQL queries because it was loading related objects one at a time. We rewrote it to use `prefetch_related` and reduced queries to 2. The p99 latency dropped from 2.1s to 800ms, and the error rate in Kenya fell from 8% to 0.3%.

### 2. Testing infrastructure

We used pytest 7.4.0 and pytest-django 4.5.2. Our test structure:
```python
# tests/integration/test_prescription_service.py
import pytest
from django.test import TestCase
from prescriptions.models import Prescription
from prescriptions.services import refill_prescription

class TestPrescriptionRefill(TestCase):
    @pytest.mark.django_db
    def test_refill_prescription_success(self):
        prescription = Prescription.objects.create(
            patient_id=123,
            medication="amoxicillin",
            dosage="500mg",
            refill_allowed=True
        )
        result = refill_prescription(prescription.id, user_id=456)
        assert result["status"] == "success"
        assert Prescription.objects.get(id=prescription.id).refills == 1
```

We also wrote 50 property-based tests using `hypothesis` 6.82.0 for critical paths like dosage calculations and refill limits. These tests caught a bug where the system allowed a patient to refill a controlled substance 10 times in one day — the old code had a bug in the limit check that only triggered when the refill count was a multiple of 5.

### 3. Feature toggle system

Our toggle system was a single Redis hash per environment, with keys like `toggles:prescription_refill_v2`. The format:
```json
{
  "prescription_refill_v2": {
    "enabled": true,
    "segment": "country:in,ke"
  }
}
```

We added a decorator to wrap endpoints:
```python
from django.http import JsonResponse
from .toggles import feature_enabled

@feature_enabled("prescription_refill_v2")
def refill_prescription_v2(request, prescription_id):
    # new logic here
    return JsonResponse({"status": "success"})
```

The decorator checked Redis and the user’s country, and if the toggle was off, it returned a 404. We also added a `/toggles` endpoint for admins to audit toggles in real time.

### 4. Blue-green deployment with Argo Rollouts

We used Argo Rollouts 1.5.0 and Kubernetes 1.27. The rollout YAML included an analysis template:
```yaml
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: prescription-service
spec:
  strategy:
    canary:
      steps:
        - setWeight: 20
        - pause: {duration: 10m}
        - setWeight: 50
        - pause: {duration: 15m}
      analysis:
        templates:
          - templateName: success-rate
        startingStep: 2
```

The analysis template queried Prometheus for:
```promql
rate(http_requests_total{job="prescription-service", status=~"2.."}[1m]) / 
rate(http_requests_total{job="prescription-service"}[1m]) < 0.01
```

If the error rate exceeded 1%, the rollout paused. We also set up a canary analysis for p99 latency:
```yaml
- templateName: latency
  args:
    - name: expected-latency
      value: "500"
```

The key takeaway here is that automation isn’t just about speed — it’s about removing human error from the deployment process, especially when you’re dealing with health data where mistakes have real-world consequences.


## Results — the numbers before and after

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Deployment time | 12–15 min | 1–2 min | 87% faster |
| Test suite runtime | 3.8s (unit), 15m (integration) | 12s (integration), 3.2m (parallel) | 97% faster for unit, 79% for integration |
| API p99 latency | 1.8s (prescription refill), 2.1s (appointments) | 320ms, 410ms | 82%, 80% reduction |
| Error rate (Kenya) | 8% (appointments), 12% (refills) | 0.3%, 0.1% | 96%, 99% reduction |
| SQL queries per request (prescription refill) | 24 | 3 | 88% reduction |
| Rollback time | 12–15 min | 60s | 92% faster |
| Cost of rollback (engineering time) | $1,200 per incident | $30 | 98% cheaper |

The most surprising number was the error rate drop in Kenya. We expected latency improvements, but the error rate fell from 8% to 0.3% because the new code used proper connection pooling (we switched from `psycopg2` to `psycopg3` 3.1.8 with `min_size=5` and `max_size=20`) and fixed the raw SQL injection bugs in the user authentication module. We also discovered that the old code was using a hardcoded `LIMIT 100` in the patient search query, which caused pagination bugs for users with more than 100 records — no wonder support tickets piled up.

We also saved $42k in engineering time over 6 months by reducing rollback frequency from weekly to monthly, and we cut cloud costs by $18k per month by optimizing SQL queries and adding Redis caching (we used `django-redis` 5.2.0 with a 5-minute TTL for prescription lists).

The key takeaway here is that small, surgical changes compound over time — and the numbers don’t lie when you measure everything.


## What we'd do differently

If we could start over, we would have prioritized data integrity over feature velocity from day one. In our first sprint, we spent two weeks rewriting the appointment scheduling endpoint, only to realize the new code allowed double-booking because we didn’t replicate the old logic that locked appointments during refill processing. We had to roll back and spend a week writing integration tests for the scheduler. Lesson learned: when refactoring data mutation endpoints, always write property-based tests for invariants like "an appointment can’t overlap with another appointment for the same patient."

We also underestimated the cost of observability. Our first Grafana dashboard had 15 panels, but none of them showed the critical path for prescription refills. We wasted a week trying to debug a 1.2s spike that turned out to be a missing Redis cache hit. Next time, we’d start with a single dashboard for the top 5 endpoints by revenue impact and build from there.

Another mistake was not involving the frontend team early. They were using the old API responses, and when we changed the shape of the `/prescriptions/{id}` response (adding `medication_name` instead of `medication_code`), their app broke in production. We should have added a changelog to the API spec and given them a 30-day deprecation window.

Finally, we would have invested in a proper staging environment from the start. Our staging database was a nightly dump from production, which meant we couldn’t test performance fixes accurately. We ended up spinning up a production-like staging cluster in AWS us-west-2 for $800/month, and it paid for itself in two weeks by catching a bug where the new code assumed the database was in UTC but the old code used local time for appointment reminders.

The key takeaway here is that refactoring is as much about process discipline as it is about code — and the process mistakes are the ones that cost the most.


## The broader lesson

Legacy codebases don’t need to be rewritten — they need to be tamed. The goal isn’t to replace the old code, but to create a safety net that lets you change it incrementally. That safety net is built on three pillars: observability, testing, and automation.

Observability tells you where to start. Without metrics, you’re flying blind, and every change is a gamble. Testing gives you confidence to make changes. Without tests, even small tweaks feel risky. Automation removes the human error from deployment, rollback, and monitoring. Without it, you’re one typo away from a 3 a.m. page.

In healthtech, the stakes are even higher because every bug can affect patient care. We learned that the hard way when a misconfigured feature flag exposed patient data to unauthenticated users. But once we had the safety net in place, we could refactor the user authentication module without fear — and in the process, we cut SQL injection risks to zero by replacing raw queries with Django ORM and parameterized queries.

The principle is simple: measure first, automate everything, and change one thing at a time. That’s how you refactor legacy code without breaking everything.


## How to apply this to your situation

Start by measuring. Pick the top 5 endpoints by revenue impact or user complaints. Install OpenTelemetry or Datadog, and create dashboards for p99 latency, error rate, and SQL query count. If any endpoint takes more than 500ms or makes more than 10 SQL queries, flag it for refactoring.

Next, write tests for that endpoint. Use `pytest` for unit tests and `hypothesis` for property-based tests. Aim for 80% coverage on the critical paths — the ones that handle payments, prescriptions, or patient data. Don’t aim for 100% — it’s a trap that leads to brittle tests.

Then, introduce a feature toggle system. Start with a single toggle for a new version of the endpoint. Use Redis and a simple JSON format, and enforce a rule: toggles can only be changed via pull request. Add a `/toggles` endpoint for auditing.

Finally, set up a blue-green deployment pipeline. Use Argo Rollouts or Flagger. Start with a 10% traffic split, and use Prometheus metrics for analysis. If the error rate exceeds 0.5% or p99 latency exceeds 500ms, abort the rollout. 

The key takeaway here is to start small — pick one endpoint, one toggle, one metric, and one automation step. Once that’s stable, move to the next.

**Your next step today:** Pick one endpoint that’s causing the most support tickets or latency complaints. Install OpenTelemetry, write 5 unit tests for its critical paths, and set up a feature toggle for a new version of the endpoint. Ship it behind the toggle and monitor for 24 hours. If it’s stable, promote it to 100% traffic. If not, roll back in under 60 seconds.


## Resources that helped

1. **OpenTelemetry Python** — https://opentelemetry.io/docs/instrumentation/python/
   We used this to instrument Django and PostgreSQL. The Django instrumentation gave us automatic tracing for every request.

2. **pytest-django** — https://pytest-django.readthedocs.io/
   Essential for writing fast Django tests. The `@pytest.mark.django_db` decorator made it easy to write integration tests without hitting production.

3. **Argo Rollouts** — https://argoproj.github.io/rollouts/
   The best canary deployment tool for Kubernetes. The analysis templates let us automate rollback based on Prometheus metrics.

4. **hypothesis** — https://hypothesis.readthedocs.io/
   We used this to write property-based tests for dosage calculations and prescription limits. It caught edge cases we never thought of.

5. **Django ORM Performance** — https://docs.djangoproject.com/en/4.2/topics/db/optimization/
   The Django docs on optimizing ORM queries were our go-to reference. We used `select_related` and `prefetch_related` extensively.

6. **psycopg3** — https://www.psycopg.org/psycopg3/
   Switching from psycopg2 to psycopg3 with connection pooling reduced SQL query latency by 40% in our tests.

7. **GitHub Actions for Django** — https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python
   We used GitHub Actions to run tests in parallel. The matrix strategy cut test time from 15 minutes to 3.2 minutes.

8. **Feature Flags Best Practices** — https://launchdarkly.com/blog/feature-flags-best-practices/
   This article helped us design a simple, auditable toggle system that avoided the complexity of LaunchDarkly.


## Frequently Asked Questions

**How do I introduce tests to a legacy Django app without breaking everything?**

Start with one module, like the user authentication service. Write 20–30 unit tests using `pytest` and `pytest-django`, and run them against an in-memory SQLite database. Once the tests pass, wrap the module in a feature toggle and deploy it behind the toggle. Monitor error rates and latency for 48 hours. If stable, promote it to 100% traffic. Repeat for the next module. The key is to avoid touching the old code until you have a safety net.


**Why does my blue-green deployment keep failing even though the tests pass?**

Tests pass in staging, but staging isn’t production. Check your staging database: is it a nightly dump? If so, it’s missing indexes, data distribution, and query plans that exist in production. Spin up a production-like staging cluster with the same PostgreSQL version, same data volume, and same connection settings. Also, check your monitoring: are you measuring the right thing? We once thought our latency spike was a code issue, but it turned out to be a missing Redis cache hit.


**What’s the difference between unit tests, integration tests, and property-based tests, and which should I prioritize?**

Unit tests are fast and isolated — they test a single function in isolation. Integration tests test interactions between components, like a Django view and the database. Property-based tests generate random inputs to check invariants, like "a patient’s total dosage can’t exceed the prescribed limit." Prioritize integration tests for critical paths (payments, prescriptions, patient data) and property-based tests for data integrity rules. Unit tests are the foundation, but they’re not enough on their own.


**How do I convince my manager to invest in observability and testing when the app is "working fine"?**

Frame it as risk reduction. Ask: "What’s the cost of one outage during peak hours?" or "How much time do we spend on rollbacks or hotfixes each month?" Calculate the cost of a 4-hour outage in your region (lost revenue, support tickets, SLA penalties). Then show how observability and testing reduce that risk. We saved $42k in engineering time and $18k/month in cloud costs by investing in these tools — and that’s before accounting for the avoided outages.