# 2026 alert triage: when Opsgenie slept for you

## The real problem: engineers as the triage layer

Most alerting tutorials stop at the happy path: an alert fires, a page goes out, a human looks at it. Production on-call is not that. A typical rotation receives a mix of genuine incidents and alerts that resolve themselves: latency spikes that never become outages, disk usage that recovers after a log rotation, background jobs that retry successfully on the second attempt. The paging tool has no way to know the difference, so the engineer becomes the triage layer for machines.

The documented behavior of most paging platforms is to route every alert that matches a rule. PagerDuty and Opsgenie both support deduplication and grouping, but deduplication keys are usually derived from a static alert name, and grouping windows are short. A daily batch job that runs ten minutes late every day will page every day unless someone writes a rule specifically excluding it. That rule is the manual version of what a triage layer should do automatically.

An AI triage layer sits between the monitoring system and the paging platform. It scores each incoming alert, suppresses the ones that historically resolve on their own, collapses correlated alerts into a single ticket, and attaches a short natural-language summary. The goal is not "fewer alerts" as a vanity metric. The goal is fewer wake-ups for problems that need a human, with an auditable record of everything that was suppressed.

## Prerequisites and what you will build

This design assumes:

- A paging platform with a read/write API. Opsgenie's Alert API and PagerDuty's Events API v2 both qualify. The examples below use an SDK-style client; substitute the equivalent calls for your platform.
- A metrics backend you can query for historical alert outcomes. Prometheus is the common choice; any queryable time-series store works.
- Python 3.11 or newer.
- An LLM endpoint for summaries. This can be a hosted API or a self-hosted model behind an OpenAI-compatible interface. Treat the endpoint and model name as configuration, not as a hard dependency.

The layer you build will:

- Poll the paging platform for open alerts on a fixed interval.
- Score each alert using a small, explainable model.
- Suppress alerts below a threshold, with a note explaining why.
- Group related alerts and create one incident for the group.
- Generate a short summary for the incident.
- Emit metrics about its own behavior, so you can tell whether it is working.

The scoring model is deliberately simple. A simple model you can reason about beats a complex one you cannot debug at 3 a.m.

## Step 1 — environment and test alert source

Create a virtual environment and install the client libraries you need. Pin versions and check the current release notes for each; API surfaces change.

```bash
python -m venv venv
source venv/bin/activate  # venv\Scripts\activate on Windows
pip install prometheus-api-client openai python-dotenv requests
```

Create a `.env` file. Never commit this file.

```ini
# .env
PAGER_API_KEY=your_paging_platform_key_here
LLM_API_KEY=your_llm_key_here
PROMETHEUS_URL=http://localhost:9090
```

Scope the paging API key to the smallest set of permissions that still lets the layer read alerts and write notes. A key with full account access is a liability if the process is compromised.

To generate test alerts without touching production, run a node exporter and scrape its `up` metric with Prometheus:

```yaml
scrape_configs:
  - job_name: 'node'
    static_configs:
      - targets: ['localhost:9100']
```

Add a rule that fires when the exporter disappears:

```yaml
groups:
  - name: example
    rules:
      - alert: InstanceDown
        expr: up{job="node"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Instance {{ $labels.instance }} down"
```

Reload Prometheus and stop the exporter to trigger the alert:

```bash
curl -X POST http://localhost:9090/-/reload
```

Confirm the alert reaches your paging platform. If it does not, check the integration configuration before writing any triage code. Debugging the triage layer while the alert pipeline is broken wastes time.

One behavior worth knowing: many paging platforms cache or debounce incoming alerts for a short window. Triggering several alerts in quick succession may produce a single ticket. That is platform behavior, not a bug in your code, and it interacts with the deduplication you will add later.

## Step 2 — the scoring function

The scoring model has three inputs, each normalized to the range 0 to 1:

1. **Severity**: how urgent the alert is by its own label. A common mapping is critical = 1.0, warning = 0.6, info = 0.3. These weights are a starting point, not a law of nature.
2. **Correlation**: how many related alerts fired recently. A single alert is more likely to be noise than one of twenty.
3. **Historical resolution**: how quickly similar alerts have resolved in the past. An alert type whose median resolution is two minutes is a weaker paging candidate than one whose median is two hours.

```python
import os
import time
import json
import logging
from typing import Dict, List, Optional

import requests
from prometheus_api_client import PrometheusConnect
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
llm = OpenAI(api_key=os.getenv("LLM_API_KEY"))
prom = PrometheusConnect(url=os.getenv("PROMETHEUS_URL"), disable_ssl=True)

SEVERITY_WEIGHTS = {"critical": 1.0, "warning": 0.6, "info": 0.3}
SUPPRESS_THRESHOLD = 0.7
DEFAULT_RESOLUTION_MINUTES = 10.0


def score_alert(alert: Dict) -> float:
    severity = SEVERITY_WEIGHTS.get(alert.get("priority", "info"), 0.3)

    related_count = count_recent_related_alerts(alert.get("message", ""))
    correlation = 1.0 / (1.0 + related_count)

    median_resolution = get_median_resolution(alert.get("message", ""))
    if median_resolution is None:
        median_resolution = DEFAULT_RESOLUTION_MINUTES
    resolution_score = 1.0 - min(median_resolution / 60.0, 1.0)

    final_score = (
        severity * 0.5
        + correlation * 0.25
        + resolution_score * 0.25
    )
    return round(final_score, 2)
```

Two properties make this model usable in practice. First, every term is bounded, so the final score is always between 0 and 1. Second, the weights sum to 1, so the score has a consistent interpretation: a critical alert with no correlated noise and a fast historical resolution still scores at least 0.5 from severity alone, which is below the 0.7 threshold. That is intentional. Severity alone should not be sufficient to page if the alert type reliably self-heals.

The correlation term deserves scrutiny. `1.0 / (1.0 + related_count)` means a single alert scores 1.0 and a burst of nineteen related alerts scores 0.05. The intent is that a burst is one incident, not twenty, and should be handled by grouping rather than by suppression. But the term also means a genuinely novel alert, which has no related history, gets the maximum correlation score. That is the correct default: unknown alerts should page.

## Step 3 — historical resolution time

The scoring model depends on knowing how long similar alerts have historically taken to resolve. You need to record that data yourself. The paging platform's API will give you current and recent alerts, but not a long-term aggregate.

The approach is to write a resolution timestamp into your metrics backend whenever an alert closes, then query it. For example, a small job can poll the paging platform for closed alerts, compute the time between creation and closure, and push a sample to a metric such as `alert_resolution_seconds` with the alert name as a label. Over a few weeks this gives you a per-alert-type distribution.

Querying the median is then a matter of asking the metrics backend for a quantile:

```python
def get_median_resolution(alert_name: str) -> Optional[float]:
    end = time.time()
    start = end - (30 * 24 * 3600)  # last 30 days
    query = (
        'quantile_over_time(0.5, '
        'alert_resolution_seconds{alert_name="%s"}[30d])' % alert_name
    )
    try:
        result = prom.custom_query(query=query)
        if result:
            return float(result[0]["value"][1]) / 60.0
    except Exception as exc:
        logging.warning("resolution query failed: %s", exc)
    return None
```

If you do not yet have 30 days of history, the function returns `None` and the scoring model falls back to the default. That is the right behavior for a new deployment: unknown alert types are treated conservatively.

Note the units. The metric stores seconds; the scoring function expects minutes. Mixing units is the most common source of silently wrong scores. Write a unit test that asserts the conversion.

## Step 4 — grouping, suppression and incident creation

The main loop polls for open alerts, groups them, scores each group, and either creates an incident or suppresses with an explanatory note.

```python
def group_alerts(alerts: List[Dict]) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = {}
    for alert in alerts:
        key = f"{alert.get('message', '')}:{alert.get('priority', '')}"
        grouped.setdefault(key, []).append(alert)
    return grouped
```

Grouping by message and priority is a starting point. It fails when the same underlying problem produces alerts with different messages, for example one alert per affected host. A more robust key strips volatile fields such as hostnames and timestamps from the message before grouping. Whatever key you choose, log it. When grouping goes wrong, the key is the first thing you need to see.

```python
def triage_alerts():
    while True:
        try:
            alerts = fetch_open_alerts()
            grouped = group_alerts(alerts)

            for key, alert_list in grouped.items():
                score = score_alert(alert_list[0])
                if score >= SUPPRESS_THRESHOLD:
                    logging.info("score %s — incident for %s", score, key)
                    create_incident(alert_list)
                else:
                    logging.info("score %s — suppressing %s", score, key)
                    suppress_alerts(alert_list)
        except Exception as exc:
            logging.error("triage loop error: %s", exc)

        time.sleep(30)
```

The suppression path must leave a record. A suppression with no note is indistinguishable from a lost alert, and the first time someone asks "why did we not get paged for this?" you will need the answer.

```python
def suppress_alerts(alerts: List[Dict]):
    for alert in alerts:
        note = (
            "Suppressed by AI triage layer. Score below "
            f"{SUPPRESS_THRESHOLD}. See triage logs for the factor breakdown."
        )
        try:
            add_note(alert["id"], note)
        except Exception as exc:
            logging.error("failed to annotate alert %s: %s", alert["id"], exc)
```

Incident creation collapses the group into one ticket and attaches a summary:

```python
def create_incident(alerts: List[Dict]) -> Optional[str]:
    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    alerts = sorted(alerts, key=lambda a: priority_order.get(a.get("priority"), 3))
    lead = alerts[0]

    payload = {
        "message": f"Collapsed incident: {lead.get('message', 'unknown')}",
        "description": generate_llm_summary(alerts),
        "priority": lead.get("priority", "medium"),
        "tags": ["ai-triage", "collapsed"],
    }

    try:
        return create_alert(payload)["id"]
    except Exception as exc:
        logging.error("failed to create incident: %s", exc)
        return None
```

## Step 5 — the LLM summary, and how it fails

The summary is the part engineers notice first, and the part most likely to embarrass you. Generate it from structured alert data, not from raw log lines, and constrain the output.

```python
SUMMARY_PROMPT = """You are an SRE writing a short incident summary for an on-call engineer.

Alerts:
{alerts}

Write at most 150 words covering:
- what is happening
- which systems are affected
- a plausible root cause, clearly marked as a hypothesis
- recommended first diagnostic step

Do not invent system names, hostnames or error codes that are not present in the alerts.
"""


def generate_llm_summary(alerts: List[Dict]) -> str:
    payload = json.dumps(
        [{"message": a.get("message"), "priority": a.get("priority")} for a in alerts]
    )
    try:
        response = llm.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "You write terse, factual SRE summaries."},
                {"role": "user", "content": SUMMARY_PROMPT.format(alerts=payload)},
            ],
            max_tokens=250,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        logging.error("LLM call failed: %s", exc)
        return "Automated summary unavailable — see raw alerts."
```

The failure modes are predictable:

- **Hallucinated specifics.** The model invents a hostname or error code that was not in the input. Mitigation: instruct the model not to, and validate the output against the set of terms present in the alerts. A summary containing a token that looks like a hostname but does not appear in the input should be discarded.
- **Confident root cause.** The model states a hypothesis as fact. Mitigation: require the hypothesis to be labeled as such in the prompt, and keep the summary short enough that there is no room for a narrative.
- **Latency.** A slow LLM call delays incident creation. Mitigation: set a client timeout and fall back to the raw alert text on timeout. An incident with no summary is better than an incident created two minutes late.
- **Cost.** Every alert group triggers a call. Measure this directly: log token counts per call and multiply by your provider's published price. Do not estimate from alert volume alone, because summaries vary in length. Instrument the call and read the number.

Never put secrets, credentials or customer identifiers into a prompt. If alert messages can contain them, redact before sending.

## Step 6 — observability for the triage layer itself

The triage layer is production infrastructure. It needs the same monitoring as anything else, and a silent failure is worse than no layer at all, because alerts will be suppressed and nobody will know.

Instrument at minimum:

- `ai_triage_alerts_processed_total` — counter
- `ai_triage_alerts_suppressed_total` — counter
- `ai_triage_incidents_created_total` — counter
- `ai_triage_llm_call_duration_seconds` — histogram
- `ai_triage_llm_tokens_total` — counter, labeled by direction

```python
from prometheus_client import start_http_server, Counter, Histogram

ALERTS_PROCESSED = Counter("ai_triage_alerts_processed_total", "Alerts processed")
ALERTS_SUPPRESSED = Counter("ai_triage_alerts_suppressed_total", "Alerts suppressed")
INCIDENTS_CREATED = Counter("ai_triage_incidents_created_total", "Incidents created")
LLM_CALL_DURATION = Histogram(
    "ai_triage_llm_call_duration_seconds",
    "LLM call duration",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0],
)

start_http_server(8000)
```

Add a heartbeat. The simplest version writes the current timestamp to a file or a gauge on every loop iteration, and a separate alert fires if the value stops advancing:

```python
HEARTBEAT = Gauge("ai_triage_last_loop_timestamp", "Unix time of last loop")
```

Then alert on `time() - ai_triage_last_loop_timestamp > 120`. This catches the failure mode where the process is alive but stuck, which a process-level check will miss.

## How to measure whether it works

Do not adopt a suppression threshold on faith. Measure it.

The two error types are asymmetric and you should treat them that way:

- **False suppression**: an alert was suppressed and the underlying problem became a real incident. This is the dangerous error.
- **False page**: an alert was escalated and resolved without human action. This is the expensive error.

To measure false suppressions, you need a record of what was suppressed and a way to link it to later incidents. The annotation you write in `suppress_alerts` is the join key. A weekly review can pull suppressed alerts and search for incidents that mention the same service in the following hours.

To measure false pages, count incidents closed with no human action recorded. Most paging platforms expose this through their API.

A practical procedure:

1. Run the layer in shadow mode first. Score every alert and log the decision, but do not suppress or create anything. This gives you a clean baseline.
2. After one to two weeks, compare the shadow decisions against what actually happened. Count false suppressions and false pages at several candidate thresholds.
3. Pick the threshold that keeps false suppressions at or near zero, even if it costs you some false pages.
4. Turn on suppression. Re-measure monthly.

The threshold is not a constant. It should move as your system changes. A threshold tuned during a quiet quarter will be wrong during a launch.

## Failure modes to design against

**Alert storms from a dependency.** When a cloud provider has a regional issue, it may emit hundreds of alerts across many services. A correlation term that rewards bursts will score these low and suppress them — which is exactly wrong, because the storm is the incident. Add a rule that any alert tagged as originating from a known dependency bypasses suppression and is grouped into a single dependency incident.

**The triage layer suppresses the alert that would have caught its own bug.** If the layer crashes, no alerts are suppressed, but no alerts are escalated either. The heartbeat alert is the mitigation, and it must not route through the triage layer.

**Rate limits.** Paging APIs commonly cap requests per minute. A loop that lists alerts, queries for related alerts, and creates incidents can exceed the cap during a storm. Implement exponential backoff with jitter on every API call, and treat a 429 as a signal to slow the loop rather than to drop work.

**Clock and metric lag.** Metrics scraped on an interval will lag the alert that references them. A resolution-time query that looks back from the current instant may miss the most recent samples. Widen the query window by at least two scrape intervals.

**Drift in alert messages.** If a deployment changes an alert's message format, the historical resolution lookup returns `None` for every alert of that type, and the scoring model silently falls back to the default. Log when the fallback is used, and alert if the fallback rate rises.

## Decision checklist before you deploy

- Is the suppression threshold chosen from measured data, not intuition?
- Does every suppression write an auditable note?
- Is the triage layer's own health monitored independently of itself?
- Are LLM prompts free of secrets and customer data?
- Is there a documented way to disable the layer in one step during an incident?
- Has someone reviewed the grouping key against real alert messages, including hostnames and timestamps?
- Are all API calls wrapped in retry with backoff?
- Is the LLM call time-bounded with a fallback?

If any answer is no, the layer is not ready to suppress anything in production.

## FAQ

**Can this work with PagerDuty instead of Opsgenie?**
Yes. The scoring and grouping logic is platform-independent. Replace the client calls: PagerDuty's Events API v2 uses a `dedup_key` for grouping, which maps naturally onto the grouping key described above. The main difference is API shape and rate-limit behavior, not the design.

**What if the team uses Slack as the incident surface?**
The triage layer can post summaries to a channel instead of creating tickets. The trade-off is that Slack has no escalation policy, so you still need something to page a human. Use Slack for context and the paging platform for escalation.

**Should the LLM run locally?**
If alert content is sensitive, yes. A self-hosted model behind an OpenAI-compatible endpoint requires no code change beyond the base URL. Expect higher latency and lower summary quality than a large hosted model; measure both before committing.

**How do I tune the threshold?**
Start at 0.7, run in shadow mode, and lower it only when measured false suppressions are zero. Adjust in small increments and re-measure. Never tune during an incident.

## One thing to do in the next 30 minutes

Export your last 90 days of closed incidents from your paging platform's API and compute, per alert type, the median time from creation to resolution. Write the result to a CSV with two columns: alert type and median minutes. That single file is the historical-resolution input your scoring function needs, and building it now means you can run the layer in shadow mode tomorrow instead of waiting a month for data to accumulate.
