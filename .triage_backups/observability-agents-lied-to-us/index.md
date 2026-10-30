# Observability Agents as Application Code

Most observability guidance stops at the happy path: install an agent, expose metrics, collect traces, set up alerts. That playbook was written for a world of long-lived monoliths. It says much less about what happens when the collectors themselves become a distributed system you did not plan for.

## The conventional wisdom and where it runs out

The standard playbook is: install an agent, expose metrics, collect traces, set up alerts, and you are done. This is reasonable advice, and for many systems it is sufficient. What it understates is the operational surface of the agents themselves once you move from one host to a fleet.

The failure mode is rarely telemetry quality. It is the collection tier's own resource consumption, configuration drift, and restart behavior. A typical symptom is an agent that cannot reach its own scrape target:

```
level=fatal msg="Failed to start scrape pool" err="failed to create scrape pool: dial tcp 127.0.0.1:9090: connect: connection refused"
```

That error describes a symptom. The cause is usually upstream: the agent was OOM-killed, the target had not finished starting, or a network policy change blocked the loopback path. Reading the error as "the target is down" sends you debugging the wrong component.

The mental model worth challenging is "agents are lightweight." That holds for a single host running one exporter. It stops holding when a collector is deployed per node and each instance is configured to scrape host metrics, kubelet metrics, container runtime metrics, and application endpoints on the same interval. The agent's footprint then scales with cluster size and object count, not with the size of the node it runs on.

The opposing view is worth stating fairly: if you refuse to run anything on the host or in a sidecar, you are choosing blind spots. Agents are necessary. The question is not whether to run them but whether you treat them as set-and-forget infrastructure or as software you operate.

## What happens when you follow the standard advice

Consider a representative progression.

Start with a monolith on one instance. A node exporter process uses a small, roughly constant amount of memory and a negligible fraction of a CPU. Nothing to manage.

Move to Kubernetes and the same exporter becomes a DaemonSet. Now there is one pod per node. If the exporter uses roughly 30 MiB per pod and you have 30 nodes, that is about 900 MiB of cluster memory for host metrics alone. The arithmetic is simple and worth doing before deployment: per-pod footprint multiplied by node count.

Add a collector DaemonSet and the picture changes more sharply. A collector's default configuration commonly enables a broad set of receivers and exporters. On each node it may scrape host metrics, kubelet metrics, container runtime metrics, and application endpoints. If all of those share one scrape interval and one queue, a slow endpoint backs up the queue for every other endpoint.

A concrete cascade looks like this. The kubelet's metrics endpoint has a request timeout. When the collector's scrape queue is saturated, requests to that endpoint exceed the timeout and samples are dropped:

```
level=error msg="Scrape failed" name=kubelet duration=12.4s err="Get \"http://127.0.0.1:10255/metrics\": context deadline exceeded (Client.Timeout exceeded while awaiting headers)"
```

The instinct is to raise the kubelet timeout. That treats the symptom. The cause is that the collector is doing more work per interval than its queue and worker pool can absorb, so a slow target starves the rest. The fix is to separate scrape jobs, raise the queue, or reduce what the collector is asked to do per node.

Configuration drift is the second failure mode. A single ConfigMap shared across every collector instance is fine at ten services. At fifty services with different intervals, relabeling rules, and exporters, the file grows into something no one reviews carefully. Updating it restarts every collector pod, and each restart opens a window where no telemetry is collected. The failure is silent: no error, just gaps in dashboards that are easy to attribute to the application.

The operational cost is the part that rarely appears in benchmarks. The measurable version is on-call load attributable to the collection tier: pod restarts, OOM kills, misconfigured relabeling, missed scrapes. You can measure this directly by tagging incidents with a "telemetry" category and counting them over a quarter. That number, not a vendor figure, is the one that should drive your decision.

## How to measure agent overhead honestly

Before adopting any mental model, get your own numbers. The commands below are the ones that produce evidence rather than opinion.

**Per-pod resource use.** On a cluster with metrics-server installed:

```
kubectl top pods -n observability --sort-by=memory
kubectl top pods -n observability --sort-by=cpu
```

Compare the collector's usage against its configured requests and limits, not against an absolute threshold. A collector at 400 MiB with a 512 MiB limit has less headroom than one at 600 MiB with a 2 GiB limit.

**Scrape health.** The collector exposes its own internal metrics. The relevant signals are the number of dropped spans or metric points, the queue length, and the fraction of failed scrapes. Instrument those and alert on them the same way you would alert on an application.

**Restart frequency.** Count container restarts over a week:

```
kubectl get pods -n observability -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.containerStatuses[*].restartCount}{"\n"}{end}'
```

A collector that restarts more than a handful of times a week is not stable, regardless of how it looks in a dashboard.

**Latency impact on targets.** If you scrape an endpoint, you can measure how long that scrape takes from the target's perspective. Compare the p99 of your metrics endpoint latency before and after enabling a new scrape job. The difference is the cost you are paying for that job.

**Sample loss.** If you use a pull-based system, compare the number of samples you expect (targets multiplied by interval) with the number actually stored. The gap is your loss rate. If you use a push-based pipeline, the collector's own counters report dropped data points directly.

None of these require a benchmark suite. They require that you look at the agents as systems with their own SLOs.

## A different mental model: agents as application code

The alternative to treating agents as infrastructure is to treat them as application code. That has concrete consequences:

- Agents are versioned, tested, and deployed through the same pipeline as services.
- Agents have their own resource budgets, SLOs, and error budgets.
- Agents are observable via their own telemetry, not only the telemetry they collect.

In practice this means:

1. Pin agent versions and the exact set of components enabled in each build.
2. Set memory and CPU requests and limits from load testing rather than accepting defaults.
3. Run agents in a dedicated namespace with pod disruption budgets and priority classes.
4. Monitor the agents' own process metrics alongside application metrics.
5. Use separate scrape intervals for agent telemetry and application telemetry.
6. Treat the agent configuration as versioned code, reviewed like any other change.

This is harder to adopt because it adds work to the deployment pipeline. The payoff is that when the telemetry pipeline breaks, you already have the instrumentation to diagnose it instead of discovering that your observability tooling is itself unobservable.

## Worked example: sizing a collector from first principles

The numbers below are illustrative, not measured. Substitute your own.

Assume a cluster with 40 nodes. You plan a collector DaemonSet, one pod per node. You expect each pod to handle 2,000 metric points per second and to buffer up to 30 seconds of data during a downstream outage.

Step 1: estimate steady-state memory. Suppose your load test shows the collector uses roughly 150 MiB of heap at 2,000 points per second with the receivers and exporters you have enabled. Add a 2x safety factor for bursts: 300 MiB.

Step 2: estimate buffer memory. If a point is roughly 200 bytes in memory and you buffer 30 seconds at 2,000 points per second, that is 2,000 × 30 × 200 bytes, or about 12 MB. This is small relative to heap, which is why heap, not buffer, usually drives the limit.

Step 3: set requests and limits. Request 256 MiB, limit 512 MiB. The request guarantees scheduling; the limit prevents a runaway collector from evicting application pods on the same node.

Step 4: verify. After deployment, watch actual usage against the limit for a week. If usage sits at 80 percent of the limit under normal load, the limit is too tight for the next traffic spike.

The point of the exercise is not the specific numbers. It is that a collector's resource budget should come from a load test and an explicit buffer assumption, not from a default in a manifest.

## Failure modes worth designing against

**Queue saturation causing cross-target starvation.** One slow target fills a shared queue and starves fast targets. Mitigation: separate scrape jobs with independent queues, or reduce the number of targets per collector.

**OOM kill cascades.** A collector is killed, telemetry stops, and the resulting gap is attributed to the application. Mitigation: set limits with headroom, alert on memory approaching the limit, and ensure the collector is not the lowest-priority workload on the node.

**Restart storms from configuration changes.** A shared ConfigMap change restarts every collector at once. Mitigation: use rolling updates, shard collectors by function, and avoid a single config that all instances share.

**Silent sample loss.** The collector drops data under pressure and reports it only in its own metrics. Mitigation: scrape the collector's internal metrics and alert on drop counters.

**Permission and network coupling.** Agents often need broad permissions and host network access, which interacts badly with network policies and least-privilege service accounts. Mitigation: scope permissions to what the collector actually reads, and test policy changes against the agent.

## When the conventional wisdom is right

Treating agents as first-class code is not universally necessary. The standard playbook is fine when:

- You run a small number of services, roughly fewer than twenty.
- Your services are long-lived, with few ephemeral pods.
- Telemetry volume is low relative to the collector's capacity.
- You use a managed agent where the vendor handles scaling and upgrades.

In those cases the agent's footprint is a rounding error and the simplicity of the standard setup is worth more than the rigor of treating it as code. A single monolith on one instance with a vendor agent using a small, stable amount of memory does not need a resource budget review.

The trap is assuming that what works for a small monolith will scale unchanged to a distributed system. It usually will not, and the failure is gradual rather than dramatic.

## Decision checklist

Use this to decide whether to adopt the agents-as-code model. The thresholds are illustrative; calibrate them against your own measurements.

| Criterion | Agents as infrastructure | Agents as code |
|---|---|---|
| Service count | Fewer than 20 | 20 or more |
| Service lifetime | Hours to days | Seconds to minutes |
| Telemetry volume | Well under collector capacity | Near or above capacity |
| Cluster size | Small, few nodes | Dozens of nodes or more |
| Team size | 1–2 people on call | 3 or more |
| Tolerance for sample loss | High | Low |
| Appetite for operational overhead | Low | High |

If several criteria fall in the right column, treat agents as code. If most fall in the left, the conventional playbook is likely sufficient.

## Common objections

**"Agents are supposed to be lightweight. If they are not, you configured them wrong."**

The lightweight claim holds for a single exporter on a single host. It weakens when a collector is deployed per node and configured to scrape many targets. The footprint scales with the number of targets and the number of enabled components, not with node capacity. That is a configuration property, not a defect, and it is worth measuring rather than assuming.

**"Managed services solve this."**

Managed agents reduce operational overhead but do not remove it. A container agent typically still runs as a DaemonSet, may require host network access, and needs a service account with permissions to read cluster state. Those are configuration and security decisions you still own. The vendor abstracts the agent's resource usage, but the integration surface remains yours to manage. For small teams this trade is usually good. For large fleets, the overhead shifts to managing API limits, cardinality, and pricing.

**"Just disable the exporters you do not need."**

In principle this is correct. In practice, default configurations often enable more than the operator realizes, and the configuration file grows until no one reviews it carefully. The reliable approach is to build a minimal collector image containing only the components you intend to run, so that an unneeded exporter is not present to be accidentally enabled.

**"Monitor the agents and set alerts."**

Monitoring is necessary but reactive. The failure that hurts is not the collector running out of memory; it is the gap in telemetry that follows, which makes the incident harder to diagnose. Treating agents as code moves the problem upstream: you catch resource pressure before it becomes an outage, and you have the agent's own telemetry to explain what happened.

## A starting configuration

A reasonable default for a new stack:

1. Start with a centralized collector deployment rather than a per-node DaemonSet, unless you have a specific reason to collect node-local data.
2. Pin the collector version and build a minimal image with only the receivers and exporters you need.
3. Set resource requests and limits from a load test, with an explicit buffer assumption.
4. Use push-based telemetry where the protocol supports it, and keep pull-based scraping for targets that only expose a pull endpoint.
5. Scrape the collector's own metrics with a separate job and alert on dropped data, queue length, and memory approaching the limit.
6. Version the collector configuration alongside your services and roll it out gradually.

## Summary

The standard observability playbook covers the happy path well and the collection tier poorly. The failure modes that matter at scale are the agents' own resource consumption, configuration drift, and restart behavior, and they are measurable with tools you already have.

Treating agents as application code is not a universal requirement. It is a response to specific conditions: many services, short-lived workloads, high telemetry volume, and low tolerance for sample loss. When those conditions hold, the agents are a distributed system in their own right, and they deserve the same versioning, budgeting, monitoring, and failure testing as anything else you run.

## Do this in the next 30 minutes

Run `kubectl top pods -n observability --sort-by=memory` and compare each collector pod's usage against its configured memory limit. If any pod is using more than 80 percent of its limit under normal load, raise the limit or reduce what that collector is asked to do, and note the change so you can verify it after the next traffic peak.
===END===
