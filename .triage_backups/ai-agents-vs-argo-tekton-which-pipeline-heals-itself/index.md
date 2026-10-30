# AI agents vs Argo, Tekton: which pipeline heals itself?

## What "self-healing" actually means in a pipeline

A pipeline that heals itself is not a pipeline that never fails. It is a pipeline that can classify a failure, choose a response, and execute that response without a human in the loop. That definition matters because it separates three distinct capabilities that are often conflated:

1. **Detection** — something observes that a step failed or that a service is degrading.
2. **Classification** — the system decides whether the failure is transient (retry), permanent (roll back), or unknown (page a human).
3. **Action** — the system mutates the pipeline or the workload to recover.

Argo Workflows and Tekton Pipelines both give you a Kubernetes-native substrate for steps 1 and 3. Neither ships a classifier. The interesting engineering work lives in the classifier and in the failure modes that appear when detection and action race each other.

A common mistake is to build the classifier first and bolt on detection later. The reverse order is more robust: make the pipeline's failure states observable and deterministic, then add classification on top. If the pipeline cannot tell you *why* a step failed in a structured way, no amount of model inference will recover it reliably.

## The two substrates compared

Argo Workflows runs as a custom resource definition (CRD) with a controller that schedules each step as a pod. Its defining feature is a native DAG engine: you declare dependencies, and the controller handles scheduling, retries, timeouts, and artifact passing. Workflow status is a Kubernetes object, so `kubectl get workflow -w` and kube-state-metrics both work without extra plumbing.

Tekton Pipelines also runs as CRDs — `Task`, `Pipeline`, `TaskRun`, `PipelineRun` — but its execution model is explicitly sequential with declared dependencies via `runAfter`. Each `Task` runs as a pod, and Tekton's controller resolves the ordering. Tekton's strength is its alignment with Kubernetes RBAC and GitOps tooling: every resource is a CRD you can manage with `kubectl` and reconcile from a Git repository.

Neither engine is "better" in the abstract. The choice affects how you build the classifier, because the two engines expose different failure signals at different times.

| Dimension | Argo Workflows | Tekton Pipelines |
|---|---|---|
| Execution model | Native DAG, parallel branches first-class | Sequential tasks with `runAfter` dependencies |
| Failure signal | `WorkflowFailed` event, workflow status object | `TaskRun` status conditions, per-task events |
| Recovery primitive | Delete or patch the Workflow object | Patch `TaskRun`, or abort the `PipelineRun` |
| Sidecar placement | Per-step pod, agent runs separately | Sidecar can share the task pod |
| Status visibility | Workflow object + kube-state-metrics | `TaskRun` conditions, results, and records |
| Parallelism fit | Strong | Weaker; sequential model resists fan-out |

The sidecar placement row deserves emphasis. A sidecar that shares the task pod has direct visibility into the container's resource usage — RSS, CPU throttling, file descriptors. A sidecar in a separate pod sees only what the container exposes over the network or through the Kubernetes API. This difference shapes what kinds of failures each approach can detect early.

## Argo Workflows: detection and recovery

Argo exposes three integration points for an external classifier:

1. **Argo Events** listens to Kubernetes events and webhooks, then triggers workflows or forwards payloads to an agent.
2. **Workflow annotations** let you attach metadata to steps that the agent can read and mutate.
3. **Workflow status updates** stream back to the agent through the Kubernetes API.

A minimal workflow that deploys a service and leaves room for an agent to act on failure:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: WorkflowTemplate
metadata:
  name: deploy-with-agent
spec:
  entrypoint: main
  templates:
  - name: main
    steps:
    - - name: build
        template: build-image
    - - name: deploy
        template: deploy-service
        arguments:
          artifacts:
            image:
              from: "{{steps.build.outputs.artifacts.image}}"

  - name: build-image
    container:
      image: gcr.io/kaniko-project/executor:1.23.1
      command: [/kaniko/executor]
      args: ["--context=git://github.com/acme/app",
             "--destination=us-west-2.amazonaws.com/acme/app:{{workflow.parameters.tag}}"]

  - name: deploy-service
    container:
      image: bitnami/kubectl:1.29
      command: [kubectl]
      args: ["apply", "-f", "/manifests/deploy.yaml"]
    outputs:
      artifacts:
        image:
          path: /tmp/image-sha
```

An agent watching for `WorkflowFailed` events can read the failed pod's logs, classify the failure, and patch the workflow's labels. The following example uses a generic chat-completion call; substitute whichever model client your environment provides:

```python
import kubernetes.client
from kubernetes.client.rest import ApiException

async def handle_workflow_failed(event, kube_client, model_client, slack_client):
    workflow = event.object
    name = workflow.metadata.name
    namespace = workflow.metadata.namespace

    # Collect logs from the failed step's pod.
    pods = await kube_client.list_namespaced_pod(
        namespace=namespace,
        label_selector=f"workflows.argoproj.io/workflow={name}",
    )
    logs = ""
    for pod in pods.items:
        try:
            logs += await kube_client.read_namespaced_pod_log(
                name=pod.metadata.name, namespace=namespace
            )
        except ApiException:
            continue

    summary = await model_client.complete(
        prompt=(
            "Classify this CI failure as one of: transient, permanent, unknown.\n"
            "Respond with the single word only.\n\n"
            f"{logs[-8000:]}"
        )
    )
    verdict = summary.strip().lower()

    if verdict == "transient":
        await kube_client.patch_namespaced_custom_object(
            group="argoproj.io", version="v1alpha1",
            namespace=namespace, plural="workflows", name=name,
            body={"metadata": {"labels": {"self-heal": "retry"}}},
        )
    elif verdict == "permanent":
        await kube_client.patch_namespaced_custom_object(
            group="argoproj.io", version="v1alpha1",
            namespace=namespace, plural="workflows", name=name,
            body={"metadata": {"labels": {"self-heal": "rollback"}}},
        )
    else:
        await slack_client.post("#alerts", f"Unknown failure in {name}: {verdict}")
```

Two details in this snippet are load-bearing. First, the log payload is truncated (`logs[-8000:]`) because stack traces can exceed model context windows and because sending unbounded logs to a hosted model is both slow and a data-handling problem. Second, the classifier returns a constrained vocabulary (`transient`/`permanent`/`unknown`) rather than free text. Constrained output is the single most effective way to reduce misclassification; a model asked to "explain" a failure will produce explanations, not decisions.

## Tekton Pipelines: detection and recovery

Tekton exposes two hooks for self-healing:

1. **Sidecars** that run alongside each `Task` pod, scraping logs and metrics.
2. **Results and Records** that store structured outputs you can feed into anomaly detectors.

A minimal pipeline:

```yaml
apiVersion: tekton.dev/v1
kind: Pipeline
metadata:
  name: deploy-with-sidecar
spec:
  tasks:
  - name: build-image
    taskRef:
      name: kaniko
    params:
    - name: IMAGE
      value: "us-west-2.amazonaws.com/acme/app:$(params.TAG)"
  - name: deploy-service
    taskRef:
      name: kubectl-apply
    params:
    - name: manifest
      value: "/manifests/deploy.yaml"
    runAfter:
    - build-image
```

A sidecar that shares the task pod can poll a Prometheus endpoint and annotate the `TaskRun` when an error-rate threshold is crossed:

```go
package main

import (
	"context"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
)

func main() {
	config, err := rest.InClusterConfig()
	if err != nil {
		panic(err)
	}
	client, err := kubernetes.NewForConfig(config)
	if err != nil {
		panic(err)
	}
	ctx := context.Background()

	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()
	for range ticker.C {
		// Query Prometheus: rate(http_requests_total{job="app",status=~"5.."}[5m])
		// If the value exceeds the threshold, annotate the TaskRun.
		if errorRateExceedsThreshold(ctx) {
			_, err := client.TektonV1().TaskRuns("default").Patch(
				ctx,
				"deploy-service-taskrun-123",
				types.MergePatchType,
				[]byte(`{"metadata":{"labels":{"self-heal":"pause"}}}`),
				metav1.PatchOptions{},
			)
			if err != nil {
				// Log and continue; a failed patch should not crash the sidecar.
				continue
			}
		}
	}
}
```

The shared-pod model is Tekton's real advantage here. Because the sidecar sees the container's own cgroup metrics, it can catch a memory leak or CPU throttling before the container's health check fails. A sidecar in a separate pod cannot see those metrics directly and must rely on whatever the container exports.

The cost of that advantage is Tekton's eventual consistency. A `TaskRun` status update may take several seconds to propagate. If the sidecar annotates the `TaskRun` and the next `Task` in the chain has already started, the annotation arrives too late to prevent the rollout. This is the single most common failure mode in Tekton-based self-healing, and it is a race condition, not a bug in Tekton.

## Failure modes worth designing against

### Orphaned workflows with stuck finalizers

A workflow can fail during cleanup because a finalizer cannot complete — for example, a pod stuck in `Terminating` after a network partition. The classifier sees a failure and retries, but the finalizer persists, so the workflow cannot be deleted. The retry loop then amplifies the problem: each retry spawns pods that also cannot clean up.

Mitigations: set `ttlSecondsAfterFinished` on workflows so completed objects are garbage collected, and add a finalizer-cleanup step to the agent that patches out stuck finalizers after a bounded timeout. The cleanup command is:

```bash
kubectl patch workflow <name> --type=json \
  -p='[{"op": "remove", "path": "/metadata/finalizers"}]'
```

Do not run this unconditionally. Run it only after the workflow has exceeded its expected cleanup window, and log every invocation, because removing a finalizer can orphan real resources.

### Race conditions between detection and action

In Tekton, the sidecar's annotation may not propagate before the next `Task` starts. In Argo, a similar race exists between the agent's label patch and the controller's retry logic: the controller may retry before the agent's `rollback` label is visible.

The general fix is to make the recovery action a gate rather than a suggestion. Instead of annotating and hoping, insert an explicit `Task` or step that blocks on a condition resource, and have the classifier write to that condition. The pipeline then proceeds only when the condition is resolved. This converts a race into a wait, which is deterministic.

### Misclassification by the model

A classifier that returns free text will misclassify. A stack trace containing "timeout" may be labeled as an out-of-memory condition if the model associates the two. The practical mitigations are:

- **Constrain the output vocabulary** to a small enumerable set, as shown above.
- **Add a fallback**: if the model's output does not match a known pattern, default to `unknown` and page a human. Retrying an unknown error is usually worse than paging, because it delays the human's involvement.
- **Log every classification with its input** so misclassifications can be audited after the fact.

There is a real trade-off here. A conservative classifier that defaults to `unknown` reduces false positives but increases pages. A permissive classifier that defaults to `retry` reduces pages but can loop on permanent failures. The right default depends on whether your failures are more often transient or permanent — a question you can only answer by instrumenting your pipeline and counting.

### Event-loop backpressure

An agent processing `WorkflowFailed` events asynchronously will fall behind during a cluster-wide outage, exactly when it is most needed. A single-pod agent is also a single point of failure. Sharding the agent across pods with leader election and a priority queue helps, but the queue depth must be monitored; a deep queue means the agent is processing stale events whose workflows may already have been handled.

### Inconsistent metric scraping

A sidecar that scrapes `/metrics` will miss containers that expose metrics on a non-standard port or path. The fix is a fallback that tries multiple ports and paths, but that adds latency and complexity. A better approach is to standardize the metrics contract across all containers in the pipeline and validate it at build time, so the sidecar never encounters a container it cannot scrape.

## How to measure whether self-healing is working

Do not trust latency or accuracy figures from any article, including this one. Measure them in your own environment. The instrumentation is straightforward:

- **Detection rate**: inject failures deliberately (kill a pod, corrupt an image tag, introduce a dependency timeout) and count how many the classifier catches. A fault-injection harness with a fixed set of failure scenarios gives you a repeatable number.
- **Classification accuracy**: record every classifier verdict alongside the ground-truth label for the injected fault. Compute precision and recall per class (`transient`, `permanent`, `unknown`). Precision on `permanent` matters most, because a permanent failure misclassified as transient causes a retry loop.
- **Time to recovery**: measure wall-clock time from failure detection to successful recovery. Compare against the manual baseline — how long the same failure took a human to resolve before automation existed.
- **False-positive rate**: count alerts that a human acknowledged and dismissed without action. This is the number that determines on-call quality of life.
- **Cost per run**: compare CI minutes and compute spend before and after. Argo and Tekton both run on your existing Kubernetes cluster, so the marginal cost is pod time, not a separate CI service.

A useful discipline is to run the fault-injection harness on a schedule, not just once. Self-healing pipelines rot: a classifier tuned for one failure distribution drifts as the application changes. A weekly injection run catches that drift before it reaches production.

## A decision checklist

Before choosing between Argo and Tekton for a self-healing pipeline, answer these questions:

- **How parallel is your pipeline?** If it fans out into many concurrent branches, Argo's DAG engine is a better fit. If it is mostly linear, Tekton's sequential model is simpler to reason about.
- **Do you need in-container resource visibility?** If you want to catch memory leaks and CPU throttling before health checks fail, Tekton's shared-pod sidecar model has an advantage.
- **How strict is your RBAC model?** Tekton maps naturally onto Kubernetes RBAC. Argo's controller needs broader permissions to manage pods across namespaces.
- **What is your GitOps story?** Both engines use CRDs, so both work with GitOps tools. Tekton's resources are more granular, which can make reconciliation easier or noisier depending on your tooling.
- **How will you classify failures?** If you plan to use a hosted model, budget for latency, cost, and data-handling review. A local model avoids the data-handling question but adds GPU infrastructure.
- **What is your fallback?** Every classifier needs a default for unrecognized failures. Decide whether that default is `retry`, `rollback`, or `page`, and make it explicit in the code.

## FAQ

**Can Argo and Tekton coexist in one cluster?** Yes. They are independent CRDs with independent controllers. Running both is common during a migration, though it doubles the surface area your team must maintain.

**Does self-healing replace on-call?** No. It reduces the number of pages and the time to recovery, but the classifier will encounter failures it has never seen. On-call remains necessary for the `unknown` class.

**How do I prevent a retry loop?** Cap retries at the workflow level and make the classifier's `retry` action increment a counter. When the counter exceeds the cap, escalate to `unknown` and page. Never let a retry action be unbounded.

**Is a hosted model safe for CI logs?** That depends on your data-handling policy. CI logs can contain secrets, internal hostnames, and customer data. Redact before sending, or use a local model. This is a policy question, not a technical one, and it should be answered before the classifier is built.

**What about Argo Events vs. polling the API?** Argo Events is event-driven and lower-latency. Polling the Kubernetes API is simpler and has fewer moving parts. For low-volume pipelines, polling is often sufficient; for high-volume, event-driven is worth the operational cost.

## Take action

In the next 30 minutes, instrument one pipeline to count its failure classes. Add a label to every failed run recording whether the failure was transient, permanent, or unknown, based on what the human who resolved it observed. Run the pipeline for a week, then look at the distribution. That distribution — not any article's benchmark — is the input your classifier needs.
