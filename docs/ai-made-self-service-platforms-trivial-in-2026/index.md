# AI made self-service platforms trivial in 2026

AI assistants can draft Terraform, Kubernetes manifests, and CI pipelines from a short prompt. That changes the bottleneck for self-service platforms: the hard part is no longer writing YAML, it is proving that generated infrastructure is correct, least-privileged, and compatible with the cluster it will actually run against. This article covers the failure modes that show up when teams wire an LLM into a deployment path, and the guardrails that catch them.

## Why generated infrastructure fails in ways unit tests miss

An LLM produces text that looks like a valid manifest. It does not produce a resource that has been applied to your cluster, evaluated against your admission controllers, or reconciled by your controllers. Three failure classes recur.

**Dependency and ordering errors.** Generated Terraform can be syntactically valid and pass a schema check while encoding the wrong dependency graph. A common example: an IAM role and an EKS node group are emitted without an explicit `depends_on`, so Terraform's implicit ordering does not guarantee the role exists before nodes try to assume it. The plan is valid; the apply races.

**Context pulled from untrusted input.** If a prompt includes a ticket URL or pasted text, the model may treat that content as instructions rather than context. A ticket that says "expose /health for monitoring" can turn into an ingress rule the author did not intend. This is prompt injection through metadata, and it is not visible in the generated diff unless someone reads the source of the context.

**API version and schema skew.** Models are trained on a snapshot of API surfaces. A manifest targeting `gateway.networking.k8s.io/v1alpha2` may apply cleanly against a cluster that only serves `v1`, and the controller will simply never reconcile it. The apply succeeds; the resource does nothing.

The common thread: the generated artifact is validated as text, but the failure occurs at runtime. Guardrails have to validate against the runtime environment, not just the file.

## Validate generated Terraform with policy and a real plan

The cheapest gate is a policy check against the JSON plan, not the source. Produce the plan in machine-readable form first:

```bash
terraform init -input=false
terraform plan -out=tfplan -input=false
terraform show -json tfplan > terraform_plan.json
```

Then evaluate policy against `terraform_plan.json`. Open Policy Agent (OPA) is the general-purpose option; here is a policy that rejects IAM roles with wildcard actions and roles not scoped to a specific service principal.

```rego
# iam_deny_wildcard.rego
package terraform

deny[msg] {
    role := input.resource.aws_iam_role[_]
    role.statement[_].actions[_] == "*"
    msg := sprintf("IAM role %s has wildcard actions", [role.name])
}

deny[msg] {
    role := input.resource.aws_iam_role[_]
    not role.assume_role_policy.statement[_].principal.Service == ["eks.amazonaws.com"]
    msg := sprintf("IAM role %s is not restricted to EKS service", [role.name])
}
```

Run it in CI so a pull request fails when the policy is violated:

```yaml
# .github/workflows/opa-terraform.yml
name: OPA Terraform Policy Gate
on:
  pull_request:
    paths: ["terraform/**"]
jobs:
  gatekeeper:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: open-policy-agent/setup-opa@v2
        with:
          version: v0.68.0
      - run: |
          opa eval --data iam_deny_wildcard.rego --input terraform_plan.json \
            --format pretty --fail-defined
```

Two notes on the policy. First, the `deny` rules above assume the plan JSON has been reshaped into a flat `resource` map; OPA evaluates whatever structure you feed it, so either normalize the plan or write rules against Terraform's actual `planned_values` shape. Second, `--fail-defined` makes `opa eval` exit non-zero when any `deny` rule produces output, which is what turns the policy into a CI gate.

For a worked example of the dependency failure: suppose the generated plan contains an `aws_iam_role` named `eks-node-role` and an `aws_eks_node_group` that references it. If the node group's `node_role_arn` is a literal string rather than a reference to `aws_iam_role.eks-node-role.arn`, Terraform has no edge in the graph and may create the node group first. A policy rule can require that the node group's role ARN be a reference:

```rego
deny[msg] {
    ng := input.resource.aws_eks_node_group[_]
    not startswith(ng.node_role_arn, "${aws_iam_role.")
    msg := sprintf("node group %s must reference an aws_iam_role ARN, not a literal", [ng.name])
}
```

That is a static check on the plan, and it catches the race before apply.

## Gate Kubernetes manifests against the cluster's schema

For Kubernetes, validate against the API versions your cluster actually serves rather than the versions the model remembers. `kubeconform` is a maintained schema validator; run it against the cluster's OpenAPI schema or a pinned schema bundle.

```bash
# Validate generated manifests against a pinned schema set
kubeconform -strict -summary \
  -schema-location default \
  -kubernetes-version 1.28.0 \
  ./generated/
```

The `-kubernetes-version` flag is the important part. Pinning it to the target cluster's version means a manifest using a removed or alpha API fails validation instead of silently applying.

A CI step that runs before merge:

```yaml
# .github/workflows/k8s-validate.yml
- name: Validate manifests
  run: |
    docker run --rm -v "$PWD:/work" -w /work \
      ghcr.io/yannh/kubeconform:latest \
      -strict -summary -kubernetes-version 1.28.0 ./generated/
```

This is a schema check, not a semantic one. It will not catch a `Gateway` resource that references a controller that is not installed. For that, the runtime check below is what matters.

## Verify at runtime, not just at merge

Schema and policy checks confirm the artifact is well-formed and permitted. They do not confirm the controller reconciled it or that the service behaves. Add a post-apply verification step.

For a Gateway or Ingress resource, the check is whether the resource reports a ready condition:

```bash
kubectl wait --for=condition=Programmed \
  gateway/payments-gateway -n payments --timeout=120s
```

If the Gateway controller never reconciles the resource, the wait times out and the pipeline fails, which is the signal the schema check could not give.

For a rollout, use a canary with an automated analysis step. Argo Rollouts supports an `AnalysisTemplate` that queries Prometheus and gates promotion on the result:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata:
  name: payments-canary-analysis
spec:
  metrics:
    - name: latency
      interval: 1m
      successCondition: "p95 <= 200"
      failureLimit: 3
      provider:
        prometheus:
          address: http://prometheus-operated.monitoring.svc:9090
          query: 'histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{service="payments-canary"}[2m])) by (le))'
```

The rollout references this template so the canary step pauses until the metric passes or the failure limit is hit. The threshold (`p95 <= 200`) is a policy decision, not a default; set it from your own SLO.

A synthetic load test can run before the first canary weight is set. Locust is one option:

```python
# locustfile.py
from locust import HttpUser, task, between

class PaymentsUser(HttpUser):
    wait_time = between(0.5, 2.0)

    @task
    def health_check(self):
        self.client.get("/health", headers={"Host": "payments.internal"})

    @task(3)
    def create_payment(self):
        self.client.post(
            "/v1/payments",
            json={"amount": 100, "currency": "KES"},
            headers={"Host": "payments.internal"},
        )
```

Run it headless in CI:

```yaml
- name: Synthetic canary test
  run: |
    docker run --rm \
      -v "$PWD/locust:/mnt/locust" \
      ghcr.io/locustio/locust:2.20.0 \
      --host https://istio-ingressgw \
      --locustfile /mnt/locust/locustfile.py \
      --headless -u 1000 -r 100 --run-time 5m \
      --csv=locust_results
```

The workflow should fail the rollout if the parsed CSV shows p95 above your threshold, an error rate above your threshold, or any 5xx. Those thresholds are yours to set; the point is that the gate runs against a live endpoint, not a mock.

## Sanitize the context the model sees

Prompt injection through metadata is best handled before generation. A sanitizer that strips URLs and restricts context sources to trusted systems removes the class of failure where a ticket description becomes an instruction.

```python
import re

ALLOWED_SOURCES = ("github.com", "docs.internal", "confluence.internal")

def sanitize_prompt(prompt: str) -> str:
    # Drop bare URLs; context must come from allowlisted systems.
    return re.sub(r"https?://\S+", "[redacted-url]", prompt)
```

This is a blunt filter and it will remove useful links. The tradeoff is deliberate: if the model cannot fetch arbitrary URLs, it cannot be steered by their contents. Teams that need ticket context should fetch it themselves through an API, strip the fields they do not want, and pass a structured summary rather than a URL.

## A decision checklist before wiring an LLM into deploys

- **What is the model allowed to produce?** If it can emit IAM policies or ingress rules, those need policy gates. If it can only emit a diff for human review, the risk surface is smaller.
- **What does the plan actually contain?** Run `terraform show -json` and inspect `planned_values` before trusting any policy check. A policy that reads the wrong field passes everything.
- **Which API versions does the target cluster serve?** Pin the validator to that version. Do not rely on the model's remembered schema.
- **What happens when the controller does not reconcile?** Add a `kubectl wait` on a ready condition, or an equivalent, so silent no-ops fail the pipeline.
- **What is the rollback path?** A canary with an analysis template gives an automated rollback trigger. Without one, a bad rollout needs a human to notice.
- **Where does the prompt context come from?** If any of it is user-supplied or fetched from a URL, sanitize it.

## How to measure the effect of the guardrails

Do not adopt a benchmark from an article. Measure your own pipeline. Instrument these:

- **Gate latency.** Time each step (`terraform plan`, policy eval, schema validation, synthetic test) and record the sum. Compare against your review SLA, not against a number from elsewhere.
- **Escape rate.** Count deploys that passed all gates and still required a rollback. This is the number the guardrails exist to reduce.
- **False-positive rate.** Count deploys blocked by a gate that turned out to be safe. A gate with a high false-positive rate gets disabled, so track it.
- **Time to first failure signal.** For a failing deploy, measure the interval between apply and the first failing check. A short interval means the runtime check is doing its job.

A simple starting point is to log one line per pipeline run with the gate name, duration, and pass/fail, then chart escape rate over time. If escape rate does not fall, the gates are checking the wrong thing.

## FAQ

**Does a passing OPA policy mean the Terraform is correct?**
No. It means the plan satisfies the rules you wrote. Dependency ordering, provider version drift, and runtime behavior are outside what a plan-level policy can see.

**Can I skip schema validation if I use a policy engine?**
They check different things. Policy engines check your rules against the plan; schema validators check the manifest against the API surface. A manifest can satisfy every policy and still target a removed API version.

**Is a synthetic load test a substitute for production canary analysis?**
No. A synthetic test exercises known paths with synthetic traffic. Canary analysis observes real traffic. Run both if the service is user-facing.

**How do I handle models that hallucinate provider attributes?**
Pin the provider version in the context you pass to the model, and validate the plan against that provider's schema. If the attribute does not exist in the pinned version, `terraform validate` fails before apply.

## Take one action in the next 30 minutes

Pick one generated manifest in your repository and run a schema check against your cluster's actual API version:

```bash
kubeconform -strict -summary -kubernetes-version 1.28.0 ./path/to/manifest.yaml
```

Replace `1.28.0` with your cluster's server version (`kubectl version --short` reports it). If the manifest fails, you have found a silent no-op before it reached production. If it passes, add the same command as a CI step so the next generated manifest is checked automatically.
