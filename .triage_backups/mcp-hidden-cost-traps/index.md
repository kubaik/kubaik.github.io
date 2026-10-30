# MCP hidden cost traps

Managed container platforms (MCPs) hide Kubernetes control-plane complexity behind a small set of knobs. Those knobs have defaults, and the defaults are chosen for broad compatibility rather than for cost or security. The result is a class of failure that is hard to diagnose because two unrelated-looking symptoms appear at once: a step change in the monthly bill and a burst of authorization errors from the secret store.

This article covers the coupling between those two symptoms — how secret-lease churn can look like sustained load to an autoscaler, and how missing cost attribution hides which node group is responsible.

## The symptom pair and why it misleads

A common report looks like this: the cost report shows a jump over the previous cycle, and simultaneously the monitoring dashboard shows a spike in `PermissionDenied` errors from the secret backend. A representative log line:

```
2026-09-12T14:23:07Z ERROR failed to fetch secret from Vault: permission denied (code=403)
```

The instinct is to treat these as two tickets. The cost increase looks like a capacity problem; the 403s look like an IAM problem. In practice they are frequently one problem, and the reason is that the autoscaler does not know *why* CPU went up.

The autoscaler reacts to a metric. If a workload's CPU rises because it is doing useful work, scaling out is correct. If CPU rises because every request is failing fast and retrying in a tight loop, scaling out is actively harmful: it adds capacity to a workload that will never succeed, and the added capacity is billed until the scale-down delay elapses.

A second source of confusion is token caching. Many secret clients cache a token for its full lease — a common default is a 24-hour lease for a service-account or AppRole token. When that lease expires, every replica of every service that shares the credential fails at roughly the same moment. The resulting retry storm is synchronized across the fleet, which makes it look like a load event rather than an expiry event.

The third source of confusion is cost attribution. If node groups are not tagged, the cost explorer aggregates everything under one line item, and there is no way to tell whether the increase came from the node group you changed or one you forgot about.

## Root causes

Three independent misconfigurations combine into a self-reinforcing loop.

**1. Aggressive scale-in behavior.** A short scale-in delay means transient spikes — including failure-induced spikes — translate into node additions that persist. The relevant parameters are the scale-in cooldown and the CPU target utilization. A low target (for example 0.6) means the autoscaler adds nodes while the fleet is still comfortably below saturation.

**2. Over-permissive credentials and long lease lifetimes.** If a single role is reused across many microservices, and the token lease is long, then expiry is both fleet-wide and infrequent enough that nobody has instrumented for it. If the same role also carries broad permissions — for example a binding that grants `cluster-admin` within the namespace — then a leaked credential is not just a read incident.

**3. Missing cost tags.** Without a tag on each node group, cost data cannot be joined to configuration. This is what turns a five-minute diagnosis into a week of guessing.

A plausible chain of events:

1. A developer commits a manifest or image that references an environment variable holding a credential.
2. At runtime the container cannot read the intended path in the secret store, because the role binding only covers a narrower path.
3. The application falls back to a placeholder value, fails every outbound call, and retries.
4. The retry traffic raises CPU above the autoscaler target, and nodes are added.
5. The added nodes are billed for at least the scale-in cooldown duration after the incident ends.
6. Separately, the committed credential is scraped from the repository history.

Step 4 is the coupling. Steps 1 and 3 are ordinary bugs; step 4 is what makes an ordinary bug expensive.

## Fix 1 — Make the autoscaler ignore failure traffic

The first fix is to raise the CPU target and lengthen the scale-in cooldown so that short-lived failure spikes do not convert into billed capacity. The Terraform below configures a node group and a target-tracking policy. Note that `aws_eks_node_group` does not expose `autoscaling_group_name` as an attribute in the way the snippet below assumes; in practice the autoscaling policy is attached to the ASG that backs the node group, which is discoverable via the node group's `resources` block or by referencing the ASG directly. Confirm the correct reference for your provider version before applying.

```hcl
resource "aws_eks_node_group" "prod" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "prod-standard"
  node_role_arn   = aws_iam_role.eks_node_role.arn
  subnet_ids      = aws_subnet.private[*].id

  scaling_config {
    desired_size = 3
    min_size     = 2
    max_size     = 6
  }

  tags = {
    CostCenter  = "FinTech"
    Environment = "prod"
  }

  launch_template {
    id      = aws_launch_template.prod.id
    version = "$Latest"
  }
}

resource "aws_autoscaling_policy" "cpu_target" {
  name                   = "cpu-target-policy"
  autoscaling_group_name = aws_eks_node_group.prod.resources[0].autoscaling_groups[0].name
  policy_type            = "TargetTrackingScaling"

  target_tracking_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ASGAverageCPUUtilization"
    }
    target_value       = 0.75
    scale_in_cooldown  = 900
    scale_out_cooldown = 300
  }
}
```

What each change does:

- `target_value = 0.75` raises the utilization threshold, so the autoscaler tolerates more headroom before adding nodes.
- `scale_in_cooldown = 900` keeps a newly added node for 15 minutes before it becomes eligible for removal. This is a trade: it costs money during genuine short spikes, but it prevents flapping when the spike is caused by a failure that will recur.
- The `CostCenter` tag is what makes the next step possible.

### How to measure whether this helped

Do not trust a claimed percentage. Instrument it:

1. Record the autoscaling group's `DesiredCapacity` and `InServiceCapacity` on a fixed interval (a CloudWatch metric alarm, a Prometheus exporter, or a cron job calling the provider API). Store the series.
2. Record the secret store's error count as a separate series, with the error code as a label.
3. Overlay the two series for a two-week window that includes at least one token-expiry event.
4. The metric that matters is node additions that occur within 10 minutes of a spike in secret errors. Before the fix, this count is nonzero; after the fix, it should be zero.

That comparison — node additions correlated with secret errors — is the actual verification. A dollar figure is a downstream consequence and depends on instance type, region and duration.

## Fix 2 — Shorten leases and stop committing credentials

The second fix addresses the trigger. Two parts: prevent credentials from entering the repository, and reduce the blast radius when a lease expires.

The Python below reads a secret using AppRole authentication and renews the token on an interval shorter than its lease. The `hvac` client is a real, widely used Python client for the HashiCorp Vault HTTP API; verify the method names against the version you install.

```python
import os
import time

import hvac

client = hvac.Client(url=os.getenv("VAULT_ADDR"))

client.auth.approle.login(
    role_id=os.getenv("VAULT_ROLE_ID"),
    secret_id=os.getenv("VAULT_SECRET_ID"),
)

secret = client.secrets.kv.v2.read_secret_version(path="payments/api_key")
api_key = secret["data"]["data"]["key"]

# Renew well before the lease expires so that expiry is never fleet-synchronized.
while True:
    time.sleep(300)
    client.auth.token.renew_self(increment="15m")
```

Two properties matter here. First, the renewal interval (5 minutes) is much shorter than the lease increment (15 minutes), so a single missed renewal does not immediately produce an outage. Second, because renewal is per-process and jittered by process start time, expiry is no longer synchronized across the fleet.

For the CI side, add a secret scanner to the pipeline. Any scanner that supports a severity threshold and a non-zero exit code works; the specific product matters less than the gate. A generic GitHub Actions step:

```yaml
- name: Scan for secrets
  run: |
    secret-scanner scan --severity-threshold high --exit-code 1 .
```

The important detail is `--exit-code 1`. A scanner that reports findings but exits zero is a dashboard, not a control.

### How to measure whether this helped

- Count `token_renew_success` entries in the secret store's audit log per service per hour. A healthy service renews on a predictable cadence; a gap indicates a process that has stopped renewing.
- Count secret-read failures by error code. A drop in `403` counts that is not accompanied by a drop in total requests indicates the fix worked.
- Re-run the node-addition correlation from Fix 1. If the coupling is broken, the correlation disappears even if occasional 403s remain.

## Fix 3 — Close the network and permission surface

The third fix is environment-specific, but the principle is uniform: reduce what is reachable and reduce what a compromised workload can do.

On AWS, the EKS control-plane endpoint can be configured for private access only. The `eksctl` configuration below does that and disables public access:

```yaml
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig
metadata:
  name: fintech-prod
  region: us-east-1
  version: "1.28"

vpc:
  clusterEndpoints:
    publicAccess: false
    privateAccess: true

nodeGroups:
  - name: prod-standard
    instanceType: m5.large
    desiredCapacity: 3
    iam:
      withAddonPolicies:
        autoScaler: true
```

Note that the field for endpoint access lives under `vpc.clusterEndpoints` in the `eksctl` schema, not as a top-level `privateCluster` block. Validate against the schema version you are using with `eksctl utils schema` or the project's published schema before applying.

On GCP, the equivalent concern is egress between zones and the reachability of the control plane. A regional cluster with IP aliasing and a NAT gateway avoids per-zone egress surprises:

```bash
gcloud container clusters create health-prod \
  --region us-central1 \
  --release-channel stable \
  --enable-ip-alias \
  --node-locations us-central1-a,us-central1-b \
  --num-nodes 4
```

The `--node-locations` flag pins nodes to two zones rather than spreading across all zones in the region, which reduces cross-zone traffic. Whether that is a saving or a resilience cost depends on your availability requirements — two zones is a real reduction in failure tolerance compared to three.

### How to measure whether this helped

- Query the API server audit log for requests whose source IP is outside your VPC CIDR. After disabling public access, this set should be empty.
- Break down network egress by destination zone using the provider's flow logs. Cross-zone traffic attributable to the cluster should drop.
- Confirm that the control plane is still reachable from your CI runners and bastion. A private endpoint that nobody can reach is an outage, not a fix.

## A worked estimate, with assumptions stated

The following is an illustrative calculation, not a measurement. It shows the arithmetic so you can substitute your own numbers.

Assume a node type billed at $0.096 per hour, and a scale-in cooldown of 900 seconds (15 minutes). If a failure-induced spike adds 2 nodes and the spike lasts 3 minutes, the nodes remain for at least the cooldown period after they become idle:

- Nodes added: 2
- Minimum billed duration per node: 15 minutes = 0.25 hours
- Cost per incident: 2 × 0.25 × $0.096 = $0.048

That is trivial for one incident. The cost becomes material through frequency, not magnitude. At 200 such incidents per month:

- 200 × $0.048 = $9.60 per month

Still small. The figure only becomes large when the failures are continuous rather than intermittent — for example, a token that expires and is never renewed, leaving a service in a permanent retry loop. In that case the added nodes are billed continuously, and the monthly cost is:

- 2 nodes × 730 hours × $0.096 = $140.16 per month

The lesson is that the arithmetic is easy; the hard part is knowing how many incidents occurred and how long each lasted. That is exactly what the instrumentation in Fix 1 provides, and it is why the instrumentation is not optional.

## Decision checklist

Before changing anything, confirm each of the following. Any "no" is a prerequisite, not a nice-to-have.

- [ ] Node groups carry a cost-attribution tag, and the cost explorer is grouped by it.
- [ ] The autoscaling group's desired and in-service capacity are being recorded as time series.
- [ ] Secret-store errors are recorded as a time series with the error code as a label.
- [ ] Secret tokens are renewed on an interval shorter than their lease.
- [ ] No credential is readable from a repository, a ConfigMap, or a container image layer.
- [ ] The control-plane endpoint is reachable only from expected networks.
- [ ] No service account in production holds `cluster-admin`.

## Escalation path

If the cost trend continues upward and secret errors continue after the three fixes, the remaining explanations are usually one of:

- A workload outside the node group you changed is consuming the capacity.
- A second credential path is failing independently of the one you fixed.
- The autoscaler is reacting to a metric other than the one you tuned.

Collect the following before escalating: the current node-group configuration, the secret-store audit log for the affected window, and the cost breakdown grouped by tag. Escalating with these three artifacts lets the receiving team reproduce the correlation rather than re-derive it.

## FAQ

**How do I find which resources drive the bill?**
Enable cost allocation tags in the provider's billing settings, tag every node group, and group the cost report by that tag. Without the tag, the report cannot be joined to configuration, and the investigation stalls.

**Why would revoking a secret role cause node scaling?**
Because the autoscaler reacts to CPU, not to success rate. A failing service that retries in a loop consumes CPU, and the autoscaler cannot distinguish that from useful load.

**What is a safe way to hold API credentials for a service?**
Read them at runtime from a secret manager using a short-lived, narrowly scoped identity, and renew on an interval shorter than the lease. Do not bake them into images, ConfigMaps, or environment variables that are checked into source control.

**When should the control-plane endpoint be private?**
When your compliance framework requires network isolation, or when the API server audit log shows requests from unexpected source addresses. Note that private endpoints require your CI and operator access paths to be inside the network or connected to it.

**Does raising the scale-in cooldown cost money?**
Yes. A longer cooldown means nodes that are no longer needed stay billed for longer. The trade is deliberate: it prevents flapping when spikes are failure-driven. Measure the incident frequency before and after to confirm the trade is favorable.

## Do this next

Open your Terraform configuration, find the node group that backs your busiest workload, and add the `CostCenter` tag if it is missing. Then run `terraform plan` and confirm the tag appears in the diff. Tagging takes under 30 minutes and is the prerequisite for every measurement described above — without it, you cannot tell which change helped.
