# Stop losing devs to undifferentiated work

This is a topic where the standard advice is technically correct but practically misleading. Here's the fuller picture, based on what I've seen work at scale.

## The situation (what we were trying to solve)

In 2022, the backend team at our e-commerce startup grew from 5 to 15 engineers in six months. We shipped features fast—too fast. By the end of the year, incident reports mentioned engineers spending 30% of their week on the same four tasks: wiring up new services to Kafka, setting up service discovery, configuring CI for new repos, and chasing down why a deployment broke because of a missing environment variable. Each engineer averaged 12 hours a month debugging infrastructure drifts like mismatched Docker base images or inconsistent secrets. We measured this by rolling out a simple internal survey asking: "What did you do yesterday that wasn’t writing business logic?" The answers were repetitive: "Spent 3 hours debugging a Kafka consumer that didn’t start because the group.id was missing." "Wasted a day because staging used a different Node version than prod." These weren’t knowledge gaps; they were undifferentiated work.

We tried to fix it by documenting every step in a Notion page titled "Onboarding Checklist v3." It ballooned to 150 items. Even with the checklist, new engineers still broke staging 3 times in their first two weeks because the checklist assumed local environment parity with production, which it never was. The checklist also didn’t account for the fact that our Kafka cluster version changed every quarter, so group.id conventions became outdated within weeks.

We needed a way to stop repeating the same setup over and over. We needed a paved road—an internal developer platform (IDP) that encoded our conventions once and let engineers focus on business logic instead of infrastructure details. The goal wasn’t to abstract away the cloud; it was to stop reinventing the same wheel while the road beneath us cracked.


Our platform had to meet three criteria:
1. **Portability**: Work the same locally, in staging, and in production.
2. **Repeatability**: Every engineer could run `make dev` and get the same environment.
3. **Observability by default**: Every service deployed with metrics, tracing, and logs preconfigured.


We measured the problem in three numbers: 12 hours per engineer per month on undifferentiated work, 3 staging breaks per new engineer in their first two weeks, and 30% of code reviews focused on infrastructure rather than business logic. Any solution had to reduce at least two of those numbers by 50% within six months.


*Summary: We faced repetitive, error-prone setup tasks that distracted engineers from writing features. The existing checklist approach failed because it couldn’t keep pace with changing infrastructure and didn’t enforce parity across environments.*


## What we tried first and why it didn’t work

First, we tried a monorepo with Bazel. The idea was to centralize build rules and enforce consistency at the build step. We spent three months migrating 12 services into one repo. The build time for a single service went from 45 seconds to 11 minutes because Bazel had to rebuild the entire world for every change. Even with remote caching, the median build time was 4 minutes 30 seconds. Engineers stopped running tests locally and relied on CI, which made feedback loops slower.

We also tried enforcing service discovery with Consul and a custom sidecar container that injected environment variables at runtime. It worked—until it didn’t. In staging, 15% of deployments failed because the sidecar couldn’t resolve the Consul agent due to a DNS misconfiguration. The failure only surfaced at runtime, not during the build. Rolling back meant redeploying the entire sidecar, which took 7 minutes per service. We measured this by instrumenting our deployment pipeline: the average rollback time was 6 minutes 42 seconds, and 23% of rollbacks introduced a new bug because the sidecar version didn’t match the service version.

Next, we tried GitHub Actions workflows templated with reusable workflows. We saved 30 lines of YAML per service, which was good. But the templates still required manual updates when we upgraded Node versions or changed the Docker base image. Engineers had to remember to update the workflow when they updated the Dockerfile, so the templates became outdated within weeks. We audited the workflows after three months and found 42% of services were using Node 16 even though Node 18 was the recommended version. The templates didn’t enforce version consistency; they just made it easier to copy-paste the wrong version.

Finally, we tried a custom CLI tool that wrapped `docker-compose` and `kubectl`. It worked locally but broke in CI because the CLI assumed the user had Docker installed with specific volume mounts. When CI environments changed (e.g., switching from GitHub Actions to Buildkite), the CLI failed silently, and engineers spent hours debugging why the CI logs showed "permission denied" on a volume that worked locally. We measured this by instrumenting the CLI with debug logs: 68% of failures were due to environment assumptions that didn’t hold in CI.


*Summary: We tried monorepos, service discovery sidecars, templated CI workflows, and custom CLIs. Each approach solved part of the problem but introduced new failure modes: slow builds, silent CI failures, outdated templates, and runtime errors that escaped detection.*


## The approach that worked

We stopped trying to abstract away the cloud and started building a paved road: a set of conventions, tooling, and guardrails that let engineers focus on features without reinventing the same setup. The core idea was to encode our conventions into a platform that enforced parity across environments and provided guardrails to prevent drift.

We built the platform around three layers:

1. **Environment parity with Nix**: We replaced Dockerfiles with Nix shells and flakes. A single `flake.nix` defined the runtime environment for each service, including Node version, Python packages, and system dependencies. Engineers ran `nix develop` locally and `nix build` in CI; the environment was guaranteed to be the same. We measured the first-time setup time for a new engineer: it dropped from 8 hours to 45 minutes.

2. **Golden paths with Tilt**: We used Tilt to define golden paths for local development and CI. Tilt watched the service source code and automatically rebuilt and redeployed the service in a local Kubernetes cluster (kind) or in staging. The Tiltfile was the source of truth for how a service should build, test, and deploy. Engineers ran `tilt up` and got a live-reload environment that mirrored production. After rolling this out, the number of staging breaks per new engineer dropped from 3 to 0.8 in their first two weeks.

3. **Guardrails with OPA policies**: We enforced guardrails with Open Policy Agent (OPA) policies in our CI pipeline. Policies checked for things like "no hardcoded secrets in environment variables," "Kafka group.id matches the service name," and "Docker base image is pinned to a digest." Policies ran in CI and failed the build if violated. We measured policy violations: in the first month, 12 services were blocked for hardcoded secrets, and 8 were blocked for mismatched group.id. After two months, violations dropped to 0.


We also standardized on a single observability stack: Prometheus for metrics, Grafana for dashboards, and OpenTelemetry for tracing. Every service was required to expose metrics on `/metrics` and include a trace context header. We built a Grafana dashboard template that every service could import, so dashboards were consistent and engineers didn’t have to write their own queries.


The platform wasn’t a black box. Engineers could override any convention, but doing so required a pull request to the platform repository and approval from the platform team. This made conventions explicit and discouraged one-off exceptions.


*Summary: We built a paved road with Nix for parity, Tilt for golden paths, and OPA for guardrails. The platform enforced conventions without hiding the underlying infrastructure, and engineers could override conventions with explicit approval.*


## Implementation details

### Nix for environment parity

We started by replacing Dockerfiles with Nix flakes. Each service has a `flake.nix` that defines the development environment and the runtime environment. Here’s an example for a Node.js service:

```nix
{
  description = "Node.js service flake";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        packages.default = pkgs.stdenv.mkDerivation {
          name = "node-service";
          src = ./.;
          buildInputs = with pkgs; [ nodejs_20 nodePackages.pnpm ];
          shellHook = ''
            export NODE_ENV=development
            pnpm install
          '';
        };

        devShells.default = pkgs.mkShell {
          packages = with pkgs; [ nodejs_20 nodePackages.pnpm direnv ];
          shellHook = ''
            export NODE_ENV=development
            pnpm install
          '';
        };
      }
    );
}
```

The flake defines a development shell (`devShells.default`) that includes Node.js 20 and pnpm. Engineers run `nix develop` to enter the shell, which sets up the environment automatically. The `packages.default` defines how the service builds in CI.

We measured the time to set up a new service from scratch: it dropped from 8 hours to 45 minutes. The key was that the flake was the single source of truth for the environment, not a README or a Dockerfile.


### Tilt for golden paths

Tilt automates the golden path for local development and CI. We defined a `Tiltfile` for each service that specified how to build, test, and deploy the service:

```python
# Tiltfile
allow_k8s_contexts('kind-cluster')

k8s_yaml('k8s/deployment.yaml')

docker_build(
    "my-service",
    "docker.io/myorg/my-service",
    dockerfile="Dockerfile",
    build_args={"NODE_ENV": "development"},
)

def update_manifests():
    # Generate Kubernetes manifests from the flake
    local("nix build .#k8s-manifests -o k8s-manifests")
    local("cp k8s-manifests/k8s/* k8s/")

update_manifests()

def dependencies():
    yield update_manifests()
    yield docker_build("my-service", ...)
    yield k8s_resource("my-service", port_forwards=8080)
```

The `Tiltfile` does three things:
1. It builds the Docker image using the flake-defined environment.
2. It generates Kubernetes manifests from the flake (we added a Nix derivation for `k8s-manifests`).
3. It sets up port forwarding so engineers can access the service at `localhost:8080`.

When an engineer runs `tilt up`, Tilt watches the source code and automatically rebuilds and redeploys the service. The live-reload loop reduces the feedback cycle from minutes to seconds.

We measured the time from code change to seeing the change in staging: it dropped from 4 minutes to 15 seconds with Tilt in local development.


### OPA for guardrails

We enforced guardrails with OPA policies in our CI pipeline. Policies are written in Rego and run in GitHub Actions using the `opa` CLI. Here’s an example policy that checks for hardcoded secrets in environment variables:

```rego
package kubernetes.validating

hardcoded_secrets[msg] {
  input.kind == "Deployment"
  container := input.spec.template.spec.containers[_]
  env := container.env[_]
  contains(env.value, "AK_"); msg := sprintf("Hardcoded secret in %s: %s", [container.name, env.name])
}
```

The policy runs in CI and fails the build if it finds a hardcoded secret. We also added policies for:
- Kafka group.id matches the service name.
- Docker base image is pinned to a digest.
- Service has a health check endpoint.
- No root user in the container.

We measured policy violations: in the first month, 12 services were blocked for hardcoded secrets, and 8 were blocked for mismatched group.id. After two months, violations dropped to 0, and the number of code reviews focused on infrastructure dropped by 60%.


### Observability stack

We standardized on Prometheus, Grafana, and OpenTelemetry. Every service is required to:
- Expose metrics on `/metrics` in Prometheus format.
- Include an OpenTelemetry trace context header.
- Have a Grafana dashboard template that imports from a shared library.

We built a shared Grafana dashboard template:

```json
{
  "dashboard": {
    "title": "Service Dashboard",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[1m])",
            "legendFormat": "{{service}}"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{status_code=~"5.."}[1m])",
            "legendFormat": "{{service}}"
          }
        ]
      }
    ]
  }
}
```

The dashboard template is imported by every service, so dashboards are consistent and engineers don’t have to write their own queries.

We measured the time to set up observability for a new service: it dropped from 3 hours to 15 minutes.


*Summary: The platform combined Nix for parity, Tilt for golden paths, and OPA for guardrails. Observability was standardized with Prometheus, Grafana, and OpenTelemetry. Engineers spent less time setting up environments and more time writing features.*


## Results — the numbers before and after

| Metric | Before | After | Reduction |
|--------|--------|-------|-----------|
| Time to set up new service | 8 hours | 45 minutes | 91% |
| Staging breaks per new engineer (first two weeks) | 3 | 0.8 | 73% |
| Time from code change to seeing change in staging | 4 minutes | 15 seconds | 94% |
| Undifferentiated work per engineer per month | 12 hours | 4 hours | 67% |
| Code reviews focused on infrastructure | 30% | 12% | 60% |
| Policy violations in CI | N/A | 0 after 2 months | 100% |
| Average rollback time | 6 minutes 42 seconds | 2 minutes 15 seconds | 65% |


The most surprising result was the drop in staging breaks. Before the platform, staging breaks were a rite of passage for new engineers. After Tilt and Nix, new engineers could deploy to staging without breaking anything, and the feedback loop was fast enough that they could iterate quickly. We measured this by instrumenting the staging cluster: the number of failed deployments dropped from 15% to 2% within three months.


The platform also reduced the cognitive load on engineers. We ran a survey asking engineers to rate their frustration with infrastructure tasks on a scale of 1 to 10. Before the platform, the average frustration score was 7.2. After the platform, it dropped to 3.1. The biggest improvement was in local development: before, engineers spent hours debugging environment mismatches; after, they ran `nix develop` and `tilt up`, and the environment just worked.


We also measured cost. Before the platform, engineers spent 12 hours a month on undifferentiated work. After, it dropped to 4 hours. At an average engineer cost of $80/hour, that’s a saving of $960 per engineer per month. For 15 engineers, that’s $14,400 per month or $172,800 per year.


The platform wasn’t free to build. We spent 6 engineer-months building the initial version, mostly on Nix flakes and Tiltfiles. We also spent 2 engineer-months standardizing the observability stack. The total cost was roughly $140,000 in engineering time. But the platform paid for itself in less than a year in saved engineering time alone, not counting the reduction in staging breaks and faster feedback loops.


*Summary: The platform reduced setup time by 91%, staging breaks by 73%, and undifferentiated work by 67%. It cost $140,000 to build but paid for itself in less than a year in saved engineering time.*


## What we'd do differently

If we started over, we’d change three things:

1. **Start with a single service first.** We tried to roll out the platform to all 15 services at once. It created too much churn. Next time, we’d pick one service, build the platform around it, and iterate until it felt effortless. Then we’d expand to other services. This would have reduced the initial build time from 6 engineer-months to 2.

2. **Automate the flake generation.** Writing `flake.nix` by hand for each service was tedious. We ended up writing a script that generates a `flake.nix` from a template, but we should have done it from day one. The script saved 30 minutes per service, which adds up when you have 15 services. Automating it would have saved us 7.5 engineer-hours.

3. **Measure more aggressively.** We measured setup time and staging breaks, but we didn’t measure the time engineers spent debugging environment mismatches. We only realized how bad it was when we ran the frustration survey. Next time, we’d instrument the platform from day one to capture every minute engineers spend on infrastructure tasks.


We also got the observability stack wrong at first. We started with a custom Grafana dashboard for each service, which created 15 dashboards to maintain. We ended up standardizing on a shared dashboard template, which reduced maintenance time from 3 hours per service to 15 minutes. Next time, we’d standardize observability before rolling out the platform.


*Summary: We’d start with one service, automate flake generation, and measure more aggressively. We’d also standardize observability before rolling out the platform.*


## The broader lesson

The best internal developer platforms don’t abstract away the cloud; they make the cloud safer and more predictable. They encode your conventions once and enforce them everywhere. They turn undifferentiated work into a one-time cost, not a recurring tax.

The trap most teams fall into is trying to build a platform that does everything for everyone. That’s a recipe for a bloated, fragile system that no one wants to use. Instead, build a paved road: a set of conventions, tooling, and guardrails that let engineers move fast without breaking things.

The paved road principle is simple: if a task is repetitive and error-prone, automate it once and let everyone benefit. If a task is rare and specific, leave it to the engineer. Don’t try to optimize for every edge case; optimize for the 80% case, and let engineers override the conventions when they need to.


*Summary: A good internal developer platform encodes conventions, enforces guardrails, and turns undifferentiated work into a one-time cost. The paved road principle is to automate repetitive, error-prone tasks and leave rare, specific tasks to engineers.*


## How to apply this to your situation

### Step 1: Identify the repetitive, error-prone tasks

List every task your engineers do that isn’t writing business logic. Use your incident reports, code reviews, and onboarding surveys. Look for patterns like:
- "Spent 3 hours debugging a Kafka consumer that didn’t start because the group.id was missing."
- "Wasted a day because staging used a different Node version than prod."
- "Took 2 hours to set up a new service because the README was outdated."

For each task, ask: Is this a one-time cost or a recurring tax? If it’s a recurring tax, it’s a candidate for the platform.


### Step 2: Pick one service and build the platform around it

Don’t try to roll out the platform to all services at once. Pick one service, build the platform around it, and iterate until it feels effortless. Use the service to validate your conventions and guardrails. Once it works, expand to other services.


### Step 3: Standardize observability first

Before you build anything else, standardize your observability stack. Every service should expose metrics, tracing, and logs in the same way. Use a shared dashboard template and a shared library for metrics. This will save you time later when you try to debug issues across services.


### Step 4: Enforce guardrails with OPA

Use OPA to enforce guardrails in CI. Start with simple policies like "no hardcoded secrets" and "Docker base image is pinned to a digest." Add more policies as you go. The key is to fail fast and give clear feedback in CI.


### Step 5: Measure everything

Instrument the platform to capture every minute engineers spend on infrastructure tasks. Measure setup time, staging breaks, rollback time, and frustration scores. Use the data to iterate and improve.


### Step 6: Document exceptions, not rules

When an engineer needs to override a convention, document the exception in the platform repository. This makes conventions explicit and discourages one-off exceptions. Over time, you’ll have a clear record of why exceptions were made and when they can be revisited.


*Summary: Identify repetitive tasks, start with one service, standardize observability, enforce guardrails with OPA, measure everything, and document exceptions. This is the paved road playbook.*


## Resources that helped

- **Nix**: [Nix Flakes documentation](https://nixos.wiki/wiki/Flakes) – The official documentation is dense, but the Nix community on Discord is helpful.
- **Tilt**: [Tilt documentation](https://docs.tilt.dev/) – Start with the "Getting Started" guide and the "Local Development" examples.
- **OPA**: [OPA documentation](https://www.openpolicyagent.org/docs/latest/) – The "Policy Language" and "How OPA Works" sections are essential.
- **Prometheus + Grafana**: [Prometheus Operator](https://prometheus-operator.dev/) – Use the operator to manage Prometheus and Grafana in Kubernetes.
- **OpenTelemetry**: [OpenTelemetry Collector](https://opentelemetry.io/docs/collector/) – The collector is the easiest way to get traces and metrics out of your services.
- **Book**: "Team Topologies" by Matthew Skelton and Manuel Pais – Explains how to structure platform teams and internal developer platforms.
- **Talk**: "The Golden Path" by Kelsey Hightower – A keynote from KubeCon 2019 that frames internal developer platforms as paved roads.
- **Tool**: [Nix flake templates](https://github.com/srid/nixos-flake) – A collection of Nix flake templates for common use cases.


*Summary: These resources helped us build our platform. They’re a mix of documentation, talks, and books that cover Nix, Tilt, OPA, observability, and platform engineering principles.*


## Frequently Asked Questions

**How do we get buy-in from leadership for a platform team?**

Frame the platform as a force multiplier, not an overhead. Measure the time engineers spend on undifferentiated work and the cost of staging breaks. Present the data: "Engineers spend 12 hours a month on infrastructure tasks, costing us $14,400 per month. The platform will cut that in half and pay for itself in less than a year." Leadership cares about ROI and risk reduction; show them the numbers.


**Isn’t a platform just another layer of abstraction that will slow us down?**

Not if it’s a paved road, not a black box. The platform should enforce your conventions, not hide them. Engineers should still understand how their service builds, deploys, and runs. The platform should reduce cognitive load, not increase it. Measure the time from code change to seeing the change in staging: if it’s faster, the platform is helping.


**What if our team doesn’t use Kubernetes?**

The principles still apply. Replace Kubernetes with whatever you use: EC2, ECS, Cloud Run, or even bare metal. The key is parity across environments and golden paths for local development and CI. Nix works for any environment, and Tilt can wrap any deployment tool. The platform should be environment-agnostic.


**How do we avoid creating a platform that no one wants to use?**

Involve engineers from day one. Build the platform in the open, with pull requests and reviews from the teams that will use it. Make it easy to override conventions, but document the exceptions. Measure frustration scores and iterate based on feedback. The platform should feel like a tool, not a cage.


*Summary: Leadership buy-in comes from ROI, the platform shouldn’t slow you down if it’s a paved road, it works without Kubernetes, and it must be built in the open with engineer feedback.*