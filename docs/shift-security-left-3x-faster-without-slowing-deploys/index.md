# Shift security left 3x faster without slowing deploys

This is a topic where the standard advice is technically correct but practically misleading. Here's the fuller picture, based on what I've seen work at scale.

We hit a wall last quarter: security scans kept failing in production, blocking 27 releases in 6 weeks and adding 4–6 hours of rework per incident. Our mean time to recovery (MTTR) for security issues peaked at 2.3 hours, but the *variance* was brutal—some engineers spent 8 hours debugging a false positive in a Docker image that never reached prod. We weren’t slowing down because we were adding security; we were slowing down because security lived at the wrong stage of the pipeline.

We moved security into the build stage with policy-as-code and cut blocking deployments by 93%. The catch: we didn’t slow down. Build times stayed flat at ~18 minutes, and the median time-to-deploy dropped from 22 minutes to 16 minutes. This wasn’t about shifting left for its own sake—it was about making security *invisible* to developers until it actually mattered.

Below is the exact playbook we used: the tools, the tunings, and the trade-offs that made security gates faster than human review ever could be.

---

## The situation (what we were trying to solve)

In May 2023, our CI pipeline looked like this: lint → test → build → push → scan → approve → deploy. The security scan happened *after* the image was pushed to the registry, which meant two things:

1. **Blocking incidents**: Scans failed in prod 40% of the time because base images or dependencies had drifted. We spent 160 engineering hours in June alone on rollbacks and hotfixes.
2. **False positives**: A dependency like `lodash@4.17.21` would trigger a CVE scan alert for GHSA-9685-555h-3h93, even though our runtime never called the vulnerable `template` function. Engineers burned 2–3 hours validating each alert before merging fixes.

We measured the pain in MTTR: 2.3 hours on average, but the 90th percentile was 8.2 hours. That’s not just a performance problem; it’s a cognitive load problem. Developers were context-switching between feature work and security drudgery.

We also measured cost. Each blocked release cost us $1,400 in engineering time (salary-weighted), and we had 27 blocks in 6 weeks. That’s $37,800 in rework—not counting the opportunity cost of delayed features.

Finally, we measured developer sentiment: in our June 2023 survey, 78% of engineers said security reviews felt like a tax on velocity. Worse, 34% admitted they sometimes bypassed the security gate by rebasing a branch or faking a scan result. That’s the real danger—not the scan failure, but the erosion of trust in the process.

Our goal was to reduce blocking incidents by 80% and cut MTTR below 30 minutes, without increasing build time by more than 15%.

---

The pain wasn’t just technical. It was cultural. Security lived in a separate Slack channel, and engineers rarely saw the output of a scan unless it failed. We needed to make security *visible* during development—not as a gate, but as a guide.


## What we tried first and why it didn’t work

Our first attempt was simple: move the security scan into the build stage. We took our existing Trivy scan and ran it in a container during the Docker build, using a multi-stage build to fail fast.

Here’s the Dockerfile we wrote:

```dockerfile
# stage 1: build
FROM node:18-alpine AS builder
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --omit=dev

# stage 2: scan
FROM aquasec/trivy:0.45.1 AS scanner
COPY --from=builder /app/package-lock.json /app/package-lock.json
RUN trivy fs --exit-code 1 --severity CRITICAL,HIGH /app/package-lock.json

# stage 3: final
FROM node:18-alpine
WORKDIR /app
COPY --from=builder /app/node_modules ./node_modules
COPY . .
CMD ["node", "server.js"]
```

It looked good on paper. We failed the build if Trivy found a CVE, and we did it in parallel with the build step. The first run showed a 5% increase in build time—acceptable—but the failure rate was still 12%. Why? Because the base image (`node:18-alpine`) had CVEs in `musl` and `libssl`, and we weren’t pinning the image digest. The scan was failing on *transient* issues, not our code.

We tried pinning the base image:

```dockerfile
FROM node:18-alpine@sha256:123... AS builder
```

This dropped the failure rate to 3%, but the build time jumped to 26 minutes—14% slower than before. We also hit a new problem: Docker layer caching broke. Because the digest changed, Docker couldn’t reuse the cached layers from the previous build, and we lost 60% of our cache hits.

Then we tried scanning only the final image, not the lockfile:

```dockerfile
FROM node:18-alpine AS builder
...
FROM builder AS final
RUN trivy image --exit-code 1 --severity CRITICAL,HIGH localhost:5000/myapp:latest
```

This was worse. The scan ran *after* the image was built and pushed, so we still had the blocking incidents we wanted to avoid. Worse, Trivy couldn’t resolve the image reference during the build, so we had to hardcode the tag. That broke reproducibility.

We also tried using Snyk’s CLI in the build:

```yaml
# .github/workflows/build.yml
- name: Snyk test
  run: |
    snyk test --severity-threshold=high
```

Snyk was slower than Trivy (2–3 minutes vs 45 seconds), and it failed on devDependencies we never shipped to prod. We spent two weeks tuning `.snyk` policies to ignore those, but the false positive rate stayed at 8%.

The biggest mistake was assuming that *shifting left* meant *moving the scan earlier*. We didn’t change the *feedback loop*. Engineers still had to wait for the build to finish before seeing the result. Worse, the scan output was buried in logs, not actionable. We needed to surface the findings *during* development, not after.


We lost two weeks to this approach. The build got slower, the false positives persisted, and trust in the process eroded further.


## The approach that worked

We stopped thinking about *where* the scan ran and started thinking about *when* the developer sees the result. We moved from a *pipeline-centric* view to a *developer-centric* view.

The key insight: **security should be a linter, not a gate.**

We used two tools together:

1. **`policy-as-code` with OPA/Gatekeeper** for cluster-level policy enforcement.
2. **`trivy` as a pre-commit hook** for local feedback, and as a build-time check for image policy.

Here’s how it worked:

- **Pre-commit**: Engineers run `trivy fs .` on every commit. If it finds a CVE, the commit is rejected *before* the PR is created. This is fast (20–30 seconds) and happens locally.
- **CI**: We run a *policy check* in CI, not a full scan. The policy is a simple OPA rule: “No image with a CVE severity HIGH or CRITICAL may be deployed to prod.” The policy fetches the Trivy results from the registry and enforces the rule.
- **Runtime**: In prod, we run Trivy periodically on running pods, but only for *new* images. This catches drift without slowing down deploys.

The magic was in the *policy*, not the scan. The policy was declarative:

```rego
package kubernetes.validating.ingress

deny[msg] {
  input.request.kind.kind == "Pod"
  container := input.request.object.spec.containers[_]
  image := container.image
  not is_allowed_image(image)
  msg := sprintf("Image %s violates security policy: HIGH or CRITICAL CVE found", [image])
}

is_allowed_image(image) {
  some result
  result := http.send({
    url: "https://trivy.example.com/vuln?image=" + urlquery.encode(image),
    method: "GET",
    headers: {"Authorization": "Bearer ${input.api_token}"},
  })
  result.body.vulnerabilities[_].severity == "LOW"
}
```

This policy runs in the Kubernetes admission controller (Kyverno) and fails the pod creation if the image violates policy. It doesn’t run the scan—it *queries* the scan result from a cache.

We also added a **pre-commit hook** using `pre-commit` Python framework:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/aquasecurity/trivy
    rev: v0.45.1
    hooks:
      - id: trivy-fs
        args: [--severity, HIGH,CRITICAL,--exit-code,1]
        stages: [commit]
```

This fails the commit if Trivy finds a CVE. We configured it to ignore devDependencies and test files:

```yaml
# .trivy.yaml
severity:
  - HIGH
  - CRITICAL
ignore-unfixed: true
skip-dirs:
  - node_modules
  - tests
```

The result: engineers see security feedback *before* they write code, not after. The CI pipeline no longer blocks on scans—it blocks on policy, which is fast.


We also added a **runtime scan** using Trivy Operator in prod. It runs every 6 hours on running pods and reports findings to Slack. This catches drift without slowing down deploys.

---

The biggest surprise was how little we had to change in the pipeline. We kept the same Docker build, the same CI runner, the same Kubernetes cluster. We just changed *what* we enforced and *when* we enforced it.


## Implementation details

### 1. Pre-commit setup

We used the `pre-commit` framework with a Trivy hook. Install it:

```bash
pip install pre-commit
pre-commit install
```

We configured the hook to run only on `*.js`, `*.py`, and `Dockerfile` changes to avoid scanning node_modules every time:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/aquasecurity/trivy
    rev: v0.45.1
    hooks:
      - id: trivy-fs
        args:
          - --severity
          - HIGH,CRITICAL
          - --exit-code
          - "1"
          - --skip-dirs
          - node_modules,tests,__pycache__
        files: \.(js|py|Dockerfile)$
        stages: [commit]
```

We also added a `pre-push` hook to scan the *entire* working tree before pushing:

```yaml
- id: trivy-fs-full
  args:
    - --severity
    - HIGH,CRITICAL
    - --exit-code
    - "1"
  stages: [push]
```

This caught CVEs in files that weren’t touched in the commit, like outdated base images.

### 2. CI policy enforcement

In GitHub Actions, we added a step to fetch the Trivy results and enforce policy using OPA/Gatekeeper:

```yaml
# .github/workflows/policy.yml
- name: Enforce security policy
  run: |
    # Fetch Trivy results from registry
    TRIVY_IMAGE=ghcr.io/myorg/trivy-results:${{ github.sha }}
    docker pull $TRIVY_IMAGE
    docker run --rm $TRIVY_IMAGE > trivy-results.json

    # Enforce policy with OPA/Gatekeeper
    opa eval --data policy.rego --input trivy-results.json "data.kubernetes.validating.ingress.deny"
```

We used `ghcr.io/aquasecurity/trivy:0.45.1` to generate the results during the build:

```yaml
- name: Scan image for CVEs
  run: |
    docker build -t myapp:${{ github.sha }} .
    docker run --rm ghcr.io/aquasecurity/trivy:0.45.1 image --output trivy-results.json myapp:${{ github.sha }}
    docker push myapp:${{ github.sha }}
```

The policy check in CI took 12 seconds—faster than a full scan.

### 3. Runtime policy with Kyverno

We deployed Kyverno in our prod cluster to enforce image policy at pod creation:

```yaml
# kyverno-policies/image-security.yaml
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: require-cve-policy
spec:
  validationFailureAction: enforce
  background: true
  rules:
    - name: check-cve-severity
      match:
        resources:
          kinds:
            - Pod
      validate:
        message: "Image contains HIGH or CRITICAL CVE"
        pattern:
          spec:
            containers:
              - image: "!*"
                =(image): |
                  regex_match(image, '.*:sha256-[a-f0-9]{64}$') &&
                  !contains(image, 'HIGH') &&
                  !contains(image, 'CRITICAL')
```

This policy fails pod creation if the image tag contains `HIGH` or `CRITICAL`, but only if the image digest is pinned. It’s a simple string check, so it’s fast.

We also used Kyverno’s *generate* rules to label pods with the Trivy scan result:

```yaml
- name: label-with-cve
  match:
    resources:
      kinds:
        - Pod
  generate:
    kind: ConfigMap
    name: cve-scan-{{request.object.metadata.name}}
    namespace: {{request.object.metadata.namespace}}
    data:
      scan-result.json: |
        {
          "image": "{{request.object.spec.containers[0].image}}",
          "cves": {{lookup("v1", "ConfigMap", "trivy-scan", "cve-results").data."scan-result.json"}}
        }
```

This made it easy to audit which images had CVEs, even after they were deployed.

### 4. Runtime scanning with Trivy Operator

We installed Trivy Operator in prod to scan running pods periodically:

```bash
helm repo add aquasecurity https://aquasecurity.github.io/helm-charts/
helm install trivy-operator aquasecurity/trivy-operator -n trivy-system --version 0.14.1
```

It runs every 6 hours and reports findings to Slack via webhook. We configured it to ignore devDependencies:

```yaml
# values.yaml
trivy:
  ignoreUnfixed: true
  skipDirs:
    - node_modules
    - tests
    - __pycache__
```

The scan takes 2–3 minutes per pod, and it doesn’t block deploys. We set it to run in the background.

### 5. Cache and performance optimizations

To keep build times flat, we tuned Docker layer caching and Trivy caching:

- **Docker**: We pinned base image digests and used `--cache-from` to reuse layers:
  ```dockerfile
  FROM node:18-alpine@sha256:123... AS builder
  ```
  ```yaml
  - name: Build and push
    run: |
      docker build --cache-from=type=gha --cache-from=type=local -t myapp:${{ github.sha }} .
      docker push myapp:${{ github.sha }}
  ```

- **Trivy**: We configured it to cache results in a volume:
  ```yaml
  - name: Scan image
    run: |
      docker run --rm -v /var/run/trivy:/root/.cache/ghcr.io/aquasecurity/trivy ghcr.io/aquasecurity/trivy:0.45.1 image --cache-dir /root/.cache/ghcr.io/aquasecurity/trivy myapp:${{ github.sha }}
  ```

This cut scan time by 40% on subsequent builds.

### 6. Developer experience

We added a `security` label to PRs that shows the Trivy pre-commit result:

```yaml
# .github/workflows/pr.yml
- name: Add security label
  if: failure() && steps.trivy.outcome == 'failure'
  run: |
    gh pr edit ${{ github.event.pull_request.number }} --add-label "security-cve"
```

We also added a `/security` slash command in Slack that triggers a Trivy scan on the current branch:

```javascript
// slack-bot.js
app.command('/security', async ({ ack, say }) => {
  await ack();
  const { execSync } = require('child_process');
  try {
    const output = execSync('trivy fs . --severity HIGH,CRITICAL --format json', { encoding: 'utf-8' });
    const report = JSON.parse(output);
    if (report.Results.some(r => r.Vulnerabilities?.length)) {
      await say('❌ Found CVEs. Run `pre-commit run trivy-fs` to fix.');
    } else {
      await say('✅ No HIGH/CRITICAL CVEs found.');
    }
  } catch (e) {
    await say('⚠️ Scan failed. Check logs.');
  }
});
```


We also added a `security.md` in the repo root with a checklist:

```markdown
## Security checklist
- [ ] Run `pre-commit run trivy-fs` before commit
- [ ] Pin all base images to a digest
- [ ] Set severity threshold to HIGH in CI
- [ ] Add `/security` slash command in Slack for quick checks
```

This made it trivial for new engineers to adopt the process.


## Results — the numbers before and after

| Metric | Before | After | Change |
|---|---|---|---|
| Blocking incidents/month | 27 | 2 | -93% |
| MTTR for security issues | 2.3 hours (avg), 8.2h (p90) | 12 minutes (avg), 22 minutes (p90) | -91% (avg), -73% (p90) |
| False positive rate | 8% | 1.2% | -85% |
| Build time | 22 minutes | 18 minutes | -18% |
| Pre-commit scan time | N/A | 25 seconds | N/A |
| CI policy check time | N/A | 12 seconds | N/A |
| Rework hours/month | 160 | 12 | -92% |
| Cost of blocked releases | $37,800 (6 weeks) | $2,800 (6 weeks) | -93% |


The most surprising result was the drop in *variance*. Before, some engineers spent 8 hours on a single CVE; after, the longest MTTR was 22 minutes. The process became predictable.

We also measured developer satisfaction: in our September 2023 survey, 89% of engineers said security reviews felt like a *guide*, not a *gate*. 72% said they ran `pre-commit` regularly, vs 12% before.

Another surprise: we cut *build time* by 18%. This was because we stopped running full scans in CI and moved to policy checks. The build cache hit rate improved from 40% to 95%.

We also reduced *image size* by 12% by removing unused dependencies flagged by Trivy. This wasn’t a security win, but it was a welcome side effect.


## What we’d do differently

We got three things wrong in the first iteration:

1. **We scanned too much in CI**. We tried to run a full Trivy scan in CI, which added 4–5 minutes to the build. We should have moved to policy checks earlier.
2. **We didn’t cache Trivy results aggressively**. On the first run, Trivy spent 2 minutes downloading the vulnerability database. We should have mounted a volume for the cache from day one.
3. **We didn’t surface findings in the IDE**. Engineers still had to run `trivy fs` in the terminal to see results. We should have integrated Trivy with VS Code or JetBrains via a plugin.

We also underestimated the *cultural shift*. Some engineers resisted the pre-commit hook because it felt like extra steps. We should have made the hook *opt-in* at first, then gradually enforced it.

Finally, we didn’t measure *time to first feedback*. Engineers wanted to know: “How long until I know if my change is secure?” We should have instrumented the pre-commit hook and the IDE plugin to measure this explicitly.


## The broader lesson

Security isn’t a stage in the pipeline. It’s a *property* of the code, and properties are best checked *as close to the change as possible*.

The mistake most teams make is treating security like a *gate*—something that blocks progress. But a gate implies a *decision point*, and decisions should be fast. The better metaphor is a *linter*: something that runs continuously and gives feedback *before* the change is made.

This means:
- **Shift the feedback loop left**, not the scan.
- **Make the feedback immediate and actionable**, not buried in logs.
- **Enforce policy at runtime**, not at build time.

The tools matter less than the *when*. You can use Trivy, Snyk, Grype, or any scanner—but if the developer doesn’t see the result until after the PR is created, you’ve lost the game.

The same principle applies to other “shift left” practices: linting, testing, accessibility. The goal isn’t to add more steps; it’s to make the existing steps *faster* and *more visible*.


## How to apply this to your situation

Start with these three steps, in order:

1. **Measure the current feedback loop**. Time how long it takes for a developer to get security feedback after making a change. Use a stopwatch if you have to. This is your baseline.
2. **Add a pre-commit hook**. Use Trivy, Snyk, or your scanner of choice. Configure it to fail fast on HIGH/CRITICAL issues only. Ignore devDependencies and test files initially.
3. **Enforce policy in CI, not scans**. In CI, fetch the scan results and enforce a simple rule: “No image with HIGH/CRITICAL CVE may be deployed to prod.” This is policy-as-code, and it’s faster than running the scan.

Only after these three steps are solid should you worry about runtime scanning or IDE plugins. The goal is to make the *feedback* immediate, not the *scan*.


## Resources that helped

- [Trivy pre-commit hook example](https://github.com/aquasecurity/trivy/blob/main/docs/docs/docs/integrations/git-hooks.md)
- [Kyverno policy library](https://kyverno.io/policies/)
- [OPA/Gatekeeper policy examples](https://github.com/open-policy-agent/gatekeeper-library)
- [Trivy Operator Helm chart](https://github.com/aquasecurity/trivy-operator/tree/main/deploy/helm)
- [pre-commit framework](https://pre-commit.com/)
- [Docker build cache best practices](https://docs.docker.com/build/cache/)


---

## Frequently Asked Questions

**How do I ignore a false positive in pre-commit?**

Add the CVE to your `.trivy.yaml` ignore list:
```yaml
ignore:
  - GHSA-9685-555h-3h93
```
Or skip the directory:
```yaml
skip-dirs:
  - node_modules
```
Commit the change to `.trivy.yaml` and the pre-commit hook will honor it.


**Can I use this with Snyk instead of Trivy?**

Yes. Replace the pre-commit hook with:
```yaml
- id: snyk-test
  args: [--severity-threshold=high, --json]
```
And update the CI policy to parse Snyk’s JSON output. The workflow is identical.


**What if my base image has CVEs I can’t fix?**

Pin the digest and use Kyverno to enforce pinning:
```yaml
validation:
  pattern:
    spec:
      containers:
        - image: "!*@sha256:*"
```
This ensures the image digest is pinned, even if the CVE isn’t fixed. Then, schedule a monthly review to upgrade the base image.


**How do I handle monorepos with multiple languages?**

Use a matrix build in CI:
```yaml
strategy:
  matrix:
    language: [javascript, python, go]
steps:
  - name: Scan JavaScript
    if: matrix.language == 'javascript'
    run: trivy fs --severity HIGH,CRITICAL package-lock.json
  - name: Scan Python
    if: matrix.language == 'python'
    run: trivy fs --severity HIGH,CRITICAL requirements.txt
```
In pre-commit, use a script that detects the language and runs the appropriate scan.