# Agent API versioning: why clients break on upgrade

Agent services fail in a distinctive way when their version changes. The error surfaces at the transport layer, so engineers debug the network, the load balancer, or the model provider. The actual cause is usually a contract that moved without a coordinated version bump.

This article covers the failure modes that appear after a new agent capability ships, how to tell which layer is responsible, and the checks that catch the problem before production traffic does. It assumes a service that wraps a model behind an HTTP or gRPC interface and has at least one client that is deployed separately from the server.

## The symptom is misleading

A versioning failure rarely announces itself as a versioning failure. It looks like one of these:

- A spike in 502 or 504 responses right after a deploy.
- A JSON payload that no longer matches the client's expected shape.
- A deserialization error on a field the client has never heard of.
- An exception such as `AgentCapabilityError: unsupported version 2.3.0 (minimum supported: 2.4.1)`.

That last message reads like a simple mismatch: the client asks for 2.3.0, the server only accepts 2.4.1 and above. In practice the version string is often the least interesting part of the problem. The client may be pinned correctly and the server may be running exactly what was intended. What changed is the shape of the data flowing between them.

Three layers usually overlap:

1. **Semantic versioning misuse.** A change is labelled minor or patch because it "only adds a field", but the field is required, or the server now enforces a rule that depends on it.
2. **Mutable image tags.** A deployment references `latest` or `stable`, so the binary behind a stable name changes without any version number moving.
3. **Feature-flag drift.** A capability is enabled by configuration rather than by version, and the client has no way to observe that configuration.

Because the error appears at the HTTP layer, the first instinct is to inspect latency, connection pools, and timeouts. Those are worth ruling out, but they rarely explain an error that starts within seconds of a deploy and affects one client while others stay healthy.

## What is actually happening: contract erosion

The contract between an agent and its consumers is the set of things the consumer is allowed to rely on: field names, types, required versus optional status, error codes, and the semantics of a successful response. In a well-run system that contract lives somewhere explicit, such as a versioned schema file, a protobuf definition, or a generated client library. When it lives only in the server's serialization code, the contract is whatever the current binary happens to emit.

Consider a worked example. An order-fulfillment agent communicates over gRPC. The request message gains a new `priority` field, and the release is tagged 2.5.0 on the assumption that adding a field is additive.

- The server now rejects orders that arrive without a `priority` value, because a business rule was added in the same change.
- The client was generated from the 2.4.0 definition. Protobuf ignores unknown fields on the wire, so the client sends no `priority` and never notices anything is wrong.
- The server returns `FAILED_PRECONDITION`, which the client maps to its generic capability error.

Every individual step is defensible. The release is broken because the version number described the wire format, while the breaking change was in the validation logic. That is contract erosion: the declared contract and the enforced contract diverged, and only one of them was versioned.

The same pattern appears with JSON APIs. A new required field, a tightened enum, a changed default, or a new error code can all break a client without any schema-level incompatibility that a naive diff would flag.

## Fix 1: make deployment artifacts immutable

The cheapest structural fix is to stop letting a name resolve to a different binary over time. Build the image with an explicit version tag, push it, and reference that exact tag (or its digest) in the deployment manifest.

```bash
# Build the agent image with a fixed version tag
docker build -t 123456789012.dkr.ecr.us-east-1.amazonaws.com/agent:2.5.0 \
    --build-arg PYTHON_VERSION=3.11 .
# Push the image
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-east-1.amazonaws.com
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/agent:2.5.0
```

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-service
spec:
  replicas: 3
  template:
    spec:
      containers:
        - name: agent
          image: 123456789012.dkr.ecr.us-east-1.amazonaws.com/agent:2.5.0
          env:
            - name: AGENT_VERSION
              value: "2.5.0"
```

Freezing the tag does not by itself prevent a client from breaking, but it removes one entire class of surprise: the binary a client was tested against is the binary it keeps talking to until someone deliberately changes the reference. It also makes rollback a matter of redeploying a known tag rather than reconstructing which build was live an hour ago.

Two details matter. First, `AGENT_VERSION` should be set from the same source that produced the image tag, not typed by hand, or the two will drift. Second, if you want reproducibility guarantees stronger than a tag, reference the image by digest in the manifest; a tag can be re-pushed, a digest cannot be changed.

## Fix 2: contract-first with compatibility checks in CI

Immutable tags stop silent binary drift. They do nothing about a contract change that is genuinely incompatible. For that, the contract needs to be an artifact under version control, and pull requests need to be checked against the last released version.

```yaml
# .github/workflows/contract-check.yml
name: Contract Compatibility
on: [pull_request]
jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install schema validator
        run: pip install openapi-spec-validator==0.7.1
      - name: Validate new spec
        run: openapi-spec-validator ./specs/agent-api.yaml
      - name: Compatibility check
        run: |
          python scripts/compare_specs.py \
            --old specs/agent-api-2.4.0.yaml \
            --new specs/agent-api-2.5.0.yaml
```

The comparison script should fail the build when it finds a change that a client generated from the old spec could not survive. The categories worth checking are:

- A field that was optional becomes required.
- A field is removed or renamed.
- An enum loses a value the client may send, or gains one the client may receive without a documented fallback.
- An error code is added to a set the client treats as exhaustive.
- A default value changes.

For a JSON API, a schema diff tool can detect the first, second, and fifth categories mechanically. The third and fourth usually need a short allowlist or a human review step, because they depend on how clients handle unknown values. If the diff reports a breaking change, the build fails and the author either raises the major version or makes the change backward compatible.

The important property is not the specific tool. It is that the contract is a file, the file is diffed on every pull request, and the diff has the authority to block a merge.

## Fix 3: make configuration observable

Feature flags move behaviour without moving a version number, which is exactly why they cause version-shaped failures. The standard mitigation is to expose the effective configuration through a health or status endpoint, so a client or an operator can see what the server will actually do.

```python
# health.py
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

FLAGS = {
    "new_order_priority": True,
    "strict_currency_check": False,
}

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({
            "status": "ok",
            "version": "2.5.0",
            "flags": [name for name, on in FLAGS.items() if on],
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

if __name__ == "__main__":
    HTTPServer(("0.0.0.0", 8080), HealthHandler).serve_forever()
```

Two rules make this useful rather than decorative. The endpoint must report the *effective* flag state, read from the same source the request path reads from, not a hardcoded copy. And the same binary must be deployed to every environment, with the flag source doing the differentiating; if staging runs different code, the health endpoint only tells you about the code, not the configuration.

Clients can then check the reported version and flags before sending a request that depends on them, and degrade gracefully instead of failing at deserialization time. Operators get a fast answer to "is this environment configured the way I think it is", which is the question that otherwise gets answered by reading logs for twenty minutes.

## How to measure whether the fix worked

None of the above is verifiable without instrumentation. Set up the following before the next release, so you have a baseline to compare against.

**1. Contract diff in CI.** The pipeline should print an explicit pass or fail for compatibility against the last released spec. Treat a missing result as a failure.

**2. Integration test against the exact released artifact.** Start the container using the pinned tag, then run a client test that sends a payload exercising the new fields and asserts both the status code and the presence of the expected version field in the response. The test must use the pinned tag, not a locally built image, or it is testing something other than what will run in production.

**3. Health endpoint check.** Query the status endpoint and confirm the reported version and flag list match the intended state for that environment. Script this, because it is exactly the kind of check that gets skipped manually.

**4. Error-rate comparison.** Instrument a counter for your capability or deserialization error class, labelled by client version if possible. Record the count per minute for the hour before and the hour after the change. The useful signal is the shape: a step change that begins at the deploy time and affects one client version points at the contract, while a gradual rise across all clients points elsewhere. Percentages from someone else's system will not tell you what your baseline is; measure your own.

**5. Artifact identity.** Confirm the digest of the deployed image matches the digest recorded as the CI artifact for that tag.

If all five hold, the release is consistent. If any fails, you have located the layer at which the version and the contract diverged.

## Failure modes to watch for

**The additive change that is not additive.** Adding a field is safe only if nothing enforces it. Check whether validation, routing, or a downstream service reads the new field conditionally on its presence.

**Unknown-field handling.** Some deserializers reject unknown fields by default. Before assuming a new field is backward compatible, confirm how each client's serialization library handles fields it does not recognise, and whether that behaviour is configurable.

**Enum extension.** Adding a value to an enum that clients treat as exhaustive causes failures at the client, not the server. Version the enum's open-endedness explicitly, or document that clients must tolerate unknown values.

**Flag flips without a version change.** A flag that changes response shape is a contract change wearing a configuration costume. Either version it, or guarantee that both states are valid for every supported client.

**Rollback that does not roll back.** If the old binary is restored but the flag state or the contract file is not, the rollback is incomplete. Rollback procedures should name every artifact that moves together.

**Multiple clients, one version.** A server version can be compatible with one client and incompatible with another. Track which client versions are actually sending traffic, so "we only support 2.4.1 and above" is a statement about real consumers rather than an assumption.

## Decision checklist

Use this when planning a change to an agent's interface.

- Does the change alter anything a client can observe: field presence, required status, enum values, error codes, defaults, or timing guarantees?
- If yes, is it breaking for at least one deployed client version? If you cannot answer, find out which client versions are live before merging.
- Is the change expressed as a version bump, a flag, or both? Prefer one mechanism with a clear owner.
- Is the contract file updated in the same pull request as the implementation?
- Does CI diff the contract against the last release and fail on breaking changes?
- Is the deployed artifact pinned by tag or digest, and is that reference recorded?
- Does the status endpoint report the effective version and flags?
- Is there a rollback step that covers code, configuration, and contract together?

## FAQ

**Can an optional field still break a client?**
Yes. Optionality is a property of the schema; whether a client tolerates the field depends on its deserialization configuration and on whether any server-side logic reacts to the field's presence. Verify both.

**Should the version live in the URL, a header, or the payload?**
Any of these works if it is consistent and if the server rejects unsupported versions explicitly rather than guessing. The failure mode to avoid is a version that is transmitted but never checked.

**When is an image digest preferable to a tag?**
When you need a guarantee that the artifact cannot change, such as for reproducibility or audit. Tags are easier for humans to read; digests are unambiguous. Many teams use a readable tag in the manifest and record the resolved digest alongside it.

**How do you test compatibility with an older client without deploying the old server?**
Run the previous artifact locally, for example with a container runtime or a test container library, and execute the current client's test suite against it. This catches cases where the new client sends something the old server cannot parse.

**Do feature flags belong in a versioning strategy at all?**
They belong in the rollout strategy. If a flag can change the response contract, it needs the same compatibility review as a schema change, because clients cannot see flag state unless you expose it.

**What is the minimum viable version gate?**
A contract file under version control, a CI step that diffs it against the last release, and a rule that a failing diff blocks the merge. Everything else is refinement.

## Take this action in the next 30 minutes

Open your deployment manifest for the agent service and find every image reference. If any of them uses `latest`, `stable`, or another mutable tag, replace it with the exact version tag of the build currently running in production, then confirm the running pod's image digest matches the digest recorded for that tag in your registry. That single change removes the most common source of "the version did not change but the behaviour did".
===END===
