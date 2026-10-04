# AI interviews break systems: 5 real mistakes in 2026

AI coding assistants produce code that is internally consistent but implicitly assumes an environment: a Linux runner, a Docker daemon at `/var/run/docker.sock`, root-equivalent privileges, and a recent toolchain. When that code lands in a real pipeline, the build fails with an error that names a symptom, not a cause. This article walks through five recurring failure modes, what each error actually means, and how to make a repository survive a heterogeneous set of runners.

## The error and why it misleads

A representative symptom looks like this:

```
Error: exec: "docker": executable file not found in $PATH
Command failed: docker build --platform linux/amd64 -t myapp:latest .
```

The message points at a missing binary. The actual problem is usually a mismatch between the assumptions baked into the generated workflow and the environment the workflow runs in. `docker` may be absent, or it may be present but pointed elsewhere by `DOCKER_HOST`, or the runner may use Podman, buildah, or Kaniko and expose no Docker CLI at all. The error is a red herring; the failure is the gap between assumed and actual environment.

A typical failure mode is a developer who validates locally on macOS with Docker Desktop, where the daemon runs as root and the socket sits at a well-known path, then pushes to a Linux CI runner that behaves differently. The build "works on my machine" and fails in CI, and the error text gives no hint about which assumption broke.

## The three layers that actually diverge

Environment drift between local, CI, and production shows up in three layers.

**Runtime assumptions.** The generated code expects a container runtime with a predictable binary name, socket path, and default build behavior. Real fleets mix GitHub-hosted runners, self-hosted runners, macOS laptops, and on-prem clusters.

**Environment variables.** `DOCKER_HOST`, `DOCKER_CERT_PATH`, and `DOCKER_TLS_VERIFY` are frequently set on developer machines and absent or differently set in CI. Conversely, a CI runner may set `DOCKER_HOST` to a Podman socket while the workflow still invokes `docker` directly.

**Privileged access.** Generated Dockerfiles and workflows often assume volume mounts, `--device` flags, or `--privileged` are available. Many CI runners disable privileged mode for security, and SELinux or AppArmor in enforcing mode blocks mounts that succeed elsewhere.

Tool version drift is a fourth, quieter layer: a workflow written against a recent buildx or Compose v2 will fail on a runner pinned to an older release, and the error rarely names the version mismatch.

## Failure mode 1: assuming a Docker binary and socket

This is the most common cause. The workflow calls `docker build`, but the runner has no `docker` binary, or has one that cannot reach a daemon.

The fix is to stop assuming the host provides a runtime. Two patterns work.

First, run the job inside a container image that ships the CLI:

```yaml
jobs:
  build:
    runs-on: ubuntu-24.04
    container:
      image: docker:25.0-cli
    steps:
      - uses: actions/checkout@v4
      - run: docker buildx build --platform linux/amd64 -t myapp:latest .
```

Second, make scripts runtime-agnostic so the same code path works under Docker or Podman:

```bash
#!/usr/bin/env bash
set -euo pipefail

RUNTIME=${DOCKER_RUNTIME:-docker}

if ! command -v "$RUNTIME" >/dev/null 2>&1; then
  if command -v podman >/dev/null 2>&1; then
    echo "docker not found; falling back to podman" >&2
    RUNTIME="podman"
  else
    echo "ERROR: neither docker nor podman found in PATH" >&2
    exit 1
  fi
fi

"$RUNTIME" run --rm myapp:latest
```

Note the quoting on `"$RUNTIME"` and the explicit failure path. A script that silently proceeds when no runtime exists converts a clear error into a confusing one later.

Version pinning matters for the same reason. Installing a pinned Compose release avoids silent breakage when the runner's default changes:

```yaml
- name: Install Docker Compose
  run: |
    curl -SL https://github.com/docker/compose/releases/download/v2.23.0/docker-compose-linux-x86_64 -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
```

Substitute whatever version the project actually validates against; the point is that the version is explicit and reviewable.

## Failure mode 2: environment variable leakage and privilege

The second cause is environment variables that exist locally and not in CI, or vice versa. A workflow that assumes a TCP-exposed daemon fails with:

```
Cannot connect to the Docker daemon at tcp://localhost:2375. Is the docker daemon running?
```

The generated workflow set `DOCKER_HOST=tcp://localhost:2375`, but the runner uses a Unix socket, or no socket at all. Most teams disable TCP daemon exposure for security, so this assumption is both fragile and a bad default.

The durable fix is to remove `DOCKER_HOST` from the workflow and rely on a container image that provides the runtime. If an explicit socket is unavoidable, set it deliberately:

```yaml
env:
  DOCKER_HOST: unix:///run/user/1001/podman/podman.sock
steps:
  - run: docker build -t myapp:latest .
```

Privileged access is the sibling problem. A workflow that runs `docker run --privileged` fails on runners that disallow it:

```
docker: Got permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock.
```

Prefer user namespaces over privileged mode where the workload permits:

```yaml
- run: docker run --rm --userns=keep-id myapp:latest
```

For GPU workloads, the NVIDIA Container Toolkit provides `--gpus` without requiring full privileged mode:

```yaml
- run: docker run --rm --gpus all myapp:latest
```

Architecture mismatch belongs in this layer too. A build that succeeds on `amd64` can fail on `arm64` when a base image or a compiled dependency has no arm64 variant:

```
ERROR: failed to solve: process "/bin/sh -c apk add --no-cache build-base" did not complete successfully: exit 1
```

Build for both explicitly, or pin the platform the runner actually provides:

```yaml
- run: docker buildx build --platform linux/amd64,linux/arm64 -t myapp:latest --push .
```

```yaml
- run: docker build --platform linux/amd64 -t myapp:latest .
```

## Failure mode 3: alternative runtimes and mandatory access control

Teams that build with buildah, Kaniko, or KinD have no Docker daemon. A `docker build` step fails with the socket error above even though a working build path exists.

For buildah:

```yaml
- name: Build with buildah
  run: |
    buildah bud --platform linux/amd64 -t myapp:latest .
    buildah push myapp:latest oci:myapp:latest
```

For KinD, build first, then load the image into the cluster:

```yaml
- name: Create KinD cluster
  run: |
    kind create cluster --name myapp-ci
    kind load docker-image myapp:latest
```

SELinux in enforcing mode produces a distinct class of failure when a container mounts a host path:

```
Error response from daemon: error while creating mount source path '/var/lib/docker/volumes/myapp/_data': mkdir /var/lib/docker/volumes/myapp/_data: permission denied
```

The correct fix depends on who controls the path. If the image owns the directory, label it at build time:

```dockerfile
VOLUME ["/data"]
RUN chcon -Rt container_file_t /data
```

If the image cannot be modified, relax the label for that container only:

```yaml
- run: docker run --rm --security-opt label=disable myapp:latest
```

`--security-opt label=disable` removes a security boundary. Treat it as a diagnostic step and a temporary measure, not a default.

Custom socket paths from Colima or Podman produce a variant of the same error:

```
Cannot connect to the Docker daemon at unix:///Users/runner/.colima/docker.sock. Is the docker daemon running?
```

Setting `DOCKER_HOST` to the correct path works, but a container image with the runtime baked in is less fragile because it does not depend on the host's socket layout at all.

## Worked example: diagnosing a permission failure

Consider a build that succeeds on a developer's laptop and fails in CI with `permission denied` on the Docker socket. Reasoning through it step by step:

1. The error names the socket, so the daemon exists and the CLI found it. The problem is authorization, not discovery.
2. On the laptop, Docker Desktop runs the daemon as root and the developer's user is in the `docker` group. In CI, the job runs as an unprivileged user that is not in that group.
3. Adding the CI user to the `docker` group grants root-equivalent access to the host, which is a poor trade in a shared runner.
4. The better fix is to avoid needing the host socket: run the build inside a container image that provides its own runtime, or use a rootless runtime such as Podman with a user socket.

The same reasoning applies to `USER node` in a Dockerfile combined with a runner that expects a different UID. The mismatch is between the image's declared user and the permissions the runtime grants that user, not between "correct" and "incorrect" code.

## How to verify a fix

Verification means reproducing the failure conditions, not just confirming the happy path.

Run the workflow on more than one runner class: a GitHub-hosted Linux runner, a hosted macOS runner if the project supports it, and a self-hosted runner with a different runtime. The build should succeed on each without repository changes.

Add a job that actually runs the container and asserts behavior:

```yaml
jobs:
  test:
    needs: build
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4
      - run: docker run --rm myapp:latest test
```

For a service, poll a health endpoint with a bounded retry loop so a slow start does not read as a failure:

```yaml
- name: Wait for container
  run: |
    for i in $(seq 1 30); do
      if curl -sf http://localhost:8080/health | grep -q '"status":"ok"'; then
        echo "Container healthy"
        exit 0
      fi
      sleep 1
    done
    echo "Container failed to become healthy" >&2
    exit 1
```

Exercise the runtime-agnostic script under both runtimes using a matrix:

```yaml
jobs:
  test:
    strategy:
      matrix:
        runtime: ["docker", "podman"]
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4
      - run: DOCKER_RUNTIME=${{ matrix.runtime }} ./run.sh
```

Finally, verify the image works where it will actually run. For a Kubernetes target, load it into a local cluster and wait for readiness:

```bash
kind create cluster
kubectl apply -f k8s/deployment.yaml
kubectl wait --for=condition=ready pod -l app=myapp --timeout=60s
kubectl logs -l app=myapp
```

If the pod fails, the logs distinguish an image problem from a scheduling or configuration problem.

## How to measure whether drift is still a problem

Rather than trusting a single green build, instrument the pipeline:

- **Time-to-first-successful-build per runner class.** Compare the median across runner types. A large spread indicates environment-specific assumptions.
- **Failure rate by error signature.** Group CI failures by the first error line. A cluster of `executable file not found` or `Cannot connect to the Docker daemon` messages localizes the assumption.
- **Rerun rate.** Count jobs that pass only after a retry. High rerun rates often indicate flaky environment setup rather than flaky tests.

These are cheap to collect from existing CI logs and require no new tooling. The absolute numbers matter less than the trend after a change.

## Prevention checklist

Before accepting a generated system, confirm each of these:

- The repository declares its container runtime and version, and CI pins the same version.
- No step assumes `docker` exists on the host; either the job runs in a runtime image or the script detects the runtime.
- `DOCKER_HOST` is either unset or set explicitly and documented.
- The build targets the architectures the runners actually provide.
- No step requires privileged mode; if one does, the reason is documented.
- SELinux or AppArmor status is recorded, and any label relaxation is scoped to one container.
- A validation script runs in CI before the build:

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "Validating environment..."

if command -v docker >/dev/null 2>&1; then
  echo "docker: $(docker --version)"
elif command -v podman >/dev/null 2>&1; then
  echo "podman: $(podman --version)"
else
  echo "ERROR: no container runtime found in PATH" >&2
  exit 1
fi

if [ -S /var/run/docker.sock ]; then
  echo "docker socket present at /var/run/docker.sock"
else
  echo "note: /var/run/docker.sock not present (expected under Podman or rootless setups)"
fi

case "$(uname -m)" in
  x86_64)  echo "architecture: amd64" ;;
  aarch64|arm64) echo "architecture: arm64" ;;
  *) echo "WARNING: unrecognized architecture $(uname -m)" >&2 ;;
esac

if command -v getenforce >/dev/null 2>&1; then
  echo "SELinux: $(getenforce)"
fi

echo "Environment validation complete."
```

- A Makefile or wrapper centralizes build commands so no workflow hardcodes a runtime:

```makefile
build:
	@if command -v docker >/dev/null 2>&1; then docker buildx build -t myapp:latest --platform linux/amd64 .; \
	elif command -v podman >/dev/null 2>&1; then podman build -t myapp:latest .; \
	else echo "no container runtime available" >&2; exit 1; fi
```

- The README states the validated runtime, version, architecture, runner type, and privilege requirements.

## Related errors and what they usually mean

| Error | Common cause | First thing to check |
|---|---|---|
| `permission denied` connecting to the Docker socket | CI user not in the `docker` group, or rootless socket ownership | Whether the job needs the host socket at all |
| `failed to solve: process did not complete successfully` | Missing build dependency or wrong base-image architecture | The failing `RUN` line and the platform flag |
| `no space left on device` | Layer cache filling the runner disk | Cache pruning and runner disk size |
| `the input device is not a TTY` | `-it` flags used in a non-interactive CI job | Remove `-it` in CI |
| `unable to find user node: no matching entries in passwd file` | `USER node` declared without creating the user | The Dockerfile's user creation step |
| `image was built with older schema version` | Old buildx producing a legacy manifest | The pinned buildx version |

## Escalation path when fixes do not resolve the failure

1. **Read the runner logs, not just the step output.** Runner-level logs often show environment setup failures the step output omits.
2. **Reproduce locally with an emulator.** Tools such as `act` run GitHub Actions workflows on a local machine and can expose assumptions that only hold in the real runner.
3. **Inspect daemon state.** `docker info` and `docker events --filter 'event=die'` (or `podman system info` and `podman events`) show whether the daemon is reachable and what it rejected.
4. **Compare runner classes.** If a self-hosted runner succeeds where a hosted runner fails, the difference is in kernel version, cgroup layout, or preinstalled tooling. Check `uname -a`, `stat /sys/fs/cgroup/cgroup.controllers`, `getenforce`, and `df -h`.
5. **Fall back to a runtime-independent build.** Buildpacks such as Paketo produce an image without requiring a Docker daemon on the build host:

```yaml
- uses: buildpacks/github-actions/setup-pack@v5
- run: pack build myapp --builder paketobuildpacks/builder:base
```

## FAQ

**Why does a Dockerfile work locally but fail in CI?**

Local setups commonly run Docker Desktop, where the daemon runs as root and the socket is at a well-known path. CI runners may use Colima, Podman, or no daemon at all, with different socket paths and privilege rules. The Dockerfile encodes an assumption about the host, and the host differs.

**How do you support Docker and Podman interchangeably?**

Centralize build and run commands in a wrapper script or Makefile that detects the available runtime and fails loudly when none is found. Avoid hardcoding `docker` in workflow steps.

**Why does a build fail with `permission denied` even when the job runs as root?**

Under SELinux in enforcing mode, root is still subject to mandatory access control. The container's mount path needs the correct label, or the container needs a scoped label relaxation. Check with `getenforce` before changing anything else.

**Is a container image in CI always better than installing the runtime on the runner?**

For reproducibility, yes: the image pins the CLI and its version. The trade-off is that mounting the host socket into that container reintroduces host dependence, so prefer a job container that performs the build itself when possible.

## What matters when reviewing a generated system

Generated code is not the problem; unstated assumptions are. A system is portable when it runs under more than one container runtime, on more than one architecture, without privileged mode, and validates its own environment before building. Those properties are testable, and they are the ones worth checking in review.

The practical consequence for review processes is that "it builds locally" carries little weight. What carries weight is a pipeline that has been run against the runner classes the project actually uses, with the environment recorded and the runtime pinned.

## Do this in the next 30 minutes

Add a `scripts/validate-env.sh` to the repository using the script above, wire it into the workflow as a step that runs before the build, and push a branch to confirm it prints the runtime, socket status, architecture, and SELinux state on your CI runner. If the output differs from what the Dockerfile assumes, you have found the first assumption to fix.
