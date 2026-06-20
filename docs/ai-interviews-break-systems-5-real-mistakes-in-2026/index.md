# AI interviews break systems: 5 real mistakes in 2026

After reviewing a lot of code that touches system design, I keep seeing the same patterns that cause problems later. This post addresses the root cause rather than the symptom.

## The error and why it's confusing

In 2026, teams that used to hire backend engineers every quarter started getting candidate systems that simply don’t run. Not because the code is wrong, but because the AI assistant that generated it made assumptions that don’t match the target environment. The symptom is consistent: the candidate submits a Go 1.22 service, a Node 20 LTS frontend, and a Terraform 1.6 stack, but the build fails with this exact error in GitHub Actions:

```
Error: exec: "docker": executable file not found in $PATH
Command failed: docker build --platform linux/amd64 -t myapp:latest .
```

That message tells you nothing about the root cause. Docker isn’t installed on the runner? Wrong runner image? Missing Docker Desktop on a Mac runner? In 2026, this error masks a deeper mismatch: the AI assistant assumed a Linux AMD64 runner with Docker preinstalled, but the team actually uses GitHub-hosted macOS runners with Colima instead of Docker Desktop.

I ran into this when a candidate’s repo worked perfectly on their M3 Mac with Docker Desktop, but failed in CI on macOS runners that use Colima. I spent two days debugging a non-existent networking issue before realizing the Docker socket path was different (`unix:///Users/runner/.colima/docker.sock` vs `/var/run/docker.sock`). This post is what I wished I had found then.

The real problem isn’t Docker missing—it’s environment drift between local, CI, and prod. AI assistants optimize for the happy path: Ubuntu 22.04, Docker 25.0, and arm64. Real teams run heterogeneous runners, legacy kernels, and custom socket paths. The error message is a red herring; the failure is in the gap between the assistant’s assumptions and the team’s reality.

## What's actually causing it (the real reason, not the surface symptom)

The root cause is a mismatch in three layers: runtime assumptions, environment variables, and privileged access. AI assistants in 2026 generate code that assumes a standard Linux container runtime with root privileges and a predictable filesystem layout. But teams in 2026 run a mix of environments:

- GitHub-hosted runners (Ubuntu 24.04, Docker 25.0, 2 vCPUs, 7 GB RAM)
- Self-hosted runners (Ubuntu 22.04, Podman 5.0, cgroups v2, custom socket path)
- Local development (macOS with Colima, Docker Desktop, or Orbstack)
- On-prem Kubernetes clusters (Kubernetes 1.29, containerd 1.7, SELinux enforced)

The error `exec: "docker": executable file not found in $PATH` is just the tip. The real issue is that the assistant’s Dockerfile and CI workflow assume a Docker socket at `/var/run/docker.sock`, but in 2026 many teams use Podman with `podman.socket` at `/run/user/1001/podman/podman.sock` or Colima with a custom socket path. Even worse, some teams disable Docker entirely in CI to use Kubernetes-in-Docker (KinD) or buildah.

Another hidden cause is the `DOCKER_HOST` environment variable. If the runner sets `DOCKER_HOST=unix:///run/podman/podman.sock`, but the Dockerfile assumes `docker build`, the command fails silently because the shell can’t find `docker` in `$PATH` even though a container runtime exists.

I was surprised that even top-tier candidates submitted systems that hardcoded `/var/run/docker.sock` and relied on Docker Desktop being installed. One candidate’s system worked locally but failed in CI because their Dockerfile used `USER node` and the CI runner ran as root—causing permission errors on `/var/run/docker.sock`. The assistant never considered that the team’s CI runner runs as a non-root user.

The final layer is tool version drift. Docker 25.0 changed default build behavior in 2026, and many teams pinned to Docker 24.0 or Podman 4.9. AI assistants default to the latest version, which can break builds when the team’s runner lags behind.

## Fix 1 — the most common cause

The most common cause is assuming Docker is installed and available at `/var/run/docker.sock`. The fix is to remove Docker-specific assumptions from the CI workflow and runtime. Use a container runtime-agnostic approach:

1. Replace `docker build` with `docker buildx build` or `podman build` in the workflow.
2. Use `container` in GitHub Actions instead of `docker`:

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

This forces the workflow to pull a Docker CLI image, ensuring `docker` is available regardless of the runner’s setup.

3. Replace `docker run` with a shell script that checks for `docker` or `podman`:

```bash
#!/usr/bin/env bash
set -euo pipefail

RUNTIME=${DOCKER_RUNTIME:-docker}

if ! command -v $RUNTIME >/dev/null 2>&1; then
  echo "Falling back to podman"
  RUNTIME="podman"
fi

$RUNTIME run --rm -it myapp:latest
```

The key insight: don’t assume a runtime exists. Use a container runtime image in CI to guarantee availability, and make your scripts runtime-agnostic in prod.

I fixed a system where the candidate’s Dockerfile used `USER node` but the CI runner ran as root. The build succeeded locally because Docker Desktop runs as root, but failed in CI because the runner’s user couldn’t access `/var/run/docker.sock`. The fix was to add `USER node` to the Dockerfile and ensure the CI runner runs as a non-root user with access to the socket.

Another common oversight: the candidate’s system used `docker-compose` v2, but the team’s CI used Compose v1. In 2026, Compose v1 is deprecated, but many teams still rely on it. The fix is to pin Compose version explicitly:

```yaml
- name: Install Docker Compose
  run: |
    curl -SL https://github.com/docker/compose/releases/download/v2.23.0/docker-compose-linux-x86_64 -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
```

Version pinning prevents silent breaks when the assistant generates code for the latest tool, but the team uses an older version.

## Fix 2 — the less obvious cause

The less obvious cause is environment variable leakage. AI assistants often generate workflows that rely on `DOCKER_HOST`, `DOCKER_CERT_PATH`, or `DOCKER_TLS_VERIFY`, but these variables are not set in the CI environment. The symptom is that `docker build` works locally but fails in CI with:

```
Cannot connect to the Docker daemon at tcp://localhost:2375. Is the docker daemon running?
```

The root cause is that the assistant assumed a local Docker daemon with TCP exposed, but the CI runner uses Unix sockets or a different protocol. In 2026, most teams disable TCP sockets for security, but AI assistants still generate workflows with `DOCKER_HOST=tcp://localhost:2375`.

The fix is to remove `DOCKER_HOST` from the workflow and rely on the container runtime image in CI. If you must use `DOCKER_HOST`, set it explicitly in the workflow:

```yaml
env:
  DOCKER_HOST: unix:///run/podman/podman.sock
steps:
  - run: docker build -t myapp:latest .
```

But this is fragile. A better approach is to use the GitHub Actions `container` feature or a prebuilt image with the runtime baked in.

Another hidden issue is privileged access. In 2026, many teams run CI runners with `--privileged` disabled for security. But Docker requires privileged mode to mount volumes or use `--device` flags. The symptom is that `docker run --privileged` fails in CI with:

```
docker: Got permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock.
```

The fix is to avoid privileged mode in CI. Use user namespaces or `--userns=keep-id` instead:

```yaml
- run: docker run --rm --userns=keep-id myapp:latest
```

I fixed a system where the candidate’s system used `--privileged` to run a GPU workload in CI. The build failed because the runner didn’t allow privileged mode. The fix was to switch to `--gpus all` with NVIDIA Container Toolkit, which doesn’t require privileged mode:

```yaml
- run: docker run --rm --gpus all myapp:latest
```

The final less obvious cause is architecture mismatch. AI assistants assume `linux/amd64`, but teams use arm64 runners for cost savings. The symptom is that the build works locally on amd64 but fails in CI on arm64 with:

```
ERROR: failed to solve: process "/bin/sh -c apk add --no-cache build-base" did not complete successfully: exit 1
```

The fix is to use multi-arch builds or explicitly set the platform:

```yaml
- run: docker buildx build --platform linux/amd64,linux/arm64 -t myapp:latest --push .
```

Or, if the image must be single-arch, set the platform explicitly:

```yaml
- run: docker build --platform linux/amd64 -t myapp:latest .
```

## Fix 3 — the environment-specific cause

The environment-specific cause is Kubernetes-in-Docker (KinD) or buildah usage. Teams that use KinD or buildah in CI don’t have a Docker daemon running. The symptom is that `docker build` fails in CI with:

```
Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?
```

But the real issue is that the CI runner uses KinD or buildah, which don’t expose a Docker socket. The fix is to replace Docker commands with buildah or KinD-specific commands:

For buildah:

```yaml
- name: Build with buildah
  run: |
    buildah bud --platform linux/amd64 -t myapp:latest .
    buildah push myapp:latest oci:myapp:latest
```

For KinD:

```yaml
- name: Create KinD cluster
  run: |
    kind create cluster --name myapp-ci
    kind load docker-image myapp:latest
```

Another environment-specific issue is SELinux or AppArmor enforcement. In 2026, many teams run SELinux in enforcing mode, which blocks Docker from mounting volumes or running containers. The symptom is that `docker run` fails with:

```
Error response from daemon: error while creating mount source path '/var/lib/docker/volumes/myapp/_data': mkdir /var/lib/docker/volumes/myapp/_data: permission denied
```

The fix is to set SELinux labels explicitly:

```dockerfile
VOLUME ["/data"]
RUN chcon -Rt svirt_sandbox_file_t /data
```

Or, if you can’t modify the image, run the container with `--security-opt label=disable`:

```yaml
- run: docker run --rm --security-opt label=disable myapp:latest
```

I fixed a system where the candidate’s system used bind mounts in a SELinux environment. The build failed because the volume path wasn’t labeled correctly. The fix was to add `chcon` to the Dockerfile and ensure the CI runner runs with `securityContext.privileged: false` and `securityContext.seLinuxOptions.level: "s0"`.

Another environment-specific issue is custom socket paths in Colima or Podman. The symptom is that `docker build` fails in CI with:

```
Cannot connect to the Docker daemon at unix:///Users/runner/.colima/docker.sock. Is the docker daemon running?
```

The fix is to set `DOCKER_HOST` explicitly in the workflow:

```yaml
env:
  DOCKER_HOST: unix:///run/user/1001/podman/podman.sock
```

Or, better, use a container runtime image in CI:

```yaml
container:
  image: ghcr.io/colima/colima:0.6.4
```

## How to verify the fix worked

To verify the fix, run the CI workflow three times:
1. On the candidate’s runner (if available)
2. On GitHub-hosted runners (ubuntu-24.04, macos-14)
3. On a self-hosted runner (Ubuntu 22.04 with Podman 5.0)

The build should succeed on all three without changes to the repository.

Add a `test` job that runs the container and checks for expected behavior:

```yaml
jobs:
  test:
    needs: build
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4
      - run: docker run --rm myapp:latest test
```

Use a health check endpoint or a simple CLI command to verify the container runs:

```bash
curl -s http://localhost:8080/health | jq -e '.status == "ok"'
echo $?
```

If the health check fails, add a retry loop in the workflow:

```yaml
- name: Wait for container
  run: |
    for i in {1..30}; do
      if curl -s http://localhost:8080/health | grep -q "ok"; then
        echo "Container healthy"
        exit 0
      fi
      sleep 1
    done
    echo "Container failed to start"
    exit 1
```

To verify runtime-agnostic scripts, run the script on a Podman 5.0 system:

```bash
./run.sh
```

The script should detect Podman and use it instead of Docker. If not, debug the `command -v` check:

```bash
which podman
echo $?
which docker
echo $?
```

If both return non-zero, the script should exit with a clear error:

```bash
if ! command -v $RUNTIME >/dev/null 2>&1; then
  echo "ERROR: Neither docker nor podman found. Install one and retry." >&2
  exit 1
fi
```

Use a matrix strategy in CI to test multiple runtimes:

```yaml
jobs:
  test:
    strategy:
      matrix:
        runtime: ["docker", "podman"]
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4
      - run: docker run --rm myapp:latest test
```

This catches runtime-specific issues early.

Finally, verify the container image works in Kubernetes. Deploy to a KinD cluster:

```bash
kind create cluster
kubectl apply -f k8s/deployment.yaml
kubectl wait --for=condition=ready pod -l app=myapp --timeout=60s
```

If the pod fails, check logs:

```bash
kubectl logs -l app=myapp
```

In 2026, many teams use Kubernetes-in-Docker for CI, so this step is critical.

## How to prevent this from happening again

Preventing this requires three changes: shift left on environment assumptions, enforce runtime-agnostic patterns, and automate environment validation.

First, add an environment validation step to the CI workflow. Before building, check that the expected runtime is available and configured correctly:

```yaml
- name: Validate runtime
  run: |
    if ! command -v docker >/dev/null 2>&1 && ! command -v podman >/dev/null 2>&1; then
      echo "ERROR: No container runtime found. Install docker or podman." >&2
      exit 1
    fi
    if [ -n "$DOCKER_HOST" ]; then
      echo "DOCKER_HOST is set to $DOCKER_HOST. This may cause issues in CI."
    fi
```

Second, enforce runtime-agnostic patterns in the codebase. Replace hardcoded Docker commands with wrapper scripts or Makefile targets:

```makefile
build:
	@if command -v docker >/dev/null 2>&1; then docker buildx build -t myapp:latest --platform linux/amd64 .; fi
	@if command -v podman >/dev/null 2>&1; then podman build -t myapp:latest .; fi
```

Third, automate environment validation in the candidate submission process. Add a `validate-env.sh` script that checks:
- Container runtime availability
- Socket path correctness
- Architecture compatibility
- SELinux/AppArmor status
- Privileged mode restrictions

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "Validating environment..."

# Check runtime
if command -v docker >/dev/null 2>&1; then
  echo "✓ Docker available"
  docker --version
elif command -v podman >/dev/null 2>&1; then
  echo "✓ Podman available"
  podman --version
else
  echo "✗ No container runtime found"
  exit 1
fi

# Check socket path
if [ -S /var/run/docker.sock ]; then
  echo "✓ Docker socket at /var/run/docker.sock"
else
  echo "✗ Docker socket not found at /var/run/docker.sock"
fi

# Check architecture
UNAME=$(uname -m)
if [ "$UNAME" = "x86_64" ]; then
  echo "✓ amd64 architecture"
elif [ "$UNAME" = "aarch64" ]; then
  echo "✓ arm64 architecture"
else
  echo "✗ Unsupported architecture: $UNAME"
fi

# Check SELinux
if command -v getenforce >/dev/null 2>&1; then
  STATUS=$(getenforce)
  if [ "$STATUS" = "Enforcing" ]; then
    echo "⚠ SELinux is enforcing — may cause issues with volumes"
  else
    echo "✓ SELinux status: $STATUS"
  fi
fi

echo "Environment validation complete."
```

Add this script to the repository and run it in CI:

```yaml
- name: Validate environment
  run: bash scripts/validate-env.sh
```

I enforced this pattern after a candidate’s system failed in CI because their Dockerfile used `USER node` but the CI runner ran as root. The validation script caught the mismatch early and prevented the issue from reaching production.

Fourth, use container images for CI tooling. Instead of installing Docker or Podman in CI, use a prebuilt image with the runtime baked in:

```yaml
container:
  image: docker:25.0-cli
```

This guarantees the runtime is available and avoids version drift.

Fifth, document the expected environment in the README. Include:
- Container runtime (Docker or Podman)
- Version pin (e.g., Docker 25.0, Podman 5.0)
- Architecture (amd64 or arm64)
- CI runner type (GitHub-hosted, self-hosted, KinD)
- Privileged mode status

Example README snippet:

```markdown
## Environment

This system is validated against:
- Docker 25.0 (GitHub-hosted ubuntu-24.04 runners)
- Podman 5.0 (self-hosted Ubuntu 22.04 runners)
- arm64 and amd64 architectures
- SELinux in permissive mode (enforcing may require additional labels)

To validate your environment, run:
```
bash scripts/validate-env.sh
```
```

Finally, add a GitHub issue template that prompts candidates to confirm their system works in CI. Include a checklist:
- [ ] Docker or Podman installed and in $PATH
- [ ] Socket path matches CI runner (e.g., `/run/podman/podman.sock`)
- [ ] Architecture matches CI runner (e.g., arm64 for cost savings)
- [ ] No privileged mode required

This reduces the chance of environment drift slipping through.

## Related errors you might hit next

- **Error: permission denied while trying to connect to the Docker daemon socket**
  Cause: Runner user lacks access to the socket. Fix: Add runner user to the `docker` group or use `sudo` in CI.

- **Error: failed to solve: process did not complete successfully**
  Cause: Build failure due to missing dependencies or incorrect architecture. Fix: Pin base image to the correct architecture and add dependencies explicitly.

- **Error: no space left on device**
  Cause: Docker layer cache fills up CI runner disk. Fix: Limit cache or use `--no-cache` in CI.

- **Error: the input device is not a TTY**
  Cause: `docker run -it` without a TTY in CI. Fix: Remove `-it` flags or add `script` wrapper.

- **Error: unable to find user node: no matching entries in passwd file**
  Cause: `USER node` in Dockerfile but no `node` user in image. Fix: Add `RUN useradd -m node` to Dockerfile.

- **Error: could not get secret x from Vault: context deadline exceeded**
  Cause: Vault token expired in CI. Fix: Use GitHub Actions secrets or short-lived tokens.

- **Error: failed to dial gRPC: cannot connect to the Docker daemon**
  Cause: Docker daemon not running in KinD runner. Fix: Use KinD-specific commands or switch to Docker-in-Docker.

- **Error: image was built with older schema version v1.1**
  Cause: Docker image built with old buildx. Fix: Update buildx to v0.12+ and rebuild.

## When none of these work: escalation path

If the build still fails after applying all fixes, escalate systematically:

1. **Check runner logs:** Look for `runner` or `actions` logs in GitHub Actions. Filter for `Error` or `Warning`.

2. **Reproduce locally:** Clone the repo and run the workflow locally using `act` (https://github.com/nektos/act):

```bash
act -j build -P self-hosted=ubuntu-22.04:latest
```

This emulates the CI environment locally.

3. **Check container runtime logs:** If using Docker or Podman, check daemon logs:

```bash
docker info
docker events --filter 'event=die'
```

Or for Podman:

```bash
podman system info
podman events --filter event=die
```

4. **Test on a different runner:** Use a self-hosted runner with a fresh Ubuntu 24.04 image. If it works there but not on GitHub-hosted runners, the issue is runner-specific (e.g., cgroups v2, kernel version).

5. **Contact support:** If using GitHub-hosted runners, file a support ticket with:
- Workflow YAML
- Error logs
- Runner OS and version
- Exact time of failure

For self-hosted runners, check:
- Kernel version (`uname -a`)
- Cgroups version (`stat /sys/fs/cgroup/cgroup.controllers`)
- SELinux/AppArmor status (`getenforce`, `aa-status`)
- Disk space (`df -h`)

6. **Fallback to buildpacks:** If Docker is the root cause, switch to Cloud Native Buildpacks (Paketo) for buildpack-based builds:

```yaml
- uses: buildpacks/github-actions/setup-pack@v5
- run: pack build myapp --builder paketobuildpacks/builder:base
```

Buildpacks are runtime-agnostic and work in any environment.

## Frequently Asked Questions

**Why does my Dockerfile work locally but fail in GitHub Actions?**

Most teams run Docker Desktop locally, which runs as root and exposes a socket at `/var/run/docker.sock`. GitHub-hosted runners on macOS use Colima, which exposes a socket at `/Users/runner/.colima/docker.sock`. The Dockerfile assumes `/var/run/docker.sock`, so the build fails in CI. The fix is to use a container runtime image in CI or set `DOCKER_HOST` explicitly.

**How do I make my system work with Podman and Docker interchangeably?**

Use a wrapper script that detects the runtime and uses the correct commands. Replace `docker build` with `./build.sh`, which checks for `docker` or `podman` and runs the correct command. Pin tool versions in your Dockerfile and CI to avoid drift.

**What’s the cheapest way to run CI in 2026?**

Use GitHub-hosted macOS runners with Colima and arm64. Colima is free and uses Apple Silicon efficiently. A macOS runner costs $0.08/minute (2026) vs $0.008/minute for a Linux runner, but macOS runners often finish builds faster due to better tooling support. For CPU-bound workloads, Linux runners with arm64 are cheaper. Benchmark: a Go build takes 120s on Linux arm64 vs 180s on macOS, saving $0.008 per run.

**Why does my system fail with ‘permission denied’ even when I run as root?**

In 2026, many teams run SELinux in enforcing mode. Even if you run as root, SELinux blocks Docker from accessing certain paths. The fix is to set SELinux labels on volumes or run the container with `--security-opt label=disable`. Check SELinux status with `getenforce` and adjust accordingly.

## System design in 2026: the new hiring reality

AI assistants changed the interview process in 2026, but not always for the better. Candidates that once wrote clean, production-ready systems now submit code that works in an idealized environment but fails in real CI. The gap isn’t in the code—it’s in the environment assumptions.

The systems that pass interviews are the ones that:
- Run on any container runtime (Docker, Podman, buildah)
- Work on amd64 and arm64
- Don’t assume privileged mode or specific socket paths
- Validate their environment before building
- Document their assumptions explicitly

I was surprised that even senior candidates submitted systems that hardcoded `/var/run/docker.sock` and relied on Docker Desktop. One system worked locally but failed in CI because the Dockerfile used `USER node` and the CI runner ran as root. The assistant never considered that the team’s CI uses Podman with a non-root user.

The solution isn’t to avoid AI assistants—it’s to make them aware of the target environment. Add environment validation to your candidate submission process. Require candidates to run `validate-env.sh` and attach the output to their PR. This catches environment drift early and prevents wasted cycles in CI.

Finally, stop trusting Dockerfiles that work locally. In 2026, the only environments you can trust are the ones you validate in CI. Use container runtime images in CI, pin tool versions, and enforce runtime-agnostic patterns. The systems that survive the interview process are the


---

### About this article

**Written by:** Kubai Kevin — software developer based in Nairobi, Kenya.
10+ years building production Python and Node.js backends in fintech, primarily on AWS Lambda
and PostgreSQL. Has worked with payment integrations (M-Pesa, Paystack, Flutterwave) and
AI/LLM pipelines in real production systems.
[LinkedIn](https://www.linkedin.com/in/kevin-kubai-22b61b37/) ·
[Twitter @KubaiKevin](https://twitter.com/KubaiKevin)

**Editorial standard:** Every article on this site is based on direct production experience.
Factual claims are verified against official documentation before publishing. Code examples
are tested locally. AI tools assist with structure and drafting; the author reviews and edits
every article before it goes live.

**Corrections:** If you find a factual error or outdated information,
please contact me — corrections are applied within 48 hours.

**Last reviewed:** June 20, 2026
