# Distroless images: smaller attack surface

A container image is a filesystem, a process entrypoint, and a set of metadata. Hardening it means shrinking what's in that filesystem and what that process can do. The usual starting point is a general-purpose base image such as `ubuntu:22.04` or `node:20`. Those images are built to be useful for many jobs, so they include a shell, package managers, and debugging tools. In a build environment that is helpful. In a runtime container it is a liability: every extra binary is another potential entry point, and every extra package is another CVE to track.

The recurring failure mode is treating image hardening as a one-time cleanup — switch to Alpine, run a scanner once, call it done — after which the next base image bump reintroduces a shell or a vulnerable library. Hardening has to be a repeatable pipeline step. The sequence below is a practical one: measure the baseline, choose a minimal base, move to distroless where it fits, make the scanner block bad builds, and verify the result with the same tools an attacker would use.

## Prerequisites

You need a container runtime (Docker or Podman), a registry you can push to, and a CI system that can run a scanner. The examples use Docker and GitHub Actions, but the same steps work with GitLab CI, Jenkins, or any runner that can execute shell commands.

You also need to know two things about your application: which dynamic libraries it links against at runtime, and whether it needs a shell for its entrypoint script. If the answer to the second is yes, distroless will require a small refactor — more on that later.

## Step 1: Measure the image you have before changing anything

Before choosing a base image, get a baseline. Two numbers matter: the number of installed packages and the number of known vulnerabilities. The package count is available without a scanner on Debian-derived images:

```bash
docker run --rm --entrypoint sh your-image:latest -c 'dpkg -l | wc -l'
```

For vulnerabilities, use a scanner. Trivy is a widely used open-source option that works on local images and in CI:

```bash
trivy image --severity HIGH,CRITICAL your-image:latest
```

Record the counts. This baseline is what you compare against after each change. Without it, there is no way to tell whether a switch to a smaller base actually helped or just moved the CVEs around.

A common trap at this stage: scanning only the `latest` tag. If deployment uses a digest or a version tag, scan that exact reference. Tags move; digests don't. When comparing two images, compare the same way each time — same scanner version, same database update, same severity filter — otherwise the delta reflects the scanner rather than the image.

## Step 2: Pick a minimal base image that matches your runtime

The first instinct is Alpine. Alpine uses musl libc instead of glibc, and that difference matters. Native Node.js modules, Python wheels with C extensions, and Go binaries built with cgo can all behave differently on musl. Some prebuilt wheels won't load at all. The failure mode is usually a runtime error such as `Error loading shared library libc.musl-x86_64.so.1` or a segfault that doesn't reproduce on a glibc-based development machine.

If dependencies are pure (no native extensions), Alpine is a fine choice and the image is small. If there are native dependencies, a better fit is a slim variant of the same distribution used for building — for example `debian:bookworm-slim` or `python:3.12-slim`. Slim variants strip documentation, man pages, and compilers but keep glibc, so native modules keep working.

A comparison of the main options:

| Base | libc | Typical use | Watch out for |
|---|---|---|---|
| `alpine:3.20` | musl | Pure-language apps, static Go binaries | Native modules, DNS resolution differences |
| `debian:bookworm-slim` | glibc | Apps with native extensions | Still includes apt and a shell |
| `gcr.io/distroless/static-debian12` | none (static) | Go and Rust static binaries | No shell, no package manager |
| `gcr.io/distroless/base-debian12` | glibc | Apps needing libc but not a shell | Debugging is harder |

Distroless images, maintained by Google, contain only the runtime dependencies the app needs — no shell, no package manager, no `apt`. The `static` variant has no libc at all; the `base` variant includes glibc. There are language-specific variants (`java`, `python3`, `nodejs`) that bundle the runtime but still omit the shell.

## Step 3: Build a multi-stage Dockerfile that discards the build toolchain

The most reliable way to get a small runtime image is to build in one stage and copy only the artifacts into the final stage. Here is a Go example that produces a static binary and runs it on distroless:

```dockerfile
# syntax=docker/dockerfile:1
FROM golang:1.22-bookworm AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags='-s -w' -o /out/app ./cmd/app

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/app /app
USER nonroot:nonroot
ENTRYPOINT ["/app"]
```

Three things in that final stage matter. `CGO_ENABLED=0` produces a static binary so it doesn't need libc. `-ldflags='-s -w'` strips debug symbols, which reduces size and slightly reduces the information an attacker gets from the binary. `USER nonroot:nonroot` runs as a non-root user — the distroless `nonroot` tag includes that user, and running as root inside a container is a privilege escalation risk if the container is compromised.

For Python, the pattern is similar but the final stage uses `gcr.io/distroless/python3-debian12`. The virtualenv or site-packages are copied into the image. One gotcha: distroless Python expects the app to be importable; `pip install` cannot run at runtime because there is no shell or package manager. Install dependencies in the build stage and copy them.

For Node.js, use `gcr.io/distroless/nodejs20-debian12`. The entrypoint is `node`, so `CMD` should be the path to the script, not `npm start` — there is no npm in the image.

A subtlety worth planning for: the build stage and the runtime stage should agree on architecture and libc. Building with `GOOS=linux` on an ARM machine and deploying to x86 will fail at exec time, not build time. Pin `--platform` in the build command or in the Dockerfile when the CI runner and the target differ.

## Step 4: Fix the entrypoint-script problem

Distroless images have no shell. Any `ENTRYPOINT` or `CMD` that starts with `sh -c` or uses shell syntax will fail with an error like `exec: "/bin/sh": stat /bin/sh: no such file or directory`. This is the most common reason a distroless migration stalls.

The fix is to make the container run a single executable directly. If a start script currently sets environment variables, waits for a database, or runs migrations, move that logic into the application or into an init container. Instead of:

```dockerfile
CMD ["sh", "-c", "node server.js"]
```

use:

```dockerfile
CMD ["node", "server.js"]
```

If environment variable substitution is needed, do it at deploy time (the orchestrator can set env vars) rather than in a shell. If the app must wait for a dependency, use the platform's health-check or init-container mechanism — Kubernetes `initContainers` or a Compose `depends_on` with a healthcheck — rather than a shell loop.

A second common trap: some base images set `ENTRYPOINT` to a shell script. When switching to distroless, that entrypoint disappears. Always check the base image's Dockerfile or documentation for its default `ENTRYPOINT` and `CMD`, and override them explicitly.

## Step 5: Add a vulnerability scanner that fails the build

Scanning is only useful if it blocks. A scanner that prints a report but exits zero produces a dashboard nobody reads. Configure the scanner to fail on HIGH and CRITICAL severities, and set a policy for what counts as acceptable.

With Trivy in GitHub Actions:

```yaml
name: build-and-scan
on: [push]
jobs:
  scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build image
        run: docker build -t app:${{ github.sha }} .
      - name: Scan image
        uses: aquasecurity/trivy-action@0.24.0
        with:
          image-ref: app:${{ github.sha }}
          format: table
          exit-code: '1'
          ignore-unfixed: true
          severity: HIGH,CRITICAL
```

`ignore-unfixed: true` is a pragmatic default: it skips vulnerabilities that have no available fix, so the build doesn't fail on something that cannot be remediated. The trade-off is that those vulnerabilities are being accepted. Review the ignore list periodically and document why each is accepted.

A second scanner worth running is Grype, which uses a different vulnerability database and can catch things Trivy misses. Running two scanners is not paranoia — their databases update on different schedules and have different coverage. If only one is run, keep its database updated; a stale database is worse than no scanner because it gives false confidence.

The blocking policy also needs an owner and an expiry. A practical rule: an exception must name the CVE, the reason it is unreachable or unfixable, and a date to re-check. Exceptions without an expiry become permanent.

## Step 6: Verify the runtime, not just the build

A clean scan at build time doesn't guarantee a clean runtime. The image can be modified after the scan (by a compromised registry or a malicious layer), and the running container can have a different filesystem than the image if volumes are mounted.

Verify three things after deployment:

1. **The running image digest matches what was scanned.** In Kubernetes, `kubectl get pod <pod> -o jsonpath='{.status.containerStatuses[0].imageID}'` shows the digest. Compare it to the digest that was scanned.
2. **The process is not running as root.** `kubectl exec <pod> -- id` will fail on distroless because there is no `id` binary, so check the pod spec's `securityContext.runAsNonRoot: true` and `runAsUser`. If the pod spec doesn't set it, the image's `USER` directive may still apply, but explicit is better.
3. **No unexpected open ports or capabilities.** Use `kubectl debug` to inspect a distroless pod; don't add a shell to the production image just for debugging.

A useful verification step is to run the image locally with a read-only root filesystem and see if it still starts:

```bash
docker run --rm --read-only --tmpfs /tmp app:latest
```

If the app writes to disk outside `/tmp`, it will fail. That failure is informative: it tells you where the app expects to write and whether that should be a mounted volume instead. Many hardened deployments run with `readOnlyRootFilesystem: true` in Kubernetes, so catching this early avoids a production surprise.

## Edge cases and errors

**DNS resolution on Alpine.** musl's DNS resolver behaves differently from glibc's, particularly with search domains and `ndots`. A service that resolves fine on Debian can fail on Alpine with `no such host`. If that happens, check `/etc/resolv.conf` and consider switching to a glibc-based slim image.

**CA certificates.** Distroless `static` images include CA certificates, but an image built `FROM scratch` must copy them explicitly: `COPY --from=build /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/`. Without them, outbound HTTPS calls fail with `x509: certificate signed by unknown authority`.

**Timezones.** Distroless images include tzdata in the `base` and language variants, but `static` does not. If the app formats timestamps in a local timezone, either use UTC everywhere or copy tzdata into the image.

**Debugging without a shell.** `kubectl exec -it <pod> -- sh` will fail on distroless. Use `kubectl debug` to attach an ephemeral container with a shell and the same process namespace. This is the intended workflow, and it is safer than shipping a shell in the production image.

**Scanner false positives on distroless.** Distroless images sometimes report CVEs for packages present in the image metadata but not actually used at runtime. Check whether the vulnerable binary is reachable. If it isn't, document the exception rather than switching back to a larger base image just to make the scanner quiet.

**Silent failures at startup.** A distroless container that crashes immediately produces almost no diagnostic output. If the app logs to stdout, capture it. If the entrypoint is wrong, the error is usually a single line from the runtime (`exec: ... no such file or directory`) before exit. Run the image locally with `docker run --rm <image>` before pushing; the failure is much cheaper to diagnose there than in a cluster.

## Frequently Asked Questions

**Does distroless mean no package manager in production?**
Correct. Distroless images have no `apt`, `apk`, or `yum`. All dependencies must be installed in the build stage and copied into the final image. This is the point: it removes a common attacker tool and reduces image size. If something must be installed at runtime, distroless is the wrong base for that container.

**Is Alpine more secure than Debian slim?**
Not inherently. Alpine has a smaller default package set, which reduces attack surface, but musl compatibility issues can force extra packages or workarounds for native modules. The more important factor is how often the image is rebuilt and rescanned. A Debian slim image rebuilt weekly with a scanner in CI is usually in better shape than an Alpine image that hasn't been rebuilt in six months.

**How should vulnerabilities with no fix be handled?**
Most scanners can ignore unfixed vulnerabilities or add them to an allowlist with an expiry date. The policy should be explicit: accept the risk, document why, and set a reminder to re-check when a fix is released. Silently ignoring them is how a known CVE stays in production for years.

**Can distroless be used with a language that needs a JIT compiler?**
Yes, for the languages that have distroless variants. The `java` variant includes a JRE, and the `nodejs` variant includes Node.js. The constraint is the same: no shell, no package manager. If the app spawns subprocesses that expect a shell, those will fail; rewrite them to call executables directly.

**How much smaller is a distroless image?**
It depends entirely on the app and its dependencies, so measure rather than assume. Compare `docker image ls` for the old and new tags, and compare the package counts from Step 1. The more useful comparison is the vulnerability delta from the scanner, since that is what the hardening is for.

## The next 30 minutes

Open the main service's Dockerfile and change the final stage to a distroless base that matches the runtime — `gcr.io/distroless/static-debian12:nonroot` for a static Go binary, or the appropriate language variant otherwise. Then run `trivy image --severity HIGH,CRITICAL <your-image>` against both the old and new image and compare the counts. If the new image fails to start, check the entrypoint first: a `sh -c` command is the most likely cause.
