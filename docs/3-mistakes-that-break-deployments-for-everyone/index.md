# 3 mistakes that break deployments for everyone

Deployment tooling is usually chosen by the person who understands the stack best, then handed to people who do not. The result is a pipeline that is fast for its author and hostile to everyone else. The failure is rarely a single bad tool choice; it is a mismatch between the defaults that suit an expert and the guardrails a newcomer needs. This article covers three recurring mistakes, what each one actually breaks, and how to fix them without adding a second platform to maintain.

## The tension you are actually managing

Two people can share one repository and have opposite requirements for the same pipeline.

One is the engineer who wrote most of the system. They want fast feedback, direct access to logs, the ability to run arbitrary commands against the environment, and a rollback path they can trigger in seconds. Every abstraction between them and the running system is friction.

The other is someone who joined recently, possibly without deep infrastructure experience. They need the pipeline to refuse dangerous actions, to explain what it did, and to fail loudly and specifically when something is wrong. Every implicit assumption is a trap.

A pipeline tuned only for the first person accumulates hidden state: a runner with a warm cache, a locally installed CLI, a database container that was started manually months ago. None of that is written down, so none of it survives contact with a new contributor. A pipeline tuned only for the second person adds so many gates that the senior engineer routes around it, and the guardrails stop mattering.

The three mistakes below are the most common ways this tension turns into an outage or a stalled onboarding.

## Mistake 1: mutable build hosts

A self-hosted runner, a long-lived CI VM, or a build machine that persists between jobs is convenient. It keeps dependency caches warm, avoids re-pulling base images, and makes builds faster on the second run. It also accumulates state that nobody tracks.

### What breaks

The classic failure mode is disk exhaustion. A runner that has been alive for weeks fills its cache volume during a large dependency update and then hangs on the next job rather than failing fast. The job sits in a queued or running state until someone notices. Meanwhile every subsequent job is blocked behind it.

Less obvious is version drift. If the build host has a toolchain installed at the OS level, the version of that toolchain is now part of your build definition, but it is not in the repository. A new hire who builds locally gets a different result than CI. The failure surfaces as a confusing "works on my machine" bug that costs hours to trace.

### The fix

Make build hosts disposable. Whether you use ephemeral cloud instances, container-based runners, or a hosted CI service, the property that matters is that each job starts from a known image and leaves nothing behind. Caches should be explicit artifacts keyed by lockfile hash, not incidental state on a disk.

If you run self-hosted runners, the standard pattern is a controller that creates a runner per job and destroys it afterward. The runner image is versioned in Git, so the toolchain is part of the repository rather than a property of a machine.

### How to measure whether you have this problem

Instrument two things: the age of your build hosts and the failure mode of the last ten failed jobs. If any host has been running for more than a few days, or if any failure was a hang rather than an error, you have mutable-host problems. A simple check on a Linux runner:

```bash
uptime -p
df -h /var/lib/docker
```

If the uptime is measured in weeks and the disk is above 80 percent, the next large dependency update is a coin flip.

## Mistake 2: implicit local prerequisites

The pipeline assumes the developer's machine is already set up. The README says "clone and run," but the actual requirements are a specific Docker version, a cloud CLI, credentials in a particular file, and a database container that was started by hand.

### What breaks

New contributors fail at step zero, before they can even reproduce the failure they are trying to fix. The failure is not interesting, so it does not get fixed; it just becomes a tax on every new hire. Senior engineers never see it because their machines have been configured for years.

A related failure mode is that the setup instructions drift from reality. Someone adds a dependency on a new CLI tool, updates their own machine, and forgets the README. The instructions are now wrong in a way that only a newcomer can detect.

### The fix

Put the environment in the repository. A container definition that describes the build and run environment, plus a single setup script that installs the host-level prerequisites, removes most of the ambiguity. The key is that the setup script is idempotent and checks for what it needs rather than assuming.

Here is a minimal container definition for a Python service:

```dockerfile
FROM python:3.11-slim-bookworm
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "main.py"]
```

And a setup script that a new hire can run without reading it first:

```bash
#!/usr/bin/env bash
set -euo pipefail

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker not found. Install it from https://docs.docker.com/engine/install/ and re-run."
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "Docker is installed but the daemon is not reachable. Start Docker and re-run."
  exit 1
fi

echo "Environment looks ready."
```

The script does not try to install Docker for the user, because package managers differ and a script that guesses wrong is worse than no script. It checks and tells the user exactly what to do.

### How to measure whether you have this problem

Ask someone who has never touched the repository to set it up while you watch, without helping. Time how long it takes to get a successful local build. If it takes more than thirty minutes, or if you had to intervene more than once, the prerequisites are implicit.

## Mistake 3: opaque failures

When a deployment fails, the pipeline reports something generic. "Build failed." "Application error." "Health check failed." The actual cause is buried in a log the new hire does not know how to find.

### What breaks

Debugging time explodes. A failure that should take five minutes to diagnose takes thirty because the error message does not name the component that failed. New hires learn to fear the pipeline, which is the opposite of what you want.

The deeper problem is that opaque failures hide real defects in the pipeline itself. If a health check fails without saying which check failed or what it received, you cannot tell whether the application is broken or the check is misconfigured. That ambiguity is where multi-hour incidents come from.

### The fix

Make every failure name its cause. A health check should log the endpoint it called, the status code or error it received, and the timeout it used. A build failure should print the failing command and its exit code. A deploy failure should distinguish between "the new version did not become healthy" and "the deploy command itself failed."

This is mostly a matter of not swallowing errors. A shell script with `set -e` and no output on failure is a common culprit. So is a CI step that captures output into a variable and never prints it.

### How to measure whether you have this problem

Take the last five failed deployments and read only the top-level error message shown in the CI UI. If you cannot tell from that message alone which component failed and roughly why, the failures are opaque.

## A worked example: rollback that actually works

Rollback is where the three mistakes compound. If build hosts are mutable, you may not be able to rebuild the previous version. If prerequisites are implicit, the person doing the rollback may not have the right CLI. If failures are opaque, you will not know whether the rollback succeeded.

The reliable pattern is to make every deployed artifact addressable by an immutable identifier, and to keep the last few around. Tag images with the Git commit SHA, and reference that tag in the deploy step.

```bash
# Build and tag with the commit SHA
SHA=$(git rev-parse --short HEAD)
docker build -t myapp:${SHA} .
docker tag myapp:${SHA} 123456789012.dkr.ecr.us-east-1.amazonaws.com/myapp:${SHA}
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/myapp:${SHA}
```

To roll back, redeploy the previous SHA. The exact command depends on your runtime; the property that matters is that the deploy step takes an image reference as input rather than always building from the current branch.

```bash
# Redeploy a specific image tag
aws ecs update-service \
  --cluster myapp-cluster \
  --service myapp-service \
  --force-new-deployment \
  --task-definition myapp:42
```

Two things make this work. First, the image registry must retain old tags for long enough to matter; a lifecycle policy that keeps the last ten tags is usually enough. Second, the deploy step must be able to run without a build, so a broken build pipeline does not block a rollback.

### Why this is worth the setup cost

A rollback that takes thirty seconds changes how a team behaves. People deploy more often because the cost of a mistake is low. That in turn makes each change smaller, which makes failures easier to diagnose. The investment is a few hours of pipeline work; the return is a different deployment culture.

## Choosing defaults: a decision checklist

When you are setting up or revising a pipeline, work through these questions in order. The answers usually point to one option.

1. **Who is the least experienced person who will deploy?** If the answer is "someone who has never used a CLI," your pipeline needs a UI or a single command that does everything.
2. **What happens if the build host is destroyed right now?** If the answer is "we lose the cache and it takes an extra two minutes," you are fine. If the answer is "we cannot rebuild," you have a problem.
3. **How long does setup take on a clean machine?** Measure it. Do not estimate.
4. **What does a failed deploy look like in the CI UI?** If it is one line with no component name, fix that before adding any other tooling.
5. **Can you roll back without building?** If not, add image tagging by commit SHA and a deploy step that accepts an image reference.
6. **What is the blast radius of a bad deploy?** If it is the whole production environment, add a staging step. If staging is already there, make sure a new hire can deploy to it without help.
7. **How much does this cost at your current deploy frequency?** Compute it from your actual numbers rather than a vendor's marketing page.

The last question deserves a note. Deployment costs are usually dominated by the compute that runs your application, not by the CI minutes. Optimizing CI cost before you have fixed the onboarding and rollback problems is usually a distraction.

## Failure modes to watch for after you fix the obvious ones

Once build hosts are disposable, prerequisites are explicit, and failures name their cause, a few subtler problems tend to surface.

**Cache poisoning.** An explicit cache keyed by lockfile hash is safe. A cache keyed by branch name is not, because two branches can produce incompatible artifacts. Key caches on the inputs that determine their contents.

**Secret sprawl.** Making setup easy often means making credentials easy to obtain. Keep the number of places a secret can live as small as possible, and make the pipeline fetch secrets at runtime rather than storing them in the repository or on the build host.

**Health checks that pass too easily.** A health check that returns 200 as soon as the process starts will pass before the application is ready to serve traffic. The check should exercise the dependency the application actually needs, such as a database connection, and fail if that dependency is unavailable.

**Environments that drift.** If staging and production are configured differently, a deploy that passes staging can fail in production for reasons unrelated to the code. Keep the configuration difference to a small, explicit set of variables.

## FAQ

### Should the pipeline build from a Dockerfile or use a buildpack?

Use a Dockerfile when you need control over the system libraries, the base image, or the runtime version. Use a buildpack when you want the platform to manage those choices and you are willing to accept its defaults. The tradeoff is control versus convenience, not correctness. If you choose a buildpack, pin the language version explicitly so an upstream default change does not silently alter your runtime.

### How many environments do you need?

Two is the minimum for anything with users: one that resembles production and one that is production. A third, ephemeral environment per pull request is useful for teams that deploy frequently, but it adds cost and configuration surface. Add it when the cost of a bad merge exceeds the cost of running it.

### What if the senior engineer refuses to use the pipeline?

That is a signal, not a personality problem. Ask what the pipeline makes slower or more annoying. Common answers are "it takes too long to get logs" and "I cannot run a one-off command against the environment." Both are fixable without removing the guardrails that protect everyone else.

### How do you keep the README from going stale?

Make the setup script the source of truth and have the README describe what the script does rather than duplicating the commands. If the script is the only place the commands live, it cannot drift from itself.

### Is a self-hosted runner worth it?

It depends on whether you need capabilities a hosted runner does not provide, such as a specific architecture, a private network, or a larger machine type. If you do not have such a requirement, hosted runners remove an entire class of maintenance work. If you do run self-hosted runners, make them ephemeral.

## What to do in the next thirty minutes

Pick the last failed deployment in your CI system and read only the top-level error message. If it does not tell you which component failed and roughly why, rewrite that error message to include the failing step, the command it ran, and its exit code. That single change is the highest-leverage fix available, because it makes every future failure cheaper to diagnose.
