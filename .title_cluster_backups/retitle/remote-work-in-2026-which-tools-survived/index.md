# Remote work in 2026: which tools survived

Most remote-work tooling assumes a developer on fiber in a major hub. When a team spans high-latency regions, metered connections, and grids that drop for hours, that assumption becomes the failure mode. A chat client that hangs on a 250 ms socket timeout, a CI runner that re-downloads a 1 GB image on every job, a deploy step with no retry — each is individually small and collectively fatal to throughput.

This article covers how to measure your actual constraints, then build a stack that degrades gracefully: a self-hosted CI runner, an async chat bridge, retry and power-failure handling, and enough observability to debug without a live call. Everything here is reproducible with commands you can run yourself.

## Measure first, then choose

Before changing tooling, quantify the problem. Latency to your source host and package registries drives CI flakiness more than raw bandwidth does.

```bash
for i in $(seq 1 10); do
  curl -s -o /dev/null -w "%{time_total}\n" https://api.github.com/users/octocat
done
```

Collect ten samples and compare the median, not the best case. A median above roughly 300 ms to your primary Git host means every CI step that makes many small API calls inherits that penalty multiplied by call count. Also measure:

```bash
# DNS resolution time alone
dig api.github.com | grep "Query time"

# Time to first byte from a container registry
curl -s -o /dev/null -w "connect=%{time_connect} ttfb=%{time_starttransfer}\n" \
  https://registry-1.docker.io/v2/
```

Record these numbers before and after any change. Without a baseline, "it feels faster" is not evidence, and vendor claims about latency are not your latency.

### Runner placement options

| Option | Typical monthly cost | Failure domain | Suits |
|---|---|---|---|
| Cloud-hosted runners | Usage-based, can be high | Provider outage | Teams with reliable local links and card billing |
| Self-hosted VPS in a nearby region | Low, fixed | Provider outage | Teams needing predictable cost and low latency to the repo host |
| Local hardware (SBC or mini-PC) | Hardware + power | Power and ISP | Teams with solar/battery and a local ISP |

The tradeoff is control versus operational burden. A VPS gives you a stable IP, IPv6, and a provider SLA. Local hardware gives you the lowest latency and survives WAN outages but adds power, cooling, and disk-failure risk to your list.

## Set up the base environment

Install the common dependencies on the host:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y docker.io docker-compose-plugin git jq build-essential
```

Install a current Node LTS via NodeSource (the exact patch version will move; check with `node --version` after install):

```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
node --version
```

### Register a self-hosted runner

Download the runner archive matching your architecture. The example below is arm64; substitute `x64` for Intel/AMD hosts, and verify the current release tag on the runner's releases page rather than hardcoding one.

```bash
mkdir ~/actions-runner && cd ~/actions-runner
curl -o actions-runner.tar.gz -L \
  https://github.com/actions/runner/releases/download/v2.317.0/actions-runner-linux-arm64-2.317.0.tar.gz
tar xzf ./actions-runner.tar.gz
```

Register it with a token scoped to the repository (use the token GitHub generates for runner registration, not a broad personal access token):

```bash
./config.sh --url https://github.com/your-org/your-repo \
  --token YOUR_REGISTRATION_TOKEN \
  --name "runner-01"
```

Install and start as a service:

```bash
sudo ./svc.sh install
sudo ./svc.sh start
```

A common failure: the runner tries to pull `node:20` and stalls on a slow link. Pre-pull images explicitly and pin digests for reproducibility:

```bash
docker pull node:20-alpine
docker image inspect node:20-alpine --format '{{index .RepoDigests 0}}'
```

Pin the digest in your workflow so a re-tag upstream cannot silently change your build.

## Build the async chat bridge

The point of an async bridge is that GitHub events land in a chat room as messages, and nobody has to attend a meeting to learn a build failed. Matrix is one option: a federated, open protocol with self-hostable servers. The same pattern works with any chat system that exposes an HTTP send endpoint.

The bridge's job is narrow: receive a webhook, format a message, POST it to the room. Keep it that small. A bridge that also tries to parse commits and run commands becomes a second, worse CI system.

### Config shape

```json
{
  "bridge": {
    "name": "GitHub Bridge",
    "matrix_host": "http://localhost:8008",
    "matrix_user": "@github-bridge:yourdomain.com",
    "matrix_password": "YOUR_PASSWORD",
    "github_token": "ghp_YOUR_TOKEN",
    "rooms": {
      "!github-updates:yourdomain.com": {
        "repo": "your-org/your-repo"
      }
    }
  }
}
```

Never commit this file. Load secrets from environment variables or a secrets manager, and mount the config read-only into the container.

### Workflow that posts status

```yaml
name: Async Push
on:
  push:
    branches: [main]
jobs:
  build:
    runs-on: ["self-hosted", "linux", "arm64"]
    steps:
      - uses: actions/checkout@v4
      - name: Build
        run: |
          npm ci
          npm run build
      - name: Notify chat
        if: always()
        run: |
          curl -sS -X POST -H "Content-Type: application/json" \
            -d "{\"text\":\"Build ${{ job.status }} for ${{ github.sha }}\"}" \
            "${{ secrets.MATRIX_ROOM_URL }}"
      - name: Deploy
        if: success()
        run: ./scripts/deploy.sh
```

Two details matter. `if: always()` ensures the notification fires on failure — the case you most need to hear about. And the room URL lives in a secret, not in the workflow file, so rotating it does not require a code change.

A failure mode worth naming: pointing the notification at a public federation endpoint from inside a corporate network often fails because outbound traffic to that host is blocked. Send to your own homeserver's client endpoint instead, and confirm with a single manual `curl` before wiring it into CI.

## Handle the failure cases as first-class

In a constrained-network environment, power cuts, throttling, and transient network errors are the normal path, not exceptions. Build for them directly.

### Power and UPS state

```bash
#!/bin/bash
# ups-check.sh
STATUS=$(apcaccess status | awk '/^STATUS/ {print $3}')
if [ "$STATUS" != "ONLINE" ]; then
  curl -sS -X POST -H "Content-Type: application/json" \
    -d "{\"text\":\"Power event. UPS status: ${STATUS}\"}" \
    "$MATRIX_ROOM_URL"
fi
```

Install the daemon that provides `apcaccess`:

```bash
sudo apt install -y apcupsd
```

A real trap: some UPS units report `STATUS ONLINE` while running on a nearly depleted battery during a brownout. Status alone is not enough — also read the battery charge field and alert on a low threshold, not just on a status change.

For battery/solar hosts without a UPS, check thermal throttling on a Raspberry Pi:

```bash
ssh pi@local-pi "vcgencmd get_throttled"
```

A non-zero value indicates the board has throttled due to undervoltage or heat — both common when running CI on solar power.

### Local registry cache

If pulling images from a public registry is slow or throttled, run a pull-through cache:

```bash
sudo apt install -y docker-registry
sudo systemctl enable --now docker-registry
```

Configure Docker to use it:

```json
{
  "registry-mirrors": ["http://localhost:5000"]
}
```

```bash
sudo systemctl restart docker
```

Measure the effect rather than assuming it: time `docker pull node:20-alpine` before and after, with the image removed between runs (`docker rmi node:20-alpine`). The cache only helps on repeat pulls; the first pull still crosses the WAN.

### Retry with backoff

```yaml
- name: Deploy with retry
  if: success()
  run: |
    MAX_RETRIES=3
    RETRY_DELAY=5
    for i in $(seq 1 $MAX_RETRIES); do
      if ./scripts/deploy.sh; then
        echo "Deploy succeeded"
        exit 0
      fi
      echo "Deploy failed, retry $i/$MAX_RETRIES in ${RETRY_DELAY}s"
      sleep $RETRY_DELAY
      RETRY_DELAY=$((RETRY_DELAY * 2))
    done
    echo "Max retries reached"
    exit 1
```

To know whether this helps, instrument the deploy step to emit an outcome label (success, retry-then-success, failed) to your metrics backend. Compare the failure rate over a week before and after enabling retries. Do not trust a single anecdotal improvement.

## Observability you can actually debug with

You cannot fix what you cannot see, and on an unreliable network you will be debugging remotely.

### Metrics

Run a Prometheus-compatible time-series database and a node exporter on the runner host. Scrape the exporter on its default port and store the data locally:

```bash
# node exporter exposes host metrics on :9100
./node_exporter &
```

Point your metrics database at `localhost:9100` as a scrape target, then build panels for:

- Runner CPU, memory, and disk I/O
- Chat bridge process uptime and restart count
- CI queue depth and job duration percentiles
- Deploy step outcome counts

Watch disk I/O specifically: on a single-board computer with an SD card, CI workloads can saturate the card and cause both slowness and early failure. An M.2 SSD avoids this.

### Logs

Ship logs to a local aggregator and query them by repo, job, and error string. The value is being able to answer "what did the deploy step print at 03:14 when the link dropped?" without SSH-ing into a machine that may be offline.

### Tests for async behavior

```javascript
// test/bridge.spec.js
const assert = require('node:assert');
const axios = require('axios');

(async () => {
  const start = Date.now();
  await axios.post('http://localhost:3000/webhook', { text: 'test', delay: 500 });
  const elapsed = Date.now() - start;
  assert.ok(elapsed <= 1000, `bridge took ${elapsed}ms`);
})();
```

Run it with the Node test runner or any test framework you already use:

```bash
node test/bridge.spec.js
```

The point of the test is a bound, not a precise number: the bridge must respond within a stated budget even when a downstream call is slow. Set the budget from your measured latency, not from a guess.

## A worked sizing example

Suppose a team wants a runner that survives a WAN outage and keeps cost predictable. Reason through it:

1. Build peak memory is about 1.5 GB (Node build plus Docker overhead).
2. The chat bridge and metrics agent together use about 200 MB.
3. Add headroom of 50% for spikes: 1.7 GB × 1.5 ≈ 2.6 GB.
4. A host with 4 GB RAM and 2 vCPU covers this with margin.
5. Storage: the OS plus two pinned images plus build cache. Measure with `docker system df` after a week of real jobs, then size the disk to twice that figure.

These numbers are illustrative — substitute your own measurements. The method is what matters: measure peak usage, add explicit headroom, then size the machine. Do not size from a blog post's numbers, including these.

## Decision checklist before you migrate

- Baseline latency to your Git host and registry is recorded, with ten samples and a median.
- You know your peak build memory and disk usage from real jobs.
- Secrets for the bridge and deploy live outside the repository.
- The deploy step has retries with backoff and emits an outcome metric.
- Notifications fire on failure, not only on success.
- You can read logs for a job that ran while you were offline.
- You have a documented plan for a multi-hour power or WAN outage: what queues, what fails loudly, what fails silently.

If any line is unchecked, fix that before adding more tooling. Adding an AI triage layer on top of a stack you cannot observe just moves the mystery one layer up.

## FAQ

**Should the runner be local or on a VPS?**
Local hardware wins on latency and survives WAN outages; a VPS wins on operational stability and a provider SLA. If your main pain is flaky CI due to latency, local or nearby-region hosting helps. If your pain is a machine that dies when the power does, a VPS in a stable region is the safer default.

**How do I know if the chat bridge is the problem or the network is?**
Log the time spent inside the bridge separately from the time spent on the outbound request. If the bridge's own processing is fast and the outbound call is slow, it is the network. If both are slow, the bridge is overloaded. One timing log per request answers this.

**Do I need a full metrics stack for a small team?**
No. A single node exporter plus a lightweight time-series database is enough to see CPU, memory, disk, and job duration. Add log aggregation when you find yourself SSH-ing to read logs during an incident.

**What is the most common silent failure?**
A notification that fires only on success, so a failed build produces no signal and the team discovers it hours later. Always send on failure.

## Do this in the next 30 minutes

Run the latency loop from the first section ten times against your primary Git host and write down the median. If it exceeds 300 ms, open an issue in your repo titled "CI latency baseline" with the number and a link to the ten raw samples. That single recorded measurement turns every later tooling argument into a comparison against evidence.
===END===
