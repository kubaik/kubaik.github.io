# Kernel and EBS Tuning That Cuts EC2 CPU Overhead

Most cost-reduction guides stop at instance-type selection and spot fleets. Those levers are real, but they sit above a layer that many teams never inspect: the kernel networking stack and the storage configuration underneath the application. This article covers what can be tuned below the application layer, how to measure whether a change actually helped, and where the common failure modes are.

The techniques here apply to any Linux workload on EC2 — Node, Python, Go, or JVM — because they operate at the sysctl and block-device level. Nothing requires recompiling a kernel or building a custom AMI.

## Prerequisites and what you'll build

You need an AWS account, an EC2 instance running a modern Linux distribution (Amazon Linux 2023 or Ubuntu 22.04 LTS are both reasonable), and SSH access. The commands assume a root-capable user or `sudo`.

By the end you will have:
- A TCP congestion-control change applied and persisted via `/etc/sysctl.d`.
- A compressed swap configuration that reduces the chance of OOM kills during RSS spikes.
- A gp3 volume provisioned above its default IOPS, with a documented method for verifying the change.
- A measurement harness (iperf3, node_exporter, k6) that lets you decide whether any of this helped.

A note on expectations: the size of the improvement depends heavily on your workload's traffic shape, connection lifetime, and I/O profile. Long-lived HTTP/2 connections and bulk transfers tend to benefit most from congestion-control changes. Request/response APIs with small payloads may see almost nothing. Measure before you conclude.

## Step 1 — set up the environment

1. SSH into the instance.

```bash
ssh -i ~/.ssh/prod-key.pem ec2-user@<instance-ip>
```

2. Confirm the OS and kernel.

```bash
cat /etc/os-release
uname -a
# Example: Amazon Linux 2023, kernel 6.1.x
```

3. Install the measurement tools you'll need.

```bash
sudo dnf install -y iperf3 bcc-tools jq
# On Debian/Ubuntu: sudo apt install -y iperf3 bpfcc-tools jq
```

4. Baseline the network before any tweaks. Run an iperf3 server on the instance and a client from another host in the same AZ. Record the retransmit count.

```bash
# On the server:
iperf3 -s -p 5201

# On the client:
iperf3 -c <SERVER_IP> -p 5201 -t 60 -R --logfile baseline.json
```

5. Parse the JSON to extract retransmits.

```bash
jq '.end.sum_retransmits' baseline.json
```

Write the number down. It is the only meaningful comparison point you have.

6. Check the EBS baseline. gp3 volumes default to 3,000 IOPS and 125 MiB/s throughput, independent of volume size. (gp2 volumes, by contrast, scale IOPS with capacity at 3 IOPS/GiB.) Confirm the current setting with the AWS CLI.

```bash
aws ec2 describe-volumes \
  --volume-ids vol-0abcdef1234567890 \
  --query 'Volumes[0].[VolumeType,Iops,Throughput]'
```

If the volume is gp2, the query returns no IOPS field; gp2's performance is derived from size.

## Step 2 — core implementation

### TCP congestion control

The Linux kernel supports pluggable congestion-control algorithms. The two most commonly discussed are CUBIC (the long-standing default) and BBR (Bottleneck Bandwidth and Round-trip propagation time), which models the bottleneck link rather than reacting to packet loss.

On the relevant kernels, BBR is compiled in but not selected by default. Check the current state:

```bash
sysctl net.ipv4.tcp_congestion_control
# net.ipv4.tcp_congestion_control = cubic

sysctl net.core.default_qdisc
# net.core.default_qdisc = fq_codel
```

List the algorithms the running kernel actually offers:

```bash
sysctl net.ipv4.tcp_available_congestion_control
```

If `bbr` does not appear in that list, the module is not available and the rest of this section will not work on that kernel. Do not assume; check.

If it is available, apply it:

```bash
sudo sysctl -w net.core.default_qdisc=fq
sudo sysctl -w net.ipv4.tcp_congestion_control=bbr
```

Note the pairing: BBR is designed to work with the `fq` queueing discipline, not `fq_codel`. Setting `net.core.default_qdisc=bbr` is a common mistake — `bbr` is a congestion-control algorithm, not a qdisc, and the kernel will reject it.

Persist the change:

```bash
sudo tee /etc/sysctl.d/10-bbr.conf > /dev/null <<'EOF'
net.core.default_qdisc=fq
net.ipv4.tcp_congestion_control=bbr
EOF
sudo sysctl --system
```

Re-run the iperf3 test and compare:

```bash
iperf3 -c <SERVER_IP> -p 5201 -t 60 -R --logfile after.json
jq '.end.sum_retransmits' after.json
```

If the retransmit count did not move, the change is not helping your traffic pattern. That is a valid result, and the honest response is to revert it rather than keep a change that adds complexity for no benefit.

### Memory: swap and compressed swap

EC2 instances with no swap configured will invoke the OOM killer when memory pressure spikes. Whether that is acceptable depends on the workload. For services with occasional RSS spikes — JVM heaps, Node processes with large buffers, Python workers loading large datasets — a swap file paired with a compressed cache can turn a hard kill into a slowdown.

Two mechanisms are commonly confused:

- **zram** creates a compressed block device in RAM and uses it as swap. Fast, but consumes RAM.
- **zswap** is a compressed cache in front of a real swap device. Pages are compressed before being written out; only pages that cannot be compressed effectively reach the disk.

The two are alternatives, not complements. Pick one.

A zram setup using `zram-generator` (available on Amazon Linux 2023 and recent Fedora/RHEL derivatives):

```bash
sudo dnf install -y zram-generator
sudo tee /etc/systemd/zram-generator.conf > /dev/null <<'EOF'
[zram0]
zram-size = min(ram / 4, 4096)
compression-algorithm = zstd
EOF
sudo systemctl daemon-reload
sudo systemctl start systemd-zram-setup@zram0.service
```

This sizes the zram device at one quarter of RAM, capped at 4 GiB, and uses zstd. Verify:

```bash
swapon --show
zramctl
```

If you prefer a conventional swap file instead:

```bash
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo "/swapfile none swap sw 0 0" | sudo tee -a /etc/fstab
```

Tune swappiness. The default of 60 is aggressive for a server. A value of 10 tells the kernel to prefer reclaiming page cache over swapping anonymous memory, which is usually the right trade-off for a service:

```bash
sudo sysctl -w vm.swappiness=10
echo "vm.swappiness=10" | sudo tee /etc/sysctl.d/10-swap.conf
```

### EBS: raising gp3 IOPS

gp3 defaults to 3,000 IOPS and 125 MiB/s, and can be provisioned up to 16,000 IOPS and 1,000 MiB/s independently of volume size. Unlike gp2, you pay separately for provisioned IOPS above the baseline.

Before raising IOPS, estimate what the workload actually needs. A worked example:

- Assume 800 requests per second.
- Assume each request reads an 8 KiB object from disk (this is a strong assumption — check your actual I/O pattern with `iostat -x 1` before trusting it).
- 800 × 8 KiB = 6,400 KiB/s = 6.25 MiB/s.
- If each of those reads is a separate 8 KiB I/O operation, that is roughly 800 IOPS, well under the 3,000 gp3 baseline.

That arithmetic suggests the default is already sufficient for this hypothetical workload. The reason to provision more IOPS is not request rate but burst behavior: a periodic job, a cache miss storm, or a batch process that issues thousands of small random reads in a short window. Measure with `iostat` and CloudWatch `VolumeReadOps`/`VolumeWriteOps` before paying for headroom you do not use.

To raise IOPS on a gp3 volume, no detach is required. `modify-volume` works on attached volumes and the change takes effect after a short optimization window:

```bash
aws ec2 modify-volume \
  --volume-id vol-0abcdef1234567890 \
  --iops 6000 \
  --throughput 250
```

Check the modification state:

```bash
aws ec2 describe-volumes-modifications \
  --volume-ids vol-0abcdef1234567890
```

The volume remains mounted and usable throughout. The `modifying` state can persist for hours on large volumes; it does not block I/O.

A caution on the arithmetic above: if you are using gp2, IOPS are not directly settable. You either resize the volume (3 IOPS per GiB) or migrate to gp3. Migration is done by snapshot-and-restore or by `modify-volume --volume-type gp3`, which is online.

## Step 3 — edge cases and failure modes

### BBR is not available on the running kernel

The algorithm name is `bbr`, not `bbr2`. Some older documentation and forum posts refer to a `bbr2` variant; that was a development branch and is not a stable kernel interface. If `bbr` is not in `net.ipv4.tcp_available_congestion_control`, the module is not present.

Detect and fail cleanly rather than silently applying a no-op:

```bash
if sysctl net.ipv4.tcp_available_congestion_control | grep -qw bbr; then
  sudo sysctl -w net.core.default_qdisc=fq
  sudo sysctl -w net.ipv4.tcp_congestion_control=bbr
else
  echo "BBR not available on this kernel: $(uname -r)" >&2
  exit 1
fi
```

### Setting the qdisc to `bbr`

As noted above, `net.core.default_qdisc=bbr` is invalid. The kernel accepts the write but the qdisc is not applied, and the congestion control falls back. Verify after applying:

```bash
sysctl net.core.default_qdisc
sysctl net.ipv4.tcp_congestion_control
```

If the first returns anything other than `fq` (or `fq_codel`, which also works acceptably with BBR in practice), the configuration is wrong.

### zram and swappiness interaction

If you use zram for swap, a high `vm.swappiness` will cause the kernel to push anonymous pages into the compressed device aggressively. On a memory-constrained instance with a CPU-bound workload, the compression cost can show up as a latency regression. Monitor CPU utilization after enabling zram. If it rises without a corresponding drop in OOM events, the configuration is not paying for itself.

A reasonable starting point for zram-backed swap on a server is `vm.swappiness=100` (the kernel treats zram as fast) or a conventional value like 60 if you want to be conservative. The value of 10 that is appropriate for disk-backed swap is often wrong for zram. Test both.

### EBS modification throttling

`modify-volume` is rate-limited per volume. Repeated modifications (for example, from a CI job that runs on every deploy) will be rejected. If you manage volumes declaratively, ensure the tooling is idempotent: read the current IOPS first and skip the call if it already matches.

```bash
current=$(aws ec2 describe-volumes \
  --volume-ids vol-0abcdef1234567890 \
  --query 'Volumes[0].Iops' --output text)
if [ "$current" != "6000" ]; then
  aws ec2 modify-volume --volume-id vol-0abcdef1234567890 --iops 6000
fi
```

## Step 4 — observability and verification

### CloudWatch

A minimal dashboard for validating these changes:

- CPU utilization, 1-minute period.
- `EBSReadOps` and `EBSWriteOps`, 1-minute period.
- `NetworkPacketsIn`/`NetworkPacketsOut` to correlate with retransmit changes.

CloudWatch does not expose TCP retransmits directly. The kernel does, via `/proc/net/snmp`, and the standard way to export it is `node_exporter`.

### node_exporter

Download the current release for your architecture from the Prometheus project's releases page, extract it, and run it as a systemd unit:

```bash
tar xf node_exporter-*.linux-amd64.tar.gz
cd node_exporter-*.linux-amd64
sudo install -m 0755 node_exporter /usr/local/bin/
sudo tee /etc/systemd/system/node_exporter.service > /dev/null <<'EOF'
[Unit]
Description=Prometheus node_exporter
After=network.target

[Service]
User=nobody
ExecStart=/usr/local/bin/node_exporter
Restart=always

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl enable --now node_exporter
```

Scrape it from Prometheus:

```yaml
scrape_configs:
  - job_name: 'ec2'
    static_configs:
      - targets: ['<instance-ip>:9100']
```

The retransmit metric is `node_netstat_Tcp_RetransSegs`. A ratio against total segments sent gives a useful signal:

```yaml
- alert: HighRetransmitRatio
  expr: |
    rate(node_netstat_Tcp_RetransSegs[5m])
    / rate(node_netstat_Tcp_OutSegs[5m]) > 0.02
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "Elevated TCP retransmit ratio on {{ $labels.instance }}"
```

The 2% threshold is a starting point, not a universal constant. Establish your own baseline first.

### Load test with k6

```javascript
// load-test.js
import http from 'k6/http';
import { check } from 'k6';

export const options = {
  vus: 100,
  duration: '3m',
};

export default function () {
  const res = http.get('http://localhost:3000/api/status');
  check(res, {
    'status is 200': (r) => r.status === 200,
    'latency < 100 ms': (r) => r.timings.duration < 100,
  });
}
```

Run this before and after each change. Compare p50, p95, and p99, not just the mean. Congestion-control changes often shift the tail rather than the median.

## How to know whether any of this worked

The honest answer is that you cannot know without measuring on your own workload. The following table describes what to instrument and what a meaningful change looks like, rather than asserting results:

| Change | What to measure | Where | What a real improvement looks like |
|---|---|---|---|
| TCP congestion control | Retransmit ratio; p95 latency on bulk transfers | `node_netstat_Tcp_RetransSegs`, k6 | Retransmit ratio falls; p95 improves on long-lived connections |
| zram or swap | OOM kill count; CPU utilization; swap-in rate | `dmesg`, CloudWatch CPU, `vmstat 1` | OOM kills stop without a CPU regression |
| gp3 IOPS increase | Queue depth; I/O wait; p99 request latency | `iostat -x 1`, CloudWatch `VolumeQueueLength` | I/O wait falls; p99 improves under burst |

If a change does not move the relevant metric, revert it. Configuration that persists without benefit is technical debt.

## A decision checklist

Before applying any of these:

1. Have you confirmed the running kernel supports the feature you are about to enable? (`sysctl net.ipv4.tcp_available_congestion_control`, `modinfo zram`, `uname -r`)
2. Do you have a baseline measurement from the last 24 hours that you can compare against?
3. Is the change persisted in `/etc/sysctl.d` or an equivalent, so it survives a reboot?
4. Have you verified the change took effect, rather than assuming the command succeeded?
5. Do you have a rollback plan that does not require console access?
6. If you are changing EBS IOPS, have you confirmed with `iostat` that the workload is actually I/O-bound?

## FAQ

**Does this work on Graviton instances?**
Yes. The kernel interfaces are identical across architectures. The `node_exporter` and `iperf3` binaries differ by architecture; download the `arm64` builds.

**Do sysctl changes require a reboot?**
No. `sysctl -w` applies immediately, and `sysctl --system` reloads from the configuration directory.

**Is `bbr2` a real thing?**
It was an experimental development branch of BBR, not a stable kernel interface. The stable algorithm is `bbr`. If you see `bbr2` in a tutorial, the tutorial is out of date.

**Can I apply these changes with Terraform?**
Yes. Use `user_data` to run a shell script at first boot that writes the sysctl files and enables zram. For EBS, use the `aws_ebs_volume` resource's `iops` attribute on gp3 volumes; Terraform will call `modify-volume` if the attribute changes.

**Will this reduce my bill by a specific percentage?**
No. The bill reduction depends on whether the changes reduce CPU utilization enough to justify a smaller instance type, or eliminate a scaling event. The way to find out is to apply the changes, measure CPU and latency for a full billing cycle, and then evaluate whether a resize is safe.

## What to do in the next 30 minutes

SSH into one production instance and run the following, then record the output somewhere you can find it in a week:

```bash
echo "=== kernel ==="; uname -r
echo "=== congestion control ==="; sysctl net.ipv4.tcp_congestion_control
echo "=== available algorithms ==="; sysctl net.ipv4.tcp_available_congestion_control
echo "=== qdisc ==="; sysctl net.core.default_qdisc
echo "=== swap ==="; swapon --show
echo "=== swappiness ==="; sysctl vm.swappiness
echo "=== retransmit counter ==="; grep Tcp_RetransSegs /proc/net/snmp
echo "=== disk ==="; lsblk -d -o NAME,SIZE,ROTA
echo "=== iostat ==="; iostat -x 1 3
```

That snapshot is your baseline. Without it, any change you make afterward is unfalsifiable. With it, you can decide in a week whether the kernel and storage tuning described here is worth keeping on your workload — or whether the real bottleneck is somewhere you have not looked yet.
