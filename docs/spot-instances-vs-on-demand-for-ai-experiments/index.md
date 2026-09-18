# Spot instances vs on-demand for AI experiments

infrastructure patterns is easy to demo and hard to keep honest at scale. The workaround gets copy-pasted forward long after the original reason is forgotten. This is the writeup with the mistakes left in, not edited out.

## Why this comparison matters right now

AI experimentation budgets are under pressure. A typical fine-tuning run on a 7B parameter model with LoRA uses around 12 GPU-hours on an A100. At on-demand rates of roughly $3.50 per GPU-hour, that's $42 per experiment. Multiply by 50 experiments a week across a small team and you're looking at $8,400 a month before you've even served a single user. The instinct is to cut experiments. The better move is to cut the cost per experiment.

The two levers that actually move the needle [are spot instances](/gpu-spot-instances-reserved-for-ai-costs/) (or their equivalents on other clouds) and aggressive checkpointing with preemptible-friendly orchestration. But there's a catch: spot instances are not a drop-in replacement for on-demand. They come with a 2-minute warning before termination on AWS, and if your training loop can't survive a mid-epoch kill, you'll spend more time babysitting than you saved in dollars.

A common failure mode here is teams enabling spot, watching their first job get preempted at epoch 3, and immediately concluding spot is unusable. The real issue is almost always that the training script has no resumable checkpoint, or the checkpoint saves only at epoch boundaries while the job runs 6-hour epochs. The part that trips people up is the interaction between checkpoint frequency and preemption rate, and that's what this post actually covers.

## Option A — how it works and where it shines

Option A is the spot-instance pattern: request capacity from a pool of unused cloud VMs at a 60–90% discount, accept that the provider can reclaim it with short notice, and design your training job to survive that. On AWS, this means EC2 Spot with a capacity-optimized allocation strategy across multiple instance types. On GCP, it's Spot VMs (formerly preemptible). On Azure, it's Spot Virtual Machines.

The core mechanic is the interruption notice. AWS gives you a 2-minute warning via the instance metadata service and an EventBridge event. GCP gives you 30 seconds via a metadata endpoint that flips to `TRUE`. Azure gives you 30 seconds via Scheduled Events. Your job either handles that signal or it doesn't.

Where spot shines is embarrassingly parallel work and long training runs with frequent checkpoints. A hyperparameter sweep over 200 configurations, each taking 20 minutes, is a perfect fit: if 10% get preempted, you lose 10% of the work and rerun it. A single 8-hour training run is a worse fit unless you checkpoint every 10–15 minutes.

The discount is real but variable. In 2026, typical spot discounts for GPU instances (A10G, L4, A100) hover between 55% and 75% off on-demand, depending on region and time of day. CPU instance discounts are deeper, often 70–90%. The catch is that the discount is not guaranteed, and during capacity crunches spot prices can spike toward on-demand.

```python
# spot_handler.py — minimal interruption handler for AWS EC2 Spot
import requests
import signal
import sys
import threading
import time

CHECKPOINT_INTERVAL = 600  # seconds

def watch_for_termination():
    while True:
        try:
            r = requests.get(
                "http://169.254.169.254/latest/meta-data/spot/termination-time",
                timeout=2,
            )
            if r.status_code == 200:
                print(f"Termination notice: {r.text}", file=sys.stderr)
                # Trigger emergency checkpoint
                save_checkpoint(emergency=True)
                sys.exit(0)
        except requests.exceptions.RequestException:
            pass
        time.sleep(5)

threading.Thread(target=watch_for_termination, daemon=True).start()
```

## Option B — how it works and where it shines

Option B is the on-demand-plus-autoscaling pattern: run on on-demand instances, but scale the fleet down to zero between experiments and use a queue (SQS, Pub/Sub, or a simple Redis list) to feed jobs to workers. The win here isn't the per-hour rate; it's the utilization rate. Most teams running AI experiments leave idle GPU instances running overnight and on weekends. A single A100 left on for a weekend costs about $168 at $3.50/hour. Ten of them cost $1,680.

On-demand shines when your job can't tolerate interruption, when you need guaranteed capacity for a deadline, or when your workload is short enough that checkpoint overhead dominates. A 5-minute inference benchmark doesn't need spot; it needs to not run on a machine that's been idle for 3 hours.

The autoscaling pattern is well-supported: AWS Auto Scaling groups with GPU instance types, GCP Managed Instance Groups, or Kubernetes with Karpenter 0.37+ or Cluster Autoscaler. Karpenter in particular has gotten good at provisioning GPU nodes quickly, though cold-start times for GPU nodes are still 3–8 minutes, which matters if your experiments are short.

The other on-demand lever is commitment: 1-year or 3-year Savings Plans or Reserved Instances cut 30–60% off on-demand for steady-state workloads. If you know you'll run at least 4 GPU instances continuously, a Compute Savings Plan is usually cheaper than spot once you account for the engineering time spent on interruption handling.

```yaml
# karpenter-nodepool.yaml — GPU nodepool with aggressive consolidation
apiVersion: karpenter.sh/v1
kind: NodePool
metadata:
  name: gpu-experiments
spec:
  template:
    spec:
      requirements:
        - key: karpenter.sh/capacity-type
          operator: In
          values: ["spot", "on-demand"]
        - key: node.kubernetes.io/instance-type
          operator: In
          values: ["g5.xlarge", "g5.2xlarge", "g6.xlarge"]
  disruption:
    consolidationPolicy: WhenEmptyOrUnderutilized
    consolidateAfter: 5m
  limits:
    nvidia.com/gpu: 32
```

## Head-to-head: performance

Performance here means two things: raw throughput and effective throughput after preemptions. Raw throughput on spot and on-demand is identical — same hardware. The difference is effective throughput.

With a 10% preemption rate and checkpoints every 15 minutes on a job with 6-hour epochs, a typical team sees effective throughput drop by 15–25% versus on-demand, because each preemption costs you the work since the last checkpoint plus the 3–5 minutes to get a new instance and reload state. With checkpoints every 2 minutes, that drop falls to 5–8%. The trade-off is checkpoint I/O: saving a 7B model's optimizer state is roughly 50–80 GB, and writing that every 2 minutes to S3 at 200 MB/s takes 4–6 minutes, which is longer than the interval. That's the trap: checkpointing too often makes the job checkpoint-bound.

The practical sweet spot for most teams is checkpointing every 8–12 minutes to a local NVMe drive, with async upload to S3. That keeps the preemption penalty around 10% while keeping I/O off the critical path.

On-demand has no preemption penalty but suffers from queue latency if you're scaling from zero. GPU node cold start on EKS with Karpenter is typically 3–8 minutes in 2026. If your experiments are 10 minutes long, that's a 30–80% overhead. The fix is a small warm pool of 1–2 on-demand instances that stay up, which reintroduces some idle cost but bounds it.

| Metric | Spot (checkpoint 10 min) | On-demand (warm pool) |
|---|---|---|
| Effective throughput vs raw | 88–92% | 95–99% |
| Preemption penalty per event | ~10 min work + 4 min restart | 0 |
| Cold start (scale from zero) | 3–8 min | 3–8 min |
| Cost per A100-hour (typical) | $0.90–$1.60 | $3.20–$3.80 |
| Engineering overhead | Medium–high | Low |

## Head-to-head: developer experience

Spot is worse to develop against, and anyone who says otherwise is selling something. The interruption handler is easy to write; making the whole training pipeline idempotent and resumable is not. You need deterministic data loading (seed your shuffles), checkpoint versioning, and a way to detect and discard partially-written checkpoints. A common bug is a checkpoint that writes the model weights but not the optimizer state, so a resumed job silently trains from a worse starting point and you don't notice until the eval numbers are off by 4%.

On-demand is boring, and boring is a feature. Your job runs, you read the logs, you move on. The DX cost is mostly in cost visibility: without tagging and a budget alarm, on-demand instances quietly accumulate. AWS Budgets with a $500 threshold and a Slack webhook catches most of this.

The tooling matters here. Weights & Biases 0.17 and MLflow 2.12 both handle resumable runs reasonably well, but neither will save you from a non-idempotent data pipeline. If you're using Ray 2.9+ for distributed training, its fault tolerance is genuinely good — Ray Train will restart workers on preemption if you configure `max_failures`. If you're rolling your own `torchrun` loop, you're writing the resume logic yourself.

A specific gotcha: on AWS, the spot termination notice endpoint returns 404 until a termination is imminent, and some HTTP clients treat 404 as an exception rather than a status code. If your handler catches `requests.exceptions.RequestException` and exits on any exception, you'll kill your job on the first poll. The handler above checks `status_code == 200` explicitly for this reason.

## Head-to-head: operational cost

Let's put numbers on it. Assume a team running 50 experiments a week, each averaging 4 GPU-hours on an A100-equivalent. That's 200 GPU-hours per week, or ~870 per month.

- On-demand at $3.50/hour: **$3,045/month** in compute, plus ~$400 in idle warm-pool overhead, so ~$3,445.
- Spot at a 65% discount ($1.23/hour): **$1,070/month** in compute, plus ~15% effective-throughput loss (you need ~1,000 GPU-hours to do 870 hours of work), so ~$1,230, plus engineering time.

The engineering time is the honest part. Building and maintaining a resumable spot pipeline is realistically 2–4 weeks of initial work and 2–4 hours a month of maintenance. At a fully-loaded engineer cost of ~$100/hour, that's $8,000–$16,000 up front and $200–$400/month ongoing. If your compute bill is $3,445/month, spot saves you ~$2,200/month, so the payback period is 4–7 months. If your compute bill is $30,000/month, payback is under a month.

This is the threshold most teams get wrong. Spot doesn't pay off at small scale unless you already have the resumable infrastructure for other reasons. It pays off enormously at large scale. The crossover is roughly $8,000–$10,000/month in GPU spend.

The third option, which people forget, is to make experiments cheaper rather than instances cheaper. LoRA instead of full fine-tuning cuts GPU-hours by 5–10x. Distillation cuts inference cost. Quantization to int8 or int4 cuts serving cost 2–4x. These are often bigger levers than spot, and they don't require interruption handling.

## The decision framework I use

I default to on-demand with a warm pool and a queue, and I move to spot when one of three conditions holds:

1. Monthly GPU spend is above ~$8,000 and the workload is checkpointable.
2. The workload is embarrassingly parallel (sweeps, batch inference) where a lost unit just gets requeued.
3. The team already has resumable training infrastructure for other reasons (multi-day runs, spot-tolerant frameworks like Ray).

I stay on on-demand when experiments are short (under 15 minutes), when deadlines are tight and a preemption would blow a demo, or when the team is small enough that 2 weeks of spot engineering is 10% of total engineering capacity for the quarter.

The framework is deliberately conservative because the failure mode of spot is not "we spent more money," it's "we spent more money and the results are subtly wrong because a resumed run used a stale checkpoint." That's a worse outcome than overpaying on-demand.

There's also a hybrid that works well: run the long training jobs on spot and the short eval/inference jobs on on-demand. The long jobs benefit most from the discount and are the easiest to checkpoint; the short jobs benefit most from instant availability. This split typically captures 70–80% of the spot savings with 20% of the operational complexity.

## My recommendation (and when to ignore it)

Use spot if your monthly GPU spend exceeds ~$8,000, your training jobs run longer than 20 minutes, and you have or can build idempotent checkpointing. Use on-demand with a warm pool and autoscaling if you're below that threshold, or if your experiments are short, or if your team is small enough that the engineering time is the real constraint.

The weakness in my preferred option (spot for large teams) is that it's fragile in ways that don't show up in benchmarks. A resumed run that silently diverges, a checkpoint that races with a termination notice, a data loader that isn't seeded — these are the bugs that cost you a week and a paper deadline. If you don't have the engineering discipline to test resume paths, don't use spot. The discount is not worth a wrong result.

The weakness in on-demand is that it's easy to let it drift. Without tagging and budget alarms, idle instances accumulate and you're back to the original problem, just paying full price for it. The autoscaling pattern only saves money if you actually scale down.

## Frequently Asked Questions

**How much cheaper are spot instances than on-demand for GPU training in 2026?**
Typical discounts for GPU instances (A10G, L4, A100) run 55–75% off on-demand, varying by region and time. CPU instances often see 70–90% discounts. The discount is not fixed; during capacity crunches spot prices can approach on-demand. Always check current spot pricing for your specific instance type and region before committing to a savings estimate.

**What happens to my training job when a spot instance is preempted?**
On AWS you get a 2-minute warning via the instance metadata endpoint and an EventBridge event. On GCP and Azure it's 30 seconds. If your job doesn't handle the signal, it's killed and you lose everything since the last checkpoint. The fix is a background thread that polls the metadata endpoint and triggers an emergency checkpoint on notice.

**Why does my resumed training run produce worse results than an uninterrupted one?**
The most common cause is a checkpoint that saves model weights but not optimizer state, or a data loader whose shuffle isn't seeded, so the resumed run sees a different data order. Both cause silent divergence. Test your resume path explicitly by killing a job mid-epoch and comparing the final loss to an uninterrupted run.

**Is spot worth it for a small team running 20 experiments a week?**
Usually not. If your monthly GPU spend is under ~$8,000, the engineering cost of building and maintaining resumable spot infrastructure typically exceeds the compute savings, with a payback period of 4–7 months. Spend that effort on making experiments cheaper instead — LoRA, distillation, or quantization often cut GPU-hours more than spot cuts GPU prices.

## Final verdict

Spot instances are the bigger lever, but only above a spend threshold that most small teams don't hit. Below ~$8,000/month in GPU spend, the honest answer is that on-demand with good autoscaling and a warm pool is cheaper once you account for engineering time. Above that threshold, spot with 10-minute checkpoints and a tested resume path is the single biggest cost reduction available, and it's not close.

The action for today: open your cloud billing console, filter GPU instances for the last 30 days, and write down two numbers — total GPU spend and average utilization (idle hours divided by total hours). If spend is above $8,000 and utilization is below 60%, you have both a spot opportunity and an autoscaling opportunity, and the autoscaling one is cheaper to fix first. Start there before you touch spot.


---

### About this article

**Written by:** [Kubai Kevin](/about/) — software developer based in Nairobi, Kenya, with 10+ years building production systems in fintech and AI.

**How this article was produced:** This site uses an automated LLM pipeline designed and maintained by the author. Topics are selected from real production experience. Drafts pass automated quality gates (minimum length, uniqueness, concrete metrics, versioned tools, code samples, absence of filler). Individual line-by-line human editing is not performed on every post before publication. Specific numbers, benchmarks and cost figures are illustrative; verify them against current official documentation before production use.

**Corrections:** Report errors via the [contact page](/contact/). Corrections are applied promptly.

**Last generated:** September 2026
