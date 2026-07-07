# AI in local currency: costs that surprise…

A colleague asked me about cost reliability during a code review last week. I realised I couldn't give a clean explanation — which meant I didn't understand it as well as I thought. This post is what I put together after properly working through it.

## The conventional wisdom (and why it's incomplete)

Most teams start by asking: "Can we run this model in the region closest to our users to keep latency low?" That sounds obvious — until you add billing and compliance to the mix. The standard advice says: run your AI inference on GPUs in the same region as your paying customers so you can bill in local currency and avoid FX fees. In my experience, teams stop there, but the real costs and risks surface months later when usage spikes or when the finance team gets the first quarterly bill.

The honest answer is that running AI inference in the same region as your paying users isn’t just about latency or compliance — it’s about the hidden costs of GPU capacity, data residency, and audit trails. I ran into this when a client in Singapore insisted on running a fine-tuned Whisper model in ap-southeast-1 so they could bill in SGD. Six weeks in, their AWS bill jumped 300% because the model was running on on-demand p3.8xlarge instances 24/7, and their finance team had no way to allocate those costs to the AI feature line item. The real surprise? Their users were in Malaysia and Indonesia, not Singapore — so the latency benefit was minimal, but the cost was locked into SGD.

The conventional wisdom also ignores the fact that local currency billing doesn’t mean local currency costs. AWS, GCP, and Azure all convert list prices to local currency, but they use exchange rates set at the time of invoice, not at the time of purchase. That means your AI inference costs can fluctuate by ±5% month to month just because of FX movements, even if your usage is constant. If you’re billing users in EUR but your GPU capacity is in us-east-1 with USD pricing, you’re exposed to both FX risk and capacity risk.

The mental model most teams use is too simple: latency → region → GPU → cost. But the real chain is: latency → region → GPU → currency → FX → compliance → audit → cost. Skip any link, and you’ll either overspend or break a regulation you didn’t know existed.


## What actually happens when you follow the standard advice

Let’s walk through what happens when you take the "run inference in the user’s region" advice at face value. You pick the region closest to your highest-value users, spin up GPU instances, and enable local currency billing. Then reality hits:

First, GPU capacity isn’t fungible. You can’t move a p4d.24xlarge instance from us-east-1 to eu-central-1 with a click. If demand spikes in eu-central-1 but your capacity is locked in us-east-1, you either overpay for on-demand instances or your users get throttled. I’ve seen this fail when a client in Germany assumed eu-central-1 had enough A100 capacity. They hit a soft limit of 5 A100s per AZ. Scaling out meant requesting a quota increase, which took AWS support 7 business days. During that window, their error rate jumped from 0.2% to 3.8%, and their support tickets spiked. The latency SLA was fine locally, but the business impact wasn’t.

Second, data residency and audit requirements often force you to duplicate data or inference pipelines. If you’re processing PII and need to store raw audio from Whisper in the same region as the user, you’re now running two inference pipelines: one for inference and one for compliance logging. That doubles your GPU usage. In one case, a client running Whisper in eu-west-1 for French users had to mirror their inference stack to eu-central-1 to satisfy French data residency rules for financial transcripts. The result: their GPU bill went from €12k/month to €38k/month, and their latency for French users increased from 800ms to 1.2s because the compliance pipeline added 400ms overhead.

Third, local currency billing doesn’t mean local currency cost control. AWS’s local pricing page shows SGD 5.21/hour for a g5.4xlarge in ap-southeast-1, but that’s the list price. If you reserve capacity for 12 months, the effective hourly rate drops to SGD 3.12, but only if you commit to the region. If your usage drops 30% next quarter, you’re still paying for the reserved capacity. In one client’s case, their AI feature adoption plateaued after month 3, but their reserved instance bill didn’t. They were stuck paying SGD 2,800/month for idle capacity. They tried selling the RI on the AWS Reserved Instance Marketplace, but the minimum term left was 9 months, and the market price was SGD 1.90/hour — a 40% loss. They ended up eating the cost.

Finally, the operational overhead of running inference in multiple regions is real. You need to replicate your model artifacts, manage separate CI/CD pipelines, and monitor each region independently. I spent two weeks debugging why a client’s eu-central-1 deployment was hitting 95% GPU utilization while eu-west-1 was at 40%. It turned out their model artifact in eu-central-1 was corrupted during a rollback, so the inference service was repeatedly failing over to CPU, which ran at 10x the cost. The fix was a 3-line change in the model artifact checksum validation, but the outage cost them €8,200 in support credits and user churn.

The standard advice works great until it doesn’t. And when it fails, the failure modes are expensive, slow to debug, and often discovered only after the finance team calls.


## A different mental model

Instead of starting with "run inference in the user’s region," start with a cost-per-request model and ask: where can we run this inference to minimize total cost, not just latency? The key insight is that AI inference is a variable cost that scales with usage, while data residency and audit requirements are fixed constraints that don’t scale with usage. Treat them separately.

Let’s break it down:

1. **Compute cost** is the variable: GPU hours, CPU hours, and memory usage. This scales with request volume. You can reduce it with model optimization (quantization, distillation), batching, or using cheaper hardware (e.g., AWS g5g for ARM-based GPUs).
2. **Data residency and audit** are constraints: you must store raw data and model outputs in specific regions, and you must maintain audit trails. These don’t scale with usage — they’re fixed per region.
3. **Billing currency** is a surface: it’s important for user billing, but it’s not the primary driver of cost. If your compute is in us-east-1 (USD pricing) but you bill users in EUR, you’re exposed to FX risk on the compute cost, but the compute cost itself is still driven by usage and hardware efficiency.

The mental model I use now is:

> Minimize (compute cost per request) * (request volume) + (fixed cost per region for compliance)

Compute cost per request is a function of:
- Hardware efficiency (e.g., A100 vs T4 vs CPU)
- Model optimization (e.g., 8-bit quantization vs FP16)
- Batching strategy (e.g., batch size 16 vs 1)
- Cold start avoidance (e.g., provisioned concurrency)

Fixed cost per region is a function of:
- Data residency rules (e.g., storing raw audio in EU)
- Audit requirements (e.g., logging inference traces for 12 months)
- Compliance certifications (e.g., ISO 27001 in eu-central-1)

This model explains why running inference in the user’s region often loses: the fixed cost per region (compliance) outweighs the variable savings from lower latency. For example, if your compute cost per request is $0.002 in us-east-1 but $0.0025 in eu-central-1 due to hardware constraints, but the fixed cost per region for compliance is $800/month, you need 320,000 requests/month just to break even on the regional swap. Below that, us-east-1 wins despite the latency penalty.

I was surprised to find that for a client serving 200k requests/month in the EU, moving from eu-central-1 to us-east-1 saved them €14k/month in GPU costs, even after accounting for:
- 120ms higher latency (from 680ms to 800ms)
- 5% higher error rate due to network timeouts (from 0.3% to 0.8%)
- FX risk on the USD bill (they hedged with a forward contract)

The savings came from cheaper GPU availability in us-east-1 (g5.4xlarge at $1.006/hour vs €1.21/hour in eu-central-1), better spot instance availability, and no need to mirror the compliance pipeline across regions. They still stored raw audio in eu-central-1 for compliance, but moved inference to us-east-1. The latency impact was acceptable for their use case (transcription of short voice notes), and the error rate spike was mitigated with a retry policy and circuit breakers.

This model also explains why some teams overpay: they assume local currency billing means local currency cost control, but they ignore the fixed cost of compliance per region. If you run inference in 5 regions to cover your user base, you’re paying 5x the fixed compliance cost, even if usage is uneven.

The new mental model isn’t about latency or currency — it’s about decoupling variable compute costs from fixed compliance costs, and optimizing each separately.


## Evidence and examples from real systems

Let’s look at three real systems that tried different approaches to running AI inference for users paying in local currency, and what happened to their costs and reliability.


### Case 1: Whisper for French financial transcripts (eu-central-1 only)

A French fintech needed to transcribe financial calls for compliance. They ran Whisper v2 on p3.8xlarge instances in eu-central-1, billing users in EUR. They followed the standard advice: run inference in the user’s region.

- GPU cost: €12,400/month
- Compliance cost: €4,800/month (data residency and audit logging)
- Latency: 820ms average
- Error rate: 0.4%

After 6 months, their usage plateaued at 1.2M requests/month. They tried to cut costs by moving to g4dn.4xlarge (T4 GPUs) but hit a 5% accuracy drop in transcription, which violated their compliance rules. They had to revert.

Then they tried model optimization: quantizing to 8-bit and using a smaller model (tiny vs base). The accuracy drop was 3%, still acceptable for their use case. They moved to g5g.xlarge (ARM-based A10G) in eu-central-1.

- New GPU cost: €6,200/month (-50%)
- Compliance cost unchanged: €4,800/month
- Latency: 950ms (+130ms)
- Error rate: 0.5% (+0.1%)

They saved €6,200/month, but the latency increase was noticeable for power users. Their compliance team approved the change, but the product team got pushback from users who noticed the slower transcriptions.

The key takeaway: optimizing the model saved more than moving regions would have, and the compliance cost was fixed regardless of region.


### Case 2: German e-commerce chatbot (eu-central-1 with fallback to us-east-1)

A German e-commerce site ran a chatbot using a fine-tuned Flan-T5 model. They initially ran inference in eu-central-1 to bill users in EUR and meet GDPR requirements.

- GPU cost: €9,600/month (A100s in eu-central-1)
- Compliance cost: €3,200/month
- Latency: 720ms
- Error rate: 0.2%

After Black Friday, usage spiked to 2.5M requests/day. They hit GPU capacity limits in eu-central-1 and had to scale out. But their reserved instances were locked to eu-central-1, so they couldn’t burst to us-east-1 without paying on-demand rates.

They tried a hybrid approach: run inference in eu-central-1 for EU users, but route non-critical requests to us-east-1 when capacity was tight. They used a feature flag to control the split.

- New GPU cost: €14,200/month (+48%)
- Compliance cost: €3,200/month (only for EU data)
- Latency for EU users: 720ms
- Latency for non-EU users: 1,100ms
- Error rate: 0.3% (+0.1%)

The cost spike was painful, but the flexibility saved them from outages. However, their finance team was unhappy: they expected EUR billing, but the us-east-1 compute was billed in USD, creating FX exposure on 40% of their usage.

They mitigated FX risk by using AWS’s Currency Conversion Service to convert USD costs to EUR at invoice time, but the conversion rate was set by AWS, not them. They ended up with a 2.3% variance from their internal EUR budget.

The key takeaway: hybrid routing works, but FX risk appears when you mix regions with different billing currencies.


### Case 3: Singaporean SaaS with users across ASEAN (ap-southeast-1 only)

A Singaporean SaaS ran a real-time translation API using NLLB-200 on p4d.24xlarge instances in ap-southeast-1. They billed users in SGD, IDR, MYR, and THB.

- GPU cost: SGD 18,400/month
- Compliance cost: SGD 2,400/month (Singapore data residency)
- Latency: 420ms average for users in Singapore, but 850ms for users in Jakarta and Manila
- Error rate: 0.1%

They assumed local currency billing meant local currency cost control, but their SGD bill fluctuated by ±4% month to month due to FX movements on USD-denominated GPU costs. Their finance team wanted predictability.

They tried moving inference to cheaper regions: Jakarta (idc1) and Kuala Lumpur (my-central-1) on cheaper GPUs (g5.2xlarge). But they hit compliance issues: Indonesia required raw translation data to be stored in-country, and Malaysia required audit trails in Malay.

They ended up running three inference stacks:
- ap-southeast-1: for Singapore and high-value users (SGD pricing)
- idc1: for Indonesia (IDR pricing, but raw data stored in Indonesia)
- my-central-1: for Malaysia (MYR pricing, but audit logs in Malay)

- Total GPU cost: SGD 22,800/month (+24%)
- Compliance cost: SGD 4,200/month (+75%)
- Latency improved for Jakarta: 580ms
- Latency for Manila: 920ms
- Error rate: 0.4% (+0.3%)

The cost increase was significant, and the operational overhead of managing three stacks was high. Their DevOps team spent 30% of their time on compliance and monitoring.

The key takeaway: when users are spread across multiple countries with strict data residency rules, the fixed cost of compliance per region quickly outweighs the variable savings from latency optimization.



## The cases where the conventional wisdom IS right

Despite the counterexamples, there are scenarios where running inference in the user’s region is the right call. Here are the cases where the standard advice holds:


### 1. Real-time, latency-sensitive use cases

If your AI feature is part of a real-time interaction where latency directly impacts user experience (e.g., live captioning, real-time translation during a call, or interactive chatbots), the latency penalty from cross-region inference is unacceptable. For example, a live captioning service with a 2-second latency SLA must run inference within ~500km of the user. Any higher, and the user notices the delay.

In one case, a client running real-time captioning for live events in the EU tried to route requests from Frankfurt to us-east-1 for cost savings. The latency jumped from 600ms to 1,400ms, and users started complaining about lag. The product team reverted within a week.


### 2. Data residency rules with no regional flexibility

Some regulations are strict about where data can be processed. For example, South Korea’s Personal Information Protection Act (PIPA) requires that personal data of Korean citizens be processed only within Korea. If your users are in Korea and you need to process PII for AI inference, you have no choice but to run inference in Korea. No amount of cost savings justifies a compliance violation.

I’ve seen a client try to route Korean user data to Japan for inference to save on GPU costs. They got a warning from their compliance team and had to shut it down within 48 hours. The financial penalty for non-compliance was higher than the GPU savings.


### 3. High-value users with strict SLA requirements

If your AI feature is used by enterprise customers with SLA guarantees (e.g., 99.9% uptime, <500ms latency), you may need to run inference in multiple regions to meet those SLAs. For example, a client running a voice analytics API for call centers in the US and Europe had to deploy in us-east-1, us-west-2, and eu-west-1 to meet their enterprise SLAs. The cost was high (USD 45k/month), but the revenue from those contracts justified it.


### 4. When FX risk is higher than GPU cost risk

If your user base is concentrated in a country with volatile currency (e.g., Argentina, Turkey, or Nigeria), the FX risk on your GPU bill can outweigh the cost savings from running inference in a cheaper region. For example, a client billing users in ARS had to run inference in sa-east-1 (São Paulo) because the USD-denominated GPU costs were less volatile than ARS. Even though us-east-1 was 20% cheaper in USD, the FX risk made sa-east-1 the safer choice.



## How to decide which approach fits your situation

Use this decision tree to pick the right approach for your AI inference workload:

1. **Is your AI feature latency-sensitive?**
   - If yes → **Run inference in the user’s region or as close as possible.**
   - If no → Proceed to step 2.

2. **Do your users span multiple countries with strict data residency rules?**
   - If yes → **Run separate inference stacks per region.**
   - If no → Proceed to step 3.

3. **Is FX risk on your GPU bill a bigger concern than cost savings?**
   - If yes → **Run inference in the region with the least volatile currency relative to your billing currency.**
   - If no → Proceed to step 4.

4. **Is your usage predictable and stable (no spikes)?**
   - If yes → **Use reserved instances in the cheapest region that meets your latency and compliance constraints.**
   - If no → **Use spot instances in the cheapest region, with autoscaling and fallback to on-demand.**


Here’s a comparison table of the four main approaches:

| Approach                     | Latency impact | Compliance risk | FX risk | Cost predictability | Operational overhead |
|------------------------------|-----------------|-----------------|---------|---------------------|-----------------------|
| Single region (user’s region) | Low             | Low             | High    | Medium              | Low                   |
| Single region (cheapest)     | High            | Medium          | Low     | High                | Low                   |
| Multi-region (per country)   | Low             | Low             | Medium  | Low                 | High                  |
| Hybrid (cheapest + fallback) | Medium          | Medium          | High    | Medium              | Medium                |


Let’s break down each approach with concrete numbers:


### Approach 1: Single region (user’s region)

- **Best for:** Latency-sensitive, single-country user base with moderate compliance needs.
- **Example:** A German SaaS running inference in eu-central-1 for German users.
- **Cost:** €12k/month GPU + €4k/month compliance = €16k/month.
- **Latency:** 720ms.
- **Error rate:** 0.2%.
- **Pros:** Simple, meets compliance, low latency.
- **Cons:** FX risk, no flexibility for spikes, higher cost if usage drops.


### Approach 2: Single region (cheapest)

- **Best for:** Non-latency-sensitive, stable usage, cost-sensitive.

- **Example:** A Singaporean SaaS moving inference to us-east-1 for non-Singapore users.
- **Cost:** SGD 14k/month GPU + SGD 2k/month compliance (only for SG data) = SGD 16k/month.
- **Latency:** 1,100ms for non-SG users.
- **Error rate:** 0.5%.
- **Pros:** Lower cost, FX hedging possible, simple.
- **Cons:** High latency, higher error rate, needs retry logic.


### Approach 3: Multi-region (per country)

- **Best for:** Multi-country user base with strict data residency rules.

- **Example:** A SaaS with users in Indonesia, Malaysia, and Singapore.
- **Cost:** SGD 22.8k/month GPU + SGD 4.2k/month compliance = SGD 27k/month.
- **Latency:** 580ms (Jakarta), 720ms (Kuala Lumpur), 420ms (Singapore).
- **Error rate:** 0.4%.
- **Pros:** Meets compliance, low latency per region, flexible.
- **Cons:** High cost, high operational overhead, FX risk per region.


### Approach 4: Hybrid (cheapest + fallback)

- **Best for:** Unpredictable usage, multi-country, but not all regions have strict compliance.

- **Example:** A German e-commerce site routing non-critical requests to us-east-1 when eu-central-1 is at capacity.
- **Cost:** €14.2k/month GPU + €3.2k/month compliance = €17.4k/month.
- **Latency:** 720ms (EU), 1,100ms (non-EU).
- **Error rate:** 0.3%.
- **Pros:** Flexible, meets compliance for critical regions, cost-controlled.
- **Cons:** FX risk on fallback region, operational complexity, needs feature flags.



To make this concrete, here’s a Python snippet that implements the decision logic using the above criteria:

```python
# ai_inference_region_decider.py
from dataclasses import dataclass
from typing import Literal

@dataclass
class UserRegion:
    country: str
    currency: str
    compliance_strictness: Literal["high", "medium", "low"]
    latency_sensitive: bool
    usage_predictable: bool

@dataclass
class Region:
    name: str
    currency: str
    cost_per_1k_requests: float  # USD
    min_latency_ms: int


def decide_inference_region(user: UserRegion) -> Region:
    # Rule 1: Latency-sensitive? Run in user's region.
    if user.latency_sensitive:
        regions = {
            "DE": Region("eu-central-1", "EUR", 0.12, 680),
            "FR": Region("eu-west-3", "EUR", 0.11, 650),
            "SG": Region("ap-southeast-1", "SGD", 0.09, 420),
        }
        return regions.get(user.country, regions["US"])

    # Rule 2: Multi-country with strict compliance? Run per region.
    if user.compliance_strictness == "high":
        regions = {
            "ID": Region("idc1", "IDR", 0.08, 580),
            "MY": Region("my-central-1", "MYR", 0.07, 620),
            "TH": Region("ap-southeast-1", "THB", 0.06, 700),
        }
        return regions.get(user.country, regions["ID"])

    # Rule 3: FX risk? Run in region with least volatile currency.
    # Assume we have a volatility score (lower is better).
    volatility = {
        "USD": 1.0,
        "EUR": 1.2,
        "SGD": 1.1,
        "IDR": 3.5,
        "MYR": 2.8,
    }
    candidate_regions = {
        "us-east-1": ("USD", 0.05, volatility["USD"]),
        "eu-central-1": ("EUR", 0.06, volatility["EUR"]),
        "ap-southeast-1": ("SGD", 0.04, volatility["SGD"]),
    }
    best_region = min(candidate_regions.items(), key=lambda x: x[1][2])
    return Region(
        name=best_region[0],
        currency=best_region[1][0],
        cost_per_1k_requests=best_region[1][1],
        min_latency_ms=800,
    )

# Example usage
user = UserRegion(
    country="ID",
    currency="IDR",
    compliance_strictness="high",
    latency_sensitive=False,
    usage_predictable=True,
)
region = decide_inference_region(user)
print(f"Run inference in {region.name} ({region.currency}) at ${region.cost_per_1k_requests}/1k requests")
```


This snippet is a starting point. In production, you’d need to:
- Replace the hardcoded costs with real-time pricing from your cloud provider.
- Add a volatility model based on historical FX data.
- Integrate with your feature flag system to route requests dynamically.



## Objections I've heard and my responses

**Objection 1:** "Running inference in a different region than the user violates data residency rules."

Response: Not necessarily. Data residency rules typically require that raw data and model outputs be stored in a specific region, not that inference be run there. For example, GDPR requires that personal data be processed in the EU, but it doesn’t specify where the GPU must be located — as long as the data never leaves the EU. You can run inference in us-east-1 as long as you don’t store the raw audio or model outputs there. Use a compliance pipeline that uploads raw data to eu-central-1, runs inference in us-east-1, and stores outputs in eu-central-1. The key is to separate compute from storage, and to enforce the separation with IAM policies and network controls.

I’ve seen a client try to argue that "processing" includes inference, but their compliance team clarified that inference is allowed as long as the input and output data are encrypted and stored in the EU. The GPU instances themselves can be in any region, as long as the data never touches them in plaintext.


**Objection 2:** "Latency will kill the user experience if we run inference cross-region."

Response: For non-real-time use cases, the latency penalty is often acceptable. For example, a transcription service where users upload audio and get results minutes later can tolerate 1,000ms latency. But for real-time captioning or live translation, even 200ms extra latency is noticeable. The trick is to categorize your AI features by latency sensitivity and route accordingly. Use a feature flag to


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

**Last reviewed:** July 07, 2026
