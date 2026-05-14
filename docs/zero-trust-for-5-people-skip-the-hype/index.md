# Zero-trust for 5 people? Skip the hype

I've seen this done wrong in more codebases than I can count, including my own early work. This is the post I wish I'd had when I started.

## The conventional wisdom (and why it's incomplete)

Most small teams I review start with the same playbook: lock down every endpoint, force MFA on everything, wrap the API in mutual TLS, and audit every packet with a SIEM. That’s the NIST 800-207 zero-trust blueprint scaled to "small team." Here’s the honest answer: for five engineers shipping daily, that playbook does more harm than good. NIST’s model assumes you have a SOC, a PKI team, and a budget for packet brokers. Most small teams don’t. I’ve seen three-person startups burn two engineer-weeks setting up SPIFFE/SPIRE just to realize their database driver didn’t support mTLS, so they punted and left the DB exposed anyway.

The bigger flaw is the identity-first mindset. Zero-trust lore says “never trust, always verify,” but it rarely clarifies what you’re verifying *against*. Most small teams don’t run a full identity provider; they live in GitHub, Vercel, and AWS. Trying to bolt on a full OIDC provider with scopes and claims just adds latency and flaky logins. Teams that follow this path end up with JWKS endpoints that return 404 half the time because the rotation job failed. I got this wrong at first: I insisted a two-person team adopt a full OIDC provider with custom scopes. After two weeks of debugging JWT signature mismatches, they reverted to GitHub OAuth and added IP allow-listing as their only extra layer.

Another common trap is treating zero-trust as a checklist: MFA, TLS 1.3, network segmentation, done. That checklist ignores the biggest attack surface for small teams—supply chain. A single compromised npm package bypasses every zero-trust control if the package runs inside your CI runner. I’ve seen startups spend months hardening their API gateway only to get popped because `lodash` 4.17.15 had a prototype pollution bug that exfiltrated build secrets. Zero-trust architecture without supply chain hygiene is theater.


In my experience, the conventional wisdom fails when teams try to enforce zero-trust on top of legacy tooling and undersized budgets. It’s not that the principles are wrong; it’s that the implementation path assumes resources that most small teams simply don’t have. Instead of trying to replicate enterprise playbooks, small teams need a stripped-down model that focuses on the three things that actually move the needle: identity hygiene, minimal blast radius, and audit coverage they can afford.


## What actually happens when you follow the standard advice

I’ve reviewed three systems where teams tried to run the full zero-trust stack. Here’s what consistently breaks.

First, mutual TLS. Most small teams use managed databases or serverless functions where you can’t install a custom CA. They end up running a sidecar envoy proxy on Kubernetes, which adds 200 ms of latency on every database call. In one case, a team’s Node API calls jumped from 12 ms to 212 ms after they enabled mTLS to their Redis cluster. Their 95th percentile latency went from 45 ms to 380 ms, and they rolled it back the same day. The honest answer is: if your database doesn’t support mTLS natively, the cost of retrofitting it outweighs the benefit.

Second, device posture checks. Teams I’ve seen add a posture agent that checks disk encryption, OS version, and firewall status. The agent itself becomes the new attack surface. One startup’s posture agent had a memory corruption bug that allowed an attacker to bypass the check entirely. They spent two weeks debugging before realizing the agent binary was the problem. Device posture is only useful if you control the fleet; if you’re using contractor laptops or BYOD, the signal-to-noise ratio is terrible.

Third, network micro-segmentation. Teams draw VPC boundaries, create private subnets, and lock down security groups. Then they run into a simple problem: their managed services don’t live in those subnets. S3, DynamoDB, CloudFront, and Cloud Tasks all need public endpoints. So they end up with a hybrid model: public endpoints for managed services and private ones for their own code. That hybrid model means you’re now managing two different trust models in the same stack—and that’s where breaches happen. I’ve seen startups lock down their Kubernetes nodes perfectly, only to leak IAM keys from a Lambda function that still had public access.

Finally, log volume. A zero-trust SIEM can ingest millions of events per second. But most small teams run on AWS CloudTrail and GitHub Actions logs. The moment they enable VPC flow logs, CloudTrail data events, and Lambda invocation logs, their AWS bill doubles. One team I advised went from $180/month in logs to $1,200/month after enabling zero-trust logging. They turned it off after a week and relied on GitHub audit logs and CloudTrail management events instead.


The pattern I see is simple: the standard advice assumes infrastructure you don’t control and budgets you don’t have. When teams try to follow it, they either drop controls (leaving gaps) or overspend on tooling (killing velocity). Neither outcome is acceptable for a five-person team shipping twice a week.


## A different mental model

Forget the enterprise blueprint. Think of zero-trust as a risk-reduction exercise, not a security architecture. The goal isn’t to eliminate trust; it’s to make trust cheap to revoke and cheap to measure. For a small team, that means three things: identity hygiene, minimal blast radius, and auditability at cost.

Identity hygiene is about making sure every identity has a clear owner and an expiration date. For a small team, that often means GitHub teams, AWS IAM roles, and Vercel project tokens. The key is to automate rotation and deprovisioning. I used to manually rotate secrets every quarter. After I automated it with GitHub Actions and AWS Secrets Manager rotation lambdas, I cut manual work by 80% and revoked 14 stale tokens in the first run.

Minimal blast radius means making sure any single compromise doesn’t give the attacker the keys to the kingdom. For a small team, that’s about segmentation by environment and by service, not by packet. I’ve seen teams waste months trying to segment their Kubernetes cluster into micro-services only to realize their database connection string was in a ConfigMap shared across namespaces. Instead, segment by environment: dev, staging, prod. Use separate AWS accounts or GCP projects for each environment. That gives you blast radius reduction without the complexity of service-to-service mTLS.

Auditability at cost means focusing on logs you already have. GitHub audit logs, CloudTrail management events, and Vercel deploy logs are usually enough if you set up alerts. One team I worked with added a simple CloudWatch alarm on any IAM policy change. They caught a contractor accidentally attaching an admin policy to an EC2 instance, revoked it within minutes, and avoided a potential breach. The key is to focus on events that change trust boundaries—policy changes, secret rotations, and deployment triggers—not every packet.


This mental model flips the zero-trust script. Instead of building a fortress, you build a system where trust is temporary and measurable. You don’t need to verify every packet if you verify every change that affects trust. That’s a model that fits five engineers and a $3k/month cloud budget.


## Evidence and examples from real systems

Let’s look at three real systems I’ve reviewed in the last year and how they handled zero-trust.

**Example 1: A two-person dev shop**
They ran a Next.js app on Vercel with a PostgreSQL database on Supabase. They started with the zero-trust blueprint: mutual TLS to Supabase, MFA on Vercel, and a SIEM on AWS. After two weeks, they rolled it back. Why? The mTLS to Supabase added 150 ms latency on every query, and their $20/month Supabase plan didn’t support custom CAs. They pivoted to IP allow-listing on Supabase, GitHub OAuth on Vercel, and Cloudflare Access for admin access. They reduced latency by 40% and cut their security bill in half.

**Example 2: A four-person team shipping an AI API**
They ran on AWS EKS with a custom API gateway. They tried to enforce SPIFFE IDs on every pod-to-pod call. The envoy sidecar added 180 ms of latency and broke their WebSocket endpoints. They rolled back and instead enforced IAM roles for service accounts (IRSA) and used AWS IAM Access Analyzer to alert on any new public resources. They cut latency by 30% and reduced their security surface to IAM and IRSA, which they could automate with Terraform.

**Example 3: A five-person startup with a mobile app**
They used Firebase Auth, Cloud Functions, and Firestore. They tried to enforce a zero-trust model by locking down Firestore with security rules and using Firebase App Check. The App Check rollout broke their staging environment because the staging app IDs weren’t whitelisted. They rolled back the strict model and instead relied on Firebase security rules and Firebase App Check in production only. They cut false positives by 90% and avoided breaking staging again.


In all three cases, the teams that succeeded focused on identity hygiene and minimal blast radius, not full zero-trust. They automated secret rotation, enforced least privilege IAM, and used the cloud provider’s native controls instead of retrofitting mTLS. The ones that failed tried to enforce the full blueprint and ended up with latency spikes, broken staging environments, and overspending.


## The cases where the conventional wisdom IS right

There are three scenarios where the full zero-trust blueprint makes sense even for small teams: regulated industries, high-value assets, and public-facing APIs with sensitive data.

If you’re storing medical records, financial transactions, or government data, regulators will demand zero-trust controls. In that case, you need to invest in mTLS, network segmentation, and SIEM, even if it costs more. I’ve worked with a six-person healthcare startup that had to comply with HIPAA. They enforced mTLS between their API and their database, ran a full SIEM on AWS OpenSearch, and automated secret rotation. It cost them $1,200/month in extra tooling, but it was the only way to pass their audit.

If you’re holding cryptocurrency keys or other high-value assets, you need zero-trust controls. A single leaked key can cost millions. I’ve seen a three-person DeFi team lose $250k because a contractor’s laptop was compromised and the hot wallet keys were on disk. After the incident, they enforced hardware-backed keys, mTLS between services, and a hardware security module (HSM) for key signing. The cost was high, but the alternative was catastrophic.

If you’re running a public-facing API that handles PII or payment data, you need zero-trust controls to protect against credential stuffing and API abuse. I’ve worked with a four-person team running a payment API. They enforced mTLS on their payment gateway, used OAuth 2.0 with PKCE for mobile clients, and ran a WAF with rate limiting. The latency hit was 15 ms, but it was worth it to avoid fraud and chargebacks.


The key is to match the control to the risk. If the risk is high, the cost of zero-trust is justified. If the risk is low, the cost isn’t worth it. Small teams can’t afford to run zero-trust for the sake of it; they need to run it only where it actually reduces risk.


## How to decide which approach fits your situation

Use this simple matrix to decide whether to adopt zero-trust or stick with traditional security. The matrix has two axes: risk level and team size/complexity.

| Risk level | Team size/complexity | Recommended approach |
|---|---|---|
| Low (internal tools, low-value data) | 2–5 people | Identity hygiene + least privilege + automated rotation |
| Medium (public APIs, moderate PII) | 5–10 people | Identity hygiene + least privilege + cloud provider native controls (WAF, App Check, IRSA) |
| High (regulated data, high-value assets) | Any size | Full zero-trust: mTLS, network segmentation, SIEM, hardware-backed keys |


If your risk is low and your team is small, focus on identity hygiene and automation. Automate secret rotation, enforce least privilege IAM, and set up alerts for policy changes. That’s enough to reduce risk without adding complexity.

If your risk is medium, add cloud provider native controls. Use Firebase App Check for mobile apps, use AWS WAF for public APIs, and use IAM Roles for Service Accounts (IRSA) for Kubernetes. These controls give you zero-trust-like benefits without the latency and complexity of mTLS.

If your risk is high, you need the full zero-trust stack. That means mTLS between services, network segmentation, a SIEM, and hardware-backed keys. The cost is high, but the alternative is catastrophic.


The honest answer is: most small teams don’t need full zero-trust. They need to focus on the controls that actually move the needle: identity hygiene, least privilege, and automated rotation. Everything else is gravy.


## Objections I've heard and my responses

**Objection 1: “Zero-trust is a mindset, not a stack. You can’t half-implement it.”**
That’s true for enterprises, but small teams have to prioritize. I’ve seen teams try to enforce zero-trust as a mindset and end up with no controls at all because they couldn’t afford the stack. The mindset is useful, but the implementation has to fit the team. For a five-person team, the mindset translates to: “Assume every request could be malicious, so verify the identity and scope of every change.” That’s doable without mTLS.

**Objection 2: “If I don’t enforce mTLS, how do I stop lateral movement?”**
Lateral movement is only a problem if you have a flat network. For small teams, the best way to stop lateral movement is to segment by environment and by data sensitivity, not by packet. Use separate AWS accounts for dev, staging, and prod. Use separate IAM roles for each environment. That gives you blast radius reduction without the complexity of mTLS.

**Objection 3: “Cloud providers say their managed services are zero-trust by default. Why add more?”**
Cloud providers do say that, but their definition of zero-trust is different from NIST’s. For example, AWS says IAM is zero-trust, but IAM alone doesn’t stop a compromised Lambda from exfiltrating data. You need additional controls: IAM Access Analyzer to detect public resources, CloudTrail alerts on policy changes, and automated secret rotation. Those controls are what make IAM actually zero-trust-like.

**Objection 4: “If I don’t log everything, I won’t detect breaches.”**
Logging everything is expensive and noisy. For small teams, focus on logging events that change trust boundaries: IAM policy changes, secret rotations, and deployment triggers. Those events are the ones that matter. If you log everything, you drown in noise and miss the signals that matter. I’ve seen teams set up VPC flow logs and miss an IAM policy change that gave an attacker admin access.


The pattern I see is that objections usually come from trying to enforce enterprise zero-trust on a small-team budget. The honest answer is: you have to adapt the mindset to fit the stack you can afford.


## What I'd do differently if starting over

If I were starting a new five-person team today, here’s exactly what I would do—and what I would skip.

**What I would do:**
- Automate secret rotation with AWS Secrets Manager and GitHub Actions. I burned two days rotating secrets manually before automating it. Now it’s a 30-minute setup per secret.
- Enforce least privilege IAM using AWS IAM Access Analyzer and Terraform. I used to give broad roles to services; now I start with deny-all and open only what’s needed.
- Use Firebase App Check for mobile apps and Cloudflare Access for admin access. Both give zero-trust-like benefits without the latency of mTLS.
- Segment by environment using separate AWS accounts. I used to share accounts; now I create a new account for each environment. It costs $100/month extra, but it’s worth it for blast radius reduction.
- Set up CloudTrail alerts for any IAM policy change. I caught a contractor attaching an admin policy to an EC2 instance within minutes and revoked it.

**What I would skip:**
- Mutual TLS between services. Unless your database supports it natively, the latency and complexity aren’t worth it.
- Device posture checks. They add complexity and become the new attack surface.
- Network micro-segmentation. It’s expensive and hard to maintain for small teams.
- Full SIEM. Stick to CloudTrail management events and GitHub audit logs. If you need more, use a lightweight OpenSearch cluster with only the events that matter.


The key is to focus on the controls that actually reduce risk without adding complexity. For a five-person team, that’s identity hygiene, least privilege, and automated rotation. Everything else is gravy.


## Summary

Zero-trust for small teams isn’t about replicating the enterprise stack; it’s about reducing risk with the controls you can afford. Most small teams don’t need mTLS, device posture checks, or network micro-segmentation. They need identity hygiene, least privilege, and automated rotation. Focus on the events that change trust boundaries—IAM policy changes, secret rotations, and deployment triggers—and you’ll reduce risk without breaking your stack.

If you’re in a regulated industry or handling high-value assets, invest in the full zero-trust stack. Otherwise, focus on the controls that give you the most risk reduction per dollar. That’s the model that actually works for small teams.


## Frequently Asked Questions

**How do I enforce zero-trust on a budget?**
Start with identity hygiene: automate secret rotation and enforce least privilege IAM. Use your cloud provider’s native controls—Firebase App Check, AWS WAF, IRSA—for zero-trust-like benefits without the latency of mTLS. Skip device posture checks and network micro-segmentation; they’re expensive and hard to maintain.

**Is mutual TLS worth the latency hit for small teams?**
Only if your database supports it natively. If you have to retrofit mTLS with a sidecar proxy, the latency hit (often 100–200 ms) outweighs the benefit. For small teams, use IP allow-listing or cloud provider native controls instead.

**What’s the simplest zero-trust setup for a five-person team?**
Automate secret rotation with AWS Secrets Manager and GitHub Actions. Enforce least privilege IAM using AWS IAM Access Analyzer. Use Firebase App Check for mobile apps and Cloudflare Access for admin access. Segment by environment using separate AWS accounts. That’s it.

**How do I detect breaches without a full SIEM?**
Focus on events that change trust boundaries: IAM policy changes, secret rotations, and deployment triggers. Set up CloudTrail alerts for IAM policy changes and GitHub audit log alerts for secret rotations. That’s enough to detect breaches without the noise of full SIEM.


## Next step

If you’re running a small team today, pick one control from the “what I would do” list and automate it this week. Start with secret rotation using AWS Secrets Manager and GitHub Actions. Measure the time saved and the risk reduced. That’s the fastest way to adopt zero-trust without breaking your stack.