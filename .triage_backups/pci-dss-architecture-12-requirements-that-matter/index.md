# PCI DSS architecture: 12 requirements that matter

PCI DSS is often treated as a compliance checkbox: scan the network, fill out the self-assessment questionnaire, get the certificate. But the standard's 12 requirements encode a set of architectural constraints. Ignore them during design and the rework arrives later, usually after a penetration test or a QSA review, when changes are most expensive.

This article treats the requirements as design drivers rather than audit items. It walks through the ones that most often force structural decisions — network segmentation, key management, logging, secure development, scope reduction, and testing — and shows how a typical cardholder data environment (CDE) gets organized around them. The examples use AWS service names because they are widely documented, but the patterns apply to any cloud.

## Why scope is the real design problem

A payment system that touches card data — even if it only stores a token or passes a PAN through — inherits PCI DSS scope. Scope drives everything downstream: the more systems in scope, the more controls you must apply, the more evidence you must produce, and the more places an auditor can find a gap.

The standard is public and detailed, but its architectural implications are buried in requirement language that reads like policy. The translation that matters is concrete: where to put the CDE boundary, how to handle keys, what to log and how to protect it, and how to keep developers away from production card data. Each of the six steps below maps to one or more requirements and ends with a decision you can evaluate on its own.

## Prerequisites and the target skeleton

Readers should be comfortable with VPC networking, IAM, and basic cryptography concepts such as symmetric encryption and key wrapping. The goal is not a full payment gateway but the skeleton of a CDE: a segmented network, a key management flow, a logging pipeline, and a CI/CD gate. Each piece is small enough to reason about and to test.

## Step 1 — Draw the CDE boundary before anything else

Requirement 1 covers network security controls. The architectural consequence: you need a clearly defined boundary around systems that store, process, or transmit cardholder data. In AWS, that typically means a separate VPC, or a set of subnets with strict security groups and NACLs, plus rules that deny all traffic except explicitly required flows.

A common mistake is to place the payment service in the same VPC as the rest of the application and rely on application-level authentication. That pulls every instance in the VPC into scope. A stronger pattern is a dedicated VPC for the CDE, connected to the application VPC only through a narrow set of endpoints — a private link endpoint or a transit gateway with route table isolation. The design goal is that compromise of the general application tier does not grant network access to the CDE.

Once the boundary exists, stricter controls inside it become cheap to apply: no internet egress, no SSH from the corporate network, narrowly scoped IAM roles. Logging also becomes tractable, because you know exactly which flow logs and VPC flow logs belong to the CDE.

A minimal security group setup for a CDE subnet looks like this (Terraform, illustrative):

```hcl
resource "aws_security_group" "cde_api" {
  name        = "cde-api"
  description = "Allow only from app tier on 443"
  vpc_id      = aws_vpc.cde.id

  ingress {
    from_port       = 443
    to_port         = 443
    protocol        = "tcp"
    security_groups = [aws_security_group.app_tier.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = []
  }
}
```

The empty egress block is deliberate: no outbound traffic from the CDE except to specific internal endpoints such as KMS and logging. A frequent trap is to allow broad egress for package updates and then forget to lock it down. That weakens the isolation Requirement 1 is meant to demonstrate and makes data exfiltration easier.

**Decision to make:** is the CDE a separate VPC or a set of subnets? The separate VPC is easier to demonstrate and to audit; subnets in a shared VPC require you to prove that security groups and NACLs prevent lateral movement, which is harder to evidence.

## Step 2 — Encrypt data at rest and in transit, but design around keys

Requirement 3 covers protection of stored cardholder data; Requirement 4 covers encryption in transit. The architectural driver is not the algorithm — it is key management. Storing encrypted PANs requires a documented process for key generation, distribution, rotation, retirement, and split knowledge. In AWS, that usually means KMS customer managed keys with automatic rotation enabled where possible, and never storing keys alongside the data they protect.

The standard pattern is envelope encryption: the payment service calls KMS to encrypt a data key, then uses that data key with AES-256-GCM to encrypt the PAN. The encrypted data key is stored with the ciphertext. The service's IAM role needs `kms:Decrypt` on the specific key, and that role should not be assumable by developers directly.

This forces a service boundary. You cannot let the general application tier decrypt PANs. Only the CDE service should hold decrypt permission, and only for the specific key it needs. Everything else works with tokens.

A common failure mode is a single KMS key shared across all environments and data types. Rotation then becomes catastrophic because you cannot rotate without re-encrypting everything, and least privilege is impossible to express. Separate keys per environment and per data classification, with key policies that deny decryption to anyone except the specific service role, avoid both problems.

In transit, TLS 1.2 or later is table stakes. The architectural point is to terminate TLS at the CDE edge and re-encrypt internally if traffic crosses subnets, or to use mTLS between services. A service mesh provides mTLS but adds operational complexity; a simpler approach is to keep the CDE small enough that internal traffic stays within a single security group and never leaves the VPC.

**Worked example of key separation.** Suppose you have three environments (dev, staging, production) and two data classes (PANs, tokens). A single-key design gives you one key to rotate and one blast radius. A separated design gives six keys: dev-PAN, dev-token, staging-PAN, staging-token, prod-PAN, prod-token. Rotating the production PAN key then touches only the production payment service, and a staging compromise cannot decrypt production ciphertext even if the ciphertext leaks. The cost is six key policies to maintain, which is trivial compared to a rotation project that requires coordinated re-encryption across every environment.

## Step 3 — Log everything, then protect the logs

Requirement 10 is the one that most often forces a logging pipeline redesign. You must log all access to cardholder data, all administrative actions, and all authentication events, and you must protect those logs from modification. The architectural consequence: logs cannot live on the same host as the application, and they cannot be writable by the application user.

A standard pattern is to ship logs to a centralized logging account or a dedicated archive bucket with S3 Object Lock in compliance mode and a retention policy. Use CloudTrail for API activity, VPC Flow Logs for network flows, and application logs for access events. The application writes to stdout; a sidecar or agent forwards the records. The application never has write access to the final store.

If an attacker compromises the CDE host, they should not be able to delete or alter logs. The log forwarder therefore uses a separate IAM role with write-only permission to the log store, and the application role must not have `s3:PutObject` on the log bucket. A common trap is granting the application role broad S3 permissions for unrelated reasons, which accidentally hands it log deletion rights.

A minimal log archive bucket policy (illustrative) denies deletion and requires encryption in transit:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "DenyDelete",
      "Effect": "Deny",
      "Principal": "*",
      "Action": "s3:DeleteObject",
      "Resource": "arn:aws:s3:::cde-logs/*"
    },
    {
      "Sid": "RequireTLS",
      "Effect": "Deny",
      "Principal": "*",
      "Action": "s3:*",
      "Resource": "arn:aws:s3:::cde-logs/*",
      "Condition": {"Bool": {"aws:SecureTransport": "false"}}
    }
  ]
}
```

You also need to log access to the logs themselves. CloudTrail data events on the log bucket provide that record. Note that a bucket policy is only one layer: an account administrator can still change the policy. Sending logs to a separate logging account with a restrictive policy that denies deletion even to the CDE account's root is the stronger pattern, and the one auditors most often expect.

**Decision to make:** same-account log archive or separate logging account? Same-account is simpler to operate but leaves logs reachable by anyone who gains admin in the CDE account. Separate-account costs one more account and a cross-account delivery role, and removes that path.

## Step 4 — Secure the development process

Requirement 6 covers secure development. The architectural driver: you need a CI/CD pipeline that enforces code review, separates duties, and prevents developers from deploying directly to production. In practice that means infrastructure as code, peer review on every change, and a deployment pipeline that uses a separate role with no human access.

A common pattern is GitOps: changes merge to main, a pipeline builds and tests, and a separate deployment role applies changes to the CDE. Developers have no console access to the CDE account; they get read-only access for debugging through a break-glass process that is logged and requires approval.

If developers can deploy directly, separation of duties cannot be enforced and there is no guarantee that only reviewed code reaches the CDE. This often forces a multi-account structure: a shared services account for CI/CD, a CDE account for production, and a development account for non-production. AWS Organizations service control policies can restrict what the CDE account is used for.

A full SCP that allows only the pipeline role is complex to write and maintain. A simpler first step is an IAM permissions boundary on developer roles that denies `sts:AssumeRole` into the CDE account. A common trap: a developer with admin in the dev account can still assume a role in the CDE account if the trust policy is too broad. Check the trust policy, not just the permissions policy — the trust policy is what decides who can enter.

## Step 5 — Tokenize early to reduce scope

Requirement 3 also recognizes tokenization as a way to reduce scope. The architectural decision: if the PAN can be replaced with a token before it reaches your systems, those systems fall out of scope. This is often the highest-leverage change available. A tokenization service sits at the edge, receives the PAN from the payment form through a JavaScript integration that posts directly to the tokenization provider, and returns a token. The application never sees the PAN.

If you build your own tokenization, the token vault becomes the most sensitive part of the system and must sit inside the CDE. Keep the vault small, give it a narrow API, and use format-preserving encryption only if downstream systems genuinely require it. Many teams are better served by a payment processor's tokenization than by building and auditing a vault themselves.

Scope reduction changes both the audit effort and the blast radius. If the application is compromised, the attacker gets tokens, not PANs. The trade-off is dependency: you now rely on the provider's availability and on the correctness of the integration, so the boundary between your systems and theirs needs the same scrutiny as any other trust boundary.

## Step 6 — Monitor and test the boundary

Requirement 11 covers security testing. The architectural driver: continuous monitoring of the CDE boundary plus regular penetration testing. In AWS, that means GuardDuty, Security Hub, and Config rules that alert on changes to security groups, IAM policies, or encryption settings. You also need file integrity monitoring on CDE hosts, which can be done with a host-based agent or with immutable infrastructure — container images that are replaced rather than patched.

A common failure mode is to treat monitoring as an afterthought and enable it only after an incident. Build it into the CDE from day one: enable VPC Flow Logs, CloudTrail, and GuardDuty in the CDE account, and route findings to a security account. Use AWS Config rules to detect public S3 buckets or unencrypted volumes.

## How to verify it works

Each control above has a test that takes minutes and catches the most common misconfigurations.

- **Segmentation:** from an app-tier instance, run `nc -zv <cde-host> 22`. It should time out. Repeat for any port that is not on the approved flow list.
- **Key isolation:** using the application role's credentials, run `aws kms decrypt --ciphertext-blob fileb://ciphertext.bin`. It should return `AccessDenied`. Then run the same command with the CDE service role; it should succeed.
- **Log immutability:** using the application role, attempt `aws s3 rm s3://cde-logs/<object>`. It should fail. Confirm the same object is still present via the logging account.
- **Pipeline separation:** attempt to assume the CDE deployment role from a developer role. It should be denied by the trust policy or permissions boundary.
- **Detection coverage:** make a benign change that should trigger an alert — for example, add an overly broad ingress rule to a non-production security group in the CDE account — and confirm the finding reaches the security account within your expected window.

If any of these succeed when they should fail, you have found a real gap before an auditor does.

## Frequently Asked Questions

**What is the minimum scope for PCI DSS?**

The minimum scope is any system that stores, processes, or transmits cardholder data, plus any system that can impact the security of the CDE. In practice that includes the CDE network, the tokenization service if self-hosted, the logging pipeline, and the CI/CD pipeline that deploys to the CDE. Outsourcing tokenization and using a payment processor's hosted payment page removes the most sensitive systems from scope.

**Is a separate VPC required for PCI compliance?**

Not strictly. It is the simplest way to demonstrate segmentation, because a separate VPC with strict security groups and no internet egress is easy to show as isolated. Subnets within a shared VPC are acceptable if you can prove that security groups and NACLs prevent lateral movement, but that proof is harder to produce and to audit.

**How often should keys be rotated?**

PCI DSS requires key rotation but does not prescribe a frequency. A common practice is annual rotation for keys protecting data at rest, plus immediate rotation on suspected compromise. KMS automatic rotation rotates the backing key annually and satisfies the requirement in many cases, but the rotation policy should be documented and tested rather than assumed.

**Can logs live in the same account as the CDE?**

They can, but it is risky: an attacker with admin in the CDE account can delete them. Sending logs to a separate logging account with a bucket policy that denies deletion even to the CDE account's root is the stronger pattern, and same-account storage is a recurring audit finding.

**Does tokenization remove a system from scope entirely?**

It removes the systems that never see a PAN. The tokenization service itself, if self-hosted, remains in scope and becomes the most sensitive component. If a third party provides tokenization, their environment is their responsibility, but your integration and the token-handling paths on your side still need review.

## Where to go from here

Open the IAM policy for your application role and search for `kms:Decrypt`. If it appears, the general application tier can decrypt card data, which is both a scope problem and a least-privilege problem. Remove that permission and move decryption into a dedicated service role that only the CDE service can assume. That single change is a concrete first step toward the boundary the rest of this article describes.
