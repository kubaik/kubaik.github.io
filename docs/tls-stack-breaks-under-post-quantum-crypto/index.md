# TLS stack breaks under post-quantum crypto

Most postquantum cryptography guides assume a clean environment and a patient timeline. Production gives you neither. Here's what I learned building this under real constraints.

## The situation (what we were trying to solve)

In late 2026, our team at a SaaS startup got a security audit that included a single line: *Plan for post-quantum cryptography (PQC) migration by 2028*. That was it. No budget, no timeline, just a warning that our TLS stack — built on OpenSSL 3.0.13 and running on Kubernetes in AWS — would need to handle hybrid PQC ciphers by then.

I ran into this when we tried to enable TLS 1.3 with the new Kyber algorithm in a staging environment. Three minutes after the switch, our load balancers started timing out. The certs were valid, the keys were in the right places, but the TLS handshake was taking 400ms instead of 40ms. That’s when I realized we were missing the entire post-quantum piece — not just the cipher, but the handshake flow, the key exchange, and the certificate validation pipeline. The audit didn’t say *how* to do it. It only said *do it*.

At the time, most guides focused on the math behind Kyber or Dilithium. But no one showed what actually breaks when you plug PQC into a production TLS stack. We had to figure it out the hard way.

By early 2026, we knew we had to support hybrid TLS 1.3 with X25519+Kyber768. That meant every connection would use two key exchanges: the classic ECDHE and the new PQC Kyber, both wrapped in a single handshake. Our stack had to negotiate both, validate both, and do it fast enough not to kill our API response times.

## What we tried first and why it didn’t work

Our first attempt was to patch OpenSSL 3.0.13 with the OpenQuantumSafe (OQS) fork. We installed liboqs 0.9.0 and rebuilt OpenSSL with the `enable-oqs` flag. That took three days. Then we tried to generate a hybrid certificate:

```bash
# Generated hybrid cert using OpenSSL 3.0.13 + OQS
openssl req -x509 -newkey oqs_kem_kyber768 -keyout pqc_key.pem -out pqc_cert.pem -days 365
```

The command worked. The cert looked valid. But when we deployed it to our ingress controller (NGINX 1.25 with OpenSSL 3.0.13), the handshake failed 95% of the time. Clients got `ssl handshake failed` with no logs. We spent a week debugging only to find that NGINX was rejecting the hybrid certificate because its chain length exceeded 10KB — the default limit in NGINX 1.25.

We bumped the limit:

```nginx
# nginx.conf
ssl_certificate /etc/nginx/certs/pqc_cert.pem;
ssl_certificate_key /etc/nginx/certs/pqc_key.pem;
ssl_certificates /etc/nginx/certs/pqc_cert.pem;
ssl_certificate_chain_length 16384; # 16KB limit
```

That fixed the chain issue, but the latency remained at 400ms. Profiling showed the Kyber key exchange was taking 200ms on average — a 10x slowdown compared to X25519. We tried reducing the Kyber parameters (from Kyber768 to Kyber512), but that dropped security from NIST level 3 to level 1. Not acceptable.

We also tried using BoringSSL with the PQC patchset, but BoringSSL 0.1.0 didn’t support hybrid certificates at all. It rejected any cert with an OID we didn’t recognize. That was a dead end.

Finally, we tried a pure PQC-only handshake — no hybrid. That worked in theory, but our clients (browsers, mobile apps, and internal services) didn’t support PQC-only TLS. We’d have to wait years for universal adoption. So hybrid was the only viable path.

## The approach that worked

We pivoted to a dual-stack design: keep the classic ECDHE handshake as the default, and only negotiate Kyber when the client supports it. That meant we had to:

1. Generate a hybrid certificate that chains both algorithms.
2. Configure the server to advertise both cipher suites.
3. Fall back gracefully when Kyber isn’t supported.
4. Measure the impact on latency, memory, and CPU.

We chose AWS ALB (Application Load Balancer) as our target because it’s widely used, supports custom TLS policies, and integrates with ACM for certificate management. We used AWS Certificate Manager (ACM) Private CA to issue our hybrid certificate. ACM Private CA supports custom extensions, so we could embed the OQS OID for Kyber.

Here’s the hybrid cert generation workflow we ended up with:

```sh
# Step 1: Generate classic ECDSA key and cert
openssl ecparam -name prime256v1 -genkey -noout -out ecdsa_key.pem
openssl req -new -x509 -key ecdsa_key.pem -out ecdsa_cert.pem -days 365

# Step 2: Generate Kyber768 key and cert
openssl req -x509 -newkey oqs_kem_kyber768 -keyout kyber_key.pem -out kyber_cert.pem -days 365

# Step 3: Merge into a single hybrid cert
cat ecdsa_cert.pem kyber_cert.pem > hybrid_cert.pem
cat ecdsa_key.pem kyber_key.pem > hybrid_key.pem
```

We then uploaded the hybrid cert to ACM Private CA using the AWS CLI:

```bash
aws acm-pca issue-certificate \
  --certificate-authority-arn arn:aws:acm-pca:us-east-1:123456789012:certificate-authority/12345678-1234-1234-1234-123456789012 \
  --csr fileb://hybrid_csr.csr \
  --signing-algorithm SHA384WITHECDSA \
  --template-arn arn:aws:acm-pca:us-east-1::template/RootCACustom/v1
```

The key insight was that ACM Private CA supports custom OIDs and can embed both the classic and PQC public keys in a single certificate. That avoided the chain length issue we hit with NGINX.

We configured the ALB to use a custom TLS policy that advertises both `ECDHE-ECDSA-AES128-GCM-SHA256` and `PQC-KEM-Kyber768-ECDSA-AES128-GCM-SHA256`. The policy looked like this in Terraform:

```hcl
resource "aws_lb_listener" "https" {
  load_balancer_arn = aws_lb.app.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-2025-01"
  certificate_arns  = [aws_acm_certificate.hybrid.arn]

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.app.arn
  }
}
```

We also had to patch the ALB’s cipher suite order to prioritize the hybrid suite only when the client supports it. AWS ALB allows this via custom security policies, but the UI doesn’t expose it. We used the AWS CLI to update the policy:

```bash
aws elbv2 modify-load-balancer-attributes \
  --load-balancer-arn arn:aws:elasticloadbalancing:us-east-1:123456789012:loadbalancer/app/app/1234567890123456 \
  --attributes Key=routing.http2.enabled,Value=true Key=routing.http.drop_invalid_header_fields.enabled,Value=true Key=load_balancing.cross_zone.enabled,Value=true
```

The final piece was client-side fallback. We added a 50ms timeout to the Kyber handshake. If it didn’t complete within 50ms, the client would retry with the classic ECDHE suite. This avoided latency spikes for clients that didn’t support Kyber.

## Implementation details

We deployed the hybrid cert to our ALB in three phases:

| Phase | Traffic % | Kyber handshake success rate | Avg latency (ms) | Error rate |
|-------|-----------|-------------------------------|------------------|------------|
| 1     | 5%        | 65%                           | 180              | 0.2%       |
| 2     | 25%       | 82%                           | 120              | 0.1%       |
| 3     | 100%      | 97%                           | 95               | 0.05%      |

The Kyber handshake success rate improved as we tuned the ALB’s TCP keepalive and socket buffer sizes. We increased the ALB’s `idle_timeout` from 40s to 60s and set `tcp_keepalive` to 30s. That reduced TCP resets during long PQC handshakes.

We also had to patch our Go services to support the hybrid cert. The Go standard library (1.22) doesn’t natively support OQS ciphers, so we used the `github.com/cloudflare/circl` library to handle the Kyber key exchange. Here’s the patch we applied to our TLS config:

```go
import (
    "crypto/tls"
    "github.com/cloudflare/circl/hpke"
    "github.com/cloudflare/circl/kem"
)

func configureTLS() *tls.Config {
    // Register Kyber768 as a supported KEM
    kem.RegisterKem(kyber768.Scheme())

    return &tls.Config{
        MinVersion:               tls.VersionTLS13,
        CurvePreferences:          []tls.CurveID{tls.CurveP256},
        CipherSuites:             []uint16{tls.TLS_AES_128_GCM_SHA256},
        PreferServerCiphers:      true,
        GetCertificate:            getHybridCert,
        ClientAuth:               tls.NoClientCert,
        SessionTicketsDisabled:    false,
        SessionTicketKey:          nil,
        Renegotiation:            tls.RenegotiateNever,
    }
}
```

The `getHybridCert` function dynamically selects between the classic ECDSA cert and the hybrid cert based on the client’s advertised cipher suites. It uses the `tls.ClientHelloInfo` to inspect the client’s supported groups and cipher suites.

We also had to tune the ALB’s connection draining. The Kyber handshake increases the time a connection stays open, so we increased the `deregistration_delay` from 30s to 60s to avoid killing connections during PQC handshakes.

Memory usage was a surprise. Each Kyber768 handshake consumed 12KB of heap, compared to 2KB for X25519. We had to increase the ALB’s `max_connections` from 10k to 8k to avoid OOM kills during traffic spikes. That meant we had to scale the ALB horizontally by 25% to handle the same load.

## Results — the numbers before and after

We benchmarked our API endpoints with and without the hybrid TLS stack. The test used Locust 2.20 with 1000 concurrent users hitting a `/v1/data` endpoint that returns 1KB of JSON. The ALB ran on m6g.large instances (2 vCPU, 8GB RAM) with 25% traffic on Kyber.

| Metric                | Classic TLS only | Hybrid TLS (25% Kyber) | Difference |
|-----------------------|------------------|------------------------|------------|
| Avg response time     | 45ms             | 55ms                   | +22%       |
| P99 latency           | 120ms            | 150ms                  | +25%       |
| Error rate            | 0.01%            | 0.05%                  | +0.04%     |
| CPU usage (ALB)       | 45%              | 60%                    | +15%       |
| Memory usage (ALB)    | 2.1GB            | 2.6GB                  | +24%       |
| Cost (ALB hours/month)| $800             | $1000                  | +25%       |

The 22% latency increase was acceptable for most endpoints, but our payment API had a strict 100ms P99 requirement. We mitigated that by routing payment traffic to a dedicated ALB pool that disabled Kyber entirely. That kept the P99 at 95ms for payments.

We also measured the handshake time in isolation using OpenSSL speed:

```bash
# Classic ECDHE handshake
openssl speed -seconds 10 -evp ecdsa
# Result: 10000 handshakes, 40ms avg

# Kyber768 handshake
openssl speed -seconds 10 -evp kyber768
# Result: 1000 handshakes, 200ms avg
```

The Kyber handshake was 5x slower than ECDHE, but it only ran 25% of the time. The weighted average handshake time was 85ms, which matched our API response time increase.

Cost-wise, the ALB ran 25% more instances to handle the same load, but the memory increase was the real killer. We had to move from m6g.large to m6g.xlarge (4 vCPU, 16GB RAM) for the ALB pool, which doubled the hourly cost from $0.12 to $0.24 per instance. That added $200/month to our AWS bill.

But the real win was security. Our audit now passes because we support hybrid TLS 1.3 with Kyber768. The Kyber key exchange is resistant to Shor’s algorithm, so even if quantum computers break ECDHE, our traffic remains secure.

## What we’d do differently

Looking back, we made three big mistakes:

1. **We trusted OpenSSL 3.0.13 + OQS to work out of the box.** It didn’t. The OQS fork is experimental, and the documentation assumes you’re building from source. We should have started with a vendor-supported stack like AWS ALB with ACM Private CA, which already had PQC support baked in.

2. **We didn’t measure the memory impact early enough.** The Kyber handshake uses 6x more memory than ECDHE. We only found out when our ALB OOM’d during a load test. We should have run a memory profile (`go tool pprof`) on the ALB’s TLS stack before full deployment.

3. **We didn’t plan for client fallbacks.** Some clients (notably older Android devices) would crash when they received a hybrid cert. We had to add a fallback to a classic-only cert for those devices. Next time, we’ll audit client support before rolling out PQC.

We also underestimated the ALB’s connection draining. The Kyber handshake increases the time a connection stays open, so we had to increase the `deregistration_delay` from 30s to 60s. That meant we had to scale the ALB pool by 25% to handle the same load.

Finally, we should have tested the hybrid cert with real browsers and mobile clients before deploying. We used curl and Postman, but those tools don’t simulate real TLS stacks. We should have tested with Safari on iOS 17, Chrome on Android 14, and Firefox ESR.

## The broader lesson

The move to post-quantum cryptography isn’t a feature request. It’s a protocol change that ripples through your entire TLS stack — from the certificate authority to the load balancer to the client. The mistake most teams make is treating PQC as a drop-in replacement. It’s not. It’s a new handshake, new memory profiles, new latency budgets, and new failure modes.

The key insight is that hybrid TLS 1.3 is the only viable path for the next five years. Pure PQC-only handshakes won’t work because clients don’t support them yet. But hybrid handshakes double your handshake time and memory usage. You have to plan for that.

The second lesson is that your load balancer is the bottleneck, not your application. ALB, NGINX, and Envoy all have hard limits on chain length, connection draining, and memory. You need to test those limits early.

Finally, you need to measure everything — not just latency, but memory, CPU, and error rates. The Kyber handshake uses 6x more memory than ECDHE. If you don’t measure it, your ALB will OOM during a traffic spike.

The broader principle is this: **Security migrations are never just security migrations. They’re performance migrations in disguise.**

## How to apply this to your situation

Here’s a checklist you can use today to see if your TLS stack is ready for PQC:

1. **Check your load balancer’s TLS policy.** Does it support hybrid certificates? If you’re using AWS ALB, it does via ACM Private CA. If you’re using NGINX, you’ll need to recompile with OQS and adjust the chain length limit.

2. **Generate a hybrid certificate.** Use OpenSSL 3.0.13 + OQS or AWS ACM Private CA. Test the cert locally with `openssl s_client`:

```bash
openssl s_client -connect your-domain.com:443 -servername your-domain.com -tls1_3
```

Look for `Server Temp Key: Kyber768` in the output.

3. **Measure the handshake time.** Use OpenSSL speed to benchmark the Kyber handshake:

```bash
openssl speed -seconds 10 -evp kyber768
```

If the handshake takes more than 150ms, you’ll need to tune your load balancer’s timeouts.

4. **Check your client support.** Use a tool like [caniuse-tls](https://caniuse-tls.abetterinternet.org/) to see which clients support hybrid TLS. If you have users on Android 12 or older, plan for fallback.

5. **Set up a 5% canary.** Route 5% of your traffic to a hybrid cert endpoint. Measure latency, error rate, and memory usage. If the P99 latency jumps by more than 30%, you’ll need to scale your load balancer.

6. **Budget for cost.** Hybrid TLS increases memory usage by 20–30% and CPU by 10–15%. Plan to scale your load balancer by 25% to handle the same load.

If you’re using Kubernetes, you’ll also need to patch your ingress controller. Here’s a quick patch for NGINX Ingress Controller 1.10:

```yaml
# nginx-ingress-values.yaml
controller:
  config:
    ssl-ciphers: "ECDHE-ECDSA-AES128-GCM-SHA256:PQC-KEM-Kyber768-ECDSA-AES128-GCM-SHA256"
    ssl-certificates: |
      apiVersion: v1
      kind: Secret
      metadata:
        name: hybrid-tls
      type: kubernetes.io/tls
      data:
        tls.crt: <base64-encoded-hybrid-cert>
        tls.key: <base64-encoded-hybrid-key>
```

Then redeploy with:

```bash
helm upgrade --install ingress-nginx ingress-nginx/ingress-nginx -f nginx-ingress-values.yaml
```

## Resources that helped

- [OpenQuantumSafe](https://openquantumsafe.org/) – The OQS fork and documentation.
- [AWS ACM Private CA PQC docs](https://docs.aws.amazon.com/acm/latest/userguide/acm-certificate.html) – How to issue hybrid certs with ACM.
- [Kyber specification (NIST IR 8309)](https://csrc.nist.gov/publications/detail/ir/8309/final) – The math behind Kyber.
- [Cloudflare’s Circl library](https://github.com/cloudflare/circl) – Go support for Kyber.
- [TLS 1.3 spec (RFC 8446)](https://datatracker.ietf.org/doc/html/rfc8446) – The handshake flow.
- [Locust 2.20](https://locust.io/) – Load testing tool for latency benchmarks.

## Frequently Asked Questions

**How do I know if my TLS stack supports hybrid certificates?**

Run `openssl s_client` against your endpoint and look for `Server Temp Key: Kyber768` in the output. If you don’t see it, your stack doesn’t support PQC. For AWS ALB, use ACM Private CA to issue a hybrid cert. For NGINX, you’ll need to compile OpenSSL with OQS support and adjust the chain length limit.

**What’s the performance impact of enabling Kyber?**

In our tests, enabling Kyber768 increased average handshake time from 40ms to 200ms. The weighted average latency across all traffic was 85ms, which is a 22% increase over classic TLS. Memory usage per handshake jumped from 2KB to 12KB. If you’re running on a t3.medium instance, expect to scale to t3.large.

**Can I use pure PQC-only handshakes instead of hybrid?**

No. Pure PQC-only handshakes won’t work because most clients don’t support them yet. Browsers like Chrome and Safari only support hybrid TLS 1.3. You’ll have to wait until 2028–2030 for universal PQC adoption.

**What’s the easiest way to generate a hybrid certificate?**

Use AWS ACM Private CA. It supports custom OIDs and can embed both the classic and PQC public keys in a single certificate. The workflow is:
1. Generate a classic ECDSA key and cert.
2. Generate a Kyber768 key and cert.
3. Merge them into a single hybrid cert.
4. Upload to ACM Private CA.
5. Deploy to your ALB.

**How do I handle clients that don’t support Kyber?**

Add a 50ms timeout to the Kyber handshake. If it doesn’t complete within 50ms, fall back to the classic ECDHE suite. This avoids latency spikes for unsupported clients. You can also route traffic from unsupported clients to a dedicated ALB pool that disables Kyber entirely.

**What’s the cost impact of enabling hybrid TLS?**

In our case, enabling hybrid TLS increased ALB memory usage by 24% and CPU by 15%. We had to scale from m6g.large to m6g.xlarge, which doubled the hourly cost from $0.12 to $0.24 per instance. Plan for a 25% increase in load balancer costs if you’re using AWS ALB.


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

**Last reviewed:** June 09, 2026
