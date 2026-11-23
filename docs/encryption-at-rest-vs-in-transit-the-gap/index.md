# Encryption at rest vs in transit: the gap

Encryption at rest and encryption in transit are often treated as two checkboxes on a compliance form. In reality, they protect against different threats, fail in different ways, and cost different amounts to operate. The gap between what the documentation promises and what actually happens under load is where most teams get surprised.

Consider a typical microservice that stores user records in a database. The database encrypts the disk (encryption at rest), and the service connects over TLS (encryption in transit). Both boxes are ticked. Yet a misconfigured client that disables certificate verification, or a backup that gets copied to an unencrypted staging bucket, can silently break the guarantee. The problem is not that the primitives are weak; it is that the operational boundaries are fuzzy.

This post examines the tradeoffs between encryption at rest and in transit, focusing on where implementations diverge from expectations. It covers the mechanisms, a concrete implementation with code, how to measure effectiveness, failure modes, and when each approach is the wrong tool. By the end, you will have a decision framework and a specific action to take in the next 30 minutes.

## Where the documentation ends and reality begins

Official docs for cloud providers and databases describe encryption at rest as "data encrypted on disk using AES-256" and in transit as "TLS 1.2+ enforced." What they often omit is the fine print: encryption at rest protects against physical disk theft but not against a compromised application with valid credentials. Encryption in transit protects against network eavesdropping but not against a misconfigured client that falls back to plaintext.

A common misconception is that enabling encryption at rest automatically encrypts backups. In many managed databases, automated backups are encrypted with the same key, but manual snapshots exported to object storage may not inherit that setting. Similarly, encryption in transit for service-to-service communication often relies on mutual TLS (mTLS), but many teams enable it only at the ingress and leave internal traffic unencrypted.

The gap widens when you consider key management. Encryption at rest is only as strong as the key management system (KMS). If the KMS key is stored alongside the data (e.g., in the same S3 bucket), the encryption is effectively defeated. Encryption in transit depends on certificate authority (CA) trust chains; a compromised CA or a misissued certificate can break the guarantee.

The real problem this post solves is: how to reason about the tradeoffs between encryption at rest and in transit so you can allocate effort where it actually reduces risk, rather than where it satisfies an auditor.

## The mechanisms: what each actually protects

Encryption at rest protects data when it is not moving: on disk, in backups, in object storage. It mitigates threats like physical media theft, improper disposal of drives, and unauthorized access to storage snapshots. The mechanism is usually block-level encryption (e.g., LUKS for Linux, BitLocker for Windows) or application-level encryption (e.g., encrypting specific fields before storing).

Encryption in transit protects data as it moves between systems: client to server, service to service, database to application. It mitigates eavesdropping, man-in-the-middle (MITM) attacks, and data tampering. The mechanism is TLS (or its predecessor SSL), which provides confidentiality and integrity through symmetric encryption (e.g., AES-GCM) and asymmetric key exchange (e.g., ECDHE).

The tradeoff is not either/or; they are complementary. But they have different cost profiles. Encryption at rest adds CPU overhead for encryption/decryption at the storage layer, but modern CPUs with AES-NI make this negligible. Encryption in transit adds latency due to the TLS handshake and encryption overhead, plus operational complexity for certificate management.

A key difference is the attack surface. Encryption at rest is static: once data is encrypted, it stays encrypted until decrypted by an authorized process. Encryption in transit is dynamic: every connection must be secured, and a single misconfigured client can downgrade to plaintext. This is why in-transit encryption often fails silently.

Consider the threat models:

| Threat | Encryption at rest | Encryption in transit |
|--------|-------------------|----------------------|
| Physical disk theft | Protects | No effect |
| Network eavesdropping | No effect | Protects |
| Compromised application | No effect (app has keys) | No effect (app terminates TLS) |
| Misconfigured client | No effect | Can fail open (plaintext) |
| Backup leakage | Protects if backups encrypted | No effect |
| Insider access to storage | Protects if keys segregated | No effect |

The table shows that neither alone is sufficient. The tradeoff is in where you invest: at-rest encryption is mostly a one-time configuration with ongoing key management, while in-transit encryption requires continuous monitoring and certificate lifecycle management.

## Implementing both without the common pitfalls

A typical implementation uses a managed database with encryption at rest enabled and TLS enforced for connections. But the devil is in the details. For example, PostgreSQL supports SSL/TLS connections, but the default `ssl` setting is `off` in many distributions. You must set `ssl = on` in `postgresql.conf` and configure `ssl_cert_file` and `ssl_key_file`. Even then, clients can choose not to use SSL unless you enforce it via `hostssl` entries in `pg_hba.conf`.

Here is a minimal example of enforcing TLS in PostgreSQL:

```conf
# postgresql.conf
ssl = on
ssl_cert_file = '/etc/ssl/certs/server.crt'
ssl_key_file = '/etc/ssl/private/server.key'
```

```conf
# pg_hba.conf
# TYPE  DATABASE        USER            ADDRESS                 METHOD
hostssl all             all             0.0.0.0/0               scram-sha-256
```

The `hostssl` directive ensures that only SSL connections are allowed. Without it, clients can connect without encryption.

For encryption at rest, if you are using a cloud provider's managed database, it is often a checkbox. But if you are self-hosting, you might use LUKS for disk encryption. Here is a simplified example of encrypting a block device with LUKS:

```bash
# Create a new encrypted volume (simplified)
cryptsetup luksFormat /dev/sdb1
cryptsetup open /dev/sdb1 encrypted_volume
mkfs.ext4 /dev/mapper/encrypted_volume
mount /dev/mapper/encrypted_volume /mnt/data
```

This encrypts the entire volume. But note: the key is derived from a passphrase or key file. If the key is stored on the same server, the encryption is only effective against physical theft, not against a compromised OS.

A more robust approach for application-level encryption at rest is to encrypt specific fields before storing them, using a library like `cryptography` in Python. Here is a runnable example that encrypts a field using AES-GCM:

```python
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import os
import base64

key = AESGCM.generate_key(bit_length=256)
aesgcm = AESGCM(key)
nonce = os.urandom(12)
plaintext = b"sensitive data"
ciphertext = aesgcm.encrypt(nonce, plaintext, None)
# Store nonce + ciphertext
encoded = base64.b64encode(nonce + ciphertext).decode('utf-8')
print(encoded)
```

This approach gives you control over keys, but it shifts the burden of key management to you. You must store the key securely, rotate it, and ensure it is not logged.

For encryption in transit, enforcing TLS on the client side is equally important. In Go, for example, the default `http.Client` verifies certificates, but many teams disable verification for testing and forget to re-enable it. Here is a correct configuration:

```go
import (
    "crypto/tls"
    "net/http"
)

func main() {
    tlsConfig := &tls.Config{
        MinVersion: tls.VersionTLS12,
    }
    client := &http.Client{
        Transport: &http.Transport{
            TLSClientConfig: tlsConfig,
        },
    }
    // Use client
}
```

Setting `MinVersion` to TLS 1.2 or higher prevents downgrade attacks. Never set `InsecureSkipVerify: true` in production.

## Measuring what you actually get

You cannot measure encryption directly, but you can measure its absence. For in-transit encryption, monitor for plaintext connections. In PostgreSQL, you can query `pg_stat_ssl` to see which connections are using SSL:

```sql
SELECT pid, ssl, version, cipher FROM pg_stat_ssl;
```

If `ssl` is false for any connection, you have a gap. For encryption at rest, you can verify that disk encryption is active using tools like `cryptsetup status` or cloud provider APIs. But measuring key management hygiene is harder: check for keys stored in code, keys with overly permissive access, and rotation policies.

A practical metric is the percentage of connections using TLS. In a service mesh like Istio, you can use telemetry to see the percentage of mTLS traffic. For example, the `istio_requests_total` metric with `connection_security_policy` label can show `mutual_tls` vs `none`.

Another metric is the time to rotate a key or certificate. If rotation takes days, your encryption is brittle. Automate rotation with tools like cert-manager for Kubernetes or AWS Certificate Manager.

## Failure modes: where encryption breaks

The most common failure mode for encryption in transit is the fallback to plaintext. Many protocols and libraries allow downgrade for compatibility. For example, SMTP with STARTTLS can be stripped by an attacker, resulting in plaintext email. Similarly, some database drivers will connect without TLS if the server does not support it, unless you explicitly require it.

Another failure mode is certificate expiry. An expired certificate causes connection failures, but a more insidious problem is a certificate that is valid but issued by an untrusted CA. If the client does not properly validate the certificate chain, it may accept a MITM certificate.

For encryption at rest, the failure mode is key compromise. If the key is stored in the same location as the data, or if access to the KMS is too broad, encryption provides little protection. A related failure is incomplete encryption: encrypting the database but not the backups, or encrypting the primary disk but not the swap file.

A subtle failure mode is the confusion between encryption at rest and encryption in use. Encryption at rest does not protect data while it is being processed in memory. If an attacker gains access to the application's memory, they can read plaintext. This is why some teams use enclaves (e.g., AWS Nitro Enclaves) for sensitive operations, but that is beyond the scope of this post.

## When encryption at rest or in transit is the wrong choice

Encryption is not free. It adds CPU overhead, latency, and operational complexity. In some cases, the tradeoff is not worth it.

For example, encrypting data at rest in a development environment with no sensitive data adds unnecessary complexity. Similarly, enforcing TLS for internal health checks between services in a trusted network may add latency without meaningful security benefit, especially if the network is already isolated.

However, the more common mistake is under-encrypting. The decision should be based on threat modeling. If the data is sensitive (PII, credentials, financial records), encrypt both at rest and in transit. If the data is public, encryption may still be required for compliance, but the risk is lower.

A wrong choice is relying solely on encryption at rest for data that is transmitted over untrusted networks. For example, storing encrypted backups in cloud storage but transferring them over FTP without TLS. The encryption at rest is irrelevant if the data is intercepted in transit.

Another wrong choice is using encryption in transit as a substitute for encryption at rest. TLS protects data in transit, but once it reaches the server, it is decrypted and stored in plaintext on disk. If the disk is stolen, the data is exposed.

The counterintuitive part is that encryption at rest can sometimes increase risk if it leads to a false sense of security. Teams may assume that because the disk is encrypted, they can be lax about access controls. But encryption at rest does not protect against a compromised application with valid credentials.

## Frequently Asked Questions

**Does encryption at rest protect data in backups?**
It depends on how backups are configured. In many managed databases, automated backups inherit the encryption settings of the primary instance. However, manual snapshots exported to object storage may not be encrypted unless you explicitly enable encryption on the storage bucket. Always verify backup encryption separately.

**Is TLS 1.2 still secure for encryption in transit?**
TLS 1.2 is considered secure when configured with strong cipher suites and proper certificate validation. However, TLS 1.3 is preferred because it removes legacy features and reduces handshake latency. As of 2026, most major providers support TLS 1.3, and it is recommended to enable it where possible.

**Can I use encryption at rest without a KMS?**
Yes, you can use application-level encryption with keys stored in a file or environment variable, but this is less secure. A KMS provides centralized key management, audit logging, and rotation. For production systems, a KMS is strongly recommended.

**What is the performance impact of encryption in transit?**
The impact is typically small with modern hardware. TLS handshakes add latency on connection establishment, but session resumption and TLS 1.3 reduce this. The encryption itself uses AES-GCM, which is hardware-accelerated on most CPUs. The overhead is usually negligible compared to network latency.

## Next step: audit your TLS enforcement in 30 minutes

Open your database configuration file (e.g., `pg_hba.conf` for PostgreSQL) and check for `hostssl` entries. If you see `host` instead of `hostssl`, clients can connect without encryption. Change at least one entry to `hostssl` and reload the configuration. Then, run a query against `pg_stat_ssl` to confirm that new connections use SSL. This single change closes a common gap between documented encryption in transit and actual enforcement.

## Official documentation

- [PostgreSQL documentation](https://www.postgresql.org/docs/)
- [Kubernetes documentation](https://kubernetes.io/docs/home/)

---

### About this article

**Written by:** [Kubai Kevin](/about/) — software developer based in Nairobi, Kenya.

**How this article was produced:** This site uses an automated LLM pipeline designed and maintained by the author. Before publication, drafts are screened automatically for minimum length, near-duplicate content, invented or unsourced claims and citations, unverifiable product names, and broken internal links. Individual line-by-line human editing is not performed on every post. Figures, benchmarks and scenarios are illustrative unless a source is linked; verify them against official documentation before relying on them in production. See the [AI content policy](/ai-content-policy/).

**Corrections:** Report errors via the [contact page](/contact/). Corrections are applied promptly.

**Last generated:** November 2026
