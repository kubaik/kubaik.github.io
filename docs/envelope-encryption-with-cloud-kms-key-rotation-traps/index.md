# Envelope encryption with cloud KMS: key rotation traps

Envelope encryption is the standard way to encrypt application data with a cloud KMS key. It is also the place where key rotation failures hide: a deployment looks healthy, then old records stop decrypting after a key change. The failure is rarely in the cryptography itself. It is almost always in what was stored alongside the ciphertext.

This article covers the key hierarchy, a working Python implementation, the concrete rotation traps, and how to test the whole path before it reaches production.

## Why encrypting data directly with a KMS key is a dead end

Managed KMS services are built to protect small payloads. AWS KMS, for example, documents a 4 KB limit on the plaintext passed to its symmetric `Encrypt` operation. They are not designed to encrypt database rows, files, or messages directly.

The reasons are technical and economic:

- **Network cost per operation.** Every `Encrypt`/`Decrypt` call is a round-trip to a highly available, audited service, billed per request. Encrypting a 1 MB file in 4 KB chunks would require roughly 256 KMS calls per file. That arithmetic is simple and the result is slow and expensive at any volume.
- **Availability coupling.** KMS keys are regional resources. If every read path calls KMS, a KMS outage becomes an application outage.
- **Throughput ceilings.** KMS request quotas are real. Bulk data encryption will hit them.

Envelope encryption removes KMS from the hot path. A data key is generated locally or by KMS, used to encrypt the payload with a fast local cipher, and then discarded. Only the data key is wrapped by KMS. To decrypt, the wrapped key is unwrapped once, then the payload is decrypted locally.

The pattern:

1. Obtain a data key (plaintext plus a wrapped copy).
2. Encrypt the payload locally with the plaintext data key.
3. Store the wrapped data key with the ciphertext.
4. Discard the plaintext data key from memory.
5. To decrypt, unwrap the data key via KMS, then decrypt locally.

The data key is sealed inside the KMS key. That seal is the part rotation affects, and it is where most deployments break.

## Prerequisites: IAM, key policies, and a local crypto library

Before writing code, confirm the environment has the necessary permissions and libraries:

- An IAM principal with `kms:GenerateDataKey` and `kms:Decrypt` on the KMS key. `GenerateDataKey` creates the data key; `Decrypt` unwraps it. `kms:Encrypt` is not needed for the `GenerateDataKey` flow.
- A symmetric KMS key for encryption and decryption. Record the key ARN. AWS KMS supports automatic annual rotation, which creates new key material while keeping the same key ARN.
- A local cryptographic library. In Python, `cryptography` is standard. In Go, `crypto/aes` and `crypto/cipher` in the standard library are sufficient.

Install dependencies:

```bash
pip install boto3 cryptography
```

In production, use IAM roles rather than long-lived access keys. For local testing, any credentials with the permissions above work.

## Step 1: Generate a data key with KMS

`GenerateDataKey` returns both a plaintext data key and a ciphertext blob (the wrapped key). The plaintext key is used only in memory; the wrapped blob is what gets persisted.

```python
import boto3

kms = boto3.client('kms')
key_id = 'arn:aws:kms:us-east-1:123456789012:key/abcd1234-...'  # your KMS key ARN

def generate_data_key(key_id):
    response = kms.generate_data_key(
        KeyId=key_id,
        KeySpec='AES_256'  # or 'AES_128'
    )
    return response['Plaintext'], response['CiphertextBlob'], response['KeyId']
```

The response includes `KeyId`, the ARN of the key that produced the data key. AWS KMS also embeds the key ID inside the ciphertext blob, so a bare `Decrypt` call can resolve the correct key. Storing the key ID separately is still worth doing: it makes records self-describing for audits, cross-account access, and disaster recovery, and it lets you validate that a record was produced by the key you expect.

## Step 2: Encrypt your data locally with the data key

AES-256-GCM is a reasonable default because it provides authenticated encryption. The one rule that cannot be broken: generate a unique nonce for every encryption operation under the same key. Reusing a nonce with GCM leaks plaintext relationships and destroys authentication guarantees.

```python
import os
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

def encrypt_data(plaintext, data_key):
    aesgcm = AESGCM(data_key)
    nonce = os.urandom(12)  # 96-bit nonce for GCM
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)
    return nonce + ciphertext  # prepend nonce for storage
```

The output is the 12-byte nonce followed by the ciphertext and authentication tag. The nonce does not need to be secret, but it must be stored with the ciphertext and must never repeat for a given key.

If the payload is large, stream it through AES-GCM rather than loading it whole, and still generate a fresh nonce per message. If you need to encrypt multiple fields under one data key, either use a fresh nonce per field or derive per-field subkeys; never reuse a nonce.

## Step 3: Store the wrapped key and ciphertext together

Three things must be persisted: the encrypted data (nonce plus ciphertext), the wrapped data key, and the KMS key ID. Optionally store the algorithm and a format version for future migrations.

```python
import json
import base64

def package_encrypted_data(nonce_ciphertext, wrapped_key, kms_key_id):
    return json.dumps({
        'data': base64.b64encode(nonce_ciphertext).decode('utf-8'),
        'wrapped_key': base64.b64encode(wrapped_key).decode('utf-8'),
        'kms_key_id': kms_key_id
    })
```

In a database, these can be separate columns (`ciphertext`, `wrapped_key`, `kms_key_id`) or a single serialized blob. Separate columns make it easier to query and re-wrap keys later without touching the payload.

## Step 4: Decrypt by unwrapping the data key with KMS

Decryption reverses the process: unwrap the data key with KMS, then decrypt locally.

```python
def decrypt_data(package):
    obj = json.loads(package)
    wrapped_key = base64.b64decode(obj['wrapped_key'])
    kms_key_id = obj['kms_key_id']
    response = kms.decrypt(
        CiphertextBlob=wrapped_key,
        KeyId=kms_key_id  # optional; acts as a safety check
    )
    data_key = response['Plaintext']
    nonce_ciphertext = base64.b64decode(obj['data'])
    nonce = nonce_ciphertext[:12]
    ciphertext = nonce_ciphertext[12:]
    aesgcm = AESGCM(data_key)
    plaintext = aesgcm.decrypt(nonce, ciphertext, None)
    return plaintext
```

The `KeyId` parameter in `Decrypt` is optional. When provided, KMS verifies that the ciphertext was produced under that key and fails otherwise. That check is useful, but it is also the source of one of the traps below.

## The rotation trap: what actually breaks

Automatic rotation on a symmetric KMS key creates new backing key material while keeping the same key ARN, and KMS retains older versions for decryption. A wrapped data key produced before rotation still unwraps correctly after rotation, because the ciphertext blob identifies the version that produced it. So annual automatic rotation, by itself, does not break envelope encryption.

The real traps are the ones that change the key identity or the reference used at decrypt time:

**1. Replacing the key instead of rotating it.** If a team creates a new KMS key (new ARN) and switches the application to it, data wrapped under the old key still decrypts — but only if the old key remains enabled and the record points at it. If the old key is scheduled for deletion or its policy is tightened, old records become unreadable. Deletion in AWS KMS is irreversible after the waiting period, so this is a data-loss event, not a bug.

**2. Hardcoding the current key ARN in the decrypt path.** If `decrypt_data` passes a hardcoded `KeyId` rather than the stored one, every record wrapped under a different key fails with `InvalidCiphertextException`, even though the correct key is still available. The stored key ID is what makes old records decryptable.

**3. Storing an alias instead of a key ARN.** Aliases are pointers. An alias can be repointed to a different key, so a record that stores only the alias may later resolve to the wrong key. Store the key ARN or key ID, not the alias.

**4. Cross-region and cross-account replication.** KMS keys are regional. Copying ciphertext to another region without the wrapped key, or without a key available there, produces undecryptable data. Multi-Region keys exist specifically to keep key material consistent across regions; otherwise the record must be re-wrapped with a key in the destination region as part of the copy.

**5. Deleting or disabling old keys during cleanup.** A rotation cleanup that disables old key versions or schedules old keys for deletion looks tidy and silently breaks historical data. Old keys must remain enabled for as long as any record depends on them.

The common thread: the wrapped key and the key identity must travel with the ciphertext, and the decrypt path must use the stored identity rather than a globally configured one.

## Edge cases: key policies, grants, and HSM-backed keys

**Key policy versus IAM policy.** In AWS KMS, the key policy is the primary access control. An IAM policy granting `kms:Decrypt` does not help if the key policy does not permit the principal. `AccessDeniedException` on decrypt usually means one of the two was missed; check both.

**Grants.** Grants delegate permissions without editing the key policy. They are useful for temporary access, but they must be revoked when no longer needed. A stale grant is a standing permission.

**HSM-backed keys.** Keys in a custom key store backed by a hardware security module keep key material inside the HSM. Envelope encryption works identically, but expect additional latency and cost, and plan disaster recovery around the HSM cluster, since key material cannot be exported.

**Data key caching.** Some SDKs offer data key caching to reduce KMS calls. This trades a larger exposure window for lower latency and cost. If used, keep the TTL short and cap the number of uses per cached key.

## Verification: test rotation and decryption

Rotation behavior should be tested explicitly rather than assumed. The test has three parts: encrypt, change the key, decrypt both old and new records.

```python
plaintext = b"secret message"
data_key, wrapped_key, key_id = generate_data_key(key_id)
nonce_ciphertext = encrypt_data(plaintext, data_key)
package = package_encrypted_data(nonce_ciphertext, wrapped_key, key_id)

# Change the key here: enable automatic rotation, or create a new key
# and update the application's configured key ARN.

# Old record must still decrypt.
assert decrypt_data(package) == plaintext

# New record must use the current key.
new_data_key, new_wrapped_key, new_key_id = generate_data_key(key_id)
new_nonce_ciphertext = encrypt_data(b"new secret", new_data_key)
new_package = package_encrypted_data(new_nonce_ciphertext, new_wrapped_key, new_key_id)
assert decrypt_data(new_package) == b"new secret"
```

Two notes on this test. First, AWS KMS automatic rotation is annual and there is no `rotate-key` command; to force a new version in a test, use a separate test key or wait for the rotation schedule. Second, a local KMS emulator can exercise the `GenerateDataKey` and `Decrypt` paths without touching production keys, but confirm that the emulator models key versions, since that is the behavior under test.

What to instrument in a real deployment:

- **Decrypt failures by error type.** A rise in `InvalidCiphertextException` after a key change is the signature of trap 2.
- **Key ID distribution in stored records.** A query grouping records by stored key ID shows how many depend on old keys.
- **Key state.** Track which keys are enabled, disabled, or pending deletion, and alert when a key with dependent records is scheduled for deletion.

The comparison to make is simple: for each stored key ID, count records, and confirm the key is enabled. Any key pending deletion with a nonzero record count is a data-loss risk.

## Choosing between cloud KMS and self-managed HSM

The envelope pattern is the same either way; the choice is about control, operations, and compliance.

| Aspect | Cloud KMS | Self-managed HSM |
|--------|-----------|------------------|
| Operational burden | Low; provider manages availability and patching | High; patching, HA, and backups are yours |
| Cost model | Per request and per key | Capital expense plus operational cost |
| Key material control | Provider holds material unless BYOK/HYOK is used | You hold material |
| Compliance | Meets many standards, with published validation | Depends on the module |
| Latency | Network round-trip | Local if on-premises |
| Scalability | High | Limited by hardware |

Use cloud KMS when operational overhead matters more than holding key material directly. Use a self-managed HSM when compliance or policy requires full control of key material. Envelope encryption applies to both, and the rotation traps are the same shape: store the wrapped key and the key identity with the ciphertext, and never let the decrypt path guess.

## Frequently Asked Questions

**How does envelope encryption work with AWS KMS?**

`GenerateDataKey` returns a plaintext data key and a copy wrapped under the KMS key. The plaintext key encrypts the payload locally and is then discarded. The wrapped key is stored with the ciphertext. To decrypt, `Decrypt` unwraps the data key, and the payload is decrypted locally. Large payloads never pass through KMS.

**What happens when a KMS key is rotated?**

Automatic rotation creates new backing key material under the same key ARN. KMS retains older versions for decryption, so data wrapped before rotation still decrypts. Problems arise when a new key replaces the old one and the old key is disabled or deleted, or when the decrypt path hardcodes a key ID instead of using the stored one.

**Can I use envelope encryption with an HSM?**

Yes. Envelope encryption is a pattern, not a product. With an HSM, the data key is generated or wrapped using the HSM (for example via PKCS#11), the payload is encrypted locally, and the wrapped key is stored. Cloud KMS services are managed HSMs, so the pattern is identical.

**Why do I get InvalidCiphertextException when decrypting?**

Common causes: the ciphertext blob is corrupted or truncated, the wrong key is specified, or the decrypt call passes a hardcoded key ID that does not match the key that produced the record. If a new key replaced an old one, records wrapped under the old key fail unless the old key ID is used and the old key remains enabled.

## Next step: audit your key IDs

In the next 30 minutes, search the codebase for every KMS decrypt call and check what `KeyId` it passes. If it is hardcoded or comes from configuration rather than from the stored record, that is trap 2 waiting to fire. Change it to read the key ID stored alongside the ciphertext, then run the encrypt-rotate-decrypt test above against a non-production key to confirm old records still decrypt.
