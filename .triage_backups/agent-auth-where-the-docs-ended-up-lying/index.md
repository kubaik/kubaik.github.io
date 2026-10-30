# Agent auth: where the docs ended up lying

Agent authentication looks simple on paper: issue a token, verify a signature, rotate keys on a schedule. That model holds up when the agent runs on a cloud VM with a stable clock, uninterruptible power, and a network that does not drop packets mid-handshake. It degrades in interesting ways when the agent is a USSD gateway on a battery-backed single-board computer, a feature phone with no real-time clock, or a device sitting behind a carrier gateway that rewrites your payload.

This article walks through the three layers that actually determine whether agent authentication survives production — device anchoring, network-level binding, and token lifecycle design — then gives a worked implementation, a failure-mode analysis, and a decision checklist for when the standard advice is the wrong choice.

## Why the documented stack assumes more than you have

The standard guidance for public clients is OAuth2 with PKCE, short-lived access tokens, and periodic key rotation. That guidance is correct. It simply carries hidden preconditions:

- A clock that is accurate to within the token's validity window.
- Storage that an attacker with physical access cannot trivially read.
- A transport that delivers your full payload without truncation or rewriting.
- A runtime you control end to end.

Each precondition maps to a specific failure mode when violated. Clock drift breaks `exp` and `iat` validation. Removable or readable storage breaks key confidentiality. SMS and USSD transports impose hard payload limits that fragment or truncate tokens. Carrier gateways may rewrite headers. None of these are exotic in constrained deployments; they are the default.

The useful mental model is that agent authentication is not one problem but three, and each has a different anchor of trust.

## Layer 1: Device anchoring

An agent must prove it is running on a specific physical device, not merely that it holds a valid token. Candidate anchors, with honest tradeoffs:

- **SIM ICCID.** Immutable for the life of the SIM, survives factory resets, does not survive a SIM swap. This is a strong anchor against device cloning but a weak one against subscriber takeover.
- **IMEI.** Nominally unique, practically spoofable. Grey-market handsets and modems are commonly reprogrammed, so IMEI alone should never be treated as a unique device identity.
- **Trusted execution environment (TEE) root of trust.** A tamper-resistant boot chain on ARMv8 and later. Not a full hardware security module; adequate for holding a signing key that survives power loss.
- **Fused one-time-programmable (OTP) secrets.** Some single-board compute modules expose a secret fused into the SoC at manufacture. This is not an HSM, but it is a reasonable place to bind a key.

A practical composite device identifier is `SHA-256(IMEI || ICCID || last-4-of-MSISDN)`. The reasoning: IMEI alone is spoofable, ICCID alone changes on SIM swap, and the MSISDN suffix adds a subscriber-side component that is cheap to collect. Combining them means an attacker must control all three simultaneously.

Two caveats worth stating plainly. First, hashing identifiers does not make them secret — ICCID and IMEI have low entropy and are guessable or enumerable. Hashing here is for correlation and privacy, not for security. Second, a composite identifier is only as strong as its weakest component; if IMEI is trivially cloned, the composite is effectively ICCID plus a subscriber suffix.

## Layer 2: Network-level binding

Binding identity to the network session, rather than only to the device, catches subscriber takeover that device anchoring misses. The relevant carrier-side identifiers are the IMSI and the MSISDN, both of which are sensitive personal data under most privacy regimes.

The design pattern that works: store a keyed hash of the IMSI, using a per-carrier salt held in your own key management, never the raw value. This lets you detect a change in the subscriber identity associated with a device without retaining the identifier itself. The salt must be treated as a secret; if it leaks, the hashes become enumerable because IMSI space is structured and finite.

A token request from a USSD agent might carry:

```json
{
  "sub": "agent:ussd:ke:carrier:254712345678",
  "iss": "https://auth.example.net/agents",
  "jti": "a1b2c3d4e5f6",
  "iat": 1717020800,
  "exp": 1717021100,
  "device": {
    "iccid": "8962000000000000001",
    "imei_hash": "sha256:...",
    "imsi_hash": "sha256:...",
    "salt_id": "carrier-a-v3"
  }
}
```

Note `salt_id` rather than the salt itself. The salt lives server-side; the token only references which salt version was used so rotation is possible without invalidating in-flight tokens.

## Layer 3: Token lifecycle design

Short-lived tokens and periodic key rotation are correct defaults. On constrained transports they collide with payload limits. A JWT signed with RSA-2048 and carrying `iss`, `sub`, `iat`, `exp`, `jti`, plus five custom claims is already in the high hundreds of bytes. Adding a nonce and several device identifiers pushes it past a kilobyte.

USSD and SMS impose much smaller practical limits. SMS is 140 bytes of payload in a single GSM-7 message, or 160 characters for a 7-bit message; concatenation is possible but multiplies loss probability. USSD session payloads are smaller still and vary by carrier.

The pattern that resolves this is the **split token**:

- A short, opaque **reference token** (for example 16 bytes, base64url-encoded) travels over the constrained channel.
- The full **payload token** stays server-side or in a local cache.
- The agent exchanges the reference token for the payload token over a channel that can carry more bytes.

This reduces on-air size dramatically, at the cost of an extra round trip. That round trip introduces a new attack surface: if the reference token alone is sufficient to fetch the payload, a replayed reference token yields the payload. The mitigation is to require a fresh device-bound signature on every fetch, so possession of the reference token is not sufficient.

```http
POST /token/fetch HTTP/1.1
Host: auth.example.net
Content-Type: application/json

{
  "ref": "a1b2c3d4e5f6",
  "sig": "ed25519:...",
  "device_id": "sha256:..."
}
```

The server verifies the signature over a server-issued challenge, checks the device binding, and only then returns the payload. A static signature over a fixed message is not enough; it must cover a nonce or timestamp to prevent replay.

## Worked example: sizing a token against a transport limit

Suppose a design has a payload token of 900 bytes and the transport is single-message SMS with a 140-byte payload limit. Naive delivery requires `ceil(900 / 140) = 7` concatenated messages.

Assume, illustratively, that each individual SMS has a 3% delivery failure probability and that concatenated messages fail if any segment fails. The probability a 7-segment message is delivered intact is `0.97^7 ≈ 0.809`, so about 19% of OTP deliveries fail. For a 2-segment message it is `0.97^2 ≈ 0.941`, about 6% failure. For a single message it is 3%.

That arithmetic is illustrative, not measured; substitute your own carrier's observed per-segment failure rate. The point is that fragmentation cost compounds, which is why shrinking the payload matters more than it first appears.

Now apply the split-token design. The reference token is 16 bytes, well within a single message. The 900-byte payload token is fetched over a data channel, where the failure mode is a retry, not a lost OTP. The transport-level failure probability drops to the single-message case.

**How to measure this for your deployment:** instrument the send path to record, per message, the number of segments and whether the delivery receipt arrived within the timeout. Aggregate to get per-segment failure rate and per-message intact-delivery rate. Compare against the arithmetic above to confirm the model holds.

## A minimal implementation

The following is a reference implementation for an agent running on a 64-bit ARM single-board computer acting as a USSD gateway. It uses Python 3.11, FastAPI, Ed25519 via a standard library, and Redis for the token cache.

```bash
sudo apt update
sudo apt install -y python3.11 python3-pip redis-server
pip install fastapi uvicorn redis pynacl pyjwt
```

### Device anchor extraction

```python
# agent/device.py
import hashlib
import serial


def get_sim_iccid(port='/dev/ttyUSB2'):
    with serial.Serial(port, 115200, timeout=1) as ser:
        ser.write(b'AT+CICCID\r')
        line = ser.readline().decode().strip()
        if line.startswith('+ICCID:'):
            return line.split(':', 1)[1].strip()
    return None


def get_imei(port='/dev/ttyUSB2'):
    with serial.Serial(port, 115200, timeout=1) as ser:
        ser.write(b'AT+CGSN\r')
        line = ser.readline().decode().strip()
        if line.isdigit() and len(line) == 15:
            return line
    return None


def hash_identifier(value, salt_id, salt_value):
    """Keyed hash of a low-entropy identifier.

    The salt is a secret held server-side; it is passed in here only
    because this process runs in the trusted part of the agent.
    """
    material = f"{salt_id}:{salt_value}:{value}".encode()
    return hashlib.sha256(material).hexdigest()
```

Two things to note. The AT command responses must be validated, not trusted blindly — a modem can return stale or malformed data. And the keyed hash is only as strong as the salt's confidentiality.

### Token generation with Ed25519

```python
# agent/token.py
import os
import time
import secrets

import jwt
from nacl import signing

KEY_PATH = '/opt/agent/keys/ed25519.key'
ISSUER = 'https://auth.example.net/agents'
SECRET = os.environ['AGENT_JWT_SECRET']  # fail loudly if unset


def load_or_create_signing_key():
    if os.path.exists(KEY_PATH):
        with open(KEY_PATH, 'rb') as f:
            return signing.SigningKey(f.read())
    os.makedirs(os.path.dirname(KEY_PATH), exist_ok=True)
    key = signing.SigningKey.generate()
    write_key_atomically(KEY_PATH, key.encode())
    return key


def write_key_atomically(path, data):
    """Write to a temp file, fsync, then rename.

    A rename is atomic on POSIX filesystems, so a power loss mid-write
    leaves either the old key or the new key, never a truncated file.
    """
    tmp = f"{path}.tmp"
    with open(tmp, 'wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    dir_fd = os.open(os.path.dirname(path), os.O_DIRECTORY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def generate_tokens(signing_key, device, ttl_seconds=90):
    ref_token = secrets.token_urlsafe(12)  # ~16 chars, opaque

    payload = {
        'sub': 'agent:ussd:ke:carrier:254712345678',
        'iss': ISSUER,
        'jti': secrets.token_urlsafe(16),
        'iat': int(time.time()),
        'exp': int(time.time()) + ttl_seconds,
        'device': {
            'iccid': device['iccid'],
            'imei_hash': device['imei_hash'],
            'imsi_hash': device['imsi_hash'],
            'salt_id': device['salt_id'],
        },
    }

    # Sign the canonical claim set with Ed25519, then embed the signature.
    message = jwt.encode(payload, SECRET, algorithm='HS256').encode()
    payload['device_sig'] = signing_key.sign(message).signature.hex()

    return ref_token, jwt.encode(payload, SECRET, algorithm='HS256')
```

The `write_key_atomically` helper is the part teams most often skip. Without the fsync-then-rename sequence, a power loss during key rotation can leave a truncated key file and a device that can no longer authenticate.

### Token fetch endpoint

```python
# server.py
import json

import redis
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

app = FastAPI()
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
bearer = HTTPBearer()


@app.post('/token/fetch')
def fetch_token(
    credentials: HTTPAuthorizationCredentials = Depends(bearer),
    device_id: str = Header(...),
    sig: str = Header(...),
):
    ref_token = credentials.credentials
    cached = redis_client.get(f'ref:{ref_token}')
    if not cached:
        raise HTTPException(status_code=404, detail='Token not found')

    payload = json.loads(cached)

    if payload['device']['iccid'] != device_id:
        raise HTTPException(status_code=403, detail='Device mismatch')

    if not verify_signature(payload, sig):
        raise HTTPException(status_code=403, detail='Invalid signature')

    return payload
```

This endpoint is intentionally minimal. In production it should also enforce a per-device rate limit, reject replayed `jti` values, and verify that the signature covers a server-issued nonce rather than a static message.

## Failure modes and how to detect them

### Clock drift invalidates tokens

If a device's clock is wrong by more than the token's validity window, every token it presents will be rejected as expired or not-yet-valid. The symptom is a sudden, device-specific spike in authentication failures with no corresponding change in credentials.

**Detection:** log `iat` and the server's `now` for every rejected token. A cluster of rejections where `now - iat` is consistently large (or negative) points to clock drift, not to a credential problem. **Mitigation:** prefer a time source that does not depend on the same network as the data path; GPS or carrier-provided time in the SMS header are common fallbacks. Do not assume NTP over a lossy cellular link is reliable.

### SIM swap defeats device-only binding

If the only anchor is the ICCID and the attacker controls the subscriber's SIM, the attacker can present the new SIM's ICCID and pass device checks. Binding to a keyed IMSI hash plus a `last_sim_swap` timestamp lets the server detect that the subscriber identity changed and reject tokens minted before the change.

**Detection:** record the IMSI hash at token issuance and compare on every refresh. A mismatch is a swap event; a match with a stale issuance timestamp is a replay. **Caveat:** there is an inherent race. If the agent mints a token in the window between swap initiation and completion, the token carries the old hash and the server cannot distinguish it from a legitimate pre-swap token. Short TTLs narrow this window; they do not eliminate it.

### Cache stampede on the fetch path

When many agents refresh simultaneously — for example, after a network outage ends — the fetch endpoint and its backing cache see a burst. Without coordination, the cache can saturate and drop connections, which cascades into authentication failures.

**Detection:** measure cache CPU and connection count during the minute following a known outage. **Mitigation:** a short-lived distributed lock (a Redis `SET NX PX` with a small TTL) around payload generation, plus jittered token TTLs so refreshes do not align.

### Transport truncation and fragmentation

Payloads larger than the transport's single-message limit fragment, and fragmented messages fail more often. **Detection:** record segment count per message and correlate failure rate with segment count. **Mitigation:** shrink the on-air payload via a split token, or compress before sending. Compression helps only if the payload is compressible; a base64-encoded signature is close to incompressible, so measure before assuming a win.

### Key rotation corruption

Writing a new key in place risks a truncated file if power is lost mid-write. **Detection:** validate the key file on boot and alert if it fails to parse. **Mitigation:** write-temp-fsync-rename, as shown above.

### Trusted-execution key generation limits

Some TEEs do not support RSA key generation above a certain size, and some do not support it at all. Attempting it fails at runtime, often only on the specific hardware revision. **Detection:** generate the key once during provisioning and fail the build if it does not succeed, rather than discovering the limitation in the field. **Mitigation:** Ed25519 keys are 32 bytes and avoid the problem entirely, at the cost of compatibility with legacy clients that only speak RSA.

## Choosing the right stack

| Scenario | Recommended anchor | Token approach | Main residual risk |
|---|---|---|---|
| Stable power, reliable network, controlled hardware | Standard OAuth2 + PKCE, short-lived JWTs | Single signed token | Key compromise |
| Constrained transport, intermittent power | Composite device ID + keyed IMSI hash | Split token with device-bound fetch | Clock drift, SIM swap race |
| High-value transactions | Hardware-backed key storage or certified secure element | Hardware-signed, short-lived | Provisioning cost and complexity |
| Legacy clients limited to RSA | RSA-2048 with a trusted key store | Single signed token | Larger payloads, slower signing |

The choice is a tradeoff between the cost of the anchor and the value of the asset being protected. A hardware-backed anchor is worth it when the asset is a financial transaction. It is overkill when the asset is a low-value session on a device that will be replaced in a year.

## When this approach is the wrong choice

- **High-value financial flows.** Residual SIM-swap and clock-drift risk is too high. Use hardware-backed key storage or a certified secure element. The cost per device rises, but so does the assurance.
- **Devices with stable power and network.** If the agent runs in a data center or a well-connected branch, the standard guidance applies unchanged. Adding split tokens and GPS time sync there buys nothing and adds latency.
- **Legacy runtimes limited to RSA and SHA-1.** Ed25519 is not available. The right move is usually to isolate the legacy agent behind a gateway that terminates modern cryptography, rather than to weaken the whole system to the weakest client.

## A 30-minute action

Pick one agent deployment and add two log lines: the device's `iat` at token issuance and the server's `now` at verification, plus the transport segment count for the delivery. Run for a day, then plot `now - iat` and failure rate against segment count. That single chart will tell you whether your dominant failure mode is clock drift or fragmentation — and those two have completely different fixes.
