# JWT sessions: revocation, rotation, storage

JWTs are stateless; sessions are not. The moment a user logs out or an administrator revokes access, the stateless promise collides with the requirement to invalidate something already issued. The gap between documented JWT behaviour and production session management is where teams get burned. This article closes that gap without pretending JWTs are something they are not.

## The stateless claim: why JWT revocation feels like a hack

The JWT specification (RFC 7519) defines a token format, not a session protocol. It says nothing about revocation, rotation, or storage. That is by design: JWTs are meant to be self-contained and verifiable without a database lookup. The moment a token must be invalidated before its expiry, state re-enters the system. Blocklists, short expiries, and refresh token rotation are all attempts to reconcile that tension.

The underlying problem is that JWTs are often treated as a drop-in replacement for server-side sessions without an evaluation of the trade-offs. A server-side session is a pointer to state stored on the server. A JWT is the state itself, signed. Revoking a server-side session is a delete. Revoking a JWT is a distributed systems problem: every verifier must learn that the token is no longer valid, or the token's validity window must be short enough that the residual risk is acceptable.

## Why a blocklist is not a revocation strategy

A blocklist (or denylist) is the most intuitive revocation mechanism: store the `jti` (JWT ID) of revoked tokens in a fast data store and check it on every request. Its pitfalls are numerous:

- **Unbounded growth**: unless entries expire at the token's `exp`, the blocklist grows forever. Entries need a TTL equal to the token's remaining lifetime.
- **Latency**: every request now requires a lookup, eroding the stateless advantage. If the store is unavailable, the system either fails open (a security hole) or fails closed (an availability hole).
- **Race conditions**: a token can be used between the time it is revoked and the time the blocklist entry propagates to all verifiers.
- **Scope creep**: the result is a session store with extra steps.

A blocklist is acceptable for low-volume, high-security operations, such as revoking a single administrator token after a suspected compromise. It is not a session management strategy.

## Refresh token rotation: the mechanism that scales

The practical answer for most applications is to keep access tokens short-lived (minutes) and use refresh tokens with rotation and reuse detection. This is the pattern described in OAuth 2.0 and OpenID Connect, and it is well-documented in the OAuth 2.0 Security Best Current Practice (RFC 9700, which superseded the earlier draft).

The mechanism:

1. The client authenticates and receives an access token (for example, 15 minutes) and a refresh token (for example, 7 days).
2. When the access token expires, the client sends the refresh token to the token endpoint.
3. The server validates the refresh token, issues a new access token and a new refresh token, and invalidates the old refresh token.
4. If an old refresh token is used again (reuse detection), the server revokes the entire token family — all refresh tokens descended from the original grant — and forces re-authentication.

This provides revocation: logging out a user means revoking their refresh token family. The access token remains valid until it expires, but that window is short. For immediate revocation, a blocklist for access tokens is still required, but the blast radius is small.

### Implementation sketch

A minimal refresh token rotation implementation in Node.js against a SQL database (Postgres syntax) might look like this. The schema stores a hash of the refresh token, not the token itself, to limit damage from a database leak.

```sql
CREATE TABLE refresh_tokens (
  id UUID PRIMARY KEY,
  user_id UUID NOT NULL,
  token_hash TEXT NOT NULL,
  family_id UUID NOT NULL,
  expires_at TIMESTAMPTZ NOT NULL,
  revoked BOOLEAN NOT NULL DEFAULT FALSE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_refresh_tokens_hash ON refresh_tokens (token_hash);
CREATE INDEX idx_refresh_tokens_family ON refresh_tokens (family_id);
```

```javascript
const crypto = require('crypto');
const { v4: uuidv4 } = require('uuid');

async function issueRefreshToken(db, userId, familyId = null) {
  const token = crypto.randomBytes(32).toString('hex');
  const hash = crypto.createHash('sha256').update(token).digest('hex');
  const id = uuidv4();
  const fam = familyId || uuidv4();
  const expiresAt = new Date(Date.now() + 7 * 24 * 60 * 60 * 1000);
  await db.query(
    'INSERT INTO refresh_tokens (id, user_id, token_hash, family_id, expires_at) VALUES ($1, $2, $3, $4, $5)',
    [id, userId, hash, fam, expiresAt]
  );
  return { token, familyId: fam };
}

async function rotateRefreshToken(db, oldToken) {
  const hash = crypto.createHash('sha256').update(oldToken).digest('hex');
  const res = await db.query(
    'SELECT id, user_id, family_id, expires_at, revoked FROM refresh_tokens WHERE token_hash = $1',
    [hash]
  );
  if (res.rowCount === 0) throw new Error('Invalid refresh token');
  const row = res.rows[0];
  if (row.revoked) {
    // Reuse detected: revoke entire family
    await db.query('UPDATE refresh_tokens SET revoked = TRUE WHERE family_id = $1', [row.family_id]);
    throw new Error('Refresh token reuse detected');
  }
  if (new Date(row.expires_at) < new Date()) throw new Error('Refresh token expired');
  // Revoke old token
  await db.query('UPDATE refresh_tokens SET revoked = TRUE WHERE id = $1', [row.id]);
  // Issue new token in same family
  return issueRefreshToken(db, row.user_id, row.family_id);
}
```

This is a simplified example. Production code needs transaction isolation around the select-then-update sequence, rate limiting on the token endpoint, and a job that deletes expired rows. But it demonstrates the core idea: rotation plus reuse detection yields a revocation mechanism that does not require checking every access token against a blocklist.

### Worked example: computing the revocation window

Suppose access tokens live 15 minutes and refresh tokens live 7 days. A user logs out at 10:00:00, and the server revokes the refresh token family at 10:00:01. Any access token issued at 09:59:00 remains valid until 10:14:00 — a residual window of just under 14 minutes. If the same deployment adds a blocklist checked on every request, the window collapses to the blocklist write latency, but every request now pays a lookup.

The arithmetic is simple and worth doing explicitly for a given product: residual exposure equals (token issue time + access token lifetime) minus (revocation time). Shortening the access token lifetime from 15 minutes to 5 minutes cuts the worst-case residual window by two-thirds, at the cost of three times as many refresh calls per active session. That trade-off, not a rule of thumb, is the decision to make.

## Storage: where the token lives determines the threat model

The storage choice is not about convenience; it is about which attacks the design defends against. The three common options:

| Storage location | XSS risk | CSRF risk | Persistence | Typical use case |
|------------------|----------|-----------|-------------|------------------|
| `localStorage`   | High (any JS can read) | None (not sent automatically) | Yes | Single-page apps with strict CSP |
| `sessionStorage` | High | None | Tab lifetime | Short-lived sessions |
| `HttpOnly` cookie | Low (not accessible to JS) | High (sent automatically) | Configurable | Server-rendered apps, or SPAs with CSRF protection |

No option is universally best. The decision hinges on the threat model:

- With a strong Content Security Policy (CSP) and no third-party scripts, `localStorage` can be acceptable. XSS remains a risk: a single injected script can exfiltrate the token.
- `HttpOnly` cookies prevent token theft via XSS but require CSRF protection (for example, `SameSite=Strict`, or anti-CSRF tokens). They also require careful CORS configuration if the API is on a different origin.
- In-memory storage (a JavaScript variable) is the most resistant to persistence-based attacks, but the token is lost on page refresh and must be re-fetched using a refresh token stored in an `HttpOnly` cookie.

A common pattern for SPAs is to store the access token in memory and the refresh token in an `HttpOnly`, `Secure`, `SameSite=Strict` cookie. The refresh endpoint is the only one that accepts the cookie, and it returns a new access token in the response body. This limits the attack surface: XSS cannot read the refresh token, and CSRF is mitigated by `SameSite` and by the fact that the refresh endpoint requires a POST with a custom header, which cannot be sent cross-origin without a CORS preflight.

### Cookie flags that matter

- `HttpOnly`: prevents JavaScript access.
- `Secure`: only sent over HTTPS.
- `SameSite=Strict` or `Lax`: mitigates CSRF. `Strict` is safer but breaks some cross-site navigation flows; `Lax` is a reasonable default.
- `Path=/refresh`: limits the cookie to the refresh endpoint, reducing exposure.
- `Max-Age` or `Expires`: align with refresh token lifetime.

## Measuring what actually matters

Two metrics are worth tracking:

1. **Access token lifetime**: the window during which a revoked user can still access resources. Shorter is safer but increases refresh traffic.
2. **Revocation propagation delay**: the time between a revocation event (logout, admin action) and the moment the token is rejected by all verifiers. With a blocklist, this is the blocklist write latency plus cache TTL. With short-lived tokens only, it is the remaining token lifetime.

To measure the first, log token issuance and revocation timestamps and compare them to actual request authentication times. To measure the second, emit a structured log line at revocation time and another when a verifier first rejects the affected token, then compute the delta in your log aggregation tool. A histogram of "time since revocation" for rejected requests shows whether the strategy has a long tail. This is instrumentation, not a benchmark: the numbers depend entirely on the deployment, so they must be collected per system rather than borrowed.

## Failure modes: what breaks in production

Even with rotation and careful storage, several failure modes recur:

- **Clock skew**: `exp` and `nbf` claims are validated against the server's clock. If servers are not time-synchronized (for example, via NTP), tokens may be rejected prematurely or accepted after expiry. Allow a small leeway — commonly 30 seconds — in the JWT library configuration.
- **Key rotation without overlap**: when signing keys are rotated, tokens signed with the previous key become invalid immediately unless verifiers hold both keys. Use a key ID (`kid`) header and publish a JWKS endpoint that includes both keys during the overlap period.
- **Refresh token race conditions**: if a client makes two concurrent requests with the same refresh token, one succeeds and the other triggers reuse detection, revoking the family. This happens with multiple tabs or aggressive retries. Mitigate by serializing refresh requests on the client, or by allowing a short grace period for the old token after rotation, at the cost of weaker reuse detection.
- **Storage leakage via logging**: tokens in URLs, logs, or error messages. Never log full tokens; log a hash or the `jti`.
- **Improper CORS**: with cookies, `Access-Control-Allow-Origin` must not be `*` when `Access-Control-Allow-Credentials` is true. This is a common misconfiguration.

### Failure-mode analysis: the multi-tab logout

Consider a user with three tabs open. Tab A logs out, revoking the refresh token family. Tabs B and C still hold a valid access token and will continue to work until it expires. When they attempt a refresh, they receive an error and the client should redirect to login. If the client treats refresh failure as a transient error and retries, it will hammer the token endpoint with a revoked token. The correct client behaviour is to distinguish "refresh token invalid or revoked" (terminal, log out) from "network error" (retryable), and to broadcast logout across tabs using a `BroadcastChannel` or `storage` event. This is a client-side design decision that the server cannot enforce.

## When JWTs are the wrong choice

JWTs are not a universal replacement for sessions. They are a poor fit when:

- Immediate, global revocation with no tolerance for a validity window is required. A server-side session store (for example, Redis) with a session ID in a cookie is simpler and more predictable.
- The application is a traditional server-rendered monolith with a single backend. The stateless benefit is minimal, and the complexity of rotation and storage is not justified.
- The user base is small and traffic is low. The overhead of a session store is negligible, and the operational simplicity of server-side sessions wins.

A recurring architectural mistake is adopting JWTs because they are perceived as modern, without evaluating revocation requirements. If a product has an admin panel that must revoke access instantly, or handles sensitive data, a server-side session with a short idle timeout may be the better default.

The counterintuitive part is that the more revocation matters, the less suitable JWTs become. Their strength is stateless verification; their weakness is stateful invalidation. Choose based on which property is needed more.

## Decision checklist

Before committing to a JWT session design, answer these questions in writing:

- What is the maximum acceptable window during which a revoked user retains access? This sets the access token lifetime ceiling.
- Is instant revocation required for any role or action? If yes, a blocklist or server-side session is mandatory for that path.
- Where does the refresh token live, and can JavaScript read it? If yes, an XSS finding becomes a full account takeover.
- Is the refresh endpoint protected against CSRF, and is CORS configured without wildcards alongside credentials?
- Does the client distinguish terminal refresh failures from transient ones, and does it coordinate logout across tabs?
- Is there a key rotation plan with an overlap period at least as long as the maximum access token lifetime?
- Are tokens excluded from logs, URLs, and error messages?

If any answer is "don't know", that gap is the priority, not the token format.

## Frequently Asked Questions

### How do I revoke a JWT immediately?

A JWT cannot be revoked in a purely stateless system. Immediate revocation requires state: either a blocklist of revoked `jti` values checked on every request, or a session store that tracks valid tokens. The practical approach is short-lived access tokens (for example, 5–15 minutes) plus refresh token revocation, accepting a small window of validity for the access token. For immediate revocation, combine a blocklist with a short TTL.

### Should a JWT be stored in localStorage or cookies?

It depends on the threat model. `localStorage` is vulnerable to XSS but immune to CSRF. `HttpOnly` cookies are immune to XSS token theft but require CSRF protection. For most SPAs, storing the access token in memory and the refresh token in an `HttpOnly`, `Secure`, `SameSite` cookie is a balanced approach. Avoid `localStorage` for long-lived tokens when third-party scripts are present.

### What happens if a refresh token is stolen?

With rotation and reuse detection, a stolen refresh token can be used once. When the legitimate client presents the old token, the server detects reuse and revokes the entire token family, forcing re-authentication. This limits the attacker's window. Without rotation, a stolen refresh token grants persistent access until expiry.

### How often should signing keys be rotated?

There is no universal answer. A common practice is periodic rotation (for example, every 90 days) with an overlap period during which both old and new keys are published in the JWKS endpoint. The overlap should be at least as long as the maximum access token lifetime so active tokens are not invalidated. Automate the rotation and ensure verifiers refresh their key cache regularly.

## The next 30 minutes: audit your token lifetimes

Open the authentication configuration (for example, `auth.js`, `config/passport.js`, or the identity provider's settings) and find the access token expiry. If it exceeds the maximum revocation window identified in the checklist above, shorten it and note the resulting increase in refresh traffic. Then check the refresh logic: does it invalidate the old token on use and detect reuse? If not, that is the next change. Finally, verify that the refresh token is stored in an `HttpOnly`, `Secure`, `SameSite` cookie and that the CORS policy does not combine `*` with credentials. These three checks close the most common JWT session gaps.
