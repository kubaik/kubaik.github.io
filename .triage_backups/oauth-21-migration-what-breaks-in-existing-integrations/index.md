# OAuth 2.1 migration: what breaks in existing integrations

OAuth 2.1 is not a new grant type or a new endpoint. It is the IETF's consolidation of the OAuth 2.0 security best practices (RFC 9700) and PKCE (RFC 7636) into a single core specification, while removing the grants and flows that have been the source of most real-world vulnerabilities. The draft (`draft-ietf-oauth-v2-1`) tightens rules that many integrations were built to ignore: the implicit grant and the resource owner password credentials grant are gone, PKCE is mandatory for every authorization code flow, redirect URIs must be compared with exact string matching, and bearer tokens must not be sent in query parameters.

If an integration still uses the implicit flow for a single-page app, relies on a wildcard redirect URI, or passes `access_token` in a query string because it was easy, the migration is not a version bump. It is a set of breaking changes that surface as failed token requests, rejected redirects, and silent 401s. This article maps which of those changes hit which kind of integration, and gives a sequence for migrating without drowning in overlapping failures.

## Which OAuth 2.0 features OAuth 2.1 removes or changes

The draft is explicit about what it drops. The two removed grants are the implicit grant (`response_type=token`) and the resource owner password credentials grant (`grant_type=password`). Both were deprecated in the OAuth 2.0 Security Best Current Practice for concrete reasons: the implicit grant exposes access tokens in the URL fragment, where they leak through browser history, referrer headers, and logs; the password grant teaches users to hand their credentials to the client, defeating the point of delegated authorization.

PKCE becomes mandatory for all authorization code flows, including confidential clients. In OAuth 2.0, PKCE was optional for server-side apps. In OAuth 2.1, every authorization request must include `code_challenge` and `code_challenge_method`, and every token request must include the matching `code_verifier`. A client that omits PKCE is rejected by a compliant authorization server.

Redirect URI matching changes from prefix or wildcard matching to exact string comparison. The authorization server compares the `redirect_uri` in the authorization request to the registered URI using simple string comparison. That means `https://app.example.com/callback` and `https://app.example.com/callback/` are different URIs. A trailing slash, a different case in the path, or an extra query parameter can cause the request to fail.

The spec also forbids bearer tokens in URI query parameters. The `access_token` must be sent in the `Authorization` header (or in the request body for form-encoded requests, though the header is preferred). Any integration that appends `?access_token=...` to a URL breaks against a strict resource server.

Finally, the spec tightens refresh token rules for public clients: refresh tokens must be sender-constrained or rotated, and a public client must not receive a refresh token unless it is bound to the client instance. The exact mechanism is left to the implementation (DPoP, mTLS, or rotation), but the old pattern of a long-lived refresh token stored in `localStorage` for a browser app no longer fits the model.

| OAuth 2.0 feature | OAuth 2.1 status | Common breakage |
|---|---|---|
| Implicit grant (`response_type=token`) | Removed | SPA stops getting tokens |
| Password grant (`grant_type=password`) | Removed | CLI or legacy app login fails |
| PKCE optional for confidential clients | Mandatory for all code flows | Token request rejected without `code_verifier` |
| Redirect URI wildcard/prefix match | Exact string match | Redirect rejected on trailing slash or case |
| Bearer token in query string | Forbidden | Resource server returns 401 |
| Long-lived refresh token for public clients | Must be rotated or sender-constrained | Refresh fails after rotation policy |

## The failures that show up first in real integrations

When teams start testing against an OAuth 2.1-compliant authorization server, failures tend to arrive in a predictable order. The first is usually the redirect URI mismatch, because it happens before any token is issued. A client registered with `https://app.example.com/callback` that sends `https://app.example.com/callback?state=abc` is rejected if the server does exact matching on the full URI including query. The spec allows the server to ignore the query component if it is registered separately, but many implementations do not. The error is usually `invalid_request` with a description like "redirect_uri does not match registered value."

The second failure is the missing PKCE parameters. A server-side client that never implemented PKCE sends an authorization request without `code_challenge`, and the authorization server returns an error at the authorization endpoint, not the token endpoint. The error is often `invalid_request` with "code_challenge required." This is a hard failure that cannot be worked around by retrying.

The third failure is the token request rejection. If the client sends a `code_verifier` that does not match the `code_challenge` — for example, because it used the wrong hash algorithm or a different encoding — the token endpoint returns `invalid_grant`. This is easy to misdiagnose as an expired code, because the error code is the same. The fix is to verify that the `code_verifier` is the same string used to compute the challenge, and that the challenge method matches (`S256` is the only method that should be used; `plain` is discouraged).

The fourth failure is the resource server rejecting a token that used to work. If the client was sending the token in a query parameter, the resource server may now return `401 Unauthorized` with a `WWW-Authenticate` header indicating that the token must be in the `Authorization` header. This failure is silent in logs that only record the URL path, because the token is no longer in the URL.

The fifth failure is refresh token rotation. A public client that receives a refresh token and does not handle rotation works until the first refresh, then fails with `invalid_grant` because the old refresh token was invalidated when a new one was issued. The client must store the new refresh token and use it for the next refresh. If the client makes concurrent requests, it may race and invalidate its own token.

## A migration sequence that minimizes breakage

The order of work matters. Trying to fix everything at once leads to a long debugging session where multiple failures mask each other. A sequence that isolates each change is more reliable.

Start by inventorying every client and every flow. For each client, record the grant type, the redirect URIs, whether PKCE is used, and how the token is sent to the resource server. This inventory is the basis for the migration plan. A simple spreadsheet is enough; the goal is to know which clients are affected by each change.

Next, enable PKCE for all authorization code clients, even if the server does not yet require it. This is a non-breaking change for servers that support PKCE, and it prepares the client for the mandatory requirement. The client must generate a `code_verifier`, compute the `code_challenge` using `S256`, and send both. A minimal implementation in JavaScript looks like this:

```javascript
// Generate a PKCE code verifier and challenge
function base64urlEncode(buffer) {
  return btoa(String.fromCharCode(...new Uint8Array(buffer)))
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=+$/, '');
}

async function generatePKCE() {
  const array = new Uint8Array(32);
  crypto.getRandomValues(array);
  const verifier = base64urlEncode(array);
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(verifier));
  const challenge = base64urlEncode(digest);
  return { verifier, challenge, method: 'S256' };
}
```

Send the `code_challenge` and `code_challenge_method` in the authorization request, and the `code_verifier` in the token request. If the server supports PKCE, the flow works. If it does not, the server ignores the parameters, and the flow still works. This is a safe first step.

After PKCE is in place, fix the redirect URIs. Compare the registered URI to the one the client sends, character by character. Remove any wildcards, trailing slashes, or query parameters that are not registered. If the client needs to pass state, use the `state` parameter, not a query parameter on the redirect URI. The `state` parameter is returned to the client and is the correct place for CSRF protection and application state.

Then migrate the clients that use removed grants. For an SPA using the implicit grant, the replacement is the authorization code flow with PKCE. The SPA no longer receives the token in the fragment; it receives an authorization code, which it exchanges for a token at the token endpoint. This requires a token endpoint request from the browser, which is allowed for public clients with PKCE. If the SPA cannot make a token request (for example, because of CORS restrictions), a backend-for-frontend pattern is the alternative: the SPA talks to its own backend, and the backend performs the token exchange.

For a CLI or legacy app using the password grant, the replacement is the device authorization grant (RFC 8628) or a browser-based authorization code flow with PKCE. The device grant is designed for input-constrained devices: the app displays a code and a URL, the user completes authorization on another device, and the app polls the token endpoint. This is a significant change for a CLI, but it is the compliant option once the password grant is removed.

Finally, update the resource servers to reject tokens in query parameters and to require the `Authorization` header. This is a server-side change that can be rolled out with a deprecation period. Log requests that use the query parameter, and after the logs show no legitimate use, enforce the header requirement. To measure readiness, count distinct clients still sending `access_token` in the query string over a fixed window (for example, by parsing the query string in the access log and grouping by client ID or user agent). The migration is done when that count reaches zero for a full business cycle, not when it looks small.

## Handling refresh tokens for public clients

The refresh token change is the one that tends to be underestimated. In OAuth 2.0, a public client could receive a refresh token and use it indefinitely. In OAuth 2.1, the authorization server must either rotate refresh tokens on each use or bind them to the client instance using a sender-constrained mechanism. Rotation is the more common implementation because it does not require client-side cryptography beyond TLS.

With rotation, each refresh request returns a new refresh token and invalidates the old one. The client must store the new token and discard the old one. If the client makes two refresh requests concurrently, the second request uses an invalidated token and fails. The client must serialize refresh requests or handle the failure by re-authenticating. A common pattern is a single-flight refresh: the first request starts the refresh, and subsequent requests wait for the result.

```javascript
// Single-flight refresh in a browser client
let refreshPromise = null;

async function refreshToken() {
  if (!refreshPromise) {
    refreshPromise = doRefresh().finally(() => {
      refreshPromise = null;
    });
  }
  return refreshPromise;
}

async function doRefresh() {
  const stored = localStorage.getItem('refresh_token');
  const response = await fetch('/token', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({
      grant_type: 'refresh_token',
      refresh_token: stored,
      client_id: 'my-spa',
    }),
  });
  if (!response.ok) {
    // Re-authenticate
    throw new Error('refresh failed');
  }
  const data = await response.json();
  localStorage.setItem('refresh_token', data.refresh_token);
  localStorage.setItem('access_token', data.access_token);
  return data.access_token;
}
```

This code is illustrative and assumes the token endpoint is on the same origin or has CORS configured. In practice, storing refresh tokens in `localStorage` is itself a risk; a backend-for-frontend with an HttpOnly cookie is safer, but that is a larger architectural change. The key point is that rotation must be handled, or the client breaks on the second refresh.

A worked example makes the race concrete. Suppose a dashboard fires three API calls on page load, each of which independently notices the access token is expired and calls `refreshToken()`. Without single-flight, all three read the same stored refresh token, all three send it, the server rotates on the first request and invalidates it, and the second and third requests fail with `invalid_grant`. The user is logged out even though nothing was actually wrong. With the `refreshPromise` guard above, the first call starts the refresh, the other two await the same promise, and only one request reaches the token endpoint. The cost is a few lines of code; the alternative is intermittent logout reports that are hard to reproduce.

## What to expect after migration, and the limits of those expectations

A compliant OAuth 2.1 implementation should eliminate entire classes of vulnerabilities: token leakage through the URL fragment, credential phishing through the password grant, authorization code interception without PKCE, and open redirects through wildcard redirect URIs. The security benefit is real and documented in the OAuth 2.0 Security Best Current Practice.

The operational benefit is less certain and depends on the implementation. Refresh token rotation makes a failure mode explicit rather than silent, which can reduce the class of tickets where a session quietly dies. But the migration itself produces more tickets in the short term, because the failures are hard failures that block login. The migration is not invisible to users; it requires coordination and a communication plan.

The limits are important. OAuth 2.1 does not solve every OAuth problem. It does not mandate a specific token format (JWT or opaque), it does not define authorization policy, and it does not prevent a misconfigured client from requesting excessive scopes. It also does not eliminate the need for TLS, because the spec still requires HTTPS for all endpoints except loopback redirects. A team that migrates to OAuth 2.1 but keeps a permissive scope model has not solved the authorization problem.

Another limit: not all authorization servers implement OAuth 2.1 yet. A client that is ready for OAuth 2.1 may still be talking to a server that supports only OAuth 2.0. In that case, the client should be written to work with both: include PKCE, use exact redirect URIs, and send tokens in the header. These changes are backward-compatible with most OAuth 2.0 servers, so the client can be migrated before the server. To confirm compatibility before committing, run the same client against both server versions and diff the authorization and token responses; the client should behave identically apart from the stricter error cases.

## A decision checklist before flipping the switch

Use this as a gate for each client, not for the fleet as a whole. A client is ready when every line is yes or explicitly waived with a reason.

- Grant type: the client uses `authorization_code` (or `urn:ietf:params:oauth:grant-type:device_code` on input-constrained devices). No `token` response type, no `password` grant.
- PKCE: the authorization request includes `code_challenge` and `code_challenge_method=S256`; the token request includes the matching `code_verifier`.
- Redirect URIs: the registered value and the sent value are byte-identical, including scheme, host, port, path, and trailing slash.
- Token transport: the resource server reads the token from the `Authorization` header, and the client never appends `access_token` to a query string.
- Refresh: the client stores the rotated refresh token and serializes concurrent refreshes through a single-flight guard.
- Client authentication: a public client sends `client_id` only; a confidential client authenticates at the token endpoint without leaking the secret to the browser.
- Scopes: the requested scopes are a subset of what the server has registered for the client.

## What to watch out for

The most common trap is assuming that the authorization server's error messages are accurate. A server that returns `invalid_grant` for a PKCE mismatch may also return `invalid_grant` for an expired code or a revoked token. The client should log the full error response and the request parameters (redacting secrets) so that the cause can be identified. Without that logging, debugging is guesswork.

The second trap is the redirect URI with a query string. Some clients append `?redirect_uri=...` to the authorization request, and the server may interpret the nested query as part of the redirect URI. The spec requires the `redirect_uri` parameter to be URL-encoded, and the value must match the registered URI exactly. A client that builds the authorization URL by string concatenation is likely to get this wrong. Use a URL builder that encodes parameters correctly.

The third trap is the token endpoint authentication method. OAuth 2.1 does not remove client secret authentication for confidential clients, but it does require that public clients not use a client secret. A public client that sends a `client_secret` (even an empty one) may be rejected by a strict server. The client should send only `client_id` and the PKCE parameters.

The fourth trap is the scope change. Some servers that implement OAuth 2.1 also tighten scope handling, requiring that requested scopes be a subset of registered scopes. A client that requests `openid profile email` when only `openid` is registered fails. This is not strictly an OAuth 2.1 requirement, but it is a common companion change. Check the server's documentation for scope rules.

The fifth trap is assuming the browser can always reach the token endpoint. Some authorization servers do not send CORS headers on the token endpoint, and some deployments block the token endpoint at the edge. A single-page app that worked with the implicit grant (which needs no token request) can fail here. Test the token endpoint from the browser origin early, and fall back to a backend-for-frontend if CORS is not supported.

## The broader lesson

OAuth 2.1 is a case study in a pattern that appears across infrastructure standards: the specification catches up to the security research, and the integrations that were built on the looser version break. The lesson is not that OAuth 2.0 was wrong; it is that the optional features in a specification become the default features in practice, and the defaults become the attack surface. The implicit grant was optional in OAuth 2.0, but it was the easiest flow to implement for a single-page app, so it became common. PKCE was optional, so it was often skipped. The migration is the cost of those shortcuts.

The principle is to treat optional security features as mandatory in your own integrations, even before the specification requires them. A client that uses PKCE, exact redirect URIs, and header-based tokens today is already compatible with OAuth 2.1 and is more secure against OAuth 2.0 attacks. The migration then becomes a verification step rather than a rewrite.

## Applying this to your integration

The first step is to inventory the clients and flows, as described above. The second step is to run a compatibility test against a known OAuth 2.1-compliant server. If you do not have one, use a public test suite or a local implementation. The test should cover the authorization request, the token request, the refresh, and the resource request.

A concrete action for the next 30 minutes: open the client configuration for your oldest integration and check the registered redirect URIs. Compare them to the URIs the client actually sends, character by character. If you find a wildcard, a trailing slash mismatch, or a query parameter, fix it now. That single change removes the most common OAuth 2.1 failure before it happens.

## Frequently Asked Questions

**What is the difference between OAuth 2.0 and OAuth 2.1?**
OAuth 2.1 is a consolidation of OAuth 2.0 and several extensions, notably PKCE and the security best practices. It removes the implicit and password grants, makes PKCE mandatory for all authorization code flows, requires exact redirect URI matching, and forbids bearer tokens in query parameters. It is not a new protocol; it is a stricter profile of the existing one.

**Will OAuth 2.1 break my existing OAuth 2.0 integration?**
It breaks integrations that rely on removed features: the implicit grant, the password grant, wildcard redirect URIs, or query-parameter tokens. Integrations that already use the authorization code flow with PKCE, exact redirect URIs, and header-based tokens are likely to work without changes. The breakage is concentrated in older or simpler clients.

**How do I migrate a single-page app from the implicit grant to OAuth 2.1?**
Replace the implicit grant with the authorization code flow with PKCE. The SPA redirects to the authorization endpoint with a `code_challenge`, receives an authorization code, and exchanges it at the token endpoint with the `code_verifier`. If the SPA cannot make the token request directly, use a backend-for-frontend that performs the exchange.

**Do I need to rotate refresh tokens for a public client?**
Yes. OAuth 2.1 requires that refresh tokens for public clients be either rotated on each use or sender-constrained. Rotation is the more common implementation. The client must store the new refresh token and handle concurrent refresh requests with a single-flight pattern.

**Does OAuth 2.1 require a specific token format?**
No. The spec leaves the token format to the implementation. JWT and opaque tokens are both permitted; the choice affects introspection, revocation, and how much information the resource server can read without a lookup, but it is not a conformance requirement.
