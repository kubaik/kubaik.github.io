# CSRF and SameSite cookies: what breaks

CSRF is not a bug you fix once. It is a class of request-forgery that reappears every time you add a new cookie, a new subdomain, or a new browser default. The SameSite cookie attribute changed the default behavior for cross-site requests, but it did not make CSRF disappear. It moved the failure mode from "any cross-site POST works" to "exactly which cross-site POSTs still carry credentials, and which ones do not." This post walks through the mechanics, the browser defaults, the edge cases that bite teams in production, and a minimal runnable defense. By the end you will be able to look at a cookie header and predict whether a cross-site request will include it — and you will know the one file to check first when a form mysteriously stops submitting.

## What CSRF actually is, before SameSite enters the picture

CSRF exploits the fact that browsers attach cookies to requests based on the target origin, not the origin that initiated the request. If a user is logged into `app.example.com` and visits `evil.example.net`, a form on the evil page can POST to `app.example.com/transfer` and the browser will include the session cookie because the request targets the app's origin. The server sees a valid session and processes the transfer.

The classic defenses are:

- **Synchronizer token pattern**: the server embeds a random token in the form, stores it in the session, and rejects requests where the submitted token does not match.
- **Double-submit cookie**: the server sets a random value in a cookie and also expects it in a request header or form field; an attacker on another origin cannot read the cookie to copy it into the header.
- **Origin/Referer checking**: reject state-changing requests whose `Origin` header is not the expected site.
- **SameSite cookies**: instruct the browser not to send the cookie on cross-site requests, or only on top-level navigations.

SameSite is the newest of these and the one most often misunderstood, because it is enforced by the browser, not the server. That means your server-side tests will pass while real browsers silently drop the cookie.

## Prerequisites: what you need to reproduce this

You need a server that sets cookies with explicit `SameSite` values, a second origin to act as the attacker, and a browser that follows current cookie defaults. A local setup with two ports on `localhost` is not enough, because `localhost` is treated as a single site for cookie purposes in most browsers. Use two hostnames that resolve to the same machine, for example `app.local.test` and `evil.local.test`, added to your hosts file. Serve over HTTP for local testing; the `Secure` attribute is ignored on `http://localhost` in most browsers, but on custom hostnames you will need HTTPS or you will spend an hour debugging why `Secure` cookies do not appear.

You also need to know which browser you are testing against. Chrome, Firefox, and Safari all implement SameSite, but their treatment of "site" versus "origin" differs for subdomains and for schemes. The specification defines "site" as scheme plus registrable domain (eTLD+1). That means `https://app.example.com` and `https://api.example.com` are the same site, while `http://app.example.com` and `https://app.example.com` are different sites because the scheme differs. This is the single most common source of confusion when a cookie works in one environment and not another.

## Step 1: Set SameSite explicitly and know the three values

The `SameSite` attribute accepts three values: `Strict`, `Lax`, and `None`. If you omit it, modern browsers treat the cookie as `Lax` by default. That default was a deliberate change to reduce CSRF, but it is not a security guarantee, because `Lax` still sends cookies on top-level GET navigations.

Here is a minimal Go server that sets a session cookie with each mode, so you can compare behavior:

```go
package main

import (
	"fmt"
	"net/http"
)

func main() {
	http.HandleFunc("/set", func(w http.ResponseWriter, r *http.Request) {
		mode := r.URL.Query().Get("mode")
		if mode == "" {
			mode = "Lax"
		}
		cookie := &http.Cookie{
			Name:     "session",
			Value:    "abc123",
			Path:     "/",
			HttpOnly: true,
			Secure:   false, // set true behind HTTPS
			SameSite: parseSameSite(mode),
		}
		http.SetCookie(w, cookie)
		fmt.Fprintf(w, "set session with SameSite=%s\n", mode)
	})

	http.HandleFunc("/echo", func(w http.ResponseWriter, r *http.Request) {
		c, err := r.Cookie("session")
		if err != nil {
			fmt.Fprintln(w, "no session cookie")
			return
		}
		fmt.Fprintf(w, "session=%s\n", c.Value)
	})

	http.ListenAndServe(":8080", nil)
}

func parseSameSite(s string) http.SameSite {
	switch s {
	case "Strict":
		return http.SameSiteStrictMode
	case "None":
		return http.SameSiteNoneMode
	default:
		return http.SameSiteLaxMode
	}
}
```

Run it, visit `/set?mode=Strict`, then load a page on your second hostname that issues a cross-site request to `/echo`. With `Strict`, the cookie is absent. With `Lax`, it is absent for POST and present for top-level GET navigation. With `None`, it is present for both, provided `Secure` is set (browsers reject `SameSite=None` without `Secure`).

The gotcha: `SameSite=None` without `Secure` is rejected by Chrome and Firefox, and the cookie is not stored at all. The console warning is easy to miss. If your integration test runs over plain HTTP and sets `SameSite=None`, the cookie will not exist in the browser, and your test will fail for the wrong reason.

## Step 2: Understand what Lax actually permits

`Lax` is the default, and it is permissive enough to be dangerous if you assume it blocks all CSRF. It allows the cookie on top-level navigations that use safe methods (GET, HEAD, OPTIONS). It blocks the cookie on cross-site POST, PUT, DELETE, and on iframes, image tags, and fetch requests from another origin.

That means a cross-site GET to a state-changing endpoint still carries the session cookie. If your application has any GET endpoint that mutates state — `/logout`, `/delete?id=...`, `/transfer?amount=...` — `Lax` does not protect it. The fix is not to change the cookie; the fix is to stop using GET for state changes. This is a design rule, not a cookie setting.

The other subtlety: `Lax` treats a top-level navigation as safe even if it is initiated by a link on an attacker's page. A user clicking a link to `https://app.example.com/transfer?to=attacker` sends the cookie. If that endpoint performs the transfer on GET, CSRF is still possible. Browsers are doing what the spec says; the application is violating HTTP semantics.

## Step 3: Add a server-side token so SameSite is not your only defense

SameSite is a browser control. You do not control which browser the user runs, and older browsers may ignore the attribute entirely. A robust defense combines SameSite with a synchronizer token or double-submit cookie. The token check runs on the server and does not depend on browser behavior.

A minimal double-submit implementation in Go:

```go
package main

import (
	"crypto/rand"
	"encoding/hex"
	"net/http"
)

func setCSRFCookie(w http.ResponseWriter) string {
	b := make([]byte, 32)
	if _, err := rand.Read(b); err != nil {
		panic(err)
	}
	token := hex.EncodeToString(b)
	http.SetCookie(w, &http.Cookie{
		Name:     "csrf",
		Value:    token,
		Path:     "/",
		HttpOnly: false, // must be readable by JS to copy into header
		Secure:   true,
		SameSite: http.SameSiteLaxMode,
	})
	return token
}

func requireCSRF(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet || r.Method == http.MethodHead {
			next(w, r)
			return
		}
		cookie, err := r.Cookie("csrf")
		if err != nil {
			http.Error(w, "missing csrf cookie", http.StatusForbidden)
			return
		}
		header := r.Header.Get("X-CSRF-Token")
		if header == "" || header != cookie.Value {
			http.Error(w, "csrf token mismatch", http.StatusForbidden)
			return
		}
		next(w, r)
	}
}
```

The client must read the `csrf` cookie and send it back in the `X-CSRF-Token` header on every state-changing request. An attacker on another origin cannot read the cookie value because of the same-origin policy, so they cannot forge the header. This works even if the browser ignores SameSite.

The gotcha: if you set the `csrf` cookie with `HttpOnly: true`, JavaScript cannot read it, and the double-submit pattern breaks. That is a common mistake when teams apply a blanket `HttpOnly` policy to all cookies. The session cookie should be `HttpOnly`; the CSRF cookie must not be.

## Step 4: Handle the cases SameSite does not cover

SameSite does not protect against attacks that originate from the same site. If an attacker can run JavaScript on a subdomain — for example, a user-content subdomain like `user.example.com` — that script is same-site with `app.example.com` and can read cookies that are not `HttpOnly`, and can issue same-site requests with credentials. This is why cookie scoping matters: set cookies on the most specific hostname possible, and avoid sharing a registrable domain between trusted and untrusted content.

The second case is OAuth and payment redirects. When a user returns from an external identity provider or a payment processor, the browser performs a top-level navigation from the external site to yours. With `SameSite=Strict`, the session cookie is not sent on that first request, so the user appears logged out. With `Lax`, the cookie is sent because it is a top-level GET. This is the reason `Strict` breaks many login flows, and why `Lax` is the common default for session cookies. If you need `Strict`, you must handle the first request without a session and redirect to a URL that re-establishes it.

The third case is cross-site POSTs that are legitimately required, such as a payment processor's webhook or a form embedded in a partner site. Those requests will not carry a `Lax` or `Strict` cookie. The correct pattern is to not rely on cookies for those endpoints; use a signed request or a shared secret in the body or header, and verify it server-side.

## Step 5: Verify with a real cross-site request, not a unit test

Unit tests that use an HTTP client do not enforce SameSite, because the client is not a browser. You will get false confidence. The verification must happen in a browser, with two distinct sites.

A practical checklist:

1. Set the session cookie with `SameSite=Lax` and `Secure`.
2. From a page on the second hostname, submit a cross-site form POST to your endpoint. The request must arrive without the session cookie, and the server must reject it.
3. From the same page, navigate with a top-level GET to a read-only endpoint. The cookie should be present.
4. From the same page, navigate with a top-level GET to a state-changing endpoint. If the cookie is present and the endpoint mutates state, you have a bug — fix the endpoint, not the cookie.
5. Repeat with `SameSite=Strict` and confirm the cookie is absent on the first navigation from the external site.
6. Repeat with `SameSite=None; Secure` and confirm the cookie is present on cross-site POST, then confirm your CSRF token check still rejects the request.

If step 6 passes without a token check, your application is vulnerable to CSRF despite the cookie attribute. That is the whole point: `SameSite=None` is sometimes necessary, and it removes the browser-level protection.

## Edge cases and errors that show up in production

**The cookie disappears after a redirect from an external site.** This is `Strict` behaving as specified. The first request after the redirect has no session. If your app redirects to a login page, the user sees a login screen even though they just authenticated. The fix is to use `Lax` for the session cookie, or to handle the first request as unauthenticated and then redirect to a same-site URL that sets the session.

**The cookie is set but not sent on API calls.** If the frontend is on `app.example.com` and the API is on `api.example.com`, they are same-site (same scheme and registrable domain), so `Lax` cookies are sent. If the frontend is on `app.example.com` and the API is on `api.other.com`, they are cross-site, and `Lax` cookies are not sent on fetch. The error in the browser console is a CORS error, not a cookie error, which sends teams down the wrong path. Check the `Set-Cookie` header and the request's `Cookie` header in the network tab before touching CORS config.

**`SameSite=None` is rejected in Safari.** Safari has historically been stricter about third-party cookies and about `SameSite=None`. If your integration depends on cross-site cookies, test in Safari specifically. A cookie that works in Chrome may be dropped in Safari, and the failure is silent.

**The CSRF token is cached.** If your HTML is cached by a CDN and contains a per-session CSRF token, users will receive a token that does not match their session. The server rejects the request. The fix is to not cache pages that embed per-session tokens, or to fetch the token via a separate uncached request.

**Double-submit fails behind a proxy that rewrites cookies.** Some reverse proxies or API gateways rewrite cookie headers. If the `csrf` cookie is rewritten but the `X-CSRF-Token` header is not, the comparison fails. Verify the cookie value at the edge matches what the application sees.

## A decision table for cookie attributes

| Scenario | Session cookie | CSRF cookie | Why |
|---|---|---|---|
| Standard web app, same-site frontend and API | `SameSite=Lax; Secure; HttpOnly` | `SameSite=Lax; Secure` (not HttpOnly) | Lax allows top-level GET after external redirects; token check covers POST |
| App requires cross-site POST (embedded form, partner integration) | `SameSite=None; Secure; HttpOnly` | `SameSite=None; Secure` (not HttpOnly) | None is required for cross-site; token check is mandatory because browser protection is off |
| High-security app with no external redirects | `SameSite=Strict; Secure; HttpOnly` | `SameSite=Strict; Secure` (not HttpOnly) | Strict blocks all cross-site sending; handle first navigation without session |
| API-only service consumed by third-party sites | No cookie; use bearer token | N/A | Cookies are the wrong tool for cross-site API auth |

## Frequently Asked Questions

**Does SameSite=Lax prevent all CSRF?**
No. Lax still sends cookies on top-level GET navigations, so any state-changing GET endpoint remains exploitable. It also does not protect against same-site attacks, such as a compromised subdomain. Lax reduces the attack surface but does not replace a server-side token check.

**Why is my session cookie not sent after redirect from PayPal or Auth0?**
This is usually `SameSite=Strict`. The redirect from the external provider is a cross-site navigation, and Strict cookies are not sent on the first request. Switch the session cookie to `Lax`, or handle the first request as unauthenticated and redirect to a same-site URL that re-establishes the session.

**Can I use SameSite=None without Secure?**
No. Browsers reject `SameSite=None` cookies that lack the `Secure` attribute, and the cookie is not stored. You must serve over HTTPS. During local development, `http://localhost` is often treated as a secure context, but custom hostnames are not, so local tests can behave differently from production.

**Is a CSRF token still needed if I use SameSite=Strict?**
Defense in depth says yes. SameSite is enforced by the browser, and you do not control the browser version or extensions. A server-side token check is independent of browser behavior and catches cases where the attribute is ignored or stripped. The cost is small: one random value per session and one header comparison.

## The one thing to check first

Open your browser's developer tools, load your app, and inspect the `Set-Cookie` header on the login response. Confirm that every cookie has an explicit `SameSite` value, that the session cookie has `HttpOnly`, and that the CSRF cookie does not. If any cookie is missing `SameSite`, the browser is defaulting it to `Lax`, and you are relying on a default you did not choose. Fix that header before you touch any CSRF middleware — it takes five minutes and eliminates the most common cause of "it works in Postman but not in the browser."

## Official documentation

- [OAuth 2.0 resources](https://oauth.net/2/)

---

### About this article

**Written by:** [Kubai Kevin](/about/) — software developer based in Nairobi, Kenya.

**How this article was produced:** This site uses an automated LLM pipeline designed and maintained by the author. Before publication, drafts are screened automatically for minimum length, near-duplicate content, invented or unsourced claims and citations, unverifiable product names, and broken internal links. Individual line-by-line human editing is not performed on every post. Figures, benchmarks and scenarios are illustrative unless a source is linked; verify them against official documentation before relying on them in production. See the [AI content policy](/ai-content-policy/).

**Corrections:** Report errors via the [contact page](/contact/). Corrections are applied promptly.

**Last generated:** November 2026
