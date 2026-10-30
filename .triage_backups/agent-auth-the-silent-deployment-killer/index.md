# Agent auth: the silent deployment killer

Agent identity looks simple until it has to survive real traffic. A default configuration is fine right up until it isn't, and the failure usually surfaces as a wall of `403 InvalidToken` responses rather than as an obvious infrastructure error. This article covers what comes after the happy path: the constraints that break textbook identity stacks, the layered fixes that hold up, and how to measure whether any of them actually helped.

## Why the documented stack breaks in the field

The common starting point is workload identity with short-lived certificates, JWTs with short expiry, and a cache in front of validation. That stack is correct for stateless services on reliable networks. It tends to crack under four constraints that documentation rarely addresses.

**Network jitter.** An agent on a moving vehicle can lose connectivity for 15–30 seconds and regain it. During that window the workload API client cannot renew its identity, the short-lived JWT expires, and a cache entry may evict. The downstream service sees a 403 and the client retries — precisely when a user is waiting for a reply.

**Device diversity.** Feature phones, KaiOS devices, and low-end Android handsets run stripped-down clients that cannot complete long mTLS handshakes. A handshake that exceeds a client's TCP timeout fails before any identity logic runs.

**Power loss during rotation.** Kiosks on solar or unreliable grid power drop out without warning. When one returns, its identity client tries to renew immediately. If many return at once, they can synchronize into a retry storm that saturates the certificate authority and degrades unrelated services.

**Cross-border attestation.** Agents in one country attesting to services in another require identity strings that encode cluster, namespace, and region. That extra data inflates certificate chains, which breaks clients with hard chain-size limits.

None of these are exotic. They are the normal operating conditions of distributed agents outside well-provisioned data centers.

## Identity as a state machine, not a static ID

The first conceptual shift is treating identity as stateful. Instead of a fixed identity that is either valid or not, the token carries the conditions under which it may be used:

- **Region lock** — an agent registered in one region must not authenticate against another region's services, even with a valid identity.
- **Battery threshold** — below a configured level, the agent may send heartbeats but not payloads.
- **Network class** — agents on slow links receive a degraded identity with a shorter expiry.

These become additional claims in the token payload:

```json
{
  "spiffe_id": "spiffe://example/region-a/toll-collector/agent-123",
  "region_lock": "region-a",
  "min_battery_pct": 20,
  "max_network_class": "2G"
}
```

The value of this model is that policy decisions move from static configuration into the token itself. A validating service can reject a request for a policy reason without calling back to a central authority, which matters when the authority is the thing that is unreachable.

## A sidecar identity provider with a fallback token

A lightweight sidecar identity provider sits next to the agent. It speaks mTLS to the upstream identity server for certificate renewal, but it also caches a fallback JWT that the agent can use when the upstream server is unreachable. The fallback token has a longer expiry and is signed by a regional key rather than a global one.

The trade-off is explicit: a longer-lived regional token is a weaker guarantee than a freshly issued short-lived certificate. It is worth taking only when the alternative is total authentication failure during a partition. The design should make the fallback path visible — log every use, and alert when the fallback rate rises, because a rising fallback rate is an early warning that the primary path is degrading.

## Adaptive retry and backoff in the client

The agent client should detect its own network class and adjust retry timing accordingly. A workable approach:

- Measure round-trip time to a regional health endpoint.
- Use a short base delay on fast links and a much longer one on slow links.
- Skip renewal attempts when battery is low and use the fallback token instead.

A worked example of the delay calculation, with illustrative numbers: start from a 50 ms base delay and double per attempt, capped at 5 seconds.

- Attempt 1: 50 ms
- Attempt 2: 100 ms
- Attempt 3: 200 ms
- Attempt 4: 400 ms

On a slow link, multiply the base by 3 before applying the same progression:

- Attempt 1: 150 ms
- Attempt 2: 300 ms
- Attempt 3: 600 ms
- Attempt 4: 1,200 ms

The point is not these specific numbers but that the client stops hammering a struggling authority. Without backoff, a returning fleet of agents produces the retry storm described earlier.

## Cross-region attestation caching

Rather than having every validating service call a remote certificate authority, a cross-region cache replicates identity documents between regions with a short TTL. A service in one region validates against a local cache entry instead of a remote authority. The cache is refreshed by a low-bandwidth gossip protocol between regional proxies.

This reduces authority load and removes a network round trip from the validation path, but it introduces a consistency window. A revoked identity may remain valid in a remote cache until the TTL expires. Choose the TTL with that in mind: shorter TTLs tighten revocation but increase gossip traffic.

## Battery-aware certificate rotation

Rotation should be gated on battery state rather than running on a fixed schedule. A common policy:

- Above 80%: rotate on the normal interval.
- Between 40% and 80%: rotate less frequently.
- Below 40%: skip rotation until power returns.

This is a small amount of policy code, but it prevents certificate rotation from competing with other work during low-power events, which is a frequent cause of crashes on constrained devices.

## A minimal sidecar implementation

The following Go program listens on a local port, obtains an identity from the workload API, and issues a signed JWT with the policy claims described above.

```go
package main

import (
	"context"
	"log"
	"net/http"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/google/uuid"
	"github.com/spiffe/go-spiffe/v2/workloadapi"
)

type SIP struct {
	spiffeClient *workloadapi.Client
	jwtSecret    []byte
}

type policyClaims struct {
	RegionLock      string `json:"region_lock"`
	MinBatteryPct   int    `json:"min_battery_pct"`
	MaxNetworkClass string `json:"max_network_class"`
	jwt.RegisteredClaims
}

func (s *SIP) handler(w http.ResponseWriter, r *http.Request) {
	id, err := s.spiffeClient.GetSpiffeID(context.Background())
	if err != nil {
		http.Error(w, "no identity", http.StatusForbidden)
		return
	}

	now := time.Now()
	claims := policyClaims{
		RegionLock:      "region-a",
		MinBatteryPct:   20,
		MaxNetworkClass: "2G",
		RegisteredClaims: jwt.RegisteredClaims{
			Subject:   id.String(),
			ExpiresAt: jwt.NewNumericDate(now.Add(24 * time.Hour)),
			Issuer:    "sip-region-a",
			IssuedAt:  jwt.NewNumericDate(now),
			NotBefore: jwt.NewNumericDate(now),
			ID:        uuid.New().String(),
		},
	}

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	tokenString, err := token.SignedString(s.jwtSecret)
	if err != nil {
		http.Error(w, "token error", http.StatusInternalServerError)
		return
	}

	w.Write([]byte(tokenString))
}

func main() {
	ctx := context.Background()

	client, err := workloadapi.New(ctx)
	if err != nil {
		log.Fatalf("workload API client: %v", err)
	}

	sip := &SIP{
		spiffeClient: client,
		jwtSecret:    []byte("replace-with-a-secret-from-your-secret-store"),
	}

	http.HandleFunc("/token", sip.handler)
	log.Fatal(http.ListenAndServe(":8081", nil))
}
```

Two corrections matter here relative to a naive version. First, custom claims belong in the struct, not in a nested `Claims` map on `RegisteredClaims` — the latter does not serialize as intended. Second, the signing secret must come from a secret store, not a literal in source.

## An agent client with fallback and backoff

```go
package agent

import (
	"bytes"
	"context"
	"fmt"
	"math"
	"net/http"
	"time"
)

type Agent struct {
	sipURL       string
	spireURL     string
	batteryPct   int
	networkClass string
}

func (a *Agent) getToken(ctx context.Context) (string, error) {
	client := &http.Client{Timeout: 2 * time.Second}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, a.sipURL+"/token", nil)
	if err == nil {
		if resp, err := client.Do(req); err == nil {
			defer resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				buf := new(bytes.Buffer)
				if _, err := buf.ReadFrom(resp.Body); err == nil {
					return buf.String(), nil
				}
			}
		}
	}

	req, err = http.NewRequestWithContext(ctx, http.MethodGet, a.spireURL+"/spire/token", nil)
	if err != nil {
		return "", err
	}
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	buf := new(bytes.Buffer)
	if _, err := buf.ReadFrom(resp.Body); err != nil {
		return "", err
	}
	return buf.String(), nil
}

func (a *Agent) baseDelay() time.Duration {
	switch a.networkClass {
	case "2G":
		return 1500 * time.Millisecond
	case "3G":
		return 500 * time.Millisecond
	default:
		return 50 * time.Millisecond
	}
}

func (a *Agent) adaptiveRetry(ctx context.Context, url string, payload []byte) error {
	const maxAttempts = 4
	base := a.baseDelay()

	for attempt := 0; attempt < maxAttempts; attempt++ {
		delay := time.Duration(float64(base) * math.Pow(2, float64(attempt)))
		if max := 5 * time.Second; delay > max {
			delay = max
		}

		token, err := a.getToken(ctx)
		if err != nil {
			time.Sleep(delay)
			continue
		}

		req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
		if err != nil {
			return err
		}
		req.Header.Set("Authorization", "Bearer "+token)
		req.Header.Set("Content-Type", "application/json")

		resp, err := http.DefaultClient.Do(req)
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode < 500 {
				return nil
			}
		}

		time.Sleep(delay)
	}

	return fmt.Errorf("failed after %d attempts", maxAttempts)
}
```

The important behavioral change is that a 4xx response stops the loop. Retrying a request that was rejected for a policy reason will never succeed and only adds load.

## A cross-region attestation cache

```python
import json
import redis


class AttestationCache:
    def __init__(self, redis_url, region, ttl_seconds=300):
        self.redis = redis.Redis.from_url(redis_url)
        self.region = region
        self.ttl = ttl_seconds

    def publish_svid(self, spiffe_id, svid_pem, expires_at):
        payload = {
            "spiffe_id": spiffe_id,
            "svid_pem": svid_pem,
            "expires_at": expires_at.isoformat(),
            "region": self.region,
        }
        self.redis.xadd("attestation:stream", {"data": json.dumps(payload)})

    def get_svid(self, spiffe_id):
        cached = self.redis.get(f"attestation:{spiffe_id}")
        if cached:
            return json.loads(cached)
        return None

    def store_svid(self, spiffe_id, data):
        self.redis.setex(f"attestation:{spiffe_id}", self.ttl, json.dumps(data))
```

Note that `get_svid` reads the local cache only. Replicating from the stream belongs in a background consumer, not in the request path — reading the stream on every lookup defeats the purpose of the cache and adds latency under load.

## Battery-aware rotation

```bash
#!/bin/bash
set -euo pipefail

BATTERY=$(cat /sys/class/power_supply/BAT0/capacity)

if [ "$BATTERY" -gt 80 ]; then
  spire-agent api rotate
elif [ "$BATTERY" -gt 40 ]; then
  sleep 30
  spire-agent api rotate
else
  echo "battery below threshold, deferring rotation"
  exit 0
fi
```

The `-ttl` flag shown in some examples is not a documented argument for this command; rotation timing should be controlled by the agent's configured SVID TTL and by when this script runs.

## How to measure whether any of this helped

No table of results belongs here, because results depend entirely on the deployment. What matters is knowing what to instrument.

**403 rate per thousand requests, split by cause.** Log the token error reason, not just the status code. `InvalidToken`, `InvalidIssuer`, and clock-skew failures have different fixes.

**Fallback token usage rate.** Count how often the sidecar serves a fallback token. A rising rate precedes a rise in 403s and is the earliest useful signal.

**Certificate authority request rate.** Instrument the authority's request counter. A synchronized retry storm shows up as a sharp spike, not a gradual rise.

**P99 latency for agent-to-service calls, segmented by network class.** Aggregate latency hides the slow-link population that is actually failing.

**Rotation failures correlated with battery level.** Log battery state alongside every rotation attempt.

To compare before and after a change, capture these five metrics for a fixed window, apply one change, and capture the same window again. Comparing a single latency number before and after a change tells you almost nothing on its own.

## Failure modes worth designing for

**Clock skew on offline agents.** Hardware clocks drift. An agent that has been offline may believe its token is valid after the issuer has rotated keys, producing an issuer mismatch. A clock sync step before renewal fixes it, at the cost of a delay on the first request after power loss.

**Certificate chain bloat.** Adding regional data to identity strings inflates certificate chains. Clients with hard chain-size limits fail the handshake. Shorter identifiers or a trimmed chain help, but trimming can break cross-region validation — test both paths.

**Cache eviction storms.** When a region recovers, all agents reconnect at once and fill the cache. Under an LRU policy, this evicts entries that are still needed. An LFU policy, a larger memory limit, or admission control on reconnection all reduce the effect.

**Handshake timeouts on slow links.** If the mTLS handshake exceeds the client's TCP timeout, no identity logic runs at all. Session resumption and a reduced cipher list cut handshake time, but verify that every device in the fleet supports the ciphers you keep.

**Gossip storms after partition healing.** When a partition heals, all regions may replicate at once. Jittered backoff of 0–5 seconds between publishers spreads the load.

## When this stack is the wrong choice

The layered approach is overkill when agents run on well-provisioned devices with stable power and fast networks, when there is a single region with no cross-border traffic, or when occasional multi-second latency spikes during outages are acceptable. In those cases the simpler stack — workload identity, short-lived tokens, a validation cache — is sufficient and much easier to operate. The added layers exist to handle unreliable power, slow links, and constrained devices. If none of those apply, they are pure complexity.

## FAQ

**Why do agents get `403 InvalidToken` when the token has not expired?**

Usually the token is valid but the validating environment disagrees. A short-lived token can expire during a connectivity gap while the validation cache evicts the key at the same time, so the service rejects a token that was valid moments earlier. Clock skew on offline devices makes it worse: a drifting hardware clock can make an agent believe a token is still valid after the issuer has rotated. The fix is typically a fallback token with a longer expiry plus a clock sync before renewal.

**What is a sidecar identity provider and why use one?**

It is a local service that brokers identity tokens for the agent, speaking mTLS to the upstream identity server and caching a fallback token signed by a regional key. It exists because the upstream server is often unreachable during power loss or partitions, and short-lived certificates expire during those windows. The fallback keeps the agent authenticating instead of failing every request.

**How should certificate rotation work on battery-powered devices?**

Gate rotation on battery level rather than a fixed schedule. Rotate normally above 80%, less often between 40% and 80%, and defer below 40% until power returns. This prevents rotation from competing with other work during low-power events, which is a common cause of crashes on constrained devices.

**Why does cross-region attestation add latency, and how is it reduced?**

It requires reaching an authority in another region, adding a round trip on top of the handshake. A cross-region cache replicates identity documents with a short TTL so validation can happen locally. The trade-off is a revocation window: an identity revoked in one region may remain valid in another until the TTL expires.

## Do this in the next 30 minutes

Open your agent client and find the retry loop that handles authentication failures. Add a branch that stops retrying on any 4xx response and logs the token error reason separately from the status code. Then run a short load test against a slow-link profile and record the 403 count split by reason. That single change — distinguishing "retry because the network failed" from "never retry because policy rejected this" — removes a large class of self-inflicted load and gives you the data to decide whether anything else in this article applies to you.
