# Saved $23k by rewriting backend for Lagos traffic

This is a topic where the standard advice is technically correct but practically misleading. Here's the fuller picture, based on what I've seen work at scale.

## The situation (what we were trying to solve)

In late 2022, our Lagos-based fintech, PayPesa, launched a new micro-lending product targeting merchants in open-air markets. We expected 5,000 daily loan applications. The first week hit 12,000. By week three, we were at 47,000. Our stack was a Node.js API behind an AWS ALB in us-east-1, with PostgreSQL RDS in eu-central-1. I chose that region because it had the best latency to our US investors’ dashboards, not because of African traffic patterns.

The first failure was DNS. We used Route 53 with latency-based routing. Nigerian users were routed to eu-central-1 or us-east-1, adding 300–400ms to every TCP handshake. Our synthetic Pingdom checks in Lagos showed 180ms pings to eu-central-1, but real users in Balogun Market were seeing 550ms page loads and 1.2s API calls. The product team reported drop-off at the loan application step: 42% of users who reached the form never submitted it.

Our infrastructure bill was $8,200 that month. The AWS calculator projected $27,000 at 100k daily users. We knew we’d hit that in eight weeks. The board asked for a latency plan and a cost ceiling.

**Summary:** We launched globally but served Africa via European and US regions, causing 400ms+ latency for real users and an $18k gap between projected and actual AWS bills.

## What we tried first and why it didn’t work

First attempt: AWS Global Accelerator. We set it up in one afternoon. Traffic from Lagos hit the accelerator endpoint, which tunneled to eu-central-1. Page load times dropped to 380ms—better, but still painful for form-heavy UIs. Cost: $1,100/month for 30k requests/day. The latency improvement didn’t justify the bill.

Second attempt: Cloudflare Workers for dynamic request routing. We wrote a simple worker that checked the user’s IP against a MaxMind GeoIP2 database and forwarded traffic to the nearest AWS region. The worker itself was 40 lines of JavaScript, deployed in 5 minutes. But Cloudflare’s edge in Lagos (CPT) was still routing back to eu-central-1 for origin. We saw 240ms latency, but the origin fetch added 150ms, so we were at 390ms again. Cost: $900/month for the worker plus $1,800 for the extra Frankfurt RDS instance we spun up for redundancy.

Third attempt: Multi-region PostgreSQL with logical replication. We set up an RDS read-replica in ap-south-1 (Mumbai) and used pgpool-II to route reads. Writes still went to eu-central-1. The replication lag was 1.8 seconds during peak. Loan data became stale mid-form, causing validation failures. We rolled it back after two days. Cost after rollback: $3,200 in wasted provisioned IOPS and snapshot storage.

**Summary:** We tried three quick fixes—Global Accelerator, Cloudflare Workers, and multi-region DB—but each added cost without solving the latency root cause: origin location.

## The approach that worked

The breakthrough came from reading Cloudflare’s 2022 Africa Network Map. Lagos, Johannesburg, and Nairobi all had Cloudflare edge POPs with direct fiber links to African IXPs. Instead of routing to Frankfurt or Virginia, we moved the origin to a Cloudflare R2 bucket for static assets and a Cloudflare Workers KV namespace for session state, with an originless setup for the API itself.

We rewrote the API in Go, compiled to a single binary, and deployed it as a Cloudflare Worker. The Worker accepted loan applications, validated them against a JSON schema, calculated risk scores using a tiny WASM model, and returned a decision in under 20ms—all within the Lagos POP. The database stayed in eu-central-1 for compliance, but we added a Cloudflare Durable Object for idempotency keys and a local SQLite cache in the Worker for hot loan data. 

The key insight was offloading state to the edge. We stopped sending every API call to the origin and started using Workers KV for user sessions, Durable Objects for request deduplication, and R2 for receipts and IDs. The result: 94% of API traffic never left Lagos.

**Summary:** Moving the compute to the edge, rewriting in Go, and using Workers KV/Durable Objects cut origin dependency and reduced latency by 90%.

## Implementation details

We split the project into two phases. Phase 1 was the rewrite; Phase 2 was the edge deployment.

**Phase 1: Go rewrite**
We started with a Node.js API that was 3,200 lines. We rewrote it in Go 1.21, keeping the same endpoints but removing ORM overhead. The new binary was 14MB, started in 8ms, and handled 16k concurrent connections on a $5 DigitalOcean droplet in Lagos. The old Node.js service needed a c5.2xlarge ($165/month) to handle 4k concurrent connections. Memory usage dropped from 1.2GB to 80MB.

Code snippet: simplified loan validation in Go.
```go
package main

import (
	"encoding/json"
	"net/http"
	"time"
)

type LoanRequest struct {
	Amount      int       `json:"amount"`
	TenureDays  int       `json:"tenure_days"`
	MerchantID  string    `json:"merchant_id"`
	BusinessAge int       `json:"business_age_months"`
	LastRepayment time.Time `json:"last_repayment_at"`
}

func validateLoan(w http.ResponseWriter, r *http.Request) {
	decoder := json.NewDecoder(r.Body)
	var req LoanRequest
	if err := decoder.Decode(&req); err != nil {
		http.Error(w, "invalid_json", http.StatusBadRequest)
		return
	}
	
	// Basic risk checks
	if req.Amount < 1000 || req.Amount > 500000 {
		http.Error(w, "amount_out_of_range", http.StatusBadRequest)
		return
	}
	if req.TenureDays < 7 || req.TenureDays > 90 {
		http.Error(w, "tenure_invalid", http.StatusBadRequest)
		return
	}
	if req.BusinessAge < 3 {
		http.Error(w, "business_too_new", http.StatusBadRequest)
		return
	}
	
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"approved": true,
		"reason":   "passed_basic_checks",
	})
}

func main() {
	http.HandleFunc("/validate", validateLoan)
	http.ListenAndServe(":8080", nil)
}
```

**Phase 2: Cloudflare Workers deployment**
We compiled the Go binary to WASM using TinyGo 0.28 and wrapped it in a Worker. The Worker script was 200 lines:
```javascript
// worker.js
export default {
  async fetch(request, env) {
    const url = new URL(request.url)
    if (url.pathname === '/validate') {
      const req = await request.json()
      const result = await GO_WASM.validateLoan(req)
      return new Response(JSON.stringify(result), {
        headers: { 'Content-Type': 'application/json' }
      })
    }
    return new Response('Not found', { status: 404 })
  }
}
```

We used Durable Objects for idempotency keys to prevent duplicate loan submissions from flaky connections. One Durable Object instance per merchant ID kept state in memory, synced to Workers KV every 5 seconds. We stored receipts in R2 with lifecycle rules to delete after 30 days.

**Cost control**
Workers cost $5 per million requests. Workers KV is $0.50 per GB per month. Durable Objects are $0.50 per active object per month. We capped Durable Objects at 20k merchants, which cost $10k/month at peak. We also set Cloudflare caching for static assets (CSS, JS) at 1 hour TTL, reducing bandwidth by 68%.

**Summary:** Rewriting in Go and deploying as a Cloudflare Worker with Durable Objects and R2 cut both latency and complexity, while capping edge costs at predictable rates.

## Results — the numbers before and after

**Latency**
- Before: 1.2s median API response time, 4.1s p95
- After: 18ms median, 45ms p95
- Improvement: 98% reduction in median, 99% in p95

**User drop-off**
- Before: 42% of users abandoned at loan form submission
- After: 8% abandonment during submission
- Improvement: 34 percentage points lower drop-off

**Cost**
- Monthly AWS bill before: $8,200 at 47k users
- Projected AWS bill at 100k users: $27,000
- Actual bill after rewrite: $4,200 (Cloudflare + DO droplet)
- Savings: $22,800 per month at projected scale
- Payback period: 12 days

**Scalability**
- Our DigitalOcean droplet handled 16k concurrent connections at 15% CPU
- Cloudflare Workers auto-scaled to 120k requests/minute during a flash sale
- No database changes were needed; origin stayed in eu-central-1 for compliance

**Reliability**
- We measured 99.95% uptime over 90 days vs 99.7% before
- Durable Objects reduced duplicate submissions by 92% during network flaps

**Summary:** After the rewrite, median API latency dropped from 1.2s to 18ms, user drop-off fell by 34 points, and monthly costs dropped from a projected $27k to $4.2k, with a 12-day payback.

## What we’d do differently

1. **Don’t move the database to the edge**
We considered replicating PostgreSQL to Johannesburg or Lagos, but compliance rules forbade cross-border writes. We should have asked for regional write replicas from day one. Instead, we accepted 150ms write latency to eu-central-1, which hurt loan confirmation speed. A regional read-replica with logical replication would have cut that to 30ms.

2. **Cache risk models at the edge**
The WASM risk model was 2MB and loaded on every request. We should have precomputed risk scores for common merchant profiles and cached them in Workers KV with a 10-minute TTL. That would have cut WASM load time from 8ms to 0.3ms.

3. **Use Workers Analytics Engine from day one**
We added it after two weeks. It showed that 34% of errors were schema validation failures due to mobile keyboards sending `,` instead of `.` in decimal fields. We fixed the frontend validation in one day, but if we’d had the analytics earlier, we could have saved 8 hours of debugging.

4. **Avoid TinyGo for complex logic**
TinyGo 0.28 didn’t support CGO, so we couldn’t use some standard library packages. We spent three days rewriting a UUID generator. Next time, we’ll use Go 1.21 with WASMGC or stick to Cloudflare’s Rust/WASM toolchain for heavy lifting.

**Summary:** We’d move to regional DB replicas, cache risk models, add analytics upfront, and avoid TinyGo for production logic to save time and improve accuracy.

## The broader lesson

The lesson isn’t about Cloudflare Workers or Go. It’s that **origin location is the new bottleneck**. When you serve global users, the cost and latency penalty of routing to a far-off origin isn’t just network overhead—it’s compounded by ORM roundtrips, JSON parsing, and validation logic that all execute on distant CPUs. Moving compute to the edge doesn’t just cut latency; it forces you to simplify state management, reduce roundtrips, and cache aggressively. The result is orders-of-magnitude cheaper infrastructure and happier users.

Teams that still treat Africa as a ‘secondary market’ route traffic to EU or US origins and then wonder why their AWS bill explodes and their user drop-off is high. The fix isn’t a bigger database or a faster CDN—it’s moving the compute to where the users are, even if it means rewriting the API in a language that compiles to WASM.

Another surprise: the edge forces you to confront state hygiene. Durable Objects, Workers KV, and R2 make you think carefully about where state lives and when it syncs. That discipline reduces bugs and makes the system easier to reason about. We went from 3,200 lines of Node.js to 1,800 lines of Go and 200 lines of Worker JS, with fewer moving parts.

**Summary:** Origin location is the new bottleneck: moving compute to the edge reduces latency by 90%+, cuts costs by 85%, and forces state discipline that reduces bugs.

## How to apply this to your situation

1. **Measure first**
Use WebPageTest or Lighthouse from Lagos, Nairobi, and Johannesburg. Record median and p95 API latency and page load times. If your origin is outside Africa, expect +300ms to +600ms. That’s your ceiling improvement.

2. **Rewrite the API in Go or Rust**
Compile to WASM and deploy on Cloudflare Workers, Fly.io, or Deno Deploy. Avoid Node.js for edge-first APIs—its startup time and memory usage are killers.

3. **Offload state aggressively**
- Use Durable Objects for user sessions and idempotency keys
- Use Workers KV for hot data (merchant profiles, risk scores)
- Use R2 for receipts, IDs, and static assets
- Keep the origin only for writes that require ACID compliance

4. **Set strict cost caps**
In Cloudflare, set Workers KV max storage to 1GB and Durable Objects to 10k instances. In Fly.io, cap the number of VMs to 5. This prevents bill shock during flash sales.

5. **Add edge analytics**
Enable Workers Analytics Engine or Cloudflare Logs. Look for patterns in errors, timeouts, and validation failures. Most teams I’ve seen fix 2–3 critical issues within a week of adding analytics.

6. **Compliance check**
If your origin must stay in one region, add a regional read-replica and route reads to the edge. Write latency will still hurt, but read performance will improve.

**Next step:** Clone the PayPesa edge template from our GitHub (link below), deploy it to Cloudflare Workers, and run a synthetic load test from Lagos. Measure latency and cost for one week. If your median API time drops below 50ms and your bill stays under $100, you’ve validated the approach.

**Summary:** Measure latency from African locations, rewrite your API in Go/Rust for WASM, offload state to Durable Objects/KV/R2, set cost caps, add edge analytics, and validate with a one-week test.

## Resources that helped

- [Cloudflare Africa Network Map 2022](https://www.cloudflare.com/network-map/) – showed us direct fiber links in Lagos, Johannesburg, and Nairobi
- [TinyGo 0.28 release notes](https://tinygo.org/releases/0.28.0/) – we used it for Go-to-WASM, but later switched to Go 1.21 WASMGC
- [Cloudflare Workers KV docs](https://developers.cloudflare.com/workers/learning/how-kv-works) – taught us how to structure edge data
- [Go 1.21 WASMGC proposal](https://github.com/golang/go/issues/42372) – better than TinyGo for larger binaries
- [DigitalOcean Lagos datacenter](https://www.digitalocean.com/docs/platform/availability-matrix/) – cheap fallback for origin when Cloudflare had hiccups
- [WebPageTest Africa nodes](https://www.webpagetest.org/) – used to verify latency improvements
- [Workers Analytics Engine guide](https://developers.cloudflare.com/analytics/analytics-engine/) – showed us how to track edge errors

**Summary:** These resources—Cloudflare’s Africa map, Go WASM, Workers KV docs, and WebPageTest nodes—were critical to designing and validating our edge-first rewrite.

## Frequently Asked Questions

**How do I handle database writes from the edge if my origin is in Europe?**
Use a regional read-replica for reads and logical replication to keep it near real-time. Accept 100–300ms write latency to the origin. If ACID is critical, batch writes and use a queue (e.g., Cloudflare Queues or AWS SQS) to reduce roundtrips. We saw 99.9% data consistency with 250ms average replication lag during peak.

**What if my compliance rules require all data to stay in-country?**
Deploy a regional read-replica in the same country as your users, but keep writes to the origin only. Use a VPN or private link to the origin for writes. This adds complexity but keeps compliance. In Nigeria, we used a DigitalOcean droplet in Lagos for edge compute and a private VPC link to our Frankfurt RDS for writes.

**Is Go the only option for edge-first APIs?**
No. Rust, Zig, and even C compiled to WASM work. Cloudflare also supports Python (Pyodide), but startup time is higher. We benchmarked Go vs Rust; Go’s 8ms startup beat Rust’s 22ms at 16k concurrency. If you need a full runtime (e.g., for Python libraries), consider Deno or Bun on Fly.io instead of Workers.

**How do I prevent Cloudflare Workers bill shock during flash sales?**
Set a Workers KV storage limit (e.g., 1GB) and Durable Objects to 10k instances. Use Cloudflare’s cost estimator: it’s $5 per million requests and $0.50 per GB for KV. During our flash sale, we hit 120k requests/minute and paid $870 for the day—well below our $2k cap. Always set billing alerts in Cloudflare.

**Can I use this approach for mobile apps or IoT devices?**
Yes. We used the same Workers endpoint for our React Native app and for USSD endpoints. The Workers script is just an HTTP handler; clients don’t care where it runs. We saw 18ms latency from a USSD device in Lagos to the Worker, compared to 450ms to our old EU origin.

**What tools did you use to monitor latency and errors at the edge?**
Cloudflare Logs (100k events/day), Workers Analytics Engine, and a small Grafana dashboard pulling from Cloudflare’s GraphQL API. We also used Sentry for error tracking, with a custom integration to capture Workers console logs. The combination gave us sub-2-second visibility into edge errors.

**Summary:** These FAQs cover compliance, language choices, cost controls, and monitoring for edge-first APIs—practical answers from real deployments.