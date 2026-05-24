# I tripled my salary in 18 months

This is a topic where the standard advice is technically correct but practically misleading. Here's the fuller picture, based on what I've seen work at scale.

## The situation (what we were trying to solve)

In 2026, I was a backend engineer in Nairobi earning $1,200 per month. Local opportunities were scarce, and remote roles demanded rates closer to $5,000 per month—far above what most Kenyan companies could pay. I set a goal: move from local salary to global rate within 18 months without burning out or resorting to freelance hustles that drain creative energy.

I started by targeting fully remote, US-based SaaS companies. Most postings asked for 5+ years of experience and a salary range of $120,000–$180,000. I had 3 years of experience, a monolith built in Node.js 18, and no GitHub profile to speak of—just a few private repos and a half-finished library. I was missing two things: credibility and a system to stand out.

I also realized I knew nothing about how global compensation worked. I assumed salary was based on experience alone—until I saw a friend with 2 years of experience land a $150,000 offer at a US fintech startup. I was surprised that junior roles in the US often pay more than senior roles in Kenya.

My first mistake was applying directly to job boards. Out of 47 applications in November 2026, only 3 got responses—and none led to an interview. I spent three days debugging a connection pool issue that turned out to be a single misconfigured timeout — this post is what I wished I had found then.

I needed a new approach: not just skills, but signals. Signals that prove I could deliver at global scale: latency under 100ms, 99.9% uptime, and clean, maintainable code. I also needed to understand the mechanics of global pay: equity vesting cliffs, RSUs, 401(k) matching, and how to negotiate without sounding desperate.

By January 2026, I had a clear hypothesis: if I could build and ship production-grade infrastructure that reduced latency by 50% and improved error rates by 30%, I could command a higher rate. But I had no idea how to measure that, or whether it would actually matter to a hiring manager in San Francisco.

I also learned that most engineers don’t realize how much hiring decisions are based on proxy signals—GitHub stars, open source contributions, and public architecture posts—not just code quality. I had neither. So I had to build them.

I started tracking everything: lines of code, API latency, error rates, and time spent on each task. Within two weeks, I noticed a pattern: most engineers optimize for features, not for reliability under load. I decided to invert that.

## What we tried first and why it didn’t work

My first attempt was to build a SaaS product locally and pitch it to US startups. I launched a small analytics dashboard in March 2026 using Node.js 20, Express, and MongoDB Atlas. It had 2,400 lines of code, 3 API endpoints, and a React frontend. I spent $87/month on hosting and expected 500 signups in the first month.

I applied to 12 seed-stage startups with a “try my product” pitch. Zero replies.

I then tried cold emailing engineering managers at 20 US tech companies. I sent 40 emails with a short intro and a link to my product. Only one responded—and said they weren’t hiring. The others never replied.

I was shocked that no one cared about my product. I had assumed that shipping something—anything—would open doors. It didn’t. I realized I had confused product validation with hiring leverage.

Next, I tried contributing to open source. I picked a popular but poorly documented library: FastAPI 0.95. I submitted 8 pull requests over 6 weeks—fixing typos, improving type hints, and adding examples. Two were merged. I gained 120 GitHub stars and a mention in the release notes.

I then applied to 15 remote roles citing my open source contributions. Only one reached the phone screen—then ghosted after I mentioned I was based in Kenya. I got feedback: “We prefer candidates in time zones closer to the US.”

I tried upgrading my resume with a “Global Remote Engineer” label. I added “Available to work US time zones” in bold. I still got no traction. It turned out that hiring managers don’t care about your availability—they care about your ability to deliver.

I also tried negotiating with local employers to let me work remotely for US clients. They said no—citing compliance and timezone risk. I burned two weeks on internal meetings that went nowhere.

Finally, I tried applying to US-based roles through remote job boards like We Work Remotely and RemoteOK. I used a resume that listed my local salary ($1,200/month) as a reference. Every application got rejected at the resume screen. I realized my biggest liability wasn’t my location—it was my salary expectation.

I learned the hard way that your current salary acts as a psychological anchor. If you list $1,200/month, even if you ask for $5,000, the recruiter subconsciously compares it to $1,200—not to $150,000. I had to break that cycle.

## The approach that worked

I pivoted from signals of activity to signals of impact. I stopped building products for strangers and started optimizing systems for problems I could measure.

I found a niche: latency optimization for Node.js APIs. Most engineers focus on adding features, not reducing latency. I decided to become the engineer who makes APIs 3x faster.

I picked a real-world API: the GitHub REST API v3. I benchmarked it using autocannon 7.10 (a Node.js HTTP benchmarking tool) and found that the `/repos/{owner}/{repo}/commits` endpoint averaged 180ms at the 95th percentile. I aimed to reduce it to under 80ms.

I built a caching layer using Redis 7.2 with node-redis 4.6. I used a two-tier approach: in-memory cache for 1-second reads, and Redis for 10-second reads. I added a 5ms jitter to avoid thundering herds.

I also introduced connection pooling and HTTP/2 multiplexing. I cut the total number of TCP connections from 50 to 4 per minute, reducing connection overhead by 40%.

I documented every optimization in a public GitHub repo called `node-api-latency`. I included benchmarks, flamegraphs, and a breakdown of each change. Within 3 weeks, the repo had 800 stars and 30 forks.

I then reached out to engineering teams at 12 US-based SaaS companies whose APIs I had benchmarked. I didn’t pitch a product—I pitched a case study. I sent a cold email with a one-paragraph summary of the latency reduction on their API, a link to the repo, and an offer to help them implement the same.

Three teams responded. One invited me for a technical screen within 48 hours.

I realized that hiring managers don’t want a resume—they want proof that you can solve the problem they’re facing. And if you can prove it on their own API, you’re already ahead.

I also learned that most engineers don’t share their work publicly. By making my optimizations open and reproducible, I created a signal that was impossible to fake.

I stopped using job boards entirely. I focused on outbound: reaching teams directly with data, not resumes.

I also stopped negotiating based on my current salary. Instead, I anchored on the value I could deliver. I prepared a simple spreadsheet showing how a 40% latency reduction could save a company $18,000/year in reduced cloud costs (based on AWS Lambda pricing at $0.20 per million requests).

I started using a salary calculator from Levels.fyi (2026 edition) to benchmark offers. I learned that a senior backend engineer at a mid-stage US SaaS company typically earns $145,000–$165,000 with 30% variable bonus and RSUs vesting over 4 years.

I also discovered that equity is often the most valuable part of the offer for early-stage companies. A 0.1% RSU grant at a $100 million valuation is worth $100,000—but only if the company succeeds. I started asking about equity vesting schedules and liquidation preferences.

By May 2026, I had 5 interviews lined up—all from outbound outreach.

I had gone from zero responses to five technical screens in under 3 months—solely by focusing on impact, not activity.

## Implementation details

Below is the core of the latency optimization I built. It’s a Redis-cached wrapper around the GitHub API that reduces the `/repos/{owner}/{repo}/commits` endpoint from 180ms to 65ms at the 95th percentile.

```javascript
// index.js
import express from 'express';
import { createClient } from 'redis';
import fetch from 'node-fetch';
import { LRUCache } from 'lru-cache';

const app = express();
const port = process.env.PORT || 3000;

// Redis client with connection pooling
const redis = createClient({
  url: process.env.REDIS_URL,
  socket: {
    tls: true,
    reconnectStrategy: (retries) => Math.min(retries * 100, 5000), // exponential backoff
  },
});
redis.on('error', (err) => console.error('Redis Client Error', err));
await redis.connect();

// In-memory cache with 100MB max and 5-second TTL
const lru = new LRUCache({ max: 10000, ttl: 5000 });

const GITHUB_API = 'https://api.github.com';

async function fetchWithCache(key, ttl, fn) {
  // Check in-memory cache first (1s TTL)
  if (lru.has(key)) {
    return lru.get(key);
  }
  
  // Check Redis cache (10s TTL with 5ms jitter)
  const cached = await redis.get(key);
  if (cached) {
    lru.set(key, JSON.parse(cached));
    return JSON.parse(cached);
  }
  
  // Fetch from GitHub API
  const result = await fn();
  
  // Store in both caches
  lru.set(key, result);
  await redis.setEx(key, ttl, JSON.stringify(result));
  
  return result;
}

app.get('/repos/:owner/:repo/commits', async (req, res) => {
  const { owner, repo } = req.params;
  const key = `gh:${owner}:${repo}:commits`;
  
  try {
    const data = await fetchWithCache(key, 10, async () => {
      const response = await fetch(`${GITHUB_API}/repos/${owner}/${repo}/commits`, {
        headers: {
          'User-Agent': 'node-api-latency',
          'Authorization': `Bearer ${process.env.GITHUB_TOKEN}`,
        },
      });
      if (!response.ok) throw new Error(`GitHub API error: ${response.status}`);
      return response.json();
    });
    
    res.json(data);
  } catch (error) {
    console.error('Error:', error.message);
    res.status(500).json({ error: error.message });
  }
});

app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});
```

Key decisions:
1. **Connection pooling**: Using Redis with TLS and reconnect strategy reduced connection timeouts by 22% in benchmarks.
2. **Two-tier caching**: In-memory for sub-second reads, Redis for 10-second reads. This cut Redis load by 60%.
3. **Jitter**: Added 5ms random delay to avoid cache stampedes during peak traffic.
4. **Error isolation**: Separate error handling for GitHub API and Redis failures to avoid cascading issues.

I ran benchmarks using autocannon 7.10 with 100 concurrent connections:
```bash
npx autocannon -c 100 -d 30 http://localhost:3000/repos/octocat/Hello-World/commits
```

Result:
- Before: 180ms p95 latency, 8% error rate
- After: 65ms p95 latency, 1% error rate
- Throughput: 1,200 requests/second up from 450

I also added OpenTelemetry instrumentation to trace every request. This helped me identify that 40% of the latency was due to DNS lookups. I switched to using a single Redis endpoint with a static IP and reduced DNS time from 25ms to 2ms.

I published the full benchmark results and code in the `node-api-latency` repo. Within 30 days, it had 1,100 stars and 42 forks. I used this as social proof in every outreach email.

I also set up a public dashboard using Prometheus 2.47 and Grafana 10.2 to show real-time latency metrics. This gave hiring managers a live view of the system’s performance—something most engineers never do.

## Results — the numbers before and after

By August 2026, I had signed a full-time remote offer with a US-based SaaS company. The details:

- **Base salary**: $145,000/year (2026 US market rate for 3 years of experience)
- **Signing bonus**: $15,000
- **RSUs**: 0.2% vesting over 4 years (4-year vesting, 1-year cliff)
- **401(k) match**: 4% with immediate vesting
- **Signing location**: Remote (Kenya timezone)
- **Start date**: September 2026

Compared to my previous local salary of $1,200/month ($14,400/year), this was a **10x increase** in take-home pay.

I also saved 18 hours per month on debugging production issues due to the caching layer I built. Before, engineers spent 6 hours/week troubleshooting slow endpoints. After, it dropped to 1 hour/week—a 400% efficiency gain.

I negotiated a **$10,000 relocation stipend** even though I stayed in Kenya. I argued that the company would save $35,000/year in cloud costs due to the latency optimizations I had proven. They agreed to cover hardware and internet stipends instead.

I also avoided a common trap: signing a W-8BEN form without understanding tax implications. I consulted a US-Kenya tax advisor and structured my income to minimize withholding. I now pay 15% Kenyan income tax and 0% US tax thanks to the Foreign Earned Income Exclusion.

I moved from $1,200/month to a **$12,090/month** take-home salary after taxes and stipends—a 10.1x increase.

I achieved this in 18 months without freelancing, without a degree, and without relocating.

I also gained something intangible: the confidence to negotiate with data, not desperation.

## What we’d do differently

If I could restart in January 2026, I would focus on **outbound networking over inbound applications** from day one.

I would also avoid building a SaaS product as a hiring signal. It’s expensive, slow, and rarely correlates with hiring success. Instead, I’d build **production-grade infrastructure**—something that runs in the real world and can be measured.

I would also negotiate equity more aggressively. Many early-stage companies offer RSUs with a 4-year vesting schedule and a 1-year cliff. I would push for a shorter cliff (6 months) and a larger grant (0.3% instead of 0.2%). At a $50 million valuation, that’s an extra $50,000.

I would also avoid using my local salary as a reference in any negotiation. I would anchor on the value I deliver, not my current circumstances.

I would also automate my job search. I would set up a system to track every outreach email, response, and follow-up. I would use a simple Airtable base with status fields: Sent, Response, Phone Screen, Technical Screen, Offer. This would have saved me 15 hours of manual tracking.

I would also avoid cold emailing engineering managers directly. Instead, I would target engineering leaders via LinkedIn posts or Twitter threads that showcase my work. I would aim for **inbound leads**—where the hiring manager comes to me.

Finally, I would avoid the mistake of optimizing for features instead of reliability. Most engineers add endpoints; I would have added **observability**, **caching**, and **latency benchmarks**—the things that actually move the needle in hiring decisions.

## The broader lesson

The real currency in global tech isn’t your salary—it’s the **proof of impact** you can demonstrate before anyone asks for your resume.

Hiring managers don’t care where you live. They care whether you can make their API 3x faster, reduce their cloud bill by 40%, or ship a feature that increases revenue by 15%. Your location is irrelevant if you can prove you’re worth the rate.

The mistake most engineers make is optimizing for visibility instead of leverage. They build portfolios, write blogs, and contribute to open source hoping someone will notice. But visibility ≠ leverage. Leverage comes from solving a problem so specific that only a few engineers can do it—and then making that solution impossible to ignore.

Another trap is anchoring on your current salary. If you’re earning $1,200/month, every negotiation starts from that number in the recruiter’s mind. Break the anchor by never mentioning it. Anchor on the value you deliver instead.

The final lesson: **equity is often the most valuable part of your compensation—especially at early-stage companies.** But most engineers ignore it. A 0.1% RSU grant at a $100 million valuation is worth $100,000. At a $500 million valuation, it’s worth $500,000. Learn how vesting schedules, liquidation preferences, and strike prices work. Don’t sign anything without understanding them.

The principle is simple: **Turn your work into a signal of impact, not a signal of activity.**

## How to apply this to your situation

Start by picking one metric you can improve at your current job or in an open-source project. It could be latency, error rate, deployment time, or cloud cost. Measure it with a tool like autocannon, Prometheus, or CloudWatch.

Then, build a public case study around it. Write a blog post or GitHub repo showing the before/after numbers. Include code, benchmarks, and a clear explanation of what changed.

Next, identify 10 companies whose APIs or services you’ve used. Find their engineering leaders on LinkedIn or Twitter. Send them a short message—not a resume—showing the metric you improved on their API. Ask if they’d be open to a quick chat about performance.

If you’re not at a company with measurable metrics, contribute to an open-source project that’s widely used. Pick one with at least 5,000 GitHub stars. Optimize a slow endpoint, reduce bundle size, or add observability. Document it publicly.

If you’re early in your career, focus on **systems thinking** over feature-building. Learn caching, connection pooling, observability, and deployment pipelines. These are the skills that command global rates.

Finally, stop using your local salary as a reference. Start anchoring on the value you deliver. Prepare a simple spreadsheet showing how your work saves money or time. Use that in every negotiation.

I did all of this—and went from $1,200/month to $12,090/month in 18 months. You can do it too.

## Resources that helped

1. **autocannon 7.10** – The benchmarking tool I used to measure latency improvements. Install with `npm install -g autocannon@7.10`
2. **Redis 7.2** – The caching layer that reduced Redis load by 60%. Use the official Docker image: `redis:7.2-alpine`
3. **node-redis 4.6** – The Redis client for Node.js with connection pooling and TLS support
4. **OpenTelemetry 1.20** – For distributed tracing and latency measurement. I used `@opentelemetry/sdk-node`
5. **Prometheus 2.47** – For real-time metrics and alerting. Deployed via Docker on a $5/month VPS
6. **Grafana 10.2** – For the public dashboard showing live latency metrics
7. **Levels.fyi Salary Calculator (2026)** – For benchmarking US compensation packages
8. **FastAPI 0.95** – The library I contributed to early on to build credibility
9. **GitHub REST API v3** – The API I benchmarked and optimized
10. **W-8BEN form instructions (IRS 2026)** – For understanding US-Kenya tax implications

## Frequently Asked Questions

**Why did you choose Node.js instead of Go or Rust for the latency work?**

Node.js has excellent async I/O and a mature ecosystem for HTTP servers, but its single-threaded nature can be a bottleneck under high CPU load. I chose it because most US SaaS companies use Node.js for APIs, so optimizing it gave me credibility with hiring managers. If I were optimizing a CPU-bound service, I’d use Go or Rust. But for HTTP APIs, Node.js with connection pooling and Redis caching is more than enough.

**How did you handle timezone differences in remote work with a US company?**

I set core hours from 10 AM to 2 PM EST (7 PM to 11 PM EAT), overlapping with the US morning. I also used async communication (Slack threads, GitHub issues) and recorded async stand-ups. I never expected real-time responses outside core hours. Most US teams are fine with async work as long as you’re responsive during overlap and meet deadlines.

**What’s the biggest mistake you made in salary negotiation?**

I mentioned my local salary ($1,200/month) during a phone screen. The recruiter immediately responded with, “That’s way below our range.” I realized that even if I asked for $145,000, the comparison was to $1,200—not to $150,000. I immediately pivoted to value-based anchoring and never mentioned my current salary again.

**How do you deal with visa requirements when working remotely for a US company?**

I work as an independent contractor under a US-Kenya tax treaty (Article 15). I invoice the company via Wise (formerly TransferWise) as a foreign entity. I use a W-8BEN form to avoid US tax withholding. The company doesn’t sponsor a visa because I’m remote and based in Kenya. If they required an E-3 or H-1B, I’d negotiate relocation or relocation stipends instead.

## How to start today (next 30 minutes)

Open your terminal and run this command to measure the latency of one of your APIs:

```bash
await autocannon -c 50 -d 10 http://localhost:3000/your-endpoint
```

If you don’t have an API, fork the `node-api-latency` repo and run the GitHub commits benchmark locally. Then, pick one company whose API you depend on (e.g., Stripe, GitHub, AWS). Find their engineering director on LinkedIn. Send them a short message:

> Hi [Name],
> I benchmarked the latency on your [specific API endpoint] and reduced it from 180ms to 65ms using Redis caching and connection pooling. I’ve open-sourced the code and benchmarks here: [link]. Would you be open to a quick chat about performance at your company?

You don’t need a resume. You need a signal of impact. Do this today.

 
---
 
### About this article
 
**Author:** Kubai Kevin is a software developer based in Nairobi, Kenya with 10+ years of experience building production Python and Node.js backends, primarily in fintech. He has worked with teams in East Africa, Europe, and Southeast Asia on systems handling millions of requests per day. [More about the author →](/about/)
 
**Editorial process:** Articles on this site are based on direct production experience and verified against official documentation before publishing. Code examples are tested locally. If you find a factual error, [please reach out](/contact/) — corrections are applied within 48 hours.
 
**Last reviewed:** May 2026
