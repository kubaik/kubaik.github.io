# Got a remote tech job: what actually worked

The thing that frustrated me most when learning this was that every tutorial assumed a clean slate. Real systems never are. Here's how it actually goes.

## The error and why it's confusing

You’ve shipped a project, aced the take-home, and nailed the technical screen. Then the recruiter calls with the same excuse: “We’ve gone with another candidate.” The rejection email reads like a template—“cultural fit,” “team dynamics,” “budget constraints.” None of it feels real, because your code worked, your GitHub is clean, and your LeetCode scores are solid. The disconnect isn’t in the code or the algorithm. It’s in the signals you thought mattered versus the signals that actually move the needle. Most bootcamp grads and junior devs fixate on the wrong metrics: perfecting a personal website with 100% lighthouse scores, writing 300-line blog posts about one API call, or chasing 100k GitHub stars on a toy project. Those things don’t scale to a distributed team where the real question is: “Can this person ship something that doesn’t break when the timezone changes and the on-call engineer is asleep?”

I burned six months optimizing a full-stack app for Web Vitals before realizing no hiring manager ever asked for those numbers. The actual pain point was: the app crashed at 3 a.m. UTC when the database connection timed out after 30 minutes of idle time. The recruiter never said that. They said “cultural fit.” What tipped the scale was the GitHub README that showed a one-line `psql` command that fixed the idle timeout. The hiring manager replied within 48 hours.

The error isn’t the rejection email—it’s the assumption that “good code” is the same as “production-ready.” It isn’t. Good code is code that works when people you can’t see are using it at 2 a.m. on a Saturday.

---

## What's actually causing it (the real reason, not the surface symptom)

The hiring pipeline isn’t optimized for “best code.” It’s optimized for “lowest risk of pager duty tonight.” The signals that matter are the ones that prove you can prevent outages, not the ones that prove you can write a sorting algorithm. Most junior developers over-index on LeetCode and under-index on incident logs, uptime dashboards, and rollback scripts. The real failure isn’t in the code review; it’s in the absence of evidence that the code will survive a real load.

I learned this the hard way when I joined a startup as the first backend hire. My onboarding ticket was to add a feature flag to an existing microservice. The service used an in-memory cache that evicted keys after 5 minutes. The first time we deployed to staging, the cache warmed up, the service OOM-killed, and the entire staging environment crashed for 12 minutes. The incident report landed in my lap with a single question: “Why didn’t this happen in your local tests?” The answer was brutal: my local Docker image had a 2 GB memory limit, while staging had 512 MB. The cache eviction policy was written for “ideal conditions,” not “out of memory.”

The root cause wasn’t the code. It was the assumption that staging would behave like my laptop. The real metric that matters is: “Does the system degrade gracefully when resources are constrained?” Most junior developers don’t build that metric into their projects. They build features, not resilience.

---

## Fix 1 — the most common cause

**Symptom pattern:** You get ghosted after the take-home or the technical screen. The recruiter says “cultural fit” or “team dynamics,” but the real signal they’re missing is: “Can you handle a flaky connection?” The most common cause is that your project doesn’t run behind a proxy or a CDN. Many junior devs build apps that assume `localhost:3000` and a stable internet connection. In production, those assumptions collapse under latency, DNS timeouts, and regional outages.

**Fix:** Run your project behind a reverse proxy (Nginx or Caddy) and simulate a slow network with `tc` (traffic control). Add a 200 ms delay and 1% packet loss. If the app still works, you’re halfway there. If it crashes or times out, you’ve found the symptom that kills 80% of remote job pipelines.

```bash
# Install traffic control (Linux/macOS)
sudo apt install iproute2

# Add 200 ms delay and 1% packet loss
sudo tc qdisc add dev lo root netem delay 200ms loss 1%

# Run your app on port 3000
npm start

# Test with curl
curl -v http://localhost:3000/api/health
```

After adding the delay, I watched a React app I built for a take-home crash because the frontend assumed the API would respond in under 100 ms. The recruiter never said “your frontend is too fragile.” They just moved on to the next candidate. The fix was to add a 500 ms timeout in the Axios config and a loading skeleton. The app passed the next round.

**Summary:** If your project doesn’t survive a 200 ms delay and 1% packet loss, it won’t survive a real user’s connection. Run the traffic control test before you submit the take-home.

---

## Fix 2 — the less obvious cause

**Symptom pattern:** Your LinkedIn profile shows 500+ connections, but recruiters still send you generic “we’re hiring” messages. The less obvious cause is that your GitHub profile doesn’t show a single project that deploys to a real domain with HTTPS and a custom subdomain. Most junior devs host projects on `localhost`, GitHub Pages, or Render’s ephemeral URLs. Those domains don’t count as “production” because they expire, change, or get rate-limited.

**Fix:** Deploy one project to a real domain with HTTPS and a custom subdomain. Use Fly.io, Railway, or Render. Add a CI/CD pipeline that runs tests on every push. The domain doesn’t need to be fancy—just `api.yourname.dev` or `app.yourname.dev`. The key is that the URL stays the same, the TLS certificate is valid, and the endpoint responds within 500 ms under load.

```yaml
# Fly.io fly.toml example
app = "my-api"

[[services]]
  internal_port = 3000
  protocol = "tcp"

  [[services.ports]]
    port = 80
    handlers = ["http"]

  [[services.ports]]
    port = 443
    handlers = ["tls", "http"]
```

I spent three weeks building a personal portfolio with a React frontend and a Node backend. I hosted it on Vercel and Netlify, but both URLs were ephemeral. When I added `api.kevin.dev` on Fly.io with a PostgreSQL database, the recruiter I’d been chatting with for two months finally replied: “Your API is live? Can you show me the uptime graph?” I added a Uptime Kuma dashboard and linked it in my README. She scheduled an onsite the next day.

**Summary:** A project on a real domain with HTTPS and uptime monitoring is worth 10 GitHub stars. Deploy one project this week and link it in your profile.

---

## Fix 3 — the environment-specific cause

**Symptom pattern:** You pass the technical screen but fail the final round because the hiring manager asks: “How would you debug a memory leak in a Node.js service that only happens in production?” The environment-specific cause is that your local tests don’t simulate memory pressure or container limits. Most junior devs test memory usage with `process.memoryUsage()` in dev tools, which shows 50 MB, but the staging environment runs in a 256 MB container. The leak only appears when the heap grows beyond 256 MB.

**Fix:** Simulate memory pressure in Docker with `memory-swappiness` and `memory-limit`. Add a script that allocates 100 MB every 100 ms until the container OOM-kills. If your app survives, you’ve proven it handles memory pressure. If not, you’ve found the leak before the hiring manager does.

```Dockerfile
# Dockerfile
FROM node:18-alpine

# Limit memory to 256 MB
RUN echo 'vm.swappiness=0' >> /etc/sysctl.conf

# Set memory limit in Docker run
CMD ["node", "--max-old-space-size=256", "server.js"]

# Run with memory limit
# docker run --memory=256m --memory-swappiness=0 my-node-app
```

```javascript
// server.js
setInterval(() => {
  const arr = new Array(1000000).fill(0);
  console.log('Memory:', process.memoryUsage().heapUsed / 1024 / 1024 + ' MB');
}, 100);
```

I built a GraphQL resolver that cached results in a global variable. Locally, the cache cleared after 10k requests. In staging, with 256 MB memory, the cache grew unbounded and OOM-killed after 12 minutes. The hiring manager’s question wasn’t theoretical—it was a trap. The fix was to add an LRU cache with a 100 MB limit. The app passed the final round.

**Summary:** If your app OOM-kills in a 256 MB container, it will OOM-kill in production. Simulate the limit before the interview.

---

## How to verify the fix worked

After you’ve deployed a project behind a proxy, added a real domain, and simulated memory pressure, the next step is to prove it works under load. The fastest way is to use k6 to simulate 100 concurrent users for 5 minutes. If the error rate is under 0.1% and p95 latency is under 500 ms, you’ve met the bar most remote teams use for “production-ready.”

```javascript
// k6 script
import http from 'k6/http';
import { check } from 'k6';

export let options = {
  vus: 100,
  duration: '5m',
};

export default function() {
  let res = http.get('https://api.yourname.dev/health');
  check(res, {
    'status is 200': (r) => r.status === 200,
    'response time < 500ms': (r) => r.timings.duration < 500,
  });
}
```

I ran this exact script on a project I built for a take-home. The first run failed with 12% errors because the database connection pool was set to 5. I increased it to 50, reran the test, and the error rate dropped to 0.1%. The recruiter replied within an hour with an offer. The key wasn’t the code—it was the proof that the code could handle load.

**Summary:** Run a 5-minute load test with 100 users. If error rate < 0.1% and p95 latency < 500 ms, you’ve proven production readiness.

---

## How to prevent this from happening again

The fastest way to avoid the “ghosted after take-home” trap is to build a single project that runs the entire pipeline: code → test → build → deploy → monitor → alert. The project doesn’t need to be complex—just a REST API with a database, a frontend, and a CI/CD pipeline that deploys to a real domain. The moment you automate the pipeline, you stop assuming and start proving.

I built a URL shortener with Next.js, PostgreSQL, and Fly.io. The entire pipeline is defined in a single `fly.toml` and a GitHub Actions workflow. Every push triggers a deploy to `short.kevin.dev`. The pipeline includes a health check, a load test, and an uptime monitor. When the recruiter asked for a production example, I sent the domain and the uptime graph. No follow-up questions.

**Prevention checklist:**
- One project with a real domain and HTTPS
- CI/CD pipeline that runs tests and deploys on every push
- Load test that simulates 100 users for 5 minutes
- Uptime monitor with a public dashboard
- README that links to the domain, the dashboard, and the load test results

**Summary:** Automate one project from code to monitor. The moment you stop assuming and start proving, the rejections turn into offers.

---

## Related errors you might hit next

- **502 Bad Gateway on Fly.io:** The app crashes immediately after deploy. Fix: Increase memory limit in `fly.toml` from 256 MB to 512 MB and add a health check endpoint.
- **PostgreSQL connection refused on Railway:** The database URL in the environment variables is malformed. Fix: Use `postgresql://user:pass@host:port/db` and escape special characters in the password.
- **CORS errors on production frontend:** The backend doesn’t set `Access-Control-Allow-Origin` for the production domain. Fix: Add the domain to the CORS middleware and redeploy.
- **GitHub Actions deploy fails with “no space left on device”:** The runner’s disk is full. Fix: Add a cleanup step in the workflow to remove node_modules after build.

**Summary:** These errors are symptoms of missing production checks. Add them to your pipeline before the next interview.

---

## When none of these work: escalation path

If you’ve deployed a project with a real domain, simulated memory pressure, run a 100-user load test, and still get ghosted, the issue isn’t technical—it’s signaling. The hiring manager wants proof that you can prevent outages, not that you can write a sorting algorithm. The escalation path is to add an incident log to your README that shows how you handled a simulated outage.

**How to escalate:**
1. Simulate an outage by killing the database container.
2. Write a rollback script that reverts to the last stable image.
3. Add the script to your project’s README with a timestamp and a one-line explanation.
4. Link the README in your LinkedIn profile.

I did this after three rejections. I killed the PostgreSQL container, ran the rollback script, and the app was back online in 30 seconds. I added the log to my README and sent it to the recruiter. She replied: “This is the first candidate who sent proof they can handle a real outage.” The offer arrived the next day.

**Summary:** If rejections persist, simulate an outage, write a rollback script, and document it. Send the log to the recruiter. That’s the signal that moves the needle.

---

## Frequently Asked Questions

**What’s the minimum project I need to get a remote job?**
Build one project that runs behind a real domain with HTTPS and a CI/CD pipeline. It doesn’t need to be complex—a URL shortener, a todo API, or a weather app. The key is that it deploys to a domain that doesn’t change and responds under load. Most junior devs overbuild projects with 10 microservices. One small project with a clear README and uptime dashboard is enough.

**How do I explain memory leaks in an interview if I’ve never seen one?**
Say: “I simulated memory pressure in Docker with a 256 MB limit and a leaky script. The app OOM-killed after 12 minutes. I fixed it by adding an LRU cache with a 100 MB limit. The error rate dropped to 0%.” Bring the Dockerfile, the script, and the fix to the interview. The hiring manager isn’t testing your knowledge—they’re testing your debugging process.

**Why do recruiters ignore GitHub with 100+ stars but reply to a project on a real domain?**
GitHub stars measure popularity, not production readiness. A real domain measures uptime, latency, and resilience. Recruiters look for signals that you can prevent outages, not that you can write a sorting algorithm. The domain is proof that the project survived a real load.

**What’s the fastest way to build credibility with a hiring manager?**
Automate a rollback script for your project. Simulate an outage, run the script, and document the log. Send the log to the recruiter with a one-line explanation: “This is how I’d handle a production outage.” Most candidates don’t do this. The ones who do get offers.

---

## The one project that gets remote offers

| Project | Domain | Load Test | Uptime Monitor | Rollback Script |
|---|---|---|---|---|
| URL shortener | short.yourname.dev | 100 users, 5 min, 0.1% errors | UptimeRobot dashboard | Fly.io rollback command |
| Todo API | api.yourname.dev | 100 users, 5 min, 0.1% errors | Uptime Kuma dashboard | Railway rollback button |
| Weather app | weather.yourname.dev | 100 users, 5 min, 0.1% errors | Freshping dashboard | Render rollback command |

The table above lists the minimal viable projects that get remote offers. Pick one, deploy it to a real domain, run the load test, and add the uptime monitor and rollback script to your README. Send the README link to the recruiter. That’s the signal that moves the needle from “cultural fit” to “hired.”

**Summary:** Build one project. Deploy it to a real domain. Add a load test, an uptime monitor, and a rollback script. Link the README in your profile. That’s the project that gets remote offers.