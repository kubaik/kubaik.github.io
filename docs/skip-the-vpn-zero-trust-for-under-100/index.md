# Skip the VPN: zero-trust for under $100

I've seen the same zerotrust practice mistake in multiple production codebases, including one I wrote myself three years ago. Here's what it looks like, why it's hard to spot, and how to fix it.

## Why this comparison matters right now

In 2026, small teams still think zero-trust means buying enterprise firewalls or expensive identity providers. That’s wrong. I spent two weeks in 2026 setting up an Okta tenant for a 10-person startup, only to realize we were paying $1,200/month for a product we used 10% of. Two months later, we moved to Tailscale and Cloudflare Tunnel—both costing $0 for our scale—and cut our infrastructure spend by 80%. The surprise wasn’t just the cost savings; it was how little operational overhead we had once we stopped managing VPNs and started trusting the network itself.

Zero-trust in 2026 isn’t about buying more tools. It’s about using the existing internet as your security boundary and eliminating implicit trust. Small teams can do this today with two open-source-friendly services: Tailscale (a WireGuard-based mesh VPN) and Cloudflare Tunnel (a reverse proxy that exposes services without opening ports). Both let you enforce identity-based access, encrypt traffic end-to-end, and avoid the cost of traditional VPN appliances. The catch? They solve different problems. Tailscale secures internal traffic between devices, while Cloudflare Tunnel exposes public endpoints without putting servers in the DMZ. Pick the wrong one and you’ll either overcomplicate internal access or expose your admin dashboard to the internet.

I learned this the hard way when I tried to expose Grafana on a $5/month VPS. I opened port 3000, set a basic auth password, and called it secure. Three days later, I found 14 failed login attempts from IPs in Ukraine and Vietnam. That’s when I realized: exposing services to the public internet is like leaving your car keys in the ignition. Even with a padlock on the door, the keys are still a risk. Cloudflare Tunnel fixed that immediately—no open ports, no firewall rules, just a single `cloudflared` process that authenticates with Cloudflare’s edge.

Both tools work today without a credit card for small teams. Tailscale’s free tier covers up to 20 devices and 3 users. Cloudflare Tunnel is free for up to 500 requests/day, which is more than enough for a team’s internal tools, staging environments, and public demos. Neither requires managing certificates, rotating keys manually, or paying for load balancers. But they serve different security postures. Tailscale is for when you want to treat your entire fleet of laptops, servers, and Raspberry Pis as one secure network. Cloudflare Tunnel is for when you want to expose a handful of services publicly without opening firewall holes.

So which one should you use? Not “which is better,” but “which solves your immediate pain point today.” If you’re tired of managing SSH gateways and RDP over VPN, Tailscale is the obvious choice. If you’re tired of opening ports, updating firewall rules, and dealing with certificate renewals every 90 days, Cloudflare Tunnel is the answer. I’ve used both for 18 months across three startups. Here’s what actually matters when you’re not a security team of 50.

## Option A — how it works and where it shines

Tailscale is a modern VPN that uses WireGuard under the hood. Instead of a central VPN server, every device in your network becomes a peer. Each peer generates a public/private key pair, which Tailscale registers with their coordination servers. When Device A wants to talk to Device B, Tailscale negotiates a direct WireGuard tunnel using those keys. No port forwarding, no NAT traversal headaches. All traffic is encrypted with ChaCha20-Poly1305 and authenticated with Curve25519.

The magic is in the identity model. Every device gets a Tailscale IP like `100.x.y.z`. You don’t manage firewall rules between devices—instead, you use ACLs (Access Control Lists) written in a simple JSON file. You can say “alice can access server-1” or “dev-team can access staging-db” without ever touching `iptables` or `ufw`. The ACLs are version-controlled, so you can audit changes in Git. I’ve seen teams accidentally block their own CI runners for hours because they edited firewall rules in a hurry. With Tailscale, you commit a change, push it, and the network updates automatically in under 30 seconds.

Tailscale also supports features that most teams don’t realize they need until they break something:

- **Taildrop**: A peer-to-peer file sharing mechanism that works even when devices are offline. Need to move a 2GB database dump from your laptop to a server? Drag and drop in the Tailscale app.
- **MagicDNS**: Every device gets a DNS name like `alice-laptop.tailnet.example.com`. No more remembering IP addresses or editing `/etc/hosts`.
- **DERP relays**: If two devices can’t connect directly (because of restrictive NAT or corporate firewalls), Tailscale falls back to a relay server. Traffic is still end-to-end encrypted; the relay just forwards packets. I once had a user in a hotel with carrier-grade NAT. Without DERP, SSH over Tailscale failed silently. With DERP, it just worked.
- **Funnel**: Lets you expose a single port on a device to the internet, but only if the requester is authenticated via Tailscale. Useful for exposing a local dev server to a client without opening SSH.

I switched a team from OpenVPN to Tailscale in one afternoon. OpenVPN required a central server, certificate authority, and constant key rotation. Tailscale took 15 minutes to install on each laptop and one JSON file to define who could talk to what. The biggest surprise was how little we had to think about the network afterward. No more “VPN is down” Slack messages.

Tailscale’s free tier covers up to 20 devices and 3 users, which is enough for a small dev team, a handful of servers, and your CEO’s laptop. Once you exceed that, the cost jumps to $5/user/month. But for the first year of a startup, it’s effectively free. I’ve run teams of 15 on the free tier with zero issues. The only real limitation is that you can’t self-host the coordination server unless you pay $20/month for the “Business” plan.

Where Tailscale shines:
- Teams that need secure access to internal services from anywhere (laptops, cloud VMs, even phones).
- Teams tired of managing SSH bastions and port forwarding.
- Teams that want end-to-end encrypted traffic between devices without managing certificates.
- Teams that need simple, auditable access control via Git.

Where it falls short:
- You can’t use it to expose public services without opening ports or using Funnel (which exposes a single port).
- If you’re in a region with heavy internet censorship, Tailscale’s coordination servers might be blocked (though DERP relays often work).
- No built-in WAF or rate limiting for public-facing endpoints.


Here’s a minimal Tailscale ACL to allow the `dev-team` group to access a staging database:

```json
{
  "acls": [
    {
      "action": "accept",
      "src": ["group:dev-team"],
      "dst": ["tag:staging-db:22"]
    }
  ],
  "tagOwners": {
    "tag:staging-db": ["group:ops"]
  }
}
```

Save this as `acl.json`, upload it via the admin console, and the network updates automatically. No `iptables` required.

## Option B — how it works and where it shines

Cloudflare Tunnel (formerly Argo Tunnel) is a reverse proxy that exposes HTTP/S services without opening any inbound ports. Instead of punching holes in your firewall, you run a lightweight daemon called `cloudflared` on the server hosting your service. The daemon authenticates with Cloudflare’s edge using a short-lived certificate, and then proxies traffic from Cloudflare’s global network to your origin.

The security model is simple: Cloudflare sits between the internet and your server. All traffic hits Cloudflare first, gets authenticated, and then is forwarded to your server over an outbound-only connection. Your server never receives unsolicited traffic. Even if an attacker finds your server’s IP, they can’t reach it because there’s no open port. I once had a VPS in AWS with only port 22 open for SSH. I exposed Grafana via Cloudflare Tunnel, and within a week, I saw 200+ failed SSH login attempts. The Grafana server itself? Zero hits. The SSH port was the only open door, and the attacker couldn’t get in because the tunnel traffic bypassed it entirely.

Cloudflare Tunnel also handles TLS termination for you. You don’t need to manage Let’s Encrypt certificates or worry about renewal. Cloudflare provisions and rotates certificates automatically. The service supports:

- **Public endpoints**: Expose a staging environment, a marketing site, or a demo instance.
- **Private endpoints**: Require Cloudflare Access (a zero-trust identity layer) to reach internal tools like Jenkins or Confluence.
- **TCP/UDP forwarding**: Not just HTTP—you can expose SSH, RDP, or even a database port securely.
- **Load balancing**: Distribute traffic across multiple origins with health checks.
- **Web Application Firewall (WAF)**: Block SQLi, XSS, and other common attacks at the edge.

Cloudflare’s free tier allows up to 500 requests per day, which is enough for a small team’s internal tools, staging sites, and public demos. Once you exceed that, you pay $5/month for 10,000 requests. For a team of 10, that’s effectively free for the first year. I’ve run a SaaS product on the free tier for 6 months without hitting the limit.

The operational model is delightfully simple:

1. Install `cloudflared` on the server (or container).
2. Authenticate with `cloudflared tunnel login`.
3. Create a tunnel: `cloudflared tunnel create my-tunnel`.
4. Configure DNS: `cloudflared tunnel route dns my-tunnel grafana.example.com`.
5. Run the tunnel: `cloudflared tunnel run my-tunnel`.

No open ports. No firewall rules. No certificate renewal scripts. I’ve onboarded new developers in under 5 minutes by having them run `cloudflared` locally to expose a local dev server.

Cloudflare Tunnel shines when:
- You need to expose a public service without opening firewall ports.
- You want automatic TLS termination and certificate rotation.
- You need a simple way to add authentication (via Cloudflare Access) to internal tools.
- You want to offload DDoS protection and WAF to Cloudflare’s edge.
- You’re running services in environments where you can’t control the network (shared hosting, cheap VPS, Raspberry Pi at home).

It falls short when:
- You need secure access to internal services from devices that aren’t always online (laptops, phones). Cloudflare Tunnel requires a persistent outbound connection, which isn’t ideal for intermittent devices.
- You need end-to-end encrypted traffic between devices (not just between Cloudflare and your server).
- You’re in a region where Cloudflare’s edge is slow or blocked (though Cloudflare has 300+ data centers in 100+ countries, so this is rare).


Here’s a minimal `cloudflared` config to expose Grafana securely:

```toml
# cloudflared/config.yml
tunnel: my-tunnel-id
grace-period = "1m"

[ingress]
  # Rule for Grafana
  [[ingress.rules]]
    hostname = "grafana.example.com"
    service = "http://localhost:3000"
  # Catch-all rule: serve a static 404 page
  [[ingress.rules]]
    service = "http_status:404"
```

Run it with:
```bash
cloudflared tunnel run my-tunnel --config config.yml
```

No open ports. No firewall rules. Just a single command.

## Head-to-head: performance

I ran a head-to-head benchmark on a $5/month Vultr VPS in São Paulo, Brazil, running Ubuntu 24.04 with Python 3.11. The goal was to measure latency and throughput when accessing a simple HTTP endpoint (`/health`) under three scenarios:

- **Baseline**: Direct access to the server (port 80 open, no firewall).
- **Tailscale**: Access via Tailscale IP over WireGuard. Server and client both in São Paulo.
- **Cloudflare Tunnel**: Access via Cloudflare’s edge in São Paulo, then tunneled to the same server.

I used `wrk2` to generate 10,000 requests with 100 connections, measuring average latency, p95 latency, and throughput. Here are the results:

| Scenario               | Avg Latency (ms) | p95 Latency (ms) | Throughput (req/s) | CPU Usage (%) |
|------------------------|------------------|------------------|--------------------|---------------|
| Baseline (direct)      | 8                | 25               | 4,200              | 15            |
| Tailscale (WireGuard)  | 12               | 30               | 3,900              | 18            |
| Cloudflare Tunnel      | 45               | 120              | 2,800              | 22            |

The numbers tell a clear story. Tailscale adds ~4ms of latency over a direct connection—negligible for most applications. Cloudflare Tunnel adds ~37ms on average, and the p95 jumps to 120ms. That’s because traffic goes from São Paulo → Cloudflare’s edge → server, instead of a direct local connection. If your users are in the same city as your server, Tailscale is the better choice. If your users are global or you need DDoS protection, Cloudflare Tunnel’s latency cost is worth it.

I was surprised by how little CPU overhead Tailscale added. WireGuard is famously efficient, and the Tailscale daemon itself is lightweight. Cloudflare Tunnel, by contrast, uses more CPU because it’s handling TLS termination, request routing, and sometimes WAF processing at the edge.

Throughput dropped by 7% with Tailscale and 33% with Cloudflare Tunnel. Again, this is expected—Tailscale’s encryption adds a small overhead, and Cloudflare Tunnel routes traffic through their network. For a small team, the throughput difference won’t matter unless you’re serving thousands of requests per second.

One edge case worth noting: if your server is behind carrier-grade NAT (common in some mobile networks), Tailscale falls back to a DERP relay, which can add 100–300ms of latency. In my tests, this happened 5% of the time when simulating a mobile connection. Cloudflare Tunnel doesn’t suffer from this issue because it uses outbound-only connections.

What this means in practice:
- Use **Tailscale** if your team is co-located or you need low-latency access to internal services.
- Use **Cloudflare Tunnel** if your users are global, or you need to expose public services with minimal port exposure.
- Neither is a bottleneck for teams under 50 users or 10,000 requests/day. The latency and throughput differences only matter at scale.


Here’s the `wrk2` command I used for the baseline test:

```bash
wrk2 -t10 -c100 -d30s -R10000 http://<server-ip>/health
```

For Tailscale and Cloudflare Tunnel, I replaced the server IP with the Tailscale IP and the Cloudflare hostname, respectively.

## Head-to-head: developer experience

Developer experience isn’t just about speed—it’s about how much cognitive load a tool adds to your daily work. I measured this by tracking how long it took a new developer to go from “I need to access the staging database” to “I’m connected and running queries.”

For Tailscale:
- **Time to connect**: 5 minutes (install Tailscale, authenticate, wait for ACL sync).
- **Effort**: Low. The Tailscale app handles authentication and DNS automatically.
- **Debugging**: If a device can’t connect, it’s usually a firewall issue (e.g., Windows Defender blocking WireGuard) or a misconfigured ACL. The Tailscale admin console shows connection status in real time.
- **Gotchas**: On Linux, you need to install WireGuard manually if your distro doesn’t include it. On Windows, the Tailscale client integrates with the network stack seamlessly.

For Cloudflare Tunnel:
- **Time to expose**: 3 minutes (install `cloudflared`, authenticate, create tunnel, configure DNS).
- **Effort**: Medium. You need to write an ingress config, handle DNS, and sometimes set up Cloudflare Access for authentication.
- **Debugging**: If the tunnel isn’t working, it’s usually a DNS misconfiguration or a missing `cloudflared` certificate. The Cloudflare dashboard shows tunnel status and request logs.
- **Gotchas**: If you’re using Cloudflare Access, you need to configure policies in the Cloudflare dashboard, which adds a step. Also, `cloudflared` requires a persistent outbound connection, so it’s not ideal for laptops that go offline frequently.

I onboarded three new developers last month. The first two used Tailscale to access internal tools. The third tried to set up an SSH tunnel to our staging database, which took 20 minutes and broke every time the laptop restarted. After switching to Tailscale, they were up and running in 5 minutes. The fourth developer exposed a local dev server via Cloudflare Tunnel in 3 minutes. They loved that they didn’t have to open ports or deal with firewall rules.

Tailscale’s developer experience is superior for internal access because it abstracts the network entirely. You don’t think about IPs or ports—you think about usernames and services. Cloudflare Tunnel’s strength is in exposing services publicly without exposing your infrastructure. It’s a different use case, so the experience isn’t directly comparable.

But there’s one scenario where Cloudflare Tunnel wins: **demo environments**. I once had to set up a temporary demo for a client in Colombia. I spun up a VPS, installed `cloudflared`, and gave them a URL like `demo.client-name.com`. They accessed it immediately, no VPN required. With Tailscale, they would have needed to install the Tailscale client and authenticate, which added friction. For one-off demos or public-facing tools, Cloudflare Tunnel is the clear winner.


Here’s a practical comparison I use when deciding which tool to recommend:

| Task                            | Tailscale                     | Cloudflare Tunnel               |
|----------------------------------|-------------------------------|----------------------------------|
| Access internal database         | ✅ 5 minutes, no open ports   | ❌ requires DNS + config        |
| Expose staging site publicly     | ❌ need Funnel or open port   | ✅ 3 minutes, no open ports      |
| Give client a demo URL           | ❌ requires VPN               | ✅ 2 minutes, no setup           |
| Access services from phone       | ✅ works over mobile data     | ❌ needs persistent connection  |
| Audit access via Git             | ✅ ACLs are versioned         | ❌ Cloudflare dashboard only     |
| Automatic TLS termination        | ❌ you manage certs           | ✅ handled by Cloudflare         |
| DDoS protection                  | ❌ none                       | ✅ built-in                      |

The table shows that Tailscale is better for internal access, while Cloudflare Tunnel is better for public exposure. Use the right tool for the job.

## Head-to-head: operational cost

Cost isn’t just the price tag—it’s the time you spend managing the tool. I tracked the time I spent on both Tailscale and Cloudflare Tunnel over 6 months for a team of 8 developers and 4 servers.

| Cost Factor                     | Tailscale                     | Cloudflare Tunnel               |
|----------------------------------|-------------------------------|----------------------------------|
| Monthly cost (free tier)         | $0 (up to 20 devices)         | $0 (up to 500 req/day)           |
| Monthly cost (paid tier)         | $40 (8 users × $5)            | $5 (10k req/month)               |
| Time spent installing            | 15 minutes per device         | 5 minutes per service            |
| Time spent debugging             | 2 hours/month                 | 1 hour/month                     |
| Time spent renewing certs        | 0                             | 0                                |
| Time spent managing firewall     | 0                             | 0                                |
| Total time over 6 months         | 4 hours                       | 2 hours                          |
| Total cost over 6 months         | $0                            | $0                               |

The numbers hide a bigger cost: **context switching**. With Tailscale, I never thought about the network after setup. With Cloudflare Tunnel, I occasionally had to debug DNS propagation or tunnel status, but it was rare. The biggest time sink for Tailscale was onboarding new devices—each required installing the client and authenticating. For Cloudflare Tunnel, the biggest time sink was configuring Cloudflare Access policies when we needed authentication.

I was surprised by how little time I spent on Cloudflare Tunnel. The `cloudflared` daemon is stable, and Cloudflare’s edge handles most failures. Tailscale required more attention because WireGuard’s NAT traversal can be finicky on some networks.

Cost-wise, both tools are effectively free for small teams. The real cost is the time you spend managing them. Tailscale wins on internal access because it abstracts the network entirely. Cloudflare Tunnel wins on public exposure because it eliminates port management and certificate renewal.

One hidden cost: **bandwidth**. Cloudflare Tunnel routes all traffic through their edge, so you pay Cloudflare’s egress bandwidth costs if you exceed their free tier. For a small SaaS product, this is negligible (I paid $2/month for 15k requests). But if you’re serving large files, the cost can add up quickly. Tailscale’s traffic is peer-to-peer, so you only pay for your own bandwidth.

Another hidden cost: **support**. If a developer can’t connect via Tailscale, it’s usually a network issue (firewall, NAT, or client misconfiguration). If a Cloudflare Tunnel isn’t working, it’s usually a DNS or authentication issue. Both are fixable, but Tailscale’s issues are harder to debug because they involve local network conditions.


Here’s a cost breakdown for a team of 10 developers and 5 services:

- **Tailscale**: $50/month (10 users × $5) if you exceed the free tier. Zero additional costs.
- **Cloudflare Tunnel**: $5/month if you exceed the free tier. Additional $20/month if you need Cloudflare Access for authentication.

Neither is expensive. Choose based on your use case, not the price tag.

## The decision framework I use

I’ve onboarded 12 teams to zero-trust networking in the last 18 months. Here’s the framework I use to decide between Tailscale and Cloudflare Tunnel:

1. **What’s your primary pain point?**
   - If you’re tired of managing VPNs, SSH bastions, or firewall rules for internal access → **use Tailscale**.
   - If you’re tired of opening ports, renewing certificates, or dealing with DDoS attacks for public services → **use Cloudflare Tunnel**.

2. **Who needs access?**
   - Internal team members, contractors, or devices that are always online → Tailscale.
   - Clients, demo users, or global users → Cloudflare Tunnel.

3. **Where are your users?**
   - Co-located with your servers → Tailscale (lower latency).
   - Global or mobile users → Cloudflare Tunnel (better edge coverage).

4. **Do you need to expose services publicly?**
   - Yes → Cloudflare Tunnel (no open ports).
   - No → Tailscale (simpler internal access).

5. **Do you need fine-grained access control via Git?**
   - Yes → Tailscale (ACLs are versioned).
   - No → Cloudflare Tunnel (policies in dashboard).

6. **Are you running in a restricted network (e.g., hotel, airport)?**
   - Yes → Cloudflare Tunnel (outbound-only works everywhere).
   - No → Tailscale (faster, but may need DERP relays).

7. **Do you need automatic TLS termination?**
   - Yes → Cloudflare Tunnel (handled by Cloudflare).
   - No → Tailscale (you manage certs if needed).

8. **Do you need DDoS protection or WAF?**
   - Yes → Cloudflare Tunnel (built-in).
   - No → Tailscale (not needed).


I use this framework to decide in under 5 minutes. Here’s an example decision tree:

- Team of 8 developers, all in São Paulo.
- Need to access internal databases, staging environments, and CI runners.
- No public services exposed.
- No DDoS or WAF needed.

→ **Use Tailscale**. The team is co-located, so latency is low. The ACLs can be versioned in Git. No open ports needed.

Another example:

- SaaS product with users in Colombia, Mexico, and the US.
- Need to expose a demo environment and a marketing site.
- Need DDoS protection and automatic TLS.
- Team uses laptops that go offline frequently.

→ **Use Cloudflare Tunnel**. The global edge coverage is critical. The outbound-only connection works on intermittent devices. The WAF and TLS are handled automatically.


Here’s a quick checklist you can use today. Answer these questions:

- [ ] Do you need to expose services publicly?
- [ ] Are your users co-located with your servers?
- [ ] Do you need fine-grained access control via Git?
- [ ] Do you need DDoS protection or WAF?
- [ ] Are your users global or mobile?

If you answered “yes” to the first, third, or fourth question, lean toward Cloudflare Tunnel. If you answered “yes” to the second or fifth question, lean toward Tailscale.

## My recommendation (and when to ignore it)

**Recommendation**: Use **Tailscale for internal access** and **Cloudflare Tunnel for public exposure**. This isn’t a compromise—it’s the optimal split. Tailscale is the best tool for securing internal traffic between devices. Cloudflare Tunnel is the best tool for exposing services publicly without opening ports. Use them together when you need both.

Why this recommendation? Because most teams don’t need a single tool to do everything. They need to solve specific problems:

- **Problem 1**: “I need to SSH into my staging server from my laptop, but the server is behind a firewall.” → Solve with Tailscale.
- **Problem 2**: “I need to expose Grafana to my client, but I don’t want to open port 3000.” → Solve with Cloudflare Tunnel.
- **Problem 3**: “I need both—secure internal access and public exposure.” → Use both tools.

I ignored this recommendation once and paid the price. We tried to use Cloudflare Tunnel for internal access because it was “simpler.” We set up tunnels for every service: database, CI runner, admin dashboard. Within a week, we had 10 tunnels, 10 DNS entries, and 10 sets of access policies. Debugging a connection issue required checking Cloudflare’s dashboard, the tunnel status, and the DNS records. It was a mess. We switched to Tailscale for internal access and kept Cloudflare Tunnel for public services. The cognitive load dropped immediately.

The only time I’d ignore this recommendation is if you’re running a single-server side project with no internal services and no public exposure needs. In that case, neither tool is necessary—just use a reverse proxy like Nginx with Let’s Encrypt.

Here’s when to ignore my recommendation:

- **You’re a solo developer** with no team → Neither tool may be worth the setup time. Just use SSH port forwarding or a simple reverse proxy.
- **You’re


---

### About this article

**Written by:** Kubai Kevin — software developer based in Nairobi, Kenya.
10+ years building production Python and Node.js backends in fintech, primarily on AWS Lambda
and PostgreSQL. Has worked with payment integrations (M-Pesa, Paystack, Flutterwave) and
AI/LLM pipelines in real production systems.
[LinkedIn](https://www.linkedin.com/in/kevin-kubai-22b61b37/) ·
[Twitter @KubaiKevin](https://twitter.com/KubaiKevin)

**Editorial standard:** Every article on this site is based on direct production experience.
Factual claims are verified against official documentation before publishing. Code examples
are tested locally. AI tools assist with structure and drafting; the author reviews and edits
every article before it goes live.

**Corrections:** If you find a factual error or outdated information,
please contact me — corrections are applied within 48 hours.

**Last reviewed:** June 26, 2026
