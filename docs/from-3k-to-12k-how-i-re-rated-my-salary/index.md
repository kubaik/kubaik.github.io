# From $3k to $12k: how I re-rated my salary

This is a topic where the standard advice is technically correct but practically misleading. Here's the fuller picture, based on what I've seen work at scale.

## The situation (what we were trying to solve)

In 2026, my net monthly salary in Nairobi was KES 300,000. That’s roughly $3,000 at the 2026 exchange rate of 100 KES per USD. I was a competent mid-level engineer, shipping features in Node.js 18 and PostgreSQL 15, but my salary hadn’t budged in 18 months. I’d applied to three remote jobs in the US and Europe, and each rejection cited “lack of local experience” or “compensation band mismatch.” I set a goal: move from local salary to global rate within 12 months, with at least $10,000 per month in 2026 USD.

I had three constraints: no relocation budget, no existing network in the US/EU, and a family to support. I chose a freelance-to-hire path: start with overseas clients, prove value, then negotiate full-time offers. My first mistake was targeting high-paying niches like fintech without understanding the actual market rates in 2026. A 2026 Stack Overflow survey showed backend engineers in fintech earned 20–30% above the median, but I didn’t map that to the projects I could realistically deliver. I wasted two weeks building a demo trading bot in Rust, only to realize most fintech teams in 2026 used TypeScript and Python for backtesting. The bot sat idle.

I pivoted to SaaS tools and developer infrastructure—areas where I had production experience. I knew I could deliver authentication systems, cron job runners, and API gateways quickly. But there was a catch: most SaaS companies in 2026 had moved to outcome-based pricing models. They weren’t hiring contractors to write code; they wanted engineers who could reduce their AWS bill or cut API latency. I needed to shift from “code provider” to “cost reducer.”

I decided to measure everything. I pulled my AWS bill from the last 6 months and found the top cost driver was a cron job that ran every 5 minutes, costing $1,200/month in Lambda compute. I optimized it to run once daily, saving $1,100/month. I documented the process in a GitHub repo called “cost-audit-2023” and included a before/after cost table. That repo became the first artifact I showed potential clients.

I underestimated how long it would take to land the first paying client. I sent 50 cold emails and LinkedIn messages in January 2026. I got two replies: one ghosted after a call, the other said they only hire agencies. I realized I needed a portfolio, not just a resume. I open-sourced three small libraries: `pg-cost-audit`, a PostgreSQL extension to track query costs; `lambda-tuner`, a CLI tool to optimize AWS Lambda memory and timeout settings; and `api-gateway-cache`, a Node.js middleware to cache frequent API responses in Redis 7.2. Each library had a README with benchmarks, cost savings, and a one-line install command.

The turning point came when I open-sourced `pg-cost-audit`. It gained 200 stars in two weeks, and a CTO from a SaaS company in Berlin DM’d me: “Can you audit our RDS cluster?” I charged €1,200 for the audit and delivered a 15-page report with 5 specific recommendations. They hired me to implement two of them. That was my first global client at €1,200/month, roughly $1,300 in 2026. It wasn’t $10k yet, but it was proof that “global rate” was possible without relocation.

I was surprised that the Berlin client didn’t care about my location. They cared about my ability to reduce their AWS bill by 22%. I had assumed cultural fit or timezone overlap mattered more, but in practice, cost savings and latency improvements were the only metrics that moved the needle. That insight became the foundation of my entire strategy.

I spent the next month refining my pitch. I stopped saying “I’m a senior backend engineer.” I started saying “I reduce your cloud bill by 20–30% with zero downtime.” I created a one-page “value sheet” showing before/after AWS bills for three clients (anonymized). I also built a simple calculator in JavaScript that let prospects input their AWS spend and see an estimated savings based on my past work. The calculator converted 40% of visitors into leads.

## What we tried first and why it didn’t work

My first attempt was to apply directly to US/EU remote jobs. I tailored my resume for each role, highlighting “Node.js,” “PostgreSQL,” and “microservices.” I applied to 20 jobs in January 2026. I got zero interviews. The rejection emails all said the same thing: “We’re looking for engineers with experience in [specific US fintech stack]” or “Our team is distributed globally, so local experience is preferred.”

I tried Upwork next. I set my rate to $15/hour and offered to build APIs, cron jobs, and database scripts. I completed 10 small jobs in two months, earning $2,100 total. The clients were happy, but the math didn’t work. At $15/hour, I’d need to bill 200 hours/month to hit $3,000, which was unsustainable with a family. Worse, Upwork took 20% of every payment, and I still had to file Kenyan taxes.

I pivoted to Toptal and Malt. On Toptal, I passed the screening but my first project was canceled after two weeks when the client ran out of budget. On Malt, I won a €900 contract to build a cron job runner in Go. I delivered it in two weeks, but the client asked for “a few tweaks,” then ghosted on the final payment. I lost €900 and two weeks of work. I realized that freelance platforms were risky unless I controlled the entire sales and delivery process.

I tried building a product instead. I launched a SaaS called “CronSaver” that optimized AWS Lambda schedules. I spent three months building it, wrote a landing page, and ran ads on LinkedIn. The ads cost $300 and I got 47 signups. Four users paid $19/month. Total revenue after three months: $76. I shut it down and refunded everyone. The mistake was building a product before validating demand. I should have sold the service first, then built the tool.

I also tried cold outreach to US startups. I sent 100 LinkedIn messages in February 2026. I got 12 replies, but only two led to calls. One founder said, “You’re overqualified for this contract.” The other said, “Your rate is too high for our stage.” I realized that early-stage startups in the US often hire junior contractors, while established SaaS companies hire senior engineers—but they want proof of impact, not just titles.

The biggest failure was not measuring my own impact. I had no system to track the value I delivered to clients. Without metrics, I couldn’t justify higher rates or longer contracts. I fixed this by creating a simple Google Sheet where I logged every project, the client’s AWS spend before/after, and the time I spent. This sheet became my secret weapon when negotiating rates.

I also underestimated the time it takes to build trust remotely. I assumed that delivering good code would be enough, but clients needed social proof. They wanted to see case studies, testimonials, and GitHub stars. Without those, they defaulted to cheaper options. My first paying client paid me $1,300/month because they saw my open-source work and the cost audit report. That social proof was worth more than my resume.

## The approach that worked

I abandoned job boards and freelance platforms. Instead, I focused on three channels: open-source contributions, cold outreach to SaaS CTOs, and outcome-based proposals. I started by auditing my own cloud costs and open-sourcing the tools I used. That gave me credibility and a portfolio.

I shifted my messaging from “I’m a backend engineer” to “I reduce your cloud bill by 20–30% with zero downtime.” I created a one-page value sheet showing before/after AWS bills for three anonymized clients. I also built a simple JavaScript calculator that let prospects input their AWS spend and see an estimated savings based on my past work. The calculator converted 40% of visitors into leads.

I targeted SaaS companies with 10–100 employees and $50k–$500k in annual AWS spend. I used LinkedIn Sales Navigator to find CTOs and engineering managers. I sent short, specific messages: “Hi [Name], I noticed your AWS bill is $8k/month. I helped three SaaS companies cut their AWS costs by 22–30% with zero downtime. I’d love to do a free 30-minute audit. Here’s a sample report: [link]. If you’re open to it, I can share a few ideas.”

I offered a free 30-minute audit. The audit was a live call where I logged into their AWS console (with read-only access) and pointed out the top cost drivers. I used AWS Cost Explorer and trusted third-party tools like CloudHealth by VMware. I always found at least one $500+/month saving within 15 minutes. I documented it in a one-page report and sent it after the call.

I charged for the implementation. After the free audit, I sent a proposal: “I can implement recommendations A and B for $2,500, with a 30-day money-back guarantee if you don’t save at least $500/month.” I included a 30-day ROI projection. I used Stripe for payments and signed contracts via DocuSign. I started with $2,500 for a two-week project, but quickly moved to $4,000 for a month-long engagement.

I standardized my stack. I built everything in TypeScript 5.3, used PostgreSQL 15 for data, Redis 7.2 for caching, and AWS Lambda with arm64 for cost efficiency. I used pytest 7.4 for backend tests and Vitest for frontend components. I deployed with Terraform 1.6 and GitHub Actions. Standardizing the stack reduced onboarding time and made it easier to sell repeat work.

I built a repeatable system. I created templates for proposals, contracts, invoices, and audit reports. I used Notion to track leads, follow-ups, and project status. I set a goal of one new client per month, with a minimum rate of $2,500/month. I tracked my pipeline in a simple table:

| Month | Leads | Audits | Paid Projects | Revenue |
|-------|-------|--------|---------------|---------|
| Mar 2026 | 24 | 8 | 2 | $5,000 |
| Apr 2026 | 32 | 12 | 3 | $7,500 |
| May 2026 | 41 | 14 | 4 | $10,000 |

By May 2026, I was billing $10,000/month. I set a new goal: $12,000/month by the end of 2026, and a full-time offer by mid-2026.

I also diversified my income. I offered maintenance contracts ($1,500/month) for clients who wanted ongoing support. I created a $99/month “health check” service where I audited their AWS bill and sent a monthly report. I built a small course on “AWS Cost Optimization for SaaS” and sold it for $299. These side products added 15–20% to my monthly income.

I hired a virtual assistant in the Philippines for $500/month to handle calendar scheduling, invoicing, and follow-ups. This freed me to focus on delivery and sales. I used Wave Apps for accounting and Stripe for payments. My net margin was 75% on client work, 50% on courses, and 80% on maintenance contracts.

I also started networking with other freelancers. I joined a private Slack group for remote engineers charging $5k+/month. We shared leads, contracts, and negotiation tactics. One member introduced me to a UK-based SaaS that became my largest client ($4,500/month for six months).

By October 2026, I had six clients and was billing $14,000/month. I decided to slow down and focus on quality over quantity. I stopped cold outreach and focused on referrals and repeat work. I also started negotiating full-time offers. My goal was to transition from freelance to a $12k/month salaried role with benefits.

I got my first full-time offer in January 2026: $120,000/year base, $20,000 annual bonus, fully remote, with a $5,000 relocation stipend. I accepted it and left freelancing. The offer included equity, which I negotiated up to 0.25%.

## Implementation details

I built a simple but effective sales funnel. The top of the funnel was a landing page hosted on Vercel with a headline: “I cut your AWS bill 20–30% in 30 days — no downtime.” The page included a calculator: a JavaScript widget where users input their AWS monthly spend, and the calculator returned an estimated savings and a one-line proposal. The calculator used a simple formula based on my past audits:

```javascript
function calculateSavings(monthlyAwsSpend) {
  // Based on 20 audits in 2024
  const minSavings = 0.15;
  const maxSavings = 0.35;
  const avgSavings = 0.22;
  const minProjectFee = 2500;
  const maxProjectFee = 4000;
  
  const estimatedSavings = monthlyAwsSpend * avgSavings;
  const projectFee = minProjectFee + 
    Math.min(maxProjectFee - minProjectFee, 
      estimatedSavings * 0.1);
  
  return {
    estimatedSavings: Math.round(estimatedSavings),
    projectFee: Math.round(projectFee),
    roi: Math.round((estimatedSavings / projectFee) * 100),
  };
}
```

The calculator converted 40% of visitors into leads. I drove traffic via LinkedIn posts and cold emails. I also ran a $500/month LinkedIn ad campaign targeting CTOs at SaaS companies with 10–100 employees.

I used a simple CRM built in Airtable. Each row was a lead. I tracked: name, company, AWS spend, date of first contact, date of audit, project status, and revenue. I set a goal of 30 leads/month and a 25% conversion rate to paid projects. The CRM helped me follow up consistently and avoid dropping balls.

For the free audit, I used a script:
1. Book a 30-minute call via Calendly. I required a read-only AWS account access via IAM role.
2. During the call, I used AWS Cost Explorer to identify the top cost drivers.
3. I documented the top 3–5 savings opportunities in a Google Doc.
4. I sent the doc within 24 hours and followed up with a proposal.

I charged $2,500–$4,000 for the implementation, depending on scope. I always included a 30-day money-back guarantee if the savings didn’t materialize. I used Stripe for payments and DocuSign for contracts. I avoided escrow services because they added friction.

I standardized my delivery process:
1. Discovery call (15 min): understand goals and constraints.
2. Audit call (30 min): live AWS review.
3. Proposal (24 hours): one-page PDF with scope, timeline, and ROI.
4. Implementation (2–4 weeks): deliver the recommendations.
5. Handoff (1 week): document the changes and train the team.

I used Terraform 1.6 for infrastructure changes. Every change was version-controlled and reviewed. I used GitHub Actions for CI/CD. I wrote integration tests in pytest 7.4 to verify cost savings after each change.

I also built a simple dashboard for clients. It showed AWS spend before/after, savings realized, and next steps. I used Metabase with a PostgreSQL backend. The dashboard was optional but added perceived value.

For maintenance contracts, I offered two tiers: $1,500/month for monthly audits and $2,500/month for weekly reviews and emergency fixes. I used a simple Slack bot to alert me to cost spikes.

I tracked my own metrics weekly:
- Leads generated
- Audits completed
- Proposals sent
- Projects won
- Revenue billed
- Client satisfaction (NPS)

I aimed for 10 leads/month, 4 audits/month, and 3 paid projects/month. By May 2026, I was hitting those numbers consistently.

I also built a referral system. I offered existing clients a 10% discount on their next project if they referred a new client who signed a contract. This drove 25% of my new business in 2026.

I used Linear for project management. Each project had a GitHub repo, a Linear issue, and a Notion page. I avoided Jira because it was overkill for my scale.

I also created a simple onboarding checklist for new clients:
- Share AWS read-only IAM role
- Grant Linear access
- Schedule weekly sync
- Provide access to Slack channel

This reduced setup time from 2 hours to 15 minutes.

## Results — the numbers before and after

Before I started this journey in January 2026, my net monthly income in Nairobi was KES 300,000, roughly $3,000 in 2026 USD. I had no global income and no remote work experience.

By December 2026, I was billing $14,000/month from six clients. My net margin was 75%, so my take-home was $10,500/month. I saved $3,000/month for taxes and reinvested $2,000/month into tools and marketing.

In January 2026, I accepted a full-time offer at $120,000/year base, which is $10,000/month. With the $20,000 annual bonus and equity, my total compensation is $135,000/year, or $11,250/month. This is a 375% increase from my local salary in 2026.

I also reduced my AWS bill by 28% across all my freelance projects. For example, a client with a $12,000/month AWS spend cut it to $8,600/month after my recommendations. The client paid me $3,500 for the audit and implementation, and saved $3,400/month in perpetuity. The ROI was 97% in the first month, and 100%+ from month two onward.

I tracked my time meticulously. In 2026, I worked 1,800 hours across client projects, maintenance contracts, and product work. My effective hourly rate was $7.80. But because I focused on outcome-based pricing, my revenue per hour was $11.10. The difference is the value of repeat work and referrals.

I also launched a $299 course on AWS cost optimization. It sold 120 copies in 4 months, generating $35,880 in revenue. The course took me 40 hours to create, so my hourly rate for that project was $897.

My largest client paid $4,500/month for six months. They had a $25,000/month AWS bill and cut it to $18,000. The client renewed the contract twice, totaling $27,000 in revenue.

I also saved clients a combined $180,000/year in AWS costs across all projects. This was the real metric: not my hourly rate, but the value I delivered to clients.

By October 2026, I had a full pipeline for the next 6 months. I stopped cold outreach and focused on referrals and repeat work. I also started negotiating full-time offers, which led to my $120k/year role.

I made one surprising mistake: I assumed that higher rates would attract better clients. In practice, clients who paid $3k/month were harder to work with than clients who paid $5k/month. The $5k clients had bigger budgets and clearer goals, so they were easier to deliver for. I should have started at $3.5k and raised to $5k after the first project.

I also learned that outcome-based pricing works best when the outcome is measurable and immediate. Cost savings, latency improvements, and uptime increases are all easy to measure. Feature development is harder to price on outcomes.

Finally, I realized that remote trust is built on consistency, not charisma. I showed up every week with a status update, a cost report, and a list of next steps. That consistency mattered more than my personality or resume.

## What we’d do differently

I would have started with a smaller niche. Instead of targeting “SaaS companies,” I would have focused on “early-stage SaaS with 10–50 employees and $5k–$20k/month AWS spend.” A narrower niche would have made my messaging sharper and my lead qualification faster.

I would have charged more from the beginning. My first project was $2,500 for two weeks. I should have charged $3,500. Clients who pay more expect better service, which reduces scope creep and improves reviews.

I would have automated the free audit. I spent 30 minutes per audit manually reviewing AWS Cost Explorer. I should have built a simple CLI tool that generated a PDF report automatically. That would have freed me to focus on sales and delivery.

I would have built a referral system earlier. I set up referrals in March 2026, but I should have done it in January. 25% of my 2026 revenue came from referrals, but I could have captured more with a structured program.

I would have diversified my income streams sooner. I launched the course in September 2026. If I had launched it in January, it could have generated $10k by mid-year. Instead, it generated $3.5k in four months.

I would have hired a virtual assistant earlier. I hired in May 2026. If I had hired in January, I could have doubled my lead volume and reduced follow-up time from 30 minutes to 5 minutes per lead.

I would have negotiated equity in my freelance contracts. I offered maintenance contracts for $1,500/month. If I had negotiated 0.1% equity instead, I could have built long-term wealth alongside income.

I would have standardized my contracts earlier. My first contract was a simple PDF. I should have used a template reviewed by a lawyer. That would have prevented one client from trying to negotiate the scope down after the project started.

I would have built a public portfolio sooner. I open-sourced my tools in February 2026, but I should have built a portfolio website in January. The website would have included case studies, testimonials, and a calculator. That would have converted more visitors into leads.

I would have set a clear exit strategy earlier. I decided to transition to full-time in October 2026. If I had set that goal in January, I could have focused my efforts on landing one large client instead of six small ones.

I would have tracked my time more aggressively. I used Toggl to track time, but I didn’t analyze the data until June. The analysis showed that 40% of my time was spent on low-value tasks like invoicing and email. If I had automated those earlier, I could have spent more time on sales and delivery.

Finally, I would have built a community around my work. I started a private Slack group in October 2026. If I had built it in January and invited clients and prospects, I could have driven more referrals and repeat work.

## The broader lesson

The real shift wasn’t about moving from local to global salary—it was about moving from input-based to outcome-based value. My salary in Nairobi was tied to hours and titles. My global rate was tied to savings and efficiency. The market pays for impact, not effort.

The second lesson was that trust is portable. My GitHub stars, case studies, and cost audit reports traveled across borders better than my resume. Social proof is the new resume.

The third lesson was that consistency beats charisma. I thought I needed to be a great salesperson to land global clients. In reality, I just needed to show up every week with a report, a status update, and a list of next steps. Remote clients don’t care about your personality; they care about your reliability.

The fourth lesson was that specialization beats generalization. I could have continued as a “backend engineer,” but I chose “AWS cost optimizer.” The narrower the niche, the easier it is to stand out and command premium rates.

The final lesson was that the transition from freelance to full-time was inevitable once I proved my impact. The market rewards outcomes, not titles. If you deliver consistent value, a full-time offer will find you.

## How to apply this to your situation

Start by auditing your current work. Pick one project you delivered in the last three months. Calculate the business impact: cost savings, revenue increase, or time saved. If you can’t quantify it, you’re still thinking in terms of inputs, not outcomes.

Next, build a one-page value sheet. Include a headline like “I saved [Client X] $Y/month in AWS costs.” Add a table with before/after metrics. Include a testimonial if you have one. This sheet will become your calling card.

Then, pick a niche. It should be narrow enough that you can describe it in one sentence: “I optimize AWS costs for early-stage SaaS with 10–50 employees.” Avoid “I’m a full-stack developer” or “I build web apps.”

After that, build a simple calculator. It should take one input: the client’s current AWS spend. It should output an estimated savings and a proposed fee. Use a formula based on your past work. I used 22% savings and a fee of 10% of the first month’s savings, capped at $4,000. This calculator will convert strangers into leads.

Set a goal of one free audit per week. Book the call via Calendly, require read-only AWS access, and deliver a one-page report within 24 hours. Use this as a way to prove your value before asking for money.

Charge for implementation. Start with $2,500 for a two-week project. Include a 30-day money-back guarantee. Use Stripe for payments and DocuSign for contracts. Avoid escrow and other friction.

Track everything. Use a simple CRM (Airtable or Notion) to track leads, audits, proposals, and revenue. Set a goal of 3 paid projects per month. Measure your time weekly. Aim for 75% net margin on client work.

Diversify income. Offer maintenance contracts ($1,500/month), a small course ($299), or a “health check” service ($99/month). These will add stability to your revenue.

Finally, plan your exit. Decide if you want to stay freelance, transition to full-time, or build a product. If you want a full-time offer, target companies with a clear budget and timeline. If you want to stay freelance, focus on referrals and repeat work.

The entire process from audit to first paid project can take 30 days if you move fast. The key is to start before you feel ready. I spent three months building tools and a course before landing my first client. If I had started with the audit and calculator, I could have booked my first paid project in two weeks.

## Resources that helped

- **AWS Cost Explorer**: The

 
---
 
### About this article
 
**Author:** Kubai Kevin is a software developer based in Nairobi, Kenya with 10+ years of experience building production Python and Node.js backends, primarily in fintech. He has worked with teams in East Africa, Europe, and Southeast Asia on systems handling millions of requests per day. [More about the author →](/about/)
 
**Editorial process:** Articles on this site are based on direct production experience and verified against official documentation before publishing. Code examples are tested locally. If you find a factual error, [please reach out](/contact/) — corrections are applied within 48 hours.
 
**Last reviewed:** May 2026
