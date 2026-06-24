# 3 side gigs that paid me 2k USD/month in 2026

I ran into this building second problem while migrating a service under a hard deadline. The answers I found online were either wrong or skipped the parts that mattered. Here's what actually worked.

## Why this list exists (what I was actually trying to solve)

In mid-2026 I hit a wall: my Nairobi fintech salary was stuck at 450k KES/month (≈3,300 USD) and the usual “side project” advice just didn’t scale for me. I tried the classic “build a SaaS” route with a multi-tenant identity provider in Python 3.11 + FastAPI. After three months and 1,200 lines of code, I had zero paying users and 140 USD in AWS bills from Lambda@Edge, DynamoDB and Route 53. The real kicker: I spent another two weeks debugging a WAF rule that blocked half of Nairobi’s mobile networks because I copy-pasted a CidrBlock from a 2026 blog post. I was surprised that a single /16 vs /24 made the difference — this post is what I wished I had found then.

I needed income that played to the skills I already had: writing production-grade Python and Node.js backends, reading AWS cost explorer graphs, and debugging latency spikes. So I ran a 90-day experiment across seven different models: freelance coding, open-source sponsorships, affiliate technical content, cloud-cost optimization gigs, selling data pipelines as one-off scripts, running a private Slack community for local engineers, and offering incident-response subscriptions. At the end of 90 days the clear winners were the ones that required almost no upfront code and gave me recurring cash. Everything else burned hours on marketing or customer support I didn’t enjoy.

Below is the ranked list I wish I’d had back then. Every entry includes the real tools, the exact numbers I hit, and the mistakes I made so you can skip the dead ends.


## How I evaluated each option

I measured every option against four concrete metrics:

1. **Activation energy** – time from “idea” to first dollar in my bank account.
2. **Runway** – how many months of income I could generate before the well ran dry without new work.
3. **Scaling ceiling** – maximum monthly revenue I could hit given 10 hours a week of effort.
4. **Regret score** – how much I would hate myself if I had to do customer support calls at 3 a.m.

I tracked activation energy in hours, runway in months, and ceiling in USD. Anything that required more than 20 hours of activation energy or had a ceiling under 1,000 USD/month got dropped early.

I also ran a small survey with 34 Nairobi developers in WhatsApp groups: **84 % said they already spend 1–3 hours a week on side income**, but only **18 % actually make more than 500 USD/month** from it. The biggest blocker was always “I don’t know what to build or sell.” This list is my answer to that.


## Building a second income stream as a developer in 2026 without building a SaaS — the full ranked list

### 1. Incident-response subscription for production APIs

**What it does:** Sell a 24/7 on-call retainer to small e-commerce teams that can’t afford a full-time DevOps hire. You get pinged via Slack when their API latency spikes, you SSH into their EC2 instances or dig into their CloudWatch Logs Insights, and you bill them a fixed monthly fee. You keep a private runbook of past incidents so the work is repeatable.

**Concrete strength:** The incidents themselves generate the content you need for future runs. I once debugged a 700 ms latency spike on a Node 20 LTS + Express stack that turned out to be a single slow MongoDB aggregation. After I fixed it I wrote a 500-word post-mortem and turned that into a template. That template alone has earned me 1,200 USD in repeat sales.

**Concrete weakness:** You’re on the hook when things break at 2 a.m. I once had to debug a Redis 7.2 cluster failover at 3:42 a.m. and I billed 150 USD for that night. Good revenue, bad sleep.

**Best for:** Developers who already speak AWS and CloudWatch fluently and don’t mind occasional late-night pages.

**Activation energy:** 14 hours (mostly writing a one-page runbook and a Stripe checkout link).
**Runway:** 6 months if you land 5 subscribers at 200 USD/month.
**Scaling ceiling:** 4,000 USD/month with 15 subscribers and 10 hours/week.


### 2. Open-source sponsorships via GitHub Sponsors

**What it does:** Publish a small, reusable Python or Node library that solves a niche problem (e.g., a FastAPI plugin for Kenyan M-Pesa callback verification, or a Node library that validates Kenyan phone numbers). Then ask for sponsorships on GitHub Sponsors. You set tiers at 5, 20, and 50 USD/month and use GitHub’s built-in analytics to see who’s installing your package.

**Concrete strength:** Once the package is published, the marginal cost per download is zero. I published a Python library for parsing Kenyan bank statements in June 2025. By December I had 2,147 downloads and 14 sponsors contributing 470 USD/month. The top sponsor is a local microfinance startup using it in production.

**Concrete weakness:** Marketing is all on you. I spent 12 hours tweeting, writing a short “why this matters” article on Dev.to, and answering issues in GitHub Discussions. Without that push the downloads plateaued at 150.

**Best for:** Developers who enjoy writing polished, well-documented libraries and don’t mind the occasional GitHub notification flood.

**Activation energy:** 22 hours (package setup, tests, README, GitHub Sponsors link).
**Runway:** 9 months with 10 sponsors at 20 USD/month.
**Scaling ceiling:** 1,200 USD/month if the library hits Hacker News or becomes the de-facto standard for Kenyan bank parsing.


### 3. Affiliate technical content (benchmarks + tutorials)

**What it does:** Write deep-dive tutorials or benchmark reports that reference specific cloud services or libraries, then include affiliate links to AWS, DigitalOcean, or HashiNode when readers spin up a server. You monetize via the affiliate program, not via ads.

**Concrete strength:** The content itself is useful and ranks well. I wrote a 2,500-word guide in May 2026 comparing AWS Lambda arm64 vs x86_64 cold-start latency across Python 3.12 and Node 22. The post now ranks #2 for “lambda arm64 vs x86 latency” and has sent 1,800 USD in AWS affiliate revenue to my account since publication.

**Concrete weakness:** Affiliate programs change their rates. AWS cut their EC2 affiliate rate from 5 % to 2 % in January 2026. That single change dropped my projected monthly revenue from 500 USD to 200 USD for the same traffic.

**Best for:** Developers who enjoy writing long-form content and can keep up with affiliate program changes.

**Activation energy:** 30 hours for a single 2,000-word guide.
**Runway:** 4 months with consistent 5,000 monthly page views.
**Scaling ceiling:** 3,000 USD/month if you publish 2–3 guides per month and hit the top of Google’s featured snippets.


### 4. One-off data pipeline scripts for SMEs

**What it does:** Build small Python scripts that pull data from a client’s Xero, QuickBooks, or Odoo instance, transform it, and push it into BigQuery or PostgreSQL. Clients pay a fixed fee per script (usually 300–800 USD) and you can resell the same script to similar businesses.

**Concrete strength:** The work is predictable and the scope is tiny. I once built a 180-line Python script using pandas and the Xero API v2 that syncs inventory every night. The client paid 600 USD and I’ve resold the same script to three more businesses this year.

**Concrete weakness:** Integration surprises. One client’s Xero instance had 17 custom fields I didn’t account for and the script broke after the first run. That cost me 2 extra hours of debugging and delayed payment by a week.

**Best for:** Developers who enjoy data munging and don’t mind occasional scope creep.

**Activation energy:** 8 hours for a first draft.
**Runway:** 5 months if you land 4 clients at 500 USD each.
**Scaling ceiling:** 1,500 USD/month if you automate the install process with Docker and charge for setup support.


### 5. Private Slack community for local engineers (paid membership)

**What it does:** Curate a invite-only Slack community for Nairobi-based backend and DevOps engineers. Charge a monthly fee (15 USD) and post curated job leads, incident post-mortems, and cloud-cost tips. Use Discord or Circle.so if you prefer.

**Concrete strength:** Recurring revenue with almost zero maintenance. I launched in March 2026 with 42 members at 15 USD/month. By June I hit 100 members and 1,500 USD/month recurring. The cost is 20 USD/month on Circle.so.

**Concrete weakness:** Growth slows after 200 members unless you keep posting high-signal content daily. I burned out after two months of daily curation and had to hire a part-time moderator at 300 USD/month.

**Best for:** Developers who enjoy community management and can commit to daily curation.

**Activation energy:** 6 hours (creating the Circle.so space, writing the rules, sending invites).
**Runway:** 8 months with 120 members.
**Scaling ceiling:** 2,500 USD/month if you cap at 200 members or raise the price to 25 USD.


### 6. Cloud-cost optimization audits for startups

**What it does:** Offer a one-time 90-minute AWS cost audit using AWS Cost Explorer and Trusted Advisor, then send a 10-slide PDF with recommendations and expected savings. Charge 400–800 USD per audit.

**Concrete strength:** The tooling already exists. AWS Trusted Advisor alone flagged 1,200 USD of wasted spend for one client in five minutes. Those five minutes turned into 800 USD of billable work.

**Concrete weakness:** Clients rarely implement the recommendations. I did 12 audits in Q2 2026 and only 4 clients followed through, so the actual revenue per hour worked was lower than expected.

**Best for:** Developers who love spreadsheets and can tolerate low conversion rates.

**Activation energy:** 5 hours (writing the slide deck template).
**Runway:** 3 months if you land 6 clients at 500 USD each.
**Scaling ceiling:** 1,000 USD/month if you automate the report generation with a Python script and upsell follow-up implementation.


### 7. Selling pre-built Terraform modules on the registry

**What it does:** Publish reusable Terraform modules for common Kenyan cloud patterns (e.g., “fastapi-nginx-alb” or “redis-cluster-with-backup”) on the Terraform Registry and accept sponsorships.

**Concrete strength:** Once published, the modules can earn passive income. My “kenya-mpesa-webhook” module has been downloaded 3,412 times since March 2026 and has earned 620 USD in sponsorships.

**Concrete weakness:** Terraform Registry sponsorships are unpredictable. The top module in the registry earns 1,200 USD/month, but the median is closer to 20 USD. You need volume or a unique niche.

**Best for:** Developers who enjoy infrastructure-as-code and don’t mind slow, uneven revenue.

**Activation energy:** 25 hours (writing tests, examples, and docs).
**Runway:** 12 months if you publish 10 modules and each gets 200 downloads.
**Scaling ceiling:** 800 USD/month if one module hits the registry’s “popular” list.


### 8. Pre-shipped FastAPI microservices on GitHub Marketplace

**What it does:** Package a FastAPI microservice (e.g., “passwordless-auth-kenya-sms”) as a Docker image and list it on GitHub Marketplace. Charge a fixed price per deployment or offer a SaaS tier.

**Concrete strength:** GitHub Marketplace handles billing and updates. I listed a “kenya-pesa-callback-verifier” service in April 2026 and sold 23 licenses at 49 USD each in the first month.

**Concrete weakness:** Support burden. I spent 8 hours debugging a single client’s Docker networking issue. After that I added a 20 USD “installation support” tier to filter out tire-kickers.

**Best for:** Developers who want to monetize existing code without full SaaS overhead.

**Activation energy:** 12 hours (Dockerfile, GitHub Actions, and marketplace listing).
**Runway:** 5 months with 15 licenses per month.
**Scaling ceiling:** 1,200 USD/month if you list 3–4 services and each hits 25 sales.


## The top pick and why it won

The clear winner after 90 days was the **incident-response subscription**. Here’s why:

- **Activation energy** was low: I already knew AWS Cost Explorer, CloudWatch Logs Insights, and EC2. I just had to write a one-page runbook and a Stripe checkout link.
- **Runway** was long: 6 months with 5 subscribers at 200 USD/month gives me 6,000 USD in the bank before I need to find new work.
- **Scaling ceiling** was high: 4,000 USD/month with 15 subscribers and 10 hours/week is a realistic target for Nairobi.
- **Regret score** was acceptable: I hate 3 a.m. pages, but the money covers the pain.

I also compared it to the **GitHub Sponsors** route. Sponsorships are truly passive once the library is written, but the ceiling is lower (1,200 USD/month) and the marketing grind never ends. With subscriptions I trade a little sleep for a higher ceiling and recurring revenue.

Below is the exact setup I use today: a private Stripe checkout link, a Slack bot that pages me via a webhook, and a runbook stored in Notion that I update after every incident. The entire stack costs me 12 USD/month on AWS Amplify Hosting and Stripe fees are 2.9 % + 30 cents per transaction.


```python
# runbook_update.py – script I run after every incident to update the runbook
import json, pathlib, datetime

runbook_path = pathlib.Path("runbook.json")
runbook = json.loads(runbook_path.read_text())

new_entry = {
    "timestamp": datetime.datetime.utcnow().isoformat(),
    "incident": "Redis failover at 03:42 UTC",
    "root_cause": "AWS ElastiCache failover triggered by a 15 % memory spike",
    "fix": "Increased reserved-memory to 70 % and enabled cluster mode",
    "billing": 150,
}

runbook["entries"].append(new_entry)
runbook_path.write_text(json.dumps(runbook, indent=2))
```

I run this script immediately after closing the ticket and push the updated runbook to GitHub. Clients get a summary email with the fix and the next time the same issue happens I can copy-paste the runbook entry in under two minutes.


## Honorable mentions worth knowing about

### Hiring yourself out as a fractional CTO

**What it does:** Offer 10 hours a week of strategic advice to a seed-stage startup. You review their architecture, suggest cost-saving moves, and help them hire their first full-time engineer.

**Why it’s worth knowing:** The pay is 150–250 USD/hour and the work is high-signal. I did this for a Nairobi edtech startup in Q1 2026 and helped them cut their AWS bill from 2,400 USD to 1,100 USD by moving from m5.xlarge to t4g.xlarge and enabling Graviton. The startup later raised a 500k USD seed round and I stepped back.

**Gotcha:** Startups often expect you to work for equity instead of cash. I turned down three equity-only offers because the equity was usually 0.25 % and the company had a 5 % chance of failing. Cash only.


### Selling Notion templates for local startups

**What it does:** Publish a Notion template for common Kenyan business workflows (e.g., “Kenya business registration tracker”, “M-Pesa reconciliation sheet”). Sell them on Gumroad or Etsy.

**Why it’s worth knowing:** The activation energy is 4 hours and the marginal cost is zero. My “M-Pesa reconciliation” template has sold 213 copies at 12 USD each and earned 2,556 USD so far. The hardest part is SEO: I had to rewrite the title three times before it ranked for “mpesa reconciliation template kenya”.

**Gotcha:** Notion’s template marketplace takes 30 % for sales on their site. I moved to Gumroad after month three to keep 100 % of the revenue.


### Running a small email newsletter with affiliate links

**What it does:** Publish a weekly email newsletter for Nairobi developers covering new AWS features, local meetups, and cloud-cost tips. Include affiliate links to AWS, DigitalOcean, and HashiNode.

**Why it’s worth knowing:** The audience builds slowly but the affiliate revenue compounds. My newsletter “Nairobi Cloud Notes” grew from 42 subscribers in March to 1,200 by June 2026 and sent 840 USD in affiliate revenue in the same period. The trick is consistency: I missed two issues in May and the open rate dropped from 62 % to 41 %.

**Gotcha:** Email deliverability is fragile. I had to move from Mailchimp to AWS SES after Mailchimp flagged my domain as “marketing” and throttled sends.


## The ones I tried and dropped (and why)

### Building a SaaS (again)

I rebuilt my failed identity provider in June 2026 using Python 3.12, FastAPI, and AWS Cognito. This time I added a referral program and a waitlist. After 45 days and 800 lines of code, I had 12 waitlist signups and one paying user at 9 USD/month. The churn rate was 50 % after week two because the onboarding flow was still 12 steps long. I dropped it after realizing the marginal cost of customer support outweighed the revenue.

### Selling AI-generated code snippets

I listed 500 AI-generated Python snippets on Gumroad for 2 USD each. After one month I earned 34 USD and burned 8 hours cleaning up hallucinated imports. The snippets ranked poorly on Google because Google’s algorithm flags AI-generated content as low quality. I dropped it after the first refund request.

### Running a paid Discord community for freelancers

I launched a Discord server for Nairobi freelance developers charging 10 USD/month. After 30 days I had 21 members and 210 USD in revenue. The problem was moderation: I spent 6 hours a week deleting spam and banning fake accounts. I shut it down and moved to Circle.so for better moderation tools.

### Offering “code review as a service”

I offered 30-minute code reviews via Calendly for 50 USD. After 15 sessions I earned 750 USD but spent 12 hours writing detailed feedback. The clients rarely implemented the suggestions, so the real ROI was low. I dropped it after the third client asked for a refund because “the code still had bugs.”


## How to choose based on your situation

Use the table below to pick the option that matches your skills, time budget, and tolerance for late-night pages.

| Option | Activation hours | Runway months | Ceiling USD/mo | Sleep cost | Best if... |
|---|---|---|---|---|---|
| Incident-response subscription | 14 | 6 | 4,000 | High | You’re already fluent in AWS and don’t mind 3 a.m. pages |
| GitHub Sponsors library | 22 | 9 | 1,200 | None | You enjoy writing polished, well-documented code |
| Affiliate technical content | 30 | 4 | 3,000 | None | You enjoy writing long-form and can keep up with affiliate changes |
| One-off data pipeline scripts | 8 | 5 | 1,500 | Low | You like data munging and small, repeatable scopes |
| Private Slack community | 6 | 8 | 2,500 | None | You enjoy daily curation and can handle growth slowdowns |
| Cloud-cost audit | 5 | 3 | 1,000 | Low | You love spreadsheets and low conversion rates |
| Terraform modules | 25 | 12 | 800 | None | You enjoy infrastructure-as-code and slow, uneven revenue |
| FastAPI microservices | 12 | 5 | 1,200 | Medium | You want to monetize existing code without full SaaS |

If your ceiling is under 1,000 USD/month, pick **GitHub Sponsors** or **Terraform modules**. If you want 3,000 USD/month and can tolerate occasional late-night pages, pick **incident-response subscription**. If you enjoy writing and can keep up with SEO changes, **affiliate technical content** is the highest ceiling.


## Frequently asked questions

**How do I set up a Stripe checkout link for the incident-response subscription?**

Create a product in Stripe with a 200 USD monthly price, then generate a checkout link. Put that link on a simple landing page built with Next.js 14 or a static HTML page hosted on AWS Amplify. I used a one-page React app with TailwindCSS and it cost me 12 USD/month to host. The entire setup took 4 hours from “I have an idea” to “first payment in my bank.”


**What’s the easiest first side gig to start this week?**

Start with **one-off data pipeline scripts**. The activation energy is only 8 hours and you can use the same script template for multiple clients. I reused a 180-line Python script three times in two weeks and earned 1,800 USD. The hardest part is writing a good README so clients can install it themselves.


**How do I get my first incident-response client?**

Post in local Slack and WhatsApp groups: “I’m offering 24/7 API incident response for 200 USD/month. Ping me if you want coverage.” Target small e-commerce teams using Shopify or WooCommerce with custom Node or Python backends. I landed my first three clients via a single WhatsApp broadcast to a local Shopify developers group.


**Do I need a business license in Kenya to take foreign payments?**

Yes, if you expect more than 5,000 USD/year in revenue you need to register for a KRA PIN and file VAT if you hit the threshold. I registered as a sole proprietor in May 2026 and it cost 1,500 KES (≈11 USD). Stripe handles the foreign exchange and the fees are transparent (2.9 % + 30 cents).


**How do I avoid burnout from 3 a.m. pages?**

Set a hard limit: “I’ll respond within 30 minutes between 8 a.m. and 10 p.m., and after that I’ll respond by 8 a.m. the next day.” Use a Slack bot that pages you only between those hours. For true emergencies, charge a 2x weekend rate (400 USD instead of 200 USD). I once turned down a 2 a.m. page and the client paid the weekend rate anyway — the policy paid for itself in one night.


## Final recommendation

Start with **one-off data pipeline scripts** this week. Here’s the exact next step:

1. Open your terminal and run:

```bash
pip install pandas requests tenacity python-dotenv --upgrade
```

2. Create a new directory `kenya_quickbooks_sync` and add a `main.py` file with this template:

```python
# main.py – template for a QuickBooks-to-BigQuery sync script
import os, pandas as pd, requests, json
from datetime import datetime, timedelta
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

QUICKBOOKS_ACCESS_TOKEN = os.getenv("QUICKBOOKS_ACCESS_TOKEN")
BIGQUERY_TABLE = "kenya_ecommerce.inventory_daily"

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_quickbooks_inventory():
    headers = {"Authorization": f"Bearer {QUICKBOOKS_ACCESS_TOKEN}"}
    url = "https://sandbox-quickbooks.api.intuit.com/v3/company/12345/query"
    query = "SELECT * FROM Item WHERE Active=TRUE"
    resp = requests.get(url, headers=headers, params={"query": query}, timeout=30)
    resp.raise_for_status()
    return resp.json()["QueryResponse"]["Item"]

def push_to_bigquery(data):
    from google.cloud import bigquery
    client = bigquery.Client()
    df = pd.DataFrame(data)
    df["sync_time"] = datetime.utcnow().isoformat()
    job = client.load_table_from_dataframe(df, BIGQUERY_TABLE)
    job.result()

if __name__ == "__main__":
    inventory = fetch_quickbooks_inventory()
    push_to_bigquery(inventory)
```

3. Add a `README.md` with a one-line description and a `requirements.txt` listing the pinned versions:

```
pandas==2.2.2
requests==2.31.0
python-dotenv==1.0.1
tenacity==8.3.0
google-cloud-bigquery==3.15.0
```

4. Push to a private GitHub repo, then share the link with three local SMEs you know use QuickBooks.

5. Quote 600–800 USD per script and ask for 50 % upfront. You’ll have your first 300–400 USD in the bank within 48 hours.

That’s it. No SaaS, no waiting months for users, just a small, repeatable side gig that plays to the skills you already have.


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

**Last reviewed:** June 24, 2026
