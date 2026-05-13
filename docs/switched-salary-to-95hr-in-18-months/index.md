# Switched salary to $95/hr in 18 months

This is a topic where the standard advice is technically correct but practically misleading. Here's the fuller picture, based on what I've seen work at scale.

**## The situation (what we were trying to solve)**

In early 2022, I was stuck earning $18 per hour as a junior developer in Accra, building internal tools for a local fintech that paid in Ghana cedi. The rent for my one-bedroom flat was 1,200 cedi, and the dollar exchange rate was 6.2 cedi to 1 USD. After converting my take-home pay, I could only afford 30% of my rent in dollars. I needed a way to earn in USD so I could save, invest, and eventually move abroad for better career growth. I knew I had to transition from a local salary to a global rate, but I didn't know where to start. I had no open source contributions, no public portfolio, and my GitHub profile was empty except for a few school projects. I decided to document the journey publicly to hold myself accountable, which forced me to actually follow through on promises like "I’ll release this library" or "I’ll write a blog post."

The first step was to identify what I could realistically offer to global clients. I had three years of experience building CRUD apps with Django and React, but I lacked specialized skills that command premium rates. I knew I couldn’t compete with senior engineers charging $150/hr right away. My goal was to reach $50/hr within a year, then double that within the next six months. I started by calculating my financial runway: I had about $2,000 in savings, which would last about three months if I quit my job immediately. So, I decided to transition while keeping my local job, working nights and weekends on freelance projects.

I set three non-negotiable constraints for the transition: (1) I would only work with clients who paid in USD or stablecoins, (2) I would only take on projects that aligned with my long-term interests (developer tools, DevOps, and backend systems), and (3) I would avoid exploitative gig platforms like Upwork’s lowest tiers, where rates often start at $10/hr. My hypothesis was that if I could deliver value quickly and reliably, I could command higher rates by demonstrating expertise in specific niches rather than being a generalist.


**Summary:** I started earning $18/hr in Accra but needed USD income to cover rent and save. I set a goal to reach $50/hr within a year, then double it, by targeting USD-paying clients in niche areas like DevOps and developer tools while avoiding exploitative platforms.


**## What we tried first and why it didn’t work**

My first attempt was signing up for Upwork and Fiverr, targeting gigs like "I will build a Django REST API for $200." Within two weeks, I completed three projects at $200 each, averaging 40 hours of work per project. After accounting for platform fees (20% on Upwork, 20% on Fiverr), taxes, and the time spent on proposals, my effective hourly rate was closer to $12/hr. Worse, these gigs were one-off scripts with no scope for follow-up work, so I was trading time for low-value projects. One client asked me to build a scraper for a site that blocked bots aggressively, and I spent 12 hours debugging before giving up. Another client wanted a "simple" Django app that turned out to require real-time WebSocket updates, which I hadn’t budgeted for. The projects were never as simple as the client promised.

Then I tried cold emailing. I sent 50 cold emails to small US-based startups, offering to build their MVP for $3,000. Only two replied, and both ghosted after I sent a detailed proposal. One startup said they’d get back to me in a week; they never did. Another asked for a 15-minute call, then said they’d decided to use no-code tools instead. Cold outreach felt like shouting into a void. I realized I needed a way to demonstrate value before asking for money, not the other way around.

Next, I tried contributing to open source. I picked a popular Python library called `requests-oauthlib`, fixed a minor bug, and opened a pull request. The maintainer took two months to review it. When he finally commented, he said the bug wasn’t actually a bug but a misuse of the library. I felt embarrassed. I then tried writing a tutorial on how to use Django with PostgreSQL, but it got 12 views in three months. Open source felt like a slow burn with unclear ROI. Meanwhile, my local job was consuming 40 hours a week, leaving little energy for side projects that might pay off someday.

Finally, I tried joining a local freelance marketplace where clients were based in Europe and North America but paid in USD. The platform took a 15% cut, and the rates were still low—$25/hr for backend work. I took on two clients, but the communication overhead was brutal. One client wanted daily standups at 3 AM my time because they were in Berlin. Another client kept changing requirements every other day, leading to scope creep. I ended up working 60 hours a week to deliver a project that paid $1,500, netting me about $10/hr after fees and taxes. I was burning out without making progress toward my goal.


**Summary:** Cold outreach, gig platforms, and local freelance marketplaces failed because they either paid too little, had unpredictable clients, or required too much unpaid time. Open source contributions didn’t yield immediate results either. I needed a different approach—one that leveraged my existing skills while building credibility in a way that global clients would pay for.


**## The approach that worked**

The breakthrough came when I realized I needed to combine three things: a public portfolio that demonstrated real expertise, a niche that was underserved but willing to pay, and a repeatable process for delivering value quickly. I chose the niche of "developer tooling for small teams" because small companies often have problems that big tech companies have already solved internally, so they’re willing to pay for off-the-shelf solutions.

I started by writing a small Python library called `django-env-health`, which checks if a Django application’s environment variables match the expected schema. It was inspired by a frustrating experience where a production app crashed because of a missing environment variable. I open-sourced it on GitHub and wrote a short README with installation instructions and an example. Within a week, it got 50 stars and a few issues. One issue led to a bug fix that improved the library’s reliability for users in India and Nigeria, where server configurations often vary.

Next, I used the library as a case study to attract clients. I wrote a blog post titled "How We Cut Django Downtime by 40% with Environment Health Checks" and published it on Dev.to and Hacker News. The post got 8,000 views in three days and led to two consulting gigs: one to help a Berlin-based startup debug a Django deployment issue, and another to review a client’s infrastructure setup. Both clients paid $75/hr and $85/hr respectively, and both projects were scoped to 10 hours each. The Berlin client later referred me to another company, leading to a $5,000 contract for a three-month retainer.

The key insight was that the library itself wasn’t the product—it was the credibility signal. Clients weren’t paying for the library; they were paying for the expertise it represented. By solving a specific pain point (environment misconfigurations) and making the solution reproducible, I could demonstrate value before pitching a project. I also started using the library as a conversation starter in cold emails: "I noticed your team uses Django. We open-sourced a tool to prevent environment-related outages—here’s how it works."

I then doubled down on writing and speaking. I published three more tools: `django-async-cron`, a drop-in replacement for Django’s built-in cron that runs tasks asynchronously; `django-split-settings`, a library for managing Django settings in multiple files; and `django-money-trace`, a library to track financial transactions in multi-currency apps. Each tool targeted a specific pain point that small teams faced but big companies had already solved internally. The libraries accumulated 200+ stars combined, and each one led to consulting leads. By month 12, I had 14 open source projects with a total of 1,200 stars on GitHub.


**Summary:** The winning strategy was to build small, useful tools in a niche (Django developer tooling) and use them as credibility signals to attract higher-paying clients. Writing and open source became the bridge between my local salary and global rates, turning side projects into a pipeline for consulting work.


**## Implementation details**

To execute this strategy, I followed a repeatable process for each library:

1. **Identify the pain point.** I used my own frustration as a guide. For `django-env-health`, it was the third time in six months that an environment variable mismatch caused a production outage. I wrote a script to validate all environment variables at startup, then generalized it into a reusable library.

2. **Ship minimal, working code.** Each library had to do one thing well. `django-async-cron` replaced Django’s synchronous cron jobs with async tasks, reducing database locks during peak hours. It was 150 lines of code and took me two days to write and test.

3. **Document thoroughly.** Every library included a README with installation instructions, usage examples, and a clear explanation of the problem it solved. I used MkDocs for documentation and hosted it on GitHub Pages. One library (`django-split-settings`) got adopted by a team in India because the docs included a migration guide from Django’s built-in settings file.

4. **Market the solution.** I published each library on Dev.to, Hacker News, and Reddit’s r/django. I also shared them in Django community Slack channels and Twitter threads. The goal wasn’t just visibility—it was to position myself as someone who understood Django’s internals well enough to build extensions.

5. **Convert attention into leads.** I added a consulting section to my GitHub profile and personal website, listing services like "Django Infrastructure Review" and "DevOps Automation Setup" at $95/hr. I included a link to my Calendly to book a 30-minute discovery call. Within three months, 60% of my inbound leads came from GitHub or my website.

Here’s an example of the README for `django-env-health`:

```markdown
# django-env-health

Check your Django app's environment variables against the expected schema at startup.

## Why?

Environment mismatches cause 40% of production outages in Django apps (based on anecdotal reports from 12 teams I interviewed).

## Installation

```bash
pip install django-env-health
```

## Usage

Add to `INSTALLED_APPS`:

```python
# settings.py
INSTALLED_APPS = [
    ...,
    'env_health',
]
```

Define your expected variables in `settings.py`:

```python
ENV_HEALTH = {
    'DATABASE_URL': {'required': True, 'type': 'str'},
    'SECRET_KEY': {'required': True, 'type': 'str'},
    'DEBUG': {'required': False, 'type': 'bool', 'default': False},
}
```

On startup, the library will validate your environment and raise an exception if any required variables are missing or of the wrong type.
```


I also used a lightweight analytics tool called [GoatCounter](https://www.goatcounter.com) to track traffic to my GitHub and website. This helped me see which libraries were gaining traction. For example, `django-env-health` got 300 unique visitors in the first month, while `django-async-cron` got 800. The data told me which problems were most urgent for small teams.


**Summary:** Each library followed a strict pattern: identify a real pain point, write minimal working code, document it thoroughly, market it in the right communities, and convert attention into consulting leads. The process was repeatable and scalable, turning side projects into a pipeline for higher-paying work.


**## Results — the numbers before and after**

Before starting this transition in January 2022, my monthly income was fixed at $360 (18 USD/hr * 200 hours/month). After taxes and rent, I had about $100 left to save each month. I was living paycheck to paycheck in USD terms, even though my local salary was covering my rent in cedi.

Twelve months later, in January 2023, my monthly income from freelance and consulting was $3,800, all paid in USD or EUR. This included $1,200 from retainers, $1,500 from fixed-scope projects, and $1,100 from ad-hoc consulting. My effective hourly rate across all projects was $72/hr, up from $18/hr. The highest-paying client was a German startup that paid $95/hr for a three-month retainer to review their Django and PostgreSQL infrastructure. They renewed the contract for another three months at the same rate.

By month 18 (July 2023), my monthly income reached $5,200, with 70% of it recurring from retainers. My effective hourly rate had stabilized at $95/hr across all projects. I had also reduced my time spent on prospecting from 10 hours a week to 2 hours a week, because most leads came from my open source projects and blog posts. I was able to save $2,500 a month after all expenses, which gave me a six-month financial runway.

Here’s a breakdown of the income trajectory:

| Month       | Income (USD) | Effective Hourly Rate | Projects Type                     |
|-------------|--------------|-----------------------|-----------------------------------|
| Jan 2022    | $360         | $18                   | Local job only                    |
| Jul 2022    | $1,800       | $55                   | 2 retainers, 3 fixed-scope        |
| Jan 2023    | $3,800       | $72                   | 3 retainers, 2 ad-hoc, 1 referral |
| Jul 2023    | $5,200       | $95                   | 4 retainers, 1 product contract   |

I also measured the time spent to land each client. In the first six months, it took an average of 20 hours of unpaid time (proposals, calls, setup) to land a $2,000 project. By the 12th month, it took 4 hours of unpaid time to land a $3,000 project, because clients were coming to me through GitHub or my blog.

One surprising result was the compounding effect of open source. The `django-env-health` library alone led to $8,000 in consulting revenue over 12 months, not from the library itself but from the credibility it provided. Clients would say, "I saw your library and thought you could help with our Django setup." The library acted as a filter—only teams that valued reliability and automation reached out, which meant higher-quality leads.

I also tracked my time investment. From January 2022 to July 2023, I spent about 800 hours on open source, blogging, and marketing. That’s roughly 400 hours a year, or 8 hours a week. The ROI was 6.5x: for every hour spent on side projects, I earned $6.50 in consulting revenue within a year. This convinced me to keep investing in public work even when it felt slow.


**Summary:** In 18 months, I increased my effective hourly rate from $18 to $95, grew monthly income from $360 to $5,200, and reduced prospecting time from 10 hours/week to 2 hours/week. Open source and writing were the primary drivers of this growth, with the highest-paying client paying $95/hr for a retainer.


**## What we’d do differently**

If I could go back, I would have started with a clearer monetization strategy for my open source projects. In the beginning, all my libraries were MIT-licensed with no commercial clauses. While this helped adoption, it also meant I couldn’t offer commercial support or dual-license features for clients who wanted guaranteed SLAs. Two clients later asked if I could add a paid feature to `django-async-cron` for enterprise support, but I couldn’t because the library was fully open source.

I’d also have set up a proper business entity earlier. For the first 12 months, I invoiced clients as an individual, which made it harder to open a business bank account or accept payments via Stripe Connect. In month 13, I registered an LLC in Delaware (cost: $125) and set up Stripe to automate invoice generation. This reduced my invoicing time from 2 hours a week to 10 minutes, and made it easier to accept EUR payments via Wise.

Another mistake was not tracking client acquisition cost early enough. I spent $300 on Google Ads for one library’s GitHub page, but didn’t track how many leads it generated. I later realized the ads led to no paying clients, so I stopped them. I should have used a simple UTM parameter to track traffic sources, which would have saved $300 and two weeks of ad spend.

I also would have specialized faster. In the first six months, I took on projects in React, Go, and AWS, even though my strongest skill was Django. This diluted my messaging and made it harder to position myself as an expert. By month 12, I only took on Django-related projects, and my lead conversion rate increased by 30% because clients knew exactly what I could do.

Finally, I’d have automated more of the onboarding process. I used to spend 1–2 hours per client setting up contracts, invoices, and project boards. After switching to a template-based system with HelloSign for contracts and Stripe for invoices, I reduced onboarding time to 15 minutes. This freed up time to focus on high-value work instead of administrative tasks.


**Summary:** I would have dual-licensed libraries, set up an LLC earlier, tracked marketing spend with UTM parameters, specialized in one stack (Django), and automated onboarding to save time and reduce friction.


**## The broader lesson**

The principle that drove this transition was **credibility arbitrage**: using public, reproducible work to signal expertise and attract higher-paying clients without cold outreach or bidding wars. Most developers wait for permission to charge premium rates—they look for a job title, a degree, or a referral. But the market doesn’t care about titles; it cares about outcomes. If you can consistently solve a specific problem for a specific audience, the rate will follow.

This approach works because small teams and startups are starved for expertise in areas that big tech has already solved internally. They don’t need a full-time engineer; they need someone who can unblock them quickly. By open-sourcing tools that solve these niche problems, you position yourself as the person who understands the domain deeply enough to build extensions. The library itself is not the product—it’s the proof that you can deliver value without a long hiring process.

The second lesson is that **compounding visibility is more powerful than sporadic outreach**. One well-written blog post or one useful library can generate leads for years. I still get consulting inquiries from a 2022 blog post about Django performance tuning. The key is to focus on one niche and go deep, not wide. If you try to build tools for React, Python, and cloud infrastructure at the same time, you’ll dilute your signal and confuse potential clients.

Finally, **financial runway is the hidden variable in career transitions**. Most developers quit their jobs too early or stay too long because they can’t afford to take the risk. By keeping my local job while building the side income, I reduced the pressure to take low-paying gigs. This gave me the space to experiment with open source and writing without financial desperation driving my decisions.


**Summary:** The broader lesson is credibility arbitrage—using public work to signal expertise and attract high-paying clients. Compounding visibility in a niche niche is more powerful than sporadic outreach, and financial runway is crucial for experimentation without desperation.


**## How to apply this to your situation**

Start by asking: *What pain point do I encounter regularly that no one has solved with a simple tool?* It could be a build step that fails unpredictably, a deployment script that breaks every Friday, or a configuration file that grows unreadable. Build a minimal solution to that problem and open-source it. Don’t aim for perfection—aim for "it works for me" and then generalize it.

Next, document the solution thoroughly. Write a README that answers: What problem does this solve? How do I install it? How do I use it? Include a short code example. Publish it on Dev.to, Hacker News, and Reddit’s r/Python or r/django (depending on your stack). If your tool is for JavaScript developers, post it on r/javascript. The goal is to get at least 100 views and 20 stars within a month.

Then, add a consulting section to your GitHub profile and personal website. List services like "DevOps Automation Review" or "Library Architecture Audit" at a rate higher than your local salary. Include a Calendly link for a 30-minute discovery call. Most developers underprice their services because they fear rejection. Start at $75/hr even if your first client is a friend—it trains you to say no to low-ball offers.

Finally, track everything. Use a simple spreadsheet to record time spent on open source, leads generated, and revenue earned. After three months, calculate your ROI: for every hour spent on side projects, how much consulting revenue did you earn within a year? If the number is less than 2x, double down on the niche or the marketing channel.

Here’s a step-by-step playbook:

1. Pick a niche (e.g., Django, React, AWS, PostgreSQL).
2. Identify a recurring pain point in that niche.
3. Build a minimal tool (under 200 lines of code).
4. Open-source it and document it thoroughly.
5. Publish it in the right communities.
6. Add a consulting section to your profiles.
7. Track leads and revenue for three months.
8. Double down on what works or pivot if the ROI is less than 2x.


**Summary:** To apply this, start by building a minimal tool for a recurring pain point in your niche, document it thoroughly, publish it in the right communities, add a consulting section to your profiles, and track ROI for three months before doubling down.


**## Resources that helped**

- **Books:** *The $100 Startup* by Chris Guillebeau (for the mindset that small tools can generate income), *Shape Up* by Basecamp (for product thinking in small tools)
- **Tools:** GitHub (for hosting code and attracting leads), MkDocs (for clean documentation), GoatCounter (for lightweight analytics), Calendly (for scheduling calls), Stripe (for invoicing), Wise (for receiving EUR payments)
- **Communities:** Dev.to (for tutorials), Hacker News (for visibility), r/django and r/Python (for niche discussions), Django Community Slack (for direct feedback)
- **Templates:** I used [Cookiecutter](https://github.com/cookiecutter/cookiecutter) to scaffold Python projects, which saved hours of setup time. The `cookiecutter-pypackage` template was particularly useful.


Here’s an example of a Cookiecutter template command I used to start a new library:

```bash
cookiecutter https://github.com/audreyr/cookiecutter-pypackage
project_name="django-env-health"
project_slug="django_env_health"
version="0.1.0"
Select open_source_license="MIT"
```


**Summary:** The key resources were books for mindset, tools for automation and analytics, and communities for visibility. Using templates like Cookiecutter reduced setup time and kept projects consistent.


**## Frequently Asked Questions**

**How do I price my first consulting gig if I have no experience?**
Start at $75/hr for the first three clients, even if it feels high. Charge by the hour for fixed-scope work, and offer a 10% discount for upfront payment. After three successful projects, raise your rate to $95/hr. The goal is to train clients to pay you what you’re worth, not to underprice yourself out of the market.


**Is open source necessary, or can I just blog?**
Open source is a stronger signal than blogging alone because it shows you can write production-grade code, not just tutorials. However, if you’re not comfortable open-sourcing code, start with a detailed technical blog post that solves a specific problem. The key is to demonstrate expertise in a way that clients can verify—whether through code or prose.


**How long does it take to see results?**
It took me 6 months to land my first $2,000 project, and 12 months to reach $3,800/month. The first 3–6 months are the hardest because you’re building credibility from scratch. After that, leads compound: one blog post or library can generate inquiries for years. Set a 12-month goal and track small wins monthly.


**What if my local salary is enough to live on, but I want to transition anyway?**
Treat the transition as a side project first. Dedicate 8–10 hours a week to building tools, writing, and networking. Use your local salary as runway to experiment without financial pressure. Once your side income reaches 50% of your local salary, start reducing hours at your job or shifting to part-time. I reduced from 40 hours to 20 hours a week at my local job after 15 months of side income.


**Should I quit my job immediately once I get a client?**
No. Use your local job as a safety net until your side income is stable for 6 months. Many freelancers underestimate the time it takes to find the next client or deal with payment delays. Keep your job until you have three months of expenses saved in USD, plus a pipeline of at least two paid projects.


**Summary:** Start pricing at $75/hr for first clients, use open source as a stronger signal than blogging, expect 6–12 months for results, treat transition as a side project while keeping local job, and only quit after 6 months of stable side income.


Start by open-sourcing a minimal tool for a recurring pain point in your niche this week.