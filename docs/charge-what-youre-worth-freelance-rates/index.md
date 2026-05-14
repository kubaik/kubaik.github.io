# Charge what you're worth: freelance rates

A colleague asked me about this last week and I realised I couldn't explain it cleanly. Writing this post forced me to think it through properly — which is usually how it goes.

## Why I wrote this (the problem I kept hitting)

I once sent a freelance proposal for a Django REST project with a clear scope: three endpoints, one model, and a week of work. The client came back with, “This looks good, but our budget is $800.” I countered with $1,600. They replied, “That’s too much for a junior.” I had no idea how to justify the number, so I split the difference—$1,200—and still felt guilty.

After six similar conversations in three months, I realized I was pricing by vibes, not by the work. I started tracking every project: hours, complexity, tech stack, client industry, and final take-home pay. What shocked me was the spread—rates ranged from $35/hour for a WordPress plugin to $180/hour for a Go microservice. The real issue wasn’t skill; it was the absence of a repeatable framework to translate scope into dollars.

Most freelancers I’ve met use one of three mental models: the “I need to eat” floor, the “market average” guess, or the “whatever the client accepts” ceiling. None of those models survive three client conversations. I needed something that turned vagueness into a number anyone could check.

This guide is the framework I wish existed that day. It doesn’t just give numbers; it shows how to derive them from your actual constraints and how to defend them without sounding like you’re reading from a script.


## Prerequisites and what you'll build

You don’t need a fancy toolkit—just a browser, a calculator, and willingness to confront your own numbers. I’ll use three artifacts you can copy immediately: a rate calculator spreadsheet, a client-facing rate card template, and a client conversation script. Everything is plain text so you can adapt it in Google Sheets, Notion, or Excel.

By the end, you’ll have:
- A personal hourly floor based on your expenses, taxes, and desired profit.
- A project rate that folds in scope, complexity, and buffer.
- A client-facing rate card that converts your internal math into language clients understand.
- A script to negotiate without justifying yourself.

I’ll include real numbers from my 2023 freelance year: $182k revenue, $54k expenses, $26k taxes, and $102k take-home. Those figures matter because they’re not averages—they’re mine. If your situation differs, swap in yours and recalculate.


## Step 1 — set up the environment

1. Open a fresh Google Sheet titled “Freelance Rate Calculator.”
   Why: Spreadsheets force you to confront assumptions. A napkin math session in your head will drift; a sheet won’t.

2. Create four tabs: 
   - Expenses
   - Taxes
   - Hourly Floor
   - Project Multiplier

3. Fill the Expenses tab with real monthly numbers. I once missed health insurance for six months because I used an old estimate. The actual premium added $420/month.
   | Category | Amount ($) | Notes |
   | Business costs | 320 | Hosting, domains, tools |
   | Health insurance | 420 | 2023 COBRA cost |
   | Rent & utilities | 1800 | Split 20% home office |
   | Groceries | 550 | Keep real, not “living cheap” |
   | Transport | 220 | Public transit + rideshare |
   | Savings & buffer | 1200 | Emergency fund top-up |
   | **Total** | **4510** |

4. In the Taxes tab, add three rows: Federal, State, Self-employment. I learned the hard way that self-employment tax is 15.3% on top of income tax. If you’re in California, add an extra 9.3% for state.
   - Federal bracket: 24% (2023 brackets)
   - State tax: 9.3%
   - Self-employment: 15.3%
   - Effective tax rate: 24% + 9.3% + (24% * 15.3%) ≈ 38.6%

5. In the Hourly Floor tab, type:
   
   ```
   Total monthly expenses = 4510
   Desired monthly profit = 5000
   Target monthly income = Total expenses + Desired profit
   Hours you can bill/month = 120 (16 days × 7.5 hours)
   Hourly floor = Target monthly income / Hours you can bill/month
   ```

   Plug in your numbers. My sheet returned $79/hour. That felt low the first time, but it’s the floor—not the rate you’ll charge. It’s what keeps the lights on if you get zero clients next month.

6. Save a copy of the sheet as “Rate Calculator v1.2024” and set the share link to “Anyone with the link can view.” You’ll send this to clients who ask for “transparency.”

7. Create a new Google Doc called “Client Rate Card.” Copy this template:
   
   ```markdown
   # Project Rates

   - Discovery call: $120 flat (1 hour max)
   - Backend API endpoint: $180 flat
   - Frontend component with tests: $240 flat
   - Bug fix under 3 hours: $80/hour (capped at 3 hours)
   - Bug fix over 3 hours: $110/hour
   - Monthly retainer (5 days/month): $3,200
   ```

   Update the numbers to match your hourly floor plus 30–50% buffer for project work. The flat fees remove the dreaded “you’re nickel-and-diming me” conversation.


## Step 2 — core implementation

1. Convert your hourly floor into project rates using three variables: scope, complexity, and urgency.
   Why: Clients don’t buy hours; they buy outcomes. A CRM integration is different from a real-time payment system, even if both take 40 hours.

2. Build a simple multiplier table in the Project Multiplier tab. I stole this from a friend who runs a dev shop and refined it over two years.
   | Complexity | Multiplier | Example |
   | Low | 1.3 | Static site rebuild |
   | Medium | 1.7 | REST API + React dashboard |
   | High | 2.2 | WebSocket microservice + mobile hooks |
   | Extreme | 3.0 | HIPAA-compliant ETL pipeline |

3. For each new project, fill a one-pager:
   - Scope: “Build a dashboard showing daily revenue for 10 stores.”
   - Complexity: “Medium—REST endpoints, PostgreSQL, D3 charts.”
   - Urgency: “Client needs it in 3 weeks.”
   - Buffers: “Add 15% for unknowns.”

4. Calculate:
   
   ```
   Project rate = (Estimated hours × Hourly floor) × Complexity multiplier × Buffer
   ```

   Example: 40 hours × $79 × 1.7 × 1.15 ≈ $6,160. Round to $6,200. 

5. Present the rate in the client rate card format, not as an hourly number. I once quoted $1,500 for a WordPress plugin and got pushback until I said, “This covers 20 hours of development, 3 rounds of revisions, and 6 months of support.” The client accepted immediately.

6. Add a “Not included” section to the rate card:
   - Third-party API costs
   - Domain registration
   - On-site meetings beyond Zoom
   - Scope creep beyond initial spec

   This prevented a client from demanding a free redesign after I delivered the MVP.


## Step 3 — handle edge cases and errors

1. The “budget too low” client: Use the rate card as a filter. If the client’s budget is 30% below your calculated project rate, decline with:
   
   ```
   “Our minimum project rate for this scope is $6,200. Anything less puts us both at risk for delays or shortcuts. If budget is tight, we can reduce scope or phase the work.”
   ```

   I once took a 25% haircut on a $4,500 project to “build trust.” I delivered late, burned out, and the client still haggled over every invoice. Never again.

2. The “hourly doubt” client: Anchor to flat rates only. If they insist on hourly, cap it at 1.5× your project rate. Example: $6,200 project becomes $9,300 hourly cap at $116/hour (80 hours max). This keeps the conversation bounded.

3. The “scope creep” client: Before every sprint, send a one-paragraph scope recap with:
   - What’s included
   - What’s extra
   - Price per extra hour or fixed fee

   I made a mistake by assuming “everyone knows what’s included.” A client added a real-time chat feature mid-project. I absorbed the cost because I didn’t have a signed change order. Now I send a Google Doc link and ask them to “click Accept” before starting the extra work.

4. The “I’ll pay later” client: Require a 30% deposit before starting. I once shipped a full e-commerce site for a client who vanished after delivery. The deposit clause saved $4,200 in unpaid invoices.


## Step 4 — add observability and tests

1. Track every project in a single Airtable base with these fields:
   - Client name
   - Project name
   - Start date
   - End date
   - Hours logged
   - Actual revenue
   - Scope changes
   - Client satisfaction (1–5)

2. Run a quarterly “rate audit.” Pull the last 12 projects and calculate:
   - Average hours per project vs. estimate
   - Average revenue per hour
   - Percentage of scope changes

   My 2023 audit showed I underestimated frontend work by 28% on average. I added a 1.4× frontend multiplier and reduced revision cycles by 40%.

3. Build a simple Slack bot that posts weekly:
   - Hours logged
   - Revenue generated
   - Client NPS (from a one-question survey)

   I built it in 45 minutes using Python, Slack’s Incoming Webhooks, and Google Sheets API. The bot surfaced a client who rated me 2/5 but never complained directly. I reached out, fixed a small bug, and turned them into a 5/5.

4. Add a “kill switch” rule: If any project’s actual revenue/hour drops below 80% of your hourly floor, pause the project. I kept a $2,800 project running for three months at $22/hour because I was “almost done.” The kill switch rule would have flagged it at $47/hour after 60 hours.


## Real results from running this

I ran this framework from July 2023 to June 2024. Here are the raw numbers:

- Projects booked: 22
- Total revenue: $182,300
- Expenses: $54,100 (includes $12,800 for a new laptop and health insurance)
- Taxes paid: $26,400
- Take-home: $101,800
- Average revenue/hour: $114
- Top 20% of projects: $52,000 revenue, $2,364/hour average
- Bottom 20% of projects: $18,300 revenue, $52/hour average

The bottom 20% projects were all fixed-scope WordPress rebuilds for local businesses. They took less time but also had the highest revision rates. I raised the flat fee by 40% and added a “design freeze” clause after two rounds of changes.

The top 20% were SaaS integrations and real-time dashboards for tech startups. They required deeper scoping upfront and a retainer model after delivery. One client renewed for $3,200/month for 12 months.

I also tracked client NPS. Projects that used the rate card template scored 4.6/5; projects that negotiated hourly scored 3.2/5. The difference wasn’t skill—it was clarity.


## Common questions and variations

**What if I’m just starting out and have no portfolio?**

Use a “starter rate” 20% below your hourly floor for the first five projects. My first five projects were at $55/hour instead of $79. I treated them as paid learning: I measured time to delivery, client satisfaction, and built three case studies. After the fifth project, I raised the rate to $79 and never looked back. You’ll lose money on paper, but the portfolio pays long-term.

**How do I justify a 3× multiplier for a “high-complexity” project?**

Break the multiplier into line items the client can see:
- “This requires a senior engineer at $116/hour instead of a mid-level at $79/hour (+$37/hour).”
- “We’re adding a QA engineer for 8 hours at $45/hour (+$360) because the spec touches patient data.”
- “We’re earmarking 15 hours for security review (+$1,740).”

Sum the line items and show the total. Clients accept complexity when it’s itemized, not abstract.

**What if the client pushes back on the deposit?**

Use the “deposit vs. trust” framing: “A 30% deposit covers our upfront costs for licenses, hosting, and dedicated hours. If we don’t deliver, you get a full refund. If you trust us enough to pay later, we’ll add a 15% late-fee clause.” 

I had one client refuse the deposit for a $12k project. I walked away. They came back two weeks later with the deposit and a signed contract.

**How do I handle currency differences for international clients?**

Never quote in their currency unless they insist. Quote in USD and add a clause: “Payments processed via Wise or PayPal; currency conversion fees are client responsibility.” I had a German client ask for EUR. I quoted $4,800 and said, “This is $4,800 USD. If you pay in EUR, the conversion rate is locked on the day of payment.” They paid immediately.


## Where to go from here

Pick one artifact from this guide—either the rate calculator or the client rate card—and deploy it within 48 hours. Send the rate card to your next three leads without waiting for a custom quote. Measure the responses: acceptance rate, negotiation pushback, and invoice disputes. After 30 days, run the rate audit and adjust your multipliers or flat fees. Repeat until your revenue per hour stabilizes above your floor. The goal isn’t to maximize rates; it’s to maximize clarity so you can sleep at night.


## Frequently Asked Questions

**How do freelancers set rates without experience?**

Start with your monthly expenses plus a 25% profit buffer. Divide by 120 billable hours. If your expenses are $3,200 and you want $2,000 profit, your floor is ($3,200 + $2,000) / 120 = $43/hour. Use this as a starter rate and raise it every three projects using the audit data. Many developers overprice early because they confuse confidence with competence. Track real hours, not gut feelings.


**What’s a fair rate for a junior developer with 6 months of experience?**

A fair junior rate is 60–70% of an established freelancer’s floor. If your floor is $79/hour, a junior should charge $47–$55/hour for simple tasks like bug fixes or static sites. Avoid underpricing just to “get experience.” Clients perceive low rates as low quality, and juniors end up doing unpaid revisions. Charge enough to respect your time, then upsell learning opportunities as paid mentorship blocks.


**How can I raise my rates without losing clients?**

Raise rates only on new projects. For existing retainer clients, grandfather them at the old rate for six months, then raise by 10–15% on renewal. Give 60 days notice and frame it as “added security features and priority support.” I raised a retainer from $2,400 to $2,800/month for a SaaS client. They accepted because the value increased, not because they were emotionally attached to the old price.


**What tools track freelance income and expenses accurately?**

Use Wave for invoicing and basic expense tracking. It’s free and integrates with bank accounts. For deeper analysis, export Wave data monthly into Google Sheets and run the rate audit. I tried QuickBooks for six months and abandoned it—too much overhead for simple freelancing. Wave gave me 80% of the insight with 20% of the effort.