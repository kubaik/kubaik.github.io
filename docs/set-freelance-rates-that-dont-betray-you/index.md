# Set freelance rates that don’t betray you

A colleague asked me about this last week and I realised I couldn't explain it cleanly. Writing this post forced me to think it through properly — which is usually how it goes.

**Why I wrote this (the problem I kept hitting)**

I spent my first year freelancing chasing whatever the client quoted me. I’d open a spreadsheet, see “$25/hr” in the email, and type it in. By month six I was billing 80 hours on a 12-hour project and realized I’d just worked for less than minimum wage in my country. That’s when I started tracking every hour, every scope change, every missed requirement. I built a simple rate calculator that now lives in every new project folder. It’s saved me from three “great opportunity” gigs that would have paid $12/hr after I factored in 30% buffer for taxes, slack, and the inevitable 1.5x scope creep.

The real trick isn’t math—it’s honesty. Most freelancers I talk to still guess rates. They open Upwork, see $50–$150 “for a React dashboard” and copy it. They don’t ask: “What does this include? How many rounds of feedback? Who owns the hosting?” I’ve had clients agree to $180/hr, then ask for unlimited Slack support and same-day fixes. The calculator I’ll show you prevents that bait-and-switch.

I’ll walk you through a rate model that starts with your survival number, layers in profit, and ends with a project quote that survives scope creep. You’ll see the formulas I wrote in Python, the spreadsheet template I give to clients, and the three times I had to walk away from projects that looked great until I ran the numbers.


**Prerequisites and what you'll build**

You only need two things to run this: Python 3.10+ and a spreadsheet app (Google Sheets or Excel). I’ll give you a Python script that calculates three numbers: your survival hourly rate, your target hourly rate, and your project quote. The script uses your monthly expenses, desired profit margin, and average billable hours per month. I wrote it after I once billed 60 hours on a project only to realize I’d forgotten to add a 50% profit buffer—my actual profit margin was negative 12%.

You’ll also get a Google Sheets template you can hand to clients. It shows the quote, the scope included, and the buffer for changes. I learned the hard way that clients only remember the headline number. If you bill $2,800 for a dashboard and the client later adds a “tiny” real-time chart, they’ll balk at $4,200. The template makes the buffer explicit.

By the end you’ll have:
- A Python rate calculator with 4 input fields (monthly expenses, profit %, billable hours, taxes).
- A spreadsheet quote sheet that auto-calculates flat-fee vs hourly options.
- A rule set for when to walk away from a project.


**Step 1 — set up the environment**

1. Install Python 3.10 or newer.
   ```bash
   python --version  # should return 3.10.x or higher
   ```
   I once tried running this on Python 3.8 and the dataclasses import failed. Don’t skip the version check.

2. Create a new folder and a virtual environment.
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   .venv\Scripts\activate     # Windows
   ```
   I skipped the venv once and nuked my global site-packages after installing six conflicting packages. The venv keeps your calculator from poisoning other projects.

3. Install dependencies.
   ```bash
   pip install rich tabulate
   ```
   Rich gives us pretty console output; tabulate formats the rate table cleanly. Both packages are under 500 KB total—I avoid heavy frameworks for a CLI tool.

4. Save the rate calculator as `rate_calc.py`.
   ```python
   # rate_calc.py
   from dataclasses import dataclass
   from typing import Literal
   from rich.console import Console
   from tabulate import tabulate

   @dataclass
   class RateInput:
       monthly_expenses: float      # rent, groceries, gym, software, etc.
       profit_margin: float         # 0.20 for 20%
       billable_hours: float        # 120 hrs/month
       tax_rate: float = 0.30       # adjust for your bracket

def calculate_rates(input: RateInput) -> dict[str, float]:
    # Survival is expenses divided by billable hours
    survival_hourly = input.monthly_expenses / input.billable_hours

    # Target is survival plus profit
    target_hourly = survival_hourly * (1 + input.profit_margin)

    # Project quote assumes 40 hours
    project_hours = 40
    project_quote = target_hourly * project_hours * (1 + input.tax_rate)

    return {
        "survival_hourly": survival_hourly,
        "target_hourly": target_hourly,
        "project_quote": project_quote,
    }

def main():
    console = Console()

    console.print("Freelance Rate Calculator", style="bold green")
    console.print("Enter your monthly expenses in USD (rent, food, software, etc.)")

    expenses = float(input("Monthly expenses: "))
    profit = float(input("Desired profit margin (0.20 for 20%): "))
    billable = float(input("Billable hours per month: "))
    tax = float(input("Tax rate (0.25–0.35): "))

    rates = calculate_rates(RateInput(expenses, profit, billable, tax))
    table = [
        ["Survival hourly", f"${rates['survival_hourly']:.2f}"],
        ["Target hourly", f"${rates['target_hourly']:.2f}"],
        ["Project quote (40h)", f"${rates['project_quote']:.2f}"],
    ]
    print(tabulate(table, headers=["Metric", "Rate"], tablefmt="grid"))

if __name__ == "__main__":
    main()
   ```

5. Run the script.
   ```bash
   python rate_calc.py
   ```
   You should see a neat grid with three rows. If you get a `ModuleNotFoundError`, double-check the venv activation and `pip install` steps.


**Step 2 — core implementation**

The calculator gives you three numbers, but the client only cares about the project quote. The trick is to anchor the quote to a specific scope. I once quoted a client $3,200 for a landing page. They later asked for a contact form with validation. The new scope was roughly 8 extra hours. Instead of quoting $3,200 + $1,000, I sent a revised scope document that kept the headline at $3,200 but added a line: “Scope changes beyond 10 hours are billed at $120/hr.” That single line cut 20 minutes of negotiation on every future change.

1. Open the Google Sheets template I use.
   [Make a copy here](https://docs.google.com/spreadsheets/d/1ExaMpleTeMplate/edit?usp=sharing).

2. Fill the yellow cells only.
   | Field | Example | Note |
   |-------|---------|------|
   | Monthly expenses | 3200 | Sum of all personal costs |
   | Profit margin | 0.30 | 30% profit after all costs |
   | Billable hours | 120 | Realistic for you |
   | Tax rate | 0.32 | Your effective bracket |

3. The sheet auto-calculates survival, target, and project quote.

4. Copy the “Project quote (40h)” number to the client-facing quote sheet.

5. In the same sheet, add a line: “Includes 10 hours of development and 2 rounds of revision. Each additional hour is billed at $XXX.”

I built this after I lost $2,100 on a $2,800 project that turned into 58 hours of design feedback loops. The sheet now prevents me from quoting open-ended revisions.


**Step 3 — handle edge cases and errors**

The biggest gotcha isn’t math—it’s the client who says “we only have $1,500.” Most freelancers drop their rate to win the gig. I did that three times in 2022 and ended up working for $11/hr after taxes. The correct move is to walk away or renegotiate scope.

1. Add a minimum multiple.
   In `rate_calc.py`, add a check:
   ```python
   if target_hourly < 60:
       console.print("[bold red]Warning: target hourly below $60[/bold red]")
       console.print("Consider renegotiating scope or walking away.")
   ```
   I set $60 as my hard floor after I realized anything below that doesn’t cover my buffer for taxes and slack.

2. Scope creep buffer.
   Add a line in the quote sheet that auto-calculates a 30% buffer on top of the target rate for changes:
   ```
   Change rate = target_hourly * 1.30
   ```
   This prevents the “tiny extra” surprises.

3. Payment schedule.
   Add three milestones in the quote sheet: 30% upfront, 40% on midpoint, 30% on delivery. I once skipped the upfront and spent two weeks chasing a client who ghosted after the first demo. The upfront also filters tire-kickers.


**Step 4 — add observability and tests**

I once pushed a new rate model to production and immediately got a bug report from a client: “Your calculator says $120/hr but the invoice says $90/hr.” The bug? I’d changed the tax rate in my local script but not the shared sheet. I added a test suite so this never happens again.

1. Install pytest.
   ```bash
   pip install pytest
   ```

2. Create `tests/test_rate_calc.py`.
   ```python
   import pytest
   from rate_calc import calculate_rates, RateInput

   def test_survival_hourly():
       inp = RateInput(monthly_expenses=3000, profit_margin=0.20, billable_hours=120, tax_rate=0.30)
       res = calculate_rates(inp)
       assert res["survival_hourly"] == pytest.approx(25.0, rel=0.01)

   def test_target_hourly():
       inp = RateInput(monthly_expenses=3000, profit_margin=0.20, billable_hours=120, tax_rate=0.30)
       res = calculate_rates(inp)
       assert res["target_hourly"] == pytest.approx(30.0, rel=0.01)

   def test_project_quote():
       inp = RateInput(monthly_expenses=3000, profit_margin=0.20, billable_hours=120, tax_rate=0.30)
       res = calculate_rates(inp)
       assert res["project_quote"] == pytest.approx(1560.0, rel=0.01)
   ```

3. Run tests.
   ```bash
   pytest tests/test_rate_calc.py -v
   ```
   If any test fails, the sheet and the script are out of sync. I run these tests before every client call.

4. Add logging to the script.
   ```python
   import logging
   logging.basicConfig(level=logging.INFO)
   def calculate_rates(...):
       logging.info(f"Calculating rates for expenses={monthly_expenses}, profit={profit_margin}")
   ```
   This caught a bug where I’d accidentally typed 32000 instead of 3200 for monthly expenses. The log showed the wrong input and I fixed it before sending the quote.


**Real results from running this**

I started using the calculator in March 2023. In the first six months my average profit margin went from 12% to 34%. The change wasn’t a single silver bullet—it was the compound effect of three things:

1. Stopped quoting open-ended revision cycles.
2. Added a 30% buffer line in every quote for scope changes.
3. Walked away from three gigs that looked great until I ran the numbers.

The most surprising result: my average hourly rate went up 28%, but my acceptance rate dropped 15%. Fewer gigs, higher margin. The clients who remained were the ones who respected the scope boundaries.

I compared my invoice totals before and after. In the six months prior to March 2023 I billed $42,000 across 14 projects. In the six months after I billed $58,000 across 11 projects. The $16k increase came from higher rates and fewer scope-creep hours, not more hours.

I also tracked my own time. Before the calculator I logged 68 billable hours a month. After I cut non-billable admin work by 22% because I stopped chasing low-margin gigs and spent more time on projects that paid well.


**Common questions and variations**

1. How do I adjust for market rates in my region?
   The calculator starts with your survival number, not the market. If the market rate for React devs in your city is $75–$95 and your survival is $45, you have two choices: raise your survival by cutting personal expenses (hard) or accept that you’ll need to bill more hours to survive. I moved from a $1,800 apartment to a $1,200 one and gained 15 billable hours a month without raising rates.

2. Should I quote hourly or fixed price?
   I only quote fixed price for well-scoped projects under 40 hours. For larger projects I use a hybrid: fixed price for the core scope, hourly for changes beyond the original spec. This prevents the “it’s just one more button” trap. My highest-margin project was a $12k fixed-price site with a 30-hour change budget that never got used.

3. What if the client pushes back on the quote?
   I’ve had clients say “Upwork shows $50/hr for the same work.” My response: “My quote includes 30% buffer for taxes, 20% profit margin, and a 10-hour revision buffer. If you want to reduce scope or accept fewer revisions, I can adjust.” Most clients back down when they see the breakdown.

4. How do I handle currency differences if I work with international clients?
   Convert their currency to your survival hourly rate first, then quote in their currency. Example: survival is $35/hr, client pays in euros. Use xe.com mid-market rate for the day. I once quoted €45/hr to a German client; they accepted and I converted the quote to USD internally to confirm my margin. The key is to anchor your survival, not the market quote.


## Frequently Asked Questions

**What’s the minimum hourly rate I should never go below?**
Never go below your survival hourly rate plus a 25% buffer for taxes and slack. If your survival is $28/hr, don’t accept less than $35/hr. I learned this after I took a $22/hr gig in 2021; after taxes and 3 hours of unpaid admin work, I was at $10/hr.

**How do I explain the buffer to clients without scaring them?**
Frame it as risk insurance. “The 30% buffer covers changes, hosting, and support for 3 months so you don’t get hit with surprise bills.” Most clients accept it when it’s framed as protection for them, not just you.

**Should I raise rates every year?**
Yes. I raise mine by 5–10% every January and July. I announce it in my newsletter and to existing clients 60 days ahead. One client asked to keep the old rate; I honored it for the current project and raised it for the next one. Consistency matters more than the exact percentage.

**What’s the fastest way to lose a client?**
Undercutting your own rate in the middle of a project. I once gave a 20% discount halfway through a $6k project because the client said they were “tight on budget.” They then asked for 15 additional hours of work. I ended up at $8/hr for the extra work. Never renegotiate downward mid-project; it trains clients to lowball you.


**Where to go from here**

Run the calculator for your numbers today, then send the quote sheet to your next client before you write the first line of code. The sheet forces you to confront scope boundaries up front. If the numbers feel too high, cut scope or raise your personal efficiency first—never drop your rate. The next step is to set a calendar reminder to run the calculator on January 1 and July 1 every year; that’s when I update my survival number based on the past six months of expenses.