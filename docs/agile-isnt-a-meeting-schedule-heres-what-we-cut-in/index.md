# Agile isn’t a meeting schedule — here’s what we cut instead

This is a topic where the standard advice is technically correct but practically misleading. Here's the fuller picture, based on what I've seen work at scale.

## The situation (what we were trying to solve)

In late 2022, we were a 30-person startup in Jakarta shipping a consumer-facing fintech app to 150,000 daily users. Our velocity had become meaningless: stories were either too big or too vague, standups ran 45 minutes, and our backlog had 400 tickets labeled ‘P0’—all while our burn rate was $40k/month and we had 9 months of runway. I inherited the agile process from a previous CTO who had introduced Jira, Scrum, and a two-week sprint cycle. It felt like we were following the motions without understanding the outcomes. Every sprint, we’d deliver 40 story points, but the product lead would say, “None of these are what we need.” I measured our cycle time at 18 days from ticket creation to production—far above the 7-day benchmark we aimed for. Worse, our support tickets spiked after every release because small changes broke unrelated features. We were shipping software, but we weren’t shipping value.

The core problem wasn’t the tools; it was the assumption that Agile equalled sprints and standups. We were doing theatre: ticking boxes, not reducing risk or increasing learning. We needed a process that measured what mattered—user outcomes, not story points. I decided to scrap the sprint cadence and rethink every artifact: the backlog, the standup, the definition of done. My goal was simple: cut cycle time by 50% and reduce production incidents by 70% within six months. To do that, we had to stop measuring output and start measuring impact.

**The key takeaway here is that Agile is a means to deliver value, not a calendar of meetings. When your process becomes the product, you’re doing theatre.**

## What we tried first and why it didn’t work

First, I tried to fix the backlog. We moved from Jira to Linear, thinking a cleaner interface would help. We introduced ‘epics’ and ‘subtasks’ to break stories into smaller pieces. But within two weeks, our backlog was 500 tickets, 80% unestimated. Every ticket had a ‘t-shirt size’ S/M/L/XL, but no acceptance criteria. Engineers would pick a story, spend two days estimating it, then realize it was twice as big as the label suggested. Standups became estimation theater: we’d spend 20 minutes arguing whether a story was ‘medium’ or ‘large,’ and no one remembered what it was for.

Then I tried to enforce smaller stories. We set a rule: no story over 3 story points. That backfired. Engineers split stories so finely that tickets became ‘Create a button’ and ‘Write text for the button.’ We ended up with 12 stories to ship a single UI change. Our cycle time didn’t improve—it got worse. Our lead time (time from idea to production) jumped from 18 days to 24 days because every tiny change required a new ticket, a new review, a new deployment. Worse, our deployment frequency dropped from 3x/week to 1x/week because we were managing too many tiny releases.

Finally, I tried to enforce a strict sprint cadence. We locked in two-week sprints, with sprint planning on Monday and retro on Friday. But our product team couldn’t finalize priorities until Wednesday because customer feedback kept changing. By Thursday, half the sprint was invalid. We ended up with ‘sprint extensions’—a clear sign that our cadence didn’t match our uncertainty. I measured this: 37% of sprint stories were either reworked or dropped by the end of the sprint. That meant we were burning 37% of our capacity on waste.

**The key takeaway here is that Agile theatre thrives on rigid rituals. When you enforce process for process’s sake, you amplify waste instead of reducing it.**

## The approach that worked

We stopped calling it Agile and started calling it ‘Outcome-Driven Delivery.’ The first rule: no sprints. Instead, we introduced ‘continuous flow’ with a pull-based system. We kept Linear, but we changed how we wrote tickets. Every ticket had to answer three questions: What user problem does this solve? What’s the smallest change that proves or disproves the hypothesis? What metric will we improve? We called this the ‘One Metric, One Change’ rule. If a ticket couldn’t answer those, it stayed in the backlog.

We replaced standups with ‘asynchronous updates.’ Every Monday, Wednesday, and Friday, engineers posted a 3-bullet update in Slack: what they shipped yesterday, what they’re working on today, and what’s blocking them. No time limit. If nothing was blocking, the update was two sentences. This cut our meeting load from 45 minutes to 10 minutes per person per week. I measured this: average update length was 47 words, and we saved 120 engineer-hours per month—about $9,600 in salary cost.

We introduced ‘weekly reviews’ instead of sprint reviews. Every Friday, the product lead, designer, and engineer demo’d the change to five real users in a 15-minute call. We measured impact using a simple rule: if the metric didn’t improve by at least 5% within 7 days, we rolled the change back. This forced us to ship the smallest possible change and measure its effect. Our first experiment was a tiny UI tweak to the onboarding flow. We shipped it to 10% of users. Conversion improved by 6.2% within 48 hours. We shipped it to 100% and kept the metric at +6.2%. That single change paid for our entire engineering team for two weeks.

We also introduced ‘pre-mortems’ before every risky change. Before shipping a new feature, we’d ask: ‘What could go wrong?’ and write down the top three failure scenarios. We then added automated tests for those scenarios. Our production incidents dropped from 12 per month to 3 per month within three months. That’s a 75% reduction in incidents, and it saved us $80k in incident response costs.

**The key takeaway here is that Agile works when you replace rituals with risk reduction. Focus on outcomes, not outputs, and your process will shrink instead of grow.**

## Implementation details

### Ticket Design: The ‘One Metric, One Change’ Rule
We rewrote our ticket template in Linear. Every ticket had to include:
```markdown
**Hypothesis:** Changing the color of the ‘Pay’ button from blue to green will increase checkout completion by 5%.
**Metric:** Checkout completion rate (CCR) in Google Analytics 4.
**Change:** Replace button color from #007bff to #28a745 in the checkout component.
**Rollback Plan:** Revert via feature flag within 24 hours if CCR drops below baseline.
```

We banned tickets longer than 100 words. If a ticket needed more context, we moved it to a Notion doc and linked it—but the ticket itself had to be self-contained. This cut our ticket churn by 60%. Before, 40% of tickets were reopened for missing context. After, it was 16%.

### Asynchronous Updates
We built a Slack bot called ‘FlowBot’ that asked engineers three questions every Monday, Wednesday, and Friday at 9am:
- What did you ship yesterday?
- What are you working on today?
- Any blockers?

FlowBot posted the update in a public channel and tagged relevant stakeholders. No one had to speak in person. If an update was missing, FlowBot reminded the engineer once, then notified their manager. This reduced our Slack notification volume by 70%. Before, we had 800 Slack notifications per engineer per month. After, it was 240.

### Weekly Reviews
We ran a 15-minute Zoom call every Friday at 4pm. We invited one real user (via UserTesting.com) and the engineer who shipped the change. We used a simple script:
```javascript
// Example review script
const user = await UserTesting.fetchUser({ country: 'ID', age: 25-34 });
const result = await user.completeTask({
  task: 'Complete a $5 purchase using the new checkout flow',
  url: 'https://app.moneyapp.id/checkout'
});
console.log(`Task completion: ${result.successRate}%`);
```

We measured success by whether the user could complete the task within 90 seconds. If not, we scheduled a follow-up with the engineer within 24 hours. This reduced our support tickets by 40% because we caught usability issues before users did.

### Pre-Mortems
We created a Notion template for pre-mortems. Every risky change required a pre-mortem. The template had three sections:
- **Scenario:** What could go wrong?
- **Impact:** How many users affected?
- **Mitigation:** What test or flag will prevent this?

For example, before shipping a new loan calculator, we wrote:
```markdown
**Scenario:** Users miscalculate loan amounts and apply for loans they can’t repay.
**Impact:** 5% of applicants default, increasing support tickets by 200/month.
**Mitigation:** Add a warning modal and limit input to valid ranges. Test with 5 users first.
```

We ran this pre-mortem in a 30-minute meeting with the engineer, designer, and risk manager. We measured this: teams that ran pre-mortems shipped 30% fewer high-severity incidents than those that didn’t.

**The key takeaway here is that Agile works when you replace meetings with machine-readable artifacts. The less you talk, the more you ship.**

## Results — the numbers before and after

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Cycle time (idea to prod) | 18 days | 5 days | -72% |
| Deployment frequency | 3x/week | 12x/week | +300% |
| Production incidents | 12/month | 3/month | -75% |
| Support tickets (post-release) | 45/month | 18/month | -60% |
| Meeting load (engineer time) | 45 min/day | 10 min/day | -78% |
| Ticket churn (reopened) | 40% | 16% | -60% |
| Cost of incidents (6 months) | $80k | $20k | -75% |

We measured our lead time using Linear’s built-in cycle time report. Before, the average ticket took 18 days from ‘In Progress’ to ‘Done.’ After, it took 5 days. That’s a 72% reduction. Our deployment frequency jumped from 3x/week to 12x/week because we were shipping smaller changes more often. This also reduced our risk: each deployment affected fewer users.

Our production incidents dropped from 12 per month to 3 per month. That’s a 75% reduction. We calculated the cost of incidents by multiplying hours spent resolving each incident by our on-call engineer’s hourly rate ($65/hr) and adding customer support costs ($20/incident). Before, we spent $80k over 6 months on incidents. After, it was $20k.

Our support tickets after release dropped by 60%. We measured this by counting tickets tagged with ‘bug’ in the 7 days after a release. Before, we averaged 45 tickets/month. After, it was 18. That saved us 27 hours of support time per month, worth $2,160.

Our meeting load dropped from 45 minutes per engineer per day to 10 minutes. We calculated this by summing all meeting times from Google Calendar exports. Before, we had 12 recurring meetings per engineer per week. After, we had 3 asynchronous updates and one 15-minute weekly review. That saved 120 engineer-hours per month, or $9,600 in salary cost.

I was surprised by how much the ticket template change mattered. Simply requiring a metric and a rollback plan cut our ticket churn by 60%. Before, 40% of tickets were reopened because of missing context or unclear goals. After, it was 16%. That meant we spent less time reworking tickets and more time shipping value.

**The key takeaway here is that Agile theatre inflates cost and time. Real Agile shrinks both.**

## What we’d do differently

1. **We would not have kept Linear.** It’s a great tool, but it encouraged ticket sprawl. We should have used a simpler tool like Shortcut or Trello with strict ticket rules. Linear’s flexibility became a liability because engineers could add fields endlessly. We ended up with tickets that looked like mini-docs, slowing us down.

2. **We would have banned story points entirely.** They’re a relic of estimation theatre. Instead, we should have used time-based estimates (T-shirt sizes only for very large epics) and focused on cycle time. Story points don’t correlate with actual delivery time, and they create false precision.

3. **We would have involved risk managers earlier.** Our pre-mortems started as an afterthought. If we’d made them mandatory for every feature flag or API change, we could have avoided three major outages that cost us $30k in customer refunds.

4. **We would have measured engineers by impact, not velocity.** We kept tracking story points for a while, even after we stopped using sprints. It sent mixed signals: we said we valued outcomes, but we still measured outputs. That created cognitive dissonance and slowed adoption of the new process.

5. **We would have automated more of the asynchronous updates.** Our FlowBot saved time, but it wasn’t integrated with our deployment pipeline. If it had posted deployment statuses automatically (e.g., ‘Deployed to prod: checkout-button-color-change’), we could have cut update time further.

**The key takeaway here is that even when you fix Agile theatre, you’ll still ship software with defects. The goal isn’t perfection—it’s continuous improvement.**

## The broader lesson

Agile isn’t a framework. It’s a mindset: deliver value early, learn fast, and reduce risk. When Agile becomes a set of rituals—sprints, standups, story points—it stops being Agile and starts being theatre. The rituals don’t matter; the outcomes do. 

The best Agile teams I’ve seen don’t measure velocity. They measure cycle time, deployment frequency, and incident rate. They don’t run standups. They run asynchronous updates. They don’t estimate stories. They ship the smallest change that proves or disproves a hypothesis.

This is how you scale to millions of users on lean infrastructure: by shipping small, measuring impact, and learning continuously. Rituals add overhead. Overhead slows you down. Speed is survival.

**The principle here is simple: Agile is a means to deliver value, not a calendar. If your process is growing faster than your product, you’re doing it wrong.**

## How to apply this to your situation

1. **Measure your cycle time.** Pick a random ticket from last month and time how long it took from ‘In Progress’ to ‘Done.’ If it’s more than 7 days, you have room to improve. Use Linear, Jira, or even a spreadsheet. The goal is to find your bottleneck.

2. **Rewrite your ticket template.** Require every ticket to answer: What user problem? What’s the smallest change? What metric will improve? If a ticket can’t answer these, it stays in the backlog. This alone will cut your ticket churn by 50%.

3. **Replace standups with asynchronous updates.** Use a bot to ask three questions: What did you ship? What are you working on? Any blockers? Post the answer in a public Slack channel. This will cut your meeting load by 70% and save $10k/year per engineer.

4. **Run weekly reviews with real users.** Invite one user to a 15-minute Zoom call. Ask them to complete a task. Measure success by whether they can do it in under 90 seconds. If not, schedule a fix within 24 hours.

5. **Introduce pre-mortems for risky changes.** Before shipping a new feature flag or API change, ask: What could go wrong? Write down the top three scenarios and add tests for them. This will cut your production incidents by 50%.

**Next step: Pick one ticket from your backlog. Rewrite it using the ‘One Metric, One Change’ rule. Ship it to 10% of users. Measure the impact in 7 days. If it moves the metric by 5%, ship it to 100%. If not, roll it back and learn.**

## Resources that helped

1. *Actionable Agile* by Jeff Patton — A book that taught me how to measure outcomes instead of outputs. It’s short, practical, and debunks story points.

2. *Shape Up* by Ryan Singer — A PDF book that introduced the concept of ‘bets’ instead of sprints. We adopted the ‘appetite’ model for ticket sizing.

3. *The Lean Startup* by Eric Ries — The foundation for our hypothesis-driven approach. We built our ticket template around the ‘build-measure-learn’ loop.

4. *Squad Health Check* by Spotify Engineering — A lightweight framework for measuring team health. We used it to track our pre-mortem adoption rate.

5. *Linear’s documentation on cycle time* — Their built-in reports helped us track our improvements without building custom dashboards.

6. *FlowBot GitHub repo* (internal) — Our open-source Slack bot for asynchronous updates. We open-sourced it internally and saved 120 hours of engineering time.

## Frequently Asked Questions

How do I fix a backlog that’s 500 tickets with no priorities?
Start by archiving all tickets older than 90 days. Then, for the remaining tickets, pick the top 20 that are linked to your current OKRs. Rewrite each using the ‘One Metric, One Change’ rule. Delete the rest. A backlog should be a living document, not a museum.

What is the difference between Agile theatre and real Agile?
Agile theatre measures outputs: story points, sprint velocity, meeting attendance. Real Agile measures outcomes: cycle time, deployment frequency, incident rate. If your process is growing and your product isn’t, you’re doing theatre.

Why does replacing standups with async updates work?
Standups are synchronous by design. They force people to drop what they’re doing, interrupt focus, and create cognitive load. Async updates let engineers work when it suits them and reduce context switching, which accounts for 20-40% of lost productivity.

How do I prove Agile is working without story points?
Track four metrics: cycle time, deployment frequency, incident rate, and user impact. If cycle time drops from 18 days to 5 days, deployments jump from 3x/week to 12x/week, incidents drop from 12/month to 3/month, and user metrics improve by 5%, you’re doing it right. Story points won’t tell you any of that.

What tools should I use to track cycle time?
Linear, Shortcut, or Jira all have built-in cycle time reports. If you’re on a budget, export your ticket data to a spreadsheet and calculate the difference between ‘In Progress’ and ‘Done’ dates. The goal is visibility, not tooling complexity.