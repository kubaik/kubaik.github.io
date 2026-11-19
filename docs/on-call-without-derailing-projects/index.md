# On-call without derailing projects

On-call rotations and project work are two jobs with the same eight hours. The on-call engineer is expected to triage alerts, write postmortems, and still deliver the feature they committed to. When the pager goes off at 2am, the next day's project work doesn't disappear — it just gets compressed. The result is a slow erosion of delivery pace that looks like a staffing problem but is really a scheduling problem. This post is about the mechanics of separating the two: how to define on-call boundaries, how to make alerting precise enough that on-call is quiet, and how to hand off work so that neither the project nor the rotation absorbs the other's cost. The concrete problem: teams treat on-call as background noise and then wonder why sprint velocity collapses every time someone joins the rotation.

## The real cost is context switching, not the pages themselves

Most teams measure on-call by page volume. A quiet week is assumed to be a good week. But the expensive part of on-call is not the 3am page — it's the low-grade background load of knowing you might be paged, plus the morning triage that fragments focus. A single interruption during a deep-work block can cost more than the interruption itself because re-establishing context (reading the stack trace, remembering the design decision, finding the right file) takes time that isn't tracked anywhere.

The pattern that causes the most damage is what I'd call the *absorbed rotation*: the on-call engineer is nominally on rotation but also expected to attend all project standups, review PRs, and make design decisions. The rotation is real but invisible in planning. When an incident lands, the project absorbs the cost silently, and the team learns to under-commit rather than fix the structure.

The fix is structural: on-call must be a *visible* allocation of capacity, not an assumption that the engineer will "just handle it." That means the rotation needs a defined scope, a quiet baseline, and a handoff protocol that the whole team follows — not just the person holding the pager.

## Prerequisites: what you need before changing the rotation

You can't fix on-call scheduling without three things in place. Skip any of these and the changes won't stick.

1. **An alert inventory with ownership.** Every alert that can page a human needs a named owning service and a documented runbook. If you don't have this, start by exporting your alert rules and listing which ones have fired in the last 30 days. Alerts that never fire are candidates for deletion; alerts that fire often without action are candidates for demotion to a dashboard.

2. **A rotation tool that supports overrides and escalation.** Most incident tools (PagerDuty, Opsgenie, Grafana OnCall, and the open-source Alertmanager with a routing tree) support schedule overrides and escalation policies. If you're still using a shared phone and a spreadsheet, the rest of this post will be hard to apply. The minimum viable setup is a schedule with a primary and secondary, an escalation policy that fires after a documented timeout, and a way to override for planned absence.

3. **A definition of "project work" that excludes on-call capacity.** This sounds obvious, but most sprint plans assume full availability. If your team runs two-week sprints and one engineer is on-call for one of those weeks, the plan should reflect roughly half capacity for that person — not because they'll be paged constantly, but because the background load and triage time are real. The exact fraction depends on your alert volume; the point is to make it explicit rather than discovered at retro.

## Step 1: Define the on-call contract in writing

The first implementation step is a one-page document that states what on-call covers and what it doesn't. This is not bureaucracy — it's the thing you point to when someone asks the on-call engineer to join a design review during their rotation week.

A useful contract has four clauses:

- **Scope:** which services and which severity levels page the on-call engineer. Everything else goes to a ticket or a dashboard.
- **Response expectations:** what "acknowledge" means (e.g., within 5 minutes during business hours, 15 minutes out of hours) and what "resolve" means (e.g., mitigate within 1 hour, root-cause within the next business day). These are targets, not guarantees; document them so the team can measure against them.
- **Exclusions:** project work that is explicitly paused during rotation. Typically this is deep-work tasks — feature development, large refactors — while code review and small bug fixes remain allowed because they're interruptible.
- **Handoff:** when and how the rotation transfers, including what the outgoing engineer must write down.

The contract should be reviewed at every retro. If it's not being followed, the retro is where you find out why.

A common trap: teams write the contract but don't enforce the exclusions. The on-call engineer still attends every meeting because "it's just one meeting." The contract only works if the team treats exclusions as real — which means the project lead has to plan around them, not the on-call engineer.

## Step 2: Make alerting precise enough that on-call is quiet

The single biggest lever for balancing on-call with project work is reducing the number of pages. A rotation that pages twice a week is a different job from one that pages twice a month. The goal is not zero pages — some alerts are real — but every page should be actionable and every non-actionable alert should be removed or demoted.

A worked example using Prometheus alerting rules (the same logic applies to any metrics-based alerting system):

```yaml
# Before: pages on any 5xx rate above 1% for 1 minute.
# This fires on deploy blips, health-check noise, and single-client retry storms.
- alert: HighErrorRate
  expr: rate(http_requests_total{status=~"5.."}[1m]) / rate(http_requests_total[1m]) > 0.01
  for: 1m
  labels:
    severity: page

# After: pages only when the error rate is sustained and affects a meaningful
# share of traffic, with a longer window to ride out deploys.
- alert: HighErrorRate
  expr: |
    (
      sum(rate(http_requests_total{status=~"5.."}[5m]))
      /
      sum(rate(http_requests_total[5m]))
    ) > 0.05
    and
    sum(rate(http_requests_total[5m])) > 1
  for: 10m
  labels:
    severity: page
  annotations:
    runbook_url: https://runbooks.example.com/high-error-rate
```

The changes: a 5-minute window instead of 1 minute, a 10-minute `for` duration, a higher threshold (5% instead of 1%), and a minimum traffic floor so low-traffic services don't page on a single error. The runbook link is not optional — an alert without a runbook is a page that requires the engineer to rediscover the system at 3am.

This is the kind of change that teams commonly resist because it feels like "ignoring errors." The counter-argument: an alert that fires on deploy blips trains engineers to ignore pages, which is worse than not alerting at all. The 5% threshold and 10-minute window are illustrative — the right values depend on your traffic shape and error budget. The principle is to alert on sustained, user-impacting conditions, not on instantaneous noise.

After changing thresholds, run a review: for each alert, ask "what action does this page require?" If the answer is "look at the dashboard and close it," demote it to a dashboard annotation or a ticket. If the answer is "restart the service," the alert is a symptom of a missing automation — file a ticket to fix the root cause and keep the alert until then.

## Step 3: Structure the rotation so project work has a home

Once the contract exists and alert volume is under control, the rotation itself needs structure. The goal is to make on-call a predictable, bounded role rather than an ambient state.

**Use weekly rotations with a fixed handoff time.** Daily rotations sound humane but they create a handoff every day, and each handoff loses context. Weekly rotations (Monday morning handoff) give the on-call engineer enough time to learn the current incident state and enough continuity to handle multi-day issues. For teams with very high page volume, a follow-the-sun model with regional rotations is better than forcing one timezone to absorb nights.

**Pair the rotation with a secondary.** The secondary is not a backup for vacation — it's the first escalation target when the primary doesn't acknowledge within the response window. This matters for project work because it means the primary can be in a meeting or on a train without the pager becoming a crisis. The secondary should be someone not on the critical path for the current project milestone, if possible.

**Schedule project work around the rotation, not through it.** The project lead should know who is on-call and plan interruptible work for that week. This is a planning discipline, not a tool: in the sprint plan, mark the on-call engineer's capacity as reduced and assign them tasks that can be paused (code review, documentation, small bug fixes) rather than tasks that require long uninterrupted blocks (complex feature work, migrations).

A common failure mode: the on-call engineer is assigned a large feature and told to "just pause it if you get paged." Pausing a complex feature for an incident and resuming it the next day is expensive — the context reload alone can eat an hour. Better to not assign that work in the first place.

## Step 4: Write a handoff that actually transfers context

The handoff is where most rotations leak. The outgoing engineer knows about three ongoing issues, a flaky alert, and a deploy that's half-rolled-out. If none of that is written down, the incoming engineer starts from zero and the first page becomes a discovery exercise.

A useful handoff has five fields, and it should take ten minutes to write:

1. **Open incidents and their state:** what's broken, what's been tried, what's next.
2. **Recent changes:** deploys, config changes, or feature flags that went out during the rotation and might be related to future pages.
3. **Noisy alerts:** which alerts fired and whether they were actionable. This feeds the alert-review loop from Step 2.
4. **Upcoming risks:** planned deploys, traffic events, or maintenance windows in the next week.
5. **Project status for the on-call engineer:** what work was paused and where to pick it up.

The fifth field is the one teams skip, and it's the one that connects on-call to project work. If the outgoing engineer was mid-task when their rotation ended, the incoming engineer doesn't need to continue that task — but the project lead does need to know it was paused so it can be rescheduled.

## Edge cases and errors

**The alert that pages but has no owner.** This happens when a service is deprecated but its alerts remain. The fix is an ownership audit: every alert rule must have a `team` label or an entry in an ownership file. Alerts without an owner should be deleted, not reassigned to the on-call engineer by default.

**The on-call engineer who is also the only person who can deploy.** If the deployment process requires a specific person, the rotation is a single point of failure. The fix is to document the deploy process and grant deploy access to the secondary as well. This is a prerequisite for the rotation to function, not a nice-to-have.

**The project deadline that "can't slip."** When a deadline is fixed and an engineer is on-call, the team is choosing to accept risk. The right move is to make the risk explicit: either reduce scope for that week or bring in someone else to cover the rotation. Silently expecting the on-call engineer to absorb both is how burnout happens.

**The alert that fires only during business hours.** Some alerts are noise during the day but meaningful at night (e.g., a batch job that fails). Use time-based routing in your alerting tool to send these to a ticket during business hours and to the pager out of hours, or suppress them entirely if they're not user-impacting.

## Verification: how to know the balance is working

You can't verify "balance" directly, but you can measure the inputs. Track these four metrics over a month:

- **Pages per rotation:** the count of pages that required action. A downward trend after threshold changes is the signal that Step 2 worked.
- **Acknowledged-within-target rate:** the share of pages acknowledged within the response window. If this drops, the rotation is under-staffed or the response targets are unrealistic.
- **Paused project tasks per rotation:** the count of tasks that were started and paused due to a page. This should be low; if it's high, the project plan isn't respecting the rotation.
- **Handoff completeness:** a spot-check that each handoff has the five fields. This is qualitative, but it's the leading indicator for whether the next rotation starts from a known state.

A simple way to collect the first two is to export the incident history from your on-call tool and count. The third requires the project lead to note pauses in the sprint board — a label like `paused-oncall` is enough.

## Frequently Asked Questions

**How do you balance on-call with a full-time project?**
Treat on-call as a capacity allocation, not an add-on. Reduce the on-call engineer's project commitments for the rotation week, assign interruptible work, and make the reduction visible in the sprint plan. The exact reduction depends on page volume, but the principle is that the plan should reflect the rotation rather than discovering it at retro.

**What is a reasonable on-call rotation length?**
Weekly rotations are common because they balance continuity against fatigue. Daily rotations create frequent handoffs and context loss; monthly rotations concentrate fatigue and make it hard to plan project work around them. The right length depends on page volume and team size, but a week is a reasonable default for most teams.

**How many pages per shift is too many?**
There's no universal number, but a useful heuristic is that if the on-call engineer can't complete a focused task during a shift, the rotation is too noisy. Some teams target fewer than one actionable page per shift; others accept more during incidents. The key is to review pages monthly and demote or fix any alert that doesn't require action.

**Should on-call engineers attend project standups?**
Attendance is fine if it's short and interruptible, but the on-call engineer should not be the decision-maker for project blockers during their rotation. The project lead should have a backup decision path so that a page doesn't stall the project. If standups are long or require deep context, consider excusing the on-call engineer during their rotation week.

## Next step: audit your noisiest alert today

Open your alerting configuration and find the alert that fired most often in the last 30 days. For that alert, check three things: does it have a runbook, does it require an action other than "look and close," and what is its `for` duration? If it fails any of these, change one parameter — raise the threshold, extend the `for` window, or demote it to a dashboard — and note the change in your team's alert-review doc. That single edit is the first step toward a rotation that leaves room for project work.

## Official documentation

- [Prometheus documentation](https://prometheus.io/docs/)
- [NIST SP 800-61: Computer Security Incident Handling Guide](https://csrc.nist.gov/pubs/sp/800/61/r2/final)
- [Google SRE Book: Monitoring Distributed Systems](https://sre.google/sre-book/monitoring-distributed-systems/)

---

### About this article

**Written by:** [Kubai Kevin](/about/) — software developer based in Nairobi, Kenya.

**How this article was produced:** This site uses an automated LLM pipeline designed and maintained by the author. Before publication, drafts are screened automatically for minimum length, near-duplicate content, invented or unsourced claims and citations, unverifiable product names, and broken internal links. Individual line-by-line human editing is not performed on every post. Figures, benchmarks and scenarios are illustrative unless a source is linked; verify them against official documentation before relying on them in production. See the [AI content policy](/ai-content-policy/).

**Corrections:** Report errors via the [contact page](/contact/). Corrections are applied promptly.

**Last generated:** November 2026
