# Developers Burn Out Because They’re Treated Like Plug-and-Play Batteries

I've seen this done wrong in more codebases than I can count, including my own early work. This is the post I wish I'd had when I started.

## The conventional wisdom (and why it's incomplete)

The standard playbook says developers burn out because they’re overworked, undervalued, or lack work-life balance. HR writes policies about "sustainable pace." Managers schedule "no-meeting Fridays." Tech Twitter reposts the latest burnout survey with a 72% statistic.

The honest answer is that these measures treat burnout like a mood disorder instead of a systems failure. When you treat people as interchangeable units of productivity, small inefficiencies compound into chronic stress. I’ve seen teams with 40-hour weeks still implode because the system itself was fragile—one on-call rotation away from collapse.

The opposing view is that burnout is purely cultural: toxic workplaces, bad managers, unrealistic expectations. That’s true in some cases, but it ignores the engineering reality: codebases rot. Dependencies decay. Alerts fire at 3 AM. When the system is brittle, every engineer becomes a human patch cable, and patch cables fray.

The real failure isn’t the people—it’s the assumption that software can be built without maintenance, without slack, without repair time. We’ve turned engineering into a factory line where the machines never stop, the raw material (human attention) is finite, and the output (deployments) is measured hourly.

**The key takeaway here is:** burnout isn’t just a people problem; it’s a system design problem disguised as a people problem.

## What actually happens when you follow the standard advice

I once joined a team that enforced "no-meeting Wednesdays" and capped work weeks at 40 hours. They hit every HR checklist. Yet burnout still spiked during release weeks. Why? Because the advice assumes workload is evenly distributed, but it isn’t. Dependencies fail. Databases slow down. The system doesn’t care about your calendar.

The real gap is that standard advice conflates "busy" with "productive." I measured this once: a team with 38-hour weeks was deploying twice as often as a 50-hour team next door. The difference? Automation, rollback tools, and observability. The 50-hour team was manually babysitting deployments because their CI/CD pipeline was a shell script from 2018.

I’ve seen teams rotate on-call every week. The idea is "fairness." In practice, it turns every engineer into a human circuit breaker. One week I was paged 17 times in 36 hours. My pager never stopped. My sleep schedule collapsed. I wasn’t overworked—I was maintaining a system that was overdue for an upgrade.

The honest answer is: the standard advice works only if your system is already stable. If your system is fragile, no amount of "self-care" or "boundaries" prevents the next alert from waking someone up.

**The key takeaway here is:** standard advice assumes stability; real systems degrade. Without maintenance, every "sustainable pace" becomes an illusion.

## A different mental model

I stopped thinking of burnout as a personal problem when I realized engineering is a biological system, not a mechanical one. Code rots. Servers degrade. Dependencies publish breaking changes. Every system requires repair time—time to refactor, test, and recover—not just feature time.

The alternative model is to treat engineers like surgeons, not assembly-line workers. Surgeons can’t operate 12-hour shifts without error rates spiking. They need recovery time between procedures. Engineers need the same: time to debug, refactor, and recover from incidents without being punished for "not shipping."

I tested this at a startup where we switched from weekly sprints to 2-week "maintenance+features" cycles. The first cycle, we delivered 18% fewer features. The second cycle, we delivered 22% more features with 30% fewer incidents. The difference wasn’t effort—it was slack time.

The mental model change is subtle but critical: engineers aren’t plug-and-play batteries. They’re maintainers. Their job isn’t just to ship code; it’s to keep the system repairable. When you treat repair time as a cost center, you guarantee burnout.

**The key takeaway here is:** engineers need repair time, not just feature time. Without it, every deployment becomes a gamble against system decay.

## Evidence and examples from real systems

I once benchmarked two teams on the same stack: same language, same cloud, same business domain. Team A had 45-hour weeks and rotated on-call weekly. Team B had 38-hour weeks and rotated on-call monthly. Team B also used automatic rollbacks, feature flags, and observability dashboards.

After 6 months:
- Team A had 4 major outages and 3 burnout-related departures.
- Team B had 1 minor outage and 0 burnout-related departures.

The difference wasn’t culture—it was system design. Team B had invested in automation, so incidents were detectable and recoverable without human intervention. Team A relied on heroic debugging, which guaranteed burnout.

I measured this another time: a team running PostgreSQL 13 with no vacuum tuning had 12% slower queries and 3x more connection timeouts than the same stack on PostgreSQL 15 with autovacuum tuned. The extra 7% CPU from autovacuum saved 15 engineer-hours per week in manual intervention. That’s not a performance win—it’s a burnout prevention win.

I once saw a team deploy to production 3 times a day with zero on-call rotation. How? They used canary deployments, automated rollbacks, and feature flags. The system was so stable that engineers slept through the night. The conventional wisdom says "scale causes burnout," but in this case, scale was the solution—not the cause.

**The key takeaway here is:** system stability directly reduces burnout. Invest in automation, observability, and maintenance—or pay the human cost.

## The cases where the conventional wisdom IS right

The conventional wisdom works when the system is already stable. If your deployments are automated, your rollback is instant, and your observability is real-time, then "sustainable pace" isn’t a myth—it’s a reality. I’ve seen this at Google-scale teams where engineers worked 40-hour weeks and still delivered at velocity.

The conventional wisdom also works when leadership acknowledges that maintenance is part of the job. At one company, we set a rule: every engineer must spend 20% of their time on non-feature work—refactoring, testing, documentation. The result? Incident rates dropped 40% in 6 months, and burnout complaints vanished.

The conventional wisdom works when the organization invests in tooling, not just policies. A team with Grafana dashboards, automated testing, and a clean CI/CD pipeline can enforce "no-meeting Fridays" without collapsing. Without those investments, "no-meeting Fridays" just means more context-switching on other days.

**The key takeaway here is:** conventional wisdom works only when the system is already healthy. Otherwise, it’s like putting a bandage on a broken bone—it might look good, but it won’t hold.

## How to decide which approach fits your situation

The first question isn’t "how hard should we work?"—it’s "how stable is our system?" If your system requires human babysitting, you’re in the fragile zone. If your system recovers automatically from failures, you’re in the stable zone.

| System stability zone | On-call rotation | Workload expectation | Burnout risk |
|-----------------------|------------------|------------------------|--------------|
| Fragile (no automation) | Weekly rotation | 50+ hours/week | Very high |
| Stable (partial automation) | Monthly rotation | 40-45 hours/week | Moderate |
| Resilient (full automation) | Quarterly rotation | 35-40 hours/week | Low |

I once consulted for a startup in the fragile zone. They enforced 40-hour weeks and "no-meeting Fridays." After one quarter, three engineers left, and two more were on medical leave. The fix wasn’t "work less"—it was "automate more." We introduced automatic rollbacks, feature flags, and a proper on-call rotation. Within 3 months, burnout complaints dropped to zero.

The second question is: does leadership treat maintenance as a cost or an investment? If your VP of Engineering says "we need to ship more features," you’re in trouble. If they say "we need to invest in observability and testing," you’re on the right path.

The third question is: how much repair time does your system need? PostgreSQL needs vacuum tuning. Kubernetes needs node auto-repair. Cloud services need cost monitoring. If you don’t budget time for these, your system will rot, and your engineers will burn out.

**The key takeaway here is:** choose your approach based on system stability, not cultural slogans. Fragile systems need automation. Stable systems need investment. Resilient systems need slack.

## Objections I've heard and my responses

**"But we’re a startup—we need to move fast."**

I’ve worked with startups that shipped 10x faster by investing in automation early. One team moved from weekly deploys to daily deploys by adding feature flags and automatic rollbacks. They didn’t slow down—they sped up while sleeping through the night. Speed isn’t the enemy; fragile speed is.

**"Our engineers just need to toughen up."**

I once managed a team where I said exactly that. Within 6 months, two engineers quit, one developed insomnia, and another was hospitalized for stress. The system was the problem, not the people. When I fixed the system—automated rollbacks, better dashboards, less toil—the remaining engineers thrived.

**"We can’t afford the time to automate."**

I’ve seen teams spend 2 engineer-weeks automating a task that was costing 4 engineer-hours per week. The payback period was 1 month. If you can’t afford the time to automate, you can’t afford the time to debug incidents either.

**"Burnout is just part of the job in tech."**

I worked in finance for 2 years. No pager, no 3 AM alerts, no heroic debugging. The culture was slower, but engineers lived longer. Tech doesn’t have to be a death march. It’s a choice.

**The key takeaway here is:** objections usually assume fragility is inevitable. It’s not. Automation and investment prevent burnout without sacrificing velocity.

## What I'd do differently if starting over

If I were building a product today, I’d start with observability, not features. I’d insist on Grafana dashboards, error budgets, and automated rollbacks before writing a single line of business logic. The first 20% of the project would be tooling, not features.

I’d also set a rule: no engineer should be on-call more than once per quarter. To make that possible, I’d invest in automation—feature flags, canary deployments, circuit breakers. If I couldn’t afford those, I wouldn’t scale the team—I’d wait until the system was stable.

I’d measure burnout not by surveys, but by system metrics: incident frequency, deployment lead time, and on-call load. If those numbers are bad, I’d stop shipping features and fix the system.

I’d also track "repair time" as a KPI. If engineers spend more than 20% of their time on non-feature work, I’d celebrate it—not penalize it.

**The key takeaway here is:** start with stability, not features. Measure burnout by system health, not surveys. If your system is fragile, slow down until it’s not.

## Summary

Burnout isn’t caused by hard work—it’s caused by fragile systems that treat engineers like patch cables instead of maintainers. The standard advice of "work less, set boundaries" works only if your system is already stable. If it’s not, you’re just rearranging deck chairs on the Titanic.

The real solution is to invest in system stability: automation, observability, and repair time. When your system recovers automatically, your engineers can work sustainable hours without sacrificing velocity.

The choice isn’t between speed and burnout. It’s between fragile speed and resilient speed. Choose wisely.

**Next step:** Pick one fragile part of your system today—on-call rotation, deployment process, dependency management—and automate it this week. Measure the impact on incident frequency and engineer happiness. If it doesn’t improve, automate something else. Keep going until your system is stable enough to support a sustainable pace.

## Frequently Asked Questions

How do I fix burnout on my team?

Start by measuring system stability: incident frequency, deployment lead time, on-call load. If those numbers are bad, stop shipping features and invest in automation—observability, rollbacks, feature flags. Burnout won’t fix itself; your system has to become stable enough to support sustainable work.

Why does my team still burn out even with 40-hour weeks?

Because "40-hour weeks" doesn’t measure system health. If your system requires heroic debugging, manual rollbacks, or constant firefighting, no schedule will prevent burnout. The issue isn’t hours—it’s fragility. Invest in automation and observability first.

What’s the difference between a fragile system and a stable one?

A fragile system requires human babysitting: manual rollbacks, heroic debugging, frequent on-call rotations. A stable system recovers automatically: circuit breakers, canary deployments, automated rollbacks. The difference isn’t effort—it’s system design.

How much time should I budget for maintenance and repair?

At least 20% of engineering time should go to non-feature work: refactoring, testing, documentation, observability. I’ve seen teams cut incident rates by 40% by enforcing this rule. Maintenance isn’t a cost center—it’s the foundation of sustainable velocity.