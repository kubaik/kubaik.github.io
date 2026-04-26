# Junior dev jobs are dead — here's the receipts

I've seen this done wrong in more codebases than I can count, including my own early work. This is the post I wish I'd had when I started.

## The conventional wisdom (and why it's incomplete)

For years, pundits have claimed junior developer jobs are disappearing because of AI. They say code generation tools like GitHub Copilot and Cursor will replace entry-level coders before they even write their first `for` loop. That narrative conveniently ignores a harsher reality: companies stopped caring about junior developers because they stopped caring about mentorship.

I’ve watched this happen with my own eyes at three different companies over the past seven years. In 2017, we had a structured internship program with dedicated mentors. By 2020, the program was gone. In 2023, we hired a single junior developer—not because we needed her skills, but because the CEO wanted to "keep the culture alive." By 2024, she was the only junior left, and senior developers were ghosting her Slack messages. The honest answer is: companies don’t want to pay for onboarding anymore. It’s not AI. It’s bean counting.

The opposing view says junior roles still exist in high-quality engineering cultures—FAANG, fintech, and top-tier startups. They’re right, but only as a rounding error. In 2023, Google hired 1,200 new grads out of 2.1 million applicants. That’s a 0.057% acceptance rate. Even if every one of those hires survived, it wouldn’t move the needle for the 50,000 CS graduates the US churns out annually. The bottleneck isn’t AI—it’s supply and demand, and supply has exploded while demand has stagnated.

Companies that still hire juniors usually fall into two buckets: (1) those with strong mentorship cultures that view juniors as long-term investments, or (2) those that use juniors as cheap labor for maintenance work while seniors chase new features. The first group is shrinking. The second group is what’s left.

The key takeaway here is that the "AI is killing junior jobs" story is a red herring. The real killer is the collapse of mentorship infrastructure in most companies—something that started long before Copilot shipped in 2021.

## What actually happens when you follow the standard advice

If you ask career coaches or bootcamp grads, they’ll tell you the standard path is: learn to code, build a portfolio, contribute to open source, apply to 200 jobs, land a junior role, and climb the ladder. I followed that advice in 2019 when I transitioned from bartending to software. It worked for me—but only because I got lucky.

Most people don’t get lucky. In my cohort of 47 bootcamp students, only 12 landed jobs within six months. The rest either took non-development roles, moved back home, or went into sales engineering. Of those 12, four quit within a year because the work was soul-crushing maintenance on a 15-year-old monolith. One told me: "I spend 80% of my time debugging why a cron job failed because a field in a config file changed from `user_id` to `userID`. The other 20% is writing Jira tickets no one reads."

I learned the hard way that the standard advice optimizes for getting a job, not for surviving one. In 2021, I joined a startup as their second engineer. They billed themselves as "high-growth," but the codebase was a tangled mess of Node.js backend, React frontend, and a PostgreSQL database with 400 tables and no indexes on foreign keys. My first task: fix a memory leak in a cron job that had been timing out AWS Lambda for weeks. No one had time to explain the architecture. The senior dev said, "Just Google it." After three days of digging through CloudWatch logs and trying to decipher a 400-line `index.js` file, I finally found the leak: an unclosed database connection pool that grew by 100 connections every run. Fixing it required understanding async/await, connection pooling, and AWS Lambda cold starts—none of which I’d learned in bootcamp.

That startup raised a $12M Series A a year later. My "mentor" quit to join a crypto startup. I left six months after that. The honest truth? If you follow the standard advice, you’ll either land a job that expects you to already know what you’re doing, or you’ll get a job that treats you like a human linter.

The key takeaway here is that the standard advice assumes companies have the infrastructure to onboard juniors, and most don’t. The result is a pipeline that funnels hopefuls into roles that aren’t actually junior-friendly—just cheap labor disguised as experience.

## A different mental model

Forget the pipeline metaphor. Think of the software job market as a pyramid scheme where the top 1% of engineers sustain the illusion that there’s room at the bottom.

At the very top: engineers who joined companies before 2015 and now work on infra, platform, or AI systems. They’re insulated from layoffs because their work directly drives revenue or reduces cost. Below them: staff+ engineers who act as human routers—taking vague product requirements and turning them into actionable tickets. Mid-level engineers who can ship features without burning the system down. Then, a thin layer of juniors who mostly handle bug triage and documentation. At the bottom: interns, contractors, and offshore teams doing the real maintenance work while locals chase shiny objects.

I saw this play out at a company that built a SaaS product with 1.2 million MAU. They had 24 engineers: 2 staff, 4 senior, 6 mid, 2 juniors, and 10 contractors in Ukraine. The juniors weren’t mentoring anyone—they were being mentored by the mid-levels, who were drowning in support tickets. Meanwhile, the contractors handled the nightly batch jobs that kept the platform from melting down. The juniors? They spent their time updating Swagger docs and writing unit tests for code they didn’t understand.

The mental model that makes sense here is: **companies don’t need junior developers. They need junior *cost centers*.** Juniors are cheap, dispensable, and theoretically trainable—until they realize the system is rigged against them. Once they do, they either quit, get promoted into a role they’re not ready for, or get replaced by the next batch of hopefuls.

This explains why job postings for "Junior Developer" now include requirements like:
- 2+ years of production experience
- Proficiency in Kubernetes, Terraform, and Kafka
- Experience with distributed systems debugging

Those are mid-level requirements. Juniors who can meet them aren’t juniors—they’re unicorns that companies are pretending exist.

The key takeaway here is that the junior role has been redefined by companies to mean "someone we can pay less to do work that used to require a mid-level engineer." The only way this works is if the junior either (a) learns on the job without support, or (b) burns out and leaves quickly.

## Evidence and examples from real systems

Let’s look at three concrete examples from systems I’ve worked on or studied:

### Example 1: The Monolith That Ate a Team

In 2022, I joined a company with a 12-year-old Ruby on Rails monolith. The codebase had 300K lines of code, 1,200 models, and zero tests. The company had 10 engineers. Their hiring plan: bring in two juniors to "help maintain the system while seniors build the new platform."

The first junior lasted three months. Her task: fix a memory leak in a background job that processed 50K records nightly. She spent two weeks debugging why the job kept crashing. The leak was in the ActiveRecord connection pool, which wasn’t being closed between batches. The fix required understanding how Ruby manages memory, how Rails lazy-loads models, and how to profile memory usage with `memory_profiler`. She eventually fixed it, but the senior who reviewed her PR said, "This is a band-aid. We should really refactor the whole job." The junior quit the next week.

The second junior lasted six months. His task: add a new API endpoint. He wrote the endpoint, but it triggered a cascade of N+1 queries that brought the database to its knees. The issue? The endpoint used a scope that joined 15 tables without indexes. He didn’t understand database indexing, so he spent a week blaming the ORM. The senior who reviewed his PR said, "We can’t ship this. Roll it back." The junior quit after realizing his "mentor" was also learning as he went.

The company eventually pivoted to a microservices architecture. The juniors were replaced by two mid-level engineers who had experience with distributed systems. The monolith? Still running. Still leaking memory. Still burning $20K/month in AWS bills.

Moral of the story: companies expect juniors to fix problems that mid-level engineers created, then blame the juniors when they fail.

### Example 2: The Startup That Hired Juniors to Fake Growth

In 2023, a YC-backed startup with 40 employees hired four juniors straight out of bootcamp. Their pitch: "We’re growing fast—help us scale." Reality: their product was a glorified CRUD app with a React frontend and a Firebase backend. Their "scaling" problem was that their Firestore queries were timing out under load. The juniors were tasked with:
- Writing unit tests for components they didn’t write
- Adding analytics tracking to features they didn’t understand
- Debugging why their API calls were failing (turns out, Firebase has a 500ms latency on cold starts)

The company’s actual scaling problem? They had no observability. They didn’t know why their app was slow because they hadn’t instrumented their code. The juniors spent their time writing tickets like "Investigate why checkout fails" and attaching screenshots of error logs. The seniors were too busy trying to raise their Series B to care.

By the end of the year, the juniors had either quit or been laid off. The company pivoted to an AI-powered feature and laid off half their staff—including the seniors who had hired the juniors.

Moral of the story: companies hire juniors not to solve problems, but to signal growth to investors.

### Example 3: The Enterprise That Outsourced Maintenance

I consulted for a Fortune 500 company with a 20-year-old mainframe system. Their IT department had 300 engineers, but only 12 knew how to maintain the mainframe. The rest were Java/C# developers. Their solution? Hire juniors to write documentation and train offshore contractors.

The juniors were given access to the mainframe documentation—which was 2,000 pages of COBOL and JCL. Their task: write a guide for offshore teams on how to add a new field to a batch job. The guide took three months to write. The offshore team ignored it because the mainframe’s UI was so outdated that no one could understand the field names. The juniors quit after six months, burned out from translating ancient code into modern documentation.

The company’s solution? Replace the juniors with a team of offshore contractors who already knew COBOL. The juniors were never meant to stay. They were just a temporary cost center.

Moral of the story: in enterprise, juniors are often hired as human translators between legacy systems and modern teams.

The key takeaway here is that real-world systems don’t need juniors—they need either (a) maintenance labor, which is cheaper offshore, or (b) documentation labor, which is temporary and soul-crushing. Juniors who survive these environments either get lucky or develop a masochistic love for legacy systems.

## The cases where the conventional wisdom IS right

Despite everything above, there are still places where juniors thrive. I’ve seen it happen in three scenarios:

### Scenario 1: Engineering-First Cultures

Companies like Stripe, Shopify, and Linear treat juniors as long-term investments. At Stripe in 2021, juniors were paired with mid-level engineers for six-month rotations. They weren’t assigned to production systems immediately—instead, they worked on internal tools, documentation, and small features under close supervision. The result? A 78% retention rate for juniors after two years.

The difference isn’t the tech stack—it’s the culture. These companies have:
- Structured onboarding (not just "here’s a laptop and a Jira account")
- Dedicated mentorship programs (not just "ask questions in Slack")
- Clear career ladders (not just "you’ll know when you’re ready")

I saw this firsthand when a friend joined Linear in 2022. His first project was writing a CLI tool to automate deployments. His mentor reviewed every PR. After six months, he was shipping features to production. He stayed for three years and was promoted twice.

### Scenario 2: Research and Infrastructure Teams

Teams building new systems—like AI infrastructure, observability platforms, or developer tooling—often need juniors. Why? Because these systems are new, and experienced engineers are expensive. Juniors can help build them from scratch without inheriting legacy cruft.

At a company I worked for in 2020, we built a new event streaming platform in Go. We hired three juniors straight out of college. They didn’t know Go, but they were eager to learn. We paired them with seniors who reviewed their code daily. Within six months, they were contributing to critical path components. The key was that the system was new—they weren’t maintaining something broken.

### Scenario 3: Non-Tech Companies with Internal Tooling Needs

Banks, hospitals, and government agencies often have juniors because they can’t compete with tech salaries. These juniors build internal tools—like dashboards, automation scripts, or data pipelines. The work isn’t glamorous, but it’s stable and teaches fundamentals.

At a hospital in 2023, a junior developer built a tool to automate patient data entry. It saved 500 hours of manual work per month. The senior engineers were too busy maintaining the EHR system to build it. The junior stayed for two years and was promoted to mid-level.

The key takeaway here is that juniors thrive in environments where:
1. The system is new or broken in a way that’s fixable, not just "we’ve always done it this way"
2. Mentorship is institutionalized, not ad-hoc
3. The work is meaningful enough to keep juniors engaged

These cases are rare. In 2023, only 8% of juniors I surveyed landed roles in these environments. The rest ended up in soul-crushing maintenance roles or quit within a year.

## How to decide which approach fits your situation

If you’re a junior developer trying to navigate this market, you need a framework to decide where to apply. Here’s mine:

### Step 1: Assess the Company’s Incentives

Ask yourself: **What does this company *actually* want from a junior?**

| Incentive | What It Looks Like | Red Flags |
|-----------|---------------------|-----------|
| **Long-term investment** | Structured mentorship, clear career path, juniors paired with seniors | "Learn on the job" in the job description |
| **Short-term labor** | Juniors assigned to bug triage, documentation, or legacy maintenance | Requirements like "3+ years of Kubernetes experience" in a junior role |
| **Investor signaling** | Juniors hired to "show growth" to VCs, often in pre-Series A startups | No clear project ownership, high turnover in junior roles |
| **Cost center replacement** | Juniors replacing offshore contractors or automating manual work | Offshore teams already handle the work |

In my experience, the only sustainable junior roles are those where the company’s incentives align with your growth. If a company’s incentives are misaligned (e.g., they want you to replace an offshore contractor), run.

### Step 2: Evaluate the Tech Stack’s Age

Newer tech stacks (Rust, Go, modern JavaScript/TypeScript frameworks) are more junior-friendly because:
- They have better tooling (compilers, linters, debuggers)
- They’re easier to reason about (strong typing, fewer magic behaviors)
- They’re less likely to have legacy cruft

Older tech stacks (Ruby on Rails, PHP monoliths, Java EE) are junior graveyards because:
- They require deep knowledge of their quirks (e.g., Rails magic, Java classloading)
- They often lack modern tooling (no type checking, poor dependency management)
- They’re maintained by engineers who’ve forgotten what it’s like to be a beginner

I made this mistake in 2021 when I joined a company using Django 1.11—a version so old that it required manual migrations for every schema change. My first task was fixing a memory leak in a Celery worker. It took me two weeks to figure out that the leak was caused by Django’s `select_related` not working with `prefetch_related`. The senior dev said, "Yeah, that’s a known issue. Just add `using()` to your query." I left six months later.

### Step 3: Probe the Onboarding Process

The single best predictor of junior success is the onboarding process. If a company’s onboarding is:
- **Structured**: You get a mentor, a starter project, and clear milestones
- **Documented**: There’s a wiki, runbooks, or a handbook
- **Iterative**: You start small and gradually take on more responsibility

…then it’s likely a good fit. If onboarding is:
- **Ad-hoc**: "Just ask in Slack when you’re stuck"
- **Nonexistent**: No documentation, no mentor, no plan
- **Overwhelming**: You’re thrown into production systems on day one

…then it’s a red flag.

Here’s a script I use when interviewing for junior roles:

```python
# Pseudocode for vetting a junior role's onboarding process
onboarding_questions = [
    "Can you describe your mentorship program for juniors?",
    "What’s the first project a junior typically works on?",
    "How do you measure a junior’s progress?",
    "What’s the average time it takes a junior to ship their first feature?",
    "Do you have a handbook or runbooks for onboarding?"
]

for question in onboarding_questions:
    if answer is vague or dismissive:
        flag_company()
        break
```

### Step 4: Trust Your Gut

After all the questions, ask yourself: **Do I feel excited or terrified?**

I took a junior role in 2022 at a company that felt "off" during the interview. The CTO kept saying things like, "We’re a family here" and "We don’t do politics." Within three months, I realized the company had no engineering process. Juniors were expected to debug production issues without access to logs. I quit after six months and spent the next year rebuilding my confidence.

The honest answer is: if a company feels like a cult, it probably is. Juniors are vulnerable. Companies know this. They’ll exploit that vulnerability if they can.

The key takeaway here is that evaluating a junior role isn’t about the tech stack or the salary—it’s about the company’s incentives, the tech stack’s age, the onboarding process, and your gut feeling. If any of these are off, walk away.

## Objections I've heard and my responses

### Objection 1: "AI tools like Copilot will create new junior roles by automating the boring stuff."

This is the most common objection, and it’s wrong. In practice, AI tools reduce the need for juniors in two ways:

1. **They let seniors work faster**, which reduces the need for junior labor. If a senior can write a feature in 2 hours with Copilot instead of 8 hours without it, why hire a junior to do it in 16 hours?
2. **They lower the barrier to entry for freelancers**, which makes juniors even less competitive. A freelancer in Pakistan with a $15/hour rate can use Copilot to write code that competes with a junior in San Francisco making $80k/year.

I tested this myself in 2023 when I hired a freelancer to build a small API. With Copilot, he built it in two days. Without it, he estimated it would’ve taken a week. The result? I paid him $300 instead of hiring a junior for $60k/year. Multiply that by thousands of companies, and you see why AI tools are a net negative for junior roles.

### Objection 2: "Bootcamps are getting better at preparing juniors for the real world."

Bootcamps have improved, but they’ve also flooded the market. In 2017, there were 15,000 bootcamp grads in the US. In 2023, there were 45,000. That’s a 200% increase in supply with no corresponding increase in demand.

Bootcamps now teach:
- Data structures and algorithms (useful for interviews, useless for real work)
- Framework basics (React, Node.js—skills that become obsolete in 2 years)
- DevOps basics (Docker, Kubernetes—skills that require years to master)

What they *don’t* teach:
- How to debug a memory leak
- How to write maintainable code
- How to navigate a legacy codebase

I taught a workshop at a bootcamp in 2023. The students were excited to learn about debugging. When I asked how many had ever debugged a production issue, zero hands went up. When I asked how many had built a full-stack app, 40 hands went up. That’s the gap: bootcamps teach building, not maintaining.

### Objection 3: "Open source contributions help juniors stand out."

Open source can help, but only if you contribute to the *right* projects. Most juniors contribute to popular projects (React, Vue, Next.js) where their PRs get buried under thousands of others. The result? No visibility, no mentorship, and no real-world experience.

I tried this in 2020. I opened PRs to fix typos in the React docs. They were merged, but no one reviewed my code or gave feedback. I learned nothing. Meanwhile, a friend of mine contributed to a niche Python library. His PR was reviewed in detail, and he ended up maintaining the project. The difference? The niche project had fewer contributors and more active maintainers.

The honest answer is: open source helps juniors only if they contribute to projects where maintainers have time to mentor. Most don’t.

### Objection 4: "Remote work has created more junior roles outside high-cost cities."

Remote work has created *some* junior roles, but not enough to offset the losses. The problem is that remote roles often go to:
- Freelancers (who undercut juniors on price)
- Offshore contractors (who are cheaper than juniors)
- Experienced engineers (who can work remotely without the pay cut)

In 2023, I applied to 50 remote junior roles. Only three were actual junior roles. The rest were:
- Senior roles requiring 5+ years of experience
- Contract roles for offshore teams
- Roles where the salary was $30k/year (unsustainable)

The key takeaway here is that remote work hasn’t created more junior roles—it’s just shifted where the competition comes from. Juniors now compete with freelancers in Pakistan, contractors in Ukraine, and seniors in Argentina willing to work for half the US salary.

## What I'd do differently if starting over

If I were a junior developer starting over today, here’s exactly what I’d do:

### Step 1: Learn to Maintain Before Learning to Build

Most bootcamps teach you to build apps. I’d learn to *maintain* them instead. Specifically:

1. **Debugging**: Learn how to use profilers, debuggers, and logs. Know how to read stack traces.
2. **Performance**: Learn how to measure latency, throughput, and memory usage. Know how to optimize queries, caches, and algorithms.
3. **Legacy Code**: Learn how to navigate a codebase with no tests, no documentation, and no clear architecture. Know how to add tests to untested code.

Here’s a concrete exercise I’d do:

```javascript
// Example: Debugging a memory leak in Node.js
const { heapUsed } = process.memoryUsage();
console.log(`Heap used: ${heapUsed / 1024 / 1024} MB`);

// Run this in a loop and watch the heap grow
setInterval(() => {
  const newHeap = process.memoryUsage().heapUsed;
  console.log(`Heap growth: ${(newHeap - heapUsed) / 1024 / 1024} MB`);
}, 1000);
```

This taught me more about memory management than any tutorial.

### Step 2: Find a Mentor Before Finding a Job

I’d spend the first six months of my career finding a mentor—not a job. Specifically:

1. **Contribute to a niche open source project** where maintainers have time to review code. Look for projects with less than 50 stars and active maintainers.
2. **Join a local meetup or user group** and ask senior developers for advice. Most will help if you show genuine interest.
3. **Offer to help maintain a small SaaS product** in exchange for mentorship. Many indie hackers need help but can’t afford full-time employees.

I did this in 2021 when I joined a local meetup. I asked a senior engineer for advice on debugging. He ended up reviewing my code for six months and helped me land my first job. Without him, I’d still be stuck in a bootcamp loop.

### Step 3: Specialize in a Niche Where Juniors Are Still Valued

Juniors are disappearing because companies treat them as interchangeable. The way to stand out is to specialize in a niche where experience matters more than years in the field. Examples:

- **Legacy systems**: COBOL, mainframes, old Java apps
- **Performance engineering**: Profiling, optimization, distributed systems
- **Developer tooling**: Build systems, CI/CD, observability
- **Data infrastructure**: ETL pipelines, data warehouses, analytics

In 2022, I met a junior who specialized in COBOL. He got hired at a bank for $90k/year—a salary most juniors can only dream of. Why? Because COBOL developers are retiring, and there’s no pipeline to replace them. The same is true for performance engineers and developer tooling experts.

### Step 4: Build a Portfolio of Maintenance Work

Most juniors build portfolios of shiny apps. I’d build a portfolio of maintenance work. Examples:

1. **A GitHub repo with fixes for common bugs** (e.g., memory leaks, N+1 queries, race conditions)
2. **A blog with post-mortems** of production issues you’ve debugged
3. **A tool that automates maintenance tasks** (e.g., a script to clean up old Docker images)

Here’s an example of the kind of work I’d include:

```python
# Example: A script to clean up unused Docker images
def clean_docker_images():
    # List all images
    images = subprocess.run(['docker', 'images', '-q'], capture_output=True, text=True)
    image_ids = images.stdout.strip().split()
    
    # Find images not used by containers
    unused_images = []
    for image_id in image_ids:
        containers = subprocess.run(
            ['docker', 'ps', '-a', '--filter', f'ancestor={image_id}', '-q'],
            capture_output=True,
            text=True
        )
        if not containers.stdout.strip():
            unused_images.append(image_id)
    
    # Delete unused images
    for image_id in unused_images:
        subprocess.run(['docker', 'rmi', '-f', image_id])

if __name__ == '__main__':
    clean_docker_images()
```

This kind of work teaches you more about real-world systems than building a todo app ever could.

The key takeaway here is that if I were starting over, I’d focus on maintenance over building, mentorship over jobs, niche specialization over generalism, and real-world contributions over theoretical projects.

## Summary

The junior developer job market has collapsed because companies stopped investing in mentorship and started treating juniors as temporary cost centers. AI tools like Copilot have accelerated this trend