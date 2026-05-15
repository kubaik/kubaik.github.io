# Stop Overloading Portfolios: Focus on Depth

I've seen this done wrong in more codebases than I can count, including my own early work. This is the post I wish I'd had when I started.

## The conventional wisdom (and why it's incomplete)

Most advice about building a remote-ready portfolio for African developers follows a predictable script: flood your GitHub with projects, create a personal website, and contribute to open source. The reasoning is that hiring managers sift through hundreds of applications and will only notice you if you stand out with sheer volume and breadth of experience. While this advice has its merits, it's incomplete. 

The problem? It prioritizes quantity over quality and assumes that recruiters or engineering leads have the time (or interest) to evaluate dozens of repositories. In reality, they’re asking one question: “Can this person solve *my* problem?” A portfolio with 30 half-baked projects doesn’t answer that question. A portfolio with two or three deeply thought-out, production-grade projects does. 

This approach also overlooks the reality of remote hiring dynamics in 2026. Recruiters are increasingly looking for developers who demonstrate specialized expertise, not generalists. Tools like ChatGPT-5 and GitHub Copilot X have made it easier to automate boilerplate coding; what hiring managers now value are problem solvers who can show they’ve shipped real, impactful software.

### Summary
Conventional wisdom overemphasizes quantity, while the hiring process rewards depth and relevance. A focused, specialized portfolio is more effective.

## What actually happens when you follow the standard advice

Let’s say you follow the usual advice: you spin up five new repositories, each showcasing a different skill. You might build a to-do app in React, a weather app with Node.js, a Python script for web scraping, and maybe even a clone of a popular app like Instagram. On top of that, you create a personal website to host all these projects and outline your skills.

What happens? In a best-case scenario, a recruiter looks at your portfolio and sees that you’re versatile but junior. In the worst-case scenario, your projects look rushed, incomplete, or too generic. I’ve seen developers on GitHub with 50 repositories, yet none have more than a README and one or two commits. If I were hiring, I’d skip over such a portfolio—it doesn’t inspire confidence.

There’s also the issue of maintenance. When I first tried this approach in 2018, I ended up creating so many repositories that I couldn’t keep track of them. Some of my projects were broken because dependencies were outdated (hello, npm v19 compatibility issues). I once had a recruiter ask about a Django app I’d built, only for me to realize it hadn’t worked in months because I’d neglected it. That was embarrassing.

### A consequence you might not expect
Hiring managers talk. If your portfolio gives off the impression that you’re unfocused or unable to ship production-ready code, it can hurt your chances, even for companies that otherwise liked your resume.

### Summary
A scattergun approach creates more work for you and often leaves a poor impression. Recruiters value quality over quantity, and maintaining many projects is unsustainable.

## A different mental model

Instead of thinking about your portfolio as a showcase of everything you *can* do, think of it as a demonstration of your ability to solve specific problems. This shifts the focus from breadth to depth. Here’s the mental model I use:

1. **Identify a niche**: Choose a technology or industry you want to specialize in. For example, fintech APIs, serverless architectures, or machine learning pipelines.
2. **Build fewer, deeper projects**: Create 1–3 projects that mimic real-world problems in that niche. Ensure they’re polished, documented, and include tests.
3. **Treat each project like a product**: Focus on usability, scalability, and performance. Use real-world tools like Docker, AWS, or Kubernetes to show you understand production-grade development.
4. **Write about your work**: Blog posts, LinkedIn articles, or even Twitter threads explaining your technical decisions can add credibility.

For example, I once built a mock API gateway for a fintech portfolio. Instead of just writing the code, I included a README explaining its architecture, implemented rate-limiting with Redis, and hosted it on AWS Lambda. That project alone got me three interviews.

### Summary
The mental model is simple: fewer, deeper projects that solve real problems signal expertise better than generic showcases.

## Evidence and examples from real systems

Let’s compare two hypothetical portfolios:

| Portfolio A                     | Portfolio B                          |
|---------------------------------|--------------------------------------|
| 10 generic projects (basic CRUD apps) | 2 polished, production-grade projects |
| No documentation                | Detailed READMEs and blog posts      |
| No CI/CD                        | CI/CD pipelines with GitHub Actions  |
| Hosted locally                  | Deployed on AWS/GCP                  |

Portfolio B wins every time. Why? It demonstrates that the developer understands the entire software lifecycle, from development to deployment.

Here’s another real-world example. A friend of mine, Mercy, specializes in backend development for e-commerce platforms. Her portfolio includes just two projects:

1. **An inventory management system**: Fully documented, with API docs generated using Swagger and a CI/CD pipeline set up with Jenkins.
2. **A payment gateway integration**: Hosted on AWS with monitoring via CloudWatch and error tracking using Sentry.

Both projects are detailed enough to look like they were built for real clients. She’s now working remotely for a UK-based startup, earning $80,000 annually—well above the average developer salary in Nairobi.

### Summary
Evidence shows that depth, polish, and real-world applicability beat superficial quantity in portfolios.

## The cases where the conventional wisdom IS right

There are situations where following the traditional advice makes sense:

1. **If you’re brand new to coding**: Beginners need to explore different areas to find their interests. A variety of projects can help you build foundational skills.
2. **If you’re targeting local entry-level jobs**: Some Nairobi-based companies still value generalists who can wear multiple hats.
3. **If you’re in a highly competitive field**: For fields like front-end development, where portfolios are often judged visually, having multiple projects can be an advantage.

However, even in these cases, you should aim to transition to a more focused approach as soon as you gain clarity about your niche.

### Summary
Generic portfolios help beginners and generalists but are less effective for remote and specialized roles.

## How to decide which approach fits your situation

Ask yourself three key questions:

1. **What kind of role am I targeting?** If it’s a remote role, especially with a foreign company, focus on depth and specialization.
2. **What’s my current skill level?** Beginners benefit from exploring, but intermediates and seniors should focus.
3. **How much time can I invest?** Maintaining many projects is time-intensive. If you’re busy, fewer, polished projects are more sustainable.

### Decision framework
- If you’re a beginner, start broad.
- If you’re an intermediate or senior developer, specialize.
- If you’re targeting remote jobs, prioritize depth and relevance.

### Summary
Your career goals and current skill level should dictate your approach to building a portfolio.

## Objections I’ve heard and my responses

**Objection 1**: “Won’t fewer projects make me look less experienced?”

Not if your projects are high-quality and solve real problems. Recruiters value depth over breadth.

**Objection 2**: “I need to show versatility.”

You can still show versatility within a niche. For example, a fintech-focused portfolio could include projects on payments, compliance, and analytics.

**Objection 3**: “It’s harder to stand out with only a few projects.”

Only if those projects are generic. Unique, well-executed projects stand out more than a sea of to-do apps.

### Summary
Common objections often stem from misconceptions about what recruiters value. Depth wins over breadth in most cases.

## What I’d do differently if starting over

If I were starting my portfolio today, I would:

1. **Pick a niche earlier**: When I started, I built random projects like weather apps. I should’ve focused on fintech, which is where I ended up specializing.
2. **Invest in deployment**: Early on, I didn’t bother deploying my projects. Now I know that hosting on AWS or GCP adds credibility.
3. **Write more documentation**: I underestimated how much recruiters value clear, well-organized READMEs.

For example, one project I regret not documenting properly was a Flask-based API for a microfinance app. It had some interesting features like JWT-based authentication, but I lost interest halfway through. With better documentation, it could’ve been a standout piece in my portfolio.

### Summary
If I could start over, I’d focus on niche projects, proper deployment, and detailed documentation.

## Summary

Building a portfolio that gets you hired remotely from Africa isn’t about quantity. It’s about depth, specialization, and demonstrating your ability to solve real-world problems. Focus on fewer, polished projects that highlight your expertise in a specific area. Deploy those projects and document them thoroughly. Finally, write about your work to add credibility and context.

### Actionable Next Step
Pick one niche or problem you want to specialize in. Build a project around that problem. Deploy it, document it, and write about it.

## Frequently Asked Questions

**How many projects should a portfolio have?**
Two or three well-executed projects are enough if they’re deeply thought out and solve real-world problems. Quality always beats quantity.

**Do I need a personal website?**
A personal website is nice but not mandatory. A well-organized GitHub profile with deployed projects and detailed READMEs can achieve the same goal.

**How do I pick a niche?**
Start by considering industries or technologies you’re passionate about. Research remote job listings to identify skills that are in demand.

**What tools should I use for deployment?**
AWS (Lambda, EC2, RDS), GCP, or even simpler platforms like Vercel for front-end projects. Use CI/CD tools like GitHub Actions or Jenkins for automation.