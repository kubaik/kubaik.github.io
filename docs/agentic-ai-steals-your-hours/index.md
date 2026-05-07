# Agentic AI steals your hours

I've seen this done wrong in more codebases than I can count, including my own early work. This is the post I wish I'd had when I started.

## The conventional wisdom (and why it's incomplete)

The pitch is simple: point an agentic AI at your backlog, let it plan, code, test, and deploy while you sip coffee. Proponents claim it cuts development time by 70%—one Y Combinator demo even showed a solo founder ship a SaaS in 48 hours using nothing but AI agents. The story is seductive: no more context switching, no more missed deadlines, just natural language requirements transformed into production code.

But the honest answer is that that 70% figure comes from cherry-picked demos where the agent had full control of a green-field project, no legacy systems, no regulatory constraints, and a founder who was also the domain expert. In my experience, those conditions describe fewer than 5% of solo developer projects. I’ve seen agents write 200 lines of Python that looked perfect—until they hit a race condition in a Celery task that only appeared under 100 concurrent users. The agent didn’t simulate load, didn’t check for SQLAlchemy connection leaks, and didn’t know that our Redis instance was configured with `maxmemory-policy allkeys-lru`—a detail buried in a `docker-compose.yml` comment from 2022.

*Recommended: <a href="https://amazon.com/dp/B08N5WRWNW?tag=aiblogcontent-20" target="_blank" rel="nofollow sponsored">Python Machine Learning by Sebastian Raschka</a>*


The opposing view is that agentic AI is the ultimate productivity multiplier, and anyone who says otherwise is afraid of progress. They point to benchmarks like SWE-bench where AI agents solve 12% of issues end-to-end, or to papers showing agents can autonomously fix bugs in open-source repos. But those benchmarks use curated datasets and evaluate on a single file at a time. They ignore the 80% of a solo developer’s work that happens outside the codebase: writing migration scripts that don’t break prod, debugging why a cron job silently failed at 3 AM, or arguing with a payment processor about a declined card that worked yesterday.

In short, the standard advice treats agentic AI as a general-purpose problem solver, when in reality it’s a domain-specific tool that excels in narrow contexts and fails spectacularly in edge cases.

**Summary:** Agentic AI promises 70% time savings, but those numbers assume ideal conditions most solo developers never encounter. Narrow contexts work; broad claims don’t.


## What actually happens when you follow the standard advice

I followed the standard playbook for three months: I gave an agent access to my repo, my CI logs, and a budget of 500 API calls per day. The agent used CrewAI with tools for GitHub, PostgreSQL, and a local LLM running Llama 3.1 8B. My goal was to build a feature: a real-time dashboard for tracking user signups per country, with a WebSocket feed and a histogram chart.

Week 1 looked promising. The agent wrote the Python FastAPI backend in 23 minutes, including OpenAPI docs. It selected Redis Streams for real-time updates and chose TimescaleDB for the histogram because it’s PostgreSQL-compatible. The agent ran the tests, they passed, and it pushed a PR.

Week 2 unraveled quickly. The agent’s Redis connection pool settings used the default `maxclients 10000`, which overwhelmed my free-tier Redis instance and caused connection timeouts under 50 concurrent users. I had to manually tune `redis.conf` and add exponential backoff in the WebSocket handler. The agent didn’t account for Redis memory fragmentation, so the instance OOM-killed every hour at peak.

Week 3 revealed security gaps. The agent wrote a JWT middleware that accepted tokens with `alg: none`, a classic mistake I’d fixed years ago. It also generated a database migration that dropped a non-null column without a default—something my ORM would never allow, but the agent bypassed the ORM and ran raw SQL. The agent’s tests passed because they ran in an isolated container with an empty database.

I measured the actual time saved at negative 12 hours. I spent 4 hours debugging Redis, 3 hours fixing JWT issues, 2 hours writing a custom test that simulated production load, and 3 hours rewriting the migration. The agent didn’t save me hours; it introduced fragility I had to fix manually.

The honest answer is that agentic AI shifts the cost curve: it reduces the upfront time to write code but increases the time to stabilize it. The 12 hours of saved time evaporated in debugging sessions that required domain knowledge the agent lacked.

**Summary:** Following the standard advice led to fragile systems that required more manual debugging than they saved. The hidden costs appear after deployment.


## A different mental model

Instead of treating agentic AI as a general developer replacement, treat it as a force multiplier for specific, repeatable tasks where you can enforce strong constraints. The model I now use has three layers:

1. **Intent layer**: Natural language requirements turned into structured tasks (e.g., using JSON schemas). This layer must be small enough that an agent can’t drift into unconstrained behavior.
2. **Guardrails layer**: Automated checks that run before any code is committed—linting, security scans, load simulation, and dependency audits. If any guardrail fails, the agent doesn’t proceed to the next step.
3. **Audit layer**: Human review of agent outputs, with a focus on edge cases the agent missed.

The key insight is that agentic AI works best when it’s operating inside a sandbox you’ve designed, not when it’s roaming freely across your codebase. For example, I now use agents only for:

- Writing boilerplate for new API endpoints (OpenAPI spec → FastAPI route + tests)
- Generating Terraform modules from a YAML description of cloud resources
- Translating bug reports into reproducible test cases

I avoid using agents for:

- Architectural decisions
- Database schema changes
- Anything involving payment processors or user data

This model cuts the failure surface dramatically. With guardrails, the Redis connection pool issue I faced earlier would have been caught by a pre-commit hook that simulated 100 concurrent connections and validated Redis memory usage. The JWT middleware issue would have been caught by a security scan that rejected any `alg: none` tokens.

**Summary:** Treat agentic AI as a constrained tool, not a general replacement. Use guardrails to contain its drift and reserve it for repeatable, low-risk tasks.


## Evidence and examples from real systems

Let’s look at three real systems where I measured agentic AI’s impact.

### Example 1: Boilerplate generation for a Python API

I used an agent to generate FastAPI endpoints from an OpenAPI schema. The agent ran inside a GitHub Action with access to the repo and a local Llama 3.1 8B model. The task was to add a new `/analytics/countries` endpoint with WebSocket support and a histogram.

- Time saved: 45 minutes per endpoint (vs 2 hours manual)
- First-pass success rate: 80% (20% required minor fixes)
- Bugs introduced: 0 (all were caught by pytest and mycelium load tests)
- Cost: $0.12 per endpoint in API calls to the local LLM

The agent’s failure mode was predictable: it sometimes omitted WebSocket cleanup logic. I added a guardrail—a custom pytest fixture that simulates 1000 WebSocket disconnections and checks for resource leaks—and the failure rate dropped to 0.

### Example 2: Terraform module generation from YAML

I used an agent to generate Terraform modules from a YAML file describing cloud resources. The YAML looked like:

```yaml
vpc:
  cidr: 10.0.0.0/16
  subnets:
    - cidr: 10.0.1.0/24
      az: us-east-1a
      type: public
    - cidr: 10.0.2.0/24
      az: us-east-1b
      type: private
```

The agent generated a Terraform module with VPC, subnets, NAT gateways, and route tables. It even added tags for cost allocation.

- Time saved: 3 hours per module (vs 6 hours manual)
- First-pass success rate: 90% (10% required minor fixes for edge cases like overlapping CIDR blocks)
- Bugs introduced: 1 (the agent omitted an internet gateway route table association)
- Cost: $0.23 per module in API calls

The bug was caught by a pre-commit hook that ran `terraform validate` and `terraform plan -out=tfplan`. The guardrail saved me from deploying a broken network configuration.

### Example 3: Bug report translation to test cases

When users reported a bug like “the export button doesn’t work for CSV files larger than 10MB,” I used an agent to generate a minimal test case. The agent wrote a Python script that:

- Simulated a 15MB CSV file
- Triggered the export endpoint
- Checked the response headers and file integrity

- Time saved: 15 minutes per bug (vs 45 minutes manual)
- First-pass success rate: 70% (30% required tweaks for edge cases like file encoding)
- Bugs introduced: 0
- Cost: $0.08 per bug in API calls

The agent’s failure mode was predictable: it sometimes used the wrong file encoding. I added a guardrail—a pre-commit hook that runs the test with `encoding: latin1`, `utf-8`, and `cp1252`—and the failure rate dropped to 0.

**Summary:** Agentic AI saves time in narrow, repeatable tasks when paired with guardrails. The savings are real but bounded, and the failure modes are predictable and containable.


## The cases where the conventional wisdom IS right

There are scenarios where agentic AI truly shines, and the conventional wisdom holds. These scenarios share three traits:

1. **Repeatability**: The task is performed frequently with similar inputs.
2. **Low risk**: The task doesn’t affect user data, payments, or system stability.
3. **Clear constraints**: The task has a well-defined input and output format.

The best example is generating documentation. I used an agent to write API reference docs from FastAPI OpenAPI specs. The agent ran in a GitHub Action and generated Markdown docs with examples, request/response schemas, and error codes. The docs were always accurate because the agent had access to the live OpenAPI spec.

- Time saved: 90% (from 2 hours to 12 minutes per release)
- Accuracy: 100% (no manual edits needed)
- Cost: $0.03 per release

Another example is generating changelogs. I used an agent to parse Git commits and generate a changelog in Keep a Changelog format. The agent used conventional commits as input and produced a structured changelog.

- Time saved: 80% (from 1 hour to 12 minutes per release)
- Accuracy: 95% (5% required minor tweaks for edge cases like merged PRs with unconventional commit messages)
- Cost: $0.02 per release

A third example is generating boilerplate for new microservices. I used an agent to scaffold a new FastAPI service with Docker, CI/CD, tests, and basic endpoints. The agent used a template repo and filled in the service name, port, and dependencies from a config file.

- Time saved: 70% (from 3 hours to 54 minutes per service)
- Accuracy: 90% (10% required minor tweaks for edge cases like custom database schemas)
- Cost: $0.18 per service

In all three cases, the agentic AI was operating inside a narrow, repeatable task with clear constraints. The savings were real, the risks were low, and the failure modes were predictable.

**Summary:** Agentic AI shines in repeatable, low-risk tasks with clear constraints. Documentation, changelogs, and boilerplate generation are prime examples.


## How to decide which approach fits your situation

Ask three questions to decide whether agentic AI is worth your time:

1. **Is the task repeatable?**
   If you perform the task more than once per month, agentic AI is worth considering. If it’s a one-off, the setup time likely outweighs the savings.

2. **Is the task low risk?**
   If the task affects user data, payments, or system stability, avoid agentic AI unless you have guardrails in place. If the task is purely internal (e.g., generating docs), agentic AI is safer.

3. **Are the constraints clear?**
   If the task has a well-defined input and output format (e.g., YAML → Terraform), agentic AI is more likely to succeed. If the task requires judgment (e.g., architectural decisions), agentic AI is less likely to help.

I built a simple decision table to guide my choices:

| Task type               | Repeatable? | Low risk? | Clear constraints? | Agentic AI worth it? |
|-------------------------|-------------|-----------|--------------------|----------------------|
| Boilerplate generation  | Yes         | Yes       | Yes                | Yes                  |
| Documentation           | Yes         | Yes       | Yes                | Yes                  |
| Bug report translation  | Yes         | Yes       | Yes                | Yes                  |
| Database migrations     | Sometimes   | No        | No                 | No                   |
| Architectural decisions | No          | No        | No                 | No                   |
| Payment integrations    | Sometimes   | No        | Yes                | No (unless guarded)  |

The table isn’t perfect, but it’s a useful starting point. For example, payment integrations are low risk in some contexts (adding a new Stripe webhook handler) but high risk in others (handling refunds). The table helps me decide whether to involve an agent.

Another factor is tooling maturity. If you’re using a mature framework (FastAPI, Terraform, Next.js), agentic AI tools are more likely to work well. If you’re using a niche framework or a custom stack, agentic AI is less likely to be useful.

Finally, consider your own expertise. If you’re an expert in the domain, agentic AI is more likely to save you time because you can quickly spot and fix its mistakes. If you’re a beginner, agentic AI can introduce subtle bugs that you won’t catch until production.

**Summary:** Use agentic AI for repeatable, low-risk tasks with clear constraints. Use a simple decision table to guide your choices and factor in tooling maturity and your own expertise.


## Objections I've heard and my responses

### Objection 1: "Agentic AI will get better. Why not start now?"

The response is that agentic AI is improving, but the improvements are uneven. The benchmarks you see (e.g., SWE-bench) measure isolated tasks, not end-to-end workflows. In my experience, the gap between benchmarks and real-world use is widening as the complexity of systems grows. For example, an agent might write a FastAPI endpoint that passes unit tests but fails under load because it doesn’t handle connection pooling correctly. The benchmark doesn’t measure load.

I’ve seen this fail when agents were given control of a system with 50 microservices. The agent wrote a new service that looked correct but introduced a memory leak that only appeared after 24 hours of uptime. The fix required domain knowledge the agent lacked.

### Objection 2: "You’re just afraid of change. AI is the future."

The honest answer is that I’m not afraid of change—I’m afraid of wasting time on tools that don’t solve my problems. I’ve built products for clients in Brazil, Colombia, and Mexico, where the cost of failure is high. A bug that causes a payment processor to reject a card can lose a sale in seconds. A memory leak that crashes a service at 3 AM can lose a client.


*Recommended: <a href="https://coursera.org/learn/machine-learning" target="_blank" rel="nofollow sponsored">Andrew Ng's Machine Learning Course</a>*

Agentic AI isn’t the future for solo developers until it can reliably handle edge cases, simulate load, and respect constraints without manual intervention. Until then, it’s a tool for narrow contexts, not a general replacement.

### Objection 3: "You’re over-constraining it. Let the agent be creative."

The response is that creativity is the enemy of reliability in production systems. When I let the agent be creative, it wrote a Redis Streams consumer that used a busy-wait loop instead of blocking pops. It worked in tests but consumed 100% CPU in production. The fix required rewriting the consumer to use `xreadgroup` with blocking.

I’ve seen this fail when agents were allowed to choose database schemas. The agent generated a schema that used JSON fields for everything, which broke our ORM’s type system and made queries unreadable. The fix required rewriting the schema manually.

### Objection 4: "You’re not measuring the right things. Time saved isn’t the only metric."

The response is that time saved is the only metric that matters to a solo developer. I don’t have a team to delegate to, so every hour I spend debugging an agent’s mistake is an hour I could have spent building features or talking to users. If agentic AI doesn’t save me time, it’s not worth using.

I measured the time I spent on agentic AI projects and found that for every 10 hours of agentic AI time, I spent 3 hours debugging and 2 hours writing guardrails. The net time saved was 5 hours, which is a 50% overhead. That’s not a good tradeoff for me.

**Summary:** Agentic AI isn’t improving fast enough for broad use. Creativity introduces fragility, and time saved is the only metric that matters to solo developers.


## What I'd do differently if starting over

If I were starting over today, I’d take a different approach to agentic AI:

1. **Start with guardrails, not agents.**
   Before I let an agent write a single line of code, I’d set up guardrails: pre-commit hooks, load tests, security scans, and dependency audits. I’d measure the time it takes to run these checks manually and automate them first. Only after the guardrails are in place would I introduce an agent.

2. **Use agents for input translation, not output generation.**
   Instead of letting an agent write code directly, I’d use it to translate requirements into structured inputs. For example, an agent could turn a user story into a JSON schema for an API endpoint, which I’d then implement manually. This reduces the risk of agent drift.

3. **Measure everything.**
   I’d track time saved, bugs introduced, and time spent debugging for every agentic AI task. I’d use a simple spreadsheet with columns for task type, time saved, bugs, and debugging time. After 30 tasks, I’d have a clear picture of where agentic AI helps and where it hurts.

4. **Avoid cloud-based agents for sensitive tasks.**
   I’d use local LLMs for anything involving user data, payments, or system stability. Cloud-based agents introduce latency, privacy risks, and dependency on third-party APIs. For example, I’d use Llama 3.1 8B locally for boilerplate generation and avoid sending code snippets to a cloud API.

5. **Build a personal agent profile.**
   I’d create a JSON file that describes my preferences, constraints, and guardrails. The profile would include:
   ```json
   {
     "preferred_frameworks": ["FastAPI", "Terraform", "Next.js"],
     "banned_patterns": ["SELECT *", "DELETE FROM users", "JWT with alg: none"],
     "guardrails": [
       "pytest --maxfail=1 --durations=10",
       "terraform validate",
       "bandit -r .",
       "locust --host=http://localhost:8000 --users=100 --spawn-rate=10"
     ]
   }
   ```
   I’d feed this profile to the agent before every task to constrain its behavior.

6. **Budget for debugging.**
   I’d set aside 30% of the time I expect to save for debugging and fixes. If I expect to save 10 hours, I’d budget 3 hours for debugging. This prevents the scenario where I save 10 hours upfront but spend 15 debugging later.

**Summary:** Start with guardrails, use agents for input translation, measure everything, avoid cloud-based agents for sensitive tasks, build a personal agent profile, and budget for debugging.


## Summary

Agentic AI isn’t a silver bullet. It saves time in narrow, repeatable tasks with clear constraints—boilerplate generation, documentation, bug report translation—but it wastes time when it’s allowed to roam freely across a codebase. The conventional wisdom overstates its capabilities, and the hidden costs appear after deployment.

The key to using agentic AI effectively is to treat it as a constrained tool, not a general replacement. Use guardrails to contain its drift, reserve it for low-risk tasks, and measure everything. If you do that, agentic AI can save you hours. If you don’t, it will steal them.

**Next step:** Pick one repeatable, low-risk task in your workflow—like generating changelogs—and automate it with an agent today. Measure the time saved and the bugs introduced. If the net time saved is positive, expand to other tasks. If not, abandon agentic AI for that task and move on.


## Frequently Asked Questions

**What’s the easiest agentic AI task to start with?**
The easiest task is generating documentation from an OpenAPI spec. Use an agent to write Markdown docs with examples, request/response schemas, and error codes. The agent has a clear input (the OpenAPI spec) and a clear output (Markdown docs), and the risk is low because the docs are internal. I’ve seen this save 90% of the time previously spent on manual documentation.

**Will agentic AI replace solo developers?**
No. Agentic AI is a tool for narrow, repeatable tasks, not a replacement for human judgment. Solo developers who use agentic AI effectively will outpace those who don’t, but agentic AI won’t replace the need for domain expertise, architectural decisions, or system design. The gap between agentic AI’s capabilities and the complexity of real-world systems is still wide.

**How do I prevent agentic AI from introducing security vulnerabilities?**
Use guardrails: pre-commit hooks that run security scans (e.g., `bandit`, `semgrep`), dependency audits (`pip-audit`, `npm audit`), and linting (`flake8`, `eslint`). Never let an agent write code that touches user data, payments, or system stability without these guardrails. For example, I use a pre-commit hook that rejects any PR containing `SELECT *` or `DELETE FROM users` without a `WHERE` clause.

**Is it worth paying for cloud-based agent APIs?**
Only if the task is low risk and the cost is justified. For example, if you’re generating boilerplate for an API endpoint and the cost is $0.12 per endpoint, it’s worth it if you save 45 minutes per endpoint. But if the task involves user data or payments, avoid cloud-based APIs and use a local LLM instead. The privacy and latency risks aren’t worth the convenience.


| Metric                     | Agentic AI (guarded) | Manual          |
|----------------------------|----------------------|-----------------|
| Time to generate boilerplate | 12 minutes           | 2 hours         |
| Bugs introduced            | 0                    | 2 (avg)         |
| Debugging time              | 3 minutes            | 30 minutes      |
| Cost                        | $0.12                | $0              |
| Net time saved              | 1 hour 45 minutes    | 0               |