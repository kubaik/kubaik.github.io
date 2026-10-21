# Docs that survive: the bus factor fix

Most engineering teams treat documentation as a write-only artifact: someone writes a runbook, a design doc, a README, and then everyone moves on. The document rots quietly until the person who wrote it leaves. Then the team discovers that the "documentation" was never really documentation — it was a personal memory aid, full of implicit context that only its author could decode. The real problem isn't that docs go stale; it's that they were never designed to be read by anyone who wasn't already in the author's head.

This is an explainer about writing documentation that survives the departure of its author. Not documentation that is merely present, but documentation that lets a new engineer — or a teammate three time zones away — act without asking the author a question. The goal is to reduce the bus factor of written knowledge: the number of people who must be hit by a bus before a piece of knowledge is lost forever.

## The one-paragraph version

Documentation survives its author when it is written for a reader who has no access to the author's context. That means every doc must state its purpose, its audience, its assumptions, and the steps or decisions it encodes, in a form that can be verified without asking a human. It must live in a place the team already looks (usually the repo, next to the code), it must be versioned with the code, and it must be tested — either by a new team member following it or by automated checks. The rest of this article is about how to do that without turning documentation into a second job.

## Why this concept confuses people

Two mental models collide. The first is the **personal notebook** model: documentation is a place to jot down what you did so you don't forget. The second is the **product manual** model: documentation is an artifact produced for a reader who is not you. Most teams say they want the second but reward the first, because the first is fast and the second is slow. A personal notebook can be written in five minutes; a product manual takes an hour. When deadlines press, the notebook wins.

The confusion deepens because the two models look identical on the surface. Both produce a markdown file with a title and some bullet points. The difference is in what is left unsaid. A personal notebook assumes the reader knows why the document exists, what problem it solves, which environment it applies to, and what "done" looks like. A product manual states all of that explicitly. When the author leaves, the notebook becomes a cryptic artifact; the manual remains usable.

There is also a cultural confusion: many engineers believe that good code is self-documenting, so documentation is only for the parts that code cannot express. That is half true. Code can express *what* it does; it rarely expresses *why* a particular trade-off was chosen, what alternatives were rejected, or what operational assumptions the code depends on. Those are exactly the things a new reader needs and exactly the things that vanish when the author leaves.

## The mental model that makes it click

Think of documentation as a **contract with a future reader**. The contract says: if you follow these steps, in this environment, with these assumptions, you will get this result. A contract is only valid if both parties can verify it. The author verifies it by writing it; the reader verifies it by executing it. If the reader cannot execute it without asking the author a question, the contract is incomplete.

A useful analogy is a recipe. A recipe that says "cook until done" is not a recipe; it's a note to self. A recipe that says "bake at 180°C for 25 minutes, until a skewer inserted in the center comes out clean" is a recipe, because a reader who has never cooked that dish can verify the result. The difference is not length; it's the presence of verifiable checkpoints. Documentation that survives its author is full of verifiable checkpoints: commands to run, outputs to expect, error messages to recognize, and decisions with stated reasons.

The contract model also tells you what to leave out. A contract does not need to explain the history of the legal system; it needs to state the terms. Similarly, a runbook does not need to explain what Kubernetes is; it needs to state which cluster, which namespace, which command, and what a successful result looks like. Context that the reader can get elsewhere should be linked, not inlined. Context that only the author knows must be written down.

## A concrete worked example

Consider a runbook for rotating a database credential. A personal-notebook version might look like this:

```markdown
# Rotate DB password

1. Update the secret in Vault.
2. Restart the app.
3. Check it works.
```

This is unusable by anyone who wasn't there. Which Vault path? Which app? What does "check it works" mean — which endpoint, which log line, which dashboard? What happens if the restart fails halfway through? What is the rollback? A reader who has never done this will have to ask the author at least four questions.

A survivable version looks like this:

```markdown
# Rotate the primary database password

**Audience:** on-call engineer with production access.
**Applies to:** the `orders-api` service in the `prod-eu` cluster.
**Prerequisites:** `vault` CLI logged in to the prod instance; `kubectl` context set to `prod-eu`.

## Steps

1. Generate a new password and write it to Vault:
   ```bash
   NEW_PW=$(openssl rand -base64 32)
   vault kv put secret/orders-api/db password="$NEW_PW"
   ```
   Expected: `====== Secret Path ======` output with version incremented.

2. Trigger a rolling restart so pods pick up the new secret:
   ```bash
   kubectl -n orders rollout restart deployment/orders-api
   kubectl -n orders rollout status deployment/orders-api --timeout=5m
   ```
   Expected: `deployment "orders-api" successfully rolled out`.

3. Verify the app can reach the database:
   ```bash
   curl -fsS https://orders.prod.example.com/healthz | jq .db
   ```
   Expected: `"ok"`. If you see `"error"`, check the pod logs for `password authentication failed` and roll back by re-running step 1 with the previous secret version.

## Rollback

If step 2 fails, the old password is still valid until the pods restart. Re-run step 1 with the previous secret version from Vault's version history, then restart again.
```

Notice what changed. The survivable version states audience, scope, and prerequisites. It gives exact commands with expected outputs. It defines what "check it works" means in terms of an HTTP endpoint and a JSON field. It includes a rollback path. None of that is optional if the doc is to survive its author.

The same principle applies to design docs. A design doc that says "we chose Kafka because it's better" is a note to self. A design doc that says "we chose Kafka over SQS because we need replayable ordered logs for the audit service; the cost is operational complexity, which we accept because the audit service is the only consumer" is a contract with a future reader, who can now evaluate whether the decision still holds.

## How this connects to things you already know

If you have ever written an incident postmortem, you already know the shape of survivable documentation. A good postmortem states the timeline, the impact, the root cause, and the action items, with enough detail that someone who wasn't in the incident can understand what happened. It does not say "the database was slow"; it says "p99 latency on `orders-api` rose from a baseline of 200ms to 4s at 14:02 UTC, correlating with connection pool exhaustion on the primary." The same specificity that makes a postmortem useful makes a runbook useful.

Similarly, if you have ever onboarded a new engineer, you know the questions they ask. Those questions are a map of what your documentation is missing. The first week of onboarding is the most honest documentation review you will ever get, because the new engineer has no context to fill in the gaps. If you can capture those questions and turn them into doc updates, you are doing exactly the work this article describes.

On the security side, documentation that survives its author is also documentation that survives an audit. A runbook that names the exact IAM role, the exact secret path, and the exact command to rotate a credential is both easier to follow and easier to review for least privilege. Docs that say "use the admin key" are a security incident waiting to happen, because the next reader will use whatever key they can find. Writing down the specific, scoped credential is a security control as much as a usability one.

## Common misconceptions, corrected

**Misconception: documentation should be comprehensive.** In practice, comprehensive documentation is usually unread. The goal is not to document everything; it is to document the things a reader cannot infer. A README that explains how to run the tests and where the service is deployed is more valuable than a 40-page architecture document that nobody updates. Write the minimum that lets a reader act, and link to the rest.

**Misconception: the code is the documentation.** Code tells you what happens; it does not tell you why, or what assumptions it depends on, or what to do when it fails at 3 a.m. The code is necessary but not sufficient. A short "why" comment or a linked design doc is often the difference between a five-minute fix and a two-hour investigation.

**Misconception: documentation lives in a wiki.** Wikis are where documentation often goes to die, because they are disconnected from the code and from the review process. Documentation that lives in the repo, next to the code it describes, gets updated when the code changes, because the same pull request can touch both. Documentation in a separate wiki drifts immediately.

**Misconception: writing docs takes too long.** The expensive part is not writing; it is the repeated questions and the slow onboarding that result from not writing. A runbook that takes an hour to write can save that hour many times over. The trick is to write docs incrementally, as part of the work, rather than as a separate project.

**Misconception: only senior engineers can write good docs.** The best docs are often written by the person who just learned the thing, because they still remember what was confusing. A new hire who writes down the steps they followed to get the service running locally has produced a more useful doc than a senior engineer who has forgotten what it was like not to know.

## The advanced version (once the basics are solid)

Once you have the basic contract model, you can start treating documentation as code. That means versioning it, reviewing it, and testing it.

**Version it with the code.** Keep runbooks and design docs in the same repository as the service they describe. When a pull request changes a deployment manifest, it should also update the runbook that references it. Reviewers should treat a stale doc as a bug.

**Test the commands.** Any command in a runbook should be runnable in a staging environment. Some teams add a small script that extracts fenced code blocks from markdown and runs the safe ones in CI. This catches the most common failure mode: a command that worked six months ago but now references a renamed resource.

**Use a doc linter.** Tools such as `markdownlint` catch formatting issues, and link checkers (for example, `lychee` or `markdown-link-check`) catch references to dashboards, wikis, and internal pages that have moved. A doc with a dead link is a doc that has already started to rot.

**Write decision records.** For architectural decisions, a short Architecture Decision Record (ADR) is more durable than a long design doc, because it captures the context, the decision, and the consequences in a format that is easy to review and easy to supersede. The format is simple: title, status, context, decision, consequences. The value is that a future reader can see not just what was decided but why, and can tell whether the reasons still hold.

**Make docs part of the definition of done.** A feature is not done until the runbook, the README, or the design doc is updated. This is a cultural change more than a technical one, but it is the single highest-leverage change available. If docs are optional, they will be skipped; if they are part of the review checklist, they will be written.

**Measure doc freshness.** A useful proxy: for each doc file, compare the timestamp of its last commit to the timestamp of the last commit touching the code it describes. A doc that has not been touched in a year while the code has changed weekly is probably stale. This is not a perfect metric, but it is a useful signal for where to look. To compute it, run `git log -1 --format=%cs -- <doc-path>` and `git log -1 --format=%cs -- <code-path>` and compare the two dates; a large gap, with the code date newer, flags the doc for review.

## A failure-mode analysis

It helps to name the specific ways survivable docs fail, because each has a different fix.

**The silent rename.** A resource is renamed — a deployment, a Vault path, a dashboard — and the doc still references the old name. The reader follows the doc, gets a "not found" error, and has no way to know whether the doc is wrong or the environment is. *Fix:* test commands in staging and treat a failing command as a broken doc, not a reader error.

**The missing precondition.** The doc assumes a CLI is already authenticated, a context is already set, or a feature flag is already enabled. The reader runs the command and gets a permission error that looks like an outage. *Fix:* state prerequisites explicitly, as the worked example does.

**The unstated success criterion.** "Check it works" is not a check. The reader cannot distinguish success from partial failure. *Fix:* name the endpoint, the field, the log line, or the dashboard panel that confirms success.

**The orphaned decision.** A design doc explains a choice but not the alternatives or the conditions under which the choice should be revisited. A future reader either preserves a decision that no longer makes sense or reverses it without understanding what it protected. *Fix:* record context, decision, and consequences, and state what would justify revisiting it.

**The review gap.** Docs live outside the repo, so code review never sees them. They drift until someone notices. *Fix:* co-locate docs with code and make doc changes part of the same pull request.

## A decision checklist

Before calling a doc finished, check each item. A "no" is a gap a future reader will hit.

- Does it name its audience and the access level that audience needs?
- Does it name the exact environment, cluster, namespace, or account it applies to?
- Does it list prerequisites, including authentication state?
- Does every command appear verbatim, with the expected output next to it?
- Does it define success in terms of something observable, not "it works"?
- Does it include a rollback or recovery path for the risky steps?
- Does it live in the repo next to the code it describes?
- Is it reviewed in the same pull request as the change it accompanies?
- For a decision doc: are the alternatives and the conditions for revisiting stated?

## Frequently Asked Questions

**How do I convince a team to write documentation?** Start with the docs that save the most time: onboarding, runbooks for common alerts, and the README. Make the cost of missing docs visible by tracking how often someone asks a question that a doc could answer. Make doc updates part of code review, not a separate task. The goal is to make the right thing the easy thing.

**What is the minimum viable documentation for a service?** A README that states what the service does, how to run it locally, how to run its tests, and where it is deployed. A runbook for the top three alerts. A link to the design doc or ADR for the major decisions. That is enough to survive most author departures.

**How do I keep documentation from going stale?** Keep it in the repo, review it with the code, and test the commands. The most effective single practice is to require a doc update in any pull request that changes a documented behavior. Staleness is a process problem, not a writing problem.

**Should I use a wiki or a repo?** Repo, for anything that describes code or operations. Wikis are fine for team-level information that does not change with the code, but they are a poor home for runbooks and design docs because they are disconnected from the review process.

## Quick reference

| Element | Personal notebook | Survivable doc |
|---|---|---|
| Audience | Implicit (me) | Explicit (on-call engineer, new hire) |
| Scope | Implicit | Explicit (service, cluster, environment) |
| Steps | Vague ("restart the app") | Exact commands with expected output |
| Failure handling | Missing | Rollback steps and error messages |
| Location | Personal notes, wiki | Repo, next to code |
| Review | None | Part of code review |
| Testing | None | Commands run in CI or staging |

## One thing to do in the next 30 minutes

Open the README of the service you own and read it as if you were a new hire on your first day. Find the first sentence that assumes knowledge the reader does not have — a missing prerequisite, an unexplained command, a vague success criterion — and rewrite that sentence to be verifiable. Commit the change. That single edit is the smallest unit of documentation that survives its author.
