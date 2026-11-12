# 7 ADR traps that cause stale decisions

Architecture decision records (ADRs) are meant to be the memory of a codebase: a short, dated record of a significant choice, the forces behind it, and what it costs. In practice, many ADR logs become a graveyard of records that are forgotten, contradicted, or silently reversed. The difficulty is not writing an ADR; it is writing one that stays trustworthy two years later.

This article covers seven specific traps that turn ADRs from a useful tool into documentation debt. Each trap gets the same treatment: what it is, why it feels attractive, how it fails, and a concrete fix. It closes with a prioritisation order, a minimal template, and a 30-minute audit.

## 1. The template trap: a format so heavy nobody fills it in

**What it is.** A structured template for capturing decisions, usually with sections like Context, Decision, Status, and Consequences. Michael Nygard's original ADR format is deliberately minimal. Some teams replace it with templates of a dozen or more fields, hoping to capture every nuance.

**Why it looks attractive.** A well-designed template enforces consistency and makes decisions comparable across time. It forces the author to articulate the problem and the trade-offs, which is half the work.

**How it fails.** Overly detailed templates become a barrier to writing at all. Authors procrastinate because filling in fifteen sections feels like writing a full design document. The decision goes undocumented, or the record is padded with boilerplate that adds nothing. The ADR becomes a bureaucratic exercise rather than a communication tool.

**When it applies.** Teams new to ADRs that want guardrails. A minimal template — title, status, context, decision, consequences — is usually enough.

**When it fails.** When the template demands information nobody has at decision time (for example, a precise cost projection for an option that was rejected on qualitative grounds), or when it duplicates a design document. The ADR should be a lightweight index into the reasoning, not the whole story.

**Fix.** Start with the minimal format and add a section only after the same gap has bitten you more than once. If a field is routinely filled with "N/A" or "see design doc", delete the field.

## 2. The status trap: decisions stuck in "Proposed"

**What it is.** A lifecycle field — proposed, accepted, rejected, deprecated, superseded — that tells a reader whether a decision is binding or historical. This is core to the ADR concept.

**Why it looks attractive.** Status makes it immediately clear which decisions are in force. It also gives a clean way to handle reversals: the old record is not deleted, it is marked superseded by a new one.

**How it fails.** Statuses are forgotten. A record stays "proposed" long after the decision shipped, or stays "accepted" after the team quietly reversed it. Without discipline the field becomes actively misleading, which is worse than having no field: a reader who trusts a wrong status makes a wrong assumption about the system.

**When it applies.** Every ADR needs a status. It is the single most important piece of metadata in the file.

**When it fails.** When no one owns the log and no process touches statuses. Drift is the default state, not an exception.

**Fix.** Pick a lightweight review cadence — quarterly is common — and audit statuses. A script that lists every ADR whose status is "proposed" and whose file is older than 30 days is enough to prompt action. Treat "proposed" as an open question with an owner, not a resting state.

## 3. The orphan trap: storing ADRs where nobody looks

**What it is.** ADRs need a home: a `/docs/adr` directory in the repository, a wiki, a dedicated tool, or a project management system.

**Why it looks attractive.** Keeping ADRs in the repository version-controls them alongside the code they describe. Developers encounter them while browsing, and a pull request can include the ADR that justifies it.

**How it fails.** If ADRs live in a separate wiki or tool, they are quickly orphaned: developers forget they exist and the wiki becomes a graveyard. Even in-repo ADRs are overlooked if the directory is not linked from the README. A new developer who cannot find the decision log within a few minutes will not look again.

**When it applies.** In-repo is generally best for engineering decisions that affect the codebase. Cross-team or organisational decisions may need a shared location, but that location should still be linked from the relevant repositories.

**When it fails.** When the location is undiscoverable, or when there are two locations and neither is authoritative.

**Fix.** Start with `/docs/adr` and add a prominent link in the README. A command-line helper that scaffolds the directory structure and numbering (the `adr-tools` family of scripts is one example) reduces the friction of creating a record correctly.

## 4. The numbering trap: inconsistent IDs that break references

**What it is.** ADRs are typically numbered sequentially (0001, 0002, …) so they can be cited from commit messages, code comments, and other ADRs ("Supersedes ADR-0005").

**Why it looks attractive.** Sequential numbers are a stable, human-readable identifier and easy to say out loud in a review.

**How it fails.** If numbers are assigned ad hoc by different people, collisions and gaps appear. If a record is deleted or renumbered, every reference to it breaks. Date-based or UUID-based schemes are unique but much less pleasant to cite.

**When it applies.** Any team that references ADRs from other artefacts. Consistency matters more than the scheme.

**When it fails.** When there is no single source of truth for the next number. Manual numbering under concurrency — two people opening pull requests the same afternoon — produces collisions.

**Fix.** Generate ADRs with a tool that assigns the next number, or, if numbering manually, take the highest existing number plus one and check it in code review. Never renumber an existing record; if it was wrong, supersede it.

## 5. The context trap: writing the decision without the problem

**What it is.** The Context section explains the forces at play: the problem, the constraints, and the alternatives considered. It is what makes an ADR useful to someone who was not in the room.

**Why it looks attractive.** A good context section stops future readers from re-litigating a decision without understanding why it was made, and it helps new team members get up to speed.

**How it fails.** Authors skip context because it feels obvious at the time. The record says "we will use PostgreSQL" without saying what problem that solved or which alternatives lost. Months later nobody remembers, and the record cannot answer the only question a reader has: *should this still be true?*

**When it applies.** Every ADR. Even a seemingly trivial decision benefits from a sentence or two of context.

**When it fails.** When context is too vague ("we need a database") or too detailed (the entire design document pasted in). The useful middle is a few paragraphs that capture the essential trade-offs and at least one rejected alternative with the reason it lost.

**Fix.** Review ADRs in pull requests and ask, "What was the alternative, and why did it lose?" Add a checklist item: "Does the Context section name the problem and at least one alternative?"

## 6. The workflow trap: no trigger in the development process

**What it is.** ADRs work best when they are part of the workflow: created when a significant decision is made, reviewed alongside the code, and used in onboarding.

**Why it looks attractive.** Integrating ADRs into pull requests means they are written at the right time and reviewed by peers. Linking from code comments helps future maintainers.

**How it fails.** Without integration, ADRs are an afterthought. Teams write them too late, after the decision is already implemented and the alternatives are no longer recoverable in detail, or not at all, because nothing triggers the act of writing.

**When it applies.** Any team that wants ADRs to be more than a compliance exercise.

**When it fails.** When the process is too heavy. Requiring an ADR for every minor decision produces fatigue and then blanket exemption.

**Fix.** Define what counts as significant — choosing a framework, changing a data model, adopting a new external service, changing an authentication scheme — and add a pull-request checklist item: "Does this change require an ADR?" For decisions made in meetings, name a scribe who files the record within a day.

## 7. The staleness trap: never revisiting or superseding

**What it is.** ADRs are meant to be immutable: once accepted, they are not edited. When the decision changes, a new ADR supersedes the old one, preserving the history of how thinking evolved.

**Why it looks attractive.** Immutability gives an audit trail. You can see why things are the way they are, and why they used to be different.

**How it fails.** Teams forget to supersede. A record stays "accepted" long after the team moved on. The log then contradicts reality, and once a reader has been burned by one wrong record they stop trusting all of them.

**When it applies.** Whenever a decision is reversed or materially modified.

**When it fails.** When there is no trigger to revisit. Without a periodic review, stale records accumulate silently.

**Fix.** Schedule a review (quarterly is a reasonable default). For each "accepted" record ask: is this still true? If not, write a new ADR that supersedes it and update the old one's status and a "superseded by" pointer. Listing records by last-modified date is a crude but effective way to spot ones nobody has touched in a year.

## A worked example: what a stale record costs

Consider a common sequence in a service that started small.

1. A team chooses a single relational database for a new service and writes ADR-0003: "Use PostgreSQL as the primary datastore", status accepted. The context notes that the expected write volume is low and that operational simplicity matters more than horizontal scale.
2. Two years later, write volume grows and the team shards the service across three databases. No new ADR is written; the change is visible only in code and infrastructure configuration.
3. A new engineer reads ADR-0003, sees "accepted", and concludes the system has a single datastore. They design a reporting query that joins across what they believe is one database. It fails in staging, and they spend a day discovering that the architecture is different from the documented one.

The cost here is not the day of debugging. It is the loss of trust: the engineer now treats the ADR log as unreliable and stops reading it, which makes every subsequent record less useful.

The fix is cheap. When the sharding decision was made, a short ADR-0017 would have recorded the trigger (write volume exceeded what a single instance handled acceptably), the options (vertical scaling, read replicas, sharding), the chosen path, and the consequences (cross-shard joins are no longer possible; reporting must go through a separate pipeline). ADR-0003 would then be marked "superseded by ADR-0017". A reader arriving at either file lands on the current truth in one hop.

## How to prioritise which traps to fix first

The traps are not equally damaging. If the log is new or neglected, fix them in this order:

| Priority | Trap | Why it ranks here | First concrete move |
|---|---|---|---|
| 1 | Orphan (discoverability) | If nobody can find the records, nothing else matters | Move to `/docs/adr`, link from README |
| 2 | Status | A wrong status is worse than no status | Audit "proposed" records older than 30 days |
| 3 | Context | Poor context makes records useless to future readers | Add a review question about rejected alternatives |
| 4 | Workflow | Without a trigger, records are written late or never | Add an ADR checklist item to pull requests |
| 5 | Staleness | Stale records erode trust in the whole log | Schedule a quarterly supersession review |
| 6 | Template | A heavy template slows writing but is recoverable | Start minimal; add fields only when needed |
| 7 | Numbering | Annoying, but rarely blocks anyone | Use a generator or enforce next-integer in review |

This order reflects day-to-day usability. A discoverable, current log with thin context is far more valuable than a beautifully formatted log that is out of date.

## A minimal template to adopt

If you are starting from scratch, a short template like the following is enough. It is based on Nygard's format.

```markdown
# ADR-0001: Use PostgreSQL for the primary datastore

## Status

Accepted

## Context

The order management service needs a relational database with transactional
guarantees for order state changes. The team already operates PostgreSQL in
production. Alternatives considered: MySQL (less operational familiarity on
the team), a document store (order state changes span multiple entities and
benefit from multi-row transactions).

## Decision

Use PostgreSQL as the primary datastore for the order service.

## Consequences

- Positive: existing operational runbooks and backup tooling apply.
- Negative: the database is stateful; failover and backup restore must be
  tested, not assumed.
- Neutral: an access layer (ORM or query builder) still has to be chosen.
```

Two properties matter more than the section list. First, the Context names at least one alternative and why it lost. Second, the Consequences include something negative; a record with only positive consequences is usually a record that has not been thought through.

## What to instrument if you want to know whether your log is healthy

Claims about ADR health are easy to make and hard to verify, so measure instead of asserting. Three cheap signals, all computable from the repository:

- **Discoverability:** time how long it takes someone unfamiliar with the repo to find the decision log from a fresh clone. If it takes more than a few minutes or requires asking, the README link is missing or buried.
- **Status drift:** count records whose status is "proposed" and whose file has not changed in 30 days, and records whose status is "accepted" but which reference technology no longer present in the dependency manifest. Both counts should trend toward zero after a review.
- **Supersession coverage:** for each record that describes a component later rewritten or replaced, check whether a superseding record exists. Sampling ten recent major changes is usually enough to see the pattern.

None of these require a special tool; a few lines of `grep` and `find` over the ADR directory, run on a schedule, produce the numbers.

## Frequently asked questions

**What is an architecture decision record?**
A short text file that captures a significant decision about a system's architecture: the context, the decision, and its consequences. It is stored alongside the code and versioned with it, giving a historical record of why the system looks the way it does. Records are treated as immutable; when a decision changes, a new record supersedes the old one.

**How do you get a team to write them?**
Start with one record for a recent decision and show that it answers a question that would otherwise require digging through chat history. Add a pull-request checklist item rather than a mandate. Avoid requiring an ADR for every decision; scope it to significant ones so the requirement stays credible.

**Where should ADRs live?**
In the code repository, typically under `/docs/adr`, linked from the README. That keeps them versioned with the code and discoverable. Cross-team decisions may also be published to a shared location, but the repository copy should be authoritative to avoid two divergent versions.

**How often should the log be reviewed?**
Quarterly works for most teams. Check each accepted record for continued accuracy, supersede the ones that no longer hold, and chase records stuck in "proposed". The goal is a log that can be trusted, not a review ritual for its own sake.

**Should an ADR ever be edited?**
Fix typos and broken links freely. Do not rewrite the decision or its context after the fact; that destroys the audit trail. If the decision changed, write a new record and mark the old one superseded.

## Your next step: audit the log in 30 minutes

From the root of the repository, list every record still marked as proposed:

```bash
grep -rl '^## Status' docs/adr --include='*.md' | while read -r f; do
  status=$(awk '/^## Status/{getline; getline; print; exit}' "$f" | tr -d '[:space:]')
  [ "$status" = "Proposed" ] && echo "$f"
done
```

For each file it prints, make one decision: promote it to accepted, mark it rejected, or supersede it with a new record. Then open the README and confirm there is a link to `docs/adr`. If there is no ADR directory yet, create one and write a single record for the most recent significant decision the team made — context, decision, consequences, including at least one negative consequence.
