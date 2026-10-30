# Turn AI coding telemetry into a salary negotiation case

AI coding assistants generate a stream of telemetry: session start and end times, which repository was open, which editor was used, how many completions were accepted. Most of that data is accessible through the vendor's API and almost none of it is ever looked at. This article shows how to turn it into a small, honest dataset you can bring to a compensation conversation.

The goal is not to prove that AI did your job. The goal is to make visible work that was previously invisible: the boilerplate, the YAML, the test scaffolding, the migration scripts. If that work has been absorbed by tooling, and the tooling is not free, then the productivity delta is a legitimate input into a pay discussion. It is not a substitute for judgment, architecture, or incident response — those remain human.

## What you actually need

- A recent Node LTS runtime (Node 20.x is a reasonable floor).
- Git.
- An AI coding assistant whose vendor exposes session or usage data. GitHub Copilot has a documented usage API; other assistants expose activity logs in different shapes. If your vendor does not expose anything, you cannot build this and you should stop here rather than estimate.
- A ticketing system you can query (Jira, Linear, GitHub Issues). The script below stubs this out; you will replace the stub with a real call.

What you will build is a CLI that:

1. Pulls the last 90 days of assistant sessions from the vendor API.
2. Maps each session to a ticket using a deterministic rule (branch name, commit message, or repo-to-project mapping).
3. Emits a CSV with one row per ticket: ticket id, wall-clock session hours, a confidence field, and the mapping rule used.

That CSV is the raw material. The interpretation is yours.

## Step 1 — project setup

```bash
mkdir ai-telemetry-report && cd ai-telemetry-report
npm init -y
npm install dotenv@16.3.1 node-fetch@3.3.2 csv-writer@1.4.0
```

Create `.env`:

```
GITHUB_TOKEN=ghp_your_token_here
GITHUB_USER=your_github_username
```

Add strict TypeScript:

```bash
npm install -D typescript@5.4.5 ts-node@10.9.2 @types/node@20.12.2
npx tsc --init
```

Set `"strict": true` in `tsconfig.json`. Timestamp parsing is where off-by-one errors hide, and strict mode catches the ones that matter.

## Step 2 — fetch sessions

Create `src/index.ts`. Note that the GraphQL field names below are illustrative of the *shape* you want (session id, start, end, repository, editor). Check your vendor's current schema before running — API surfaces change and field names differ between products. The pagination logic, error handling, and CSV writing are the parts that transfer.

```typescript
import fs from 'fs'
import { parseArgs } from 'node:util'
import fetch from 'node-fetch'
import { createObjectCsvWriter } from 'csv-writer'

const args = parseArgs({
  options: {
    since: { type: 'string', default: '2026-01-01' },
    until: { type: 'string', default: new Date().toISOString().slice(0, 10) },
  },
})

interface AssistantSession {
  id: string
  startedAt: string
  endedAt: string
  repository: { nameWithOwner: string }
  editor: string
}

interface Page {
  nodes: AssistantSession[]
  pageInfo: { hasNextPage: boolean; endCursor: string | null }
}

async function fetchSessions(
  since: string,
  until: string,
  cursor: string | null = null,
): Promise<AssistantSession[]> {
  const query = `
    query Sessions($since: DateTime!, $until: DateTime!, $cursor: String) {
      user(login: $user) {
        sessions(first: 100, since: $since, until: $until, after: $cursor) {
          nodes {
            id
            startedAt
            endedAt
            repository { nameWithOwner }
            editor
          }
          pageInfo { hasNextPage endCursor }
        }
      }
    }
  `

  const res = await fetch('https://api.github.com/graphql', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${process.env.GITHUB_TOKEN}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      query,
      variables: { since, until, cursor, user: process.env.GITHUB_USER },
    }),
  })

  if (!res.ok) throw new Error(`API error: ${res.status}`)
  const json = (await res.json()) as any
  const page: Page = json.data.user.sessions

  if (!page.pageInfo.hasNextPage) return page.nodes
  const next = await fetchSessions(since, until, page.pageInfo.endCursor)
  return page.nodes.concat(next)
}
```

The pagination recursion is the important part. A single request returns at most 100 nodes; anything beyond that is silently dropped if you ignore `pageInfo`.

## Step 3 — map sessions to tickets

Mapping is where most people cheat. Do not guess. Pick one deterministic rule and record which rule you used, so a reviewer can reproduce the mapping.

A defensible rule: the branch name contains a ticket key (for example `feature/PROJ-1421-add-retry`). Extract the key with a regex, and if there is no match, mark the session as `unmapped` rather than assigning it to a nearby ticket.

```typescript
interface TicketRow {
  ticket: string
  hours: number
  rule: string
}

function extractTicket(branch: string | null): { key: string; rule: string } {
  if (!branch) return { key: 'unmapped', rule: 'no-branch' }
  const match = branch.match(/([A-Z][A-Z0-9]+-\d+)/)
  if (match) return { key: match[1], rule: 'branch-regex' }
  return { key: 'unmapped', rule: 'no-key-in-branch' }
}

function sessionHours(start: string, end: string): number {
  const ms = new Date(end).getTime() - new Date(start).getTime()
  if (Number.isNaN(ms) || ms < 0) return 0
  return ms / 3_600_000
}

function buildRows(
  sessions: AssistantSession[],
  branchFor: (s: AssistantSession) => string | null,
): TicketRow[] {
  const byTicket = new Map<string, TicketRow>()
  for (const s of sessions) {
    const { key, rule } = extractTicket(branchFor(s))
    const row = byTicket.get(key) ?? { ticket: key, hours: 0, rule }
    row.hours += sessionHours(s.startedAt, s.endedAt)
    byTicket.set(key, row)
  }
  return [...byTicket.values()]
}
```

Two things to notice. First, `sessionHours` returns wall-clock time, not "saved time". Wall-clock time includes reading, thinking, waiting for tests, and coffee. Do not relabel it as savings. Second, sessions with no ticket key land in an `unmapped` bucket, which is honest and also tells you how good your mapping rule is.

## Step 4 — write the CSV

```typescript
async function writeCSV(rows: TicketRow[], since: string, until: string) {
  const writer = createObjectCsvWriter({
    path: `report_${since}_${until}.csv`,
    header: [
      { id: 'ticket', title: 'TICKET' },
      { id: 'hours', title: 'SESSION_HOURS' },
      { id: 'rule', title: 'MAPPING_RULE' },
    ],
  })
  await writer.writeRecords(rows)
}

;(async () => {
  const sessions = await fetchSessions(args.values.since, args.values.until)
  const rows = buildRows(sessions, () => null) // replace with real branch lookup
  await writeCSV(rows, args.values.since, args.values.until)
  console.log(`Wrote ${rows.length} rows`)
})()
```

Run it:

```bash
npx ts-node src/index.ts --since 2026-01-01 --until 2026-03-31
```

## Step 5 — retries and rate limits

Vendor APIs rate-limit. Handle 429 with the `x-ratelimit-reset` header rather than a fixed sleep, and cap retries so a persistent failure surfaces instead of hanging.

```typescript
import { setTimeout as sleep } from 'timers/promises'

async function safeFetch(url: string, opts: any, retries = 3): Promise<any> {
  const res = await fetch(url, opts)
  if (res.ok) return res

  if (res.status === 429 && retries > 0) {
    const reset = Number(res.headers.get('x-ratelimit-reset') ?? '5')
    const waitMs = Math.max(reset * 1000 - Date.now(), 1000)
    await sleep(waitMs)
    return safeFetch(url, opts, retries - 1)
  }

  if (res.status >= 500 && retries > 0) {
    await sleep(2 ** (4 - retries) * 1000)
    return safeFetch(url, opts, retries - 1)
  }

  throw new Error(`HTTP ${res.status}`)
}
```

Note the difference from a naive retry: a 429 waits until the reset time the server gave you; a 5xx uses exponential backoff. Mixing these up wastes your quota.

## Step 6 — a test that would actually catch a bug

Testing that an empty response returns an empty array is fine but shallow. Test the arithmetic and the mapping rule, because those are where wrong numbers enter the report.

```typescript
import { extractTicket, sessionHours } from '../src/index'

describe('sessionHours', () => {
  it('computes 1.5 hours from a 90 minute gap', () => {
    expect(sessionHours('2026-01-01T10:00:00Z', '2026-01-01T11:30:00Z')).toBeCloseTo(1.5)
  })

  it('returns 0 for inverted timestamps', () => {
    expect(sessionHours('2026-01-01T11:30:00Z', '2026-01-01T10:00:00Z')).toBe(0)
  })
})

describe('extractTicket', () => {
  it('pulls a ticket key from a branch name', () => {
    expect(extractTicket('feature/PROJ-1421-add-retry')).toEqual({
      key: 'PROJ-1421',
      rule: 'branch-regex',
    })
  })

  it('marks unmapped when no key is present', () => {
    expect(extractTicket('main')).toEqual({ key: 'unmapped', rule: 'no-key-in-branch' })
  })
})
```

Run with coverage:

```bash
npx jest --coverage --coverageReporters=text
```

Coverage percentage is not the point. The point is that the two functions producing your negotiation numbers have tests pinning their behaviour.

## How to measure the productivity delta honestly

The tool above gives you session hours per ticket. That is not the same as hours saved. To get from one to the other, you need a before/after measurement, and the honest way to do it is a controlled comparison:

1. Pick a class of ticket (for example, "add a new REST endpoint").
2. Find 10+ historical tickets of that class completed before assistant adoption. Record cycle time from first commit to merge.
3. Find 10+ tickets of the same class completed after adoption. Record the same metric.
4. Compare medians, not means — outliers dominate small samples.
5. Report the delta with the sample sizes next to it.

If you cannot assemble both samples, you do not have a delta. You have a session count. Say so.

What you should instrument:

- Cycle time per ticket (first commit → merge), from your git host.
- Session hours per ticket, from the script above.
- Rework rate (commits after review approval that touch the same lines), to check you are not trading speed for churn.

What you should not do: multiply session hours by an hourly rate and call the product "savings". Session hours are time the assistant was open, not time you would otherwise have billed.

## Worked example (illustrative numbers)

Suppose over a quarter you collect:

- 40 tickets of class "add endpoint", pre-adoption median cycle time: 6.0 days.
- 40 tickets of class "add endpoint", post-adoption median cycle time: 4.5 days.
- Median session hours per ticket post-adoption: 3.0.

The cycle-time delta is 1.5 days per ticket. Over 40 tickets that is 60 ticket-days of cycle time, but cycle time is elapsed, not effort — a chunk of that is queue time, not your time. To convert to effort you need a second measurement: how many hours per day you actually touched those tickets. Suppose your commit timestamps show 2.0 hours of active work on a typical pre-adoption ticket and 1.4 hours post-adoption. That is 0.6 hours of effort delta per ticket, or 24 hours across 40 tickets.

Twenty-four hours is a real number. It is not "£10k". If your fully-loaded hourly cost is £60, the effort delta is £1,440 — a defensible figure you can put in front of a manager, with the method attached. The temptation to inflate it is exactly what gets the whole exercise dismissed.

## Failure modes to watch for

**Timezone drift.** APIs return ISO timestamps with offsets; some tools return naive local times. If you mix the two, session durations can be off by hours. Always parse with `new Date(iso)` and check `Number.isNaN` before trusting the result. Never hand-add an offset unless the API documents that it omits one.

**Session fragmentation.** A single logical task often spans several sessions across repos or editors. Summing per-session and then attributing each to a ticket will over-count if the same ticket appears in multiple repos. Aggregate by ticket key, not by session.

**Shared credentials.** If your team shares one API token, your "personal" telemetry includes everyone else's sessions. Use a personal token, or filter by the `user` field in the response.

**Mapping leakage.** A loose regex will pull ticket keys out of commit messages that mention unrelated issues. Keep the rule strict and count the `unmapped` bucket; if it is large, your mapping is wrong, not your productivity.

**Survivorship bias in the before/after sample.** If you only compare tickets that shipped, you ignore the ones that were abandoned — and abandonment rates may differ between the two periods.

## A decision checklist before you bring this to a manager

- Can you state the measurement method in two sentences without hedging?
- Are your sample sizes large enough that a median is meaningful (rule of thumb: at least 10 per group)?
- Have you separated elapsed time from effort time?
- Have you kept session hours and saved hours in distinct columns?
- Can a colleague reproduce your CSV from your script and your mapping rule?
- Do you have a proposal that is about scope and impact, not just "AI made me faster"?
- Are you prepared for the counter-argument that the assistant is a company-provided tool, and therefore the productivity accrues to the company? (A reasonable response: the tool is an input; the judgment about where to apply it is the skill being compensated.)

## Common questions

**What if my employer does not expose assistant telemetry?**
Then you cannot build the session-hours column. You can still build the cycle-time column from git history, which is often the more persuasive metric anyway, because it is independent of any AI vendor.

**Should I show the raw CSV in the meeting?**
Show the summary and the method. Keep the raw CSV available as an appendix. The method is what survives scrutiny; the raw file invites arguments about individual rows.

**What if the assistant is company-licensed?**
The licence is a cost the company already chose to pay. The relevant question is whether the productivity delta changes your scope or your level, not whether you personally bought the tool.

**How do I handle a role in a different cost-of-living market?**
Anchor to the market you are hired into, then present the productivity data as supporting evidence for a level or scope adjustment, not as a multiplier on a local rate.

## Do this in the next 30 minutes

Pick one ticket class you worked on in the last quarter, open your git host's UI, and record the cycle time (first commit to merge) for five pre-adoption and five post-adoption tickets of that class. Paste them into a spreadsheet and compute both medians. If the medians differ, you have the beginning of a real, reproducible argument. If they do not, you have saved yourself from making a claim you cannot defend.
