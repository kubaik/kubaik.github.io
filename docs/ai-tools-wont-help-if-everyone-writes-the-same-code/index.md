# AI tools won’t help if everyone writes the same code

## The problem: AI amplifies whatever your team already rewards

AI coding assistants do not create new team dynamics; they amplify existing ones. A team that already rewards raw output over verification will find that AI makes the output side cheaper and the verification side more expensive. A team that already has a small group of senior engineers doing most of the review will find that group absorbing an even larger share of the load.

The failure mode is consistent enough to describe generically. A subset of engineers adopts AI assistance and ships features quickly. Because their throughput is visible and their defects are not yet visible, they accumulate social capital. Meanwhile, the engineers who write tests, document interfaces, and wait for CI before merging become the de facto cleanup crew. Review comments shift from technical critique to process complaints. Review latency rises even though individual diffs get smaller. Eventually the reviewers disengage, and the team has two classes of contributors: those who generate code and those who are accountable for it.

This is not a tooling problem, and it is not solved by banning or mandating AI. It is a measurement and incentive problem. The rest of this article describes a concrete approach: pick one metric that captures real-world impact, instrument it cheaply, and make the feedback private to each engineer while keeping the aggregate visible.

## Why policy-first approaches fail

Three interventions are common and reliably fail. Understanding why is useful before designing anything.

**Blanket review mandates.** A rule like "all AI-assisted code must be reviewed by a human before merge" sounds reasonable. In practice it relocates work rather than reducing it. Engineers who are optimizing for speed will mark a pull request ready for review and let someone else do the cleanup. The review queue grows, the same senior people absorb the load, and review comments become meta-commentary about process instead of substance. The mandate changes nothing about who bears the cost.

**Automated AI detection.** Tools that flag AI-generated diffs by inspecting commit metadata, watermarking tokens, or stylistic heuristics are brittle by construction. Any signal embedded in the commit can be stripped. Any heuristic based on naming patterns or formatting produces false positives on legitimate code. A false positive that blocks a hotfix during an incident is expensive in a way that is hard to recover from socially—the tool becomes something people route around rather than something they trust.

**Senior mentorship programs.** Asking senior engineers to coach others on responsible AI use fails on timing. Coaching happens after the code is already in staging and the on-call rotation is already paged. The mentor absorbs the cost of someone else's speed, and burnout follows. When the mentors quietly stop reviewing AI-heavy changes, the two-class split becomes permanent.

The common thread: all three approaches treat AI use as a binary to be policed. None of them change the underlying incentive, which is that shipping fast is rewarded and cleaning up is not.

## The metric that changes behavior: incident rate per author

The intervention that tends to work is unglamorous. Stop measuring AI usage. Start measuring outcomes per person.

Pick one metric that everyone already agrees is broken. The most useful candidate in most teams is **incident count attributable to an author within a fixed window after their code merges**. Thirty days is a reasonable default; adjust based on your deploy cadence and how quickly incidents surface.

Why incidents rather than coverage, PR count, or lines changed:

- Coverage is trivially gameable by AI-generated tests that assert nothing meaningful.
- PR count rewards fragmentation.
- Lines changed rewards verbosity and punishes refactoring.
- Incidents are expensive, hard to fake, and directly connected to customer impact.

The metric does not need to be perfect. Attribution is genuinely hard when multiple authors touch a service, when incidents stem from configuration rather than code, or when a defect ships from an old change. Accept a coarse signal. The goal is not a performance review artifact; it is a feedback loop that makes consequences visible at the moment of decision.

### Defining the metric precisely

Before building anything, write down the definition. A workable starting point:

- **Numerator:** count of production incidents in the trailing 30 days where the author of the most recent change to the implicated code path is identified as the responsible party.
- **Denominator:** count of merged pull requests by that author in the same window, or simply report the raw count if PR volume is similar across the team.
- **Exclusions:** incidents caused by third-party outages, infrastructure changes not tied to a code merge, and incidents where attribution is genuinely ambiguous.

Record the exclusions in writing. Ambiguity in the definition is where trust in the metric erodes.

## Instrumentation: what to collect and from where

The data you need already exists in most organizations. You are joining two systems that were never designed to talk to each other.

**From your source control host:** merged pull requests in the trailing window, with author identity, merge timestamp, additions, deletions, and the list of files touched. Both GitHub's GraphQL API and GitLab's REST API expose all of this. Use the API rather than scraping the web UI; rate limits are documented and stable.

**From your incident management system:** incidents in the same window, with the responder or assignee identity and creation timestamp. PagerDuty, Opsgenie, and similar tools all expose this via documented REST endpoints.

**The join key** is a stable identity. Email address is the usual choice, but it only works if the same address is used across both systems. Verify this before building; mismatched identities produce a dashboard that silently under-counts.

**Storage:** a single table keyed by author and period is sufficient. Any managed key-value or relational store works. Do not over-invest here.

**Compute:** a scheduled job that runs every few hours and writes one row per author per period. A serverless function on a timer is adequate. The workload is small: a few hundred API calls and a few hundred writes per run.

**Presentation:** a dashboard with per-author views. The critical design decision is access control, discussed below.

### A minimal handler

The following illustrates the shape of the job. It fetches merged pull requests and incidents for each team member and writes one summary row per person. Error handling is intentionally left as an exercise; in production, wrap each author's processing in a try/except so one failure does not abort the run.

```python
import os
from datetime import datetime, timedelta, timezone

import boto3
import requests

TABLE_NAME = os.environ["METRICS_TABLE"]
GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
GITHUB_ORG = os.environ["GITHUB_ORG"]
PAGERDUTY_TOKEN = os.environ["PAGERDUTY_TOKEN"]
WINDOW_DAYS = 30

dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table(TABLE_NAME)


def fetch_merged_prs(login, cutoff):
    """Return merged PRs authored by `login` since `cutoff`."""
    query = """
    query($org: String!, $cursor: String) {
      organization(login: $org) {
        repositories(first: 50, after: $cursor) {
          pageInfo { hasNextPage endCursor }
          nodes {
            name
            pullRequests(first: 50, states: MERGED, orderBy: {field: UPDATED_AT, direction: DESC}) {
              nodes {
                number
                title
                mergedAt
                additions
                deletions
                author { login }
              }
            }
          }
        }
      }
    }
    """
    headers = {"Authorization": f"bearer {GITHUB_TOKEN}"}
    results = []
    cursor = None

    while True:
        response = requests.post(
            "https://api.github.com/graphql",
            json={"query": query, "variables": {"org": GITHUB_ORG, "cursor": cursor}},
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()["data"]["organization"]["repositories"]

        for repo in payload["nodes"]:
            for pr in repo["pullRequests"]["nodes"]:
                merged_at = datetime.fromisoformat(pr["mergedAt"].replace("Z", "+00:00"))
                if merged_at < cutoff:
                    continue
                if pr["author"] and pr["author"]["login"] == login:
                    results.append(
                        {
                            "repo": repo["name"],
                            "number": pr["number"],
                            "additions": pr["additions"],
                            "deletions": pr["deletions"],
                            "merged_at": merged_at.isoformat(),
                        }
                    )

        if not payload["pageInfo"]["hasNextPage"]:
            break
        cursor = payload["pageInfo"]["endCursor"]

    return results


def fetch_incidents(email, since):
    """Return incidents assigned to `email` since the given ISO timestamp."""
    headers = {
        "Authorization": f"Token token={PAGERDUTY_TOKEN}",
        "Accept": "application/vnd.pagerduty+json;version=2",
    }
    params = {"since": since, "until": datetime.now(timezone.utc).isoformat()}
    response = requests.get(
        "https://api.pagerduty.com/incidents",
        headers=headers,
        params=params,
        timeout=30,
    )
    response.raise_for_status()
    incidents = response.json()["incidents"]
    return [i for i in incidents if email in [a.get("email") for a in i.get("assignments", [])]]


def lambda_handler(event, context):
    cutoff = datetime.now(timezone.utc) - timedelta(days=WINDOW_DAYS)
    since = cutoff.isoformat()

    ssm = boto3.client("ssm")
    params = ssm.get_parameters_by_path(
        Path="/team/emails", Recursive=True, WithDecryption=True
    )["Parameters"]

    period = cutoff.strftime("%Y-%m")

    for param in params:
        email = param["Value"]
        login = email.split("@")[0]  # replace with an explicit mapping if needed

        try:
            prs = fetch_merged_prs(login, cutoff)
            incidents = fetch_incidents(email, since)
        except Exception as exc:  # noqa: BLE001 - log and continue
            print(f"skipping {email}: {exc}")
            continue

        total_lines = sum(p["additions"] + p["deletions"] for p in prs)
        avg_pr_size = total_lines / len(prs) if prs else 0

        table.put_item(
            Item={
                "author": email,
                "period": period,
                "incident_count": len(incidents),
                "pr_count": len(prs),
                "avg_pr_size": round(avg_pr_size, 1),
                "computed_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    return {"statusCode": 200}
```

Two notes on this code. First, deriving the source-control login from the email local part is a shortcut that will break for anyone whose login differs from their email prefix; maintain an explicit mapping instead. Second, the GraphQL query above fetches a bounded page of repositories and pull requests for illustration. In production you would paginate pull requests within each repository and filter server-side by author and merge date where the API supports it.

### The weekly summary

A scheduled message that posts aggregate, anonymized movement to a team channel keeps the metric present without putting individuals on the spot. The following posts the five largest incident counts without naming authors.

```javascript
const { App } = require("@slack/bolt");
const { DynamoDBClient, ScanCommand } = require("@aws-sdk/client-dynamodb");

const app = new App({
  token: process.env.SLACK_BOT_TOKEN,
  signingSecret: process.env.SLACK_SIGNING_SECRET,
});

const ddb = new DynamoDBClient({ region: process.env.AWS_REGION });

app.message("weekly-metrics", async ({ say }) => {
  const result = await ddb.send(
    new ScanCommand({ TableName: process.env.METRICS_TABLE, Limit: 200 })
  );

  const items = (result.Items || []).map((item) => ({
    incidents: Number(item.incident_count.N),
    prCount: Number(item.pr_count.N),
    avgPrSize: Number(item.avg_pr_size.N),
    period: item.period.S,
  }));

  const totalIncidents = items.reduce((sum, i) => sum + i.incidents, 0);
  const totalPrs = items.reduce((sum, i) => sum + i.prCount, 0);
  const overallAvgSize = totalPrs
    ? items.reduce((sum, i) => sum + i.avgPrSize * i.prCount, 0) / totalPrs
    : 0;

  const withIncidents = items.filter((i) => i.incidents > 0).length;

  await say(
    [
      `*Weekly engineering metrics* (${items[0]?.period ?? "n/a"})`,
      `• Merged PRs: ${totalPrs}`,
      `• Incidents attributed: ${totalIncidents}`,
      `• Contributors with at least one incident: ${withIncidents} of ${items.length}`,
      `• Mean PR size: ${overallAvgSize.toFixed(0)} lines`,
    ].join("\n")
  );
});

(async () => {
  await app.start(process.env.PORT || 3000);
})();
```

## The design decision that matters most: private detail, public aggregate

The difference between a metric that improves a team and one that poisons it is who sees what.

**Per-author detail should be visible only to that author.** Each engineer can open a view showing their own incident count, PR count, and average PR size over time. Nobody else can see the breakdown. This converts the metric from a ranking into a mirror. The question an engineer asks themselves shifts from "am I beating my colleagues" to "is my own trend moving the right direction."

**Aggregates should be visible to everyone.** Total incidents, total merged PRs, distribution of PR sizes, and the spread between the highest and lowest contributors. Aggregates let the team see whether the overall system is improving without exposing individuals.

**Nothing should be pushed to individuals automatically.** No direct messages, no mentions in a channel, no weekly email with a personal score. Push notifications turn a reflective tool into a surveillance system, and the first thing people do with a surveillance system is route around it.

This combination is what makes the feedback loop work. Consequences become visible at the moment of decision—an engineer about to merge an AI-assisted change knows their name is attached to whatever happens next—without the social cost of public comparison.

## Pairing the metric with a lightweight rule

Metrics alone change attention; they do not change behavior reliably. A single enforceable rule, tightly coupled to the metric, closes the gap.

A workable example: **every pull request description must contain a test plan, marked with a recognizable prefix.** The content can be minimal. The requirement is that it exists and is specific enough that a reviewer can tell whether it was written after the change or before it.

Enforce it with a CI check rather than human vigilance. A small script that reads the pull request body and fails the check if the marker is absent is enough. This removes the rule from the realm of social enforcement, where it would be applied unevenly, and puts it in the build pipeline, where it is applied identically to everyone.

The rule should be tied to the metric. If the metric is incidents per author, the rule should be something that plausibly reduces incidents. "Test plan in the description" qualifies. "Be careful with AI" does not, because it is not checkable and creates no shared expectation.

## Failure modes to watch for

**Gaming by fragmentation.** When average PR size becomes visible, engineers may split large changes into many small ones to move the number. The metric improves; the underlying risk does not. Watch the PR count alongside the average size. If the count rises sharply while the average falls, the metric is being gamed. The remedy is not a new rule but a conversation about what the metric is for.

**Attribution disputes.** The first time someone is told an incident is theirs, they will disagree. Have a written definition of attribution and a documented process for contesting it. If attribution is arbitrary, the metric loses credibility within weeks.

**Metric fixation.** A team that optimizes incident count will eventually under-ship. Pair the incident metric with a throughput signal—merged PRs, deployment frequency, or lead time—so that the two are read together. A team with zero incidents and zero deploys is not healthy.

**Timezone and handoff effects.** In distributed teams, incidents are often discovered by someone other than the author. If your incident system assigns based on who responded rather than who wrote the code, the metric will attribute incidents to the on-call engineer. Fix the join before you trust the number.

**Silent disengagement.** If per-author data leaks into performance reviews without warning, engineers will stop trusting the tool and stop using it honestly. Decide in advance what the data is and is not used for, and say so explicitly.

## How to measure this on your own team

You do not need a dashboard to start. You need one query and one honest conversation.

**Step 1.** Export the last 30 days of merged pull requests from your source control host, including author and merge date. Most hosts offer a CSV export or a one-line API call.

**Step 2.** Export the same window of incidents from your incident management tool, including the responder identity.

**Step 3.** Join the two on email address in a spreadsheet. Count incidents per author.

**Step 4.** Sort by incident count and compute the ratio between the highest and lowest contributors. If the gap exceeds roughly 2×, you have a measurable imbalance worth addressing.

**Step 5.** Share the aggregate distribution with the team—not the per-person breakdown—and ask whether the current review and testing practices are producing the outcomes people expect.

The number itself is not the point. The point is that the conversation moves from opinion to evidence, and that the evidence is about outcomes rather than tooling choices.

## What to do in the next 30 minutes

Open your incident management tool, export the last 30 days of incidents as CSV, and open the result in a spreadsheet. Add a column for the author of the most recent change to the implicated code path, using your source control history. Count incidents per author. Sort descending. If the top contributor has more than twice the incidents of the median, you have found the metric worth instrumenting—and the first conversation worth having.
