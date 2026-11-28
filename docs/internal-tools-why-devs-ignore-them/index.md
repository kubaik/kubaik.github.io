# Internal tools: why devs ignore them

Internal tools fail for a boring reason: they're built for the person who requested them, not the person who has to use them. The request usually arrives as a ticket — "we need a dashboard for X" — and gets built as a form with a submit button. Then the same team that asked for it keeps using curl and a spreadsheet, and the tool becomes a graveyard of good intentions.

The problem isn't engineering quality. It's that internal tools are judged on adoption, not on features, and adoption is governed by a few unglamorous properties: latency, discoverability, failure behavior, and whether the tool lets a developer stay in their terminal. This post covers how to design for those properties, with concrete patterns you can apply to a tool you already own.

## The real problem: internal tools compete with `curl` and a shell history

Public-facing products compete with other products. Internal tools compete with the path of least resistance that already exists. A developer who needs to check a feature flag can grep the codebase, call the flag service's API with `curl`, or open the internal dashboard. If the dashboard takes four seconds to load and requires a VPN plus SSO re-auth, the developer will use `curl` and never mention the dashboard again.

This means the bar for an internal tool is not "better than nothing." It's "faster than the workaround the user already has." That's a much higher bar, and it changes what you build first. A CLI that wraps the same API as the dashboard will often get more usage than the dashboard, because it composes with the shell. A read-only API endpoint with good docs can beat a UI entirely for the audience that reads docs.

The design consequence: before writing UI code, find out what the current workaround is and how long it takes. If you can't beat it, don't build. If you can, build the thinnest thing that beats it and ship that.

## Prerequisites: what you need before writing UI code

You need three things before the first line of frontend code:

1. **A stable API contract.** The tool's UI should be a client of an API that also serves the CLI and any automation. If the UI talks directly to the database or to internal services through a bespoke backend-for-frontend, you've coupled the tool to one interface and made the CLI a second-class citizen.
2. **An identity story that doesn't require a new login.** If your org uses an SSO provider, the tool should sit behind the same session. A separate username/password table is a non-starter; it also means you now own password resets.
3. **A way to measure usage without a product analytics stack.** Server access logs are enough. If you can't answer "how many distinct users hit this endpoint last week," you can't tell whether the tool is working.

A useful exercise: write the API contract as an OpenAPI spec or a typed interface before the UI. If the contract is awkward to call from a shell, it will be awkward in the UI too, and you'll find that out early.

## Step 1: Design the API as the product, the UI as one client

The most common structural mistake in internal tools is treating the UI as the product and the API as an implementation detail. Invert it. The API is the product; the UI, CLI, and any scheduled jobs are clients.

Here's a minimal example of an API-first internal tool in Python using FastAPI, which is a reasonable default because it generates an OpenAPI schema you can feed to client generators:

```python
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="flag-admin", version="0.1.0")

class Flag(BaseModel):
    name: str
    enabled: bool
    rollout_pct: int = 0

# In-memory for illustration; replace with your store.
FLAGS: dict[str, Flag] = {}

def current_user() -> str:
    # Replace with your SSO integration.
    return "dev@example.com"

@app.get("/flags/{name}", response_model=Flag)
def get_flag(name: str, user: str = Depends(current_user)):
    if name not in FLAGS:
        raise HTTPException(status_code=404, detail="flag not found")
    return FLAGS[name]

@app.put("/flags/{name}", response_model=Flag)
def put_flag(name: str, flag: Flag, user: str = Depends(current_user)):
    if flag.rollout_pct < 0 or flag.rollout_pct > 100:
        raise HTTPException(status_code=422, detail="rollout_pct must be 0-100")
    FLAGS[name] = flag
    return flag
```

The important design choices here are not the framework. They're:

- **Every mutation is idempotent and explicit.** `PUT` with a full representation means a retry after a timeout doesn't double-apply. Internal tools get retried by scripts; design for it.
- **Errors are structured.** `HTTPException` with a `detail` string is fine for a human, but if the CLI is going to parse errors, return a JSON body with a stable `code` field. A CLI that has to regex a prose error message is a CLI that breaks when someone edits the message.
- **Auth is a dependency, not middleware sprinkled per-route.** When you add a new endpoint, you get auth by default. The opposite pattern — opt-in auth — is how internal tools end up with an unauthenticated debug endpoint.

Once the API exists, generate a CLI. You don't need a full CLI framework to start; a shell function or a small script that calls the API is enough. The point is that the CLI and the UI share one contract.

## Step 2: Make the CLI the primary interface, not an afterthought

Developers live in a terminal. A CLI that wraps the API and handles auth via the same SSO token (or a personal access token issued by the same identity provider) will get used during incidents, in scripts, and in CI. A UI will not.

A minimal CLI in Python using `httpx` and `argparse`:

```python
import argparse
import os
import sys
import httpx

BASE = os.environ.get("FLAG_ADMIN_URL", "http://localhost:8000")
TOKEN = os.environ.get("FLAG_ADMIN_TOKEN")

def client() -> httpx.Client:
    if not TOKEN:
        sys.exit("FLAG_ADMIN_TOKEN is not set")
    return httpx.Client(base_url=BASE, headers={"Authorization": f"Bearer {TOKEN}"}, timeout=5.0)

def cmd_get(args):
    r = client().get(f"/flags/{args.name}")
    if r.status_code == 404:
        sys.exit(f"flag {args.name!r} not found")
    r.raise_for_status()
    print(r.json())

def cmd_set(args):
    body = {"name": args.name, "enabled": args.enabled, "rollout_pct": args.pct}
    r = client().put(f"/flags/{args.name}", json=body)
    r.raise_for_status()
    print(r.json())

p = argparse.ArgumentParser(prog="flagctl")
sub = p.add_subparsers(required=True)

g = sub.add_parser("get")
g.add_argument("name")
g.set_defaults(func=cmd_get)

s = sub.add_parser("set")
s.add_argument("name")
s.add_argument("--enabled", action=argparse.BooleanOptionalAction, default=True)
s.add_argument("--pct", type=int, default=0)
s.set_defaults(func=cmd_set)

args = p.parse_args()
args.func(args)
```

The CLI is short because the API carries the logic. That's the payoff of API-first design: adding a new operation means adding one subcommand and one endpoint, not a new backend and a new frontend.

A trap here: don't make the CLI require interactive auth on every invocation. If it opens a browser for SSO on each call, developers will script around it with a long-lived token stored in an environment variable, which is exactly the credential sprawl you were trying to avoid. Support a token from the environment or a config file, document its scope, and make rotation easy.

## Step 3: Build the UI for the 20% of tasks the CLI can't do

The UI earns its place when the task is inherently visual or when the user needs to compare many records at once. Good candidates: browsing a list with filters, viewing a diff, approving a batch of changes. Bad candidates: anything a single API call can do.

Two UI properties matter more than any framework choice:

**Latency on first paint.** Internal tools are often deployed behind a VPN and an SSO proxy. Each of those adds round trips. If your UI makes five API calls on load and each is 200ms, the page feels slow even though the backend is fast. Load one summary endpoint first, then fetch details lazily. This is the same advice as for public apps, but the penalty for ignoring it is higher because the baseline (a shell command) is instant.

**Failure behavior.** Internal tools hit internal services that are sometimes down. A UI that shows a spinner forever when a dependency is unavailable trains users to distrust it. Show partial data with an explicit "stale" indicator, and make the error message name the failing dependency. "Flag service unreachable" is actionable; "Something went wrong" is not.

A tested pattern is to render the last known good response with a timestamp and a banner when a refresh fails. Users can still read the data they need, and they know it might be stale. This is more work than a spinner but it's the difference between a tool people use during an incident and one they abandon.

## Step 4: Handle the edge cases that make internal tools untrustworthy

Internal tools fail differently from public products because the failure modes are operational. Three that come up repeatedly:

### Concurrent edits silently overwrite each other

If two engineers open the same record and both save, the last write wins. For a flag that controls a rollout, that's a correctness bug. Use an `ETag` or a version field and reject writes with a stale version. In the API above, add a `version: int` to the `Flag` model and return `409 Conflict` when the submitted version doesn't match the stored one. The UI can then show a diff and let the user re-apply.

### Authorization is checked at the UI layer only

A common pattern is to hide a button in the UI for users without permission, while the API endpoint accepts the request from anyone authenticated. This is the internal-tool equivalent of client-side validation. Enforce authorization in the API, and make the UI's hiding a convenience, not a control. The test is simple: call the endpoint directly with a low-privilege token and confirm it's rejected.

### The tool depends on a service that gets renamed or decommissioned

Internal services move. A tool that hardcodes a hostname in a config file will break silently when the service moves. Use service discovery or a config value that's owned by the service's team, and add a health check that the tool surfaces. A dashboard that shows "dependency: OK" is more trustworthy than one that shows nothing until it breaks.

## Step 5: Measure adoption with the logs you already have

You don't need a product analytics vendor to know whether an internal tool is used. Server access logs, parsed by path and status code, answer most questions:

- **Distinct users per week.** If the number is flat or declining after launch, the tool isn't solving the problem.
- **Error rate by endpoint.** A high 4xx rate on a specific endpoint usually means the UI is calling it with the wrong shape, or users are hitting a validation rule they don't understand.
- **p95 latency per endpoint.** Internal tools are often deployed on small instances; a slow query can make the whole tool feel broken.

A useful check is to compare the tool's usage against the workaround it was meant to replace. If the flag dashboard has 3 weekly users but the flag service's own API logs show 40 weekly calls from scripts, the dashboard isn't the interface people chose — and that's fine, as long as you know it and invest in the CLI instead.

## Verification: a 10-minute test before you call it done

Before declaring an internal tool ready, run this check with someone who didn't build it:

1. Ask them to complete the primary task from a cold start — no browser tab open, no documentation. Time it.
2. Ask them to do the same task from the CLI, if a CLI exists. Time it.
3. Break a dependency (stop the backing service) and ask them to interpret what the tool shows. If they can't tell what's wrong, the error handling needs work.
4. Check the API directly with a low-privilege token for one mutation endpoint. Confirm it's rejected.

If step 1 takes more than twice as long as step 2, the UI is the wrong primary interface for that task, and you should invest in the CLI first.

## Frequently Asked Questions

### Should internal tools use the same auth as public products?

Usually yes, if your public products use an SSO provider that your employees already have accounts with. Reusing the same identity provider means no separate password store, no separate password reset flow, and consistent offboarding when someone leaves. The main exception is tools that need to be reachable from outside the corporate network; those need a separate access model, and that's a security decision, not a UI decision.

### How do you get engineers to stop using spreadsheets for internal data?

The spreadsheet wins because it's fast, familiar, and editable. To displace it, the replacement needs to be faster for the specific task and exportable back to a spreadsheet format. A read-only API plus a CSV export endpoint often gets adopted faster than a polished UI, because it slots into the workflow people already have.

### What's the minimum viable monitoring for an internal tool?

Access logs with status codes and latency, grouped by endpoint and user, are enough to answer adoption and reliability questions. Add an alert on the 5xx rate for the tool's own API, not on the dependencies it calls — dependency alerts belong to the dependency's owner. If the tool is used during incidents, also alert when it's unreachable, because that's when people will notice.

### When should an internal tool be retired?

When its distinct weekly users drop below the cost of maintaining it, or when the workflow it supports has moved elsewhere. The signal is usually that the only traffic is from a scheduled job that someone forgot to turn off. Check the access logs for the last 90 days; if there's no human traffic and one cron caller, delete the cron job first, then the tool.

## The next 30 minutes: check one tool's access logs

Pick one internal tool your team owns and open its access log for the last seven days. Count distinct users per day and the p95 latency for its main endpoint. If distinct users are in single digits and the p95 is over a second, the tool is not being used for the reason it was built — and the fix is probably a CLI or an API endpoint, not a redesign. Write down the current workaround each user relies on, and compare it to the tool's primary path. That comparison is the design brief for whatever you build next.

## Official documentation

- [FastAPI documentation](https://fastapi.tiangolo.com/)
- [Stripe: idempotent requests](https://docs.stripe.com/api/idempotent_requests)
- [Google SRE Book: Monitoring Distributed Systems](https://sre.google/sre-book/monitoring-distributed-systems/)

---

### About this article

**Written by:** [Kubai Kevin](/about/) — software developer based in Nairobi, Kenya.

**How this article was produced:** This site uses an automated LLM pipeline designed and maintained by the author. Before publication, drafts are screened automatically for minimum length, near-duplicate content, invented or unsourced claims and citations, unverifiable product names, and broken internal links. Individual line-by-line human editing is not performed on every post. Figures, benchmarks and scenarios are illustrative unless a source is linked; verify them against official documentation before relying on them in production. See the [AI content policy](/ai-content-policy/).

**Corrections:** Report errors via the [contact page](/contact/). Corrections are applied promptly.

**Last generated:** November 2026
