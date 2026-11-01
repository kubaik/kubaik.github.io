# STRIDE is a checklist, not a threat model

Most teams that say they do threat modeling are running a brainstorming session with a STRIDE-shaped agenda. They write six categories on a whiteboard, list a few scary things under each, and call it done. That is a taxonomy exercise, and it produces a document that is long on nouns and short on decisions. The claim that follows is narrow: STRIDE is a useful checklist for prompting memory, but it is not a model of your system, and treating it as one is a common reason threat models end up as PDFs nobody opens after the review meeting. Attack trees force reasoning about how an adversary actually reaches an asset — and that is the part that changes what gets built. The two belong together, but in the opposite order from how most teams use them.

The problem worth solving is the gap between a threat-model artifact that satisfies a compliance checkbox and a threat model that changes a design decision before code ships. A team that has produced a STRIDE table, filed it, and then shipped a feature with the same authorization bug the table was supposed to catch has not failed at security review. It has failed at modeling, and the fix is structural rather than cultural.

## The steelman: why STRIDE workshops are the default

The case for the status quo is genuinely strong, and it is worth stating before knocking it down. STRIDE — Spoofing, Tampering, Repudiation, Information disclosure, Denial of service, Elevation of privilege — is memorable, vendor-neutral, and maps cleanly onto the categories auditors and security reviewers already recognize. It is cheap to teach. A facilitator can run a two-hour session with engineers who have never done threat modeling and get a non-empty list of concerns out the other side. Microsoft's threat modeling guidance has used this framing for years, and the mnemonic has spread because it works as a memory aid. For an organization that has never done any structured security review, going from zero to a STRIDE session is a real improvement.

The second part of the steelman is that STRIDE forces breadth. Left to their own devices, engineers tend to think about the threats they already know how to fix — usually injection and authentication. STRIDE's six categories push the conversation toward repudiation and information disclosure, which are the ones people forget. A checklist that reliably surfaces forgotten categories has value.

None of that is wrong. The problem is what happens after the session.

## Where the STRIDE workshop falls apart

A STRIDE list is a set of categories, not a set of paths. "Elevation of privilege" is not a threat; it is a heading. The actual threat is something like: an authenticated user can change the `tenant_id` in a request body and the handler trusts it, so they can read another tenant's invoices. That sentence has an actor, a precondition, a mechanism, and an asset. The category heading has none of those. When the output of a session is a list of headings with one-line examples underneath, the team has no way to rank the items, no way to tell whether a given mitigation actually closes the path, and no way to know when they are done.

The second failure is that STRIDE sessions are usually run against a diagram of the system, and the diagram is usually wrong. It shows the services and the queues and maybe the database, but it does not show the trust boundaries that actually matter in a modern backend: the boundary between the public load balancer and the service mesh, the boundary between the application and the payment provider's webhook endpoint, the boundary between the job runner and the credentials it holds. If the diagram flattens those, the STRIDE prompts fire against the wrong edges.

The third failure is temporal. A STRIDE workshop is a point-in-time event. Backend systems change weekly. The threat model does not. Six months later the service has a new async worker, a new third-party integration, and a new admin endpoint, and the document describes none of them. The team is not being lazy; the process has no mechanism for staying current because it was designed as an event, not a practice.

## Attack trees are the model; STRIDE is the prompt

The better mental model is to invert the relationship. An attack tree starts from an asset and works backward. The root is the thing the adversary wants — read another tenant's data, move money out of a merchant account, take the service down during a peak. The children are the ways to get there, and each of those has children, until you reach leaves that are concrete enough to test or mitigate. That structure gives you three things a STRIDE list does not: a ranking (the shortest path to the root is the scariest), a completeness check (every branch either has a mitigation or is explicitly accepted), and a maintenance story (when the system changes, you add or prune branches).

STRIDE's job in this model is to prompt at each node. When expanding a branch, running the six categories against that specific step is a fast way to avoid missing an angle. "To reach the root, the attacker must forge a session token" — at that node, Spoofing, Tampering, and Information disclosure are the relevant prompts. The exercise is not a STRIDE pass over the whole system; it is a STRIDE pass over one edge of one tree. That is a much smaller, much more productive unit of work.

The practical consequence is that the artifact changes. Instead of a table of categories, the output is a tree per critical asset plus a decision log that says which branches are mitigated, which are accepted, and which are still open. That is a document an engineer can act on.

## A worked example: the multi-tenant invoice endpoint

Take a concrete backend shape that is common in payments-adjacent systems: a REST endpoint `GET /invoices/{id}` that returns an invoice for the authenticated user's tenant. The asset is another tenant's invoice data. The root of the tree is "read an invoice belonging to a different tenant."

Here is the tree, expanded two levels, with STRIDE prompts applied at each node. The numbering is for reference only; no costs or likelihoods are asserted.

```text
ROOT: read another tenant's invoice
├── 1. Guess or enumerate an invoice id
│   ├── 1a. Ids are sequential integers           [Information disclosure]
│   ├── 1b. Ids are UUIDv4 but leaked in logs     [Information disclosure]
│   └── 1c. Ids are exposed in a public share link [Information disclosure]
├── 2. Authenticate as a user in the target tenant
│   ├── 2a. Steal a session token                 [Spoofing]
│   ├── 2b. Forge a JWT with a weak/leaked key    [Spoofing, Tampering]
│   └── 2c. Reuse a token after role change       [Elevation of privilege]
├── 3. Authenticate as yourself but confuse the tenant resolution
│   ├── 3a. Send `tenant_id` in the request body  [Tampering]
│   ├── 3b. Send `X-Tenant-Id` header the gateway trusts [Tampering]
│   └── 3c. Exploit a cached authorization decision keyed only on user id [Elevation of privilege]
└── 4. Reach the data through a side channel
    ├── 4a. Error messages leak existence of ids  [Information disclosure]
    └── 4b. Timing difference reveals tenant     [Information disclosure]
```

Now the useful part: each leaf gets a status. 1a is mitigated by using unguessable ids. 1b stays open until log scrubbing is in place. 3a is the branch that most often ships broken, because the handler reads `tenant_id` from the body and the authorization check reads it from the session — two sources of truth. 3c is the one that bites in systems that added a cache for latency and keyed it on the wrong field.

The tree tells you what to test. For 3a, the test is: authenticate as tenant A, send a request with `tenant_id` set to tenant B in the body, assert 403 or 404. For 3c, the test is: warm the cache as user A in tenant A, change the user's tenant association, replay, assert the old decision is not served. Those are concrete, automatable checks. A STRIDE table saying "Tampering: validate inputs" produces neither.

### Working the tree by hand

A worked expansion shows why the tree changes decisions. Start at the root and ask the STRIDE prompts about the step, not the system.

At node 3, the mechanism is "the request carries a tenant identifier and the server resolves it." Spoofing: can the caller claim a tenant they are not in? Only if the identifier is trusted from the caller rather than derived from the session. Tampering: can the identifier be changed mid-flight? Yes, if it travels in a body field or a client-settable header. Elevation of privilege: does the resolved tenant grant access the caller's role does not allow? Only if authorization and resolution read different fields.

That chain of questions produces a design rule, not a list item: derive the tenant from the authenticated session, ignore any tenant identifier the client supplies, and assert that the authorization layer reads the same value the query uses. The rule is testable, reviewable in a diff, and narrow enough to enforce in code review. A STRIDE row reading "Tampering — validate all input" is none of those things.

At node 4, the mechanism is "the response differs depending on whether the invoice exists and whether it belongs to the caller." Information disclosure: does a 404 for missing and a 403 for forbidden let an attacker distinguish existence across tenants? Yes, if both cases can be probed. The design rule is to return the same status and body shape for "not found" and "not yours," and to keep timing roughly constant by avoiding a fast path that skips the ownership query. That rule is a two-line test and a one-line code convention.

Two nodes, two design rules, two tests. That ratio — every branch that matters ending in a rule and a test — is the difference between a model and a taxonomy.

## The decision rule: when to use which

A defensible rule for a team is this. Use STRIDE as a prompt at the leaf of an attack tree, not as the organizing structure of the model. Use attack trees whenever the asset is one an adversary would pay for or a regulator would fine you over — money movement, tenant data, authentication state, admin surfaces. Use a plain STRIDE checklist when doing a first pass on a system nobody has modeled and the immediate need is to discover what the assets even are; that is a legitimate use of the checklist as a discovery tool, and it is where the common advice is still right.

The corollary is that not every service needs a tree. A stateless formatting service behind an authenticated gateway with no data store does not justify one. The tree is for assets, and if a service holds no asset, the tree is overhead. The failure mode in the other direction — treeing everything — is how threat modeling becomes a tax that teams route around.

A short comparison makes the division of labor concrete:

| Question | STRIDE checklist | Attack tree |
| --- | --- | --- |
| What does it produce? | A list of categories with examples | Paths from attacker to asset |
| Can it be ranked? | No, categories are peers | Yes, by path length and feasibility |
| Does it show when you are done? | No | Yes, when every leaf has a status |
| How does it stay current? | Re-run the workshop | Diff the tree in the same PR as the code |
| Best first use | Asset and boundary discovery | Modeling a specific high-value asset |
| Main failure mode | Filed and forgotten | Over-applied to services with no asset |

## Where the common advice is still right

Two pieces of the conventional wisdom hold up. First, do the modeling before the code, not after. A tree drawn against a design doc catches the 3a-style bug in an afternoon; the same tree drawn against a shipped service catches it in a postmortem. Second, involve the people who will operate the system. The on-call engineer knows which endpoints get hammered and which third-party webhooks are flaky, and those are the branches that matter. A threat model produced by a security team in isolation will miss the operational edges.

There is also a version of the STRIDE-first approach that works: when a team is new to this, run one STRIDE session to inventory assets and trust boundaries, then convert the output into trees for the two or three assets that clearly matter. The session is a bootstrap, not the deliverable.

## Keeping the tree alive with a small amount of code

The maintenance problem is real, and the answer is not a heavyweight tool. A tree that lives in a YAML file in the repo, next to the service it describes, and that is rendered by a script in CI, stays current because it is reviewed in the same pull request as the change that affects it. The example below reads a tree from YAML, checks that every leaf has a status, and fails if any leaf is marked `open` without an owner or `accepted` without a reason. That is enough to stop the drift where a new branch is added and nobody notices it is unmitigated.

```python
# threat_tree.py — run in CI: python threat_tree.py tree.yaml
import sys
import yaml

VALID_STATUS = {"mitigated", "accepted", "open"}

def walk(node, path, errors):
    name = node.get("name", "<unnamed>")
    here = path + [name]
    if "children" in node and node["children"]:
        for child in node["children"]:
            walk(child, here, errors)
        return
    status = node.get("status")
    if status not in VALID_STATUS:
        errors.append(f"leaf {' > '.join(here)}: invalid status {status!r}")
        return
    if status == "open" and not node.get("owner"):
        errors.append(f"leaf {' > '.join(here)}: open with no owner")
    if status == "accepted" and not node.get("reason"):
        errors.append(f"leaf {' > '.join(here)}: accepted with no reason")

def main(path):
    with open(path) as f:
        tree = yaml.safe_load(f)
    errors = []
    walk(tree, [], errors)
    if errors:
        print("\n".join(errors))
        sys.exit(1)
    print("threat tree ok")

if __name__ == "__main__":
    main(sys.argv[1])
```

And the YAML it validates, trimmed to the branches that matter:

```yaml
# tree.yaml
name: read another tenant's invoice
children:
  - name: confuse tenant resolution
    children:
      - name: tenant_id in request body
        status: mitigated
        test: tests/test_tenant_isolation.py::test_body_tenant_id_rejected
      - name: authorization cache keyed on user id only
        status: open
        owner: payments-platform
  - name: reach data via error messages
    status: accepted
    reason: 404 for both missing and forbidden; no existence leak
```

The point is not the script. The point is that the tree is now a file with a schema, it is diffable, and the CI check makes an unowned open branch a build failure. That is what turns threat modeling from an event into a practice. Teams that adopt this pattern commonly find the tree stays roughly current because the PR that adds an endpoint is the PR that adds the branch.

### Measuring whether it is working

Claims that a process "works" are cheap. Two measurements are cheap and honest, and both can be computed from data the team already has.

The first is branch freshness. For each tree file, record the commit date of the last change to the tree and the last change to the service's source. Track the gap over time. The signal is not a specific number; it is whether tree edits cluster in the same commits as code edits. A tree that changes only in quarterly sweeps is stale by construction.

The second is test coverage of mitigated leaves. The schema above already carries a `test` field on mitigated leaves. Count leaves where `status: mitigated` and `test` is empty. That count should be zero; anything else is a mitigation asserted without a check. Both numbers come from `git log` and a grep over the YAML, so no new tooling is required.

## A short checklist for the next tree you draw

When sitting down to draw one, the questions that separate a useful tree from a decorative one are:

- What is the asset, stated as a sentence an adversary would agree with? "Read another tenant's invoice" beats "data confidentiality."
- What are the trust boundaries the attacker must cross, and are they on the diagram? If the boundary between the service and the payment provider's webhook is not drawn, the tree will miss the replay branch.
- Does every leaf have a status and, if open, an owner and a test? A leaf without a test is a wish.
- Is the shortest path to the root mitigated? If the cheapest branch is open, the rest of the tree is decoration.
- When did the tree last change, and did it change in the same PR as the code? If not, it is already stale.

## The next concrete step

Open the repository for the service that handles money or tenant data, pick the single endpoint that would hurt most if it leaked, and write its attack tree as a `tree.yaml` in that repo — root, two levels, status on every leaf. Do not run a STRIDE workshop first. Draw the tree, mark the leaves that cannot honestly be called mitigated as `open`, and file one ticket per open leaf with the test that would close it. Thirty minutes gets a model that names a real path and a real test, which is more than most STRIDE tables ever produce.
