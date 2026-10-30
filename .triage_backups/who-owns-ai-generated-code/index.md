# Who owns AI-generated code?

When an AI assistant writes a large share of a pull request, the interesting question is not who typed the lines. It is who is accountable when those lines misbehave in production at 2 a.m. Teams that adopt AI pair programming without answering that question tend to hit the same failure modes: reviewers rubber-stamp plausible-looking code, PRs balloon in size, and a generated query or auth flow slips past tests that only cover the happy path.

This article lays out a workable ownership model for AI-assisted development, the failure modes it prevents, and a concrete way to measure whether it is working in your repository.

## The ownership question, stated precisely

Code ownership is not about authorship. It is about who signs off on behavior. Three distinct responsibilities are usually bundled together and need to be separated when an AI is in the loop:

1. **Authorship** — who produced the characters in the diff. With AI assistance this is often mixed and hard to attribute line by line.
2. **Review** — who read the change, understood the invariant it is supposed to uphold, and confirmed the code enforces that invariant.
3. **Accountability** — who is paged when the code fails, who writes the postmortem, and who signs the compliance artifact.

A useful default: treat the AI like a junior contributor who can write code but cannot sign off, cannot merge, and does not carry a pager. The human reviewer owns the behavior of the merged artifact regardless of how many lines the AI produced. That single sentence resolves most of the ambiguity, but it has to be enforced mechanically, not just stated in a wiki page.

## Failure modes when ownership is implicit

These are the patterns that show up repeatedly when teams let AI assistance in without an explicit ownership rule.

**Ownership drift.** Developers begin to treat AI suggestions as pre-reviewed. The diff looks clean, the tests pass, and the reviewer assumes someone else checked the logic. A typical consequence is a generated database query that bypasses a row-level security policy because the model had no knowledge of that policy. The query returns far more rows than intended, and a replica or downstream service runs out of memory.

**Review inflation.** Because generating code is cheap, PRs grow. A change that would have been 40 lines becomes 400, including helper functions with no context. Reviewers spend their entire budget understanding structure instead of verifying invariants, and review quality drops even as review volume rises.

**Happy-path tests.** AI-generated tests frequently assert the behavior of the code that was just written, which means they confirm the implementation rather than the requirement. Edge cases — token rotation, currency rounding, partial failures — are exactly what the generated tests tend to miss.

**Cost creep.** Per-seat AI tooling is usually a small line item until advanced features (chat, custom rules, security scanning) are enabled. The budget then grows faster than headcount, and the cost is easy to miss because it is spread across many small invoices.

**Prompt laundering.** If the AI writes the prompt as well as the code, the business invariant never gets written down by a human. The resulting code is a plausible reconstruction of the spec rather than an enforcement of it.

## A workable ownership model

The model below has four rules. Each one is designed to be enforceable in CI, because rules that live only in documentation get ignored under deadline pressure.

**Rule 1 — The human writes the prompt.** Every AI-assisted PR must include a human-written prompt in the description stating the business invariant and the security boundary. The AI may respond to the prompt; it may not author it. This forces the requirement to exist in prose before it exists in code.

**Rule 2 — Files carry an owner tag.** Every file begins with a tag such as `// @owner: payments-team`. The AI may edit beneath the tag but may not move or delete it. A linter fails the build if the tag is missing or malformed. This keeps a named human team accountable for the file regardless of who edited it last.

**Rule 3 — AI-generated functions carry review stubs.** Each generated function includes a `// @review-notes: TODO` block that the human reviewer must fill in. The presence of an empty stub blocks the merge. This is a forcing function: it makes silent approval impossible without an explicit act of omission.

**Rule 4 — A security gate runs on every PR.** A static analysis pass must return zero high-severity findings before approval. The gate should encode the invariants your unit tests do not cover — authorization boundaries, parameterized queries, secret handling.

A fifth rule is optional but effective: a **human veto**. Any reviewer can flag a PR as AI-heavy and request a human rewrite. The threshold is a policy choice; the important part is that the veto exists and is socially acceptable to use.

## Worked example: catching a bypassed policy

Consider a generated endpoint that looks up a user's recent transactions. The model produces something like the following, which passes a naive test because it returns the right rows for the test user:

```python
def get_recent_transactions(user_id):
    query = f"SELECT * FROM transactions WHERE user_id = {user_id} ORDER BY created_at DESC LIMIT 50"
    return db.execute(query)
```

Two problems are invisible to a happy-path test. First, the query is built by string interpolation, which is an injection risk. Second, it does not filter by tenant, so in a multi-tenant schema it can return rows belonging to another tenant if the identifier is guessed or reused.

A reviewer following Rule 3 has to write review notes, and the act of writing them surfaces the question "what is the security boundary here?" The prompt required by Rule 1 should have answered it: "This endpoint returns only transactions belonging to the authenticated user's tenant; it must never accept a tenant identifier from the request." With that invariant written down, the correct implementation is obvious:

```python
def get_recent_transactions(user_id, tenant_id):
    query = (
        "SELECT * FROM transactions "
        "WHERE user_id = %s AND tenant_id = %s "
        "ORDER BY created_at DESC LIMIT 50"
    )
    return db.execute(query, (user_id, tenant_id))
```

The security gate catches the first version if it is configured to flag string-built SQL. The prompt catches it earlier if the invariant is stated. Neither catches it if both are absent and the reviewer trusts the passing tests.

## Measuring whether the model is working

Do not adopt a metric you cannot compute from data you already collect. The following are all derivable from git history, CI logs, and your issue tracker.

- **Median PR size in changed lines.** Compare a baseline window before the policy to a window after. Rising size is a signal that generation is outrunning review.
- **Median time from first review request to approval.** Falling time is good only if review comments per PR stay stable or rise. Falling time with falling comments usually means rubber-stamping.
- **Review comments per PR, split by whether they reference behavior or style.** A simple heuristic: count comments containing words like "invariant", "boundary", "tenant", "authorization", "edge case" separately from comments about naming or formatting.
- **Security-gate findings per 100 PRs, by severity.** A gate that never fires is either unnecessary or misconfigured. Track the ratio of findings to merges.
- **Incidents in the 30 days following merge, attributed to changed files.** This is the metric that matters most and is the hardest to attribute cleanly; even a coarse version is more useful than none.
- **Tooling spend per active contributor per month.** Pull this from your billing export, not from memory.

To measure AI contribution itself, the most robust signal available in most repositories is the presence of the review stub and the prompt in the PR description, not a line-count percentage. Line attribution is noisy because formatting, refactoring, and generated boilerplate all distort the count. If you do track a percentage, label it clearly as a heuristic and do not use it as a gate.

## A decision checklist before you scale

Before rolling AI assistance out beyond a pilot repository, confirm each of the following:

- [ ] The ownership rule is written down and names a human role, not a tool.
- [ ] The prompt requirement is enforced by a CI check, not by convention.
- [ ] The owner tag format is pinned to a fixed list of team names stored in the repository.
- [ ] The security gate runs on every PR and blocks on high-severity findings.
- [ ] The review-stub requirement blocks merge when empty.
- [ ] A veto path exists and has been used at least once without social penalty.
- [ ] Tooling cost per contributor is tracked in a dashboard, not in a spreadsheet someone updates manually.
- [ ] The pilot repository is not security-critical.

If any box is unchecked, fix that before expanding. The cost of retrofitting an ownership model onto a large codebase is much higher than the cost of starting with one.

## Common objections

**"The tests pass, so the code is fine."** Tests confirm the behavior you thought to assert. They do not confirm the invariant you forgot to write down. The security gate and the written prompt are what cover that gap.

**"Reviewers don't have time to fill in review notes."** They have time to review; the stub just makes the review visible. If the notes take more than a few minutes per function, the function is too large to review safely, which is itself useful information.

**"Line-count thresholds are arbitrary."** They are. That is why they should be a veto trigger for human judgment, not an automatic rejection. The threshold's job is to start a conversation, not to end one.

**"Local autocomplete has no network latency, so performance is a non-issue."** Latency from the model is usually not the bottleneck. The bottlenecks are CI time added by the security gate and the review time added by larger diffs. Measure both before assuming the tool is free.

## Action for the next 30 minutes

Open the repository you are most likely to pilot on. Create `.cursor/rules.json` at the root with the policy below, commit it on a branch, and open a draft PR. Then take the next AI-assisted change in your queue and check it against each rule: is there a human-written prompt, an owner tag, a filled review stub, and a passing security gate?

```json
{
  "prompts": { "required_fields": ["business_invariant", "security_boundary"] },
  "ownership_tags": { "required": true, "tag_format": "// @owner: <team>" },
  "review_stubs": { "required": true, "stub_format": "// @review-notes: TODO" },
  "security_gate": { "severity_threshold": "HIGH", "fail_build": true },
  "veto_quorum": 1
}
```

If the change fails any rule, do not merge it. Add the missing prompt, tag, or review notes first. The point of the exercise is not the file itself; it is that the ownership model becomes something a machine checks rather than something a person remembers.
