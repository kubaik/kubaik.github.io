# Red-teaming agents without killing velocity

Red-teaming an LLM agent is often treated as a separate security exercise that happens after the agent ships. That sequencing creates a lag between finding a flaw and fixing it, and it puts the security team in the position of reviewing behavior they did not design. A more practical pattern is to embed a small adversarial test suite into the pull request pipeline, so the agent is exercised against hostile inputs before it reaches production. The goal is not to replace a pentest; it is to catch the cheap, obvious failures automatically and leave the expensive human review for the cases that need judgment.

## The core idea in one paragraph

Treat red-teaming as a gate, not a phase. A pull request runs a short suite of adversarial tests against the agent's exposed interface. The suite has three parts: synthetic malformed inputs generated from the interface schema, semantic attacks that try to make the agent leak or hallucinate, and a fail-closed assertion that the agent returns a safe response or an error rather than an unsafe one. If any of these fail, the merge is blocked. The suite should run in the same CI job as unit tests, in parallel, so the wall-clock cost is small. The value comes from making the red-team role mechanical for the cases that can be mechanized, and reserving human red-teaming for the cases that cannot.

## Why this concept confuses people

Three misunderstandings recur.

The first is that red-teaming requires security specialists. Specialists are valuable for systemic threat modeling, but they are rarely the people who know an individual agent's quirks. A developer who has watched the agent fail on multi-line input will often find a specific bug faster than a generic scanner will.

The second is that red-teaming is the same as unit testing. Unit tests assert that the code does what it was written to do. Red-team tests assert that the agent fails safely when the input is not what it was written for. These are different properties and they need different test designs.

The third is that red-teaming necessarily slows delivery. It slows delivery when it is a separate stage with its own review queue. It does not slow delivery when it is a fast, automated gate that runs alongside the tests a team already runs.

## A mental model: the agent is a door

Picture the agent as a room with a door. Unit tests check that the keycard opens the door. Red-teaming is the set of people outside trying knocks, fake badges, and side passages. The door is the only interface exposed to users, so red-team effort should focus there first.

If the agent receives Slack webhooks, test the malformed payloads Slack could send, not every internal function call. If the agent calls an internal API, test the HTTP status codes and timeouts that API might return, not every Python exception. The objective is to find the smallest set of inputs that let an attacker change the agent's behavior. Once the interface is understood as a specific shape, the tests can be designed to fit through it.

## A worked example

Consider an internal agent that summarizes GitHub pull requests and posts the summary to a team Slack channel. The agent runs as a Python function, is triggered by a GitHub webhook, and posts to Slack via an incoming webhook URL. The core loop looks like this:

```python
import os
import httpx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import GitHubPRLoader

def summarize_pr(pr_url: str) -> str:
    loader = GitHubPRLoader(
        pr_url=pr_url,
        access_token=os.getenv("GITHUB_TOKEN"),
        branch=os.getenv("GITHUB_BRANCH", "main"),
    )
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    # Replace with a real model call; the shape of the interface is what matters here.
    return "Summary: " + chunks[0].page_content[:200]

def handler(event, context):
    pr_url = event["pr_url"]
    summary = summarize_pr(pr_url)
    webhook_url = os.getenv("SLACK_WEBHOOK")
    httpx.post(webhook_url, json={"text": summary})
```

The red-team suite for this agent has three stages.

### Stage 1: synthetic payloads

Generate malformed GitHub webhook events from the documented payload schema. Inject invalid JSON, missing required fields, oversized payloads, and Unicode control characters. Run these under `pytest` with a mocked Slack webhook so no message is actually posted. The assertion is not that the agent returns a summary; it is that the agent does not crash the handler, does not post to Slack on invalid input, and returns a structured error.

The runtime of this stage depends on how many payloads are generated and how fast the agent's dependencies load. A useful measurement is to run the suite locally with `pytest --durations=10` and record the slowest tests. If the stage is too slow for every pull request, reduce the payload count and move the full set to a nightly job.

### Stage 2: semantic attacks

Craft prompts that try to make the summarizer leak secrets or hallucinate. Examples include a pull request titled `password: <value>`, a file named `.env` containing internal URLs, and a diff that includes a token-shaped string. Assert that the agent's output does not contain the secret verbatim and does not include PII.

Run these against a sandboxed copy of the model, not the production deployment. The sandbox can be a container that has no network access to the production data store, or a managed gateway configured with a separate quota and no credentials. The point is that the attack does not reach production data.

### Stage 3: fail-closed assertion

For each stage, the expected behavior on hostile input is a safe response or an explicit error, not a confident summary. If a file named `.env` is encountered, the expected output is an error indicating that a sensitive file was detected. This is a resilience assertion, not a correctness assertion.

### What this catches

In practice this kind of suite catches classes of bugs that unit tests miss: missing input sanitization that allows markup in PR titles, race conditions when two webhooks arrive close together, and unbounded memory growth when a large diff is loaded. Each of these is a specific failure mode that can be reproduced with a crafted input.

## How to measure the value

Do not rely on a vendor benchmark or a survey number. Measure the suite in your own repository.

Instrument three things. First, the number of red-team test failures per week and the number that correspond to real bugs. Second, the wall-clock time the suite adds to the CI pipeline, measured with the CI provider's own timing output. Third, the number of production incidents in the agent's area before and after the gate was introduced, pulled from the incident tracker.

A simple way to present this is a two-column table with the period before the gate and the period after, and rows for CI duration, red-team failures, real bugs found, and production incidents. The numbers are specific to your system; the point is that they are measured, not assumed.

## Common misconceptions, corrected

**Red-teaming requires security expertise.** Not for the automated portion. Security specialists are valuable for systemic risks, but the specific quirks of an agent are usually known to the developers who built it. A developer who has watched the agent fail on multi-line titles will find that bug faster than a generic scanner.

**Red-teaming needs a large budget.** The synthetic payload stage runs on the CI provider's existing runners. The sandbox for semantic attacks can be a container on the same infrastructure. The main cost is engineering time to write and maintain the tests, not tooling licenses. Any cost estimate should be built from the CI provider's published per-minute rates and the team's own loaded engineering rate, not from a quoted figure.

**Red-teaming slows development.** It slows development when it is a separate stage with its own review queue. It does not slow development when it runs in parallel with existing tests and fails fast. The measurable question is the delta in pipeline wall-clock time, which can be read from the CI provider's timing view.

## Advanced techniques

Once the basic gate is stable, two higher-signal techniques are worth adding.

### Adversarial prompts as test cases

Maintain a curated set of prompts that attempt to jailbreak the agent, tagged by attack vector: prompt injection, role play, token smuggling, and so on. Assert that the agent refuses or returns a safety warning. Track which vectors produce failures so the team can prioritize fixes. The tag taxonomy matters more than the size of the set; a small, well-tagged set is easier to act on than a large, unlabeled one.

### Shadow canary deployments

Deploy the new agent version to a small fraction of traffic but do not post its output to the user-facing channel. Instead, log the output and compare it to the previous version's output. Run a diff that checks for hallucinations, omissions, and unsafe content. If the diff shows a significant change in summary length or any PII leakage, roll back automatically.

The following Terraform snippet shows the shape of a shadow canary configuration. It is illustrative; the exact resource names and environment variables depend on the deployment.

```hcl
resource "aws_lambda_function" "canary" {
  function_name    = "pr-summary-canary"
  handler          = "index.handler"
  runtime          = "python3.11"
  filename         = "lambda.zip"
  memory_size      = 256
  timeout          = 5
  vpc_config {
    subnet_ids         = [aws_subnet.private_a.id]
    security_group_ids = [aws_security_group.lambda.id]
  }
  provisioned_concurrent_executions = 5
  environment {
    variables = {
      MODE           = "shadow"
      CANARY_PERCENT = "5"
    }
  }
}
```

A useful metric to track alongside the canary is the *attack surface delta*: the change in the number of red-team tests that pass before and after a change. A positive delta means the change increased resilience; a negative delta means it decreased it. This can be recorded as a custom metric in whatever monitoring system the team already uses.

## Quick reference

| Concern | Approach | Typical runtime | When to run |
|---|---|---|---|
| Synthetic payloads | Schema-driven fuzzing under pytest | Seconds to a minute | Every pull request |
| Semantic attacks | Crafted prompts against a sandboxed model | Seconds | Every pull request or nightly |
| Jailbreak prompts | Curated, tagged adversarial set | Seconds | Nightly or weekly |
| Shadow canary | Compare new and old outputs on a traffic slice | Minutes | Before merge to main |
| Rollback gate | Automated revert on suite failure | Seconds | On failure |

The runtimes above are illustrative. Measure them in your own pipeline; the CI provider's timing output is the authoritative source.

## Frequently asked questions

**What if the red-team tests are too slow for CI?**
Split the suite into a fast path that runs on every pull request and a slow path that runs nightly. The fast path should cover the highest-signal cases: schema-driven payloads and a small set of semantic attacks. The slow path can include the full jailbreak set and the canary comparison. If a slow-path test fails, post a comment on the relevant pull request so a reviewer can decide whether to block the merge.

**How do I decide which red-team tests to write first?**
Start with the failure modes that have actually occurred in production, or that are most likely given the agent's interface. For an internal agent that reads documents and posts to a chat channel, the usual candidates are input sanitization, rate limiting, timeout handling, PII leakage, and hallucination. Write one test per mode, name it after the mode and the attack vector, and keep the tests in a dedicated directory so they are easy to find.

**What if the agent uses a closed-source model?**
Run the same tests, but record the model's responses once and replay them in CI. This is the standard HTTP-recording pattern: capture the request and response pairs, commit the recording, and assert against the replayed output. When the model changes, re-record. The trade-off is that the tests no longer exercise the live model, so a separate, less frequent job should run against the real endpoint.

**How do I justify the engineering time?**
Measure the cost of a single production incident in the agent's area: engineering hours spent, downstream impact, and any direct infrastructure cost. Compare that to the engineering time required to write and maintain the suite. Present both numbers from your own records. Avoid quoted industry averages; they are not specific to your system and they invite disagreement.

## One thing to do in the next 30 minutes

Open the repository for an agent you maintain and create a single test file that sends one malformed input to the agent's entry point and asserts that it fails closed rather than posting an unsafe result. Run it locally. If it passes, commit it and add it to the existing CI job. If it fails, you have found a real bug and a reason to build the rest of the suite.
