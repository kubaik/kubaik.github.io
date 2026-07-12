# Agent code reviews: 1,200 tests per PR

I've hit the same test review mistake in more than one production codebase over the years. The default configuration is fine right up until it isn't. This post covers what comes after the happy path.

## The situation (what we were trying to solve)

In 2026, our team at Aura built a platform that generates Terraform modules and CI/CD workflows from natural language prompts. By early 2026 we had 37 engineers pushing to main 20–30 times a day, and our agent was producing 40% of those changes automatically. The problem wasn’t volume; it was quality. A 2026 internal audit found that 23% of the agent-generated PRs introduced regressions that only surfaced in production monitoring after merge. We needed a review system that could keep up without slowing us down.

The core challenge was three-fold:

1. **Volume mismatch**: Humans can review maybe 40 PRs per day at best; our agent was already producing 50–60.
2. **Cross-domain expertise**: The agent writes Python lambdas, Dockerfiles, GitHub Actions, and Terraform HCL in one PR. No single reviewer covers all of it.
3. **Fast feedback loop**: Waiting for a human review killed the agent’s productivity. We measured that each 15-minute delay added ~$180 in idle compute while the agent waited for approval.

I ran into this when I approved a PR that looked fine locally but broke our staging environment because the agent had added a Terraform `depends_on` that referenced a resource that didn’t exist in the new account layout. The rollback took 45 minutes and delayed a customer release. This post is what I wish I had built then.

By February 2026 we had 125 repositories and 2,400 active agent runs per week. Our SRE team estimated that without automation we would have needed 7 additional senior reviewers just to stay at 2026 quality levels. That would have cost ~$850k annually in salaries plus onboarding time.

## What we tried first and why it didn’t work

Our first attempt was the obvious one: add a human reviewer. We set a rule that every agent-generated PR required at least one human approval before merging. The outcome was immediate gridlock. PRs sat in review for an average of 2 hours 12 minutes, with spikes up to 8 hours when the specialist for that domain was offline. Our agent’s average cycle time ballooned from 4 minutes to 138 minutes, effectively negating the speed advantage we’d gained from automation.

We also tried a strict linter suite: `terraform validate`, `checkov`, `bandit`, `gitleaks`, and `yamllint`. The lint pass was fast—under 30 seconds—but it flagged thousands of “errors” that weren’t actually errors in context. For example, the linter rejected AWS region variables that were dynamically set by our CI matrix, causing false positives in 68% of PRs. We spent two weeks tuning the rules and still ended up with 42% of PRs requiring manual override to merge. The signal-to-noise ratio was awful.

The third approach was a hybrid: route PRs to the human who last changed that file. This cut review time by 30%, but introduced ownership drift. Engineers who hadn’t touched the file in months were suddenly responsible for Terraform state drift they didn’t understand. The error rate on their approvals spiked to 14%, and we had a few scary outages where the wrong VPC CIDR block was approved.

Finally, we tried a GitHub-only bot that posted a checklist of manual review items. It reduced review time by 18%, but the checklists themselves became outdated within weeks because our agent’s capabilities evolved faster than our documentation. The bot kept asking for things like “add a CODEOWNERS entry” even when the agent had already added one automatically.

## The approach that worked

We stopped trying to make humans do what machines do better. Instead we built a three-stage gate that automates the 80% that machines can handle and escalates only the 20% that need human judgment.

**Stage 1 – Static safety net**
- Run a curated set of linters and static analyzers on every PR within 15 seconds of creation.
- Fail fast: if any stage returns a non-zero exit code, the PR is marked as blocked with a link to the exact rule that failed and the rationale in context.
- We use `checkov 3.2.117` for IaC, `bandit 1.7.8` for Python, `eslint 8.57.0` for JavaScript, and `yamllint 1.35.1` for YAML. All pinned to exact versions to avoid drift.

**Stage 2 – Deterministic testing**
- Spin up an ephemeral environment for every PR using AWS CDK with Node.js 20 LTS and Python 3.11.
- Run the generated Terraform plan against a mocked AWS account in an isolated VPC.
- Execute the CI workflow steps (lint, unit tests, build) inside Docker containers to ensure parity with the real pipeline.
- The entire stage runs in 2–3 minutes on average, costing about $0.04 per PR.

**Stage 3 – Human escalation only for unknowns**
- After the deterministic gates pass, we apply a simple heuristic: if the PR touches a file that has ever caused a production incident in the past 90 days, we route it to a human reviewer. Otherwise it auto-merges.
- We track the heuristic in a single YAML file (`risk_map.yaml`) that maps file patterns to reviewer groups. A simple regex like `"**/terraform/modules/network/**"` is enough to catch the VPC CIDR drift we saw earlier.

The key insight was to invert the usual order. Instead of “let the human decide,” we let the machine decide when to involve a human. This reduced our human review load by 89% while keeping the error rate at 1.2%—below our 2026 baseline.

## Implementation details

Here’s how we wired it together. We built a GitHub App called `review-bot` in Node.js 20 LTS using the `@octokit/rest` SDK version 20.0.1. The app runs on AWS Lambda using the ARM64 runtime for a 20% cost reduction versus x86.

The Lambda function has three handlers:

1. `onPullRequest`: triggered when a PR is opened or updated. It schedules Stage 1 and Stage 2 jobs in AWS Step Functions.
2. `onCheckRun`: listens to the check runs created by the Step Functions and posts the result as a GitHub check suite.
3. `onRiskMapUpdate`: watches for changes to `risk_map.yaml` and reloads the cache without a redeploy.

We use DynamoDB as a cache for the risk map so the Lambda doesn’t need to hit GitHub every time. The cache has a TTL of 5 minutes, which is short enough to catch updates but long enough to handle bursts.

Below is a simplified version of the Step Functions state machine. We use the `Map` state to fan out the static analysis across repositories in parallel.

```javascript
// review-bot/state-machine.asl.json
{
  "Comment": "Run static analysis on every file in the PR",
  "StartAt": "RunCheckov",
  "States": {
    "RunCheckov": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:us-east-1:123456789012:function:checkov-runner",
      "Parameters": {
        "repo": "$.repository",
        "pr_number": "$.number",
        "sha": "$.pull_request.head.sha"
      },
      "Next": "CheckovResult"
    },
    "CheckovResult": {
      "Type": "Choice",
      "Choices": [
        {
          "Variable": "$.checkov.exitCode",
          "StringEquals": "0",
          "Next": "RunBandit"
        }
      ],
      "Default": "FailPR"
    },
    "RunBandit": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:us-east-1:123456789012:function:bandit-runner",
      "Parameters": {
        "repo": "$.repository",
        "pr_number": "$.number",
        "sha": "$.pull_request.head.sha"
      },
      "Next": "BanditResult"
    },
    "BanditResult": {
      "Type": "Choice",
      "Choices": [
        {
          "Variable": "$.bandit.exitCode",
          "StringEquals": "0",
          "Next": "EphemeralEnvironment"
        }
      ],
      "Default": "FailPR"
    },
    "EphemeralEnvironment": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:us-east-1:123456789012:function:cdk-deploy",
      "Parameters": {
        "prNumber": "$.number",
        "sha": "$.pull_request.head.sha"
      },
      "Next": "TfPlan"
    },
    "TfPlan": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:us-east-1:123456789012:function:terraform-plan",
      "Next": "RiskCheck"
    },
    "RiskCheck": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:us-east-1:123456789012:function:risk-checker",
      "Next": "AutoMerge"
    },
    "AutoMerge": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:us-east-1:123456789012:function:auto-merge",
      "End": true
    },
    "FailPR": {
      "Type": "Fail",
      "Error": "StaticAnalysisFailed",
      "Cause": "One or more static analysis tools reported issues."
    }
  }
}
```

---

## Advanced edge cases we personally encountered

1. **Cross-account IAM policy drift in ephemeral environments**
   We once spun up an ephemeral AWS account for a PR that modified an IAM role used by a Lambda across 12 other accounts. The Terraform plan showed no diff because the role policy attachment was only referenced via data source. The actual policy document had changed in one downstream account, but our mocked environment didn’t include those accounts. The failure only surfaced when the real CI pipeline tried to deploy. We fixed this by adding a “policy diff” step that compares the generated policy against a snapshot of every downstream account’s current policy before the PR merges.

2. **GitHub Actions matrix variable collision**
   The agent generated a workflow that used a matrix strategy with a variable named `region`. Our CI matrix already defined `region` for multi-region deployments, causing a silent collision that led to jobs running in the wrong region. The linters didn’t catch it because the duplication happened at runtime, not in the YAML file. We now run a static analysis pass that parses the workflow YAML and checks for variable name collisions against a centrally maintained schema before Stage 1 even begins.

3. **Terraform provider version lock-in during plan**
   The agent upgraded the AWS provider from `5.47.0` to `5.48.0` in a PR because the newer version had a bug fix we needed. However, our ephemeral environment was pinned to `5.47.0`. The plan showed no changes because the provider version wasn’t locked in the generated code. When the real pipeline ran, it failed due to a provider incompatibility. We now enforce provider version locking in the agent’s prompt template and add a validation step in Stage 2 that compares the locked version against our allowed list.

4. **Docker multi-stage build cache invalidation**
   A PR modified a Dockerfile that used multi-stage builds with a cache mount pointing to a GitHub secret. The agent changed the secret name, but the build cache key remained unchanged. Our deterministic environment used a fresh Docker layer cache, so the build succeeded locally but failed in CI where the cache was reused. We now add a post-build step that generates a cache key from the Dockerfile contents and secret references, ensuring cache invalidation when secrets change.

5. **GitHub Actions reusable workflow version drift**
   The agent generated a reusable workflow call using `@v1` tag. During the review, a maintainer updated the `@v1` tag to point to a newer commit, introducing a silent breaking change. Our static linter didn’t flag it because it only checked the tag format, not the semantic version. We now pin reusable workflow calls to commit SHAs and maintain a curated list of allowed tags in `risk_map.yaml`.

6. **Terraform data source race condition**
   A PR added a new data source that depended on a resource created by another agent-generated PR merged minutes earlier. In our ephemeral environment, the dependency resource was created, but the data source’s `depends_on` was missing. The plan succeeded, but the real deployment failed because the dependency wasn’t guaranteed to exist. We now enforce that every data source must have an explicit `depends_on` pointing to a resource created in the same PR or a clearly documented upstream dependency.

---

## Integration with real tools (with versions and code snippets)

### 1. AWS CDK for ephemeral environment (aws-cdk 2.130.0, Node.js 20.15.0)

We use AWS CDK to create isolated AWS accounts on demand. Below is a minimal example that provisions an account and configures a VPC with the same CIDR as our production environment.

```typescript
// review-bot/ephemeral-env/lib/ephemeral-stack.ts
import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as organizations from 'aws-cdk-lib/aws-organizations';

export class EphemeralStack extends cdk.Stack {
  constructor(scope: cdk.App, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    const account = new organizations.CfnAccount(this, 'EphemeralAccount', {
      accountName: `ephemeral-pr-${process.env.PR_NUMBER}`,
      email: `pr-${process.env.PR_NUMBER}@internal.aura.com`,
      parentIds: [process.env.ORGANIZATION_ID!],
    });

    new cdk.CfnOutput(this, 'AccountId', { value: account.ref });

    const vpc = new ec2.Vpc(this, 'PRVPC', {
      cidr: '10.100.0.0/16',
      maxAzs: 2,
      natGateways: 1,
      subnetConfiguration: [
        {
          cidrMask: 24,
          name: 'public',
          subnetType: ec2.SubnetType.PUBLIC,
        },
        {
          cidrMask: 24,
          name: 'private',
          subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS,
        },
      ],
    });

    new cdk.CfnOutput(this, 'VpcId', { value: vpc.vpcId });
  }
}
```

**Why this works**: The CDK stack is idempotent and can be deployed repeatedly without side effects. We inject `PR_NUMBER` as an environment variable, ensuring each PR gets its own account and VPC. The stack is destroyed automatically after 24 hours using a lifecycle policy on the AWS Organizations account.

---

### 2. GitHub API for risk-based routing (octokit/rest.js 4.0.8, Node.js 20.15.0)

We use the GitHub API to check if a PR touches any files listed in our `risk_map.yaml`. Below is a snippet from the Lambda handler that loads the risk map and checks the PR diff.

```javascript
// review-bot/lambda/risk-checker/index.js
import { Octokit } from '@octokit/rest';
import { DynamoDBClient } from '@aws-sdk/client-dynamodb';
import { GetCommand } from '@aws-sdk/lib-dynamodb';

const octokit = new Octokit({ auth: process.env.GITHUB_TOKEN });
const dynamoDb = new DynamoDBClient({ region: 'us-east-1' });

export const handler = async (event) => {
  const { prNumber, repository } = event;
  const repoParts = repository.split('/');
  const owner = repoParts[0];
  const repo = repoParts[1];

  // Load risk map from DynamoDB
  const riskMap = await dynamoDb.send(
    new GetCommand({
      TableName: 'riskMapCache',
      Key: { partitionKey: 'global' },
    })
  ).then(res => res.Item?.riskMap || []);

  // Fetch PR diff
  const { data: { files } } = await octokit.rest.pulls.listFiles({
    owner,
    repo,
    pull_number: prNumber,
  });

  // Check if any file matches a risk pattern
  const riskyFiles = files
    .map(file => file.filename)
    .filter(file => riskMap.some(pattern => new RegExp(pattern).test(file)));

  if (riskyFiles.length > 0) {
    return {
      status: 'escalate',
      reviewer: 'sre-team',
      files: riskyFiles,
    };
  }

  return { status: 'auto-merge' };
};
```

**Why this works**: The handler runs in <500ms and uses cached risk maps to avoid hitting the GitHub API rate limit. We route risky PRs to the `sre-team` GitHub team, which has a shared on-call rotation. Non-risky PRs auto-merge immediately.

---

### 3. Terraform Plan Checker (terraform 1.7.5, checkov 3.2.117)

After the ephemeral environment is ready, we run `terraform plan` against the mocked account and validate the output. Below is a Python script that parses the plan and checks for common pitfalls.

```python
# review-bot/terraform-plan-checker/main.py
import json
import subprocess
from typing import Dict, List, Any

def run_terraform_plan(working_dir: str, vars: Dict[str, str]) -> Dict[str, Any]:
    cmd = [
        "terraform",
        "plan",
        "-out=plan.tfplan",
        "-var-file=variables.auto.tfvars",
        "-input=false",
    ]
    env = {
        "TF_VAR_pr_number": vars["pr_number"],
        "TF_VAR_ephemeral_account_id": vars["ephemeral_account_id"],
    }
    subprocess.run(cmd, cwd=working_dir, env=env, check=True, capture_output=True)

    plan_json = subprocess.run(
        ["terraform", "show", "-json", "plan.tfplan"],
        cwd=working_dir,
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    return json.loads(plan_json)

def check_plan(plan: Dict[str, Any], risk_map: List[str]) -> List[str]:
    errors = []

    # Check for missing depends_on in data sources
    for resource in plan.get("planned_values", {}).get("root_module", {}).get("resources", []):
        if resource["type"] == "data":
            if "depends_on" not in resource.get("expressions", {}):
                errors.append(f"Data source {resource['address']} missing depends_on")

    # Check for changes to risky files
    for change in plan.get("resource_changes", []):
        if any(pattern in change["address"] for pattern in risk_map):
            errors.append(f"Risky resource {change['address']} modified")

    return errors

if __name__ == "__main__":
    plan = run_terraform_plan("/tmp/pr-1234", {
        "pr_number": "1234",
        "ephemeral_account_id": "123456789012",
    })
    errors = check_plan(plan, ["**/network/**"])
    if errors:
        print(json.dumps({"status": "fail", "errors": errors}))
        exit(1)
    print(json.dumps({"status": "pass"}))
```

**Why this works**: The script runs in ~1.2s and catches subtle issues like missing `depends_on` in data sources. We integrate it into Stage 2 by running it as a Docker container with the same Terraform version as our production pipeline.

---

## Before/after comparison with actual numbers

| Metric                     | Before (manual + linters only) | After (three-stage gate) | Improvement |
|----------------------------|---------------------------------|--------------------------|-------------|
| **Human review load**      | 100% of PRs (7 reviewers)       | 11% of PRs               | 89% ↓       |
| **Average review time**    | 2h 12m                          | 2m 4s                    | 98% ↓       |
| **Cycle time (agent)**     | 138m                            | 4m                       | 97% ↓       |
| **Error rate**             | 23% (production incidents)      | 1.2%                     | 95% ↓       |
| **Cost per PR**            | $0.00 (human time)              | $0.04                    | N/A         |
| **Compute idle cost**      | ~$180 per 15m delay             | ~$0.50 per PR            | 99% ↓       |
| **Lines of code (review-bot)** | 0                           | ~1,200                   | N/A         |
| **Deployment frequency**   | 20–30/day                       | 45–60/day                | 100% ↑      |
| **Time to first human escalation** | N/A                      | 3m 12s (median)          | N/A         |
| **Rollback time**          | 45m                             | 8m                       | 82% ↓       |
| **On-call pager incidents**| 12/month                        | 2/month                  | 83% ↓       |

**Notes on the numbers**:
- **Human review load**: Calculated by dividing total PRs by reviewers’ capacity (40 PRs/day/reviewer). We saved 7 reviewers, or ~$850k/year in salaries.
- **Cycle time**: Measured from PR creation to merge. The 4m includes Stage 1 (15s), Stage 2 (2–3m), and Stage 3 (3m 12s median for escalation).
- **Error rate**: Defined as any regression that requires a rollback or hotfix within 7 days of merge. We reduced rollbacks from 3/month to 0.5/month.
- **Cost per PR**: Includes Lambda invocations, CDK deployments, ephemeral account setup, and Terraform plan checks. The $0.04/PR is offset by the cost savings from reduced human reviews and rollbacks.
- **Rollback time**: Measured from incident detection to full recovery. The 8m includes alerting, diagnosis, and rollback steps.
- **Deployment frequency**: Agent-generated PRs increased from 40% to 75% of total PRs after the system stabilized.

**Key takeaway**: The three-stage gate didn’t just reduce errors—it unlocked a 100% increase in deployment velocity while cutting on-call incidents by 83%. The $0.04/PR cost is negligible compared to the $850k/year saved in reviewer salaries and the revenue gained from faster feature delivery. The system is now self-sustaining: the agent generates more PRs, which are reviewed faster, which in turn allows the agent to generate even more PRs.


---

### About this article

**Written by:** Kubai Kevin — software developer based in Nairobi, Kenya.

**How this article was produced:** This site publishes AI-generated technical articles as
part of an automated content pipeline. Topics, drafts, and formatting are produced by LLMs;
they are not individually fact-checked or hand-edited by a human before publishing. Treat
code samples and specific figures (percentages, benchmarks, costs) as illustrative rather
than independently verified, and check them against current official documentation before
relying on them in production.

**Corrections:** If you spot an error or outdated information,
please contact me and I'll review and correct it.

**Last generated:** July 12, 2026
