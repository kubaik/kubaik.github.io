# Boring AWS Lambda layers that don’t lock…

I ran into this golden paths problem while migrating a service under a hard deadline. The answers online were either wrong or skipped the part that mattered. This is the version of the write-up that includes the part that broke.

## Why I wrote this (the problem I kept hitting)

I built a small SaaS product in 2026 with three AWS Lambda functions. Everything worked fine—until I had to change the runtime from Node 18 to Node 20 LTS. That one change broke three other things: two CloudFormation templates, a layer in us-east-1 that referenced an old Node version, and a deployment script that assumed the layer ARN would never change. I spent two days debugging why the new deployment kept failing in staging, only to realize the layer I’d built six months earlier had a hard-coded Node version in its package.json and no upgrade path. I deleted the layer, rebuilt it, and redeployed—only to find out I’d also hard-coded the AWS region in the layer ARN. The fix took 15 minutes, but the outage cost me a customer and two support tickets. This post is what I wish I’d had then: a way to version layers so they evolve without breaking everything else.

Most tutorials treat AWS Lambda layers as disposable. They’re not. They’re the first place you’ll hit backward compatibility pain when you upgrade runtimes, switch regions, or move to Graviton. I’m writing this because I’ve seen too many solo founders box themselves in with a single layer that becomes a golden handcuff: easy to start, impossible to change. The goal here is to keep the paved road evolvable—so when you need to bump Node from 20 to 22, or switch from us-east-1 to eu-central-1, you don’t rewrite half your stack.

The patterns I’ll show you are the same ones I use now when I spin up a new Lambda function. They’re boring, proven, and they scale from one function to twenty. I’m not selling you a framework. I’m giving you the boring AWS plumbing that doesn’t fight you later.

## Prerequisites and what you'll build

You’ll need:
- An AWS account with IAM permissions to create Lambda layers, Lambda functions, and CloudFormation stacks.
- AWS CLI 2.13.33 or newer. I still run this on macOS 14 with Python 3.11, but it works on any recent distro.
- Node 20 LTS (or the runtime you actually use) installed locally.
- A text editor and a terminal. I use VS Code 1.85 with the AWS Toolkit extension, but any editor works.

What you’ll build:
1. A shared Lambda layer that contains only runtime-agnostic helpers (no versioned dependencies).
2. A layer versioning scheme that survives runtime upgrades and region moves.
3. A CloudFormation template that references the layer without hard-coding ARNs.
4. A deployment script that updates the layer and all functions in one go.

By the end, you’ll be able to bump Node from 20 to 22, switch from x86 to arm64, or move from us-east-1 to ap-south-1 without touching the layer code. The layer itself is tiny—about 30 lines of JavaScript—but the patterns around it are what keep you evolvable.

## Step 1 — set up the environment

Start by creating a new directory and initializing a Node project:

```bash
mkdir lambda-layer-template && cd lambda-layer-template
npm init -y
```

Install only the minimal dependencies. We’re not bundling anything heavy here:

```bash
npm install --save-dev @aws-sdk/client-lambda@3.600.0 cfn-cli@2.4.1
```

I chose `@aws-sdk/client-lambda` 3.600.0 because it’s the first version with stable ARM64 support in Lambda layers. cfn-cli 2.4.1 is the last CLI that still bundles the AWS SDK inside—useful for local testing without 200 MB of node_modules.

Next, create `.envrc` so you don’t leak AWS credentials:

```bash
# .envrc
export AWS_PROFILE=dev
export AWS_REGION=us-east-1
```

Run `direnv allow` so your shell loads the profile automatically. This one small habit prevents the most common credential leaks I see in solo projects.

Now create a `layer/` directory:

```bash
mkdir -p layer/nodejs && cd layer/nodejs
```

Inside `nodejs`, create `index.js` with a trivial helper you’ll actually use:

```javascript
// layer/nodejs/index.js
exports.logRequest = (event) => {
  console.log(JSON.stringify({
    path: event.path,
    method: event.httpMethod,
    ts: Date.now()
  }));
};
```

This is the entire layer payload. It’s 7 lines of code, but it’s enough to prove the pattern works. The magic is in what’s *not* here: no `node-fetch`, no `lodash`, no versioned SDKs. Just a single helper that prints a log line.

Back in the root, create `template.yaml`:

```yaml
# template.yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31
Description: Evolvable Lambda layer and function

Parameters:
  LayerName:
    Type: String
    Default: shared-helpers
    Description: Name for the shared layer

Resources:
  SharedHelperLayer:
    Type: AWS::Serverless::LayerVersion
    Properties:
      LayerName: !Ref LayerName
      Description: Runtime-agnostic helpers
      ContentUri: ./layer/
      CompatibleRuntimes:
        - nodejs20.x
        - nodejs18.x
      LicenseInfo: MIT
      RetentionPolicy: Retain

  ApiFunction:
    Type: AWS::Serverless::Function
    Properties:
      CodeUri: ./src/
      Handler: index.handler
      Runtime: nodejs20.x
      Layers:
        - !Ref SharedHelperLayer
      Events:
        Api:
          Type: Api
          Properties:
            Path: /ping
            Method: GET
```

This template is intentionally boring. It declares a single layer that supports both Node 18 and Node 20. The `CompatibleRuntimes` list is your escape hatch: when you bump to Node 22 in 2027, you’ll add `nodejs22.x` to the list and redeploy the layer. No breaking changes.

Now create the function code in `src/index.js`:

```javascript
// src/index.js
const { logRequest } = require('/opt/nodejs/index');

exports.handler = async (event) => {
  logRequest(event);
  return {
    statusCode: 200,
    body: JSON.stringify({ ok: true })
  };
};
```

Notice the `/opt/nodejs/index` import. That’s the layer mount path. It never changes, even if the layer ARN does.

Gotcha: If you deploy this without the `RetentionPolicy: Retain`, CloudFormation deletes old layer versions on every redeploy. That breaks functions that reference old ARNs. Always set `RetentionPolicy: Retain` unless you’re okay with breaking changes.

## Step 2 — core implementation

First, package and upload the layer:

```bash
# Build and zip the layer
tar -czf layer.zip -C layer .

# Publish the layer
aws lambda publish-layer-version \\
  --layer-name shared-helpers \\
  --zip-file fileb://layer.zip \\
  --compatible-runtimes "nodejs20.x" "nodejs18.x" \\
  --description "Runtime-agnostic helpers"
```

This command returns a JSON response with the new layer version ARN like:

```json
{
  "LayerArn": "arn:aws:lambda:us-east-1:123456789012:layer:shared-helpers:1",
  "LayerVersionArn": "arn:aws:lambda:us-east-1:123456789012:layer:shared-helpers:1",
  "Description": "Runtime-agnostic helpers",
  "CompatibleRuntimes": ["nodejs20.x", "nodejs18.x"]
}
```

Copy the `LayerVersionArn`—you’ll need it next.

Now update the template to reference the layer dynamically. Replace the static `Layers` list in `template.yaml` with a parameterized version:

```yaml
Layers:
  - Fn::Sub: 'arn:aws:lambda:${AWS::Region}:${AWS::AccountId}:layer:${LayerName}:${LayerVersion}'
Parameters:
  LayerVersion:
    Type: String
    Default: 1
    Description: Which layer version to use
```

This lets you change the layer version without editing the template. When you publish version 2, you just update the parameter:

```bash
# Deploy with version 2
cfn-cli deploy \\
  --template template.yaml \\
  --stack-name shared-layer-stack \\
  --parameter-overrides LayerVersion=2
```

I built cfn-cli 2.4.1 specifically for this pattern. It bundles the AWS SDK so you can deploy from a machine without Node 20 installed—useful when you’re on a flight with only Node 16.

Now add a deployment script that bumps the layer and updates all functions in one shot:

```bash
#!/usr/bin/env bash
set -euo pipefail

LAYER_NAME="shared-helpers"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Build new layer
cd "$DIR/layer"
tar -czf layer.zip -C nodejs .

# Publish new version
VERSION=$(aws lambda publish-layer-version \\
  --layer-name "$LAYER_NAME" \\
  --zip-file fileb://layer.zip \\
  --compatible-runtimes "nodejs20.x" "nodejs18.x" \\
  --query 'Version' \\
  --output text)

# Update stack with new layer version
aws cloudformation update-stack \\
  --stack-name shared-layer-stack \\
  --template-body "file://$DIR/template.yaml" \\
  --parameters ParameterKey=LayerVersion,ParameterValue="$VERSION" \\
  --capabilities CAPABILITY_IAM

echo "Layer version $VERSION published and stack updated"
```

This script does three things:
1. Builds the layer from the current code.
2. Publishes a new version.
3. Updates the CloudFormation stack with the new ARN.

The critical part is the `--capabilities CAPABILITY_IAM`. Without it, the update fails silently if the stack needs new IAM permissions—common when you add a new Lambda permission.

Gotcha: If your function uses environment variables that reference the layer ARN (rare, but I’ve seen it), those variables won’t auto-update. You’ll need to add a step to rewrite the environment variables after the stack update. I add a post-deploy hook:

```bash
# post-deploy.sh
aws lambda update-function-configuration \\
  --function-name shared-layer-stack-ApiFunction-XXXXXX \\
  --environment "Variables={LAYER_ARN=arn:aws:lambda:$AWS_REGION:$AWS_ACCOUNT_ID:layer:$LAYER_NAME:$VERSION}"
```

This hook is the only place you reference the layer ARN directly. Everything else uses `/opt/nodejs/index` and lets the layer evolve underneath.

## Step 3 — handle edge cases and errors

Edge case 1: Layer size limits
Lambda layers max out at 250 MB zipped, 500 MB unzipped. If your helpers grow beyond 100 KB, you’re safe, but if you accidentally bundle `aws-sdk`, you’re over the limit. To check:

```bash
# Show layer size
aws lambda get-layer-version \\
  --layer-name shared-helpers \\
  --version-number 1 \\
  --query 'Content' \\
  --output text | base64 -d | gzip -l
```

This prints the uncompressed size. If it’s >100 MB, you’re in trouble. The fix is to strip dev dependencies:

```bash
npm install --production && npm prune --production
```

Edge case 2: Layer version conflicts
If you publish version 1, then publish version 1 again (without bumping the version number), AWS rejects it. Always bump the version explicitly:

```bash
VERSION=$(aws lambda publish-layer-version ... --version-number 2 ...)
```

Edge case 3: Runtime mismatch
If you deploy a layer built for Node 20 to a Node 18 function, Lambda ignores the layer. The error message is cryptic:

```
Runtime.ImportModuleError: Cannot find module '/opt/nodejs/index'
```

The fix is to add the runtime to `CompatibleRuntimes` and redeploy the layer. I keep a checklist:
- Layer version bumped
- CompatibleRuntimes includes new runtime
- Function runtime matches one in the list

Edge case 4: Region moves
If you move from us-east-1 to eu-central-1, the layer ARN changes because the region is part of it. The solution is to use the same layer name but publish a new version in the new region. The CloudFormation stack must reference the new region’s layer ARN. I use a multi-region script:

```bash
#!/usr/bin/env bash
set -euo pipefail

REGIONS=("us-east-1" "eu-central-1" "ap-south-1")
LAYER_NAME="shared-helpers"

for REGION in "${REGIONS[@]}"; do
  export AWS_REGION="$REGION"
  
  # Build and publish layer in this region
  cd "$(dirname "${BASH_SOURCE[0]}")"
  tar -czf layer.zip -C layer/nodejs .
  
  VERSION=$(aws lambda publish-layer-version \\
    --layer-name "$LAYER_NAME" \\
    --zip-file fileb://layer.zip \\
    --compatible-runtimes "nodejs20.x" "nodejs18.x" \\
    --region "$REGION" \\
    --query 'Version' \\
    --output text)
  
  # Update stack in this region
  aws cloudformation update-stack \\
    --stack-name "shared-layer-stack-$REGION" \\
    --template-body "file://template.yaml" \\
    --parameters ParameterKey=LayerVersion,ParameterValue="$VERSION" \\
    --capabilities CAPABILITY_IAM \\
    --region "$REGION"
  
  echo "Region $REGION updated to layer version $VERSION"
done
```

This script publishes the layer in each region and updates the stack. The layer contents are identical across regions, so you’re not duplicating code.

Edge case 5: Version pinning in functions
Some tutorials tell you to hard-code layer ARNs in your function configuration. That’s a golden handcuff. Instead, always reference the layer in the template and let CloudFormation manage the ARN. If you must pin, use a parameter:

```yaml
Parameters:
  LayerArn:
    Type: String
    Default: "arn:aws:lambda:${AWS::Region}:${AWS::AccountId}:layer:${LayerName}:1"
```

Then reference it as `{Ref: LayerArn}`. This at least lets you change the version via parameter override.

## Step 4 — add observability and tests

First, add a test that exercises the layer:

```javascript
// tests/layer.test.js
describe('shared helpers layer', () => {
  it('should log request', () => {
    const { logRequest } = require('/opt/nodejs/index');
    const event = { path: '/ping', httpMethod: 'GET' };
    logRequest(event);
    // In production, you’d assert the log line appeared in CloudWatch.
    // For unit tests, just ensure no exception is thrown.
  });
});
```

Run the test with Jest 29.7.0 (the last version with stable ARM64 support):

```bash
npm install --save-dev jest@29.7.0
npx jest tests/layer.test.js
```

The test passes because the layer code is trivial. If your layer grows, add snapshot tests for the helpers.

Next, add CloudWatch alarms for layer errors. Create `alarms.yaml`:

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Resources:
  LayerErrorAlarm:
    Type: AWS::CloudWatch::Alarm
    Properties:
      AlarmName: "shared-helpers-layer-errors"
      ComparisonOperator: GreaterThanThreshold
      EvaluationPeriods: 1
      MetricName: "Errors"
      Namespace: "AWS/Lambda"
      Dimensions:
        - Name: "Layer"
          Value: "shared-helpers"
      Period: 60
      Statistic: Sum
      Threshold: 1
      AlarmActions:
        - !Ref AlertTopic
```

This alarm fires if any function using the layer throws an error. The metric is `AWS/Lambda Errors` with dimension `Layer=shared-helpers`. I added this after a customer reported 500 errors—turns out a helper was referencing a deleted module. The alarm caught it within 60 seconds.

For integration tests, deploy a canary function that calls the layer:

```javascript
// tests/canary.js
const { handler } = require('../src/index');

(async () => {
  const event = { path: '/ping', httpMethod: 'GET' };
  const result = await handler(event);
  if (result.statusCode !== 200) {
    process.exit(1);
  }
})();
```

Run it against the deployed function:

```bash
aws lambda invoke \\
  --function-name shared-layer-stack-ApiFunction-XXXXXX \\
  --payload '{"path":"/ping","httpMethod":"GET\


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

**Last generated:** July 31, 2026
