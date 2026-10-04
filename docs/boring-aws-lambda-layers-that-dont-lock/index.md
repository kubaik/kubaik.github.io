# Boring AWS Lambda layers that don’t lock…

## The problem layers are supposed to solve

A Lambda layer is a ZIP archive that the Lambda service extracts into `/opt` inside the execution environment. Because the mount path is stable, application code can import from `/opt/...` without knowing which layer version supplied it. That indirection is the whole point: it lets the layer change underneath a function without the function's code changing.

The failure mode is that teams treat the layer as a one-off artifact rather than a versioned dependency. A single layer is created once, its ARN is pasted into a CloudFormation template or a function's configuration, and everything works until one of three things happens:

- The runtime is upgraded (for example, `nodejs18.x` to `nodejs20.x`), and the layer's `CompatibleRuntimes` list no longer covers the new runtime.
- The workload moves to a different region, and the ARN—which embeds the region—no longer resolves.
- The architecture changes from `x86_64` to `arm64`, and the compiled contents of the layer no longer match.

Each of these is a configuration change, not a code change, but a hard-coded ARN turns it into an outage. The patterns below keep the layer evolvable so these transitions are routine.

## What you'll build

1. A shared layer containing only runtime-agnostic helpers, with no versioned third-party dependencies.
2. A versioning scheme where the layer version is a stack parameter rather than a literal ARN.
3. A CloudFormation template that composes the layer ARN from `AWS::Region`, `AWS::AccountId`, the layer name and the version.
4. A deployment script that publishes a new layer version and updates every dependent stack in one pass.

The layer payload itself is deliberately tiny. The value is in the plumbing around it.

## Prerequisites

- An AWS account with permission to create Lambda layers, Lambda functions and CloudFormation stacks.
- AWS CLI v2 installed and configured with a named profile.
- The runtime you target installed locally (Node.js 20 LTS is used in the examples).
- A text editor and a shell.

## Step 1 — scaffold the project

```bash
mkdir lambda-layer-template && cd lambda-layer-template
npm init -y
mkdir -p layer/nodejs src
```

Create the layer payload. Keep it free of anything versioned so it never needs rebuilding for a dependency bump:

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

The Lambda Node.js runtime adds `/opt/nodejs` to the module resolution path, so consumers import it as if it were a normal package:

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

Note the import path: `/opt/nodejs/index`. It is fixed by the runtime and does not change when the layer version changes. That is what makes the function immune to layer churn.

## Step 2 — declare the layer and function in a template

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

  LayerVersion:
    Type: String
    Default: "1"
    Description: Which published layer version to attach

Resources:
  ApiFunction:
    Type: AWS::Serverless::Function
    Properties:
      CodeUri: ./src/
      Handler: index.handler
      Runtime: nodejs20.x
      Layers:
        - !Sub 'arn:${AWS::Partition}:lambda:${AWS::Region}:${AWS::AccountId}:layer:${LayerName}:${LayerVersion}'
      Events:
        Api:
          Type: Api
          Properties:
            Path: /ping
            Method: GET
```

Two things matter here. First, the ARN is composed with `Fn::Sub` from pseudo-parameters, so the same template works in any region and any account. Second, the layer version is a stack parameter, so promoting a new version is a parameter override rather than a template edit.

If you prefer to let CloudFormation own the layer's lifecycle, `AWS::Lambda::LayerVersion` and `AWS::Serverless::LayerVersion` both work. The trade-off is that CloudFormation-created layers are tied to the stack that created them, which complicates sharing a layer across stacks. Publishing the layer out-of-band and referencing it by ARN keeps the layer independent of any one stack; that is the pattern used here.

### The retention gotcha

When a layer is created inside a stack, deleting or updating the stack can delete layer versions that other functions still reference. `AWS::Lambda::LayerVersion` supports a `RetentionPolicy` property; setting it to `Retain` prevents CloudFormation from deleting the version on stack teardown. Functions that reference a deleted version fail at invocation with a `LayerVersionNotFound` error, and the failure is not obvious from the function's own configuration. For layers managed inside a stack that other stacks depend on, `Retain` is the safe default.

## Step 3 — publish the layer

Package and publish:

```bash
cd layer
tar -czf ../layer.zip .
cd ..

aws lambda publish-layer-version \
  --layer-name shared-helpers \
  --zip-file fileb://layer.zip \
  --compatible-runtimes "nodejs20.x" "nodejs18.x" \
  --description "Runtime-agnostic helpers"
```

The response includes the new version number and its ARN:

```json
{
  "LayerArn": "arn:aws:lambda:us-east-1:123456789012:layer:shared-helpers",
  "LayerVersionArn": "arn:aws:lambda:us-east-1:123456789012:layer:shared-helpers:1",
  "Version": 1,
  "Description": "Runtime-agnostic helpers",
  "CompatibleRuntimes": ["nodejs20.x", "nodejs18.x"]
}
```

The version number is what you feed into the `LayerVersion` parameter. The ARN is what the template reconstructs.

Note that `CompatibleRuntimes` is advisory metadata, not an enforcement mechanism. Lambda does not reject a layer whose `CompatibleRuntimes` list omits the function's runtime; the layer is still mounted. The list is used by the console and by tooling to surface mismatches. Treat it as documentation that keeps humans honest, and add the new runtime to it whenever you upgrade.

### A worked example: bumping the runtime

Suppose a stack currently runs `nodejs18.x` and you want to move to `nodejs20.x`. The steps, in order, are:

1. Rebuild the layer if any part of it is runtime-specific. Pure JavaScript helpers are not, so no rebuild is needed; if the layer contains native modules, they must be rebuilt against the Node 20 ABI.
2. Publish a new layer version with `nodejs20.x` added to `--compatible-runtimes`.
3. Change the function's `Runtime` property in the template to `nodejs20.x`.
4. Pass the new layer version as the `LayerVersion` parameter.
5. Deploy.

Because the function imports from `/opt/nodejs/index` and the ARN is composed from the parameter, steps 3 and 4 are the only changes. No function code is touched. This is the payoff the pattern is designed to deliver.

## Step 4 — automate publish and deploy

A script that publishes a new version and updates the stack:

```bash
#!/usr/bin/env bash
set -euo pipefail

LAYER_NAME="shared-helpers"
STACK_NAME="shared-layer-stack"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$DIR/layer"
tar -czf ../layer.zip .
cd "$DIR"

VERSION=$(aws lambda publish-layer-version \
  --layer-name "$LAYER_NAME" \
  --zip-file fileb://layer.zip \
  --compatible-runtimes "nodejs20.x" "nodejs18.x" \
  --query 'Version' \
  --output text)

aws cloudformation deploy \
  --template-file template.yaml \
  --stack-name "$STACK_NAME" \
  --parameter-overrides LayerVersion="$VERSION" \
  --capabilities CAPABILITY_IAM

echo "Layer version $VERSION published and stack updated"
```

`aws cloudformation deploy` is preferred over `update-stack` for this because it creates the stack if it does not exist and waits for completion. `--capabilities CAPABILITY_IAM` is required whenever the template can create IAM resources; without it the deploy fails with an explicit error naming the capability, not silently.

### A worked example: moving regions

The ARN embeds the region, so a layer published in `us-east-1` does not exist in `eu-central-1`. The layer must be published once per region. The contents are identical; only the ARN differs.

```bash
#!/usr/bin/env bash
set -euo pipefail

REGIONS=("us-east-1" "eu-central-1" "ap-south-1")
LAYER_NAME="shared-helpers"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$DIR/layer"
tar -czf ../layer.zip .
cd "$DIR"

for REGION in "${REGIONS[@]}"; do
  VERSION=$(aws lambda publish-layer-version \
    --layer-name "$LAYER_NAME" \
    --zip-file fileb://layer.zip \
    --compatible-runtimes "nodejs20.x" "nodejs18.x" \
    --region "$REGION" \
    --query 'Version' \
    --output text)

  aws cloudformation deploy \
    --template-file template.yaml \
    --stack-name "shared-layer-stack-$REGION" \
    --parameter-overrides LayerVersion="$VERSION" \
    --capabilities CAPABILITY_IAM \
    --region "$REGION"

  echo "Region $REGION updated to layer version $VERSION"
done
```

Because the template derives the ARN from `AWS::Region`, no template change is needed per region. The only per-region input is the version number, which the script obtains from the publish call.

## Step 5 — edge cases and failure modes

### Layer size limits

Lambda enforces a 50 MB limit on the zipped layer as uploaded via `publish-layer-version`, and a 250 MB unzipped limit on the extracted contents. The unzipped size is what actually matters, because a layer that unpacks past 250 MB will fail. To inspect the size of a published version:

```bash
aws lambda get-layer-version \
  --layer-name shared-helpers \
  --version-number 1 \
  --query 'Content.[CodeSize,UncompressedCodeSize]' \
  --output text
```

`UncompressedCodeSize` is reported directly by the API, so no local decoding is necessary. If a layer is approaching the limit, the usual cause is a bundled SDK. The AWS SDK for JavaScript v3 is available in the Lambda Node.js runtimes, so bundling it into a layer duplicates functionality and inflates the archive. Strip dev dependencies before packaging:

```bash
npm install --omit=dev
npm prune --production
```

### Runtime import failures

The error `Runtime.ImportModuleError: Cannot find module '/opt/nodejs/index'` almost always means one of: the layer is not attached to the function, the layer was attached but the module path inside the ZIP is wrong, or the layer contents were built for a different architecture. The Node.js runtime expects the module under `nodejs/` at the ZIP root for Node runtimes; a ZIP whose top-level directory is `layer/nodejs/` will not resolve. Verify the archive layout:

```bash
unzip -l layer.zip
```

The listing should show `nodejs/index.js`, not `layer/nodejs/index.js`.

### Architecture mismatch

Layers built for `x86_64` and `arm64` are not interchangeable when they contain native binaries. A pure-JavaScript layer works on both. If a layer contains compiled code, publish it once per architecture or build it as a multi-architecture package. The function's `Architectures` property must match the layer's.

### Version pinning

Hard-coding a layer ARN in a function's configuration creates exactly the coupling this pattern avoids. If a literal ARN must be used somewhere—for example, a function managed outside CloudFormation—expose it as a parameter with a default so it can be overridden without editing the resource. Environment variables that capture the layer ARN are a similar trap: they do not update when the layer version changes, so any consumer reading them will silently use a stale version.

## Step 6 — verification and observability

### Verify the attached layer

After deploying, confirm which layer version a function is actually using rather than trusting the template:

```bash
aws lambda get-function-configuration \
  --function-name shared-layer-stack-ApiFunction-XXXXXX \
  --query 'Layers[].Arn' \
  --output text
```

This returns the resolved ARN, including the version suffix. Compare it against the version the stack was deployed with. A mismatch means the deploy did not complete or the function is managed by something else.

### Test the layer locally

The layer's helpers can be tested without deploying by requiring them directly from the source path:

```javascript
// tests/layer.test.js
const { logRequest } = require('../layer/nodejs/index');

describe('shared helpers layer', () => {
  it('logs without throwing', () => {
    const event = { path: '/ping', httpMethod: 'GET' };
    expect(() => logRequest(event)).not.toThrow();
  });
});
```

```bash
npm install --save-dev jest
npx jest tests/layer.test.js
```

The import path differs from the deployed one (`/opt/nodejs/index`), which is worth noting: the test exercises the code, not the mount path. To verify the mount path, invoke the deployed function.

### Invoke the deployed function

```bash
aws lambda invoke \
  --function-name shared-layer-stack-ApiFunction-XXXXXX \
  --payload '{"path":"/ping","httpMethod":"GET"}' \
  --cli-binary-format raw-in-base64-out \
  /tmp/out.json

cat /tmp/out.json
```

A `200` response confirms the layer mounted and the import resolved. A `Runtime.ImportModuleError` confirms it did not.

### Alarm on invocation errors

CloudWatch can alarm on Lambda errors for the function that consumes the layer:

```yaml
LayerErrorAlarm:
  Type: AWS::CloudWatch::Alarm
  Properties:
    AlarmName: "shared-helpers-consumer-errors"
    ComparisonOperator: GreaterThanThreshold
    EvaluationPeriods: 1
    MetricName: "Errors"
    Namespace: "AWS/Lambda"
    Dimensions:
      - Name: "FunctionName"
        Value: !Ref ApiFunction
    Period: 60
    Statistic: Sum
    Threshold: 1
    AlarmActions:
      - !Ref AlertTopic
```

Alarm on the consuming function, not on the layer. Lambda does not emit per-layer error metrics; the errors surface against the function that failed to import. `Threshold: 1` over a 60-second period is sensitive enough to catch a broken deploy quickly and quiet enough to avoid noise from isolated failures.

### How to measure the cost of the pattern

The overhead of a separate layer versus bundling helpers into each function is straightforward to measure. Compare, for a representative function, the value of `CodeSize` returned by `aws lambda get-function-configuration` with and without the helpers bundled, and the cold-start duration reported in CloudWatch's `InitDuration` metric for the same function. A shared layer typically reduces per-function package size at the cost of one additional mount during init. Instrument both, deploy both, and compare the distributions rather than a single invocation.

## Decision checklist

Before adopting a shared layer, confirm:

- The helpers are runtime-agnostic, or you accept rebuilding per runtime.
- The layer will be published out-of-band, or you accept stack-scoped lifecycle.
- `RetentionPolicy: Retain` is set if the layer lives in a stack others depend on.
- The template composes the ARN from pseudo-parameters rather than a literal.
- The layer version is a stack parameter.
- `CompatibleRuntimes` is updated alongside every runtime upgrade.
- The ZIP layout matches what the runtime expects (`nodejs/` at the root for Node.js).
- The function's `Architectures` matches the layer's.
- An alarm exists on the consuming function's `Errors` metric.
- A verification step reads back the attached ARN after deploy.

If any of these cannot be satisfied, the layer will become a coupling point rather than an abstraction, and the first runtime upgrade will be as painful as the problem the layer was meant to solve.

## Do this in the next 30 minutes

Pick one existing Lambda function that imports from a layer via a hard-coded ARN. Run `aws lambda get-function-configuration --function-name <name> --query 'Layers[].Arn' --output text` to capture the current ARN, then replace the literal in its template with `!Sub 'arn:${AWS::Partition}:lambda:${AWS::Region}:${AWS::AccountId}:layer:${LayerName}:${LayerVersion}'` and add `LayerName` and `LayerVersion` parameters with the current values as defaults. Deploy, then re-run the same command and confirm the resolved ARN is unchanged. You have now made that function survive a region move and a runtime upgrade without a code edit.
