# AWS CloudWatch Synthetics for AI pipelines

## Why endpoint-only canaries miss real outages

A health check that returns 200 proves one thing: the process answered. It does not prove that a user can complete the flow. In an AI pipeline the user flow typically spans several hops — an inference endpoint, a queue or job store, a payment rail, and a confirmation step. Any one of those can fail while the endpoint the canary pings stays green.

The failure mode is easy to describe and hard to notice. An upstream dependency starts returning intermittent 503s. Your endpoint-only canary still gets a 200 from the front door because the front door is up. Meanwhile users abandon checkout. The pager stays quiet until a human notices the drop in conversions, which is usually much later than the first failed request.

The fix is not more canaries. It is a canary that runs the same sequence of calls a real user triggers, asserts on each step, and alarms only when the whole journey fails or when a specific step fails repeatedly. This article walks through building that, from a local service to a scheduled canary with alarms and a dashboard.

## What you will build

1. A small Python service that simulates an AI pipeline endpoint with realistic latency and an occasional upstream failure.
2. A CloudWatch Synthetics canary script that performs a multi-step journey: request a prediction, submit a payment, confirm the result.
3. CloudWatch alarms that fire on journey failure, not on a single transient error.
4. Terraform that wires the canary, IAM role, bucket, and alarm together reproducibly.

The payment step is modelled on a mobile-money style API. Substitute whatever rail you actually use; the structure is the same.

## Prerequisites

- An AWS account with billing alarms configured. Synthetics is inexpensive per run, but a misconfigured payload or an aggressive schedule can multiply cost quickly.
- Node.js 20 LTS for the canary script.
- Python 3.11 or later for the simulated service.
- Terraform 1.5 or later.
- AWS CLI configured with credentials that can create IAM roles, S3 buckets, Lambda-backed canaries, and CloudWatch alarms.

## Step 1 — a service that behaves like the real thing

Install dependencies:

```bash
sudo apt update && sudo apt install -y python3.11 python3.11-venv git
python3.11 -m venv ./venv
source ./venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install flask==3.0.3 gunicorn==21.2.0 boto3==1.34.23
```

Create `app.py`:

```python
from flask import Flask, request, jsonify
import random
import time

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        body = request.get_json(silent=True) or {}
        # Simulate variable inference latency.
        time.sleep(random.uniform(0.05, 0.25))
        # Simulate an occasional upstream flake so the canary has something to detect.
        if random.random() < 0.02:
            return jsonify({"error": "upstream_timeout"}), 503
        return jsonify({"prediction": body.get('text', '')[:50]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
```

Note the change from the naive version: `request.json` raises on malformed input, so use `get_json(silent=True)` and default to an empty dict. A canary that sends a slightly wrong payload should get a clean 400-class response, not a stack trace.

Run it:

```bash
gunicorn --bind 0.0.0.0:8000 app:app
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"hello"}'
```

Expect a 200 with a short prediction most of the time, and a 503 roughly 2% of the time. That 2% is a deliberate injection so you can watch the canary fail and recover.

## Step 2 — write the journey canary

Create a directory `canary` and a file `journey.js`. The script uses the Synthetics runtime helpers rather than raw HTTP, so failures are recorded as step failures in the run report.

```javascript
const synthetics = require('Synthetics');
const log = require('SyntheticsLogger');

const TARGET_ENDPOINT = process.env.TARGET_ENDPOINT;
const PAYMENT_ENDPOINT = process.env.PAYMENT_ENDPOINT;
const PAYMENT_TOKEN = process.env.PAYMENT_TOKEN;

async function post(url, headers, body) {
  const response = await synthetics.getUrl({
    url,
    headers,
    method: 'POST',
    body,
  });
  return response;
}

const journey = async function () {
  // Step 1: inference request.
  const predictResponse = await post(
    `${TARGET_ENDPOINT}/predict`,
    { 'Content-Type': 'application/json' },
    JSON.stringify({ text: 'Generate a summary please' })
  );

  if (predictResponse.statusCode !== 200) {
    throw new Error(`AI endpoint failed: ${predictResponse.statusCode}`);
  }

  // Step 2: payment submission.
  const paymentResponse = await post(
    `${PAYMENT_ENDPOINT}/v1/payment`,
    {
      'Authorization': `Bearer ${PAYMENT_TOKEN}`,
      'Content-Type': 'application/json',
    },
    JSON.stringify({
      phone: '254712345678',
      amount: 100,
      reference: 'AI_JOB_001',
    })
  );

  if (paymentResponse.statusCode !== 200) {
    throw new Error(`Payment step failed: ${paymentResponse.statusCode}`);
  }

  // Step 3: confirmation.
  log.info(JSON.stringify({
    step: 'confirmation',
    status: 'ok',
    target: TARGET_ENDPOINT,
  }));
};

exports.handler = async () => {
  return await journey();
};
```

Two corrections worth noting against a common first draft:

- `synthetics.getUrl` takes a request options object. Passing `bodyS3Location: { bucket: '', key: '' }` with empty strings is not meaningful and can confuse the runtime; omit it.
- Credentials and endpoints belong in environment variables, not inline in the script. Anything hardcoded ends up in the canary artifact and in CloudWatch Logs.

Package it:

```bash
cd canary
npm init -y
npm install --save-dev jest@29.7.0
zip -r journey.zip journey.js package.json
```

The Synthetics runtime supplies its own Node modules; you only need to ship your script and any third-party packages you import.

## Step 3 — provision with Terraform

Create `main.tf`:

```hcl
terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.40"
    }
  }
}

provider "aws" {
  region = var.region
}

data "aws_caller_identity" "current" {}

resource "aws_s3_bucket" "canary_bucket" {
  bucket        = "${data.aws_caller_identity.current.account_id}-ai-pipeline-canary"
  force_destroy = true
}

resource "aws_iam_role" "canary_role" {
  name = "ai-pipeline-canary-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "canary_basic" {
  role       = aws_iam_role.canary_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "canary_secrets" {
  name = "canary-secrets-policy"
  role = aws_iam_role.canary_role.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["secretsmanager:GetSecretValue"]
      Resource = ["arn:aws:secretsmanager:${var.region}:${data.aws_caller_identity.current.account_id}:secret:payment-api-key-*"]
    }]
  })
}

resource "aws_sns_topic" "oncall" {
  name = "ai-pipeline-oncall"
}

resource "aws_synthetics_canary" "ai_pipeline_journey" {
  name                 = "ai-pipeline-real-user-journey"
  artifact_s3_location = "s3://${aws_s3_bucket.canary_bucket.id}/canary-artifacts/"
  execution_role_arn   = aws_iam_role.canary_role.arn
  handler              = "journey.handler"
  zip_file             = "s3://${aws_s3_bucket.canary_bucket.id}/journey.zip"
  runtime_version      = "syn-nodejs-puppeteer-5.9"
  start_canary         = true

  run_config {
    timeout_in_seconds = 120
    environment_variables = {
      TARGET_ENDPOINT  = var.target_endpoint
      PAYMENT_ENDPOINT = var.payment_endpoint
    }
  }

  schedule {
    expression = "rate(5 minutes)"
  }
}

resource "aws_cloudwatch_alarm" "journey_failure_alarm" {
  alarm_name          = "ai-pipeline-journey-failure-alarm"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods  = 2
  datapoints_to_alarm = 2
  metric_name         = "Failed"
  namespace           = "AWS/CloudWatchSynthetics"
  period              = 300
  statistic           = "Sum"
  threshold           = 1
  treat_missing_data  = "notBreaching"
  alarm_description   = "Full user journey failed on two consecutive runs"
  alarm_actions       = [aws_sns_topic.oncall.arn]
  dimensions = {
    CanaryName = aws_synthetics_canary.ai_pipeline_journey.name
  }
}

output "canary_arn" {
  value = aws_synthetics_canary.ai_pipeline_journey.arn
}
```

Two deliberate choices differ from a naive setup:

- `timeout_in_seconds` is set to 120 rather than 900. The AWS Lambda maximum is 900 seconds, but a canary that runs for 15 minutes on a 5-minute schedule overlaps itself and produces confusing metrics. Keep the timeout comfortably below the schedule interval.
- The alarm requires two consecutive failed runs before firing. A single failed run on a 2% injected failure rate is noise; two consecutive failures on independent runs is a much stronger signal.

Apply:

```bash
terraform init
terraform apply
```

Upload the zip after the bucket exists:

```bash
aws s3 cp canary/journey.zip \
  s3://$(aws sts get-caller-identity --query Account --output text)-ai-pipeline-canary/journey.zip
```

The canary picks up the artifact on its next scheduled run.

## Step 4 — failure modes and how to handle them

| Failure | Typical cause | Mitigation |
|---|---|---|
| Canary times out at the payment step | Third-party sandbox is slow or down | Set a per-request timeout, retry once, and assert on the final result rather than the first attempt |
| Canary passes but users still fail | Journey omits a step users actually take | Add the missing step; the canary is only as good as the sequence it models |
| Alarm fires on transient blips | Threshold set to a single failure | Require consecutive failures, or alarm on a rolling success percentage |
| Canary cannot reach a private ALB | ALB in a private subnet without a route from the canary's VPC | Run the canary in a VPC configuration that has a path to the ALB, or expose the endpoint through a public entry point |
| Artifact storage grows without bound | Screenshots retained on every run | Configure artifact retention; Synthetics supports a retention period on the artifact location |

A note on the runtime: the `syn-nodejs-puppeteer-*` runtimes bundle a headless browser, which is what makes them useful if you later need to drive a real UI. If your journey is pure HTTP, that browser is dead weight — it increases cold-start time and artifact size. For API-only journeys, consider a lighter runtime or accept the overhead knowingly.

Per-request timeouts matter more than people expect. A third-party API that hangs does not fail fast; it holds the canary open until the Lambda timeout, which consumes the full run budget and delays the next run.

```javascript
const response = await synthetics.getUrl({
  url: `${PAYMENT_ENDPOINT}/v1/payment`,
  headers: {
    'Authorization': `Bearer ${PAYMENT_TOKEN}`,
    'Content-Type': 'application/json',
  },
  method: 'POST',
  body: JSON.stringify({ phone: '254712345678', amount: 100, reference: 'AI_JOB_001' }),
});
```

There is no per-request timeout parameter in the Synthetics request options. Enforce it by keeping the canary timeout low and by treating any non-2xx response as a failure. If you need a hard per-call deadline, wrap the call in a `Promise.race` against a timer and throw on timeout.

## Step 5 — observability

Emit structured logs so you can query by step:

```javascript
log.info(JSON.stringify({
  step: 'ai_prediction',
  status: predictResponse.statusCode,
  region: process.env.AWS_REGION,
}));
```

Then in CloudWatch Logs Insights:

```
fields @timestamp, step, status
| filter step = "ai_prediction"
| stats count() by status
```

Build a dashboard with three widgets:

- `SuccessPercent` for the canary, so you see partial degradation before the alarm fires.
- `Duration` p99 of the full journey, to catch slow-but-passing runs.
- `Failed` count, split by canary name if you run more than one.

The `SuccessPercent` metric is the most useful of the three. A journey that succeeds 99% of the time is not healthy if the missing 1% clusters in a five-minute window.

## Step 6 — test the canary logic locally

You can unit-test the journey function without AWS by mocking the Synthetics module:

```javascript
const synthetics = require('Synthetics');
const { handler } = require('./journey');

jest.mock('Synthetics', () => ({
  getUrl: jest.fn(),
}));

jest.mock('SyntheticsLogger', () => ({
  info: jest.fn(),
}));

describe('AI pipeline journey', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('succeeds when both steps return 200', async () => {
    synthetics.getUrl
      .mockResolvedValueOnce({ statusCode: 200 })
      .mockResolvedValueOnce({ statusCode: 200 });
    await handler();
    expect(synthetics.getUrl).toHaveBeenCalledTimes(2);
  });

  it('throws when the AI endpoint returns 503', async () => {
    synthetics.getUrl.mockResolvedValueOnce({ statusCode: 503 });
    await expect(handler()).rejects.toThrow(/AI endpoint failed: 503/);
  });

  it('throws when the payment step returns 401', async () => {
    synthetics.getUrl
      .mockResolvedValueOnce({ statusCode: 200 })
      .mockResolvedValueOnce({ statusCode: 401 });
    await expect(handler()).rejects.toThrow(/Payment step failed: 401/);
  });
});
```

Run:

```bash
npx jest journey.test.js
```

The important detail: `jest.mock('Synthetics', ...)` must match the module name you `require`. In the canary runtime the module is provided by the platform; locally you mock it. The earlier draft mocked `'Synthetics'` but required `'Synthetics'` — keep those strings identical or the mock silently does nothing and the test hits the network.

## How to measure whether this is working

Do not trust a before/after table someone hands you. Instrument it yourself:

1. Record the number of pages your on-call rotation receives per week, from your paging tool, for four weeks before and four weeks after.
2. Record mean time to detect: the timestamp of the first failed canary run versus the timestamp of the first user-visible symptom in your support queue or error tracker.
3. Record the false-positive rate: failed canary runs that turned out not to correspond to a real user-facing problem, divided by total failed runs.
4. Record the Synthetics cost from the AWS Cost Explorer, filtered to the CloudWatch Synthetics service.

Those four numbers are the only ones that matter, and you can compute all of them from data you already have.

## Common questions

**Can I use a different payment rail?**
Yes. Replace the payment step with a call to whatever sandbox endpoint you use. The structure is identical: build a payload, send it with an auth header from an environment variable or Secrets Manager, assert on the status code. Keep the credentials out of the script.

**Can I run the canary from multiple regions?**
Yes. Define a second `aws_synthetics_canary` resource with a different name, region, and `TARGET_ENDPOINT`. Terraform handles this naturally via a provider alias or a separate module invocation. The value of multi-region canaries is distinguishing a regional outage from a global one; if only one region fails, the problem is likely regional infrastructure, not your code.

**What if the AI service is asynchronous?**
Model the polling loop in the canary. Post the job, receive a job ID, then poll the status endpoint until it completes or a deadline passes:

```javascript
const deadline = Date.now() + 30000;
let status = 'pending';

while (status === 'pending' && Date.now() < deadline) {
  const jobResponse = await synthetics.getUrl({
    url: `${TARGET_ENDPOINT}/jobs/${jobId}`,
    headers: { 'Content-Type': 'application/json' },
  });
  const body = JSON.parse(jobResponse.body);
  status = body.status;
  if (status === 'pending') {
    await new Promise((resolve) => setTimeout(resolve, 2000));
  }
}

if (status !== 'completed') {
  throw new Error(`Job did not complete within deadline; last status: ${status}`);
}
```

Keep the deadline well under the canary timeout so a stuck job produces a clean failure rather than a Lambda timeout.

**Should I use a Python runtime instead?**
Python runtimes exist for Synthetics, but for API-only journeys the runtime choice matters less than the journey design. If you need to drive a browser, the Puppeteer-based Node runtime is the more direct fit. If your journey is pure HTTP, pick whichever your team can maintain and keep the timeout budget in mind.

## One action for the next 30 minutes

Open the CloudWatch Synthetics console, choose **Create canary**, and pick the API canary blueprint. Paste the `journey.js` handler above, set `TARGET_ENDPOINT` to a real service you own, and create the canary. When the first run completes, open the run report and look at which step it reached. If it failed, you have just found the first step in your pipeline that a real user would also have failed at — and you found it before a customer did.
