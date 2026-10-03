# Resize Images at the Edge: A CloudFront and Lambda Guide

Tutorials usually show the happy path: an image goes in, a smaller image comes out. Production adds the parts that break — concurrent uploads, format negotiation, cold starts, and logs that grow faster than your traffic. This article builds a CloudFront + Lambda@Edge resizing pipeline and then works through the failure modes that show up once real clients hit it.

## What you will build

A CloudFront distribution in front of a private S3 bucket. An origin-response Lambda@Edge function intercepts the S3 response, resizes the image to one of three breakpoints (300 px, 600 px, 1200 px), negotiates WebP versus JPEG from the `Accept` header, and returns the transformed bytes with a long-lived cache header.

Design goals:

- Stream the object rather than buffering it fully in memory.
- Cache aggressively, because a resize is expensive and the result is deterministic for a given input and width.
- Fall back to the original bytes whenever anything goes wrong, so a resize bug degrades quality rather than availability.
- Emit enough telemetry to tell a slow resize apart from a cold start apart from an origin timeout.

### Prerequisites

- Node.js 20 LTS and npm.
- An AWS account with permissions for CloudFront, Lambda@Edge, S3, IAM and CloudWatch.
- An S3 bucket in a region you can reach, plus a domain you control if you want a custom distribution domain.
- The AWS CLI configured with a default region and JSON output.

Lambda@Edge has a hard constraint worth internalising before you write any code: functions must be created in `us-east-1` and are replicated to edge locations, and there is no environment-variable support at the edge. Configuration must be baked into the bundle or fetched at runtime.

## Step 1 — environment and bucket

Install Node 20 LTS (the exact patch version does not matter; pin whatever your CI uses):

```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs
node -v
```

Bootstrap the project:

```bash
mkdir edge-image-resize && cd edge-image-resize
npm init -y
npm install --save-dev aws-cdk typescript ts-node @types/node
npx cdk init app --language typescript
```

Create the bucket with versioning and public access blocked:

```bash
aws s3api create-bucket --bucket photos-example --region af-south-1 \
  --create-bucket-configuration LocationConstraint=af-south-1
aws s3api put-bucket-versioning --bucket photos-example \
  --versioning-configuration Status=Enabled
aws s3api put-public-access-block --bucket photos-example \
  --public-access-block-configuration \
  "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
```

Note the `--create-bucket-configuration` flag: outside `us-east-1`, `CreateBucket` requires an explicit `LocationConstraint` or it fails with `InvalidLocationConstraint`.

### Grant CloudFront read access

Use an origin access control (OAC), the current mechanism; origin access identity (OAI) is the older one and does not support newer features such as SSE-KMS reads. The bucket policy grants `cloudfront.amazonaws.com` `s3:GetObject` only when the request originates from your specific distribution:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {"Service": "cloudfront.amazonaws.com"},
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::photos-example/*",
      "Condition": {
        "StringEquals": {
          "AWS:SourceArn": "arn:aws:cloudfront::YOUR_ACCOUNT_ID:distribution/YOUR_DISTRIBUTION_ID"
        }
      }
    }
  ]
}
```

The `AWS:SourceArn` condition is what makes this safe. Without it, any CloudFront distribution in any account could read the bucket.

## Step 2 — the resize function

Two properties matter more than the resize itself: do not buffer the whole image, and never let a failure return a 5xx when the original bytes would do.

```javascript
const sharp = require('sharp');

const BREAKPOINTS = [300, 600, 1200];
const MAX_WIDTH = 1200;

exports.handler = async (event) => {
  const response = event.Records[0].cf.response;
  const request = event.Records[0].cf.request;

  // Only transform successful image responses.
  if (response.status !== '200') return response;

  const accept = (request.headers['accept'] || [{}])[0].value || '';
  const wantsWebp = accept.includes('image/webp');
  const format = wantsWebp ? 'webp' : 'jpeg';

  const querystring = request.querystring || '';
  const params = new URLSearchParams(querystring);
  const requested = parseInt(params.get('w'), 10);
  const width = BREAKPOINTS.includes(requested) ? requested : MAX_WIDTH;

  try {
    const body = Buffer.from(response.body, 'base64');
    const pipeline = sharp(body, {
      failOn: 'none',
      sequentialRead: true,
      limitInputPixels: false
    }).resize({ width, withoutEnlargement: true });

    const output = await (format === 'webp'
      ? pipeline.webp({ quality: 72 })
      : pipeline.jpeg({ quality: 78 })).toBuffer();

    response.body = output.toString('base64');
    response.bodyEncoding = 'base64';
    response.headers['content-type'] = [
      { key: 'Content-Type', value: `image/${format}` }
    ];
    response.headers['cache-control'] = [
      { key: 'Cache-Control', value: 'public, max-age=31536000, immutable' }
    ];
    return response;
  } catch (err) {
    // Fall through to the original object rather than failing the request.
    console.error(JSON.stringify({ msg: 'resize_failed', error: err.message }));
    return response;
  }
};
```

Points worth calling out:

- `withoutEnlargement: true` prevents upscaling a 200 px source to 1200 px, which wastes bandwidth and looks worse than the original.
- `limitInputPixels: false` disables sharp's decompression-bomb guard. Only do this if uploads are authenticated and size-capped upstream — otherwise you have handed attackers a memory-exhaustion primitive.
- The `catch` block returns the untransformed response. A resize failure should never be a user-visible error.
- `Cache-Control: immutable` is only honest if the object at that URL never changes. With versioned buckets, key on a content hash or a version ID in the path, or you will serve stale bytes forever.

### Why streaming matters

`sharp` is backed by libvips, which can process images in a streaming fashion when you give it a file path or a stream. The Lambda@Edge response object, however, arrives as a base64 string in `event.Records[0].cf.response.body`, so you are already holding the encoded object in memory before sharp sees it. Lambda@Edge caps the response body it can generate, and the function's memory ceiling is small compared with a standard Lambda.

The practical consequence: keep originals small, cap upload size, and treat edge resizing as a convenience tier. If you need to process 25 MB camera RAW files, do it in a standard Lambda triggered by S3 `ObjectCreated`, write the derivatives back to S3, and let the edge serve pre-computed variants. That architecture also removes cold-start cost from the request path entirely.

## Step 3 — infrastructure

The same stack expressed in Terraform, using an origin access control:

```hcl
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "us-east-1" # Lambda@Edge functions must live here
}

resource "aws_cloudfront_origin_access_control" "oac" {
  name                              = "photos-oac"
  origin_access_control_origin_type = "s3"
  signing_behavior                  = "always"
  signing_protocol                  = "sigv4"
}

resource "aws_s3_bucket" "photos" {
  bucket = "photos-example"
}

resource "aws_s3_bucket_versioning" "photos" {
  bucket = aws_s3_bucket.photos.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_cloudfront_distribution" "s3_distribution" {
  enabled         = true
  is_ipv6_enabled = true

  origin {
    domain_name              = aws_s3_bucket.photos.bucket_regional_domain_name
    origin_id                = "s3-photos"
    origin_access_control_id = aws_cloudfront_origin_access_control.oac.id
  }

  default_cache_behavior {
    allowed_methods        = ["GET", "HEAD"]
    cached_methods         = ["GET", "HEAD"]
    target_origin_id       = "s3-photos"
    viewer_protocol_policy = "redirect-to-https"
    min_ttl                = 0
    default_ttl            = 31536000
    max_ttl                = 31536000

    forwarded_values {
      query_string = true # the ?w= breakpoint must reach the origin request
      cookies { forward = "none" }
    }

    lambda_function_association {
      event_type   = "origin-response"
      lambda_arn   = aws_lambda_function.resize.qualified_arn
      include_body = true
    }
  }

  restrictions {
    geo_restriction { restriction_type = "none" }
  }

  viewer_certificate {
    cloudfront_default_certificate = true
  }
}

resource "aws_lambda_function" "resize" {
  provider      = aws
  function_name = "edge-image-resize"
  handler       = "index.handler"
  runtime       = "nodejs20.x"
  role          = aws_iam_role.lambda_exec.arn
  filename      = "lambda.zip"
  memory_size   = 512
  timeout       = 5
  publish       = true
}

resource "aws_iam_role" "lambda_exec" {
  name = "edge-image-resize-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = ["lambda.amazonaws.com", "edgelambda.amazonaws.com"] }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}
```

Two details that cause most of the support tickets here:

1. **`query_string = true` on the cache behavior.** If you forward the query string but do not include it in the cache key, every viewer gets whichever breakpoint was cached first. CloudFront's cache key is configured separately from forwarding; make sure `?w=600` and `?w=1200` are distinct cache entries.
2. **The `edgelambda.amazonaws.com` service principal.** A Lambda@Edge execution role that only trusts `lambda.amazonaws.com` will deploy and then fail at the edge with an opaque permission error.

## Failure modes and how to detect them

### Cold starts

Lambda@Edge functions run in a constrained environment and are not kept warm on demand. A cold start includes container initialisation plus loading the `sharp` native module and libvips, which is the expensive part. The symptom is a bimodal latency distribution: most requests fast, a minority several times slower, clustered after periods of low traffic.

Measure it rather than guess. Emit the container's own start time and compare it with the request time:

```javascript
const CONTAINER_START = Date.now();

// inside the handler
const initMs = Date.now() - CONTAINER_START;
console.log(JSON.stringify({ msg: 'invocation', initMs }));
```

If `initMs` is small, the container was warm. If it is large, you paid initialisation. Plot the distribution of `initMs` over time — that tells you the cold-start rate directly. Scheduled "warming" invocations are a common mitigation, but they only help if the scheduler hits the same edge locations your users do, which you cannot control. For latency-critical paths, pre-computed derivatives in S3 are a more reliable answer than warming.

### Format negotiation

Never trust the `Accept` header alone to predict whether a client can decode a format. Clients advertise capabilities they implement incorrectly, and proxies sometimes rewrite the header. The robust pattern is content negotiation with a correctness check you control:

1. Parse `Accept` for `image/webp` or `image/avif`.
2. If the client claims support, serve the modern format.
3. Keep the `Vary: Accept` header on the response so caches do not serve WebP to a client that asked for JPEG.

```javascript
response.headers['vary'] = [{ key: 'Vary', value: 'Accept' }];
```

Omitting `Vary` is the single most common cause of "the image is broken for some users" reports: a shared cache stores the WebP variant and hands it to a client that never advertised support. The `Vary` header is what makes the cache key include the `Accept` header.

### Origin timeouts

CloudFront's origin response timeout and the origin's own behaviour are separate knobs. If the origin is slow — a large object read, a throttled bucket — CloudFront returns a 504 regardless of what your Lambda does, because the Lambda never sees a response. Distinguish this from a Lambda error by checking which side logged. A 504 with no corresponding Lambda invocation in CloudWatch is an origin problem; a 502 with a Lambda log line is your code.

Keep the origin timeout below the Lambda timeout so that a slow origin surfaces as a 504 you can alert on, rather than a Lambda that gets killed mid-flight.

### Log volume

Edge functions log per request, and per-request logging at the edge is expensive in a way that surprises people. A single `console.log` per invocation, multiplied by request rate and replicated across edge locations, can dominate your CloudWatch bill. Log structured, sampled events rather than free text, and set a retention policy on the log group. If you need per-request detail, sample it: log 1 in 100 requests in full, and log only aggregate counters for the rest.

## Observability: what to instrument

Lambda@Edge cannot be scraped by Prometheus — there is no long-lived process to scrape, and the function runs at edge locations you do not control. The workable pattern is to emit metrics as structured log lines and let CloudWatch Logs metric filters turn them into metrics.

Instrument these four things:

- `resize_duration_ms` — the time inside sharp, labelled by breakpoint and output format.
- `init_ms` — container initialisation time, to quantify cold starts.
- `resize_error` — a counter of caught exceptions, labelled by error class.
- `bytes_out` — the size of the transformed body, to verify that WebP is actually smaller.

```javascript
const start = Date.now();
try {
  const output = await pipeline.toBuffer();
  console.log(JSON.stringify({
    msg: 'resize_ok',
    width,
    format,
    durationMs: Date.now() - start,
    bytesOut: output.length
  }));
  // ...
} catch (err) {
  console.log(JSON.stringify({
    msg: 'resize_error',
    width,
    format,
    errorClass: err.constructor.name
  }));
}
```

Then create a metric filter on `{ $.msg = "resize_ok" }` extracting `$.durationMs`, and alarm on its p99. This is the difference between knowing that resizes are slow and knowing *which breakpoint* is slow — a distinction that matters because a 1200 px WebP encode is roughly an order of magnitude more expensive than a 300 px JPEG.

## Decision checklist

Before deploying edge resizing, confirm each of these:

- **Are originals small enough to hold in memory at the edge?** If not, pre-compute derivatives in a standard Lambda.
- **Is the cache key correct?** Query string included, `Vary: Accept` set, immutable URLs actually immutable.
- **Does every failure path return the original?** A resize bug should never be a 5xx.
- **Is the execution role trusted by `edgelambda.amazonaws.com`?** Otherwise the deploy succeeds and the edge fails.
- **Are you logging sampled, structured events with a retention policy?** Otherwise logs become your largest line item.
- **Do you have a p99 alarm on resize duration?** Without one, a regression is invisible until users complain.
- **Have you capped upload size and authenticated uploads?** `limitInputPixels: false` is only safe behind those controls.

## A worked sizing example

Suppose your originals average 2 MB and you serve 1 million resized images per month, with a 95% cache hit rate at the edge.

- Origin fetches: 1,000,000 × 0.05 = 50,000.
- Lambda invocations: 50,000 (only cache misses invoke the origin-response function).
- At an illustrative 400 ms per invocation, total compute time is 50,000 × 0.4 = 20,000 seconds.
- Lambda@Edge is billed on request count and GB-seconds; at 512 MB, that is 20,000 × 0.5 = 10,000 GB-seconds.

Substitute your own measured `resize_duration_ms` p50 and your real hit rate. The point of the exercise is that cache hit rate, not resize speed, dominates the bill — going from 95% to 99% hit rate cuts compute by 80%, which is almost always cheaper to achieve than optimising the encoder.

## Next.js integration

If the frontend is a Next.js app, point `next/image` at the distribution with a custom loader so that Next.js does not double-optimise:

```javascript
// next.config.js
module.exports = {
  images: {
    remotePatterns: [
      { protocol: 'https', hostname: 'cdn.example.com' }
    ],
    deviceSizes: [300, 600, 1200]
  }
};
```

```javascript
// components/imageLoader.js
export default function imageLoader({ src, width, quality }) {
  return `https://cdn.example.com/${src}?w=${width}&q=${quality || 75}`;
}
```

The loader must only emit widths that the edge function recognises. If `next/image` requests a width outside your `BREAKPOINTS` list, the function falls back to `MAX_WIDTH` and you silently serve a larger image than requested — which is correct but wasteful. Keep the two lists in sync, ideally by generating both from one constant.

## What to do in the next 30 minutes

Open your CloudFront distribution's cache statistics and compute the hit rate for your image paths over the last 24 hours. If it is below 90%, the highest-leverage fix is almost always the cache key — a missing `Vary: Accept`, a query string that is forwarded but not cached, or a short TTL on assets that never change. Fix that before you touch the resize function.
