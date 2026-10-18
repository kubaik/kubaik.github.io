# Deploy a solo stack for under $20/month

Most deployment tutorials stop at the happy path. What follows is the less glamorous part: a pipeline that keeps running when you are the only person on call, built from boring, well-documented services with published prices.

## The problem with day-one scale thinking

A common failure mode for solo developers is provisioning for an audience that does not exist yet. The stack gets chosen for the load it might see, not the load it has. The result is usually the same: an idle database instance, a container orchestrator with one container, and a monitoring bill that grows with alert volume rather than with traffic.

The alternative is to pick the smallest set of managed services that covers deployment, observability, and backups, then measure whether they are actually too small. That measurement is cheap. Over-provisioning is not.

This article walks through a pipeline with four moving parts:

- A three-stage CI workflow: test, build, deploy
- A serverless backend on AWS Lambda with a container image, fronted by API Gateway
- A managed PostgreSQL instance on Amazon RDS with automated backups
- CloudWatch metrics and a single alert path to a chat webhook

Cost targets below are arithmetic from published on-demand prices and stated assumptions, not measurements of a real bill. Treat them as a sizing exercise you should redo with your own traffic.

## Prerequisites

Three things are required before starting:

1. A GitHub account and a repository containing a working application. Language is irrelevant to the pipeline; the examples use Python.
2. An AWS account with billing alerts configured. Create a budget before deploying anything, because a forgotten NAT gateway or an open RDS instance is the classic way to turn a $15 project into a several-hundred-dollar one.
3. A domain you can point at AWS. Route 53 hosted zones cost $0.50/month per zone; the registrar fee is separate.

Everything else is created along the way.

## Step 1 — repository and database

Start with a repository layout that keeps application code, tests, and workflow definitions separate:

- `src/` for the application
- `tests/` for pytest
- `.github/workflows/` for CI definitions
- `Dockerfile` for the container image
- `requirements.txt` with pinned versions

Pin your dependencies. Unpinned installs mean that a rebuild of the same commit can produce a different image, which turns a trivial redeploy into a debugging session. Pinning makes builds reproducible and makes the diff between two deployments meaningful.

```python
# requirements.txt
fastapi==0.110.2
uvicorn==0.29.0
pydantic==2.7.1
sqlalchemy==2.0.29
psycopg2-binary==2.9.9
pytest==8.1.1
httpx==0.27.0
```

Next, create the database. In the RDS console:

- Engine: PostgreSQL
- Template: Free tier
- Instance class: `db.t4g.micro`
- Storage: 20 GB gp3
- Storage autoscaling: enabled, maximum 100 GB
- Public access: No
- Security group: inbound only from the Lambda function's security group
- Automated backups: enabled, 7-day retention
- Encryption: AWS managed key
- Log exports: none

Connect once from a machine inside the VPC, or through a bastion, and create the application role:

```sql
CREATE DATABASE myapp;
CREATE USER appuser WITH PASSWORD 'replace-with-a-generated-secret';
GRANT ALL PRIVILEGES ON DATABASE myapp TO appuser;
```

Store the password in GitHub Secrets as `DB_PASSWORD` and the endpoint as `DB_HOST`. Do not put either in the repository.

The application connects with a pool that survives serverless execution:

```python
# src/main.py
from fastapi import FastAPI
from sqlalchemy import create_engine, text
import os

DB_HOST = os.getenv("DB_HOST")
DB_PASSWORD = os.getenv("DB_PASSWORD")

DATABASE_URL = f"postgresql+psycopg2://appuser:{DB_PASSWORD}@{DB_HOST}/myapp"
engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=300)

app = FastAPI()

@app.get("/")
def read_root():
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        return {"status": "ok", "db_ok": result.fetchone()[0] == 1}
```

Two settings matter here. `pool_pre_ping=True` issues a lightweight check before handing out a connection, which prevents the `psycopg2.OperationalError: connection already closed` error that appears when a pooled connection has been dropped by the server. `pool_recycle=300` retires connections before PostgreSQL's own idle timeout can close them underneath you.

## Step 2 — the pipeline

The workflow runs tests against a throwaway PostgreSQL container, builds a container image, pushes it to Amazon ECR, and updates the Lambda function.

```yaml
# .github/workflows/deploy.yml
name: Deploy to Lambda

on:
  push:
    branches: [main]

env:
  AWS_REGION: us-east-1
  ECR_REPO: myapp-lambda
  LAMBDA_FUNCTION: myapp-api

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16.2
        ports:
          - 5432:5432
        env:
          POSTGRES_PASSWORD: testpass
          POSTGRES_USER: appuser
          POSTGRES_DB: myapp
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -r requirements.txt
      - run: pytest tests/

  build-and-deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ env.AWS_REGION }}
      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v2
      - name: Build, tag, and push image to Amazon ECR
        id: build-image
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          ECR_REPO: ${{ env.ECR_REPO }}
          IMAGE_TAG: ${{ github.sha }}
        run: |
          docker build -t $ECR_REGISTRY/$ECR_REPO:$IMAGE_TAG .
          docker push $ECR_REGISTRY/$ECR_REPO:$IMAGE_TAG
          echo "image=$ECR_REGISTRY/$ECR_REPO:$IMAGE_TAG" >> $GITHUB_OUTPUT
      - name: Deploy to Lambda
        id: deploy-lambda
        uses: aws-actions/aws-lambda-deploy@v4
        with:
          function-name: ${{ env.LAMBDA_FUNCTION }}
          image-uri: ${{ steps.build-image.outputs.image }}
          package-type: Image
          timeout: 30
          memory-size: 2048
          environment-variables: DB_HOST=${{ secrets.DB_HOST }},DB_PASSWORD=${{ secrets.DB_PASSWORD }}
          publish: true
```

Container images are worth the extra build step for one reason: native dependencies. Lambda layers and ZIP bundles routinely break when a Python package expects a shared library such as `libpq` that is not present in the runtime. A container image pins the whole userland, so the build and the runtime agree.

Create the Lambda function once, before the first pipeline run:

- Runtime: `Amazon Linux 2023`
- Architecture: `arm64`
- Timeout: 30 seconds
- Memory: 2048 MB
- Ephemeral storage: 512 MB
- Environment variables: `DB_HOST`, `DB_PASSWORD`
- VPC: the same VPC as RDS, with at least two private subnets
- Security group: outbound to the internet, inbound from API Gateway

Then attach an HTTP API in API Gateway with a Lambda proxy integration, route `ANY /{proxy+}`, payload format 2.0, deployed to a stage named `prod`. HTTP APIs are materially cheaper than REST APIs and support what this stack needs.

Finally, create a Route 53 alias record pointing your subdomain at the API Gateway endpoint.

### On the IAM user

The deploy user needs write access to Lambda and ECR, and read access to CloudWatch Logs. Managed policies such as `AWSLambda_FullAccess` are broader than necessary; a scoped inline policy is better practice, but if you start with managed policies, restrict the user to a single account and rotate the access key on a schedule. Long-lived static keys in GitHub Secrets are the weakest link in this design; OIDC-based role assumption removes them entirely and is worth migrating to.

## Step 3 — connection handling and failure modes

The first deployment of a VPC-attached Lambda that talks to RDS usually produces a 502 from API Gateway, with `Task timed out after N seconds` in CloudWatch Logs. The cause is almost always one of four things.

**Lambda is outside the VPC.** If the function is not attached to the VPC containing RDS, it cannot route to a private endpoint. The connection hangs until the timeout. Fix: attach the function to the same VPC and subnets.

**The security group does not permit the traffic.** The RDS security group must allow inbound on 5432 from the Lambda's security group, not from a CIDR range. Referencing the security group by ID keeps the rule correct when addresses change.

**The password or host is wrong.** This surfaces as an authentication failure rather than a timeout, but it is worth checking first because it is the cheapest to rule out.

**The connection is dropped between invocations.** Lambda freezes execution environments between requests. A pooled connection can be closed by the database while the environment is frozen, and the next invocation finds a dead socket. This is what `pool_pre_ping` and `pool_recycle` address.

A connection pool sized for serverless:

```python
# src/db.py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DB_HOST = os.getenv("DB_HOST")
DB_PASSWORD = os.getenv("DB_PASSWORD")

engine = create_engine(
    f"postgresql+psycopg2://appuser:{DB_PASSWORD}@{DB_HOST}/myapp",
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=300,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
```

`pool_size=5` with `max_overflow=10` allows up to 15 concurrent connections per execution environment. Multiply that by the number of concurrent Lambda environments to get the worst-case connection count your database must accept. A `db.t4g.micro` has a `max_connections` ceiling that is easy to exceed if you set the pool too high and traffic spikes. If connection count becomes the constraint, an RDS Proxy in front of the instance multiplexes client connections and reduces churn; it is billed per vCPU-hour of the underlying instance, so check current pricing before assuming it is cheap.

### Measuring instead of guessing

Latency and throughput claims are only meaningful if you can reproduce them. A simple load test against a staging endpoint:

```bash
vegeta attack -duration=30s -rate=100 -targets=targets.txt | vegeta report
```

Run it against a staging deployment, not production, and record four numbers: requests per second sustained, p50 and p99 latency, and error rate. Repeat at two or three rates to find where p99 starts climbing. That curve, not a single number, tells you when to add memory or move off Lambda.

The same discipline applies to cost. Instrument it by reading the AWS Cost Explorer grouped by service after a full billing cycle, and compare against the arithmetic below. Do not trust a projected figure over an actual invoice.

### Cost arithmetic

The following is illustrative, based on published on-demand prices and stated assumptions. Substitute your own region and usage.

Assume 100,000 requests per month, each consuming 200 ms of billed duration at 2048 MB:

- Lambda bills GB-seconds. 2048 MB = 2 GB. 2 GB × 0.2 s = 0.4 GB-seconds per request.
- 100,000 requests × 0.4 = 40,000 GB-seconds per month.
- At the x86 rate of $0.0000166667 per GB-second, that is 40,000 × 0.0000166667 ≈ $0.67.
- ARM64 is priced about 20% lower, so ≈ $0.53.

Add the request charge (roughly $0.20 per million requests, so about $0.02 here) and API Gateway HTTP API charges (roughly $1.00 per million requests, so about $0.10). The dominant cost is the database: a `db.t4g.micro` with 20 GB gp3 storage runs in the region of $12–13 per month on demand, less if reserved. Log ingestion and storage add a small amount that scales with how much you log.

The point of showing the arithmetic is that the database is roughly 80% of the bill. Optimizing Lambda memory saves cents; choosing the right database size saves dollars. If you need to cut further, the levers in order of impact are: reserved instances or Savings Plans for RDS, shorter log retention, and moving to a smaller instance class once you have measured actual CPU.

## Step 4 — observability and tests

CloudWatch is sufficient for a single-service deployment provided you configure it deliberately rather than accepting defaults.

Enable Lambda Insights for per-invocation memory and duration data, then set alarms at a fraction of your configured limits rather than at the limit itself. An alarm that fires only when the function has already timed out is a post-mortem, not a warning. A reasonable starting point is a duration alarm at roughly 80% of the configured timeout and a memory alarm at roughly 80% of the configured memory, adjusted once you have real percentiles.

Structured logging makes those logs searchable:

```python
# src/main.py
import logging
from fastapi import Request
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info("request_start", extra={
        "method": request.method,
        "path": request.url.path,
        "query": dict(request.query_params),
    })
    response = await call_next(request)
    logger.info("request_end", extra={
        "status_code": response.status_code,
        "method": request.method,
        "path": request.url.path,
    })
    return response

@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    logger.error("unhandled_exception", extra={
        "error": str(exc),
        "path": request.url.path,
    })
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error"},
    )
```

A health endpoint that reports database reachability gives your alerting something unambiguous to poll:

```python
@app.get("/health")
def health():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "healthy", "db": "ok"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}, 500
```

Wire a CloudWatch alarm on the Lambda `Errors` metric to an SNS topic, and have SNS deliver to a chat webhook. One alarm on errors and one on duration covers most solo-project incidents. Resist adding more until you have been woken up by something the existing alarms missed.

Tests run against a real PostgreSQL instance in CI, which catches dialect and migration problems that SQLite would hide:

```python
# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

@pytest.fixture(scope="module")
def setup_db():
    with engine.connect() as conn:
        conn.execute(text("CREATE TABLE IF NOT EXISTS test_table (id SERIAL PRIMARY KEY)"))
        conn.commit()

def test_health(setup_db):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_db_connection(setup_db):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["db_ok"] == 1
```

Note the fixture is declared as a parameter of each test that needs it. Ordering plugins and markers like `@pytest.mark.order` are a code smell: if tests only pass in a specific sequence, they are sharing state and will fail intermittently under parallel execution.

Finally, confirm that automated backups are enabled with a retention window you can live with, and take a manual snapshot before any schema migration. Snapshots within your retained window cost nothing extra; the discipline of taking one before a destructive change is what saves you.

## Decision checklist before you commit

- Is the database the dominant cost line? If yes, that is where to optimize first.
- Have you measured p99 latency at your expected peak, not just at idle?
- Does the Lambda security group reference the RDS security group by ID?
- Are connection limits on the instance class above your worst-case pool size times concurrency?
- Is there exactly one alert path, and has it been tested by forcing a failure?
- Are credentials in a secret store, and is there a rotation plan?
- Can you restore from backup, and have you tried it?

## FAQ

**Can a static frontend use this API?**
Yes. Host the frontend on any static host and point it at the API Gateway endpoint. Remember to configure CORS on the API; the browser will enforce it regardless of what the backend allows.

**What about WebSockets?**
API Gateway HTTP APIs do not support WebSockets; REST APIs and the WebSocket API type do. For most solo projects, Server-Sent Events or polling over HTTP is simpler and cheaper, and the connection lifecycle is easier to reason about.

**How should secrets be rotated?**
Use Secrets Manager with a rotation schedule, or eliminate static credentials entirely by using IAM authentication for RDS and OIDC for GitHub Actions. The second option is more work up front and removes a whole class of credential-leak incidents.

**What if traffic outgrows Lambda?**
The signal is p99 latency rising as concurrency increases, or persistent throttling. At that point a small EC2 instance or a container service becomes cheaper per request than Lambda's per-invocation pricing. Do not migrate preemptively; migrate when the measurement says so.

**Is GitHub Actions the only option?**
No. Any CI system that can build a container and call the AWS CLI will work. The workflow structure — test, build, push, deploy — is portable; only the YAML differs.

## Do this in the next 30 minutes

Open the AWS Billing console, create a budget with an alert threshold you would be comfortable losing, and confirm the alert email address is one you actually read. Then check whether your current database instance has automated backups enabled and a retention window longer than one day. Those two settings prevent the majority of expensive solo-project incidents, and neither takes more than a few minutes to verify.
