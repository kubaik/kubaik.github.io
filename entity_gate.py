#!/usr/bin/env python3
"""
entity_gate.py - flags drafts/posts that present versioned products/benchmarks that cannot be verified.

Why: the generator is closed-book and is *required* to emit tool+version numbers and metrics, so it invents
products ("Kagi Router v0.9"), versions, and benchmark tables. This is the failure that gets AI blogs
rejected for low-value / misleading content, and the existing anecdote gate does not see it.

Offline (default): allowlist check only - deterministic, no network, used by triage.
Online (--online / verify_online=True): PyPI + npm registry lookup with on-disk cache, used in CI before publish.
"""

import json, re, urllib.request, urllib.error
from pathlib import Path

KNOWN = {n.lower() for n in """
Python Node Node.js Go Golang Rust Java Kotlin Swift Ruby PHP Zig TypeScript JavaScript Deno Bun Postgres PostgreSQL MySQL MariaDB SQLite Redis Valkey
MongoDB Cassandra DynamoDB ClickHouse DuckDB Kafka RabbitMQ NATS Pulsar Kubernetes K8s Docker Terraform OpenTofu Pulumi Helm Argo ArgoCD Flux Istio Envoy
Linkerd Prometheus Grafana Loki Tempo Jaeger OpenTelemetry Datadog Sentry Nginx HAProxy Traefik Caddy Linux Ubuntu Debian Alpine React Vue Angular Svelte
Next.js Nuxt Remix Astro HTMX FastAPI Django Flask Express NestJS Spring Rails Laravel pgvector Pydantic SQLAlchemy Celery Temporal Inngest Cursor Windsurf
Copilot Claude GPT Gemini Llama Mistral Qwen DeepSeek Ollama LangChain LangGraph LlamaIndex Pytest Jest Vitest Playwright Cypress Locust k6 GitHub GitLab
Jenkins Tekton Backstage Vault Consul Nomad Cloudflare Vercel Netlify Fly.io Supabase Firebase Stripe Paystack Flutterwave M-Pesa Lambda AWS GCP Azure
OpenAI Anthropic Groq Cerebras Bedrock Vertex PagerDuty Opsgenie FireHydrant Splunk Elastic Elasticsearch OpenSearch Kibana Trivy Snyk Semgrep Dependabot
Renovate Wasm WebAssembly Wasmtime eBPF Cilium Falco Kyverno OPA Gatekeeper gRPC GraphQL REST HTTP OAuth TLS QUIC WebAuthn Android iOS Chrome Firefox Safari
PyTorch TensorFlow SageMaker LangSmith HuggingFace Hugging Gradio Streamlit Pandas NumPy Polars Spark Airflow dbt Snowflake BigQuery Databricks Redshift Athena Fargate ECS EKS GKE AKS S3 EC2 CloudFront CloudWatch Cloud Run Kinesis SQS SNS RDS Aurora Neon PlanetScale CockroachDB Mongo Vercel Heroku Render Railway Sentry Honeycomb Lightstep Zipkin Fluentd Logstash Beats Vagrant Podman Containerd CRI-O Kustomize Skaffold Tilt Telepresence Crossplane Keycloak Auth0 Okta Clerk Cognito Twilio SendGrid Resend Postmark Algolia Meilisearch Typesense Qdrant Pinecone Weaviate Milvus Chroma FAISS Triton vLLM TGI ONNX TensorRT Ray Dask Prefect Dagster Luigi Metabase Superset Grafana Tailwind Vite Webpack esbuild Turbopack Rollup pnpm npm yarn pip uv Poetry Cargo Maven Gradle Git Bash Zsh Vim VS Code Terraform Ansible Chef Puppet Nix
""".split()}
# "Name v1.2" / "Name 1.2.3" / "Two Words v4.8"
VER = re.compile(
    r"\b([A-Z][A-Za-z0-9.+#-]{2,}(?:\s+[A-Z][A-Za-z0-9.+#-]{1,}){0,2})\s+v?(\d{1,3}\.\d{1,3}(?:\.\d{1,3})?)\b"
)
CODE = re.compile(r"```.*?```", re.S)
_CACHE_PATH = Path(".entity_cache.json")


def versioned_products(text: str):
    text = CODE.sub(" ", text)
    out = {}
    for m in VER.finditer(text):
        name, ver = m.group(1).strip(), m.group(2)
        words = name.lower().replace("  ", " ").split()
        if any(w in KNOWN or w.rstrip("s") in KNOWN for w in words):
            continue
        if name.split()[0] in {
            "The",
            "This",
            "That",
            "Step",
            "Phase",
            "Table",
            "Figure",
            "Version",
            "Section",
            "Part",
            "Chapter",
            "Rank",
        }:
            continue
        out.setdefault(name, set()).add(ver)
    return out


def _http_json(url):
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, headers={"User-Agent": "entity-gate/1.0"}),
            timeout=8,
        ) as r:
            return json.loads(r.read().decode())
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None


def _exists_online(name, versions, cache):
    key = f"{name}|{','.join(sorted(versions))}"
    if key in cache:
        return cache[key]
    slugs = {
        name.lower().replace(" ", "-"),
        name.lower().replace(" ", ""),
        name.lower().replace(" ", "_"),
    }
    ok = False
    for s in slugs:
        j = _http_json(f"https://pypi.org/pypi/{s}/json")
        if j and any(
            v.split(".")[:2] == ver.split(".")[:2]
            for v in j.get("releases", {})
            for ver in versions
        ):
            ok = True
            break
        j = _http_json(f"https://registry.npmjs.org/{s}")
        if j and any(
            v.split(".")[:2] == ver.split(".")[:2]
            for v in j.get("versions", {})
            for ver in versions
        ):
            ok = True
            break
    cache[key] = ok
    return ok


def unverified_products(text: str, verify_online: bool = False):
    prods = versioned_products(text)
    if not verify_online:
        return sorted(prods)
    cache = json.loads(_CACHE_PATH.read_text()) if _CACHE_PATH.exists() else {}
    bad = [n for n, v in prods.items() if not _exists_online(n, v, cache)]
    _CACHE_PATH.write_text(json.dumps(cache))
    return sorted(bad)


def gate(text: str, max_unverified: int = 3, verify_online: bool = False):
    """Return None if OK, else a rejection reason string (same contract as blog_system._reject_if_*)."""
    bad = unverified_products(text, verify_online)
    if len(bad) > max_unverified:
        return f"{len(bad)} unverifiable versioned products presented as fact: {', '.join(bad[:6])}"
    return None
