"""
taxonomy.py
===========
Deterministic, controlled-vocabulary site tags.

WHY THIS EXISTS
---------------
blog_system.py used to build `post.tags` from `_derive_hashtags_from_keywords()`,
a TWEET-hashtag helper that matched keys by raw substring. Result on the live
site: "rag" matched "sto-rag-e" (logging post tagged "Rag Vectorsearch"),
"coding" matched "en-coding", "ai" matched "m-ai-ntain", and CamelCase junk such
as "UvVsPoetry" / "SeniorTitle" / "Active" became public /tag/ archive pages.
Tag archives are crawlable listing pages; incoherent ones are exactly the
low-value, auto-generated archive pages that AdSense / helpful-content
reviewers sample.

Contract
--------
  derive_site_tags(title, keywords, content) -> list[str]   (1..4 display tags)

* Only tags from CANONICAL below can ever be emitted -> bounded, human-readable
  taxonomy that cannot grow into thousands of singleton archives.
* Matching is word-boundary based (no substring leaks).
* A tag needs evidence: in the title, in the LLM keywords, or >= 3 mentions in
  prose (code fences excluded).
* Twitter hashtags are untouched and stay a separate concern.
"""

import re
from typing import Dict, Iterable, List, Tuple

# display tag -> regex alternatives (matched case-insensitively with \b guards)
CANONICAL: Dict[str, Tuple[str, ...]] = {
    # --- AI / agents ---
    "AI Agents": (r"agents?", r"agentic", r"multi-agent", r"human-in-the-loop"),
    "LLMs": (
        r"llms?",
        r"large language models?",
        r"fine-tun\w*",
        r"prompt\w*",
        r"guardrails?",
    ),
    "RAG": (r"rag", r"retrieval[- ]augmented", r"reranking", r"hybrid search"),
    "Vector Search": (
        r"vector (?:databases?|search|index\w*)",
        r"pgvector",
        r"qdrant",
        r"weaviate",
        r"milvus",
        r"embeddings?",
    ),
    "MCP": (r"mcp", r"model context protocol", r"a2a"),
    "MLOps": (r"mlops", r"llmops", r"model evals?", r"evaluation harness\w*"),
    # --- platform / infra ---
    "Kubernetes": (r"kubernetes", r"k8s", r"karpenter", r"keda", r"argo ?cd", r"helm"),
    "Containers": (r"docker", r"containers?", r"distroless", r"container images?"),
    "AWS": (r"aws", r"amazon web services", r"ec2", r"s3", r"lambda"),
    "Azure": (r"azure",),
    "Google Cloud": (r"google cloud", r"gcp", r"bigquery"),
    "Cloud Cost": (
        r"finops",
        r"egress costs?",
        r"cloud costs?",
        r"unit economics",
        r"token economics",
    ),
    "Terraform": (r"terraform", r"opentofu", r"infrastructure as code", r"iac"),
    "Platform Engineering": (
        r"platform engineering",
        r"internal developer platforms?",
        r"golden paths?",
        r"platform teams?",
    ),
    "CI/CD": (
        r"ci/cd",
        r"gitops",
        r"canary",
        r"blue-green",
        r"progressive delivery",
        r"rollbacks?",
    ),
    "Serverless": (r"serverless", r"cold starts?", r"faas"),
    # --- reliability / observability ---
    "Observability": (
        r"observability",
        r"opentelemetry",
        r"tracing",
        r"structured logging",
        r"slos?",
        r"monitoring",
    ),
    "Reliability": (
        r"reliability",
        r"sre",
        r"incidents?",
        r"postmortems?",
        r"on-call",
        r"chaos engineering",
        r"circuit breakers?",
        r"graceful degrad\w*",
    ),
    "Performance": (
        r"performance",
        r"latency",
        r"throughput",
        r"load testing",
        r"capacity planning",
        r"core web vitals",
        r"profiling",
    ),
    # --- data ---
    "PostgreSQL": (r"postgres(?:ql)?", r"pgbouncer"),
    "Databases": (
        r"databases?",
        r"sql",
        r"schema migrations?",
        r"replicas?",
        r"sharding",
        r"isolation levels?",
        r"denormaliz\w*",
        r"connection pool\w*",
    ),
    "Redis": (r"redis", r"valkey", r"dragonfly"),
    "Messaging": (
        r"kafka",
        r"redpanda",
        r"nats",
        r"message queues?",
        r"event streams?",
        r"outbox",
        r"webhooks?",
        r"backpressure",
    ),
    "Data Engineering": (
        r"clickhouse",
        r"duckdb",
        r"data warehouse",
        r"change data capture",
        r"cdc",
    ),
    # --- architecture ---
    "Distributed Systems": (
        r"distributed",
        r"multi-region",
        r"active-active",
        r"locks?",
        r"consensus",
        r"leader election",
        r"fencing",
        r"idempotency",
        r"sagas?",
        r"cap theorem",
        r"event sourcing",
        r"cqrs",
    ),
    "API Design": (
        r"apis?",
        r"rest(?:ful)?\s+apis?",
        r"restful",
        r"graphql",
        r"grpc",
        r"pagination",
        r"rate limit\w*",
        r"api versioning",
    ),
    "Software Architecture": (
        r"architecture",
        r"microservices?",
        r"monolith\w*",
        r"monorepo",
        r"architecture decision records?",
        r"adrs?",
    ),
    "Testing": (
        r"testing",
        r"flaky tests?",
        r"contract tests?",
        r"property-based",
        r"staging environments?",
    ),
    # --- security ---
    "Security": (
        r"security",
        r"threat model\w*",
        r"stride",
        r"vulnerabilit\w*",
        r"sbom",
        r"supply chain",
        r"secrets?",
        r"encryption",
        r"csrf",
        r"sql injection",
        r"waf",
    ),
    "Identity & Access": (
        r"oauth",
        r"jwt",
        r"rbac",
        r"abac",
        r"passkeys?",
        r"webauthn",
        r"spiffe",
        r"mtls",
        r"mutual tls",
        r"least[- ]privilege",
        r"zero[- ]trust",
        r"authenticat\w*",
        r"authoriz\w*",
    ),
    # --- fintech / Africa ---
    "Payments": (
        r"payments?",
        r"ledgers?",
        r"double-entry",
        r"settlement",
        r"reconciliation",
        r"iso 20022",
        r"pci dss",
        r"tokenization",
        r"wallets?",
    ),
    "Mobile Money": (
        r"m-pesa",
        r"mpesa",
        r"mobile money",
        r"ussd",
        r"paystack",
        r"flutterwave",
        r"papss",
    ),
    "Fintech": (r"fintech", r"kyc", r"open banking", r"fraud detection"),
    "Africa Tech": (
        r"africa\w*",
        r"nairobi",
        r"kenya\w*",
        r"nigeria\w*",
        r"east africa",
    ),
    # --- languages ---
    "Python": (
        r"python",
        r"fastapi",
        r"django",
        r"flask",
        r"asyncpg",
        r"sqlalchemy",
        r"uv",
        r"poetry",
        r"pip-tools",
    ),
    "Go": (r"golang", r"go concurrency", r"goroutines?"),
    "Rust": (r"rust",),
    "Java": (r"java", r"jvm", r"virtual threads"),
    "TypeScript": (r"typescript", r"node\.?js", r"bun", r"deno"),
    "WebAssembly": (r"webassembly", r"wasm"),
    # --- career ---
    "Engineering Careers": (
        r"careers?",
        r"promotion",
        r"senior(?:ity)?",
        r"senior engineers?",
        r"staff engineers?",
        r"portfolio",
        r"hiring",
        r"interview\w*",
        r"mentor\w*",
        r"remote (?:work|roles?|engineering)",
        r"freelanc\w*",
        r"compensation",
    ),
    "Engineering Management": (
        r"engineering managers?",
        r"technical debt",
        r"design docs?",
        r"code review",
        r"engineering documentation",
        r"meeting load",
    ),
}

# tags that only count when they appear in the title/keywords (too generic for
# body-mention evidence): prevents "Performance"/"Security" riding on 3 passing words
TITLE_OR_KEYWORD_ONLY = {
    "Performance",
    "Security",
    "Databases",
    "API Design",
    "Africa Tech",
    "Software Architecture",
}

_COMPILED: Dict[str, re.Pattern] = {
    tag: re.compile(r"(?<![\w])(?:" + "|".join(alts) + r")(?![\w])", re.I)
    for tag, alts in CANONICAL.items()
}
_FENCE = re.compile(r"```.*?```", re.S)
_FALLBACK = "Backend Engineering"
MIN_BODY_MENTIONS = 3


def _count(pattern: re.Pattern, text: str) -> int:
    return len(pattern.findall(text)) if text else 0


def derive_site_tags(
    title: str,
    keywords: Iterable[str] = (),
    content: str = "",
    max_tags: int = 4,
) -> List[str]:
    """Return 1..max_tags canonical display tags, strongest evidence first."""
    kw_text = " ; ".join(k for k in (keywords or []) if isinstance(k, str))
    body = _FENCE.sub(" ", content or "")
    scored: List[Tuple[int, str]] = []
    for tag, pat in _COMPILED.items():
        in_title = _count(pat, title or "")
        in_kw = _count(pat, kw_text)
        mentions = _count(pat, body)
        if tag in TITLE_OR_KEYWORD_ONLY:
            if not (in_title or in_kw):
                continue
            score = 5 * in_title + 3 * in_kw + min(mentions, 5)
        else:
            if not (in_title or in_kw or mentions >= MIN_BODY_MENTIONS):
                continue
            score = 5 * in_title + 3 * in_kw + min(mentions, 10)
        scored.append((score, tag))
    scored.sort(key=lambda x: (-x[0], x[1]))
    tags = [t for _, t in scored[:max_tags]]
    return tags or [_FALLBACK]
