#!/usr/bin/env python3
"""
post_enhancer.py - code-driven IMPROVE pass for docs/*/post.json (idempotent, backs up, no LLM).

Reads triage_report/verdicts.csv (from triage.py) and, per post, applies only the fixes tagged there:
  strip_experience_claims  remove sentences asserting invented first-hand metrics/incidents
  add_inline_links         insert 'Related reading' with the 3 most similar KEPT posts (TF-IDF)
  add_sources              append 'Official documentation' links for tools the post actually names
  flag_unverified_products write post slug to enhance_report.json -> regeneration (never auto-edited)
  retitle                  report only (title shape needs regeneration)

Usage:
  python post_enhancer.py --docs docs --verdicts triage_report/verdicts.csv --backup .quality_review_backups            # dry run
  python post_enhancer.py ... --confirm [--verify-links]
"""

import argparse, csv, json, re, shutil, sys, urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import triage

MARK = "<!-- enhanced:v1 -->"
FOOTER_SENTINEL = "### About this article"
FOOTER = """---

### About this article

**Written by:** [Kubai Kevin](/about/), software developer based in Nairobi, Kenya.

**How this article was produced:** This article was drafted by an automated LLM pipeline and passed automated checks (length, uniqueness, citation and claim filters). It has not been individually reviewed or edited line by line by a human before publication. Figures, benchmarks and scenarios are illustrative unless a source is linked; verify against official documentation before relying on them in production. See the [AI content policy](/ai-content-policy/).

**Corrections:** Report errors via the [contact page](/contact/).

**Last generated:** {date}
"""
DOCS = {  # tool mention (regex) -> (label, official URL). Official vendor/standards docs only.
    r"\bPostgre(?:SQL|s)\b": (
        "PostgreSQL documentation",
        "https://www.postgresql.org/docs/",
    ),
    r"\bRedis\b": ("Redis documentation", "https://redis.io/docs/"),
    r"\bKubernetes\b|\bK8s\b": (
        "Kubernetes documentation",
        "https://kubernetes.io/docs/home/",
    ),
    r"\bTerraform\b": (
        "Terraform documentation",
        "https://developer.hashicorp.com/terraform/docs",
    ),
    r"\bDocker\b": ("Docker documentation", "https://docs.docker.com/"),
    r"\bOpenTelemetry\b|\bOTel\b": (
        "OpenTelemetry documentation",
        "https://opentelemetry.io/docs/",
    ),
    r"\bPrometheus\b": ("Prometheus documentation", "https://prometheus.io/docs/"),
    r"\bKafka\b": (
        "Apache Kafka documentation",
        "https://kafka.apache.org/documentation/",
    ),
    r"\bLambda\b": (
        "AWS Lambda Developer Guide",
        "https://docs.aws.amazon.com/lambda/",
    ),
    r"\bPydantic\b": ("Pydantic documentation", "https://docs.pydantic.dev/"),
    r"\bFastAPI\b": ("FastAPI documentation", "https://fastapi.tiangolo.com/"),
    r"\bpgvector\b": ("pgvector README", "https://github.com/pgvector/pgvector"),
    r"\bTemporal\b": ("Temporal documentation", "https://docs.temporal.io/"),
    r"\bOWASP\b.{0,40}\bLLM\b|\bLLM\b.{0,40}\bOWASP\b": (
        "OWASP Top 10 for LLM Applications",
        "https://genai.owasp.org/llm-top-10/",
    ),
    r"\bMCP\b|Model Context Protocol": (
        "Model Context Protocol docs",
        "https://modelcontextprotocol.io/",
    ),
    r"\bClickHouse\b": ("ClickHouse documentation", "https://clickhouse.com/docs"),
    r"\bDuckDB\b": ("DuckDB documentation", "https://duckdb.org/docs/"),
    r"\bgRPC\b": ("gRPC documentation", "https://grpc.io/docs/"),
    r"Core Web Vitals|\bINP\b|\bLCP\b": (
        "web.dev: Web Vitals",
        "https://web.dev/articles/vitals",
    ),
    r"\bArgo ?CD\b": ("Argo CD documentation", "https://argo-cd.readthedocs.io/"),
    r"\bFlux\b": ("Flux documentation", "https://fluxcd.io/flux/"),
    r"\bReact\b": ("React documentation", "https://react.dev/"),
    r"\bNext\.js\b": ("Next.js documentation", "https://nextjs.org/docs"),
    r"\bHTMX\b": ("htmx documentation", "https://htmx.org/docs/"),
    r"\bBackstage\b": ("Backstage documentation", "https://backstage.io/docs/"),
    r"\bOAuth\b": ("OAuth 2.0 resources", "https://oauth.net/2/"),
    r"\bWebAuthn\b|\bPasskeys?\b": (
        "W3C Web Authentication",
        "https://www.w3.org/TR/webauthn-3/",
    ),
    r"\bM-?Pesa\b|\bDaraja\b": (
        "Safaricom Daraja API",
        "https://developer.safaricom.co.ke/",
    ),
    r"\bPaystack\b": ("Paystack documentation", "https://paystack.com/docs/"),
    r"\bFlutterwave\b": (
        "Flutterwave documentation",
        "https://developer.flutterwave.com/docs",
    ),
    r"\bIdempotency\b|\bidempotent\b": (
        "Stripe: idempotent requests",
        "https://docs.stripe.com/api/idempotent_requests",
    ),
    r"\bLocust\b": ("Locust documentation", "https://docs.locust.io/"),
    r"\bLangGraph\b": (
        "LangGraph documentation",
        "https://langchain-ai.github.io/langgraph/",
    ),
    r"\bAnthropic\b|\bClaude\b": (
        "Anthropic documentation",
        "https://docs.anthropic.com/",
    ),
    r"\bOpenAI\b|\bGPT-": (
        "OpenAI platform documentation",
        "https://platform.openai.com/docs",
    ),
    r"\beBPF\b": ("ebpf.io: What is eBPF?", "https://ebpf.io/what-is-ebpf/"),
    r"\bWebAssembly\b|\bWasm\b": ("WebAssembly", "https://webassembly.org/"),
    r"\bRust\b": ("The Rust Programming Language", "https://doc.rust-lang.org/book/"),
    r"\bGolang\b|\bGo (?:service|routine|code)": (
        "Go documentation",
        "https://go.dev/doc/",
    ),
}
DOCS = {re.compile(k): v for k, v in DOCS.items()}


def link_ok(url: str) -> bool:
    try:
        req = urllib.request.Request(
            url, method="HEAD", headers={"User-Agent": "post-enhancer/1.0"}
        )
        with urllib.request.urlopen(req, timeout=8) as r:
            return r.status < 400
    except Exception:
        return False


def strip_claims(text: str):
    """Drop sentences that assert invented first-hand evidence. Code fences/tables/headings untouched."""
    out, removed, in_code = [], 0, False
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            in_code = not in_code
            out.append(line)
            continue
        if in_code or line.startswith(("#", "|", "- ", "* ", "1.")) or len(line) < 40:
            out.append(line)
            continue
        parts = re.split(r"(?<=[.!?])\s+", line)
        keep = []
        for p in parts:
            if len(p) > 25 and (
                triage.ANECDOTE.search(p)
                or triage.ANEC2.search(p)
                or (triage.EXP.search(p) and re.search(triage.NUM, p))
            ):
                removed += 1
            else:
                keep.append(p)
        out.append(" ".join(keep))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)), removed


LINK_RX = re.compile(
    r"\[([^\]]+)\]\((?:https?://kubaik\.github\.io)?/([a-z0-9][a-z0-9-]*)/?\)"
)
RESERVED = {
    "about",
    "contact",
    "privacy-policy",
    "terms-of-service",
    "ai-content-policy",
    "dmca",
    "author",
    "tag",
    "page",
}


def fix_dead_links(text, live):
    """[anchor](/gone-slug/) -> anchor, for slugs with no post.json (deleted/retired)."""
    n = 0

    def f(m):
        nonlocal n
        if m.group(2) in live or m.group(2) in RESERVED:
            return m.group(0)
        n += 1
        return m.group(1)

    return LINK_RX.sub(f, text), n


def related(slug, posts, sims, k=3):
    return [s for s in sims.get(slug, []) if s in posts][:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="docs")
    ap.add_argument("--verdicts", default="triage_report/verdicts.csv")
    ap.add_argument("--backup", default=".quality_review_backups")
    ap.add_argument("--base-url", default="https://kubaik.github.io")
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--verify-links", action="store_true")
    ap.add_argument("--max-posts", type=int, default=0, help="0 = all")
    a = ap.parse_args()
    docs = Path(a.docs)
    rows = [r for r in csv.DictReader(open(a.verdicts)) if r["verdict"] == "IMPROVE"]
    posts = triage.load(docs)
    deleted = {
        r["slug"] for r in csv.DictReader(open(a.verdicts)) if r["verdict"] == "DELETE"
    }
    keep = {s: d for s, d in posts.items() if s not in deleted}
    # similarity among KEPT posts only, so we never link to something about to be deleted
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    slugs = list(keep)
    M = TfidfVectorizer(
        stop_words="english", sublinear_tf=True, min_df=2
    ).fit_transform(
        [
            keep[s]["title"]
            + " "
            + " ".join(triage.STRIP.sub(" ", keep[s]["content"]).split()[:300])
            for s in slugs
        ]
    )
    S = cosine_similarity(M)
    sims = {}
    for i, s in enumerate(slugs):
        sims[s] = [slugs[j] for j in S[i].argsort()[::-1] if j != i][:6]
    report = {"changed": [], "needs_regeneration": [], "retitle": []}
    ok_cache = {}
    for n, r in enumerate(rows):
        if a.max_posts and n >= a.max_posts:
            break
        slug = r["slug"]
        p = docs / slug / "post.json"
        if slug not in keep or not p.exists():
            continue
        d = json.loads(p.read_text("utf-8"))
        c = d["content"]
        fixes = [f for f in r["fixes"].split(",") if f]
        if "flag_unverified_products" in fixes:
            report["needs_regeneration"].append(slug)
        if "retitle" in fixes:
            report["retitle"].append(slug)
        c, dead_links = fix_dead_links(c, set(keep))
        if dead_links:
            report.setdefault("dead_links_fixed", []).append(
                {"slug": slug, "count": dead_links}
            )
            if a.confirm:
                d["content"] = c
                p.write_text(json.dumps(d, indent=2, ensure_ascii=False), "utf-8")
        if MARK in c:
            continue
        head, sep, old_footer = c.partition("\n---\n\n" + FOOTER_SENTINEL)
        if not sep:
            head, old_footer = c, ""
        c = head
        migrated = bool(sep) and (
            "Editorial standard" in old_footer
            or "direct production experience" in old_footer
            or "Topics are selected from real production experience" in old_footer
            or "Last reviewed" in old_footer
        )
        gen_date = d.get("created_at", "")[:10]
        try:
            gen_date = datetime.fromisoformat(gen_date).strftime("%B %Y")
        except ValueError:
            gen_date = "unknown"
        removed = 0
        if "strip_experience_claims" in fixes:
            c, removed = strip_claims(c)
        tail = []
        if "add_inline_links" in fixes:
            rel = related(slug, keep, sims)
            if rel:
                tail.append(
                    "## Related reading\n\n"
                    + "\n".join(
                        f"- [{keep[x]['title']}]({a.base_url}/{x}/)" for x in rel
                    )
                )
        if "add_sources" in fixes and "official documentation" not in c.lower():
            body = triage.STRIP.sub(" ", c)
            links = []
            for rx, (label, url) in DOCS.items():
                if rx.search(body):
                    if a.verify_links and not ok_cache.setdefault(url, link_ok(url)):
                        continue
                    links.append(f"- [{label}]({url})")
            if links:
                tail.append("## Official documentation\n\n" + "\n".join(links[:5]))
        if not tail and not removed and not migrated:
            continue
        new = (
            c.rstrip()
            + "\n\n"
            + ("\n\n".join(tail) + "\n\n" if tail else "")
            + FOOTER.format(date=gen_date)
            + f"\n{MARK}\n"
        )
        report["changed"].append(
            {
                "slug": slug,
                "sentences_removed": removed,
                "sections_added": len(tail),
                "footer_migrated": migrated,
            }
        )
        if a.confirm:
            bk = Path(a.backup) / slug
            bk.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, bk / "post.json")
            d["content"] = new
            d["audit_enhanced_at"] = datetime.now(timezone.utc).isoformat()
            p.write_text(json.dumps(d, indent=2, ensure_ascii=False), "utf-8")
    Path("enhance_report.json").write_text(json.dumps(report, indent=2))
    print(
        f"{'APPLIED' if a.confirm else 'DRY RUN'}: {len(report['changed'])} posts changed, "
        f"{len(report['needs_regeneration'])} need regeneration, {len(report['retitle'])} need retitle"
    )


if __name__ == "__main__":
    main()
