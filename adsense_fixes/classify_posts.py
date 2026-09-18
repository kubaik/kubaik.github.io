#!/usr/bin/env python3
"""
adsense_fixes/classify_posts.py
================================
Read-only classifier: walk docs/*/post.json and say whether each post
falls in DELETE, IMPROVE, or OK under the AdSense content-audit rules.

Does not tombstone, rewrite, or queue anything.

Usage:
    python adsense_fixes/classify_posts.py
    python adsense_fixes/classify_posts.py --docs ./docs
    python adsense_fixes/classify_posts.py --json report.json
    python adsense_fixes/classify_posts.py --only DELETE
    python adsense_fixes/classify_posts.py --slug stop-putting-llm-keys-in-env


    # inspect only
python adsense_fixes/classify_posts.py --only DELETE
python adsense_fixes/classify_posts.py --only IMPROVE

# strip IMPROVE sentences + tombstone the real DELETE list
python adsense_fixes/content_audit_verdicts.py --apply --improve-only

python adsense_fixes/content_audit_verdicts.py --apply --delete-only
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from adsense_fixes.claim_gate import check_claims
from adsense_fixes.policy_risk import is_static_page, topic_policy_violation, SKIP_SITE_DIRS

try:
    from adsense_fixes.canonical_guard import _MERGE_STUB_RE
except ImportError:
    _MERGE_STUB_RE = re.compile(
        r"merged into \[([^\]]+)\]\((/[^\s)]+/)\)",
        re.IGNORECASE,
    )


SKIP_DIRS = set(SKIP_SITE_DIRS)
MIN_WORDS_DELETE = 1500
MIN_WORDS_IMPROVE = 1800

# Known-bad slugs from the manual audit. Still classified by content first;
# this only adds a reason when the slug is present.
KNOWN_DELETE = {
    "47-ai-code-debt-in-2026",
    "ab-pricing-tests-that-lifted-arr-18",
    "ai-api-docs-what-humans-still-write-in-2026",
    "ai-ate-our-niche-heres-how-we-picked-a-new-one",
    "ai-codebases-guardrails-vs-chaos-in-2026",
    "ai-codes-hidden-cves-exposed",
    "ai-scans-your-code-real-risks-and-defenses",
    "ai-services-slow-fix-the-network-first",
    "batteries-included-offline-mode-for-unreliable-networks",
    "claude-4-prompts-broke-my-prod",
    "claude-code-in-2026-one-year-of-lessons-learned",
    "dagster-replaced-airflow-in-2026-heres-the-stack-we",
    "engineering-interviews-what-ai-wants-now",
    "freelance-rates-after-ai-took-the-boilerplate",
    "hiring-engineers-the-3-things-ai-cares-about-now",
    "internal-dev-portal-that-developers-open-daily",
    "kubernetes-in-2026-what-teams-offload",
    "land-a-45k-remote-job-africas-2026-map",
    "latency-spike-after-llm-feature-rollout",
    "llm-evals-the-two-metrics-that-bite-you",
    "llm-security-mistakes-in-2026-owasp-top-10-fixes",
    "money-moves-need-stricter-parsers",
    "negotiate-salary-2026-with-ai-resume-tool",
    "roll-your-own-migrations-why-blue-green-fails-you",
    "scan-10k-repos-daily-without-false-positives",
    "senior-devs-flee-big-tech-not-for-money",
    "skip-the-zero-trust-hype-for-5-person-teams",
    "stop-shipping-daily-the-burn-rate-math",
    "survive-ai-swipes-with-niche-picks",
    "treat-every-user-input-as-hostile-in-2026",
    "why-your-model-evaluation-missed-a-production-regression",
    "zero-trust-mcp-servers-auth-trap-youll-hit-first",
    "2026-technical-screens-what-actually-works",
    "vc-insights",
    "ai-tools-for-african-devs-what-actually-works-now",
    "api-attacks-rise-300-in-2026-whats-breaking-first",
    "mcp-servers-2026-why-your-agent-keeps-crashing",
    "3-side-gigs-developers-use-for-12kmo-in-2026",
    "compound-nairobi-6-dev-income-streams-2026",
    "ship-production-grade-tools-in-90-minutes-with-the",
}

KNOWN_IMPROVE = {
    "stop-putting-llm-keys-in-env",
    "handoff-latency-kills-systems-60-fix-is-free",
    "world-models-wont-save-your-backend",
    "tech-salaries-2026-what-corrected-what-grew",
    "handle-starlink-4g-fallbacks-in-fastapi-0115",
    "2026-islands-astros-hydration-edge",
    "3-iac-tools-in-2026-which-one-bites-you-first",
    "htmx-5-things-i-dropped-to-make-it-work",
}

_BOILERPLATE = [
    "class {topic_slug}Client",
    "class Client:",
    "{topic_slug}",
    "{topic}",
    "topic_slug",
]

_CRITICAL_FILLER = [
    "in today's rapidly evolving",
    "in the ever-changing landscape",
    "in today's fast-paced",
    "as an ai language model",
    "as a large language model",
    "harness the power of",
    "unlock the potential of",
    "paradigm shift",
    "game-changer",
]


@dataclass
class Verdict:
    slug: str
    title: str
    category: str  # DELETE | IMPROVE | OK | MISSING
    word_count: int
    reasons: List[str] = field(default_factory=list)
    claim_hits: List[str] = field(default_factory=list)
    has_code: bool = False
    is_stub: bool = False
    path: str = ""


def _word_count(text: str) -> int:
    return len((text or "").split())


def classify_post(slug: str, data: Dict, path: Path) -> Verdict:
    title = (data.get("title") or slug).strip()
    content = data.get("content") or ""
    wc = _word_count(content)
    lower = content.lower()
    v = Verdict(
        slug=slug,
        title=title,
        category="OK",
        word_count=wc,
        has_code="```" in content,
        path=str(path),
    )

    if data.get("redirect_to") or _MERGE_STUB_RE.search(content):
        v.is_stub = True
        v.category = "DELETE"
        v.reasons.append("Merge-redirect stub / thin placeholder")
        return v

    if not title or not content.strip():
        v.category = "DELETE"
        v.reasons.append("Empty title or body")
        return v

    title_hit = topic_policy_violation(title)
    if title_hit:
        v.category = "DELETE"
        v.reasons.append(f"Policy-risk title: {title_hit}")

    claims = check_claims(content, title)
    v.claim_hits = list(claims.hits)
    if claims.blocked:
        v.reasons.extend(f"Claim: {r}" for r in claims.reasons)

    for marker in _BOILERPLATE:
        if marker in content:
            v.reasons.append(f"Template boilerplate: {marker}")
            break

    for phrase in _CRITICAL_FILLER:
        if phrase in lower:
            v.reasons.append(f"AI-filler phrase: {phrase}")
            break

    if wc < MIN_WORDS_DELETE:
        v.reasons.append(f"Thin content ({wc} < {MIN_WORDS_DELETE} words)")
    elif wc < MIN_WORDS_IMPROVE:
        v.reasons.append(f"Below target length ({wc} < {MIN_WORDS_IMPROVE} words)")

    technical = bool(re.search(
        r"\b(python|fastapi|redis|postgres|kubernetes|docker|llm|agent|api|react)\b",
        f"{title} {content}",
        re.IGNORECASE,
    ))
    if technical and content.count("```") < 2:
        v.reasons.append("Technical post with fewer than two code fences")

    # Decide bucket
    delete_signals = {
        "Merge-redirect stub / thin placeholder",
        "Empty title or body",
    }
    hard = [
        r for r in v.reasons
        if r.startswith("Policy-risk title:")
        or r.startswith("Template boilerplate:")
        or r.startswith("AI-filler phrase:")
        or r.startswith("Thin content")
        or r in delete_signals
        or "fake_authority" in v.claim_hits
        or "named_org_stat" in v.claim_hits
    ]
    # Multiple fabricated-authority hits, or income bait, or known-delete + claims
    named_claim_count = v.claim_hits.count("named_org_stat") + v.claim_hits.count("fake_authority")

    if v.category == "DELETE":
        return v

    if hard and (wc < MIN_WORDS_IMPROVE or title_hit or named_claim_count >= 1 or slug in KNOWN_DELETE):
        v.category = "DELETE"
        if slug in KNOWN_DELETE and "On manual DELETE list" not in v.reasons:
            v.reasons.append("On manual DELETE list")
        return v

    if v.reasons or slug in KNOWN_IMPROVE:
        v.category = "IMPROVE"
        if slug in KNOWN_IMPROVE and "On manual IMPROVE list" not in v.reasons:
            v.reasons.append("On manual IMPROVE list")
        return v

    if slug in KNOWN_DELETE:
        v.category = "DELETE"
        v.reasons.append("On manual DELETE list")
        return v

    v.category = "OK"
    return v


def scan(docs_dir: Path, only_slug: Optional[str] = None) -> List[Verdict]:
    results: List[Verdict] = []
    if not docs_dir.exists():
        return results

    slugs = [only_slug] if only_slug else sorted(
        p.name for p in docs_dir.iterdir()
        if p.is_dir() and p.name not in SKIP_DIRS
    )

    for slug in slugs:
        if is_static_page(slug):
            continue
        post_dir = docs_dir / slug
        pj = post_dir / "post.json"
        if not post_dir.exists():
            results.append(Verdict(
                slug=slug, title="", category="MISSING",
                word_count=0, reasons=["Directory not found"], path=str(post_dir),
            ))
            continue
        if not pj.exists():
            # HTML-only orphan
            html = post_dir / "index.html"
            if html.exists():
                results.append(Verdict(
                    slug=slug, title=slug, category="DELETE",
                    word_count=0,
                    reasons=["HTML orphan — no post.json"],
                    path=str(html),
                ))
            else:
                results.append(Verdict(
                    slug=slug, title="", category="MISSING",
                    word_count=0, reasons=["No post.json"], path=str(post_dir),
                ))
            continue
        try:
            data = json.loads(pj.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            results.append(Verdict(
                slug=slug, title="", category="DELETE",
                word_count=0, reasons=["Unreadable post.json"], path=str(pj),
            ))
            continue
        results.append(classify_post(slug, data, pj))
    return results


def print_report(results: List[Verdict], only: Optional[str] = None) -> None:
    if only:
        results = [r for r in results if r.category == only]

    buckets: Dict[str, List[Verdict]] = {"DELETE": [], "IMPROVE": [], "OK": [], "MISSING": []}
    for r in results:
        buckets.setdefault(r.category, []).append(r)

    print("Content Audit Classifier (read-only)")
    print("=" * 72)
    print(
        f"Scanned {len(results)}  |  "
        f"DELETE {len(buckets['DELETE'])}  "
        f"IMPROVE {len(buckets['IMPROVE'])}  "
        f"OK {len(buckets['OK'])}  "
        f"MISSING {len(buckets['MISSING'])}"
    )
    print("=" * 72)

    for cat in ("DELETE", "IMPROVE", "MISSING", "OK"):
        rows = buckets.get(cat, [])
        if not rows:
            continue
        print(f"\n{cat} ({len(rows)})")
        print("-" * 72)
        for r in rows:
            reasons = "; ".join(r.reasons) if r.reasons else "—"
            print(f"  /{r.slug}/")
            print(f"      {r.word_count}w  code={r.has_code}  {r.title[:70]}")
            print(f"      {reasons}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Classify published posts as DELETE / IMPROVE / OK"
    )
    parser.add_argument("--docs", default="./docs", help="docs/ directory")
    parser.add_argument("--json", dest="json_out", help="Write full report JSON")
    parser.add_argument("--only", choices=["DELETE", "IMPROVE", "OK", "MISSING"])
    parser.add_argument("--slug", help="Classify a single slug")
    args = parser.parse_args(argv)

    if args.slug and is_static_page(args.slug):
        print(f"SKIPPED /{args.slug}/ — static site page, not a blog post")
        return 0

    results = scan(Path(args.docs), only_slug=args.slug)
    if not results:
        print(f"No posts found under {Path(args.docs).resolve()}")
        print("Run this from the repo root so ./docs exists.")
        return 1

    print_report(results, only=args.only)

    if args.json_out:
        payload = [asdict(r) for r in results]
        Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
