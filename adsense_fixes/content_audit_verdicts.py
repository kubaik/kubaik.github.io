#!/usr/bin/env python3
"""
adsense_fixes/content_audit_verdicts.py
=======================================
Apply the AdSense content-triage verdicts: tombstone DELETE posts,
strip-or-queue IMPROVE posts.

Default is dry-run. Nothing is mutated unless --apply is passed.

Usage:
    python adsense_fixes/content_audit_verdicts.py
    python adsense_fixes/content_audit_verdicts.py --apply
    python adsense_fixes/content_audit_verdicts.py --apply --delete-only
    python adsense_fixes/content_audit_verdicts.py --apply --improve-only
    python blog_system.py apply-verdicts --apply
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from adsense_fixes.claim_gate import check_claims
from adsense_fixes.policy_risk import topic_policy_violation

try:
    from adsense_fixes.canonical_guard import _MERGE_STUB_RE
except ImportError:
    _MERGE_STUB_RE = re.compile(
        r"merged into \[([^\]]+)\]\((/[^\s)]+/)\)",
        re.IGNORECASE,
    )


DOCS_DIR = Path("./docs")
QUEUE_PATH = Path("./regeneration_queue.json")
REMOVED_LOG = DOCS_DIR / "_removed_posts.json"
SKIP_DIRS = {"static", "tag", "author"}

DELETE_VERDICTS: Dict[str, str] = {
    "47-ai-code-debt-in-2026": "Fabricated studies + queue needs_review",
    "ab-pricing-tests-that-lifted-arr-18": "Fabricated ProfitWell stats",
    "ai-api-docs-what-humans-still-write-in-2026": "Fabricated DORA stats",
    "ai-ate-our-niche-heres-how-we-picked-a-new-one": "Fabricated McKinsey Kenya stats",
    "ai-codebases-guardrails-vs-chaos-in-2026": "Fabricated African Tech Policy Forum survey",
    "ai-codes-hidden-cves-exposed": "Fabricated Snyk / LF / OWASP benchmarks",
    "ai-scans-your-code-real-risks-and-defenses": "Fabricated Trail of Bits study",
    "ai-services-slow-fix-the-network-first": "Fabricated CNCF survey",
    "batteries-included-offline-mode-for-unreliable-networks": "Unsourced market research",
    "claude-4-prompts-broke-my-prod": "Fabricated Stanford HAI stat",
    "claude-code-in-2026-one-year-of-lessons-learned": "Fabricated McKinsey stat",
    "dagster-replaced-airflow-in-2026-heres-the-stack-we": "Invented sentiment survey",
    "engineering-interviews-what-ai-wants-now": "Fabricated KSEA / NTRG studies",
    "freelance-rates-after-ai-took-the-boilerplate": "Fabricated Upwork survey",
    "hiring-engineers-the-3-things-ai-cares-about-now": "Fabricated DevIQ Labs study",
    "internal-dev-portal-that-developers-open-daily": "Fabricated Dev Interrupted study",
    "kubernetes-in-2026-what-teams-offload": "Fabricated CNCF survey",
    "land-a-45k-remote-job-africas-2026-map": "Unverifiable salary-map claims",
    "latency-spike-after-llm-feature-rollout": "Invented incident numbers",
    "llm-evals-the-two-metrics-that-bite-you": "Fabricated eval benchmarks",
    "llm-security-mistakes-in-2026-owasp-top-10-fixes": "Fake OWASP attribution",
    "money-moves-need-stricter-parsers": "Unsourced fintech incident stats",
    "negotiate-salary-2026-with-ai-resume-tool": "Thin career bait + invented comps",
    "roll-your-own-migrations-why-blue-green-fails-you": "Fabricated migration-failure stats",
    "scan-10k-repos-daily-without-false-positives": "Fabricated scanner benchmarks",
    "senior-devs-flee-big-tech-not-for-money": "Unverifiable attrition anecdotes",
    "skip-the-zero-trust-hype-for-5-person-teams": "Fabricated survey claims",
    "stop-shipping-daily-the-burn-rate-math": "Invented burn-rate figures",
    "survive-ai-swipes-with-niche-picks": "Thin / keyword-farm",
    "treat-every-user-input-as-hostile-in-2026": "Fabricated security stats",
    "why-your-model-evaluation-missed-a-production-regression": "Invented eval-miss numbers",
    "zero-trust-mcp-servers-auth-trap-youll-hit-first": "Fabricated auth-incident stats",
    "2026-technical-screens-what-actually-works": "Queue needs_review + fabricated hiring claims",
    "vc-insights": "HTML orphan / generic meta",
    "ai-tools-for-african-devs-what-actually-works-now": "Thin + 404 live",
    "api-attacks-rise-300-in-2026-whats-breaking-first": "Thin + unsourced 300% headline",
    "mcp-servers-2026-why-your-agent-keeps-crashing": "Lowest quality score, thin-ish",
    "3-side-gigs-developers-use-for-12kmo-in-2026": "Income-claim bait",
    "compound-nairobi-6-dev-income-streams-2026": "Income-claim bait cluster",
    "ship-production-grade-tools-in-90-minutes-with-the": "Truncated slug + low-value how-to",
}

IMPROVE_VERDICTS: Dict[str, str] = {
    "stop-putting-llm-keys-in-env": "Strip fake audit percentages; keep the secrets-gateway angle",
    "handoff-latency-kills-systems-60-fix-is-free": "Drop unsourced 60%; add real code",
    "world-models-wont-save-your-backend": "Strip invented benchmarks; keep architecture argument",
    "tech-salaries-2026-what-corrected-what-grew": "Remove unofficial compensation figures",
    "handle-starlink-4g-fallbacks-in-fastapi-0115": "Keep; strip first-person incident lines",
    "2026-islands-astros-hydration-edge": "Raise depth; strip unsourced numbers",
    "3-iac-tools-in-2026-which-one-bites-you-first": "Keep comparison; require sourced claims only",
    "htmx-5-things-i-dropped-to-make-it-work": "Fix structure; strip autobiography",
}

TOMBSTONE_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>This post has been removed</title>
    <meta name="robots" content="noindex, follow">
    <link rel="canonical" href="https://kubaik.github.io/">
    <style>
        body {{ font-family: system-ui, sans-serif; margin: 0; background: #fafafa; }}
        .tombstone {{ text-align: center; padding: 4rem 1.5rem; max-width: 480px; margin: 0 auto; }}
        .tombstone h1 {{ font-size: 1.4rem; margin-bottom: 0.75rem; }}
        .tombstone p {{ color: #666; line-height: 1.5; }}
        .tombstone a {{
            display: inline-block; margin-top: 1.25rem; color: #fff;
            background: #6366f1; padding: 0.6rem 1.25rem;
            border-radius: 6px; text-decoration: none; font-weight: 600;
        }}
    </style>
</head>
<body>
    <div class="tombstone">
        <h1>This post has been removed</h1>
        <p>It didn't meet our current content quality bar and has been taken down as part of a routine editorial review. Sorry for the dead link.</p>
        <a href="/">Browse the blog</a>
    </div>
</body>
</html>
"""

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CODE_FENCE = re.compile(r"```[\s\S]*?```")


def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def _post_json_path(slug: str) -> Path:
    return DOCS_DIR / slug / "post.json"


def _discover_merge_stubs() -> Dict[str, str]:
    extra = {}
    if not DOCS_DIR.exists():
        return extra
    for post_dir in DOCS_DIR.iterdir():
        if not post_dir.is_dir() or post_dir.name in SKIP_DIRS:
            continue
        pj = post_dir / "post.json"
        if not pj.exists():
            # HTML-only orphan under a known delete slug is handled elsewhere
            continue
        try:
            data = json.loads(pj.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            extra[post_dir.name] = "Unreadable post.json"
            continue
        content = data.get("content", "") or ""
        if _MERGE_STUB_RE.search(content) or data.get("redirect_to"):
            extra[post_dir.name] = "Merge-redirect stub (thin / self-canonical risk)"
    return extra


def _slug_exists(slug: str) -> bool:
    return (DOCS_DIR / slug).exists()


def tombstone_slug(slug: str, reason: str, apply: bool) -> str:
    post_dir = DOCS_DIR / slug
    if not post_dir.exists():
        return f"MISSING  /{slug}/  ({reason})"
    if not apply:
        return f"DRY-RUN delete  /{slug}/  ({reason})"

    post_dir.mkdir(parents=True, exist_ok=True)
    for stale in list(post_dir.iterdir()):
        if stale.name == "index.html":
            continue
        if stale.is_dir():
            shutil.rmtree(stale, ignore_errors=True)
        else:
            stale.unlink(missing_ok=True)
    (post_dir / "index.html").write_text(TOMBSTONE_HTML, encoding="utf-8")

    log = _load_json(REMOVED_LOG, {})
    log[slug] = {
        "removed_at": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "source": "content_audit_verdicts",
    }
    REMOVED_LOG.parent.mkdir(parents=True, exist_ok=True)
    REMOVED_LOG.write_text(json.dumps(log, indent=2), encoding="utf-8")
    return f"TOMBSTONED /{slug}/  ({reason})"


def _strip_risky_sentences(content: str) -> Tuple[str, int]:
    """Drop sentences the claim gate would flag. Preserve fenced code."""
    fences: List[str] = []

    def _mask(match: re.Match) -> str:
        fences.append(match.group(0))
        return f"\x00CODE{len(fences) - 1}\x00"

    masked = _CODE_FENCE.sub(_mask, content or "")
    kept_parts: List[str] = []
    removed = 0
    for block in masked.split("\n\n"):
        if "\x00CODE" in block or block.strip().startswith("#"):
            kept_parts.append(block)
            continue
        sentences = _SENTENCE_SPLIT.split(block)
        kept_sents = []
        for sent in sentences:
            probe = check_claims(sent, "")
            if probe.blocked:
                removed += 1
                continue
            kept_sents.append(sent)
        rebuilt = " ".join(s.strip() for s in kept_sents if s.strip())
        if rebuilt:
            kept_parts.append(rebuilt)
    new_content = "\n\n".join(kept_parts)
    for i, fence in enumerate(fences):
        new_content = new_content.replace(f"\x00CODE{i}\x00", fence)
    return new_content, removed


def _enqueue_improve(slug: str, reason: str) -> None:
    queue = _load_json(QUEUE_PATH, [])
    if not isinstance(queue, list):
        queue = []
    queue = [e for e in queue if e.get("slug") != slug]
    queue.append({
        "slug": slug,
        "action": "regenerate",
        "status": None,
        "reason": f"content_audit_verdicts: {reason}",
        "queued_at": datetime.now(timezone.utc).isoformat(),
    })
    QUEUE_PATH.write_text(json.dumps(queue, indent=2), encoding="utf-8")


def improve_slug(slug: str, reason: str, apply: bool) -> str:
    pj = _post_json_path(slug)
    if not pj.exists():
        return f"MISSING  /{slug}/  (improve skipped — {reason})"

    data = json.loads(pj.read_text(encoding="utf-8"))
    title = data.get("title", slug)
    content = data.get("content", "")
    before = len(content.split())
    cleaned, removed = _strip_risky_sentences(content)
    after = len(cleaned.split())
    title_hit = topic_policy_violation(title)
    claim = check_claims(cleaned, title)

    if title_hit or after < 1800:
        demote_reason = title_hit or f"Improve strip left {after} words (< 1800)"
        return tombstone_slug(slug, f"demoted from improve: {demote_reason}", apply)

    if not apply:
        still = "; ".join(claim.reasons) if claim.blocked else "clean after strip"
        return (
            f"DRY-RUN improve /{slug}/  words {before}->{after}, "
            f"stripped {removed} sentence(s), remaining={still}"
        )

    data["content"] = cleaned
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    data["audit_improved_at"] = data["updated_at"]
    data["audit_improve_reason"] = reason
    pj.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    md = DOCS_DIR / slug / "index.md"
    if md.exists() or True:
        md.write_text(f"# {title}\n\n{cleaned}", encoding="utf-8")
    if claim.blocked:
        _enqueue_improve(slug, "; ".join(claim.reasons))
        return (
            f"IMPROVED /{slug}/  stripped {removed} sentence(s), "
            f"{after} words, queued for regenerate ({'; '.join(claim.reasons)})"
        )
    return f"IMPROVED /{slug}/  stripped {removed} sentence(s), {after} words, clean"


def run(apply: bool, delete_only: bool, improve_only: bool) -> int:
    delete_map = dict(DELETE_VERDICTS)
    delete_map.update(_discover_merge_stubs())

    print("Content Audit Verdicts")
    print("=" * 64)
    print(f"Mode        : {'APPLY' if apply else 'DRY-RUN'}")
    print(f"Docs        : {DOCS_DIR.resolve()}")
    print(f"Delete list : {len(delete_map)} slugs (includes live merge stubs)")
    print(f"Improve list: {len(IMPROVE_VERDICTS)} slugs")
    print("=" * 64)

    actions: List[str] = []
    if not improve_only:
        print("\nDELETE")
        for slug, reason in sorted(delete_map.items()):
            if slug in IMPROVE_VERDICTS:
                continue
            line = tombstone_slug(slug, reason, apply)
            actions.append(line)
            print(f"  {line}")

    if not delete_only:
        print("\nIMPROVE")
        for slug, reason in sorted(IMPROVE_VERDICTS.items()):
            if slug in delete_map:
                line = tombstone_slug(slug, f"also on delete list: {reason}", apply)
            else:
                line = improve_slug(slug, reason, apply)
            actions.append(line)
            print(f"  {line}")

    applied = sum(1 for a in actions if a.startswith(("TOMBSTONED", "IMPROVED")))
    missing = sum(1 for a in actions if a.startswith("MISSING"))
    print("\n" + "=" * 64)
    print(f"Done. apply={apply} changed={applied} missing={missing}")
    if apply:
        print("Next: python blog_system.py build")
    else:
        print("Re-run with --apply to tombstone / strip.")
    return 0


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(description="Apply AdSense content audit verdicts")
    parser.add_argument("--apply", action="store_true", help="Write changes (default: dry-run)")
    parser.add_argument("--delete-only", action="store_true")
    parser.add_argument("--improve-only", action="store_true")
    args = parser.parse_args(argv)
    return run(args.apply, args.delete_only, args.improve_only)


if __name__ == "__main__":
    raise SystemExit(main())
