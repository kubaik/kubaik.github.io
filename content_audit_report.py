"""
content_audit_report.py
========================
Run this INSIDE the repo (same directory as blog_system.py, next to docs/)
to produce the actual DELETE / IMPROVE verdict list for every published
post. This is the tool an outside reviewer would need repo access to
generate — run it yourself and paste the output back if you want a
verdict-by-verdict review of specific posts.

Usage:
    python content_audit_report.py                # human-readable report
    python content_audit_report.py --json out.json

Verdicts (any ONE hard-fail = DELETE; else IMPROVE if warnings; else OK):

DELETE if any of:
  - Unfilled template boilerplate (BOILERPLATE_FALLBACK_MARKERS)
  - word_count < 1800 (matches the site's own publish-time gate — see
    MIN_WORD_PURGE fix in blog_system.py)
  - monetization_data.review_status == "recovered_from_markdown" AND
    word_count < 1800 (recovered posts with no real body)
  - near-duplicate of another published post (Jaccard shingle overlap > 0.6)

IMPROVE if OK on the above but missing any of:
  - "### About this article" E-E-A-T footer
  - a markdown table (comparison signal)
  - a FAQ section
  - a versioned tool/library reference (e.g. "Python 3.12")
  - a concrete metric (%, ms, rps, cost, latency)
  - fewer than 2 fenced code blocks (for posts tagged as technical)

Everything else: OK, no action needed.
"""
import argparse
import json
import re
import sys
from pathlib import Path

DOCS_DIR = Path("./docs")

BOILERPLATE_FALLBACK_MARKERS = [
    "class {topic_slug}Client", "class Client:", "max_retries = config.get",
    "{topic_slug}", "{topic}", "topic_slug",
]

HARD_MIN_WORDS = 1800


def _count_words(text: str) -> int:
    return len(text.split())


def _shingles(text: str, n: int = 5) -> set:
    words = re.sub(r"[^\w\s]", "", text.lower()).split()
    return {" ".join(words[i:i + n]) for i in range(max(len(words) - n + 1, 0))}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def load_posts(docs_dir: Path):
    posts = []
    for post_dir in sorted(docs_dir.iterdir()):
        if not post_dir.is_dir() or post_dir.name == "static":
            continue
        pj = post_dir / "post.json"
        if not pj.exists():
            continue
        try:
            data = json.loads(pj.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  ! could not parse {pj}: {e}", file=sys.stderr)
            continue
        posts.append((post_dir.name, data))
    return posts


def audit(docs_dir: Path):
    posts = load_posts(docs_dir)
    shingle_cache = {slug: _shingles(d.get("content", "")) for slug, d in posts}

    report = {"delete": [], "improve": [], "ok": []}

    for slug, data in posts:
        content = data.get("content", "")
        title = data.get("title", "")
        lower = content.lower()
        wc = _count_words(content)
        reasons = []

        if any(m in content for m in BOILERPLATE_FALLBACK_MARKERS):
            reasons.append("Unfilled template boilerplate present")

        if wc < HARD_MIN_WORDS:
            reasons.append(f"Thin content: {wc} words (< {HARD_MIN_WORDS})")

        if data.get("monetization_data", {}).get("review_status") == "recovered_from_markdown" and wc < HARD_MIN_WORDS:
            reasons.append("Recovered stub with no real body")

        # near-duplicate check against every other post
        my_shingles = shingle_cache[slug]
        for other_slug, other_shingles in shingle_cache.items():
            if other_slug == slug:
                continue
            sim = _jaccard(my_shingles, other_shingles)
            if sim > 0.6:
                reasons.append(f"{sim:.0%} near-duplicate of '{other_slug}'")
                break

        if reasons:
            report["delete"].append({"slug": slug, "title": title, "word_count": wc, "reasons": reasons})
            continue

        warnings = []
        if "### about this article" not in lower:
            warnings.append("Missing E-E-A-T author footer")
        if "|" not in content:
            warnings.append("No comparison/data table")
        if "frequently asked questions" not in lower and "## faq" not in lower:
            warnings.append("No FAQ section (FAQPage schema opportunity)")
        if not re.search(r"\b(python\s*3\.\d+|node\.?js\s*\d+|postgres(?:ql)?\s*\d+|redis\s*\d+|kubernetes\s*1\.\d+)\b", content, re.I):
            warnings.append("No versioned tool/library reference")
        if not re.search(r"\b(\d+%|\d+\s*ms|\d+\s*rps|\$\d+)\b", content):
            warnings.append("No concrete metric")
        if content.count("```") < 4:
            warnings.append("Fewer than 2 fenced code blocks")

        if warnings:
            report["improve"].append({"slug": slug, "title": title, "word_count": wc, "warnings": warnings})
        else:
            report["ok"].append({"slug": slug, "title": title, "word_count": wc})

    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="./docs")
    ap.add_argument("--json", default=None, help="write JSON report to this path")
    args = ap.parse_args()

    docs_dir = Path(args.docs)
    if not docs_dir.exists():
        print(f"docs dir not found: {docs_dir}", file=sys.stderr)
        sys.exit(1)

    report = audit(docs_dir)

    print(f"\n=== Content Audit: {docs_dir} ===")
    print(f"DELETE:  {len(report['delete'])}")
    print(f"IMPROVE: {len(report['improve'])}")
    print(f"OK:      {len(report['ok'])}\n")

    print("--- DELETE ---")
    for p in report["delete"]:
        print(f"  {p['slug']}  ({p['word_count']}w)  — {'; '.join(p['reasons'])}")

    print("\n--- IMPROVE ---")
    for p in report["improve"]:
        print(f"  {p['slug']}  ({p['word_count']}w)  — {'; '.join(p['warnings'])}")

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"\nWrote JSON report to {args.json}")


if __name__ == "__main__":
    main()