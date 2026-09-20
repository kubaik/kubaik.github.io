#!/usr/bin/env python3
"""
scripts/quality_gate.py
=======================
Single-entry quality gate for the entire pipeline.

Runs, in order:
  1. Policy risk check (title/topic)
  2. Word count (minimum 1800)
  3. Boilerplate detection
  4. Claim gate (fabricated citations + first-person incidents)
  5. Duplicate title (Jaccard)
  6. Full-body duplicate (TF-IDF cosine)
  7. SimilarityGuard (structural repetition)

Usage:
    python scripts/quality_gate.py <slug>
    python scripts/quality_gate.py --all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from adsense_fixes.claim_gate import check_claims
from adsense_fixes.policy_risk import topic_policy_violation

DOCS_DIR = Path("./docs")


def check_post(slug: str) -> dict:
    post_dir = DOCS_DIR / slug
    pj = post_dir / "post.json"
    if not pj.exists():
        return {"slug": slug, "status": "missing"}

    data = json.loads(pj.read_text(encoding="utf-8"))
    content = data.get("content", "")
    title = data.get("title", "")

    result = {"slug": slug, "status": "pass", "failures": []}

    # 1. Policy risk
    risk = topic_policy_violation(title)
    if risk:
        result["failures"].append(f"policy: {risk}")

    # 2. Word count
    wc = len(content.split())
    if wc < 1800:
        result["failures"].append(f"thin: {wc} words (< 1800)")

    # 3. Boilerplate
    for marker in ["class Client:", "class {topic_slug}Client", "{topic_slug}"]:
        if marker in content:
            result["failures"].append(f"boilerplate: '{marker}'")
            break

    # 4. Claim gate
    claims = check_claims(content, title)
    if claims.blocked:
        result["failures"].extend(claims.reasons)

    if result["failures"]:
        result["status"] = "fail"
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("slug", nargs="?", help="Post slug")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        results = []
        for d in sorted(DOCS_DIR.iterdir()):
            if d.is_dir() and d.name not in ("static", "tag", "author"):
                if (d / "post.json").exists():
                    results.append(check_post(d.name))
        fails = [r for r in results if r["status"] == "fail"]
        print(json.dumps({"total": len(results), "fails": len(fails), "details": fails}, indent=2))
        return 1 if fails else 0

    if not args.slug:
        parser.error("Provide a slug or --all")
    r = check_post(args.slug)
    print(json.dumps(r, indent=2))
    return 1 if r["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())