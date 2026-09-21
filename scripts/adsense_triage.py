#!/usr/bin/env python3
"""
scripts/adsense_triage.py
=========================
Classify every post in docs/ for AdSense-readiness triage.

Reads post.json for every published post, runs the same detection
primitives blog_system.py uses at publish time (narrative title, first-
person incident, fabricated citation), and writes three bucket files:

    .adsense_triage/delete.json
    .adsense_triage/improve.json
    .adsense_triage/keep.json

The classification is deliberately conservative: when in doubt, a post
goes to DELETE rather than IMPROVE, because AdSense review samples
randomly and a single bad post in the sample can trigger rejection.

Usage:
    python scripts/adsense_triage.py --docs-dir ./docs
    python scripts/adsense_triage.py --docs-dir ./docs --print-summary
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Reuse the exact gates blog_system.py already uses, so triage decisions
# match publish-time decisions and can't drift.
from blog_system import (
    _reject_if_narrative_title,
    _reject_if_fabricated_citation,
    _flag_fabricated_anecdotes,
)


_SKIP_DIRS = {"static", "tag", "author", "page"}


def _load_post(post_dir: Path) -> dict | None:
    pj = post_dir / "post.json"
    if not pj.exists():
        return None
    try:
        return json.loads(pj.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None


def _classify(slug: str, data: dict) -> dict:
    """
    Return {"bucket": "delete"|"improve"|"keep", "reasons": [...]}.

    DELETE if any of:
      - narrative title (fabricated-incident title)
      - fabricated named-source citation in body
      - thin content (< 1800 words)

    IMPROVE if any of:
      - first-person incident in body (fixable by rewriting sentences)

    KEEP otherwise.
    """
    title = (data.get("title") or "").strip()
    content = (data.get("content") or "").strip()
    wc = len(content.split())

    delete_reasons: list[str] = []
    improve_reasons: list[str] = []

    narrative = _reject_if_narrative_title(title)
    if narrative:
        delete_reasons.append(narrative)

    citation = _reject_if_fabricated_citation(content)
    if citation:
        delete_reasons.append(citation)

    if wc < 1800:
        delete_reasons.append(f"thin content ({wc} words)")

    incidents = _flag_fabricated_anecdotes(content)
    if incidents:
        improve_reasons.append(f"{len(incidents)} first-person incident(s)")

    if delete_reasons:
        return {"bucket": "delete", "reasons": delete_reasons, "word_count": wc}
    if improve_reasons:
        return {"bucket": "improve", "reasons": improve_reasons, "word_count": wc}
    return {"bucket": "keep", "reasons": [], "word_count": wc}


def _run_triage(docs_dir: Path) -> dict:
    out = {
        "generated_at": datetime.now().isoformat(),
        "docs_dir": str(docs_dir),
        "delete": [],
        "improve": [],
        "keep": [],
    }

    for post_dir in sorted(docs_dir.iterdir()):
        if not post_dir.is_dir() or post_dir.name in _SKIP_DIRS:
            continue
        data = _load_post(post_dir)
        if not data:
            continue
        title = (data.get("title") or "").strip()
        if not title:
            # Not a real post — skip silently.
            continue
        result = _classify(post_dir.name, data)
        entry = {
            "slug": post_dir.name,
            "title": title,
            "word_count": result["word_count"],
            "reasons": result["reasons"],
        }
        out[result["bucket"]].append(entry)

    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs-dir", default="./docs")
    parser.add_argument(
        "--out-dir", default="./.adsense_triage",
        help="Directory for the three bucket JSON files.",
    )
    parser.add_argument(
        "--print-summary", action="store_true",
        help="Print per-bucket counts and worst-10 examples.",
    )
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir)
    if not docs_dir.exists():
        print(f"docs dir not found: {docs_dir}")
        return 1

    out = _run_triage(docs_dir)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for bucket in ("delete", "improve", "keep"):
        (out_dir / f"{bucket}.json").write_text(
            json.dumps(out[bucket], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    total = sum(len(out[b]) for b in ("delete", "improve", "keep"))
    print(f"Triaged {total} posts → {out_dir}/")
    for bucket in ("delete", "improve", "keep"):
        print(f"  {bucket:8s}: {len(out[bucket]):4d} posts")

    if args.print_summary and out["delete"]:
        print("\nDELETE preview (worst 10):")
        for entry in out["delete"][:10]:
            print(f"  {entry['slug']}")
            for r in entry["reasons"][:2]:
                print(f"    - {r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())