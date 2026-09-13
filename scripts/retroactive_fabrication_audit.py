"""
scripts/retroactive_fabrication_audit.py
==========================================
Re-scores every already-published post against the CURRENT
_flag_fabricated_anecdotes() / _reject_if_fabricated_citation() logic in
blog_system.py, and pushes hits into regeneration_queue.json — the same
file scripts/quality_gate.py already writes to, so the existing
automated regen pipeline (no manual review step) picks them up on the
next scheduled run.

WHY THIS EXISTS
---------------
blog_system.py's fabrication gates only run at GENERATION time, on new
content. They were tightened over time (see the "FIX (found in review,
2026)" comment on _FABRICATED_CITATION_PATTERNS in blog_system.py) but
that tightening only protects posts written AFTER the fix — the 501
posts already published before it are never re-checked. This script
closes that gap without any manual review: it's read-only against
docs/, it only ever appends to regeneration_queue.json, and it reuses
the exact same gate functions blog_system.py itself uses, so there is
no drift between "what blocks a new post" and "what flags an old one."

USAGE
-----
    python scripts/retroactive_fabrication_audit.py --docs-dir ./docs

Recommended: run this as a step in .github/workflows/blog-automation.yml
right after the existing "Full-corpus content quality gate" step, so
flagged posts merge into the same queue and get picked up by whatever
already consumes regeneration_queue.json:

    - name: 🔎 Retroactive fabrication audit
      run: python scripts/retroactive_fabrication_audit.py --docs-dir ./docs --queue ./regeneration_queue.json
"""

import argparse
import json
import sys
from pathlib import Path

# blog_system.py lives at the repo root, one level up from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blog_system import (  # noqa: E402  (import after sys.path fix, intentional)
    _flag_fabricated_anecdotes,
    _reject_if_fabricated_citation,
)

_SKIP_DIRS = {"static", "tag", "author"}


def audit(docs_dir: Path, queue_path: Path) -> int:
    """
    Scan docs_dir for posts that trip the current fabrication gates.
    Returns the number of newly-flagged posts. Idempotent: re-running
    won't duplicate an already-queued slug.
    """
    queue = []
    if queue_path.exists():
        try:
            queue = json.loads(queue_path.read_text(encoding="utf-8"))
            if not isinstance(queue, list):
                print(f"  ⚠️  {queue_path} did not contain a JSON list — starting fresh.")
                queue = []
        except json.JSONDecodeError:
            print(f"  ⚠️  {queue_path} was not valid JSON — starting fresh.")
            queue = []

    existing_slugs = {q.get("slug") for q in queue if isinstance(q, dict)}

    if not docs_dir.exists():
        print(f"{docs_dir} not found — nothing to audit.")
        return 0

    flagged = 0
    checked = 0

    for post_dir in sorted(docs_dir.iterdir()):
        if not post_dir.is_dir() or post_dir.name in _SKIP_DIRS:
            continue
        post_json = post_dir / "post.json"
        if not post_json.exists():
            continue

        try:
            data = json.loads(post_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"  ⚠️  Skipping {post_dir.name}: post.json is not valid JSON.")
            continue

        checked += 1
        content = data.get("content", "")
        if not content:
            continue

        citation_issue = _reject_if_fabricated_citation(content)
        anecdote_hits = _flag_fabricated_anecdotes(content)

        if citation_issue or anecdote_hits:
            flagged += 1
            reason = citation_issue or (
                f"{len(anecdote_hits)} unverifiable first-person anecdote(s): "
                + "; ".join(anecdote_hits[:3])
            )
            print(f"  ⚠️  {post_dir.name}: {reason}")

            if post_dir.name not in existing_slugs:
                queue.append({
                    "slug": post_dir.name,
                    "title": data.get("title", post_dir.name),
                    "reason": f"retroactive_fabrication_audit: {reason}",
                    "action": "regenerate",
                })
                existing_slugs.add(post_dir.name)

    queue_path.write_text(json.dumps(queue, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{checked} post(s) checked, {flagged} flagged this run.")
    print(f"Queue now has {len(queue)} total entr{'y' if len(queue) == 1 else 'ies'} at {queue_path}")
    return flagged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retroactively re-score existing posts against the "
                     "current fabrication gates and queue hits for automated regeneration."
    )
    parser.add_argument("--docs-dir", default="./docs", help="Path to the docs/ output directory.")
    parser.add_argument(
        "--queue",
        default="./regeneration_queue.json",
        help="Path to the shared regeneration queue (same file quality_gate.py writes to).",
    )
    args = parser.parse_args()

    audit(Path(args.docs_dir), Path(args.queue))