"""
scripts/process_citation_queue.py
====================================
Bounded, automated consumer for citation_audit_report.json.

WHY THIS EXISTS
---------------
citation_audit_report.json is produced by adsense_credibility_cleanup.py's
`scan` mode: 33 posts / 46 sentences with fabricated-sounding institutional
citations ("A 2026 study by OWASP tracked 1,240 CVEs...", attributed to
OWASP, Datadog, LinearB, ProfitWell, Upwork, Hired, and an unverifiable
"DevIQ Labs"). adsense_credibility_cleanup.py deliberately does NOT
auto-strip these — its `review` mode logs a human editorial decision per
match to citation_audit_decisions.json by design, which is the right
default for a one-off manual pass but leaves nothing to close the loop
automatically going forward, and conflicts with keeping the regular
pipeline fully automated.

This is that automated closer — scoped ONLY to citation_audit_report.json,
same as process_regeneration_queue.py is scoped only to
regeneration_queue.json's "retroactive_fabrication_audit:" entries. It
does not touch citation_audit_decisions.json or replace the option to
review manually; it just means a flagged post doesn't sit fully live and
unresolved indefinitely if nobody runs `review` by hand.

HOW IT REMOVES CONTENT
-----------------------
Reuses strip_fabricated_content() from process_regeneration_queue.py
UNCHANGED — no reimplementation, no drift. That function already strips
both citation-pattern spans (_FABRICATED_CITATION_PATTERNS, checked
against the same URL/_LEGITIMATE_SOURCE_CONTEXT allow-lists
_reject_if_fabricated_citation() uses) and fabricated-anecdote sentences
(_SKIP_PATTERNS) in one pass — a superset of what this queue needs, which
is fine: if a citation-flagged post also happens to contain an anecdote,
cleaning both in one edit is strictly better than two separate partial
edits touching the same post.json.

For each queued post:
  1. Run strip_fabricated_content() on post.content.
  2. Re-run _reject_if_fabricated_citation() and _flag_fabricated_anecdotes()
     on the RESULT. Only write if the result now comes back clean on
     BOTH gates — same "don't ship a partial fix" rule
     process_regeneration_queue.py already applies.
  3. If not clean, leave post.json untouched and mark the entry
     "needs_review" so a human can use adsense_credibility_cleanup.py's
     `review` mode for that specific post instead.

Unlike regeneration_queue.json, citation_audit_report.json entries don't
carry a `status` field (adsense_credibility_cleanup.py's `scan` mode
just overwrites the whole file each run). This script adds one
("pending" | "stripped" | "needs_review" | "resolved_elsewhere") and
persists it back to the same file, exactly like the status field
process_regeneration_queue.py already relies on for regeneration_queue.json.
If citation_audit_report.json is ever regenerated from scratch, statuses
reset with it — that's a re-scan starting clean, not a bug.

SAFETY MODEL
------------
Identical to process_regeneration_queue.py:
  - --max-posts bounds how many posts get edited per run.
  - Every post.json is copied to --backup-dir before being modified.
  - Without --confirm, this only reports what it would change.
  - A post this can't fully clean is left alone and flagged, not forced.

USAGE
-----
    # Dry run:
    python scripts/process_citation_queue.py --docs-dir ./docs --queue ./citation_audit_report.json

    # Apply, bounded:
    python scripts/process_citation_queue.py \\
        --docs-dir ./docs \\
        --queue ./citation_audit_report.json \\
        --max-posts 5 \\
        --backup-dir ./.citation_backups \\
        --confirm
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Running as `python scripts/process_citation_queue.py` from the repo root
# puts scripts/ on sys.path[0], not the repo root, so blog_system (a
# top-level module) fails to import. Insert both the repo root (for
# blog_system) and this file's own directory (for the sibling
# process_regeneration_queue module) explicitly — same fix applied to
# scripts/delete_confirmed_stubs.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from blog_system import (  # noqa: E402
    _reject_if_fabricated_citation,
    _flag_fabricated_anecdotes,
)
from process_regeneration_queue import (  # noqa: E402
    strip_fabricated_content,
    _backup_post_json,
)

_PENDING_STATUSES = {None, "pending"}


def run(
    docs_dir: Path,
    queue_path: Path,
    max_posts: int,
    backup_dir: Path,
    confirm: bool,
) -> int:
    if not queue_path.exists():
        print(f"{queue_path} not found — nothing to process.")
        return 0

    try:
        queue = json.loads(queue_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"⚠️  {queue_path} is not valid JSON — aborting without changes.")
        return 1

    pending = [
        e for e in queue
        if isinstance(e, dict) and e.get("status") in _PENDING_STATUSES
    ]

    if not pending:
        print("No pending citation-audit entries in the queue. Nothing to do.")
        return 0

    batch = pending[:max_posts]
    deferred = pending[max_posts:]

    print(f"{len(pending)} pending citation entr{'y' if len(pending) == 1 else 'ies'} in queue.")
    print(f"{'Processing' if confirm else 'Would process'} {len(batch)} this run "
          f"(--max-posts {max_posts}):\n")

    if not confirm:
        for e in batch:
            print(f"  {e['slug']}  | {e.get('title', '')}  "
                  f"({len(e.get('matches', []))} flagged sentence(s))")
        if deferred:
            print(f"\n{len(deferred)} more deferred to a future run: "
                  f"{', '.join(d['slug'] for d in deferred)}")
        print("\nDry run — nothing was written. Re-run with --confirm to apply.")
        return 0

    backup_dir.mkdir(parents=True, exist_ok=True)
    cleaned, needs_review = 0, 0

    for entry in batch:
        slug = entry["slug"]
        post_json = docs_dir / slug / "post.json"
        if not post_json.exists():
            print(f"  ⚠️  {slug}: post.json missing (likely retired by another "
                  f"script this run) — marking resolved.")
            entry["status"] = "resolved_elsewhere"
            continue

        try:
            data = json.loads(post_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"  ⚠️  {slug}: post.json is not valid JSON — skipping, left in queue.")
            continue

        if data.get("redirect_to"):
            print(f"  ⏭️  {slug}: already a redirect stub (see auto_retire_duplicates.py) "
                  f"— nothing to fix, marking resolved.")
            entry["status"] = "resolved_elsewhere"
            continue

        content = data.get("content", "")
        new_content, removed_count = strip_fabricated_content(content)

        still_flagged = (
            _reject_if_fabricated_citation(new_content) is not None
            or bool(_flag_fabricated_anecdotes(new_content))
        )

        if still_flagged:
            print(f"  ⚠️  {slug}: removed {removed_count} sentence(s) but the gate "
                  f"still flags the result — leaving post.json untouched, "
                  f"marking for manual review (see adsense_credibility_cleanup.py "
                  f"`review` mode for this specific post).")
            entry["status"] = "needs_review"
            needs_review += 1
            continue

        backup_path = _backup_post_json(post_json, backup_dir, slug)
        data["content"] = new_content
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        post_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"  ✅ {slug}: removed {removed_count} sentence(s), gate now clean. "
              f"Backup at {backup_path}")
        entry["status"] = "stripped"
        cleaned += 1

    queue_path.write_text(json.dumps(queue, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{cleaned} post(s) cleaned, {needs_review} flagged for manual review, "
          f"{len(deferred)} deferred.")
    if cleaned:
        print("   Run 'python blog_system.py build' to regenerate the affected "
              "index.html files before the sitemap and OG steps.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bounded, automated consumer for citation_audit_report.json."
    )
    parser.add_argument("--docs-dir", default="./docs")
    parser.add_argument("--queue", default="./citation_audit_report.json")
    parser.add_argument("--max-posts", type=int, default=5)
    parser.add_argument("--backup-dir", default="./.citation_backups")
    parser.add_argument("--confirm", action="store_true", help="Actually write changes. Omit for a dry run.")
    args = parser.parse_args()

    sys.exit(run(
        docs_dir=Path(args.docs_dir),
        queue_path=Path(args.queue),
        max_posts=args.max_posts,
        backup_dir=Path(args.backup_dir),
        confirm=args.confirm,
    ))