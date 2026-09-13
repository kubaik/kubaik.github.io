#!/usr/bin/env python3
"""
scripts/auto_retire_duplicates.py

Bounded, automated cleanup for the duplicate-topic clusters that
adsense_fixes/topic_dedup.py already detects (find_clusters()).

WHY THIS EXISTS
---------------
topic_dedup.check_topic_duplicate() stops NEW duplicates at generation
time. It does nothing about the clusters that already exist in the
corpus (17 pairs >= 45% on the live site, per topic_dedup.py's own
docstring). find_clusters() can report those clusters, but nothing
acts on them — someone has to read the CLI output and delete posts by
hand.

This script closes that loop WITHOUT a full --confirm-everywhere
free-for-all. It applies the same "bounded blast radius" pattern used
elsewhere in this repo (VelocityController capping posts/day,
quality_gate.py capping what it queues):

  --max-deletions N   caps how many posts this script can remove in a
                       single run, no matter how many clusters exist.
  --min-score S        only acts on cluster pairs scoring >= S (i.e. the
                       highest-confidence half of the corpus, per the
                       17-pairs-at-45% baseline). Lower-confidence pairs
                       are left in the report for a human to glance at.
  --confirm             without this flag, the script only prints what
                       it WOULD do. Nothing is deleted or written.
  backup-then-delete    every retired post's full directory is copied to
                       --backup-dir before anything is removed, so a bad
                       call is always reversible.
  redirect stub          the retired post's URL is not simply 404'd. Its
                       directory is replaced with a small noindex,
                       meta-refresh HTML stub pointing at the keeper
                       post, so:
                         - any inbound/backlinks to the old URL don't 404
                         - scripts/generate_sitemap.py already excludes
                           noindex + meta-refresh pages, so the retired
                           URL drops out of the sitemap on the next build
                           with no extra wiring
                         - adsense_fixes/link_validator.py will no longer
                           see the slug as "published" (no post.json), so
                           any internal links still pointing at it get
                           stripped/reported the same way a genuinely
                           deleted post already would be

Within a cluster, topic_dedup.find_clusters() already sorts by word
count descending (longest = keep). This script always keeps c[0] and
only ever considers retiring c[1:].

WHAT THIS DOES NOT DO
----------------------
- It does not touch clusters or pairs below --min-score. Those stay in
  the printed report for manual review, same as before.
- It does not modify the keeper post at all.
- It does not rebuild the site (sitemap, homepage, posts.json, RSS).
  Run scripts/rebuild_site.py or `python blog_system.py build`
  afterward — the existing blog-automation.yml workflow already does
  this on every scheduled run via the sitemap/OG-image/build steps.

USAGE
-----
    # Dry run — see exactly what would be retired, nothing touched
    python scripts/auto_retire_duplicates.py --docs-dir ./docs

    # Actually retire, capped at 2 posts this run
    python scripts/auto_retire_duplicates.py \\
        --docs-dir ./docs \\
        --max-deletions 2 \\
        --min-score 0.50 \\
        --backup-dir ./.duplicate_backups \\
        --confirm

Recommended CI wiring (bounded — see module docstring for why the cap
matters here):

    - name: 🧹 Auto-retire confirmed duplicate posts (bounded)
      run: |
        python scripts/auto_retire_duplicates.py \\
          --docs-dir ./docs \\
          --max-deletions 2 \\
          --min-score 0.50 \\
          --backup-dir ./.duplicate_backups \\
          --confirm
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# adsense_fixes/ lives at the repo root, one level up from scripts/
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adsense_fixes.topic_dedup import find_clusters, _jaccard  # noqa: E402

DEFAULT_MAX_DELETIONS = 2
DEFAULT_MIN_SCORE = 0.50
DEFAULT_BACKUP_DIR = REPO_ROOT / ".duplicate_backups"
DEFAULT_BASE_URL = "https://kubaik.github.io"
DEFAULT_REPORT_PATH = REPO_ROOT / "duplicate_retirement_report.json"

_REDIRECT_STUB_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="robots" content="noindex,follow">
<meta http-equiv="refresh" content="0; url={keeper_url}">
<link rel="canonical" href="{keeper_url}">
<title>Moved — {keeper_title}</title>
</head>
<body>
<p>This post has been merged into <a href="{keeper_url}">{keeper_title}</a>. If you are not redirected automatically, follow the link.</p>
</body>
</html>
"""


def build_candidates(
    docs_dir: Path,
    min_score: float,
) -> List[Dict]:
    """
    Run topic_dedup.find_clusters() and flatten every (keeper, duplicate)
    pair across all clusters into a single list, scored and sorted
    highest-confidence first.

    Each item: {score, keep_slug, keep_title, dup_slug, dup_title, dup_words}
    """
    clusters = find_clusters(docs_dir)
    candidates: List[Dict] = []

    for cluster in clusters:
        keep = cluster[0]  # longest word count = keep, per find_clusters()
        for dup in cluster[1:]:
            score = _jaccard(keep["keys"], dup["keys"])
            if score >= min_score:
                candidates.append({
                    "score": score,
                    "keep_slug": keep["slug"],
                    "keep_title": keep["title"],
                    "dup_slug": dup["slug"],
                    "dup_title": dup["title"],
                    "dup_words": dup["words"],
                })

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates


def backup_post_dir(post_dir: Path, backup_dir: Path) -> Path:
    """Copy a post's full directory to backup_dir/{slug}-{timestamp}/."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = backup_dir / f"{post_dir.name}-{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(post_dir, dest)
    return dest


def write_redirect_stub(post_dir: Path, keeper_slug: str, keeper_title: str, base_url: str) -> None:
    """
    Remove everything in post_dir and replace it with a single index.html
    redirect stub. Deliberately does NOT write a post.json — that's what
    makes every other script in this repo (link_validator, internal_linker,
    topic_dedup, quality_gate, generate_sitemap) treat the slug as no
    longer published, without needing any special-case code added to them.
    """
    shutil.rmtree(post_dir)
    post_dir.mkdir(parents=True, exist_ok=True)
    keeper_url = f"{base_url.rstrip('/')}/{keeper_slug}/"
    stub_html = _REDIRECT_STUB_TEMPLATE.format(
        keeper_url=keeper_url,
        keeper_title=keeper_title.replace("<", "&lt;").replace(">", "&gt;"),
    )
    (post_dir / "index.html").write_text(stub_html, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs-dir", type=str, default=str(REPO_ROOT / "docs"))
    parser.add_argument("--max-deletions", type=int, default=DEFAULT_MAX_DELETIONS,
                        help=f"Cap on how many posts this run can retire (default: {DEFAULT_MAX_DELETIONS})")
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE,
                        help=f"Only retire pairs scoring >= this (default: {DEFAULT_MIN_SCORE})")
    parser.add_argument("--backup-dir", type=str, default=str(DEFAULT_BACKUP_DIR),
                        help="Where full post directories are copied before deletion")
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL,
                        help="Site base URL, used to build the redirect stub's target URL")
    parser.add_argument("--report", type=str, default=str(DEFAULT_REPORT_PATH),
                        help="Where to write a JSON report of what happened this run")
    parser.add_argument("--confirm", action="store_true",
                        help="Actually retire posts. Without this flag, dry-run only.")
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir)
    backup_dir = Path(args.backup_dir)
    report_path = Path(args.report)

    if not docs_dir.exists():
        print(f"ERROR: {docs_dir} not found.", file=sys.stderr)
        return 1

    candidates = build_candidates(docs_dir, args.min_score)
    print(
        f"auto_retire_duplicates: {len(candidates)} duplicate pair(s) at or above "
        f"{args.min_score:.0%} (from find_clusters())."
    )

    if not candidates:
        print("Nothing to retire this run.")
        report_path.write_text(json.dumps({
            "ran_at": datetime.now(timezone.utc).isoformat(),
            "confirm": args.confirm,
            "candidates_found": 0,
            "retired": [],
        }, indent=2))
        return 0

    # Never retire the same dup_slug twice in one run, and never retire a
    # slug that is itself acting as a keeper elsewhere in this batch.
    keeper_slugs = {c["keep_slug"] for c in candidates}
    seen_dups = set()
    to_retire: List[Dict] = []
    for c in candidates:
        if len(to_retire) >= args.max_deletions:
            break
        if c["dup_slug"] in seen_dups:
            continue
        if c["dup_slug"] in keeper_slugs:
            # This slug is the keeper in another pair — never retire it
            # even if it also shows up as a "dup" against something else.
            continue
        to_retire.append(c)
        seen_dups.add(c["dup_slug"])

    skipped = len(candidates) - len(to_retire)
    print(
        f"  -> {len(to_retire)} selected for this run "
        f"(capped at --max-deletions {args.max_deletions}); "
        f"{skipped} left for a future run / manual review."
    )

    report_entries = []
    for c in to_retire:
        post_dir = docs_dir / c["dup_slug"]
        action = "RETIRE" if args.confirm else "WOULD RETIRE (dry-run)"
        print(
            f"  {action}  {c['score']:.0%}  /{c['dup_slug']}/  "
            f"({c['dup_words']}w)  -> keeper /{c['keep_slug']}/"
        )

        entry = {
            "score": round(c["score"], 3),
            "retired_slug": c["dup_slug"],
            "retired_title": c["dup_title"],
            "keeper_slug": c["keep_slug"],
            "keeper_title": c["keep_title"],
            "backup_path": None,
            "applied": False,
        }

        if not post_dir.exists():
            print(f"    ⚠️  {post_dir} does not exist on disk — skipping.")
            entry["error"] = "post_dir_missing"
            report_entries.append(entry)
            continue

        if args.confirm:
            backup_path = backup_post_dir(post_dir, backup_dir)
            write_redirect_stub(post_dir, c["keep_slug"], c["keep_title"], args.base_url)
            entry["backup_path"] = str(backup_path)
            entry["applied"] = True
            print(f"    ✅ backed up to {backup_path}, replaced with redirect stub")

        report_entries.append(entry)

    report_path.write_text(json.dumps({
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "confirm": args.confirm,
        "min_score": args.min_score,
        "max_deletions": args.max_deletions,
        "candidates_found": len(candidates),
        "candidates_skipped_this_run": skipped,
        "retired": report_entries,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport written to {report_path}")

    if not args.confirm:
        print("\nDry run only — nothing was changed. Re-run with --confirm to apply.")

    return 0


if __name__ == "__main__":
    sys.exit(main())