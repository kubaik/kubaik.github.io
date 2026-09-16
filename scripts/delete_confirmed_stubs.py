#!/usr/bin/env python3
"""
scripts/delete_confirmed_stubs.py
====================================
Deletes the merge-redirect stub posts identified in the 2026-09-15 audit
(delete_list.csv) and rebuilds the site so the homepage, pagination,
related-posts blocks, RSS feed, posts.json, and sitemap.xml all stop
referencing them.

WHY A DEDICATED SCRIPT INSTEAD OF A MANUAL `rm -rf`
-----------------------------------------------------
1. Backs up each post directory before deleting it (same convention as
   auto_retire_duplicates.py / process_regeneration_queue.py — nothing in
   this pipeline deletes without a backup).
2. Re-validates each slug against canonical_guard.is_merge_stub() right
   before deleting, rather than trusting a hardcoded list blindly — if a
   slug's content.json changed since the audit (post recovered, retitled,
   etc.), it's skipped and reported rather than deleted anyway.
3. Optionally re-discovers the CURRENT set of merge stubs from docs/ via
   --auto-detect, so this script stays useful for future cleanup passes,
   not just this one-time list.
4. Calls `python blog_system.py build` afterward so every page that
   referenced the deleted slugs (related-posts blocks, tag pages, RSS,
   posts.json, sitemap.xml) is regenerated against the new state — a bare
   `rm -rf` would leave those stale until the next scheduled run.

USAGE
-----
    # Dry run against the confirmed 2026-09-15 list (default, safe):
    python scripts/delete_confirmed_stubs.py

    # Actually delete + rebuild:
    python scripts/delete_confirmed_stubs.py --confirm

    # Re-discover merge stubs from current docs/ instead of the hardcoded
    # list (use for future cleanup passes, not this one):
    python scripts/delete_confirmed_stubs.py --auto-detect --confirm

    # Skip the rebuild step (e.g. if you're about to run
    # `python blog_system.py build` yourself as part of a larger script):
    python scripts/delete_confirmed_stubs.py --confirm --no-rebuild
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

# Running as `python scripts/delete_confirmed_stubs.py` from the repo root
# puts scripts/ on sys.path[0], not the repo root — so `adsense_fixes` and
# `blog_post` (both top-level modules) fail to import even though the repo
# is correctly checked out. Insert the repo root (this file's parent's
# parent) explicitly rather than relying on cwd or PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adsense_fixes.canonical_guard import is_merge_stub
from blog_post import BlogPost

# The 13 slugs confirmed in the 2026-09-15 audit (AUDIT_GUIDE.md /
# delete_list.csv): auto_retire_duplicates.py merge-redirect stubs that
# were self-canonical and not noindexed at audit time. Kept as the default
# so a bare `--confirm` run is deterministic and doesn't silently pick up
# whatever else looks like a stub today.
CONFIRMED_STUB_SLUGS: List[str] = [
    "4-ai-tools-that-level-the-dev-salary-gap-in-2026",
    "5k-remote-roles-nairobi-lagos-playbook",
    "ai-agents-pinching-our-own-ai-budget",
    "ai-rollouts-feature-flags-in-2026",
    "ai-wont-fix-your-legacy-code-but-my-script-did",
    "build-a-portfolio-that-beats-ai-clones-in-2026",
    "fintech-apis-broke-in-nigeria-heres-why",
    "micro-saas-niches-surviving-ai-commoditisation",
    "ownership-drift-after-velocity-spikes",
    "pick-saas-niches-ai-cant-erode",
    "purpose-built-ai-vs-general-platforms",
    "senior-devs-flee-when-systems-rot",
    "year-one-claude-codes-slow-burn-wins",
]


def _discover_current_stubs(docs_dir: Path) -> List[str]:
    """Re-scan docs/ for posts currently matching the merge-stub pattern."""
    found = []
    for post_dir in sorted(docs_dir.iterdir()):
        if not post_dir.is_dir() or post_dir.name in ("static", "tag", "author"):
            continue
        post_json = post_dir / "post.json"
        if not post_json.exists():
            continue
        try:
            import json
            data = json.loads(post_json.read_text(encoding="utf-8"))
            post = BlogPost.from_dict(data)
        except Exception as e:
            print(f"  ⚠️  Skipping {post_dir.name}: could not load post.json ({e})")
            continue
        if is_merge_stub(post):
            found.append(post_dir.name)
    return found


def run(docs_dir: Path, slugs: List[str], backup_dir: Path,
        confirm: bool, rebuild: bool) -> int:
    to_delete = []
    for slug in slugs:
        post_dir = docs_dir / slug
        if not post_dir.exists():
            print(f"  ⏭️  {slug} — no longer on disk, skipping")
            continue

        post_json = post_dir / "post.json"
        if not post_json.exists():
            print(f"  ⚠️  {slug} — no post.json found, skipping (not a normal stub)")
            continue

        import json
        try:
            data = json.loads(post_json.read_text(encoding="utf-8"))
            post = BlogPost.from_dict(data)
        except Exception as e:
            print(f"  ⚠️  {slug} — could not parse post.json ({e}), skipping")
            continue

        if not is_merge_stub(post):
            print(
                f"  ⚠️  {slug} — no longer matches the merge-stub pattern "
                f"(content may have changed since the audit) — skipping, "
                f"review manually if this is unexpected."
            )
            continue

        to_delete.append(slug)

    if not to_delete:
        print("Nothing to delete.")
        return 0

    print(f"\n{len(to_delete)} confirmed stub post(s) to delete:")
    for slug in to_delete:
        print(f"  - {slug}")

    if not confirm:
        print("\nDry run — nothing was deleted. Re-run with --confirm to apply.")
        return 0

    backup_dir.mkdir(parents=True, exist_ok=True)
    deleted = 0
    for slug in to_delete:
        post_dir = docs_dir / slug
        backup_target = backup_dir / slug
        try:
            if backup_target.exists():
                shutil.rmtree(backup_target)
            shutil.copytree(post_dir, backup_target)
            shutil.rmtree(post_dir)
            deleted += 1
            print(f"  🗑️  Deleted {slug} (backed up to {backup_target})")
        except OSError as e:
            print(f"  ❌ Failed to delete {slug}: {e}")

    print(f"\n✅ {deleted}/{len(to_delete)} post(s) deleted. Backups in {backup_dir}/")

    if rebuild:
        print("\nRebuilding site (python blog_system.py build)...")
        result = subprocess.run(
            [sys.executable, "blog_system.py", "build"],
            capture_output=False,
        )
        if result.returncode != 0:
            print("❌ Rebuild failed — check output above. "
                  "Deleted posts remain deleted; re-run `python blog_system.py build` "
                  "manually once fixed.")
            return 1
        print("✅ Site rebuilt — homepage, pagination, related posts, RSS, "
              "posts.json, and sitemap.xml no longer reference the deleted slugs.")
    else:
        print("\n--no-rebuild passed — remember to run "
              "`python blog_system.py build` before this is deployed.")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Delete confirmed merge-stub posts and rebuild the site."
    )
    parser.add_argument("--docs-dir", default="./docs")
    parser.add_argument("--backup-dir", default="./.stub_deletion_backups")
    parser.add_argument(
        "--auto-detect", action="store_true",
        help="Re-scan docs/ for current merge stubs instead of using the "
             "hardcoded 2026-09-15 confirmed list."
    )
    parser.add_argument(
        "--confirm", action="store_true",
        help="Actually delete + rebuild. Omit for a dry run."
    )
    parser.add_argument(
        "--no-rebuild", action="store_true",
        help="Skip `python blog_system.py build` after deleting."
    )
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir)
    if not docs_dir.exists():
        print(f"❌ {docs_dir} not found.")
        sys.exit(1)

    if args.auto_detect:
        print(f"Scanning {docs_dir} for current merge-stub posts...")
        slugs = _discover_current_stubs(docs_dir)
        print(f"Found {len(slugs)} stub(s) on disk right now.\n")
    else:
        slugs = CONFIRMED_STUB_SLUGS
        print(f"Using the {len(slugs)} slugs confirmed in the 2026-09-15 audit.\n")

    sys.exit(run(
        docs_dir=docs_dir,
        slugs=slugs,
        backup_dir=Path(args.backup_dir),
        confirm=args.confirm,
        rebuild=not args.no_rebuild,
    ))