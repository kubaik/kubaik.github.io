#!/usr/bin/env python3
"""
delete_posts_without_source.py

Finds published posts under docs/ that have no post.json (the pipeline's
source of truth) and deletes them, safely.

By default this runs in DRY RUN mode and only prints what it would do.
Pass --execute to actually delete.

Usage:
    python3 delete_posts_without_source.py                       # dry run
    python3 delete_posts_without_source.py --keep-markdown       # dry run, spare recoverable posts
    python3 delete_posts_without_source.py --keep-markdown --execute
    python3 delete_posts_without_source.py --execute --block     # also block topic regeneration

For each post deleted (with --execute) the script:
  1. backs up docs/<slug>/ to <backup-dir>/<slug>/   (default .triage_backups)
  2. removes docs/<slug>/ and docs/static/og/<slug>*.png
  3. repairs links in remaining posts that pointed at it
     ([text](/slug/) -> text), in their post.json "content"
  4. logs it to docs/_deleted_no_source.json
  5. with --block, adds it to the topic blocklist via quality_gate
     so blog_system.py won't regenerate the topic

Directories with index.md but no post.json are recoverable through
`python3 blog_system.py cleanup`; --keep-markdown leaves them alone.

Afterwards run: python3 blog_system.py build
"""

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

# Directories static_site_generator.py writes directly. These are real site
# pages, not blog posts, and never have a post.json.
NON_POST_DIR_NAMES = {
    "static",
    "about",
    "author",
    "contact",
    "dmca",
    "privacy-policy",
    "terms-of-service",
    "ai-content-policy",
    "tag",
    "page",
}

# Same link shape delete_posts.py repairs: [text](/slug/) or full-URL form.
LINK_RX = re.compile(
    r"\[([^\]]+)\]\((?:https?://kubaik\.github\.io)?/([a-z0-9][a-z0-9-]*)/?\)"
)


def find_posts_without_source(docs_dir: Path, keep_markdown: bool) -> dict:
    """Returns {"delete": [...], "recoverable": [...]} slugs."""
    result = {"delete": [], "recoverable": []}

    if not docs_dir.exists():
        print(f"Docs directory not found: {docs_dir}")
        return result

    for post_dir in sorted(docs_dir.iterdir()):
        if (
            not post_dir.is_dir()
            or post_dir.name in NON_POST_DIR_NAMES
            or post_dir.name.startswith((".", "_"))
        ):
            continue

        if (post_dir / "post.json").exists():
            continue  # has a source of truth

        if keep_markdown and (post_dir / "index.md").exists():
            result["recoverable"].append(post_dir.name)
            continue

        result["delete"].append(post_dir.name)

    return result


def read_title(post_dir: Path, slug: str) -> str:
    """Best-effort title for the blocklist: index.md heading, else the slug."""
    md = post_dir / "index.md"
    if md.exists():
        try:
            for line in md.read_text("utf-8").splitlines():
                if line.startswith("# "):
                    return line[2:].strip()
        except OSError:
            pass
    return slug.replace("-", " ")


def repair_links(docs_dir: Path, dead: set, execute: bool) -> int:
    """Replace links to deleted slugs with their plain text in remaining posts."""
    fixed = 0
    for pj in docs_dir.glob("*/post.json"):
        if pj.parent.name in dead:
            continue
        try:
            data = json.loads(pj.read_text("utf-8"))
        except ValueError:
            continue

        count = 0

        def sub(m):
            nonlocal count
            if m.group(2) in dead:
                count += 1
                return m.group(1)
            return m.group(0)

        new_content = LINK_RX.sub(sub, data.get("content", ""))
        if count:
            fixed += count
            if execute:
                data["content"] = new_content
                pj.write_text(json.dumps(data, indent=2, ensure_ascii=False), "utf-8")
    return fixed


def main():
    parser = argparse.ArgumentParser(
        description="Delete published posts that have no post.json source of truth."
    )
    parser.add_argument(
        "--docs-dir",
        default="./docs",
        help="Path to the docs/ output directory (default: ./docs)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete. Without this, only a dry-run report is printed.",
    )
    parser.add_argument(
        "--keep-markdown",
        action="store_true",
        help="Don't delete posts that still have index.md; recover them via "
        "'python3 blog_system.py cleanup' instead.",
    )
    parser.add_argument(
        "--backup-dir",
        default=".triage_backups",
        help="Where to copy each post before deleting (default: .triage_backups)",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip the backup copy (not recommended).",
    )
    parser.add_argument(
        "--block",
        action="store_true",
        help="Add deleted topics to the blocklist (needs quality_gate.py) "
        "so blog_system.py won't regenerate them.",
    )
    parser.add_argument(
        "--blocklist",
        default=None,
        help="Blocklist path (default: quality_gate.BLOCKLIST_PATH)",
    )
    args = parser.parse_args()

    quality_gate = None
    if args.block:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        try:
            import quality_gate  # type: ignore
        except ImportError:
            print(
                "--block requires quality_gate.py next to this script; not found. Aborting."
            )
            sys.exit(2)

    docs_dir = Path(args.docs_dir)
    found = find_posts_without_source(docs_dir, args.keep_markdown)
    to_delete = found["delete"]
    recoverable = found["recoverable"]

    print("=" * 70)
    print("POSTS WITH NO post.json (no source of truth)")
    print("=" * 70)

    if recoverable:
        print(f"\nRecoverable (has index.md, --keep-markdown set): {len(recoverable)}")
        for slug in recoverable:
            print(f"  ~ {slug}")
        print("  -> run 'python3 blog_system.py cleanup' to restore their post.json")

    if not to_delete:
        print("\nNothing to delete.")
        sys.exit(0)

    print(f"\nTo be deleted: {len(to_delete)}")
    for slug in to_delete:
        print(f"  x {slug}")

    dead = set(to_delete)
    link_count = repair_links(docs_dir, dead, execute=False)
    og_count = (
        sum(len(list((docs_dir / "static" / "og").glob(f"{s}*"))) for s in to_delete)
        if (docs_dir / "static" / "og").exists()
        else 0
    )
    print(f"\nLinks in remaining posts that would be repaired: {link_count}")
    print(f"OG images that would be removed: {og_count}")

    if not args.execute:
        print("\n" + "=" * 70)
        print(
            f"DRY RUN: nothing was deleted. Re-run with --execute to remove "
            f"{len(to_delete)} post(s)."
        )
        if not args.no_backup:
            print(f"Each will be backed up to {args.backup_dir}/<slug>/ first.")
        print("=" * 70)
        sys.exit(0)

    log_path = docs_dir / "_deleted_no_source.json"
    removed_log = json.loads(log_path.read_text()) if log_path.exists() else {}
    now = datetime.now(timezone.utc).isoformat()

    deleted = []
    block_entries = []
    for slug in to_delete:
        post_dir = docs_dir / slug
        title = read_title(post_dir, slug)
        try:
            if not args.no_backup:
                shutil.copytree(
                    post_dir, Path(args.backup_dir) / slug, dirs_exist_ok=True
                )
            shutil.rmtree(post_dir)
            for og in (docs_dir / "static" / "og").glob(f"{slug}*"):
                og.unlink(missing_ok=True)
        except OSError as e:
            print(f"Failed to delete {slug}: {e}")
            continue

        removed_log[slug] = {
            "deleted_at": now,
            "reason": "missing post.json (no source of truth)",
            "source": "delete_posts_without_source.py",
        }
        block_entries.append(
            {
                "slug": slug,
                "title": title,
                "reason": "missing post.json (no source of truth)",
                "deleted_at": now,
            }
        )
        deleted.append(slug)
        print(f"Deleted: {slug}")

    # Only repair links to posts that were actually removed.
    fixed = repair_links(docs_dir, set(deleted), execute=True)

    log_path.write_text(json.dumps(removed_log, indent=2, ensure_ascii=False))

    if args.block and block_entries:
        blocklist = (
            Path(args.blocklist)
            if args.blocklist
            else Path(quality_gate.BLOCKLIST_PATH)
        )
        added = quality_gate.add_to_blocklist(block_entries, blocklist)
        print(f"blocklist: +{added} entries -> {blocklist}")

    print("\n" + "=" * 70)
    print(f"Deleted {len(deleted)}/{len(to_delete)} post(s). Log: {log_path}")
    if not args.no_backup:
        print(f"Backups: {args.backup_dir}/")
    print(f"Links repaired in remaining posts: {fixed}")
    print("Next: python3 blog_system.py build  (regenerates sitemap, RSS, index pages)")
    print("=" * 70)


if __name__ == "__main__":
    main()
