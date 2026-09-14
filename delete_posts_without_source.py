#!/usr/bin/env python3
"""
delete_posts_without_source.py

Finds published posts under docs/ that have no post.json (the pipeline's
source of truth — see blog_system.py's own error: "A post without a source
of truth cannot be regenerated, fact-checked, or corrected — this must not
ship.") and deletes them.

By default this runs in DRY RUN mode and only prints what it would do.
Pass --execute to actually delete.

Usage:
    python3 delete_posts_without_source.py                # dry run (safe)
    python3 delete_posts_without_source.py --execute       # actually delete
    python3 delete_posts_without_source.py --docs-dir ./docs --execute
    python3 delete_posts_without_source.py --keep-markdown --execute
        # if a post has index.md but no post.json, don't delete it —
        # just report it, since blog_system.py's `cleanup_posts()` command
        # can recover those automatically instead of losing them.

Behavior:
    - Walks each subdirectory of docs/, skipping known non-post site
      directories (static, about, author, contact, dmca, privacy-policy,
      terms-of-service, ai-content-policy, tag, page) — those are generated
      directly by static_site_generator.py and never have a post.json by
      design, so they are not "missing a source" in any meaningful sense.
    - A post dir is "missing source" if post.json does not exist in it.
    - With --keep-markdown (recommended), a dir that has index.md but no
      post.json is left alone and just reported, since running
      `python3 blog_system.py cleanup` can regenerate post.json from that
      markdown without losing the post. Only dirs with NEITHER post.json
      NOR index.md are treated as truly unrecoverable.
    - Without --keep-markdown, ANY dir missing post.json is deleted,
      matching the literal ask ("delete posts without post.json").
    - Writes a JSON log of everything removed to docs/_deleted_no_source.json
      so the removal is auditable later.
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Directories static_site_generator.py writes directly (about/, contact/,
# dmca/, etc. — see _generate_static_pages(), _generate_author_page(),
# _generate_dmca_page(), _generate_ai_content_policy_page(), and the
# paginated /page/N/ archive and /tag/<slug>/ archive pages). These are
# real site pages, NOT blog posts, and were never written with a
# post.json in the first place — they must never be treated as
# "missing their source of truth."
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


def find_posts_without_source(docs_dir: Path, keep_markdown: bool) -> dict:
    """Returns {"delete": [...], "recoverable": [...]} slugs."""
    result = {"delete": [], "recoverable": []}

    if not docs_dir.exists():
        print(f"Docs directory not found: {docs_dir}")
        return result

    for post_dir in sorted(docs_dir.iterdir()):
        if not post_dir.is_dir() or post_dir.name in NON_POST_DIR_NAMES:
            continue

        post_json = post_dir / "post.json"
        if post_json.exists():
            continue  # has a source of truth — leave it alone

        markdown_path = post_dir / "index.md"
        if keep_markdown and markdown_path.exists():
            result["recoverable"].append(post_dir.name)
            continue

        result["delete"].append(post_dir.name)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Delete published posts that have no post.json source of truth."
    )
    parser.add_argument(
        "--docs-dir", default="./docs",
        help="Path to the docs/ output directory (default: ./docs)",
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Actually delete. Without this flag, only a dry-run report is printed.",
    )
    parser.add_argument(
        "--keep-markdown", action="store_true",
        help="Don't delete posts that still have index.md — those can be "
             "recovered instead via 'python3 blog_system.py cleanup'.",
    )
    args = parser.parse_args()

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
            print(f"  ~ {slug}  (run 'python3 blog_system.py cleanup' to restore post.json)")

    if not to_delete:
        print("\nNothing to delete.")
        if recoverable:
            print(
                "\nAll affected posts are recoverable — none will be deleted "
                "because --keep-markdown was set."
            )
        sys.exit(0)

    print(f"\nTo be deleted: {len(to_delete)}")
    for slug in to_delete:
        print(f"  ✗ {slug}")

    if not args.execute:
        print("\n" + "=" * 70)
        print(f"DRY RUN — nothing was deleted. Re-run with --execute to remove "
              f"{len(to_delete)} post(s).")
        print("=" * 70)
        sys.exit(0)

    log_path = docs_dir / "_deleted_no_source.json"
    removed_log = json.loads(log_path.read_text()) if log_path.exists() else {}

    deleted_count = 0
    for slug in to_delete:
        post_dir = docs_dir / slug
        try:
            shutil.rmtree(post_dir)
            removed_log[slug] = {
                "deleted_at": datetime.now().isoformat(),
                "reason": "missing post.json (no source of truth)",
            }
            deleted_count += 1
            print(f"Deleted: {slug}")
        except OSError as e:
            print(f"Failed to delete {slug}: {e}")

    log_path.write_text(json.dumps(removed_log, indent=2))

    print("\n" + "=" * 70)
    print(f"Deleted {deleted_count}/{len(to_delete)} post(s). Log: {log_path}")
    print("Run 'python3 blog_system.py build' to regenerate the site "
          "(sitemap, index pages, etc.) so it no longer references them.")
    print("=" * 70)


if __name__ == "__main__":
    main()