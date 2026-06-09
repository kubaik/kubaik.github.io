#!/usr/bin/env python3
"""
delete_posts_by_date.py
=======================
Delete blog posts from docs/ by date range.

Reads created_at from each post.json and deletes the directory if the
post falls within the specified --from and --to dates (both inclusive).

Always shows a dry-run preview first. Actual deletion only happens
after confirmation (or when --yes is passed).

Usage
-----
  # Preview posts between two dates (dry run, no deletion)
  python delete_posts_by_date.py --from 2025-01-01 --to 2025-03-31

  # Delete posts in range after confirmation prompt
  python delete_posts_by_date.py --from 2025-01-01 --to 2025-03-31 --delete

  # Delete without confirmation prompt
  python delete_posts_by_date.py --from 2025-01-01 --to 2025-03-31 --delete --yes

  # Delete a single day
  python delete_posts_by_date.py --from 2025-06-15 --to 2025-06-15 --delete

  # Delete everything before a date
  python delete_posts_by_date.py --from 2000-01-01 --to 2025-12-31 --delete

  # Delete everything from a date onwards
  python delete_posts_by_date.py --from 2026-01-01 --to 2099-12-31 --delete

  # Point at a non-default docs directory
  python delete_posts_by_date.py --from 2025-01-01 --to 2025-06-30 --docs /path/to/docs --delete

  # List all posts with their dates (no deletion)
  python delete_posts_by_date.py --list
"""

import argparse
import json
import shutil
import sys
from datetime import date, datetime
from pathlib import Path
from typing import List, Tuple


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_date_arg(value: str, label: str) -> date:
    """Parse a YYYY-MM-DD string into a date, exit on failure."""
    try:
        return datetime.strptime(value.strip(), "%Y-%m-%d").date()
    except ValueError:
        print(f"Error: --{label} must be in YYYY-MM-DD format, got: {value!r}")
        sys.exit(1)


def _extract_date(raw: str) -> date:
    """
    Extract a date from a created_at string.
    Handles ISO 8601 with or without time component.
    Returns date(1970, 1, 1) on failure so the post is never falsely matched.
    """
    if not raw:
        return date(1970, 1, 1)
    try:
        # Strip timezone offset or Z, take date part only
        clean = raw.strip().replace("Z", "+00:00")
        if "T" in clean:
            clean = clean.split("T")[0]
        return datetime.strptime(clean[:10], "%Y-%m-%d").date()
    except Exception:
        return date(1970, 1, 1)


def _word_count(content: str) -> int:
    return len(content.split())


# ── Post loader ───────────────────────────────────────────────────────────────

def load_all_posts(docs_dir: Path) -> List[Tuple[str, dict, date]]:
    """
    Return [(slug, post_data, created_date), ...] sorted oldest-first.
    Skips non-post directories (static, tag, author, etc.).
    """
    results = []
    skip_names = {"static", "tag", "author", "dmca", "ai-content-policy",
                  "about", "contact", "privacy-policy", "terms-of-service"}

    for post_dir in docs_dir.iterdir():
        if not post_dir.is_dir() or post_dir.name in skip_names:
            continue
        post_json = post_dir / "post.json"
        if not post_json.exists():
            continue
        try:
            with open(post_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            created = _extract_date(data.get("created_at", ""))
            results.append((post_dir.name, data, created))
        except Exception as e:
            print(f"  ⚠️  Could not read {post_dir.name}/post.json: {e}")

    results.sort(key=lambda x: x[2])
    return results


# ── Listing ───────────────────────────────────────────────────────────────────

def list_all_posts(docs_dir: Path) -> None:
    posts = load_all_posts(docs_dir)
    if not posts:
        print("No posts found.")
        return

    SEP = "─" * 74
    print(f"\n{SEP}")
    print(f"All posts in {docs_dir}  ({len(posts)} total, oldest first)")
    print(SEP)

    for slug, data, created in posts:
        title = data.get("title", slug)[:60]
        wc = _word_count(data.get("content", ""))
        print(f"  {created}   {title}")
        print(f"             slug: {slug}   words: {wc:,}")

    print(SEP + "\n")


# ── Range filter ──────────────────────────────────────────────────────────────

def filter_by_range(
    posts: List[Tuple[str, dict, date]],
    from_date: date,
    to_date: date,
) -> List[Tuple[str, dict, date]]:
    return [
        (slug, data, created)
        for slug, data, created in posts
        if from_date <= created <= to_date
    ]


# ── Report ────────────────────────────────────────────────────────────────────

def print_range_report(
    matched: List[Tuple[str, dict, date]],
    all_posts: List[Tuple[str, dict, date]],
    from_date: date,
    to_date: date,
    dry_run: bool,
) -> None:
    SEP = "─" * 74
    label = "DRY RUN — " if dry_run else ""

    print(f"\n{SEP}")
    print(
        f"{label}Posts from {from_date} to {to_date}  "
        f"({len(matched)} matched  /  {len(all_posts)} total)"
    )
    print(SEP)

    if not matched:
        print("  No posts fall within this date range.")
        print(SEP + "\n")
        return

    total_words = sum(_word_count(d.get("content", "")) for _, d, _ in matched)

    for slug, data, created in matched:
        title = data.get("title", slug)[:60]
        wc = _word_count(data.get("content", ""))
        print(f"  🗑  {created}   {title}")
        print(f"               slug : {slug}")
        print(f"               words: {wc:,}")

    print(SEP)
    print(
        f"  Total posts to delete : {len(matched)}")
    print(
        f"  Total words removed   : {total_words:,}")
    print(
        f"  Posts remaining after : {len(all_posts) - len(matched)}")
    print(SEP + "\n")


# ── Deletion ──────────────────────────────────────────────────────────────────

def delete_posts(
    matched: List[Tuple[str, dict, date]],
    docs_dir: Path,
) -> int:
    deleted = 0
    for slug, _, _ in matched:
        post_dir = docs_dir / slug
        if post_dir.exists():
            shutil.rmtree(post_dir)
            print(f"  Deleted: {post_dir}")
            deleted += 1
        else:
            print(f"  ⚠️  Already gone: {post_dir}")
    return deleted


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Delete blog posts from docs/ by date range.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--docs", default="./docs",
        help="Path to docs directory (default: ./docs)",
    )
    parser.add_argument(
        "--from", dest="from_date", metavar="YYYY-MM-DD", default=None,
        help="Start of date range (inclusive). Required unless --list is used.",
    )
    parser.add_argument(
        "--to", dest="to_date", metavar="YYYY-MM-DD", default=None,
        help="End of date range (inclusive). Required unless --list is used.",
    )
    parser.add_argument(
        "--delete", action="store_true",
        help="Actually delete the matched posts (default is preview only).",
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="Skip the confirmation prompt when --delete is used.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all posts with their dates and exit (no deletion).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    docs_dir = Path(args.docs).resolve()

    if not docs_dir.exists():
        print(f"Error: docs directory not found: {docs_dir}")
        sys.exit(1)

    # ── List mode ─────────────────────────────────────────────────────────────
    if args.list:
        list_all_posts(docs_dir)
        sys.exit(0)

    # ── Validate date args ────────────────────────────────────────────────────
    if not args.from_date or not args.to_date:
        print("Error: --from and --to are required (or use --list to browse all posts).")
        print("  Example: python delete_posts_by_date.py --from 2025-01-01 --to 2025-06-30")
        sys.exit(1)

    from_date = _parse_date_arg(args.from_date, "from")
    to_date = _parse_date_arg(args.to_date,   "to")

    if from_date > to_date:
        print(
            f"Error: --from ({from_date}) must be on or before --to ({to_date})."
        )
        sys.exit(1)

    print(f"Docs dir  : {docs_dir}")
    print(f"Date range: {from_date}  →  {to_date}")

    # ── Load & filter ─────────────────────────────────────────────────────────
    all_posts = load_all_posts(docs_dir)
    if not all_posts:
        print(f"\nNo posts found in {docs_dir}")
        sys.exit(0)

    matched = filter_by_range(all_posts, from_date, to_date)

    # ── Always show the preview report ───────────────────────────────────────
    print_range_report(
        matched, all_posts, from_date, to_date,
        dry_run=not args.delete,
    )

    if not matched:
        sys.exit(0)

    # ── Preview-only mode (no --delete flag) ──────────────────────────────────
    if not args.delete:
        print("  This is a preview. Pass --delete to remove these posts.")
        print("  Example:")
        print(
            f"    python delete_posts_by_date.py "
            f"--from {from_date} --to {to_date} --delete\n"
        )
        sys.exit(0)

    # ── Deletion mode ─────────────────────────────────────────────────────────
    if not args.yes:
        answer = input(
            f"  Permanently delete {len(matched)} post(s)? [y/N] "
        ).strip().lower()
        if answer not in ("y", "yes"):
            print("  Aborted. Nothing was deleted.\n")
            sys.exit(0)

    print()
    deleted = delete_posts(matched, docs_dir)

    SEP = "─" * 74
    print(f"\n{SEP}")
    print(
        f"  Done. Deleted {deleted} post director{'y' if deleted == 1 else 'ies'}.")
    print(f"  Run 'python blog_system.py build' to regenerate the site.")
    print(SEP + "\n")


if __name__ == "__main__":
    main()
