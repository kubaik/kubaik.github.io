"""
deduplicate_posts.py
--------------------
Finds near-duplicate blog posts and deletes the oldest one.

Similarity is measured by comparing post titles using a simple
word-overlap (Jaccard) score. Posts whose titles share >= THRESHOLD
of their words are considered duplicates.

Usage:
    python deduplicate_posts.py            # dry-run, shows what would be deleted
    python deduplicate_posts.py --delete   # actually deletes duplicates + rebuilds site
    python deduplicate_posts.py --threshold 0.4  # adjust sensitivity (default 0.5)
"""

import json
import shutil
import argparse
import yaml
from pathlib import Path
from datetime import datetime


# ── Configuration ─────────────────────────────────────────────────────────────

DOCS_DIR        = Path("./docs")
DEFAULT_THRESHOLD = 0.5   # 0.0 = any overlap, 1.0 = identical titles only


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_posts() -> list[dict]:
    """Load all post.json files from the docs directory."""
    posts = []
    if not DOCS_DIR.exists():
        print(f"ERROR: docs directory not found at {DOCS_DIR.resolve()}")
        return posts

    for post_dir in sorted(DOCS_DIR.iterdir()):
        if not post_dir.is_dir() or post_dir.name == "static":
            continue
        post_json = post_dir / "post.json"
        if not post_json.exists():
            continue
        try:
            with open(post_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["_dir"] = post_dir          # attach directory path
            posts.append(data)
        except Exception as e:
            print(f"  Warning: could not read {post_json}: {e}")

    posts.sort(key=lambda p: p.get("created_at", ""), reverse=True)
    print(f"Loaded {len(posts)} posts.")
    return posts


def tokenise(title: str) -> set[str]:
    """Lowercase, strip punctuation, split into words. Filter short stop-words."""
    import re
    STOP = {"a", "an", "the", "to", "in", "of", "for", "and", "or",
            "is", "are", "with", "how", "your", "my", "our", "its",
            "on", "at", "by", "from", "this", "that", "best", "using"}
    words = re.sub(r"[^\w\s]", "", title.lower()).split()
    return {w for w in words if w not in STOP and len(w) > 2}


def jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two sets."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def find_duplicates(posts: list[dict], threshold: float) -> list[tuple[dict, dict]]:
    """
    Return pairs (keep, delete) where 'delete' is the older duplicate.
    We keep the NEWER post (higher created_at).
    """
    pairs = []
    seen  = set()

    for i, post_a in enumerate(posts):
        for j, post_b in enumerate(posts):
            if j <= i:
                continue
            key = (i, j)
            if key in seen:
                continue

            tokens_a = tokenise(post_a.get("title", ""))
            tokens_b = tokenise(post_b.get("title", ""))
            score    = jaccard(tokens_a, tokens_b)

            if score >= threshold:
                seen.add(key)
                # posts are sorted newest-first, so post_a is newer
                pairs.append((post_a, post_b, score))

    return pairs


def format_date(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return iso


def delete_post(post: dict):
    """Delete the post directory and all its contents."""
    post_dir = post["_dir"]
    if post_dir.exists():
        shutil.rmtree(post_dir)
        print(f"  🗑  Deleted: {post_dir.name}  ({post.get('title', '?')})")
    else:
        print(f"  Warning: directory not found: {post_dir}")


def rebuild_site():
    """Rebuild the static site after deletions."""
    import sys
    import os

    print("\nRebuilding site...")
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        os.chdir(Path(__file__).parent)

        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)

        from blog_system          import BlogSystem
        from static_site_generator import StaticSiteGenerator

        blog_system = BlogSystem(config)
        generator   = StaticSiteGenerator(blog_system)
        generator.generate_site()
        print("Site rebuilt successfully.")
    except Exception as e:
        print(f"Warning: site rebuild failed: {e}")
        print("Run  python blog_system.py build  manually to rebuild.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Find and remove near-duplicate blog posts.")
    parser.add_argument("--delete",    action="store_true",
                        help="Actually delete duplicates (default is dry-run only).")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Similarity threshold 0–1 (default {DEFAULT_THRESHOLD}). "
                             "Lower = more aggressive deduplication.")
    parser.add_argument("--no-rebuild", action="store_true",
                        help="Skip site rebuild after deletion.")
    args = parser.parse_args()

    print("=" * 60)
    print("  Blog Post Deduplication Tool")
    print(f"  Threshold : {args.threshold}")
    print(f"  Mode      : {'DELETE' if args.delete else 'DRY-RUN (pass --delete to apply)'}")
    print("=" * 60)

    posts = load_posts()
    if not posts:
        print("No posts found. Exiting.")
        return

    duplicates = find_duplicates(posts, args.threshold)

    if not duplicates:
        print("\n✅  No duplicate posts found.")
        return

    print(f"\nFound {len(duplicates)} duplicate pair(s):\n")
    deleted_slugs = set()

    for keep, delete, score in duplicates:
        print(f"  Similarity : {score:.0%}")
        print(f"  KEEP   [{format_date(keep.get('created_at',''))}]  {keep.get('title','?')}")
        print(f"  DELETE [{format_date(delete.get('created_at',''))}]  {delete.get('title','?')}")
        print()

        if args.delete:
            slug = delete.get("slug", "")
            if slug and slug not in deleted_slugs:
                delete_post(delete)
                deleted_slugs.add(slug)
            elif slug in deleted_slugs:
                print(f"  (already deleted in this run: {slug})")

    if args.delete:
        print(f"\nDeleted {len(deleted_slugs)} post(s).")
        if not args.no_rebuild:
            rebuild_site()
    else:
        print("Dry-run complete. No files were changed.")
        print("Run with --delete to apply changes.")


if __name__ == "__main__":
    main()