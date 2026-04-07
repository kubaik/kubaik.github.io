"""
delete_fallback_posts.py

Detects and deletes blog posts generated from the local fallback template
in blog_system.py (_generate_fallback_post).

Usage:
    python delete_fallback_posts.py          # dry run — shows what would be deleted
    python delete_fallback_posts.py --delete # actually deletes the posts
"""

import json
import shutil
import sys
from pathlib import Path


DOCS_DIR = Path("./docs")

# ── Fingerprints of fallback-generated posts ──────────────────────────────────
# These match the exact strings hardcoded in _generate_fallback_post()

FALLBACK_TITLE_PREFIX  = "Understanding "
FALLBACK_TITLE_SUFFIX  = ": A Complete Guide"
FALLBACK_CONTENT_PROBE = "is a crucial aspect of modern technology"
FALLBACK_TAGS          = {"technology", "development", "guide"}
FALLBACK_SEO_KEYWORDS  = {"guide", "tutorial", "best practices"}


def is_fallback_post(data: dict) -> tuple[bool, list[str]]:
    """
    Returns (is_fallback, reasons) where reasons lists which fingerprints matched.
    A post is considered a fallback if it matches ANY 2+ fingerprints (robust against
    minor edits to the template).
    """
    reasons = []

    title = data.get("title", "")
    if title.startswith(FALLBACK_TITLE_PREFIX) and title.endswith(FALLBACK_TITLE_SUFFIX):
        reasons.append(f'Title matches pattern: "{title}"')

    content = data.get("content", "")
    if FALLBACK_CONTENT_PROBE in content:
        reasons.append(f'Content contains fallback probe text: "{FALLBACK_CONTENT_PROBE}"')

    tags = set(data.get("tags", []))
    overlap = tags & FALLBACK_TAGS
    if len(overlap) >= 2:
        reasons.append(f"Tags overlap with fallback tags: {overlap}")

    seo = set(data.get("seo_keywords", []))
    seo_overlap = seo & FALLBACK_SEO_KEYWORDS
    if len(seo_overlap) >= 2:
        reasons.append(f"SEO keywords overlap with fallback keywords: {seo_overlap}")

    # Needs at least 2 signals to be flagged (avoids false positives)
    return len(reasons) >= 2, reasons


def scan_docs(dry_run: bool = True):
    if not DOCS_DIR.exists():
        print(f"❌ Directory not found: {DOCS_DIR.resolve()}")
        sys.exit(1)

    fallback_dirs  = []
    skipped        = []
    error_count    = 0

    for post_dir in sorted(DOCS_DIR.iterdir()):
        if not post_dir.is_dir() or post_dir.name == "static":
            continue

        post_json = post_dir / "post.json"
        if not post_json.exists():
            skipped.append(post_dir.name)
            continue

        try:
            with open(post_json, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  ⚠️  Could not read {post_dir.name}/post.json: {e}")
            error_count += 1
            continue

        flagged, reasons = is_fallback_post(data)
        if flagged:
            fallback_dirs.append((post_dir, data.get("title", "???"), reasons))

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"  {'DRY RUN — ' if dry_run else ''}Fallback Post Scanner")
    print(f"{'=' * 65}")
    print(f"  Docs directory : {DOCS_DIR.resolve()}")
    print(f"  Posts scanned  : {sum(1 for d in DOCS_DIR.iterdir() if d.is_dir() and d.name != 'static')}")
    print(f"  Fallback posts : {len(fallback_dirs)}")
    print(f"  Skipped (no JSON): {len(skipped)}")
    print(f"  Read errors    : {error_count}")
    print(f"{'=' * 65}\n")

    if not fallback_dirs:
        print("✅ No fallback posts detected.")
        return

    print(f"{'🔍 WOULD DELETE' if dry_run else '🗑️  DELETING'} {len(fallback_dirs)} fallback post(s):\n")

    deleted = 0
    for post_dir, title, reasons in fallback_dirs:
        print(f"  📁 {post_dir.name}")
        print(f"     Title   : {title}")
        print(f"     Signals : {len(reasons)}")
        for r in reasons:
            print(f"       • {r}")

        if not dry_run:
            try:
                shutil.rmtree(post_dir)
                print(f"     ✅ Deleted")
                deleted += 1
            except OSError as e:
                print(f"     ❌ Failed to delete: {e}")
        print()

    print(f"{'=' * 65}")
    if dry_run:
        print(f"  DRY RUN complete. {len(fallback_dirs)} post(s) would be deleted.")
        print(f"  Run with --delete to actually remove them.")
    else:
        print(f"  Done. {deleted}/{len(fallback_dirs)} post(s) deleted.")
        print(f"  Run 'python blog_system.py build' to rebuild the site.")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    dry_run = "--delete" not in sys.argv
    if not dry_run:
        confirm = input(
            "⚠️  This will permanently delete matched post directories.\n"
            "    Type 'yes' to confirm: "
        ).strip().lower()
        if confirm != "yes":
            print("Aborted.")
            sys.exit(0)

    scan_docs(dry_run=dry_run)