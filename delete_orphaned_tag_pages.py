"""
delete_orphaned_tag_pages.py

One-time cleanup for tag pages that reference zero current posts (left
behind after posts were deleted elsewhere in the pipeline, before the fix
to _generate_tag_pages that now rebuilds docs/tag/ from scratch on every
run). After this runs once, the regenerated pipeline should keep it clean
going forward — this script is for the pages that already exist right now.

Run from repo root:
    python delete_orphaned_tag_pages.py              # dry-run
    python delete_orphaned_tag_pages.py --delete      # actually git rm
"""
import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path

DOCS_DIR = Path("./docs")
TAGS_DIR = DOCS_DIR / "tag"
EXCLUDE_DIRS = {
    "static", "tag", "author", "page",
    "contact", "about", "privacy-policy", "terms-of-service",
    "dmca", "ai-content-policy",
}


def count_live_tags() -> Counter:
    counts = Counter()
    for post_dir in DOCS_DIR.iterdir():
        if not post_dir.is_dir() or post_dir.name in EXCLUDE_DIRS:
            continue
        pj = post_dir / "post.json"
        if not pj.exists():
            continue
        try:
            data = json.loads(pj.read_text(encoding="utf-8"))
        except Exception:
            continue
        for tag in data.get("tags", []):
            counts[tag.strip().lower().replace(" ", "-")] += 1
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete", action="store_true")
    args = parser.parse_args()

    live_tag_counts = count_live_tags()

    if not TAGS_DIR.exists():
        print("No docs/tag/ directory found — nothing to check.")
        return

    orphaned, stale_threshold = [], []
    for tag_dir in TAGS_DIR.iterdir():
        if not tag_dir.is_dir():
            continue
        count = live_tag_counts.get(tag_dir.name, 0)
        if count == 0:
            orphaned.append(tag_dir.name)
        elif count < 2:  # below even the base "qualifying" threshold
            stale_threshold.append((tag_dir.name, count))

    print(
        f"Orphaned tag pages (0 live posts, should not exist at all): {len(orphaned)}")
    for t in orphaned:
        print(f"  docs/tag/{t}/")

    if stale_threshold:
        print(
            f"\nAlso below the 2-post 'qualifying' threshold (should also be removed): {len(stale_threshold)}")
        for t, c in stale_threshold:
            print(f"  docs/tag/{t}/  ({c} live posts)")

    to_remove = orphaned + [t for t, _ in stale_threshold]

    with open("remove_orphaned_tags.sh", "w") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        for t in to_remove:
            f.write(f"git rm -r \"docs/tag/{t}\"\n")
    print(f"\nWrote remove_orphaned_tags.sh ({len(to_remove)} tag pages)")

    if args.delete:
        for t in to_remove:
            subprocess.run(["git", "rm", "-r", f"docs/tag/{t}"], check=True)
        print(
            f"Ran git rm on {len(to_remove)} orphaned tag pages. Review `git status`, then commit.")
    else:
        print("Dry-run only. Re-run with --delete to actually remove them.")


if __name__ == "__main__":
    main()
