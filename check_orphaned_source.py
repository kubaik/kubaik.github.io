"""
check_orphaned_source.py

For the 273 dirs with neither post.json nor markdown: do they have a live
index.html? If yes, these are published pages with zero recoverable source
in the repo — can't be updated/regenerated/corrected through the normal
pipeline at all, only re-written from scratch (losing whatever made them
rank, if anything) or scraped back out of their own rendered HTML.

Run from repo root: python check_orphaned_source.py
"""
from pathlib import Path

DOCS_DIR = Path("./docs")
EXCLUDE_DIRS = {"static", "tag", "author", "page", "contact", "about"}


def main():
    live_no_source, dead_dirs = [], []

    for post_dir in DOCS_DIR.iterdir():
        if not post_dir.is_dir() or post_dir.name in EXCLUDE_DIRS:
            continue
        if (post_dir / "post.json").exists():
            continue
        has_md = any(post_dir.glob("*.md"))
        if has_md:
            continue
        index = post_dir / "index.html"
        if index.exists() and index.stat().st_size > 500:
            live_no_source.append(post_dir.name)
        else:
            dead_dirs.append(post_dir.name)

    print(
        f"Live pages with NO recoverable source (post.json or .md): {len(live_no_source)}")
    for s in live_no_source[:50]:
        print(f"  docs/{s}/index.html  <- serving, unrecoverable source")

    print(
        f"\nEmpty/dead directories (no index.html either — likely git artifacts): {len(dead_dirs)}")
    for s in dead_dirs[:20]:
        print(f"  docs/{s}/")


if __name__ == "__main__":
    main()
