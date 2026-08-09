"""
audit_markdown_fallback.py

Checks the ~275 posts with no post.json: do they actually get served with
the generic from_markdown_file() metadata (templated meta_description,
'recovered'/'blog' tags, no SEO keywords), or is there another metadata
source (e.g. front-matter in the .md file itself) this triage missed?

Run from repo root: python audit_markdown_fallback.py
"""
import re
from pathlib import Path

DOCS_DIR = Path("./docs")
EXCLUDE_DIRS = {"static", "tag", "author", "page", "contact", "about"}

GENERIC_META_PATTERN = re.compile(r'^Blog post about .+$')


def find_markdown_file(post_dir: Path):
    for candidate in ("index.md", "post.md", f"{post_dir.name}.md"):
        p = post_dir / candidate
        if p.exists():
            return p
    md_files = list(post_dir.glob("*.md"))
    return md_files[0] if md_files else None


def has_frontmatter(md_text: str) -> bool:
    return md_text.startswith("---\n") or md_text.startswith("+++\n")


def main():
    no_json = [d for d in DOCS_DIR.iterdir()
               if d.is_dir() and d.name not in EXCLUDE_DIRS and not (d / "post.json").exists()]

    hitting_generic_fallback, has_own_frontmatter, no_markdown_found = [], [], []

    for post_dir in no_json:
        md = find_markdown_file(post_dir)
        if md is None:
            no_markdown_found.append(post_dir.name)
            continue
        text = md.read_text(encoding="utf-8", errors="ignore")
        if has_frontmatter(text):
            has_own_frontmatter.append(post_dir.name)
        else:
            # no front-matter -> from_markdown_file()'s generic fallback is
            # what actually generates this post's metadata at build time
            hitting_generic_fallback.append(post_dir.name)

    print(f"Posts with no post.json: {len(no_json)}")
    print(
        f"  -> hitting generic fallback metadata (real problem): {len(hitting_generic_fallback)}")
    for s in hitting_generic_fallback[:50]:
        print(f"     docs/{s}/")
    print(
        f"  -> have their own front-matter (probably fine, investigate parser): {len(has_own_frontmatter)}")
    print(
        f"  -> no markdown file found either (broken/orphaned dir): {len(no_markdown_found)}")
    for s in no_markdown_found[:20]:
        print(f"     docs/{s}/")


if __name__ == "__main__":
    main()
