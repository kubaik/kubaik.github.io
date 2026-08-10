"""
reconstruct_post_json.py

For posts with a live index.html but no post.json/markdown: scrape the
rendered HTML back into a post.json so these posts re-enter the normal
pipeline (can be regenerated/refreshed/corrected/citation-checked like
everything else). This is a one-time recovery step, not a replacement for
fixing WHY they lost their source (see the CI guard below).

Reconstructed fields are best-effort: title/content/meta_description come
straight from the rendered page. tags/seo_keywords default empty since
they generally aren't recoverable from rendered HTML — the next scheduled
refresh pass should repopulate them properly.

Requires: pip install beautifulsoup4 --break-system-packages

Run from repo root: python reconstruct_post_json.py --dry-run
                     python reconstruct_post_json.py   # writes files
"""
import argparse
import json
import re
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup

DOCS_DIR = Path("./docs")
EXCLUDE_DIRS = {
    "static", "tag", "author", "page",
    "contact", "about", "privacy-policy", "terms-of-service",
    "dmca", "ai-content-policy",
}


def slug_to_title(slug: str) -> str:
    return slug.replace("-", " ").strip().capitalize()


def reconstruct_one(post_dir: Path) -> dict | None:
    index = post_dir / "index.html"
    if not index.exists():
        return None
    soup = BeautifulSoup(index.read_text(
        encoding="utf-8", errors="ignore"), "html.parser")

    title_tag = soup.find("h1") or soup.find("title")
    title = title_tag.get_text(
        strip=True) if title_tag else slug_to_title(post_dir.name)

    meta_desc_tag = soup.find("meta", attrs={"name": "description"})
    meta_description = meta_desc_tag.get(
        "content", "").strip() if meta_desc_tag else ""

    # Article body: prefer a semantic <article>/<main>, fall back to the
    # largest <div> by text length so this works across template versions.
    article = soup.find("article") or soup.find("main")
    if article is None:
        divs = soup.find_all("div")
        article = max(divs, key=lambda d: len(d.get_text()), default=None)

    content_html = str(article) if article else ""
    content_text = article.get_text("\n", strip=True) if article else ""

    og_image = soup.find("meta", attrs={"property": "og:image"})
    featured_image = og_image.get("content", "") if og_image else ""

    now = datetime.now().isoformat()

    return {
        "title": title,
        "content": content_text,
        # keep raw HTML too, for manual comparison
        "content_html_snapshot": content_html,
        "slug": post_dir.name,
        "tags": ["reconstructed"],
        "meta_description": meta_description,
        "featured_image": featured_image,
        "created_at": now,
        "updated_at": now,
        "seo_keywords": [],
        "affiliate_links": [],
        "monetization_data": {"ad_slots": 3, "affiliate_count": 0, "review_status": "reconstructed_no_source"},
        "twitter_hashtags": "",
        "_reconstruction_note": "post.json rebuilt from rendered HTML; no original source "
        "existed in the repo at reconstruction time. Flagged for "
        "priority refresh — tags/seo_keywords/monetization data "
        "are placeholders, not real.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    reconstructed, failed = 0, []
    for post_dir in DOCS_DIR.iterdir():
        if not post_dir.is_dir() or post_dir.name in EXCLUDE_DIRS:
            continue
        if (post_dir / "post.json").exists() or any(post_dir.glob("*.md")):
            continue
        data = reconstruct_one(post_dir)
        if data is None or not data["content"]:
            failed.append(post_dir.name)
            continue
        if args.dry_run:
            print(f"[dry-run] would write docs/{post_dir.name}/post.json "
                  f"({len(data['content'])} chars recovered)")
        else:
            (post_dir / "post.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
            print(f"Reconstructed docs/{post_dir.name}/post.json")
        reconstructed += 1

    print(
        f"\n{'Would reconstruct' if args.dry_run else 'Reconstructed'}: {reconstructed}")
    print(
        f"Failed (no article content found, needs manual look): {len(failed)}")
    for f in failed[:20]:
        print(f"  docs/{f}/")


if __name__ == "__main__":
    main()
