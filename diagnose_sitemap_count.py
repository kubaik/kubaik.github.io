"""
diagnose_sitemap_count.py

Breaks down every URL in sitemap.xml by category, to explain a gap between
"N articles" shown on the homepage and the sitemap's total URL count.
Doesn't guess — just counts what's actually there.

Run from repo root: python diagnose_sitemap_count.py
"""
import re
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

SITEMAP = Path("./docs/sitemap.xml")

CATEGORIES = [
    ("homepage", re.compile(r'^https?://[^/]+/$')),
    ("tag_archive", re.compile(r'/tag/')),
    ("author_page", re.compile(r'/author/')),
    ("pagination", re.compile(r'/page/\d+/')),
    ("utility_page", re.compile(
        r'/(about|contact|privacy-policy|terms-of-service|dmca|ai-content-policy)/$'
    )),
]


def categorize(url: str) -> str:
    for name, pattern in CATEGORIES:
        if pattern.search(url):
            return name
    return "article"  # everything else — this should match your article count


def main():
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    tree = ET.parse(SITEMAP)
    urls = [el.text for el in tree.getroot().findall(".//sm:loc", ns)]

    counts = Counter(categorize(u) for u in urls)

    print(f"Total URLs in sitemap: {len(urls)}\n")
    for category, count in counts.most_common():
        print(f"  {category:15s} {count}")

    print(f"\n'article' count above should match your homepage's article count.")
    print(f"If it doesn't, list a few article-category URLs to spot-check:")
    article_urls = [u for u in urls if categorize(u) == "article"]
    for u in article_urls[:10]:
        print(f"  {u}")
    if len(article_urls) > 10:
        print(f"  ... and {len(article_urls) - 10} more")


if __name__ == "__main__":
    main()
