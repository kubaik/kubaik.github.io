#!/usr/bin/env python3
"""
scripts/generate_sitemap.py

Generates sitemap.xml for kubaik.github.io from published posts.
Run this as part of the build pipeline after all posts are generated.

Usage:
    python scripts/generate_sitemap.py --output-dir ./_site --base-url https://kubaik.github.io

The script:
  1. Discovers all HTML files in the output directory
  2. Extracts canonical URLs and last-modified dates from meta tags
  3. Applies priority weights (homepage > about > posts > tags)
  4. Splits into a sitemap index + per-category sitemaps when count > 1000
  5. Writes sitemap.xml (or sitemap-index.xml + sitemap-posts.xml) to output-dir
"""

import argparse
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

# Priority and changefreq by URL pattern
URL_RULES = [
    # (pattern, priority, changefreq)
    (r"^/$", "1.0", "daily"),
    (r"^/about/$", "0.9", "monthly"),
    (r"^/contact/$", "0.8", "monthly"),
    (r"^/privacy-policy/$", "0.5", "yearly"),
    (r"^/terms-of-service/$", "0.5", "yearly"),
    (r"^/tag/[^/]+/$", "0.6", "weekly"),
    (r"^/[^/]+/$", "0.7", "weekly"),  # article pages
]

EXCLUDE_PATTERNS = [
    r"^/404",
    r"^/drafts/",
    r"^/staging/",
]


def get_url_meta(pattern_rules: list, path: str) -> tuple[str, str]:
    """Return (priority, changefreq) for a given URL path."""
    for pattern, priority, changefreq in pattern_rules:
        if re.match(pattern, path):
            return priority, changefreq
    return "0.5", "monthly"


def is_excluded(path: str, exclude_patterns: list) -> bool:
    return any(re.match(p, path) for p in exclude_patterns)


def extract_meta_from_html(html_path: Path) -> dict:
    """Extract canonical URL and article:modified_time from HTML file."""
    content = html_path.read_text(encoding="utf-8", errors="replace")

    canonical = None
    modified = None

    canonical_match = re.search(
        r'<link[^>]+rel=["\']canonical["\'][^>]+href=["\']([^"\']+)["\']', content
    )
    if canonical_match:
        canonical = canonical_match.group(1)

    # Try article:modified_time first, fall back to article:published_time
    for meta_name in ("article:modified_time", "article:published_time"):
        m = re.search(
            rf'<meta[^>]+property=["\']{re.escape(meta_name)}["\'][^>]+content=["\']([^"\']+)["\']',
            content,
        )
        if m:
            modified = m.group(1)
            break

    return {"canonical": canonical, "modified": modified}


def discover_pages(site_dir: Path) -> list[dict]:
    """Walk site directory, collect all index.html files."""
    pages = []
    for html_file in sorted(site_dir.rglob("index.html")):
        rel_parent = html_file.parent.relative_to(site_dir).as_posix()
        if rel_parent == ".":
            rel_path = "/"
        else:
            rel_path = "/" + rel_parent.rstrip("/") + "/"

        if is_excluded(rel_path, EXCLUDE_PATTERNS):
            continue

        meta = extract_meta_from_html(html_file)
        pages.append(
            {
                "path": rel_path,
                "canonical": meta.get("canonical"),
                "modified": meta.get("modified"),
                "file": html_file,
            }
        )

    return pages


def format_lastmod(iso_str: str | None) -> str:
    """Return W3C date (YYYY-MM-DD) from ISO 8601 string, or today."""
    if iso_str:
        try:
            dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def build_sitemap_xml(pages: list[dict], base_url: str) -> str:
    """Build a single sitemap.xml string."""
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"',
        '        xmlns:image="http://www.google.com/schemas/sitemap-image/1.1">',
    ]

    for page in pages:
        canonical = page.get("canonical") or (
            base_url.rstrip("/") + page["path"])
        priority, changefreq = get_url_meta(URL_RULES, page["path"])
        lastmod = format_lastmod(page.get("modified"))

        lines.append("  <url>")
        lines.append(f"    <loc>{escape(canonical)}</loc>")
        lines.append(f"    <lastmod>{lastmod}</lastmod>")
        lines.append(f"    <changefreq>{changefreq}</changefreq>")
        lines.append(f"    <priority>{priority}</priority>")
        lines.append("  </url>")

    lines.append("</urlset>")
    return "\n".join(lines)


def build_sitemap_index(sitemaps: list[dict], base_url: str) -> str:
    """Build a sitemap index XML string when there are >1000 URLs."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    ]
    for sm in sitemaps:
        lines.append("  <sitemap>")
        lines.append(
            f"    <loc>{escape(base_url.rstrip('/') + '/' + sm['filename'])}</loc>")
        lines.append(f"    <lastmod>{today}</lastmod>")
        lines.append("  </sitemap>")
    lines.append("</sitemapindex>")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate sitemap.xml")
    parser.add_argument(
        "--output-dir",
        default="./_site",
        help="Path to the built site directory",
    )
    parser.add_argument(
        "--base-url",
        default="https://kubaik.github.io",
        help="Base URL of the site",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Max URLs per sitemap file (default: 1000)",
    )
    args = parser.parse_args()

    site_dir = Path(args.output_dir)
    if not site_dir.exists():
        print(
            f"ERROR: Output directory does not exist: {site_dir}", file=sys.stderr)
        return 1

    print(f"Scanning: {site_dir}")
    pages = discover_pages(site_dir)
    print(f"Found {len(pages)} pages")

    base_url = args.base_url.rstrip("/")

    if len(pages) <= args.chunk_size:
        # Single sitemap
        xml = build_sitemap_xml(pages, base_url)
        out_path = site_dir / "sitemap.xml"
        out_path.write_text(xml, encoding="utf-8")
        print(f"Written: {out_path} ({len(pages)} URLs)")
    else:
        # Sitemap index with chunks
        chunks = [
            pages[i: i + args.chunk_size]
            for i in range(0, len(pages), args.chunk_size)
        ]
        sitemaps = []
        for idx, chunk in enumerate(chunks, start=1):
            filename = f"sitemap-{idx}.xml"
            xml = build_sitemap_xml(chunk, base_url)
            out_path = site_dir / filename
            out_path.write_text(xml, encoding="utf-8")
            sitemaps.append({"filename": filename})
            print(f"Written: {out_path} ({len(chunk)} URLs)")

        index_xml = build_sitemap_index(sitemaps, base_url)
        index_path = site_dir / "sitemap.xml"
        index_path.write_text(index_xml, encoding="utf-8")
        print(f"Written: {index_path} (index with {len(chunks)} sitemaps)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
