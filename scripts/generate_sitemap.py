#!/usr/bin/env python3
"""
scripts/generate_sitemap.py

Generates a single sitemap.xml for kubaik.github.io.
Run this as part of the build pipeline after all posts are generated.

This version always creates ONE sitemap.xml (no sitemap index/chunking).
"""

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape


# Priority and changefreq by URL pattern
URL_RULES = [
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
    for pattern, priority, changefreq in pattern_rules:
        if re.match(pattern, path):
            return priority, changefreq
    return "0.5", "monthly"


def is_excluded(path: str, exclude_patterns: list) -> bool:
    return any(re.match(p, path) for p in exclude_patterns)


def extract_meta_from_html(html_path: Path) -> dict:
    content = html_path.read_text(encoding="utf-8", errors="replace")

    canonical = None
    modified = None

    canonical_match = re.search(
        r'<link[^>]+rel=["\']canonical["\'][^>]+href=["\']([^"\']+)["\']', content
    )
    if canonical_match:
        canonical = canonical_match.group(1)

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
    pages = []
    for html_file in sorted(site_dir.rglob("index.html")):
        rel_parent = html_file.parent.relative_to(site_dir).as_posix()
        rel_path = "/" if rel_parent == "." else "/" + \
            rel_parent.rstrip("/") + "/"

        if is_excluded(rel_path, EXCLUDE_PATTERNS):
            continue

        meta = extract_meta_from_html(html_file)
        pages.append({
            "path": rel_path,
            "canonical": meta.get("canonical"),
            "modified": meta.get("modified"),
        })
    return pages


def format_lastmod(iso_str: str | None) -> str:
    if iso_str:
        try:
            dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def build_sitemap_xml(pages: list[dict], base_url: str) -> str:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a single sitemap.xml")
    parser.add_argument("--output-dir", default="./_site",
                        help="Path to the built site directory")
    parser.add_argument(
        "--base-url", default="https://kubaik.github.io", help="Base URL of the site")
    args = parser.parse_args()

    site_dir = Path(args.output_dir)
    if not site_dir.exists():
        print(
            f"ERROR: Output directory does not exist: {site_dir}", file=sys.stderr)
        return 1

    print(f"Scanning: {site_dir}")
    pages = discover_pages(site_dir)
    print(f"Found {len(pages)} pages")

    xml = build_sitemap_xml(pages, args.base_url)
    out_path = site_dir / "sitemap.xml"
    out_path.write_text(xml, encoding="utf-8")

    print(f"Generated single sitemap: {out_path} ({len(pages)} URLs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
