#!/usr/bin/env python3
"""
scripts/generate_sitemap.py

Generates a single, Google-friendly sitemap.xml for kubaik.github.io.
Includes support for image sitemaps (OG images).

Run after the site is built.
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

try:
    from bs4 import BeautifulSoup
except ImportError:
    print("ERROR: beautifulsoup4 is required. Add it to requirements.txt")
    sys.exit(1)


# URL priority and changefreq rules
URL_RULES = {
    "/":                    ("1.0", "daily"),
    "/about/":              ("0.9", "monthly"),
    "/contact/":            ("0.8", "monthly"),
    "/privacy-policy/":     ("0.5", "yearly"),
    "/terms-of-service/":   ("0.5", "yearly"),
}


def get_priority_and_changefreq(path: str) -> tuple[str, str]:
    """Return priority and changefreq based on URL path."""
    if path in URL_RULES:
        return URL_RULES[path]
    if path.startswith("/tag/"):
        return "0.6", "weekly"
    return "0.7", "weekly"  # Default for articles


def is_excluded(path: str) -> bool:
    excluded = {"/404/", "/drafts/", "/staging/", "/admin/"}
    return any(path.startswith(p) for p in excluded)


def extract_page_data(html_path: Path, base_url: str) -> dict | None:
    """Extract canonical URL, lastmod, and OG image from HTML."""
    try:
        content = html_path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(content, "html.parser")

        # Canonical URL
        canonical_tag = soup.find("link", rel="canonical")
        canonical = canonical_tag["href"] if canonical_tag else None

        # Last modified / published time
        lastmod = None
        for prop in ("article:modified_time", "article:published_time"):
            meta = soup.find("meta", property=prop)
            if meta and meta.get("content"):
                lastmod = meta["content"]
                break

        # OG Image (for image sitemap)
        og_image = None
        og_tag = soup.find("meta", property="og:image")
        if og_tag and og_tag.get("content"):
            og_image = og_tag["content"]

        return {
            "canonical": canonical,
            "lastmod": lastmod,
            "og_image": og_image,
        }
    except Exception as e:
        print(f"Warning: Could not parse {html_path}: {e}")
        return None


def discover_pages(site_dir: Path, base_url: str) -> list[dict]:
    """Find all pages and extract relevant metadata."""
    pages = []
    for html_file in sorted(site_dir.rglob("index.html")):
        rel_path = html_file.parent.relative_to(site_dir).as_posix()
        url_path = "/" if rel_path == "." else f"/{rel_path}/"

        if is_excluded(url_path):
            continue

        data = extract_page_data(html_file, base_url)
        if data:
            pages.append({
                "path": url_path,
                "canonical": data["canonical"],
                "lastmod": data["lastmod"],
                "og_image": data["og_image"],
            })
    return pages


def format_lastmod(iso_str: str | None) -> str:
    """Convert ISO datetime to YYYY-MM-DD format."""
    if not iso_str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def build_sitemap_xml(pages: list[dict], base_url: str) -> str:
    """Build a Google-compatible sitemap with optional image support."""
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"',
        '        xmlns:image="http://www.google.com/schemas/sitemap-image/1.1">',
    ]

    for page in pages:
        loc = page.get("canonical") or (base_url.rstrip("/") + page["path"])
        lastmod = format_lastmod(page.get("lastmod"))
        priority, changefreq = get_priority_and_changefreq(page["path"])

        lines.append("  <url>")
        lines.append(f"    <loc>{escape(loc)}</loc>")
        lines.append(f"    <lastmod>{lastmod}</lastmod>")
        lines.append(f"    <changefreq>{changefreq}</changefreq>")
        lines.append(f"    <priority>{priority}</priority>")

        # Add image if OG image exists
        if page.get("og_image"):
            lines.append("    <image:image>")
            lines.append(
                f"      <image:loc>{escape(page['og_image'])}</image:loc>")
            lines.append("    </image:image>")

        lines.append("  </url>")

    lines.append("</urlset>")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Google-friendly sitemap.xml")
    parser.add_argument("--output-dir", default="./docs",
                        help="Built site directory")
    parser.add_argument(
        "--base-url", default="https://kubaik.github.io", help="Site base URL")
    args = parser.parse_args()

    site_dir = Path(args.output_dir)
    if not site_dir.exists():
        print(f"ERROR: Directory not found: {site_dir}", file=sys.stderr)
        return 1

    print(f"Scanning {site_dir} for pages...")
    pages = discover_pages(site_dir, args.base_url)
    print(f"Found {len(pages)} pages")

    xml_content = build_sitemap_xml(pages, args.base_url)
    output_file = site_dir / "sitemap.xml"
    output_file.write_text(xml_content, encoding="utf-8")

    print(f"✅ Generated single sitemap: {output_file} ({len(pages)} URLs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
