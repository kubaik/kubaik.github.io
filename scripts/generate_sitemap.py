#!/usr/bin/env python3
"""
scripts/generate_sitemap.py

Generates a single, Google-friendly sitemap.xml for kubaik.github.io.
Includes support for image sitemaps (OG images).

Run after the site is built.

FIXES applied (2026-07):
  - Pages whose <meta name="robots"> contains "noindex" are now EXCLUDED
    from the sitemap. Previously all 912 /tag/* archive pages (which are
    noindex,follow) were submitted, causing persistent "Submitted URL
    marked noindex" warnings in Search Console.
  - Output is deduplicated by final <loc>. Previously, noindex "this post
    was merged" redirect stubs resolved their canonical to the live post's
    URL, so a single URL could appear in the sitemap dozens of times
    (one <url> entry was found duplicated 23x).
  - Pages missing a canonical tag are now skipped with a warning instead
    of silently falling back to a possibly-wrong path-derived URL.
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
    """Extract canonical URL, lastmod, robots directive, and OG image from HTML."""
    try:
        content = html_path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(content, "html.parser")

        # Robots directive — skip noindex pages entirely
        robots_tag = soup.find("meta", attrs={"name": "robots"})
        robots_content = (robots_tag.get("content", "")
                          if robots_tag else "").lower()
        if "noindex" in robots_content:
            return None

        # Canonical URL — required. A page without one shouldn't be in the
        # sitemap since we can't be sure what URL Google should index it as.
        canonical_tag = soup.find("link", rel="canonical")
        canonical = canonical_tag["href"] if canonical_tag and canonical_tag.get(
            "href") else None
        if not canonical:
            print(f"  Warning: skipping {html_path} — no canonical tag found")
            return None

        # Skip meta-refresh redirect stubs even if somehow not marked noindex
        if soup.find("meta", attrs={"http-equiv": lambda v: v and v.lower() == "refresh"}):
            return None

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
    """Find all indexable pages and extract relevant metadata, deduplicated by canonical URL."""
    pages_by_url: dict[str, dict] = {}

    for html_file in sorted(site_dir.rglob("index.html")):
        rel_path = html_file.parent.relative_to(site_dir).as_posix()
        url_path = "/" if rel_path == "." else f"/{rel_path}/"

        if is_excluded(url_path):
            continue

        data = extract_page_data(html_file, base_url)
        if not data:
            continue

        loc = data["canonical"]

        # Dedup: if we've already seen this canonical URL, keep the entry
        # with the most complete metadata (prefer one that has an OG image
        # and a real lastmod) rather than just the first one encountered.
        existing = pages_by_url.get(loc)
        if existing is None:
            pages_by_url[loc] = {
                "path": url_path,
                "canonical": loc,
                "lastmod": data["lastmod"],
                "og_image": data["og_image"],
            }
        else:
            if not existing.get("og_image") and data.get("og_image"):
                existing["og_image"] = data["og_image"]
            if not existing.get("lastmod") and data.get("lastmod"):
                existing["lastmod"] = data["lastmod"]

    return list(pages_by_url.values())


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

    # Sort for deterministic, diff-friendly output
    for page in sorted(pages, key=lambda p: p["canonical"]):
        loc = page["canonical"]
        lastmod = format_lastmod(page.get("lastmod"))
        priority, changefreq = get_priority_and_changefreq(page["path"])

        lines.append("  <url>")
        lines.append(f"    <loc>{escape(loc)}</loc>")
        lines.append(f"    <lastmod>{lastmod}</lastmod>")
        lines.append(f"    <changefreq>{changefreq}</changefreq>")
        lines.append(f"    <priority>{priority}</priority>")

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

    print(f"Scanning {site_dir} for indexable pages...")
    pages = discover_pages(site_dir, args.base_url)
    print(f"Found {len(pages)} indexable, deduplicated pages")

    xml_content = build_sitemap_xml(pages, args.base_url)
    output_file = site_dir / "sitemap.xml"
    output_file.write_text(xml_content, encoding="utf-8")

    print(f"✅ Generated single sitemap: {output_file} ({len(pages)} URLs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
