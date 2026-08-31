#!/usr/bin/env python3
"""
audit_dead_links.py — pre-deploy 404 / sitemap auditor for kubaik.github.io

Run this AFTER `python static_site_generator.py` (or blog_system.py build),
against the freshly generated ./docs directory, and BEFORE you push. It
answers the exact question Search Console is asking you: "which URLs does
this site currently link to, or list in the sitemap, that don't actually
resolve to a real 200 page in this build?"

It does not depend on the adsense_fixes/* package (not present in this
upload), so it will catch anything that package's audit-links mode might
miss due to a stale cache, and it's runnable standalone in CI.

Usage:
    python audit_dead_links.py [--docs ./docs] [--fail-on-error]

Exit code is 1 if any dead internal link or dead sitemap entry is found
and --fail-on-error is passed — wire this into your GitHub Actions build
step so a broken link can never ship again.
"""
from __future__ import annotations
import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlsplit

SCRIPT_RE = re.compile(r'<script\b[^>]*>.*?</script>', re.S | re.I)
# Negative lookbehind so this matches a real `href="..."` attribute but not
# `data-tag-href="..."`, `data-xhref="..."`, etc. — those are inert data
# attributes for JS to read, not navigable <a> links, and matching them was
# producing false-positive "dead link" reports.
HREF_RE = re.compile(r'(?<![-\w])href="([^"]+)"')

# Anything you deliberately allow to be an external/anchor/mailto/etc link
# and don't want flagged as "internal but unresolved".
SKIP_PREFIXES = ("http://", "https://", "mailto:", "tel:", "#", "javascript:")


def route_exists(docs_dir: Path, path: str) -> bool:
    """
    Mirrors how GitHub Pages actually resolves a path, so this catches the
    same things Search Console will:
      /foo/        -> docs/foo/index.html
      /foo         -> docs/foo.html  OR docs/foo/index.html (GH Pages will
                       redirect foo -> foo/, which is exactly the source of
                       the 2x "Page with redirect" entries in GSC — treat
                       missing trailing slash as a WARNING, not a pass)
      /foo.png     -> docs/foo.png (literal file)
    """
    path = path.split("#")[0].split("?")[0]
    if path in ("", "/"):
        return (docs_dir / "index.html").exists()

    rel = path.lstrip("/")
    candidate = docs_dir / rel

    if rel.endswith("/"):
        return (candidate / "index.html").exists()

    # Literal file (image, .xml, .txt, etc.)
    if candidate.exists() and candidate.is_file():
        return True

    # Directory without trailing slash — resolves via a 301 on GH Pages,
    # not a clean 200. Caller treats this as "needs trailing slash" below.
    if candidate.is_dir() and (candidate / "index.html").exists():
        return "redirect"

    return False


def collect_internal_hrefs(html_path: Path) -> list[str]:
    text = html_path.read_text(encoding="utf-8", errors="ignore")
    text = SCRIPT_RE.sub("", text)  # drop inline JS — it isn't markup
    hrefs = HREF_RE.findall(text)
    return [h for h in hrefs if not h.startswith(SKIP_PREFIXES)]


def audit_links(docs_dir: Path) -> dict:
    dead, needs_slash, ok = [], [], 0
    html_files = sorted(docs_dir.rglob("index.html")) + sorted(
        p for p in docs_dir.glob("*.html")
    )
    for html_file in html_files:
        for href in collect_internal_hrefs(html_file):
            result = route_exists(docs_dir, href)
            if result is True:
                ok += 1
            elif result == "redirect":
                needs_slash.append((str(html_file.relative_to(docs_dir)), href))
            else:
                dead.append((str(html_file.relative_to(docs_dir)), href))
    return {"dead": dead, "needs_slash": needs_slash, "ok": ok}


def audit_sitemap(docs_dir: Path, base_url: str) -> dict:
    sitemap_path = docs_dir / "sitemap.xml"
    if not sitemap_path.exists():
        return {"error": "sitemap.xml not found", "dead": [], "needs_slash": []}

    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    tree = ET.parse(sitemap_path)
    dead, needs_slash = [], []
    for url_el in tree.getroot().findall("sm:url", ns):
        loc = url_el.find("sm:loc", ns).text.strip()
        parsed = urlsplit(loc)
        path = parsed.path
        result = route_exists(docs_dir, path)
        if result is True:
            continue
        elif result == "redirect":
            needs_slash.append(loc)
        else:
            dead.append(loc)
    return {"dead": dead, "needs_slash": needs_slash}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="./docs")
    ap.add_argument("--base-url", default="https://kubaik.github.io")
    ap.add_argument("--fail-on-error", action="store_true")
    args = ap.parse_args()

    docs_dir = Path(args.docs)
    if not docs_dir.exists():
        print(f"ERROR: {docs_dir} does not exist. Run the generator first.")
        sys.exit(2)

    print(f"Auditing {docs_dir.resolve()} ...\n")

    link_report = audit_links(docs_dir)
    sitemap_report = audit_sitemap(docs_dir, args.base_url)

    print("=" * 70)
    print("INTERNAL <a href> AUDIT (crawled from rendered HTML)")
    print("=" * 70)
    print(f"  OK links:                {link_report['ok']}")
    print(f"  Dead links (404):        {len(link_report['dead'])}")
    for src, href in link_report["dead"]:
        print(f"    ✗ {src}  ->  {href}")
    print(f"  Missing trailing slash (301 on GH Pages, wastes crawl budget "
          f"and can surface as 'Page with redirect'): {len(link_report['needs_slash'])}")
    for src, href in link_report["needs_slash"]:
        print(f"    ⚠ {src}  ->  {href}  (should be '{href.rstrip('/')}/ ')")

    print()
    print("=" * 70)
    print("SITEMAP.XML AUDIT")
    print("=" * 70)
    if sitemap_report.get("error"):
        print(f"  {sitemap_report['error']}")
    else:
        print(f"  Dead sitemap entries:     {len(sitemap_report['dead'])}")
        for loc in sitemap_report["dead"]:
            print(f"    ✗ {loc}")
        print(f"  Sitemap entries needing trailing slash: "
              f"{len(sitemap_report['needs_slash'])}")
        for loc in sitemap_report["needs_slash"]:
            print(f"    ⚠ {loc}")

    total_problems = (
        len(link_report["dead"])
        + len(link_report["needs_slash"])
        + len(sitemap_report.get("dead", []))
        + len(sitemap_report.get("needs_slash", []))
    )
    print()
    print(f"TOTAL ISSUES: {total_problems}")

    if args.fail_on_error and total_problems:
        sys.exit(1)


if __name__ == "__main__":
    main()