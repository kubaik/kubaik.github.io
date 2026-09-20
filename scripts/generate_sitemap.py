#!/usr/bin/env python3
"""
scripts/generate_sitemap.py
===========================
Sitemap VALIDATOR for the file produced by static_site_generator.py.

HISTORY
-------
This script used to write docs/sitemap.xml directly, duplicating
StaticSiteGenerator._generate_sitemap() (which runs inside
`python blog_system.py build`). Two writers to one artifact is a drift
risk — the two implementations had different noindex-exclusion logic at
various points in this repo's history, and whichever ran last silently
won. Ownership is now consolidated into the SSG; this script reads the
file SSG produced and reports on it.

CHECKS
------
  1. docs/sitemap.xml exists and is well-formed XML.
  2. Every post URL listed has a corresponding docs/<slug>/post.json.
  3. No noindexed post (merge-redirect stub or fabrication-quarantined)
     is listed in the sitemap.
  4. Required static pages (/, /about/, /contact/, /privacy-policy/,
     /terms-of-service/, /tag/, /dmca/, /ai-content-policy/) are present.
  5. (Warning only) Indexable posts that exist on disk but aren't in the
     sitemap — these are indexable-by-default and should be listed.

EXIT CODES
----------
  0 — clean, or only warnings, or --strict not passed
  1 — with --strict: any hard validation issue exits 1. Without --strict,
      the script always exits 0 so it can be called from the workflow's
      non-blocking audit step without taking the job down.

Usage:
    python scripts/generate_sitemap.py --validate-only
    python scripts/generate_sitemap.py --output-dir ./docs --validate-only
    python scripts/generate_sitemap.py --output-dir ./docs --strict --validate-only
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Optional dependencies — degrade gracefully if the adsense_fixes package
# isn't importable in a stripped-down environment.
try:
    from adsense_fixes.canonical_guard import is_merge_stub
except Exception:  # pragma: no cover
    def is_merge_stub(_post) -> bool:
        return False

try:
    from adsense_fixes.queue_noindex_guard import get_quarantined_slugs
except Exception:  # pragma: no cover
    def get_quarantined_slugs(_path) -> set:
        return set()


_SKIP_DIRS = {"static", "tag", "author", "page"}
_SITEMAP_NS = "{http://www.sitemaps.org/schemas/sitemap/0.9}"

_REQUIRED_STATIC_PATHS = {
    "/",
    "/about/",
    "/contact/",
    "/privacy-policy/",
    "/terms-of-service/",
    "/tag/",
    "/dmca/",
    "/ai-content-policy/",
}


def _load_post_slugs(docs_dir: Path) -> tuple[set[str], set[str]]:
    """
    Scan docs_dir for post directories and return:
      (all_slugs, noindex_slugs)

    noindex_slugs combines:
      - merge-redirect stubs (auto_retire_duplicates.py output)
      - posts currently quarantined in regeneration_queue.json
        (unresolved fabricated anecdotes/citations)
    Mirrors the exact same `_noindex_slugs` computation that
    static_site_generator.generate_site() performs, so the sitemap check
    here and the sitemap writer there always agree on what should be
    excluded.
    """
    all_slugs: set[str] = set()
    noindex_slugs: set[str] = set()

    quarantine = get_quarantined_slugs(Path("./regeneration_queue.json"))
    noindex_slugs |= quarantine

    if not docs_dir.exists():
        return all_slugs, noindex_slugs

    for post_dir in docs_dir.iterdir():
        if not post_dir.is_dir() or post_dir.name in _SKIP_DIRS:
            continue
        pj = post_dir / "post.json"
        if not pj.exists():
            continue
        try:
            data = json.loads(pj.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            continue
        if not data.get("title", "").strip():
            continue
        all_slugs.add(post_dir.name)
        try:
            if is_merge_stub(data):
                noindex_slugs.add(post_dir.name)
        except Exception:
            pass

    return all_slugs, noindex_slugs


def _parse_sitemap_locs(sitemap_path: Path) -> list[str]:
    """Return all <loc> text values, namespace-aware."""
    tree = ET.parse(sitemap_path)
    root = tree.getroot()
    return [
        loc_el.text.strip()
        for loc_el in root.iter(f"{_SITEMAP_NS}loc")
        if loc_el.text and loc_el.text.strip()
    ]


def _url_to_slug(loc: str, base_url: str) -> str | None:
    """
    If loc is a single-segment post URL under base_url (e.g.
    https://example.com/my-post/), return "my-post". Otherwise None
    (covers the homepage, /about/, /tag/x/, /page/2/, external URLs,
    etc.).
    """
    prefix = base_url.rstrip("/") + "/"
    if not loc.startswith(prefix):
        return None
    path = loc[len(prefix):].rstrip("/")
    if not path or "/" in path:
        return None
    return path


def validate(
    docs_dir: Path,
    base_url: str,
    strict: bool = False,
) -> int:
    base_url = base_url.rstrip("/")
    sitemap_path = docs_dir / "sitemap.xml"

    issues: list[str] = []
    warnings: list[str] = []

    if not sitemap_path.exists():
        print(f"❌ sitemap.xml not found at {sitemap_path}")
        print("   (Was `blog_system.py build` run before this validation?)")
        return 1 if strict else 0

    try:
        locs = _parse_sitemap_locs(sitemap_path)
    except ET.ParseError as e:
        print(f"❌ sitemap.xml is not well-formed XML: {e}")
        return 1 if strict else 0

    print(f"sitemap.xml: {len(locs)} URLs")

    all_slugs, noindex_slugs = _load_post_slugs(docs_dir)

    sitemap_slugs: set[str] = set()
    for loc in locs:
        slug = _url_to_slug(loc, base_url)
        if slug is None:
            continue
        sitemap_slugs.add(slug)
        if slug not in all_slugs:
            issues.append(f"orphaned URL (no post.json for /{slug}/)")
        if slug in noindex_slugs:
            issues.append(f"noindexed post leaked into sitemap: /{slug}/")

    # Static pages present?
    loc_set = set(locs)
    for path in sorted(_REQUIRED_STATIC_PATHS):
        url = f"{base_url}{path}"
        if url not in loc_set:
            warnings.append(f"static page missing from sitemap: {path}")

    # Indexable posts present on disk but missing from sitemap
    expected_indexable = all_slugs - noindex_slugs
    missing = expected_indexable - sitemap_slugs
    if missing:
        for slug in sorted(missing)[:10]:
            warnings.append(f"indexable post not in sitemap: /{slug}/")
        if len(missing) > 10:
            warnings.append(f"... and {len(missing) - 10} more")

    # Also flag noindexed posts that exist on disk — for the record, so the
    # log shows the exclusion is actually happening, not just not-checking.
    if noindex_slugs:
        print(f"  {len(noindex_slugs)} noindexed post(s) excluded by design")

    for w in warnings:
        print(f"⚠️  {w}")
    for i in issues:
        print(f"❌ {i}")

    if issues:
        print(f"\n{len(issues)} issue(s), {len(warnings)} warning(s)")
        return 1 if strict else 0

    print(f"\n✅ sitemap.xml valid ({len(warnings)} warning(s))")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the SSG-generated docs/sitemap.xml. "
            "This script is read-only; sitemap.xml is written by "
            "StaticSiteGenerator._generate_sitemap() inside "
            "`python blog_system.py build`."
        )
    )
    parser.add_argument(
        "--output-dir", "--docs-dir",
        dest="output_dir",
        default="./docs",
        help="Directory containing sitemap.xml (default: ./docs)",
    )
    parser.add_argument(
        "--base-url",
        default="https://kubaik.github.io",
        help="Site base URL, no trailing slash (default: %(default)s)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Accepted for backwards compatibility. This script only "
            "validates; it never writes sitemap.xml."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero on any validation issue (warnings still exit 0).",
    )
    args = parser.parse_args()

    return validate(
        docs_dir=Path(args.output_dir),
        base_url=args.base_url,
        strict=args.strict,
    )


if __name__ == "__main__":
    raise SystemExit(main())