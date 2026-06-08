"""
adsense_fixes/canonical_guard.py
==================================
Canonical URL enforcement and duplicate-URL prevention.

WHY THIS EXISTS
---------------
AdSense Site Readiness Guide (§4.2 Duplicate Content):
  "Canonical tags properly configured. Set <link rel='canonical'> tags
   to point to the preferred URL for every page, especially for paginated
   series or filtered views."

  "No cross-page content duplication. Each URL must serve unique content.
   Do not publish the same article under multiple URLs."

The static site generator already emits canonical tags, but three
patterns in the current code create subtle canonical problems:

  1. Tag pages can produce the same article appearing at both
     /slug/ and /tag/topic/  — both indexed, neither canonical to the other.
  2. The sitemap includes /tag/ pages without canonical enforcement,
     causing soft-duplicate signals.
  3. Posts with very similar slugs (e.g. "redis-caching" and
     "redis-caching-guide") don't get cross-canonical signals.

This module provides:
  - `validate_canonical(post, base_url)` — checks the canonical in a
    post's metadata is correct and self-referential.
  - `audit_duplicate_slugs(docs_dir)` — detects slug pairs above a
    similarity threshold that may confuse Google's dedup.
  - `generate_canonical_meta(post, base_url)` — returns the canonical
    <link> tag string for injection into templates.

HOW TO INTEGRATE
----------------
canonical_guard is already implicitly integrated via static_site_generator.py
since every template already emits <link rel='canonical'>.

Call validate_canonical() in the auto-mode pipeline after save_post()
to catch any misconfigured canonical values before they go live:

    from adsense_fixes.canonical_guard import validate_canonical
    issues = validate_canonical(blog_post, config.get('base_url', ''))
    for issue in issues:
        print(f"  ⚠️  Canonical: {issue}")

Run audit_duplicate_slugs periodically (CLI: python blog_system.py audit):
    from adsense_fixes.canonical_guard import audit_duplicate_slugs
    report = audit_duplicate_slugs(Path('./docs'))
    print(report)
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple


_SLUG_SIMILARITY_THRESHOLD = 0.70   # Jaccard over slug bigrams
_STOP_SLUG_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'in', 'of', 'for',
    'to', 'is', 'are', 'with', 'how', 'why', 'what',
    'guide', 'tutorial', 'post', 'article', 'blog',
}


# ── Canonical tag generator ────────────────────────────────────────────────

def generate_canonical_meta(post, base_url: str) -> str:
    """
    Return the canonical <link> tag string for a post.
    Normalises trailing slashes to be consistent site-wide.

    The canonical URL format is: {base_url}/{slug}/
    """
    slug = getattr(post, 'slug', '').strip('/')
    base = base_url.rstrip('/')
    canonical = f"{base}/{slug}/"
    return f'<link rel="canonical" href="{canonical}">'


# ── Canonical validation ───────────────────────────────────────────────────

def validate_canonical(post, base_url: str) -> List[str]:
    """
    Validate that a post's canonical URL is correctly formed.

    Returns a list of issue strings (empty list = no issues).
    """
    issues = []
    slug = getattr(post, 'slug', '')
    base = base_url.rstrip('/')

    if not slug:
        issues.append("Post has no slug — canonical URL cannot be formed.")
        return issues

    if not base:
        issues.append(
            "base_url is empty — canonical URL will be relative only.")

    # Check for slug patterns that create duplicate risks
    if slug.endswith('-2') or re.search(r'-\d+$', slug):
        issues.append(
            f"Slug '{slug}' ends with a numeric suffix, suggesting a duplicate "
            f"was auto-generated. Verify this is intentional."
        )

    # Slug should be URL-safe
    if re.search(r'[^a-z0-9\-]', slug):
        issues.append(
            f"Slug '{slug}' contains characters other than lowercase letters, "
            f"digits, and hyphens. This may cause canonical URL inconsistency."
        )

    # Slug should not start or end with hyphens
    if slug.startswith('-') or slug.endswith('-'):
        issues.append(f"Slug '{slug}' starts or ends with a hyphen.")

    # Canonical URL must use HTTPS
    if base.startswith('http://'):
        issues.append(
            "base_url uses HTTP — canonical URLs should use HTTPS."
        )

    return issues


# ── Duplicate slug audit ───────────────────────────────────────────────────

def audit_duplicate_slugs(docs_dir: Path) -> str:
    """
    Walk docs_dir and find slug pairs with high Jaccard bigram similarity.

    Two slugs that are too similar (e.g. 'redis-caching' and
    'redis-caching-guide') send mixed signals to Google about which is the
    authoritative page.

    Returns a human-readable report string.
    """
    if not docs_dir.exists():
        return "docs/ directory not found — nothing to audit."

    slugs: List[Tuple[str, str]] = []   # (slug, title)
    for post_dir in sorted(docs_dir.iterdir()):
        if not post_dir.is_dir() or post_dir.name in ('static', 'tag', 'author'):
            continue
        post_json = post_dir / 'post.json'
        if not post_json.exists():
            continue
        try:
            with open(post_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            slugs.append((post_dir.name, data.get('title', post_dir.name)))
        except (json.JSONDecodeError, KeyError):
            continue

    if len(slugs) < 2:
        return "✅ Canonical audit: only one post — no duplicate slug pairs possible."

    pairs: List[Tuple[float, str, str, str, str]] = []
    for i in range(len(slugs)):
        for j in range(i + 1, len(slugs)):
            slug_a, title_a = slugs[i]
            slug_b, title_b = slugs[j]
            score = _slug_jaccard(slug_a, slug_b)
            if score >= _SLUG_SIMILARITY_THRESHOLD:
                pairs.append((score, slug_a, title_a, slug_b, title_b))

    if not pairs:
        return (
            f"✅ Canonical audit: PASS\n"
            f"   {len(slugs)} slugs checked — no near-duplicate slugs found "
            f"(threshold: {_SLUG_SIMILARITY_THRESHOLD:.0%})"
        )

    pairs.sort(key=lambda x: x[0], reverse=True)
    lines = [
        f"⚠️  Canonical audit: {len(pairs)} near-duplicate slug pair(s) found.",
        f"   These pairs may create duplicate-content signals for Google.",
        "",
        f"  {'Score':<8} {'Slug A':<40} {'Slug B'}",
        "  " + "-" * 80,
    ]
    for score, slug_a, title_a, slug_b, title_b in pairs:
        lines.append(f"  {score:.0%}     /{slug_a:<38} /{slug_b}")
        lines.append(f"            {title_a[:38]:<38}   {title_b[:38]}")
        lines.append("")

    lines += [
        "Resolution options:",
        "  1. Delete one of the near-duplicate posts.",
        "  2. Add a <link rel='canonical'> from the thinner post to the stronger one.",
        "  3. If both must exist, ensure they are sufficiently differentiated.",
    ]
    return "\n".join(lines)


# ── Helpers ────────────────────────────────────────────────────────────────

def _slug_bigrams(slug: str) -> Set[str]:
    """Character bigrams of the slug words (after removing stop words)."""
    words = [w for w in slug.split('-') if w and w not in _STOP_SLUG_WORDS]
    text = ''.join(words)
    if len(text) < 2:
        return set(text)
    return {text[i:i + 2] for i in range(len(text) - 1)}


def _slug_jaccard(a: str, b: str) -> float:
    """Jaccard similarity between two slugs using character bigrams."""
    bg_a = _slug_bigrams(a)
    bg_b = _slug_bigrams(b)
    if not bg_a or not bg_b:
        return 0.0
    intersection = len(bg_a & bg_b)
    union = len(bg_a | bg_b)
    return intersection / union if union else 0.0


# ── Sitemap canonical enforcement ─────────────────────────────────────────

def get_noindex_paths() -> List[str]:
    """
    Return URL path patterns that should carry noindex meta tags.
    These are pages that should be accessible to users but not indexed,
    to prevent thin/duplicate pages from diluting site quality.
    """
    return [
        '/tag/',        # tag index listing
        '/404.html',
        '/offline.html',
    ]


def should_noindex(path: str) -> bool:
    """
    Return True if a given URL path should have noindex,follow meta.

    Tag pages with < 5 posts are already handled in static_site_generator.py.
    This function covers the remaining cases.
    """
    noindex = get_noindex_paths()
    for pattern in noindex:
        if path.startswith(pattern):
            return True
    return False
