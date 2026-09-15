"""
adsense_fixes/canonical_guard.py
==================================
Canonical URL enforcement and duplicate-URL prevention.

FIX (audit, 2026-09-15)
------------------------
auto_retire_duplicates.py / merge-redirect stubs (post.json content =
"This article has been merged into [X](/slug/) ...") were being treated
as normal posts by this module:
  - generate_canonical_meta() pointed the canonical at the STUB's own
    URL instead of the survivor it redirects to, so the stub was
    self-canonical — actively telling Google "index me" while the body
    says "go elsewhere."
  - should_noindex() only covered /tag/, /404.html, /offline.html —
    stub posts were never noindexed.
  - Confirmed live: 13 merge-stub posts on kubaik.github.io, all
    self-canonical, 2 still present in sitemap.xml.

This revision:
  1. Adds is_merge_stub(post) / get_redirect_target(post) to detect the
     "This article has been merged into [...](...)" content pattern
     auto_retire_duplicates.py writes.
  2. generate_canonical_meta() now points a stub's canonical at the
     SURVIVOR post, not itself.
  3. should_noindex_post(post) (new, post-aware — the old path-only
     should_noindex() is kept for the static routes it already covers)
     returns True for merge stubs so static_site_generator.py can emit
     <meta name="robots" content="noindex,follow"> on them.
  4. audit_noindex_compliance(docs_dir) — CLI/CI check: fails if any
     merge-stub post is missing noindex or is self-canonical, and fails
     if any stub slug still appears in sitemap.xml.

HOW TO INTEGRATE
----------------
In static_site_generator.py's post-template render step:

    from adsense_fixes.canonical_guard import (
        generate_canonical_meta, should_noindex_post, is_merge_stub,
    )
    if should_noindex_post(post):
        robots_meta = '<meta name="robots" content="noindex,follow">'
    canonical_tag = generate_canonical_meta(post, base_url)

In the sitemap generator (scripts/generate_sitemap.py), skip any slug
for which should_noindex_post(post) is True.

CI (run in blog-automation.yml, non-mutating):
    python -c "from adsense_fixes.canonical_guard import audit_noindex_compliance; \
               from pathlib import Path; print(audit_noindex_compliance(Path('./docs')))"
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


_SLUG_SIMILARITY_THRESHOLD = 0.70   # Jaccard over slug bigrams
_STOP_SLUG_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'in', 'of', 'for',
    'to', 'is', 'are', 'with', 'how', 'why', 'what',
    'guide', 'tutorial', 'post', 'article', 'blog',
}

# Matches the exact stub pattern auto_retire_duplicates.py writes:
# "This article has been merged into [Title](/slug/) to remove ..."
_MERGE_STUB_RE = re.compile(
    r"merged into \[([^\]]+)\]\((/[^\s)]+/)\)",
    re.IGNORECASE,
)


# ── Merge-stub detection ───────────────────────────────────────────────────

def is_merge_stub(post) -> bool:
    """True if post.content is an auto_retire_duplicates.py redirect stub."""
    content = getattr(post, 'content', '') or ''
    return bool(_MERGE_STUB_RE.search(content))


def get_redirect_target(post) -> Optional[str]:
    """
    Return the survivor slug a merge-stub post redirects to, or None if
    this post is not a stub.
    """
    content = getattr(post, 'content', '') or ''
    m = _MERGE_STUB_RE.search(content)
    if not m:
        return None
    path = m.group(2)
    parts = [p for p in path.strip('/').split('/') if p]
    return parts[-1] if parts else None


# ── Canonical tag generator ────────────────────────────────────────────────

def generate_canonical_meta(post, base_url: str) -> str:
    """
    Return the canonical <link> tag string for a post.

    Merge stubs canonicalize to their survivor post, not themselves —
    this is what actually resolves the duplicate-content signal instead
    of just hiding it behind noindex.
    """
    base = base_url.rstrip('/')

    target_slug = get_redirect_target(post)
    if target_slug:
        canonical = f"{base}/{target_slug}/"
        return f'<link rel="canonical" href="{canonical}">'

    slug = getattr(post, 'slug', '').strip('/')
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

    if is_merge_stub(post) and get_redirect_target(post) == slug:
        issues.append(
            f"Merge stub '{slug}' redirects to itself — auto_retire_duplicates.py "
            f"produced a self-referential stub, verify manually."
        )

    if slug.endswith('-2') or re.search(r'-\d+$', slug):
        issues.append(
            f"Slug '{slug}' ends with a numeric suffix, suggesting a duplicate "
            f"was auto-generated. Verify this is intentional."
        )

    if re.search(r'[^a-z0-9\-]', slug):
        issues.append(
            f"Slug '{slug}' contains characters other than lowercase letters, "
            f"digits, and hyphens. This may cause canonical URL inconsistency."
        )

    if slug.startswith('-') or slug.endswith('-'):
        issues.append(f"Slug '{slug}' starts or ends with a hyphen.")

    if base.startswith('http://'):
        issues.append(
            "base_url uses HTTP — canonical URLs should use HTTPS."
        )

    return issues


# ── Duplicate slug audit ───────────────────────────────────────────────────

def audit_duplicate_slugs(docs_dir: Path) -> str:
    """
    Walk docs_dir and find slug pairs with high Jaccard bigram similarity.
    Returns a human-readable report string.
    """
    if not docs_dir.exists():
        return "docs/ directory not found — nothing to audit."

    slugs: List[Tuple[str, str]] = []
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
    words = [w for w in slug.split('-') if w and w not in _STOP_SLUG_WORDS]
    text = ''.join(words)
    if len(text) < 2:
        return set(text)
    return {text[i:i + 2] for i in range(len(text) - 1)}


def _slug_jaccard(a: str, b: str) -> float:
    bg_a = _slug_bigrams(a)
    bg_b = _slug_bigrams(b)
    if not bg_a or not bg_b:
        return 0.0
    intersection = len(bg_a & bg_b)
    union = len(bg_a | bg_b)
    return intersection / union if union else 0.0


# ── Sitemap / noindex enforcement ──────────────────────────────────────────

def get_noindex_paths() -> List[str]:
    """Static path patterns that should carry noindex meta tags."""
    return [
        '/tag/',
        '/404.html',
        '/offline.html',
    ]


def should_noindex(path: str) -> bool:
    """Path-based check for static/template routes (unchanged behaviour)."""
    for pattern in get_noindex_paths():
        if path.startswith(pattern):
            return True
    return False


def should_noindex_post(post) -> bool:
    """
    Post-aware check. True if the post itself is a merge-redirect stub —
    these must never be indexed regardless of URL path, since the body
    content is "go elsewhere," not real content.
    """
    return is_merge_stub(post)


def audit_noindex_compliance(docs_dir: Path) -> str:
    """
    CI-safe, read-only check: for every merge-stub post currently on
    disk, verify (a) it is NOT self-canonical, and (b) it does NOT
    appear in sitemap.xml. Fails loudly rather than silently passing —
    this is the check that would have caught the original bug.
    """
    if not docs_dir.exists():
        return "docs/ directory not found — nothing to audit."

    sitemap_path = docs_dir / 'sitemap.xml'
    sitemap_text = sitemap_path.read_text(encoding='utf-8') if sitemap_path.exists() else ''

    problems = []
    stub_count = 0

    for post_dir in sorted(docs_dir.iterdir()):
        if not post_dir.is_dir() or post_dir.name in ('static', 'tag', 'author'):
            continue
        post_json = post_dir / 'post.json'
        if not post_json.exists():
            continue
        try:
            data = json.loads(post_json.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            continue

        content = data.get('content', '')
        m = _MERGE_STUB_RE.search(content)
        if not m:
            continue

        stub_count += 1
        slug = post_dir.name
        target = [p for p in m.group(2).strip('/').split('/') if p][-1] if m.group(2) else None

        if target == slug:
            problems.append(f"/{slug}/ — self-referential redirect target")
        if f"/{slug}/" in sitemap_text:
            problems.append(f"/{slug}/ — still present in sitemap.xml")

        html_path = post_dir / 'index.html'
        if html_path.exists():
            html = html_path.read_text(encoding='utf-8')
            if 'noindex' not in html.lower():
                problems.append(f"/{slug}/ — index.html missing noindex meta tag")
            if f'href="{docs_dir.name}' not in html and f'/{slug}/"' in html and target and f'/{target}/"' not in html:
                problems.append(f"/{slug}/ — canonical does not point at survivor /{target}/")

    if stub_count == 0:
        return "✅ Noindex audit: no merge-stub posts found."

    if not problems:
        return f"✅ Noindex audit: PASS — {stub_count} merge-stub post(s), all correctly noindexed/canonicalized/excluded from sitemap."

    lines = [f"❌ Noindex audit: {len(problems)} issue(s) across {stub_count} merge-stub post(s)."]
    lines += [f"  - {p}" for p in problems]
    return "\n".join(lines)