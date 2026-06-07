"""
adsense_fixes/link_validator.py
=================================
Validates internal links injected by internal_linker.py.

WHY THIS EXISTS
---------------
Google AdSense's Site Readiness Guide (§5.1) requires:
  "No links to missing or error pages. Internal links must not 404.
   Regularly audit internal links especially in auto-generated content."

`inject_internal_links()` injects links based on token-overlap scoring
but never verifies that the target slug actually exists in docs/.
If a post has been purged via `purge_low_quality_posts()` or was never
saved (generation failed), any internal links pointing to it will 404.

This module provides:
  1. `validate_post_links(post, docs_dir)` — called before save_post()
     to strip any injected links whose targets don't exist yet.
  2. `audit_all_internal_links(docs_dir)` — crawls all post.json files
     and reports any links to missing slugs.

HOW TO INTEGRATE
----------------
In blog_system.py, after inject_internal_links() and before save_post():

    from adsense_fixes.link_validator import validate_post_links

    # Strip links to non-existent slugs (silently)
    removed = validate_post_links(blog_post, blog_system.output_dir)
    if removed:
        print(f"  🔗 Link validator removed {len(removed)} unresolvable link(s): "
              f"{', '.join(removed)}")

Add an 'audit-links' CLI command:
    elif mode == 'audit-links':
        from adsense_fixes.link_validator import audit_all_internal_links
        report = audit_all_internal_links(Path('./docs'))
        print(report)
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple


# Matches Markdown links: [anchor text](/slug/) or [anchor text](/base/slug/)
_MD_INTERNAL_LINK_RE = re.compile(
    r"\[([^\]]+)\]\((/[^\s)]*?/)\)",
    re.UNICODE,
)


def _extract_slug_from_path(path: str) -> str:
    """
    Extract the leaf slug from an internal link path.

    /redis-caching/           → redis-caching
    /blog/redis-caching/      → redis-caching
    /redis-caching            → redis-caching
    """
    # Strip leading/trailing slashes and split
    parts = [p for p in path.strip("/").split("/") if p]
    if not parts:
        return ""
    return parts[-1]


def _get_published_slugs(docs_dir: Path) -> Set[str]:
    """
    Return the set of all valid published slugs in docs_dir.
    A slug is valid if docs_dir/{slug}/post.json exists.
    """
    if not docs_dir.exists():
        return set()
    slugs = set()
    for item in docs_dir.iterdir():
        if item.is_dir() and (item / "post.json").exists():
            slugs.add(item.name)
    return slugs


def validate_post_links(post, docs_dir: Path) -> List[str]:
    """
    Scan post.content for internal Markdown links and remove any
    that point to slugs not present in docs_dir.

    Modifies post.content in-place.
    Returns a list of removed link targets (slugs) for logging.
    """
    published = _get_published_slugs(docs_dir)
    # Also add the current post's own slug so self-links aren't flagged
    # (they won't be added to docs yet but are valid destinations)
    own_slug = getattr(post, "slug", "")
    if own_slug:
        published.add(own_slug)

    content = post.content
    removed: List[str] = []

    def _maybe_strip(match: re.Match) -> str:
        anchor_text = match.group(1)
        path = match.group(2)
        slug = _extract_slug_from_path(path)

        if not slug:
            # Absolute path with no identifiable slug — keep it
            return match.group(0)

        if slug in published:
            # Target exists — keep the link
            return match.group(0)

        # Target does not exist — degrade to plain text, no link
        removed.append(slug)
        return anchor_text

    post.content = _MD_INTERNAL_LINK_RE.sub(_maybe_strip, content)
    return removed


def audit_all_internal_links(docs_dir: Path) -> str:
    """
    Audit all published posts for broken internal links.
    Returns a human-readable report string.

    Run this periodically (e.g. monthly) to catch links broken by post deletion.
    """
    published = _get_published_slugs(docs_dir)
    if not docs_dir.exists():
        return "docs/ directory not found — nothing to audit."

    # (source_slug, anchor_text, target_slug)
    broken: List[Tuple[str, str, str]] = []
    checked_posts = 0

    for post_dir in sorted(docs_dir.iterdir()):
        if not post_dir.is_dir() or post_dir.name in ("static", "tag", "author"):
            continue
        post_json = post_dir / "post.json"
        if not post_json.exists():
            continue

        try:
            with open(post_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            content = data.get("content", "")
            source_slug = post_dir.name
            checked_posts += 1

            for match in _MD_INTERNAL_LINK_RE.finditer(content):
                anchor = match.group(1)
                path = match.group(2)
                target_slug = _extract_slug_from_path(path)
                if target_slug and target_slug not in published:
                    broken.append((source_slug, anchor, target_slug))

        except (json.JSONDecodeError, KeyError):
            continue

    if not broken:
        return (
            f"✅ Internal link audit: PASS\n"
            f"   {checked_posts} posts checked — no broken internal links found."
        )

    lines = [
        f"❌ Internal link audit: {len(broken)} broken link(s) found across "
        f"{checked_posts} posts checked.",
        "",
        f"{'Source Post':<40} {'Anchor Text':<30} {'Missing Target'}",
        "-" * 90,
    ]
    for source, anchor, target in broken:
        lines.append(f"  /{source:<38} {anchor[:28]:<30} /{target}/")

    lines += [
        "",
        "Resolution options:",
        "  1. Re-publish the missing posts.",
        "  2. Remove the links manually from the source posts.",
        "  3. Run the auto-fix: validate_post_links() on each source post.",
    ]
    return "\n".join(lines)
