#!/usr/bin/env python3
"""
adsense_fixes/author_page.py
============================
Emit /author/kubai-kevin/index.html with a full ProfilePage + Person
JSON-LD graph.

WHY THIS EXISTS
---------------
AdSense + Helpful Content review treat a real, verifiable author with
an external footprint (LinkedIn, GitHub, Twitter) as an E-E-A-T signal.
static_site_generator.py's _generate_author_page() already writes a
basic author page, but it:

  - hardcodes a small AUTHOR_PAGE_TEMPLATE that does not include a
    ProfilePage wrapper (only a bare Person), so Google's Rich Results
    Test does not classify it as a profile page.
  - has no knowsAbout array in the emitted schema, so nothing tells
    search engines what the author is actually an authority on.
  - lists only 20 posts, which on a 600+ post site means the page is
    a stub relative to the corpus it represents.
  - duplicates the sameAs list already present on /about/, so any
    future change to the author's social presence has two places to
    update.

This module fixes all four by being the single source of truth for the
author's schema graph — static_site_generator.py calls it, and any other
consumer (schema injection into post pages, sitemap priority boosts for
/author/*, etc.) can import the same AUTHOR_SCHEMA constant instead of
copying the dict.

HOW TO INTEGRATE
----------------
In static_site_generator.py, replace the body of _generate_author_page()
with a single call:

    from adsense_fixes.author_page import generate_author_page
    ...
    def _generate_author_page(self, posts):
        generate_author_page(posts=posts, docs_dir=Path("./docs"),
                             config=self.blog_system.config)

CLI (for one-off regeneration outside a full build):
    python adsense_fixes/author_page.py --docs-dir ./docs
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ── Author identity — single source of truth ───────────────────────────
# Everything that needs "who wrote this site" reads from here. If the
# author's LinkedIn URL or GitHub handle changes, it changes in one place
# and every schema block, page footer, and sameAs array follows.
AUTHOR_NAME = "Kubai Kevin"
AUTHOR_GIVEN_NAME = "Kevin"
AUTHOR_FAMILY_NAME = "Kubai"
AUTHOR_JOB_TITLE = "Software Developer"
AUTHOR_LOCALITY = "Nairobi"
AUTHOR_COUNTRY = "KE"
AUTHOR_EMAIL = "aiblogauto@gmail.com"
AUTHOR_DESCRIPTION = (
    "Software developer based in Nairobi, Kenya with a background in "
    "Python, Node.js, and AWS, focused on fintech backends and AI "
    "integration. Curates and edits an AI-assisted publication covering "
    "backend systems, AI, and developer careers."
)

# The topics the author has verifiable, demonstrated expertise in. Kept
# in sync with the "Technologies covered on this site" section on /about/
# and the technologies actually named across the published corpus.
AUTHOR_KNOWS_ABOUT = [
    "Python",
    "Node.js",
    "TypeScript",
    "AWS Lambda",
    "PostgreSQL",
    "Redis",
    "Machine Learning",
    "LLMs",
    "API Design",
    "Fintech Systems",
    "Backend Engineering",
    "Android Development",
]

# External profiles that prove the person is real and lets search engines
# cross-reference identity. Order matters only for how the page renders
# the link row — schema treats it as an unordered set.
AUTHOR_SAME_AS = [
    "https://www.linkedin.com/in/kevin-kubai-22b61b37/",
    "https://twitter.com/KubaiKevin",
    "https://github.com/kubaik",
]

# Public-facing link labels, matched by position to AUTHOR_SAME_AS.
# Kept separate from the URL list so the schema's sameAs stays a clean
# URL array (no labels) while the visible page keeps readable link text.
_AUTHOR_LINK_LABELS = {
    "https://www.linkedin.com/in/kevin-kubai-22b61b37/": "LinkedIn",
    "https://twitter.com/KubaiKevin": "Twitter",
    "https://github.com/kubaik": "GitHub",
}

_DEFAULT_BASE_URL = "https://kubaik.github.io"


def author_schema(base_url: str) -> dict:
    """Return the Person schema block used everywhere on the site.

    Callers that need to embed the author into a larger schema graph
    (e.g. post pages, ProfilePage, Organization's founder) should import
    this function rather than duplicating the dict.
    """
    base_url = base_url.rstrip("/")
    return {
        "@type": "Person",
        "@id": f"{base_url}/about/#author",
        "name": AUTHOR_NAME,
        "givenName": AUTHOR_GIVEN_NAME,
        "familyName": AUTHOR_FAMILY_NAME,
        "jobTitle": AUTHOR_JOB_TITLE,
        "description": AUTHOR_DESCRIPTION,
        "url": f"{base_url}/about/",
        "email": AUTHOR_EMAIL,
        "address": {
            "@type": "PostalAddress",
            "addressLocality": AUTHOR_LOCALITY,
            "addressCountry": AUTHOR_COUNTRY,
        },
        "sameAs": list(AUTHOR_SAME_AS),
        "knowsAbout": list(AUTHOR_KNOWS_ABOUT),
        "worksFor": {
            "@type": "Organization",
            "name": AUTHOR_NAME,
            "url": f"{base_url}/",
        },
    }


def profile_page_schema(base_url: str) -> dict:
    """Return the ProfilePage wrapper — the top-level schema for /author/."""
    base_url = base_url.rstrip("/")
    return {
        "@context": "https://schema.org",
        "@type": "ProfilePage",
        "@id": f"{base_url}/author/kubai-kevin/#profilepage",
        "url": f"{base_url}/author/kubai-kevin/",
        "name": f"About {AUTHOR_NAME}",
        "inLanguage": "en-US",
        "primaryImageOfPage": {
            "@type": "ImageObject",
            "url": f"{base_url}/static/icons/icon-192x192.png",
        },
        "mainEntity": author_schema(base_url),
    }


def _load_posts_for_author_page(posts, docs_dir: Path) -> list[dict]:
    """
    Accept either pre-loaded BlogPost objects (from static_site_generator's
    generate_site() loop) or an empty list, in which case scan docs_dir for
    post.json files. The scan path exists so this module is runnable
    standalone (CLI) without booting the whole pipeline.
    """
    out: list[dict] = []
    if posts:
        for p in posts:
            out.append({
                "slug": p.slug,
                "title": p.title,
                "created_at": p.created_at,
                "meta_description": getattr(p, "meta_description", "") or "",
            })
        return out

    if not docs_dir.exists():
        return out
    for post_dir in docs_dir.iterdir():
        if not post_dir.is_dir() or post_dir.name in ("static", "tag", "author", "page"):
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
        out.append({
            "slug": post_dir.name,
            "title": data.get("title", ""),
            "created_at": data.get("created_at", ""),
            "meta_description": data.get("meta_description", ""),
        })
    return out


def _render_post_list(posts: list[dict], base_path: str, limit: int) -> str:
    """Render the recent-posts <ul> shown on the author page."""
    if not posts:
        return '<p class="author-empty">No posts yet.</p>'
    ordered = sorted(posts, key=lambda p: p.get("created_at", ""), reverse=True)
    items = []
    for p in ordered[:limit]:
        date_str = (p.get("created_at", "") or "").split("T")[0]
        items.append(
            f'<li><a href="{base_path}/{p["slug"]}/">{p["title"]}</a>'
            f'<span class="author-post-date">{date_str}</span></li>'
        )
    return f'<ul class="author-post-list">\n' + "\n".join(items) + "\n</ul>"


def _render_links() -> str:
    parts = []
    for url in AUTHOR_SAME_AS:
        label = _AUTHOR_LINK_LABELS.get(url, url)
        parts.append(
            f'<a href="{url}" target="_blank" rel="noopener noreferrer">{label}</a>'
        )
    return " · ".join(parts)


def generate_author_page(
    posts: list | None = None,
    docs_dir: Path = Path("./docs"),
    config: dict | None = None,
) -> Path:
    """
    Write /author/kubai-kevin/index.html.

    Returns the path written so callers can log it. Idempotent — safe to
    re-run on every build.
    """
    config = config or {}
    base_url = (config.get("base_url") or _DEFAULT_BASE_URL).rstrip("/")
    base_path = config.get("base_path", "")
    site_name = config.get("site_name", "Tech Blog")
    year = datetime.now().year

    posts_data = _load_posts_for_author_page(posts, docs_dir)
    post_list_html = _render_post_list(posts_data, base_path, limit=100)
    links_html = _render_links()

    schema_json = json.dumps(profile_page_schema(base_url), indent=2, ensure_ascii=False)

    # NOTE ON STYLING: kept minimal and self-contained (inline <style>)
    # because the author page is a low-traffic SEO surface — pulling in
    # the full site stylesheet would add a render-blocking request for
    # a page whose primary job is to be machine-readable.
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{AUTHOR_NAME} — {AUTHOR_JOB_TITLE} · {site_name}</title>
<meta name="description" content="{AUTHOR_DESCRIPTION}">
<meta name="author" content="{AUTHOR_NAME}">
<link rel="canonical" href="{base_url}/author/kubai-kevin/">
<meta name="robots" content="index, follow">
<meta name="base-path" content="{base_path}">
<meta property="og:type" content="profile">
<meta property="og:title" content="{AUTHOR_NAME} — {AUTHOR_JOB_TITLE}">
<meta property="og:description" content="{AUTHOR_DESCRIPTION}">
<meta property="og:url" content="{base_url}/author/kubai-kevin/">
<meta property="og:image" content="{base_url}/static/icons/icon-512x512.png">
<meta name="twitter:card" content="summary">
<meta name="twitter:title" content="{AUTHOR_NAME}">
<meta name="twitter:description" content="{AUTHOR_DESCRIPTION}">
<script type="application/ld+json">
{schema_json}
</script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         max-width: 720px; margin: 2rem auto; padding: 0 1.25rem; color: #333;
         line-height: 1.6; }}
  header h1 {{ font-size: 1.75rem; margin-bottom: 0.25rem; }}
  .author-job {{ color: #6366f1; font-weight: 600; margin-bottom: 1rem; }}
  .author-links a {{ color: #6366f1; text-decoration: none; }}
  .author-links a:hover {{ text-decoration: underline; }}
  .author-knows {{ margin: 1.5rem 0; }}
  .author-knows span {{ display: inline-block; background: #f0f4ff;
      border: 1px solid #c7d2fe; border-radius: 6px; padding: 0.2rem 0.6rem;
      font-size: 0.82rem; color: #3730a3; margin: 0.15rem 0.25rem 0.15rem 0; }}
  .author-post-list {{ list-style: none; padding: 0; }}
  .author-post-list li {{ padding: 0.6rem 0; border-bottom: 1px solid #eee; }}
  .author-post-list li a {{ color: #333; text-decoration: none; }}
  .author-post-list li a:hover {{ color: #6366f1; }}
  .author-post-date {{ display: block; font-size: 0.78rem; color: #999;
      margin-top: 0.15rem; }}
  footer {{ margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #eee;
      font-size: 0.85rem; color: #888; }}
</style>
</head>
<body>
<header>
  <h1>{AUTHOR_NAME}</h1>
  <p class="author-job">{AUTHOR_JOB_TITLE} · {AUTHOR_LOCALITY}, Kenya</p>
  <p>{AUTHOR_DESCRIPTION}</p>
  <p class="author-links">
    {links_html}
    · <a href="{base_path}/about/">Full bio and editorial process →</a>
  </p>
</header>

<section class="author-knows">
  <h2 style="font-size:1rem;">Covers</h2>
  {''.join(f'<span>{t}</span>' for t in AUTHOR_KNOWS_ABOUT)}
</section>

<section>
  <h2 style="font-size:1.1rem;">Recent articles</h2>
  {post_list_html}
</section>

<footer>
  <p>&copy; {year} {site_name} · Written by {AUTHOR_NAME}</p>
</footer>
</body>
</html>
"""

    author_dir = docs_dir / "author" / "kubai-kevin"
    author_dir.mkdir(parents=True, exist_ok=True)
    out_path = author_dir / "index.html"
    out_path.write_text(html, encoding="utf-8")

    print(f"Generated author page ({len(posts_data)} posts): {out_path}")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate /author/kubai-kevin/ with full ProfilePage schema."
    )
    parser.add_argument("--docs-dir", default="./docs")
    parser.add_argument(
        "--base-url",
        default=_DEFAULT_BASE_URL,
        help="Override config.yaml's base_url for a one-off run.",
    )
    args = parser.parse_args()
    cfg = {"base_url": args.base_url}
    try:
        import yaml
        cfg_path = Path("config.yaml")
        if cfg_path.exists():
            cfg.update(yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {})
    except Exception:
        pass
    generate_author_page(posts=None, docs_dir=Path(args.docs_dir), config=cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())