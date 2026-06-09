#!/usr/bin/env python3
"""
test_schema_validator.py
========================
Local test script that:
  1. Loads every post.json found under docs/
  2. Runs enrich_article_schema() + validate_article_schema() on each post
  3. Prints a per-post report (pass / warnings / failures)
  4. Optionally deletes posts that fail validation, by age (oldest first)
     with a user-supplied count cap.

Usage
-----
  # Validate only (no deletions)
  python test_schema_validator.py

  # Validate and delete the 5 oldest failing posts (dry-run first)
  python test_schema_validator.py --delete 5

  # Validate and delete ALL failing posts (dry-run first)
  python test_schema_validator.py --delete all

  # Skip the dry-run confirmation prompt
  python test_schema_validator.py --delete 5 --yes

  # Point at a different docs directory
  python test_schema_validator.py --docs /path/to/docs --delete 3
"""

import argparse
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Inline copy of adsense_fixes/schema_validator.py logic ────────────────
# (so the script runs standalone without needing the full project installed)

_REQUIRED_ARTICLE_PROPS = {"headline", "author", "datePublished", "image"}

_ISO_DATE_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}(:\d{2})?([+-]\d{2}:\d{2}|Z)?)?$"
)


def _normalise_date(raw: str) -> Optional[str]:
    if not raw:
        return None
    if re.match(r"^\d{4}-\d{2}-\d{2}$", raw):
        return raw
    if "T" in raw:
        return raw.split("T")[0]
    return None


def _build_logo(base_url: str) -> Dict:
    return {
        "@type": "ImageObject",
        "url": f"{base_url}/static/icons/icon-512x512.png",
        "width": 512,
        "height": 512,
    }


def _build_publisher(base_url: str, config: Dict) -> Dict:
    site_name = config.get("site_name", "Tech Blog")
    return {
        "@type": "Organization",
        "name": site_name,
        "url": base_url,
        "logo": _build_logo(base_url),
        "sameAs": [
            "https://twitter.com/KubaiKevin",
            "https://www.linkedin.com/in/kevin-kubai-22b61b37/",
        ],
    }


def enrich_article_schema(
    schemas: List[Dict],
    post_data: Dict,
    base_url: str,
    config: Dict,
) -> List[Dict]:
    """
    Mirrors adsense_fixes.schema_validator.enrich_article_schema but accepts
    a plain post_data dict instead of a BlogPost object.
    """
    base = base_url.rstrip("/")
    slug = post_data.get("slug", "")
    og_image = f"{base}/static/og/{slug}.svg"

    for schema in schemas:
        if schema.get("@type") not in ("Article", "BlogPosting"):
            continue

        # 1. Image with dimensions
        if "image" not in schema:
            schema["image"] = {
                "@type": "ImageObject",
                "url": og_image,
                "width": 1200,
                "height": 630,
                "caption": post_data.get("title", ""),
            }

        # 2. Publisher with logo
        if "publisher" not in schema:
            schema["publisher"] = _build_publisher(base, config)
        elif "logo" not in schema.get("publisher", {}):
            schema["publisher"]["logo"] = _build_logo(base)

        # 3. Normalise dates
        for date_field in ("datePublished", "dateModified"):
            raw = schema.get(date_field, "")
            normalised = _normalise_date(raw)
            if normalised:
                schema[date_field] = normalised

        # 4. inLanguage default
        schema.setdefault("inLanguage", "en-US")

        # 5. mainEntityOfPage
        if "mainEntityOfPage" not in schema:
            schema["mainEntityOfPage"] = {
                "@type": "WebPage",
                "@id": f"{base}/{slug}/",
            }

    return schemas


def validate_article_schema(schemas: List[Dict]) -> List[str]:
    """
    Direct copy of adsense_fixes.schema_validator.validate_article_schema.
    Returns a list of issue strings (empty = valid).
    """
    issues = []

    for schema in schemas:
        schema_type = schema.get("@type", "")

        if schema_type in ("Article", "BlogPosting"):
            for prop in _REQUIRED_ARTICLE_PROPS:
                if prop not in schema:
                    issues.append(
                        f"Article schema missing required property: '{prop}'"
                    )

            for date_field in ("datePublished", "dateModified"):
                date_val = schema.get(date_field, "")
                if date_val and not _ISO_DATE_RE.match(str(date_val)):
                    issues.append(
                        f"'{date_field}' value '{date_val}' is not valid ISO 8601."
                    )

            author = schema.get("author", {})
            if isinstance(author, dict):
                if "name" not in author:
                    issues.append("Article author is missing 'name' property.")
                if "@type" not in author:
                    issues.append(
                        "Article author is missing '@type' (should be 'Person')."
                    )
            elif not author:
                issues.append("Article 'author' is empty.")

            image = schema.get("image", {})
            if isinstance(image, dict):
                if "url" not in image:
                    issues.append("Article image is missing 'url'.")
            elif not image:
                issues.append(
                    "Article schema missing 'image' — required for Rich Results."
                )

            headline = schema.get("headline", "")
            if headline and len(headline) > 110:
                issues.append(
                    f"Article headline is {len(headline)} chars — Google recommends < 110."
                )

        elif schema_type == "BreadcrumbList":
            items = schema.get("itemListElement", [])
            if not items:
                issues.append("BreadcrumbList has no itemListElement entries.")
            for item in items:
                if "item" not in item or "name" not in item:
                    issues.append(
                        "BreadcrumbList item is missing 'name' or 'item' URL."
                    )

    return issues


# ── Schema builder (mirrors static_site_generator._generate_article_schema) ──

def build_schemas_from_post(post_data: Dict, base_url: str) -> List[Dict]:
    """
    Reconstruct the same schema list that _generate_article_schema() would build,
    using only data from post.json.
    """
    slug = post_data.get("slug", "")
    title = post_data.get("title", "")
    meta_description = post_data.get("meta_description", "")
    created_at = post_data.get("created_at", "")
    updated_at = post_data.get("updated_at", created_at)
    seo_keywords = post_data.get("seo_keywords", [])
    content = post_data.get("content", "")
    word_count = post_data.get("word_count") or len(content.split())
    reading_time = max(1, round(word_count / 200))

    schemas = [
        {
            "@type": "Article",
            "@id": f"{base_url}/{slug}/#article",
            "headline": title,
            "description": meta_description,
            "datePublished": created_at,
            "dateModified": updated_at,
            "wordCount": word_count,
            "timeRequired": f"PT{reading_time}M",
            "articleSection": "Technology",
            "inLanguage": "en-US",
            "author": {
                "@type": "Person",
                "@id": f"{base_url}/about/#author",
                "name": "Kubai Kevin",
                "jobTitle": "Software Developer",
                "url": f"{base_url}/about/",
                "sameAs": [
                    "https://www.linkedin.com/in/kevin-kubai-22b61b37/",
                    "https://twitter.com/KubaiKevin",
                ],
                "knowsAbout": [
                    "Python", "Node.js", "TypeScript", "AWS",
                    "Backend Systems", "AI", "Machine Learning",
                ],
            },
            "publisher": {
                "@type": "Organization",
                "name": "Tech Blog",
                "url": base_url,
            },
            "mainEntityOfPage": {
                "@type": "WebPage",
                "@id": f"{base_url}/{slug}/",
            },
            "keywords": ", ".join(seo_keywords[:8]) if seo_keywords else "",
        },
        {
            "@type": "BreadcrumbList",
            "itemListElement": [
                {
                    "@type": "ListItem",
                    "position": 1,
                    "name": "Home",
                    "item": f"{base_url}/",
                },
                {
                    "@type": "ListItem",
                    "position": 2,
                    "name": title,
                    "item": f"{base_url}/{slug}/",
                },
            ],
        },
    ]
    return schemas


# ── Post loader ───────────────────────────────────────────────────────────────

def load_posts(docs_dir: Path) -> List[Tuple[str, Dict]]:
    """
    Return list of (slug, post_data) sorted oldest-first by created_at.
    Only directories containing a post.json are included.
    """
    results = []
    for post_dir in docs_dir.iterdir():
        if not post_dir.is_dir() or post_dir.name == "static":
            continue
        post_json = post_dir / "post.json"
        if not post_json.exists():
            continue
        try:
            with open(post_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            results.append((post_dir.name, data))
        except Exception as e:
            print(f"  ⚠️  Could not load {post_dir.name}/post.json: {e}")

    # Sort oldest first so --delete N removes the oldest N failures
    results.sort(
        key=lambda x: x[1].get("created_at", "1970-01-01")
    )
    return results


# ── Validation runner ─────────────────────────────────────────────────────────

def _status_icon(issues: List[str]) -> str:
    if not issues:
        return "✅"
    # Distinguish warnings (headline length, missing image URL) from hard failures
    hard = [i for i in issues if "missing required" in i or "missing 'name'" in i
            or "missing '@type'" in i or "BreadcrumbList" in i
            or "is not valid ISO" in i or "author' is empty" in i]
    return "🛑" if hard else "⚠️ "


def run_validation(
    docs_dir: Path,
    base_url: str,
    config: Dict,
    verbose: bool = True,
) -> List[Tuple[str, Dict, List[str]]]:
    """
    Validate all posts. Returns list of (slug, post_data, issues) for
    every post — including those that pass (issues=[]).
    """
    posts = load_posts(docs_dir)
    if not posts:
        print(f"No posts found in {docs_dir}")
        return []

    results = []
    pass_count = warn_count = fail_count = 0

    SEP = "─" * 70
    print(f"\n{SEP}")
    print(f"Schema Validator — {len(posts)} post(s) in {docs_dir}")
    print(SEP)

    for slug, post_data in posts:
        schemas = build_schemas_from_post(post_data, base_url)
        schemas = enrich_article_schema(schemas, post_data, base_url, config)
        issues = validate_article_schema(schemas)

        icon = _status_icon(issues)
        title = post_data.get("title", slug)[:65]
        created = post_data.get("created_at", "?")[:10]

        if verbose:
            print(f"\n{icon}  [{created}]  {title}")
            print(f"     slug: {slug}")
            if issues:
                for issue in issues:
                    print(f"     • {issue}")
            else:
                print(f"     All required properties present.")

        hard_issues = [i for i in issues if any(kw in i for kw in (
            "missing required", "missing 'name'", "missing '@type'",
            "BreadcrumbList", "is not valid ISO", "author' is empty",
        ))]

        if not issues:
            pass_count += 1
        elif hard_issues:
            fail_count += 1
        else:
            warn_count += 1

        results.append((slug, post_data, issues))

    print(f"\n{SEP}")
    print(
        f"Summary: {pass_count} passed  |  "
        f"{warn_count} warnings  |  {fail_count} hard failures"
    )
    print(SEP)
    return results


# ── Deletion helper ───────────────────────────────────────────────────────────

def _is_hard_failure(issues: List[str]) -> bool:
    return any(kw in i for i in issues for kw in (
        "missing required", "missing 'name'", "missing '@type'",
        "BreadcrumbList", "is not valid ISO", "author' is empty",
    ))


def delete_failing_posts(
    results: List[Tuple[str, Dict, List[str]]],
    docs_dir: Path,
    count,           # int or "all"
    dry_run: bool = True,
    yes: bool = False,
) -> None:
    """
    Delete the oldest `count` posts that have hard validation failures.
    Posts are already sorted oldest-first (load_posts sorts by created_at).
    """
    failing = [(slug, data, issues) for slug, data, issues in results
               if _is_hard_failure(issues)]

    if not failing:
        print("\n✅  No hard-failure posts to delete.")
        return

    # How many to delete
    if count == "all":
        to_delete = failing
    else:
        to_delete = failing[:int(count)]

    SEP = "─" * 70
    label = "DRY RUN — " if dry_run else ""
    print(f"\n{SEP}")
    print(f"{label}Posts queued for deletion ({len(to_delete)} of {len(failing)} failing):")
    print(SEP)

    for slug, data, issues in to_delete:
        created = data.get("created_at", "?")[:10]
        title = data.get("title", slug)[:60]
        hard = [i for i in issues if any(kw in i for kw in (
            "missing required", "missing 'name'", "missing '@type'",
            "BreadcrumbList", "is not valid ISO", "author' is empty",
        ))]
        print(f"  🗑  [{created}]  {title}")
        print(f"       slug: {slug}")
        for h in hard:
            print(f"       reason: {h}")

    if dry_run:
        print(f"\n  This is a DRY RUN — nothing has been deleted.")
        if not yes:
            answer = input(
                f"\n  Proceed with deleting these {len(to_delete)} post(s)? [y/N] "
            ).strip().lower()
            if answer not in ("y", "yes"):
                print("  Aborted.")
                return
        # Re-run as a real delete
        delete_failing_posts(
            results, docs_dir, count, dry_run=False, yes=True
        )
        return

    # Actual deletion
    deleted = 0
    for slug, data, _ in to_delete:
        post_dir = docs_dir / slug
        if post_dir.exists():
            shutil.rmtree(post_dir)
            print(f"  Deleted: {post_dir}")
            deleted += 1
        else:
            print(f"  ⚠️  Directory not found (already gone?): {post_dir}")

    print(
        f"\n  Deleted {deleted} post director{'y' if deleted == 1 else 'ies'}.")
    print(f"  Run 'python blog_system.py build' to regenerate the site.\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate schema markup on existing docs/ posts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--docs",
        default="./docs",
        help="Path to the docs directory (default: ./docs)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Override base_url (default: read from config.yaml or use https://kubaik.github.io)",
    )
    parser.add_argument(
        "--delete",
        metavar="N|all",
        default=None,
        help=(
            "Delete failing posts. Pass a number (oldest N first) or 'all'. "
            "A dry-run preview is always shown first."
        ),
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt when deleting.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print the summary and failures (suppress per-post detail for passing posts).",
    )
    parser.add_argument(
        "--failures-only",
        action="store_true",
        help="Only print posts that have issues.",
    )
    return parser.parse_args()


def load_config(config_path: str = "config.yaml") -> Dict:
    try:
        import yaml  # type: ignore
        if Path(config_path).exists():
            with open(config_path, "r") as f:
                return yaml.safe_load(f) or {}
    except ImportError:
        pass

    # Fallback: read base_url from config.yaml manually without PyYAML
    if Path(config_path).exists():
        with open(config_path, "r") as f:
            for line in f:
                m = re.match(
                    r"^\s*base_url\s*:\s*['\"]?([^'\"\n]+)['\"]?", line)
                if m:
                    return {"base_url": m.group(1).strip(), "site_name": "Tech Blog"}
    return {}


def main():
    args = parse_args()
    docs_dir = Path(args.docs).resolve()

    if not docs_dir.exists():
        print(f"Error: docs directory not found: {docs_dir}")
        sys.exit(1)

    config = load_config()
    base_url = (
        args.base_url
        or config.get("base_url", "https://kubaik.github.io")
    ).rstrip("/")

    print(f"Base URL : {base_url}")
    print(f"Docs dir : {docs_dir}")

    # Decide verbosity
    verbose = not args.quiet

    results = run_validation(docs_dir, base_url, config, verbose=verbose)

    if args.failures_only and verbose:
        # Re-print only the ones with issues (already printed above, but
        # --failures-only + --quiet makes the most sense; this is a no-op
        # for the summary which always prints)
        pass

    if not results:
        sys.exit(0)

    # Deletion
    if args.delete is not None:
        raw = args.delete.strip().lower()
        if raw == "all":
            count = "all"
        else:
            try:
                count = int(raw)
                if count <= 0:
                    raise ValueError
            except ValueError:
                print(
                    f"Error: --delete must be a positive integer or 'all', got: {args.delete!r}")
                sys.exit(1)

        delete_failing_posts(
            results,
            docs_dir,
            count=count,
            dry_run=True,
            yes=args.yes,
        )

    # Exit code: 1 if any hard failures remain (after potential deletion)
    remaining_failures = [
        (slug, data, issues) for slug, data, issues in results
        if _is_hard_failure(issues)
    ]
    if remaining_failures and args.delete is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
