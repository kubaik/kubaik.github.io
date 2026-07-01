#!/usr/bin/env python3
"""
scripts/title_validator.py

Validates and fixes article titles during content generation.

The site has truncated titles like:
  "Side-channel prompt attacks: the 60ms latency you"
  "AI observability: real-time dashboards without the"
  "AWS Local Zones vs LocalStack: AI microservices in"

This script:
  1. Validates titles against length and completeness rules
  2. Stores full titles in a `full_title` metadata field
  3. Generates a SERP-optimised display title (max 60 chars)
  4. Provides a batch fix mode for existing posts

Usage (single title):
    python scripts/title_validator.py --title "Side-channel prompt attacks: the 60ms latency you should measure"

Usage (batch fix from JSON feed of posts):
    python scripts/title_validator.py --batch posts_manifest.json --output fixed_titles.json
"""

import argparse
import json
import re
import sys
from typing import Optional

MAX_DISPLAY_TITLE = 60
MAX_FULL_TITLE = 110

# Words that should not end a display title
WEAK_ENDINGS = {
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "and",
    "or", "but", "with", "by", "from", "as", "is", "are", "was",
    "you", "your", "we", "our", "this", "that", "it", "its",
    "which", "when", "where", "how", "why", "what", "who",
}

# Prepositions and conjunctions to avoid ending on
PREPOSITIONS = {"in", "on", "at", "to", "for", "of",
                "with", "by", "from", "into", "through", "about"}


class TitleValidationError(Exception):
    pass


def last_word(title: str) -> str:
    cleaned = title.rstrip(".,;:!?\"'…").strip()
    parts = cleaned.split()
    return parts[-1].lower() if parts else ""


def is_truncated(title: str) -> bool:
    """Heuristic: title appears truncated if it ends with a weak word."""
    lw = last_word(title)
    return lw in WEAK_ENDINGS


def truncate_at_word_boundary(text: str, max_len: int, suffix: str = "…") -> str:
    """Truncate text at a word boundary without cutting words."""
    if len(text) <= max_len:
        return text
    # Find last space before max_len - len(suffix)
    cut = max_len - len(suffix)
    pos = text.rfind(" ", 0, cut)
    if pos == -1:
        pos = cut
    truncated = text[:pos].rstrip(".,;:–—")
    return truncated + suffix


def generate_display_title(full_title: str, max_len: int = MAX_DISPLAY_TITLE) -> str:
    """
    Generate a display title that:
    - Is at most max_len characters
    - Does not end with a weak/truncated word
    - Ends at a natural break point (colon, dash) if possible
    """
    if len(full_title) <= max_len and not is_truncated(full_title):
        return full_title

    # If there's a colon or dash, try to use just the first part
    for sep in [": ", " — ", " – ", " - "]:
        if sep in full_title:
            first_part = full_title.split(sep)[0].strip()
            if MAX_DISPLAY_TITLE * 0.6 <= len(first_part) <= max_len:
                return first_part

    # Fall back to truncation at word boundary
    display = truncate_at_word_boundary(full_title, max_len)

    # Ensure it doesn't end weakly after truncation
    if is_truncated(display.rstrip("…").strip()):
        # Shorten further to remove the weak ending
        words = display.rstrip("… ").split()
        while words and words[-1].lower().rstrip(".,;:") in WEAK_ENDINGS:
            words.pop()
        display = " ".join(words) + "…"

    return display


def validate_title(title: str, full_title: Optional[str] = None) -> dict:
    """
    Validate a title and return a result dict:
    {
        "valid": bool,
        "display_title": str,
        "full_title": str,
        "errors": [str],
        "warnings": [str]
    }
    """
    errors = []
    warnings = []

    effective_full = full_title or title

    # Check for obvious truncation
    if is_truncated(title):
        errors.append(
            f"TRUNCATED_TITLE: Ends with weak word '{last_word(title)}'. "
            "The content generation pipeline cut the title too short. "
            "Fix: store the complete title in `full_title` and generate display title with this script."
        )

    # Length checks on display title
    if len(title) < 20:
        errors.append(f"TITLE_TOO_SHORT: {len(title)} chars (min 20)")

    if len(title) > MAX_DISPLAY_TITLE:
        warnings.append(
            f"TITLE_TOO_LONG_FOR_SERP: Display title is {len(title)} chars. "
            f"Google typically shows ~60 chars in SERPs. "
            "Consider shortening."
        )

    # All caps
    if title == title.upper() and len(title) > 5:
        errors.append("TITLE_ALL_CAPS: All-caps titles look spammy in SERPs.")

    # Repetitive "2026" usage
    if title.count("2026") > 1:
        warnings.append(
            "TITLE_YEAR_REPETITION: '2026' appears multiple times in the title.")

    # Clickbait number padding (excessive specificity)
    numbers = re.findall(r"\b\d+%\b", title)
    if len(numbers) > 2:
        warnings.append(
            f"TITLE_NUMBER_OVERLOAD: {len(numbers)} percentage claims in title. "
            "Looks clickbait-heavy. Reduce to at most 1-2."
        )

    display_title = generate_display_title(effective_full)

    return {
        "valid": len(errors) == 0,
        "display_title": display_title,
        "full_title": effective_full,
        "errors": errors,
        "warnings": warnings,
    }


def batch_validate(posts: list[dict]) -> list[dict]:
    """
    Validate titles for a batch of posts.
    Each post dict should have at minimum: {"slug": "...", "title": "...", "full_title": "..."}
    Returns list of post dicts with "title_validation" key added.
    """
    results = []
    for post in posts:
        title = post.get("title", "")
        full_title = post.get("full_title", title)
        result = validate_title(title, full_title)
        post_copy = dict(post)
        post_copy["title_validation"] = result
        # Auto-fix: replace display title with corrected version
        if not result["valid"] or post_copy.get("title") != result["display_title"]:
            post_copy["display_title_suggested"] = result["display_title"]
        results.append(post_copy)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and fix article titles")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--title", help="Validate a single title string")
    group.add_argument(
        "--batch", help="Path to JSON file with array of post objects")
    parser.add_argument(
        "--full-title", help="Full title for single-title mode (if different from display)")
    parser.add_argument("--output", help="Output JSON file for batch mode")
    args = parser.parse_args()

    if args.title:
        result = validate_title(args.title, args.full_title)
        print(f"\nDisplay title: {result['display_title']}")
        print(f"Full title:    {result['full_title']}")
        print(f"Valid:         {result['valid']}")
        if result["errors"]:
            print("\nErrors:")
            for e in result["errors"]:
                print(f"  ❌ {e}")
        if result["warnings"]:
            print("\nWarnings:")
            for w in result["warnings"]:
                print(f"  ⚠  {w}")
        return 0 if result["valid"] else 1

    # Batch mode
    batch_path = args.batch
    with open(batch_path) as f:
        posts = json.load(f)

    results = batch_validate(posts)

    invalid_count = sum(
        1 for p in results if not p["title_validation"]["valid"])
    warning_count = sum(
        1 for p in results if p["title_validation"]["warnings"])

    print(f"\nBatch validation: {len(results)} posts")
    print(f"  ❌ Invalid titles: {invalid_count}")
    print(f"  ⚠  Titles with warnings: {warning_count}")

    for post in results:
        val = post["title_validation"]
        if not val["valid"]:
            print(f"\n  [{post.get('slug', '?')}]")
            print(f"    Current:   {post.get('title', '')}")
            print(f"    Suggested: {val['display_title']}")
            for e in val["errors"]:
                print(f"    ❌ {e}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults written to: {args.output}")

    return 0 if invalid_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
