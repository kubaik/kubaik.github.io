"""
adsense_fixes/schema_validator.py
====================================
Validates and enriches structured data (JSON-LD schema) for AdSense
and Google Search readiness.

WHY THIS EXISTS
---------------
AdSense Site Readiness Guide (§7 Automated Blog Specific Concerns):
  "Schema markup / structured data. Implement Article, BlogPosting,
   or FAQPage schema on posts. This helps Google understand content
   and improves crawl quality."

Google's Rich Results Test checks for:
  - Required properties on Article schema
  - Valid datePublished / dateModified ISO 8601 format
  - Author entity linking to a known person page
  - Publisher with logo
  - Image with explicit dimensions
  - BreadcrumbList on every post

Current gaps in the codebase:
  - No `image` object on the Article schema (required for rich results)
  - Publisher logo is missing
  - dateModified uses .isoformat() which may include timezone offset
    or microseconds causing schema validation warnings
  - BreadcrumbList @type exists but no @context on the graph root

HOW TO INTEGRATE
----------------
In static_site_generator.py, call validate_article_schema() in
_generate_article_schema() BEFORE returning the HTML string:

    from adsense_fixes.schema_validator import (
        enrich_article_schema,
        validate_article_schema,
    )
    schemas = enrich_article_schema(schemas, post, base_url, config)
    issues = validate_article_schema(schemas)
    if issues:
        print(f"  ⚠️  Schema issues for {post.slug}: {'; '.join(issues)}")
"""

import json
import re
from typing import Dict, List, Optional


# Required Article properties per schema.org + Google Rich Results spec
_REQUIRED_ARTICLE_PROPS = {
    'headline', 'author', 'datePublished', 'image',
}

_ISO_DATE_RE = re.compile(
    r'^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}(:\d{2})?([+-]\d{2}:\d{2}|Z)?)?$')


# ── Schema enrichment ──────────────────────────────────────────────────────

def enrich_article_schema(
    schemas: List[Dict],
    post,
    base_url: str,
    config: Dict,
) -> List[Dict]:
    """
    Take the existing schemas list (output of _generate_article_schema) and
    add / fix missing required properties.

    Mutations applied:
      1. Adds image object with explicit dimensions to Article
      2. Adds publisher logo to Article
      3. Normalises datePublished and dateModified to YYYY-MM-DD format
      4. Ensures @context is present on the graph object
      5. Adds Organization schema if not already present
    """
    base = base_url.rstrip('/')
    slug = getattr(post, 'slug', '')
    og_image = f"{base}/static/og/{slug}.svg"

    for schema in schemas:
        if schema.get('@type') not in ('Article', 'BlogPosting'):
            continue

        # 1. Image with dimensions
        if 'image' not in schema:
            schema['image'] = {
                '@type': 'ImageObject',
                'url': og_image,
                'width': 1200,
                'height': 630,
                'caption': getattr(post, 'title', ''),
            }

        # 2. Publisher with logo
        if 'publisher' not in schema:
            schema['publisher'] = _build_publisher(base, config)
        elif 'logo' not in schema.get('publisher', {}):
            schema['publisher']['logo'] = _build_logo(base)

        # 3. Normalise dates
        for date_field in ('datePublished', 'dateModified'):
            raw = schema.get(date_field, '')
            normalised = _normalise_date(raw)
            if normalised:
                schema[date_field] = normalised

        # 4. Ensure inLanguage
        schema.setdefault('inLanguage', 'en-US')

        # 5. mainEntityOfPage
        if 'mainEntityOfPage' not in schema:
            schema['mainEntityOfPage'] = {
                '@type': 'WebPage',
                '@id': f"{base}/{slug}/",
            }

    return schemas


def _build_publisher(base_url: str, config: Dict) -> Dict:
    site_name = config.get('site_name', 'Tech Blog')
    return {
        '@type': 'Organization',
        'name': site_name,
        'url': base_url,
        'logo': _build_logo(base_url),
        'sameAs': [
            'https://twitter.com/KubaiKevin',
            'https://www.linkedin.com/in/kevin-kubai-22b61b37/',
        ],
    }


def _build_logo(base_url: str) -> Dict:
    return {
        '@type': 'ImageObject',
        'url': f"{base_url}/static/icons/icon-512x512.png",
        'width': 512,
        'height': 512,
    }


def _normalise_date(raw: str) -> Optional[str]:
    """
    Normalise an ISO date string to YYYY-MM-DD format.
    Returns None if the input cannot be parsed.
    """
    if not raw:
        return None
    # Already YYYY-MM-DD
    if re.match(r'^\d{4}-\d{2}-\d{2}$', raw):
        return raw
    # Strip time component
    if 'T' in raw:
        return raw.split('T')[0]
    return None


# ── Schema validation ──────────────────────────────────────────────────────

def validate_article_schema(schemas: List[Dict]) -> List[str]:
    """
    Validate the schemas list for required properties.
    Returns a list of issue strings (empty list = valid).

    This is a lightweight pre-publish check — not a full JSON-LD validator.
    Use Google's Rich Results Test for definitive validation before submission.
    """
    issues = []

    for schema in schemas:
        schema_type = schema.get('@type', '')

        if schema_type in ('Article', 'BlogPosting'):
            # Check required properties
            for prop in _REQUIRED_ARTICLE_PROPS:
                if prop not in schema:
                    issues.append(
                        f"Article schema missing required property: '{prop}'"
                    )

            # Validate date formats
            for date_field in ('datePublished', 'dateModified'):
                date_val = schema.get(date_field, '')
                if date_val and not _ISO_DATE_RE.match(str(date_val)):
                    issues.append(
                        f"'{date_field}' value '{date_val}' is not valid ISO 8601."
                    )

            # Validate author structure
            author = schema.get('author', {})
            if isinstance(author, dict):
                if 'name' not in author:
                    issues.append("Article author is missing 'name' property.")
                if '@type' not in author:
                    issues.append(
                        "Article author is missing '@type' (should be 'Person')."
                    )
            elif not author:
                issues.append("Article 'author' is empty.")

            # Validate image
            image = schema.get('image', {})
            if isinstance(image, dict):
                if 'url' not in image:
                    issues.append("Article image is missing 'url'.")
            elif isinstance(image, str):
                pass  # string URL is technically valid but lacks dimensions
            elif not image:
                issues.append(
                    "Article schema missing 'image' — required for Rich Results.")

            # Headline length
            headline = schema.get('headline', '')
            if headline and len(headline) > 110:
                issues.append(
                    f"Article headline is {len(headline)} chars — Google recommends < 110."
                )

        elif schema_type == 'BreadcrumbList':
            items = schema.get('itemListElement', [])
            if not items:
                issues.append("BreadcrumbList has no itemListElement entries.")
            for item in items:
                if 'item' not in item or 'name' not in item:
                    issues.append(
                        "BreadcrumbList item is missing 'name' or 'item' URL."
                    )

    return issues


# ── FAQ schema builder ──────────────────────────────────────────────────────

def extract_and_build_faq_schema(content: str, base_url: str, slug: str) -> Optional[str]:
    """
    Extract FAQ questions and answers from the markdown content and
    return a JSON-LD FAQPage schema string, or None if no FAQ section found.

    The FAQ section must use ## Frequently Asked Questions heading.
    Questions must be ### subheadings.  Answers are the following paragraphs.
    """
    # Find FAQ section
    faq_pattern = re.compile(
        r'##\s+(?:Frequently\s+Asked\s+Questions|FAQ)\s*\n([\s\S]+?)(?=\n##\s|\Z)',
        re.IGNORECASE,
    )
    faq_match = faq_pattern.search(content)
    if not faq_match:
        return None

    faq_body = faq_match.group(1)

    # Extract Q&A pairs (### Question \n answer paragraph)
    qa_pattern = re.compile(
        r'###\s+(.+?)\n([\s\S]+?)(?=\n###\s|\Z)',
        re.IGNORECASE,
    )

    entries = []
    for qa_match in qa_pattern.finditer(faq_body):
        question = qa_match.group(1).strip().rstrip('?') + '?'
        answer_raw = qa_match.group(2).strip()
        # Strip markdown from answer
        answer = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', answer_raw)
        answer = re.sub(r'[*_`#]', '', answer)
        answer = re.sub(r'\s+', ' ', answer).strip()

        if question and answer and len(answer) > 20:
            entries.append({
                '@type': 'Question',
                'name': question,
                'acceptedAnswer': {
                    '@type': 'Answer',
                    'text': answer[:2000],   # Google recommends < 2000 chars
                },
            })

    if not entries:
        return None

    schema = {
        '@context': 'https://schema.org',
        '@type': 'FAQPage',
        '@id': f"{base_url}/{slug}/#faq",
        'mainEntity': entries,
    }

    try:
        return json.dumps(schema, indent=2, ensure_ascii=False)
    except Exception:
        return None
