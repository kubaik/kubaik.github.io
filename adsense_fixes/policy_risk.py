"""
adsense_fixes/policy_risk.py
============================
Shared title/topic bans used by content_audit_verdicts.py and blog_system.py.

These patterns produced the AdSense-risk corpus: income-claim bait,
sensational unsourced percentages, and “side gig / salary hack” posts.
"""
from __future__ import annotations

import re
from typing import List, Optional


# Title or content_topics line that must never be generated or kept.
_BANNED_TOPIC_RE = re.compile(
    r"(?:"
    r"\bside[\s-]?gigs?\b|"
    r"\bpassive income\b|"
    r"\bincome streams?\b|"
    r"\bsalary hacks?\b|"
    r"\bnegotiate salary\b|"
    r"\bland a \$?\d+k\b|"
    r"\b(?:make|earn|paid(?:\s+me)?|profit)\b.{0,24}\$?\d+k\s*/?\s*mo\b|"
    r"\bvc[\s-]?insights\b|"
    r"\bmake \$?\d+k\b"
    r")",
    re.IGNORECASE,
)

_SENSATIONAL_PCT_RE = re.compile(
    r"\b(?:cut|reduce(?:d)?|improv(?:e|ed)|sav(?:e|ed)|increas(?:e|ed)|"
    r"lift(?:ed)?|rose|rise|grew)\b.{0,40}\b\d{1,3}\s*%\b",
    re.IGNORECASE,
)


def topic_policy_violation(text: str) -> Optional[str]:
    """Return a short reason if this title/topic must not be published."""
    if not text:
        return None
    if _BANNED_TOPIC_RE.search(text):
        return (
            "Income/salary/side-gig bait — low-value monetization content "
            "and unverifiable earnings claims."
        )
    if _SENSATIONAL_PCT_RE.search(text):
        return "Headline contains an unsourced performance percentage."
    return None


def filter_safe_topics(topics: List[str]) -> List[str]:
    return [t for t in topics if not topic_policy_violation(t)]


# Site chrome under docs/ — never classify, tombstone, or queue these.
STATIC_PAGE_SLUGS = frozenset({
    "about",
    "contact",
    "privacy-policy",
    "privacy",
    "terms-of-service",
    "terms",
    "tos",
    "dmca",
    "ai-content-policy",
    "cookie-policy",
    "cookies",
    "disclaimer",
    "legal",
    "page",
})

SKIP_SITE_DIRS = frozenset({
    "static",
    "tag",
    "author",
    "tags",
    "authors",
    "assets",
    "css",
    "js",
    "images",
}) | STATIC_PAGE_SLUGS


def is_static_page(slug: str) -> bool:
    """True for about/contact/legal and other non-post dirs."""
    if not slug:
        return True
    name = slug.strip("/").split("/")[-1].lower()
    return name in SKIP_SITE_DIRS
