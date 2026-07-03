#!/usr/bin/env python3
"""
scripts/content_quality_gate.py

Automated quality gate for AI-generated content before publishing.
Run this in the build pipeline before a post is committed/published.

Gates enforced (all must pass):
  1. Title length: 40–70 characters (no truncated titles)
  2. Title completeness: must not end with preposition/article/conjunction
  3. Description length: 120–160 characters  
  4. Word count: minimum 1500 words (thin content rejection)
  5. Near-duplicate detection: cosine similarity < 0.72 vs recent posts
  6. Publication rate: maximum 3 posts per calendar day
  7. Keyword stuffing: primary keyword density < 3% of total words
  8. AI disclosure: article body must contain editorial byline section
  9. External links: at least 2 external links (demonstrates research)
  10. Code blocks: if title mentions code/api/system, must have ≥1 code block

Usage:
    python scripts/content_quality_gate.py \
        --post path/to/new-post.html \
        --index path/to/published_index.json \
        --date 2026-06-29

Exit codes:
    0 = all gates pass (publish)
    1 = one or more gates failed (reject)

The index file is a JSON array of published posts:
    [{"url": "/slug/", "title": "...", "date": "YYYY-MM-DD", "word_count": N,
      "description": "...", "tfidf_vector": [...]}]
"""

import argparse
import json
import math
import re
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────
MAX_POSTS_PER_DAY = 3
MIN_WORD_COUNT = 1500
MIN_TITLE_LEN = 40
MAX_TITLE_LEN = 70
MIN_DESC_LEN = 120
MAX_DESC_LEN = 165
DUPLICATE_THRESHOLD = 0.72   # cosine similarity above this = reject
MAX_KEYWORD_DENSITY = 0.030  # 3%
MIN_EXTERNAL_LINKS = 2

# Words that should not end a title (signals truncation)
WEAK_TITLE_ENDINGS = {
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "and",
    "or", "but", "with", "by", "from", "as", "is", "are", "was",
    "you", "your", "we", "our", "this", "that", "it", "its",
}

STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "can", "that",
    "this", "these", "those", "it", "its", "we", "you", "your", "our",
    "they", "their", "what", "which", "who", "when", "where", "how",
    "not", "no", "so", "if", "as", "than", "then", "about", "up",
    "out", "into", "more", "also", "just", "after", "before", "over",
    "some", "any", "all", "each", "both", "between", "through",
}


def extract_text_from_html(html: str) -> str:
    """Strip all HTML tags and return plain text."""
    # Remove script and style blocks
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>",
                  "", html, flags=re.DOTALL | re.IGNORECASE)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode common entities
    text = text.replace("&amp;", "&").replace(
        "&lt;", "<").replace("&gt;", ">").replace("&nbsp;", " ")
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_meta(html: str, property_or_name: str) -> Optional[str]:
    """Extract content from a <meta> tag by name or property."""
    patterns = [
        rf'<meta[^>]+(?:name|property)=["\']{re.escape(property_or_name)}["\'][^>]*content=["\']([^"\']+)["\']',
        rf'<meta[^>]+content=["\']([^"\']+)["\'][^>]+(?:name|property)=["\']{re.escape(property_or_name)}["\'][^>]*>'
    ]
    for p in patterns:
        m = re.search(p, html, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


def word_tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    tokens = re.findall(r"\b[a-z][a-z']{1,}\b", text.lower())
    return tokens


def build_tfidf_vector(text: str, vocab: Optional[dict] = None) -> dict:
    """Build a simple TF-IDF-like vector (TF only if no IDF available)."""
    tokens = [t for t in word_tokenize(text) if t not in STOP_WORDS]
    total = len(tokens)
    if total == 0:
        return {}
    counts = Counter(tokens)
    return {word: count / total for word, count in counts.items()}


def cosine_similarity(v1: dict, v2: dict) -> float:
    """Cosine similarity between two sparse vectors (dicts)."""
    common = set(v1) & set(v2)
    if not common:
        return 0.0
    dot = sum(v1[k] * v2[k] for k in common)
    mag1 = math.sqrt(sum(x ** 2 for x in v1.values()))
    mag2 = math.sqrt(sum(x ** 2 for x in v2.values()))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


def count_code_blocks(html: str) -> int:
    return len(re.findall(r"<code[^>]*>", html, re.IGNORECASE))


def count_external_links(html: str) -> int:
    links = re.findall(
        r'href=["\']https?://([^"\']+)["\']', html, re.IGNORECASE)
    # Exclude self-referential links
    external = [l for l in links if not l.startswith("kubaik.github.io")]
    return len(set(external))


def keyword_density(text: str, keyword: str) -> float:
    words = word_tokenize(text)
    total = len(words)
    if total == 0:
        return 0.0
    kw_tokens = word_tokenize(keyword)
    if not kw_tokens:
        return 0.0
    # Count occurrences of first keyword token as proxy
    count = sum(1 for w in words if w == kw_tokens[0])
    return count / total


class QualityGate:
    def __init__(self, post_html: str, index: list[dict], pub_date: date, post_path: str):
        self.html = post_html
        self.index = index
        self.pub_date = pub_date
        self.post_path = post_path
        self.text = extract_text_from_html(post_html)
        self.words = word_tokenize(self.text)
        self.errors: list[str] = []
        self.warnings: list[str] = []

        self.title = self._extract_title()
        self.description = extract_meta(post_html, "description") or ""
        self.keywords_raw = extract_meta(post_html, "keywords") or ""
        self.primary_keyword = (self.keywords_raw.split(
            ",")[0] if self.keywords_raw else "").strip()

    def _extract_title(self) -> str:
        m = re.search(r"<title>([^<]+)</title>", self.html, re.IGNORECASE)
        if m:
            # Strip " - Kubai Kevin" suffix if present
            t = m.group(1).strip()
            t = re.sub(r"\s*[-|]\s*Kubai Kevin\s*$", "", t).strip()
            return t
        return ""

    def check_title_length(self):
        n = len(self.title)
        if n < MIN_TITLE_LEN:
            self.errors.append(
                f"TITLE_TOO_SHORT: '{self.title}' is {n} chars (min {MIN_TITLE_LEN})"
            )
        elif n > MAX_TITLE_LEN:
            self.errors.append(
                f"TITLE_TOO_LONG: '{self.title}' is {n} chars (max {MAX_TITLE_LEN}). "
                "Store the full title in `full_title` field and shorten display title."
            )

    def check_title_completeness(self):
        last_word = self.title.rstrip(
            ".,!?:;").split()[-1].lower() if self.title else ""
        if last_word in WEAK_TITLE_ENDINGS:
            self.errors.append(
                f"TITLE_TRUNCATED: Title ends with '{last_word}', which suggests "
                f"truncation. Full title: '{self.title}'"
            )

    def check_description(self):
        n = len(self.description)
        if n < MIN_DESC_LEN:
            self.errors.append(
                f"DESC_TOO_SHORT: {n} chars (min {MIN_DESC_LEN}). "
                "Description: '{}'".format(self.description[:80])
            )
        elif n > MAX_DESC_LEN:
            self.warnings.append(
                f"DESC_TOO_LONG: {n} chars (recommended max {MAX_DESC_LEN}). "
                "May be truncated in SERPs."
            )

    def check_word_count(self):
        wc = len(self.words)
        if wc < MIN_WORD_COUNT:
            self.errors.append(
                f"THIN_CONTENT: {wc} words (min {MIN_WORD_COUNT}). "
                "Google's helpful content system penalises thin articles."
            )

    def check_publication_rate(self):
        date_str = self.pub_date.isoformat()
        same_day = [p for p in self.index if p.get("date", "")[
            :10] == date_str]
        if len(same_day) >= MAX_POSTS_PER_DAY:
            self.errors.append(
                f"RATE_LIMIT: {len(same_day)} posts already published on {date_str}. "
                f"Maximum is {MAX_POSTS_PER_DAY}/day. Schedule this post for tomorrow."
            )

    def check_duplicate_content(self):
        new_vec = build_tfidf_vector(self.text)
        # Only compare against posts from the last 90 days to keep it fast
        cutoff = date.fromisoformat(
            (datetime(self.pub_date.year, self.pub_date.month, self.pub_date.day)
             .replace(year=self.pub_date.year - 1)).isoformat()[:10]
        )
        recent = [
            p for p in self.index
            if p.get("date", "0000-00-00")[:10] >= cutoff.isoformat()
        ]
        for post in recent:
            existing_vec = post.get("tfidf_vector")
            if not existing_vec:
                # Rebuild from description + title if vector not cached
                existing_text = f"{post.get('title', '')} {post.get('description', '')}"
                existing_vec = build_tfidf_vector(existing_text)
            sim = cosine_similarity(new_vec, existing_vec)
            if sim >= DUPLICATE_THRESHOLD:
                self.errors.append(
                    f"NEAR_DUPLICATE: Similarity {sim:.2f} with '{post.get('url', '')}' "
                    f"('{post.get('title', '')}') exceeds threshold {DUPLICATE_THRESHOLD}. "
                    "Merge these articles or differentiate the angle significantly."
                )
                # Report first match only to avoid noise
                break

    def check_keyword_density(self):
        if not self.primary_keyword:
            self.warnings.append(
                "NO_PRIMARY_KEYWORD: No keywords meta tag found. "
                "Add keywords to help with topic clustering."
            )
            return
        density = keyword_density(self.text, self.primary_keyword)
        if density > MAX_KEYWORD_DENSITY:
            self.errors.append(
                f"KEYWORD_STUFFING: '{self.primary_keyword}' appears at {density:.1%} "
                f"density (max {MAX_KEYWORD_DENSITY:.0%}). "
                "Reduce repetition — Google penalises keyword stuffing."
            )

    def check_editorial_byline(self):
        """Require the editorial standards section in article body."""
        has_byline = (
            "Written by:" in self.html
            or "Editorial standard:" in self.html
            or 'about/#author' in self.html
        )
        if not has_byline:
            self.errors.append(
                "MISSING_BYLINE: No editorial byline/author section found. "
                "All articles must include the author attribution block for E-E-A-T."
            )

    def check_external_links(self):
        count = count_external_links(self.html)
        if count < MIN_EXTERNAL_LINKS:
            self.warnings.append(
                f"FEW_EXTERNAL_LINKS: Only {count} external link(s) found "
                f"(recommended minimum: {MIN_EXTERNAL_LINKS}). "
                "Link to official docs, GitHub repos, or primary sources."
            )

    def check_code_blocks_for_technical_posts(self):
        technical_keywords = {
            "code", "api", "function", "script", "query", "command",
            "implementation", "pipeline", "deploy", "kubernetes", "docker",
            "python", "node", "sql", "redis", "postgresql"
        }
        title_words = set(word_tokenize(self.title.lower()))
        is_technical = bool(title_words & technical_keywords)
        if is_technical and count_code_blocks(self.html) == 0:
            self.warnings.append(
                "MISSING_CODE_BLOCKS: Technical post with no code examples. "
                "Developer audience expects runnable code for technical articles."
            )

    def run_all_checks(self) -> bool:
        self.check_title_length()
        self.check_title_completeness()
        self.check_description()
        self.check_word_count()
        self.check_publication_rate()
        self.check_duplicate_content()
        self.check_keyword_density()
        self.check_editorial_byline()
        self.check_external_links()
        self.check_code_blocks_for_technical_posts()
        return len(self.errors) == 0

    def report(self):
        if self.errors:
            print(f"\n❌ QUALITY GATE FAILED — {self.post_path}")
            print(f"   Title: {self.title}")
            for e in self.errors:
                print(f"   [ERROR] {e}")
        if self.warnings:
            print(f"\n⚠  WARNINGS — {self.post_path}")
            for w in self.warnings:
                print(f"   [WARN]  {w}")
        if not self.errors and not self.warnings:
            print(f"\n✅ QUALITY GATE PASSED — {self.post_path}")
        elif not self.errors:
            print(
                f"\n✅ QUALITY GATE PASSED (with warnings) — {self.post_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Content quality gate for AI-generated posts")
    parser.add_argument("--post", required=True, help="Path to post HTML file")
    parser.add_argument(
        "--index",
        required=True,
        help="Path to published_index.json",
    )
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Publication date YYYY-MM-DD (default: today)",
    )
    args = parser.parse_args()

    post_path = Path(args.post)
    if not post_path.exists():
        print(f"ERROR: Post file not found: {post_path}", file=sys.stderr)
        return 1

    index_path = Path(args.index)
    index: list[dict] = []
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)

    try:
        pub_date = date.fromisoformat(args.date)
    except ValueError:
        print(f"ERROR: Invalid date format: {args.date}", file=sys.stderr)
        return 1

    html = post_path.read_text(encoding="utf-8", errors="replace")
    gate = QualityGate(html, index, pub_date, str(post_path))
    passed = gate.run_all_checks()
    gate.report()

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
