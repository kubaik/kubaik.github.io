"""
adsense_fixes/similarity_guard.py
===================================
Content similarity detection for AdSense compliance.

WHY THIS EXISTS
---------------
Google AdSense's Site Readiness Guide (§7) requires:
  "Run each generated post through Copyscape, Grammarly's plagiarism
   checker, or similar tool before publishing to confirm < 10%
   similarity with existing web content."

This module provides FOUR layers of similarity protection:

  Layer 0 — Topic-key collision detection  (NEW)
    Catches the most common LLM content-farm failure mode: two posts
    that cover the *same underlying topic* but were prompted with a
    different headline number, vendor name, or percentage
    (e.g. "...200% latency spike" vs "...300% latency hit"), which
    defeats plain word-shingle Jaccard because the specific tokens
    differ throughout the piece even though the topic is identical.
    This layer normalises numbers/units before extracting a topic
    fingerprint from the title + first paragraph, so these collisions
    are caught even when body-level Jaccard looks low.

  Layer 1 — Cross-post fingerprinting (MinHash + Jaccard)
    Detects near-duplicate posts within your own site.
    This is the most reliable layer and requires no external API.

  Layer 2 — Structural repetition detection
    Detects posts that share the same sentence structures or paragraphs
    even when the topic words differ (a common LLM failure mode).

  Layer 3 — Optional Copyscape API integration
    If COPYSCAPE_USERNAME and COPYSCAPE_API_KEY env vars are set,
    performs a live web similarity check. Costs ~$0.05/check.
    Skip this for internal use only; use it before submission.

HOW TO INTEGRATE
----------------
In blog_system.py, after _validate_content_quality():

    from adsense_fixes.similarity_guard import SimilarityGuard

    guard = SimilarityGuard(docs_dir=blog_system.output_dir)
    result = guard.check(blog_post)
    if result.is_blocked:
        print(f"🛑 SIMILARITY BLOCK: {result.reason}")
        sys.exit(1)
    if result.warnings:
        for w in result.warnings:
            print(f"  ⚠️  Similarity warning: {w}")

IMPORTANT — CALLER MUST FAIL CLOSED
------------------------------------
If guard.check() raises, the caller MUST treat that as a hard failure
(sys.exit(1)), not swallow it as non-fatal. A duplicate-content gate
that silently no-ops on error provides no protection at all. See the
CLI integration patch shipped alongside this file.
"""

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────

# Jaccard similarity above this → BLOCK (post is too similar to existing)
_CROSS_SITE_BLOCK_THRESHOLD = 0.35

# Jaccard similarity above this → WARN
_CROSS_SITE_WARN_THRESHOLD = 0.20

# Fraction of shared n-grams (structural repetition) above this → WARN
_STRUCTURAL_WARN_THRESHOLD = 0.40

# Topic-key Jaccard above this → BLOCK (same underlying topic, different numbers)
_TOPIC_KEY_BLOCK_THRESHOLD = 0.55

# Topic-key Jaccard above this → WARN
_TOPIC_KEY_WARN_THRESHOLD = 0.40

# Minimum content length to bother checking (very short posts skip similarity)
_MIN_CHARS_TO_CHECK = 2000

# Copyscape API endpoint
_COPYSCAPE_URL = "https://www.copyscape.com/api/"

# Common stopwords stripped before building a topic key. Small and
# deliberately conservative — we want technical nouns/verbs to survive.
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "for",
    "with", "without", "at", "by", "from", "that", "this", "these", "those",
    "is", "are", "was", "were", "be", "been", "being", "it", "its", "we",
    "our", "you", "your", "i", "how", "why", "what", "when", "actually",
    "here", "s", "vs", "after", "before", "than", "into", "about", "did",
    "does", "do", "just", "still", "really", "very", "so", "as", "if",
}


# ─────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────

@dataclass
class SimilarityResult:
    is_blocked: bool = False
    reason: str = ""
    warnings: List[str] = field(default_factory=list)
    jaccard_scores: Dict[str, float] = field(default_factory=dict)
    topic_key_scores: Dict[str, float] = field(default_factory=dict)
    structural_score: float = 0.0
    copyscape_similarity: Optional[float] = None


# ─────────────────────────────────────────────────────────────────
# Text normalisation helpers
# ─────────────────────────────────────────────────────────────────

_CODE_FENCE_RE = re.compile(r"```[\s\S]*?```")
_HEADING_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^\)]+\)")
_MARKUP_RE = re.compile(r"[*_`>|#\[\]()~]")
_WHITESPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z]+|\d+(?:\.\d+)?")


def _normalize_text(text: str) -> str:
    """Strip Markdown, code blocks, and normalise whitespace."""
    text = _CODE_FENCE_RE.sub(" ", text)
    text = _HEADING_RE.sub(" ", text)
    text = _LINK_RE.sub(r"\1", text)
    text = _MARKUP_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.lower().strip()


def _topic_tokens(text: str) -> List[str]:
    """
    Tokenize into pure alphabetic words and numbers, discarding all
    punctuation (colons, hyphens, percent signs, etc.) so that
    "post-quantum" / "post quantum", or "300%" / "300 percent",
    tokenize identically instead of being treated as distinct terms.
    Every numeric token is then collapsed to a single '<num>'
    placeholder — this is the key fix for the "same topic, different
    headline number" content-farm pattern: "200% latency spike" and
    "300% latency hit" should collide on topic even though their
    literal digits differ. Body-level Jaccard (Layer 1) still uses the
    untouched numbers, so genuinely distinct benchmark posts aren't
    over-merged there — only this topic-key layer normalises numbers.
    """
    tokens = _TOKEN_RE.findall(text.lower())
    return ["<num>" if t[0].isdigit() else t for t in tokens]


def _topic_key_terms(title: str, content: str) -> Set[str]:
    """
    Extract a compact set of topic-defining terms from the title and
    the first ~40 words of body content (the part that carries the
    "what is this post actually about" signal, before specifics vary).
    """
    lead = " ".join(content.split()[:40])
    combined = f"{title} {lead}"
    words = [w for w in _topic_tokens(combined)
             if w == "<num>" or (w not in _STOPWORDS and len(w) > 2)]
    # Keep both unigrams and bigrams — bigrams capture compound technical
    # terms ("zero trust", "post quantum") that unigrams alone would blur.
    bigrams = {f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)}
    return set(words) | bigrams


def _shingles(text: str, k: int = 5) -> Set[str]:
    """
    Produce a set of k-character shingles from text.
    Shingles are the foundation of MinHash similarity estimation.
    k=5 is a good balance between sensitivity and false-positive rate
    for English prose of 1,500–5,000 words.
    """
    words = text.split()
    if len(words) < k:
        # Fall back to character-level shingles for very short text
        chars = text.replace(" ", "")
        return {chars[i:i+k] for i in range(len(chars) - k + 1)}
    return {" ".join(words[i:i+k]) for i in range(len(words) - k + 1)}


def _word_ngrams(text: str, n: int = 4) -> Set[str]:
    """Produce word-level n-grams for structural analysis."""
    words = re.sub(r"[^\w\s]", "", text).split()
    return {" ".join(words[i:i+n]) for i in range(len(words) - n + 1)}


def _jaccard(a: Set, b: Set) -> float:
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0.0


def _fingerprint(text: str) -> str:
    """SHA-256 fingerprint of the normalised content."""
    normalized = _normalize_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────────
# Fingerprint index (on-disk cache for fast lookups)
# ─────────────────────────────────────────────────────────────────

class FingerprintIndex:
    """
    Lightweight on-disk index of content fingerprints and shingle sets.
    Avoids re-computing shingles for every existing post on each run.
    """

    def __init__(self, docs_dir: Path, index_file: Path = None):
        self.docs_dir = docs_dir
        self.index_file = index_file or Path(".similarity_index.json")
        self._index: Dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self.index_file.exists():
            return
        try:
            with open(self.index_file, "r", encoding="utf-8") as f:
                self._index = json.load(f)
        except (json.JSONDecodeError, OSError):
            self._index = {}

    def _save(self) -> None:
        try:
            with open(self.index_file, "w", encoding="utf-8") as f:
                json.dump(self._index, f)
        except OSError as e:
            print(f"  ⚠️  Could not save similarity index: {e}")

    def build(self, force: bool = False) -> None:
        """
        Walk docs_dir and index every post.json.
        Only processes posts not already in the index (incremental).
        """
        if not self.docs_dir.exists():
            return

        updated = False
        for post_dir in self.docs_dir.iterdir():
            if not post_dir.is_dir() or post_dir.name in ("static", "tag", "author"):
                continue
            post_json = post_dir / "post.json"
            if not post_json.exists():
                continue

            slug = post_dir.name
            mtime = post_json.stat().st_mtime

            # Skip if already indexed at this mtime
            if (
                not force
                and slug in self._index
                and self._index[slug].get("mtime") == mtime
            ):
                continue

            try:
                with open(post_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                content = data.get("content", "")
                title = data.get("title", "")
                if len(content) < _MIN_CHARS_TO_CHECK:
                    continue
                normalized = _normalize_text(content)
                fp = hashlib.sha256(normalized.encode()).hexdigest()
                shingle_list = sorted(_shingles(normalized, k=5))[
                    :2000]  # cap for disk size
                topic_terms = sorted(_topic_key_terms(title, content))
                self._index[slug] = {
                    "mtime": mtime,
                    "fingerprint": fp,
                    "shingles": shingle_list,
                    "topic_terms": topic_terms,
                    "title": title,
                }
                updated = True
            except Exception:
                continue

        if updated:
            self._save()

    def get_all(self) -> Dict[str, dict]:
        return self._index

    def add(self, slug: str, content: str, title: str = "") -> None:
        """Add or update a single entry (used after publishing a new post)."""
        if len(content) < _MIN_CHARS_TO_CHECK:
            return
        normalized = _normalize_text(content)
        fp = hashlib.sha256(normalized.encode()).hexdigest()
        shingle_list = sorted(_shingles(normalized, k=5))[:2000]
        topic_terms = sorted(_topic_key_terms(title, content))
        self._index[slug] = {
            "mtime": 0,  # will be updated on next build()
            "fingerprint": fp,
            "shingles": shingle_list,
            "topic_terms": topic_terms,
            "title": title,
        }
        self._save()


# ─────────────────────────────────────────────────────────────────
# Main SimilarityGuard class
# ─────────────────────────────────────────────────────────────────

class SimilarityGuard:
    """
    Checks a new BlogPost for similarity against existing posts
    and optionally against the live web via Copyscape.
    """

    def __init__(
        self,
        docs_dir: Path,
        copyscape_user: str = None,
        copyscape_key: str = None,
    ):
        self.docs_dir = docs_dir
        self.copyscape_user = copyscape_user or os.getenv(
            "COPYSCAPE_USERNAME", "")
        self.copyscape_key = copyscape_key or os.getenv(
            "COPYSCAPE_API_KEY", "")
        self._fp_index = FingerprintIndex(docs_dir)
        self._fp_index.build()

    def check(self, post) -> SimilarityResult:
        """
        Run all similarity checks on the post.
        Returns a SimilarityResult; caller decides how to act on it.
        Caller MUST treat an exception from this method as a hard
        failure (fail closed), not a soft/non-fatal warning.
        """
        result = SimilarityResult()
        content = getattr(post, "content", "")
        title = getattr(post, "title", "")

        if len(content) < _MIN_CHARS_TO_CHECK:
            result.warnings.append(
                f"Content too short ({len(content)} chars) for similarity check — skipped."
            )
            return result

        slug = getattr(post, "slug", "")
        normalized = _normalize_text(content)
        existing = self._fp_index.get_all()

        # ── Layer 0: Topic-key collision detection ─────────────────────────
        # Runs FIRST and can block on its own, independent of body Jaccard,
        # because it's specifically designed to catch cases where body
        # Jaccard is deflated by swapped numbers/vendor names.
        candidate_topic_terms = _topic_key_terms(title, content)
        for existing_slug, entry in existing.items():
            if existing_slug == slug:
                continue
            existing_topic_terms = set(entry.get("topic_terms", []))
            if not existing_topic_terms:
                continue
            topic_score = _jaccard(candidate_topic_terms, existing_topic_terms)
            if topic_score > 0.05:
                result.topic_key_scores[existing_slug] = round(topic_score, 3)

            if topic_score >= _TOPIC_KEY_BLOCK_THRESHOLD:
                result.is_blocked = True
                result.reason = (
                    f"Post covers the same underlying topic as existing post "
                    f"'{entry.get('title', existing_slug)}' (/{existing_slug}/) "
                    f"(topic-key similarity {topic_score:.0%}, threshold "
                    f"{_TOPIC_KEY_BLOCK_THRESHOLD:.0%}). This is the 'same topic, "
                    f"different headline number' pattern — pick a genuinely "
                    f"distinct angle or topic."
                )
                return result

            if topic_score >= _TOPIC_KEY_WARN_THRESHOLD:
                result.warnings.append(
                    f"Topic-key overlap ({topic_score:.0%}) with "
                    f"'{entry.get('title', existing_slug)}' (/{existing_slug}/). "
                    f"Verify this post isn't just a reworded restatement."
                )

        # ── Layer 1: Cross-post fingerprint comparison ────────────────────────
        candidate_shingles = _shingles(normalized, k=5)

        for existing_slug, entry in existing.items():
            if existing_slug == slug:
                continue

            # Fast path: exact fingerprint match
            existing_fp = entry.get("fingerprint", "")
            candidate_fp = hashlib.sha256(normalized.encode()).hexdigest()
            if existing_fp == candidate_fp:
                result.is_blocked = True
                result.reason = (
                    f"Exact content duplicate of '{entry.get('title', existing_slug)}' "
                    f"(/{existing_slug}/). Post will not be published."
                )
                return result

            # Jaccard similarity
            existing_shingles = set(entry.get("shingles", []))
            if not existing_shingles:
                continue
            score = _jaccard(candidate_shingles, existing_shingles)
            if score > 0.01:
                result.jaccard_scores[existing_slug] = round(score, 3)

            if score >= _CROSS_SITE_BLOCK_THRESHOLD:
                result.is_blocked = True
                result.reason = (
                    f"Post is {score:.0%} similar to existing post "
                    f"'{entry.get('title', existing_slug)}' (/{existing_slug}/). "
                    f"Threshold: {_CROSS_SITE_BLOCK_THRESHOLD:.0%}. "
                    f"This level of similarity risks AdSense rejection for duplicate content."
                )
                return result

            if score >= _CROSS_SITE_WARN_THRESHOLD:
                result.warnings.append(
                    f"Moderate similarity ({score:.0%}) with "
                    f"'{entry.get('title', existing_slug)}' (/{existing_slug}/). "
                    f"Consider diverging the angle or examples used."
                )

        # ── Layer 2: Structural repetition detection ──────────────────────────
        # This catches posts that share the same sentence patterns (a common
        # LLM failure mode where the structure is identical but nouns differ).
        structural_score = self._structural_repetition_score(
            normalized, existing)
        result.structural_score = structural_score
        if structural_score >= _STRUCTURAL_WARN_THRESHOLD:
            result.warnings.append(
                f"High structural repetition score ({structural_score:.0%}) — "
                f"this post shares many sentence-level patterns with existing posts. "
                f"LLM may be templating. Verify the post reads as genuinely unique."
            )

        # ── Layer 3: Copyscape (optional) ─────────────────────────────────────
        if self.copyscape_user and self.copyscape_key:
            copyscape_sim = self._copyscape_check(content)
            result.copyscape_similarity = copyscape_sim
            if copyscape_sim is not None and copyscape_sim > 0.10:
                result.is_blocked = True
                result.reason = (
                    f"Copyscape reports {copyscape_sim:.0%} similarity with existing web content "
                    f"(threshold: 10%). Do not publish — this content may be near-verbatim "
                    f"LLM reproduction of copyrighted material."
                )
                return result
            if copyscape_sim is not None and copyscape_sim > 0.05:
                result.warnings.append(
                    f"Copyscape similarity: {copyscape_sim:.0%} — borderline. "
                    f"Review the post for any verbatim phrases from common sources."
                )

        return result

    def _structural_repetition_score(
        self,
        normalized_candidate: str,
        existing_index: Dict[str, dict],
    ) -> float:
        """
        Compute the maximum fraction of 4-grams in this post that appear
        in ANY existing post. High scores indicate templated writing.
        """
        if not existing_index:
            return 0.0

        candidate_ngrams = _word_ngrams(normalized_candidate, n=4)
        if not candidate_ngrams:
            return 0.0

        # Build a union of all n-grams from all existing posts
        # (we compare against the union, not each post, to detect
        # broad structural patterns — phrases that appear in many posts)
        all_existing_ngrams: Set[str] = set()
        for entry in existing_index.values():
            # n-grams aren't stored in the index; derive from shingles
            # as a proxy (shingles are 5-word; use 4-word for structure)
            for shingle in entry.get("shingles", [])[:500]:
                words = shingle.split()
                if len(words) >= 4:
                    all_existing_ngrams.add(" ".join(words[:4]))

        if not all_existing_ngrams:
            return 0.0

        overlap = len(candidate_ngrams & all_existing_ngrams)
        return overlap / len(candidate_ngrams)

    def _copyscape_check(self, content: str) -> Optional[float]:
        """
        Query the Copyscape API for web similarity.
        Returns a float 0.0–1.0 or None if the check fails.

        API docs: https://www.copyscape.com/api/instructions.php
        Cost: ~$0.05 per check (charged from your Copyscape balance).
        """
        try:
            import urllib.parse
            import urllib.request

            # Copyscape accepts up to 5,000 words of text
            excerpt = " ".join(content.split()[:4500])
            params = urllib.parse.urlencode({
                "u": self.copyscape_user,
                "k": self.copyscape_key,
                "o": "csearch",
                "e": "UTF-8",
                "c": "5",
                "t": excerpt,
            })
            url = f"{_COPYSCAPE_URL}?{params}"
            req = urllib.request.Request(url, method="GET")
            req.add_header("User-Agent", "AutoBlog/1.0 SimilarityGuard")

            with urllib.request.urlopen(req, timeout=30) as resp:
                xml_body = resp.read().decode("utf-8")

            # Parse similarity from XML response
            # <percentmatched> field holds the highest match percentage
            match = re.search(
                r"<percentmatched>(\d+)</percentmatched>", xml_body)
            if match:
                return int(match.group(1)) / 100.0

            # If no results, Copyscape returns <count>0</count>
            no_result = re.search(r"<count>0</count>", xml_body)
            if no_result:
                return 0.0

            print(
                f"  ⚠️  Copyscape returned unexpected response: {xml_body[:200]}")
            return None

        except Exception as e:
            print(f"  ⚠️  Copyscape check failed (non-fatal): {e}")
            return None

    def update_index(self, post) -> None:
        """
        Add the newly published post to the similarity index so future
        posts are checked against it immediately.
        Call this after save_post() succeeds.
        """
        self._fp_index.add(
            slug=getattr(post, "slug", ""),
            content=getattr(post, "content", ""),
            title=getattr(post, "title", ""),
        )
