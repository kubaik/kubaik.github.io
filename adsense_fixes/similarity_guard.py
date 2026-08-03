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
# NOTE: lowered from 0.55 → 0.45 alongside widening the topic-key window from
# the first ~40 words to ~150 words (see _topic_key_terms below). The wider
# window pulls in more shared vocabulary by default, which pushes raw scores
# up across the board — so the old 0.55 cut-off would now trigger on posts
# that are merely on the same beat, not truly duplicate. 0.45 is a starting
# estimate, not a measured value: run the audit CLI at the bottom of this
# file (`python similarity_guard.py audit ./docs`) against your live corpus
# and look at the score distribution before trusting this in production —
# nudge it up or down based on what you see there.
_TOPIC_KEY_BLOCK_THRESHOLD = 0.45

# Topic-key Jaccard above this → WARN
_TOPIC_KEY_WARN_THRESHOLD = 0.35

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


_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def _topic_key_terms(title: str, content: str) -> Tuple[Set[str], Set[str]]:
    """
    Extract topic-defining terms from the title and the first ~150 words
    of body content, split into UNIGRAMS and BIGRAMS (returned separately,
    not merged) so callers can weight them differently — see
    _topic_key_combined_score for why that separation matters.

    WIDENED from ~40 → ~150 words: with this blog's post structure, the
    first 40 words are frequently a hook/anecdote opener that doesn't
    name the actual topic yet — the real topic vocabulary only shows up
    once the intro paragraph gets going. 150 words reliably captures the
    intro without pulling in enough of the body to start matching on
    generic supporting vocabulary shared by unrelated posts.

    Year mentions (e.g. "in 2026") are stripped before tokenizing rather
    than collapsed to '<num>' like other numbers, since on a blog where
    nearly every post opens with "...in 2026" the year is calendar
    boilerplate, not topic content. Other numbers (counts, percentages,
    dollar amounts) still collapse to '<num>' as before.
    """
    lead = " ".join(content.split()[:150])
    combined = _YEAR_RE.sub(" ", f"{title} {lead}")
    words = [w for w in _topic_tokens(combined)
             if w == "<num>" or (w not in _STOPWORDS and len(w) > 2)]
    bigrams = {f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)}
    return set(words), bigrams


# Bigrams ("offline_first", "starlink_4g", "feature_flags") only match when
# two posts share a specific multi-word concept — they're a much stronger
# duplicate signal than a shared single word. Unrelated-but-adjacent posts
# in the same technical space (e.g. two different infra-reliability posts)
# routinely share several generic unigrams ("systems", "cloud", "api",
# "patterns") without covering the same actual angle; a flat Jaccard over
# a merged unigram+bigram set let that generic overlap carry as much
# weight as a genuine phrase match, which is what made real duplicates
# (Starlink/4G, offline-first, feature-flags) score in the same 29-33%
# band as clearly-unrelated pairs (Pulumi/Terraform vs. unrelated infra
# posts) on the live corpus. Weighting bigrams much more heavily is meant
# to pull those two clusters apart — re-run the audit CLI after this
# change to confirm it actually does, on your real data.
_BIGRAM_WEIGHT = 0.75
_UNIGRAM_WEIGHT = 0.25


def _topic_key_combined_score(entry_a: dict, entry_b: dict) -> float:
    """Combine bigram + unigram topic-key overlap into one weighted score.
    Both entries are dicts with 'topic_unigrams' and 'topic_bigrams' lists
    (the shape stored in FingerprintIndex / built ad hoc for a candidate).
    Used by both SimilarityGuard.check() and the audit CLI so they can
    never silently disagree on how a score is computed."""
    bigram_score = _jaccard(
        set(entry_a.get("topic_bigrams", [])),
        set(entry_b.get("topic_bigrams", [])),
    )
    unigram_score = _jaccard(
        set(entry_a.get("topic_unigrams", [])),
        set(entry_b.get("topic_unigrams", [])),
    )
    return _BIGRAM_WEIGHT * bigram_score + _UNIGRAM_WEIGHT * unigram_score


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
                topic_unigrams, topic_bigrams = _topic_key_terms(
                    title, content)
                self._index[slug] = {
                    "mtime": mtime,
                    "fingerprint": fp,
                    "shingles": shingle_list,
                    "topic_unigrams": sorted(topic_unigrams),
                    "topic_bigrams": sorted(topic_bigrams),
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
        topic_unigrams, topic_bigrams = _topic_key_terms(title, content)
        self._index[slug] = {
            "mtime": 0,  # will be updated on next build()
            "fingerprint": fp,
            "shingles": shingle_list,
            "topic_unigrams": sorted(topic_unigrams),
            "topic_bigrams": sorted(topic_bigrams),
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
        candidate_unigrams, candidate_bigrams = _topic_key_terms(
            title, content)
        candidate_entry = {
            "topic_unigrams": sorted(candidate_unigrams),
            "topic_bigrams": sorted(candidate_bigrams),
        }
        for existing_slug, entry in existing.items():
            if existing_slug == slug:
                continue
            if not entry.get("topic_unigrams") and not entry.get("topic_bigrams"):
                continue
            topic_score = _topic_key_combined_score(candidate_entry, entry)
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


# ─────────────────────────────────────────────────────────────────
# Audit CLI — measure the topic-key threshold against a live corpus
# ─────────────────────────────────────────────────────────────────
#
# The 150-word window and 0.45/0.35 thresholds above are an informed
# guess, not a measured result. Before relying on them in production,
# run this against your actual docs/ folder:
#
#   python similarity_guard.py audit ./docs
#
# It recomputes topic_unigrams/topic_bigrams for every published post using
# the CURRENT window/threshold constants, scores every pair with the same
# bigram-weighted formula production uses, and prints:
#   - the score distribution (so you can see where a natural cut-off falls)
#   - every pair that would BLOCK or WARN under the current thresholds
#
# Read through the printed pairs: if genuinely distinct posts show up in
# the BLOCK list, raise _TOPIC_KEY_BLOCK_THRESHOLD; if known-duplicate
# topics you expected to catch aren't there, lower it. Nothing here
# writes to .similarity_index.json or touches your posts — it's read-only.
def _audit_topic_key_threshold(docs_dir: str, top_n: int = 20) -> None:
    import itertools
    import statistics

    docs_path = Path(docs_dir)
    index = FingerprintIndex(docs_path)
    index.build(force=True)
    entries = index.get_all()

    if len(entries) < 2:
        print(
            f"Only {len(entries)} post(s) indexed under {docs_dir} — need at least 2 to compare.")
        return

    slugs = list(entries.keys())
    scores: List[Tuple[str, str, float]] = []
    for slug_a, slug_b in itertools.combinations(slugs, 2):
        score = _topic_key_combined_score(entries[slug_a], entries[slug_b])
        if score > 0.0:
            scores.append((slug_a, slug_b, score))

    scores.sort(key=lambda t: t[2], reverse=True)
    raw_scores = [s for _, _, s in scores]

    print(f"Indexed posts : {len(entries)}")
    print(f"Pairs compared: {len(list(itertools.combinations(slugs, 2)))}")
    print(f"Non-zero pairs: {len(scores)}")
    if raw_scores:
        print(
            f"Score stats   : min={min(raw_scores):.2f} "
            f"median={statistics.median(raw_scores):.2f} "
            f"mean={statistics.mean(raw_scores):.2f} "
            f"max={max(raw_scores):.2f}"
        )

    # ALWAYS show the highest-scoring pairs, regardless of whether they
    # cross WARN/BLOCK. This matters most when nothing crosses the
    # current thresholds — that could mean your corpus is genuinely
    # diverse, or it could mean the thresholds are miscalibrated and
    # blind to real near-duplicates sitting just below the cut-off.
    # You can't tell the difference from summary stats alone; you have
    # to look at the actual title pairs.
    print(
        f"\nTop {min(top_n, len(scores))} highest-scoring pairs (regardless of threshold):")
    for slug_a, slug_b, score in scores[:top_n]:
        title_a = entries[slug_a].get("title", slug_a)
        title_b = entries[slug_b].get("title", slug_b)
        flag = (
            "BLOCK" if score >= _TOPIC_KEY_BLOCK_THRESHOLD else
            "WARN " if score >= _TOPIC_KEY_WARN_THRESHOLD else
            "     "
        )
        print(f"  [{flag}] {score:.0%}  '{title_a}'  ≈  '{title_b}'")
    if not scores:
        print("  (no overlapping pairs at all — corpus shares zero topic terms pairwise)")

    blocked = [t for t in scores if t[2] >= _TOPIC_KEY_BLOCK_THRESHOLD]
    warned = [t for t in scores if _TOPIC_KEY_WARN_THRESHOLD <=
              t[2] < _TOPIC_KEY_BLOCK_THRESHOLD]

    print(
        f"\nWould BLOCK ({len(blocked)} pair(s), score >= "
        f"{_TOPIC_KEY_BLOCK_THRESHOLD:.2f}) beyond what's shown above, if any:"
    )
    for slug_a, slug_b, score in blocked[top_n:top_n + 30]:
        title_a = entries[slug_a].get("title", slug_a)
        title_b = entries[slug_b].get("title", slug_b)
        print(f"  {score:.0%}  '{title_a}'  ≈  '{title_b}'")

    print(
        f"\nWould WARN ({len(warned)} pair(s), "
        f"{_TOPIC_KEY_WARN_THRESHOLD:.2f} <= score < {_TOPIC_KEY_BLOCK_THRESHOLD:.2f}) "
        "beyond what's shown above, if any:"
    )
    for slug_a, slug_b, score in warned[top_n:top_n + 30]:
        title_a = entries[slug_a].get("title", slug_a)
        title_b = entries[slug_b].get("title", slug_b)
        print(f"  {score:.0%}  '{title_a}'  ≈  '{title_b}'")

    print(
        "\nLook at the top-N list above:\n"
        "  - If the highest pair(s) genuinely cover the same angle, pick a\n"
        "    threshold just below their score and set BLOCK there.\n"
        "  - If even the top pair is clearly a different topic, this corpus\n"
        "    may just be diverse — leaving the thresholds high (or this layer\n"
        "    rarely firing) is correct, not broken."
    )


if __name__ == "__main__":
    import sys as _sys

    if len(_sys.argv) >= 3 and _sys.argv[1] == "audit":
        _top_n = int(_sys.argv[3]) if len(_sys.argv) >= 4 else 20
        _audit_topic_key_threshold(_sys.argv[2], top_n=_top_n)
    else:
        print(
            "Usage: python similarity_guard.py audit <docs_dir> [top_n]\n"
            "  e.g. python similarity_guard.py audit ./docs\n"
            "       python similarity_guard.py audit ./docs 30\n"
            "Scores every pair of published posts using the CURRENT "
            "topic-key window/thresholds, always prints the top-N highest-\n"
            "scoring pairs (regardless of threshold) so you can calibrate "
            "even when nothing currently crosses WARN/BLOCK."
        )
