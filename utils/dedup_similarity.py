"""
dedup_similarity.py

Single source of truth for "how similar are two blog posts" across this
repo. Previously blog_system.py's ContentDuplicateGate (runs at publish
time) and content_quality_scanner.py's audit (runs offline) each had their
own hand-rolled TF-IDF implementation — different tokenizers, different
stopword lists, and critically, the gate used raw term-frequency with NO
IDF weighting while the scanner used full-corpus IDF plus 3x title
weighting. A comment in blog_system.py claimed "the standalone CI gate and
the live generation pipeline agree on what counts as a duplicate," but
that was only true of the numeric threshold (0.60 == 0.60) — the two
tools were computing similarity in different vector spaces entirely, so
the same threshold number meant different things in each place. That's
exactly why posts existed in the corpus at 0.73 similarity (per the
scanner) despite an active 0.60 gate at publish time: the gate's own
math scored that pair below its own threshold.

Both blog_system.py and content_quality_scanner.py now import from here.
There is exactly one tokenizer, one IDF calculation, and one cosine
function in this repo — change it here and both the publish-time gate and
the offline audit move together.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Tuple

# Unified stopword list (superset of the two lists that previously existed
# separately in blog_system.py and content_quality_scanner.py).
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for",
    "with", "is", "are", "was", "were", "be", "been", "being", "this",
    "that", "it", "its", "as", "at", "by", "from", "we", "you", "your",
    "i", "our", "will", "what", "how", "why", "when", "which", "these",
    "those", "not", "no", "do", "does", "did", "can", "could", "should",
    "would", "into", "than", "then", "so", "if", "just", "more", "most",
    "have", "has", "had", "may", "might", "they", "their", "who",
    "about", "up", "out", "also", "after", "before", "over", "some",
    "any", "all", "each", "both", "between", "through",
}

# DUPLICATE_SIMILARITY_THRESHOLD — how similar (0-1) two posts' bodies can
# be before they're treated as duplicates.
#
# PATCH (2026, dedup hardening round 2): previously 0.60. The offline
# audit found *confirmed* near-duplicate pairs already live in the
# corpus at 0.52-0.73 similarity under this exact vectorizer, meaning
# 0.60 was too loose to have caught them even if both tools had agreed
# on the math the whole time. Lowered to 0.45 so pairs like those don't
# require lucky wording to get caught. If this proves too aggressive in
# practice (i.e. it starts blocking genuinely distinct posts that just
# share a lot of domain vocabulary), raise it back up in small steps and
# re-run the audit rather than reverting to 0.60 outright.
DUPLICATE_SIMILARITY_THRESHOLD = 0.45

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, drop stopwords and 1-2 char tokens."""
    if not text:
        return []
    words = _TOKEN_RE.findall(text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def document_tokens(title: str, content: str) -> List[str]:
    """Title words count 3x — duplicate framing is very visible in titles
    for this repo's clusters, so weighting them up front makes near-dupes
    stand out from cosine similarity of the token multiset."""
    return tokenize(title or "") * 3 + tokenize(content or "")


def build_idf(all_doc_tokens: Iterable[List[str]]) -> Dict[str, float]:
    """Standard smoothed IDF: log(N / (1 + df)) + 1, computed once per
    corpus snapshot and reused for every vector built against it."""
    doc_lists = list(all_doc_tokens)
    n_docs = max(len(doc_lists), 1)
    df = Counter()
    for tokens in doc_lists:
        for term in set(tokens):
            df[term] += 1
    return {term: math.log(n_docs / (1 + count)) + 1 for term, count in df.items()}


def tfidf_vector(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    """Vectorize one document's tokens against a precomputed corpus IDF.
    Terms absent from the corpus (e.g. a brand-new candidate post's rare
    terms) fall back to an IDF of 1.0 rather than being dropped, so a new
    post's distinctive vocabulary still counts toward its own vector."""
    if not tokens:
        return {}
    tf = Counter(tokens)
    total = max(sum(tf.values()), 1)
    return {
        term: (count / total) * idf.get(term, 1.0)
        for term, count in tf.items()
    }


def cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[t] * b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def build_corpus_vectors(
    documents: Dict[str, Tuple[str, str]]
) -> Dict[str, Dict[str, float]]:
    """documents: {key: (title, content)} -> {key: tfidf_vector}.

    Builds one shared IDF table across every document passed in, then
    vectorizes each one against it. This is the function both
    ContentDuplicateGate (candidate + every already-published post) and
    the scanner's corpus-wide audit (every post at once) should call, so
    a given (title, content) pair always produces the same vector
    regardless of which tool is asking.
    """
    doc_tokens = {key: document_tokens(title, content)
                  for key, (title, content) in documents.items()}
    idf = build_idf(doc_tokens.values())
    return {key: tfidf_vector(tokens, idf) for key, tokens in doc_tokens.items()}


def most_similar(
    target_key: str,
    documents: Dict[str, Tuple[str, str]],
) -> Tuple[str, float]:
    """Convenience helper: given target_key's document is already in
    `documents`, return (best_matching_key, best_similarity) among all
    the others. Returns ("", 0.0) if there's nothing to compare against."""
    vectors = build_corpus_vectors(documents)
    target_vec = vectors.get(target_key, {})
    best_key, best_sim = "", 0.0
    for key, vec in vectors.items():
        if key == target_key:
            continue
        sim = cosine(target_vec, vec)
        if sim > best_sim:
            best_sim, best_key = sim, key
    return best_key, best_sim
