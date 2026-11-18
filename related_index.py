"""
related_index.py
================
Code-driven internal linking that scales to thousands of posts.

Replaces the per-request pure-Python TF-IDF cosine in static_site_generator
(O(N^2) dict intersections, with both vector norms recomputed for every pair:
~9M pair evaluations at N=3000, i.e. minutes-to-hours per build).

This builds ONE sparse TF-IDF matrix and computes top-k neighbours in row
chunks (memory bounded), so a build is O(N * nnz) and takes seconds.

Quality rules (internal links are an editorial signal, not filler):
  * Code fences, URLs, the generated "About this article" footer and
    "[CLAIM REMOVED]" markers are stripped before vectorising, so shared
    boilerplate cannot create fake similarity.
  * Title terms are weighted 3x.
  * Bigrams are included, so "connection pooling" beats "connection" + "pooling".
  * max_df drops terms that appear in >40% of posts (site-wide vocabulary).
  * Tag overlap is a small tie-break boost only.
  * Fewer than k results (even zero) is correct: no padding with unrelated posts.
"""

import re
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

RELATED_MIN_SIM = 0.08  # cosine floor (sublinear-tf + bigram scale)
TAG_BOOST = 0.02

_FENCE = re.compile(r"```.*?```", re.S)
_FOOTER = re.compile(r"\n---\s*\n+### About this article.*\Z", re.S)
_CLAIM = re.compile(r"[ \t]*\[CLAIM REMOVED[^\]]*\]")
_URL = re.compile(r"https?://\S+")
_NONWORD = re.compile(r"[^A-Za-z' ]+")


def clean_for_similarity(text: str) -> str:
    t = text or ""
    t = _FOOTER.sub("", t)
    t = _FENCE.sub(" ", t)
    t = _CLAIM.sub(" ", t)
    t = _URL.sub(" ", t)
    return _NONWORD.sub(" ", t)


def build_related_index(
    posts: Sequence,
    stop_words: Optional[Iterable[str]] = None,
    top_k: int = 8,
    min_sim: float = RELATED_MIN_SIM,
    chunk: int = 512,
) -> Optional[Dict[str, List[Tuple[float, str]]]]:
    """slug -> [(score, other_slug), ...] best first.

    Returns None if scikit-learn/numpy are unavailable so the caller can fall
    back to its legacy implementation instead of failing the build.
    """
    try:
        import numpy as np
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
    except Exception:
        return None

    n = len(posts)
    if n < 2:
        return {p.slug: [] for p in posts}

    slugs: List[str] = []
    docs: List[str] = []
    tags: List[Set[str]] = []
    for p in posts:
        title = getattr(p, "full_title", None) or p.title
        docs.append((f"{title} " * 3) + clean_for_similarity(p.content))
        slugs.append(p.slug)
        tags.append({t.lower() for t in (p.tags or [])})

    stops = set(ENGLISH_STOP_WORDS) | set(stop_words or ())
    vec = TfidfVectorizer(
        lowercase=True,
        stop_words=sorted(stops),
        token_pattern=r"(?u)\b[a-z][a-z']+\b",
        sublinear_tf=True,
        ngram_range=(1, 2),
        min_df=2 if n >= 20 else 1,
        max_df=0.4 if n >= 20 else 1.0,
        max_features=200_000,
        dtype=np.float32,
    )
    try:
        X = vec.fit_transform(docs)  # L2-normalised rows
    except ValueError:  # empty vocabulary
        return {s: [] for s in slugs}

    index: Dict[str, List[Tuple[float, str]]] = {}
    k_cand = min(top_k * 3, n - 1)
    for start in range(0, n, chunk):
        sims = (X[start : start + chunk] @ X.T).toarray()
        for i in range(sims.shape[0]):
            gi = start + i
            row = sims[i]
            row[gi] = -1.0
            cand = (
                np.argpartition(-row, k_cand - 1)[:k_cand] if k_cand < n else range(n)
            )
            scored: List[Tuple[float, str]] = []
            for j in cand:
                sim = float(row[j])
                if sim <= 0.0:
                    continue
                sim += TAG_BOOST * len(tags[gi] & tags[j])
                if sim >= min_sim:
                    scored.append((sim, slugs[j]))
            scored.sort(key=lambda x: (-x[0], x[1]))  # deterministic
            index[slugs[gi]] = scored[:top_k]
    return index
