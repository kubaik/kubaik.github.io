"""
adsense_fixes/topic_diversity_gate.py
Reject topics that are too close to the last N published titles (corpus-level anti-farm).

Usage (inside blog_system.py generation path, before save_post):

    from adsense_fixes.topic_diversity_gate import is_too_similar
    too_close, match, score = is_too_similar(blog_post.title)
    if too_close:
        # discard draft and retry with a different topic
        raise DuplicateContentError(f"Topic too similar to '{match}' (score={score:.2f})")
"""

import json
import re
from pathlib import Path
from typing import List, Tuple

_STOP = {
    "a",
    "an",
    "the",
    "to",
    "in",
    "of",
    "for",
    "and",
    "or",
    "is",
    "are",
    "with",
    "how",
    "your",
    "my",
    "our",
    "its",
    "on",
    "at",
    "by",
    "from",
    "this",
    "that",
    "best",
    "using",
    "guide",
    "complete",
    "vs",
    "versus",
    "why",
    "when",
    "where",
    "what",
    "who",
    "most",
    "every",
}


def _tokens(title: str) -> set:
    words = re.sub(r"[^\w\s]", "", title.lower()).split()
    return {w for w in words if w not in _STOP and len(w) > 2}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def is_too_similar(
    new_title: str,
    docs_dir: Path = Path("./docs"),
    window: int = 40,
    threshold: float = 0.42,
) -> Tuple[bool, str, float]:
    """
    Return (too_close, nearest_title, score).
    Compares against the most recently modified post.json titles.
    """
    titles: List[str] = []
    if docs_dir.exists():
        candidates = [
            d
            for d in docs_dir.iterdir()
            if d.is_dir() and d.name not in ("static", "tag", "author")
        ]
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for d in candidates:
            pj = d / "post.json"
            if not pj.exists():
                continue
            try:
                t = json.loads(pj.read_text(encoding="utf-8")).get("title", "")
                if t:
                    titles.append(t)
                    if len(titles) >= window:
                        break
            except Exception:
                continue

    nt = _tokens(new_title)
    best, best_t = 0.0, ""
    for t in titles:
        s = _jaccard(nt, _tokens(t))
        if s > best:
            best, best_t = s, t
    return best >= threshold, best_t, best
