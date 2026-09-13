"""
adsense_fixes/topic_dedup.py
==============================
Second, independent duplicate-topic gate operating on TITLE + declared
meta_description keywords, rather than body text.

WHY THIS EXISTS
---------------
adsense_fixes/similarity_guard.py's topic-key score is measured on the
first ~150 words of the article BODY. Run against the live corpus
(`python similarity_guard.py audit ./docs`):

    Indexed posts : 501
    Score stats   : min=0.00 median=0.02 mean=0.03 max=0.30

Max pairwise score across the entire site is 30% — under even the 35%
WARN threshold — yet real duplicate-topic pairs exist, e.g. "AI
rollouts: feature flags in 2026" vs "AI rollouts live or die by
flags" (confirmed same angle, scores only 27%). This site's generator
paraphrases too aggressively at the sentence level for word-overlap on
body text to survive; the topic reuse is visible in the TITLE and
declared keywords, not the prose.

This module doesn't replace similarity_guard.py — it's a second,
independent signal at the intent level. A post can pass body-similarity
and still get blocked here if it's asking the same underlying question
as an existing post in different words.

HOW TO INTEGRATE
----------------
Add as a hard gate in blog_system.py's auto-mode pipeline, in the same
place SimilarityGuard is already checked, before save_post():

    from adsense_fixes.topic_dedup import check_topic_duplicate
    dup = check_topic_duplicate(candidate_title, candidate_meta_description, Path("./docs"))
    if dup:
        print(f"  🛑 Topic-key match ({dup['score']:.0%}) with existing post "
              f"'{dup['title']}' (/{dup['slug']}/) — regenerating with a different angle.")
        # same retry-with-different-angle flow SimilarityGuard already triggers

CLI (retroactive, read-only — finds existing clusters, changes nothing):
    python adsense_fixes/topic_dedup.py ./docs
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

_STOP = {
    "the", "a", "an", "in", "of", "for", "and", "or", "to", "is", "are",
    "with", "how", "why", "what", "your", "our", "when", "2026", "ai",
    "vs", "after", "before", "new", "we", "i", "still", "did", "not",
    "that", "this", "from", "into", "over", "under", "than", "then",
}

# Jaccard threshold, measured on TITLE keywords only. Tuned against this
# site's actual corpus (see module docstring). Note: meta_description was
# tried first and rejected — this generator writes a different invented
# statistic into every meta_description (e.g. "Cut AI rollout latency 84%"
# vs "Reduce AI incident minutes 40%" for the SAME underlying topic), so
# including it actively DILUTES the signal: the confirmed "AI rollouts"
# duplicate pair scores 50% on title alone but only 24% once meta is
# blended in. Title-only, checked against the live corpus:
#   501 posts, 5,824 non-zero pairs, 17 pairs >= 45% — matching manual
#   review almost exactly (16 pairs found by hand). meta_description is
#   kept in the corpus loader for future use but intentionally excluded
#   from the score itself.
_BLOCK_THRESHOLD = 0.45

_SKIP_DIRS = {"static", "tag", "author"}


def _keyset(title: str, meta: str) -> Set[str]:
    text = f"{title} {meta}"
    words = re.sub(r"[^\w\s]", " ", text.lower()).split()
    return {w for w in words if w not in _STOP and len(w) > 3}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _load_corpus(docs_dir: Path) -> List[Dict]:
    entries = []
    if not docs_dir.exists():
        return entries
    for d in sorted(docs_dir.iterdir()):
        if not d.is_dir() or d.name in _SKIP_DIRS:
            continue
        pj = d / "post.json"
        if not pj.exists():
            continue
        try:
            data = json.loads(pj.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        title = data.get("title", "")
        meta = data.get("meta_description", "")
        entries.append({
            "slug": d.name,
            "title": title,
            "meta": meta,
            # Title-only, deliberately excluding meta — see _BLOCK_THRESHOLD
            # comment above for why blending it in hurts the signal here.
            "keys": _keyset(title, ""),
            "words": len(data.get("content", "").split()),
        })
    return entries


# ── Live gate: call this before publishing a new candidate post ────────────

def check_topic_duplicate(
    candidate_title: str,
    candidate_meta: str,
    docs_dir: Path,
    exclude_slug: Optional[str] = None,
) -> Optional[Dict]:
    """
    Compare a candidate post's title+meta keyword set against every
    already-published post. Returns the highest-scoring match dict
    ({slug, title, score}) if it's >= _BLOCK_THRESHOLD, else None.
    """
    # Title-only, matching the corpus loader — see _BLOCK_THRESHOLD comment.
    candidate_keys = _keyset(candidate_title, "")
    if not candidate_keys:
        return None

    best = None
    for entry in _load_corpus(docs_dir):
        if exclude_slug and entry["slug"] == exclude_slug:
            continue
        score = _jaccard(candidate_keys, entry["keys"])
        if score >= _BLOCK_THRESHOLD and (best is None or score > best["score"]):
            best = {"slug": entry["slug"], "title": entry["title"], "score": score}

    return best


# ── Retroactive audit: find clusters in the existing corpus ────────────────

def find_clusters(docs_dir: Path) -> List[List[Dict]]:
    """
    Union-find clustering of all published posts by title+meta keyword
    Jaccard similarity at _BLOCK_THRESHOLD. Returns a list of clusters
    (each a list of entry dicts, longest word-count first), excluding
    singletons.
    """
    entries = _load_corpus(docs_dir)
    parent = {e["slug"]: e["slug"] for e in entries}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            a, b = entries[i], entries[j]
            if not a["keys"] or not b["keys"]:
                continue
            if _jaccard(a["keys"], b["keys"]) >= _BLOCK_THRESHOLD:
                union(a["slug"], b["slug"])

    clusters: Dict[str, List[Dict]] = {}
    for e in entries:
        clusters.setdefault(find(e["slug"]), []).append(e)

    result = [c for c in clusters.values() if len(c) > 1]
    for c in result:
        c.sort(key=lambda e: -e["words"])  # longest first == recommended KEEP
    return result


def _cli(docs_dir: Path) -> None:
    clusters = find_clusters(docs_dir)
    print(
        f"{len(clusters)} title-level duplicate cluster(s) found "
        f"(threshold {_BLOCK_THRESHOLD:.0%}, read-only — nothing changed):\n"
    )
    for c in clusters:
        keep = c[0]
        print(f"KEEP  {keep['words']:5d}w  {keep['slug']}  | {keep['title']}")
        for e in c[1:]:
            print(f"  DEL {e['words']:5d}w  {e['slug']}  | {e['title']}")
        print()

    if not clusters:
        print("No clusters at this threshold — corpus is diverse at the title/intent level.")


if __name__ == "__main__":
    docs = Path(sys.argv[1] if len(sys.argv) > 1 else "./docs")
    _cli(docs)