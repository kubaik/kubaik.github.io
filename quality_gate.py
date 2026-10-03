#!/usr/bin/env python3
"""
quality_gate.py - single source of truth for "would this topic/post hurt AdSense approval?"

Used by blog_system.py (generation), delete_posts.py (blocklist writer) and improve_posts.py (validator).

  * Blocklist   : blocked_topics.json (repo root, committed). Written by delete_posts.py; every deleted post's
                  title/slug is remembered so the pipeline can never regenerate it.
  * Topic gate  : rejects first-person / numeric-claim topics ("How we cut our bill 68%") before any LLM call.
  * Post gate   : rejects drafts whose density of invented first-hand claims / unverifiable products is high.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

BLOCKLIST_PATH = Path("blocked_topics.json")

_STOP = {
    "the",
    "a",
    "an",
    "of",
    "to",
    "in",
    "for",
    "and",
    "or",
    "on",
    "with",
    "is",
    "are",
    "your",
    "you",
    "how",
    "why",
    "what",
    "when",
    "vs",
    "from",
    "at",
    "by",
    "it",
    "that",
    "this",
    "do",
    "does",
    "not",
    "now",
    "2026",
}

# first-person / incident-style / numeric-outcome topics
_FIRST_PERSON = re.compile(
    r"\b(we|our|us|i|my|me|ours|we've|we're)\b|\bhow we\b|numbers from our|we know\b",
    re.I,
)
_NUMERIC_CLAIM = re.compile(
    r"\d\s*%|\$\s?\d|\d+(?:\.\d+)?\s?(?:ms|s)\b\s*(?:to|→)|\b\d+\s?(?:k|K|TB|MAU|QPS)\+?|\b\d+x\b|\b\d{2,}\+|\bfrom \d"
)


# implicit first-hand/incident narratives: "that worked", "the production incident caused by...", "real metrics"
# push the LLM to invent a story and a result even without a pronoun
_NARRATIVE = re.compile(
    r"\bthat (?:actually |finally |quietly )?(?:worked|restored|reduced|cut|saved|killed|made|broke)\b"
    r"|\bwhat (?:actually )?(?:worked|broke|happened|leaked)\b|\b(?:real|actual) (?:metrics|numbers|results)\b"
    r"|\bincident (?:caused|that)\b|\bthe production incident\b|\bpost-?mortem\b|\bwar stor",
    re.I,
)


def _tokens(text: str) -> set:
    return {
        w
        for w in re.findall(r"[a-z0-9]+", text.lower())
        if w not in _STOP and len(w) > 1
    }


# ── blocklist ───────────────────────────────────────────────────────────────
def load_blocklist(path: Path = BLOCKLIST_PATH) -> List[Dict]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def add_to_blocklist(entries: List[Dict], path: Path = BLOCKLIST_PATH) -> int:
    existing = load_blocklist(path)
    seen = {e.get("slug") for e in existing}
    new = [e for e in entries if e.get("slug") not in seen]
    if new:
        Path(path).write_text(
            json.dumps(existing + new, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return len(new)


def blocked_match(
    candidate: str, blocklist: Optional[List[Dict]] = None, threshold: float = 0.5
) -> Optional[Dict]:
    """Return the blocklist entry a topic/title is too close to, else None.
    Token Jaccard >= threshold, or >= 75% of the smaller token set contained in the other (min 3 tokens).
    """
    bl = load_blocklist() if blocklist is None else blocklist
    ct = _tokens(candidate)
    if len(ct) < 2:
        return None
    for e in bl:
        for ref in (e.get("title", ""), e.get("slug", "").replace("-", " ")):
            rt = _tokens(ref)
            if len(rt) < 2:
                continue
            inter = len(ct & rt)
            if inter / len(ct | rt) >= threshold:
                return e
            if min(len(ct), len(rt)) >= 3 and inter / min(len(ct), len(rt)) >= 0.75:
                return e
    return None


# ── topic gate ──────────────────────────────────────────────────────────────
def topic_problem(topic: str, blocklist: Optional[List[Dict]] = None) -> Optional[str]:
    if _FIRST_PERSON.search(topic):
        return "first-person topic (implies first-hand experience the author cannot substantiate)"
    if _NUMERIC_CLAIM.search(topic):
        return "topic embeds a numeric outcome claim"
    if _NARRATIVE.search(topic):
        return (
            "topic is an incident/outcome narrative (invites invented first-hand story)"
        )
    hit = blocked_match(topic, blocklist)
    if hit:
        return f"matches previously deleted post '{hit.get('slug')}'"
    return None


def filter_topics(
    topics: List[str], blocklist: Optional[List[Dict]] = None
) -> List[str]:
    bl = load_blocklist() if blocklist is None else blocklist
    return [t for t in topics if not topic_problem(t, bl)]


# ── post gate ───────────────────────────────────────────────────────────────
def fabrication_problem(
    title: str, content: str, max_density: float = 0.10, max_unverified: int = 3
) -> Optional[str]:
    """Reject drafts dominated by invented first-hand evidence. Same scorer triage.py used to pick the DELETE list."""
    import triage  # local import: triage imports entity_gate

    sc = triage.score({"title": title, "content": content})
    if sc["fab_density"] >= max_density:
        return (
            f"invented first-hand evidence density {sc['fab_density']:.2f} >= {max_density} "
            f"({sc['exp_claims']} experience claims, {sc['unverified_products']} unverifiable products)"
        )
    if sc["title_first"] and sc["anec"] >= 3:
        return "title promises first-hand results"
    return None


def blocked_title_problem(
    title: str, blocklist: Optional[List[Dict]] = None
) -> Optional[str]:
    hit = blocked_match(title, blocklist, threshold=0.6)
    return f"title too close to deleted post '{hit['slug']}'" if hit else None
