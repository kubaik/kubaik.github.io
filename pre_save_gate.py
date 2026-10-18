"""
pre_save_gate.py — drop-in, fail-closed gate for titles and topic overlap.

Covers what the existing gates miss:
  * _reject_if_narrative_title only fires on first-person+verb or
    verb+number. It passes "$150/month", "50k EPS", "300M users",
    "2 paths tested" etc.
  * check_topic_duplicate runs only in one CLI path and scores title
    similarity, so 3+ posts on one subject all got through.

API
  problem = title_problem(title, content)               -> Optional[str]
  problem = topic_overlap_problem(title, docs_dir, exclude_slug=None)
  pre_save_gate(post, docs_dir)                          -> raises PreSaveGateError
  check_before_save(title, content, docs_dir, slug=None) -> Optional[str]
      (string form for the generation retry loop)
"""

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

# Tuned against your audit output: dropped "in production", "real attacks",
# "real mistakes", "caught", "that worked" — those are descriptive, not claims
# of personal experience.
_EXPERIENCE = re.compile(
    r"\b(we|our|ours|i|my|me|us)\b|"
    r"\b(tested|tried|survived|blew up|broke our|what i|months in|"
    r"weeks in|years in|after \d+ (?:days?|weeks?|months?|years?)|we missed)\b",
    re.I,
)

# Quantified claims: % / x multipliers / money / k-M-B suffix / "N times"
_FIGURE = re.compile(
    r"[$€£]\s?\d[\d,.]*\s?(?:[kKmMbB]\b)?|"
    r"\b\d+(?:\.\d+)?\s?(?:%|x\b|[kKmMbB]\b|times\b|percent\b|million\b|billion\b|thousand\b)",
    re.I,
)
# Backing must be unambiguous: a link, an explicit assumption/illustrative
# label, or visible arithmetic ("3 x 40 = 120"). Generic words such as
# "example", "model", "budget", "target", "workload" appear in nearly every
# technical paragraph and made the old check pass everything.
_BACKING = re.compile(
    r"https?://|illustrat|hypothetical|assum(?:e|es|ed|ing|ption)|"
    r"back-of-the-envelope|worked example|rough estimate|"
    r"\d[\d,.]*\s*[x×*/+\-]\s*\$?\d[\d,.]*\s*=\s*\$?\d",
    re.I,
)

_STOP = {
    "the",
    "and",
    "for",
    "with",
    "your",
    "that",
    "this",
    "why",
    "how",
    "what",
    "when",
    "now",
    "still",
    "not",
    "you",
    "are",
    "can",
    "does",
    "vs",
    "from",
    "into",
    "over",
    "than",
    "when",
    "which",
    "most",
    "more",
    "less",
    "actually",
    "really",
    "real",
    "ways",
    "tools",
    "patterns",
    "mistakes",
    "traps",
    "rules",
    "skills",
    "metrics",
    "lessons",
    "things",
    "tips",
    "guide",
    "playbook",
    "checklist",
    "stack",
    "2025",
    "2026",
    "2027",
    "production",
    "teams",
    "systems",
    "beats",
    "break",
    "breaks",
}

TOPIC_MIN_SHARED = 3
TOPIC_MIN_RATIO = 0.5
TOPIC_MAX_EXISTING = 1  # allow the 2nd post on a subject, block the 3rd


class PreSaveGateError(ValueError):
    pass


def _figure_patterns(fig: str) -> List["re.Pattern"]:
    """Regexes that match this figure as a WHOLE token (never a substring of
    another number such as the 20 in 2026), including common spellings."""
    f = fig.strip()
    m = re.match(
        r"([$€£]?)\s?(\d[\d,.]*)\s?([kKmMbB]|million|billion|thousand|times|x|%|percent)?",
        f,
    )
    if not m:
        return [re.compile(re.escape(f), re.I)]
    cur, num, suf = m.group(1), m.group(2).rstrip(".,"), (m.group(3) or "").lower()
    n = re.escape(num)
    alts = []
    if cur:
        alts.append(re.escape(cur) + r"\s?" + n + r"(?:\s?[kKmMbB]\b)?")
    if suf in ("k", "thousand"):
        digits = num.replace(",", "")
        alts += [n + r"\s?(?:k|thousand)\b"]
        if digits.isdigit():
            alts.append(re.escape(f"{int(digits) * 1000:,}"))
    elif suf in ("m", "million"):
        alts += [n + r"\s?(?:m|million)\b"]
    elif suf in ("b", "billion"):
        alts += [n + r"\s?(?:b|billion)\b"]
    elif suf in ("%", "percent"):
        alts += [n + r"\s?(?:%|percent)"]
    elif suf in ("times", "x"):
        alts += [n + r"\s?(?:x|times)\b"]
    elif not cur:
        alts.append(n)
    return [re.compile(r"(?<![\d.,])(?:" + "|".join(alts) + r")(?![\d])", re.I)]


def _figure_supported(fig: str, body: str) -> bool:
    """Supported = the same figure appears in the body within 250 chars of a
    link, an explicit assumption/illustrative label, or visible arithmetic."""
    for rx in _figure_patterns(fig):
        for m in rx.finditer(body):
            w = body[max(0, m.start() - 250) : m.end() + 250]
            if _BACKING.search(w):
                return True
    return False


def title_problem(title: str, content: str) -> Optional[str]:
    if not title:
        return None
    m = _EXPERIENCE.search(title)
    if m:
        return (
            f"title implies first-hand experience ('{m.group(0)}'): " f"{title[:60]!r}"
        )
    for fm in _FIGURE.finditer(title):
        # bare list counts ("7 ways") are not matched by _FIGURE
        if not _figure_supported(fm.group(0), content):
            return (
                f"title figure '{fm.group(0).strip()}' has no link, calculation "
                f"or scenario label in the body: {title[:60]!r}"
            )
    return None


# ── topic overlap ───────────────────────────────────────────────────────────
_cache: Dict[str, object] = {"t": 0.0, "dir": None, "data": {}}


def _tokens(title: str) -> Set[str]:
    return {w for w in re.findall(r"[a-z]{3,}", title.lower()) if w not in _STOP}


def _existing(docs_dir: Path) -> Dict[str, Set[str]]:
    now = time.time()
    if _cache["dir"] == str(docs_dir) and now - float(_cache["t"]) < 30:
        return _cache["data"]  # type: ignore
    data: Dict[str, Set[str]] = {}
    for d in Path(docs_dir).iterdir():
        pj = d / "post.json"
        if d.is_dir() and pj.exists():
            try:
                t = json.loads(pj.read_text(encoding="utf-8")).get("title", "")
            except Exception:
                continue
            data[d.name] = _tokens(t or d.name.replace("-", " "))
    _cache.update(t=now, dir=str(docs_dir), data=data)
    return data


def topic_overlap_problem(
    title: str, docs_dir: Path, exclude_slug: Optional[str] = None
) -> Optional[str]:
    mine = _tokens(title)
    if len(mine) < 2:
        return None
    hits = []
    for slug, toks in _existing(Path(docs_dir)).items():
        if slug == exclude_slug or not toks:
            continue
        shared = mine & toks
        if (
            len(shared) >= TOPIC_MIN_SHARED
            and len(shared) / min(len(mine), len(toks)) >= TOPIC_MIN_RATIO
        ):
            hits.append((slug, sorted(shared)))
    if len(hits) > TOPIC_MAX_EXISTING:
        names = ", ".join(s for s, _ in hits[:4])
        return (
            f"topic already covered by {len(hits)} published posts ({names}); "
            f"another post on this subject splits ranking signals and reads as scaled content"
        )
    return None


def check_before_save(
    title: str, content: str, docs_dir: Path, slug: Optional[str] = None
) -> Optional[str]:
    return title_problem(title, content) or topic_overlap_problem(title, docs_dir, slug)


def pre_save_gate(post, docs_dir: Path) -> None:
    problem = check_before_save(
        post.title, post.content, Path(docs_dir), getattr(post, "slug", None)
    )
    if problem:
        raise PreSaveGateError(f"Refusing to save '{post.title}': {problem}")
