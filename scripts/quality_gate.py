#!/usr/bin/env python3
"""
scripts/quality_gate.py
=======================
Full-corpus content quality scanner.

Scores every post in docs/ against a composite quality rubric, writes a
CSV report, and appends any post scoring below --min-score to
regeneration_queue.json for the existing automated regeneration pipeline
to consume.

NON-NEGOTIABLE BEHAVIOUR
------------------------
  1. Report-only. Never fails the job. Never blocks a deploy. Never
     deletes a post. Deletion lives in content_quality_scanner.py's own
     prune/dedupe modes (--confirm required) and in workflow_dispatch.yml's
     report-only watchdogs, not here.
  2. Never modifies post.json. The only files this script writes are
     the CSV report (--csv) and appends to the queue file (--queue).
  3. Idempotent. Re-running does not duplicate queue entries that are
     already present for the same slug.
  4. Degrades gracefully. If utils/content_quality_scanner.py is
     importable and exposes a usable score function, that scorer is used.
     Otherwise, a fallback composite scorer running the same checks
     blog_system.py's _validate_content_quality() enforces is used
     instead. Either way, the output schema is identical.

SCORING (0-100, higher = better)
--------------------------------
The composite score starts at 100 and deducts for each detected issue:

  Weight  Check
  ------  ------------------------------------------------------------
   40     Word count below 1800 (thin content)
   25     Fabricated-citation pattern (named-source, no URL nearby)
   20     First-person / narrative incident in body
   25     Fabricated-incident or unverifiable-number narrative in TITLE
   15     Boilerplate fallback markers (template leakage)
   10     AI-filler phrase ("dive into", "game-changer", etc.)
    8     No fenced code block
    8     No version-pinned tool reference
    5     No markdown table
    5     No FAQ section
    5     E-E-A-T footer missing ("### About this article")
    3     Generic opener in first 200 chars
    8     Vague attribution ("The report showed" / "studies suggest")
           — this is a weaker signal than a named source, but it's a
           claim being made with no verifiable referent either way.

Threshold defaults to 50. Tuning is done via --min-score, not by
editing this file.

FIX (2026-09-21): the first version of this scorer was too narrow to
catch the actual failure modes in the corpus. It missed:
  - Fabricated-incident narratives in the TITLE (never scanned).
  - Body sentences where the incident pattern appears mid-sentence
    instead of at position 0 (regex was ^-anchored).
  - Narrative-number claims using verbs like "cost us", "burned",
    "survived" that weren't in the verb list.
  - Vague attribution like "The report showed" (flagged as a named
    citation, which it isn't).
This rewrite adds three new detection layers (title, mid-sentence
narrative, vague-attribution) and fixes the false-positive on generic
attribution. Weighting is tuned so a post with a fabricated title
scores in the 40-60 range, not 100.

Usage:
    python scripts/quality_gate.py --docs-dir ./docs --min-score 50 \\
        --csv ./quality_report_full.csv --queue ./regeneration_queue.json
    python scripts/quality_gate.py --docs-dir ./docs --no-queue  # report only
    python scripts/quality_gate.py --docs-dir ./docs --slug my-post  # single
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ── Optional imports — degrade gracefully ──────────────────────────────
try:
    from adsense_fixes.claim_gate import check_claims as _check_claims
except Exception:  # pragma: no cover
    _check_claims = None

try:
    from adsense_fixes.policy_risk import topic_policy_violation as _policy_violation
except Exception:  # pragma: no cover
    _policy_violation = None

# Scorer provided by content_quality_scanner — preferred if importable.
try:
    from utils.content_quality_scanner import score_post as _external_scorer
except Exception:  # pragma: no cover
    try:
        from utils.content_quality_scanner import score_content as _external_scorer
    except Exception:
        _external_scorer = None


# ── Fallback scorers (used when content_quality_scanner is unavailable) ─
_SKIP_DIRS = {"static", "tag", "author", "page"}

_BOILERPLATE_MARKERS = [
    "class {topic_slug}Client",
    "class Client:",
    "max_retries = config.get",
    "{topic_slug}",
    "{topic}",
    "topic_slug",
]

_AI_FILLER_PHRASES = [
    "in today's rapidly evolving",
    "in the ever-changing landscape",
    "in today's fast-paced",
    "in the ever-evolving",
    "harness the power of",
    "unlock the potential of",
    "paradigm shift",
    "game-changer",
    "revolutionize",
    "state-of-the-art",
    "cutting-edge technology",
    "dive into",
    "delve into",
    "it's important to note",
    "needless to say",
    "comprehensive guide",
    "this article will",
    "we will explore",
    "in conclusion",
    "let's explore",
    "let's dive",
    "look no further",
    "in this blog post",
    "stay tuned",
]

# ── Fabricated citations: NAMED source only ────────────────────────────
# A named-source pattern must have an actual proper-noun-looking token in
# the source slot. "The report showed" is intentionally NOT matched here —
# it's vague attribution (handled by _VAGUE_ATTRIBUTION_RE below), not a
# fabricated citation to a specific organisation, and flagging it as one
# was producing false positives that pushed already-passable posts below
# threshold.
_FABRICATED_CITATION_RE = re.compile(
    # "a/the 2026 <study|survey|report> by/from <Capitalized Name>"
    r"\b(?:a|an|the)\s+(?:20\d\d\s+)?"
    r"(?:study|paper|report|research|survey|analysis|benchmark)\s+"
    r"(?:by|from)\s+([A-Z][\w&.'-]+(?:\s+[A-Z][\w&.'-]+){0,4})|"
    # "according to <Capitalized Name>"
    r"\baccording to\s+(?:a\s+|an\s+|the\s+)?"
    r"([A-Z][\w&.'-]+(?:\s+[A-Z][\w&.'-]+){0,4})|"
    # "<Org>'s <study|research|red-team|findings>"
    r"\b([A-Z][\w&.'-]+)'s\s+"
    r"(?:study|research|red[\s-]?team|findings|report|analysis|survey)\b|"
    # "<NamedSource> survey/study that found/shows/tracked"
    r"\b(?:20\d\d\s+)?([A-Z][\w&.'-]+(?:\s+[A-Z][\w&.'-]+){0,3})\s+"
    r"(?:survey|study|report|analysis|benchmark)\s+"
    r"(?:that\s+|which\s+)?(?:found|shows?|tracked|reveals?|says?|showed)\b",
)

# Vague attribution is a real quality signal (a claim is made with no
# verifiable referent), but it's NOT the same failure mode as naming a
# real organisation. Scored at a smaller weight so it doesn't drown out
# the named-citation and narrative checks.
_VAGUE_ATTRIBUTION_RE = re.compile(
    r"\b(?:The|A|An|Recent|New|Industry|Independent)\s+"
    r"(?:report|survey|study|analysis|benchmark|data|research)\s+"
    r"(?:showed|found|revealed|suggested|indicated|said|estimated|tracked)\b",
    re.IGNORECASE,
)

# ── First-person / narrative incident patterns (BODY) ──────────────────
# Broadened from the original:
#   - No longer anchored at ^ — matches anywhere in the sentence, since
#     production posts frequently embed these mid-sentence.
#   - Verb list extended to include narrative-cost and scale verbs that
#     were slipping through: cost us, burned, survived, wasted, lost,
#     took us/me, flooded, ended up, kept [verb]-ing.
_FIRST_PERSON_INCIDENT_RE = re.compile(
    r"(?:"
    # "I <verb>" / "I've <verb>" / "I've been <verb>ing" with a broad
    # but concrete-claim verb set.
    r"\bI(?:'ve|'m|\s+have)?\s+(?:spent|built|shipped|deployed|migrated|"
    r"worked\s+(?:on|with|at)|ran\s+into|debugged|broke|fixed|caught|"
    r"hit\s+a|dealt\s+with|went\s+through|had\s+to|ended\s+up|"
    r"kept\s+seeing|kept\s+hitting|kept\s+running|was\s+surprised|"
    r"watched|witnessed|saw\s+firsthand|actually\s+used|regret\s+shipping|"
    r"regret\s+building|regret\s+picking|cost\s+us|cost\s+me|"
    r"burned|survived|wasted|lost|took\s+us|took\s+me|"
    r"flooded|had\s+to\s+roll\s+back)\b"
    r"|"
    # "A colleague" / "This took me" openers.
    r"\bA\s+colleague\b|"
    r"\bThis\s+took\s+me\b|"
    # "cost us <duration>" / "burned $N" / "survived <count>" — narrative
    # cost/scale claims with a leading verb (no pronoun required).
    r"\b(?:cost|saved|burned|wasted|lost|spent)\s+"
    r"(?:us|me|our|my|the\s+team)?\s*"
    r"(?:three|two|four|five|six|seven|eight|nine|ten|\d+)\s+"
    r"(?:hours?|days?|weeks?|months?|years?)\b|"
    r"\b(?:burned|saved|cost|spent)\s+\$\d+(?:k|K|,\d+)?\b|"
    r"\bsurvived\s+(?:[\d,]+|\d+k|\d+\s+thousand)\s+\w+"
    r")",
    re.IGNORECASE,
)

# ── Title-level fabricated-narrative detection (NEW) ───────────────────
# Titles that describe a specific past-tense first-person action ("Added
# circuit breakers 3 days after...") or that lead with a first-person
# pronoun + concrete verb ("I actually used...", "I regret shipping...")
# are the shape of a fabricated case study. The body-level check never
# sees them because they're in the title, not the content.
_TITLE_NARRATIVE_VERBS = (
    r"Added|Killed|Saved|Cut|Dumped|Launched|Survived|Burned|"
    r"Replaced|Grew|Found|Built|Shipped|Migrated|Reduced|Trimmed|"
    r"Dropped|Stopped|Started|Kept|Fixed|Broke|Caught|Made|Lost|"
    r"Gained|Spent|Wasted|Survived"
)
_TITLE_INCIDENT_PREFIX_RE = re.compile(
    r"^(?:" + _TITLE_NARRATIVE_VERBS + r")\b",
    re.IGNORECASE,
)
_TITLE_FIRST_PERSON_RE = re.compile(
    r"\b(?:I|We|My|Our|Us)\b\s+"
    r"(?:actually|regret|hit|spent|built|shipped|migrated|"
    r"debugged|survived|burned|killed|launched|added|replaced|"
    r"cut|saved|dumped|found|grew|reduced|trimmed|dropped|"
    r"lost|wasted|caught|broke|fixed)\b",
    re.IGNORECASE,
)
_TITLE_SPECIFIC_NUMBER_RE = re.compile(
    r"\$\d+(?:k|K|,\d+)?(?:/?(?:month|mo|year|yr|week|wk))?|"
    r"\b\d+\s*(?:%|hours?|days?|weeks?|months?)\b|"
    r"\b\d{1,3}(?:,\d{3})+\s+(?:agents|requests|users|events)\b",
)
_TITLE_SPECIFIC_PERCENT_RE = re.compile(r"\b\d{1,3}\s*%\b")

# Body-level narrative-number claims (mid-sentence).
_BODY_NUMBER_NARRATIVE_RE = re.compile(
    r"\b(?:cost|saved|burned|spent|wasted|lost|gained|"
    r"cut|reduced|added|shipped|migrated|built|survived)\s+"
    r"(?:us|me|our|my|the\s+team)?\s*"
    r"(?:\$\d+(?:k|K|,\d+)?|"
    r"\d+\s*(?:hours?|days?|weeks?|months?|years?)|"
    r"[\d,]+\s*(?:agents|requests|users|events|repos?))\b",
    re.IGNORECASE,
)

_VERSIONED_TOOL_RE = re.compile(
    r"\b([A-Za-z][A-Za-z0-9+.#_-]{1,24})\s+v?(\d{1,3}(?:\.\d{1,3}){0,2})"
    r"(?:\s*(?:LTS|lts))?\b",
)

_GENERIC_OPENERS = (
    "in this", "today we", "welcome to", "this guide covers",
    "if you're looking", "are you looking", "have you ever",
    "whether you're a beginner", "this post will",
)

_CODE_FENCE_RE = re.compile(r"```[\s\S]*?```")
_INLINE_CODE_RE = re.compile(r"`[^`]+`")


# ── Data model ─────────────────────────────────────────────────────────
@dataclass
class PostScore:
    slug: str
    score: int
    word_count: int
    issues: list = field(default_factory=list)
    title: str = ""
    created_at: str = ""


def _strip_code(text: str) -> str:
    text = _CODE_FENCE_RE.sub(" ", text)
    text = _INLINE_CODE_RE.sub(" ", text)
    return text


def _is_legitimate_source_window(window: str) -> bool:
    """Return True if a citation-looking match is actually referencing
    primary documentation or a real URL — those aren't fabricated and
    shouldn't be penalised."""
    if "http://" in window or "https://" in window:
        return True
    if re.search(
        r"\b(documentation|docs\b|changelog|release notes|readme|"
        r"rfc|specification|error message|log output|github\.com)\b",
        window, re.IGNORECASE,
    ):
        return True
    return False


def _score_title(slug: str, title: str, issues: list) -> int:
    """
    Title-only detection layer. Returns points to deduct (0 or more).
    All three checks are gated on the title being a narrative-shaped
    claim, not just "has a number" — a title like "Postgres 17" is fine,
    but "Saved $3k/month by ditching Kubernetes" is not.
    """
    deduct = 0
    if not title:
        return 0

    has_first_person = bool(_TITLE_FIRST_PERSON_RE.search(title))
    has_narrative_verb = bool(_TITLE_INCIDENT_PREFIX_RE.match(title.strip()))
    has_specific_number = bool(
        _TITLE_SPECIFIC_NUMBER_RE.search(title)
        or _TITLE_SPECIFIC_PERCENT_RE.search(title)
    )

    if has_first_person:
        deduct += 20
        issues.append("title: first-person narrative claim")

    if has_narrative_verb:
        deduct += 15
        issues.append("title: past-tense action narrative")

    # A narrative title that ALSO carries a specific number claim
    # (percentage, dollar figure, time span) is the strongest signal —
    # that's the shape of an invented case study.
    if has_specific_number and (has_first_person or has_narrative_verb):
        deduct += 15
        issues.append("title: specific-number claim in narrative title")

    return deduct


def _score_post_fallback(slug: str, data: dict) -> PostScore:
    """
    Inline composite scorer. Runs the same checks blog_system.py's
    _validate_content_quality() enforces, collapsed into a single 0-100
    score, PLUS the title-level and mid-sentence narrative checks that
    the previous version was missing.
    """
    content = data.get("content", "") or ""
    title = data.get("title", "") or ""
    issues: list = []
    score = 100

    wc = len(content.split())
    if wc < 1800:
        score -= 40
        issues.append(f"thin content ({wc} words < 1800)")

    # ── Title-level checks (run first — most damaging signal) ──────────
    score -= _score_title(slug, title, issues)

    # ── Body checks ────────────────────────────────────────────────────
    stripped = _strip_code(content)

    # Named-source fabrication (URL / primary-docs window excluded)
    named_citation_hit = False
    for m in _FABRICATED_CITATION_RE.finditer(stripped):
        window = stripped[max(0, m.start() - 200): m.end() + 200]
        if _is_legitimate_source_window(window):
            continue
        named_citation_hit = True
        score -= 25
        issues.append(f"fabricated citation: {m.group(0)[:60]!r}")
        break

    # Vague attribution (weaker signal — 8 points, separate from above)
    if not named_citation_hit:
        vm = _VAGUE_ATTRIBUTION_RE.search(stripped)
        if vm:
            score -= 8
            issues.append(f"vague attribution: {vm.group(0)[:60]!r}")

    # First-person / narrative incident (mid-sentence, broadened verbs)
    body_incident_hit = False
    for sent in re.split(r"(?<=[.!?])\s+", stripped):
        sent = sent.strip()
        if len(sent) < 15:
            continue
        if _FIRST_PERSON_INCIDENT_RE.search(sent):
            score -= 20
            issues.append(f"first-person incident: {sent[:80]!r}")
            body_incident_hit = True
            break

    # Narrative-number claim (mid-sentence, verb + unit)
    if not body_incident_hit:
        nm = _BODY_NUMBER_NARRATIVE_RE.search(stripped)
        if nm:
            score -= 15
            issues.append(f"narrative number claim: {nm.group(0)[:60]!r}")

    # Boilerplate markers
    for marker in _BOILERPLATE_MARKERS:
        if marker in content:
            score -= 15
            issues.append(f"boilerplate: {marker!r}")
            break

    # AI filler
    lower = content.lower()
    detected_filler = [p for p in _AI_FILLER_PHRASES if p in lower]
    if detected_filler:
        score -= 10
        issues.append(
            f"AI filler ({len(detected_filler)}): {detected_filler[0]!r}"
        )

    # Code / versions / tables / FAQ / E-E-A-T
    if content.count("```") < 2:
        score -= 8
        issues.append("no fenced code block")
    if not _VERSIONED_TOOL_RE.search(content):
        score -= 8
        issues.append("no version-pinned tool")
    if "|" not in content:
        score -= 5
        issues.append("no markdown table")
    if "## faq" not in lower and "frequently asked questions" not in lower:
        score -= 5
        issues.append("no FAQ section")
    if "### About this article" not in content:
        score -= 5
        issues.append("E-E-A-T footer missing")

    # Generic opener
    first_200 = content[:200].lower()
    for opener in _GENERIC_OPENERS:
        if first_200.startswith(opener) or f"\n{opener}" in first_200:
            score -= 3
            issues.append(f"generic opener: {opener!r}")
            break

    if _policy_violation:
        try:
            v = _policy_violation(title)
            if v:
                score -= 25
                issues.append(f"policy risk: {v}")
        except Exception:
            pass

    if _check_claims:
        try:
            res = _check_claims(content, title)
            if res.blocked:
                for r in res.reasons:
                    score -= 15
                    issues.append(f"claim gate: {r}")
        except Exception:
            pass

    score = max(0, min(100, score))
    return PostScore(
        slug=slug,
        score=score,
        word_count=wc,
        issues=issues,
        title=title,
        created_at=data.get("created_at", ""),
    )


def _score_post(slug: str, data: dict) -> PostScore:
    """
    Score a single post. Prefers content_quality_scanner if importable;
    falls back to the inline composite scorer.
    """
    if _external_scorer is not None:
        try:
            result = _external_scorer(data)
            if isinstance(result, dict) and "score" in result:
                return PostScore(
                    slug=slug,
                    score=int(result["score"]),
                    word_count=len((data.get("content") or "").split()),
                    issues=list(result.get("issues", [])),
                    title=data.get("title", ""),
                    created_at=data.get("created_at", ""),
                )
        except Exception as e:
            print(
                f"  WARN external scorer raised on {slug} ({e}); "
                f"falling back to inline scorer"
            )
    return _score_post_fallback(slug, data)


# ── Corpus walk ────────────────────────────────────────────────────────
def _load_corpus(docs_dir: Path, only_slug: str = None):
    posts = []
    if not docs_dir.exists():
        return posts
    for post_dir in sorted(docs_dir.iterdir()):
        if not post_dir.is_dir() or post_dir.name in _SKIP_DIRS:
            continue
        if only_slug and post_dir.name != only_slug:
            continue
        pj = post_dir / "post.json"
        if not pj.exists():
            continue
        try:
            data = json.loads(pj.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
            print(f"  WARN skipping unreadable {pj}: {e}")
            continue
        if not data.get("content"):
            continue
        posts.append((post_dir.name, data))
    return posts


# ── Queue integration ──────────────────────────────────────────────────
def _load_queue(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        return {entry.get("slug", ""): entry for entry in raw if entry.get("slug")}
    return {}


def _write_queue(path: Path, queue: dict) -> None:
    path.write_text(
        json.dumps(queue, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _append_to_queue(queue: dict, post: PostScore, min_score: int) -> bool:
    existing = queue.get(post.slug)
    reason = (
        f"quality_gate: score {post.score}/100 < {min_score}. "
        f"Issues: {'; '.join(post.issues[:5]) or 'none recorded'}"
    )

    if existing is None:
        queue[post.slug] = {
            "slug": post.slug,
            "title": post.title,
            "reason": reason,
            "status": "unresolved",
            "source": "quality_gate",
            "score": post.score,
            "queued_at": datetime.now().isoformat(),
        }
        return True

    if existing.get("source") == "quality_gate":
        if existing.get("score") != post.score or existing.get("reason") != reason:
            existing["score"] = post.score
            existing["reason"] = reason
            existing["title"] = post.title
            return True
    return False


# ── Report writer ──────────────────────────────────────────────────────
def _write_csv(path: Path, scores) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "slug", "score", "word_count", "title", "created_at",
            "issue_count", "issues",
        ])
        for s in sorted(scores, key=lambda x: x.score):
            w.writerow([
                s.slug,
                s.score,
                s.word_count,
                s.title,
                s.created_at,
                len(s.issues),
                " | ".join(s.issues[:8]),
            ])


# ── Entry point ────────────────────────────────────────────────────────
def run(
    docs_dir: Path,
    min_score: int,
    csv_path,
    queue_path,
    only_slug: str = None,
) -> int:
    print(f"Quality gate: scanning {docs_dir} (min score {min_score})")
    corpus = _load_corpus(docs_dir, only_slug)
    if not corpus:
        print("  no posts found — nothing to score")
        return 0

    scores = [_score_post(slug, data) for slug, data in corpus]
    below = [s for s in scores if s.score < min_score]
    scores_sorted = sorted(scores, key=lambda s: s.score)

    print(f"  scored {len(scores)} posts")
    print(f"  below threshold ({min_score}): {len(below)}")
    if scores_sorted:
        print(
            f"  score range: {scores_sorted[0].score} - {scores_sorted[-1].score}"
        )

    if below:
        print("\n  Worst 15:")
        for s in below[:15]:
            print(
                f"    {s.score:3d}  {s.slug}  "
                f"({'; '.join(s.issues[:2]) or 'no issues listed'})"
            )

    if csv_path is not None:
        _write_csv(csv_path, scores)
        print(f"\n  CSV report written: {csv_path}")

    if queue_path is not None:
        queue = _load_queue(queue_path)
        added = 0
        for s in below:
            if _append_to_queue(queue, s, min_score):
                added += 1
        if added:
            _write_queue(queue_path, queue)
            print(f"  queued {added} post(s) into {queue_path}")
        else:
            print(f"  queue unchanged ({queue_path})")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Full-corpus content quality scanner (report-only)."
    )
    parser.add_argument("--docs-dir", default="./docs")
    parser.add_argument("--min-score", type=int, default=50)
    parser.add_argument("--csv", default="./quality_report_full.csv")
    parser.add_argument("--queue", default="./regeneration_queue.json")
    parser.add_argument("--no-csv", action="store_true")
    parser.add_argument("--no-queue", action="store_true")
    parser.add_argument("--slug", default=None)
    parser.add_argument("legacy_slug", nargs="?", default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--all", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    csv_path = None if args.no_csv else Path(args.csv)
    queue_path = None if args.no_queue else Path(args.queue)
    only_slug = args.slug or args.legacy_slug

    return run(
        docs_dir=Path(args.docs_dir),
        min_score=args.min_score,
        csv_path=csv_path,
        queue_path=queue_path,
        only_slug=only_slug,
    )


if __name__ == "__main__":
    raise SystemExit(main())