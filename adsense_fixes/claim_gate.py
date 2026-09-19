"""
adsense_fixes/claim_gate.py
===========================
Fail-closed pre-publish gate for fabricated citations and unverifiable
first-person incidents.

WHY THIS EXISTS
---------------
AdSense / Helpful Content reviewers treat invented "2026 study by OWASP
found 68%" sentences and "I spent three days debugging…" war stories as
misleading content. This site already strips some of those after the fact
(regeneration_queue.json) — this module stops new drafts from landing
with the same patterns.

HOW TO INTEGRATE
----------------
Call AFTER inject_personal_intro() / inject_eeat_signals() and BEFORE
save_post(), and again inside save_post() so no other entry point can
bypass the auto-mode loop.

    from adsense_fixes.claim_gate import check_claims, ClaimGateError

    result = check_claims(post.content, post.title)
    if result.blocked:
        raise ClaimGateError("; ".join(result.reasons))
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List


class ClaimGateError(Exception):
    """Raised when a draft contains fabricated-authority or incident claims."""


_CODE_FENCE_RE = re.compile(r"```[\s\S]*?```")
_INLINE_CODE_RE = re.compile(r"`[^`]+`")

# "A 2026 study by X", "According to a 2025 survey from Y"
_FAKE_AUTHORITY_RE = re.compile(
    r"\b(?:"
    r"(?:a|an)\s+20\d{2}\s+(?:study|survey|report|analysis|benchmark|paper)\s+by|"
    r"(?:according\s+to|per)\s+(?:a\s+)?(?:20\d{2}\s+)?"
    r"(?:study|survey|report|analysis|benchmark)\s+(?:by|from)|"
    r"(?:study|survey|report|analysis)\s+by\s+[A-Z][A-Za-z0-9&.\- ]{2,40}"
    r"\s+(?:found|showed|tracked|reported|estimated)"
    r")\b",
    re.IGNORECASE,
)

# Named org + precise statistic in the same short window, no URL required
# to trip — the point is the model invents both the org and the number.
_NAMED_ORG_STAT_RE = re.compile(
    r"\b("
    r"OWASP|McKinsey|Gartner|Forrester|Datadog|Snyk|CNCF|DORA|"
    r"Stanford|MIT|Linux Foundation|Upwork|ProfitWell|LinearB|"
    r"Trail of Bits|Cloudflare|Stripe|HashiCorp|"
    r"Kenya Software Engineering Association|African Tech Policy Forum|"
    r"Nairobi Tech Research Group|DevIQ Labs|Dev Interrupted"
    r")\b"
    r".{0,120}?"
    r"\b(\d{1,3}\s*%|\d+(?:\.\d+)?\s*x)\b",
    re.IGNORECASE | re.DOTALL,
)

# Concrete unverifiable experience — opinion verbs (think/believe) are
# intentionally excluded. Matches blog_system._SKIP_PATTERNS intent.
_EXPERIENCE_RE = re.compile(
    r"^(?:"
    r"A colleague\b|"
    r"This took me\b|"
    r"I(?:'ve|'m|\s+have)?\s+(?:spent|built|shipped|deployed|migrated|"
    r"worked\s+(?:on|with|at)|ran\s+into|debugged|broke|fixed|caught|"
    r"hit\s+a|dealt\s+with|went\s+through|had\s+to|ended\s+up|"
    r"kept\s+seeing|was\s+surprised|watched|witnessed|saw\s+firsthand|"
    r"tried\s+\w+\s+first)\b"
    r")",
    re.IGNORECASE | re.MULTILINE,
)

# Headline-only sensational percentages with no source.
_HEADLINE_PCT_RE = re.compile(
    r"\b(?:cut|reduce(?:d)?|improv(?:e|ed)|sav(?:e|ed)|increas(?:e|ed)|"
    r"lift(?:ed)?|rose|rise)\b.{0,40}\b\d{1,3}\s*%\b",
    re.IGNORECASE,
)

# Footer disclaimer is not a pass for inventing named studies.
_DISCLAIMER_RE = re.compile(
    r"figures are illustrative|illustrative; verify",
    re.IGNORECASE,
)


@dataclass
class ClaimGateResult:
    blocked: bool = False
    reasons: List[str] = field(default_factory=list)
    hits: List[str] = field(default_factory=list)


def _plain_text(content: str) -> str:
    text = _CODE_FENCE_RE.sub(" ", content or "")
    text = _INLINE_CODE_RE.sub(" ", text)
    return text


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_MD_PREFIX = re.compile(r"^[\s>*_\-#]+")
_FILLER_LINE_RE = re.compile(
    r"\b(?:game-changer|paradigm shift|harness the power of|"
    r"unlock the potential of|in today's rapidly evolving|"
    r"in the ever-changing landscape|in today's fast-paced|"
    r"as an ai language model)\b",
    re.IGNORECASE,
)


def _line_is_flagged(text: str) -> bool:
    probe = _MD_PREFIX.sub("", (text or "")).strip()
    if not probe:
        return False
    if _EXPERIENCE_RE.match(probe):
        return True
    if _FAKE_AUTHORITY_RE.search(probe):
        return True
    if _FILLER_LINE_RE.search(probe):
        return True
    return False


def strip_flagged_language(content: str) -> tuple[str, int]:
    """Remove fabricated-authority / first-person / filler sentences.

    Uses the same regexes as check_claims(), including markdown prefixes
    (*I spent…*, > I ran into…) so improve actually shrinks the IMPROVE list.
    """
    fences: List[str] = []

    def _mask(match: re.Match) -> str:
        fences.append(match.group(0))
        return f"\x00CODE{len(fences) - 1}\x00"

    masked = _CODE_FENCE_RE.sub(_mask, content or "")
    removed = 0
    kept_lines: List[str] = []

    for line in masked.splitlines(keepends=True):
        raw = line.rstrip("\n")
        ending = "\n" if line.endswith("\n") else ""
        if "\x00CODE" in raw:
            kept_lines.append(line)
            continue
        pieces = _SENTENCE_SPLIT.split(raw) if raw.strip() else [raw]
        kept_pieces: List[str] = []
        for piece in pieces:
            if _line_is_flagged(piece):
                removed += 1
                continue
            kept_pieces.append(piece)
        rebuilt = " ".join(p.strip() for p in kept_pieces if p.strip())
        if rebuilt:
            kept_lines.append(rebuilt + ending)
        elif not raw.strip():
            kept_lines.append(ending)

    new_content = "".join(kept_lines)
    for i, fence in enumerate(fences):
        new_content = new_content.replace(f"\x00CODE{i}\x00", fence)
    return new_content, removed


def check_claims(content: str, title: str = "") -> ClaimGateResult:
    """Return a structured result. Caller decides raise vs retry."""
    result = ClaimGateResult()
    body = _plain_text(content)
    blob = f"{title}\n{body}"

    if _FAKE_AUTHORITY_RE.search(blob):
        result.reasons.append(
            "Unsourced 'study/survey/report by' phrasing — treat as a "
            "fabricated citation."
        )
        result.hits.append("fake_authority")

    org_stat = _NAMED_ORG_STAT_RE.search(blob)
    if org_stat:
        result.reasons.append(
            f"Named organization '{org_stat.group(1)}' paired with a precise "
            "statistic and no verifiable source URL."
        )
        result.hits.append("named_org_stat")

    experience_hits = _EXPERIENCE_RE.findall(body)
    if experience_hits:
        result.reasons.append(
            f"Unverifiable first-person incident language "
            f"({len(experience_hits)} sentence(s))."
        )
        result.hits.append("first_person_incident")

    if title and _HEADLINE_PCT_RE.search(title):
        result.reasons.append(
            "Headline contains an unsourced performance percentage."
        )
        result.hits.append("headline_pct")

    result.blocked = bool(result.reasons)
    return result


def assert_claims_clean(content: str, title: str = "") -> None:
    """Fail-closed helper for save_post() and the auto-mode loop."""
    result = check_claims(content, title)
    if result.blocked:
        raise ClaimGateError("; ".join(result.reasons))


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sample_title = sys.argv[1] if len(sys.argv) > 1 else ""
    sample_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    sample_body = sample_path.read_text(encoding="utf-8") if sample_path else ""
    out = check_claims(sample_body, sample_title)
    print("BLOCKED" if out.blocked else "OK")
    for reason in out.reasons:
        print(f"  - {reason}")
