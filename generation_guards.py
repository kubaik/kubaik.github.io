"""
generation_guards.py - generation-time guards against the two defects that produced the
DELETE / IMPROVE backlog (see triage_final/):

  1. Repeated section skeleton   -> section ROLES instead of concrete headings in the prompt,
                                    plus skeleton_clone_problem() reject gate before save.
  2. No primary sources          -> ensure_sources() adds verified official/standards links at
                                    save time (no LLM, no invented URLs); posts with nothing
                                    safe to cite are flagged needs_source_review, not blocked.

Pure functions; no network except link verification inside ensure_sources (optional).
"""

import json
import re
from pathlib import Path
from typing import Optional, Tuple

GENERIC_H2 = {
    "related reading",
    "official documentation",
    "faq",
    "frequently asked questions",
    "summary",
    "quick reference",
    "common questions",
    "conclusion",
    "sources",
    "references",
    "further reading",
}
MIN_OUTLINE = 4  # shorter outlines are too small to compare
MAX_SAME_OUTLINE = 2  # reject if >= 3 published posts already share the outline
MAX_SAME_TAIL = 5  # reject if >= 6 published posts end with the same 2 sections

# format -> section ROLES. The model must write its own, topic-specific heading for each.
_ROLES = {
    "deep_dive": [
        "the gap between documented behaviour and production reality",
        "the mechanism",
        "a concrete implementation",
        "how to measure it",
        "the failure modes",
        "when this is the wrong choice",
        "the next concrete step",
    ],
    "tutorial": [
        "the problem being solved",
        "prerequisites",
        "the core implementation steps",
        "edge cases and errors",
        "verification",
        "the next concrete step",
    ],
    "opinion": [
        "the stance, stated plainly",
        "why the common advice falls short",
        "the better mental model",
        "where the common advice is still right",
        "a decision rule",
        "the next concrete step",
    ],
    "comparison": [
        "why the comparison matters",
        "each option on its own terms",
        "head-to-head on the axes that matter for this topic",
        "who should pick which",
        "the next concrete step",
    ],
    "case_study": [
        "the situation, framed as a typical scenario",
        "the mechanism at fault",
        "the fix",
        "what to monitor afterwards",
        "the next concrete step",
    ],
    "explainer": [
        "the confusion being resolved",
        "the underlying model",
        "worked examples",
        "common misreadings",
        "the next concrete step",
    ],
    "listicle": [
        "why the list exists",
        "the items, each with when it applies and when it fails",
        "how to prioritise them",
        "the next concrete step",
    ],
    "troubleshooting": [
        "the exact symptom",
        "the likely causes in order of probability",
        "how to confirm each cause",
        "the fix for each",
        "prevention",
        "the next concrete step",
    ],
}


def section_role_block(format_name: str, topic: str) -> str:
    roles = _ROLES.get(format_name) or _ROLES["deep_dive"]
    lines = "\n".join(f"- {r}" for r in roles)
    return (
        f'Cover these section ROLES for "{topic}", in an order that fits the topic. For EACH '
        "role write your own heading that names the specific thing being discussed (a tool, "
        "error, mechanism or decision). NEVER use a generic heading such as 'Decision checklist', "
        "'What to do next', 'Summary', 'Common questions' or 'Do this in the next N minutes'. "
        "Merge or drop roles that do not fit:\n" + lines
    )


def _norm(h: str) -> str:
    h = re.sub(r"\d+", "N", h.lower().strip())
    return re.sub(r"[^a-z N]", "", h).strip()


def outline(content: str) -> tuple:
    return tuple(
        x
        for x in (_norm(h) for h in re.findall(r"^## (.+)$", content, re.M))
        if x and x not in GENERIC_H2
    )


def skeleton_clone_problem(
    content: str, docs_dir: Path, exclude_slug: Optional[str] = None
) -> Optional[str]:
    mine = outline(content)
    if len(mine) < MIN_OUTLINE:
        return None
    same_full = same_tail = 0
    for pj in Path(docs_dir).glob("*/post.json"):
        if exclude_slug and pj.parent.name == exclude_slug:
            continue
        try:
            other = outline(json.loads(pj.read_text("utf-8")).get("content", ""))
        except (OSError, ValueError):
            continue
        if len(other) < MIN_OUTLINE:
            continue
        same_full += other == mine
        same_tail += other[-2:] == mine[-2:]
    if same_full > MAX_SAME_OUTLINE:
        return f"template-clone outline: {same_full} published posts share this exact section skeleton"
    if same_tail > MAX_SAME_TAIL:
        return f"template-clone ending: {same_tail} published posts end with the same two sections"
    return None


def ensure_sources(content: str, verify: bool = True) -> Tuple[str, int]:
    """Append '## Official documentation' with verified links when the post cites nothing
    external. Returns (content, n_links_added). Idempotent."""
    import add_sources as src
    import triage

    if src.has_external(content) or src.HEAD.search(content):
        return content, 0
    links = src.pick(triage.STRIP.sub(" ", content), verify, {})
    if not links:
        return content, 0
    section = "## Official documentation\n\n" + "\n".join(links)
    m = src.FOOTER_SPLIT.search(content)
    new = (
        (content[: m.start()].rstrip() + "\n\n" + section + "\n" + content[m.start() :])
        if m
        else content.rstrip() + "\n\n" + section + "\n"
    )
    return new, len(links)
