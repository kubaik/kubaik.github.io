"""
tweet_entropy.py
================
High-variance entropy engine for visibility_automator.py.

Drop this file next to visibility_automator.py and replace the relevant
sections with the imports / calls documented below.

Four systems in one file:
  1. DynamicMetricPool        – randomised pain-point / metric slots
  2. HookAlternationEngine    – cross-run style deduplication
  3. LLM_NEGATIVE_CONSTRAINTS – copy-paste system-prompt block
  4. VariantTemplate          – multi-variant template expansion
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

# ══════════════════════════════════════════════════════════════════════════════
# 1. DYNAMIC METRIC & FRICTION POOL
# ══════════════════════════════════════════════════════════════════════════════
#
# Usage in visibility_automator.py:
#
#   from tweet_entropy import DynamicMetricPool
#   pool = DynamicMetricPool()
#   metric = pool.draw(seed=post.slug)          # deterministic per post
#   pain   = pool.draw_pain(seed=post.slug)     # friction phrase
#
# Then pass {metric} and {pain} as format vars inside your templates.
# ──────────────────────────────────────────────────────────────────────────────


class DynamicMetricPool:
    """
    Provides deterministic-but-varied pain metrics and friction phrases
    so every tweet gets a different concrete anchor.

    Deterministic = same slug → same pick on every run (idempotent).
    Varied        = different slugs draw from different buckets so the
                    same number never appears twice in a row on the timeline.
    """

    # ── Financial metrics ────────────────────────────────────────────────────
    _FINANCIAL: List[str] = [
        "$3k/mo",
        "$12k+ a year",
        "$800 a month",
        "over $4,000/mo",
        "roughly $6k each quarter",
        "$15k in wasted API credits",
        "under $200/month",
        "half their cloud budget",
        "a four-figure monthly bill",
        "tens of thousands a year",
    ]

    # ── Time / effort sinks ──────────────────────────────────────────────────
    _TIME: List[str] = [
        "40+ developer hours",
        "three weekends straight",
        "two sprints chasing the same bug",
        "a week of oncall incidents",
        "half a quarter debugging this",
        "six hours in a single postmortem",
        "days of painful log triage",
        "an entire sprint on one config issue",
        "three consecutive all-nighters",
        "weeks of back-and-forth in PRs",
    ]

    # ── Structural / complexity friction ────────────────────────────────────
    _STRUCTURAL: List[str] = [
        "hundreds of messy logs",
        "a tangled dependency graph",
        "five separate config files for one service",
        "a 2,000-line migration nobody wanted to touch",
        "a broken pipeline with no owner",
        "duplicate endpoints across four microservices",
        "a 47-column query with no index",
        "stale docs that contradict the code",
        "three overlapping monitoring tools",
        "a codebase where half the tests are skipped",
    ]

    # ── Compound pain phrases (metric + context in one phrase) ───────────────
    _PAIN_PHRASES: List[str] = [
        "paying three vendors to solve the same problem",
        "waiting on builds that take 18 minutes to fail",
        "shipping to staging only to find prod breaks differently",
        "rewriting the same boilerplate in every new service",
        "losing a senior dev because the codebase was unmaintainable",
        "discovering the bottleneck was a single misconfigured timeout",
        "rolling back a release at 2am over a typo in a config value",
        "debugging a race condition that only appeared under load",
        "explaining to a VP why the dashboards were lying for six weeks",
        "watching a cache stampede take down a service on launch day",
        "deploying a hotfix without understanding the root cause",
        "paying for 99.9% uptime SLA then sitting through a 4-hour outage",
    ]

    # ── Buckets by modulo to guarantee rotation across consecutive posts ─────
    _BUCKETS = ["_FINANCIAL", "_TIME", "_STRUCTURAL", "_PAIN_PHRASES"]

    def _seed_idx(self, seed: str, pool: List[str], bucket_offset: int = 0) -> int:
        h = int(hashlib.md5(
            f"{seed}:{bucket_offset}".encode()).hexdigest(), 16)
        return h % len(pool)

    def draw(self, seed: str) -> str:
        """Return a concrete metric phrase (financial, time, or structural)."""
        bucket_name = self._BUCKETS[
            int(hashlib.md5(seed.encode()).hexdigest(), 16) % (
                len(self._BUCKETS) - 1)
        ]
        pool: List[str] = getattr(self, bucket_name)
        return pool[self._seed_idx(seed, pool)]

    def draw_financial(self, seed: str) -> str:
        return self._FINANCIAL[self._seed_idx(seed, self._FINANCIAL)]

    def draw_time(self, seed: str) -> str:
        return self._TIME[self._seed_idx(seed, self._TIME, bucket_offset=1)]

    def draw_structural(self, seed: str) -> str:
        return self._STRUCTURAL[self._seed_idx(seed, self._STRUCTURAL, bucket_offset=2)]

    def draw_pain(self, seed: str) -> str:
        """Return a compound friction phrase."""
        return self._PAIN_PHRASES[self._seed_idx(seed, self._PAIN_PHRASES, bucket_offset=3)]


# ══════════════════════════════════════════════════════════════════════════════
# 2. HOOK ALTERNATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
#
# Persists the last-used hook FAMILY to disk so consecutive runs can never
# repeat the same structural layout.
#
# Usage in visibility_automator.py  (inside _build_single_tweet or the caller):
#
#   from tweet_entropy import HookAlternationEngine
#   engine = HookAlternationEngine()
#   idx = engine.next_template_idx(post.slug, len(_TWEET_TEMPLATES))
#   template = _TWEET_TEMPLATES[idx]
#   engine.record(idx)        # call AFTER the tweet is successfully posted
# ──────────────────────────────────────────────────────────────────────────────

# Map every template index to a broad "family" label.
# Templates that share a family are never used consecutively.
TEMPLATE_FAMILIES = {
    0:  "admission",        # honest admission / teams get it wrong
    1:  "number_lead",      # specific result / number
    2:  "paradox",          # less X = more Y
    3:  "contrarian",       # everyone says… but
    4:  "before_after",     # before / after
    5:  "observation",      # teams that nail X do one thing
    6:  "analyzed",         # dozens reviewed / same 3 mistakes
    7:  "dev_prod_gap",     # works in dev, breaks in prod
    8:  "confession",       # most teams ship broken X
    9:  "opinion",          # unpopular take
    10: "tool_discovery",   # pattern that replaces 200 lines
    11: "seniority",        # senior devs approach X differently
    12: "cost",             # cut overhead by doing less
    13: "docs_gap",         # docs are good; prod isn't covered
    14: "result_timeline",  # latency issue / single config line
    15: "open_question",    # how long did X take your team
}

# Families that must never follow each other (adjacency ban list).
# Format: {family: [banned_next_families]}
_ADJACENCY_BANS: dict = {
    "admission":    ["confession", "observation"],
    "confession":   ["admission", "observation"],
    "observation":  ["admission", "confession"],
    "number_lead":  ["result_timeline"],
    "result_timeline": ["number_lead"],
    "contrarian":   ["opinion"],
    "opinion":      ["contrarian"],
    "before_after": ["dev_prod_gap"],
    "dev_prod_gap": ["before_after"],
}


class HookAlternationEngine:
    """
    Prevents consecutive tweets from using the same structural hook family.

    State is persisted to `.hook_state.json` so the constraint holds across
    separate process runs (i.e., separate GitHub Actions runs).
    """

    _STATE_FILE = Path(".hook_state.json")

    def __init__(self, state_file: Optional[Path] = None):
        self._path = state_file or self._STATE_FILE
        self._state: dict = self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def next_template_idx(self, seed: str, n_templates: int) -> int:
        """
        Return a template index that:
        1. Is not the same family as the last-used template.
        2. Is not in an adjacency-banned family relative to the last.
        3. Falls back to full slug-hash if all families are banned (rare).
        """
        last_family = self._state.get("last_family", "")
        banned_families: set = {last_family}
        banned_families.update(_ADJACENCY_BANS.get(last_family, []))

        # Build a candidate list that avoids banned families
        candidates = [
            idx for idx in range(n_templates)
            if TEMPLATE_FAMILIES.get(idx, "") not in banned_families
        ]

        if not candidates:
            candidates = list(range(n_templates))   # full fallback

        # Deterministic within candidates so the same slug always maps to the
        # same "allowed" template (idempotent regeneration).
        h = int(hashlib.md5(f"{seed}:{last_family}".encode()).hexdigest(), 16)
        return candidates[h % len(candidates)]

    def record(self, template_idx: int) -> None:
        """
        Call this after the tweet is successfully posted (or composed for
        dry-run mode). Persists state for the next run.
        """
        family = TEMPLATE_FAMILIES.get(template_idx, f"unknown_{template_idx}")
        self._state["last_family"] = family
        self._state["last_idx"] = template_idx
        self._state["recorded_at"] = int(time.time())
        self._save()

    def last_family(self) -> str:
        return self._state.get("last_family", "")

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> dict:
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save(self) -> None:
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._state, f, indent=2)
        except Exception as exc:
            print(f"⚠️  HookAlternationEngine: could not save state — {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. LLM NEGATIVE CONSTRAINTS  (system-prompt block for EnhancedTweetGenerator)
# ══════════════════════════════════════════════════════════════════════════════
#
# Usage  — paste the constant into your LLM system prompt:
#
#   from tweet_entropy import LLM_NEGATIVE_CONSTRAINTS
#
#   system_prompt = f"""
#   You are a senior technical copywriter writing X/Twitter hooks. ...
#
#   {LLM_NEGATIVE_CONSTRAINTS}
#   """
# ──────────────────────────────────────────────────────────────────────────────

LLM_NEGATIVE_CONSTRAINTS: str = """
══════════════════════════════════════════════════════
BANNED HOOK PATTERNS — NEVER USE ANY OF THESE
══════════════════════════════════════════════════════

BANNED OPENERS (first 5 words of tweet body):
  ✗ "Most teams burn..."
  ✗ "Most developers don't..."
  ✗ "Most engineers still..."
  ✗ "Here is why..."
  ✗ "Here's why..."
  ✗ "Let me show you..."
  ✗ "Let's dive into..."
  ✗ "Let's be honest..."
  ✗ "The truth about..."
  ✗ "Nobody talks about..."
  ✗ "No one tells you..."
  ✗ "This is how..."
  ✗ "This changes everything..."
  ✗ "Stop using..."
  ✗ "You should know..."
  ✗ "Did you know..."
  ✗ "Fun fact:"
  ✗ "Hot take:"   (overused — rephrase as: "Unpopular view:" or just state the opinion)

BANNED PHRASES (anywhere in the tweet):
  ✗ "game-changer" / "game changer"
  ✗ "it's important to note"
  ✗ "dive into" / "delve into"
  ✗ "leverage" (use "use" instead)
  ✗ "unlock the potential"
  ✗ "harness the power"
  ✗ "seamlessly"
  ✗ "revolutionize" / "revolutionise"
  ✗ "cutting-edge"
  ✗ "state-of-the-art"
  ✗ "paradigm shift"
  ✗ "full breakdown" (prefer: "full walkthrough" / "what I found" / "the details")
  ✗ "thread below ↓"  (prefer: "full guide 👇" or "here's the pattern 👇")
  ✗ "in today's X landscape"
  ✗ "needless to say"
  ✗ "at the end of the day"
  ✗ "it goes without saying"

BANNED STRUCTURAL PATTERNS:
  ✗ [Shocking claim]\\n\\n[Restate same claim slightly differently]\\n\\nFull breakdown 👇
     → This 3-line pattern is over-indexed and kills trust. Vary the structure.
  ✗ Consecutive tweets using the same subject ("Most teams...", "Most engineers...")
  ✗ Any tweet that is just a rephrased version of a previous hook in this session.

STYLE REQUIREMENTS:
  ✓ Use third person OR a direct "you" voice — never first person ("I burned...")
    unless the hook is explicitly a case study ("After 6 months in production: ...")
  ✓ If you use a number, it must be concrete and plausible: "$3k/mo" not "$millions"
  ✓ Vary sentence length: mix short punchy lines with one longer explanatory line
  ✓ End with ONE clear action cue — do not stack two CTAs
  ✓ The hook must be self-contained: a reader with no context should understand
    what the problem is without clicking through
══════════════════════════════════════════════════════
"""


# ══════════════════════════════════════════════════════════════════════════════
# 4. VARIANT TEMPLATE SYSTEM  (replaces static _TWEET_TEMPLATES strings)
# ══════════════════════════════════════════════════════════════════════════════
#
# Each VariantTemplate holds multiple opener variants and optional body
# variants.  When rendered, it randomly (but deterministically, seeded by
# the post slug) selects from each variant list.
#
# Usage in visibility_automator.py:
#
#   from tweet_entropy import VARIANT_TEMPLATES, render_variant_template
#
#   # Replace _TWEET_TEMPLATES[idx] with:
#   raw = render_variant_template(VARIANT_TEMPLATES[idx], post.slug, **format_vars)
# ──────────────────────────────────────────────────────────────────────────────

class VariantTemplate:
    """
    A tweet template that holds multiple alternative phrasings for each
    structural slot so the output varies even when the same template index
    is selected.

    Slots in the body strings use standard Python .format() syntax.
    Available format vars (same as existing templates):
      {topic}   — extracted topic phrase
      {teaser}  — meta description teaser
      {url}     — post URL (with optional read-time annotation)
      {tags}    — hashtag string
      {bait}    — reply-bait question
      {metric}  — dynamic metric from DynamicMetricPool.draw()
      {pain}    — compound friction phrase from DynamicMetricPool.draw_pain()
    """

    def __init__(
        self,
        family: str,
        openers: List[str],
        bodies: List[str],
        closers: Optional[List[str]] = None,
    ):
        """
        Parameters
        ----------
        family  : str
            Hook family label (must match TEMPLATE_FAMILIES).
        openers : List[str]
            Alternative first lines / opening stanzas.  One is picked per render.
        bodies  : List[str]
            Alternative middle bodies.  One is picked per render.
        closers : List[str] | None
            Alternative CTA / closing lines.  If None, a shared default pool is used.
        """
        self.family = family
        self.openers = openers
        self.bodies = bodies
        self.closers = closers or [
            "Full walkthrough 👇\n{url}\n\n{tags}{bait}",
            "Here's the pattern 👇\n{url}\n\n{tags}{bait}",
            "What I found 👇\n{url}\n\n{tags}{bait}",
            "The details 👇\n{url}\n\n{tags}{bait}",
        ]

    def render(self, seed: str, **kwargs) -> str:
        def _pick(lst: List[str], offset: int) -> str:
            h = int(hashlib.md5(
                f"{seed}:{self.family}:{offset}".encode()).hexdigest(), 16)
            return lst[h % len(lst)]

        opener = _pick(self.openers, 0)
        body = _pick(self.bodies,   1)
        closer = _pick(self.closers,  2)

        raw = "\n\n".join(part for part in [
                          opener, body, closer] if part.strip())
        try:
            return raw.format(**kwargs)
        except KeyError:
            # Gracefully fall back if a slot is missing from kwargs
            return raw


# ── The 16 variant templates (indexed to match TEMPLATE_FAMILIES) ─────────────

VARIANT_TEMPLATES: List[VariantTemplate] = [

    # 0 — ADMISSION
    VariantTemplate(
        family="admission",
        openers=[
            "{topic} has a failure mode that most teams hit at month three.",
            "The standard advice on {topic} breaks under real traffic.",
            "Teams that get {topic} right the first time are rarer than the tutorials suggest.",
            "Six months into production, {topic} starts behaving differently than the docs imply.",
        ],
        bodies=[
            "{teaser}",
            "The gap between the tutorial and the production postmortem is wide.\n\n{teaser}",
            "Here's what the docs don't cover:\n\n{teaser}",
        ],
    ),

    # 1 — NUMBER LEAD
    VariantTemplate(
        family="number_lead",
        openers=[
            "{metric} — that's what {topic} was costing before one config change fixed it.",
            "Cutting {metric} came down to a single decision about {topic}.",
            "The {topic} optimisation that recovered {metric} in two weeks.",
            "{metric} saved by rethinking one assumption about {topic}.",
        ],
        bodies=[
            "{teaser}",
            "The measurement came first. Then the fix.\n\n{teaser}",
            "Not a refactor. Not a rewrite. One lever.\n\n{teaser}",
        ],
    ),

    # 2 — PARADOX
    VariantTemplate(
        family="paradox",
        openers=[
            "Less {topic} code tends to mean fewer incidents.",
            "The teams with the cleanest {topic} setups also have the shortest runbooks.",
            "Reducing {topic} complexity by 60% required adding exactly zero new tools.",
            "The counterintuitive part of {topic}: doing less is often the performance fix.",
        ],
        bodies=[
            "{teaser}",
            "The reasoning behind it:\n\n{teaser}",
            "Why this works in practice:\n\n{teaser}",
        ],
    ),

    # 3 — CONTRARIAN
    VariantTemplate(
        family="contrarian",
        openers=[
            "The conventional framing of {topic} makes the hard part invisible.",
            "Everyone reaches for the same {topic} solution — and it's not the right starting point.",
            "The {topic} tutorial consensus is accurate. It's also incomplete in a specific way.",
            "Saying '{topic} is just X' is technically true and practically misleading.",
        ],
        bodies=[
            "The part that's missing:\n\n{teaser}",
            "{teaser}",
            "What the typical guide skips:\n\n{teaser}",
        ],
    ),

    # 4 — BEFORE / AFTER
    VariantTemplate(
        family="before_after",
        openers=[
            "Before: {pain}.\nAfter: a five-step checklist that catches 90% of it.",
            "Before the fix: {metric} gone every month on {topic}.\nAfter: one config line.",
            "The {topic} workflow before and after one structural change.",
        ],
        bodies=[
            "{teaser}",
            "What changed (and what it took to find it):\n\n{teaser}",
            "The diff that made the difference:\n\n{teaser}",
        ],
    ),

    # 5 — OBSERVATION
    VariantTemplate(
        family="observation",
        openers=[
            "Teams that ship {topic} cleanly have one thing in common.",
            "After reviewing a lot of {topic} implementations — the good ones share a pattern.",
            "The {topic} setups that survive scaling look different from the ones that don't.",
            "There's a consistent tell in teams that have {topic} under control.",
        ],
        bodies=[
            "{teaser}",
            "The pattern:\n\n{teaser}",
            "What separates them:\n\n{teaser}",
        ],
    ),

    # 6 — ANALYZED
    VariantTemplate(
        family="analyzed",
        openers=[
            "Reviewed dozens of {topic} setups in production codebases.",
            "Same three {topic} mistakes keep appearing across unrelated teams.",
            "After going through a lot of {topic} code — the failure modes cluster.",
            "The {topic} problem isn't unique to one team. The pattern repeats.",
        ],
        bodies=[
            "{teaser}",
            "The three that show up most:\n\n{teaser}",
            "What the repeat offenders have in common:\n\n{teaser}",
        ],
    ),

    # 7 — DEV / PROD GAP
    VariantTemplate(
        family="dev_prod_gap",
        openers=[
            "{topic} holds up fine in a local environment.\nThe first real traffic event is a different story.",
            "The {topic} setup that works on a dev laptop has a specific failure mode under load.",
            "Every {topic} tutorial works. Then production adds latency, concurrent users, and real data.",
            "The gap between '{topic} works in staging' and '{topic} works in prod' is non-trivial.",
        ],
        bodies=[
            "{teaser}",
            "Why this happens — and how to close the gap:\n\n{teaser}",
            "The specific conditions that expose it:\n\n{teaser}",
        ],
    ),

    # 8 — CONFESSION
    VariantTemplate(
        family="confession",
        openers=[
            "A team ships broken {topic} to production and doesn't know it for weeks.",
            "The {topic} bug that's live right now in more codebases than anyone will admit.",
            "The gap between 'it works' and 'it works correctly' is invisible in {topic} until it isn't.",
            "Most {topic} problems in production were technically working — just not correctly.",
        ],
        bodies=[
            "{teaser}",
            "The lesson that finally sticks:\n\n{teaser}",
            "What the postmortem always shows:\n\n{teaser}",
        ],
    ),

    # 9 — OPINION
    VariantTemplate(
        family="opinion",
        openers=[
            "Unpopular view: the standard {topic} mental model teaches the wrong thing first.",
            "Contested: most {topic} guides optimise for the tutorial case, not the production case.",
            "Disagree with the consensus on {topic} here — and have the incident data to back it up.",
            "The {topic} advice that's technically correct but consistently leads teams into a wall.",
        ],
        bodies=[
            "The mental model that actually holds under load:\n\n{teaser}",
            "{teaser}",
            "What changes when you reframe it:\n\n{teaser}",
        ],
    ),

    # 10 — TOOL DISCOVERY
    VariantTemplate(
        family="tool_discovery",
        openers=[
            "A {topic} pattern that replaces {metric} of boilerplate with a single abstraction.",
            "The {topic} approach that cut the implementation from 200 lines to 20.",
            "Replaced three {topic} tools with one pattern — and the oncall schedule got quieter.",
            "The {topic} setup that eliminated {pain} entirely.",
        ],
        bodies=[
            "Setup and full walkthrough:\n\n{teaser}",
            "{teaser}",
            "How it works in practice:\n\n{teaser}",
        ],
    ),

    # 11 — SENIORITY
    VariantTemplate(
        family="seniority",
        openers=[
            "Senior engineers approach {topic} with a different question than junior ones.",
            "The {topic} mental shift that separates the engineers who debug fast from the ones who don't.",
            "There's a specific moment when {topic} stops being a puzzle and starts being a system.",
            "The {topic} instinct that takes years to develop — and a shortcut to build it faster.",
        ],
        bodies=[
            "{teaser}",
            "The shift:\n\n{teaser}",
            "What changes after it clicks:\n\n{teaser}",
        ],
    ),

    # 12 — COST
    # Uses {fin} (financial metric) and {time} (time/effort metric) — both
    # provided by build_single_tweet_v2 via DynamicMetricPool.draw_financial/draw_time.
    VariantTemplate(
        family="cost",
        openers=[
            "Cut {topic} overhead by removing what looked necessary but wasn't.",
            "The {topic} bill came down to {fin} after one architectural decision.",
            "Less {topic} instrumentation, not more, was the fix that cut {fin} in spend.",
            "The {topic} audit found {fin} going to infrastructure that was actively making things worse.",
        ],
        bodies=[
            "{teaser}",
            "The counter-intuitive part:\n\n{teaser}",
            "Where the spend was actually going:\n\n{teaser}",
        ],
    ),

    # 13 — DOCS GAP
    VariantTemplate(
        family="docs_gap",
        openers=[
            "The {topic} documentation is accurate. It covers the wrong 80% of the problem.",
            "Official {topic} docs: solid.\nWhat they don't cover: the part that breaks in production.",
            "Every {topic} guide covers the setup. Almost none cover what happens at month four.",
            "The {topic} docs are thorough right up until the first edge case that actually matters.",
        ],
        bodies=[
            "{teaser}",
            "What's missing:\n\n{teaser}",
            "The gap:\n\n{teaser}",
        ],
    ),

    # 14 — RESULT / TIMELINE
    # Uses {fin} for financial cost and {time} for time-based metrics.
    VariantTemplate(
        family="result_timeline",
        openers=[
            "A {topic} issue was adding 400ms latency. Root cause: one config value.",
            "The {topic} incident that cost {fin} traced back to a single default setting.",
            "{time} chasing a {topic} regression. Root cause: one assumption nobody had questioned.",
            "Three weeks on a {topic} problem. Fix took {time} once we knew where to look.",
        ],
        bodies=[
            "{teaser}",
            "What the investigation found:\n\n{teaser}",
            "The fix and how to prevent it:\n\n{teaser}",
        ],
    ),

    # 15 — OPEN QUESTION
    VariantTemplate(
        family="open_question",
        openers=[
            "How long did it take your team to actually get {topic} right?",
            "What was the {topic} assumption that cost the most time to unlearn?",
            "At what point did {topic} stop feeling like a configuration problem and start feeling like a design problem?",
            "What finally made {topic} click after the tutorial version stopped working?",
        ],
        bodies=[
            "{teaser}",
            "What finally made it click for this implementation:\n\n{teaser}",
            "The pattern that resolved it:\n\n{teaser}",
        ],
    ),
]

# Sanity check: indices must align with TEMPLATE_FAMILIES
assert len(VARIANT_TEMPLATES) == len(TEMPLATE_FAMILIES), (
    f"VARIANT_TEMPLATES has {len(VARIANT_TEMPLATES)} entries but "
    f"TEMPLATE_FAMILIES has {len(TEMPLATE_FAMILIES)} keys."
)


def render_variant_template(
    template: VariantTemplate,
    seed: str,
    **format_vars,
) -> str:
    """
    Convenience wrapper.  Equivalent to template.render(seed, **format_vars).

    Example
    -------
    from tweet_entropy import VARIANT_TEMPLATES, render_variant_template, DynamicMetricPool

    pool   = DynamicMetricPool()
    engine = HookAlternationEngine()
    idx    = engine.next_template_idx(post.slug, len(VARIANT_TEMPLATES))

    tweet  = render_variant_template(
        VARIANT_TEMPLATES[idx],
        seed   = post.slug,
        topic  = extracted_topic,
        teaser = extracted_teaser,
        url    = post_url,
        tags   = hashtag_str,
        bait   = reply_bait,
        metric = pool.draw(post.slug),
        pain   = pool.draw_pain(post.slug),
    )

    engine.record(idx)   # persist state AFTER successful post
    """
    return template.render(seed, **format_vars)


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRATION PATCH  (drop-in replacement for _build_single_tweet)
# ══════════════════════════════════════════════════════════════════════════════
#
# Copy this function into visibility_automator.py and replace the existing
# _build_single_tweet() with it.  It uses all four systems above.
# ──────────────────────────────────────────────────────────────────────────────

def build_single_tweet_v2(post, base_url: str, hook_style: str = "auto") -> str:
    """
    Entropy-aware replacement for visibility_automator._build_single_tweet().

    Changes vs original:
    - Uses HookAlternationEngine to prevent consecutive same-family hooks.
    - Uses VariantTemplate so even the same template index varies its wording.
    - Uses DynamicMetricPool to inject concrete, varied metrics into hooks.
    - Preserves all original trimming logic.
    """
    # Import here so this file can be used standalone for testing
    from visibility_automator import (
        _extract_topic_phrase,
        _extract_teaser,
        _get_hashtags_for_post,
        _pick_reply_bait,
        _trim_to_budget,
    )

    metric_pool = DynamicMetricPool()
    engine = HookAlternationEngine()

    if hook_style == "auto":
        idx = engine.next_template_idx(post.slug, len(VARIANT_TEMPLATES))
    else:
        from visibility_automator import _STYLE_MAP
        idx = _STYLE_MAP.get(hook_style, 0)

    template = VARIANT_TEMPLATES[idx]
    topic = _extract_topic_phrase(post.title, max_words=3)

    raw_desc = getattr(post, "meta_description", "") or ""
    teaser = _extract_teaser(raw_desc, max_chars=120)

    post_url = f"{base_url}/{post.slug}"
    reading_time = getattr(post, "reading_time_minutes", None)
    url_line = f"{post_url}  ({reading_time} min read)" if reading_time else post_url

    hashtags = _get_hashtags_for_post(post)
    bait = _pick_reply_bait(post.slug)
    metric = metric_pool.draw(post.slug)
    pain = metric_pool.draw_pain(post.slug)
    fin = metric_pool.draw_financial(post.slug)
    time_val = metric_pool.draw_time(post.slug)

    tweet = render_variant_template(
        template,
        seed=post.slug,
        topic=topic,
        teaser=teaser,
        url=url_line,
        tags=hashtags,
        bait=bait,
        metric=metric,
        pain=pain,
        fin=fin,
        time=time_val,
    )

    # ── Trim to 280 (same cascade as original) ───────────────────────────────
    if len(tweet) > 280:
        # Measure skeleton cost without teaser
        skeleton = render_variant_template(
            template,
            seed=post.slug,
            topic=topic,
            teaser="",
            url=url_line,
            tags=hashtags,
            bait=bait,
            metric=metric,
            pain=pain,
        )
        teaser_budget = max(20, 280 - len(skeleton))
        teaser = _trim_to_budget(teaser, teaser_budget)
        tweet = render_variant_template(
            template,
            seed=post.slug,
            topic=topic,
            teaser=teaser,
            url=url_line,
            tags=hashtags,
            bait=bait,
            metric=metric,
            pain=pain,
        )

    if len(tweet) > 280:
        skeleton = render_variant_template(
            template,
            seed=post.slug,
            topic=topic,
            teaser="",
            url=post_url,
            tags=hashtags,
            bait="",
            metric=metric,
            pain=pain,
        )
        teaser_budget = max(20, 280 - len(skeleton))
        teaser = _trim_to_budget(teaser, teaser_budget)
        tweet = render_variant_template(
            template,
            seed=post.slug,
            topic=topic,
            teaser=teaser,
            url=post_url,
            tags=hashtags,
            bait="",
            metric=metric,
            pain=pain,
        )

    if len(tweet) > 280:
        tweet = _trim_to_budget(tweet, 277)

    # Persist alternation state so the next run picks a different family
    engine.record(idx)

    return tweet


# ══════════════════════════════════════════════════════════════════════════════
# QUICK SELF-TEST  (python tweet_entropy.py)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("TWEET ENTROPY ENGINE — SELF-TEST")
    print("=" * 70)

    pool = DynamicMetricPool()
    engine = HookAlternationEngine(state_file=Path(".hook_state_test.json"))

    slugs = [
        "redis-caching-patterns",
        "postgres-index-optimization",
        "kubernetes-cost-cutting",
        "ai-agents-in-production",
        "typescript-strict-mode-traps",
        "fastapi-rate-limiting",
        "llm-fine-tuning-guide",
        "docker-compose-production",
    ]

    families_used = []
    print("\n── Simulating 8 consecutive posts ──\n")
    for slug in slugs:
        idx = engine.next_template_idx(slug, len(VARIANT_TEMPLATES))
        family = TEMPLATE_FAMILIES[idx]
        metric = pool.draw(slug)
        pain = pool.draw_pain(slug)

        print(f"Slug  : {slug}")
        print(f"Idx   : {idx}  Family : {family}")
        print(f"Metric: {metric}")
        print(f"Pain  : {pain}")

        # Render a sample
        sample = VARIANT_TEMPLATES[idx].render(
            slug,
            topic="Database Indexing",
            teaser="The index that PostgreSQL won't use without a specific cast.",
            url=f"https://kubaik.github.io/{slug}",
            tags="#Postgres #BackendDev",
            bait="\n\nWhat was your worst index miss in prod?",
            metric=metric,
            pain=pain,
        )
        print("Tweet preview:")
        for line in sample.splitlines():
            print(f"  │ {line}")
        print(f"  ({len(sample)} chars)\n")

        engine.record(idx)
        families_used.append(family)

    print("── Family sequence (no two consecutive should match) ──")
    for i, fam in enumerate(families_used):
        marker = ""
        if i > 0 and families_used[i] == families_used[i - 1]:
            marker = " ⚠️  REPEAT!"
        print(f"  {i+1}. {fam}{marker}")

    print("\n── LLM Negative Constraints block (excerpt) ──")
    print(LLM_NEGATIVE_CONSTRAINTS[:400] + "...\n")

    # Clean up test state file
    Path(".hook_state_test.json").unlink(missing_ok=True)
    print("Self-test complete.")
