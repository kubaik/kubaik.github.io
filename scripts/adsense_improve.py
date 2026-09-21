#!/usr/bin/env python3
"""
scripts/adsense_improve.py
==========================
Surgically rewrite the posts in .adsense_triage/improve.json to remove
unverifiable first-person incidents and add missing structure.

WHY THIS EXISTS
---------------
The AdSense triage step (scripts/adsense_triage.py) classifies every
published post into delete / improve / keep. The improve bucket contains
posts that are structurally sound (correct length, no named-source
fabrication, non-narrative title) but contain one or more fabricated
first-person incidents — sentences like "I spent three days debugging a
connection pool issue that turned out to be a single misconfigured
timeout." Those are unverifiable claims presented as lived experience,
which is exactly what Google's Helpful Content signal flags.

This script rewrites those sentences in-place without touching the rest
of the article. Model calls go through scripts/blog_llm_client.py —
a standalone provider chain that reads API keys directly from the
environment and doesn't require config.yaml or BlogSystem to boot.

SAFETY
------
1. Every post.json is backed up to .adsense_backups/{slug}/post.json
   before being rewritten. A bad rewrite can be reverted individually:

       cp .adsense_backups/{slug}/post.json docs/{slug}/post.json
       python blog_system.py build

2. --dry-run is the default. Nothing is written until --apply.

3. Every rewrite is validated before it's saved. A rewrite that:
     - is truncated or incomplete
     - drops more than 15% of the original word count
     - still contains first-person incidents
     - still contains fabricated named-source citations
     - still contains a narrative title pattern in the body
   is rejected and the original post.json is left untouched.

4. Batches are bounded by --limit so one run can't burn the whole
   provider chain. The default of 10 posts per run keeps each batch
   under ~5 minutes.

ENVIRONMENT
-----------
Set at least one LLM provider key before running. Run:

    python scripts/blog_llm_client.py --check

to verify which providers are currently working. The chain is tried in
order and the first that succeeds is used.

AFTER RUNNING
-------------
Always rebuild the site so the updated content is reflected in HTML:

    python blog_system.py build

Usage:
    python scripts/adsense_improve.py --dry-run
    python scripts/adsense_improve.py --apply --limit 10
    python scripts/adsense_improve.py --apply --limit 0   # all in one go
    python scripts/adsense_improve.py --apply --slug my-post-slug
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Sibling import: blog_llm_client lives in this same scripts/ directory.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# Reuse the same detection primitives blog_system.py's publish-time gate
# uses, so the validation here and the publish-time gate can never drift.
# NOTE (2026-09-21): _reject_if_narrative_title is deliberately NOT
# imported. It's designed for titles and flags past-tense action verbs at
# the start of a string ("Added X"), which are incidents in a title but
# perfectly normal technical prose in a body. Body-level incident
# detection delegates to _flag_fabricated_anecdotes(), which uses the
# two-tier check.
from blog_system import (
    _flag_fabricated_anecdotes,
    _reject_if_fabricated_citation,
    _check_expansion_completeness,
    MIN_ACCEPTABLE_WORDS,
)

# Standalone LLM client — reads provider keys directly from os.environ,
# tries each configured provider in order, returns raw text.
from blog_llm_client import generate_text, configured_providers


_DOCS_DIR = Path("./docs")
_TRIAGE_DIR = Path("./.adsense_triage")
_BACKUP_DIR = Path("./.adsense_backups")


# ─────────────────────────────────────────────────────────────────────
# Prompt sentinel handling
# ─────────────────────────────────────────────────────────────────────

# The prompt wraps the source post between these markers. The LLM
# occasionally echoes the closing marker into its output, which then
# fails the "ends on a finished sentence" check in
# _check_expansion_completeness(). Strip them before validation and
# before saving.
_PROMPT_SENTINEL_RE = re.compile(
    r'^\s*-{2,}\s*(?:END\s+(?:OF\s+)?POST|BEGIN\s+POST)\s*-{2,}\s*$',
    re.IGNORECASE,
)


def _strip_prompt_sentinels(text: str) -> str:
    """
    Remove trailing prompt sentinel lines the LLM may have echoed. Only
    strips from the end — a sentinel appearing mid-body is left alone
    since it's more likely real content (e.g. inside a code block).
    """
    lines = text.rstrip().split("\n")
    while lines and _PROMPT_SENTINEL_RE.match(lines[-1]):
        lines.pop()
    return "\n".join(lines).rstrip()


# ─────────────────────────────────────────────────────────────────────
# Bucket loading
# ─────────────────────────────────────────────────────────────────────

def _load_improve_bucket(triage_dir: Path) -> list[dict]:
    """Load the improve bucket produced by scripts/adsense_triage.py."""
    path = triage_dir / "improve.json"
    if not path.exists():
        print(f"No improve bucket found at {path} — run adsense_triage.py first.")
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"Could not read {path}: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────────────

def _build_prompt(post_data: dict) -> list[dict]:
    """
    Build the system + user messages sent to the LLM. The prompt is
    deliberately narrow: preserve structure and code, only rewrite the
    specific categories of sentence that violate AdSense quality.

    The rules mirror the publish-time gate in blog_system.py so a rewrite
    that would fail that gate isn't even requested here.
    """
    title = post_data.get("title", "")
    content = post_data.get("content", "")

    system = (
        "You are a careful technical editor. You rewrite existing blog "
        "posts to remove fabricated first-person incidents and add missing "
        "structure, WITHOUT changing the topic or inventing new claims.\n"
        "\n"
        "ABSOLUTE RULES — a rewrite that breaks any of these is discarded:\n"
        "1. Do NOT invent specific personal incidents, specific bugs that "
        "happened to you, or specific dollar/time metrics presented as your "
        "own. Rewrite any such sentence as a typical/common industry "
        "pattern. Examples:\n"
        "     BAD:  'I spent three days debugging a connection pool issue.'\n"
        "     GOOD: 'A connection pool issue that consumes three days of "
        "debugging is usually a single misconfigured timeout.'\n"
        "     BAD:  'We spent two weeks tuning the rules and still ended up "
        "with 42% of PRs requiring review.'\n"
        "     GOOD: 'Rule-tuning efforts commonly plateau around 40% manual "
        "review, even after two weeks of iteration.'\n"
        "     BAD:  'It cost $1,800 in engineering time and $600 in infra.'\n"
        "     GOOD: 'The engineering and infra cost for this pattern is "
        "commonly in the low thousands of dollars.'\n"
        "\n"
        "2. Do NOT invent named third-party sources ('a 2026 Gartner survey', "
        "'a Datadog report', 'the 2026 Stack Overflow Developer Survey'). "
        "Remove any such attribution and describe the mechanism instead.\n"
        "\n"
        "3. Preserve all code examples, headings, tables, and technical "
        "content verbatim unless a specific sentence is a fabricated "
        "incident. Do NOT restructure the article. Do NOT reorder sections.\n"
        "\n"
        "4. Preserve the '### About this article' E-E-A-T footer exactly as "
        "it is, including the date.\n"
        "\n"
        "5. Preserve or increase the total word count. A rewrite that is "
        "significantly shorter than the original will be rejected.\n"
        "\n"
        "6. If a '## Frequently Asked Questions' section is not already "
        "present, add one with 3-4 real developer questions a reader would "
        "actually search for, each answered in 3-5 sentences. If one is "
        "already present, leave it alone.\n"
        "\n"
        "7. Return ONLY the complete rewritten markdown body. Do NOT "
        "include the '--- BEGIN POST ---' or '--- END POST ---' "
        "delimiters in your output. Do not repeat the title. Do not add "
        "meta commentary. Do not wrap the whole output in a code fence. "
        "No preamble."
    )

    user = (
        f"Title: {title!r}\n\n"
        "Rewrite the post below to comply with rules 1-7. "
        "Output only the full post markdown.\n\n"
        f"--- BEGIN POST ---\n{content}\n--- END POST ---"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ─────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────

def _body_has_fabricated_incident(body: str) -> str | None:
    """
    Body-specific version of the narrative check.

    _reject_if_narrative_title() is designed for TITLES: it flags
    first-person pronouns AND past-tense action verbs at the start of the
    string. In a body, only the first of those is unambiguously an
    incident claim — "Added retry logic" is a normal way to describe a
    change, whereas "I added retry logic" is a fabricated incident.

    This delegates to _flag_fabricated_anecdotes(), which uses the
    two-tier check from blog_system.py, so there's one source of truth
    for what counts as a body-level incident.
    """
    incidents = _flag_fabricated_anecdotes(body)
    if incidents:
        return (
            f"body still contains {len(incidents)} first-person "
            f"incident(s), first: {incidents[0][:60]!r}"
        )
    return None


def _validate_rewrite(original: str, rewritten: str) -> str | None:
    """
    Return None if the rewrite is acceptable, else a short human-readable
    reason it should be rejected. The rewrite is discarded and the
    original kept whenever this returns a non-None value.
    """
    if not rewritten or not rewritten.strip():
        return "empty rewrite"

    # Strip any echoed prompt sentinel before the truncation check, so a
    # well-formed rewrite doesn't fail because it ended on "--- END POST ---".
    rewritten = _strip_prompt_sentinels(rewritten)

    # Truncation / mid-generation cutoff / mid-code-block ending
    issues = _check_expansion_completeness(rewritten)
    if issues:
        return f"incomplete rewrite: {'; '.join(issues)}"

    # Length floor and drift check
    original_wc = len(original.split())
    new_wc = len(rewritten.split())
    if new_wc < MIN_ACCEPTABLE_WORDS:
        return f"rewrite too short ({new_wc} < {MIN_ACCEPTABLE_WORDS})"
    if new_wc < original_wc * 0.85:
        return (
            f"rewrite dropped too much content "
            f"({original_wc} → {new_wc}, more than 15% loss)"
        )

    # Content-level gates must all clear after the rewrite
    if _reject_if_fabricated_citation(rewritten):
        return "rewrite still contains a fabricated citation"

    incident_problem = _body_has_fabricated_incident(rewritten)
    if incident_problem:
        return incident_problem

    return None


# ─────────────────────────────────────────────────────────────────────
# Single-post processing
# ─────────────────────────────────────────────────────────────────────

async def _improve_one(
    slug: str,
    dry_run: bool,
    verbose: bool,
) -> str:
    """
    Rewrite one post. Returns a status string for the run log. Never
    raises — every failure path returns a FAIL/SKIP/REJECT string so the
    batch loop can continue.
    """
    post_json = _DOCS_DIR / slug / "post.json"
    if not post_json.exists():
        return f"SKIP {slug}: post.json not found"

    try:
        post_data = json.loads(post_json.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return f"SKIP {slug}: post.json unreadable ({e})"

    content = post_data.get("content", "")
    if not content.strip():
        return f"SKIP {slug}: empty content"

    messages = _build_prompt(post_data)

    # Token budget scales with input length so long posts have headroom
    # for a complete response. The 6500 floor matches blog_system.py's
    # generation default; the 16000 ceiling caps provider cost.
    budget = min(16000, max(6500, len(content) // 4 + 3000))

    if verbose:
        print(f"\n  [{slug}] sending {len(content.split())} words, "
              f"budget {budget} tokens")

    try:
        rewritten = await generate_text(
            messages=messages,
            max_tokens=budget,
            verbose=verbose,
        )
    except Exception as e:
        return f"FAIL {slug}: all providers failed ({e})"

    # Strip any sentinel the LLM echoed before validation, so the string
    # we validate is the same string we write.
    rewritten = _strip_prompt_sentinels(rewritten)

    rejection = _validate_rewrite(content, rewritten)
    if rejection:
        return f"REJECT {slug}: {rejection}"

    if dry_run:
        return (
            f"DRY-RUN {slug}: rewrite would be "
            f"{len(rewritten.split())} words "
            f"(was {len(content.split())})"
        )

    # ── Backup the original before writing ────────────────────────────
    post_backup_dir = _BACKUP_DIR / slug
    post_backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(post_json, post_backup_dir / "post.json")

    index_md = post_json.parent / "index.md"
    if index_md.exists():
        shutil.copy2(index_md, post_backup_dir / "index.md")

    # ── Write the rewrite ─────────────────────────────────────────────
    post_data["content"] = rewritten
    post_data["updated_at"] = datetime.now().isoformat()
    post_data["_adsense_improved"] = True

    try:
        post_json.write_text(
            json.dumps(post_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        index_md.write_text(
            f"# {post_data.get('title', '')}\n\n{rewritten}",
            encoding="utf-8",
        )
    except OSError as e:
        # Restore from backup if the write failed partway. Better to
        # leave the original in place than a half-written post.json.
        shutil.copy2(post_backup_dir / "post.json", post_json)
        return f"FAIL {slug}: write failed, restored original ({e})"

    return (
        f"OK {slug}: {len(content.split())} → "
        f"{len(rewritten.split())} words"
    )


# ─────────────────────────────────────────────────────────────────────
# Batch runner
# ─────────────────────────────────────────────────────────────────────

async def _run_batch(args) -> int:
    # ── Load the improve bucket ───────────────────────────────────────
    triage_dir = Path(args.triage_dir)
    bucket = _load_improve_bucket(triage_dir)
    if not bucket:
        return 1

    # ── Optional --slug override for single-post debugging ────────────
    if args.slug:
        bucket = [e for e in bucket if e.get("slug") == args.slug]
        if not bucket:
            print(f"Slug {args.slug!r} not in improve bucket.")
            return 1

    # ── Apply --limit ─────────────────────────────────────────────────
    limit = args.limit if args.limit > 0 else len(bucket)
    todo = bucket[:limit]

    dry_run = not args.apply
    print(
        f"Improving {len(todo)} of {len(bucket)} posts "
        f"(dry_run={dry_run}, limit={limit})"
    )
    print(f"docs dir:   {_DOCS_DIR}")
    print(f"backup dir: {_BACKUP_DIR}")

    # ── Report which providers will be tried ──────────────────────────
    configured = configured_providers()
    if not configured:
        print()
        print("❌ No LLM providers configured. Set at least one of:")
        print("   DEEPSEEK_API_KEY, GROQ_API_KEY, GEMINI_API_KEY,")
        print("   OPENROUTER_API_KEY, MISTRAL_API_KEY, GITHUB_TOKEN,")
        print("   NVIDIA_API_KEY, ZAI_API_KEY")
        print()
        print("Verify with: python scripts/blog_llm_client.py --check")
        return 1
    print(f"providers:  {' → '.join(configured)}")
    print()

    ok = 0
    dry = 0
    rejected = 0
    skipped = 0
    failed = 0

    for entry in todo:
        slug = entry.get("slug", "").strip()
        if not slug:
            continue

        result = await _improve_one(slug, dry_run=dry_run, verbose=not args.quiet)
        print(f"  {result}")

        if result.startswith("OK "):
            ok += 1
        elif result.startswith("DRY-RUN "):
            dry += 1
        elif result.startswith("REJECT "):
            rejected += 1
        elif result.startswith("SKIP "):
            skipped += 1
        else:
            failed += 1

    print()
    if dry_run:
        print(f"Dry run complete: {dry} would be rewritten, "
              f"{rejected} rejected by validator, {skipped} skipped, "
              f"{failed} failed.")
        print("Run with --apply to actually rewrite.")
    else:
        print(f"Batch complete: {ok} rewritten, {rejected} rejected, "
              f"{skipped} skipped, {failed} failed.")
        if ok:
            print()
            print("Next steps:")
            print("  1. python blog_system.py build")
            print("  2. Re-run triage to confirm the improved posts moved:")
            print("       python scripts/adsense_triage.py --docs-dir ./docs --print-summary")
        if rejected:
            print()
            print("Rejected rewrites were discarded — the original post.json")
            print("is unchanged. Inspect the REJECT lines above to see why.")
        if failed:
            print()
            print("Failures were likely LLM provider issues. Check provider")
            print("connectivity with:")
            print("    python scripts/blog_llm_client.py --check")

    return 0


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite the posts in .adsense_triage/improve.json to remove "
            "fabricated first-person incidents and add missing structure. "
            "Backs up each post.json first; defaults to dry-run."
        )
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually write the rewrites and back up the originals.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview only. This is the default when --apply is omitted.",
    )
    parser.add_argument(
        "--limit", type=int, default=10,
        help=(
            "Max posts to process this run. Use 0 to process the entire "
            "improve bucket. Default: 10 — bounded so one run can't burn "
            "the whole provider chain."
        ),
    )
    parser.add_argument(
        "--slug", default=None,
        help="Rewrite a single post by slug (for debugging).",
    )
    parser.add_argument(
        "--triage-dir", default="./.adsense_triage",
        help="Directory containing improve.json (default: ./.adsense_triage).",
    )
    parser.add_argument(
        "--backup-dir", default="./.adsense_backups",
        help="Where to back up originals (default: ./.adsense_backups).",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-provider attempt lines; only print final status.",
    )
    args = parser.parse_args()

    # --dry-run is accepted for symmetry even though it's the default. If
    # neither flag was passed, print an explicit reminder so nobody thinks
    # the default is "write".
    if not args.apply and not args.dry_run:
        print("No action flag supplied. Defaulting to --dry-run.")
        print("Pass --apply to actually rewrite posts.\n")

    # Allow --backup-dir override to flow through to the module-level
    # constant used by _improve_one().
    global _BACKUP_DIR
    _BACKUP_DIR = Path(args.backup_dir)

    return asyncio.run(_run_batch(args))


if __name__ == "__main__":
    raise SystemExit(main())