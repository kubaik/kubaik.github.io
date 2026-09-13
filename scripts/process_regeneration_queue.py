#!/usr/bin/env python3
"""
scripts/process_regeneration_queue.py

Consumes the fabrication-audit entries in regeneration_queue.json and
closes the loop that scripts/retroactive_fabrication_audit.py left
open: that script only ever WRITES to the queue, nothing reads it.

WHY "STRIP-AND-KEEP" RATHER THAN DELETE-AND-REGENERATE
--------------------------------------------------------
A flagged post is usually a decent article with one bad sentence in
it (an invented statistic, a fabricated "when I tested this..."
anecdote) — not a bad article. Deleting the whole post throws away
everything else in it and costs a slug/URL/backlink. This script
removes only the sentence(s) that trip the SAME two gate functions
blog_system.py already uses at generation time:

    _reject_if_fabricated_citation(content)  -> str | None
    _flag_fabricated_anecdotes(content)      -> List[str]

There is no reimplementation of the fabrication regexes here — the
real functions are imported and called directly, exactly like
retroactive_fabrication_audit.py already does. That matters because
_reject_if_fabricated_citation() only reports its FIRST match, and
allow-lists a hit if a real URL or legitimate-documentation phrase
(e.g. "official docs", "changelog") appears within 200 characters of
it — logic this script does not try to duplicate. Instead:

  1. Call the real gate function against the CURRENT content.
  2. If it flags something, locate the exact matched phrase it names
     in its reason string, find the sentence that phrase sits inside,
     and remove that whole sentence.
  3. Re-run the gate against the new content and repeat (bounded), so
     later matches are evaluated with correct, up-to-date context —
     never against a stale offset.
  4. After stripping, re-run BOTH gates on the full resulting content.
     If anything is still flagged, the post is left untouched and
     reported as "needs manual review" instead of being force-edited.
     This script never guesses past what the real gates confirm.

SCOPE — READ THIS BEFORE WIRING INTO CI
-----------------------------------------
This script only processes queue entries written by
retroactive_fabrication_audit.py (reason string starts with
"retroactive_fabrication_audit:"). Entries written by quality_gate.py
(thin content, near-duplicate, weak meta, etc.) are left untouched and
reported as skipped — stripping a sentence doesn't fix thin content or
a weak meta_description, so bolting that logic in here would be
unsafe. Those either need a real regen (existing
`python blog_system.py auto` content pipeline) or a human, and staying
out of their way is intentional, not an oversight.

USAGE
-----
    # Dry run — see exactly what would be stripped, nothing written
    python scripts/process_regeneration_queue.py --docs-dir ./docs

    # Apply, bounded to 5 posts this run
    python scripts/process_regeneration_queue.py \\
        --docs-dir ./docs \\
        --max-posts 5 \\
        --backup-dir ./.fabrication_backups \\
        --confirm

Recommended CI wiring, right after the retroactive fabrication audit
step (bounded, same pattern as auto_retire_duplicates.py):

    - name: 🧹 Process fabrication regeneration queue (bounded)
      run: |
        python scripts/process_regeneration_queue.py \\
          --docs-dir ./docs \\
          --max-posts 5 \\
          --backup-dir ./.fabrication_backups \\
          --confirm

After this runs, rebuild the site (scripts/rebuild_site.py or
`python blog_system.py build`) so the edited post.json is re-rendered
to HTML — the existing blog-automation.yml build/sitemap/OG steps
already do this on every scheduled run.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from blog_system import (  # noqa: E402
    _flag_fabricated_anecdotes,
    _reject_if_fabricated_citation,
)

DEFAULT_MAX_POSTS = 5
DEFAULT_BACKUP_DIR = REPO_ROOT / ".fabrication_backups"
DEFAULT_QUEUE = REPO_ROOT / "regeneration_queue.json"
DEFAULT_DOCS_DIR = REPO_ROOT / "docs"
DEFAULT_REPORT_PATH = REPO_ROOT / "fabrication_strip_report.json"
MAX_STRIP_ITERATIONS = 8

_QUEUE_SOURCE_PREFIX = "retroactive_fabrication_audit:"

# Matches the phrase _reject_if_fabricated_citation() quotes in its own
# reason string: "unverifiable named-source citation: '<phrase>'"
_CITATION_REASON_RE = re.compile(r": '(.+)'$")


def _tidy_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _sentence_bounds(content: str, around_index: int) -> Tuple[int, int]:
    """
    Return (start, end) offsets of the sentence containing around_index,
    using simple terminal-punctuation scanning. Good enough given the
    final full-content re-verification step backstops any imprecision:
    if a boundary is slightly off, the worst case is either an adjacent
    fragment gets pulled in/left behind, and the re-verify step below
    catches whether the result actually passes the real gates before
    anything is written.
    """
    before = content[:around_index]
    backs = list(re.finditer(r"[.!?]\s+", before))
    start = backs[-1].end() if backs else 0

    after = content[around_index:]
    fwd = re.search(r"[.!?]", after)
    end = around_index + fwd.end() if fwd else len(content)

    return start, end


def _strip_one_citation_hit(content: str) -> Optional[Tuple[str, str]]:
    """
    Run the real _reject_if_fabricated_citation() gate once. If it flags
    something, remove the sentence containing the exact phrase it named
    and return (new_content, removed_sentence). Returns None if nothing
    was flagged, or if the flagged phrase couldn't be located verbatim
    in the content (defensive — should not normally happen).
    """
    issue = _reject_if_fabricated_citation(content)
    if not issue:
        return None
    m = _CITATION_REASON_RE.search(issue)
    if not m:
        return None
    phrase = m.group(1)
    idx = content.find(phrase)
    if idx == -1:
        return None
    start, end = _sentence_bounds(content, idx)
    removed = content[start:end].strip()
    new_content = _tidy_whitespace(content[:start] + content[end:])
    return new_content, removed


def _strip_one_anecdote_hit(content: str) -> Optional[Tuple[str, str]]:
    """
    Run the real _flag_fabricated_anecdotes() gate once. It returns
    sentences already truncated to 80 chars, matching a sentence's
    actual start (the gate matches _SKIP_PATTERNS against sentence
    starts). Locate that prefix in content and remove the full sentence.
    """
    hits = _flag_fabricated_anecdotes(content)
    if not hits:
        return None
    prefix = hits[0][:40]
    idx = content.find(prefix)
    if idx == -1:
        return None
    start, end = _sentence_bounds(content, idx)
    removed = content[start:end].strip()
    new_content = _tidy_whitespace(content[:start] + content[end:])
    return new_content, removed


def strip_fabricated_content(content: str) -> Tuple[str, List[str]]:
    """
    Repeatedly call the real gate functions against the current content
    and remove one flagged sentence at a time, bounded by
    MAX_STRIP_ITERATIONS. Returns (new_content, removed_sentences).
    """
    working = content
    removed: List[str] = []

    for _ in range(MAX_STRIP_ITERATIONS):
        progressed = False

        citation_result = _strip_one_citation_hit(working)
        if citation_result:
            working, removed_sentence = citation_result
            if removed_sentence:
                removed.append(removed_sentence)
            progressed = True

        anecdote_result = _strip_one_anecdote_hit(working)
        if anecdote_result:
            working, removed_sentence = anecdote_result
            if removed_sentence:
                removed.append(removed_sentence)
            progressed = True

        if not progressed:
            break

    return working.strip(), removed


def load_queue(queue_path: Path) -> List[Dict]:
    if not queue_path.exists():
        return []
    try:
        data = json.loads(queue_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"  ⚠️  {queue_path} was not valid JSON — treating as empty.")
        return []
    return data if isinstance(data, list) else []


def is_fabrication_entry(entry: Dict) -> bool:
    reason = entry.get("reason", "")
    return isinstance(reason, str) and reason.startswith(_QUEUE_SOURCE_PREFIX)


def backup_post_json(post_json: Path, backup_dir: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir.mkdir(parents=True, exist_ok=True)
    dest = backup_dir / f"{post_json.parent.name}-{timestamp}.post.json"
    shutil.copy2(post_json, dest)
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs-dir", type=str, default=str(DEFAULT_DOCS_DIR))
    parser.add_argument("--queue", type=str, default=str(DEFAULT_QUEUE))
    parser.add_argument("--max-posts", type=int, default=DEFAULT_MAX_POSTS,
                        help=f"Cap on how many posts this run can edit (default: {DEFAULT_MAX_POSTS})")
    parser.add_argument("--backup-dir", type=str, default=str(DEFAULT_BACKUP_DIR),
                        help="Where the original post.json is copied before editing")
    parser.add_argument("--report", type=str, default=str(DEFAULT_REPORT_PATH),
                        help="Where to write a JSON report of what happened this run")
    parser.add_argument("--confirm", action="store_true",
                        help="Actually write changes. Without this flag, dry-run only.")
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir)
    queue_path = Path(args.queue)
    backup_dir = Path(args.backup_dir)
    report_path = Path(args.report)

    queue = load_queue(queue_path)
    fabrication_entries = [e for e in queue if isinstance(e, dict) and is_fabrication_entry(e)]
    other_entries = [e for e in queue if not (isinstance(e, dict) and is_fabrication_entry(e))]

    print(
        f"process_regeneration_queue: {len(queue)} total entr{'y' if len(queue)==1 else 'ies'} in queue — "
        f"{len(fabrication_entries)} from retroactive_fabrication_audit, "
        f"{len(other_entries)} from other sources (skipped, out of scope for strip-and-keep)."
    )

    to_process = fabrication_entries[: args.max_posts]
    remaining = fabrication_entries[args.max_posts:]
    print(
        f"  -> processing {len(to_process)} this run "
        f"(capped at --max-posts {args.max_posts}); "
        f"{len(remaining)} left for a future run."
    )

    report_entries = []
    processed_slugs = set()

    for entry in to_process:
        slug = entry.get("slug", "")
        post_dir = docs_dir / slug
        post_json_path = post_dir / "post.json"

        result = {
            "slug": slug,
            "queue_reason": entry.get("reason", ""),
            "status": None,
            "removed_sentences": [],
            "backup_path": None,
        }

        if not post_json_path.exists():
            result["status"] = "SKIPPED — post.json not found (post may already be retired/deleted)"
            report_entries.append(result)
            continue

        try:
            data = json.loads(post_json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            result["status"] = "SKIPPED — post.json is not valid JSON"
            report_entries.append(result)
            continue

        content = data.get("content", "")
        if not content:
            result["status"] = "SKIPPED — post has no content"
            report_entries.append(result)
            continue

        new_content, removed = strip_fabricated_content(content)

        if not removed:
            result["status"] = (
                "SKIPPED — could not locate a removable sentence for the flagged "
                "issue (gate may need manual review — see queue_reason)"
            )
            report_entries.append(result)
            continue

        # Re-verify against the REAL gates on the full resulting content.
        # If anything is still flagged, don't guess further — leave the
        # post untouched and flag for a human.
        still_flagged = bool(_reject_if_fabricated_citation(new_content)) or bool(
            _flag_fabricated_anecdotes(new_content)
        )
        if still_flagged:
            result["status"] = (
                "SKIPPED — still flagged by the real gates after stripping "
                f"{len(removed)} sentence(s) (hit MAX_STRIP_ITERATIONS or a "
                "residual case); needs manual review"
            )
            result["removed_sentences"] = removed
            report_entries.append(result)
            continue

        result["removed_sentences"] = removed

        if args.confirm:
            backup_path = backup_post_json(post_json_path, backup_dir)
            data["content"] = new_content
            post_json_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            result["status"] = "APPLIED"
            result["backup_path"] = str(backup_path)
            processed_slugs.add(slug)
            print(f"  ✅ {slug}: removed {len(removed)} sentence(s), backed up to {backup_path}")
        else:
            result["status"] = "WOULD APPLY (dry-run)"
            print(f"  🔎 {slug}: would remove {len(removed)} sentence(s) —")
            for s in removed:
                print(f"       - {s[:140]}{'…' if len(s) > 140 else ''}")

        report_entries.append(result)

    # Remove successfully-processed entries from the queue so re-running
    # this script doesn't reprocess them. Entries left untouched (other
    # sources, skipped, still-flagged) stay in the queue.
    if args.confirm and processed_slugs:
        new_queue = [e for e in queue if not (isinstance(e, dict) and e.get("slug") in processed_slugs)]
        queue_path.write_text(json.dumps(new_queue, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nQueue updated: removed {len(processed_slugs)} resolved entr{'y' if len(processed_slugs)==1 else 'ies'} -> {queue_path}")

    report_path.write_text(json.dumps({
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "confirm": args.confirm,
        "max_posts": args.max_posts,
        "fabrication_entries_in_queue": len(fabrication_entries),
        "other_entries_skipped": len(other_entries),
        "processed_this_run": report_entries,
        "remaining_for_future_runs": len(remaining),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Report written to {report_path}")

    if not args.confirm:
        print("\nDry run only — nothing was changed. Re-run with --confirm to apply.")

    return 0


if __name__ == "__main__":
    sys.exit(main())