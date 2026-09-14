"""
scripts/process_regeneration_queue.py
========================================
Bounded, automated consumer for the fabrication entries
scripts/retroactive_fabrication_audit.py appends to regeneration_queue.json.

WHY THIS EXISTS
---------------
retroactive_fabrication_audit.py only ever appends to the queue file —
nothing consumed it, so flagged posts sat there indefinitely (the same
gap auto_retire_duplicates.py closes for topic_dedup.py's clusters).
This is that consumer, scoped ONLY to entries this repo's own
fabrication scan produced (identified by the "retroactive_fabrication_audit:"
reason prefix) — quality_gate.py's lower-level score-based entries are
left untouched for manual review, since "score < 50" doesn't tell you
WHAT to remove the way a specific regex match does.

HOW IT REMOVES CONTENT
-----------------------
This does NOT reimplement the fabrication rules. It imports the exact
same gate functions and pattern lists blog_system.py's generation-time
gates use (_FABRICATED_CITATION_PATTERNS, _LEGITIMATE_SOURCE_CONTEXT,
_SKIP_PATTERNS, _reject_if_fabricated_citation, _flag_fabricated_anecdotes)
so there is no drift between "what blocks a new post" and "what this
strips from an old one." For each queued post:
  1. Split content into paragraphs, mask fenced code blocks.
  2. Within each non-code paragraph, split into sentences and drop any
     sentence that (a) matches _SKIP_PATTERNS (fabricated first-person
     anecdote opener) or (b) contains a _FABRICATED_CITATION_PATTERNS
     match not covered by either of that gate's own allow-lists (a
     nearby URL, or nearby legitimate-documentation context).
  3. Re-run _reject_if_fabricated_citation() and _flag_fabricated_anecdotes()
     on the RESULT. Only if the result now comes back clean does this
     write the change — if stripping sentence-by-sentence didn't fully
     clear it (e.g. the fabricated claim spans multiple sentences), the
     post is left untouched and stays in the queue with a note, rather
     than shipping a partial, possibly-still-fabricated edit.

SAFETY MODEL
------------
Same posture as auto_retire_duplicates.py:
  - --max-posts bounds how many posts get edited per run.
  - Every post.json is copied to --backup-dir before being modified.
  - Without --confirm, this only reports what it would change.
  - A post this can't fully clean is left alone and flagged, not forced.

USAGE
-----
    # Dry run:
    python scripts/process_regeneration_queue.py --docs-dir ./docs --queue ./regeneration_queue.json

    # Apply, bounded:
    python scripts/process_regeneration_queue.py \\
        --docs-dir ./docs \\
        --queue ./regeneration_queue.json \\
        --max-posts 5 \\
        --backup-dir ./.fabrication_backups \\
        --confirm
"""

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from blog_system import (  # noqa: E402
    _FABRICATED_CITATION_PATTERNS,
    _LEGITIMATE_SOURCE_CONTEXT,
    _SKIP_PATTERNS,
    _reject_if_fabricated_citation,
    _flag_fabricated_anecdotes,
)

_QUEUE_MARKER = "retroactive_fabrication_audit:"
_CONTEXT_WINDOW = 200  # matches the window blog_system.py's own gate uses


def _mask_code_blocks(content: str) -> Tuple[str, List[str]]:
    blocks: List[str] = []

    def _mask(m: "re.Match") -> str:
        blocks.append(m.group(0))
        return f"\x00CODE{len(blocks) - 1}\x00"

    return re.sub(r"```[\s\S]*?```", _mask, content), blocks


def _restore_code_blocks(content: str, blocks: List[str]) -> str:
    for i, block in enumerate(blocks):
        content = content.replace(f"\x00CODE{i}\x00", block)
    return content


def _sentence_spans(text: str) -> List[Tuple[int, int, str]]:
    """(start, end, sentence_text) for each sentence, positions into `text`."""
    spans = []
    pos = 0
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if not sent:
            continue
        idx = text.find(sent, pos)
        if idx == -1:
            continue
        spans.append((idx, idx + len(sent), sent))
        pos = idx + len(sent)
    return spans


def _find_fabricated_citation_spans(text: str) -> List[Tuple[int, int]]:
    """
    Match spans in `text` that trip a citation pattern and survive neither
    allow-list — same two checks _reject_if_fabricated_citation() applies,
    just returning every span instead of stopping at the first.
    """
    bad_spans = []
    for pattern in _FABRICATED_CITATION_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            window = text[max(0, m.start() - _CONTEXT_WINDOW): m.end() + _CONTEXT_WINDOW]
            if re.search(r"https?://", window):
                continue
            if _LEGITIMATE_SOURCE_CONTEXT.search(window):
                continue
            bad_spans.append((m.start(), m.end()))
    return bad_spans


def strip_fabricated_content(content: str) -> Tuple[str, int]:
    """
    Remove sentences that trip either fabrication gate, preserving
    everything else including code blocks. Returns (new_content, removed_count).
    """
    masked, code_blocks = _mask_code_blocks(content)
    paragraphs = masked.split("\n\n")
    new_paragraphs = []
    removed = 0

    for para in paragraphs:
        # Skip structural paragraphs entirely — same convention
        # internal_linker.py already uses for headings/code/tables/quotes,
        # extended here to bullet and numbered list blocks. Without this,
        # a multi-line list with no blank line between items is one
        # "paragraph" to split(\"\\n\\n\"); sentence-splitting that whole
        # block on punctuation cuts across list-item boundaries, so one
        # flagged citation partway through a list can pull adjacent,
        # unrelated list items out with it. Confirmed on a real post: a
        # 9-row cost-comparison list lost ~30 unrelated lines this way
        # before this guard was added. A citation fabricated INSIDE a list
        # item is rarer and left for manual review (still caught by the
        # re-check below, which will flag the post as still-dirty) rather
        # than risking further collateral damage from list-aware splitting.
        stripped_para = para.strip()
        is_list_block = bool(re.match(r'^(-|\*|\d+\.)\s', stripped_para))
        if "\x00CODE" in para or stripped_para.startswith(("#", "|", ">")) or is_list_block:
            new_paragraphs.append(para)
            continue

        bad_spans = _find_fabricated_citation_spans(para)
        kept_sentences = []
        for start, end, sent in _sentence_spans(para):
            stripped_sent = sent.strip()
            if not stripped_sent:
                continue

            if _SKIP_PATTERNS.match(stripped_sent):
                removed += 1
                continue

            overlaps_bad_span = any(
                start < b_end and end > b_start for b_start, b_end in bad_spans
            )
            if overlaps_bad_span:
                removed += 1
                continue

            kept_sentences.append(sent)

        new_para = " ".join(kept_sentences).strip()
        if new_para:
            new_paragraphs.append(new_para)
        # A paragraph that becomes entirely empty is dropped, not kept as
        # a blank line — avoids leaving stray double gaps in the output.

    new_content = "\n\n".join(new_paragraphs)
    return _restore_code_blocks(new_content, code_blocks), removed


def _backup_post_json(post_json: Path, backup_dir: Path, slug: str) -> Path:
    dest = backup_dir / f"{slug}.post.json"
    if dest.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dest = backup_dir / f"{slug}.post.json__{stamp}"
    shutil.copy2(post_json, dest)
    return dest


def run(
    docs_dir: Path,
    queue_path: Path,
    max_posts: int,
    backup_dir: Path,
    confirm: bool,
) -> int:
    if not queue_path.exists():
        print(f"{queue_path} not found — nothing to process.")
        return 0

    try:
        queue = json.loads(queue_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"⚠️  {queue_path} is not valid JSON — aborting without changes.")
        return 1

    pending = [
        e for e in queue
        if isinstance(e, dict)
        and str(e.get("reason", "")).startswith(_QUEUE_MARKER)
        and e.get("status", "pending") == "pending"
    ]

    if not pending:
        print("No pending fabrication-audit entries in the queue. Nothing to do.")
        return 0

    batch = pending[:max_posts]
    deferred = pending[max_posts:]

    print(f"{len(pending)} pending fabrication entr{'y' if len(pending) == 1 else 'ies'} in queue.")
    print(f"{'Processing' if confirm else 'Would process'} {len(batch)} this run "
          f"(--max-posts {max_posts}):\n")

    if not confirm:
        for e in batch:
            print(f"  {e['slug']}  | {e.get('title', '')}")
        if deferred:
            print(f"\n{len(deferred)} more deferred to a future run: "
                  f"{', '.join(d['slug'] for d in deferred)}")
        print("\nDry run — nothing was written. Re-run with --confirm to apply.")
        return 0

    backup_dir.mkdir(parents=True, exist_ok=True)
    cleaned, needs_review = 0, 0

    for entry in batch:
        slug = entry["slug"]
        post_json = docs_dir / slug / "post.json"
        if not post_json.exists():
            print(f"  ⚠️  {slug}: post.json missing (likely retired by another script this run) — marking resolved.")
            entry["status"] = "resolved_elsewhere"
            continue

        try:
            data = json.loads(post_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"  ⚠️  {slug}: post.json is not valid JSON — skipping, left in queue.")
            continue

        if data.get("redirect_to"):
            print(f"  ⏭️  {slug}: already a redirect stub (see auto_retire_duplicates.py) "
                  f"— nothing to fix, marking resolved.")
            entry["status"] = "resolved_elsewhere"
            continue

        content = data.get("content", "")
        new_content, removed_count = strip_fabricated_content(content)

        still_flagged = (
            _reject_if_fabricated_citation(new_content) is not None
            or bool(_flag_fabricated_anecdotes(new_content))
        )

        if still_flagged:
            print(f"  ⚠️  {slug}: removed {removed_count} sentence(s) but the gate "
                  f"still flags the result — leaving post.json untouched, "
                  f"marking for manual review.")
            entry["status"] = "needs_review"
            needs_review += 1
            continue

        backup_path = _backup_post_json(post_json, backup_dir, slug)
        data["content"] = new_content
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        post_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"  ✅ {slug}: removed {removed_count} sentence(s), gate now clean. "
              f"Backup at {backup_path}")
        entry["status"] = "stripped"
        cleaned += 1

    queue_path.write_text(json.dumps(queue, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{cleaned} post(s) cleaned, {needs_review} flagged for manual review, "
          f"{len(deferred)} deferred.")
    if cleaned:
        print("   Run 'python blog_system.py build' to regenerate the affected "
              "index.html files before the sitemap and OG steps.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bounded, automated consumer for fabrication-audit queue entries."
    )
    parser.add_argument("--docs-dir", default="./docs")
    parser.add_argument("--queue", default="./regeneration_queue.json")
    parser.add_argument("--max-posts", type=int, default=5)
    parser.add_argument("--backup-dir", default="./.fabrication_backups")
    parser.add_argument("--confirm", action="store_true", help="Actually write changes. Omit for a dry run.")
    args = parser.parse_args()

    sys.exit(run(
        docs_dir=Path(args.docs_dir),
        queue_path=Path(args.queue),
        max_posts=args.max_posts,
        backup_dir=Path(args.backup_dir),
        confirm=args.confirm,
    ))