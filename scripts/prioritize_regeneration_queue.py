"""
scripts/prioritize_regeneration_queue.py
===========================================
process_regeneration_queue.py processes queue entries in file order and is
bounded by --max-posts. With 443 unprocessed entries and a 5-per-run cap,
file order becomes the de facto priority — and right now that's insertion
order from retroactive_fabrication_audit.py, not severity.

Confirmed from a live dry run (2026-09-15): the 5 entries selected had no
particular severity, while posts like
'7-ways-to-earn-extra-in-2026-without-shipping-saas' (58 flagged sentences)
and 'launch-saas-in-6-weeks-the-ai-shortcut-i-actually-used' (43) sat in the
438-deferred tail behind lighter cases.

This script re-sorts regeneration_queue.json in place, worst-first, by the
anecdote count already embedded in each entry's `reason` string (e.g.
"9 unverifiable first-person anecdote(s)..."). It does NOT touch post
content and does NOT touch entry status — it only changes ordering, so it's
safe to run before every process_regeneration_queue.py invocation with no
risk to the "backup + re-verify before write" safety model that script
already has.

Entries already resolved (status in {"stripped", "resolved_elsewhere"}) are
sorted to the back regardless of severity, since they don't need processing.

HOW TO INTEGRATE
----------------
In blog-automation.yml, run this immediately before process_regeneration_queue.py:

    - name: Prioritize fabrication queue by severity
      run: python scripts/prioritize_regeneration_queue.py --queue ./regeneration_queue.json

    - name: Process fabrication queue (worst offenders first)
      run: |
        python scripts/process_regeneration_queue.py \\
          --docs-dir ./docs --queue ./regeneration_queue.json \\
          --max-posts 5 --backup-dir ./.fabrication_backups --confirm

USAGE
-----
    # Preview the new order without writing:
    python scripts/prioritize_regeneration_queue.py --queue ./regeneration_queue.json

    # Apply:
    python scripts/prioritize_regeneration_queue.py --queue ./regeneration_queue.json --confirm
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

_RESOLVED_STATUSES = {"stripped", "resolved_elsewhere"}
_ANECDOTE_COUNT_RE = re.compile(r"(\d+)\s+unverifiable")


def _severity(entry: Dict) -> int:
    """Extract the flagged-sentence count from an entry's reason string."""
    m = _ANECDOTE_COUNT_RE.search(entry.get("reason", ""))
    return int(m.group(1)) if m else 0


def prioritize(queue: List[Dict]) -> List[Dict]:
    """
    Return a new list: unresolved entries first (worst severity first),
    then resolved entries (order among these doesn't matter).
    """
    unresolved = [e for e in queue if e.get("status") not in _RESOLVED_STATUSES]
    resolved = [e for e in queue if e.get("status") in _RESOLVED_STATUSES]
    unresolved.sort(key=_severity, reverse=True)
    return unresolved + resolved


def run(queue_path: Path, confirm: bool) -> int:
    if not queue_path.exists():
        print(f"❌ {queue_path} not found.")
        return 1

    try:
        queue = json.loads(queue_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"❌ {queue_path} is not valid JSON: {e}")
        return 1

    reordered = prioritize(queue)

    unresolved_count = sum(1 for e in reordered if e.get("status") not in _RESOLVED_STATUSES)
    print(f"{len(queue)} total entries, {unresolved_count} unresolved.")
    print("\nTop 10 by severity after reordering:")
    for e in reordered[:10]:
        status = e.get("status") or "unprocessed"
        print(f"  {_severity(e):>3} flagged sentence(s)  [{status:<18}]  {e.get('slug')}")

    if not confirm:
        print("\nDry run — nothing was written. Re-run with --confirm to apply.")
        return 0

    queue_path.write_text(json.dumps(reordered, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✅ {queue_path} reordered worst-first and written.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reorder regeneration_queue.json by fabrication severity (worst first)."
    )
    parser.add_argument("--queue", default="./regeneration_queue.json")
    parser.add_argument("--confirm", action="store_true", help="Actually write the reordered queue. Omit for a dry run.")
    args = parser.parse_args()
    sys.exit(run(Path(args.queue), args.confirm))