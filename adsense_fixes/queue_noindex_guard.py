"""
adsense_fixes/queue_noindex_guard.py
======================================
Closes the dead-end in scripts/process_regeneration_queue.py: a post
that reaches status="needs_review" (fabricated claim spans multiple
sentences, sentence-stripping can't fully clear it) currently has NO
automated path to resolution and stays fully live/indexed indefinitely.
Same problem for the ~444 entries still at status=None waiting for their
turn under --max-posts 5.

This does not try to auto-write replacement prose (that's exactly the
kind of one-off editorial judgment call the bounded, backup-and-diff
posture of process_regeneration_queue.py deliberately avoids making
automatically). Instead it removes the AdSense/Search exposure while
the queue drains: any post whose slug currently appears in
regeneration_queue.json gets noindex'd. The moment
process_regeneration_queue.py marks it "stripped", it's automatically
re-included on the next site build.

HOW TO INTEGRATE
----------------
Run in blog-automation.yml BEFORE static_site_generator.py builds, and
again after process_regeneration_queue.py runs (statuses change):

    from adsense_fixes.queue_noindex_guard import get_quarantined_slugs
    quarantined = get_quarantined_slugs(Path("./regeneration_queue.json"))

In static_site_generator.py's post-render step:

    if post.slug in quarantined:
        robots_meta = '<meta name="robots" content="noindex,follow">'

In the sitemap generator, skip any slug in `quarantined` the same way
should_noindex_post() stubs are skipped (see canonical_guard.py).

This is intentionally a pure read of regeneration_queue.json — it does
not mutate the queue or post content. process_regeneration_queue.py
remains the only writer of post content.
"""

import json
from pathlib import Path
from typing import Dict, List, Set


# Statuses that mean "this post's fabrication issue is NOT resolved" —
# keep it out of the index until process_regeneration_queue.py clears it.
_UNRESOLVED_STATUSES = {None, "needs_review"}

# Statuses that mean the queue entry is closed and the post is safe to
# index again (either cleaned, or determined elsewhere not to need it).
_RESOLVED_STATUSES = {"stripped", "resolved_elsewhere"}


def get_quarantined_slugs(queue_path: Path) -> Set[str]:
    """
    Return the set of slugs that must be noindexed right now because
    their regeneration_queue.json entry is not yet resolved.
    """
    if not queue_path.exists():
        return set()

    try:
        queue = json.loads(queue_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()

    quarantined: Set[str] = set()
    for entry in queue:
        slug = entry.get("slug")
        status = entry.get("status")
        if not slug:
            continue
        if status in _UNRESOLVED_STATUSES:
            quarantined.add(slug)
    return quarantined


def quarantine_report(queue_path: Path) -> str:
    """Human-readable summary for CI logs / the auto-mode run output."""
    if not queue_path.exists():
        return "No regeneration_queue.json found — nothing quarantined."

    try:
        queue = json.loads(queue_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "regeneration_queue.json is not valid JSON — treating as empty."

    by_status: Dict[str, List[str]] = {}
    for entry in queue:
        status = entry.get("status") or "unprocessed"
        by_status.setdefault(status, []).append(entry.get("slug", "?"))

    total = len(queue)
    quarantined = sum(
        len(v) for k, v in by_status.items()
        if k == "unprocessed" or k == "needs_review"
    )

    lines = [
        f"🔒 Fabrication queue: {total} entries, {quarantined} currently noindexed pending cleanup.",
    ]
    for status, slugs in sorted(by_status.items()):
        flag = " (QUARANTINED)" if status in ("unprocessed", "needs_review") else ""
        lines.append(f"   {status:<20} {len(slugs):>4}{flag}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    qp = Path(sys.argv[1] if len(sys.argv) > 1 else "./regeneration_queue.json")
    print(quarantine_report(qp))