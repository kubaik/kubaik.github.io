#!/usr/bin/env python3
"""
scripts/reconcile_cleanup_plan.py

cleanup_plan_v2.json's summary.total_posts (520) does not match the live
docs/ post count (446 at last check) when this repo was cloned for review.
That gap means the plan was generated against a docs/ state that no longer
exists — some slugs it references (either as the action's own `slug` or as
a `survivor_slug` a stub redirects to) may have been deleted since, by
content_quality_scanner.py prune/dedupe, adsense_compliance_audit.py, or a
manual `git rm`.

Chaining scripts against a stale plan is the actual risk: e.g.
adsense_compliance_audit.py's safe_to_delete() logic trusts that a
duplicate's survivor_slug is still live before approving a delete. If the
plan is stale, that check can pass on bad data.

This script does not implement any new deletion logic. It re-validates
the existing plan's entries against docs/ as it actually is right now,
drops anything that references a slug that's gone, and writes a fresh
cleanup_plan_v2.json with recomputed summary counts. If nothing changed,
it says so and exits without writing.

USAGE
-----
    python scripts/reconcile_cleanup_plan.py
    python scripts/reconcile_cleanup_plan.py --plan cleanup_plan_v2.json --docs-dir ./docs
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PLAN = REPO_ROOT / "cleanup_plan_v2.json"
DEFAULT_DOCS = REPO_ROOT / "docs"


def live_slugs(docs_dir: Path) -> set:
    """Slugs that actually exist on disk right now (have a post directory,
    post.json or not — a stub redirect dir counts as 'live' too since it
    can still be a valid survivor target)."""
    return {p.name for p in docs_dir.iterdir() if p.is_dir()}


def reconcile(plan: dict, live: set) -> tuple[dict, list[str]]:
    dropped = []
    kept_actions = []

    for action in plan.get("actions", []):
        slug = action.get("slug")
        survivor = action.get("survivor_slug")

        missing = []
        if slug and slug not in live:
            missing.append(f"slug '{slug}' not in docs/")
        if survivor and survivor not in live:
            missing.append(f"survivor_slug '{survivor}' not in docs/")

        if missing:
            dropped.append(f"{slug or '?'}: " + "; ".join(missing))
            continue

        kept_actions.append(action)

    flagged = [
        s for s in plan.get("thin_content_flagged_for_manual_review", [])
        if (s.get("slug") if isinstance(s, dict) else s) in live
    ]

    noindex_only = sum(1 for a in kept_actions if a.get(
        "action") == "NOINDEX_ONLY")
    merge_redirect = sum(1 for a in kept_actions if a.get(
        "action") == "MERGE_REDIRECT")
    unaffected = len(live) - len({a.get("slug") for a in kept_actions})

    new_plan = {
        "summary": {
            "total_posts": len(live),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "reconciled_from": plan.get("summary", {}).get("generated_at", "unknown"),
            "threshold": plan.get("summary", {}).get("threshold"),
            "merge_redirect": merge_redirect,
            "noindex_only": noindex_only,
            "unaffected": unaffected,
            "dropped_stale_actions": len(dropped),
        },
        "actions": kept_actions,
        "thin_content_flagged_for_manual_review": flagged,
    }
    return new_plan, dropped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=str, default=str(DEFAULT_PLAN))
    parser.add_argument("--docs-dir", type=str, default=str(DEFAULT_DOCS))
    parser.add_argument("--write", action="store_true",
                        help="Actually overwrite the plan file. Default is dry-run (prints the diff only).")
    args = parser.parse_args()

    plan_path = Path(args.plan)
    docs_dir = Path(args.docs_dir)

    if not plan_path.exists():
        print(f"No plan at {plan_path} — nothing to reconcile.")
        return 0
    if not docs_dir.exists():
        print(f"ERROR: {docs_dir} does not exist.")
        return 1

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    live = live_slugs(docs_dir)

    old_total = plan.get("summary", {}).get("total_posts", "?")
    print(
        f"Plan claims total_posts={old_total}; docs/ actually has {len(live)} post directories.")

    new_plan, dropped = reconcile(plan, live)

    if not dropped:
        print("No stale references found — plan is consistent with docs/ as-is.")
        if old_total != len(live):
            print(f"(total_posts count itself was stale though: {old_total} -> {len(live)}"
                  f"{' — rewriting summary only' if args.write else ', use --write to fix'})")
            if args.write:
                plan_path.write_text(json.dumps(
                    new_plan, indent=2), encoding="utf-8")
                print(f"Wrote updated summary to {plan_path}")
        return 0

    print(
        f"\n{len(dropped)} stale action(s) found (referencing a slug no longer in docs/):")
    for d in dropped:
        print(f"  DROP: {d}")

    if args.write:
        plan_path.write_text(json.dumps(new_plan, indent=2), encoding="utf-8")
        print(f"\nWrote reconciled plan -> {plan_path}")
        print(
            f"  actions: {len(plan.get('actions', []))} -> {len(new_plan['actions'])}")
    else:
        print(f"\nDry run — plan not written. Re-run with --write to apply.")

    return 0


if __name__ == "__main__":
    exit(main())
