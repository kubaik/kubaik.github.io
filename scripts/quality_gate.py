#!/usr/bin/env python3
"""
scripts/quality_gate.py

Full-corpus wiring for utils/content_quality_scanner.py.

WHY THIS EXISTS
----------------
content_quality_scanner.py already scores every post in docs/ on every
run (score_posts() and compute_similarities() operate on the full list
load_posts() returns — there is no sampling in the scoring itself). What
it does NOT do is:
  1. write a CSV covering the full corpus (its `report` mode CSV only
     contains the worst --per-month posts per calendar month, by design,
     since it's meant as a "worst offenders" report)
  2. run automatically as part of blog-automation.yml — it's a CLI tool
     a human has to remember to invoke.

This script imports the real scoring functions directly (no reimplementation,
no drift risk), scores 100% of docs/, writes a full-coverage CSV, and queues
anything under MIN_PASS_SCORE into regeneration_queue.json using the exact
same {"slug", "reason", "priority"} shape adsense_compliance_audit.py's
execute_improve() already reads and appends to. It does not delete or
rewrite anything itself — deletion/dedup stays in content_quality_scanner.py
prune/dedupe (which require --confirm) and the report-only watchdog steps
already in workflow_dispatch.yml. This keeps the "no manual review" property:
failing posts get queued for the existing automated regeneration pipeline,
nothing here waits on a human.

USAGE
-----
    python scripts/quality_gate.py
    python scripts/quality_gate.py --min-score 55
    python scripts/quality_gate.py --docs-dir ./docs --csv ./quality_report_full.csv

Exit code is always 0 (report-only step; does not fail the CI job) unless
the docs/ directory itself is missing or unreadable.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
UTILS_DIR = REPO_ROOT / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from content_quality_scanner import load_posts, compute_similarities, score_posts  # noqa: E402

DEFAULT_MIN_SCORE = 50.0
DEFAULT_CSV = REPO_ROOT / "quality_report_full.csv"
DEFAULT_QUEUE = REPO_ROOT / "regeneration_queue.json"


def write_full_csv(posts, csv_path: Path) -> None:
    fieldnames = [
        "month", "slug", "score", "word_count",
        "duplication_score", "depth_score", "meta_score",
        "structure_score", "affiliate_density_score",
        "most_similar_slug", "max_similarity",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in sorted(posts, key=lambda p: p.score):
            s = p.subscores
            writer.writerow({
                "month": p.month_key,
                "slug": p.slug,
                "score": p.score,
                "word_count": p.word_count,
                "duplication_score": s.get("duplication"),
                "depth_score": s.get("depth"),
                "meta_score": s.get("meta"),
                "structure_score": s.get("structure"),
                "affiliate_density_score": s.get("affiliate_density"),
                "most_similar_slug": p.most_similar_slug,
                "max_similarity": round(p.max_similarity, 3),
            })


def reason_for(p) -> str:
    s = p.subscores
    worst_axis = min(s, key=lambda k: s[k])
    axis_detail = {
        "duplication": f"near-duplicate of {p.most_similar_slug} (sim={p.max_similarity:.2f})",
        "depth": f"thin content ({p.word_count} words)",
        "meta": "weak/missing meta_description",
        "structure": "no code/table structure",
        "affiliate_density": "affiliate/ad density too high relative to length",
    }
    return f"quality_gate: score {p.score} < threshold — worst axis: {axis_detail.get(worst_axis, worst_axis)}"


def queue_failures(failing, queue_path: Path) -> int:
    existing = json.loads(queue_path.read_text()
                          ) if queue_path.exists() else []
    existing_slugs = {e["slug"] for e in existing}

    added = 0
    for p in failing:
        if p.slug in existing_slugs:
            continue
        existing.append({
            "slug": p.slug,
            "reason": reason_for(p),
            "priority": "high" if p.score < 30 else "normal",
        })
        added += 1

    if added:
        queue_path.write_text(json.dumps(existing, indent=2))
    return added


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs-dir", type=str,
                        default=str(REPO_ROOT / "docs"))
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE,
                        help=f"Posts scoring below this are queued for regeneration (default: {DEFAULT_MIN_SCORE})")
    parser.add_argument("--csv", type=str, default=str(DEFAULT_CSV),
                        help="Full-corpus CSV output path. Use '' to skip.")
    parser.add_argument("--queue", type=str, default=str(DEFAULT_QUEUE),
                        help="regeneration_queue.json path to append failures to.")
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir)
    posts = load_posts(docs_dir)
    total = len(posts)
    print(f"quality_gate: loaded {total} posts from {docs_dir}")

    if total == 0:
        print("quality_gate: nothing to score, exiting cleanly.")
        return 0

    compute_similarities(posts)
    score_posts(posts)

    if args.csv:
        csv_path = Path(args.csv)
        write_full_csv(posts, csv_path)
        print(
            f"quality_gate: wrote full-corpus report ({total} rows) -> {csv_path}")

    failing = [p for p in posts if p.score < args.min_score]
    print(
        f"quality_gate: {len(failing)}/{total} posts below min-score {args.min_score}")

    if failing:
        queue_path = Path(args.queue)
        added = queue_failures(failing, queue_path)
        print(
            f"quality_gate: queued {added} new post(s) for regeneration -> {queue_path}")
        for p in sorted(failing, key=lambda p: p.score)[:20]:
            print(f"  {p.score:5.1f}  {p.slug}")
        if len(failing) > 20:
            print(f"  ... and {len(failing) - 20} more (see {args.csv})")

    print("quality_gate: report-only step, nothing was deleted or blocked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
