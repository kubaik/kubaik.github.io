"""
scripts/auto_retire_duplicates.py
====================================
Bounded, automated retirement of confirmed duplicate-topic posts found by
adsense_fixes/topic_dedup.py's find_clusters().

WHY THIS EXISTS
---------------
find_clusters() is read-only reporting. Nothing consumed its output —
same gap as regeneration_queue.json before retroactive_fabrication_audit.py
existed. This is that consumer for duplicate-topic clusters specifically.

Run against the live corpus, every non-keep cluster member scores
50-67% pairwise against its specific keeper (see topic_dedup.py's
score_vs_keep field) — i.e. every cluster currently found is a
high-confidence match, not a borderline one. --min-score still exists
as a dial for the future, in case looser clustering ever produces
weaker pairs.

SAFETY MODEL
------------
This script deletes content, so it does NOT get the "obviously safe,
always automate" treatment retroactive_fabrication_audit.py's read-only
scan got. Instead:
  1. --max-deletions caps blast radius per run (same philosophy as
     VelocityController's daily post cap) — a bad match can only take
     out a small, boundable number of posts before a human notices,
     rather than potentially the whole matching set in one run.
  2. --min-score only acts on the higher-confidence half of a cluster's
     pairwise scores.
  3. Nothing is ever actually deleted from disk. The retired post's
     directory is copied to --backup-dir first, in full, before
     anything is touched.
  4. The retired post's directory is REPLACED (not removed) with a
     noindex + meta-refresh redirect stub pointing at the keeper, so
     inbound links/backlinks/bookmarks land somewhere real instead of
     404ing. generate_sitemap.py already excludes both noindex pages
     and meta-refresh stubs (verified directly in that script's
     extract_page_data(), not assumed), so the retired URL drops out
     of the next sitemap build with no separate wiring needed.
  5. The stub's post.json is intentionally kept, not deleted, and
     carries `"redirect_to": "<keeper-slug>"`. This matters more than
     it looks: deleting post.json outright would make
     link_validator.py's _get_published_slugs() (which gates purely on
     post.json existing) start reporting every post that links to this
     now-retired slug as "broken" forever, even though the URL works
     fine — it redirects. Keeping a thin post.json avoids manufacturing
     permanent false positives in the link audit. The `redirect_to`
     marker is what stops topic_dedup.py's corpus loader from
     re-matching the stub against its own keeper and "retiring" it a
     second time on the next run (see the skip check added to
     adsense_fixes/topic_dedup.py's _load_corpus()).
  6. The stub's `created_at` is copied from the ORIGINAL post, not set
     to today. velocity_controller.py derives its daily publish count
     live from every post.json's created_at — stamping today's date on
     an administrative merge would look like a real new publish and
     silently eat into that day's actual publishing quota.
  7. Without --confirm, this only prints what it would do. No files are
     touched. Exit code is always 0 in dry-run mode.

USAGE
-----
    # Dry run — see what would happen, nothing is written:
    python scripts/auto_retire_duplicates.py --docs-dir ./docs

    # Actually retire, bounded:
    python scripts/auto_retire_duplicates.py \\
        --docs-dir ./docs \\
        --max-deletions 2 \\
        --min-score 0.50 \\
        --backup-dir ./.duplicate_backups \\
        --base-url "https://kubaik.github.io" \\
        --confirm

To undo a retirement: copy the backed-up directory from --backup-dir
back into docs/, overwriting the stub. Nothing else needs to change —
the moment its post.json no longer has `redirect_to`, every other
script (topic_dedup, link_validator, the sitemap generator) treats it
as a normal live post again on their next read.
"""

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adsense_fixes.topic_dedup import find_clusters  # noqa: E402


_REDIRECT_STUB_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - Kubai Kevin</title>
    <meta name="robots" content="noindex,follow">
    <link rel="canonical" href="{keeper_url}">
    <meta http-equiv="refresh" content="0; url={keeper_url}">
    <meta name="description" content="This article has been merged into a more complete post on the same topic.">
</head>
<body>
    <p>This post has been merged into a more complete article:
       <a href="{keeper_url}">{keeper_title}</a>.
       If you are not redirected automatically,
       <a href="{keeper_url}">click here</a>.</p>
</body>
</html>
"""


def _backup_post(post_dir: Path, backup_dir: Path) -> Path:
    dest = backup_dir / post_dir.name
    if dest.exists():
        # Don't clobber an earlier backup of the same slug from a prior run —
        # append a timestamp instead, so nothing is ever silently overwritten.
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dest = backup_dir / f"{post_dir.name}__{stamp}"
    shutil.copytree(post_dir, dest)
    return dest


def _stub_post_json(original: dict, retired_slug: str, keeper_slug: str, keeper_title: str) -> dict:
    return {
        "slug": retired_slug,
        "title": original.get("title", retired_slug),
        "created_at": original.get("created_at", ""),  # preserved — see docstring point 6
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "content": (
            f"This article has been merged into "
            f"[{keeper_title}](/{keeper_slug}/) to remove duplicate-topic "
            f"content and consolidate ranking signal for this subject."
        ),
        "meta_description": f"Merged into {keeper_title}.",
        "tags": [],
        "seo_keywords": [],
        "redirect_to": keeper_slug,
        "noindex": True,
    }


def _retire_post(
    post_dir: Path,
    keeper_slug: str,
    keeper_title: str,
    base_url: str,
    backup_dir: Path,
) -> None:
    backup_path = _backup_post(post_dir, backup_dir)

    with open(post_dir / "post.json", "r", encoding="utf-8") as f:
        original = json.loads(f.read())
    title = original.get("title", post_dir.name)

    keeper_url = f"{base_url.rstrip('/')}/{keeper_slug}/"
    stub_html = _REDIRECT_STUB_TEMPLATE.format(
        title=title,
        keeper_url=keeper_url,
        keeper_title=keeper_title,
    )

    # Replace the whole directory's contents rather than editing in place,
    # so no stray files from the original post (e.g. a per-post asset)
    # survive the merge.
    shutil.rmtree(post_dir)
    post_dir.mkdir(parents=True, exist_ok=True)

    (post_dir / "index.html").write_text(stub_html, encoding="utf-8")
    (post_dir / "post.json").write_text(
        json.dumps(
            _stub_post_json(original, post_dir.name, keeper_slug, keeper_title),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"  🗄️  Backed up to {backup_path}")
    print(f"  ↪️  {post_dir.name} → redirect stub → /{keeper_slug}/")


def run(
    docs_dir: Path,
    max_deletions: int,
    min_score: float,
    backup_dir: Path,
    base_url: str,
    confirm: bool,
    report_path: Path,
) -> int:
    clusters = find_clusters(docs_dir)

    candidates: List[Dict] = []
    for cluster in clusters:
        keep = cluster[0]
        for member in cluster[1:]:
            if member.get("score_vs_keep", 0.0) >= min_score:
                candidates.append({
                    "slug": member["slug"],
                    "title": member["title"],
                    "words": member["words"],
                    "score": round(member["score_vs_keep"], 4),
                    "keeper_slug": keep["slug"],
                    "keeper_title": keep["title"],
                })

    if not candidates:
        print(f"No duplicate posts scoring >= {min_score:.0%} against their cluster's keeper. Nothing to do.")
        Path(report_path).write_text(json.dumps({
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "dry_run": not confirm, "min_score": min_score, "max_deletions": max_deletions,
            "total_candidates_found": 0, "acted": [], "deferred": [],
        }, indent=2), encoding="utf-8")
        return 0

    # Highest-confidence matches first, so if max_deletions truncates the
    # list, what gets acted on this run is the strongest evidence available.
    candidates.sort(key=lambda c: -c["score"])
    acted_candidates, deferred_candidates = candidates[:max_deletions], candidates[max_deletions:]

    print(f"{len(candidates)} duplicate post(s) qualify (score >= {min_score:.0%}).")
    print(f"{'Retiring' if confirm else 'Would retire'} {len(acted_candidates)} this run "
          f"(--max-deletions {max_deletions}):\n")

    for c in acted_candidates:
        print(f"  {c['score']:.0%}  {c['slug']}  →  {c['keeper_slug']}  | {c['title']}")
    if deferred_candidates:
        print(f"\n{len(deferred_candidates)} more qualify but are deferred to a future run "
              f"(--max-deletions cap): {', '.join(d['slug'] for d in deferred_candidates)}")

    acted: List[Dict] = []

    if not confirm:
        print("\nDry run — nothing was written. Re-run with --confirm to apply.")
    else:
        backup_dir.mkdir(parents=True, exist_ok=True)
        print()
        for c in acted_candidates:
            post_dir = docs_dir / c["slug"]
            if not post_dir.exists() or not (post_dir / "post.json").exists():
                print(f"  ⚠️  Skipping {c['slug']}: already retired or missing (likely by an earlier run).")
                continue
            existing = json.loads((post_dir / "post.json").read_text(encoding="utf-8"))
            if existing.get("redirect_to"):
                print(f"  ⏭️  Skipping {c['slug']}: already a redirect stub.")
                continue
            _retire_post(
                post_dir=post_dir,
                keeper_slug=c["keeper_slug"],
                keeper_title=c["keeper_title"],
                base_url=base_url,
                backup_dir=backup_dir,
            )
            acted.append(c)

        print(f"\n✅ Retired {len(acted)} post(s). Backups at {backup_dir}/")
        print("   Run 'python blog_system.py build' to regenerate index/tag pages "
              "before the sitemap and OG steps.")

    Path(report_path).write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dry_run": not confirm,
        "min_score": min_score,
        "max_deletions": max_deletions,
        "total_candidates_found": len(candidates),
        "acted": acted,
        "deferred": [c["slug"] for c in deferred_candidates],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Report written to {report_path}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bounded, automated retirement of confirmed duplicate-topic posts."
    )
    parser.add_argument("--docs-dir", default="./docs")
    parser.add_argument("--max-deletions", type=int, default=2)
    parser.add_argument("--min-score", type=float, default=0.50)
    parser.add_argument("--backup-dir", default="./.duplicate_backups")
    parser.add_argument("--base-url", default="https://kubaik.github.io")
    parser.add_argument("--confirm", action="store_true", help="Actually write changes. Omit for a dry run.")
    parser.add_argument("--report", default="./duplicate_retirement_report.json")
    args = parser.parse_args()

    sys.exit(run(
        docs_dir=Path(args.docs_dir),
        max_deletions=args.max_deletions,
        min_score=args.min_score,
        backup_dir=Path(args.backup_dir),
        base_url=args.base_url,
        confirm=args.confirm,
        report_path=Path(args.report),
    ))