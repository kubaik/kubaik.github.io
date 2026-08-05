#!/usr/bin/env python3
"""
corpus_audit_and_remediate.py

Full-corpus duplicate/thin-content audit and remediation for the
kubaik.github.io blog (docs/<slug>/post.json + index.html).

WHY THIS EXISTS
----------------
utils/delete_similar_posts.py and cleanup_plan.json already do most of this
work, but cleanup_plan.json is a stale, one-time snapshot (563 of 795 posts
audited) and the CI job that runs similarity checks is configured
report-only, so approved actions never actually get applied. This script:

  1. Re-scans the ENTIRE current corpus (all docs/*/post.json), not a cached
     subset, using the same lightweight TF-IDF approach as
     utils/delete_similar_posts.py (title x4, meta_description x2, tags,
     keywords -- content body deliberately excluded, matching the existing
     tool's rationale that body text skews similarity toward shared
     boilerplate rather than actual topic overlap).
  2. Merges results with any pending actions already sitting in
     cleanup_plan.json (if present) so nothing already decided gets lost.
  3. Writes an updated, reviewable action plan: cleanup_plan_v2.json.
  4. With --apply, executes MERGE_REDIRECT and NOINDEX_ONLY actions using
     the EXACT redirect-stub format already live in this repo (verified
     against docs/11-patterns-for-eventual-consistency-that-dont-wake/
     index.html): noindex + 0s meta-refresh + canonical to the survivor URL.
     This keeps output consistent with pages already deployed.
  5. Separately flags (report-only, never auto-deletes) any *unexpectedly*
     thin index.html -- i.e. under WORD_COUNT_FLOOR words -- that is NOT
     already a redirect stub and NOT a known short policy page. This is a
     safety net for future generation runs that might emit broken/empty
     pages, not a routine action.

This script never deletes a post directory. Deletion of content that may
have inbound links/backlinks is a business decision, not something to
automate blindly -- MERGE_REDIRECT preserves the URL (as a redirect) and
folds link equity into the survivor, which is almost always the right
call for a monetized site. Use --hard-delete-empty-only for the rare case
of a genuinely broken/empty stub that isn't a valid redirect (see below).

USAGE
-----
    # Safe default: scan everything, write cleanup_plan_v2.json, change nothing
    python corpus_audit_and_remediate.py

    # Same, but more/less aggressive duplicate threshold
    python corpus_audit_and_remediate.py --threshold 0.80

    # Actually write redirect stubs / inject noindex for approved actions
    python corpus_audit_and_remediate.py --apply

    # Only apply actions that were already in cleanup_plan.json
    # (skip the fresh full-corpus rescan)
    python corpus_audit_and_remediate.py --apply --no-rescan

Requires: numpy, scikit-learn (already in requirements.txt for this repo).
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Fixed: Navigate up one directory to reach the project root
REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
BASE_URL = "https://kubaik.github.io"  # override with --base-url if needed

DEFAULT_THRESHOLD = 0.75
WORD_COUNT_FLOOR = 300  # below this AND not a known stub/policy page => flag

# Directories that are pages, not blog posts -- never touched by this script.
NON_POST_DIRS = {
    "static", "tag", "about", "contact", "privacy-policy", "terms-of-service",
    "author", "dmca", "ai-content-policy", "vc-insights", "your-data",
}

REDIRECT_STUB_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<link rel="canonical" href="{survivor_url}">
<meta http-equiv="refresh" content="0; url={survivor_url}">
<meta name="robots" content="noindex">
</head>
<body>
<p>This article has been merged into a more complete version:
<a href="{survivor_url}">{survivor_title}</a></p>
</body>
</html>
"""


# ── Text / corpus helpers (mirrors utils/delete_similar_posts.py) ──────────

def clean(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def post_to_text(data: dict) -> str:
    title = clean(data.get("title", ""))
    description = clean(data.get("meta_description", ""))
    tags = clean(" ".join(data.get("tags", []) or []))
    keywords = clean(" ".join(data.get("seo_keywords", []) or []))
    return f"{title} {title} {title} {title} {description} {description} {tags} {keywords}"


def load_posts(docs_dir: Path) -> list:
    posts = []
    if not docs_dir.exists():
        print(f"ERROR: {docs_dir} does not exist.", file=sys.stderr)
        return posts
    for post_dir in sorted(docs_dir.iterdir()):
        if not post_dir.is_dir() or post_dir.name in NON_POST_DIRS:
            continue
        post_json = post_dir / "post.json"
        if not post_json.exists():
            continue
        try:
            data = json.loads(post_json.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"  WARN: skipping {post_dir.name}: {e}", file=sys.stderr)
            continue
        data["_slug"] = post_dir.name
        data["_dir"] = post_dir
        data["_text"] = post_to_text(data)
        data["_content_len"] = len(data.get("content", "") or "")
        posts.append(data)
    return posts


def build_similarity_matrix(posts: list) -> np.ndarray:
    texts = [p["_text"] for p in posts]
    vectorizer = TfidfVectorizer(min_df=1, stop_words="english")
    tfidf = vectorizer.fit_transform(texts)
    return cosine_similarity(tfidf)


def find_pairs(posts: list, sim_matrix: np.ndarray, threshold: float) -> list:
    pairs = []
    n = len(posts)
    for i, j in combinations(range(n), 2):
        score = float(sim_matrix[i, j])
        if score >= threshold:
            pairs.append((i, j, score))
    return sorted(pairs, key=lambda t: -t[2])


def union_find_groups(n: int, pairs: list) -> list:
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, j, _ in pairs:
        union(i, j)

    groups = {}
    for idx in range(n):
        groups.setdefault(find(idx), []).append(idx)
    return [g for g in groups.values() if len(g) > 1]


def pick_survivor(group: list, posts: list) -> int:
    # Keep the post with the most content; tie-break by earliest created_at.
    def sort_key(idx):
        p = posts[idx]
        return (-p["_content_len"], p.get("created_at", ""))
    return sorted(group, key=sort_key)[0]


# ── Action plan construction ────────────────────────────────────────────────

def build_action_plan(posts: list, threshold: float) -> dict:
    if len(posts) < 2:
        return {"summary": {"total_posts": len(posts)}, "actions": []}

    sim_matrix = build_similarity_matrix(posts)
    pairs = find_pairs(posts, sim_matrix, threshold)
    groups = union_find_groups(len(posts), pairs)

    actions = []
    for group in groups:
        survivor_idx = pick_survivor(group, posts)
        survivor = posts[survivor_idx]
        for idx in group:
            if idx == survivor_idx:
                continue
            loser = posts[idx]
            score = float(sim_matrix[survivor_idx, idx])
            # High similarity -> redirect the loser into the survivor.
            # Borderline similarity -> just noindex it (don't destroy a
            # possibly-distinct page's indexability by redirecting it).
            action = "MERGE_REDIRECT" if score >= (
                threshold + 0.05) else "NOINDEX_ONLY"
            actions.append({
                "slug": loser["_slug"],
                "action": action,
                "similarity": round(score, 3),
                "survivor_slug": survivor["_slug"],
                "survivor_title": survivor.get("title", survivor["_slug"]),
                "loser_title": loser.get("title", loser["_slug"]),
                "cluster_size": len(group),
            })

    summary = {
        "total_posts": len(posts),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "threshold": threshold,
        "merge_redirect": sum(1 for a in actions if a["action"] == "MERGE_REDIRECT"),
        "noindex_only": sum(1 for a in actions if a["action"] == "NOINDEX_ONLY"),
        "unaffected": len(posts) - len({a["slug"] for a in actions}),
    }
    return {"summary": summary, "actions": actions}


def merge_with_existing_plan(new_plan: dict, existing_path: Path) -> dict:
    """Union in any actions from an existing cleanup_plan.json that the
    fresh rescan didn't happen to reproduce (e.g. threshold differences),
    so nothing previously approved silently disappears."""
    if not existing_path.exists():
        return new_plan
    try:
        existing = json.loads(existing_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return new_plan

    seen_slugs = {a["slug"] for a in new_plan["actions"]}
    added = 0
    for a in existing.get("actions", []):
        if a["slug"] not in seen_slugs:
            new_plan["actions"].append(a)
            seen_slugs.add(a["slug"])
            added += 1
    if added:
        new_plan["summary"]["carried_over_from_existing_plan"] = added
    return new_plan


# ── Thin-content safety net (report only) ───────────────────────────────────

def flag_thin_pages(docs_dir: Path, word_floor: int) -> list:
    flagged = []
    for post_dir in sorted(docs_dir.iterdir()):
        if not post_dir.is_dir() or post_dir.name in NON_POST_DIRS:
            continue
        idx = post_dir / "index.html"
        if not idx.exists():
            continue
        html = idx.read_text(encoding="utf-8", errors="ignore")
        if "has been merged into a more complete version" in html:
            continue  # legitimate, already-applied redirect stub
        text = re.sub(r"<script.*?</script>", " ", html, flags=re.S)
        text = re.sub(r"<style.*?</style>", " ", text, flags=re.S)
        text = re.sub(r"<[^>]+>", " ", text)
        word_count = len(text.split())
        if word_count < word_floor:
            flagged.append({"slug": post_dir.name, "word_count": word_count})
    return flagged


# ── Apply actions ────────────────────────────────────────────────────────

def apply_actions(plan: dict, docs_dir: Path, base_url: str, dry_run: bool) -> None:
    for a in plan["actions"]:
        post_dir = docs_dir / a["slug"]
        idx = post_dir / "index.html"
        if not post_dir.exists():
            print(
                f"  SKIP {a['slug']}: directory not found (already removed?)")
            continue
        if not idx.exists():
            print(f"  SKIP {a['slug']}: index.html not found")
            continue

        current = idx.read_text(encoding="utf-8", errors="ignore")
        if "has been merged into a more complete version" in current:
            print(f"  SKIP {a['slug']}: already a redirect stub")
            continue

        if a["action"] == "MERGE_REDIRECT":
            survivor_url = f"{base_url}/{a['survivor_slug']}/"
            stub = REDIRECT_STUB_TEMPLATE.format(
                title=a["loser_title"],
                survivor_url=survivor_url,
                survivor_title=a["survivor_title"],
            )
            print(
                f"  {'[DRY RUN] ' if dry_run else ''}MERGE_REDIRECT {a['slug']} -> {a['survivor_slug']}")
            if not dry_run:
                idx.write_text(stub, encoding="utf-8")

        elif a["action"] == "NOINDEX_ONLY":
            print(f"  {'[DRY RUN] ' if dry_run else ''}NOINDEX_ONLY {a['slug']}")
            if not dry_run:
                if 'name="robots"' in current:
                    updated = re.sub(
                        r'<meta name="robots" content="[^"]*">',
                        '<meta name="robots" content="noindex, follow">',
                        current,
                    )
                else:
                    updated = current.replace(
                        "</head>",
                        '<meta name="robots" content="noindex, follow">\n</head>',
                        1,
                    )
                idx.write_text(updated, encoding="utf-8")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs-dir", default=str(DOCS_DIR),
                    help="Path to docs/ directory")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help="Cosine similarity threshold for near-duplicate grouping (default 0.75)")
    ap.add_argument("--base-url", default=BASE_URL,
                    help="Site base URL for redirect targets")
    ap.add_argument("--word-floor", type=int, default=WORD_COUNT_FLOOR,
                    help="Flag pages with fewer words than this that aren't known stubs (report-only)")
    ap.add_argument("--no-rescan", action="store_true",
                    help="Skip the fresh full-corpus TF-IDF scan; only apply what's already in cleanup_plan.json")
    ap.add_argument("--apply", action="store_true",
                    help="Actually write redirect stubs / inject noindex. Default is dry-run.")
    ap.add_argument("--out", default="cleanup_plan_v2.json",
                    help="Where to write the action plan")
    args = ap.parse_args()

    docs_dir = Path(args.docs_dir)
    existing_plan_path = REPO_ROOT / "cleanup_plan.json"

    if args.no_rescan:
        plan = json.loads(existing_plan_path.read_text(
            encoding="utf-8")) if existing_plan_path.exists() else {"summary": {}, "actions": []}
    else:
        print(f"Loading posts from {docs_dir} ...")
        posts = load_posts(docs_dir)
        print(f"Loaded {len(posts)} posts (full corpus, not a cached subset).")
        print(
            f"Scanning for near-duplicates at threshold {args.threshold} ...")
        plan = build_action_plan(posts, args.threshold)
        plan = merge_with_existing_plan(plan, existing_plan_path)

    thin = flag_thin_pages(docs_dir, args.word_floor)
    plan["thin_content_flagged_for_manual_review"] = thin

    out_path = REPO_ROOT / args.out
    out_path.write_text(json.dumps(
        plan, indent=2, default=str), encoding="utf-8")

    print("\n--- Summary ---")
    print(json.dumps(plan.get("summary", {}), indent=2))
    print(f"Pending actions: {len(plan['actions'])}")
    print(f"Thin-content flags (report only, review manually): {len(thin)}")
    print(f"Plan written to: {out_path}")

    if args.apply:
        print(f"\nApplying {len(plan['actions'])} actions ...")
        apply_actions(plan, docs_dir, args.base_url, dry_run=False)
        print("\nDone. Re-run scripts/generate_sitemap.py afterward so redirected")
        print("slugs are dropped from sitemap.xml, then commit the changes.")
    else:
        print("\nDry run only -- no files changed. Re-run with --apply to execute.")
        apply_actions(plan, docs_dir, args.base_url, dry_run=True)


if __name__ == "__main__":
    main()
