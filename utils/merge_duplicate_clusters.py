#!/usr/bin/env python3
"""
merge_duplicate_clusters.py

Merge/redirect pass for near-duplicate post CLUSTERS in the kubaik.github.io
AI-blog repo. Companion to content_quality_scanner.py, and built directly on
top of it: it imports content_quality_scanner's own post-loading and
TF-IDF/cosine-similarity code, so a post is only ever considered a
"duplicate" here using the exact same methodology already reflected in
quality_report.csv.

WHY THIS EXISTS (vs. the existing utils/delete_similar_posts.py)
------------------------------------------------------------------
utils/delete_similar_posts.py and utils/deduplicate_posts.py both already
detect near-duplicates and DELETE the losers outright. For a monetized,
search-indexed site that's the wrong move:

  - Any of those posts that Google has already crawled/indexed become 404s
    -> lost equity, worse crawl-budget signals, broken backlinks.
  - Nothing rewrites the internal links (homepage, tag pages, other posts'
    body text, related-posts blocks) that point AT the deleted slug -> the
    survivor's internal link graph is left full of dead links.
  - No audit trail: once shutil.rmtree() runs, the only record is a git diff.

This script does a MERGE instead of a delete:
  1. Groups posts into clusters using content_quality_scanner's TF-IDF
     cosine similarity over the FULL corpus (title x3 + content), not just
     each post's single nearest neighbor -- the printed quality_report.csv
     only stores one nearest-neighbor per post, which is why it shows the
     pgvector/pinecone problem as several separate pairs instead of the one
     7-post cluster it actually is. This script reconstructs the real
     clusters with union-find over the full pairwise matrix.
  2. Picks ONE canonical post per cluster (highest composite quality score
     from content_quality_scanner's own scoring -- rewards depth, meta
     quality, structure, low affiliate density; tie-break: earliest
     created_at, since an older URL is more likely to already carry
     backlinks/index history worth preserving).
  3. For every non-canonical post in the cluster:
       - backs up the full post directory (index.html, post.json, images)
         to .duplicate_merge_backups/<timestamp>/<slug>/
       - replaces docs/<slug>/index.html with a lightweight redirect stub:
         <link rel="canonical"> + <meta http-equiv="refresh"> pointing at
         the canonical post. This is the standard fallback for a "301" on
         GitHub Pages, which has no server-side redirect support.
       - removes post.json (content_quality_scanner.load_posts() already
         treats a directory with no post.json as "already merged" and
         skips it, so future scanner runs won't re-flag it)
  4. Rewrites internal links repo-wide: every docs/**/index.html that links
     to a merged slug gets that href swapped to point at the canonical post
     directly, so users and crawlers don't have to follow a redirect hop.
  5. Strips the merged slugs' <url> blocks out of docs/sitemap.xml (a full
     `python blog_system.py build` will regenerate this cleanly anyway --
     this just keeps things correct if you don't rebuild immediately).
  6. Writes an append-only merge log (.duplicate_merge_log.json) and a
     human-readable CSV (merge_plan.csv) recording exactly what happened.

IT NEVER TOUCHES ANYTHING BY DEFAULT.
--------------------------------------
- Default mode is a dry run: prints the cluster plan, writes merge_plan.csv,
  changes nothing on disk.
- Actually applying requires BOTH --apply AND --confirm.
- Every removed post is backed up before its directory is touched, so it's
  always reversible (git history is a second safety net on top of that).

USAGE
-----
    # Dry run: see the clusters and merge plan, write merge_plan.csv
    python utils/merge_duplicate_clusters.py

    # Same, but only look within one calendar month
    python utils/merge_duplicate_clusters.py --month 2026-06

    # Stricter/looser clustering (default 0.55; the original report showed
    # confirmed duplicate pairs from 0.59 up to 0.84)
    python utils/merge_duplicate_clusters.py --threshold 0.6

    # Force-keep a specific slug as canonical even if its score is lower
    python utils/merge_duplicate_clusters.py --keep pgvector-at-100k-qps-what-broke-first

    # Never touch a specific slug at all (excluded from clustering entirely)
    python utils/merge_duplicate_clusters.py --exclude some-slug-to-leave-alone

    # Actually perform the merge (backs up + redirects + rewrites links)
    python utils/merge_duplicate_clusters.py --apply --confirm

    # After applying, regenerate the site so the homepage, tag pages,
    # sitemap, RSS, and related-posts links no longer reference merged
    # slugs at all:
    python blog_system.py build

Lives in utils/ and resolves docs/, backups, etc. relative to the repo root
(parent of utils/), the same convention as content_quality_scanner.py,
deduplicate_posts.py, and delete_similar_posts.py. Works whether you run it
as `python utils/merge_duplicate_clusters.py` from the repo root or
`python merge_duplicate_clusters.py` from inside utils/.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

# Make content_quality_scanner importable regardless of cwd, and reuse its
# post-loading / scoring / TF-IDF code directly rather than re-implementing
# a second, possibly-inconsistent version of the same logic.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import content_quality_scanner as cqs  # noqa: E402

REPO_ROOT = cqs.REPO_ROOT
DOCS_DIR = cqs.DOCS_DIR
SITEMAP_PATH = DOCS_DIR / "sitemap.xml"
BACKUP_ROOT = REPO_ROOT / ".duplicate_merge_backups"
MERGE_LOG_PATH = REPO_ROOT / ".duplicate_merge_log.json"
DEFAULT_MERGE_PLAN_CSV = REPO_ROOT / "merge_plan.csv"

# Same default as the confirmed-duplicate pairs already surfaced in
# quality_report.csv (0.59 - 0.84), set conservatively low enough to catch
# paraphrased duplicates without over-merging distinct-but-related posts.
DEFAULT_THRESHOLD = 0.55


# --------------------------------------------------------------------------
# Clustering (union-find over the FULL pairwise similarity matrix)
# --------------------------------------------------------------------------

def build_full_similarity_matrix(posts: List["cqs.Post"]) -> Dict[str, Dict[str, float]]:
    """
    {slug: {other_slug: similarity}} for every pair, using
    content_quality_scanner's own TF-IDF vectors and cosine function so
    clustering matches the report exactly. O(n^2) cosine comparisons --
    fine at this corpus size (matches the "174936 pairs" the scanner
    already computes for a single nearest-neighbor pass).
    """
    tfidf = cqs._build_tfidf(posts)
    slugs = [p.slug for p in posts]
    n = len(slugs)
    print(f"Computing full pairwise similarity across {n} posts "
          f"({n * (n - 1) // 2} pairs)... this may take a moment.")

    matrix: Dict[str, Dict[str, float]] = {s: {} for s in slugs}
    for i in range(n):
        si = slugs[i]
        vi = tfidf[si]
        for j in range(i + 1, n):
            sj = slugs[j]
            sim = cqs._cosine(vi, tfidf[sj])
            if sim > 0:
                matrix[si][sj] = sim
                matrix[sj][si] = sim
    return matrix


class UnionFind:
    def __init__(self, items: List[str]):
        self.parent = {item: item for item in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: str, y: str) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[rx] = ry


def find_clusters(
    posts: List["cqs.Post"],
    matrix: Dict[str, Dict[str, float]],
    threshold: float,
    exclude: Set[str],
) -> List[List["cqs.Post"]]:
    """Union-find any two posts with similarity >= threshold. Returns only
    clusters with 2+ members; excluded slugs are never merged into anything
    (they're dropped from consideration entirely, not just protected as
    canonical)."""
    eligible = [p for p in posts if p.slug not in exclude]
    slugs = [p.slug for p in eligible]
    uf = UnionFind(slugs)

    for si in slugs:
        for sj, sim in matrix.get(si, {}).items():
            if sj in uf.parent and sim >= threshold:
                uf.union(si, sj)

    by_slug = {p.slug: p for p in eligible}
    groups: Dict[str, List["cqs.Post"]] = {}
    for s in slugs:
        root = uf.find(s)
        groups.setdefault(root, []).append(by_slug[s])

    return [g for g in groups.values() if len(g) > 1]


def cluster_max_similarity(cluster: List["cqs.Post"], matrix: Dict[str, Dict[str, float]]) -> float:
    best = 0.0
    for i, pi in enumerate(cluster):
        for pj in cluster[i + 1:]:
            best = max(best, matrix.get(pi.slug, {}).get(pj.slug, 0.0))
    return best


# --------------------------------------------------------------------------
# Canonical selection
# --------------------------------------------------------------------------

def pick_canonical(
    cluster: List["cqs.Post"],
    keep_longest: bool,
    forced_keep: Set[str],
) -> "cqs.Post":
    forced = [p for p in cluster if p.slug in forced_keep]
    if forced:
        # If multiple forced-keeps land in the same cluster (user error),
        # still need a single canonical -- prefer the highest score among them.
        return max(forced, key=lambda p: p.score)

    if keep_longest:
        return max(cluster, key=lambda p: p.word_count)

    def sort_key(p: "cqs.Post"):
        # Primary: highest composite quality score.
        # Tie-break: earliest created_at (older URL = more likely to carry
        # existing backlinks/index history worth preserving). Posts with an
        # unparseable date are pushed to the back of the tie-break so a
        # dated post is always preferred as canonical over an undated one.
        ts = p.created_at.timestamp() if p.created_at else float("inf")
        return (p.score, -ts)

    return max(cluster, key=sort_key)


# --------------------------------------------------------------------------
# Redirect stub generation
# --------------------------------------------------------------------------

REDIRECT_STUB_TMPL = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} — moved</title>
    <link rel="canonical" href="{canonical_url}">
    <meta http-equiv="refresh" content="0; url={canonical_url}">
    <meta name="robots" content="index, follow">
</head>
<body>
    <p>This article has been merged into a more complete, up-to-date post.
    If you are not redirected automatically,
    <a href="{canonical_url}">continue to {canonical_title}</a>.</p>
</body>
</html>
"""


def write_redirect_stub(post: "cqs.Post", canonical: "cqs.Post", base_url: str) -> None:
    canonical_url = f"{base_url}/{canonical.slug}/"
    stub_html = REDIRECT_STUB_TMPL.format(
        title=cqs._html_escape(post.title) if hasattr(
            cqs, "_html_escape") else post.title,
        canonical_url=canonical_url,
        canonical_title=canonical.title,
    )
    (post.path / "index.html").write_text(stub_html, encoding="utf-8")


# --------------------------------------------------------------------------
# Internal link rewriting (repo-wide)
# --------------------------------------------------------------------------

def rewrite_internal_links(slug_map: Dict[str, str], docs_dir: Path) -> int:
    """
    slug_map: {old_slug: new_slug} for every merged post.
    Rewrites href="/<old_slug>/" and href="{base_url}/<old_slug>/" (with or
    without a trailing slash) to point at the canonical slug, across every
    index.html under docs/. Returns the number of files changed.
    """
    if not slug_map:
        return 0

    # Matches href="/old-slug" or href="/old-slug/" or an absolute URL
    # variant, but only when old-slug is the WHOLE path segment (anchored
    # with a required trailing "/" or end-quote) so e.g. "pgvector-cut"
    # never accidentally matches inside "pgvector-cut-costs-80...".
    patterns = {
        old: re.compile(
            r'(href="(?:https?://[^"]+)?/' + re.escape(old) + r')(/?)(")'
        )
        for old in slug_map
    }

    changed = 0
    for html_file in docs_dir.rglob("index.html"):
        try:
            text = html_file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        original = text
        for old, new in slug_map.items():
            text = patterns[old].sub(
                lambda m, new=new: f'href="/{new}/"', text
            )
        if text != original:
            html_file.write_text(text, encoding="utf-8")
            changed += 1
    return changed


def strip_from_sitemap(slugs: Set[str], sitemap_path: Path, base_url: str) -> int:
    """Remove <url>...</url> blocks whose <loc> is one of the merged slugs.
    A full `blog_system.py build` regenerates the sitemap anyway; this just
    keeps it correct in the meantime."""
    if not slugs or not sitemap_path.exists():
        return 0
    text = sitemap_path.read_text(encoding="utf-8")
    removed = 0
    for slug in slugs:
        loc = f"{base_url}/{slug}/"
        block_re = re.compile(
            r"[ \t]*<url>\s*<loc>" + re.escape(loc) + r"</loc>.*?</url>\s*",
            re.DOTALL,
        )
        text, n = block_re.subn("", text)
        removed += n
    if removed:
        sitemap_path.write_text(text, encoding="utf-8")
    return removed


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def write_merge_plan_csv(clusters_plan: List[dict], path: Path) -> None:
    rows = []
    for cluster in clusters_plan:
        rows.append({
            "cluster_id": cluster["cluster_id"],
            "role": "canonical",
            "slug": cluster["canonical"].slug,
            "title": cluster["canonical"].title,
            "score": cluster["canonical"].score,
            "word_count": cluster["canonical"].word_count,
            "created_at": cluster["canonical"].created_at.isoformat() if cluster["canonical"].created_at else "",
            "max_similarity_in_cluster": round(cluster["max_similarity"], 3),
        })
        for p in cluster["redirects"]:
            rows.append({
                "cluster_id": cluster["cluster_id"],
                "role": "redirect -> " + cluster["canonical"].slug,
                "slug": p.slug,
                "title": p.title,
                "score": p.score,
                "word_count": p.word_count,
                "created_at": p.created_at.isoformat() if p.created_at else "",
                "max_similarity_in_cluster": round(cluster["max_similarity"], 3),
            })

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "cluster_id", "role", "slug", "title", "score", "word_count",
            "created_at", "max_similarity_in_cluster",
        ])
        writer.writeheader()
        writer.writerows(rows)


def print_plan(clusters_plan: List[dict]) -> None:
    total_redirects = sum(len(c["redirects"]) for c in clusters_plan)
    print(f"\n{'=' * 78}")
    print(f"  Found {len(clusters_plan)} duplicate cluster(s), "
          f"{total_redirects} post(s) would be merged/redirected")
    print(f"{'=' * 78}")

    for cluster in clusters_plan:
        canonical = cluster["canonical"]
        print(f"\nCluster {cluster['cluster_id']}  "
              f"(max intra-cluster similarity: {cluster['max_similarity']:.2f})")
        print(f"  \u2713 KEEP (canonical)  score={canonical.score:>5.1f}  "
              f"words={canonical.word_count:>5}  {canonical.slug}")
        for p in cluster["redirects"]:
            print(f"    \u2192 redirect          score={p.score:>5.1f}  "
                  f"words={p.word_count:>5}  {p.slug}")


# --------------------------------------------------------------------------
# Apply
# --------------------------------------------------------------------------

def apply_merges(clusters_plan: List[dict], docs_dir: Path, base_url: str) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = BACKUP_ROOT / timestamp
    backup_dir.mkdir(parents=True, exist_ok=True)

    slug_map: Dict[str, str] = {}
    merged_slugs: Set[str] = set()
    log_entries = []

    for cluster in clusters_plan:
        canonical = cluster["canonical"]
        for p in cluster["redirects"]:
            if not p.path.exists():
                print(f"  SKIP (already gone): {p.slug}")
                continue

            dest = backup_dir / p.slug
            shutil.copytree(p.path, dest)

            write_redirect_stub(p, canonical, base_url)

            post_json = p.path / "post.json"
            if post_json.exists():
                post_json.unlink()
            index_md = p.path / "index.md"
            if index_md.exists():
                index_md.unlink()

            slug_map[p.slug] = canonical.slug
            merged_slugs.add(p.slug)

            log_entries.append({
                "timestamp": timestamp,
                "cluster_id": cluster["cluster_id"],
                "merged_slug": p.slug,
                "canonical_slug": canonical.slug,
                "similarity_to_cluster": round(cluster["max_similarity"], 3),
                "backup_path": str(dest.relative_to(REPO_ROOT)),
            })

            print(f"  Merged: {p.slug}  -> {canonical.slug}  "
                  f"(backed up to {dest.relative_to(REPO_ROOT)})")

    if not slug_map:
        print("\nNothing to apply.")
        return

    print(f"\nRewriting internal links across {docs_dir}/ ...")
    changed = rewrite_internal_links(slug_map, docs_dir)
    print(f"  Updated links in {changed} file(s).")

    print("Stripping merged slugs from sitemap.xml ...")
    removed = strip_from_sitemap(merged_slugs, SITEMAP_PATH, base_url)
    print(f"  Removed {removed} <url> block(s) from {SITEMAP_PATH.name}.")

    existing_log = []
    if MERGE_LOG_PATH.exists():
        try:
            existing_log = json.loads(
                MERGE_LOG_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing_log = []
    existing_log.extend(log_entries)
    MERGE_LOG_PATH.write_text(json.dumps(
        existing_log, indent=2), encoding="utf-8")

    print(f"\nDone. {len(slug_map)} post(s) merged into "
          f"{len({c['canonical'].slug for c in clusters_plan})} canonical post(s).")
    print(f"Backups saved under {backup_dir.relative_to(REPO_ROOT)}/ "
          f"-- restore by copying a folder back into {docs_dir.relative_to(REPO_ROOT)}/.")
    print(f"Merge log appended to {MERGE_LOG_PATH.relative_to(REPO_ROOT)}.")
    print("\nNEXT STEP: rebuild the site so the homepage, tag pages, RSS feed, "
          "and related-posts\nlinks fully drop the merged slugs (this script's "
          "sitemap/link edits cover the\nstatic files that already exist on disk, "
          "but a rebuild is still the clean way\nto regenerate everything from "
          "the now-smaller post set):")
    print("    python blog_system.py build")
    print("Then commit the change (backups, redirect stubs, rewritten links, "
          "updated sitemap).")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Cosine similarity threshold for clustering (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--month", type=str, default=None,
                        help="Only cluster within one calendar month, e.g. 2026-06")
    parser.add_argument("--keep", action="append", default=[],
                        help="Slug to force as canonical if it appears in a cluster. Repeatable.")
    parser.add_argument("--exclude", action="append", default=[],
                        help="Slug to leave out of clustering entirely (never merged, never a canonical). Repeatable.")
    parser.add_argument("--keep-longest", action="store_true",
                        help="Pick canonical by word count instead of composite quality score")
    parser.add_argument("--min-cluster-size", type=int, default=2,
                        help="Minimum posts in a group to treat it as a cluster (default: 2)")
    parser.add_argument("--apply", action="store_true",
                        help="Perform the merge (still requires --confirm). Without this flag, dry run only.")
    parser.add_argument("--confirm", action="store_true",
                        help="Required together with --apply to actually write changes.")
    parser.add_argument("--docs-dir", type=str, default=str(DOCS_DIR),
                        help=f"Path to docs/ (default: {DOCS_DIR})")
    parser.add_argument("--base-url", type=str, default=None,
                        help="Site base URL, e.g. https://kubaik.github.io (default: read from config.yaml)")
    parser.add_argument("--csv", type=str, default=str(DEFAULT_MERGE_PLAN_CSV),
                        help=f"Where to write the merge plan CSV (default: {DEFAULT_MERGE_PLAN_CSV}). Use '' to skip.")
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir)

    base_url = args.base_url
    if not base_url:
        try:
            import yaml
            config = yaml.safe_load(
                (REPO_ROOT / "config.yaml").read_text(encoding="utf-8"))
            base_url = (config.get("base_url") or "").rstrip("/")
        except Exception:
            base_url = ""
    if not base_url:
        print("ERROR: could not determine base_url from config.yaml. Pass --base-url explicitly.")
        sys.exit(1)

    print(f"Loading posts from {docs_dir}/ ...")
    posts = cqs.load_posts(docs_dir)
    print(f"Loaded {len(posts)} posts with post.json.")

    if args.month:
        posts = [p for p in posts if p.month_key == args.month]
        print(f"Filtered to month {args.month}: {len(posts)} posts.")

    if len(posts) < 2:
        print("Not enough posts to compare (need at least 2). Exiting.")
        return

    # Score first (composite score drives canonical selection), using the
    # existing nearest-neighbor pass so subscores are populated exactly as
    # they are in quality_report.csv.
    cqs.compute_similarities(posts)
    cqs.score_posts(posts)

    matrix = build_full_similarity_matrix(posts)

    exclude_set = set(args.exclude)
    keep_set = set(args.keep)
    clusters = find_clusters(posts, matrix, args.threshold, exclude_set)
    clusters = [c for c in clusters if len(c) >= args.min_cluster_size]

    if not clusters:
        print(f"\nNo clusters found at threshold={args.threshold}. "
              f"Try lowering --threshold if you expected matches.")
        return

    clusters_plan = []
    for i, cluster in enumerate(clusters, 1):
        canonical = pick_canonical(cluster, args.keep_longest, keep_set)
        redirects = [p for p in cluster if p.slug != canonical.slug]
        clusters_plan.append({
            "cluster_id": i,
            "canonical": canonical,
            "redirects": redirects,
            "max_similarity": cluster_max_similarity(cluster, matrix),
        })

    print_plan(clusters_plan)

    if args.csv:
        csv_path = Path(args.csv)
        write_merge_plan_csv(clusters_plan, csv_path)
        print(f"\nMerge plan written to {csv_path}")

    if not args.apply:
        print("\nThis was a dry run. Re-run with --apply --confirm to actually "
              "back up, write redirect stubs, and rewrite internal links.")
        return

    if not args.confirm:
        print("\n--apply was set but --confirm was not. Refusing to write "
              "changes -- pass both flags together to actually apply the merge.")
        return

    apply_merges(clusters_plan, docs_dir, base_url)


if __name__ == "__main__":
    main()
