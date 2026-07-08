#!/usr/bin/env python3
"""
scripts/audit_corpus_duplicates.py
===================================
Retroactive near-duplicate audit AND cleanup for the ALREADY-PUBLISHED
corpus.

WHY THIS EXISTS
---------------
blog_system.py already contains solid duplicate-detection logic
(PreFlightIndex, ContentDuplicateGate, SimilarityGuard) — but those
only run against NEW posts at generation time. They have never been
run against the ~800 posts that were published before those gates
existed (or that slipped through before recent threshold tuning).

This script re-uses the same TF-IDF cosine-similarity approach as
ContentDuplicateGate in blog_system.py (same tokenizer, same stopword
list, same math) so a post that would be blocked today is reported
here too, clusters existing posts by mutual similarity, and can
optionally clean up each cluster automatically by keeping one post and
removing the rest.

DEFAULT BEHAVIOUR IS READ-ONLY (dry run). Nothing is deleted unless you
pass --delete AND --yes (or answer the interactive confirmation).

CLEANUP RULE
------------
Within each cluster, one post is kept and the others are removed:
  --keep newest   (default) keep the most recently created post,
                  remove the older one(s) — matches "these are stale
                  duplicates, the newest rewrite is the one to keep."
  --keep oldest   keep the post with the most tenure/likely the most
                  externally indexed and linked one, remove the newer
                  duplicate(s) instead. Consider this if you care about
                  preserving existing backlinks/search history more
                  than freshness.
"Created" is read from post.json's created_at field, falling back to
the post directory's filesystem mtime if that field is missing.

REMOVAL MODE
------------
  --mode hard-delete   (default) removes the entire post directory.
                       Simplest, but the URL starts 404ing immediately,
                       which can look like link rot to search engines
                       and drops any backlink equity the removed post
                       had accumulated.
  --mode redirect-stub replaces the removed post's index.html with a
                       lightweight HTML redirect (meta-refresh + 301-
                       style rel=canonical) pointing at the kept post,
                       and deletes its post.json so it's excluded from
                       future audits/generation indexes. This preserves
                       link equity and is the safer default for a site
                       that already has Search Console history —
                       consider this over hard-delete unless the
                       removed posts have no meaningful traffic/links.

After deleting or stubbing anything, re-run your sitemap generator
(scripts/generate_sitemap.py) and re-deploy — this script only touches
docs/<slug>/ directories, not sitemap.xml or your CDN cache.

USAGE
-----
    # Dry run — see what WOULD happen, nothing is changed:
    python scripts/audit_corpus_duplicates.py --docs docs --threshold 0.72

    # Actually clean up, keeping the newest post per cluster:
    python scripts/audit_corpus_duplicates.py --docs docs --threshold 0.72 \\
        --delete --keep newest --mode redirect-stub --yes
        
        python scripts/audit_corpus_duplicates.py --docs docs --threshold 0.72 --delete --keep newest --mode redirect-stub --yes

    # Write a machine-readable report without deleting anything:
    python scripts/audit_corpus_duplicates.py --docs docs --json report.json
    
    

Increase --threshold toward 0.72 (the live ContentDuplicateGate
threshold) for a stricter "would this be blocked today" view, or lower
it toward 0.40 to catch softer topical overlap worth diverging
(recommended for review only — do not auto-delete at low thresholds,
since below ~0.65 some flagged pairs are legitimately related-but-
distinct posts, not duplicates).
"""

import argparse
import json
import math
import re
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "can", "that",
    "this", "these", "those", "it", "its", "we", "you", "your", "our",
    "they", "their", "what", "which", "who", "when", "where", "how",
    "not", "no", "so", "if", "as", "than", "then", "about", "up",
    "out", "into", "more", "also", "just", "after", "before", "over",
    "some", "any", "all", "each", "both", "between", "through",
}

# Index/cache files elsewhere in the pipeline that key on post slug.
# When a post is removed we prune stale entries from these too, so
# blog_system.py's own gates don't keep comparing against ghosts.
_KNOWN_SLUG_KEYED_CACHES = [
    ".similarity_index.json",
    ".preflight_index.json",
]

_REDIRECT_STUB_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<link rel="canonical" href="{target_url}">
<meta http-equiv="refresh" content="0; url={target_url}">
<meta name="robots" content="noindex">
</head>
<body>
<p>This article has been merged into a more complete version:
<a href="{target_url}">{title}</a></p>
</body>
</html>
"""


# ─────────────────────────────────────────────────────────────────
# Similarity (same algorithm as blog_system.py's ContentDuplicateGate)
# ─────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b[a-z][a-z']{1,}\b", text.lower())


def _tfidf_vector(text: str) -> Dict[str, float]:
    tokens = [t for t in _tokenize(text) if t not in _STOP_WORDS]
    if not tokens:
        return {}
    counts = Counter(tokens)
    total = len(tokens)
    return {word: count / total for word, count in counts.items()}


def _cosine(v1: Dict[str, float], v2: Dict[str, float]) -> float:
    common = set(v1) & set(v2)
    if not common:
        return 0.0
    dot = sum(v1[k] * v2[k] for k in common)
    mag1 = math.sqrt(sum(x ** 2 for x in v1.values()))
    mag2 = math.sqrt(sum(x ** 2 for x in v2.values()))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


def _parse_created_at(data: dict, post_dir: Path) -> float:
    """
    Return a sortable timestamp for a post: created_at from post.json
    if present and parseable, else the post directory's mtime.
    """
    raw = data.get("created_at") or data.get("updated_at")
    if raw:
        try:
            return datetime.fromisoformat(raw).timestamp()
        except ValueError:
            pass
    try:
        return post_dir.stat().st_mtime
    except OSError:
        return 0.0


def load_corpus(docs_dir: Path) -> Dict[str, dict]:
    corpus = {}
    for post_dir in sorted(docs_dir.iterdir()):
        if not post_dir.is_dir():
            continue
        post_json = post_dir / "post.json"
        if not post_json.exists():
            continue
        try:
            data = json.loads(post_json.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        content = data.get("content", "")
        if len(content) < 500:
            continue
        corpus[post_dir.name] = {
            "title": data.get("title", post_dir.name),
            "vector": _tfidf_vector(content),
            "created_at": _parse_created_at(data, post_dir),
        }
    return corpus


def find_clusters(corpus: Dict[str, dict], threshold: float) -> List[List[Tuple[str, str, float]]]:
    """
    Union-find style clustering: any two posts scoring >= threshold are
    linked into the same cluster, then clusters of size > 1 are returned
    with the pairwise score that triggered each link.
    """
    slugs = list(corpus.keys())
    parent = {s: s for s in slugs}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    edges = []
    n = len(slugs)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = slugs[i], slugs[j]
            score = _cosine(corpus[a]["vector"], corpus[b]["vector"])
            if score >= threshold:
                edges.append((a, b, score))
                union(a, b)

    groups: Dict[str, List[str]] = {}
    for s in slugs:
        groups.setdefault(find(s), []).append(s)

    clusters = []
    for members in groups.values():
        if len(members) < 2:
            continue
        member_set = set(members)
        cluster_edges = [(a, b, s) for a, b, s in edges
                         if a in member_set and b in member_set]
        clusters.append(cluster_edges)
    return clusters


# ─────────────────────────────────────────────────────────────────
# Cleanup actions
# ─────────────────────────────────────────────────────────────────

def resolve_cluster_actions(
    members: List[str],
    corpus: Dict[str, dict],
    keep: str,
) -> Tuple[str, List[str]]:
    """
    Decide which slug in a cluster to keep and which to remove.
    Returns (keep_slug, [remove_slug, ...]).
    """
    ordered = sorted(members, key=lambda s: corpus[s]["created_at"])
    if keep == "newest":
        keep_slug = ordered[-1]
    else:  # "oldest"
        keep_slug = ordered[0]
    remove_slugs = [s for s in members if s != keep_slug]
    return keep_slug, remove_slugs


def _prune_slug_from_caches(slug: str, docs_dir: Path) -> None:
    """Remove stale entries for a deleted slug from known index caches."""
    for cache_name in _KNOWN_SLUG_KEYED_CACHES:
        # These caches live at the project root in blog_system.py's
        # convention, i.e. alongside docs_dir's parent, not inside docs/.
        cache_path = docs_dir.parent / cache_name
        if not cache_path.exists():
            continue
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, dict) and slug in data:
            del data[slug]
            try:
                cache_path.write_text(json.dumps(data), encoding="utf-8")
            except OSError:
                pass


def remove_post(
    slug: str,
    keep_slug: str,
    docs_dir: Path,
    base_url: str,
    mode: str,
    corpus: Dict[str, dict],
) -> None:
    post_dir = docs_dir / slug
    if not post_dir.exists():
        return

    if mode == "redirect-stub":
        target_url = f"{base_url.rstrip('/')}/{keep_slug}/"
        stub = _REDIRECT_STUB_TEMPLATE.format(
            title=corpus[keep_slug]["title"],
            target_url=target_url,
        )
        index_html = post_dir / "index.html"
        index_html.write_text(stub, encoding="utf-8")
        post_json = post_dir / "post.json"
        if post_json.exists():
            post_json.unlink()
        # Remove any other generated assets (og images etc.) except the
        # stub itself, so the directory stops looking like a full post.
        for child in post_dir.iterdir():
            if child.name != "index.html":
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)
    else:  # hard-delete
        shutil.rmtree(post_dir)

    _prune_slug_from_caches(slug, docs_dir)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs", default="docs",
                        help="Path to the docs/ output directory")
    parser.add_argument("--threshold", type=float, default=0.55,
                        help="Cosine similarity threshold to consider a pair duplicate (default 0.55)")
    parser.add_argument("--json", default=None,
                        help="Optional path to write a JSON report")
    parser.add_argument("--delete", action="store_true",
                        help="Actually remove/redirect the non-kept post in each cluster. "
                        "Without this flag the script only reports (dry run).")
    parser.add_argument("--keep", choices=["newest", "oldest"], default="newest",
                        help="Which post in each cluster to keep (default: newest)")
    parser.add_argument("--mode", choices=["hard-delete", "redirect-stub"], default="hard-delete",
                        help="How to remove a non-kept post (default: hard-delete). "
                        "redirect-stub preserves the URL as a redirect to the kept post "
                        "instead of letting it 404 — recommended if these posts may "
                        "already have backlinks or search impressions.")
    parser.add_argument("--base-url", default="https://kubaik.github.io",
                        help="Site base URL, used to build redirect targets in --mode redirect-stub")
    parser.add_argument("--yes", action="store_true",
                        help="Skip the interactive confirmation prompt before deleting")
    parser.add_argument("--min-threshold-for-delete", type=float, default=0.65,
                        help="Safety floor: refuses --delete if --threshold is set below this "
                        "(default 0.65). Low thresholds catch topically-related but "
                        "legitimately distinct posts, which should be reviewed, not "
                        "auto-deleted.")
    args = parser.parse_args()

    docs_dir = Path(args.docs)
    if not docs_dir.exists():
        raise SystemExit(f"docs dir not found: {docs_dir}")

    if args.delete and args.threshold < args.min_threshold_for_delete:
        raise SystemExit(
            f"Refusing to --delete at --threshold={args.threshold} "
            f"(below --min-threshold-for-delete={args.min_threshold_for_delete}). "
            f"Re-run without --delete to review this threshold's clusters first, "
            f"or raise --threshold, or explicitly lower --min-threshold-for-delete "
            f"if you've reviewed the report and are confident."
        )

    corpus = load_corpus(docs_dir)
    print(f"Loaded {len(corpus)} posts. Computing pairwise similarity "
          f"(this is O(n^2) — a few thousand posts is fine, tens of "
          f"thousands will need blocking/LSH instead)...")

    clusters = find_clusters(corpus, args.threshold)
    clusters.sort(key=lambda c: max(s for _, _, s in c), reverse=True)

    print(
        f"\nFound {len(clusters)} duplicate cluster(s) at threshold {args.threshold}:\n")

    report = []
    planned_removals: List[Tuple[str, str]] = []  # (remove_slug, keep_slug)

    for i, cluster_edges in enumerate(clusters, 1):
        members = sorted({a for a, b, s in cluster_edges} |
                         {b for a, b, s in cluster_edges})
        keep_slug, remove_slugs = resolve_cluster_actions(
            members, corpus, args.keep)

        print(f"Cluster {i} ({len(members)} posts) — keep: /{keep_slug}/")
        for slug in members:
            tag = "KEEP  " if slug == keep_slug else "REMOVE"
            created = datetime.fromtimestamp(
                corpus[slug]["created_at"]).strftime("%Y-%m-%d")
            print(f"   [{tag}] /{slug}/  ({created})  — {corpus[slug]['title']}")
        for a, b, s in sorted(cluster_edges, key=lambda e: -e[2])[:5]:
            print(f"     {a}  <->  {b}   similarity={s:.2f}")
        print()

        for slug in remove_slugs:
            planned_removals.append((slug, keep_slug))

        report.append({
            "keep": keep_slug,
            "members": [
                {
                    "slug": s,
                    "title": corpus[s]["title"],
                    "created_at": datetime.fromtimestamp(corpus[s]["created_at"]).isoformat(),
                    "action": "keep" if s == keep_slug else "remove",
                }
                for s in members
            ],
            "top_edges": [{"a": a, "b": b, "similarity": round(s, 3)}
                          for a, b, s in sorted(cluster_edges, key=lambda e: -e[2])[:5]],
        })

    if args.json:
        Path(args.json).write_text(json.dumps(
            report, indent=2), encoding="utf-8")
        print(f"Wrote JSON report to {args.json}")

    if not planned_removals:
        print("Nothing to remove.")
        return

    print(
        f"Planned removals: {len(planned_removals)} post(s), mode={args.mode}, keep={args.keep}")

    if not args.delete:
        print("\nDry run only — no files were changed. Re-run with --delete to apply "
              "(add --yes to skip confirmation).")
        return

    if not args.yes:
        confirm = input(
            f"\nThis will {'redirect' if args.mode == 'redirect-stub' else 'permanently delete'} "
            f"{len(planned_removals)} post director{'y' if len(planned_removals)==1 else 'ies'} "
            f"under {docs_dir}/. Type 'yes' to continue: "
        )
        if confirm.strip().lower() != "yes":
            print("Aborted. No changes made.")
            return

    for slug, keep_slug in planned_removals:
        remove_post(slug, keep_slug, docs_dir,
                    args.base_url, args.mode, corpus)
        print(
            f"  {'redirected' if args.mode == 'redirect-stub' else 'deleted'}: /{slug}/  -> /{keep_slug}/")

    print(
        f"\nDone. {len(planned_removals)} post(s) processed. "
        f"Remember to regenerate sitemap.xml (scripts/generate_sitemap.py) "
        f"and redeploy before these changes are visible to search engines."
    )


if __name__ == "__main__":
    main()
