#!/usr/bin/env python3
"""
content_backlog_cleanup.py

Cleans up an existing docs/ backlog of AI-generated blog posts based on the
same scoring dimensions your content_quality_scanner.py already computes
(SCORE, DUP, DEPTH, META, STRUCT, ADS + pairwise similarity).

WHAT IT DOES
------------
For every near-duplicate pair above a similarity threshold, it decides a
"survivor" (higher SCORE, or if tied, greater DEPTH, or if still tied, older
publish date since it likely has more accumulated backlinks/indexing) and a
"loser". For each loser it plans one of three actions:

  1. MERGE_REDIRECT  - loser is very similar (>= merge_threshold) to survivor.
                        Loser page gets <meta name="robots" content="noindex">
                        + <link rel="canonical" href="survivor_url"> + a
                        client-side redirect after a short delay, and is
                        removed from sitemap.xml. This preserves the URL
                        (no 404s for anyone who already indexed/bookmarked it)
                        while telling Google unambiguously not to index it
                        as a separate page.

  2. NOINDEX_ONLY    - loser is moderately similar (>= noindex_threshold but
                        < merge_threshold) OR scores below the thin-content
                        floor on its own (low DEPTH/META/STRUCT). Gets
                        noindex + canonical pointing at itself is NOT set;
                        instead it's flagged for a human rewrite pass. It
                        stays live (no redirect) since it isn't a true
                        duplicate, just weak.

  3. ARCHIVE         - loser is a near-exact duplicate (>= archive_threshold,
                        e.g. 0.85+) of an existing survivor. Physically moved
                        out of docs/ into _archive/ (outside the Pages build
                        output) so it stops being served at all. A thin
                        redirect stub is left behind at the old URL.

Every run first writes a JSON action plan to review. Nothing is modified
until you pass --apply.

USAGE
-----
  # 1. Dry run - just look at the plan
  python content_backlog_cleanup.py plan --docs-dir docs --out cleanup_plan.json

  # 2. Apply it after you've read cleanup_plan.json
  python content_backlog_cleanup.py apply --plan cleanup_plan.json --docs-dir docs

  # 3. Regenerate sitemap.xml to drop archived/redirected URLs
  python content_backlog_cleanup.py sitemap --docs-dir docs --base-url https://kubaik.github.io

ASSUMPTIONS (adjust the CONFIG block below to match your repo)
----------------------------------------------------------------
- Each post lives at docs/<slug>/index.html with a sibling docs/<slug>/post.json
  containing at least: {"title": ..., "published": "YYYY-MM-DD", ...}
- post.json also stores a "body" or "content" field used for similarity
  (falls back to stripping index.html if not present).
- sitemap.xml lives at docs/sitemap.xml with one <url><loc>...</loc></url>
  entry per post.
"""

import argparse
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False
    import difflib

# --------------------------------------------------------------------------
# CONFIG - tune these before running
# --------------------------------------------------------------------------
MERGE_THRESHOLD = 0.72     # near-duplicate: redirect + canonical + noindex
ARCHIVE_THRESHOLD = 0.85   # essentially the same post: physically remove
NOINDEX_THRESHOLD = 0.55   # weaker overlap or independently thin: flag only
THIN_DEPTH_FLOOR = 60      # DEPTH score below this = thin regardless of dup
THIN_WORDCOUNT_FLOOR = 500  # raw safety net if DEPTH isn't available

REDIRECT_DELAY_SECONDS = 3


@dataclass
class Post:
    slug: str
    dir: Path
    title: str = ""
    published: str = ""
    word_count: int = 0
    text: str = ""
    depth_score: Optional[float] = None
    overall_score: Optional[float] = None


def load_posts(docs_dir: Path) -> list[Post]:
    posts = []
    for sub in sorted(docs_dir.iterdir()):
        if not sub.is_dir():
            continue
        pj = sub / "post.json"
        idx = sub / "index.html"
        if not pj.exists() or not idx.exists():
            continue
        try:
            data = json.loads(pj.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  ! skipping {sub.name}: bad post.json ({e})")
            continue

        body = data.get("body") or data.get("content") or ""
        if not body:
            html = idx.read_text(encoding="utf-8", errors="ignore")
            body = re.sub("<[^>]+>", " ", html)

        text = re.sub(r"\s+", " ", body).strip()
        posts.append(Post(
            slug=sub.name,
            dir=sub,
            title=data.get("title", sub.name),
            published=data.get("published", data.get("date", "")),
            word_count=len(text.split()),
            text=text,
            depth_score=data.get("depth_score"),
            overall_score=data.get("quality_score"),
        ))
    return posts


def pairwise_similarity(posts: list[Post]):
    """Returns list of (i, j, similarity) for i < j, similarity descending."""
    texts = [p.text for p in posts]
    n = len(posts)
    pairs = []

    if HAVE_SKLEARN:
        vec = TfidfVectorizer(stop_words="english", max_features=20000)
        matrix = vec.fit_transform(texts)
        sims = cosine_similarity(matrix)
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((i, j, float(sims[i][j])))
    else:
        print("  ! sklearn not installed, falling back to difflib "
              "(slower, install scikit-learn for large corpora)")
        for i in range(n):
            for j in range(i + 1, n):
                ratio = difflib.SequenceMatcher(
                    None, texts[i][:5000], texts[j][:5000]
                ).ratio()
                pairs.append((i, j, ratio))

    pairs.sort(key=lambda x: -x[2])
    return pairs


def choose_survivor(a: Post, b: Post) -> tuple[Post, Post]:
    """Returns (survivor, loser)."""
    a_score = a.overall_score if a.overall_score is not None else a.word_count
    b_score = b.overall_score if b.overall_score is not None else b.word_count
    if a_score != b_score:
        return (a, b) if a_score > b_score else (b, a)
    # tie-break on age: older survives (more time to accrue links/index)
    if a.published and b.published:
        return (a, b) if a.published < b.published else (b, a)
    return (a, b)


def build_clusters(posts: list[Post], pairs: list[tuple[int, int, float]]) -> list[set]:
    """Union-Find over any pair meeting NOINDEX_THRESHOLD, so a chain of
    A~B~C~D (even if A and D individually score below threshold) is treated
    as one topic cluster rather than independent pairwise decisions."""
    parent = list(range(len(posts)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for i, j, sim in pairs:
        if sim >= NOINDEX_THRESHOLD:
            union(i, j)

    groups = {}
    for idx in range(len(posts)):
        groups.setdefault(find(idx), set()).add(idx)

    return [g for g in groups.values() if len(g) > 1]


def build_plan(docs_dir: Path, debug: bool = False,
               noindex_threshold: float = None,
               merge_threshold: float = None,
               archive_threshold: float = None) -> dict:
    global NOINDEX_THRESHOLD, MERGE_THRESHOLD, ARCHIVE_THRESHOLD
    if noindex_threshold is not None:
        NOINDEX_THRESHOLD = noindex_threshold
    if merge_threshold is not None:
        MERGE_THRESHOLD = merge_threshold
    if archive_threshold is not None:
        ARCHIVE_THRESHOLD = archive_threshold
    print(
        f"[thresholds] noindex={NOINDEX_THRESHOLD} merge={MERGE_THRESHOLD} archive={ARCHIVE_THRESHOLD}")

    print(f"Loading posts from {docs_dir} ...")
    posts = load_posts(docs_dir)
    print(f"Loaded {len(posts)} posts.")

    if debug:
        empty = sum(1 for p in posts if p.word_count == 0)
        zero_score = sum(1 for p in posts if p.overall_score is None)
        print(f"[debug] sklearn available: {HAVE_SKLEARN}")
        print(f"[debug] posts with 0 extracted words: {empty} / {len(posts)}")
        print(
            f"[debug] posts missing quality_score in post.json: {zero_score} / {len(posts)}")
        if posts:
            wc = sorted(p.word_count for p in posts)
            print(
                f"[debug] word_count min/median/max: {wc[0]}/{wc[len(wc)//2]}/{wc[-1]}")

    print("Computing pairwise similarity ...")
    pairs = pairwise_similarity(posts)

    if debug:
        print("[debug] top 15 similarity pairs (any threshold):")
        for i, j, sim in pairs[:15]:
            print(f"    {sim:.3f}  {posts[i].slug}  <->  {posts[j].slug}")
        if pairs and pairs[0][2] == 0.0:
            print("[debug] WARNING: top similarity is 0.0 - text extraction "
                  "is likely broken (empty 'body'/'content' fields and "
                  "index.html stripping produced nothing). Check post.json "
                  "field names against load_posts().")

    pair_sim = {}
    for i, j, sim in pairs:
        pair_sim[(i, j)] = sim
        pair_sim[(j, i)] = sim

    clusters = build_clusters(posts, pairs)
    print(f"Found {len(clusters)} duplicate cluster(s) "
          f"({sum(len(c) for c in clusters)} posts involved).")

    decided = {}   # slug -> action dict

    for cluster_idxs in clusters:
        cluster = [posts[i] for i in cluster_idxs]
        # cornerstone = highest overall_score, tie-break on word_count
        cornerstone = max(
            cluster,
            key=lambda p: (p.overall_score if p.overall_score is not None else -1,
                           p.word_count),
        )
        others = [p for p in cluster if p.slug != cornerstone.slug]

        # max similarity of each "other" to ANY member of the cluster,
        # used only to decide ARCHIVE vs MERGE_REDIRECT severity
        idx_by_slug = {p.slug: i for i, p in enumerate(posts)}
        for loser in others:
            li = idx_by_slug[loser.slug]
            best_sim = max(
                (pair_sim.get((li, idx_by_slug[m.slug]), 0.0)
                 for m in cluster if m.slug != loser.slug),
                default=0.0,
            )
            if best_sim >= ARCHIVE_THRESHOLD:
                action = "ARCHIVE"
            else:
                # any post that's part of a confirmed cluster (>=2 members)
                # gets consolidated via redirect rather than left orphaned,
                # since the goal is one strong cornerstone per topic, not
                # N noindexed near-duplicates sitting around unindexed.
                action = "MERGE_REDIRECT" if len(
                    cluster) >= 3 or best_sim >= MERGE_THRESHOLD else "NOINDEX_ONLY"

            decided[loser.slug] = {
                "slug": loser.slug,
                "action": action,
                "similarity": round(best_sim, 3),
                "survivor_slug": cornerstone.slug,
                "survivor_title": cornerstone.title,
                "loser_title": loser.title,
                "cluster_size": len(cluster),
            }

    # independently thin posts that weren't caught by similarity at all
    for p in posts:
        if p.slug in decided:
            continue
        thin_by_depth = p.depth_score is not None and p.depth_score < THIN_DEPTH_FLOOR
        thin_by_wc = p.word_count < THIN_WORDCOUNT_FLOOR
        if thin_by_depth or thin_by_wc:
            decided[p.slug] = {
                "slug": p.slug,
                "action": "NOINDEX_ONLY",
                "similarity": None,
                "survivor_slug": None,
                "survivor_title": None,
                "loser_title": p.title,
                "reason": f"thin content (depth={p.depth_score}, words={p.word_count})",
            }

    summary = {
        "total_posts": len(posts),
        "archive": sum(1 for v in decided.values() if v["action"] == "ARCHIVE"),
        "merge_redirect": sum(1 for v in decided.values() if v["action"] == "MERGE_REDIRECT"),
        "noindex_only": sum(1 for v in decided.values() if v["action"] == "NOINDEX_ONLY"),
        "unaffected": len(posts) - len(decided),
    }
    return {"summary": summary, "actions": list(decided.values())}


# --------------------------------------------------------------------------
# APPLY
# --------------------------------------------------------------------------

NOINDEX_TAG = '<meta name="robots" content="noindex, follow">'
CANONICAL_TMPL = '<link rel="canonical" href="{url}">'
REDIRECT_SCRIPT_TMPL = """<script>
  setTimeout(function() {{
    window.location.replace("{url}");
  }}, {delay}000);
</script>
<p style="text-align:center;padding:2rem;">
  This post has been merged into
  <a href="{url}">a more complete article</a>. Redirecting...
</p>"""


def inject_head_tags(html: str, tags: list[str]) -> str:
    insertion = "\n".join(tags) + "\n</head>"
    if "</head>" in html:
        return html.replace("</head>", insertion, 1)
    return insertion + html  # no <head>, just prepend (shouldn't happen)


def apply_plan_hard_delete(plan: dict, docs_dir: Path) -> list[str]:
    """Permanently remove every flagged post's directory. No stubs, no
    redirects, no noindex tags - the URL simply stops resolving (404).
    Use this when you want the duplicate/thin posts gone outright rather
    than preserved-but-hidden."""
    log = []
    for action in plan["actions"]:
        slug = action["slug"]
        post_dir = docs_dir / slug
        if not post_dir.exists():
            log.append(f"SKIP {slug}: directory not found (already removed?)")
            continue
        shutil.rmtree(post_dir)
        log.append(f"DELETED {slug} (was {action['action']}, "
                   f"similarity={action.get('similarity')}, "
                   f"cluster_size={action.get('cluster_size', 1)})")
    return log


def apply_plan(plan: dict, docs_dir: Path, base_url: str, archive_dir: Path):
    archive_dir.mkdir(parents=True, exist_ok=True)
    log = []

    for action in plan["actions"]:
        slug = action["slug"]
        post_dir = docs_dir / slug
        idx = post_dir / "index.html"
        if not idx.exists():
            log.append(f"SKIP {slug}: index.html missing")
            continue

        html = idx.read_text(encoding="utf-8", errors="ignore")
        survivor_url = None
        if action.get("survivor_slug"):
            survivor_url = f"{base_url.rstrip('/')}/{action['survivor_slug']}/"

        if action["action"] == "NOINDEX_ONLY":
            html = inject_head_tags(html, [NOINDEX_TAG])
            idx.write_text(html, encoding="utf-8")
            log.append(
                f"NOINDEX  {slug} (reason={action.get('reason', action.get('similarity'))})")

        elif action["action"] == "MERGE_REDIRECT":
            tags = [NOINDEX_TAG]
            if survivor_url:
                tags.append(CANONICAL_TMPL.format(url=survivor_url))
            html = inject_head_tags(html, tags)
            # replace body content with redirect stub, keep head intact
            body_replaced = re.sub(
                r"<body[^>]*>.*</body>",
                f"<body>{REDIRECT_SCRIPT_TMPL.format(url=survivor_url, delay=REDIRECT_DELAY_SECONDS)}</body>",
                html, flags=re.DOTALL,
            )
            idx.write_text(body_replaced, encoding="utf-8")
            log.append(f"REDIRECT {slug} -> {action['survivor_slug']}")

        elif action["action"] == "ARCHIVE":
            dest = archive_dir / slug
            if dest.exists():
                shutil.rmtree(dest)
            shutil.move(str(post_dir), str(dest))
            # leave a thin stub at the old location so the URL still resolves
            post_dir.mkdir(parents=True, exist_ok=True)
            tags = [NOINDEX_TAG]
            if survivor_url:
                tags.append(CANONICAL_TMPL.format(url=survivor_url))
            stub = (
                "<!DOCTYPE html><html><head>"
                + "\n".join(tags)
                + f'<meta charset="utf-8"><title>Moved</title></head>'
                + f"<body>{REDIRECT_SCRIPT_TMPL.format(url=survivor_url, delay=REDIRECT_DELAY_SECONDS)}</body></html>"
            )
            (post_dir / "index.html").write_text(stub, encoding="utf-8")
            log.append(
                f"ARCHIVE  {slug} -> _archive/{slug}, stub redirects to {action['survivor_slug']}")

    return log


def regenerate_sitemap(docs_dir: Path, base_url: str):
    """Rebuild sitemap.xml excluding anything noindexed/archived/redirected."""
    entries = []
    for sub in sorted(docs_dir.iterdir()):
        if not sub.is_dir():
            continue
        idx = sub / "index.html"
        if not idx.exists():
            continue
        html = idx.read_text(encoding="utf-8", errors="ignore")
        if 'name="robots" content="noindex' in html:
            continue
        pj = sub / "post.json"
        lastmod = ""
        if pj.exists():
            try:
                data = json.loads(pj.read_text(encoding="utf-8"))
                lastmod = data.get("published", data.get("date", ""))
            except Exception:
                pass
        entries.append((f"{base_url.rstrip('/')}/{sub.name}/", lastmod))

    xml_lines = ['<?xml version="1.0" encoding="UTF-8"?>',
                 '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">']
    for url, lastmod in entries:
        xml_lines.append("  <url>")
        xml_lines.append(f"    <loc>{url}</loc>")
        if lastmod:
            xml_lines.append(f"    <lastmod>{lastmod}</lastmod>")
        xml_lines.append("  </url>")
    xml_lines.append("</urlset>")

    (docs_dir / "sitemap.xml").write_text("\n".join(xml_lines), encoding="utf-8")
    print(f"sitemap.xml rewritten with {len(entries)} indexable URLs "
          f"(was {sum(1 for _ in docs_dir.iterdir())} total post dirs).")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_plan = sub.add_parser(
        "plan", help="Compute the cleanup plan (no changes made).")
    p_plan.add_argument("--docs-dir", default="docs", type=Path)
    p_plan.add_argument("--out", default="cleanup_plan.json", type=Path)
    p_plan.add_argument("--debug", action="store_true",
                        help="Print the top 15 pairwise similarities found, "
                        "regardless of threshold, plus text-extraction "
                        "stats. Use this to sanity-check a 0-cluster result.")
    p_plan.add_argument("--noindex-threshold", type=float, default=None,
                        help=f"Override NOINDEX_THRESHOLD (default {NOINDEX_THRESHOLD})")
    p_plan.add_argument("--merge-threshold", type=float, default=None,
                        help=f"Override MERGE_THRESHOLD (default {MERGE_THRESHOLD})")
    p_plan.add_argument("--archive-threshold", type=float, default=None,
                        help=f"Override ARCHIVE_THRESHOLD (default {ARCHIVE_THRESHOLD})")

    p_apply = sub.add_parser(
        "apply", help="Apply a previously generated plan.")
    p_apply.add_argument("--plan", required=True, type=Path)
    p_apply.add_argument("--docs-dir", default="docs", type=Path)
    p_apply.add_argument("--base-url", default="https://kubaik.github.io")
    p_apply.add_argument("--archive-dir", default="_archive", type=Path)
    p_apply.add_argument("--hard-delete", action="store_true",
                         help="Permanently rm -rf every flagged post's directory "
                         "instead of noindex/redirect/archive-with-stub. "
                         "URLs will 404. Cannot be undone except via git.")

    p_sitemap = sub.add_parser(
        "sitemap", help="Regenerate sitemap.xml excluding noindexed pages.")
    p_sitemap.add_argument("--docs-dir", default="docs", type=Path)
    p_sitemap.add_argument("--base-url", default="https://kubaik.github.io")

    args = ap.parse_args()

    if args.cmd == "plan":
        plan = build_plan(args.docs_dir, debug=args.debug,
                          noindex_threshold=args.noindex_threshold,
                          merge_threshold=args.merge_threshold,
                          archive_threshold=args.archive_threshold)
        args.out.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        print("\n=== SUMMARY ===")
        for k, v in plan["summary"].items():
            print(f"  {k}: {v}")
        print(f"\nFull plan written to {args.out}. Review it, then run:")
        print(
            f"  python {sys.argv[0]} apply --plan {args.out} --docs-dir {args.docs_dir}")

    elif args.cmd == "apply":
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        n = len(plan["actions"])

        if args.hard_delete:
            print(f"About to PERMANENTLY DELETE {n} post directories from "
                  f"{args.docs_dir} (no stubs, no redirects - URLs will 404).")
            confirm = input('Type "delete" to confirm: ')
            if confirm.strip().lower() != "delete":
                print("Aborted, nothing deleted.")
                return
            log = apply_plan_hard_delete(plan, args.docs_dir)
        else:
            log = apply_plan(plan, args.docs_dir,
                             args.base_url, args.archive_dir)

        for line in log:
            print(line)
        print(f"\n{len(log)} posts modified. Now run:")
        print(
            f"  python {sys.argv[0]} sitemap --docs-dir {args.docs_dir} --base-url {args.base_url}")

    elif args.cmd == "sitemap":
        regenerate_sitemap(args.docs_dir, args.base_url)


if __name__ == "__main__":
    main()
