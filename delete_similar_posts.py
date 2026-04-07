"""
delete_similar_posts.py

Detects blog posts that are too similar in subject matter using TF-IDF
vectorization and cosine similarity, then lets you delete the duplicates
while keeping the best version of each group.

How similarity is determined:
  - Compares post titles + meta descriptions + tags (lightweight, no large deps)
  - Uses TF-IDF to turn text into vectors, then cosine similarity to score pairs
  - Posts scoring above the threshold are grouped as "near-duplicates"
  - Within each group, the OLDEST post is kept (assumption: first = original)
    unless you pass --keep-longest to keep the one with the most content instead

Usage:
    python delete_similar_posts.py                        # dry run, threshold 0.75
    python delete_similar_posts.py --threshold 0.85       # stricter matching
    python delete_similar_posts.py --threshold 0.65       # more aggressive
    python delete_similar_posts.py --keep-longest         # keep richest post, not oldest
    python delete_similar_posts.py --delete               # actually delete
    python delete_similar_posts.py --delete --threshold 0.80 --keep-longest
"""

import json
import shutil
import sys
import re
from pathlib import Path
from itertools import combinations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DOCS_DIR = Path("./docs")
DEFAULT_THRESHOLD = 0.75   # 0.0 = no similarity, 1.0 = identical


# ── Text helpers ──────────────────────────────────────────────────────────────

def clean(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def post_to_text(data: dict) -> str:
    """
    Build a representative text fingerprint for a post.
    Weights: title (x4) > meta_description (x2) > tags (x1).
    Content is intentionally excluded — it's too long and skews similarity
    toward shared boilerplate phrases rather than actual subject matter.
    """
    title       = clean(data.get("title", ""))
    description = clean(data.get("meta_description", ""))
    tags        = clean(" ".join(data.get("tags", [])))
    keywords    = clean(" ".join(data.get("seo_keywords", [])))

    # Repeat title/description to give them more weight in TF-IDF
    return f"{title} {title} {title} {title} {description} {description} {tags} {keywords}"


def content_length(data: dict) -> int:
    return len(data.get("content", ""))


# ── Load all posts ────────────────────────────────────────────────────────────

def load_posts(docs_dir: Path) -> list[dict]:
    posts = []
    for post_dir in sorted(docs_dir.iterdir()):
        if not post_dir.is_dir() or post_dir.name == "static":
            continue
        post_json = post_dir / "post.json"
        if not post_json.exists():
            continue
        try:
            with open(post_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["_dir"]  = post_dir          # attach path for deletion
            data["_text"] = post_to_text(data)
            posts.append(data)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  ⚠️  Skipping {post_dir.name}: {e}")
    return posts


# ── Similarity matrix ─────────────────────────────────────────────────────────

def build_similarity_matrix(posts: list[dict]) -> np.ndarray:
    texts = [p["_text"] for p in posts]
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),   # unigrams + bigrams catch "machine learning" etc.
        min_df=1,
        sublinear_tf=True,    # log-scale TF dampens very frequent terms
        stop_words="english"
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    return cosine_similarity(tfidf_matrix)


# ── Group near-duplicates (union-find) ───────────────────────────────────────

def find_groups(posts: list[dict], sim_matrix: np.ndarray, threshold: float) -> list[list[int]]:
    """
    Union-Find: merge any two posts whose similarity >= threshold into the same group.
    Returns a list of groups, each group is a list of post indices.
    Only groups with 2+ posts are returned (singletons ignored).
    """
    n = len(posts)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for i, j in combinations(range(n), 2):
        if sim_matrix[i, j] >= threshold:
            union(i, j)

    # Collect groups
    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    return [g for g in groups.values() if len(g) > 1]


# ── Pick which post to KEEP within a group ────────────────────────────────────

def pick_keeper(group: list[int], posts: list[dict], keep_longest: bool) -> int:
    if keep_longest:
        return max(group, key=lambda i: content_length(posts[i]))
    else:
        # Keep oldest (smallest created_at string — ISO dates sort lexicographically)
        return min(group, key=lambda i: posts[i].get("created_at", ""))


# ── Main ──────────────────────────────────────────────────────────────────────

def run(threshold: float, keep_longest: bool, dry_run: bool):
    if not DOCS_DIR.exists():
        print(f"❌ Docs directory not found: {DOCS_DIR.resolve()}")
        sys.exit(1)

    print(f"\nLoading posts from {DOCS_DIR.resolve()} ...")
    posts = load_posts(DOCS_DIR)

    if len(posts) < 2:
        print("Not enough posts to compare (need at least 2).")
        return

    print(f"Loaded {len(posts)} posts. Building similarity matrix ...")
    sim_matrix = build_similarity_matrix(posts)

    print(f"Finding groups with similarity >= {threshold} ...\n")
    groups = find_groups(posts, sim_matrix, threshold)

    # ── Summary header ────────────────────────────────────────────────────────
    total_to_delete = sum(len(g) - 1 for g in groups)

    print("=" * 70)
    print(f"  {'DRY RUN — ' if dry_run else ''}Similar Post Scanner")
    print("=" * 70)
    print(f"  Threshold      : {threshold}  (higher = stricter)")
    print(f"  Keep strategy  : {'longest content' if keep_longest else 'oldest post'}")
    print(f"  Posts scanned  : {len(posts)}")
    print(f"  Similar groups : {len(groups)}")
    print(f"  Posts to delete: {total_to_delete}")
    print("=" * 70)

    if not groups:
        print("\n✅ No similar post groups found at this threshold.")
        print("   Try lowering --threshold if you expected matches.")
        return

    deleted = 0

    for group_num, group in enumerate(groups, 1):
        keeper_idx   = pick_keeper(group, posts, keep_longest)
        to_delete    = [i for i in group if i != keeper_idx]
        keeper       = posts[keeper_idx]

        print(f"\n── Group {group_num} ({'%d similar posts' % len(group)}) {'─' * 40}")

        # Show similarity scores between all pairs in the group
        print(f"  Similarity scores within group:")
        for i, j in combinations(group, 2):
            score = sim_matrix[i, j]
            print(f"    {posts[i]['title'][:40]:<42} ↔  {posts[j]['title'][:40]:<42}  {score:.3f}")

        print(f"\n  ✅ KEEPING  [{keeper.get('created_at','?')[:10]}]  {keeper['title']}")
        print(f"     Slug: {keeper['_dir'].name}")
        print(f"     Content length: {content_length(keeper):,} chars")

        for idx in to_delete:
            post = posts[idx]
            action = "🗑️  DELETING" if not dry_run else "🔍 WOULD DELETE"
            print(f"\n  {action}  [{post.get('created_at','?')[:10]}]  {post['title']}")
            print(f"     Slug: {post['_dir'].name}")
            print(f"     Content length: {content_length(post):,} chars")

            if not dry_run:
                try:
                    shutil.rmtree(post["_dir"])
                    print(f"     ✅ Deleted")
                    deleted += 1
                except OSError as e:
                    print(f"     ❌ Failed: {e}")

    # ── Footer ────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    if dry_run:
        print(f"  DRY RUN complete.")
        print(f"  {total_to_delete} post(s) across {len(groups)} group(s) would be deleted.")
        print(f"  Run with --delete to apply, or adjust --threshold first.")
    else:
        print(f"  Done. {deleted}/{total_to_delete} post(s) deleted.")
        print(f"  Run 'python blog_system.py build' to rebuild the site.")
    print("=" * 70 + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]

    dry_run      = "--delete"       not in args
    keep_longest = "--keep-longest" in args

    threshold = DEFAULT_THRESHOLD
    if "--threshold" in args:
        idx = args.index("--threshold")
        try:
            threshold = float(args[idx + 1])
            if not 0.0 < threshold < 1.0:
                raise ValueError
        except (IndexError, ValueError):
            print("❌ --threshold must be a float between 0.0 and 1.0")
            sys.exit(1)

    if not dry_run:
        print(f"⚠️  About to DELETE similar posts (threshold={threshold}).")
        confirm = input("   Type 'yes' to confirm: ").strip().lower()
        if confirm != "yes":
            print("Aborted.")
            sys.exit(0)

    run(threshold=threshold, keep_longest=keep_longest, dry_run=dry_run)