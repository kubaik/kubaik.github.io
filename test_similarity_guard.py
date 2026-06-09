#!/usr/bin/env python3
"""
test_similarity_guard.py
========================
Local test script that:
  1. Builds a similarity index from every post.json found under docs/
  2. Runs all three similarity layers (cross-post Jaccard, structural
     repetition, optional Copyscape) on every post against every other
  3. Prints a per-post report (pass / warn / blocked)
  4. Optionally deletes posts that exceed the block threshold, by age
     (oldest first) with a user-supplied count cap

Usage
-----
  # Check all posts, report only (no deletions)
  python test_similarity_guard.py

  # Delete the 5 oldest blocked posts (shows dry-run first)
  python test_similarity_guard.py --delete 5

  # Delete ALL blocked posts, skip confirmation
  python test_similarity_guard.py --delete all --yes

  # Delete ALL blocked posts but keep the newest post in each duplicate cluster
  python test_similarity_guard.py --delete all --retain-newest --yes

  # Also check for structural repetition warnings
  python test_similarity_guard.py --structural

  # Point at a different docs directory
  python test_similarity_guard.py --docs /path/to/docs

  # Show only posts with issues (suppress clean posts)
  python test_similarity_guard.py --failures-only

  # Raise or lower the block / warn thresholds
  python test_similarity_guard.py --block-threshold 0.40 --warn-threshold 0.25
"""

import argparse
import hashlib
import json
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# ── Similarity constants (mirrors similarity_guard.py defaults) ───────────────

DEFAULT_BLOCK_THRESHOLD = 0.10
DEFAULT_WARN_THRESHOLD = 0.20
STRUCTURAL_WARN_THRESHOLD = 0.40
MIN_CHARS_TO_CHECK = 1000


# ── Text normalisation ────────────────────────────────────────────────────────

_CODE_FENCE_RE = re.compile(r"```[\s\S]*?```")
_HEADING_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^\)]+\)")
_MARKUP_RE = re.compile(r"[*_`>|#\[\]()~]")
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    text = _CODE_FENCE_RE.sub(" ", text)
    text = _HEADING_RE.sub(" ", text)
    text = _LINK_RE.sub(r"\1", text)
    text = _MARKUP_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.lower().strip()


def _shingles(text: str, k: int = 5) -> Set[str]:
    words = text.split()
    if len(words) < k:
        chars = text.replace(" ", "")
        return {chars[i:i+k] for i in range(max(0, len(chars) - k + 1))}
    return {" ".join(words[i:i+k]) for i in range(len(words) - k + 1)}


def _word_ngrams(text: str, n: int = 4) -> Set[str]:
    words = re.sub(r"[^\w\s]", "", text).split()
    return {" ".join(words[i:i+n]) for i in range(len(words) - n + 1)}


def _jaccard(a: Set, b: Set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _fingerprint(normalized: str) -> str:
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class PostSimilarityResult:
    slug: str
    title: str
    created_at: str
    is_blocked: bool = False
    block_reason: str = ""
    warnings: List[str] = field(default_factory=list)
    # {other_slug: jaccard_score}
    top_matches: Dict[str, float] = field(default_factory=dict)
    structural_score: float = 0.0
    skipped: bool = False
    skip_reason: str = ""
    # The slug of the most-similar other post — used as the cluster key
    # when --retain-newest groups duplicates to keep only the newest.
    closest_match_slug: str = ""


# ── Post loader ───────────────────────────────────────────────────────────────

def load_posts(docs_dir: Path) -> List[Tuple[str, Dict]]:
    """Return (slug, post_data) sorted oldest-first."""
    results = []
    for post_dir in docs_dir.iterdir():
        if not post_dir.is_dir() or post_dir.name in ("static", "tag", "author"):
            continue
        post_json = post_dir / "post.json"
        if not post_json.exists():
            continue
        try:
            with open(post_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            results.append((post_dir.name, data))
        except Exception as e:
            print(f"  ⚠️  Could not load {post_dir.name}/post.json: {e}")
    results.sort(key=lambda x: x[1].get("created_at", "1970-01-01"))
    return results


# ── Index builder ─────────────────────────────────────────────────────────────

def build_index(posts: List[Tuple[str, Dict]]) -> Dict[str, dict]:
    """
    Build an in-memory similarity index.
    {slug: {fingerprint, shingles, title, created_at}}
    """
    index: Dict[str, dict] = {}
    for slug, data in posts:
        content = data.get("content", "")
        if len(content) < MIN_CHARS_TO_CHECK:
            continue
        normalized = _normalize(content)
        fp = _fingerprint(normalized)
        shingle_set = _shingles(normalized, k=5)
        index[slug] = {
            "fingerprint": fp,
            "shingles":    shingle_set,
            "title":       data.get("title", slug),
            "created_at":  data.get("created_at", "")[:10],
        }
    return index


# ── Structural repetition ─────────────────────────────────────────────────────

def _structural_score(
    normalized_candidate: str,
    index: Dict[str, dict],
    exclude_slug: str,
) -> float:
    """
    Fraction of 4-grams in the candidate that appear in the union of
    all OTHER indexed posts. High = templated LLM writing.
    """
    candidate_ngrams = _word_ngrams(normalized_candidate, n=4)
    if not candidate_ngrams:
        return 0.0

    all_ngrams: Set[str] = set()
    for slug, entry in index.items():
        if slug == exclude_slug:
            continue
        for shingle in entry.get("shingles", set()):
            words = shingle.split() if isinstance(shingle, str) else []
            if len(words) >= 4:
                all_ngrams.add(" ".join(words[:4]))

    if not all_ngrams:
        return 0.0

    overlap = len(candidate_ngrams & all_ngrams)
    return overlap / len(candidate_ngrams)


# ── Per-post check ────────────────────────────────────────────────────────────

def check_post(
    slug: str,
    data: Dict,
    index: Dict[str, dict],
    block_threshold: float,
    warn_threshold: float,
    check_structural: bool,
) -> PostSimilarityResult:
    title = data.get("title", slug)
    created_at = data.get("created_at", "")[:10]
    result = PostSimilarityResult(
        slug=slug, title=title, created_at=created_at)

    content = data.get("content", "")
    if len(content) < MIN_CHARS_TO_CHECK:
        result.skipped = True
        result.skip_reason = (
            f"Content too short ({len(content)} chars < {MIN_CHARS_TO_CHECK})"
        )
        return result

    normalized = _normalize(content)
    candidate_fp = _fingerprint(normalized)
    candidate_shingles = _shingles(normalized, k=5)

    best_score = 0.0
    best_match_slug = ""

    for other_slug, entry in index.items():
        if other_slug == slug:
            continue

        # Exact duplicate
        if entry["fingerprint"] == candidate_fp:
            result.is_blocked = True
            result.block_reason = (
                f"EXACT DUPLICATE of '{entry['title']}' (/{other_slug}/)"
            )
            result.closest_match_slug = other_slug
            return result

        score = _jaccard(candidate_shingles, entry["shingles"])
        if score > 0.01:
            result.top_matches[other_slug] = round(score, 4)

        # Track the single best match for cluster grouping
        if score > best_score:
            best_score = score
            best_match_slug = other_slug

        if score >= block_threshold:
            result.is_blocked = True
            result.block_reason = (
                f"{score:.1%} similar to '{entry['title']}' (/{other_slug}/) "
                f"— exceeds block threshold {block_threshold:.0%}"
            )
            # Don't break early; finish scanning so top_matches is complete

        elif score >= warn_threshold:
            result.warnings.append(
                f"{score:.1%} similarity with '{entry['title']}' (/{other_slug}/)"
            )

    # Record the closest match slug for cluster-based retention logic
    result.closest_match_slug = best_match_slug

    # Keep only the top 5 matches for readability
    result.top_matches = dict(
        sorted(result.top_matches.items(), key=lambda x: -x[1])[:5]
    )

    if check_structural:
        s_score = _structural_score(normalized, index, exclude_slug=slug)
        result.structural_score = round(s_score, 4)
        if s_score >= STRUCTURAL_WARN_THRESHOLD:
            result.warnings.append(
                f"High structural repetition score {s_score:.1%} "
                f"— post may be templated (LLM re-using sentence patterns)"
            )

    return result


# ── Report printer ────────────────────────────────────────────────────────────

def _icon(r: PostSimilarityResult) -> str:
    if r.skipped:
        return "⏭ "
    if r.is_blocked:
        return "🛑"
    if r.warnings:
        return "⚠️ "
    return "✅"


def print_report(
    results: List[PostSimilarityResult],
    failures_only: bool = False,
    verbose: bool = True,
) -> None:
    SEP = "─" * 72

    pass_count = sum(
        1 for r in results if not r.is_blocked and not r.warnings and not r.skipped)
    warn_count = sum(1 for r in results if r.warnings and not r.is_blocked)
    blocked_count = sum(1 for r in results if r.is_blocked)
    skipped_count = sum(1 for r in results if r.skipped)

    print(f"\n{SEP}")
    print(f"Similarity Guard — {len(results)} post(s) checked")
    print(
        f"  ✅ Clean: {pass_count}  |  ⚠️  Warnings: {warn_count}  "
        f"|  🛑 Blocked: {blocked_count}  |  ⏭  Skipped: {skipped_count}"
    )
    print(SEP)

    for r in results:
        icon = _icon(r)
        if failures_only and not r.is_blocked and not r.warnings:
            continue

        print(f"\n{icon}  [{r.created_at}]  {r.title[:65]}")
        if verbose:
            print(f"     slug: {r.slug}")

        if r.skipped:
            print(f"     skipped: {r.skip_reason}")
            continue

        if r.is_blocked:
            print(f"     BLOCKED: {r.block_reason}")

        if r.warnings:
            for w in r.warnings:
                print(f"     ⚠  {w}")

        if verbose and r.top_matches:
            top = list(r.top_matches.items())[:3]
            pairs = "  |  ".join(f"/{s}/: {v:.1%}" for s, v in top)
            print(f"     top matches → {pairs}")

        if verbose and r.structural_score > 0:
            print(f"     structural repetition: {r.structural_score:.1%}")

    print(f"\n{SEP}")

    # Blocked list summary
    blocked = [r for r in results if r.is_blocked]
    if blocked:
        print(f"\n🛑  BLOCKED POSTS ({len(blocked)}) — oldest first:")
        for r in blocked:
            print(f"  [{r.created_at}]  {r.slug}")
            print(f"           {r.block_reason}")
    print()


# ── Cluster builder ───────────────────────────────────────────────────────────

def _build_clusters(blocked: List[PostSimilarityResult]) -> Dict[str, List[PostSimilarityResult]]:
    """
    Group blocked posts into duplicate clusters using Union-Find so that
    transitive duplicates (A≈B and B≈C) end up in the same cluster.

    Returns {canonical_cluster_key: [PostSimilarityResult, ...]}
    where each list is sorted oldest-first (same order as input).

    The canonical key is the slug of the NEWEST post in the cluster —
    that is the one we want to RETAIN.
    """
    # Build parent map for Union-Find
    all_slugs = [r.slug for r in blocked]
    parent: Dict[str, str] = {s: s for s in all_slugs}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]   # path compression
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Union every blocked post with its closest match (if that match is
    # also in the blocked set — i.e. both sides of the pair are blocked)
    blocked_slugs = set(all_slugs)
    for r in blocked:
        if r.closest_match_slug and r.closest_match_slug in blocked_slugs:
            union(r.slug, r.closest_match_slug)

    # Group by root
    groups: Dict[str, List[PostSimilarityResult]] = {}
    for r in blocked:
        root = find(r.slug)
        groups.setdefault(root, []).append(r)

    # Re-key each group by the slug of its newest member so the canonical
    # key always identifies the post we are KEEPING.
    keyed: Dict[str, List[PostSimilarityResult]] = {}
    for members in groups.values():
        # newest = last in oldest-first list
        newest_slug = members[-1].slug
        keyed[newest_slug] = members

    return keyed


# ── Deletion ──────────────────────────────────────────────────────────────────

def delete_blocked_posts(
    results: List[PostSimilarityResult],
    docs_dir: Path,
    count,                  # int or "all"
    retain_newest: bool,    # if True, keep the newest post in each cluster
    dry_run: bool = True,
    yes: bool = False,
) -> None:
    """
    Delete blocked posts.

    retain_newest=True  →  for every duplicate cluster, spare the newest post
                           and delete only the older copies.
    retain_newest=False →  delete all blocked posts up to `count`, oldest first.
    """
    blocked = [r for r in results if r.is_blocked]

    if not blocked:
        print("✅  No blocked posts to delete.")
        return

    SEP = "─" * 72

    if retain_newest:
        # ── Cluster mode: keep newest, delete all others ──────────────────────
        clusters = _build_clusters(blocked)
        to_delete = []
        to_retain = []

        for newest_slug, members in clusters.items():
            # members is oldest-first; the last entry is the newest
            to_retain.append(members[-1])
            to_delete.extend(members[:-1])   # everything except the newest

        # Still respect the numeric cap when retain_newest + a number are combined
        if count != "all":
            to_delete = to_delete[:int(count)]

        label = "DRY RUN — " if dry_run else ""
        print(f"\n{SEP}")
        print(
            f"{label}Cluster-aware deletion  "
            f"({len(to_delete)} delete  |  {len(to_retain)} retain)"
        )
        print(SEP)

        if to_retain:
            print(f"\n  🟢  RETAINED (newest in each cluster):")
            for r in to_retain:
                print(f"       [{r.created_at}]  {r.title[:60]}")
                print(f"                slug: {r.slug}")

        if to_delete:
            print(f"\n  🗑   QUEUED FOR DELETION (older duplicates):")
            for r in to_delete:
                print(f"       [{r.created_at}]  {r.title[:60]}")
                print(f"                slug  : {r.slug}")
                print(f"                reason: {r.block_reason}")

    else:
        # ── Simple mode: delete oldest N blocked posts ────────────────────────
        if count == "all":
            to_delete = blocked
        else:
            to_delete = blocked[:int(count)]

        label = "DRY RUN — " if dry_run else ""
        print(f"\n{SEP}")
        print(
            f"{label}Posts queued for deletion "
            f"({len(to_delete)} of {len(blocked)} blocked):"
        )
        print(SEP)

        for r in to_delete:
            print(f"  🗑  [{r.created_at}]  {r.title[:60]}")
            print(f"       slug   : {r.slug}")
            print(f"       reason : {r.block_reason}")

    if not to_delete:
        print("\n  Nothing to delete after applying retention rules.")
        return

    if dry_run:
        print(f"\n  Dry run complete — nothing deleted yet.")
        if not yes:
            answer = input(
                f"\n  Proceed with deleting {len(to_delete)} post(s)? [y/N] "
            ).strip().lower()
            if answer not in ("y", "yes"):
                print("  Aborted.")
                return
        # Re-run as real delete (pass same retain_newest so logic is consistent)
        delete_blocked_posts(
            results, docs_dir, count,
            retain_newest=retain_newest,
            dry_run=False,
            yes=True,
        )
        return

    # ── Actual filesystem deletion ────────────────────────────────────────────
    deleted = 0
    for r in to_delete:
        post_dir = docs_dir / r.slug
        if post_dir.exists():
            shutil.rmtree(post_dir)
            print(f"  Deleted: {post_dir}")
            deleted += 1
        else:
            print(f"  ⚠️  Already gone: {post_dir}")

    print(
        f"\n  Deleted {deleted} post director{'y' if deleted == 1 else 'ies'}.")
    print(f"  Run 'python blog_system.py build' to regenerate the site.\n")


# ── Pair matrix (optional deep-dive) ─────────────────────────────────────────

def print_similarity_matrix(
    results: List[PostSimilarityResult],
    warn_threshold: float,
) -> None:
    """Print every cross-post pair that exceeds warn_threshold."""
    seen:  set = set()
    pairs: list = []
    for r in results:
        for other_slug, score in r.top_matches.items():
            key = tuple(sorted([r.slug, other_slug]))
            if key in seen or score < warn_threshold:
                continue
            seen.add(key)
            pairs.append((score, r.slug, other_slug))

    if not pairs:
        print("No cross-post pairs exceed the warn threshold.")
        return

    pairs.sort(reverse=True)
    print(f"\n{'─'*72}")
    print(f"Cross-post similarity pairs (≥ {warn_threshold:.0%}):")
    print(f"{'─'*72}")
    for score, slug_a, slug_b in pairs:
        print(f"  {score:.1%}   /{slug_a}/")
        print(f"         /{slug_b}/")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Check all posts in docs/ for duplicate/similar content.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--docs", default="./docs",
        help="Path to docs directory (default: ./docs)",
    )
    parser.add_argument(
        "--block-threshold", type=float, default=DEFAULT_BLOCK_THRESHOLD,
        help=f"Jaccard score that blocks a post (default: {DEFAULT_BLOCK_THRESHOLD})",
    )
    parser.add_argument(
        "--warn-threshold", type=float, default=DEFAULT_WARN_THRESHOLD,
        help=f"Jaccard score that warns (default: {DEFAULT_WARN_THRESHOLD})",
    )
    parser.add_argument(
        "--structural", action="store_true",
        help="Also run structural repetition detection (Layer 2)",
    )
    parser.add_argument(
        "--matrix", action="store_true",
        help="Print all cross-post similarity pairs above warn threshold",
    )
    parser.add_argument(
        "--failures-only", action="store_true",
        help="Only print posts with warnings or blocks",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-post detail (summary + failures only)",
    )
    parser.add_argument(
        "--delete", metavar="N|all", default=None,
        help="Delete blocked posts: oldest N first, or 'all'",
    )
    parser.add_argument(
        "--retain-newest", action="store_true",
        help=(
            "When deleting, keep the newest post in each duplicate cluster "
            "and remove only the older copies. "
            "Pairs naturally with --delete all."
        ),
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="Skip the deletion confirmation prompt",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    docs_dir = Path(args.docs).resolve()

    if not docs_dir.exists():
        print(f"Error: docs directory not found: {docs_dir}")
        sys.exit(1)

    print(f"Docs dir         : {docs_dir}")
    print(f"Block threshold  : {args.block_threshold:.0%}")
    print(f"Warn threshold   : {args.warn_threshold:.0%}")
    print(f"Structural check : {'on' if args.structural else 'off'} "
          f"(use --structural to enable)")
    if args.retain_newest:
        print(f"Retain newest    : on — newest post per cluster will be spared")

    # ── Load & index ──────────────────────────────────────────────────────────
    posts = load_posts(docs_dir)
    if not posts:
        print(f"\nNo posts found in {docs_dir}")
        sys.exit(0)

    print(f"Posts found      : {len(posts)}\n")
    print("Building similarity index...", end=" ", flush=True)
    index = build_index(posts)
    print(f"indexed {len(index)} post(s).")

    # ── Run checks ────────────────────────────────────────────────────────────
    results: List[PostSimilarityResult] = []
    for slug, data in posts:
        r = check_post(
            slug=slug,
            data=data,
            index=index,
            block_threshold=args.block_threshold,
            warn_threshold=args.warn_threshold,
            check_structural=args.structural,
        )
        results.append(r)

    # ── Report ────────────────────────────────────────────────────────────────
    print_report(
        results,
        failures_only=args.failures_only,
        verbose=not args.quiet,
    )

    if args.matrix:
        print_similarity_matrix(results, warn_threshold=args.warn_threshold)

    # ── Deletion ──────────────────────────────────────────────────────────────
    if args.delete is not None:
        raw = args.delete.strip().lower()
        if raw == "all":
            count: object = "all"
        else:
            try:
                count = int(raw)
                if count <= 0:
                    raise ValueError
            except ValueError:
                print(
                    f"Error: --delete must be a positive integer or 'all', "
                    f"got: {args.delete!r}"
                )
                sys.exit(1)

        delete_blocked_posts(
            results,
            docs_dir,
            count=count,
            retain_newest=args.retain_newest,
            dry_run=True,
            yes=args.yes,
        )

    # Exit 1 if any blocked posts remain and deletion was not requested
    blocked_remaining = [r for r in results if r.is_blocked]
    if blocked_remaining and args.delete is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
