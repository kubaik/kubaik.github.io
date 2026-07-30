#!/usr/bin/env python3
"""
content_quality_scanner.py

Local (offline) tool for the kubaik.github.io AI-blog repo.

WHAT IT DOES
------------
1. Scans every docs/<slug>/post.json in the repo (the canonical per-post
   data record written by static_site_generator.py).
2. Groups posts by the month they were published (created_at).
3. Scores each post on a 0-100 quality scale using signals that matter for
   AdSense approval / SEO, weighted heavily toward the biggest real risk in
   this repo: near-duplicate / "programmatic doorway page" content, not
   just word count.
4. Prints the N lowest-scoring posts per month (a report), and, only when
   explicitly asked, moves the losing posts to a local backup folder and
   removes them from docs/ so they stop being published/indexed.

IT NEVER DELETES ANYTHING BY DEFAULT.
--------------------------------------
- Default mode is `report`: read-only, prints a ranked table, writes a CSV.
- `prune` mode requires --confirm and always backs up the full post
  directory (index.html, post.json, images, etc.) to
  .quality_review_backups/<timestamp>/<slug>/ before removing it from docs/,
  so removal is reversible (git also still has history if committed).
- After a prune, re-run the site build (`python blog_system.py build`) so
  the sitemap, RSS feed, homepage, tag pages, and related-posts links are
  regenerated without the removed posts.

SCORING (higher = better, 0-100)
---------------------------------
Weighted composite of:
  - duplication_score   (35%) - 1 - max TF-IDF cosine similarity to any
                                 other post published within a lookback
                                 window. Near-duplicate posts (the
                                 "postgres-17-replaced-X / -replaces-Y /
                                 -buried-Z" pattern in this repo) score low.
  - depth_score          (25%) - word count relative to the corpus median,
                                 penalizing thin posts heavily below 1200
                                 words.
  - meta_score           (15%) - meta_description length/quality (empty,
                                 too short <50 chars, too long >170 chars,
                                 or a generic boilerplate opener all score
                                 low).
  - structure_score      (10%) - presence of code blocks / tables (has_code,
                                 has_table) as a light signal of substantive,
                                 non-listicle content. Not applicable to
                                 every post, so it's a small weight.
  - affiliate_density    (15%) - affiliate_count and ad_slots relative to
                                 word count. Posts that are unusually
                                 ad/affiliate-heavy relative to their length
                                 read as low-quality/spammy to reviewers.

Within each calendar month, the posts with the LOWEST composite score are
the "least quality content per month" the tool flags.

USAGE
-----
    # Read-only report, top 5 worst posts per month, written to CSV
    python content_quality_scanner.py report

    # Same, but only show the worst 3 per month and only for 2026-06
    python content_quality_scanner.py report --per-month 3 --month 2026-06

    # Dry run of a prune (shows exactly what WOULD be deleted, deletes nothing)
    python content_quality_scanner.py prune --per-month 3

    # Actually prune (backs up first, then removes from docs/)
    python content_quality_scanner.py prune --per-month 3 --confirm

    # Prune but exclude specific slugs you want to keep no matter what
    python content_quality_scanner.py prune --per-month 3 --confirm \\
        --exclude my-best-post-slug --exclude another-slug

    # Dry run: find and preview fixes for broken meta_description fields
    # (too long/short/missing/generic) instead of deleting those posts
    python content_quality_scanner.py fix-meta

    # Actually apply the fixes (backs up post.json + index.html first)
    python content_quality_scanner.py fix-meta --confirm

    # Only fix one month, with custom length thresholds
    python content_quality_scanner.py fix-meta --month 2026-06 \\
        --min-len 60 --max-len 155 --confirm

Lives in utils/ and defaults to the repo's docs/ and backup folders
regardless of what directory you run it from (same convention as
utils/deduplicate_posts.py and utils/delete_similar_posts.py). Can still be
run from anywhere:

    python utils/content_quality_scanner.py report --per-month 5
    cd utils && python content_quality_scanner.py report --per-month 5

Both work identically.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Resolve paths relative to the repo root (parent of utils/), not the
# current working directory, so this script works whether you run it as
# `python utils/content_quality_scanner.py` from the repo root or
# `python content_quality_scanner.py` from inside utils/. Matches the
# convention already used by utils/deduplicate_posts.py and
# utils/delete_similar_posts.py.
REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
BACKUP_ROOT = REPO_ROOT / ".quality_review_backups"

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for",
    "with", "is", "are", "was", "were", "be", "been", "being", "this",
    "that", "it", "its", "as", "at", "by", "from", "we", "you", "your",
    "i", "our", "will", "what", "how", "why", "when", "which", "these",
    "those", "not", "no", "do", "does", "did", "can", "could", "should",
    "would", "into", "than", "then", "so", "if", "just", "more", "most",
}

WEAK_META_OPENERS = (
    "this post", "in this article", "a guide to", "learn about",
    "an overview", "this tutorial", "this article", "we will",
    "you will learn",
)

# Practical Google snippet guideline: descriptions much beyond ~155-160
# chars get truncated in the SERP, and under ~50 chars reads as thin/lazy.
# These are the targets used by `fix-meta` mode (independent of the
# slightly looser 170-char threshold used in the report's meta_score, which
# only needs to catch outliers, not enforce an exact snippet length).
META_MIN_LEN = 50
META_MAX_LEN = 160
META_IDEAL_MIN = 70


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class Post:
    slug: str
    path: Path
    title: str = ""
    content: str = ""
    meta_description: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: Optional[datetime] = None
    word_count: int = 0
    has_code: bool = False
    has_table: bool = False
    affiliate_count: int = 0
    ad_slots: int = 0

    # filled in after corpus-wide analysis
    max_similarity: float = 0.0
    most_similar_slug: str = ""
    score: float = 0.0
    subscores: Dict[str, float] = field(default_factory=dict)

    @property
    def month_key(self) -> str:
        if not self.created_at:
            return "unknown"
        return f"{self.created_at.year:04d}-{self.created_at.month:02d}"


def _parse_date(raw: str) -> Optional[datetime]:
    if not raw:
        return None
    raw = raw.strip()
    for candidate in (raw, raw.replace("Z", "")):
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            continue
    # last resort: grab a leading YYYY-MM-DD
    m = re.match(r"(\d{4}-\d{2}-\d{2})", raw)
    if m:
        try:
            return datetime.fromisoformat(m.group(1))
        except ValueError:
            return None
    return None


def load_posts(docs_dir: Path) -> List[Post]:
    posts = []
    if not docs_dir.exists():
        print(
            f"ERROR: {docs_dir} does not exist. Run this from the repo root.")
        sys.exit(1)

    for post_dir in sorted(docs_dir.iterdir()):
        if not post_dir.is_dir():
            continue
        post_json = post_dir / "post.json"
        if not post_json.exists():
            # No post.json means it's a stub/redirect page (already merged
            # into a canonical post) or a non-post directory (e.g. "static",
            # "author", "about", "tag", "page"). Skip either way -- nothing
            # to score or safely delete here.
            continue
        try:
            data = json.loads(post_json.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  WARN: could not parse {post_json}: {e}")
            continue

        monetization = data.get("monetization_data") or {}
        posts.append(
            Post(
                slug=data.get("slug", post_dir.name),
                path=post_dir,
                title=data.get("title", ""),
                content=data.get("content", "") or "",
                meta_description=data.get("meta_description", "") or "",
                tags=data.get("tags", []) or [],
                created_at=_parse_date(data.get("created_at", "")),
                word_count=data.get("word_count") or len(
                    (data.get("content") or "").split()),
                has_code=bool(data.get("has_code")),
                has_table=bool(data.get("has_table")),
                affiliate_count=monetization.get("affiliate_count", len(
                    data.get("affiliate_links", []) or [])),
                ad_slots=monetization.get("ad_slots", 0),
            )
        )
    return posts


# --------------------------------------------------------------------------
# Duplication detection (TF-IDF cosine similarity, no external deps)
# --------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def _build_tfidf(posts: List[Post]) -> Dict[str, Dict[str, float]]:
    """Return {slug: {term: tfidf_weight}} using title + content."""
    doc_tokens: Dict[str, List[str]] = {}
    for p in posts:
        # Title words count extra (x3) since duplicate framing is very
        # visible in titles for this repo's clusters.
        doc_tokens[p.slug] = _tokenize(p.title) * 3 + _tokenize(p.content)

    df = Counter()
    for tokens in doc_tokens.values():
        for term in set(tokens):
            df[term] += 1

    n_docs = max(len(posts), 1)
    tfidf: Dict[str, Dict[str, float]] = {}
    for slug, tokens in doc_tokens.items():
        tf = Counter(tokens)
        total = max(sum(tf.values()), 1)
        weights = {}
        for term, count in tf.items():
            idf = math.log(n_docs / (1 + df[term])) + 1
            weights[term] = (count / total) * idf
        tfidf[slug] = weights
    return tfidf


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[t] * b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_similarities(posts: List[Post]) -> None:
    """Fill in max_similarity / most_similar_slug for every post in place."""
    if len(posts) < 2:
        return
    tfidf = _build_tfidf(posts)
    slugs = [p.slug for p in posts]
    n = len(slugs)
    print(f"Computing pairwise similarity across {n} posts "
          f"({n * (n - 1) // 2} pairs)... this may take a moment.")

    by_slug = {p.slug: p for p in posts}
    for i in range(n):
        si = slugs[i]
        best_sim, best_slug = 0.0, ""
        vi = tfidf[si]
        for j in range(n):
            if i == j:
                continue
            sj = slugs[j]
            sim = _cosine(vi, tfidf[sj])
            if sim > best_sim:
                best_sim, best_slug = sim, sj
        by_slug[si].max_similarity = best_sim
        by_slug[si].most_similar_slug = best_slug


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def score_posts(posts: List[Post]) -> None:
    if not posts:
        return
    word_counts = sorted(p.word_count for p in posts)
    median_wc = word_counts[len(word_counts) // 2] or 1

    for p in posts:
        # 1. Duplication (35%) - lower similarity to nearest neighbor = better
        duplication_score = max(0.0, 1.0 - p.max_similarity) * 100

        # 2. Depth (25%) - relative to corpus median, hard penalty <1200 words
        if p.word_count <= 0:
            depth_score = 0.0
        else:
            ratio = p.word_count / median_wc
            depth_score = min(100.0, ratio * 100)
            if p.word_count < 1200:
                depth_score *= 0.4  # hard penalty for thin content

        # 3. Meta description quality (15%)
        meta = p.meta_description.strip()
        if not meta:
            meta_score = 0.0
        elif len(meta) < 50 or len(meta) > 170:
            meta_score = 40.0
        elif any(meta.lower().startswith(w) for w in WEAK_META_OPENERS):
            meta_score = 50.0
        else:
            meta_score = 100.0

        # 4. Structure (10%) - light bonus, not a penalty if absent
        structure_score = 50.0
        if p.has_code:
            structure_score += 25.0
        if p.has_table:
            structure_score += 25.0
        structure_score = min(structure_score, 100.0)

        # 5. Affiliate/ad density (15%) - penalize high ad/affiliate count
        #    relative to length (spammy-per-word ratio)
        per_1000_words = (p.word_count / 1000.0) or 1.0
        affiliate_ratio = p.affiliate_count / per_1000_words
        ad_ratio = p.ad_slots / per_1000_words
        density_penalty = affiliate_ratio * 15 + max(0.0, ad_ratio - 2) * 10
        affiliate_density_score = max(0.0, 100.0 - density_penalty)

        composite = (
            duplication_score * 0.35
            + depth_score * 0.25
            + meta_score * 0.15
            + structure_score * 0.10
            + affiliate_density_score * 0.15
        )

        p.subscores = {
            "duplication": round(duplication_score, 1),
            "depth": round(depth_score, 1),
            "meta": round(meta_score, 1),
            "structure": round(structure_score, 1),
            "affiliate_density": round(affiliate_density_score, 1),
        }
        p.score = round(composite, 1)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def group_by_month(posts: List[Post]) -> Dict[str, List[Post]]:
    groups: Dict[str, List[Post]] = defaultdict(list)
    for p in posts:
        groups[p.month_key].append(p)
    return dict(sorted(groups.items()))


def print_report(groups: Dict[str, List[Post]], per_month: int, csv_path: Optional[Path]) -> List[Post]:
    flagged: List[Post] = []
    csv_rows = []

    for month, month_posts in groups.items():
        ranked = sorted(month_posts, key=lambda p: p.score)
        worst = ranked[:per_month]
        flagged.extend(worst)

        print(f"\n=== {month} ({len(month_posts)} posts) ===")
        print(
            f"{'SCORE':>6}  {'DUP':>5} {'DEPTH':>5} {'META':>5} {'STRUCT':>6} {'ADS':>5}  SLUG")
        for p in worst:
            s = p.subscores
            print(f"{p.score:6.1f}  {s['duplication']:5.1f} {s['depth']:5.1f} "
                  f"{s['meta']:5.1f} {s['structure']:6.1f} {s['affiliate_density']:5.1f}  {p.slug}")
            if p.most_similar_slug:
                print(f"        (most similar to: {p.most_similar_slug}, "
                      f"similarity={p.max_similarity:.2f})")
            csv_rows.append({
                "month": month,
                "slug": p.slug,
                "score": p.score,
                "word_count": p.word_count,
                "duplication_score": s["duplication"],
                "depth_score": s["depth"],
                "meta_score": s["meta"],
                "structure_score": s["structure"],
                "affiliate_density_score": s["affiliate_density"],
                "most_similar_slug": p.most_similar_slug,
                "max_similarity": round(p.max_similarity, 3),
                "created_at": p.created_at.isoformat() if p.created_at else "",
            })

    if csv_path:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()) if csv_rows else
                                    ["month", "slug", "score"])
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nWrote CSV report to {csv_path}")

    return flagged


# --------------------------------------------------------------------------
# Prune (delete) mode
# --------------------------------------------------------------------------

def prune_posts(flagged: List[Post], confirm: bool, exclude: List[str]) -> None:
    exclude_set = set(exclude)
    to_remove = [p for p in flagged if p.slug not in exclude_set]

    if not to_remove:
        print("\nNothing to prune (all flagged posts were excluded).")
        return

    print(
        f"\n{'PRUNE (LIVE)' if confirm else 'PRUNE (DRY RUN — nothing will be deleted)'}")
    print(f"{len(to_remove)} post(s) selected for removal:\n")
    for p in to_remove:
        print(f"  - {p.slug}  (score={p.score}, month={p.month_key})")

    if not confirm:
        print("\nThis was a dry run. Re-run with --confirm to actually back up "
              "and delete these posts.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = BACKUP_ROOT / timestamp
    backup_dir.mkdir(parents=True, exist_ok=True)

    removed = []
    for p in to_remove:
        if not p.path.exists():
            print(f"  SKIP (already gone): {p.slug}")
            continue
        dest = backup_dir / p.slug
        shutil.copytree(p.path, dest)
        shutil.rmtree(p.path)
        removed.append(p.slug)
        print(f"  Removed: {p.slug}  (backed up to {dest})")

    print(f"\nDone. {len(removed)} post(s) removed from {DOCS_DIR}/.")
    print(
        f"Backups saved under {backup_dir}/ — restore by copying a folder back into {DOCS_DIR}/.")
    print("\nNEXT STEP: regenerate the site so sitemap.xml, rss.xml, the homepage,")
    print("tag pages, and related-posts links no longer reference the removed posts:")
    print("    python blog_system.py build")
    print("Then commit the change (deleted docs/<slug>/ dirs + regenerated site files).")


# --------------------------------------------------------------------------
# Meta-description fixer
# --------------------------------------------------------------------------
#
# Investigating the "META 40.0" scores flagged in the report showed the
# underlying meta_description text is usually fine, well-written, specific
# copy — it's just 170-220 characters, past the ~155-160 char point where
# Google truncates the snippet in search results. That's a much better fix
# than deleting the post: shorten the existing description in place rather
# than losing the whole post over one metadata field.
#
# `fix-meta` mode:
#   - too long (> META_MAX_LEN)  -> truncate at the last full sentence that
#     fits, or the last clause/word boundary as a fallback. Never invents
#     new text for these, since the original wording is already good.
#   - missing / too short (< META_MIN_LEN) -> pull the lead paragraph out of
#     the post body (before the first heading), strip markdown, and trim it
#     to fit. This is an extractive fallback (no external API calls), so
#     it's always worth a human skim afterward.
#   - starts with a generic filler opener ("This post...", "In this
#     article...") -> strip the filler phrase and re-capitalize what's left,
#     falling back to the extractive method if that leaves too little.
#
# Every fix updates BOTH post.json's meta_description AND the three
# corresponding tags in index.html (<meta name="description">,
# og:description, twitter:description), after backing up the original
# post.json + index.html to
#   .quality_review_backups/meta_fixes/<timestamp>/<slug>/
# so it's reversible even after --confirm.

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_MD_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s", re.MULTILINE)
_MD_STRIP_RE = re.compile(r"[`*_>#]|\[([^\]]+)\]\([^)]+\)")


def _strip_markdown_light(text: str) -> str:
    """Strip common markdown syntax for use in an extracted meta description."""
    # Turn [label](url) into just label
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Drop emphasis/heading/quote/code markers
    text = re.sub(r"[`*_>#]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_fallback_description(content: str, max_len: int = META_MAX_LEN) -> str:
    """Pull the lead paragraph (before the first heading) out of the post
    body as a stand-in meta description when one is missing or too thin."""
    lead = _MD_HEADING_RE.split(content, maxsplit=1)[0]
    lead = _strip_markdown_light(lead)
    if not lead:
        # No text before the first heading (rare) -- fall back to whatever
        # the first heading's following paragraph is.
        lead = _strip_markdown_light(content)
    return truncate_description(lead, max_len=max_len)


def truncate_description(desc: str, max_len: int = META_MAX_LEN) -> str:
    """Shorten desc to at most max_len chars without cutting mid-word,
    preferring to cut at the end of a full sentence. Falls back to a
    word-boundary cut with a trailing ellipsis so it reads as intentionally
    trimmed rather than an abrupt/broken sentence."""
    desc = desc.strip()
    if len(desc) <= max_len:
        return desc

    sentences = _SENTENCE_SPLIT_RE.split(desc)
    built = ""
    for sentence in sentences:
        candidate = (built + " " + sentence).strip() if built else sentence
        if len(candidate) <= max_len:
            built = candidate
        else:
            break
    if len(built) >= META_IDEAL_MIN:
        return built

    # No single sentence fit cleanly (or the first one is already too long)
    # -- fall back to a hard word-boundary cut, leaving room for an ellipsis.
    budget = max_len - 1  # reserve 1 char for the ellipsis
    truncated = desc[:budget]
    last_space = truncated.rfind(" ")
    if last_space > META_IDEAL_MIN:
        truncated = truncated[:last_space]
    return truncated.rstrip(" ,;:-") + "…"


def diagnose_meta(post: "Post", min_len: int = META_MIN_LEN, max_len: int = META_MAX_LEN) -> Optional[Dict[str, str]]:
    """Return {"reason": ..., "suggested": ...} if post.meta_description has
    a fixable problem, else None."""
    meta = (post.meta_description or "").strip()

    if not meta:
        return {"reason": "missing", "suggested": extract_fallback_description(post.content, max_len)}

    if len(meta) > max_len:
        return {"reason": f"too long ({len(meta)} chars > {max_len})",
                "suggested": truncate_description(meta, max_len)}

    if len(meta) < min_len:
        # Try extending with the post's lead paragraph first; if that's
        # still not usable, fall back to the extractive method outright.
        extra = extract_fallback_description(post.content, max_len)
        suggested = extra if len(extra) >= META_IDEAL_MIN else meta
        return {"reason": f"too short ({len(meta)} chars < {min_len})", "suggested": suggested}

    lowered = meta.lower()
    for opener in WEAK_META_OPENERS:
        if lowered.startswith(opener):
            remainder = meta[len(opener):].lstrip(" ,:-")
            if remainder:
                remainder = remainder[0].upper() + remainder[1:]
            if len(remainder) >= META_IDEAL_MIN:
                return {"reason": f'generic opener ("{opener}...")', "suggested": remainder}
            return {"reason": f'generic opener ("{opener}...")',
                    "suggested": extract_fallback_description(post.content, max_len)}

    return None


def _html_escape(s: str) -> str:
    """Match Jinja2's default autoescape so we can find/replace the exact
    string as it appears inside an HTML attribute."""
    return (
        s.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def apply_meta_fix(post: "Post", new_meta: str, confirm: bool) -> bool:
    """Update meta_description in post.json and the three matching <meta>
    tags in index.html. Backs up both files first. Returns True on success."""
    post_json_path = post.path / "post.json"
    index_html_path = post.path / "index.html"

    if not post_json_path.exists():
        print(f"  SKIP {post.slug}: post.json not found")
        return False

    if not confirm:
        return True  # dry run: nothing to write, caller already printed the diff

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = BACKUP_ROOT / "meta_fixes" / timestamp / post.slug
    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(post_json_path, backup_dir / "post.json")
    if index_html_path.exists():
        shutil.copy2(index_html_path, backup_dir / "index.html")

    # 1. post.json
    data = json.loads(post_json_path.read_text(encoding="utf-8"))
    old_meta = data.get("meta_description", "")
    data["meta_description"] = new_meta
    post_json_path.write_text(json.dumps(
        data, indent=2, ensure_ascii=False), encoding="utf-8")

    # 2. index.html -- replace the escaped old value in all three tags
    if index_html_path.exists() and old_meta:
        html = index_html_path.read_text(encoding="utf-8")
        old_escaped = _html_escape(old_meta)
        new_escaped = _html_escape(new_meta)
        count = html.count(old_escaped)
        if count == 0:
            print(f"  WARN {post.slug}: old meta_description not found verbatim in index.html "
                  f"(post.json updated; you may need to rebuild the page instead: "
                  f"python blog_system.py build)")
        else:
            html = html.replace(old_escaped, new_escaped)
            index_html_path.write_text(html, encoding="utf-8")

    return True


def run_fix_meta(posts: List[Post], min_len: int, max_len: int, confirm: bool, exclude: List[str]) -> None:
    exclude_set = set(exclude)
    issues = []
    for p in sorted(posts, key=lambda p: (p.month_key, p.slug)):
        if p.slug in exclude_set:
            continue
        diagnosis = diagnose_meta(p, min_len=min_len, max_len=max_len)
        if diagnosis:
            issues.append((p, diagnosis))

    if not issues:
        print("\nNo meta_description issues found in the selected posts.")
        return

    print(f"\n{'FIX-META (LIVE)' if confirm else 'FIX-META (DRY RUN — nothing will be written)'}")
    print(f"{len(issues)} post(s) with a fixable meta_description issue:\n")

    fixed = 0
    for p, diagnosis in issues:
        print(f"  {p.slug}  [{p.month_key}]  — {diagnosis['reason']}")
        print(
            f"    old ({len(p.meta_description or '')} chars): {p.meta_description!r}")
        print(
            f"    new ({len(diagnosis['suggested'])} chars): {diagnosis['suggested']!r}")
        ok = apply_meta_fix(p, diagnosis["suggested"], confirm=confirm)
        if confirm and ok:
            fixed += 1
        print()

    if not confirm:
        print("This was a dry run. Re-run with --confirm to write these changes "
              "(post.json + index.html), after backing up both files.")
    else:
        print(f"Done. Fixed {fixed}/{len(issues)} post(s).")
        print(f"Backups saved under {BACKUP_ROOT / 'meta_fixes'}/.")
        print("\nNEXT STEP: spot-check a few of the extractive fixes (the ones flagged "
              "'missing' or 'too short'), then rebuild so cached pages/sitemap metadata "
              "stay consistent:")
        print("    python blog_system.py build")
        print("Then commit the change.")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["report", "prune", "fix-meta"],
                        help="report = read-only; prune = remove flagged posts; "
                        "fix-meta = shorten/repair meta_description in place")
    parser.add_argument("--per-month", type=int, default=5,
                        help="How many lowest-scoring posts to flag per month (report/prune only, default: 5)")
    parser.add_argument("--month", type=str, default=None,
                        help="Only process one month, e.g. 2026-06")
    parser.add_argument("--csv", type=str, default=str(REPO_ROOT / "quality_report.csv"),
                        help="Path to write the CSV report (report mode). Use '' to skip.")
    parser.add_argument("--confirm", action="store_true",
                        help="Actually write changes (prune deletes / fix-meta rewrites); otherwise dry run")
    parser.add_argument("--exclude", action="append", default=[],
                        help="Slug to skip, even if flagged. Repeatable.")
    parser.add_argument("--docs-dir", type=str, default=str(DOCS_DIR),
                        help=f"Path to the docs/ directory (default: {DOCS_DIR}, resolved from the repo root regardless of cwd)")
    parser.add_argument("--min-len", type=int, default=META_MIN_LEN,
                        help=f"fix-meta only: minimum acceptable meta_description length (default: {META_MIN_LEN})")
    parser.add_argument("--max-len", type=int, default=META_MAX_LEN,
                        help=f"fix-meta only: maximum acceptable meta_description length (default: {META_MAX_LEN})")
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir)
    print(f"Loading posts from {docs_dir}/ ...")
    posts = load_posts(docs_dir)
    print(f"Loaded {len(posts)} posts with post.json.")

    if args.month:
        posts = [p for p in posts if p.month_key == args.month]
        print(f"Filtered to month {args.month}: {len(posts)} posts.")

    if not posts:
        print("No posts to analyze. Exiting.")
        return

    if args.mode == "fix-meta":
        run_fix_meta(posts, min_len=args.min_len, max_len=args.max_len,
                     confirm=args.confirm, exclude=args.exclude)
        return

    compute_similarities(posts)
    score_posts(posts)
    groups = group_by_month(posts)

    csv_path = Path(args.csv) if args.csv else None
    flagged = print_report(groups, args.per_month, csv_path)

    if args.mode == "prune":
        prune_posts(flagged, confirm=args.confirm, exclude=args.exclude)


if __name__ == "__main__":
    main()
