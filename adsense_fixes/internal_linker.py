"""
adsense_fixes/internal_linker.py
==================================
Automated contextual internal linking for AdSense readiness.

The AdSense guide explicitly requires:
  "Internal linking between posts — Auto-published posts should include
   contextual internal links to related posts. Orphan pages with no
   internal links signal low site quality."

HOW TO INTEGRATE
----------------
1. Import this module in blog_system.py:
       from adsense_fixes.internal_linker import inject_internal_links

2. Call it AFTER inject_eeat_signals() in the auto mode block:
       inject_internal_links(blog_post, existing_posts_index)

   Where `existing_posts_index` is built once per run:
       from adsense_fixes.internal_linker import build_posts_index
       existing_posts_index = build_posts_index(Path('./docs'))

3. The function modifies blog_post.content in-place.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Maximum number of internal links to inject per article.
# 2–4 is the sweet spot: enough to signal site quality, not enough to look spammy.
MAX_INTERNAL_LINKS = 3

# Minimum paragraph length (chars) before we consider it for link injection.
# Short paragraphs (e.g. code block captions) are skipped.
MIN_PARAGRAPH_CHARS = 120

# Minimum keyword overlap for a post to be considered related.
_STOP = {
    "a", "an", "the", "to", "in", "of", "for", "and", "or", "is", "are",
    "with", "how", "your", "my", "our", "its", "on", "at", "by", "from",
    "this", "that", "be", "was", "were", "has", "have", "had", "do", "does",
    "did", "will", "would", "can", "could", "should", "may", "might", "must",
    "not", "but", "if", "when", "then", "than", "so", "as", "it", "its",
    "about", "after", "before", "between", "through", "during", "without",
}


def _tokenise(text: str) -> set:
    words = re.sub(r"[^\w\s]", "", text.lower()).split()
    return {w for w in words if w not in _STOP and len(w) > 3}


def _score_overlap(tokens_a: set, tokens_b: set) -> float:
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def build_posts_index(docs_dir: Path) -> List[Dict]:
    """
    Build a lightweight index of all published posts.
    Returns list of dicts: {title, slug, tokens, url}
    """
    index = []
    if not docs_dir.exists():
        return index
    for post_dir in docs_dir.iterdir():
        if not post_dir.is_dir() or post_dir.name in ("static", "tag", "author"):
            continue
        post_json = post_dir / "post.json"
        if not post_json.exists():
            continue
        try:
            with open(post_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            title = data.get("title", "")
            slug = data.get("slug", post_dir.name)
            keywords = data.get("seo_keywords", []) + data.get("tags", [])
            tokens = _tokenise(title + " " + " ".join(keywords))
            index.append({
                "title": title,
                "slug": slug,
                "tokens": tokens,
            })
        except Exception:
            pass
    return index


def _find_anchor_opportunity(
    paragraph: str,
    candidate_title: str,
    candidate_tokens: set,
) -> Optional[Tuple[str, str]]:
    """
    Find a naturally occurring phrase in `paragraph` that can serve as anchor
    text for a link to the candidate post.

    Returns (original_phrase, anchor_text) or None.
    Strategy:
    1. Look for any 2-4 word substring of the paragraph that overlaps strongly
       with the candidate's keyword tokens.
    2. Prefer noun-phrase patterns (capitalised terms, tech tool names).
    3. Never return a phrase inside backticks or a code block.
    """
    # Skip if paragraph looks like a code reference or heading
    if paragraph.strip().startswith(("#", "```", "|", ">")):
        return None

    # Extract 2–4 word windows from the paragraph
    words = paragraph.split()
    best_phrase = None
    best_score = 0.0

    for n in (3, 2, 4):  # prefer 3-word phrases
        for i in range(len(words) - n + 1):
            window = words[i: i + n]
            # Skip windows that contain markdown punctuation
            raw = " ".join(window)
            if any(c in raw for c in ("*", "_", "`", "[", "]", "(", ")")):
                continue
            phrase_tokens = _tokenise(raw)
            score = _score_overlap(phrase_tokens, candidate_tokens)
            if score > best_score and score >= 0.25:
                best_score = score
                best_phrase = raw

    if best_phrase:
        return (best_phrase, best_phrase)
    return None


def inject_internal_links(post, posts_index: List[Dict], base_path: str = "") -> None:
    """
    Inject contextual internal links into post.content in-place.

    Parameters
    ----------
    post : BlogPost
        The post being prepared for publication. Content is modified in-place.
    posts_index : List[Dict]
        Output of build_posts_index() — the existing published posts.
    base_path : str
        The site base_path from config (e.g. "" for root or "/blog").
    """
    if not posts_index:
        return

    # Tokenise the current post for comparison
    current_tokens = _tokenise(
        post.title + " " + " ".join(post.seo_keywords or []))

    # Rank candidates by overlap score, excluding the post itself
    candidates = []
    for entry in posts_index:
        if entry["slug"] == post.slug:
            continue
        score = _score_overlap(current_tokens, entry["tokens"])
        if score >= 0.10:  # at least 10% token overlap
            candidates.append((score, entry))
    candidates.sort(key=lambda x: x[0], reverse=True)

    if not candidates:
        print(
            f"  ℹ️  No related posts found for internal linking: '{post.title}'")
        return

    top_candidates = candidates[:MAX_INTERNAL_LINKS]

    # Split content into paragraphs (preserve code blocks)
    # We process only markdown paragraphs outside fenced code blocks.
    code_blocks: List[str] = []

    def _mask_code(m):
        code_blocks.append(m.group(0))
        return f"\x00CODE{len(code_blocks) - 1}\x00"

    masked = re.sub(r"```[\s\S]*?```", _mask_code, post.content)

    paragraphs = masked.split("\n\n")
    injected_count = 0
    used_slugs: set = set()

    for idx, para in enumerate(paragraphs):
        if injected_count >= MAX_INTERNAL_LINKS:
            break
        # Skip short, heading, code-placeholder, or already-linked paragraphs
        if len(para) < MIN_PARAGRAPH_CHARS:
            continue
        if para.strip().startswith("#"):
            continue
        if "\x00CODE" in para:
            continue
        if "](http" in para or "](/":  # already has links
            continue

        for score, candidate in top_candidates:
            if candidate["slug"] in used_slugs:
                continue
            result = _find_anchor_opportunity(
                para, candidate["title"], candidate["tokens"])
            if result:
                original_phrase, anchor_text = result
                link_url = f"{base_path}/{candidate['slug']}/"
                linked = f"[{anchor_text}]({link_url})"
                # Only replace first occurrence to avoid duplicate links in one paragraph
                paragraphs[idx] = para.replace(original_phrase, linked, 1)
                used_slugs.add(candidate["slug"])
                injected_count += 1
                print(
                    f"  🔗 Internal link injected: '{anchor_text}' → /{candidate['slug']}/"
                )
                break  # one link per paragraph

    # Reassemble
    new_content = "\n\n".join(paragraphs)

    # Restore code blocks
    for i, block in enumerate(code_blocks):
        new_content = new_content.replace(f"\x00CODE{i}\x00", block)

    post.content = new_content

    if injected_count == 0:
        print(
            f"  ℹ️  No anchor opportunities found for internal links in '{post.title}'")
    else:
        print(
            f"  ✅ {injected_count} internal link(s) injected into '{post.title}'")
