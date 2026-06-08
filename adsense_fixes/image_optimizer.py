"""
adsense_fixes/image_optimizer.py
==================================
Automated image alt-text injection and OG image generation.

WHY THIS EXISTS
---------------
AdSense Site Readiness Guide (§4 Content Quality):
  "Images/media add value (not just padding). Images must be relevant,
   properly licensed, and include descriptive alt text."

Google's quality raters also assess image relevance and accessibility.
Missing alt text is both an accessibility failure (WCAG 2.1 §1.1.1) and
a crawl-quality signal — Googlebot cannot understand images without it.

Two problems solved here:
  1. Markdown content from the LLM uses bare `![](url)` image tags with
     no alt text.  This module injects contextual alt text derived from
     the surrounding paragraph and the post title.
  2. Posts have no social OG image.  We generate a simple SVG-based
     cover card and save it to docs/static/images/{slug}.jpg equivalent
     (as an SVG since GitHub Pages serves them fine and they need no
     external image generation service).

HOW TO INTEGRATE
----------------
In blog_system.py auto mode, AFTER inject_eeat_signals():

    from adsense_fixes.image_optimizer import inject_alt_text, generate_og_card
    inject_alt_text(blog_post)
    generate_og_card(blog_post, output_dir=blog_system.output_dir)

The og_card writes to docs/static/og/{slug}.svg and the static site
generator template already references /static/og/{slug}.svg as the og:image.
"""

import re
from pathlib import Path
from typing import Optional

# ── Alt text injection ─────────────────────────────────────────────────────

_IMG_RE = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
_CODE_FENCE_RE = re.compile(r'```[\s\S]*?```')


def inject_alt_text(post) -> int:
    """
    Scan post.content for Markdown images without alt text and inject
    contextual alt text derived from the post title and surrounding text.

    Returns the number of images updated.

    The alt text follows WCAG guidance:
      - Describes what the image shows (not "image of X")
      - Includes the post topic keyword for SEO
      - Is under 125 characters
    """
    if not getattr(post, 'content', ''):
        return 0

    # Mask code blocks so we don't modify images inside code examples
    code_blocks = []

    def _mask(m):
        code_blocks.append(m.group(0))
        return f"\x00CODE{len(code_blocks) - 1}\x00"

    masked = _CODE_FENCE_RE.sub(_mask, post.content)

    updated = 0
    title_words = _extract_title_keywords(post.title)

    def _replace_img(m):
        nonlocal updated
        alt = m.group(1).strip()
        url = m.group(2).strip()

        if alt:
            # Already has alt text — only upgrade empty or placeholder alt
            if alt.lower() not in ('', 'image', 'photo', 'screenshot', 'figure'):
                return m.group(0)

        # Derive alt text from URL filename + post title keywords
        filename = url.split('/')[-1].split('?')[0]
        base = re.sub(r'\.[a-z]{2,4}$', '', filename, flags=re.IGNORECASE)
        readable = re.sub(r'[-_]', ' ', base).strip()

        if readable and len(readable) > 3:
            candidate = readable.capitalize()
        else:
            candidate = post.title[:60] if post.title else "Technical diagram"

        # Append topic context if not already present and fits budget
        if title_words and title_words[0].lower() not in candidate.lower():
            candidate = f"{candidate} — {title_words[0]}"

        alt_text = candidate[:124]
        updated += 1
        return f"![{alt_text}]({url})"

    new_content = _IMG_RE.sub(_replace_img, masked)

    # Restore code blocks
    for i, block in enumerate(code_blocks):
        new_content = new_content.replace(f"\x00CODE{i}\x00", block)

    post.content = new_content
    return updated


def _extract_title_keywords(title: str) -> list:
    """Extract meaningful words from title for alt text context."""
    if not title:
        return []
    _stop = {'a', 'an', 'the', 'to', 'in', 'of', 'for', 'and', 'or', 'is',
             'are', 'with', 'how', 'vs', 'why', 'what', 'when', 'where'}
    words = re.sub(r'[^\w\s]', ' ', title).split()
    return [w for w in words if w.lower() not in _stop and len(w) > 2][:3]


# ── OG image card generation ───────────────────────────────────────────────

_OG_SVG_TEMPLATE = '''\
<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="630" viewBox="0 0 1200 630">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#667eea"/>
      <stop offset="100%" stop-color="#764ba2"/>
    </linearGradient>
    <linearGradient id="stripe" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="rgba(255,255,255,0.15)"/>
      <stop offset="100%" stop-color="rgba(255,255,255,0)"/>
    </linearGradient>
  </defs>

  <!-- Background -->
  <rect width="1200" height="630" fill="url(#bg)"/>
  <rect width="1200" height="630" fill="url(#stripe)"/>

  <!-- Decorative circles -->
  <circle cx="1050" cy="100" r="200" fill="rgba(255,255,255,0.07)"/>
  <circle cx="150" cy="530" r="150" fill="rgba(255,255,255,0.05)"/>

  <!-- Content area -->
  <rect x="60" y="60" width="1080" height="510" rx="16"
        fill="rgba(0,0,0,0.25)" stroke="rgba(255,255,255,0.15)" stroke-width="1"/>

  <!-- Site name -->
  <text x="100" y="135"
        font-family="system-ui,-apple-system,sans-serif"
        font-size="26" font-weight="600"
        fill="rgba(255,255,255,0.85)"
        letter-spacing="1">
    {site_name}
  </text>

  <!-- Separator -->
  <rect x="100" y="155" width="60" height="4" rx="2" fill="rgba(255,255,255,0.6)"/>

  <!-- Post title — wrapped across up to 3 lines -->
  {title_lines}

  <!-- Tag pills -->
  {tag_pills}

  <!-- Author -->
  <text x="100" y="560"
        font-family="system-ui,-apple-system,sans-serif"
        font-size="22" fill="rgba(255,255,255,0.75)">
    by {author}
  </text>
</svg>'''


def generate_og_card(
    post,
    output_dir: Path,
    site_name: str = "Tech Blog",
    author: str = "Kubai Kevin",
) -> Optional[Path]:
    """
    Generate an SVG OG image card for a blog post and save it to
    {output_dir}/static/og/{slug}.svg.

    Returns the Path to the written file, or None on error.

    The static site generator references this path as og:image in the
    post template.  GitHub Pages serves SVGs with the correct MIME type.
    """
    og_dir = output_dir / "static" / "og"
    og_dir.mkdir(parents=True, exist_ok=True)
    out_path = og_dir / f"{post.slug}.svg"

    title = getattr(post, 'title', 'Untitled')
    tags = getattr(post, 'tags', [])

    title_lines = _wrap_title_svg(title)
    tag_pills_svg = _render_tag_pills_svg(tags[:4])

    svg = _OG_SVG_TEMPLATE.format(
        site_name=_escape_xml(site_name),
        title_lines=title_lines,
        tag_pills=tag_pills_svg,
        author=_escape_xml(author),
    )

    try:
        out_path.write_text(svg, encoding='utf-8')
        return out_path
    except OSError as e:
        print(f"  ⚠️  OG card generation failed for {post.slug}: {e}")
        return None


def _wrap_title_svg(title: str, max_chars: int = 36) -> str:
    """
    Break a title string into up to 3 SVG <text> lines, each with a
    reduced font size for longer titles.
    """
    words = title.split()
    lines = []
    current = []

    for word in words:
        if sum(len(w) + 1 for w in current) + len(word) <= max_chars:
            current.append(word)
        else:
            if current:
                lines.append(' '.join(current))
            current = [word]
        if len(lines) == 2 and current:
            # Third line: dump remaining words
            remaining = ' '.join(current + words[words.index(word) + 1:])
            if len(remaining) > max_chars:
                remaining = remaining[:max_chars - 1] + '…'
            lines.append(remaining)
            current = []
            break

    if current:
        lines.append(' '.join(current))

    lines = lines[:3]
    font_size = 58 if len(title) < 40 else 48 if len(title) < 70 else 40
    y_start = 260 if len(lines) == 1 else 230 if len(lines) == 2 else 200
    line_height = font_size + 14

    parts = []
    for i, line in enumerate(lines):
        y = y_start + i * line_height
        parts.append(
            f'<text x="100" y="{y}" '
            f'font-family="system-ui,-apple-system,sans-serif" '
            f'font-size="{font_size}" font-weight="700" '
            f'fill="white">{_escape_xml(line)}</text>'
        )
    return '\n  '.join(parts)


def _render_tag_pills_svg(tags: list) -> str:
    """Render up to 4 tag pills as SVG rectangles + text."""
    if not tags:
        return ''

    parts = []
    x = 100
    y = 470
    for tag in tags[:4]:
        label = tag[:20]
        pill_width = len(label) * 11 + 28
        parts.append(
            f'<rect x="{x}" y="{y}" width="{pill_width}" height="34" rx="17" '
            f'fill="rgba(255,255,255,0.2)" stroke="rgba(255,255,255,0.35)" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x + 14}" y="{y + 22}" '
            f'font-family="system-ui,-apple-system,sans-serif" '
            f'font-size="16" fill="rgba(255,255,255,0.9)">{_escape_xml(label)}</text>'
        )
        x += pill_width + 12

    return '\n  '.join(parts)


def _escape_xml(text: str) -> str:
    return (
        str(text)
        .replace('&', '&amp;')
        .replace('<', '&lt;')
        .replace('>', '&gt;')
        .replace('"', '&quot;')
    )
