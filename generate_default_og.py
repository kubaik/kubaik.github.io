#!/usr/bin/env python3
"""
generate_default_og.py
-----------------------
Generates docs/static/og-default.png — a branded 1200x630 fallback
Open Graph / Twitter card image for any page that doesn't have its own
per-post image (tag/topic pages, the 404 page, category pages, etc).

Previously, _generate_tag_pages() in static_site_generator.py referenced
"{base_url}/static/og-default.png" as the og:image / twitter:image for
every tag archive page, but nothing in the codebase ever generated that
file. Every tag page (dozens of them, across 800+ posts) was shipping a
broken image reference — social shares and link-preview bots got no
thumbnail at all.

This reuses the same gradient identity as generate_icons.py so every
social card across the site (per-post cards, PWA icons, this default
card) looks like it belongs to the same brand.

Usage:
    pip install Pillow
    python generate_default_og.py

Output: docs/static/og-default.png
"""

from pathlib import Path

OUTPUT_PATH = Path("docs/static/og-default.png")

# Brand colours — matches generate_icons.py's indigo -> violet gradient
GRAD_START = (102, 126, 234)   # #667eea
GRAD_END = (118, 75, 162)      # #764ba2
TEXT_COLOUR = (255, 255, 255)
SUBTEXT_COLOUR = (230, 230, 250)

# Standard OG/Twitter large-card dimensions
OG_WIDTH = 1200
OG_HEIGHT = 630

_FONT_CANDIDATES_BOLD = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)
_FONT_CANDIDATES_REGULAR = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def lerp_colour(c1, c2, t):
    """Linear interpolate between two RGB tuples."""
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def _load_font(size: int, bold: bool = False):
    from PIL import ImageFont

    candidates = _FONT_CANDIDATES_BOLD if bold else _FONT_CANDIDATES_REGULAR
    for candidate in candidates:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _wrap_text(draw, text: str, font, max_width: int) -> list:
    """Greedy word-wrap using actual glyph measurements."""
    words = text.split()
    lines, current = [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def generate_default_og(
    site_name: str = "Kubai Kevin",
    tagline: str = "Real Systems. Real Failures. Real Fixes.",
    output_path: Path = OUTPUT_PATH,
) -> Path:
    """Render and save the branded default OG/Twitter card. Returns the path."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (OG_WIDTH, OG_HEIGHT))
    draw = ImageDraw.Draw(img)

    # Vertical gradient background
    for y in range(OG_HEIGHT):
        t = y / (OG_HEIGHT - 1)
        draw.line([(0, y), (OG_WIDTH, y)],
                  fill=lerp_colour(GRAD_START, GRAD_END, t))

    # Subtle diagonal accent band so the card doesn't look like a flat
    # placeholder even without a real photo/illustration.
    accent = Image.new("RGBA", (OG_WIDTH, OG_HEIGHT), (0, 0, 0, 0))
    accent_draw = ImageDraw.Draw(accent)
    accent_draw.polygon(
        [(OG_WIDTH * 0.62, 0), (OG_WIDTH, 0),
         (OG_WIDTH, OG_HEIGHT), (OG_WIDTH * 0.78, OG_HEIGHT)],
        fill=(255, 255, 255, 18),
    )
    img.paste(Image.alpha_composite(
        img.convert("RGBA"), accent).convert("RGB"))

    title_font = _load_font(76, bold=True)
    tagline_font = _load_font(36)

    left_margin = 80
    max_text_width = OG_WIDTH - (left_margin * 2)

    title_lines = _wrap_text(draw, site_name, title_font, max_text_width)
    tagline_lines = _wrap_text(draw, tagline, tagline_font, max_text_width)

    y = 220
    for line in title_lines:
        draw.text((left_margin, y), line, font=title_font, fill=TEXT_COLOUR)
        y += 88

    y += 20
    for line in tagline_lines:
        draw.text((left_margin, y), line,
                  font=tagline_font, fill=SUBTEXT_COLOUR)
        y += 48

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG", optimize=True)
    print(f"Generated {output_path} ({OG_WIDTH}x{OG_HEIGHT})")
    return output_path


if __name__ == "__main__":
    generate_default_og()
