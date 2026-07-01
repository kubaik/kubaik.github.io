#!/usr/bin/env python3
"""
scripts/generate_og_images.py

Generates per-article Open Graph images (1200×630 PNG) at build time.
Eliminates the problem of every article sharing the same generic app icon.

Requirements:
    pip install Pillow

Usage:
    python scripts/generate_og_images.py \
        --posts-dir ./_site \
        --output-dir ./_site/static/og \
        --font-dir ./static/fonts

The script:
  1. Reads each post's HTML to extract title and primary tag
  2. Renders a 1200×630 PNG with:
     - Site brand background (indigo gradient matching theme-color #6366f1)
     - Article title (wrapped, white text)
     - Site name and author name at bottom
     - Primary tag label
  3. Saves to /static/og/{slug}.png
  4. Updates the HTML file's og:image meta tag to point to the generated image

After running this script, rebuild the site or patch the HTML directly.
The generated images are committed to the repository and served as static files.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("ERROR: Pillow not installed. Run: pip install Pillow", file=sys.stderr)
    sys.exit(1)

# Canvas dimensions
WIDTH = 1200
HEIGHT = 630

# Brand colours (matching #6366f1 indigo theme)
BG_TOP = (60, 50, 150)       # dark indigo
BG_BOTTOM = (99, 102, 241)   # #6366f1
ACCENT = (165, 180, 252)     # indigo-300
WHITE = (255, 255, 255)
DARK = (30, 27, 75)          # indigo-950

# Font paths — fall back to default if custom fonts not available
DEFAULT_FONT_SIZE_TITLE = 52
DEFAULT_FONT_SIZE_SUB = 28
DEFAULT_FONT_SIZE_BRAND = 22

SITE_NAME = "Kubai Kevin"
SITE_URL = "kubaik.github.io"


def load_font(font_dir: Path, filename: str, size: int) -> ImageFont.FreeTypeFont:
    """Load a TTF font, falling back to PIL default."""
    if font_dir:
        candidates = [
            font_dir / filename,
            font_dir / "Inter-Bold.ttf",
            font_dir / "Inter-Regular.ttf",
        ]
        for path in candidates:
            if path.exists():
                try:
                    return ImageFont.truetype(str(path), size)
                except Exception:
                    pass
    # PIL built-in fallback (low quality but always available)
    return ImageFont.load_default()


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    """Word-wrap text to fit within max_width pixels."""
    words = text.split()
    lines = []
    current = ""
    for word in words:
        test = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_gradient_background(img: Image.Image) -> None:
    """Draw a vertical gradient from BG_TOP to BG_BOTTOM."""
    draw = ImageDraw.Draw(img)
    for y in range(HEIGHT):
        ratio = y / HEIGHT
        r = int(BG_TOP[0] + (BG_BOTTOM[0] - BG_TOP[0]) * ratio)
        g = int(BG_TOP[1] + (BG_BOTTOM[1] - BG_TOP[1]) * ratio)
        b = int(BG_TOP[2] + (BG_BOTTOM[2] - BG_TOP[2]) * ratio)
        draw.line([(0, y), (WIDTH, y)], fill=(r, g, b))


def draw_decorative_elements(draw: ImageDraw.ImageDraw) -> None:
    """Add subtle geometric decoration (two semi-transparent circles)."""
    # Large circle, top-right
    draw.ellipse(
        [(WIDTH - 200, -150), (WIDTH + 150, 200)],
        fill=(255, 255, 255, 15),
    )
    # Smaller circle, bottom-left
    draw.ellipse(
        [(-80, HEIGHT - 180), (220, HEIGHT + 80)],
        fill=(255, 255, 255, 10),
    )


def generate_og_image(
    title: str,
    tag: str,
    output_path: Path,
    font_dir: Path | None = None,
) -> None:
    """Generate a single OG image and save to output_path."""
    # Use RGBA for transparent circles, then convert to RGB for PNG
    img = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 255))
    draw_gradient_background(img)

    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    draw_decorative_elements(draw_overlay)
    img = Image.alpha_composite(img, overlay).convert("RGB")

    draw = ImageDraw.Draw(img)

    font_title = load_font(font_dir, "Inter-Bold.ttf", DEFAULT_FONT_SIZE_TITLE)
    font_sub = load_font(font_dir, "Inter-Regular.ttf", DEFAULT_FONT_SIZE_SUB)
    font_brand = load_font(font_dir, "Inter-Bold.ttf", DEFAULT_FONT_SIZE_BRAND)

    # ── Tag label (top-left) ──
    if tag:
        tag_text = f"#{tag}"
        tag_x, tag_y = 64, 52
        tag_bbox = draw.textbbox((tag_x, tag_y), tag_text, font=font_sub)
        draw.rounded_rectangle(
            [tag_bbox[0] - 12, tag_bbox[1] - 8, tag_bbox[2] + 12, tag_bbox[3] + 8],
            radius=6,
            fill=(255, 255, 255, 40),
        )
        draw.text((tag_x, tag_y), tag_text, font=font_sub, fill=ACCENT)

    # ── Title (centred vertically) ──
    padding = 64
    max_title_width = WIDTH - padding * 2
    lines = wrap_text(draw, title, font_title, max_title_width)
    # Limit to 3 lines, truncate with ellipsis
    if len(lines) > 3:
        lines = lines[:3]
        lines[2] = lines[2][:-3].rstrip() + "…"

    line_height = DEFAULT_FONT_SIZE_TITLE + 16
    total_text_height = len(lines) * line_height
    y_start = (HEIGHT - total_text_height) // 2 - 20  # slightly above centre

    for i, line in enumerate(lines):
        y = y_start + i * line_height
        # Subtle shadow
        draw.text((padding + 2, y + 2), line,
                  font=font_title, fill=(0, 0, 0, 80))
        draw.text((padding, y), line, font=font_title, fill=WHITE)

    # ── Bottom bar: site name + URL ──
    bottom_y = HEIGHT - 80
    draw.line([(padding, bottom_y - 16), (WIDTH - padding,
              bottom_y - 16)], fill=(255, 255, 255, 60), width=1)
    draw.text((padding, bottom_y), SITE_NAME, font=font_brand, fill=WHITE)
    url_bbox = draw.textbbox((0, 0), SITE_URL, font=font_sub)
    url_x = WIDTH - padding - (url_bbox[2] - url_bbox[0])
    draw.text((url_x, bottom_y + 2), SITE_URL, font=font_sub, fill=ACCENT)

    # Save as RGB PNG (no alpha needed for og:image)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(output_path), "PNG", optimize=True)


def extract_slug_from_path(html_path: Path, site_dir: Path) -> str:
    rel = html_path.parent.relative_to(site_dir)
    return str(rel).replace("/", "-").replace("\\", "-").strip("-") or "index"


def extract_post_data(html: str) -> dict:
    """Extract title and primary tag from HTML."""
    title = ""
    tag = ""

    # og:title is more reliable than <title> (no site suffix)
    m = re.search(
        r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)["\']', html, re.IGNORECASE)
    if m:
        title = m.group(1).strip()

    if not title:
        m = re.search(r"<title>([^<]+)</title>", html, re.IGNORECASE)
        if m:
            title = re.sub(r"\s*[-|]\s*Kubai Kevin\s*$",
                           "", m.group(1)).strip()

    # First keyword as primary tag
    m = re.search(
        r'<meta[^>]+name=["\']keywords["\'][^>]+content=["\']([^"\']+)["\']', html, re.IGNORECASE)
    if m:
        tags = [t.strip() for t in m.group(1).split(",")]
        tag = tags[0] if tags else ""

    return {"title": title, "tag": tag}


def update_og_image_in_html(html: str, new_url: str) -> str:
    """Replace the og:image content attribute with the new generated image URL."""
    # og:image
    html = re.sub(
        r'(<meta[^>]+property=["\']og:image["\'][^>]+content=["\'])([^"\']+)(["\'])',
        lambda m: m.group(1) + new_url + m.group(3),
        html,
        flags=re.IGNORECASE,
    )
    # twitter:image
    html = re.sub(
        r'(<meta[^>]+name=["\']twitter:image["\'][^>]+content=["\'])([^"\']+)(["\'])',
        lambda m: m.group(1) + new_url + m.group(3),
        html,
        flags=re.IGNORECASE,
    )
    # Fix the dimensions meta (these are now correct at 1200×630)
    html = re.sub(
        r'(<meta[^>]+property=["\']og:image:width["\'][^>]+content=["\'])\d+(["\'])',
        lambda m: m.group(1) + "1200" + m.group(2),
        html,
        flags=re.IGNORECASE,
    )
    html = re.sub(
        r'(<meta[^>]+property=["\']og:image:height["\'][^>]+content=["\'])\d+(["\'])',
        lambda m: m.group(1) + "630" + m.group(2),
        html,
        flags=re.IGNORECASE,
    )
    return html


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate per-article OG images")
    parser.add_argument("--posts-dir", required=True,
                        help="Site output directory containing HTML files")
    parser.add_argument("--output-dir", required=True,
                        help="Where to write generated PNG files")
    parser.add_argument("--font-dir", default=None,
                        help="Optional directory containing TTF fonts (Inter recommended)")
    parser.add_argument(
        "--base-url", default="https://kubaik.github.io", help="Site base URL")
    parser.add_argument("--patch-html", action="store_true",
                        help="Patch og:image in HTML files after generating images")
    parser.add_argument("--limit", type=int, default=0,
                        help="Only process N files (0 = all, for testing)")
    args = parser.parse_args()

    site_dir = Path(args.posts_dir)
    output_dir = Path(args.output_dir)
    font_dir = Path(args.font_dir) if args.font_dir else None
    base_url = args.base_url.rstrip("/")

    html_files = list(site_dir.rglob("index.html"))
    if args.limit:
        html_files = html_files[: args.limit]

    print(f"Processing {len(html_files)} HTML files...")
    generated = 0
    skipped = 0
    errors = 0

    for html_path in html_files:
        try:
            slug = extract_slug_from_path(html_path, site_dir)
            if not slug or slug == "index":
                # Homepage — skip, use default OG image
                skipped += 1
                continue

            output_path = output_dir / f"{slug}.png"

            # Skip if already generated (incremental builds)
            if output_path.exists():
                skipped += 1
                continue

            html = html_path.read_text(encoding="utf-8", errors="replace")
            data = extract_post_data(html)

            if not data["title"]:
                print(f"  SKIP (no title): {html_path}")
                skipped += 1
                continue

            generate_og_image(
                title=data["title"],
                tag=data["tag"],
                output_path=output_path,
                font_dir=font_dir,
            )

            if args.patch_html:
                new_image_url = f"{base_url}/static/og/{slug}.png"
                patched_html = update_og_image_in_html(html, new_image_url)
                html_path.write_text(patched_html, encoding="utf-8")

            generated += 1
            if generated % 50 == 0:
                print(f"  Generated {generated} images...")

        except Exception as e:
            print(f"  ERROR processing {html_path}: {e}", file=sys.stderr)
            errors += 1

    print(
        f"\nDone. Generated: {generated}, Skipped: {skipped}, Errors: {errors}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
