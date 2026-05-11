#!/usr/bin/env python3
"""
generate_icons.py
-----------------
Generates all required PWA icon sizes using Pillow only.
No system libraries (cairo, libxml2, etc.) required.

Usage:
    pip install Pillow
    python generate_icons.py

Output: docs/static/icons/*.png
"""

from pathlib import Path

ICON_SIZES = [72, 96, 128, 144, 152, 192, 384, 512]
OUTPUT_DIR = Path("docs/static/icons")

# Brand colours (matching your site's indigo→violet gradient)
GRAD_START = (102, 126, 234)   # #667eea
GRAD_END = (118,  75, 162)   # #764ba2
TEXT_COLOUR = (255, 255, 255)


def lerp_colour(c1, c2, t):
    """Linear interpolate between two RGB tuples."""
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def draw_rounded_rect(draw, xy, radius, fill):
    """Draw a rounded rectangle using Pillow's built-in shapes."""
    from PIL import ImageDraw
    x0, y0, x1, y1 = xy
    r = radius
    # Four corner circles + three rectangles
    draw.ellipse([x0, y0, x0 + 2*r, y0 + 2*r], fill=fill)
    draw.ellipse([x1 - 2*r, y0, x1, y0 + 2*r], fill=fill)
    draw.ellipse([x0, y1 - 2*r, x0 + 2*r, y1], fill=fill)
    draw.ellipse([x1 - 2*r, y1 - 2*r, x1, y1], fill=fill)
    draw.rectangle([x0 + r, y0, x1 - r, y1], fill=fill)
    draw.rectangle([x0, y0 + r, x1, y1 - r], fill=fill)


def make_icon(size: int) -> "Image":
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # ── Gradient background ───────────────────────────────────────
    # Draw row-by-row vertical gradient, then apply rounded mask
    grad_layer = Image.new("RGBA", (size, size))
    grad_draw = ImageDraw.Draw(grad_layer)
    for y in range(size):
        t = y / (size - 1)
        colour = lerp_colour(GRAD_START, GRAD_END, t)
        grad_draw.line([(0, y), (size, y)], fill=colour + (255,))

    # Rounded-rectangle mask
    radius = max(4, size // 5)          # ~20% corner radius (like iOS icons)
    mask = Image.new("L", (size, size), 0)
    mask_draw = ImageDraw.Draw(mask)
    draw_rounded_rect(mask_draw, [0, 0, size, size], radius, 255)

    img.paste(grad_layer, mask=mask)
    draw = ImageDraw.Draw(img)

    # ── "TB" text ─────────────────────────────────────────────────
    label = "TB"
    font_size = max(10, int(size * 0.40))

    font = None
    # Try to load a bold system font; fall back to Pillow's default
    font_candidates = [
        # macOS
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial Bold.ttf",
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        # Windows
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    try:
        from PIL import ImageFont
        for path in font_candidates:
            if Path(path).exists():
                font = ImageFont.truetype(path, font_size)
                break
        if font is None:
            # Scale up the built-in bitmap font as best we can
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    # Centre the text
    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (size - text_w) / 2 - bbox[0]
    y = (size - text_h) / 2 - bbox[1]

    # Subtle drop shadow
    shadow_offset = max(1, size // 64)
    draw.text((x + shadow_offset, y + shadow_offset), label,
              font=font, fill=(0, 0, 0, 80))
    draw.text((x, y), label, font=font, fill=TEXT_COLOUR + (255,))

    return img


def generate_icons():
    try:
        from PIL import Image  # noqa: F401 — just checking it's available
    except ImportError:
        print("Pillow is not installed. Run:  pip install Pillow")
        raise SystemExit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for size in ICON_SIZES:
        out_path = OUTPUT_DIR / f"icon-{size}x{size}.png"
        img = make_icon(size)
        img.save(out_path, "PNG", optimize=True)
        print(f"  ✓  {out_path.name}  ({size}×{size})")

    # Also write the SVG source for designers
    svg_source = """\
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%"   stop-color="#667eea"/>
      <stop offset="100%" stop-color="#764ba2"/>
    </linearGradient>
  </defs>
  <rect width="512" height="512" rx="100" ry="100" fill="url(#g)"/>
  <text x="256" y="310" text-anchor="middle"
        font-family="Arial, Helvetica, sans-serif"
        font-weight="800" font-size="220" fill="white">TB</text>
</svg>
"""
    svg_path = OUTPUT_DIR / "icon-source.svg"
    svg_path.write_text(svg_source, encoding="utf-8")
    print(f"\n  SVG source saved → {svg_path}")
    print(f"\nDone! {len(ICON_SIZES)} icons written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    generate_icons()
