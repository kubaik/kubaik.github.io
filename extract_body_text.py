"""
extract_body_text.py — diagnostic + reusable extractor.

Instead of guessing which tag wraps the article body (the bug in the
previous version — find("article")/find("main")/largest <div> all came up
empty for 270/280 posts, which is a parser miss, not 270 actually-empty
pages), this strips known chrome (nav, header, footer, script, style,
forms, and anything matching common non-content class/id patterns) from
the whole document and keeps what's left. Much less sensitive to the
exact template.

Run standalone first to sanity-check against a real page:
    python extract_body_text.py docs/solid-in-action/index.html
"""
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup

STRIP_TAGS = ["nav", "header", "footer", "script", "style", "form", "aside"]
STRIP_CLASS_ID_PATTERNS = re.compile(
    r"(nav|header|footer|sidebar|related-articles|related-posts|comments|"
    r"breadcrumb|social-share|newsletter|cookie|author-box)", re.IGNORECASE
)


def extract_body_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag_name in STRIP_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    for tag in soup.find_all(True):
        if tag.attrs is None or tag.parent is None:
            continue  # already decomposed as part of a parent removed above
        classes = " ".join(tag.get("class") or [])
        tag_id = tag.get("id") or ""
        if STRIP_CLASS_ID_PATTERNS.search(classes) or STRIP_CLASS_ID_PATTERNS.search(tag_id):
            tag.decompose()

    body = soup.find("body") or soup
    return body.get_text(" ", strip=True)


if __name__ == "__main__":
    path = Path(sys.argv[1])
    html = path.read_text(encoding="utf-8", errors="ignore")
    text = extract_body_text(html)
    print(f"Extracted {len(text.split())} words")
    print("---first 500 chars---")
    print(text[:500])
