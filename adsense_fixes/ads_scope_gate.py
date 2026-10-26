#!/usr/bin/env python3
"""
ads_scope_gate.py - post-build pass over ./docs enforcing "ads only on real publisher content".

AdSense policy: no Google ads/loader on pages without substantial publisher content,
and nothing on pages that are noindexed, redirects, errors, or navigation-only.

For every docs/**/index.html (and 404.html) it removes the adsbygoogle loader, the
google-adsense-account meta and any <ins class="adsbygoogle"> unit when the page is:
  * noindex / meta-refresh stub
  * a utility or listing page (tag/, page/, author/, about, contact, legal, 404, offline)
  * a post whose visible text is below --min-words
Idempotent; run AFTER `blog_system.py build` and BEFORE the Pages upload.

  python adsense_fixes/ads_scope_gate.py --docs ./docs --min-words 1200 [--check]
--check exits 1 if any page would change (CI guard) and writes nothing.
"""

import argparse, re, sys
from pathlib import Path

UTILITY_TOP = {
    "tag",
    "page",
    "author",
    "about",
    "contact",
    "privacy-policy",
    "terms-of-service",
    "dmca",
    "ai-content-policy",
    "static",
}
UTILITY_FILES = {"404.html", "offline.html"}

LOADER = re.compile(r"\s*<script\b[^>]*adsbygoogle\.js[^>]*>\s*</script>", re.I)
PRECONNECT = re.compile(r"\s*<link\b[^>]*pagead2\.googlesyndication\.com[^>]*>", re.I)
ACCT_META = re.compile(
    r'\s*<meta\b[^>]*name=["\']google-adsense-account["\'][^>]*>', re.I
)
INS = re.compile(
    r'\s*<ins\b[^>]*class=["\'][^"\']*adsbygoogle[^"\']*["\'][\s\S]*?</ins>'
    r"(?:\s*<script>\s*\(adsbygoogle\s*=\s*window\.adsbygoogle\s*\|\|\s*\[\]\)\.push\(\{\}\);?\s*</script>)?",
    re.I,
)
NOINDEX = re.compile(r'<meta\b[^>]*name=["\']robots["\'][^>]*noindex', re.I)
REFRESH = re.compile(r'http-equiv=["\']refresh["\']', re.I)


def visible_words(html: str) -> int:
    h = re.sub(
        r"<(script|style|nav|header|footer|aside|form)\b[\s\S]*?</\1>",
        " ",
        html,
        flags=re.I,
    )
    return len(re.sub(r"<[^>]+>", " ", h).split())


def should_strip(path: Path, docs: Path, html: str, min_words: int):
    rel = path.relative_to(docs)
    if rel.name in UTILITY_FILES or (
        len(rel.parts) > 1 and rel.parts[0] in UTILITY_TOP
    ):
        return "utility_page"
    if rel.parts == ("index.html",):
        return None  # homepage keeps the loader (site verification); no units are rendered there
    if NOINDEX.search(html) or REFRESH.search(html):
        return "noindex_or_redirect"
    if (path.parent / "post.json").exists() and visible_words(html) < min_words:
        return "thin_post"
    return None


def strip(html: str) -> str:
    for rx in (INS, LOADER, PRECONNECT, ACCT_META):
        html = rx.sub("", html)
    return html


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="./docs")
    ap.add_argument("--min-words", type=int, default=1200)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    docs = Path(a.docs)
    changed, reasons = [], {}
    pages = list(docs.glob("**/index.html")) + [
        docs / n for n in UTILITY_FILES if (docs / n).exists()
    ]
    for p in pages:
        if "static" in p.relative_to(docs).parts[:1]:
            continue
        html = p.read_text(encoding="utf-8", errors="ignore")
        if not re.search(r"adsbygoogle|google-adsense-account", html, re.I):
            continue
        why = should_strip(p, docs, html, a.min_words)
        if not why:
            continue
        new = strip(html)
        if new != html:
            changed.append(p)
            reasons[why] = reasons.get(why, 0) + 1
            if not a.check:
                p.write_text(new, encoding="utf-8")
    print(
        f"ads_scope_gate: {'would change' if a.check else 'changed'} {len(changed)} pages {reasons}"
    )
    if a.check and changed:
        sys.exit(1)


if __name__ == "__main__":
    main()
