#!/usr/bin/env python3
"""check_indexing_hygiene.py — fail the deploy if the build would create GSC errors.

Checks (all code-driven, no manual step):
  1. Every <loc> in sitemap.xml ends with '/', maps to a real docs/<slug>/index.html,
     is not noindex, not a meta-refresh stub, not a tombstone.     -> 404 / redirect / noindex
  2. Every internal link in built HTML resolves to a real page.     -> 404
  3. No internal link or canonical omits the trailing slash.        -> "Page with redirect"
  4. Sitemap never lists a slug present in retired_slugs.json.      -> publish-then-delete loop
Usage: python utilsIndexing/check_indexing_hygiene.py [--docs docs] [--base https://kubaik.github.io] [--strict]
Exit 1 on any ERROR (or WARN with --strict).
"""

import argparse, json, re, sys
from pathlib import Path
from urllib.parse import urlsplit

ROBOTS = re.compile(r'<meta[^>]+name=["\']robots["\'][^>]+content=["\']([^"\']*)', re.I)
REFRESH = re.compile(r'<meta[^>]+http-equiv=["\']refresh["\']', re.I)
HREF = re.compile(r'(?:href|src)=["\']([^"\'#?]+)', re.I)
CANON = re.compile(r'<link[^>]+rel=["\']canonical["\'][^>]+href=["\']([^"\']+)', re.I)
TOMB = "This post has been removed"
SKIP_PREFIX = ("/static/", "/tag/page", "/cdn-cgi/")


def local_path(docs: Path, base: str, url: str):
    p = urlsplit(url)
    if p.netloc and not base.endswith(p.netloc):
        return None
    path = p.path or "/"
    if path.endswith("/"):
        return docs / path.lstrip("/") / "index.html"
    return docs / path.lstrip("/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="docs")
    ap.add_argument("--base", default="https://kubaik.github.io")
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()
    docs, base = Path(a.docs), a.base.rstrip("/")
    errors, warns = [], []

    retired = set()
    rp = Path("retired_slugs.json")
    if rp.exists():
        retired = set(json.loads(rp.read_text()).keys())

    sm = docs / "sitemap.xml"
    if not sm.exists():
        print("ERROR: sitemap.xml missing")
        sys.exit(1)
    locs = re.findall(r"<loc>([^<]+)</loc>", sm.read_text(encoding="utf-8"))
    for loc in locs:
        if not loc.endswith("/") and "." not in loc.rsplit("/", 1)[-1]:
            errors.append(f"sitemap: slash-less URL (redirects): {loc}")
        f = local_path(
            docs,
            base,
            loc if loc.endswith("/") or "." in loc.rsplit("/", 1)[-1] else loc + "/",
        )
        if f is None:
            errors.append(f"sitemap: foreign host: {loc}")
            continue
        if not f.exists():
            errors.append(f"sitemap: 404 target: {loc}")
            continue
        html = f.read_text(encoding="utf-8", errors="ignore")
        m = ROBOTS.search(html)
        if m and "noindex" in m.group(1).lower():
            errors.append(f"sitemap: noindex page listed: {loc}")
        if REFRESH.search(html):
            errors.append(f"sitemap: meta-refresh stub listed: {loc}")
        if TOMB in html:
            errors.append(f"sitemap: tombstone listed: {loc}")
        slug = urlsplit(loc).path.strip("/")
        if slug in retired:
            errors.append(f"sitemap: retired slug listed: {slug}")

    missing = {}
    for f in docs.rglob("index.html"):
        html = f.read_text(encoding="utf-8", errors="ignore")
        rel = f.relative_to(docs).as_posix()
        for href in set(HREF.findall(html)):
            if href.startswith(("mailto:", "tel:", "javascript:", "data:", "//")):
                continue
            if href.startswith(("http://", "https://")) and not href.startswith(base):
                continue
            path = urlsplit(href).path if href.startswith("http") else href
            if not path.startswith("/") or path.startswith(SKIP_PREFIX):
                continue
            last = path.rsplit("/", 1)[-1]
            if not path.endswith("/") and "." not in last and path != "":
                warns.append(f"{rel}: slash-less internal link {path}")
                path += "/"
            target = docs / path.lstrip("/")
            target = target / "index.html" if path.endswith("/") else target
            if not target.exists():
                missing.setdefault(path, []).append(rel)
        for c in CANON.findall(html):
            cp = urlsplit(c).path
            if cp and not cp.endswith("/") and "." not in cp.rsplit("/", 1)[-1]:
                errors.append(f"{rel}: canonical without trailing slash: {c}")
    for path, srcs in sorted(missing.items()):
        errors.append(
            f"dead internal link {path} (from {len(srcs)} page(s), e.g. {srcs[0]})"
        )

    for w in warns[:50]:
        print("WARN ", w)
    for e in errors[:200]:
        print("ERROR", e)
    print(
        f"\nsitemap URLs: {len(locs)} | errors: {len(errors)} | warnings: {len(warns)}"
    )
    sys.exit(1 if errors or (a.strict and warns) else 0)


if __name__ == "__main__":
    main()
