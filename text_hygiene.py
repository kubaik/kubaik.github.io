"""text_hygiene — one place that decides what "clean plain text" means.

WHY THIS EXISTS
---------------
Live pages showed titles/excerpts such as ``isn&#x27;t`` and ``aren&#x27;t``.
Root cause: ``static_site_generator._safe_excerpt()`` HTML-escaped its output,
and the result was then handed to Jinja (``autoescape=True``), to JSON
(``posts.json``) and to the RSS escaper — each of which escapes *again*.
``'`` -> ``&#x27;`` -> ``&amp;#x27;`` -> the browser prints ``&#x27;``.

THE RULE (enforced by this module + tests)
------------------------------------------
* Stored data (post.json, BlogPost fields) is PLAIN TEXT. Never HTML-escaped.
* Escaping happens exactly once, at the output boundary:
    - Jinja templates  -> autoescape does it, pass plain text.
    - f-string HTML    -> wrap with ``html.escape(..., quote=True)``.
    - JSON / JSON-LD   -> json.dumps does it, pass plain text.
    - RSS/XML          -> ``_escape_xml`` does it, pass plain text.

CLI
---
    python text_hygiene.py scan                # report dirty post.json + built HTML
    python text_hygiene.py fix                 # rewrite dirty post.json (plain text)
    python text_hygiene.py fix-html            # repair already-built docs/ in place
    python text_hygiene.py check-html          # exit 1 if built HTML is double-escaped (CI gate)

All commands accept ``--docs DIR`` (default: auto-detected, see _resolve_docs) and ``--dry-run`` for fix*.
USING IT FROM utils/ OR scripts/
--------------------------------
The module lives at the repo root next to blog_post.py. Scripts in utils/ or
scripts/ that rewrite post.json should do:

    import sys; from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from text_hygiene import clean_post_dict      # before json.dump(...)

or run the CLI from anywhere:  python path/to/text_hygiene.py scan

After ``fix`` run a normal site rebuild; ``fix-html`` is a stop-gap that repairs
the published HTML without a rebuild.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------

# A real character reference: &#27; &#x27; &amp; &quot; &nbsp; ...
ENTITY_RE = re.compile(
    r"&(?:#[0-9]{1,7}|#[xX][0-9a-fA-F]{1,6}|[A-Za-z][A-Za-z0-9]{1,31});"
)

# The same thing but escaped one level too far: &amp;#x27; &amp;amp; &amp;quot;
DOUBLE_ESCAPED_RE = re.compile(
    r"&amp;(?:#[0-9]{1,7}|#[xX][0-9a-fA-F]{1,6}|"
    r"amp|lt|gt|quot|apos|nbsp|mdash|ndash|hellip|rsquo|lsquo|rdquo|ldquo);"
)

_ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS_RE = re.compile(r"\s+")


def has_entities(text: Optional[str]) -> bool:
    """True if ``text`` contains HTML character references (it should be plain)."""
    return bool(text) and bool(ENTITY_RE.search(text))


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------


def normalize_text(text, max_passes: int = 6):
    """Return ``text`` as clean plain text.

    * Repeatedly un-escapes HTML entities until stable. Repetition matters:
      ``&amp;#x27;`` needs two passes. Bounded so pathological input can't loop.
    * Removes zero-width / control characters and collapses whitespace.

    Non-strings are returned untouched, so this is safe to call on optional
    fields. Idempotent: normalize_text(normalize_text(x)) == normalize_text(x).
    """
    if not isinstance(text, str) or not text:
        return text
    out = text
    for _ in range(max_passes):
        nxt = html.unescape(out)
        if nxt == out:
            break
        out = nxt
    out = _ZERO_WIDTH_RE.sub("", out)
    out = _CONTROL_RE.sub("", out)
    return _WS_RE.sub(" ", out).strip()


def normalize_list(items):
    """normalize_text over a list of strings (tags, keywords); drops empties."""
    if not isinstance(items, (list, tuple)):
        return items
    cleaned = [normalize_text(i) if isinstance(i, str) else i for i in items]
    return [i for i in cleaned if i not in ("", None)]


# Fields of a post.json that are plain text. `content` is deliberately NOT here:
# markdown bodies may legitimately contain `&lt;` inside code samples.
PLAIN_TEXT_FIELDS = ("title", "full_title", "meta_description")
PLAIN_LIST_FIELDS = ("tags", "seo_keywords")


def clean_post_dict(data: Dict) -> Tuple[Dict, List[str]]:
    """Return (cleaned_copy, changed_field_names) for a post.json dict."""
    cleaned = dict(data)
    changed: List[str] = []
    for k in PLAIN_TEXT_FIELDS:
        v = cleaned.get(k)
        if isinstance(v, str):
            nv = normalize_text(v)
            if nv != v:
                cleaned[k] = nv
                changed.append(k)
    for k in PLAIN_LIST_FIELDS:
        v = cleaned.get(k)
        if isinstance(v, list):
            nv = normalize_list(v)
            if nv != v:
                cleaned[k] = nv
                changed.append(k)
    return cleaned, changed


# --------------------------------------------------------------------------
# Built-HTML inspection / repair
# --------------------------------------------------------------------------

# Regions where a single escape layer is correct and a double one is a bug.
# We never touch <pre>/<code>/<script> bodies: a post that *documents* HTML
# entities legitimately contains "&amp;lt;" there.
_PROTECTED_BLOCK_RE = re.compile(
    r"(<pre\b.*?</pre>|<code\b.*?</code>|<script\b(?![^>]*application/ld\+json).*?</script>)",
    re.IGNORECASE | re.DOTALL,
)


def find_double_escaped(html_text: str) -> List[str]:
    """Return up to 20 snippets of double-escaped text outside code blocks."""
    stripped = _PROTECTED_BLOCK_RE.sub(" ", html_text)
    hits = []
    for m in DOUBLE_ESCAPED_RE.finditer(stripped):
        s = max(0, m.start() - 40)
        hits.append(stripped[s : m.end() + 20].replace("\n", " "))
        if len(hits) >= 20:
            break
    return hits


def repair_html_text(html_text: str) -> Tuple[str, int]:
    """Undo ONE level of over-escaping everywhere except protected code blocks.

    ``&amp;#x27;`` -> ``&#x27;`` (browser then renders an apostrophe).
    Returns (new_text, replacements).
    """
    parts = _PROTECTED_BLOCK_RE.split(html_text)
    total = 0
    for i, part in enumerate(parts):
        # re.split with one capture group: odd indexes are the protected blocks.
        if i % 2 == 1:
            continue
        fixed, n = _fix_segment(part)
        parts[i] = fixed
        total += n
    return "".join(parts), total


def _fix_segment(segment: str) -> Tuple[str, int]:
    # Only rewrite the exact over-escaped sequences the buggy pipeline produced
    # (DOUBLE_ESCAPED_RE). Anything else — "AT&amp;T", "?a=1&amp;b=2" — is a
    # correct single escape and is left alone.
    return DOUBLE_ESCAPED_RE.subn(lambda m: "&" + m.group(0)[5:], segment)


def repair_posts_json_text(raw: str) -> Tuple[str, int]:
    """posts.json is plain text consumed via textContent: fully unescape fields."""
    data = json.loads(raw)
    changes = 0
    for p in data if isinstance(data, list) else []:
        for k in ("title", "meta_description"):
            v = p.get(k)
            if isinstance(v, str):
                nv = normalize_text(v)
                if nv != v:
                    p[k] = nv
                    changes += 1
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False), changes


# --------------------------------------------------------------------------
# Site-level operations
# --------------------------------------------------------------------------

_SKIP_DIRS = {"static", ".git", "node_modules"}


def _iter_post_jsons(docs: Path) -> Iterable[Path]:
    for d in sorted(docs.iterdir()):
        if d.is_dir() and d.name not in _SKIP_DIRS and not d.name.startswith("."):
            p = d / "post.json"
            if p.exists():
                yield p


def _iter_html(docs: Path) -> Iterable[Path]:
    for p in docs.rglob("*.html"):
        if not any(
            part in _SKIP_DIRS or part.startswith(".")
            for part in p.relative_to(docs).parts
        ):
            yield p


def scan_sources(docs: Path) -> List[Tuple[Path, List[str]]]:
    dirty = []
    for pj in _iter_post_jsons(docs):
        try:
            data = json.loads(pj.read_text(encoding="utf-8"))
        except Exception:
            continue
        _, changed = clean_post_dict(data)
        if changed:
            dirty.append((pj, changed))
    return dirty


def fix_sources(docs: Path, dry_run: bool = False) -> int:
    n = 0
    for pj, changed in scan_sources(docs):
        data = json.loads(pj.read_text(encoding="utf-8"))
        cleaned, _ = clean_post_dict(data)
        print(
            f"  {'would fix' if dry_run else 'fixed'} {pj.parent.name}: {', '.join(changed)}"
        )
        if not dry_run:
            pj.write_text(
                json.dumps(cleaned, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        n += 1
    return n


def scan_built_html(docs: Path) -> List[Tuple[Path, List[str]]]:
    bad = []
    for p in _iter_html(docs):
        try:
            hits = find_double_escaped(p.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        if hits:
            bad.append((p, hits))
    return bad


def fix_built_html(docs: Path, dry_run: bool = False) -> Tuple[int, int]:
    files = reps = 0
    for p in _iter_html(docs):
        text = p.read_text(encoding="utf-8", errors="ignore")
        if not DOUBLE_ESCAPED_RE.search(text):
            continue
        fixed, n = repair_html_text(text)
        if n:
            files += 1
            reps += n
            if not dry_run:
                p.write_text(fixed, encoding="utf-8")
    pj = docs / "posts.json"
    if pj.exists():
        fixed, n = repair_posts_json_text(pj.read_text(encoding="utf-8"))
        if n:
            reps += n
            files += 1
            if not dry_run:
                pj.write_text(fixed, encoding="utf-8")
    return files, reps


def assert_no_double_escaping(docs: Path, max_report: int = 10) -> None:
    """Build-time gate. Raises RuntimeError if generated HTML is double-escaped."""
    bad = scan_built_html(docs)
    if not bad:
        return
    lines = [f"{len(bad)} generated page(s) contain double-escaped HTML entities:"]
    for p, hits in bad[:max_report]:
        lines.append(f"  {p.relative_to(docs)}: …{hits[0]}…")
    raise RuntimeError("\n".join(lines))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _resolve_docs(explicit: Optional[str]) -> Path:
    """Find docs/ regardless of the current working directory."""
    if explicit:
        return Path(explicit)
    here = Path(__file__).resolve().parent
    for cand in (Path.cwd() / "docs", here / "docs", here.parent / "docs"):
        if cand.is_dir():
            return cand
    return Path("docs")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("command", choices=["scan", "fix", "fix-html", "check-html"])
    ap.add_argument(
        "--docs",
        default=None,
        help="site output dir (default: auto-detect ./docs, then <script dir>/docs, "
        "then <script dir>/../docs — so it works from the repo root, from utils/ "
        "or scripts/, or when this file is copied into utils/)",
    )
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    docs = _resolve_docs(a.docs)
    if not docs.is_dir():
        print(f"docs dir not found: {docs}")
        return 2

    if a.command == "scan":
        src = scan_sources(docs)
        built = scan_built_html(docs)
        print(f"post.json with entities in title/description/tags: {len(src)}")
        for pj, ch in src[:20]:
            print(f"  {pj.parent.name}: {', '.join(ch)}")
        print(f"built HTML pages with double-escaped entities: {len(built)}")
        for p, hits in built[:20]:
            print(f"  {p.relative_to(docs)}: …{hits[0]}…")
        return 1 if (src or built) else 0
    if a.command == "fix":
        n = fix_sources(docs, a.dry_run)
        print(
            f"{n} post.json file(s) {'would be ' if a.dry_run else ''}cleaned. Rebuild the site next."
        )
        return 0
    if a.command == "fix-html":
        files, reps = fix_built_html(docs, a.dry_run)
        print(
            f"{files} file(s), {reps} replacement(s) {'would be ' if a.dry_run else ''}applied."
        )
        return 0
    if a.command == "check-html":
        try:
            assert_no_double_escaping(docs)
        except RuntimeError as e:
            print(e)
            return 1
        print("OK: no double-escaped entities in built HTML.")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
