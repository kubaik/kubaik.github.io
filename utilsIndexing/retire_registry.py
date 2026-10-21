"""retire_registry.py — stops "publish, later delete -> 404 in GSC" from recurring.

Works on shallow CI checkouts (no git history needed):
  * published_slugs.json  — ledger of every slug that has ever been live (appended each build)
  * retired_slugs.json    — slugs in the ledger that are no longer live, with retired_at
Any deletion — by script, by hand, or by `git rm` — is detected on the next build because
the slug is in the ledger but has no docs/<slug>/post.json.

If config `tombstone_retired: true` (default), recently retired slugs get a 200 +
`noindex, follow` page (never listed in sitemap) so Google moves them to
"Excluded by noindex" instead of repeating 404s. Older ones are left as plain 404s,
which is the correct permanent state and carries no quality penalty.

CLI:
  python utilsIndexing/retire_registry.py import-gsc "Not found (404).csv"   # add URLs exported from GSC
"""

import csv, json, sys
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path
from urllib.parse import urlsplit

LEDGER, RETIRED = Path("published_slugs.json"), Path("retired_slugs.json")
RESERVED = {
    "tag",
    "static",
    "about",
    "contact",
    "privacy-policy",
    "terms-of-service",
    "dmca",
    "ai-content-policy",
    "author",
    "page",
    "cdn-cgi",
    "admin",
    "private",
}
MARK = "This post has been removed"
_NOW = lambda: datetime.now(timezone.utc).isoformat()


def _load(p: Path) -> dict:
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _save(p: Path, d: dict):
    p.write_text(json.dumps(d, indent=2, sort_keys=True), encoding="utf-8")


def _tombstone_html(site_name: str, base_path: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Post removed — {escape(site_name)}</title>
<meta name="robots" content="noindex, follow">
<style>html,body{{margin:0;min-height:100%;display:flex;align-items:center;justify-content:center;font-family:system-ui,sans-serif;color:#333;background:#fff}}
.t{{text-align:center;padding:2rem;max-width:480px}}a{{display:inline-block;margin-top:1.25rem;color:#fff;background:#4f46e5;padding:.6rem 1.25rem;border-radius:6px;text-decoration:none;font-weight:600}}</style>
</head><body><div class="t"><h1>{MARK}</h1>
<p>This article was retired because it did not meet our content quality standard.</p>
<a href="{base_path}/">Browse the blog</a></div></body></html>"""


def sync_ledger_and_tombstones(live_slugs, docs: Path, config: dict) -> dict:
    live = set(live_slugs)
    docs = Path(docs)
    ledger, retired = _load(LEDGER), _load(RETIRED)
    now = _NOW()

    for s in live:
        ledger.setdefault(s, {"first_seen": now})
        retired.pop(s, None)  # resurrected -> no longer retired

    for s in ledger:
        if s in live or s in RESERVED or s in retired:
            continue
        if (
            docs / s / "post.json"
        ).exists():  # on disk but filtered this build: not a deletion
            continue
        retired[s] = {"retired_at": now, "source": "ledger_diff"}

    _save(LEDGER, ledger)
    _save(RETIRED, retired)

    written = 0
    if config.get("tombstone_retired", True):
        cutoff = datetime.now(timezone.utc) - timedelta(
            days=int(config.get("tombstone_days", 180))
        )
        html = _tombstone_html(
            config.get("site_name", "Blog"), config.get("base_path", "")
        )
        for s, meta in retired.items():
            if s in live or s in RESERVED or not s or "/" in s:
                continue
            try:
                when = datetime.fromisoformat(meta["retired_at"])
                if when.tzinfo is None:
                    when = when.replace(tzinfo=timezone.utc)
            except Exception:
                when = datetime.now(timezone.utc)
            if when < cutoff:
                continue
            d = docs / s
            if (d / "post.json").exists():
                continue
            d.mkdir(parents=True, exist_ok=True)
            (d / "index.html").write_text(html, encoding="utf-8")
            written += 1
    print(f"  🪦 retire_registry: {len(retired)} retired, {written} tombstones written")
    return retired


def import_gsc_csv(csv_path: str):
    """Add every slug from a GSC 'Not found (404)' / 'Page with redirect' export."""
    retired, ledger = _load(RETIRED), _load(LEDGER)
    added = 0
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            url = row.get("URL") or row.get("url") or next(iter(row.values()), "")
            slug = urlsplit(url).path.strip("/")
            if not slug or "/" in slug or slug in RESERVED:
                continue
            if slug not in retired and not Path("docs", slug, "post.json").exists():
                retired[slug] = {"retired_at": _NOW(), "source": "gsc_import"}
                ledger.setdefault(slug, {"first_seen": "unknown"})
                added += 1
    _save(RETIRED, retired)
    _save(LEDGER, ledger)
    print(f"imported {added} slug(s)")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "import-gsc":
        import_gsc_csv(sys.argv[2])
    else:
        print(__doc__)
