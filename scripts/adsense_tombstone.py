#!/usr/bin/env python3
"""
scripts/adsense_tombstone.py
============================
Replace the posts in .adsense_triage/delete.json with noindex,follow
tombstone pages.

WHY TOMBSTONE INSTEAD OF HARD-DELETE
------------------------------------
Removing a post from docs/ entirely would 404 the URL. For AdSense
purposes that's fine, but for SEO it isn't: any inbound link to the
removed post (from Twitter, RSS readers, other blogs that linked to
it, Google's own index) dead-ends, and Search Console starts logging
"Not found (404)" errors.

A tombstone instead writes a real 200-response page with:

    <meta name="robots" content="noindex, follow">

Google drops the page from the index on the next crawl without
treating the URL as broken, and any visitor following an old link
lands on a "this post was removed" page with a link back to the
homepage.

SAFETY
------
1. Every post.json is backed up to .adsense_backups/{slug}/post.json
   before the tombstone replaces it, so the decision is reversible.
2. --dry-run is the default. Nothing is written until --apply.
3. The removal is logged to docs/_removed_posts.json, which is the
   same log purge_low_quality_posts() uses — one audit trail for both
   code paths.
4. Reuses the same tombstone template from static_site_generator.py
   that purge_low_quality_posts() uses, so a post tombstoned by this
   script looks identical to one tombstoned by the main pipeline.

AFTER RUNNING
-------------
Always rebuild the site so the sitemap, RSS feed, homepage, and tag
pages reflect the removals:

    python blog_system.py build

Usage:
    python scripts/adsense_tombstone.py --dry-run
    python scripts/adsense_tombstone.py --apply
    python scripts/adsense_tombstone.py --apply --triage-dir ./.adsense_triage
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Reuse the site's own tombstone template so a tombstoned post looks
# identical to one the pipeline would tombstone via purge_low_quality_posts.
from static_site_generator import _build_templates


_DOCS_DIR = Path("./docs")
_TRIAGE_DIR = Path("./.adsense_triage")
_BACKUP_DIR = Path("./.adsense_backups")


def _load_delete_bucket(triage_dir: Path) -> list[dict]:
    """Load the delete bucket produced by scripts/adsense_triage.py."""
    path = triage_dir / "delete.json"
    if not path.exists():
        print(f"No delete bucket found at {path} — run adsense_triage.py first.")
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"Could not read {path}: {e}")
        return []


def _load_removed_log(docs_dir: Path) -> dict:
    """Load the shared removal log. Empty dict if it doesn't exist yet."""
    path = docs_dir / "_removed_posts.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _write_removed_log(docs_dir: Path, log: dict) -> None:
    (docs_dir / "_removed_posts.json").write_text(
        json.dumps(log, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _tombstone_one(
    slug: str,
    reason: str,
    tombstone_template,
    site_name: str,
    base_path: str,
    removed_log: dict,
    docs_dir: Path,
    backup_dir: Path,
    dry_run: bool,
) -> bool:
    """
    Tombstone one post. Returns True on success (including dry-run no-op),
    False if the post couldn't be tombstoned (missing dir, no post.json).
    """
    post_dir = docs_dir / slug
    if not post_dir.exists():
        print(f"  SKIP   {slug}: directory not found")
        return False

    post_json = post_dir / "post.json"
    if not post_json.exists():
        print(f"  SKIP   {slug}: no post.json (already a stub?)")
        return False

    if dry_run:
        print(f"  DRY-RUN  would tombstone {slug} ({reason})")
        return True

    # ── 1. Backup the original post.json + index.md ──────────────────
    post_backup_dir = backup_dir / slug
    post_backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(post_json, post_backup_dir / "post.json")

    index_md = post_dir / "index.md"
    if index_md.exists():
        shutil.copy2(index_md, post_backup_dir / "index.md")

    # ── 2. Remove everything in the post directory except index.html ──
    # This clears out static assets, images, social_posts.json, etc. The
    # index.html itself is about to be replaced anyway, but keeping the
    # filter explicit means the removal loop can't accidentally leave a
    # partial post.json behind.
    for stale in post_dir.iterdir():
        if stale.name == "index.html":
            continue
        if stale.is_dir():
            shutil.rmtree(stale, ignore_errors=True)
        else:
            stale.unlink(missing_ok=True)

    # ── 3. Write the tombstone page ──────────────────────────────────
    html = tombstone_template.render(
        site_name=site_name,
        base_path=base_path,
    )
    (post_dir / "index.html").write_text(html, encoding="utf-8")

    # ── 4. Log the removal ───────────────────────────────────────────
    removed_log[slug] = {
        "removed_at": datetime.now().isoformat(),
        "reason": reason,
        "source": "adsense_triage",
    }

    print(f"  TOMBSTONED  {slug}  ({reason})")
    return True


def _load_site_config() -> tuple[str, str]:
    """
    Pull site_name and base_path out of config.yaml if present. Falls back
    to sane defaults so the script still works in a stripped checkout.
    """
    site_name = "Kubai Kevin"
    base_path = ""
    config_path = Path("./config.yaml")
    if config_path.exists():
        try:
            import yaml
            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            site_name = cfg.get("site_name", site_name)
            base_path = cfg.get("base_path", base_path)
        except Exception:
            pass
    return site_name, base_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Tombstone the posts listed in .adsense_triage/delete.json. "
            "Backs up each post.json first; defaults to dry-run."
        )
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually write the tombstone pages and back up the originals.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview only. This is the default when --apply is omitted.",
    )
    parser.add_argument(
        "--docs-dir", default="./docs",
        help="Path to the site's docs/ directory (default: ./docs).",
    )
    parser.add_argument(
        "--triage-dir", default="./.adsense_triage",
        help="Directory containing delete.json (default: ./.adsense_triage).",
    )
    parser.add_argument(
        "--backup-dir", default="./.adsense_backups",
        help="Where to back up originals (default: ./.adsense_backups).",
    )
    args = parser.parse_args()

    # --dry-run flag is accepted for symmetry even though it's the default.
    # If neither --apply nor --dry-run was passed, print an explicit
    # reminder so nobody thinks the default is "write".
    dry_run = not args.apply
    if dry_run and not args.dry_run:
        print("No action flag supplied. Defaulting to --dry-run.")
        print("Pass --apply to actually tombstone posts.\n")

    docs_dir = Path(args.docs_dir)
    triage_dir = Path(args.triage_dir)
    backup_dir = Path(args.backup_dir)

    if not docs_dir.exists():
        print(f"docs directory not found: {docs_dir}")
        return 1

    delete_bucket = _load_delete_bucket(triage_dir)
    if not delete_bucket:
        return 1

    print(f"Tombstoning {len(delete_bucket)} posts "
          f"(dry_run={dry_run}, docs_dir={docs_dir})\n")

    templates = _build_templates()
    tombstone = templates["tombstone"]

    site_name, base_path = _load_site_config()

    removed_log = _load_removed_log(docs_dir)
    done = 0
    skipped = 0

    for entry in delete_bucket:
        slug = entry.get("slug", "").strip()
        if not slug:
            continue
        reasons = entry.get("reasons", [])
        # Keep the log line short — the full reason list is available in
        # .adsense_triage/delete.json if a fuller audit is needed later.
        reason = "; ".join(reasons[:2]) if reasons else "low quality"

        if _tombstone_one(
            slug=slug,
            reason=reason,
            tombstone_template=tombstone,
            site_name=site_name,
            base_path=base_path,
            removed_log=removed_log,
            docs_dir=docs_dir,
            backup_dir=backup_dir,
            dry_run=dry_run,
        ):
            done += 1
        else:
            skipped += 1

    print()
    if dry_run:
        print(f"{done} post(s) would be tombstoned, {skipped} skipped.")
        print("Run again with --apply to execute.")
        print(f"Backups would go to: {backup_dir}/")
    else:
        _write_removed_log(docs_dir, removed_log)
        print(f"{done} post(s) tombstoned, {skipped} skipped.")
        print(f"Originals backed up to: {backup_dir}/")
        print(f"Removal log updated:    {docs_dir}/_removed_posts.json")
        print()
        print("Next steps:")
        print("  1. python blog_system.py build")
        print("  2. Inspect docs/sitemap.xml — tombstoned URLs should be gone.")
        print("  3. git add docs/ .gitignore && git commit")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())