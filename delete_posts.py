#!/usr/bin/env python3
"""
delete_posts.py - scan docs/, identify every post that should be deleted, delete it, and make sure it can never
come back. No verdict file is needed: the scan (triage.analyze) runs every time, so results always match the
current contents of docs/.

For each post the scan marks DELETE it:
  1. backs up docs/<slug>/ to .triage_backups/<slug>/
  2. removes docs/<slug>/ and docs/static/og/<slug>*.png
  3. writes {slug,title,reason} to blocked_topics.json  -> blog_system.py will never regenerate the topic
  4. records it in docs/_removed_posts.json
  5. repairs links in the remaining posts that pointed at a deleted post ([text](/slug/) -> text)

Safe by default: DRY RUN unless --confirm. Idempotent.

  python delete_posts.py                              # dry run: scans docs/ and lists what would be deleted
  python delete_posts.py --confirm                    # delete everything the scan marks DELETE
  python delete_posts.py --confirm --max 25           # worst offenders first, 25 per run
  python delete_posts.py --slug some-slug --confirm   # restrict to one flagged post (repeatable)
  python delete_posts.py --force-slug some-slug --confirm   # delete a post the scan did NOT flag
Optional: --verdicts FILE.csv / --list FILE.txt to use a saved report instead of scanning.
"""

import argparse
import csv
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import quality_gate
import triage

LINK_RX = re.compile(
    r"\[([^\]]+)\]\((?:https?://kubaik\.github\.io)?/([a-z0-9][a-z0-9-]*)/?\)"
)


def targets_from_scan(docs: Path):
    rows, _ = triage.analyze(docs)
    return sorted(
        (r for r in rows if r["verdict"] == "DELETE"),
        key=lambda r: -float(r.get("fab_density") or 0),
    )


def targets_from_csv(path: Path):
    rows = [
        r
        for r in csv.DictReader(open(path, encoding="utf-8"))
        if r["verdict"] == "DELETE"
    ]
    return sorted(rows, key=lambda r: -float(r.get("fab_density") or 0))


def targets_from_list(path: Path, docs: Path):
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        p, _, reason = line.partition("\t")
        slug = Path(p.strip()).parent.name
        title = slug.replace("-", " ")
        pj = docs / slug / "post.json"
        if pj.exists():
            try:
                title = json.loads(pj.read_text("utf-8")).get("title", title)
            except ValueError:
                pass
        out.append({"slug": slug, "title": title, "reasons": reason or "triage DELETE"})
    return out


def repair_links(docs: Path, dead: set, confirm: bool) -> int:
    fixed = 0
    for pj in docs.glob("*/post.json"):
        if pj.parent.name in dead:
            continue
        try:
            d = json.loads(pj.read_text("utf-8"))
        except ValueError:
            continue
        n = 0

        def f(m):
            nonlocal n
            if m.group(2) in dead:
                n += 1
                return m.group(1)
            return m.group(0)

        new = LINK_RX.sub(f, d.get("content", ""))
        if n:
            fixed += n
            if confirm:
                d["content"] = new
                pj.write_text(json.dumps(d, indent=2, ensure_ascii=False), "utf-8")
    return fixed


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="docs")
    ap.add_argument(
        "--verdicts", help="use a saved verdicts.csv instead of scanning docs/"
    )
    ap.add_argument(
        "--list",
        help="use a saved delete.txt (path<TAB>reason per line) instead of scanning docs/",
    )
    ap.add_argument(
        "--force-slug",
        action="append",
        help="also delete this slug even if the scan did not flag it (repeatable)",
    )
    ap.add_argument(
        "--slug", action="append", help="delete only this slug (repeatable)"
    )
    ap.add_argument(
        "--max", type=int, default=0, help="max deletions this run (0 = all)"
    )
    ap.add_argument("--backup-dir", default=".triage_backups")
    ap.add_argument("--blocklist", default=str(quality_gate.BLOCKLIST_PATH))
    ap.add_argument(
        "--confirm", action="store_true", help="actually delete (default is a dry run)"
    )
    return ap


def run_pass(a, budget: int = 0) -> int:
    """One scan + delete pass. Returns the number of posts deleted (or, in a dry run, flagged)."""
    docs = Path(a.docs)
    if a.list:
        targets = targets_from_list(Path(a.list), docs)
    elif a.verdicts:
        targets = targets_from_csv(Path(a.verdicts))
    else:
        targets = targets_from_scan(docs)
    for fs in a.force_slug or []:
        if not any(t["slug"] == fs for t in targets):
            title = fs.replace("-", " ")
            try:
                title = json.loads((docs / fs / "post.json").read_text("utf-8")).get(
                    "title", title
                )
            except (OSError, ValueError):
                pass
            targets.append({"slug": fs, "title": title, "reasons": "manually forced"})
    if a.slug:
        wanted = set(a.slug) | set(a.force_slug or [])
        targets = [t for t in targets if t["slug"] in wanted]
    live = [t for t in targets if (docs / t["slug"]).exists()]
    already = len(targets) - len(live)
    if budget:
        live = live[:budget]

    print(
        f"{'DELETING' if a.confirm else 'DRY RUN —'} {len(live)} post(s) flagged for deletion"
        + (f" ({already} previously deleted)" if already else "")
    )
    removed_log_path = docs / "_removed_posts.json"
    removed_log = (
        json.loads(removed_log_path.read_text("utf-8"))
        if removed_log_path.exists()
        else {}
    )
    block_entries = []
    for t in live:
        slug = t["slug"]
        print(f"  {slug}  ::  {t.get('reasons', '')[:100]}")
        block_entries.append(
            {
                "slug": slug,
                "title": t.get("title", slug.replace("-", " ")),
                "reason": t.get("reasons", "triage DELETE"),
                "deleted_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        if not a.confirm:
            continue
        shutil.copytree(docs / slug, Path(a.backup_dir) / slug, dirs_exist_ok=True)
        shutil.rmtree(docs / slug)
        for og in (docs / "static" / "og").glob(f"{slug}*"):
            og.unlink(missing_ok=True)
        removed_log[slug] = {
            "removed_at": block_entries[-1]["deleted_at"],
            "reason": t.get("reasons", ""),
            "source": "delete_posts.py",
        }

    # Links to deleted posts: previously-deleted ones count too, so reruns stay consistent.
    dead = {t["slug"] for t in targets}
    fixed = repair_links(docs, dead, a.confirm)

    if a.confirm and live:
        added = quality_gate.add_to_blocklist(block_entries, Path(a.blocklist))
        removed_log_path.write_text(
            json.dumps(removed_log, indent=2, ensure_ascii=False), "utf-8"
        )
        print(f"blocklist: +{added} entries -> {a.blocklist}")
    print(f"links repaired in remaining posts: {fixed}")
    return len(live)


def main():
    a = build_parser().parse_args()
    scan_mode = not (a.list or a.verdicts or a.slug or a.force_slug)
    total, remaining = 0, a.max
    for n in range(1, 6):  # re-scan until stable: deleting one post of a duplicate pair
        done = run_pass(
            a, remaining
        )  # can change which sibling is flagged on the next scan
        total += done
        if not (a.confirm and scan_mode and done) or (a.max and total >= a.max):
            break
        remaining = a.max - total if a.max else 0
        print(
            f"\n--- re-scanning docs/ after pass {n} (duplicate pairs can resolve differently once a sibling is gone) ---"
        )
    if a.confirm:
        print(f"\nTotal deleted this run: {total}")
    else:
        print(
            "\nDry run only. Re-run with --confirm to delete (it re-scans until no more posts are flagged). "
            "Then: python improve_posts.py --confirm, python blog_system.py build, commit."
        )


if __name__ == "__main__":
    main()
