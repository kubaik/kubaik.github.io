#!/usr/bin/env python3
"""
migrate_domain.py - repoint a site from one hostname to another across a repo.

Lives in utils/ and can be run from anywhere: the repo root is found by
walking up from this file until a .git folder (or blog_system.py + config.yaml)
is seen. Override with --root.

Typical use (blog.kubailabs.com -> kubailabs.com), from the repo root:

    python utils/migrate_domain.py --old blog.kubailabs.com --new kubailabs.com            # dry run
    python utils/migrate_domain.py --old blog.kubailabs.com --new kubailabs.com --apply    # write

Also sweep up leftovers from the original github.io address in the same pass:

    python utils/migrate_domain.py --old blog.kubailabs.com --old kubaik.github.io \
                                   --new kubailabs.com --apply

What it rewrites
  * URL forms of every --old host:  https://old  http://old  //old  and the
    JSON-escaped https:\\/\\/old.  All become  https://NEW.
  * The bare hostname ("old" with no scheme), but only where that is safe:
      - hosts that are NOT *.github.io: everywhere (e.g. SITE_URL = "blog...").
      - *.github.io hosts: only in --bare-files, because "kubaik.github.io" is
        also the repository name and appears in paths and docs that must not
        change. Defaults cover the self-link checks found in this repo.

Safety
  * Dry run by default; --apply is required to write anything.
  * Skips .git, virtualenvs, node_modules, caches and every ".*_backups" folder,
    changelog files, binaries, and files over 25 MB.
  * A .json file that parsed before must still parse after, or it is left alone.
  * Preserves line endings. Idempotent: a second run changes nothing.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

TEXT_EXTS = {
    ".py",
    ".yml",
    ".yaml",
    ".json",
    ".md",
    ".html",
    ".htm",
    ".xml",
    ".txt",
    ".cfg",
    ".toml",
    ".ini",
    ".csv",
    ".js",
    ".css",
    ".svg",
    ".rss",
    ".atom",
}
TEXT_NAMES = {"ads.txt", "robots.txt", "CNAME", "_redirects"}
SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    ".idea",
    ".vscode",
    ".mypy_cache",
    ".pytest_cache",
}
SKIP_FILE_GLOBS = ["*changelog*", "*CHANGELOG*", "*.bak", "*.min.js"]
MAX_BYTES = 25 * 1024 * 1024

DEFAULT_GITHUB_BARE_FILES = [
    "scripts/generate_og_images.py",  # SITE_URL = "kubaik.github.io"
    "triage.py",  # own-domain check for external links
    "scripts/final_triage.py",  # ext_links(base="kubaik.github.io")
    "add_sources.py",  # SELF = "kubaik.github.io"
]


def find_repo_root(start: Path) -> Path:
    """Walk up from `start` to the repo root (.git, or blog_system.py + config.yaml)."""
    for d in [start, *start.parents]:
        if (d / ".git").exists() or (
            (d / "blog_system.py").exists() and (d / "config.yaml").exists()
        ):
            return d
    return start.parent  # script sits in <root>/utils/ by convention


def is_text_candidate(path: Path) -> bool:
    return path.suffix.lower() in TEXT_EXTS or path.name in TEXT_NAMES


def skipped_dir(name: str) -> bool:
    return name in SKIP_DIRS or (name.startswith(".") and name.endswith("_backups"))


def build_patterns(old: str) -> Tuple[re.Pattern, re.Pattern]:
    h = re.escape(old)
    tail = r"(?![\w-])(?!\.\w)"  # not part of a longer hostname
    # https://old  http://old  //old  and JSON-escaped  https:\/\/old
    url = re.compile(r"(?<![\w.-])(?:https?:)?\\?/\\?/" + h + tail, re.IGNORECASE)
    bare = re.compile(r"(?<![\w.@/\\-])" + h + tail, re.IGNORECASE)
    return url, bare


def rewrite(
    text: str, olds: List[str], new: str, bare_ok: Dict[str, bool]
) -> Tuple[str, int]:
    total = 0
    for old in olds:
        url_re, bare_re = build_patterns(old)
        text, n1 = url_re.subn("https://" + new, text)
        total += n1
        if bare_ok[old]:
            text, n2 = bare_re.subn(new, text)
            total += n2
    return text, total


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--old", action="append", required=True, help="hostname to replace (repeatable)"
    )
    ap.add_argument("--new", required=True, help="new hostname, e.g. kubailabs.com")
    ap.add_argument(
        "--root",
        default=None,
        help="repo root (default: auto-detected from this script's location)",
    )
    ap.add_argument(
        "--apply", action="store_true", help="write changes (default is a dry run)"
    )
    ap.add_argument(
        "--bare-files",
        action="append",
        default=None,
        help="extra files (relative to root) where a bare *.github.io host is replaced",
    )
    ap.add_argument(
        "--top", type=int, default=15, help="how many files to list in the summary"
    )
    args = ap.parse_args()

    root = (
        Path(args.root).resolve()
        if args.root
        else find_repo_root(Path(__file__).resolve().parent)
    )
    if not root.is_dir():
        print(f"error: root {root} is not a directory", file=sys.stderr)
        return 2
    new = args.new.strip().lower().rstrip("/")
    olds = [o.strip().lower().rstrip("/") for o in args.old]
    if any(o == new for o in olds):
        print("error: --old and --new are the same host", file=sys.stderr)
        return 2

    bare_files = set(DEFAULT_GITHUB_BARE_FILES) | set(args.bare_files or [])
    self_path = Path(__file__).resolve()

    per_file: Dict[str, int] = {}
    json_skipped: List[str] = []
    errors: List[str] = []
    scanned = 0

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not skipped_dir(d)]
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.resolve() == self_path or not is_text_candidate(p):
                continue
            if any(fnmatch.fnmatch(fn, g) for g in SKIP_FILE_GLOBS):
                continue
            try:
                if p.stat().st_size > MAX_BYTES:
                    continue
                with open(p, "r", encoding="utf-8", newline="") as f:
                    text = f.read()
            except (UnicodeDecodeError, OSError):
                continue
            scanned += 1

            rel = p.relative_to(root).as_posix()
            bare_ok = {
                o: (not o.endswith(".github.io")) or rel in bare_files for o in olds
            }
            new_text, n = rewrite(text, olds, new, bare_ok)
            if n == 0:
                continue

            if p.suffix.lower() == ".json":
                try:
                    json.loads(text)
                except Exception:
                    pass  # was already invalid; do not hold it to a stricter standard
                else:
                    try:
                        json.loads(new_text)
                    except Exception as e:
                        json_skipped.append(f"{rel} ({e})")
                        continue

            per_file[rel] = n
            if args.apply:
                tmp = p.with_name(p.name + ".migrate.tmp")
                try:
                    with open(tmp, "w", encoding="utf-8", newline="") as f:
                        f.write(new_text)
                    os.replace(tmp, p)
                except OSError as e:
                    errors.append(f"{rel}: {e}")
                    tmp.unlink(missing_ok=True)

    total = sum(per_file.values())
    mode = "APPLIED" if args.apply else "DRY RUN (nothing written; add --apply)"
    print(f"{mode}: {', '.join(olds)}  ->  {new}")
    print(f"repo root: {root}")
    print(
        f"scanned {scanned} text files; {len(per_file)} would change / changed; {total} replacements"
    )
    for rel, n in sorted(per_file.items(), key=lambda kv: (-kv[1], kv[0]))[: args.top]:
        print(f"  {n:5d}  {rel}")
    if len(per_file) > args.top:
        print(f"  ... and {len(per_file) - args.top} more files")
    for item in json_skipped:
        print(f"WARNING left unchanged (would break JSON): {item}")
    for item in errors:
        print(f"ERROR could not write: {item}")
    if args.apply and not errors:
        print(
            "\nNext: python blog_system.py build   then re-run this script without --apply;"
            "\nit should report 0 replacements."
        )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
