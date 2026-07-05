#!/usr/bin/env python3
"""
rebuild_site.py

Run this from the root of your kubaik.github.io repo (where blog_system.py
lives) to regenerate the site — sitemap.xml, homepage, posts.json, RSS feed,
etc. — from whatever posts are actually still in docs/ right now.

This is exactly what fixes the stale-sitemap problem: sitemap.xml is only
ever regenerated when `blog_system.py build` runs, so if posts were deleted
without a rebuild afterward, the sitemap keeps listing them. Running this
script rebuilds it fresh, dropping any URL that no longer has a matching
post directory.

USAGE:
    python3 scripts/rebuild_site.py                # cleanup (fallback/similar/dup) + rebuild
    python3 scripts/rebuild_site.py --build-only   # skip cleanup, just rebuild the sitemap/site
    python3 scripts/rebuild_site.py --push         # also git add/commit/push when done

You can run it from anywhere (repo root, scripts/, or elsewhere) — it finds
the repo root itself based on this file's location and chdir's there first.

Safe to run multiple times — it only touches docs/ and only commits if
something actually changed.
"""

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def run(cmd, input_text=None, check=True):
    """Run a shell command, streaming output, returning (returncode, stdout)."""
    result = subprocess.run(
        cmd,
        input=input_text,
        text=True,
        capture_output=True,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if check and result.returncode != 0:
        print(f"❌ Command failed: {' '.join(cmd)}")
        sys.exit(result.returncode)
    return result.returncode, result.stdout


def count_sitemap_urls(docs_dir: Path) -> int:
    sitemap = docs_dir / "sitemap.xml"
    if not sitemap.exists():
        return 0
    content = sitemap.read_text(encoding="utf-8")
    return len(re.findall(r"<loc>", content))


def find_repo_root(start: Path) -> Path:
    """Walk upward from *start* until a directory containing blog_system.py
    is found. This lets the script live in scripts/ (or anywhere else in the
    repo) and still work no matter where it's invoked from."""
    current = start.resolve()
    for _ in range(6):  # repo is never nested more than a few levels deep
        if (current / "blog_system.py").exists():
            return current
        if current.parent == current:
            break
        current = current.parent
    return start.resolve()  # fall back; the existence check below will catch it


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild the blog site locally.")
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Skip the fallback/similar/duplicate cleanup scripts, just rebuild.",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Commit and push docs/ changes to origin/main when done.",
    )
    args = parser.parse_args()

    repo_root = find_repo_root(Path(__file__).parent)
    if not (repo_root / "blog_system.py").exists():
        print("❌ Could not locate blog_system.py by walking up from this script's folder.")
        print("   Make sure this script lives somewhere inside your kubaik.github.io repo")
        print("   (e.g. repo_root/scripts/rebuild_site.py).")
        sys.exit(1)

    os.chdir(repo_root)
    print(
        f"── Using repo root: {repo_root} ──────────────────────────────────────")

    docs_dir = repo_root / "docs"

    print("── Installing dependencies ──────────────────────────────────────────")
    deps = [
        "pyyaml", "aiohttp", "requests", "scikit-learn", "numpy",
        "tweepy", "markdown", "jinja2", "pygments",
    ]
    rc, _ = run([sys.executable, "-m", "pip", "install",
                "--quiet", *deps], check=False)
    if rc != 0:
        # Some systems (Debian/Homebrew Python) block system-wide pip installs
        # ("externally-managed-environment"). Retry with the standard escape
        # hatch rather than failing outright.
        print("   Retrying with --break-system-packages...")
        run([sys.executable, "-m", "pip", "install",
            "--quiet", "--break-system-packages", *deps])

    before = count_sitemap_urls(docs_dir)
    print(
        f"── Sitemap URL count before: {before} ────────────────────────────────")

    if not args.build_only:
        print("\n── Checking for fallback posts ──────────────────────────────────────")
        _, out = run(
            [sys.executable, "utils/delete_fallback_posts.py"], check=False)
        if "would be deleted" in out:
            run([sys.executable, "utils/delete_fallback_posts.py",
                "--delete"], input_text="yes\n")

        print("\n── Checking for similar posts ───────────────────────────────────────")
        _, out = run(
            [sys.executable, "utils/delete_similar_posts.py", "--threshold", "0.75"],
            check=False,
        )
        if "would be deleted" in out:
            run(
                [sys.executable, "utils/delete_similar_posts.py",
                    "--delete", "--threshold", "0.75"],
                input_text="yes\n",
            )

        print("\n── Checking for duplicate posts ─────────────────────────────────────")
        _, out = run(
            [sys.executable, "utils/deduplicate_posts.py", "--threshold", "0.5"],
            check=False,
        )
        if "DELETE" in out:
            run([sys.executable, "utils/deduplicate_posts.py",
                "--delete", "--no-rebuild"])
    else:
        print("── Skipping cleanup scripts (--build-only) ──────────────────────────")

    print("\n── Rebuilding site (sitemap, homepage, posts.json, RSS, etc.) ───────")
    run([sys.executable, "blog_system.py", "build"])

    after = count_sitemap_urls(docs_dir)
    print(
        f"\n── Sitemap URL count after:  {after} ────────────────────────────────")
    print(
        f"── Removed {before - after} stale URL(s) from sitemap.xml ────────────")

    if args.push:
        run(["git", "add", "docs/"])
        diff_rc, _ = run(["git", "diff", "--cached", "--quiet"], check=False)
        if diff_rc == 0:
            print("\nNo changes to commit.")
        else:
            timestamp = datetime.now(
                timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            run(["git", "commit", "-m",
                f"chore: manual rebuild — sync sitemap with live posts [{timestamp}]"])
            run(["git", "push", "origin", "main"])
            print("\n✅ Pushed rebuilt site to origin/main.")
    else:
        print("\nℹ️  Changes were made locally in docs/ but NOT committed/pushed.")
        print("   Review with 'git status' / 'git diff', then commit and push yourself,")
        print("   or re-run this script with --push to do it automatically.")


if __name__ == "__main__":
    main()
