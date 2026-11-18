#!/usr/bin/env python3
"""
retag_posts.py - one-shot, idempotent, NON-destructive tag migration.

Rewrites ONLY `tags` with canonical taxonomy tags (taxonomy.py). Never touches
content, title, created_at or updated_at, so
it cannot fake freshness. Safe to run on every build: when tags already equal the
derived set nothing is written.

  python scripts/retag_posts.py --docs ./docs            # dry-run report
  python scripts/retag_posts.py --docs ./docs --apply    # write post.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from taxonomy import CANONICAL, derive_site_tags  # noqa: E402

SKIP = {
    "static",
    "tag",
    "author",
    "page",
    "about",
    "contact",
    "dmca",
    "privacy-policy",
    "terms-of-service",
    "ai-content-policy",
}
VALID = set(CANONICAL) | {"Backend Engineering"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="./docs")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    changed = total = 0
    for d in sorted(Path(a.docs).iterdir()):
        pj = d / "post.json"
        if not d.is_dir() or d.name in SKIP or not pj.exists():
            continue
        try:
            data = json.loads(pj.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        total += 1
        old = data.get("tags") or []
        if old and all(t in VALID for t in old) and len(old) <= 4:
            continue  # already canonical -> idempotent no-op
        new = derive_site_tags(
            data.get("full_title") or data.get("title", ""),
            data.get("seo_keywords") or [],
            data.get("content", ""),
        )
        if new == old:
            continue
        changed += 1
        print(f"{d.name}: {old} -> {new}")
        if a.apply:
            data["tags"] = new
            pj.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
    print(
        f"\n{changed}/{total} posts {'retagged' if a.apply else 'would be retagged (dry-run)'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
