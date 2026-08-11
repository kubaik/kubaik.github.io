"""
check_tag_thresholds.py

Confirms indexed tag pages actually have >=8 posts (the pipeline's own
threshold for treating a tag page as substantial enough to index), by
counting how many post.json files reference each tag directly — doesn't
trust the rendered HTML, counts the real source data.

Run from repo root: python check_tag_thresholds.py
"""
import json
from collections import Counter
from pathlib import Path

DOCS_DIR = Path("./docs")
EXCLUDE_DIRS = {
    "static", "tag", "author", "page",
    "contact", "about", "privacy-policy", "terms-of-service",
    "dmca", "ai-content-policy",
}

tag_counts = Counter()
for post_dir in DOCS_DIR.iterdir():
    if not post_dir.is_dir() or post_dir.name in EXCLUDE_DIRS:
        continue
    pj = post_dir / "post.json"
    if not pj.exists():
        continue
    try:
        data = json.loads(pj.read_text(encoding="utf-8"))
    except Exception:
        continue
    for tag in data.get("tags", []):
        tag_counts[tag.strip().lower()] += 1

sample_tags = ["5gbackendoptimization",
               "africandeveloperplatforms", "africatech", "agenticai", "ai"]
print("Sampled tags (should all be >= 8 if the indexing is intentional):")
for t in sample_tags:
    print(f"  {t}: {tag_counts.get(t, 0)} posts")

print(
    f"\nAll tags with >= 8 posts (expected to be indexed): {sum(1 for c in tag_counts.values() if c >= 8)}")
print(
    f"All tags with < 8 posts (expected to be noindex): {sum(1 for c in tag_counts.values() if c < 8)}")
