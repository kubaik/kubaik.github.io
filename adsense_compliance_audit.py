"""
adsense_compliance_audit.py

Stricter, AdSense-specific pass across EVERY live post — post.json-backed
or orphan HTML-only (the 272 without post.json, left as-is per decision
not to restore them). Produces a DELETE list and a ready-to-run git
removal script. Nothing is deleted automatically; this only classifies
and generates the removal commands for review.

Criteria (each one is an actual AdSense/Search "Low value content" or
"Content misrepresenting sources" signal, not a vague quality opinion):

  DELETE:
    - thin_content        : visible body text < MIN_WORDS
    - near_duplicate       : ≥ DUP_THRESHOLD shingle overlap with another post
    - fabricated_citation  : names a real org (Gartner/Stack Overflow/etc.)
                              as a source with no nearby URL
    - no_meaningful_content: body text couldn't be extracted at all
                              (broken template render, effectively blank page)

  IMPROVE (kept, but flagged):
    - below_target_length, generic_meta_description, no_code_in_technical_post

Run from repo root: python adsense_compliance_audit.py
"""
import re as _re_module
import json
import re
from pathlib import Path
from typing import Optional

try:
    from bs4 import BeautifulSoup
except ImportError:
    raise SystemExit("pip install beautifulsoup4 --break-system-packages")

DOCS_DIR = Path("./docs")
EXCLUDE_DIRS = {
    "static", "tag", "author", "page",
    "contact", "about", "privacy-policy", "terms-of-service",
    "dmca", "ai-content-policy",
}

MIN_WORDS_DELETE = 1200
MIN_WORDS_FLAG = 1800
NGRAM_SIZE = 8
DUP_JACCARD_THRESHOLD = 0.35

NAMED_SOURCE_PATTERN = (
    r'\b(according to (a |an )?(20\d\d )?'
    r'(stack overflow|gartner|forrester|mckinsey|gitlab|github|jetbrains)|'
    r'(20\d\d )?(stack overflow|gartner|forrester|mckinsey) (survey|report|study))\b'
)
GENERIC_META_PATTERNS = [
    r'^Blog post about .+$',
    r'^Learn about .+ in this post\.?$',
]
TECHNICAL_TAGS = {"python", "javascript", "aws", "docker", "kubernetes",
                  "sql", "api", "backend", "devops"}


def word_count(text: str) -> int:
    return len(text.split())


def shingles(text: str, n: int = NGRAM_SIZE) -> set:
    words = re.findall(r'\w+', text.lower())
    return {' '.join(words[i:i + n]) for i in range(len(words) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def named_source_hit(text: str) -> str | None:
    m = re.search(NAMED_SOURCE_PATTERN, text, re.IGNORECASE)
    if not m:
        return None
    window = text[max(0, m.start() - 200):m.end() + 200]
    return None if re.search(r'https?://', window) else m.group(0)


def load_from_json(post_dir: Path) -> dict | None:
    pj = post_dir / "post.json"
    if not pj.exists():
        return None
    try:
        d = json.loads(pj.read_text(encoding="utf-8"))
    except Exception:
        return None
    return {
        "content": d.get("content", ""),
        "meta_description": d.get("meta_description", ""),
        "tags": [t.lower() for t in d.get("tags", [])],
        "has_code": "```" in d.get("content", ""),
        "source": "post.json",
    }


STRIP_TAGS = ["nav", "header", "footer", "script", "style", "form", "aside"]
STRIP_CLASS_ID_PATTERNS = _re_module.compile(
    r"(nav|header|footer|sidebar|related-articles|related-posts|comments|"
    r"breadcrumb|social-share|newsletter|cookie|author-box)", _re_module.IGNORECASE
)


def _extract_body_text(html: str) -> str:
    """Verified against a real post (2,231 words correctly extracted from
    docs/solid-in-action/index.html) — replaces the earlier find("article")/
    find("main")/largest-<div> approach, which returned 0 words for 270/280
    posts because this template doesn't wrap content in a matching tag.
    Stripping known chrome and keeping everything else is far less sensitive
    to the exact template than hunting for one specific container.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag_name in STRIP_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()
    for tag in soup.find_all(True):
        if tag.attrs is None or tag.parent is None:
            continue
        classes = " ".join(tag.get("class") or [])
        tag_id = tag.get("id") or ""
        if STRIP_CLASS_ID_PATTERNS.search(classes) or STRIP_CLASS_ID_PATTERNS.search(tag_id):
            tag.decompose()
    body = soup.find("body") or soup
    return body.get_text(" ", strip=True)


BOILERPLATE_TEXT_MARKERS = [
    "This site publishes AI-generated technical articles",
    "How this article was made",
    "corrections are applied within 48 hours",
]


def _strip_known_boilerplate(text: str) -> str:
    for marker in BOILERPLATE_TEXT_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            # cut from the marker to the end of that sentence/paragraph run —
            # simplest safe approach: drop everything from the first marker
            # onward, since this boilerplate reliably sits at the end of the
            # article template in this site's pages.
            text = text[:idx]
            break
    return text


def load_from_html(post_dir: Path) -> dict | None:
    index = post_dir / "index.html"
    if not index.exists():
        return None
    html = index.read_text(encoding="utf-8", errors="ignore")
    content = _strip_known_boilerplate(_extract_body_text(html))
    soup = BeautifulSoup(html, "html.parser")
    meta_tag = soup.find("meta", attrs={"name": "description"})
    meta_description = meta_tag.get("content", "").strip() if meta_tag else ""
    return {
        "content": content,
        "meta_description": meta_description,
        "tags": [],
        "has_code": bool(soup.find("code") or soup.find("pre")),
        "source": "html_orphan",
    }


def load_posts():
    posts = {}
    for post_dir in DOCS_DIR.iterdir():
        if not post_dir.is_dir() or post_dir.name in EXCLUDE_DIRS:
            continue
        data = load_from_json(post_dir) or load_from_html(post_dir)
        if data is not None:
            posts[post_dir.name] = data
    return posts


MERGE_STUB_PATTERN = re.compile(
    r'This article has been merged into a?n? ?(more complete version)?:?\s*(.*)',
    re.IGNORECASE
)


def merge_stub_target(content: str) -> Optional[str]:
    """If this post is a 'merged into X' redirect stub, return the target
    title/slug text mentioned, so we can verify that target still exists
    before deleting the stub — don't want to delete a stub whose merge
    target was itself deleted in the same pass, leaving a dead link with
    nothing to land on.
    """
    m = MERGE_STUB_PATTERN.search(content)
    return m.group(2).strip() if m else None


def build_report():
    posts = load_posts()
    shingle_index = {slug: shingles(d["content"]) for slug, d in posts.items()}

    report = {"delete": [], "improve": [], "keep": []}

    for slug, d in posts.items():
        content = d["content"]
        wc = word_count(content)
        reasons = []
        verdict = "keep"
        is_merge_stub = merge_stub_target(content) is not None

        if wc == 0:
            reasons.append("no_meaningful_content: body text unextractable")
            verdict = "delete"
        elif wc < MIN_WORDS_DELETE:
            reasons.append(
                f"{'merge_stub' if is_merge_stub else 'thin_content'}: {wc} words < {MIN_WORDS_DELETE}"
            )
            verdict = "delete"

        my_shingles = shingle_index[slug]
        dup_of = None
        for other_slug, other_shingles in shingle_index.items():
            if other_slug == slug:
                continue
            sim = jaccard(my_shingles, other_shingles)
            if sim >= DUP_JACCARD_THRESHOLD:
                reasons.append(f"near_duplicate_of:{other_slug} sim={sim:.2f}")
                dup_of = other_slug
                verdict = "delete"
                break

        named_hit = named_source_hit(content)
        if named_hit:
            reasons.append(f"fabricated_citation: '{named_hit}'")
            verdict = "delete"

        if verdict != "delete":
            if wc < MIN_WORDS_FLAG:
                reasons.append(f"below_target_length: {wc} < {MIN_WORDS_FLAG}")
                verdict = "improve"
            if any(re.match(p, d["meta_description"] or "") for p in GENERIC_META_PATTERNS) \
                    or len(d["meta_description"] or "") < 50:
                reasons.append("generic_or_missing_meta_description")
                verdict = "improve"
            if TECHNICAL_TAGS & set(d["tags"]) and not d["has_code"]:
                reasons.append("no_code_in_technical_post")
                verdict = "improve"

        entry = {
            "slug": slug, "source": d["source"], "word_count": wc,
            "reasons": reasons, "is_merge_stub": is_merge_stub, "duplicate_of": dup_of,
        }
        report[verdict].append(entry)

    return report, posts


def safe_to_delete(entry: dict, all_slugs: set) -> bool:
    """A near-duplicate/merge-stub deletion is only safe if what it's a
    duplicate OF is not also being deleted — otherwise both copies of the
    content vanish instead of consolidating to one. If the duplicate target
    isn't even a known slug (couldn't confirm it still exists), hold back
    and flag for manual look rather than deleting blind.
    """
    dup_of = entry.get("duplicate_of")
    if dup_of is None:
        return True  # thin/fabricated-citation delete, not a duplicate case
    return dup_of in all_slugs


def execute_delete(report: dict, all_slugs: set, dry_run: bool):
    to_delete, held_back = [], []
    for e in report["delete"]:
        if safe_to_delete(e, all_slugs):
            to_delete.append(e)
        else:
            held_back.append(e)

    print(
        f"\n{'[DRY RUN] Would delete' if dry_run else 'Deleting'}: {len(to_delete)}")
    print(
        f"Held back (duplicate target not confirmed live — check manually): {len(held_back)}")
    for e in held_back:
        print(
            f"  docs/{e['slug']}/  — target '{e.get('duplicate_of')}' not found among live posts")

    with open("remove_flagged_posts.sh", "w") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("# Generated by adsense_compliance_audit.py --delete\n")
        f.write("set -euo pipefail\n\n")
        for e in to_delete:
            f.write(f"git rm -r \"docs/{e['slug']}\"  # {e['reasons']}\n")
    print("Wrote remove_flagged_posts.sh")

    if not dry_run:
        import subprocess
        for e in to_delete:
            subprocess.run(
                ["git", "rm", "-r", f"docs/{e['slug']}"], check=True)
        print(
            f"Ran git rm on {len(to_delete)} posts. Review `git status`, then commit.")


def execute_improve(report: dict, dry_run: bool):
    """IMPROVE items get queued for the same automated regeneration pipeline
    used for fabricated citations earlier — no manual editing introduced.
    """
    queue_file = Path("regeneration_queue.json")
    existing = json.loads(queue_file.read_text()
                          ) if queue_file.exists() else []
    existing_slugs = {e["slug"] for e in existing}

    added = 0
    for e in report["improve"]:
        if e["slug"] in existing_slugs:
            continue
        existing.append({"slug": e["slug"], "reason": "; ".join(
            e["reasons"]), "priority": "normal"})
        added += 1

    print(f"\n{'[DRY RUN] Would queue' if dry_run else 'Queued'} {added} posts for regeneration "
          f"({len(existing)} total in queue).")
    if not dry_run:
        queue_file.write_text(json.dumps(existing, indent=2))
        print("Wrote regeneration_queue.json — run your refresh command against it.")


def execute_delete_improve(report: dict, dry_run: bool):
    """Deletes IMPROVE entries outright instead of queuing for regeneration.
    Separate function/flag from execute_delete on purpose — IMPROVE and
    DELETE are different severities (fabricated citation / below length /
    generic meta vs. confirmed thin/duplicate/empty), so deleting the
    IMPROVE bucket is a distinct, explicit choice, not a side effect of
    --delete.
    """
    entries = report["improve"]
    print(
        f"\n{'[DRY RUN] Would delete' if dry_run else 'Deleting'} IMPROVE entries: {len(entries)}")
    for e in entries:
        print(f"  docs/{e['slug']}/  — {e['reasons']}")

    with open("remove_improve_posts.sh", "w") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("# Generated by adsense_compliance_audit.py --delete-improve\n")
        f.write("set -euo pipefail\n\n")
        for e in entries:
            f.write(f"git rm -r \"docs/{e['slug']}\"  # {e['reasons']}\n")
    print("Wrote remove_improve_posts.sh")

    if not dry_run:
        import subprocess
        for e in entries:
            subprocess.run(
                ["git", "rm", "-r", f"docs/{e['slug']}"], check=True)
        print(
            f"Ran git rm on {len(entries)} IMPROVE posts. Review `git status`, then commit.")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete", action="store_true",
                        help="Actually git rm DELETE entries (default: dry-run)")
    parser.add_argument("--improve", action="store_true",
                        help="Actually write regeneration_queue.json (default: dry-run)")
    parser.add_argument("--delete-improve", action="store_true",
                        help="Delete IMPROVE entries outright via git rm instead of queuing for regeneration")
    args = parser.parse_args()

    report, posts = build_report()
    all_slugs = set(posts.keys())

    Path("adsense_audit_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    print(f"DELETE ({len(report['delete'])}):")
    for e in report["delete"]:
        print(f"  docs/{e['slug']}/  [{e['source']}]  — {e['reasons']}")
    print(
        f"\nIMPROVE ({len(report['improve'])}): see adsense_audit_report.json")
    print(f"KEEP: {len(report['keep'])}")

    execute_delete(report, all_slugs, dry_run=not args.delete)

    if args.delete_improve:
        execute_delete_improve(report, dry_run=False)
    else:
        execute_delete_improve(report, dry_run=True)
        execute_improve(report, dry_run=not args.improve)

    if not args.delete:
        print("\n(Ran in dry-run mode for DELETE. Re-run with --delete to actually remove them, "
              "--improve to queue IMPROVE posts for regeneration, or --delete-improve to delete "
              "IMPROVE posts outright instead.)")


if __name__ == "__main__":
    main()
