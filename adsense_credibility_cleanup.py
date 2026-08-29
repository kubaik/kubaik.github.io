#!/usr/bin/env python3
"""
adsense_credibility_cleanup.py

Run this locally, from the same directory as blog_system.py / config.yaml
(your repo root), to address the two AdSense findings:

  1. About page vs. article-footer contradiction on human review
  2. Fabricated institutional citations (e.g. "A study by iHub Research
     found...", "A 2025 paper from Makerere University showed...")

IMPORTANT — what I found reading your actual source before writing this:

  * The About-page contradiction is *already fixed in source*. The
    "Editorial process" section in static_site_generator.py
    (_generate_author_page, ~line 2751) is honest and matches the
    article-footer disclosure almost word for word. The old
    "I read every article before it goes live" text is NOT in your
    current templates. That means your live docs/about/index.html is
    almost certainly a STALE BUILD — generated before this section was
    rewritten — and was never rebuilt. So step 1 below is not a content
    rewrite, it's just "run the build so the honest version ships."
    If the rebuilt About page still shows the old text, the fix has to
    happen in static_site_generator.py itself, not here.

  * Fabricated citations live inside each post's `content` field
    (see BlogPost.to_dict/from_dict) as plain markdown text, under
    docs/<slug>/post.json. There's no existing gate that catches
    "attributed to a real institution + specific stat" patterns —
    your quality gates check length/uniqueness/filler, not factual
    attribution. So this needs a new scan, which is step 2 below.

This script deliberately does NOT auto-invent replacement text for
flagged citations. Silently swapping a fake citation for another
guessed sentence would just reintroduce the same problem (unverified
claims shipped without a human looking at them) — which is the exact
thing under review right now. Instead it:

  - rebuilds the site so the About page matches source (safe, automatic)
  - scans every post and produces a report of suspect sentences
  - can auto-strip the two already-confirmed fabricated citations
    (iHub Research, Makerere University) down to a neutral, honest
    version -- with a visible marker so you know it was touched
  - leaves every other match for you to read and decide on, and logs
    that human decision (approve / rewrite / delete) back into an
    audit trail file, which itself is evidence of real editorial review

Usage:
    python adsense_credibility_cleanup.py rebuild
    python adsense_credibility_cleanup.py scan
    python adsense_credibility_cleanup.py fix-known --apply
    python adsense_credibility_cleanup.py review
    python adsense_credibility_cleanup.py all --apply
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

DOCS_DIR = Path("./docs")
REPORT_PATH = Path("./citation_audit_report.json")
DECISIONS_PATH = Path("./citation_audit_decisions.json")

BANNED_ABOUT_PHRASES = [
    "I read every article before it goes live",
    "I correct errors, add specifics from my own experience",
    "remove anything that feels generic or that I can't personally verify",
]

# ── Step 1: rebuild ──────────────────────────────────────────────────────

def rebuild_site():
    """Regenerate the static site from current templates, then verify the
    About page no longer contains the old contradictory claim."""
    if not Path("config.yaml").exists():
        print("config.yaml not found. Run this from your repo root.")
        sys.exit(1)

    import yaml
    from blog_system import BlogSystem
    from static_site_generator import StaticSiteGenerator

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("Rebuilding site from current templates...")
    blog_system = BlogSystem(config)
    StaticSiteGenerator(blog_system).generate_site()

    about_file = DOCS_DIR / "about" / "index.html"
    if not about_file.exists():
        print(f"  ⚠️  Expected {about_file} to exist after build — check output_dir.")
        return

    about_html = about_file.read_text(encoding="utf-8")
    still_bad = [p for p in BANNED_ABOUT_PHRASES if p in about_html]

    if still_bad:
        print("  ❌ Rebuilt About page STILL contains contradictory claims:")
        for p in still_bad:
            print(f"       - \"{p}\"")
        print("     This text isn't coming from static_site_generator.py's")
        print("     current template, so it must be hardcoded somewhere else")
        print("     (a CMS override, a hand-edited file, or a second template")
        print("     path). Search the repo for it directly:")
        print("       grep -rn \"I read every article\" .")
    else:
        print("  ✅ Rebuilt About page matches the honest editorial-process")
        print("     copy. If the live site still shows old text, the fix is")
        print("     just: commit + push docs/, since GitHub Pages serves")
        print("     whatever's in the repo, not what's in your working tree.")


# ── Step 2: scan for fabricated citations ────────────────────────────────

# Patterns that catch "attributed claim + specific number" — the exact
# shape of the two confirmed fabrications (iHub Research, Makerere
# University) and anything structurally similar elsewhere in the corpus.
CITATION_PATTERNS = [
    re.compile(
        r'(?P<sentence>[^.]*?\b(?:study|paper|report|survey|research|analysis)\b'
        r'\s+(?:by|from)\s+[^.]*?\d{1,3}\s?%[^.]*\.)',
        re.IGNORECASE,
    ),
    re.compile(
        r'(?P<sentence>[^.]*?\baccording to\s+(?:a\s+)?[A-Z][\w&.\'\-]*'
        r'(?:\s+[A-Z][\w&.\'\-]*){0,5}[^.]*\d{1,3}\s?%[^.]*\.)',
    ),
    re.compile(
        r'(?P<sentence>[^.]*?\b(?:20\d{2})\s+(?:study|paper|report)\s+'
        r'(?:by|from)\s+[^.]*\.)',
        re.IGNORECASE,
    ),
]

# The two citations already confirmed as fabricated during the AdSense
# audit — exact substrings, safe to auto-flag with high confidence.
KNOWN_FABRICATED_SNIPPETS = [
    "iHub Research",
    "Makerere University",
]


def iter_posts():
    """Yield (post_dir, post_json_path, data) for every post under DOCS_DIR."""
    if not DOCS_DIR.exists():
        print(f"{DOCS_DIR} not found. Run this from your repo root.")
        sys.exit(1)
    for post_dir in sorted(DOCS_DIR.iterdir()):
        if not post_dir.is_dir() or post_dir.name in ("static", "tag"):
            continue
        post_json = post_dir / "post.json"
        if not post_json.exists():
            continue
        try:
            with open(post_json, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  ⚠️  Couldn't read {post_json}: {e}")
            continue
        yield post_dir, post_json, data


def scan_citations():
    """Walk every post's content, flag sentences that attribute a specific
    statistic to a named source, and write a report for human review."""
    findings = []
    checked = 0

    for post_dir, post_json, data in iter_posts():
        checked += 1
        content = data.get("content", "") or ""
        matches = []

        for pattern in CITATION_PATTERNS:
            for m in pattern.finditer(content):
                sentence = m.group("sentence").strip()
                if sentence and sentence not in [x["sentence"] for x in matches]:
                    matches.append({
                        "sentence": sentence,
                        "known_fabricated": any(
                            snip.lower() in sentence.lower()
                            for snip in KNOWN_FABRICATED_SNIPPETS
                        ),
                    })

        if matches:
            findings.append({
                "slug": post_dir.name,
                "post_json": str(post_json),
                "title": data.get("title", ""),
                "matches": matches,
            })

    REPORT_PATH.write_text(
        json.dumps(findings, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    total_matches = sum(len(f["matches"]) for f in findings)
    known = sum(
        1 for f in findings for m in f["matches"] if m["known_fabricated"]
    )
    print(f"Checked {checked} posts.")
    print(f"Flagged {total_matches} suspect citation sentences across "
          f"{len(findings)} posts.")
    print(f"  - {known} match the already-confirmed fabricated pattern "
          f"(iHub Research / Makerere University style).")
    print(f"  - {total_matches - known} are new pattern matches — read these, "
          f"the regex is deliberately loose and WILL include false positives.")
    print(f"\nFull report written to: {REPORT_PATH.resolve()}")
    print("Open it and read every sentence. For each one, decide:")
    print("  KEEP    — it's a real, verifiable, correctly-cited source")
    print("  REWRITE — the point is valid but drop the fake attribution")
    print("  DELETE  — the claim itself doesn't hold up, cut it")
    print("\nRecord your decisions with: python adsense_credibility_cleanup.py review")


# ── Step 3: auto-fix the two already-confirmed fabrications ─────────────

def fix_known(apply: bool):
    """Strip the two confirmed-fabricated citations to a neutral, honest
    framing. Everything else stays untouched — see module docstring for why."""
    if not REPORT_PATH.exists():
        print(f"No {REPORT_PATH} found — run `scan` first.")
        sys.exit(1)

    findings = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    touched = 0

    for entry in findings:
        known = [m for m in entry["matches"] if m["known_fabricated"]]
        if not known:
            continue

        post_json = Path(entry["post_json"])
        with open(post_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        content = data.get("content", "")
        changed = False

        for m in known:
            sentence = m["sentence"]
            if sentence not in content:
                continue  # already edited or content changed since scan
            marker = (
                "[CLAIM REMOVED — previously cited a source/statistic that "
                "could not be verified; see citation_audit_report.json]"
            )
            if apply:
                content = content.replace(sentence, marker)
            changed = True

        if changed:
            touched += 1
            print(f"{'FIXED' if apply else 'WOULD FIX'}: {entry['slug']}")
            for m in known:
                print(f"    - {m['sentence'][:110]}...")

            if apply:
                data["content"] = content
                data["updated_at"] = datetime.now().isoformat()
                data.setdefault("monetization_data", {})["review_status"] = (
                    "credibility_reviewed_" + datetime.now().strftime("%Y%m%d")
                )
                with open(post_json, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

    if not apply:
        print(f"\n{touched} posts would be changed. Re-run with --apply to write changes.")
    else:
        print(f"\n{touched} posts updated. Now run:")
        print("  python adsense_credibility_cleanup.py rebuild")
        print("to regenerate the HTML from the updated post.json files.")


# ── Step 3b: bulk-delete every flagged sentence, not just the known ones ─

def delete_all_flagged(apply: bool):
    """Strip EVERY sentence in the report — known fabrications and the
    looser regex matches alike — down to a visible removal marker, and
    log each one as a 'delete' decision automatically. Use this instead
    of `review` when you've decided the safe default for anything with
    an invented attribution is "cut it," not "rewrite it in place.\""""
    if not REPORT_PATH.exists():
        print(f"No {REPORT_PATH} found — run `scan` first.")
        sys.exit(1)

    findings = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    decisions = {}
    if DECISIONS_PATH.exists():
        decisions = json.loads(DECISIONS_PATH.read_text(encoding="utf-8"))

    touched = 0
    total_sentences = 0

    for entry in findings:
        if not entry["matches"]:
            continue

        post_json = Path(entry["post_json"])
        with open(post_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        content = data.get("content", "")
        changed = False

        for m in entry["matches"]:
            sentence = m["sentence"]
            key = f"{entry['slug']}::{sentence[:80]}"

            if sentence not in content:
                continue  # already edited since scan, or scan/content drifted

            total_sentences += 1
            marker = (
                "[CLAIM REMOVED — previously cited a source/statistic that "
                "could not be verified; see citation_audit_report.json]"
            )
            if apply:
                content = content.replace(sentence, marker)
                decisions[key] = {
                    "slug": entry["slug"],
                    "sentence": sentence,
                    "decision": "delete",
                    "reviewed_at": datetime.now().isoformat(),
                    "method": "bulk_delete_flagged",
                }
            changed = True

        if changed:
            touched += 1
            print(f"{'DELETED' if apply else 'WOULD DELETE'} {len(entry['matches'])} "
                  f"sentence(s) in: {entry['slug']}")

            if apply:
                data["content"] = content
                data["updated_at"] = datetime.now().isoformat()
                data.setdefault("monetization_data", {})["review_status"] = (
                    "credibility_reviewed_" + datetime.now().strftime("%Y%m%d")
                )
                with open(post_json, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

    if apply:
        DECISIONS_PATH.write_text(
            json.dumps(decisions, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\n{touched} posts edited, {total_sentences} sentences removed.")
        print(f"Decisions logged to {DECISIONS_PATH.resolve()}")
        print("Now run:")
        print("  python adsense_credibility_cleanup.py rebuild")
        print("to regenerate the HTML from the updated post.json files.")
    else:
        print(f"\n{touched} posts / {total_sentences} sentences would be changed.")
        print("Re-run with --apply to actually write changes.")


# ── Step 4: log human review decisions for the remaining matches ────────

def review():
    """Interactive pass through every non-known flagged sentence, so a real
    human decision gets recorded — this file itself is the evidence that
    editorial review happened, which is the thing under review."""
    if not REPORT_PATH.exists():
        print(f"No {REPORT_PATH} found — run `scan` first.")
        sys.exit(1)

    findings = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    decisions = {}
    if DECISIONS_PATH.exists():
        decisions = json.loads(DECISIONS_PATH.read_text(encoding="utf-8"))

    for entry in findings:
        for m in entry["matches"]:
            if m["known_fabricated"]:
                continue  # handled by fix-known
            key = f"{entry['slug']}::{m['sentence'][:80]}"
            if key in decisions:
                continue

            print("\n" + "=" * 70)
            print(f"Post: {entry['title']}  ({entry['slug']})")
            print(f"Sentence:\n  {m['sentence']}")
            choice = input("Decision [k=keep / r=needs rewrite / d=delete claim / s=skip]: ").strip().lower()
            if choice == "s":
                continue
            decisions[key] = {
                "slug": entry["slug"],
                "sentence": m["sentence"],
                "decision": {"k": "keep", "r": "rewrite", "d": "delete"}.get(choice, "unresolved"),
                "reviewed_at": datetime.now().isoformat(),
            }
            DECISIONS_PATH.write_text(
                json.dumps(decisions, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    print(f"\nDecisions saved to {DECISIONS_PATH.resolve()}")
    print("Sentences marked 'rewrite' or 'delete' still need the actual edit")
    print("made by hand in the post's post.json content field — this log is")
    print("your record of the review, not the edit itself.")


# ── entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("rebuild", help="Regenerate the site so the About page matches source")
    sub.add_parser("scan", help="Scan all posts for fabricated-citation patterns")
    p_fix = sub.add_parser("fix-known", help="Strip the two confirmed fabrications (iHub Research, Makerere University)")
    p_fix.add_argument("--apply", action="store_true", help="Write changes (default: dry run)")
    p_del = sub.add_parser("delete-flagged", help="Strip EVERY flagged sentence in the report, not just the known ones")
    p_del.add_argument("--apply", action="store_true", help="Write changes (default: dry run)")
    sub.add_parser("review", help="Interactively log a decision for each remaining flagged sentence")
    p_all = sub.add_parser("all", help="Run rebuild + scan + fix-known in sequence")
    p_all.add_argument("--apply", action="store_true", help="Write changes during fix-known (default: dry run)")

    args = parser.parse_args()

    if args.command == "rebuild":
        rebuild_site()
    elif args.command == "scan":
        scan_citations()
    elif args.command == "fix-known":
        fix_known(apply=args.apply)
    elif args.command == "delete-flagged":
        delete_all_flagged(apply=args.apply)
    elif args.command == "review":
        review()
    elif args.command == "all":
        rebuild_site()
        print()
        scan_citations()
        print()
        fix_known(apply=args.apply)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()