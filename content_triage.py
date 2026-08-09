"""
content_triage.py (v2) — Automated DELETE / IMPROVE / KEEP classifier.

Changes from v1:
- Citation detection split into two tiers:
    * citation_named_source_review: STRICT — matches a real, named org
      (Stack Overflow, Gartner, McKinsey, Forrester, GitLab, GitHub,
      JetBrains) with no URL nearby. This is the dangerous case
      (readers can go looking for the source and find it doesn't
      exist). Treat as IMPROVE and prioritize.
    * citation_broad_advisory: the old generic "X% of teams" / "survey
      found" pattern. This has a real false-positive rate against
      legitimate "typical figure" phrasing the system prompt allows.
      Reported separately, does NOT alone trigger IMPROVE — surfaced
      for optional human/LLM-judge spot-check, not automatic action.
- Posts skipped because post.json is missing/unparseable are now
  listed explicitly (report["skipped"]) instead of silently vanishing,
  so a mismatch against the live post count is visible.
"""
import json
import re
from pathlib import Path

DOCS_DIR = Path("./docs")

MIN_WORDS_KEEP = 1200
MIN_WORDS_FLAG = 1800
MIN_CODE_BLOCKS_TECHNICAL = 1
NGRAM_SIZE = 8
DUP_JACCARD_THRESHOLD = 0.35

NAMED_SOURCE_PATTERN = (
    r'\b(according to (a |an )?(20\d\d )?'
    r'(stack overflow|gartner|forrester|mckinsey|gitlab|github|jetbrains)|'
    r'(20\d\d )?(stack overflow|gartner|forrester|mckinsey) (survey|report|study))\b'
)
BROAD_ADVISORY_PATTERNS = [
    r'\b(survey|study|report)\s+(of|found|showed)\b',
    r'\b\d{1,3}%\s+of\s+(teams|developers|companies|engineers)\b',
]


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
    if re.search(r'https?://', window):
        return None  # sourced — has a nearby URL, not fabricated
    return m.group(0)


def broad_advisory_hits(text: str) -> list:
    return [p for p in BROAD_ADVISORY_PATTERNS if re.search(p, text, re.IGNORECASE)]


def load_posts():
    posts, skipped = [], []
    for post_dir in DOCS_DIR.iterdir():
        if not post_dir.is_dir() or post_dir.name in ("static", "tag", "author", "page"):
            continue
        pj = post_dir / "post.json"
        if not pj.exists():
            skipped.append(
                {"slug": post_dir.name, "reason": "no post.json found"})
            continue
        try:
            data = json.loads(pj.read_text(encoding="utf-8"))
        except Exception as e:
            skipped.append(
                {"slug": post_dir.name, "reason": f"parse error: {e}"})
            continue
        posts.append((post_dir.name, data))
    return posts, skipped


def main():
    posts, skipped = load_posts()
    shingle_index = {slug: shingles(data.get("content", ""))
                     for slug, data in posts}

    report = {"delete": [], "improve": [], "keep": [], "skipped": skipped}

    for slug, data in posts:
        content = data.get("content", "")
        tags = [t.lower() for t in data.get("tags", [])]
        wc = word_count(content)
        code_blocks = content.count("```") // 2
        reasons = []
        advisory = []
        verdict = "keep"

        if wc < MIN_WORDS_KEEP:
            reasons.append(f"thin_content: {wc} words < {MIN_WORDS_KEEP}")
            verdict = "delete"

        my_shingles = shingle_index[slug]
        for other_slug, other_shingles in shingle_index.items():
            if other_slug == slug:
                continue
            sim = jaccard(my_shingles, other_shingles)
            if sim >= DUP_JACCARD_THRESHOLD:
                reasons.append(f"near_duplicate_of:{other_slug} sim={sim:.2f}")
                verdict = "delete"
                break

        named_hit = named_source_hit(content)
        if named_hit:
            reasons.append(f"citation_named_source_review: '{named_hit}'")
            if verdict != "delete":
                verdict = "improve"

        broad = broad_advisory_hits(content)
        if broad:
            advisory.append(f"citation_broad_advisory: {broad}")

        technical_tags = {"python", "javascript", "aws", "docker", "kubernetes",
                          "sql", "api", "backend", "devops"}
        if technical_tags & set(tags) and code_blocks < MIN_CODE_BLOCKS_TECHNICAL:
            reasons.append("technical_tag_no_code_block")
            if verdict != "delete":
                verdict = "improve"

        if wc < MIN_WORDS_FLAG and verdict == "keep":
            reasons.append(
                f"below_target_length: {wc} words < {MIN_WORDS_FLAG}")
            verdict = "improve"

        if not data.get("meta_description") or len(data.get("meta_description", "")) < 50:
            reasons.append("missing_or_short_meta_description")
            if verdict == "keep":
                verdict = "improve"

        entry = {"slug": slug, "title": data.get("title", ""), "word_count": wc,
                 "reasons": reasons, "advisory_only": advisory}
        report[verdict].append(entry)

    Path("triage_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"Posts scanned: {len(posts)}  |  Skipped (no/bad post.json): {len(skipped)}")
    if skipped:
        print("  Skipped slugs (check these manually — invisible to this audit otherwise):")
        for s in skipped[:30]:
            print(f"    docs/{s['slug']}/  — {s['reason']}")

    print(f"\nDELETE ({len(report['delete'])}):")
    for e in report["delete"][:50]:
        print(f"  docs/{e['slug']}/  — {e['reasons']}")

    print(f"\nIMPROVE ({len(report['improve'])}):")
    for e in report["improve"][:50]:
        print(f"  docs/{e['slug']}/  — {e['reasons']}")

    strict = [e for e in report["improve"] if any(
        "named_source" in r for r in e["reasons"])]
    print(
        f"\n  → of which flagged for STRICT named-source citation (prioritize these {len(strict)}):")
    for e in strict:
        print(f"    docs/{e['slug']}/")

    advisory_only = [e for e in (
        report["improve"] + report["keep"]) if e.get("advisory_only")]
    print(
        f"\nAdvisory-only (broad pattern, not auto-actioned): {len(advisory_only)} — spot-check a sample")

    print(f"\nKEEP: {len(report['keep'])}")
    print("\nFull machine-readable report: triage_report.json")


if __name__ == "__main__":
    main()
