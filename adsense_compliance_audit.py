"""
adsense_compliance_audit.py  (v2)

Classifies every live post as DELETE / IMPROVE / KEEP and writes reviewable
removal scripts. Nothing is deleted unless --delete is passed.

Fixes vs. v1:
  * near-duplicate: v1 flagged BOTH members of a pair (each found the other),
    and safe_to_delete() always passed -> both copies were deleted. Now
    duplicates are clustered; the best copy survives, only the rest go.
  * template boilerplate shared by most pages no longer inflates similarity
    (shingles present in >5% of posts are ignored).
  * O(n^2) Jaccard replaced by an inverted index (scales to thousands).
  * mass-delete guard: refuses to delete >40% of the corpus without --force.
  * --delete-improve removed (deleting salvageable posts is never a side
    effect of an audit); IMPROVE posts are queued for regeneration.
"""

import argparse
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

try:
    from bs4 import BeautifulSoup
except ImportError:
    raise SystemExit("pip install beautifulsoup4")

DOCS_DIR = Path("./docs")
EXCLUDE_DIRS = {
    "static",
    "tag",
    "author",
    "page",
    "contact",
    "about",
    "privacy-policy",
    "terms-of-service",
    "dmca",
    "ai-content-policy",
}

MIN_WORDS_DELETE = 1200
MIN_WORDS_FLAG = 1800
NGRAM = 8
DUP_JACCARD = 0.35
BOILERPLATE_DF = 0.05  # ignore shingles that appear in >5% of posts
MAX_DELETE_FRACTION = 0.40

NAMED_SOURCE = re.compile(
    r"\b(according to (a |an )?(20\d\d )?"
    r"(stack overflow|gartner|forrester|mckinsey|gitlab|github|jetbrains|idc|deloitte)|"
    r"(20\d\d )?(stack overflow|gartner|forrester|mckinsey|idc) (survey|report|study))\b",
    re.I,
)
GENERIC_META = [
    re.compile(p)
    for p in (r"^Blog post about .+$", r"^Learn about .+ in this post\.?$")
]
TECH_TAGS = {
    "python",
    "javascript",
    "aws",
    "docker",
    "kubernetes",
    "sql",
    "api",
    "backend",
    "devops",
}
STRIP_TAGS = ["nav", "header", "footer", "script", "style", "form", "aside"]
STRIP_CLS = re.compile(
    r"(nav|header|footer|sidebar|related-articles|related-posts|comments|"
    r"breadcrumb|social-share|newsletter|cookie|author-box)",
    re.I,
)
BOILERPLATE_MARKERS = [
    "This site publishes AI-generated technical articles",
    "How this article was made",
    "How this article was produced",
    "corrections are applied within 48 hours",
]


def words(text: str) -> int:
    return len(text.split())


def shingle_set(text: str) -> Set[int]:
    w = re.findall(r"\w+", text.lower())
    return {hash(" ".join(w[i : i + NGRAM])) for i in range(len(w) - NGRAM + 1)}


def named_source_hit(text: str) -> Optional[str]:
    m = NAMED_SOURCE.search(text)
    if not m:
        return None
    window = text[max(0, m.start() - 200) : m.end() + 200]
    return None if re.search(r"https?://", window) else m.group(0)


def body_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for n in STRIP_TAGS:
        for t in soup.find_all(n):
            t.decompose()
    for t in soup.find_all(True):
        if t.attrs is None or t.parent is None:
            continue
        if STRIP_CLS.search(" ".join(t.get("class") or [])) or STRIP_CLS.search(
            t.get("id") or ""
        ):
            t.decompose()
    text = (soup.find("body") or soup).get_text(" ", strip=True)
    for m in BOILERPLATE_MARKERS:
        i = text.find(m)
        if i != -1:
            text = text[:i]
            break
    return text


def load_post(d: Path) -> Optional[dict]:
    pj = d / "post.json"
    if pj.exists():
        try:
            j = json.loads(pj.read_text(encoding="utf-8"))
            c = j.get("content", "")
            return {
                "content": c,
                "meta": j.get("meta_description", ""),
                "tags": [t.lower() for t in j.get("tags", [])],
                "has_code": "```" in c,
                "source": "post.json",
            }
        except Exception:
            pass
    idx = d / "index.html"
    if idx.exists():
        html = idx.read_text(encoding="utf-8", errors="ignore")
        if re.search(r'http-equiv=["\']refresh', html, re.I) and re.search(
            r"noindex", html, re.I
        ):
            return None  # retired redirect stub, not a live post
        soup = BeautifulSoup(html, "html.parser")
        mt = soup.find("meta", attrs={"name": "description"})
        return {
            "content": body_text(html),
            "meta": (mt.get("content", "") if mt else "").strip(),
            "tags": [],
            "has_code": bool(soup.find("pre") or soup.find("code")),
            "source": "html_orphan",
        }
    return None


def load_posts() -> Dict[str, dict]:
    out = {}
    for d in DOCS_DIR.iterdir():
        if d.is_dir() and d.name not in EXCLUDE_DIRS:
            p = load_post(d)
            if p:
                out[d.name] = p
    return out


def duplicate_clusters(posts: Dict[str, dict]) -> Dict[str, str]:
    """Return {slug_to_delete: canonical_slug}. Best copy (most words, then
    alphabetical) survives; others in its cluster are marked."""
    sh = {s: shingle_set(p["content"]) for s, p in posts.items()}
    df: Dict[int, int] = defaultdict(int)
    for s in sh.values():
        for h in s:
            df[h] += 1
    cutoff = max(3, int(len(posts) * BOILERPLATE_DF))
    for s in sh:
        sh[s] = {h for h in sh[s] if df[h] <= cutoff}

    inv: Dict[int, List[str]] = defaultdict(list)
    for slug, s in sh.items():
        for h in s:
            inv[h].append(slug)

    parent = {s: s for s in posts}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for slug, s in sh.items():
        if not s:
            continue
        counts: Dict[str, int] = defaultdict(int)
        for h in s:
            for o in inv[h]:
                if o != slug:
                    counts[o] += 1
        for o, inter in counts.items():
            union = len(s) + len(sh[o]) - inter
            if union and inter / union >= DUP_JACCARD:
                parent[find(slug)] = find(o)

    clusters: Dict[str, List[str]] = defaultdict(list)
    for s in posts:
        clusters[find(s)].append(s)
    result = {}
    for members in clusters.values():
        if len(members) > 1:
            keep = sorted(members, key=lambda m: (-words(posts[m]["content"]), m))[0]
            for m in members:
                if m != keep:
                    result[m] = keep
    return result


def build_report(min_words: int):
    posts = load_posts()
    dups = duplicate_clusters(posts)
    report = {"delete": [], "improve": [], "keep": []}
    for slug, d in posts.items():
        c, wc, reasons, verdict = d["content"], words(d["content"]), [], "keep"
        if wc == 0:
            reasons.append("no_meaningful_content")
            verdict = "delete"
        elif wc < min_words:
            reasons.append(f"thin_content: {wc} < {min_words}")
            verdict = "delete"
        if slug in dups:
            reasons.append(f"near_duplicate_of:{dups[slug]}")
            verdict = "delete"
        hit = named_source_hit(c)
        if hit:
            reasons.append(f"unsourced_named_citation: '{hit}'")
            verdict = "delete"
        if verdict == "keep":
            if wc < MIN_WORDS_FLAG:
                reasons.append(f"below_target_length: {wc}")
                verdict = "improve"
            if len(d["meta"] or "") < 50 or any(
                p.match(d["meta"] or "") for p in GENERIC_META
            ):
                reasons.append("generic_or_missing_meta_description")
                verdict = "improve"
            if TECH_TAGS & set(d["tags"]) and not d["has_code"]:
                reasons.append("no_code_in_technical_post")
                verdict = "improve"
        report[verdict].append(
            {
                "slug": slug,
                "source": d["source"],
                "word_count": wc,
                "reasons": reasons,
                "duplicate_of": dups.get(slug),
            }
        )
    return report, posts


def write_script(path: str, entries: List[dict]):
    with open(path, "w") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        for e in entries:
            f.write(f'git rm -r "docs/{e["slug"]}"  # {"; ".join(e["reasons"])}\n')
    Path(path).chmod(0o755)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--delete", action="store_true", help="git rm DELETE entries (default dry-run)"
    )
    ap.add_argument(
        "--improve", action="store_true", help="write regeneration_queue.json"
    )
    ap.add_argument("--min-words", type=int, default=MIN_WORDS_DELETE)
    ap.add_argument(
        "--force", action="store_true", help="allow deleting >40%% of corpus"
    )
    a = ap.parse_args()

    report, posts = build_report(a.min_words)
    Path("adsense_audit_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    n, nd = len(posts), len(report["delete"])
    print(
        f"posts={n} delete={nd} improve={len(report['improve'])} keep={len(report['keep'])}"
    )
    for e in report["delete"]:
        print(f"  DELETE docs/{e['slug']}/ [{e['source']}] {e['reasons']}")

    write_script("remove_flagged_posts.sh", report["delete"])
    print("wrote remove_flagged_posts.sh")

    if a.delete:
        if n and nd / n > MAX_DELETE_FRACTION and not a.force:
            raise SystemExit(
                f"Refusing: {nd}/{n} > {MAX_DELETE_FRACTION:.0%}. Inspect report, use --force."
            )
        for e in report["delete"]:
            subprocess.run(["git", "rm", "-r", f"docs/{e['slug']}"], check=True)
        print("git rm done; review `git status` and commit.")

    q = Path("regeneration_queue.json")
    queue = json.loads(q.read_text()) if q.exists() else []
    seen = {e["slug"] for e in queue}
    new = [
        {"slug": e["slug"], "reason": "; ".join(e["reasons"]), "priority": "normal"}
        for e in report["improve"]
        if e["slug"] not in seen
    ]
    print(
        f"{'queued' if a.improve else '[dry-run] would queue'} {len(new)} IMPROVE posts"
    )
    if a.improve:
        q.write_text(json.dumps(queue + new, indent=2))


if __name__ == "__main__":
    main()
