"""
title_claim_audit.py — report-only. Flags (1) titles promising first-hand
experience or measured results the body doesn't support, and (2) topic
clusters of posts competing for the same subject. Writes title_audit.json
and title_audit_queue.json (entries: retitle | merge_candidate). Deletes nothing.

Run from repo root: python title_claim_audit.py
"""

import json
import re
from pre_save_gate import title_problem
from collections import defaultdict
from pathlib import Path

DOCS = Path("./docs")
SKIP = {
    "static",
    "tag",
    "author",
    "page",
    "about",
    "contact",
    "privacy-policy",
    "terms-of-service",
    "dmca",
    "ai-content-policy",
}

# Titles implying the author personally did/saw/measured something
EXPERIENCE = re.compile(
    r"\b(we|our|i|my|me|us)\b|\b(tested|tried|survived|blew up|broke our|"
    r"what i|months in|in production|real attacks?|real mistakes?|that worked|"
    r"caught|we missed)\b",
    re.I,
)
# Quantified claims in a title: percentages, money, multipliers, 'Nk'
CLAIM = re.compile(
    r"\d+(?:\.\d+)?\s?(?:%|x\b|k\b|m\b)|[$€£]\s?\d+[\d,.]*[kKmM]?|\b\d+\s?(?:percent|times)\b",
    re.I,
)
STOP = {
    "the",
    "a",
    "an",
    "of",
    "to",
    "in",
    "for",
    "and",
    "or",
    "vs",
    "with",
    "your",
    "that",
    "this",
    "is",
    "are",
    "why",
    "how",
    "what",
    "when",
    "now",
    "still",
    "2025",
    "2026",
    "ai",
    "you",
    "it",
    "on",
    "at",
    "by",
    "not",
    "no",
    "do",
    "dont",
}


def load(d: Path):
    pj = d / "post.json"
    if not pj.exists():
        return None
    try:
        j = json.loads(pj.read_text(encoding="utf-8"))
    except Exception:
        return None
    return j.get("title") or d.name.replace("-", " "), j.get("content", "")


def norm(s: str) -> str:
    return re.sub(r"[\s,]", "", s.lower())


def claim_supported(claim: str, body: str) -> bool:
    """A quantified title claim is supported only if the same figure appears
    in the body with a link within 250 chars, or in a worked calculation
    (an '=' or 'assum' within 250 chars of it)."""
    for m in re.finditer(re.escape(claim.strip()), body, re.I):
        w = body[max(0, m.start() - 250) : m.end() + 250]
        if re.search(r"https?://", w) or re.search(
            r"=|assum|illustrat|hypothetical|example", w, re.I
        ):
            return True
    return False


def tokens(title: str):
    return {w for w in re.findall(r"[a-z]{3,}", title.lower()) if w not in STOP}


def main():
    posts = {}
    for d in DOCS.iterdir():
        if d.is_dir() and d.name not in SKIP:
            r = load(d)
            if r:
                posts[d.name] = r

    retitle = []
    for slug, (title, body) in posts.items():
        tp = title_problem(title, body)
        reasons = [tp] if tp else []
        if reasons:
            retitle.append(
                {"slug": slug, "title": title, "reasons": reasons, "action": "retitle"}
            )

    # topic clusters: shared >=3 distinctive tokens, union-find
    parent = {s: s for s in posts}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    tk = {s: tokens(t) for s, (t, _) in posts.items()}
    inv = defaultdict(list)
    for s, ts in tk.items():
        for t in ts:
            inv[t].append(s)
    for s, ts in tk.items():
        cnt = defaultdict(int)
        for t in ts:
            if len(inv[t]) <= 40:  # ignore overly common words
                for o in inv[t]:
                    if o != s:
                        cnt[o] += 1
        for o, n in cnt.items():
            if n >= 3 and n / min(len(ts), len(tk[o]) or 1) >= 0.5:
                parent[find(s)] = find(o)
    groups = defaultdict(list)
    for s in posts:
        groups[find(s)].append(s)
    merges = []
    for members in groups.values():
        if len(members) >= 3:  # 2-post overlaps are tolerable; 3+ is cannibalization
            best = max(members, key=lambda m: len(posts[m][1]))
            merges.append(
                {
                    "keep": best,
                    "merge_or_retire": [m for m in members if m != best],
                    "titles": {m: posts[m][0] for m in members},
                    "action": "merge_candidate",
                }
            )

    Path("title_audit.json").write_text(
        json.dumps({"retitle": retitle, "clusters": merges}, indent=2), encoding="utf-8"
    )
    Path("title_audit_queue.json").write_text(
        json.dumps(
            [
                {
                    "slug": r["slug"],
                    "reason": "; ".join(r["reasons"]),
                    "action": "retitle",
                }
                for r in retitle
            ]
            + [
                {
                    "slug": s,
                    "reason": f"topic cluster; keeper={m['keep']}",
                    "action": "merge_candidate",
                }
                for m in merges
                for s in m["merge_or_retire"]
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"posts={len(posts)}  titles_to_fix={len(retitle)}  topic_clusters(3+)={len(merges)}"
    )
    for r in retitle[:25]:
        print(f"  RETITLE {r['slug']}: {r['reasons']}")
    for m in merges[:15]:
        print(f"  CLUSTER keep={m['keep']}  retire={m['merge_or_retire']}")


if __name__ == "__main__":
    main()
