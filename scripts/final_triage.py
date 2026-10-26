#!/usr/bin/env python3
"""
final_triage.py - deterministic, report-first DELETE / IMPROVE / KEEP verdicts.

Targets the signals that the length/fabrication gates already in the repo do NOT
cover, and that drive "scaled content abuse" / "low-value content" rejections:

  1. quarantined posts   : noindexed by fabrication quarantine and still unresolved
                           in regeneration_queue.json  -> DELETE (never regenerate:
                           the topic goes to blocked_topics.json)
  2. skeleton clones     : posts whose H2 outline (generic sections removed) is shared
                           with >= CLONE_GROUP_MAX other posts, or whose title
                           opens with a number-listicle AND shares the outline
                           -> IMPROVE (queue restructure), DELETE only if the group's
                           weakest members are also topically redundant (title Jaccard)
  3. topic redundancy    : title-token Jaccard >= TITLE_DUP within a cluster -> keep best
  4. no primary sources  : zero outbound non-self links -> IMPROVE

Usage:
  python scripts/final_triage.py --docs ./docs --queue ./regeneration_queue.json --out ./triage_final
  python scripts/final_triage.py ... --max-delete-frac 0.15   (safety cap, default 0.15)
Writes: delete.txt, improve.csv, keep.txt, verdicts.json. Never deletes anything.
"""

import argparse, csv, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

GENERIC_H2 = {
    "related reading",
    "official documentation",
    "faq",
    "frequently asked questions",
    "summary",
    "quick reference",
    "common questions",
    "conclusion",
    "sources",
    "references",
    "further reading",
}
TAIL_GROUP_MAX = 5  # same last-two-sections in more than this many posts
CLONE_GROUP_MAX = 6  # same outline in more than this many posts => template clone
TITLE_DUP = 0.72
MIN_OUTLINE = 4  # outlines shorter than this are too small to compare
STOP = set(
    "a an the of for to in on and or with why how what when your you is are 2026 2025".split()
)


def norm_h2(h: str) -> str:
    h = re.sub(r"\d+", "N", h.lower().strip())
    return re.sub(r"[^a-z N]", "", h).strip()


def outline(content: str):
    return tuple(
        x
        for x in (norm_h2(h) for h in re.findall(r"^## (.+)$", content, re.M))
        if x and x not in GENERIC_H2
    )


def toks(title: str):
    return {
        w
        for w in re.findall(r"[a-z0-9]+", title.lower())
        if w not in STOP and not w.isdigit()
    }


def jac(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def load(docs: Path):
    posts = {}
    for pj in docs.glob("*/post.json"):
        try:
            j = json.loads(pj.read_text(encoding="utf-8"))
        except Exception:
            continue
        html = pj.parent / "index.html"
        h = html.read_text(errors="ignore") if html.exists() else ""
        j["_noindex"] = bool(re.search(r"noindex", h, re.I))
        posts[pj.parent.name] = j
    return posts


def ext_links(content: str, base="kubaik.github.io"):
    return [
        u for u in re.findall(r"\]\((https?://[^)\s]+)\)", content) if base not in u
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="./docs")
    ap.add_argument("--queue", default="./regeneration_queue.json")
    ap.add_argument("--out", default="./triage_final")
    ap.add_argument("--max-delete-frac", type=float, default=0.15)
    a = ap.parse_args()

    docs, out = Path(a.docs), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    posts = load(docs)
    if not posts:
        sys.exit("no posts found")

    unresolved = set()
    qp = Path(a.queue)
    if qp.exists():
        for e in json.loads(qp.read_text() or "[]"):
            if (
                e.get("status") in (None, "needs_review", "pending")
                and e.get("action") == "regenerate"
            ):
                unresolved.add(e.get("slug"))

    outlines = {s: outline(p.get("content", "")) for s, p in posts.items()}
    freq = Counter(o for o in outlines.values() if len(o) >= MIN_OUTLINE)
    tails = Counter(o[-2:] for o in outlines.values() if len(o) >= MIN_OUTLINE)
    tk = {s: toks(p.get("title", "")) for s, p in posts.items()}

    # title-redundancy clusters (union-find, inverted index on tokens)
    inv = defaultdict(set)
    for s, t in tk.items():
        for w in t:
            inv[w].add(s)
    parent = {s: s for s in posts}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for s, t in tk.items():
        cand = set().union(*(inv[w] for w in t)) - {s} if t else set()
        for o in cand:
            if jac(t, tk[o]) >= TITLE_DUP:
                parent[find(s)] = find(o)
    clusters = defaultdict(list)
    for s in posts:
        clusters[find(s)].append(s)
    redundant = {}
    for m in clusters.values():
        if len(m) > 1:
            best = max(m, key=lambda x: (len(posts[x].get("content", "").split()), x))
            for x in m:
                if x != best:
                    redundant[x] = best

    verdicts = {}
    for s, p in posts.items():
        c, why, fix, v = p.get("content", ""), [], [], "KEEP"
        if p["_noindex"] and s in unresolved:
            v = "DELETE"
            why.append(
                "quarantined_unverifiable_claims (noindexed, regeneration unresolved)"
            )
        if s in redundant:
            why.append(f"topic_redundant_with:{redundant[s]}")
            v = "DELETE"
        o = outlines[s]
        if len(o) >= MIN_OUTLINE and freq[o] > CLONE_GROUP_MAX:
            why.append(f"template_clone_outline x{freq[o]}")
            fix.append("restructure_outline")
            if v == "KEEP":
                v = "IMPROVE"
        if len(o) >= MIN_OUTLINE and tails[o[-2:]] > TAIL_GROUP_MAX:
            why.append(f"template_clone_ending x{tails[o[-2:]]}")
            fix.append("restructure_outline")
            if v == "KEEP":
                v = "IMPROVE"
        if not ext_links(c):
            why.append("no_primary_sources")
            fix.append("add_sources")
            if v == "KEEP":
                v = "IMPROVE"
        if p.get("author") is None and "Written by" not in c:
            fix.append("byline")
        verdicts[s] = {
            "verdict": v,
            "reasons": why,
            "fixes": sorted(set(fix)),
            "title": p.get("title"),
            "words": len(c.split()),
        }

    dels = [s for s, x in verdicts.items() if x["verdict"] == "DELETE"]
    cap = int(len(posts) * a.max_delete_frac)
    if len(dels) > cap:  # keep the worst offenders only; the rest degrade to IMPROVE
        dels.sort(
            key=lambda s: (
                0 if "quarantined" in " ".join(verdicts[s]["reasons"]) else 1,
                verdicts[s]["words"],
            )
        )
        for s in dels[cap:]:
            verdicts[s]["verdict"] = "IMPROVE"
            verdicts[s]["fixes"].append("dedupe_manually_capped")
        dels = dels[:cap]

    (out / "verdicts.json").write_text(json.dumps(verdicts, indent=1))
    (out / "delete.txt").write_text(
        "".join(
            f"docs/{s}/index.html\t{'; '.join(verdicts[s]['reasons'])}\n"
            for s in sorted(dels)
        )
    )
    (out / "keep.txt").write_text(
        "".join(
            f"docs/{s}/\n"
            for s, x in sorted(verdicts.items())
            if x["verdict"] == "KEEP"
        )
    )
    with open(out / "improve.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "fixes", "reasons", "words", "title"])
        for s, x in sorted(verdicts.items()):
            if x["verdict"] == "IMPROVE":
                w.writerow(
                    [
                        f"docs/{s}/",
                        "|".join(x["fixes"]),
                        "; ".join(x["reasons"]),
                        x["words"],
                        x["title"],
                    ]
                )
    cnt = Counter(x["verdict"] for x in verdicts.values())
    print(
        f"posts={len(posts)} DELETE={cnt['DELETE']} IMPROVE={cnt['IMPROVE']} KEEP={cnt['KEEP']}"
    )


if __name__ == "__main__":
    main()
