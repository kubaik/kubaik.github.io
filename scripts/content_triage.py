#!/usr/bin/env python3
"""
scripts/content_triage.py
=========================
Code-driven DELETE / IMPROVE / KEEP verdicts for every docs/<slug>/post.json.
No LLM, no network, deterministic: same corpus -> same verdicts.

Usage
-----
  python scripts/content_triage.py --docs docs --out triage_out
  python scripts/content_triage.py --docs docs --out triage_out --fix
  python scripts/content_triage.py --docs docs --out triage_out --max-delete-fraction 0.3

Outputs (in --out): verdicts.json, delete.txt, improve.txt, keep.txt, summary.txt
Exit codes: 0 ok, 3 DELETE share above --max-delete-fraction (safety brake).

Each detector maps to a defect class Google calls out for scaled / low-value
content: thin pages, near-duplicates, templated structure, unverifiable
first-person experience, unsourced numeric claims, filler, junk taxonomy.
Thresholds are heuristics; review delete.txt before acting on it.
"""

import argparse
import hashlib
import json
import re
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

SKIP_DIRS = {
    "static",
    "tag",
    "author",
    "about",
    "contact",
    "dmca",
    "page",
    "privacy-policy",
    "terms-of-service",
    "ai-content-policy",
}

# ── thresholds ──────────────────────────────────────────────────────────────
MIN_WORDS_DELETE = 1000
MIN_WORDS_IMPROVE = 1500
DUP_JACCARD_DELETE = 0.50  # 5-word shingles
TEMPLATE_H2_SHARE = 0.08  # an H2 reused by >= 8% of posts is boilerplate
TEMPLATE_MIN_POSTS = 8
TEMPLATE_POST_FRACTION = 0.5  # >= half of a post's H2s are boilerplate
STAT_DENSITY_IMPROVE = 6.0  # numeric claims per 1000 prose words
FILLER_HITS_IMPROVE = 2
FIRST_PERSON_IMPROVE = 1
FIRST_PERSON_SEVERE = 4  # with unsourced stats -> DELETE
TAG_MISS_FRACTION = 0.5

FIRST_PERSON_RE = re.compile(
    r"\b(in my experience|i['’]ve (seen|run|built|shipped|worked|debugged|migrated)|"
    r"i once\b|when i (worked|was)|at my (previous|last|old)|we (migrated|shipped|rolled out|"
    r"cut|reduced|saved) |my team (and i )?(migrated|shipped|built)|"
    r"on (a|one) (client|project) (i|we)\b|i (personally )?(tested|measured|benchmarked))",
    re.I,
)
FILLER_RE = re.compile(
    r"(in today['’]s (fast[- ]paced|digital|ever[- ]changing)|in the ever[- ]evolving|"
    r"in the (world|realm) of|it['’]s no secret|let['’]s dive (in|into)|without further ado|"
    r"game[- ]changer|unlock the (power|potential)|\bdelve\b|navigate the (complex|landscape)|"
    r"in conclusion,|at the end of the day)",
    re.I,
)
STAT_RE = re.compile(
    r"(?<![\w.])\d[\d,]*(?:\.\d+)?\s?(?:%|ms\b|s\b|x\b|×|gb\b|mb\b|tb\b|kb\b|rps\b|req/s|qps\b|"
    r"k\b|m\b|\$|usd\b|percent\b|times\b|fold\b)",
    re.I,
)
CODE_FENCE_RE = re.compile(r"```.*?```", re.S)
LINK_RE = re.compile(r"\[[^\]]*\]\((https?://[^)\s]+)\)")
H2_RE = re.compile(r"^##\s+(.+?)\s*$", re.M)


def words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def prose(md: str) -> str:
    return CODE_FENCE_RE.sub(" ", md or "")


def norm_h2(h: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", h.lower()).strip()


def shingles(text: str, k: int = 5):
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return {hash(" ".join(toks[i : i + k])) for i in range(max(0, len(toks) - k + 1))}


def load(docs: Path):
    posts = {}
    for d in sorted(docs.iterdir()):
        if not d.is_dir() or d.name in SKIP_DIRS:
            continue
        pj = d / "post.json"
        if not pj.exists():
            continue
        try:
            data = json.loads(pj.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            posts[d.name] = {"_broken": True, "_dir": d}
            continue
        data["_dir"] = d
        posts[d.name] = data
    return posts


def duplicate_losers(posts: dict) -> dict:
    """slug -> slug it duplicates. Keeps the EARLIEST post of each cluster."""
    sh, inv = {}, {}
    for slug, p in posts.items():
        if p.get("_broken"):
            continue
        s = shingles(prose(p.get("content", "")))
        sh[slug] = s
        for h in s:
            if h % 16 == 0:  # ~6% sample is enough to find candidate pairs
                inv.setdefault(h, []).append(slug)
    cand = Counter()
    for slugs in inv.values():
        if 1 < len(slugs) <= 50:
            for i, a in enumerate(slugs):
                for b in slugs[i + 1 :]:
                    cand[(a, b) if a < b else (b, a)] += 1
    order = lambda s: (posts[s].get("created_at", ""), s)
    losers = {}
    for (a, b), shared in cand.items():
        if shared < 3:
            continue
        inter = len(sh[a] & sh[b])
        union = len(sh[a] | sh[b]) or 1
        if inter / union >= DUP_JACCARD_DELETE:
            keep, drop = sorted((a, b), key=order)
            losers.setdefault(drop, keep)
    return losers


def evaluate(posts: dict):
    n_valid = sum(1 for p in posts.values() if not p.get("_broken"))
    h2_counts = Counter()
    for p in posts.values():
        if p.get("_broken"):
            continue
        for h in {norm_h2(x) for x in H2_RE.findall(p.get("content", ""))}:
            h2_counts[h] += 1
    boiler = set()
    if n_valid >= TEMPLATE_MIN_POSTS:
        boiler = {h for h, c in h2_counts.items() if c / n_valid >= TEMPLATE_H2_SHARE}
    dups = duplicate_losers(posts)

    verdicts = {}
    for slug, p in posts.items():
        if p.get("_broken"):
            verdicts[slug] = {"verdict": "DELETE", "reasons": ["Unreadable post.json"]}
            continue
        body = p.get("content", "") or ""
        pr = prose(body)
        wc = words(pr)
        title = (p.get("title") or "").strip()
        full_title = (p.get("full_title") or "").strip()
        md = p.get("monetization_data") or {}
        human = isinstance(md, dict) and md.get("review_status") == "human_reviewed"
        delete, improve = [], []

        # thin
        if wc < MIN_WORDS_DELETE:
            delete.append(f"Thin content ({wc} words < {MIN_WORDS_DELETE})")
        elif wc < MIN_WORDS_IMPROVE:
            improve.append(f"Short ({wc} words < {MIN_WORDS_IMPROVE})")

        # duplicate
        if slug in dups:
            delete.append(
                f"Near-duplicate of docs/{dups[slug]}/ (>= {DUP_JACCARD_DELETE:.0%} shared 5-word shingles)"
            )

        # truncated title
        if title.endswith(("…", "...")) and (not full_title or full_title == title):
            improve.append("Truncated title with no recoverable full_title")

        # boilerplate outline
        h2s = [norm_h2(x) for x in H2_RE.findall(body)]
        if h2s and boiler:
            share = sum(1 for h in h2s if h in boiler) / len(h2s)
            if share >= TEMPLATE_POST_FRACTION:
                improve.append(
                    f"Templated outline ({share:.0%} of H2s reused across corpus)"
                )

        # filler
        filler = len(FILLER_RE.findall(pr))
        if filler >= FILLER_HITS_IMPROVE:
            improve.append(f"LLM filler phrases x{filler}")

        # first-person experience claims without human review
        fp = 0 if human else len(FIRST_PERSON_RE.findall(pr))
        if fp >= FIRST_PERSON_IMPROVE:
            improve.append(
                f"Unverifiable first-person experience claims x{fp} (not human_reviewed)"
            )

        # unsourced numeric claims
        ext_links = len(LINK_RE.findall(body))
        stats = len(STAT_RE.findall(pr))
        density = stats / max(wc, 1) * 1000
        if density >= STAT_DENSITY_IMPROVE and ext_links == 0:
            improve.append(
                f"Unsourced numeric claims ({stats}, {density:.1f}/1000 words, 0 external links)"
            )
            if fp >= FIRST_PERSON_SEVERE:
                delete.append(
                    "Fabricated-experience pattern: many first-person claims + unsourced figures"
                )

        # junk tags
        tags = [t for t in (p.get("tags") or []) if isinstance(t, str)]
        if tags:
            low = pr.lower()
            miss = [
                t
                for t in tags
                if not all(w in low for w in re.findall(r"[a-z0-9]+", t.lower())[:2])
            ]
            if len(miss) / len(tags) >= TAG_MISS_FRACTION:
                improve.append(f"Tags absent from body: {', '.join(miss[:4])}")

        if delete:
            verdicts[slug] = {
                "verdict": "DELETE",
                "reasons": delete + improve,
                "words": wc,
            }
        elif improve:
            verdicts[slug] = {"verdict": "IMPROVE", "reasons": improve, "words": wc}
        else:
            verdicts[slug] = {"verdict": "KEEP", "reasons": [], "words": wc}
    return verdicts


def apply_fix(posts: dict, verdicts: dict, backup_root: Path) -> int:
    """Safe, deterministic fix only: drop tags that never appear in the body."""
    changed = 0
    for slug, v in verdicts.items():
        p = posts[slug]
        if p.get("_broken") or v["verdict"] == "DELETE":
            continue
        if not any(r.startswith("Tags absent") for r in v["reasons"]):
            continue
        low = prose(p.get("content", "")).lower()
        keep = [
            t
            for t in p.get("tags", [])
            if all(w in low for w in re.findall(r"[a-z0-9]+", t.lower())[:2])
        ]
        if keep == p.get("tags"):
            continue
        pj = p["_dir"] / "post.json"
        dest = backup_root / slug
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pj, dest / "post.json")
        data = json.loads(pj.read_text(encoding="utf-8"))
        data["tags"] = keep
        pj.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        changed += 1
    return changed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="docs")
    ap.add_argument("--out", default="triage_out")
    ap.add_argument("--fix", action="store_true")
    ap.add_argument("--max-delete-fraction", type=float, default=0.25)
    a = ap.parse_args(argv)

    docs = Path(a.docs)
    if not docs.is_dir():
        print(f"docs dir not found: {docs}", file=sys.stderr)
        return 2
    posts = load(docs)
    if not posts:
        print("no posts found", file=sys.stderr)
        return 2
    verdicts = evaluate(posts)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "verdicts.json").write_text(
        json.dumps(verdicts, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for name in ("DELETE", "IMPROVE", "KEEP"):
        lines = [
            f"{docs.as_posix()}/{s}/\t{'; '.join(v['reasons']) or '-'}"
            for s, v in sorted(verdicts.items())
            if v["verdict"] == name
        ]
        (out / f"{name.lower()}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )

    c = Counter(v["verdict"] for v in verdicts.values())
    total = len(verdicts)
    reasons = Counter(
        r.split(" (")[0].split(":")[0].split(" x")[0]
        for v in verdicts.values()
        for r in v["reasons"]
    )
    summary = [f"{datetime.now():%Y-%m-%d %H:%M}  posts={total}"]
    summary += [
        f"{k:8s} {c.get(k, 0):5d}  {c.get(k, 0) / total:.0%}"
        for k in ("DELETE", "IMPROVE", "KEEP")
    ]
    summary += ["", "Top reasons:"] + [
        f"  {n:5d}  {r}" for r, n in reasons.most_common(12)
    ]
    (out / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))

    if a.fix:
        n = apply_fix(posts, verdicts, Path(".triage_backups"))
        print(f"--fix: removed absent tags from {n} posts (backups in .triage_backups)")

    if c.get("DELETE", 0) / total > a.max_delete_fraction:
        print(
            f"ABORT: DELETE share {c['DELETE'] / total:.0%} > {a.max_delete_fraction:.0%}. "
            "Thresholds likely wrong; review before deleting.",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
