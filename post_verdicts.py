#!/usr/bin/env python3
"""
post_verdicts.py - deterministic, code-only content triage (REPORT ONLY).

Reads docs/<slug>/post.json and classifies every post:

  DELETE   unrecoverable by code: thin, templated/boilerplate, near-duplicate
           of an older post, or built on invented first-person incidents /
           unsourced named statistics.
  IMPROVE  salvageable: structural / metadata / taxonomy defects, borderline
           length, light unqualified-number density.
  KEEP     passes all gates.

Outputs (in --out):  verdicts.json  delete.txt  improve.txt  summary.txt
delete.txt / improve.txt hold one slug per line (feed to delete_posts.py --list
or the regeneration queue). Nothing is modified or deleted here.

No LLM, no network: runs in seconds on thousands of posts, identical result on
every run (stable ordering, no randomness).
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from taxonomy import CANONICAL

    _VALID_TAGS = set(CANONICAL) | {"Backend Engineering"}
except Exception:  # taxonomy optional for this script
    _VALID_TAGS = None

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

_FENCE = re.compile(r"```.*?```", re.S)
_FOOTER = re.compile(r"\n---\s*\n+### About this article.*\Z", re.S)
_TABLE = re.compile(r"^\s*\|.*\|\s*$", re.M)
_HTML = re.compile(r"<[^>]+>")
_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_WORD = re.compile(r"\b\w+\b")

# invented first-person incidents (the generation prompt forbids these)
_INCIDENT = re.compile(
    r"\b(?:when I (?:joined|worked|was)|at my (?:last|previous|old) (?:job|company|startup)"
    r"|(?:I|we)\s+(?:migrated|shipped|burned|got paged|was paged|lost \$|saved \$|cut (?:our|the) (?:bill|costs?|latency))"
    r"|(?:my|our) (?:team|client|startup) (?:saw|found|noticed|measured|lost|saved|shipped)"
    r"|in my experience|I(?:'ve| have) seen (?:this|it) (?:happen|fail))\b",
    re.I,
)
# named statistics / authorities with no link in the same paragraph
_NAMED_STAT = re.compile(
    r"\b(?:according to (?:a |the )?(?:\d{4} )?[A-Z][\w&]+|a \d{4} (?:survey|report|study)"
    r"|studies (?:show|found|suggest)|research (?:shows|from|by)"
    r"|(?:Gartner|Forrester|McKinsey|IDC|Deloitte|Statista)\b)",
)
_PCT = re.compile(r"\b\d{1,3}(?:\.\d+)?\s?%")
_QUALIFIER = re.compile(
    r"\b(?:assum\w+|illustrative|hypothetical|example|for instance|suppose|say|if|"
    r"roughly|typical(?:ly)?|default|documented|e\.g\.)\b",
    re.I,
)


def prose(text: str) -> str:
    t = _FOOTER.sub("", text or "")
    t = _FENCE.sub(" ", t)
    t = _TABLE.sub(" ", t)
    t = _HTML.sub(" ", t)
    t = _LINK.sub(r"\1", t)
    return t


def paragraphs(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", prose(text)) if p.strip()]


def norm_para(p: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\W+", " ", p.lower())).strip()


def load(docs: Path) -> List[Dict]:
    posts = []
    for d in sorted(docs.iterdir()):
        pj = d / "post.json"
        if not d.is_dir() or d.name in SKIP or not pj.exists():
            continue
        try:
            data = json.loads(pj.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        data["_slug"] = d.name
        data["_path"] = str(pj)
        posts.append(data)
    return posts


def near_duplicates(posts: List[Dict], threshold: float) -> Dict[str, str]:
    """newer_slug -> older_slug for cosine >= threshold. Needs scikit-learn."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except Exception:
        return {}
    if len(posts) < 2:
        return {}
    docs = [prose(p.get("content", "")) for p in posts]
    X = TfidfVectorizer(
        sublinear_tf=True,
        stop_words="english",
        min_df=1,
        ngram_range=(1, 2),
        max_features=200_000,
    ).fit_transform(docs)
    out: Dict[str, str] = {}
    n = len(posts)
    for s in range(0, n, 512):
        sims = (X[s : s + 512] @ X.T).toarray()
        for i in range(sims.shape[0]):
            gi = s + i
            for j in (sims[i] >= threshold).nonzero()[0]:
                if j == gi:
                    continue
                a, b = posts[gi], posts[int(j)]
                # keep older (ties: slug order); mark the newer one
                ka = (a.get("created_at", ""), a["_slug"])
                kb = (b.get("created_at", ""), b["_slug"])
                if ka > kb:
                    out.setdefault(a["_slug"], b["_slug"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="./docs")
    ap.add_argument("--out", default="./triage_verdicts")
    ap.add_argument("--min-delete-words", type=int, default=600)
    ap.add_argument("--min-keep-words", type=int, default=1000)
    ap.add_argument("--boilerplate-delete", type=float, default=0.35)
    ap.add_argument("--dup-delete", type=float, default=0.85)
    ap.add_argument("--dup-improve", type=float, default=0.70)
    ap.add_argument("--boilerplate-min-posts", type=int, default=4)
    a = ap.parse_args()

    posts = load(Path(a.docs))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # corpus-wide paragraph frequency (templated paragraphs recur across posts)
    para_posts: Dict[str, set] = defaultdict(set)
    for p in posts:
        for para in paragraphs(p.get("content", "")):
            if len(_WORD.findall(para)) >= 12:
                para_posts[hashlib.md5(norm_para(para).encode()).hexdigest()].add(
                    p["_slug"]
                )

    dup_hard = near_duplicates(posts, a.dup_delete)
    dup_soft = near_duplicates(posts, a.dup_improve)

    verdicts = []
    for p in posts:
        slug, content = p["_slug"], p.get("content", "")
        text = prose(content)
        words = len(_WORD.findall(text))
        paras = paragraphs(content)

        bp_words = 0
        for para in paras:
            w = len(_WORD.findall(para))
            if (
                w >= 12
                and len(para_posts[hashlib.md5(norm_para(para).encode()).hexdigest()])
                >= a.boilerplate_min_posts
            ):
                bp_words += w
        bp_ratio = bp_words / words if words else 0.0

        incidents = len(_INCIDENT.findall(text))
        named_unsourced = 0
        pct_unqualified = 0
        for para in paras:
            if _NAMED_STAT.search(para) and "](" not in para and "http" not in para:
                named_unsourced += 1
            for sent in re.split(r"(?<=[.!?])\s+", para):
                if _PCT.search(sent) and not _QUALIFIER.search(sent):
                    pct_unqualified += 1
        pct_density = pct_unqualified / (words / 1000) if words else 0.0

        h2 = len(re.findall(r"^##\s+\S", content or "", re.M))
        desc = (p.get("meta_description") or "").strip()
        tags = p.get("tags") or []
        md = p.get("monetization_data") or {}

        delete: List[str] = []
        improve: List[str] = []

        if words < a.min_delete_words:
            delete.append(f"Thin content ({words} prose words < {a.min_delete_words})")
        elif words < a.min_keep_words:
            improve.append(f"Borderline length ({words} prose words)")
        if bp_ratio >= a.boilerplate_delete:
            delete.append(
                f"Templated/boilerplate ({bp_ratio:.0%} of prose shared with >= {a.boilerplate_min_posts} other posts)"
            )
        elif bp_ratio >= 0.15:
            improve.append(f"Repeated paragraphs ({bp_ratio:.0%})")
        if slug in dup_hard:
            delete.append(
                f"Near-duplicate of older post '{dup_hard[slug]}' (cosine >= {a.dup_delete})"
            )
        elif slug in dup_soft:
            improve.append(
                f"Overlaps older post '{dup_soft[slug]}' (cosine >= {a.dup_improve}): merge or differentiate"
            )
        if incidents >= 3:
            delete.append(
                f"Invented first-person incidents ({incidents}); unverifiable experience claims"
            )
        elif incidents:
            improve.append(
                f"First-person incident phrasing x{incidents}: rewrite impersonally"
            )
        if named_unsourced >= 3:
            delete.append(
                f"Unsourced named statistics/authorities in {named_unsourced} paragraphs"
            )
        elif named_unsourced:
            improve.append(
                f"Unsourced named statistic in {named_unsourced} paragraph(s)"
            )
        if pct_density > 6:
            improve.append(
                f"High unqualified-percentage density ({pct_density:.1f}/1k words): label illustrative or remove"
            )
        if h2 < 3:
            improve.append(f"Weak structure ({h2} H2 sections)")
        if len(desc) < 70:
            improve.append("Missing/short meta description")
        if _VALID_TAGS is not None and (
            not tags or any(t not in _VALID_TAGS for t in tags)
        ):
            improve.append(
                "Non-canonical tags: run scripts/retag_posts.py --apply (programmatic fix)"
            )
        if md.get("needs_quality_review"):
            improve.append("Flagged needs_quality_review (recovered post)")

        verdict = "DELETE" if delete else ("IMPROVE" if improve else "KEEP")
        verdicts.append(
            {
                "slug": slug,
                "path": p["_path"],
                "verdict": verdict,
                "reasons": delete + improve,
                "metrics": {
                    "prose_words": words,
                    "boilerplate_ratio": round(bp_ratio, 3),
                    "incident_hits": incidents,
                    "unsourced_named": named_unsourced,
                    "pct_per_1k": round(pct_density, 2),
                    "h2": h2,
                },
            }
        )

    (out / "verdicts.json").write_text(json.dumps(verdicts, indent=2), encoding="utf-8")
    (out / "delete.txt").write_text(
        "".join(v["slug"] + "\n" for v in verdicts if v["verdict"] == "DELETE"),
        encoding="utf-8",
    )
    (out / "improve.txt").write_text(
        "".join(v["slug"] + "\n" for v in verdicts if v["verdict"] == "IMPROVE"),
        encoding="utf-8",
    )
    c = Counter(v["verdict"] for v in verdicts)
    reasons = Counter(
        r.split(" (")[0].split(":")[0] for v in verdicts for r in v["reasons"]
    )
    lines = [
        f"posts={len(verdicts)} "
        + " ".join(f"{k}={c.get(k, 0)}" for k in ("DELETE", "IMPROVE", "KEEP")),
        "",
    ]
    lines += [f"{n:5d}  {r}" for r, n in reasons.most_common()]
    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
