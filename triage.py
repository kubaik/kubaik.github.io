#!/usr/bin/env python3
"""triage.py - deterministic DELETE / IMPROVE verdicts for docs/*/post.json (no LLM, no network).
Report only. To act on it: delete_posts.py (DELETE) then improve_posts.py (IMPROVE) then post_enhancer.py.
"""

import argparse, csv, json, re, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import entity_gate

SKIP = {"static", "tag", "author", "about", "contact", "dmca", "page"}
NUM = r"(?:\$\s?\d[\d,.]*\s?[kKmMbB]?|\d[\d,.]*\s?(?:%|ms|s\b|x\b|k\b|M\b|TB|GB|MAU|QPS|req|requests|teams|engineers|users|services|agents|incidents|weeks|months|days))"
FIRST = r"\b(?:we|our|i|my|me)\b"
VERB = r"(?:cut|reduced|shipped|migrated|spent|ran|measured|pulled|tested|built|joined|lost|saved|launched|moved|replaced|burned|deployed|wrote|hit|saw|got|paid|dropped|rolled|switched|scaled|handled|served|processed)"
ANECDOTE = re.compile(rf"{FIRST}\b[^.\n]{{0,90}}\b{VERB}\b[^.\n]{{0,120}}{NUM}", re.I)
ANEC2 = re.compile(
    r"\b(?:at (?:my|our) (?:company|startup|last job)|a (?:\d+-person|\d+-engineer) (?:startup|team)|one (?:solo )?founder I (?:worked|spoke)|(?:team|client|company) I (?:worked|consulted)|in production (?:at|for) (?:my|our)|\d+ teams? (?:we|I) (?:surveyed|talked|interviewed|pulled))",
    re.I,
)
ATTRIB = re.compile(
    r"\b(?:according to|a (?:\d{4} )?(?:survey|report|study)|(?:Datadog|Gartner|Forrester|OWASP|McKinsey|IDC|Stack Overflow|LinearB|DORA|CNCF|Snyk|Sonatype)\b[^.\n]{0,60}(?:found|reported|showed|says|estimates)|studies (?:show|found)|research (?:shows|found)|survey revealed)",
    re.I,
)
TEMPORAL = re.compile(
    r"\b(20\d\d)\b\s*(?:and|to|–|-)\s*\1\b|\blate 20\d\d[^.\n]{0,80}\b(?:March|January|February|April|May|June)\s+20\d\d",
    re.I,
)
LINK = re.compile(r"\]\((https?://[^)\s]+)\)")
HEAD = re.compile(r"^#{2,3}\s+(.+?)\s*$", re.M)
PAST = r"(?:\w+ed|spent|ran|built|saw|got|lost|paid|wrote|took|found|hit|made|had|knew|thought|went|came|chose|kept|left|put|set|began|broke|fixed|shipped|added|increased|observed|measured|reduced|delivered|showed|cut|dropped|moved|switched|rolled|pulled|joined|tried|learned|realized|noticed|discovered|decided)"
EXP = re.compile(
    rf"\b(?:I|we|We|My|Our|our|my)\b(?:'ve|’ve)?\s+(?:\w+\s+){{0,5}}{PAST}\b"
)
STRIP = re.compile(r"```.*?```", re.S)


def sents(t):
    t = STRIP.sub(" ", t)
    return [
        s.strip() for s in re.split(r"(?<=[.!?])\s+|\n{2,}", t) if len(s.strip()) > 25
    ]


def load(docs):
    out = {}
    for p in sorted(docs.glob("*/post.json")):
        if p.parent.name in SKIP:
            continue
        try:
            d = json.loads(p.read_text("utf-8"))
        except Exception:
            continue
        d["_path"] = str(p.parent / "index.md")
        d["_slug"] = p.parent.name
        out[p.parent.name] = d
    return out


def similarity(posts, thr=0.40):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        return {}
    slugs = list(posts)
    docs = [
        posts[s]["title"]
        + " "
        + posts[s]["title"]
        + " "
        + " ".join(STRIP.sub(" ", posts[s]["content"]).split()[:250])
        for s in slugs
    ]
    m = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2), min_df=2, sublinear_tf=True
    ).fit_transform(docs)
    S = cosine_similarity(m)
    pairs = {}
    for i in range(len(slugs)):
        for j in range(i + 1, len(slugs)):
            if S[i, j] >= thr:
                pairs[(slugs[i], slugs[j])] = float(S[i, j])
    return pairs


def score(d):
    c = d["content"]
    ss = sents(c)
    n = max(len(ss), 1)
    anec = [s for s in ss if ANECDOTE.search(s) or ANEC2.search(s)]
    attr = [s for s in ss if ATTRIB.search(s)]
    links = [u for u in LINK.findall(c) if "kubaik.github.io" not in u]
    internal = len(re.findall(r"\]\((?:/|https?://kubaik\.github\.io)", c))
    heads = HEAD.findall(c)
    words = len(STRIP.sub(" ", c).split())
    temporal = len(TEMPORAL.findall(c))
    title_first = bool(re.search(r"\b(we|our|i|my)\b", d["title"], re.I)) or bool(
        re.search(
            r"\b(how we|what we|we (cut|built|moved|replaced))\b", d["title"], re.I
        )
    )
    exp = sum(1 for x in ss if EXP.search(x))
    unv = len(entity_gate.unverified_products(c))
    dens = round((exp + len(anec) + 2 * unv) / n, 3)
    return dict(
        exp_claims=exp,
        unverified_products=unv,
        fab_density=dens,
        words=words,
        anec=len(anec),
        anec_ratio=round(len(anec) / n, 3),
        attr_unsourced=len(attr) if not links else max(0, len(attr) - len(links)),
        ext_links=len(links),
        internal_links=internal,
        temporal=temporal,
        title_first=title_first,
        faq="frequently asked" in c.lower(),
        n_heads=len(heads),
        has_sources_section=bool(
            re.search(r"^#{2,3}\s+(sources|references|further reading)", c, re.I | re.M)
        ),
    )


def verdict(sc, dup_loser):
    why, fix = [], []
    if dup_loser:
        why.append(f"Near-duplicate of {dup_loser}")
    if sc["fab_density"] >= 0.25:
        why.append(
            f"Invented first-hand evidence/benchmarks ({sc['exp_claims']} experience claims, {sc['unverified_products']} unverifiable products)"
        )
    if sc["title_first"] and sc["anec"] >= 4:
        why.append("Title promises first-hand results the author cannot substantiate")
    if sc["temporal"] and sc["anec"] >= 4:
        why.append("Incoherent invented timeline")
    if sc["words"] < 900:
        why.append("Thin content")
    if why:
        return "DELETE", why, fix
    if sc["fab_density"] >= 0.08:
        fix.append("strip_experience_claims")
    if sc["unverified_products"] >= 3:
        fix.append("flag_unverified_products")
    if sc["ext_links"] == 0:
        fix.append("add_sources")
    if sc["internal_links"] == 0:
        fix.append("add_inline_links")
    if sc["title_first"]:
        fix.append("retitle")
    return "IMPROVE", why, fix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="docs")
    ap.add_argument("--out", default="triage_report")
    a = ap.parse_args()
    docs = Path(a.docs)
    posts = load(docs)
    pairs = similarity(posts)
    loser = {}
    for (x, y), s in sorted(pairs.items(), key=lambda kv: -kv[1]):
        sx, sy = score(posts[x]), score(posts[y])
        keep, drop = (
            (x, y)
            if (
                sx["ext_links"],
                -sx["anec"],
                posts[x]["created_at"] < posts[y]["created_at"],
            )
            >= (
                sy["ext_links"],
                -sy["anec"],
                posts[y]["created_at"] < posts[x]["created_at"],
            )
            else (y, x)
        )
        if drop not in loser and keep not in loser:
            loser[drop] = keep
    rows = []
    for s, d in posts.items():
        sc = score(d)
        v, why, fix = verdict(sc, loser.get(s))
        rows.append(
            dict(
                slug=s,
                path=d["_path"],
                verdict=v,
                reasons=" | ".join(why),
                fixes=",".join(fix),
                title=d["title"],
                created=d.get("created_at", "")[:10],
                **sc,
            )
        )
    Path(a.out).mkdir(exist_ok=True)
    with open(f"{a.out}/verdicts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    Path(f"{a.out}/delete.txt").write_text(
        "\n".join(
            f'{r["path"]}\t{r["reasons"]}'
            for r in sorted(
                (r for r in rows if r["verdict"] == "DELETE"),
                key=lambda r: -float(r["fab_density"]),
            )
        )
        + "\n"
    )
    Path(f"{a.out}/improve.txt").write_text(
        "\n".join(r["path"] for r in rows if r["verdict"] == "IMPROVE") + "\n"
    )
    c = Counter(r["verdict"] for r in rows)
    print(c, "dup pairs:", len(pairs))
    return rows


if __name__ == "__main__":
    main()
