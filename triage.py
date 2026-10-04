#!/usr/bin/env python3
"""triage.py - deterministic DELETE / IMPROVE verdicts for docs/*/post.json (no LLM, no network).
Report only. To act on it: delete_posts.py (DELETE) then improve_posts.py (IMPROVE) then post_enhancer.py.
"""

import argparse, csv, json, re, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import entity_gate

SKIP = {
    "static",
    "tag",
    "author",
    "about",
    "contact",
    "dmca",
    "page",
    "terms-of-service",
    "privacy-policy",
    "ai-content-policy",
    "privacy",
    "terms",
}
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


LAST_COVERAGE = {}


def _md_post(slug, md_path):
    """Synthesize a post dict from index.md when post.json is missing/corrupt, so EVERY post directory is reviewed."""
    txt = md_path.read_text("utf-8") if md_path.exists() else ""
    lines = txt.split("\n")
    title = (
        lines[0][2:].strip()
        if lines and lines[0].startswith("# ")
        else slug.replace("-", " ")
    )
    body = (
        "\n".join(lines[1:]).strip()
        if lines and lines[0].startswith("# ")
        else txt.strip()
    )
    first = next(
        (p.strip() for p in body.split("\n\n") if p.strip() and not p.startswith("#")),
        "",
    )
    now = ""
    return {
        "title": title,
        "content": body,
        "slug": slug,
        "tags": [],
        "meta_description": first[:155],
        "featured_image": "",
        "created_at": now,
        "updated_at": now,
        "seo_keywords": [],
        "affiliate_links": [],
        "monetization_data": {},
        "twitter_hashtags": "",
    }


def load(docs):
    """Every content directory under docs/ is reviewed. Reserved/site pages are skipped by name; a directory with
    no post.json (or an unparseable one) falls back to index.md; one with neither is reported as an empty post.
    """
    out, skipped, repaired = {}, [], []
    for d in sorted(Path(docs).iterdir()):
        if not d.is_dir() or d.name.startswith((".", "_")):
            continue
        if d.name in SKIP:
            skipped.append(d.name)
            continue
        pj, md = d / "post.json", d / "index.md"
        data = None
        if pj.exists():
            try:
                data = json.loads(pj.read_text("utf-8"))
            except Exception:
                data = None
        if data is None:
            if not md.exists() and not pj.exists():
                skipped.append(d.name)
                continue  # e.g. legal pages: index.html only
            data = _md_post(d.name, md)
            data["_no_json"] = True
            repaired.append(d.name)
        data.setdefault("title", d.name.replace("-", " "))
        data.setdefault("content", "")
        data.setdefault("created_at", "")
        data["_path"] = str(md)
        data["_slug"] = d.name
        out[d.name] = data
    LAST_COVERAGE.update(
        reviewed=len(out), skipped=skipped, rebuilt_from_markdown=repaired
    )
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


def analyze(docs: Path, dup_threshold: float = 0.40):
    """Scan docs/*/post.json and return (rows, dup_pairs). One row per post:
    slug, path, verdict (DELETE|IMPROVE), reasons, fixes, title, created + all score() fields.
    Pure function of the files on disk: no verdict file is read or required."""
    posts = load(Path(docs))
    pairs = similarity(posts, dup_threshold) if posts else {}
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
    cov = LAST_COVERAGE
    print(
        f"reviewed {cov.get('reviewed', 0)} post directories in {docs}/ "
        f"(skipped {len(cov.get('skipped', []))} site/reserved dirs"
        + (
            f"; {len(cov['rebuilt_from_markdown'])} had no readable post.json and were reviewed from index.md"
            if cov.get("rebuilt_from_markdown")
            else ""
        )
        + ")"
    )
    return rows, pairs


def write_report(rows, out):
    Path(out).mkdir(exist_ok=True)
    if rows:
        with open(f"{out}/verdicts.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    Path(f"{out}/delete.txt").write_text(
        "\n".join(
            f'{r["path"]}\t{r["reasons"]}'
            for r in sorted(
                (r for r in rows if r["verdict"] == "DELETE"),
                key=lambda r: -float(r["fab_density"]),
            )
        )
        + "\n"
    )
    Path(f"{out}/improve.txt").write_text(
        "\n".join(r["path"] for r in rows if r["verdict"] == "IMPROVE") + "\n"
    )


def main():
    ap = argparse.ArgumentParser(
        description="Scan docs/ and print DELETE/IMPROVE verdicts; optionally write a report."
    )
    ap.add_argument("--docs", default="docs")
    ap.add_argument("--out", default="triage_report")
    ap.add_argument(
        "--no-report", action="store_true", help="print summary only, write nothing"
    )
    a = ap.parse_args()
    rows, pairs = analyze(Path(a.docs))
    if not a.no_report:
        write_report(rows, a.out)
    print(Counter(r["verdict"] for r in rows), "dup pairs:", len(pairs))
    return rows


if __name__ == "__main__":
    main()
