#!/usr/bin/env python3
"""
corpus_triage.py — code-driven DELETE / IMPROVE / KEEP verdict per post.json.

Targets defects actually observed on the live site (not generic heuristics):
  * truncated display titles ("…") with no recoverable full title
  * irrelevant appended "Official documentation" link (e.g. Redis docs on a
    staging-environments article) = fake sourcing signal
  * junk tags that never appear in the body (tag "Rag" on a JWT post)
  * LLM filler openers
  * template skeleton: same H2 sequence reused corpus-wide (scaled content)
  * first-person experience claims in unreviewed LLM text (fabricated E-E-A-T)
  * thin posts

Usage
  python scripts/corpus_triage.py --docs docs --out triage_out            # report only
  python scripts/corpus_triage.py --docs docs --out triage_out --fix      # apply safe programmatic fixes
Outputs: triage_out/verdicts.json, delete.txt, improve.txt, summary.txt
Exit 3 if DELETE fraction > --max-delete-fraction (mass-delete guard).
"""

import argparse, json, re, shutil, sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

SKIP = {
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
MIN_WORDS_DELETE = 1000
MIN_WORDS_IMPROVE = 1500
SKELETON_DF = 0.08  # an H2 present in >=8% of posts is boilerplate
SKELETON_SHARE = 0.5  # post is templated if >=50% of its H2s are boilerplate
FILLER = [
    re.compile(p, re.I)
    for p in (
        r"this is the writeup that includes",
        r"the failure is quiet",
        r"here is the part that usually goes wrong",
        r"the conventional (advice|wisdom) .{0,60}(incomplete|wrong)",
        r"in today'?s (fast-paced|ever-evolving)",
        r"let'?s dive (in|into)",
    )
]
FIRST_PERSON = re.compile(
    r"\b(I|we) (ran|built|shipped|migrated|measured|saw|found|deployed|lost|burned|"
    r"debugged|learned)\b|\b(in my experience|when I|at my (previous )?(job|company))\b"
)
STOP = set(
    "the a an and or of to in for on with is are be by as at from that this it".split()
)


def toks(s):
    return [t for t in re.findall(r"[a-z0-9]{3,}", s.lower()) if t not in STOP]


# Headings appended by the pipeline itself (not chosen by the model): counting
# them as "boilerplate" inflated the templated-structure flag.
PIPELINE_H2 = {
    "related reading",
    "official documentation",
    "related articles",
    "frequently asked questions",
    "faq",
}


def h2s(md):
    body = re.sub(r"```.*?```", "", md, flags=re.S)
    hs = [
        re.sub(r"[^a-z ]", "", h.lower()).strip()
        for h in re.findall(r"^## (.+)$", body, flags=re.M)
    ]
    return [h for h in hs if h not in PIPELINE_H2]


def doc_section(md):
    m = re.search(
        r"\n## Official documentation\n(.*?)(?=\n## |\n---\n|\Z)", md, flags=re.S
    )
    return m


def irrelevant_doc_links(md):
    m = doc_section(md)
    if not m:
        return False
    body = md[: m.start()].lower()
    for label, url in re.findall(r"\[([^\]]+)\]\((https?://[^)]+)\)", m.group(1)):
        host = urlparse(url).netloc.lower().replace("www.", "").split(".")
        keys = {
            k
            for k in (toks(label) + host[:-1])
            if k not in {"docs", "documentation", "official", "developer"}
        }
        if any(len(re.findall(r"\b%s\b" % re.escape(k), body)) >= 3 for k in keys):
            return False
    return True


def junk_tags(tags, md, tag_freq=None, min_corpus=3):
    low = md.lower()
    out = []
    for t in tags:
        if tag_freq is not None and tag_freq.get(t, 0) >= min_corpus:
            continue  # used across the corpus = real taxonomy, not junk
        parts = [
            w.lower() for w in re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])", t)
        ] or [t.lower()]
        if sum(low.count(w) for w in parts) < 2:
            out.append(t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="docs")
    ap.add_argument("--out", default="triage_out")
    ap.add_argument("--fix", action="store_true")
    ap.add_argument("--max-delete-fraction", type=float, default=0.25)
    ap.add_argument("--skeleton-df", type=float, default=0.04)
    ap.add_argument("--backup", default=".triage_backups")
    a = ap.parse_args()

    posts = {}
    for d in sorted(Path(a.docs).iterdir()):
        pj = d / "post.json"
        if d.is_dir() and d.name not in SKIP and pj.exists():
            try:
                posts[d.name] = (pj, json.loads(pj.read_text(encoding="utf-8")))
            except Exception:
                continue
    n = max(len(posts), 1)
    df = Counter()
    for _, (_, j) in posts.items():
        df.update(set(h2s(j.get("content", ""))))
    tag_freq = Counter(t for _, (_, j) in posts.items() for t in set(j.get("tags", [])))
    boiler = {h for h, c in df.items() if c / n >= a.skeleton_df and h}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "top_h2.txt").write_text(
        "\n".join(f"{c}\t{c/n:.0%}\t{h}" for h, c in df.most_common(40)) + "\n"
    )

    verdicts, fixes = {}, {}
    for slug, (pj, j) in posts.items():
        md = j.get("content", "")
        body = md.split("\n---\n\n### About this article")[0]
        words = len(body.split())
        title, full = j.get("title", ""), j.get("full_title") or ""
        why_del, why_imp = [], []

        if words < MIN_WORDS_DELETE:
            why_del.append(f"thin content ({words} words)")
        elif words < MIN_WORDS_IMPROVE:
            why_imp.append(f"short ({words} words)")
        if title.rstrip().endswith(("…", "...")) and (
            not full or full.rstrip().endswith(("…", "..."))
        ):
            why_imp.append("truncated title, no full_title (regenerate title)")
        elif title.rstrip().endswith(("…", "...")):
            pass  # full_title exists; the build renders it (not a defect)
        if irrelevant_doc_links(md):
            why_imp.append("irrelevant 'Official documentation' link (strip)")
            fixes.setdefault(slug, set()).add("doc")
        jt = junk_tags(j.get("tags", []), body, tag_freq)
        if jt:
            why_imp.append(f"tags absent from body: {jt}")
            fixes.setdefault(slug, set()).add("tags")
        prose = "\n".join(
            l
            for l in re.sub(r"```.*?```", "", body, flags=re.S).split("\n")
            if not l.lstrip().startswith("#")
        )
        if any(f.search(prose[:900]) for f in FILLER):
            why_imp.append("LLM filler opener")
        hs = h2s(body)
        share = sum(h in boiler for h in hs) / len(hs) if hs else 0
        if share >= SKELETON_SHARE and len(hs) >= 4:
            why_imp.append(
                f"templated structure ({share:.0%} corpus-wide boilerplate H2s)"
            )
        fp = len(FIRST_PERSON.findall(re.sub(r"```.*?```", "", body, flags=re.S)))
        if (
            fp >= 3
            and (j.get("monetization_data") or {}).get("review_status")
            != "human_reviewed"
        ):
            why_del.append(
                f"{fp} unverifiable first-person experience claims in unreviewed LLM text"
            )
        elif fp:
            why_imp.append(f"{fp} first-person claim(s) to neutralise")
        if share >= 0.8 and why_imp and words < 1800:
            why_del.append("templated + short: no unique value")

        AUTO = ("irrelevant 'Official", "tags absent", "truncated display title")
        needs_rewrite = [r for r in why_imp if not r.startswith(AUTO)]
        auto_only = [r for r in why_imp if r.startswith(AUTO)]
        verdicts[slug] = {
            "path": f"{a.docs}/{slug}/post.json",
            "verdict": (
                "DELETE"
                if why_del
                else (
                    "IMPROVE"
                    if needs_rewrite
                    else ("FIX_AUTO" if auto_only else "KEEP")
                )
            ),
            "reasons": why_del + why_imp,
            "words": words,
        }

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "verdicts.json").write_text(json.dumps(verdicts, indent=2))
    cnt = Counter(v["verdict"] for v in verdicts.values())
    for name in ("DELETE", "IMPROVE", "FIX_AUTO"):
        (out / f"{name.lower()}.txt").write_text(
            "\n".join(s for s, v in verdicts.items() if v["verdict"] == name) + "\n"
        )
    (out / "summary.txt").write_text(json.dumps(cnt, indent=2))
    print(dict(cnt), f"of {len(verdicts)}")

    if a.fix:
        bdir = Path(a.backup)
        bdir.mkdir(exist_ok=True)
        for slug, kinds in fixes.items():
            pj, j = posts[slug]
            shutil.copy2(pj, bdir / f"{slug}.post.json")
            if "doc" in kinds:
                m = doc_section(j["content"])
                if m:
                    j["content"] = (
                        j["content"][: m.start()] + j["content"][m.end() :]
                    ).replace("\n\n\n\n", "\n\n")
            if "tags" in kinds:
                bad = set(
                    junk_tags(
                        j.get("tags", []),
                        j["content"].split("\n---\n\n### About this article")[0],
                        tag_freq,
                    )
                )
                j["tags"] = [t for t in j.get("tags", []) if t not in bad]
            pj.write_text(json.dumps(j, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"applied safe fixes to {len(fixes)} posts")

    if cnt["DELETE"] / n > a.max_delete_fraction:
        print(
            f"ABORT: DELETE fraction {cnt['DELETE']/n:.0%} > guard {a.max_delete_fraction:.0%}; inspect first"
        )
        sys.exit(3)


if __name__ == "__main__":
    main()
