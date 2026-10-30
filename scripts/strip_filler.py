#!/usr/bin/env python3
"""
strip_filler.py — deterministic, LLM-free removal of filler/first-person
SENTENCES from posts listed by corpus_triage (no rewrite, no updated_at churn
beyond the edited posts, backups kept).

  python scripts/strip_filler.py --docs docs --verdicts triage_out/verdicts.json            # dry run
  python scripts/strip_filler.py --docs docs --verdicts triage_out/verdicts.json --apply
  add --first-person to also drop sentences with unverifiable first-person claims.

Only prose is touched (fenced code is protected); only the first 1200 chars
are scanned for filler openers; a post is skipped if removal would leave the
intro under 40 words.
"""

import argparse, json, re, shutil
from pathlib import Path
from datetime import datetime, timezone

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
SENT = re.compile(r"(?<=[.!?])\s+")


def clean(text, pats, scan=None):
    parts = re.split(r"(```.*?```)", text, flags=re.S)
    removed, used = [], 0
    for i, part in enumerate(parts):
        if part.startswith("```"):
            used += len(part)
            continue
        out = []
        for para in re.split(r"(\n\n)", part):
            if para.startswith(("#", "|", "- ", "* ", "1.")) or para == "\n\n":
                out.append(para)
                used += len(para)
                continue
            keep = []
            for s in SENT.split(para):
                in_scope = scan is None or used < scan
                if in_scope and any(p.search(s) for p in pats):
                    removed.append(s.strip())
                else:
                    keep.append(s)
                used += len(s) + 1
            out.append(" ".join(keep))
        parts[i] = "".join(out)
    return "".join(parts), removed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="docs")
    ap.add_argument("--verdicts", default="triage_out/verdicts.json")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--first-person", action="store_true")
    ap.add_argument("--backup", default=".triage_backups")
    a = ap.parse_args()
    v = json.load(open(a.verdicts))
    changed = 0
    for slug, x in v.items():
        want_filler = any("filler" in r for r in x["reasons"])
        want_fp = a.first_person and any("first-person" in r for r in x["reasons"])
        if not (want_filler or want_fp):
            continue
        pj = Path(a.docs) / slug / "post.json"
        j = json.loads(pj.read_text(encoding="utf-8"))
        body, sep, footer = j["content"].partition("\n---\n\n### About this article")
        removed = []
        if want_filler:
            body, r = clean(body, FILLER, scan=1200)
            removed += r
        if want_fp:
            body, r = clean(body, [FIRST_PERSON])
            removed += r
        if not removed:
            continue
        intro = body.split("\n## ")[0]
        if len(intro.split()) < 40:
            print(f"SKIP {slug}: intro would be too short")
            continue
        print(f"{'APPLY' if a.apply else 'WOULD'} {slug}:")
        for s in removed:
            print("   -", s[:110])
        if a.apply:
            Path(a.backup).mkdir(exist_ok=True)
            shutil.copy2(pj, Path(a.backup) / f"{slug}.strip.post.json")
            j["content"] = body + sep + footer
            j["updated_at"] = datetime.now(timezone.utc).isoformat()
            pj.write_text(json.dumps(j, indent=2, ensure_ascii=False), encoding="utf-8")
        changed += 1
    print(f"{changed} post(s) {'edited' if a.apply else 'would be edited'}")


if __name__ == "__main__":
    main()
