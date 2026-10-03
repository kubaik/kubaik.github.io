#!/usr/bin/env python3
"""
improve_posts.py - LLM rewrite of every IMPROVE-verdict post so it is honest, grounded and genuinely useful.

Uses the SAME provider fallback chain as blog_system.py (DeepSeek -> Groq -> Gemini -> OpenRouter -> ...), so it
needs no new secrets. Per post it:
  1. strips the old footer / related / docs tails (code re-adds them afterwards)
  2. asks the LLM to rewrite: no first-person or invented evidence, no unverifiable products, numbers only
     if documented / shown / labelled illustrative, keep the technical substance and correct code
  3. VALIDATES the rewrite with the same gates the pipeline uses (fabrication density, entity gate, length,
     code-fence balance, code retained, no H1, title/meta sanity). Failing output is retried with the
     failure reasons fed back; if it still fails the ORIGINAL IS LEFT UNTOUCHED and logged
  4. backs up post.json, writes post.json + index.md, stamps llm_improved_at (resume-safe / idempotent)
  5. runs post_enhancer.py to append related-reading, official-docs links and the honest AI-disclosure footer

Bounded + resumable: `--limit` posts per run (default 10), worst offenders first, already-improved posts skipped.

  python improve_posts.py                          # dry run: lists what would be sent to the LLM
  python improve_posts.py --confirm --limit 10     # rewrite 10 posts
  python improve_posts.py --confirm --slug my-slug # one post
  python improve_posts.py --confirm --limit 0      # everything (long; mind provider rate limits)
Run delete_posts.py first so no LLM budget is spent on posts that are going away.
"""

import argparse
import asyncio
import csv
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import entity_gate
import quality_gate
import triage

FOOTER_SPLIT = "\n---\n\n### About this article"
TAIL_RX = re.compile(
    r"\n## (?:Related reading|Official documentation)\n.*?(?=\n## |\Z)", re.S
)
MARKER_RX = re.compile(r"<!--\s*enhanced:v\d+\s*-->")
CODE_RX = re.compile(r"```.*?```", re.S)

SYSTEM = (
    "You are a meticulous senior technical editor. You rewrite AI-drafted engineering articles so that every "
    "sentence is accurate, honest and useful to a developer. You never invent evidence. Output exactly in the "
    "requested delimiter format and nothing else."
)

PROMPT = """Rewrite the article below.

KEEP: the topic, the genuinely useful technical substance, the overall teaching order, and all CORRECT code
(fix any bugs; keep language tags). Keep it a strong, standalone article of at least {min_words} words.

REMOVE OR REPLACE:
1. Every first-hand claim: "I", "we", "our team", "my company", "a client", "we measured/shipped/cut/migrated".
   Rewrite impersonally ("teams commonly...", "a typical failure mode is...", "the documented behavior is...").
2. Invented evidence: benchmark tables, survey/customer/user counts, "X% of teams", dollar savings, named
   studies, invented timelines ("in March 2026 we..."). Replace a fake benchmark with HOW TO MEASURE it
   (what to instrument, what command, what to compare).
3. Any product/library/version you are not 100% sure exists. Names the checker flagged as unverifiable:
   {unverified}. Remove them or describe the category instead ("a managed LLM gateway").
4. Numbers: keep ONLY (a) documented defaults/limits, (b) arithmetic shown step by step from stated
   assumptions, (c) figures explicitly labelled "illustrative".
5. Filler, hype, repeated points, generic intros, and identical boilerplate sections. Do not pad: where the
   article is thin, ADD depth with a worked example (reasoning shown), a failure-mode analysis, or a decision
   checklist.

STRUCTURE: ## headings only (no # H1, no title heading). Fit the sections to the topic. Add a table only for a
real comparison; add an FAQ only if there are real follow-up questions. End with ONE specific action the reader
can take in the next 30 minutes. Do NOT add links, "Related reading", "Sources" or author/footer text.

{feedback}Sentences the checker flagged as invented first-hand evidence (rewrite or delete them):
{flagged}

OUTPUT FORMAT (exactly):
===TITLE===
<title, max 60 chars, a topic or claim; no first-person, no incident, no invented result>
===META===
<meta description, 110-155 chars, specific, no "Learn about">
===BODY===
<full markdown article>

CURRENT TITLE: {title}

CURRENT ARTICLE:
{body}
"""


def split_content(content: str):
    head = content.split(FOOTER_SPLIT)[0]
    head = MARKER_RX.sub("", head)
    head = TAIL_RX.sub("", head)
    return head.strip()


def flagged_sentences(body: str, limit: int = 12):
    out = []
    for s in triage.sents(body):
        if (
            triage.ANECDOTE.search(s)
            or triage.ANEC2.search(s)
            or (triage.EXP.search(s) and re.search(triage.NUM, s))
        ):
            out.append("- " + s[:220])
        if len(out) >= limit:
            break
    return "\n".join(out) or "- (none matched; still apply rules 1-5)"


def parse(raw: str):
    m = re.search(
        r"===TITLE===\s*(.*?)\s*===META===\s*(.*?)\s*===BODY===\s*(.*)\Z", raw, re.S
    )
    if not m:
        return None
    body = m.group(3).strip()
    body = re.sub(r"^```(?:markdown|md)?\s*\n|\n```\s*$", "", body).strip()
    return m.group(1).strip().strip('"'), m.group(2).strip().strip('"'), body


def required_words(orig_words: int) -> int:
    """>=1500 always; otherwise 70% of the original, capped at 2000 (an original inflated with invented
    material should not force the rewrite to keep inflating)."""
    return max(1500, min(int(orig_words * 0.7), 2000))


def validate(
    orig_words: int,
    orig_codes: int,
    title: str,
    meta: str,
    body: str,
    verify_online: bool,
    orig_unverified: int = 0,
):
    errs = []
    words = len(CODE_RX.sub(" ", body).split())
    need = required_words(orig_words)
    if words < need:
        errs.append(f"too short: {words} words (need >= {need})")
    if body.count("```") % 2:
        errs.append("unbalanced code fences")
    if orig_codes and len(CODE_RX.findall(body)) < max(1, int(orig_codes * 0.6)):
        errs.append("dropped too many code examples")
    if re.search(r"^# [^#]", CODE_RX.sub("", body), re.M):
        errs.append("contains an H1 heading")
    if re.search(r"\]\(https?://", body):
        errs.append("contains links (code adds links)")
    sc = triage.score({"title": title, "content": body})
    if sc["fab_density"] >= 0.06:
        errs.append(
            f"still too much first-hand/invented evidence (density {sc['fab_density']}, "
            f"{sc['exp_claims']} experience claims)"
        )
    bad = entity_gate.unverified_products(body, verify_online)
    # online check is authoritative (limit 2). The offline allowlist cannot know every real tool, so offline
    # we only demand a clear reduction versus the original instead of an absolute number.
    limit = 2 if verify_online else max(2, orig_unverified // 2)
    if len(bad) > limit:
        errs.append("unverifiable products remain: " + ", ".join(bad[:6]))
    if not (8 <= len(title) <= 70):
        errs.append("title length out of range")
    if re.search(r"\b(we|our|i|my)\b", title, re.I):
        errs.append("first-person title")
    if quality_gate.blocked_title_problem(title):
        errs.append("title matches a deleted post")
    if not (60 <= len(meta) <= 170):
        errs.append("meta description length out of range")
    return errs


async def improve_one(bs, d: dict, attempts: int, verify_online: bool):
    body = split_content(d["content"])
    orig_words = len(CODE_RX.sub(" ", body).split())
    orig_codes = len(CODE_RX.findall(body))
    _orig_bad = entity_gate.unverified_products(body)
    unverified = ", ".join(_orig_bad) or "(none)"
    flagged = flagged_sentences(body)
    feedback = ""
    last_errs = []
    for n in range(1, attempts + 1):
        prompt = PROMPT.format(
            min_words=max(1800, min(int(orig_words * 0.8), 2200)),
            unverified=unverified,
            flagged=flagged,
            feedback=feedback,
            title=d["title"],
            body=body,
        )
        raw = await bs._call_api_with_fallback(
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=7000,
        )
        parsed = parse(raw)
        if not parsed:
            last_errs = [
                "output did not follow the ===TITLE===/===META===/===BODY=== format"
            ]
        else:
            title, meta, new_body = parsed
            last_errs = validate(
                orig_words,
                orig_codes,
                title,
                meta,
                new_body,
                verify_online,
                len(_orig_bad),
            )
            if not last_errs:
                return title, meta, new_body, n, []
        feedback = (
            "PREVIOUS ATTEMPT WAS REJECTED. Fix these problems:\n"
            + "\n".join(f"- {e}" for e in last_errs)
            + "\n\n"
        )
    return None, None, None, attempts, last_errs


def load_candidates(verdicts: Path, docs: Path, slugs):
    rows = [
        r
        for r in csv.DictReader(open(verdicts, encoding="utf-8"))
        if r["verdict"] == "IMPROVE"
    ]
    if slugs:
        rows = [r for r in rows if r["slug"] in set(slugs)]
    rows.sort(key=lambda r: -float(r.get("fab_density") or 0))
    out = []
    for r in rows:
        pj = docs / r["slug"] / "post.json"
        if not pj.exists():
            continue
        d = json.loads(pj.read_text("utf-8"))
        if d.get("llm_improved_at"):
            continue
        out.append((r, d, pj))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="docs")
    ap.add_argument("--verdicts", default="triage_report/verdicts.csv")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--slug", action="append")
    ap.add_argument("--limit", type=int, default=10, help="posts per run (0 = all)")
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--backup-dir", default=".quality_review_backups")
    ap.add_argument(
        "--verify-online", action="store_true", help="PyPI/npm check of product names"
    )
    ap.add_argument(
        "--no-finalize", action="store_true", help="skip post_enhancer.py afterwards"
    )
    ap.add_argument("--confirm", action="store_true")
    a = ap.parse_args()

    docs = Path(a.docs)
    cands = load_candidates(Path(a.verdicts), docs, a.slug)
    if a.limit:
        cands = cands[: a.limit]
    print(
        f"{'REWRITING' if a.confirm else 'DRY RUN —'} {len(cands)} post(s) with the LLM"
    )
    if not a.confirm:
        for r, d, _ in cands:
            print(f"  {r['slug']}  fab_density={r['fab_density']}  fixes={r['fixes']}")
        print("Dry run only (no LLM calls made). Re-run with --confirm.")
        return

    import yaml
    from blog_system import BlogSystem

    cfg = yaml.safe_load(open(a.config, encoding="utf-8")) or {}
    bs = BlogSystem(cfg)
    report = {"improved": [], "failed": []}
    for r, d, pj in cands:
        slug = r["slug"]
        print(f"\n→ {slug}")
        try:
            title, meta, body, tries, errs = asyncio.run(
                improve_one(bs, d, a.attempts, a.verify_online)
            )
        except Exception as exc:  # provider chain exhausted etc.
            report["failed"].append(
                {"slug": slug, "errors": [f"LLM call failed: {exc}"]}
            )
            print(f"  ✗ LLM call failed: {exc}")
            continue
        if errs:
            report["failed"].append({"slug": slug, "errors": errs})
            print("  ✗ rejected after retries (original kept): " + "; ".join(errs))
            continue
        bk = Path(a.backup_dir) / slug
        bk.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pj, bk / "post.json.pre_llm")
        old_title = d["title"]
        if quality_gate.topic_problem(
            old_title, []
        ):  # first-person / numeric-outcome title
            d["title"] = title  # only retitle posts whose title was the problem
        d["content"] = body
        d["meta_description"] = meta
        now = datetime.now(timezone.utc).isoformat()
        d["updated_at"] = now
        d["llm_improved_at"] = now
        pj.write_text(json.dumps(d, indent=2, ensure_ascii=False), "utf-8")
        (pj.parent / "index.md").write_text(f"# {d['title']}\n\n{body}\n", "utf-8")
        report["improved"].append(
            {"slug": slug, "attempts": tries, "retitled": d["title"] != old_title}
        )
        print(
            f"  ✓ improved in {tries} attempt(s)"
            + (f"; retitled → {d['title']}" if d["title"] != old_title else "")
        )
    Path("improve_report.json").write_text(json.dumps(report, indent=2))
    print(
        f"\nDone: {len(report['improved'])} improved, {len(report['failed'])} left unchanged (see improve_report.json)"
    )
    if report["failed"]:
        print(
            "Posts that cannot be fixed automatically: re-run triage.py, review, then delete_posts.py --slug <slug>."
        )
    if report["improved"] and not a.no_finalize:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve().parent / "post_enhancer.py"),
                "--docs",
                a.docs,
                "--verdicts",
                a.verdicts,
                "--backup",
                a.backup_dir,
                "--verify-links",
                "--confirm",
            ],
            check=False,
        )


if __name__ == "__main__":
    main()
