#!/usr/bin/env python3
"""
adsense_compliance_audit.py  (v4 — tombstone-safe + LLM improve)

Classifies every post as DELETE / IMPROVE / KEEP (plus RETIRED for tombstones)
and writes reviewable removal scripts. Nothing is deleted unless --delete is
passed. Nothing is rewritten unless --improve-llm is passed.

Changes vs. v3:
  * ROOT-CAUSE FIX for "deleted posts come back": the 24 DELETE entries were
    not real posts, they were TOMBSTONE pages that the build itself writes on
    every run (StaticSiteGenerator.generate_site -> retire_registry
    .sync_ledger_and_tombstones). `git rm` removed them, the next pipeline run
    recreated them from the ledger. Tombstones are now classified RETIRED:
    never git-rm'd, never counted against the delete fraction. They are
    noindex,follow, carry no ads and no content, so they are not an AdSense
    problem; leaving them keeps 200+noindex for already-indexed URLs.
  * To really forget a retired slug, pass --purge-tombstones --ledger <file>
    (repeatable). The slug is removed from the ledger FIRST, then the
    directory, so the build cannot recreate it.
  * Genuinely deleted posts are also pushed into the pipeline's topic
    blocklist (quality_gate) so the generator does not rewrite the same title.
  * --improve-llm: IMPROVE posts are rewritten by the LLM through the same
    provider fallback chain as improve_posts.py (BlogSystem
    ._call_api_with_fallback), validated, backed up, and resume-safe.

Changes vs. v2:
  * Tombstone / removed-post HTML directories are now explicitly detected and
    classified DELETE (they were previously thin-content false positives).
  * Jaccard threshold lowered to 0.28 and boilerplate DF to 0.04 for tighter
    near-duplicate clustering on formulaic AI output.
  * MIN_WORDS_DELETE raised to 1800; MIN_WORDS_FLAG to 2200.
  * Named-source pattern expanded (OWASP, Datadog, LinearB, iHub, Makerere).
  * Missing E-E-A-T footer ("About this article" / AI content policy link)
    now forces IMPROVE.
  * Mass-delete guard lowered to 25% of corpus.
  * IMPROVE entries written to regeneration_queue.json with priority.
"""

import argparse
import asyncio
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
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

MIN_WORDS_DELETE = 1800
MIN_WORDS_FLAG = 2200
NGRAM = 8
DUP_JACCARD = 0.28
BOILERPLATE_DF = 0.04
MAX_DELETE_FRACTION = 0.25
BACKUP_DIR = Path(".quality_review_backups")
FOOTER_SPLIT = "\n---\n\n### About this article"
CODE_RX = re.compile(r"```.*?```", re.S)
TAIL_RX = re.compile(
    r"\n## (?:Related reading|Official documentation)\n.*?(?=\n## |\Z)", re.S
)
MARKER_RX = re.compile(r"<!--\s*enhanced:v\d+\s*-->")

NAMED_SOURCE = re.compile(
    r"\b(according to (a |an )?(20\d\d )?"
    r"(stack overflow|gartner|forrester|mckinsey|gitlab|github|jetbrains|idc|deloitte|"
    r"owasp|datadog|linearb|ihub|makerere)|"
    r"(20\d\d )?(stack overflow|gartner|forrester|mckinsey|idc|owasp) (survey|report|study))\b",
    re.I,
)
GENERIC_META = [
    re.compile(p)
    for p in (
        r"^Blog post about .+$",
        r"^Learn about .+ in this post\.?$",
        r"^Practical guide to .+$",
    )
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
    "llm",
    "agent",
    "rag",
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
    "individually reviewed or edited line by line",
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
    window = text[max(0, m.start() - 250) : m.end() + 250]
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
            return None
        if "Post removed" in html or "This post has been removed" in html:
            return {
                "content": "",
                "meta": "",
                "tags": [],
                "has_code": False,
                "source": "tombstone",
            }
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
    sh = {s: shingle_set(p["content"]) for s, p in posts.items() if p["content"]}
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
    report = {"delete": [], "improve": [], "keep": [], "retired": []}
    for slug, d in posts.items():
        c, wc, reasons, verdict = d["content"], words(d["content"]), [], "keep"
        if d["source"] == "tombstone":
            # Pipeline-owned page (noindex, no ads, no content). NOT deletable
            # by git rm: the build rewrites it from the retire ledger.
            report["retired"].append(
                {
                    "slug": slug,
                    "source": d["source"],
                    "word_count": 0,
                    "reasons": ["tombstone (pipeline-managed, noindex)"],
                    "duplicate_of": None,
                }
            )
            continue
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
            if len(d["meta"] or "") < 60 or any(
                p.match(d["meta"] or "") for p in GENERIC_META
            ):
                reasons.append("generic_or_missing_meta_description")
                verdict = "improve"
            if TECH_TAGS & set(d["tags"]) and not d["has_code"]:
                reasons.append("no_code_in_technical_post")
                verdict = "improve"
            if "### About this article" not in c and "AI content policy" not in c:
                reasons.append("missing_eeat_footer")
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


# --------------------------------------------------------------------------
# Tombstone / ledger handling
# --------------------------------------------------------------------------
def _slug_of(entry) -> Optional[str]:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        return entry.get("slug")
    return None


def purge_from_ledger(ledger: Path, slugs: Set[str]) -> int:
    """Remove slugs from a JSON ledger shaped as {slug: {...}}, [slug, ...],
    [{"slug": ...}, ...] or {"retired": [...]}. Returns number removed."""
    if not ledger.exists():
        print(f"  ledger not found, skipped: {ledger}")
        return 0
    data = json.loads(ledger.read_text(encoding="utf-8"))
    removed = 0

    def clean(node):
        nonlocal removed
        if isinstance(node, dict):
            for k in [k for k in node if k in slugs]:
                del node[k]
                removed += 1
            for k, v in node.items():
                node[k] = clean(v)
            return node
        if isinstance(node, list):
            out = []
            for item in node:
                if _slug_of(item) in slugs:
                    removed += 1
                else:
                    out.append(clean(item))
            return out
        return node

    shutil.copy2(ledger, ledger.with_suffix(ledger.suffix + ".bak"))
    ledger.write_text(
        json.dumps(clean(data), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  {ledger}: removed {removed} entr{'y' if removed == 1 else 'ies'}")
    return removed


def block_deleted_topics(entries: List[dict], posts: Dict[str, dict]):
    """Best-effort: tell the topic generator not to recreate deleted posts."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import quality_gate  # type: ignore
    except Exception as exc:
        print(f"  (blocklist not updated: quality_gate unavailable: {exc})")
        return
    adder = next(
        (
            getattr(quality_gate, n)
            for n in ("add_to_blocklist", "block_posts", "append_blocklist")
            if hasattr(quality_gate, n)
        ),
        None,
    )
    if not adder:
        print(
            "  (blocklist not updated: quality_gate has no add_to_blocklist(); "
            "see README note — deleted topics may be regenerated)"
        )
        return
    items = [{"slug": e["slug"], "title": e["slug"].replace("-", " ")} for e in entries]
    try:
        adder(items)
        print(f"  blocklisted {len(items)} deleted slug(s) in quality_gate")
    except Exception as exc:
        print(f"  (blocklist update failed: {exc})")


def write_script(path: str, entries: List[dict]):
    with open(path, "w") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        for e in entries:
            f.write(f'git rm -r "docs/{e["slug"]}"  # {"; ".join(e["reasons"])}\n')
    Path(path).chmod(0o755)


# --------------------------------------------------------------------------
# LLM improve (same provider chain / contract as improve_posts.py)
# --------------------------------------------------------------------------
LLM_SYSTEM = (
    "You are a meticulous senior technical editor. You rewrite AI-drafted engineering "
    "articles so that every sentence is accurate, honest and useful to a developer. "
    "You never invent evidence. Output exactly in the requested delimiter format and "
    "nothing else."
)

LLM_PROMPT = """Rewrite the article below so it passes an AdSense content-quality review.

PROBLEMS FOUND BY THE AUDIT (fix every one):
{problems}

KEEP: the topic, the genuinely useful technical substance, the teaching order, and all
CORRECT code (fix bugs; keep language tags). The finished article must be a strong,
standalone piece of AT LEAST {min_words} words (code blocks do not count).

REMOVE OR REPLACE:
1. First-hand claims ("I", "we", "our team", "a client", "we measured/shipped/migrated").
   Write impersonally ("teams commonly...", "the documented behaviour is...").
2. Invented evidence: benchmark tables, survey/customer counts, "X% of teams", dollar
   savings, named studies/reports, invented timelines. Replace a fake benchmark with HOW TO
   MEASURE it (what to instrument, which command, what to compare).
3. Any product, library or version you are not 100% sure exists.
4. Numbers: keep only documented defaults/limits, arithmetic shown step by step from stated
   assumptions, or figures explicitly labelled "illustrative".
5. Filler, hype, repetition, generic intros. Do NOT pad: where thin, add depth with a worked
   example (reasoning shown), a failure-mode analysis, or a decision checklist.

STRUCTURE: ## headings only (no # H1, no title heading). Add a table only for a real
comparison. End with ONE specific action the reader can take in the next 30 minutes.
Do NOT add links, "Related reading", "Sources", or author/footer text.
{code_rule}
{feedback}OUTPUT FORMAT (exactly):
===TITLE===
<title, max 60 chars, no first-person, no invented result>
===META===
<meta description, 110-155 chars, specific, must not start with "Learn about", "Blog post about" or "Practical guide to">
===BODY===
<full markdown article>

CURRENT TITLE: {title}

CURRENT ARTICLE:
{body}
"""

REASON_TEXT = {
    "below_target_length": "Too short. Expand with real depth (worked examples, failure modes, checklists).",
    "generic_or_missing_meta_description": "Meta description is missing/generic. Write a specific 110-155 char one.",
    "no_code_in_technical_post": "Technical post has no code. Add at least one correct, runnable, language-tagged example.",
    "missing_eeat_footer": "(Footer is added automatically afterwards; do not write one.)",
}


def split_content(content: str) -> str:
    head = content.split(FOOTER_SPLIT)[0]
    head = MARKER_RX.sub("", head)
    head = TAIL_RX.sub("", head)
    return head.strip()


def parse_llm(raw: str):
    m = re.search(
        r"===TITLE===\s*(.*?)\s*===META===\s*(.*?)\s*===BODY===\s*(.*)\Z", raw, re.S
    )
    if not m:
        return None
    body = m.group(3).strip()
    # Only unwrap when the WHOLE answer is wrapped in a ```markdown fence; a bare
    # trailing ``` is usually the closing fence of the article's last code block.
    if re.match(r"^```(?:markdown|md)\s*\n", body):
        body = re.sub(r"^```(?:markdown|md)\s*\n", "", body)
        body = re.sub(r"\n```\s*$", "", body).strip()
    return m.group(1).strip().strip('"'), m.group(2).strip().strip('"'), body


def validate_rewrite(title, meta, body, tags, orig_codes, orig_body) -> List[str]:
    errs = []
    prose = len(CODE_RX.sub(" ", body).split())
    if prose < MIN_WORDS_FLAG:
        errs.append(f"too short: {prose} prose words (need >= {MIN_WORDS_FLAG})")
    if body.count("```") % 2:
        errs.append("unbalanced code fences")
    n_codes = len(CODE_RX.findall(body))
    if orig_codes and n_codes < max(1, int(orig_codes * 0.6)):
        errs.append("dropped too many code examples")
    if TECH_TAGS & set(tags) and n_codes == 0:
        errs.append("technical post still has no code block")
    if re.search(r"^# [^#]", CODE_RX.sub("", body), re.M):
        errs.append("contains an H1 heading")
    if re.search(r"\]\(https?://", body):
        errs.append("contains links (the enhancer adds links)")
    if named_source_hit(body):
        errs.append(f"unsourced named citation: {named_source_hit(body)!r}")
    if not (8 <= len(title) <= 70) or re.search(r"\b(we|our|i|my)\b", title, re.I):
        errs.append("bad title (length or first-person)")
    if not (60 <= len(meta) <= 170) or any(p.match(meta) for p in GENERIC_META):
        errs.append("meta description length out of range or generic")
    # Optional: reuse the repo's own fabrication/entity gates when available.
    try:
        import improve_posts  # type: ignore

        orig_words = len(CODE_RX.sub(" ", orig_body).split())
        errs += [
            e
            for e in improve_posts.validate(
                orig_words, orig_codes, title, meta, body, False
            )
            if not e.startswith(("too short", "meta description", "dropped too many"))
            and e not in errs
        ]
    except Exception:
        pass
    return errs


async def llm_rewrite(bs, slug: str, d: dict, reasons: List[str], attempts: int):
    body = split_content(d.get("content", ""))
    orig_codes = len(CODE_RX.findall(body))
    tags = [t.lower() for t in d.get("tags", [])]
    problems = (
        "\n".join("- " + REASON_TEXT.get(r.split(":")[0], r) for r in reasons)
        or "- general quality pass"
    )
    code_rule = (
        "CODE: this is a technical post; include at least one correct fenced code example.\n"
        if TECH_TAGS & set(tags)
        else ""
    )
    feedback, errs = "", []
    for n in range(1, attempts + 1):
        prompt = LLM_PROMPT.format(
            problems=problems,
            min_words=MIN_WORDS_FLAG + 100,
            code_rule=code_rule,
            feedback=feedback,
            title=d.get("title", slug),
            body=body,
        )
        raw = await bs._call_api_with_fallback(
            [
                {"role": "system", "content": LLM_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=7000,
        )
        parsed = parse_llm(raw)
        if not parsed:
            errs = ["output did not follow ===TITLE===/===META===/===BODY=== format"]
        else:
            title, meta, new_body = parsed
            errs = validate_rewrite(title, meta, new_body, tags, orig_codes, body)
            if not errs:
                return title, meta, new_body, n, []
        feedback = (
            "PREVIOUS ATTEMPT WAS REJECTED. Fix these problems:\n"
            + "\n".join(f"- {e}" for e in errs)
            + "\n\n"
        )
    return None, None, None, attempts, errs


def run_llm_improve(report: dict, args) -> List[str]:
    """Rewrite IMPROVE posts that have a post.json. Returns improved slugs."""
    import yaml

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from blog_system import BlogSystem  # type: ignore

    cands, skipped_done, footer_only = [], [], []
    for e in report["improve"]:
        if args.slug and e["slug"] not in args.slug:
            continue
        pj = DOCS_DIR / e["slug"] / "post.json"
        if e["source"] != "post.json" or not pj.exists():
            print(f"  skip {e['slug']}: no post.json (html-only orphan)")
            continue
        try:
            d = json.loads(pj.read_text(encoding="utf-8"))
        except Exception:
            continue
        if d.get("llm_improved_at") and not args.reimprove:
            skipped_done.append(e)
            continue
        # footer-only failures are fixed by post_enhancer.py, not by a rewrite
        if all(r.startswith("missing_eeat_footer") for r in e["reasons"]):
            footer_only.append(e)
            continue
        cands.append((e, d, pj))
    # thinnest first
    cands.sort(key=lambda t: t[0]["word_count"])
    total = len(cands)
    if args.limit:
        cands = cands[: args.limit]
    print(f"LLM-improving {len(cands)} of {total} pending IMPROVE post(s)")
    if skipped_done:
        from collections import Counter

        why = Counter(r.split(":")[0] for e in skipped_done for r in e["reasons"])
        print(
            f"  {len(skipped_done)} IMPROVE post(s) skipped: already LLM-rewritten "
            f"(llm_improved_at set, e.g. by improve_posts.py in CI). Remaining reasons: "
            f"{dict(why)}. Use --reimprove to rewrite them again."
        )
    if footer_only:
        print(
            f"  {len(footer_only)} post(s) only lack the footer: run post_enhancer.py "
            "(no LLM needed)."
        )

    cfg = yaml.safe_load(open(args.config, encoding="utf-8")) or {}
    bs = BlogSystem(cfg)
    improved, failed = [], []
    for e, d, pj in cands:
        slug = e["slug"]
        print(f"\n→ {slug}  ({'; '.join(e['reasons'])})")
        try:
            title, meta, body, tries, errs = asyncio.run(
                llm_rewrite(bs, slug, d, e["reasons"], args.attempts)
            )
        except Exception as exc:
            failed.append({"slug": slug, "errors": [f"LLM call failed: {exc}"]})
            print(f"  ✗ LLM call failed: {exc}")
            continue
        if errs:
            failed.append({"slug": slug, "errors": errs})
            print("  ✗ rejected after retries (original kept): " + "; ".join(errs))
            continue
        bk = BACKUP_DIR / slug
        bk.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pj, bk / "post.json.pre_llm")
        now = datetime.now(timezone.utc).isoformat()
        d["content"], d["meta_description"] = body, meta
        d["updated_at"] = d["llm_improved_at"] = now
        d.setdefault("slug", slug)
        pj.write_text(json.dumps(d, indent=2, ensure_ascii=False), "utf-8")
        (pj.parent / "index.md").write_text(f"# {d['title']}\n\n{body}\n", "utf-8")
        improved.append(slug)
        print(f"  ✓ improved in {tries} attempt(s)")

    Path("improve_report.json").write_text(
        json.dumps({"improved": improved, "failed": failed}, indent=2)
    )
    print(
        f"\nDone: {len(improved)} improved, {len(failed)} unchanged (improve_report.json)"
    )

    if improved and not args.no_finalize:
        enh = Path(__file__).resolve().parent / "post_enhancer.py"
        if enh.exists():
            subprocess.run(
                [
                    sys.executable,
                    str(enh),
                    "--docs",
                    str(DOCS_DIR),
                    "--backup",
                    str(BACKUP_DIR),
                    "--verify-links",
                    "--confirm",
                ],
                check=False,
            )
        else:
            print(
                "post_enhancer.py not found: footer/links NOT added; run it before publishing."
            )
    return improved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--delete", action="store_true", help="git rm DELETE entries (default dry-run)"
    )
    ap.add_argument(
        "--improve", action="store_true", help="write regeneration_queue.json"
    )
    ap.add_argument(
        "--improve-llm",
        action="store_true",
        help="LLM-rewrite IMPROVE posts (writes post.json)",
    )
    ap.add_argument(
        "--limit", type=int, default=10, help="max LLM rewrites per run; 0 = all"
    )
    ap.add_argument("--slug", action="append", help="restrict --improve-llm to slug(s)")
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument(
        "--reimprove", action="store_true", help="redo posts already llm_improved_at"
    )
    ap.add_argument("--no-finalize", action="store_true", help="skip post_enhancer.py")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--min-words", type=int, default=MIN_WORDS_DELETE)
    ap.add_argument(
        "--force", action="store_true", help="allow deleting >25%% of corpus"
    )
    ap.add_argument(
        "--purge-tombstones",
        action="store_true",
        help="forget RETIRED slugs for good: needs --ledger; removes them from the ledger, then git rm",
    )
    ap.add_argument(
        "--ledger",
        action="append",
        default=[],
        help="JSON file(s) the build reads retired slugs from (e.g. the one used by "
        "utilsIndexing/retire_registry.py). Repeatable.",
    )
    a = ap.parse_args()

    report, posts = build_report(a.min_words)
    Path("adsense_audit_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    n, nd, nr = len(posts), len(report["delete"]), len(report["retired"])
    print(
        f"posts={n} delete={nd} improve={len(report['improve'])} "
        f"keep={len(report['keep'])} retired(tombstones)={nr}"
    )
    for e in report["delete"]:
        print(f"  DELETE docs/{e['slug']}/ [{e['source']}] {e['reasons']}")
    if nr:
        print(
            f"  {nr} tombstone(s) left alone: the build recreates them from the retire "
            "ledger. Use --purge-tombstones --ledger <file> to forget them."
        )

    write_script("remove_flagged_posts.sh", report["delete"])
    print("wrote remove_flagged_posts.sh")

    if a.delete and report["delete"]:
        if n and nd / max(1, n - nr) > MAX_DELETE_FRACTION and not a.force:
            raise SystemExit(
                f"Refusing: {nd}/{n - nr} > {MAX_DELETE_FRACTION:.0%}. Inspect report, use --force."
            )
        for e in report["delete"]:
            subprocess.run(["git", "rm", "-r", f"docs/{e['slug']}"], check=True)
        block_deleted_topics(report["delete"], posts)
        print("git rm done; review `git status` and commit.")

    if a.purge_tombstones:
        if not a.ledger:
            raise SystemExit(
                "--purge-tombstones needs --ledger <file>: without removing the slugs "
                "from the ledger the next build recreates the tombstones."
            )
        slugs = {e["slug"] for e in report["retired"]}
        for lg in a.ledger:
            purge_from_ledger(Path(lg), slugs)
        for s in sorted(slugs):
            subprocess.run(["git", "rm", "-r", "-f", f"docs/{s}"], check=True)
        print(f"purged {len(slugs)} tombstone(s); commit ledger + docs together.")

    q = Path("regeneration_queue.json")
    queue = json.loads(q.read_text()) if q.exists() else []
    seen = {e["slug"] for e in queue}
    new = [
        {
            "slug": e["slug"],
            "reason": "; ".join(e["reasons"]),
            "priority": "high" if "thin" in "; ".join(e["reasons"]) else "normal",
            "source": "adsense_audit",
        }
        for e in report["improve"]
        if e["slug"] not in seen
    ]
    print(
        f"{'queued' if a.improve else '[dry-run] would queue'} {len(new)} IMPROVE posts"
    )
    if a.improve:
        q.write_text(json.dumps(queue + new, indent=2))

    if a.improve_llm:
        done = set(run_llm_improve(report, a))
        if done and q.exists():
            cur = json.loads(q.read_text())
            q.write_text(
                json.dumps(
                    [
                        x
                        for x in cur
                        if not (
                            x.get("source") == "adsense_audit" and x.get("slug") in done
                        )
                    ],
                    indent=2,
                )
            )
        print(
            "Run the site build next so docs/*/index.html is regenerated from post.json."
        )
    elif report["improve"]:
        print(
            f"hint: --improve-llm rewrites the {len(report['improve'])} IMPROVE posts with the LLM"
        )


if __name__ == "__main__":
    main()
