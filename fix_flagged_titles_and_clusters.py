"""
fix_flagged_titles_and_clusters.py — acts on title_audit_queue.json.

  merge_candidate -> RETIRE: back up the post dir, replace it with a
                     noindex + canonical + meta-refresh stub pointing at the
                     keeper, rewrite internal links in other posts to the
                     keeper, and record the title in .retired_topics.json.
  retitle         -> re-checked with pre_save_gate.title_problem (so audit
                     false positives are skipped). Fix order:
                       1. --overrides file   (explicit {slug: title})
                       2. deterministic wording rewrite (free)
                       3. LLM, same mechanism as improve_posts.py:
                          BlogSystem._call_api_with_fallback (the pipeline's
                          provider chain: DeepSeek -> Groq -> Gemini -> ...),
                          ===TITLE=== output format, up to --attempts tries
                          with the gate's rejection reasons fed back.
                     Failures leave the original title untouched and are
                     logged to fix_report.json. Slug (URL) is always kept.

Dry-run by default (no LLM calls). --confirm to write. Bounded per run,
everything backed up. Run from repo root, then `python blog_system.py build`.

  python fix_flagged_titles_and_clusters.py
  python fix_flagged_titles_and_clusters.py --confirm --llm
  python fix_flagged_titles_and_clusters.py --confirm --llm --slug my-slug
"""

import argparse
import asyncio
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pre_save_gate import title_problem, topic_overlap_problem

BASE_URL = "https://kubaik.github.io"

STUB = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Moved: {title}</title>
<meta name="robots" content="noindex, follow">
<link rel="canonical" href="{base}/{keeper}/">
<meta http-equiv="refresh" content="0; url={base_path}/{keeper}/">
</head><body>
<p>This article has been merged into a more complete version:
<a href="{base_path}/{keeper}/">{keeper_title}</a>.</p>
</body></html>
"""

_WORD_MAP = [
    (r"\bwe missed\b", "teams miss"),
    (r"\bblew up\b", "failed"),
    (r"\btested\b", "compared"),
    (r"\btried\b", "compared"),
    (r"\bsurvived\b", "still hold up"),
    (r"\bwhat i\b", "what"),
    (r"\bour\b", "your"),
    (r"\bwe\b", "teams"),
    (r"\bmy\b", "your"),
    (r"\bmonths in\b", "long-term"),
    (r"\bweeks in\b", "early"),
]

# ── LLM prompt (same style/contract as improve_posts.py) ────────────────────
SYSTEM = (
    "You are a meticulous senior technical editor. You write accurate, honest "
    "article titles. You never imply first-hand experience or invented results. "
    "Output exactly in the requested delimiter format and nothing else."
)

PROMPT = """Write ONE replacement title for the article below.

The current title was rejected: {problem}

RULES:
- Max 60 characters; a TOPIC or a CLAIM the article actually supports.
- No first-person ("I", "we", "my", "our") and no implied personal experience
  ("tested", "tried", "survived", "after N months", "what broke for us").
- No dollar amounts, percentages, multipliers, user/event counts or "Nx" figures
  unless that exact figure is documented, shown as arithmetic, or labelled
  illustrative in the article body. When unsure, describe qualitatively
  ("on a small budget", "at high volume").
- Do not start with a past-tense action verb (Cut, Saved, Burned, Built...).
- No filler ("Complete", "Ultimate", "Guide to"). Keep the same subject so the
  URL slug still fits.
- Must differ from these existing titles:
{existing}

{feedback}OUTPUT FORMAT (exactly):
===TITLE===
<the title>

CURRENT TITLE: {title}

ARTICLE EXCERPT (first 400 words):
{excerpt}
"""


def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def save(p: Path, d: dict):
    p.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")


def backup(src: Path, backup_dir: Path):
    dst = backup_dir / src.name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


# ── retire ──────────────────────────────────────────────────────────────────
def repoint_links(
    docs: Path, old: str, new: str, backup_dir: Path, confirm: bool
) -> int:
    pat = re.compile(r"\]\(((?:https?://[^)/\s]+)?)/" + re.escape(old) + r"/?([#)])")
    n = 0
    for d in docs.iterdir():
        pj = d / "post.json"
        if not pj.exists() or d.name == old:
            continue
        data = load(pj)
        new_content, k = pat.subn(
            lambda m: f"]({m.group(1)}/{new}/{m.group(2) if m.group(2) == '#' else ')'}",
            data.get("content", ""),
        )
        if k:
            n += k
            if confirm:
                backup(d, backup_dir / "linkfix")
                data["content"] = new_content
                save(pj, data)
                md = d / "index.md"
                if md.exists():
                    md.write_text(
                        f"# {data.get('title', '')}\n\n{new_content}", encoding="utf-8"
                    )
    return n


def retire(
    docs: Path, slug: str, keeper: str, backup_dir: Path, confirm: bool, base_path: str
):
    src, kp = docs / slug, docs / keeper
    if not (src / "post.json").exists():
        return f"skip {slug}: no post.json (already retired?)"
    if not (kp / "post.json").exists():
        return f"skip {slug}: keeper {keeper} missing"
    title = load(src / "post.json").get("title", slug)
    ktitle = load(kp / "post.json").get("title", keeper)
    if not confirm:
        return f"would retire {slug} -> {keeper}"
    backup(src, backup_dir)
    shutil.rmtree(src)
    src.mkdir()
    (src / "index.html").write_text(
        STUB.format(
            title=title,
            keeper=keeper,
            keeper_title=ktitle,
            base=BASE_URL,
            base_path=base_path,
        ),
        encoding="utf-8",
    )
    rt = Path(".retired_topics.json")
    items = json.loads(rt.read_text()) if rt.exists() else []
    items.append(
        {
            "slug": slug,
            "title": title,
            "keeper": keeper,
            "date": datetime.now().date().isoformat(),
        }
    )
    rt.write_text(json.dumps(items, indent=2))
    fixed = repoint_links(docs, slug, keeper, backup_dir, True)
    return f"retired {slug} -> {keeper} (links repointed: {fixed})"


# ── retitle ─────────────────────────────────────────────────────────────────
def deterministic_title(title: str, content: str) -> Optional[str]:
    t = title
    for pat, rep in _WORD_MAP:
        t = re.sub(pat, rep, t, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip(" :-–—")
    if len(t) > 60 or len(t.split()) < 3:
        return None
    return t if not title_problem(t, content) and t != title else None


def parse_title(raw: str) -> Optional[str]:
    m = re.search(r"===TITLE===\s*(.+?)\s*(?:===|\Z)", raw or "", re.S)
    if not m:
        return None
    t = (
        m.group(1)
        .strip()
        .splitlines()[0]
        .strip()
        .strip('"')
        .strip("'")
        .strip("#")
        .strip()
    )
    return t or None


def title_errors(
    cand: str, old: str, content: str, docs: Path, slug: str, existing_lower: set
) -> List[str]:
    errs = []
    if len(cand) > 60:
        errs.append(f"{len(cand)} chars; max is 60")
    if len(cand.split()) < 3:
        errs.append("too short (min 3 words)")
    if cand.strip().lower() == old.strip().lower() or cand.lower() in existing_lower:
        errs.append("identical to an existing title")
    p = title_problem(cand, content)
    if p:
        errs.append(p)
    p = topic_overlap_problem(cand, docs, exclude_slug=slug)
    if p:
        errs.append(p)
    return errs


async def llm_title(
    bs, d: dict, problem: str, docs: Path, slug: str, existing: List[str], attempts: int
):
    """Same call path as improve_posts.improve_one: bs._call_api_with_fallback,
    delimiter output, retry with the validator's reasons fed back."""
    content, old = d.get("content", ""), d.get("title", "")
    excerpt = " ".join(content.split()[:400])
    existing_lower = {t.lower() for t in existing}
    feedback, last = "", []
    for n in range(1, attempts + 1):
        prompt = PROMPT.format(
            problem=problem,
            existing="\n".join(f'- "{t}"' for t in existing[:15]) or "- (none)",
            feedback=feedback,
            title=old,
            excerpt=excerpt,
        )
        raw = await bs._call_api_with_fallback(
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=200,
        )
        cand = parse_title(raw)
        if not cand:
            last = ["output did not follow the ===TITLE=== format"]
        else:
            last = title_errors(cand, old, content, docs, slug, existing_lower)
            if not last:
                return cand, n, []
        feedback = (
            "PREVIOUS ATTEMPT WAS REJECTED. Fix these problems:\n"
            + "\n".join(f"- {e}" for e in last)
            + "\n\n"
        )
    return None, attempts, last


def apply_title(d: Path, new_title: str, backup_dir: Path, llm: bool):
    backup(d, backup_dir / "retitle")
    data = load(d / "post.json")
    data["title"] = new_title
    if "full_title" in data:
        data["full_title"] = new_title
    now = datetime.now(timezone.utc).isoformat()
    data["updated_at"] = now
    if llm:
        data["llm_retitled_at"] = now
    save(d / "post.json", data)
    if (d / "index.md").exists():
        (d / "index.md").write_text(
            f"# {new_title}\n\n{data['content']}\n", encoding="utf-8"
        )


def make_bs(config_path: str):
    import yaml
    from blog_system import BlogSystem

    return BlogSystem(yaml.safe_load(open(config_path, encoding="utf-8")) or {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="./docs")
    ap.add_argument("--queue", default="title_audit_queue.json")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--backup-dir", default=".title_cluster_backups")
    ap.add_argument("--max-retire", type=int, default=10)
    ap.add_argument("--max-retitle", type=int, default=10)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--slug", action="append", help="limit retitle step to these slugs")
    ap.add_argument("--base-path", default="")
    ap.add_argument(
        "--llm", action="store_true", help="use the pipeline's LLM provider chain"
    )
    ap.add_argument("--overrides", help="JSON {slug: new_title}; validated by the gate")
    ap.add_argument("--confirm", action="store_true")
    a = ap.parse_args()

    docs, bdir = Path(a.docs), Path(a.backup_dir)
    queue = json.loads(Path(a.queue).read_text())
    audit = (
        json.loads(Path("title_audit.json").read_text())
        if Path("title_audit.json").exists()
        else {"clusters": []}
    )
    keeper_of = {s: c["keep"] for c in audit["clusters"] for s in c["merge_or_retire"]}
    retire_slugs = set(keeper_of)
    if a.confirm:
        bdir.mkdir(exist_ok=True)

    print(f"mode: {'WRITE' if a.confirm else 'DRY RUN'}{' + LLM' if a.llm else ''}")
    report = {"retired": [], "retitled": [], "failed": [], "needs_attention": []}

    done = 0
    if not a.slug:
        for e in queue:
            if e["action"] != "merge_candidate" or done >= a.max_retire:
                continue
            keeper = keeper_of.get(e["slug"])
            if not keeper or keeper in retire_slugs:
                print(f"  skip {e['slug']}: keeper unknown or itself being retired")
                continue
            msg = retire(docs, e["slug"], keeper, bdir, a.confirm, a.base_path)
            print("  " + msg)
            if msg.startswith("retired"):
                report["retired"].append(e["slug"])
            done += 1

    overrides = json.loads(Path(a.overrides).read_text()) if a.overrides else {}
    existing = []
    for p in docs.glob("*/post.json"):
        try:
            existing.append(load(p).get("title", ""))
        except Exception:
            pass
    existing = [t for t in existing if t]

    bs = None
    if a.llm and a.confirm:
        try:
            bs = make_bs(a.config)
        except Exception as exc:
            print(f"  ✗ could not initialise BlogSystem for LLM use: {exc}")

    n = 0
    for e in queue:
        if e["action"] != "retitle" or n >= a.max_retitle:
            continue
        if a.slug and e["slug"] not in a.slug:
            continue
        slug = e["slug"]
        d = docs / slug
        if not (d / "post.json").exists():
            continue
        data = load(d / "post.json")
        title, content = data.get("title", ""), data.get("content", "")
        problem = title_problem(title, content)
        if not problem:
            print(f"  ok   {slug}: passes tuned gate (audit false positive)")
            continue

        new, via = overrides.get(slug), "override"
        if new and title_problem(new, content):
            print(
                f"  override for {slug} rejected by gate: {title_problem(new, content)}"
            )
            new = None
        if not new:
            new, via = deterministic_title(title, content), "rewrite"
        if not new and a.llm and not a.confirm:
            print(f"  LLM  {slug}: would ask the LLM to retitle ({problem})")
            continue
        if not new and bs:
            try:
                new, tries, errs = asyncio.run(
                    llm_title(bs, data, problem, docs, slug, existing, a.attempts)
                )
            except Exception as exc:  # provider chain exhausted etc.
                report["failed"].append(
                    {"slug": slug, "errors": [f"LLM call failed: {exc}"]}
                )
                print(f"  ✗ {slug}: LLM call failed: {exc}")
                continue
            via = f"llm x{tries}"
            if not new:
                report["failed"].append({"slug": slug, "errors": errs})
                print(
                    f"  ✗ {slug}: rejected after retries (kept original): {'; '.join(errs)}"
                )
                continue
        if not new:
            report["needs_attention"].append(
                {"slug": slug, "title": title, "problem": problem}
            )
            print(f"  TODO {slug}: {problem}  (use --llm --confirm or --overrides)")
            continue

        print(
            f"  {'set ' if a.confirm else 'would set'} {slug} [{via}]: {title!r} -> {new!r}"
        )
        if a.confirm:
            apply_title(d, new, bdir, llm=via.startswith("llm"))
            existing.append(new)
            report["retitled"].append(
                {"slug": slug, "old": title, "new": new, "via": via}
            )
        n += 1

    if a.confirm:
        Path("fix_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False)
        )
        print(
            f"\nDone: {len(report['retired'])} retired, {len(report['retitled'])} retitled, "
            f"{len(report['failed'])} failed, {len(report['needs_attention'])} need attention "
            f"(fix_report.json)"
        )
    print("Next: python blog_system.py build && python title_claim_audit.py")


if __name__ == "__main__":
    main()
