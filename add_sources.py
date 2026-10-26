#!/usr/bin/env python3
"""
add_sources.py - code-driven primary-source linking for posts that cite nothing external.

Fixes the IMPROVE list from scripts/final_triage.py (reason: no_primary_sources) WITHOUT an LLM,
so no URL can be hallucinated: links come only from post_enhancer.DOCS (vendor docs) plus the
STANDARDS map below (OWASP / NIST / W3C / Google SRE), selected by what the post actually discusses.
Each URL is HEAD/GET-verified (--verify-links) and dropped if dead.

Posts with no match are left untouched and listed in sources_unmatched.txt (nothing safe to cite).

  python add_sources.py --docs ./docs                      # dry run
  python add_sources.py --docs ./docs --verify-links --confirm --max 100
Backs up post.json to --backup-dir before writing. Idempotent.
"""

import argparse, json, re, shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import post_enhancer as pe
import triage

# topic regex (matched against post body, code stripped) -> (label, URL). Standards bodies only.
STANDARDS = {
    r"(?i)prompt injection|llm (security|guardrail)|agent (auth|permission)|least[- ]privilege.{0,30}agent|model poisoning|data poisoning": (
        "OWASP Top 10 for LLM Applications",
        "https://owasp.org/www-project-top-10-for-large-language-model-applications/",
    ),
    r"(?i)\bWCAG\b|accessib(le|ility)": (
        "W3C: Web Content Accessibility Guidelines (WCAG) 2.2",
        "https://www.w3.org/TR/WCAG22/",
    ),
    r"(?i)post-?mortem|blameless": (
        "Google SRE Book: Postmortem Culture",
        "https://sre.google/sre-book/postmortem-culture/",
    ),
    r"(?i)incident response|on-?call|alert (fatigue|triage)": (
        "NIST SP 800-61: Computer Security Incident Handling Guide",
        "https://csrc.nist.gov/pubs/sp/800/61/r2/final",
    ),
    r"(?i)audit (log|trail)|log management": (
        "NIST SP 800-92: Guide to Computer Security Log Management",
        "https://csrc.nist.gov/pubs/sp/800/92/final",
    ),
    r"(?i)slo|error budget|monitoring|observability|alerting": (
        "Google SRE Book: Monitoring Distributed Systems",
        "https://sre.google/sre-book/monitoring-distributed-systems/",
    ),
    r"(?i)ai risk|responsible ai|ai governance": (
        "NIST AI Risk Management Framework",
        "https://www.nist.gov/itl/ai-risk-management-framework",
    ),
    r"(?i)opentelemetry|distributed tracing|\btraces?\b": (
        "OpenTelemetry documentation",
        "https://opentelemetry.io/docs/",
    ),
}
STANDARDS = {re.compile(k): v for k, v in STANDARDS.items()}
MIN_STD_HITS = 3
HEAD = re.compile(r"^## Official documentation\s*$", re.M | re.I)
FOOTER_SPLIT = re.compile(r"\n---\n\n### About this article")
SELF = "kubaik.github.io"


def has_external(c: str) -> bool:
    return any(SELF not in u for u in re.findall(r"\]\((https?://[^)\s]+)\)", c))


def pick(body: str, verify: bool, cache: dict):
    links, seen = [], set()
    for table in (pe.DOCS, STANDARDS):
        for rx, (label, url) in table.items():
            need = (
                1 if table is pe.DOCS else MIN_STD_HITS
            )  # standards: topic must be a real theme
            if url in seen or len(rx.findall(body)) < need:
                continue
            if verify and not cache.setdefault(url, pe.link_ok(url)):
                continue
            seen.add(url)
            links.append(f"- [{label}]({url})")
    return links[:5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="docs")
    ap.add_argument("--backup-dir", default=".sources_backups")
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--verify-links", action="store_true")
    ap.add_argument("--confirm", action="store_true")
    a = ap.parse_args()
    docs, bk = Path(a.docs), Path(a.backup_dir)
    cache, done, unmatched, skipped = {}, 0, [], 0
    for pj in sorted(docs.glob("*/post.json")):
        slug = pj.parent.name
        try:
            d = json.loads(pj.read_text("utf-8"))
        except ValueError:
            continue
        c = d.get("content", "")
        if has_external(c) or HEAD.search(c):
            skipped += 1
            continue
        links = pick(triage.STRIP.sub(" ", c), a.verify_links, cache)
        if not links:
            unmatched.append(slug)
            continue
        if a.max and done >= a.max:
            break
        section = "## Official documentation\n\n" + "\n".join(links)
        m = FOOTER_SPLIT.search(c)
        new = (
            (c[: m.start()].rstrip() + "\n\n" + section + "\n" + c[m.start() :])
            if m
            else c.rstrip() + "\n\n" + section + "\n"
        )
        done += 1
        print(f"{'ADD' if a.confirm else 'would add'} {len(links)} source(s) -> {slug}")
        if a.confirm:
            (bk / slug).mkdir(parents=True, exist_ok=True)
            shutil.copy2(pj, bk / slug / "post.json")
            d["content"] = new
            pj.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
            md = pj.parent / "index.md"
            if md.exists() and c.strip() in md.read_text("utf-8"):
                md.write_text(
                    md.read_text("utf-8").replace(c.strip(), new.strip()),
                    encoding="utf-8",
                )
    Path("sources_unmatched.txt").write_text(
        "\n".join(unmatched) + ("\n" if unmatched else "")
    )
    print(
        f"{'updated' if a.confirm else 'would update'} {done}; already sourced {skipped}; "
        f"no safe source to add {len(unmatched)} (see sources_unmatched.txt)"
    )
    if not a.confirm:
        print("Dry run. Re-run with --confirm.")


if __name__ == "__main__":
    main()
