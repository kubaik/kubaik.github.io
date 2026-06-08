"""
adsense_fixes/content_freshness.py
=====================================
Automated content freshness signals and stale-content detection.

WHY THIS EXISTS
---------------
AdSense Site Readiness Guide (§4 Content Quality):
  "Content updated regularly. Continually publish new, fresh content.
   Sites with no updates for weeks are penalised. Set an automated
   publishing schedule."

Google's quality rater guidelines also assess whether a page's content
reflects the current state of a topic. Technology content especially
becomes stale fast: tool versions, API endpoints, and best practices
change constantly.

This module provides:
  1. `mark_stale_posts(docs_dir, days_threshold)` — identifies posts
     that haven't been updated and are likely to contain outdated info.
  2. `inject_freshness_footer(post)` — adds/updates a "last verified"
     signal in the article footer, which Google's quality raters look
     for as a trust signal.
  3. `get_publishing_schedule_status(docs_dir)` — reports whether
     the publishing cadence looks natural or has gaps.

HOW TO INTEGRATE
----------------
Run weekly from GitHub Actions:
    from adsense_fixes.content_freshness import (
        mark_stale_posts,
        get_publishing_schedule_status,
    )
    stale = mark_stale_posts(Path('./docs'), days_threshold=180)
    print(get_publishing_schedule_status(Path('./docs')))

The inject_freshness_footer() is called automatically in blog_system.py
inside inject_eeat_signals() — it updates the "Last reviewed" date
in the E-E-A-T footer every time a post is regenerated.
"""

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple


# Posts older than this (days) without an update are considered stale.
# 180 days = ~6 months, which matches Google's freshness decay window
# for evergreen technical content.
STALE_THRESHOLD_DAYS = 180

# Posts in these technology categories decay faster
_FAST_DECAY_TAGS = {
    'llm', 'ai', 'generativeai', 'chatgpt', 'openai', 'agentic',
    'cloudnative', 'kubernetes', 'devops', 'devsecops',
}


# ── Stale content detection ────────────────────────────────────────────────

def mark_stale_posts(
    docs_dir: Path,
    days_threshold: int = STALE_THRESHOLD_DAYS,
) -> List[Dict]:
    """
    Walk docs_dir and return posts that are likely stale (last updated
    more than days_threshold days ago).

    For fast-decay topic posts (AI, cloud, etc.), the threshold is halved.

    Returns a list of dicts:
      { slug, title, last_updated, days_old, fast_decay, priority }
    """
    stale = []

    if not docs_dir.exists():
        return stale

    for post_dir in sorted(docs_dir.iterdir()):
        if not post_dir.is_dir() or post_dir.name in ('static', 'tag', 'author'):
            continue
        post_json = post_dir / 'post.json'
        if not post_json.exists():
            continue

        try:
            with open(post_json, 'r', encoding='utf-8') as f:
                data = json.load(f)

            slug = post_dir.name
            title = data.get('title', slug)
            updated_raw = data.get('updated_at', data.get('created_at', ''))
            tags = [t.lower() for t in data.get('tags', [])]

            updated_date = _parse_date(updated_raw)
            if not updated_date:
                continue

            days_old = (date.today() - updated_date).days
            is_fast_decay = bool(
                set(tags) & _FAST_DECAY_TAGS
            )
            threshold = days_threshold // 2 if is_fast_decay else days_threshold

            if days_old >= threshold:
                stale.append({
                    'slug': slug,
                    'title': title,
                    'last_updated': updated_date.isoformat(),
                    'days_old': days_old,
                    'fast_decay': is_fast_decay,
                    'priority': 'high' if days_old > threshold * 1.5 else 'medium',
                })

        except (json.JSONDecodeError, KeyError, ValueError):
            continue

    stale.sort(key=lambda x: x['days_old'], reverse=True)
    return stale


# ── Publishing cadence analysis ────────────────────────────────────────────

def get_publishing_schedule_status(docs_dir: Path) -> str:
    """
    Analyse the publishing cadence over the last 90 days.

    Returns a human-readable report.  A healthy cadence for AdSense
    purposes is:
      - At least one post per week
      - No gap longer than 14 days
      - No more than 4 posts per day (spam signal threshold)
    """
    if not docs_dir.exists():
        return "docs/ not found."

    dates: List[date] = []
    for post_dir in docs_dir.iterdir():
        if not post_dir.is_dir() or post_dir.name in ('static', 'tag', 'author'):
            continue
        post_json = post_dir / 'post.json'
        if not post_json.exists():
            continue
        try:
            with open(post_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            d = _parse_date(data.get('created_at', ''))
            if d:
                dates.append(d)
        except (json.JSONDecodeError, KeyError):
            continue

    if not dates:
        return "No published posts found."

    dates.sort()
    cutoff = date.today() - timedelta(days=90)
    recent = [d for d in dates if d >= cutoff]

    lines = [
        "📅 Publishing Cadence Report (last 90 days)",
        "=" * 50,
        f"  Total posts published : {len(dates)}",
        f"  Posts in last 90 days : {len(recent)}",
        f"  First post            : {dates[0]}",
        f"  Most recent post      : {dates[-1]}",
        f"  Days since last post  : {(date.today() - dates[-1]).days}",
    ]

    # Detect gaps
    if len(recent) >= 2:
        gaps = []
        for i in range(1, len(recent)):
            gap = (recent[i] - recent[i - 1]).days
            if gap > 14:
                gaps.append((recent[i - 1], recent[i], gap))
        if gaps:
            lines.append(f"\n  ⚠️  Publishing gaps > 14 days detected:")
            for start, end, gap_days in gaps:
                lines.append(f"     {start} → {end} ({gap_days} days)")
        else:
            lines.append("\n  ✅ No publishing gaps > 14 days.")

    # Detect velocity spikes
    daily_counts: Dict[date, int] = {}
    for d in recent:
        daily_counts[d] = daily_counts.get(d, 0) + 1

    spikes = [(d, cnt) for d, cnt in daily_counts.items() if cnt > 4]
    if spikes:
        lines.append(f"\n  ⚠️  High-velocity days (> 4 posts) detected:")
        for d, cnt in sorted(spikes):
            lines.append(f"     {d}: {cnt} posts")
    else:
        lines.append("  ✅ No velocity spikes detected.")

    # Days since last post warning
    days_since = (date.today() - dates[-1]).days
    if days_since > 14:
        lines.append(
            f"\n  ⚠️  Last post was {days_since} days ago. "
            f"AdSense reviewers expect regular activity. "
            f"Consider publishing a new post before re-submitting for review."
        )
    else:
        lines.append(
            f"\n  ✅ Recent activity looks healthy ({days_since} days since last post)."
        )

    return "\n".join(lines)


# ── Freshness footer update ────────────────────────────────────────────────

def inject_freshness_footer(post) -> None:
    """
    Update the 'Last reviewed' date in the E-E-A-T footer in post.content.

    This is a lightweight call that updates only the date line.
    It is safe to call on every auto-mode run — it is idempotent.

    If no E-E-A-T footer is present, this is a no-op (inject_eeat_signals
    in blog_system.py handles the initial injection).
    """
    if not getattr(post, 'content', ''):
        return

    today_str = datetime.now().strftime('%B %d, %Y')
    # Pattern matches: **Last reviewed:** Month DD, YYYY
    reviewed_pattern = r'(\*\*Last reviewed:\*\*\s*)([^\n]+)'

    import re
    new_content = re.sub(
        reviewed_pattern,
        lambda m: f"{m.group(1)}{today_str}",
        post.content,
    )
    post.content = new_content


# ── Helpers ────────────────────────────────────────────────────────────────

def _parse_date(raw: str) -> date | None:
    """Parse an ISO datetime string to a date object."""
    if not raw:
        return None
    # Strip time component
    date_part = raw.split('T')[0] if 'T' in raw else raw.strip()
    try:
        return date.fromisoformat(date_part)
    except (ValueError, AttributeError):
        return None


def stale_report(docs_dir: Path) -> str:
    """Human-readable stale posts report for the CLI."""
    stale = mark_stale_posts(docs_dir)
    if not stale:
        return (
            f"✅ Content freshness: PASS\n"
            f"   No posts exceed the {STALE_THRESHOLD_DAYS}-day staleness threshold."
        )

    lines = [
        f"⚠️  Content freshness: {len(stale)} stale post(s) detected.",
        f"   These posts should be reviewed and updated before re-submitting to AdSense.",
        "",
        f"  {'Priority':<8} {'Days Old':<10} {'Slug':<40} Title",
        "  " + "-" * 90,
    ]
    for item in stale[:20]:
        decay_marker = " 🔥" if item['fast_decay'] else ""
        lines.append(
            f"  {item['priority']:<8} {item['days_old']:<10} "
            f"{item['slug'][:38]:<40} {item['title'][:40]}{decay_marker}"
        )

    lines += [
        "",
        "  🔥 = Fast-decay topic (AI/cloud/DevOps — stale faster than others)",
        "",
        "  To re-queue a stale post for update, delete its post.json and re-run",
        "  'python blog_system.py auto' with the same topic — the similarity guard",
        "  will detect the existing content and update rather than duplicate.",
    ]
    return "\n".join(lines)
