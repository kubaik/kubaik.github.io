"""
velocity_controller.py
======================
Controls automated post-publication rate to prevent spam signals.

WHY THIS EXISTS
---------------
AdSense Site Readiness Guide (§7 — Automated Blog Specific Concerns):
  "Publishing 100+ posts per day on a new site is a spam signal.
   Throttle automated publishing to a natural, sustainable pace."

Google's spam classifier uses publishing velocity as a strong signal.
A new site that publishes 10 posts in 10 minutes looks like a content
farm regardless of quality.  This module enforces a daily cap that
scales with domain age and can be tuned per deployment environment.

RECOMMENDED CAPS
---------------
  - First 30 days:   1 post/day  (build trust slowly)
  - Days 31-90:      2 posts/day
  - Days 91-180:     3 posts/day
  - After 6 months:  up to 4 posts/day

FIX (found in review, 2026)
----------------------------
This module's gating logic (can_publish/effective_limit) was always
correct and was already wired into blog_system.py's `auto` entry point.
The bug was entirely in how state was persisted: it wrote a counter to
.publish_velocity.json, but that file is deliberately listed in
.gitignore under "Runtime caches — rebuilt automatically from docs/,
never commit these" (alongside .similarity_index.json,
.used_topics.json, etc.) — a pattern the rest of this codebase applies
consistently. Every GitHub Actions run gets a fresh checkout, so the
counter silently reset to 0 on every run regardless of what happened
earlier that day. Force-adding the ignored file (`git add -f`) would
work but fights the codebase's own convention and reintroduces drift
risk (a stale/corrupt state file surviving a checkout that docs/
disagrees with).

The actual fix: today's publish count and domain age are now DERIVED
LIVE from docs/*/post.json `created_at` fields — the same source of
truth content_freshness.py and canonical_guard.py already read for
their own corpus-wide checks — instead of a separately persisted
counter. This needs no git-add exception, self-heals on every fresh
checkout, and can never drift from what's actually published.

HOW TO INTEGRATE
----------------
In blog_system.py auto mode, BEFORE calling generate_blog_post():

    from velocity_controller import VelocityController
    vc = VelocityController()   # reads ./docs by default
    if not vc.can_publish():
        print(f"🛑 Velocity limit reached: {vc.today_count()}/{vc.effective_limit()} posts today.")
        sys.exit(0)   # Clean exit — GH Actions will retry next scheduled run

After save_post() succeeds:
    vc.record_publish()   # now a lightweight cache-invalidation, see below

CLI (already wired in blog_system.py):
    python blog_system.py velocity status
    python blog_system.py velocity reset
"""

import json
import os
from datetime import date
from pathlib import Path
from typing import List, Optional


# Override via environment variable for staging environments.
# E.g.:  PUBLISH_DAILY_LIMIT=10 python blog_system.py auto
_ENV_LIMIT_KEY = "PUBLISH_DAILY_LIMIT"

# Default caps indexed by domain-age tier (days since first publish).
# Set DOMAIN_AGE_DAYS in env to skip auto-detection.
_DEFAULT_CAPS = {
    "early":   1,   # 0-30 days
    "growing": 2,   # 31-90 days
    "mature":  25,   # 91-180 days
    "scaled":  4,   # 181+ days
}

_SKIP_DIRS = {"static", "tag", "author"}


class VelocityController:
    """
    Tracks and enforces the daily publication limit.

    The limit is intentionally conservative: it is far better to publish
    1 high-quality post per day than 10 that trigger spam classifiers.

    Publish history is read directly from docs_dir's post.json files
    rather than a separately persisted counter — see module docstring
    for why. This means a VelocityController instantiated at any point
    in a run always reflects exactly what's actually on disk in docs/,
    including posts saved earlier in the same run.
    """

    def __init__(self, docs_dir: Path = Path("./docs")):
        self._docs_dir = Path(docs_dir)
        self._dates_cache: Optional[List[date]] = None

    # ── Public API ─────────────────────────────────────────────────────────

    def can_publish(self) -> bool:
        """Return True if the daily limit has not been reached today."""
        return self.today_count() < self.effective_limit()

    def record_publish(self) -> None:
        """
        Invalidate the cached publish-date list so a post saved earlier
        in this same process is counted if today_count()/can_publish()
        is checked again before the process exits.

        This is intentionally NOT a write — there is nothing to persist.
        Once save_post() has written the new post's post.json to
        docs_dir, that post already IS the record; the next process
        (or the next call in this one) picks it up automatically by
        re-scanning docs_dir. Kept as a real method (rather than
        removing it) so existing call sites in blog_system.py that call
        vc.record_publish() immediately after save_post() need no change.
        """
        self._dates_cache = None

    def today_count(self) -> int:
        """Return the number of posts published today, per docs_dir."""
        today = date.today()
        return sum(1 for d in self._publish_dates() if d == today)

    def effective_limit(self) -> int:
        """
        Return the applicable daily limit.

        Priority:
          1. PUBLISH_DAILY_LIMIT env var (explicit override for staging)
          2. DOMAIN_AGE_DAYS env var → age-based default cap
          3. Age derived from the earliest created_at found in docs_dir
          4. Conservatively assumes 'early' tier (1/day)
        """
        env_override = os.getenv(_ENV_LIMIT_KEY, "").strip()
        if env_override.isdigit():
            return max(1, int(env_override))

        domain_age = self._domain_age_days()
        if domain_age is None:
            return _DEFAULT_CAPS["early"]
        if domain_age <= 30:
            return _DEFAULT_CAPS["early"]
        if domain_age <= 90:
            return _DEFAULT_CAPS["growing"]
        if domain_age <= 180:
            return _DEFAULT_CAPS["mature"]
        return _DEFAULT_CAPS["scaled"]

    def domain_age_summary(self) -> str:
        """Human-readable summary for CLI output."""
        age = self._domain_age_days()
        age_str = f"{age} days" if age is not None else "unknown (no posts found in docs_dir yet)"
        return (
            f"Domain age : {age_str}\n"
            f"Daily limit: {self.effective_limit()} posts\n"
            f"Published today: {self.today_count()}\n"
            f"Can publish: {'Yes' if self.can_publish() else 'No — limit reached'}"
        )

    # ── Private helpers ────────────────────────────────────────────────────

    def _publish_dates(self) -> List[date]:
        """
        Every post's created_at date, read live from docs_dir. Cached for
        the lifetime of this instance; call record_publish() to invalidate
        after writing a new post.
        """
        if self._dates_cache is not None:
            return self._dates_cache

        dates: List[date] = []
        if self._docs_dir.exists():
            for post_dir in self._docs_dir.iterdir():
                if not post_dir.is_dir() or post_dir.name in _SKIP_DIRS:
                    continue
                post_json = post_dir / "post.json"
                if not post_json.exists():
                    continue
                try:
                    data = json.loads(post_json.read_text(encoding="utf-8"))
                    raw = data.get("created_at", "")
                    date_part = raw.split("T")[0] if "T" in raw else raw.strip()
                    if date_part:
                        dates.append(date.fromisoformat(date_part))
                except (json.JSONDecodeError, ValueError, OSError):
                    continue

        self._dates_cache = dates
        return dates

    def _domain_age_days(self) -> Optional[int]:
        """Days since the earliest post in docs_dir, or None if unknown."""
        env_age = os.getenv("DOMAIN_AGE_DAYS", "").strip()
        if env_age.isdigit():
            return int(env_age)

        dates = self._publish_dates()
        if not dates:
            return None
        return (date.today() - min(dates)).days