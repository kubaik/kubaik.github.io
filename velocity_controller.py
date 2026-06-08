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
  - Days 31–90:      2 posts/day
  - Days 91–180:     3 posts/day
  - After 6 months:  up to 4 posts/day

HOW TO INTEGRATE
----------------
In blog_system.py auto mode, BEFORE calling generate_blog_post():

    from velocity_controller import VelocityController
    vc = VelocityController()
    if not vc.can_publish():
        print(f"🛑 Velocity limit reached: {vc.today_count()}/{vc.effective_limit()} posts today.")
        sys.exit(0)   # Clean exit — GH Actions will retry tomorrow

After save_post() succeeds:
    vc.record_publish()

CLI (already wired in blog_system.py):
    python blog_system.py velocity status
    python blog_system.py velocity reset
"""

import json
import os
from datetime import date
from pathlib import Path
from typing import Optional


_VELOCITY_FILE = Path(".publish_velocity.json")

# Override via environment variable for staging environments.
# E.g.:  PUBLISH_DAILY_LIMIT=10 python blog_system.py auto
_ENV_LIMIT_KEY = "PUBLISH_DAILY_LIMIT"

# Default caps indexed by domain-age tier (days since first publish).
# Set DOMAIN_AGE_DAYS in env to skip auto-detection.
_DEFAULT_CAPS = {
    "early":   1,   # 0–30 days
    "growing": 2,   # 31–90 days
    "mature":  3,   # 91–180 days
    "scaled":  4,   # 181+ days
}


class VelocityController:
    """
    Tracks and enforces the daily publication limit.

    The limit is intentionally conservative: it is far better to publish
    1 high-quality post per day than 10 that trigger spam classifiers.
    """

    def __init__(self, state_file: Path = _VELOCITY_FILE):
        self._file = state_file
        self._state = self._load()

    # ── Public API ─────────────────────────────────────────────────────────

    def can_publish(self) -> bool:
        """Return True if the daily limit has not been reached today."""
        return self.today_count() < self.effective_limit()

    def record_publish(self) -> None:
        """
        Increment today's counter.  Call once immediately after save_post()
        succeeds — not before, to avoid counting failed attempts.
        """
        today = date.today().isoformat()
        self._state.setdefault("days", {})
        self._state["days"][today] = self._state["days"].get(today, 0) + 1
        # Record first-ever publish date so age-based caps can be applied.
        if "first_publish" not in self._state:
            self._state["first_publish"] = today
        self._save()

    def today_count(self) -> int:
        """Return the number of posts published today."""
        today = date.today().isoformat()
        return self._state.get("days", {}).get(today, 0)

    def effective_limit(self) -> int:
        """
        Return the applicable daily limit.

        Priority:
          1. PUBLISH_DAILY_LIMIT env var (explicit override for staging)
          2. DOMAIN_AGE_DAYS env var → age-based default cap
          3. Age derived from first_publish date in state file
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
        age_str = f"{age} days" if age is not None else "unknown"
        return (
            f"Domain age : {age_str}\n"
            f"Daily limit: {self.effective_limit()} posts\n"
            f"Published today: {self.today_count()}\n"
            f"Can publish: {'Yes' if self.can_publish() else 'No — limit reached'}"
        )

    # ── Private helpers ────────────────────────────────────────────────────

    def _domain_age_days(self) -> Optional[int]:
        """Days since first ever publish, or None if unknown."""
        env_age = os.getenv("DOMAIN_AGE_DAYS", "").strip()
        if env_age.isdigit():
            return int(env_age)
        fp = self._state.get("first_publish")
        if not fp:
            return None
        try:
            delta = date.today() - date.fromisoformat(fp)
            return delta.days
        except (ValueError, TypeError):
            return None

    def _load(self) -> dict:
        if not self._file.exists():
            return {}
        try:
            with open(self._file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def _save(self) -> None:
        try:
            with open(self._file, "w", encoding="utf-8") as f:
                json.dump(self._state, f, indent=2)
        except OSError as e:
            print(f"⚠️  VelocityController: could not save state: {e}")
