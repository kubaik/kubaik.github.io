"""
adsense_fixes/ads_txt_sync.py
=============================
Write a valid ads.txt to BOTH repo root and docs/ so GitHub Pages and
the raw repository URL resolve the same publisher line.

WHY THIS EXISTS
---------------
AdSense site status for kubaik.github.io showed ads.txt = "Not found"
(2026-08-27) while the generator only wrote ./docs/ads.txt. User Pages
sites serve from /docs OR the repo root depending on the Pages source
setting. Writing both paths removes that split-brain.

The IAB line format is:
    google.com, pub-XXXXXXXXXXXXXXXX, DIRECT, f08c47fec0942fa0

HOW TO INTEGRATE
----------------
Replace StaticSiteGenerator._generate_ads_txt() with:

    from adsense_fixes.ads_txt_sync import sync_ads_txt
    sync_ads_txt(self.blog_system.config.get("google_adsense_id", ""))

Also call from CI before deploy so a build that skips generate_site()
cannot ship without ads.txt.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional


ADSENSE_CERT_AUTHORITY_ID = "f08c47fec0942fa0"
_REQUIRED_DOMAIN = "google.com"
_REQUIRED_RELATIONSHIP = "DIRECT"


class AdsTxtError(Exception):
    """Invalid publisher id or failed write."""


def publisher_id(adsense_id: str) -> str:
    """Accept ca-pub-N, pub-N, or bare digits. Return digits only."""
    raw = (adsense_id or "").strip()
    raw = raw.replace("ca-pub-", "").replace("pub-", "")
    raw = raw.replace(" ", "")
    if not raw.isdigit() or len(raw) < 10:
        raise AdsTxtError(
            f"Invalid google_adsense_id {adsense_id!r}. "
            "Expected ca-pub-<digits> from the AdSense account."
        )
    return raw


def render_ads_txt(adsense_id: str) -> str:
    pub = publisher_id(adsense_id)
    return (
        f"{_REQUIRED_DOMAIN}, pub-{pub}, "
        f"{_REQUIRED_RELATIONSHIP}, {ADSENSE_CERT_AUTHORITY_ID}\n"
    )


def ads_txt_targets(repo_root: Path) -> List[Path]:
    docs = repo_root / "docs"
    return [repo_root / "ads.txt", docs / "ads.txt"]


def sync_ads_txt(
    adsense_id: str,
    repo_root: Path = Path("."),
) -> List[Path]:
    """
    Write ads.txt to repo root and docs/. Returns paths written.
    Raises AdsTxtError if the publisher id is missing/invalid.
    """
    if not (adsense_id or "").strip():
        raise AdsTxtError(
            "google_adsense_id is empty — refusing to skip ads.txt. "
            "Set google_adsense_id in config.yaml."
        )

    body = render_ads_txt(adsense_id)
    repo_root = Path(repo_root)
    docs = repo_root / "docs"
    docs.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    for target in ads_txt_targets(repo_root):
        target.write_text(body, encoding="utf-8")
        written.append(target)
    return written


def verify_ads_txt(adsense_id: str, repo_root: Path = Path(".")) -> List[str]:
    """Return issue strings (empty = pass). Safe for CI."""
    issues: List[str] = []
    try:
        expected = render_ads_txt(adsense_id)
    except AdsTxtError as exc:
        return [str(exc)]

    for target in ads_txt_targets(Path(repo_root)):
        if not target.exists():
            issues.append(f"missing {target}")
            continue
        actual = target.read_text(encoding="utf-8")
        if actual.strip() != expected.strip():
            issues.append(
                f"{target} content mismatch.\n"
                f"  expected: {expected.strip()}\n"
                f"  actual:   {actual.strip()}"
            )
    return issues


if __name__ == "__main__":
    import sys
    import yaml

    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    config_path = root / "config.yaml"
    adsense_id = ""
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        adsense_id = cfg.get("google_adsense_id", "")

    written = sync_ads_txt(adsense_id, repo_root=root)
    for path in written:
        print(f"Wrote {path}")
    leftover = verify_ads_txt(adsense_id, repo_root=root)
    if leftover:
        print("VERIFY FAIL:")
        for item in leftover:
            print(f"  - {item}")
        sys.exit(1)
    print("VERIFY PASS")
