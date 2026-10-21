"""url_utils.py — single source of truth for public post URLs.

Every outbound URL (tweets, replies, RSS, IndexNow, schema, internal links)
MUST come from here. GitHub Pages 301s /slug -> /slug/, so any slash-less URL
we publish shows up in Search Console as "Page with redirect".
"""

from urllib.parse import urlsplit, urlunsplit

BASE_URL = "https://kubaik.github.io"


def public_post_url(slug: str, base_url: str = BASE_URL) -> str:
    return f"{base_url.rstrip('/')}/{slug.strip('/')}/"


def normalize_url(url: str) -> str:
    """Force trailing slash on extensionless paths; leave files/queries alone."""
    p = urlsplit(url)
    path = p.path or "/"
    last = path.rsplit("/", 1)[-1]
    if not path.endswith("/") and "." not in last:
        path += "/"
    return urlunsplit((p.scheme, p.netloc, path, p.query, p.fragment))
