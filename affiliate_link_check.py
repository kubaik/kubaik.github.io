"""
affiliate_link_check.py — replaces monetization_validator.py.

Does NOT scrape amazon.com or rotate user agents (that file's approach reads
as bot-detection evasion against a third party, unrelated to AdSense and a
real ToS/reputational risk to keep in a production repo). This only checks
that your own outbound links are well-formed and reachable via HEAD.
"""
import re
import sys
import requests


def check_affiliate_tag_format(tag: str) -> list:
    issues = []
    if len(tag) > 20:
        issues.append(f"Tag '{tag}' exceeds Amazon's 20-char limit")
    if not re.fullmatch(r'[A-Za-z0-9_-]+', tag):
        issues.append(f"Tag '{tag}' contains disallowed characters")
    return issues


def check_link_reachable(url: str, timeout: int = 10) -> dict:
    try:
        r = requests.head(url, timeout=timeout, allow_redirects=True)
        return {"url": url, "status": r.status_code, "final_url": r.url}
    except requests.exceptions.RequestException as e:
        return {"url": url, "error": str(e)}


if __name__ == "__main__":
    urls = sys.argv[1:] or []
    for u in urls:
        print(check_link_reachable(u))
