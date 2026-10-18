import hashlib
import re
from typing import List, Tuple

# Neutral framings only. v1 used first-person lines ("the resource I'd
# actually use", "I'd point a colleague...") — that manufactures first-hand
# endorsement the AI pipeline cannot have, contradicting the site's own
# AI-content disclosure and Google's guidance on misleading affiliate content.
_CTA_TEMPLATES = [
    "Further reading: {link}.",
    "Related resource: {link}.",
    "For a deeper treatment of this topic, see {link}.",
    "Reference material on this: {link}.",
]
DISCLOSURE = (
    "*Disclosure: this page contains affiliate links. If you buy "
    "through them the site may earn a commission at no extra cost to you.*"
)


def _h(seed: str) -> int:
    return int(hashlib.md5(seed.encode()).hexdigest(), 16)


class MonetizationManager:
    """Affiliate injection. Disabled unless config['affiliate_enabled'] is true.

    Catalog comes from config['affiliate_catalog'] (human-verified entries:
    {topics:[...], url, text, keywords:[...], commission}). Hardcoded
    unverified ASINs were removed: a wrong product link is a misleading-
    content risk.
    """

    def __init__(self, config):
        self.config = config
        self.enabled = bool(config.get("affiliate_enabled", False))
        self.catalog = config.get("affiliate_catalog", []) or []

    def inject_affiliate_links(self, content: str, topic: str) -> Tuple[str, list]:
        if not self.enabled:
            return content, []
        links, enhanced = [], content
        for s in self._get_affiliate_suggestions(topic)[:2]:
            points = self._find_insertion_points(enhanced, s["keywords"])
            if not points:
                continue
            idx = points[_h(f"{topic}:{s['url']}") % len(points)]  # deterministic
            link = f'<a href="{s["url"]}" target="_blank" rel="nofollow sponsored noopener">{s["text"]}</a>'
            cta = _CTA_TEMPLATES[
                _h(f"cta:{topic}:{s['url']}") % len(_CTA_TEMPLATES)
            ].format(link=link)
            lines = enhanced.split("\n")
            lines.insert(idx + 1, f"\n*{cta}*")
            enhanced = "\n".join(lines)
            links.append(
                {
                    "url": s["url"],
                    "text": s["text"],
                    "commission_rate": s.get("commission", 0.0),
                }
            )
        if links and DISCLOSURE not in enhanced:
            enhanced = DISCLOSURE + "\n\n" + enhanced
        return enhanced, links

    def _get_affiliate_suggestions(self, topic: str) -> List[dict]:
        t = topic.lower()
        out = []
        for item in self.catalog:
            if any(
                re.search(rf"\b{re.escape(k.lower())}\b", t)
                for k in item.get("topics", [])
            ):
                out.append(
                    {
                        "url": item["url"],
                        "text": item["text"],
                        "keywords": item.get("keywords", []),
                        "commission": item.get("commission", 0.0),
                    }
                )
        return out

    def _find_insertion_points(self, content: str, keywords: list) -> List[int]:
        """Index of the LAST line of a plain prose paragraph mentioning a
        keyword. Skips code fences, tables, quotes, headings, lists, HTML."""
        lines = content.split("\n")
        pts, in_code = [], False
        for i, line in enumerate(lines):
            s = line.strip()
            if s.startswith(("```", "~~~")):
                in_code = not in_code
                continue
            if in_code or not s:
                continue
            if (
                s.startswith((">", "#", "<", "|"))
                or re.match(r"^([-*+]|\d+\.)\s", s)
                or "|" in s
            ):
                continue
            nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
            if nxt:  # not the end of the paragraph
                continue
            if any(
                re.search(rf"\b{re.escape(k.lower())}\b", s.lower()) for k in keywords
            ):
                pts.append(i)
        return pts

    def generate_ad_slots(self, content: str) -> dict:
        n = len(content.split("\n"))
        return {
            "header": 2,
            "middle": n // 2,
            "footer": max(0, n - 3),
            "ad_slots": 3,
            "affiliate_count": 0,
        }
