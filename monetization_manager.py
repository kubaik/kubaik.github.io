import hashlib
import random
from typing import Dict, List, Tuple

# FIX: every affiliate insertion previously used the identical literal
# template "*Recommended: <a ...>text</a>*" regardless of topic or post.
# Across hundreds of posts that is the same "scaled content abuse" signature
# already fixed for intro sentences elsewhere in this pipeline (identical
# templated text at scale is trivially detectable and is explicitly called
# out in Google's affiliate-content guidance as a low-value pattern). Rotate
# through non-generic framings, deterministically per-post so it stays
# reproducible/automated.
_CTA_TEMPLATES = [
    "If you want to go deeper on this, {link} is worth a look.",
    "{link} covers this in more detail than fits here.",
    "For a hands-on reference, {link} is a solid next step.",
    "I'd point a colleague hitting this toward {link}.",
    "{link} is the resource I'd actually use for this.",
]


def _select_cta(seed: str) -> str:
    idx = int(hashlib.md5(seed.encode()).hexdigest(), 16) % len(_CTA_TEMPLATES)
    return _CTA_TEMPLATES[idx]


class MonetizationManager:
    """Handles automated monetization features"""

    def __init__(self, config):
        self.config = config
        self.affiliate_programs = {
            'amazon': {
                'tag': config.get('amazon_affiliate_tag', 'aiblogcontent-20'),
                'api_key': config.get('amazon_api_key'),
                'keywords': ['tools', 'software', 'books', 'course', 'equipment']
            },
            'tech_products': {
                'categories': ['software', 'cloud', 'development', 'ai-tools'],
                'commission_rate': 0.05
            }
        }

    def inject_affiliate_links(self, content: str, topic: str) -> tuple:
        """Inject relevant affiliate links into content"""
        affiliate_links = []
        enhanced_content = content

        suggestions = self._get_affiliate_suggestions(topic)

        for suggestion in suggestions[:3]:  # Limit to 3 per post
            link_html = f'<a href="{suggestion["url"]}" target="_blank" rel="nofollow sponsored">{suggestion["text"]}</a>'

            insertion_points = self._find_insertion_points(
                content, suggestion["keywords"])

            if insertion_points:
                insert_at = random.choice(insertion_points)
                cta = _select_cta(seed=f"{topic}:{suggestion['url']}").format(
                    link=link_html)
                lines = enhanced_content.split('\n')
                if insert_at < len(lines):
                    lines[insert_at] += f"\n\n*{cta}*\n"
                enhanced_content = '\n'.join(lines)

                affiliate_links.append({
                    'url': suggestion['url'],
                    'text': suggestion['text'],
                    'commission_rate': suggestion.get('commission', 0.05)
                })

        return enhanced_content, affiliate_links

    def _get_affiliate_suggestions(self, topic: str) -> list:
        """Get relevant affiliate suggestions based on topic"""
        suggestions = []
        topic_lower = topic.lower()

        # AI/ML Tools
        if any(term in topic_lower for term in ['ai', 'machine learning', 'data science']):
            suggestions.extend([
                {
                    'url': f'https://amazon.com/dp/B08N5WRWNW?tag={self.affiliate_programs["amazon"]["tag"]}',
                    'text': 'Python Machine Learning by Sebastian Raschka',
                    'keywords': ['python', 'learning', 'algorithm'],
                    'commission': 0.04
                },
                {
                    'url': 'https://coursera.org/learn/machine-learning',
                    'text': 'Andrew Ng\'s Machine Learning Course',
                    'keywords': ['course', 'learn', 'training'],
                    'commission': 0.10
                }
            ])

        # Web Development
        if any(term in topic_lower for term in ['web', 'frontend', 'backend', 'javascript']):
            suggestions.extend([
                {
                    'url': f'https://amazon.com/dp/B07C3KLQWX?tag={self.affiliate_programs["amazon"]["tag"]}',
                    'text': 'Eloquent JavaScript Book',
                    'keywords': ['javascript', 'programming', 'web'],
                    'commission': 0.04
                },
                {
                    'url': 'https://digitalocean.com',
                    'text': 'DigitalOcean Cloud Hosting',
                    'keywords': ['hosting', 'deploy', 'server'],
                    'commission': 0.25
                }
            ])

        # DevOps/Cloud
        if any(term in topic_lower for term in ['devops', 'cloud', 'aws', 'docker']):
            suggestions.extend([
                {
                    'url': f'https://amazon.com/dp/B0816Q9F6Z?tag={self.affiliate_programs["amazon"]["tag"]}',
                    'text': 'Docker Deep Dive by Nigel Poulton',
                    'keywords': ['docker', 'container', 'devops'],
                    'commission': 0.04
                }
            ])

        return suggestions

    def _find_insertion_points(self, content: str, keywords: list) -> list:
        """Find good places to insert affiliate links.

        FIX: previously this scanned every line for keyword matches with no
        awareness of markdown structure. If the matched line happened to
        fall inside a fenced code block (```), a table row, or a
        blockquote, the injected "*Recommended: <a>...</a>*" text would
        land inside that structure and corrupt it (e.g. a link inserted
        mid-code-block renders as a broken/garbled code sample; inside a
        table row it breaks the table). This site's articles are
        code-heavy, so that was a real, frequent risk rather than a
        theoretical one.

        Fix: track fence state (toggle on ``` or ~~~ boundaries) while
        scanning, and exclude any line that is inside a code block, is a
        table row (or table separator like |---|---|), or is a
        blockquote line, from the candidate insertion points.
        """
        lines = content.split('\n')
        insertion_points = []
        in_code_block = False

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Toggle fence state on ``` or ~~~ boundaries. The fence line
            # itself is never a valid insertion point either way.
            if stripped.startswith('```') or stripped.startswith('~~~'):
                in_code_block = not in_code_block
                continue

            if in_code_block:
                continue

            # Blockquote lines (> ...) shouldn't get content appended,
            # or the appended text either merges into the quote or breaks
            # it depending on renderer.
            if stripped.startswith('>'):
                continue

            # Table rows (|cell|cell|) and separator rows (|---|---|)
            if self._is_table_row(stripped):
                continue

            if any(keyword.lower() in line.lower() for keyword in keywords):
                insertion_points.append(i)

        return insertion_points

    @staticmethod
    def _is_table_row(stripped_line: str) -> bool:
        """Detect a markdown table row or header-separator row."""
        if not stripped_line:
            return False
        # Header/body row: contains at least one pipe with content around it
        if '|' in stripped_line:
            return True
        return False

    def generate_ad_slots(self, content: str) -> dict:
        """Generate ad slot positions in content"""
        lines = content.split('\n')
        total_lines = len(lines)

        ad_slots = {
            'header': 2,
            'middle': total_lines // 2,
            'footer': total_lines - 3,
            'ad_slots': 3,
            'affiliate_count': 0
        }

        return ad_slots
