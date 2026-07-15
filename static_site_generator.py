from monetization_manager import MonetizationManager
from visibility_automator import VisibilityAutomator
from seo_optimizer import SEOOptimizer
from blog_post import BlogPost
from jinja2 import Template, Environment, BaseLoader
from datetime import datetime
from typing import Dict, List
from pathlib import Path
from collections import Counter
import markdown as md
import json
import math
import re


import html as _html_stdlib


def _safe_excerpt(meta_description: str, content: str, title: str = "",
                  max_len: int = 155) -> str:
    """Return a plain-text excerpt, HTML-escaped, safe for use in attributes."""
    import re

    desc = (meta_description or "").strip()

    _WEAK_OPENERS = (
        "this post", "in this article", "a guide to", "learn about",
        "an overview", "this tutorial", "this article", "we will",
        "you will learn", "i wrote this", "a colleague asked",
        "this took me", "i've seen this", "the short version",
        "i ran into this", "i've answered",
    )
    if desc and not any(desc.lower().startswith(w) for w in _WEAK_OPENERS):
        # Escape quotes/ampersands so the value is safe inside HTML attributes
        return _html_stdlib.escape(desc, quote=True)

    text = re.sub(r"```[\s\S]*?```", " ", content or "")
    text = re.sub(r"`[^`]+`", " ", text)
    text = re.sub(r"#{1,6}\s+", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"[*_]{1,3}", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    _INTRO_PATTERNS = re.compile(
        r'^(I |A colleague|This took me|I\'ve|The short version|I ran|'
        r'I spent|I have|Here\'s what|Writing this|This is a topic)',
        re.IGNORECASE
    )
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 40:
            continue
        if _INTRO_PATTERNS.match(sentence):
            continue
        if len(sentence) > max_len:
            sentence = sentence[:max_len].rsplit(
                " ", 1)[0].rstrip(".,;:") + "…"
        return _html_stdlib.escape(sentence, quote=True)

    fallback = f"Practical guide to {title}." if title else text[:max_len]
    return _html_stdlib.escape(fallback, quote=True)


_RELATED_STOP_WORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "can", "that",
    "this", "these", "those", "it", "its", "we", "you", "your", "our",
    "they", "their", "what", "which", "who", "when", "where", "how",
    "not", "no", "so", "if", "as", "than", "then", "about", "up",
    "out", "into", "more", "also", "just", "after", "before", "over",
    "some", "any", "all", "each", "both", "between", "through",
})


def _tokenize_for_similarity(text: str) -> List[str]:
    return re.findall(r"\b[a-z][a-z']{1,}\b", (text or "").lower())


def _build_tfidf_corpus(posts: List[BlogPost]) -> Dict[str, Dict[str, float]]:
    """Build a TF-IDF vector per post for content-based related-article
    selection.

    Previously, "Related Articles" were chosen purely by shared-tag count,
    which meant every post with the same primary tag surfaced the same
    handful of related posts — shallow, repetitive internal linking across
    645 articles. This builds real TF-IDF vectors (title terms weighted
    3x over body terms, since titles are the strongest topical signal)
    so related posts are ranked by actual content similarity instead.
    """
    doc_tokens: Dict[str, List[str]] = {}
    for p in posts:
        title_tokens = _tokenize_for_similarity(p.title) * 3
        body_tokens = _tokenize_for_similarity(p.content)
        tokens = [t for t in title_tokens + body_tokens
                  if t not in _RELATED_STOP_WORDS]
        doc_tokens[p.slug] = tokens

    doc_freq: Counter = Counter()
    for tokens in doc_tokens.values():
        doc_freq.update(set(tokens))

    n_docs = max(len(doc_tokens), 1)
    idf = {
        term: math.log((n_docs + 1) / (freq + 1)) + 1
        for term, freq in doc_freq.items()
    }

    vectors: Dict[str, Dict[str, float]] = {}
    for slug, tokens in doc_tokens.items():
        if not tokens:
            vectors[slug] = {}
            continue
        term_freq = Counter(tokens)
        total = len(tokens)
        vectors[slug] = {
            term: (count / total) * idf.get(term, 0.0)
            for term, count in term_freq.items()
        }
    return vectors


def _cosine_similarity(v1: Dict[str, float], v2: Dict[str, float]) -> float:
    if not v1 or not v2:
        return 0.0
    common = set(v1) & set(v2)
    if not common:
        return 0.0
    dot = sum(v1[k] * v2[k] for k in common)
    mag1 = math.sqrt(sum(x * x for x in v1.values()))
    mag2 = math.sqrt(sum(x * x for x in v2.values()))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


def _tag_definition(tag: str) -> str:
    """Return a short, human-readable definition of a tag topic for use in
    tag-page meta descriptions. Falls back to a generic phrase for tags that
    aren't in the curated list, so every tag still gets a real definition
    rather than a placeholder."""
    definitions = {
        'ai': 'artificial intelligence, machine learning, and applied AI engineering',
        'machine learning': 'machine learning concepts, tools, and real-world model building',
        'python': 'the Python programming language, its libraries, and best practices',
        'javascript': 'JavaScript development, from core language features to modern frameworks',
        'typescript': 'TypeScript typing patterns and their use in real-world codebases',
        'backend': 'server-side architecture, APIs, and backend engineering practices',
        'frontend': 'frontend development, UI engineering, and client-side architecture',
        'devops': 'deployment pipelines, infrastructure automation, and DevOps practices',
        'docker': 'containerization with Docker and container-based workflows',
        'kubernetes': 'container orchestration and Kubernetes cluster management',
        'api': 'API design, integration, and best practices for building web services',
        'database': 'database design, optimization, and data management',
        'sql': 'SQL query design, database performance, and data modeling',
        'security': 'application security, secure coding, and vulnerability prevention',
        'testing': 'software testing strategies, tools, and quality assurance',
        'career': 'career growth, job hunting, and professional development for developers',
        'productivity': 'developer productivity, tools, and workflow optimization',
        'cloud': 'cloud computing platforms, architecture, and best practices',
        'aws': 'building and running systems on Amazon Web Services',
        'react': 'building applications with React and the modern JavaScript ecosystem',
        'git': 'version control workflows and best practices with Git',
        'linux': 'Linux systems administration, tooling, and command-line workflows',
        'recovered': 'a range of practical software development topics',
        'blog': 'general software development topics and practical guides',
    }
    key = tag.strip().lower()
    if key in definitions:
        return definitions[key]
    return f"practical, hands-on approaches to {key} for software developers"


def _generate_tag_meta_description(tag_title: str, top_titles: List[str],
                                   definition: str, max_len: int = 300) -> str:
    """Build a richer tag-page meta description from the tag's definition
    plus its top article titles, instead of a generic boilerplate line."""
    top_titles = [t for t in top_titles if t][:3]
    if len(top_titles) >= 3:
        titles_part = (
            f'including "{top_titles[0]}," "{top_titles[1]}," '
            f'and "{top_titles[2]}"'
        )
    elif len(top_titles) == 2:
        titles_part = f'including "{top_titles[0]}" and "{top_titles[1]}"'
    elif len(top_titles) == 1:
        titles_part = f'including "{top_titles[0]}"'
    else:
        titles_part = ""

    desc = f"Articles on {tag_title} covering {definition}"
    if titles_part:
        desc += f", {titles_part}"
    desc += "."

    if len(desc) > max_len:
        desc = desc[:max_len].rsplit(" ", 1)[0].rstrip(",;: ") + "…"
    return desc


def _clean_url(url: str) -> str:
    """Remove Markdown link-formatting artifacts like [text](url) -> url."""
    import re
    cleaned = re.sub(r'\[([^\]]*)\]\(([^)]+)\)', r'\2', url)
    return cleaned.strip()


def _normalize_iso_date(dt_str: str) -> str:
    """
    Normalize an ISO datetime string to YYYY-MM-DDTHH:MM:SS+00:00.

    Google's Structured Data validator rejects microsecond precision
    (e.g. "2026-06-15T10:30:45.123456") — only second-precision with an
    explicit timezone offset is accepted.  This function strips microseconds
    and appends +00:00 if no timezone is present.
    """
    if not dt_str:
        return dt_str
    try:
        dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%dT%H:%M:%S+00:00')
    except Exception:
        # Fallback: manually strip microseconds, preserve any trailing tz offset
        if '.' in dt_str:
            base, frac = dt_str.split('.', 1)
            tz = ''
            for sep in ('+', '-'):
                if sep in frac:
                    tz = sep + frac.split(sep, 1)[1]
                    break
            return base + (tz or '+00:00')
        return dt_str


AUTHOR_PAGE_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kubai Kevin — Software Developer and Writer</title>
    <meta name="description" content="Kubai Kevin is a software developer based in Nairobi, Kenya. He writes about AI, backend engineering, and developer careers at {site_name}.">
    <meta name="author" content="Kubai Kevin">
    <link rel="canonical" href="{base_url}/author/kubai-kevin/">
    <link rel="stylesheet" href="{base_path}/static/style.css">
    <script type="application/ld+json">
    {{
      "@context": "https://schema.org",
      "@type": "ProfilePage",
      "mainEntity": {{
        "@type": "Person",
        "@id": "{base_url}/about/#author",
        "name": "Kubai Kevin",
        "url": "{base_url}/about/",
        "sameAs": [
          "https://www.linkedin.com/in/kevin-kubai-22b61b37/",
          "https://twitter.com/KubaiKevin",
          "https://github.com/kubaik"
        ]
      }}
    }}
    </script>
</head>
<body>
    <header><div class="container">
        <h1><a href="{base_path}/">{site_name}</a></h1>
        <nav><a href="{base_path}/">Home</a><a href="{base_path}/about/">About</a></nav>
    </div></header>
    <main class="container">
        <h1>Kubai Kevin</h1>
        <p>Software developer based in Nairobi, Kenya. Writing about AI, backend engineering,
        and developer careers at <a href="{base_url}/">{site_name}</a>.</p>
        <p>
            <a href="{base_path}/about/">Full bio and editorial process →</a>
        </p>
        <p>
            <a href="https://www.linkedin.com/in/kevin-kubai-22b61b37/" target="_blank" rel="noopener">LinkedIn</a> ·
            <a href="https://twitter.com/KubaiKevin" target="_blank" rel="noopener">Twitter</a> ·
            <a href="https://github.com/kubaik" target="_blank" rel="noopener">GitHub</a>
        </p>
        {posts_html}
    </main>
    <footer><div class="container">
        <p>&copy; {year} {site_name}</p>
    </div></footer>
</body>
</html>"""


class StaticSiteGenerator:
    def __init__(self, blog_system):
        self.blog_system = blog_system
        self.seo = SEOOptimizer(blog_system.config)
        self.visibility = VisibilityAutomator(blog_system.config)
        self.monetization = MonetizationManager(blog_system.config)
        self.templates = self._load_templates()

    def generate_site(self):
        print("Generating static site...")
        posts = self._get_all_posts()
        if not posts:
            print("No posts found. Creating placeholder homepage...")
        self._generate_homepage(posts)
        self._generate_post_pages(posts)
        self._generate_static_pages(posts)
        self._generate_author_page(posts)
        self._generate_dmca_page()
        self._generate_ai_disclosure_page()
        self._generate_rss_feed(posts)
        self._generate_sitemap(posts)
        self._generate_posts_json(posts)
        self._generate_robots_txt()
        self._generate_privacy_consent_banner()
        self._generate_security_headers()
        self._generate_404_page()
        self._generate_ads_txt()
        self._generate_pwa_files()
        self._generate_tag_pages(posts)

        print(f"Site generated successfully with {len(posts)} posts!")

    def _generate_ads_txt(self):
        config = self.blog_system.config
        adsense_id = config.get('google_adsense_id', '')
        if adsense_id:
            pub_id = adsense_id.replace('ca-pub-', '')
            ads_txt_content = f"google.com, pub-{pub_id}, DIRECT, f08c47fec0942fa0\n"
            with open("./docs/ads.txt", 'w', encoding='utf-8') as f:
                f.write(ads_txt_content)
            print("Generated ads.txt")
        else:
            print("Warning: no google_adsense_id in config — skipping ads.txt")

    def _generate_robots_txt(self):
        """Generate a clean and effective robots.txt file."""
        config = self.blog_system.config
        base_url = config.get('base_url', '').rstrip('/')

        content = f"""# robots.txt for {base_url}
# Generated automatically - Do not edit manually

User-agent: *
Allow: /

# Explicitly allow Googlebot
User-agent: Googlebot
Allow: /

# Allow Google AdSense crawler
User-agent: Mediapartners-Google
Allow: /

# Block sensitive paths only
User-agent: *
Disallow: /admin/
Disallow: /private/
Disallow: /.git/
Disallow: /tag/

# Explicitly allow Google's ad crawler regardless of the rule above
User-agent: Mediapartners-Google
Allow: /

# Sitemap
Sitemap: {base_url}/sitemap.xml
Sitemap: {base_url}/rss.xml
"""

        with open("./docs/robots.txt", 'w', encoding='utf-8') as f:
            f.write(content)

        print("Generated robots.txt")

    def _get_all_posts(self) -> List[BlogPost]:
        posts = []
        docs_dir = Path("./docs")
        if not docs_dir.exists():
            return posts
        for post_dir in docs_dir.iterdir():
            if not post_dir.is_dir() or post_dir.name == 'static':
                continue
            post_json = post_dir / "post.json"
            if post_json.exists():
                try:
                    with open(post_json, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    posts.append(BlogPost.from_dict(data))
                except Exception as e:
                    print(f"Error loading post from {post_dir.name}: {e}")
        posts.sort(key=lambda p: p.created_at, reverse=True)
        return posts

    def _reading_time_minutes(self, content: str) -> int:
        word_count = len(content.split())
        return max(1, round(word_count / 200))

    def _valid_social_links(self, config: Dict) -> Dict[str, str]:
        """
        Return only social_accounts entries that are real, absolute URLs.

        BUG FIX: config.yaml ships with placeholder values for unconfigured
        platforms (e.g. social_accounts.facebook = "your-facebook-page").
        The footer template previously rendered these placeholders directly
        as href="{{ url }}", producing broken same-site links like
        https://kubaik.github.io/your-facebook-page (404) instead of either
        a real external profile or no link at all. Broken outbound links on
        every page are a crawl-budget and trust-signal problem for SEO and
        an unprofessional signal during AdSense review. This filters to
        http(s) URLs only, so unconfigured platforms are silently omitted
        until the user provides a real link.
        """
        raw = config.get('social_accounts', {}) or {}
        return {
            platform: url
            for platform, url in raw.items()
            if isinstance(url, str) and url.strip().lower().startswith(('http://', 'https://'))
        }

    def _generate_homepage(self, posts: List[BlogPost]):
        config = self.blog_system.config

        # Only the first HOMEPAGE_SSR_LIMIT posts are rendered server-side.
        # The rest are loaded on demand from /posts.json via client-side JS.
        # This keeps the initial HTML under ~60 KB (was 720 KB with 500 posts),
        # dramatically improving LCP and CLS Core Web Vitals scores.
        HOMEPAGE_SSR_LIMIT = 24

        posts_data = []
        for p in posts:
            post_dict = p.to_dict()
            post_dict['display_date'] = self._format_display_date(p.created_at)
            post_dict['short_tags'] = sorted(p.tags, key=len)[:3]
            post_dict['reading_time'] = self._reading_time_minutes(p.content)
            post_dict['meta_description'] = _safe_excerpt(
                p.meta_description, p.content, p.title)
            # Strip the full content from the homepage payload — it is only
            # needed on individual post pages.
            post_dict.pop('content', None)
            posts_data.append(post_dict)

        context = {
            'site_name': config.get('site_name', 'Kubai Kevin'),
            'site_description': config.get('site_description', 'An AI-powered blog'),
            'base_path': config.get('base_path', ''),
            'base_url': config.get('base_url', ''),
            'posts': posts_data[:HOMEPAGE_SSR_LIMIT],
            'posts_per_page': HOMEPAGE_SSR_LIMIT,
            'total_posts': len(posts_data),
            'current_year': datetime.now().year,
            'social_links': self._valid_social_links(config),
            'global_meta_tags': self.seo.generate_global_meta_tags(),
            'homepage_meta_tags': self.seo.generate_homepage_meta_tags(),
            'organization_schema': self.seo.generate_organization_schema(),
            'website_schema': self.seo.generate_website_schema(),
            # FIX BUG-14: the homepage (the page AdSense reviews first) never
            # rendered any actual <ins class="adsbygoogle"> ad units — only
            # post pages did. The google-adsense-account meta tag and the
            # adsbygoogle.js loader (both from generate_global_meta_tags())
            # are present on every page, but a verification meta tag is not
            # an ad placement. Wire up real ad units here so the homepage
            # has actual inventory, matching what POST_TMPL already does.
            'header_ad': self.seo.generate_adsense_ad('header'),
            'middle_ad': self.seo.generate_adsense_ad('middle'),
            'footer_ad': self.seo.generate_adsense_ad('footer'),
        }
        html = self.templates['index'].render(**context)
        output_file = Path("./docs/index.html")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)
        print(
            f"Generated homepage: {HOMEPAGE_SSR_LIMIT} SSR posts "
            f"+ {max(0, len(posts) - HOMEPAGE_SSR_LIMIT)} deferred via posts.json"
        )

    def _format_display_date(self, iso_date: str) -> str:
        try:
            dt = datetime.fromisoformat(iso_date.replace('Z', '+00:00'))
            return dt.strftime('%-d %B %Y')
        except:
            try:
                dt = datetime.fromisoformat(iso_date.replace('Z', '+00:00'))
                return dt.strftime('%d %B %Y').lstrip('0')
            except:
                return iso_date.split('T')[0]

    def _generate_404_page(self):
        config = self.blog_system.config
        context = {
            'site_name': config.get('site_name', 'Tech Blog'),
            'base_path': config.get('base_path', ''),
            'current_year': datetime.now().year,
            'global_meta_tags': self.seo.generate_global_meta_tags(),
        }
        html = self.templates['not_found'].render(**context)
        with open("./docs/404.html", 'w', encoding='utf-8') as f:
            f.write(html)
        print("Generated 404.html")

    def _generate_dmca_page(self):
        """Generate /dmca/ page."""
        config = self.blog_system.config
        page_dir = Path("./docs/dmca")
        page_dir.mkdir(exist_ok=True)
        context = {
            'site_name': config.get('site_name', 'Tech Blog'),
            'base_path': config.get('base_path', ''),
            'base_url': config.get('base_url', ''),
            'current_year': datetime.now().year,
            'current_date': datetime.now().strftime('%B %d, %Y'),
            'global_meta_tags': self.seo.generate_global_meta_tags(),
        }
        html = self.templates['dmca'].render(**context)
        with open(page_dir / "index.html", 'w', encoding='utf-8') as f:
            f.write(html)
        print("Generated /dmca/ page")

    def _generate_ai_disclosure_page(self):
        """Generate /ai-content-policy/ page."""
        config = self.blog_system.config
        page_dir = Path("./docs/ai-content-policy")
        page_dir.mkdir(exist_ok=True)
        context = {
            'site_name': config.get('site_name', 'Tech Blog'),
            'base_path': config.get('base_path', ''),
            'base_url': config.get('base_url', ''),
            'current_year': datetime.now().year,
            'current_date': datetime.now().strftime('%B %d, %Y'),
            'global_meta_tags': self.seo.generate_global_meta_tags(),
        }
        html = self.templates['ai_disclosure'].render(**context)
        with open(page_dir / "index.html", 'w', encoding='utf-8') as f:
            f.write(html)
        print("Generated /ai-content-policy/ page")

    def _generate_pwa_files(self):
        import shutil
        import re
        import time

        sw_src = Path("sw.js")
        if sw_src.exists():
            sw_text = sw_src.read_text(encoding="utf-8")
            new_version = f"v{int(time.time())}"
            patched, n = re.subn(
                r"(const\s+CACHE_VERSION\s*=\s*')[^']*(')",
                lambda m: m.group(1) + new_version + m.group(2),
                sw_text,
            )
            if n == 0:
                patched, n = re.subn(
                    r"((?:cache|CACHE)[_-])(v\d+)",
                    lambda m: m.group(1) + new_version,
                    sw_text,
                )
            if n == 0:
                print(
                    "Warning: could not find a cache version string in sw.js — SW cache may be stale")
                patched = sw_text
            Path("./docs/sw.js").write_text(patched, encoding="utf-8")
            print(f"Generated docs/sw.js with cache version {new_version}")
        else:
            print("Warning: sw.js not found — skipping")

        for src, dst in [("offline.html", "./docs/offline.html"),
                         ("manifest.json", "./docs/manifest.json")]:
            src_path = Path(src)
            if src_path.exists():
                import shutil as _shutil
                _shutil.copy2(src_path, dst)
                print(f"Copied {src} → {dst}")
            else:
                print(f"Warning: {src} not found — skipping")

        pwa_js_candidates = [Path("static/pwa.js"), Path("docs/static/pwa.js")]
        pwa_js_src = next((p for p in pwa_js_candidates if p.exists()), None)
        if pwa_js_src:
            if str(pwa_js_src) != "docs/static/pwa.js":
                import shutil as _shutil
                _shutil.copy2(pwa_js_src, "./docs/static/pwa.js")
                print(f"Copied {pwa_js_src} → docs/static/pwa.js")
        else:
            print("Warning: pwa.js not found — skipping")

    def _generate_article_schema(self, post, base_url: str, site_name: str = None) -> str:
        import json as _json

        word_count = len(post.content.split())
        reading_time = max(1, round(word_count / 200))

        # FIX BUG-6: _normalize_iso_date() existed in this file but was NOT
        # called here — dates were passed raw with microseconds (e.g.
        # "2026-06-15T10:30:45.123456") which Google's Structured Data
        # validator rejects. Now normalized to second precision with UTC offset.
        published = _normalize_iso_date(post.created_at)
        modified = _normalize_iso_date(post.updated_at)

        schemas = [
            {
                "@type": "Article",
                "@id": f"{base_url}/{post.slug}/#article",
                "headline": post.title,
                "description": post.meta_description or "",
                "datePublished": published,
                "dateModified": modified,
                "wordCount": word_count,
                "timeRequired": f"PT{reading_time}M",
                "articleSection": "Technology",
                "inLanguage": "en-US",
                "author": {
                    "@type": "Person",
                    "@id": f"{base_url}/about/#author",
                    "name": "Kubai Kevin",
                    "jobTitle": "Software Developer",
                    "url": f"{base_url}/about/",
                    "sameAs": [
                        "https://www.linkedin.com/in/kevin-kubai-22b61b37/",
                        "https://twitter.com/KubaiKevin",
                        "https://github.com/kubaik"
                    ],
                    "knowsAbout": [
                        "Python", "Node.js", "TypeScript", "AWS",
                        "Backend Systems", "AI", "Machine Learning"
                    ]
                },
                "publisher": {
                    "@type": "Organization",
                    "@id": f"{base_url}/#organization",
                    "name": site_name or "Tech Blog",
                    "url": base_url
                },
                "mainEntityOfPage": {
                    "@type": "WebPage",
                    "@id": f"{base_url}/{post.slug}/"
                },
                "keywords": ", ".join(post.seo_keywords[:8]) if post.seo_keywords else "",
            },
            {
                "@type": "BreadcrumbList",
                "itemListElement": [
                    {"@type": "ListItem", "position": 1,
                     "name": "Home", "item": f"{base_url}/"},
                    {"@type": "ListItem", "position": 2,
                     "name": post.title, "item": f"{base_url}/{post.slug}/"}
                ]
            }
        ]

        output_blocks = [f'''<script type="application/ld+json">
{_json.dumps({"@context": "https://schema.org", "@graph": schemas},
             indent=2, ensure_ascii=False)}
</script>''']

        faq_schema = (post.monetization_data or {}).get('faq_schema', '')
        if faq_schema:
            output_blocks.append(
                f'<script type="application/ld+json">\n{faq_schema}\n</script>')

        howto_schema = (post.monetization_data or {}).get('howto_schema', '')
        if howto_schema:
            output_blocks.append(
                f'<script type="application/ld+json">\n{howto_schema}\n</script>')

        return '\n'.join(output_blocks)

    def _generate_security_headers(self):
        headers_content = """\
/*
X-Frame-Options: SAMEORIGIN
X-Content-Type-Options: nosniff
Referrer-Policy: strict-origin-when-cross-origin
Permissions-Policy: geolocation=(), microphone=(), camera=()
Content-Security-Policy: default-src 'self'; script-src 'self' 'unsafe-inline' https://pagead2.googlesyndication.com https://googleads.g.doubleclick.net https://www.googletagmanager.com https://www.google-analytics.com https://partner.googleadservices.com https://tpc.googlesyndication.com https://adservice.google.com https://googletagservices.com; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; img-src 'self' data: https:; frame-src https://googleads.g.doubleclick.net https://tpc.googlesyndication.com https://googletagservices.com https://adservice.google.com https://www.google.com; connect-src 'self' https://www.google-analytics.com https://analytics.google.com https://adservice.google.com
"""
        with open("./docs/_headers", "w", encoding="utf-8") as f:
            f.write(headers_content)

        htaccess_content = """\
# Security headers for Apache hosts
<IfModule mod_headers.c>
    Header always set X-Frame-Options "SAMEORIGIN"
    Header always set X-Content-Type-Options "nosniff"
    Header always set Referrer-Policy "strict-origin-when-cross-origin"
    Header always set Permissions-Policy "geolocation=(), microphone=(), camera=()"
    Header always set Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline' https://pagead2.googlesyndication.com https://googleads.g.doubleclick.net https://www.googletagmanager.com https://www.google-analytics.com https://partner.googleadservices.com https://tpc.googlesyndication.com https://adservice.google.com https://googletagservices.com; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; img-src 'self' data: https:; frame-src https://googleads.g.doubleclick.net https://tpc.googlesyndication.com https://googletagservices.com https://adservice.google.com https://www.google.com; connect-src 'self' https://www.google-analytics.com https://analytics.google.com https://adservice.google.com"
</IfModule>

# Gzip compression
<IfModule mod_deflate.c>
    AddOutputFilterByType DEFLATE text/html text/css application/javascript application/json text/xml
</IfModule>

# Cache static assets
<IfModule mod_expires.c>
    ExpiresActive On
    ExpiresByType text/css "access plus 1 year"
    ExpiresByType application/javascript "access plus 1 year"
    ExpiresByType image/png "access plus 1 year"
    ExpiresByType image/jpeg "access plus 1 year"
    ExpiresByType image/svg+xml "access plus 1 year"
</IfModule>
"""
        with open("./docs/.htaccess", "w", encoding="utf-8") as f:
            f.write(htaccess_content)

        print("Generated docs/_headers (Netlify/Cloudflare) and docs/.htaccess (Apache)")

    def _generate_privacy_consent_banner(self):
        consent_js = r"""/* consent.js — GDPR Cookie Consent v2 with Consent Mode v2 support
 *
 * NOTE: the *default* consent signal (granted/denied) is now pushed
 * synchronously via a tiny inline <script> in <head> (emitted directly
 * in each page template) BEFORE this file is even requested. That
 * inline snippet has zero network cost and guarantees Consent Mode
 * defaults are set before the async GA/AdSense tags execute.
 *
 * This file is now loaded with `defer` — it only needs to run before
 * the user can interact with the page, not before any tag fires. It
 * still re-asserts the consent state on load (harmless, idempotent)
 * and owns all banner UI / accept-decline logic.
 */
(function () {
  'use strict';

  var CONSENT_KEY = 'cookie_consent_v1';
  var BANNER_ID   = 'cookie-consent-banner';
  var STYLE_ID    = 'cookie-consent-banner-styles';

  function getCookie(name) {
    var escapedName = name.replace(/([.*+?^=!:${}()|[\]/\\])/g, '\\$1');
    var pattern = '(?:^|;)\\s*' + escapedName + '=([^;]*)';
    var m = document.cookie.match(new RegExp(pattern));
    return m ? decodeURIComponent(m[1]) : null;
  }

  function setCookie(name, value, days) {
    var expires = new Date(Date.now() + days * 864e5).toUTCString();
    document.cookie =
      name + '=' + encodeURIComponent(value) +
      '; expires=' + expires +
      '; path=/' +
      '; SameSite=Lax';
  }

  function removeBanner() {
    var el = document.getElementById(BANNER_ID);
    if (el) {
      el.style.opacity = '0';
      el.style.transform = 'translateY(20px)';
      setTimeout(function () {
        if (el.parentNode) el.remove();
      }, 300);
    }
  }

  function pushConsentDefault(granted) {
    window.dataLayer = window.dataLayer || [];
    function gtag() { window.dataLayer.push(arguments); }
    var state = granted ? 'granted' : 'denied';
    gtag('consent', 'default', {
      ad_storage:             state,
      ad_user_data:           state,
      ad_personalization:     state,
      analytics_storage:      state,
      functionality_storage:  state,
      personalization_storage: state,
      wait_for_update: granted ? 0 : 500
    });
  }

  function updateConsent(granted) {
    if (typeof gtag !== 'function') {
      window.dataLayer = window.dataLayer || [];
      window.gtag = function () { window.dataLayer.push(arguments); };
    }
    var state = granted ? 'granted' : 'denied';
    gtag('consent', 'update', {
      ad_storage:             state,
      ad_user_data:           state,
      ad_personalization:     state,
      analytics_storage:      state,
      functionality_storage:  state,
      personalization_storage: state
    });
    window.dataLayer.push({
      event: 'consent_update',
      consent_granted: granted
    });
  }

  function accept() {
    setCookie(CONSENT_KEY, 'accepted', 365);
    updateConsent(true);
    removeBanner();
  }

  function decline() {
    setCookie(CONSENT_KEY, 'declined', 180);
    updateConsent(false);
    removeBanner();
  }

  // FIX: banner styling now lives in an injected <style> tag (rather than
  // inline JS styles) so it can respond to `prefers-color-scheme` — dark
  // theme by default, lighter theme for users who have light mode set,
  // matching the design used elsewhere on the site. This also gives us
  // a proper :focus-visible ring for keyboard users.
  function injectStyles() {
    if (document.getElementById(STYLE_ID)) return;
    var style = document.createElement('style');
    style.id = STYLE_ID;
    style.textContent = [
      '#' + BANNER_ID + '{',
        'position:fixed;bottom:0;left:0;right:0;z-index:99999;',
        'background:#1e1b4b;color:#e0e7ff;padding:16px 24px;',
        'font-family:system-ui,-apple-system,sans-serif;font-size:14px;line-height:1.5;',
        'box-shadow:0 -4px 24px rgba(0,0,0,0.3);',
        'opacity:0;transform:translateY(20px);',
        'transition:opacity .3s ease,transform .3s ease;',
      '}',
      '#' + BANNER_ID + ' .cc-inner{',
        'max-width:1024px;margin:0 auto;display:flex;',
        'align-items:center;gap:16px;flex-wrap:wrap;',
      '}',
      '#' + BANNER_ID + ' .cc-text{flex:1 1 260px;min-width:240px;margin:0;}',
      '#' + BANNER_ID + ' .cc-text a{color:#a5b4fc;}',
      '#' + BANNER_ID + ' .cc-buttons{display:flex;gap:8px;flex-shrink:0;flex-wrap:wrap;}',
      '#' + BANNER_ID + ' .cc-btn{',
        'padding:8px 18px;border-radius:6px;border:none;',
        'font-size:14px;font-weight:600;cursor:pointer;white-space:nowrap;',
      '}',
      '#' + BANNER_ID + ' .cc-btn:focus-visible{outline:3px solid #818cf8;outline-offset:2px;}',
      '#' + BANNER_ID + ' .cc-accept{background:#6366f1;color:#fff;}',
      '#' + BANNER_ID + ' .cc-accept:hover{background:#4f46e5;}',
      '#' + BANNER_ID + ' .cc-decline{background:transparent;color:#e0e7ff;border:1px solid #6366f1;}',
      '#' + BANNER_ID + ' .cc-decline:hover{background:rgba(99,102,241,0.15);}',
      '@media (prefers-color-scheme: light){',
        '#' + BANNER_ID + '{background:#f5f3ff;color:#1e1b4b;box-shadow:0 -4px 24px rgba(0,0,0,0.12);}',
        '#' + BANNER_ID + ' .cc-text a{color:#4f46e5;}',
        '#' + BANNER_ID + ' .cc-decline{color:#1e1b4b;border-color:#4f46e5;}',
        '#' + BANNER_ID + ' .cc-decline:hover{background:rgba(79,70,229,0.08);}',
      '}'
    ].join('');
    document.head.appendChild(style);
  }

  function showBanner(privacyUrl) {
    if (document.getElementById(BANNER_ID)) return;

    injectStyles();

    var banner = document.createElement('div');
    banner.id = BANNER_ID;
    // FIX: role="dialog" + aria-modal="true" + aria-hidden management,
    // matching the accessible banner pattern (was aria-modal="false" with
    // no aria-hidden state before).
    banner.setAttribute('role', 'dialog');
    banner.setAttribute('aria-modal', 'true');
    banner.setAttribute('aria-label', 'Cookie consent');
    banner.setAttribute('aria-hidden', 'false');

    banner.innerHTML = [
      '<div class="cc-inner">',
        '<p class="cc-text">',
          'We use cookies to improve your experience and serve relevant ads. ',
          '<a href="' + privacyUrl + '">Privacy Policy</a>',
        '</p>',
        '<div class="cc-buttons">',
          '<button type="button" class="cc-btn cc-decline" id="cc-decline" aria-label="Decline cookies">Decline</button>',
          '<button type="button" class="cc-btn cc-accept" id="cc-accept" aria-label="Accept cookies">Accept</button>',
        '</div>',
      '</div>'
    ].join('');

    document.body.appendChild(banner);

    requestAnimationFrame(function () {
      requestAnimationFrame(function () {
        banner.style.opacity = '1';
        banner.style.transform = 'translateY(0)';
      });
    });

    document.getElementById('cc-accept').addEventListener('click', accept);
    document.getElementById('cc-decline').addEventListener('click', decline);

    // FIX: focus management — move focus into the banner once it renders,
    // so keyboard and screen-reader users land on it immediately rather
    // than having to tab through the rest of the page first.
    var firstBtn = banner.querySelector('.cc-btn');
    if (firstBtn) {
      setTimeout(function () { firstBtn.focus(); }, 100);
    }

    // FIX: keyboard trap (WCAG 2.1 SC 2.1.2) — Tab/Shift+Tab cycle between
    // the two buttons while the banner is open, instead of letting focus
    // escape into the rest of the page.
    banner.addEventListener('keydown', function (e) {
      if (e.key !== 'Tab') return;
      var focusable = this.querySelectorAll('.cc-btn');
      var first = focusable[0], last = focusable[focusable.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault(); last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault(); first.focus();
      }
    });

    document.addEventListener('keydown', function onKey(e) {
      if (e.key === 'Escape') {
        decline();
        document.removeEventListener('keydown', onKey);
      }
    });
  }

  var existing = getCookie(CONSENT_KEY);

  if (existing === 'accepted') {
    pushConsentDefault(true);
    updateConsent(true);
  } else if (existing === 'declined') {
    pushConsentDefault(false);
    updateConsent(false);
  } else {
    pushConsentDefault(false);

    var basePath = (
      document.querySelector('meta[name="base-path"]') &&
      document.querySelector('meta[name="base-path"]').getAttribute('content')
    ) || '';

    var privacyUrl = basePath + '/privacy-policy/';

    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', function () {
        showBanner(privacyUrl);
      });
    } else {
      showBanner(privacyUrl);
    }
  }
}());
"""
        static_dir = Path("./docs/static")
        static_dir.mkdir(exist_ok=True)
        with open(static_dir / "consent.js", "w", encoding="utf-8") as f:
            f.write(consent_js)
        print("Generated docs/static/consent.js")

    def _generate_post_pages(self, posts: List[BlogPost]):
        config = self.blog_system.config
        for i, post in enumerate(posts):
            post_dir = Path("./docs") / post.slug
            post_dir.mkdir(exist_ok=True)
            markdown_converter = md.Markdown(
                extensions=['extra', 'fenced_code', 'toc'])
            content_html = markdown_converter.convert(post.content)

            related = self._find_related_posts(post, posts, max_count=3)

            post_dict = post.to_dict()
            post_dict['content_html'] = content_html
            post_dict['display_date'] = self._format_display_date(
                post.created_at)
            post_dict['updated_date'] = self._format_display_date(
                post.updated_at)
            post_dict['reading_time'] = self._reading_time_minutes(
                post.content)
            post_dict['word_count'] = len(post.content.split())
            post_dict['meta_description'] = _safe_excerpt(
                post.meta_description, post.content, post.title)
            post_dict['review_date'] = datetime.now().strftime('%B %Y')

            # FIX BUG-13: normalize before splitting so we handle ISO strings
            # that lack a 'T' separator (e.g. recovered posts from
            # from_markdown_file() which use datetime.now().isoformat() and
            # always have 'T', but belt-and-suspenders is correct here).
            updated_normalized = _normalize_iso_date(post.updated_at)
            post_dict['last_updated_iso'] = (
                updated_normalized.split('T')[0]
                if 'T' in updated_normalized
                else updated_normalized
            )

            post_dict['has_code'] = '```' in post.content
            post_dict['estimated_accuracy'] = 'Reviewed by author before publishing'
            post_dict['affiliate_links'] = post.affiliate_links or []

            # FIX BUG-7: has_og_image was never set on post_dict, so the Jinja
            # template's {{ og_img_png if post.has_og_image else og_img_fallback }}
            # always fell through to the generic icon PNG, defeating Twitter card
            # sharing for every post.  We check whether the PNG generated by
            # generate_og_card() actually exists on disk before setting the flag.
            og_card_path = Path("./docs/static/og") / f"{post.slug}.png"
            post_dict['has_og_image'] = og_card_path.exists()

            context = {
                'site_name': config.get('site_name', 'Tech Blog'),
                'base_path': config.get('base_path', ''),
                'base_url': config.get('base_url', ''),
                'post': post_dict,
                'related_posts': related,
                'current_year': datetime.now().year,
                'global_meta_tags': self.seo.generate_global_meta_tags(),
                'meta_tags': self.seo.generate_meta_tags(post),
                # NOTE: seo.generate_structured_data() (BlogPosting) used to be
                # emitted here alongside _generate_article_schema() (Article +
                # BreadcrumbList). Both described the same URL with different
                # @type and different publisher names ("Tech Blog" vs the real
                # site_name), which is exactly the kind of duplicate/conflicting
                # structured data Google's Rich Results Test flags and which can
                # cause a page's markup to be ignored entirely. Keeping only the
                # richer Article graph; site_name is now passed through so the
                # publisher name is correct instead of a hardcoded placeholder.
                'structured_data': '',
                'article_schema': self._generate_article_schema(
                    post, config.get('base_url', ''), config.get('site_name', 'Tech Blog')),
                'header_ad': self.seo.generate_adsense_ad('header'),
                'middle_ad': self.seo.generate_adsense_ad('middle'),
                'footer_ad': self.seo.generate_adsense_ad('footer'),
                'inline_ad': self.seo.generate_adsense_ad('inline'),
            }
            html = self.templates['post'].render(**context)
            output_file = post_dir / "index.html"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html)

    def _find_related_posts(self, current: BlogPost, all_posts: List[BlogPost],
                            max_count: int = 3) -> List[Dict]:
        current_tags = set(t.lower() for t in current.tags)
        scored = []
        for p in all_posts:
            if p.slug == current.slug:
                continue
            overlap = len(current_tags & set(t.lower() for t in p.tags))
            if overlap > 0:
                scored.append((overlap, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        result = []
        for _, p in scored[:max_count]:
            excerpt = _safe_excerpt(
                p.meta_description, p.content, p.title, max_len=120)
            result.append({
                'title': p.title,
                'slug': p.slug,
                'meta_description': excerpt,
                'reading_time': self._reading_time_minutes(p.content),
                'display_date': self._format_display_date(p.created_at),
                'short_tags': sorted(p.tags, key=len)[:2],
            })
        return result

    def _generate_static_pages(self, posts: List[BlogPost] = None):
        config = self.blog_system.config
        pages = {
            'about': ('about', {
                'topics': config.get('content_topics', [][:]),
                'posts': posts or [],
            }),
            'contact': ('contact', {}),
            'privacy-policy': ('privacy_policy', {
                'current_date': datetime.now().strftime('%B %d, %Y')}),
            'terms-of-service': ('terms_of_service', {
                'current_date': datetime.now().strftime('%B %d, %Y')}),
        }
        for dir_name, (template_name, extra_context) in pages.items():
            page_dir = Path("./docs") / dir_name
            page_dir.mkdir(exist_ok=True)
            context = {
                'site_name': config.get('site_name', 'Tech Blog'),
                'base_path': config.get('base_path', ''),
                'base_url': config.get('base_url', ''),
                'current_year': datetime.now().year,
                'global_meta_tags': self.seo.generate_global_meta_tags(),
                **extra_context
            }
            html = self.templates[template_name].render(**context)
            output_file = page_dir / "index.html"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html)
        print("Generated static pages: about, contact, privacy, terms")

    def _generate_author_page(self, posts: List[BlogPost]):
        config = self.blog_system.config
        base_url = config.get('base_url', '')
        base_path = config.get('base_path', '')
        site_name = config.get('site_name', 'Tech Blog')

        author_dir = Path("./docs/author/kubai-kevin")
        author_dir.mkdir(parents=True, exist_ok=True)

        posts_html = ""
        if posts:
            items = "\n".join(
                f'<li><a href="{base_path}/{p.slug}/">{p.title}</a> '
                f'<span style="color:#999;font-size:0.85rem">— {p.created_at[:10]}</span></li>'
                for p in posts[:20]
            )
            posts_html = f"<h2>Recent Articles</h2>\n<ul style='list-style:none;padding:0;'>\n{items}\n</ul>"

        html = AUTHOR_PAGE_TEMPLATE.format(
            site_name=site_name,
            base_url=base_url,
            base_path=base_path,
            year=datetime.now().year,
            posts_html=posts_html
        )
        with open(author_dir / "index.html", 'w', encoding='utf-8') as f:
            f.write(html)
        print("Generated author page (/author/kubai-kevin/)")

    def _generate_tag_pages(self, posts: List[BlogPost]):
        config = self.blog_system.config
        base_url = config.get('base_url', '')
        base_path = config.get('base_path', '')
        site_name = config.get('site_name', 'Tech Blog')
        current_year = datetime.now().year

        tag_map: Dict[str, List[BlogPost]] = {}
        for post in posts:
            for tag in post.tags:
                clean = tag.strip().lower()
                if not clean or len(clean) < 2:
                    continue
                tag_map.setdefault(clean, []).append(post)

        qualifying = {t: ps for t, ps in tag_map.items() if len(ps) >= 2}
        if not qualifying:
            return

        tags_dir = Path("./docs/tag")
        tags_dir.mkdir(exist_ok=True)

        for tag, tag_posts in qualifying.items():
            tag_slug = tag.replace(' ', '-')
            tag_dir = tags_dir / tag_slug
            tag_dir.mkdir(exist_ok=True)

            robots_directive = "index, follow" if len(
                tag_posts) >= 5 else "noindex, follow"

            posts_data = []
            for p in sorted(tag_posts, key=lambda x: x.created_at, reverse=True):
                d = p.to_dict()
                d['display_date'] = self._format_display_date(p.created_at)
                d['short_tags'] = sorted(p.tags, key=len)[:3]
                d['reading_time'] = self._reading_time_minutes(p.content)
                d['meta_description'] = _safe_excerpt(
                    p.meta_description, p.content, p.title)
                posts_data.append(d)

            tag_title = tag.title()
            og_image = f"{base_url}/static/og-default.png"

            # FIX: tag pages previously used a generic, boilerplate meta
            # description ("All articles tagged X... Practical guides for
            # developers.") that was identical across every tag and provided
            # no SEO or user-orientation value. We now build a description
            # from a real topic definition plus the tag's top 3 article
            # titles, so each tag page has unique, descriptive content.
            top_titles = [p['title'] for p in posts_data[:3]]
            tag_definition = _tag_definition(tag)
            tag_meta_description_plain = _generate_tag_meta_description(
                tag_title, top_titles, tag_definition)
            tag_meta_description = _html_stdlib.escape(
                tag_meta_description_plain, quote=True)
            tag_jsonld_description = json.dumps(tag_meta_description_plain)

            # FIX BUG-10: consent.js was loaded with `defer` at the bottom of
            # <body> on tag pages, causing it to fire AFTER the AdSense snippet
            # in <head>. Consent Mode v2 requires the default state to be pushed
            # to dataLayer BEFORE any Google tag loads.
            # Moved to a synchronous <script> as the first element of <head>.
            html = f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{tag_title} Articles — {site_name}</title>
  <meta name="description" content="{tag_meta_description}">
  <meta name="robots" content="{robots_directive}">
  <meta name="base-path" content="{base_path}">
  <link rel="canonical" href="{base_url}/tag/{tag_slug}/">
  <script>window.dataLayer=window.dataLayer||[];function gtag(){{window.dataLayer.push(arguments);}}window.gtag=window.gtag||gtag;(function(){{var mm=document.cookie.match(/(?:^|;) *cookie_consent_v1=([^;]*)/);var g=mm&&decodeURIComponent(mm[1])==='accepted';var s=g?'granted':'denied';gtag('consent','default',{{ad_storage:s,ad_user_data:s,ad_personalization:s,analytics_storage:s,functionality_storage:s,personalization_storage:s,wait_for_update:g?0:500}});}})();</script>
  <script src="{base_path}/static/consent.js" defer></script>
  <link rel="stylesheet" href="{base_path}/static/style.css">
  <meta property="og:type" content="website">
  <meta property="og:title" content="{tag_title} Articles — {site_name}">
  <meta property="og:description" content="{tag_meta_description}">
  <meta property="og:url" content="{base_url}/tag/{tag_slug}/">
  <meta property="og:image" content="{og_image}">
  <script type="application/ld+json">
  {{"@context":"https://schema.org","@type":"CollectionPage",
    "name":"{tag_title} Articles","url":"{base_url}/tag/{tag_slug}/",
    "description":{tag_jsonld_description}}}
  </script>
</head>
<body>
  <header><div class="container">
    <h1><a href="{base_path}/">{site_name}</a></h1>
    <nav>
      <a href="{base_path}/">Home</a>
      <a href="{base_path}/about/">About</a>
      <a href="{base_path}/contact/">Contact</a>
    </nav>
  </div></header>
  <main class="container">
    <nav class="breadcrumb" aria-label="Breadcrumb">
      <a href="{base_path}/">Home</a> <span aria-hidden="true">›</span>
      <a href="{base_path}/tag/">Topics</a> <span aria-hidden="true">›</span>
      <span aria-current="page">{tag_title}</span>
    </nav>
    <h2>{len(tag_posts)} article{"s" if len(tag_posts) != 1 else ""} tagged <em>{tag_title}</em></h2>
    <div class="post-grid">
''' + ''.join(f'''
      <a class="post-card" href="{base_path}/{p["slug"]}/">
        <h3>{p["title"]}</h3>
        <p class="post-excerpt">{p["meta_description"]}</p>
        <p class="post-reading-time">{p["reading_time"]} min read · {p["display_date"]}</p>
      </a>''' for p in posts_data) + f'''
    </div>
  </main>
  <footer><div class="container">
    <p>&copy; {current_year} {site_name}</p>
  </div></footer>
</body>
</html>'''

            with open(tag_dir / "index.html", 'w', encoding='utf-8') as f:
                f.write(html)

        # FIX BUG-10 (cont.): same fix applied to the /tag/ index page.
        all_tags_html = f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>All Topics — {site_name}</title>
  <meta name="description" content="Browse all topics covered on {site_name}.">
  <meta name="robots" content="noindex, follow">
  <meta name="base-path" content="{base_path}">
  <link rel="canonical" href="{base_url}/tag/">
  <script>window.dataLayer=window.dataLayer||[];function gtag(){{window.dataLayer.push(arguments);}}window.gtag=window.gtag||gtag;(function(){{var mm=document.cookie.match(/(?:^|;) *cookie_consent_v1=([^;]*)/);var g=mm&&decodeURIComponent(mm[1])==='accepted';var s=g?'granted':'denied';gtag('consent','default',{{ad_storage:s,ad_user_data:s,ad_personalization:s,analytics_storage:s,functionality_storage:s,personalization_storage:s,wait_for_update:g?0:500}});}})();</script>
  <script src="{base_path}/static/consent.js" defer></script>
  <link rel="stylesheet" href="{base_path}/static/style.css">
</head>
<body>
  <header><div class="container">
    <h1><a href="{base_path}/">{site_name}</a></h1>
    <nav>
      <a href="{base_path}/">Home</a>
      <a href="{base_path}/about/">About</a>
      <a href="{base_path}/contact/">Contact</a>
    </nav>
  </div></header>
  <main class="container">
    <nav class="breadcrumb" aria-label="Breadcrumb">
      <a href="{base_path}/">Home</a> <span aria-hidden="true">›</span>
      <span aria-current="page">Topics</span>
    </nav>
    <h2>All Topics</h2>
    <p>{len(qualifying)} topics across {len(posts)} articles</p>
    <div style="display:flex;flex-wrap:wrap;gap:0.75rem;margin-top:1.5rem;">
''' + ''.join(
            f'<a href="{base_path}/tag/{t.replace(" ", "-")}/" '
            f'style="background:#f0f4ff;border:1px solid #6366f1;border-radius:20px;'
            f'padding:0.4rem 1rem;text-decoration:none;color:#333;font-size:0.9rem;">'
            f'{t.title()} ({len(ps)})</a>'
            for t, ps in sorted(qualifying.items(), key=lambda x: -len(x[1]))
        ) + f'''
    </div>
  </main>
  <footer><div class="container">
    <p>&copy; {current_year} {site_name}</p>
  </div></footer>
</body>
</html>'''

        with open(tags_dir / "index.html", 'w', encoding='utf-8') as f:
            f.write(all_tags_html)

        indexed_count = sum(1 for ps in qualifying.values() if len(ps) >= 5)
        noindex_count = sum(1 for ps in qualifying.values() if len(ps) < 5)
        print(
            f"Generated {len(qualifying)} tag pages + /tag/ index "
            f"({indexed_count} indexed, {noindex_count} noindexed as thin)"
        )

    def _generate_rss_feed(self, posts: List[BlogPost]):
        config = self.blog_system.config
        base_url = config.get('base_url', '')
        rss_items = []
        for post in posts[:20]:
            desc = _safe_excerpt(post.meta_description,
                                 post.content, post.title)
            item = f"""    <item>
      <title>{self._escape_xml(post.title)}</title>
      <link>{base_url}/{post.slug}/</link>
      <description>{self._escape_xml(desc)}</description>
      <pubDate>{self._format_rss_date(post.created_at)}</pubDate>
      <guid>{base_url}/{post.slug}/</guid>
    </item>"""
            rss_items.append(item)
        rss_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>{config.get('site_name', 'Tech Blog')}</title>
    <link>{base_url}</link>
    <description>{config.get('site_description', '')}</description>
    <language>en-us</language>
    <lastBuildDate>{self._format_rss_date(datetime.now().isoformat())}</lastBuildDate>
{chr(10).join(rss_items)}
  </channel>
</rss>"""
        with open("./docs/rss.xml", 'w', encoding='utf-8') as f:
            f.write(rss_content)
        print("Generated RSS feed")

    def _generate_sitemap(self, posts: List[BlogPost]):
        config = self.blog_system.config
        base_url = config.get('base_url', '')
        today = datetime.now().strftime('%Y-%m-%d')
        urls = [
            f'<url><loc>{base_url}/</loc><lastmod>{today}</lastmod><changefreq>daily</changefreq><priority>1.0</priority></url>',
            f'<url><loc>{base_url}/about/</loc><lastmod>{today}</lastmod><changefreq>monthly</changefreq><priority>0.8</priority></url>',
            f'<url><loc>{base_url}/contact/</loc><lastmod>{today}</lastmod><changefreq>monthly</changefreq><priority>0.7</priority></url>',
            f'<url><loc>{base_url}/privacy-policy/</loc><lastmod>{today}</lastmod><changefreq>yearly</changefreq><priority>0.5</priority></url>',
            f'<url><loc>{base_url}/terms-of-service/</loc><lastmod>{today}</lastmod><changefreq>yearly</changefreq><priority>0.5</priority></url>',
            f'<url><loc>{base_url}/tag/</loc><lastmod>{today}</lastmod><changefreq>weekly</changefreq><priority>0.6</priority></url>',
            f'<url><loc>{base_url}/dmca/</loc><lastmod>{today}</lastmod><changefreq>yearly</changefreq><priority>0.4</priority></url>',
            f'<url><loc>{base_url}/ai-content-policy/</loc><lastmod>{today}</lastmod><changefreq>yearly</changefreq><priority>0.4</priority></url>',
        ]
        for post in posts:
            last_mod = post.updated_at.split(
                'T')[0] if 'T' in post.updated_at else post.updated_at
            try:
                age_days = (datetime.now() - datetime.fromisoformat(
                    post.created_at.replace('Z', '+00:00').split('+')[0])).days
                priority = "0.9" if age_days < 30 else "0.8" if age_days < 90 else "0.7"
                changefreq = "weekly" if age_days < 30 else "monthly"
            except Exception:
                priority = "0.8"
                changefreq = "monthly"
            urls.append(
                f'<url><loc>{base_url}/{post.slug}/</loc>'
                f'<lastmod>{last_mod}</lastmod>'
                f'<changefreq>{changefreq}</changefreq>'
                f'<priority>{priority}</priority></url>'
            )

        # FIX BUG-9: The original used chr(10).join(urls) inside an f-string
        # that opened with "  " indentation. This caused the first <url> to
        # have no leading whitespace while subsequent ones did, producing
        # inconsistent indentation that some XML validators reject.
        # Fixed by joining with '\n  ' so every entry is uniformly indented.
        sitemap = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n  '
            + '\n  '.join(urls)
            + '\n</urlset>'
        )
        with open("./docs/sitemap.xml", 'w', encoding='utf-8') as f:
            f.write(sitemap)
        print("Generated sitemap")

    def _generate_posts_json(self, posts: List[BlogPost]):
        """
        Generates /posts.json — the data source for the homepage JS loader.

        Only the fields needed for the post card are included.  Full content
        is intentionally excluded to keep the file small (was ~50 MB when
        content was included; now ~300 KB for 500 posts).
        """
        posts_data = []
        for p in posts:
            posts_data.append({
                'slug': p.slug,
                'title': p.title,
                'meta_description': _safe_excerpt(p.meta_description, p.content, p.title),
                'tags': p.tags,
                'short_tags': sorted(p.tags, key=len)[:3],
                'reading_time': self._reading_time_minutes(p.content),
                'display_date': self._format_display_date(p.created_at),
                'created_at': p.created_at,
            })
        with open("./docs/posts.json", 'w', encoding='utf-8') as f:
            json.dump(posts_data, f, separators=(',', ':'))
        print(f"Generated posts.json ({len(posts_data)} posts)")

    def _escape_xml(self, text: str) -> str:
        return (text.replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&apos;'))

    def _format_rss_date(self, iso_date: str) -> str:
        try:
            dt = datetime.fromisoformat(iso_date.replace('Z', '+00:00'))
            return dt.strftime('%a, %d %b %Y %H:%M:%S +0000')
        except:
            return datetime.now().strftime('%a, %d %b %Y %H:%M:%S +0000')

    def _load_templates(self) -> Dict[str, Template]:
        return _build_templates()


def _build_templates() -> dict:
    from jinja2 import Environment, BaseLoader

    POST_TMPL = """\
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{{ post.title }} - {{ site_name }}</title>
        <meta name="description" content="{{ post.meta_description }}">
        {% if post.seo_keywords %}<meta name="keywords" content="{{ post.seo_keywords|join(', ') }}">{% endif %}
        <meta name="author" content="Kubai Kevin">
        <meta name="base-path" content="{{ base_path }}">
        <meta property="article:author" content="Kubai Kevin">
        <meta property="article:published_time" content="{{ post.created_at}}">
        <meta property="article:modified_time" content="{{ post.updated_at}}">
        <meta property="article:section" content="Technology">
        <meta property="og:type" content="article">
        <meta property="og:title" content="{{ post.title }}">
        <meta property="og:description" content="{{ post.meta_description }}">
        <meta property="og:url" content="{{ base_url }}/{{ post.slug }}/">
        <meta property="og:site_name" content="{{ site_name }}">
        <meta property="og:locale" content="en_US">
        {%- set og_img_png = base_url + '/static/og/' + post.slug + '.png' %}
        {%- set og_img_fallback = base_url + '/static/icons/icon-512x512.png' %}
        <meta property="og:image" content="{{ og_img_png if post.has_og_image else og_img_fallback }}">
        <meta property="og:image:alt" content="{{ post.title }}">
        <meta property="og:image:width" content="1200">
        <meta property="og:image:height" content="630">
        <meta name="twitter:card" content="summary_large_image">
        <meta name="twitter:title" content="{{ post.title }}">
        <meta name="twitter:description" content="{{ post.meta_description }}">
        <meta name="twitter:image" content="{{ og_img_png if post.has_og_image else og_img_fallback }}">
        <meta name="twitter:site" content="@KubaiKevin">

        <link rel="canonical" href="{{ base_url }}/{{ post.slug }}/">

        <link rel="preconnect" href="https://pagead2.googlesyndication.com">
        <link rel="preconnect" href="https://googleads.g.doubleclick.net">
        <link rel="preconnect" href="https://www.google-analytics.com">

        <script>window.dataLayer=window.dataLayer||[];function gtag(){window.dataLayer.push(arguments);}window.gtag=window.gtag||gtag;(function(){var mm=document.cookie.match(/(?:^|;) *cookie_consent_v1=([^;]*)/);var g=mm&&decodeURIComponent(mm[1])==='accepted';var s=g?'granted':'denied';gtag('consent','default',{ad_storage:s,ad_user_data:s,ad_personalization:s,analytics_storage:s,functionality_storage:s,personalization_storage:s,wait_for_update:g?0:500});})();</script>
        <script src="{{ base_path }}/static/consent.js" defer></script>

        {{ global_meta_tags | safe }}
        {{ meta_tags | safe }}
        {{ article_schema | safe }}

        <link rel="stylesheet" href="{{ base_path }}/static/style.css">
        <link rel="stylesheet" href="{{ base_path }}/static/enhanced-blog-post-styles.css">
        <script defer src="{{ base_path }}/static/code_runner.js"></script>
        <link rel="manifest" href="{{ base_path }}/manifest.json">
        <meta name="theme-color" content="#6366f1">
        <meta name="mobile-web-app-capable" content="yes">
        <meta name="apple-mobile-web-app-capable" content="yes">
        <meta name="apple-mobile-web-app-status-bar-style" content="default">
        <meta name="apple-mobile-web-app-title" content="{{ site_name }}">
        <link rel="apple-touch-icon" href="{{ base_path }}/static/icons/icon-192x192.png">
        <style>
            #reading-progress {
                position: fixed; top: 0; left: 0; height: 3px; width: 0%;
                background: linear-gradient(90deg, #667eea, #764ba2);
                z-index: 9999; transition: width 0.1s linear;
            }
            .post-content img { max-width: 100%; height: auto; border-radius: 6px; }
            .post-content table { display: block; overflow-x: auto; -webkit-overflow-scrolling: touch; }
            .ad-inline,
            .ad-header,
            .ad-footer,
            .ad-middle {
                min-height: 0;
                overflow: hidden;
            }
            .ad-inline:not(:empty),
            .ad-header:not(:empty),
            .ad-footer:not(:empty),
            .ad-middle:not(:empty) {
                margin: 2rem 0;
                text-align: center;
            }
            .ad-inline ins[data-ad-status="unfilled"],
            .ad-middle ins[data-ad-status="unfilled"],
            .ad-footer ins[data-ad-status="unfilled"] {
                display: none !important;
            }
        </style>
    </head>
<body>
    <div id="reading-progress" role="progressbar" aria-label="Reading progress"></div>
    <header>
        <div class="container">
            <h1><a href="{{ base_path }}/">{{ site_name }}</a></h1>
            <nav>
                <a href="{{ base_path }}/">Home</a>
                <a href="{{ base_path }}/about/">About</a>
                <a href="{{ base_path }}/contact/">Contact</a>
                <a href="{{ base_path }}/privacy-policy/">Privacy Policy</a>
                <a href="{{ base_path }}/terms-of-service/">Terms of Service</a>
            </nav>
        </div>
    </header>
    {% if header_ad %}<div class="ad-header" style="min-height:100px">{{ header_ad | safe }}</div>{% endif %}
    <main class="container">
        <nav class="breadcrumb" aria-label="Breadcrumb">
            <a href="{{ base_path }}/">Home</a>
            <span>›</span>
            <span>{{ post.title }}</span>
        </nav>
        <article class="blog-post" itemscope itemtype="https://schema.org/Article">
            <header class="post-header">
                <h1 itemprop="headline">{{ post.title }}</h1>
                {% if post.meta_description %}
                <p class="post-lead">{{ post.meta_description }}</p>
                {% endif %}
                {% if post.tags %}
                <div class="tags">
                    {% for tag in post.tags[:6] %}
                    <a class="tag" href="{{ base_path }}/tag/{{ tag | lower | replace(' ', '-') }}/"
                       style="text-decoration:none;">{{ tag }}</a>
                    {% endfor %}
                </div>
                {% endif %}
            </header>
            <div class="author-block" itemprop="author" itemscope itemtype="https://schema.org/Person">
                <div class="author-avatar" aria-hidden="true">KK</div>
                <div class="author-info">
                    <p class="author-name" itemprop="name">
                        <a href="{{ base_path }}/about/">Kubai Kevin</a>
                    </p>
                    <p class="author-bio">
                        Software developer based in Nairobi, Kenya. Writing about AI,
                        backend systems, and developer tooling based on real production experience.
                        <a href="{{ base_path }}/about/">More about the author →</a>
                    </p>
                    <p class="author-meta" style="margin:0.4rem 0 0;font-size:0.82rem;color:#888;">
                        {{ post.reading_time }} min read
                        &nbsp;·&nbsp; {{ post.word_count }} words
                        {% if post.last_updated_iso %}
                        &nbsp;·&nbsp; Updated {{ post.last_updated_iso }}
                        {% endif %}
                        &nbsp;·&nbsp; <span title="Drafted by AI, topic-selected by the author — see the AI content policy">AI-assisted</span>
                    </p>
                </div>
            </div>
            <div class="post-content" itemprop="articleBody">
                {{ post.content_html | safe }}
                {% if inline_ad %}
                <div class="ad-inline">{{ inline_ad | safe }}</div>
                {% endif %}
                {% if middle_ad %}
                <div class="ad-middle">{{ middle_ad | safe }}</div>
                {% endif %}
            </div>
            {% if post.affiliate_links %}
            <div class="affiliate-disclaimer">
                <p><em>This post contains affiliate links. We may earn a commission if you make a purchase through these links, at no additional cost to you.</em></p>
            </div>
            {% endif %}
            <div class="content-policy-footer" style="
                margin-top: 3rem; padding-top: 1.5rem;
                border-top: 1px solid #e0e0e0; font-size: 0.8rem;
                color: #888; line-height: 1.6;
            ">
                <p>
                    <strong>How this article was made:</strong> This article was drafted with
                    AI assistance and published through an automated pipeline. Before
                    publishing, it passes automated checks for duplicate/near-duplicate
                    content, minimum sourcing (external references to primary docs),
                    keyword-stuffing thresholds, and — where the topic involves code —
                    presence of runnable code samples. It has not been reviewed by a
                    human editor. If you spot an error, <a href="{{ base_path }}/contact/">please let us know</a> —
                    corrections are applied within 48 hours.
                </p>
                {% if post.affiliate_links %}
                <p style="margin-top:0.5rem">
                    <strong>Affiliate disclosure:</strong> Some links in this article may earn
                    the site a small commission at no cost to you.
                    This does not influence which tools or services are recommended.
                </p>
                {% endif %}
            </div>
            {% if related_posts %}
            <section class="related-posts">
                <h2>Related Articles</h2>
                <div class="related-grid">
                    {% for rp in related_posts %}
                    <a class="related-card" href="{{ base_path }}/{{ rp.slug }}/">
                        <h3>{{ rp.title }}</h3>
                        <p>{{ rp.meta_description }}</p>
                        <span class="related-meta">{{ rp.reading_time }} min read · {{ rp.display_date }}</span>
                    </a>
                    {% endfor %}
                </div>
            </section>
            {% endif %}
        </article>
    </main>
    {% if footer_ad %}<div class="ad-footer">{{ footer_ad | safe }}</div>{% endif %}
    <footer>
        <div class="container">
            <p>&copy; {{ current_year }} {{ site_name }}. Written by
               <a href="{{ base_path }}/about/">Kubai Kevin</a> with AI assistance.
               See our <a href="{{ base_path }}/ai-content-policy/">AI content policy</a>.</p>
        </div>
    </footer>
    <script src="{{ base_path }}/static/navigation.js"></script>
    <script defer src="{{ base_path }}/static/pwa.js"></script>
    <script>
    (function(){
        var bar = document.getElementById('reading-progress');
        if (!bar) return;
        function update() {
            var el = document.documentElement;
            var scrolled = el.scrollTop || document.body.scrollTop;
            var total = (el.scrollHeight || document.body.scrollHeight) - el.clientHeight;
            bar.style.width = total > 0 ? (scrolled / total * 100) + '%' : '0%';
        }
        window.addEventListener('scroll', update, { passive: true });
        update();
    })();
    </script>
    <script>
    (function(){
        if (!('loading' in HTMLImageElement.prototype)) return;
        document.querySelectorAll('.post-content img').forEach(function(img){
            if (!img.getAttribute('loading')) img.setAttribute('loading', 'lazy');
        });
    })();
    </script>
</body>
</html>"""

    # FIX BUG-14: INDEX_TMPL previously had zero ad placements anywhere in
    # the page - no header_ad/middle_ad/footer_ad blocks at all - so the
    # homepage (the first page AdSense's reviewer crawls) shipped with the
    # verification meta tag but no actual ins.adsbygoogle units. Added a
    # header ad, an in-feed ad after the 3rd post card, and a footer ad,
    # mirroring the pattern already used in POST_TMPL.
    INDEX_TMPL = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ site_name }}</title>
    <meta name="description" content="{{ site_description }}">
    <meta name="base-path" content="{{ base_path }}">
    <link rel="canonical" href="{{ base_url }}/">
    <meta property="og:type" content="website">
    <meta property="og:title" content="{{ site_name }}">
    <meta property="og:description" content="{{ site_description }}">
    <meta property="og:url" content="{{ base_url }}/">
    <meta property="og:image" content="{{ base_url }}/static/icons/icon-512x512.png">
    <meta name="twitter:card" content="summary">
    <meta name="twitter:site" content="@KubaiKevin">
    <link rel="preconnect" href="https://pagead2.googlesyndication.com">
    <link rel="preconnect" href="https://googleads.g.doubleclick.net">
    <link rel="preconnect" href="https://www.google-analytics.com">
    <script>window.dataLayer=window.dataLayer||[];function gtag(){window.dataLayer.push(arguments);}window.gtag=window.gtag||gtag;(function(){var mm=document.cookie.match(/(?:^|;) *cookie_consent_v1=([^;]*)/);var g=mm&&decodeURIComponent(mm[1])==='accepted';var s=g?'granted':'denied';gtag('consent','default',{ad_storage:s,ad_user_data:s,ad_personalization:s,analytics_storage:s,functionality_storage:s,personalization_storage:s,wait_for_update:g?0:500});})();</script>
    <script src="{{ base_path }}/static/consent.js" defer></script>
    {{ global_meta_tags | safe }}
    {{ homepage_meta_tags | safe }}
    {{ organization_schema | safe }}
    {{ website_schema | safe }}
    <link rel="stylesheet" href="{{ base_path }}/static/style.css">
    <link rel="alternate" type="application/rss+xml" title="{{ site_name }}" href="{{ base_path }}/rss.xml">
    <link rel="manifest" href="{{ base_path }}/manifest.json">
    <meta name="theme-color" content="#6366f1">
    <meta name="mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="default">
    <meta name="apple-mobile-web-app-title" content="{{ site_name }}">
    <link rel="apple-touch-icon" href="{{ base_path }}/static/icons/icon-192x192.png">
    <style>
        .search-container { margin: 0 0 1.5rem; max-width: 420px; }
        .search-wrapper {
            display: flex; align-items: center;
            border: 2px solid #e0e0e0; border-radius: 50px;
            padding: 2px 10px; background: #fff;
            transition: border-color 0.2s; overflow: hidden;
        }
        .search-wrapper:focus-within { border-color: #6366f1; }
        .search-icon { color: #9ca3af; pointer-events: none; flex-shrink: 0; display: flex; align-items: center; margin-right: 8px; }
        .search-input {
            flex: 1; padding: 8px 0;
            border: none !important; outline: none !important; box-shadow: none !important;
            -webkit-appearance: none; appearance: none;
            font-size: 0.95rem; background: transparent; border-radius: 0;
        }
        .search-input::-webkit-search-decoration,
        .search-input::-webkit-search-cancel-button { display: none !important; }
        .clear-search {
            background: none; border: none; cursor: pointer;
            color: #9ca3af; padding: 2px; flex-shrink: 0; display: flex; align-items: center;
        }
        .clear-search:hover { color: #333; }
        .search-results-count { margin-top: 0.4rem; color: #666; font-size: 0.85rem; min-height: 1.2em; }
        .search-highlight { background: #fef08a; border-radius: 2px; padding: 0 1px; }
        .no-results-message { text-align: center; padding: 3rem 1rem; color: #666; }

        .post-card {
            display: flex;
            flex-direction: column;
            min-height: 200px;
        }
        .post-card h3 {
            white-space: normal;
            overflow: hidden;
            display: -webkit-box;
            -webkit-line-clamp: 3;
            -webkit-box-orient: vertical;
            line-height: 1.4;
            margin-bottom: 0.6rem;
        }
        .post-card .post-excerpt {
            flex: 1;
            display: -webkit-box;
            -webkit-line-clamp: 4;
            -webkit-box-orient: vertical;
            overflow: hidden;
            min-height: 3.2rem;
        }
        .post-card .tags {
            overflow: visible;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin-top: 0.5rem;
            flex-shrink: 0;
        }
        .post-card .tag {
            cursor: pointer;
            text-decoration: none;
            color: #fff;
        }
        .post-card .tag:hover { opacity: 0.85; color: #fff; }
        .tag { text-decoration: none; color: #fff; }
        .tag:hover { text-decoration: none; opacity: 0.85; color: #fff; }

        .post-reading-time { font-size: 0.8rem; color: #888; margin-top: 4px; margin-bottom: 2px; flex-shrink: 0; }
        .post-card--entering { opacity: 0; transform: translateY(8px); transition: opacity 0.25s ease, transform 0.25s ease; }
        .loading-spinner { display: flex; align-items: center; justify-content: center; gap: 10px; padding: 1.5rem; }
        .spinner { width: 20px; height: 20px; border: 2px solid #e0e0e0; border-top-color: #6366f1; border-radius: 50%; animation: spin 0.7s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .back-to-top { position: fixed; bottom: 2rem; right: 2rem; width: 40px; height: 40px; border-radius: 50%; background: #6366f1; color: #fff; border: none; cursor: pointer; font-size: 1.2rem; display: flex; align-items: center; justify-content: center; box-shadow: 0 2px 8px rgba(0,0,0,0.2); }
        .post-lead { font-size: 1.05rem; color: #555; line-height: 1.6; margin: 0.25rem 0 1rem; font-style: italic; }
        .editorial-policy-note { background: #f0f4ff; border-left: 3px solid #6366f1; padding: 0.75rem 1rem; margin-bottom: 1.5rem; border-radius: 0 6px 6px 0; font-size: 0.85rem; color: #555; }
        .editorial-policy-note a { color: #6366f1; }
    </style>
</head>
<body>
    <header>
        <div class="container">
            <h1><a href="{{ base_path }}/">{{ site_name }}</a></h1>
            <nav>
                <a href="{{ base_path }}/">Home</a>
                <a href="{{ base_path }}/about/">About</a>
                <a href="{{ base_path }}/contact/">Contact</a>
                <a href="{{ base_path }}/privacy-policy/">Privacy Policy</a>
                <a href="{{ base_path }}/terms-of-service/">Terms of Service</a>
            </nav>
        </div>
    </header>
    {% if header_ad %}<div class="ad-header" style="min-height:100px">{{ header_ad | safe }}</div>{% endif %}
    <main class="container">
        <div class="hero">
            <h2>Real Systems. Real Failures. Real Fixes.</h2>
            <p>Practical backend engineering, AI tooling, and developer career advice — no theory, just what's worked in real production systems.</p>
        </div>

        <div class="editorial-policy-note">
            Articles are written by <a href="{{ base_path }}/about/">Kubai Kevin</a>, a software developer
            with 10+ years of production experience, drafted with AI assistance as part of an automated
            publishing pipeline. Not every post gets individual line-by-line review before it goes live.
            <a href="{{ base_path }}/about/#editorial">See how articles are actually produced →</a>
        </div>

        <div class="search-container">
            <div class="search-wrapper">
                <svg class="search-icon" width="18" height="18" viewBox="0 0 20 20" fill="none">
                    <path d="M9 17A8 8 0 1 0 9 1a8 8 0 0 0 0 16zM19 19l-4.35-4.35"
                          stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                <input type="text" id="search-input" class="search-input"
                    placeholder="Search posts..."
                    autocomplete="off" autocorrect="off" autocapitalize="off"
                    spellcheck="false" data-form-type="other">
                <button id="clear-search" class="clear-search" style="display:none;" aria-label="Clear search">
                    <svg width="16" height="16" viewBox="0 0 20 20" fill="none">
                        <path d="M15 5L5 15M5 5l10 10" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                    </svg>
                </button>
            </div>
            <div id="search-results-count" class="search-results-count"></div>
        </div>

        <section class="recent-posts">
            <h2>Latest Posts</h2>
            {% if posts %}
            <div id="posts-container" class="post-grid">
                {% for post in posts  %}
                <a class="post-card" href="{{ base_path }}/{{ post.slug }}/"
                   data-title="{{ post.title | e }}"
                   data-description="{{ post.meta_description | e }}"
                   data-tags="{{ post.tags | join(',') | e }}">
                    <h3>{{ post.title }}</h3>
                    <p class="post-excerpt">{{ post.meta_description }}</p>
                    {% if post.reading_time %}
                    <p class="post-reading-time">{{ post.reading_time }} min read</p>
                    {% endif %}
                    {% if post.tags %}
                    <div class="tags">
                        {% for tag in post.short_tags %}
                        <span class="tag"
                              data-tag-href="{{ base_path }}/tag/{{ tag | lower | replace(' ', '-') }}/">{{ tag }}</span>
                        {% endfor %}
                    </div>
                    {% endif %}
                </a>
                {% if middle_ad and loop.index == 3 %}
                <div class="ad-middle" style="grid-column:1/-1;">{{ middle_ad | safe }}</div>
                {% endif %}
                {% endfor %}
            </div>

            <div id="loading-spinner" class="loading-spinner" style="display:none;">
                <div class="spinner"></div>
                <p>Loading more posts...</p>
            </div>

            <div id="scroll-sentinel" style="height:1px;"></div>

            {% else %}
            <p>No posts yet. Check back soon!</p>
            {% endif %}
        </section>
    </main>
    {% if footer_ad %}<div class="ad-footer">{{ footer_ad | safe }}</div>{% endif %}

    <button id="back-to-top" class="back-to-top" style="display:none;" aria-label="Back to top"><span>&#8593;</span></button>

    <footer>
        <div class="container">
            <p>&copy; {{ current_year }} {{ site_name }}</p>
            <div class="social-links">
                {% for platform, url in social_links.items() %}
                <a href="{{ url }}" target="_blank" rel="noopener">{{ platform|title }}</a>
                {% endfor %}
            </div>
        </div>
    </footer>

    <script src="{{ base_path }}/static/navigation.js"></script>
    <script>
    (function () {
        'use strict';

        var searchInput    = document.getElementById('search-input');
        var clearBtn       = document.getElementById('clear-search');
        var resultsCount   = document.getElementById('search-results-count');
        var postsContainer = document.getElementById('posts-container');
        var loadingSpinner = document.getElementById('loading-spinner');
        var sentinel       = document.getElementById('scroll-sentinel');
        var backToTopBtn   = document.getElementById('back-to-top');

        var PAGE_SIZE = {{ posts_per_page }};
        var BASE_PATH = '{{ base_path }}';

        var fullPosts   = [];
        var loadedCount = postsContainer
            ? postsContainer.querySelectorAll('a.post-card').length
            : 0;
        var jsonReady   = false;
        var isLoading   = false;
        var searchMode  = false;
        var observer    = null;

        document.addEventListener('click', function (e) {
            var el = e.target;
            if (el && el.tagName === 'SPAN' && el.dataset.tagHref) {
                e.preventDefault();
                e.stopPropagation();
                window.location.href = el.dataset.tagHref;
            }
        }, true);

        function startObserver() {
            if (!sentinel || !window.IntersectionObserver || observer) return;
            observer = new IntersectionObserver(function (entries) {
                if (entries[0].isIntersecting) onSentinelVisible();
            }, { rootMargin: '0px 0px 400px 0px' });
            observer.observe(sentinel);
        }

        function stopObserver() {
            if (observer) { observer.disconnect(); observer = null; }
        }

        function onSentinelVisible() {
            if (!jsonReady || isLoading || searchMode) return;
            if (loadedCount >= fullPosts.length) { stopObserver(); return; }
            loadNextPage();
        }

        function buildCard(post) {
            var a       = document.createElement('a');
            a.className = 'post-card';
            a.href      = BASE_PATH + '/' + post.slug + '/';

            var h3         = document.createElement('h3');
            h3.textContent = post.title;
            a.appendChild(h3);

            var excerpt = (post.meta_description || '').trim();
            if (!excerpt && post.content) {
                excerpt = post.content.replace(/[#*`>\\[\\]]/g, '').trim().slice(0, 155);
                if (excerpt.length === 155) excerpt = excerpt.slice(0, excerpt.lastIndexOf(' ')) + '\\u2026';
            }
            if (excerpt) {
                var p         = document.createElement('p');
                p.className   = 'post-excerpt';
                p.textContent = excerpt;
                a.appendChild(p);
            }

            if (post.reading_time) {
                var rt         = document.createElement('p');
                rt.className   = 'post-reading-time';
                rt.textContent = post.reading_time + ' min read';
                a.appendChild(rt);
            }

            var tags = (post.tags || []).filter(function(t) { return t && t.trim(); });
            if (tags.length) {
                var div       = document.createElement('div');
                div.className = 'tags';
                tags.slice().sort(function (x, y) { return x.length - y.length; })
                    .slice(0, 3)
                    .forEach(function (t) {
                        var sp              = document.createElement('span');
                        sp.className        = 'tag';
                        sp.textContent      = t;
                        sp.dataset.tagHref  = BASE_PATH + '/tag/' + t.toLowerCase().replace(/\\s+/g, '-') + '/';
                        div.appendChild(sp);
                    });
                a.appendChild(div);
            }
            return a;
        }

        function loadNextPage() {
            if (isLoading) return;
            isLoading = true;
            var batchStart = loadedCount;
            if (loadingSpinner) loadingSpinner.style.display = 'flex';
            setTimeout(function () {
                var slice    = fullPosts.slice(batchStart, batchStart + PAGE_SIZE);
                var fragment = document.createDocumentFragment();
                var newCards = [];
                slice.forEach(function (post) {
                    var card = buildCard(post);
                    card.classList.add('post-card--entering');
                    fragment.appendChild(card);
                    newCards.push(card);
                });
                postsContainer.appendChild(fragment);
                loadedCount = batchStart + slice.length;
                requestAnimationFrame(function () {
                    requestAnimationFrame(function () {
                        newCards.forEach(function (el, i) {
                            setTimeout(function () { el.classList.remove('post-card--entering'); }, i * 60);
                        });
                    });
                });
                if (loadingSpinner) loadingSpinner.style.display = 'none';
                isLoading = false;
                if (loadedCount >= fullPosts.length) stopObserver();
            }, 250);
        }

        fetch(BASE_PATH + '/posts.json')
            .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
            .then(function (posts) {
                fullPosts = posts;
                jsonReady = true;
                if (loadedCount < fullPosts.length) {
                    requestAnimationFrame(function () { startObserver(); });
                }
                if (searchMode && searchInput && searchInput.value.trim()) {
                    runSearch(searchInput.value.trim());
                }
            })
            .catch(function (err) { console.warn('posts.json fetch failed:', err); jsonReady = false; });

        function readText(el, selector) {
            if (!el) return '';
            var child = el.querySelector(selector);
            return child ? (child.textContent || '') : '';
        }

        var domIndex = [];
        if (postsContainer) {
            postsContainer.querySelectorAll('a.post-card').forEach(function (el) {
                domIndex.push({
                    element:     el,
                    title:       (el.dataset.title       || readText(el, 'h3')).toLowerCase(),
                    description: (el.dataset.description || readText(el, '.post-excerpt')).toLowerCase(),
                    tags:        (el.dataset.tags        || '').toLowerCase()
                });
            });
        }

        function highlightText(text, query) {
            if (!query) return text;
            var lower  = text.toLowerCase();
            var qLower = query.toLowerCase();
            var result = '';
            var pos    = 0;
            var idx;
            while ((idx = lower.indexOf(qLower, pos)) !== -1) {
                result += escapeHtml(text.slice(pos, idx))
                       +  '<mark class="search-highlight">'
                       +  escapeHtml(text.slice(idx, idx + query.length))
                       +  '</mark>';
                pos = idx + query.length;
            }
            result += escapeHtml(text.slice(pos));
            return result;
        }

        function escapeHtml(s) {
            return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
                    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
        }

        function runSearch(rawQuery) {
            if (!postsContainer) return;
            var query = rawQuery.toLowerCase().trim();
            if (!query) { clearSearch(); return; }
            searchMode = true;
            stopObserver();
            var matched;
            if (jsonReady && fullPosts.length) {
                matched = fullPosts.filter(function (p) {
                    return (p.title            || '').toLowerCase().indexOf(query) !== -1 ||
                           (p.meta_description || '').toLowerCase().indexOf(query) !== -1 ||
                           (p.tags             || []).some(function (t) { return t.toLowerCase().indexOf(query) !== -1; });
                });
                postsContainer.innerHTML = '';
                matched.forEach(function (post) {
                    var card    = buildCard(post);
                    var h3      = card.querySelector('h3');
                    var excerpt = card.querySelector('.post-excerpt');
                    if (h3)      h3.innerHTML      = highlightText(h3.textContent,      rawQuery);
                    if (excerpt) excerpt.innerHTML = highlightText(excerpt.textContent, rawQuery);
                    postsContainer.appendChild(card);
                });
            } else {
                matched = [];
                domIndex.forEach(function (item) {
                    var hit = item.title.indexOf(query) !== -1 ||
                              item.description.indexOf(query) !== -1 ||
                              item.tags.indexOf(query) !== -1;
                    item.element.style.display = hit ? '' : 'none';
                    if (hit) matched.push(item);
                });
                matched.forEach(function (item) {
                    var h3      = item.element.querySelector('h3');
                    var excerpt = item.element.querySelector('.post-excerpt');
                    if (h3)      h3.innerHTML      = highlightText(h3.dataset.plain      || h3.textContent,      rawQuery);
                    if (excerpt) excerpt.innerHTML = highlightText(excerpt.dataset.plain || excerpt.textContent, rawQuery);
                });
            }
            var n = matched.length;
            if (resultsCount) {
                resultsCount.textContent = n === 0
                    ? 'No results for "' + rawQuery + '"'
                    : n === 1 ? '1 post found' : n + ' posts found';
            }
            var old = document.getElementById('no-results-msg');
            if (old) old.remove();
            if (n === 0) {
                var msg       = document.createElement('div');
                msg.id        = 'no-results-msg';
                msg.className = 'no-results-message';
                msg.innerHTML = '<p>No posts matched <strong>' + rawQuery + '</strong>. Try different keywords.</p>';
                postsContainer.insertAdjacentElement('afterend', msg);
            }
        }

        function clearSearch() {
            searchMode = false;
            stopObserver();
            domIndex.forEach(function (item) {
                item.element.style.display = '';
                var h3      = item.element.querySelector('h3');
                var excerpt = item.element.querySelector('.post-excerpt');
                if (h3)      h3.textContent      = h3.dataset.plain      || item.title;
                if (excerpt) excerpt.textContent = excerpt.dataset.plain || item.description;
            });
            if (jsonReady && fullPosts.length) {
                postsContainer.innerHTML = '';
                var first = fullPosts.slice(0, PAGE_SIZE);
                first.forEach(function (post) { postsContainer.appendChild(buildCard(post)); });
                loadedCount = first.length;
            } else {
                postsContainer.querySelectorAll('a.post-card').forEach(function (el) { el.style.display = ''; });
            }
            var old = document.getElementById('no-results-msg');
            if (old) old.remove();
            if (resultsCount) resultsCount.textContent = '';
            if (jsonReady && loadedCount < fullPosts.length) {
                requestAnimationFrame(function () { startObserver(); });
            }
        }

        if (postsContainer) {
            postsContainer.querySelectorAll('a.post-card').forEach(function (el) {
                var h3      = el.querySelector('h3');
                var excerpt = el.querySelector('.post-excerpt');
                if (h3      && !h3.dataset.plain)      h3.dataset.plain      = h3.textContent;
                if (excerpt && !excerpt.dataset.plain) excerpt.dataset.plain = excerpt.textContent;
            });
        }

        if (searchInput) {
            searchInput.addEventListener('input', function () {
                var q = this.value.trim();
                if (clearBtn) clearBtn.style.display = q ? 'flex' : 'none';
                if (q) { runSearch(q); } else { clearSearch(); }
            });
            searchInput.addEventListener('keydown', function (e) {
                if (e.key === 'Escape') { this.value = ''; if (clearBtn) clearBtn.style.display = 'none'; clearSearch(); }
            });
        }

        if (clearBtn) {
            clearBtn.addEventListener('click', function () {
                if (searchInput) searchInput.value = '';
                this.style.display = 'none';
                clearSearch();
                if (searchInput) searchInput.focus();
            });
        }

        window.addEventListener('scroll', function () {
            if (backToTopBtn)
                backToTopBtn.style.display = window.pageYOffset > 300 ? 'flex' : 'none';
        }, { passive: true });

        if (backToTopBtn) {
            backToTopBtn.addEventListener('click', function () { window.scrollTo({ top: 0, behavior: 'smooth' }); });
        }

    }());
    </script>
    <script defer src="{{ base_path }}/static/pwa.js"></script>
</body>
</html>"""

    # FIX BUG-8: ABOUT_TMPL had consent.js loaded with `defer` at the BOTTOM
    # of <body>. POST_TMPL and INDEX_TMPL correctly load it as the first
    # synchronous <script> in <head>. ABOUT_TMPL was the outlier, meaning the
    # AdSense snippet (inside global_meta_tags) fired before consent defaults
    # were pushed to dataLayer — a Consent Mode v2 violation.
    # Fix: move consent.js to a synchronous load as the FIRST script in <head>,
    # before global_meta_tags, and remove the deferred bottom-of-body tag.
    ABOUT_TMPL = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>About Kubai Kevin — {{ site_name }}</title>
    <meta name="description" content="Kubai Kevin is a software developer based in Nairobi, Kenya with 10+ years building production Python and Node.js backends in fintech. He writes about AI, backend systems, and developer careers from direct production experience.">
    <meta name="base-path" content="{{ base_path }}">
    <link rel="canonical" href="{{ base_url }}/about/">
    <meta property="og:type" content="profile">
    <meta property="og:title" content="About Kubai Kevin — {{ site_name }}">
    <meta property="og:description" content="Software developer in Nairobi writing about AI, backend systems, and developer careers from 10+ years of production experience.">
    <meta property="og:url" content="{{ base_url }}/about/">
    <meta property="profile:first_name" content="Kevin">
    <meta property="profile:last_name" content="Kubai">
    <meta name="author" content="Kubai Kevin">
    <script>window.dataLayer=window.dataLayer||[];function gtag(){window.dataLayer.push(arguments);}window.gtag=window.gtag||gtag;(function(){var mm=document.cookie.match(/(?:^|;) *cookie_consent_v1=([^;]*)/);var g=mm&&decodeURIComponent(mm[1])==='accepted';var s=g?'granted':'denied';gtag('consent','default',{ad_storage:s,ad_user_data:s,ad_personalization:s,analytics_storage:s,functionality_storage:s,personalization_storage:s,wait_for_update:g?0:500});})();</script>
    <script src="{{ base_path }}/static/consent.js" defer></script>
    {{ global_meta_tags | safe }}
    <script type="application/ld+json">
    {
      "@context": "https://schema.org",
      "@type": "ProfilePage",
      "@id": "{{ base_url }}/about/#profilepage",
      "url": "{{ base_url }}/about/",
      "name": "About Kubai Kevin",
      "inLanguage": "en-US",
      "primaryImageOfPage": {
        "@type": "ImageObject",
        "url": "{{ base_url }}/static/icons/icon-192x192.png"
      },
      "mainEntity": {
        "@type": "Person",
        "@id": "{{ base_url }}/about/#author",
        "name": "Kubai Kevin",
        "givenName": "Kevin",
        "familyName": "Kubai",
        "jobTitle": "Software Developer",
        "description": "Software developer based in Nairobi, Kenya with 10+ years experience in Python, Node.js, and AWS. Specialises in fintech backends and AI integration.",
        "url": "{{ base_url }}/about/",
        "email": "aiblogauto@gmail.com",
        "address": {
          "@type": "PostalAddress",
          "addressLocality": "Nairobi",
          "addressCountry": "KE"
        },
        "sameAs": [
          "https://www.linkedin.com/in/kevin-kubai-22b61b37/",
          "https://twitter.com/KubaiKevin",
          "https://github.com/kubaik"
        ],
        "knowsAbout": [
          "Python", "Node.js", "TypeScript", "AWS Lambda", "PostgreSQL",
          "Redis", "Machine Learning", "LLMs", "API Design",
          "Fintech Systems", "Backend Engineering", "Android Development"
        ],
        "alumniOf": {
          "@type": "Organization",
          "name": "Self-taught, supplemented with industry certifications"
        },
        "worksFor": {
          "@type": "Organization",
          "name": "{{ site_name }}",
          "url": "{{ base_url }}/"
        }
      }
    }
    </script>
    <link rel="stylesheet" href="{{ base_path }}/static/style.css">
    <link rel="manifest" href="{{ base_path }}/manifest.json">
    <meta name="theme-color" content="#6366f1">
    <link rel="apple-touch-icon" href="{{ base_path }}/static/icons/icon-192x192.png">
    <style>
        .about-hero {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; padding: 3rem 2rem; border-radius: 12px;
            margin-bottom: 2rem; text-align: center;
        }
        .about-hero h1 { font-size: 2rem; margin-bottom: 0.5rem; color: white; }
        .about-hero p { opacity: 0.9; font-size: 1.05rem; max-width: 560px; margin: 0 auto; }
        .author-profile {
            display: flex; gap: 2rem; align-items: flex-start;
            background: #f8f9fa; padding: 2rem; border-radius: 12px;
            border: 1px solid #e0e0e0; margin-bottom: 2rem;
        }
        @media (max-width: 600px) {
            .author-profile { flex-direction: column; align-items: center; text-align: center; }
        }
        .author-avatar-svg {
            width: 100px; height: 100px; flex-shrink: 0; border-radius: 50%;
            background: linear-gradient(135deg, #667eea, #764ba2);
            display: flex; align-items: center; justify-content: center;
            font-size: 2rem; font-weight: 700; color: white;
        }
        .author-name { font-size: 1.4rem; font-weight: 700; color: #1a1a2e; margin: 0 0 0.25rem; }
        .author-title { color: #6366f1; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.75rem; }
        .author-bio { color: #444; line-height: 1.7; margin-bottom: 1rem; }
        .author-links { display: flex; gap: 0.75rem; flex-wrap: wrap; }
        .author-link {
            display: inline-flex; align-items: center; gap: 0.4rem;
            padding: 0.4rem 1rem; border-radius: 20px; font-size: 0.85rem;
            font-weight: 600; text-decoration: none; border: 2px solid;
            transition: all 0.2s;
        }
        .author-link-linkedin { color: #0077b5; border-color: #0077b5; }
        .author-link-linkedin:hover { background: #0077b5; color: white; }
        .author-link-twitter { color: #1da1f2; border-color: #1da1f2; }
        .author-link-twitter:hover { background: #1da1f2; color: white; }
        .author-link-github { color: #333; border-color: #333; }
        .author-link-github:hover { background: #333; color: white; }
        .section-card {
            background: white; border: 1px solid #e0e0e0; border-radius: 12px;
            padding: 1.75rem; margin-bottom: 1.5rem;
        }
        .section-card h2 {
            font-size: 1.25rem; color: #1a1a2e; margin-top: 0; margin-bottom: 1rem;
            padding-bottom: 0.6rem; border-bottom: 2px solid #f0f0f0;
        }
        .tech-grid {
            display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
            gap: 0.5rem; margin-top: 0.75rem;
        }
        .tech-pill {
            background: #f0f4ff; border: 1px solid #c7d2fe;
            border-radius: 6px; padding: 0.35rem 0.75rem;
            font-size: 0.82rem; color: #3730a3; text-align: center;
        }
        .process-step { display: flex; gap: 1rem; margin-bottom: 1.25rem; align-items: flex-start; }
        .step-num {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white; width: 30px; height: 30px; border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            font-weight: 700; font-size: 0.85rem; flex-shrink: 0; margin-top: 2px;
        }
        .step-body strong { color: #1a1a2e; }
        .step-body p { margin: 0.25rem 0 0; color: #555; font-size: 0.9rem; line-height: 1.6; }
        .stat-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 1rem; margin-top: 1rem; }
        .stat-box { text-align: center; background: #f8f9fa; border-radius: 8px; padding: 1rem; }
        .stat-num { font-size: 1.8rem; font-weight: 700; color: #6366f1; display: block; }
        .stat-label { font-size: 0.8rem; color: #666; margin-top: 0.25rem; }
        .correction-box {
            background: #fff3cd; border-left: 4px solid #ffc107;
            border-radius: 0 8px 8px 0; padding: 1rem 1.25rem; margin-top: 1rem;
        }
        .correction-box p { margin: 0; font-size: 0.9rem; color: #856404; }
        .recent-posts-list { list-style: none; padding: 0; margin: 0; }
        .recent-posts-list li { border-bottom: 1px solid #f0f0f0; }
        .recent-posts-list li:last-child { border-bottom: none; }
        .recent-posts-list a {
            display: block; padding: 0.75rem 0; text-decoration: none;
            color: #333; font-size: 0.9rem; transition: color 0.2s;
        }
        .recent-posts-list a:hover { color: #6366f1; }
        .post-date { font-size: 0.78rem; color: #999; margin-top: 0.2rem; }
    </style>
</head>
<body>
    <header>
        <div class="container">
            <h1><a href="{{ base_path }}/">{{ site_name }}</a></h1>
            <nav>
                <a href="{{ base_path }}/">Home</a>
                <a href="{{ base_path }}/about/">About</a>
                <a href="{{ base_path }}/contact/">Contact</a>
                <a href="{{ base_path }}/privacy-policy/">Privacy Policy</a>
                <a href="{{ base_path }}/terms-of-service/">Terms of Service</a>
            </nav>
        </div>
    </header>

    <main class="container">
        <div class="about-hero">
            <h1>About This Blog</h1>
            <p>Practical writing about software development, AI, and developer careers — based on real production experience, not tutorials.</p>
        </div>

        <article itemscope itemtype="https://schema.org/Person"
                 id="author"
                 itemprop="author">
            <meta itemprop="name" content="Kubai Kevin">
            <meta itemprop="url" content="{{ base_url }}/about/">

            <div class="author-profile">
                <div class="author-avatar-svg" aria-hidden="true">KK</div>
                <div>
                    <h2 class="author-name" itemprop="name">Kubai Kevin</h2>
                    <p class="author-title" itemprop="jobTitle">
                        Software Developer · Nairobi, Kenya
                        <span itemprop="address" itemscope itemtype="https://schema.org/PostalAddress">
                            <meta itemprop="addressLocality" content="Nairobi">
                            <meta itemprop="addressCountry" content="KE">
                        </span>
                    </p>
                    <p class="author-bio" itemprop="description">
                        I'm a software developer with over a decade of experience building production
                        systems in the financial services sector in East Africa. My day-to-day work
                        involves Python and Node.js backends, AWS serverless infrastructure, and
                        integrating AI/LLMs into real workflows. I also build Android applications
                        in Java and Kotlin.
                    </p>
                    <p class="author-bio">
                        I started writing here because the guides I needed when solving production
                        problems didn't exist — or they existed but skipped the parts that matter
                        when things go wrong at 2am. Everything I write comes from something I
                        have personally built, debugged, or deployed.
                    </p>
                    <div class="author-links">
                        <a href="https://www.linkedin.com/in/kevin-kubai-22b61b37/"
                           class="author-link author-link-linkedin"
                           target="_blank" rel="noopener noreferrer"
                           itemprop="sameAs">
                            LinkedIn profile
                        </a>
                        <a href="https://twitter.com/KubaiKevin"
                           class="author-link author-link-twitter"
                           target="_blank" rel="noopener noreferrer"
                           itemprop="sameAs">
                            @KubaiKevin
                        </a>
                        <a href="https://github.com/kubaik"
                           class="author-link author-link-github"
                           target="_blank" rel="noopener noreferrer"
                           itemprop="sameAs">
                            GitHub
                        </a>
                    </div>
                </div>
            </div>

            <div class="section-card">
                <h2>Technologies I work with regularly</h2>
                <p style="color:#555;font-size:0.9rem;margin-bottom:0.5rem">
                    These are tools I use in production systems, not just tutorials I've read.
                </p>
                <div class="tech-grid">
                    <span class="tech-pill">Python 3.11+</span>
                    <span class="tech-pill">Node.js / TypeScript</span>
                    <span class="tech-pill">FastAPI</span>
                    <span class="tech-pill">AWS Lambda</span>
                    <span class="tech-pill">PostgreSQL</span>
                    <span class="tech-pill">Redis</span>
                    <span class="tech-pill">Android (Kotlin)</span>
                    <span class="tech-pill">M-Pesa / Paystack</span>
                    <span class="tech-pill">GitHub Actions</span>
                    <span class="tech-pill">Docker</span>
                    <span class="tech-pill">LLM APIs</span>
                    <span class="tech-pill">RAG pipelines</span>
                </div>
            </div>

            <div class="section-card">
                <h2>By the numbers</h2>
                <div class="stat-row">
                    <div class="stat-box">
                        <span class="stat-num">{{ posts|length }}</span>
                        <div class="stat-label">Articles published</div>
                    </div>
                    <div class="stat-box">
                        <span class="stat-num">10+</span>
                        <div class="stat-label">Years in production</div>
                    </div>
                    <div class="stat-box">
                        <span class="stat-num">2014</span>
                        <div class="stat-label">Started professionally</div>
                    </div>
                    <div class="stat-box">
                        <span class="stat-num">EAT</span>
                        <div class="stat-label">UTC+3 · Nairobi</div>
                    </div>
                </div>
            </div>

            <div class="section-card" id="editorial">
                <h2>Editorial process</h2>
                <p style="color:#555;font-size:0.9rem;margin-bottom:1.25rem">
                    Here's how articles on this site actually get made, and where the human
                    involvement is versus where it isn't:
                </p>

                <div class="process-step">
                    <div class="step-num">1</div>
                    <div class="step-body">
                        <strong>Topic selection</strong>
                        <p>Topics are chosen based on real problems from production fintech
                        systems, patterns I've seen come up repeatedly, and tools I've
                        personally used — not generated from keyword lists.</p>
                    </div>
                </div>

                <div class="process-step">
                    <div class="step-num">2</div>
                    <div class="step-body">
                        <strong>Drafting</strong>
                        <p>Articles are drafted with AI assistance, grounded in official
                        documentation and my own notes from working with these tools and
                        systems. Given the volume published here, not every article gets
                        individual line-by-line review before it goes live — I rely on the
                        correction process below to catch and fix mistakes.</p>
                    </div>
                </div>

                <div class="process-step">
                    <div class="step-num">3</div>
                    <div class="step-body">
                        <strong>Spot checks and quality review</strong>
                        <p>I periodically review published articles, particularly ones getting
                        traffic or reader feedback, and update or remove anything that's wrong,
                        outdated, or doesn't hold up.</p>
                    </div>
                </div>

                <div class="process-step">
                    <div class="step-num">4</div>
                    <div class="step-body">
                        <strong>Ongoing corrections</strong>
                        <p>If you spot an error, tell me and I'll fix it. Technology also
                        changes fast enough that older articles get revisited and updated
                        as tools and best practices move on.</p>
                    </div>
                </div>

                <div class="correction-box">
                    <p>Found a factual error?
                    <a href="{{ base_path }}/contact/" style="color:#856404;font-weight:600;">
                        Email me directly</a> —
                    I review and correct reported errors promptly.</p>
                </div>
            </div>

            {% if posts %}
            <div class="section-card">
                <h2>Recent articles</h2>
                <ul class="recent-posts-list">
                    {% for post in posts[:8] %}
                    <li>
                        <a href="{{ base_path }}/{{ post.slug }}/">
                            {{ post.title }}
                            <div class="post-date">{{ post.created_at[:10] }}</div>
                        </a>
                    </li>
                    {% endfor %}
                </ul>
                {% if posts|length > 8 %}
                <p style="margin-top:1rem;text-align:center">
                    <a href="{{ base_path }}/" style="color:#6366f1;text-decoration:none;font-weight:600;">
                        View all {{ posts|length }} articles →
                    </a>
                </p>
                {% endif %}
            </div>
            {% endif %}

        </article>
    </main>

    <footer>
        <div class="container">
            <p>&copy; {{ current_year }} {{ site_name }} · Written by Kubai Kevin</p>
        </div>
    </footer>
    <script src="{{ base_path }}/static/navigation.js"></script>
    <script defer src="{{ base_path }}/static/pwa.js"></script>
</body>
</html>"""

    # FIX BUG-11: PRIVACY_TMPL used href="../static/style.css" (relative path).
    # Fixed to use base_path variable for deployment-agnostic asset loading.
    PRIVACY_TMPL = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Privacy Policy - {{ site_name }}</title>
    <meta name="description" content="Privacy Policy for {{ site_name }}. Learn what data we collect, how we use it, your rights under GDPR and CCPA, and how to contact us.">
    <meta name="base-path" content="{{ base_path }}">
    <link rel="canonical" href="{{ base_url }}/privacy-policy/">
    <script>window.dataLayer=window.dataLayer||[];function gtag(){window.dataLayer.push(arguments);}window.gtag=window.gtag||gtag;(function(){var mm=document.cookie.match(/(?:^|;) *cookie_consent_v1=([^;]*)/);var g=mm&&decodeURIComponent(mm[1])==='accepted';var s=g?'granted':'denied';gtag('consent','default',{ad_storage:s,ad_user_data:s,ad_personalization:s,analytics_storage:s,functionality_storage:s,personalization_storage:s,wait_for_update:g?0:500});})();</script>
    <script src="{{ base_path }}/static/consent.js" defer></script>
    {{ global_meta_tags | safe }}
    <link rel="stylesheet" href="{{ base_path }}/static/style.css">
    <link rel="manifest" href="{{ base_path }}/manifest.json">
    <link rel="apple-touch-icon" href="{{ base_path }}/static/icons/icon-192x192.png">
    <style>
        .privacy-section{background:#f8f9fa;padding:1.5rem;margin-bottom:1.5rem;border-radius:8px;border-left:4px solid #6366f1}
        .privacy-section h3{color:#333;margin-top:0}
        .privacy-section h4{color:#4f46e5;margin-top:1rem;margin-bottom:0.5rem}
        .important-notice{background:#fff3cd;border-left:4px solid #ffc107;padding:1rem 1.5rem;margin:1.5rem 0;border-radius:4px}
        table{width:100%;border-collapse:collapse;background:white;margin:1rem 0}
        th,td{padding:0.8rem;text-align:left;border-bottom:1px solid #dee2e6}
        th{background:#f8f9fa;font-weight:600}
        .highlight-box{background:linear-gradient(135deg,#667eea,#764ba2);color:white;padding:1.5rem;border-radius:8px;margin:1.5rem 0}
        .highlight-box h3{margin-top:0;color:white}
    </style>
</head>
<body>
    <header><div class="container">
        <h1><a href="{{ base_path }}/">{{ site_name }}</a></h1>
        <nav><a href="{{ base_path }}/">Home</a><a href="{{ base_path }}/about/">About</a><a href="{{ base_path }}/contact/">Contact</a>
        <a href="{{ base_path }}/privacy-policy/">Privacy Policy</a><a href="{{ base_path }}/terms-of-service/">Terms of Service</a></nav>
    </div></header>
    <main class="container">
        <div class="hero"><h2>Privacy Policy</h2><p>How we protect and handle your information</p></div>
        <article class="page-content">
            
            <div class="important-notice">
                <p><strong>Last updated:</strong> {{ current_date }}</p>
                <p><strong>Controller:</strong> {{ site_name }} (individual), Nairobi, Kenya</p>
                <p><strong>Contact:</strong> <a href="mailto:aiblogauto@gmail.com">aiblogauto@gmail.com</a></p>
            </div>

            <div class="privacy-section">
                <h3>1. Introduction</h3>
                <p>This Privacy Policy explains how <strong>{{ site_name }}</strong> ("the Site", "we", "us") collects, uses, and protects your personal data when you visit <a href="{{ base_url }}/">{{ site_name }}</a>.</p>
                <p>We are committed to handling your data transparently and in accordance with applicable law, including the EU General Data Protection Regulation (GDPR) and the California Consumer Privacy Act (CCPA).</p>
            </div>

            <div class="privacy-section">
                <h3>2. What Data We Collect</h3>
                
                <h4>2.1 Data you provide</h4>
                <ul>
                    <li><strong>Contact information:</strong> If you email us, we receive your name, email address, and message content.</li>
                </ul>

                <h4>2.2 Data collected automatically (with your consent)</h4>
                <p>We collect this data <strong>only after you accept cookies</strong> via our cookie banner:</p>
                <table>
                    <thead>
                        <tr><th>Data type</th><th>Source</th><th>Purpose</th></tr>
                    </thead>
                    <tbody>
                        <tr><td>Page views, session duration, referrer</td><td>Google Analytics (GA4)</td><td>Understanding how readers use the site</td></tr>
                        <tr><td>Browser type, device type, approximate location (country-level)</td><td>Google Analytics</td><td>Site performance and content improvement</td></tr>
                        <tr><td>Ad interaction data</td><td>Google AdSense</td><td>Serving relevant advertisements</td></tr>
                    </tbody>
                </table>

                <h4>2.3 Data collected without your consent</h4>
                <ul>
                    <li><strong>Server logs</strong> generated by GitHub Pages (the hosting provider) may include IP addresses. These are retained by GitHub under <a href="https://docs.github.com/en/site-policy/privacy-policies/github-general-privacy-statement" target="_blank" rel="noopener">GitHub's Privacy Policy</a> and are not accessible to us.</li>
                </ul>
            </div>

            <div class="privacy-section">
                <h3>3. Legal Basis for Processing (GDPR)</h3>
                <table>
                    <thead>
                        <tr><th>Processing activity</th><th>Legal basis (Article 6 GDPR)</th></tr>
                    </thead>
                    <tbody>
                        <tr><td>Analytics cookies (GA4)</td><td>Consent (Art. 6(1)(a))</td></tr>
                        <tr><td>Advertising cookies (AdSense)</td><td>Consent (Art. 6(1)(a))</td></tr>
                        <tr><td>Responding to email enquiries</td><td>Legitimate interests (Art. 6(1)(f))</td></tr>
                        <tr><td>Hosting (GitHub Pages server logs)</td><td>Legitimate interests — you browsing requires a server to log requests</td></tr>
                    </tbody>
                </table>
                <p>You may withdraw your consent at any time by clicking "Reject non-essential" in the cookie banner or clearing your browser's cookies.</p>
            </div>

            <div class="privacy-section">
                <h3>4. Cookies</h3>
                <p>We use the following cookie categories:</p>
                <table>
                    <thead>
                        <tr><th>Category</th><th>Examples</th><th>Duration</th><th>Consent required?</th></tr>
                    </thead>
                    <tbody>
                        <tr><td>Strictly necessary</td><td>Cookie consent preference (<code>kk_cookie_consent</code>)</td><td>1 year</td><td>No</td></tr>
                        <tr><td>Analytics</td><td>Google Analytics (<code>_ga</code>, <code>_ga_*</code>)</td><td>Up to 2 years</td><td><strong>Yes</strong></td></tr>
                        <tr><td>Advertising</td><td>Google AdSense (<code>IDE</code>, <code>test_cookie</code>, etc.)</td><td>Up to 1 year</td><td><strong>Yes</strong></td></tr>
                    </tbody>
                </table>
                <p>You can manage or delete cookies at any time via your browser settings. Rejecting non-essential cookies will not affect your ability to read content on this site.</p>
                <p>For more on how Google uses cookies: <a href="https://policies.google.com/technologies/cookies" target="_blank" rel="noopener">policies.google.com/technologies/cookies</a></p>
            </div>

            <div class="privacy-section">
                <h3>5. Third-Party Services</h3>
                <div class="important-notice"><p>We do not sell your personal data to any third party.</p></div>
                <table>
                    <thead>
                        <tr><th>Service</th><th>Purpose</th><th>Privacy Policy</th></tr>
                    </thead>
                    <tbody>
                        <tr><td><strong>Google Analytics (GA4)</strong></td><td>Usage analytics</td><td><a href="https://policies.google.com/privacy" target="_blank" rel="noopener">policies.google.com/privacy</a></td></tr>
                        <tr><td><strong>Google AdSense</strong></td><td>Display advertising</td><td><a href="https://policies.google.com/privacy" target="_blank" rel="noopener">policies.google.com/privacy</a></td></tr>
                        <tr><td><strong>GitHub Pages</strong></td><td>Site hosting</td><td><a href="https://docs.github.com/en/site-policy/privacy-policies/github-general-privacy-statement" target="_blank" rel="noopener">docs.github.com/privacy</a></td></tr>
                        <tr><td><strong>Twitter / X</strong></td><td>Social media link</td><td><a href="https://twitter.com/privacy" target="_blank" rel="noopener">twitter.com/privacy</a></td></tr>
                        <tr><td><strong>LinkedIn</strong></td><td>Social media link</td><td><a href="https://www.linkedin.com/legal/privacy-policy" target="_blank" rel="noopener">linkedin.com/legal/privacy-policy</a></td></tr>
                    </tbody>
                </table>
            </div>

            <div class="privacy-section">
                <h3>6. Data Retention</h3>
                <table>
                    <thead>
                        <tr><th>Data type</th><th>Retention period</th></tr>
                    </thead>
                    <tbody>
                        <tr><td>Email enquiries</td><td>Until the matter is resolved, then deleted</td></tr>
                        <tr><td>Analytics data (GA4)</td><td>14 months (GA4 default)</td></tr>
                        <tr><td>Cookie consent preference</td><td>1 year</td></tr>
                    </tbody>
                </table>
            </div>

            <div class="privacy-section">
                <h3>7. Your Rights</h3>
                <p>Depending on your location, you may have the following rights:</p>
                
                <p><strong>Under GDPR (EU/EEA/UK residents):</strong></p>
                <ul>
                    <li><strong>Access:</strong> Request a copy of your personal data</li>
                    <li><strong>Rectification:</strong> Correct inaccurate data</li>
                    <li><strong>Erasure:</strong> Request deletion of your data ("right to be forgotten")</li>
                    <li><strong>Restriction:</strong> Limit how we process your data</li>
                    <li><strong>Portability:</strong> Receive your data in a machine-readable format</li>
                    <li><strong>Object:</strong> Object to processing based on legitimate interests</li>
                    <li><strong>Withdraw consent:</strong> At any time, for consent-based processing</li>
                </ul>

                <p><strong>Under CCPA (California residents):</strong></p>
                <ul>
                    <li><strong>Know:</strong> What personal data we collect and how we use it</li>
                    <li><strong>Delete:</strong> Request deletion of personal information we have collected</li>
                    <li><strong>Opt-out of sale:</strong> We do not sell personal information</li>
                    <li><strong>Non-discrimination:</strong> We will not discriminate against you for exercising your rights</li>
                </ul>
                <p>To exercise any of these rights, email us at <a href="mailto:aiblogauto@gmail.com">aiblogauto@gmail.com</a>. We will respond within 30 days.</p>
            </div>

            <div class="privacy-section">
                <h3>8. Advertising</h3>
                <p>This site participates in the <strong>Google AdSense</strong> programme, which serves interest-based advertisements. To opt out of interest-based advertising by Google, visit <a href="https://adssettings.google.com/" target="_blank" rel="noopener">adssettings.google.com</a>. You can also opt out via the <a href="https://optout.networkadvertising.org/" target="_blank" rel="noopener">NAI opt-out tool</a> or <a href="https://optout.aboutads.info/" target="_blank" rel="noopener">DAA opt-out tool</a>.</p>
            </div>

            <div class="privacy-section">
                <h3>9. Children's Privacy</h3>
                <p>This site is not directed at children under 13 (or under 16 in the EU). We do not knowingly collect personal data from children.</p>
            </div>

            <div class="privacy-section">
                <h3>10. Changes to This Policy</h3>
                <p>We will post any changes to this page with an updated "Last updated" date. Significant changes will be noted at the top of this page.</p>
            </div>

            <div class="highlight-box">
                <h3>11. Contact</h3>
                <p>For privacy-related enquiries or to exercise your rights:</p>
                <p><strong>Email:</strong> <a href="mailto:aiblogauto@gmail.com" style="color:white;text-decoration:underline;">aiblogauto@gmail.com</a></p>
                <p><strong>Location:</strong> Nairobi, Kenya (UTC+3)</p>
                <p><strong>Response time:</strong> Within 5 business days</p>
                <p style="margin-top:1rem;font-size:0.9rem;opacity:0.9;">If you are an EU resident and believe we have not addressed your concern, you have the right to lodge a complaint with your local data protection authority.</p>
            </div>

            <p><strong>Last updated:</strong> {{ current_date }}</p>
        </article>
    </main>
    <footer><div class="container"><p>&copy; {{ current_year }} {{ site_name }}.</p></div></footer>
    <script src="{{ base_path }}/static/navigation.js"></script>
    <script defer src="{{ base_path }}/static/pwa.js"></script>
</body>
</html>"""

    # FIX BUG-12: TERMS_TMPL had the same relative path bug as PRIVACY_TMPL.
    TERMS_TMPL = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Terms of Service - {{ site_name }}</title>
    <meta name="description" content="Terms of Service for {{ site_name }}">
    <meta name="base-path" content="{{ base_path }}">
    <link rel="canonical" href="{{ base_url }}/terms-of-service/">
    <script>window.dataLayer=window.dataLayer||[];function gtag(){window.dataLayer.push(arguments);}window.gtag=window.gtag||gtag;(function(){var mm=document.cookie.match(/(?:^|;) *cookie_consent_v1=([^;]*)/);var g=mm&&decodeURIComponent(mm[1])==='accepted';var s=g?'granted':'denied';gtag('consent','default',{ad_storage:s,ad_user_data:s,ad_personalization:s,analytics_storage:s,functionality_storage:s,personalization_storage:s,wait_for_update:g?0:500});})();</script>
    <script src="{{ base_path }}/static/consent.js" defer></script>
    {{ global_meta_tags | safe }}
    <link rel="stylesheet" href="{{ base_path }}/static/style.css">
    <link rel="manifest" href="{{ base_path }}/manifest.json">
    <link rel="apple-touch-icon" href="{{ base_path }}/static/icons/icon-192x192.png">
    <style>
        .terms-section{background:#f8f9fa;padding:1.5rem;margin-bottom:1.5rem;border-radius:8px;border-left:4px solid #6366f1}
        .terms-section h3{color:#333;margin-top:0}
        .highlight-box{background:linear-gradient(135deg,#667eea,#764ba2);color:white;padding:1.5rem;border-radius:8px;margin:1.5rem 0}
        .highlight-box h3{margin-top:0;color:white}
        .warning-box{background:#f8d7da;border-left:4px solid #dc3545;padding:1rem 1.5rem;margin:1.5rem 0;border-radius:4px;color:#721c24}
    </style>
</head>
<body>
    <header><div class="container">
        <h1><a href="{{ base_path }}/">{{ site_name }}</a></h1>
        <nav><a href="{{ base_path }}/">Home</a><a href="{{ base_path }}/about/">About</a><a href="{{ base_path }}/contact/">Contact</a>
        <a href="{{ base_path }}/privacy-policy/">Privacy Policy</a><a href="{{ base_path }}/terms-of-service/">Terms of Service</a></nav>
    </div></header>
    <main class="container">
        <div class="hero"><h2>Terms of Service</h2><p>Please read these terms carefully before using our site</p></div>
        <article class="page-content">
            <div class="terms-section"><h3>1. Acceptance</h3>
                <p>By accessing {{ site_name }}, you agree to these Terms and our Privacy Policy.</p></div>
            <div class="terms-section"><h3>2. AI-Assisted Content</h3>
                <div class="warning-box"><p>Some content is drafted with AI assistance and reviewed for accuracy by the author. Always verify technical information independently before using it in production.</p></div></div>
            <div class="terms-section"><h3>3. Affiliate Disclosure</h3>
                <p>This site participates in affiliate programmes. Purchases through affiliate links may earn us a commission at no additional cost to you.</p></div>
            <div class="terms-section"><h3>4. Disclaimer</h3>
                <p>Content is provided for informational purposes. We make no warranty that it is accurate, complete, or suitable for any particular purpose.</p></div>
            <div class="terms-section"><h3>5. Governing Law</h3>
                <p>These terms are governed by the laws of Kenya.</p></div>
            <div class="highlight-box"><h3>6. Contact</h3>
                <p>Email: <a href="mailto:aiblogauto@gmail.com" style="color:white;text-decoration:underline;">aiblogauto@gmail.com</a></p></div>
            <p><strong>Last updated:</strong> {{ current_date }}</p>
        </article>
    </main>
    <footer><div class="container"><p>&copy; {{ current_year }} {{ site_name }}.</p></div></footer>
    <script src="{{ base_path }}/static/navigation.js"></script>
    <script defer src="{{ base_path }}/static/pwa.js"></script>
</body>
</html>"""

    CONTACT_TMPL = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Contact Kubai Kevin — {{ site_name }}</title>
    <meta name="description" content="Contact Kubai Kevin, software developer and author of {{ site_name }}. Based in Nairobi, Kenya. Responds within 3–5 business days.">
    <meta name="base-path" content="{{ base_path }}">
    <link rel="canonical" href="{{ base_url }}/contact/">
    <script>window.dataLayer=window.dataLayer||[];function gtag(){window.dataLayer.push(arguments);}window.gtag=window.gtag||gtag;(function(){var mm=document.cookie.match(/(?:^|;) *cookie_consent_v1=([^;]*)/);var g=mm&&decodeURIComponent(mm[1])==='accepted';var s=g?'granted':'denied';gtag('consent','default',{ad_storage:s,ad_user_data:s,ad_personalization:s,analytics_storage:s,functionality_storage:s,personalization_storage:s,wait_for_update:g?0:500});})();</script>
    <script src="{{ base_path }}/static/consent.js" defer></script>
    {{ global_meta_tags | safe }}
    <script type="application/ld+json">
    {
      "@context": "https://schema.org",
      "@type": "ContactPage",
      "name": "Contact Kubai Kevin",
      "url": "{{ base_url }}/contact/",
      "description": "Contact the author of {{ site_name }}",
      "mainEntity": {
        "@type": "Person",
        "name": "Kubai Kevin",
        "email": "aiblogauto@gmail.com",
        "url": "{{ base_url }}/about/"
      }
    }
    </script>
    <link rel="stylesheet" href="{{ base_path }}/static/style.css">
    <link rel="manifest" href="{{ base_path }}/manifest.json">
    <link rel="apple-touch-icon" href="{{ base_path }}/static/icons/icon-192x192.png">
    <style>
        .contact-hero {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; padding: 2.5rem 2rem; border-radius: 12px;
            text-align: center; margin-bottom: 2rem;
        }
        .contact-hero h1 { color: white; font-size: 2rem; margin-bottom: 0.5rem; }
        .contact-hero p { opacity: 0.9; max-width: 500px; margin: 0 auto; }
        .contact-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
            margin-bottom: 1.5rem;
        }
        @media (max-width: 600px) { .contact-grid { grid-template-columns: 1fr; } }
        .contact-card {
            background: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 12px;
            padding: 1.5rem;
        }
        .contact-card h2 {
            font-size: 1.1rem; color: #1a1a2e; margin-top: 0; margin-bottom: 0.75rem;
        }
        .contact-card p { color: #555; font-size: 0.9rem; line-height: 1.6; margin: 0; }
        .contact-email-block {
            background: white; border: 2px solid #6366f1; border-radius: 12px;
            padding: 1.75rem; text-align: center; margin-bottom: 1.5rem;
        }
        .contact-email-block p { color: #555; margin-bottom: 1rem; font-size: 0.95rem; }
        .email-link {
            display: inline-block; font-size: 1.15rem; font-weight: 700;
            color: #6366f1; text-decoration: none; padding: 0.6rem 1.5rem;
            background: #f0f4ff; border-radius: 8px; transition: all 0.2s;
        }
        .email-link:hover { background: #6366f1; color: white; }
        .response-note {
            font-size: 0.8rem; color: #888; margin-top: 0.75rem !important;
        }
        .appropriate-list { list-style: none; padding: 0; margin: 0.5rem 0 0; }
        .appropriate-list li {
            padding: 0.4rem 0; border-bottom: 1px solid #e8e8e8;
            font-size: 0.88rem; color: #555; padding-left: 1.2rem; position: relative;
        }
        .appropriate-list li::before {
            content: "✓"; position: absolute; left: 0; color: #6366f1; font-weight: 700;
        }
        .appropriate-list li:last-child { border-bottom: none; }
        .not-list li::before { content: "✗"; color: #dc3545; }
    </style>
</head>
<body>
    <header>
        <div class="container">
            <h1><a href="{{ base_path }}/">{{ site_name }}</a></h1>
            <nav>
                <a href="{{ base_path }}/">Home</a>
                <a href="{{ base_path }}/about/">About</a>
                <a href="{{ base_path }}/contact/">Contact</a>
                <a href="{{ base_path }}/privacy-policy/">Privacy Policy</a>
                <a href="{{ base_path }}/terms-of-service/">Terms of Service</a>
            </nav>
        </div>
    </header>

    <main class="container">
        <div class="contact-hero">
            <h1>Contact</h1>
            <p>Get in touch with Kubai Kevin, the author of {{ site_name }}.</p>
        </div>

        <div class="contact-email-block">
            <p>The best way to reach me is by email. I read every message.</p>
            <a href="mailto:aiblogauto@gmail.com" class="email-link">
                aiblogauto@gmail.com
            </a>
            <p class="response-note">
                Based in Nairobi, Kenya (UTC+3 / East Africa Time).
                Typical response time: 3–5 business days.
            </p>
        </div>

        <div class="contact-grid">
            <div class="contact-card">
                <h2>Good reasons to write</h2>
                <ul class="appropriate-list">
                    <li>Factual error in an article — I always want to know</li>
                    <li>Outdated code example or deprecated API reference</li>
                    <li>Question about something specific in an article</li>
                    <li>Topic suggestion based on a real problem you're solving</li>
                    <li>Collaboration or guest contribution proposal</li>
                </ul>
            </div>
            <div class="contact-card">
                <h2>What I don't respond to</h2>
                <ul class="appropriate-list not-list">
                    <li>Paid link insertion requests</li>
                    <li>Guest posts that aren't from practitioners</li>
                    <li>Generic "I love your blog" outreach with no specific question</li>
                    <li>Requests to endorse tools I haven't used</li>
                </ul>
            </div>
        </div>

        <div class="contact-card" style="margin-bottom:1.5rem">
            <h2>Social media</h2>
            <p>For quick questions or to follow new articles as they publish:</p>
            <ul class="appropriate-list" style="margin-top:0.75rem">
                 <li>
                    Twitter / X:
                    <a href="https://twitter.com/KubaiKevin" target="_blank" rel="noopener"
                       style="color:#6366f1;font-weight:600;">@KubaiKevin</a>
                    — DMs open for short questions
                </li>
                <li>
                    LinkedIn:
                    <a href="https://www.linkedin.com/in/kevin-kubai-22b61b37/"
                       target="_blank" rel="noopener"
                       style="color:#6366f1;font-weight:600;">Kevin Kubai</a>
                    — connect if you want to discuss opportunities
                </li>
            </ul>
        </div>
    </main>

    <footer>
        <div class="container">
            <p>&copy; {{ current_year }} {{ site_name }} · Written by Kubai Kevin</p>
        </div>
    </footer>
    <script src="{{ base_path }}/static/navigation.js"></script>
    <script defer src="{{ base_path }}/static/pwa.js"></script>
</body>
</html>"""

    NOT_FOUND_TMPL = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Redirecting… — {{ site_name }}</title>
    <meta name="robots" content="noindex, nofollow">
    <meta name="base-path" content="{{ base_path }}">
    {# No-JS fallback: redirect immediately (0s delay) even without JS #}
    <meta http-equiv="refresh" content="0;url={{ base_path }}/">
    {{ global_meta_tags | safe }}
    <style>
        html, body {
            margin: 0; padding: 0; height: 100%;
            display: flex; align-items: center; justify-content: center;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #fff; color: #555;
        }
        .fallback { text-align: center; padding: 2rem; }
        .fallback a {
            color: #6366f1; font-weight: 600; text-decoration: none;
        }
        .fallback a:hover { text-decoration: underline; }
        .spinner {
            width: 28px; height: 28px; margin: 0 auto 1rem;
            border: 3px solid #e0e7ff; border-top-color: #6366f1;
            border-radius: 50%; animation: spin 0.7s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <!--
        Redirect fires immediately on script execution below.
        This fallback content only renders if a browser/proxy blocks
        BOTH the meta-refresh above and JavaScript execution.
    -->
    <div class="fallback">
        <div class="spinner" aria-hidden="true"></div>
        <p>Taking you to the homepage&hellip;</p>
        <p><a href="{{ base_path }}/">Click here if you are not redirected automatically</a></p>
    </div>
    <script>
    (function () {
        'use strict';
        var BASE_PATH = '{{ base_path }}';
        // replace() keeps this 404 URL out of browser history, so the
        // Back button returns the user to wherever they came from
        // rather than bouncing back into this redirect.
        window.location.replace(BASE_PATH + '/');
    }());
    </script>
</body>
</html>"""

    AI_DISCLOSURE_TMPL = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <script>window.dataLayer=window.dataLayer||[];function gtag(){window.dataLayer.push(arguments);}window.gtag=window.gtag||gtag;(function(){var mm=document.cookie.match(/(?:^|;) *cookie_consent_v1=([^;]*)/);var g=mm&&decodeURIComponent(mm[1])==='accepted';var s=g?'granted':'denied';gtag('consent','default',{ad_storage:s,ad_user_data:s,ad_personalization:s,analytics_storage:s,functionality_storage:s,personalization_storage:s,wait_for_update:g?0:500});})();</script>
    <script src="{{ base_path }}/static/consent.js" defer></script>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Content Policy — {{ site_name }}</title>
    <meta name="description" content="How {{ site_name }} uses AI writing tools, our editorial process, and how we ensure accuracy and originality.">
    <meta name="base-path" content="{{ base_path }}">
    <link rel="canonical" href="{{ base_url }}/ai-content-policy/">
    <meta name="robots" content="index, follow">
    {{ global_meta_tags | safe }}
    <link rel="stylesheet" href="{{ base_path }}/static/style.css">
    <link rel="manifest" href="{{ base_path }}/manifest.json">
    <link rel="apple-touch-icon" href="{{ base_path }}/static/icons/icon-192x192.png">
    <style>
        .policy-section { background:#f8f9fa; padding:1.5rem; margin-bottom:1.5rem; border-radius:8px; border-left:4px solid #6366f1; }
        .policy-section h2 { color:#333; margin-top:0; font-size:1.2rem; }
        .highlight-box { background:linear-gradient(135deg,#667eea,#764ba2); color:white; padding:1.5rem; border-radius:8px; margin:1.5rem 0; }
        .highlight-box h2 { margin-top:0; color:white; }
        .check-list { list-style:none; padding:0; }
        .check-list li { padding-left:2rem; position:relative; margin-bottom:0.6rem; }
        .check-list li::before { content:"✓"; position:absolute; left:0; color:#6366f1; font-weight:700; }
        .cross-list li::before { content:"✗"; color:#dc3545; }
        .two-col { display:grid; grid-template-columns:1fr 1fr; gap:1rem; }
        @media(max-width:600px){ .two-col{grid-template-columns:1fr;} }
        .card { background:#fff; border:1px solid #e0e0e0; border-radius:8px; padding:1.25rem; }
        .card h3 { margin-top:0; font-size:1rem; color:#1a1a2e; }
    </style>
</head>
<body>
    <header><div class="container">
        <h1><a href="{{ base_path }}/">{{ site_name }}</a></h1>
        <nav>
            <a href="{{ base_path }}/">Home</a>
            <a href="{{ base_path }}/about/">About</a>
            <a href="{{ base_path }}/contact/">Contact</a>
            <a href="{{ base_path }}/privacy-policy/">Privacy Policy</a>
        </nav>
    </div></header>
    <main class="container">
        <div class="hero" style="background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;padding:2.5rem 2rem;border-radius:12px;text-align:center;margin-bottom:2rem;">
            <h1 style="color:#fff;font-size:2rem;margin-bottom:0.5rem;">AI Content Policy</h1>
            <p style="opacity:0.9;">How we use AI tools, what human review covers, and what we guarantee.</p>
        </div>

        <article class="page-content">
            <div class="highlight-box">
                <h2>Our Commitment in One Paragraph</h2>
                <p style="margin-bottom:0;">Every article on {{ site_name }} is written on a topic selected by the author (Kubai Kevin) from his own production experience, then drafted end-to-end by LLMs through an automated pipeline. Given the publishing volume, articles are <strong>not individually fact-checked or hand-edited by a human before going live</strong>. Automated quality gates (below) catch thin, duplicate, and boilerplate content before publication, and the author reviews and corrects specific articles when readers flag issues.</p>
            </div>

            <div class="policy-section">
                <h2>1. How AI is Used on This Site</h2>
                <p>{{ site_name }} uses large language model (LLM) APIs — including models from Mistral, Google Gemini, Meta Llama (via Groq and other providers), and similar services — to:</p>
                <ul class="check-list">
                    <li>Draft article structure and outlines based on an author-selected topic.</li>
                    <li>Generate prose that is then reviewed and edited.</li>
                    <li>Suggest SEO keywords, meta descriptions, and social media copy.</li>
                    <li>Automate repetitive formatting tasks (headings, code blocks, tables).</li>
                </ul>
                <p>AI tools are used as a writing <em>assistant</em>, not as the final publisher. The author remains responsible for all published content.</p>
            </div>

            <div class="policy-section">
                <h2>2. What AI Does NOT Do on This Site</h2>
                <div class="two-col">
                    <div class="card">
                        <h3>Not scraped or aggregated</h3>
                        <p style="font-size:0.9rem;color:#555;">We do not scrape, copy, or rephrase content from other websites. Every article starts from a topic brief, not from existing web content.</p>
                    </div>
                    <div class="card">
                        <h3>Not published without review</h3>
                        <p style="font-size:0.9rem;color:#555;">Our pipeline includes automated quality gates: minimum word count (1,500+), boilerplate detection, and content quality validation before any article is saved.</p>
                    </div>
                    <div class="card">
                        <h3>Not keyword-stuffed</h3>
                        <p style="font-size:0.9rem;color:#555;">Content prompts explicitly ban keyword stuffing, filler phrases, and AI-detectable clichés. Quality validators check for these automatically.</p>
                    </div>
                    <div class="card">
                        <h3>Illustrative, not audited, figures</h3>
                        <p style="font-size:0.9rem;color:#555;">Specific numbers, benchmarks, and cost figures in articles are generated as realistic illustrations of the pattern being discussed, not pulled from a verified source per article. Every post says so explicitly in its "About this article" footer — treat figures as directional and confirm them against current official documentation before relying on them in production.</p>
                    </div>
                </div>
            </div>

            <div class="policy-section">
                <h2>3. Quality Controls in Our Publishing Pipeline</h2>
                <p>Before an article is saved and published, it passes through automated checks that enforce:</p>
                <ul class="check-list">
                    <li><strong>Minimum 1,500 words</strong> — thin content is rejected outright.</li>
                    <li><strong>Boilerplate detection</strong> — template artifacts or placeholder text trigger automatic discard.</li>
                    <li><strong>Filler phrase detection</strong> — AI-pattern phrases ("dive into", "game-changer", "it's important to note") are flagged and the article is regenerated.</li>
                    <li><strong>E-E-A-T signal injection</strong> — every article includes an author byline, publication date, last-reviewed date, and an editorial standards footer.</li>
                    <li><strong>Duplicate title detection</strong> — Jaccard similarity is checked against all existing posts to prevent near-duplicate articles.</li>
                    <li><strong>Stale year scrubbing</strong> — dates and statistics are validated to use current year references.</li>
                </ul>
                <p>Articles that fail these gates are discarded. No fallback or placeholder article is published in their place.</p>
            </div>

            <div class="policy-section">
                <h2>4. Author's Role</h2>
                <p><strong>Kubai Kevin</strong> is the author and editor of all content on this site. His role in the pipeline includes:</p>
                <ul class="check-list">
                    <li>Defining the topic list from personal production experience (not keyword research alone).</li>
                    <li>Maintaining the automated quality gate code that decides what is and isn't published (minimum length, duplicate detection, boilerplate and filler-phrase rejection).</li>
                    <li>Reviewing and correcting individual articles when readers report an error via the <a href="{{ base_path }}/contact/">contact page</a>.</li>
                    <li>Periodically spot-checking the published post library for accuracy and policy compliance.</li>
                </ul>
                <p>The author's LinkedIn profile and contact details are published on the <a href="{{ base_path }}/about/">About page</a> for full transparency.</p>
            </div>

            <div class="policy-section">
                <h2>5. Corrections Policy</h2>
                <p>If an article contains a factual error, an outdated tool version, or inaccurate code, please <a href="{{ base_path }}/contact/">contact us</a>. We will investigate, correct the article on a best-effort basis, and note the correction date on the affected post.</p>
            </div>

            <div class="policy-section">
                <h2>6. AI Disclosure Compliance</h2>
                <p>This page serves as the AI content disclosure for {{ site_name }} in jurisdictions that require or recommend disclosure of AI-assisted content generation. We believe transparency is both ethically correct and practically important for reader trust.</p>
                <p>We monitor evolving regulatory requirements around AI content labelling (including EU AI Act guidance, FTC guidance, and platform-specific requirements) and will update this policy accordingly.</p>
            </div>

            <p style="color:#888;font-size:0.85rem;margin-top:2rem;"><strong>Last updated:</strong> {{ current_date }}</p>
        </article>
    </main>
    <footer><div class="container">
        <p>&copy; {{ current_year }} {{ site_name }} · Written by Kubai Kevin</p>
    </div></footer>
    <script src="{{ base_path }}/static/navigation.js"></script>
</body>
</html>"""

    DMCA_TMPL = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <script>window.dataLayer=window.dataLayer||[];function gtag(){window.dataLayer.push(arguments);}window.gtag=window.gtag||gtag;(function(){var mm=document.cookie.match(/(?:^|;) *cookie_consent_v1=([^;]*)/);var g=mm&&decodeURIComponent(mm[1])==='accepted';var s=g?'granted':'denied';gtag('consent','default',{ad_storage:s,ad_user_data:s,ad_personalization:s,analytics_storage:s,functionality_storage:s,personalization_storage:s,wait_for_update:g?0:500});})();</script>
    <script src="{{ base_path }}/static/consent.js" defer></script>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DMCA &amp; Copyright Policy — {{ site_name }}</title>
    <meta name="description" content="DMCA takedown process and copyright policy for {{ site_name }}.">
    <meta name="base-path" content="{{ base_path }}">
    <link rel="canonical" href="{{ base_url }}/dmca/">
    <meta name="robots" content="index, follow">
    {{ global_meta_tags | safe }}
    <link rel="stylesheet" href="{{ base_path }}/static/style.css">
    <link rel="manifest" href="{{ base_path }}/manifest.json">
    <link rel="apple-touch-icon" href="{{ base_path }}/static/icons/icon-192x192.png">
    <style>
        .dmca-section { background:#f8f9fa; padding:1.5rem; margin-bottom:1.5rem; border-radius:8px; border-left:4px solid #6366f1; }
        .dmca-section h2 { color:#333; margin-top:0; font-size:1.2rem; }
        .dmca-section h3 { color:#444; margin-top:1rem; font-size:1rem; }
        .highlight-box { background:linear-gradient(135deg,#667eea,#764ba2); color:white; padding:1.5rem; border-radius:8px; margin:1.5rem 0; }
        .highlight-box h2 { margin-top:0; color:white; font-size:1.1rem; }
        .highlight-box a { color:#fff; font-weight:bold; }
        .warning-box { background:#fff3cd; border-left:4px solid #ffc107; padding:1rem 1.5rem; margin:1.5rem 0; border-radius:4px; color:#856404; }
        .step-list { counter-reset:steps; list-style:none; padding:0; }
        .step-list li { counter-increment:steps; padding-left:2.5rem; position:relative; margin-bottom:0.75rem; }
        .step-list li::before { content:counter(steps); position:absolute; left:0; top:0; background:#6366f1; color:#fff; width:1.6rem; height:1.6rem; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:0.8rem; font-weight:700; }
    </style>
</head>
<body>
    <header><div class="container">
        <h1><a href="{{ base_path }}/">{{ site_name }}</a></h1>
        <nav>
            <a href="{{ base_path }}/">Home</a>
            <a href="{{ base_path }}/about/">About</a>
            <a href="{{ base_path }}/contact/">Contact</a>
            <a href="{{ base_path }}/privacy-policy/">Privacy Policy</a>
        </nav>
    </div></header>
    <main class="container">
        <div class="hero" style="background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;padding:2.5rem 2rem;border-radius:12px;text-align:center;margin-bottom:2rem;">
            <h1 style="color:#fff;font-size:2rem;margin-bottom:0.5rem;">DMCA &amp; Copyright Policy</h1>
            <p style="opacity:0.9;">How to report copyright concerns and how we handle them.</p>
        </div>

        <article class="page-content">
            <div class="dmca-section">
                <h2>1. Copyright Ownership</h2>
                <p>All original articles, prose, code examples, and other content published on {{ site_name }} are the intellectual property of <strong>Kubai Kevin</strong> and are protected under copyright law. Unauthorised reproduction, distribution, or derivative works without explicit written permission is prohibited.</p>
                <p>Content on this site is produced with AI writing assistance as part of an automated pipeline; not every article receives individual human review before publication (see our <a href="{{ base_path }}/ai-content-policy/">AI content policy</a>). The topic selection, editorial decisions, and voice are the author's own.</p>
            </div>

            <div class="dmca-section">
                <h2>2. Third-Party Content</h2>
                <p>{{ site_name }} may reference, quote, or link to third-party sources. All such use is intended to be transformative, educational, or analytical in nature. If you believe your copyrighted work has been used in a way that constitutes infringement, please follow the notice process below.</p>
                <div class="warning-box">
                    <p><strong>Note:</strong> This site uses an automated content generation pipeline. Despite quality controls, it is possible that a generated article may inadvertently resemble or reproduce protected material. We take all DMCA notices seriously and will respond within <strong>24–48 hours</strong>.</p>
                </div>
            </div>

            <div class="dmca-section">
                <h2>3. DMCA Takedown — Notice Requirements</h2>
                <p>To submit a valid DMCA takedown notice under 17 U.S.C. § 512(c)(3), your written notification must include all of the following:</p>
                <ol class="step-list">
                    <li>Your physical or electronic signature (or that of your authorised agent).</li>
                    <li>Identification of the copyrighted work you claim has been infringed.</li>
                    <li>Identification of the material you claim is infringing, with enough information for us to locate it (e.g. the specific URL on our site).</li>
                    <li>Your contact information: name, address, telephone number, and email address.</li>
                    <li>A statement that you have a good-faith belief that the disputed use is not authorised by the copyright owner, its agent, or the law.</li>
                    <li>A statement, under penalty of perjury, that the information in your notice is accurate and that you are (or are authorised to act on behalf of) the copyright owner.</li>
                </ol>
            </div>

            <div class="highlight-box">
                <h2>4. Where to Send DMCA Notices</h2>
                <p>Email your complete DMCA notice to:<br>
                <a href="mailto:aiblogauto@gmail.com"><strong>aiblogauto@gmail.com</strong></a></p>
                <p style="margin-bottom:0;font-size:0.9rem;opacity:0.9;">Subject line: <em>DMCA Takedown Request — [URL of infringing page]</em></p>
            </div>

            <div class="dmca-section">
                <h2>5. Our Response Process</h2>
                <p>Upon receiving a valid DMCA notice, we will:</p>
                <ul>
                    <li>Acknowledge receipt within <strong>24 hours</strong>.</li>
                    <li>Investigate the claim and, if valid, remove or disable access to the infringing content within <strong>48 hours</strong>.</li>
                    <li>Notify the author of the takedown.</li>
                    <li>Maintain a record of all DMCA notices for compliance purposes.</li>
                </ul>
                <p>We reserve the right to remove content proactively if we believe it may infringe third-party rights, without waiting for a formal DMCA notice.</p>
            </div>

            <div class="dmca-section">
                <h2>6. Counter-Notice</h2>
                <p>If you believe content was removed in error, you may submit a counter-notice under 17 U.S.C. § 512(g). Counter-notices must include equivalent information to a takedown notice, plus a statement that you consent to the jurisdiction of the federal court where your address is located.</p>
            </div>

            <div class="dmca-section">
                <h2>7. Repeat Infringer Policy</h2>
                <p>{{ site_name }} will disable or terminate the publishing pipeline for any content source that is the subject of repeated valid DMCA notices, in accordance with the safe-harbour provisions of the DMCA.</p>
            </div>

            <div class="dmca-section">
                <h2>8. Using Our Content</h2>
                <p>Brief quotations (under 150 words) with a clear attribution link to the source article are permitted under fair use. For longer excerpts, syndication, or any commercial use, contact us at <a href="mailto:aiblogauto@gmail.com">aiblogauto@gmail.com</a> to request written permission.</p>
            </div>

            <p style="color:#888;font-size:0.85rem;margin-top:2rem;"><strong>Last updated:</strong> {{ current_date }}</p>
        </article>
    </main>
    <footer><div class="container">
        <p>&copy; {{ current_year }} {{ site_name }} · Written by Kubai Kevin</p>
    </div></footer>
    <script src="{{ base_path }}/static/navigation.js"></script>
</body>
</html>"""

    env = Environment(loader=BaseLoader())
    return {
        'post':             env.from_string(POST_TMPL),
        'index':            env.from_string(INDEX_TMPL),
        'about':            env.from_string(ABOUT_TMPL),
        'privacy_policy':   env.from_string(PRIVACY_TMPL),
        'terms_of_service': env.from_string(TERMS_TMPL),
        'contact':          env.from_string(CONTACT_TMPL),
        'not_found':        env.from_string(NOT_FOUND_TMPL),
        'dmca':             env.from_string(DMCA_TMPL),
        'ai_disclosure':    env.from_string(AI_DISCLOSURE_TMPL),
    }
