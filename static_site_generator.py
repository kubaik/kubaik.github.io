from monetization_manager import MonetizationManager
from visibility_automator import VisibilityAutomator
from seo_optimizer import SEOOptimizer
from blog_post import BlogPost
from jinja2 import Template, Environment, BaseLoader
from datetime import datetime
from typing import Dict, List
from pathlib import Path
import markdown as md
import json
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


def _clean_url(url: str) -> str:
    """Remove Markdown link-formatting artifacts like [text](url) -> url."""
    import re
    cleaned = re.sub(r'\[([^\]]*)\]\(([^)]+)\)', r'\2', url)
    return cleaned.strip()


def _normalize_iso_date(dt_str: str) -> str:
    """Normalize an ISO datetime string to YYYY-MM-DDTHH:MM:SS+00:00."""
    if not dt_str:
        return dt_str
    try:
        dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%dT%H:%M:%S+00:00')
    except Exception:
        if '.' in dt_str:
            base, frac = dt_str.split('.', 1)
            tz = ''
            for sep in ('+', '-'):
                if sep in frac:
                    tz = sep + frac.split(sep, 1)[1]
                    break
            return base + (tz or '+00:00')
        return dt_str


# Global Template String Constants
AUTHOR_PAGE_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kubai Kevin — Software Developer and Writer</title>
    <meta name="description" content="Kubai Kevin is a software developer based in Nairobi, Kenya. He writes about AI, backend engineering, and developer careers at {site_name}.">
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

POST_TMPL = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ post.title }} — {{ site_name }}</title>
    <meta name="description" content="{{ post.meta_description }}">
    <link rel="stylesheet" href="{{ base_path }}/static/style.css">
    {{ meta_tags | safe }}
    {{ article_schema | safe }}
</head>
<body>
    <header><div class="container">
        <h1><a href="{{ base_path }}/">{{ site_name }}</a></h1>
        <nav><a href="{{ base_path }}/">Home</a><a href="{{ base_path }}/about/">About</a></nav>
    </div></header>
    <main class="container" itemscope itemtype="https://schema.org/TechArticle">
        {% if header_ad %}<div class="ad-header">{{ header_ad | safe }}</div>{% endif %}
        <article>
            <h1 itemprop="headline">{{ post.title }}</h1>
            <div class="author-box">
                <p class="author-meta">
                    {{ post.reading_time }} min read &nbsp;·&nbsp; {{ post.word_count }} words
                    {% if post.last_updated_iso %} &nbsp;·&nbsp; Updated {{ post.last_updated_iso }} {% endif %}
                </p>
            </div>
            <div class="post-content" itemprop="articleBody">
                {{ post.content_html | safe }}
                {% if inline_ad %}<div class="ad-inline">{{ inline_ad | safe }}</div>{% endif %}
                {% if middle_ad %}<div class="ad-middle">{{ middle_ad | safe }}</div>{% endif %}
            </div>
        </article>
        {% if footer_ad %}<div class="ad-footer">{{ footer_ad | safe }}</div>{% endif %}
    </main>
    <footer><div class="container"><p>&copy; {{ current_year }} {{ site_name }}</p></div></footer>
</body>
</html>"""

INDEX_TMPL = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ site_name }} — {{ site_description }}</title>
    <link rel="stylesheet" href="{{ base_path }}/static/style.css">
    {{ homepage_meta_tags | safe }}
</head>
<body>
    <header><div class="container">
        <h1><a href="{{ base_path }}/">{{ site_name }}</a></h1>
        <nav><a href="{{ base_path }}/">Home</a><a href="{{ base_path }}/about/">About</a></nav>
    </div></header>
    <main class="container">
        <section class="recent-posts">
            <h2>Latest Posts</h2>
            {% if posts %}
            <div id="posts-container" class="post-grid">
                {% for post in posts %}
                <a class="post-card" href="{{ base_path }}/{{ post.slug }}/">
                    <h3>{{ post.title }}</h3>
                    <p class="post-excerpt">{{ post.meta_description }}</p>
                    <p class="post-reading-time">{{ post.reading_time }} min read</p>
                </a>
                {% endfor %}
            </div>
            {% endif %}
        </section>
    </main>
    <footer><div class="container"><p>&copy; {{ current_year }} {{ site_name }}</p></div></footer>
</body>
</html>"""

ABOUT_TMPL = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>About — {{ site_name }}</title>
    <link rel="stylesheet" href="{{ base_path }}/static/style.css">
</head>
<body>
    <header><div class="container">
        <h1><a href="{{ base_path }}/">{{ site_name }}</a></h1>
        <nav><a href="{{ base_path }}/">Home</a><a href="{{ base_path }}/about/">About</a></nav>
    </div></header>
    <main class="container">
        <article class="page-content">
            <h2>About Our Editorial Process</h2>
            <p>Articles are rigorously planned, human-authored, and thoroughly fact-checked before publishing.</p>
            {% if posts %}
            <div class="section-card">
                <h2>Recent articles</h2>
                <ul class="recent-posts-list">
                    {% for post in posts[:8] %}
                    <li><a href="{{ base_path }}/{{ post.slug }}/">{{ post.title }}</a></li>
                    {% endfor %}
                </ul>
            </div>
            {% endif %}
        </article>
    </main>
    <footer><div class="container"><p>&copy; {{ current_year }} {{ site_name }} · Written by Kubai Kevin</p></div></footer>
</body>
</html>"""

PRIVACY_TMPL = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Privacy Policy - {{ site_name }}</title>
    <link rel="stylesheet" href="{{ base_path }}/static/style.css">
</head>
<body>
    <header><div class="container"><h1><a href="{{ base_path }}/">{{ site_name }}</a></h1></div></header>
    <main class="container">
        <h2>Privacy Policy</h2>
        <p>Last updated: {{ current_date }}</p>
        <div class="privacy-section">
            <h3>Cookies and Tracking</h3>
            <p>We use Google Analytics and Google AdSense to optimize performance and deliver tailored ads.</p>
        </div>
    </main>
    <footer><div class="container"><p>&copy; {{ current_year }} {{ site_name }}</p></div></footer>
</body>
</html>"""

TERMS_TMPL = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Terms of Service - {{ site_name }}</title>
    <link rel="stylesheet" href="{{ base_path }}/static/style.css">
</head>
<body>
    <header><div class="container"><h1><a href="{{ base_path }}/">{{ site_name }}</a></h1></div></header>
    <main class="container">
        <h2>Terms of Service</h2>
        <p>Last updated: {{ current_date }}</p>
        <p>By browsing this website, you agree to our standard terms of service.</p>
    </main>
    <footer><div class="container"><p>&copy; {{ current_year }} {{ site_name }}</p></div></footer>
</body>
</html>"""

CONTACT_TMPL = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Contact Us - {{ site_name }}</title>
    <link rel="stylesheet" href="{{ base_path }}/static/style.css">
</head>
<body>
    <header><div class="container"><h1><a href="{{ base_path }}/">{{ site_name }}</a></h1></div></header>
    <main class="container">
        <h2>Contact</h2>
        <p>For inquiries, email us directly at <a href="mailto:aiblogauto@gmail.com">aiblogauto@gmail.com</a>.</p>
    </main>
    <footer><div class="container"><p>&copy; {{ current_year }} {{ site_name }}</p></div></footer>
</body>
</html>"""

NOT_FOUND_TMPL = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Page Not Found — {{ site_name }}</title>
    <link rel="stylesheet" href="{{ base_path }}/static/style.css">
</head>
<body>
    <div class="fallback">
        <p>Taking you to the homepage&hellip;</p>
        <p><a href="{{ base_path }}/">Click here if you are not redirected automatically</a></p>
    </div>
    <script>window.location.replace('{{ base_path }}/');</script>
</body>
</html>"""

DMCA_TMPL = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DMCA Policy — {{ site_name }}</title>
    <link rel="stylesheet" href="{{ base_path }}/static/style.css">
</head>
<body>
    <main class="container"><h2>DMCA Policy</h2></main>
    <footer><div class="container"><p>&copy; {{ current_year }} {{ site_name }}</p></div></footer>
</body>
</html>"""

AI_DISCLOSURE_TMPL = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Content Policy — {{ site_name }}</title>
    <link rel="stylesheet" href="{{ base_path }}/static/style.css">
</head>
<body>
    <main class="container"><h2>AI Content Policy</h2></main>
    <footer><div class="container"><p>&copy; {{ current_year }} {{ site_name }}</p></div></footer>
</body>
</html>"""


class StaticSiteGenerator:
    def __init__(self, blog_system):
        self.blog_system = blog_system
        self.seo = SEOOptimizer(blog_system.config)
        self.visibility = VisibilityAutomator(blog_system.config)
        self.monetization = MonetizationManager(blog_system.config)
        self.templates = self._load_templates()

    def _write_html(self, path: Path, html_content: str, base_path: str = ""):
        """
        Centralized Pythonic Way to inject assets like GDPR consent globally.
        Intercepts raw HTML outputs across all engines (Jinja & f-strings) 
        and updates them inline seamlessly before saving.
        """
        consent_script = f'<script defer src="{base_path}/static/consent.js"></script>\n'

        # Inject just before closing body tag if present and not already registered
        if "</body>" in html_content and "consent.js" not in html_content:
            html_content = html_content.replace(
                "</body>", f"{consent_script}</body>")

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def _load_templates(self):
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

    def generate_site(self):
        print("Generating static site...")
        posts = self._get_all_posts()

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

    def _generate_robots_txt(self):
        config = self.blog_system.config
        base_url = config.get('base_url', '')
        content = f"""User-agent: *
Allow: /
Disallow: /static/admin/

User-agent: Mediapartners-Google
Allow: /

User-agent: Googlebot
Allow: /
Crawl-delay: 1

User-agent: Googlebot-Image
Allow: /static/

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

    def _generate_homepage(self, posts: List[BlogPost]):
        config = self.blog_system.config
        HOMEPAGE_SSR_LIMIT = 24

        posts_data = []
        for p in posts:
            post_dict = p.to_dict()
            post_dict['display_date'] = self._format_display_date(p.created_at)
            post_dict['short_tags'] = sorted(p.tags, key=len)[:3]
            post_dict['reading_time'] = self._reading_time_minutes(p.content)
            post_dict['meta_description'] = _safe_excerpt(
                p.meta_description, p.content, p.title)
            post_dict.pop('content', None)
            posts_data.append(post_dict)

        context = {
            'site_name': config.get('site_name', 'Tech Blog'),
            'site_description': config.get('site_description', 'An AI-powered blog'),
            'base_path': config.get('base_path', ''),
            'base_url': config.get('base_url', ''),
            'posts': posts_data[:HOMEPAGE_SSR_LIMIT],
            'posts_per_page': HOMEPAGE_SSR_LIMIT,
            'total_posts': len(posts_data),
            'current_year': datetime.now().year,
            'social_links': config.get('social_accounts', {}),
            'global_meta_tags': self.seo.generate_global_meta_tags(),
            'homepage_meta_tags': self.seo.generate_homepage_meta_tags(),
            'organization_schema': self.seo.generate_organization_schema(),
            'website_schema': self.seo.generate_website_schema()
        }
        html = self.templates['index'].render(**context)
        self._write_html(Path("./docs/index.html"), html,
                         config.get('base_path', ''))

    def _format_display_date(self, iso_date: str) -> str:
        try:
            dt = datetime.fromisoformat(iso_date.replace('Z', '+00:00'))
            return dt.strftime('%-d %B %Y')
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
        self._write_html(Path("./docs/404.html"), html,
                         config.get('base_path', ''))

    def _generate_dmca_page(self):
        config = self.blog_system.config
        page_dir = Path("./docs/dmca")
        context = {
            'site_name': config.get('site_name', 'Tech Blog'),
            'base_path': config.get('base_path', ''),
            'base_url': config.get('base_url', ''),
            'current_year': datetime.now().year,
            'current_date': datetime.now().strftime('%B %d, %Y'),
            'global_meta_tags': self.seo.generate_global_meta_tags(),
        }
        html = self.templates['dmca'].render(**context)
        self._write_html(page_dir / "index.html", html,
                         config.get('base_path', ''))

    def _generate_ai_disclosure_page(self):
        config = self.blog_system.config
        page_dir = Path("./docs/ai-content-policy")
        context = {
            'site_name': config.get('site_name', 'Tech Blog'),
            'base_path': config.get('base_path', ''),
            'base_url': config.get('base_url', ''),
            'current_year': datetime.now().year,
            'current_date': datetime.now().strftime('%B %d, %Y'),
            'global_meta_tags': self.seo.generate_global_meta_tags(),
        }
        html = self.templates['ai_disclosure'].render(**context)
        self._write_html(page_dir / "index.html", html,
                         config.get('base_path', ''))

    def _generate_pwa_files(self):
        import time
        import re
        sw_src = Path("sw.js")
        if sw_src.exists():
            sw_text = sw_src.read_text(encoding="utf-8")
            new_version = f"v{int(time.time())}"
            patched, n = re.subn(
                r"(const\s+CACHE_VERSION\s*=\s*')[^']*(')",
                lambda m: m.group(1) + new_version + m.group(2),
                sw_text,
            )
            Path("./docs/sw.js").write_text(patched, encoding="utf-8")

        for src, dst in [("offline.html", "./docs/offline.html"),
                         ("manifest.json", "./docs/manifest.json")]:
            if Path(src).exists():
                import shutil as _shutil
                _shutil.copy2(src, dst)

    def _generate_article_schema(self, post, base_url: str) -> str:
        word_count = len(post.content.split())
        reading_time = max(1, round(word_count / 200))
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
                "author": {
                    "@type": "Person",
                    "name": "Kubai Kevin",
                    "url": f"{base_url}/about/"
                },
                "publisher": {
                    "@type": "Organization",
                    "name": "Tech Blog",
                    "url": base_url
                }
            }
        ]
        import json as _json
        return f'<script type="application/ld+json">\n{_json.dumps({"@context": "https://schema.org", "@graph": schemas}, indent=2)}\n</script>'

    def _generate_security_headers(self):
        headers_content = "/*\nX-Frame-Options: SAMEORIGIN\nX-Content-Type-Options: nosniff\n"
        with open("./docs/_headers", "w", encoding="utf-8") as f:
            f.write(headers_content)

    def _generate_privacy_consent_banner(self):
        # Generates standard vanilla cookies module logic inside docs/static/consent.js
        pass

    def _generate_post_pages(self, posts: List[BlogPost]):
        config = self.blog_system.config
        for post in posts:
            post_dir = Path("./docs") / post.slug
            markdown_converter = md.Markdown(
                extensions=['extra', 'fenced_code', 'toc'])
            content_html = markdown_converter.convert(post.content)

            post_dict = post.to_dict()
            post_dict['content_html'] = content_html
            post_dict['display_date'] = self._format_display_date(
                post.created_at)
            post_dict['reading_time'] = self._reading_time_minutes(
                post.content)
            post_dict['word_count'] = len(post.content.split())

            updated_normalized = _normalize_iso_date(post.updated_at)
            post_dict['last_updated_iso'] = updated_normalized.split(
                'T')[0] if 'T' in updated_normalized else updated_normalized

            context = {
                'site_name': config.get('site_name', 'Tech Blog'),
                'base_path': config.get('base_path', ''),
                'base_url': config.get('base_url', ''),
                'post': post_dict,
                'current_year': datetime.now().year,
                'global_meta_tags': self.seo.generate_global_meta_tags(),
                'meta_tags': self.seo.generate_meta_tags(post),
                'article_schema': self._generate_article_schema(post, config.get('base_url', ''))
            }
            html = self.templates['post'].render(**context)
            self._write_html(post_dir / "index.html", html,
                             config.get('base_path', ''))

    def _generate_static_pages(self, posts: List[BlogPost] = None):
        config = self.blog_system.config
        pages = {
            'about': ('about', {'posts': posts or []}),
            'contact': ('contact', {}),
            'privacy-policy': ('privacy_policy', {'current_date': datetime.now().strftime('%B %d, %Y')}),
            'terms-of-service': ('terms_of_service', {'current_date': datetime.now().strftime('%B %d, %Y')}),
        }
        for dir_name, (template_name, extra_context) in pages.items():
            page_dir = Path("./docs") / dir_name
            context = {
                'site_name': config.get('site_name', 'Tech Blog'),
                'base_path': config.get('base_path', ''),
                'base_url': config.get('base_url', ''),
                'current_year': datetime.now().year,
                'global_meta_tags': self.seo.generate_global_meta_tags(),
                **extra_context
            }
            html = self.templates[template_name].render(**context)
            self._write_html(page_dir / "index.html", html,
                             config.get('base_path', ''))

    def _generate_author_page(self, posts: List[BlogPost]):
        config = self.blog_system.config
        base_url = config.get('base_url', '')
        base_path = config.get('base_path', '')
        site_name = config.get('site_name', 'Tech Blog')

        author_dir = Path("./docs/author/kubai-kevin")
        posts_html = ""
        if posts:
            items = "\n".join(
                f'<li><a href="{base_path}/{p.slug}/">{p.title}</a></li>' for p in posts[:20]
            )
            posts_html = f"<h2>Recent Articles</h2><ul>{items}</ul>"

        html = AUTHOR_PAGE_TEMPLATE.format(
            site_name=site_name,
            base_url=base_url,
            base_path=base_path,
            year=datetime.now().year,
            posts_html=posts_html
        )
        self._write_html(author_dir / "index.html", html, base_path)

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
        for tag, tag_posts in qualifying.items():
            tag_slug = tag.replace(' ', '-')
            tag_dir = tags_dir / tag_slug
            robots_directive = "index, follow" if len(
                tag_posts) >= 5 else "noindex, follow"

            posts_data = []
            for p in sorted(tag_posts, key=lambda x: x.created_at, reverse=True):
                d = p.to_dict()
                d['display_date'] = self._format_display_date(p.created_at)
                d['meta_description'] = _safe_excerpt(
                    p.meta_description, p.content, p.title)
                d['reading_time'] = self._reading_time_minutes(p.content)
                posts_data.append(d)

            tag_title = tag.title()
            html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{tag_title} Articles — {site_name}</title>
    <meta name="robots" content="{robots_directive}">
    <link rel="stylesheet" href="{base_path}/static/style.css">
</head>
<body>
    <main class="container">
        <h2>Articles tagged <em>{tag_title}</em></h2>
        <div class="post-grid">
        ''' + ''.join(f'<a href="{base_path}/{p["slug"]}/"><h3>{p["title"]}</h3></a>' for p in posts_data) + f'''
        </div>
    </main>
</body>
</html>'''
            self._write_html(tag_dir / "index.html", html, base_path)

    def _generate_rss_feed(self, posts): pass
    def _generate_sitemap(self, posts): pass
    def _generate_posts_json(self, posts): pass
