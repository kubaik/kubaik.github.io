import json
import markdown as md
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from jinja2 import Template, Environment, BaseLoader

from blog_post import BlogPost
from seo_optimizer import SEOOptimizer
from visibility_automator import VisibilityAutomator
from monetization_manager import MonetizationManager


# ─────────────────────────────────────────────────────────────────
# FIX: module-level excerpt helper used by templates + Python code
# ─────────────────────────────────────────────────────────────────

def _safe_excerpt(meta_description: str, content: str, title: str = "",
                  max_len: int = 155) -> str:
    """
    Return a non-empty excerpt string for display.
    Priority: meta_description → first meaningful sentence in content → title.
    """
    import re

    desc = (meta_description or "").strip()
    if desc:
        return desc

    # Strip markdown and extract plain text
    text = re.sub(r"```[\s\S]*?```", " ", content or "")
    text = re.sub(r"`[^`]+`", " ", text)
    text = re.sub(r"#{1,6}\s+", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"[*_]{1,3}", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    sentences = re.split(r'(?<=[.!?])\s+', text)
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) >= 40:
            if len(sentence) > max_len:
                sentence = sentence[:max_len].rsplit(
                    " ", 1)[0].rstrip(".,;:") + "…"
            return sentence

    # Last resort: use first max_len chars of plain text or the title
    fallback = text[:max_len] if text else (title or "")
    if len(fallback) == max_len:
        fallback = fallback.rsplit(" ", 1)[0] + "…"
    return fallback


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
        self._generate_static_pages()
        self._generate_rss_feed(posts)
        self._generate_sitemap(posts)
        self._generate_posts_json(posts)
        self._generate_robots_txt()
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
        config = self.blog_system.config
        base_url = config.get('base_url', '')
        content = f"""User-agent: *
Allow: /
Disallow: /static/admin/
Disallow: /*.json$

Crawl-delay: 1

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
        posts_data = []
        for p in posts:
            post_dict = p.to_dict()
            post_dict['display_date'] = self._format_display_date(p.created_at)
            post_dict['short_tags'] = sorted(p.tags, key=len)[:3]
            post_dict['reading_time'] = self._reading_time_minutes(p.content)
            # ── FIX: guarantee excerpt is never empty on the homepage card ──
            post_dict['meta_description'] = _safe_excerpt(
                p.meta_description, p.content, p.title
            )
            posts_data.append(post_dict)
        context = {
            'site_name': config.get('site_name', 'Tech Blog'),
            'site_description': config.get('site_description', 'An AI-powered blog'),
            'base_path': config.get('base_path', ''),
            'base_url': config.get('base_url', ''),
            'posts': posts_data,
            'posts_per_page': 10,
            'current_year': datetime.now().year,
            'social_links': config.get('social_accounts', {}),
            'global_meta_tags': self.seo.generate_global_meta_tags(),
            'homepage_meta_tags': self.seo.generate_homepage_meta_tags(),
            'organization_schema': self.seo.generate_organization_schema(),
            'website_schema': self.seo.generate_website_schema()
        }
        html = self.templates['index'].render(**context)
        output_file = Path("./docs/index.html")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"Generated homepage with {len(posts)} posts")

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
            # ── FIX: guarantee meta_description is populated on post pages ──
            post_dict['meta_description'] = _safe_excerpt(
                post.meta_description, post.content, post.title
            )
            context = {
                'site_name': config.get('site_name', 'Tech Blog'),
                'base_path': config.get('base_path', ''),
                'base_url': config.get('base_url', ''),
                'post': post_dict,
                'related_posts': related,
                'current_year': datetime.now().year,
                'global_meta_tags': self.seo.generate_global_meta_tags(),
                'meta_tags': self.seo.generate_meta_tags(post),
                'structured_data': self.seo.generate_structured_data(post),
                'header_ad': self.seo.generate_adsense_ad('header'),
                'middle_ad': self.seo.generate_adsense_ad('middle'),
                'footer_ad': self.seo.generate_adsense_ad('footer')
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
            # ── FIX: related-post cards also get a guaranteed excerpt ──
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

    def _generate_static_pages(self):
        config = self.blog_system.config
        pages = {
            'about': ('about', {'topics': config.get('content_topics', [])[:]}),
            'contact': ('contact', {}),
            'privacy-policy': ('privacy_policy', {'current_date': datetime.now().strftime('%B %d, %Y')}),
            'terms-of-service': ('terms_of_service', {'current_date': datetime.now().strftime('%B %d, %Y')})
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

    def _generate_rss_feed(self, posts: List[BlogPost]):
        config = self.blog_system.config
        base_url = config.get('base_url', '')
        rss_items = []
        for post in posts[:20]:
            # ── FIX: RSS description also gets the safe excerpt ──
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
            f'<url><loc>{base_url}/</loc><lastmod>{today}</lastmod><priority>1.0</priority></url>',
            f'<url><loc>{base_url}/about/</loc><lastmod>{today}</lastmod><priority>0.8</priority></url>',
            f'<url><loc>{base_url}/contact/</loc><lastmod>{today}</lastmod><priority>0.7</priority></url>',
            f'<url><loc>{base_url}/privacy-policy/</loc><lastmod>{today}</lastmod><priority>0.5</priority></url>',
            f'<url><loc>{base_url}/terms-of-service/</loc><lastmod>{today}</lastmod><priority>0.5</priority></url>',
        ]
        for post in posts:
            last_mod = post.updated_at.split(
                'T')[0] if 'T' in post.updated_at else post.updated_at
            urls.append(
                f'<url><loc>{base_url}/{post.slug}/</loc><lastmod>{last_mod}</lastmod><priority>0.9</priority></url>'
            )
        sitemap = f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  {chr(10).join(urls)}
</urlset>"""
        with open("./docs/sitemap.xml", 'w', encoding='utf-8') as f:
            f.write(sitemap)
        print("Generated sitemap")

    def _generate_posts_json(self, posts: List[BlogPost]):
        posts_data = []
        for p in posts:
            d = p.to_dict()
            d['reading_time'] = self._reading_time_minutes(p.content)
            # ── FIX: posts.json (used by JS infinite scroll) also gets safe excerpt ──
            d['meta_description'] = _safe_excerpt(
                p.meta_description, p.content, p.title
            )
            posts_data.append(d)
        with open("./docs/posts.json", 'w', encoding='utf-8') as f:
            json.dump(posts_data, f, indent=2)
        print("Generated posts.json")

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
    {{ global_meta_tags | safe }}
    {{ meta_tags | safe }}
    {{ structured_data | safe }}
    <link rel="stylesheet" href="{{ base_path }}/static/style.css">
    <link rel="stylesheet" href="{{ base_path }}/static/enhanced-blog-post-styles.css">
    <script defer src="{{ base_path }}/static/code_runner.js"></script>
</head>
<body>
    {{ header_ad | safe }}
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

        <nav class="breadcrumb" aria-label="Breadcrumb">
            <a href="{{ base_path }}/">Home</a>
            <span>›</span>
            <span>{{ post.title }}</span>
        </nav>

        <article class="blog-post" itemscope itemtype="https://schema.org/Article">
            <header class="post-header">
                <h1 itemprop="headline">{{ post.title }}</h1>

                {# ── FIX: show meta_description as a lead paragraph on the post page ── #}
                {% if post.meta_description %}
                <p class="post-lead">{{ post.meta_description }}</p>
                {% endif %}

                <div class="post-meta-row">
                    <span>
                        <time datetime="{{ post.created_at }}" itemprop="datePublished">
                            {{ post.display_date }}
                        </time>
                    </span>
                    {% if post.updated_date and post.updated_date != post.display_date %}
                    <span>
                        Updated
                        <time datetime="{{ post.updated_at }}" itemprop="dateModified">
                            {{ post.updated_date }}
                        </time>
                    </span>
                    {% endif %}
                    {% if post.reading_time %}
                    <span>{{ post.reading_time }} min read</span>
                    {% endif %}
                    {% if post.word_count %}
                    <span>{{ post.word_count | int }} words</span>
                    {% endif %}
                </div>

                {% if post.tags %}
                <div class="tags">
                    {% for tag in post.tags[:6] %}
                    <span class="tag">{{ tag }}</span>
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
                </div>
            </div>

            <div class="post-content" itemprop="articleBody">
                {{ post.content_html | safe }}
                {{ middle_ad | safe }}
            </div>

            {% if post.affiliate_links %}
            <div class="affiliate-disclaimer">
                <p><em>This post contains affiliate links. We may earn a commission if you make a purchase through these links, at no additional cost to you.</em></p>
            </div>
            {% endif %}

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
    {{ footer_ad | safe }}
    <footer>
        <div class="container">
            <p>&copy; {{ current_year }} {{ site_name }}. Written by
               <a href="{{ base_path }}/about/">Kubai Kevin</a>.
               Content reviewed for accuracy before publishing.</p>
        </div>
    </footer>
    <script src="{{ base_path }}/static/navigation.js"></script>
</body>
</html>"""

    INDEX_TMPL = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ site_name }}</title>
    <meta name="description" content="{{ site_description }}">
    {{ global_meta_tags | safe }}
    {{ homepage_meta_tags | safe }}
    {{ organization_schema | safe }}
    {{ website_schema | safe }}
    <link rel="stylesheet" href="{{ base_path }}/static/style.css">
    <link rel="alternate" type="application/rss+xml" title="{{ site_name }}" href="{{ base_path }}/rss.xml">
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
        .post-reading-time { font-size: 0.8rem; color: #888; margin-top: 4px; }
        .post-card--entering { opacity: 0; transform: translateY(8px); transition: opacity 0.25s ease, transform 0.25s ease; }
        .loading-spinner { display: flex; align-items: center; justify-content: center; gap: 10px; padding: 1.5rem; }
        .spinner { width: 20px; height: 20px; border: 2px solid #e0e0e0; border-top-color: #6366f1; border-radius: 50%; animation: spin 0.7s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .back-to-top { position: fixed; bottom: 2rem; right: 2rem; width: 40px; height: 40px; border-radius: 50%; background: #6366f1; color: #fff; border: none; cursor: pointer; font-size: 1.2rem; display: flex; align-items: center; justify-content: center; box-shadow: 0 2px 8px rgba(0,0,0,0.2); }
        .post-lead { font-size: 1.05rem; color: #555; line-height: 1.6; margin: 0.25rem 0 1rem; font-style: italic; }
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
        <div class="hero">
            <h2>Welcome to {{ site_name }}</h2>
            <p>{{ site_description }}</p>
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
                {% for post in posts[:posts_per_page] %}
                <a class="post-card" href="{{ base_path }}/{{ post.slug }}/"
                   data-title="{{ post.title | e }}"
                   data-description="{{ post.meta_description | e }}"
                   data-tags="{{ post.tags | join(',') | e }}">
                    <h3>{{ post.title }}</h3>
                    {# ── FIX: meta_description is guaranteed non-empty by _generate_homepage ── #}
                    <p class="post-excerpt">{{ post.meta_description }}</p>
                    {% if post.reading_time %}
                    <p class="post-reading-time">{{ post.reading_time }} min read</p>
                    {% endif %}
                    {% if post.tags %}
                    <div class="tags">
                        {% for tag in post.short_tags %}
                        <span class="tag">{{ tag }}</span>
                        {% endfor %}
                    </div>
                    {% endif %}
                </a>
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

            // ── FIX: use meta_description from posts.json (already safe-excerpted) ──
            var excerpt = (post.meta_description || '').trim();
            if (!excerpt && post.content) {
                // client-side last resort: first 155 chars of content
                excerpt = post.content.replace(/[#*`>\[\]]/g, '').trim().slice(0, 155);
                if (excerpt.length === 155) excerpt = excerpt.slice(0, excerpt.lastIndexOf(' ')) + '…';
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
            var tags = post.tags || [];
            if (tags.length) {
                var div       = document.createElement('div');
                div.className = 'tags';
                tags.slice().sort(function (x, y) { return x.length - y.length; })
                    .slice(0, 3)
                    .forEach(function (t) {
                        var sp         = document.createElement('span');
                        sp.className   = 'tag';
                        sp.textContent = t;
                        div.appendChild(sp);
                    });
                a.appendChild(div);
            }
            return a;
        }

        function loadNextPage() {
            isLoading = true;
            if (loadingSpinner) loadingSpinner.style.display = 'flex';
            setTimeout(function () {
                var slice    = fullPosts.slice(loadedCount, loadedCount + PAGE_SIZE);
                var fragment = document.createDocumentFragment();
                var newCards = [];
                slice.forEach(function (post) {
                    var card = buildCard(post);
                    card.classList.add('post-card--entering');
                    fragment.appendChild(card);
                    newCards.push(card);
                });
                postsContainer.appendChild(fragment);
                loadedCount += slice.length;
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
                if (loadedCount < fullPosts.length) startObserver();
                if (searchMode && searchInput && searchInput.value.trim()) runSearch(searchInput.value.trim());
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
            if (jsonReady && loadedCount < fullPosts.length) startObserver();
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
</body>
</html>"""

    ABOUT_TMPL = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>About - {{ site_name }}</title>
    <meta name="description" content="Kubai Kevin is a software developer based in Nairobi writing about AI, backend systems, and developer careers.">
    {{ global_meta_tags | safe }}
    <link rel="stylesheet" href="../static/style.css">
    <link rel="canonical" href="{{ base_url }}/about/">
    <style>
        .about-section{background:#f8f9fa;padding:1.5rem 2rem;margin-bottom:1.5rem;border-radius:8px;border-left:4px solid #6366f1}
        .about-section h2{color:#333;margin-top:0;margin-bottom:1rem}
        .feature-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:1.5rem;margin:1.5rem 0}
        .feature-card{background:white;padding:1.5rem;border-radius:8px;border:2px solid #e0e0e0}
        .feature-card h3{color:#6366f1;margin-top:0}
        .author-card{background:white;padding:1.5rem 2rem;border-radius:8px;border:2px solid #6366f1;margin-bottom:1.5rem;display:flex;gap:1.5rem;align-items:flex-start}
        .author-avatar-lg{width:72px;height:72px;border-radius:50%;background:linear-gradient(135deg,#667eea,#764ba2);display:flex;align-items:center;justify-content:center;color:white;font-weight:700;font-size:1.6rem;flex-shrink:0}
        .author-card h3{margin-top:0;color:#333;font-size:1.2rem}
        .author-card .credentials{color:#6366f1;font-size:0.88rem;margin-bottom:0.75rem;font-weight:600}
        .stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:1rem;margin:1.5rem 0}
        .stat-card{background:#f0f4ff;padding:1.5rem;border-radius:8px;text-align:center;border:2px solid #6366f1}
        .stat-number{font-size:2rem;font-weight:bold;color:#6366f1;display:block;margin-bottom:0.25rem}
        .cta-box{background:#fff3cd;border-left:4px solid #ffc107;padding:1.5rem;border-radius:8px;margin:1.5rem 0;text-align:center}
        .cta-button{display:inline-block;background:linear-gradient(135deg,#667eea,#764ba2);color:white;padding:0.8rem 1.8rem;border-radius:8px;text-decoration:none;font-weight:600;margin-top:0.8rem}
        .process-step{display:flex;gap:1rem;margin-bottom:1rem;align-items:flex-start}
        .step-num{background:#6366f1;color:white;width:28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:0.85rem;flex-shrink:0;margin-top:2px}
        .author-avatar-lg {width: 100px;height: 100px;border-radius: 50%;overflow: hidden;flex-shrink: 0;}
        .author-photo {width: 100%;height: 100%;object-fit: cover;}
        .paragraph-spacing {margin-bottom: 16px;}
        .linkedin-link {font-weight: 600;font-size: 1.1rem;text-decoration: none;}
        .linkedin-link:hover {text-decoration: underline;}
    </style>
</head>
<body>
    <header><div class="container">
        <h1><a href="../">{{ site_name }}</a></h1>
        <nav><a href="../">Home</a><a href="../about/">About</a><a href="../contact/">Contact</a>
        <a href="../privacy-policy/">Privacy Policy</a><a href="../terms-of-service/">Terms of Service</a></nav>
    </div></header>
    <main class="container">
        <div class="hero"><h2>About {{ site_name }}</h2><p>Practical technology writing from a working developer</p></div>
        <article class="page-content" itemscope itemtype="https://schema.org/Person">

            <div class="about-section">
                <h2>The Author</h2>
                <div class="author-card">
                    <!-- Profile Image -->
                    <div class="author-avatar-lg">
                        <img src="../static/photo.jpg" 
                            alt="Kubai Kevin" 
                            class="author-photo">
                    </div>

                    <div>
                        <h3 itemprop="name">Kubai Kevin</h3>
                        <p class="credentials" itemprop="jobTitle">
                            Software Developer · Nairobi, Kenya
                        </p>

                        <p itemprop="description" class="paragraph-spacing" >
                                Software Engineer building production systems since 2014, with experience in the financial services industry. 
                                I specialize in Python backends, Node.js (TypeScript), Android (Java/Kotlin), and AWS serverless architectures.
                            </p>

                            <p class="paragraph-spacing >
                                My work focuses on API design, automation, and integrating AI/LLMs into practical, maintainable workflows, 
                                with an emphasis on performance, cost, and scalability.
                            </p>

                            <p class="paragraph-spacing >
                                I write about real-world engineering tradeoffs and what actually works in production.
                            </p>
                            <p>
                            <a href="https://www.linkedin.com/in/kevin-kubai-22b61b37/" target="_blank" rel="noopener noreferrer" class="linkedin-link">
                                View full experience on LinkedIn →
                            </a>
                        </p>
                    </div>
                </div>
            </div>

            <div class="about-section">
                <h2>What This Blog Covers</h2>
                <p>The topics on this blog fall into a few areas I work with regularly and care about enough to have strong opinions on:</p>
                <div class="feature-grid">
                    <div class="feature-card"><h3>AI &amp; LLMs</h3><p>Practical applications of language models, prompt engineering, retrieval systems, and what actually works beyond the demos. I am particularly interested in the gap between benchmark performance and production reliability.</p></div>
                    <div class="feature-card"><h3>Backend Systems</h3><p>APIs, databases, queues, and the distributed systems problems that come up in real products. I focus on the decisions that matter at 10,000 req/min, not just the ones that matter at 10 million.</p></div>
                    <div class="feature-card"><h3>Developer Tools</h3><p>CI/CD, observability, and developer experience. What saves real time and what sounds good on paper but adds friction. I try to include concrete before/after numbers wherever possible.</p></div>
                    <div class="feature-card"><h3>Tech Careers</h3><p>How the industry is changing, what skills matter in 2026, and honest analysis of the trends that will and will not last. I pay particular attention to what is happening in African tech markets.</p></div>
                </div>
            </div>

            <div class="about-section">
                <h2>How Articles Are Written</h2>
                <p>I use a combination of personal experience and AI-assisted drafting. Here is the exact process:</p>
                <div class="process-step"><div class="step-num">1</div><p><strong>Topic selection</strong> — I pick topics I have direct experience with or have researched recently for work. I do not write about things I cannot personally verify.</p></div>
                <div class="process-step"><div class="step-num">2</div><p><strong>AI-assisted drafting</strong> — I use language models (Claude, GPT-4o) to generate a first draft based on my specifications. The prompt I use requires specific tool names, version numbers, benchmarks, and honest tradeoffs.</p></div>
                <div class="process-step"><div class="step-num">3</div><p><strong>Review and editing</strong> — I read every draft and correct errors. If I cannot verify a specific claim against documentation or my own experience, I remove it. Code examples are tested locally where practical.</p></div>
                <div class="process-step"><div class="step-num">4</div><p><strong>My take section</strong> — Every article includes a section with my personal opinion on the topic — something that reflects what I actually believe based on production experience, not just what the documentation says.</p></div>
                <p style="margin-top:1rem">If you find a factual error, please <a href="../contact/">contact me</a>. I take corrections seriously and update articles when errors are found.</p>
            </div>

            <div class="stat-grid">
                <div class="stat-card"><span class="stat-number">{{ posts|length }}</span><span>Posts published</span></div>
                <div class="stat-card"><span class="stat-number">2014</span><span>Started coding professionally</span></div>
                <div class="stat-card"><span class="stat-number">Free</span><span>Always and forever</span></div>
            </div>

            <div class="cta-box">
                <h3>Questions or Corrections?</h3>
                <p>I respond to every email. Factual corrections are especially welcome.</p>
                <a href="../contact/" class="cta-button">Contact Me</a>
            </div>
        </article>
    </main>
    <footer><div class="container"><p>&copy; {{ current_year }} {{ site_name }} · Written by Kubai Kevin</p></div></footer>
    <script src="../static/navigation.js"></script>
</body>
</html>"""

    PRIVACY_TMPL = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Privacy Policy - {{ site_name }}</title>
    <meta name="description" content="Privacy Policy for {{ site_name }}">
    {{ global_meta_tags | safe }}
    <link rel="stylesheet" href="../static/style.css">
    <link rel="canonical" href="{{ base_url }}/privacy-policy/">
    <style>
        .privacy-section{background:#f8f9fa;padding:1.5rem;margin-bottom:1.5rem;border-radius:8px;border-left:4px solid #6366f1}
        .privacy-section h3{color:#333;margin-top:0}
        .important-notice{background:#fff3cd;border-left:4px solid #ffc107;padding:1rem 1.5rem;margin:1.5rem 0;border-radius:4px}
        table{width:100%;border-collapse:collapse;background:white}
        th,td{padding:0.8rem;text-align:left;border-bottom:1px solid #dee2e6}
        th{background:#f8f9fa;font-weight:600}
        .highlight-box{background:linear-gradient(135deg,#667eea,#764ba2);color:white;padding:1.5rem;border-radius:8px;margin:1.5rem 0}
        .highlight-box h3{margin-top:0;color:white}
    </style>
</head>
<body>
    <header><div class="container">
        <h1><a href="../">{{ site_name }}</a></h1>
        <nav><a href="../">Home</a><a href="../about/">About</a><a href="../contact/">Contact</a>
        <a href="../privacy-policy/">Privacy Policy</a><a href="../terms-of-service/">Terms of Service</a></nav>
    </div></header>
    <main class="container">
        <div class="hero"><h2>Privacy Policy</h2><p>How we protect and handle your information</p></div>
        <article class="page-content">
            <div class="privacy-section"><h3>1. Introduction</h3>
                <p><strong>{{ site_name }}</strong> is committed to protecting your privacy. By accessing this site, you agree to this Privacy Policy.</p></div>
            <div class="privacy-section"><h3>2. Information We Collect</h3>
                <ul>
                    <li><strong>Contact information</strong> when you contact us: name, email, message</li>
                    <li><strong>Usage data</strong> via Google Analytics: pages visited, time on site, browser type</li>
                    <li><strong>Cookie data</strong> from Google Analytics and Google AdSense</li>
                </ul></div>
            <div class="privacy-section"><h3>3. How We Use Information</h3>
                <ul>
                    <li>To respond to your inquiries</li>
                    <li>To improve content based on what readers find useful</li>
                    <li>To serve relevant advertisements via Google AdSense</li>
                </ul></div>
            <div class="privacy-section"><h3>4. Cookies</h3>
                <table>
                    <thead><tr><th>Type</th><th>Purpose</th><th>Duration</th></tr></thead>
                    <tbody>
                        <tr><td>Analytics</td><td>Google Analytics usage tracking</td><td>Up to 2 years</td></tr>
                        <tr><td>Advertising</td><td>Google AdSense ad targeting</td><td>Up to 1 year</td></tr>
                    </tbody>
                </table>
                <p>You can disable cookies in your browser settings. Google's privacy policy: <a href="https://policies.google.com/privacy" target="_blank" rel="noopener">policies.google.com/privacy</a></p></div>
            <div class="privacy-section"><h3>5. Third Parties</h3>
                <div class="important-notice"><p>We do not sell your personal information. We use Google Analytics and Google AdSense, which have their own privacy policies.</p></div></div>
            <div class="highlight-box"><h3>6. Contact</h3>
                <p>Email: <a href="mailto:aiblogauto@gmail.com" style="color:white;text-decoration:underline;">aiblogauto@gmail.com</a></p></div>
            <p><strong>Last updated:</strong> {{ current_date }}</p>
        </article>
    </main>
    <footer><div class="container"><p>&copy; {{ current_year }} {{ site_name }}.</p></div></footer>
    <script src="../static/navigation.js"></script>
</body>
</html>"""

    TERMS_TMPL = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Terms of Service - {{ site_name }}</title>
    <meta name="description" content="Terms of Service for {{ site_name }}">
    {{ global_meta_tags | safe }}
    <link rel="stylesheet" href="../static/style.css">
    <link rel="canonical" href="{{ base_url }}/terms-of-service/">
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
        <h1><a href="../">{{ site_name }}</a></h1>
        <nav><a href="../">Home</a><a href="../about/">About</a><a href="../contact/">Contact</a>
        <a href="../privacy-policy/">Privacy Policy</a><a href="../terms-of-service/">Terms of Service</a></nav>
    </div></header>
    <main class="container">
        <div class="hero"><h2>Terms of Service</h2><p>Please read these terms carefully before using our site</p></div>
        <article class="page-content">
            <div class="terms-section"><h3>1. Acceptance</h3>
                <p>By accessing {{ site_name }}, you agree to these Terms and our Privacy Policy.</p></div>
            <div class="terms-section"><h3>2. AI-Assisted Content</h3>
                <div class="warning-box"><p>Some content on this site is drafted with AI assistance and reviewed for accuracy by the author. Always verify technical information independently before using it in production.</p></div></div>
            <div class="terms-section"><h3>3. Affiliate Disclosure</h3>
                <p>This site participates in affiliate programmes. Purchases made through affiliate links may earn us a commission at no additional cost to you.</p></div>
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
    <script src="../static/navigation.js"></script>
</body>
</html>"""

    CONTACT_TMPL = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Contact - {{ site_name }}</title>
    <meta name="description" content="Contact Kubai Kevin at {{ site_name }}">
    {{ global_meta_tags | safe }}
    <link rel="stylesheet" href="../static/style.css">
    <link rel="canonical" href="{{ base_url }}/contact/">
    <style>
        .contact-method{background:#f8f9fa;padding:1.5rem;margin-bottom:1.5rem;border-radius:8px;border-left:4px solid #6366f1}
        .contact-method h3{color:#333;margin-top:0}
        .contact-email{color:#6366f1;font-weight:600;font-size:1.1rem}
        .contact-footer{background:linear-gradient(135deg,#667eea,#764ba2);color:white;padding:1.5rem;border-radius:8px;margin-top:1.5rem}
    </style>
</head>
<body>
    <header><div class="container">
        <h1><a href="../">{{ site_name }}</a></h1>
        <nav><a href="../">Home</a><a href="../about/">About</a><a href="../contact/">Contact</a>
        <a href="../privacy-policy/">Privacy Policy</a><a href="../terms-of-service/">Terms of Service</a></nav>
    </div></header>
    <main class="container">
        <div class="hero"><h2>Contact</h2><p>Get in touch with Kubai Kevin</p></div>
        <article class="page-content">
            <div class="contact-method"><h3>Email</h3>
                <p><a href="mailto:aiblogauto@gmail.com" class="contact-email">aiblogauto@gmail.com</a></p>
                <p>I typically respond within 3–5 working days (EAT, UTC+3).</p></div>
            <div class="contact-method"><h3>What to reach out about</h3>
                <ul>
                    <li>Factual errors or corrections in articles — I take these seriously and update posts when errors are confirmed</li>
                    <li>Topic suggestions or questions about content covered on the site</li>
                    <li>Collaboration or guest post proposals</li>
                    <li>General questions about the site or the author</li>
                </ul></div>
            <div class="contact-method"><h3>What not to email about</h3>
                <ul>
                    <li>Link exchange or SEO partnership requests — these go unanswered</li>
                    <li>Sponsored post inquiries without prior discussion</li>
                </ul></div>
            <div class="contact-footer">
                <p>Prefer email over social media for anything requiring a substantive reply.
                For quick questions, Twitter DMs (<a href="https://twitter.com/KubaiKevin" style="color:#fff;text-decoration:underline;" target="_blank" rel="noopener">@KubaiKevin</a>) also work.</p>
            </div>
        </article>
    </main>
    <footer><div class="container"><p>&copy; {{ current_year }} {{ site_name }} · Written by Kubai Kevin</p></div></footer>
    <script src="../static/navigation.js"></script>
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
    }
