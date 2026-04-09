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
        self._generate_ads_txt()
        print(f"Site generated successfully with {len(posts)} posts!")

    def _generate_ads_txt(self):
        config = self.blog_system.config
        adsense_id = config.get('google_adsense_id', '')
        if adsense_id:
            pub_id = adsense_id.replace('ca-pub-', '')
            ads_txt_content = f"google.com, pub-{pub_id}, DIRECT, f08c47fec0942fa0\n"
            with open("./docs/ads.txt", 'w', encoding='utf-8') as f:
                f.write(ads_txt_content)
            print("Generated ads.txt file")

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

    def _generate_homepage(self, posts: List[BlogPost]):
        config = self.blog_system.config
        posts_data = []
        for p in posts:
            post_dict = p.to_dict()
            post_dict['display_date'] = self._format_display_date(p.created_at)
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
            return dt.strftime('%-d %b %Y')
        except:
            try:
                dt = datetime.fromisoformat(iso_date.replace('Z', '+00:00'))
                return dt.strftime('%d %b %Y').lstrip('0')
            except:
                return iso_date.split('T')[0]

    def _generate_post_pages(self, posts: List[BlogPost]):
        config = self.blog_system.config
        for post in posts:
            post_dir = Path("./docs") / post.slug
            post_dir.mkdir(exist_ok=True)
            markdown_converter = md.Markdown(
                extensions=['extra', 'codehilite', 'toc'])
            content_html = markdown_converter.convert(post.content)
            post_dict = post.to_dict()
            post_dict['content_html'] = content_html
            post_dict['display_date'] = self._format_display_date(
                post.created_at)
            ad_slots = post.monetization_data if post.monetization_data else {}
            context = {
                'site_name': config.get('site_name', 'AI Blog'),
                'base_path': config.get('base_path', ''),
                'base_url': config.get('base_url', ''),
                'post': post_dict,
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
                'site_name': config.get('site_name', 'AI Blog'),
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
            item = f"""    <item>
      <title>{self._escape_xml(post.title)}</title>
      <link>{base_url}/{post.slug}/</link>
      <description>{self._escape_xml(post.meta_description)}</description>
      <pubDate>{self._format_rss_date(post.created_at)}</pubDate>
      <guid>{base_url}/{post.slug}/</guid>
    </item>"""
            rss_items.append(item)
        rss_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>{config.get('site_name', 'AI Blog')}</title>
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
        urls = [
            f'<url><loc>{base_url}/</loc><priority>1.0</priority></url>',
            f'<url><loc>{base_url}/about/</loc><priority>0.8</priority></url>',
            f'<url><loc>{base_url}/contact/</loc><priority>0.8</priority></url>',
            f'<url><loc>{base_url}/privacy-policy/</loc><priority>0.5</priority></url>',
            f'<url><loc>{base_url}/terms-of-service/</loc><priority>0.5</priority></url>'
        ]
        for post in posts:
            urls.append(
                f'<url><loc>{base_url}/{post.slug}/</loc><priority>0.9</priority></url>')
        sitemap = f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  {chr(10).join(urls)}
</urlset>"""
        with open("./docs/sitemap.xml", 'w', encoding='utf-8') as f:
            f.write(sitemap)
        print("Generated sitemap")

    def _generate_posts_json(self, posts: List[BlogPost]):
        posts_data = [p.to_dict() for p in posts]
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
        template_strings = {
            "post": """<!DOCTYPE html>
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
        <article class="blog-post">
            <header class="post-header">
                <h1>{{ post.title }}</h1>
                {% if post.tags %}
                <div class="tags">
                    {% for tag in post.tags[:6] %}
                    <span class="tag">{{ tag }}</span>
                    {% endfor %}
                </div>
                {% endif %}
                <div class="post-meta">
                    <time datetime="{{ post.created_at }}">{{ post.display_date }}</time>
                </div>
            </header>
            <div class="post-content">
                {{ post.content_html | safe }}
                {{ middle_ad | safe }}
            </div>
            {% if post.affiliate_links %}
            <div class="affiliate-disclaimer">
                <p><em>This post contains affiliate links. We may earn a commission if you make a purchase through these links, at no additional cost to you.</em></p>
            </div>
            {% endif %}
        </article>
    </main>
    {{ footer_ad | safe }}
    <footer>
        <div class="container">
            <p>&copy; {{ current_year }} {{ site_name }}.</p>
        </div>
    </footer>
    <script src="{{ base_path }}/static/navigation.js"></script>
</body>
</html>""",

            "index": """<!DOCTYPE html>
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
        .search-container { margin: 0; max-width: 420px; }
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
        .search-input::-webkit-search-cancel-button,
        .search-input::-webkit-credentials-auto-fill-button,
        .search-input::-ms-clear { display: none !important; width: 0 !important; }
        .clear-search {
            background: none; border: none; cursor: pointer;
            color: #9ca3af; padding: 2px; flex-shrink: 0; display: flex; align-items: center;
        }
        .clear-search:hover { color: #333; }
        .search-results-count { margin-top: 0.4rem; color: #666; font-size: 0.85rem; min-height: 1.2em; }
        .search-highlight { background: #fef08a; border-radius: 2px; padding: 0 1px; }
        .no-results-message { text-align: center; padding: 3rem 1rem; color: #666; }
        .no-results-message svg { width: 48px; height: 48px; margin-bottom: 1rem; color: #9ca3af; display: block; margin: 0 auto 1rem; }
        .no-results-message h3 { margin: 0 0 0.5rem; color: #333; }
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

        <!-- Search Bar -->
        <div class="search-container">
            <div class="search-wrapper">
                <svg class="search-icon" width="18" height="18" viewBox="0 0 20 20" fill="none">
                    <path d="M9 17A8 8 0 1 0 9 1a8 8 0 0 0 0 16zM19 19l-4.35-4.35"
                          stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                <input type="text" id="search-input" class="search-input"
                    placeholder="Search by title, description, or tags....."
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
                    <article class="post-card"> 
                        <h3><a href="{{ base_path }}/{{ post.slug }}/">{{ post.title }}</a></h3> 
                        <p class="post-excerpt">{{ post.meta_description }}</p> 
                        {% if post.tags %}
                        <div class="tags"> 
                            {% for tag in post.tags[:3] %}
                            <span class="tag">{{ tag }}</span> 
                            {% endfor %} 
                        </div> 
                        {% endif %}
                    </article>
                {% endfor %}
            </div>

            <div id="loading-spinner" class="loading-spinner" style="display:none;">
                <div class="spinner"></div>
                <p>Loading more posts...</p>
            </div>

            {% if posts|length > posts_per_page %}
            <div id="load-more-container" class="load-more-container">
                <button id="load-more" type="button" class="load-more-button">
                    <span class="button-text">Load More Posts</span>
                    <span class="button-icon">↓</span>
                </button>
            </div>
            {% endif %}

            <div class="scroll-options">
                <label class="toggle-switch" hidden>
                    <input type="checkbox" id="infinite-scroll-toggle">
                    <span class="slider"></span>
                    Enable Infinite Scroll
                </label>
            </div>
            {% else %}
            <p>No posts yet. Check back soon!</p>
            {% endif %}
        </section>
    </main>

    <button id="back-to-top" class="back-to-top" style="display:none;"><span>↑</span></button>

    <footer>
        <div class="container">
            <p>&copy; {{ current_year }} {{ site_name }}.</p>
            <div class="social-links">
                {% for platform, url in social_links.items() %}
                <a href="{{ url }}" target="_blank" rel="noopener">{{ platform|title }}</a>
                {% endfor %}
            </div>
        </div>
    </footer>

    <script src="{{ base_path }}/static/navigation.js"></script>
    <script>
    document.addEventListener('DOMContentLoaded', function () {

        // ── ALL SHARED STATE UP FRONT ─────────────────────────────────────────
        var searchInput          = document.getElementById('search-input');
        var clearSearchBtn       = document.getElementById('clear-search');
        var resultsCount         = document.getElementById('search-results-count');
        var postsContainer       = document.getElementById('posts-container');
        var loadMoreButton       = document.getElementById('load-more');
        var loadingSpinner       = document.getElementById('loading-spinner');
        var infiniteScrollToggle = document.getElementById('infinite-scroll-toggle');
        var backToTopButton      = document.getElementById('back-to-top');

        var postsPerPage          = {{ posts_per_page }};
        var allPosts              = [];
        var currentPage           = 1;
        var isLoading             = false;
        var searchActive          = false;
        var infiniteScrollEnabled = localStorage.getItem('infiniteScroll') === 'true';

        if (infiniteScrollToggle) infiniteScrollToggle.checked = infiniteScrollEnabled;

        // ── SEED FROM DOM ─────────────────────────────────────────────────────
        // FIX: Reads the server-rendered post cards to populate allPosts as a
        // reliable fallback when posts.json cannot be fetched (e.g. base_path
        // mismatch on GitHub Pages project sites). This ensures search and
        // load-more always have data to work with even on fetch failure.
        function seedFromDOM() {
            if (!postsContainer) return;
            allPosts = Array.from(postsContainer.querySelectorAll('article.post-card')).map(function(el) {
                var anchor = el.querySelector('h3 a');
                var href   = anchor ? (anchor.getAttribute('href') || '') : '';
                var slug   = href.replace(/\\/+$/, '').split('/').filter(Boolean).pop() || '';
                return {
                    slug:             slug,
                    title:            anchor ? anchor.textContent.trim() : '',
                    meta_description: el.querySelector('.post-excerpt') ? el.querySelector('.post-excerpt').textContent.trim() : '',
                    tags:             Array.from(el.querySelectorAll('.tag')).map(function(t) { return t.textContent.trim(); }),
                    created_at:       el.querySelector('time') ? (el.querySelector('time').getAttribute('datetime') || '') : ''
                };
            });
        }

        // ── FETCH ALL POSTS JSON ──────────────────────────────────────────────
        // FIX: Changed from '{{ base_path }}/posts.json' to 'posts.json'.
        // posts.json is always a sibling of index.html in docs/, so a bare
        // relative URL resolves correctly regardless of base_path or the
        // GitHub Pages hosting prefix (user page vs project page).
        // Previously '{{ base_path }}/posts.json' would render as e.g.
        // '/my-repo/posts.json' which is a 404 when docs/ is the site root.
        fetch('posts.json')
            .then(function(r) {
                if (!r.ok) throw new Error('HTTP ' + r.status);
                return r.json();
            })
            .then(function(posts) {
                allPosts = posts;
                updateLoadMoreButton();
                if (searchInput && searchInput.value.trim()) {
                    performSearch(searchInput.value.trim().toLowerCase());
                }
            })
            .catch(function(err) {
                console.warn('posts.json fetch failed, falling back to DOM seed:', err);
                seedFromDOM();
                updateLoadMoreButton();
            });

        // ── HELPERS ──────────────────────────────────────────────────────────
        function escapeRe(s) {
            return s.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\\\$&');
        }

        function highlight(text, query) {
            return text.replace(
                new RegExp('(' + escapeRe(query) + ')', 'gi'),
                '<span class="search-highlight">$1</span>'
            );
        }

        function formatDisplayDate(iso) {
            try {
                var d = new Date(iso);
                return d.getDate() + ' ' +
                    ['January','February','March','April','May','June',
                     'July','August','September','October','November','December'][d.getMonth()] +
                    ' ' + d.getFullYear();
            } catch(e) { return iso.split('T')[0]; }
        }

        function createPostElement(post) {
            var article = document.createElement('article');
            article.className = 'post-card';

            var h3   = document.createElement('h3');
            var link = document.createElement('a');
            link.href        = '{{ base_path }}/' + post.slug + '/';
            link.textContent = post.title;
            h3.appendChild(link);

            var excerpt = document.createElement('p');
            excerpt.className   = 'post-excerpt';
            excerpt.textContent = post.meta_description;

            article.appendChild(h3);
            article.appendChild(excerpt);

            if (post.tags && post.tags.length > 0) {
                var tagsDiv = document.createElement('div');
                tagsDiv.className = 'tags';
                post.tags.slice(0, 3).forEach(function(t) {
                    var span = document.createElement('span');
                    span.className   = 'tag';
                    span.textContent = t;
                    tagsDiv.appendChild(span);
                });
                article.appendChild(tagsDiv);
            }

            var meta = document.createElement('div');
            meta.className = 'post-meta';
            var time = document.createElement('time');
            time.dateTime    = post.created_at;
            time.textContent = formatDisplayDate(post.created_at);
            article.appendChild(meta);

            return article;
        }

        // ── SEARCH ───────────────────────────────────────────────────────────
        function performSearch(query) {
            if (!allPosts.length) return;
            searchActive = true;

            var matched = allPosts.filter(function(post) {
                var t = (post.title            || '').toLowerCase();
                var d = (post.meta_description || '').toLowerCase();
                var g = (post.tags             || []).map(function(x) { return x.toLowerCase(); });
                return t.indexOf(query) !== -1 ||
                       d.indexOf(query) !== -1 ||
                       g.some(function(x) { return x.indexOf(query) !== -1; });
            });

            postsContainer.innerHTML = '';
            matched.forEach(function(post) {
                var el      = createPostElement(post);
                var link    = el.querySelector('h3 a');
                var excerpt = el.querySelector('.post-excerpt');
                if (link)    link.innerHTML    = highlight(link.textContent,    query);
                if (excerpt) excerpt.innerHTML = highlight(excerpt.textContent, query);
                postsContainer.appendChild(el);
            });

            var lmc = document.getElementById('load-more-container');
            if (lmc) lmc.style.display = 'none';

            var n = matched.length;
            resultsCount.textContent = n === 0 ? 'No results for "' + query + '"'
                                     : n === 1 ? '1 post found'
                                     : n + ' posts found';

            var noRes = document.getElementById('no-results');
            if (n === 0) {
                if (!noRes) {
                    noRes = document.createElement('div');
                    noRes.id        = 'no-results';
                    noRes.className = 'no-results-message';
                    noRes.innerHTML =
                        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor">' +
                        '<circle cx="11" cy="11" r="8"/>' +
                        '<path d="m21 21-4.35-4.35"/></svg>' +
                        '<h3>No posts found</h3>' +
                        '<p>Try different keywords or clear the search.</p>';
                    postsContainer.after(noRes);
                }
            } else if (noRes) {
                noRes.remove();
            }
        }

        function clearSearch() {
            searchActive = false;
            postsContainer.innerHTML = '';
            allPosts.slice(0, postsPerPage).forEach(function(post) {
                postsContainer.appendChild(createPostElement(post));
            });
            currentPage = 1;
            resultsCount.textContent = '';
            var noRes = document.getElementById('no-results');
            if (noRes) noRes.remove();
            updateLoadMoreButton();
        }

        if (searchInput) {
            searchInput.addEventListener('input', function () {
                var q = this.value.trim().toLowerCase();
                clearSearchBtn.style.display = q ? 'flex' : 'none';
                if (q) { performSearch(q); } else { clearSearch(); }
            });

            clearSearchBtn.addEventListener('click', function () {
                searchInput.value = '';
                this.style.display = 'none';
                clearSearch();
                searchInput.focus();
            });

            searchInput.addEventListener('keydown', function (e) {
                if (e.key === 'Escape') {
                    this.value = '';
                    clearSearchBtn.style.display = 'none';
                    clearSearch();
                }
            });
        }

        // ── LOAD MORE / INFINITE SCROLL ───────────────────────────────────────
        if (loadMoreButton) {
            loadMoreButton.addEventListener('click', function (e) {
                e.preventDefault();
                loadMorePosts();
            });
        }

        if (infiniteScrollToggle) {
            infiniteScrollToggle.addEventListener('change', function () {
                infiniteScrollEnabled = this.checked;
                localStorage.setItem('infiniteScroll', infiniteScrollEnabled);
                if (infiniteScrollEnabled) {
                    if (loadMoreButton) loadMoreButton.style.display = 'none';
                    enableInfiniteScroll();
                } else {
                    disableInfiniteScroll();
                    updateLoadMoreButton();
                }
            });
        }

        if (infiniteScrollEnabled) {
            enableInfiniteScroll();
            if (loadMoreButton) loadMoreButton.style.display = 'none';
        }

        window.addEventListener('scroll', function () {
            if (backToTopButton)
                backToTopButton.style.display = window.pageYOffset > 300 ? 'flex' : 'none';
        });

        if (backToTopButton) {
            backToTopButton.addEventListener('click', function() {
                window.scrollTo({ top: 0, behavior: 'smooth' });
            });
        }

        function loadMorePosts() {
            if (isLoading || searchActive || !hasMorePosts()) return;
            isLoading = true;
            if (loadingSpinner) loadingSpinner.style.display = 'block';
            if (loadMoreButton) { loadMoreButton.disabled = true; loadMoreButton.style.opacity = '0.6'; }

            setTimeout(function() {
                currentPage++;
                var start = (currentPage - 1) * postsPerPage;
                allPosts.slice(start, start + postsPerPage).forEach(function(post, i) {
                    var el = createPostElement(post);
                    el.style.opacity   = '0';
                    el.style.transform = 'translateY(20px)';
                    postsContainer.appendChild(el);
                    setTimeout(function() {
                        el.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
                        el.style.opacity    = '1';
                        el.style.transform  = 'translateY(0)';
                    }, i * 100);
                });
                if (loadingSpinner) loadingSpinner.style.display = 'none';
                if (loadMoreButton) { loadMoreButton.disabled = false; loadMoreButton.style.opacity = '1'; }
                updateLoadMoreButton();
                isLoading = false;
            }, 500);
        }

        function hasMorePosts() { return currentPage * postsPerPage < allPosts.length; }

        function updateLoadMoreButton() {
            if (searchActive) return;
            var show      = hasMorePosts() && !infiniteScrollEnabled;
            var container = document.getElementById('load-more-container');
            if (loadMoreButton) loadMoreButton.style.display = show ? 'block' : 'none';
            if (container)      container.style.display      = show ? 'block' : 'none';
        }

        function enableInfiniteScroll()  { window.addEventListener('scroll',    infiniteScrollHandler); }
        function disableInfiniteScroll() { window.removeEventListener('scroll', infiniteScrollHandler); }

        function infiniteScrollHandler() {
            if (isLoading || searchActive || !hasMorePosts()) return;
            if (window.pageYOffset + window.innerHeight >= document.documentElement.scrollHeight - 1000)
                loadMorePosts();
        }
    });
    </script>
</body>
</html>""",

            "about": """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>About - {{ site_name }}</title>
    <meta name="description" content="About {{ site_name }}">
    {{ global_meta_tags | safe }}
    <link rel="stylesheet" href="../static/style.css">
    <link rel="canonical" href="{{ base_url }}/about/">
    <style>
        .about-section { background:#f8f9fa; padding:2rem; margin-bottom:2rem; border-radius:8px; border-left:4px solid #6366f1; }
        .about-section h2 { color:#333; margin-top:0; margin-bottom:1rem; font-size:1.5rem; }
        .about-section h3 { color:#555; margin-top:1.5rem; margin-bottom:1rem; font-size:1.2rem; }
        .about-section p { line-height:1.8; margin-bottom:1rem; }
        .about-section ul { margin-left:1.5rem; line-height:1.8; }
        .about-section ul li { margin-bottom:0.5rem; }
        .mission-box { background:linear-gradient(135deg,#667eea 0%,#764ba2 100%); color:white; padding:2rem; border-radius:8px; margin:2rem 0; text-align:center; }
        .mission-box h2 { margin-top:0; color:white; font-size:2rem; }
        .mission-box p { font-size:1.1rem; line-height:1.8; margin-bottom:0; }
        .feature-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(250px,1fr)); gap:1.5rem; margin:2rem 0; }
        .feature-card { background:white; padding:1.5rem; border-radius:8px; border:2px solid #e0e0e0; transition:transform 0.3s ease,border-color 0.3s ease; }
        .feature-card:hover { transform:translateY(-5px); border-color:#6366f1; }
        .feature-card h3 { color:#6366f1; margin-top:0; margin-bottom:1rem; font-size:1.3rem; }
        .feature-icon { font-size:2.5rem; margin-bottom:1rem; display:block; }
        .stats-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:1.5rem; margin:2rem 0; }
        .stat-card { background:#f0f4ff; padding:1.5rem; border-radius:8px; text-align:center; border:2px solid #6366f1; }
        .stat-number { font-size:2.5rem; font-weight:bold; color:#6366f1; display:block; margin-bottom:0.5rem; }
        .stat-label { color:#555; font-size:1rem; }
        .topics-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:1rem; margin:1.5rem 0; }
        .topic-tag { background:white; padding:1rem; border-radius:8px; border-left:4px solid #8b5cf6; font-weight:500; color:#333; }
        .cta-box { background:#fff3cd; border-left:4px solid #ffc107; padding:2rem; border-radius:8px; margin:2rem 0; text-align:center; }
        .cta-box h3 { margin-top:0; color:#333; font-size:1.5rem; }
        .cta-button { display:inline-block; background:linear-gradient(135deg,#667eea 0%,#764ba2 100%); color:white; padding:1rem 2rem; border-radius:8px; text-decoration:none; font-weight:600; margin-top:1rem; transition:transform 0.3s ease; }
        .cta-button:hover { transform:scale(1.05); }
        .team-section { background:#f0f4ff; padding:2rem; border-radius:8px; border-left:4px solid #8b5cf6; }
    </style>
</head>
<body>
    <header>
        <div class="container">
            <h1><a href="../">{{ site_name }}</a></h1>
            <nav>
                <a href="../">Home</a><a href="../about/">About</a><a href="../contact/">Contact</a>
                <a href="../privacy-policy/">Privacy Policy</a><a href="../terms-of-service/">Terms of Service</a>
            </nav>
        </div>
    </header>
    <main class="container">
        <div class="hero">
            <h2>About {{ site_name }}</h2>
            <p>Your trusted source for AI and technology insights</p>
        </div>
        <article class="page-content">
            <div class="about-section">
                <h2>Welcome to {{ site_name }}</h2>
                <p>{{ site_name }} is an innovative AI-powered technology blog dedicated to delivering high-quality, informative content on the latest developments in artificial intelligence, machine learning, and emerging technologies.</p>
                  <h2>🎯 Our Mission</h2>
                <p>To empower readers with knowledge about artificial intelligence and technology through accessible, accurate, and engaging content.</p>
            </div>
            <div class="about-section">
                <h2>📊 What We Do</h2>
                <div class="feature-grid">
                    <div class="feature-card"><span class="feature-icon">📰</span><h3>Breaking News</h3><p>Latest AI and tech announcements and industry developments.</p></div>
                    <div class="feature-card"><span class="feature-icon">🔬</span><h3>In-Depth Analysis</h3><p>Comprehensive breakdowns of complex topics and emerging technologies.</p></div>
                    <div class="feature-card"><span class="feature-icon">💡</span><h3>Practical Guides</h3><p>Step-by-step tutorials for implementing AI solutions and tools.</p></div>
                    <div class="feature-card"><span class="feature-icon">🚀</span><h3>Future Trends</h3><p>Forward-looking analysis of where AI and technology are headed.</p></div>
                    <div class="feature-card"><span class="feature-icon">⚖️</span><h3>Ethics & Policy</h3><p>Discussions on AI ethics, regulations, and responsible development.</p></div>
                    <div class="feature-card"><span class="feature-icon">🎓</span><h3>Educational Content</h3><p>Learning resources for beginners to advanced practitioners.</p></div>
                </div>
            </div>
            <div class="about-section">
                <h2>💻 Topics We Cover</h2>
                <div class="topics-grid">{% for topic in topics %}<div class="topic-tag">{{ topic }}</div>{% endfor %}</div>
            </div>
            <div class="stats-grid">
                <div class="stat-card"><span class="stat-number">24/7</span><span class="stat-label">Content Publishing</span></div>
                <div class="stat-card"><span class="stat-number">100%</span><span class="stat-label">AI-Powered</span></div>
                <div class="stat-card"><span class="stat-number">∞</span><span class="stat-label">Learning & Improving</span></div>
            </div>
            <div class="cta-box">
                <h3>📬 Stay Connected</h3>
                <p>Want to stay updated with the latest AI and technology insights?</p>
                <a href="../contact/" class="cta-button">Contact Us</a>
            </div>
        </article>
    </main>
    <footer><div class="container"><p>&copy; {{ current_year }} {{ site_name }}.</p></div></footer>
    <script src="../static/navigation.js"></script>
</body>
</html>""",

            "privacy_policy": """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Privacy Policy - {{ site_name }}</title>
    <meta name="description" content="Privacy Policy for {{ site_name }}">
    {{ global_meta_tags | safe }}
    <link rel="stylesheet" href="../static/style.css">
    <link rel="canonical" href="{{ base_url }}/privacy-policy/">
    <style>
        .privacy-section { background:#f8f9fa; padding:1.5rem; margin-bottom:1.5rem; border-radius:8px; border-left:4px solid #6366f1; }
        .privacy-section h3 { color:#333; margin-top:0; margin-bottom:1rem; font-size:1.3rem; }
        .privacy-section h4 { color:#555; margin-top:1rem; margin-bottom:0.5rem; font-size:1.05rem; }
        .privacy-section ul { margin-left:1.5rem; line-height:1.8; }
        .privacy-section ul li { margin-bottom:0.5rem; }
        .highlight-box { background:linear-gradient(135deg,#667eea 0%,#764ba2 100%); color:white; padding:1.5rem; border-radius:8px; margin:2rem 0; }
        .highlight-box h3 { margin-top:0; color:white; }
        .important-notice { background:#fff3cd; border-left:4px solid #ffc107; padding:1rem 1.5rem; margin:1.5rem 0; border-radius:4px; }
        .table-container { overflow-x:auto; margin:1.5rem 0; }
        table { width:100%; border-collapse:collapse; background:white; }
        th,td { padding:1rem; text-align:left; border-bottom:1px solid #dee2e6; }
        th { background:#f8f9fa; font-weight:600; color:#333; }
        .toc { background:#f0f4ff; padding:1.5rem; border-radius:8px; margin-bottom:2rem; }
        .toc h3 { margin-top:0; color:#333; }
        .toc ul { list-style:none; padding-left:0; }
        .toc li { margin-bottom:0.5rem; }
        .toc a { color:#6366f1; text-decoration:none; }
        .toc a:hover { text-decoration:underline; }
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
            <div class="toc"><h3>Table of Contents</h3><ul>
                <li><a href="#introduction">1. Introduction</a></li>
                <li><a href="#information-collection">2. Information We Collect</a></li>
                <li><a href="#how-we-use">3. How We Use Your Information</a></li>
                <li><a href="#cookies">4. Cookies and Tracking</a></li>
                <li><a href="#third-party">5. Third-Party Services</a></li>
                <li><a href="#your-rights">6. Your Privacy Rights</a></li>
                <li><a href="#contact">7. Contact Information</a></li>
            </ul></div>
            <div id="introduction" class="privacy-section">
                <h3>1. Introduction</h3>
                <p><strong>{{ site_name }}</strong> is committed to protecting your privacy. By accessing our Site, you agree to this Privacy Policy.</p>
            </div>
            <div id="information-collection" class="privacy-section">
                <h3>2. Information We Collect</h3>
                <ul>
                    <li><strong>Contact Information:</strong> Name, email, message content when you contact us.</li>
                    <li><strong>Device Information:</strong> Device type, OS, browser type and version.</li>
                    <li><strong>Usage Data:</strong> Pages visited, time spent, links clicked.</li>
                    <li><strong>Location Data:</strong> General geographic location based on IP address.</li>
                </ul>
            </div>
            <div id="how-we-use" class="privacy-section">
                <h3>3. How We Use Your Information</h3>
                <ul>
                    <li>Operate, maintain, and improve the Site</li>
                    <li>Respond to your inquiries</li>
                    <li>Monitor and analyze usage trends</li>
                    <li>Comply with legal obligations</li>
                </ul>
            </div>
            <div id="cookies" class="privacy-section">
                <h3>4. Cookies and Tracking</h3>
                <p>We use cookies to improve your experience. You can control cookies through your browser settings.</p>
                <div class="table-container"><table>
                    <thead><tr><th>Cookie Type</th><th>Purpose</th><th>Duration</th></tr></thead>
                    <tbody>
                        <tr><td><strong>Essential</strong></td><td>Basic site functionality</td><td>Session</td></tr>
                        <tr><td><strong>Analytics</strong></td><td>Track site usage</td><td>Up to 2 years</td></tr>
                        <tr><td><strong>Advertising</strong></td><td>Relevant advertisements</td><td>Up to 1 year</td></tr>
                    </tbody>
                </table></div>
            </div>
            <div id="third-party" class="privacy-section">
                <h3>5. Third-Party Services</h3>
                <ul>
                    <li><strong>Google Analytics:</strong> Site usage analysis. <a href="https://policies.google.com/privacy" target="_blank" rel="noopener">Google Privacy Policy</a></li>
                    <li><strong>Google AdSense:</strong> Display advertising. <a href="https://policies.google.com/technologies/ads" target="_blank" rel="noopener">Google Advertising Policy</a></li>
                </ul>
                <div class="important-notice"><p><strong>Important:</strong> We do not sell, rent, or trade your personal information to third parties.</p></div>
            </div>
            <div id="your-rights" class="privacy-section">
                <h3>6. Your Privacy Rights</h3>
                <ul>
                    <li><strong>Correction:</strong> Request correction of inaccurate information</li>
                    <li><strong>Deletion:</strong> Request deletion of your personal information</li>
                </ul>
                <p>Contact us at <a href="mailto:aiblogauto@gmail.com">aiblogauto@gmail.com</a> to exercise these rights.</p>
            </div>
            <div id="contact" class="highlight-box">
                <h3>7. Contact Information</h3>
                <p><strong>Email:</strong> <a href="mailto:aiblogauto@gmail.com" style="color:white;text-decoration:underline;">aiblogauto@gmail.com</a></p>
                <p>We aim to respond within 48 hours on business days.</p>
            </div>
            <div style="margin-top:2rem;padding:1rem;background:#f8f9fa;border-radius:8px;">
                <p><strong>Last Updated:</strong> {{ current_date }}</p>
            </div>
        </article>
    </main>
    <footer><div class="container"><p>&copy; {{ current_year }} {{ site_name }}.</p></div></footer>
    <script src="../static/navigation.js"></script>
</body>
</html>""",

            "terms_of_service": """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Terms of Service - {{ site_name }}</title>
    <meta name="description" content="Terms of Service for {{ site_name }}">
    {{ global_meta_tags | safe }}
    <link rel="stylesheet" href="../static/style.css">
    <link rel="canonical" href="{{ base_url }}/terms-of-service/">
    <style>
        .terms-section { background:#f8f9fa; padding:1.5rem; margin-bottom:1.5rem; border-radius:8px; border-left:4px solid #6366f1; }
        .terms-section h3 { color:#333; margin-top:0; margin-bottom:1rem; font-size:1.3rem; }
        .terms-section h4 { color:#555; margin-top:1rem; margin-bottom:0.5rem; }
        .terms-section ul,.terms-section ol { margin-left:1.5rem; line-height:1.8; }
        .terms-section li { margin-bottom:0.5rem; }
        .highlight-box { background:linear-gradient(135deg,#667eea 0%,#764ba2 100%); color:white; padding:1.5rem; border-radius:8px; margin:2rem 0; }
        .highlight-box h3 { margin-top:0; color:white; }
        .important-notice { background:#fff3cd; border-left:4px solid #ffc107; padding:1rem 1.5rem; margin:1.5rem 0; border-radius:4px; }
        .warning-box { background:#f8d7da; border-left:4px solid #dc3545; padding:1rem 1.5rem; margin:1.5rem 0; border-radius:4px; color:#721c24; }
        .toc { background:#f0f4ff; padding:1.5rem; border-radius:8px; margin-bottom:2rem; }
        .toc h3 { margin-top:0; color:#333; }
        .toc ul { list-style:none; padding-left:0; }
        .toc li { margin-bottom:0.5rem; }
        .toc a { color:#6366f1; text-decoration:none; }
        .toc a:hover { text-decoration:underline; }
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
            <div class="important-notice"><p><strong>Important:</strong> By accessing our Site, you agree to these Terms. If you do not agree, you must not use our Site.</p></div>
            <div class="toc"><h3>Table of Contents</h3><ul>
                <li><a href="#acceptance">1. Acceptance of Terms</a></li>
                <li><a href="#ai-content">2. AI-Generated Content Disclaimer</a></li>
                <li><a href="#use-license">3. License to Use Site</a></li>
                <li><a href="#third-party">4. Third-Party Links & Affiliate Disclosure</a></li>
                <li><a href="#disclaimer">5. Disclaimers & Limitation of Liability</a></li>
                <li><a href="#governing-law">6. Governing Law</a></li>
                <li><a href="#contact">7. Contact Information</a></li>
            </ul></div>
            <div id="acceptance" class="terms-section">
                <h3>1. Acceptance of Terms</h3>
                <p>By accessing {{ site_name }}, you agree to these Terms of Service and our Privacy Policy. We reserve the right to modify these Terms at any time; continued use constitutes acceptance.</p>
            </div>
            <div id="ai-content" class="terms-section">
                <h3>2. AI-Generated Content Disclaimer</h3>
                <div class="warning-box"><p><strong>Notice:</strong> The majority of content on {{ site_name }} is generated using artificial intelligence technology.</p></div>
                <p>AI-generated content may contain inaccuracies. Content on this Site does not constitute legal, financial, or medical advice. Always verify important information independently and consult qualified professionals.</p>
            </div>
            <div id="use-license" class="terms-section">
                <h3>3. License to Use Site</h3>
                <p>We grant you a limited, non-exclusive, non-transferable license to access and view the Site for personal, non-commercial use. You may not reproduce, distribute, or create derivative works without our written consent.</p>
            </div>
            <div id="third-party" class="terms-section">
                <h3>4. Third-Party Links & Affiliate Disclosure</h3>
                <div class="important-notice"><p>We participate in affiliate marketing programs and may earn commissions from purchases made through links on our Site. This does not affect the price you pay.</p></div>
                <p>We are not responsible for the content or practices of third-party websites linked from our Site.</p>
            </div>
            <div id="disclaimer" class="terms-section">
                <h3>5. Disclaimers & Limitation of Liability</h3>
                <div class="warning-box"><p><strong>THE SITE AND ALL CONTENT ARE PROVIDED "AS IS" WITHOUT WARRANTIES OF ANY KIND.</strong></p></div>
                <p>TO THE MAXIMUM EXTENT PERMITTED BY LAW, {{ site_name }} SHALL NOT BE LIABLE FOR ANY INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING FROM YOUR USE OF THE SITE.</p>
            </div>
            <div id="governing-law" class="terms-section">
                <h3>6. Governing Law</h3>
                <p>These Terms are governed by the laws of Kenya. Any disputes shall be resolved in the courts of Kenya.</p>
            </div>
            <div id="contact" class="highlight-box">
                <h3>7. Contact Information</h3>
                <p><strong>Email:</strong> <a href="mailto:aiblogauto@gmail.com" style="color:white;text-decoration:underline;">aiblogauto@gmail.com</a></p>
                <p>Include "Legal Notice - Terms of Service" in your subject line.</p>
            </div>
            <div style="margin-top:2rem;padding:1rem;background:#f8f9fa;border-radius:8px;">
                <p><strong>Last Updated:</strong> {{ current_date }}</p>
            </div>
        </article>
    </main>
    <footer><div class="container"><p>&copy; {{ current_year }} {{ site_name }}.</p></div></footer>
    <script src="../static/navigation.js"></script>
</body>
</html>""",

            "contact": """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Contact Us - {{ site_name }}</title>
    <meta name="description" content="Contact {{ site_name }}">
    {{ global_meta_tags | safe }}
    <link rel="stylesheet" href="../static/style.css">
    <link rel="canonical" href="{{ base_url }}/contact/">
    <style>
        .contact-method { background:#f8f9fa; padding:1.5rem; margin-bottom:1.5rem; border-radius:8px; border-left:4px solid #6366f1; }
        .contact-method h3 { color:#333; margin-bottom:1rem; font-size:1.3rem; }
        .contact-method ul { margin-left:1.5rem; line-height:1.8; }
        .contact-method ul li { margin-bottom:0.5rem; }
        .contact-email { color:#6366f1; text-decoration:none; font-weight:600; font-size:1.1rem; }
        .contact-email:hover { text-decoration:underline; }
        .response-time { color:#666; font-style:italic; margin-top:0.5rem; font-size:0.9rem; }
        .faq-section { background:#f0f4ff; border-left-color:#8b5cf6; }
        .faq-item { margin-bottom:1.5rem; }
        .faq-item h4 { color:#555; margin-bottom:0.5rem; }
        .contact-footer { background:linear-gradient(135deg,#667eea 0%,#764ba2 100%); color:white; padding:1.5rem; border-radius:8px; margin-top:2rem; }
    </style>
</head>
<body>
    <header><div class="container">
        <h1><a href="../">{{ site_name }}</a></h1>
        <nav><a href="../">Home</a><a href="../about/">About</a><a href="../contact/">Contact</a>
        <a href="../privacy-policy/">Privacy Policy</a><a href="../terms-of-service/">Terms of Service</a></nav>
    </div></header>
    <main class="container">
        <div class="hero"><h2>Contact Us</h2><p>Get in touch with the {{ site_name }} team</p></div>
        <article class="page-content">
            <p>We'd love to hear from you! Whether you have questions, feedback, or collaboration opportunities, feel free to reach out.</p>
            <div class="contact-method">
                <h3>📧 Email</h3>
                <p><a href="mailto:aiblogauto@gmail.com" class="contact-email">aiblogauto@gmail.com</a></p>
                <p class="response-time">We typically respond within 5 working days</p>
            </div>
            <div class="contact-method">
                <h3>💬 What We Cover</h3>
                <ul>
                    <li><strong>General Inquiries:</strong> Questions about our content or AI technology</li>
                    <li><strong>Collaboration:</strong> Partnership opportunities and guest posting</li>
                    <li><strong>Technical Support:</strong> Issues accessing content</li>
                    <li><strong>Feedback:</strong> Suggestions for improvement or topic requests</li>
                </ul>
            </div>
            <div class="contact-method">
                <h3>⏰ Office Hours</h3>
                <p><strong>Monday – Friday:</strong> 10:00 AM – 4:00 PM (EAT)</p>
                <p>Weekend messages will be reviewed on the next business day.</p>
            </div>
            <div class="contact-method faq-section">
                <h3>❓ Frequently Asked Questions</h3>
                <div class="faq-item">
                    <h4>How often do you publish new content?</h4>
                    <p>We publish fresh AI-related content regularly, covering the latest developments in artificial intelligence and related technologies.</p>
                </div>
                <div class="faq-item">
                    <h4>Can I republish your content?</h4>
                    <p>Please contact us for permission. We're open to partnerships with proper attribution.</p>
                </div>
                <div class="faq-item">
                    <h4>Do you accept sponsored content?</h4>
                    <p>Yes, we consider sponsored content that aligns with our editorial standards. Please reach out for guidelines and pricing.</p>
                </div>
            </div>
            <div class="contact-footer">
                <p><strong>Before reaching out, please check our FAQ section – your question might already be answered!</strong></p>
                <p>We look forward to hearing from you.</p>
            </div>
        </article>
    </main>
    <footer><div class="container"><p>&copy; {{ current_year }} {{ site_name }}.</p></div></footer>
    <script src="../static/navigation.js"></script>
</body>
</html>"""
        }

        env = Environment(loader=BaseLoader())
        templates = {}
        for name, template_str in template_strings.items():
            templates[name] = env.from_string(template_str)
        return templates
