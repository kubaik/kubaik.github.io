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
        # self._generate_ads_txt()
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

    def _reading_time_minutes(self, content: str) -> int:
        """Estimate reading time at 200 words per minute."""
        word_count = len(content.split())
        return max(1, round(word_count / 200))

    def _generate_homepage(self, posts: List[BlogPost]):
        config = self.blog_system.config
        posts_data = []
        for p in posts:
            post_dict = p.to_dict()
            post_dict['display_date'] = self._format_display_date(p.created_at)
            post_dict['short_tags'] = sorted(p.tags, key=len)[
                :3]  # shortest 3 tags
            post_dict['reading_time'] = self._reading_time_minutes(p.content)
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
        for post in posts:
            post_dir = Path("./docs") / post.slug
            post_dir.mkdir(exist_ok=True)
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
        # Templates are stored in a separate helper to keep this method short.
        return _build_templates()


# ---------------------------------------------------------------------------
# Template builder — kept outside the class so the heredoc approach works.
# ---------------------------------------------------------------------------
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
                    {% if post.reading_time %}
                    <span class="reading-time"> &middot; {{ post.reading_time }} min read</span>
                    {% endif %}
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
</html>"""

    # -------------------------------------------------------------------------
    # INDEX template
    # KEY DESIGN:
    #   Each card is  <a class="post-card" href="...">  with NO inner <a>.
    #   CSS (.post-card) gives it display:flex so it stretches to row height.
    #   .post-excerpt has flex:1 to push tags toward the bottom.
    #   JS createPostElement() mirrors this structure exactly so load-more
    #   cards look and behave identically to the server-rendered ones.
    # -------------------------------------------------------------------------
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

        <div class="search-container">
            <div class="search-wrapper">
                <svg class="search-icon" width="18" height="18" viewBox="0 0 20 20" fill="none">
                    <path d="M9 17A8 8 0 1 0 9 1a8 8 0 0 0 0 16zM19 19l-4.35-4.35"
                          stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                <input type="text" id="search-input" class="search-input"
                    placeholder="Search"
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
                <a class="post-card" href="{{ base_path }}/{{ post.slug }}/">
                    <h3>{{ post.title }}</h3>
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

    <button id="back-to-top" class="back-to-top" style="display:none;"><span>&#8593;</span></button>

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

        var searchInput     = document.getElementById('search-input');
        var clearSearchBtn  = document.getElementById('clear-search');
        var resultsCount    = document.getElementById('search-results-count');
        var postsContainer  = document.getElementById('posts-container');
        var loadingSpinner  = document.getElementById('loading-spinner');
        var sentinel        = document.getElementById('scroll-sentinel');
        var backToTopButton = document.getElementById('back-to-top');

        var postsPerPage = {{ posts_per_page }};
        var allPosts     = [];
        var currentPage  = 1;
        var isLoading    = false;
        var searchActive = false;
        var observer     = null;

        // Reads server-rendered <a class="post-card"> elements as fallback
        function seedFromDOM() {
            if (!postsContainer) return;
            allPosts = Array.from(postsContainer.querySelectorAll('a.post-card')).map(function(el) {
                var href = el.getAttribute('href') || '';
                var slug = href.replace(/\/+$/, '').split('/').filter(Boolean).pop() || '';
                return {
                    slug:             slug,
                    title:            el.querySelector('h3') ? el.querySelector('h3').textContent.trim() : '',
                    meta_description: el.querySelector('.post-excerpt') ? el.querySelector('.post-excerpt').textContent.trim() : '',
                    tags:             Array.from(el.querySelectorAll('.tag')).map(function(t) { return t.textContent.trim(); }),
                    created_at:       ''
                };
            });
        }

        fetch('{{ base_path }}/posts.json')
            .then(function(r) {
                if (!r.ok) throw new Error('HTTP ' + r.status);
                return r.json();
            })
            .then(function(posts) {
                allPosts = posts;
                startObserver();
                if (searchInput && searchInput.value.trim()) {
                    performSearch(searchInput.value.trim().toLowerCase());
                }
            })
            .catch(function(err) {
                console.warn('posts.json fetch failed, falling back to DOM seed:', err);
                seedFromDOM();
                startObserver();
            });

        // ── IntersectionObserver: fires when sentinel enters the viewport ──
        function startObserver() {
            if (!sentinel || !window.IntersectionObserver) return;
            observer = new IntersectionObserver(function(entries) {
                if (entries[0].isIntersecting) loadMorePosts();
            }, { rootMargin: '0px 0px 300px 0px' });
            observer.observe(sentinel);
        }

        function stopObserver() {
            if (observer) { observer.disconnect(); observer = null; }
        }

        function escapeRe(s) {
            return s.replace(/[.*+?^${}()|[\\\\]\\\\]/g, '\\\\\\\\$&');
        }

        function highlight(text, query) {
            return text.replace(
                new RegExp('(' + escapeRe(query) + ')', 'gi'),
                '<span class="search-highlight">$1</span>'
            );
        }

        // Mirrors <a class="post-card" href="..."><h3>...</h3><p class="post-excerpt">...</p><div class="tags">...</div></a>
        function createPostElement(post) {
            var card       = document.createElement('a');
            card.className = 'post-card';
            card.href      = '{{ base_path }}/' + post.slug + '/';

            var h3         = document.createElement('h3');
            h3.textContent = post.title;

            var excerpt         = document.createElement('p');
            excerpt.className   = 'post-excerpt';
            excerpt.textContent = post.meta_description;

            card.appendChild(h3);
            card.appendChild(excerpt);

            if (post.tags && post.tags.length > 0) {
                var tagsDiv       = document.createElement('div');
                tagsDiv.className = 'tags';
                post.tags.slice().sort(function(a, b) { return a.length - b.length; }).slice(0, 3).forEach(function(t) {
                    var span         = document.createElement('span');
                    span.className   = 'tag';
                    span.textContent = t;
                    tagsDiv.appendChild(span);
                });
                card.appendChild(tagsDiv);
            }

            return card;
        }

        function performSearch(query) {
            if (!allPosts.length) return;
            searchActive = true;
            stopObserver();

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
                var heading = el.querySelector('h3');
                var excerpt = el.querySelector('.post-excerpt');
                if (heading) heading.innerHTML = highlight(heading.textContent, query);
                if (excerpt) excerpt.innerHTML = highlight(excerpt.textContent, query);
                postsContainer.appendChild(el);
            });

            var n = matched.length;
            resultsCount.textContent = n === 0 ? 'No results for "' + query + '"'
                                     : n === 1 ? '1 post found'
                                     : n + ' posts found';

            var noRes = document.getElementById('no-results');
            if (n === 0) {
                if (!noRes) {
                    noRes           = document.createElement('div');
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
            searchActive             = false;
            postsContainer.innerHTML = '';
            currentPage              = 1;
            allPosts.slice(0, postsPerPage).forEach(function(post) {
                postsContainer.appendChild(createPostElement(post));
            });
            resultsCount.textContent = '';
            var noRes = document.getElementById('no-results');
            if (noRes) noRes.remove();
            startObserver();
        }

        if (searchInput) {
            searchInput.addEventListener('input', function () {
                var q = this.value.trim().toLowerCase();
                clearSearchBtn.style.display = q ? 'flex' : 'none';
                if (q) { performSearch(q); } else { clearSearch(); }
            });
            clearSearchBtn.addEventListener('click', function () {
                searchInput.value  = '';
                this.style.display = 'none';
                clearSearch();
                searchInput.focus();
            });
            searchInput.addEventListener('keydown', function (e) {
                if (e.key === 'Escape') {
                    this.value                   = '';
                    clearSearchBtn.style.display = 'none';
                    clearSearch();
                }
            });
        }

        window.addEventListener('scroll', function () {
            if (backToTopButton)
                backToTopButton.style.display = window.pageYOffset > 300 ? 'flex' : 'none';
        }, { passive: true });

        if (backToTopButton) {
            backToTopButton.addEventListener('click', function() {
                window.scrollTo({ top: 0, behavior: 'smooth' });
            });
        }

        function loadMorePosts() {
            if (isLoading || searchActive || !hasMorePosts()) {
                if (!hasMorePosts()) stopObserver();
                return;
            }
            isLoading = true;
            if (loadingSpinner) loadingSpinner.style.display = 'flex';

            setTimeout(function() {
                currentPage++;
                var start    = (currentPage - 1) * postsPerPage;
                var fragment = document.createDocumentFragment();
                var newCards = [];
                allPosts.slice(start, start + postsPerPage).forEach(function(post) {
                    var el = createPostElement(post);
                    el.classList.add('post-card--entering');
                    fragment.appendChild(el);
                    newCards.push(el);
                });
                postsContainer.appendChild(fragment);
                // Double rAF ensures the browser has painted the entering state before animating
                requestAnimationFrame(function() {
                    requestAnimationFrame(function() {
                        newCards.forEach(function(el, i) {
                            setTimeout(function() { el.classList.remove('post-card--entering'); }, i * 60);
                        });
                    });
                });
                if (loadingSpinner) loadingSpinner.style.display = 'none';
                isLoading = false;
                if (!hasMorePosts()) stopObserver();
            }, 300);
        }

        function hasMorePosts() { return currentPage * postsPerPage < allPosts.length; }
    });
    </script>
</body>
</html>"""

    # FIX: About page now includes an author/team section for E-E-A-T signals
    ABOUT_TMPL = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>About - {{ site_name }}</title>
    <meta name="description" content="About {{ site_name }} — who we are and what we cover">
    {{ global_meta_tags | safe }}
    <link rel="stylesheet" href="../static/style.css">
    <link rel="canonical" href="{{ base_url }}/about/">
    <style>
        .about-section{background:#f8f9fa;padding:1.5rem 2rem;margin-bottom:1.5rem;border-radius:8px;border-left:4px solid #6366f1}
        .about-section h2{color:#333;margin-top:0;margin-bottom:1rem}
        .feature-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:1.5rem;margin:1.5rem 0}
        .feature-card{background:white;padding:1.5rem;border-radius:8px;border:2px solid #e0e0e0}
        .feature-card h3{color:#6366f1;margin-top:0}
        .author-card{background:white;padding:1.5rem;border-radius:8px;border:2px solid #6366f1;margin-bottom:1.5rem}
        .author-card h3{margin-top:0;color:#333}
        .author-card .credentials{color:#666;font-size:0.9rem;margin-bottom:0.75rem}
        .stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:1rem;margin:1.5rem 0}
        .stat-card{background:#f0f4ff;padding:1.5rem;border-radius:8px;text-align:center;border:2px solid #6366f1}
        .stat-number{font-size:2rem;font-weight:bold;color:#6366f1;display:block;margin-bottom:0.25rem}
        .cta-box{background:#fff3cd;border-left:4px solid #ffc107;padding:1.5rem;border-radius:8px;margin:1.5rem 0;text-align:center}
        .cta-button{display:inline-block;background:linear-gradient(135deg,#667eea,#764ba2);color:white;padding:0.8rem 1.8rem;border-radius:8px;text-decoration:none;font-weight:600;margin-top:0.8rem}
    </style>
</head>
<body>
    <header><div class="container">
        <h1><a href="../">{{ site_name }}</a></h1>
        <nav><a href="../">Home</a><a href="../about/">About</a><a href="../contact/">Contact</a>
        <a href="../privacy-policy/">Privacy</a><a href="../terms-of-service/">Terms</a></nav>
    </div></header>
    <main class="container">
        <div class="hero"><h2>About {{ site_name }}</h2><p>Practical technology writing for developers and builders</p></div>
        <article class="page-content">

            <!-- E-E-A-T: Author/team section — critical for AdSense approval -->
            <div class="about-section">
                <h2>Who Writes Here</h2>
                <div class="author-card">
                    <h3>Kubai Kevin</h3>
                    <p class="credentials">Software developer based in Nairobi, Kenya &bull; Writing about AI, backend systems, and developer tooling</p>
                    <p>I have been building software professionally for several years, working across web development, automation, and AI-assisted tools. This blog documents what I learn along the way — the problems I encounter in real projects, the tradeoffs I have had to make, and the tools that have actually proved useful in production.</p>
                    <p>I started this blog because most technical content either explains concepts without showing real tradeoffs, or gives advice that only applies at the scale of large companies. My goal is to write about what actually works for developers building real things at realistic scales.</p>
                </div>
            </div>

            <div class="about-section">
                <h2>What This Blog Covers</h2>
                <p>The topics on this blog fall into a few areas that I work with regularly:</p>
                <div class="feature-grid">
                    <div class="feature-card"><h3>AI &amp; LLMs</h3><p>Practical applications of language models, prompt engineering, retrieval systems, and what actually works beyond the demos.</p></div>
                    <div class="feature-card"><h3>Backend Systems</h3><p>APIs, databases, queues, and the distributed systems problems that come up in real products — not textbook examples.</p></div>
                    <div class="feature-card"><h3>Developer Tools</h3><p>CI/CD, observability, developer experience. What saves real time and what sounds good on paper but adds friction.</p></div>
                    <div class="feature-card"><h3>Tech Careers</h3><p>How the industry is changing, what skills matter, and honest analysis of the trends that will and will not last.</p></div>
                </div>
            </div>

            <div class="about-section">
                <h2>Editorial Standards</h2>
                <p>Some posts on this site are drafted with AI assistance and then reviewed and edited for accuracy. Where AI drafts are used, they are checked against documentation, tested against real systems where possible, and updated when errors are found.</p>
                <p>If you spot an error or something that does not match your experience, the contact page has my email address. I take corrections seriously.</p>
            </div>

            <div class="stat-grid">
                <div class="stat-card"><span class="stat-number">{{ posts|length }}</span><span>Posts published</span></div>
                <div class="stat-card"><span class="stat-number">Weekly</span><span>Publishing cadence</span></div>
                <div class="stat-card"><span class="stat-number">Free</span><span>Always and forever</span></div>
            </div>

            <div class="cta-box">
                <h3>Get in Touch</h3>
                <p>Questions, corrections, or topic suggestions are welcome.</p>
                <a href="../contact/" class="cta-button">Contact Me</a>
            </div>
        </article>
    </main>
    <footer><div class="container"><p>&copy; {{ current_year }} {{ site_name }}.</p></div></footer>
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
        <a href="../privacy-policy/">Privacy</a><a href="../terms-of-service/">Terms</a></nav>
    </div></header>
    <main class="container">
        <div class="hero"><h2>Privacy Policy</h2></div>
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
        <a href="../privacy-policy/">Privacy</a><a href="../terms-of-service/">Terms</a></nav>
    </div></header>
    <main class="container">
        <div class="hero"><h2>Terms of Service</h2></div>
        <article class="page-content">
            <div class="terms-section"><h3>1. Acceptance</h3>
                <p>By accessing {{ site_name }}, you agree to these Terms and our Privacy Policy.</p></div>
            <div class="terms-section"><h3>2. AI-Assisted Content</h3>
                <div class="warning-box"><p>Some content on this site is drafted with AI assistance and reviewed for accuracy. Always verify technical information independently before using it in production.</p></div></div>
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
    <meta name="description" content="Contact {{ site_name }}">
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
        <a href="../privacy-policy/">Privacy</a><a href="../terms-of-service/">Terms</a></nav>
    </div></header>
    <main class="container">
        <div class="hero"><h2>Contact</h2></div>
        <article class="page-content">
            <div class="contact-method"><h3>Email</h3>
                <p><a href="mailto:aiblogauto@gmail.com" class="contact-email">aiblogauto@gmail.com</a></p>
                <p>I typically respond within 3–5 working days.</p></div>
            <div class="contact-method"><h3>What to reach out about</h3>
                <ul>
                    <li>Factual errors or corrections in articles</li>
                    <li>Topic suggestions or questions about content</li>
                    <li>Collaboration or guest post proposals</li>
                    <li>General questions about the site</li>
                </ul></div>
            <div class="contact-method"><h3>Response time</h3>
                <p>Monday to Friday, EAT (UTC+3). Weekend messages are read on the next business day.</p></div>
            <div class="contact-footer">
                <p>Prefer email over social media for anything that requires a substantive reply.</p>
            </div>
        </article>
    </main>
    <footer><div class="container"><p>&copy; {{ current_year }} {{ site_name }}.</p></div></footer>
    <script src="../static/navigation.js"></script>
</body>
</html>"""

    env = Environment(loader=BaseLoader())
    return {
        'post':           env.from_string(POST_TMPL),
        'index':          env.from_string(INDEX_TMPL),
        'about':          env.from_string(ABOUT_TMPL),
        'privacy_policy': env.from_string(PRIVACY_TMPL),
        'terms_of_service': env.from_string(TERMS_TMPL),
        'contact':        env.from_string(CONTACT_TMPL),
    }
