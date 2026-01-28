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
        """Main method to generate the entire static site"""
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
        """Generate ads.txt file for AdSense verification"""
        config = self.blog_system.config
        adsense_id = config.get('google_adsense_id', '')
        
        if adsense_id:
            # Remove 'ca-pub-' prefix if present
            pub_id = adsense_id.replace('ca-pub-', '')
            
            ads_txt_content = f"google.com, pub-{pub_id}, DIRECT, f08c47fec0942fa0\n"
            
            with open("./docs/ads.txt", 'w', encoding='utf-8') as f:
                f.write(ads_txt_content)
            
            print("Generated ads.txt file")

    def _get_all_posts(self) -> List[BlogPost]:
        """Load all blog posts from the docs directory"""
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
        """Generate the homepage with post listings"""
        config = self.blog_system.config
        
        context = {
            'site_name': config.get('site_name', 'Tech Blog'),
            'site_description': config.get('site_description', 'An AI-powered blog'),
            'base_path': config.get('base_path', ''),
            'base_url': config.get('base_url', ''),
            'posts': [p.to_dict() for p in posts],
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

    def _generate_post_pages(self, posts: List[BlogPost]):
        """Generate individual post pages"""
        config = self.blog_system.config
        
        for post in posts:
            post_dir = Path("./docs") / post.slug
            post_dir.mkdir(exist_ok=True)
            
            markdown_converter = md.Markdown(extensions=['extra', 'codehilite', 'toc'])
            content_html = markdown_converter.convert(post.content)
            
            post_dict = post.to_dict()
            post_dict['content_html'] = content_html
            
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
        """Generate about, contact, privacy, and terms pages"""
        config = self.blog_system.config
        
        # Map directory names (with hyphens) to template names (with underscores)
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
        """Generate RSS feed"""
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
        """Generate XML sitemap"""
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
            urls.append(f'<url><loc>{base_url}/{post.slug}/</loc><priority>0.9</priority></url>')
        
        sitemap = f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  {chr(10).join(urls)}
</urlset>"""
        
        with open("./docs/sitemap.xml", 'w', encoding='utf-8') as f:
            f.write(sitemap)
        
        print("Generated sitemap")

    def _generate_posts_json(self, posts: List[BlogPost]):
        """Generate JSON file with all posts for JavaScript loading"""
        posts_data = [p.to_dict() for p in posts]
        
        with open("./docs/posts.json", 'w', encoding='utf-8') as f:
            json.dump(posts_data, f, indent=2)
        
        print("Generated posts.json")

    def _escape_xml(self, text: str) -> str:
        """Escape XML special characters"""
        return (text.replace('&', '&amp;')
                   .replace('<', '&lt;')
                   .replace('>', '&gt;')
                   .replace('"', '&quot;')
                   .replace("'", '&apos;'))

    def _format_rss_date(self, iso_date: str) -> str:
        """Convert ISO date to RFC 822 format for RSS"""
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
    </head>
    <body>
        <!-- Header Ad Slot -->
        {{ header_ad | safe }}
        
        <header>
            <div class="container">
                <h1><a href="{{ base_path }}/">{{ site_name }}</a></h1>
                <nav>
                    <a href="{{ base_path }}/">Home</a>
                    <a href="{{ base_path }}/about/">About</a>
                    <a href="{{ base_path }}/contact/">Contact</a>
                    <a href="{{ base_path }}/privacy-policy/">Privacy Policy</a>
                    <a href="{{ base_path }}/terms-of-service/">Terms</a>
                </nav>
            </div>
        </header>
        <main class="container">
            <article class="blog-post">
                <header class="post-header">
                    <h1>{{ post.title }}</h1>
                    <div class="post-meta">
                        <time datetime="{{ post.created_at }}">{{ post.created_at.split('T')[0] }}</time>
                        {% if post.tags %}
                        <div class="tags">
                            {% for tag in post.tags %}
                            <span class="tag">{{ tag }}</span>
                            {% endfor %}
                        </div>
                        {% endif %}
                    </div>
                </header>
                <div class="post-content">
                    {{ post.content_html | safe }}
                    
                    <!-- Middle Ad Slot -->
                    {{ middle_ad | safe }}
                </div>
                
                <!-- Affiliate Disclaimer -->
                {% if post.affiliate_links %}
                <div class="affiliate-disclaimer">
                    <p><em>This post contains affiliate links. We may earn a commission if you make a purchase through these links, at no additional cost to you.</em></p>
                </div>
                {% endif %}
            </article>
        </main>
        
        <!-- Footer Ad Slot -->
        {{ footer_ad | safe }}
        
        <footer>
            <div class="container">
                <p>&copy; {{ current_year }} {{ site_name }}. Powered by AI.</p>
            </div>
        </footer>
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
                    <a href="{{ base_path }}/terms-of-service/">Terms</a>
                </nav>
            </div>
        </header>
        <main class="container">
            <div class="hero">
                <h2>Welcome to {{ site_name }}</h2>
                <p>{{ site_description }}</p>
            </div>

            <section class="recent-posts">
                <h2>Latest Posts</h2>
                {% if posts %}
                <div id="posts-container" class="post-grid">
                    {% for post in posts[:posts_per_page] %}
                    <article class="post-card" data-aos="fade-up" data-aos-delay="{{ loop.index0 * 100 }}">
                        <h3><a href="{{ base_path }}/{{ post.slug }}/">{{ post.title }}</a></h3>
                        <p class="post-excerpt">{{ post.meta_description }}</p>
                        <div class="post-meta">
                            <time datetime="{{ post.created_at }}">{{ post.created_at.split('T')[0] }}</time>
                            {% if post.tags %}
                            <div class="tags">
                                {% for tag in post.tags[:3] %}
                                <span class="tag">{{ tag }}</span>
                                {% endfor %}
                            </div>
                            {% endif %}
                        </div>
                    </article>
                    {% endfor %}
                </div>
                
                <!-- Loading Spinner -->
                <div id="loading-spinner" class="loading-spinner" style="display: none;">
                    <div class="spinner"></div>
                    <p>Loading more posts...</p>
                </div>
                
                <!-- Load More Button -->
                {% if posts|length > posts_per_page %}
                <div id="load-more-container" class="load-more-container">
                    <button id="load-more" class="load-more-button">
                        <span class="button-text">Load More Posts</span>
                        <span class="button-icon">↓</span>
                    </button>
                </div>
                {% endif %}
                
                <!-- Infinite Scroll Toggle -->
                <div class="scroll-options">
                    <label class="toggle-switch">
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
        
        <!-- Back to Top Button -->
        <button id="back-to-top" class="back-to-top" style="display: none;">
            <span>↑</span>
        </button>
        
        <footer>
            <div class="container">
                <p>&copy; {{ current_year }} {{ site_name }}. Powered by AI.</p>
                <div class="social-links">
                    {% for platform, url in social_links.items() %}
                    <a href="{{ url }}" target="_blank" rel="noopener">{{ platform|title }}</a>
                    {% endfor %}
                </div>
            </div>
        </footer>
        
        <!-- Enhanced JavaScript for Load More and Infinite Scroll -->
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                const loadMoreButton = document.getElementById('load-more');
                const loadingSpinner = document.getElementById('loading-spinner');
                const postsContainer = document.getElementById('posts-container');
                const infiniteScrollToggle = document.getElementById('infinite-scroll-toggle');
                const backToTopButton = document.getElementById('back-to-top');
                
                let currentPage = 1;
                const postsPerPage = {{ posts_per_page }};
                let allPosts = [];
                let isLoading = false;
                let infiniteScrollEnabled = localStorage.getItem('infiniteScroll') === 'true';
                
                infiniteScrollToggle.checked = infiniteScrollEnabled;
                
                fetch('{{ base_path }}/posts.json')
                    .then(response => response.json())
                    .then(posts => {
                        allPosts = posts;
                        updateLoadMoreButton();
                    })
                    .catch(error => {
                        console.error('Error loading posts data:', error);
                    });
                
                if (loadMoreButton) {
                    loadMoreButton.addEventListener('click', function() {
                        loadMorePosts();
                    });
                }
                
                infiniteScrollToggle.addEventListener('change', function() {
                    infiniteScrollEnabled = this.checked;
                    localStorage.setItem('infiniteScroll', infiniteScrollEnabled);
                    
                    if (infiniteScrollEnabled) {
                        loadMoreButton.style.display = 'none';
                        enableInfiniteScroll();
                    } else {
                        disableInfiniteScroll();
                        updateLoadMoreButton();
                    }
                });
                
                if (infiniteScrollEnabled) {
                    enableInfiniteScroll();
                    loadMoreButton.style.display = 'none';
                }
                
                window.addEventListener('scroll', function() {
                    if (window.pageYOffset > 300) {
                        backToTopButton.style.display = 'flex';
                    } else {
                        backToTopButton.style.display = 'none';
                    }
                });
                
                backToTopButton.addEventListener('click', function() {
                    window.scrollTo({ top: 0, behavior: 'smooth' });
                });
                
                function loadMorePosts() {
                    if (isLoading || !hasMorePosts()) return;
                    
                    isLoading = true;
                    showLoadingSpinner();
                    
                    setTimeout(() => {
                        currentPage++;
                        const startIndex = (currentPage - 1) * postsPerPage;
                        const endIndex = startIndex + postsPerPage;
                        const nextPosts = allPosts.slice(startIndex, endIndex);
                        
                        nextPosts.forEach((post, index) => {
                            const postElement = createPostElement(post);
                            postElement.style.opacity = '0';
                            postElement.style.transform = 'translateY(20px)';
                            postsContainer.appendChild(postElement);
                            
                            setTimeout(() => {
                                postElement.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
                                postElement.style.opacity = '1';
                                postElement.style.transform = 'translateY(0)';
                            }, index * 100);
                        });
                        
                        hideLoadingSpinner();
                        updateLoadMoreButton();
                        isLoading = false;
                    }, 500);
                }
                
                function enableInfiniteScroll() {
                    window.addEventListener('scroll', infiniteScrollHandler);
                }
                
                function disableInfiniteScroll() {
                    window.removeEventListener('scroll', infiniteScrollHandler);
                }
                
                function infiniteScrollHandler() {
                    if (isLoading || !hasMorePosts()) return;
                    
                    const scrollTop = window.pageYOffset;
                    const windowHeight = window.innerHeight;
                    const docHeight = document.documentElement.scrollHeight;
                    
                    if (scrollTop + windowHeight >= docHeight - 1000) {
                        loadMorePosts();
                    }
                }
                
                function hasMorePosts() {
                    return currentPage * postsPerPage < allPosts.length;
                }
                
                function updateLoadMoreButton() {
                    if (!loadMoreButton) return;
                    
                    if (hasMorePosts() && !infiniteScrollEnabled) {
                        loadMoreButton.style.display = 'block';
                        const remainingPosts = allPosts.length - (currentPage * postsPerPage);
                        const buttonText = loadMoreButton.querySelector('.button-text');
                        buttonText.textContent = 'Load More Posts (' + remainingPosts + ' remaining)';
                    } else {
                        loadMoreButton.style.display = 'none';
                    }
                }
                
                function showLoadingSpinner() {
                    if (loadingSpinner) {
                        loadingSpinner.style.display = 'block';
                    }
                    if (loadMoreButton) {
                        loadMoreButton.disabled = true;
                        loadMoreButton.style.opacity = '0.6';
                    }
                }
                
                function hideLoadingSpinner() {
                    if (loadingSpinner) {
                        loadingSpinner.style.display = 'none';
                    }
                    if (loadMoreButton) {
                        loadMoreButton.disabled = false;
                        loadMoreButton.style.opacity = '1';
                    }
                }
                
                function createPostElement(post) {
                    const article = document.createElement('article');
                    article.className = 'post-card';
                    
                    const title = document.createElement('h3');
                    const titleLink = document.createElement('a');
                    titleLink.href = '{{ base_path }}/' + post.slug + '/';
                    titleLink.textContent = post.title;
                    title.appendChild(titleLink);
                    
                    const excerpt = document.createElement('p');
                    excerpt.className = 'post-excerpt';
                    excerpt.textContent = post.meta_description;
                    
                    const meta = document.createElement('div');
                    meta.className = 'post-meta';
                    
                    const time = document.createElement('time');
                    time.dateTime = post.created_at;
                    time.textContent = post.created_at.split('T')[0];
                    
                    meta.appendChild(time);
                    
                    if (post.tags && post.tags.length > 0) {
                        const tags = document.createElement('div');
                        tags.className = 'tags';
                        
                        post.tags.slice(0, 3).forEach(tag => {
                            const tagSpan = document.createElement('span');
                            tagSpan.className = 'tag';
                            tagSpan.textContent = tag;
                            tags.appendChild(tagSpan);
                        });
                        
                        meta.appendChild(tags);
                    }
                    
                    article.appendChild(title);
                    article.appendChild(excerpt);
                    article.appendChild(meta);
                    
                    return article;
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
    </head>
    <body>
        <header>
            <div class="container">
                <h1><a href="../">{{ site_name }}</a></h1>
                <nav>
                    <a href="../">Home</a>
                    <a href="../about/">About</a>
                    <a href="../contact/">Contact</a>
                    <a href="../privacy-policy/">Privacy Policy</a>
                    <a href="../terms-of-service/">Terms</a>
                </nav>
            </div>
        </header>
        <main class="container">
            <div class="hero">
                <h2>About Tech Blog</h2>
                <p>Learn more about our mission and what we do</p>
            </div>
            <article class="page-content">
                <h1>About {{ site_name }}</h1>
                <p>Welcome to {{ site_name }}, an innovative AI-powered blog that delivers high-quality, informative content on various technology topics.</p>
                
                <h2>Our Content Philosophy</h2>
                <p>We believe in the responsible use of AI technology to create content that genuinely helps our readers.</p>
                
                <h2>Topics We Cover</h2>
                <ul>
                    {% for topic in topics %}
                    <li>{{ topic }}</li>
                    {% endfor %}
                </ul>
                
                <h2>Contact Us</h2>
                <p>Have questions? Visit our <a href="../contact/">contact page</a>.</p>
            </article>
        </main>
        <footer>
            <div class="container">
                <p>&copy; {{ current_year }} {{ site_name }}. Powered by AI.</p>
            </div>
        </footer>
    </body>
    </html>""",

                "privacy_policy": """<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Privacy Policy - {{ site_name }}</title>
        <meta name="description" content="Privacy Policy for {{ site_name }}">
        {{ global_meta_tags | safe }}
        <link rel="stylesheet" href="../static/style.css">
        <link rel="canonical" href="{{ base_url }}/privacy-policy/">
    </head>
    <body>
        <header>
            <div class="container">
                <h1><a href="../">{{ site_name }}</a></h1>
                <nav>
                    <a href="../">Home</a>
                    <a href="../about/">About</a>
                    <a href="../contact/">Contact</a>
                    <a href="../privacy-policy/">Privacy Policy</a>
                    <a href="../terms-of-service/">Terms</a>
                </nav>
            </div>
        </header>
        <main class="container">
            <div class="hero">
                <h2>Privacy Policy</h2>
                <p>How we protect and handle your information</p>
            </div>
            <article class="page-content">
                <p><strong>Last updated:</strong> October 02, 2025</p>

                <h2>Introduction</h2>
                <p><strong>AI Tech Blog</strong> ("we," "our," or "us") is committed to protecting your privacy. This Privacy Policy explains how we collect, use, and safeguard your information when you use our website.</p>

                <h2>Information We Collect</h2>
                <ul>
                <li><strong>Personal Information:</strong> such as your email address, if you voluntarily contact us or subscribe to updates.</li>
                <li><strong>Non-Personal Information:</strong> such as browser type, device type, and usage data, collected automatically for analytics and site performance.</li>
                <li><strong>Cookies & Tracking:</strong> we may use cookies, analytics tools, and similar technologies to improve the user experience and deliver relevant ads.</li>
                </ul>

                <h2>How We Use Your Information</h2>
                <ul>
                <li>To operate and maintain the Site.</li>
                <li>To improve content and personalize user experience.</li>
                <li>To send newsletters, updates, or promotional content (if you opt in).</li>
                <li>To comply with legal obligations.</li>
                </ul>

                <h2>Sharing of Information</h2>
                <p>We do not sell or rent your personal information. We may share data with trusted third-party service providers (e.g., analytics, hosting, advertising networks) only as necessary to operate the Site.</p>

                <h2>Third-Party Links</h2>
                <p>Our Site may contain links to third-party websites or affiliate products. We are not responsible for the privacy practices or content of those external sites.</p>

                <h2>Data Retention</h2>
                <p>We retain personal information only for as long as necessary to fulfill the purposes described in this Privacy Policy or as required by law.</p>

                <h2>Your Privacy Rights</h2>
                <p>Depending on your location, you may have rights to access, update, or delete your personal data. To exercise these rights, contact us at <a href="mailto:aiblogauto@gmail.com">aiblogauto@gmail.com</a>.</p>

                <h2>Children's Privacy</h2>
                <p>The Site is not intended for children under 13. We do not knowingly collect information from children. If you believe we have unintentionally collected data from a child, please contact us.</p>

                <h2>Security</h2>
                <p>We implement reasonable security measures to protect your information. However, no method of transmission or storage is 100% secure, and we cannot guarantee absolute security.</p>

                <h2>Changes to This Privacy Policy</h2>
                <p>We may update this Privacy Policy from time to time. Any updates will be posted on this page with the revised "Last updated" date.</p>

                <h2>Contact Us</h2>
                <p>If you have any questions or concerns about this Privacy Policy, email us at 
                <a href="mailto:aiblogauto@gmail.com">aiblogauto@gmail.com</a>.
                </p>
            </article>
            </main>

        <footer>
            <div class="container">
                <p>&copy; {{ current_year }} {{ site_name }}. Powered by AI.</p>
            </div>
        </footer>
    </body>
    </html>""",

                "terms_of_service": """<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Terms of Service - {{ site_name }}</title>
        <meta name="description" content="Terms of Service for {{ site_name }}">
        {{ global_meta_tags | safe }}
        <link rel="stylesheet" href="../static/style.css">
        <link rel="canonical" href="{{ base_url }}/terms-of-service/">
    </head>
    <body>
        <header>
            <div class="container">
                <h1><a href="../">{{ site_name }}</a></h1>
                <nav>
                    <a href="../">Home</a>
                    <a href="../about/">About</a>
                    <a href="../contact/">Contact</a>
                    <a href="../privacy-policy/">Privacy Policy</a>
                    <a href="../terms-of-service/">Terms</a>
                </nav>
            </div>
        </header>
        <main class="container">
        <div class="hero">
            <h2>Terms of Service</h2>
            <p>Please read these terms carefully before using our site</p>
        </div>
        <article class="page-content">
            <p><strong>Last updated:</strong> October 02, 2025</p>

            <h2>Agreement to Terms</h2>
            <p>By accessing <strong>AI Tech Blog</strong> (the "Site," "we," "our," or "us"), you agree to be bound by these Terms of Service (the "Terms"). If you do not agree, please do not use our services.</p>

            <h2>Use of Our Services</h2>
            <ul>
            <li>You may browse and read the content for personal, non-commercial purposes.</li>
            <li>You agree not to misuse the Site or interfere with its operation.</li>
            <li>Automated scraping, unauthorized reposting, or commercial use of our content is prohibited unless expressly authorized.</li>
            </ul>

            <h2>AI-Generated Content Disclaimer</h2>
            <p>The articles, posts, and recommendations published on AI Tech Blog are automatically generated using artificial intelligence. While we strive for accuracy, we cannot guarantee that all information is correct, complete, or up to date. Content should not be considered professional, legal, medical, or financial advice. Users are responsible for verifying information before acting on it.</p>

            <h2>Intellectual Property Rights</h2>
            <p>All content, branding, and design elements on the Site are owned or licensed by us, unless otherwise stated. You may not reproduce, distribute, or create derivative works without prior written permission.</p>

            <h2>Third-Party Links & Affiliates</h2>
            <ul>
            <li>Some articles may include links to third-party websites, affiliate products, or sponsored content.</li>
            <li>We are not responsible for the content, policies, or actions of any third-party site.</li>
            <li>When you click on affiliate links, we may earn a commission at no extra cost to you.</li>
            </ul>

            <h2>Limitation of Liability</h2>
            <p>The Site is provided "as is" without warranties of any kind. We are not liable for any damages, losses, or consequences resulting from the use of our AI-generated content or third-party services.</p>

            <h2>Privacy</h2>
            <p>Your use of the Site is also governed by our <a href="../privacy-policy/">Privacy Policy</a>, which describes how we collect, use, and store information.</p>

            <h2>Termination of Use</h2>
            <p>We reserve the right to suspend or terminate access to the Site at our discretion, without notice, if you violate these Terms.</p>

            <h2>Governing Law</h2>
            <p>These Terms are governed by and interpreted under the laws of Kenya, without regard to conflict of law principles.</p>

            <h2>Updates to Terms</h2>
            <p>We may update these Terms from time to time. Changes will be posted with an updated "Last updated" date. Continued use of the Site indicates acceptance of the revised Terms.</p>

            <h2>Contact Information</h2>
            <p>For questions or concerns about these Terms, please contact us at: 
            <a href="mailto:aiblogauto@gmail.com">aiblogauto@gmail.com</a>
            </p>
        </article>
        </main>
        <footer>
            <div class="container">
                <p>&copy; {{ current_year }} {{ site_name }}. Powered by AI.</p>
            </div>
        </footer>
    </body>
    </html>""",

                "contact": """<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Contact Us - {{ site_name }}</title>
        <meta name="description" content="Contact {{ site_name }}">
        {{ global_meta_tags | safe }}
        <link rel="stylesheet" href="../static/style.css">
        <link rel="canonical" href="{{ base_url }}/contact/">
    </head>
    <body>
        <header>
            <div class="container">
                <h1><a href="../">{{ site_name }}</a></h1>
                <nav>
                    <a href="../">Home</a>
                    <a href="../about/">About</a>
                    <a href="../contact/">Contact</a>
                    <a href="../privacy-policy/">Privacy Policy</a>
                    <a href="../terms-of-service/">Terms</a>
                </nav>
            </div>
        </header>
        <main class="container">
            <div class="hero">
                <h2>Contact Us</h2>
                <p>Get in touch with the AI Tech Blog team</p>
            </div>
            <article class="page-content">
                <p>We'd love to hear from you!</p>
                
                <div class="contact-info">
                    <h2>Get in Touch</h2>
                    <div class="contact-method">
                        <h3>Email</h3>
                        <p><a href="mailto:aiblogauto@gmail.com" class="contact-email">aiblogauto@gmail.com</a></p>
                    </div>
                </div>
            </article>
        </main>
        <footer>
            <div class="container">
                <p>&copy; {{ current_year }} {{ site_name }}. Powered by AI.</p>
            </div>
        </footer>
    </body>
    </html>"""
            }

            env = Environment(loader=BaseLoader())
            templates = {}
            for name, template_str in template_strings.items():
                templates[name] = env.from_string(template_str)
            return templates